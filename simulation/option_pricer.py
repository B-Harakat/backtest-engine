"""
Option pricing for the synthetic market: turns a Bates spot/vol state into a Black-Scholes
price and greeks for a single (strike, right, expiration) contract.

The market maker prices WITH Black-Scholes, using a synthetic implied-volatility surface derived
from the Bates path's instantaneous variance `V_t`. So Bates-SV/jump realism enters through the
pricing vol (which fluctuates with the path), while execution prices are vanilla BS.

Consistency with the engine:
  - European exercise (correct for cash-settled index products like XSP), `r = 0.0`, `q = 0.0`
    — the exact conventions `engine.greeks` assumes.
  - `vega`/`rho` columns are stored DIVIDED BY 100, matching what the real ThetaData path
    (`engine.thetadata_client`) writes into its warmed frames. Do NOT re-scale downstream.
  - `gamma`/`vanna`/`charm` come from `engine.greeks.compute_second_order_greeks` (already the
    engine's standard per-unit convention), so synthetic and real data are comparable.
  - `time_to_expiry_years` / `settlement_time` match `engine.greeks`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time

from engine.entities import Right
from engine.greeks import DEFAULT_CASH_SETTLEMENT_TIME_ET, compute_second_order_greeks, time_to_expiry_years


# Black-Scholes helpers, stdlib only (mirror engine.greeks's normal pdf/cdf).
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


#: Moneyness-smile steepness used by `synthetic_implied_vol`. Kept small so the surface stays
#: roughly flat by default.
_SKEW_SLOPE = 0.05


def synthetic_implied_vol(spot_vol: float, strike: float, spot: float, time_to_expiry_yrs: float) -> float:
    """
    Map an instantaneous vol (sqrt{V_t}) into a pricing IV. This is the hook where you'd put a
    real calibrated surface (e.g. SABR / SSVI / spline). The default is intentionally simple:
    instantaneous vol + a mild smile skew that weakens as time-to-expiry grows and tilts by
    moneyness (downward for OTM calls => skew smirks, negligible by default via _SKEW_SLOPE).
    """
    base = spot_vol
    if time_to_expiry_yrs <= 0 or spot <= 0:
        return base
    moneyness = strike / spot  # >1 OTM call / ITM put
    skew = _SKEW_SLOPE * (moneyness - 1.0) / (1.0 + 10.0 * time_to_expiry_yrs)
    return max(base + skew, 1e-4)


@dataclass(frozen=True)
class PricedOption:
    """Cross-sectional price + greeks for one (strike, right) at one instant."""

    strike: float
    right: Right
    implied_vol: float
    price: float          # mid; the spread is applied by the market maker for bid/ask.
    delta: float
    gamma: float
    theta: float
    vega: float           # standard per-unit convention (this repo), i.e. ThetaData /100.
    rho: float            # same -- in this repo's convention.
    vanna: float
    charm: float


def price_option(
    spot: float,
    strike: float,
    right: Right,
    expiration: date,
    as_of: datetime,
    spot_vol: float,
    r: float = 0.0,
    q: float = 0.0,
    settlement_time: time = DEFAULT_CASH_SETTLEMENT_TIME_ET,
) -> PricedOption:
    """
    Price an option under BS using ``synthetic_implied_vol(spot_vol, ...)`` as the IV.
    ``as_of`` is the bar timestamp (naive ET) used to compute time-to-expiry.
    """
    T = time_to_expiry_years(as_of, expiration, settlement_time)
    iv = synthetic_implied_vol(spot_vol, strike, spot, T)
    if T <= 0:
        # Settlement instant — only intrinsic value, all greeks 0. The engine never trades a
        # 16:00 bar, but be safe.
        intrinsic = max(spot - strike, 0.0) if right == "CALL" else max(strike - spot, 0.0)
        return PricedOption(
            strike=strike, right=right, implied_vol=iv, price=intrinsic,
            delta=1.0 if (right == "CALL" and spot > strike) else (0.0 if right == "PUT" and spot < strike else 0.0),
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0, vanna=0.0, charm=0.0,
        )

    S, K, T_y, sigma = spot, strike, T, iv
    sqrt_T = math.sqrt(max(T_y, 1e-12))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T_y) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc_q = math.exp(-q * T_y)

    if right == "CALL":
        price = S * math.exp(-q * T_y) * _norm_cdf(d1) - K * math.exp(-r * T_y) * _norm_cdf(d2)
        delta = math.exp(-q * T_y) * _norm_cdf(d1)
    else:
        price = K * math.exp(-r * T_y) * _norm_cdf(-d2) - S * math.exp(-q * T_y) * _norm_cdf(-d1)
        delta = -math.exp(-q * T_y) * _norm_cdf(-d1)

    pdf_d1 = _norm_pdf(d1)
    vega = S * disc_q * pdf_d1 * sqrt_T          # standard per-unit before /100
    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
    theta_call = (
        -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_T)
        - r * K * math.exp(-r * T_y) * _norm_cdf(d2)
        + q * S * disc_q * _norm_cdf(d1)
    )
    theta_put = (
        -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_T)
        + r * K * math.exp(-r * T_y) * _norm_cdf(-d2)
        - q * S * disc_q * _norm_cdf(-d1)
    )
    theta = theta_call if right == "CALL" else theta_put
    rho_call = K * T_y * math.exp(-r * T_y) * _norm_cdf(d2) * 1.0
    rho_put = -K * T_y * math.exp(-r * T_y) * _norm_cdf(-d2) * 1.0
    rho = rho_call if right == "CALL" else rho_put

    # Second-order greeks from the engine builder for consistency.
    g2 = compute_second_order_greeks(
        underlying_price=S, strike=K, time_to_expiry_yrs=T_y, implied_vol=sigma,
        right=right, r=r, q=q,
    )

    return PricedOption(
        strike=K,
        right=right,
        implied_vol=iv,
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega / 100.0,   # this repo's stored convention (== ThetaData /100)
        rho=rho / 100.0,     # same
        vanna=g2.vanna,
        charm=g2.charm,
    )
