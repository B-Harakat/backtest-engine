"""
Second-order greeks, computed in Python from the implied vol ThetaData already gives us — NOT
re-derived independently. Using their IV keeps these consistent with the 1st-order greeks the
Options Standard plan provides.

Model conventions matched to ThetaData's methodology (Black-Scholes, European exercise, correct
for cash-settled index products like XSP/SPX weeklies/RUT/NDX/VIX; not correct for American-style
equity/ETF options, but this engine never simulates physical settlement for those). Dividend
yield q = 0 by default; no external deps beyond stdlib (normal pdf/cdf via math.erf).

Units (all standard per-unit conventions):
    delta:  d(price)/d(underlying)
    gamma:  d(delta)/d(underlying)
    vanna:  d(delta)/d(vol), vol in decimal (0.20 = 20%)
    charm:  d(delta)/d(time), in "per year" units (divide by 365 for per-day decay)
ThetaData's returned `rho`/`vega` are divided by 100 to reach these standard units (done once in
`engine.thetadata_client`); gamma/vanna/charm here are already standard per-unit.

Risk-free rate:
    `engine.thetadata_client` passes `rate_value=0.0` (its `ASSUMED_RATE_PCT`) on every greeks
    request, and this module computes with `r=0.0`, keeping both sides on the same rate. Change
    them together if you want a nonzero rate — for short-dated contracts it barely matters.

Time-to-expiration, 0DTE wrinkle:
    ThetaData's "latest" methodology computes TTE from the actual quote timestamp for contracts
    under 7 DTE (real elapsed time, same as `time_to_expiry_years` here) and floors same-day
    expirations at ~1 hour to avoid the T→0 blowup in Black-Scholes. `time_to_expiry_years` below
    does NOT apply that floor, so in the final hour of a 0DTE session the IV-derived
    gamma/vanna/charm can very slightly diverge from ThetaData's own delta/theta/vega. Minor and
    bounded; only matters in the closing minutes of expiration day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal

Right = Literal["CALL", "PUT"]

# Default settlement cutoff for computing time-to-expiration. XSP and SPX weeklies/0DTE are
# PM-settled against the 4:00pm ET close, but this can vary by product/expiration cadence (SPX
# standard monthlies are AM-settled) — override via run_backtest's `settlement_time` if needed.
DEFAULT_CASH_SETTLEMENT_TIME_ET = time(16, 0)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def time_to_expiry_years(as_of: datetime, expiration: date, settlement_time: time = DEFAULT_CASH_SETTLEMENT_TIME_ET) -> float:
    """
    Years between `as_of` and the contract's settlement instant. Uses actual elapsed time (not
    whole trading days), which matters for 0DTE contracts where T can be a few hours, not a day.
    """
    settlement_dt = datetime.combine(expiration, settlement_time)
    delta_seconds = (settlement_dt - as_of).total_seconds()
    return max(delta_seconds, 0.0) / (365.0 * 24.0 * 3600.0)


@dataclass(frozen=True)
class SecondOrderGreeks:
    gamma: float
    vanna: float
    charm: float


def compute_second_order_greeks(
    underlying_price: float,
    strike: float,
    time_to_expiry_yrs: float,
    implied_vol: float,
    right: Right,
    r: float = 0.0,
    q: float = 0.0,
) -> SecondOrderGreeks:
    """
    Black-Scholes gamma, vanna, charm. `right` only affects charm (via the dividend term); with
    q=0 (ThetaData's default), charm is numerically identical for calls and puts.
    """
    S, K, T, sigma = underlying_price, strike, time_to_expiry_yrs, implied_vol

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return SecondOrderGreeks(gamma=0.0, vanna=0.0, charm=0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc_q = math.exp(-q * T)
    pdf_d1 = _norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
    vanna = -disc_q * pdf_d1 * d2 / sigma

    charm_common = pdf_d1 * ((r - q) / (sigma * sqrt_T) - d2 / (2.0 * T))
    if right == "CALL":
        charm = -disc_q * (charm_common - q * _norm_cdf(d1))
    else:
        charm = -disc_q * (charm_common + q * _norm_cdf(-d1))

    return SecondOrderGreeks(gamma=gamma, vanna=vanna, charm=charm)