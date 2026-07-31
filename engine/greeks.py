"""
Second-order greeks, computed in Python from the implied vol ThetaData already gives us —
NOT re-derived independently. Using their IV (rather than fitting our own) is what keeps these
numbers consistent with the 1st-order greeks you're paying for on the Standard plan.

Model conventions matched to ThetaData's stated methodology (see their "Option Greeks" article):
  - Black-Scholes, European exercise (correct for cash-settled index products like XSP/SPX
    weeklies/RUT/NDX/VIX; NOT correct for American-style equity/ETF options — this engine never
    simulates physical settlement for those at all, see engine.runner's force-close mechanism,
    so European-style Greeks are used uniformly regardless of the underlying's real-world style)
  - dividend yield q = 0 by default (ThetaData ignores dividends unless you pass annual_div)
  - no external deps beyond stdlib — normal pdf/cdf via math.erf, not scipy

One thing ThetaData's docs call out explicitly: their returned `rho` and `vega` values must be
divided by 100 to reach the standard per-unit convention. Gamma/vanna/charm below are computed
in the same standard per-unit convention as a result — see NOTE_ON_UNITS below before comparing
against a different provider or a spreadsheet model.

NOTE_ON_UNITS:
    delta:  d(price)/d(underlying), no scaling
    gamma:  d(delta)/d(underlying), no scaling
    vanna:  d(delta)/d(vol), where vol is in decimal (0.20 = 20%) — divide by 100 if you want
            "per 1 vol point" to match how ThetaData scales vega
    charm:  d(delta)/d(time), in "per year" units — divide by 365 for "per day" decay, which is
            usually the more intuitive unit to look at for 0DTE/short-dated contracts

Risk-free rate:
    ThetaData defaults to SOFR (a few percent, not 0) unless you pass `rate_value` explicitly on
    the request. Rather than guess at their SOFR value and hope it roughly matches this module's
    `r=0.0` default, `engine.thetadata_client` explicitly passes `rate_value=0.0` (see its
    `ASSUMED_RATE_PCT` constant) on every greeks request — forcing BOTH their 1st-order greeks
    and this module's 2nd-order greeks to use the exact same r, guaranteeing consistency instead
    of hoping for it. Change `ASSUMED_RATE_PCT` there (and the default here) together if you want
    a nonzero rate — for short-dated contracts the simplification barely matters either way.

Time-to-expiration, and a consistency wrinkle for 0DTE:
    ThetaData's default ("latest") greeks methodology computes TTE from the actual quote
    timestamp for contracts under 7 DTE (real elapsed time, same idea as
    `time_to_expiry_years` below), and a whole-number DTE beyond that. Their API reference also
    notes "latest" floors at a minimum of ~1 hour for same-day expirations specifically (to avoid
    the numerical blowup of T→0 in Black-Scholes). This module's `time_to_expiry_years` does NOT
    apply that floor by default — for a 0DTE contract in its final hour, our IV-derived
    gamma/vanna/charm will therefore very slightly diverge from what's internally consistent with
    ThetaData's own delta/theta/vega in that last hour. This is a minor, bounded effect (it only
    matters in the closing minutes of expiration day), but worth knowing about if you see anything
    odd in gamma right at the end of a 0DTE session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal

Right = Literal["CALL", "PUT"]

# Default settlement cutoff for computing time-to-expiration, used unless overridden.
# XSP and SPX weeklies/0DTE are PM-settled against the normal 4:00pm ET close, but this varies by
# product AND by expiration cadence (SPX's standard monthly expirations are AM-settled against a
# special print) — VERIFY this against current OCC/Cboe specs for whatever you actually trade,
# and override via run_backtest's `settlement_time` parameter rather than editing this default.
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
    Black-Scholes gamma, vanna, charm. `right` only affects charm (via the dividend term) — with
    q=0 (ThetaData's default), charm is numerically identical for calls and puts, which is a
    useful sanity check if you ever see them diverge with q=0 still set.
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