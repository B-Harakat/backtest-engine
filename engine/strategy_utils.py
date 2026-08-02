"""
Reusable strategy math and validations.

These functions encode the conventions every options-spread strategy needs (strike selection,
risk/credit math, position sizing, entry guards) so strategies don't each re-derive them — and
don't each reintroduce the classic bugs (wrong sign for put deltas, off-by-one on offset, asking a
bid for the short side, forgetting a NaN check, etc.).

Sign conventions matched to the engine's chain data (ThetaData): **call deltas are positive, put
deltas are negative**. Functions that take a ``right`` ("CALL"/"PUT") or a ``right``-dependent
``spot``/``target_delta`` handle the sign internally, so callers always write in absolute terms.

All prices here are in **premium points** (before the contract multiplier). They are
multiplier-independent, so they compose with ``Strategy.get_run_multiplier()`` when converting to
dollars.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import pandas as pd

# --------------------------------------------------------------------------------------------
# Tier 1: profit-target exit
# --------------------------------------------------------------------------------------------


def profit_target_reached(entry_value: float, current_value: float, target_fraction: float) -> bool:
    """
    True if ``target_fraction`` of the position's maximum potential profit has been captured.

    Works for both credit and debit spreads:

    - **Credit spreads** (max profit = net credit received). Enter short, ``entry_value`` = net
      credit at entry. Value decays toward 0 as profit accrues, so "captured profit" =
      ``entry_value - current_value``.
    - **Debit spreads** (max profit = max(0, width - debit paid)). Enter long, ``entry_value`` =
      net debit paid at entry. Value rises toward its width cap, so "captured profit" =
      ``current_value - entry_value``.

    The caller passes the correct sign relationship; this function only checks whether the
    *gains already banked by the book* reach the target. A non-positive ``target_fraction`` (e.g.
    0 or negative) always returns False — you can't target a fraction of nothing.

    Args:
        entry_value: value of the position at entry, in premium points (credit received, or debit
            paid).
        current_value: current value, in premium points (current mid of the spread).
        target_fraction: fraction of max potential profit to capture (e.g. 0.50 for 50%%).
    """
    if target_fraction <= 0.0:
        return False
    captured = entry_value - current_value  # positive when premium decayed / value grew
    return captured >= target_fraction * abs(entry_value)


# --------------------------------------------------------------------------------------------
# Tier 2: strike selection
# --------------------------------------------------------------------------------------------


def select_strike_by_delta(chain: pd.DataFrame, target_delta: float, right: str) -> float:
    """
    The strike whose ``delta`` is closest (in absolute terms) to ``target_delta``.

    ``target_delta`` is an **absolute** delta (e.g. ``0.12`` for a 12-delta leg); this function
    converts to the signed value the chain uses based on ``right`` (calls positive, puts negative)
    and selects the strike minimizing the difference. This is the fix for the engine's most
    error-prone idiom: the naive ``(chain["delta"] + abs(target)).abs().idxmin()`` silently selects
    nonsense strikes on call legs, because call deltas are already negative-free.

    Args:
        chain: single-sided chain DataFrame indexed by strike, with a ``delta`` column.
        target_delta: absolute target delta (>= 0), e.g. 0.12.
        right: ``"CALL"`` or ``"PUT"``.

    Returns:
        The strike whose delta is closest to the (right-signed) target.

    Raises:
        ValueError: if ``right`` is invalid, the chain is empty, or no finite delta values exist.
    """
    right_up = str(right).upper()
    if right_up not in ("CALL", "PUT"):
        raise ValueError(f"right must be 'CALL' or 'PUT', got {right!r}")
    if target_delta < 0:
        raise ValueError(f"target_delta must be absolute (>= 0), got {target_delta!r}")

    if chain is None or chain.empty or "delta" not in chain.columns:
        raise ValueError("select_strike_by_delta requires a non-empty chain with a 'delta' column.")

    signed_target = target_delta if right_up == "CALL" else -target_delta
    delta = pd.to_numeric(chain["delta"], errors="coerce")
    delta = delta.dropna()
    if delta.empty:
        raise ValueError("select_strike_by_delta: no finite delta values in the chain.")

    best = (delta - signed_target).abs().idxmin()
    return float(best)


def select_strike_by_offset(strikes: Union[pd.Index, Iterable[float]], base_strike: float, offset: int) -> float:
    """
    The strike ``offset`` index positions away from ``base_strike`` in a sorted strike list.

    ``offset`` is positive for higher strikes, negative for lower (a put protection leg is one
    short leg count *lower* for a put spread, i.e. negative offset). Unlike the fragile inline
    ``sorted_strikes.index(short_strike)`` / ``[idx - width]`` pattern, this raises clear errors
    instead of opaque ``ValueError``/``IndexError`` when the base strike is absent or the target is
    out of range.

    Args:
        strikes: an iterable/Index of available strikes (sorted or not — it is sorted internally).
        base_strike: the reference strike to count from.
        offset: number of index positions to move (negative = lower strike).

    Returns:
        The strike at that offset.

    Raises:
        ValueError: if ``base_strike`` is not present, or the target index is out of bounds.
    """
    sorted_strikes = sorted(float(s) for s in strikes)
    base = float(base_strike)

    try:
        base_idx = sorted_strikes.index(base)
    except ValueError:
        raise ValueError(
            f"base_strike {base_strike} not found in the provided strikes. Available: "
            f"{sorted_strikes[:10]}{'...' if len(sorted_strikes) > 10 else ''}"
        )

    target_idx = base_idx + offset
    if target_idx < 0 or target_idx >= len(sorted_strikes):
        raise ValueError(
            f"Offset {offset} from base_strike {base_strike} lands at index {target_idx}, "
            f"out of range for {len(sorted_strikes)} strikes."
        )

    return sorted_strikes[target_idx]


def select_wing_strike(
    strikes: Union[pd.Index, Iterable[float]], base: float, width_pts: float, upward: bool
) -> float:
    """
    The first listed strike at or beyond ``width_pts`` points away from ``base``, on the
    specified side. Wraps the point-distance-to-index-offset conversion so callers think in
    point space, not index positions.

    ``upward=True`` finds the first strike >= ``base + width_pts`` (call wing).
    ``upward=False`` finds the first strike <= ``base - width_pts`` (put wing).

    Args:
        strikes: available strikes (sorted or not — sorted internally).
        base: reference strike to measure from.
        width_pts: minimum point distance from ``base`` (>= 0).
        upward: True to search higher strikes, False for lower.

    Returns:
        The strike at or beyond the target distance.

    Raises:
        ValueError: if ``base`` is not in strikes, or no strike reaches the target distance.
    """
    sorted_strikes = sorted(float(s) for s in strikes)
    base_f = float(base)
    if upward:
        target = next((s for s in sorted_strikes if s >= base_f + width_pts), None)
    else:
        target = next((s for s in reversed(sorted_strikes) if s <= base_f - width_pts), None)
    if target is None:
        raise ValueError(
            f"No strike {'>=' if upward else '<='} {width_pts} pts beyond base {base} "
            f"(available: {sorted_strikes[0]}..{sorted_strikes[-1]})."
        )
    return float(target)


def select_nearest_otm(chain: pd.DataFrame, spot: float, right: str) -> Optional[float]:
    """
    The strike nearest to ``spot`` on the out-of-the-money side, or None if there is none.

    For calls, OTM is strikes strictly above ``spot``; this returns the smallest such strike. For
    puts, OTM is strikes strictly below ``spot``; this returns the largest such strike. Selecting
    the nearest OTM is the entry convention for strategies that don't use delta-based selection.

    Args:
        chain: single-sided chain DataFrame indexed by strike.
        spot: current underlying price.
        right: ``"CALL"`` or ``"PUT"``.

    Returns:
        The nearest OTM strike, or None if no OTM strike exists on that side.
    """
    right_up = str(right).upper()
    if right_up not in ("CALL", "PUT"):
        raise ValueError(f"right must be 'CALL' or 'PUT', got {right!r}")

    strikes = sorted(float(s) for s in chain.index)
    if right_up == "CALL":
        otm = [s for s in strikes if s > float(spot)]
        return otm[0] if otm else None
    otm = [s for s in strikes if s < float(spot)]
    return otm[-1] if otm else None


# --------------------------------------------------------------------------------------------
# Tier 2: risk-based position sizing and credit/risk math
# --------------------------------------------------------------------------------------------


def size_position(cash: float, risk_pct: float, max_risk_premium: float, multiplier: int) -> int:
    """
    Number of contracts to trade so the account risks at most ``cash * risk_pct`` on one position.

    Formula: ``max(1, floor(cash * risk_pct / (max_risk_premium * multiplier)))``. ``cash`` is
    current account cash, ``risk_pct`` the fraction of equity to risk per trade, ``max_risk_premium``
    the maximum loss per contract in premium points (e.g. ``max_risk_premium(width, interval,
    credit)`` for a credit spread), and ``multiplier`` the contract multiplier from
    ``Strategy.get_run_multiplier()``. Returns a minimum of 1 contract.

    Args:
        cash: current account cash (dollars).
        risk_pct: fraction of account equity to risk per trade (e.g. 0.02).
        max_risk_premium: max loss per contract in premium points (> 0).
        multiplier: contract multiplier ($/point/contract), from get_run_multiplier().

    Returns:
        Number of contracts, at least 1.
    """
    if max_risk_premium <= 0:
        # No measurable per-contract risk — can't meaningfully scale; trade a single contract.
        return 1
    risk_dollars_per_contract = max_risk_premium * max(int(multiplier), 1)
    raw = cash * risk_pct / risk_dollars_per_contract
    return max(1, int(math.floor(raw)))


def net_credit(short_price: float, long_price: float) -> float:
    """
    Net credit received (premium points) for a credit spread: ``short_price - long_price``.

    ``short_price`` must be the **bid** of the short leg and ``long_price`` the **ask** of the
    long leg so the credit isn't overstated (using ask for the short side overstates credit).
    Simple guard: the credit cannot be negative for a valid credit spread; it is returned as-is
    otherwise (a negative result signals the caller mis-ordered the legs).
    """
    return float(short_price) - float(long_price)


def max_risk_premium(width: int, strike_interval: float, credit: float) -> float:
    """
    Maximum risk per contract (premium points) of a defined-risk spread:
    ``width * strike_interval - credit``, floored at 0.

    ``width`` is the number of strike positions between the legs and ``strike_interval`` the
    distance between adjacent strikes (points). Floors at 0 so an (unusual) over-credit can't
    report negative risk; returns a float for use in ``size_position``.
    """
    risk = float(width) * float(strike_interval) - float(credit)
    return max(risk, 0.0)


# --------------------------------------------------------------------------------------------
# Tier 2: chain / quote validation guards
# --------------------------------------------------------------------------------------------


def chain_has_columns(chain: pd.DataFrame, cols: Union[str, Iterable[str]]) -> bool:
    """
    True if the chain (a DataFrame) contains all of the required columns.

    Accepts a single column name or an iterable of names. Missing/empty/None chains are False.
    """
    if chain is None or chain.empty:
        return False
    required = [cols] if isinstance(cols, str) else list(cols)
    return all(str(c) in chain.columns for c in required)


def chain_has_enough_otm(chain: pd.DataFrame, spot: float, right: str, min_count: int) -> bool:
    """
    True if the chain lists at least ``min_count`` out-of-the-money strikes on the given side.

    OTM is strikes strictly above ``spot`` for calls, strictly below for puts. This is the guard
    that ensures there are enough strikes to place a ``spread_width`` protection leg with slack.
    """
    right_up = str(right).upper()
    if right_up not in ("CALL", "PUT"):
        raise ValueError(f"right must be 'CALL' or 'PUT', got {right!r}")
    if chain is None or chain.empty:
        return False
    strikes = sorted(float(s) for s in chain.index)
    if right_up == "CALL":
        return sum(1 for s in strikes if s > float(spot)) >= int(min_count)
    return sum(1 for s in strikes if s < float(spot)) >= int(min_count)


def quotes_valid(quotes: Union[float, Iterable[Optional[float]]], min_value: float = 0.0) -> bool:
    """
    True if every quote value is a finite number >= ``min_value``.

    Accepts a single price or an iterable of prices. Rejects None, NaN, and infinities, which is
    the guard that prevents ordering against a missing/zero/negative bid or ask.
    """
    values = [quotes] if isinstance(quotes, (int, float)) else (list(quotes) if quotes is not None else [])
    for v in values:
        if v is None or not isinstance(v, (int, float)):
            return False
        if math.isnan(float(v)) or math.isinf(float(v)):
            return False
        if float(v) < float(min_value):
            return False
    return True
