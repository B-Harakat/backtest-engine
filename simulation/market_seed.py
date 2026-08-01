"""
Derive the synthetic-market opening seeds (`s0` and `V_0`) from the real cached historical
ThetaData chain for a given 0DTE session day.

Assumption: the user runs a real `scripts/run_backtest.py` for the target date *before*
`scripts/run_mc.py`. That real run warms the local Parquet cache under `data/chains/` for the
day's 0DTE expiration. This module reads that cached chain — no network, no new ThetaData calls —
to seed the Bates simulator from the market's actual opening level and ATM volatility.

Naming/meaning:
  - `s0` (opening spot): the chain's `underlying_price` at the session's first minute that has
    real (non-padded) data. ThetaData pads the very first bar (09:30) with zeros/NaNs, so we scan
    forward past that to the first valid quote (usually 09:31).
  - `V_0` (opening IV of the ATM option): `implied_vol` of the strike closest to that spot on the
    SAME-DAY chain, at that opening minute. It's an annualized decimal (0.1447 = 14.47%);
    `BatesParams.V_0` takes VARIANCE, so `V_0 = iv**2`.

Expected cache layout (see `engine.data_store`):
    `data_dir/chains/{underlying}_{day:YYYYMMDD}_call_chain.parquet` (call side), a
    (timestamp, strike) MultiIndex with `underlying_price` and `implied_vol` columns.

Raises with a clear error if the cache is absent or contains no valid opening quote.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from engine.calendar import session_open
from engine.data_store import DEFAULT_DATA_DIR, _chain_path_for

#: How many minutes past the session open to scan for the first real quote. The feed pads the
#: open with a zero bar; real quotes normally appear within 1-2 minutes, so 15 is generous.
MAX_OPEN_MINUTES = 15


@dataclass(frozen=True)
class OpeningMarket:
    """Seeds for one simulated session, derived from the real cached chain."""

    spot: float            # s0: opening underlying price.
    v0: float              # V_0: opening ATM implied vol (decimal, e.g. 0.1447).
    atm_strike: float      # the struck option whose IV was used as V_0.
    opening_timestamp: datetime  # the first minute with a real quote (usually 09:31).


def _atom_zero_or_nan(x) -> bool:
    try:
        return bool(pd.isna(x)) or bool(float(x) <= 0)
    except (TypeError, ValueError):
        return True


def opening_market_from_cache(
    underlying: str,
    day: date,
    data_dir: Path = DEFAULT_DATA_DIR,
    max_minutes: int = MAX_OPEN_MINUTES,
) -> OpeningMarket:
    """Read the opening spot/ATM-IV seeds for ``day`` from the real cached 0DTE chain."""
    # The "call" chain is the one that carries the full strike set + underlying_price we want.
    path = _chain_path_for(underlying, day, data_dir, right="call")
    if not path.exists():
        raise FileNotFoundError(
            f"No cached chain for {underlying} {day} at {path}. Run a real backtest for this "
            f"date first (scripts/run_backtest.py --start {day} --end {day}) so the ThetaData "
            f"cache is populated, before running a synthetic Monte Carlo."
        )
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["timestamp", "strike"]:
        raise ValueError(f"Unexpected chain layout in {path} -- expected (timestamp, strike) MultiIndex.")
    for col in ("underlying_price", "implied_vol"):
        if col not in df.columns:
            raise ValueError(f"Chain {path} is missing column {col!r}; cannot derive opening seeds.")

    ts_level = df.index.get_level_values("timestamp")
    open_ts = pd.Timestamp(session_open(day))
    for minute in pd.date_range(open_ts, periods=int(max_minutes) + 1, freq="min"):
        sub = df.loc[ts_level == minute]
        if sub.empty:
            continue
        spot_series = sub["underlying_price"].dropna()
        if spot_series.empty:
            continue
        valid = sub[~sub["implied_vol"].map(_atom_zero_or_nan)]
        if valid.empty:
            continue
        spot = float(spot_series.iloc[0])
        strikes = valid.index.get_level_values("strike").unique()
        atm = float(min(strikes, key=lambda k: abs(k - spot)))
        atm_iv = float(valid.loc[(valid.index.get_level_values("strike") == atm), "implied_vol"].iloc[0])
        if atm_iv <= 0:
            continue
        return OpeningMarket(
            spot=spot,
            v0=atm_iv**2,  # V_0 is VARIANCE in BatesParams; IV^2 == variance.
            atm_strike=atm,
            opening_timestamp=minute.to_pydatetime(),
        )

    raise ValueError(
        f"No valid opening quote found in {path} within {max_minutes} min of {open_ts}. "
        f"The cache may be empty for that minute; re-run the real backtest for {day}."
    )
