"""
Trading calendar helpers — the master clock the time loop iterates over.

Uses `pandas_market_calendars` for the actual XNYS holiday/early-close schedule rather than
hand-rolling one — that library already correctly encodes things like Thanksgiving, the day
after Thanksgiving's early close, Christmas Eve early closes in years it applies, etc. Don't
reinvent this; getting a holiday calendar subtly wrong is a classic source of silent backtest
bugs (phantom trading days, or missing an early-close flatten).

Timestamp convention: naive wall-clock America/New_York, matching `engine.thetadata_client`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterator

import pandas as pd
import pandas_market_calendars as mcal

ET = "America/New_York"
_CALENDAR = mcal.get_calendar("XNYS")

# Bars are timestamped at the START of their interval (matches ThetaData's OHLC/quote/greeks
# convention — see thetadata_client.py). A 1-minute bar timestamped 15:59 covers [15:59, 16:00),
# i.e. the last minute of the regular session. We therefore do NOT emit a 16:00 bar — the session
# close itself is a settlement/mark-to-market instant handled separately (see settlement.py),
# not a tradeable bar.


def trading_days(start: date, end: date) -> pd.DatetimeIndex:
    """All XNYS session dates in [start, end], holidays/weekends excluded."""
    schedule = _CALENDAR.schedule(start_date=start, end_date=end)
    return schedule.index


def session_bounds(day: date) -> tuple[datetime, datetime]:
    """
    (first_bar_start, last_bar_start) for a single session, naive ET, respecting early closes
    (e.g. day after Thanksgiving). Returns the timestamp of the FIRST minute bar and the LAST
    minute bar (not the close time itself — see module docstring).
    """
    open_ts, close_ts = _session_open_close(day)
    first_bar = open_ts
    last_bar = close_ts - pd.Timedelta(minutes=1)
    return first_bar.to_pydatetime(), last_bar.to_pydatetime()


def _session_open_close(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    schedule = _CALENDAR.schedule(start_date=day, end_date=day)
    if schedule.empty:
        raise ValueError(f"{day} is not an XNYS trading day")
    open_ts = schedule.iloc[0]["market_open"].tz_convert(ET).tz_localize(None)
    close_ts = schedule.iloc[0]["market_close"].tz_convert(ET).tz_localize(None)
    return open_ts, close_ts


def session_open(day: date) -> datetime:
    """The actual market open instant for `day` (same as session_bounds's first_bar)."""
    open_ts, _ = _session_open_close(day)
    return open_ts.to_pydatetime()


def session_close(day: date) -> datetime:
    """
    The actual market close instant for `day` — unlike `session_bounds`'s `last_bar`, this is
    NOT shifted back a minute for the bar-start convention. Respects early closes.

    Use this (paired with `session_open`) when you need the true [open, close] window a session's
    data can ever cover — e.g. when warming a contract's or chain's full-day cache. Requesting a
    wider window (like the full calendar day) will make `data_store.compute_gaps`/
    `_chain_compute_gaps` see a permanent, unsatisfiable "gap" outside trading hours and re-fetch
    it on every single call, forever, since no data will ever exist there to close that gap.
    """
    _, close_ts = _session_open_close(day)
    return close_ts.to_pydatetime()


def trading_minutes(start: date, end: date) -> Iterator[datetime]:
    """
    Every 1-minute bar-start timestamp across [start, end], in order, skipping non-trading days
    and respecting early closes. This is the sequence the runner's time loop iterates over.
    """
    for day in trading_days(start, end):
        first_bar, last_bar = session_bounds(day.date())
        ts = first_bar
        while ts <= last_bar:
            yield ts
            ts += pd.Timedelta(minutes=1)


def is_last_bar_of_day(ts: datetime, day: date | None = None) -> bool:
    """True if `ts` is the final minute bar of its session (respects early closes)."""
    _, last_bar = session_bounds(day if day is not None else ts.date())
    return ts == last_bar