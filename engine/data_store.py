"""
Local-storage-first data access.

`get_bars()` is the single entry point the rest of the engine should ever call for price/quote/
greeks data. It transparently handles: read local cache -> figure out what's missing -> fetch only
the missing range -> persist -> return the full requested range.

The actual ThetaData HTTP calls are injected as a `fetch_fn` callback rather than imported directly,
so this module has zero knowledge of ThetaData's API shape and can be unit-tested with a fake
fetcher. Wire it up with `engine.thetadata_client.fetch_from_thetadata` in the runner.

One Parquet file per contract for its entire lifetime (options are short-lived; the underlying
is a single growing file). This is intentionally simple — if the underlying file ever gets
unwieldy, revisit monthly partitioning then, not before.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

import pandas as pd
import numpy as np

from engine.entities import Contract
from engine.calendar import session_close, session_open

logger = logging.getLogger(__name__)

FetchFn = Callable[[Contract, datetime, datetime], pd.DataFrame]

# Distinct from FetchFn: fetches every listed strike for an (underlying, expiration) at once, as
# a (timestamp, strike)-MultiIndex DataFrame -- not a single contract's time series. Used by
# DataProvider's chain_snapshot(), wired up with
# engine.thetadata_client.fetch_chain_from_thetadata in the runner. This is how the engine sources
# "what's the underlying price right now" and "what strikes are listed" without a separate
# Index/Stock subscription -- see thetadata_client's module docstring.
ChainFetchFn = Callable[[str, date, datetime, datetime, str], pd.DataFrame]

DEFAULT_DATA_DIR = Path(os.environ.get("BACKTEST_DATA_DIR", "data"))


def local_path_for(contract: Contract, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    subfolder = "options" if contract.is_option else "underlying"
    return data_dir / subfolder / f"{contract.key}.parquet"


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Cached file {path} does not have a DatetimeIndex — inspect it manually.")
    return df


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_parquet(path)


def compute_gaps(
    cached: pd.DataFrame, start: datetime, end: datetime
) -> list[tuple[datetime, datetime]]:
    """
    Return the sub-range(s) of [start, end] not covered by `cached`.

    Simple version: only checks the leading and trailing edges of the cached range. It does NOT
    detect holes in the middle (e.g. a previous partial fetch that skipped a week). That's a
    deliberate simplification — start here, and only add interior-gap detection if you actually
    observe stale/missing data slipping through (holidays/weekends are the caller's problem to
    filter out via the trading calendar, not this function's).

    IMPORTANT — callers must request a range the data can actually cover. If `[start, end]`
    extends beyond what will ever exist (e.g. midnight-to-midnight for a feed that only ever has
    regular-trading-hours data), the "gap" at that edge is unsatisfiable: `cached_start`/
    `cached_end` will never move to close it, so every single call will detect the same gap and
    re-fetch it — forever, on every run, even with a fully "cached" range in every meaningful
    sense. This is exactly the bug that motivated this warning: see `engine.calendar.session_open`/
    `session_close` for how callers should scope a warm request to a real session window instead.
    """
    if cached.empty:
        return [(start, end)]

    cached_start, cached_end = cached.index.min(), cached.index.max()
    gaps: list[tuple[datetime, datetime]] = []

    if start < cached_start:
        gaps.append((start, cached_start))
    if end > cached_end:
        gaps.append((cached_end, end))

    return gaps


def get_bars(
    contract: Contract,
    start: datetime,
    end: datetime,
    fetch_fn: FetchFn,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """
    Return all bars for `contract` covering [start, end], fetching and persisting any missing
    range from `fetch_fn` first.

    `fetch_fn(contract, gap_start, gap_end) -> DataFrame` must return a DataFrame indexed by
    tz-consistent datetime, with whatever columns you want cached (OHLC, bid, ask, greeks, iv —
    see engine.thetadata_client for the expected shape).
    """
    path = local_path_for(contract, data_dir)
    cached = _load_cache(path)

    gaps = compute_gaps(cached, start, end)
    if gaps:
        logger.debug("get_bars(%s): requested [%s, %s], %d gap(s) to fetch: %s", contract.key, start, end, len(gaps), gaps)
        fetched_frames = []
        for gap_start, gap_end in gaps:
            t0 = perf_counter()
            frame = fetch_fn(contract, gap_start, gap_end)
            elapsed = perf_counter() - t0
            logger.debug(
                "get_bars(%s): fetched gap [%s, %s] -> %d rows in %.2fs",
                contract.key, gap_start, gap_end, len(frame), elapsed,
            )
            fetched_frames.append(frame)
        cached = pd.concat([cached, *fetched_frames])
        cached = cached[~cached.index.duplicated(keep="last")].sort_index()
        _save_cache(cached, path)

    return cached.loc[start:end]


def cache_status(contract: Contract, data_dir: Path = DEFAULT_DATA_DIR) -> Optional[tuple[datetime, datetime, int]]:
    """Quick inspection helper: (earliest, latest, row_count) for a contract's cache, or None."""
    path = local_path_for(contract, data_dir)
    cached = _load_cache(path)
    if cached.empty:
        return None
    return cached.index.min(), cached.index.max(), len(cached)


def _chain_path_for(underlying: str, expiration: date, data_dir: Path = DEFAULT_DATA_DIR, right: str = "call") -> Path:
    # Calls keep the original (pre-multi-right) filename so existing downloaded caches aren't
    # orphaned by this parameter's addition -- only a non-default right gets a suffix.
    suffix = "" if right == "call" else f"_{right}"
    return data_dir / "chains" / f"{underlying}_{expiration:%Y%m%d}{suffix}_chain.parquet"


def _load_chain_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["timestamp", "strike"]:
        raise ValueError(f"Cached chain file {path} does not have the expected (timestamp, strike) MultiIndex.")
    return df


def _save_chain_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_parquet(path)


def _chain_compute_gaps(cached: pd.DataFrame, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Same idea as `compute_gaps`, but reads the `timestamp` level of a (timestamp, strike)
    MultiIndex rather than a plain DatetimeIndex — same leading/trailing-edge-only simplification
    applies (see compute_gaps's docstring)."""
    if cached.empty:
        return [(start, end)]

    ts_level = cached.index.get_level_values("timestamp")
    cached_start, cached_end = ts_level.min(), ts_level.max()
    gaps: list[tuple[datetime, datetime]] = []

    if start < cached_start:
        gaps.append((start, cached_start))
    if end > cached_end:
        gaps.append((cached_end, end))

    return gaps


def get_chain(
    underlying: str,
    expiration: date,
    start: datetime,
    end: datetime,
    fetch_fn: ChainFetchFn,
    data_dir: Path = DEFAULT_DATA_DIR,
    right: str = "call",
) -> pd.DataFrame:
    """
    Same local-storage-first pattern as `get_bars` — read local cache, fetch only what's missing,
    persist, return the full range — generalized for chain data (every strike for one expiration
    and one side (`right`: "call" or "put"), a (timestamp, strike) MultiIndex instead of a plain
    per-contract DatetimeIndex). Cached under `data_dir/chains/...` — see `_chain_path_for`.
    """
    path = _chain_path_for(underlying, expiration, data_dir, right)
    cached = _load_chain_cache(path)

    gaps = _chain_compute_gaps(cached, start, end)
    if gaps:
        logger.debug(
            "get_chain(%s, %s, %s): requested [%s, %s], %d gap(s) to fetch: %s",
            underlying, expiration, right, start, end, len(gaps), gaps,
        )
        fetched_frames = []
        for gap_start, gap_end in gaps:
            t0 = perf_counter()
            frame = fetch_fn(underlying, expiration, gap_start, gap_end, right)
            elapsed = perf_counter() - t0
            logger.debug(
                "get_chain(%s, %s, %s): fetched gap [%s, %s] -> %d rows in %.2fs",
                underlying, expiration, right, gap_start, gap_end, len(frame), elapsed,
            )
            fetched_frames.append(frame)
        cached = pd.concat([cached, *fetched_frames])
        cached = cached[~cached.index.duplicated(keep="last")].sort_index()
        _save_chain_cache(cached, path)

    ts_level = cached.index.get_level_values("timestamp")
    return cached.loc[(ts_level >= start) & (ts_level <= end)]


class DataProvider:
    """
    Thin in-memory layer over `get_bars`/`get_chain`, used by the runner/strategy during the
    actual backtest loop. Both already do their own Parquet-level caching, but calling them every
    single minute of the loop would still mean a disk read + gap-check on every call — this class
    holds one already-loaded DataFrame per contract (and per chain) for the life of a backtest
    run, so the loop itself never touches disk once warmed.

    Usage: call `warm(contract, start, end)` once per contract before the time loop starts (this
    is where the one-time Parquet-read-or-ThetaData-fetch cost happens), then use
    `bar_at`/`bars_up_to` freely inside the loop. `chain_snapshot` is lazier — it warms the whole
    session for an (underlying, expiration) automatically the first time it's requested, since a
    strategy typically doesn't know in advance which expiration's chain it'll want.
    """

    def __init__(self, fetch_fn: FetchFn, data_dir: Path = DEFAULT_DATA_DIR, chain_fetch_fn: Optional[ChainFetchFn] = None):
        self.fetch_fn = fetch_fn
        self.data_dir = data_dir
        self.chain_fetch_fn = chain_fetch_fn
        self._frames: dict[str, pd.DataFrame] = {}
        self._chains: dict[tuple[str, date, str], pd.DataFrame] = {}
        self._quote_arrays: dict[str, Optional[tuple]] = {}  # lazily built by quote_at()

    def warm(self, contract: Contract, start: datetime, end: datetime) -> None:
        self._frames[contract.key] = get_bars(contract, start, end, self.fetch_fn, self.data_dir)
        self._quote_arrays.pop(contract.key, None)  # invalidate any cached fast-path arrays

    def bars_up_to(self, contract: Contract, ts: datetime) -> pd.DataFrame:
        """All warmed bars for `contract` at or before `ts`. Empty DataFrame if never warmed.
        For a lookback window (multiple rows) — for a single most-recent row, use `bar_at`
        instead, which avoids this method's O(n) slice-construction cost."""
        df = self._frames.get(contract.key)
        if df is None:
            return pd.DataFrame()
        return df.loc[:ts]

    def bar_at(self, contract: Contract, ts: datetime) -> Optional[pd.Series]:
        """
        Most recent warmed bar for `contract` at or before `ts`, or None if none exists yet.

        Uses a direct binary-search index lookup (`searchsorted`) rather than routing through
        `bars_up_to`'s `.loc[:ts]` — this method is called once per open position on EVERY bar of
        the backtest loop, and `.loc[:ts]` was rebuilding an ever-growing DataFrame slice from
        scratch on every single call just to take its last row. That was the dominant cost of a
        fully-cached backtest run once the calendar-recomputation bug (see runner.py) was fixed.

        For just bid/ask (the actual hot path — mark-to-market and fill-matching don't need the
        rest of the row), use `quote_at` instead: `.iloc[idx]` here still reconstructs a full
        pandas Series on every call (dtype-inference across the row's columns), which remained
        the dominant cost even after this method's own O(n)-slice fix.
        """
        df = self._frames.get(contract.key)
        if df is None or df.empty:
            return None
        idx = df.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return None
        return df.iloc[idx]

    def _quote_arrays_for(self, contract: Contract) -> Optional[tuple]:
        key = contract.key
        if key not in self._quote_arrays:
            df = self._frames.get(key)
            if df is None or df.empty or "bid" not in df.columns or "ask" not in df.columns:
                self._quote_arrays[key] = None
            else:
                ts_arr = df.index.values  # datetime64[ns], already sortable/searchsortable directly
                self._quote_arrays[key] = (ts_arr, df["bid"].to_numpy(), df["ask"].to_numpy())
        return self._quote_arrays[key]

    def quote_at(self, contract: Contract, ts: datetime) -> Optional[tuple[float, float]]:
        """
        (bid, ask) for `contract`'s most recent tick at or before `ts`, or None. This is the
        fast path for the backtest loop's hottest lookups (mark-to-market, order fill-matching):
        it indexes directly into cached numpy arrays rather than going through pandas' per-row
        `.iloc[]` (which reconstructs a full typed Series every call — see `bar_at`'s docstring).
        Use `bar_at`/`get_greeks` instead when you need more than bid/ask.
        """
        arrays = self._quote_arrays_for(contract)
        if arrays is None:
            return None
        ts_arr, bid_arr, ask_arr = arrays
        idx = ts_arr.searchsorted(np.datetime64(ts), side="right") - 1
        if idx < 0:
            return None
        bid, ask = bid_arr[idx], ask_arr[idx]
        if np.isnan(bid) or np.isnan(ask):
            return None
        return float(bid), float(ask)

    def is_warmed(self, contract: Contract) -> bool:
        return contract.key in self._frames

    def warm_chain(self, underlying: str, expiration: date, start: datetime, end: datetime, right: str = "call") -> None:
        if self.chain_fetch_fn is None:
            raise RuntimeError(
                "No chain_fetch_fn configured for this DataProvider -- pass chain_fetch_fn to "
                "DataProvider(...) or run_backtest(...)."
            )
        self._chains[(underlying, expiration, right)] = get_chain(
            underlying, expiration, start, end, self.chain_fetch_fn, self.data_dir, right
        )

    def chain_snapshot(self, underlying: str, expiration: date, ts: datetime, right: str = "call") -> pd.DataFrame:
        """
        Full option chain (every listed strike, one side: `right` = "call" or "put") as of the
        most recent available tick at or before `ts`, indexed by strike. Lazily warms the window
        from THE DAY THIS IS FIRST CALLED (`ts.date()`) through `expiration`'s session close on
        first request for a given (underlying, expiration, right) — not just the single instant
        asked for — so a second call anywhere in that same window (this run, or a future rerun
        reading from the persisted Parquet cache) doesn't refetch. Empty DataFrame if no
        `chain_fetch_fn` is configured, or if there's genuinely no data (e.g. no listed expiration
        that day). Calls and puts are cached and warmed completely independently of each other —
        a strategy that needs both (e.g. an iron condor) calls this twice, once per side.

        IMPORTANT: warming from `ts.date()` rather than `expiration` is deliberate, not
        incidental — for a 0DTE call this is the same date either way, but a strategy that wants
        TODAY's chain for an option expiring days from now (e.g. to pick delta-targeted strikes
        for a new position, or to check strikes ahead of a roll) needs data available STARTING
        FROM TODAY, not starting from the expiration date. Warming `[session_open(expiration),
        session_close(expiration)]` unconditionally — this method's original implementation —
        meant every timestamp in the cache was on the expiration date, so a lookup on any earlier
        day always fell before the entire cached range and silently returned empty, no error.
        Same bug class as the one `engine.runner`'s per-contract lazy-warm had; see its docstring.

        Scoped to actual market sessions (open -> close), NOT full calendar days — requesting
        midnight-to-midnight would include hours no option ever has data for, which
        `_chain_compute_gaps` can never resolve (see its docstring): it would silently re-fetch
        that same unsatisfiable pre-/post-market range on every single call, forever, since no
        data will ever exist there to close the gap.
        """
        key = (underlying, expiration, right)
        if key not in self._chains:
            if self.chain_fetch_fn is None:
                return pd.DataFrame()
            warm_start = session_open(ts.date())
            warm_end = session_close(expiration)
            self.warm_chain(underlying, expiration, warm_start, warm_end, right)

        df = self._chains[key]
        if df.empty:
            return df

        ts_level = df.index.get_level_values("timestamp")
        sub = df.loc[ts_level <= ts]
        if sub.empty:
            return pd.DataFrame()

        latest_ts = sub.index.get_level_values("timestamp").max()
        return sub.loc[latest_ts]