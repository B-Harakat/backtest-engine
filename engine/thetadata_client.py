"""
ThetaData client — rewritten for the `thetadata` Python library (gRPC-based, authenticates over
HTTPS; no ThetaTerminal process to run). See https://docs.thetadata.us/Python-Library/Getting-Started.html

Scoped to what an Options Standard subscription actually has access to: OHLC bars and
first-order greeks (delta, theta, vega, rho, epsilon, lambda, implied_vol) — NOT second-order
(gamma) or third-order greeks, which need a higher tier. That's exactly why `engine.greeks`
exists: it computes gamma/vanna/charm locally from the implied_vol this tier does give us.

Only two ThetaData calls per contract-date-range: `option_history_ohlc` and
`option_history_greeks_first_order`. The greeks endpoint already returns the NBBO bid/ask used
in its own calculation, so a separate quote call isn't needed — this is a real simplification
over the earlier REST/ThetaTerminal design, which needed three calls per contract.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.entities import Contract
from engine.greeks import compute_second_order_greeks, time_to_expiry_years

DEFAULT_INTERVAL = "1m"
ET = "America/New_York"

# Passed to BOTH ThetaData's `rate_value` param below AND `engine.greeks.compute_second_order_greeks`'s
# `r` argument (that module's default is also 0.0 — keep the two in sync if you change this).
# ThetaData defaults to SOFR (a few percent, not 0) if `rate_value` is left unset; letting them
# use their SOFR default while we assumed r=0 locally would silently make our gamma/vanna/charm
# inconsistent with their delta/theta/vega/rho. Forcing both sides to the same explicit value
# removes that ambiguity entirely, at the cost of a small (and for short-dated XSP, negligible)
# realism trade-off versus using the true SOFR rate.
ASSUMED_RATE_PCT = 0.0

# ThetaData already ignores dividends by default (matches engine.greeks's q=0 default) — passed
# explicitly here anyway so the assumption is visible in the request, not just an implicit default
# we're hoping stays in sync.
ASSUMED_ANNUAL_DIVIDEND = 0.0

# ThetaData's docs: "Multi-day requests are limited to 1 month of data" for every history
# endpoint used here. Chunk any wider request into <=1-month windows and concatenate.
_MAX_CHUNK_DAYS = 28

# Repo-root-relative path to creds.txt (email on line 1, password on line 2) -- this file lives
# at /home/claude/xsp_backtester/creds.txt in this project, i.e. two directories up from here
# (engine/thetadata_client.py -> engine/ -> repo root). Resolved explicitly rather than relying
# on ThetaClient()'s own default of "creds.txt in the current working directory", since that
# depends on where the script happens to be launched from (an IDE, a different cwd, etc.) rather
# than the repo's actual layout.
CREDS_FILE = Path(__file__).resolve().parent.parent / "creds.txt"

_client = None  # lazily-constructed module-level singleton


def _get_client():
    """
    Auth priority: explicit CREDS_FILE (email/password) if it exists at the repo root, otherwise
    fall back to ThetaClient()'s own default discovery (THETADATA_API_KEY env var / .env file /
    cwd-relative creds.txt) -- see the `thetadata` package's own auth docs for the full list.
    Constructed once per process, not once per call.
    """
    global _client
    if _client is None:
        from thetadata import ThetaClient

        if CREDS_FILE.exists():
            _client = ThetaClient(creds_file=str(CREDS_FILE), dataframe_type="pandas")
        else:
            _client = ThetaClient(dataframe_type="pandas")
    return _client


def _chunked_date_ranges(start: date, end: date, max_days: int = _MAX_CHUNK_DAYS):
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + timedelta(days=max_days - 1))
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(days=1)


def _right_str(contract: Contract) -> str:
    return "call" if contract.right == "CALL" else "put"


def _ensure_naive_et(df: pd.DataFrame) -> pd.DataFrame:
    """
    The rest of the engine standardizes on naive wall-clock America/New_York timestamps (see
    calendar.py's module docstring). The Python library's docs don't explicitly state whether
    the returned `timestamp` column is tz-aware or already naive-local — handle both: convert to
    ET and strip tz info if aware, pass through unchanged if already naive. VERIFY this against
    the first real response you get back; if timestamps come back looking shifted by a fixed
    number of hours against what you expect (e.g. bars starting at 14:30 instead of 09:30), the
    library is returning naive UTC rather than naive ET and this function needs a `tz_localize
    ("UTC").tz_convert(ET).tz_localize(None)` path added instead.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(ET).tz_localize(None)
    return df


def _ensure_naive_et_column(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Same idea as `_ensure_naive_et`, but for a `timestamp` column rather than the index —
    needed by the chain path, which sets a (timestamp, strike) MultiIndex after normalizing."""
    if df.empty or col not in df.columns:
        return df
    if pd.api.types.is_datetime64tz_dtype(df[col]):
        df = df.copy()
        df[col] = df[col].dt.tz_convert(ET).dt.tz_localize(None)
    return df


def _index_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Promote the `timestamp` column to the index and drop the symbol/expiration/strike/right
    columns — redundant once you're fetching one specific contract at a time, since the Contract
    object already carries that identity."""
    if df.empty:
        return df
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns=[c for c in ("symbol", "expiration", "strike", "right") if c in df.columns])
    return _ensure_naive_et(df)


def _fetch_or_empty(fn, **kwargs) -> pd.DataFrame:
    """
    Calls a ThetaData client method, returning an empty DataFrame instead of letting
    `NoDataFoundError` propagate. A contract can legitimately have zero trades (OHLC) or zero
    quotes (greeks) for an otherwise-correctly-scoped request -- e.g. a thin, far-out-of-the-money
    strike that's listed and has occasional bid/ask activity but never actually printed a trade
    on a given day. That's a real, valid data state, not an error, and shouldn't crash a backtest;
    it should just mean this particular bar/gap has nothing to contribute. This is a different
    case from the earlier "requested a date range the contract can never have data for at all"
    bug (e.g. asking for a contract's OHLC weeks after it expired) -- that one is a genuine
    caller mistake worth surfacing; this one is normal, expected market microstructure.
    """
    from thetadata.errors import NoDataFoundError

    try:
        return fn(**kwargs)
    except NoDataFoundError:
        return pd.DataFrame()


def fetch_option_ohlc(contract: Contract, start_date: date, end_date: date) -> pd.DataFrame:
    client = _get_client()
    frames = []
    for chunk_start, chunk_end in _chunked_date_ranges(start_date, end_date):
        frames.append(
            _fetch_or_empty(
                client.option_history_ohlc,
                symbol=contract.underlying,
                expiration=contract.expiration,
                strike=f"{contract.strike:.2f}",
                right=_right_str(contract),
                start_date=chunk_start,
                end_date=chunk_end,
                interval=DEFAULT_INTERVAL,
            )
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _index_by_timestamp(combined)


def fetch_option_greeks_first_order(contract: Contract, start_date: date, end_date: date) -> pd.DataFrame:
    client = _get_client()
    frames = []
    for chunk_start, chunk_end in _chunked_date_ranges(start_date, end_date):
        frames.append(
            _fetch_or_empty(
                client.option_history_greeks_first_order,
                symbol=contract.underlying,
                expiration=contract.expiration,
                strike=f"{contract.strike:.2f}",
                right=_right_str(contract),
                start_date=chunk_start,
                end_date=chunk_end,
                interval=DEFAULT_INTERVAL,
                annual_dividend=ASSUMED_ANNUAL_DIVIDEND,
                rate_value=ASSUMED_RATE_PCT,
                version="latest",
            )
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = _index_by_timestamp(combined)

    # Per ThetaData's Option-Greeks article: rho and vega must be divided by 100 to reach the
    # standard per-unit convention. Do this once, here, so nothing downstream has to remember it.
    for col in ("rho", "vega"):
        if col in df.columns:
            df[col] = df[col] / 100.0

    return df


def _merge_option_frames(ohlc: pd.DataFrame, greeks: pd.DataFrame) -> pd.DataFrame:
    # No column-name overlap between the two responses (ohlc: open/high/low/close/volume/count/
    # vwap; greeks: bid/ask/delta/theta/vega/rho/epsilon/lambda/implied_vol/iv_error/
    # underlying_timestamp/underlying_price) — a plain join is safe, no suffixing needed.
    return ohlc.join(greeks, how="outer").sort_index()


def _compute_row_second_order_greeks(
    ts: datetime, strike: float, underlying_price, iv, right: str, expiration: date
) -> tuple[float, float, float]:
    """Shared by both the per-contract path (fixed strike/right/expiration) and the chain path
    (strike varies per row, right/expiration fixed for the whole chain fetch)."""
    if pd.isna(iv) or pd.isna(underlying_price) or iv <= 0:
        return float("nan"), float("nan"), float("nan")

    t_years = time_to_expiry_years(ts, expiration)
    greeks_2 = compute_second_order_greeks(
        underlying_price=float(underlying_price),
        strike=float(strike),
        time_to_expiry_yrs=t_years,
        implied_vol=float(iv),
        right=right,
        r=ASSUMED_RATE_PCT / 100.0,  # kept in sync with the rate_value passed to ThetaData above
    )
    return greeks_2.gamma, greeks_2.vanna, greeks_2.charm


def _add_second_order_greeks(df: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    if df.empty or "implied_vol" not in df.columns:
        return df

    gammas, vannas, charms = [], [], []
    for ts, row in df.iterrows():
        gamma, vanna, charm = _compute_row_second_order_greeks(
            ts.to_pydatetime(), contract.strike, row.get("underlying_price"), row.get("implied_vol"),
            contract.right, contract.expiration,
        )
        gammas.append(gamma)
        vannas.append(vanna)
        charms.append(charm)

    df = df.copy()
    df["gamma"] = gammas
    df["vanna"] = vannas
    df["charm"] = charms
    return df


def _add_second_order_greeks_chain(df: pd.DataFrame, expiration: date, right: str) -> pd.DataFrame:
    """Same as `_add_second_order_greeks`, but for a (timestamp, strike)-MultiIndex chain
    DataFrame — each row supplies its own strike instead of one fixed contract.strike."""
    if df.empty or "implied_vol" not in df.columns:
        return df

    gammas, vannas, charms = [], [], []
    for (ts, strike), row in df.iterrows():
        ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        gamma, vanna, charm = _compute_row_second_order_greeks(
            ts_py, strike, row.get("underlying_price"), row.get("implied_vol"), right, expiration
        )
        gammas.append(gamma)
        vannas.append(vanna)
        charms.append(charm)

    df = df.copy()
    df["gamma"] = gammas
    df["vanna"] = vannas
    df["charm"] = charms
    return df


def fetch_chain_from_thetadata(
    underlying: str, expiration: date, gap_start: datetime, gap_end: datetime, right: str = "call"
) -> pd.DataFrame:
    """
    The ChainFetchFn passed to `engine.data_store.get_chain`. Fetches the FULL option chain
    (every listed strike, one side at a time) for `underlying`'s `expiration`, covering
    [gap_start, gap_end], via `option_history_greeks_first_order(strike="*")`.

    This is how the engine sources both "what's the underlying price right now" and "what
    strikes are actually listed" using only an Options Standard subscription — every row already
    carries `underlying_price`, and the set of `strike` values present at each tick IS the listed
    strike list, with no separate Index/Stock-tier endpoint (`index_history_price`) or
    listing-tier endpoint (`option_list_strikes`) required. `right` is `"call"` or `"put"` — one
    side per call, since a single (timestamp, strike) MultiIndex can't hold both a call and a put
    row at the same strike; call this once per side for a strategy that needs both (e.g. an iron
    condor), which is exactly what `Strategy.get_chain_snapshot(..., right=...)` does.

    Returns a DataFrame with a (timestamp, strike) MultiIndex.
    """
    client = _get_client()
    start_date, end_date = gap_start.date(), gap_end.date()
    frames = []
    for chunk_start, chunk_end in _chunked_date_ranges(start_date, end_date):
        frames.append(
            _fetch_or_empty(
                client.option_history_greeks_first_order,
                symbol=underlying,
                expiration=expiration,
                strike="*",
                right=right,
                start_date=chunk_start,
                end_date=chunk_end,
                interval=DEFAULT_INTERVAL,
                annual_dividend=ASSUMED_ANNUAL_DIVIDEND,
                rate_value=ASSUMED_RATE_PCT,
                version="latest",
            )
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return pd.DataFrame()

    for col in ("rho", "vega"):
        if col in combined.columns:
            combined[col] = combined[col] / 100.0

    combined = combined.drop(columns=[c for c in ("symbol", "expiration", "right") if c in combined.columns])
    combined = _ensure_naive_et_column(combined, "timestamp")
    combined = combined.set_index(["timestamp", "strike"]).sort_index()
    combined = _add_second_order_greeks_chain(combined, expiration, right=right.upper())

    ts_level = combined.index.get_level_values("timestamp")
    return combined.loc[(ts_level >= gap_start) & (ts_level <= gap_end)]


def fetch_from_thetadata(contract: Contract, gap_start: datetime, gap_end: datetime) -> pd.DataFrame:
    """
    The FetchFn passed to `engine.data_store.get_bars`. Fetches by whole calendar day (ThetaData's
    date params are date-granular) covering [gap_start, gap_end], then trims to the exact
    requested timestamp range before returning.

    Only fetches option contracts. A bare underlying Contract (no expiration/strike/right) has no
    fetch path here — sourcing underlying/index price history requires a separate ThetaData
    Index/Stock subscription tier beyond Options Standard. Use `fetch_chain_from_thetadata`
    instead (via Strategy.get_chain_snapshot), which sources the underlying price from the
    options chain data you already have access to.
    """
    if not contract.is_option:
        raise NotImplementedError(
            f"fetch_from_thetadata was called with a non-option contract ({contract}). Underlying/"
            "index price fetching isn't supported here — it requires a separate ThetaData Index/"
            "Stock subscription tier. Use Strategy.get_chain_snapshot() / fetch_chain_from_thetadata "
            "instead, which sources the underlying price from options chain data."
        )

    start_date, end_date = gap_start.date(), gap_end.date()
    ohlc = fetch_option_ohlc(contract, start_date, end_date)
    greeks = fetch_option_greeks_first_order(contract, start_date, end_date)
    df = _merge_option_frames(ohlc, greeks)
    df = _add_second_order_greeks(df, contract)

    if df.empty:
        return df

    return df.loc[gap_start:gap_end]