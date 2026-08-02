#!/usr/bin/env python3
"""
One-time downloader for VIX daily open/close prints, written to data/vix_data.csv.

This is a download-only helper: it fetches Cboe's official daily VIX history CSV once and
re-writes it as a minimal two-column series for use by the backtest engine.

Rationale / mapping (see the design choices agreed for this task):
  - The engine has NO intraday VIX subscription (ThetaData Options Standard has no underlying/
    index feed -- engine.thetadata_client.fetch_from_thetadata raises for non-option contracts),
    and VIX is a distinct Cboe product with its own calendar/settlement. So this file does NOT go
    through the engine's data store at all; it pulls Cboe's daily history directly.
  - Cboe's daily CSV is OHLC-per-day (no intraday stamps). We map:
        vix_open  <- daily OPEN   (the first published ~09:30/09:31 print of the session)
        vix_close <- daily CLOSE  (the final ~15:59/16:00 settlement close print)
    The daily close IS the last trade at/before the 16:00 close, so no time-of-day slicing is
    needed -- Cboe has already collapsed the day into its official open/close prints.
  - Row dates come from the engine's XNYS trading calendar (engine.calendar.trading_days), so a
    day only appears if it is an actual XNYS session; VIX rows on non-XNYS days are filtered out.
  - CSV layout: date (YYYYMMDD), vix_open, vix_close.

Output: data/vix_data.csv. It is overwritten on each run (download-once semantics).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Ensure the repo root is importable so we can reuse engine.calendar.trading_days() for the
# trading-calendar row set (consistent with the rest of the repo).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.calendar import trading_days  # noqa: E402

# Cboe's official VIX daily history (fully free, no auth, updated daily). Header: DATE,OPEN,HIGH,LOW,CLOSE.
VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

# Download range (inclusive) for the rows of vix_data.csv.
START_DATE = date(2023, 1, 1)
END_DATE = date(2026, 7, 31)

# Repo-root-relative output path, matching the convention used by engine.data_store.
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "vix_data.csv"


def _fetch_csv(url: str) -> bytes:
    """Download ``url`` and return the raw bytes. Uses urllib (stdlib) so no extra deps."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "backtest-engine-vix-download"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _parse_rows(text: str) -> dict[date, tuple[str, str]]:
    """
    Parse Cboe's CSV into {date: (open, close)}.

    Cboe uses MM/DD/YYYY (sometimes without leading zeros, e.g. 8/5/1990), so the date column is
    parsed with the loose %-format. We keep OPEN and CLOSE as their original numeric strings to
    preserve Cboe's precision verbatim. Rows with an unparsable date or a missing close are
    skipped.
    """
    lines = text.splitlines()
    if not lines:
        return {}
    header = lines[0].strip().upper()
    if "DATE" not in header or "CLOSE" not in header:
        raise ValueError(f"Unexpected Cboe CSV header: {lines[0]!r}")

    out: dict[date, tuple[str, str]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        date_str = parts[0]
        open_str = parts[1]
        close_str = parts[-1]  # CLOSE is the last column

        try:
            mm, dd, yyyy = date_str.split("/")
            d = date(int(yyyy), int(mm), int(dd))
        except (ValueError, IndexError):
            continue

        if not open_str or not close_str:
            continue
        out[d] = (open_str, close_str)

    return out


def main(argv: list[str] | None = None) -> Path:
    # Build the ordered row dates from the engine's XNYS calendar.
    trading_ts = trading_days(START_DATE, END_DATE)
    row_dates = [ts.date() for ts in trading_ts]
    if not row_dates:
        raise RuntimeError(f"No XNYS trading days found in [{START_DATE}, {END_DATE}].")

    print(f"Downloading VIX history from {VIX_HISTORY_URL} ...")
    raw = _fetch_csv(VIX_HISTORY_URL)
    text = raw.decode("utf-8")

    parsed = _parse_rows(text)
    if not parsed:
        raise RuntimeError("Failed to parse any rows from the Cboe VIX history CSV.")

    print(f"Parsed {len(parsed)} total trading dates from Cboe.")

    # Align the XNYS row dates to the VIX prints; drop any XNYS day with no matching VIX row.
    lines = ["date,vix_open,vix_close"]
    matched = 0
    for d in row_dates:
        row = parsed.get(d)
        if row is None:
            print(f"WARNING: no VIX print for XNYS trading day {d} -- skipped.")
            continue
        open_str, close_str = row
        lines.append(f"{d:%Y%m%d},{open_str},{close_str}")
        matched += 1

    if matched == 0:
        raise RuntimeError(
            f"No VIX rows matched any XNYS day in [{START_DATE}, {END_DATE}] -- "
            "check the date range and Cboe data availability."
        )

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {matched} rows ({len(row_dates) - matched} XNYS days had no VIX print) to {DEFAULT_OUTPUT}")
    return DEFAULT_OUTPUT


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface a clear oneline error from the downloader
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
