"""
Daily VIX data access for strategies.

ThetaData's Options Standard tier does not include index-level data such as VIX, so the engine
pulls VIX daily prints from a separate, free, pre-fetched CSV written by ``vix-download.py`` at
the repo root (source: Cboe's official VIX daily history). This module is a thin read-only
lookup layer over that file — no subscription, no real-time feed.

Data file (default ``<data_dir>/vix_data.csv``):

    date,vix_open,vix_close
    20230103,23.090000,22.900000

    - ``date``     : the trading day, ``YYYYMMDD`` (the format vix-download.py writes).
    - ``vix_open`` : the VIX print at the session open (~09:30/09:31 ET).
    - ``vix_close``: the VIX print at the ~16:00 ET close (the final official print).

``field`` is ``"open"`` (same-day open print) or ``"close"`` (prior-day/current-day close). Both
are used for volatility-regime gating: the prior day's close tells you the regime before the
session starts; the same-day open captures overnight moves.

The CSV is read once per process and cached, so repeated per-bar calls (e.g. a VIX filter checked
in ``on_bar``) don't re-parse the file. Missing file, missing day, or unknown field return ``None``.
"""

from __future__ import annotations

import functools
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from engine.data_store import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# Filename written by vix-download.py (repo root). Kept at the data root, not in a subdir, to
# match where vix-download.py already wrote it.
_VIX_CSV_FILENAME = "vix_data.csv"


@functools.lru_cache(maxsize=None)
def _load_vix_map(path: Path) -> dict[date, dict[str, Optional[float]]]:
    """
    Parse the VIX CSV into ``{date: {"open": float|None, "close": float|None}}``.

    Cached in-process so strategies can query per bar without re-reading the file. Accepts both
    ``YYYYMMDD`` (what vix-download.py writes) and ``YYYY-MM-DD`` dates, and upper/lower/mixed
    column names, for robustness if the file is ever hand-edited.
    """
    from datetime import datetime as _dt

    result: dict[date, dict[str, Optional[float]]] = {}

    def _parse_date(raw: str) -> Optional[date]:
        raw = raw.strip()
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return _dt.strptime(raw, fmt).date()
            except ValueError:
                continue
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("VIX data file %s not found — get_daily_vix will return None.", path)
        return result
    except OSError as exc:
        logger.warning("Could not read VIX data file %s: %s", path, exc)
        return result

    lines = text.splitlines()
    if not lines:
        return result

    # Resolve column indices from the header (case-insensitive).
    header = [h.strip().lower() for h in lines[0].split(",")]
    try:
        i_date = header.index("date")
    except ValueError:
        logger.warning("VIX data file %s has no 'date' column; ignoring.", path)
        return result
    i_open = header.index("vix_open") if "vix_open" in header else (header.index("open") if "open" in header else None)
    i_close = header.index("vix_close") if "vix_close" in header else (header.index("close") if "close" in header else None)

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        d = _parse_date(parts[i_date])
        if d is None:
            continue

        def _num(idx) -> Optional[float]:
            if idx is None or idx >= len(parts):
                return None
            try:
                return float(parts[idx])
            except ValueError:
                return None

        result[d] = {"open": _num(i_open), "close": _num(i_close)}

    return result


def get_daily_vix(day: date, field: str, data_dir: Path = DEFAULT_DATA_DIR) -> Optional[float]:
    """
    The VIX print for ``day`` on the given ``field``.

    ``field`` is ``"open"`` (the session's opening print) or ``"close"`` (the session's closing
    print). Returns ``None`` if the VIX CSV is missing, the day is not present, or the field is
    unknown / has no value for that day.
    """
    key = field.strip().lower()
    if key not in ("open", "close"):
        raise ValueError(f"field must be 'open' or 'close', got {field!r}")

    path = Path(data_dir) / _VIX_CSV_FILENAME
    vix_map = _load_vix_map(path)
    row = vix_map.get(day)
    if row is None:
        return None
    return row.get(key)
