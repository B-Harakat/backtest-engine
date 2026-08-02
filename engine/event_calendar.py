"""
Event-day avoidance calendar.

Scheduled macro releases (FOMC decisions, CPI, NFP, PPI) create intraday volatility spikes that
are structurally hostile to premium-selling strategies — being short gamma through a data release
is a different risk/reward than a normal session. Research in this corpus ranks event-day
avoidance as the second strongest strategy-level filter.

The events are a static, manually-maintained CSV of known dates (updated quarterly — the Fed,
BLS, and BEA publish these months in advance), NOT a dynamic API or real-time feed.

Data file (default ``<data_dir>/events/event_days.csv``):

    date,event
    2024-01-31,FOMC
    2024-02-13,CPI
    2024-03-08,NFP
    2024-03-12,CPI
    2024-03-20,FOMC

    - ``date``  : the release date, ``YYYY-MM-DD``.
    - ``event`` : a short label ("FOMC", "CPI", "NFP", "PPI", ...) — informational.

``is_event_day(day)`` returns True only for days explicitly listed. A missing file, an empty
file, or a day that is not listed all return False, so the feature is safe to use before the CSV
is populated (it simply never filters anything out).
"""

from __future__ import annotations

import functools
import logging
from datetime import date
from pathlib import Path
from typing import Optional, Set

from engine.data_store import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# Matches engine.data_store's layout: event dates live under a subfolder of the data root.
_EVENTS_SUBDIR = "events"
_EVENTS_FILENAME = "event_days.csv"


@functools.lru_cache(maxsize=None)
def _load_event_days(path: Path) -> Set[date]:
    """
    Parse the event CSV into a set of dates. Cached in-process so per-bar calls in ``on_bar``
    don't re-read the file. Only the ``date`` column matters; the ``event`` label is ignored for
    the lookup (kept in the file for human readability / filtering by label later).
    """
    from datetime import datetime as _dt

    events: Set[date] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("Event calendar %s not found — is_event_day will always return False.", path)
        return events
    except OSError as exc:
        logger.warning("Could not read event calendar %s: %s", path, exc)
        return events

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("DATE"):
            continue
        date_str = stripped.split(",")[0].strip()
        try:
            events.add(_dt.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            logger.debug("Skipping unparsable event date %r in %s.", date_str, path)
            continue
    return events


def is_event_day(day: date, data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """
    True if ``day`` is a scheduled macro event (FOMC/CPI/NFP/PPI) per the manually-maintained
    ``event_days.csv``. False if the file is absent, empty, or the day is not listed.
    """
    path = Path(data_dir) / _EVENTS_SUBDIR / _EVENTS_FILENAME
    return day in _load_event_days(path)
