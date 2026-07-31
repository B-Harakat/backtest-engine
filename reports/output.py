"""
Run-output writers: persistently tree-based run/monte-carlo reports under ``output/``.

Layout (strategy class name UPPERCASED):

    output/<STRATEGY>/HISTORY/             -- one backtest run; files overwrite per run
        trade_log.csv                       fills (incl. expiration settlements)
        cash_equity.csv                     daily granularity: one row per trading day
        summary.json                        build_summary() statistics

    output/<STRATEGY>/MONTE-CARLO/         -- one synthetic Monte-Carlo experiment
        <date>.csv                          per day; rows = stats, columns = one per param set
        _summary.csv                        rollup: each stat averaged across days per param set

The HISTORY files reuse ``reports.report`` primitives (``build_summary``, ``fills_to_dataframe``);
the MONTE-CARLO stats are the same ``build_summary`` statistics, but gathered per path and pooled
into ``{stat}_mean / {stat}_p05 / {stat}_p95`` rows so N Monte-Carlo paths collapse into one
column.
"""

from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from reports.report import build_summary, fills_to_dataframe

if TYPE_CHECKING:
    from engine.runner import BacktestResult
    from simulation.monte_carlo import PathResult


def _strategy_dir(output_root: Path, strategy_cls: type) -> Path:
    return output_root / strategy_cls.__name__.upper()


def daily_equity_frame(result: "BacktestResult") -> pd.DataFrame:
    """Resample a run's minute equity curve to one row per trading day: `date, cash, equity`."""
    ec = result.equity_curve
    if ec.empty:
        return pd.DataFrame(columns=["date", "cash", "equity"])
    daily = ec.resample("1D").last().dropna(how="all")
    out = pd.DataFrame({"date": daily.index, "cash": daily["cash"], "equity": daily["equity"]})
    return out.reset_index(drop=True)


# --- HISTORY -------------------------------------------------------------------------------


def write_history(
    strategy_cls: type,
    result: "BacktestResult",
    starting_cash: float,
    output_root: Path,
) -> dict:
    """Write trade_log.csv, cash_equity.csv and summary.json under
    ``output/<STRATEGY>/HISTORY/``. Returns `build_summary()`'s summary dict."""
    out = _strategy_dir(output_root, strategy_cls) / "HISTORY"
    out.mkdir(parents=True, exist_ok=True)

    fills_to_dataframe(result.fills).to_csv(out / "trade_log.csv", index=False)
    daily_equity_frame(result).to_csv(out / "cash_equity.csv", index=False)

    summary = build_summary(result, starting_cash)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


# --- MONTE-CARLO ----------------------------------------------------------------------------


def _pool(stat_key, values) -> dict:
    """Distribution row for one stat across pooled (path/daily) samples."""
    vals = [v for v in values if v is not None]
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {stat_key + "_mean": None, stat_key + "_p05": None, stat_key + "_p95": None}
    return {
        stat_key + "_mean": float(np.mean(arr)),
        stat_key + "_p05": float(np.percentile(arr, 5)),
        stat_key + "_p95": float(np.percentile(arr, 95)),
    }


def daily_mc_frame(paths_for_day: list["PathResult"]) -> pd.DataFrame:
    """
    Build a per-day MONTE-CARLO frame: rows = ``{stat}_mean/{stat}_p05/{stat}_p95`` for each
    report.py stat, columns = one per param set (labelled by its param values). Stats for a
    (day, param set) are pooled across the N paths by reading each path's cached ``summary``
    (computed once by ``run_mc``) -- the heavy per-bar result payload is never touched here, so
    this stays memory-light even over many paths.
    """
    # Group paths by param_set_index, preserving a stable ordering.
    by_param: dict[int, list[PathResult]] = {}
    order: list[int] = []
    for r in paths_for_day:
        if r.param_set_index not in by_param:
            order.append(r.param_set_index)
        by_param.setdefault(r.param_set_index, []).append(r)

    # Stat key order from the first summary available (sorted for determinism).
    stat_keys: list[str] | None = None
    for prs in by_param.values():
        if prs and prs[0].summary:
            stat_keys = sorted(prs[0].summary.keys())
            break
    if stat_keys is None:
        return pd.DataFrame()

    # Row names: {stat}_mean / {stat}_p05 / {stat}_p95 in stat_keys order.
    row_names = [f"{k}_{suf}" for k in stat_keys for suf in ("mean", "p05", "p95")]

    # Columns: labelled param sets; cells = pooled stat value for that (day, param set).
    data: dict[str, list] = {}
    for p_idx in order:
        prs = by_param[p_idx]
        values = {k: [] for k in stat_keys}
        for pr in prs:
            s = pr.summary
            for k in stat_keys:
                values[k].append(s.get(k))
            # Free the per-path heavy result reference once its summary has been consumed.
            pr.result = None
        colvals = []
        for k in stat_keys:
            pooled = _pool(k, values[k])
            colvals.append(pooled[f"{k}_mean"])
            colvals.append(pooled[f"{k}_p05"])
            colvals.append(pooled[f"{k}_p95"])
        data[_param_label(prs[0].param_set)] = colvals

    df = pd.DataFrame(data, index=row_names)
    df.index.name = "stat"
    return df


def _param_label(params: dict) -> str:
    """Short label for a param set, e.g. 'w=2|stop=1.5|t=10:00'."""
    parts = []
    for k, v in params.items():
        if isinstance(v, pd.Timestamp):
            v = v.strftime("%H:%M")
        elif isinstance(v, time):
            v = v.strftime("%H:%M")
        parts.append(f"{k}={v}")
    return "|".join(parts) if parts else "default"


def write_monte_carlo(
    strategy_cls: type,
    results: list["PathResult"],
    output_root: Path,
) -> None:
    """
    Write output/<STRATEGY>/MONTE-CARLO/<date>.csv per day (columns = param sets, rows = pooled
    stats) and _summary.csv (each stat averaged across days per param set).
    """
    out = _strategy_dir(output_root, strategy_cls) / "MONTE-CARLO"
    out.mkdir(parents=True, exist_ok=True)

    # Group results by day for per-day files.
    by_day: dict[date, list[PathResult]] = {}
    for r in results:
        by_day.setdefault(r.day, []).append(r)

    day_frames: dict[date, pd.DataFrame] = {}
    for day in sorted(by_day):
        frame = daily_mc_frame(by_day[day])
        if not frame.empty:
            day_frames[day] = frame
            frame.to_csv(out / f"{day.isoformat()}.csv")  # rows = stat names, cols = param sets

    # _summary.csv: rollup each stat (mean/p05/p95) across the days' frames, per param set.
    rollup = _build_rollup(day_frames)
    rollup.to_csv(out / "_summary.csv")


def _build_rollup(day_frames: dict[date, pd.DataFrame]) -> pd.DataFrame:
    """
    Summarize the per-day MONTE-CARLO frames: one frame whose rows = stats, columns = param
    sets, where each cell = the average of that stat's value across the days.
    """
    if not day_frames:
        return pd.DataFrame()
    days = sorted(day_frames)

    # First day defines the stat-row order and the parameter-label columns.
    first = day_frames[days[0]]
    stat_names = list(first.index)
    param_labels = list(first.columns)

    rollup_cols: dict[str, list] = {}
    for label in param_labels:
        # Gather this stat across all days for this param set.
        row_vals = []
        for stat in stat_names:
            vals = [
                float(day_frames[d].at[stat, label])
                for d in days if label in day_frames[d].columns and stat in day_frames[d].index
            ]
            vals = [v for v in vals if not pd.isna(v)]
            row_vals.append(float(np.mean(vals)) if vals else None)
        rollup_cols[label] = row_vals

    df = pd.DataFrame(rollup_cols, index=stat_names)
    df.index.name = "stat"
    return df
