"""
Monte Carlo harness for the synthetic-market backtest.

Drives ``engine.runner.run_backtest`` on paths synthesized by ``simulation.bates`` and priced by
``simulation.market_maker``, over a grid of strategy parameter sets, and aggregates the result
distribution (per parameter set) across paths.

Key mechanics:

  * **Determinism**: a master ``seed`` derives a distinct per-path seed so any given
    ``(day, param set, path index)`` reproduces the exact same session and result when re-run
    with the same ``seed``/``n_paths``/``params``. Pass ``seed=None`` for fresh runs.
  * **Isolation**: each path writes its synthetic Parquet cache to its OWN temporary ``data_dir``.
    Parameter sets sharing a path reuse that path's ``data_dir`` (the market is identical within
    a path), but paths never share one, preventing cross-path contamination and keeping synthetic
    data out of the real ``data/``.
  * **Strategy grid**: ``param_grid`` is a list of dicts, each full set of constructor kwargs for
    ``strategy_cls`` (e.g. ``{"spread_width": 2, "short_leg_stop_multiple": 1.5, "entry_time": ...}``).
    The strategy is instantiated fresh per run because strategies hold mutable per-session state.

The target context is intraday 0DTE XSP (one session per path), so each day is both ``start_date``
and ``end_date`` and ``settlement_style="cash"`` with the default 4:00pm PM settlement.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Type

import numpy as np
import pandas as pd

from engine.costs import CostFn, ibkr_standard_option_cost
from engine.entities import DEFAULT_OPTION_MULTIPLIER
from engine.runner import BacktestResult, run_backtest
from engine.strategy import Strategy
from reports.report import build_summary
from simulation.bates import BatesParams, simulate_session
from simulation.market_maker import SpreadConfig, StrikeGrid, SyntheticMarketMaker


@dataclass(frozen=True)
class DayMarket:
    """Per-session market parameters for one 0DTE day: date, opening spot, and Bates params."""

    day: date
    s0: float
    bates: BatesParams


@dataclass
class MCConfig:
    """Top-level configuration for one Monte Carlo experiment.

    ``per_day_seeds`` enumerates the (day, s0, bates) sessions to simulate -- one per trading
    day in the window. ``n_paths`` is the number of paths PER DAY; total runs =
    len(per_day_seeds) x n_paths x len(param_grid).
    """

    strategy_cls: Type[Strategy]
    param_grid: list[dict]               # expanded list of constructor kwargs; one dict per set.
    per_day_seeds: list[DayMarket]       # one session seed per trading day.
    n_paths: int = 100
    seed: Optional[int] = 0              # master seed; None => fresh randomness.
    starting_cash: float = 10_000.0
    multiplier: int = DEFAULT_OPTION_MULTIPLIER
    settlement_time: time = time(16, 0)
    cost_fn: Optional[CostFn] = ibkr_standard_option_cost
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    strike_grid: StrikeGrid = field(default_factory=StrikeGrid)
    # Explicit underlying symbol; None => inferred from the strategy's grid default (partial
    # override means the grid usually omits `underlying`, so this falls back to "XSP").
    _underlying: Optional[str] = None

    @property
    def days(self) -> list[date]:
        return [dm.day for dm in self.per_day_seeds]

    @property
    def underlying(self) -> str:
        return self._underlying or _underlying_of(self)



def expand_grid(spec: dict) -> list[dict]:
    """
    Expand a sparse grid spec into the full list of parameter-set dicts (the Cartesian product).

    ``spec`` is a ``{param_name: [candidate_value, ...]}`` mapping; run_mc then builds one dict
    per combination. Because only the listed params are overridden, any strategy constructor
    arg NOT named in the spec keeps its own default (partial override) -- see the constructor
    interaction rule. Non-hashable candidate values (e.g. ``datetime.time``) are fine.
    """
    if not isinstance(spec, dict):
        raise TypeError(f"grid spec must be a dict of {{param: [values]}}, got {type(spec).__name__}")
    if not spec:
        # Empty spec => a single parameter set with ALL constructor defaults.
        return [{}]
    keys = list(spec.keys())
    combos = itertools.product(*(spec[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in combos]


@dataclass
class PathResult:
    day: date
    path_index: int
    param_set_index: int
    param_set: dict
    result: BacktestResult
    summary: dict = field(default_factory=dict)  # build_summary() output, cached so the heavy
    #                                            result payload can be freed after summarization.

    def free_heavy(self) -> None:
        """Release the heavy per-bar data (minute equity curve, fills, realized-P&L events) held
        by ``self.result`` so a long/high-N run doesn't retain them all in memory. The cached
        ``self.summary`` dict (which contains ending_equity, max_drawdown, win rate, fills count,
        realized P&L, etc.) remains intact, so downstream aggregation/reporting stays valid."""
        try:
            self.result.equity_curve = None
            self.result.fills = None
            self.result.realized_pnl_events = None
        except Exception:
            pass


def summarize_result(result: BacktestResult, starting_cash: float) -> dict:
    """Compute the full report.py statistics for one result (the per-path summary). Cached on the
    PathResult so reporting doesn't re-run build_summary or touch the heavy payload again."""
    return build_summary(result, starting_cash)


def _path_seed(master: Optional[int], day: date, path_index: int) -> Optional[int]:
    if master is None:
        return None
    # Day-aware, stable, distinct seed per (day, path): the master is salted by the day's
    # ordinal so (i) each (day, path, param-set) is reproducible bit-for-bit, and (ii) running a
    # single day via --day gives the EXACT same paths as a multi-day run containing that day
    # with the same master seed (a pure function of master + day, independent of list position).
    salt = master + day.toordinal() * 1_000_003
    return int(np.random.default_rng(salt + path_index).integers(0, 2**63 - 1))


def run_mc(config: MCConfig, on_result=None) -> list[PathResult]:
    """
    Run the Monte Carlo and return one ``PathResult`` per (day, path, param-set). Each trading
    day in ``config.per_day_seeds`` is simulated independently with its own opening spot/Bates
    params, N paths, and the full parameter grid. Days never share a cache dir.

    Memory handling (see RAM notes): passing ``on_result`` enables STREAMING mode. ``on_result``
    is called with each completed ``PathResult`` and the heavy per-bar payload is freed
    immediately after; ``run_mc`` then returns an empty list. Without ``on_result`` (backward
    compatible), all results are accumulated and returned — fine for small runs, but for long /
    high-``n_paths`` runs prefer ``on_result`` so results are consumed/freed as they're produced
    instead of retaining every minute equity curve in memory.
    """
    results: list[PathResult] = []

    for dm in config.per_day_seeds:
        for path_i in range(config.n_paths):
            path_seed = _path_seed(config.seed, dm.day, path_i)
            path = simulate_session(dm.day, dm.s0, dm.bates, seed=path_seed)
            if path.empty:
                continue

            mm = SyntheticMarketMaker(
                path=path,
                underlying=config.underlying,
                spread=config.spread,
                strike_grid=config.strike_grid,
                seed=path_seed,
            )

            # One temporary cache dir per (day, path), shared across that path's param sets.
            with TemporaryDirectory(prefix="mc_cache_") as tmp:
                data_dir = Path(tmp)
                for p_idx, params in enumerate(config.param_grid):
                    strategy = config.strategy_cls(**params)
                    result = run_backtest(
                        strategy,
                        dm.day,
                        dm.day,
                        starting_cash=config.starting_cash,
                        multiplier=config.multiplier,
                        settlement_style="cash",
                        settlement_time=config.settlement_time,
                        fetch_fn=mm.fetch_fn,
                        chain_fetch_fn=mm.chain_fetch_fn,
                        data_dir=data_dir,
                        cost_fn=config.cost_fn,
                        warm_open_chain=False,  # synthetic runs supply their own s0/V0; never warm real chains
                    )
                    pr = PathResult(
                        day=dm.day,
                        path_index=path_i,
                        param_set_index=p_idx,
                        param_set=params,
                        result=result,
                        summary=summarize_result(result, config.starting_cash),
                    )
                    # Free the heavy per-bar payload as soon as it's been summarized, so a long
                    # run doesn't hold every minute equity curve in memory (reporting and
                    # aggregation read the cached scalar `summary` instead).
                    pr.free_heavy()

                    if on_result is not None:
                        on_result(pr)
                    else:
                        results.append(pr)

    return results


def _underlying_of(config: MCConfig) -> str:
    """Best-effort underlying symbol from the first param set (strategies carry ``underlying``)."""
    for p in config.param_grid:
        if isinstance(p.get("underlying"), str):
            return p["underlying"]
    return "XSP"


# --- aggregation -----------------------------------------------------------------------------


def aggregate_results(results: list[PathResult], starting_cash: float) -> pd.DataFrame:
    """
    Collapse per-(day, path, param-set) results into one summary row per parameter set.

    Combined columns describe the distribution ACROSS ALL days x paths (ending equity, total
    return, max drawdown, win rate, fills, realized P&L at mean/p05/p50/p95). Per-day columns
    (suffixed ``_day_<iso>_...``) break out the same distribution for each individual day, so a
    multi-day run shows both the pooled numbers and per-day behavior. Reads the cached
    ``PathResult.summary`` scalars only — the heavy per-bar payload is not needed here (it has
    already been freed by run_mc), so this is memory-light regardless of how many results there
    are.
    """
    if not results:
        raise ValueError("No results to aggregate -- did run_mc return an empty list?")

    by_param: dict[int, list[PathResult]] = {}
    labels: dict[int, dict] = {}
    for r in results:
        by_param.setdefault(r.param_set_index, []).append(r)
        labels[r.param_set_index] = r.param_set

    days = sorted({r.day for r in results})

    def _summary_val(r: PathResult, key: str):
        return r.summary.get(key)

    def _distcols(rs: list) -> dict:
        end_eqs = [v for v in (_summary_val(r, "ending_equity") for r in rs) if v is not None]
        if not end_eqs:
            return {}
        ret = [e / starting_cash * 100.0 - 100.0 for e in end_eqs]
        realized = [v for v in (_summary_val(r, "total_realized_pnl") for r in rs) if v is not None]
        max_dd = [v for v in (_summary_val(r, "max_drawdown_pct") for r in rs) if v is not None]
        win_rates = [v for v in (_summary_val(r, "win_rate_pct") for r in rs) if v is not None]
        n_fills = [v for v in (_summary_val(r, "num_fills") for r in rs) if v is not None]
        return {
            "ending_equity_mean": float(np.mean(end_eqs)),
            "ending_equity_std": float(np.std(end_eqs)),
            "ending_equity_p05": float(np.percentile(end_eqs, 5)),
            "ending_equity_p50": float(np.percentile(end_eqs, 50)),
            "ending_equity_p95": float(np.percentile(end_eqs, 95)),
            "total_return_mean_pct": float(np.mean(ret)),
            "total_return_p05_pct": float(np.percentile(ret, 5)),
            "total_return_p95_pct": float(np.percentile(ret, 95)),
            "realized_pnl_mean": float(np.mean(realized)) if realized else np.nan,
            "realized_pnl_std": float(np.std(realized)) if realized else np.nan,
            "max_drawdown_mean_pct": float(np.mean(max_dd)) if max_dd else np.nan,
            "max_drawdown_p05_pct": float(np.percentile(max_dd, 5)) if max_dd else np.nan,  # worst 5%
            "win_rate_mean_pct": float(np.mean(win_rates)) if win_rates else np.nan,
            "fills_mean": float(np.mean(n_fills)) if n_fills else np.nan,
        }

    rows = []
    for p_idx in sorted(by_param):
        prs = by_param[p_idx]
        row = {
            "param_set_index": p_idx,
            **{f"param_{k}": str(v) for k, v in labels[p_idx].items()},
            "n_paths": len(prs),
            **{f"n_paths_day_{d.isoformat()}": sum(1 for r in prs if r.day == d) for d in days},
            **_distcols(prs),
        }
        # Per-day breakdown columns, prefixed so they sit alongside the combined ones.
        for d in days:
            day_prs = [r for r in prs if r.day == d]
            for k, v in _distcols(day_prs).items():
                row[f"{k}_day_{d.isoformat()}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
