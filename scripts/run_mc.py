#!/usr/bin/env python3
"""
CLI for the synthetic-market Monte Carlo harness (`simulation.monte_carlo`).

Generic and model/strategy agnostic (0DTE is the only hard constraint): point it at any
strategy via a `--strategy module.path:ClassName` spec (exactly like `scripts/run_backtest.py`),
and the strategy's own module-level ``GRID`` dict defines the parameter space. run_mc imports
that module, expands ``GRID`` into the Cartesian product of parameter sets, and runs each set
over N simulated 0DTE XSP sessions (Bates spot/vol paths priced by simulation.market_maker).

Market-model and runtime knobs come from an optional TOML config (`--config`, see
`mc_config.toml`); CLI flags override the file; the file overrides built-in defaults.

Usage:
    python scripts/run_mc.py --strategy strategies.vertical_spread_strategy:VerticalSpreadStrategy \
        --day 2026-01-15 --s0 694 --config mc_config.toml --n-paths 20 --seed 3
    # no --config: built-in defaults; --seed -1 for fresh randomness.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from datetime import date, time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.calendar import trading_days
from engine.costs import cost_model_from_name
from engine.data_store import DEFAULT_DATA_DIR
from reports.output import write_monte_carlo
from simulation import monte_carlo
from simulation.bates import BatesParams
from simulation.config import load_config
from simulation.market_maker import SpreadConfig, StrikeGrid
from simulation.market_seed import opening_market_from_cache


def load_strategy(spec: str) -> tuple[type, list[dict]]:
    """
    Given ``module.path:ClassName``, import the module, find the class, and read its module-level
    ``GRID`` dict. Returns ``(strategy_cls, expanded_param_grid)``. A missing ``GRID`` yields a
    single default parameter set (all constructor defaults) so any strategy can be run as-is.
    """
    module_path, sep, class_name = spec.partition(":")
    if not sep:
        raise ValueError(f"--strategy must be 'module.path:ClassName', got {spec!r}")
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"{module_path!r} has no attribute {class_name!r}") from exc

    grid_spec = getattr(module, "GRID", {}) or {}
    if not isinstance(grid_spec, dict):
        raise TypeError(f"{module_path}.GRID must be a dict {{param: [values]}}, got {type(grid_spec).__name__}")
    return cls, monte_carlo.expand_grid(grid_spec)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the synthetic-market Monte Carlo harness.")
    p.add_argument(
        "--strategy", required=True,
        help="Strategy spec 'module.path:ClassName' (like run_backtest). Its module-level GRID "
             "defines the parameter grid.",
    )
    p.add_argument(
        "--day", default=None,
        help="Single 0DTE session date YYYY-MM-DD (shorthand for --start == --end == --day). "
             "Mutually exclusive with --start/--end.",
    )
    p.add_argument(
        "--start", default=None,
        help="Session window start YYYY-MM-DD (inclusive). Use with --end for a multi-day run.",
    )
    p.add_argument(
        "--end", default=None,
        help="Session window end YYYY-MM-DD (inclusive). Use with --start for a multi-day run.",
    )
    p.add_argument(
        "--ticker", default=None,
        help="Underlying symbol. Defaults to the strategy constructor's `underlying` default "
             "(e.g. 'XSP'). Used to locate the cached chain for opening s0/V_0.",
    )
    p.add_argument(
        "--s0", type=float, default=None,
        help="Opening spot per path. Default: derived from the cached real chain for each --day "
             "-- only used when a day's chain is missing. Overrides the config/cache.",
    )
    p.add_argument(
        "--v0", type=float, default=None,
        help="Opening implied vol (decimal, e.g. 0.15) for V_0. Default: derived from the cached "
             "real chain for each --day (ATM IV) -- only used when a day's chain is missing. "
             "Overrides the config/cache.",
    )
    p.add_argument(
        "--config", default=None,
        help="Optional TOML config for Bates/spread/strike/runtime knobs (see mc_config.toml).",
    )
    p.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR),
        help="Local Parquet cache directory to read opening-market seeds from (default: the real "
             "data dir where run_backtest writes chains).",
    )
    # Runtime overrides (CLI wins over config file).
    p.add_argument("--n-paths", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="Master seed; use -1 for fresh randomness.")
    p.add_argument("--cash", type=float, default=None, dest="starting_cash")
    p.add_argument("--multiplier", type=int, default=None)
    p.add_argument("--settlement-time", default=None, help="HH:MM 24h ET (default 16:00).")
    p.add_argument("--cost-model", default=None, choices=["ibkr", "none"])
    p.add_argument("--output-dir", default="output",
                   help="Root output dir (default: output/) -- MONTE-CARLO files are written to "
                        "output/<STRATEGY>/MONTE-CARLO/.")
    return p.parse_args(argv)


def _merge(overrides: dict, base: dict) -> dict:
    merged = dict(base)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def _default_underlying(strategy_cls: type) -> str:
    """Best-effort underlying symbol from the strategy's ``__init__`` default, else 'XSP'."""
    try:
        sig = inspect.signature(strategy_cls.__init__)
        default = sig.parameters.get("underlying")
        if default is not None and default.default is not inspect.Parameter.empty:
            return default.default
    except (TypeError, ValueError):
        pass
    return "XSP"


def _resolve_days(args) -> list[date]:
    """Resolve the list of trading days from the --day xor --start/--end CLI form."""
    if args.day is not None:
        if args.start is not None or args.end is not None:
            raise ValueError("Use either --day OR --start/--end, not both.")
        return [date.fromisoformat(args.day)]
    if args.start is None or args.end is None:
        raise ValueError("Must provide --day, or both --start and --end.")
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise ValueError("--end must be >= --start.")
    return [ts.date() for ts in trading_days(start, end)]


def main(argv: Optional[list[str]] = None) -> dict:
    args = parse_args(argv)
    days = _resolve_days(args)

    # Layer config file -> CLI.
    loaded = load_config(Path(args.config)) if args.config else {
        "bates": {}, "spread": {}, "strike_grid": {}, "runtime": {},
    }
    runtime = _merge(
        {
            "n_paths": args.n_paths,
            "starting_cash": args.starting_cash,
            "multiplier": args.multiplier,
            "settlement_time": args.settlement_time,
            "cost_model": args.cost_model,
            "seed": args.seed,
            "ticker": args.ticker,
            "s0": args.s0,
        },
        loaded["runtime"],
    )

    strategy_cls, param_grid = load_strategy(args.strategy)
    underlying = runtime.get("ticker") or _default_underlying(strategy_cls)
    data_dir = Path(args.data_dir)

    # Resolve opening market seeds PER DAY. Precedence (per the multi-day design): the cached
    # real chain for that day wins when present; --s0/--v0/[runtime] s0/[bates] V_0 are used only
    # as a fallback for days whose chain isn't cached. V_0 is stored/computed as VARIANCE
    # (IV^2), matching BatesParams.V_0; --v0 is an IV decimal.
    per_day = []
    explicit_s0 = args.s0 if args.s0 is not None else runtime.get("s0")
    config_v0_var = loaded["bates"].get("V_0")
    for d in days:
        om = None
        try:
            om = opening_market_from_cache(underlying, d, data_dir=data_dir)
        except (FileNotFoundError, ValueError):
            om = None

        if om is not None:
            s0 = om.spot
            v0_var = om.v0
        else:
            if explicit_s0 is None:
                raise RuntimeError(
                    f"No cached chain for {underlying} {d} (data_dir={data_dir}) and no --s0/"
                    f"[runtime] s0 fallback. Run a real backtest for this date first "
                    f"(run_backtest --start {d} --end {d}), or pass --s0."
                )
            if args.v0 is not None:
                v0_var = args.v0**2
            elif config_v0_var is not None:
                v0_var = config_v0_var
            else:
                raise RuntimeError(
                    f"No cached chain for {underlying} {d} (data_dir={data_dir}) and no --v0/"
                    f"[bates] V_0 fallback. Run a real backtest for this date first, or pass --v0."
                )
            s0 = explicit_s0

        bates_kwargs = dict(loaded["bates"])
        bates_kwargs["V_0"] = v0_var
        per_day.append(monte_carlo.DayMarket(day=d, s0=float(s0), bates=BatesParams(**bates_kwargs)))

    spread = SpreadConfig(**loaded["spread"])
    strike_grid = StrikeGrid(**loaded["strike_grid"])

    # seed: config default unless CLI passed; CLI -1 => fresh randomness (None).
    seed: Optional[int]
    if args.seed is not None:
        seed = None if args.seed == -1 else args.seed
    else:
        seed = _coerce_seed(runtime.get("seed"))

    settlement_time = _parse_time(runtime.get("settlement_time", "16:00"), default=time(16, 0))
    cost_model = runtime.get("cost_model", "ibkr")

    config = monte_carlo.MCConfig(
        strategy_cls=strategy_cls,
        param_grid=param_grid,
        per_day_seeds=per_day,
        n_paths=int(runtime.get("n_paths", 100)),
        seed=seed,
        starting_cash=float(runtime.get("starting_cash", 10_000.0)),
        multiplier=int(runtime.get("multiplier", 100)),
        settlement_time=settlement_time,
        cost_fn=cost_model_from_name(cost_model),
        spread=spread,
        strike_grid=strike_grid,
        _underlying=underlying,
    )

    seed_summary = "; ".join(
        f"{dm.day.isoformat()}: s0={dm.s0:.2f}, V0_vol={dm.bates.V_0**0.5:.4f}" for dm in per_day
    )
    print(f"Running Monte Carlo: {config.n_paths} paths/day x {len(config.param_grid)} param sets "
          f"(strategy={args.strategy.split(':')[-1]}, ticker={underlying}, days={[d.isoformat() for d in config.days]}, "
          f"seed={seed})\n  market seeds: {seed_summary}")
    results = monte_carlo.run_mc(config)
    summary = monte_carlo.aggregate_results(results, starting_cash=config.starting_cash)

    # Report every param column (varies per strategy), the combined distribution, and the
    # per-day distribution columns (suffixed _day_<iso>_...).
    param_cols = [c for c in summary.columns if c.startswith("param_")]
    dist_cols = [
        "ending_equity_mean", "ending_equity_p05", "ending_equity_p50", "ending_equity_p95",
        "total_return_mean_pct", "total_return_p05_pct", "total_return_p95_pct",
        "max_drawdown_mean_pct", "max_drawdown_p05_pct",
        "win_rate_mean_pct", "fills_mean", "realized_pnl_mean",
    ]
    day_cols = sorted(c for c in summary.columns if "_day_" in c and c.startswith(("total_return_mean_pct", "win_rate_mean_pct", "max_drawdown_mean_pct")))
    present = _unique_cols([*param_cols, "n_paths", *dist_cols, *day_cols])
    print("\n=== Per-parameter-set distribution (combined over all days x paths) ===")
    print(summary[present].to_string(index=False))

    # Write the MONTE-CARLO output tree.
    write_monte_carlo(
        strategy_cls,
        results,
        output_root=Path(args.output_dir),
    )
    mc_dir = Path(args.output_dir) / strategy_cls.__name__.upper() / "MONTE-CARLO"
    print(f"\nMONTE-CARLO written to: {mc_dir.resolve()}")
    return summary


def _unique_cols(cols: list) -> list:
    """Ordered de-duplicate, preserving first occurrence."""
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _parse_time(raw: Optional[str], default: time) -> time:
    if not raw:
        return default
    try:
        hour, _, minute = raw.partition(":")
        return time(int(hour), int(minute) if minute else 0)
    except ValueError as exc:
        raise ValueError(f"Bad --settlement-time {raw!r}; expected HH:MM") from exc


def _coerce_seed(raw) -> Optional[int]:
    if raw is None:
        return None
    val = int(raw)
    return None if val == -1 else val


if __name__ == "__main__":
    main()
