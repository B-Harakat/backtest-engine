#!/usr/bin/env python3
"""
CLI entrypoint.

Usage:
    python scripts/run_backtest.py --strategy strategies.my_strategy:MyStrategy \\
        --start 2026-01-15 --end 2026-01-16 --cash 10000 --ticker XSP

--strategy takes a "module.path:ClassName" spec (like uvicorn's app spec) rather than a fixed
import, so you can point this at any strategy module in strategies/ without editing this file.
The class must be constructible with no arguments (or accept an `underlying=` keyword, if you
pass --ticker) — if your strategy needs other parameters, give them sensible defaults in
__init__ and adjust in code, or extend this script with your own flags.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure `engine`/`strategies`/`reports` are importable when this script is run directly
# (e.g. `python scripts/run_backtest.py`) rather than as an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.data_store import DEFAULT_DATA_DIR
from engine.entities import DEFAULT_OPTION_MULTIPLIER
from engine.greeks import DEFAULT_CASH_SETTLEMENT_TIME_ET
from engine.runner import run_backtest
from engine.thetadata_client import fetch_from_thetadata
from reports.report import print_summary, save_report


def load_strategy_class(spec: str):
    module_path, sep, class_name = spec.partition(":")
    if not sep:
        raise ValueError(f"--strategy must be 'module.path:ClassName', got {spec!r}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"{module_path!r} has no attribute {class_name!r}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the options backtest engine.")
    parser.add_argument("--strategy", required=True, help="e.g. strategies.my_strategy:MyStrategy")
    parser.add_argument(
        "--ticker", default=None,
        help="Underlying symbol, passed to the strategy's constructor as underlying=<TICKER> "
             "(default: whatever the strategy class itself defaults to, e.g. 'XSP')",
    )
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--cash", type=float, default=10_000.0, help="Starting cash (default: 10000)")
    parser.add_argument("--slippage", type=float, default=0.0, help="Flat per-contract slippage (default: 0.0)")
    parser.add_argument(
        "--multiplier", type=int, default=DEFAULT_OPTION_MULTIPLIER,
        help=f"Contract multiplier, $ per point per contract (default: {DEFAULT_OPTION_MULTIPLIER})",
    )
    parser.add_argument(
        "--settlement-style", choices=["cash", "physical"], default="cash",
        help="'cash' for cash-settled European-style products (XSP/SPX weeklies/RUT/NDX/VIX); "
             "'physical' for American-style equity/ETF options -- these are never actually "
             "physically settled, positions are force-closed via a market order before "
             "expiration instead (default: cash)",
    )
    parser.add_argument(
        "--settlement-time", default=DEFAULT_CASH_SETTLEMENT_TIME_ET.strftime("%H:%M"),
        help=f"Cash-settlement cutoff, HH:MM 24h ET -- only used with --settlement-style=cash "
             f"(default: {DEFAULT_CASH_SETTLEMENT_TIME_ET.strftime('%H:%M')})",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Local Parquet cache directory")
    parser.add_argument("--output-dir", default="backtest_results", help="Where to write reports")
    parser.add_argument("--debug", action="store_true", help="Verbose logging: shows every cache-gap fetch and how long it took")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(message)s" if args.debug else "%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )


    strategy_cls = load_strategy_class(args.strategy)
    strategy_kwargs = {}
    if args.ticker is not None:
        strategy_kwargs["underlying"] = args.ticker
    try:
        strategy = strategy_cls(**strategy_kwargs)
    except TypeError as exc:
        if args.ticker is not None:
            raise ValueError(
                f"{args.strategy!r} doesn't accept an `underlying=` constructor argument -- "
                f"--ticker only works with strategies that do (see strategies/my_strategy.py)."
            ) from exc
        raise

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    settlement_time = datetime.strptime(args.settlement_time, "%H:%M").time()

    result = run_backtest(
        strategy,
        start_date,
        end_date,
        starting_cash=args.cash,
        slippage=args.slippage,
        multiplier=args.multiplier,
        settlement_style=args.settlement_style,
        settlement_time=settlement_time,
        fetch_fn=fetch_from_thetadata,
        data_dir=Path(args.data_dir),
    )

    summary = save_report(result, starting_cash=args.cash, output_dir=Path(args.output_dir))
    print_summary(summary)
    print(f"\nReports written to: {Path(args.output_dir).resolve()}")
    return summary


if __name__ == "__main__":
    main()