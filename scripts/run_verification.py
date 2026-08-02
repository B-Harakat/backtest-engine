#!/usr/bin/env python3
"""
Verify that a backtest's reported statistics are consistent with the actual fill stream.

This script has the SAME command-line interface as ``scripts/run_backtest.py``. It runs
``run_backtest`` with the given arguments (exactly as running run_backtest standalone), which
populates ``output/<ticker>/<strategy>/HISTORY/`` with ``trade_log.csv`` / ``cash_equity.csv`` /
``summary.json``. It then:

  1. reads ``summary.json`` (the reported statistics),
  2. independently recomputes the fill-derivable statistics by replaying the rows of the written
     ``trade_log.csv`` through a small position-accounting model written from scratch in this file,
  3. prints a comparison and exits non-zero (1) if anything mismatches.

Only statistics directly derivable from the fill stream and the (flat-book) equity endpoint are
checked: ending equity / return, realized P&L, commission, fill and closing-trade counts,
win rate, avg win/loss, profit factor. Curve-dependent stats (drawdown, Sharpe, VaR, ...) need the
minute equity curve and are not independently recomputed here.

Exit codes: 0 = all statistics match; 1 = a mismatch; 2 = the backtest failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure `engine`/`strategies`/`reports` are importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.costs import cost_model_from_name  # noqa: E402
from engine.data_store import DEFAULT_DATA_DIR  # noqa: E402
from engine.entities import DEFAULT_OPTION_MULTIPLIER, Contract  # noqa: E402
from engine.greeks import DEFAULT_CASH_SETTLEMENT_TIME_ET  # noqa: E402
from engine.runner import run_backtest  # noqa: E402
from engine.thetadata_client import fetch_from_thetadata  # noqa: E402

# Tolerances for "recomputed == reported".
REL_TOL = 1e-6      # relative tolerance for dollar/premium quantities
ABS_TOL = 1e-6      # absolute tolerance for percentages / counts


# --------------------------------------------------------------------------------------------
# CLI — identical to run_backtest.py
# --------------------------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    import importlib  # local to keep the file flat

    p = argparse.ArgumentParser(description="Verify reported backtest statistics against the trade log.")
    p.add_argument("--strategy", required=True, help="e.g. strategies.my_strategy:MyStrategy")
    p.add_argument("--ticker", default=None, help="Underlying symbol, passed to the strategy as underlying= (default: strategy default)")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--cash", type=float, default=10_000.0, help="Starting cash (default: 10000)")
    p.add_argument("--multiplier", type=int, default=DEFAULT_OPTION_MULTIPLIER)
    p.add_argument("--settlement-style", choices=["cash", "physical"], default="cash")
    p.add_argument("--settlement-time", default=DEFAULT_CASH_SETTLEMENT_TIME_ET.strftime("%H:%M"))
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir", default="output")
    p.add_argument("--cost-model", default="ibkr")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def load_strategy_class(spec: str):
    import importlib

    module_path, sep, class_name = spec.partition(":")
    if not sep:
        raise ValueError(f"--strategy must be 'module.path:ClassName', got {spec!r}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"{module_path!r} has no attribute {class_name!r}") from exc


# --------------------------------------------------------------------------------------------
# Independent replay of the written trade_log.csv
# --------------------------------------------------------------------------------------------


def _contract_from_key(key: str) -> Contract:
    """Parse an engine canonical contract key (e.g. 'XSP_20260706_C756' or 'XSP') into a Contract."""
    parts = key.split("_")
    if len(parts) == 1:
        return Contract(underlying=parts[0])
    underlying, exp = parts[0], parts[1]
    right = "CALL" if parts[2][0] == "C" else "PUT"
    return Contract(
        underlying=underlying,
        expiration=datetime.strptime(exp, "%Y%m%d").date(),
        strike=float(parts[2][1:]),
        right=right,
    )


def replay_trade_log(csv_path: Path, starting_cash: float, multiplier: int) -> dict:
    """
    Recompute fill-derivable statistics from trade_log.csv using a small, self-contained
    position-accounting model (written here, not reusing engine.ledger).

    Ledger conventions (re-derived, matching the engine):
      - BUY fills have positive qty, SELL negative qty.
      - cash -= qty*price*multiplier ; cash -= commission.
      - Realized P&L on a close = closing_qty*(fill.price - avg_price)*multiplier*sign,
        sign = +1 for longs, -1 for shorts.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    positions: dict[str, tuple[int, float]] = {}
    realized: list[float] = []
    cash = float(starting_cash)
    total_commission = 0.0

    for _, row in df.iterrows():
        key = str(row["contract"])
        qty = int(row["qty"])
        price = float(row["price"])
        comm = float(row["commission"])
        mult = multiplier

        cash -= qty * price * mult  # cash -= qty*price*mult
        cash -= comm                # cash -= commission
        total_commission += comm

        cur_qty, cur_avg = positions.get(key, (0, 0.0))

        if cur_qty == 0 or (cur_qty > 0) == (qty > 0):
            new_qty = cur_qty + qty
            if new_qty:
                cur_avg = (cur_qty * cur_avg + qty * price) / new_qty
            positions[key] = (new_qty, cur_avg)
        else:
            closing_qty = min(abs(cur_qty), abs(qty))
            sign = 1 if cur_qty > 0 else -1
            realized.append(closing_qty * (price - cur_avg) * mult * sign)
            new_qty = cur_qty + qty
            if new_qty == 0:
                cur_avg = 0.0
            elif (new_qty > 0) == (cur_qty > 0):
                pass  # partial close: basis unchanged
            else:
                cur_avg = price  # flipped: excess is a new position at this price
            positions[key] = (new_qty, cur_avg)

    flat = all(q == 0 for q, _ in positions.values())
    pnls = realized
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    num_closes = len(pnls)

    ending_equity = float(cash) if flat else None
    return {
        "ending_equity": ending_equity,
        "total_return_pct": ((ending_equity / starting_cash - 1.0) * 100.0) if (flat and starting_cash) else None,
        "num_fills": len(df),
        "num_orders_filled": int(df[df["reason"] == "ORDER"].shape[0]),
        "num_closing_trades": num_closes,
        "win_rate_pct": (len(wins) / num_closes * 100.0) if num_closes else None,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None,
        "total_realized_pnl": float(sum(pnls)),
        "total_commission": total_commission,
    }


# --------------------------------------------------------------------------------------------
# Comparison and reporting
# --------------------------------------------------------------------------------------------


def _close(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    if b == 0.0:
        return abs(a - b) <= ABS_TOL
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def compare(reported: dict, derived: dict) -> tuple[bool, list[str]]:
    problems: list[str] = []
    ok = True
    for key in derived:
        match = _close(reported.get(key), derived[key])
        ok = ok and match
        if not match:
            problems.append(f"{key}: reported={reported.get(key)!r} derived={derived[key]!r}")
    return ok, problems


def main(argv: Optional[list[str]] = None) -> int:
    import logging

    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    strategy_cls = load_strategy_class(args.strategy)
    strategy_kwargs = {}
    if args.ticker is not None:
        strategy_kwargs["underlying"] = args.ticker
    try:
        strategy = strategy_cls(**strategy_kwargs)
    except TypeError as exc:
        if args.ticker is not None:
            raise ValueError(
                f"{args.strategy!r} doesn't accept an `underlying=` constructor argument — "
                f"--ticker only works with strategies that do."
            ) from exc
        raise

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    settlement_time = datetime.strptime(args.settlement_time, "%H:%M").time()
    cost_fn = cost_model_from_name(args.cost_model)

    # Run the backtest exactly as run_backtest.py does — this writes
    # output/<ticker>/<strategy>/HISTORY/{trade_log.csv, cash_equity.csv, summary.json}.
    result = run_backtest(
        strategy,
        start_date,
        end_date,
        starting_cash=args.cash,
        multiplier=args.multiplier,
        settlement_style=args.settlement_style,
        settlement_time=settlement_time,
        fetch_fn=fetch_from_thetadata,
        data_dir=Path(args.data_dir),
        cost_fn=cost_fn,
    )
    from reports.output import write_history

    history_dir = Path(args.output_dir) / strategy.underlying / strategy_cls.__name__ / "HISTORY"
    write_history(
        strategy_cls, result,
        starting_cash=args.cash,
        output_root=Path(args.output_dir),
        ticker=strategy.underlying,
    )
    print(f"HISTORY written to: {history_dir.resolve()}")

    # Reported statistics come from the summary.json already on disk.
    with open(history_dir / "summary.json", "r", encoding="utf-8") as f:
        reported = json.load(f)

    # Independently recompute from the trade_log.csv already on disk.
    derived = replay_trade_log(
        history_dir / "trade_log.csv",
        starting_cash=args.cash,
        multiplier=int(reported.get("multiplier", result.multiplier)),
    )

    ok, problems = compare(reported, derived)

    header = f"{'stat':<22}{'reported':>16}{'derived':>16}{'match':>10}"
    print("\n" + header)
    print("-" * len(header))
    for key in derived:
        print(f"{key:<22}{str(reported.get(key)):>16}{str(derived[key]):>16}"
              f"{'OK' if _close(reported.get(key), derived[key]) else '** MISMATCH **':>10}")

    if not ok:
        print("\nMISMATCHES FOUND:")
        for p in problems:
            print(f"  * {p}")
        print("\nVERIFICATION FAILED — reported statistics do not match an independent replay of the trade log.")
        return 1

    print("\nVERIFICATION PASSED — reported statistics match an independent replay of the trade log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
