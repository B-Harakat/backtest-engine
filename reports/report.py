"""
Reporting: trade log CSV, equity curve CSV + PNG plot, and a plain-text/JSON summary.

Deliberately not a tearsheet system (see the architecture guideline, section 9) — this covers
"can I tell if the strategy worked" without building HTML report machinery before you've even
validated the engine against real data. Build something fancier once you trust the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless — this runs from a CLI script, not a notebook
import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from engine.runner import BacktestResult


def fills_to_dataframe(fills: list) -> pd.DataFrame:
    rows = [
        {
            "timestamp": f.timestamp,
            "contract": f.contract.key,
            "qty": f.qty,
            "price": f.price,
            "reason": f.reason,
            "group_id": f.group_id,
        }
        for f in fills
    ]
    return pd.DataFrame(rows)


def build_summary(result: "BacktestResult", starting_cash: float) -> dict:
    equity = result.equity_curve["equity"]
    if equity.empty:
        raise ValueError("equity_curve is empty — nothing to summarize")

    total_return_pct = (equity.iloc[-1] / starting_cash - 1.0) * 100.0

    running_max = equity.cummax()
    drawdown_pct = (equity - running_max) / running_max * 100.0
    max_drawdown_pct = float(drawdown_pct.min())

    events = result.realized_pnl_events
    wins = [e.pnl for e in events if e.pnl > 0]
    losses = [e.pnl for e in events if e.pnl < 0]
    num_closes = len(events)

    return {
        "starting_cash": starting_cash,
        "ending_equity": float(equity.iloc[-1]),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": max_drawdown_pct,
        "num_bars": len(equity),
        "num_fills": len(result.fills),
        "num_closing_trades": num_closes,
        "win_rate_pct": (len(wins) / num_closes * 100.0) if num_closes else None,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        # ratio of gross profit to gross loss; None (not infinite) when there are no losses to
        # divide by, since "no losing trades yet" isn't meaningfully "infinite edge"
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None,
        "total_realized_pnl": float(result.realized_pnl),
    }


def print_summary(summary: dict) -> None:
    print("=== Backtest Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:>20}: {value:,.2f}")
        else:
            print(f"{key:>20}: {value}")


def plot_equity_curve(equity_curve: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    equity_curve["equity"].plot(ax=ax)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Account Equity ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_report(result: "BacktestResult", starting_cash: float, output_dir: Path) -> dict:
    """
    Writes trade_log.csv, equity_curve.csv, equity_curve.png, and summary.json to `output_dir`.
    Returns the summary dict (also used by the CLI to print to stdout).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fills_to_dataframe(result.fills).to_csv(output_dir / "trade_log.csv", index=False)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    plot_equity_curve(result.equity_curve, output_dir / "equity_curve.png")

    summary = build_summary(result, starting_cash)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary