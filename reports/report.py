"""
Reporting: trade log CSV, equity curve CSV + PNG plot, and a plain-text/JSON summary.

The summary carries a professional statistical battery computed from data already in the
`BacktestResult`: risk-adjusted returns (Sharpe/Sortino/Calmar/Omega), drawdown depth & duration,
tail risk (VaR/CVaR, skew/kurtosis), and per-trade stats (expectancy, streaks, holding period,
turnover, fee drag). Everything here is derived from the equity curve, fills, and realized-P&L
events the engine already produces.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from engine.runner import BacktestResult


_TRADING_DAYS_PER_YEAR = 252.0


def fills_to_dataframe(fills: list) -> pd.DataFrame:
    rows = [
        {
            "timestamp": f.timestamp,
            "contract": f.contract.key,
            "qty": f.qty,
            "price": f.price,
            "reason": f.reason,
            "group_id": f.group_id,
            "commission": f.commission,
        }
        for f in fills
    ]
    return pd.DataFrame(rows)


# --- daily-return statistics -------------------------------------------------------------


def _daily_returns(equity: pd.Series) -> pd.Series:
    """Resample the (1-minute) equity curve to daily closes and produce daily simple returns."""
    daily_equity = equity.resample("1D").last().dropna()
    return daily_equity.pct_change().dropna()


def _annualized_return_pct(equity: pd.Series, starting_cash: float, years: float) -> float:
    if years <= 0 or starting_cash <= 0:
        return 0.0
    ending = float(equity.iloc[-1])
    if ending <= 0:
        return -100.0
    return (ending / starting_cash) ** (1.0 / years) - 1.0


def _years_covering(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return (equity.index[-1] - equity.index[0]).days / 365.0


def compute_risk_return_metrics(equity: pd.Series, starting_cash: float) -> dict:
    """Return- and risk-adjusted metrics computed from the daily-return series."""
    daily = _daily_returns(equity)
    has_returns = len(daily) >= 2

    std = float(daily.std()) if has_returns else 0.0
    mean = float(daily.mean()) if has_returns else 0.0

    # downside deviation: std of returns below zero (the Sortino denominator)
    downside = daily[daily < 0] if has_returns else daily
    downside_std = float(downside.std()) if len(downside) >= 2 else None

    ann_vol = std * np.sqrt(_TRADING_DAYS_PER_YEAR) if has_returns else None
    ann_ret = _annualized_return_pct(equity, starting_cash, _years_covering(equity))
    cagr = ann_ret  # alias for readability in Calmar

    running_max = equity.cummax()
    drawdown_pct = (equity - running_max) / running_max * 100.0
    max_dd = float(drawdown_pct.min()) if not equity.empty else None
    # Ulcer index = sqrt(mean of squared drawdown %) -- depth-AND-duration weighted
    ulcer = float(np.sqrt((drawdown_pct**2).mean())) if not equity.empty else None
    pain = (ann_ret / (ulcer / 100.0)) if (ulcer and ulcer > 0) else None

    sortino = None
    if downside_std is not None and downside_std > 0:
        sortino = mean / downside_std * np.sqrt(_TRADING_DAYS_PER_YEAR)

    calmar = (cagr / -(max_dd / 100.0)) if (max_dd and max_dd < 0) else None

    # Omega ratio (threshold 0): probability-weighted ratio of gains to losses on daily returns
    omega = None
    if has_returns:
        gains = daily[daily > 0].sum()
        losses = -daily[daily < 0].sum()
        if losses > 0:
            omega = float(gains / losses)

    return {
        "ann_volatility_pct": (ann_vol * 100.0) if ann_vol is not None else None,
        "annualized_return_pct": ann_ret * 100.0,
        "cagr_pct": cagr * 100.0,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "ulcer_index_pct": ulcer,
        "pain_ratio": pain,
        "omega_ratio": omega,
    }


def compute_drawdown_metrics(equity: pd.Series) -> dict:
    """Drawdown duration / frequency statistics from the (minute) equity curve."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max  # fraction, <= 0
    underwater = drawdown < 0

    episodes = []
    start = None
    for ts, is_dd in underwater.items():
        if is_dd and start is None:
            start = ts
        elif not is_dd and start is not None:
            episodes.append((start, ts))
            start = None
    if start is not None:
        episodes.append((start, underwater.index[-1]))

    total_days = (underwater.index[-1] - underwater.index[0]).days if len(underwater) > 1 else 0

    if episodes:
        durations_days = [(end - st).days for st, end in episodes]
        max_dur_days = max(durations_days)
        avg_dur_days = sum(durations_days) / len(durations_days)
        # average % of the run spent underwater = sum of episode durations / total span
        underwater_days = sum(durations_days)
        pct_underwater = (underwater_days / total_days * 100.0) if total_days > 0 else None
    else:
        max_dur_days = avg_dur_days = pct_underwater = None

    return {
        "max_drawdown_duration_days": max_dur_days,
        "avg_time_underwater_days": avg_dur_days,
        "time_underwater_pct": pct_underwater,
        "num_drawdown_episodes": len(episodes),
    }


def compute_tail_metrics(daily: pd.Series, trade_pnls: list) -> dict:
    """VaR / CVaR / moment / extreme-loss metrics from daily returns and per-trade P&L."""
    var_95 = var_99 = cvar_95 = skew = kurt = largest_daily = None
    if len(daily) >= 1:
        largest_daily = float(daily.min()) * 100.0
    if len(daily) >= 2:
        var_95 = float(daily.quantile(0.05)) * 100.0
        var_99 = float(daily.quantile(0.01)) * 100.0
        tail = daily[daily <= daily.quantile(0.05)]
        if len(tail) >= 1:
            cvar_95 = float(tail.mean()) * 100.0
        skew = float(daily.skew())
        kurt = float(daily.kurt())

    largest_trade = min(trade_pnls) if trade_pnls else None
    return {
        "var_95_pct": var_95,
        "var_99_pct": var_99,
        "cvar_95_pct": cvar_95,
        "daily_return_skew": skew,
        "daily_return_kurtosis": kurt,
        "largest_daily_loss_pct": largest_daily,
        "largest_single_trade_loss": largest_trade,
    }


# --- trade-level statistics --------------------------------------------------------------


def _cumulative_trade_pnl(events) -> dict:
    """Consecutive-win/loss streaks and max drawdown-in-trade-count computed from the ordered
    realized-P&L events (a proxy for 'did closing trades keep the book green')."""
    pnls = [e.pnl for e in events]
    best_win_streak = cur_win = 0
    best_loss_streak = cur_loss = 0
    for pnl in pnls:
        if pnl > 0:
            cur_win += 1
            cur_loss = 0
        elif pnl < 0:
            cur_loss += 1
            cur_win = 0
        else:  # breakeven trade breaks neither streak
            cur_win = 0
            cur_loss = 0
        best_win_streak = max(best_win_streak, cur_win)
        best_loss_streak = max(best_loss_streak, cur_loss)

    # drawdown on the cumulative closing-trade P&L line
    cum = np.cumsum(pnls)
    running_peak = np.maximum.accumulate(cum)
    dd = cum - running_peak
    max_trade_dd = float(dd.min()) if len(dd) else None

    return {
        "max_consecutive_wins": best_win_streak,
        "max_consecutive_losses": best_loss_streak,
        "max_trade_drawdown": max_trade_dd,
    }


def _position_open_intervals(fills: list) -> list[tuple[datetime, datetime]]:
    """
    Reconstruct per-contract 'open' intervals from the fill stream (which includes settlement
    EXPIRY fills that flatten positions), by tracking each contract key's running signed qty.
    Returns a list of (start, end) timestamps for every distinct position-open episode.
    """
    from collections import defaultdict

    by_contract: dict[str, int] = defaultdict(int)
    intervals: list[tuple[datetime, datetime]] = []
    in_interval: dict[str, datetime] = {}

    ordered = sorted(fills, key=lambda f: f.timestamp)
    for f in ordered:
        key = f.contract.key
        before = by_contract[key]
        after = before + f.qty
        by_contract[key] = after

        if before == 0 and after != 0:
            in_interval[key] = f.timestamp  # position just opened
        elif before != 0 and after == 0 and key in in_interval:
            intervals.append((in_interval.pop(key), f.timestamp))  # fully closed
    # Ignore any still-open intervals at the end (positions never closed in this sample).
    return intervals


def compute_trade_metrics(
    events: list,
    fills: list,
    equity: pd.Series,
    multiplier: int,
    total_commission: float,
    starting_cash: float,
) -> dict:
    """Per-trade statistics computed entirely from realized-P&L events and the fill stream."""
    pnls = [e.pnl for e in events]
    num_closes = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    breakeven = [p for p in pnls if p == 0]

    expectancy = (sum(pnls) / num_closes) if num_closes else None
    payoff = (
        (sum(wins) / len(wins)) / abs(sum(losses) / len(losses))
    ) if wins and losses else None

    # turnover: gross $ traded across all real order fills (per contract qty * price * multiplier),
    # expressed as a multiple of starting equity so it's comparable regardless of account size.
    gross = sum(abs(f.qty) * f.price * multiplier for f in fills if f.reason == "ORDER")
    turnover_ratio = (gross / starting_cash) if starting_cash else None

    # holding period / time-in-market from reconstructed position intervals
    intervals = _position_open_intervals(fills)
    total_bars = max(len(equity), 1)
    open_bars = sum(
        1 for ts in equity.index if any(st <= ts <= en for st, en in intervals)
    ) if intervals else 0
    time_in_market_pct = (open_bars / total_bars * 100.0) if intervals else 0.0

    avg_holding_days = None
    if intervals:
        durations_days = [(en - st).days for st, en in intervals]
        avg_holding_days = sum(durations_days) / len(durations_days)

    streak_metrics = _cumulative_trade_pnl(events)

    trade_pnl_skew = None
    if len(pnls) >= 2:
        trade_pnl_skew = float(pd.Series(pnls).skew())

    total_pnl = sum(pnls)
    cost_ratio = None
    if total_commission and total_pnl != 0:
        cost_ratio = total_commission / abs(total_pnl)

    return {
        "expectancy": expectancy,
        "payoff_ratio": payoff,
        "turnover_ratio": turnover_ratio,
        "avg_holding_days": avg_holding_days,
        "time_in_market_pct": time_in_market_pct,
        "pct_positive_trades": (len(wins) / num_closes * 100.0) if num_closes else None,
        "pct_negative_trades": (len(losses) / num_closes * 100.0) if num_closes else None,
        "pct_breakeven_trades": (len(breakeven) / num_closes * 100.0) if num_closes else None,
        "trade_pnl_skew": trade_pnl_skew,
        "round_trip_cost_ratio": cost_ratio,
        **streak_metrics,
    }


def build_summary(
    result: "BacktestResult",
    starting_cash: float,
) -> dict:
    equity = result.equity_curve["equity"]
    if equity.empty:
        raise ValueError("equity_curve is empty — nothing to summarize")

    daily = _daily_returns(equity)
    events = result.realized_pnl_events
    pnls = [e.pnl for e in events]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    num_closes = len(events)

    total_return_pct = (equity.iloc[-1] / starting_cash - 1.0) * 100.0

    running_max = equity.cummax()
    drawdown_pct = (equity - running_max) / running_max * 100.0
    max_drawdown_pct = float(drawdown_pct.min())

    # Sharpe ratio, annualized: resample the (1-minute) equity curve to daily closes, take
    # daily simple returns, and scale their mean/std by sqrt(252). None when there aren't at
    # least two trading days (no meaningful daily-return series).
    sharpe_ratio = None
    if len(daily) >= 2 and float(daily.std()) > 0:
        sharpe_ratio = float(daily.mean() / daily.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))

    # Number of real (non-settlement) orders that actually filled. `result.fills` counts fill
    # *events* (one per leg/group), so multiple fills can share one order; `num_orders_filled`
    # is the distinct submitted orders that got at least partially filled. EXPIRY_CASH_SETTLEMENT
    # fills are synthesized by the engine (single artificial "EXPIRY" order id), so they're
    # excluded rather than folded in.
    num_orders_filled = len({f.order_id for f in result.fills if f.reason == "ORDER"})
    total_commission = float(sum(f.commission for f in result.fills))

    metrics = {
        "starting_cash": starting_cash,
        "ending_equity": float(equity.iloc[-1]),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "num_bars": len(equity),
        "num_fills": len(result.fills),
        "num_orders_filled": num_orders_filled,
        "num_closing_trades": num_closes,
        "win_rate_pct": (len(wins) / num_closes * 100.0) if num_closes else None,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        # ratio of gross profit to gross loss; None (not infinite) when there are no losses to
        # divide by, since "no losing trades yet" isn't meaningfully "infinite edge"
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None,
        "total_realized_pnl": float(result.realized_pnl),
        # total transaction cost (USD) paid across all order fills -- commission + fees charged
        # by the configured cost model (see engine/costs.py). Excludes expiration settlements.
        "total_commission": total_commission,
    }

    # Merge the statistical batteries (risk/return, drawdown, tail risk, per-trade).
    metrics.update(compute_risk_return_metrics(equity, starting_cash))
    metrics.update(compute_drawdown_metrics(equity))
    metrics.update(compute_tail_metrics(daily, pnls))
    metrics.update(
        compute_trade_metrics(
            events, result.fills, equity, multiplier=result.multiplier,
            total_commission=total_commission, starting_cash=starting_cash,
        )
    )
    return metrics


def print_summary(summary: dict) -> None:
    print("=== Backtest Summary ===")
    for key, value in summary.items():
        if value is None:
            print(f"{key:>20}: n/a")
        elif isinstance(value, bool):
            print(f"{key:>20}: {value}")
        elif isinstance(value, int):
            print(f"{key:>20}: {value:,}")
        elif isinstance(value, float):
            print(f"{key:>20}: {value:,.2f}")
        else:
            print(f"{key:>20}: {value}")