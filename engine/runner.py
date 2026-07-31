"""
Composition root — `run_backtest()` is the one function that wires everything else in `engine/`
together into an actual backtest. No threading, no scheduler: this is a plain loop over
`calendar.trading_minutes()`.

Per-bar sequence, matching the architecture guideline section 4:
    1. strategy.on_bar(ts)      -- strategy may submit orders
    2. fill engine processes all open orders against this bar's quotes
    3. on the last bar of a session: strategy.before_close(ts), then process any orders it
       submitted there, then expiration handling for anything expiring today (see
       `settlement_style` below)
    4. mark-to-market equity snapshot recorded for this bar

There is no separate "underlying index" feed here: ThetaData's Index/Stock history endpoints
require a subscription tier beyond Options Standard. The underlying's price is instead sourced
from data you already have access to — per-position, from `option_history_greeks_first_order`'s
embedded `underlying_price` column (used below for cash settlement), and for pre-trade strike
selection, via `Strategy.get_chain_snapshot()` (see engine.thetadata_client.fetch_chain_from_thetadata).

This engine assumes ONE underlying (and one multiplier, one settlement style) per backtest run —
set via `run_backtest`'s parameters, not inferred from the traded contracts. Trading several
different underlyings with different multipliers/settlement conventions in the same run isn't
supported; run separate backtests instead.

Settlement styles:
    "cash" (default) — the underlying is a cash-settled, European-style product (XSP, SPX
        weeklies/0DTE, RUT, NDX, VIX). Expiring positions are cash-settled at intrinsic value —
        see engine.settlement.expire_and_settle.
    "physical" — the underlying's real-world options are American-style and physically settled
        (single-stock/ETF options). This engine NEVER simulates physical settlement: instead,
        any position still open on its own expiration day is force-closed via an ordinary market
        order (at that contract's own bid/ask) during the day's last bar, before the actual
        exercise/assignment decision would occur (which happens after the close). This does NOT
        model early-assignment risk on short positions held on earlier days — a counterparty
        could in principle exercise early at any time before expiration, which this engine has no
        visibility into or control over. It only guarantees the engine itself never carries a
        position through to its own expiration/physical settlement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from engine.calendar import session_bounds, session_close, session_open, trading_minutes
from engine.data_store import DEFAULT_DATA_DIR, ChainFetchFn, DataProvider, FetchFn
from engine.entities import DEFAULT_OPTION_MULTIPLIER, Fill, Order
from engine.fill_engine import Quote, process_pending_orders
from engine.greeks import DEFAULT_CASH_SETTLEMENT_TIME_ET
from engine.ledger import Portfolio
from engine.settlement import expire_and_settle
from engine.strategy import Strategy
from engine.thetadata_client import fetch_chain_from_thetadata, fetch_from_thetadata

logger = logging.getLogger(__name__)

SettlementStyle = Literal["cash", "physical"]


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame  # columns: cash, equity; indexed by timestamp, one row per bar
    fills: list[Fill]
    realized_pnl: float
    realized_pnl_events: list  # list[engine.ledger.RealizedPnLEvent], for win-rate/profit-factor reporting


def _build_quotes_for_orders(data: DataProvider, orders: list[Order], ts: datetime) -> dict[str, Quote]:
    quotes: dict[str, Quote] = {}
    for order in orders:
        if order.status != "OPEN" or order.contract.key in quotes:
            continue
        result = data.quote_at(order.contract, ts)
        if result is None:
            continue
        bid, ask = result
        quotes[order.contract.key] = Quote(bid=bid, ask=ask)
    return quotes


def _process_new_orders(
    strategy: Strategy,
    data: DataProvider,
    portfolio: Portfolio,
    open_orders: list[Order],
    ts: datetime,
    slippage: float,
    session_start: datetime,
    session_end: datetime,
) -> None:
    for order in strategy._drain_new_orders():
        if not data.is_warmed(order.contract):
            # Strategy submitted an order on a contract it never called self.watch() for.
            # Warm it lazily rather than fail — but this is the less predictable path; prefer
            # calling self.watch(contract) in initialize() for anything you know about in advance.
            #
            # Option contracts get scoped to the window between the day the ORDER was actually
            # submitted and the contract's OWN expiration date's session close -- not the whole
            # backtest range (which could span weeks either side of a contract's real trading
            # window) and not JUST the expiration day's session either. That second mistake is a
            # real bug this exact line used to have: for a 0DTE contract, entry day == expiration
            # day, so scoping to "just the expiration day" happened to look correct and every
            # strategy tested against this engine so far was 0DTE. The first time a genuinely
            # multi-day position was tried, it broke silently: the order sat with literally no
            # quote to fill against for every day between submission and expiration (since only
            # the expiration day's session was ever fetched), then finally filled on the
            # expiration day's first bar instead -- days late, at a completely different price,
            # with no error raised to reveal any of this had happened. `compute_gaps` cannot ever
            # resolve a gap outside a session's actual hours (see its docstring) which is why the
            # window must be bounded correctly up front, not widened after the fact.
            if order.contract.is_option:
                warm_start = session_open(order.submitted_at.date())
                warm_end = session_close(order.contract.expiration)
            else:
                warm_start, warm_end = session_start, session_end
            data.warm(order.contract, warm_start, warm_end)
        open_orders.append(order)

    quotes = _build_quotes_for_orders(data, open_orders, ts)
    fills = process_pending_orders(open_orders, quotes, ts, slippage=slippage)
    for fill in fills:
        portfolio.apply_fill(fill)

    open_orders[:] = [o for o in open_orders if o.status == "OPEN"]


def _mark_to_market_prices(data: DataProvider, portfolio: Portfolio, ts: datetime) -> dict[str, float]:
    prices: dict[str, float] = {}
    for key, pos in portfolio.positions.items():
        result = data.quote_at(pos.contract, ts)
        if result is not None:
            bid, ask = result
            prices[key] = (bid + ask) / 2.0
            continue
        # Fallback for the rare case a contract has no bid/ask at all (thin/no-quote data) --
        # uncommon in practice, since this engine always fetches bid/ask alongside OHLC, so it's
        # fine for this rarely-hit branch to go through the slower full-row bar_at().
        row = data.bar_at(pos.contract, ts)
        if row is not None and "close" in row.index and not pd.isna(row["close"]):
            prices[key] = row["close"]
    return prices


def _settlement_price(data: DataProvider, portfolio: Portfolio, day: date, settlement_time: time) -> Optional[float]:
    """
    The underlying's settlement print for `day`, read from one of the actually-expiring
    positions' own already-fetched bar data — specifically the `underlying_price` column that
    `option_history_greeks_first_order` returns for every row. This needs no additional ThetaData
    call: that data is already sitting in the position's warmed frame from whatever fetch brought
    in its option data in the first place. Only used for `settlement_style="cash"`.

    Still subject to the same settlement-timing caveat as before — see `engine/settlement.py`'s
    docstring on settlement conventions varying by product and expiration cadence. "Most recent
    tick at or before the settlement instant" is a best-effort proxy if no tick exists exactly at
    `settlement_time`.
    """
    settlement_ts = datetime.combine(day, settlement_time)
    expiring_positions = [
        pos for pos in portfolio.positions.values()
        if pos.contract.is_option and pos.contract.expiration == day
    ]
    for pos in expiring_positions:
        row = data.bar_at(pos.contract, settlement_ts)
        if row is not None and "underlying_price" in row.index and not pd.isna(row["underlying_price"]):
            return float(row["underlying_price"])
    return None


def _force_close_expiring_positions(
    data: DataProvider,
    portfolio: Portfolio,
    day: date,
    ts: datetime,
    slippage: float,
) -> None:
    """
    Physical-settlement avoidance (`settlement_style="physical"`): force-closes every open
    position expiring on `day` via an ordinary market order, filled at that contract's own
    bid/ask, during the day's last bar — before the actual exercise/assignment decision would be
    made (which happens after the close). This converts what would have been a physical
    settlement into a normal closing trade.

    Does NOT model early-assignment risk on short American-style positions held on days BEFORE
    their own expiration — a counterparty could in principle exercise early at any time, which
    this engine has no visibility into. It only guarantees the engine itself never carries a
    position through to its own expiration.

    Raises if any expiring position can't be filled (e.g. no quote available at this exact
    instant) — silently leaving it open would violate the "never physically settled" guarantee
    this whole mechanism exists to provide.
    """
    expiring_positions = [
        pos for pos in portfolio.positions.values()
        if pos.contract.is_option and pos.contract.expiration == day
    ]
    if not expiring_positions:
        return

    close_orders = [
        Order(
            contract=pos.contract,
            side="SELL_TO_CLOSE" if pos.qty > 0 else "BUY_TO_CLOSE",
            qty=abs(pos.qty),
            order_type="MARKET",
            submitted_at=ts,
        )
        for pos in expiring_positions
    ]

    quotes = _build_quotes_for_orders(data, close_orders, ts)
    fills = process_pending_orders(close_orders, quotes, ts, slippage=slippage)
    for fill in fills:
        portfolio.apply_fill(fill)

    unfilled = [o for o in close_orders if o.status != "FILLED"]
    if unfilled:
        raise RuntimeError(
            f"Could not force-close {len(unfilled)} physically-settled position(s) expiring on "
            f"{day} before expiration (no fillable quote at {ts}) — refusing to let these reach "
            f"physical settlement. Contracts: {[o.contract.key for o in unfilled]}"
        )


def run_backtest(
    strategy: Strategy,
    start_date: date,
    end_date: date,
    starting_cash: float,
    slippage: float = 0.0,
    multiplier: int = DEFAULT_OPTION_MULTIPLIER,
    settlement_style: SettlementStyle = "cash",
    settlement_time: time = DEFAULT_CASH_SETTLEMENT_TIME_ET,
    fetch_fn: FetchFn = fetch_from_thetadata,
    chain_fetch_fn: Optional[ChainFetchFn] = fetch_chain_from_thetadata,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> BacktestResult:
    session_start = datetime.combine(start_date, time.min)
    session_end = datetime.combine(end_date, time.max)

    data = DataProvider(fetch_fn=fetch_fn, data_dir=data_dir, chain_fetch_fn=chain_fetch_fn)
    portfolio = Portfolio(starting_cash=starting_cash, multiplier=multiplier)
    strategy.attach(portfolio, data, start_date, end_date)

    strategy.initialize()

    open_orders: list[Order] = []
    equity_rows: list[dict] = []
    current_day: Optional[date] = None
    current_day_last_bar: Optional[datetime] = None

    for ts in trading_minutes(start_date, end_date):
        if ts.date() != current_day:
            current_day = ts.date()
            # session_bounds() rebuilds pandas_market_calendars' full schedule -- expensive
            # (holiday rules, session-day construction). Computing this once per day here, rather
            # than calling is_last_bar_of_day(ts) on every one of the ~390 bars in the day (which
            # was recomputing the same schedule from scratch on every single bar), was the actual
            # dominant cost of a fully-cached run -- not Python itself, not pandas, not I/O.
            _, current_day_last_bar = session_bounds(current_day)
            logger.info("Processing %s", current_day)

        strategy._set_clock(ts)
        strategy.on_bar(ts)
        _process_new_orders(strategy, data, portfolio, open_orders, ts, slippage, session_start, session_end)

        if ts == current_day_last_bar:
            strategy.before_close(ts)
            _process_new_orders(strategy, data, portfolio, open_orders, ts, slippage, session_start, session_end)

            day = ts.date()
            if settlement_style == "cash":
                settlement_price = _settlement_price(data, portfolio, day, settlement_time)
                if settlement_price is not None:
                    expire_and_settle(portfolio, day, settlement_price, settlement_time)
                elif any(pos.contract.is_option and pos.contract.expiration == day for pos in portfolio.positions.values()):
                    # We have positions expiring today but couldn't find a settlement price —
                    # don't silently skip this, it would leave a "zombie" expired position.
                    raise RuntimeError(
                        f"Positions expiring on {day} but no settlement price found — none of the "
                        f"expiring positions' own bar data has an underlying_price at/near "
                        f"{settlement_time}. Check that contract data covers this date/time."
                    )
            else:  # "physical"
                _force_close_expiring_positions(data, portfolio, day, ts, slippage)

        mid_prices = _mark_to_market_prices(data, portfolio, ts)
        equity = portfolio.mark_to_market(mid_prices)
        equity_rows.append({"timestamp": ts, "cash": portfolio.cash, "equity": equity})

    equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
    return BacktestResult(
        equity_curve=equity_curve,
        fills=portfolio.fills,
        realized_pnl=portfolio.realized_pnl,
        realized_pnl_events=portfolio.realized_pnl_log,
    )