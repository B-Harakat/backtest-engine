"""
Base Strategy class.

Subclass this and override `initialize`, `on_bar`, and optionally `before_close`. Everything
else here (`get_bars`, `get_greeks`, `get_chain_snapshot`, `submit_order`, `positions`, `cash`,
`get_datetime`) is the engine-provided surface, wired up by `runner.py`'s `attach()` call before
the backtest starts — mirrors Lumibot's ergonomics (`self.submit_order(...)`,
`self.get_last_price(...)`) without any of its threading/scheduling machinery. See the
architecture guideline, section 8.

There is no separate "current underlying price" argument passed into `on_bar` — ThetaData's
Index/Stock price history requires a subscription tier beyond Options Standard. Use
`get_chain_snapshot()` instead, which sources the underlying price (and the real listed strikes)
from options chain data you already have access to.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Optional

import pandas as pd

from engine.data_store import DataProvider
from engine.entities import Contract, Order, OrderSide, Position
from engine.ledger import Portfolio

_GREEK_COLUMNS = ("delta", "gamma", "theta", "vega", "rho", "vanna", "charm", "implied_vol", "underlying_price")


class Strategy:
    def __init__(self) -> None:
        self._portfolio: Optional[Portfolio] = None
        self._data: Optional[DataProvider] = None
        self._current_ts: Optional[datetime] = None
        self._new_orders: list[Order] = []
        self._backtest_start: Optional[date] = None
        self._backtest_end: Optional[date] = None

    # --- wired up by runner.py; not meant to be called from strategy code ---
    def attach(self, portfolio: Portfolio, data: DataProvider, start_date: date, end_date: date) -> None:
        self._portfolio = portfolio
        self._data = data
        self._backtest_start = start_date
        self._backtest_end = end_date

    def _set_clock(self, ts: datetime) -> None:
        self._current_ts = ts

    def _drain_new_orders(self) -> list[Order]:
        orders, self._new_orders = self._new_orders, []
        return orders

    # --- lifecycle hooks: override these ---
    def initialize(self) -> None:
        """Called once before the backtest starts."""

    def on_bar(self, ts: datetime) -> None:
        """Called once per minute bar during regular trading hours."""

    def before_close(self, ts: datetime) -> None:
        """Called once, on the last bar of each session (respects early closes) — the natural
        place for 0DTE flatten-before-close logic. Any orders submitted here are processed
        against this same bar's quotes before end-of-day expiration settlement runs."""

    # --- engine-provided helpers ---
    def get_datetime(self) -> datetime:
        return self._current_ts

    def get_bars(self, contract: Contract, lookback: int = 1) -> pd.DataFrame:
        """Most recent `lookback` bars for `contract`, up to and including now."""
        return self._data.bars_up_to(contract, self._current_ts).tail(lookback)

    def get_greeks(self, contract: Contract) -> Optional[dict]:
        """Most recent greeks snapshot for `contract` at or before now, or None if no data yet."""
        row = self._data.bar_at(contract, self._current_ts)
        if row is None:
            return None
        return {col: row[col] for col in _GREEK_COLUMNS if col in row.index}

    def get_chain_snapshot(self, underlying: str, expiration: date, right: str = "call") -> pd.DataFrame:
        """
        Full option chain (every listed strike, one side: `right` = "call" or "put") as of the
        most recent available tick at or before now, indexed by strike, with an
        `underlying_price` column. This is the source for both "what's the current underlying
        price" and "what strikes are listed" — see engine.thetadata_client.fetch_chain_from_thetadata
        for why this comes from the options chain rather than a separate index feed. Empty
        DataFrame if there's no data yet (e.g. no listed expiration for that date). Calls and
        puts are fetched/cached completely independently — call this once per side for a
        strategy that needs both (e.g. an iron condor).
        """
        return self._data.chain_snapshot(underlying, expiration, self._current_ts, right)

    def get_quote(self, contract: Contract) -> Optional[dict]:
        """{'bid': ..., 'ask': ...} for `contract` at or before now, or None if unavailable."""
        row = self._data.bar_at(contract, self._current_ts)
        if row is None or "bid" not in row.index or "ask" not in row.index:
            return None
        return {"bid": row["bid"], "ask": row["ask"]}

    def submit_order(
        self,
        contract: Contract,
        side: OrderSide,
        qty: int,
        limit_price: float,
        group_id: Optional[str] = None,
    ) -> Order:
        """
        Submit a LIMIT order at `limit_price` — you set the price, the engine never guesses one
        for you. This is the only order type reachable from strategy code: there's no market-
        order path here, so a strategy can never silently accept an unknown/moving fill price.

        The only market orders this engine ever creates are the automatic
        force-close-before-expiration orders for `settlement_style="physical"` (see
        `engine.runner._force_close_expiring_positions`) — those bypass this method entirely,
        since they exist specifically to guarantee a fill and must never risk resting unfilled.

        An unfilled limit order simply stays open and is re-checked against every subsequent
        bar's bid/ask until it fills (or you cancel it yourself) — there's no automatic
        cancellation or conversion to market.
        """
        order = Order(
            contract=contract,
            side=side,
            qty=qty,
            order_type="LIMIT",
            limit_price=limit_price,
            submitted_at=self._current_ts,
            group_id=group_id,
        )
        self._new_orders.append(order)
        return order

    def positions(self) -> list[Position]:
        return list(self._portfolio.positions.values())

    def get_position(self, contract: Contract) -> Optional[Position]:
        return self._portfolio.positions.get(contract.key)

    def cash(self) -> float:
        return self._portfolio.cash

    def watch(self, contract: Contract, start: Optional[datetime] = None, end: Optional[datetime] = None) -> None:
        """
        Pre-fetch/pre-cache `contract`'s data for [start, end] (defaults to the full backtest
        window). Call this from `initialize()` for every contract you know in advance you'll
        need — it's the difference between one warm-up fetch and (potentially) a fetch triggered
        mid-loop the first time you submit an order on a contract you never mentioned. The runner
        will lazily warm any contract you didn't, but pre-declaring is the more predictable path,
        especially the first time you run against a cold local cache.
        """
        start = start or datetime.combine(self._backtest_start, time.min)
        end = end or datetime.combine(self._backtest_end, time.max)
        self._data.warm(contract, start, end)