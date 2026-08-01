"""
Base Strategy class.

Subclass this and override `initialize`, `on_bar`, and optionally `before_close`. Everything
else here (`get_bars`, `get_greeks`, `get_chain_snapshot`, `submit_order`, `positions`, `cash`,
`get_datetime`) is the engine-provided surface, wired up by `runner.py`'s `attach()` call before
the backtest starts. There is no threading or scheduling machinery.

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

    def get_chain_snapshot(self, underlying: str, expiration: date, right: str) -> pd.DataFrame:
        """
        Full option chain (every listed strike, one side: `right` = "call" or "put") as of the
        most recent available tick at or before now, indexed by strike, with an
        `underlying_price` column. This is the source for both "what's the current underlying
        price" and "what strikes are listed" (see engine.thetadata_client.fetch_chain_from_thetadata).
        Empty DataFrame if there's no data yet (e.g. no listed expiration for that date). The
        chain is always single-sided, so `right` is REQUIRED — pass "call" OR "put" explicitly;
        there is no default side. Calls and puts are fetched/cached independently — call this
        once per side for a strategy that needs both (e.g. an iron condor).
        """
        return self._data.chain_snapshot(underlying, expiration, self._current_ts, right)

    def get_quote(self, contract: Contract) -> Optional[dict]:
        """`{'bid': ..., 'ask': ...}` for `contract` at or before now, or None if unavailable.
        Uses the numpy-array `quote_at` fast path (see data_store.py) rather than `bar_at`."""
        result = self._data.quote_at(contract, self._current_ts)
        if result is None:
            return None
        bid, ask = result
        return {"bid": bid, "ask": ask}

    def submit_order(
        self,
        contract: Contract,
        side: OrderSide,
        qty: int,
        limit_price: Optional[float] = None,
        order_type: OrderType = "LIMIT",
        group_id: Optional[str] = None,
    ) -> Order:
        """
        Submit an order. `order_type` is "LIMIT" (default) or "MARKET".

        For LIMIT orders you set the price via `limit_price` and the engine only fills once the
        market touches it — you are never filled at a price worse than your limit. For MARKET
        orders `limit_price` must be None and the engine fills at the current touch (ask on a
        buy, bid on a sell) on the next bar it processes. There is no slippage model: MARKET
        fills are perfect executions at the book.
        """
        if order_type not in ("LIMIT", "MARKET"):
            raise ValueError(f"Unknown order_type {order_type!r} — must be 'LIMIT' or 'MARKET'")
        if order_type == "LIMIT" and limit_price is None:
            raise ValueError("LIMIT orders require a limit_price")

        order = Order(
            contract=contract,
            side=side,
            qty=qty,
            order_type=order_type,
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
        """Pre-fetch/pre-cache `contract`'s data for [start, end] (default: the full backtest
        window). Call this from `initialize()` for every contract you know in advance you'll
        need, so its data is fetched once up front rather than lazily mid-loop the first time you
        submit an order on it."""
        start = start or datetime.combine(self._backtest_start, time.min)
        end = end or datetime.combine(self._backtest_end, time.max)
        self._data.warm(contract, start, end)