"""
Matches pending Orders against the current bar's bid/ask and produces Fills.

Deliberately has no knowledge of legs, groups, or spreads (see entities.py docstring) — it fills
each order independently against its own contract's quote. No knowledge of positions/cash either;
that's the ledger's job. This module's only responsibility: given "here are the open orders" and
"here's the current bid/ask for every contract with an open order", decide which orders fill,
at what price, and produce the resulting Fills. The caller (runner.py) is responsible for handing
the resulting Fills to the ledger to update cash/positions.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import NamedTuple, Optional

from engine.costs import CostFn
from engine.entities import Fill, Order

BUY_SIDES = ("BUY_TO_OPEN", "BUY_TO_CLOSE")
SELL_SIDES = ("SELL_TO_OPEN", "SELL_TO_CLOSE")


class Quote(NamedTuple):
    bid: float
    ask: float


def _is_buy(order: Order) -> bool:
    return order.side in BUY_SIDES


def _valid(quote: Optional[Quote]) -> bool:
    if quote is None:
        return False
    return not (math.isnan(quote.bid) or math.isnan(quote.ask))


def process_pending_orders(
    orders: list[Order],
    quotes: dict[str, Quote],
    ts: datetime,
    cost_fn: Optional[CostFn] = None,
) -> list[Fill]:
    """
    Attempt to fill every OPEN order against `quotes` (keyed by `Contract.key`) for the current
    bar. Orders that can't fill this bar (no quote available, or a limit price not yet touched)
    stay OPEN and are re-checked on the next call with the next bar's quotes.

    Mutates matched orders in place (status -> FILLED, fill_price, filled_at) and returns the
    resulting Fills. A MARKET order fills at the touch (bid if selling, ask if buying); a LIMIT
    order fills at the intersection of the touch and its limit price (i.e. no worse than the
    limit). There is no slippage model: fills are perfect executions at the book.

    When `cost_fn` is provided, each produced fill carries a `commission` (transaction cost in
    USD) computed from `cost_fn(side, qty, fill_price)` — the costs module sums this over a
    fill's contracts and applies per-order minimums. Pass cost_fn=None for free execution.
    """
    fills: list[Fill] = []

    for order in orders:
        if order.status != "OPEN":
            continue

        quote = quotes.get(order.contract.key)
        if not _valid(quote):
            continue  # no usable quote this bar — order stays pending

        is_buy = _is_buy(order)

        if order.order_type == "MARKET":
            fill_price = quote.ask if is_buy else quote.bid
        else:  # LIMIT
            if is_buy:
                if quote.ask > order.limit_price:
                    continue  # market's ask is above what we're willing to pay — not marketable
                fill_price = min(quote.ask, order.limit_price)
            else:
                if quote.bid < order.limit_price:
                    continue  # market's bid is below what we're willing to accept — not marketable
                fill_price = max(quote.bid, order.limit_price)

        commission = cost_fn(order.side, order.qty, fill_price) if cost_fn else 0.0

        order.status = "FILLED"
        order.fill_price = fill_price
        order.filled_at = ts

        signed_qty = order.qty if is_buy else -order.qty
        fills.append(
            Fill(
                order_id=order.id,
                contract=order.contract,
                qty=signed_qty,
                price=fill_price,
                timestamp=ts,
                reason="ORDER",
                group_id=order.group_id,
                commission=commission,
            )
        )

    return fills