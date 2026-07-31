"""
Portfolio ledger: cash balance, open positions (keyed by contract.key), and an append-only fill
log (which doubles as your trade log for later analysis — see reports/report.py).

`apply_fill` is the only mutating entry point, used for BOTH ordinary order fills (from
fill_engine.py) and expiration cash-settlements (from settlement.py) — settlement is modeled as
"a fill at the intrinsic value," reusing the same cost-basis/realized-P&L logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from engine.entities import DEFAULT_OPTION_MULTIPLIER, Contract, Fill, Position


@dataclass
class RealizedPnLEvent:
    timestamp: datetime
    contract: Contract
    qty_closed: int
    pnl: float


class Portfolio:
    def __init__(self, starting_cash: float, multiplier: int = DEFAULT_OPTION_MULTIPLIER):
        self.cash = starting_cash
        self.multiplier = multiplier
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.realized_pnl_log: list[RealizedPnLEvent] = []

    @property
    def realized_pnl(self) -> float:
        return sum(e.pnl for e in self.realized_pnl_log)

    def apply_fill(self, fill: Fill) -> None:
        self.cash -= fill.qty * fill.price * self.multiplier
        self.cash -= fill.commission  # transaction costs hit cash directly (commission + fees)
        self.fills.append(fill)

        existing = self.positions.get(fill.contract.key)

        if existing is None:
            if fill.qty != 0:
                self.positions[fill.contract.key] = Position(
                    contract=fill.contract,
                    qty=fill.qty,
                    avg_price=fill.price,
                    opened_at=fill.timestamp,
                    group_id=fill.group_id,
                )
            return

        adding_to_position = (existing.qty > 0) == (fill.qty > 0)

        if adding_to_position:
            new_qty = existing.qty + fill.qty
            existing.avg_price = (existing.qty * existing.avg_price + fill.qty * fill.price) / new_qty
            existing.qty = new_qty
            return

        # Opposite-direction fill: reduces, fully closes, or flips the position.
        closing_qty = min(abs(existing.qty), abs(fill.qty))
        direction_sign = 1 if existing.qty > 0 else -1
        realized = closing_qty * (fill.price - existing.avg_price) * self.multiplier * direction_sign
        self.realized_pnl_log.append(
            RealizedPnLEvent(timestamp=fill.timestamp, contract=fill.contract, qty_closed=closing_qty, pnl=realized)
        )

        new_qty = existing.qty + fill.qty
        if new_qty == 0:
            del self.positions[fill.contract.key]
        elif (new_qty > 0) == (existing.qty > 0):
            # partial close — direction unchanged, cost basis on the remainder is unchanged
            existing.qty = new_qty
        else:
            # flipped through zero — the excess is a brand-new position at this fill's price
            existing.qty = new_qty
            existing.avg_price = fill.price
            existing.opened_at = fill.timestamp

    def mark_to_market(self, mid_prices: dict[str, float]) -> float:
        """Total account equity: cash + sum of open positions marked at `mid_prices` (keyed by
        `Contract.key`, typically (bid+ask)/2 from the current bar). Positions missing from
        `mid_prices` (e.g. no quote this bar) fall back to their cost basis rather than crashing."""
        equity = self.cash
        for key, pos in self.positions.items():
            price = mid_prices.get(key, pos.avg_price)
            equity += pos.market_value(price, self.multiplier)
        return equity