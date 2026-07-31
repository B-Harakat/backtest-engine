"""
Core data model for the engine.

Deliberately flat and leg-agnostic: a Position/Order/Fill always refers to exactly one
contract. Multi-leg strategies (spreads, condors, etc.) are built by your strategy code
submitting several independent single-leg orders, optionally tagged with a shared
`group_id` for reporting. The engine itself never needs to know a "spread" exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional
from uuid import uuid4

# Default option contract multiplier ($ per point of intrinsic value per contract) — 100 is the
# standard for most US equity/ETF/index options, but not universal (e.g. "Mini" contracts use
# 10). This engine assumes one multiplier per backtest run, set via run_backtest's `multiplier`
# parameter, not inferred from the underlying — override the default there, not here.
DEFAULT_OPTION_MULTIPLIER = 100

OrderSide = Literal["BUY_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_CLOSE"]
OrderType = Literal["MARKET", "LIMIT"]
OrderStatus = Literal["OPEN", "FILLED", "CANCELLED"]
Right = Literal["CALL", "PUT"]


@dataclass(frozen=True)
class Contract:
    """
    Identifies a single tradeable instrument: either the underlying itself
    (expiration=None, strike=None, right=None) or a specific option contract.
    """

    underlying: str
    expiration: Optional[date] = None
    strike: Optional[float] = None
    right: Optional[Right] = None

    def __post_init__(self) -> None:
        is_option = self.expiration is not None or self.strike is not None or self.right is not None
        if is_option and not (self.expiration and self.strike is not None and self.right):
            raise ValueError(
                f"Contract {self} is missing one of expiration/strike/right — "
                "an option contract needs all three, or none (for the underlying)."
            )

    @property
    def is_option(self) -> bool:
        return self.expiration is not None

    @property
    def key(self) -> str:
        """Canonical string id — used as the Parquet filename stem and dict key everywhere."""
        if not self.is_option:
            return self.underlying
        right_code = "C" if self.right == "CALL" else "P"
        return f"{self.underlying}_{self.expiration:%Y%m%d}_{right_code}{self.strike:g}"

    def __str__(self) -> str:
        return self.key


@dataclass
class Order:
    contract: Contract
    side: OrderSide
    qty: int
    order_type: OrderType = "MARKET"
    limit_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    group_id: Optional[str] = None  # optional tag, e.g. "iron_condor_2026-01-17", for reporting only
    status: OrderStatus = "OPEN"
    fill_price: Optional[float] = None
    filled_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"Order qty must be positive (direction is encoded in `side`), got {self.qty}")
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("LIMIT orders require a limit_price")


@dataclass
class Position:
    contract: Contract
    qty: int  # signed: positive = long, negative = short
    avg_price: float  # per-contract basis, before multiplier
    opened_at: datetime
    group_id: Optional[str] = None

    def market_value(self, mark_price: float, multiplier: int = DEFAULT_OPTION_MULTIPLIER) -> float:
        return self.qty * mark_price * multiplier

    def unrealized_pnl(self, mark_price: float, multiplier: int = DEFAULT_OPTION_MULTIPLIER) -> float:
        return (mark_price - self.avg_price) * self.qty * multiplier


@dataclass
class Fill:
    order_id: str
    contract: Contract
    qty: int  # signed, same convention as Position.qty
    price: float
    timestamp: datetime
    reason: str = "ORDER"  # "ORDER" | "EXPIRY_CASH_SETTLEMENT"
    group_id: Optional[str] = None