"""
Expiration handling for cash-settled, European-style options (the settlement mechanics used by
index products like XSP, SPX weeklies/0DTE, RUT, NDX, VIX).

Adapted from the pattern in Lumibot's `backtesting_broker.py`
(`settle_expired_option_contract` / `cash_settle_options_contract` / `_intrinsic_value_per_contract`),
stripped of the equity-option branches (assignment/exercise/early-assignment) that don't apply to
a cash-settled index product.

Settlement is modeled as a Fill at the contract's intrinsic value, applied through the same
`Portfolio.apply_fill` used for ordinary trades — see ledger.py's docstring for why.

This module only implements the CASH-settlement path. For underlyings whose real-world options
are physically settled (American-style equity/ETF options), this engine never simulates physical
settlement at all — see `engine.runner`'s `_force_close_expiring_positions`, which force-closes
any such position via an ordinary market order before its own expiration instead. Which path a
backtest run uses is chosen once via `run_backtest`'s `settlement_style` parameter, not per
contract (this engine assumes a single underlying per run — see the architecture guideline).
"""

from __future__ import annotations

from datetime import date, datetime, time

from engine.entities import Contract, Fill
from engine.greeks import DEFAULT_CASH_SETTLEMENT_TIME_ET
from engine.ledger import Portfolio


def intrinsic_value(contract: Contract, underlying_settlement_price: float) -> float:
    if not contract.is_option:
        raise ValueError(f"intrinsic_value called on a non-option contract: {contract}")
    if contract.right == "CALL":
        return max(0.0, underlying_settlement_price - contract.strike)
    return max(0.0, contract.strike - underlying_settlement_price)


def expire_and_settle(
    ledger: Portfolio,
    as_of_date: date,
    underlying_settlement_price: float,
    settlement_time: time = DEFAULT_CASH_SETTLEMENT_TIME_ET,
) -> list[Fill]:
    """
    Cash-settle every open position expiring on `as_of_date` at `underlying_settlement_price`.

    IMPORTANT: `underlying_settlement_price` must be the actual official settlement print for
    the contract's expiration cadence — NOT just "the last minute bar's close." Settlement timing
    and price source vary by product AND by expiration cadence within the same product: XSP and
    SPX's weekly/0DTE expirations are PM-settled against the normal 4:00pm ET close, but SPX's
    *standard monthly* expirations are AM-settled against a special opening print the next
    morning — verify this against current OCC/Cboe specs for whichever underlying and expiration
    cadence you're actually trading; passing the wrong settlement price/time here silently
    corrupts expiration P&L. `settlement_time` (threaded from `run_backtest`'s
    `settlement_time` parameter) only controls the Fill's recorded timestamp — the actual price
    source verification is on you.

    Out-of-the-money contracts settle at zero — they still produce a Fill (reason=
    "EXPIRY_CASH_SETTLEMENT", price=0.0) so the trade log shows the position expiring worthless
    rather than silently vanishing.
    """
    settlement_ts = datetime.combine(as_of_date, settlement_time)
    settlement_fills: list[Fill] = []

    expiring_positions = [
        pos for pos in ledger.positions.values()
        if pos.contract.is_option and pos.contract.expiration == as_of_date
    ]

    for pos in expiring_positions:
        price = intrinsic_value(pos.contract, underlying_settlement_price)
        fill = Fill(
            order_id="EXPIRY",
            contract=pos.contract,
            qty=-pos.qty,  # flatten: opposite sign of the held position
            price=price,
            timestamp=settlement_ts,
            reason="EXPIRY_CASH_SETTLEMENT",
            group_id=pos.group_id,
        )
        ledger.apply_fill(fill)
        settlement_fills.append(fill)

    return settlement_fills