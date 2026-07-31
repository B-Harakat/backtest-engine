"""
Expiration handling for cash-settled, European-style options (the settlement mechanics used by
index products like XSP, SPX weeklies/0DTE, RUT, NDX, VIX).

Settlement is modeled as a Fill at the contract's intrinsic value, applied through the same
`Portfolio.apply_fill` used for ordinary trades — see ledger.py's docstring for why.

This module only implements the CASH-settlement path. For underlyings whose real-world options
are physically settled (American-style equity/ETF options), this engine never simulates physical
settlement at all — see `engine.runner`'s `_force_close_expiring_positions`. Which path a run uses
is chosen once via `run_backtest`'s `settlement_style` parameter, not per contract (this engine
assumes a single underlying per run).
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

    IMPORTANT: `underlying_settlement_price` must be the actual official settlement print for the
    contract's expiration cadence — NOT just "the last minute bar's close." Settlement timing and
    price source vary by product AND by expiration cadence within the same product (e.g. XSP and
    SPX weeklies/0DTE are PM-settled against the 4:00pm ET close, while SPX standard monthlies are
    AM-settled against a special opening print). Verify against current OCC/Cboe specs; passing
    the wrong price/time silently corrupts expiration P&L. `settlement_time` only controls the
    Fill's recorded timestamp.

    Out-of-the-money contracts settle at zero — they still produce a Fill (reason=
    "EXPIRY_CASH_SETTLEMENT", price=0.0) so the trade log shows the position expiring worthless.
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