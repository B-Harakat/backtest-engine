"""
0DTE call vertical (debit) spread — written specifically to exercise the engine's multi-leg
support, not as a strategy with real trading edge.

At 10:30am ET, buys the nearest out-of-the-money call (the long leg) and sells a further OTM
call (the short leg, `spread_width` listed strikes further out), both on the same expiration,
simultaneously, tagged with a shared `group_id` so the trade log identifies them as one spread —
see engine/entities.py's module docstring: "Multi-leg strategies... submitting several
independent single-leg orders, optionally tagged with a shared group_id."

The two legs are then managed INDEPENDENTLY of each other for the rest of the day: the short leg
carries its own stop-loss (if the cost to buy it back rises past `short_leg_stop_multiple` times
the credit collected for it, it's bought back on its own), while the long leg is left completely
alone regardless of what happens to the short leg. Whichever leg(s) are still open at the end of
the day settle/force-close through the normal engine mechanisms (see run_backtest's
`settlement_style`) exactly as if they were two entirely unrelated single-leg positions — because,
as far as the engine's ledger is concerned, that's exactly what they are.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Optional

import pandas as pd

from engine.entities import Contract
from engine.strategy import Strategy

logger = logging.getLogger(__name__)

ENTRY_TIME = time(10, 30)


class VerticalSpreadStrategy(Strategy):
    def __init__(
        self,
        underlying: str = "XSP",
        quantity: int = 1,
        spread_width: int = 2,  # how many listed strikes further out the short leg is
        short_leg_stop_multiple: float = 2.0,  # buy back the short leg alone if its cost doubles
    ):
        super().__init__()
        self.underlying = underlying
        self.quantity = quantity
        self.spread_width = spread_width
        self.short_leg_stop_multiple = short_leg_stop_multiple

        self._entered_on: Optional[date] = None
        self._warmed_chain_on: Optional[date] = None

        # Per-trading-day state for whichever spread is currently open (reset on each new entry).
        self._long_leg: Optional[Contract] = None
        self._short_leg: Optional[Contract] = None
        self._short_leg_entry_credit: Optional[float] = None
        self._short_leg_closed = False
        self._group_id: Optional[str] = None

    def initialize(self) -> None:
        pass  # legs aren't known until we see the day's chain at 10:30am

    def on_bar(self, ts: datetime) -> None:
        if self._warmed_chain_on != ts.date():
            # First bar of a new session -- warm the whole day's chain right away rather than
            # waiting for the entry time (see strategies/my_strategy.py for why this is separate
            # from the entry-time check below).
            self.get_chain_snapshot(self.underlying, ts.date())
            self._warmed_chain_on = ts.date()

        if ts.time() == ENTRY_TIME and self._entered_on != ts.date():
            self._enter_spread(ts)

        if self._short_leg is not None and not self._short_leg_closed:
            self._check_short_leg_stop(ts)

    def _enter_spread(self, ts: datetime) -> None:
        expiration = ts.date()
        chain = self.get_chain_snapshot(self.underlying, expiration)
        if chain.empty or not {"underlying_price", "ask", "bid"}.issubset(chain.columns):
            logger.warning("No usable chain for %s on %s -- skipping spread entry.", self.underlying, expiration)
            self._entered_on = expiration
            return

        spot = float(chain["underlying_price"].iloc[0])
        otm_strikes = sorted(s for s in chain.index if s > spot)
        if len(otm_strikes) <= self.spread_width:
            logger.warning(
                "Not enough OTM strikes (%d) for a %d-wide spread on %s %s -- skipping.",
                len(otm_strikes), self.spread_width, self.underlying, expiration,
            )
            self._entered_on = expiration
            return

        long_strike = otm_strikes[0]
        short_strike = otm_strikes[self.spread_width]
        long_ask = chain.loc[long_strike, "ask"]
        short_bid = chain.loc[short_strike, "bid"]
        if pd.isna(long_ask) or pd.isna(short_bid):
            logger.warning("Missing bid/ask for chosen spread legs on %s %s -- skipping.", self.underlying, expiration)
            self._entered_on = expiration
            return

        self._long_leg = Contract(underlying=self.underlying, expiration=expiration, strike=long_strike, right="CALL")
        self._short_leg = Contract(underlying=self.underlying, expiration=expiration, strike=short_strike, right="CALL")
        self._short_leg_entry_credit = float(short_bid)
        self._short_leg_closed = False
        self._group_id = f"{self.underlying}_{expiration:%Y%m%d}_vertical"

        self.submit_order(self._long_leg, "BUY_TO_OPEN", self.quantity, limit_price=float(long_ask), group_id=self._group_id)
        self.submit_order(self._short_leg, "SELL_TO_OPEN", self.quantity, limit_price=float(short_bid), group_id=self._group_id)

        self._entered_on = expiration
        logger.info(
            "Entered %s vertical: +%s @ %.2f / -%s @ %.2f",
            self.underlying, self._long_leg.key, long_ask, self._short_leg.key, short_bid,
        )

    def _check_short_leg_stop(self, ts: datetime) -> None:
        """
        Independent leg management -- this is the part of the strategy specifically meant to
        exercise "close one leg of a multi-leg position independently of the other." If the short
        leg's cost to buy back has risen past `short_leg_stop_multiple` times the credit
        collected for it, close JUST that leg; the long leg is left completely alone and keeps
        riding toward its own settlement regardless of what happens here.
        """
        quote = self.get_quote(self._short_leg)
        if quote is None:
            return  # no data yet this bar -- try again next bar

        stop_price = self._short_leg_entry_credit * self.short_leg_stop_multiple
        if quote["ask"] < stop_price:
            return  # not stopped out yet

        self.submit_order(
            self._short_leg, "BUY_TO_CLOSE", self.quantity,
            limit_price=float(quote["ask"]), group_id=self._group_id,
        )
        self._short_leg_closed = True
        logger.info(
            "Short leg %s stopped out at %.2f (>= %.1fx entry credit %.2f) -- long leg %s left open.",
            self._short_leg.key, quote["ask"], self.short_leg_stop_multiple, self._short_leg_entry_credit, self._long_leg.key,
        )