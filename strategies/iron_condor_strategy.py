"""
0DTE short iron condor — written to exercise the engine's multi-leg support across BOTH option
rights simultaneously (a call spread + a put spread in one position), not as a strategy with real
trading edge.

At 10:30am ET, sells the nearest OTM call and buys a further OTM call (`spread_width` listed
strikes further out — the call credit spread / upside wing), AND sells the nearest OTM put and
buys a further OTM put (same `spread_width`, the put credit spread / downside wing) — four
independent single-leg orders, all tagged with one shared `group_id`. Calls and puts are fetched
from two independent chain snapshots (`Strategy.get_chain_snapshot(..., right="call"/"put")`) --
see engine/thetadata_client.py and engine/data_store.py, which cache and warm each side
completely separately.

Both SHORT legs (the ones actually at risk) carry their own independent stop-loss, checked and
managed completely independently of each other and of the two long legs: a rally can stop out the
short call while the short put and both long legs are left alone, a selloff can stop out the short
put while the short call and both long legs are left alone, both can happen on the same day, or
neither may happen at all -- whatever remains open at end of day settles/force-closes through the
normal engine mechanisms exactly as if all four legs were entirely unrelated single-leg positions,
because that's exactly what they are as far as the ledger is concerned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional

import pandas as pd

from engine.entities import Contract
from engine.strategy import Strategy

logger = logging.getLogger(__name__)

ENTRY_TIME = time(10, 30)


@dataclass
class _ShortLeg:
    contract: Contract
    entry_credit: float
    closed: bool = False


class IronCondorStrategy(Strategy):
    def __init__(
        self,
        underlying: str = "XSP",
        quantity: int = 1,
        spread_width: int = 2,  # how many listed strikes separate each short leg from its long wing
        short_leg_stop_multiple: float = 2.0,  # buy back a short leg alone if its cost doubles
    ):
        super().__init__()
        self.underlying = underlying
        self.quantity = quantity
        self.spread_width = spread_width
        self.short_leg_stop_multiple = short_leg_stop_multiple

        self._entered_on: Optional[date] = None
        self._warmed_chains_on: Optional[date] = None

        self._short_call: Optional[_ShortLeg] = None
        self._short_put: Optional[_ShortLeg] = None
        self._group_id: Optional[str] = None

    def initialize(self) -> None:
        pass  # legs aren't known until we see the day's chains at 10:30am

    def on_bar(self, ts: datetime) -> None:
        if self._warmed_chains_on != ts.date():
            # First bar of a new session -- warm both sides' chains right away rather than
            # waiting for the entry time (see strategies/my_strategy.py for why this is separate
            # from the entry-time check below). Calls and puts are independent fetches/caches.
            self.get_chain_snapshot(self.underlying, ts.date(), right="call")
            self.get_chain_snapshot(self.underlying, ts.date(), right="put")
            self._warmed_chains_on = ts.date()

        if ts.time() == ENTRY_TIME and self._entered_on != ts.date():
            self._enter_condor(ts)

        if self._short_call is not None and not self._short_call.closed:
            self._check_short_leg_stop(self._short_call)
        if self._short_put is not None and not self._short_put.closed:
            self._check_short_leg_stop(self._short_put)

    def _enter_condor(self, ts: datetime) -> None:
        expiration = ts.date()
        call_chain = self.get_chain_snapshot(self.underlying, expiration, right="call")
        put_chain = self.get_chain_snapshot(self.underlying, expiration, right="put")

        required_cols = {"underlying_price", "bid", "ask"}
        if call_chain.empty or put_chain.empty or not required_cols.issubset(call_chain.columns) or not required_cols.issubset(put_chain.columns):
            logger.warning("No usable call+put chains for %s on %s -- skipping condor entry.", self.underlying, expiration)
            self._entered_on = expiration
            return

        spot = float(call_chain["underlying_price"].iloc[0])
        otm_calls = sorted(s for s in call_chain.index if s > spot)
        otm_puts = sorted((s for s in put_chain.index if s < spot), reverse=True)  # nearest-to-spot first

        if len(otm_calls) <= self.spread_width or len(otm_puts) <= self.spread_width:
            logger.warning(
                "Not enough OTM strikes for a %d-wide condor on %s %s (calls=%d, puts=%d) -- skipping.",
                self.spread_width, self.underlying, expiration, len(otm_calls), len(otm_puts),
            )
            self._entered_on = expiration
            return

        short_call_strike, long_call_strike = otm_calls[0], otm_calls[self.spread_width]
        short_put_strike, long_put_strike = otm_puts[0], otm_puts[self.spread_width]

        short_call_bid = call_chain.loc[short_call_strike, "bid"]
        long_call_ask = call_chain.loc[long_call_strike, "ask"]
        short_put_bid = put_chain.loc[short_put_strike, "bid"]
        long_put_ask = put_chain.loc[long_put_strike, "ask"]
        if any(pd.isna(p) for p in (short_call_bid, long_call_ask, short_put_bid, long_put_ask)):
            logger.warning("Missing bid/ask for chosen condor legs on %s %s -- skipping.", self.underlying, expiration)
            self._entered_on = expiration
            return

        short_call = Contract(underlying=self.underlying, expiration=expiration, strike=short_call_strike, right="CALL")
        long_call = Contract(underlying=self.underlying, expiration=expiration, strike=long_call_strike, right="CALL")
        short_put = Contract(underlying=self.underlying, expiration=expiration, strike=short_put_strike, right="PUT")
        long_put = Contract(underlying=self.underlying, expiration=expiration, strike=long_put_strike, right="PUT")

        self._group_id = f"{self.underlying}_{expiration:%Y%m%d}_condor"
        self._short_call = _ShortLeg(contract=short_call, entry_credit=float(short_call_bid))
        self._short_put = _ShortLeg(contract=short_put, entry_credit=float(short_put_bid))

        self.submit_order(short_call, "SELL_TO_OPEN", self.quantity, limit_price=float(short_call_bid), group_id=self._group_id)
        self.submit_order(long_call, "BUY_TO_OPEN", self.quantity, limit_price=float(long_call_ask), group_id=self._group_id)
        self.submit_order(short_put, "SELL_TO_OPEN", self.quantity, limit_price=float(short_put_bid), group_id=self._group_id)
        self.submit_order(long_put, "BUY_TO_OPEN", self.quantity, limit_price=float(long_put_ask), group_id=self._group_id)

        self._entered_on = expiration
        logger.info(
            "Entered %s condor: call spread -%s@%.2f/+%s@%.2f, put spread -%s@%.2f/+%s@%.2f",
            self.underlying, short_call.key, short_call_bid, long_call.key, long_call_ask,
            short_put.key, short_put_bid, long_put.key, long_put_ask,
        )

    def _check_short_leg_stop(self, leg: _ShortLeg) -> None:
        """
        Independent leg management -- this is the part of the strategy specifically meant to
        exercise "close one leg of a multi-leg position independently of the others," now with
        two short legs (call and put) that can each be stopped out on their own schedule, driven
        by opposite market moves, without ever touching each other or either long leg.
        """
        quote = self.get_quote(leg.contract)
        if quote is None:
            return  # no data yet this bar -- try again next bar

        stop_price = leg.entry_credit * self.short_leg_stop_multiple
        if quote["ask"] < stop_price:
            return  # not stopped out yet

        self.submit_order(leg.contract, "BUY_TO_CLOSE", self.quantity, limit_price=float(quote["ask"]), group_id=self._group_id)
        leg.closed = True
        logger.info(
            "Short leg %s stopped out at %.2f (>= %.1fx entry credit %.2f) -- other legs left open.",
            leg.contract.key, quote["ask"], self.short_leg_stop_multiple, leg.entry_credit,
        )