"""
0DTE short iron condor — written to exercise the engine's multi-leg support across BOTH option
rights simultaneously (a call credit spread + a put credit spread in one position), not as a
strategy with real trading edge. run_mc-compatible: 0DTE (entry day == expiration day), with a
module-level ``GRID`` for parameter exploration.

Each session at ``entry_time`` ET, sells the nearest OTM call and buys a further OTM call
(`spread_width` listed strikes further out — the upside call wing), AND sells the nearest OTM
put and buys a further OTM put (same `spread_width` — the downside put wing). Four independent
single-leg orders, all tagged with one shared ``group_id``. Calls and puts come from two
independent chain snapshots (`Strategy.get_chain_snapshot(..., right="call"/"put")`), which are
cached/warmed separately.

Both SHORT legs (the actually-at-risk ones) carry their own independent stop: a short leg is
bought back on its own if its current ask rises past `short_leg_stop_multiple` x its entry credit.
Each short leg is managed independently of the other and of the two long wings. Whatever remains
open at end of day settles/force-closes through the normal engine mechanisms (as independent
single-leg positions, which is exactly what they are to the ledger).
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

# Monte Carlo grid (consumed by simulation.monte_carlo.expand_grid / scripts/run_mc.py via the
# --strategy spec). Sparse: only the params listed here are varied across the Cartesian product;
# any other constructor arg keeps its own default (partial override). `entry_time` is a
# constructor param, see __init__.
GRID = {
    "spread_width": [2, 3],
    "short_leg_stop_multiple": [1.5, 3.0],
    "entry_time": [time(10, 0), time(11, 0)],
}


@dataclass
class _ShortLeg:
    contract: Contract
    entry_credit: float
    closed: bool = False


class IronCondorStrategy(Strategy):
    """4-leg 0DTE short iron condor: short call + long call (call wing) and short put + long put
    (put wing), each short leg independently stop-managed."""

    def __init__(
        self,
        underlying: str = "XSP",
        quantity: int = 1,
        spread_width: int = 2,  # listed strikes between each short leg and its long wing
        short_leg_stop_multiple: float = 2.0,  # buy back a short leg alone if its cost exceeds this multiple of its credit
        entry_time: time = ENTRY_TIME,  # intraday time (ET) to enter the condor each session
    ):
        super().__init__()
        self.underlying = underlying
        self.quantity = quantity
        self.spread_width = spread_width
        self.short_leg_stop_multiple = short_leg_stop_multiple
        self.entry_time = entry_time

        self._entered_on: Optional[date] = None
        self._warmed_chains_on: Optional[date] = None
        self._short_call: Optional[_ShortLeg] = None
        self._short_put: Optional[_ShortLeg] = None
        self._group_id: Optional[str] = None

    def initialize(self) -> None:
        pass  # legs aren't known until we see the day's chains at the entry time

    def on_bar(self, ts: datetime) -> None:
        if self._warmed_chains_on != ts.date():
            # Warm both sides' chains on the first bar of each session (calls and puts are
            # independent fetches/caches), separate from the entry-time check below.
            self.get_chain_snapshot(self.underlying, ts.date(), right="call")
            self.get_chain_snapshot(self.underlying, ts.date(), right="put")
            self._warmed_chains_on = ts.date()

        if ts.time() == self.entry_time and self._entered_on != ts.date():
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
        if (call_chain.empty or put_chain.empty
                or not required_cols.issubset(call_chain.columns)
                or not required_cols.issubset(put_chain.columns)):
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
        """Close one short leg alone (buy-to-close at its current ask) if its cost has risen past
        `short_leg_stop_multiple` x the credit it collected at entry, leaving the other legs (and
        both long wings) untouched. Exercises the engine's independent multi-leg close."""
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
