"""
At 10:30am ET each session, buys 1 contract of the 0DTE (same-day-expiring) call on `underlying`
whose strike is the nearest out-of-the-money strike above the current underlying price. Both the
current price and the real listed strikes come from a single options-chain snapshot
(Strategy.get_chain_snapshot) rather than a separate index feed -- see
engine/thetadata_client.py's fetch_chain_from_thetadata for why. Holds to expiration -- how that
expiration is handled (cash settlement vs. force-closing before physical settlement) is chosen
once for the whole backtest via run_backtest's `settlement_style`, not by this strategy.

Submits a LIMIT order at the chosen contract's current ask (Strategy.submit_order no longer
supports market orders at all -- see its docstring). Priced off the CHAIN's own bid/ask for that
strike rather than a fresh quote lookup on the contract itself, since the contract hasn't been
warmed yet at the moment of submission (that only happens once the order is processed) -- the
chain snapshot we already fetched to pick the strike already carries every strike's bid/ask too,
so no extra data access is needed. Limiting at the ask is just this example's own choice (roughly
matches what a market order would have paid); pick a different price entirely if you want, e.g.
mid, or the ask minus a few cents to trade more passively.
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


class MyStrategy(Strategy):
    def __init__(self, underlying: str = "XSP", quantity: int = 1):
        super().__init__()
        self.underlying = underlying
        self.quantity = quantity
        self._bought_on: Optional[date] = None  # last date we entered a position, so we only buy once per day
        self._warmed_chain_on: Optional[date] = None  # last date we warmed the chain for, so we only do it once/day

    def initialize(self) -> None:
        pass  # the contract to trade isn't known until we see the day's price at 10:30am

    def on_bar(self, ts: datetime) -> None:
        if self._warmed_chain_on != ts.date():
            # First bar of a new session -- warm the whole day's chain right away rather than
            # waiting for the entry time. get_chain_snapshot() always fetches/caches the full
            # session on first touch regardless of when it's called; calling it here just moves
            # WHEN that happens earlier in the day, decoupled from the entry-time check below.
            self.get_chain_snapshot(self.underlying, ts.date())
            self._warmed_chain_on = ts.date()

        if ts.time() != ENTRY_TIME:
            return
        if self._bought_on == ts.date():
            return  # already entered today

        contract, limit_price = self._pick_entry_contract(ts.date())
        if contract is None:
            # No listed 0DTE expiration today, no chain data available yet, or the chosen
            # strike's ask is missing/NaN -- skip the day rather than guess a price or trade a
            # contract that may not actually exist. See _pick_entry_contract's docstring.
            logger.warning("No usable 0DTE OTM call entry found for %s on %s -- skipping entry.", self.underlying, ts.date())
            self._bought_on = ts.date()  # don't retry later bars today; try again tomorrow
            return

        self.submit_order(contract, "BUY_TO_OPEN", self.quantity, limit_price=limit_price)
        self._bought_on = ts.date()

    def _pick_entry_contract(self, expiration: date) -> tuple[Optional[Contract], Optional[float]]:
        """
        Picks the nearest-out-of-the-money-call and its limit price, both from the SAME chain
        snapshot -- the strike list, the underlying price, and every strike's own bid/ask all
        come from one `option_history_greeks_first_order(strike="*")` fetch, so there's no
        separate "get a quote for the contract I just chose" step (which wouldn't work yet
        anyway, since the contract itself isn't warmed until the order is actually processed).

        Deliberately does not fall back to a guessed strike/price if the chain is empty or the
        chosen strike has no usable ask: an empty chain usually means there's no same-day
        expiration listed for that date, and silently substituting a computed value would mean
        trading (or mispricing) a contract that may not actually exist. Returns (None, None) in
        that case; the caller skips the day.
        """
        chain = self.get_chain_snapshot(self.underlying, expiration)
        if chain.empty or "underlying_price" not in chain.columns or "ask" not in chain.columns:
            return None, None

        spot = float(chain["underlying_price"].iloc[0])
        otm_strikes = [s for s in chain.index if s > spot]
        if not otm_strikes:
            return None, None
        strike = min(otm_strikes)

        ask = chain.loc[strike, "ask"]
        if pd.isna(ask):
            return None, None

        contract = Contract(underlying=self.underlying, expiration=expiration, strike=strike, right="CALL")
        return contract, float(ask)