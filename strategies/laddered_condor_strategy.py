"""
Laddered, delta-targeted, rolling iron condor -- a deliberately "kitchen sink" strategy meant to
stress several engine capabilities AT ONCE, not to have real trading edge. Where the other
example strategies each isolated one capability, this one combines:

  - MULTI-DAY, MULTI-EXPIRATION positions held concurrently: enters a new iron condor every
    trading day, each one targeting an expiration `days_to_expiration` trading days out. At
    steady state there are multiple condors on DIFFERENT expirations open at once (a genuine
    ladder) -- every other strategy here was 0DTE (entry day == expiration day); this is the
    first to actually hold a position across multiple sessions. This is exactly the scenario that
    exposed a real multi-day lazy-warm bug in engine/runner.py (see that file's history) --
    an order submitted today for a contract expiring days later had literally no quote to fill
    against until its expiration day arrived. Fixed there; exercised here.
  - DELTA-TARGETED strike selection: strikes are chosen by nearest delta to a target (using the
    chain's own `delta` column), not by counting strikes away from the money.
  - DYNAMIC position sizing: quantity scales with current cash via self.cash(), not a fixed
    constant.
  - ROLLING a threatened short leg (not just closing it): uses Strategy.get_bars() to compute a
    trailing reference price from the leg's OWN recent history, and if breached, closes that
    strike and opens a new, further-out-of-the-money one on the SAME expiration and group_id, in
    the same bar -- a close-then-reopen pattern, not just a close.
  - PORTFOLIO-LEVEL risk gating: inspects self.positions() before entering a new condor and
    skips the entry entirely if too many condors are already open concurrently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Optional

import pandas as pd

from engine.calendar import trading_days
from engine.entities import Contract
from engine.strategy import Strategy

logger = logging.getLogger(__name__)

ENTRY_TIME = time(10, 30)


@dataclass
class _RollableLeg:
    contract: Contract
    entry_credit: float
    roll_count: int = 0
    active: bool = True  # False once rolling has given up (no further strike to roll into) --
                         # without this, the leg keeps getting "managed" against a contract we no
                         # longer hold a position in, resubmitting BUY_TO_CLOSE against it forever


@dataclass
class _ActiveCondor:
    expiration: date
    group_id: str
    quantity: int
    short_call: _RollableLeg
    long_call: Contract
    short_put: _RollableLeg
    long_put: Contract
    ever_opened: bool = False  # guards against treating a not-yet-filled entry as "already closed"


class LadderedCondorStrategy(Strategy):
    def __init__(
        self,
        underlying: str = "XSP",
        days_to_expiration: int = 2,          # each day's new condor targets this many trading days out
        target_short_delta: float = 0.20,     # |delta| target for short legs (call: +0.20, put: -0.20)
        target_long_delta: float = 0.08,      # |delta| target for the protective long wings
        risk_pct_per_trade: float = 0.02,     # fraction of current cash risked per new condor
        max_concurrent_condors: int = 3,      # portfolio-level cap on simultaneously open condors
        roll_lookback_bars: int = 60,         # trailing window (in bars) used for the roll trigger
        roll_trigger_multiple: float = 1.5,   # roll if current ask > trigger_multiple * trailing min ask
        assumed_multiplier: int = 100,        # only used for this strategy's OWN sizing heuristic --
                                               # Strategy doesn't expose the real configured multiplier,
                                               # so this is an approximation; keep it in sync with
                                               # run_backtest's `multiplier` if you change either.
    ):
        super().__init__()
        self.underlying = underlying
        self.days_to_expiration = days_to_expiration
        self.target_short_delta = target_short_delta
        self.target_long_delta = target_long_delta
        self.risk_pct_per_trade = risk_pct_per_trade
        self.max_concurrent_condors = max_concurrent_condors
        self.roll_lookback_bars = roll_lookback_bars
        self.roll_trigger_multiple = roll_trigger_multiple
        self.assumed_multiplier = assumed_multiplier

        self._entered_on: Optional[date] = None
        self._warmed_chains_on: Optional[date] = None
        self._active_condors: list[_ActiveCondor] = []

    def initialize(self) -> None:
        pass

    def on_bar(self, ts: datetime) -> None:
        if self._warmed_chains_on != ts.date():
            self.get_chain_snapshot(self.underlying, ts.date(), right="call")
            self.get_chain_snapshot(self.underlying, ts.date(), right="put")
            self._warmed_chains_on = ts.date()

        # Clean up condors that have FULLY settled/closed, BEFORE this bar's new-entry check --
        # a condor entered on this exact bar has orders that haven't been processed/filled yet
        # (that happens after on_bar returns), so checking cleanup first means a freshly-entered
        # condor is never in this list yet at all, and one entered on an earlier bar has already
        # had at least one full bar cycle to actually fill.
        self._active_condors = [c for c in self._active_condors if not self._is_condor_fully_closed(c)]

        if ts.time() == ENTRY_TIME and self._entered_on != ts.date():
            self._maybe_enter_new_condor(ts)

        for condor in self._active_condors:
            self._manage_leg(condor, condor.short_call, "CALL")
            self._manage_leg(condor, condor.short_put, "PUT")

    def _is_condor_fully_closed(self, condor: _ActiveCondor) -> bool:
        """
        True once this condor's current legs (which may have changed strike due to rolling --
        see _manage_leg) were confirmed open at some point and are now ALL closed, whether
        because they settled at expiration or were closed/rolled away entirely.

        Deliberately distinguishes "not yet filled" from "filled and now closed" via
        `ever_opened`: without it, a condor whose entry orders simply haven't been processed yet
        looks identical to one that's fully closed (zero open positions either way) and would be
        purged from tracking before it ever had a chance to actually open. Without THIS cleanup
        existing at all, `_active_condors` would only ever grow, and `max_concurrent_condors`
        would trip once and never release -- both are real bugs this strategy hit during testing.
        """
        open_keys = {p.contract.key for p in self.positions()}
        condor_keys = {condor.short_call.contract.key, condor.long_call.key, condor.short_put.contract.key, condor.long_put.key}
        currently_open = not condor_keys.isdisjoint(open_keys)
        if currently_open:
            condor.ever_opened = True
            return False
        return condor.ever_opened

    # --- entry -----------------------------------------------------------------------------

    def _maybe_enter_new_condor(self, ts: datetime) -> None:
        self._entered_on = ts.date()  # only ever attempt once per day, success or not

        if len(self._active_condors) >= self.max_concurrent_condors:
            logger.info(
                "Skipping new condor on %s -- already at max_concurrent_condors=%d.",
                ts.date(), self.max_concurrent_condors,
            )
            return

        expiration = self._nth_trading_day_after(ts.date(), self.days_to_expiration)

        call_chain = self.get_chain_snapshot(self.underlying, expiration, right="call")
        put_chain = self.get_chain_snapshot(self.underlying, expiration, right="put")
        required = {"underlying_price", "bid", "ask", "delta"}
        if call_chain.empty or put_chain.empty or not required.issubset(call_chain.columns) or not required.issubset(put_chain.columns):
            logger.warning("No usable chains for %s expiring %s -- skipping condor entry.", self.underlying, expiration)
            return

        short_call_strike = self._pick_strike_by_delta(call_chain, self.target_short_delta)
        long_call_strike = self._pick_strike_by_delta(call_chain, self.target_long_delta, min_strike=short_call_strike)
        short_put_strike = self._pick_strike_by_delta(put_chain, -self.target_short_delta)
        long_put_strike = self._pick_strike_by_delta(put_chain, -self.target_long_delta, max_strike=short_put_strike)

        if None in (short_call_strike, long_call_strike, short_put_strike, long_put_strike):
            logger.warning("Could not find delta-targeted strikes for %s expiring %s -- skipping.", self.underlying, expiration)
            return

        short_call_bid = call_chain.loc[short_call_strike, "bid"]
        long_call_ask = call_chain.loc[long_call_strike, "ask"]
        short_put_bid = put_chain.loc[short_put_strike, "bid"]
        long_put_ask = put_chain.loc[long_put_strike, "ask"]
        if any(pd.isna(p) for p in (short_call_bid, long_call_ask, short_put_bid, long_put_ask)):
            logger.warning("Missing bid/ask for delta-targeted legs on %s expiring %s -- skipping.", self.underlying, expiration)
            return

        max_width = max(long_call_strike - short_call_strike, short_put_strike - long_put_strike)
        quantity = self._position_size(max_width)

        short_call = Contract(underlying=self.underlying, expiration=expiration, strike=short_call_strike, right="CALL")
        long_call = Contract(underlying=self.underlying, expiration=expiration, strike=long_call_strike, right="CALL")
        short_put = Contract(underlying=self.underlying, expiration=expiration, strike=short_put_strike, right="PUT")
        long_put = Contract(underlying=self.underlying, expiration=expiration, strike=long_put_strike, right="PUT")
        group_id = f"{self.underlying}_{expiration:%Y%m%d}_ladder{len(self._active_condors)}"

        self.submit_order(short_call, "SELL_TO_OPEN", quantity, limit_price=float(short_call_bid), group_id=group_id)
        self.submit_order(long_call, "BUY_TO_OPEN", quantity, limit_price=float(long_call_ask), group_id=group_id)
        self.submit_order(short_put, "SELL_TO_OPEN", quantity, limit_price=float(short_put_bid), group_id=group_id)
        self.submit_order(long_put, "BUY_TO_OPEN", quantity, limit_price=float(long_put_ask), group_id=group_id)

        self._active_condors.append(
            _ActiveCondor(
                expiration=expiration,
                group_id=group_id,
                quantity=quantity,
                short_call=_RollableLeg(contract=short_call, entry_credit=float(short_call_bid)),
                long_call=long_call,
                short_put=_RollableLeg(contract=short_put, entry_credit=float(short_put_bid)),
                long_put=long_put,
            )
        )
        logger.info(
            "Entered condor #%d (%s) expiring %s, qty=%d: -%s/+%s (calls), -%s/+%s (puts)",
            len(self._active_condors), group_id, expiration, quantity,
            short_call.key, long_call.key, short_put.key, long_put.key,
        )

    def _position_size(self, spread_width_points: float) -> int:
        """Rough, deliberately simple dynamic sizing: risk a fixed fraction of current cash
        against this condor's worst-case width. `assumed_multiplier` is an approximation since
        Strategy doesn't expose the real configured multiplier -- see __init__'s docstring."""
        risk_budget = self.cash() * self.risk_pct_per_trade
        max_loss_per_contract = max(spread_width_points, 0.01) * self.assumed_multiplier
        return max(1, int(risk_budget / max_loss_per_contract))

    def _nth_trading_day_after(self, today: date, n: int) -> date:
        window_end = today + timedelta(days=n * 3 + 7)  # generous padding for weekends/holidays
        days = [d.date() for d in trading_days(today, window_end) if d.date() > today]
        return days[n - 1]

    def _pick_strike_by_delta(
        self, chain: pd.DataFrame, target_delta: float, min_strike: Optional[float] = None, max_strike: Optional[float] = None
    ) -> Optional[float]:
        """Strike whose `delta` is closest to `target_delta`, optionally restricted to strikes
        strictly above `min_strike` (call wings) or strictly below `max_strike` (put wings) so a
        long wing can never land on or inside its own short strike."""
        candidates = chain
        if min_strike is not None:
            candidates = candidates.loc[candidates.index > min_strike]
        if max_strike is not None:
            candidates = candidates.loc[candidates.index < max_strike]
        candidates = candidates.dropna(subset=["delta"])
        if candidates.empty:
            return None
        return (candidates["delta"] - target_delta).abs().idxmin()

    # --- ongoing management -----------------------------------------------------------------

    def _manage_leg(self, condor: _ActiveCondor, leg: _RollableLeg, right: str) -> None:
        """
        Independent per-leg, per-condor management: uses this leg's OWN recent bar history
        (Strategy.get_bars) as a trailing reference rather than a fixed entry-credit multiple --
        if the current ask has risen too far above the trailing minimum ask, ROLL (close this
        strike, open a new further-out-of-the-money one on the same expiration/group) instead of
        just closing. Completely independent of every other leg, in this condor or any other.
        """
        if not leg.active:
            return  # rolling already gave up on this leg (see below) -- nothing left to manage

        bars = self.get_bars(leg.contract, lookback=self.roll_lookback_bars)
        if bars.empty or "ask" not in bars.columns:
            return
        trailing_min_ask = bars["ask"].min()
        if pd.isna(trailing_min_ask):
            return

        quote = self.get_quote(leg.contract)
        if quote is None:
            return

        if quote["ask"] <= trailing_min_ask * self.roll_trigger_multiple:
            return  # not triggered

        chain = self.get_chain_snapshot(self.underlying, condor.expiration, right="call" if right == "CALL" else "put")
        if chain.empty or "delta" not in chain.columns:
            new_strike = None
        else:
            new_strike = self._pick_strike_by_delta(
                chain, self.target_short_delta if right == "CALL" else -self.target_short_delta,
                min_strike=leg.contract.strike if right == "CALL" else None,
                max_strike=leg.contract.strike if right == "PUT" else None,
            )

        self.submit_order(leg.contract, "BUY_TO_CLOSE", condor.quantity, limit_price=float(quote["ask"]), group_id=condor.group_id)
        logger.info(
            "Rolling %s leg %s: closed at %.2f (>= %.1fx trailing min %.2f).",
            right, leg.contract.key, quote["ask"], self.roll_trigger_multiple, trailing_min_ask,
        )

        if new_strike is None:
            logger.warning("No further-OTM strike available to roll %s into -- leg closed, not replaced.", right)
            leg.active = False  # stop managing this leg -- its contract reference is now stale
            return

        new_row = chain.loc[new_strike]
        if pd.isna(new_row.get("bid")):
            logger.warning("New roll strike %s has no usable bid -- leg closed, not replaced.", new_strike)
            leg.active = False  # same reason -- don't keep re-checking a stale contract reference
            return

        new_contract = Contract(underlying=self.underlying, expiration=condor.expiration, strike=new_strike, right=right)
        self.submit_order(new_contract, "SELL_TO_OPEN", condor.quantity, limit_price=float(new_row["bid"]), group_id=condor.group_id)

        leg.contract = new_contract
        leg.entry_credit = float(new_row["bid"])
        leg.roll_count += 1
        logger.info("Rolled %s leg to new strike %s @ %.2f (roll #%d).", right, new_contract.key, new_row["bid"], leg.roll_count)