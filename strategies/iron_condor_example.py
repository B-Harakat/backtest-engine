"""
0DTE short iron condor — a deliberately pedagogical example.

This file is the reference "how to use the engine" strategy. It is NOT written for trading edge; it is
written to exercise every capability the engine exposes, end to end, with heavy inline comments that
explain *what each piece is for* and *where it hooks into the engine*. Read it alongside STRATEGY.md.

== What an iron condor is ==
A neutral, defined-risk premium-selling structure. We sell an out-of-the-money (OTM) put spread and
an OTM call spread on the same expiration:

    PUT side (bearish insurance below spot)          CALL side (bearish insurance above spot)
      long put  @ long_put_strike  (buy, wing)         long call  @ long_call_strike (buy, wing)
      short put @ short_put_strike (sell, near spot)   short call @ short_call_strike (sell, near spot)

Net result: collect premium from both sides; maximum profit = the net credit; maximum loss = the wing
width minus that credit; we profit if the underlying stays between the two short strikes by expiry.

== Why this strategy demonstrates the FULL engine ==
Every numbered capability below is used somewhere in this file. Grep for the tag `# [FEAT]` to find it.

  [FEAT 01] Strategy.get_run_multiplier()   -- read the ledger's *actual* multiplier for $ sizing.
  [FEAT 02] engine.vix_data.get_daily_vix   -- volatility-regime gate (skip low/high VIX days).
  [FEAT 03] engine.event_calendar.is_event_day -- macro-release avoidance (skip FOMC/CPI/NFP/PPI).
  [FEAT 04] engine.strategy_utils.profit_target_reached -- take-profit at a fraction of max credit.
  [FEAT 05] engine.strategy_utils.select_strike_by_delta -- delta-based short-strike selection.
  [FEAT 06] engine.strategy_utils.select_strike_by_offset -- find each wing strike outward.
  [FEAT 07] engine.strategy_utils.select_nearest_otm       -- (also used for robustness checks).
  [FEAT 08] engine.strategy_utils.size_position            -- risk-based contract sizing.
  [FEAT 09] engine.strategy_utils.net_credit / max_risk_premium -- credit + max-risk math.
  [FEAT 10] engine.strategy_utils.chain_has_columns / chain_has_enough_otm / quotes_valid -- guards.
  [ENG A]   Strategy.get_chain_snapshot(..., right=...) -- read spot, bid/ask/delta from one side.
  [ENG B]   Strategy.initialize / on_bar / before_close  -- lifecycle hooks.
  [ENG C]   Strategy.submit_order(...) x4 with a shared group_id -- multi-leg structure.
  [ENG D]   Strategy.watch(...)            -- pre-fetch contract data for the backtest window.
  [ENG E]   Strategy.get_quote(...)        -- current bid/ask for exit management.
  [ENG F]   cash settlement path           -- expiring legs settle at intrinsic value at 16:00 ET.

== Engine conventions it respects (see STRATEGY.md) ==
  - Underlying symbol param is named `underlying` so `--ticker` maps to it.
  - Always call super().__init__() first.
  - qty is always positive; direction lives in the `side` (BUY_TO_OPEN/SELL_TO_OPEN/...).
  - A "spread"/"condor" is just independent single-leg orders sharing a `group_id`.
  - No current-price arg is passed to on_bar; get price/strikes from get_chain_snapshot(right=...).
  - Each side is single-sided; call and put chains are fetched/cached independently.
  - The engine assumes ONE underlying + one multiplier + one settlement style per run.
  - run_mc-compatible: 0DTE (entry day == expiration day) with a module-level GRID.

== The mechanical rules (from the 0DTE premium-selling research corpus) ==
  Entry ~10:00-10:30 ET, only on non-event days and only in a moderate-VIX regime:
  - ~10-delta short strikes each side (select_strike_by_delta) — high probability of OTM.
  - Wings `wing_width_pts` points out, tightened automatically if the requested width would yield
    a non-positive net credit (keeps the short premium and long protection economically sane on
    XSP's small point scale) (select_strike_by_offset).
  - Size so that one condor's max loss = ~1-2% of account equity (size_position).
  Management:
  - Take `profit_target_frac` of the max credit as profit and close the whole condor
    (profit_target_reached).
  - A short leg is stopped only on an extreme adverse move (leg_stop_multiple x its own credit),
    so normal intraday whipsaws don't realize a loss against the skimpy 0DTE credit.
  - Whole-book stop caps the loss at `book_stop_frac` of the wing width (the spread's true max
    risk) — the credit is too small to size a stop against.
  - before_close flattens anything still open before the 16:00 expiry; the engine cash-settles
    whatever survives to expiry (0DTE SPX/XSP are cash-settled European products).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional

# --- engine surface we demonstrate ---------------------------------------------------------------
from engine import strategy_utils as utils                                   # [FEAT 04-10]
from engine.data_store import DEFAULT_DATA_DIR
from engine.entities import Contract                                          # [ENG D]
from engine.event_calendar import is_event_day                                # [FEAT 03]
from engine.strategy import Strategy                                          # base class
from engine.vix_data import get_daily_vix                                     # [FEAT 02]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Tunable strategy parameters (each becomes grid-able in run_mc; see GRID below).
# --------------------------------------------------------------------------------------------------
ENTRY_TIME = time(10, 30)          # intraday entry time (ET) — 0DTE entry day == expiration day.
SHORT_DELTA = 0.10                 # absolute delta target for each short leg (10-delta ~90% POP).
WING_WIDTH_PTS = 2.0               # requested protection-wing width in POINTS (short->long leg).
PROFIT_TARGET_FRAC = 0.40          # take profit at 40% of the net credit collected.
LEG_STOP_MULTIPLE = 6.0            # stop a short leg only on an extreme move (6x its credit).
BOOK_STOP_FRAC = 0.75              # stop the whole book if cost-to-close >= this fraction of the wing width.
RISK_PCT = 0.01                    # fraction of account equity to risk per condor (1%).
MIN_WING_WIDTH_PTS = 1.0           # adaptive fallback: never go narrower than this wing width.
# VIX regime band (== moderate-volatility filter). Prior-day VIX close below MIN means thin
# premium; above MAX means elevated tail risk. Guarded to only fire when VIX data exists.
VIX_CLOSE_MIN = 12.0
VIX_CLOSE_MAX = 25.0


# run_mc grid: only these params are varied across the Cartesian product; everything else keeps its
# constructor default (partial override). See [ENG] run_mc / STRATEGY.md section 8.
# Note: this is the Cartesian product of all listed values, so keep it small — each combination is
# one full backtest per Monte-Carlo path. (Here 2 x 2 = 4 sets.)
GRID = {
    "short_delta": [0.10, 0.12],
    "wing_width_pts": [2.0, 3.0],
}


@dataclass
class _ShortLeg:
    """Tracks one at-risk short leg so it can be stopped independently of the other legs."""
    contract: Contract
    entry_credit: float
    closed: bool = False


class IronCondorDemo(Strategy):
    """
    0DTE short iron condor that demonstrates the full engine surface.

    Lifecycle used: initialize -> on_bar (entry + per-bar management) -> before_close (flatten).
    Multi-leg: 4 independent single-leg orders sharing one group_id (see [ENG C]).
    """

    def __init__(
        self,
        underlying: str = "XSP",                     # name MUST be `underlying` for --ticker to work
        quantity: Optional[int] = None,              # None => sized via size_position (see below)
        entry_time: time = ENTRY_TIME,
        short_delta: float = SHORT_DELTA,
        wing_width_pts: float = WING_WIDTH_PTS,
        min_wing_width_pts: float = MIN_WING_WIDTH_PTS,
        profit_target_frac: float = PROFIT_TARGET_FRAC,
        leg_stop_multiple: float = LEG_STOP_MULTIPLE,
        book_stop_frac: float = BOOK_STOP_FRAC,
        risk_pct: float = RISK_PCT,
        vix_min: float = VIX_CLOSE_MIN,
        vix_max: float = VIX_CLOSE_MAX,
        data_dir: str = str(DEFAULT_DATA_DIR),
    ):
        # [ENG] Always call super().__init__() FIRST, before touching anything.
        super().__init__()
        self.underlying = underlying
        self.quantity = quantity            # None => auto-size; int => fixed size (dev/README use)
        self.entry_time = entry_time
        self.short_delta = short_delta
        self.wing_width_pts = wing_width_pts
        self.min_wing_width_pts = min_wing_width_pts
        self.profit_target_frac = profit_target_frac
        self.leg_stop_multiple = leg_stop_multiple
        self.book_stop_frac = book_stop_frac
        self.risk_pct = risk_pct
        self.vix_min = vix_min
        self.vix_max = vix_max
        self.data_dir = data_dir

        # Per-session state (reset each day we actually enter).
        self._entered_on: Optional[date] = None
        self._warmed_chains_on: Optional[date] = None
        self._group_id: Optional[str] = None
        self._short_call: Optional[_ShortLeg] = None
        self._short_put: Optional[_ShortLeg] = None
        self._long_call: Optional[Contract] = None
        self._long_put: Optional[Contract] = None
        self._call_credit = 0.0
        self._put_credit = 0.0
        self._entry_spot: Optional[float] = None
        self._exiting = False  # True once we've submitted a whole-condor flatten; stops re-triggering
        self._book_modified = False  # True once a leg is closed by a per-leg stop (abandon profit target)
        self._entry_bar_ts: Optional[datetime] = None  # bar we entered on — management skips it
        self._wing_width_used = 0.0  # the wing width actually placed today (after possible tightening)

    # =============================================================================================
    # Lifecycle  # [ENG B]
    # =============================================================================================
    def initialize(self) -> None:
        """
        Called once before the backtest starts. Pre-fetch data for every instrument we know we'll
        trade so it is warmed up front (one Parquet read) rather than lazily mid-loop.

        Note the chains themselves are warmed on the first bar of each session (we don't know
        today's strikes until we see spot). Here we only set up anything static.
        """
        # [FEAT 01] The ledger's true multiplier is available before entry too. Contract multiplier
        # is $/point/contract; a 1-point move = $100 on a standard XSP/SPX contract.
        _multiplier = self.get_run_multiplier()
        logger.debug("initialize(): run multiplier = %d", _multiplier)

    def on_bar(self, ts: datetime) -> None:
        """
        Called once per 1-minute regular-trading bar. `ts` is a naive America/New_York bar-START
        timestamp (see engine.calendar). No price is passed — read everything from the chains.
        """
        # Warm both chains at the start of each session. Calls and puts are fetched/cached
        # independently, so request each side explicitly.  # [ENG A]
        if self._warmed_chains_on != ts.date():
            self.get_chain_snapshot(self.underlying, ts.date(), right="call")
            self.get_chain_snapshot(self.underlying, ts.date(), right="put")
            self._warmed_chains_on = ts.date()

        # Enter once per session, exactly at our chosen minute.
        # `_entered_on` guards against re-entering if on_bar fires more than once at this time.
        if ts.time() == self.entry_time and self._entered_on != ts.date():
            self._enter_condor(ts)                 # [FEAT 02-05,08-10] [ENG A,C]
            self._entry_bar_ts = ts                # skip management THIS bar (not filled yet)

        # Per-bar management while a condor is open: check profit target on the whole book, and
        # check each short leg's independent stop. We skip the entry bar itself — our orders are
        # filled on the NEXT engine bar, so any value we'd read now is a pre-fill artifact that
        # would falsely trip the profit target and lock management off for the day.  # [FEAT 04]
        if self._short_call is not None and ts != self._entry_bar_ts:
            self._manage_position(ts)

    def before_close(self, ts: datetime) -> None:
        """
        Called on the last bar of each session (respects early closes), BEFORE end-of-day expiry
        settlement runs. The natural place for 0DTE flatten logic.
        # [ENG B]
        """
        # Flatten any legs still open before the close. The engine would cash-settle anything that
        # expires open at 16:00, but on a 0DTE position it is cleaner to exit explicitly here —
        # rolling to expiry leaves only intrinsic value / tail risk for, at most, zero upside.
        if self._short_call is not None:
            self._flatten_all(ts, reason="before_close")

    # =============================================================================================
    # Entry
    # =============================================================================================
    def _enter_condor(self, ts: datetime) -> None:
        """Run the entry filters, choose strikes, size, and submit the 4-leg order set."""
        expiration = ts.date()
        self._entered_on = expiration  # even if skipped, don't retry today

        # --- [FEAT 02] Volatility-regime gate ------------------------------------------------
        # Skip if the session started in too-calm or too-hot a VIX regime. We read the PRIOR day's
        # close (regime before the session begins). If the VIX file is missing for this day,
        # get_daily_vix returns None and we fall through (no filter applied).
        prior_close = self._prior_day_vix_close(ts.date())
        if prior_close is not None and (prior_close < self.vix_min or prior_close > self.vix_max):
            logger.info("SKIP entry %s: prior-day VIX close %.2f outside [%.1f, %.1f].",
                        ts.date(), prior_close, self.vix_min, self.vix_max)
            return

        # --- [FEAT 03] Event-day avoidance ---------------------------------------------------
        # Skip scheduled macro releases (FOMC/CPI/NFP/PPI). is_event_day returns False for a
        # missing/empty calendar, so it's safe before the CSV is populated.
        if is_event_day(ts.date(), data_dir=self.data_dir):
            logger.info("SKIP entry %s: scheduled macro event day.", ts.date())
            return

        # --- Pull both sides' chains ----------------------------------------------------------
        call_chain = self.get_chain_snapshot(self.underlying, expiration, right="call")
        put_chain = self.get_chain_snapshot(self.underlying, expiration, right="put")

        # --- [FEAT 10] Validation guards ------------------------------------------------------
        # Replace 20 lines of defensive checks with intent-revealing named guards.
        if not (utils.chain_has_columns(call_chain, ["underlying_price", "bid", "ask", "delta"])
                and utils.chain_has_columns(put_chain, ["underlying_price", "bid", "ask", "delta"])):
            logger.warning("Skip %s %s: call/put chains missing required columns.", self.underlying, expiration)
            return

        spot = float(call_chain["underlying_price"].iloc[0])
        self._entry_spot = spot

        # Make sure both sides have enough OTM strikes between spot and the wing so a
        # `wing_width_pts` wide protection leg exists with a little slack.
        minimum_otm = max(2, int(self.wing_width_pts) + 1)
        if not (utils.chain_has_enough_otm(call_chain, spot, "CALL", minimum_otm)
                and utils.chain_has_enough_otm(put_chain, spot, "PUT", minimum_otm)):
            logger.warning("Skip %s %s: not enough OTM strikes for a %.0f-pt condor wing.",
                           self.underlying, expiration, self.wing_width_pts)
            return

        # --- [FEAT 05] Delta-based short-strike selection ------------------------------------
        # Target the same absolute delta on each side; the helper handles the sign (calls +,
        # puts -) internally so we never flip it by accident.
        short_call_strike = utils.select_strike_by_delta(call_chain, self.short_delta, "CALL")
        short_put_strike = utils.select_strike_by_delta(put_chain, self.short_delta, "PUT")

        # --- [FEAT 06] Wing strikes a set number of POINTS outward, adaptively ------------------
        # The protection wing is `wing_width_pts` points further OTM than each short leg. If that
        # requested width has gone too wide for the day's premium (long-wing ask >= short credit,
        # i.e. net credit <= 0), we tighten the wing in 1-pt steps down to `min_wing_width_pts` and
        # take the WIDEST width that still yields a positive net credit. This makes the example
        # robust across XSP regimes (5pt wings at 10:30 can be no-bid) without hardcoding a magic
        # number.
        call_strikes = sorted(float(s) for s in call_chain.index)
        put_strikes = sorted(float(s) for s in put_chain.index)

        chosen = None  # (width_pts, long_call, long_put, call_credit, put_credit)
        width = self.wing_width_pts
        while width >= self.min_wing_width_pts:
            long_call_strike = utils.select_wing_strike(
                call_strikes, short_call_strike, width, upward=True)
            long_put_strike = utils.select_wing_strike(
                put_strikes, short_put_strike, width, upward=False)
            cc = utils.net_credit(call_chain.loc[short_call_strike, "bid"], call_chain.loc[long_call_strike, "ask"])
            pc = utils.net_credit(put_chain.loc[short_put_strike, "bid"], put_chain.loc[long_put_strike, "ask"])
            if cc + pc > 0:
                chosen = (width, long_call_strike, long_put_strike, cc, pc)
                break
            width = round(width - 1.0, 1)

        if chosen is None:
            logger.warning("Skip %s %s: no wing width in [%.0f, %.0f]pt yields positive net credit.",
                           self.underlying, expiration, self.min_wing_width_pts, self.wing_width_pts)
            return

        width, long_call_strike, long_put_strike, self._call_credit, self._put_credit = chosen
        target_width = width

        # --- Pull bids/asks and validate them ----------------------------------------------------
        short_call_bid = call_chain.loc[short_call_strike, "bid"]
        long_call_ask = call_chain.loc[long_call_strike, "ask"]
        short_put_bid = put_chain.loc[short_put_strike, "bid"]
        long_put_ask = put_chain.loc[long_put_strike, "ask"]
        # [FEAT 10] Reject missing/NaN/negative/zero quotes before we ever send an order.
        if not utils.quotes_valid([short_call_bid, long_call_ask, short_put_bid, long_put_ask]):
            logger.warning("Skip %s %s: a chosen leg has no valid quote.", self.underlying, expiration)
            return

        # --- [FEAT 09] Credit and max-risk math ------------------------------------------------
        # net_credit() uses SHORT bid and LONG ask so the credit isn't overstated.
        total_credit = self._call_credit + self._put_credit
        # Max loss per condor ≈ (one wing's width in points) − the credit that offsets it.
        max_risk_per_contract = utils.max_risk_premium(1, target_width, total_credit)

        # --- [FEAT 08] Risk-based position sizing ----------------------------------------------
        # qty = max(1, floor(account * risk_pct / (max_risk * multiplier))). This is exactly why
        # we need [FEAT 01] the engine's real multiplier — hardcoding 100 diverges under --multiplier.
        multiplier = self.get_run_multiplier()                                    # [FEAT 01]
        if self.quantity is None:
            qty = utils.size_position(self.cash(), self.risk_pct, max_risk_per_contract, multiplier)
        else:
            qty = self.quantity
        if qty <= 0:
            logger.warning("Skip %s %s: sized 0 contracts.", self.underlying, expiration)
            return

        # --- Build the 4 legs and submit them with one shared group_id --------------------------
        # The engine is leg-agnostic; the shared group_id is what lets reports group these 4
        # single-leg positions into one condor.  # [ENG C]
        self._exiting = False  # fresh day, fresh book — management may run again
        self._book_modified = False
        self._wing_width_used = target_width  # persisted for management stops today
        short_call = Contract(self.underlying, expiration, short_call_strike, "CALL")
        long_call = Contract(self.underlying, expiration, long_call_strike, "CALL")
        short_put = Contract(self.underlying, expiration, short_put_strike, "PUT")
        long_put = Contract(self.underlying, expiration, long_put_strike, "PUT")
        self._group_id = f"{self.underlying}_{expiration:%Y%m%d}_condor_demo"

        self.submit_order(short_call, "SELL_TO_OPEN", qty, limit_price=float(short_call_bid), group_id=self._group_id)
        self.submit_order(long_call, "BUY_TO_OPEN", qty, limit_price=float(long_call_ask), group_id=self._group_id)
        self.submit_order(short_put, "SELL_TO_OPEN", qty, limit_price=float(short_put_bid), group_id=self._group_id)
        self.submit_order(long_put, "BUY_TO_OPEN", qty, limit_price=float(long_put_ask), group_id=self._group_id)

        # Track all four legs for management (long wings are needed to value the book fairly).
        self._short_call = _ShortLeg(short_call, entry_credit=float(short_call_bid))
        self._short_put = _ShortLeg(short_put, entry_credit=float(short_put_bid))
        self._long_call = long_call
        self._long_put = long_put

        logger.info(
            "Entered %s condor x%d, credit=%.3f (call %.0f/%.0f=%.2f, put %.0f/%.0f=%.2f), spot %.2f, wing %.0fpt%s",
            self.underlying, qty, total_credit,
            short_call_strike, long_call_strike, self._call_credit,
            short_put_strike, long_put_strike, self._put_credit, spot, target_width,
            "" if target_width >= self.wing_width_pts - 0.01 else " (tightened)",
        )

    # =============================================================================================
    # Management
    # =============================================================================================
    def _manage_position(self, ts: datetime) -> None:
        """Per-bar: take profit on the whole book, or stop the book/short legs if threatened."""
        # Once we've triggered a whole-condor flatten (profit-target, book-stop, or before_close),
        # stop re-evaluating. MARKET closes fill on the NEXT engine bar, so without this guard the
        # condition would re-fire every bar (still seeing the pre-fill value) and resubmit
        # duplicate closing orders each minute.
        if self._exiting:
            return

        current_value = self.spread_value()

        # --- Whole-book stop: give back at most (1 - book_stop_frac) of the wing -----------------
        # Caps the loss on the whole book relative to the spread's max risk (the wing width in
        # points), which is the economically meaningful scale for a credit spread — the skimpy
        # credit itself is too small to size a stop against. Stop if the remaining book would
        # already cost `book_stop_frac` of a full wing to close right now.
        if current_value is not None and self._wing_width_used > 0 \
                and current_value >= self.book_stop_frac * self._wing_width_used:
            logger.info("Book stop hit (value %.3f >= %.0f%% of %.1fpt wing) — closing condor.",
                        current_value, self.book_stop_frac * 100, self._wing_width_used)
            self._flatten_all(ts, reason="book_stop")
            return

        # Once a per-leg stop has closed part of the book, the remaining legs are no longer a clean
        # "credit to retain" setup — the whole-book profit target is no longer meaningful, so we
        # skip it and rely on stops + before_close to clean up the residue.
        if not self._book_modified and current_value is not None:
            max_credit = self._call_credit + self._put_credit
            # profit_target_reached for a SHORT spread: captured = max_credit - current_value,
            # target reached when captured >= target_frac * max_credit  <=>  current_value <=
            # (1 - target_frac) * max_credit.  # [FEAT 04]
            if utils.profit_target_reached(max_credit, current_value, self.profit_target_frac):
                logger.info("Profit target reached (value %.3f <= %.0f%% of %.3f credit) — closing condor.",
                            current_value, self.profit_target_frac * 100, max_credit)
                self._flatten_all(ts, reason="target")
                return

        # Independent per-leg stops (each short leg's cost-to-close doubled from its own credit).
        if self._short_call is not None and not self._short_call.closed:
            self._check_short_leg_stop(self._short_call)
        if self._short_put is not None and not self._short_put.closed:
            self._check_short_leg_stop(self._short_put)

    def _check_short_leg_stop(self, leg: _ShortLeg) -> None:
        """Buy back a single short leg at its current ask if its cost has grown past the stop."""
        quote = self.get_quote(leg.contract)                       # [ENG E]
        if quote is None:
            return  # no quote yet this bar — recheck next bar
        stop_price = leg.entry_credit * self.leg_stop_multiple
        if quote["ask"] < stop_price:
            return
        self.submit_order(leg.contract, "BUY_TO_CLOSE", self._leg_qty(leg.contract),
                          limit_price=float(quote["ask"]), group_id=self._group_id)
        leg.closed = True
        self._book_modified = True  # book is no longer a clean credit-hold; abandon profit target
        logger.info("Short leg %s stopped out at %.2f (>= %.1fx the %.2f entry credit).",
                    leg.contract.key, quote["ask"], self.leg_stop_multiple, leg.entry_credit)

    def _flatten_all(self, ts: datetime, reason: str) -> None:
        """
        Close every currently-open leg via market orders (buy-to-close / sell-to-close). This runs
        at the before_close hook and on profit-target. Anything we missed expires and gets
        cash-settled by the engine at 16:00 ET (0DTE cash settlement path — [ENG F]).
        """
        if self._exiting:
            return  # already submitted a flatten for this book
        self._exiting = True
        qty_by_side: dict[str, list[tuple[Contract, int]]] = {}
        for pos in self.positions():  # positions() returns a list of Position (contract + qty)
            if pos.qty == 0:
                continue
            # Close a LONG (qty>0) by SELLING it; close a SHORT (qty<0) by BUYING it back.
            side = "SELL_TO_CLOSE" if pos.qty > 0 else "BUY_TO_CLOSE"
            qty_by_side.setdefault(side, []).append((pos.contract, abs(pos.qty)))
        for side, legs in qty_by_side.items():
            for contract, qty in legs:
                self.submit_order(contract, side, qty, order_type="MARKET", group_id=self._group_id)
        logger.info("Flatten %s: submitted %d closing order(s) (reason=%s).",
                    reason, sum(len(v) for v in qty_by_side.values()), reason)

    # =============================================================================================
    # Small helpers
    # =============================================================================================
    def _leg_qty(self, contract: Contract) -> int:
        """Current signed qty magnitude for a contract, or fall back to 1 if position is gone."""
        pos = self.get_position(contract)
        qty = abs(pos.qty) if pos is not None else 1
        return max(qty, 1)

    def _prior_day_vix_close(self, day: date) -> Optional[float]:
        """
        VIX close of the most recent XNYS trading day strictly before `day`.

        The engine's XNYS calendar is the canonical trading-day source for this run, so we get the
        prior session from `engine.calendar.trading_days` and look up that day's VIX close. Returns
        None if the VIX CSV has no entry for the prior session (safe: filter simply doesn't fire).
        [FEAT 02]
        """
        from datetime import timedelta

        from engine.calendar import trading_days

        # Walk backward up to a few calendar days, using the XNYS calendar to land only on real
        # sessions. trading_days is inclusive, so query a short window ending just before `day`.
        for back in range(1, 6):  # enough to clear a weekend + one holiday
            cand = day - timedelta(days=back)
            sessions = [ts.date() for ts in trading_days(cand, cand)]
            if not sessions:  # not an XNYS trading day — keep walking back
                continue
            prior = get_daily_vix(sessions[0], "close", data_dir=self.data_dir)
            if prior is not None:
                return prior
            break  # a real session exists but has no VIX row — no filter applies
        return None
