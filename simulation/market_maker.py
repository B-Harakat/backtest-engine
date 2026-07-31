"""
Synthetic market maker: builds the engine's fetch functions (``FetchFn`` / ``ChainFetchFn``)
from a single Bates path so that ``engine.runner.run_backtest`` / ``engine.data_store`` can
drive a backtest off synthesized prices with no knowledge that the "feed" isn't ThetaData.

The market maker is BOUND to ONE simulated session path (spot+vol per minute bar). Because the
target use case is intraday 0DTE XSP, each Monte Carlo path = one trading day = one backtest
run (``start_date == end_date == contract.expiration``), so "the whole path" is the natural
unit handed to the engine's ``get_bars``/``get_chain`` (which slice what they need and cache it
per contract under an isolated ``data_dir``).

SCOPE LIMITATION: this market maker only serves data for the session(s) its path covers and, for
the per-contract ``fetch_fn``, only for contracts expiring ON that path's day. Requesting a
contract that expires a different day (or a range with no path data) returns an empty frame,
which ``engine.data_store.compute_gaps`` can never resolve -- the engine treats it as a
perpetual unsatisfiable gap. For 0DTE intraday runs this never arises; do NOT use this maker for
multi-day / non-0DTE strategies without extending it (generate one path per day and bind a maker
per day).

Data shape contract (must match every consumer):

  Per-contract ``FetchFn(contract, gap_start, gap_end) -> DataFrame``:
      index        : DatetimeIndex of 1-minute bar starts (naive ET)
      columns      : bid, ask, close, underlying_price,
                     delta, gamma, theta, vega, rho, vanna, charm, implied_vol
      consumed by  : engine.data_store.quote_at (bid/ask),
                     engine.runner._mark_to_market_prices (mid, close fallback),
                     engine.runner._settlement_price (underlying_price at ~16:00),
                     engine.strategy.get_greeks / get_quote.

  Per-chain ``ChainFetchFn(underlying, expiration, gap_start, gap_end, right) -> DataFrame``:
      index        : (timestamp, strike) MultiIndex
      columns      : bid, ask, underlying_price, delta (+ the other greeks)
      consumed by  : engine.data_store.chain_snapshot -> Strategy.get_chain_snapshot
                     (strike set + underlying_price + delta for selection).

Spread model: the engine has NO slippage and fills at the touch, so execution quality is
entirely up to the synthetic bid/ask. We model a constant proportional half-spread with an
absolute tick floor; bid is never negative. Configurable via ``SpreadConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from engine.entities import Contract
from simulation.option_pricer import price_option

# Greeks the per-contract fetch should carry (mirrors engine.strategy._GREEK_COLUMNS).
_GREEK_COLS = ("delta", "gamma", "theta", "vega", "rho", "vanna", "charm", "implied_vol")


@dataclass(frozen=True)
class SpreadConfig:
    """
    Bid/ask width model for synthetic quotes. No slippage in the engine, so this is the entire
    execution-cost lever. ``half_spread_frac`` is the proportion of mid price used for each
    side (0.01 = 1% both ways => 2% total width). ``tick_size`` floors the absolute half-width
    to a market's minimum tick (XSP options trade in 0.05 ticks); ``spread_noise_std`` adds
    proportional per-bar jitter (0 = deterministic).
    """

    half_spread_frac: float = 0.005      # 0.5% each side => 1% total quoted width.
    tick_size: float = 0.05              # half-width floor in price points.
    spread_noise_std: float = 0.0        # std of a multiplicative (0-mean) noise on the width.


def _spread_half_width(mid: float, spread_cfg: SpreadConfig, rng: np.random.Generator) -> float:
    noise = 1.0
    if spread_cfg.spread_noise_std > 0:
        noise = max(0.05, 1.0 + spread_cfg.spread_noise_std * rng.standard_normal())
    return max(mid * spread_cfg.half_spread_frac * noise, spread_cfg.tick_size * 0.5)


@dataclass(frozen=True)
class StrikeGrid:
    """
    How the synthetic chain enumerates strikes around the spot. ``step`` is the strike spacing,
    ``margin_frac`` the fractional band on each side of spot, and ``min_strikes`` a floor on the
    count per side so even a quiet day yields enough strikes for a ``spread_width`` far leg.
    """

    step: float = 1.0
    margin_frac: float = 0.15        # 15% band each side of spot.
    min_strikes: int = 10

    def strikes_around(self, spot: float) -> list[float]:
        lo = max(1.0, spot * (1.0 - self.margin_frac))
        hi = spot * (1.0 + self.margin_frac)
        lo = np.floor(lo / self.step) * self.step
        hi = np.ceil(hi / self.step) * self.step
        vals = np.arange(lo, hi + self.step, self.step)
        while len(vals) < 2 * self.min_strikes:
            lo -= self.step
            hi += self.step
            vals = np.arange(lo, hi + self.step, self.step)
        return [round(float(v), 6) for v in vals]


class SyntheticMarketMaker:
    """
    Binds a Bates path (``path: DataFrame`` indexed by naive-ET timestamps with ``spot`` and
    ``vol``) to the engine's fetch-fn interface. One instance per simulated session.
    """

    def __init__(
        self,
        path: pd.DataFrame,
        underlying: str = "XSP",
        spread: SpreadConfig = SpreadConfig(),
        strike_grid: StrikeGrid = StrikeGrid(),
        seed: Optional[int] = None,
    ):
        if path.empty or not isinstance(path.index, pd.DatetimeIndex):
            raise ValueError("SyntheticMarketMaker requires a non-empty DateTime-indexed path")
        required = {"spot", "vol"}
        missing = required - set(path.columns)
        if missing:
            raise ValueError(f"path must have columns {sorted(required)}, missing {sorted(missing)}")
        self.path = path
        self.underlying = underlying
        self.spread = spread
        self.strike_grid = strike_grid
        self._rng = np.random.default_rng(seed)

    # --- helpers ---------------------------------------------------------------------------

    def _timestamps_in(self, gap_start: datetime, gap_end: datetime) -> pd.DatetimeIndex:
        idx = self.path.index
        return idx[(idx >= np.datetime64(gap_start)) & (idx <= np.datetime64(gap_end))]

    def _bid_ask(self, mid: float) -> tuple[float, float]:
        hw = _spread_half_width(mid, self.spread, self._rng)
        bid = max(mid - hw, 0.0)
        ask = mid + hw
        return round(float(bid), 6), round(float(ask), 6)

    def _price_contract_row(self, contract: Contract, ts: pd.Timestamp) -> Optional[dict]:
        """Pricing row for one contract at one timestamp, or None if the contract's day isn't
        this path's day / spot unavailable."""
        t = ts.to_pydatetime()
        # This market maker is a single-session feed: only serve the session(s) this path covers.
        if contract.expiration != t.date():
            return None
        spot = float(self.path.loc[ts, "spot"])
        vol = float(self.path.loc[ts, "vol"])
        if spot <= 0:
            return None
        po = price_option(
            spot=spot, strike=float(contract.strike), right=contract.right,
            expiration=contract.expiration, as_of=t, spot_vol=vol,
        )
        bid, ask = self._bid_ask(po.price)
        row = {"bid": bid, "ask": ask, "close": po.price, "underlying_price": spot}
        for g in _GREEK_COLS:
            row[g] = getattr(po, g)
        return row

    # --- engine FetchFn --------------------------------------------------------------------

    def fetch_fn(self, contract: Contract, gap_start: datetime, gap_end: datetime) -> pd.DataFrame:
        """Per-contract time series (see module docstring for shape contract)."""
        rows = []
        for ts in self._timestamps_in(gap_start, gap_end):
            row = self._price_contract_row(contract, ts)
            if row is not None:
                row["timestamp"] = ts
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df[["bid", "ask", "close", "underlying_price", *_GREEK_COLS]]

    # --- engine ChainFetchFn ---------------------------------------------------------------

    def chain_fetch_fn(
        self,
        underlying: str,
        expiration: date,
        gap_start: datetime,
        gap_end: datetime,
        right: str = "call",
    ) -> pd.DataFrame:
        """Full one-side chain as a (timestamp, strike) MultiIndex (see module docstring)."""
        right_upper = "CALL" if right == "call" else "PUT"
        rows = []
        for ts in self._timestamps_in(gap_start, gap_end):
            spot = float(self.path.loc[ts, "spot"])
            vol = float(self.path.loc[ts, "vol"])
            if spot <= 0:
                continue
            t = ts.to_pydatetime()
            for strike in self.strike_grid.strikes_around(spot):
                po = price_option(
                    spot=spot, strike=strike, right=right_upper,
                    expiration=expiration, as_of=t, spot_vol=vol,
                )
                bid, ask = self._bid_ask(po.price)
                rows.append(
                    {
                        "timestamp": ts, "strike": strike,
                        "bid": bid, "ask": ask, "underlying_price": spot,
                        "delta": po.delta, "gamma": po.gamma, "theta": po.theta,
                        "vega": po.vega, "rho": po.rho, "vanna": po.vanna,
                        "charm": po.charm, "implied_vol": po.implied_vol,
                    }
                )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index(["timestamp", "strike"]).sort_index()
        return df
