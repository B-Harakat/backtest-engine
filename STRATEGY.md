# Writing Strategies

Reference for authoring a strategy file (e.g. `strategies/iron_condor_example.py`) for the
backtest engine. Covers file layout, the full engine-provided API, lifecycle hooks, ordering
conventions, multi-leg composition, Monte Carlo support, and all engine utility functions.

Existing examples to read alongside this guide:

- `strategies/call_credit_spread_example.py` — two-leg debit spread with an independent short-leg stop.
- `strategies/bull_put_spread_example.py` — two-leg credit spread with an independent short-leg stop.
- `strategies/iron_condor_example.py` — four-leg condor across both rights; exercises every engine capability.

---

## 1. What a strategy is

A strategy is a **Python class that subclasses `engine.strategy.Strategy`**. During a backtest the
engine:

1. Instantiates your class (your `__init__`), once per run.
2. Calls `attach(...)` then `initialize()`.
3. For every 1-minute trading bar, calls `on_bar(ts)`.
4. On the last bar of each session, calls `before_close(ts)`.
5. Orders you submitted are filled by the engine; mark-to-market and expiration handling happen
   automatically.

You write **decision logic only**. You never manage cash, fills, positions, or data fetching —
you use the engine-provided helpers to read data and submit orders.

There is **no threading or scheduler**, and there is **no current-price argument** passed into
`on_bar`. To see the underlying's price, use `get_chain_snapshot(..., right="call"/"put")` (which
sources the underlying price from option chain data; `right` is required — the chain is
single-sided).

---

## 2. File layout

One strategy class per file in `strategies/`. A file has three optional parts:

1. **Module docstring** — what the strategy does.
2. **Module-level constants** and a `GRID` dict (see §10) for Monte Carlo parameter exploration.
3. **The `Strategy` subclass** — the class itself.

The engine locates a strategy by spec string `module.path:ClassName`:
`strategies.call_credit_spread_example.CallCreditSpreadExample`.

### Minimal skeleton

```python
from datetime import date, time

from engine.entities import Contract
from engine.strategy import Strategy


class MyNewStrategy(Strategy):
    def __init__(self, underlying: str = "XSP", quantity: int = 1):
        super().__init__()
        self.underlying = underlying
        self.quantity = quantity

    def initialize(self):
        pass

    def on_bar(self, ts):
        pass

    def before_close(self, ts):
        pass
```

---

## 3. Constructor contract

Your `__init__` takes **tunable parameters as keyword arguments with defaults**.

Key conventions:

- Use `underlying: str = "XSP"` as the default underlying symbol — `run_backtest --ticker` and
  `run_mc --ticker` map to this exact `underlying=` argument. A different param name means
  `--ticker` won't work.
- **Always call `super().__init__()`** before touching anything.
- Do NOT instantiate a `Portfolio`, `DataProvider`, etc. — the engine wires those up via `attach()`.
- Constructor args become grid-able parameters for Monte Carlo; see §10.

---

## 4. Lifecycle hooks

### `initialize()`

Called once before the backtest starts. Use it to call `self.watch(...)` for every contract you
know in advance you'll trade, so data is fetched once up front instead of lazily mid-loop.

### `on_bar(ts: datetime)`

Called once per minute bar during regular trading hours. `ts` is a naive wall-clock
America/New_York bar-start timestamp. This is where most decision logic (entries, exits, stops)
lives. Submit orders here; they fill against this bar's quotes.

**Important**: Orders submitted on bar `ts` are filled on the **next** bar the engine processes.
If you read `spread_value()` or `get_quote()` on the entry bar itself, those values are pre-fill
artifacts. Skip management on the entry bar.

### `before_close(ts: datetime)`

Called on the last bar of each session (respects early closes). The natural place for 0DTE
flatten-before-close logic. Orders submitted here are processed against this same bar's quotes
**before** end-of-day expiration settlement runs.

---

## 5. Data-reading helpers

All methods on your strategy; none require touching `DataProvider` directly.

| Helper | Returns | Notes |
|---|---|---|
| `self.get_datetime()` | `datetime` | current bar timestamp |
| `self.get_bars(contract, lookback=1)` | `pd.DataFrame` | most recent `lookback` bars, up to and including now |
| `self.get_quote(contract)` | `dict` / `None` | `{"bid": ..., "ask": ...}` at or before now |
| `self.get_greeks(contract)` | `dict` / `None` | delta, gamma, theta, vega, rho, vanna, charm, implied_vol, underlying_price |
| `self.get_chain_snapshot(underlying, expiration, right)` | `pd.DataFrame` | full ONE-side chain indexed by strike, with `underlying_price` and `delta` columns. `right` is REQUIRED — `"call"` or `"put"`. Calls and puts are fetched/cached independently; request each side once if you need both |
| `self.cash()` | `float` | current account cash |
| `self.positions()` | `list[Position]` | currently open positions |
| `self.get_position(contract)` | `Position` / `None` | open position for a specific contract |
| `self.get_run_multiplier()` | `int` | contract multiplier used by the ledger (default 100); use this instead of hardcoding |
| `self.spread_value()` | `float` / `None` | §6 |
| `self.aggregate_greeks()` | `dict[str, float]` | §6 |

### Getting the current underlying price

```python
chain = self.get_chain_snapshot(self.underlying, expiration, right="call")
spot = float(chain["underlying_price"].iloc[0])
```

### Pre-fetching data

```python
from engine.entities import Contract

def initialize(self):
    c = Contract(underlying="XSP", expiration=date(2026, 1, 16), strike=695.0, right="CALL")
    self.watch(c)
```

---

## 6. Position-level helpers

### `self.spread_value() -> Optional[float]`

Cost-to-close of **every** currently-open position, in premium points **per contract** (not scaled
by qty). Short legs (qty < 0) valued at their current ask, long legs (qty > 0) at their current
bid. Returns `None` if any open leg has no usable quote this bar — caller rechecks next bar.

Returns per-contract value so it's directly comparable to per-contract entry credits/debits used
by profit-target logic.

```python
current_value = self.spread_value()
if current_value is not None and current_value <= (1.0 - target_frac) * total_credit:
    self._flatten_all(ts, reason="target")
```

### `self.aggregate_greeks() -> dict[str, float]`

Summed raw (per-unit, before multiplier scaling) greeks across all open positions. Positive delta
= net long, negative = net short. Gamma, theta, vega, rho, vanna, and charm are signed by
position direction (long legs add, short legs subtract).

Returns a dict with keys `delta`, `gamma`, `theta`, `vega`, `rho`, `vanna`, `charm`,
`implied_vol`, and `underlying_price`. Positions with no greek data this bar are silently skipped.

```python
book = self.aggregate_greeks()
if abs(book["delta"]) > 50.0:
    self._flatten_all(ts, reason="delta_exceeded")
```

---

## 7. Ordering

### Building a `Contract`

```python
from engine.entities import Contract

c = Contract(underlying="XSP", expiration=date(2026, 1, 16), strike=695.0, right="CALL")
```

`right` is `"CALL"` or `"PUT"`. The canonical key (used for dict keys and report filenames) is
`XSP_20260116_C695`.

### `self.submit_order(...)`

```python
order = self.submit_order(
    contract=c,
    side="BUY_TO_OPEN",        # BUY_TO_OPEN | SELL_TO_OPEN | BUY_TO_CLOSE | SELL_TO_CLOSE
    qty=1,
    order_type="LIMIT",        # "LIMIT" (default) or "MARKET"
    limit_price=2.50,          # required for LIMIT; must be None for MARKET
    group_id="my_group",       # optional, for reporting multi-leg positions together
)
```

Conventions and rules:

- **`qty` is always positive**; direction is encoded in `side`. `BUY_TO_OPEN`/`BUY_TO_CLOSE` are
  buys, `SELL_TO_OPEN`/`SELL_TO_CLOSE` are sells.
- **LIMIT** orders require a `limit_price` and fill only once the market touches it — never at a
  price worse than your limit.
- **MARKET** orders require `limit_price=None` and fill at the current touch (ask on a buy, bid on
  a sell) on the **next** bar the engine processes. There is **no slippage model** — MARKET fills
  are perfect executions at the book.
- You are **never auto-closed at expiration** by ordering; call `submit_order(...,
  SELL_TO_CLOSE/BUY_TO_CLOSE)` for manual exits, or let the engine's expiration/settlement
  handle positions still open at expiry (`settlement_style` chosen in `run_backtest`).

---

## 8. Multi-leg positions and `group_id`

The engine is **leg-agnostic**: every `Order`/`Position`/`Fill` refers to exactly **one contract**.
A spread or condor is several independent single-leg orders your strategy submits, optionally
tagged with a shared `group_id` so reports can group them.

```python
group = f"{self.underlying}_{expiration:%Y-%m-%d}_call_spread"
self.submit_order(short, "SELL_TO_OPEN", qty=1, limit_price=short_bid, group_id=group)
self.submit_order(long,  "BUY_TO_OPEN",  qty=1, limit_price=long_ask,  group_id=group)
```

The ledger treats them as unrelated single legs; your strategy code manages them together.

---

## 9. Engine utility functions (`engine.strategy_utils`)

All functions live in `engine.strategy_utils`. Import as:

```python
from engine import strategy_utils as utils
```

All prices in these functions are **premium points** (before contract multiplier). Use
`self.get_run_multiplier()` when converting to dollars.

### Strike selection

**`select_strike_by_delta(chain, target_delta, right) -> float`**

The strike whose `delta` is closest (in absolute terms) to `target_delta`. `target_delta` is an
absolute delta (e.g. `0.12`); the function handles signing internally based on `right`.

```python
short_call = utils.select_strike_by_delta(call_chain, 0.10, "CALL")
short_put  = utils.select_strike_by_delta(put_chain, 0.10, "PUT")
```

**`select_strike_by_offset(strikes, base_strike, offset) -> float`**

The strike `offset` index positions away from `base_strike` in a sorted strike list. Positive
offset = higher strike, negative = lower.

```python
long_call = utils.select_strike_by_offset(call_strikes, short_strike, 2)    # 2 strikes above
long_put  = utils.select_strike_by_offset(put_strikes, short_strike, -2)    # 2 strikes below
```

**`select_wing_strike(strikes, base, width_pts, upward) -> float`**

The first listed strike at or beyond `width_pts` **points** away from `base`. `upward=True` for
call wings (strike >= base + width_pts), `upward=False` for put wings (strike <= base - width_pts).
Prefer this over `select_strike_by_offset` when you think in point distances rather than index
positions.

```python
long_call = utils.select_wing_strike(call_strikes, short_call_strike, 2.0, upward=True)
long_put  = utils.select_wing_strike(put_strikes, short_put_strike, 2.0, upward=False)
```

**`select_nearest_otm(chain, spot, right) -> Optional[float]`**

The strike nearest to `spot` on the out-of-the-money side. For calls, the smallest strike above
spot. For puts, the largest strike below spot. Returns `None` if no OTM strike exists.

### Profit target

**`profit_target_reached(entry_value, current_value, target_fraction) -> bool`**

True if `target_fraction` of the maximum potential profit has been captured. Works for both credit
and debit spreads: `captured = entry_value - current_value`. Pass `spread_value()` as
`current_value`.

```python
if utils.profit_target_reached(max_credit, self.spread_value(), 0.50):
    self._flatten_all(ts, reason="target")
```

### Risk math

**`net_credit(short_price, long_price) -> float`**

Net credit received: `short_price - long_price`. `short_price` must be the **bid** of the short
leg, `long_price` the **ask** of the long leg.

**`max_risk_premium(width, strike_interval, credit) -> float`**

Maximum risk per contract: `width * strike_interval - credit`, floored at 0. `width` is the
number of strike positions between legs, `strike_interval` the distance between adjacent strikes.

**`size_position(cash, risk_pct, max_risk_premium, multiplier) -> int`**

Number of contracts to risk at most `cash * risk_pct` per trade. Formula:
`max(1, floor(cash * risk_pct / (max_risk * multiplier)))`.

```python
qty = utils.size_position(self.cash(), 0.01, max_risk_per_contract, self.get_run_multiplier())
```

### Chain validation

**`chain_has_columns(chain, cols) -> bool`**

True if the chain DataFrame contains all required columns. Accepts a single column name or an
iterable. `None`/empty chains return False.

**`chain_has_enough_otm(chain, spot, right, min_count) -> bool`**

True if the chain lists at least `min_count` out-of-the-money strikes on the given side.

**`quotes_valid(quotes, min_value=0.0) -> bool`**

True if every quote value is a finite number >= `min_value`. Rejects `None`, `NaN`, and infinities.

---

## 10. Monte Carlo support: the `GRID` spec

`run_mc` discovers a strategy's parameter space from a module-level `GRID` dict. `GRID` maps
**parameter name → list of candidate values**. `run_mc` builds the Cartesian product of these
values into parameter sets, each passed via `**kwargs` to your constructor. **Parameters not
listed in `GRID` keep their constructor defaults.**

```python
GRID = {
    "short_delta": [0.10, 0.12],
    "wing_width_pts": [2.0, 3.0],
    "entry_time": [time(10, 0), time(11, 0)],
}
```

- The grid lives **in the strategy file** alongside the class.
- `run_mc` imports the module via `--strategy` and reads `module.GRID`.
- A strategy **without** a `GRID` runs via `run_mc` as a single default-parameter set.

Constraint: the synthetic market maker is scoped to **0DTE intraday** sessions. A strategy
designed for `run_mc` should be 0DTE-compatible (entry day == expiration day).

---

## 11. External data filters

These are optional imports for filtering trades by market regime.

### VIX lookup (`engine.vix_data`)

```python
from engine.vix_data import get_daily_vix

vix = get_daily_vix(some_date, "close")    # prior day's closing print
vix = get_daily_vix(some_date, "open")     # session open print
```

Returns `None` if the VIX CSV is missing or the day isn't present — safe to use without a guard.
Data source: `data/vix_data.csv` (populated by `vix-download.py` from Cboe).

### Event-day avoidance (`engine.event_calendar`)

```python
from engine.event_calendar import is_event_day

if is_event_day(ts.date()):
    return  # skip FOMC, CPI, NFP, PPI days
```

Returns `False` if the event CSV is missing or the day isn't listed — safe to call before the
file is populated. Data source: `data/events/event_days.csv`.

---

## 12. Running your strategy

### Real backtest (ThetaData data)

```bash
python scripts/run_backtest.py --strategy strategies.iron_condor_example:IronCondorDemo \
    --start 2023-01-01 --end 2023-01-13 --cash 10000 --ticker XSP
```

Flags: `--multiplier`, `--settlement-style cash|physical`, `--settlement-time HH:MM`,
`--cost-model ibkr|none`, `--data-dir`, `--output-dir`, `--debug`.

### Monte Carlo (synthetic Bates paths)

```bash
python scripts/run_mc.py --strategy strategies.iron_condor_example:IronCondorDemo \
    --day 2026-01-15 --n-paths 100 --seed 7 --config mc_config.toml
```

Flags: `--day` or `--start/--end`, `--n-paths`, `--seed` (-1 for fresh randomness), `--config`,
`--ticker`, `--s0`, `--v0` (fallback when cached chain is missing), `--cash`, `--multiplier`,
`--settlement-time`, `--cost-model`, `--output-dir`.

### Output & reporting

Both entry points write under `output/<ticker>/<StrategyClassName>/`:

| Run mode | Subdirectory | Contents |
|---|---|---|
| `run_backtest` | `HISTORY/` | `trade_log.csv`, `cash_equity.csv`, `summary.json` |
| `run_mc` | `MONTE-CARLO/` | `<date>.csv` per day, `_summary.csv` (rolled up across days) |

MC columns are labelled by parameter values (e.g. `short_delta=0.1|wing_width_pts=2.0`); rows
are pooled statistics (`{stat}_mean`, `{stat}_p05`, `{stat}_p95`).

---

## 13. Conventions and gotchas

- Always `super().__init__()` in your constructor.
- `qty` is always positive; direction lives in the `side`.
- The underlying symbol param must be named `underlying` for `--ticker` to work.
- You never manage cash/positions/fills — only submit orders and read data.
- No price argument is passed to `on_bar`; get price/strikes from
  `get_chain_snapshot(..., right=...)`.
- Multi-leg structures are independent single contracts sharing a `group_id`.
- For `run_mc`, define a `GRID` dict and keep the strategy 0DTE-compatible (entry day ==
  expiration day).
- Class names become output-folder names — keep them unique and descriptive.
- Constructor parameter names become MC column labels — keep them short and descriptive.
- Orders submitted on bar `ts` fill on the next bar; skip management on the entry bar.
