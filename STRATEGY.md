# Writing Strategies

This is the guide to authoring a strategy file (e.g. `strategies/vertical_spread_strategy.py`)
for the backtest engine. It covers the file layout, the engine-provided API a strategy has access
to, the lifecycle hooks, ordering/position conventions, multi-leg composition, and how strategies
connect to `run_backtest`, `run_mc`, and the generated reports.

Existing examples to read alongside this guide:

- `strategies/my_strategy.py` — minimal 0DTE single-leg call buy.
- `strategies/vertical_spread_strategy.py` — a two-leg debit spread with an independent short-leg stop.
- `strategies/iron_condor_strategy.py` — a four-leg condor across both rights.
- `strategies/laddered_condor_strategy.py` — multi-day, delta-targeted, rolling "kitchen sink" example.

---

## 1. What a strategy is

A strategy is a **Python class that subclasses `engine.strategy.Strategy`**. During a backtest the
engine:

1. Instantiates your class (your `__init__`), once per run.
2. Calls `attach(...)` then `initialize()`.
3. For every 1-minute trading bar, calls `on_bar(ts)`.
4. On the last bar of each session, calls `before_close(ts)`.
5. Any orders your strategy submitted are filled by the engine, and mark-to-market/expiration
   handling happens automatically.

You write **decision logic only**. You never manage cash, fills, positions, or data fetching —
you use the engine-provided helpers to query data and submit orders, and the engine does the rest.

There is **no threading or scheduler**, and there is **no current-price argument** passed into
`on_bar`. To see the underlying's price, use `get_chain_snapshot()` (which sources the underlying
price from option chain data rather than a separate index feed).

---

## 2. File layout

There is no enforced file layout, but the convention is **one strategy class per file** in
`strategies/`. A strategy file has three optional parts:

1. **Module docstring** — what the strategy does and the engine capabilities it exercises.
2. **Module-level constants** and a `GRID` dict (see §7) for Monte Carlo parameter exploration.
3. **The `Strategy` subclass** — the class itself.

The engine locates a strategy by spec string `module.path:ClassName`, so for
`strategies/vertical_spread_strategy.py` the spec is
`strategies.vertical_spread_strategy:VerticalSpreadStrategy`.

### Minimal skeleton

```python
from datetime import date, time

from engine.entities import Contract
from engine.strategy import Strategy


class MyNewStrategy(Strategy):
    def __init__(self, underlying: str = "XSP", quantity: int = 1, entry_time: time = time(10, 30)):
        super().__init__()
        self.underlying = underlying
        self.quantity = quantity
        self.entry_time = entry_time

    def initialize(self):
        # Pre-declare contracts you always trade (optional but recommended).
        pass

    def on_bar(self, ts):
        # Called once per minute bar. Submit orders via self.submit_order(...).
        pass

    def before_close(self, ts):
        # Optional: flatten/before-close logic on the session's last bar.
        pass
```

---

## 3. Constructor contract

Your `__init__` takes your strategy's **tunable parameters as keyword arguments with defaults**.

Key conventions:

- Use `underlying: str = "XSP"` as the default underlying symbol — `run_backtest --ticker` and
  `run_mc --ticker` map to this exact `underlying=` argument. Choosing a different param name means
  `--ticker` won't work for your strategy.
- **Always call `super().__init__()`** before touching anything.
- Do NOT instantiate a `Portfolio`, `DataProvider`, etc. — the engine wires those up via `attach()`.
- Constructor args become grid-able parameters for Monte Carlo; see §7.

---

## 4. Lifecycle hooks

### `initialize()`
Called once before the backtest starts. Use it to call `self.watch(...)` for every contract you
know in advance you'll trade, so data is fetched once up front instead of lazily mid-loop.

### `on_bar(ts: datetime)`
Called once per minute bar during regular trading hours. `ts` is a naive wall-clock
America/New_York bar-start timestamp (see `engine/calendar.py`). This is where most decision
logic (entries, exits, stops) lives. Submit orders here; they fill against this bar's quotes.

### `before_close(ts: datetime)`
Called on the last bar of each session (respects early closes). The natural place for 0DTE
flatten-before-close logic. Orders submitted here are processed against this same bar's quotes
**before** end-of-day expiration settlement runs.

---

## 5. Reading data (engine-provided helpers)

All of these are methods on your strategy; none require you to touch `DataProvider` directly.

| Helper | Returns | Notes |
|---|---|---|
| `self.get_datetime()` | `datetime` | current bar timestamp |
| `self.get_bars(contract, lookback=1)` | `pd.DataFrame` | most recent `lookback` bars for a contract, up to and including now |
| `self.get_quote(contract)` | `dict` / `None` | `{"bid": ..., "ask": ...}` at or before now |
| `self.get_greeks(contract)` | `dict` / `None` | most recent greeks (delta/gamma/theta/vega/rho/vanna/charm/implied_vol/underlying_price) |
| `self.get_chain_snapshot(underlying, expiration, right="call")` | `pd.DataFrame` | full one-side chain indexed by strike, with `underlying_price` and `delta` columns |
| `self.cash()` | `float` | current account cash |
| `self.positions()` | `list[Position]` | currently open positions |
| `self.get_position(contract)` | `Position` / `None` | open position for a specific contract |

### Getting the current underlying price
There is no separate index-price argument. Use the chain snapshot:

```python
chain = self.get_chain_snapshot(self.underlying, expiration, right="call")
spot = float(chain["underlying_price"].iloc[0])   # current underlying
```

### Pre-fetching data
```python
from engine.entities import Contract

def initialize(self):
    c = Contract(underlying="XSP", expiration=date(2026, 1, 16), strike=695.0, right="CALL")
    self.watch(c)   # warm the full backtest window
```

---

## 6. Ordering

### Building a `Contract`
A `Contract` identifies one instrument. It is either the underlying (all fields None) or an
option (must have `expiration`, `strike`, and `right` together).

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
    group_id="my_2026-01-16",  # optional, for reporting multi-leg positions together
)
```

Conventions and rules:

- **`qty` is always positive**; the direction is encoded in `side`. `BUY_TO_OPEN`/`BUY_TO_CLOSE`
  are buys, `SELL_TO_OPEN`/`SELL_TO_CLOSE` are sells.
- **LIMIT** orders require a `limit_price` and fill only once the market touches it — never at a
  price worse than your limit.
- **MARKET** orders require `limit_price=None` and fill at the current touch (ask on a buy, bid on
  a sell) on the next bar the engine processes. There is **no slippage model** — MARKET fills are
  perfect executions at the book.
- You are **never auto-closed at expiration** by ordering; call `submit_order(...,
  SELL_TO_CLOSE/BUY_TO_CLOSE)` for manual exits, or let the engine's expiration/settlement handle
  positions still open at their expiry (`settlement_style` chosen in `run_backtest`).

---

## 7. Multi-leg positions and `group_id`

The engine is **leg-agnostic**: every `Order`/`Position`/`Fill` refers to exactly **one contract**.
A "spread" or "condor" is just several independent single-leg orders your strategy submits,
optionally tagged with a shared `group_id` so the reports can group them.

```python
# A call credit spread = two independent orders on one group.
group = f"{self.underlying}_{expiration:%Y-%m-%d}_call_spread"
self.submit_order(short, "SELL_TO_OPEN", qty=1, limit_price=short_bid, order_type="LIMIT", group_id=group)
self.submit_order(long,  "BUY_TO_OPEN",  qty=1, limit_price=long_ask,  order_type="LIMIT", group_id=group)
```

The ledger/fill engine treat them as unrelated single legs; your strategy code is what manages
them together (e.g. checking `self.get_position(short_leg)` and `self.get_position(long_leg)`).

---

## 8. Optional Monte Carlo support: the `GRID` spec

`run_mc` explores a parameter grid per strategy, discovered from a module-level `GRID` dict in the
strategy file. `GRID` maps **parameter name → list of candidate values**. `run_mc` builds the
Cartesian product of these values into a set of parameter dicts, each passed via `**kwargs` to
your constructor. **Parameters not listed in `GRID` keep their constructor defaults** (partial
override).

```python
GRID = {
    "spread_width": [2, 3],
    "short_leg_stop_multiple": [1.5, 3.0],
    "entry_time": [time(10, 0), time(11, 0)],   # any iterable; non-hashable values are fine
}
```

- The grid lives **in the strategy file** alongside the class, so a strategy carries its own
  parameter space.
- `run_mc` finds it by importing the module via the `--strategy` spec and reading `module.GRID`.
- A strategy **without** a `GRID` runs fine via `run_mc` as a single default-parameter set.

There is one Monte-Carlo constraint to keep in mind: the engine's synthetic market maker is
scoped to **0DTE intraday** sessions. A strategy designed for `run_mc` should therefore be
0DTE-compatible (each path is one future- simulated session, entry day == expiration day).

---

## 9. Running your strategy

### Real backtest (ThetaData data)
```bash
python scripts/run_backtest.py \
  --strategy strategies.vertical_spread_strategy:VerticalSpreadStrategy \
  --start 2025-02-25 --end 2025-03-15 --cash 10000 --ticker XSP
```

### Monte Carlo (synthetic Bates paths)
```bash
python scripts/run_mc.py \
  --strategy strategies.vertical_spread_strategy:VerticalSpreadStrategy \
  --start 2025-02-25 --end 2025-03-15 --ticker XSP --n-paths 2 --seed 7 --cash 10000
```

### Output & reporting

Both entry points write reports under `output/`, organized by **your strategy class name,
uppercased**.

- `run_backtest` writes `output/<STRATEGY>/HISTORY/`:
  - `trade_log.csv` — every fill (including expiration settlements).
  - `cash_equity.csv` — daily granularity (`date`, `cash`, `equity`).
  - `summary.json` — full statistics battery.
- `run_mc` writes `output/<STRATEGY>/MONTE-CARLO/`:
  - `<date>.csv` per day — columns are parameter sets, rows are pooled statistics
    (`{stat}_mean/{stat}_p05/{stat}_p95`) per report.
  - `_summary.csv` — each statistic averaged across days per parameter set.

Because the output folder is the **class name** (`VERTICALSPREADSTRATEGY`), two different
classes with the same name would collide — keep class names unique. Your constructor parameter
values are used to label Monte-Carlo columns (e.g. `spread_width=2|entry_time=10:00`), so give
parameters short, descriptive names.

---

## 10. Gotchas and conventions summary

- Always `super().__init__()` in your constructor.
- `qty` is always positive; direction lives in the `side`.
- The underlying symbol param should be named `underlying` for `--ticker` to work.
- You never manage cash/positions/fills — only submit orders and read data.
- No price argument is passed to `on_bar`; get price/strikes from `get_chain_snapshot()`.
- Multi-leg structures are just independently-managed single contracts sharing a `group_id`.
- For `run_mc`, define a `GRID` dict and keep the strategy 0DTE-compatible.
- Class names become output-folder names — keep them unique and descriptive.
