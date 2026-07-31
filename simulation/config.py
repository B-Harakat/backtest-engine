"""
TOML config loader for the synthetic-market Monte Carlo harness.

Loads the optional `--config` file (e.g. `mc_config.toml`) into the parameter dataclasses the
harness uses (`BatesParams`, `SpreadConfig`, `StrikeGrid`) plus a small `runtime` dict. CLI flags
take precedence over file values; file values take precedence over the dataclasses' defaults.
Model/strategy agnostic — it maps a fixed TOML schema onto the market-model dataclasses.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_RUNTIME_DEFAULTS = {
    "n_paths": 100,
    "starting_cash": 10_000.0,
    "multiplier": 100,
    "settlement_time": "16:00",
    "cost_model": "ibkr",
    "seed": 0,
}


def _section(conf: dict, name: str) -> dict:
    sec = conf.get(name, {})
    return {k: v for k, v in sec.items() if v is not None}


def load_config(path: Path) -> dict[str, Any]:
    """
    Read a TOML config file and return three keys usable by run_mc:

      * ``bates``      -> dict of BatesParams field overrides (may be empty).
      * ``spread``     -> dict of SpreadConfig field overrides (may be empty).
      * ``strike_grid``-> dict of StrikeGrid field overrides (may be empty).
      * ``runtime``    -> dict of CLI-default knobs (n_paths, starting_cash, multiplier,
                          settlement_time, cost_model, seed) merged over built-in defaults.

    Empty/missing sections collapse to the dataclasses' own defaults when the caller applies
    them. The loader does NOT instantiate the dataclasses; that stays with the caller so callers
    can layer CLI overrides on top (dict.update) before constructing.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    runtime = dict(_RUNTIME_DEFAULTS)
    runtime.update({k: v for k, v in _section(raw, "runtime").items() if v is not None})
    return {
        "bates": _section(raw, "bates"),
        "spread": _section(raw, "spread"),
        "strike_grid": _section(raw, "strike_grid"),
        "runtime": runtime,
    }

