"""
Transaction cost modeling — broker-style order costs applied to every order fill.

The engine fills orders at a clean book price (no slippage); real brokerages charge per-contract
commissions plus regulatory/clearing fees. This module models those costs so a backtest's P&L
reflects actual trading friction.

Default model: IBKR Pro "Tiered" U.S. options pricing (first tier, ≤10,000 contracts/month), the
schedule Interactive Brokers applies to index options like XSP. All figures are USD. The pricing
changes occasionally; tiered per-volume breaks below 0.65/contract would need monthly
running-total machinery (out of scope; this assumes the base ≤10k/month tier).

Summary of the modeled IBKR tiered schedule (per contract, per order):
    - IBKR commission by premium:
          premium <  $0.05  ->  $0.25 /contract
          $0.05 <= premium <  $0.10 ->  $0.50 /contract
          premium >= $0.10  ->  $0.65 /contract
    - Commission minimum per ORDER: $1.00
    - Third-party fees (always passed through, NOT subject to the order minimum):
          OCC clearing           0.02500 /contract
          Options Regulatory Fee 0.02295 /contract
          FINRA CAT              0.00030 /contract

Cboe's SPX execution surcharge is deliberately NOT included: it applies to SPX/SPXW, not to XSP
(Mini-SPX), which the engine treats as its cash-settled index product.

A model is just a callable `(side, qty, fill_price) -> total_cost_dollars`. The default
`ibkr_standard_option_cost` is wired in by the runner; pass `cost_fn=None` to disable costs.
"""

from __future__ import annotations

from typing import Callable, Optional

# IBKR base commission per contract, keyed by premium thresholds (USD).
_IBKR_PREMIUM_TIERS: list[tuple[float, float]] = [
    (0.05, 0.25),   # premium <  $0.05
    (0.10, 0.50),   # premium >= $0.05 and < $0.10
    (float("inf"), 0.65),  # premium >= $0.10
]

# Per-contract third-party fees (USD), always passed through.
_OCC_CLEARING_FEE = 0.02500
_OPTIONS_REGULATORY_FEE = 0.02295
_FINRA_CAT_FEE = 0.00030

_MIN_COMMISSION_PER_ORDER = 1.00


def _ibkr_commission_per_contract(fill_price: float) -> float:
    """IBKR tiered commission per contract based on the option's premium (fill price)."""
    for threshold, rate in _IBKR_PREMIUM_TIERS:
        if fill_price < threshold:
            return rate
    return _IBKR_PREMIUM_TIERS[-1][1]


def _third_party_fees_per_contract() -> float:
    return _OCC_CLEARING_FEE + _OPTIONS_REGULATORY_FEE + _FINRA_CAT_FEE


def ibkr_standard_option_cost(
    side: str,
    qty: int,
    fill_price: float,
) -> float:
    """
    Total all-in transaction cost (USD) for a single order fill of `qty` contracts.

    Asset-agnostic: IBKR charges US listed options uniformly per contract regardless of the
    underlying symbol. The per-order minimum applies to the IBKR commission only; third-party
    fees are always charged per contract on top. Cboe's index execution surcharge on SPX/SPXW is
    intentionally not modeled (it doesn't apply to XSP/equity/ETF options); backtesting standard
    SPX itself would understate true costs slightly.
    """
    per_contract_commission = _ibkr_commission_per_contract(fill_price)
    commission = max(per_contract_commission * qty, _MIN_COMMISSION_PER_ORDER)
    third_party = _third_party_fees_per_contract() * qty
    return commission + third_party


CostFn = Callable[[str, int, float], float]

_DEFAULT_COST_FN: Callable[[Optional[str]], Optional[CostFn]] = {
    "ibkr": ibkr_standard_option_cost,
}


def cost_model_from_name(name: Optional[str]) -> Optional[CostFn]:
    """
    Resolve a configured cost-model name to a callable (or None for no costs).
    Case-insensitive. Recognized: "ibkr" (default), "none"/"" /None -> no costs.
    """
    if not name:
        return None
    key = name.strip().lower()
    if key in ("none", "off", "no", "0"):
        return None
    if key in _DEFAULT_COST_FN:
        return _DEFAULT_COST_FN[key]
    raise ValueError(
        f"Unknown cost model {name!r} — supported: {', '.join(_DEFAULT_COST_FN)} or 'none'"
    )
