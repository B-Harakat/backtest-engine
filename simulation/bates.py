"""
Bates model path simulator: Heston stochastic volatility + Merton-style log-normal jumps.

Generates a single intraday (one trading session) spot + vol path on the engine's exact
minute grid (naive America/New_York bar-start timestamps, ``session_open`` -> last bar
before ``session_close`` -- see ``engine.calendar``). Bar *start* times follow the engine's
convention: a bar timestamped 15:59 covers [15:59, 16:00) and 16:00 itself is the settlement
instant, not a tradeable bar.

The SDEs (Bates 1996; jumps follow a Poisson process with normally distributed jump size):

    dS_t / S_t = (mu - lambda * k_bar) * dt + sqrt(V_t) * dB^s_t + (e^J - 1) * dN_t
    dV_t        = kappa * (theta - V_t) * dt + sigma_v * sqrt(V_t) * dB^v_t

    corr(dB^s_t, dB^v_t) = rho
    J ~ Normal(mu_j - 0.5 * sigma_j^2, sigma_j^2)   (so E[e^J] = e^{mu_j})
    k_bar = E[e^J - 1] = e^{mu_j} - 1               (compensator; mu excludes this term)
    N_t  = Poisson(rate = lambda) jump counting process

``mu`` is the *continuous* drift (annualized). The jump compensation ``-lambda * k_bar`` is
applied so the total drift is ``mu`` under the risk-neutral-ish convention used by the
pricer. Parameters are in calibrateable units; defaults are annotated hand-set values meant
to be a reasonable 0DTE XSP starting point, NOT a calibrated fit -- replace with your own
calibration before trusting outputs.

Discretization is Euler-many-step with the **Full Truncation** fix for Heston variance
(V at a time step is floored at 0 for the diffusion term when negative; positivity is
enforced on the value itself), which is the standard stable choice for a CIR process.

Only the Bates layer is here; converting an IV \u2192 price/greeks lives in
``option_pricer.py``, and turning a path into the engine's expected fetch schema lives in
``market_maker.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from engine.calendar import session_close, session_open


def _session_minute_grid(day: date) -> pd.DatetimeIndex:
    """All minute bar-start timestamps for one XNYS session (naive ET), last bar 15:59."""
    start = session_open(day)
    end = session_close(day)
    return pd.date_range(start=start, end=end, freq="min", inclusive="left")


@dataclass(frozen=True)
class BatesParams:
    """
    Parameters for the Bates model. All drift/vol/size params are ANNUALIZED in decimal form
    (0.20 = 20%). ``V_0`` is the initial variance. ``seed`` (optional) seeds the RNG for
    reproducible runs; pass ``None`` for fresh randomness.
    """

    mu: float = 0.0          # continuous (diffusive) drift, annualized. 0 = neutral.
    kappa: float = 3.0       # Heston mean-reversion speed for variance.
    theta: float = 0.04      # Heston long-run variance (0.04 = 20% vol).
    sigma_v: float = 0.30    # Heston vol-of-vol (annualized).
    rho: float = -0.7        # spot/vol correlation. Negative = leverage effect (usual for equities/indices).
    V_0: float = 0.04        # initial variance (0.04 = 20% vol).
    # --- Merton jump component ---
    lambda_: float = 0.5     # jump intensity (expected jumps per year).
    mu_j: float = 0.0        # mean log-jump size (0 = symmetric; positive = upward jumps).
    sigma_j: float = 0.02    # jump size std (log scale). 0.02 is a small "event" jump.


def simulate_session(
    day: date,
    s0: float,
    params: BatesParams,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate one session's spot+vol path on the engine's minute grid, starting from spot
    ``s0`` at ``session_open(day)``. Returns a DataFrame indexed by naive-ET bar timestamps
    with columns ``spot`` and ``vol`` (vol = sqrt(variance) per bar, i.e. instantaneous vol).
    Mirrors ``engine.calendar.trading_minutes`` for a single day (skips non-trading days).

    The opening bar is the initial state; subsequent bars step forward by 1 minute via Euler
    with Full Truncation variance. At each step a Poisson draw decides if a jump lands.
    """
    grid = _session_minute_grid(day)
    if len(grid) == 0:
        return pd.DataFrame(columns=["spot", "vol"])

    rng = np.random.default_rng(seed)
    n = len(grid)

    dt = 1.0 / (365.0 * 24.0 * 60.0)  # one minute, in years

    # Correlated spot/vol Brownian innovations for this path.
    z = rng.standard_normal((n, 2))
    dB_s = z[:, 0]
    dB_v = params.rho * z[:, 0] + np.sqrt(1.0 - params.rho**2) * z[:, 1]

    # Pre-draw infinitesimal Poisson jumps (log-jump magnitude) per step.
    n_jumps = rng.poisson(params.lambda_ * dt, size=n)
    # For any step with >0 expected jumps the Poisson draw gives >=0; we still draw a size
    # so a jump event is non-zero. Using the same magnitudes regardless avoids a subtle bias.
    log_jump = rng.normal(params.mu_j - 0.5 * params.sigma_j**2, params.sigma_j, size=n)

    spot = np.empty(n)
    vol2 = np.empty(n)  # variance
    spot[0] = s0
    vol2[0] = max(params.V_0, 0.0)

    k_bar = np.exp(params.mu_j) - 1.0
    drift_continuous = params.mu - params.lambda_ * k_bar

    for i in range(1, n):
        v_prev = vol2[i - 1]
        v_prev_floored = max(v_prev, 0.0)
        v_new = (
            v_prev
            + params.kappa * (params.theta - v_prev) * dt
            + params.sigma_v * np.sqrt(v_prev_floored) * np.sqrt(dt) * dB_v[i]
        )
        vol2[i] = max(v_new, 0.0)  # Full Truncation: floor the value itself

        v_eff = v_prev_floored
        ret = (
            drift_continuous * dt
            + np.sqrt(v_eff * dt) * dB_s[i]
            + n_jumps[i] * log_jump[i]
        )
        spot[i] = spot[i - 1] * np.exp(ret)

    return pd.DataFrame(
        {"spot": spot, "vol": np.sqrt(vol2)},
        index=grid.rename("timestamp"),
    )
