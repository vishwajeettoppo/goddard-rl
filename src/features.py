"""Observation and reward-shaping features for the Goddard rocket.

The 12 base features and the two burnout features are laid out exactly as in
CARL's ``goddard_code/environment.py``.  All non-base feature modes in this
project intentionally use 14 observations so that checkpoints can be shared
between ablations R2--R6.

Numba entry points operate on scalars and caller-provided output arrays; the
CPython implementation is independent and is useful for inspection/tests.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

# Shared physical constants.  env.py re-exports these to keep one source of
# truth without a circular import between the physics and feature modules.
G = 9.81
M_DRY = 1000.0
RHO_0 = 1.225
H_SCALE = 8500.0
DT = 0.1
MAX_STEPS = 2000
DRAG_SURCHARGE = 1.05
H_SCALE_FIXED = 1000.0
CE_FACTOR = 0.01
PENALTY_COEF = 20.0
HINT_EARLY_FRAC = 0.05

# Numerical target-time mode codes (strings are used by the public API).
TARGET_REV1 = 0
TARGET_REV2_QUADRATIC = 1
TARGET_MONOTONE = 2

# Observation mode codes used inside JIT-compiled loops.
FEATURE_BASE = 0
FEATURE_BURNOUT = 1
FEATURE_RESIDUAL = 2

FEATURE_MODE_CODES = {
    "base": FEATURE_BASE,
    "burnout": FEATURE_BURNOUT,
    "burnout_rev1": FEATURE_BURNOUT,
    "burnout_monotonic": FEATURE_BURNOUT,
    "residual": FEATURE_RESIDUAL,
    "energy": FEATURE_BURNOUT,
}
TARGET_MODE_CODES = {
    "rev1": TARGET_REV1,
    "rev2_quadratic": TARGET_REV2_QUADRATIC,
    "monotone_calibrated": TARGET_MONOTONE,
}

# A three-parameter, monotone saturating Weibull curve.  These coefficients
# interpolate the training-distribution calibration points (x=m_fuel_0/1000)
# (0.5, 35.5), (2, 118), and (4, 172), with no downturn or negative tail.
_MONOTONE_LIMIT = 205.7573422649152
_MONOTONE_TAU = 2.317884139728519
_MONOTONE_POWER = 1.0848634217102737


def _check_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


def feature_code(feature_mode: str) -> int:
    """Return the JIT-friendly integer code for an observation mode."""

    try:
        return FEATURE_MODE_CODES[feature_mode]
    except KeyError as exc:
        choices = ", ".join(sorted(FEATURE_MODE_CODES))
        raise ValueError(f"Unknown feature_mode {feature_mode!r}; choose {choices}") from exc


def target_code(target_mode: str) -> int:
    """Return the JIT-friendly integer code for a burnout-target mode."""

    try:
        return TARGET_MODE_CODES[target_mode]
    except KeyError as exc:
        choices = ", ".join(sorted(TARGET_MODE_CODES))
        raise ValueError(f"Unknown target_mode {target_mode!r}; choose {choices}") from exc


def obs_dim(feature_mode: str = "base") -> int:
    """Observation width: 12 for R0/R1 and 14 for every later ablation."""

    return 12 if feature_code(feature_mode) == FEATURE_BASE else 14


@njit(cache=True)
def rho_numba(h: float) -> float:
    """Exponential atmosphere, using sea-level density below ground."""

    return RHO_0 * math.exp(-max(h, 0.0) / H_SCALE)


@njit(cache=True)
def dynamic_pressure_numba(h: float, v: float) -> float:
    rho = rho_numba(h)
    return 0.5 * rho * v * v


@njit(cache=True)
def t_target_monotone_numba(m_fuel_0: float, T_max: float, Q_max: float) -> float:
    """Monotone saturating target calibrated on training-fuel rollouts.

    T_max and Q_max remain explicit arguments so all target laws share one
    calling convention.  The fitted calibration is a function of initial fuel;
    c_e=0.01*T_max makes unconstrained full-burn time independent of T_max.
    """

    # Keep the signature physically explicit; these values do not alter the
    # three calibration points supplied for this ablation.
    _ = T_max
    _ = Q_max
    x = max(m_fuel_0, 0.0) / 1000.0
    if x <= 0.0:
        return 0.0
    log_z = _MONOTONE_POWER * (math.log(x) - math.log(_MONOTONE_TAU))
    # exp(-z) is below machine precision well before this branch.  The branch
    # also prevents overflow for arbitrarily large, finite fuel inputs.
    if log_z > 7.0:
        return _MONOTONE_LIMIT
    z = math.exp(log_z)
    return _MONOTONE_LIMIT * (-math.expm1(-z))


def t_target_monotone(m_fuel_0: float, T_max: float = 200000.0, Q_max: float = 30000.0) -> float:
    """Python counterpart of :func:`t_target_monotone_numba`."""

    _check_positive("T_max", T_max)
    _check_positive("Q_max", Q_max)
    if not math.isfinite(m_fuel_0) or m_fuel_0 < 0.0:
        raise ValueError("m_fuel_0 must be finite and non-negative")
    return float(t_target_monotone_numba(float(m_fuel_0), float(T_max), float(Q_max)))


@njit(cache=True)
def burn_target_time_numba(
    m_fuel_0: float,
    T_max: float,
    Q_max: float,
    mode: int,
) -> float:
    c_e = CE_FACTOR * T_max
    if mode == TARGET_REV1:
        return 4.0 * m_fuel_0 * c_e / T_max
    if mode == TARGET_MONOTONE:
        return t_target_monotone_numba(m_fuel_0, T_max, Q_max)
    x = m_fuel_0 / 1000.0
    # The max(10, ...) guard does not alter CARL's formula anywhere in its
    # training range (300--5000 kg), but prevents a negative tail if extrapolated.
    return max(10.0, 75.0 * x - 8.0 * x * x)


def burn_target_time(
    m_fuel_0: float,
    T_max: float = 200000.0,
    Q_max: float = 30000.0,
    mode: str = "rev2_quadratic",
) -> float:
    """Compute a burnout target using ``rev1``, CARL's quadratic, or monotone law."""

    _check_positive("T_max", T_max)
    _check_positive("Q_max", Q_max)
    if not math.isfinite(m_fuel_0) or m_fuel_0 < 0.0:
        raise ValueError("m_fuel_0 must be finite and non-negative")
    return float(
        burn_target_time_numba(
            float(m_fuel_0), float(T_max), float(Q_max), target_code(mode)
        )
    )


def default_target_mode(feature_mode: str) -> str:
    """Target law implied by a feature mode unless explicitly overridden."""

    feature_code(feature_mode)  # validate
    if feature_mode == "burnout_rev1":
        return "rev1"
    if feature_mode == "burnout_monotonic":
        return "monotone_calibrated"
    return "rev2_quadratic"


def audit_quadratic_target() -> dict[str, Any]:
    """Explain the unsafe extrapolation of CARL's unclipped quadratic target.

    For ``f(x)=75*x-8*x**2``, ``f'(x)=75-16*x``.  Thus the vertex is at
    x=4.6875 and f decreases above it; the positive root is x=9.375, after
    which the unclipped target is negative.  The guarded implementation clips
    to 10 seconds, while the monotone law remains positive and non-decreasing.
    """

    return {
        "formula": "t_target = max(10, 75*x - 8*x^2), x = m_fuel_0/1000",
        "derivative": "75 - 16*x",
        "vertex_x": 75.0 / 16.0,
        "decreases_for_x_above": 75.0 / 16.0,
        "positive_root_x": 75.0 / 8.0,
        "negative_unclipped_for_x_above": 75.0 / 8.0,
        "vertex_m_fuel_0": 1000.0 * 75.0 / 16.0,
        "negative_unclipped_above_m_fuel_0": 1000.0 * 75.0 / 8.0,
        "unclipped_at_x_10": 75.0 * 10.0 - 8.0 * 10.0**2,
        "guarded_at_x_10": 10.0,
        "monotone_limit_seconds": _MONOTONE_LIMIT,
    }


@njit(cache=True, inline="always")
def _constant_hint_numba(
    m_fuel: float,
    t: float,
    t_target: float,
    T_max: float,
) -> float:
    if m_fuel <= 0.0:
        return 0.0
    if t < HINT_EARLY_FRAC * t_target:
        return 1.0
    c_e = CE_FACTOR * T_max
    dt_rem = max(t_target - t, DT)
    hint = m_fuel * c_e / (T_max * dt_rem)
    return min(1.0, max(0.0, hint))


@njit(cache=True)
def fill_observation_numba(
    h: float,
    v: float,
    m: float,
    m_fuel_0: float,
    T_max: float,
    cd: float,
    Q_max: float,
    t: float,
    t_target: float,
    mode: int,
    hint_override: float,
    out: np.ndarray,
) -> None:
    """Fill a 12- or 14-element float32 observation without allocating.

    ``hint_override >= 0`` replaces the usual constant-throttle hint.  The
    batched residual environment uses this to insert its singular-arc hint.
    Pass a negative value for the standard hint.
    """

    rho = rho_numba(h)
    q = 0.5 * rho * v * v
    q_ratio = q / Q_max
    m_fuel = max(m - M_DRY, 0.0)
    c_e = CE_FACTOR * T_max

    out[0] = rho / RHO_0
    out[1] = v / math.sqrt(2.0 * Q_max / RHO_0)
    out[2] = q_ratio
    out[3] = 1.0 - q_ratio
    out[4] = m_fuel / m_fuel_0
    out[5] = M_DRY / m
    out[6] = T_max / (m * G)

    mass_ratio = max(m / M_DRY, 1.0e-12)
    e_alt = h + v * v / (2.0 * G) + (c_e / G) * math.log(mass_ratio)
    e_0 = (c_e / G) * math.log((M_DRY + m_fuel_0) / M_DRY)
    out[7] = e_alt / max(e_0, 1.0)
    out[8] = math.log(Q_max / 30000.0)
    out[9] = math.log(T_max / 200000.0)
    out[10] = math.log(m_fuel_0 / 1500.0)
    out[11] = cd

    if mode != FEATURE_BASE:
        denominator = max(t_target, 1.0e-9)
        t_rem = min(1.0, max(-1.0, (t_target - t) / denominator))
        if hint_override >= 0.0:
            hint = min(1.0, hint_override)
        else:
            hint = _constant_hint_numba(m_fuel, t, t_target, T_max)
        out[12] = t_rem
        out[13] = hint

    width = 12 if mode == FEATURE_BASE else 14
    for j in range(width):
        out[j] = min(10.0, max(-10.0, out[j]))


@njit(cache=True)
def build_obs_numba(
    h: float,
    v: float,
    m: float,
    m_fuel_0: float,
    T_max: float,
    cd: float,
    Q_max: float,
    t: float = 0.0,
    t_target: float = 10.0,
    mode: int = FEATURE_BASE,
    hint_override: float = -1.0,
) -> np.ndarray:
    """Allocated Numba observation builder for single-state compiled callers."""

    width = 12 if mode == FEATURE_BASE else 14
    out = np.empty(width, dtype=np.float32)
    fill_observation_numba(
        h, v, m, m_fuel_0, T_max, cd, Q_max, t, t_target, mode, hint_override, out
    )
    return out


def build_observation(
    h: float,
    v: float,
    m: float,
    m_fuel_0: float,
    T_max: float,
    cd: float,
    Q_max: float,
    t: float = 0.0,
    t_target: float | None = None,
    feature_mode: str = "base",
    hint_override: float | None = None,
    target_mode: str | None = None,
) -> np.ndarray:
    """Build one observation with transparent NumPy/Python arithmetic.

    In ``residual`` mode, omitting ``hint_override`` computes the singular-arc
    law lazily from ``src.env``; supplying it avoids that optional import.
    """

    values = (h, v, m, m_fuel_0, T_max, cd, Q_max, t)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Observation inputs must be finite")
    _check_positive("m_fuel_0", float(m_fuel_0))
    _check_positive("T_max", float(T_max))
    _check_positive("Q_max", float(Q_max))
    if m <= 0.0:
        raise ValueError("m must be positive")
    code = feature_code(feature_mode)
    if t_target is None:
        selected_target = target_mode or default_target_mode(feature_mode)
        t_target = burn_target_time(m_fuel_0, T_max, Q_max, selected_target)
    if not math.isfinite(t_target) or t_target < 0.0:
        raise ValueError("t_target must be finite and non-negative")

    rho = RHO_0 * math.exp(-max(h, 0.0) / H_SCALE)
    q = 0.5 * rho * v * v
    q_ratio = q / Q_max
    m_fuel = max(m - M_DRY, 0.0)
    c_e = CE_FACTOR * T_max
    e_alt = h + v * v / (2.0 * G) + (c_e / G) * math.log(max(m / M_DRY, 1.0e-12))
    e_0 = (c_e / G) * math.log((M_DRY + m_fuel_0) / M_DRY)
    result = [
        rho / RHO_0,
        v / math.sqrt(2.0 * Q_max / RHO_0),
        q_ratio,
        1.0 - q_ratio,
        m_fuel / m_fuel_0,
        M_DRY / m,
        T_max / (m * G),
        e_alt / max(e_0, 1.0),
        math.log(Q_max / 30000.0),
        math.log(T_max / 200000.0),
        math.log(m_fuel_0 / 1500.0),
        cd,
    ]
    if code != FEATURE_BASE:
        t_rem = float(np.clip((t_target - t) / max(t_target, 1.0e-9), -1.0, 1.0))
        if hint_override is not None:
            hint = float(np.clip(hint_override, 0.0, 1.0))
        elif code == FEATURE_RESIDUAL:
            # Lazy import avoids a module cycle; Numba callers pass the hint in.
            from .env import singular_arc_u

            hint = singular_arc_u(h, v, m, T_max, cd, Q_max)
        elif m_fuel <= 0.0:
            hint = 0.0
        elif t < HINT_EARLY_FRAC * t_target:
            hint = 1.0
        else:
            hint = m_fuel * c_e / (T_max * max(t_target - t, DT))
            hint = float(np.clip(hint, 0.0, 1.0))
        result.extend((t_rem, hint))
    return np.clip(np.asarray(result, dtype=np.float32), -10.0, 10.0)


# Concise public alias used by evaluation/inspection code.
build_obs = build_observation


@njit(cache=True)
def potential_numba(
    h: float,
    m: float,
    m_fuel_0: float,
    t: float,
    t_target: float,
    terminal: bool = False,
    c_pot: float = 1.0,
    c_burn: float = 1.0,
    h_scale: float = H_SCALE,
) -> float:
    """Potential for R5; exactly zero in every terminal state."""

    if terminal:
        return 0.0
    m_fuel = max(m - M_DRY, 0.0)
    fuel_frac = m_fuel / m_fuel_0
    altitude_term = 1.0 - math.exp(-h / h_scale)
    early = max(0.0, 1.0 - t / max(t_target, 1.0e-9))
    burn_indicator = 1.0 if m_fuel > 0.0 else 0.0
    return c_pot * fuel_frac * altitude_term - c_burn * burn_indicator * early * early


def potential(
    h: float,
    m: float,
    m_fuel_0: float,
    t: float,
    t_target: float,
    terminal: bool = False,
    c_pot: float = 1.0,
    c_burn: float = 1.0,
    h_scale: float = H_SCALE,
) -> float:
    """Python form of the physically grounded R5 potential function."""

    _check_positive("m_fuel_0", float(m_fuel_0))
    _check_positive("h_scale", float(h_scale))
    if terminal:
        return 0.0
    m_fuel = max(float(m) - M_DRY, 0.0)
    fuel_frac = m_fuel / m_fuel_0
    altitude_term = 1.0 - math.exp(-float(h) / h_scale)
    early = max(0.0, 1.0 - float(t) / max(float(t_target), 1.0e-9))
    indicator = 1.0 if m_fuel > 0.0 else 0.0
    return float(c_pot * fuel_frac * altitude_term - c_burn * indicator * early * early)


# Descriptive aliases for callers that prefer explicit R5 terminology.
potential_energy = potential
potential_energy_numba = potential_numba


__all__ = [
    "CE_FACTOR",
    "DRAG_SURCHARGE",
    "DT",
    "FEATURE_BASE",
    "FEATURE_BURNOUT",
    "FEATURE_RESIDUAL",
    "HINT_EARLY_FRAC",
    "H_SCALE",
    "H_SCALE_FIXED",
    "MAX_STEPS",
    "M_DRY",
    "PENALTY_COEF",
    "RHO_0",
    "TARGET_MONOTONE",
    "TARGET_REV1",
    "TARGET_REV2_QUADRATIC",
    "G",
    "audit_quadratic_target",
    "build_obs",
    "build_obs_numba",
    "build_observation",
    "burn_target_time",
    "burn_target_time_numba",
    "default_target_mode",
    "dynamic_pressure_numba",
    "feature_code",
    "fill_observation_numba",
    "obs_dim",
    "potential",
    "potential_energy",
    "potential_energy_numba",
    "potential_numba",
    "rho_numba",
    "t_target_monotone",
    "t_target_monotone_numba",
    "target_code",
]
