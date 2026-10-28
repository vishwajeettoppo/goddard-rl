"""Shared RK4 Goddard-rocket physics and scalar/batched environments.

The default stepper is the common evaluator used by every baseline and RL
ablation: fourth-order integration, exact fuel-out splitting, and binary-search
max-Q thrust limiting with a 5% drag surcharge on clamped steps.  The batched
implementation runs the same scalar Numba kernel inside a parallel ``prange``;
there is deliberately no second, approximate vectorized physics implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numba import njit, prange

from .features import (
    CE_FACTOR,
    DRAG_SURCHARGE,
    DT,
    FEATURE_RESIDUAL,
    H_SCALE,
    H_SCALE_FIXED,
    HINT_EARLY_FRAC,
    M_DRY,
    MAX_STEPS,
    PENALTY_COEF,
    RHO_0,
    G,
    build_obs_numba,
    build_observation,
    burn_target_time,
    burn_target_time_numba,
    default_target_mode,
    dynamic_pressure_numba,
    feature_code,
    fill_observation_numba,
    obs_dim,
    potential,
    potential_numba,
    rho_numba,
    target_code,
)

# Domain randomization from configs/train.yaml.
M_FUEL_RANGE = (300.0, 5000.0)
T_MAX_RANGE = (100000.0, 300000.0)
CD_RANGE = (1.0, 2.0)
Q_MAX_RANGE = (3500.0, 60000.0)
TW_MIN = 1.1
EVAL_T_MAX = 200000.0
EVAL_CD = 1.5
N_BISECT = 8

EVAL_M_FUEL_GRID = (500.0, 2000.0, 4000.0)
EVAL_Q_MAX_GRID = (5000.0, 7500.0, 10000.0, 15000.0, 30000.0, 45000.0)
EVAL_PAIRS = tuple((mf, q) for mf in EVAL_M_FUEL_GRID for q in EVAL_Q_MAX_GRID)

REWARD_MODES = {"base": 0, "burnout_penalty": 1, "potential_energy": 2}


# ---------------------------------------------------------------------------
# Scalar JIT physics
# ---------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _acceleration_numba(h: float, v: float, m: float, thrust: float, cd: float) -> float:
    rho = rho_numba(h)
    drag = cd * rho * v * abs(v)
    return (thrust - drag) / m - G


@njit(cache=True)
def rk4_step_numba(
    h: float,
    v: float,
    m: float,
    thrust: float,
    cd: float,
    c_e: float,
    dt: float,
) -> tuple[float, float, float]:
    """One scalar RK4 step, including wet-mass integration while burning."""

    burning = thrust > 0.0 and m > M_DRY
    if m <= M_DRY:
        m = M_DRY
        thrust = 0.0
        burning = False
    elif thrust <= 0.0:
        thrust = 0.0
    mdot = -thrust / c_e if burning else 0.0

    k1_h = v
    k1_v = _acceleration_numba(h, v, m, thrust, cd)
    k1_m = mdot

    h2 = h + 0.5 * dt * k1_h
    v2 = v + 0.5 * dt * k1_v
    m2 = m + 0.5 * dt * k1_m
    k2_h = v2
    k2_v = _acceleration_numba(h2, v2, m2, thrust, cd)
    k2_m = mdot

    h3 = h + 0.5 * dt * k2_h
    v3 = v + 0.5 * dt * k2_v
    m3 = m + 0.5 * dt * k2_m
    k3_h = v3
    k3_v = _acceleration_numba(h3, v3, m3, thrust, cd)
    k3_m = mdot

    h4 = h + dt * k3_h
    v4 = v + dt * k3_v
    m4 = m + dt * k3_m
    k4_h = v4
    k4_v = _acceleration_numba(h4, v4, m4, thrust, cd)
    k4_m = mdot

    h_next = h + (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)
    v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    m_next = m + (dt / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m)
    if burning:
        m_next = max(m_next, M_DRY)
    elif m <= M_DRY:
        m_next = M_DRY
    # With zero thrust and fuel remaining, dm/dt=0 and wet mass is preserved.
    return h_next, v_next, m_next


@njit(cache=True)
def advance_fuel_exact_numba(
    h: float,
    v: float,
    m: float,
    thrust: float,
    cd: float,
    c_e: float,
    dt: float = DT,
) -> tuple[float, float, float]:
    """Advance one step, splitting exactly at fuel exhaustion when necessary."""

    if thrust <= 0.0 or m <= M_DRY:
        return rk4_step_numba(h, v, m, 0.0, cd, c_e, dt)
    t_burn = (m - M_DRY) * c_e / thrust
    if t_burn >= dt:
        return rk4_step_numba(h, v, m, thrust, cd, c_e, dt)
    if t_burn > 0.0:
        h_next, v_next, m_next = rk4_step_numba(h, v, m, thrust, cd, c_e, t_burn)
        m_next = M_DRY
        return rk4_step_numba(h_next, v_next, m_next, 0.0, cd, c_e, dt - t_burn)
    return rk4_step_numba(h, v, m, 0.0, cd, c_e, dt)


@njit(cache=True)
def physics_step_numba(
    h: float,
    v: float,
    m: float,
    u_cmd: float,
    T_max: float,
    cd: float,
    Q_max: float,
    c_e: float,
    dt: float = DT,
    n_bisect: int = N_BISECT,
) -> tuple[float, float, float, float, bool]:
    """Shared max-Q-clamped step; return state, applied thrust, and clamp flag."""

    u = min(1.0, max(0.0, u_cmd))
    T_des = 0.0 if m <= M_DRY else u * T_max
    h_next, v_next, m_next = advance_fuel_exact_numba(h, v, m, T_des, cd, c_e, dt)
    violated = T_des > 0.0 and dynamic_pressure_numba(h_next, v_next) > Q_max
    if not violated:
        return h_next, v_next, m_next, T_des, False

    lo = 0.0
    hi = T_des
    cd_eff = DRAG_SURCHARGE * cd
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        h_mid, v_mid, _ = advance_fuel_exact_numba(h, v, m, mid, cd_eff, c_e, dt)
        if dynamic_pressure_numba(h_mid, v_mid) <= Q_max:
            lo = mid
        else:
            hi = mid
    h_next, v_next, m_next = advance_fuel_exact_numba(h, v, m, lo, cd_eff, c_e, dt)
    return h_next, v_next, m_next, lo, True


@njit(cache=True)
def u_cap_safe_numba(
    h: float,
    v: float,
    m: float,
    T_max: float,
    cd: float,
    Q_max: float,
    safety: float = 0.99,
) -> float:
    """CARL's one-step lookahead throttle cap."""

    if m <= M_DRY:
        return 0.0
    rho = rho_numba(h)
    v_lim = math.sqrt(2.0 * Q_max / rho)
    drag = cd * rho * v * abs(v)
    T_needed = m * ((v_lim - v) / DT + G) + drag
    u_max = safety * T_needed / T_max
    return min(1.0, max(0.0, u_max))


@njit(cache=True)
def singular_arc_u_numba(
    h: float,
    v: float,
    m: float,
    T_max: float,
    cd: float,
    Q_max: float = -1.0,
    use_cap: bool = False,
    t: float = -1.0,
    t_switch: float = -1.0,
) -> float:
    """JIT Pontryagin switching surface and equation-8 feedback law."""

    if m <= M_DRY:
        return 0.0
    c_e = CE_FACTOR * T_max
    rho = rho_numba(h)
    drag = cd * rho * v * abs(v)
    surface = drag * (1.0 + max(v, 0.0) / c_e) - m * G
    full_thrust = (
        t < t_switch if t_switch >= 0.0 and t >= 0.0 else surface < 0.0
    )
    if full_thrust:
        u = 1.0
    else:
        numerator = (c_e + v) * c_e * (2.0 * G + (v * v) / H_SCALE)
        denominator = 2.0 * c_e * c_e + 4.0 * c_e * v + v * v
        if abs(denominator) < 1.0e-12:
            u = 0.0
        else:
            T_sing = drag + m * numerator / denominator
            u = min(1.0, max(0.0, T_sing / T_max))
    if use_cap and Q_max > 0.0:
        u = min(u, u_cap_safe_numba(h, v, m, T_max, cd, Q_max))
    return u


# ---------------------------------------------------------------------------
# Transparent Python helpers and wrappers
# ---------------------------------------------------------------------------


def rho(h: float) -> float:
    return RHO_0 * math.exp(-max(float(h), 0.0) / H_SCALE)


def dyn_pressure(h: float, v: float) -> float:
    return 0.5 * rho(h) * float(v) ** 2


dynamic_pressure = dyn_pressure


def rk4_step(
    state: Sequence[float],
    thrust_force: float,
    cd: float,
    c_e: float,
    dt: float = DT,
) -> np.ndarray:
    """Array-friendly wrapper around the exact scalar RK4 kernel."""

    if len(state) != 3:
        raise ValueError("state must contain (h, v, m)")
    result = rk4_step_numba(
        float(state[0]), float(state[1]), float(state[2]),
        float(thrust_force), float(cd), float(c_e), float(dt),
    )
    return np.asarray(result, dtype=np.float64)


def advance_fuel_exact(
    state: Sequence[float],
    thrust_force: float,
    cd: float,
    c_e: float,
    dt: float = DT,
) -> np.ndarray:
    """Array-friendly exact fuel-out-splitting integrator."""

    if len(state) != 3:
        raise ValueError("state must contain (h, v, m)")
    result = advance_fuel_exact_numba(
        float(state[0]), float(state[1]), float(state[2]),
        float(thrust_force), float(cd), float(c_e), float(dt),
    )
    return np.asarray(result, dtype=np.float64)


def u_cap_safe(
    h: float,
    v: float,
    m: float,
    T_max: float,
    cd: float,
    Q_max: float,
    safety: float = 0.99,
) -> float:
    """Python implementation of CARL's lookahead cap (not the env clamp)."""

    if m <= M_DRY:
        return 0.0
    rho_value = rho(h)
    v_lim = math.sqrt(2.0 * Q_max / rho_value)
    drag = cd * rho_value * v * abs(v)
    T_needed = m * ((v_lim - v) / DT + G) + drag
    return float(np.clip(safety * T_needed / T_max, 0.0, 1.0))


def singular_arc_u(
    h: float,
    v: float,
    m: float,
    T_max: float,
    cd: float,
    Q_max: float | None = None,
    use_cap: bool = False,
    t: float | None = None,
    t_switch: float | None = None,
) -> float:
    """Unconstrained-optimal feedback with surface or explicit-time switching.

    If ``t_switch`` is supplied, ``t`` selects the full-thrust arc for
    ``t < t_switch``.  Otherwise the Pontryagin switching surface is used.
    """

    if m <= M_DRY:
        return 0.0
    if t_switch is not None and t is None:
        raise ValueError("t is required when t_switch is supplied")
    c_e = CE_FACTOR * T_max
    rho_value = rho(h)
    drag = cd * rho_value * v * abs(v)
    surface = drag * (1.0 + max(v, 0.0) / c_e) - m * G
    full_thrust = t < t_switch if t_switch is not None else surface < 0.0
    if full_thrust:
        u = 1.0
    else:
        numerator = (c_e + v) * c_e * (2.0 * G + v * v / H_SCALE)
        denominator = 2.0 * c_e * c_e + 4.0 * c_e * v + v * v
        if abs(denominator) < 1.0e-12:
            u = 0.0
        else:
            T_sing = drag + m * numerator / denominator
            u = float(np.clip(T_sing / T_max, 0.0, 1.0))
    if use_cap and Q_max is not None:
        u = min(u, u_cap_safe(h, v, m, T_max, cd, Q_max))
    return float(u)


def is_eval_pair(m_fuel_0: float, Q_max: float, relative_radius: float = 0.0) -> bool:
    """Whether a sample is an excluded held-out ``(fuel, Q_max)`` pair."""

    for eval_fuel, eval_q in EVAL_PAIRS:
        if relative_radius <= 0.0:
            if m_fuel_0 == eval_fuel and Q_max == eval_q:
                return True
        elif (
            abs(m_fuel_0 - eval_fuel) <= relative_radius * eval_fuel
            and abs(Q_max - eval_q) <= relative_radius * eval_q
        ):
            return True
    return False


def _draw_params(
    rng: np.random.Generator,
    exclude_eval: bool,
    exclusion_radius: float,
) -> dict[str, float]:
    for _ in range(10000):
        m_fuel_0 = float(np.exp(rng.uniform(math.log(M_FUEL_RANGE[0]), math.log(M_FUEL_RANGE[1]))))
        T_max = float(np.exp(rng.uniform(math.log(T_MAX_RANGE[0]), math.log(T_MAX_RANGE[1]))))
        cd = float(rng.uniform(*CD_RANGE))
        Q_max = float(np.exp(rng.uniform(math.log(Q_MAX_RANGE[0]), math.log(Q_MAX_RANGE[1]))))
        if T_max / ((M_DRY + m_fuel_0) * G) < TW_MIN:
            continue
        if exclude_eval and is_eval_pair(m_fuel_0, Q_max, exclusion_radius):
            continue
        return {"m_fuel_0": m_fuel_0, "T_max": T_max, "cd": cd, "Q_max": Q_max}
    raise RuntimeError("Parameter rejection sampling failed after 10000 draws")


def sample_params(
    rng: np.random.Generator | None = None,
    *,
    exclude_eval: bool = True,
    exclusion_radius: float = 0.0,
) -> dict[str, float]:
    """Sample the training distribution with thrust-to-weight and eval guards."""

    if rng is None:
        rng = np.random.default_rng()
    if exclusion_radius < 0.0:
        raise ValueError("exclusion_radius must be non-negative")
    return _draw_params(rng, exclude_eval, exclusion_radius)


def _validate_params(params: Mapping[str, float]) -> dict[str, float]:
    required = ("m_fuel_0", "T_max", "cd", "Q_max")
    try:
        result = {name: float(params[name]) for name in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"params must define numeric {required}") from exc
    for name, value in result.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    return result


def _reward_code(reward_mode: str) -> int:
    try:
        return REWARD_MODES[reward_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown reward_mode {reward_mode!r}; choose {sorted(REWARD_MODES)}") from exc


def h_ref_ballistic(m_fuel_0: float, T_max: float) -> float:
    c_e = CE_FACTOR * T_max
    delta_v = c_e * math.log((M_DRY + m_fuel_0) / M_DRY)
    return delta_v * delta_v / (2.0 * G)


def full_burn_time(m_fuel_0: float, T_max: float) -> float:
    return m_fuel_0 * (CE_FACTOR * T_max) / T_max


# ---------------------------------------------------------------------------
# Single environment
# ---------------------------------------------------------------------------


class GoddardEnv:
    """Gym-like scalar environment using the shared deterministic evaluator."""

    ACT_DIM = 1

    def __init__(
        self,
        seed: int | None = None,
        params: Mapping[str, float] | None = None,
        *,
        feature_mode: str = "base",
        reward_mode: str = "base",
        target_mode: str | None = None,
        use_actor_cap: bool = False,
        n_bisect: int = N_BISECT,
        dt: float = DT,
        max_steps: int | None = MAX_STEPS,
        gamma: float = 1.0,
        burnout_penalty_coef: float = PENALTY_COEF,
        potential_coeff: float = 1.0,
        burn_potential_coeff: float = 1.0,
        potential_h_scale: float = H_SCALE,
        record_trajectory: bool = False,
        exclude_eval: bool = True,
        exclusion_radius: float = 0.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self._fixed_params = _validate_params(params) if params is not None else None
        self.feature_mode = feature_mode
        self._feature_code = feature_code(feature_mode)
        self.OBS_DIM = obs_dim(feature_mode)
        self.obs_dim = self.OBS_DIM
        self.reward_mode = reward_mode
        self._reward_code = _reward_code(reward_mode)
        self.target_mode = target_mode or default_target_mode(feature_mode)
        self._target_code = target_code(self.target_mode)
        self.use_actor_cap = bool(use_actor_cap)
        if n_bisect < 0:
            raise ValueError("n_bisect must be non-negative")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive or None")
        self.n_bisect = int(n_bisect)
        self.dt = float(dt)
        self.max_steps = max_steps
        self.gamma = float(gamma)
        self.burnout_penalty_coef = float(burnout_penalty_coef)
        self.potential_coeff = float(potential_coeff)
        self.burn_potential_coeff = float(burn_potential_coeff)
        self.potential_h_scale = float(potential_h_scale)
        self.record_trajectory = bool(record_trajectory)
        self.exclude_eval = bool(exclude_eval)
        self.exclusion_radius = float(exclusion_radius)

        self.params: dict[str, float] | None = None
        self.state = np.zeros(3, dtype=np.float64)
        self.m_fuel = 0.0
        self.launched = False
        self.done = False
        self.terminated = False
        self.truncated = False
        self.h_apogee = 0.0
        self.max_h = 0.0
        self.n_steps = 0
        self.n_clamp = 0
        self.t_now = 0.0
        self.t_target = 0.0
        self.h_ref = 0.0
        self.burnout_penalty_applied = False
        self.burnout_penalty = 0.0
        self.burnout_time: float | None = None
        self.trajectory: list[dict[str, Any]] = []

    def get_obs(self) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("reset() must be called before get_obs()")
        h, v, m = self.state
        hint_override = None
        if self._feature_code == FEATURE_RESIDUAL:
            hint_override = singular_arc_u(
                h, v, m, self.params["T_max"], self.params["cd"], self.params["Q_max"]
            )
        return build_observation(
            h, v, m,
            self.params["m_fuel_0"], self.params["T_max"], self.params["cd"],
            self.params["Q_max"], self.t_now, self.t_target,
            feature_mode=self.feature_mode, hint_override=hint_override,
        )

    def _obs(self) -> np.ndarray:
        """CARL-compatible private observation alias."""

        return self.get_obs()

    def u_cap_safe(self) -> float:
        if self.params is None:
            raise RuntimeError("reset() must be called before u_cap_safe()")
        h, v, m = self.state
        return u_cap_safe(
            h, v, m, self.params["T_max"], self.params["cd"], self.params["Q_max"]
        )

    def reset(
        self,
        seed: int | None = None,
        *,
        params: Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if params is not None:
            self.params = _validate_params(params)
        elif self._fixed_params is not None:
            self.params = dict(self._fixed_params)
        else:
            self.params = sample_params(
                self.rng,
                exclude_eval=self.exclude_eval,
                exclusion_radius=self.exclusion_radius,
            )
        p = self.params
        self.m_fuel = p["m_fuel_0"]
        self.state = np.array([0.0, 0.0, M_DRY + self.m_fuel], dtype=np.float64)
        self.launched = False
        self.done = False
        self.terminated = False
        self.truncated = False
        self.h_apogee = 0.0
        self.max_h = 0.0
        self.n_steps = 0
        self.n_clamp = 0
        self.t_now = 0.0
        self.t_target = burn_target_time(
            p["m_fuel_0"], p["T_max"], p["Q_max"], self.target_mode
        )
        self.h_ref = h_ref_ballistic(p["m_fuel_0"], p["T_max"])
        self.burnout_penalty_applied = False
        self.burnout_penalty = 0.0
        self.burnout_time = None
        self.trajectory = []
        info = {
            "params": dict(p),
            "h_ref": self.h_ref,
            "t_target": self.t_target,
            "full_burn_time": full_burn_time(p["m_fuel_0"], p["T_max"]),
        }
        if self.record_trajectory:
            self.trajectory.append(self._trajectory_record(0.0, 0.0, False, 0.0))
        return self.get_obs(), info

    def _trajectory_record(
        self, u_cmd: float, u_applied: float, clamped: bool, reward: float
    ) -> dict[str, Any]:
        h, v, m = self.state
        return {
            "step": self.n_steps,
            "t": self.t_now,
            "h": float(h),
            "v": float(v),
            "m": float(m),
            "q": dyn_pressure(h, v),
            "u_cmd": float(u_cmd),
            "u_applied": float(u_applied),
            "clamped": bool(clamped),
            "reward": float(reward),
        }

    def get_trajectory(self) -> list[dict[str, Any]]:
        return list(self.trajectory)

    def step(self, u: float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called after episode termination; call reset()")
        if self.params is None:
            raise RuntimeError("reset() must be called before step()")
        p = self.params
        u_cmd = float(np.clip(float(u), 0.0, 1.0))
        actor_cap = self.u_cap_safe() if self.use_actor_cap else 1.0
        u_limited = min(u_cmd, actor_cap)
        h, v, m = (float(value) for value in self.state)
        phi_before = 0.0
        if self._reward_code == 2:
            phi_before = potential(
                h, m, p["m_fuel_0"], self.t_now, self.t_target,
                c_pot=self.potential_coeff, c_burn=self.burn_potential_coeff,
                h_scale=self.potential_h_scale,
            )

        h_next, v_next, m_next, thrust, clamped = physics_step_numba(
            h, v, m, u_limited, p["T_max"], p["cd"], p["Q_max"],
            CE_FACTOR * p["T_max"], self.dt, self.n_bisect,
        )
        u_applied = thrust / p["T_max"]
        if clamped:
            self.n_clamp += 1
        fuel_before = self.m_fuel
        self.state = np.array([h_next, v_next, m_next], dtype=np.float64)
        self.m_fuel = max(m_next - M_DRY, 0.0)
        self.launched = self.launched or v_next > 0.0
        self.n_steps += 1
        self.t_now += self.dt
        self.h_apogee = max(self.h_apogee, h_next)
        self.max_h = self.h_apogee

        apogee = self.launched and v_next <= 0.0
        dry_nonlaunch = m_next <= M_DRY and v_next <= 0.0 and self.n_steps > 0
        self.terminated = bool(apogee or dry_nonlaunch)
        self.truncated = bool(
            not self.terminated and self.max_steps is not None and self.n_steps >= self.max_steps
        )
        self.done = self.terminated or self.truncated

        reward = (h_next - h) / H_SCALE_FIXED
        penalty = 0.0
        fuel_out = fuel_before > 0.0 and self.m_fuel <= 0.0
        if fuel_out:
            self.burnout_time = self.t_now
            if self._reward_code == 1 and not self.burnout_penalty_applied:
                deficit = max(0.0, self.t_target - self.t_now) / max(self.t_target, 1.0e-9)
                deficit = float(np.clip(deficit, 0.0, 1.0))
                penalty = -self.burnout_penalty_coef * deficit * deficit
                self.burnout_penalty_applied = True
                self.burnout_penalty = penalty
                reward += penalty

        shaping = 0.0
        if self._reward_code == 2:
            phi_after = 0.0 if self.done else potential(
                h_next, m_next, p["m_fuel_0"], self.t_now, self.t_target,
                c_pot=self.potential_coeff, c_burn=self.burn_potential_coeff,
                h_scale=self.potential_h_scale,
            )
            shaping = self.gamma * phi_after - phi_before
            reward += shaping

        q_next = dyn_pressure(h_next, v_next)
        info: dict[str, Any] = {
            "h": h_next,
            "v": v_next,
            "m": m_next,
            "q": q_next,
            "q_over_Qmax": q_next / p["Q_max"],
            "throttle_commanded": u_cmd,
            "throttle_actor_capped": u_limited,
            "throttle_applied": u_applied,
            "actor_cap": actor_cap,
            "clamp_active": bool(clamped),
            "reward_step": reward,
            "reward_altitude": (h_next - h) / H_SCALE_FIXED,
            "burnout_penalty_step": penalty,
            "potential_shaping_step": shaping,
        }
        if self.done:
            info.update({
                "h_apogee": self.h_apogee,
                "h_ref": self.h_ref,
                "apogee_fraction": self.h_apogee / max(self.h_ref, 1.0e-9),
                "n_clamp_steps": self.n_clamp,
                "n_steps": self.n_steps,
                "t_target": self.t_target,
                "burnout_time": self.burnout_time if self.burnout_time is not None else -1.0,
                "burnout_penalty_total": self.burnout_penalty,
                **p,
            })
        if self.record_trajectory:
            self.trajectory.append(self._trajectory_record(u_cmd, u_applied, clamped, reward))
        return self.get_obs(), float(reward), self.terminated, self.truncated, info


# CARL's class name is useful when porting existing rollout code.
RocketEnv = GoddardEnv


# ---------------------------------------------------------------------------
# Parallel batched environment
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _batch_obs_kernel(
    h: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    m_fuel_0: np.ndarray,
    T_max: np.ndarray,
    cd: np.ndarray,
    Q_max: np.ndarray,
    t_now: np.ndarray,
    t_target: np.ndarray,
    mode: int,
    obs: np.ndarray,
) -> None:
    for i in prange(h.shape[0]):
        hint = -1.0
        if mode == FEATURE_RESIDUAL:
            hint = singular_arc_u_numba(h[i], v[i], m[i], T_max[i], cd[i], Q_max[i])
        fill_observation_numba(
            h[i], v[i], m[i], m_fuel_0[i], T_max[i], cd[i], Q_max[i],
            t_now[i], t_target[i], mode, hint, obs[i],
        )


@njit(parallel=True, cache=True)
def _batch_step_kernel(
    actions: np.ndarray,
    h: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    m_fuel_0: np.ndarray,
    T_max: np.ndarray,
    cd: np.ndarray,
    Q_max: np.ndarray,
    t_now: np.ndarray,
    t_target: np.ndarray,
    launched: np.ndarray,
    done: np.ndarray,
    h_apogee: np.ndarray,
    n_steps: np.ndarray,
    n_clamp: np.ndarray,
    burnout_detected: np.ndarray,
    burnout_penalty_applied: np.ndarray,
    burnout_penalty_total: np.ndarray,
    burnout_time: np.ndarray,
    obs: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    throttle_applied: np.ndarray,
    clamp_active: np.ndarray,
    q_out: np.ndarray,
    penalty_step: np.ndarray,
    shaping_step: np.ndarray,
    mode: int,
    reward_mode: int,
    use_actor_cap: bool,
    n_bisect: int,
    dt: float,
    max_steps: int,
    gamma: float,
    burnout_coef: float,
    c_pot: float,
    c_burn: float,
    potential_h_scale: float,
) -> None:
    for i in prange(h.shape[0]):
        rewards[i] = 0.0
        terminated[i] = False
        truncated[i] = False
        throttle_applied[i] = 0.0
        clamp_active[i] = False
        penalty_step[i] = 0.0
        shaping_step[i] = 0.0
        q_out[i] = dynamic_pressure_numba(h[i], v[i])

        if not done[i]:
            u_cmd = min(1.0, max(0.0, actions[i]))
            if use_actor_cap:
                cap = u_cap_safe_numba(h[i], v[i], m[i], T_max[i], cd[i], Q_max[i])
                u_cmd = min(u_cmd, cap)
            phi_before = 0.0
            if reward_mode == 2:
                phi_before = potential_numba(
                    h[i], m[i], m_fuel_0[i], t_now[i], t_target[i], False,
                    c_pot, c_burn, potential_h_scale,
                )
            h_old = h[i]
            fuel_before = max(m[i] - M_DRY, 0.0)
            h_new, v_new, m_new, thrust, clamped = physics_step_numba(
                h[i], v[i], m[i], u_cmd, T_max[i], cd[i], Q_max[i],
                CE_FACTOR * T_max[i], dt, n_bisect,
            )
            h[i] = h_new
            v[i] = v_new
            m[i] = m_new
            t_now[i] += dt
            n_steps[i] += 1
            launched[i] = launched[i] or v_new > 0.0
            h_apogee[i] = max(h_apogee[i], h_new)
            if clamped:
                n_clamp[i] += 1
            throttle_applied[i] = thrust / T_max[i]
            clamp_active[i] = clamped
            q_out[i] = dynamic_pressure_numba(h_new, v_new)

            apogee = launched[i] and v_new <= 0.0
            dry_nonlaunch = m_new <= M_DRY and v_new <= 0.0 and n_steps[i] > 0
            terminated[i] = apogee or dry_nonlaunch
            truncated[i] = (
                not terminated[i] and max_steps > 0 and n_steps[i] >= max_steps
            )
            done[i] = terminated[i] or truncated[i]

            rewards[i] = (h_new - h_old) / H_SCALE_FIXED
            fuel_after = max(m_new - M_DRY, 0.0)
            fuel_out = fuel_before > 0.0 and fuel_after <= 0.0
            if fuel_out:
                burnout_detected[i] = True
                burnout_time[i] = t_now[i]
                if reward_mode == 1 and not burnout_penalty_applied[i]:
                    deficit = max(0.0, t_target[i] - t_now[i]) / max(t_target[i], 1.0e-9)
                    deficit = min(1.0, max(0.0, deficit))
                    penalty = -burnout_coef * deficit * deficit
                    burnout_penalty_applied[i] = True
                    burnout_penalty_total[i] += penalty
                    penalty_step[i] = penalty
                    rewards[i] += penalty
            if reward_mode == 2:
                phi_after = 0.0
                if not done[i]:
                    phi_after = potential_numba(
                        h_new, m_new, m_fuel_0[i], t_now[i], t_target[i], False,
                        c_pot, c_burn, potential_h_scale,
                    )
                shaping = gamma * phi_after - phi_before
                shaping_step[i] = shaping
                rewards[i] += shaping

        hint = -1.0
        if mode == FEATURE_RESIDUAL:
            hint = singular_arc_u_numba(h[i], v[i], m[i], T_max[i], cd[i], Q_max[i])
        fill_observation_numba(
            h[i], v[i], m[i], m_fuel_0[i], T_max[i], cd[i], Q_max[i],
            t_now[i], t_target[i], mode, hint, obs[i],
        )


class BatchedGoddardEnv:
    """N independent environments stepped by one ``@njit(parallel=True)`` kernel.

    Parameters are sampled in common-random-number blocks: each of ``n_blocks``
    parameter sets is repeated for ``n_twins`` adjacent environments.  Done
    environments remain done and return zero reward while other slots finish;
    call :meth:`reset` to begin a fresh batch.
    """

    ACT_DIM = 1

    def __init__(
        self,
        n_envs: int | None = None,
        *,
        n_blocks: int | None = None,
        n_twins: int | None = None,
        seed: int | None = None,
        params: Mapping[str, float] | None = None,
        feature_mode: str = "base",
        reward_mode: str = "base",
        target_mode: str | None = None,
        use_actor_cap: bool = False,
        n_bisect: int = N_BISECT,
        dt: float = DT,
        max_steps: int | None = MAX_STEPS,
        gamma: float = 1.0,
        burnout_penalty_coef: float = PENALTY_COEF,
        potential_coeff: float = 1.0,
        burn_potential_coeff: float = 1.0,
        potential_h_scale: float = H_SCALE,
        exclude_eval: bool = True,
        exclusion_radius: float = 0.0,
    ) -> None:
        n_envs, n_blocks, n_twins = self._resolve_shape(n_envs, n_blocks, n_twins)
        self.n_envs = n_envs
        self.n_blocks = n_blocks
        self.n_twins = n_twins
        self.rng = np.random.default_rng(seed)
        self._fixed_params = _validate_params(params) if params is not None else None
        self.feature_mode = feature_mode
        self._feature_code = feature_code(feature_mode)
        self.OBS_DIM = obs_dim(feature_mode)
        self.obs_dim = self.OBS_DIM
        self.reward_mode = reward_mode
        self._reward_code = _reward_code(reward_mode)
        self.target_mode = target_mode or default_target_mode(feature_mode)
        self._target_code = target_code(self.target_mode)
        self.use_actor_cap = bool(use_actor_cap)
        self.n_bisect = int(n_bisect)
        self.dt = float(dt)
        self.max_steps = max_steps
        self._max_steps_kernel = -1 if max_steps is None else int(max_steps)
        self.gamma = float(gamma)
        self.burnout_penalty_coef = float(burnout_penalty_coef)
        self.potential_coeff = float(potential_coeff)
        self.burn_potential_coeff = float(burn_potential_coeff)
        self.potential_h_scale = float(potential_h_scale)
        self.exclude_eval = bool(exclude_eval)
        self.exclusion_radius = float(exclusion_radius)
        if self.n_bisect < 0 or self.dt <= 0.0:
            raise ValueError("n_bisect must be non-negative and dt positive")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive or None")

        n = self.n_envs
        self.h = np.zeros(n, dtype=np.float64)
        self.v = np.zeros(n, dtype=np.float64)
        self.m = np.full(n, M_DRY, dtype=np.float64)
        self.m_fuel_0 = np.zeros(n, dtype=np.float64)
        self.T_max = np.zeros(n, dtype=np.float64)
        self.cd = np.zeros(n, dtype=np.float64)
        self.Q_max = np.zeros(n, dtype=np.float64)
        self.t_now = np.zeros(n, dtype=np.float64)
        self.t_target = np.zeros(n, dtype=np.float64)
        self.launched = np.zeros(n, dtype=np.bool_)
        self.done = np.zeros(n, dtype=np.bool_)
        self.h_apogee = np.zeros(n, dtype=np.float64)
        self.max_h = self.h_apogee
        self.n_steps = np.zeros(n, dtype=np.int64)
        self.n_clamp = np.zeros(n, dtype=np.int64)
        self.burnout_detected = np.zeros(n, dtype=np.bool_)
        self.burnout_penalty_applied = np.zeros(n, dtype=np.bool_)
        self.burnout_penalty_total = np.zeros(n, dtype=np.float64)
        self.burnout_time = np.full(n, -1.0, dtype=np.float64)

        self.obs = np.zeros((n, self.OBS_DIM), dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float64)
        self.terminated = np.zeros(n, dtype=np.bool_)
        self.truncated = np.zeros(n, dtype=np.bool_)
        self.throttle_applied = np.zeros(n, dtype=np.float64)
        self.clamp_active = np.zeros(n, dtype=np.bool_)
        self.q = np.zeros(n, dtype=np.float64)
        self.penalty_step = np.zeros(n, dtype=np.float64)
        self.shaping_step = np.zeros(n, dtype=np.float64)
        self._has_reset = False

    @staticmethod
    def _resolve_shape(
        n_envs: int | None, n_blocks: int | None, n_twins: int | None
    ) -> tuple[int, int, int]:
        if n_envs is None:
            blocks = 32 if n_blocks is None else int(n_blocks)
            twins = 8 if n_twins is None else int(n_twins)
            total = blocks * twins
        else:
            total = int(n_envs)
            if n_blocks is None and n_twins is None:
                twins = 8 if total % 8 == 0 else 1
                blocks = total // twins
            elif n_blocks is None:
                twins = int(n_twins)
                if total % twins:
                    raise ValueError("n_envs must be divisible by n_twins")
                blocks = total // twins
            elif n_twins is None:
                blocks = int(n_blocks)
                if total % blocks:
                    raise ValueError("n_envs must be divisible by n_blocks")
                twins = total // blocks
            else:
                blocks, twins = int(n_blocks), int(n_twins)
                if blocks * twins != total:
                    raise ValueError("n_envs must equal n_blocks*n_twins")
        if total <= 0 or blocks <= 0 or twins <= 0:
            raise ValueError("n_envs, n_blocks, and n_twins must be positive")
        return total, blocks, twins

    @property
    def state(self) -> np.ndarray:
        return np.column_stack((self.h, self.v, self.m))

    @property
    def m_fuel(self) -> np.ndarray:
        return np.maximum(self.m - M_DRY, 0.0)

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {
            "m_fuel_0": self.m_fuel_0,
            "T_max": self.T_max,
            "cd": self.cd,
            "Q_max": self.Q_max,
        }

    def _install_params(self, block_params: Sequence[Mapping[str, float]]) -> None:
        if len(block_params) != self.n_blocks:
            raise ValueError(f"Expected {self.n_blocks} block parameter sets")
        validated = [_validate_params(p) for p in block_params]
        for block, p in enumerate(validated):
            start = block * self.n_twins
            stop = start + self.n_twins
            self.m_fuel_0[start:stop] = p["m_fuel_0"]
            self.T_max[start:stop] = p["T_max"]
            self.cd[start:stop] = p["cd"]
            self.Q_max[start:stop] = p["Q_max"]

    def _sample_blocks(self) -> list[dict[str, float]]:
        if self._fixed_params is not None:
            return [dict(self._fixed_params) for _ in range(self.n_blocks)]
        return [
            sample_params(
                self.rng,
                exclude_eval=self.exclude_eval,
                exclusion_radius=self.exclusion_radius,
            )
            for _ in range(self.n_blocks)
        ]

    def reset(
        self,
        seed: int | None = None,
        *,
        params: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if params is None:
            blocks = self._sample_blocks()
        elif isinstance(params, Mapping):
            blocks = [params for _ in range(self.n_blocks)]
        else:
            blocks = list(params)
        self._install_params(blocks)
        self.h.fill(0.0)
        self.v.fill(0.0)
        self.m[:] = M_DRY + self.m_fuel_0
        self.t_now.fill(0.0)
        for i in range(self.n_envs):
            self.t_target[i] = burn_target_time_numba(
                self.m_fuel_0[i], self.T_max[i], self.Q_max[i], self._target_code
            )
        self.launched.fill(False)
        self.done.fill(False)
        self.h_apogee.fill(0.0)
        self.n_steps.fill(0)
        self.n_clamp.fill(0)
        self.burnout_detected.fill(False)
        self.burnout_penalty_applied.fill(False)
        self.burnout_penalty_total.fill(0.0)
        self.burnout_time.fill(-1.0)
        self.terminated.fill(False)
        self.truncated.fill(False)
        self.rewards.fill(0.0)
        self.throttle_applied.fill(0.0)
        self.clamp_active.fill(False)
        self.penalty_step.fill(0.0)
        self.shaping_step.fill(0.0)
        self.q[:] = 0.0
        _batch_obs_kernel(
            self.h, self.v, self.m, self.m_fuel_0, self.T_max, self.cd,
            self.Q_max, self.t_now, self.t_target, self._feature_code, self.obs,
        )
        self._has_reset = True
        info = {
            "params": {key: value.copy() for key, value in self.params.items()},
            "t_target": self.t_target.copy(),
            "n_blocks": self.n_blocks,
            "n_twins": self.n_twins,
        }
        return self.obs, info

    def get_obs(self) -> np.ndarray:
        return self.obs

    def reset_indices(self, indices: Sequence[int], *, resample: bool = True) -> None:
        """Restart selected environments in place (PPO auto-reset).

        Parameters of a restarted environment are re-drawn from the training
        distribution unless the environment was constructed with fixed
        ``params``.  Everything else is reset exactly like :meth:`reset`.
        """

        if not self._has_reset:
            raise RuntimeError("reset() must be called before reset_indices()")
        idx = np.asarray(list(indices), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < self.n_envs)]
        if idx.size == 0:
            return
        if resample and self._fixed_params is None:
            for i in idx:
                p = sample_params(
                    self.rng,
                    exclude_eval=self.exclude_eval,
                    exclusion_radius=self.exclusion_radius,
                )
                self.m_fuel_0[i] = p["m_fuel_0"]
                self.T_max[i] = p["T_max"]
                self.cd[i] = p["cd"]
                self.Q_max[i] = p["Q_max"]
                self.t_target[i] = burn_target_time_numba(
                    p["m_fuel_0"], p["T_max"], p["Q_max"], self._target_code
                )
        self.h[idx] = 0.0
        self.v[idx] = 0.0
        self.m[idx] = M_DRY + self.m_fuel_0[idx]
        self.t_now[idx] = 0.0
        self.launched[idx] = False
        self.done[idx] = False
        self.h_apogee[idx] = 0.0
        self.n_steps[idx] = 0
        self.n_clamp[idx] = 0
        self.burnout_detected[idx] = False
        self.burnout_penalty_applied[idx] = False
        self.burnout_penalty_total[idx] = 0.0
        self.burnout_time[idx] = -1.0
        self.terminated[idx] = False
        self.truncated[idx] = False
        self.rewards[idx] = 0.0
        self.throttle_applied[idx] = 0.0
        self.clamp_active[idx] = False
        self.penalty_step[idx] = 0.0
        self.shaping_step[idx] = 0.0
        self.q[idx] = 0.0
        _batch_obs_kernel(
            self.h, self.v, self.m, self.m_fuel_0, self.T_max, self.cd,
            self.Q_max, self.t_now, self.t_target, self._feature_code, self.obs,
        )

    def u_cap_safe(self) -> np.ndarray:
        result = np.empty(self.n_envs, dtype=np.float64)
        for i in range(self.n_envs):
            result[i] = u_cap_safe_numba(
                self.h[i], self.v[i], self.m[i], self.T_max[i], self.cd[i], self.Q_max[i]
            )
        return result

    def step(
        self, actions: Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        action_array = np.asarray(actions, dtype=np.float64)
        if action_array.shape != (self.n_envs,):
            raise ValueError(f"actions must have shape ({self.n_envs},), got {action_array.shape}")
        if not np.isfinite(action_array).all():
            raise ValueError("actions must be finite")
        # np.ascontiguousarray avoids a copy in the usual float64 case and
        # ensures Numba sees a dense action vector for float32 policy outputs.
        action_array = np.ascontiguousarray(action_array)
        _batch_step_kernel(
            action_array, self.h, self.v, self.m, self.m_fuel_0, self.T_max,
            self.cd, self.Q_max, self.t_now, self.t_target, self.launched,
            self.done, self.h_apogee, self.n_steps, self.n_clamp,
            self.burnout_detected, self.burnout_penalty_applied,
            self.burnout_penalty_total, self.burnout_time, self.obs,
            self.rewards, self.terminated, self.truncated,
            self.throttle_applied, self.clamp_active, self.q,
            self.penalty_step, self.shaping_step, self._feature_code,
            self._reward_code, self.use_actor_cap, self.n_bisect, self.dt,
            self._max_steps_kernel, self.gamma, self.burnout_penalty_coef,
            self.potential_coeff, self.burn_potential_coeff, self.potential_h_scale,
        )
        info = {
            "h": self.h,
            "v": self.v,
            "m": self.m,
            "q": self.q,
            "q_over_Qmax": self.q / self.Q_max,
            "throttle_applied": self.throttle_applied,
            "clamp_active": self.clamp_active,
            "burnout_penalty_step": self.penalty_step,
            "potential_shaping_step": self.shaping_step,
            "h_apogee": self.h_apogee,
            "n_steps": self.n_steps,
            "n_clamp_steps": self.n_clamp,
            "burnout_time": self.burnout_time,
            "burnout_penalty_total": self.burnout_penalty_total,
        }
        return self.obs, self.rewards, self.terminated, self.truncated, info


__all__ = [
    "CD_RANGE",
    "CE_FACTOR",
    "DRAG_SURCHARGE",
    "DT",
    "EVAL_M_FUEL_GRID",
    "EVAL_PAIRS",
    "EVAL_Q_MAX_GRID",
    "HINT_EARLY_FRAC",
    "H_SCALE",
    "H_SCALE_FIXED",
    "MAX_STEPS",
    "M_DRY",
    "M_FUEL_RANGE",
    "N_BISECT",
    "Q_MAX_RANGE",
    "RHO_0",
    "TW_MIN",
    "T_MAX_RANGE",
    "BatchedGoddardEnv",
    "G",
    "GoddardEnv",
    "RocketEnv",
    "advance_fuel_exact",
    "advance_fuel_exact_numba",
    "build_obs_numba",
    "dyn_pressure",
    "dynamic_pressure",
    "full_burn_time",
    "h_ref_ballistic",
    "is_eval_pair",
    "obs_dim",
    "physics_step_numba",
    "potential",
    "potential_numba",
    "rho",
    "rk4_step",
    "rk4_step_numba",
    "sample_params",
    "singular_arc_u",
    "singular_arc_u_numba",
    "u_cap_safe",
    "u_cap_safe_numba",
]
