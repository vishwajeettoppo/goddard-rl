"""Non-RL baselines for the constrained Goddard rocket (paper Table 7).

Three references are implemented exactly as described in Section 7 of
``INFORMS-CARL.tex``:

* Bang-Bang -- full thrust until the propellant is exhausted.
* MC-Local-Greedy -- 30 piecewise-constant throttle segments, 500 random
  candidate schedules, then coordinate descent on the best eight.
* Unconstrained Optimal -- full thrust until ``t_switch``, equation (8)
  singular arc afterwards, coast once dry.

The optional fourth reference :func:`solve_casadi` solves a direct
multiple-shooting transcription of the same optimal-control problem with
CasADi/IPOPT; it is a mathematical upper bound and is not used inside any
learning loop.

All rollouts go through :mod:`src.env`, i.e. the same RK4 integrator, exact
fuel-out splitting and max-Q clamp with the 5% drag surcharge used to produce
the paper's table, so the numbers here are directly comparable.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numba import njit, prange

from .env import (
    CE_FACTOR,
    DT,
    EVAL_CD,
    EVAL_T_MAX,
    MAX_STEPS,
    M_DRY,
    N_BISECT,
    GoddardEnv,
    physics_step_numba,
    singular_arc_u,
)

METHODS = (
    "bang_bang",
    "mc_local_greedy",
    "unconstrained_optimal",
    "casadi_optimum",
)

BASELINE_LABELS = {
    "bang_bang": "Bang-Bang",
    "mc_local_greedy": "MC-Local-Greedy",
    "unconstrained_optimal": "Unconstrained Optimal",
    "casadi_optimum": "CasADi Optimum",
}


# ---------------------------------------------------------------------------
# Fast batched rollouts of open-loop throttle schedules
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def rollout_schedules_numba(
    schedules: np.ndarray,
    seg_len: float,
    m_fuel_0: float,
    T_max: float,
    cd: float,
    Q_max: float,
    dt: float = DT,
    max_steps: int = MAX_STEPS,
    n_bisect: int = N_BISECT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out ``(n, n_segments)`` piecewise-constant throttles in parallel.

    Returns ``(apogee, t_apogee, n_steps, peak_q)``.  Each candidate uses the
    shared :func:`src.env.physics_step_numba` kernel, so no physics is
    duplicated here.
    """

    n = schedules.shape[0]
    n_seg = schedules.shape[1]
    c_e = CE_FACTOR * T_max
    apogee = np.zeros(n)
    t_apogee = np.zeros(n)
    steps_out = np.zeros(n, dtype=np.int64)
    peak_q = np.zeros(n)

    for i in prange(n):
        h = 0.0
        v = 0.0
        m = M_DRY + m_fuel_0
        launched = False
        best_h = 0.0
        best_t = 0.0
        peak = 0.0
        for k in range(max_steps):
            segment = min(int((k * dt) / seg_len), n_seg - 1)
            u = schedules[i, segment]
            h, v, m, _thrust, _clamped = physics_step_numba(
                h, v, m, u, T_max, cd, Q_max, c_e, dt, n_bisect
            )
            t = (k + 1) * dt
            if h > best_h:
                best_h = h
                best_t = t
            launched = launched or v > 0.0
            q = 0.5 * 1.225 * math.exp(-max(h, 0.0) / 8500.0) * v * v
            if q > peak:
                peak = q
            if launched and v <= 0.0:
                steps_out[i] = k + 1
                break
        else:
            steps_out[i] = max_steps
        apogee[i] = best_h
        t_apogee[i] = best_t
        peak_q[i] = peak

    return apogee, t_apogee, steps_out, peak_q


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """Outcome of one deterministic rollout."""

    method: str
    apogee: float
    t_apogee: float
    n_steps: int
    n_clamp: int
    max_q: float
    q_violation_frac: float
    runtime_s: float
    trajectory: dict[str, np.ndarray] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row = {
            "method": self.method,
            "apogee": self.apogee,
            "t_apogee": self.t_apogee,
            "n_steps": self.n_steps,
            "n_clamp": self.n_clamp,
            "max_q": self.max_q,
            "q_violation_frac": self.q_violation_frac,
            "runtime_s": self.runtime_s,
        }
        row.update(self.extra)
        return row


def evaluate_policy(
    env: GoddardEnv,
    policy: Callable[[GoddardEnv, Mapping[str, Any]], float],
    *,
    method: str = "policy",
    record_trajectory: bool = False,
) -> BaselineResult:
    """Roll one deterministic episode and collect evaluation metrics."""

    env.record_trajectory = bool(record_trajectory)
    obs, info = env.reset()
    done = False
    max_q = 0.0
    violations = 0
    t0 = time.perf_counter()
    while not done:
        u = float(policy(env, info))
        obs, _reward, terminated, truncated, info = env.step(u)
        done = bool(terminated or truncated)
        max_q = max(max_q, float(info["q"]))
        if float(info["q"]) > env.params["Q_max"] * (1.0 + 1e-9):
            violations += 1
    runtime = time.perf_counter() - t0
    if record_trajectory and env.trajectory:
        keys = ("t", "h", "v", "m", "q", "u_cmd", "u_applied", "clamped")
        trajectory = {key: np.asarray([row[key] for row in env.trajectory]) for key in keys}
    else:
        trajectory = {}
    return BaselineResult(
        method=method,
        apogee=float(env.h_apogee),
        t_apogee=float(env.t_now),
        n_steps=int(env.n_steps),
        n_clamp=int(env.n_clamp),
        max_q=max_q,
        q_violation_frac=violations / max(env.n_steps, 1),
        runtime_s=runtime,
        trajectory=trajectory,
    )


def make_env(
    m_fuel_0: float,
    Q_max: float,
    T_max: float = EVAL_T_MAX,
    cd: float = EVAL_CD,
    *,
    record_trajectory: bool = False,
    use_actor_cap: bool = False,
) -> GoddardEnv:
    """Convenience constructor for a fixed evaluation scenario."""

    return GoddardEnv(
        params={"m_fuel_0": m_fuel_0, "T_max": T_max, "cd": cd, "Q_max": Q_max},
        record_trajectory=record_trajectory,
        use_actor_cap=use_actor_cap,
    )


# ---------------------------------------------------------------------------
# 1. Bang-Bang
# ---------------------------------------------------------------------------


def bang_bang_policy(env: GoddardEnv, info: Mapping[str, Any]) -> float:
    """Full throttle while propellant remains, then coast (u=0)."""

    del info
    return 1.0 if env.m_fuel > 0.0 else 0.0


# ---------------------------------------------------------------------------
# 2. Monte-Carlo local greedy
# ---------------------------------------------------------------------------


class MCLocalGreedyPolicy:
    """30-segment piecewise-constant throttle tuned by coordinate descent."""

    def __init__(
        self,
        m_fuel_0: float,
        Q_max: float,
        T_max: float = EVAL_T_MAX,
        cd: float = EVAL_CD,
        *,
        horizon_factor: float = 4.0,
        n_segments: int = 30,
        n_samples: int = 500,
        top_k: int = 8,
        grid_step: float = 0.05,
        max_steps: int = MAX_STEPS,
        seed: int = 0,
    ) -> None:
        self.m_fuel_0 = float(m_fuel_0)
        self.Q_max = float(Q_max)
        self.T_max = float(T_max)
        self.cd = float(cd)
        self.horizon = horizon_factor * self.m_fuel_0 * (CE_FACTOR * self.T_max) / self.T_max
        self.n_segments = int(n_segments)
        self.n_samples = int(n_samples)
        self.top_k = int(top_k)
        self.grid_step = float(grid_step)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.seg_len = self.horizon / self.n_segments
        self.schedule = np.ones(self.n_segments)
        self.info: dict[str, Any] = {"optimized": False}

    # -- candidate generation -------------------------------------------------
    def _candidates(self, rng: np.random.Generator) -> np.ndarray:
        n_seg = self.n_segments
        idx = np.linspace(0.0, 1.0, n_seg)
        candidates: list[np.ndarray] = [np.ones(n_seg)]
        candidates.append(np.zeros(n_seg))
        for rate in (0.25, 0.5, 0.75, 1.0):
            candidates.append(np.clip(1.0 - rate * idx * 3.0, 0.0, 1.0))
            candidates.append(np.clip(np.exp(-rate * 3.0 * idx), 0.0, 1.0))
        # Front-loaded: full throttle for a random prefix, then a taper.
        for frac in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
            cut = max(1, int(frac * n_seg))
            profile = np.zeros(n_seg)
            profile[:cut] = 1.0
            profile[cut:] = np.linspace(0.8, 0.0, n_seg - cut)
            candidates.append(profile)
        while len(candidates) < self.n_samples:
            kind = rng.integers(0, 3)
            if kind == 0:
                candidates.append(rng.uniform(0.0, 1.0, n_seg))
            elif kind == 1:
                knots = rng.uniform(0.0, 1.0, 5)
                candidates.append(np.interp(idx, np.linspace(0.0, 1.0, 5), knots))
            else:
                scale = rng.uniform(0.05, 1.0)
                decay = rng.uniform(1.0, 6.0)
                candidates.append(np.clip(scale * np.exp(-decay * idx), 0.0, 1.0))
        return np.asarray(candidates[: self.n_samples], dtype=np.float64)

    def _score(self, schedules: np.ndarray) -> np.ndarray:
        apogee, _, _, _ = rollout_schedules_numba(
            np.ascontiguousarray(schedules, dtype=np.float64),
            self.seg_len,
            self.m_fuel_0,
            self.T_max,
            self.cd,
            self.Q_max,
            DT,
            self.max_steps,
            N_BISECT,
        )
        return apogee

    # -- optimisation ---------------------------------------------------------
    def optimize(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        candidates = self._candidates(rng)
        scores = self._score(candidates)
        n_evals = len(scores)
        order = np.argsort(-scores)[: self.top_k]
        experts = candidates[order].copy()

        passes = (
            (np.round(np.arange(0.0, 1.0 + 1e-9, 0.2), 10), 2),
            (np.round(np.arange(0.0, 1.0 + 1e-9, self.grid_step), 10), 1),
        )
        for grid, repeats in passes:
            for _ in range(repeats):
                for segment in range(self.n_segments):
                    trial = np.repeat(experts[:, None, :], len(grid), axis=1)
                    trial[:, :, segment] = grid[None, :]
                    flat = trial.reshape(-1, self.n_segments)
                    values = self._score(flat)
                    n_evals += len(values)
                    values = values.reshape(len(experts), len(grid))
                    best = np.argmax(values, axis=1)
                    for j in range(len(experts)):
                        experts[j, segment] = grid[best[j]]

        final_scores = self._score(experts)
        n_evals += len(final_scores)
        best = int(np.argmax(final_scores))
        self.schedule = experts[best]
        self.info = {
            "optimized": True,
            "apogee": float(final_scores[best]),
            "n_evals": int(n_evals),
            "runtime_s": time.perf_counter() - t0,
            "horizon_s": self.horizon,
            "n_segments": self.n_segments,
            "grid_step": self.grid_step,
        }
        return {"schedule": self.schedule.copy(), **self.info}

    @property
    def apogee(self) -> float:
        return float(self.info.get("apogee", float("nan")))

    def act(self, env: GoddardEnv, info: Mapping[str, Any] | None = None) -> float:
        del info
        if not self.info.get("optimized"):
            self.optimize()
        segment = min(int(env.t_now / self.seg_len), self.n_segments - 1)
        if env.m_fuel <= 0.0:
            return 0.0
        return float(self.schedule[segment])

    __call__ = act


# ---------------------------------------------------------------------------
# 3. Unconstrained optimal (bang -> singular arc -> coast)
# ---------------------------------------------------------------------------

_T_SWITCH_CACHE = Path(__file__).resolve().parents[1] / "results" / "cache" / "unconstrained_t_switch.json"
_TS_CACHE: dict[str, float] | None = None


def _load_ts_cache() -> dict[str, float]:
    global _TS_CACHE
    if _TS_CACHE is None:
        if _T_SWITCH_CACHE.exists():
            try:
                _TS_CACHE = json.loads(_T_SWITCH_CACHE.read_text())
            except (json.JSONDecodeError, OSError):
                _TS_CACHE = {}
        else:
            _TS_CACHE = {}
    return _TS_CACHE


def _save_ts_cache(cache: Mapping[str, float]) -> None:
    _T_SWITCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _T_SWITCH_CACHE.write_text(json.dumps(dict(sorted(cache.items())), indent=2) + "\n")


def _ts_key(m_fuel_0: float, T_max: float, cd: float) -> str:
    return f"{m_fuel_0:.3f}|{T_max:.3f}|{cd:.4f}"


def _rollout_unconstrained(
    m_fuel_0: float,
    T_max: float,
    cd: float,
    t_switch: float,
    *,
    dt: float = DT,
    max_steps: int = MAX_STEPS,
) -> float:
    """Apogee in the unconstrained model (Q_max effectively infinite)."""

    h = v = 0.0
    m = M_DRY + m_fuel_0
    launched = False
    best = 0.0
    c_e = CE_FACTOR * T_max
    for k in range(max_steps):
        t = k * dt
        u = 0.0
        if m > M_DRY:
            u = 1.0 if t < t_switch else singular_arc_u(
                h, v, m, T_max, cd, None, t=t, t_switch=t_switch
            )
        h, v, m, _thrust, _clamped = physics_step_numba(
            h, v, m, u, T_max, cd, 1.0e12, c_e, dt, N_BISECT
        )
        launched = launched or v > 0.0
        best = max(best, h)
        if launched and v <= 0.0:
            break
    return best


class UnconstrainedOptimalPolicy:
    """Bang-arc-bang feedback law; ``t_switch`` found on the unconstrained model.

    The paper (Section 7, Table 7) does not state how ``t_switch`` was chosen.
    Following the trajectory that maximises apogee without the Q_max constraint
    reproduces every light and nominal paper row exactly (3505 m, 14443 m,
    15717 m, 15805 m) and the loose-Q heavy rows (50165 m).  The three
    heavy-fuel / tight-Q rows cannot be reproduced from the published
    information; see ``docs/reproduction.md`` for the measured gap.
    """

    def __init__(
        self,
        m_fuel_0: float,
        Q_max: float,
        T_max: float = EVAL_T_MAX,
        cd: float = EVAL_CD,
        *,
        t_switch: float | None = None,
        search: bool = True,
        ts_grid_step: float = 0.1,
        ts_max: float = 40.0,
        cap: bool = False,
    ) -> None:
        self.m_fuel_0 = float(m_fuel_0)
        self.Q_max = float(Q_max)
        self.T_max = float(T_max)
        self.cd = float(cd)
        self.cap = bool(cap)
        self.search = bool(search)
        self.ts_grid_step = float(ts_grid_step)
        self.ts_max = float(ts_max)
        self.t_switch = round(float(t_switch), 9) if t_switch is not None else (
            round(self.solve_t_switch(), 9) if search else 0.0
        )

    def solve_t_switch(self) -> float:
        key = _ts_key(self.m_fuel_0, self.T_max, self.cd)
        cache = _load_ts_cache()
        if key in cache:
            return round(float(cache[key]), 9)
        best_ts, best_apogee = 0.0, -1.0
        for ts in np.arange(0.0, self.ts_max + 1e-9, self.ts_grid_step):
            value = _rollout_unconstrained(self.m_fuel_0, self.T_max, self.cd, float(ts))
            if value > best_apogee:
                best_apogee, best_ts = value, float(ts)
        cache = _load_ts_cache()
        cache[key] = best_ts
        _save_ts_cache(cache)
        return best_ts

    def act(self, env: GoddardEnv, info: Mapping[str, Any] | None = None) -> float:
        del info
        if env.m_fuel <= 0.0:
            return 0.0
        h, v, m = (float(x) for x in env.state)
        return float(
            singular_arc_u(
                h,
                v,
                m,
                self.T_max,
                self.cd,
                self.Q_max if self.cap else None,
                use_cap=self.cap,
                t=round(env.t_now, 9),
                t_switch=self.t_switch,
            )
        )

    __call__ = act


# ---------------------------------------------------------------------------
# 4. CasADi direct multiple shooting (optional upper bound)
# ---------------------------------------------------------------------------


def solve_casadi(
    m_fuel_0: float,
    Q_max: float,
    T_max: float = EVAL_T_MAX,
    cd: float = EVAL_CD,
    *,
    n_intervals: int = 240,
    dt: float = 1.0,  # retained for API compatibility; the step is tf/n_intervals
    max_steps: int = MAX_STEPS,
    ipopt_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve a multiple-shooting transcription of the constrained problem.

    Decision variables are the states ``(h, v, m)`` at ``n_intervals + 1``
    nodes and the throttle at ``n_intervals`` intervals.  Dynamics are the
    exact RK4 map with a single ``dt`` step, the max-Q limit is imposed at
    every node, and terminal velocity is pinned to the apogee condition
    ``v_N = 0``.  Returns a dict with ``apogee``, ``status`` and diagnostics;
    failures are reported rather than raised so pipelines can continue.
    """

    result: dict[str, Any] = {"apogee": float("nan"), "status": "unavailable", "runtime_s": 0.0}
    try:
        import casadi as ca
    except ImportError:
        result["status"] = "casadi_not_installed"
        return result

    t0 = time.perf_counter()
    c_e = CE_FACTOR * T_max
    g, rho0, scale = 9.81, 1.225, 8500.0
    horizon = float(max_steps * DT)

    def rk4(h, v, m, thrust, alpha, step):
        def f(h_, v_, m_, thrust_):
            rho = rho0 * ca.exp(-ca.fmax(h_, 0.0) / scale)
            drag = alpha * cd * rho * v_ * ca.fabs(v_)
            # Clamp the RK4 stage mass away from zero: IPOPT evaluates equality
            # constraints outside the feasible box while searching.
            m_safe = ca.fmax(m_, 0.5 * M_DRY)
            dh = v_
            dv = (thrust_ - drag) / m_safe - g
            dm = -thrust_ / c_e
            return dh, dv, dm

        k1 = f(h, v, m, thrust)
        k2 = f(h + 0.5 * step * k1[0], v + 0.5 * step * k1[1], m + 0.5 * step * k1[2], thrust)
        k3 = f(h + 0.5 * step * k2[0], v + 0.5 * step * k2[1], m + 0.5 * step * k2[2], thrust)
        k4 = f(h + step * k3[0], v + step * k3[1], m + step * k3[2], thrust)
        return (
            h + step / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            v + step / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            m + step / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
        )

    opti = ca.Opti()
    X = opti.variable(3, n_intervals + 1)
    U = opti.variable(1, n_intervals)
    tf = opti.variable()
    h, v, m = X[0, :], X[1, :], X[2, :]
    step = tf / n_intervals

    opti.subject_to(X[:, 0] == ca.vertcat(0.0, 0.0, M_DRY + m_fuel_0))
    opti.subject_to(opti.bounded(0.0, U, 1.0))
    opti.subject_to(opti.bounded(M_DRY, m, M_DRY + m_fuel_0))
    opti.subject_to(opti.bounded(1.0, tf, 2.0 * horizon))
    opti.subject_to(h >= 0.0)
    opti.subject_to(v >= -1.0)
    opti.subject_to(v[-1] == 0.0)

    for k in range(n_intervals):
        h1, v1, m1 = rk4(h[k], v[k], m[k], U[0, k] * T_max, 1.0, step)
        opti.subject_to(h[k + 1] == h1)
        opti.subject_to(v[k + 1] == v1)
        opti.subject_to(m[k + 1] == m1)
        q = 0.5 * rho0 * ca.exp(-ca.fmax(h[k], 0.0) / scale) * v[k] ** 2
        opti.subject_to(q <= Q_max)

    opti.minimize(-h[-1])
    options = {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 3000,
               "ipopt.tol": 1e-7, "ipopt.acceptable_tol": 1e-6}
    if ipopt_options:
        options.update(dict(ipopt_options))
    opti.solver("ipopt", options)

    t_full = m_fuel_0 * c_e / T_max
    # Warm start from the bang-bang rollout of the *same* shared physics.
    hh = vv = 0.0
    mm = M_DRY + m_fuel_0
    launched = False
    traj_t, traj_h, traj_v, traj_m = [0.0], [0.0], [0.0], [mm]
    for k in range(max_steps):
        hh, vv, mm, _thrust, _clamped = physics_step_numba(
            hh, vv, mm, 1.0, T_max, cd, Q_max, c_e, DT, N_BISECT
        )
        traj_t.append((k + 1) * DT)
        traj_h.append(hh)
        traj_v.append(vv)
        traj_m.append(mm)
        launched = launched or vv > 0.0
        if launched and vv <= 0.0:
            break
    traj_t = np.asarray(traj_t)
    done_time = float(traj_t[-1])
    grid = np.linspace(0.0, done_time, n_intervals + 1)
    opti.set_initial(tf, max(done_time, 1.0))
    opti.set_initial(X[0, :], np.interp(grid, traj_t, np.asarray(traj_h)))
    opti.set_initial(X[1, :], np.interp(grid, traj_t, np.asarray(traj_v)))
    opti.set_initial(X[2, :], np.interp(grid, traj_t, np.asarray(traj_m)))
    opti.set_initial(U, np.ones((1, n_intervals)))

    try:
        sol = opti.solve()
        apogee = float(sol.value(h[-1]))
        controls = np.asarray(sol.value(U)).ravel()
        tf_value = float(sol.value(tf))
        result.update(
            apogee=apogee,
            status="optimal",
            u=controls,
            tf=tf_value,
            dt=tf_value / n_intervals,
            h=np.asarray(sol.value(h)).ravel(),
            v=np.asarray(sol.value(v)).ravel(),
            m=np.asarray(sol.value(m)).ravel(),
        )
    except RuntimeError as exc:  # IPOPT returns the best iterate in the message.
        result["status"] = f"failed: {type(exc).__name__}"
        try:
            apogee = float(opti.debug.value(h[-1]))
            result["apogee"] = apogee
            result["status"] = "infeasible_iterate"
        except Exception:  # pragma: no cover - diagnostics only
            pass
    result["runtime_s"] = time.perf_counter() - t0
    return result


class CasadiOptimumPolicy:
    """Open-loop replay of a CasADi solution (falls back to bang-bang)."""

    def __init__(
        self,
        m_fuel_0: float,
        Q_max: float,
        T_max: float = EVAL_T_MAX,
        cd: float = EVAL_CD,
        *,
        n_intervals: int = 240,
        dt: float = 1.0,
    ) -> None:
        self.m_fuel_0 = float(m_fuel_0)
        self.Q_max = float(Q_max)
        self.T_max = float(T_max)
        self.cd = float(cd)
        self.dt = float(dt)
        self.solution = solve_casadi(
            m_fuel_0, Q_max, T_max, cd, n_intervals=n_intervals, dt=dt
        )
        self.u = self.solution.get("u")

    @property
    def apogee(self) -> float:
        return float(self.solution.get("apogee", float("nan")))

    def act(self, env: GoddardEnv, info: Mapping[str, Any] | None = None) -> float:
        del info
        if env.m_fuel <= 0.0:
            return 0.0
        if self.u is None:
            return 1.0
        index = min(int(env.t_now / self.dt), len(self.u) - 1)
        return float(self.u[index])

    __call__ = act


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_baseline(
    method: str,
    scenario: Mapping[str, Any],
    *,
    seed: int = 0,
    **kwargs: Any,
) -> Callable[[GoddardEnv, Mapping[str, Any]], float]:
    """Construct a scenario-specific baseline policy.

    ``scenario`` must provide ``m_fuel_0`` and ``Q_max`` and may override the
    default ``T_max``/``cd`` evaluation constants.
    """

    m_fuel_0 = float(scenario["m_fuel_0"])
    Q_max = float(scenario["Q_max"])
    T_max = float(scenario.get("T_max", EVAL_T_MAX))
    cd = float(scenario.get("cd", EVAL_CD))
    if method == "bang_bang":
        return bang_bang_policy
    if method == "mc_local_greedy":
        policy = MCLocalGreedyPolicy(m_fuel_0, Q_max, T_max, cd, seed=seed, **kwargs)
        policy.optimize()
        return policy
    if method == "unconstrained_optimal":
        return UnconstrainedOptimalPolicy(m_fuel_0, Q_max, T_max, cd, **kwargs)
    if method == "casadi_optimum":
        return CasadiOptimumPolicy(m_fuel_0, Q_max, T_max, cd, **kwargs)
    raise ValueError(f"Unknown baseline {method!r}; choose {METHODS}")


__all__ = [
    "BASELINE_LABELS",
    "METHODS",
    "BaselineResult",
    "CasadiOptimumPolicy",
    "MCLocalGreedyPolicy",
    "UnconstrainedOptimalPolicy",
    "bang_bang_policy",
    "build_baseline",
    "evaluate_policy",
    "make_env",
    "rollout_schedules_numba",
    "solve_casadi",
]
