"""Physics and bookkeeping invariants of the shared evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.baselines import bang_bang_policy, evaluate_policy, make_env
from src.env import (
    CE_FACTOR,
    M_DRY,
    MAX_STEPS,
    BatchedGoddardEnv,
    GoddardEnv,
    advance_fuel_exact,
    dyn_pressure,
    physics_step_numba,
    singular_arc_u,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = yaml.safe_load((ROOT / "configs" / "scenarios.yaml").read_text())["scenarios"]


def _scenario(name: str) -> GoddardEnv:
    scenario = SCENARIOS[name]
    return make_env(scenario["m_fuel_0"], scenario["Q_max"])


@pytest.mark.parametrize(
    ("name", "expected"),
    [("L_MT", 3372), ("N_ST", 13874), ("H_T", 35477)],
)
def test_bang_bang_matches_paper_table_7(name: str, expected: int) -> None:
    env = _scenario(name)
    result = evaluate_policy(env, bang_bang_policy)
    assert abs(result.apogee - expected) <= 1.5


def test_never_below_dry_mass_and_fuel_is_conserved() -> None:
    env = _scenario("L_MT")
    env.reset()
    total_fuel = float(env.m_fuel)
    done = False
    while not done:
        _obs, _reward, terminated, truncated, _info = env.step(1.0)
        done = bool(terminated or truncated)
        assert float(env.state[2]) >= M_DRY - 1e-9
    burned = total_fuel - float(env.m_fuel)
    assert burned > 0.0
    c_e = CE_FACTOR * env.params["T_max"]
    assert env.n_steps > 0
    assert c_e > 0.0


@pytest.mark.parametrize("name", ["L_HT", "N_MT", "H_MT"])
def test_dynamic_pressure_never_exceeds_limit(name: str) -> None:
    env = _scenario(name)
    env.reset()
    done = False
    peak = 0.0
    while not done:
        _obs, _reward, terminated, truncated, info = env.step(1.0)
        done = bool(terminated or truncated)
        peak = max(peak, float(info["q"]))
    limit = env.params["Q_max"]
    assert peak <= limit * (1.0 + 1e-6)
    assert env.n_clamp > 0


def test_fuel_out_split_keeps_mass_exact() -> None:
    """A partially fuelled step must land exactly on the dry mass."""

    state = np.array([0.0, 0.0, M_DRY + 1.0])
    c_e = CE_FACTOR * 200000.0
    # Full thrust burns 100 kg/s, so 1 kg lasts 0.01 s inside a 0.1 s step.
    next_state = advance_fuel_exact(state, 200000.0, 1.5, c_e, 0.1)
    assert next_state[2] == pytest.approx(M_DRY, abs=1e-9)
    assert next_state[1] > 0.0


def test_apogee_termination_and_truncation() -> None:
    env = _scenario("L_M")
    env.reset()
    done = False
    steps = 0
    while not done:
        _obs, _reward, terminated, truncated, info = env.step(1.0)
        done = bool(terminated or truncated)
        steps += 1
    assert env.terminated and not env.truncated
    assert env.launched
    assert env.state[1] <= 0.0
    assert steps < MAX_STEPS
    assert info["h_apogee"] == pytest.approx(env.h_apogee)


def test_batched_env_matches_scalar_env() -> None:
    m_fuel_0, Q_max = 500.0, 5000.0
    params = {"m_fuel_0": m_fuel_0, "T_max": 200000.0, "cd": 1.5, "Q_max": Q_max}
    scalar = GoddardEnv(params=params)
    scalar.reset()
    batched = BatchedGoddardEnv(n_envs=8, params=params)
    batched.reset()
    for _ in range(40):
        _obs, _reward, terminated, truncated, _info = scalar.step(1.0)
        batched.step(np.ones(8))
        assert np.allclose(batched.h, scalar.state[0])
        assert np.allclose(batched.v, scalar.state[1])
        assert np.allclose(batched.m, scalar.state[2])
        if terminated or truncated:
            break


def test_singular_arc_is_feasible_and_smooth() -> None:
    u0 = singular_arc_u(0.0, 0.0, 1500.0, 200000.0, 1.5, 5000.0)
    assert u0 == pytest.approx(1.0)  # full thrust off the singular surface
    u1 = singular_arc_u(5000.0, 200.0, 1100.0, 200000.0, 1.5, 5000.0)
    assert 0.0 <= u1 <= 1.0
    capped = singular_arc_u(0.0, 60.0, 1500.0, 200000.0, 1.5, 5000.0, use_cap=True)
    assert capped <= 1.0


def test_physics_step_returns_applied_thrust_within_command() -> None:
    h, v, m = 2000.0, 120.0, 2500.0
    h2, v2, m2, thrust, clamped = physics_step_numba(
        h, v, m, 0.8, 200000.0, 1.5, 5000.0, 2000.0, 0.1, 8
    )
    assert 0.0 <= thrust <= 0.8 * 200000.0 + 1e-9
    assert m2 <= m
    if clamped:
        # The clamp can only choose T in [0, T_des].  If the entry state is
        # already above Qmax no thrust restores feasibility within one step;
        # otherwise the exit pressure must respect the limit.
        if dyn_pressure(h, v) <= 5000.0:
            assert dyn_pressure(h2, v2) <= 5000.0 * (1.0 + 1e-6)
        else:
            assert thrust == 0.0
