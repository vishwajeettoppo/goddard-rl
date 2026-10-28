"""Validate the shared RK4 evaluator against paper Table 7.

Run: .venv/bin/python scripts/validate_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env import BatchedGoddardEnv, GoddardEnv, singular_arc_u  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "configs" / "scenarios.yaml").read_text())
SCENARIOS = CONFIG["scenarios"]
CONSTS = CONFIG["physical_constants"]


def roll(env: GoddardEnv, policy) -> float:
    obs, info = env.reset()
    done = False
    reward = 0.0
    while not done:
        u = policy(env, info)
        obs, reward, term, trunc, info = env.step(u)
        done = term or trunc
    return env.h_apogee


def run_bang_bang():
    print("=== Bang-Bang (u=1) vs Table 7 ===")
    errors = []
    for name, sc in SCENARIOS.items():
        env = GoddardEnv(
            params={
                "m_fuel_0": sc["m_fuel_0"],
                "T_max": CONSTS["Tmax"],
                "cd": CONSTS["cd"],
                "Q_max": sc["Q_max"],
            },
            feature_mode="base",
            record_trajectory=False,
        )
        apogee = roll(env, lambda e, i: 1.0)
        ref = sc["paper_apogee_m"]["bang_bang"]
        err = apogee - ref
        errors.append(abs(err))
        print(f"{name:5s} fuel={sc['m_fuel_0']:5.0f} Q={sc['Q_max']:6.0f} "
              f"ours={apogee:9.2f} paper={ref:8d} diff={err:+8.2f}")
    print(f"max |diff| = {max(errors):.2f} m, mean = {np.mean(errors):.2f} m")
    return max(errors)


def run_unconstrained():
    print("\n=== Unconstrained Optimal vs Table 7 ===")
    errors = []
    for name, sc in SCENARIOS.items():
        env = GoddardEnv(
            params={
                "m_fuel_0": sc["m_fuel_0"],
                "T_max": CONSTS["Tmax"],
                "cd": CONSTS["cd"],
                "Q_max": sc["Q_max"],
            },
        )
        apogee = roll(env, lambda e, i: singular_arc_u(
            float(e.state[0]), float(e.state[1]), float(e.state[2]),
            CONSTS["Tmax"], CONSTS["cd"], sc["Q_max"], use_cap=False,
        ))
        ref = sc["paper_apogee_m"]["unconstrained_optimal"]
        err = apogee - ref
        errors.append(abs(err))
        print(f"{name:5s} ours={apogee:9.2f} paper={ref:8d} diff={err:+8.2f}")
    print(f"max |diff| = {max(errors):.2f} m, mean = {np.mean(errors):.2f} m")
    return max(errors)


def bench_batched():
    print("\n=== BatchedGoddardEnv speed benchmark (256 envs) ===")
    import time
    env = BatchedGoddardEnv(n_blocks=32, n_twins=8, seed=0)
    env.reset()
    actions = np.full(env.n_envs, 1.0)
    t0 = time.time()
    for _ in range(1000):
        env.step(actions)
    dt = time.time() - t0
    print(f"1000 steps x 256 envs: {dt:.3f}s  ({dt / 1000 * 1e6:.1f} us/step)")
    print(f"apogee mean={env.h_apogee.mean():.1f} m  range=({env.h_apogee.min():.0f}, {env.h_apogee.max():.0f})")


if __name__ == "__main__":
    e1 = run_bang_bang()
    e2 = run_unconstrained()
    bench_batched()
    print(f"\nBang-bang max error: {e1:.2f} m | Unconstrained max error: {e2:.2f} m")
