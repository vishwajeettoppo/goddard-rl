"""Roll every non-RL baseline out on the 18 evaluation scenarios.

Examples
--------
python scripts/run_baselines.py --methods bang_bang,unconstrained_optimal
python scripts/run_baselines.py --methods mc_local_greedy --scenarios L_MT,N_ST
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import METHODS, build_baseline, evaluate_policy, make_env  # noqa: E402

COLUMNS = [
    "scenario",
    "m_fuel_0",
    "Q_max",
    "method",
    "apogee",
    "t_apogee",
    "n_steps",
    "n_clamp",
    "max_q",
    "q_violation_frac",
    "runtime_s",
    "paper_apogee",
    "pct_of_paper",
    "pct_of_best_known",
    "opt_runtime_s",
    "replay_apogee",
]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "scenarios.yaml").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--scenarios", default=None, help="comma-separated subset")
    parser.add_argument("--out", default="results/baselines.csv")
    parser.add_argument("--traj-dir", default="results/trajectories")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config()
    constants = config["physical_constants"]
    all_scenarios = config["scenarios"]
    names = args.scenarios.split(",") if args.scenarios else list(all_scenarios)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s) {unknown}; choose from {METHODS}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame(columns=COLUMNS)
    if out_path.exists() and not args.no_resume:
        existing = pd.read_csv(out_path)
        if not set(COLUMNS).issubset(existing.columns):
            existing = pd.DataFrame(columns=COLUMNS)
    done = set(zip(existing.get("scenario", []), existing.get("method", [])))

    traj_dir = ROOT / args.traj_dir
    rows: list[dict] = []
    for name in names:
        scenario = all_scenarios[name]
        m_fuel_0 = float(scenario["m_fuel_0"])
        Q_max = float(scenario["Q_max"])
        paper = scenario["paper_apogee_m"]
        for method in methods:
            if (name, method) in done:
                continue
            t_build = time.perf_counter()
            policy = build_baseline(method, {"m_fuel_0": m_fuel_0, "Q_max": Q_max}, seed=args.seed)
            build_seconds = time.perf_counter() - t_build
            env = make_env(
                m_fuel_0, Q_max, record_trajectory=True
            )
            result = evaluate_policy(env, policy, method=method, record_trajectory=True)
            paper_key = {
                "bang_bang": "bang_bang",
                "mc_local_greedy": "mc_local_greedy",
                "unconstrained_optimal": "unconstrained_optimal",
                "casadi_optimum": "unconstrained_optimal",
            }[method]
            reference = float(paper[paper_key])
            if method == "casadi_optimum":
                reference = max(float(value) for value in paper.values())
            replay_apogee = result.apogee
            reported_apogee = result.apogee
            if method == "casadi_optimum":
                # The open-loop NLP solution is the mathematical optimum; its
                # replay in the RK4 environment is recorded separately because
                # Q-limited arcs are open-loop unstable.
                nlp_apogee = float(getattr(policy, "apogee", float("nan")))
                if nlp_apogee == nlp_apogee:
                    reported_apogee = nlp_apogee
            if result.trajectory:
                traj_dir.mkdir(parents=True, exist_ok=True)
                import numpy as np

                np.savez_compressed(
                    traj_dir / f"{name}__{method}.npz", **result.trajectory
                )
            rows.append(
                {
                    "scenario": name,
                    "m_fuel_0": m_fuel_0,
                    "Q_max": Q_max,
                    "method": method,
                    "apogee": round(reported_apogee, 3),
                    "t_apogee": round(result.t_apogee, 2),
                    "n_steps": result.n_steps,
                    "n_clamp": result.n_clamp,
                    "max_q": round(result.max_q, 3),
                    "q_violation_frac": round(result.q_violation_frac, 6),
                    "runtime_s": round(result.runtime_s, 3),
                    "paper_apogee": reference,
                    "pct_of_paper": round(100.0 * reported_apogee / reference, 2),
                    "opt_runtime_s": round(build_seconds, 3),
                    "replay_apogee": round(replay_apogee, 3),
                }
            )
            print(
                f"{name:5s} {method:22s} apogee={reported_apogee:9.1f} "
                f"paper={reference:8.0f} ({100.0 * reported_apogee / reference:6.2f}%) "
                f"clamp={result.n_clamp:4d} {result.runtime_s:6.2f}s"
            )

    df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    if not df.empty:
        df = df.drop_duplicates(subset=["scenario", "method"], keep="last")
        best_known = df["apogee"].groupby(df["scenario"]).max().rename("best_known_ours")
        paper_best = df.groupby("scenario")["paper_apogee"].max().rename("paper_best")
        best = pd.concat([best_known, paper_best], axis=1).max(axis=1)
        df["best_known"] = df["scenario"].map(best)
        df["pct_of_best_known"] = (100.0 * df["apogee"] / df["best_known"]).round(2)
        df = df[[*COLUMNS, "best_known"]]
        df = df.sort_values(["scenario", "method"]).reset_index(drop=True)
        df.to_csv(out_path, index=False)
        print(f"\nWrote {len(df)} rows to {out_path.relative_to(ROOT)}")
        pivot = df.pivot(index="scenario", columns="method", values="apogee")
        print(pivot.round(0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
