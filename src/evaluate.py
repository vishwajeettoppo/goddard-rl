"""Evaluate trained agents and baselines on the held-out 18-scenario grid.

The dynamics are deterministic, so every method gets exactly one rollout per
scenario.  Trained agents are evaluated with the Beta mean (CARL's table used
the mean), and optionally with the Beta mode to expose the train/eval mismatch
that R6 targets.

Examples
--------
python src/evaluate.py --ablations R0,R1,R4 --out results/eval_rl.csv
python src/evaluate.py --ablations all --eval-mode mode --out results/eval_mode.csv
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import GoddardEnv  # noqa: E402
from src.ppo import BetaActorCritic  # noqa: E402

ABLATION_NAMES = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")

COLUMNS = [
    "scenario",
    "m_fuel_0",
    "group",
    "method",
    "eval_mode",
    "apogee",
    "t_apogee",
    "n_steps",
    "n_clamp",
    "max_q",
    "q_violation_frac",
    "runtime_s",
]


def load_scenarios() -> dict[str, Any]:
    config = yaml.safe_load((ROOT / "configs" / "scenarios.yaml").read_text())
    return config


def group_of(name: str) -> str:
    return {"L": "light", "N": "nominal", "H": "heavy"}[name[0]]


def rollout_agent(
    model: BetaActorCritic,
    env: GoddardEnv,
    *,
    eval_mode: str = "mean",
    record_trajectory: bool = False,
) -> dict[str, Any]:
    env.record_trajectory = bool(record_trajectory)
    obs, _info = env.reset()
    done = False
    max_q = 0.0
    violations = 0
    start = time.perf_counter()
    while not done:
        throttle, _logp, _value = model.act(
            obs[None, :],
            deterministic=eval_mode == "mean",
            mode=eval_mode == "mode",
        )
        u = float(throttle[0])
        obs, _reward, terminated, truncated, info = env.step(u)
        done = bool(terminated or truncated)
        max_q = max(max_q, float(info["q"]))
        if float(info["q"]) > env.params["Q_max"] * (1.0 + 1e-9):
            violations += 1
    runtime = time.perf_counter() - start
    out = {
        "apogee": float(env.h_apogee),
        "t_apogee": float(env.t_now),
        "n_steps": int(env.n_steps),
        "n_clamp": int(env.n_clamp),
        "max_q": max_q,
        "q_violation_frac": violations / max(env.n_steps, 1),
        "runtime_s": runtime,
    }
    if record_trajectory:
        keys = ("t", "h", "v", "m", "q", "u_cmd", "u_applied", "clamped")
        out["trajectory"] = {
            key: np.asarray([row[key] for row in env.trajectory]) for key in keys
        }
    return out


def find_checkpoints(pattern: str) -> list[Path]:
    paths = [Path(p) for p in sorted(glob.glob(str(ROOT / pattern)))]
    return [p for p in paths if p.suffix == ".pt"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablations", default="all", help="R0,R1,... or all")
    parser.add_argument("--checkpoint-glob", default="results/checkpoints/{ablation}_seed*.pt")
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--eval-mode", default="mean", choices=["mean", "mode", "sample"])
    parser.add_argument("--out", default="results/eval_rl.csv")
    parser.add_argument("--traj-dir", default="results/trajectories")
    parser.add_argument("--include-baselines", action="store_true", default=True)
    return _run(parser.parse_args(argv))


def _run(args: argparse.Namespace) -> int:
    config = load_scenarios()
    scenarios = config["scenarios"]
    names = list(scenarios) if args.scenarios is None else args.scenarios.split(",")
    ablations = list(ABLATION_NAMES) if args.ablations == "all" else args.ablations.split(",")

    checkpoints: list[tuple[str, int, Path]] = []
    for ablation in ablations:
        pattern = args.checkpoint_glob.format(ablation=ablation)
        for path in find_checkpoints(pattern):
            try:
                seed = int(path.stem.split("seed")[-1])
            except ValueError:
                seed = -1
            checkpoints.append((ablation, seed, path))
    if not checkpoints:
        raise SystemExit("No checkpoints found; run src/train.py first")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for ablation, seed, path in checkpoints:
        model, meta = BetaActorCritic.load(path)
        method = f"{ablation}_seed{seed}"
        target_mode = meta.get("target_mode")
        feature_mode = meta.get("feature_mode", "base")
        for name in names:
            scenario = scenarios[name]
            m_fuel_0 = float(scenario["m_fuel_0"])
            Q_max = float(scenario["Q_max"])
            env = GoddardEnv(
                params={
                    "m_fuel_0": m_fuel_0,
                    "T_max": config["physical_constants"]["Tmax"],
                    "cd": config["physical_constants"]["cd"],
                    "Q_max": Q_max,
                },
                feature_mode=feature_mode,
                target_mode=target_mode,
                record_trajectory=True,
            )
            result = rollout_agent(model, env, eval_mode=args.eval_mode, record_trajectory=True)
            traj_dir = ROOT / args.traj_dir
            traj_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                traj_dir / f"{name}__{method}.npz", **result["trajectory"]
            )
            rows.append(
                {
                    "scenario": name,
                    "m_fuel_0": m_fuel_0,
                    "group": group_of(name),
                    "method": method,
                    "eval_mode": args.eval_mode,
                    "apogee": round(result["apogee"], 3),
                    "t_apogee": round(result["t_apogee"], 2),
                    "n_steps": result["n_steps"],
                    "n_clamp": result["n_clamp"],
                    "max_q": round(result["max_q"], 3),
                    "q_violation_frac": round(result["q_violation_frac"], 6),
                    "runtime_s": round(result["runtime_s"], 3),
                }
            )
            print(
                f"{method:16s} {name:5s} apogee={result['apogee']:9.1f} "
                f"clamp={result['n_clamp']:4d} {result['runtime_s']:5.2f}s",
                flush=True,
            )

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["ablation"] = df["method"].str.split("_seed").str[0]
    df["seed"] = df["method"].str.split("_seed").str[-1].astype(int)
    df.to_csv(out_path, index=False)

    table = summarise(df, config, out_path.parent)
    print("\n=== Ablation summary (mean over seeds) ===")
    print(table.to_string(index=False))
    print(f"\nWrote {out_path.relative_to(ROOT)} and results/ablation_table.csv")
    return 0


def summarise(df: pd.DataFrame, config: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    """Build the report table: mean/std apogee and percent of best-known."""

    baselines_path = ROOT / "results" / "baselines.csv"
    best_known: dict[str, float] = {}
    paper_best: dict[str, float] = {}
    for name, scenario in config["scenarios"].items():
        paper_best[name] = float(max(scenario["paper_apogee_m"].values()))
        best_known[name] = paper_best[name]
    baseline_rows: list[dict[str, Any]] = []
    if baselines_path.exists():
        base = pd.read_csv(baselines_path)
        for name, group in base.groupby("scenario"):
            best_known[name] = max(best_known.get(name, 0.0), float(group["apogee"].max()))
        baseline_rows = base.to_dict("records")

    grouped = (
        df.groupby(["ablation", "scenario"])["apogee"]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped["pct_best"] = grouped.apply(
        lambda row: 100.0 * row["mean"] / best_known.get(row["scenario"], row["mean"]), axis=1
    )
    grouped["group"] = grouped["scenario"].map(group_of)

    summary_rows: list[dict[str, Any]] = []
    for ablation, group in grouped.groupby("ablation"):
        summary_rows.append(
            {
                "method": ablation,
                "kind": "RL",
                "mean_apogee": round(float(group["mean"].mean()), 1),
                "std_over_seeds": round(float(group["std"].mean(skipna=True)), 1),
                "mean_pct_best": round(float(group["pct_best"].mean()), 2),
                "worst_pct_best": round(float(group["pct_best"].min()), 2),
                "light_pct_best": round(
                    float(group.loc[group["group"] == "light", "pct_best"].mean()), 2
                ),
                "nominal_pct_best": round(
                    float(group.loc[group["group"] == "nominal", "pct_best"].mean()), 2
                ),
                "heavy_pct_best": round(
                    float(group.loc[group["group"] == "heavy", "pct_best"].mean()), 2
                ),
            }
        )

    if baseline_rows:
        base = pd.DataFrame(baseline_rows)
        base["pct_best"] = base.apply(
            lambda row: 100.0 * row["apogee"] / best_known.get(row["scenario"], row["apogee"]),
            axis=1,
        )
        base["group"] = base["scenario"].map(group_of)
        for method, group in base.groupby("method"):
            summary_rows.append(
                {
                    "method": method,
                    "kind": "baseline",
                    "mean_apogee": round(float(group["apogee"].mean()), 1),
                    "std_over_seeds": 0.0,
                    "mean_pct_best": round(float(group["pct_best"].mean()), 2),
                    "worst_pct_best": round(float(group["pct_best"].min()), 2),
                    "light_pct_best": round(
                        float(group.loc[group["group"] == "light", "pct_best"].mean()), 2
                    ),
                    "nominal_pct_best": round(
                        float(group.loc[group["group"] == "nominal", "pct_best"].mean()), 2
                    ),
                    "heavy_pct_best": round(
                        float(group.loc[group["group"] == "heavy", "pct_best"].mean()), 2
                    ),
                }
            )

    table = pd.DataFrame(summary_rows).sort_values(["kind", "mean_pct_best"], ascending=[True, False])
    table.to_csv(out_dir / "ablation_table.csv", index=False)
    grouped.to_csv(out_dir / "ablation_per_scenario.csv", index=False)
    return table


if __name__ == "__main__":
    raise SystemExit(main())
