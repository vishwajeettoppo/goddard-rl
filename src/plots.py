"""Figures for the report, slides and demo notebook.

All plots read the artefacts produced by the other entry points, so they can be
regenerated at any time without retraining:

* ``results/logs/<ablation>_seed<seed>.jsonl`` -- PPO update logs.
* ``results/baselines.csv`` -- non-RL baselines on the 18 scenarios.
* ``results/eval_rl.csv`` -- trained agents on the same scenarios.
* ``results/trajectories/*.npz`` -- deterministic rollouts.

Run ``python src/plots.py`` to regenerate every figure in ``results/figures``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ABLATIONS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")
LABELS = {
    "R0": "R0 unimodal Beta",
    "R1": "R1 boundary Beta",
    "R2": "R2 + burnout penalty",
    "R3": "R3 + BC warm start",
    "R4": "R4 + residual action",
    "R5": "R5 + potential shaping",
    "R6": "R6 + Q-cap / mode eval",
    "bang_bang": "Bang-Bang",
    "mc_local_greedy": "MC-Local-Greedy",
    "unconstrained_optimal": "Unconstrained Optimal",
    "casadi_optimum": "CasADi optimum",
}
COLORS = {
    "R0": "#d1495b",
    "R1": "#edae49",
    "R2": "#00798c",
    "R3": "#30638e",
    "R4": "#003d5b",
    "R5": "#8f2d56",
    "R6": "#5c8001",
    "bang_bang": "#8d99ae",
    "mc_local_greedy": "#2a9d8f",
    "unconstrained_optimal": "#e76f51",
    "casadi_optimum": "#000000",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def load_logs(log_dir: Path | list[Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {}
    directories = log_dir if isinstance(log_dir, list) else [log_dir]
    paths = [path for directory in directories for path in sorted(directory.glob("*.jsonl"))]
    for path in paths:
        ablation = path.name.split("_seed")[0]
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        frames.setdefault(ablation, []).append(pd.DataFrame(rows))
    return {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}


def _aggregate(frames: list[pd.DataFrame], column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged = pd.concat([frame[["update", column]] for frame in frames])
    grouped = merged.groupby("update")[column]
    mean = grouped.mean()
    std = grouped.std().fillna(0.0)
    return mean.index.to_numpy(), mean.to_numpy(), std.to_numpy()


def plot_learning_curves(logs: dict[str, pd.DataFrame], fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ablation in ABLATIONS:
        if ablation not in logs:
            continue
        frames = [logs[ablation]]
        x, mean, std = _aggregate(frames, "mean_apogee_train")
        axes[0].plot(x, mean, label=LABELS.get(ablation, ablation), color=COLORS.get(ablation))
        axes[0].fill_between(x, mean - std, mean + std, color=COLORS.get(ablation), alpha=0.15)
        entropy_x, entropy_mean, entropy_std = _aggregate(frames, "entropy")
        axes[1].plot(
            entropy_x,
            entropy_mean,
            label=LABELS.get(ablation, ablation),
            color=COLORS.get(ablation),
        )
        axes[1].fill_between(
            entropy_x,
            entropy_mean - entropy_std,
            entropy_mean + entropy_std,
            color=COLORS.get(ablation),
            alpha=0.15,
        )
    axes[0].set_xlabel("PPO update")
    axes[0].set_ylabel("mean apogee during training (m)")
    axes[0].set_title("Training apogee (training distribution)")
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("Beta policy entropy (nats)")
    axes[1].set_title("Policy entropy: R0 collapse vs boundary mode")
    axes[1].axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    axes[0].legend(frameon=False, ncol=1)
    fig.tight_layout()
    path = fig_dir / "learning_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_entropy_collapse(logs: dict[str, pd.DataFrame], fig_dir: Path) -> Path:
    fig, axis = plt.subplots(figsize=(5.4, 3.4))
    for ablation in ("R0", "R1"):
        if ablation not in logs:
            continue
        x, mean, std = _aggregate([logs[ablation]], "entropy")
        axis.plot(x, mean, label=LABELS.get(ablation, ablation), color=COLORS.get(ablation))
        axis.fill_between(x, mean - std, mean + std, color=COLORS.get(ablation), alpha=0.15)
    axis.set_xlabel("PPO update")
    axis.set_ylabel("policy entropy (nats)")
    axis.set_title("Entropy of the Beta head during training")
    axis.legend(frameon=False)
    finite = [
        value
        for ablation in ("R0", "R1")
        if ablation in logs
        for value in _aggregate([logs[ablation]], "entropy")[1]
        if np.isfinite(value)
    ]
    if finite:
        low, high = min(finite), max(finite)
        pad = 0.08 * max(high - low, 0.05)
        axis.set_ylim(low - pad, high + pad)
    fig.tight_layout()
    path = fig_dir / "entropy_collapse.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def load_eval(eval_path: Path) -> pd.DataFrame:
    if not eval_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(eval_path)
    if "ablation" not in frame.columns:
        frame["ablation"] = frame["method"].str.split("_seed").str[0]
    return frame


def best_known_table(eval_df: pd.DataFrame, baselines: pd.DataFrame, scenarios: dict[str, Any]) -> dict[str, float]:
    best: dict[str, float] = {
        name: float(max(values["paper_apogee_m"].values())) for name, values in scenarios.items()
    }
    for frame in (eval_df, baselines):
        if frame.empty:
            continue
        for name, group in frame.groupby("scenario"):
            best[name] = max(best.get(name, 0.0), float(group["apogee"].max()))
    return best


def plot_ablation_bars(
    eval_df: pd.DataFrame, baselines: pd.DataFrame, scenarios: dict[str, Any], fig_dir: Path
) -> Path:
    best = best_known_table(eval_df, baselines, scenarios)
    rows: list[dict[str, Any]] = []
    for ablation in ABLATIONS:
        subset = eval_df[eval_df["ablation"] == ablation]
        if subset.empty:
            continue
        per_scenario = subset.groupby("scenario")["apogee"].mean()
        pct = [100.0 * value / best[name] for name, value in per_scenario.items()]
        rows.append({"method": ablation, "pct": float(np.mean(pct)), "std": float(np.std(pct))})
    for method, group in baselines.groupby("method"):
        pct = [100.0 * row.apogee / best[row.scenario] for row in group.itertuples()]
        rows.append({"method": method, "pct": float(np.mean(pct)), "std": float(np.std(pct))})
    table = pd.DataFrame(rows).sort_values("pct")
    fig, axis = plt.subplots(figsize=(6.4, 0.36 * len(table) + 1.2))
    axis.barh(
        [LABELS.get(name, name) for name in table["method"]],
        table["pct"],
        xerr=table["std"],
        color=[COLORS.get(name, "#555") for name in table["method"]],
        alpha=0.88,
    )
    axis.set_xlabel("% of best-known apogee (mean over 18 scenarios)")
    axis.set_xlim(60, 106)
    axis.set_title("Ablation ladder: each change adds one ingredient")
    for index, value in enumerate(table["pct"]):
        axis.text(min(value + 0.6, 100.8), index, f"{value:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    path = fig_dir / "ablation_bars.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_heatmap(
    eval_df: pd.DataFrame, baselines: pd.DataFrame, scenarios: dict[str, Any], fig_dir: Path
) -> Path:
    best = best_known_table(eval_df, baselines, scenarios)
    matrix: dict[str, dict[str, float]] = {}
    for method, group in baselines.groupby("method"):
        matrix[method] = {
            row.scenario: 100.0 * row.apogee / best[row.scenario] for row in group.itertuples()
        }
    for ablation in ABLATIONS:
        subset = eval_df[eval_df["ablation"] == ablation]
        if subset.empty:
            continue
        per = subset.groupby("scenario")["apogee"].mean()
        matrix[ablation] = {name: 100.0 * value / best[name] for name, value in per.items()}
    frame = pd.DataFrame(matrix).T
    order = [name for name in scenarios if name in frame.columns]
    frame = frame[order]
    methods = [m for m in ("casadi_optimum", "mc_local_greedy", "unconstrained_optimal", "bang_bang") if m in frame.index]
    methods += [m for m in ABLATIONS if m in frame.index]
    frame = frame.loc[methods]
    fig, axis = plt.subplots(figsize=(11.0, 0.34 * len(frame) + 1.6))
    image = axis.imshow(frame.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=55, vmax=100)
    axis.set_xticks(range(len(frame.columns)), frame.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(frame.index)), [LABELS.get(m, m) for m in frame.index])
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            value = frame.iloc[i, j]
            if np.isfinite(value):
                axis.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.6)
    axis.set_title("% of best-known apogee per scenario")
    fig.colorbar(image, ax=axis, shrink=0.7, label="% of best known")
    fig.tight_layout()
    path = fig_dir / "heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_trajectory(
    scenario: str,
    methods: list[str],
    traj_dir: Path,
    fig_dir: Path,
) -> Path | None:
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.0), sharex=True)
    plotted = False
    candidates = [f"{scenario}__{method}.npz" for method in methods]
    candidates += [f"{scenario}__{method}_seed42.npz" for method in methods]
    candidates += [f"{scenario}__{method}_seed101.npz" for method in methods]
    plotted_methods: set[str] = set()
    for path in candidates:
        file = traj_dir / path
        if not file.exists():
            continue
        base_method = path.split("__")[1].replace(".npz", "").split("_seed")[0]
        if base_method in plotted_methods:
            continue
        plotted_methods.add(base_method)
        data = np.load(file)
        method = path.split("__")[1].replace(".npz", "")
        label = LABELS.get(method.split("_seed")[0], method)
        color = COLORS.get(method.split("_seed")[0])
        axes[0].plot(data["t"], data["h"], label=label, color=color)
        axes[1].plot(data["t"], data["v"], color=color)
        axes[2].plot(data["t"], data["u_applied"], color=color)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_ylabel("altitude (m)")
    axes[1].set_ylabel("velocity (m/s)")
    axes[2].set_ylabel("throttle (applied)")
    axes[2].set_xlabel("time (s)")
    axes[0].set_title(f"Scenario {scenario}: altitude, velocity, throttle")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = fig_dir / f"trajectory_{scenario}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_policy_shapes(fig_dir: Path, scenario: str = "H_ST") -> Path | None:
    """Beta densities along one deterministic rollout for R0 and R1.

    The unimodal head (epsilon = 1) can only put a bump near its mean; the
    boundary head (epsilon = 0.01) can put mass on 0 and 1, which is where
    bang-bang-like throttles live.  R4 is excluded because its Beta is over the
    residual latent, not over the throttle.
    """

    from src.env import GoddardEnv
    from src.ppo import BetaActorCritic

    checkpoints = {}
    for ablation in ("R0", "R1"):
        found = sorted((ROOT / "results" / "checkpoints").glob(f"{ablation}_seed42.pt"))
        if found:
            checkpoints[ablation] = found[0]
    if len(checkpoints) < 2:
        return None

    grid = np.linspace(0.001, 0.999, 200)
    times = (5.0, 20.0, 40.0)
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(3.4 * len(checkpoints), 3.1), sharey=True)
    if len(checkpoints) == 1:
        axes = [axes]
    scenario_cfg = yaml.safe_load((ROOT / "configs" / "scenarios.yaml").read_text())["scenarios"][scenario]
    for axis, (ablation, path) in zip(axes, checkpoints.items(), strict=False):
        model, meta = BetaActorCritic.load(path)
        env = GoddardEnv(
            params={
                "m_fuel_0": scenario_cfg["m_fuel_0"],
                "T_max": 200000.0,
                "cd": 1.5,
                "Q_max": scenario_cfg["Q_max"],
            },
            feature_mode=meta.get("feature_mode", "base"),
            target_mode=meta.get("target_mode"),
        )
        obs, _ = env.reset()
        done = False
        while not done:
            throttle, _logp, _value = model.act(obs[None, :], deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(float(throttle[0]))
            done = bool(terminated or truncated)
            for stamp in times:
                if abs(env.t_now - stamp) < 0.05:
                    with torch.no_grad():
                        alpha, beta = model.concentrations(torch.as_tensor(obs[None, :]))
                        dist = torch.distributions.Beta(alpha, beta)
                        log_density = dist.log_prob(torch.as_tensor(grid, dtype=torch.float32))
                    density = np.exp(log_density.detach().numpy())
                    axis.plot(grid, density, label=f"t = {stamp:.0f} s")
        axis.set_title(LABELS.get(ablation, ablation))
        axis.set_xlabel("throttle u")
        axis.set_xlim(0, 1)
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Beta density")
    fig.suptitle("Beta policy shape along one rollout (axes are the sampled latent)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = fig_dir / "policy_shapes.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ppo_diagnostics(logs: dict[str, pd.DataFrame], fig_dir: Path) -> Path:
    """Approximate KL, value loss and entropy per PPO update.

    These are the three numbers that tell you whether PPO is actually learning in a
    stable way: the KL guard should stay below its threshold, the value loss should
    fall or plateau, and the entropy should not collapse before the policy is good.
    """

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4))
    panels = (
        ("kl", "approximate KL between old and new policy", True),
        ("value_loss", "critic loss (reward units: 1 = 1000 m)", True),
        ("entropy", "policy entropy (nats)", False),
    )
    for axis, (column, label, log_y) in zip(axes, panels, strict=False):
        for ablation in ABLATIONS:
            if ablation not in logs:
                continue
            x, mean, std = _aggregate([logs[ablation]], column)
            axis.plot(x, mean, label=LABELS.get(ablation, ablation), color=COLORS.get(ablation))
            if not log_y:
                axis.fill_between(x, mean - std, mean + std, color=COLORS.get(ablation), alpha=0.15)
        axis.set_xlabel("PPO update")
        axis.set_ylabel(label)
        if log_y:
            axis.set_yscale("log")
    axes[0].axhline(0.05, color="black", linestyle=":", linewidth=0.8)
    axes[0].axhline(0.075, color="black", linestyle="--", linewidth=0.8)
    from matplotlib.lines import Line2D

    axes[0].legend(
        handles=[
            Line2D([0], [0], color="black", linestyle=":", linewidth=0.8, label="target KL 0.05"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="early stop at 1.5x"),
        ],
        frameon=False,
        fontsize=6,
        loc="lower left",
    )
    axes[0].set_title("KL stayed tiny except for the residual action")
    axes[1].set_title("Critic loss grows with altitude scale")
    axes[2].set_title("Entropy: BC starts low, all drift down")
    axes[2].axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = fig_dir / "ppo_diagnostics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_seed_spread(eval_df: pd.DataFrame, fig_dir: Path) -> Path | None:
    if eval_df.empty:
        return None
    grouped = eval_df.groupby(["ablation", "scenario"])["apogee"].agg(["mean", "std", "min", "max"]).reset_index()
    fig, axis = plt.subplots(figsize=(7.2, 3.4))
    for index, ablation in enumerate(ABLATIONS):
        subset = grouped[grouped["ablation"] == ablation]
        if subset.empty:
            continue
        axis.errorbar(
            index + 1,
            subset["mean"].mean(),
            yerr=[[(subset["mean"] - subset["min"]).mean()], [(subset["max"] - subset["mean"]).mean()]],
            fmt="o",
            color=COLORS.get(ablation),
            capsize=4,
        )
    axis.set_xticks(range(1, len(ABLATIONS) + 1), [LABELS[a] for a in ABLATIONS], rotation=20, ha="right")
    axis.set_ylabel("apogee (m), mean over scenarios")
    axis.set_title("Seed spread per ablation (min/mean/max over seeds and scenarios)")
    fig.tight_layout()
    path = fig_dir / "seed_spread.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--fig-dir", default="results/figures")
    parser.add_argument("--eval", default="results/eval_rl.csv")
    parser.add_argument("--baselines", default="results/baselines.csv")
    parser.add_argument("--scenarios", default="configs/scenarios.yaml")
    parser.add_argument("--scenario", default="H_ST", help="scenario for the profile plot")
    parser.add_argument(
        "--traj-methods",
        default="bang_bang,mc_local_greedy,unconstrained_optimal,R4",
    )
    args = parser.parse_args(argv)

    fig_dir = ROOT / args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    _style()

    config = yaml.safe_load((ROOT / args.scenarios).read_text())
    scenarios = config["scenarios"]
    logs = load_logs([ROOT / args.log_dir, ROOT / "results" / "checkpoints"])
    eval_df = load_eval(ROOT / args.eval)
    baselines = pd.read_csv(ROOT / args.baselines) if (ROOT / args.baselines).exists() else pd.DataFrame()

    outputs = []
    if logs:
        outputs.append(plot_learning_curves(logs, fig_dir))
        outputs.append(plot_entropy_collapse(logs, fig_dir))
    if logs:
        outputs.append(plot_ppo_diagnostics(logs, fig_dir))
    shapes = plot_policy_shapes(fig_dir)
    if shapes:
        outputs.append(shapes)
    if not eval_df.empty and not baselines.empty:
        outputs.append(plot_ablation_bars(eval_df, baselines, scenarios, fig_dir))
        outputs.append(plot_heatmap(eval_df, baselines, scenarios, fig_dir))
        spread = plot_seed_spread(eval_df, fig_dir)
        if spread:
            outputs.append(spread)
    trajectory = plot_trajectory(
        args.scenario,
        [m.strip() for m in args.traj_methods.split(",")],
        ROOT / "results" / "trajectories",
        fig_dir,
    )
    if trajectory:
        outputs.append(trajectory)
    for path in outputs:
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
