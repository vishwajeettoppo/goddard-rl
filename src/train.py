"""Train PPO agents for the Goddard ablation ladder R0--R6.

Each ablation changes exactly one thing relative to its parent, following
``ProjectPlan.md`` and CARL's revisit history:

======  =====================================================================
R0      Base 12-D observation, unimodal Beta head (``epsilon = 1``).
R1      R0 with the boundary-mode Beta head (``epsilon = 0.01``).
R2      R1 + burnout features and the one-sided early-burnout penalty.
R3      R2 + behaviour-cloning warm start from the singular-arc expert.
R4      R2 + residual action ``u = clip(u_singular + delta)``.
R5      R2 with potential-based energy shaping instead of the penalty.
R6      R2 + actor-side Q_max safety cap during training and mode-based eval.
======  =====================================================================

Example
-------
python src/train.py --ablation R1 --seed 42 --updates 120
python src/train.py --all --quick            # smoke test of the whole ladder
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import (  # noqa: E402
    CE_FACTOR,
    EVAL_CD,
    EVAL_T_MAX,
    MAX_STEPS,
    M_DRY,
    BatchedGoddardEnv,
    GoddardEnv,
    singular_arc_u,
)
from src.ppo import (  # noqa: E402
    BetaActorCritic,
    PPOConfig,
    Rollout,
    compute_gae,
    normalise_advantages,
)

ABLATIONS: dict[str, dict[str, Any]] = {
    "R0": {
        "description": "Baseline PPO, base observations, unimodal Beta head.",
        "beta_epsilon": 1.0,
        "feature_mode": "base",
        "reward_mode": "base",
        "target_mode": None,
        "residual": False,
        "use_actor_cap": False,
        "behavior_cloning": False,
    },
    "R1": {
        "description": "R0 with a boundary-mode Beta head.",
        "beta_epsilon": 0.01,
        "feature_mode": "base",
        "reward_mode": "base",
        "target_mode": None,
        "residual": False,
        "use_actor_cap": False,
        "behavior_cloning": False,
    },
    "R2": {
        "description": "R1 + burnout features + one-sided early-burnout penalty.",
        "beta_epsilon": 0.01,
        "feature_mode": "burnout",
        "reward_mode": "burnout_penalty",
        "target_mode": "rev1",  # linear in fuel: 4 * t_full = 0.04 * m_fuel_0
        "residual": False,
        "use_actor_cap": False,
        "behavior_cloning": False,
    },
    "R3": {
        "description": "R2 + behaviour-cloning warm start from the singular-arc law.",
        "beta_epsilon": 0.01,
        "feature_mode": "burnout",
        "reward_mode": "burnout_penalty",
        "target_mode": "rev1",
        "residual": False,
        "use_actor_cap": False,
        "behavior_cloning": True,
    },
    "R4": {
        "description": "R2 + residual action around the singular-arc feedback law.",
        "beta_epsilon": 0.01,
        "feature_mode": "residual",
        "reward_mode": "burnout_penalty",
        "target_mode": "rev1",
        "residual": True,
        "use_actor_cap": False,
        "behavior_cloning": False,
    },
    "R5": {
        "description": "R2 with potential-based energy shaping replacing the penalty.",
        "beta_epsilon": 0.01,
        "feature_mode": "burnout",
        "reward_mode": "potential_energy",
        "target_mode": "rev2_quadratic",
        "residual": False,
        "use_actor_cap": False,
        "behavior_cloning": False,
    },
    "R6": {
        "description": "R2 + actor-side Q_max cap in training and mode-based evaluation.",
        "beta_epsilon": 0.01,
        "feature_mode": "burnout",
        "reward_mode": "burnout_penalty",
        "target_mode": "rev1",
        "residual": False,
        "use_actor_cap": True,
        "behavior_cloning": False,
    },
}


def load_train_config(path: Path | None = None) -> dict[str, Any]:
    path = path or ROOT / "configs" / "train.yaml"
    return yaml.safe_load(path.read_text())


def resolve_ablation(name: str) -> dict[str, Any]:
    if name not in ABLATIONS:
        raise SystemExit(f"Unknown ablation {name!r}; choose from {sorted(ABLATIONS)}")
    return dict(ABLATIONS[name])


def build_env(
    ablation: Mapping[str, Any],
    *,
    seed: int,
    config: PPOConfig,
    quick: bool = False,
) -> BatchedGoddardEnv:
    env = BatchedGoddardEnv(
        n_blocks=config.n_blocks,
        n_twins=config.n_twins,
        seed=seed,
        feature_mode=str(ablation["feature_mode"]),
        reward_mode=str(ablation["reward_mode"]),
        target_mode=ablation.get("target_mode"),
        use_actor_cap=bool(ablation.get("use_actor_cap", False)),
        gamma=config.gamma,
        exclude_eval=True,
    )
    env.reset()
    return env


class PPOTrainer:
    def __init__(
        self,
        ablation_name: str,
        ablation: Mapping[str, Any],
        config: PPOConfig,
        *,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.name = ablation_name
        self.ablation = dict(ablation)
        self.config = config
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env = build_env(ablation, seed=seed, config=config)
        self.model = BetaActorCritic(
            self.env.OBS_DIM,
            config.hidden_sizes,
            beta_epsilon=float(ablation["beta_epsilon"]),
            residual=bool(ablation.get("residual", False)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr_start)
        self.history: list[dict[str, Any]] = []
        self.episode_apogees: deque[float] = deque(maxlen=256)
        self.episode_fuel: deque[float] = deque(maxlen=256)
        self.episode_clamp: deque[float] = deque(maxlen=256)
        self.total_episodes = 0
        self.total_steps = 0

    # -- rollout --------------------------------------------------------------
    def collect(self) -> Rollout:
        cfg = self.config
        obs = self.env.get_obs().copy()
        steps = cfg.rollout_steps
        obs_buf = np.zeros((steps, self.env.n_envs, self.env.OBS_DIM), dtype=np.float32)
        act_buf = np.zeros((steps, self.env.n_envs), dtype=np.float32)
        logp_buf = np.zeros((steps, self.env.n_envs), dtype=np.float32)
        val_buf = np.zeros((steps, self.env.n_envs), dtype=np.float32)
        rew_buf = np.zeros((steps, self.env.n_envs), dtype=np.float32)
        done_buf = np.zeros((steps, self.env.n_envs), dtype=np.float32)
        base_buf = np.zeros((steps, self.env.n_envs), dtype=np.float32)
        apogees: list[float] = []
        clamps: list[float] = []

        for t in range(steps):
            base = obs[:, -1].copy() if self.model.residual else np.zeros(self.env.n_envs)
            actions, log_probs, values = self.model.act(obs)
            obs_buf[t] = obs
            act_buf[t] = actions.astype(np.float32)
            logp_buf[t] = log_probs.astype(np.float32)
            val_buf[t] = values.astype(np.float32)
            base_buf[t] = base.astype(np.float32)

            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            done = np.logical_or(terminated, truncated)
            rew_buf[t] = rewards.astype(np.float32)
            done_buf[t] = done.astype(np.float32)

            finished = np.flatnonzero(done)
            if finished.size:
                for i in finished:
                    apogees.append(float(info["h_apogee"][i]))
                    self.episode_apogees.append(float(info["h_apogee"][i]))
                    clamps.append(float(info["n_clamp_steps"][i]) / max(int(info["n_steps"][i]), 1))
                    self.episode_clamp.append(clamps[-1])
                self.total_episodes += int(finished.size)
                self.env.reset_indices(finished)
                obs = self.env.get_obs().copy()
            else:
                obs = next_obs
            self.total_steps += self.env.n_envs

        last_input = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            last_values = self.model.value(last_input).cpu().numpy().astype(np.float64)

        return Rollout(
            obs=obs_buf,
            actions=act_buf,
            log_probs=logp_buf,
            values=val_buf,
            rewards=rew_buf,
            dones=done_buf,
            base_actions=base_buf,
            last_values=last_values,
            episode_apogees=apogees,
            episode_clamp_fracs=clamps,
        )

    # -- PPO update -----------------------------------------------------------
    def update(self, rollout: Rollout, progress: float) -> dict[str, float]:
        cfg = self.config
        adv, returns = compute_gae(
            rollout.rewards.astype(np.float64),
            rollout.values.astype(np.float64),
            rollout.dones.astype(np.float64),
            rollout.last_values if rollout.last_values is not None else np.zeros(self.env.n_envs),
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        adv = normalise_advantages(adv)

        obs = rollout.obs.reshape(-1, self.env.OBS_DIM)
        actions = rollout.actions.reshape(-1)
        old_logp = rollout.log_probs.reshape(-1)
        base = rollout.base_actions.reshape(-1)
        adv_flat = adv.reshape(-1).astype(np.float32)
        ret_flat = returns.reshape(-1).astype(np.float32)

        n = obs.shape[0]
        batch_size = min(cfg.mini_batch_size, n)
        lr = cfg.lr_start + (cfg.lr_end - cfg.lr_start) * progress
        ent_coef = cfg.ent_coef_start + (cfg.ent_coef_end - cfg.ent_coef_start) * progress
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        base_t = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv_flat, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret_flat, dtype=torch.float32, device=self.device)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_updates = 0
        stop_early = False
        for _epoch in range(cfg.ppo_epochs):
            permutation = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = permutation[start : start + batch_size]
                if idx.numel() < 2:
                    continue
                b_obs, b_act = obs_t[idx], act_t[idx]
                b_old_logp, b_adv, b_ret = old_logp_t[idx], adv_t[idx], ret_t[idx]
                b_base = base_t[idx]

                logp = self.model.log_prob(b_obs, b_act, base=b_base)
                ratio = torch.exp(logp - b_old_logp)
                surrogate = torch.minimum(
                    ratio * b_adv,
                    torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv,
                )
                policy_loss = -surrogate.mean()
                value = self.model.value(b_obs)
                value_loss = nn.functional.smooth_l1_loss(value, b_ret)
                entropy = self.model.entropy(b_obs).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (logp - b_old_logp)).mean().item()
                    stats["policy_loss"] += float(policy_loss.item())
                    stats["value_loss"] += float(value_loss.item())
                    stats["entropy"] += float(entropy.item())
                    stats["kl"] += approx_kl
                    n_updates += 1
                    if approx_kl > 1.5 * cfg.target_kl:
                        stop_early = True
                if stop_early:
                    break
            if stop_early:
                break

        for key in stats:
            stats[key] /= max(n_updates, 1)
        stats["lr"] = lr
        stats["ent_coef"] = ent_coef
        stats["n_minibatches"] = float(n_updates)
        stats["early_stop"] = float(stop_early)
        return stats

    # -- behaviour cloning ----------------------------------------------------
    def behaviour_clone(self, steps: int = 400, batch_size: int = 4096) -> dict[str, float]:
        """Warm-start the actor to imitate the singular-arc expert."""

        rng = np.random.default_rng(self.seed + 991)
        obs_rows: list[np.ndarray] = []
        action_rows: list[float] = []
        env = GoddardEnv(
            feature_mode=str(self.ablation["feature_mode"]),
            target_mode=self.ablation.get("target_mode"),
            record_trajectory=False,
        )
        episodes = 24
        for _ in range(episodes):
            obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = False
            while not done:
                h, v, m = (float(x) for x in env.state)
                u = singular_arc_u(
                    h, v, m, env.params["T_max"], env.params["cd"], env.params["Q_max"],
                    use_cap=True,
                )
                obs_rows.append(obs.copy())
                action_rows.append(u)
                obs, _reward, terminated, truncated, _info = env.step(u)
                done = bool(terminated or truncated)
        obs_array = np.asarray(obs_rows, dtype=np.float32)
        action_array = np.asarray(action_rows, dtype=np.float32)

        data_obs = torch.as_tensor(obs_array, device=self.device)
        data_act = torch.as_tensor(action_array, device=self.device)
        optimiser = torch.optim.Adam(self.model.actor_head.parameters(), lr=1e-3)
        losses: list[float] = []
        for _ in range(steps):
            idx = torch.randint(0, data_obs.shape[0], (batch_size,), device=self.device)
            alpha, beta = self.model.concentrations(data_obs[idx])
            mean = alpha / (alpha + beta)
            loss = nn.functional.mse_loss(mean, data_act[idx])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        return {
            "bc_samples": float(len(obs_rows)),
            "bc_final_mse": float(np.mean(losses[-20:])),
            "bc_initial_mse": float(np.mean(losses[:20])),
        }

    # -- training loop --------------------------------------------------------
    def train(self, updates: int | None = None) -> dict[str, Any]:
        cfg = self.config
        updates = cfg.n_updates if updates is None else int(updates)
        start = time.perf_counter()
        info: dict[str, Any] = {}
        if self.ablation.get("behavior_cloning"):
            info = self.behaviour_clone()
            print(
                f"[{self.name}] BC warm start: samples={info['bc_samples']:.0f} "
                f"mse {info['bc_initial_mse']:.4f} -> {info['bc_final_mse']:.4f}"
            )

        for update_index in range(updates):
            progress = update_index / max(updates - 1, 1)
            rollout = self.collect()
            stats = self.update(rollout, progress)
            record = {
                "update": update_index,
                "progress": round(progress, 4),
                "mean_apogee_train": float(np.mean(self.episode_apogees))
                if self.episode_apogees
                else float("nan"),
                "mean_apogee_last16": float(np.mean(list(self.episode_apogees)[-16:]))
                if self.episode_apogees
                else float("nan"),
                "mean_clamp_frac": float(np.mean(self.episode_clamp))
                if self.episode_clamp
                else float("nan"),
                "episodes": self.total_episodes,
                "steps": self.total_steps,
                "wall_s": round(time.perf_counter() - start, 2),
                **{key: round(value, 6) for key, value in stats.items()},
            }
            self.history.append(record)
            if update_index % 5 == 0 or update_index == updates - 1:
                print(
                    f"[{self.name}] u{update_index:4d} apogee={record['mean_apogee_train']:9.1f} "
                    f"ent={record['entropy']:+.3f} kl={record['kl']:.4f} "
                    f"pi={record['policy_loss']:+.4f} v={record['value_loss']:.3f} "
                    f"ep={record['episodes']} {record['wall_s']:.1f}s",
                    flush=True,
                )
        metadata = {
            "ablation": self.name,
            "seed": self.seed,
            "obs_dim": self.env.OBS_DIM,
            "beta_epsilon": float(self.ablation["beta_epsilon"]),
            "feature_mode": self.ablation["feature_mode"],
            "reward_mode": self.ablation["reward_mode"],
            "target_mode": self.ablation.get("target_mode"),
            "residual": bool(self.ablation.get("residual", False)),
            "use_actor_cap": bool(self.ablation.get("use_actor_cap", False)),
            "description": self.ablation.get("description", ""),
            "updates": updates,
            "wall_s": round(time.perf_counter() - start, 2),
            "bc": info or None,
        }
        return metadata

    def save(self, path: Path, metadata: Mapping[str, Any]) -> None:
        self.model.save(path, extra=dict(metadata))
        log_path = path.with_suffix(".jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            for record in self.history:
                handle.write(json.dumps(record) + "\n")
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(dict(metadata), indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation", default=None, help="R0..R6")
    parser.add_argument("--all", action="store_true", help="train every ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (with --all)")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--out-dir", default="results/checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test budget")
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = load_train_config(Path(args.config) if args.config else None)
    config = PPOConfig.from_dict(raw["ppo"])
    if args.updates is not None:
        config.n_updates = args.updates
    if args.n_envs is not None:
        config.n_envs = args.n_envs
        config.n_blocks = max(1, args.n_envs // config.n_twins)
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    if args.gae_lambda is not None:
        config.gae_lambda = args.gae_lambda
    if args.quick:
        config.n_updates = args.updates or 2
        config.rollout_steps = args.rollout_steps or 128
        config.mini_batch_size = 2048

    if args.all:
        names = list(ABLATIONS)
    elif args.ablation:
        names = [args.ablation]
    else:
        raise SystemExit("Pass --ablation R0..R6 or --all")

    if args.seeds:
        seeds = [int(value) for value in args.seeds.split(",")]
    elif args.all:
        seeds = [int(value) for value in raw["ppo"]["seeds"]]
    else:
        seeds = [args.seed]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for name in names:
        ablation = resolve_ablation(name)
        for seed in seeds:
            target = out_dir / f"{name}_seed{seed}.pt"
            if target.exists():
                print(f"[{name}] seed {seed}: checkpoint exists, skipping")
                continue
            trainer = PPOTrainer(name, ablation, config, seed=seed, device=args.device)
            metadata = trainer.train()
            trainer.save(target, metadata)
            summary.append(metadata)
            print(
                f"[{name}] seed {seed} done in {metadata['wall_s']:.1f}s -> "
                f"{target.relative_to(ROOT)}"
            )
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
