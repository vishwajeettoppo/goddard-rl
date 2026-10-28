"""PPO with a Beta policy head for the constrained Goddard rocket.

The actor predicts the two concentration parameters of a Beta distribution over
the throttle; the critic predicts the state value.  The implementation follows
CARL's Appendix C as closely as the paper states it: separate 256x256 tanh MLPs,
orthogonal initialisation, Gamma = 1, GAE(lambda = 0.95), clipped PPO with target
KL early stopping, Huber value loss, linear learning-rate and entropy anneals.

Two policy-head modes are supported because the paper's central finding is that
the boundary mode (``beta_epsilon < 1``) is what escapes the unimodal collapse:

* ``beta_epsilon = 1`` -- unimodal Beta, the R0 baseline.
* ``beta_epsilon = 0.01`` -- boundary-seeking Beta, R1 and later.

Evaluation may use the Beta mean, the Beta mode, or a sample; :meth:`policy_action`
implements all three so the train/eval mismatch can be measured (R6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta

ACTION_EPS = 1e-6


def orthogonal_init(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


class BetaActorCritic(nn.Module):
    """Separate tanh actor/critic trunks with a Beta action head."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        *,
        beta_epsilon: float = 0.01,
        residual: bool = False,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if beta_epsilon <= 0.0:
            raise ValueError("beta_epsilon must be positive")
        self.obs_dim = int(obs_dim)
        self.beta_epsilon = float(beta_epsilon)
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)

        actor_layers: list[nn.Module] = []
        width = self.obs_dim
        for hidden in hidden_sizes:
            actor_layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        self.actor_body = nn.Sequential(*actor_layers)
        self.actor_head = nn.Linear(width, 2)

        critic_layers: list[nn.Module] = []
        width = self.obs_dim
        for hidden in hidden_sizes:
            critic_layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        self.critic_body = nn.Sequential(*critic_layers)
        self.critic_head = nn.Linear(width, 1)

        for module in self.actor_body:
            if isinstance(module, nn.Linear):
                orthogonal_init(module, math.sqrt(2.0))
        for module in self.critic_body:
            if isinstance(module, nn.Linear):
                orthogonal_init(module, math.sqrt(2.0))
        orthogonal_init(self.actor_head, 0.01)
        orthogonal_init(self.critic_head, 1.0)

    # -- distribution helpers -------------------------------------------------
    def concentrations(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.actor_head(self.actor_body(obs))
        alpha = nn.functional.softplus(raw[:, 0]) + self.beta_epsilon
        beta = nn.functional.softplus(raw[:, 1]) + self.beta_epsilon
        return alpha, beta

    def distribution(self, obs: torch.Tensor) -> Beta:
        alpha, beta = self.concentrations(obs)
        return Beta(alpha, beta)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_body(obs)).squeeze(-1)

    # -- action parameterisation ---------------------------------------------
    def base_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Singular-arc hint stored in the last observation entry (R4)."""

        if not self.residual:
            return torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)
        return obs[:, -1].clamp(0.0, 1.0)

    def to_throttle(self, z: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if not self.residual:
            return z.clamp(ACTION_EPS, 1.0 - ACTION_EPS)
        delta = (2.0 * z - 1.0) * self.residual_scale
        return (base + delta).clamp(ACTION_EPS, 1.0 - ACTION_EPS)

    def to_latent(self, throttle: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`to_throttle` for PPO importance ratios."""

        if not self.residual:
            return throttle.clamp(ACTION_EPS, 1.0 - ACTION_EPS)
        z = ((throttle - base) / self.residual_scale + 1.0) * 0.5
        return z.clamp(ACTION_EPS, 1.0 - ACTION_EPS)

    def log_prob(
        self, obs: torch.Tensor, throttle: torch.Tensor, *, base: torch.Tensor | None = None
    ) -> torch.Tensor:
        if base is None:
            base = self.base_action(obs)
        z = self.to_latent(throttle, base)
        return self.distribution(obs).log_prob(z)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Analytic Beta differential entropy (torch has no Beta.entropy)."""

        alpha, beta = self.concentrations(obs)
        a, b = alpha, beta
        log_b = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        return (
            log_b
            - (a - 1.0) * torch.digamma(a)
            - (b - 1.0) * torch.digamma(b)
            + (a + b - 2.0) * torch.digamma(a + b)
        )

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        *,
        deterministic: bool = False,
        mode: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(throttle, log_prob, value)`` for a batch of observations."""

        tensor = torch.as_tensor(obs, dtype=torch.float32)
        alpha, beta = self.concentrations(tensor)
        mean = alpha / (alpha + beta)
        if mode:
            denom = alpha + beta - 2.0
            mode_value = torch.where(denom > 1e-6, (alpha - 1.0) / denom.clamp_min(1e-6), mean)
            latent = torch.where(
                ((alpha <= 1.0) | (beta <= 1.0)), mode_value, mean
            ).clamp(ACTION_EPS, 1.0 - ACTION_EPS)
        elif deterministic:
            latent = mean.clamp(ACTION_EPS, 1.0 - ACTION_EPS)
        else:
            latent = self.distribution(tensor).sample().clamp(ACTION_EPS, 1.0 - ACTION_EPS)
        base = self.base_action(tensor)
        throttle = self.to_throttle(latent, base)
        log_prob = self.distribution(tensor).log_prob(latent)
        value = self.value(tensor)
        return (
            throttle.numpy(),
            log_prob.numpy(),
            value.numpy(),
        )

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path, extra: Mapping[str, Any] | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "obs_dim": self.obs_dim,
            "beta_epsilon": self.beta_epsilon,
            "residual": self.residual,
            "residual_scale": self.residual_scale,
            "hidden_sizes": [
                module.out_features
                for module in self.actor_body
                if isinstance(module, nn.Linear)
            ],
        }
        if extra:
            payload["meta"] = dict(extra)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["BetaActorCritic", dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            int(payload["obs_dim"]),
            tuple(payload.get("hidden_sizes", (256, 256))),
            beta_epsilon=float(payload["beta_epsilon"]),
            residual=bool(payload.get("residual", False)),
            residual_scale=float(payload.get("residual_scale", 0.5)),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, dict(payload.get("meta", {}))


@dataclass
class PPOConfig:
    """Optimiser and rollout hyperparameters (configs/train.yaml ``ppo`` block)."""

    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr_start: float = 4.0e-4
    lr_end: float = 2.0e-4
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 2
    mini_batch_size: int = 8192
    target_kl: float = 0.05
    rollout_steps: int = 1024
    n_envs: int = 256
    n_blocks: int = 32
    n_twins: int = 8
    n_updates: int = 120
    hidden_sizes: tuple[int, ...] = (256, 256)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PPOConfig":
        known = {f for f in cls.__dataclass_fields__}
        values = {key: value for key, value in payload.items() if key in known}
        if "hidden_sizes" in values:
            values["hidden_sizes"] = tuple(int(v) for v in values["hidden_sizes"])
        return cls(**values)


def _linear_anneal(start: float, end: float, progress: float) -> float:
    return float(start + (end - start) * min(max(progress, 0.0), 1.0))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over a ``(T, N)`` rollout."""

    steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_gae = np.zeros(n_envs, dtype=np.float64)
    for t in reversed(range(steps)):
        next_value = last_values if t == steps - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def normalise_advantages(advantages: np.ndarray) -> np.ndarray:
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + 1e-8)


@dataclass
class Rollout:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    base_actions: np.ndarray
    last_values: np.ndarray | None = None
    episode_apogees: list[float] = field(default_factory=list)
    episode_clamp_fracs: list[float] = field(default_factory=list)


__all__ = [
    "ACTION_EPS",
    "BetaActorCritic",
    "PPOConfig",
    "Rollout",
    "compute_gae",
    "normalise_advantages",
    "orthogonal_init",
]
