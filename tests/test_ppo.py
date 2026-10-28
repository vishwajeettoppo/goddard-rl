"""Beta policy head, GAE and PPO sanity checks."""

from __future__ import annotations

import numpy as np
import torch

from src.ppo import (
    BetaActorCritic,
    PPOConfig,
    compute_gae,
    normalise_advantages,
)


def _model(**kwargs) -> BetaActorCritic:
    torch.manual_seed(0)
    return BetaActorCritic(14, (32, 32), beta_epsilon=0.01, **kwargs)


def test_concentrations_respect_epsilon() -> None:
    model = _model()
    obs = torch.zeros(5, 14)
    alpha, beta = model.concentrations(obs)
    assert torch.all(alpha >= 0.01)
    assert torch.all(beta >= 0.01)


def test_beta_entropy_matches_monte_carlo() -> None:
    model = _model()
    obs = torch.zeros(64, 14)
    analytic = model.entropy(obs).mean().item()
    samples = model.distribution(obs).sample((4000,))
    empirical = -model.distribution(obs).log_prob(samples).mean().item()
    assert abs(analytic - empirical) < 0.05


def test_mean_and_mode_actions_are_bounded() -> None:
    model = _model()
    obs = torch.rand(16, 14) * 2 - 1
    for mode in (False, True):
        actions, logp, values = model.act(obs.numpy(), deterministic=True, mode=mode)
        assert np.all(actions > 0.0) and np.all(actions < 1.0)
        assert np.all(np.isfinite(logp))
        assert values.shape == (16,)


def test_residual_actions_stay_in_bounds_and_invert() -> None:
    model = _model(residual=True, residual_scale=0.5)
    obs = torch.zeros(8, 14)
    obs[:, -1] = torch.linspace(0.0, 1.0, 8)
    actions, _logp, _values = model.act(obs.numpy(), deterministic=True)
    assert np.all(actions >= 0.0) and np.all(actions <= 1.0)
    base = model.base_action(obs)
    latent = model.to_latent(torch.as_tensor(actions, dtype=torch.float32), base)
    recovered = model.to_throttle(latent, base)
    assert torch.allclose(recovered, torch.as_tensor(actions, dtype=torch.float32), atol=1e-6)


def test_gae_matches_discounted_return_when_lambda_one() -> None:
    rewards = np.array([[1.0], [1.0], [1.0]])
    values = np.zeros((3, 1))
    dones = np.zeros((3, 1))
    adv, returns = compute_gae(
        rewards, values, dones, np.zeros(1), gamma=1.0, gae_lambda=1.0
    )
    assert np.allclose(adv[:, 0], [3.0, 2.0, 1.0])
    assert np.allclose(returns, adv)


def test_gae_stops_at_termination() -> None:
    rewards = np.array([[1.0], [1.0], [1.0]])
    values = np.zeros((3, 1))
    dones = np.array([[0.0], [1.0], [0.0]])
    adv, _ = compute_gae(rewards, values, dones, np.zeros(1), gamma=1.0, gae_lambda=1.0)
    assert adv[1, 0] == 1.0
    assert adv[0, 0] == 2.0


def test_advantage_normalisation() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    norm = normalise_advantages(values)
    assert abs(norm.mean()) < 1e-6
    assert abs(norm.std() - 1.0) < 1e-6


def test_ppo_config_round_trip() -> None:
    config = PPOConfig.from_dict({"gamma": 1.0, "hidden_sizes": [64, 64], "unknown": 1})
    assert config.hidden_sizes == (64, 64)
    assert config.gamma == 1.0


def test_save_load_round_trip(tmp_path) -> None:
    model = _model(residual=True)
    path = tmp_path / "agent.pt"
    model.save(path, extra={"ablation": "R4"})
    loaded, meta = BetaActorCritic.load(path)
    assert meta["ablation"] == "R4"
    obs = torch.zeros(3, 14)
    assert torch.allclose(model.value(obs), loaded.value(obs))
