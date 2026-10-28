"""Observation builders, burnout targets and the potential-shaping audit."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src.features import (
    audit_quadratic_target,
    build_observation,
    burn_target_time,
    obs_dim,
    potential,
    t_target_monotone,
)


def test_observation_widths() -> None:
    assert obs_dim("base") == 12
    assert obs_dim("burnout") == 14
    assert obs_dim("residual") == 14
    assert obs_dim("energy") == 14


def test_base_observation_matches_carl_layout() -> None:
    obs = build_observation(
        h=0.0,
        v=0.0,
        m=1500.0,
        m_fuel_0=500.0,
        T_max=200000.0,
        cd=1.5,
        Q_max=5000.0,
        t=0.0,
        feature_mode="base",
    )
    assert obs.shape == (12,)
    assert obs[0] == pytest.approx(1.0)  # rho / rho_0
    assert obs[2] == pytest.approx(0.0)  # q / Qmax at rest
    assert obs[4] == pytest.approx(1.0)  # fuel fraction
    assert obs[7] == pytest.approx(1.0)  # energy ratio: all of it is chemical at rest
    assert obs[11] == pytest.approx(1.5)


def test_burnout_features_clip_and_hint_ranges() -> None:
    obs = build_observation(
        h=1000.0,
        v=50.0,
        m=1200.0,
        m_fuel_0=500.0,
        T_max=200000.0,
        cd=1.5,
        Q_max=5000.0,
        t=10.0,
        t_target=20.0,
        feature_mode="burnout",
    )
    assert obs.shape == (14,)
    assert -1.0 <= obs[12] <= 1.0
    assert 0.0 <= obs[13] <= 1.0


def test_quadratic_target_turns_over_within_training_range() -> None:
    audit = audit_quadratic_target()
    assert audit["vertex_m_fuel_0"] == pytest.approx(4687.5)
    assert audit["negative_unclipped_above_m_fuel_0"] == pytest.approx(9375.0)
    # The unclipped formula is already negative at x = 10 (10000 kg).
    assert audit["unclipped_at_x_10"] < 0.0


def test_monotone_target_is_monotone_and_positive() -> None:
    fuels = np.linspace(300.0, 5000.0, 40)
    values = [t_target_monotone(float(fuel)) for fuel in fuels]
    assert all(value > 0.0 for value in values)
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(values))
    # Calibration points from the paper's quadratic, reused for the monotone fit.
    for fuel, expected in ((500.0, 35.5), (2000.0, 118.0), (4000.0, 172.0)):
        assert t_target_monotone(fuel) == pytest.approx(expected, rel=0.02)


def test_rev1_target_is_linear_in_fuel() -> None:
    assert burn_target_time(500.0, mode="rev1") == pytest.approx(20.0)
    assert burn_target_time(4000.0, mode="rev1") == pytest.approx(160.0)


def test_potential_is_zero_in_terminal_states() -> None:
    assert potential(5000.0, 1100.0, 4000.0, 100.0, 160.0, terminal=True) == 0.0
    value = potential(1000.0, 3000.0, 4000.0, 10.0, 160.0)
    assert math.isfinite(value)
