"""Synthetic cohorts for tests, tutorials, and continuous integration.

All arrays are generated in memory. They are intentionally small, clearly
synthetic, and contain no rows or summaries derived from study participants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Cohort


@dataclass(frozen=True)
class SyntheticConfig:
    n_source: int = 64
    n_prior: int = 24
    n_target: int = 24
    horizon: int = 48
    baseline_dim: int = 6
    missing_rate: float = 0.05
    seed: int = 7

    def validate(self) -> None:
        if min(self.n_source, self.n_prior, self.n_target) < 2:
            raise ValueError("each synthetic cohort needs at least two participants")
        if self.horizon < 8:
            raise ValueError("horizon must be at least eight")
        if self.baseline_dim < 2:
            raise ValueError("baseline_dim must be at least two")
        if not 0 <= self.missing_rate < 0.5:
            raise ValueError("missing_rate must be in [0, 0.5)")


def _make_cohort(
    rng: np.random.Generator,
    n: int,
    config: SyntheticConfig,
    *,
    population_shift: float,
    phase_shift: float,
    with_actions: bool,
) -> Cohort:
    horizon = config.horizon
    baseline = rng.normal(size=(n, config.baseline_dim)).astype(np.float32)
    hour = np.arange(horizon, dtype=np.float32) % 24
    day = np.arange(horizon, dtype=np.float32) // 24

    # Two daily activity peaks, with a population-specific timing shift.
    morning = np.exp(-0.5 * ((hour - (8.0 + phase_shift)) / 2.2) ** 2)
    evening = 0.85 * np.exp(-0.5 * ((hour - (17.0 + phase_shift)) / 3.0) ** 2)
    circadian = morning + evening
    weekend = np.where((day % 7) >= 5, -0.18, 0.0)

    availability = rng.random((n, horizon)) < 0.72
    actions = (
        availability & (rng.random((n, horizon)) < 0.35)
        if with_actions
        else np.zeros((n, horizon), dtype=bool)
    )

    participant_level = 5.25 + 0.32 * baseline[:, 0] + 0.18 * baseline[:, 1]
    latent = rng.normal(scale=0.30, size=(n, horizon)).astype(np.float32)
    for t in range(1, horizon):
        latent[:, t] += 0.62 * latent[:, t - 1]

    action_effect = 0.12 * actions.astype(np.float32)
    action_effect[:, 1:] += 0.06 * actions[:, :-1]
    log_rate = (
        participant_level[:, None]
        + population_shift
        + 0.95 * circadian[None, :]
        + weekend[None, :]
        + latent
        + action_effect * (1.0 + 0.20 * baseline[:, [2]])
    )
    raw = np.maximum(np.exp(log_rate) - 1.0, 0.0)

    # Create realistic inactivity without using any empirical distribution.
    inactive_probability = np.clip(0.55 - 0.42 * circadian, 0.08, 0.70)
    inactive = rng.random((n, horizon)) < inactive_probability[None, :]
    raw[inactive] *= rng.uniform(0.0, 0.08, size=int(inactive.sum()))
    steps = rng.poisson(np.clip(raw, 0.0, 5000.0)).astype(np.float32)

    observed = rng.random((n, horizon)) >= config.missing_rate
    observed[:, 0] = True
    return Cohort(
        steps=steps,
        actions=actions.astype(np.float32),
        baseline=baseline,
        observed=observed,
    )


def make_synthetic_cohorts(
    config: SyntheticConfig = SyntheticConfig(),
) -> dict[str, Cohort]:
    """Return observational, prior-intervention, and target cohorts.

    The target cohort is returned only to validate the synthetic demo. The
    training pipeline never uses it for fitting or calibration.
    """

    config.validate()
    root = np.random.SeedSequence(config.seed)
    source_seed, prior_seed, target_seed = root.spawn(3)
    return {
        "source": _make_cohort(
            np.random.default_rng(source_seed),
            config.n_source,
            config,
            population_shift=0.12,
            phase_shift=-0.4,
            with_actions=False,
        ),
        "prior": _make_cohort(
            np.random.default_rng(prior_seed),
            config.n_prior,
            config,
            population_shift=0.0,
            phase_shift=0.0,
            with_actions=True,
        ),
        "target": _make_cohort(
            np.random.default_rng(target_seed),
            config.n_target,
            config,
            population_shift=-0.24,
            phase_shift=1.2,
            with_actions=True,
        ),
    }


def protocol_only_target_profile(horizon: int) -> np.ndarray:
    """Construct a toy pre-deployment target from synthetic protocol facts.

    This function represents an expert-specified calibration target. It does
    not inspect the synthetic target cohort returned by
    :func:`make_synthetic_cohorts`.
    """

    hour = np.arange(horizon, dtype=np.float32) % 24
    morning = np.exp(-0.5 * ((hour - 9.0) / 2.4) ** 2)
    evening = 0.90 * np.exp(-0.5 * ((hour - 18.0) / 3.1) ** 2)
    return (35.0 + 330.0 * (morning + evening)).astype(np.float32)
