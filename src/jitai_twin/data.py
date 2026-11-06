"""In-memory data contracts used by the public pipeline.

The package deliberately does not ship data readers tied to any restricted
study. Projects using real data should map their own access-controlled source
into :class:`Cohort` objects outside this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cohort:
    """A collection of equally long participant trajectories.

    Parameters
    ----------
    steps
        Non-negative outcome trajectories with shape ``(participants, time)``.
    actions
        Binary intervention indicators with the same shape as ``steps``.
    baseline
        Numeric baseline embeddings with shape ``(participants, features)``.
        A private application may construct these embeddings from harmonized
        survey text, but raw survey responses should remain outside the repo.
    observed
        Boolean observation mask with the same shape as ``steps``.
    """

    steps: np.ndarray
    actions: np.ndarray
    baseline: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", np.asarray(self.steps, dtype=np.float32))
        object.__setattr__(self, "actions", np.asarray(self.actions, dtype=np.float32))
        object.__setattr__(self, "baseline", np.asarray(self.baseline, dtype=np.float32))
        object.__setattr__(self, "observed", np.asarray(self.observed, dtype=bool))
        self.validate()

    @property
    def n_participants(self) -> int:
        return int(self.steps.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.steps.shape[1])

    @property
    def baseline_dim(self) -> int:
        return int(self.baseline.shape[1])

    def validate(self) -> None:
        if self.steps.ndim != 2:
            raise ValueError("steps must have shape (participants, time)")
        if self.actions.shape != self.steps.shape:
            raise ValueError("actions must match the shape of steps")
        if self.observed.shape != self.steps.shape:
            raise ValueError("observed must match the shape of steps")
        if self.baseline.ndim != 2 or self.baseline.shape[0] != self.steps.shape[0]:
            raise ValueError("baseline must have one row per participant")
        if not np.isfinite(self.steps).all() or (self.steps < 0).any():
            raise ValueError("steps must be finite and non-negative")
        if not np.isfinite(self.baseline).all():
            raise ValueError("baseline must be finite")
        if not np.isin(self.actions, [0.0, 1.0]).all():
            raise ValueError("actions must be binary")
        if not self.observed.any(axis=1).all():
            raise ValueError("each participant must have an observed outcome")

    def take(self, indices: np.ndarray) -> "Cohort":
        indices = np.asarray(indices)
        return Cohort(
            steps=self.steps[indices],
            actions=self.actions[indices],
            baseline=self.baseline[indices],
            observed=self.observed[indices],
        )


@dataclass(frozen=True)
class TrajectoryScaler:
    """Global ``log1p`` standardization fitted on observed source outcomes."""

    mean: float
    scale: float

    @classmethod
    def fit(cls, cohort: Cohort) -> "TrajectoryScaler":
        values = np.log1p(cohort.steps[cohort.observed])
        scale = float(values.std())
        return cls(mean=float(values.mean()), scale=max(scale, 1e-6))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if (values < 0).any():
            raise ValueError("raw outcomes must be non-negative")
        return ((np.log1p(values) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        raw = np.expm1(values * self.scale + self.mean)
        return np.maximum(raw, 0.0).astype(np.float32)
