"""Pre-deployment target construction for linear reward tilting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TrajectoryScaler


@dataclass(frozen=True)
class Calibration:
    guidance: np.ndarray
    base_profile: np.ndarray
    target_profile: np.ndarray


def build_linear_guidance(
    base_samples: np.ndarray,
    target_profile: np.ndarray,
    scaler: TrajectoryScaler,
    *,
    max_absolute_guidance: float = 2.5,
) -> Calibration:
    """Build ``v = mu_target - mu_base`` in the model's operating space."""

    base_samples = np.asarray(base_samples, dtype=np.float32)
    target_profile = np.asarray(target_profile, dtype=np.float32).reshape(-1)
    if base_samples.ndim != 2:
        raise ValueError("base_samples must have shape (draws, time)")
    if base_samples.shape[1] != target_profile.shape[0]:
        raise ValueError("target profile length must match the sample horizon")
    if (base_samples < 0).any() or (target_profile < 0).any():
        raise ValueError("calibration operates on non-negative raw outcomes")
    base_profile = base_samples.mean(axis=0)
    guidance = scaler.transform(target_profile) - scaler.transform(base_profile)
    guidance = np.clip(
        guidance, -max_absolute_guidance, max_absolute_guidance
    ).astype(np.float32)
    return Calibration(
        guidance=guidance,
        base_profile=base_profile.astype(np.float32),
        target_profile=target_profile,
    )
