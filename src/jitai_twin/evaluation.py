"""Small dependency-free fidelity metrics for synthetic validation."""

from __future__ import annotations

import numpy as np


def wasserstein_1d(left: np.ndarray, right: np.ndarray, points: int = 256) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size == 0 or right.size == 0:
        raise ValueError("Wasserstein inputs must be non-empty")
    quantiles = np.linspace(0.0, 1.0, points)
    return float(np.mean(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles))))


def _participant_acf1(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    numerator = (centered[:, :-1] * centered[:, 1:]).sum(axis=1)
    denominator = (centered * centered).sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-8,
    )


def _active_run_lengths(values: np.ndarray) -> np.ndarray:
    threshold = np.quantile(values, 0.65)
    runs: list[int] = []
    for row in values > threshold:
        current = 0
        for active in row:
            if active:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
    return np.asarray(runs or [0], dtype=np.float64)


def fidelity_metrics(real: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    """Compare marginal, temporal, and between-participant structure."""

    real = np.asarray(real, dtype=np.float64)
    synthetic = np.asarray(synthetic, dtype=np.float64)
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError("real and synthetic must be 2-D with a shared horizon")
    real_log = np.log1p(np.maximum(real, 0.0))
    synthetic_log = np.log1p(np.maximum(synthetic, 0.0))
    real_user_mean = real_log.mean(axis=1)
    synthetic_user_mean = synthetic_log.mean(axis=1)
    return {
        "marginal_w1_log": wasserstein_1d(real_log, synthetic_log),
        "mean_profile_mae_log": float(
            np.abs(real_log.mean(axis=0) - synthetic_log.mean(axis=0)).mean()
        ),
        "acf1_w1": wasserstein_1d(
            _participant_acf1(real_log), _participant_acf1(synthetic_log)
        ),
        "active_run_w1": wasserstein_1d(
            _active_run_lengths(real_log), _active_run_lengths(synthetic_log)
        ),
        "between_user_sd_error": float(
            abs(real_user_mean.std() - synthetic_user_mean.std())
        ),
    }
