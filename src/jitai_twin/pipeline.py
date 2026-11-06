"""End-to-end synthetic demonstration of the three-stage JITAI-Twin flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .calibration import build_linear_guidance
from .data import TrajectoryScaler
from .diffusion import DiffusionConfig, GaussianDiffusion
from .evaluation import fidelity_metrics
from .model import CausalConditionalDenoiser
from .synthetic import (
    SyntheticConfig,
    make_synthetic_cohorts,
    protocol_only_target_profile,
)
from .training import TrainConfig, fit_stage, generate


@dataclass(frozen=True)
class DemoConfig:
    synthetic: SyntheticConfig = SyntheticConfig()
    hidden_dim: int = 24
    depth: int = 2
    diffusion_steps: int = 16
    pretrain_updates: int = 30
    finetune_updates: int = 20
    batch_size: int = 16
    guidance_scale: float = 0.8
    seed: int = 19

    @classmethod
    def quick(cls, seed: int = 19) -> "DemoConfig":
        return cls(
            synthetic=SyntheticConfig(
                n_source=20,
                n_prior=10,
                n_target=10,
                horizon=24,
                baseline_dim=4,
                missing_rate=0.05,
                seed=seed,
            ),
            hidden_dim=12,
            depth=1,
            diffusion_steps=6,
            pretrain_updates=3,
            finetune_updates=2,
            batch_size=8,
            guidance_scale=0.6,
            seed=seed,
        )


def run_synthetic_demo(
    output_dir: str | Path,
    config: DemoConfig = DemoConfig(),
    *,
    device: str = "cpu",
) -> dict[str, object]:
    """Run all stages without reading any external data."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    cohorts = make_synthetic_cohorts(config.synthetic)
    source = cohorts["source"]
    prior = cohorts["prior"]
    target = cohorts["target"]
    scaler = TrajectoryScaler.fit(source)

    model = CausalConditionalDenoiser(
        baseline_dim=source.baseline_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
    )
    diffusion = GaussianDiffusion(DiffusionConfig(steps=config.diffusion_steps))

    pretrain_loss = fit_stage(
        model,
        diffusion,
        source,
        scaler,
        stage="pretrain",
        config=TrainConfig(
            updates=config.pretrain_updates,
            batch_size=config.batch_size,
            seed=config.seed,
        ),
        device=device,
    )
    finetune_loss = fit_stage(
        model,
        diffusion,
        prior,
        scaler,
        stage="finetune",
        config=TrainConfig(
            updates=config.finetune_updates,
            batch_size=config.batch_size,
            learning_rate=1e-3,
            seed=config.seed + 1,
        ),
        device=device,
    )

    base = generate(
        model,
        diffusion,
        target,
        scaler,
        seed=config.seed + 2,
        device=device,
    )
    expert_target = protocol_only_target_profile(target.horizon)
    calibration = build_linear_guidance(base, expert_target, scaler)
    calibrated = generate(
        model,
        diffusion,
        target,
        scaler,
        seed=config.seed + 2,
        guidance=calibration.guidance,
        guidance_scale=config.guidance_scale,
        device=device,
    )

    summary: dict[str, object] = {
        "release": "synthetic-only",
        "config": asdict(config),
        "training": {
            "pretrain_initial_loss": pretrain_loss[0],
            "pretrain_final_loss": pretrain_loss[-1],
            "finetune_initial_loss": finetune_loss[0],
            "finetune_final_loss": finetune_loss[-1],
        },
        "uncalibrated": fidelity_metrics(target.steps, base),
        "calibrated": fidelity_metrics(target.steps, calibrated),
        "calibration": {
            "guidance_l1_mean": float(np.abs(calibration.guidance).mean()),
            "target_profile_mean": float(expert_target.mean()),
            "base_profile_mean": float(base.mean()),
            "calibrated_profile_mean": float(calibrated.mean()),
        },
        "privacy": {
            "external_files_read": 0,
            "participant_data_written": 0,
            "target_outcomes_used_for_training_or_calibration": False,
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
