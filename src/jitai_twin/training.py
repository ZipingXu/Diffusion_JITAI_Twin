"""Training helpers for the observational and prior-deployment stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import Cohort, TrajectoryScaler
from .diffusion import GaussianDiffusion
from .model import CausalConditionalDenoiser


@dataclass(frozen=True)
class TrainConfig:
    updates: int = 100
    batch_size: int = 16
    learning_rate: float = 2e-3
    seed: int = 11

    def validate(self) -> None:
        if self.updates < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("training configuration values must be positive")


def fit_stage(
    model: CausalConditionalDenoiser,
    diffusion: GaussianDiffusion,
    cohort: Cohort,
    scaler: TrajectoryScaler,
    *,
    stage: str,
    config: TrainConfig = TrainConfig(),
    device: str | torch.device = "cpu",
) -> list[float]:
    """Fit one stage and return the optimization loss history."""

    config.validate()
    torch.manual_seed(config.seed)
    model.to(device)
    diffusion.to(device)
    model.set_training_stage(stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("the selected stage has no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)

    clean = torch.from_numpy(scaler.transform(cohort.steps)).to(device)
    actions = torch.from_numpy(cohort.actions).to(device)
    baseline = torch.from_numpy(cohort.baseline).to(device)
    observed = torch.from_numpy(cohort.observed).to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    model.train()
    history: list[float] = []
    for _ in range(config.updates):
        indices = torch.randint(
            0,
            cohort.n_participants,
            (min(config.batch_size, cohort.n_participants),),
            generator=generator,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.loss(
            model,
            clean[indices],
            actions[indices],
            baseline[indices],
            observed[indices],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history


@torch.no_grad()
def generate(
    model: CausalConditionalDenoiser,
    diffusion: GaussianDiffusion,
    cohort: Cohort,
    scaler: TrajectoryScaler,
    *,
    seed: int = 0,
    guidance: np.ndarray | None = None,
    guidance_scale: float = 1.0,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Generate one synthetic trajectory per baseline/action row."""

    model.to(device).eval()
    diffusion.to(device)
    actions = torch.from_numpy(cohort.actions).to(device)
    baseline = torch.from_numpy(cohort.baseline).to(device)
    guidance_tensor = None
    if guidance is not None:
        guidance_tensor = torch.as_tensor(guidance, dtype=torch.float32, device=device)
    transformed = diffusion.sample(
        model,
        actions,
        baseline,
        seed=seed,
        guidance=guidance_tensor,
        guidance_scale=guidance_scale,
    )
    return scaler.inverse(transformed.cpu().numpy())
