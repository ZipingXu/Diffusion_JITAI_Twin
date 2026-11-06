"""Minimal DDPM training and reward-tilted sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import CausalConditionalDenoiser


@dataclass(frozen=True)
class DiffusionConfig:
    steps: int = 24
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def validate(self) -> None:
        if self.steps < 2:
            raise ValueError("diffusion steps must be at least two")
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")


class GaussianDiffusion(nn.Module):
    """Variance-preserving diffusion with optional linear reward tilting."""

    def __init__(self, config: DiffusionConfig = DiffusionConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        betas = torch.linspace(config.beta_start, config.beta_end, config.steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_previous = torch.cat((torch.ones(1), alpha_bar[:-1]))
        posterior_variance = betas * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-12))

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bar[timesteps].unsqueeze(1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def loss(
        self,
        model: CausalConditionalDenoiser,
        clean: torch.Tensor,
        actions: torch.Tensor,
        baseline: torch.Tensor,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = clean.shape[0]
        timesteps = torch.randint(
            0, self.config.steps, (batch,), device=clean.device
        )
        noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timesteps, noise)
        predicted = model(noisy, actions, baseline, timesteps)
        squared_error = (predicted - noise).square()
        if observed is not None:
            weights = observed.to(dtype=squared_error.dtype)
            return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)
        return squared_error.mean()

    @torch.no_grad()
    def sample(
        self,
        model: CausalConditionalDenoiser,
        actions: torch.Tensor,
        baseline: torch.Tensor,
        *,
        seed: int = 0,
        guidance: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Draw trajectories, optionally from a linearly tilted distribution.

        For a guidance vector ``v``, the reverse process approximates draws
        from ``p(x) exp(guidance_scale * <v, x>)``. The additive shift is
        applied inside every reverse step rather than after generation.
        """

        if actions.ndim != 2 or baseline.ndim != 2:
            raise ValueError("actions and baseline must be two-dimensional")
        if actions.shape[0] != baseline.shape[0]:
            raise ValueError("actions and baseline batch sizes must match")
        device = actions.device
        generator = torch.Generator(device=device).manual_seed(seed)
        values = torch.randn(
            actions.shape, device=device, dtype=actions.dtype, generator=generator
        )
        if guidance is not None:
            guidance = guidance.to(device=device, dtype=values.dtype).reshape(1, -1)
            if guidance.shape[1] != actions.shape[1]:
                raise ValueError("guidance length must equal the trajectory horizon")
            guidance = guidance * float(guidance_scale)

        for step in range(self.config.steps - 1, -1, -1):
            batch_steps = torch.full(
                (actions.shape[0],), step, device=device, dtype=torch.long
            )
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bar[step]
            sigma = (1.0 - alpha_bar).sqrt()
            model_values = values
            if guidance is not None:
                shift = (1.0 - alpha_bar) / alpha_bar.sqrt() * guidance
                model_values = values + shift
            predicted_noise = model(model_values, actions, baseline, batch_steps)
            if guidance is not None:
                predicted_noise = predicted_noise - sigma / alpha_bar.sqrt() * guidance
            mean = (
                values
                - self.betas[step] / sigma.clamp_min(1e-8) * predicted_noise
            ) / alpha.sqrt()
            if step > 0:
                noise = torch.randn(
                    values.shape,
                    device=device,
                    dtype=values.dtype,
                    generator=generator,
                )
                values = mean + self.posterior_variance[step].sqrt() * noise
            else:
                values = mean
        return values
