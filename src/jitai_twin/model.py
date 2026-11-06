"""A compact temporally consistent denoiser used by the public example."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    """One-dimensional convolution that never reads future positions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(in_channels, out_channels, kernel_size, padding=0)
        self.left_padding = kernel_size - 1

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(values, (self.left_padding, 0)))


class ChannelLayerNorm(nn.Module):
    """Layer normalization across channels independently at every time."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.conv1 = CausalConv1d(channels, channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.conv2 = CausalConv1d(channels, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return values + hidden


def diffusion_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding.shape[1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[1]))
    return embedding


class CausalConditionalDenoiser(nn.Module):
    """Small action- and baseline-conditioned causal denoising network.

    The production study used a state-space TimeWeaver backbone. This compact
    convolutional network exposes the same essential contracts while keeping
    the synthetic example fast enough for CPU-only continuous integration.
    """

    def __init__(
        self,
        baseline_dim: int,
        hidden_dim: int = 32,
        depth: int = 3,
        adapter_rank: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.action_projection = nn.Conv1d(1, hidden_dim, kernel_size=1, bias=False)
        self.position_projection = nn.Conv1d(2, hidden_dim, kernel_size=1, bias=False)
        self.baseline_projection = nn.Linear(baseline_dim, hidden_dim)
        self.adapter_down = nn.Linear(baseline_dim, adapter_rank, bias=False)
        self.adapter_up = nn.Linear(adapter_rank, hidden_dim, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        self.diffusion_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(CausalResidualBlock(hidden_dim) for _ in range(depth))
        self.output_norm = ChannelLayerNorm(hidden_dim)
        self.output_projection = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        noisy: torch.Tensor,
        actions: torch.Tensor,
        baseline: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.ndim != 2 or actions.shape != noisy.shape:
            raise ValueError("noisy and actions must have shape (batch, time)")
        hidden = self.input_projection(noisy.unsqueeze(1))
        hidden = hidden + self.action_projection(actions.unsqueeze(1))
        positions = torch.arange(
            noisy.shape[1], device=noisy.device, dtype=noisy.dtype
        )
        daily_angle = 2.0 * math.pi * positions / 24.0
        position_features = torch.stack((daily_angle.sin(), daily_angle.cos()))
        hidden = hidden + self.position_projection(
            position_features.unsqueeze(0).expand(noisy.shape[0], -1, -1)
        )
        baseline_features = self.baseline_projection(baseline)
        baseline_features = baseline_features + self.adapter_up(self.adapter_down(baseline))
        hidden = hidden + baseline_features.unsqueeze(-1)
        time_features = self.diffusion_projection(
            diffusion_embedding(timesteps, self.hidden_dim)
        )
        hidden = hidden + time_features.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(F.silu(self.output_norm(hidden))).squeeze(1)

    def set_training_stage(self, stage: str) -> None:
        """Select trainable parameters for observational pretraining or PEFT."""

        if stage not in {"pretrain", "finetune"}:
            raise ValueError("stage must be 'pretrain' or 'finetune'")
        for parameter in self.parameters():
            parameter.requires_grad = stage == "pretrain"
        if stage == "pretrain":
            # Observational data contain no recorded target intervention.
            for parameter in self.action_projection.parameters():
                parameter.requires_grad = False
            # The low-rank adapter is reserved for small-deployment adaptation.
            for module in (self.adapter_down, self.adapter_up):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        else:
            # Fine-tune intervention injection, low-rank baseline adaptation,
            # and the input/output projections while preserving the backbone.
            for module in (
                self.input_projection,
                self.action_projection,
                self.adapter_down,
                self.adapter_up,
                self.output_projection,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True
