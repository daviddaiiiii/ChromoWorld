"""Flow-matching architecture used by the March 2026 ChromoLift checkpoint."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = -math.log(10000.0) / (half - 1)
    frequencies = torch.exp(
        torch.arange(half, dtype=torch.float32, device=timesteps.device) * scale
    )
    embedding = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((embedding.sin(), embedding.cos()), dim=1)
    return F.pad(embedding, (0, 1)) if dim % 2 else embedding


class AdaLNZeroMLP(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int, mid_mult: int = 4):
        super().__init__()
        mid = max(hidden_dim * 2, cond_dim * mid_mult)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, mid),
            nn.SiLU(),
            nn.Linear(mid, 2 * hidden_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(cond)
        dim = out.shape[-1] // 2
        return 1.0 + out[:, :dim], out[:, dim:]


class FeatureFlowMatchingLifter(nn.Module):
    """Map seven 2D physical descriptors to seven 3D descriptors."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        condition_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.time_embed_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, self.time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim * 4, self.time_embed_dim * 4),
        )
        self.in_proj = nn.Linear(input_dim + self.time_embed_dim * 4, hidden_dim)
        self.blocks = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks))
        self.adaln_mlps = nn.ModuleList(
            AdaLNZeroMLP(condition_dim, hidden_dim) for _ in range(num_blocks)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.adaln_out = AdaLNZeroMLP(condition_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, target_dim)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, extra: dict
    ) -> torch.Tensor:
        time = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim).to(x))
        h = self.in_proj(torch.cat((x, time), dim=-1))
        condition = extra.get("condition")
        for block, adaln in zip(self.blocks, self.adaln_mlps):
            if condition is None:
                scale, shift = torch.ones_like(h), torch.zeros_like(h)
            else:
                scale, shift = adaln(condition)
            h = block(h)
            h = F.layer_norm(h, (h.shape[-1],))
            h = F.silu(h * scale + shift)
        h = self.out_norm(h)
        if condition is None:
            scale, shift = torch.ones_like(h), torch.zeros_like(h)
        else:
            scale, shift = self.adaln_out(condition)
        return self.out_proj(h * scale + shift)
