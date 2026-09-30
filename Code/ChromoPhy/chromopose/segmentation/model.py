"""3D U-Net with multi-head outputs for ChromoPose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Target layout: (C, Z, Y, X)
CH_FLOW_X = 0
CH_FLOW_Y = 1
CH_FLOW_Z = 2
CH_FOREGROUND = 3
CH_DISTANCE = 4
CH_SKELETON = 5
CH_BOUNDARY = 6
CH_EMB_0 = 7

# Base supervision channels (without optional embedding).
BASE_TARGET_CHANNELS = 7

# Optional per-skeleton-voxel embedding dimension (instance-consistent direction code).
EMBEDDING_DIM = 4
TARGET_CHANNELS = BASE_TARGET_CHANNELS + EMBEDDING_DIM  # 11

TARGET_CHANNEL_NAMES = (
    "flow_x",
    "flow_y",
    "flow_z",
    "foreground",
    "distance",
    "skeleton",
    "boundary",
    "emb_0",
    "emb_1",
    "emb_2",
    "emb_3",
)

AXIS_ZYX = ("Z", "Y", "X")

# Soft chromosome-count prior (used in postprocess scoring, never hard-clipped).
CHROMOSOME_COUNT_PRIOR = (50, 70)
CHROMOSOME_COUNT_SOFT_CENTER = 60.0

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Down3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        return x, self.pool(x)

class Up3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            dz = skip.shape[-3] - x.shape[-3]
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, (0, dx, 0, dy, 0, dz))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

@dataclass
class ChromoPoseHeadConfig:
    use_embedding: bool = True
    embedding_dim: int = EMBEDDING_DIM

class UNet3DChromoPose(nn.Module):
    """
    Shared encoder-decoder with dedicated prediction heads:
      foreground, distance, flow, skeleton, boundary, [embedding]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        levels: int = 4,
        head_config: Optional[ChromoPoseHeadConfig] = None,
    ):
        super().__init__()
        self.head_config = head_config or ChromoPoseHeadConfig()
        self.use_embedding = bool(self.head_config.use_embedding)
        self.embedding_dim = int(self.head_config.embedding_dim) if self.use_embedding else 0
        self.out_channels = (
            TARGET_CHANNELS if self.use_embedding else BASE_TARGET_CHANNELS
        )

        chs: List[int] = [int(base_channels * (2**i)) for i in range(levels)]
        self.inc = ConvBlock3D(in_channels, chs[0])
        self.downs = nn.ModuleList()
        for i in range(levels - 1):
            self.downs.append(Down3D(chs[i], chs[i + 1]))
        self.bot = ConvBlock3D(chs[-1], chs[-1] * 2)
        self.ups = nn.ModuleList()
        rev = list(reversed(chs))
        in_ch = chs[-1] * 2
        for i in range(levels - 1):
            self.ups.append(Up3D(in_ch, rev[i], rev[i + 1]))
            in_ch = rev[i + 1]

        feat_ch = chs[0]
        self.foreground_head = nn.Conv3d(feat_ch, 1, kernel_size=1)
        self.distance_head = nn.Conv3d(feat_ch, 1, kernel_size=1)
        self.flow_head = nn.Conv3d(feat_ch, 3, kernel_size=1)
        self.skeleton_head = nn.Conv3d(feat_ch, 1, kernel_size=1)
        self.boundary_head = nn.Conv3d(feat_ch, 1, kernel_size=1)
        self.embedding_head = (
            nn.Conv3d(feat_ch, self.embedding_dim, kernel_size=1) if self.use_embedding else None
        )

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        skips = [x0]
        x_down = x0
        for down in self.downs:
            s, x_down = down(x_down)
            skips.append(s)
        x_bot = self.bot(x_down)
        x_up = x_bot
        for up, skip in zip(self.ups, reversed(skips)):
            x_up = up(x_up, skip)
        return x_up

    def forward_heads(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "foreground": self.foreground_head(feat),
            "distance": self.distance_head(feat),
            "flow": self.flow_head(feat),
            "skeleton": self.skeleton_head(feat),
            "boundary": self.boundary_head(feat),
        }
        if self.embedding_head is not None:
            out["embedding"] = self.embedding_head(feat)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._decode(x)
        heads = self.forward_heads(feat)
        parts = [
            heads["flow"][:, 0:1],  # flow_x
            heads["flow"][:, 1:2],  # flow_y
            heads["flow"][:, 2:3],  # flow_z
            heads["foreground"],
            heads["distance"],
            heads["skeleton"],
            heads["boundary"],
        ]
        if self.embedding_head is not None:
            parts.append(heads["embedding"])
        return torch.cat(parts, dim=1)

def build_model(
    *,
    backbone: str = "unet3d",
    in_channels: int = 1,
    base_channels: int = 32,
    levels: int = 4,
    use_embedding: bool = True,
    embedding_dim: int = EMBEDDING_DIM,
) -> nn.Module:
    if str(backbone).lower() not in {"unet3d", "unet"}:
        raise ValueError(f"unknown backbone: {backbone}")
    return UNet3DChromoPose(
        in_channels=in_channels,
        base_channels=base_channels,
        levels=levels,
        head_config=ChromoPoseHeadConfig(use_embedding=use_embedding, embedding_dim=embedding_dim),
    )

def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    backbone: str,
    base_channels: int,
    levels: int,
    use_embedding: bool,
    device: torch.device,
) -> nn.Module:
    model = build_model(
        backbone=backbone,
        base_channels=int(base_channels),
        levels=int(levels),
        use_embedding=bool(use_embedding),
    ).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    if state and str(next(iter(state))).startswith("module."):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
