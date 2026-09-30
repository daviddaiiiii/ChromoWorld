#!/usr/bin/env python3
"""Lift 7D 2D morphology features into the 3D descriptor space."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import FeatureFlowMatchingLifter


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "Checkpoint/ChromoPhy/ChromoLift/lifter_final.pth"
)
FEATURES = (
    "norm_size", "eccentricity", "solidity", "circularity",
    "bulkiness", "tortuosity", "inertia_ratio",
)


def load_lifter(checkpoint: Path, device: torch.device):
    """Load the exact architecture and fitted scalers used by the UMAP suite."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    scaler_mip = ckpt["scaler_mip"]
    scaler_volume = ckpt["scaler_volume"]
    scaler_image = ckpt.get("scaler_image")
    n_features = int(scaler_mip.n_features_in_)
    if n_features != len(FEATURES) or int(scaler_volume.n_features_in_) != n_features:
        raise ValueError("Checkpoint is not a 7D ChromoLift feature model")
    condition_dim = int(scaler_image.n_features_in_) if scaler_image is not None else 1536
    model = FeatureFlowMatchingLifter(
        input_dim=n_features,
        target_dim=n_features,
        condition_dim=condition_dim,
        hidden_dim=256,
        num_blocks=4,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, scaler_mip, scaler_volume, scaler_image


@torch.no_grad()
def lift_features(
    model: FeatureFlowMatchingLifter,
    features: np.ndarray,
    scaler_mip,
    scaler_volume,
    device: torch.device,
    *,
    condition: np.ndarray | None = None,
    scaler_image=None,
    steps: int = 50,
    batch_size: int = 4096,
) -> np.ndarray:
    """Use the same Euler flow and scaling as the H100 live UMAP pipeline."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(FEATURES):
        raise ValueError("features must have shape [N, 7]")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinity")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if condition is not None:
        condition = np.asarray(condition, dtype=np.float32)
        if scaler_image is None:
            raise ValueError("Checkpoint does not contain an image-condition scaler")
        if condition.shape != (len(features), scaler_image.n_features_in_):
            raise ValueError("condition shape does not match checkpoint or feature rows")
        if not np.isfinite(condition).all():
            raise ValueError("condition contains NaN or infinity")
    normalized = scaler_mip.transform(features).astype(np.float32)
    output = np.empty_like(normalized)
    dt = 1.0 / steps
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        x = torch.as_tensor(normalized[start:stop], device=device)
        c = None
        if condition is not None:
            c_norm = scaler_image.transform(condition[start:stop]).astype(np.float32)
            c = torch.as_tensor(c_norm, device=device)
        for step in range(steps):
            t = torch.full((len(x),), step * dt, dtype=torch.float32, device=device)
            x = x + model(x, t, extra={"condition": c}) * dt
        output[start:stop] = x.cpu().numpy()
    return scaler_volume.inverse_transform(output).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with features [N,7] and optional condition [N,C]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    with np.load(args.input) as data:
        features = data["features"]
        condition = data["condition"] if "condition" in data.files else None
    device = torch.device(args.device)
    model, scaler_mip, scaler_volume, scaler_image = load_lifter(args.checkpoint, device)
    lifted = lift_features(
        model, features, scaler_mip, scaler_volume, device,
        condition=condition, scaler_image=scaler_image,
        steps=args.steps, batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=lifted, feature_names=np.asarray(FEATURES))
    print(f"Saved {len(lifted)} lifted instances to {args.output}")


if __name__ == "__main__":
    main()
