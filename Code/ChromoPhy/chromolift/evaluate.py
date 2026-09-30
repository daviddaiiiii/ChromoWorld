#!/usr/bin/env python3
"""Evaluate ChromoLift on paired seven-feature descriptors and optional conditions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from infer import DEFAULT_CHECKPOINT, FEATURES, lift_features, load_lifter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Held-out NPZ: features [N,7], targets [N,7], optional condition [N,C]")
    parser.add_argument("--output", required=True, type=Path, help="JSON metrics")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        features, targets = data["features"], np.asarray(data["targets"], dtype=np.float64)
        condition = data["condition"] if "condition" in data.files else None
    if targets.shape != features.shape or targets.ndim != 2 or targets.shape[1] != 7 or len(targets) == 0:
        raise ValueError("features and targets must have the same nonempty [N,7] shape")
    if not np.isfinite(targets).all():
        raise ValueError("targets contain nonfinite values")
    device = torch.device(args.device)
    model, mip_scaler, volume_scaler, image_scaler = load_lifter(args.checkpoint, device)
    predicted = lift_features(model, features, mip_scaler, volume_scaler, device,
                              condition=condition, scaler_image=image_scaler,
                              steps=args.steps, batch_size=args.batch_size)
    error = predicted - targets
    normalized_error = volume_scaler.transform(predicted) - volume_scaler.transform(targets)
    result = {"instances": len(targets), "checkpoint": str(args.checkpoint),
              "image_conditioned": condition is not None, "steps": args.steps,
              "per_feature": {name: {"mae": float(np.abs(error[:, i]).mean()),
                                     "rmse": float(np.sqrt(np.mean(error[:, i] ** 2)))}
                              for i, name in enumerate(FEATURES)},
              "standardized_rmse": float(np.sqrt(np.mean(normalized_error ** 2)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved paired-descriptor metrics to {args.output}")


if __name__ == "__main__":
    main()
