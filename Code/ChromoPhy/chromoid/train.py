#!/usr/bin/env python3
"""Build the labeled lifted-2D gallery used by ChromoID's 3D retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

LIFT_DIR = Path(__file__).resolve().parents[1] / "chromolift"
sys.path.insert(0, str(LIFT_DIR))
from infer import DEFAULT_CHECKPOINT, FEATURES, lift_features, load_lifter  # noqa: E402


def is_supported_modality(base_name: str) -> bool:
    if base_name.startswith("Banding_"):
        return True
    parts = base_name.split("_")
    return len(parts) >= 4 and parts[0].startswith("id") and parts[2] == "FISH"


def physical_vector(instance: dict) -> np.ndarray | None:
    values = []
    for name in FEATURES:
        value = instance.get(name)
        if value is None and name == "norm_size":
            value = next((instance[k] for k in ("norm_area", "norm_volume") if instance.get(k) is not None), None)
        if value is None:
            return None
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
    array = np.asarray(values, dtype=np.float32)
    return array if np.isfinite(array).all() else None


def build_gallery(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    model, scaler_mip, scaler_volume, scaler_image = load_lifter(args.checkpoint, device)
    gallery_x: list[np.ndarray] = []
    gallery_y: list[np.ndarray] = []
    pending_x: list[np.ndarray] = []
    pending_y: list[int] = []
    pending_c: list[np.ndarray] = []

    def flush() -> None:
        if not pending_x:
            return
        condition = None if args.no_image_condition else np.stack(pending_c)
        lifted = lift_features(
            model, np.stack(pending_x), scaler_mip, scaler_volume, device,
            condition=condition, scaler_image=scaler_image,
            steps=args.steps, batch_size=args.batch_size,
        )
        gallery_x.append(lifted)
        gallery_y.append(np.asarray(pending_y, dtype=np.int32))
        pending_x.clear()
        pending_y.clear()
        pending_c.clear()

    with args.input_jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if not is_supported_modality(str(record.get("base_name", ""))):
                continue
            conditions = None
            if not args.no_image_condition:
                if args.image_feature_dir is None:
                    raise ValueError("--image-feature-dir is required with image conditioning")
                path = args.image_feature_dir / Path(record["npy_file"]).name
                if not path.is_file():
                    raise FileNotFoundError(path)
                conditions = np.load(path)
            for index, instance in enumerate(record.get("instances", [])):
                cid = instance.get("pred_class_id", instance.get("class_id"))
                if not isinstance(cid, int) or not 0 <= cid <= 23:
                    continue
                physical = physical_vector(instance)
                if physical is None:
                    continue
                pending_x.append(physical)
                pending_y.append(cid)
                if conditions is not None:
                    if index >= len(conditions):
                        raise ValueError(f"Image features have fewer rows than instances: {path}")
                    pending_c.append(np.asarray(conditions[index]).reshape(-1).astype(np.float32))
                if len(pending_x) >= args.batch_size:
                    flush()
    flush()
    if not gallery_x:
        raise ValueError("No valid Banding/FISH instances found")
    return np.concatenate(gallery_x), np.concatenate(gallery_y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--image-feature-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--no-image-condition", action="store_true")
    args = parser.parse_args()
    gallery_x, gallery_y = build_gallery(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, lifted_combined_x=gallery_x, lifted_combined_y=gallery_y)
    print(f"Saved {len(gallery_y)} lifted gallery instances to {args.output}")


if __name__ == "__main__":
    main()
