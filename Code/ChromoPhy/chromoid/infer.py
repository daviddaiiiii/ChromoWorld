#!/usr/bin/env python3
"""Assign 24-class chromosome IDs to 3D descriptors by lifted-gallery kNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


DEFAULT_GALLERY = (
    Path(__file__).resolve().parents[3]
    / "Artifacts/ChromoID/gallery_7d_cache.npz"
)
DEFAULT_LIFTER = (
    Path(__file__).resolve().parents[3]
    / "Checkpoint/ChromoPhy/ChromoLift/lifter_final.pth"
)
EPS = 1e-6


def predict_ids(
    gallery: np.ndarray,
    gallery_y: np.ndarray,
    query: np.ndarray,
    scaler_volume,
    *,
    k: int = 5,
) -> np.ndarray:
    """Euclidean kNN after the same 3D standardization used on H100."""
    gallery = np.asarray(gallery, dtype=np.float32)
    gallery_y = np.asarray(gallery_y, dtype=np.int32)
    query = np.asarray(query, dtype=np.float32)
    if gallery.ndim != 2 or query.ndim != 2 or gallery.shape[1] != 7 or query.shape[1] != 7:
        raise ValueError("gallery and query must have shape [N, 7]")
    if len(gallery) != len(gallery_y) or len(gallery) == 0 or k < 1:
        raise ValueError("gallery labels, gallery rows, or k are invalid")
    if not np.isfinite(gallery).all() or not np.isfinite(query).all():
        raise ValueError("descriptors contain NaN or infinity")
    if len(query) == 0:
        return np.empty(0, dtype=np.int32)
    gallery_scaled = scaler_volume.transform(gallery)
    query_scaled = scaler_volume.transform(query)
    neighbors = NearestNeighbors(n_neighbors=min(k, len(gallery)), metric="euclidean")
    neighbors.fit(gallery_scaled)
    distances, indices = neighbors.kneighbors(query_scaled)
    predictions = np.empty(len(query), dtype=np.int32)
    for row, (row_d, row_i) in enumerate(zip(distances, indices)):
        weights: dict[int, float] = {}
        for distance, index in zip(row_d, row_i):
            cid = int(gallery_y[index])
            weights[cid] = weights.get(cid, 0.0) + 1.0 / (float(distance) + EPS)
        best = max(weights.values())
        winners = [cid for cid, weight in weights.items() if weight == best]
        predictions[row] = winners[0] if len(winners) == 1 else int(gallery_y[row_i[0]])
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with 3D features [N,7]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--lifter-checkpoint", type=Path, default=DEFAULT_LIFTER)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    with np.load(args.gallery) as data:
        gallery = data["lifted_combined_x"]
        labels = data["lifted_combined_y"]
    with np.load(args.input) as data:
        query = data["features"]
    checkpoint = torch.load(args.lifter_checkpoint, map_location="cpu", weights_only=False)
    predictions = predict_ids(gallery, labels, query, checkpoint["scaler_volume"], k=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, class_id=predictions)
    print(f"Saved {len(predictions)} IDs to {args.output}")


if __name__ == "__main__":
    main()
