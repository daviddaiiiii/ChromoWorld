#!/usr/bin/env python3
"""Score ChromoID retrieval on held-out 3D descriptors with reference labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from infer import DEFAULT_GALLERY, DEFAULT_LIFTER, predict_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Held-out NPZ with features [N,7] and class_id [N]")
    parser.add_argument("--output", required=True, type=Path, help="JSON metrics")
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--lifter-checkpoint", type=Path, default=DEFAULT_LIFTER)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        features, truth = data["features"], np.asarray(data["class_id"])
    if truth.shape != (len(features),) or len(truth) == 0 or not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("class_id must be a nonempty integer vector matching the feature rows")
    if np.any((truth < 0) | (truth >= 24)):
        raise ValueError("Reference class IDs must be in 0..23")
    with np.load(args.gallery, allow_pickle=False) as data:
        gallery, labels = data["lifted_combined_x"], data["lifted_combined_y"]
    checkpoint = torch.load(args.lifter_checkpoint, map_location="cpu", weights_only=False)
    predicted = predict_ids(gallery, labels, features, checkpoint["scaler_volume"], k=args.k)
    if np.any((predicted < 0) | (predicted >= 24)):
        raise ValueError("Gallery contains reference labels outside 0..23")
    confusion = np.zeros((24, 24), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    support = confusion.sum(axis=1)
    recalls = np.divide(confusion.diagonal(), support, out=np.zeros(24), where=support > 0)
    result = {"instances": len(truth), "k": args.k, "gallery": str(args.gallery),
              "accuracy": float(np.mean(predicted == truth)),
              "macro_recall_present_classes": float(recalls[support > 0].mean()),
              "class_ids": list(range(24)), "support": support.tolist(),
              "confusion_true_rows_predicted_columns": confusion.tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved retrieval metrics to {args.output}")


if __name__ == "__main__":
    main()
