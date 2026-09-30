#!/usr/bin/env python3
"""Evaluate a ChromoPix checkpoint, or previously generated polygon TXT files."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from release_paths import resolve_release_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default="Dataset/2D/test/image")
    parser.add_argument("--label-dir", default="Dataset/2D/test/label")
    parser.add_argument("--checkpoint", default="Checkpoint/ChromoPix/chromopix.pth")
    parser.add_argument("--output-dir", default="Code/ChromoPix/outputs/evaluation")
    parser.add_argument("--prediction-dir", help="Reuse predictions; skips checkpoint inference")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--polygon-mask-size", type=int, default=1024)
    parser.add_argument("--class-filter", choices=("present-in-gt", "all"), default="present-in-gt",
                        help="Historical figure evaluation filters predicted classes by GT presence")
    args = parser.parse_args()
    path = lambda value: resolve_release_path(value, __file__)
    output = path(args.output_dir)
    prediction = path(args.prediction_dir) if args.prediction_dir else output / "predictions"
    analysis = CODE_ROOT / "Analyze/fig2-3"
    if not args.prediction_dir:
        subprocess.run([
            sys.executable, str(analysis / "get-polygon_PNG2TXT.py"),
            "--image_path", str(path(args.image_dir)), "--out_path", str(prediction),
            "--checkpoint", str(path(args.checkpoint)), "--gpu_id", args.gpu_ids,
            "--batch_size", str(args.batch_size), "--img_h", str(args.img_size),
            "--img_w", str(args.img_size),
        ], check=True)
    subprocess.run([
        sys.executable, str(analysis / "get-metrics_TXT2CSV.py"),
        "--gt_path", str(path(args.label_dir)),
        "--baseline_path", json.dumps({"ChromoPix": str(prediction)}),
        "--class_configurable", json.dumps({"ChromoPix": args.class_filter == "present-in-gt"}),
        "--output_path", str(output / "reports"),
        "--polygon_mask_size", str(args.polygon_mask_size),
    ], check=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_config.json").write_text(json.dumps({
        "prediction_dir": str(prediction), "label_dir": str(path(args.label_dir)),
        "checkpoint": str(path(args.checkpoint)) if not args.prediction_dir else None,
        "class_filter": args.class_filter, "polygon_mask_size": args.polygon_mask_size,
        "inference_image_size": args.img_size if not args.prediction_dir else None,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
