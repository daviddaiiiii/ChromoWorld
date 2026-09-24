#!/usr/bin/env python3
"""Run Cellpose-SAM 3D instance segmentation on TIFF volumes.

The input is one or more native 3D TIFF stacks with ZYX voxel order.  The
output is a label TIFF with one positive integer per instance.  Optional
resizing is performed only for inference; the saved mask is returned to the
original voxel grid with nearest-neighbour interpolation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="3D TIFF file or directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Cellpose-SAM pretrained model file or model directory.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU inference.")
    parser.add_argument(
        "--size-zyx",
        type=int,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="Optional inference size. Native size is used by default.",
    )
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--flow-threshold", type=float, default=0.1)
    parser.add_argument("--cellprob-threshold", type=float, default=1.0)
    parser.add_argument("--min-size", type=int, default=2000)
    parser.add_argument("--flow3d-smooth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-volume-voxels", type=int, default=2000)
    parser.add_argument("--min-extent-ratio", type=float, default=0.1)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def collect_tiffs(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in TIFF_SUFFIXES else []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in TIFF_SUFFIXES
    )


def read_volume(path: Path) -> tuple[np.ndarray, str]:
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        array = np.asarray(series.asarray())
        axes = str(series.axes or "")
    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D TIFF stack, got shape={array.shape} axes={axes!r} for {path}"
        )
    return array, axes or "ZYX"


def resize_image(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(volume.shape) == target_shape:
        return volume.astype(np.float32, copy=False)
    from scipy.ndimage import zoom

    factors = tuple(target / source for target, source in zip(target_shape, volume.shape))
    return zoom(volume.astype(np.float32), factors, order=1, mode="nearest")


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(mask.shape) == target_shape:
        return mask.astype(np.int32, copy=False)
    from scipy.ndimage import zoom

    factors = tuple(target / source for target, source in zip(target_shape, mask.shape))
    resized = zoom(mask.astype(np.float32), factors, order=0, mode="nearest")
    return np.rint(resized).astype(np.int32)


def filter_and_reindex(
    mask: np.ndarray,
    min_volume_voxels: int,
    min_extent_ratio: float,
) -> tuple[np.ndarray, dict[str, int]]:
    ids = [int(value) for value in np.unique(mask) if int(value) > 0]
    kept: list[tuple[int, int]] = []
    removed_small = 0
    removed_sparse = 0
    for instance_id in ids:
        instance = mask == instance_id
        volume = int(instance.sum())
        if volume < min_volume_voxels:
            removed_small += 1
            continue
        coords = np.argwhere(instance)
        if coords.size == 0:
            removed_small += 1
            continue
        extent = np.prod(coords.max(axis=0) - coords.min(axis=0) + 1)
        if extent <= 0 or volume / float(extent) < min_extent_ratio:
            removed_sparse += 1
            continue
        kept.append((volume, instance_id))

    kept.sort(key=lambda item: (-item[0], item[1]))
    output_dtype = np.uint16 if len(kept) <= np.iinfo(np.uint16).max else np.uint32
    filtered = np.zeros(mask.shape, dtype=output_dtype)
    for new_id, (_, old_id) in enumerate(kept, start=1):
        filtered[mask == old_id] = new_id
    return filtered, {
        "n_input_instances": len(ids),
        "n_output_instances": len(kept),
        "n_removed_small": removed_small,
        "n_removed_sparse": removed_sparse,
    }


def run_cellpose(volume: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import torch
    from cellpose import core as cellpose_core
    from cellpose import models as cellpose_models

    use_gpu = not args.cpu and bool(torch.cuda.is_available()) and cellpose_core.use_gpu()
    device = None
    if use_gpu:
        torch.cuda.set_device(int(args.gpu_id))
        device = torch.device(f"cuda:{int(args.gpu_id)}")
    model = cellpose_models.CellposeModel(
        gpu=use_gpu,
        device=device,
        pretrained_model=str(args.model),
    )
    image = np.expand_dims(volume.astype(np.float32, copy=False), axis=1)
    result = model.eval(
        image,
        z_axis=0,
        channel_axis=1,
        batch_size=int(args.batch_size),
        diameter=args.diameter,
        flow_threshold=float(args.flow_threshold),
        cellprob_threshold=float(args.cellprob_threshold),
        min_size=int(args.min_size),
        resample=True,
        do_3D=True,
        flow3D_smooth=int(args.flow3d_smooth),
    )
    return np.asarray(result[0], dtype=np.int32)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        mask,
        compression="zlib",
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )


def process_one(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output_dir / f"{path.stem}_mask.tif"
    if output_path.exists() and not args.force:
        return {
            "input_file": path.name,
            "output_file": output_path.name,
            "skipped": True,
        }

    t0 = time.perf_counter()
    raw, input_axes = read_volume(path)
    native_shape = tuple(int(value) for value in raw.shape)
    inference_shape = tuple(int(value) for value in (args.size_zyx or native_shape))
    inference_volume = resize_image(raw, inference_shape)
    mask = run_cellpose(inference_volume, args)
    mask = resize_mask(mask, native_shape)
    if args.no_filter:
        mask, filter_stats = filter_and_reindex(mask, 0, 0.0)
    else:
        mask, filter_stats = filter_and_reindex(
            mask,
            int(args.min_volume_voxels),
            float(args.min_extent_ratio),
        )
    save_mask(output_path, mask)
    return {
        "input_file": path.name,
        "output_file": output_path.name,
        "input_axes": input_axes,
        "native_shape_zyx": list(native_shape),
        "inference_shape_zyx": list(inference_shape),
        "mask_dtype": str(mask.dtype),
        "elapsed_sec": float(time.perf_counter() - t0),
        "skipped": False,
        **filter_stats,
    }


def main() -> None:
    args = parse_args()
    files = collect_tiffs(args.input)
    if args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        raise SystemExit(f"No TIFF files found under {args.input}")
    if not args.model.exists():
        raise SystemExit(f"Cellpose-SAM model does not exist: {args.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}", flush=True)
        try:
            record = process_one(path, args)
        except Exception as error:
            record = {
                "input_file": path.name,
                "skipped": False,
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"ERROR: {record['error']}", flush=True)
        records.append(record)

    report_path = args.report_json or args.output_dir / "segmentation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "chromoworld.3d_segmentation.v1",
        "model": str(args.model),
        "input_root": str(args.input),
        "output_root": str(args.output_dir),
        "parameters": {
            "size_zyx": list(args.size_zyx) if args.size_zyx else None,
            "diameter": args.diameter,
            "flow_threshold": args.flow_threshold,
            "cellprob_threshold": args.cellprob_threshold,
            "min_size": args.min_size,
            "flow3d_smooth": args.flow3d_smooth,
            "min_volume_voxels": args.min_volume_voxels,
            "min_extent_ratio": args.min_extent_ratio,
            "no_filter": args.no_filter,
        },
        "n_files": len(files),
        "files": records,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"n_files": len(files), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
