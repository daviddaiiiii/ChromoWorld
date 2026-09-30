#!/usr/bin/env python3
"""Run raw 4D DAPI through ChromoPose preprocessing, registration, and tracking.

Each *-DAPI.tif input produces one <basename>/ folder containing six stage
directories. Image stages contain calibrated TZYX intensities; mask stages
contain TZYX instance labels on the registered image grid.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile


HERE = Path(__file__).resolve().parent
CHROMOPOSE = HERE.parent.parent / "ChromoPhy" / "chromopose"
DEFAULT_CHECKPOINT = HERE.parents[2] / "Checkpoint" / "ChromoPhy" / "ChromoPose" / "last.pth"
STAGES = ("raw", "preprocess", "affine", "segment", "post", "track")
PIPELINE_VERSION = "chromoworld.4d_mask_pipeline.v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw *-DAPI.tif file or flat directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu-ids", default="0", help="Passed to ChromoPose infer.py")
    parser.add_argument("--profile", choices=("auto", "legacy-v2", "legacy-paired"), default="auto")
    parser.add_argument("--anchors", type=Path, help="Optional tracker anchors JSON, keyed by sample basename")
    parser.add_argument("--pattern", default="*-DAPI.tif", help="Glob for directory input")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--stop-after", choices=STAGES, default="track", help="Pilot one or more early stages")
    return parser.parse_args()


def file_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def save_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sample_name(path: Path) -> str:
    if path.suffix.lower() not in (".tif", ".tiff") or not path.stem.endswith("-DAPI"):
        raise ValueError(f"Expected a *-DAPI.tif(f) input: {path}")
    return path.stem.removesuffix("-DAPI")


def source_files(path: Path, pattern: str, max_files: int) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob(pattern))
    files = [file for file in files if file.is_file()]
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No raw DAPI TIFFs found at {path}")
    names = [sample_name(file) for file in files]
    if len(names) != len(set(names)):
        raise ValueError("Input sample basenames are not unique")
    return files


def tiff_info(path: Path, *, label: bool = False) -> dict[str, Any]:
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        axes = str(series.axes)
        shape = tuple(int(value) for value in series.shape)
        dtype = np.dtype(series.dtype)
    if axes == "ZYX" and len(shape) == 3:
        shape = (1, *shape)
    elif axes != "TZYX" or len(shape) != 4:
        raise ValueError(f"Expected ZYX/TZYX TIFF, got {axes} {shape}: {path}")
    if label and dtype.kind != "u":
        raise ValueError(f"Expected unsigned integer labels, got {dtype}: {path}")
    if not label and dtype != np.dtype("uint16"):
        raise ValueError(f"Expected uint16 DAPI, got {dtype}: {path}")
    return {"shape_tzyx": list(shape), "axes": axes, "dtype": dtype.name, "bytes": path.stat().st_size}


def run_command(script: Path, *arguments: str | Path) -> None:
    command = [sys.executable, str(script), *(str(item) for item in arguments)]
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def relocate_report_paths(temporary: Path, destination: Path) -> None:
    """Keep component JSON reports valid after committing a staged directory."""
    old_prefix, new_prefix = str(temporary), str(destination)

    def replace(value):
        if isinstance(value, str):
            return value.replace(old_prefix, new_prefix)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {replace(key): replace(item) for key, item in value.items()}
        return value

    for report in temporary.rglob("*.json"):
        data = json.loads(report.read_text(encoding="utf-8"))
        updated = replace(data)
        if updated != data:
            save_json(report, updated)


def finished_stage(
    sample_dir: Path,
    stage: str,
    filename: str,
    producer: Callable[[Path], None],
    *,
    label: bool,
    expected_shape: tuple[int, ...] | None,
) -> tuple[Path, dict[str, Any]]:
    destination = sample_dir / stage
    output = destination / filename
    if destination.exists():
        marker = destination / "stage.json"
        if not marker.is_file() or not output.is_file():
            raise RuntimeError(f"Incomplete stage; inspect before retrying: {destination}")
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        info = tiff_info(output, label=label)
        if recorded.get("output") != filename or recorded.get("tiff") != info:
            raise RuntimeError(f"Stage output changed after completion: {destination}")
        if expected_shape and tuple(info["shape_tzyx"]) != expected_shape:
            raise RuntimeError(f"Stage shape mismatch: {output}")
        print(f"[skip] {stage}: {output}", flush=True)
        return output, info

    temporary = Path(tempfile.mkdtemp(prefix=f".{stage}-", dir=sample_dir))
    try:
        producer(temporary)
        temporary_output = temporary / filename
        info = tiff_info(temporary_output, label=label)
        if expected_shape and tuple(info["shape_tzyx"]) != expected_shape:
            raise ValueError(f"{stage} shape {info['shape_tzyx']} != expected {expected_shape}")
        relocate_report_paths(temporary, destination)
        save_json(temporary / "stage.json", {"stage": stage, "output": filename, "tiff": info})
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"[done] {stage}: {output} {info['shape_tzyx']}", flush=True)
    return output, info


def resolve_profile(source: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    prefix = sample_name(source) + "-Tag-Chr"
    paired = any(
        item.name.startswith(prefix) and item.suffix.lower() in (".tif", ".tiff")
        for item in source.parent.iterdir()
    )
    return "legacy-paired" if paired else "legacy-v2"


def load_anchors(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    anchors = data.get("anchors", data)
    if not isinstance(anchors, dict):
        raise ValueError("Anchors JSON must contain an object keyed by sample basename")
    return anchors


def anchor_index(name: str, count: int, anchors: dict[str, Any]) -> int:
    entry = anchors.get(name, {})
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid anchor entry for {name}")
    value = entry.get("representative_t", entry.get("window_start", count // 2))
    index = int(value)
    if not 0 <= index < count:
        raise ValueError(f"Anchor t={index} outside 0..{count - 1} for {name}")
    return index


def process_sample(source: Path, args: argparse.Namespace, anchors: dict[str, Any]) -> None:
    name = sample_name(source)
    sample_dir = args.output_dir / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    profile = resolve_profile(source, args.profile)
    checkpoint = args.checkpoint.resolve()
    config = {
        "version": PIPELINE_VERSION, "sample": name, "source": file_stat(source),
        "profile": profile, "checkpoint": file_stat(checkpoint) if checkpoint.is_file() else str(checkpoint),
        "gpu_ids": args.gpu_ids, "anchors": file_stat(args.anchors) if args.anchors else None,
    }
    manifest_path = sample_dir / "pipeline.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(f"Pipeline inputs/options changed for {name}: {manifest_path}")
    else:
        if any(sample_dir.iterdir()):
            raise ValueError(f"Nonempty output without pipeline manifest: {sample_dir}")
        save_json(manifest_path, config)

    original = tiff_info(source)
    raw_name = f"{name}-DAPI.tif"
    raw_path, raw_info = finished_stage(
        sample_dir, "raw", raw_name,
        lambda stage: shutil.copy2(source, stage / raw_name),
        label=False, expected_shape=tuple(original["shape_tzyx"]),
    )
    if args.stop_after == "raw":
        return
    pre_path, pre_info = finished_stage(
        sample_dir, "preprocess", raw_name,
        lambda stage: run_command(CHROMOPOSE / "preprocess.py", "--input", raw_path,
                                  "--output", stage, "--profile", profile),
        label=False, expected_shape=tuple(raw_info["shape_tzyx"]),
    )
    if args.stop_after == "preprocess":
        return
    count = raw_info["shape_tzyx"][0]
    selected_anchor = anchor_index(name, count, anchors)
    affine_path, affine_info = finished_stage(
        sample_dir, "affine", raw_name,
        lambda stage: run_command(CHROMOPOSE / "affine.py", "--raw", raw_path,
                                  "--preprocessed", pre_path, "--output", stage,
                                  "--anchor-t", str(selected_anchor),
                                  "--report", stage / "transforms.json"),
        label=False, expected_shape=None,
    )
    if affine_info["shape_tzyx"][0] != count:
        raise ValueError("Affine changed time count")
    if args.stop_after == "affine":
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ChromoPose checkpoint not found: {checkpoint}")
    mask_shape = tuple(affine_info["shape_tzyx"])
    segment_name = f"{name}_shape.tif"
    _, _ = finished_stage(
        sample_dir, "segment", segment_name,
        lambda stage: run_command(CHROMOPOSE / "segmentation" / "infer.py",
                                  "--in_path", affine_path, "--out_path", stage,
                                  "--ckpt", checkpoint, "--gpu_id", args.gpu_ids),
        label=True, expected_shape=mask_shape,
    )
    if args.stop_after == "segment":
        return
    post_name = f"{name}_shape_postprocessed.tif"
    _, _ = finished_stage(
        sample_dir, "post", post_name,
        lambda stage: run_command(CHROMOPOSE / "postprocess.py", "--masks", sample_dir / "segment",
                                  "--raw", sample_dir / "affine", "--output", stage,
                                  "--pattern", segment_name, "--no-preview", "--workers", "1"),
        label=True, expected_shape=mask_shape,
    )
    if args.stop_after == "post":
        return

    def make_track(stage: Path) -> None:
        anchor_file = stage / "anchors.json"
        save_json(anchor_file, {"anchors": {name: {"representative_t": selected_anchor}}})
        run_command(CHROMOPOSE / "track.py", "--input", sample_dir / "post",
                    "--output", stage, "--anchors", anchor_file)

    finished_stage(sample_dir, "track", f"{name}_tracked.tif", make_track,
                   label=True, expected_shape=mask_shape)


def main() -> None:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.anchors:
        args.anchors = args.anchors.expanduser().resolve()
    if args.max_files < 0:
        raise SystemExit("--max-files must be nonnegative")
    if not CHROMOPOSE.is_dir():
        raise SystemExit(f"ChromoPose code not found: {CHROMOPOSE}")
    files = source_files(args.input, args.pattern, args.max_files)
    if args.input.is_dir() and args.output_dir == args.input:
        raise SystemExit("Output directory must differ from the raw input directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchors = load_anchors(args.anchors)
    for index, source in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {sample_name(source)}", flush=True)
        process_sample(source, args, anchors)
    print(json.dumps({"samples": len(files), "completed_through": args.stop_after,
                      "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
