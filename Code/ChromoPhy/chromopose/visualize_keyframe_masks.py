#!/usr/bin/env python3
"""Render standalone mask-only grids for the 40 selected 4D keyframes."""

from __future__ import annotations

import argparse
import colorsys
import csv
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


N_COLUMNS = 5
KEYFRAME_DIR = "keyframe_pipeline_grid_20260925"
MASK_METHODS = (
    ("ChromoPose segmentation", "chromopose", "chromopose_keyframes_segmentation"),
    ("ChromoPose postprocessed", "postprocess", "chromopose_keyframes_postprocessed"),
    ("CPSAM", "cpsam", "cpsam_keyframes_segmentation"),
    ("Fast CHIMP", "chimp", "fastchimp_keyframes_segmentation"),
)


def _read_records(path: Path, max_samples: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if max_samples is None and len(records) != 40:
        raise ValueError(f"Expected 40 keyframe records, found {len(records)} in {path}")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("--max-samples must be positive")
        records = records[:max_samples]
    if not records:
        raise ValueError(f"No keyframe records found in {path}")
    samples = [record["sample"] for record in records]
    if len(samples) != len(set(samples)):
        raise ValueError("The keyframe table contains duplicate sample names")
    return records


def _image_path(root: Path, sample: str) -> Path:
    return root / "image" / "raw_pre_affine" / "tiff" / f"{sample}-DAPI.tif"


def _mask_path(root: Path, sample: str, method: str) -> Path:
    keyframe_root = root / KEYFRAME_DIR
    if method == "chromopose":
        return keyframe_root / "current_keyframe_segmentation" / f"{sample}_shape.tif"
    if method == "postprocess":
        return (
            keyframe_root
            / "current_keyframe_postprocess"
            / f"{sample}_shape_postprocessed.tif"
        )
    return root / "mask3d" / method / f"{sample}_shape.tif"


def _source_time(record: dict[str, str], method: str) -> int:
    if method == "chromopose":
        return int(record.get("segmentation_source_t") or 0)
    if method == "postprocess":
        return int(record.get("postprocess_source_t") or 0)
    return int(record["t_zero_based"])


def _shape_and_axes(path: Path) -> tuple[tuple[int, ...], str]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return tuple(int(value) for value in series.shape), series.axes


def _read_tzyx_frame(path: Path, t: int) -> np.ndarray:
    """Read only one timepoint from a page-stacked TZYX TIFF."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        axes = series.axes
        if len(shape) != 4 or axes != "TZYX":
            raise ValueError(f"Expected a TZYX TIFF at {path}, got axes={axes}, shape={shape}")
        n_t, n_z, _, _ = shape
        if t < 0 or t >= n_t:
            raise IndexError(f"t={t} is outside [0, {n_t}) for {path}")
        frame = tif.asarray(key=range(t * n_z, (t + 1) * n_z))
    if frame.ndim != 3:
        raise ValueError(f"Expected a ZYX frame from {path}, got shape={frame.shape}")
    return np.asarray(frame)


def _resize_labels_nearest(mask: np.ndarray, target_zyx: tuple[int, int, int]) -> np.ndarray:
    """Resize categorical labels with the same floor-index rule as nearest interpolation."""
    if tuple(mask.shape) == tuple(target_zyx):
        return mask
    indices = [
        np.minimum(
            (np.arange(target, dtype=np.float64) * source / target).astype(np.intp),
            source - 1,
        )
        for source, target in zip(mask.shape, target_zyx, strict=True)
    ]
    resized = np.take(mask, indices[0], axis=0)
    resized = np.take(resized, indices[1], axis=1)
    return np.take(resized, indices[2], axis=2)


def _project_mask_front_to_back(mask_zyx: np.ndarray) -> np.ndarray:
    """Make a colored XY max projection, keeping the first nonzero Z label per pixel."""
    foreground = mask_zyx > 0
    supported = np.any(foreground, axis=0)
    if not np.any(supported):
        return np.zeros(mask_zyx.shape[1:], dtype=mask_zyx.dtype)
    z_first = np.argmax(foreground, axis=0)
    labels = np.take_along_axis(mask_zyx, z_first[None, ...], axis=0)[0]
    labels[~supported] = 0
    return labels


def _projected_label_edges(labels: np.ndarray) -> np.ndarray:
    """Compute label boundaries once in 2D."""
    edges = np.zeros(labels.shape, dtype=bool)
    diff = labels[:, 1:] != labels[:, :-1]
    edges[:, 1:] |= diff & (labels[:, 1:] > 0)
    edges[:, :-1] |= diff & (labels[:, :-1] > 0)
    diff = labels[1:, :] != labels[:-1, :]
    edges[1:, :] |= diff & (labels[1:, :] > 0)
    edges[:-1, :] |= diff & (labels[:-1, :] > 0)
    return edges


def _color_mask(mask_zyx: np.ndarray) -> np.ndarray:
    """Color the mask projection on black, without showing or blending an image."""
    labels = _project_mask_front_to_back(mask_zyx)
    max_label = int(labels.max(initial=0))
    palette = np.zeros((max_label + 1, 3), dtype=np.float32)
    for label in np.unique(labels):
        if label == 0:
            continue
        hue = (int(label) * 0.6180339887498949) % 1.0
        palette[int(label)] = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    rgb = palette[labels]
    rgb[_projected_label_edges(labels)] *= 0.35
    return rgb


def _sample_title(sample: str, t: int) -> str:
    lines = textwrap.wrap(sample, width=30, break_long_words=True, break_on_hyphens=True)
    return "\n".join([*lines, f"t = {t}"])


def _validate_and_count(
    root: Path,
    records: list[dict[str, str]],
) -> list[dict[str, object]]:
    counts: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        sample = record["sample"]
        t = int(record["t_zero_based"])
        image_path = _image_path(root, sample)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing registered DAPI image for {sample}: {image_path}")
        image_shape, image_axes = _shape_and_axes(image_path)
        if image_axes != "TZYX" or len(image_shape) != 4:
            raise ValueError(f"{sample}: expected TZYX registered DAPI, got {image_axes} {image_shape}")
        if t < 0 or t >= image_shape[0]:
            raise IndexError(f"{sample}: t={t} is outside the registered DAPI time range {image_shape[0]}")
        target_zyx = image_shape[1:]

        row: dict[str, object] = {
            "sample": sample,
            "t_zero_based": t,
            "target_affine_zyx": "x".join(map(str, target_zyx)),
        }
        for title, method, output_tag in MASK_METHODS:
            path = _mask_path(root, sample, method)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {title} mask for {sample}: {path}")
            mask_shape, mask_axes = _shape_and_axes(path)
            if mask_axes != "TZYX" or len(mask_shape) != 4:
                raise ValueError(f"{sample}: expected TZYX {title} mask, got {mask_axes} {mask_shape}")
            source_t = _source_time(record, method)
            if source_t < 0 or source_t >= mask_shape[0]:
                raise IndexError(f"{sample}: {title} source t={source_t} is outside shape={mask_shape}")
            if method in {"chromopose", "postprocess"} and mask_shape[1:] != target_zyx:
                raise ValueError(
                    f"{sample}: {title} should already match the affine DAPI grid; "
                    f"mask={mask_shape[1:]}, target={target_zyx}"
                )

            mask = _read_tzyx_frame(path, source_t)
            resized = _resize_labels_nearest(mask, target_zyx)
            n_instances = int(np.unique(resized[resized > 0]).size)
            if n_instances == 0:
                raise ValueError(f"{sample}: {title} mask is empty at keyframe t={t}")
            row[f"{output_tag}_source_zyx"] = "x".join(map(str, mask.shape))
            row[f"{output_tag}_instances"] = n_instances
            row[f"{output_tag}_resized"] = mask.shape != target_zyx
        counts.append(row)
        print(f"Validated [{index:02d}/{len(records)}] {sample} | t={t}", flush=True)
    return counts


def _render_method(
    root: Path,
    records: list[dict[str, str]],
    out_dir: Path,
    method_info: tuple[str, str, str],
) -> tuple[Path, Path]:
    title, method, output_tag = method_info
    n_rows = math.ceil(len(records) / N_COLUMNS)
    figure_height = max(5.0, n_rows * 1.8 + 1.4)
    fig = plt.figure(figsize=(16.0, figure_height), facecolor="white")
    grid = fig.add_gridspec(
        nrows=n_rows,
        ncols=N_COLUMNS,
        left=0.025,
        right=0.995,
        top=1.0 - 0.8 / figure_height,
        bottom=0.065,
        hspace=0.34,
        wspace=0.045,
    )
    for index, record in enumerate(records):
        sample = record["sample"]
        t = int(record["t_zero_based"])
        image_shape, _ = _shape_and_axes(_image_path(root, sample))
        target_zyx = image_shape[1:]
        path = _mask_path(root, sample, method)
        mask = _read_tzyx_frame(path, _source_time(record, method))
        if mask.shape != target_zyx:
            mask = _resize_labels_nearest(mask, target_zyx)

        row, col = divmod(index, N_COLUMNS)
        ax = fig.add_subplot(grid[row, col])
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.set_title(
            _sample_title(sample, t),
            fontsize=6.8,
            color="#292d33",
            pad=2.0,
            linespacing=1.0,
        )
        ax.imshow(_color_mask(mask), interpolation="nearest")
        print(f"Rendered {title}: [{index + 1:02d}/{len(records)}] {sample} | t={t}", flush=True)

    fig.suptitle(
        f"{title} mask | {len(records)} key frames in {n_rows} × {N_COLUMNS}",
        x=0.5,
        y=0.99,
        fontsize=14,
        weight="bold",
        color="#20242a",
    )
    fig.text(
        0.5,
        0.022,
        "Standalone colored mask projection only (no DAPI overlay). At each XY pixel, the first "
        "nonzero label along Z is shown. Colors encode sample-local instance IDs. Masks are "
        "nearest-neighbor resized to each sample's registered DAPI ZYX grid.",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#5a6068",
        wrap=True,
    )
    stem = f"{output_tag}_8x{N_COLUMNS}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=140, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ourslive-root", type=Path, required=True, help="Root of the local ours-live data")
    parser.add_argument(
        "--source-table",
        type=Path,
        help="Keyframe CSV (defaults to keyframe_pipeline_grid_20260925/keyframes_and_stage_sources.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory (defaults to chromopose/demo_outputs/keyframe_pipeline_grid_20260925)",
    )
    parser.add_argument("--max-samples", type=int, help="Render the first N samples for a quick layout check")
    args = parser.parse_args()

    root = args.ourslive_root.expanduser().resolve()
    source_table = args.source_table or (
        root / KEYFRAME_DIR / "keyframes_and_stage_sources.csv"
    )
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent / "demo_outputs" / KEYFRAME_DIR
    )
    records = _read_records(source_table, args.max_samples)
    counts = _validate_and_count(root, records)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_path = out_dir / "keyframe_mask_instance_counts.csv"
    with counts_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(counts[0].keys()))
        writer.writeheader()
        writer.writerows(counts)

    outputs: list[Path] = [counts_path]
    for method in MASK_METHODS:
        outputs.extend(_render_method(root, records, out_dir, method))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
