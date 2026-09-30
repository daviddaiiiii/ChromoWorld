#!/usr/bin/env python3
"""Render representative keyframes as five stage-specific sample-grid figures."""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn.functional as F


STAGES = (
    ("Raw DAPI", "raw", "raw"),
    ("Preprocessed", "pre", "preprocessed"),
    ("Registered DAPI", "affine", "registered"),
    ("ChromoPose segmentation", "segmentation", "segmentation"),
    ("Postprocessed", "postprocess", "postprocessed"),
)


def _stage_paths(
    root: Path,
    sample: str,
    segmentation_dir: Path | None = None,
    postprocess_dir: Path | None = None,
) -> dict[str, Path]:
    return {
        "raw": root / "image" / "raw" / f"{sample}-DAPI.tif",
        "pre": root / "image" / "raw_pre" / "tiff" / f"{sample}-DAPI.tif",
        "affine": root / "image" / "raw_pre_affine" / "tiff" / f"{sample}-DAPI.tif",
        "segmentation": (segmentation_dir or root / "mask3d" / "chrp3d_affine") / f"{sample}_shape.tif",
        "postprocess": (postprocess_dir or root / "mask3d" / "chrp3d_affine_post") / f"{sample}_shape_postprocessed.tif",
    }


def _tiff_shape(path: Path) -> tuple[int, ...]:
    with tifffile.TiffFile(path) as tif:
        return tuple(int(v) for v in tif.series[0].shape)


def _read_tzyx_frame(path: Path, t: int) -> np.ndarray:
    """Read just one T frame from a page-stacked TZYX TIFF."""
    with tifffile.TiffFile(path) as tif:
        shape = tuple(int(v) for v in tif.series[0].shape)
        if len(shape) != 4:
            raise ValueError(f"Expected TZYX TIFF at {path}, got shape={shape}")
        n_t, n_z, _, _ = shape
        if t < 0 or t >= n_t:
            raise IndexError(f"t={t} is outside [0, {n_t}) for {path}")
        frame = tif.asarray(key=range(t * n_z, (t + 1) * n_z))
    if frame.ndim != 3:
        raise ValueError(f"Could not read one ZYX frame from {path}; got {frame.shape}")
    return np.asarray(frame)


def _resize_image_for_mask(volume_zyx: np.ndarray, target_zyx: tuple[int, int, int]) -> np.ndarray:
    """Align an affine DAPI volume to the saved mask grid for visualization."""
    if tuple(volume_zyx.shape) == tuple(target_zyx):
        return np.asarray(volume_zyx, dtype=np.float32)
    tensor = torch.from_numpy(np.asarray(volume_zyx, dtype=np.float32))[None, None]
    with torch.inference_mode():
        resized = F.interpolate(
            tensor,
            size=target_zyx,
            mode="trilinear",
            align_corners=False,
        )
    return resized[0, 0].numpy()


def _display_limits(image: np.ndarray, recorded: list[float] | None = None) -> tuple[float, float]:
    if recorded is not None and len(recorded) == 2:
        low, high = float(recorded[0]), float(recorded[1])
        if np.isfinite(low) and np.isfinite(high) and high > low:
            return low, high
    finite = np.asarray(image)[np.isfinite(image)]
    positive = finite[finite > 0]
    values = positive if positive.size else finite
    if values.size == 0:
        return 0.0, 1.0
    high = float(np.percentile(values, 99.7))
    if not np.isfinite(high) or high <= 0:
        high = float(np.max(values))
    return 0.0, max(high, 1.0)


def _project_labels_at_signal_peak(mask_zyx: np.ndarray, dapi_zyx: np.ndarray) -> np.ndarray:
    """Choose one segmented label per XY pixel: the brightest labeled voxel along Z."""
    mask = np.asarray(mask_zyx)
    dapi = np.asarray(dapi_zyx)
    if mask.ndim != 3 or dapi.ndim != 3 or mask.shape != dapi.shape:
        raise ValueError(f"Expected matching ZYX mask and DAPI, got {mask.shape} and {dapi.shape}")
    foreground = mask > 0
    supported = np.any(foreground, axis=0)
    if not np.any(supported):
        return np.zeros(mask.shape[1:], dtype=mask.dtype)
    scores = np.where(foreground, dapi.astype(np.float32, copy=False), -np.inf)
    z_peak = np.argmax(scores, axis=0)
    projected = np.take_along_axis(mask, z_peak[None, ...], axis=0)[0]
    projected[~supported] = 0
    return projected


def _projected_label_edges(labels: np.ndarray) -> np.ndarray:
    """Find only the 2D projected boundaries, avoiding per-Z contour rings."""
    edge = np.zeros(labels.shape, dtype=bool)
    diff = labels[:, 1:] != labels[:, :-1]
    edge[:, 1:] |= diff & (labels[:, 1:] > 0)
    edge[:, :-1] |= diff & (labels[:, :-1] > 0)
    diff = labels[1:, :] != labels[:-1, :]
    edge[1:, :] |= diff & (labels[1:, :] > 0)
    edge[:-1, :] |= diff & (labels[:-1, :] > 0)
    return edge


def _overlay_projected_mask(
    background_mip: np.ndarray,
    mask_zyx: np.ndarray,
    dapi_zyx: np.ndarray,
    limits: tuple[float, float],
    *,
    hue_offset: float = 0.0,
    alpha: float = 0.43,
) -> np.ndarray:
    """Blend a filled 2D instance projection over DAPI, with a single clean outline."""
    low, high = limits
    gray = np.clip((background_mip.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    labels = _project_labels_at_signal_peak(mask_zyx, dapi_zyx)
    max_label = int(labels.max(initial=0))
    palette = np.zeros((max_label + 1, 3), dtype=np.float32)
    for label in np.unique(labels):
        if label == 0:
            continue
        hue = (int(label) * 0.6180339887498949 + hue_offset) % 1.0
        palette[int(label)] = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    foreground = labels > 0
    color_layer = palette[labels]
    rgb[foreground] = (1.0 - alpha) * rgb[foreground] + alpha * color_layer[foreground]
    edges = _projected_label_edges(labels)
    rgb[edges] *= 0.42
    return rgb


def _wrapped_sample_title(sample: str, t: int) -> str:
    lines = textwrap.wrap(sample, width=30, break_long_words=True, break_on_hyphens=True)
    return "\n".join([*lines, f"t = {t}"])


def _validate_records(
    root: Path,
    records: list[dict],
    segmentation_dir: Path | None,
    postprocess_dir: Path | None,
    keyframe_only_masks: bool = False,
    require_all_40: bool = True,
) -> None:
    if require_all_40 and len(records) != 40:
        raise ValueError(f"Expected the 40-sample keyframe manifest, found {len(records)} entries")
    seen: set[str] = set()
    missing: list[str] = []
    invalid: list[str] = []
    for record in records:
        sample = str(record["sample"])
        if sample in seen:
            invalid.append(f"duplicate sample: {sample}")
        seen.add(sample)
        representative = [
            frame for frame in record.get("frames", [])
            if "representative" in frame.get("roles", [])
        ]
        if len(representative) != 1:
            invalid.append(f"{sample}: expected one representative frame, found {len(representative)}")
            continue
        t = int(representative[0]["t"])
        paths = _stage_paths(root, sample, segmentation_dir, postprocess_dir)
        for stage, path in paths.items():
            if not path.is_file():
                missing.append(f"{sample} [{stage}]: {path}")
                continue
            shape = _tiff_shape(path)
            if len(shape) != 4:
                invalid.append(f"{sample} [{stage}]: t={t}, shape={shape}")
                continue
            if keyframe_only_masks and stage in {"segmentation", "postprocess"}:
                if shape[0] != 1:
                    invalid.append(f"{sample} [{stage}]: expected one-frame result, got shape={shape}")
            elif t >= shape[0]:
                invalid.append(f"{sample} [{stage}]: t={t} outside shape={shape}")
        if paths["segmentation"].is_file() and paths["postprocess"].is_file():
            seg_shape = _tiff_shape(paths["segmentation"])
            post_shape = _tiff_shape(paths["postprocess"])
            if seg_shape[1:] != post_shape[1:]:
                invalid.append(f"{sample}: segmentation/postprocess ZYX shapes differ: {seg_shape[1:]}, {post_shape[1:]}")
    if missing or invalid:
        details = "\n".join((missing + invalid)[:30])
        raise RuntimeError(f"Keyframe-stage validation failed ({len(missing)} missing, {len(invalid)} invalid):\n{details}")


def render(
    root: Path,
    manifest: Path,
    out_dir: Path,
    max_samples: int | None = None,
    segmentation_dir: Path | None = None,
    postprocess_dir: Path | None = None,
    keyframe_only_masks: bool = False,
) -> list[Path]:
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if max_samples is not None:
        records = records[:max_samples]
    _validate_records(
        root,
        records,
        segmentation_dir,
        postprocess_dir,
        keyframe_only_masks,
        require_all_40=max_samples is None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, object]] = []
    for record in records:
        sample = str(record["sample"])
        frame_meta = next(f for f in record["frames"] if "representative" in f.get("roles", []))
        t = int(frame_meta["t"])
        paths = _stage_paths(root, sample, segmentation_dir, postprocess_dir)
        seg_t = 0 if keyframe_only_masks else t
        post_t = 0 if keyframe_only_masks else t
        segmentation = _read_tzyx_frame(paths["segmentation"], seg_t)
        postprocess = _read_tzyx_frame(paths["postprocess"], post_t)
        if segmentation.shape != postprocess.shape:
            raise ValueError(f"{sample} t={t}: segmentation/postprocess shapes differ")
        csv_rows.append(
            {
                "sample": sample,
                "t_zero_based": t,
                "segmentation_source_t": seg_t,
                "postprocess_source_t": post_t,
                "segmentation_instances": int(np.unique(segmentation[segmentation > 0]).size),
                "postprocessed_instances": int(np.unique(postprocess[postprocess > 0]).size),
                **{f"{stage}_path": str(path) for stage, path in paths.items()},
            }
        )
    csv_suffix = f"_{len(records)}samples" if len(records) != 40 else ""
    csv_path = out_dir / f"keyframes_and_stage_sources{csv_suffix}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    n_cols = 5
    n_rows = math.ceil(len(records) / n_cols)
    out_paths: list[Path] = []
    for title, stage, file_tag in STAGES:
        figure_height = max(5.0, n_rows * 1.8 + 1.4)
        fig = plt.figure(figsize=(16.0, figure_height), facecolor="white")
        gs = fig.add_gridspec(
            nrows=n_rows,
            ncols=n_cols,
            left=0.025,
            right=0.995,
            top=1.0 - 0.8 / figure_height,
            bottom=0.065,
            hspace=0.34,
            wspace=0.045,
        )
        for index, record in enumerate(records):
            row, col = divmod(index, n_cols)
            sample = str(record["sample"])
            frame_meta = next(f for f in record["frames"] if "representative" in f.get("roles", []))
            t = int(frame_meta["t"])
            paths = _stage_paths(root, sample, segmentation_dir, postprocess_dir)
            ax = fig.add_subplot(gs[row, col])
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(
                _wrapped_sample_title(sample, t),
                fontsize=6.8,
                color="#292d33",
                pad=2.0,
                linespacing=1.0,
            )

            source_t = 0 if keyframe_only_masks and stage in {"segmentation", "postprocess"} else t
            panel = _read_tzyx_frame(paths[stage], source_t)
            if stage in {"raw", "pre", "affine"}:
                projection = np.max(panel, axis=0)
                recorded = frame_meta.get("raw_display") if stage == "raw" else None
                if stage == "pre":
                    recorded = frame_meta.get("preprocessed_display")
                limits = _display_limits(projection, recorded)
                ax.imshow(projection, cmap="gray", vmin=limits[0], vmax=limits[1], interpolation="nearest")
            else:
                affine = _read_tzyx_frame(paths["affine"], t)
                affine = _resize_image_for_mask(affine, tuple(int(v) for v in panel.shape))
                mip = np.max(affine, axis=0)
                limits = _display_limits(mip)
                ax.imshow(
                    _overlay_projected_mask(
                        mip,
                        panel,
                        affine,
                        limits,
                        hue_offset=0.0 if stage == "segmentation" else 0.47,
                    ),
                    interpolation="nearest",
                )
            print(f"{title}: [{index + 1:02d}/{len(records)}] {sample} | t={t}", flush=True)

        fig.suptitle(
            f"ChromoPose 4D — {title} | {len(records)} key frames in {n_rows} × {n_cols}",
            x=0.5,
            y=0.99,
            fontsize=14,
            weight="bold",
            color="#20242a",
        )
        if stage in {"segmentation", "postprocess"}:
            note = (
                "DAPI max-Z projection with a translucent filled instance mask. At each XY pixel, "
                "the shown label is from the brightest DAPI voxel with a nonzero mask; outlines are drawn "
                "once on the 2D projection, not once per Z slice. Colors are panel-local instance labels."
            )
        else:
            note = "Each tile shows the selected zero-based t as a max-Z DAPI projection; display range is per sample."
        fig.text(0.5, 0.022, note, ha="center", va="center", fontsize=7.0, color="#5a6068", wrap=True)

        suffix = f"_{len(records)}samples" if len(records) != 40 else ""
        stem = f"chromopose_keyframes_{file_tag}_{n_rows}x{n_cols}{suffix}"
        png_path = out_dir / f"{stem}.png"
        pdf_path = out_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=140, facecolor="white")
        fig.savefig(pdf_path, facecolor="white")
        plt.close(fig)
        out_paths.extend((png_path, pdf_path))

    out_paths.append(csv_path)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Local ours-live dataset root")
    parser.add_argument("--manifest", type=Path, help="40-sample representative-frame manifest")
    parser.add_argument("--out-dir", type=Path, help="Output directory (defaults under --root)")
    parser.add_argument("--segmentation-dir", type=Path, help="Override the maintained segmentation TIFF directory")
    parser.add_argument("--postprocess-dir", type=Path, help="Override the postprocessed-mask TIFF directory")
    parser.add_argument(
        "--keyframe-only-masks",
        action="store_true",
        help="Read segmentation/postprocessing from one-frame TIFFs (source t=0); other stages use manifest t",
    )
    parser.add_argument("--max-samples", type=int, help="Render only the first N samples for layout testing")
    args = parser.parse_args()
    manifest = args.manifest or (args.root / "comparison_mips_20260917" / "manifest.json")
    out_dir = args.out_dir or (args.root / "keyframe_pipeline_grid")
    paths = render(
        args.root,
        manifest,
        out_dir,
        args.max_samples,
        args.segmentation_dir,
        args.postprocess_dir,
        args.keyframe_only_masks,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
