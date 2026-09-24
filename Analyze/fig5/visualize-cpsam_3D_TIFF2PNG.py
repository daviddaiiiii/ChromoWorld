#!/usr/bin/env python3
"""Create a compact 3D CPSAM segmentation visualization.

The input volume is treated as ZYX, matching the Fig5 Cellpose-SAM mask
pipeline.  The output contains DAPI and CPSAM overlay views for a maximum
intensity projection and one selected z-slice.  CPSAM labels are instance IDs,
not chromosome classes, so each positive ID receives a deterministic colour.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile


def project_root() -> Path:
    """Find the repository root when run in-place or as a copied script."""
    resolved = Path(__file__).resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "release").is_dir():
            return candidate
    return Path.cwd()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve relative paths from the ChromoWorld project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root() / path


def read_3d_tiff(path: Path) -> tuple[np.ndarray, str]:
    """Read a three-dimensional TIFF and preserve its recorded axes string."""
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        array = np.asarray(series.asarray())
        axes = str(series.axes or "ZYX")
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF, got shape={array.shape} axes={axes!r}: {path}")
    return array, axes


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Robustly map an intensity volume to uint8 for display."""
    values = volume.astype(np.float32, copy=False)
    low, high = np.percentile(values, [1.0, 99.8])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def instance_color_bgr(instance_id: int) -> tuple[int, int, int]:
    """Return a deterministic high-contrast BGR colour for one instance ID."""
    hue = (float(instance_id) * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(round(blue * 255)), int(round(green * 255)), int(round(red * 255))


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Render a label image as a BGR colour image with a black background."""
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(mask):
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        output[mask == instance_id] = instance_color_bgr(instance_id)
    return output


def blend_mask(image_gray: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay instance colours on a grayscale image and draw boundaries."""
    base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    colours = colorize_mask(mask).astype(np.float32)
    foreground = mask > 0
    opacity = float(np.clip(alpha, 0.0, 1.0))
    output = base.copy()
    output[foreground] = base[foreground] * (1.0 - opacity) + colours[foreground] * opacity
    output = np.clip(output, 0, 255).astype(np.uint8)
    for instance_id in np.unique(mask):
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        binary = (mask == instance_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(output, contours, -1, instance_color_bgr(instance_id), 1, lineType=cv2.LINE_AA)
    return output


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    strip = np.full((36, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(strip, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([strip, image])


def resize_longest_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0 or max(image.shape[:2]) <= max_edge:
        return image
    scale = float(max_edge) / max(image.shape[:2])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", required=True, help="Processed/raw 3D DAPI TIFF used by the segmentation stage.")
    parser.add_argument("--mask", required=True, help="CPSAM integer-label TIFF from get-3dmask_TIFF2TIFF.py.")
    parser.add_argument("--output-dir", default="release/Code/Analyze/fig5/demo_outputs/cpsam")
    parser.add_argument("--slice-index", type=int, default=None, help="z index; defaults to the middle slice.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask opacity in [0, 1].")
    parser.add_argument("--max-edge", type=int, default=1600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    volume_path = resolve_project_path(args.volume)
    mask_path = resolve_project_path(args.mask)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume, volume_axes = read_3d_tiff(volume_path)
    mask, mask_axes = read_3d_tiff(mask_path)
    if volume.shape != mask.shape:
        raise SystemExit(f"Volume and mask shapes differ: {volume.shape} vs {mask.shape}")
    volume_u8 = normalize_volume(volume)
    mask = mask.astype(np.int64, copy=False)
    z_index = volume.shape[0] // 2 if args.slice_index is None else int(args.slice_index)
    if not 0 <= z_index < volume.shape[0]:
        raise SystemExit(f"slice-index must be in [0, {volume.shape[0] - 1}]")

    # The MIP label is taken from the z-plane containing the brightest DAPI voxel.
    mip_image = volume_u8.max(axis=0)
    brightest_z = volume_u8.argmax(axis=0)
    mip_mask = np.take_along_axis(mask, brightest_z[None, ...], axis=0)[0]
    slice_image = volume_u8[z_index]
    slice_mask = mask[z_index]

    panels = [
        add_title(cv2.cvtColor(mip_image, cv2.COLOR_GRAY2BGR), "DAPI maximum-intensity projection"),
        add_title(blend_mask(mip_image, mip_mask, args.alpha), "CPSAM instances on DAPI MIP"),
        add_title(cv2.cvtColor(slice_image, cv2.COLOR_GRAY2BGR), f"DAPI z={z_index}"),
        add_title(blend_mask(slice_image, slice_mask, args.alpha), f"CPSAM instances on z={z_index}"),
    ]
    panel_height = max(panel.shape[0] for panel in panels)
    panel_width = max(panel.shape[1] for panel in panels)
    normalized_panels = []
    for panel in panels:
        canvas = np.full((panel_height, panel_width, 3), 255, dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        normalized_panels.append(canvas)
    top = cv2.hconcat(normalized_panels[:2])
    bottom = cv2.hconcat(normalized_panels[2:])
    output = resize_longest_edge(cv2.vconcat([top, bottom]), args.max_edge)

    stem = volume_path.stem
    output_path = output_dir / f"{stem}_cpsam_demo.png"
    cv2.imwrite(str(output_path), output)
    manifest = {
        "schema_version": "chromoworld.fig5.cpsam_visualization.v1",
        "project": "Chromo World",
        "component": "CPSAM",
        "volume_file": str(volume_path),
        "mask_file": str(mask_path),
        "volume_axes": volume_axes,
        "mask_axes": mask_axes,
        "shape_zyx": list(volume.shape),
        "slice_index": z_index,
        "n_instances": int(np.unique(mask[mask > 0]).size),
        "nonzero_mask_voxels": int((mask > 0).sum()),
        "alpha": float(args.alpha),
        "output_file": str(output_path),
    }
    (output_dir / f"{stem}_cpsam_demo.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
