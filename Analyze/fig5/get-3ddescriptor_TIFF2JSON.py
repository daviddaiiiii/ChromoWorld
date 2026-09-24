#!/usr/bin/env python3
"""Extract per-instance 3D physical descriptors from volume and label TIFFs.

The input mask is an integer label volume (0 is background).  One JSONL record
is written per volume, with one nested instance record per positive label.
Voxel spacing is explicit and ordered as Z, Y, X; use calibrated values for
physical units instead of the default unit spacing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
from skimage.morphology import skeletonize


TIFF_SUFFIXES = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, required=True, help="3D TIFF file or directory.")
    parser.add_argument("--mask", type=Path, required=True, help="Label TIFF file or directory.")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--voxel-size-zyx",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SZ", "SY", "SX"),
        help="Voxel spacing in physical units, ordered Z Y X.",
    )
    parser.add_argument("--max-hull-points", type=int, default=5000)
    parser.add_argument("--no-intensity", action="store_true")
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


def read_tiff(path: Path) -> tuple[np.ndarray, str]:
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        array = np.asarray(series.asarray())
        axes = str(series.axes or "")
    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D TIFF stack, got shape={array.shape} axes={axes!r} for {path}"
        )
    return array, axes or "ZYX"


def pair_mask(volume_path: Path, mask_root: Path) -> Path | None:
    if mask_root.is_file():
        return mask_root
    candidates = [
        mask_root / f"{volume_path.stem}.tif",
        mask_root / f"{volume_path.stem}.tiff",
        mask_root / f"{volume_path.stem}_mask.tif",
        mask_root / f"{volume_path.stem}_mask.tiff",
        mask_root / f"{volume_path.stem}_shape.tif",
        mask_root / f"{volume_path.stem}_shape.tiff",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def physical_coordinates(coords: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    return (coords.astype(np.float64) + 0.5) * spacing[None, :]


def inertia_features(coords_phys: np.ndarray) -> dict[str, Any]:
    if len(coords_phys) < 2:
        return {
            "elongation": 0.0,
            "axis_ratio_min_max": 0.0,
            "principal_axis_std_um": [0.0, 0.0, 0.0],
            "inertia_eigenvalues": [0.0, 0.0, 0.0],
        }
    centered = coords_phys - coords_phys.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False, bias=True)
    covariance = np.atleast_2d(covariance)
    if covariance.shape != (3, 3):
        covariance = np.zeros((3, 3), dtype=np.float64)
    std_eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    inertia = np.zeros((3, 3), dtype=np.float64)
    x, y, z = centered[:, 2], centered[:, 1], centered[:, 0]
    inertia[0, 0] = np.sum(y * y + z * z)
    inertia[1, 1] = np.sum(x * x + z * z)
    inertia[2, 2] = np.sum(x * x + y * y)
    inertia[0, 1] = inertia[1, 0] = -np.sum(x * y)
    inertia[0, 2] = inertia[2, 0] = -np.sum(x * z)
    inertia[1, 2] = inertia[2, 1] = -np.sum(y * z)
    inertia_eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(inertia), 0.0))
    ratio = (
        float(inertia_eigenvalues[0] / inertia_eigenvalues[-1])
        if inertia_eigenvalues[-1] > 0
        else 0.0
    )
    return {
        "elongation": float(1.0 - ratio),
        "axis_ratio_min_max": ratio,
        "principal_axis_std_um": np.sqrt(std_eigenvalues).tolist(),
        "inertia_eigenvalues": inertia_eigenvalues.tolist(),
    }


def surface_area(mask: np.ndarray, spacing: np.ndarray) -> tuple[float, int]:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    z_faces = np.sum(center & ~padded[:-2, 1:-1, 1:-1]) + np.sum(
        center & ~padded[2:, 1:-1, 1:-1]
    )
    y_faces = np.sum(center & ~padded[1:-1, :-2, 1:-1]) + np.sum(
        center & ~padded[1:-1, 2:, 1:-1]
    )
    x_faces = np.sum(center & ~padded[1:-1, 1:-1, :-2]) + np.sum(
        center & ~padded[1:-1, 1:-1, 2:]
    )
    face_count = int(z_faces + y_faces + x_faces)
    area = float(z_faces * spacing[1] * spacing[2])
    area += float(y_faces * spacing[0] * spacing[2])
    area += float(x_faces * spacing[0] * spacing[1])
    return area, face_count


def convex_hull_volume(coords_phys: np.ndarray, max_points: int) -> float:
    if len(coords_phys) < 4:
        return 0.0
    boundary = coords_phys
    if len(boundary) > max_points:
        indices = np.linspace(0, len(boundary) - 1, max_points, dtype=np.int64)
        boundary = boundary[indices]
    try:
        return float(ConvexHull(boundary).volume)
    except Exception:
        return 0.0


def skeleton_features(mask: np.ndarray, spacing: np.ndarray) -> dict[str, float]:
    skeleton = skeletonize(mask)
    coords = np.argwhere(skeleton)
    if len(coords) <= 1:
        return {
            "skeleton_voxels": float(len(coords)),
            "skeleton_length_um": 0.0,
            "skeleton_endpoints": 0.0,
            "skeleton_branchpoints": 0.0,
            "tortuosity": 0.0,
        }
    point_set = {tuple(int(value) for value in row) for row in coords}
    endpoints = []
    branchpoints = []
    for z, y, x in point_set:
        neighbors = 0
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dz, dy, dx) != (0, 0, 0) and (z + dz, y + dy, x + dx) in point_set:
                        neighbors += 1
        if neighbors == 1:
            endpoints.append((z, y, x))
        elif neighbors >= 3:
            branchpoints.append((z, y, x))

    length = 0.0
    positive_offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) > (0, 0, 0)
    ]
    for z, y, x in point_set:
        for dz, dy, dx in positive_offsets:
            if (z + dz, y + dy, x + dx) in point_set:
                length += float(np.linalg.norm(np.asarray((dz, dy, dx), dtype=float) * spacing))

    coords_phys = physical_coordinates(coords, spacing)
    if len(endpoints) >= 2:
        endpoint_phys = physical_coordinates(np.asarray(endpoints), spacing)
        straight = 0.0
        for index in range(len(endpoint_phys)):
            distances = np.linalg.norm(endpoint_phys[index + 1 :] - endpoint_phys[index], axis=1)
            if len(distances):
                straight = max(straight, float(distances.max()))
    else:
        straight = float(np.linalg.norm(coords_phys.max(axis=0) - coords_phys.min(axis=0)))
    tortuosity = float(np.clip(1.0 - straight / length, 0.0, 1.0)) if length > 0 else 0.0
    return {
        "skeleton_voxels": float(len(coords)),
        "skeleton_length_um": length,
        "skeleton_endpoints": float(len(endpoints)),
        "skeleton_branchpoints": float(len(branchpoints)),
        "tortuosity": tortuosity,
    }


def intensity_features(volume: np.ndarray, mask: np.ndarray, spacing: np.ndarray) -> dict[str, float]:
    values = volume[mask].astype(np.float64)
    if values.size == 0:
        return {}
    median = float(np.median(values))
    weights = values - values.min()
    coords_phys = physical_coordinates(np.argwhere(mask), spacing)
    if weights.sum() > 0:
        weighted_center = (coords_phys * weights[:, None]).sum(axis=0) / weights.sum()
        displacement = float(np.linalg.norm(weighted_center - coords_phys.mean(axis=0)))
    else:
        displacement = 0.0
    return {
        "intensity_mean": float(values.mean()),
        "intensity_median": median,
        "intensity_std": float(values.std()),
        "intensity_mad": float(np.median(np.abs(values - median))),
        "intensity_p25": float(np.quantile(values, 0.25)),
        "intensity_p75": float(np.quantile(values, 0.75)),
        "integrated_intensity": float(values.sum()),
        "intensity_mass_displacement_um": displacement,
    }


def extract_instance(
    instance_mask: np.ndarray,
    volume: np.ndarray,
    union_voxels: int,
    spacing: np.ndarray,
    max_hull_points: int,
    include_intensity: bool,
) -> dict[str, Any]:
    coords = np.argwhere(instance_mask)
    volume_voxels = int(len(coords))
    physical_volume = float(volume_voxels * np.prod(spacing))
    coords_phys = physical_coordinates(coords, spacing)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_voxels = bbox_max - bbox_min + 1
    surface_um2, surface_faces = surface_area(instance_mask, spacing)
    hull_points = np.argwhere(instance_mask ^ binary_erosion(instance_mask, structure=np.ones((3, 3, 3))))
    hull_volume = convex_hull_volume(physical_coordinates(hull_points, spacing), max_hull_points)
    sphericity = (
        (math.pi ** (1.0 / 3.0)) * ((6.0 * physical_volume) ** (2.0 / 3.0)) / surface_um2
        if surface_um2 > 0 and physical_volume > 0
        else 0.0
    )
    equivalent_diameter = (
        (6.0 * physical_volume / math.pi) ** (1.0 / 3.0) if physical_volume > 0 else 0.0
    )
    row: dict[str, Any] = {
        "volume_voxels": volume_voxels,
        "volume": float(volume_voxels / union_voxels) if union_voxels > 0 else 0.0,
        "volume_fraction": float(volume_voxels / union_voxels) if union_voxels > 0 else 0.0,
        "volume_um3": physical_volume,
        "centroid_zyx_voxels": coords.mean(axis=0).tolist(),
        "centroid_zyx_um": coords_phys.mean(axis=0).tolist(),
        "bbox_min_zyx_voxels": bbox_min.tolist(),
        "bbox_max_zyx_voxels": bbox_max.tolist(),
        "bbox_size_zyx_voxels": bbox_voxels.tolist(),
        "bbox_size_zyx_um": (bbox_voxels.astype(float) * spacing).tolist(),
        "bbox_volume_fraction": float(volume_voxels / np.prod(bbox_voxels)),
        "equivalent_diameter_um": equivalent_diameter,
        "surface_area_um2": surface_um2,
        "surface_face_count": surface_faces,
        "convex_hull_volume_um3": hull_volume,
        "solidity": float(physical_volume / hull_volume) if hull_volume > 0 else 1.0,
        "sphericity": float(sphericity),
        "border_voxel_fraction": float(
            np.count_nonzero(
                instance_mask
                & (
                    (np.indices(instance_mask.shape)[0] == 0)
                    | (np.indices(instance_mask.shape)[0] == instance_mask.shape[0] - 1)
                    | (np.indices(instance_mask.shape)[1] == 0)
                    | (np.indices(instance_mask.shape)[1] == instance_mask.shape[1] - 1)
                    | (np.indices(instance_mask.shape)[2] == 0)
                    | (np.indices(instance_mask.shape)[2] == instance_mask.shape[2] - 1)
                )
            )
            / volume_voxels
            if volume_voxels > 0
            else 0.0
        ),
        "touches_volume_border": int(
            bool(
                bbox_min[0] == 0
                or bbox_min[1] == 0
                or bbox_min[2] == 0
                or bbox_max[0] == instance_mask.shape[0] - 1
                or bbox_max[1] == instance_mask.shape[1] - 1
                or bbox_max[2] == instance_mask.shape[2] - 1
            )
        ),
    }
    row.update(inertia_features(coords_phys))
    row.update(skeleton_features(instance_mask, spacing))
    if include_intensity:
        row.update(intensity_features(volume, instance_mask, spacing))
    return row


def process_pair(
    volume_path: Path,
    mask_path: Path,
    spacing: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    volume, volume_axes = read_tiff(volume_path)
    mask, mask_axes = read_tiff(mask_path)
    if volume.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: volume={volume.shape}, mask={mask.shape} for {volume_path.name}"
        )
    label_ids = [int(value) for value in np.unique(mask) if int(value) > 0]
    union_voxels = int(np.count_nonzero(mask))
    instances = []
    for instance_id in label_ids:
        instance_mask = mask == instance_id
        features = extract_instance(
            instance_mask,
            volume,
            union_voxels,
            spacing,
            int(args.max_hull_points),
            not args.no_intensity,
        )
        instances.append({"instance_id": instance_id, **features})
    return {
        "schema_version": "chromoworld.instance.v1",
        "modality": "3d",
        "volume_basename": volume_path.stem,
        "volume_file": volume_path.name,
        "mask_file": mask_path.name,
        "volume_axes": volume_axes,
        "mask_axes": mask_axes,
        "shape_zyx": list(volume.shape),
        "voxel_size_zyx": spacing.tolist(),
        "voxel_volume": float(np.prod(spacing)),
        "total_mask_voxels": union_voxels,
        "n_instances": len(instances),
        "feature_names": sorted({key for row in instances for key in row if key != "instance_id"}),
        "instances": instances,
    }


def main() -> None:
    args = parse_args()
    volumes = collect_tiffs(args.volume)
    if args.limit > 0:
        volumes = volumes[: int(args.limit)]
    if not volumes:
        raise SystemExit(f"No volume TIFF files found under {args.volume}")
    spacing = np.asarray(args.voxel_size_zyx, dtype=np.float64)
    if np.any(spacing <= 0):
        raise SystemExit("--voxel-size-zyx values must be positive")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = []
    skipped = []
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for volume_path in volumes:
            mask_path = pair_mask(volume_path, args.mask)
            if mask_path is None:
                skipped.append({"volume_file": volume_path.name, "reason": "missing_mask"})
                continue
            try:
                record = process_pair(volume_path, mask_path, spacing, args)
            except Exception as error:
                skipped.append(
                    {
                        "volume_file": volume_path.name,
                        "mask_file": mask_path.name,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    report_path = args.report_json or args.output_jsonl.with_name("descriptor_3d_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "chromoworld.3d_descriptor.v1",
        "output_jsonl": args.output_jsonl.name,
        "voxel_size_zyx": spacing.tolist(),
        "n_input_volumes": len(volumes),
        "n_output_volumes": len(records),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_instances": int(sum(record["n_instances"] for record in records)),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
