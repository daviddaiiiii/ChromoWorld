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
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    label as ndimage_label,
)
from scipy.spatial import ConvexHull
from skimage.measure import euler_number as measure_euler_number
from skimage.morphology import skeletonize


TIFF_SUFFIXES = {".tif", ".tiff"}
HISTORICAL_MIN_INSTANCE_VOXELS = 20
HISTORICAL_MIN_VOLUME_FRACTION = 0.001
HISTORICAL_MAX_VOLUME_FRACTION = 0.2


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
    parser.add_argument(
        "--with-texture",
        action="store_true",
        help="Compute masked 3D GLCM descriptors, corresponding to the 2D texture option.",
    )
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


def inertia_features(coords: np.ndarray, coords_phys: np.ndarray) -> dict[str, Any]:
    if len(coords) < 2:
        return {
            "elongation": 0.0,
            "axis_ratio_min_max": 0.0,
            "principal_axis_std_um": [0.0, 0.0, 0.0],
            "principal_axis_lengths_um": [0.0, 0.0, 0.0],
            "principal_axis_directions_zyx": [[0.0, 0.0, 0.0]] * 3,
            "major_axis_length_um": 0.0,
            "middle_axis_length_um": 0.0,
            "minor_axis_length_um": 0.0,
            "axis_ratio": 0.0,
            "eccentricity_3d": 0.0,
            "inertia_eigenvalues": [0.0, 0.0, 0.0],
        }
    # The historical v7 shape ratios use voxel coordinates, without physical
    # spacing. Keep calibrated spread as an additional physical descriptor.
    centered_voxels = coords.astype(np.float64) - coords.mean(axis=0, keepdims=True)
    inertia = np.zeros((3, 3), dtype=np.float64)
    x, y, z = centered_voxels[:, 2], centered_voxels[:, 1], centered_voxels[:, 0]
    inertia[0, 0] = np.sum(y * y + z * z)
    inertia[1, 1] = np.sum(x * x + z * z)
    inertia[2, 2] = np.sum(x * x + y * y)
    inertia[0, 1] = inertia[1, 0] = -np.sum(x * y)
    inertia[0, 2] = inertia[2, 0] = -np.sum(x * z)
    inertia[1, 2] = inertia[2, 1] = -np.sum(y * z)
    inertia_eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(inertia)))
    ratio = (
        float(inertia_eigenvalues[0] / inertia_eigenvalues[-1])
        if inertia_eigenvalues[-1] > 0
        else 0.0
    )
    centered_phys = coords_phys - coords_phys.mean(axis=0, keepdims=True)
    covariance = np.cov(centered_phys, rowvar=False, bias=True)
    covariance = np.atleast_2d(covariance)
    if covariance.shape != (3, 3):
        covariance = np.zeros((3, 3), dtype=np.float64)
    covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(covariance)
    axis_order = np.argsort(np.maximum(covariance_eigenvalues, 0.0))[::-1]
    std_eigenvalues = np.maximum(covariance_eigenvalues[axis_order], 0.0)
    axis_directions = covariance_eigenvectors[:, axis_order].T
    axis_std = np.sqrt(std_eigenvalues)
    axis_lengths = 4.0 * axis_std
    axis_ratio = (
        float(axis_std[-1] / axis_std[0]) if axis_std[0] > 0 else 0.0
    )
    return {
        "elongation": float(1.0 - ratio),
        "axis_ratio_min_max": ratio,
        "axis_ratio": axis_ratio,
        "eccentricity_3d": float(np.sqrt(max(0.0, 1.0 - axis_ratio**2))),
        "principal_axis_std_um": axis_std.tolist(),
        "principal_axis_lengths_um": axis_lengths.tolist(),
        "principal_axis_directions_zyx": axis_directions.tolist(),
        "major_axis_length_um": float(axis_lengths[0]),
        "middle_axis_length_um": float(axis_lengths[1]),
        "minor_axis_length_um": float(axis_lengths[2]),
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


def convex_hull_measurements(coords_phys: np.ndarray, max_points: int) -> tuple[float, float]:
    if len(coords_phys) < 4:
        return 0.0, 0.0
    boundary = coords_phys
    if len(boundary) > max_points:
        indices = np.linspace(0, len(boundary) - 1, max_points, dtype=np.int64)
        boundary = boundary[indices]
    try:
        hull = ConvexHull(boundary)
        return float(hull.volume), float(hull.area)
    except Exception:
        return 0.0, 0.0


def historical_solidity(coords: np.ndarray, voxel_count: int) -> float:
    """Historical v7 solidity: voxel count divided by hull volume of voxel centers."""
    if len(coords) < 4:
        return 1.0
    try:
        hull_volume = float(ConvexHull(coords).volume)
        return float(voxel_count / hull_volume) if hull_volume > 0 else 1.0
    except Exception:
        return 1.0


def radius_features(mask: np.ndarray, spacing: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Measure local 3D half-widths with a spacing-aware Euclidean distance map."""
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    distances = distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    values = distances[mask]
    if values.size == 0:
        return (
            {
                "radius_mean_um": 0.0,
                "radius_median_um": 0.0,
                "radius_std_um": 0.0,
                "radius_cv": 0.0,
                "radius_max_um": 0.0,
            },
            distances,
        )
    mean = float(values.mean())
    std = float(values.std())
    return (
        {
            "radius_mean_um": mean,
            "radius_median_um": float(np.median(values)),
            "radius_std_um": std,
            "radius_cv": float(std / mean) if mean > 0 else 0.0,
            "radius_max_um": float(values.max()),
        },
        distances,
    )


def topology_features(mask: np.ndarray) -> dict[str, float]:
    """Count 3D components, tunnels, and enclosed cavities using dual connectivity."""
    foreground_structure = generate_binary_structure(3, 3)  # 26-connected foreground
    background_structure = generate_binary_structure(3, 1)  # 6-connected background
    _, components = ndimage_label(mask, structure=foreground_structure)
    euler = int(measure_euler_number(mask, connectivity=3))

    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    background_labels, background_count = ndimage_label(
        ~padded, structure=background_structure
    )
    boundary_labels = np.concatenate(
        [
            background_labels[0].ravel(),
            background_labels[-1].ravel(),
            background_labels[:, 0].ravel(),
            background_labels[:, -1].ravel(),
            background_labels[:, :, 0].ravel(),
            background_labels[:, :, -1].ravel(),
        ]
    )
    exterior = set(int(value) for value in np.unique(boundary_labels) if value > 0)
    cavities = sum(1 for component_id in range(1, background_count + 1) if component_id not in exterior)
    tunnels = max(0, int(components) + int(cavities) - euler)
    return {
        "euler_number": float(euler),
        "connected_components_3d": float(components),
        "tunnel_count_3d": float(tunnels),
        "cavity_count_3d": float(cavities),
        "hole_count": float(tunnels + cavities),
    }


def texture_features(volume: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute masked 3D GLCM features averaged across 13 undirected directions."""
    values = volume[mask].astype(np.float64)
    defaults = {
        "texture_contrast": 0.0,
        "texture_homogeneity": 1.0,
        "texture_correlation": 1.0,
        "texture_asm": 1.0,
        "texture_entropy": 0.0,
    }
    if values.size < 4 or float(values.max()) <= float(values.min()):
        return defaults

    levels = 8
    quantized = np.zeros(mask.shape, dtype=np.uint8)
    scaled = (volume.astype(np.float64) - values.min()) / (values.max() - values.min() + 1e-8)
    quantized[mask] = np.clip((scaled[mask] * levels).astype(np.uint8), 0, levels - 1)
    directions = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) > (0, 0, 0)
    ]
    measurements = []
    for offset in directions:
        source_slices = []
        target_slices = []
        for size, delta in zip(mask.shape, offset):
            if delta > 0:
                source_slices.append(slice(0, size - delta))
                target_slices.append(slice(delta, size))
            elif delta < 0:
                source_slices.append(slice(-delta, size))
                target_slices.append(slice(0, size + delta))
            else:
                source_slices.append(slice(None))
                target_slices.append(slice(None))
        source_slices = tuple(source_slices)
        target_slices = tuple(target_slices)
        valid = mask[source_slices] & mask[target_slices]
        if not valid.any():
            continue
        source = quantized[source_slices][valid]
        target = quantized[target_slices][valid]
        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (source, target), 1.0)
        matrix /= matrix.sum()

        i, j = np.indices(matrix.shape)
        px = matrix.sum(axis=1)
        py = matrix.sum(axis=0)
        bins = np.arange(levels)
        mean_x = float((bins * px).sum())
        mean_y = float((bins * py).sum())
        std_x = float(np.sqrt(((bins - mean_x) ** 2 * px).sum()))
        std_y = float(np.sqrt(((bins - mean_y) ** 2 * py).sum()))
        correlation = float(((i - mean_x) * (j - mean_y) * matrix).sum())
        if std_x > 0 and std_y > 0:
            correlation /= std_x * std_y
        measurements.append(
            [
                float(((i - j) ** 2 * matrix).sum()),
                float((matrix / (1.0 + (i - j) ** 2)).sum()),
                correlation,
                float((matrix**2).sum()),
                float(-(matrix[matrix > 0] * np.log2(matrix[matrix > 0])).sum()),
            ]
        )
    if not measurements:
        return defaults
    mean = np.asarray(measurements).mean(axis=0)
    return dict(
        zip(
            [
                "texture_contrast",
                "texture_homogeneity",
                "texture_correlation",
                "texture_asm",
                "texture_entropy",
            ],
            [float(value) for value in mean],
        )
    )


def skeleton_features(mask: np.ndarray, spacing: np.ndarray) -> dict[str, float]:
    skeleton = skeletonize(mask)
    coords = np.argwhere(skeleton)
    if len(coords) <= 1:
        return {
            "skeleton_voxels": float(len(coords)),
            "skeleton_length_voxels": float(len(coords)),
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

    if len(endpoints) >= 2:
        endpoint_coords = np.asarray(endpoints, dtype=np.float64)
        straight = 0.0
        for index in range(len(endpoint_coords)):
            distances = np.linalg.norm(endpoint_coords[index + 1 :] - endpoint_coords[index], axis=1)
            if len(distances):
                straight = max(straight, float(distances.max()))
    elif len(endpoints) == 1:
        start = np.asarray(endpoints[0], dtype=np.float64)
        straight = float(np.linalg.norm(coords.astype(np.float64) - start[None, :], axis=1).max())
    else:
        straight = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    # Historical v7 uses the number of skeleton voxels as skeleton length.
    skeleton_length_voxels = float(len(coords))
    tortuosity = (
        float(np.clip(1.0 - straight / skeleton_length_voxels, 0.0, 1.0))
        if skeleton_length_voxels > 0
        else 0.0
    )
    return {
        "skeleton_voxels": float(len(coords)),
        "skeleton_length_voxels": skeleton_length_voxels,
        "skeleton_length_um": length,
        "skeleton_endpoints": float(len(endpoints)),
        "skeleton_branchpoints": float(len(branchpoints)),
        "tortuosity": tortuosity,
    }


def intensity_features(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: np.ndarray,
    distances: np.ndarray,
) -> dict[str, float]:
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
    edge_mask = mask & ~binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool))
    edge_values = volume[edge_mask].astype(np.float64)
    if edge_values.size == 0:
        edge_values = values
    radius_values = distances[mask]
    inner_values = values[radius_values >= np.quantile(radius_values, 0.75)]
    outer_values = values[radius_values <= np.quantile(radius_values, 0.25)]
    inner_mean = float(inner_values.mean()) if inner_values.size else median
    outer_mean = float(outer_values.mean()) if outer_values.size else median
    return {
        "intensity_mean": float(values.mean()),
        "intensity_median": median,
        "intensity_std": float(values.std()),
        "intensity_mad": float(np.median(np.abs(values - median))),
        "intensity_p25": float(np.quantile(values, 0.25)),
        "intensity_p75": float(np.quantile(values, 0.75)),
        "integrated_intensity": float(values.sum()),
        "edge_intensity_mean": float(edge_values.mean()),
        "edge_intensity_std": float(edge_values.std()),
        "intensity_mass_displacement_um": displacement,
        "center_edge_intensity_ratio": float(inner_mean / outer_mean) if outer_mean != 0 else 0.0,
    }


def extract_instance(
    instance_mask: np.ndarray,
    volume: np.ndarray,
    normalization_voxels: int,
    spacing: np.ndarray,
    max_hull_points: int,
    include_intensity: bool,
    include_texture: bool,
) -> dict[str, Any]:
    coords = np.argwhere(instance_mask)
    volume_voxels = int(len(coords))
    physical_volume = float(volume_voxels * np.prod(spacing))
    coords_phys = physical_coordinates(coords, spacing)
    volume_fraction = (
        float(volume_voxels / normalization_voxels) if normalization_voxels > 0 else 0.0
    )
    historical_feature_eligible = (
        volume_voxels >= HISTORICAL_MIN_INSTANCE_VOXELS
        and HISTORICAL_MIN_VOLUME_FRACTION <= volume_fraction <= HISTORICAL_MAX_VOLUME_FRACTION
    )
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_voxels = bbox_max - bbox_min + 1
    roi_min = np.maximum(bbox_min - 1, 0)
    roi_max = np.minimum(bbox_max + 2, np.asarray(instance_mask.shape))
    roi_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(roi_min, roi_max))
    roi_mask = instance_mask[roi_slices]
    roi_volume = volume[roi_slices]
    roi_offset = roi_min.astype(np.float64)
    bbox_size_um = bbox_voxels.astype(float) * spacing
    bbox_aspect_ratios = [
        float(bbox_size_um[0] / bbox_size_um[1]),
        float(bbox_size_um[0] / bbox_size_um[2]),
        float(bbox_size_um[1] / bbox_size_um[2]),
    ]
    surface_um2, surface_faces = surface_area(roi_mask, spacing)
    hull_points_local = np.argwhere(
        roi_mask ^ binary_erosion(roi_mask, structure=np.ones((3, 3, 3), dtype=bool))
    )
    hull_points_phys = physical_coordinates(hull_points_local + roi_offset, spacing)
    hull_volume, hull_surface_um2 = convex_hull_measurements(hull_points_phys, max_hull_points)
    sphericity = (
        (math.pi ** (1.0 / 3.0)) * ((6.0 * volume_voxels) ** (2.0 / 3.0)) / surface_faces
        if surface_faces > 0 and volume_voxels > 0
        else 0.0
    )
    compactness_3d = (
        surface_um2 / ((36.0 * math.pi) ** (1.0 / 3.0) * physical_volume ** (2.0 / 3.0))
        if surface_um2 > 0 and physical_volume > 0
        else 0.0
    )
    convexity_3d = min(hull_surface_um2 / surface_um2, 1.0) if surface_um2 > 0 else 0.0
    equivalent_diameter = (
        (6.0 * physical_volume / math.pi) ** (1.0 / 3.0) if physical_volume > 0 else 0.0
    )
    radius, distances = radius_features(roi_mask, spacing)
    topology = topology_features(roi_mask)
    skeleton = skeleton_features(roi_mask, spacing)
    intensity = (
        intensity_features(roi_volume, roi_mask, spacing, distances) if include_intensity else {}
    )
    texture = texture_features(roi_volume, roi_mask) if include_texture else {}
    touches_border = bool(
        np.any(coords == 0)
        or any(np.any(coords[:, axis] == instance_mask.shape[axis] - 1) for axis in range(3))
    )
    border_coordinates = np.zeros(volume_voxels, dtype=bool)
    for axis in range(3):
        border_coordinates |= (coords[:, axis] == 0) | (
            coords[:, axis] == instance_mask.shape[axis] - 1
        )
    border_voxels = int(border_coordinates.sum())
    row: dict[str, Any] = {
        "volume_voxels": volume_voxels,
        "volume": volume_fraction,
        "volume_fraction": volume_fraction,
        "historical_feature_eligible": bool(historical_feature_eligible),
        "volume_um3": physical_volume,
        "centroid_zyx_voxels": coords.mean(axis=0).tolist(),
        "centroid_zyx_um": coords_phys.mean(axis=0).tolist(),
        "bbox_min_zyx_voxels": bbox_min.tolist(),
        "bbox_max_zyx_voxels": bbox_max.tolist(),
        "bbox_size_zyx_voxels": bbox_voxels.tolist(),
        "bbox_size_zyx_um": bbox_size_um.tolist(),
        "bbox_aspect_ratios_zyx": bbox_aspect_ratios,
        "bbox_volume_fraction": float(volume_voxels / np.prod(bbox_voxels)),
        "extent": float(volume_voxels / np.prod(bbox_voxels)),
        "equivalent_diameter_um": equivalent_diameter,
        "surface_area_um2": surface_um2,
        "surface_area_voxels": float(surface_faces),
        "surface_face_count": surface_faces,
        "convex_hull_volume_um3": hull_volume,
        "convex_hull_surface_area_um2": hull_surface_um2,
        "convexity_3d": convexity_3d,
        "solidity": historical_solidity(coords, volume_voxels),
        "sphericity": float(sphericity),
        "compactness_3d": compactness_3d,
        "border_voxel_fraction": float(border_voxels / volume_voxels) if volume_voxels > 0 else 0.0,
        "touches_volume_border": int(touches_border),
        "skeleton_length_over_volume_cuberoot": (
            float(skeleton["skeleton_length_voxels"] / np.cbrt(volume_voxels))
            if volume_voxels > 0
            else 0.0
        ),
    }
    row.update(inertia_features(coords, coords_phys))
    row.update(radius)
    row.update(topology)
    row.update(skeleton)
    row.update(intensity)
    row.update(texture)
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
    unique_labels, label_counts = np.unique(mask, return_counts=True)
    instance_counts = {
        int(label): int(count)
        for label, count in zip(unique_labels, label_counts)
        if int(label) > 0
    }
    label_ids = sorted(instance_counts)
    # Historical normalization first excludes labels smaller than 20 voxels.
    normalization_voxels = sum(
        count for count in instance_counts.values() if count >= HISTORICAL_MIN_INSTANCE_VOXELS
    )
    instances = []
    for instance_id in label_ids:
        instance_mask = mask == instance_id
        features = extract_instance(
            instance_mask,
            volume,
            normalization_voxels,
            spacing,
            int(args.max_hull_points),
            not args.no_intensity,
            bool(args.with_texture),
        )
        instances.append({"instance_id": instance_id, **features})
    return {
        "schema_version": "chromoworld.instance.v1",
        "morphology_profile": "historical_3d_semantic_v7",
        "additional_feature_profile": "3d_analogues_of_fig4_2d_v1",
        "texture_enabled": bool(args.with_texture),
        "intensity_enabled": not bool(args.no_intensity),
        "modality": "3d",
        "volume_basename": volume_path.stem,
        "volume_file": volume_path.name,
        "mask_file": mask_path.name,
        "volume_axes": volume_axes,
        "mask_axes": mask_axes,
        "shape_zyx": list(volume.shape),
        "voxel_size_zyx": spacing.tolist(),
        "voxel_volume": float(np.prod(spacing)),
        "total_mask_voxels": int(np.count_nonzero(mask)),
        "historical_normalization_voxels": normalization_voxels,
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
        "morphology_profile": "historical_3d_semantic_v7",
        "additional_feature_profile": "3d_analogues_of_fig4_2d_v1",
        "historical_instance_filter": {
            "minimum_voxels": HISTORICAL_MIN_INSTANCE_VOXELS,
            "minimum_volume_fraction": HISTORICAL_MIN_VOLUME_FRACTION,
            "maximum_volume_fraction": HISTORICAL_MAX_VOLUME_FRACTION,
            "behavior": "record all labels; mark historical_feature_eligible instead of dropping rows",
        },
        "output_jsonl": args.output_jsonl.name,
        "voxel_size_zyx": spacing.tolist(),
        "n_input_volumes": len(volumes),
        "n_output_volumes": len(records),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_instances": int(sum(record["n_instances"] for record in records)),
        "texture_enabled": bool(args.with_texture),
        "intensity_enabled": not bool(args.no_intensity),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
