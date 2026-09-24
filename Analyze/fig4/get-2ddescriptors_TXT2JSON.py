#!/usr/bin/env python3
"""Extract raw per-instance descriptors from polygon TXT files and images.

This stage performs no filtering or normalization. It pairs each polygon TXT
file with the processed image having the same basename, rasterizes the
polygons into instance-ID masks, and writes one structured JSONL record per
image and label set. A legacy flat CSV can be requested as an optional export.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


RAW_FEATURES = [
    "area_fraction",
    "norm_size",
    "area_px",
    "perimeter_px",
    "eccentricity",
    "solidity",
    "circularity",
    "compactness",
    "extent",
    "convexity",
    "euler_number",
    "hole_count",
    "major_axis_length_px",
    "minor_axis_length_px",
    "axis_ratio",
    "orientation_deg",
    "feret_min_px",
    "feret_max_px",
    "feret_ratio",
    "equivalent_diameter_px",
    "bbox_height_px",
    "bbox_width_px",
    "bbox_aspect_ratio",
    "radius_mean_px",
    "radius_median_px",
    "radius_std_px",
    "radius_cv",
    "radius_max_px",
    "skeleton_length_px",
    "skeleton_length_over_sqrt_area",
    "skeleton_endpoints",
    "skeleton_branchpoints",
    "tortuosity",
    "overlap_fraction",
    "border_pixel_fraction",
    "touches_image_border",
    "intensity_mean",
    "intensity_median",
    "intensity_std",
    "intensity_mad",
    "intensity_p25",
    "intensity_p75",
    "integrated_intensity",
    "edge_intensity_mean",
    "edge_intensity_std",
    "mass_displacement_px",
    "center_edge_intensity_ratio",
    "texture_contrast",
    "texture_homogeneity",
    "texture_correlation",
    "texture_asm",
    "texture_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--label-role",
        nargs="+",
        default=None,
        help="One role per label directory: gt or pred. Default: infer from the path.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional flat CSV export for legacy analysis scripts.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Raw instance-ID mask directory. Defaults beside output-jsonl.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Extraction report. Defaults beside output-jsonl.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--skip-image-features",
        action="store_true",
        help="Skip intensity descriptors while retaining mask descriptors.",
    )
    parser.add_argument(
        "--with-texture",
        action="store_true",
        help="Compute selected masked GLCM texture descriptors.",
    )
    return parser.parse_args()


def parse_polygon_line(line: str) -> tuple[int, float | None, np.ndarray] | None:
    """Parse GT ``class x y ...`` or prediction ``class conf x y ...``."""

    tokens = line.strip().split()
    if len(tokens) < 7:
        return None
    try:
        class_id = int(float(tokens[0]))
        tail = [float(token) for token in tokens[1:]]
    except ValueError:
        return None

    confidence: float | None = None
    if len(tail) % 2 == 1:
        confidence = float(tail[0])
        tail = tail[1:]
    if len(tail) < 6 or len(tail) % 2:
        return None
    points = np.asarray(tail, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(points).all():
        return None
    return class_id, confidence, points


def polygon_to_mask(points: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterize normalized polygon coordinates to a binary mask."""

    xy = np.rint(points * np.asarray([width - 1, height - 1], dtype=np.float32))
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [xy.astype(np.int32)], 1)
    return mask


def load_gray_image(path: Path) -> np.ndarray:
    """Load a processed image as a float32 grayscale array."""

    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def skeleton_descriptors(mask: np.ndarray) -> dict[str, float]:
    """Return skeleton length, topology counts, and tortuosity."""

    skeleton = skeletonize(mask > 0)
    skeleton_u8 = skeleton.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton_u8, cv2.CV_16U, kernel) - skeleton_u8
    coords = np.column_stack(np.where(skeleton))
    length = int(len(coords))
    endpoints = skeleton & (neighbors == 1)
    branchpoints = skeleton & (neighbors >= 3)
    endpoint_coords = np.column_stack(np.where(endpoints))

    if length <= 1:
        return {
            "skeleton_length_px": float(length),
            "skeleton_endpoints": float(endpoint_coords.shape[0]),
            "skeleton_branchpoints": float(branchpoints.sum()),
            "tortuosity": 0.0,
        }

    if endpoint_coords.shape[0] >= 2:
        straight = max(
            float(np.linalg.norm(first - second))
            for index, first in enumerate(endpoint_coords)
            for second in endpoint_coords[index + 1 :]
        )
    elif endpoint_coords.shape[0] == 1:
        straight = float(
            np.linalg.norm(coords - endpoint_coords[0], axis=1).max()
        )
    else:
        straight = float(np.linalg.norm(coords[0] - coords[-1]))
    tortuosity = float(np.clip(1.0 - straight / length, 0.0, 1.0))
    return {
        "skeleton_length_px": float(length),
        "skeleton_endpoints": float(endpoint_coords.shape[0]),
        "skeleton_branchpoints": float(branchpoints.sum()),
        "tortuosity": tortuosity,
    }


def radius_descriptors(mask: np.ndarray) -> dict[str, float]:
    """Measure mean, spread, and maximum local half-width."""

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    values = distance[mask > 0].astype(np.float64)
    if values.size == 0:
        return {
            "radius_mean_px": 0.0,
            "radius_median_px": 0.0,
            "radius_std_px": 0.0,
            "radius_cv": 0.0,
            "radius_max_px": 0.0,
        }
    mean = float(values.mean())
    std = float(values.std())
    return {
        "radius_mean_px": mean,
        "radius_median_px": float(np.median(values)),
        "radius_std_px": std,
        "radius_cv": float(std / mean) if mean > 0 else 0.0,
        "radius_max_px": float(values.max()),
    }


def contour_descriptors(mask: np.ndarray, area: float, perimeter: float) -> dict[str, float]:
    """Measure convexity and an efficient min-area-rectangle Feret proxy."""

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return {
            "convexity": 0.0,
            "feret_min_px": 0.0,
            "feret_max_px": 0.0,
            "feret_ratio": 0.0,
        }
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_perimeter = float(cv2.arcLength(hull, True))
    _, (width, height), _ = cv2.minAreaRect(hull)
    feret_min = float(min(width, height))
    feret_max = float(max(width, height))
    return {
        "convexity": min(float(hull_perimeter / perimeter), 1.0) if perimeter > 0 else 0.0,
        "feret_min_px": feret_min,
        "feret_max_px": feret_max,
        "feret_ratio": float(feret_min / feret_max) if feret_max > 0 else 0.0,
    }


def intensity_descriptors(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Measure robust intensity and radial distribution descriptors."""

    values = gray[mask > 0].astype(np.float64)
    if values.size == 0:
        return {name: 0.0 for name in RAW_FEATURES if name.startswith(("intensity", "integrated", "edge_", "mass_", "center_"))}

    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    edge_mask = (mask > 0) & (eroded == 0)
    edge_values = gray[edge_mask].astype(np.float64)
    if edge_values.size == 0:
        edge_values = values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radius = distance[mask > 0]
    inner = values[radius >= np.quantile(radius, 0.75)]
    outer = values[radius <= np.quantile(radius, 0.25)]
    inner_mean = float(inner.mean()) if inner.size else median
    outer_mean = float(outer.mean()) if outer.size else median
    coords = np.column_stack(np.where(mask > 0)).astype(np.float64)
    weights = values - values.min()
    weight_sum = float(weights.sum())
    if weight_sum > 0:
        weighted_center = (coords * weights[:, None]).sum(axis=0) / weight_sum
        region_center = coords.mean(axis=0)
        mass_displacement = float(np.linalg.norm(weighted_center - region_center))
    else:
        mass_displacement = 0.0
    return {
        "intensity_mean": float(values.mean()),
        "intensity_median": median,
        "intensity_std": float(values.std()),
        "intensity_mad": mad,
        "intensity_p25": float(np.quantile(values, 0.25)),
        "intensity_p75": float(np.quantile(values, 0.75)),
        "integrated_intensity": float(values.sum()),
        "edge_intensity_mean": float(edge_values.mean()),
        "edge_intensity_std": float(edge_values.std()),
        "mass_displacement_px": mass_displacement,
        "center_edge_intensity_ratio": float(inner_mean / outer_mean) if outer_mean != 0 else 0.0,
    }


def texture_descriptors(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute selected masked GLCM descriptors averaged over four directions."""

    values = gray[mask > 0].astype(np.float64)
    if values.size < 4 or float(values.max()) <= float(values.min()):
        return {
            "texture_contrast": 0.0,
            "texture_homogeneity": 1.0,
            "texture_correlation": 1.0,
            "texture_asm": 1.0,
            "texture_entropy": 0.0,
        }
    levels = 8
    quantized = np.zeros_like(gray, dtype=np.uint8)
    scaled = (gray - values.min()) / (values.max() - values.min() + 1e-8)
    quantized[mask > 0] = np.clip((scaled[mask > 0] * levels).astype(np.uint8), 0, levels - 1)
    matrices = []
    for dr, dc in [(0, 1), (1, 1), (1, 0), (1, -1)]:
        height, width = mask.shape
        row_start = max(0, -dr)
        row_stop = min(height, height - dr)
        col_start = max(0, -dc)
        col_stop = min(width, width - dc)
        source = mask[row_start:row_stop, col_start:col_stop] > 0
        target = mask[row_start + dr : row_stop + dr, col_start + dc : col_stop + dc] > 0
        valid = source & target
        if not valid.any():
            continue
        local_r, local_c = np.where(valid)
        r = local_r + row_start
        c = local_c + col_start
        r2 = r + dr
        c2 = c + dc
        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (quantized[r, c], quantized[r2, c2]), 1.0)
        if matrix.sum() > 0:
            matrices.append(matrix / matrix.sum())
    if not matrices:
        return {
            "texture_contrast": 0.0,
            "texture_homogeneity": 1.0,
            "texture_correlation": 1.0,
            "texture_asm": 1.0,
            "texture_entropy": 0.0,
        }

    measurements = []
    for matrix in matrices:
        i, j = np.indices(matrix.shape)
        px = matrix.sum(axis=1)
        py = matrix.sum(axis=0)
        mux = float((np.arange(levels) * px).sum())
        muy = float((np.arange(levels) * py).sum())
        sigx = float(np.sqrt(((np.arange(levels) - mux) ** 2 * px).sum()))
        sigy = float(np.sqrt(((np.arange(levels) - muy) ** 2 * py).sum()))
        correlation = float(((i - mux) * (j - muy) * matrix).sum())
        if sigx > 0 and sigy > 0:
            correlation /= sigx * sigy
        measurements.append(
            [
                float(((i - j) ** 2 * matrix).sum()),
                float((matrix / (1.0 + (i - j) ** 2)).sum()),
                correlation,
                float((matrix**2).sum()),
                float(-(matrix[matrix > 0] * np.log2(matrix[matrix > 0])).sum()),
            ]
        )
    mean = np.asarray(measurements).mean(axis=0)
    return dict(zip(
        ["texture_contrast", "texture_homogeneity", "texture_correlation", "texture_asm", "texture_entropy"],
        mean.astype(float),
    ))


def infer_role(path: Path) -> str:
    """Infer a label role from common directory names."""

    text = "/".join(part.lower() for part in path.parts)
    if "pred" in text or "prediction" in text or "chromopix" in text:
        return "pred"
    if "gt" in text or "ground" in text or "label" in text:
        return "gt"
    return "unknown"


def resolve_roles(label_dirs: list[Path], roles: list[str] | None) -> list[str]:
    if roles is None:
        return [infer_role(path) for path in label_dirs]
    if len(roles) == 1 and len(label_dirs) > 1:
        return roles * len(label_dirs)
    if len(roles) != len(label_dirs):
        raise ValueError("--label-role must have one value per --label-dir")
    return [role.lower() for role in roles]


def find_images(image_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            mapping.setdefault(path.stem, path)
    return mapping


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable raw descriptor table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_basename",
        "image_relpath",
        "label_relpath",
        "label_role",
        "label_set_id",
        "class_id",
        "confidence",
        "instance_id",
        "mask_file",
    ] + RAW_FEATURES
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one structured JSON record per image and label set."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("label_set_id", "")),
            str(row.get("image_basename", "")),
            str(row.get("label_role", "")),
        )
        grouped.setdefault(key, []).append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (label_set_id, image_basename, label_role), group in grouped.items():
            first = group[0]
            instances = []
            for row in group:
                instances.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "image_basename",
                            "image_relpath",
                            "label_relpath",
                            "label_role",
                            "label_set_id",
                        }
                    }
                )
            record = {
                "schema_version": "chromoworld.instance.v1",
                "modality": "2d",
                "image_basename": image_basename,
                "image_file": first.get("image_relpath", ""),
                "label_file": first.get("label_relpath", ""),
                "label_role": label_role,
                "label_set_id": label_set_id,
                "n_instances": len(instances),
                "instances": instances,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_row(
    mask: np.ndarray,
    gray: np.ndarray,
    union_area: int,
    coverage_count: np.ndarray,
    skip_image_features: bool,
    with_texture: bool,
) -> dict[str, Any]:
    """Calculate one raw row from one instance mask."""

    props = regionprops(label(mask > 0))
    if not props:
        raise ValueError("empty mask")
    region = props[0]
    area = float(region.area)
    perimeter = float(region.perimeter)
    circularity = 4.0 * math.pi * area / (perimeter**2) if perimeter > 0 else 0.0
    compactness = (perimeter**2) / (4.0 * math.pi * area) if area > 0 else 0.0
    major = float(region.major_axis_length)
    minor = float(region.minor_axis_length)
    min_row, min_col, max_row, max_col = region.bbox
    bbox_height = float(max_row - min_row)
    bbox_width = float(max_col - min_col)
    contour = contour_descriptors(mask, area, perimeter)
    radius = radius_descriptors(mask)
    skeleton = skeleton_descriptors(mask)
    intensity = {} if skip_image_features else intensity_descriptors(gray, mask)
    texture = texture_descriptors(gray, mask) if with_texture else {}
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_pixels = int((mask.astype(bool) & border).sum())
    overlap_pixels = int(((coverage_count > 1) & (mask > 0)).sum())
    row: dict[str, Any] = {
        "area_fraction": float(area / union_area) if union_area > 0 else 0.0,
        "norm_size": float(area / union_area) if union_area > 0 else 0.0,
        "area_px": area,
        "perimeter_px": perimeter,
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "circularity": float(circularity),
        "compactness": float(compactness),
        "extent": float(region.extent),
        "convexity": contour["convexity"],
        "euler_number": float(region.euler_number),
        "hole_count": float(max(0, 1 - region.euler_number)),
        "major_axis_length_px": major,
        "minor_axis_length_px": minor,
        "axis_ratio": float(minor / major) if major > 0 else 0.0,
        "orientation_deg": float(math.degrees(region.orientation)),
        "feret_min_px": contour["feret_min_px"],
        "feret_max_px": contour["feret_max_px"],
        "feret_ratio": contour["feret_ratio"],
        "equivalent_diameter_px": float(region.equivalent_diameter),
        "bbox_height_px": bbox_height,
        "bbox_width_px": bbox_width,
        "bbox_aspect_ratio": float(bbox_width / bbox_height) if bbox_height > 0 else 0.0,
        **radius,
        **skeleton,
        "skeleton_length_over_sqrt_area": float(
            skeleton["skeleton_length_px"] / math.sqrt(area)
        ) if area > 0 else 0.0,
        "overlap_fraction": float(overlap_pixels / area) if area > 0 else 0.0,
        "border_pixel_fraction": float(border_pixels / area) if area > 0 else 0.0,
        "touches_image_border": int(bool(border_pixels)),
        **intensity,
        **texture,
    }
    return {name: row.get(name, "") for name in RAW_FEATURES}


def main() -> None:
    args = parse_args()
    roles = resolve_roles(args.label_dir, args.label_role)
    image_map = find_images(args.image_dir)
    output_jsonl = args.output_jsonl
    mask_dir = args.mask_dir or output_jsonl.parent / "masks_raw"
    report_json = args.report_json or output_jsonl.with_name("descriptor_report.json")
    mask_dir.mkdir(parents=True, exist_ok=True)

    label_files: list[tuple[int, Path, str]] = []
    for set_index, (label_dir, role) in enumerate(zip(args.label_dir, roles)):
        candidates = [label_dir] if label_dir.is_file() else sorted(label_dir.rglob("*.txt"))
        label_files.extend((set_index, path, role) for path in candidates if path.is_file())
    if args.limit > 0:
        label_files = label_files[: args.limit]

    rows: list[dict[str, Any]] = []
    skipped = Counter[str]()
    malformed_lines = 0
    image_count = 0

    for set_index, label_path, role in label_files:
        stem = label_path.stem
        image_path = image_map.get(stem)
        if image_path is None:
            skipped["missing_image"] += 1
            continue
        image_count += 1
        gray = load_gray_image(image_path)
        height, width = gray.shape
        parsed: list[tuple[int, float | None, np.ndarray]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = parse_polygon_line(line)
            if item is None:
                malformed_lines += 1
            else:
                parsed.append(item)
        masks = [polygon_to_mask(points, height, width) for _, _, points in parsed]
        union = np.zeros((height, width), dtype=np.uint8)
        coverage_count = np.zeros((height, width), dtype=np.uint16)
        for mask in masks:
            union = np.maximum(union, mask)
            coverage_count += mask.astype(np.uint16)
        union_area = int(union.sum())
        label_set_id = f"{role}_{set_index}"
        mask_name = f"{label_set_id}__{stem}_instance_id.png"
        mask_path = mask_dir / mask_name
        instance_map = np.zeros((height, width), dtype=np.uint16)
        for index, mask in enumerate(masks, start=1):
            instance_map[mask > 0] = index
            try:
                descriptors = compute_row(
                    mask,
                    gray,
                    union_area,
                    coverage_count,
                    args.skip_image_features,
                    args.with_texture,
                )
            except ValueError:
                skipped["empty_instance"] += 1
                continue
            image_relpath = str(image_path.relative_to(args.image_dir))
            label_relpath = str(label_path.relative_to(args.label_dir[set_index])) if args.label_dir[set_index].is_dir() else label_path.name
            row: dict[str, Any] = {
                "image_basename": stem,
                "image_relpath": image_relpath,
                "label_relpath": label_relpath,
                "label_role": role,
                "label_set_id": label_set_id,
                "class_id": int(parsed[index - 1][0]),
                "confidence": "" if parsed[index - 1][1] is None else parsed[index - 1][1],
                "instance_id": index,
                "mask_file": str(Path("masks_raw") / mask_name),
                **descriptors,
            }
            rows.append(row)
        cv2.imwrite(str(mask_path), instance_map)

    write_jsonl(output_jsonl, rows)
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    report = {
        "output_jsonl": str(output_jsonl.name),
        "output_csv": str(args.output_csv.name) if args.output_csv else None,
        "n_label_files": len(label_files),
        "n_images": image_count,
        "n_instances": len(rows),
        "malformed_lines": malformed_lines,
        "skipped": dict(skipped),
        "features": RAW_FEATURES,
        "image_features_enabled": not args.skip_image_features,
        "texture_enabled": args.with_texture,
        "schema_version": "chromoworld.instance.v1",
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
