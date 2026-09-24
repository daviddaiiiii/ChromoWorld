#!/usr/bin/env python3
"""Filter and robustly normalize the raw Fig4 descriptor JSONL.

This stage never rereads polygon TXT files. It adds auditable QC flags to an
audit table, writes only accepted instances to the final JSONL, and preserves
raw features while adding image-relative percentiles and robust z-scores. A
flat CSV export remains optional for legacy plotting code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NORMALIZE_FEATURES = [
    "area_fraction",
    "eccentricity",
    "solidity",
    "circularity",
    "compactness",
    "extent",
    "convexity",
    "axis_ratio",
    "feret_ratio",
    "radius_mean_px",
    "radius_cv",
    "skeleton_length_over_sqrt_area",
    "tortuosity",
    "intensity_mean",
    "intensity_std",
    "intensity_mad",
    "center_edge_intensity_ratio",
    "texture_contrast",
    "texture_homogeneity",
    "texture_correlation",
    "texture_asm",
    "texture_entropy",
]

HARD_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "area_fraction": (0.001, 0.20),
    "eccentricity": (0.0, 1.0),
    "solidity": (0.0, 1.0),
    "circularity": (0.0, 1.0),
    "extent": (0.0, 1.0),
    "convexity": (0.0, 1.0),
    "axis_ratio": (0.0, 1.0),
    "feret_ratio": (0.0, 1.0),
    "radius_cv": (0.0, None),
    "tortuosity": (0.0, 1.0),
    "overlap_fraction": (0.0, 1.0),
    "border_pixel_fraction": (0.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--min-area-px", type=float, default=20.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.20)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--max-overlap-fraction", type=float, default=1.0)
    parser.add_argument("--reject-border-touch", action="store_true")
    parser.add_argument("--min-instances-per-image", type=int, default=0)
    parser.add_argument("--max-instances-per-image", type=int, default=0)
    parser.add_argument(
        "--mad-z-threshold",
        type=float,
        default=6.0,
        help="Robust global outlier threshold; zero disables MAD filtering.",
    )
    parser.add_argument(
        "--outlier-features",
        default="area_fraction,eccentricity,solidity,circularity,axis_ratio,tortuosity",
        help="Comma-separated raw features used for MAD outlier filtering.",
    )
    parser.add_argument(
        "--normalization-group",
        choices=("image", "label_role", "all"),
        default="image",
        help="Group used for relative percentile and robust z-score columns.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = {
                "image_basename": record.get("image_basename", ""),
                "image_relpath": record.get("image_file", ""),
                "label_relpath": record.get("label_file", ""),
                "label_role": record.get("label_role", "unknown"),
                "label_set_id": record.get("label_set_id", ""),
            }
            for instance in record.get("instances", []):
                rows.append({**metadata, **instance})
    return rows


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def robust_location_scale(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 0.0, 1.0
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12 and array.size >= 4:
        q25, q75 = np.quantile(array, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if scale <= 1e-12:
        scale = 1.0
    return median, scale


def group_key(row: dict[str, Any], mode: str) -> str:
    if mode == "image":
        return f"{row.get('label_role', '')}::{row.get('image_basename', '')}"
    if mode == "label_role":
        return str(row.get("label_role", "unknown"))
    return "all"


def add_relative_features(
    rows: list[dict[str, Any]],
    features: list[str],
    group_mode: str,
) -> dict[str, dict[str, float]]:
    """Add within-group percentiles and robust z-scores without overwriting raw data."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row, group_mode)].append(row)
    stats: dict[str, dict[str, float]] = {}
    for key, group in grouped.items():
        for feature in features:
            values = [numeric(row.get(feature)) for row in group]
            valid = [value for value in values if value is not None]
            if not valid:
                continue
            median, scale = robust_location_scale(valid)
            ordered = np.sort(np.asarray(valid, dtype=np.float64))
            stats[f"{key}::{feature}"] = {
                "median": median,
                "scale": scale,
                "n": float(len(valid)),
            }
            for row, value in zip(group, values):
                if value is None:
                    row[f"{feature}__rel_pct"] = ""
                    row[f"{feature}__robust_z"] = ""
                    continue
                rank = float(np.searchsorted(ordered, value, side="right") / len(ordered))
                row[f"{feature}__rel_pct"] = rank
                row[f"{feature}__robust_z"] = float((value - median) / scale)
    return stats


def apply_hard_qc(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    area_px = numeric(row.get("area_px"))
    if area_px is None or area_px < args.min_area_px:
        reasons.append("small_or_missing_area_px")
    area_fraction = numeric(row.get("area_fraction"))
    if area_fraction is None or area_fraction < args.min_area_ratio:
        reasons.append("small_or_missing_area_fraction")
    if area_fraction is not None and area_fraction > args.max_area_ratio:
        reasons.append("large_area_fraction")
    for feature, (lower, upper) in HARD_BOUNDS.items():
        value = numeric(row.get(feature))
        if value is None:
            reasons.append(f"missing_{feature}")
            continue
        if lower is not None and value < lower:
            reasons.append(f"low_{feature}")
        if upper is not None and value > upper:
            reasons.append(f"high_{feature}")
    confidence = numeric(row.get("confidence"))
    if args.min_confidence is not None and (
        confidence is None or confidence < args.min_confidence
    ):
        reasons.append("low_or_missing_confidence")
    overlap = numeric(row.get("overlap_fraction"))
    if overlap is not None and overlap > args.max_overlap_fraction:
        reasons.append("excessive_overlap")
    if args.reject_border_touch and int(float(row.get("touches_image_border") or 0)):
        reasons.append("touches_image_border")
    return reasons


def apply_mad_qc(
    rows: list[dict[str, Any]],
    features: list[str],
    threshold: float,
    group_mode: str,
) -> None:
    if threshold <= 0:
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["qc_reasons"] == "":
            grouped[group_key(row, group_mode)].append(row)
    for group in grouped.values():
        for feature in features:
            values = [numeric(row.get(feature)) for row in group]
            valid = [value for value in values if value is not None]
            if len(valid) < 5:
                continue
            median, scale = robust_location_scale(valid)
            for row, value in zip(group, values):
                if value is None:
                    continue
                robust_z = abs((value - median) / scale)
                if robust_z > threshold:
                    reason = f"mad_outlier_{feature}"
                    if row["qc_reasons"] == "":
                        row["qc_reasons"] = reason
                    elif reason not in row["qc_reasons"].split(";"):
                        row["qc_reasons"] += ";" + reason


def add_image_qc(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, bool]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row.get('label_role', '')}::{row.get('image_basename', '')}"].append(row)
    image_pass: dict[str, bool] = {}
    for key, group in grouped.items():
        count = sum(row["qc_reasons"] == "" for row in group)
        reasons = []
        if args.min_instances_per_image and count < args.min_instances_per_image:
            reasons.append("too_few_instances")
        if args.max_instances_per_image and count > args.max_instances_per_image:
            reasons.append("too_many_instances")
        passed = not reasons
        image_pass[key] = passed
        for row in group:
            row["image_n_qc_pass"] = count
            row["image_qc_pass"] = int(passed)
            if reasons:
                if row["qc_reasons"]:
                    row["qc_reasons"] += ";" + ";".join(reasons)
                else:
                    row["qc_reasons"] = ";".join(reasons)
    return image_pass


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write QC rows using the same per-image JSONL schema as extraction."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("label_set_id", "")),
            str(row.get("image_basename", "")),
            str(row.get("label_role", "")),
        )
        grouped[key].append(row)
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


def make_plots(plot_dir: Path, all_rows: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    features = ["area_fraction", "eccentricity", "solidity", "circularity", "tortuosity"]
    fig, axes = plt.subplots(1, len(features), figsize=(16, 3.5), constrained_layout=True)
    for axis, feature in zip(axes, features):
        before = [numeric(row.get(feature)) for row in all_rows]
        after = [numeric(row.get(feature)) for row in accepted]
        before = [value for value in before if value is not None]
        after = [value for value in after if value is not None]
        axis.hist(before, bins=30, alpha=0.35, label="raw")
        axis.hist(after, bins=30, alpha=0.65, label="QC-pass")
        axis.set_title(feature)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(plot_dir / "core_feature_qc.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.scatter(
        [numeric(row.get("area_fraction")) for row in all_rows],
        [numeric(row.get("circularity")) for row in all_rows],
        s=6,
        alpha=0.2,
        label="raw",
    )
    axis.scatter(
        [numeric(row.get("area_fraction")) for row in accepted],
        [numeric(row.get("circularity")) for row in accepted],
        s=8,
        alpha=0.65,
        label="QC-pass",
    )
    axis.set_xlabel("area fraction")
    axis.set_ylabel("circularity")
    axis.legend(frameon=False)
    fig.savefig(plot_dir / "area_vs_circularity_qc.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    if not rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")
    for row in rows:
        row["qc_reasons"] = ";".join(apply_hard_qc(row, args))
        row["qc_pass_instance"] = int(row["qc_reasons"] == "")

    outlier_features = [
        feature.strip()
        for feature in args.outlier_features.split(",")
        if feature.strip()
    ]
    apply_mad_qc(rows, outlier_features, args.mad_z_threshold, args.normalization_group)
    add_image_qc(rows, args)
    for row in rows:
        row["qc_pass"] = int(row["qc_reasons"] == "")

    accepted = [row for row in rows if row["qc_pass"]]
    stats = add_relative_features(accepted, NORMALIZE_FEATURES, args.normalization_group)
    write_jsonl(args.output_jsonl, accepted)
    audit_jsonl = args.audit_jsonl or args.output_jsonl.with_name(
        args.output_jsonl.stem + "_audit.jsonl"
    )
    write_jsonl(audit_jsonl, rows)
    if args.output_csv is not None:
        write_csv(args.output_csv, accepted)
    if args.audit_csv is not None:
        write_csv(args.audit_csv, rows)
    if not args.no_plots:
        make_plots(args.plot_dir or args.output_jsonl.parent / "qc_plots", rows, accepted)

    image_groups = {
        f"{row.get('label_role', '')}::{row.get('image_basename', '')}"
        for row in rows
    }
    report = {
        "input_jsonl": str(args.input_jsonl.name),
        "output_jsonl": str(args.output_jsonl.name),
        "audit_jsonl": str(audit_jsonl.name),
        "output_csv": str(args.output_csv.name) if args.output_csv else None,
        "audit_csv": str(args.audit_csv.name) if args.audit_csv else None,
        "n_input_instances": len(rows),
        "n_output_instances": len(accepted),
        "n_rejected_instances": len(rows) - len(accepted),
        "n_images": len(image_groups),
        "mad_z_threshold": args.mad_z_threshold,
        "outlier_features": outlier_features,
        "normalization_group": args.normalization_group,
        "normalization_statistics": stats,
        "rejection_reasons": dict(
            Counter(
                reason
                for row in rows
                if row["qc_reasons"]
                for reason in row["qc_reasons"].split(";")
            )
        ),
    }
    report_json = args.report_json or args.output_jsonl.with_name("qc_report.json")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "normalization_statistics"}, indent=2))


if __name__ == "__main__":
    main()
