#!/usr/bin/env python3
"""Create and promote mask-guided top-six Tag candidates per time point.

Raw Tag TIFFs are never modified. ``--prepare`` creates verified candidates
under ``image/tag_processing_audit/top6``; ``--apply`` atomically promotes them
to the existing raw_pre and raw_pre_affine TIFF trees.

The six highest mask-supported local high-pass maxima are selected per frame.
Candidates below the estimated background-peak threshold are still retained
to meet the requested per-frame top-six rule and are flagged in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

import affine as affine_ops


VERSION = "tag-mask-guided-top6-v1.0"
MAX_POINTS = 6
SUPPORT_SIZE = (3, 9, 9)
PEAK_SIZE = (3, 7, 7)
SMOOTH_SIGMA = (1.1, 1.3, 1.3)
BACKGROUND_SIGMA = (3.5, 9.0, 9.0)
SPOT_WINDOW_RADIUS = (5, 12, 12)


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metadata_from_tiff(tif: tifffile.TiffFile) -> dict:
    metadata = dict(tif.imagej_metadata or {})
    if metadata:
        return metadata
    try:
        return json.loads(tif.pages[0].description or "{}")
    except (ValueError, TypeError):
        return {}


def read_frame(tif: tifffile.TiffFile, t: int) -> np.ndarray:
    return affine_ops.read_frame(tif, t)


def robust_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    return max(1.0, float(1.4826 * np.median(np.abs(values - median))))


def select_top6_frame(
    tag: np.ndarray, chromosome_mask: np.ndarray
) -> tuple[np.ndarray, list[dict], dict]:
    """Retain the raw-intensity connected blobs around the top six peaks."""
    if tag.shape != chromosome_mask.shape:
        raise ValueError(f"Tag/mask frame shape mismatch: {tag.shape} vs {chromosome_mask.shape}")

    x = tag.astype(np.float32, copy=False)
    mask = chromosome_mask > 0
    support = ndi.maximum_filter(mask.astype(np.uint8), size=SUPPORT_SIZE) > 0
    outside = ~support

    smooth = ndi.gaussian_filter(x, sigma=SMOOTH_SIGMA)
    highpass = smooth - ndi.gaussian_filter(smooth, sigma=BACKGROUND_SIGMA)
    local_max = ndi.maximum_filter(
        highpass, size=PEAK_SIZE, mode="constant", cval=-np.inf
    )

    outside_peaks = (highpass == local_max) & outside
    outside_scores = highpass[outside_peaks]
    if outside_scores.size < 32:
        outside_scores = highpass[outside]
    median = float(np.median(outside_scores)) if outside_scores.size else 0.0
    sigma = robust_sigma(outside_scores)
    threshold = max(
        float(np.percentile(outside_scores, 99.5)) if outside_scores.size else 0.0,
        median + 5.0 * sigma,
    )

    # Use non-maximum suppression so one broad plateau cannot occupy all six slots.
    candidate_mask = (highpass == local_max) & support
    coords = np.argwhere(candidate_mask)
    values = highpass[tuple(coords.T)] if coords.size else np.empty(0, dtype=np.float32)
    candidates = [
        (float(value), tuple(int(v) for v in coord))
        for coord, value in zip(coords, values)
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))

    selected: list[tuple[float, tuple[int, int, int]]] = []
    rz, ry, rx = (size // 2 for size in PEAK_SIZE)
    for score, coord in candidates:
        if any(
            abs(coord[0] - chosen[1][0]) <= rz
            and abs(coord[1] - chosen[1][1]) <= ry
            and abs(coord[2] - chosen[1][2]) <= rx
            for chosen in selected
        ):
            continue
        selected.append((score, coord))
        if len(selected) == MAX_POINTS:
            break

    output = np.zeros(tag.shape, dtype=np.uint16)
    spots = []
    connectivity = ndi.generate_binary_structure(3, 1)
    z_size, y_size, x_size = tag.shape
    for score, (z, y, x0) in selected:
        above_threshold = score >= threshold
        cutoff = (
            max(threshold, 0.20 * score)
            if above_threshold
            else max(0.0, min(threshold, 0.20 * score))
        )
        z0, z1 = max(0, z - SPOT_WINDOW_RADIUS[0]), min(z_size, z + SPOT_WINDOW_RADIUS[0] + 1)
        y0, y1 = max(0, y - SPOT_WINDOW_RADIUS[1]), min(y_size, y + SPOT_WINDOW_RADIUS[1] + 1)
        x1, x2 = max(0, x0 - SPOT_WINDOW_RADIUS[2]), min(x_size, x0 + SPOT_WINDOW_RADIUS[2] + 1)
        local_slice = (slice(z0, z1), slice(y0, y1), slice(x1, x2))
        signal = (highpass[local_slice] >= cutoff) & support[local_slice]
        components, _ = ndi.label(signal, structure=connectivity)
        peak_local = (z - z0, y - y0, x0 - x1)
        component_id = int(components[peak_local])
        blob = components == component_id if component_id else signal
        if not np.any(blob):
            blob[peak_local] = True
        target = output[local_slice]
        raw_values = np.rint(np.clip(x[local_slice], 0, 65535)).astype(np.uint16)
        target[blob] = np.maximum(target[blob], raw_values[blob])
        spots.append(
            {
                "zyx": [z, y, x0],
                "highpass_score": score,
                "raw_intensity": int(tag[z, y, x0]),
                "inside_original_mask": bool(mask[z, y, x0]),
                "above_background_threshold": bool(above_threshold),
                "kept_voxels": int(blob.sum()),
            }
        )

    return output, spots, {
        "threshold": threshold,
        "background_median": median,
        "background_robust_sigma": sigma,
        "local_peak_candidates": len(candidates),
        "candidate_count": len(spots),
        "foreground_voxels": int(mask.sum()),
        "support_voxels": int(support.sum()),
        "output_voxels": int(np.count_nonzero(output)),
        "above_threshold_count": int(sum(s["above_background_threshold"] for s in spots)),
        "below_threshold_count": int(sum(not s["above_background_threshold"] for s in spots)),
    }


def write_pre_candidate(
    raw_tag: Path, mask_path: Path, output: Path
) -> dict:
    with tifffile.TiffFile(raw_tag) as tag_file, tifffile.TiffFile(mask_path) as mask_file:
        tag_series, mask_series = tag_file.series[0], mask_file.series[0]
        if tag_series.axes != "TZYX" or mask_series.axes != "TZYX":
            raise ValueError(f"Expected TZYX for {raw_tag.name}: {tag_series.axes}, {mask_series.axes}")
        if tag_series.shape != mask_series.shape:
            raise ValueError(f"Tag/mask shape mismatch for {raw_tag.name}: {tag_series.shape}, {mask_series.shape}")
        shape = tuple(tag_series.shape)
        input_metadata = metadata_from_tiff(tag_file)
        spacing, _ = affine_ops.get_metadata(tag_file)

        metadata = {
            key: input_metadata[key]
            for key in ("unit", "spacing", "finterval", "fps")
            if key in input_metadata
        }
        metadata.update(
            axes="TZYX",
            tag_pipeline=VERSION,
            tag_selection="top six mask-supported local high-pass maxima per frame",
            source_tag=str(raw_tag),
            source_mask=str(mask_path),
            candidate_peak_threshold="outside-support p99.5 or median plus 5 robust sigma; below-threshold top-six are retained and flagged",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        if output.exists() or temporary.exists():
            raise FileExistsError(output if output.exists() else temporary)

        frames = []
        frame_hashes = []

        def planes():
            for t in range(shape[0]):
                selected, spots, stats = select_top6_frame(
                    read_frame(tag_file, t), read_frame(mask_file, t)
                )
                frames.append({"t": t, "spots": spots, "stats": stats})
                frame_hashes.append(hashlib.sha256(selected.tobytes()).hexdigest())
                yield from selected

        tifffile.imwrite(
            temporary,
            planes(),
            shape=shape,
            dtype=np.uint16,
            metadata=metadata,
            resolution=(1.0 / spacing[2], 1.0 / spacing[1]),
            photometric="minisblack",
            compression="zlib",
            bigtiff=True,
        )

    with tifffile.TiffFile(temporary) as check:
        if check.series[0].shape != shape or check.series[0].axes != "TZYX":
            raise AssertionError(f"Pre Tag output shape/axes mismatch: {check.series[0].shape}, {check.series[0].axes}")
        for t, expected in enumerate(frame_hashes):
            if hashlib.sha256(read_frame(check, t).tobytes()).hexdigest() != expected:
                raise AssertionError(f"Pre Tag read-back hash mismatch at t={t}: {raw_tag.name}")
    os.replace(temporary, output)
    return {
        "shape_tzyx": list(shape),
        "frames": frames,
        "file": fingerprint(output),
        "sha256": digest(output),
    }


def write_affine_candidate(
    pre_candidate: Path,
    affine_dapi: Path,
    transform: dict,
    output: Path,
) -> dict:
    spacing = np.asarray(transform["spacing_zyx_um"], dtype=float)
    origin = np.asarray(transform["full_origin_zyx_um"], dtype=float)
    full_shape = tuple(int(v) for v in transform["full_shape_zyx"])
    crop_start = np.asarray(transform["crop_start_zyx"], dtype=int)
    matrices = [np.asarray(v, dtype=float) for v in transform["frame_to_anchor_physical_zyx"]]

    with tifffile.TiffFile(pre_candidate) as pre_file, tifffile.TiffFile(affine_dapi) as dapi_file:
        pre_shape = tuple(pre_file.series[0].shape)
        affine_shape = tuple(dapi_file.series[0].shape)
        if len(pre_shape) != 4 or affine_shape[0] != pre_shape[0] or len(matrices) != pre_shape[0]:
            raise ValueError(f"Affine frame count mismatch: {pre_shape}, {affine_shape}")
        output_shape = tuple(int(v) for v in affine_shape[1:])
        metadata = metadata_from_tiff(dapi_file)
        metadata = {
            key: metadata[key]
            for key in ("unit", "spacing", "finterval", "fps", "origin_zyx_um", "anchor_t", "crop_start_zyx")
            if key in metadata
        }
        metadata.update(
            axes="TZYX",
            tag_pipeline=VERSION,
            affine_method="same mask-assisted rigid transforms and crop as paired raw_pre_affine DAPI",
            source_tag_pre=str(pre_candidate),
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        if output.exists() or temporary.exists():
            raise FileExistsError(output if output.exists() else temporary)
        crop = tuple(slice(int(start), int(start + size)) for start, size in zip(crop_start, output_shape))
        hashes = []
        affine_counts = []

        def planes():
            for t, matrix in enumerate(matrices):
                frame = read_frame(pre_file, t).astype(np.float32, copy=False)
                full = affine_ops.resample(frame, matrix, spacing, origin, full_shape, order=1)
                registered = np.rint(np.clip(full, 0, 65535)).astype(np.uint16)[crop]
                hashes.append(hashlib.sha256(registered.tobytes()).hexdigest())
                affine_counts.append(int(np.count_nonzero(registered)))
                yield from registered

        tifffile.imwrite(
            temporary,
            planes(),
            shape=(pre_shape[0], *output_shape),
            dtype=np.uint16,
            metadata=metadata,
            resolution=(1.0 / spacing[2], 1.0 / spacing[1]),
            photometric="minisblack",
            compression="zlib",
            bigtiff=True,
        )

    with tifffile.TiffFile(temporary) as check:
        if tuple(check.series[0].shape) != (pre_shape[0], *output_shape) or check.series[0].axes != "TZYX":
            raise AssertionError(f"Affine Tag output shape/axes mismatch: {check.series[0].shape}, {check.series[0].axes}")
        for t, expected in enumerate(hashes):
            if hashlib.sha256(read_frame(check, t).tobytes()).hexdigest() != expected:
                raise AssertionError(f"Affine Tag read-back hash mismatch at t={t}: {output.name}")
    os.replace(temporary, output)
    return {
        "shape_tzyx": [pre_shape[0], *output_shape],
        "nonzero_voxels_by_t": affine_counts,
        "file": fingerprint(output),
        "sha256": digest(output),
    }


def validate_sources(sample: dict, require_pre_unchanged: bool = True) -> dict[str, Path]:
    paths = {key: Path(value["path"]) for key, value in sample["sources"].items()}
    for key, expected in sample["sources"].items():
        path = Path(expected["path"])
        if not path.is_file() or fingerprint(path) != expected:
            raise RuntimeError(f"Source changed or disappeared after prior audit: {path}")
    if require_pre_unchanged:
        raw_tag = paths["raw_tag"]
        pre_tag = paths["pre_tag"]
        if not pre_tag.is_file() or digest(raw_tag) != digest(pre_tag):
            raise RuntimeError(f"Existing raw_pre Tag must still be an untouched raw copy before promotion: {pre_tag}")
    return paths


def prepare(release: Path) -> None:
    audit = release / "image" / "tag_processing_audit"
    prior_report = audit / "tag_processing_report.json"
    report_path = audit / "top6_processing_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing top-six report: {report_path}")
    prior = json.loads(prior_report.read_text())
    if prior.get("status") not in ("audit_candidates_ready", "applied_in_place"):
        raise ValueError(f"Unexpected prior audit status: {prior.get('status')}")

    candidate_root = audit / "top6_candidates"
    if candidate_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing top-six candidates: {candidate_root}")

    result = {
        "version": VERSION,
        "status": "preparing",
        "release_root": str(release),
        "selection": {
            "max_candidates_per_frame": MAX_POINTS,
            "top6_are_forced_even_below_threshold": True,
            "below_threshold_candidates_are_reported": True,
            "candidate_signal": "original Tag intensity in high-pass connected components",
            "support_size_zyx": SUPPORT_SIZE,
            "peak_neighborhood_zyx": PEAK_SIZE,
            "maxima_nonmaximum_suppression": True,
        },
        "samples": [],
    }
    for index, sample in enumerate(prior["samples"], start=1):
        paths = validate_sources(sample)
        raw_tag = paths["raw_tag"]
        mask_path = paths["mask_post"]
        category = sample["category"]
        pre_target = paths["pre_tag"]
        affine_target = release / "image" / "raw_pre_affine" / "tiff" / category / raw_tag.name
        pre_candidate = candidate_root / "pre" / category / raw_tag.name
        affine_candidate = candidate_root / "affine" / category / raw_tag.name
        print(f"[{index}/{len(prior['samples'])}] Selecting Tag peaks: {raw_tag.name}", flush=True)

        pre_info = write_pre_candidate(raw_tag, mask_path, pre_candidate)
        affine_info = write_affine_candidate(
            pre_candidate,
            paths["affine_dapi"],
            sample["transform"],
            affine_candidate,
        )
        frames = pre_info["frames"]
        selected_counts = [frame["stats"]["candidate_count"] for frame in frames]
        below_counts = [frame["stats"]["below_threshold_count"] for frame in frames]
        row = {
            "sample": sample["sample"],
            "category": category,
            "sources": sample["sources"],
            "transform": sample["transform"],
            "pre_target": str(pre_target),
            "affine_target": str(affine_target),
            "affine_target_before": fingerprint(affine_target) if affine_target.exists() else None,
            "candidate_pre": str(pre_candidate),
            "candidate_affine": str(affine_candidate),
            "pre": pre_info,
            "affine": affine_info,
            "frames_with_six": int(sum(count == MAX_POINTS for count in selected_counts)),
            "frames_total": len(selected_counts),
            "below_threshold_selected_total": int(sum(below_counts)),
        }
        result["samples"].append(row)
        print(
            f"  selected/frame min={min(selected_counts)} max={max(selected_counts)}; "
            f"below-threshold picks={sum(below_counts)}; affine shape={affine_info['shape_tzyx']}",
            flush=True,
        )

    result["status"] = "top6_candidates_ready"
    result["created_utc_epoch"] = time.time()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    os.replace(temporary, report_path)
    print(f"Top-six audit report: {report_path}", flush=True)


def apply(release: Path) -> None:
    report_path = release / "image" / "tag_processing_audit" / "top6_processing_report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "top6_candidates_ready":
        raise ValueError(f"Top-six candidates are not ready: {report.get('status')}")

    plan = []
    for sample in report["samples"]:
        validate_sources(sample)
        pre_target = Path(sample["pre_target"])
        affine_target = Path(sample["affine_target"])
        pre_candidate = Path(sample["candidate_pre"])
        affine_candidate = Path(sample["candidate_affine"])
        if not pre_candidate.is_file() or not affine_candidate.is_file():
            raise FileNotFoundError(f"Missing verified top-six candidates for {sample['sample']}")
        before = sample["affine_target_before"]
        if before is None:
            if affine_target.exists():
                raise RuntimeError(f"Affine Tag target appeared after prepare: {affine_target}")
        elif not affine_target.is_file() or fingerprint(affine_target) != before:
            raise RuntimeError(f"Affine Tag target changed after prepare: {affine_target}")
        if digest(pre_candidate) != sample["pre"]["sha256"]:
            raise RuntimeError(f"Prepared pre Tag candidate changed: {pre_candidate}")
        if digest(affine_candidate) != sample["affine"]["sha256"]:
            raise RuntimeError(f"Prepared affine Tag candidate changed: {affine_candidate}")
        plan.append((sample, pre_candidate, pre_target, affine_candidate, affine_target))

    for sample, pre_candidate, pre_target, affine_candidate, affine_target in plan:
        pre_target.parent.mkdir(parents=True, exist_ok=True)
        affine_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pre_candidate, pre_target)
        os.replace(affine_candidate, affine_target)
        print(f"Applied top-six Tag: {sample['sample']}", flush=True)

    report["status"] = "applied_in_place"
    report["applied_utc_epoch"] = time.time()
    report["raw_tag_unchanged"] = True
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    os.replace(temporary, report_path)
    print(f"Updated report: {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="ourslive/release directory")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="Create and verify top-six candidates")
    action.add_argument("--apply", action="store_true", help="Promote verified candidates in place")
    args = parser.parse_args()
    release = args.release.resolve()
    if not (release / "image" / "raw").is_dir():
        parser.error(f"Not a release root with image/raw: {release}")
    if args.prepare:
        prepare(release)
    else:
        apply(release)


if __name__ == "__main__":
    main()
