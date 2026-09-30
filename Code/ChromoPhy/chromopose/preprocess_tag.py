#!/usr/bin/env python3
"""Conservatively preprocess Tag TIFFs and carry them through DAPI affine maps.

This utility has two explicit stages. ``--prepare`` writes auditable candidates
and a report without changing release images. After reviewing the report and
the generated contact sheet, ``--apply`` promotes the candidates in place.
The original Tag TIFFs in ``image/raw`` remain untouched.

Example:
    python preprocess_tag.py --root /path/to/ourslive/release --prepare
    python preprocess_tag.py --root /path/to/ourslive/release --apply
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

import affine as affine_ops


VERSION = "tag-background-contrast-v1.0"
SUPPORT_RADIUS_UM = 1.2
OUTSIDE_NOISE_SIGMA = 2.5
OUTPUT_PERCENTILE = 99.9
OUTPUT_GAMMA = 0.85
OUTPUT_PEAK = 60000.0


def metadata_from_tiff(tif: tifffile.TiffFile) -> dict:
    metadata = dict(tif.imagej_metadata or {})
    if metadata:
        return metadata
    try:
        return json.loads(tif.pages[0].description or "{}")
    except (ValueError, TypeError):
        return {}


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_tzyx_frame(tif: tifffile.TiffFile, t: int) -> np.ndarray:
    series = tif.series[0]
    if series.axes != "TZYX":
        raise ValueError(f"Expected TZYX, got {series.axes} {series.shape}")
    return affine_ops.read_frame(tif, t)


def correct_tag_frame(
    tag: np.ndarray, dapi_pre: np.ndarray, spacing_zyx_um: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Subtract measured background and suppress only faint off-chromatin noise.

    No hard threshold is applied inside the generously dilated DAPI support.
    High-intensity signal outside that support is retained as a possible real
    Tag spot; only values within 2.5 robust noise SD of the local background
    are removed there.
    """
    smoothed = ndi.gaussian_filter(
        tag.astype(np.float32), sigma=(0.35, 0.55, 0.55), mode="reflect"
    )
    dapi_support = dapi_pre > 0
    if np.any(dapi_support):
        distance = ndi.distance_transform_edt(
            ~dapi_support, sampling=tuple(float(v) for v in spacing_zyx_um)
        )
        support = distance <= SUPPORT_RADIUS_UM
    else:
        # If paired DAPI has no foreground, avoid using it to delete Tag data.
        support = np.ones(tag.shape, dtype=bool)

    outside = ~support
    if np.count_nonzero(outside) < max(1024, int(0.02 * outside.size)):
        # Dense support is not a reason to infer an aggressive background mask.
        support = np.ones(tag.shape, dtype=bool)
        outside = ~support

    exterior_values = smoothed[outside]
    if exterior_values.size:
        global_background = float(np.median(exterior_values))
        global_noise = max(
            float(1.4826 * np.median(np.abs(exterior_values - global_background))),
            1.0,
        )
    else:
        global_background = float(np.median(smoothed))
        global_noise = max(
            float(1.4826 * np.median(np.abs(smoothed - global_background))), 1.0
        )

    background = np.full(tag.shape[0], global_background, dtype=np.float32)
    noise = np.full(tag.shape[0], global_noise, dtype=np.float32)
    for z in range(tag.shape[0]):
        values = smoothed[z][outside[z]]
        if values.size >= 256:
            median = float(np.median(values))
            background[z] = median
            noise[z] = max(float(1.4826 * np.median(np.abs(values - median))), 1.0)

    corrected = np.maximum(smoothed - background[:, None, None], 0.0)
    inside_mass_before = float(corrected[support].sum(dtype=np.float64))
    outside_mass_before = float(corrected[outside].sum(dtype=np.float64))
    outside_low = outside & (
        smoothed <= (background[:, None, None] + OUTSIDE_NOISE_SIGMA * noise[:, None, None])
    )
    corrected[outside_low] = 0.0
    inside_mass_after = float(corrected[support].sum(dtype=np.float64))
    outside_mass_after = float(corrected[outside].sum(dtype=np.float64))

    frame_metrics = {
        "dapi_support_fraction": float(support.mean()),
        "background_median_by_z": float(np.median(background)),
        "noise_sigma_median_by_z": float(np.median(noise)),
        "inside_excess_mass_retained": inside_mass_after / max(inside_mass_before, 1e-9),
        "outside_excess_mass_retained": outside_mass_after / max(outside_mass_before, 1e-9),
        "outside_low_noise_voxels_removed": int(np.count_nonzero(outside_low)),
        "outside_voxels": int(np.count_nonzero(outside)),
    }
    return corrected, support, frame_metrics


def sample_signal_quantile(
    tag_path: Path, dapi_path: Path, spacing: np.ndarray
) -> tuple[float, list[dict]]:
    samples = []
    first_pass = []
    with tifffile.TiffFile(tag_path) as tag_file, tifffile.TiffFile(dapi_path) as dapi_file:
        tag_series, dapi_series = tag_file.series[0], dapi_file.series[0]
        if tag_series.axes != "TZYX" or tag_series.shape != dapi_series.shape:
            raise ValueError(f"Tag/DAPI shape mismatch: {tag_series.shape}, {dapi_series.shape}")
        for t in range(tag_series.shape[0]):
            corrected, support, metrics = correct_tag_frame(
                read_tzyx_frame(tag_file, t), read_tzyx_frame(dapi_file, t), spacing
            )
            sampled = corrected[::2, ::4, ::4]
            sampled_support = support[::2, ::4, ::4]
            values = sampled[sampled_support & (sampled > 0)]
            if values.size:
                samples.append(values.astype(np.float32, copy=False))
            first_pass.append({"t": t, **metrics})
    if not samples:
        raise ValueError(f"No positive Tag signal remains after conservative background subtraction: {tag_path}")
    values = np.concatenate(samples)
    high = float(np.percentile(values, OUTPUT_PERCENTILE))
    if not np.isfinite(high) or high <= 0:
        raise ValueError(f"Invalid contrast scale {high} for {tag_path}")
    return high, first_pass


def scale_contrast(corrected: np.ndarray, high: float) -> np.ndarray:
    scaled = np.power(np.maximum(corrected / high, 0.0), OUTPUT_GAMMA) * OUTPUT_PEAK
    return np.rint(np.clip(scaled, 0.0, 65535.0)).astype(np.uint16)


def write_preprocessed_tag(
    source: Path,
    dapi_pre: Path,
    candidate: Path,
    spacing: np.ndarray,
    scale: float,
    first_pass: list[dict],
) -> dict:
    with tifffile.TiffFile(source) as input_file:
        series = input_file.series[0]
        shape = series.shape
        original_metadata = metadata_from_tiff(input_file)
    metadata = {
        key: original_metadata[key]
        for key in ("unit", "spacing", "finterval", "fps")
        if key in original_metadata
    }
    metadata.update(
        axes="TZYX",
        tag_pipeline=VERSION,
        background_method="per-z exterior median; gentle Gaussian denoise; no hard threshold inside dilated DAPI support",
        dapi_support_radius_um=SUPPORT_RADIUS_UM,
        outside_noise_threshold_sigma=OUTSIDE_NOISE_SIGMA,
        contrast_percentile=OUTPUT_PERCENTILE,
        contrast_gamma=OUTPUT_GAMMA,
        contrast_peak=OUTPUT_PEAK,
    )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    temporary = candidate.with_name(candidate.name + ".partial")
    if candidate.exists() or temporary.exists():
        raise FileExistsError(candidate if candidate.exists() else temporary)
    hashes = []
    with tifffile.TiffFile(source) as tag_file, tifffile.TiffFile(dapi_pre) as dapi_file:
        def planes():
            for t in range(shape[0]):
                corrected, _, _ = correct_tag_frame(
                    read_tzyx_frame(tag_file, t), read_tzyx_frame(dapi_file, t), spacing
                )
                frame = scale_contrast(corrected, scale)
                hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
                yield from frame

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
    with tifffile.TiffFile(temporary) as output_file:
        if output_file.series[0].shape != shape or output_file.series[0].axes != "TZYX":
            raise AssertionError(f"Preprocessed Tag shape/axes mismatch: {output_file.series[0].shape}")
        for t, expected in enumerate(hashes):
            actual = hashlib.sha256(read_tzyx_frame(output_file, t).tobytes()).hexdigest()
            if actual != expected:
                raise AssertionError(f"Preprocessed Tag failed readback verification at t={t}")
    os.replace(temporary, candidate)
    return {
        "shape_tzyx": list(shape),
        "dtype": "uint16",
        "contrast_scale_input": scale,
        "frame_background_metrics": first_pass,
        "verified_frame_sha256": hashes,
        "file": fingerprint(candidate),
    }


def resolve_sample_paths(release: Path, raw_tag: Path) -> dict[str, Path]:
    category = raw_tag.parent.name
    name = raw_tag.name
    dapi_name = name.removesuffix("-Tag.tif") + "-DAPI.tif"
    mask_name = name.removesuffix("-Tag.tif") + "_shape_postprocessed.tif"
    return {
        "raw_tag": raw_tag,
        "raw_dapi": release / "image" / "raw" / category / dapi_name,
        "pre_tag": release / "image" / "raw_pre" / "tiff" / category / name,
        "pre_dapi": release / "image" / "raw_pre" / "tiff" / category / dapi_name,
        "affine_dapi": release / "image" / "raw_pre_affine" / "tiff" / category / dapi_name,
        "mask_post": release / "mask" / "chrp3d_post" / category / mask_name,
        "affine_tag": release / "image" / "raw_pre_affine" / "tiff" / category / name,
    }


def reconstruct_and_validate_transforms(paths: dict[str, Path]) -> dict:
    if affine_ops.VERSION != "rigid-stabilization-1.1":
        raise RuntimeError(f"Unexpected affine implementation version: {affine_ops.VERSION}")
    with (
        tifffile.TiffFile(paths["raw_dapi"]) as raw_file,
        tifffile.TiffFile(paths["pre_dapi"]) as pre_file,
        tifffile.TiffFile(paths["affine_dapi"]) as affine_file,
        tifffile.TiffFile(paths["mask_post"]) as mask_file,
    ):
        shape = raw_file.series[0].shape
        for key, tif in (("pre_dapi", pre_file), ("mask_post", mask_file)):
            if tif.series[0].axes != "TZYX" or tif.series[0].shape != shape:
                raise ValueError(f"{key} must match raw DAPI shape {shape}; got {tif.series[0].shape}")
        affine_shape = affine_file.series[0].shape
        if affine_file.series[0].axes != "TZYX" or affine_shape[0] != shape[0]:
            raise ValueError(f"Affine DAPI axes/frame count mismatch: {affine_shape}")
        spacing, _ = affine_ops.get_metadata(raw_file)
        pre_spacing, _ = affine_ops.get_metadata(pre_file)
        if not np.allclose(spacing, pre_spacing):
            raise ValueError("Raw and preprocessed DAPI voxel spacing differs")
        affine_metadata = metadata_from_tiff(affine_file)
        anchor = int(affine_metadata["anchor_t"])
        crop_start = np.asarray(affine_metadata["crop_start_zyx"], dtype=int)
        output_origin = np.asarray(affine_metadata["origin_zyx_um"], dtype=float)
        output_shape = np.asarray(affine_shape[1:], dtype=int)

        pairs = []
        previous_raw = affine_ops.read_frame(raw_file, 0)
        previous_mask = affine_ops.read_frame(mask_file, 0)
        previous_prepared = affine_ops.prepare(previous_raw, previous_mask)
        for t in range(1, shape[0]):
            current_raw = affine_ops.read_frame(raw_file, t)
            current_mask = affine_ops.read_frame(mask_file, t)
            current_prepared = affine_ops.prepare(current_raw, current_mask)
            old, _, _ = affine_ops.registration_shift(previous_raw, current_raw)
            matrix, detail = affine_ops.choose(
                previous_prepared,
                current_prepared,
                previous_mask,
                current_mask,
                spacing,
                old,
            )
            pairs.append(matrix)
            previous_raw, previous_mask, previous_prepared = (
                current_raw,
                current_mask,
                current_prepared,
            )
            print(
                f"  transform {paths['raw_tag'].stem} t={t}: {detail['selected']}, uncertain={detail['uncertain']}",
                flush=True,
            )

        matrices = affine_ops.compose(pairs, anchor)
        full_origin, full_shape = affine_ops.canvas(shape[1:], spacing, matrices)
        expected_origin = full_origin + crop_start * spacing
        if not np.allclose(output_origin, expected_origin, atol=1e-7):
            raise AssertionError(
                f"Reconstructed crop origin mismatch for {paths['raw_tag'].name}: "
                f"target={output_origin.tolist()} reconstructed={expected_origin.tolist()}"
            )
        if np.any(crop_start < 0) or np.any(crop_start + output_shape > np.asarray(full_shape)):
            raise AssertionError("Affine crop lies outside reconstructed registration canvas")

        crop = tuple(
            slice(int(start), int(start + length))
            for start, length in zip(crop_start, output_shape)
        )
        dapi_hashes = []
        for t in range(shape[0]):
            pre = affine_ops.read_frame(pre_file, t)
            registered = affine_ops.resample(
                pre.astype(np.float32), matrices[t], spacing, full_origin, full_shape, order=1
            )
            registered = np.rint(np.clip(registered, 0, 65535)).astype(np.uint16)
            registered = registered[crop]
            target = affine_ops.read_frame(affine_file, t)
            if not np.array_equal(registered, target):
                different = int(np.count_nonzero(registered != target))
                raise AssertionError(
                    f"Reconstructed affine DAPI differs at t={t} for {paths['raw_tag'].name}: "
                    f"{different} voxels differ"
                )
            dapi_hashes.append(hashlib.sha256(registered.tobytes()).hexdigest())

    return {
        "spacing_zyx_um": spacing.tolist(),
        "anchor_t": anchor,
        "full_origin_zyx_um": full_origin.tolist(),
        "full_shape_zyx": list(full_shape),
        "crop_start_zyx": crop_start.tolist(),
        "output_shape_tzyx": list(affine_shape),
        "frame_to_anchor_physical_zyx": [matrix.tolist() for matrix in matrices],
        "dapi_reconstruction_exact_all_frames": True,
        "verified_dapi_frame_sha256": dapi_hashes,
    }


def write_affine_tag(
    candidate_pre: Path,
    candidate_affine: Path,
    transform: dict,
    affine_dapi: Path,
) -> dict:
    spacing = np.asarray(transform["spacing_zyx_um"], dtype=float)
    full_origin = np.asarray(transform["full_origin_zyx_um"], dtype=float)
    full_shape = tuple(int(v) for v in transform["full_shape_zyx"])
    crop_start = np.asarray(transform["crop_start_zyx"], dtype=int)
    output_shape = tuple(int(v) for v in transform["output_shape_tzyx"][1:])
    matrices = [np.asarray(matrix, dtype=float) for matrix in transform["frame_to_anchor_physical_zyx"]]

    with tifffile.TiffFile(candidate_pre) as pre_file, tifffile.TiffFile(affine_dapi) as affine_file:
        pre_shape = pre_file.series[0].shape
        if pre_shape[0] != len(matrices):
            raise ValueError("Preprocessed Tag frame count does not match affine transforms")
        affine_metadata = metadata_from_tiff(affine_file)
    metadata = {
        key: affine_metadata[key]
        for key in ("unit", "spacing", "finterval", "fps", "pipeline", "origin_zyx_um", "anchor_t", "crop_start_zyx", "crop_margin_um")
        if key in affine_metadata
    }
    metadata.update(
        axes="TZYX",
        tag_pipeline=VERSION,
        affine_method="same mask-assisted rigid-stabilization-1.1 transforms as paired DAPI",
    )
    candidate_affine.parent.mkdir(parents=True, exist_ok=True)
    temporary = candidate_affine.with_name(candidate_affine.name + ".partial")
    if candidate_affine.exists() or temporary.exists():
        raise FileExistsError(candidate_affine if candidate_affine.exists() else temporary)
    crop = tuple(
        slice(int(start), int(start + length))
        for start, length in zip(crop_start, output_shape)
    )
    affine_metrics = []
    hashes = []
    with tifffile.TiffFile(candidate_pre) as pre_file:
        def planes():
            for t, matrix in enumerate(matrices):
                pre = read_tzyx_frame(pre_file, t)
                full = affine_ops.resample(
                    pre.astype(np.float32), matrix, spacing, full_origin, full_shape, order=1
                )
                full = np.rint(np.clip(full, 0, 65535)).astype(np.uint16)
                moved = np.ascontiguousarray(full[crop])
                total_signal = int(full.sum(dtype=np.uint64))
                kept_signal = int(moved.sum(dtype=np.uint64))
                total_high = int(np.count_nonzero(full >= 30000))
                kept_high = int(np.count_nonzero(moved >= 30000))
                affine_metrics.append(
                    {
                        "t": t,
                        "positive_intensity_mass_retained": kept_signal / max(total_signal, 1),
                        "high_signal_voxels_retained_ge_30000": kept_high / max(total_high, 1),
                        "high_signal_voxels_full": total_high,
                        "high_signal_voxels_kept": kept_high,
                    }
                )
                hashes.append(hashlib.sha256(moved.tobytes()).hexdigest())
                yield from moved

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
    with tifffile.TiffFile(temporary) as output_file:
        if output_file.series[0].shape != (pre_shape[0], *output_shape):
            raise AssertionError(f"Affine Tag shape mismatch: {output_file.series[0].shape}")
        for t, expected in enumerate(hashes):
            actual = hashlib.sha256(read_tzyx_frame(output_file, t).tobytes()).hexdigest()
            if actual != expected:
                raise AssertionError(f"Affine Tag failed readback verification at t={t}")
    os.replace(temporary, candidate_affine)
    return {
        "frame_metrics": affine_metrics,
        "verified_frame_sha256": hashes,
        "file": fingerprint(candidate_affine),
    }


def prepare_release(release: Path) -> None:
    if affine_ops.VERSION != "rigid-stabilization-1.1":
        raise RuntimeError("The installed affine code does not match the released mask-assisted pipeline")
    image_root = release / "image"
    audit_root = image_root / "tag_processing_audit"
    candidate_pre_root = audit_root / "candidates" / "pre"
    candidate_affine_root = audit_root / "candidates" / "affine"
    report_path = audit_root / "tag_processing_report.json"
    if report_path.exists() or candidate_pre_root.exists() or candidate_affine_root.exists():
        raise FileExistsError("Tag audit candidates already exist; refusing to overwrite them")

    tags = sorted((image_root / "raw").rglob("*-Tag.tif"))
    if not tags:
        raise FileNotFoundError(f"No raw Tag TIFFs found under {image_root / 'raw'}")

    report = {
        "version": VERSION,
        "status": "preparing",
        "release_root": str(release),
        "policy": {
            "raw_tag_is_immutable": True,
            "inside_dilated_dapi_support_hard_threshold": False,
            "outside_support_noise_threshold_sigma": OUTSIDE_NOISE_SIGMA,
            "support_radius_um": SUPPORT_RADIUS_UM,
            "contrast_percentile": OUTPUT_PERCENTILE,
            "contrast_gamma": OUTPUT_GAMMA,
            "contrast_peak": OUTPUT_PEAK,
            "affine": "reconstructed from paired raw DAPI and chrp3d_post, then exact-compared with current affine DAPI",
        },
        "samples": [],
    }
    for index, raw_tag in enumerate(tags, start=1):
        paths = resolve_sample_paths(release, raw_tag)
        missing = [str(path) for key, path in paths.items() if key != "affine_tag" and not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required paired inputs:\n" + "\n".join(missing))
        if paths["affine_tag"].exists():
            raise FileExistsError(f"Affine Tag already exists; refusing to overwrite: {paths['affine_tag']}")
        if not filecmp.cmp(paths["raw_tag"], paths["pre_tag"], shallow=False):
            raise ValueError(f"Existing raw_pre Tag is not byte-identical to raw; refusing in-place replacement: {paths['pre_tag']}")

        print(f"[{index}/{len(tags)}] Reconstructing affine: {raw_tag.name}", flush=True)
        transform = reconstruct_and_validate_transforms(paths)
        with tifffile.TiffFile(paths["raw_tag"]) as tag_file, tifffile.TiffFile(paths["raw_dapi"]) as dapi_file:
            if tag_file.series[0].shape != dapi_file.series[0].shape:
                raise ValueError(f"Raw Tag and DAPI shape mismatch: {raw_tag.name}")
            tag_spacing, _ = affine_ops.get_metadata(tag_file)
            if not np.allclose(tag_spacing, transform["spacing_zyx_um"]):
                raise ValueError(f"Tag and DAPI physical spacing differs: {raw_tag.name}")
        candidate_pre = candidate_pre_root / raw_tag.parent.name / raw_tag.name
        candidate_affine = candidate_affine_root / raw_tag.parent.name / raw_tag.name
        scale, frame_background = sample_signal_quantile(
            paths["raw_tag"], paths["pre_dapi"], np.asarray(transform["spacing_zyx_um"])
        )
        pre_report = write_preprocessed_tag(
            paths["raw_tag"],
            paths["pre_dapi"],
            candidate_pre,
            np.asarray(transform["spacing_zyx_um"]),
            scale,
            frame_background,
        )
        affine_report = write_affine_tag(
            candidate_pre, candidate_affine, transform, paths["affine_dapi"]
        )
        report["samples"].append(
            {
                "sample": raw_tag.name.removesuffix("-Tag.tif"),
                "category": raw_tag.parent.name,
                "sources": {key: fingerprint(path) for key, path in paths.items() if key != "affine_tag"},
                "transform": transform,
                "preprocessed": pre_report,
                "affine": affine_report,
                "candidate_pre": str(candidate_pre),
                "candidate_affine": str(candidate_affine),
            }
        )
        print(
            f"  candidates ready; contrast scale={scale:.3f}; affine DAPI exact on all frames",
            flush=True,
        )

    report["status"] = "audit_candidates_ready"
    report["created_utc_epoch"] = time.time()
    audit_root.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(".json.partial")
    temporary_report.write_text(json.dumps(report, indent=2))
    os.replace(temporary_report, report_path)
    print(f"Audit report: {report_path}", flush=True)


def apply_candidates(release: Path) -> None:
    image_root = release / "image"
    audit_root = image_root / "tag_processing_audit"
    report_path = audit_root / "tag_processing_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Run --prepare first: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("status") != "audit_candidates_ready":
        raise ValueError(f"Candidates are not ready to apply (status={report.get('status')})")

    planned = []
    for sample in report["samples"]:
        transform = sample["transform"]
        if not transform.get("dapi_reconstruction_exact_all_frames"):
            raise ValueError(f"DAPI affine reconstruction was not exact: {sample['sample']}")
        candidate_pre = Path(sample["candidate_pre"])
        candidate_affine = Path(sample["candidate_affine"])
        paths = {key: Path(value["path"]) for key, value in sample["sources"].items()}
        pre_target = paths["pre_tag"]
        affine_target = release / "image" / "raw_pre_affine" / "tiff" / sample["category"] / candidate_affine.name
        if not candidate_pre.is_file() or not candidate_affine.is_file():
            raise FileNotFoundError(f"Missing candidate TIFF for {sample['sample']}")
        if affine_target.exists():
            raise FileExistsError(f"Refusing to overwrite existing affine Tag: {affine_target}")
        for key, source in paths.items():
            current = fingerprint(source)
            if current != sample["sources"][key]:
                raise RuntimeError(f"Source changed after audit: {source}")
        if not filecmp.cmp(paths["raw_tag"], pre_target, shallow=False):
            raise RuntimeError(f"Existing pre Tag ceased to match raw source: {pre_target}")
        planned.append((candidate_pre, pre_target, candidate_affine, affine_target))

    for candidate_pre, pre_target, candidate_affine, affine_target in planned:
        os.replace(candidate_pre, pre_target)
        affine_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate_affine, affine_target)
        print(f"Applied: {pre_target.name}; created {affine_target}", flush=True)

    report["status"] = "applied_in_place"
    report["applied_utc_epoch"] = time.time()
    temporary_report = report_path.with_suffix(".json.partial")
    temporary_report.write_text(json.dumps(report, indent=2))
    os.replace(temporary_report, report_path)
    print(f"Updated audit report: {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="ourslive/release directory")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="Create candidates and an audit report without replacing release data")
    action.add_argument("--apply", action="store_true", help="Promote audited candidates in place; raw Tag sources remain unchanged")
    args = parser.parse_args()
    release = args.root.resolve()
    if not (release / "image" / "raw").is_dir():
        parser.error(f"Not a release root with image/raw: {release}")
    if args.prepare:
        prepare_release(release)
    else:
        apply_candidates(release)


if __name__ == "__main__":
    main()
