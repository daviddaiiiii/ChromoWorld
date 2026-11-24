"""Render color-stable front and back previews from native tracked TIFFs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import tifffile as tf

from prepare_live_comparison import calibrated_xy, palette, read_frame

MITOSIS = "RPE1_3_F2CGD"
SIZE = 420


def physical_spacing(image):
    if image.ome_metadata:
        pixels = ET.fromstring(image.ome_metadata).find(".//{*}Pixels")
        if pixels is not None and all(pixels.get(f"PhysicalSize{axis}Unit", "µm") in ("µm", "um")
                                      for axis in "ZYX"):
            sizes = [pixels.get(f"PhysicalSize{axis}") for axis in "ZYX"]
            if all(sizes):
                return np.array(sizes, dtype=float)
    xy = calibrated_xy(image)
    z = (image.imagej_metadata or {}).get("spacing")
    return np.array([z, xy, xy], dtype=float)


def export(case, release, output, renderer):
    sys.path.insert(0, str(renderer))
    from rounded_render import build_surfaces, shade

    name = case["source"]
    group = case["group"]
    key = "mitosis" if group == "others" and name == MITOSIS else case["key"]
    folder = output / key
    folder.mkdir(parents=True, exist_ok=True)
    image_path = release / "image/raw_pre/tiff" / group / f"{name}-DAPI.tif"
    mask_path = release / "mask/chrp3d_post_tracked" / group / f"{name}.tif"
    with tf.TiffFile(image_path) as image, tf.TiffFile(mask_path) as masks:
        raw, tracked = image.series[0], masks.series[0]
        if tracked.axes != "TZYX" or raw.axes != "TZYX" or raw.shape != tracked.shape:
            raise ValueError(f"DNA and tracks have different native grids: {name}")
        if raw.shape[2] != raw.shape[3] or not np.issubdtype(tracked.dtype, np.integer):
            raise ValueError(f"Expected a square XY image and integer labels: {name}")
        spacing = physical_spacing(image)
        if not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError(f"Native voxel calibration is missing: {name}")
        depth = (raw.shape[1] - 1) * spacing[0]
        n = raw.shape[0]
        if "frames" in case and [f["t"] for f in case["frames"]] != list(range(n)):
            raise ValueError(f"Comparison and native track time points differ: {name}")
        mips = np.stack([read_frame(raw, t).max(axis=0) for t in range(n)])
        low, high = np.percentile(mips, [38, 99.7])
        for t in range(n):
            labels = read_frame(tracked, t)
            if np.min(labels) < 0:
                raise ValueError(f"Negative label in {name} at {t}")
            surfaces, _ = build_surfaces(labels, spacing, up=2)
            colors = palette(int(labels.max()))
            dna = np.uint8(255 * np.clip((mips[t]-low) / max(high-low, 1), 0, 1)**.83)
            for face, surface in enumerate(surfaces):
                suffix = "-back" if face else ""
                bounds = (-depth, 0.) if face else (0., depth)
                colored = shade(surface, colors, spacing, bounds, up=2)
                for layer, pixels in [("mask", colored), ("dna", dna[:, ::-1] if face else dna)]:
                    view = Image.fromarray(pixels).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
                    view.save(folder / f"{t}-{layer}{suffix}.webp", "WEBP", quality=91)
    print(f"Rendered {key}: {n} frames, front and back", flush=True)
    return {"key": key, "name": name,
            "spacing": spacing.tolist(), "xySize": raw.shape[3], "reference": 18 if key == "mitosis" else case.get("reference", n//2),
            "frames": [{"t": t} for t in range(n)], "experiment": None, "candidates": []}


def patient_cases(release):
    for dna in sorted((release / "image/raw_pre/tiff/patients").glob("*-DAPI.tif")):
        name = dna.stem.removesuffix("-DAPI")
        key = "patient-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        yield {"key": key, "source": name, "group": "patients"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("comparison_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--renderer", required=True, type=Path, help="Directory containing rounded_render.py")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--collection", choices=("normal","patients"), default="normal")
    args = parser.parse_args()
    cases = list(patient_cases(args.release)) if args.collection == "patients" else [
        c for c in json.loads(args.comparison_index.read_text()) if c["group"] == "others"]
    if not cases:
        parser.error(f"No matched inputs for {args.collection}")
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        jobs = [workers.submit(export, c, args.release, args.output, args.renderer) for c in cases]
        index = [job.result() for job in jobs]
    index.sort(key=lambda c: c["key"] != "mitosis")
    (args.output / "index.json").write_text(json.dumps(index, separators=(",",":")), encoding="utf-8")
