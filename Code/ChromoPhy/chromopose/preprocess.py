# Standalone DAPI preprocessing. Python >=3.10.
# Dependencies: numpy, scipy, tifffile, opencv-python-headless.
# Run: python preprocess.py --input RAW_DIR --output NEW_DIR --profile auto
# Inputs: uint16 *-DAPI.tif, ZYX/TZYX with micron spacing and XY resolution.
# auto uses legacy-paired when a matching Tag file is co-located; otherwise legacy-v2.
# Both profiles are supported production algorithms, not obsolete versions.
#!/usr/bin/env python3
"""Historical DAPI restoration/background removal, without continuity enhancement.
Recovered from the actual 2026-09-15 processing commands. Inputs are never replaced.
Two explicit profiles preserve the different historical algorithms.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import tifffile
import cv2
from scipy.ndimage import (
    gaussian_filter,
    binary_propagation,
    label,
    binary_dilation,
    generate_binary_structure,
)

structure = generate_binary_structure(3, 3)
VERSION = "historical-restoration-20260916"


def resolve_profile(src, requested):
    """Auto is for this dataset's co-located raw DAPI/Tag naming convention."""
    if requested != "auto":
        return requested
    prefix = src.stem.removesuffix("-DAPI") + "-Tag-Chr"
    paired = any(
        p.name.startswith(prefix) and p.suffix.lower() in (".tif", ".tiff")
        for p in src.parent.iterdir()
    )
    return "legacy-paired" if paired else "legacy-v2"


def read_frame(tif, t):
    s = tif.series[0]
    if s.axes == "ZYX":
        if t != 0:
            raise IndexError(t)
        return s.asarray()
    if s.axes != "TZYX":
        raise ValueError(f"Expected ZYX/TZYX, got {s.axes} {s.shape}")
    z = s.shape[1]
    if len(s.pages) == s.shape[0] * z:
        return np.stack([p.asarray() for p in s.pages[t * z : (t + 1) * z]])
    return np.asarray(tifffile.memmap(tif.filehandle.path)[t]).copy()


def metadata(tif):
    m = dict(tif.imagej_metadata or {})
    if not m:
        try:
            m = json.loads(tif.pages[0].description or "{}")
        except (ValueError, TypeError):
            m = {}
    p = tif.pages[0]

    def spacing(tag):
        v = p.tags[tag].value
        return v[1] / v[0]

    if "spacing" not in m or m.get("unit") not in ("micron", "um", "µm"):
        raise ValueError(
            "Calibrated micron spacing is required; do not assume pixel=micron"
        )
    return (float(m["spacing"]), spacing("YResolution"), spacing("XResolution")), m


def preprocess_file(src, out_dir, profile, max_frames=None):
    if not src.stem.endswith("-DAPI"):
        raise ValueError(f"Only named DAPI inputs are accepted: {src}")
    out = out_dir / src.name
    mask_out = out_dir / "foreground mask" / (src.stem[:-5] + "-foreground-mask.tif")
    report_path = out_dir / (src.stem + ".json")
    for p in (out, mask_out, report_path):
        if p.exists():
            raise FileExistsError(p)
    temp = out.with_name(out.name + ".partial")
    if temp.exists():
        raise FileExistsError(temp)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(src) as tf:
        s = tf.series[0]
        shape = s.shape if s.axes == "TZYX" else (1,) + s.shape
        if s.dtype != np.dtype("uint16"):
            raise ValueError("Historical algorithms require uint16 raw data")
        T, Z, Y, X = shape
        T = min(T, max_frames) if max_frames else T
        sp, original_meta = metadata(tf)
        maps = []
        comps = []
        if profile == "legacy-v2":
            for t in range(T):
                labels, components, *_ = candidate_components(read_frame(tf, t))
                maps.append(labels)
                comps.append(components)
            rois = build_temporal_rois(maps, comps)
        masks = []
        metrics = []
        digests = []

        def planes():
            for t in range(T):
                vol = read_frame(tf, t)
                if profile == "legacy-v2":
                    image, roi, m = restore(vol, rois[t], sp[0], sp[2])
                else:
                    image, roi, m = restore_masked(vol, sp[0])
                masks.append(roi)
                metrics.append(dict(t=t, **m))
                digests.append(hashlib.sha256(image.tobytes()).hexdigest())
                print(
                    json.dumps(dict(file=src.name, t=t, profile=profile, **m)),
                    flush=True,
                )
                yield from image

        meta = {
            k: original_meta[k]
            for k in ("spacing", "unit", "finterval", "fps")
            if k in original_meta
        }
        meta.update(axes="TZYX", pipeline=VERSION, profile=profile)
        tifffile.imwrite(
            temp,
            planes(),
            shape=(T, Z, Y, X),
            dtype="uint16",
            metadata=meta,
            resolution=(1 / sp[2], 1 / sp[1]),
            photometric="minisblack",
            compression="zlib",
        )
    with tifffile.TiffFile(temp) as tf:
        assert tf.series[0].shape == (T, Z, Y, X)
        for t in range(T):
            assert hashlib.sha256(read_frame(tf, t).tobytes()).hexdigest() == digests[t]
    os.replace(temp, out)
    tifffile.imwrite(
        mask_out,
        np.stack(masks),
        metadata={"axes": "TYX", "unit": "micron"},
        resolution=(1 / sp[2], 1 / sp[1]),
        photometric="minisblack",
        compression="zlib",
    )
    report_path.write_text(
        json.dumps(
            dict(
                version=VERSION,
                profile=profile,
                input=str(src),
                output=str(out),
                shape=[T, Z, Y, X],
                spacing_zyx_um=sp,
                frames=metrics,
                verified_sha256=digests,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Raw *-DAPI.tif file or directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Separate output directory"
    )
    parser.add_argument(
        "--profile", choices=["legacy-v2", "legacy-paired", "auto"], required=True
    )
    parser.add_argument("--pattern", default="*-DAPI.tif")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Smoke testing only; recomputes temporal ROI on this subset",
    )
    args = parser.parse_args()
    files = (
        [args.input] if args.input.is_file() else sorted(args.input.glob(args.pattern))
    )
    if not files:
        parser.error("No DAPI files matched")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if any(p.resolve().parent == args.output.resolve() for p in files):
        parser.error("Output must differ from input directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for p in files:
        preprocess_file(
            p, args.output, resolve_profile(p, args.profile), args.max_frames
        )


# Historical processing functions, command source line 383


def estimate_signal(vol):
    x = vol.astype(np.float32, copy=False)
    nz = x[x > 0]
    if nz.size == 0:
        return np.zeros_like(x), 0.0, 1.0
    bg = float(np.percentile(nz, 40.0))
    ceiling = float(np.percentile(nz, 65.0))
    m = (x > 0) & (x <= ceiling)
    estimates = []
    dx = np.diff(x, axis=2)
    mx = m[:, :, 1:] & m[:, :, :-1]
    dy = np.diff(x, axis=1)
    my = m[:, 1:, :] & m[:, :-1, :]
    if mx.any():
        estimates.append(np.median(np.abs(dx[mx])) / 0.67448975 / np.sqrt(2.0))
    if my.any():
        estimates.append(np.median(np.abs(dy[my])) / 0.67448975 / np.sqrt(2.0))
    noise = max(float(np.median(estimates)) if estimates else 2.0, 0.5)
    signal = np.maximum(x - bg - 1.25 * noise, 0.0)
    observed = gaussian_filter(signal, sigma=(0.25, 0.55, 0.55), mode="reflect")
    return observed, bg, noise


def foreground_roi(observed, noise):
    proj = gaussian_filter(observed.max(axis=0), 1.3)
    cap = float(np.percentile(proj, 99.9))
    if cap <= 0:
        return np.ones(proj.shape, bool), 0.0, 0.0
    u = np.clip(proj / cap * 65535.0, 0, 65535).astype(np.uint16)
    otsu, _ = cv2.threshold(u, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(float(otsu) / 65535.0 * cap, 6.0 * noise)
    low = max(2.0 * noise, 0.30 * high)
    grown = binary_propagation(proj >= high, mask=proj >= low)
    mask = cv2.morphologyEx(
        (grown * 255).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
    lab, n = label(mask)
    if n == 0:
        return np.ones(proj.shape, bool), low, high
    scores = [float(proj[lab == k].sum()) for k in range(1, n + 1)]
    order = np.argsort(scores)[::-1]
    dominant = lab == int(order[0] + 1)
    yy, xx = np.nonzero(dominant)
    cy, cx = yy.mean(), xx.mean()
    keep = np.zeros_like(mask, bool)
    top = scores[order[0]]
    for oi in order:
        region = lab == int(oi + 1)
        yy, xx = np.nonzero(region)
        distance = np.hypot(yy.mean() - cy, xx.mean() - cx)
        if scores[oi] >= 0.10 * top and distance <= 150.0:
            keep |= region
    keep = (
        cv2.morphologyEx(
            (keep * 255).astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        )
        > 0
    )
    keep = (
        cv2.dilate(
            (keep * 255).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        )
        > 0
    )
    return keep, low, high


def restore_masked(vol, spacing):
    observed, bg, noise = estimate_signal(vol)
    roi, low, high = foreground_roi(observed, noise)
    area = float(roi.mean())
    if not 0.01 <= area <= 0.80:
        raise ValueError(f"Implausible foreground ROI area {area:.3f}")

    zsigma = float(np.clip(0.30 / max(float(spacing), 1e-6), 0.50, 0.90))
    psf_sigma = (zsigma, 1.15, 1.15)
    estimate = np.maximum(observed, 1e-3)
    for _ in range(6):
        blurred = gaussian_filter(estimate, sigma=psf_sigma, mode="reflect")
        ratio = observed / np.maximum(blurred, 1e-3)
        estimate *= gaussian_filter(ratio, sigma=psf_sigma, mode="reflect")
    restored = gaussian_filter(estimate, sigma=(0.12, 0.28, 0.28), mode="reflect")

    # Soft 2D ROI edge plus voxel-wise signal confidence. This suppresses artifacts
    # without creating a hard artificial border around chromosomes.
    soft_roi = gaussian_filter(roi.astype(np.float32), sigma=3.0, mode="nearest")
    soft_roi[soft_roi < 0.025] = 0.0
    confidence = np.clip(observed / (6.0 * noise), 0.0, 1.0)
    confidence = gaussian_filter(confidence, sigma=(0.15, 0.35, 0.35), mode="reflect")
    cleaned = restored * soft_roi[None, :, :] * np.sqrt(confidence)
    cleaned[cleaned < 0.02 * noise] = 0.0

    positive = cleaned[cleaned > 0]
    high_scale = float(np.percentile(positive, 99.8)) if positive.size else 1.0
    high_scale = max(high_scale, 1e-6)
    normalized = np.power(np.maximum(cleaned / high_scale, 0.0), 0.90)
    out = np.rint(np.clip(normalized * 60000.0, 0.0, 65535.0)).astype(np.uint16)
    metrics = {
        "background": bg,
        "noise": noise,
        "low": low,
        "high": high,
        "roi_area": area,
        "scale": high_scale,
        "zero_fraction": float((out == 0).mean()),
    }
    return out, (roi.astype(np.uint8) * 255), metrics


# Historical processing functions, command source line 682


def odd(value):
    return max(3, int(round(value)) // 2 * 2 + 1)


def uint16_quantile_nonzero(array, q):
    hist = np.bincount(array.reshape(-1), minlength=65536).astype(np.int64, copy=False)
    hist[0] = 0
    total = int(hist.sum())
    if total == 0:
        return 0.0
    target = q / 100.0 * (total - 1)
    return float(np.searchsorted(np.cumsum(hist), target, side="right"))


def estimate_observed(volume):
    bg = uint16_quantile_nonzero(volume, 40.0)
    ceiling = uint16_quantile_nonzero(volume, 65.0)
    zstep = max(1, volume.shape[0] // 32)
    sample = volume[::zstep, ::2, ::2].astype(np.float32)
    m = (sample > 0) & (sample <= ceiling)
    estimates = []
    dx = np.diff(sample, axis=2)
    mx = m[:, :, 1:] & m[:, :, :-1]
    dy = np.diff(sample, axis=1)
    my = m[:, 1:, :] & m[:, :-1, :]
    if mx.any():
        estimates.append(np.median(np.abs(dx[mx])) / 0.67448975 / np.sqrt(2.0))
    if my.any():
        estimates.append(np.median(np.abs(dy[my])) / 0.67448975 / np.sqrt(2.0))
    noise = max(float(np.median(estimates)) if estimates else 2.0, 0.5)
    observed = volume.astype(np.float32)
    observed -= bg + 1.25 * noise
    np.maximum(observed, 0.0, out=observed)
    gaussian_filter(observed, sigma=(0.25, 0.55, 0.55), mode="reflect", output=observed)
    return observed, bg, noise


def candidate_components(volume):
    observed, bg, noise = estimate_observed(volume)
    Y, X = volume.shape[-2:]
    scale = min(Y, X) / 384.0
    projection = observed.max(axis=0)
    gaussian_filter(
        projection, sigma=max(0.8, 1.3 * scale), mode="reflect", output=projection
    )
    cap = float(np.percentile(projection, 99.9))
    if cap <= 0:
        labels = np.ones((Y, X), dtype=np.uint16)
        return (
            labels,
            [{"raw": 1.0, "centroid": ((Y - 1) / 2, (X - 1) / 2)}],
            bg,
            noise,
            0.0,
            0.0,
        )
    scaled = np.clip(projection / cap * 65535.0, 0, 65535).astype(np.uint16)
    otsu, _ = cv2.threshold(scaled, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(float(otsu) / 65535.0 * cap, 6.0 * noise)
    low = max(2.5 * noise, 0.40 * high)
    grown = binary_propagation(projection >= high, mask=projection >= low)
    clustered = cv2.morphologyEx(
        (grown * 255).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (odd(15 * scale), odd(15 * scale))
        ),
    )
    clustered = (
        cv2.dilate(
            clustered,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (odd(11 * scale), odd(11 * scale))
            ),
        )
        > 0
    )
    labels, n = label(clustered)
    labels = labels.astype(np.uint16, copy=False)
    comps = []
    for k in range(1, n + 1):
        region = labels == k
        yy, xx = np.nonzero(region)
        comps.append(
            {
                "raw": float(projection[region].sum()),
                "centroid": (float(yy.mean()), float(xx.mean())),
            }
        )
    if not comps:
        labels = np.ones((Y, X), dtype=np.uint16)
        comps = [{"raw": 1.0, "centroid": ((Y - 1) / 2, (X - 1) / 2)}]
    return labels, comps, bg, noise, low, high


def independent_selection(components, Y, X):
    cy0, cx0 = (Y - 1) / 2.0, (X - 1) / 2.0
    adjusted = []
    for comp in components:
        cy, cx = comp["centroid"]
        distance = np.hypot(cy - cy0, cx - cx0)
        adjusted.append(
            comp["raw"] * np.exp(-0.5 * (distance / (0.42 * min(Y, X))) ** 2)
        )
    dominant = int(np.argmax(adjusted))
    top = components[dominant]["raw"]
    cy, cx = components[dominant]["centroid"]
    selected = {dominant}
    for i, comp in enumerate(components):
        distance = np.hypot(comp["centroid"][0] - cy, comp["centroid"][1] - cx)
        if comp["raw"] >= 0.18 * top and distance <= 0.55 * min(Y, X):
            selected.add(i)
    return selected


def selected_mask(labels, indices):
    result = np.zeros(labels.shape, dtype=np.uint8)
    for i in indices:
        result[labels == (i + 1)] = 255
    return result


def build_temporal_rois(label_maps, all_components):
    T = len(label_maps)
    Y, X = label_maps[0].shape
    scale = min(Y, X) / 384.0
    selected = [independent_selection(c, Y, X) for c in all_components]
    radius = max(5, int(round(0.08 * min(Y, X))))
    temporal_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    for _ in range(4):
        changed = False
        for order in (range(T), range(T - 1, -1, -1)):
            for t in order:
                contexts = []
                for nt in (t - 1, t + 1):
                    if 0 <= nt < T:
                        contexts.append(
                            cv2.dilate(
                                selected_mask(label_maps[nt], selected[nt]),
                                temporal_kernel,
                            )
                            > 0
                        )
                if not contexts:
                    continue
                context = np.logical_or.reduce(contexts)
                top = max(comp["raw"] for comp in all_components[t])
                for i, comp in enumerate(all_components[t]):
                    if i in selected[t] or comp["raw"] < 0.07 * top:
                        continue
                    region = label_maps[t] == (i + 1)
                    overlap = float((region & context).sum()) / max(
                        1, int(region.sum())
                    )
                    if overlap >= 0.03:
                        selected[t].add(i)
                        changed = True
        if not changed:
            break
    rois = []
    for t in range(T):
        roi = selected_mask(label_maps[t], selected[t])
        roi = cv2.morphologyEx(
            roi,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (odd(21 * scale), odd(21 * scale))
            ),
        )
        roi = (
            cv2.dilate(
                roi,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (odd(15 * scale), odd(15 * scale))
                ),
            )
            > 0
        )
        area = float(roi.mean())
        if not 0.005 <= area <= 0.90:
            raise ValueError(f"Implausible temporal ROI area {area:.3f} at t={t}")
        rois.append(roi)
    return rois


# Historical processing functions, command source line 952


def hist_quantile(a, q):
    h = np.bincount(a.ravel(), minlength=65536).astype(np.int64, copy=False)
    h[0] = 0
    s = int(h.sum())
    return (
        float(np.searchsorted(np.cumsum(h), q / 100 * (s - 1), "right")) if s else 0.0
    )


def observed_from_raw(v):
    bg = hist_quantile(v, 40)
    ce = hist_quantile(v, 65)
    sample = v[:: max(1, len(v) // 32), ::2, ::2].astype(np.float32)
    m = (sample > 0) & (sample <= ce)
    est = []
    dx = np.diff(sample, axis=2)
    mx = m[:, :, 1:] & m[:, :, :-1]
    dy = np.diff(sample, axis=1)
    my = m[:, 1:, :] & m[:, :-1, :]
    if mx.any():
        est.append(np.median(np.abs(dx[mx])) / 0.67448975 / np.sqrt(2))
    if my.any():
        est.append(np.median(np.abs(dy[my])) / 0.67448975 / np.sqrt(2))
    noise = max(float(np.median(est)) if est else 2, 0.5)
    o = v.astype(np.float32)
    o -= bg + 1.25 * noise
    np.maximum(o, 0, out=o)
    gaussian_filter(o, (0.25, 0.55, 0.55), mode="reflect", output=o)
    return o, bg, noise


def structural_support(observed, noise, coarse_roi):
    high = (observed > 6.0 * noise) & coarse_roi[None, :, :]
    labels, n = label(high, structure=structure)
    if n == 0:
        high = (observed > 4.0 * noise) & coarse_roi[None, :, :]
        labels, n = label(high, structure=structure)
    if n == 0:
        return np.broadcast_to(coarse_roi, observed.shape).copy(), 0, 0
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max())
    cut = max(20, int(0.20 * largest))
    keep = np.zeros(len(sizes), bool)
    keep[np.flatnonzero(sizes >= cut)] = True
    keep[0] = False
    support = keep[labels]
    del labels, high, keep
    low = (observed > 1.5 * noise) & coarse_roi[None, :, :]
    for _ in range(4):
        support = binary_dilation(support, structure=structure) & low
    return support, int(np.count_nonzero(sizes[1:] >= cut)), largest


def restore(v, coarse_roi, spacing, xy_size):
    observed, bg, noise = observed_from_raw(v)
    support, kept, largest = structural_support(observed, noise, coarse_roi)
    zsig = float(np.clip(0.18 / max(spacing, 1e-6), 0.5, 1.5))
    xysig = float(np.clip(0.08 / max(xy_size, 1e-6), 0.75, 1.35))
    psf = (zsig, xysig, xysig)
    estimate = np.maximum(observed.copy(), 1e-3)
    work = np.empty_like(estimate)
    corr = np.empty_like(estimate)
    for _ in range(6):
        gaussian_filter(estimate, psf, mode="reflect", output=work)
        np.maximum(work, 1e-3, out=work)
        np.divide(observed, work, out=work)
        gaussian_filter(work, psf, mode="reflect", output=corr)
        estimate *= corr
    gaussian_filter(estimate, (0.12, 0.28, 0.28), mode="reflect", output=work)
    del estimate
    # Feather the strict 3D support slightly so no hard artificial edge is introduced.
    support_float = support.astype(np.float32)
    gaussian_filter(
        support_float, (0.45, 0.65, 0.65), mode="nearest", output=support_float
    )
    support_float[support_float < 0.02] = 0
    np.divide(observed, 6 * noise, out=corr)
    np.clip(corr, 0, 1, out=corr)
    gaussian_filter(corr, (0.15, 0.35, 0.35), mode="reflect", output=corr)
    np.sqrt(corr, out=corr)
    work *= corr
    work *= support_float
    work[work < 0.02 * noise] = 0
    sample = work[:, ::4, ::4]
    pos = sample[sample > 0]
    scale = float(np.percentile(pos, 99.8)) if pos.size else 1.0
    scale = max(scale, 1e-6)
    work /= scale
    np.maximum(work, 0, out=work)
    np.power(work, 0.90, out=work)
    work *= 60000
    np.clip(work, 0, 65535, out=work)
    out = np.rint(work).astype(np.uint16)
    projection = support.max(0).astype(np.uint8) * 255
    return (
        out,
        projection,
        {
            "bg": bg,
            "noise": noise,
            "kept": kept,
            "largest": largest,
            "zero": float((out == 0).mean()),
            "support": float(support.mean()),
        },
    )


if __name__ == "__main__":
    main()
