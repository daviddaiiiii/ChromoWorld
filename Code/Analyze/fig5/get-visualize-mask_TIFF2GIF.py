#!/usr/bin/env python3
"""Shaded fixed front/back mask-only GIFs with physical scale bars.

Left: visible labels from +Z. Right: visible labels from -Z, horizontally flipped
to match a 180-degree turn around Y (not a mirrored copy of the front image).
Every source timepoint is shown, with exactly 6 seconds total playback by default.
Same ID has fixed base RGB, without claiming tracked identity. Surface shading
changes brightness only, not instance geometry. A movie-wide GIF palette avoids
frame-dependent color flicker; the 256-color limit still approximates hues/shades.
No titles/timepoint text. The only annotation is the calibrated scale-bar label.
Shading is a visualization of mask depth, NOT measured fluorescence or recovery
of true physical shape. Source TIFFs unchanged.
Supports direct TIFFs and chr3/chr18/others folders; excludes hidden backups.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

CATEGORIES = ("chr3", "chr18", "others")
RENDER_VERSION = "shaded-front-back-scalebar-2.1"


def gif_frame_durations(steps: int, duration_seconds: float = 6.0,
                        fps: float | None = None) -> list[int]:
    """GIF delays in 10-ms units; default total playback time is exactly 6 s."""
    if fps is not None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        return [max(10, round(100 / fps) * 10)] * steps
    total_units = round(duration_seconds * 100)
    if total_units < steps:
        raise ValueError("Duration must allow at least 10 ms per GIF frame")
    return [(round((i + 1) * total_units / steps) - round(i * total_units / steps)) * 10
            for i in range(steps)]


def instance_colors(ids: np.ndarray) -> np.ndarray:
    """Direct ID -> RGB using golden-angle hues and varied saturation/brightness."""
    lut = np.zeros((int(ids.max(initial=0)) + 1, 3), dtype=np.uint8)
    used = {(0, 0, 0)}
    for rid in ids:
        rid = int(rid)
        if rid == 0:
            continue
        hue = (rid * 0.618033988749895) % 1.0
        saturation = 0.65 + 0.30 * ((rid * 37) % 101) / 100.0
        value = 0.78 + 0.22 * ((rid * 53) % 101) / 100.0
        rgb = tuple(round(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value))
        while rgb in used:
            rgb = ((rgb[0] + 1) % 256, rgb[1], rgb[2])
        used.add(rgb)
        lut[rid] = rgb
    return lut


def _load_mask(src: Path) -> np.ndarray:
    with tifffile.TiffFile(src) as tf:
        series = tf.series[0]
        axes = series.axes
        arr = series.asarray()
    if axes in ("TZYX", "ZYX"):
        mask = arr if axes == "TZYX" else arr[None]
    elif axes == "TZCYX" and arr.shape[2] == 1:
        mask = arr[:, :, 0]
    elif axes == "TCZYX" and arr.shape[1] == 1:
        mask = arr[:, 0]
    elif axes in ("QZYX", "QQYX") and arr.ndim == 4:
        mask = arr  # Legacy generic-axis files are stored as TZYX.
    else:
        raise ValueError(f"Unsupported label axes/shape: {axes}, {arr.shape}")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Expected nonnegative integer instance labels")
    if np.issubdtype(mask.dtype, np.signedinteger) and np.any(mask < 0):
        raise ValueError("Expected nonnegative integer instance labels")
    return mask


def fixed_front_back(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface visibility by depth, NEVER numeric max-label projection.

    All Z slices participate: no 3D resampling and no rotating/cropping the FOV.
    An all-background ray naturally returns zero from either end of the volume.
    """
    occupied = labels > 0
    back_depth = occupied.argmax(axis=0)
    front_depth = labels.shape[0] - 1 - occupied[::-1].argmax(axis=0)
    front = np.take_along_axis(labels, front_depth[None], axis=0)[0]
    back = np.take_along_axis(labels, back_depth[None], axis=0)[0]
    return front, back[:, ::-1]


def _spacing_um(src: Path) -> np.ndarray:
    """Respect ImageJ/shaped TIFF calibration, not the mask's array dimensions.

ImageJ metadata declares the length unit; its X/Y resolution tags are pixels
per that unit. For these files ResolutionUnit alone is not authoritative.
"""
    with tifffile.TiffFile(src) as tf:
        meta = dict(tf.imagej_metadata or {})
        if not meta:
            try:
                meta = json.loads(tf.pages[0].description or "{}")
            except (ValueError, TypeError):
                meta = {}
        factors = {"micron": 1., "microns": 1., "um": 1., "µm": 1., "μm": 1.,
                   "nm": .001, "mm": 1000.}
        factor = factors.get(str(meta.get("unit", "")).lower())
        tags = tf.pages[0].tags
        if factor is None or "spacing" not in meta or not all(k in tags for k in ("XResolution", "YResolution")):
            raise ValueError(f"No trustworthy physical XYZ calibration in {src}; cannot invent a scale bar")
        xr, yr = tags["XResolution"].value, tags["YResolution"].value
        spacing = np.array([float(meta["spacing"]), yr[1]/yr[0], xr[1]/xr[0]])*factor
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError(f"Invalid physical calibration in {src}: {spacing}")
    return spacing


def _surface_views(labels):
    occupied = labels > 0
    back_z = occupied.argmax(axis=0)
    front_z = labels.shape[0]-1-occupied[::-1].argmax(axis=0)
    front = np.take_along_axis(labels, front_z[None], axis=0)[0]
    back = np.take_along_axis(labels, back_z[None], axis=0)[0]
    return (front, front_z), (back[:, ::-1], back_z[:, ::-1])


def _shade_view(labels, z_indices, colors, spacing, z_bounds, back=False):
    """Estimate calibrated depth normals separately inside each visible ID.

Depth smoothing never mixes different IDs or paints background. Mild edge
ambient occlusion provides readable silhouette shading; bounded illumination
keeps small/distant instances visible. Full Z participates in surface selection.
"""
    sz, sy, sx = spacing
    result = np.zeros((*labels.shape, 3), np.uint8)
    light = np.array([-.45, -.55, .70]); light /= np.linalg.norm(light)
    for identity, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        binary = np.pad(labels[sl] == identity, 2)
        depth = np.pad(z_indices[sl].astype(float)*sz*(-1 if back else 1), 2)
        weight = ndi.gaussian_filter(binary.astype(float), 1.1)
        height = ndi.gaussian_filter(depth*binary, 1.1)/np.maximum(weight, 1e-8)
        edge_distance = ndi.distance_transform_edt(binary, sampling=(sy, sx))
        # A shallow illustrative bevel makes voxel silhouettes read as volumes.
        # It changes lighting normals only, NOT positions or projected labels.
        bevel = np.minimum(ndi.gaussian_filter(edge_distance, .8), 2.4*math.sqrt(sx*sy))
        height += .65*bevel
        dy, dx = np.gradient(height, sy, sx)
        normal = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        diffuse = np.maximum(0., normal @ light)
        edge_occlusion = .80+.20*np.clip(edge_distance/(2.4*math.sqrt(sx*sy)), 0, 1)
        near = (z_indices[sl]-z_bounds[0])/max(1, z_bounds[1]-z_bounds[0])
        if back:
            near = 1-near
        gain = (.46+.54*diffuse[2:-2, 2:-2]) * edge_occlusion[2:-2, 2:-2] * (.90+.10*near)
        gain = np.clip(gain, .65, 1.)
        rgb = np.rint(colors[identity]*gain[..., None]).astype(np.uint8)
        visible = labels[sl] == identity
        result[sl][visible] = rgb[visible]
    return result


def _scale_bar(width, pixel_um):
    field_um = width*pixel_um
    exponent = math.floor(math.log10(field_um))
    choices = [v*10.**e for e in range(exponent-3, exponent+1) for v in (1, 2, 5)]
    valid = [v for v in choices if 16 <= v/pixel_um <= .30*width]
    if not valid:
        valid = [v for v in choices if v/pixel_um <= .30*width]
    length_um = min(valid, key=lambda v:abs(math.log(v/(.22*field_um))))
    pixels = max(1, round(length_um/pixel_um))
    return {"length_um": length_um, "length_display_pixels": pixels,
            "actual_drawn_length_um": pixels*pixel_um,
            "display_pixel_size_um": pixel_um, "label": f"{length_um:g} µm"}


def _draw_bar(canvas, origin_x, height, bar, font):
    draw = ImageDraw.Draw(canvas)
    x, y = origin_x+12, height+7
    end = x+bar["length_display_pixels"]
    draw.line((x, y, end, y), fill="white", width=3)
    label = bar["label"]
    box = draw.textbbox((0, 0), label, font=font)
    label_width = box[2]-box[0]
    draw.text(((x+end-label_width)/2, height+12), label, fill="white", font=font)


def _font():
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, 12)
        except OSError:
            pass
    return ImageFont.load_default()


def _mask_gif(src: Path, dst: Path, args) -> dict:
    started = time.monotonic()
    mask = _load_mask(src)
    calibration_source = src
    try:
        spacing = _spacing_um(src)
    except ValueError:
        reference = getattr(args, "calibration_reference", None)
        if reference is None:
            raise
        reference = Path(reference)
        with tifffile.TiffFile(reference) as tf:
            reference_shape = tf.series[0].shape
        if len(reference_shape)==3 and mask.shape[0]==1:
            reference_shape=(1,*reference_shape)
        if tuple(reference_shape) != tuple(mask.shape):
            raise ValueError(f"Calibration reference has a different grid: {reference_shape} vs {mask.shape}")
        spacing = _spacing_um(reference)
        calibration_source = reference
    present = np.zeros(int(mask.max(initial=0)) + 1, dtype=bool)
    occupied_z = np.zeros(mask.shape[1], bool)
    for volume in mask:
        counts = np.bincount(volume.reshape(-1).astype(np.int64, copy=False))
        present[:len(counts)] |= counts > 0
        occupied_z |= np.any(volume > 0, axis=(1, 2))
    ids = np.flatnonzero(present)
    colors = instance_colors(ids)
    durations = gif_frame_durations(mask.shape[0], args.duration_seconds, args.fps)
    z_used = np.flatnonzero(occupied_z)
    z_bounds = (int(z_used[0]), int(z_used[-1])) if len(z_used) else (0, mask.shape[1]-1)

    # Resize ONLY the final 2D label projections; never drop a Z slice.
    height, width = mask.shape[-2:]
    scale = min(1.0, args.max_size / max(height, width)) if args.max_size else 1.0
    out_height, out_width = max(1, round(height*scale)), max(1, round(width*scale))
    yy = np.floor((np.arange(out_height)+.5)*height/out_height).astype(int)
    xx = np.floor((np.arange(out_width)+.5)*width/out_width).astype(int)
    out_height, out_width = len(yy), len(xx)
    footer, separator = 28, 12
    font = _font()
    bar = _scale_bar(out_width, spacing[2]*width/out_width)
    frames, schedule = [], []
    for t, volume in enumerate(mask):
        (front, front_z), (back, back_z) = _surface_views(volume)
        front_rgb = _shade_view(front, front_z, colors, spacing, z_bounds)[np.ix_(yy, xx)]
        back_rgb = _shade_view(back, back_z, colors, spacing, z_bounds, back=True)[np.ix_(yy, xx)]
        canvas = Image.new("RGB", (2 * out_width + separator, out_height + footer), "black")
        canvas.paste(Image.fromarray(front_rgb), (0, 0))
        canvas.paste(Image.fromarray(back_rgb), (out_width + separator, 0))
        draw = ImageDraw.Draw(canvas)
        _draw_bar(canvas, 0, out_height, bar, font)
        _draw_bar(canvas, out_width+separator, out_height, bar, font)
        draw.line((out_width + separator // 2, 0, out_width + separator // 2, out_height + footer),
                  fill=(80, 80, 80))
        frames.append(canvas)
        schedule.append({"frame": t, "t": t, "left_view": "+Z",
                         "right_view": "-Z, equivalent to Y=180 degrees"})
    dst.parent.mkdir(parents=True, exist_ok=True)
    # One palette for the entire movie, both panels and all brightness levels.
    sample = Image.new("RGB", (frames[0].width, sum(f.height for f in frames)))
    offset = 0
    for f in frames:
        sample.paste(f, (0, offset)); offset += f.height
    palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    temp = dst.with_suffix(".tmp.gif")
    frames[0].save(temp, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=2)
    with Image.open(temp) as result:
        encoded_frames = result.n_frames
        if encoded_frames > mask.shape[0]:
            raise RuntimeError("Unexpected extra GIF frames")
        actual_duration = 0
        for i in range(result.n_frames):
            result.seek(i)
            actual_duration += result.info.get("duration", 0)
        if actual_duration != sum(durations):
            raise RuntimeError(f"Wrong GIF duration: {actual_duration} ms")
    temp.replace(dst)
    return {"source": str(src), "gif": str(dst), "shape_tzyx": list(mask.shape),
            "renderer_version": RENDER_VERSION, "shading": "calibrated_surface_normals_visual_bevel_and_edge_occlusion",
            "surface_bevel_is_illustrative": True,
            "lighting_is_illustrative_not_intensity": True, "titles_or_time_text": False,
            "spacing_zyx_um": spacing.tolist(), "scale_bar": bar,
            "calibration_source": str(calibration_source),
            "palette_mode": "shared_movie_palette_256_colors", "encoded_gif_frames": encoded_frames,
            "identical_frames_may_coalesce_without_losing_elapsed_time": True,
            "panel_shape_yx": [out_height, out_width], "color_mode": "instances",
            "view_mode": "fixed_front_back", "rotation": False,
            "nonzero_ids": int(np.count_nonzero(ids)),
            "unique_instance_rgb": len(set(map(tuple, colors[ids[ids > 0]]))),
            "duration_ms": sum(durations), "frame_durations_ms": durations,
            "frames": schedule, "seconds": round(time.monotonic() - started, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--folders", nargs="+", required=True)
    ap.add_argument("--duration-seconds", type=float, default=6.0,
                    help="Total GIF playback duration (default: 6 seconds)")
    ap.add_argument("--fps", type=float, default=None,
                    help="Optional frame-rate override instead of fixed total duration")
    ap.add_argument("--max-size", type=int, default=300, help="Maximum 2D panel axis size; 0 = full resolution")
    ap.add_argument("--visualizer", type=Path, help="Deprecated; no external visualizer is needed")
    ap.add_argument("--num-angles", type=int, help="Deprecated and ignored; all source timepoints are shown without rotation")
    ap.add_argument("--gpu-id", type=int, help="Deprecated; fixed projections run on CPU")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report-json",type=Path,help="Optional explicit audit JSON; none is created by default")
    ap.add_argument("--calibration-reference-root", type=Path,
                    help="Matching *-DAPI TIFFs for masks without calibration; exact TZYX grid required")
    args = ap.parse_args()
    if args.report_json and args.report_json.suffix.lower()!='.json':ap.error("Report filename must end with .json")
    if (args.fps is not None and args.fps <= 0) or args.duration_seconds <= 0 or args.max_size < 0:
        ap.error("fps/duration must be positive and max-size nonnegative")
    if not 0 <= args.worker_index < args.workers:
        ap.error("worker-index must be in [0, workers)")
    jobs = []
    for folder in args.folders:
        base = args.input_root / folder
        base = base / "tiff" if (base / "tiff").is_dir() else base
        for directory in [base, *(base / c for c in CATEGORIES)]:
            if directory.is_dir():
                for src in sorted(directory.iterdir()):
                    if src.is_file() and not src.name.startswith('.') and src.suffix.lower() in (".tif", ".tiff"):
                        dst = args.output_root / folder / "gif" / src.relative_to(base).parent / f"{src.stem}_time_front_back_mask.gif"
                        jobs.append((src, dst))
    jobs = sorted(jobs)
    if not jobs:
        ap.error("No matching mask TIFFs")
    if len({dst for _, dst in jobs}) != len(jobs):
        raise ValueError("Duplicate sample GIF destinations across categories")
    jobs = [j for i, j in enumerate(jobs) if i % args.workers == args.worker_index]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"worker {args.worker_index}: {len(jobs)} fixed front/back GIFs; independent instance colors", flush=True)
    reports, errors = [], []
    references = {}
    if args.calibration_reference_root:
        for path in args.calibration_reference_root.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in ('.tif','.tiff') or not path.stem.endswith('-DAPI'):continue
            if any(part.startswith('.') for part in path.relative_to(args.calibration_reference_root).parts):continue
            name=path.stem.removesuffix('-DAPI')
            if name in references:raise ValueError(f'Duplicate calibration reference: {name}')
            references[name]=path
    for i, (src, dst) in enumerate(jobs, 1):
        if dst.exists() and not args.force:
            print(f"[{i}/{len(jobs)}] skip {src.name}", flush=True)
            continue
        try:
            name = src.stem
            previous = None
            while name != previous:
                previous = name
                for suffix in ('_shape_postprocessed','_shape','_tracked','-DAPI'):
                    name=name.removesuffix(suffix)
            args.calibration_reference = references.get(name)
            report = _mask_gif(src, dst, args)
            reports.append(report)
            print(f"[{i}/{len(jobs)}] {src.relative_to(args.input_root)}: {len(report['frames'])} t, {report['seconds']}s", flush=True)
        except Exception as exc:
            errors.append({"source": str(src), "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {errors[-1]}", flush=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True,exist_ok=True)
        args.report_json.write_text(json.dumps({"results":reports,"errors":errors},indent=2),encoding='utf-8')
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
