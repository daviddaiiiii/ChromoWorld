#!/usr/bin/env python3
"""Conservative, calibrated, all-time XYZ cropping before/after registration.

Crop detection uses DAPI only. Outputs are exact native-grid index slices:
no interpolation, intensity restoration, component deletion, or mask relabeling.
Supports grayscale ZYX/TZYX and CZYX/ZCYX/TCZYX/TZCYX (all channels retained).
Ambiguous axes/calibration require explicit overrides. Existing files are never
overwritten. Companions must already be aligned; equal shape is not proof of it.
Dependencies: numpy, scipy, tifffile. No model, GPU, or project data required.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage as ndi
import tifffile as tf

VERSION = "temporal-native-grid-auto-crop-1.0"
UNITS = {"µm": 1., "μm": 1., "um": 1., "micron": 1., "microns": 1.,
         "nm": .001, "mm": 1000., "m": 1e6}


@dataclass(frozen=True)
class CropConfig:
    margin_xy_um: float = 3.
    margin_z_um: float = 1.5
    min_seed_voxels: int = 8
    retained_signal: float = .998
    square_xy: bool = True
    axes_to_crop: str = "xyz"

    def validate(self):
        if not all(math.isfinite(x) and x >= 0 for x in (self.margin_xy_um, self.margin_z_um)):
            raise ValueError("Physical margins must be finite and nonnegative")
        if self.min_seed_voxels < 1 or not .9 <= self.retained_signal <= 1.:
            raise ValueError("min_seed_voxels >= 1 and 0.9 <= retained_signal <= 1 required")
        if self.axes_to_crop not in ("xyz", "z"):
            raise ValueError("axes_to_crop must be xyz or z")


def _json_description(handle):
    try:
        value = json.loads(handle.pages[0].description or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def calibration(handle, override=None):
    meta = dict(handle.imagej_metadata or _json_description(handle))
    meta = {k: v for k, v in meta.items() if k not in ("Labels", "Info")}
    if handle.ome_metadata:
        root = ET.fromstring(handle.ome_metadata)
        pixels = next(e for e in root.iter() if e.tag.split("}")[-1] == "Pixels")
        for e in root.iter():
            if e.tag.split("}")[-1] == "Description" and e.text:
                try:
                    extra = json.loads(e.text)
                    if isinstance(extra, dict):
                        meta.update(extra)
                except ValueError:
                    pass
        if override is None:
            spacing = []
            for axis in "ZYX":
                value = pixels.get("PhysicalSize" + axis)
                unit = pixels.get("PhysicalSize" + axis + "Unit", "µm")
                if value is None or unit not in UNITS:
                    raise ValueError("OME physical Z/Y/X sizes in known units are required; use --spacing-zyx")
                spacing.append(float(value) * UNITS[unit])
        if pixels.get("TimeIncrement"):
            meta["finterval"] = float(pixels.get("TimeIncrement"))
            meta["time_unit"] = pixels.get("TimeIncrementUnit", "s")
        meta["channel_names"] = [e.get("Name", "") for e in pixels
                                 if e.tag.split("}")[-1] == "Channel"]
    elif override is None:
        unit = str(meta.get("unit", "")).lower()
        tags = handle.pages[0].tags
        if unit not in UNITS or "spacing" not in meta or not all(k in tags for k in ("XResolution", "YResolution")):
            raise ValueError("Calibrated Z/Y/X spacing is missing; use --spacing-zyx, never guess pixel units")
        xr, yr = tags["XResolution"].value, tags["YResolution"].value
        spacing = np.array([meta["spacing"], yr[1]/yr[0], xr[1]/xr[0]]) * UNITS[unit]
    spacing = np.asarray(override if override is not None else spacing, float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Spacing override must be positive finite Z Y X in µm")
    supplied_origin = meta.get("origin_zyx_um")
    if supplied_origin is None and isinstance(meta.get("source_geometry"), dict):
        supplied_origin = meta["source_geometry"].get("origin_zyx_um")
    origin = np.zeros(3) if supplied_origin is None else np.asarray(supplied_origin, float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("Invalid origin_zyx_um")
    return spacing, origin, supplied_origin is not None, meta


class Movie:
    def __init__(self, path, axes=None, spacing=None):
        self.path = Path(path).resolve()
        self.handle = tf.TiffFile(self.path)
        try:
            if len(self.handle.series) != 1:
                raise ValueError("Multi-series TIFF is ambiguous; export the intended movie as one series first")
            series = self.handle.series[0]
            self.axes = axes or series.axes
            self.shape, self.dtype = tuple(series.shape), np.dtype(series.dtype)
            if self.axes not in ("ZYX", "TZYX", "CZYX", "ZCYX", "TCZYX", "TZCYX") or len(self.axes) != len(self.shape):
                raise ValueError(f"Ambiguous/unsupported axes {self.axes}, {self.shape}; use an explicit --axes")
            if self.dtype.kind not in "uif" or any(n < 1 for n in self.shape):
                raise ValueError("Nonempty real grayscale TIFF required")
            self.sizes = dict(zip(self.axes, self.shape))
            self.tzyx = tuple(self.sizes.get(a, 1) for a in "TZYX")
            self.frame_axes = self.axes.removeprefix("T")
            self.frame_shape = tuple(self.sizes[a] for a in self.frame_axes)
            self.spacing, self.origin, self.has_origin, self.meta = calibration(self.handle, spacing)
            self.planes = math.prod(self.frame_shape[:-2])
            self.pages = series.pages
            self.mapping = None
            if len(self.pages) != self.tzyx[0] * self.planes:
                # Large contiguous ImageJ TIFFs can store the entire stack in
                # one page. Do not repeatedly decompress the whole movie.
                try:
                    self.mapping = tf.memmap(self.path, mode="r")
                except ValueError as error:
                    raise ValueError("Unsupported packed compressed TIFF; export plane-addressable TIFF first") from error
        except BaseException:
            self.handle.close()
            raise

    def close(self):
        if self.mapping is not None:
            self.mapping._mmap.close()
        self.handle.close()

    def read(self, t):
        if not 0 <= t < self.tzyx[0]:
            raise IndexError(t)
        if self.mapping is not None:
            return np.array(self.mapping[t] if "T" in self.axes else self.mapping)
        pages = self.pages[t*self.planes:(t+1)*self.planes]
        return np.stack([p.asarray() for p in pages]).reshape(self.frame_shape)

    def dapi(self, t, channel):
        if "C" in self.axes:
            if channel is None:
                if self.sizes["C"] != 1:
                    raise ValueError("Multi-channel TIFF needs an explicit --channel DAPI index")
                channel = 0
            if not 0 <= channel < self.sizes["C"]:
                raise ValueError("DAPI --channel outside TIFF channels")
            return np.take(self.read(t), channel, axis=self.frame_axes.index("C"))
        if channel not in (None, 0):
            raise ValueError("--channel given for a TIFF with no channel axis")
        return self.read(t)


def signal_bounds(volume, config):
    """Conservative signal support, not a segmentation/instance classifier."""
    x = volume.astype(np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite image pixels cannot define a trustworthy crop")
    shape = np.array(x.shape)
    full = (np.zeros(3, int), shape)
    if float(x.max()) == float(x.min()):
        return None, dict(status="constant_no_spatial_signal")
    stride = np.maximum(1, shape // np.array([32, 192, 192]))
    sample = x[tuple(slice(None, None, int(n)) for n in stride)]
    bg = float(np.percentile(sample, 40))
    differences = np.diff(sample, axis=-1).reshape(-1)
    noise = float(np.median(abs(differences - np.median(differences))) / (.67448975*np.sqrt(2))) if len(differences) else 0.
    span = float(x.max()-x.min())
    noise = max(noise, span*1e-6, np.finfo(np.float32).eps)
    smooth = ndi.gaussian_filter(x, (.45, .8, .8), mode="nearest")
    strong = smooth > bg + 6*noise
    components, _ = ndi.label(strong, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(components.reshape(-1))
    keep = sizes >= config.min_seed_voxels
    keep[0] = False
    seeds = keep[components]
    count = int(keep.sum())
    del strong, components
    if not count:
        # In low-confidence/noisy frames a smaller FOV is not justified.
        return full, dict(status="unresolved_keep_full_frame", background=bg, noise=noise)
    support = ndi.binary_propagation(seeds, mask=smooth > bg + 2*noise,
                                    structure=np.ones((3, 3, 3)))
    points = np.where(support)
    start = np.array([int(a.min()) for a in points])
    stop = np.array([int(a.max())+1 for a in points])
    del seeds, support, points
    # Guard dim extensions using background/noise-subtracted signal, NOT all
    # nonzero pixels. Divide the omission budget across six bounding faces;
    # the union bound then protects >= retained_signal in the whole 3D box.
    weights = np.maximum(smooth-bg-3*noise, 0)
    total = float(weights.sum(dtype=np.float64))
    tail = (1-config.retained_signal)/6
    if total > 0:
        for axis in range(3):
            hist = weights.sum(axis=tuple(k for k in range(3) if k != axis), dtype=np.float64)
            occupied = np.flatnonzero(hist > 0)
            cumulative = np.cumsum(hist)
            a = max(int(occupied[0]), int(np.searchsorted(cumulative, tail*total, side="left")))
            b = min(int(occupied[-1])+1, int(np.searchsorted(cumulative, (1-tail)*total, side="left"))+1)
            start[axis] = min(start[axis], a)
            stop[axis] = max(stop[axis], b)
        retained = float(weights[tuple(slice(int(a), int(b)) for a,b in zip(start,stop))].sum(dtype=np.float64)/total)
        if retained + 1e-7 < config.retained_signal:
            return full, dict(status="signal_guard_fallback_keep_full_frame", retained_signal=retained)
    else:
        retained = 1.
    return (start, stop), dict(status="signal_supported", seed_components=count,
                              background=bg, noise=noise, retained_signal=retained,
                              start_zyx=start.tolist(), stop_zyx=stop.tolist())


def detect_crop(movie, config, channel=None):
    config.validate()
    shape = np.array(movie.tzyx[1:])
    start, stop = shape.copy(), np.zeros(3, int)
    records = []
    for t in range(movie.tzyx[0]):
        bounds, detail = signal_bounds(movie.dapi(t, channel), config)
        records.append(dict(t=t, **detail))
        if bounds is not None:
            start = np.minimum(start, bounds[0]); stop = np.maximum(stop, bounds[1])
    if np.any(stop <= start):
        start, stop = np.zeros(3, int), shape.copy()
    margin = np.ceil(np.array([config.margin_z_um, config.margin_xy_um, config.margin_xy_um])/movie.spacing).astype(int)
    start = np.maximum(0, start-margin); stop = np.minimum(shape, stop+margin)
    if config.axes_to_crop == "z":
        start[1:], stop[1:] = 0, shape[1:]
    elif config.square_xy:
        side_um = float(np.max((stop-start)[1:]*movie.spacing[1:]))
        target = np.minimum(shape[1:], np.ceil(side_um/movie.spacing[1:]-1e-9).astype(int))
        midpoint = (start[1:]+stop[1:])/2
        square_start = np.floor(midpoint-target/2).astype(int)
        square_start = np.clip(square_start, 0, shape[1:]-target)
        start[1:], stop[1:] = square_start, square_start+target
    return start, stop, records


def _slices(movie, start, stop):
    bounds = dict(zip("ZYX", zip(start, stop)))
    return tuple(slice(int(bounds[a][0]), int(bounds[a][1])) if a in bounds else slice(None)
                 for a in movie.frame_axes)


def _write(movie, output, start, stop):
    expected = []
    output_shape = tuple(int(stop["ZYX".index(a)]-start["ZYX".index(a)]) if a in "ZYX" else movie.sizes[a]
                         for a in movie.axes)
    new_origin = movie.origin+start*movie.spacing
    meta = {k: movie.meta[k] for k in ("finterval", "fps", "time_unit", "channel_names") if k in movie.meta}
    meta.update(axes=movie.axes, unit="micron", spacing=float(movie.spacing[0]),
                pipeline=VERSION, origin_zyx_um=new_origin.tolist(),
                origin_reference=("aligned_reference_origin" if movie.meta.get("crop_origin_inherited_from") else
                                  "source_origin" if movie.has_origin else "local_source_voxel_grid"),
                crop_start_zyx=start.tolist(), crop_stop_zyx=stop.tolist())

    def planes():
        for t in range(movie.tzyx[0]):
            cropped = np.ascontiguousarray(movie.read(t)[_slices(movie, start, stop)])
            expected.append(hashlib.sha256(cropped.tobytes()).hexdigest())
            yield from cropped.reshape(-1, *cropped.shape[-2:])

    tf.imwrite(output, planes(), shape=output_shape, dtype=movie.dtype, metadata=meta,
               photometric="minisblack", compression="zlib", ome=False,
               bigtiff=math.prod(output_shape)*movie.dtype.itemsize > 2**32-2**25,
               resolution=(1/movie.spacing[2], 1/movie.spacing[1]), resolutionunit="NONE")
    checked = Movie(output)
    try:
        if checked.axes != movie.axes or checked.shape != output_shape or checked.dtype != movie.dtype:
            raise AssertionError("Cropped TIFF axes/shape/dtype changed unexpectedly")
        # TIFF XY resolutions are bounded integer rationals. Decimal OME
        # spacings (e.g. 0.06 um) need not round-trip to the last double bit.
        if not np.allclose(checked.spacing, movie.spacing, rtol=1e-8, atol=0) or not np.allclose(checked.origin, new_origin):
            raise AssertionError("Cropped TIFF physical grid changed unexpectedly")
        for t, digest in enumerate(expected):
            if hashlib.sha256(np.ascontiguousarray(checked.read(t)).tobytes()).hexdigest() != digest:
                raise AssertionError(f"Native crop pixels changed at t={t}")
    finally:
        checked.close()
    return dict(input=str(movie.path), output_shape=list(output_shape), axes=movie.axes,
                dtype=str(movie.dtype), spacing_zyx_um=movie.spacing.tolist(),
                source_origin_zyx_um=movie.origin.tolist(), output_origin_zyx_um=new_origin.tolist(),
                source_origin_was_explicit=movie.has_origin,
                origin_inherited_from=movie.meta.get("crop_origin_inherited_from"),
                verified_frame_sha256=expected)


def crop_file(source, output_dir, config=None, *, channel=None, axes=None, spacing=None, apply_to=()):
    config = config or CropConfig()
    config.validate()
    paths = [Path(source).resolve(), *(Path(p).resolve() for p in apply_to)]
    output_dir = Path(output_dir).resolve()
    outputs = [output_dir/p.name for p in paths]
    report_path = output_dir/(paths[0].stem+"_crop.json")
    if len(set(paths)) != len(paths) or len(set(outputs)) != len(outputs):
        raise ValueError("Duplicate source or companion output filenames")
    if any(out in paths for out in outputs):
        raise ValueError("Crop output must not overwrite any input")
    if any(p.exists() for p in [*outputs, report_path]):
        raise FileExistsError("Crop output/report already exists; use a new directory")
    with ExitStack() as stack:
        movies = []
        stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
        for p in paths:
            movie = Movie(p, axes, spacing);stack.callback(movie.close);movies.append(movie)
        reference = movies[0]
        for other in movies[1:]:
            if other.tzyx != reference.tzyx or not np.allclose(other.spacing, reference.spacing):
                raise ValueError("Companion must have the same TZYX grid and physical spacing")
            if reference.has_origin and other.has_origin and not np.allclose(other.origin, reference.origin):
                raise ValueError("Companion physical origin differs; register it before cropping")
            if other.has_origin and not reference.has_origin:
                raise ValueError("Companion has an origin but DAPI does not; supply corresponding DAPI calibration/origin first")
            if reference.has_origin and not other.has_origin:
                # A common index crop must retain a common physical grid.
                # Only an explicitly asserted aligned companion may inherit
                # the DAPI origin, with that assumption recorded in the report.
                other.origin = reference.origin.copy()
                other.meta["crop_origin_inherited_from"] = str(reference.path)
        start, stop, frames = detect_crop(reference, config, channel)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Stage every companion before publishing any. Never modify source
        # arrays and never resize labels to make an incompatible grid fit.
        with tempfile.TemporaryDirectory(prefix=".auto-crop-", dir=output_dir) as tmp:
            temporary = Path(tmp)
            records = [_write(m, temporary/out.name, start, stop) for m,out in zip(movies,outputs)]
            if stats != {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}:
                raise RuntimeError("Input changed during cropping")
            for out,record in zip(outputs,records):
                record["output"] = str(out)
            report = dict(version=VERSION, source=str(paths[0]), config=asdict(config),
                          channel_for_detection=channel, crop_start_zyx=start.tolist(), crop_stop_zyx=stop.tolist(),
                          source_shape_tzyx=list(reference.tzyx), output_shape_tzyx=[reference.tzyx[0],*(stop-start).tolist()],
                          timepoints_preserved=True, pixel_operation="exact index slice; no resampling",
                          companion_alignment_is_user_asserted=True, frames=frames, files=records)
            staged_report = temporary/report_path.name
            staged_report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            # Output names are reserved above and rechecked against concurrent
            # writers. A publication failure removes only our own new files.
            if any(p.exists() for p in [*outputs, report_path]):
                raise FileExistsError("Crop destination changed while processing")
            published = []
            try:
                for out in [*outputs, report_path]:
                    os.link(temporary/out.name, out)  # exclusive, no overwrite
                    published.append(out)
            except BaseException:
                for out in published:
                    out.unlink()
                raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One TIFF or a flat TIFF directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*", help="Directory glob, filtered to .tif/.tiff")
    parser.add_argument("--mode", choices=("raw", "registered"), default="raw",
                        help="Default XY margin: raw=3 µm; registered=2 µm; neither mode estimates motion")
    parser.add_argument("--margin-xy-um", type=float)
    parser.add_argument("--margin-z-um", type=float, default=1.5)
    parser.add_argument("--min-seed-voxels", type=int, default=8)
    parser.add_argument("--retained-signal", type=float, default=.998)
    parser.add_argument("--axes-to-crop", choices=("xyz", "z"), default="xyz")
    parser.add_argument("--no-square-xy", action="store_true")
    parser.add_argument("--axes", help="Explicit axes when the TIFF axes are ambiguous; also applies to companions")
    parser.add_argument("--spacing-zyx", nargs=3, type=float, metavar=("Z_UM", "Y_UM", "X_UM"))
    parser.add_argument("--channel", type=int, help="Zero-based DAPI detection channel; ALL channels are written")
    parser.add_argument("--apply-to", type=Path, action="append", default=[], help="Already aligned mask/Tag TIFF; single input only")
    args = parser.parse_args()
    config = CropConfig(margin_xy_um=args.margin_xy_um if args.margin_xy_um is not None else (3. if args.mode=="raw" else 2.),
                        margin_z_um=args.margin_z_um, min_seed_voxels=args.min_seed_voxels,
                        retained_signal=args.retained_signal, square_xy=not args.no_square_xy, axes_to_crop=args.axes_to_crop)
    sources = [args.input] if args.input.is_file() else sorted(p for p in args.input.glob(args.pattern)
                 if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in (".tif", ".tiff"))
    if not sources or (args.apply_to and len(sources) != 1):
        parser.error("Need input TIFF(s); --apply-to requires a single source TIFF")
    try:
        config.validate()
        for source in sources:
            report = crop_file(source, args.output_dir, config, channel=args.channel, axes=args.axes,
                               spacing=args.spacing_zyx, apply_to=args.apply_to)
            print(json.dumps({"sample":source.name,"crop_start_zyx":report["crop_start_zyx"],
                              "output_shape_tzyx":report["output_shape_tzyx"]}), flush=True)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
