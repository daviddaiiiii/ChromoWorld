#!/usr/bin/env python3
"""Fixed +/-Z grayscale image GIFs with calibrated scale bars and no titles.

MIP is invariant to reversing the Z ray direction: the opposite view
is its horizontal mirror (a Y=180-degree camera). Default volume rendering
uses explicitly illustrative emission/absorption; it is NOT quantitative
fluorescence or reconstructed physical opacity. Both methods use every Z and T.
Movie-wide input contrast, shared output contrast/gamma, and a fixed palette avoid brightness
and palette flicker. Source TIFFs are read-only; no segmentation is consulted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

# The sibling CLI filename contains hyphens, so load its shared physical/display
# helpers by path. No imports from the author's external algorithm directory.
_helper_path = Path(__file__).with_name('get-visualize-mask_TIFF2GIF.py')
_helper_spec = importlib.util.spec_from_file_location('chromoworld_mask_gif_helpers', _helper_path)
_helpers = importlib.util.module_from_spec(_helper_spec)
sys.modules[_helper_spec.name] = _helpers
_helper_spec.loader.exec_module(_helpers)
CATEGORIES = _helpers.CATEGORIES
_spacing_um, _scale_bar, _draw_bar, _font = (
    _helpers._spacing_um, _helpers._scale_bar, _helpers._draw_bar, _helpers._font)
gif_frame_durations = _helpers.gif_frame_durations

VERSION = "image-fixed-front-back-scalebar-1.3"


def _load_image(src):
    with tifffile.TiffFile(src) as tf:
        axes = tf.series[0].axes
    try:
        arr = tifffile.memmap(src, series=0, mode="r")
    except (ValueError, TypeError):
        arr = tifffile.imread(src)
    if axes == "TZYX":
        pass
    elif axes == "ZYX":
        arr = arr[None]
    elif axes == "TZCYX" and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif axes == "TCZYX" and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif axes in ("QZYX", "QQYX") and arr.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported image axes {axes}, shape {arr.shape}")
    return arr


def _volume_view(volume, lo, hi, spacing_z, reverse, strength=2.):
    intensity = np.zeros(volume.shape[-2:], np.float32)
    transmission = np.ones_like(intensity)
    order = range(len(volume)-1, -1, -1) if not reverse else range(len(volume))
    for z in order:
        value = np.clip((volume[z].astype(np.float32)-lo)/(hi-lo), 0., 1.)
        alpha = -np.expm1(-strength*spacing_z*value**2)
        intensity += transmission*alpha*value
        transmission *= 1-alpha
    return intensity[:, ::-1] if reverse else intensity


def _display_transfer(views, low_pct, high_pct, gamma):
    """One monotonic display-only transfer across every time and both faces."""
    finite = views[np.isfinite(views)]
    if not finite.size:
        raise ValueError("No finite rendered intensities")
    lo, hi = np.percentile(finite, [low_pct, high_pct]).astype(float)
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.
    mapped = np.clip((views-lo)/(hi-lo), 0., 1.)
    return np.power(mapped, gamma), [lo, hi]


def _save_gif(frames, dst, durations):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Exact grayscale values; no independently chosen adaptive palettes.
    palette = np.repeat(np.arange(256, dtype=np.uint8), 3).tolist()
    encoded = []
    for frame in frames:
        grayscale = frame.convert("L")
        indexed = Image.frombytes("P", grayscale.size, grayscale.tobytes())
        indexed.putpalette(palette); encoded.append(indexed)
    temp = dst.with_suffix(".tmp.gif")
    encoded[0].save(temp, save_all=True, append_images=encoded[1:], duration=durations,
                    loop=0, optimize=False, disposal=2)
    with Image.open(temp) as im:
        count = im.n_frames; total = 0
        for t in range(count):
            im.seek(t); total += im.info.get("duration", 0)
        if total != sum(durations):
            raise AssertionError("GIF playback duration mismatch")
    temp.replace(dst)
    return count


def _image_gif(src, dst, args):
    start = time.monotonic(); image = _load_image(src); spacing = _spacing_um(src)
    # Only a small 2D projection stack is allocated, even for large raw movies.
    mips = np.stack([np.max(volume, axis=0) for volume in image])
    finite = mips[np.isfinite(mips)]
    high_pct = args.contrast_high_pct
    lo, hi = np.percentile(finite, [args.contrast_low_pct, high_pct]).astype(float)
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo+1.
    height, width = image.shape[-2:]
    ratio = min(1., args.max_size/max(height, width)) if args.max_size else 1.
    out_h, out_w = max(1, round(height*ratio)), max(1, round(width*ratio))
    yy = np.floor((np.arange(out_h)+.5)*height/out_h).astype(int)
    xx = np.floor((np.arange(out_w)+.5)*width/out_w).astype(int)
    bar = _scale_bar(out_w, spacing[2]*width/out_w)
    footer, separator = 28, 12; font = _font()
    durations = gif_frame_durations(len(image), args.duration_seconds)
    # Store only the displayed 2D faces, never a second full 4D source stack.
    # All faces are rendered before choosing one fixed movie-wide transfer.
    rendered = np.empty((len(image), 2, out_h, out_w), dtype=np.float32)
    for t, volume in enumerate(image):
        if args.projection == "mip":
            front = np.clip((mips[t].astype(float)-lo)/(hi-lo), 0., 1.)
            front = front[np.ix_(yy, xx)]
            back = front[:, ::-1]
        else:
            # Independent orthographic rays: selecting displayed XY rays before
            # compositing is exactly equivalent to selecting them afterwards.
            # Every Z plane is retained; no 3D interpolation or source edit.
            displayed_rays = volume[:, yy[:, None], xx]
            front = _volume_view(displayed_rays, lo, hi, spacing[0], False, args.opacity_strength)
            back = _volume_view(displayed_rays, lo, hi, spacing[0], True, args.opacity_strength)
        rendered[t, 0] = front
        rendered[t, 1] = back
    if args.projection == "volume":
        display_high_pct = args.display_high_pct
        rendered, display_range = _display_transfer(
            rendered, args.display_low_pct, display_high_pct, args.display_gamma)
        display_gamma = args.display_gamma
    else:
        display_range = [0., 1.]; display_gamma = 1.; display_high_pct = 100.
    frames=[]
    for faces in rendered:
        panels=[]
        for view in faces:
            gray=np.rint(255*view).astype(np.uint8)
            panels.append(Image.fromarray(gray).convert("RGB"))
        canvas=Image.new("RGB", (2*out_w+separator,out_h+footer),"black")
        canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(out_w+separator,0))
        _draw_bar(canvas,0,out_h,bar,font);_draw_bar(canvas,out_w+separator,out_h,bar,font)
        ImageDraw.Draw(canvas).line((out_w+separator//2,0,out_w+separator//2,out_h+footer),fill=(80,80,80))
        frames.append(canvas)
    count = _save_gif(frames,dst,durations)
    return dict(source=str(src),gif=str(dst),renderer_version=VERSION,shape_tzyx=list(image.shape),
        panel_shape_yx=[out_h,out_w],view_mode="fixed_front_back",rotation=False,titles_or_time_text=False,
        projection=args.projection,front_back_mip_equivalence=args.projection=="mip",
        all_z_and_t_used=True,duration_ms=sum(durations),frame_durations_ms=durations,
        encoded_gif_frames=count,source_timepoints=len(image),spacing_zyx_um=spacing.tolist(),scale_bar=bar,
        contrast_range=[lo,hi],contrast_percentiles=[args.contrast_low_pct,high_pct],
        contrast_policy="one fixed range per movie, same for both views",
        display_transfer_range=display_range,display_gamma=display_gamma,
        display_transfer_percentiles=[args.display_low_pct,display_high_pct] if args.projection=="volume" else None,
        display_transfer_policy="one monotonic output transfer shared across all T and both faces",
        image_values_modified=False,display_is_not_quantitative_fluorescence=True,
        opacity_strength_per_um=args.opacity_strength if args.projection=="volume" else None,
        palette_mode="fixed_256_level_grayscale",seconds=round(time.monotonic()-start,3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",required=True,type=Path);p.add_argument("--output-root",required=True,type=Path)
    p.add_argument("--folders",nargs="+",required=True)
    p.add_argument("--projection",choices=["mip","volume"],default="volume")
    p.add_argument("--opacity-strength",type=float,default=2.)
    p.add_argument("--contrast-low-pct",type=float,default=1.)
    p.add_argument("--contrast-high-pct",type=float,default=99.)
    p.add_argument("--display-low-pct",type=float,default=1.)
    p.add_argument("--display-high-pct",type=float,default=99.5)
    p.add_argument("--display-gamma",type=float,default=.7)
    p.add_argument("--duration-seconds",type=float,default=6.);p.add_argument("--max-size",type=int,default=300)
    p.add_argument("--workers",type=int,default=1);p.add_argument("--worker-index",type=int,default=0)
    p.add_argument("--limit",type=int,default=0);p.add_argument("--force",action="store_true")
    p.add_argument("--report-json",type=Path,help="Optional explicit audit JSON; none is created by default")
    args=p.parse_args()
    if args.report_json and args.report_json.suffix.lower()!='.json':p.error("Report filename must end with .json")
    if not 0 <= args.worker_index < args.workers:p.error("Invalid worker index")
    if not 0 <= args.contrast_low_pct < args.contrast_high_pct <= 100:p.error("Invalid contrast percentiles")
    if not 0 <= args.display_low_pct < args.display_high_pct <= 100:p.error("Invalid display percentiles")
    if args.display_gamma<=0:p.error("Display gamma must be positive")
    if args.duration_seconds<=0 or args.max_size<0 or args.opacity_strength<=0:p.error("Invalid rendering parameters")
    jobs=[]
    for group in args.folders:
        base=args.input_root/group
        base=base/"tiff" if (base/"tiff").is_dir() else base
        for d in [base,*(base/c for c in CATEGORIES)]:
            if not d.is_dir():continue
            for src in d.iterdir():
                if src.is_file() and not src.name.startswith('.') and src.suffix.lower() in (".tif",".tiff"):
                    if re.search(r'(?:^|[-_])tag(?:[-_]|$)',src.stem,re.I):continue
                    jobs.append((src,args.output_root/group/"gif"/src.relative_to(base).parent/f"{src.stem}_time_front_back_raw.gif"))
    jobs=sorted(jobs)
    if not jobs:p.error("No matching non-Tag image TIFFs")
    if len({dst for _,dst in jobs})!=len(jobs):raise ValueError("Duplicate GIF destinations")
    jobs=[job for i,job in enumerate(jobs) if i%args.workers==args.worker_index]
    if args.limit:jobs=jobs[:args.limit]
    print(f"worker {args.worker_index}: {len(jobs)} image GIFs; {args.projection}",flush=True)
    reports=[];errors=[]
    for i,(src,dst) in enumerate(jobs,1):
        if dst.exists() and not args.force:continue
        try:
            r=_image_gif(src,dst,args);reports.append(r)
            print(f"[{i}/{len(jobs)}] {src.name}: {r['source_timepoints']} t, {r['seconds']}s",flush=True)
        except Exception as e:
            errors.append(dict(source=str(src),error=f"{type(e).__name__}: {e}"));print(errors[-1],flush=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True,exist_ok=True)
        args.report_json.write_text(json.dumps(dict(results=reports,errors=errors),indent=2))
    if errors:raise SystemExit(1)


if __name__=="__main__":main()
