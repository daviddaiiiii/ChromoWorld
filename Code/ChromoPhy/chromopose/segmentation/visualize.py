#!/usr/bin/env python3
"""
visualize4D.py

4D TIFF (T,Z,C,Y,X; also common with a leading singleton axis e.g. (1,T,Z,Y,X)):
- per time step, raw rotation GIFs
- per time+rotate combined raw GIF
- optional mask rotation GIFs if --mask_path is provided

- vis_setup specifies input axis order + sampling ratio (after squeezing singleton dims), e.g. {"t":1.0,"z":1.0,"x":1.0,"y":1.0}
- --gif_only controls whether to keep frame image folders (default: true).
"""

from __future__ import annotations

import argparse
import csv
import colorsys
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import json
import numpy as np

# NumPy 2 removed ndarray.newbyteorder; older tifffile still calls it from read_array, so we patch FileHandle.read_array.
if int(np.__version__.split(".", 1)[0]) >= 2:
    import tifffile

    def _patched_read_array(self, dtype, count=-1, offset=0, *, out=None):
        dtype = np.dtype(dtype)
        if count < 0:
            nbytes = self._size if out is None else out.nbytes
            count = nbytes // dtype.itemsize
        else:
            nbytes = count * dtype.itemsize
        result = np.empty(count, dtype) if out is None else out
        if result.nbytes != nbytes:
            raise ValueError("size mismatch")
        assert self._fh is not None
        if offset:
            self._fh.seek(self._offset + offset)
        try:
            n = self._fh.readinto(result)
        except AttributeError:
            result[:] = np.frombuffer(self._fh.read(nbytes), dtype).reshape(result.shape)
            n = nbytes
        if n != nbytes:
            raise ValueError(f"failed to read {nbytes} bytes, got {n}")
        if not result.dtype.isnative:
            if not dtype.isnative:
                result.byteswap(True)
            result = result.view(result.dtype.newbyteorder())
        elif result.dtype.isnative != dtype.isnative:
            result.byteswap(True)
        if out is not None and hasattr(out, "flush"):
            out.flush()
        return result

    tifffile.FileHandle.read_array = _patched_read_array

import torch
import torch.nn.functional as F

MASK_BUCKET_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 178, 178),
    (255, 51, 51),
    (153, 0, 0),
    (255, 204, 153),
    (255, 102, 0),
    (255, 255, 204),
    (255, 255, 0),
    (255, 255, 102),
    (255, 239, 153),
    (255, 204, 0),
    (204, 153, 0),
    (153, 102, 0),
    (255, 153, 51),
    (204, 255, 204),
    (51, 255, 51),
    (0, 102, 0),
    (153, 255, 255),
    (0, 255, 255),
    (0, 128, 128),
    (153, 153, 255),
    (0, 0, 255),
    (216, 191, 216),
    (153, 51, 255),
    (75, 0, 130),
)

MASK_BUCKET_CLASS_NAMES: tuple[str, ...] = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "X",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "Y",
)

def build_visible_spectrum_colors(sorted_ids: list[int]) -> dict[int, tuple[int, int, int]]:
    """Map sorted_ids order (usually descending volume) to RGB from red through violet in HSV."""
    n = len(sorted_ids)
    out: dict[int, tuple[int, int, int]] = {}
    if n <= 0:
        return out
    for rank, rid in enumerate(sorted_ids):
        t = rank / max(1, n - 1)
        hue = 0.0 + (0.77 * t)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        out[int(rid)] = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
    return out

def _load_tiff_volume(tiff_path: Path) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(str(tiff_path)))

def _parse_vis_setup(raw: str) -> dict[str, float]:
    if isinstance(raw, dict):
        data = raw
    else:
        raw_s = raw.strip()
        try:
            data = json.loads(raw_s)
        except Exception:
            data = json.loads(raw_s.replace("'", '"'))
    if not isinstance(data, dict):
        raise ValueError("vis_setup must be a JSON object")
    normalized: dict[str, float] = {}
    for k, v in data.items():
        key = str(k).strip().lower()
        if key in {"time"}:
            key = "t"
        if key not in {"t", "c", "x", "y", "z"}:
            raise ValueError(f"Unsupported vis_setup axis key: {key!r}, expect one of t/c/x/y/z")
        if key in normalized:
            raise ValueError(f"duplicated axis in vis_setup: {key!r}")
        ratio = float(v)
        if not np.isfinite(ratio):
            raise ValueError(f"vis_setup[{key}] must be finite, got {v!r}")
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"vis_setup[{key}] must be in (0,1], got {v!r}")
        normalized[key] = ratio
    return normalized

def _reorder_volume_by_vis_setup(arr: np.ndarray, vis_setup: dict[str, float]) -> np.ndarray:
    """Reorder array according to vis_setup axis keys and return shape [T, Z, C, Y, X] (insert singleton T/C when absent)."""
    arr = np.asarray(arr)
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 0:
        raise ValueError(f"Input array has no dimensions after squeeze: shape={arr.shape}")

    axis_keys = [k for k in vis_setup.keys() if k in {"t", "z", "c", "x", "y"}]
    if not axis_keys:
        raise ValueError("vis_setup must include at least one axis key")

    if len(axis_keys) != squeezed.ndim:
        raise ValueError(
            f"vis_setup axis count {len(axis_keys)} does not match non-singleton dim count {squeezed.ndim} (shape={squeezed.shape})"
        )

    required = {"x", "y", "z"}
    missing = sorted(required - set(axis_keys))
    if missing:
        raise ValueError(f"vis_setup must include x,y,z; missing {missing}")

    name_to_axis = {name: idx for idx, name in enumerate(axis_keys)}
    perm: list[int] = []
    if "t" in name_to_axis:
        perm.append(name_to_axis["t"])
    if "z" in name_to_axis:
        perm.append(name_to_axis["z"])
    if "c" in name_to_axis:
        perm.append(name_to_axis["c"])
    perm.extend(name_to_axis[name] for name in ("y", "x"))

    reordered = np.transpose(squeezed, axes=perm) if len(perm) == squeezed.ndim else squeezed
    if "t" not in name_to_axis:
        reordered = np.expand_dims(reordered, axis=0)
    if "c" not in name_to_axis:
        reordered = np.expand_dims(reordered, axis=2)
    return reordered

def _parse_bool_arg(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")

def _sample_indices_from_ratio(n: int, ratio: float, sampler_name: str) -> list[int]:
    if n <= 0:
        return []
    if not np.isfinite(ratio):
        raise ValueError(f"{sampler_name} must be finite, got {ratio}")
    if ratio <= 0:
        raise ValueError(f"{sampler_name} must be in (0,1].")
    if ratio > 1:
        raise ValueError(f"{sampler_name} must be in (0,1], got {ratio}")
    if ratio == 1:
        return list(range(n))

    k = int(np.ceil(n * ratio))
    k = max(1, min(n, k))
    if k <= 1:
        return [0]

    idx = np.linspace(0, n - 1, k)
    out = sorted(set(int(round(float(x))) for x in idx))
    if not out:
        out = [0]
    return out

def infer_time_axis(shape: tuple[int, ...]) -> int:
    """
    Infer time axis from shape: among axes with length > 1, choose the one with the fewest slices.
    """
    candidates = [i for i, s in enumerate(shape) if s > 1]
    if not candidates:
        return 0
    return min(candidates, key=lambda i: (shape[i], i))

def _instance_volumes_desc_ids(masks_f: np.ndarray) -> list[int]:
    """Local instance IDs sorted by descending voxel volume."""
    m = np.asarray(masks_f)
    if m.ndim == 4 and m.shape[1] == 1:
        m = m[:, 0, :, :]
    if m.ndim != 3:
        raise ValueError(f"Expected mask 3D for volume statistics, got shape={m.shape}")

    instance_ids = np.unique(m)
    instance_ids = instance_ids[instance_ids > 0].astype(int).tolist()
    pairs: list[tuple[int, int]] = []
    for rid in instance_ids:
        vol = int((m == rid).sum())
        pairs.append((vol, int(rid)))
    pairs.sort(key=lambda x: -x[0])
    return [rid for _, rid in pairs]

def build_colors_lut_from_mask_volume(
    mask_vol: np.ndarray,
    ta: int,
    time_indices: list[int],
    depth_indices: list[int],
    spatial_indices: tuple[list[int], list[int]],
) -> tuple[np.ndarray, dict[int, int], list[int]]:
    """Build one global LUT for all sampled time steps based on total mask volume."""
    counts: dict[int, int] = {}
    for t_idx in time_indices:
        m = prepare_volume_slice(mask_vol, ta, t_idx, depth_indices=depth_indices, spatial_indices=spatial_indices)
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0, :, :]
        if m.ndim != 3:
            raise ValueError(f"Expected mask 3D for global LUT count, got shape={m.shape}")
        ids = np.asarray(m)
        ids = ids[ids > 0].astype(np.int64).ravel()
        if ids.size == 0:
            continue
        uniq, cts = np.unique(ids, return_counts=True)
        for rid, cnt in zip(uniq.astype(int), cts.astype(int)):
            counts[int(rid)] = counts.get(int(rid), 0) + int(cnt)

    if not counts:
        return (
            np.array(MASK_BUCKET_COLORS, dtype=np.uint8),
            {},
            [0 for _ in MASK_BUCKET_CLASS_NAMES],
        )

    sorted_ids = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    n_instances = len(sorted_ids)
    n_buckets = len(MASK_BUCKET_COLORS)
    if n_instances <= 0:
        return (
            np.array(MASK_BUCKET_COLORS, dtype=np.uint8),
            {},
            [0 for _ in MASK_BUCKET_CLASS_NAMES],
        )

    instance_to_class: dict[int, int] = {}
    class_counts: list[int] = [0 for _ in MASK_BUCKET_CLASS_NAMES]
    for rank, (rid, _vol) in enumerate(sorted_ids):
        # continuous axis [0, n_instances), evenly split into n_buckets segments
        bucket = min(n_buckets - 1, int(rank * n_buckets / n_instances))
        instance_to_class[int(rid)] = bucket
        class_counts[bucket] += 1

    lut = np.array(MASK_BUCKET_COLORS, dtype=np.uint8)
    return lut, instance_to_class, class_counts

def build_instance_class_lut(instance_to_class: dict[int, int]) -> np.ndarray:
    """Build a dense id->class LUT for fast vectorized mask remapping."""
    if not instance_to_class:
        return np.zeros(1, dtype=np.int32)
    max_instance_id = max(instance_to_class.keys())
    instance_lut = np.zeros(max_instance_id + 1, dtype=np.int32)
    for instance_id, class_id in instance_to_class.items():
        instance_lut[int(instance_id)] = int(class_id)
    return instance_lut

def map_mask_instances_to_bucket_class(
    mask_arr: np.ndarray,
    instance_class_lut: np.ndarray,
) -> np.ndarray:
    """Map raw instance ids in mask volume to bucket class indices."""
    m = np.asarray(mask_arr)
    if m.ndim == 4 and m.shape[1] == 1:
        m = m[:, 0, :, :]
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask for instance mapping, got shape={m.shape}")
    m = np.round(m).astype(np.int32)

    if instance_class_lut.size == 0:
        return np.zeros_like(m, dtype=np.int32)

    valid = (m > 0) & (m < instance_class_lut.size)
    out = np.zeros_like(m, dtype=np.int32)
    if not valid.any():
        return out
    out[valid] = instance_class_lut[m[valid]]
    return out

def write_mask_stats_csv(out_csv: Path, class_counts: list[int]) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "instance_count"])
        for name, cnt in zip(MASK_BUCKET_CLASS_NAMES, class_counts):
            w.writerow([name, int(cnt)])

def build_colors_lut_volume_spectrum(masks_f: np.ndarray) -> np.ndarray:
    """One frame: descending volume -> visible spectrum -> colors_lut with shape (max_id+1, 3)."""
    m = np.asarray(masks_f)
    if m.ndim == 4 and m.shape[1] == 1:
        m = m[:, 0, :, :]
    if m.ndim != 3:
        raise ValueError(f"Expected mask 3D/4D(with C=1) for color LUT, got shape={m.shape}")

    sorted_ids = _instance_volumes_desc_ids(m)
    if not sorted_ids:
        return np.zeros((1, 3), dtype=np.uint8)
    color_map = build_visible_spectrum_colors(sorted_ids)
    max_id = int(m.max())
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for rid, rgb in color_map.items():
        lut[int(rid)] = np.array(rgb, dtype=np.uint8)
    return lut

def _mask_for_rotation(masks_f: np.ndarray) -> np.ndarray:
    m = np.asarray(masks_f)
    if m.ndim == 4 and m.shape[1] == 1:
        m = m[:, 0, :, :]
    if m.ndim != 3:
        raise ValueError(f"Expected mask 3D for rotation tensorization, got shape={m.shape}")
    return np.round(m).astype(np.float32)

def write_ppm_rgb(path: Path, rgb_uint8: np.ndarray):
    rgb_uint8 = np.ascontiguousarray(rgb_uint8, dtype=np.uint8)
    H, W, C = rgb_uint8.shape
    assert C == 3
    header = f"P6\n{W} {H}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(rgb_uint8.tobytes())

def _ppm_dir_to_png_dir_pillow(ppm_dir: Path, png_dir: Path) -> None:
    from PIL import Image

    for ppm in sorted(ppm_dir.glob("frame_*.ppm")):
        png_path = png_dir / f"{ppm.stem}.png"
        Image.open(ppm).convert("RGB").save(png_path)

def _png_dir_to_gif_pillow(png_dir: Path, out_gif: Path, fps: int) -> None:
    from PIL import Image

    paths = sorted(png_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG frames under {png_dir}")
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    frames = [Image.open(p).convert("RGB") for p in paths]
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

def prepare_volume_slice(
    vol: np.ndarray,
    ta: int,
    t_idx: int,
    depth_indices: list[int] | None = None,
    spatial_indices: tuple[list[int], list[int]] | None = None,
) -> np.ndarray:
    idx = [slice(None)] * vol.ndim
    idx[ta] = t_idx
    img_t = vol[tuple(idx)].copy()
    while img_t.ndim >= 4 and img_t.shape[0] == 1:
        img_t = np.squeeze(img_t, axis=0)
    if img_t.ndim == 3:
        img_t = np.expand_dims(img_t, axis=1)
    elif img_t.ndim != 4:
        raise ValueError(f"After slicing time, expected (Z,C,Y,X) or (Z,Y,X), got {img_t.shape}")
    if depth_indices is not None:
        img_t = img_t[depth_indices]
    if spatial_indices is not None:
        y_idx, x_idx = spatial_indices
        if y_idx is not None:
            img_t = img_t[:, :, y_idx, :]
        if x_idx is not None:
            img_t = img_t[:, :, :, x_idx]
    return img_t

def contrast_range_from_img3d(img_3d: np.ndarray, proj_plane: str, contrast_low_pct: float, contrast_high_pct: float) -> tuple[float, float]:
    if proj_plane == "xy":
        proj0 = np.max(img_3d, axis=0).astype(np.float32)
    elif proj_plane == "xz":
        proj0 = np.max(img_3d, axis=1).astype(np.float32)
    else:
        proj0 = np.max(img_3d, axis=2).astype(np.float32)
    lo = float(np.percentile(proj0, contrast_low_pct))
    hi = float(np.percentile(proj0, contrast_high_pct))
    if hi <= lo + 1e-8:
        lo = float(np.min(proj0))
        hi = float(np.max(proj0))
        if hi <= lo + 1e-8:
            lo, hi = 0.0, 1.0
    return lo, hi

def _vol_proj_from_rot(rot_img: torch.Tensor, proj_plane: str) -> torch.Tensor:
    if proj_plane == "xy":
        return rot_img.max(dim=2).values.squeeze(0).squeeze(0)
    if proj_plane == "xz":
        return rot_img.max(dim=3).values.squeeze(0).squeeze(0)
    return rot_img.max(dim=4).values.squeeze(0).squeeze(0)

def _ppm_sequence_to_gif(ppm_tmp_dir: Path, frames_dir: Path, out_gif: Path, fps: int, gif_only: bool) -> None:
    keep_frames = not gif_only
    has_ffmpeg = shutil.which("ffmpeg") is not None

    if keep_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for p in frames_dir.glob("frame_*.png"):
            try:
                p.unlink()
            except Exception:
                pass
    else:
        for p in frames_dir.glob("frame_*.png"):
            try:
                p.unlink()
            except Exception:
                pass

    if has_ffmpeg:
        if keep_frames:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(ppm_tmp_dir / "frame_%04d.ppm"),
                    str(frames_dir / "frame_%04d.png"),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(frames_dir / "frame_%04d.png"),
                    "-filter_complex",
                    "[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse",
                    "-loop",
                    "0",
                    str(out_gif),
                ],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(ppm_tmp_dir / "frame_%04d.ppm"),
                    "-filter_complex",
                    "[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse",
                    "-loop",
                    "0",
                    str(out_gif),
                ],
                check=True,
            )
    else:
        if keep_frames:
            _ppm_dir_to_png_dir_pillow(ppm_tmp_dir, frames_dir)
            _png_dir_to_gif_pillow(frames_dir, out_gif, fps)
        else:
            from PIL import Image

            frames = []
            for p in sorted(ppm_tmp_dir.glob("frame_*.ppm")):
                with Image.open(p) as im:
                    frames.append(im.convert("RGB"))
            if not frames:
                raise FileNotFoundError(f"No PPM frames under {ppm_tmp_dir}")
            duration_ms = max(1, int(round(1000.0 / max(1, fps))))
            frames[0].save(
                out_gif,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )

    for p in ppm_tmp_dir.glob("frame_*.ppm"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        ppm_tmp_dir.rmdir()
    except Exception:
        pass

def render_rotation_gif_raw(img_tczyx: np.ndarray, device: torch.device, args, out_gif: Path) -> None:
    img_3d = img_tczyx[:, 0, :, :].astype(np.float32)
    lo, hi = contrast_range_from_img3d(img_3d, args.proj_plane, args.contrast_low_pct, args.contrast_high_pct)
    img_tensor_5d = torch.from_numpy(img_3d).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask_tensor_5d = torch.zeros_like(img_tensor_5d)
    n = max(1, int(args.num_angles))
    angles = [args.rotate_start_deg + i * (360.0 / n) for i in range(n)]

    frames_dir = out_gif.with_suffix("").with_name(out_gif.stem + "_frames")
    if not args.gif_only:
        frames_dir.mkdir(parents=True, exist_ok=True)
    ppm_tmp_dir = frames_dir.with_name(frames_dir.name + "_ppm_tmp")
    ppm_tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in ppm_tmp_dir.glob("frame_*.ppm"):
        try:
            p.unlink()
        except Exception:
            pass

    with torch.no_grad():
        for i, ang in enumerate(angles):
            rot_img, _ = rotate_volume_around_y(img_tensor_5d, mask_tensor_5d, ang, device=device)
            vol_proj = _vol_proj_from_rot(rot_img, args.proj_plane)
            vol_norm = (vol_proj - lo) / (hi - lo)
            vol_norm = torch.clamp(vol_norm, 0.0, 1.0)
            vol_uint8 = (vol_norm * 255.0).round().to(torch.uint8).cpu().numpy()
            rgb = np.stack([vol_uint8, vol_uint8, vol_uint8], axis=-1)
            write_ppm_rgb(ppm_tmp_dir / f"frame_{i:04d}.ppm", rgb)

    _ppm_sequence_to_gif(ppm_tmp_dir, frames_dir, out_gif, args.fps, gif_only=args.gif_only)
    print(f"  Saved: {out_gif}")

def render_rotation_gif_mask(
    img_tczyx: np.ndarray,
    masks_f: np.ndarray,
    colors_lut: np.ndarray,
    device: torch.device,
    args,
    out_gif: Path,
) -> None:
    img_3d = img_tczyx[:, 0, :, :].astype(np.float32)
    img_tensor_5d = torch.from_numpy(img_3d).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask_3d = _mask_for_rotation(masks_f)
    mask_tensor_5d = torch.from_numpy(mask_3d).to(device=device).unsqueeze(0).unsqueeze(0)
    n = max(1, int(args.num_angles))
    angles = [args.rotate_start_deg + i * (360.0 / n) for i in range(n)]

    frames_dir = out_gif.with_suffix("").with_name(out_gif.stem + "_frames")
    if not args.gif_only:
        frames_dir.mkdir(parents=True, exist_ok=True)
    ppm_tmp_dir = frames_dir.with_name(frames_dir.name + "_ppm_tmp")
    ppm_tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in ppm_tmp_dir.glob("frame_*.ppm"):
        try:
            p.unlink()
        except Exception:
            pass

    with torch.no_grad():
        for i, ang in enumerate(angles):
            _, rot_mask = rotate_volume_around_y(img_tensor_5d, mask_tensor_5d, ang, device=device)
            mask_proj = frontmost_projection(rot_mask, args.proj_plane, device=device)
            mask_proj_np = mask_proj.cpu().numpy().astype(np.int32)
            mask_proj_np = np.clip(mask_proj_np, 0, colors_lut.shape[0] - 1)
            rgb = np.zeros(mask_proj_np.shape + (3,), dtype=np.uint8)
            mask_fg = mask_proj_np != 0
            rgb[mask_fg] = colors_lut[mask_proj_np[mask_fg]]
            write_ppm_rgb(ppm_tmp_dir / f"frame_{i:04d}.ppm", rgb)

    _ppm_sequence_to_gif(ppm_tmp_dir, frames_dir, out_gif, args.fps, gif_only=args.gif_only)
    print(f"  Saved: {out_gif}")

def _time_index_for_combo_frame(i: int, n_angles: int, n_times: int) -> int:
    """Index into time_indices for rotation frame i (evenly aligned with num_angles and number of sampled times)."""
    if n_times <= 1:
        return 0
    if n_angles <= 1:
        return 0
    return int(round(i * (n_times - 1) / max(1, n_angles - 1)))

def render_time_rotate_combo_raw(
    vol: np.ndarray,
    ta: int,
    time_indices: list[int],
    depth_indices: list[int],
    spatial_indices: tuple[list[int], list[int]],
    device: torch.device,
    args,
    out_gif: Path,
) -> None:
    n_angles = max(1, int(args.num_angles))
    n_t = len(time_indices)
    frames_dir = out_gif.with_suffix("").with_name(out_gif.stem + "_frames")
    if not args.gif_only:
        frames_dir.mkdir(parents=True, exist_ok=True)
    ppm_tmp_dir = frames_dir.with_name(frames_dir.name + "_ppm_tmp")
    ppm_tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in ppm_tmp_dir.glob("frame_*.ppm"):
        try:
            p.unlink()
        except Exception:
            pass

    with torch.no_grad():
        for i in range(n_angles):
            k_time = _time_index_for_combo_frame(i, n_angles, n_t)
            t_idx = time_indices[k_time]
            img_t = prepare_volume_slice(
                vol, ta, t_idx, depth_indices=depth_indices, spatial_indices=spatial_indices
            )
            img_3d = img_t[:, 0, :, :].astype(np.float32)
            lo, hi = contrast_range_from_img3d(img_3d, args.proj_plane, args.contrast_low_pct, args.contrast_high_pct)
            ang = args.rotate_start_deg + i * (360.0 / n_angles)
            img_tensor_5d = torch.from_numpy(img_3d).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask_tensor_5d = torch.zeros_like(img_tensor_5d)
            rot_img, _ = rotate_volume_around_y(img_tensor_5d, mask_tensor_5d, ang, device=device)
            vol_proj = _vol_proj_from_rot(rot_img, args.proj_plane)
            vol_norm = (vol_proj - lo) / (hi - lo)
            vol_norm = torch.clamp(vol_norm, 0.0, 1.0)
            vol_uint8 = (vol_norm * 255.0).round().to(torch.uint8).cpu().numpy()
            rgb = np.stack([vol_uint8, vol_uint8, vol_uint8], axis=-1)
            write_ppm_rgb(ppm_tmp_dir / f"frame_{i:04d}.ppm", rgb)

    _ppm_sequence_to_gif(ppm_tmp_dir, frames_dir, out_gif, args.fps, gif_only=args.gif_only)
    print(f"  Saved: {out_gif}")

def render_time_rotate_combo_mask(
    vol: np.ndarray,
    ta: int,
    time_indices: list[int],
    depth_indices: list[int],
    spatial_indices: tuple[list[int], list[int]],
    masks_per_time: Optional[list[np.ndarray]],
    mask_vol: Optional[np.ndarray],
    colors_lut: np.ndarray,
    instance_class_lut: np.ndarray,
    device: torch.device,
    args,
    out_gif: Path,
) -> None:
    n_angles = max(1, int(args.num_angles))
    n_t = len(time_indices)
    if masks_per_time is not None:
        if len(masks_per_time) != n_t:
            raise ValueError(f"len(masks_per_time)={len(masks_per_time)} != len(time_indices)={n_t}")
    elif mask_vol is None:
        raise ValueError("mask_vol must be provided when masks_per_time is None")

    frames_dir = out_gif.with_suffix("").with_name(out_gif.stem + "_frames")
    if not args.gif_only:
        frames_dir.mkdir(parents=True, exist_ok=True)
    ppm_tmp_dir = frames_dir.with_name(frames_dir.name + "_ppm_tmp")
    ppm_tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in ppm_tmp_dir.glob("frame_*.ppm"):
        try:
            p.unlink()
        except Exception:
            pass

    with torch.no_grad():
        for i in range(n_angles):
            k_time = _time_index_for_combo_frame(i, n_angles, n_t)
            t_idx = time_indices[k_time]
            img_t = prepare_volume_slice(
                vol, ta, t_idx, depth_indices=depth_indices, spatial_indices=spatial_indices
            )
            img_3d = img_t[:, 0, :, :].astype(np.float32)
            if masks_per_time is None:
                masks_t = prepare_volume_slice(
                    mask_vol, ta, t_idx, depth_indices=depth_indices, spatial_indices=spatial_indices
                )
                masks_f = map_mask_instances_to_bucket_class(masks_t, instance_class_lut)
            else:
                masks_f = masks_per_time[k_time]
            lo, hi = contrast_range_from_img3d(img_3d, args.proj_plane, args.contrast_low_pct, args.contrast_high_pct)
            ang = args.rotate_start_deg + i * (360.0 / n_angles)
            img_tensor_5d = torch.from_numpy(img_3d).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask_3d = _mask_for_rotation(masks_f)
            mask_tensor_5d = torch.from_numpy(mask_3d).to(device=device).unsqueeze(0).unsqueeze(0)
            rot_img, rot_mask = rotate_volume_around_y(img_tensor_5d, mask_tensor_5d, ang, device=device)
            mask_proj = frontmost_projection(rot_mask, args.proj_plane, device=device)
            mask_proj_np = mask_proj.cpu().numpy().astype(np.int32)
            mask_proj_np = np.clip(mask_proj_np, 0, colors_lut.shape[0] - 1)
            mask_fg = mask_proj_np != 0
            if bool(getattr(args, "vis_overlay_image", True)):
                vol_proj = frontmost_projection(rot_img, args.proj_plane, device=device)
                vol_norm = (vol_proj - lo) / (hi - lo)
                vol_norm = torch.clamp(vol_norm, 0.0, 1.0)
                gray = (vol_norm * 255.0).round().to(torch.uint8).cpu().numpy()
                rgb = np.stack([gray, gray, gray], axis=-1)
                rgb[mask_fg] = colors_lut[mask_proj_np[mask_fg]]
            else:
                rgb = np.zeros(mask_proj_np.shape + (3,), dtype=np.uint8)
                rgb[mask_fg] = colors_lut[mask_proj_np[mask_fg]]
            write_ppm_rgb(ppm_tmp_dir / f"frame_{i:04d}.ppm", rgb)

    _ppm_sequence_to_gif(ppm_tmp_dir, frames_dir, out_gif, args.fps, gif_only=args.gif_only)
    print(f"  Saved: {out_gif}")

def rotate_volume_around_y(img_tensor_5d: torch.Tensor, mask_tensor_5d: torch.Tensor, angle_deg: float, device: torch.device):
    angle_rad = float(angle_deg) * np.pi / 180.0
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))

    theta = torch.zeros((1, 3, 4), device=device, dtype=torch.float32)
    theta[0, 0, 0] = c
    theta[0, 0, 2] = s
    theta[0, 1, 1] = 1.0
    theta[0, 2, 0] = -s
    theta[0, 2, 2] = c

    grid = F.affine_grid(theta, size=img_tensor_5d.size(), align_corners=False)
    rot_img = F.grid_sample(
        img_tensor_5d, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    rot_mask = F.grid_sample(
        mask_tensor_5d, grid, mode="nearest", padding_mode="zeros", align_corners=False
    )
    return rot_img, rot_mask

def frontmost_projection(rot_mask: torch.Tensor, proj_plane: str, device: torch.device):
    m = torch.round(rot_mask).to(torch.long)
    exist = m > 0

    if proj_plane == "xy":
        D = m.shape[2]
        depth_idx = torch.arange(D, device=device, dtype=torch.long).view(1, 1, D, 1, 1)
        depth_score = torch.where(exist, depth_idx, torch.full_like(depth_idx, -1))
        front_idx = depth_score.argmax(dim=2)
        front_ids = torch.gather(m, dim=2, index=front_idx.unsqueeze(2)).squeeze(2)
        exist_any = exist.any(dim=2)
        front_ids = torch.where(exist_any, front_ids, torch.zeros_like(front_ids))
        return front_ids.squeeze(0).squeeze(0)

    if proj_plane == "xz":
        H = m.shape[3]
        depth_idx = torch.arange(H, device=device, dtype=torch.long).view(1, 1, 1, H, 1)
        depth_score = torch.where(exist, depth_idx, torch.full_like(depth_idx, -1))
        front_idx = depth_score.argmax(dim=3)
        front_ids = torch.gather(m, dim=3, index=front_idx.unsqueeze(3)).squeeze(3)
        exist_any = exist.any(dim=3)
        front_ids = torch.where(exist_any, front_ids, torch.zeros_like(front_ids))
        return front_ids.squeeze(0).squeeze(0)

    W = m.shape[4]
    depth_idx = torch.arange(W, device=device, dtype=torch.long).view(1, 1, 1, 1, W)
    depth_score = torch.where(exist, depth_idx, torch.full_like(depth_idx, -1))
    front_idx = depth_score.argmax(dim=4)
    front_ids = torch.gather(m, dim=4, index=front_idx.unsqueeze(4)).squeeze(4)
    exist_any = exist.any(dim=4)
    front_ids = torch.where(exist_any, front_ids, torch.zeros_like(front_ids))
    return front_ids.squeeze(0).squeeze(0)

def _normalize_mask_like_volume(mask_raw: np.ndarray, vol: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_raw)

    if m.ndim == vol.ndim and m.shape == vol.shape:
        return np.round(m).astype(np.int32)

    # common case: vol was expanded from 4D to 5D with leading singleton axis
    if m.ndim == vol.ndim - 1 and vol.ndim == 5 and vol.shape[0] == 1 and m.shape == vol.shape[1:]:
        return np.expand_dims(np.round(m), axis=0).astype(np.int32)

    # also support inserting channel axis when vol has channel dimension and mask is missing it
    if (
        m.ndim == 4
        and vol.ndim == 5
        and vol.shape[2] == 1
        and m.shape == (vol.shape[0], vol.shape[1], vol.shape[3], vol.shape[4])
    ):
        return np.round(np.expand_dims(m, axis=2)).astype(np.int32)

    raise ValueError(f"Mask shape {m.shape} does not match volume shape {vol.shape} and cannot be normalized.")

def _resize_axis_linear(vol: np.ndarray, axis: int, new_size: int) -> np.ndarray:
    """Linear-resample one axis of an array to ``new_size``."""
    old_size = vol.shape[axis]
    if old_size == new_size:
        return vol
    if new_size <= 0:
        raise ValueError(f"new_size must be positive, got {new_size}")
    if old_size == 1:
        reps = [1] * vol.ndim
        reps[axis] = new_size
        return np.repeat(vol, new_size, axis=axis)

    src = np.linspace(0.0, old_size - 1.0, num=new_size, dtype=np.float64)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, old_size - 1)
    w = (src - i0).astype(np.float32)
    # Move target axis to front for easy fancy indexing, then restore.
    vol_m = np.moveaxis(vol, axis, 0)
    a0 = vol_m[i0].astype(np.float32, copy=False)
    a1 = vol_m[i1].astype(np.float32, copy=False)
    w_shape = [new_size] + [1] * (vol.ndim - 1)
    out = a0 * (1.0 - w.reshape(w_shape)) + a1 * w.reshape(w_shape)
    return np.moveaxis(out, 0, axis)

def _resize_volume_to_mask_shape(vol: np.ndarray, mask_vol: np.ndarray) -> np.ndarray:
    """Resize raw volume [T,Z,C,Y,X] spatially to match mask [T,Z,C|1,Y,X].

    Intensity volume is interpolated (linear) along mismatched axes.
    Channel count of ``vol`` is kept.
    """
    vol = np.asarray(vol)
    mask_vol = np.asarray(mask_vol)
    if vol.ndim != 5 or mask_vol.ndim != 5:
        raise ValueError(
            f"Expected 5D [T,Z,C,Y,X] volumes, got vol={vol.shape}, mask={mask_vol.shape}"
        )

    target = (
        int(mask_vol.shape[0]),
        int(mask_vol.shape[1]),
        int(vol.shape[2]),
        int(mask_vol.shape[3]),
        int(mask_vol.shape[4]),
    )
    if tuple(vol.shape) == target:
        return vol

    print(
        f"  Shape mismatch: raw {tuple(vol.shape)} vs mask {tuple(mask_vol.shape)}; "
        f"resizing raw -> {target}"
    )
    out = vol
    # Axis order: T=0, Z=1, C=2 (kept), Y=3, X=4
    for axis, new_size in ((0, target[0]), (1, target[1]), (3, target[3]), (4, target[4])):
        if out.shape[axis] != new_size:
            out = _resize_axis_linear(out, axis=axis, new_size=new_size)
    if tuple(out.shape) != target:
        raise RuntimeError(f"Resize failed: got {out.shape}, expected {target}")
    return out.astype(vol.dtype, copy=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_path", type=str, default="./OpenChromo/our_live/croped/rpe1_60x_2_0002_561.tif")
    parser.add_argument("--out_path", type=str, default="./OpenChromo/our_live/croped/")
    parser.add_argument("--mask_path", type=str, default="./OpenChromo/our_live/chrp3d/rpe1_60x_2_0002_561_shape_tracked_t15.tiff", help="Optional mask tif path (single tif or directory).")
    parser.add_argument(
        "--gif_only",
        type=_parse_bool_arg,
        default=True,
        help="If true (default), only keep final gif outputs and do not retain intermediate image folders.",
    )
    parser.add_argument(
        "--combine_gif_only",
        type=_parse_bool_arg,
        default=True,
        help="If true, skip per-time rotation gifs and only generate time+rotation combined gifs.",
    )
    parser.add_argument(
        "--vis_setup",
        type=str,
        default='{"t":1.0,"z":1.0,"x":1.0,"y":1.0}',
        help='Dimension order and sampling ratio after squeeze, e.g. {"t":0.5,"z":0.5,"x":1.0,"y":1.0}',
    )

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_angles", type=int, default=36)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--rotate_start_deg", type=float, default=0.0)
    parser.add_argument("--contrast_low_pct", type=float, default=1.0)
    parser.add_argument("--contrast_high_pct", type=float, default=99.0)
    parser.add_argument("--proj_plane", type=str, default="xy", choices=["xy", "xz", "yz"])
    args = parser.parse_args()

    tiff_path = Path(args.tiff_path)
    if not tiff_path.exists():
        raise FileNotFoundError(tiff_path)

    out_root = Path(args.out_path)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print("[1/4] Loading 4D/5D TIFF...")
    vol = _load_tiff_volume(tiff_path)
    vol_squeezed = np.squeeze(vol)
    dim_desc = [f"{i}:{size}" for i, size in enumerate(vol_squeezed.shape)]
    print(f"  squeezed raw shape (1-dim removed): {vol_squeezed.shape}")
    print(f"  dimension list (after squeeze): {dim_desc}")
    vis_setup = _parse_vis_setup(args.vis_setup)
    vol = _reorder_volume_by_vis_setup(vol_squeezed, vis_setup)
    ta = 0
    print(f"  re-ordered volume shape: {vol.shape}, vis_setup={vis_setup}")

    if vol.ndim != 5:
        raise ValueError(f"Reordered volume must be 5D [T,Z,C,Y,X], got shape={vol.shape}")

    mask_vol = None
    if args.mask_path is not None:
        mask_path = Path(args.mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        mask_raw = _load_tiff_volume(mask_path)
        mask_raw_squeezed = np.squeeze(mask_raw)
        mask_dim_desc = [f"{i}:{size}" for i, size in enumerate(mask_raw_squeezed.shape)]
        print(f"  Loaded mask: {mask_path}")
        print(f"  mask squeezed raw shape (1-dim removed): {mask_raw_squeezed.shape}")
        print(f"  mask dimension list (after squeeze): {mask_dim_desc}")
        mask_vol = np.round(_reorder_volume_by_vis_setup(mask_raw_squeezed, vis_setup)).astype(np.int32)
        print(f"  reordered mask shape: {mask_vol.shape}")
        # Align raw intensity volume to mask geometry before sampling indices.
        vol = _resize_volume_to_mask_shape(vol, mask_vol)
        print(f"  aligned raw volume shape: {vol.shape}")

    T = vol.shape[ta]
    t_ratio = vis_setup.get("t", 1.0)
    z_ratio = vis_setup.get("z", 1.0)
    y_ratio = vis_setup.get("y", 1.0)
    x_ratio = vis_setup.get("x", 1.0)
    time_indices = _sample_indices_from_ratio(T, t_ratio, "vis_setup[t]")
    if not time_indices:
        raise ValueError(f"No time indices after vis_setup[t] (T={T}, sampler={t_ratio})")
    print(f"  T={T}, sampled indices ({len(time_indices)}): {time_indices[:20]}{'...' if len(time_indices) > 20 else ''}")

    depth_probe = prepare_volume_slice(vol, ta, time_indices[0], depth_indices=None, spatial_indices=None)
    depth_count = depth_probe.shape[0]
    depth_indices = _sample_indices_from_ratio(depth_count, z_ratio, "vis_setup[z]")
    if not depth_indices:
        raise ValueError(f"No depth indices after vis_setup[z] (Z={depth_count}, sampler={z_ratio})")
    print(f"  Z={depth_count}, sampled depth indices ({len(depth_indices)}): {depth_indices[:20]}{'...' if len(depth_indices) > 20 else ''}")
    y_indices = _sample_indices_from_ratio(depth_probe.shape[2], y_ratio, "vis_setup[y]")
    x_indices = _sample_indices_from_ratio(depth_probe.shape[3], x_ratio, "vis_setup[x]")
    print(f"  Y={depth_probe.shape[2]}, sampled y indices ({len(y_indices)}): {y_indices[:20]}{'...' if len(y_indices) > 20 else ''}")
    print(f"  X={depth_probe.shape[3]}, sampled x indices ({len(x_indices)}): {x_indices[:20]}{'...' if len(x_indices) > 20 else ''}")
    spatial_indices = (y_indices, x_indices)

    stem = tiff_path.stem
    n_angles = max(1, int(args.num_angles))
    n_t = len(time_indices)

    if not args.combine_gif_only:
        print("\n[2/4] Raw rotation GIFs (per time, no mask)...")
        for k, t_idx in enumerate(time_indices):
            print(f"\n=== Raw time {k + 1}/{n_t} (index={t_idx}) ===")
            img_t = prepare_volume_slice(vol, ta, t_idx, depth_indices=depth_indices, spatial_indices=spatial_indices)
            out_gif = out_root / f"{stem}_t{t_idx:04d}_raw.gif"
            print(f"  Generating {n_angles} rotation frames...")
            render_rotation_gif_raw(img_t, device, args, out_gif)

    print(
        "\n[3/4] time+rotate combined (raw): "
        "angle i -> 360°·i/num_angles, time slot -> round(i·(N_t-1)/max(1,num_angles-1))..."
    )
    combo_raw = out_root / f"{stem}_time_rotate_raw.gif"
    render_time_rotate_combo_raw(vol, ta, time_indices, depth_indices, spatial_indices, device, args, combo_raw)

    if mask_vol is not None:
        print("\n[4/4] mask rotation GIFs...")
        colors_lut, instance_to_class, class_counts = build_colors_lut_from_mask_volume(
            mask_vol, ta, time_indices, depth_indices, spatial_indices
        )
        print(f"  Global mask instances: {sum(class_counts)} (bucketed into {len(colors_lut)} colors)")
        stats_csv = out_root / f"{stem}_mask_stats.csv"
        write_mask_stats_csv(stats_csv, class_counts)
        print(f"  Saved mask stats: {stats_csv}")
        instance_class_lut = build_instance_class_lut(instance_to_class)

        masks_per_time: Optional[list[np.ndarray]]
        if args.combine_gif_only:
            masks_per_time = None
        else:
            masks_per_time = []
            for k, t_idx in enumerate(time_indices):
                print(f"\n=== Mask prep {k + 1}/{n_t} (index={t_idx}) ===")
                masks_t = prepare_volume_slice(
                    mask_vol, ta, t_idx, depth_indices=depth_indices, spatial_indices=spatial_indices
                )
                masks_t = map_mask_instances_to_bucket_class(masks_t, instance_class_lut)
                masks_per_time.append(masks_t)

        if not args.combine_gif_only:
            for k, t_idx in enumerate(time_indices):
                print(f"\n=== Mask GIF {k + 1}/{n_t} (index={t_idx}) ===")
                img_t = prepare_volume_slice(vol, ta, t_idx, depth_indices=depth_indices, spatial_indices=spatial_indices)
                out_mask = out_root / f"{stem}_t{t_idx:04d}_mask.gif"
                render_rotation_gif_mask(img_t, masks_per_time[k], colors_lut, device, args, out_mask)

        combo_mask = out_root / f"{stem}_time_rotate_mask.gif"
        render_time_rotate_combo_mask(
            vol,
            ta,
            time_indices,
            depth_indices,
            spatial_indices,
            masks_per_time,
            mask_vol,
            colors_lut,
            instance_class_lut,
            device,
            args,
            combo_mask,
        )

    print("\nDone.")


# ---- Embedded helpers: target_visualization.py ----

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from model import (
    CH_DISTANCE,
    CH_FLOW_X,
    CH_FLOW_Y,
    CH_FLOW_Z,
    CH_FOREGROUND,
    CH_SKELETON,
)

def _mip_max(vol_zyx: np.ndarray) -> np.ndarray:
    return np.max(vol_zyx, axis=0)

def _gray_to_bgr_u8(gray_yx: np.ndarray, *, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    x = gray_yx.astype(np.float32)
    lo = float(np.percentile(x, pmin))
    hi = float(np.percentile(x, pmax))
    if hi <= lo + 1e-6:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo + 1e-6:
        u8 = np.zeros_like(x, dtype=np.uint8)
    else:
        u8 = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        u8 = (u8 * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

def _binary_mask_to_bgr(mask_yx: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    u8 = (mask_yx.astype(np.float32) > float(threshold)).astype(np.uint8) * 255
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

def _label_to_bgr(label: int) -> np.ndarray:
    """Deterministic pseudo-color for one instance id (BGR uint8)."""
    lid = int(label)
    if lid <= 0:
        return np.zeros(3, dtype=np.uint8)
    rng = np.random.default_rng(lid * 9973 + 17)
    return rng.integers(48, 224, size=3, dtype=np.uint8)

def _greedy_pred_to_gt_color_map(
    pred_mask_zyx: np.ndarray,
    gt_mask_zyx: np.ndarray,
) -> Dict[int, int]:
    """Map pred instance id -> gt instance id (for shared colors) via greedy IoU."""
    pred = np.asarray(pred_mask_zyx, dtype=np.int32)
    gt = np.asarray(gt_mask_zyx, dtype=np.int32)
    pred_ids = [int(i) for i in np.unique(pred) if i > 0]
    gt_ids = [int(i) for i in np.unique(gt) if i > 0]
    if not pred_ids or not gt_ids:
        return {}

    pairs: list[Tuple[float, int, int]] = []
    for pid in pred_ids:
        p = pred == pid
        for gid in gt_ids:
            g = gt == gid
            inter = float((p & g).sum())
            if inter <= 0:
                continue
            union = float((p | g).sum())
            pairs.append((inter / max(union, 1.0), pid, gid))
    pairs.sort(reverse=True)

    pred_to_gt: Dict[int, int] = {}
    used_gt: set[int] = set()
    for iou, pid, gid in pairs:
        if pid in pred_to_gt or gid in used_gt:
            continue
        pred_to_gt[pid] = gid
        used_gt.add(gid)
    return pred_to_gt

def _instance_mask_mip_bgr(
    mask_zyx: np.ndarray,
    *,
    color_id_map: Optional[Dict[int, int]] = None,
    unmatched_offset: int = 50000,
) -> np.ndarray:
    """
    Colored instance MIP.

    color_id_map: optional relabel (e.g. pred_id -> gt_id) so colors match another mask.
    """
    mask = np.asarray(mask_zyx)
    if mask.ndim != 3:
        raise ValueError(f"mask must be (Z,Y,X), got {mask.shape}")
    colored = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)
    for z in range(mask.shape[0]):
        sl = mask[z]
        for lid in np.unique(sl):
            lid = int(lid)
            if lid <= 0:
                continue
            cid = int(color_id_map.get(lid, lid)) if color_id_map else lid
            if color_id_map is not None and lid not in color_id_map:
                cid = lid + int(unmatched_offset)
            colored[sl == lid] = np.maximum(colored[sl == lid], _label_to_bgr(cid))
    return colored

def _distance_to_bgr(
    dist_yx: np.ndarray,
    fg_yx: np.ndarray,
    label_mip_yx: np.ndarray,
    *,
    distance_clip: float,
) -> np.ndarray:
    """Per-instance Viridis: each instance scaled by its own max (capped at distance_clip)."""
    dist = dist_yx.astype(np.float32)
    clip = max(float(distance_clip), 1e-6)
    dist_n = np.zeros_like(dist, dtype=np.float32)
    label_mip = np.asarray(label_mip_yx, dtype=np.int32)
    for iid in np.unique(label_mip):
        if int(iid) <= 0:
            continue
        sl = label_mip == int(iid)
        d = dist[sl]
        if d.size == 0:
            continue
        scale = max(min(float(d.max()), clip), 1e-6)
        dist_n[sl] = np.clip(d / scale, 0.0, 1.0)
    unlabeled = fg_yx & (label_mip <= 0)
    if bool(unlabeled.any()):
        dist_n[unlabeled] = np.clip(dist[unlabeled] / clip, 0.0, 1.0)
    u8 = (dist_n * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
    colored[~fg_yx] = 0
    return colored

def _prob_heatmap_bgr(prob_yx: np.ndarray, fg_yx: Optional[np.ndarray] = None) -> np.ndarray:
    """Skeleton / probability heatmap (inferno), masked to foreground if given."""
    p = np.clip(prob_yx.astype(np.float32), 0.0, 1.0)
    u8 = (p * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    if fg_yx is not None:
        colored[~fg_yx] = 0
    return colored

def _unit_normalize_flow_zyx(
    fx_zyx: np.ndarray,
    fy_zyx: np.ndarray,
    fz_zyx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = fx_zyx.astype(np.float32)
    fy = fy_zyx.astype(np.float32)
    fz = fz_zyx.astype(np.float32)
    mag = np.sqrt(fx * fx + fy * fy + fz * fz) + 1e-6
    return fx / mag, fy / mag, fz / mag

def _flow_rgb_fg_proj_bgr(
    fx_zyx: np.ndarray,
    fy_zyx: np.ndarray,
    fz_zyx: np.ndarray,
    fg_zyx: np.ndarray,
    *,
    unit_normalize: bool = True,
) -> np.ndarray:
    """
    Project one (z,y,x) flow vector per pixel via argmax-z on foreground.

    When unit_normalize is True, direction-only RGB (magnitude shown via hue
    brightness is not encoded). Set False only if vectors are already bounded.
    """
    if unit_normalize:
        fx, fy, fz = _unit_normalize_flow_zyx(fx_zyx, fy_zyx, fz_zyx)
    else:
        fx, fy, fz = fx_zyx.astype(np.float32), fy_zyx.astype(np.float32), fz_zyx.astype(np.float32)
    fg = np.clip(fg_zyx.astype(np.float32), 0.0, 1.0)
    z_idx = np.argmax(fg, axis=0)
    z_idx_exp = z_idx[np.newaxis, ...]
    fx_2d = np.take_along_axis(fx, z_idx_exp, axis=0)[0]
    fy_2d = np.take_along_axis(fy, z_idx_exp, axis=0)[0]
    fz_2d = np.take_along_axis(fz, z_idx_exp, axis=0)[0]
    fg_m = np.take_along_axis(fg, z_idx_exp, axis=0)[0] > 0.5

    def _chan_to_u8(ch: np.ndarray) -> np.ndarray:
        return (np.clip(ch, -1.0, 1.0) * 0.5 + 0.5) * 255.0

    rgb = np.zeros((*fx_2d.shape, 3), dtype=np.uint8)
    rgb[..., 0] = _chan_to_u8(fx_2d).astype(np.uint8)
    rgb[..., 1] = _chan_to_u8(fz_2d).astype(np.uint8)
    rgb[..., 2] = _chan_to_u8(fy_2d).astype(np.uint8)
    rgb[~fg_m] = 0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-x))

def _panels_from_channels(
    raw_zyx: np.ndarray,
    channels_czyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    distance_clip: float,
    inst_color_map: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ch = np.asarray(channels_czyx, dtype=np.float32)
    fg = np.clip(ch[CH_FOREGROUND], 0.0, 1.0)
    skel = np.clip(ch[CH_SKELETON], 0.0, 1.0)
    dist = ch[CH_DISTANCE]
    fx, fy, fz = ch[CH_FLOW_X], ch[CH_FLOW_Y], ch[CH_FLOW_Z]

    fg_mip = _mip_max(fg) > 0.5
    label_mip = mask_zyx.max(axis=0).astype(np.int32)
    p_raw = _gray_to_bgr_u8(_mip_max(raw_zyx.astype(np.float32)))
    p_fg = _binary_mask_to_bgr(_mip_max(fg), threshold=0.5)
    p_skel = _prob_heatmap_bgr(_mip_max(skel), fg_mip)
    p_inst = _instance_mask_mip_bgr(mask_zyx, color_id_map=inst_color_map)
    p_dist = _distance_to_bgr(_mip_max(dist), fg_mip, label_mip, distance_clip=distance_clip)
    p_flow = _flow_rgb_fg_proj_bgr(fx, fy, fz, fg)
    row1 = np.concatenate([p_raw, p_fg, p_skel], axis=1)
    row2 = np.concatenate([p_inst, p_dist, p_flow], axis=1)
    return row1, row2

def build_target_vis_mip_sixpack(
    raw_zyx: np.ndarray,
    target_czyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    distance_clip: float = 32.0,
) -> np.ndarray:
    """
    2x3 GT-only montage (preprocessor):
      row1 — raw | foreground | skeleton
      row2 — instance mask (colored) | distance | flow
    """
    row1, row2 = _panels_from_channels(
        raw_zyx, target_czyx, mask_zyx, distance_clip=distance_clip
    )
    return np.concatenate([row1, row2], axis=0)

def build_compare_vis_mip_eightpack(
    raw_zyx: np.ndarray,
    pred_czyx: np.ndarray,
    pred_mask_zyx: np.ndarray,
    gt_target_czyx: np.ndarray,
    gt_mask_zyx: np.ndarray,
    *,
    distance_clip: float = 32.0,
    from_logits: bool = True,
) -> np.ndarray:
    """
    4x3 MIP montage (BGR uint8):
      row1-2: prediction (raw, fg, skel heatmap | inst, dist, flow)
      row3-4: ground truth (same layout; instance colors matched to GT ids)
    """
    pred_to_gt = _greedy_pred_to_gt_color_map(pred_mask_zyx, gt_mask_zyx)

    pred_fg = _sigmoid(pred_czyx[CH_FOREGROUND]) if from_logits else np.clip(
        pred_czyx[CH_FOREGROUND], 0.0, 1.0
    )
    pred_skel = _sigmoid(pred_czyx[CH_SKELETON]) if from_logits else np.clip(
        pred_czyx[CH_SKELETON], 0.0, 1.0
    )
    pred_pack = pred_czyx.copy()
    pred_pack[CH_FOREGROUND] = pred_fg
    pred_pack[CH_SKELETON] = pred_skel

    pr1, pr2 = _panels_from_channels(
        raw_zyx, pred_pack, pred_mask_zyx,
        distance_clip=distance_clip, inst_color_map=pred_to_gt,
    )
    gr1, gr2 = _panels_from_channels(
        raw_zyx, gt_target_czyx, gt_mask_zyx, distance_clip=distance_clip
    )
    return np.concatenate([pr1, pr2, gr1, gr2], axis=0)

def save_target_vis_mip_png(
    raw_zyx: np.ndarray,
    target_czyx: np.ndarray,
    out_png: Path,
    mask_zyx: np.ndarray,
    *,
    distance_clip: float = 32.0,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    montage = build_target_vis_mip_sixpack(
        raw_zyx, target_czyx, mask_zyx, distance_clip=distance_clip
    )
    if not cv2.imwrite(str(out_png), montage):
        raise RuntimeError(f"failed to write {out_png}")

def save_compare_vis_mip_png(
    raw_zyx: np.ndarray,
    pred_czyx: np.ndarray,
    pred_mask_zyx: np.ndarray,
    gt_target_czyx: np.ndarray,
    gt_mask_zyx: np.ndarray,
    out_png: Path,
    *,
    distance_clip: float = 32.0,
    from_logits: bool = True,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    montage = build_compare_vis_mip_eightpack(
        raw_zyx,
        pred_czyx,
        pred_mask_zyx,
        gt_target_czyx,
        gt_mask_zyx,
        distance_clip=distance_clip,
        from_logits=from_logits,
    )
    if not cv2.imwrite(str(out_png), montage):
        raise RuntimeError(f"failed to write {out_png}")


if __name__ == "__main__":
    main()
