#!/usr/bin/env python3
"""4D ChromoPose3D (chrp3d) shape extraction for ChromoWorld.

Resizes each T-stack to the training crop (default 64×256×256), runs twophase
decode (no watershed refine), nearest-resizes masks back to native ZYX, and
writes zlib-compressed ``{stem}_shape.tif``.

Uses the flat ChromoPose3D support modules in this directory and resolves the
default checkpoint from ``release/Checkpoint/ChromoPhy/ChromoPose``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
_RELEASE_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "Checkpoint" / "ChromoPhy" / "ChromoPose").is_dir()
    ),
    None,
)
_CHECKPOINT_ROOT = (
    _RELEASE_ROOT / "Checkpoint" / "ChromoPhy" / "ChromoPose"
    if _RELEASE_ROOT is not None
    else ROOT / "checkpoints"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import normalize_volume, resize_volume_zyx  # noqa: E402
from model import (  # noqa: E402
    CH_DISTANCE,
    CH_FLOW_X,
    CH_FLOW_Y,
    CH_FLOW_Z,
    CH_FOREGROUND,
    load_checkpoint,
)

_DEFAULT_RUN = _CHECKPOINT_ROOT / "training_run"
_DEFAULT_CKPT = _CHECKPOINT_ROOT / "last.pth"

def _load_run_args(run_dir: Path) -> dict:
    path = run_dir / "args.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    return {}

def _resolve_model_config(args: argparse.Namespace, run_args: dict) -> dict:
    return {
        "backbone": args.backbone or str(run_args.get("backbone", "unet3d")),
        "base_ch": int(args.base_ch or run_args.get("base_ch", 32)),
        "levels": int(args.levels or run_args.get("levels", 4)),
        "use_embedding": False
        if bool(args.no_embedding)
        else bool(run_args.get("use_embedding", True)),
        "size_zyx": (
            int(args.size_z or run_args.get("crop_z", 64)),
            int(args.size_y or run_args.get("crop_y", 256)),
            int(args.size_x or run_args.get("crop_x", 256)),
        ),
        "flow_scale": float(args.flow_scale)
        if float(args.flow_scale) >= 0
        else float(run_args.get("flow_scale", 5.0)),
    }

def select_usable_gpu_ids(gpu_ids: List[int], *, min_free_gib: float = 8.0) -> List[int]:
    if not torch.cuda.is_available():
        return [0]
    min_free = int(float(min_free_gib) * (1024**3))
    usable = []
    for gpu_id in gpu_ids:
        try:
            with torch.cuda.device(int(gpu_id)):
                free_bytes, _ = torch.cuda.mem_get_info()
            if int(free_bytes) >= min_free:
                usable.append(int(gpu_id))
        except Exception:
            continue
    return usable or [int(gpu_ids[0])]

def _pred_to_inputs(pred_czyx: np.ndarray, *, flow_scale: float, fg_threshold: float):
    pred = np.asarray(pred_czyx, dtype=np.float32)
    foreground_prob = 1.0 / (1.0 + np.exp(-pred[CH_FOREGROUND]))
    foreground = foreground_prob > float(fg_threshold)
    distance = np.clip(pred[CH_DISTANCE], 0.0, None)
    flow_x, flow_y, flow_z = pred[CH_FLOW_X], pred[CH_FLOW_Y], pred[CH_FLOW_Z]
    scale = float(flow_scale)
    if scale > 0 and abs(scale - 1.0) > 1e-8:
        flow_x, flow_y, flow_z = flow_x / scale, flow_y / scale, flow_z / scale
    flow = np.stack([flow_z, flow_y, flow_x], axis=0).astype(np.float32)
    return foreground, distance, flow, flow

@torch.no_grad()
def run_twophase_on_stack(
    model: torch.nn.Module,
    volume_zyx: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    size_zyx: Tuple[int, int, int],
) -> np.ndarray:
    raw = volume_zyx.astype(np.float32)
    if raw.shape != size_zyx:
        raw, _ = resize_volume_zyx(raw, np.zeros(size_zyx, dtype=np.uint16), size_zyx)
    normalized = normalize_volume(raw)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)
    prediction = model(tensor)[0].detach().cpu().numpy()
    foreground, distance, flow, d_p = _pred_to_inputs(
        prediction,
        flow_scale=float(args.flow_scale),
        fg_threshold=float(args.fg_threshold),
    )
    masks, _, _, _ = decode_instance_mask_twophase(
        foreground,
        distance,
        flow,
        dP=d_p,
        niter_phase2=int(args.niter_phase2),
        flow_step=float(args.flow_step),
        phase2_backend="cellpose",
        device=device,
    )
    return masks.astype(np.int32)

def setup_models(
    gpu_ids: List[int],
    checkpoint_path: str | Path,
    *,
    backbone: str,
    base_ch: int,
    levels: int,
    use_embedding: bool,
) -> Dict[int, torch.nn.Module]:
    models = {}
    for gpu_id in gpu_ids:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        models[gpu_id] = load_checkpoint(
            checkpoint_path,
            backbone=backbone,
            base_channels=base_ch,
            levels=levels,
            use_embedding=use_embedding,
            device=device,
        )
    return models

def resize_mask_zyx(mask: np.ndarray, size_zyx: Tuple[int, int, int]) -> np.ndarray:
    if mask.shape == size_zyx:
        return mask.astype(np.int32, copy=False)
    dummy = np.zeros_like(mask, dtype=np.float32)
    _, mask_r = resize_volume_zyx(dummy, mask.astype(np.float32), size_zyx)
    return np.rint(mask_r).astype(np.int32)

def stem_of(path: Path) -> str:
    s = path.stem
    for suf in ("-DAPI", "_DAPI", "-dapi", "_dapi"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s

def run_volume(
    vol_tzyx: np.ndarray,
    models: Dict[int, torch.nn.Module],
    gpu_ids: List[int],
    args_ns: argparse.Namespace,
    size_zyx: Tuple[int, int, int],
) -> np.ndarray:
    n_t, nz, ny, nx = vol_tzyx.shape
    out = np.zeros((n_t, nz, ny, nx), dtype=np.int32)
    workers = max(1, len(gpu_ids))

    def _one(ti: int) -> Tuple[int, np.ndarray]:
        gid = gpu_ids[ti % len(gpu_ids)]
        model = models[gid]
        device = next(model.parameters()).device
        m = run_twophase_on_stack(model, vol_tzyx[ti], device, args_ns, size_zyx)
        if m.shape != (nz, ny, nx):
            m = resize_mask_zyx(m, (nz, ny, nx))
        return ti, m

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, ti) for ti in range(n_t)]
        for fut in tqdm(as_completed(futs), total=n_t, desc="chrp3d T"):
            ti, m = fut.result()
            out[ti] = m
    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_path", type=str, default=str(ROOT / "data" / "4D" / "raw"))
    p.add_argument("--out_path", type=str, default=str(ROOT / "data" / "4D" / "mask" / "chrp3d"))
    p.add_argument("--ckpt", type=str, default=str(_DEFAULT_CKPT))
    p.add_argument("--run-dir", type=str, default=str(_DEFAULT_RUN), help="Training run dir (args.json)")
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument(
        "--axis-setup",
        type=str,
        default="",
        help="Fallback TIFF axes JSON when metadata is missing, e.g. '{\"t\":1,\"z\":1,\"y\":1,\"x\":1}'",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--size-z", type=int, default=64)
    p.add_argument("--size-y", type=int, default=256)
    p.add_argument("--size-x", type=int, default=256)
    p.add_argument("--niter-phase2", type=int, default=200)
    p.add_argument("--flow-step", type=float, default=0.75)
    p.add_argument("--fg-threshold", type=float, default=0.5)
    p.add_argument("--flow-scale", type=float, default=-1.0)
    p.add_argument("--max-files", type=int, default=0)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"ChromoPose3D ckpt not found: {ckpt}")

    run_dir = Path(args.run_dir).resolve()
    run_args = _load_run_args(run_dir) if run_dir.is_dir() else {}
    cfg_ns = argparse.Namespace(
        backbone="",
        base_ch=0,
        levels=0,
        no_embedding=False,
        size_z=int(args.size_z),
        size_y=int(args.size_y),
        size_x=int(args.size_x),
        flow_scale=float(args.flow_scale),
    )
    cfg = _resolve_model_config(cfg_ns, run_args)
    size_zyx = tuple(cfg["size_zyx"])

    files = collect_input_files(in_path)
    axis_setup = parse_axis_setup(args.axis_setup)
    if args.max_files > 0:
        files = files[: int(args.max_files)]
    if not files:
        raise SystemExit(f"No TIFFs under {in_path}")

    gpu_ids = select_usable_gpu_ids(resolve_gpu_ids(args.gpu_id), min_free_gib=8.0)
    print(f"chrp3d | GPUs={gpu_ids} size_zyx={size_zyx} n_files={len(files)}")
    print(f"ckpt={ckpt}")

    models = setup_models(
        gpu_ids,
        ckpt,
        backbone=cfg["backbone"],
        base_ch=int(cfg["base_ch"]),
        levels=int(cfg["levels"]),
        use_embedding=bool(cfg["use_embedding"]),
    )
    ns = argparse.Namespace(
        flow_scale=float(cfg["flow_scale"]),
        fg_threshold=float(args.fg_threshold),
        niter_phase2=int(args.niter_phase2),
        flow_step=float(args.flow_step),
        phase2_backend="cellpose",
    )

    for raw_file in files:
        stem = stem_of(raw_file)
        shape_path = out_path / f"{stem}{OUTPUT_SUFFIX_SHAPE}.tif"
        if shape_path.is_file() and not args.force:
            print(f"[skip] {shape_path.name}")
            continue
        t0 = time.perf_counter()
        vol_tzyx, file_axes = load_volume_4d(raw_file, axis_setup or None)
        print(f"[{stem}] {vol_tzyx.shape} axes={file_axes}", flush=True)
        mask = run_volume(vol_tzyx, models, gpu_ids, ns, size_zyx)
        save_label_volume_4d(shape_path, mask, file_axes, kind="shape")
        print(f"[{stem}] done {time.perf_counter() - t0:.1f}s -> {shape_path}", flush=True)


# ---- Embedded helpers: decode.py ----

from typing import Literal, Optional, Union

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None

try:
    import torch
    from cellpose import dynamics as cp_dynamics
except ImportError:  # pragma: no cover
    torch = None
    cp_dynamics = None

STRUCT26 = ndi.generate_binary_structure(3, 3) if ndi is not None else None

Phase2Backend = Literal["cellpose", "euler"]

def _local_endpoints_one_step(
    fg_b: np.ndarray,
    dist: np.ndarray,
    flow_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Phase 1: coord + flow * dist for each foreground voxel."""
    coords = np.column_stack(np.where(fg_b)).astype(np.float64)
    zz, yy, xx = coords[:, 0].astype(int), coords[:, 1].astype(int), coords[:, 2].astype(int)
    sampled_flow = flow_zyx[:, zz, yy, xx].astype(np.float64)
    step = dist[zz, yy, xx].astype(np.float64)
    local_ep = np.rint(coords + sampled_flow.T * step[:, None]).astype(np.int32)
    shape = fg_b.shape
    for ax, n in enumerate(shape):
        local_ep[:, ax] = np.clip(local_ep[:, ax], 0, n - 1)
    return coords, zz, yy, xx, local_ep

def _unique_local_coords(local_ep: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized unique local endpoints -> (coords Nx3, inverse index per fg voxel)."""
    ymax, xmax = int(shape[1]), int(shape[2])
    keys = (
        local_ep[:, 0].astype(np.int64) * ymax * xmax
        + local_ep[:, 1].astype(np.int64) * xmax
        + local_ep[:, 2].astype(np.int64)
    )
    uniq_keys, inv = np.unique(keys, return_inverse=True)
    zs = uniq_keys // (ymax * xmax)
    rem = uniq_keys % (ymax * xmax)
    ys = rem // xmax
    xs = rem % xmax
    local_coords = np.stack([zs, ys, xs], axis=1).astype(np.int32)
    return local_coords, inv

def _integrate_coords_euler(
    flow_zyx: np.ndarray,
    coords: np.ndarray,
    *,
    niter: int,
    step: float,
) -> np.ndarray:
    """Euler integration from N×3 (z,y,x) coordinates."""
    out = coords.astype(np.float64, copy=True)
    for _ in range(int(niter)):
        sampled = _sample_flow_trilinear(flow_zyx, out.T)
        norm = np.linalg.norm(sampled, axis=0, keepdims=True) + 1e-6
        sampled = sampled / norm
        out = out + float(step) * sampled.T
    return out

def _follow_flows_cellpose(
    dP: np.ndarray,
    local_coords: np.ndarray,
    *,
    niter: int,
    device: Union[torch.device, str, None],
) -> np.ndarray:
    """Phase 2 via cellpose ``follow_flows`` from unique local endpoints."""
    if cp_dynamics is None or torch is None:
        raise RuntimeError("cellpose and torch are required for phase2_backend='cellpose'")
    if local_coords.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    dev = device if isinstance(device, torch.device) else torch.device(
        str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    inds = (
        local_coords[:, 0].astype(np.int32),
        local_coords[:, 1].astype(np.int32),
        local_coords[:, 2].astype(np.int32),
    )
    dP_in = np.asarray(dP, dtype=np.float32) / 5.0
    p_final = cp_dynamics.follow_flows(dP_in, inds, niter=int(niter), device=dev)
    if torch.is_tensor(p_final):
        p = p_final.detach().float().cpu().numpy()
    else:
        p = np.asarray(p_final, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.shape[0] == 3 and p.shape[1] != 3:
        p = p.T
    global_ep = np.rint(p).astype(np.int32)
    for ax, n in enumerate(dP.shape[1:]):
        global_ep[:, ax] = np.clip(global_ep[:, ax], 0, int(n) - 1)
    return global_ep

def _fill_missing_labels(pred: np.ndarray, fg_b: np.ndarray) -> np.ndarray:
    missing = fg_b & (pred == 0)
    if bool(missing.any()) and bool((pred > 0).any()):
        inds = ndi.distance_transform_edt(
            pred == 0, return_distances=False, return_indices=True
        )
        pred[missing] = pred[tuple(inds[:, missing])]
    return pred

def decode_instance_mask_ceiling(
    fg: np.ndarray,
    dist: np.ndarray,
    flow_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover instance labels from foreground, distance-to-skeleton, and unit flow.

    Args:
        fg: (Z,Y,X) bool or float foreground
        dist: (Z,Y,X) distance to skeleton
        flow_zyx: (3,Z,Y,X) unit flow in (fz, fy, fx) order

    Returns:
        pred: (Z,Y,X) uint16 instance mask
        seeds: (Z,Y,X) bool endpoint seed voxels
    """
    if ndi is None:
        raise RuntimeError("scipy is required")
    fg_b = fg.astype(bool)
    shape = fg_b.shape
    pred = np.zeros(shape, dtype=np.uint16)
    seeds = np.zeros(shape, dtype=bool)
    if not bool(fg_b.any()):
        return pred, seeds

    _coords, zz, yy, xx, local_ep = _local_endpoints_one_step(fg_b, dist, flow_zyx)
    seeds[tuple(local_ep.T)] = True
    seed_labels, _ = ndi.label(seeds, structure=STRUCT26)
    pred[zz, yy, xx] = seed_labels[tuple(local_ep.T)].astype(np.uint16)
    pred = _fill_missing_labels(pred, fg_b)
    return pred, seeds

def decode_instance_mask_twophase(
    fg: np.ndarray,
    dist: np.ndarray,
    flow_zyx: np.ndarray,
    *,
    dP: Optional[np.ndarray] = None,
    niter_phase2: int = 200,
    flow_step: float = 0.75,
    phase2_backend: Phase2Backend = "cellpose",
    device: Union[torch.device, str, None] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """
    Two-phase decode:
      Phase 1 — one-step flow*dist -> local endpoint per foreground voxel
      Phase 2 — follow flows from each unique local endpoint -> global endpoint
                (cellpose ``follow_flows`` by default, or legacy Euler)
      Instance id — 26-connected components on global endpoint seeds

    Args:
        fg: foreground mask
        dist: distance-to-skeleton
        flow_zyx: unit flow (fz, fy, fx) for phase 1
        dP: cellpose dynamics flow (fz, fy, fx), unnormalized; required for cellpose phase 2
        niter_phase2: integration steps
        flow_step: Euler step size (euler backend only)
        phase2_backend: ``cellpose`` or ``euler``
        device: torch device for cellpose phase 2

    Returns:
        pred mask, local seed mask, global seed mask, stats dict
    """
    if ndi is None:
        raise RuntimeError("scipy is required")
    fg_b = fg.astype(bool)
    shape = fg_b.shape
    pred = np.zeros(shape, dtype=np.uint16)
    local_seeds = np.zeros(shape, dtype=bool)
    global_seeds = np.zeros(shape, dtype=bool)
    empty_stats = {
        "n_local_seeds": 0,
        "n_global_seeds": 0,
        "n_global_components": 0,
        "phase2_backend": str(phase2_backend),
    }
    if not bool(fg_b.any()):
        return pred, local_seeds, global_seeds, empty_stats

    _coords, zz, yy, xx, local_ep = _local_endpoints_one_step(fg_b, dist, flow_zyx)
    local_seeds[tuple(local_ep.T)] = True

    local_coords, inv = _unique_local_coords(local_ep, shape)
    n_local = int(len(local_coords))

    backend = str(phase2_backend).lower()
    if backend == "cellpose":
        if dP is None:
            raise ValueError("dP is required when phase2_backend='cellpose'")
        global_ep = _follow_flows_cellpose(
            dP, local_coords, niter=int(niter_phase2), device=device
        )
    elif backend == "euler":
        global_ep = np.rint(
            _integrate_coords_euler(
                flow_zyx,
                local_coords.astype(np.float64),
                niter=int(niter_phase2),
                step=float(flow_step),
            )
        ).astype(np.int32)
        for ax, n in enumerate(shape):
            global_ep[:, ax] = np.clip(global_ep[:, ax], 0, n - 1)
    else:
        raise ValueError(f"unsupported phase2_backend: {phase2_backend!r}")

    global_seeds[tuple(global_ep.T)] = True
    global_labels, n_global_comp = ndi.label(global_seeds, structure=STRUCT26)
    per_local_global = global_labels[tuple(global_ep.T)].astype(np.int32)
    pred[zz, yy, xx] = per_local_global[inv].astype(np.uint16)
    pred = _fill_missing_labels(pred, fg_b)

    stats = {
        "n_local_seeds": n_local,
        "n_global_seeds": int(global_seeds.sum()),
        "n_global_components": int(n_global_comp),
        "phase2_backend": backend,
    }
    return pred, local_seeds, global_seeds, stats


# ---- TIFF axis and volume I/O helpers (formerly volume_io.py) ----

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import tifffile
import torch

INTERNAL_AXES = "TZYX"
OUTPUT_SUFFIX_SHAPE = "_shape"
OUTPUT_SUFFIX_TRACKED = "_tracked"

def parse_gpu_ids(gpu_arg):
    if isinstance(gpu_arg, int):
        return [int(gpu_arg)]
    if isinstance(gpu_arg, list):
        return [int(x) for x in gpu_arg]
    s = str(gpu_arg).strip()
    if s == "":
        return [0]
    if s.startswith("[") and s.endswith("]"):
        import ast

        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [int(x) for x in parsed]
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    return [int(s)]

def parse_axis_setup(raw: str | dict | None) -> dict[str, float]:
    """Same convention as visualize_4D_TIFF2GIF --vis_setup: key order maps dim0..dimN-1."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        data = raw
    else:
        raw_s = str(raw).strip()
        if not raw_s:
            return {}
        try:
            data = json.loads(raw_s)
        except Exception:
            data = json.loads(raw_s.replace("'", '"'))
    if not isinstance(data, dict):
        raise ValueError("axis_setup must be a JSON object")
    normalized: dict[str, float] = {}
    for k, v in data.items():
        key = str(k).strip().lower()
        if key in {"time"}:
            key = "t"
        if key not in {"t", "c", "x", "y", "z"}:
            raise ValueError(f"Unsupported axis_setup key: {key!r}, expect one of t/c/x/y/z")
        if key in normalized:
            raise ValueError(f"duplicated axis in axis_setup: {key!r}")
        ratio = float(v)
        if not np.isfinite(ratio):
            raise ValueError(f"axis_setup[{key}] must be finite, got {v!r}")
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"axis_setup[{key}] must be in (0,1], got {v!r}")
        normalized[key] = ratio
    return normalized

def axis_setup_to_file_axes(axis_setup: dict[str, float]) -> str:
    keys = [k for k in axis_setup if k in {"t", "z", "c", "x", "y"}]
    if not keys:
        raise ValueError("axis_setup must include at least one axis key")
    required = {"x", "y", "z"}
    missing = sorted(required - set(keys))
    if missing:
        raise ValueError(f"axis_setup must include x,y,z; missing {missing}")
    if len(keys) != 4 or "t" not in keys:
        raise ValueError('axis_setup for 4D volumes must include exactly t,z,x,y (4 keys, no channel)')
    return "".join(k.upper() for k in keys)

def _normalize_file_axes(axes: str) -> str:
    axes_u = axes.upper().replace("TIME", "T")
    if len(axes_u) != 4:
        raise ValueError(f"Expected 4 axis labels, got {axes!r}")
    if len(set(axes_u)) != 4:
        raise ValueError(f"Axis labels must be unique, got {axes!r}")
    for c in axes_u:
        if c not in set(INTERNAL_AXES):
            raise ValueError(f"Unsupported axis {c!r} in {axes!r}; expected subset of TZYX")
    if set(axes_u) != set(INTERNAL_AXES):
        raise ValueError(f"4D volume axes must be a permutation of TZYX, got {axes!r}")
    return axes_u

def _squeeze_leading_singletons(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 4 and out.shape[0] == 1:
        out = np.squeeze(out, axis=0)
    return out

def _read_tiff_axes(path: Path) -> str | None:
    with tifffile.TiffFile(str(path)) as tf:
        if not tf.series:
            return None
        series = tf.series[0]
        axes = getattr(series, "axes", None)
        if axes is None:
            return None
        axes_s = str(axes).strip()
        return axes_s if axes_s else None

def file_to_internal(arr: np.ndarray, file_axes: str) -> np.ndarray:
    file_axes = _normalize_file_axes(file_axes)
    perm = [file_axes.index(c) for c in INTERNAL_AXES]
    return np.transpose(arr, perm)

def internal_to_file(vol: np.ndarray, file_axes: str) -> np.ndarray:
    file_axes = _normalize_file_axes(file_axes)
    perm = [INTERNAL_AXES.index(c) for c in file_axes]
    return np.transpose(vol, perm)

def _resolve_file_axes(path: Path, arr: np.ndarray, axis_setup: dict[str, float] | None) -> str:
    """Prefer TIFF series metadata; fall back to ``--axis_setup`` when missing/invalid."""
    meta_axes = _read_tiff_axes(path)
    if meta_axes:
        try:
            return _normalize_file_axes(meta_axes)
        except ValueError as exc:
            print(f"Warning: ignoring invalid TIFF axes {meta_axes!r} on {path}: {exc}")
    if axis_setup:
        return axis_setup_to_file_axes(axis_setup)
    _exit_missing_axes(path, arr)
    raise RuntimeError("unreachable")

def _exit_missing_axes(path: Path, arr: np.ndarray) -> None:
    squeezed = _squeeze_leading_singletons(arr)
    dim_desc = [f"{i}:{size}" for i, size in enumerate(squeezed.shape)]
    print(f"No axis metadata on: {path}")
    print(f"  raw shape: {arr.shape}")
    print(f"  squeezed shape: {squeezed.shape}")
    print(f"  dimension list: {dim_desc}")
    print(
        "Specify axis order (same as visualize_4D_TIFF2GIF --vis_setup), e.g.\n"
        '  --axis_setup \'{"t":1.0,"z":1.0,"x":1.0,"y":1.0}\'\n'
        "Key order maps dimension 0,1,2,3 to t,z,x,y. For shape (64,100,401,401) use the example above (TZYX)."
    )
    sys.exit(1)

def load_volume_4d(path: Path, axis_setup: dict[str, float] | None) -> tuple[np.ndarray, str]:
    with tifffile.TiffFile(str(path)) as tf:
        if not tf.series:
            raise ValueError(f"No TIFF series in {path}")
        arr = np.asarray(tf.series[0].asarray())
    arr = _squeeze_leading_singletons(arr)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D volume, got shape={arr.shape} from {path}")
    file_axes = _resolve_file_axes(path, arr, axis_setup)
    vol_tzyx = file_to_internal(arr, file_axes).astype(np.float32)
    return vol_tzyx, file_axes

LABEL_DTYPE_META_KEY = "label_dtype"
LABEL_MAX_ID_META_KEY = "max_label_id"

def max_label_id(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    return int(np.max(arr))

def dtype_for_label_max(max_id: int) -> np.dtype:
    if max_id <= int(np.iinfo(np.uint8).max):
        return np.dtype(np.uint8)
    if max_id <= int(np.iinfo(np.uint16).max):
        return np.dtype(np.uint16)
    if max_id <= int(np.iinfo(np.uint32).max):
        return np.dtype(np.uint32)
    return np.dtype(np.int32)

def cast_label_volume(vol: np.ndarray, dtype: np.dtype | None = None) -> tuple[np.ndarray, np.dtype, int]:
    mid = max_label_id(vol)
    if dtype is None:
        dtype = dtype_for_label_max(mid)
    if mid > int(np.iinfo(dtype).max):
        raise ValueError(f"max label id {mid} exceeds storage dtype {dtype}")
    return np.asarray(vol, dtype=dtype), dtype, mid

def save_label_volume_4d(
    path: Path,
    vol_tzyx: np.ndarray,
    file_axes: str,
    *,
    kind: str = "labels",
    dtype: np.dtype | None = None,
) -> dict[str, Any]:
    vol_cast, store_dtype, max_id = cast_label_volume(vol_tzyx, dtype=dtype)
    out = internal_to_file(vol_cast, file_axes)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "axes": _normalize_file_axes(file_axes),
        LABEL_DTYPE_META_KEY: store_dtype.name,
        LABEL_MAX_ID_META_KEY: str(max_id),
    }
    tifffile.imwrite(
        str(path),
        out,
        photometric="minisblack",
        metadata=metadata,
        compression="zlib",
        compressionargs={"level": 6},
    )
    summary = {
        "kind": kind,
        "path": str(path),
        "dtype": store_dtype.name,
        "max_label_id": max_id,
        "shape_tzyx": list(vol_cast.shape),
    }
    print(
        f"  saved {kind}: dtype={store_dtype.name} max_id={max_id} "
        f"shape(T,Z,Y,X)={tuple(vol_cast.shape)}"
    )
    return summary

def resolve_gpu_ids(gpu_arg) -> List[int]:
    gpu_ids = parse_gpu_ids(gpu_arg)
    if torch.cuda.is_available():
        n_cuda = torch.cuda.device_count()
        gpu_ids = [g for g in gpu_ids if 0 <= g < n_cuda]
        if not gpu_ids:
            gpu_ids = [0]
    else:
        gpu_ids = [0]
    return gpu_ids

def _is_tiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}

def _should_skip_input_tiff(path: Path, exclude_suffixes: tuple[str, ...]) -> bool:
    stem = path.stem
    if any(stem.endswith(suffix) for suffix in exclude_suffixes):
        return True
    # macOS archive metadata (not real TIFF volumes)
    if path.name.startswith("._"):
        return True
    if "__MACOSX" in path.parts or "__macosx" in {p.lower() for p in path.parts}:
        return True
    return False

def collect_input_files(
    in_path: Path,
    *,
    require_suffix: str | None = None,
    exclude_suffixes: tuple[str, ...] = (OUTPUT_SUFFIX_SHAPE, OUTPUT_SUFFIX_TRACKED),
) -> List[Path]:
    """Collect .tif inputs from a single file or all TIFFs under a directory."""
    in_path = in_path.resolve()
    if in_path.is_file():
        if not _is_tiff(in_path):
            raise ValueError(f"Expected .tif/.tiff file: {in_path}")
        return [in_path]

    files: list[Path] = []
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        files.extend(in_path.rglob(f"*{ext}"))
    out: list[Path] = []
    for p in sorted({fp.resolve() for fp in files if fp.is_file()}):
        if _should_skip_input_tiff(p, exclude_suffixes):
            continue
        if require_suffix is not None and not p.stem.endswith(require_suffix):
            continue
        out.append(p)
    return out


if __name__ == "__main__":
    main()
