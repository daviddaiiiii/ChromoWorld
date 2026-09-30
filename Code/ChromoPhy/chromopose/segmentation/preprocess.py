#!/usr/bin/env python3
"""Prepare 4D Chimp images and instance masks as ChromoPose training targets."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from dataloader import (  # noqa: E402
    SamplePair,
    VolumeRecord,
    _sid_from_pair,
    collect_pairs,
    collect_chimp_pairs,
    load_volume_index,
    read_tiff,
    resize_volume_zyx,
    split_t_volumes,
    write_index,
)
from model import TARGET_CHANNEL_NAMES  # noqa: E402
from visualize import save_target_vis_mip_png  # noqa: E402

# ---- Target generation helpers (formerly prepare_data.py) ----

def process_one_volume(
    pair: SamplePair,
    t_idx: int,
    raw_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    out_root: Path,
    *,
    size_zyx: Tuple[int, int, int],
    distance_clip: float,
    use_embedding: bool,
    skip_existing: bool,
    refresh_vis: bool,
) -> Optional[VolumeRecord]:
    sid = _sid_from_pair(pair)
    stem = f"{sid}_t{t_idx:03d}"
    out_raw = out_root / "raw" / f"{stem}.tif"
    out_mask = out_root / "mask" / f"{stem}.tif"
    out_target = out_root / "target" / f"{stem}.tif"
    out_target_vis = out_root / "target_vis" / f"{stem}.png"

    if skip_existing and out_target.is_file() and out_mask.is_file() and out_raw.is_file():
        raw_r = tifffile.imread(out_raw).astype(np.float32)
        target = tifffile.imread(out_target).astype(np.float32)
        mask = tifffile.imread(out_mask)
        if refresh_vis or not out_target_vis.is_file():
            save_target_vis_mip_png(
                raw_r, target, out_target_vis, mask, distance_clip=float(distance_clip)
            )
        n_inst = len(np.unique(mask)) - 1
        z, y, x = mask.shape
        split = pair.folder.parent.name if pair.folder.parent.name in {"train", "test"} else ""
        return VolumeRecord(
            sid=sid,
            t=t_idx,
            raw=str(out_raw.relative_to(out_root)),
            mask=str(out_mask.relative_to(out_root)),
            target=str(out_target.relative_to(out_root)),
            target_vis=str(out_target_vis.relative_to(out_root)),
            nz=int(z),
            ny=int(y),
            nx=int(x),
            n_inst=int(n_inst),
            split=split,
            celltype=pair.celltype,
            sample_num=pair.sample_num,
        )

    raw_r, mask_in = resize_volume_zyx(raw_zyx, mask_zyx, size_zyx)
    if int(np.max(mask_in)) <= 0:
        return None
    try:
        mask_gt, target = labels_to_chromopose_target_3d(
            mask_in,
            distance_clip=distance_clip,
            use_embedding=use_embedding,
        )
    except Exception as exc:
        print(f"[WARN] skip {stem}: {exc}")
        return None
    n_inst = int(len(np.unique(mask_gt)) - 1)

    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    out_target.parent.mkdir(parents=True, exist_ok=True)
    out_target_vis.parent.mkdir(parents=True, exist_ok=True)
    if tifffile is None:
        raise RuntimeError("tifffile is required")

    tifffile.imwrite(out_raw, raw_r.astype(np.float32))
    tifffile.imwrite(out_mask, mask_gt.astype(np.uint16))
    tifffile.imwrite(
        out_target,
        target,
        metadata={"axes": "CZYX", "Channel": {"Name": list(TARGET_CHANNEL_NAMES)}},
    )
    save_target_vis_mip_png(
        raw_r, target, out_target_vis, mask_gt, distance_clip=float(distance_clip)
    )

    split = pair.folder.parent.name if pair.folder.parent.name in {"train", "test"} else ""
    return VolumeRecord(
        sid=sid,
        t=t_idx,
        raw=str(out_raw.relative_to(out_root)),
        mask=str(out_mask.relative_to(out_root)),
        target=str(out_target.relative_to(out_root)),
        target_vis=str(out_target_vis.relative_to(out_root)),
        nz=int(target.shape[1]),
        ny=int(target.shape[2]),
        nx=int(target.shape[3]),
        n_inst=n_inst,
        split=split,
        celltype=pair.celltype,
        sample_num=pair.sample_num,
    )

def process_sample(
    pair: SamplePair,
    out_root: Path,
    *,
    size_zyx: Tuple[int, int, int],
    distance_clip: float,
    use_embedding: bool,
    skip_existing: bool,
    refresh_vis: bool,
) -> List[VolumeRecord]:
    raw_4d = read_tiff(pair.raw_path)
    mask_4d = read_tiff(pair.mask_path)
    records: List[VolumeRecord] = []
    for t_idx, raw_zyx, mask_zyx in split_t_volumes(raw_4d, mask_4d):
        rec = process_one_volume(
            pair,
            t_idx,
            raw_zyx,
            mask_zyx,
            out_root,
            size_zyx=size_zyx,
            distance_clip=distance_clip,
            use_embedding=use_embedding,
            skip_existing=skip_existing,
            refresh_vis=refresh_vis,
        )
        if rec is not None:
            records.append(rec)
    return records

# Per-worker cache removed — use cell-level batch jobs instead.

def _job_from_pair_cell(
    pair: SamplePair,
    t_indices: Tuple[int, ...],
    out_root: Path,
    *,
    size_zyx: Tuple[int, int, int],
    distance_clip: float,
    use_embedding: bool,
    skip_existing: bool,
    refresh_vis: bool,
) -> Tuple:
    return (
        str(pair.folder),
        pair.celltype,
        pair.sample_num,
        str(pair.raw_path.resolve()),
        str(pair.mask_path.resolve()),
        tuple(int(t) for t in t_indices),
        str(out_root.resolve()),
        size_zyx[0],
        size_zyx[1],
        size_zyx[2],
        float(distance_clip),
        bool(use_embedding),
        bool(skip_existing),
        bool(refresh_vis),
    )

def _process_cell_job(
    job: Tuple[
        str,
        str,
        str,
        str,
        str,
        Tuple[int, ...],
        str,
        int,
        int,
        int,
        float,
        bool,
        bool,
        bool,
    ],
) -> List[dict]:
    """Process all pending T frames for one cell (single 4D read)."""
    (
        folder,
        celltype,
        sample_num,
        raw_path,
        mask_path,
        t_indices,
        out_root_str,
        sz,
        sy,
        sx,
        distance_clip,
        use_embedding,
        skip_existing,
        refresh_vis,
    ) = job
    pair = SamplePair(
        folder=Path(folder),
        celltype=celltype,
        sample_num=sample_num,
        raw_path=Path(raw_path),
        mask_path=Path(mask_path),
    )
    out_root = Path(out_root_str)
    size_zyx = (int(sz), int(sy), int(sx))
    t_set = {int(t) for t in t_indices}
    raw_4d = read_tiff(pair.raw_path)
    mask_4d = read_tiff(pair.mask_path)
    records: List[dict] = []
    for t_idx, raw_zyx, mask_zyx in split_t_volumes(raw_4d, mask_4d):
        if int(t_idx) not in t_set:
            continue
        rec = process_one_volume(
            pair,
            int(t_idx),
            raw_zyx,
            mask_zyx,
            out_root,
            size_zyx=size_zyx,
            distance_clip=float(distance_clip),
            use_embedding=bool(use_embedding),
            skip_existing=bool(skip_existing),
            refresh_vis=bool(refresh_vis),
        )
        if rec is not None:
            records.append(asdict(rec))
    return records

def _job_from_pair_t(
    pair: SamplePair,
    t_idx: int,
    out_root: Path,
    *,
    size_zyx: Tuple[int, int, int],
    distance_clip: float,
    use_embedding: bool,
    skip_existing: bool,
    refresh_vis: bool,
) -> Tuple:
    return _job_from_pair_cell(
        pair, (int(t_idx),), out_root,
        size_zyx=size_zyx, distance_clip=distance_clip,
        use_embedding=use_embedding, skip_existing=skip_existing, refresh_vis=refresh_vis,
    )

def _refresh_vis_job(job: Tuple[str, str, float, bool]) -> Optional[str]:
    """Regenerate target_vis PNG from existing raw + target."""
    raw_path, target_path, distance_clip, refresh = job
    raw_path_p = Path(raw_path)
    target_path_p = Path(target_path)
    out_png = raw_path_p.parent.parent / "target_vis" / f"{raw_path_p.stem}.png"
    if out_png.is_file() and not refresh:
        return str(out_png)
    if tifffile is None:
        raise RuntimeError("tifffile is required")
    if not raw_path_p.is_file() or not target_path_p.is_file():
        return None
    raw = tifffile.imread(raw_path_p).astype(np.float32)
    target = tifffile.imread(target_path_p).astype(np.float32)
    mask_path = target_path_p.parent.parent / "mask" / f"{raw_path_p.stem}.tif"
    mask = tifffile.imread(mask_path) if mask_path.is_file() else np.zeros(raw.shape, dtype=np.uint16)
    save_target_vis_mip_png(raw, target, out_png, mask, distance_clip=float(distance_clip))
    return str(out_png)

def _run_refresh_vis_only(out_root: Path, records: List[VolumeRecord], *, distance_clip: float, workers: int, refresh: bool) -> List[VolumeRecord]:
    jobs = []
    for rec in records:
        raw_p = out_root / rec.raw
        tgt_p = out_root / rec.target
        jobs.append((str(raw_p), str(tgt_p), float(distance_clip), bool(refresh)))
    print(f"Refreshing {len(jobs)} target_vis PNGs with {workers} workers")
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_refresh_vis_job, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="target_vis"):
            try:
                fut.result()
            except Exception as exc:
                print(f"[WARN] vis refresh failed: {exc}")
    updated: List[VolumeRecord] = []
    for rec in records:
        stem = Path(rec.raw).stem
        rec.target_vis = str(Path("target_vis") / f"{stem}.png")
        updated.append(rec)
    write_index(updated, out_root / "index.jsonl")
    return updated

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess 4D chimp data into ChromoPose targets")
    p.add_argument("--chimp-root", type=str, default="")
    p.add_argument("--source", type=str, default="")
    p.add_argument("--out-root", type=str, required=True)
    p.add_argument("--size-z", type=int, default=64)
    p.add_argument("--size-y", type=int, default=256)
    p.add_argument("--size-x", type=int, default=256)
    p.add_argument("--distance-clip", type=float, default=32.0)
    p.add_argument("--no-embedding", action="store_true")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--refresh-vis", action="store_true", help="regenerate target_vis PNG even if exists")
    p.add_argument("--refresh-vis-only", action="store_true", help="only regenerate target_vis PNGs from index")
    p.add_argument("--workers", type=int, default=16, help="parallel worker processes")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    if tifffile is None:
        raise RuntimeError("tifffile is required")
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    size_zyx = (int(args.size_z), int(args.size_y), int(args.size_x))

    if str(args.chimp_root).strip():
        pairs = collect_chimp_pairs(Path(args.chimp_root).resolve())
    elif str(args.source).strip():
        pairs = collect_pairs(Path(args.source).resolve())
    else:
        raise RuntimeError("Provide --chimp-root or --source")
    if not pairs:
        raise RuntimeError("No samples found")
    print(f"Found {len(pairs)} cell folders (ChromoPose skeleton-flow targets)")

    skip_existing = not bool(args.no_skip_existing)
    refresh_vis = bool(args.refresh_vis)
    all_records: List[VolumeRecord] = load_volume_index(out_root / "index.jsonl") if (out_root / "index.jsonl").is_file() else []
    if bool(args.no_skip_existing):
        all_records = []

    if bool(args.refresh_vis_only):
        if not all_records:
            raise RuntimeError("index.jsonl missing or empty; run preprocessing first")
        _run_refresh_vis_only(
            out_root,
            all_records,
            distance_clip=float(args.distance_clip),
            workers=int(args.workers),
            refresh=True,
        )
        print(f"Refreshed target_vis PNGs under {out_root / 'target_vis'}")
        return

    existing_keys = {(r.sid, r.t) for r in all_records}

    jobs: List[Tuple] = []
    for pair in pairs:
        sid = _sid_from_pair(pair)
        raw_4d = read_tiff(pair.raw_path)
        mask_4d = read_tiff(pair.mask_path)
        pending_t: List[int] = []
        for t_idx, _, _ in split_t_volumes(raw_4d, mask_4d):
            if skip_existing and (sid, int(t_idx)) in existing_keys and not refresh_vis:
                vis_png = out_root / "target_vis" / f"{sid}_t{int(t_idx):03d}.png"
                if vis_png.is_file():
                    continue
            pending_t.append(int(t_idx))
        if pending_t:
            jobs.append(
                _job_from_pair_cell(
                    pair,
                    tuple(pending_t),
                    out_root,
                    size_zyx=size_zyx,
                    distance_clip=float(args.distance_clip),
                    use_embedding=not bool(args.no_embedding),
                    skip_existing=skip_existing,
                    refresh_vis=refresh_vis,
                )
            )

    n_pending = sum(len(j[5]) for j in jobs)
    n_workers = max(1, int(args.workers))
    print(
        f"Scheduling {n_pending} volumes in {len(jobs)} cell jobs "
        f"with {n_workers} workers (skip_existing={skip_existing})"
    )

    if jobs:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_cell_job, job) for job in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="cells"):
                try:
                    rec_dicts = fut.result()
                except Exception as exc:
                    print(f"[WARN] worker failed: {exc}")
                    continue
                for rec_dict in rec_dicts:
                    all_records.append(VolumeRecord(**rec_dict))
                dedup = {(r.sid, r.t): r for r in all_records}
                all_records = list(dedup.values())
                write_index(all_records, out_root / "index.jsonl")

    dedup = {(r.sid, r.t): r for r in all_records}
    all_records = list(dedup.values())
    write_index(all_records, out_root / "index.jsonl")
    print(f"Wrote {len(all_records)} volumes to {out_root}")

# ---- Instance-mask to skeleton-flow targets (formerly targets.py) ----

from collections import deque

from model import (
    BASE_TARGET_CHANNELS,
    CH_BOUNDARY,
    CH_DISTANCE,
    CH_EMB_0,
    CH_FLOW_X,
    CH_FLOW_Y,
    CH_FLOW_Z,
    CH_FOREGROUND,
    CH_SKELETON,
    EMBEDDING_DIM,
    TARGET_CHANNELS,
)

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None

try:
    import fastremap
except ImportError:  # pragma: no cover
    fastremap = None

try:
    from skimage.morphology import skeletonize as skimage_skeletonize
except ImportError:  # pragma: no cover
    skimage_skeletonize = None

try:
    from cellpose import utils as cp_utils
except Exception:  # pragma: no cover
    cp_utils = None

_NEIGHBOR_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]

def _require_deps() -> None:
    if ndi is None:
        raise RuntimeError("scipy is required")
    if fastremap is None:
        raise RuntimeError("fastremap is required")
    if skimage_skeletonize is None:
        raise RuntimeError("skimage is required for 3D skeletonization")

def clean_instance_mask_3d(mask_zyx: np.ndarray, *, fill_holes: bool = True) -> np.ndarray:
    """Per-instance hole filling; keep all instances."""
    _require_deps()
    lbl = mask_zyx.astype(np.int32)
    if lbl.ndim != 3:
        raise ValueError(f"mask must be (Z,Y,X), got {lbl.shape}")
    lbl, _ = fastremap.renumber(lbl, in_place=True)
    out = np.zeros_like(lbl, dtype=np.int32)
    next_id = 1
    for inst_id, sl in enumerate(ndi.find_objects(lbl) or [], start=1):
        if sl is None:
            continue
        zsl, ysl, xsl = sl
        inst_crop = lbl[zsl, ysl, xsl] == inst_id
        if not int(inst_crop.sum()):
            continue
        inst_work = inst_crop
        if fill_holes:
            inst_work = ndi.binary_fill_holes(inst_work)
        out[zsl, ysl, xsl][inst_work] = next_id
        next_id += 1
    out, _ = fastremap.renumber(out, in_place=True)
    return out.astype(np.uint16)

def _boundary_from_label(lbl: np.ndarray, inst_mask: np.ndarray) -> np.ndarray:
    if cp_utils is not None and hasattr(cp_utils, "masks_to_edges"):
        edges = cp_utils.masks_to_edges(lbl.astype(np.int32), threshold=1.0).astype(np.float32)
        return np.clip(edges, 0.0, 1.0)
    eroded = ndi.binary_erosion(inst_mask, iterations=1)
    return (inst_mask & ~eroded).astype(np.float32)

def skeletonize_instance_3d(inst_mask: np.ndarray) -> np.ndarray:
    """Extract thin 3D medial skeleton via full skeletonize."""
    _require_deps()
    if not bool(inst_mask.any()):
        return np.zeros_like(inst_mask, dtype=bool)
    skel = skimage_skeletonize(inst_mask.astype(bool))
    if not bool(skel.any()):
        dist = ndi.distance_transform_edt(inst_mask)
        skel = (dist >= float(dist.max()) - 1e-6) & inst_mask
    return skel.astype(bool)

def _skeleton_degree(skel: np.ndarray, z: int, y: int, x: int) -> int:
    zmax, ymax, xmax = skel.shape
    deg = 0
    for dz, dy, dx in _NEIGHBOR_OFFSETS:
        zz, yy, xx = z + dz, y + dy, x + dx
        if 0 <= zz < zmax and 0 <= yy < ymax and 0 <= xx < xmax and skel[zz, yy, xx]:
            deg += 1
    return deg

def _prune_short_spurs(skel: np.ndarray, *, min_len: int = 4) -> np.ndarray:
    """Remove skeleton branches shorter than min_len voxels from endpoints."""
    out = skel.astype(bool).copy()
    if min_len <= 0 or not bool(out.any()):
        return out
    for _ in range(min_len):
        coords = np.column_stack(np.where(out))
        if len(coords) == 0:
            break
        to_remove: List[Tuple[int, int, int]] = []
        for z, y, x in coords:
            if _skeleton_degree(out, int(z), int(y), int(x)) == 1:
                to_remove.append((int(z), int(y), int(x)))
        if not to_remove:
            break
        for z, y, x in to_remove:
            out[z, y, x] = False
    return out

def _geodesic_from_endpoints(skel: np.ndarray) -> np.ndarray:
    """
    Geodesic distance on skeleton graph to the nearest endpoint (degree == 1).
    Higher values are farther from tips (toward interior).
    """
    shape = skel.shape
    geo = np.full(shape, np.inf, dtype=np.float32)
    if not bool(skel.any()):
        return geo

    coords = np.column_stack(np.where(skel))
    endpoints = [
        (int(z), int(y), int(x))
        for z, y, x in coords
        if _skeleton_degree(skel, int(z), int(y), int(x)) == 1
    ]
    if not endpoints:
        # Closed loop: seed from voxel farthest from centroid.
        center = coords.mean(axis=0)
        d2 = np.sum((coords - center) ** 2, axis=1)
        seed = tuple(int(v) for v in coords[int(np.argmax(d2))])
        endpoints = [seed]

    q: deque = deque()
    for z, y, x in endpoints:
        geo[z, y, x] = 0.0
        q.append((z, y, x))

    zmax, ymax, xmax = shape
    while q:
        z, y, x = q.popleft()
        base = float(geo[z, y, x])
        for dz, dy, dx in _NEIGHBOR_OFFSETS:
            zz, yy, xx = z + dz, y + dy, x + dx
            if zz < 0 or yy < 0 or xx < 0 or zz >= zmax or yy >= ymax or xx >= xmax:
                continue
            if not skel[zz, yy, xx]:
                continue
            step = float(np.sqrt(dz * dz + dy * dy + dx * dx))
            nd = base + step
            if nd < geo[zz, yy, xx]:
                geo[zz, yy, xx] = nd
                q.append((zz, yy, xx))
    return geo

def _tangent_speed_on_skeleton(
    skel: np.ndarray,
    geo: np.ndarray,
    *,
    speed_clip: float = 0.0,
) -> np.ndarray:
    """
    Remaining geodesic distance to the interior attractor (skeleton geo maximum).

    Zero at attractor voxels; largest near endpoints. Used to scale the tangent
    flow component so Phase-2 integration does not overshoot the branch center.
    """
    vpara = np.zeros(skel.shape, dtype=np.float32)
    if not bool(skel.any()):
        return vpara
    geo_on = geo[skel]
    finite = np.isfinite(geo_on)
    if not bool(finite.any()):
        return vpara
    geo_max = float(geo_on[finite].max())
    if geo_max <= 0.0:
        return vpara
    on_skel = skel & np.isfinite(geo)
    rem = geo_max - geo[on_skel]
    if float(speed_clip) > 0.0:
        rem = np.clip(rem, 0.0, float(speed_clip))
    else:
        rem = np.clip(rem, 0.0, geo_max)
    vpara[on_skel] = rem.astype(np.float32)
    return vpara

def _oriented_tangent_on_skeleton(skel: np.ndarray, geo: np.ndarray) -> np.ndarray:
    """
    Unit tangent on skeleton voxels pointing away from endpoints (toward interior).
    Returns (3, Z, Y, X) in (fz, fy, fx) order.
    """
    shape = skel.shape
    tangent = np.zeros((3, *shape), dtype=np.float32)
    if not bool(skel.any()):
        return tangent

    coords = np.column_stack(np.where(skel))
    zmax, ymax, xmax = shape
    for z, y, x in coords:
        z, y, x = int(z), int(y), int(x)
        best = None
        best_geo = float(geo[z, y, x])
        for dz, dy, dx in _NEIGHBOR_OFFSETS:
            zz, yy, xx = z + dz, y + dy, x + dx
            if zz < 0 or yy < 0 or xx < 0 or zz >= zmax or yy >= ymax or xx >= xmax:
                continue
            if not skel[zz, yy, xx]:
                continue
            g = float(geo[zz, yy, xx])
            if g > best_geo + 1e-6:
                best_geo = g
                best = (dz, dy, dx)
        if best is None:
            for dz, dy, dx in _NEIGHBOR_OFFSETS:
                zz, yy, xx = z + dz, y + dy, x + dx
                if zz < 0 or yy < 0 or xx < 0 or zz >= zmax or yy >= ymax or xx >= xmax:
                    continue
                if not skel[zz, yy, xx]:
                    continue
                best = (dz, dy, dx)
                break
        if best is None:
            continue
        dz, dy, dx = best
        norm = float(np.sqrt(dz * dz + dy * dy + dx * dx)) + 1e-8
        tangent[0, z, y, x] = dz / norm
        tangent[1, z, y, x] = dy / norm
        tangent[2, z, y, x] = dx / norm
    return tangent

def _instance_skeleton_targets(
    inst_mask: np.ndarray,
    *,
    blend_sigma: float = 3.0,
    prune_spur_len: int = 4,
    tangent_speed_clip: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Skeleton, blended flow, distance, and oriented tangent for one instance."""
    skel = skeletonize_instance_3d(inst_mask)
    skel = _prune_short_spurs(skel, min_len=int(prune_spur_len))
    geo = _geodesic_from_endpoints(skel)
    tangent = _oriented_tangent_on_skeleton(skel, geo)
    flow, dist = _blended_skeleton_flow_3d(
        inst_mask,
        skel,
        tangent_skel=tangent,
        geo=geo,
        blend_sigma=float(blend_sigma),
        tangent_speed_clip=float(tangent_speed_clip),
    )
    return skel, flow, dist, tangent

def _blended_skeleton_flow_3d(
    inst_mask: np.ndarray,
    skel: np.ndarray,
    *,
    tangent_skel: np.ndarray | None = None,
    geo: np.ndarray | None = None,
    blend_sigma: float = 3.0,
    tangent_speed_clip: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blended flow (not unit-normalized):
      far from axis  -> unit radial toward nearest skeleton
      near/on axis  -> tangent * remaining geodesic to interior attractor

    Radial magnitude is applied at decode via distance; tangent magnitude is
    stored in the flow vector and decays to zero at the branch center.

    Returns:
        flow (3,Z,Y,X) in (fz, fy, fx) order
        dist (Z,Y,X) EDT distance to skeleton
    """
    shape = inst_mask.shape
    flow = np.zeros((3, *shape), dtype=np.float32)
    dist = np.zeros(shape, dtype=np.float32)
    if not bool(inst_mask.any()) or not bool(skel.any()):
        return flow, dist

    if geo is None:
        geo = _geodesic_from_endpoints(skel)
    if tangent_skel is None:
        tangent_skel = _oriented_tangent_on_skeleton(skel, geo)
    vpara_skel = _tangent_speed_on_skeleton(
        skel, geo, speed_clip=float(tangent_speed_clip)
    )

    inv_skel = ~skel
    edt_dist, inds = ndi.distance_transform_edt(inv_skel, return_indices=True)

    fg = inst_mask.astype(bool)
    zz, yy, xx = np.where(fg)
    dz = inds[0, zz, yy, xx].astype(np.float32) - zz.astype(np.float32)
    dy = inds[1, zz, yy, xx].astype(np.float32) - yy.astype(np.float32)
    dx = inds[2, zz, yy, xx].astype(np.float32) - xx.astype(np.float32)
    rnorm = np.sqrt(dz * dz + dy * dy + dx * dx) + float(eps)

    dist_fg = edt_dist[zz, yy, xx].astype(np.float32)
    rz = dz / rnorm
    ry = dy / rnorm
    rx = dx / rnorm

    sz = inds[0, zz, yy, xx]
    sy = inds[1, zz, yy, xx]
    sx = inds[2, zz, yy, xx]
    tz = tangent_skel[0, sz, sy, sx]
    ty = tangent_skel[1, sz, sy, sx]
    tx = tangent_skel[2, sz, sy, sx]
    vp = vpara_skel[sz, sy, sx].astype(np.float32)

    sigma = max(float(blend_sigma), 1e-6)
    alpha = np.clip(dist_fg / sigma, 0.0, 1.0).astype(np.float32)

    fz = alpha * rz + (1.0 - alpha) * tz * vp
    fy = alpha * ry + (1.0 - alpha) * ty * vp
    fx = alpha * rx + (1.0 - alpha) * tx * vp

    dist[zz, yy, xx] = dist_fg
    flow[0, zz, yy, xx] = fz.astype(np.float32)
    flow[1, zz, yy, xx] = fy.astype(np.float32)
    flow[2, zz, yy, xx] = fx.astype(np.float32)
    return flow, dist

def _skeleton_embedding_3d(
    inst_mask: np.ndarray,
    skel: np.ndarray,
    dist: np.ndarray,
    tangent: np.ndarray,
    *,
    emb_dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    """Skeleton embedding: oriented tangent + normalized radius."""
    emb = np.zeros((emb_dim, *inst_mask.shape), dtype=np.float32)
    if emb_dim <= 0 or not bool(skel.any()):
        return emb

    skel_dist = dist[skel]
    radius_scale = float(np.median(skel_dist[skel_dist > 0])) if np.any(skel_dist > 0) else 1.0
    radius_scale = max(radius_scale, 1.0)

    emb[0][skel] = tangent[0][skel]
    emb[1][skel] = tangent[1][skel]
    if emb_dim >= 3:
        emb[2][skel] = tangent[2][skel]
    if emb_dim >= 4:
        emb[3][skel] = np.clip(dist[skel] / radius_scale, 0.0, 3.0).astype(np.float32)
    return emb

def labels_to_chromopose_target_3d(
    mask_zyx: np.ndarray,
    *,
    fill_holes: bool = True,
    distance_clip: float = 32.0,
    blend_sigma: float = 3.0,
    prune_spur_len: int = 4,
    use_embedding: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build ChromoPose supervision from instance mask.

    Returns:
        mask_clean (Z,Y,X) uint16
        target (C,Z,Y,X) float32
    """
    _require_deps()
    mask_clean = clean_instance_mask_3d(mask_zyx, fill_holes=fill_holes)
    lbl = mask_clean.astype(np.int32)
    shape = lbl.shape
    n_ch = TARGET_CHANNELS if use_embedding else BASE_TARGET_CHANNELS
    target = np.zeros((n_ch, *shape), dtype=np.float32)
    if int(lbl.max()) <= 0:
        return mask_clean, target

    inst_mask = lbl > 0
    boundary = _boundary_from_label(lbl, inst_mask)
    target[CH_FOREGROUND] = inst_mask.astype(np.float32)
    target[CH_BOUNDARY] = boundary

    skel_all = np.zeros(shape, dtype=bool)
    flow_acc = np.zeros((3, *shape), dtype=np.float32)
    dist_acc = np.zeros(shape, dtype=np.float32)
    emb_acc = (
        np.zeros((EMBEDDING_DIM, *shape), dtype=np.float32) if use_embedding else None
    )

    for inst_id, sl in enumerate(ndi.find_objects(lbl) or [], start=1):
        if sl is None:
            continue
        zsl, ysl, xsl = sl
        inst_crop = lbl[zsl, ysl, xsl] == inst_id
        if not int(inst_crop.sum()):
            continue

        skel_crop, flow_crop, dist_crop, tangent_crop = _instance_skeleton_targets(
            inst_crop,
            blend_sigma=float(blend_sigma),
            prune_spur_len=int(prune_spur_len),
            tangent_speed_clip=float(distance_clip),
        )

        z0, y0, x0 = zsl.start, ysl.start, xsl.start
        ze, ye, xe = zsl.stop, ysl.stop, xsl.stop
        flow_acc[:, z0:ze, y0:ye, x0:xe][:, inst_crop] = flow_crop[:, inst_crop]
        dist_acc[z0:ze, y0:ye, x0:xe][inst_crop] = dist_crop[inst_crop]
        skel_all[z0:ze, y0:ye, x0:xe] |= skel_crop

        if use_embedding and emb_acc is not None:
            emb_crop = _skeleton_embedding_3d(inst_crop, skel_crop, dist_crop, tangent_crop)
            for d in range(EMBEDDING_DIM):
                emb_acc[d, z0:ze, y0:ye, x0:xe][skel_crop] = emb_crop[d][skel_crop]

    clip = float(distance_clip)
    if clip > 0:
        dist_acc = np.clip(dist_acc, 0.0, clip)

    target[CH_FLOW_Z] = flow_acc[0]
    target[CH_FLOW_Y] = flow_acc[1]
    target[CH_FLOW_X] = flow_acc[2]
    target[CH_DISTANCE] = dist_acc
    target[CH_SKELETON] = skel_all.astype(np.float32)

    if use_embedding and emb_acc is not None:
        for d in range(EMBEDDING_DIM):
            target[CH_EMB_0 + d] = emb_acc[d]

    return mask_clean, target

if __name__ == "__main__":
    main()
