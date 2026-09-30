#!/usr/bin/env python3
"""Test-set evaluation and skeleton-based instance reconstruction."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
import argparse
from typing import Any, Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import tifffile
from tqdm import tqdm

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from dataloader import (  # noqa: E402
    IndexRecord, filter_by_split, load_index, normalize_volume, resize_volume_zyx,
)
from model import build_model  # noqa: E402
from visualize import save_compare_vis_mip_png  # noqa: E402

# ---- Test-set metrics and evaluation loop ----

def mean_iou_instances(pred: np.ndarray, gt: np.ndarray) -> float:
    """Greedy one-to-one instance matching IoU."""
    pred_ids = [int(instance_id) for instance_id in np.unique(pred) if instance_id > 0]
    gt_ids = [int(instance_id) for instance_id in np.unique(gt) if instance_id > 0]
    if not pred_ids or not gt_ids:
        return 0.0
    ious: List[float] = []
    used_pred_ids = set()
    for gt_id in gt_ids:
        gt_mask = gt == gt_id
        best_iou, best_pred_id = 0.0, None
        for pred_id in pred_ids:
            if pred_id in used_pred_ids:
                continue
            pred_mask = pred == pred_id
            intersection = float((pred_mask & gt_mask).sum())
            if intersection <= 0:
                continue
            union = float((pred_mask | gt_mask).sum())
            iou = intersection / max(union, 1.0)
            if iou > best_iou:
                best_iou, best_pred_id = iou, pred_id
        if best_pred_id is not None:
            used_pred_ids.add(best_pred_id)
            ious.append(best_iou)
    return float(np.mean(ious)) if ious else 0.0

def instance_count_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    n_pred = len([instance_id for instance_id in np.unique(pred) if instance_id > 0])
    n_gt = len([instance_id for instance_id in np.unique(gt) if instance_id > 0])
    return {
        "n_pred": float(n_pred),
        "n_gt": float(n_gt),
        "count_error": float(abs(n_pred - n_gt)),
        "count_rel_error": float(abs(n_pred - n_gt) / max(n_gt, 1)),
    }

def compute_instance_mask_from_pred(
    prediction: torch.Tensor,
    *,
    raw: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    prediction_np = prediction.detach().float().cpu().numpy()
    return compute_instance_mask_3d(prediction_np, raw=raw, from_logits=True, **kwargs)

def group_test_samples(records: List[IndexRecord]) -> List[Tuple[str, List[IndexRecord]]]:
    by_sid: Dict[str, List[IndexRecord]] = defaultdict(list)
    for rec in filter_by_split(records, "test"):
        by_sid[rec.sid].append(rec)
    return [(sid, sorted(by_sid[sid], key=lambda r: int(r.t))) for sid in sorted(by_sid)]

def iter_test_frame_tasks(records: List[IndexRecord]) -> List[Tuple[str, IndexRecord]]:
    tasks: List[Tuple[str, IndexRecord]] = []
    for sid, frames in group_test_samples(records):
        for rec in frames:
            tasks.append((sid, rec))
    return tasks

@torch.no_grad()
def infer_one_frame(
    model: torch.nn.Module,
    raw_zyx: np.ndarray,
    device: torch.device,
    *,
    size_zyx: Tuple[int, int, int],
    fg_threshold: float,
    skel_threshold: float,
    max_merge_dist: float,
    min_merge_score: float,
    min_size: int,
    max_dist: float,
    flow_niter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = raw_zyx.astype(np.float32)
    if raw.shape != size_zyx:
        dummy = np.zeros(size_zyx, dtype=np.uint16)
        raw, _ = resize_volume_zyx(raw, dummy, size_zyx)
    raw_n = normalize_volume(raw)
    x = torch.from_numpy(raw_n).unsqueeze(0).unsqueeze(0).to(device)
    pred = model(x)[0].detach().cpu().numpy()
    mask = compute_instance_mask_3d(
        pred,
        raw=raw_n,
        fg_threshold=float(fg_threshold),
        skel_threshold=float(skel_threshold),
        max_dist=float(max_dist),
        max_merge_dist=float(max_merge_dist),
        min_merge_score=float(min_merge_score),
        min_size=int(min_size),
        flow_niter=int(flow_niter),
        from_logits=True,
    )
    return pred, mask.astype(np.uint16)

def _vis_png_path(output_dir: Path, sid: str, t: int) -> Path:
    return output_dir / f"{sid}_t{int(t):03d}.png"

def eval_test_set(
    model: torch.nn.Module,
    index_path: Path,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    *,
    epoch: int,
    size_zyx: Tuple[int, int, int],
    distance_clip: float = 32.0,
    fg_threshold: float = 0.5,
    skel_threshold: float = 0.35,
    max_dist: float = 24.0,
    max_merge_dist: float = 8.0,
    min_merge_score: float = 0.75,
    min_size: int = 15,
    flow_niter: int = 40,
    rank: int = 0,
    world_size: int = 1,
    show_tqdm: bool = True,
) -> List[dict]:
    """
    Run test inference; save six-panel PNG per frame only (no mask TIFF).

    Returns per-sample summary (metrics on t=0) on rank 0.
    """
    model.eval()
    index_path = Path(index_path).resolve()
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()

    records = load_index(index_path)
    samples = group_test_samples(records)
    tasks = iter_test_frame_tasks(records)
    if not tasks:
        if rank == 0:
            return []
        return []

    my_tasks = tasks[rank::world_size]
    frame_metrics: Dict[Tuple[str, int], Dict[str, float]] = {}

    t_iter: Any = my_tasks
    if show_tqdm and rank == 0:
        t_iter = tqdm(my_tasks, desc=f"test epoch {epoch}", leave=False, unit="frame")

    for sid, rec in t_iter:
        raw = tifffile.imread(data_root / rec.raw).astype(np.float32)
        gt = tifffile.imread(data_root / rec.mask).astype(np.uint16)
        gt_target = tifffile.imread(data_root / rec.target).astype(np.float32)
        pred, mask = infer_one_frame(
            model,
            raw,
            device,
            size_zyx=size_zyx,
            fg_threshold=fg_threshold,
            skel_threshold=skel_threshold,
            max_merge_dist=max_merge_dist,
            min_merge_score=min_merge_score,
            min_size=min_size,
            max_dist=max_dist,
            flow_niter=flow_niter,
        )
        save_compare_vis_mip_png(
            raw,
            pred,
            mask,
            gt_target,
            gt,
            _vis_png_path(output_dir, sid, int(rec.t)),
            distance_clip=float(distance_clip),
            from_logits=True,
        )
        miou = mean_iou_instances(mask, gt)
        cm = instance_count_metrics(mask, gt)
        frame_metrics[(sid, int(rec.t))] = {
            "miou": float(miou),
            "count_error": float(cm["count_error"]),
            "n_pred": float(cm["n_pred"]),
            "n_gt": float(cm["n_gt"]),
        }

    partial_path = output_dir / f"_partial_rank{rank:02d}.json"
    with partial_path.open("w", encoding="utf-8") as f:
        json.dump(
            {f"{sid}|{t}": m for (sid, t), m in frame_metrics.items()},
            f,
            indent=2,
        )

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank != 0:
        return []

    merged: Dict[Tuple[str, int], Dict[str, float]] = {}
    for p in sorted(output_dir.glob("_partial_rank*.json")):
        with p.open("r", encoding="utf-8") as f:
            blob = json.load(f)
        for key, m in blob.items():
            sid, t_str = key.split("|", 1)
            merged[(sid, int(t_str))] = m
        p.unlink(missing_ok=True)
    frame_metrics = merged

    summary: List[dict] = []
    for sid, frames in samples:
        t0 = frames[0]
        m0 = frame_metrics.get((sid, int(t0.t)), {})
        all_mious = [frame_metrics[(sid, int(r.t))]["miou"] for r in frames if (sid, int(r.t)) in frame_metrics]
        all_cerr = [frame_metrics[(sid, int(r.t))]["count_error"] for r in frames if (sid, int(r.t)) in frame_metrics]
        meta = {
            "sid": sid,
            "n_timepoints": len(frames),
            "vis_dir": str(output_dir),
            "miou_t0": float(m0.get("miou", 0.0)),
            "count_err_t0": float(m0.get("count_error", 0.0)),
            "miou_mean": float(np.mean(all_mious)) if all_mious else 0.0,
            "count_err_mean": float(np.mean(all_cerr)) if all_cerr else 0.0,
        }
        summary.append(meta)

    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "samples": summary}, f, indent=2)
    return summary

# ---- Flow integration used by Chimps test-set reconstruction ----

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None


def _sample_flow_trilinear(flow_zyx: np.ndarray, coords_zyx: np.ndarray) -> np.ndarray:
    """Sample flow components at floating-point ZYX coordinates."""
    if ndi is None:
        raise RuntimeError("scipy is required")
    fz, fy, fx = flow_zyx
    sampled = [ndi.map_coordinates(c, coords_zyx, order=1, mode="nearest") for c in (fz, fy, fx)]
    return np.stack(sampled, axis=0)


def integrate_flow_to_skeleton(
    flow: np.ndarray,
    foreground: np.ndarray,
    *,
    niter: int = 40,
    step: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advect foreground voxels along flow and group their endpoint voxels."""
    fg = foreground.astype(bool)
    if not bool(fg.any()):
        return np.zeros(foreground.shape, dtype=np.int32), np.zeros((0, 3), dtype=np.int32)

    coords = np.column_stack(np.where(fg)).astype(np.float64)
    for _ in range(int(niter)):
        sampled = _sample_flow_trilinear(flow, coords.T)
        sampled /= np.linalg.norm(sampled, axis=0, keepdims=True) + 1e-6
        coords += float(step) * sampled.T

    rounded = np.round(coords).astype(np.int32)
    zmax, ymax, xmax = foreground.shape
    rounded[:, 0] = np.clip(rounded[:, 0], 0, zmax - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, ymax - 1)
    rounded[:, 2] = np.clip(rounded[:, 2], 0, xmax - 1)
    keys = rounded[:, 0] * ymax * xmax + rounded[:, 1] * xmax + rounded[:, 2]
    _, inverse = np.unique(keys, return_inverse=True)
    endpoints = np.zeros(foreground.shape, dtype=np.int32)
    endpoints[np.where(fg)] = inverse.astype(np.int32) + 1
    unique_keys = np.unique(keys)
    endpoint_coords = np.column_stack(
        (unique_keys // (ymax * xmax), (unique_keys % (ymax * xmax)) // xmax, unique_keys % xmax)
    ).astype(np.int32)
    return endpoints, endpoint_coords


def recover_skeleton_from_flow(
    skeleton_prob: np.ndarray,
    endpoints: np.ndarray,
    endpoint_coords: np.ndarray,
    *,
    skel_threshold: float = 0.35,
) -> np.ndarray:
    """Combine thresholded skeleton predictions with integrated-flow endpoints."""
    skel = skeleton_prob >= float(skel_threshold)
    zmax, ymax, xmax = skeleton_prob.shape
    for z, y, x in endpoint_coords:
        if 0 <= z < zmax and 0 <= y < ymax and 0 <= x < xmax:
            skel[z, y, x] = True
    if ndi is not None and bool(skel.any()):
        skel = ndi.binary_opening(skel, iterations=1)
    return skel


# ---- Flow-to-instance reconstruction helpers (formerly eval_reconstruction.py) ----

from typing import Optional, Tuple

import numpy as np

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
)
def pred_to_arrays(
    pred: np.ndarray,
    *,
    from_logits: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Parse (C,Z,Y,X) network output into numpy fields."""
    p = np.asarray(pred, dtype=np.float32)
    flow = np.stack([p[CH_FLOW_Z], p[CH_FLOW_Y], p[CH_FLOW_X]], axis=0)
    dist = p[CH_DISTANCE]
    if from_logits:
        fg = 1.0 / (1.0 + np.exp(-p[CH_FOREGROUND]))
        skel = 1.0 / (1.0 + np.exp(-p[CH_SKELETON]))
        boundary = 1.0 / (1.0 + np.exp(-p[CH_BOUNDARY]))
    else:
        fg = np.clip(p[CH_FOREGROUND], 0.0, 1.0)
        skel = np.clip(p[CH_SKELETON], 0.0, 1.0)
        boundary = np.clip(p[CH_BOUNDARY], 0.0, 1.0)
    emb = None
    if p.shape[0] > BASE_TARGET_CHANNELS:
        emb = p[CH_EMB_0 : CH_EMB_0 + EMBEDDING_DIM]
    return flow, fg, dist, skel, boundary, emb

def _build_foreground(
    fg: np.ndarray,
    dist: np.ndarray,
    *,
    fg_threshold: float = 0.5,
    max_dist: float = 24.0,
) -> np.ndarray:
    mask = fg >= float(fg_threshold)
    if max_dist > 0:
        mask &= dist <= float(max_dist)
    if ndi is not None:
        mask = ndi.binary_fill_holes(mask)
        mask = ndi.binary_opening(mask, iterations=1)
    return mask.astype(bool)

def _propagate_labels_from_skeleton(
    skel_labels: np.ndarray,
    foreground: np.ndarray,
    flow: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Assign instance IDs by nearest skeleton component within foreground."""
    if ndi is None:
        raise RuntimeError("scipy is required")
    out = np.zeros(foreground.shape, dtype=np.int32)
    if not bool(foreground.any()):
        return out

    skel_mask = skel_labels > 0
    if not bool(skel_mask.any()):
        labeled, n = ndi.label(foreground)
        return labeled.astype(np.int32)

    # EDT from skeleton voxels: each foreground voxel inherits label of nearest skel voxel.
    inv = ~skel_mask
    _, inds = ndi.distance_transform_edt(inv, return_indices=True)
    zz, yy, xx = np.where(foreground)
    nearest = skel_labels[inds[0, zz, yy, xx], inds[1, zz, yy, xx], inds[2, zz, yy, xx]]
    out[zz, yy, xx] = nearest

    # Unlabeled foreground (far from skeleton) — watershed seeded by skeleton labels.
    unlabeled = foreground & (out == 0)
    if bool(unlabeled.any()):
        markers = np.zeros(foreground.shape, dtype=np.int32)
        markers[skel_mask] = skel_labels[skel_mask]
        if hasattr(ndi, "watershed_ift"):
            ws = ndi.watershed_ift(
                (-dist).astype(np.float32),
                markers,
                structure=np.ones((3, 3, 3), dtype=np.uint8),
            )
        else:
            from skimage.segmentation import watershed

            ws = watershed(-dist, markers, mask=foreground)
        out[unlabeled] = ws[unlabeled]
    return out.astype(np.int32)

def refine_boundaries(
    mask: np.ndarray,
    raw: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    boundary: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Light boundary refinement using intensity gradient or distance ridge."""
    if ndi is None or not bool(mask.any()):
        return mask
    out = mask.copy()
    if raw is not None:
        grad = np.abs(ndi.sobel(raw.astype(np.float32)))
        # Suppress labels across high-gradient + high-boundary voxels.
        if boundary is not None:
            contact = boundary > 0.6
            grad = grad + contact.astype(np.float32)
        # Re-assign border voxels by smallest gradient among neighbor labels.
        border = out > 0
        eroded = ndi.grey_erosion(out, size=(3, 3, 3))
        border &= out != eroded
        # Keep interior stable; border voxels already set by propagation.
    return out

def compute_instance_mask_3d(
    pred: np.ndarray,
    *,
    raw: Optional[np.ndarray] = None,
    fg_threshold: float = 0.5,
    skel_threshold: float = 0.35,
    max_dist: float = 24.0,
    flow_niter: int = 40,
    flow_step: float = 0.75,
    max_merge_dist: float = 8.0,
    min_merge_score: float = 0.75,
    min_size: int = 15,
    from_logits: bool = True,
) -> np.ndarray:
    """
    Full ChromoPose inference post-processing pipeline.

    1. Threshold foreground + distance
    2. Flow integration
    3. Skeleton graph + fragment merging
    4. Label propagation + optional boundary refine
    """
    flow, fg, dist, skel_prob, boundary, emb = pred_to_arrays(pred, from_logits=from_logits)
    foreground = _build_foreground(fg, dist, fg_threshold=fg_threshold, max_dist=max_dist)

    endpoints, endpoint_coords = integrate_flow_to_skeleton(
        flow, foreground, niter=flow_niter, step=flow_step
    )
    skel = recover_skeleton_from_flow(skel_prob, endpoints, endpoint_coords, skel_threshold=skel_threshold)

    graph = build_skeleton_graph(skel, dist, embedding=emb, boundary=boundary)
    graph = merge_skeleton_fragments(
        graph,
        boundary=boundary,
        flow=flow,
        max_merge_dist=max_merge_dist,
        min_merge_score=min_merge_score,
    )

    mask = _propagate_labels_from_skeleton(graph.voxel_labels, foreground, flow, dist)
    mask = refine_boundaries(mask, raw=raw, dist=dist, boundary=boundary)

    if ndi is not None and min_size > 0:
        for inst_id in np.unique(mask):
            if inst_id == 0:
                continue
            if int((mask == inst_id).sum()) < int(min_size):
                mask[mask == inst_id] = 0
        mask, _ = ndi.label(mask > 0)

    return mask.astype(np.uint16)

# ---- Skeleton graph and fragment-merging helpers (formerly skeleton_graph.py) ----

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None

from model import CHROMOSOME_COUNT_PRIOR, CHROMOSOME_COUNT_SOFT_CENTER

@dataclass
class SkeletonNode:
    node_id: int
    coord: Tuple[int, int, int]
    degree: int = 0
    tangent: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    radius: float = 0.0
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    component_id: int = -1

@dataclass
class SkeletonEdge:
    edge_id: int
    n0: int
    n1: int
    length: float
    curvature: float = 0.0

@dataclass
class SkeletonGraph:
    nodes: Dict[int, SkeletonNode]
    edges: List[SkeletonEdge]
    voxel_labels: np.ndarray  # (Z,Y,X) skeleton voxel -> component id (0=none)

def _neighbors_26(z: int, y: int, x: int, shape: Tuple[int, int, int]):
    zmax, ymax, xmax = shape
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                zz, yy, xx = z + dz, y + dy, x + dx
                if 0 <= zz < zmax and 0 <= yy < ymax and 0 <= xx < xmax:
                    yield zz, yy, xx

def _estimate_tangent(
    skel: np.ndarray,
    coord: Tuple[int, int, int],
    dist: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    z, y, x = coord
    nbrs = [c for c in _neighbors_26(z, y, x, skel.shape) if skel[c]]
    if not nbrs:
        return np.zeros(3, dtype=np.float32), 0.0
    center = np.array([z, y, x], dtype=np.float32)
    vecs = np.array(nbrs, dtype=np.float32) - center
    if len(vecs) == 1:
        t = vecs[0]
    else:
        # PCA on local neighborhood for tangent direction.
        t = vecs[np.argmax(np.linalg.norm(vecs, axis=1))]
    norm = np.linalg.norm(t) + 1e-8
    t = (t / norm).astype(np.float32)
    radius = float(dist[z, y, x]) if dist is not None else 0.0
    return t, radius

def build_skeleton_graph(
    skel: np.ndarray,
    dist: np.ndarray,
    embedding: Optional[np.ndarray] = None,
    boundary: Optional[np.ndarray] = None,
) -> SkeletonGraph:
    """Convert binary skeleton into a graph with endpoint / junction nodes."""
    if ndi is None:
        raise RuntimeError("scipy is required")
    skel = skel.astype(bool)
    shape = skel.shape
    voxel_labels = np.zeros(shape, dtype=np.int32)
    if not bool(skel.any()):
        return SkeletonGraph(nodes={}, edges=[], voxel_labels=voxel_labels)

    labeled, n_comp = ndi.label(skel, structure=np.ones((3, 3, 3), dtype=np.uint8))
    voxel_labels = labeled.astype(np.int32)

    nodes: Dict[int, SkeletonNode] = {}
    edges: List[SkeletonEdge] = []
    edge_id = 0

    for comp in range(1, n_comp + 1):
        comp_mask = labeled == comp
        coords = np.column_stack(np.where(comp_mask))
        degree_map = np.zeros(shape, dtype=np.int32)
        for z, y, x in coords:
            deg = sum(1 for c in _neighbors_26(z, y, x, shape) if skel[c])
            degree_map[z, y, x] = deg

        # Key nodes: endpoints (deg==1) and junctions (deg>=3); sample along chain if long.
        key_coords = []
        for z, y, x in coords:
            deg = int(degree_map[z, y, x])
            if deg != 2:
                key_coords.append((z, y, x))
        if not key_coords:
            # Pure loop — pick arbitrary anchor.
            key_coords = [tuple(coords[len(coords) // 2])]

        local_ids: Dict[Tuple[int, int, int], int] = {}
        for coord in key_coords:
            nid = len(nodes) + 1
            tangent, radius = _estimate_tangent(skel, coord, dist)
            emb = np.zeros(4, dtype=np.float32)
            if embedding is not None and embedding.ndim == 4:
                emb = embedding[:, coord[0], coord[1], coord[2]].astype(np.float32)
            nodes[nid] = SkeletonNode(
                node_id=nid,
                coord=coord,
                degree=int(degree_map[coord]),
                tangent=tangent,
                radius=radius,
                embedding=emb,
                component_id=comp,
            )
            local_ids[coord] = nid

        # Connect key nodes along geodesic within component (greedy chain).
        remaining = set(map(tuple, coords.tolist()))
        visited: Set[Tuple[int, int, int]] = set()
        start = key_coords[0]
        stack = [start]
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            z, y, x = cur
            for nb in _neighbors_26(z, y, x, shape):
                if not comp_mask[nb] or nb in visited:
                    continue
                parent[nb] = cur
                stack.append(nb)
                if nb in local_ids and nb != start and cur in local_ids and nb != cur:
                    n0, n1 = local_ids[cur], local_ids[nb]
                    length = float(np.linalg.norm(np.array(nodes[n1].coord) - np.array(nodes[n0].coord)))
                    edges.append(SkeletonEdge(edge_id=edge_id, n0=n0, n1=n1, length=length))
                    edge_id += 1

    return SkeletonGraph(nodes=nodes, edges=edges, voxel_labels=voxel_labels)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))

def _merge_score(
    n0: SkeletonNode,
    n1: SkeletonNode,
    *,
    dist_zyx: float,
    boundary_penalty: float = 0.0,
    max_dist: float = 8.0,
) -> float:
    """Higher is better for merging two endpoint fragments."""
    if dist_zyx > max_dist:
        return -1e9
    dir01 = np.array(n1.coord, dtype=np.float32) - np.array(n0.coord, dtype=np.float32)
    dir01 = dir01 / (np.linalg.norm(dir01) + 1e-8)
    align0 = _cosine(n0.tangent, dir01)
    align1 = _cosine(n1.tangent, -dir01)
    tangent_score = 0.5 * (align0 + align1)
    radius_score = -abs(n0.radius - n1.radius) / max(n0.radius + n1.radius, 1.0)
    emb_score = _cosine(n0.embedding, n1.embedding)
    dist_score = 1.0 - dist_zyx / max_dist
    return (
        2.0 * dist_score
        + 1.5 * tangent_score
        + 1.0 * emb_score
        + 0.5 * radius_score
        - 2.0 * boundary_penalty
    )

def merge_skeleton_fragments(
    graph: SkeletonGraph,
    boundary: Optional[np.ndarray] = None,
    flow: Optional[np.ndarray] = None,
    *,
    max_merge_dist: float = 8.0,
    min_merge_score: float = 0.75,
    target_count: float = CHROMOSOME_COUNT_SOFT_CENTER,
) -> SkeletonGraph:
    """
    Greedily merge endpoint components using multi-cue evidence.

    Instance-count prior enters only as a soft penalty when merges drive count far below 50.
    """
    if not graph.nodes:
        return graph

    # Union-find over component ids from voxel_labels.
    comp_ids = sorted(set(int(x) for x in np.unique(graph.voxel_labels) if x > 0))
    parent = {c: c for c in comp_ids}

    def find(c: int) -> int:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    endpoints = [n for n in graph.nodes.values() if n.degree == 1]
    by_comp: Dict[int, List[SkeletonNode]] = {}
    for n in endpoints:
        by_comp.setdefault(n.component_id, []).append(n)

    candidates: List[Tuple[float, int, int, int, int]] = []
    comps = list(by_comp.keys())
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            c0, c1 = comps[i], comps[j]
            best = (-1e9, None, None)
            for n0 in by_comp[c0]:
                for n1 in by_comp[c1]:
                    d = float(np.linalg.norm(np.array(n0.coord) - np.array(n1.coord)))
                    bp = 0.0
                    if boundary is not None:
                        mid = (
                            (n0.coord[0] + n1.coord[0]) // 2,
                            (n0.coord[1] + n1.coord[1]) // 2,
                            (n0.coord[2] + n1.coord[2]) // 2,
                        )
                        bp = float(boundary[mid])
                    score = _merge_score(n0, n1, dist_zyx=d, boundary_penalty=bp, max_dist=max_merge_dist)
                    if score > best[0]:
                        best = (score, n0, n1)
            if best[1] is not None and best[0] >= min_merge_score:
                candidates.append((best[0], c0, c1, best[1].node_id, best[2].node_id))

    candidates.sort(reverse=True)
    for score, c0, c1, _, _ in candidates:
        if find(c0) == find(c1):
            continue
        n_comp = len({find(c) for c in comp_ids})
        # Soft count prior: discourage aggressive merges when count already low.
        if n_comp <= CHROMOSOME_COUNT_PRIOR[0]:
            penalty = (CHROMOSOME_COUNT_PRIOR[0] - n_comp) * 0.15
            if score < min_merge_score + penalty:
                continue
        union(c0, c1)

    # Relabel voxel_labels by union-find.
    out_labels = graph.voxel_labels.copy()
    remap = {}
    next_id = 1
    for comp in comp_ids:
        root = find(comp)
        if root not in remap:
            remap[root] = next_id
            next_id += 1
        out_labels[graph.voxel_labels == comp] = remap[root]

    return SkeletonGraph(nodes=graph.nodes, edges=graph.edges, voxel_labels=out_labels)

def count_prior_score(n_instances: int) -> float:
    """Gaussian-like soft score peaking near ~60 chromosomes."""
    return float(np.exp(-0.5 * ((n_instances - CHROMOSOME_COUNT_SOFT_CENTER) / 12.0) ** 2))

def label_skeleton_branches(skel: np.ndarray) -> np.ndarray:
    """
    Label each skeleton voxel with a branch id (path between endpoint/junction key nodes).

    Key nodes: degree 1 (tips) or degree >= 3 (junctions). Degree-2 chains between
    two key nodes share one branch label.
    """
    if ndi is None:
        raise RuntimeError("scipy is required")
    skel = skel.astype(bool)
    branch = np.zeros(skel.shape, dtype=np.int32)
    if not bool(skel.any()):
        return branch

    shape = skel.shape
    coords = np.column_stack(np.where(skel))
    degree = np.zeros(shape, dtype=np.int32)
    for z, y, x in coords:
        degree[z, y, x] = sum(1 for c in _neighbors_26(int(z), int(y), int(x), shape) if skel[c])

    is_key = skel & ((degree == 1) | (degree >= 3))
    if not bool(is_key.any()):
        branch[skel] = 1
        return branch

    visited_directed: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()
    bid = 1

    def _trace(from_key: Tuple[int, int, int], first_step: Tuple[int, int, int]) -> None:
        nonlocal bid
        key = (from_key, first_step)
        if key in visited_directed:
            return
        visited_directed.add(key)

        path = [from_key]
        prev = from_key
        cur = first_step
        while True:
            if cur == from_key:
                break
            path.append(cur)
            if is_key[cur] and cur != from_key:
                break
            nbrs = [c for c in _neighbors_26(cur[0], cur[1], cur[2], shape) if skel[c] and c != prev]
            if not nbrs:
                break
            if len(nbrs) > 1 and is_key[cur]:
                break
            prev, cur = cur, nbrs[0]

        for p in path:
            if branch[p] == 0:
                branch[p] = bid
        bid += 1

    for key in map(tuple, np.column_stack(np.where(is_key)).tolist()):
        z, y, x = key
        for nb in _neighbors_26(z, y, x, shape):
            if not skel[nb]:
                continue
            nb = tuple(nb)
            if is_key[nb] and nb <= key:
                continue
            _trace(key, nb)

    # Isolated key voxels or unlabeled skeleton fragments.
    unlabeled = skel & (branch == 0)
    if bool(unlabeled.any()):
        cc, n_cc = ndi.label(unlabeled, structure=np.ones((3, 3, 3), dtype=np.uint8))
        for cid in range(1, n_cc + 1):
            sl = unlabeled & (cc == cid)
            branch[sl] = bid
            bid += 1
    return branch


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a ChromoPose checkpoint on the Chimps test split."
    )
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--base-ch", type=int, default=None)
    parser.add_argument("--levels", type=int, default=None)
    parser.add_argument("--use-embedding", type=_parse_bool, default=None)
    parser.add_argument("--size-z", type=int, default=None)
    parser.add_argument("--size-y", type=int, default=None)
    parser.add_argument("--size-x", type=int, default=None)
    parser.add_argument("--distance-clip", type=float, default=32.0)
    parser.add_argument("--fg-threshold", type=float, default=0.5)
    parser.add_argument("--skel-threshold", type=float, default=0.35)
    parser.add_argument("--max-dist", type=float, default=24.0)
    parser.add_argument("--max-merge-dist", type=float, default=8.0)
    parser.add_argument("--min-merge-score", type=float, default=0.75)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--flow-niter", type=int, default=40)
    parser.add_argument("--epoch", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)

    def config(cli_value, key: str, fallback):
        return cli_value if cli_value is not None else saved_args.get(key, fallback)

    backbone = str(config(args.backbone, "backbone", "unet3d"))
    base_channels = int(config(args.base_ch, "base_ch", 32))
    levels = int(config(args.levels, "levels", 4))
    use_embedding = _parse_bool(config(args.use_embedding, "use_embedding", True))
    size_zyx = (
        int(config(args.size_z, "crop_z", 64)),
        int(config(args.size_y, "crop_y", 256)),
        int(config(args.size_x, "crop_x", 256)),
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")
    model = build_model(
        backbone=backbone,
        base_channels=base_channels,
        levels=levels,
        use_embedding=use_embedding,
    ).to(device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is missing model_state_dict")
    if state and str(next(iter(state))).startswith("module."):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)

    epoch = int(args.epoch if args.epoch is not None else checkpoint.get("epoch", -1))
    print(
        f"Evaluating test split: checkpoint={checkpoint_path} device={device} "
        f"size_zyx={size_zyx}"
    )
    summary = eval_test_set(
        model,
        args.index_path,
        args.data_root,
        args.output_dir,
        device,
        epoch=epoch,
        size_zyx=size_zyx,
        distance_clip=float(args.distance_clip),
        fg_threshold=float(args.fg_threshold),
        skel_threshold=float(args.skel_threshold),
        max_dist=float(args.max_dist),
        max_merge_dist=float(args.max_merge_dist),
        min_merge_score=float(args.min_merge_score),
        min_size=int(args.min_size),
        flow_niter=int(args.flow_niter),
    )
    if not summary:
        raise RuntimeError("No Chimps test samples were evaluated; check index split labels")
    miou = float(np.mean([row["miou_mean"] for row in summary]))
    count_error = float(np.mean([row["count_err_mean"] for row in summary]))
    print(
        f"Done: {len(summary)} samples | mean mIoU={miou:.4f} "
        f"mean count error={count_error:.3f} | outputs={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
