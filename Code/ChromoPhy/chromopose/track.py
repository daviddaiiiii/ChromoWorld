#!/usr/bin/env python3
"""Elastic 4-D instance tracking for chromosome masks.

The input is a directory of labelled TZYX TIFFs.  The tracker keeps the
foreground exactly as supplied and only replaces per-frame source labels by
movie-wide track-segment IDs.  A reviewed division frame is a soft anchor:
there is no hard count constraint, and merges, splits, appearances and
disappearances are recorded rather than suppressed.

This is intentionally dependency-light so it can run on the analysis node.
It uses overlap, containment, centroid motion and bounding-box proximity;
the anchor frame is used as the deterministic ID origin for both directions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile


@dataclass
class Obj:
    label: int
    volume: int
    centroid: Tuple[float, float, float]
    lo: Tuple[int, int, int]
    hi: Tuple[int, int, int]


@dataclass
class Edge:
    prev: int
    curr: int
    score: float
    iou: float
    containment: float
    centroid_score: float
    bbox_iou: float
    intersection: int


def _stem(path: Path) -> str:
    name = path.name
    for suffix in ("_shape_postprocessed.tif", "_shape.tif", ".tiff", ".tif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _objects(frame: np.ndarray) -> Dict[int, Obj]:
    frame = np.asarray(frame)
    labels = np.unique(frame)
    labels = labels[labels > 0]
    out: Dict[int, Obj] = {}
    for lab in labels.tolist():
        zyx = np.argwhere(frame == lab)
        if zyx.size == 0:
            continue
        out[int(lab)] = Obj(
            label=int(lab),
            volume=int(zyx.shape[0]),
            centroid=tuple(np.mean(zyx, axis=0).astype(float).tolist()),
            lo=tuple(np.min(zyx, axis=0).astype(int).tolist()),
            hi=tuple(np.max(zyx, axis=0).astype(int).tolist()),
        )
    return out


def _bbox_iou(a: Obj, b: Obj) -> float:
    inter = 1
    union = 1
    for k in range(3):
        left = max(a.lo[k], b.lo[k])
        right = min(a.hi[k], b.hi[k])
        inter *= max(0, right - left + 1)
        union *= max(a.hi[k], b.hi[k]) - min(a.lo[k], b.lo[k]) + 1
    return float(inter / union) if union else 0.0


def _distance(a: Obj, b: Obj) -> float:
    # z is sampled more coarsely than x/y in the canonical 64x256x256 grid.
    dz = (a.centroid[0] - b.centroid[0]) * 2.0
    dy = a.centroid[1] - b.centroid[1]
    dx = a.centroid[2] - b.centroid[2]
    return float(math.sqrt(dz * dz + dy * dy + dx * dx))


def _bbox_gap(a: Obj, b: Obj) -> float:
    gaps = []
    for k in range(3):
        if a.hi[k] < b.lo[k]:
            gaps.append(b.lo[k] - a.hi[k] - 1)
        elif b.hi[k] < a.lo[k]:
            gaps.append(a.lo[k] - b.hi[k] - 1)
        else:
            gaps.append(0)
    return float(math.sqrt((2.0 * gaps[0]) ** 2 + gaps[1] ** 2 + gaps[2] ** 2))


def _pair_edges(prev_frame: np.ndarray, curr_frame: np.ndarray,
                prev: Dict[int, Obj], curr: Dict[int, Obj],
                max_distance: float = 70.0,
                edge_min: float = 0.10) -> List[Edge]:
    """Generate sparse overlap/proximity edges for two neighbouring frames."""
    p = np.asarray(prev_frame)
    c = np.asarray(curr_frame)
    fg = (p > 0) & (c > 0)
    inter_by_pair: Dict[Tuple[int, int], int] = {}
    if np.any(fg):
        pv = p[fg].astype(np.int64, copy=False)
        cv = c[fg].astype(np.int64, copy=False)
        mult = int(max(cv.max(initial=0), 1) + 1)
        keys, counts = np.unique(pv * mult + cv, return_counts=True)
        for key, n in zip(keys.tolist(), counts.tolist()):
            inter_by_pair[(int(key // mult), int(key % mult))] = int(n)

    candidates: Dict[Tuple[int, int], Edge] = {}
    for (pl, cl), inter in inter_by_pair.items():
        a, b = prev.get(pl), curr.get(cl)
        if a is None or b is None:
            continue
        union = a.volume + b.volume - inter
        iou = float(inter / union) if union else 0.0
        contain = float(inter / min(a.volume, b.volume))
        dist = _distance(a, b)
        cscore = math.exp(-dist / 22.0)
        biou = _bbox_iou(a, b)
        score = 0.46 * iou + 0.25 * contain + 0.17 * cscore + 0.12 * biou
        if score >= edge_min or contain >= 0.18 or iou >= 0.025:
            candidates[(pl, cl)] = Edge(pl, cl, score, iou, contain, cscore, biou, inter)

    # Keep nearby no-overlap edges.  These are useful at a thin bridge opening
    # or a one-frame mask dropout, but only when the boxes are also close.
    for pl, a in prev.items():
        for cl, b in curr.items():
            if (pl, cl) in candidates:
                continue
            dist = _distance(a, b)
            if dist > max_distance or _bbox_gap(a, b) > 18.0:
                continue
            cscore = math.exp(-dist / 22.0)
            biou = _bbox_iou(a, b)
            score = 0.17 * cscore + 0.12 * biou
            if score >= 0.095:
                candidates[(pl, cl)] = Edge(pl, cl, score, 0.0, 0.0, cscore, biou, 0)

    # Only retain the strongest few alternatives per object.  This keeps
    # many-to-many topology information without connecting every nearby object.
    by_prev: Dict[int, List[Edge]] = {}
    by_curr: Dict[int, List[Edge]] = {}
    for e in candidates.values():
        by_prev.setdefault(e.prev, []).append(e)
        by_curr.setdefault(e.curr, []).append(e)
    keep = set()
    for group in (by_prev, by_curr):
        for edges in group.values():
            for e in sorted(edges, key=lambda x: x.score, reverse=True)[:4]:
                keep.add((e.prev, e.curr))
    return [candidates[k] for k in keep]


def _anchor_index(name: str, t_count: int, anchors: Dict[str, dict]) -> int:
    cfg = anchors.get(name, {})
    t = cfg.get("representative_t", cfg.get("window_start", 0))
    try:
        return max(0, min(t_count - 1, int(t)))
    except (TypeError, ValueError):
        return 0


def _assign_anchor(objs: Dict[int, Obj]) -> Dict[int, int]:
    ordered = sorted(objs.values(), key=lambda o: (o.centroid[0], o.centroid[1], o.centroid[2], -o.volume))
    return {o.label: i + 1 for i, o in enumerate(ordered)}


def _transition(prev_frame: np.ndarray, curr_frame: np.ndarray,
                prev_objs: Dict[int, Obj], curr_objs: Dict[int, Obj],
                prev_ids: Dict[int, int], next_id: int,
                frame_index: int, direction: str) -> Tuple[Dict[int, int], int, List[Edge]]:
    events: List[dict] = []
    edges = _pair_edges(prev_frame, curr_frame, prev_objs, curr_objs)
    by_prev: Dict[int, List[Edge]] = {}
    by_curr: Dict[int, List[Edge]] = {}
    for e in edges:
        by_prev.setdefault(e.prev, []).append(e)
        by_curr.setdefault(e.curr, []).append(e)
    for v in by_prev.values():
        v.sort(key=lambda e: e.score, reverse=True)
    for v in by_curr.values():
        v.sort(key=lambda e: e.score, reverse=True)

    # Every current object first inherits the strongest plausible parent.
    inherited: Dict[int, Tuple[int, Edge]] = {}
    for cl, es in by_curr.items():
        if es and (es[0].score >= 0.115 or es[0].containment >= 0.13 or es[0].iou >= 0.025):
            inherited[cl] = (es[0].prev, es[0])

    curr_ids: Dict[int, int] = {}
    # When one source object branches, only the strongest branch retains the
    # old segment ID; secondary branches receive new IDs and a split event.
    children_of_prev: Dict[int, List[int]] = {}
    for cl, (pl, _) in inherited.items():
        children_of_prev.setdefault(pl, []).append(cl)
    for pl, children in children_of_prev.items():
        ranked = sorted(children, key=lambda cl: inherited[cl][1].score, reverse=True)
        for rank, cl in enumerate(ranked):
            if pl not in prev_ids:
                continue
            if rank == 0:
                curr_ids[cl] = prev_ids[pl]
            else:
                curr_ids[cl] = next_id
                next_id += 1
                e = inherited[cl][1]
                events.append({"type": "split", "frame": frame_index, "direction": direction,
                               "parent_source": pl, "parent_track": prev_ids[pl],
                               "child_source": cl, "child_track": curr_ids[cl],
                               "score": e.score, "intersection": e.intersection})

    # If a child has multiple plausible parents, preserve the strongest parent
    # and record the other relations as a merge.  This is deliberately elastic.
    for cl, (pl, e0) in inherited.items():
        if cl not in curr_ids or pl not in prev_ids:
            continue
        for e in by_curr.get(cl, [])[1:]:
            if e.score < max(0.105, e0.score * 0.55):
                break
            if e.prev == pl or e.prev not in prev_ids:
                continue
            events.append({"type": "merge", "frame": frame_index, "direction": direction,
                           "child_source": cl, "child_track": curr_ids[cl],
                           "parent_source": e.prev, "parent_track": prev_ids[e.prev],
                           "score": e.score, "intersection": e.intersection})

    for cl, obj in curr_objs.items():
        if cl not in curr_ids:
            curr_ids[cl] = next_id
            next_id += 1
            events.append({"type": "appearance", "frame": frame_index, "direction": direction,
                           "source": cl, "track": curr_ids[cl]})
    return curr_ids, next_id, edges


def _track_movie(arr: np.ndarray, name: str, anchors: Dict[str, dict]) -> Tuple[np.ndarray, dict, List[dict], List[dict]]:
    if arr.ndim != 4:
        raise ValueError(f"Expected TZYX labels, got {arr.shape}")
    t_count = arr.shape[0]
    obj_frames = [_objects(arr[t]) for t in range(t_count)]
    anchor_t = _anchor_index(name, t_count, anchors)
    ids: List[Dict[int, int]] = [{} for _ in range(t_count)]
    next_id = 1
    ids[anchor_t] = _assign_anchor(obj_frames[anchor_t])
    next_id = len(ids[anchor_t]) + 1
    events: List[dict] = []
    edges_all: List[dict] = []

    for t in range(anchor_t + 1, t_count):
        ids[t], next_id, edges = _transition(arr[t - 1], arr[t], obj_frames[t - 1], obj_frames[t],
                                              ids[t - 1], next_id, t, "forward")
        edges_all.extend({**asdict(e), "frame": t, "direction": "forward"} for e in edges)
    for t in range(anchor_t - 1, -1, -1):
        ids[t], next_id, edges = _transition(arr[t + 1], arr[t], obj_frames[t + 1], obj_frames[t],
                                              ids[t + 1], next_id, t, "backward")
        edges_all.extend({**asdict(e), "frame": t, "direction": "backward"} for e in edges)

    # Re-run event collection with the same transition decisions in a compact
    # pass, so the TIFF and tables are generated from exactly one ID map.
    # Topology events are also inferred directly from neighbouring ID maps.
    for t in range(1, t_count):
        prev_sources = {ids[t - 1].get(k): k for k in ids[t - 1]}
        cur_to_parents: Dict[int, set] = {}
        for e in edges_all:
            if e["frame"] != t or e["direction"] != "forward":
                continue
            if e["prev"] in ids[t - 1] and e["curr"] in ids[t]:
                cur_to_parents.setdefault(ids[t][e["curr"]], set()).add(ids[t - 1][e["prev"]])
        for tr, parents in cur_to_parents.items():
            if len(parents) > 1:
                events.append({"type": "merge", "frame": t, "track": tr, "parents": sorted(parents)})
        parent_to_children: Dict[int, set] = {}
        for e in edges_all:
            if e["frame"] != t or e["direction"] != "forward":
                continue
            if e["prev"] in ids[t - 1] and e["curr"] in ids[t]:
                parent_to_children.setdefault(ids[t - 1][e["prev"]], set()).add(ids[t][e["curr"]])
        for tr, children in parent_to_children.items():
            if len(children) > 1:
                events.append({"type": "split", "frame": t, "track": tr, "children": sorted(children)})

    out = np.zeros_like(arr, dtype=np.uint32)
    nodes: List[dict] = []
    for t, frame in enumerate(arr):
        for lab, obj in obj_frames[t].items():
            tr = int(ids[t][lab])
            out[t][frame == lab] = tr
            nodes.append({"movie": name, "t": t, "source_label": lab, "track_id": tr,
                          "volume_voxels": obj.volume, "centroid_z": obj.centroid[0],
                          "centroid_y": obj.centroid[1], "centroid_x": obj.centroid[2],
                          "anchor_frame": t == anchor_t})
    report = {"version": "elastic-anchor-tracking-1.2-v6", "movie": name,
              "shape": list(arr.shape), "anchor_frame": anchor_t,
              "source_instances": int(sum(len(x) for x in obj_frames)),
              "track_segments": int(next_id - 1),
              "events": {k: sum(1 for e in events if e.get("type") == k)
                         for k in ("merge", "split", "appearance", "disappearance")}}
    return out, report, nodes, edges_all + events


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--anchors", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tables").mkdir(exist_ok=True)
    (args.output / "reports").mkdir(exist_ok=True)

    anchors: Dict[str, dict] = {}
    if args.anchors and args.anchors.exists():
        cfg = json.loads(args.anchors.read_text())
        anchors = cfg.get("anchors", cfg if isinstance(cfg, dict) else {})

    files = sorted(args.input.glob("*.tif"))
    if args.limit:
        files = files[: args.limit]
    all_reports = []
    for i, path in enumerate(files, 1):
        name = _stem(path)
        print(f"[{i}/{len(files)}] {name}", flush=True)
        arr = tifffile.imread(path)
        tracked, report, nodes, relations = _track_movie(arr, name, anchors)
        out_path = args.output / f"{name}_tracked.tif"
        with tifffile.TiffFile(path) as source:
            source_meta = dict(source.imagej_metadata or {})
            if not source_meta:
                try:
                    source_meta = json.loads(source.pages[0].description or "{}")
                except (TypeError, ValueError):
                    source_meta = {}
            metadata = {key: source_meta[key] for key in ("unit", "spacing", "finterval") if key in source_meta}
            metadata["axes"] = "TZYX"
            x_resolution = source.pages[0].tags.get("XResolution")
            y_resolution = source.pages[0].tags.get("YResolution")
            resolution = None
            if x_resolution is not None and y_resolution is not None:
                x_num, x_den = x_resolution.value
                y_num, y_den = y_resolution.value
                resolution = (x_num / x_den, y_num / y_den)
        tifffile.imwrite(out_path, tracked, compression="zlib", metadata=metadata,
                         resolution=resolution, photometric="minisblack", bigtiff=True)
        _write_csv(args.output / "tables" / f"{name}_nodes.csv", nodes)
        _write_csv(args.output / "tables" / f"{name}_relations.csv", relations)
        (args.output / "reports" / f"{name}.json").write_text(json.dumps(report, indent=2))
        all_reports.append(report)
    summary = {"version": "elastic-anchor-tracking-1.2-v6", "input": str(args.input),
               "movies": all_reports, "movie_count": len(all_reports),
               "frame_count": int(sum(r["shape"][0] for r in all_reports))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(args.output / "summary.csv", all_reports)
    print(json.dumps({"movie_count": len(all_reports), "frame_count": summary["frame_count"]}, indent=2))


if __name__ == "__main__":
    main()
