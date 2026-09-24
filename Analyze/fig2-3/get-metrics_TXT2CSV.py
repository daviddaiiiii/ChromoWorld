#!/usr/bin/env python
"""
Decoupled metric evaluation for Chromo World component predictions.

Inputs
- gt_path: directory containing GT txt files (one per image, same stem).
- baseline_path: dict-like (component name -> directory of TXT predictions).
- class_configurable (optional): JSON dict group -> bool. True = drop preds whose class is absent
  from that image's GT; False = keep all preds (detector-style). Omit the arg for legacy all-True.

Supported txt formats (per line):
- Polygon (seg):  class_id [confidence] x1 y1 x2 y2 ...   (normalized 0-1)
- BBox (det):     class_id [confidence] xc yc w h         (normalized 0-1, YOLO bbox)
                 class_id [confidence] x1 y1 x2 y2        (normalized 0-1, xyxy bbox)

Outputs (under output_path):
- overall_stats.csv: per group global metrics (precision/recall/AP; dice/iou = mean over all GT
  instances, with 0 for empty preds / no bbox match / missing polygon on that GT)
- tag_stats.csv: per group per-tag metrics
- pr_curves/<group>/pr_curve_global.png and pr_curve_tag__<tag>.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-9


def _resolve_path(value):
    """Resolve a user-supplied path relative to the Chromo World project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class Instance:
    class_id: int
    confidence: float  # GT uses 1.0
    polygon: Optional[np.ndarray]  # (N,2) normalized float64, or None
    bbox_xyxy: np.ndarray  # (4,) normalized float64 [x1,y1,x2,y2]


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _parse_line_tokens(tokens: List[str]) -> Optional[Instance]:
    if not tokens:
        return None
    if tokens[0].startswith("#"):
        return None
    class_id = int(float(tokens[0]))

    # Heuristic: if second token exists and >1.0 maybe it's coord; confidence in [0,1] usually.
    # But coords are also [0,1], so we instead use token count parity to distinguish polygon.
    # Candidate formats:
    # - polygon: class conf + 2k coords  (k>=3) OR class + 2k coords
    # - bbox: class conf + 4 coords OR class + 4 coords

    def _try_parse(conf_idx: int) -> Optional[Instance]:
        # tokens[conf_idx] is confidence
        conf = _safe_float(tokens[conf_idx])
        coords = list(map(_safe_float, tokens[conf_idx + 1 :]))
        if any(not np.isfinite(c) for c in coords):
            return None
        if len(coords) == 4:
            bbox = _coords_to_bbox(coords)
            return Instance(class_id=class_id, confidence=float(conf), polygon=None, bbox_xyxy=bbox)
        if len(coords) >= 6 and len(coords) % 2 == 0:
            poly = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
            poly = _clip01(poly)
            bbox = _poly_to_bbox(poly)
            return Instance(class_id=class_id, confidence=float(conf), polygon=poly, bbox_xyxy=bbox)
        return None

    def _try_parse_no_conf(start_idx: int) -> Optional[Instance]:
        conf = 1.0
        coords = list(map(_safe_float, tokens[start_idx:]))
        if any(not np.isfinite(c) for c in coords):
            return None
        if len(coords) == 4:
            bbox = _coords_to_bbox(coords)
            return Instance(class_id=class_id, confidence=conf, polygon=None, bbox_xyxy=bbox)
        if len(coords) >= 6 and len(coords) % 2 == 0:
            poly = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
            poly = _clip01(poly)
            bbox = _poly_to_bbox(poly)
            return Instance(class_id=class_id, confidence=conf, polygon=poly, bbox_xyxy=bbox)
        return None

    if len(tokens) < 2:
        return None

    # Prefer interpreting as: class conf ...
    inst = _try_parse(conf_idx=1)
    if inst is not None:
        return inst
    # Fallback: class ...
    inst = _try_parse_no_conf(start_idx=1)
    return inst


def _coords_to_bbox(coords4: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = coords4
    # detect YOLO xywh vs xyxy
    # If appears to be center format: (xc,yc,w,h) with w,h positive and x2,y2 likely <=1
    # Ambiguous, so: if x2 <= 1 and y2 <= 1 and (x1 + x2 / 2) <= 1.5 etc doesn't help.
    # Use a robust heuristic: if x2 >= x1 and y2 >= y1 and (x2 - x1) <= 1 and (y2 - y1) <= 1,
    # treat as xyxy; else treat as xywh.
    if (x2 >= x1) and (y2 >= y1) and (x2 - x1 <= 1.0 + 1e-6) and (y2 - y1 <= 1.0 + 1e-6):
        bbox = np.array([x1, y1, x2, y2], dtype=np.float64)
    else:
        xc, yc, w, h = x1, y1, x2, y2
        bbox = np.array([xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0], dtype=np.float64)
    return _clip01(bbox)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _poly_to_bbox(poly: np.ndarray) -> np.ndarray:
    x1 = float(np.min(poly[:, 0]))
    y1 = float(np.min(poly[:, 1]))
    x2 = float(np.max(poly[:, 0]))
    y2 = float(np.max(poly[:, 1]))
    return _clip01(np.array([x1, y1, x2, y2], dtype=np.float64))


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xA = max(float(a[0]), float(b[0]))
    yA = max(float(a[1]), float(b[1]))
    xB = min(float(a[2]), float(b[2]))
    yB = min(float(a[3]), float(b[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    areaB = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return float(inter / (areaA + areaB - inter + EPS))


def _polygon_to_mask(poly: np.ndarray, size: int = 1024) -> np.ndarray:
    # poly: normalized (N,2). Use Pillow rasterization to avoid cv2 ABI issues.
    if poly.shape[0] < 3:
        return np.zeros((size, size), dtype=np.uint8)
    pts = np.stack([poly[:, 0] * (size - 1), poly[:, 1] * (size - 1)], axis=1)
    pts = np.round(pts).astype(np.int32)
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(map(int, p)) for p in pts], outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


def _dice_iou_from_polygons(poly_pred: np.ndarray, poly_gt: np.ndarray, size: int = 1024) -> Tuple[float, float]:
    m1 = _polygon_to_mask(poly_pred, size=size)
    m2 = _polygon_to_mask(poly_gt, size=size)
    inter = float(np.logical_and(m1, m2).sum())
    s1 = float(m1.sum())
    s2 = float(m2.sum())
    union = float(np.logical_or(m1, m2).sum())
    dice = float((2.0 * inter + EPS) / (s1 + s2 + EPS))
    iou = float((inter + EPS) / (union + EPS))
    return dice, iou


def _list_txt_by_basename_recursive(root: Path) -> Dict[str, Path]:
    """
    Recursively list all .txt under root, using basename (stem) as the key.

    Example:
      root/sub/a_b_001.txt -> key="a_b_001"

    Note: if different subfolders contain the same basename, later files will overwrite earlier ones.
    We will print a warning with collision count for transparency.
    """
    out: Dict[str, Path] = {}
    collisions = 0
    for p in sorted(root.rglob("*.txt")):
        if not p.is_file():
            continue
        key = p.stem
        if key in out and out[key] != p:
            collisions += 1
        out[key] = p
    if collisions > 0:
        print(f"[WARN] `{root}` has {collisions} basename collisions while indexing txt files.")
    return out


def _load_instances_from_txt(txt_path: Path) -> List[Instance]:
    instances: List[Instance] = []
    if not txt_path.exists():
        return instances
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        inst = _parse_line_tokens(tokens)
        if inst is not None:
            instances.append(inst)
    return instances


def _parse_tags_from_key(key: str) -> List[str]:
    # key is basename (stem)
    parts = key.split("_")
    if len(parts) <= 1:
        return []
    last = parts[-1]
    if last.isdigit():
        parts = parts[:-1]
    # Treat every token as a tag and remove duplicates while preserving order.
    tags = [p for p in parts if p]
    # stable unique
    seen = set()
    uniq: List[str] = []
    for t in tags:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


@dataclass
class PRResult:
    thresholds: np.ndarray  # descending conf thresholds
    precision: np.ndarray
    recall: np.ndarray
    ap: float
    num_gt: int
    num_pred: int
    tp_total: int
    fp_total: int


def _roc_auc_tp_fp(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    ROC-AUC for separating TP (label=1) vs FP (label=0) using confidence scores.
    This is not the classical detection ROC (no TN/FN notion); it measures ranking quality:
      P(score_tp > score_fp) with tie handled as 0.5.
    """
    if scores.size == 0:
        return 0.0
    labels = labels.astype(np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Rank scores with tie handling (average rank for ties), ascending.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    i = 0
    n = sorted_scores.size
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # average rank for [i, j)
        # ranks are 1-based
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j

    # Sum ranks for positives
    rank_sum_pos = float(ranks[sorted_labels == 1].sum())
    # Mann–Whitney U for positives
    u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_pos / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _label_predictions_tp_fp(
    gt_by_image: Dict[str, List[Instance]],
    pred_by_image: Dict[str, List[Instance]],
    class_whitelist: Optional[set[int]] = None,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, labels) for predictions only:
      score = confidence
      label = 1 if prediction is TP under greedy 1-1 matching, else 0 (FP)
    """
    preds: List[Tuple[str, Instance]] = []
    gts: Dict[str, List[Instance]] = {}

    for k, g in gt_by_image.items():
        if class_whitelist is None:
            gts[k] = list(g)
        else:
            gts[k] = [x for x in g if x.class_id in class_whitelist]
    for k, p in pred_by_image.items():
        if class_whitelist is None:
            preds.extend([(k, x) for x in p])
        else:
            preds.extend([(k, x) for x in p if x.class_id in class_whitelist])

    preds.sort(key=lambda kv: float(kv[1].confidence), reverse=True)

    used: Dict[str, np.ndarray] = {k: np.zeros((len(v),), dtype=bool) for k, v in gts.items()}
    labels: List[int] = []
    scores: List[float] = []

    for img_key, pred in preds:
        scores.append(float(pred.confidence))
        gt_list = gts.get(img_key, [])
        if not gt_list:
            labels.append(0)
            continue
        best_j = -1
        best_iou = 0.0
        for j, gt in enumerate(gt_list):
            if used[img_key][j]:
                continue
            if gt.class_id != pred.class_id:
                continue
            iou = _bbox_iou_xyxy(pred.bbox_xyxy, gt.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            used[img_key][best_j] = True
            labels.append(1)
        else:
            labels.append(0)

    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _pr_at_conf(pr: PRResult, conf: float) -> Tuple[float, float]:
    if pr.num_pred == 0 or pr.thresholds.size == 0:
        return 0.0, 0.0
    idx = np.where(pr.thresholds >= conf)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    j = int(idx[-1])
    return float(pr.precision[j]), float(pr.recall[j])


def _f1(p: float, r: float) -> float:
    return float((2.0 * p * r) / (p + r + EPS))


def _compute_pr_curve(
    gt_by_image: Dict[str, List[Instance]],
    pred_by_image: Dict[str, List[Instance]],
    class_whitelist: Optional[set[int]] = None,
    iou_threshold: float = 0.5,
) -> PRResult:
    # Flatten predictions across images, keep image key for matching.
    preds: List[Tuple[str, Instance]] = []
    gts: Dict[str, List[Instance]] = {}

    for k, g in gt_by_image.items():
        if class_whitelist is None:
            gts[k] = list(g)
        else:
            gts[k] = [x for x in g if x.class_id in class_whitelist]
    for k, p in pred_by_image.items():
        if class_whitelist is None:
            preds.extend([(k, x) for x in p])
        else:
            preds.extend([(k, x) for x in p if x.class_id in class_whitelist])

    # Count GT
    num_gt = sum(len(v) for v in gts.values())
    num_pred = len(preds)

    # Sort preds by confidence desc
    preds.sort(key=lambda kv: float(kv[1].confidence), reverse=True)
    thresholds = np.array([float(inst.confidence) for _, inst in preds], dtype=np.float64)

    # For each image, track which GT used
    used: Dict[str, np.ndarray] = {}
    for k, v in gts.items():
        used[k] = np.zeros((len(v),), dtype=bool)

    tp = np.zeros((num_pred,), dtype=np.float64)
    fp = np.zeros((num_pred,), dtype=np.float64)

    for i, (img_key, pred) in enumerate(preds):
        gt_list = gts.get(img_key, [])
        if not gt_list:
            fp[i] = 1.0
            continue

        best_j = -1
        best_iou = 0.0
        for j, gt in enumerate(gt_list):
            if used[img_key][j]:
                continue
            if gt.class_id != pred.class_id:
                continue
            iou = _bbox_iou_xyxy(pred.bbox_xyxy, gt.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            used[img_key][best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum + EPS)
    recall = tp_cum / (num_gt + EPS)

    # AP: numeric integration over recall (monotonic in cum-sorted setting).
    # Use standard step-wise envelope: precision_envelope = max_{r' >= r} precision(r')
    prec_env = np.maximum.accumulate(precision[::-1])[::-1]
    # prepend (0,1) style boundary
    r = np.concatenate([[0.0], recall])
    p = np.concatenate([[prec_env[0] if len(prec_env) else 0.0], prec_env])
    ap = float(np.trapezoid(p, r))

    return PRResult(
        thresholds=thresholds,
        precision=precision,
        recall=recall,
        ap=ap,
        num_gt=int(num_gt),
        num_pred=int(num_pred),
        tp_total=int(tp.sum()),
        fp_total=int(fp.sum()),
    )


def _compute_polygon_quality_on_matches(
    gt_by_image: Dict[str, List[Instance]],
    pred_by_image: Dict[str, List[Instance]],
    class_whitelist: Optional[set[int]] = None,
    iou_threshold: float = 0.5,
    mask_size: int = 1024,
) -> Tuple[Optional[float], Optional[float], int]:
    """
    GT-centric mean dice / IoU: one score per GT instance (after whitelist).

    - Empty predictions or no bbox match above threshold for a GT -> dice=0, iou=0 for that GT.
    - Matched pair but missing polygon on pred or GT -> dice=0, iou=0.
    Images with no GT in gt_list are skipped. Same greedy matching as PR (pred sorted by conf).
    """
    dices: List[float] = []
    ious: List[float] = []
    num_gt_scored = 0
    for img_key, gt_list0 in gt_by_image.items():
        pred_list0 = pred_by_image.get(img_key, [])
        if class_whitelist is None:
            gt_list = gt_list0
            pred_list = pred_list0
        else:
            gt_list = [g for g in gt_list0 if g.class_id in class_whitelist]
            pred_list = [p for p in pred_list0 if p.class_id in class_whitelist]

        if not gt_list:
            continue

        # Greedy matching by bbox IoU (per-class), using confidence ordering
        pred_sorted = sorted(pred_list, key=lambda x: float(x.confidence), reverse=True)
        used_gt = np.zeros((len(gt_list),), dtype=bool)
        matched_pred: List[Optional[Instance]] = [None] * len(gt_list)
        for pred in pred_sorted:
            best_j = -1
            best_iou = 0.0
            for j, gt in enumerate(gt_list):
                if used_gt[j]:
                    continue
                if gt.class_id != pred.class_id:
                    continue
                iou_bb = _bbox_iou_xyxy(pred.bbox_xyxy, gt.bbox_xyxy)
                if iou_bb > best_iou:
                    best_iou = iou_bb
                    best_j = j
            if best_j >= 0 and best_iou >= iou_threshold:
                used_gt[best_j] = True
                matched_pred[best_j] = pred

        for j, gt in enumerate(gt_list):
            pred = matched_pred[j]
            if pred is not None and pred.polygon is not None and gt.polygon is not None:
                d, iou_poly = _dice_iou_from_polygons(pred.polygon, gt.polygon, size=mask_size)
                dices.append(d)
                ious.append(iou_poly)
            else:
                dices.append(0.0)
                ious.append(0.0)
            num_gt_scored += 1

    if num_gt_scored == 0:
        return None, None, 0
    return float(np.mean(dices)), float(np.mean(ious)), int(num_gt_scored)


def _compute_metrics_block(
    gt_by_image: Dict[str, List[Instance]],
    pred_by_image: Dict[str, List[Instance]],
    class_whitelist: Optional[set[int]],
    match_iou_threshold: float,
    conf_thresholds: Sequence[float] = (0.5, 0.75),
    polygon_mask_size: int = 1024,
) -> Dict[str, float]:
    # Precision/Recall/F1 computed at IoU=match_iou_threshold and confidence cutoffs
    pr_main = _compute_pr_curve(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        class_whitelist=class_whitelist,
        iou_threshold=match_iou_threshold,
    )
    p05, r05 = _pr_at_conf(pr_main, float(conf_thresholds[0]))
    p075, r075 = _pr_at_conf(pr_main, float(conf_thresholds[1]))
    f1 = _f1(p05, r05)

    # mAP: use IoU@0.5 only (i.e., AP@0.5)
    # AUC: ROC-AUC for separating TP vs FP using confidence (IoU@0.5 matching)
    pr_50 = _compute_pr_curve(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        class_whitelist=class_whitelist,
        iou_threshold=0.5,
    )
    mAP = float(pr_50.ap)
    scores, labels = _label_predictions_tp_fp(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        class_whitelist=class_whitelist,
        iou_threshold=0.5,
    )
    auc = _roc_auc_tp_fp(scores, labels)

    # Polygon dice/iou: mean over every GT instance (unmatched / empty pred -> 0)
    dice_mean, iou_mean, _num_gt_poly = _compute_polygon_quality_on_matches(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        class_whitelist=class_whitelist,
        iou_threshold=match_iou_threshold,
        mask_size=polygon_mask_size,
    )

    return {
        "num_gt": float(pr_main.num_gt),
        "num_pred": float(pr_main.num_pred),
        "precision@0.5": float(p05),
        "recall@0.5": float(r05),
        "precision@0.75": float(p075),
        "recall@0.75": float(r075),
        "mAP": float(mAP),
        "dice": 0.0 if dice_mean is None else float(dice_mean),
        "iou": 0.0 if iou_mean is None else float(iou_mean),
        "F1": float(f1),
        "AUC": float(auc),
    }


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr: Dict[str, object] = {}
            for k, v in r.items():
                if isinstance(v, float):
                    if not math.isfinite(v):
                        rr[k] = ""
                    else:
                        rr[k] = f"{v:.6f}"
                else:
                    rr[k] = v
            w.writerow(rr)


def evaluate(
    gt_path: Path,
    baselines: Dict[str, Optional[Path]],
    output_path: Path,
    iou_threshold: float = 0.5,
    polygon_mask_size: int = 1024,
    class_configurable: Optional[Dict[str, bool]] = None,
):
    gt_files = _list_txt_by_basename_recursive(gt_path)
    if not gt_files:
        raise FileNotFoundError(f"No GT txt files found under: {gt_path}")

    # Load GT for all images
    gt_by_image: Dict[str, List[Instance]] = {}
    gt_classes: set[int] = set()
    tags_by_image: Dict[str, List[str]] = {}
    dropped_gt_neg1 = 0
    print(f"[GT] Found {len(gt_files)} txt files under `{gt_path}` (recursive, basename-keyed).")
    for key, p in tqdm(gt_files.items(), total=len(gt_files), desc="Loading GT txt"):
        insts_all = _load_instances_from_txt(p)
        insts = []
        for x in insts_all:
            if int(x.class_id) == -1:
                dropped_gt_neg1 += 1
                continue
            insts.append(x)
        gt_by_image[key] = insts
        for x in insts:
            gt_classes.add(int(x.class_id))
        tags_by_image[key] = _parse_tags_from_key(key)
    print(f"[GT] Images: {len(gt_by_image)} | Instances: {sum(len(v) for v in gt_by_image.values())} | "
          f"Classes in GT: {len(gt_classes)}")
    if dropped_gt_neg1 > 0:
        print(f"[GT] Dropped {dropped_gt_neg1} instances with class_id == -1")

    # Global tag set
    all_tags: set[str] = set()
    for tlist in tags_by_image.values():
        all_tags.update(tlist)
    print(f"[GT] Tags discovered: {len(all_tags)}")

    overall_rows: List[Dict[str, object]] = []
    tag_rows: List[Dict[str, object]] = []

    for group, pred_dir in tqdm(list(baselines.items()), desc="Evaluating baselines"):
        if pred_dir is None or str(pred_dir).strip() == "":
            print(f"[{group}] pred_dir is empty (\"\") -> skip, metrics=0.")
            base = {
                "group": group,
                "num_images": len(gt_by_image),
                "num_gt": sum(len(v) for v in gt_by_image.values()),
                "num_pred": 0,
                "precision@0.5": 0.0,
                "recall@0.5": 0.0,
                "precision@0.75": 0.0,
                "recall@0.75": 0.0,
                "mAP": 0.0,
                "dice": 0.0,
                "iou": 0.0,
                "F1": 0.0,
                "AUC": 0.0,
            }
            overall_rows.append(
                dict(base)
            )
            for tag in sorted(all_tags):
                stems = [s for s, tlist in tags_by_image.items() if tag in tlist]
                tag_rows.append(
                    {
                        "group": group,
                        "tag": tag,
                        "num_images": len(stems),
                        "num_gt": sum(len(gt_by_image[s]) for s in stems),
                        "num_pred": 0,
                        "precision@0.5": 0.0,
                        "recall@0.5": 0.0,
                        "precision@0.75": 0.0,
                        "recall@0.75": 0.0,
                        "mAP": 0.0,
                        "dice": 0.0,
                        "iou": 0.0,
                        "F1": 0.0,
                        "AUC": 0.0,
                    }
                )
            continue

        if not pred_dir.exists():
            print(f"[{group}] pred_dir does not exist: `{pred_dir}` -> will output zeros.")
            # still output an empty row for traceability
            overall_rows.append(
                {
                    "group": group,
                    "num_images": len(gt_by_image),
                    "num_gt": sum(len(v) for v in gt_by_image.values()),
                    "num_pred": 0,
                    "precision@0.5": 0.0,
                    "recall@0.5": 0.0,
                    "precision@0.75": 0.0,
                    "recall@0.75": 0.0,
                    "mAP": 0.0,
                    "dice": 0.0,
                    "iou": 0.0,
                    "F1": 0.0,
                    "AUC": 0.0,
                }
            )
            for tag in sorted(all_tags):
                stems = [s for s, tlist in tags_by_image.items() if tag in tlist]
                tag_rows.append(
                    {
                        "group": group,
                        "tag": tag,
                        "num_images": len(stems),
                        "num_gt": sum(len(gt_by_image[s]) for s in stems),
                        "num_pred": 0,
                        "precision@0.5": 0.0,
                        "recall@0.5": 0.0,
                        "precision@0.75": 0.0,
                        "recall@0.75": 0.0,
                        "mAP": 0.0,
                        "dice": 0.0,
                        "iou": 0.0,
                        "F1": 0.0,
                        "AUC": 0.0,
                    }
                )
            continue

        pred_by_image: Dict[str, List[Instance]] = {}
        pred_files = _list_txt_by_basename_recursive(pred_dir)
        missing = 0
        dropped_pred_not_in_image_gt_classes = 0
        # class_configurable is None -> legacy: all groups filter preds to per-image GT classes.
        # If provided: True = ChromoPix-style (drop preds whose class not in that image's GT);
        # False = detector-style (keep all preds; extras count as FP in matching).
        if class_configurable is None:
            filter_to_image_gt_classes = True
        else:
            filter_to_image_gt_classes = bool(class_configurable.get(group, False))
        if not filter_to_image_gt_classes:
            print(f"[{group}] class_configurable=False: keeping all prediction classes (no per-image GT class filter)")

        for key in tqdm(list(gt_by_image.keys()), desc=f"[{group}] Loading predictions", leave=False):
            p = pred_files.get(key)
            if p is None:
                missing += 1
                pred_by_image[key] = []
                continue
            pred_insts = _load_instances_from_txt(p)
            if not filter_to_image_gt_classes:
                pred_by_image[key] = list(pred_insts)
                continue
            # Conditional evaluation: for each image, only keep predicted instances whose class
            # appears in this image's GT (after GT -1 filtering). Other classes are ignored.
            present_gt_classes = {int(g.class_id) for g in gt_by_image.get(key, [])}
            kept = []
            for x in pred_insts:
                if int(x.class_id) in present_gt_classes:
                    kept.append(x)
                else:
                    dropped_pred_not_in_image_gt_classes += 1
            pred_by_image[key] = kept
        print(f"[{group}] pred_dir=`{pred_dir}` | pred_txt_found={len(pred_files)} | "
              f"missing_for_gt={missing}/{len(gt_by_image)}")
        if filter_to_image_gt_classes and dropped_pred_not_in_image_gt_classes > 0:
            print(f"[{group}] Dropped {dropped_pred_not_in_image_gt_classes} predicted instances whose classes "
                  f"are not present in that image's GT")

        # Metrics (per-image GT class filter applied above when filter_to_image_gt_classes)
        class_whitelist = set(gt_classes)
        m = _compute_metrics_block(
            gt_by_image=gt_by_image,
            pred_by_image=pred_by_image,
            class_whitelist=class_whitelist,
            match_iou_threshold=float(iou_threshold),
            conf_thresholds=(0.5, 0.75),
            polygon_mask_size=int(polygon_mask_size),
        )
        overall_rows.append(
            {
                "group": group,
                "num_images": len(gt_by_image),
                "num_gt": int(m["num_gt"]),
                "num_pred": int(m["num_pred"]),
                "precision@0.5": m["precision@0.5"],
                "recall@0.5": m["recall@0.5"],
                "precision@0.75": m["precision@0.75"],
                "recall@0.75": m["recall@0.75"],
                "mAP": m["mAP"],
                "dice": m["dice"],
                "iou": m["iou"],
                "F1": m["F1"],
                "AUC": m["AUC"],
            }
        )

        # Per-tag stats
        for tag in tqdm(sorted(all_tags), desc=f"[{group}] Per-tag PR", leave=False):
            stems = [s for s, tlist in tags_by_image.items() if tag in tlist]
            gt_tag = {s: gt_by_image[s] for s in stems}
            pred_tag = {s: pred_by_image.get(s, []) for s in stems}
            mt = _compute_metrics_block(
                gt_by_image=gt_tag,
                pred_by_image=pred_tag,
                class_whitelist=class_whitelist,
                match_iou_threshold=float(iou_threshold),
                conf_thresholds=(0.5, 0.75),
                polygon_mask_size=int(polygon_mask_size),
            )
            tag_rows.append(
                {
                    "group": group,
                    "tag": tag,
                    "num_images": len(stems),
                    "num_gt": int(mt["num_gt"]),
                    "num_pred": int(mt["num_pred"]),
                    "precision@0.5": mt["precision@0.5"],
                    "recall@0.5": mt["recall@0.5"],
                    "precision@0.75": mt["precision@0.75"],
                    "recall@0.75": mt["recall@0.75"],
                    "mAP": mt["mAP"],
                    "dice": mt["dice"],
                    "iou": mt["iou"],
                    "F1": mt["F1"],
                    "AUC": mt["AUC"],
                }
            )

    _write_csv(
        output_path / "overall_stats.csv",
        fieldnames=[
            "group",
            "num_images",
            "num_gt",
            "num_pred",
            "precision@0.5",
            "recall@0.5",
            "precision@0.75",
            "recall@0.75",
            "mAP",
            "dice",
            "iou",
            "F1",
            "AUC",
        ],
        rows=overall_rows,
    )
    _write_csv(
        output_path / "tag_stats.csv",
        fieldnames=[
            "group",
            "tag",
            "num_images",
            "num_gt",
            "num_pred",
            "precision@0.5",
            "recall@0.5",
            "precision@0.75",
            "recall@0.75",
            "mAP",
            "dice",
            "iou",
            "F1",
            "AUC",
        ],
        rows=tag_rows,
    )


def _parse_baselines(s: str) -> Dict[str, str]:
    """
    Accept:
    - JSON string of dict
    - path to a json file
    """
    p = Path(s)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("baseline_path must be a JSON dict (group -> path)")
    out: Dict[str, str] = {}
    for k, v in data.items():
        out[str(k)] = "" if v is None else str(v)
    return out


def _parse_class_configurable(s: Optional[str]) -> Optional[Dict[str, bool]]:
    """
    None / empty -> None (all baselines use legacy per-image GT class filter).
    Otherwise JSON dict or path to .json: group name -> bool.
    """
    if s is None or str(s).strip() == "":
        return None
    p = Path(s)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("class_configurable must be a JSON object (group -> bool)")
    out: Dict[str, bool] = {}
    for k, v in data.items():
        if isinstance(v, bool):
            out[str(k)] = v
        elif isinstance(v, (int, float)) and v in (0, 1):
            out[str(k)] = bool(int(v))
        elif isinstance(v, str) and v.lower() in ("true", "false", "1", "0", "yes", "no"):
            out[str(k)] = v.lower() in ("true", "1", "yes")
        else:
            out[str(k)] = bool(v)
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare Chromo World components from YOLO-style TXT outputs.")
    parser.add_argument("--gt_path", type=str, default="release/Dataset/2D/test/label", help="Directory containing GT TXT files")
    parser.add_argument(
        "--baseline_path",
        type=str,
        default=json.dumps(
            {
                "ChromoPix": "Analyze/results/predictions/ChromoPix",
                # "ChromoPix_Robustness": "Analyze/results/predictions/ChromoPix_Robustness",
                "MaskRCNN": "Analyze/results/predictions/MaskRCNN",
                "MaskRCNN_Banding": "Analyze/results/predictions/MaskRCNN_Banding",
                "YOLOv11-Seg": "Analyze/results/predictions/YOLOv11-Seg",
                "YOLOv12-Seg": "Analyze/results/predictions/YOLOv12-Seg",
                "YOLOv11-Seg_Banding": "Analyze/results/predictions/YOLOv11-Seg_Banding",
                "YOLOv12-Seg_Banding": "Analyze/results/predictions/YOLOv12-Seg_Banding",
            },
            ensure_ascii=False,
        ),
        help="JSON string or JSON file: component name -> prediction TXT directory",
    )
    parser.add_argument(
        "--class_configurable",
        type=str,
        default=json.dumps(
            {
                "ChromoPix": True,
                # "ChromoPix_Robustness": True,
                "MaskRCNN": False,
                "MaskRCNN_Banding": False,
                "YOLOv11-Seg": False,
                "YOLOv12-Seg": False,
                "YOLOv11-Seg_Banding": False,
                "YOLOv12-Seg_Banding": False,
                "YOLOv11": False,
                "YOLOv12": False,
                "YOLOv26": False,
            },
            ensure_ascii=False,
        ),
        help=(
            "Optional JSON dict (or path to .json): component name -> boolean. "
            "True: drop predictions whose class is absent from that image's GT (ChromoPix-style). "
            "False: keep all predictions (Mask R-CNN / YOLO-style; extra classes become FP). "
            "Groups omitted from the dict default to False when this option is set. "
            "If this option is omitted entirely, all groups use True (legacy behavior)."
        ),
    )
    parser.add_argument("--output_path", type=str, default="Analyze/results/reports/Fig2", help="Output directory")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for bbox matching")
    parser.add_argument(
        "--polygon_mask_size",
        type=int,
        default=1024,
        help="Rasterization size for polygon dice/iou (bigger=more accurate, slower)",
    )
    args = parser.parse_args()

    gt_path = _resolve_path(args.gt_path)
    class_configurable = _parse_class_configurable(args.class_configurable)
    baselines_raw = _parse_baselines(args.baseline_path)
    baselines: Dict[str, Optional[Path]] = {}
    for k, v in baselines_raw.items():
        if not k:
            continue
        if v is None or str(v).strip() == "":
            baselines[str(k)] = None
        else:
            baselines[str(k)] = _resolve_path(str(v))
    output_path = _resolve_path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    evaluate(
        gt_path=gt_path,
        baselines=baselines,
        output_path=output_path,
        iou_threshold=float(args.iou_threshold),
        polygon_mask_size=int(args.polygon_mask_size),
        class_configurable=class_configurable,
    )
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
