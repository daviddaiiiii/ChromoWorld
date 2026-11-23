"""Select clear same-field examples from archived ChromoPix test predictions."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import cv2
import numpy as np

from prepare_atlas_predictions import build as export_predictions, read_predictions
from prepare_class_atlas import CLASSES, MODES, reference_polygons, write_atlas

GRID = 512


@dataclass
class Mask:
    cls: int
    x: int
    y: int
    pixels: np.ndarray
    area: int

    @classmethod
    def from_polygon(cls, class_id, points):
        coords = np.rint(np.array(points).reshape(-1, 2) * (GRID-1)).astype(np.int32)
        corner = coords.min(axis=0)
        coords -= corner
        width, height = coords.max(axis=0) + 1
        pixels = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(pixels, [coords], 1)
        return cls(class_id, int(corner[0]), int(corner[1]), pixels, cv2.countNonZero(pixels))

    @property
    def x1(self):
        return self.x + self.pixels.shape[1]

    @property
    def y1(self):
        return self.y + self.pixels.shape[0]


def overlaps(ref, predictions):
    """Measure mask IoU and the fraction of the reference covered by each call."""
    for pred in predictions:
        x0, x1 = max(ref.x, pred.x), min(ref.x1, pred.x1)
        y0, y1 = max(ref.y, pred.y), min(ref.y1, pred.y1)
        if x0 >= x1 or y0 >= y1:
            continue
        a = ref.pixels[y0-ref.y:y1-ref.y, x0-ref.x:x1-ref.x]
        b = pred.pixels[y0-pred.y:y1-pred.y, x0-pred.x:x1-pred.x]
        intersection = cv2.countNonZero(cv2.bitwise_and(a, b))
        if intersection:
            yield pred.cls, intersection / (ref.area + pred.area - intersection), intersection / ref.area


def quality(ref, predicted):
    hits = list(overlaps(ref, predicted))
    correct = sorted(((iou, covered) for cls, iou, covered in hits if cls == ref.cls), reverse=True)
    best_iou = correct[0][0] if correct else 0
    wrong = max((covered for cls,iou,covered in hits if cls != ref.cls), default=0)
    duplicate = max((covered for iou,covered in correct[1:]), default=0)
    return best_iou, wrong, duplicate


def curate(test, archive, old, amount):
    selected = {}
    comparisons = {}
    for mode in MODES:
        prior = {}
        for cls, name in enumerate(CLASSES):
            for index, row in enumerate(old[mode.lower()][name]):
                prior.setdefault(row["source"], []).append((cls, index, row["polygon"]))
        old_scores = {}
        field_lists = {cls: [] for cls in range(len(CLASSES))}
        for path in sorted((test / "label" / mode).glob("*.txt")):
            source = test / "image" / (path.stem + ".png")
            pred_path = archive / path.name
            if not source.is_file() or not pred_path.is_file():
                continue
            predictions = [Mask.from_polygon(cls,p) for cls,p in read_predictions(pred_path)]
            for cls, index, polygon in prior.get(path.stem, []):
                old_scores[cls,index] = quality(Mask.from_polygon(cls,polygon), predictions)
            refs = [(cls, points, quality(Mask.from_polygon(cls,points), predictions))
                    for cls, area, points in reference_polygons(path)]
            if not refs:
                continue
            # G-banding usually annotates most instances. FISH can be sparse, so
            # unrelated calls cannot be counted as false positives by this audit.
            coverage = sum(iou >= .65 and wrong <= .30 for _, _, (iou,wrong,dup) in refs)/len(refs)
            best = {}
            for cls, points, metrics in refs:
                iou, wrong, duplicate = metrics
                if iou < .76 or wrong > .30 or duplicate > .40:
                    continue
                rank = iou - .23*wrong - .1*duplicate + (.12*coverage if mode == "Banding" else 0)
                match = (float(rank), *metrics)
                if cls not in best or match > best[cls][0]:
                    best[cls] = match, points
            cell = re.match(r"^id[^_]+_(.+?)_FISH_", path.stem)
            for cls, (match, points) in best.items():
                field_lists[cls].append((match, source, points, cell[1] if cell else ""))
        print(f"{mode}: eligible fields per class", {name: len(field_lists[i]) for i,name in enumerate(CLASSES)}, flush=True)
        pool = {}
        stats = {}
        for cls, name in enumerate(CLASSES):
            choices = sorted(field_lists[cls], key=lambda row: (-row[0][0], row[1].stem))
            if not choices:
                raise ValueError(f"{mode} Chr{name} has no acceptable fields.")
            chosen = choices[:amount]
            pool[cls] = [("test",cell,source,points) for _,source,points,cell in chosen]
            stats[name] = {
                "before": [{"source": row["source"], "quality": old_scores[cls,i]} for i,row in enumerate(old[mode.lower()][name])],
                "after": [{"source": source.stem, "quality": quality_tuple[1:]} for quality_tuple,source,_,_ in chosen],
            }
        selected[mode], comparisons[mode] = pool, stats
    return selected, comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("old_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    chosen, report = curate(args.test, args.archive, json.loads(args.old_index.read_text()), 6)
    index = write_atlas(args.output, chosen, min_per_class=1)
    export_predictions(index, args.archive, args.output/"prediction")
    args.report.write_text(json.dumps(report, separators=(",",":")))
