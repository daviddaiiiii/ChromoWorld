"""Build 2D atlas previews from release images and their YOLO polygons."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageDraw, features

CLASSES = [str(n) for n in range(1, 23)] + ["X", "Y"]
MODES = ("Banding", "FISH")
SIZE = 640


def polygon_area(points):
    x, y = zip(*points)
    return abs(sum(a*d-b*c for a,b,c,d in zip(x, y, x[1:]+x[:1], y[1:]+y[:1])))/2


def reference_polygons(label):
    for row in label.read_text().splitlines():
        label_id, *values = row.split() if row.strip() else ["-1"]
        cls = int(label_id)
        if not 0 <= cls < len(CLASSES) or len(values) < 6 or len(values) % 2:
            continue
        floats = [float(v) for v in values]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in floats):
            continue
        points = list(zip(floats[::2], floats[1::2]))
        x, y = zip(*points)
        area = polygon_area(points)
        if area < .00012 or max(x)-min(x) < .016 or max(y)-min(y) < .016:
            continue
        # Sparse FISH labels can lie near the crop boundary. Prefer intact views.
        if min(x+y) < .015 or max(x+y) > .985:
            continue
        yield cls, area, points


def annotations(label):
    rows = {}
    for cls, area, points in reference_polygons(label):
        rows.setdefault(cls, []).append((area, points))
    return {cls: max(candidates, key=lambda row: row[0])[1] for cls, candidates in rows.items()}


def display_image(path):
    with Image.open(path) as frame:
        image = frame.convert("L").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    gray = np.asarray(image, dtype=float)
    lo, hi = np.percentile(gray, [55, 99.85])
    return Image.fromarray(np.uint8(255 * np.clip((gray-lo)/max(1, hi-lo), 0, 1)**.85))


def choose(records, amount):
    """Prefer held-out, distinct fields. Spread FISH picks over cell lines."""
    seen = set()
    for _ in range(amount):
        if not records:
            break
        idx = next((i for i, rec in enumerate(records) if rec[1] not in seen), 0)
        rec = records.pop(idx)
        seen.add(rec[1])
        yield rec


def write_atlas(out, selections, min_per_class):
    fields = out / "field"
    tiles = out / "instance"
    fields.mkdir(parents=True, exist_ok=True)
    tiles.mkdir(exist_ok=True)
    result = {}
    images = {}
    ext = "webp" if features.check("webp") else "png"
    for mode, pool in selections.items():
        selected = {}
        for cls, cases in pool.items():
            label = CLASSES[cls]
            selected[label] = []
            for index, (split, line, source, points) in enumerate(cases):
                key = hashlib.blake2s(f"{split}/{source.name}".encode(), digest_size=6).hexdigest()
                if key not in images:
                    img = display_image(source)
                    img.save(fields / f"{key}.{ext}", quality=91)
                    images[key] = img
                img = images[key]
                px = [(x*SIZE, y*SIZE) for x,y in points]
                xs, ys = zip(*px)
                size = max(max(xs)-min(xs), max(ys)-min(ys))*1.26
                cx,cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
                box = tuple(round(v) for v in (cx-size/2,cy-size/2,cx+size/2,cy+size/2))
                mask = Image.new("L", (SIZE, SIZE))
                ImageDraw.Draw(mask).polygon(px, fill=255)
                cutout = Image.composite(img, Image.new("L", img.size), mask)
                tile = f"{mode.lower()}-{label}-{index+1}.{ext}"
                cutout.crop(box).resize((180,180),Image.Resampling.LANCZOS).save(tiles/tile,quality=92)
                selected[label].append({"field":f"field/{key}.{ext}", "instance":f"instance/{tile}",
                                        "polygon":[[round(x,5),round(y,5)] for x,y in points],
                                        "split":split, "cellLine":line, "source":source.stem})
            if len(selected[label]) < min_per_class:
                raise ValueError(f"{mode} {label} has {len(selected[label])} usable fields, expected at least {min_per_class}")
        result[mode.lower()] = selected
    (out/"index.json").write_text(json.dumps(result,separators=(",",":")),encoding="utf-8")
    print(f"Exported {sum(len(rows) for group in result.values() for rows in group.values())} instances from {len(images)} fields")
    return result


def build(root, out, amount):
    selections = {}
    for mode in MODES:
        pool = {i: [] for i in range(len(CLASSES))}
        for split in ("test", "train"):
            base = root / split
            names = sorted((base / "label" / mode).glob("*.txt"))
            random.Random(5606).shuffle(names)
            for label in names:
                image = base / "image" / (label.stem + ".png")
                if not image.is_file():
                    continue
                cell_line = (re.match(r"^id[^_]+_(.+?)_FISH_", label.stem) if mode == "FISH" else None)
                for cls, points in annotations(label).items():
                    if split == "train" and len(pool[cls]) >= amount:
                        continue
                    pool[cls].append((split, cell_line[1] if cell_line else "", image, points))
            if all(len(v) >= amount for v in pool.values()):
                break
        selections[mode] = {cls: list(choose(cases, amount)) for cls, cases in pool.items()}
    write_atlas(out, selections, amount)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="2D dataset directory with train and test")
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, default=6)
    args = parser.parse_args()
    build(args.dataset, args.output, args.per_class)
