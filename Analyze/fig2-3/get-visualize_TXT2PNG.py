#!/usr/bin/env python3
"""
Read YOLO polygon txt files from label_path, match images in image_path by basename,
and write visualizations to output_path as PNGs named like the txt files. Optional confidence
per polygon is supported; overlapping pixels keep the class with the highest confidence.
Subfigure layout is controlled by --subpic_type: numeric keys "0","1",... each map to a list of
layers among background, text, mask, bbox, contour; panels are concatenated left-to-right.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value):
    """Resolve a user-supplied path relative to the Chromo World project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


COLOR_MAP = {
    0: (255, 178, 178), 1: (255, 51, 51), 2: (153, 0, 0), 3: (255, 204, 153),
    4: (255, 102, 0), 5: (255, 255, 204), 6: (255, 255, 0), 7: (255, 255, 102),
    8: (255, 239, 153), 9: (255, 204, 0), 10: (204, 153, 0), 11: (153, 102, 0),
    12: (204, 255, 204), 13: (51, 255, 51), 14: (0, 102, 0), 15: (153, 255, 255),
    16: (0, 255, 255), 17: (0, 128, 128), 18: (153, 153, 255), 19: (0, 0, 255),
    20: (216, 191, 216), 21: (153, 51, 255), 22: (255, 153, 51), 23: (75, 0, 130),
    24: (220, 220, 220),
}

# Text above bbox top-left: 1–22, X, Y, Unknown
CLASS_TO_LABEL = {i: str(i + 1) for i in range(22)}
CLASS_TO_LABEL[22] = "X"
CLASS_TO_LABEL[23] = "Y"
CLASS_TO_LABEL[24] = "Unknown"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

DEFAULT_SUBPIC_TYPE = {
    "class": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y",
    "0": ["background"],
    "1": ["background", "text", "mask", "bbox", "contour"],
    "2": ["background", "mask", "bbox", "contour"],
}


def load_image(path):
    path = Path(path)
    if not path.is_file():
        return None
    im = cv2.imread(str(path))
    if im is None:
        try:
            with Image.open(path) as img:
                im = np.array(img)
                if im.ndim == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                elif im.shape[-1] == 4:
                    im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
                else:
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return im


def save_png(arr, path):
    if arr is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


def _choose_grid(n: int) -> Tuple[int, int]:
    """
    Choose rows and columns so that rows*cols >= n while staying close to square.
    Panels have equal size, so the layout minimizes both aspect imbalance and waste.
    """
    if n <= 0:
        return 1, 1
    best = None
    best_rc = (1, n)
    for rows in range(1, n + 1):
        cols = int((n + rows - 1) // rows)
        waste = rows * cols - n
        squareness = abs((cols / rows) - 1.0)
        score = (squareness, waste, rows * cols)
        if best is None or score < best:
            best = score
            best_rc = (rows, cols)
    return best_rc


def _tile_panels(panels, rows: int, cols: int) -> np.ndarray:
    if not panels:
        return None
    h, w = panels[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, p in enumerate(panels):
        r = i // cols
        c = i % cols
        if r >= rows:
            break
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = p
    return canvas


def _resize_longest_edge(img: np.ndarray, max_length: int) -> np.ndarray:
    if img is None:
        return img
    ml = int(max_length)
    if ml <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= ml:
        return img
    scale = ml / float(longest)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def invert_colormap_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Invert every channel as 255 - I, normalizing non-uint8 input first.
    """
    if img_bgr is None:
        return img_bgr
    if img_bgr.dtype != np.uint8:
        tmp = img_bgr.astype(np.float32)
        tmp = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX)
        img_u8 = np.clip(tmp, 0, 255).astype(np.uint8)
    else:
        img_u8 = img_bgr
    return 255 - img_u8


def find_image_by_basename(image_dir, stem):
    """Find an image under image_dir as stem + common extensions."""
    image_dir = Path(image_dir)
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _try_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def _parse_one_label_line(parts, width, height):
    """
    Supports:
    - GT: `class x1 y1 x2 y2 ...` (no confidence, at least 3 points).
    - Prediction: `class conf x1 y1 ...` when, after dropping class+conf, the remaining token count
      is even and >= 6 (polygon coordinates).
    Avoids treating the second number on a 7-field GT line as confidence.
    """
    if len(parts) < 7:
        return None
    try:
        class_id = int(parts[0])
    except ValueError:
        return None
    if class_id < 0:
        return None

    n = len(parts)
    # Prediction line: class + conf + even number of coords, total length >= 8
    if n >= 8 and (n - 2) % 2 == 0 and (n - 2) >= 6:
        c2 = _try_float(parts[1])
        if c2 is not None:
            coords = [float(x) for x in parts[2:]]
            if len(coords) % 2 == 0 and len(coords) >= 6:
                polygon = [
                    (coords[i] * width, coords[i + 1] * height)
                    for i in range(0, len(coords), 2)
                ]
                return class_id, c2, polygon

    coords = [float(x) for x in parts[1:]]
    if len(coords) % 2 != 0 or len(coords) < 6:
        return None
    polygon = [
        (coords[i] * width, coords[i + 1] * height)
        for i in range(0, len(coords), 2)
    ]
    return class_id, 1.0, polygon


def parse_yolo_polygon_lines(label_path, width, height):
    """
    Each line: class_id [confidence] x1 y1 x2 y2 ... (normalized coordinates).
    Returns a list of (class_id, confidence, polygon in pixel coordinates).
    """
    label_path = Path(label_path)
    if not label_path.is_file():
        return []
    out = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            parsed = _parse_one_label_line(parts, width, height)
            if parsed is not None:
                out.append(parsed)
    return out


def build_pixel_wise_nms(instances, height, width):
    """
    instances: list of (class_id, conf, polygon).
    For each pixel, among instances covering it, keep the one with highest confidence;
    ties break by larger class_id.
    Returns id_map (H, W) int32 and conf_map (H, W) float32; background is -1 / -1.0.
    """
    h, w = height, width
    conf_map = np.full((h, w), -1.0, dtype=np.float32)
    id_map = np.full((h, w), -1, dtype=np.int32)

    for class_id, conf, polygon in instances:
        pts = np.array(polygon, dtype=np.int32)
        if pts.size < 6:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        m = mask.astype(bool)
        upd = m & (conf > conf_map)
        upd |= m & (conf == conf_map) & (class_id > id_map)
        conf_map[upd] = np.float32(conf)
        id_map[upd] = class_id

    return id_map, conf_map


def smooth_contour(mask_uint8):
    smoothed = cv2.GaussianBlur(mask_uint8, (3, 3), 0.5)
    _, binary = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def draw_class_overlay_bgr(base_bgr, id_map, alpha=0.5):
    """Semi-transparent per-class overlay with Gaussian-softened edges on BGR base."""
    overlay = base_bgr.astype(np.float32)
    h, w = id_map.shape
    for class_id in np.unique(id_map):
        if class_id < 0:
            continue
        rgb = COLOR_MAP.get(int(class_id), (220, 220, 220))
        color_bgr = tuple(int(c) for c in (rgb[2], rgb[1], rgb[0]))
        mask = (id_map == class_id).astype(np.uint8) * 255
        if mask.max() == 0:
            continue
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 1.0)
        soft = np.clip(soft, 0, 1)[:, :, np.newaxis]
        overlay = overlay * (1 - alpha * soft) + np.array(color_bgr, dtype=np.float32) * alpha * soft
    return np.clip(overlay, 0, 255).astype(np.uint8)


def collect_instance_regions(id_map, conf_map):
    """
    For each connected component per class_id, yield dicts with mask, bbox, and max conf in region.
    """
    regions = []
    for class_id in np.unique(id_map):
        if class_id < 0:
            continue
        bin_mask = (id_map == class_id).astype(np.uint8)
        num, labels = cv2.connectedComponents(bin_mask)
        for lid in range(1, num):
            inst = (labels == lid).astype(np.uint8)
            if inst.sum() == 0:
                continue
            conf_val = float(conf_map[inst > 0].max())
            ys, xs = np.where(inst > 0)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            w, h = max(1, x_max - x_min + 1), max(1, y_max - y_min + 1)
            regions.append(
                {
                    "class_id": int(class_id),
                    "mask": inst,
                    "bbox": (x_min, y_min, w, h),
                    "conf": conf_val,
                }
            )
    return regions


def draw_bbox_tint_bgr(img_bgr, x, y, bw, bh, color_bgr):
    """Light tint blend inside the bbox (same spirit as copy_banding)."""
    result = img_bgr
    roi = result[y : y + bh, x : x + bw]
    if roi.size == 0:
        return
    roi_f = roi.astype(np.float32)
    blend = roi_f * (1 - 0.3) + np.array(color_bgr, dtype=np.float32) * 0.3
    result[y : y + bh, x : x + bw] = np.clip(blend, 0, 255).astype(np.uint8)


def draw_textbox_top_left(img_bgr, x, y, text, color_bgr):
    """Above bbox top-left: semi-transparent white background, border, black text (may include confidence)."""
    result = img_bgr
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx0 = max(0, int(x))
    ty_bottom = max(th + 4, int(y) - 4)
    r1 = ty_bottom - th - 2
    r2 = ty_bottom + 2
    c1 = tx0
    c2 = tx0 + tw + 4
    r1, r2 = max(0, r1), min(result.shape[0], r2)
    c1, c2 = max(0, c1), min(result.shape[1], c2)
    if r2 > r1 and c2 > c1:
        patch = result[r1:r2, c1:c2].astype(np.float32)
        result[r1:r2, c1:c2] = np.clip(patch * (1 - 0.7) + 255 * 0.7, 0, 255).astype(np.uint8)
    cv2.rectangle(result, (tx0, ty_bottom - th - 2), (tx0 + tw + 4, ty_bottom + 2), color_bgr, 1)
    cv2.putText(result, text, (tx0 + 2, ty_bottom - 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _parse_class_whitelist(s):
    """
    Parse a case-insensitive list such as "1,2,...,22,X,Y,Unknown".
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None

    label_to_id = {v.lower(): k for k, v in CLASS_TO_LABEL.items()}
    out = set()
    for tok in [t.strip() for t in s.split(",") if t.strip()]:
        tl = tok.lower()
        if tl in label_to_id:
            out.add(int(label_to_id[tl]))
            continue
        try:
            out.add(int(float(tok)))
            continue
        except Exception:
            pass
        if tok.isdigit():
            v = int(tok)
            if 1 <= v <= 22:
                out.add(v - 1)
    return out if out else None


def _normalize_feature_list(v):
    """Normalize panel layers from a list, JSON array string, or CSV string."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip().lower() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip().lower() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [x.strip().lower() for x in s.split(",") if x.strip()]
    raise ValueError(f"Panel layers must be a list, tuple, or string; received {type(v)}")


def _panel_index_from_key(k):
    if isinstance(k, int) and k >= 0:
        return k
    if isinstance(k, str) and k.strip().isdigit():
        return int(k.strip())
    return None


def _parse_subpic_type(s):
    """
    Parse a JSON dictionary or JSON file describing the panel layout.
    - "class": class whitelist string
    - Non-negative integer keys define panel order and layer lists.
    - Layers: background, text, mask, bbox, and contour.
    Panels without background use a black base image.
    """
    if s is None or str(s).strip() == "":
        cfg = dict(DEFAULT_SUBPIC_TYPE)
    else:
        p = _resolve_path(str(s))
        if p.exists() and p.is_file():
            cfg = json.loads(p.read_text(encoding="utf-8"))
        else:
            cfg = json.loads(str(s))
    if not isinstance(cfg, dict):
        raise ValueError("subpic_type must be a JSON dictionary or a JSON file path")

    merged = dict(DEFAULT_SUBPIC_TYPE)
    merged.update(cfg)

    out = {"class_whitelist": _parse_class_whitelist(merged.get("class"))}
    by_idx = {}
    for k, v in merged.items():
        if k in ("class", "class_whitelist"):
            continue
        idx = _panel_index_from_key(k)
        if idx is None:
            continue
        if idx in by_idx:
            raise ValueError(f"Duplicate panel index in subpic_type: {idx}")
        by_idx[idx] = _normalize_feature_list(v)

    if not by_idx:
        raise ValueError('subpic_type needs at least one numeric panel key, for example "0": ["background"]')

    out["panel_layers"] = [by_idx[i] for i in sorted(by_idx.keys())]
    return out


def _render_one_panel(image_bgr, id_map, conf_map, alpha, features):
    """Render one panel from its requested layer list."""
    feats = [str(x).lower() for x in features]
    feat_set = set(feats)

    if "background" in feat_set:
        base = image_bgr.copy()
    else:
        base = np.zeros_like(image_bgr)

    if "mask" in feat_set:
        panel = draw_class_overlay_bgr(base.copy(), id_map, alpha=alpha)
    else:
        panel = base.copy()

    draw_text = "text" in feat_set
    regions = collect_instance_regions(id_map, conf_map)
    for reg in regions:
        cid = reg["class_id"]
        rgb = COLOR_MAP.get(cid, (220, 220, 220))
        color_bgr = tuple(int(c) for c in (rgb[2], rgb[1], rgb[0]))
        inst_mask = (reg["mask"] * 255).astype(np.uint8)
        x, y, bw, bh = reg["bbox"]
        contours = smooth_contour(inst_mask) if ("contour" in feat_set) else []

        if "contour" in feat_set and contours:
            cv2.drawContours(panel, contours, -1, color_bgr, 1)
        if "bbox" in feat_set:
            draw_bbox_tint_bgr(panel, x, y, bw, bh, color_bgr)
            cv2.rectangle(panel, (x, y), (x + bw, y + bh), color_bgr, 1)
        if draw_text:
            label = CLASS_TO_LABEL.get(cid, str(cid))
            text = f"{label} {reg['conf']:.2f}"
            draw_textbox_top_left(panel, x, y, text, color_bgr)

    return panel


def visualize_by_subpic_type(image_bgr, id_map, conf_map, alpha=0.5, subpic_type_cfg=None):
    if subpic_type_cfg is None:
        subpic_type_cfg = _parse_subpic_type(None)

    layers_list = subpic_type_cfg.get("panel_layers", [])
    if not layers_list:
        raise ValueError("panel_layers cannot be empty")

    panels = [_render_one_panel(image_bgr, id_map, conf_map, alpha=alpha, features=feats) for feats in layers_list]
    rows, cols = _choose_grid(len(panels))
    return _tile_panels(panels, rows=rows, cols=cols)


def process_one(txt_path, image_dir, output_dir, alpha, invert_colormap: bool, subpic_type_cfg: dict, max_length: int):
    stem = txt_path.stem
    img_path = find_image_by_basename(image_dir, stem)
    if img_path is None:
        return f"No image for stem: {stem}"
    img = load_image(str(img_path))
    if img is None:
        return f"Failed to read image: {img_path}"
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if invert_colormap:
        img = invert_colormap_bgr(img)
    H, W = img.shape[:2]

    instances = parse_yolo_polygon_lines(txt_path, W, H)
    wl = subpic_type_cfg.get("class_whitelist")
    if wl is not None:
        instances = [x for x in instances if int(x[0]) in wl]
    if not instances:
        id_map = np.full((H, W), -1, dtype=np.int32)
        conf_map = np.full((H, W), -1.0, dtype=np.float32)
        out_img = visualize_by_subpic_type(img, id_map, conf_map, alpha=alpha, subpic_type_cfg=subpic_type_cfg)
        out_img = _resize_longest_edge(out_img, max_length=max_length)
        out_path = Path(output_dir) / f"{stem}.png"
        save_png(out_img, out_path)
        return None

    id_map, conf_map = build_pixel_wise_nms(instances, H, W)
    if (id_map >= 0).sum() == 0:
        id_map = np.full((H, W), -1, dtype=np.int32)
        conf_map = np.full((H, W), -1.0, dtype=np.float32)
        out_img = visualize_by_subpic_type(img, id_map, conf_map, alpha=alpha, subpic_type_cfg=subpic_type_cfg)
        out_img = _resize_longest_edge(out_img, max_length=max_length)
        out_path = Path(output_dir) / f"{stem}.png"
        save_png(out_img, out_path)
        return None

    out_img = visualize_by_subpic_type(img, id_map, conf_map, alpha=alpha, subpic_type_cfg=subpic_type_cfg)
    out_img = _resize_longest_edge(out_img, max_length=max_length)
    out_path = Path(output_dir) / f"{stem}.png"
    save_png(out_img, out_path)
    return None


def main():
    parser = argparse.ArgumentParser(description="YOLO polygon TXT to PNG visualization for Chromo World")
    parser.add_argument(
        "--label_path",
        type=str,
        default="Analyze/results/predictions/ChromoPix",
        help="Directory containing TXT labels",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="release/Dataset/2D/test/image",
        help="Image directory; images are matched by basename",
    )
    parser.add_argument("--output_path", type=str, 
                        default="Analyze/results/visualizations/ChromoPix",
                        help="Output directory for PNGs")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mask overlay alpha")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count()), help="Thread pool size")
    parser.add_argument("--invert_colormap", action="store_true", help="Invert the image before rendering labels")
    parser.add_argument(
        "--subpic_type",
        type=str,
        default=json.dumps(DEFAULT_SUBPIC_TYPE, ensure_ascii=False),
        help='JSON with "class" and numeric panel keys mapped to '
        'background/text/mask/bbox/contour layers',
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum output edge length, including tiled panels; <=0 disables resizing",
    )
    args = parser.parse_args()

    label_dir = _resolve_path(args.label_path)
    image_dir = _resolve_path(args.image_path)
    output_dir = _resolve_path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    subpic_type_cfg = _parse_subpic_type(args.subpic_type)
    max_length = int(args.max_length)

    txt_files = sorted(label_dir.rglob("*.txt"), key=lambda p: str(p.relative_to(label_dir)))
    if not txt_files:
        print(f"No txt files found under: {label_dir}")
        return

    workers = max(1, args.workers)
    print(f"{len(txt_files)} txt file(s), workers={workers}")

    errors = []
    if workers == 1:
        for p in tqdm(txt_files, desc="txt2png"):
            err = process_one(
                p,
                image_dir,
                output_dir,
                args.alpha,
                args.invert_colormap,
                subpic_type_cfg,
                max_length,
            )
            if err:
                errors.append(err)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    process_one,
                    p,
                    image_dir,
                    output_dir,
                    args.alpha,
                    args.invert_colormap,
                    subpic_type_cfg,
                    max_length,
                ): p
                for p in txt_files
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="txt2png"):
                err = fut.result()
                if err:
                    errors.append(err)

    if errors:
        print(f"{len(errors)} warning(s) (showing up to 20):")
        for e in errors[:20]:
            print(f"  {e}")
    print("Done.")


if __name__ == "__main__":
    main()
