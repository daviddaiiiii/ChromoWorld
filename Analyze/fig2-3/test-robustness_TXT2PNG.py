#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("opencv-python is required. Install it with: pip install opencv-python") from e

try:
    from tqdm import tqdm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("tqdm is required. Install it with: pip install tqdm") from e


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _resolve_path(value: str) -> Path:
    """Resolve a user-supplied path relative to the Chromo World project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class SegInstance:
    class_id: int
    poly_norm: np.ndarray  # (N,2) float64 in [0,1]


def _read_yolo_seg_label(txt_path: Path) -> List[SegInstance]:
    instances: List[SegInstance] = []
    if not txt_path.exists():
        return instances
    raw = txt_path.read_text(encoding="utf-8").strip()
    if raw == "":
        return instances
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 7 or (len(toks) - 1) % 2 != 0:
            # Skip empty or malformed lines.
            continue
        try:
            class_id = int(float(toks[0]))
            coords = np.asarray([float(x) for x in toks[1:]], dtype=np.float64)
        except Exception:
            continue
        pts = coords.reshape(-1, 2)
        instances.append(SegInstance(class_id=class_id, poly_norm=pts))
    return instances


def _write_yolo_seg_label(txt_path: Path, instances: Sequence[SegInstance]) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for inst in instances:
        pts = np.clip(inst.poly_norm, 0.0, 1.0)
        flat = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
        lines.append(f"{inst.class_id} {flat}".rstrip())
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _poly_norm_to_px(poly_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = poly_norm.astype(np.float64).copy()
    pts[:, 0] *= float(w)
    pts[:, 1] *= float(h)
    return pts


def _poly_px_to_norm(poly_px: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = poly_px.astype(np.float64).copy()
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    return pts


def _apply_affine_to_points(pts_px: np.ndarray, M: np.ndarray) -> np.ndarray:
    # pts_px: (N,2), M: (2,3)
    ones = np.ones((pts_px.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([pts_px.astype(np.float64), ones], axis=1)  # (N,3)
    out = hom @ M.T  # (N,2)
    return out


def _clip_points_in_image(pts_px: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = pts_px.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
    return pts


def _rotate_and_flip(
    img: np.ndarray,
    instances: Sequence[SegInstance],
    rng: random.Random,
    max_angle_deg: float = 45.0,
    min_angle_deg: float = 0.0,
) -> Tuple[np.ndarray, List[SegInstance]]:
    h, w = img.shape[:2]
    lo = float(min_angle_deg)
    hi = float(max_angle_deg)
    if hi < lo:
        lo, hi = hi, lo
    angle = rng.uniform(lo, hi)
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float64)
    out_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    out_instances: List[SegInstance] = []
    for inst in instances:
        pts = _poly_norm_to_px(inst.poly_norm, w=w, h=h)
        pts2 = _apply_affine_to_points(pts, M)
        pts2 = _clip_points_in_image(pts2, w=w, h=h)
        out_instances.append(SegInstance(class_id=inst.class_id, poly_norm=_poly_px_to_norm(pts2, w=w, h=h)))

    # Apply independent random horizontal and vertical flips.
    do_h = rng.random() < 0.5
    do_v = rng.random() < 0.5
    if do_h:
        out_img = cv2.flip(out_img, 1)
        for i, inst in enumerate(out_instances):
            pts = _poly_norm_to_px(inst.poly_norm, w=w, h=h)
            pts[:, 0] = (w - 1) - pts[:, 0]
            out_instances[i] = SegInstance(class_id=inst.class_id, poly_norm=_poly_px_to_norm(pts, w=w, h=h))
    if do_v:
        out_img = cv2.flip(out_img, 0)
        for i, inst in enumerate(out_instances):
            pts = _poly_norm_to_px(inst.poly_norm, w=w, h=h)
            pts[:, 1] = (h - 1) - pts[:, 1]
            out_instances[i] = SegInstance(class_id=inst.class_id, poly_norm=_poly_px_to_norm(pts, w=w, h=h))

    return out_img, out_instances


def _scale_center(
    img: np.ndarray,
    instances: Sequence[SegInstance],
    scale: float,
) -> Tuple[np.ndarray, List[SegInstance]]:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, 0.0, float(scale)).astype(np.float64)
    out_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    out_instances: List[SegInstance] = []
    for inst in instances:
        pts = _poly_norm_to_px(inst.poly_norm, w=w, h=h)
        pts2 = _apply_affine_to_points(pts, M)
        pts2 = _clip_points_in_image(pts2, w=w, h=h)
        out_instances.append(SegInstance(class_id=inst.class_id, poly_norm=_poly_px_to_norm(pts2, w=w, h=h)))
    return out_img, out_instances


def _mask_from_polygon_px(poly_px: np.ndarray, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_px).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _copy_paste_instances(
    img: np.ndarray,
    instances: Sequence[SegInstance],
    rng: random.Random,
    pct: float,
) -> Tuple[np.ndarray, List[SegInstance]]:
    h, w = img.shape[:2]
    base_instances = list(instances)
    n = len(base_instances)
    if n == 0 or pct <= 0:
        return img.copy(), list(base_instances)

    k = int(round(n * pct))
    k = max(0, min(n, k))
    if k == 0:
        return img.copy(), list(base_instances)

    idxs = list(range(n))
    rng.shuffle(idxs)
    chosen = idxs[:k]

    out_img = img.copy()
    out_instances: List[SegInstance] = list(base_instances)

    for idx in chosen:
        inst = base_instances[idx]
        poly_px = _poly_norm_to_px(inst.poly_norm, w=w, h=h)
        poly_px = _clip_points_in_image(poly_px, w=w, h=h)
        mask = _mask_from_polygon_px(poly_px, w=w, h=h)
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        if bw <= 1 or bh <= 1:
            continue

        # Choose a destination that keeps the bounding box inside the image.
        max_x0 = w - bw
        max_y0 = h - bh
        if max_x0 < 0 or max_y0 < 0:
            continue
        new_x0 = rng.randint(0, max_x0) if max_x0 > 0 else 0
        new_y0 = rng.randint(0, max_y0) if max_y0 > 0 else 0

        dx = float(new_x0 - x0)
        dy = float(new_y0 - y0)

        # Crop the instance patch.
        mask_crop = mask[y0 : y1 + 1, x0 : x1 + 1]
        if mask_crop.size == 0:
            continue
        if out_img.ndim == 2:
            img_crop = out_img[y0 : y1 + 1, x0 : x1 + 1].copy()
            patch = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            dst = out_img[new_y0 : new_y0 + bh, new_x0 : new_x0 + bw]
            inv = cv2.bitwise_not(mask_crop)
            dst_bg = cv2.bitwise_and(dst, dst, mask=inv)
            dst_fg = cv2.bitwise_and(patch, patch, mask=mask_crop)
            out_img[new_y0 : new_y0 + bh, new_x0 : new_x0 + bw] = cv2.add(dst_bg, dst_fg)
        else:
            img_crop = out_img[y0 : y1 + 1, x0 : x1 + 1, :].copy()
            patch = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            dst = out_img[new_y0 : new_y0 + bh, new_x0 : new_x0 + bw, :]
            inv = cv2.bitwise_not(mask_crop)
            dst_bg = cv2.bitwise_and(dst, dst, mask=inv)
            dst_fg = cv2.bitwise_and(patch, patch, mask=mask_crop)
            out_img[new_y0 : new_y0 + bh, new_x0 : new_x0 + bw, :] = cv2.add(dst_bg, dst_fg)

        # Add the translated polygon as a new label.
        poly_px_new = poly_px + np.array([dx, dy], dtype=np.float64)
        poly_px_new = _clip_points_in_image(poly_px_new, w=w, h=h)
        out_instances.append(SegInstance(class_id=inst.class_id, poly_norm=_poly_px_to_norm(poly_px_new, w=w, h=h)))

    return out_img, out_instances


def _psf_blur(img: np.ndarray, k: int) -> np.ndarray:
    kk = int(k)
    if kk <= 0:
        return img
    if kk % 2 == 0:
        kk += 1
    return cv2.GaussianBlur(img, (kk, kk), 0)


def _iter_txt_files(root: Path) -> Iterable[Path]:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".txt"):
                yield Path(dp) / fn


def _index_images(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    if not root.exists():
        return idx
    for dp, _, fns in os.walk(root):
        for fn in fns:
            lo = fn.lower()
            if any(lo.endswith(ext) for ext in IMG_EXTS):
                p = Path(dp) / fn
                idx.setdefault(p.stem, p)
    return idx


def _ensure_dir_structure(out_root: Path, tag: str) -> Tuple[Path, Path]:
    img_dir = out_root / tag / "image"
    lab_dir = out_root / tag / "label"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lab_dir


def _save_image(dst_path: Path, img: np.ndarray) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {dst_path}")


def _flatten_name(label_root: Path, txt_path: Path, ext: str) -> str:
    """Flatten the input tree and keep only the label stem in the filename."""
    rel = txt_path.relative_to(label_root)
    return f"{rel.stem}{ext}"


def _split_group(label_root: Path, txt_path: Path) -> str:
    """Assign labels to Banding, FISH, or Other based on the first subdirectory."""
    rel = txt_path.relative_to(label_root)
    if len(rel.parts) >= 2 and rel.parts[0] in ("Banding", "FISH"):
        return rel.parts[0]
    return "Other"


def _dst_paths(out_root: Path, group: str, prefix: str, stem: str, img_ext: str) -> Tuple[Path, Path]:
    img_path = out_root / "image" / f"{prefix}_{stem}{img_ext}"
    lab_path = out_root / "label" / group / f"{prefix}_{stem}.txt"
    return img_path, lab_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create synchronized robustness images and YOLO TXT labels for Chromo World."
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="release/Dataset/2D/test/label",
        help="Input label root; TXT files are scanned recursively",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="release/Dataset/2D/test/image",
        help="Input image root; images are matched by basename",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="Analyze/results/robustness",
        help="Output root for transformed images and labels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for rotation, flipping, and copy-paste",
    )
    args = parser.parse_args()

    label_root = _resolve_path(args.label_path)
    image_root = _resolve_path(args.image_path)
    out_root = _resolve_path(args.out_path)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    # Index the requested image root first; fall back to its parent if needed.
    img_index = _index_images(image_root)
    fallback_index: Dict[str, Path] = {}
    if len(img_index) == 0 and image_root.parent.exists():
        fallback_index = _index_images(image_root.parent)

    # Keep images in one directory and group labels as Banding, FISH, or Other.
    (out_root / "image").mkdir(parents=True, exist_ok=True)
    for grp in ("Banding", "FISH", "Other"):
        (out_root / "label" / grp).mkdir(parents=True, exist_ok=True)

    txt_files = list(_iter_txt_files(label_root))
    if len(txt_files) == 0:
        raise RuntimeError(f"No TXT labels were found under: {label_root}")

    miss = 0
    done = 0
    for txt_path in tqdm(txt_files, desc="robustness", unit="img"):
        stem = txt_path.stem
        group = _split_group(label_root, txt_path)

        img_path = img_index.get(stem)
        if img_path is None:
            img_path = fallback_index.get(stem)
        if img_path is None:
            miss += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            miss += 1
            continue

        instances = _read_yolo_seg_label(txt_path)

        # Preserve the input image extension when possible.
        out_ext = img_path.suffix.lower() if img_path.suffix.lower() in IMG_EXTS else ".png"

        # 1) Random rotation and flipping between 20 and 60 degrees.
        rr_img, rr_inst = _rotate_and_flip(img, instances, rng=rng, min_angle_deg=20.0, max_angle_deg=60.0)
        rr_img_path, rr_lab_path = _dst_paths(out_root, group=group, prefix="RRRF", stem=stem, img_ext=out_ext)
        _save_image(rr_img_path, rr_img)
        _write_yolo_seg_label(rr_lab_path, rr_inst)

        # 2) Centered scaling with fixed scale factors.
        for s, prefix in [(0.50, "RS50"), (0.75, "RS75"), (1.25, "RS125"), (1.50, "RS150")]:
            si_img, si_inst = _scale_center(img, instances, scale=s)
            si_img_path, si_lab_path = _dst_paths(out_root, group=group, prefix=prefix, stem=stem, img_ext=out_ext)
            _save_image(si_img_path, si_img)
            _write_yolo_seg_label(si_lab_path, si_inst)

        # 3) Instance copy-paste augmentation.
        for pct, prefix in [(0.25, "CP25"), (0.50, "CP50"), (0.75, "CP75"), (1.00, "CP100")]:
            cp_img, cp_inst = _copy_paste_instances(img, instances, rng=rng, pct=pct)
            cp_img_path, cp_lab_path = _dst_paths(out_root, group=group, prefix=prefix, stem=stem, img_ext=out_ext)
            _save_image(cp_img_path, cp_img)
            _write_yolo_seg_label(cp_lab_path, cp_inst)

        # 4) PSF blur.
        for k, prefix in [(15, "PSF15"), (30, "PSF30"), (45, "PSF45"), (60, "PSF60")]:
            bf_img = _psf_blur(img, k=k)
            bf_img_path, bf_lab_path = _dst_paths(out_root, group=group, prefix=prefix, stem=stem, img_ext=out_ext)
            _save_image(bf_img_path, bf_img)
            _write_yolo_seg_label(bf_lab_path, instances)

        done += 1

    print(f"Done. processed={done}, total_labels={len(txt_files)}, missing_images={miss}")
    if miss > 0 and len(img_index) == 0:
        print(f"Warning: no images were indexed under image_path; tried parent directory: {image_root.parent}")


if __name__ == "__main__":
    main()
