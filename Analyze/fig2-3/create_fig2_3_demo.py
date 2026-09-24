#!/usr/bin/env python3
"""Create the two five-sample, three-column Fig2-3 qualitative panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BACKGROUND_ID = -2
UNLABELLED_GT_ID = -1

# RGB colors follow the 24-class Chromo World palette: 1-22, X, Y.
CLASS_COLORS_RGB = {
    0: (255, 178, 178), 1: (255, 51, 51), 2: (153, 0, 0),
    3: (255, 204, 153), 4: (255, 102, 0), 5: (255, 255, 204),
    6: (255, 255, 0), 7: (255, 255, 102), 8: (255, 239, 153),
    9: (255, 204, 0), 10: (204, 153, 0), 11: (153, 102, 0),
    12: (204, 255, 204), 13: (51, 255, 51), 14: (0, 102, 0),
    15: (153, 255, 255), 16: (0, 255, 255), 17: (0, 128, 128),
    18: (153, 153, 255), 19: (0, 0, 255), 20: (216, 191, 216),
    21: (153, 51, 255), 22: (255, 153, 51), 23: (75, 0, 130),
    UNLABELLED_GT_ID: (160, 160, 160),
}

SAMPLES = {
    "Banding": [
        "Banding_04318",
        "Banding_04856",
        "Banding_05241",
        "Banding_05463",
        "Banding_05605",
    ],
    "FISH": [
        "id01_HFF_FISH_00003",
        "id06_MCF-10F_FISH_00000",
        "id12_MCF-10F_FISH_00002",
        "id18_MCF-10F_FISH_00024",
        "idY_MRC5_FISH_00000",
    ],
}


def project_root() -> Path:
    """Find the ChromoWorld root when this script is run from any directory."""
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "release").is_dir():
            return candidate
    return Path.cwd()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root() / path


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        values = image.astype(np.float32)
        low, high = np.percentile(values, [1.0, 99.5])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        if high <= low:
            return np.zeros(image.shape, dtype=np.uint8)
        image = np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return image


def parse_polygons(path: Path, width: int, height: int) -> list[tuple[int, float, np.ndarray]]:
    """Read GT or prediction polygons in normalized YOLO-seg format."""
    instances = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            class_id = int(fields[0])
            confidence = 1.0
            coordinates = [float(value) for value in fields[1:]]
            if len(fields) >= 8 and (len(fields) - 2) % 2 == 0:
                confidence = float(fields[1])
                coordinates = [float(value) for value in fields[2:]]
        except ValueError:
            continue
        if len(coordinates) < 6 or len(coordinates) % 2:
            continue
        points = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
        if not np.isfinite(points).all():
            continue
        points[:, 0] = np.clip(points[:, 0] * width, 0, width - 1)
        points[:, 1] = np.clip(points[:, 1] * height, 0, height - 1)
        instances.append((class_id, confidence, np.rint(points).astype(np.int32)))
    return instances


def rasterize(instances: list[tuple[int, float, np.ndarray]], height: int, width: int) -> np.ndarray:
    """Rasterize instances; preserve unlabelled FISH GT polygons in neutral gray."""
    id_map = np.full((height, width), BACKGROUND_ID, dtype=np.int32)
    confidence_map = np.full((height, width), -np.inf, dtype=np.float32)
    for class_id, confidence, points in instances:
        display_id = UNLABELLED_GT_ID if class_id == UNLABELLED_GT_ID else class_id
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [points], 1)
        covered = polygon_mask.astype(bool)
        update = covered & (
            (confidence > confidence_map)
            | ((confidence == confidence_map) & (display_id > id_map))
        )
        id_map[update] = display_id
        confidence_map[update] = confidence
    return id_map


def overlay_mask(image: np.ndarray, id_map: np.ndarray, alpha: float) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for display_id in np.unique(id_map):
        if display_id == BACKGROUND_ID:
            continue
        mask = id_map == display_id
        rgb = CLASS_COLORS_RGB.get(int(display_id), (210, 210, 210))
        bgr = np.asarray((rgb[2], rgb[1], rgb[0]), dtype=np.float32)
        output[mask] = output[mask] * (1.0 - alpha) + bgr * alpha
    output = np.clip(output, 0, 255).astype(np.uint8)
    for display_id in np.unique(id_map):
        if display_id == BACKGROUND_ID:
            continue
        rgb = CLASS_COLORS_RGB.get(int(display_id), (210, 210, 210))
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        mask = (id_map == display_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(output, contours, -1, bgr, 1, lineType=cv2.LINE_AA)
    return output


def resize_to_tile(image: np.ndarray, edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(edge / width, edge / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    tile = np.zeros((edge, edge, 3), dtype=np.uint8)
    x = (edge - size[0]) // 2
    y = (edge - size[1]) // 2
    tile[y : y + size[1], x : x + size[0]] = resized
    return tile


def text_strip(width: int, height: int, text: str, centered: bool = False) -> np.ndarray:
    strip = np.full((height, width, 3), 248, dtype=np.uint8)
    scale = 0.72
    thickness = 1
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(12, (width - text_width) // 2) if centered else 12
    cv2.putText(strip, text, (x, height - 13), cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 25), thickness, cv2.LINE_AA)
    return strip


def make_panel(
    group: str,
    stems: list[str],
    image_dir: Path,
    gt_root: Path,
    pred_dir: Path,
    tile_edge: int,
    alpha: float,
) -> np.ndarray:
    gap = 10
    label_height = 36
    column_titles = ("Processed input", "Ground truth", "ChromoPix prediction (24 classes)")
    header_height = 44
    panel_width = 3 * tile_edge + 2 * gap
    header = np.full((header_height, panel_width, 3), 248, dtype=np.uint8)
    for column, title in enumerate(column_titles):
        cell = text_strip(tile_edge, header_height, title, centered=True)
        x = column * (tile_edge + gap)
        header[:, x : x + tile_edge] = cell
    rows = [header]
    for stem in stems:
        image_path = image_dir / f"{stem}.png"
        gt_path = gt_root / group / f"{stem}.txt"
        pred_path = pred_dir / f"{stem}.txt"
        missing = [str(path) for path in (image_path, gt_path, pred_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing demo input(s):\n" + "\n".join(missing))

        image = load_image(image_path)
        height, width = image.shape[:2]
        gt_map = rasterize(parse_polygons(gt_path, width, height), height, width)
        pred_map = rasterize(parse_polygons(pred_path, width, height), height, width)
        tiles = [
            resize_to_tile(image, tile_edge),
            resize_to_tile(overlay_mask(image, gt_map, alpha), tile_edge),
            resize_to_tile(overlay_mask(image, pred_map, alpha), tile_edge),
        ]
        row_title = text_strip(panel_width, label_height, stem)
        row = np.full((tile_edge, panel_width, 3), 255, dtype=np.uint8)
        for column, tile in enumerate(tiles):
            x = column * (tile_edge + gap)
            row[:, x : x + tile_edge] = tile
        rows.extend([row_title, row])
        if stem != stems[-1]:
            rows.append(np.full((gap, panel_width, 3), 255, dtype=np.uint8))
    title = text_strip(panel_width, 42, f"{group} | five held-out examples", centered=True)
    return np.vstack([title, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default="release/Dataset/2D/test/image")
    parser.add_argument("--gt-root", default="release/Dataset/2D/test/label")
    parser.add_argument("--pred-dir", required=True, help="Directory of ChromoPix polygon TXT predictions")
    parser.add_argument("--output-dir", default="release/Code/Analyze/fig2-3/demo_outputs")
    parser.add_argument("--tile-edge", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile_edge < 128:
        raise SystemExit("--tile-edge must be at least 128 pixels")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")
    image_dir = resolve_path(args.image_dir)
    gt_root = resolve_path(args.gt_root)
    pred_dir = resolve_path(args.pred_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for group, stems in SAMPLES.items():
        panel = make_panel(group, stems, image_dir, gt_root, pred_dir, args.tile_edge, args.alpha)
        output_path = output_dir / f"{group}_5x3.png"
        if not cv2.imwrite(str(output_path), panel):
            raise OSError(f"Could not write panel image: {output_path}")
        written.append(output_path)
        print(f"Wrote {output_path} ({panel.shape[1]} x {panel.shape[0]})")

    print(f"Created {len(written)} panels from {sum(map(len, SAMPLES.values()))} examples.")


if __name__ == "__main__":
    main()
