#!/usr/bin/env python3
"""Create a semi-transparent polygon-mask visualization for Fig2-3.

The script pairs processed images with YOLO-style polygon TXT files.  It writes
one panel per image: the processed image on the left and the same image with a
semi-transparent, class-coloured mask on the right.  Ground-truth and
prediction TXT files use the same parser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

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
}
CLASS_NAMES = {i: str(i + 1) for i in range(22)}
CLASS_NAMES.update({22: "X", 23: "Y"})


def project_root() -> Path:
    """Find the repository root when run in-place or as a copied script."""
    resolved = Path(__file__).resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "release").is_dir():
            return candidate
    return Path.cwd()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve relative paths from the ChromoWorld project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root() / path


def load_image(path: Path) -> np.ndarray:
    """Load an image and convert it to displayable uint8 BGR."""
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


def find_images(image_dir: Path) -> dict[str, Path]:
    """Index images by basename for pairing with TXT files."""
    images: dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.setdefault(path.stem, path)
    return images


def parse_polygon_file(path: Path, width: int, height: int) -> list[tuple[int, float, np.ndarray]]:
    """Parse class [confidence] x1 y1 ... polygon lines with normalized coordinates."""
    instances: list[tuple[int, float, np.ndarray]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) < 7:
                continue
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError:
                continue
            confidence = 1.0
            # Prediction lines have class + confidence + an even number of coordinates.
            if len(fields) >= 8 and (len(fields) - 2) % 2 == 0:
                try:
                    confidence = float(fields[1])
                    values = [float(value) for value in fields[2:]]
                except ValueError:
                    continue
            if len(values) < 6 or len(values) % 2:
                continue
            points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= float(width)
            points[:, 1] *= float(height)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            if not np.isfinite(points).all():
                continue
            instances.append((class_id, confidence, np.rint(points).astype(np.int32)))
    return instances


def rasterize_instances(
    instances: Iterable[tuple[int, float, np.ndarray]], height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize polygons with confidence-priority overlap resolution."""
    id_map = np.full((height, width), -1, dtype=np.int32)
    confidence_map = np.full((height, width), -np.inf, dtype=np.float32)
    for class_id, confidence, points in instances:
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [points], 1)
        covered = polygon_mask.astype(bool)
        update = covered & (
            (confidence > confidence_map)
            | ((confidence == confidence_map) & (class_id > id_map))
        )
        id_map[update] = class_id
        confidence_map[update] = confidence
    return id_map, confidence_map


def overlay_mask(image_bgr: np.ndarray, id_map: np.ndarray, alpha: float) -> np.ndarray:
    """Blend class colours into the image and draw thin instance contours."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output = image_bgr.copy().astype(np.float32)
    for class_id in np.unique(id_map):
        if class_id < 0:
            continue
        mask = id_map == int(class_id)
        rgb = CLASS_COLORS_RGB.get(int(class_id), (220, 220, 220))
        bgr = np.asarray((rgb[2], rgb[1], rgb[0]), dtype=np.float32)
        output[mask] = output[mask] * (1.0 - alpha) + bgr * alpha
    output = np.clip(output, 0, 255).astype(np.uint8)
    for class_id in np.unique(id_map):
        if class_id < 0:
            continue
        rgb = CLASS_COLORS_RGB.get(int(class_id), (220, 220, 220))
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        mask = (id_map == int(class_id)).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(output, contours, -1, bgr, 1, lineType=cv2.LINE_AA)
    return output


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    """Add a compact title strip above a panel."""
    strip = np.full((34, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(strip, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([strip, image])


def resize_longest_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0 or max(image.shape[:2]) <= max_edge:
        return image
    scale = float(max_edge) / max(image.shape[:2])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def make_contact_sheet(panels: list[np.ndarray], columns: int, tile_edge: int) -> np.ndarray | None:
    """Create a small overview sheet without changing the full-size outputs."""
    if not panels:
        return None
    tiles = [resize_longest_edge(panel, tile_edge) for panel in panels]
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    padded_tiles = []
    for tile in tiles:
        canvas = np.full((tile_height, tile_width, 3), 255, dtype=np.uint8)
        y = (tile_height - tile.shape[0]) // 2
        x = (tile_width - tile.shape[1]) // 2
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
        padded_tiles.append(canvas)
    tiles = padded_tiles
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = tile
    return sheet


def make_class_legend(cell_width: int = 120, cell_height: int = 42, columns: int = 6) -> np.ndarray:
    """Render the complete 24-class palette used by the mask overlays."""
    rows = (len(CLASS_NAMES) + columns - 1) // columns
    legend = np.full((rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)
    for class_id, class_name in CLASS_NAMES.items():
        row, column = divmod(class_id, columns)
        x0, y0 = column * cell_width, row * cell_height
        rgb = CLASS_COLORS_RGB.get(class_id, (220, 220, 220))
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(legend, (x0 + 8, y0 + 9), (x0 + 34, y0 + 34), bgr, -1)
        cv2.rectangle(legend, (x0 + 8, y0 + 9), (x0 + 34, y0 + 34), (80, 80, 80), 1)
        cv2.putText(legend, class_name, (x0 + 43, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    return legend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default="release/Dataset/2D/test/image")
    parser.add_argument("--label-dir", default="release/Dataset/2D/test/label/FISH")
    parser.add_argument("--output-dir", default="release/Code/Analyze/fig2-3/demo_outputs/mask_overlay")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask opacity in [0, 1].")
    parser.add_argument("--max-edge", type=int, default=2048, help="Maximum edge of each saved panel.")
    parser.add_argument("--max-images", type=int, default=0, help="Process only the first N labels; 0 means all.")
    parser.add_argument("--stems", default=None, help="Comma-separated image basenames to process in the given order.")
    parser.add_argument("--contact-sheet-columns", type=int, default=3)
    parser.add_argument("--contact-sheet-tile-edge", type=int, default=700)
    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = resolve_project_path(args.image_dir)
    label_dir = resolve_project_path(args.label_dir)
    output_dir = resolve_project_path(args.output_dir)
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    images = find_images(image_dir)
    label_files = sorted(label_dir.rglob("*.txt"))
    if args.stems:
        by_stem = {path.stem: path for path in label_files}
        requested = [stem.strip() for stem in args.stems.split(",") if stem.strip()]
        missing = [stem for stem in requested if stem not in by_stem]
        if missing:
            raise SystemExit(f"Requested stems were not found under {label_dir}: {missing}")
        label_files = [by_stem[stem] for stem in requested]
    elif args.max_images > 0:
        label_files = label_files[: args.max_images]
    if not label_files:
        raise SystemExit(f"No TXT labels found under {label_dir}")

    records = []
    panels = []
    for label_path in label_files:
        image_path = images.get(label_path.stem)
        if image_path is None:
            records.append({"label_file": str(label_path), "status": "missing_image"})
            continue
        try:
            image = load_image(image_path)
            height, width = image.shape[:2]
            instances = parse_polygon_file(label_path, width, height)
            id_map, _ = rasterize_instances(instances, height, width)
            overlay = overlay_mask(image, id_map, args.alpha)
            panel = cv2.hconcat([add_title(image, "processed image"), add_title(overlay, "semi-transparent mask")])
            panel = resize_longest_edge(panel, args.max_edge)
            output_path = panel_dir / f"{label_path.stem}_mask_overlay.png"
            cv2.imwrite(str(output_path), panel)
            panels.append(panel)
            records.append(
                {
                    "image_basename": label_path.stem,
                    "image_file": str(image_path),
                    "label_file": str(label_path),
                    "output_file": str(output_path),
                    "n_polygons": len(instances),
                    "n_mask_pixels": int((id_map >= 0).sum()),
                    "status": "ok",
                }
            )
        except Exception as error:
            records.append({"label_file": str(label_path), "status": "error", "error": str(error)})

    if panels and not args.no_contact_sheet:
        sheet = make_contact_sheet(panels, max(1, args.contact_sheet_columns), args.contact_sheet_tile_edge)
        if sheet is not None:
            cv2.imwrite(str(output_dir / "contact_sheet.png"), sheet)
    cv2.imwrite(str(output_dir / "class_legend_24colours.png"), make_class_legend())
    manifest = {
        "schema_version": "chromoworld.fig2_3.mask_overlay.v1",
        "project": "Chromo World",
        "component": "ChromoPix or comparison component TXT labels",
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "alpha": float(args.alpha),
        "records": records,
    }
    (output_dir / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"n_labels": len(label_files), "n_written": len(panels), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
