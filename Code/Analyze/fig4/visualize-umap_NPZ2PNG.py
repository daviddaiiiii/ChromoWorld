#!/usr/bin/env python3
"""Render any saved Fig4/Fig5 ChromoWorld UMAP embedding from an NPZ.

This is a plotting-only script: it uses saved two-dimensional coordinates and
does not fit or refit UMAP. This shared entry point lives in Analyze/fig4 and
accepts named per-space NPZ assets from both Artifacts/fig4 and Artifacts/fig5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from release_paths import find_release_root, resolve_release_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# RGB colors are shared with the Fig2-3 24-class mask visualizations.
CLASS_COLORS_RGB = np.asarray(
    [
        (255, 178, 178), (255, 51, 51), (153, 0, 0), (255, 204, 153),
        (255, 102, 0), (255, 255, 204), (255, 255, 0), (255, 255, 102),
        (255, 239, 153), (255, 204, 0), (204, 153, 0), (153, 102, 0),
        (204, 255, 204), (51, 255, 51), (0, 102, 0), (153, 255, 255),
        (0, 255, 255), (0, 128, 128), (153, 153, 255), (0, 0, 255),
        (216, 191, 216), (153, 51, 255), (255, 153, 51), (75, 0, 130),
    ],
    dtype=np.float32,
) / 255.0
CLASS_NAMES = [str(index) for index in range(1, 23)] + ["X", "Y"]
SPACE_CONFIG = {
    "Banding": ("banding", "fixed"),
    "FISH": ("fish", "fixed"),
    "Combine": ("combined", "fixed"),
    "3D": ("phy_3d", "live"),
    "MIP": ("mip_2d", "live"),
    "LiftedMIP": ("lifted_opt_mip", "live"),
    "Lifted2D": ("lifted_combined", "live"),
}
OUTPUT_SPACE_NAMES = {
    "Banding": "banding",
    "FISH": "fish",
    "Combine": "combined",
    "3D": "3d",
    "MIP": "mip",
    "LiftedMIP": "liftedmip",
    "Lifted2D": "lifted2d",
}


def project_root() -> Path:
    """Find the repository root when run in-place or as a copied script."""
    return find_release_root(__file__)


def portable_path(path: Path) -> str:
    """Keep release-local plot manifests independent of the author's machine."""
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: str | Path) -> Path:
    """Resolve relative paths from the ChromoWorld project root."""
    return resolve_release_path(value, __file__)


def stratified_indices(class_id: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Sample approximately equally per class while retaining all classes."""
    if max_points <= 0 or len(class_id) <= max_points:
        return np.arange(len(class_id), dtype=np.int64)
    rng = np.random.default_rng(seed)
    present = sorted(int(value) for value in np.unique(class_id) if 0 <= int(value) < 24)
    if not present:
        return np.empty(0, dtype=np.int64)
    budget = np.full(len(present), max_points // len(present), dtype=np.int64)
    budget[: max_points % len(present)] += 1
    selected: list[np.ndarray] = []
    for class_index, label in enumerate(present):
        candidates = np.flatnonzero(class_id == label)
        count = min(int(budget[class_index]), len(candidates))
        if count:
            selected.append(rng.choice(candidates, size=count, replace=False))
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def parse_class_names(value: str | None) -> list[int] | None:
    """Parse chromosome names such as ``1,5,13,X,Y`` into class IDs."""
    if not value:
        return None
    name_to_id = {name.lower(): index for index, name in enumerate(CLASS_NAMES)}
    selected = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in name_to_id:
            selected.append(name_to_id[token.lower()])
            continue
        try:
            class_id = int(token)
        except ValueError as error:
            raise SystemExit(f"Unknown chromosome class {token!r}; use 1-22, X, or Y") from error
        if not 0 <= class_id < 24:
            raise SystemExit(f"Numeric class ID must be between 0 and 23: {class_id}")
        selected.append(class_id)
    return list(dict.fromkeys(selected)) or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-npz", required=True, help="One named Fig4/Fig5 UMAP NPZ.")
    parser.add_argument("--space", choices=("auto", *SPACE_CONFIG), default="auto")
    parser.add_argument("--split", choices=("both", "pred", "gt"), default="both")
    parser.add_argument("--label-source", choices=("auto", "class", "mip", "canvas"),
                        default="auto", help="Fig5 labels: canvas by default for 3D/MIP; "
                        "LiftedMIP defaults to its canonical class labels.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to Code/Analyze/fig4 or fig5/demo_outputs/umap for the detected space",
    )
    parser.add_argument("--max-points", type=int, default=100000, help="Stratified points; 0 plots all points.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--highlight-classes",
        default=None,
        help="Comma-separated chromosome names to highlight, for example 1,5,13,X,Y.",
    )
    parser.add_argument("--context-alpha", type=float, default=0.12)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_path = resolve_project_path(args.embedding_npz)
    with np.load(embedding_path, allow_pickle=False) as data:
        if args.space == "auto":
            matches = [name for name, (prefix, _) in SPACE_CONFIG.items()
                       if f"{prefix}_embedding" in data.files]
            if len(matches) != 1:
                raise SystemExit(
                    "Could not infer one embedding space from NPZ; pass --space explicitly. "
                    f"Detected: {matches or 'none'}"
                )
            space = matches[0]
        else:
            space = args.space
        key_prefix, profile = SPACE_CONFIG[space]
        embedding_key = f"{key_prefix}_embedding"
        split_key = f"{key_prefix}_split"
        if embedding_key not in data:
            raise SystemExit(f"NPZ is missing {embedding_key!r}; available keys: {data.files}")
        embedding = np.asarray(data[embedding_key], dtype=np.float32)
        if profile == "fixed":
            if args.label_source not in ("auto", "class"):
                raise SystemExit("--label-source mip/canvas applies only to Fig5 embeddings")
            class_key = f"{key_prefix}_class_id"
            label_source = "class"
        elif space in ("3D", "MIP"):
            if args.label_source == "class":
                raise SystemExit("Fig5 3D/MIP embeddings use --label-source mip or canvas")
            label_source = "canvas" if args.label_source == "auto" else args.label_source
            class_key = f"y_{label_source}"
        elif space == "LiftedMIP":
            if args.label_source in ("auto", "class"):
                label_source = "class"
                class_key = f"{key_prefix}_y"
            else:
                label_source = args.label_source
                class_key = f"{key_prefix}_y_{label_source}"
        else:
            if args.label_source not in ("auto", "class"):
                raise SystemExit("Lifted2D has one class-label array; do not select mip/canvas")
            label_source = "class"
            class_key = f"{key_prefix}_y"
        if class_key not in data:
            raise SystemExit(f"NPZ is missing label array {class_key!r}; available keys: {data.files}")
        class_id = np.asarray(data[class_key], dtype=np.int32)
        split = np.asarray(data[split_key]).astype(str) if split_key in data else None

    output_dir = resolve_project_path(args.output_dir or (
        "Code/Analyze/fig4/demo_outputs" if profile == "fixed"
        else "Code/Analyze/fig5/demo_outputs/chromophy"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise SystemExit(f"Expected a two-dimensional embedding, got shape {embedding.shape}")
    if class_id.ndim != 1 or len(embedding) != len(class_id):
        raise SystemExit("Embedding and class-label arrays must be one-dimensional and row-aligned")
    if profile == "live" and args.split != "both":
        raise SystemExit("--split pred/gt is available only for Fig4 Banding/FISH/Combine embeddings")
    valid = np.isfinite(embedding).all(axis=1) & (class_id >= 0) & (class_id < 24)
    if args.split != "both":
        if split is None:
            raise SystemExit(f"The NPZ has no {split_key!r}; cannot filter split={args.split!r}")
        valid &= split == args.split
    base_indices = np.flatnonzero(valid)
    sampled_local = stratified_indices(class_id[base_indices], args.max_points, args.seed)
    indices = base_indices[sampled_local]
    if len(indices) == 0:
        raise SystemExit("No valid points remain after filtering")

    x = embedding[indices]
    labels = class_id[indices]
    highlight_classes = parse_class_names(args.highlight_classes)
    fig, ax = plt.subplots(figsize=(9.2, 7.8), constrained_layout=True)
    present = []
    if highlight_classes:
        context = ~np.isin(labels, highlight_classes)
        if context.any():
            ax.scatter(
                x[context, 0],
                x[context, 1],
                s=max(0.7, args.point_size * 0.65),
                alpha=float(np.clip(args.context_alpha, 0.0, 1.0)),
                color=(0.55, 0.55, 0.55),
                edgecolors="none",
                rasterized=True,
                label="other classes",
            )
        plot_classes = highlight_classes
    else:
        plot_classes = list(range(24))
    for label in plot_classes:
        mask = labels == label
        if not mask.any():
            continue
        present.append(label)
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            s=args.point_size,
            alpha=float(np.clip(args.alpha, 0.0, 1.0)),
            color=CLASS_COLORS_RGB[label],
            edgecolors="none",
            rasterized=True,
            label=CLASS_NAMES[label],
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=CLASS_COLORS_RGB[label],
            markeredgecolor="none",
            markersize=6,
            label=CLASS_NAMES[label],
        )
        for label in present
    ]
    if highlight_classes and np.any(~np.isin(labels, highlight_classes)):
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="", markerfacecolor=(0.55, 0.55, 0.55),
                markeredgecolor="none", markersize=6, label="other classes",
            )
        )
    ax.legend(handles=handles, title="Chromosome class", ncol=4, loc="upper left", frameon=False, fontsize=8)
    split_label = args.split if profile == "fixed" else label_source
    title = args.title or f"Chromo World 24-class UMAP — {space} — {split_label}"
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    suffix = "" if not highlight_classes else "_highlight_" + "-".join(CLASS_NAMES[label] for label in highlight_classes)
    stem = f"umap_{OUTPUT_SPACE_NAMES[space]}_{split_label}_24class{suffix}"
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=args.dpi, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    manifest = {
        "schema_version": "chromoworld.umap_24class_demo.v2",
        "project": "Chromo World",
        "embedding_file": portable_path(embedding_path),
        "space": space,
        "label_source": label_source,
        "split": args.split if profile == "fixed" else None,
        "n_valid": int(len(base_indices)),
        "n_plotted": int(len(indices)),
        "classes_plotted": [CLASS_NAMES[label] for label in present],
        "highlight_classes": [CLASS_NAMES[label] for label in highlight_classes] if highlight_classes else None,
        "features_are_precomputed": True,
        "output_png": portable_path(png_path),
        "output_svg": portable_path(svg_path),
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
