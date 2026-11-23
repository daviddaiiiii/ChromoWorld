"""Build browser assets from local ChromoWorld exports."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tifffile as tf
from PIL import Image, ImageOps

DOWNLOADS = Path.home() / "Downloads"
SITE = Path(__file__).resolve().parents[1]
ASSETS = SITE / "public/assets"
REPO = SITE.parent / "release"
RELEASE = DOWNLOADS / "ourslive/release"
REVIEW = DOWNLOADS / "ourslive/tag_candidate_review/extended_new_fluorescence"
RESAMPLE = Image.Resampling.LANCZOS


def frame(tiff, index):
    series = tiff.series[0]
    depth = series.shape[-3]
    return np.stack([p.asarray() for p in series.pages[index * depth:(index + 1) * depth]])


def save_image(image, path):
    image.save(path, "WEBP", quality=92, method=4)


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))

def class_centroids(xy, classes):
    points = []
    for label in np.unique(classes):
        if not 0 <= int(label) < 24:
            continue
        selected = xy[classes == label, :2]
        selected = selected[np.isfinite(selected).all(axis=1)]
        if len(selected):
            center = selected.mean(axis=0)
            points.append({"label": str(int(label)), "x": round(float(center[0]), 4),
                           "y": round(float(center[1]), 4), "n": len(selected)})
    return points


def fixed_assets():
    folder = ASSETS / "fish"
    folder.mkdir(parents=True, exist_ok=True)
    source = DOWNLOADS / "invert/fig3"
    name = "id01_HFF_FISH_00010"
    original = Image.open(source / f"{name}.png").convert("RGB")
    prediction = Image.open(source / f"{name}_ChromoPix.png").convert("RGB")
    dna = ImageOps.invert(original.crop((0, 0, 1024, 1024)))

    def clean_overlay(colored):
        # The archived no-text panels use a white-background display. Recover
        # their chroma separately so a dark display retains the original hues.
        from scipy import ndimage
        pixels = np.asarray(colored).astype(np.float32)
        chroma = pixels.max(axis=2) - pixels.min(axis=2)
        hue = pixels - pixels.min(axis=2, keepdims=True)
        strength = np.clip(chroma / 52., 0, .72)
        # Remove the thin boxes in the archived presentation export.
        visible = ndimage.binary_opening(chroma > 6, structure=np.ones((5, 5)))
        gray = np.asarray(dna).astype(np.float32)
        # Constrain the archive's colored overlay to visible DNA. This keeps
        # the display boxes from being mistaken for chromosome silhouettes.
        strength *= visible * np.clip((gray.mean(axis=2) - 13.) / 35., 0, 1)
        hue /= np.maximum(chroma[..., None], 1.)
        highlight = 55 + 200 * hue
        output = gray * (1 - strength[..., None]) + highlight * strength[..., None]
        return Image.fromarray(np.uint8(np.clip(output, 0, 255)))

    for key, img in [
        ("input", dna),
        ("reference", clean_overlay(original.crop((0, 1024, 1024, 2048)))),
        ("prediction", clean_overlay(prediction.crop((0, 1024, 1024, 2048)))),
    ]:
        save_image(img.resize((768, 768), RESAMPLE), folder / f"hff-{key}.webp")

    # The paired release grid contains three equal square panels per sample.
    grid = Image.open(REPO / "Code/Analyze/fig2-3/demo_outputs/FISH_5x3.png").convert("RGB")
    a = np.asarray(grid)
    rows = np.flatnonzero((a.mean(axis=(1, 2)) < 140))
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    bands = [g for g in groups if len(g) > 950]
    if len(bands) != 5:
        raise ValueError(f"Unexpected FISH grid row boundaries: {[len(g) for g in bands]}")
    for key, band in [("atlas-a", bands[1]), ("atlas-b", bands[3])]:
        y = int(band[0])
        for col, layer in enumerate(("input", "reference", "prediction")):
            x = col * 1034
            patch = grid.crop((x, y, x + 1024, y + 1024))
            save_image(patch.resize((768, 768), RESAMPLE), folder / f"{key}-{layer}.webp")

    banding_grid = Image.open(REPO / "Code/Analyze/fig2-3/demo_outputs/Banding_5x3.png").convert("RGB")
    banding = banding_grid.crop((0, 122, 1024, 1146))
    save_image(banding.resize((640, 640), RESAMPLE), ASSETS / "banding.webp")

    # Export class centroids from saved embeddings, keeping independently fitted coordinates separate.
    embeddings = {}
    for key, filename in [("banding", "Banding"), ("fish", "FISH"), ("joint", "Combine")]:
        with np.load(REPO / f"Artifacts/fig4/{filename}_umap_embed.npz", allow_pickle=False) as data:
            prefix = "combined" if filename == "Combine" else filename.lower()
            xy = data[f"{prefix}_embedding"]
            embeddings[key] = class_centroids(xy, data[f"{prefix}_class_id"])
    save_json(ASSETS / "centroids.json", embeddings)
    live_embeddings = {}
    for key, name, coords, label in [
        ("naive", "MIP", "mip_2d_embedding", "y_canvas"),
        ("geometry", "3D", "phy_3d_embedding", "y_canvas"),
        ("lifted", "LiftedMIP", "lifted_opt_mip_embedding", "lifted_opt_mip_y"),
    ]:
        with np.load(REPO / f"Artifacts/fig5/{name}_umap_embed.npz", allow_pickle=False) as data:
            live_embeddings[key] = class_centroids(data[coords], data[label])
    save_json(ASSETS / "live-centroids.json", live_embeddings)


def live_assets():
    sys.path.insert(0, str(DOWNLOADS / "patients 14 tiff 20261011/tracking_repair/algorithm"))
    from rounded_render import build_surfaces, shade
    spec = importlib.util.spec_from_file_location("glow", REVIEW / "glow_cyan.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    cases = [
        ("chr3", "chr3", "RPE-1 chr3_2026-09-08_5 CCDS6", [2, 3, 4, 5, 6, 7], [.355, .06, .06],
         REVIEW / "chr3/candidates.ome.tif", REVIEW / "chr3/image.ome.tif", "CcS2", [11, 13, 45, 49]),
        ("chr18", "chr18", "RPE-1 chr18_2026-09-09_6 CCDS5", list(range(8)), [.326, .06, .06],
         REVIEW / "chr18/group_2/candidates.ome.tif", REVIEW / "chr18/image.ome.tif", "CDS8", [9, 32, 51, 88]),
        ("mitosis", "others", "RPE1_3_F2CGD", list(range(19)), [.345, .108, .108],
         None, None, None, []),
    ]
    result = []
    from contextlib import ExitStack
    for key, group, name, times, spacing, candidate, tag, experiment, ids in cases:
        folder = ASSETS / "live" / key
        folder.mkdir(parents=True, exist_ok=True)
        sp = np.array(spacing)
        series = {"key": key, "name": name, "experiment": experiment,
                  "reference": {"chr3": 3, "chr18": 4, "mitosis": 18}[key],
                  "spacing": spacing, "candidates": ids, "frames": []}
        with ExitStack() as stack:
            raw = stack.enter_context(tf.TiffFile(RELEASE / f"image/raw_pre/tiff/{group}/{name}-DAPI.tif"))
            masks = stack.enter_context(tf.TiffFile(RELEASE / f"mask/chrp3d_post_tracked/{group}/{name}.tif"))
            print(key, raw.series[0].shape, raw.pages[0].tags["XResolution"].value, flush=True)
            if key == "mitosis":
                resolution = raw.pages[0].tags["XResolution"].value
                sp[1:] = resolution[1] / resolution[0]
                series["spacing"] = sp.tolist()
            series["xySize"] = int(raw.series[0].shape[-1])
            if candidate is not None:
                selected = stack.enter_context(tf.TiffFile(candidate))
                tagfile = stack.enter_context(tf.TiffFile(tag))
                depth = tagfile.series[0].shape[-3]
                mip_tags = {t: np.maximum.reduce([p.asarray() for p in tagfile.series[0].pages[(2*t+1)*depth:(2*t+2)*depth]]) for t in times}
                nonzero = np.concatenate([s[s > 0] for s in mip_tags.values()])
                high = float(np.percentile(nonzero, 99))
            for t in times:
                labels = frame(masks, t)
                subset = frame(selected, t) if candidate else np.zeros_like(labels)
                input_volume = frame(raw, t)
                bounds = (0., (labels.shape[0] - 1) * sp[0])
                surfaces, _ = build_surfaces(labels, sp, up=2)
                gray = np.full((int(max(labels.max(), subset.max())) + 1, 3), 205, np.uint8)
                gray[0] = 0
                if candidate:
                    foreground, _ = build_surfaces(subset, sp, up=2)
                    colored = gray.copy()
                    colored[1:] = legacy.palette.slot_to_color(key[3:])
                mip = input_volume.max(0).astype(float)
                lo, hi = np.percentile(mip, [1, 99.8])
                normalized = np.uint8(245 * np.clip((mip - lo) / max(1., hi - lo), 0, 1))
                dna = np.repeat(normalized[..., None], 3, axis=2)
                # Keep the label raster lossless and in the same view as the display.
                def export_pick(image, filename):
                    rgb = np.stack([image & 255, (image >> 8) & 255, (image >> 16) & 255], axis=-1).astype(np.uint8)
                    Image.fromarray(rgb).resize((420, 420), Image.Resampling.NEAREST).save(folder / filename)
                for face, surface in enumerate(surfaces):
                    suffix = "-back" if face else ""
                    depth_bounds = (-bounds[1], 0.) if face else bounds
                    mask_rgb = shade(surface, gray, sp, depth_bounds, up=2)
                    images = [("dna", dna[:, ::-1] if face else dna), ("mask", mask_rgb)]
                    export_pick(surface[0], f"{t}-ids{suffix}.png")
                    if candidate:
                        layer = shade(foreground[face], colored, sp, depth_bounds, up=2)
                        hit = foreground[face][0] > 0
                        candidate_rgb = mask_rgb.copy()
                        candidate_rgb[hit] = layer[hit]
                        picked = surface[0].copy()
                        picked[hit] = foreground[face][0][hit]
                        images.append(("candidates", candidate_rgb))
                        export_pick(picked, f"{t}-candidate-ids{suffix}.png")
                    for layer, rgb in images:
                        save_image(Image.fromarray(rgb).resize((420, 420), RESAMPLE), folder / f"{t}-{layer}{suffix}.webp")
                        if tag:
                            mip_tag = mip_tags[t][:, ::-1] if face else mip_tags[t]
                            save_image(legacy.glow(rgb, mip_tag, high), folder / f"{t}-{layer}-tag{suffix}.webp")
                counts = dict(zip(*np.unique(labels, return_counts=True)))
                candidate_counts = {str(i): int(np.count_nonzero(subset == i)) for i in ids}
                series["frames"].append({"t": t, "counts": {str(int(i)): int(n) for i, n in counts.items() if i > 0 and n > 0},
                                         "candidateCounts": candidate_counts})
                print("Rendered", key, t, flush=True)
        result.append(series)
    save_json(ASSETS / "live/index.json", result)


if __name__ == "__main__":
    if sys.argv[-1] == "live":
        live_assets()
    else:
        fixed_assets()
