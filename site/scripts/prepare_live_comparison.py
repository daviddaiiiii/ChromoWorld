"""Export synchronized front views from matching native 4D release TIFFs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile as tf

METHODS = {"pose": "chrp3d_post", "chimp": "chimp", "cpsam": "cpsam"}
GROUPS = {"others": "Live-cell series", "chr3": "Chr3-tagged series", "chr18": "Chr18-tagged series"}
FIG5 = "rpe1_60x_2_0002_561"
OUTPUT_SIZE = 420


def source_cases(root):
    cases = []
    for group, description in GROUPS.items():
        paths = sorted((root / "image/raw_pre/tiff" / group).glob("*-DAPI.tif"))
        for i, dna in enumerate(paths, 1):
            source = dna.stem.removesuffix("-DAPI")
            paths_by_layer = {"dna": dna, **{key: root / "mask" / stage / group / f"{source}.tif"
                                           for key, stage in METHODS.items()}}
            if all(p.is_file() for p in paths_by_layer.values()):
                label = "Fig. 5 live sequence" if source == FIG5 else f"{description} {i:02}"
                cases.append(("fig5" if source == FIG5 else f"{group}-{i:02}", group, label, source, paths_by_layer))
    return sorted(cases, key=lambda case: case[0] != "fig5")


def calibrated_xy(image):
    tags = image.pages[0].tags
    resolution = tags.get("XResolution")
    metadata = image.imagej_metadata or {}
    unit = metadata.get("unit")
    n, d = resolution.value if resolution else (0, 0)
    return float(d/n) if n and d and unit in ("micron", "um", "µm") else None


def read_frame(series, time):
    z = series.shape[1]
    return np.stack([p.asarray() for p in series.pages[time*z:(time+1)*z]])


def palette(max_label):
    """A stable, per-instance palette. Label colors are independent across methods."""
    i = np.arange(max_label+1, dtype=float)
    hue = np.uint8(i*.61803398875 % 1 * 255)
    hsv = np.stack([hue, 170+47*((i*37)%101)/100, 203+42*((i*53)%101)/100], axis=1)
    colors = np.array(Image.fromarray(np.uint8(hsv[None, :, :]), "HSV").convert("RGB"))[0]
    colors[0] = 0
    return colors


def surface(volume):
    front = (volume.shape[0]-1)-(volume[::-1] != 0).argmax(0)
    ids = np.take_along_axis(volume, front[None], axis=0)[0].astype(np.int32)
    height = front.astype(np.float32)
    x, y = np.zeros_like(height), np.zeros_like(height)
    dx = np.clip(height[:, 1:]-height[:, :-1], -2.5, 2.5)
    dy = np.clip(height[1:, :]-height[:-1, :], -2.5, 2.5)
    x[:, 1:] = dx * (ids[:, 1:] == ids[:, :-1])
    y[1:, :] = dy * (ids[1:, :] == ids[:-1, :])
    dot = (1.18+.36*x-.4*y)/np.sqrt(1+x*x+y*y)
    lighting = .65+.33*np.clip(dot,0,1)
    boundary = np.zeros_like(ids, dtype=bool)
    boundary[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    boundary[1:, :] |= ids[1:, :] != ids[:-1, :]
    lighting[boundary] *= .59
    return np.uint8(palette(int(ids.max()))[ids]*lighting[..., None])


def save(pixels, path):
    Image.fromarray(pixels).resize((OUTPUT_SIZE,OUTPUT_SIZE), Image.Resampling.LANCZOS).save(path, "WEBP", quality=87)


def export(case, out):
    key, group, label, source, paths = case
    folder = out / key
    folder.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        sources = {layer: resources.enter_context(tf.TiffFile(path)) for layer, path in paths.items()}
        series = {layer: obj.series[0] for layer, obj in sources.items()}
        reference = series["dna"]
        shape = reference.shape
        if len(shape) != 4 or shape[2] != shape[3] or any(s.shape != shape or s.axes != "TZYX" for s in series.values()):
            raise ValueError(f"Images and masks must share a square native TZYX grid: {source}")
        for key_name,s in series.items():
            if key_name != "dna" and not np.issubdtype(s.dtype, np.integer):
                raise ValueError(f"{source}: {key_name} is not an integer mask")
        n = shape[0]
        xy_size = calibrated_xy(sources["dna"])
        mip = [read_frame(reference, t).max(0) for t in range(n)]
        lo, hi = np.percentile(np.stack(mip), [38,99.7])
        frames = []
        for t, dna in enumerate(mip):
            save(np.uint8(255*np.clip((dna-lo)/max(hi-lo,1),0,1)**.83), folder / f"{t}-dna.webp")
            counts = {}
            for method in METHODS:
                volume = read_frame(series[method], t)
                if np.issubdtype(volume.dtype, np.signedinteger) and volume.min() < 0:
                    raise ValueError(f"Negative {method} label: {source}, t={t}")
                totals = np.bincount(volume.ravel())
                counts[method] = int(np.count_nonzero(totals[1:]))
                save(surface(volume), folder / f"{t}-{method}.webp")
            frames.append({"t":t, "counts":counts})
    print(f"{label}: {n} matched frames", flush=True)
    return {"key":key, "group":group, "name":label, "source":source,
            "xySize":shape[3], "spacingXY":xy_size, "reference":min(15,n-1) if key=="fig5" else n//2, "frames":frames}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("release", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda case: export(case, args.output), source_cases(args.release)))
    (args.output/"index.json").write_text(json.dumps(records,separators=(",",":")),encoding="utf-8")
