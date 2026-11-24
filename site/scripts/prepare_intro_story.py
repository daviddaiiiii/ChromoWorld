"""Prepare sourced karyotype and benchmark panels for the overview film."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save(image, directory, name):
    image.save(directory / (name + ".webp"), "WEBP", quality=94)


def chromosome_cutout(image, points):
    pts = np.array(points) * np.array(image.size)
    cx, cy = pts.mean(0)
    rx, ry = np.ceil(np.max(np.abs(pts - [cx, cy]), axis=0) + 3)
    radius = int(np.hypot(rx, ry))
    left, top = round(cx) - radius, round(cy) - radius
    mask = Image.new("L", (2 * radius, 2 * radius))
    ImageDraw.Draw(mask).polygon([tuple(xy) for xy in pts - [left, top]], fill=255)
    tile = image.crop((left, top, left + mask.width, top + mask.height))
    tile = Image.composite(tile, Image.new("L", tile.size), mask)
    axis = np.linalg.svd(pts - [cx, cy], full_matrices=False)[2][0]
    angle = (np.degrees(np.arctan2(axis[1], axis[0])) - 90) % 180
    tile = tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    return tile.crop(tile.getbbox())


def karyotype(test, specimen, dest):
    with Image.open(test / "image" / f"{specimen}.png") as source:
        img = source.convert("L")
    chromosomes = {i: [] for i in range(24)}
    for row in (test / "label/Banding" / f"{specimen}.txt").read_text().splitlines():
        cls, *coords = row.split()
        cls = int(cls)
        if cls in chromosomes:
            points = np.array([float(x) for x in coords]).reshape(-1, 2)
            chromosomes[cls].append(chromosome_cutout(img, points))
    if any(len(chromosomes[i]) != 2 for i in range(23)) or chromosomes[23]:
        raise ValueError("This demonstration expects a complete reference-annotated XX spread")

    board = Image.new("RGB", (684, 430), "black")
    d = ImageDraw.Draw(board)
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font = ImageFont.truetype(str(font_path) if font_path.exists() else "DejaVuSans.ttf", 22)
    scale = min(1.35, 106 / max(tile.height for pairs in chromosomes.values() for tile in pairs))
    for i, pairs in enumerate(chromosomes.values()):
        x, y = 44 + (i % 8) * 85, 9 + (i // 8) * 140
        for j, tile in enumerate(pairs):
            tile = tile.resize(tuple(max(1, round(v * scale)) for v in tile.size), Image.Resampling.LANCZOS)
            board.paste(tile, (x + 35*j - 18 - tile.width//2, y + 54 - tile.height//2))
        d.text((x, y + 116), str(i+1) if i<22 else "XY"[i-22], font=font, fill="#b4b4bd", anchor="mm")
    save(board, dest, "karyotype")
    for i, tile in enumerate(chromosomes[0]):
        save(tile, dest, f"karyotype-chr1-{i+1}")


def white_to_black(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    chroma = np.ptp(rgb, axis=2)
    neutral = (1 - rgb.mean(2)) ** .95
    saturation = np.clip(chroma * 1.8, 0, .82)[..., None]
    color = rgb / np.maximum(rgb.max(2)[..., None], .15)
    mapped = neutral[..., None] * (1-saturation) + color * saturation
    return Image.fromarray(np.uint8(np.clip(mapped, 0, 1)*255))


def fig2_panels(fig2_figure, dest):
    with Image.open(fig2_figure) as image:
        image = image.convert("RGB")

        def crop(bounds):
            return image.crop(tuple(round(v*s/r) for v,s,r in zip(bounds,image.size*2,(1273,1800)*2)))

        panel = crop((249,108,939,511))
        ImageDraw.Draw(panel).rectangle((0,0,round(panel.width*.053),round(panel.height*.06)),fill="white")
        save(white_to_black(panel), dest, "banding-transfer-metrics")
        # Fig. 2A artwork interiors. Fade the white paper matte at the edges
        # while keeping the original pale-blue artwork and its lighter stripes.
        for name, bounds in (("dehydrated", (39,99,226,266)),
                             ("rehydrated", (48,296,232,470))):
            panel = crop(bounds)
            rgb = np.asarray(panel, dtype=np.int16)
            alpha = np.clip((rgb[...,2] - rgb[...,0] - 1)*255./4, 0, 255)
            panel = Image.composite(panel, Image.new("RGB", panel.size), Image.fromarray(alpha.astype(np.uint8)))
            save(panel, dest, f"state-{name}")
        # The upper row of Fig. 2D uses G-banding-trained weights for all models.
        for name, left, right in (("mask-rcnn",343,640), ("v11",652,949), ("v12",961,1258)):
            save(crop((left,550,right,854)), dest, f"banding-only-{name}")


def fig5_panels(fig5, dest):
    for index,name in ((259,"mip-raw"),(264,"mip-naive"),(267,"mip-optimal")):
        with Image.open(fig5 / f"image{index}.png") as image:
            save(image.convert("RGB"),dest,name)


def pose_panels(figure, dest):
    # Image interiors in Supplementary Fig. S2c, without borders or headings.
    with Image.open(figure) as source:
        figure = source.convert("RGB")
    for name, x in (("fast-chimp", 78), ("cellpose", 325), ("chromopose", 570)):
        box = (x/1261, 591/1560, (x+232)/1261, 823/1560)
        crop = figure.crop(tuple(round(a*b) for a,b in zip(box, figure.size*2)))
        save(crop, dest, f"pose-{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("fig2_figure", type=Path, help="Raster export of the full Fig. 2 plate")
    parser.add_argument("fig5", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pose-figure", type=Path, required=True,
                        help="Raster export of the full Supplementary Fig. S2 plate")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    karyotype(args.dataset,"Banding_05535",args.output)
    fig2_panels(args.fig2_figure,args.output)
    fig5_panels(args.fig5,args.output)
    pose_panels(args.pose_figure,args.output)
    (args.output/"sources.json").write_text(json.dumps({
        "karyotype":"Dataset/2D held-out G-banding, Banding_05535; arranged by the reference annotations",
        "preparation_states":"Fig. 2A dehydrated and rehydrated chromosome schematics",
        "banding_only":"Fig. 2D top row: Mask R-CNN, YOLOv11-Seg and YOLOv12-Seg on the same held-out FISH field (id01_HFF_FISH_00023)",
        "projection":"Archival Fig. 5A/B image panels from the presentation",
        "identity_flow":"LiftedMIP_umap_embed.npz; same embedding coordinates, three sets of inferred identities",
        "pose_comparison":"Supplementary Fig. S2c; three predictions on the same live-cell frame"
    },indent=2)+"\n")


if __name__ == "__main__":
    main()
