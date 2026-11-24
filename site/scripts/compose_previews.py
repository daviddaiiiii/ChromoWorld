"""Compose clean, proportionate panels from sampled scientific assets."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parents[1] / "public/assets"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def pair(paths, labels, filename, crop=None):
    canvas = Image.new("RGB", (960, 446), "black")
    draw = ImageDraw.Draw(canvas)
    for idx, (path, label) in enumerate(zip(paths, labels)):
        region = Image.open(ASSETS / path).convert("RGB")
        if crop:
            region = region.crop(crop)
        region.thumbnail((412, 402), Image.Resampling.LANCZOS)
        canvas.paste(region, (idx*480 + (480-region.width)//2, (416-region.height)//2))
        draw.text((idx*480+240, 424), label, anchor="mt", fill="#ebe4fa", font=ImageFont.truetype(FONT, 21))
    draw.line((480,15,480,404), fill="#322a40", width=1)
    canvas.save(ASSETS / filename, "WEBP", quality=92, method=5)


def tracked_preview():
    data = ASSETS.parent.parent / "src/data"
    feature = json.loads((data / "live-feature.json").read_text())
    path = f"live/ordinary/{feature['sample']}/{feature['time']}"
    with Image.open(ASSETS / (path+"-mask.webp")) as mask:
        bounds = mask.convert("L").point(lambda x: 255 if x > 20 else 0).getbbox()
        if not bounds:
            raise ValueError("The featured frame has no visible tracked instances.")
        left, top, right, bottom = bounds
        side = min(mask.width, mask.height, max(right-left, bottom-top)+54)
        x = max(0, min(mask.width-side, (left+right-side)//2))
        y = max(0, min(mask.height-side, (top+bottom-side)//2))
    pair([path+"-dna.webp", path+"-mask.webp"], ["DNA projection", "Tracked instances"],
         "tracked-preview.webp", (x,y,x+side,y+side))

PALETTE = json.loads((ASSETS.parent.parent / "src/data/palette.json").read_text())
FONTS = ImageFont.truetype(FONT, 23)

def morphology_pair(source, keys, captions, output):
    spaces = json.loads((ASSETS / source).read_text())
    canvas = Image.new("RGB", (960, 446), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (key, caption) in enumerate(zip(keys, captions)):
        points = spaces[key]
        min_x, max_x = min(p["x"] for p in points), max(p["x"] for p in points)
        min_y, max_y = min(p["y"] for p in points), max(p["y"] for p in points)
        offset = 480 * index
        draw.line((offset + 55, 47, offset + 55, 363, offset + 425, 363), fill="#6d6478")
        for p in points:
            class_id = int(p["label"])
            label = str(class_id + 1) if class_id < 22 else ("X" if class_id == 22 else "Y")
            x = offset + 69 + 338 * (p["x"] - min_x) / max(max_x - min_x, .001)
            y = 346 - 282 * (p["y"] - min_y) / max(max_y - min_y, .001)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=PALETTE[label], outline="#b1a7c0")
        draw.text((offset + 240, 408), caption, anchor="mt", fill="#ebe4fa", font=FONTS)
    canvas.save(ASSETS / output, "WEBP", quality=94)

if __name__ == "__main__":
    pair(["fish/hff-input.webp", "fish/hff-prediction.webp"], ["DNA image", "ChromoPix prediction"], "fish-preview.webp")
    pair(["live/chr18/3-candidates-tag.webp", "live/chr18/3-dna-tag.webp"], ["Tracked instances", "DNA + Tag"], "live-preview.webp")
    pair(["live/ordinary/mitosis/18-dna.webp", "live/ordinary/mitosis/18-mask.webp"], ["DNA projection", "3D instance surfaces"], "pose-preview.webp")
    tracked_preview()
    morphology_pair("centroids.json", ["banding", "fish"], ["G-banding", "FISH"], "shape-preview.webp")
    morphology_pair("live-centroids.json", ["naive", "lifted"], ["Projected morphology", "Lifted morphology"], "lift-preview.webp")
