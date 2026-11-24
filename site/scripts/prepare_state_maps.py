"""Render the class distributions and centroids used by the identity explorer."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

SITE = Path(__file__).resolve().parents[1]
CLASSES = [str(n) for n in range(1, 23)] + ["X", "Y"]
PAPER_ORDER = CLASSES[:12] + ["X"] + CLASSES[12:22] + ["Y"]
PALETTE = json.loads((SITE / "src/data/palette.json").read_text())
SIDE = 630


def embedding(path, coord_key, label_key):
    with np.load(path, allow_pickle=False) as data:
        xy = data[coord_key].astype(float)
        labels = data[label_key]
    if xy.shape != (len(labels), 2) or not np.isfinite(xy).all() or not np.isin(labels, range(-1, 24)).all():
        raise ValueError(f"Invalid coordinates or classes in {path}")
    low, high = xy.min(axis=0), xy.max(axis=0)
    # Use one scale on both axes, centering the full extent with a narrow margin.
    unit = (xy - (low+high)/2) / (max(high-low)*1.06) + .5
    unit[:, 1] = 1 - unit[:, 1]
    pixel = np.rint(unit*(SIDE-1)).astype(int)
    flat = pixel[:, 1]*SIDE + pixel[:, 0]
    counts = np.bincount(flat, minlength=SIDE**2)
    rgb = np.array([ImageColor.getrgb(PALETTE[c]) for c in CLASSES] + [(115,115,115)])
    mean = np.stack([np.bincount(flat, weights=rgb[labels, k], minlength=SIDE**2)
                     for k in range(3)], axis=-1) / np.maximum(counts, 1)[:, None]
    strength = np.minimum(1, .51 + .49*np.log1p(counts)/np.log1p(np.percentile(counts[counts>0],97)))
    cloud = Image.fromarray((mean*strength[:,None]).reshape(SIDE,SIDE,3).astype('uint8'))
    centers = {}
    for i,c in enumerate(CLASSES):
        subset = unit[labels==i]
        if not len(subset):
            raise ValueError(f"Missing class {c} in {path}")
        centers[c] = [*subset.mean(axis=0).round(6).tolist(),len(subset)]
    return cloud, centers


def live_figure(story):
    with Image.open(story/'identity-chromoid.webp') as im:
        image = im.convert('RGB')
    coords = np.array(json.loads((story/'identity-centers.json').read_text())['chromoid'])
    if coords.shape!=(24,2) or not ((coords>0)&(coords<1)).all():
        raise ValueError("Expected 24 centroids in the coordinate system of Fig. 5E")
    # Keep the paper's aspect ratio and apply the same placement to its vector overlay.
    ratio = min(SIDE*.95/image.width,SIDE*.95/image.height)
    size = (round(image.width*ratio),round(image.height*ratio))
    origin = ((SIDE-size[0])//2,(SIDE-size[1])//2)
    cloud = Image.new('RGB',(SIDE,SIDE))
    cloud.paste(image.resize(size,Image.Resampling.LANCZOS),origin)
    normalized = (coords*size+origin)/SIDE
    return cloud, dict(zip(PAPER_ORDER,normalized.round(6).tolist()))


def cover(maps, folder):
    out = Image.new('RGB',(960,446))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf',29)
    except OSError:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',29)
    for i,(key,title) in enumerate([('banding','G-banding'),('fish','FISH'),('live','Live')]):
        with Image.open(folder/f'{key}.webp') as image:
            image = image.resize((308,308),Image.Resampling.LANCZOS).point(lambda n: round(n*.6))
        x,y = 320*i+6,38
        out.paste(image,(x,y))
        centers = maps[key]
        points = [(x+round(centers[c][0]*308),y+round(centers[c][1]*308)) for c in PAPER_ORDER]
        draw.line(points,fill='#d7d5d9',width=2)
        for c,(px,py) in zip(PAPER_ORDER,points):
            draw.ellipse((px-4,py-4,px+4,py+4),fill=PALETTE[c],outline='white',width=1)
        draw.text((x+154,365),title,anchor='mt',font=font,fill='#f2f0f4')
    out.save(folder.parent/'shape-preview.webp',quality=94,method=5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts',type=Path)
    parser.add_argument('story',type=Path,help='Fig. 5E cloud and original centroid exports from prepare_intro_story.py')
    args=parser.parse_args()
    folder=SITE/'public/assets/state-maps'
    folder.mkdir(exist_ok=True)
    maps={}
    for key,filename,coords,labels in [
        ('banding','fig4/Banding','banding_embedding','banding_class_id'),
        ('fish','fig4/FISH','fish_embedding','fish_class_id'),
        ('joint','fig4/Combine','combined_embedding','combined_class_id'),
        ('naive','fig5/MIP','mip_2d_embedding','y_canvas'),
        ('geometry','fig5/3D','phy_3d_embedding','y_canvas'),
        ('lifted','fig5/LiftedMIP','lifted_opt_mip_embedding','lifted_opt_mip_y'),
    ]:
        cloud,maps[key]=embedding(args.artifacts/f'{filename}_umap_embed.npz',coords,labels)
        cloud.save(folder/f'{key}.webp',lossless=True,method=5)
    cloud,maps['live']=live_figure(args.story)
    cloud.save(folder/'live.webp',lossless=True,method=5)
    (SITE/'src/data/state-maps.json').write_text(json.dumps(maps,separators=(',',':'))+'\n')
    cover(maps,folder)


if __name__=='__main__':
    main()
