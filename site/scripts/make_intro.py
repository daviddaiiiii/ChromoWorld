"""Render the homepage explainer from microscopy and archived morphology embeddings."""
from functools import lru_cache
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SITE = Path(__file__).resolve().parents[1]
ASSETS = SITE / "public/assets"
SOURCE = Path(os.environ.get("CHROMOWORLD_FILM_MEDIA", SITE / "src/data/overview"))
FIG4 = SITE.parent / "release/Artifacts/fig4"
W, H, FPS, DURATION = 1280, 720, 24, 333
BLACK, WHITE, MUTED = "#000000", "#f8faff", "#c1c6d4"
GRID, AXIS, HIRULE = "#151a20", "#626b77", "#262a32"
FONT = Path("/System/Library/Fonts/Supplemental")
CLASS_NAMES = [str(i) for i in range(1, 23)] + ["X", "Y"]
PAPER_COLORS = json.loads((SITE / "src/data/palette.json").read_text())
PALETTE = [ImageColor.getrgb(PAPER_COLORS[name]) for name in CLASS_NAMES]
EXAMPLE = json.loads((SOURCE / "example.json").read_text())
ACCENT = PALETTE[EXAMPLE["class_id"]]
FEATURES = (
    ("Relative area", "norm_size", ".2%"),
    ("Eccentricity", "eccentricity", ".2f"),
    ("Solidity", "solidity", ".2f"),
    ("Circularity", "circularity", ".2f"),
    ("Tortuosity", "tortuosity", ".2f"),
)


@lru_cache(maxsize=None)
def font(size, bold=False):
    avenir = Path("/System/Library/Fonts/Avenir Next.ttc")
    if avenir.is_file():
        return ImageFont.truetype(str(avenir), size, index=2 if bold else 5)
    path = FONT / ("Arial Bold.ttf" if bold else "Arial.ttf")
    return ImageFont.truetype(str(path) if path.exists() else
                              ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


@lru_cache(maxsize=130)
def picture(path, width=None):
    with Image.open(path) as image:
        im = image.convert("RGB")
    return im if width is None else im.resize((width, width), Image.Resampling.LANCZOS)


def clamp(x):
    return min(1., max(0., x))


def ramp(t, start, end):
    p = clamp((t-start)/(end-start))
    return p*p*(3-2*p)


def camera(t, start, end):
    # Slow into the cut and settle without an overshoot.
    p = clamp((t-start)/(end-start))
    return p*p*p*(10+p*(-15+6*p))


def mix(a, b, frac):
    return a + (b-a)*frac


def tone(color, alpha):
    if isinstance(color, str):
        color = ImageColor.getrgb(color)
    return tuple(round(c*clamp(alpha)) for c in color)


def blend(a, b, p):
    if p <= 0:
        return a
    if p >= 1:
        return b
    return Image.blend(a, b, p)


def light(layer, opacity):
    if opacity <= 0:
        return Image.new("RGB", layer.size)
    if opacity >= 1:
        return layer
    return ImageEnhance.Brightness(layer).enhance(opacity)


def text(im, xy, label, size=34, color=WHITE, bold=False, anchor="la"):
    ImageDraw.Draw(im).text(xy, label, fill=color, font=font(size,bold), anchor=anchor)


def dot(draw, x, y, r, color, outline=None, width=1):
    draw.ellipse((x-r,y-r,x+r,y+r),fill=color,outline=outline,width=width)


def line(im, a, b, color=AXIS, width=1):
    ImageDraw.Draw(im).line((a,b),fill=color,width=width)


def paste(im, layer, center):
    im.paste(layer,(round(center[0]-layer.width/2),round(center[1]-layer.height/2)))


@lru_cache(maxsize=140)
def fitted(layer_path, width, height):
    layer = picture(layer_path)
    scale = min(width/layer.width, height/layer.height)
    return layer.resize((round(layer.width*scale), round(layer.height*scale)), Image.Resampling.LANCZOS)


def plate(im, path, center, size, p=1.):
    if p > 0:
        paste(im, light(fitted(path,*size),p), center)


def shot(context, title):
    im = Image.new("RGB",(W,H),BLACK)
    text(im,(77,39),context.upper(),18,"#969eaa",True)
    size = min(57, int(1128*57/font(57,True).getlength(title)))
    text(im,(74,66),title,size,WHITE,True)
    return im


def caption(im, value, opacity=1.):
    if opacity <= 0:
        return
    line(im,(76,642),(1204,642),tone(HIRULE, opacity))
    size = min(26, int(1128*26/font(26).getlength(value)))
    text(im,(76,656),value,size,tone(MUTED,opacity))


def label(im, x, y, value, opacity=1., color=WHITE, size=27):
    if opacity > 0:
        text(im,(x,y),value,size,tone(color,opacity),True,"ma")


def linked(im, points, color, progress=1., marker=True):
    """Trace a thin sight-line from annotation to the exact feature it names."""
    if progress <= 0:
        return
    remaining = sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(points,points[1:])) * progress
    for a,b in zip(points,points[1:]):
        length = np.hypot(b[0]-a[0],b[1]-a[1])
        if length:
            f = min(1., remaining/length)
            line(im,a,(mix(a[0],b[0],f),mix(a[1],b[1],f)),color,2)
        remaining -= length
        if remaining <= 0:
            break
    if marker and progress >= 1.:
        dot(ImageDraw.Draw(im), *points[-1],2.8,color)


def reveal_registered(before, after, p, feather=25):
    """Reveal a layer in place. Input images share their crop and pixel grid."""
    if p <= 0:
        return before
    if p >= 1:
        return after
    width,height = before.size
    edge = mix(-feather, width+feather, p)
    alpha = np.clip((edge-np.arange(width)+feather/2)/feather,0,1)
    mask = Image.fromarray(np.uint8(255*np.broadcast_to(alpha, (height,width))))
    return Image.composite(after,before,mask)


def live(case, t, layer, width):
    return picture(SOURCE / f"live/{case}/{t}-{layer}.webp", width)


def fish(layer, width):
    return picture(SOURCE / f"fish-{layer}.webp", width)


def field_layers():
    dna = picture(SOURCE / "example-dna.webp")
    mask = Image.open(SOURCE / "example-mask.png").convert("L")
    mask = mask.resize(dna.size, Image.Resampling.NEAREST)
    edge_radius = max(1,round(dna.width/384))
    rim = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(1+2*edge_radius)),
                              mask.filter(ImageFilter.MinFilter(1+edge_radius//2*2)))
    tinted = blend(dna, Image.new("RGB", dna.size, ACCENT), .30)
    selected = Image.composite(tinted, ImageEnhance.Brightness(dna).enhance(.36), mask)
    selected.paste(ACCENT, (0, 0, *dna.size), rim)
    isolated = Image.composite(tinted, Image.new("RGB", dna.size), mask)
    isolated.paste(ACCENT, (0, 0, *dna.size), rim)
    return dna, selected, isolated, mask.getbbox()


@lru_cache(maxsize=1)
def microscopy():
    return field_layers()


def field_view(im, zoom, isolation, center, width, highlight=1.):
    dna, marked, isolated, bbox = microscopy()
    left, top, right, bottom = bbox
    # Zoom and pan into the original mask in the same coordinate system.
    side = mix(dna.width, dna.width*172/768, zoom)
    cx = mix(dna.width/2, (left + right) / 2, zoom)
    cy = mix(dna.height/2, (top + bottom) / 2, zoom)
    crop = tuple(round(c) for c in (cx-side/2, cy-side/2, cx+side/2, cy+side/2))
    field = blend(blend(dna, marked, highlight), isolated, isolation)
    tile = field.crop(crop).resize((width, width), Image.Resampling.BICUBIC)
    if zoom > .7 and isolation > 0:
        detail, _ = invariant_example()
        detail_side = right-left+2*DETAIL_PAD
        display_width = max(1,round(detail_side*width/side))
        target = detail.resize((display_width,max(1,round(display_width*detail.height/detail.width))),
                               Image.Resampling.LANCZOS)
        precise = Image.new("RGB",tile.size)
        paste(precise,target,(((right+left)/2-crop[0])*width/side,((bottom+top)/2-crop[1])*width/side))
        tile = blend(tile,precise,ramp(zoom,.7,1)*isolation)
    paste(im,tile,center)
    if highlight > 0 and zoom < .7:
        x = center[0] - width / 2 + ((left+right)/2 - crop[0]) * width / (crop[2]-crop[0])
        y = center[1] - width / 2 + ((top+bottom)/2 - crop[1]) * width / (crop[3]-crop[1])
        r = max(right-left, bottom-top) * width / side * .64
        dot(ImageDraw.Draw(im), x, y, r, None, tone(ACCENT, highlight*(1-ramp(zoom, .35, .7))), 3)


class Projection:
    def __init__(self, rect, bounds):
        self.rect = rect
        x, y, w, h = rect
        lo, hi = np.array(bounds)
        self.mid = (lo + hi) / 2
        self.scale = min(w / (hi[0]-lo[0]), h / (hi[1]-lo[1]))
        self.origin = np.array([x + w/2, y + h/2])

    def coordinates(self, points):
        return self.origin + (np.array(points)-self.mid) * [self.scale, -self.scale]

    def axes(self, im, step=5, names=True, visible=1.):
        x, y, w, h = self.rect
        for v in range(-40, 41, step):
            gx, gy = self.coordinates([v, v])
            if x+2 < gx < x+w-2:
                line(im, (gx, y), (gx, y+h), tone(GRID, visible))
            if y+2 < gy < y+h-2:
                line(im, (x, gy), (x+w, gy), tone(GRID, visible))
        line(im, (x, y+h), (x+w, y+h), tone(AXIS, visible))
        line(im, (x, y), (x, y+h), tone(AXIS, visible))
        if names:
            text(im, (x+w-1, y+h+7), "UMAP 1", 20, tone(MUTED, visible), anchor="ra")
            text(im, (x+70, y-25), "UMAP 2", 20, tone(MUTED, visible), anchor="ra")


class Embedding:
    def __init__(self, filename, prefix):
        with np.load(FIG4 / filename, allow_pickle=False) as data:
            self.xy = data[prefix + "_embedding"]
            self.labels = data[prefix + "_class_id"]
            self.ids = data[prefix + "_global_idx"]
            self.splits = data[prefix + "_split"]
        if not np.isfinite(self.xy).all() or not np.isin(self.labels, range(len(PALETTE))).all():
            raise ValueError(f"Unexpected invalid rows in {filename}")
        self.centroids = np.array([self.xy[self.labels == i].mean(0) for i in range(len(PALETTE))])
        self.bounds = np.array([self.xy.min(0) - .8, self.xy.max(0) + .8])
        # A fixed order for progressive reveal. Later points remain at their original coordinates.
        self.order = np.random.default_rng(103).permutation(len(self.xy))

    @lru_cache(maxsize=3)
    def example(self):
        rows = np.flatnonzero(self.ids == EXAMPLE["global_idx"])
        if len(rows) != 1:
            raise ValueError("Microscopy example must map to one archived row")
        row = rows[0]
        split = "gt" if EXAMPLE["if_gt"] else "pred"
        if self.labels[row] != EXAMPLE["class_id"] or self.splits[row] != split:
            raise ValueError("Microscopy example differs from its archived label")
        return self.xy[row]

    def cloud(self, projection, count=None):
        index = self.order if count is None else self.order[:count]
        return raster_cloud(self.xy[index], self.labels[index], projection)

    def alignment(self, reference):
        source, target = self.centroids.astype(float), reference.centroids.astype(float)
        centered = source - source.mean(0)
        goal = target - target.mean(0)
        u, _, vt = np.linalg.svd(centered.T @ goal)
        return centered @ (u @ vt) * (np.linalg.norm(goal)/np.linalg.norm(centered)) + target.mean(0)


def raster_cloud(points, classes, projection):
    x, y, w, h = projection.rect
    xy = np.rint(projection.coordinates(points) - [x, y]).astype(int)
    visible = np.all((xy >= 0) & (xy < [w, h]), axis=1)
    xy, labels = xy[visible], classes[visible]
    if len(xy) < 3200:
        layer = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(layer)
        radius = 3 if len(xy) < 400 else 2
        for (px, py), label in zip(xy, labels):
            dot(d, px, py, radius, PALETTE[label])
        return layer
    # Render every archived point into its pixel. Blend classes where
    # points overlap and compress density so sparse regions stay visible.
    flat = xy[:, 0] + xy[:, 1] * w
    frequency = np.bincount(flat, minlength=w*h)
    colors = np.array(PALETTE)[labels]
    occupied = frequency > 0
    rgb = np.zeros((w*h, 3), np.uint8)
    gain = np.minimum(.98, .62 + .14*np.log1p(frequency[occupied])) / frequency[occupied]
    for c in range(3):
        rgb[occupied, c] = np.bincount(flat, colors[:, c], minlength=w*h)[occupied] * gain
    return Image.fromarray(rgb.reshape(h, w, 3))



@lru_cache(maxsize=1)
def embeddings():
    combined = Embedding("Combine_umap_embed.npz", "combined")
    banding = Embedding("Banding_umap_embed.npz", "banding")
    fish_cloud = Embedding("FISH_umap_embed.npz", "fish")
    fish_cloud.example()
    return combined, banding, fish_cloud

def preparation_states(t):
    im = shot("Fig. 2A / schematic", "What changes from G-banding to FISH?")
    states = (
        (335, "dehydrated", "G-banding", "Dehydrated · distinct stripes", 0.0),
        (945, "rehydrated", "FISH", "Rehydrated · varied shapes without bands", 2.5),
    )
    for x, name, title, description, delay in states:
        p = ramp(t, delay, delay+.85)
        label(im, x, 176, title, p, WHITE, 30)
        plate(im, STORY/f"state-{name}.webp", (x, 382), (372, 299), p)
        text(im, (x, 553), description, 24, tone(MUTED,p), anchor="ma")
    line(im, (640, 176), (640, 596), tone(HIRULE, ramp(t,2.5,3.35)))
    caption(im, "Preparation changes the texture and shape cues available to the models.")
    return im


@lru_cache(maxsize=16)
def fish_overlay(width):
    dna = np.asarray(picture(SOURCE/"example-dna.webp",width),dtype=np.float32)
    probe = picture(SOURCE/"fish-probe.webp",width)
    halo = np.asarray(probe.filter(ImageFilter.GaussianBlur(2)),dtype=np.float32)
    signal = np.asarray(probe,dtype=np.float32)
    return Image.fromarray(np.uint8(np.clip(dna*.74+signal*2.3+halo*1.6,0,255)))


def virtual_fish(t):
    if t >= 13.7:
        p = camera(t,13.7,15.7)
        im = shot("Fixed cells / ChromoPix","What does shape tell us?")
        field_view(im,0.,0.,(mix(334,640,p),382),round(mix(444,452,p)),p)
        caption(im,"Follow one FISH-annotated chromosome into morphology space.",p)
        return im
    if t >= 6.3:
        im = shot("Fixed cells / ChromoPix","From a probe signal to a predicted identity")
        p = camera(t,6.3,8.2)
        w = 444
        paste(im,fish("reference",w),(334,382))
        paste(im,reveal_registered(picture(SOURCE/"example-dna.webp",w),fish("prediction",w),p),(946,382))
        label(im,334,597,"FISH annotation")
        label(im,946,597,"ChromoPix prediction",p)
        caption(im,"ChromoPix learns from both G-banding and FISH to segment and identify chromosomes.")
        return im
    im = shot("Fixed cells / FISH","A DNA probe locates chromosome 1")
    pullback = camera(t,0,1.55)
    if pullback < 1:
        field_view(im,1-pullback,1-pullback,(mix(949,640,pullback),mix(379,382,pullback)),
                   round(mix(459,490,pullback)),1-pullback)
        caption(im,"A sequence-specific fluorescent probe marks Chr1 in this fixed cell.")
        return im
    layer = ramp(t,1.55,2.72)
    split = camera(t,3.05,4.55)
    reference = ramp(t,4.60,5.58)
    x = mix(640,334,split)
    w = round(mix(490,444,split))
    paste(im,blend(picture(SOURCE/"example-dna.webp",w),fish_overlay(w),layer),(x,382))
    if reference:
        paste(im,light(fish("reference",444),reference),(946,382))
    label(im,x,597,"DNA + Chr1 probe" if layer>.40 else "DNA image")
    label(im,946,597,"FISH annotation",reference)
    caption(im,"A sequence-specific fluorescent probe marks Chr1 in this fixed cell.")
    return im

def features(im,t,opacity=1.):
    if opacity <= 0:
        return
    y = 216
    text(im,(663,y),"SHAPE MEASUREMENTS",18,tone(MUTED,opacity),True)
    for i,(name,key,spec) in enumerate(FEATURES):
        p = ramp(t,3.2+i*.42,4.0+i*.42)*opacity
        if p:
            yp = 257+i*58
            line(im,(663,yp-7),(1170,yp-7),tone(HIRULE,p))
            text(im,(663,yp+6),name,31,tone(WHITE,p))
            text(im,(1170,yp+6),format(EXAMPLE["features"][key],spec),
                 31,tone(ACCENT,p),True,"ra")


def morphology(t):
    im = shot("Morphology","Measure one chromosome")
    p = camera(t,.6,4.3)
    isolation = camera(t,1.7,4.6)
    field_view(im,p,isolation,(mix(640,344,p),382),round(mix(452,500,p)))
    features(im,t)
    label(im,344,571,"FISH-annotated Chr1",ramp(t,4.5,5.5),ACCENT)
    caption(im,"Quantify relative size and shape directly from the segmented instance.")
    return im


JOINT_RECT = (90,187,1088,385)


def zoomed_axes(embedding,progress):
    p = embedding.example()
    close = np.array([p-[1.72,1.08], p+[1.72,1.08]])
    return Projection(JOINT_RECT,mix(close,embedding.bounds,progress))


def example_dot(im,pr,size=8,opacity=1.,note=False):
    x,y = pr.coordinates(embeddings()[0].example())
    color = tone(ACCENT,opacity)
    d = ImageDraw.Draw(im)
    dot(d,x,y,size+7,None,color,2)
    dot(d,x,y,size,color)
    if note:
        linked(im,((x+16,y-10),(x+43,y-41),(x+99,y-41)),tone(ACCENT,.68))
        text(im,(x+110,y-60),"This Chr1 instance",25,WHITE,True)


def to_one_point(t):
    im = shot("Morphology / UMAP","One chromosome, one point")
    p = camera(t,.75,4.1)
    pr = zoomed_axes(embeddings()[0],0)
    vis = ramp(t,1.46,3.24)
    if vis:
        pr.axes(im,step=1,visible=vis)
    x,y=pr.coordinates(embeddings()[0].example())
    if p < 1:
        width = round(500*(.05**p))
        field_view(im,1,1,(mix(344,x,p),mix(382,y,p)),width)
    features(im,10,1-ramp(t,.5,1.85))
    label(im,344,571,"FISH-annotated Chr1",1-ramp(t,.4,1.2),ACCENT)
    opacity = ramp(t,3.2,4.2)
    if opacity:
        example_dot(im,pr,round(mix(5,8,opacity)),opacity,note=opacity>=1)
    caption(im,"UMAP gives this chromosome a position based on its five shape measurements.")
    return im


def palette(im,y=609,opacity=1.):
    if opacity <= 0:
        return
    left, step = 651,22
    d = ImageDraw.Draw(im)
    text(im,(left-15,y-3),"Chromosome class",19,tone(MUTED,opacity),False,"ra")
    for i,value in enumerate(PALETTE):
        x = left+i*step
        d.rectangle((x,y,x+17,y+7),fill=tone(value,opacity))
        if i in (0,5,11,17,21,22,23):
            text(im,(x+8,y+10),CLASS_NAMES[i],15,tone(MUTED,opacity),False,"ma")


@lru_cache(maxsize=8)
def full_cloud(which, rect):
    e = embeddings()[which]
    return e.cloud(Projection(rect,e.bounds))


def paste_cloud(im, cloud, at):
    mask = cloud.convert("L").point(lambda n: 255 if n else 0)
    im.paste(cloud,at,mask)


def population(t):
    combined = embeddings()[0]
    im = shot("Morphology / UMAP","A map of chromosome shapes")
    z = camera(t,.35,6.75)
    pr = zoomed_axes(combined,z)
    total = len(combined.xy)
    count = round(np.exp(mix(np.log(2),np.log(total),ramp(t,.4,7.9))))
    if t >= 8.1:
        cloud = full_cloud(0,JOINT_RECT)
    elif t >= 6.75:
        cloud = blend(combined.cloud(pr,min(count,103000)),
                      full_cloud(0,JOINT_RECT),ramp(t,6.75,8.1))
    else:
        cloud = combined.cloud(pr,count)
    pr.axes(im,step=1 if z<.19 else 5)
    paste_cloud(im,cloud,JOINT_RECT[:2])
    example_dot(im,pr,8,note=t<1.8)
    p=ramp(t,6.4,8.35)
    if p:
        text(im,(1186,218),f"{total:,}",53,tone(WHITE,p),True,"ra")
        text(im,(1186,284),"chromosome instances",24,tone(MUTED,p),False,"ra")
    palette(im,603,ramp(t,2.8,4.35))
    caption(im,"Each dot is one chromosome instance. Color denotes its class.")
    return im


COMPARISON_PANELS = [(75,216,526,333),(678,216,526,333)]
NATIVE_RECT = (34,20,475,296)


def draw_centroid(draw, point, color, reference_probe):
    x,y = point
    if reference_probe:
        draw.polygon([(x,y-10),(x+10,y),(x,y+10),(x-10,y)],fill=BLACK,outline=color,width=2)
    else:
        dot(draw,x,y,7,color,WHITE,1)


def native_comparison(t):
    im = shot("Morphology / UMAP","Two preparations, two maps")
    p = ramp(t,.8,2.72)
    q = ramp(t,3.78,5.32)
    for i,(name,panel,source) in enumerate(zip(
        ("G-banding","FISH"),COMPARISON_PANELS,("banding-dna.webp","example-dna.webp"),
    )):
        e=embeddings()[i+1]
        x,y,w,h=panel
        sample=Image.new("RGB",(w,h))
        paste(sample,picture(SOURCE/source,333),(w/2,h/2))
        pr=Projection(NATIVE_RECT,e.bounds)
        chart=Image.new("RGB",(w,h))
        pr.axes(chart,names=False)
        cloud=light(full_cloud(i+1,NATIVE_RECT),1-.82*q)
        paste_cloud(chart,cloud,NATIVE_RECT[:2])
        if q:
            marks=chart.copy()
            d=ImageDraw.Draw(marks)
            for point,color in zip(pr.coordinates(e.centroids),PALETTE):
                draw_centroid(d,point,color,bool(i))
            chart=blend(chart,marks,ramp(q,0,.84))
        im.paste(reveal_registered(sample,chart,p,50),(x,y))
        label(im,x+w/2,166,name,1.,WHITE,31)
        label(im,x+w/2,576,f"{len(e.xy):,} instances",p,MUTED,24)
    line(im,(640,179),(640,583),HIRULE)
    caption(im, "One marker per chromosome class" if q>.5 else "UMAP is fit independently within each preparation.")
    return im


@lru_cache(maxsize=1)
def centroid_positions():
    c,b,f=embeddings()
    end=(b.alignment(c), f.alignment(c))
    merged=np.concatenate(end)
    pr=Projection((210,162,860,407),[merged.min(0)-.7,merged.max(0)+.7])
    starts=[Projection(NATIVE_RECT,e.bounds).coordinates(e.centroids)+panel[:2]
            for e,panel in zip((b,f),COMPARISON_PANELS)]
    return starts,[pr.coordinates(v) for v in end]


def centroid_comparison(t):
    im=shot("Morphology / UMAP","A shared order across preparations")
    start,end=centroid_positions()
    p=camera(t,.1,3.25)
    b,f=[mix(a,z,p) for a,z in zip(start,end)]
    vis=ramp(t,2.25,3.8)
    d=ImageDraw.Draw(im)
    order=[*range(12),22,*range(12,22),23]
    if vis:
        for i,j in zip(order,order[1:]):
            line(im,tuple(b[i]),tuple(b[j]),tone(PALETTE[i],.26*vis),2)
    keys={0:(-35,-33),6:(-34,15),13:(23,14),21:(21,-27)}
    for j,(point_a,point_b,col) in enumerate(zip(b,f,PALETTE)):
        if vis:
            line(im,tuple(point_a),tuple(point_b),tone(col,vis*(.81 if j in keys else .27)),2)
        draw_centroid(d,point_a,col,False)
        draw_centroid(d,point_b,col,True)
        if vis and j in keys:
            dx,dy=keys[j]
            pos=point_a if j in (0,13) else point_b
            text(im,(pos[0]+dx,pos[1]+dy),CLASS_NAMES[j],25,tone(WHITE,vis),True,"mm")
    draw_centroid(d,(365,588),WHITE,False)
    text(im,(384,571),"G-banding",24,MUTED)
    draw_centroid(d,(575,588),WHITE,True)
    text(im,(596,571),"FISH",24,MUTED)
    if vis:
        line(im,(755,588),(786,588),tone(MUTED,vis),2)
        text(im,(800,571),"Same chromosome class",24,tone(MUTED,vis))
    caption(im,"The 24 class centers retain a similar order after alignment.")
    return im

@lru_cache(maxsize=1)
def fig5_frames():
    source=json.loads((SOURCE/"fig5-live/source.json").read_text())
    frames=[frame["t"] for frame in source["frames"]]
    if frames[-1] != source["isolated_anchor_t"]:
        raise ValueError("Fig. 5 tracking shot must end at the isolated chromosome frame")
    return frames


def fig5_layer(time,layer,width):
    return picture(SOURCE/f"fig5-live/{time}-{layer}.webp",width)


def pose_comparison(t):
    im = shot("ChromoPose / Supplementary Fig. S2c", "The same frame, three segmentations")
    methods = (
        ("fast-chimp", "FAST-CHIMP (EmbedSeg)", "Embeddings + watershed refinement", "Many small fragments"),
        ("cellpose", "Cellpose-SAM", "Diffusion-based flows", "Broad, joined-looking regions"),
        ("chromopose", "ChromoPose", "Chromosome-guided flows", "Slender, more continuous instances"),
    )
    detail = camera(t, 4.8, 6.8)
    reveal = ramp(t, 5.9, 7.1)
    for x, (name, method, approach, observation) in zip((252, 640, 1028), methods):
        label(im, x, 165, method, color="#d5c2ff" if name == "chromopose" else WHITE, size=26)
        text(im, (x, 200), approach, 20, MUTED, anchor="ma")
        img = picture(SOURCE/f"story/pose-{name}.webp")
        inset = detail*.065
        box = (inset*img.width, inset*img.height, (1-inset)*img.width, (1-inset)*img.height)
        paste(im, img.resize((346, 346), Image.Resampling.LANCZOS, box=box), (x, 403))
        text(im, (x, 583), observation, 21, tone(WHITE, reveal), True, "ma")
    for x in (446, 834):
        line(im, (x,171), (x,606), HIRULE)
    caption(im, "Qualitative comparison on one live-cell frame. Colors distinguish instances within each panel.")
    return im


def schematic_centerline(p, index):
    bend = np.sin(np.pi*(p+.12))
    x = 135 + 527*p
    y = 295 + 143*index - (82-20*index)*bend + (75-24*index)*p*p
    return np.array((x, y))


POSE_TUBES = (("#344e64", "#14202e", "#78c8f9"), ("#584166", "#251932", "#cc9eed"))


@lru_cache(maxsize=1)
def pose_schematic_plate():
    plate = Image.new("RGB", (W*2,H*2), BLACK)
    draw = ImageDraw.Draw(plate)
    for index, (rim, fill, _) in enumerate(POSE_TUBES):
        points = [tuple(2*schematic_centerline(p,index)) for p in np.linspace(0,1,233)]
        for radius, color in ((100,rim),(95,fill)):
            for point in points:
                dot(draw, *point, radius, color)
        for j in range(0, len(points)-7, 12):
            draw.line((points[j],points[j+7]), fill=rim, width=3)
    return plate.resize((W,H),Image.Resampling.LANCZOS)


def pose_design(t):
    im = shot("ChromoPose / decoding schematic", "Follow the bends of each chromosome")
    im.paste(pose_schematic_plate().crop((0,151,748,570)),(0,151))
    draw = ImageDraw.Draw(im)
    local_move = camera(t, 2.0, 6.7)
    follow_flow = camera(t, 8.15, 13.25)
    for index, (_, _, accent) in enumerate(POSE_TUBES):
        attractor = .57 if index == 0 else .47
        for p, offset in ((.08, -24), (.19, 24), (.31, -28), (.44, 28),
                          (.62, -26), (.74, 29), (.85, -24), (.94, 24)):
            center = schematic_centerline(p,index)
            tangent = schematic_centerline(p+.01,index)-schematic_centerline(p-.01,index)
            normal = np.array((-tangent[1], tangent[0])) / np.linalg.norm(tangent)
            starting_point = center + offset*normal
            if local_move < .98:
                line(im, tuple(starting_point), tuple(center), tone(accent, .50*(1-local_move)), 2)
            moved = mix(p,attractor,follow_flow)
            if follow_flow:
                trail = mix(p,attractor,max(0,follow_flow-.16))
                path = [tuple(schematic_centerline(v,index)) for v in np.linspace(trail,moved,9)]
                draw.line(path,fill=tone(accent,.55),width=2)
            position = schematic_centerline(moved,index) + offset*(1-local_move)*normal
            dot(draw, *position, 5, accent)
        if follow_flow > 0:
            target = schematic_centerline(attractor,index)
            dot(draw, *target, 14, None, tone(accent, follow_flow), 2)

    line(im, (756,185), (756,584), HIRULE)
    rows = (
        (202, 0., "Learned cues", "Flow direction + skeleton distance"),
        (323, 1.4, "1. Distance-guided move", "Toward the local centerline"),
        (443, 7.55, "2. Integrate and group", "Along the learned flow"),
    )
    for y, start, title, detail in rows:
        p = ramp(t, start, start+.63)
        if p:
            text(im, (797,y), title, 28, tone(WHITE,p), True)
            text(im, (797,y+38), detail, 23, tone(MUTED,p))
    text(im, (98,582), "Centerlines and voxel paths shown schematically", 20, MUTED)
    caption(im, "A 3D backbone also predicts foreground, boundaries and the skeleton itself.")
    return im


def segmentation(t):
    im=shot("ChromoPose / tracking","Follow the same chromosome")
    frames=fig5_frames()
    index=min(len(frames)-1,int(clamp((t-.75)/7.4)*len(frames)))
    t_image=frames[index]
    split=camera(t,.75,2.4)
    focus=ramp(t,2.7,3.72)
    x=mix(640,328,split)
    width=round(mix(478,463,split))
    paste(im,fig5_layer(t_image,"dna",width),(x,377))
    appear = ramp(t,2.42,3.15)
    if appear:
        mask=fig5_layer(t_image,"mask",463)
        target=fig5_layer(t_image,"track",463)
        paste(im,light(blend(mask,target,focus),appear),(952,377))
    label(im,x,590,"Live DNA projection")
    label(im,952,590,"Tracked instance" if focus>.55 else "3D instances",appear,
          "#c0ddfa" if focus>.55 else WHITE)
    caption(im,"Tracking links the 3D instances in time. Their geometry feeds identity retrieval.")
    return im


@lru_cache(maxsize=4)
def isolated_views(width,height):
    reference=fitted(SOURCE/"tracked-chromosome-3d.webp",width,height)
    projected=fitted(SOURCE/"tracked-chromosome-projection.webp",width,height)
    if reference.size != projected.size:
        raise ValueError("3D view and 2D projection must share a pixel grid")
    return reference,projected


def lift_training(t):
    im=shot("ChromoLift / self-supervised training","Learn what projection loses")
    settle=camera(t,.18,1.86)
    if settle<1:
        paste(im,light(fig5_layer(fig5_frames()[-1],"track",463),1-settle),(952,377))
    x=mix(1046,355,settle)
    width=round(mix(132,414,settle))
    body,_=isolated_views(width,260)
    paste(im,body,(x,mix(343,385,settle)))
    project=camera(t,2.02,3.83)
    descriptors=ramp(t,4.8,6.45)
    if project:
        surface,mip=isolated_views(380,258)
        paste(im,light(reveal_registered(surface,mip,project),project*(1-descriptors)),(899,385))
        line(im,(640,191),(640,545),tone(HIRULE,project))
    label(im,355,579,"Measured 3D instance",settle,"#c0ddfa")
    label(im,899,579,"Projection of the same instance",project*(1-descriptors))
    if descriptors:
        for i, name in enumerate(("Volume","Elongation","Sphericity","Solidity","Tortuosity")):
            q=ramp(t,4.9+i*.23,5.75+i*.23)
            if q:
                y=230+i*58
                left=mix(716,700,q)
                line(im,(left,y-7),(1142,y-7),tone(HIRULE,q))
                text(im,(left,y),name,31,tone(WHITE,q))
        label(im,910,579,"3D supervision",descriptors)
    caption(im,"Project the same volume into 2D. Use its measured 3D shape as the training target.")
    return im


@lru_cache(maxsize=1)
def lifted_atlas():
    file=FIG4.parent/"fig5/Lifted2D_umap_embed.npz"
    with np.load(file,allow_pickle=False) as arrays:
        count=len(arrays["lifted_combined_y"])
    return fitted(STORY/"identity-atlas.webp",592,387),count


def atlas_and_match(t):
    im=shot("ChromoID / identity retrieval","Match live shape to the lifted atlas")
    moving=camera(t,.4,2.9)
    cloud,count=lifted_atlas()
    paste(im,isolated_views(280,185)[0],(mix(366,296,moving),302))
    label(im,296,412,"Measured live 3D shape",moving,"#c0ddfa",26)
    paste(im,reveal_registered(Image.new("RGB",cloud.size),cloud,moving,85),(892,359))
    line(im,(555,175),(555,610),tone(HIRULE,moving))
    label(im,892,563,f"{count:,} lifted reference instances",moving,WHITE,25)
    label(im,892,599,"UMAP of the reference atlas",moving,MUTED,21)
    for i,(name,delay) in enumerate((("Compare with class prototypes",1.35),
                                      ("Assign slots within the cell",5.9))):
        p=ramp(t,delay,delay+.7)
        y=500+i*52
        text(im,(86,y),str(i+1),22,tone("#c9b6fa",p),True)
        text(im,(125,y),name,24,tone(WHITE,p),True)
    caption(im,"ChromoID computes both steps from physical descriptors such as volume, elongation and sphericity.")
    return im


def live_identity(t, case="chr18"):
    im=shot("ChromoPhy / fluorescence comparison","Compare candidates with fluorescence")
    p=ramp(t,1.05,2.58)
    frame=min(7,(2 if case=="chr3" else 1)+int(t*.76))
    paste(im,live(case,frame,"candidates",451),(329,379))
    dna=live(case,frame,"dna",451)
    tag=live(case,frame,"dna-tag",451)
    paste(im,blend(dna,tag,p),(951,379))
    line(im,(640,174),(640,547),HIRULE)
    label(im,329,588,f"Chr{case[3:]} candidates")
    label(im,951,588,"DNA + fluorescent tag" if p>.40 else "Live DNA")
    caption(im,"Manually reviewed candidates are shown beside the measured Tag signal.")
    return im


def patient_division(t):
    im=shot("Patient-derived tumor cells","An unequal split of chromosome 14")
    frame=min(9,int(t*1.13))
    p=ramp(t,.6,2.4)
    source=picture(SOURCE/f"patient-{frame}.webp")
    scale=source.width/420
    crop=tuple(round(v*scale) for v in (43,45,363,375))
    image=source.crop(crop)
    size=round(mix(421,471,camera(t,1.,6.5)))
    image=blend(image.convert("L").convert("RGB"),image,p)
    width=round(size*image.width/image.height)
    paste(im,image.resize((width,size),Image.Resampling.LANCZOS),(640,382))
    reveal=ramp(t,8.4,9.72)
    if reveal:
        green=PALETTE[13]
        for x,a,(u,v),count in ((191,311,(127,109),2),(1089,970,(271,210),4)):
            ink=tone(green,reveal)
            end=(640-width/2+(u*scale-crop[0])*width/image.width,382-size/2+(v*scale-crop[1])*size/image.height)
            text(im,(x,305),str(count),75,ink,True,"ma")
            label(im,x,403,"Chr14 copies",reveal,WHITE,24)
            offset=-54 if count==2 else 54
            linked(im,((a,368),(a+47*(1 if count==2 else -1),368),(end[0]+offset,end[1]),end),
                   tone(green,.62),reveal)
    caption(im,"Daughter-cell copy counts and Chr14 identities are inferred from live imaging.")
    return im


@lru_cache(maxsize=1)
def microscope_icons():
    paths=[SOURCE/f"{name}.webp" for name in ("micromanipulator","plate")]
    icons=[]
    for path,height in zip(paths,(245,250)):
        with Image.open(path) as input:
            image=input.convert("RGBA")
        image=image.crop(image.getbbox())
        icons.append(image.resize((round(height*image.width/image.height),height),Image.Resampling.LANCZOS))
    return icons


def place_alpha(im,layer,top_left,opacity=1.):
    px=tuple(round(v) for v in top_left)
    mask=layer.getchannel("A")
    im.paste(layer,px,mask if opacity>=1 else mask.point(lambda x:round(x*clamp(opacity))))


def micromanipulation(t):
    im=shot("Clonal validation / workflow","Isolate a daughter cell")
    instrument,well=microscope_icons()
    d=ImageDraw.Draw(im)
    d.ellipse((106,295,498,497),outline=AXIS,width=2)
    for x,y in ((187,400),(268,357),(291,444),(418,349),(428,437)):
        dot(d,x,y,17,None,"#3c4754",2)
        dot(d,x,y,4,MUTED)
    green=PALETTE[13]
    pickup=camera(t,.1,1.76)
    transfer=camera(t,2.12,4.38)
    target=(374,405)
    dot(d,*target,20,None,tone(green,1-.78*transfer),2)
    dot(d,*target,5,tone(green,1-transfer))
    tip=(mix(422,target[0],pickup),mix(333,target[1],pickup))
    place_alpha(im,instrument,(tip[0]-.96*instrument.width,tip[1]-.94*instrument.height),1-.68*transfer)
    destination=(round(953-well.width/2),round(391-well.height/2))
    place_alpha(im,well,destination,ramp(t,1.5,2.79))
    final=(923,381)
    if transfer:
        moving=(mix(target[0],final[0],transfer),mix(target[1],final[1],transfer)-58*np.sin(np.pi*transfer))
        if transfer<1:
            dot(d,*moving,6,green)
        dot(d,*final,12,None,tone(green,ramp(t,3.86,4.6)),2)
    label(im,301,576,"Image-guided isolation")
    label(im,953,576,"Single-cell expansion",ramp(t,1.6,2.8))
    caption(im,"Experiment schematic. Isolated daughter cells were grown as separate clones.")
    return im


def qpcr_chapter(t):
    im=shot("Clonal validation / genomic qPCR","Copy-number differences persist")
    p=camera(t,.15,1.)
    plate(im,SOURCE/"qpcr.webp",(mix(904,896,p),380),(486,435),p)
    text(im,(107,276),"Chr14 dosage",44,WHITE,True)
    text(im,(110,351),"Measured at chr14q24.1",27,MUTED)
    text(im,(110,400),"Across serial passages",27,MUTED)
    line(im,(578,186),(578,582),HIRULE)
    caption(im,"Single-cell clones retained distinct Chr14 dosage over successive passages.")
    return im


def assay_evidence(t):
    im=shot("Clonal validation / ex vivo assays","From Chr14 dosage to AKT")
    text(im,(105,182),"AKT signaling",31,WHITE,True)
    text(im,(701,182),"Drug response",31,WHITE,True)
    p=ramp(t,.3,1.25)
    q=ramp(t,.7,1.75)
    for y,name,title in ((326,"pakt","p-AKT1"),(435,"akt","AKT1")):
        text(im,(105,y-54),title,23,tone(MUTED,p))
        plate(im,SOURCE/f"{name}.webp",(343,y),(471,71),p)
    for x,name in ((149,"2-copy"),(271,"2-copy"),(389,"3-copy"),(505,"4-copy")):
        label(im,x,483,name,p,MUTED,21)
    line(im,(106,528),(199,528),tone(HIRULE,p))
    line(im,(215,528),(575,528),tone(HIRULE,p))
    label(im,149,539,"Patient 2",p,MUTED,20)
    label(im,389,539,"Patient 1",p,MUTED,20)
    line(im,(640,184),(640,578),HIRULE)
    plate(im,SOURCE/"response.webp",(942,371),(486,292),q)
    label(im,961,530,"Drug concentration (log10 mol/L)",q,MUTED,19)
    plate(im,SOURCE/"response-legend.webp",(943,589),(495,56),q)
    caption(im,"Higher-copy clones showed stronger AKT signaling and greater sensitivity to capivasertib ex vivo.")
    return im


def treatment_followup(t):
    im = shot("Clinical follow-up · Manuscript Fig. 7", "Liver metastases regressed during therapy")
    color = PAPER_COLORS["14"]
    text(im,(90,178),"Sum of target-lesion diameters",29,WHITE,True)
    x0, x1, y0, y1 = 136, 678, 265, 532
    for value in (0,-20,-40,-60):
        y = y0-value*(y1-y0)/60
        line(im,(x0,y),(x1,y),HIRULE)
        text(im,(x0-15,y),f"{value}%",20,MUTED,anchor="rm")
    line(im,(x0,y0),(x0,y1),AXIS)
    text(im,(x0,y1+15),"Baseline",21,MUTED,anchor="ma")
    text(im,(x0+(x1-x0)*8/18,y1+15),"Week 8",21,MUTED,anchor="ma")
    text(im,(x1,y1+15),"Week 18",21,MUTED,anchor="ma")
    # Manuscript v15 Results reports the target-lesion sum at baseline and
    # weeks 8 and 18. Only measured visits are marked.
    points = [(x0+(x1-x0)*week/18,y0-value*(y1-y0)/60)
              for week,value in ((0,0),(8,-31.8),(18,-47.7))]
    starts = (0,2.1,5.0)
    for i, ((x,y),start) in enumerate(zip(points,starts)):
        if i:
            p = camera(t,start-.9,start+.15)
            px, py = points[i-1]
            line(im,(px,py),(mix(px,x,p),mix(py,y,p)),color,3)
        show = ramp(t,start,start+.45)
        if show > 0:
            dot(ImageDraw.Draw(im),x,y,5.5,tone(color,show))
            if i:
                text(im,(x,y-14),("−31.8%","−47.7%")[i-1],25,tone(color,show),True,"ms")
    line(im,(756,184),(756,560),HIRULE)
    p = ramp(t,.28,1.0)
    text(im,(800,212),"Capivasertib-based therapy",29,tone(WHITE,p),True)
    text(im,(800,267),"Combined with endocrine therapy",23,tone(MUTED,p))
    text(im,(800,327),"Treatment selected independently",23,tone(MUTED,p))
    text(im,(800,357),"by the clinical team",23,tone(MUTED,p))
    q = ramp(t,5.5,6.25)
    text(im,(800,441),"Partial response",42,tone(color,q),True)
    text(im,(800,503),"No new metastatic lesions observed",23,tone(MUTED,q))
    caption(im,"CT measurements in the patient with the Chr14 trajectory, relative to pretreatment baseline.")
    return im


def broader_recovery(t):
    im = shot("Clinical follow-up · Manuscript Fig. 7", "Clinical measures also improved")
    rows = [
        ("Tumor markers","Serum CA15-3 and CEA declined."),
        ("Liver biochemistry","Markers including AST, ALT and bilirubin improved."),
        ("Daily function and symptoms","Performance status improved, with less liver pain and fatigue."),
    ]
    for i,(heading,description) in enumerate(rows):
        p = ramp(t,.4+i*1.6,1.+i*1.6)
        y = 206+i*121
        line(im,(78,y-18),(1201,y-18),tone(HIRULE,p))
        text(im,(83,y),heading,34,tone(WHITE,p),True)
        text(im,(84,y+51),description,27,tone(MUTED,p))
    caption(im,"Clinical follow-up during capivasertib-based therapy in the same patient.")
    return im


def closing(t):
    im = shot("ChromoWorld", "From chromosome dynamics to therapy hypotheses")
    color = PAPER_COLORS["14"]
    items = [
        ("Detect","Chr14 segregation"),
        ("Test","Response to AKT inhibition"),
        ("Follow","Clinical response"),
    ]
    for i,(verb,detail) in enumerate(items):
        x = 81+i*386
        p = ramp(t,.15+i*.5,.85+i*.5)
        line(im,(x,264),(x+335,264),tone(color,p),2)
        text(im,(x,294),verb,42,tone(WHITE,p),True)
        text(im,(x,353),detail,27,tone(MUTED,p))
    p = ramp(t,2.05,2.85)
    text(im,(81,495),"Live-cell imaging could help prioritize therapies for clinical validation.",30,tone(WHITE,p))
    caption(im,"Single-patient clinical follow-up.",p)
    return im


STORY = SOURCE / "story"


def karyotype_open(t):
    im = shot("Karyotype analysis", "What can a karyotype tell us?")
    plate(im, STORY / "karyotype.webp", (425, 372), (680, 421))
    d = ImageDraw.Draw(im)
    p = camera(t, 1.1, 2.25)
    if p:
        d.rounded_rectangle((86, 161, 166, 295), radius=9, outline=tone(ACCENT, p), width=2)
        line(im, (797, 183), (797, 565), tone(HIRULE, p))
    for i, (title, detail) in enumerate((("Identity", "Which chromosome?"),
                                           ("Copy number", "How many copies?"),
                                           ("Structure", "Has it changed?"))):
        q = ramp(t, 1.4+i*1.72, 2.2+i*1.72)
        y = 213+i*113
        text(im, (849, y), title, 34, tone(WHITE, q), True)
        text(im, (850, y+46), detail, 25, tone(MUTED, q))
    caption(im, "G-banded metaphase, arranged by reference chromosome annotations.")
    return im


def living_question(t):
    im = shot("Live mitosis", "Where do the chromosomes go at division?")
    frame_idx = min(18, round(mix(1, 18, camera(t, .15, 6.3))))
    paste(im, live("mitosis", frame_idx, "dna", 482), (640, 365))
    p = ramp(t, 5.2, 6.3)
    label(im, 640, 595, "DNA fluorescence during mitosis", p, WHITE, 27)
    caption(im, "The DNA signal reveals movement. Following each chromosome also requires an identity.")
    return im


def pipeline_node(im, x, y, title, detail, progress, accent=False):
    d = ImageDraw.Draw(im)
    key = "#cab6fa" if accent else MUTED
    d.rounded_rectangle((x, y, x+174, y+79), radius=11,
                        fill=tone("#17121e" if accent else "#101218", progress),
                        outline=tone("#6c588e" if accent else HIRULE, progress), width=2)
    title_size = min(23, int(23*156/font(23, True).getlength(title)))
    text(im, (x+87, y+26), title, title_size, tone(WHITE, progress), True, "mm")
    text(im, (x+87, y+56), detail, 16, tone(key, progress), False, "mm")


def architecture(t, emphasis=False):
    im = shot("ChromoWorld / system overview", "How ChromoWorld reads a live-cell karyotype")
    xs = (78, 284, 490, 696)
    upper = (("G-banding / FISH", "Fixed-cell images"), ("ChromoPix", "2D identity & masks"),
             ("ChromoLift", "2D to 3D features"), ("Reference atlas", "Labeled 3D shapes"))
    lower = (("Live 3D + time", "DNA microscopy"), ("ChromoPose", "3D instances"),
             ("Tracking", "Link across frames"), ("Live query", "Measured 3D shape"))
    stages = ((upper, 255, 1.25, "BUILD THE IDENTITY REFERENCE"),
              (lower, 442, 9.0, "FOLLOW CHROMOSOMES IN LIVING CELLS"))
    for nodes, y, delay, heading in stages:
        status = .46 if emphasis and y==255 else 1
        p = status if emphasis else ramp(t, delay-.65, delay+.2)
        text(im, (284, y-58), heading, 21, tone(MUTED, p), True)
        if y == 255:
            plate(im,SOURCE/"banding-dna.webp",(120,y-53),(84,84),p)
            plate(im,SOURCE/"example-dna.webp",(211,y-53),(84,84),p)
        else:
            paste(im,light(live("mitosis",9,"dna",88),p),(165,y-54))
        for i, (title, detail) in enumerate(nodes):
            p = status if emphasis else ramp(t, delay+i*.58, delay+i*.58+.72)
            pipeline_node(im, xs[i], y, title, detail, p, title in ("ChromoPix","ChromoLift","ChromoPose"))
            if i and p:
                linked(im, ((xs[i-1]+174,y+39),(xs[i]-4,y+39)), tone(AXIS,p),p)
    p = 1 if emphasis else ramp(t, 12.2, 13.5)
    for y in (294,481):
        linked(im, ((870,y),(903,y),(903,389),(932,389)), tone("#ab96ca",p), p, marker=False)
    pipeline_node(im, 933, 350, "ChromoID", "Shape-based retrieval", p, True)
    line(im, (1020,429), (1020,550), tone("#ab96ca",p), 2)
    text(im, (1020,576), "Identity + trajectory", 22, tone(WHITE,p), True, "mm")
    text(im, (78,575), "ChromoPhy", 29, tone("#cab6fa",p), True)
    text(im, (257,583), "Lift fixed-cell references and match live instances in 3D", 22, tone(MUTED,p))
    caption(im, "The live branch provides measured shapes and links them in time." if emphasis else
            "The 3D atlas and the live-cell measurements meet in the same physical feature space.")
    return im


def cross_domain(t):
    if t < 5.8:
        im = shot("Fig. 2B / held-out evaluation", "The same models face a different preparation")
        plate(im,STORY/"banding-transfer-metrics.webp",(454,378),(736,422))
        text(im,(882,273),"One training state",29,WHITE,True)
        text(im,(883,320),"G-banding only",23,MUTED)
        text(im,(882,395),"Two test states",29,WHITE,True)
        text(im,(883,440),"G-banding and FISH",23,MUTED)
        caption(im,"Fig. 2B. Each line follows a metric for the same trained model across two preparations.")
        return im
    im = shot("Fig. 2D / cross-preparation test", "Can a model trained on bands read FISH?")
    text(im, (77,153), "Three models trained on G-banding. One held-out FISH cell.", 25, MUTED)
    models = (("mask-rcnn", "Mask R-CNN", "No detections"),
              ("v11", "YOLOv11-Seg", "Masks with incorrect IDs"),
              ("v12", "YOLOv12-Seg", "Masks with incorrect IDs"))
    for x, (name, title, detail) in zip((252,640,1028),models):
        plate(im, STORY/f"banding-only-{name}.webp", (x,386), (367,367))
        label(im, x, 566, title, size=27)
        text(im, (x,604), detail, 22, tone(MUTED,ramp(t,7.2,8.25)), anchor="ma")
    caption(im,"Archived predictions overlaid on the same microscope acquisition.")
    return im


DETAIL_PAD = 14


@lru_cache(maxsize=1)
def invariant_example():
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    # Illustrative morphology-preserving view from the same archived Chr1 mask.
    dna, _, _, bbox = microscopy()
    x0,y0,x1,y1 = bbox
    view = (x0-DETAIL_PAD,y0-DETAIL_PAD,x1+DETAIL_PAD,y1+DETAIL_PAD)
    scale = 4
    texture = dna.crop(view).resize(((view[2]-view[0])*scale,(view[3]-view[1])*scale),Image.Resampling.BICUBIC)
    points = (np.array(json.loads((SOURCE/"example-outline.json").read_text()))-[view[0],view[1]])*scale
    mask = Image.new("L",texture.size)
    ImageDraw.Draw(mask).polygon(list(map(tuple,points)),fill=255)
    tint = blend(texture,Image.new("RGB",texture.size,ACCENT),.24)
    original = Image.composite(tint,Image.new("RGB",texture.size),mask)
    ImageDraw.Draw(original).line([tuple(p) for p in points]+[tuple(points[0])],fill=ACCENT,
                                  width=scale+1,joint="curve")
    binary = np.asarray(mask)>0
    distance = distance_transform_edt(binary)
    core = gaussian_filter(distance,1.6*scale)
    glow = .34*gaussian_filter(binary.astype(float),1.2*scale)+.66*core/max(1e-6,core.max())
    transformed = Image.fromarray(np.uint8(np.clip(glow,0,1)*255)).convert("RGB")
    return original, transformed


def invariant_learning(t):
    im = shot("ChromoPix / mixed supervision", "One chromosome, two training views")
    clean, changed = invariant_example()
    for i, (view, label_name) in enumerate(((clean,"Reference view"),(changed,"Shape-preserving view"))):
        p = ramp(t, .35+i*1.05, 1.15+i*1.05)
        x = 306 + i*666
        copy = view.copy()
        copy.thumbnail((380,288),Image.Resampling.LANCZOS)
        paste(im,light(copy,p),(x,329))
        label(im,x,476,label_name,p,WHITE,26)
    p = ramp(t, 5.0, 6.1)
    for x in (306,972):
        linked(im, ((x,514),(x,545),(640,545)),tone(ACCENT,p),p,marker=False)
    label(im,640,568,"Align the features of the same chromosome",p,ACCENT,29)
    caption(im, "Standard identity labels supervise the masks. Paired views teach feature invariance.")
    return im


def pix_cell_lines(t):
    im=shot("ChromoPix / Fig. 3A", "Chromosome identity across four cell lines")
    lines=(("hff","HFF","0.670"),("mcf","MCF-10F","0.700"),
           ("rpe","RPE1","0.716"),("mrc","MRC-5","0.661"))
    for i,(name,title,f1) in enumerate(lines):
        x=207+288*i
        p=ramp(t,.15+.38*i,.8+.38*i)
        plate(im,STORY/f"pix-line-{name}.webp",(x,360),(252,252),p)
        label(im,x,508,title,p,WHITE,27)
        text(im,(x,549),f"F1  {f1}",22,tone(MUTED,p),anchor="ma")
    caption(im,"Held-out F1 ranged from 0.661 to 0.716 across the four tested cell lines.")
    return im


def pix_scale(t):
    im=shot("ChromoPix / Fig. 3B", "How much does image scale matter?")
    for x,name,title in ((200,"small","Small field of view"),(485,"large","Large field of view")):
        plate(im,STORY/f"pix-scale-{name}.webp",(x,355),(250,250))
        label(im,x,505,title,size=24)
    p=ramp(t,1.3,2.3)
    plate(im,STORY/"pix-scale-metrics.webp",(926,351),(530,285),p)
    line(im,(635,181),(635,573),tone(HIRULE,p))
    label(im,926,490,"0.75× to 1.25×",p,"#c9b6fa",30)
    text(im,(926,530),"Moderate rescaling",22,tone(MUTED,p),anchor="ma")
    caption(im,"Moderate changes preserve performance better than the most extreme scales tested.")
    return im


def pix_perturbations(t):
    im=shot("ChromoPix / Fig. 3C", "Test the limits of ChromoPix")
    variants=(("original","Original"),("rotate","Rotate + flip"),
              ("crowd","Crowding"),("blur","Optical blur"))
    for i,(name,title) in enumerate(variants):
        x=207+288*i
        p=ramp(t,.15+.38*i,.8+.38*i)
        plate(im,STORY/f"pix-perturb-{name}.webp",(x,360),(252,252),p)
        label(im,x,508,title,p,WHITE,26)
    p=ramp(t,2.5,3.45)
    text(im,(495,550),"Small change",22,tone("#c9b6fa",p),True,"ma")
    caption(im,"Rotation and flipping had little effect.")
    return im


@lru_cache(maxsize=1)
def pix_perturbation_charts():
    im=shot("ChromoPix / Fig. 3C · held-out test set", "Performance declines as the task gets harder")
    values=json.loads((STORY/"pix-perturbation-bars.json").read_text())
    panels=(
        (74,"cp","Copy & Paste (CP)","Added chromosomes","crowd",
         ("+25%","+50%","+75%","+100%"),"Near-linear decline"),
        (660,"psf","Point Spread Function (PSF)","Blur kernel size","blur",
         ("15 px","30 px","45 px","60 px"),"Larger drops with stronger blur"),
    )
    colors=("#e7eaf2","#8b94a5","#a3cbe6","#66abc9","#9d88df","#cf9ee5")
    metrics=(("Precision","@0.5"),("F1",""),("Recall","@0.5"),
             ("Dice",""),("mAP",""),("IoU",""))
    draw=ImageDraw.Draw(im)
    for x,name,title,detail,example,levels,takeaway in panels:
        text(im,(x,158),title,28,WHITE,True)
        text(im,(x,198),detail,20,MUTED)
        plate(im,STORY/f"pix-perturb-{example}.webp",(x+501,188),(91,91))
        for offset,color,level in zip((0,85,206,284,362,440),colors,("Original","Rotate + flip",*levels)):
            draw.rectangle((x+offset,249,x+offset+10,259),fill=color)
            text(im,(x+offset+15,243),level,16,MUTED)
        left, right, top, base = x+45,x+546,293,516
        for tick in (0,.2,.4,.6,.8):
            y=round(mix(base,top,tick/.8))
            text(im,(left-11,y),f"{tick:.1f}",17,MUTED,anchor="rm")
            line(im,(left,y),(right,y),AXIS if tick==0 else HIRULE)
        for j,(metric, qualifier) in enumerate(metrics):
            center=left+(right-left)*(j+.5)/len(metrics)
            for i,(scores,color) in enumerate(zip(values[name],colors)):
                start=round(center-31+i*10.5)
                y=round(mix(base,top,scores[j]/.8))
                draw.rectangle((start,y,start+8,base-1),fill=color)
            text(im,(center,529),metric,18,WHITE,True,"ma")
            if qualifier:
                text(im,(center,552),qualifier,16,MUTED,anchor="ma")
        label(im,(left+right)/2,594,takeaway,1,"#d5c2ff",22)
    line(im,(642,159),(642,617),HIRULE)
    caption(im,"Scores decline predictably as copy-paste crowding or optical blur increases.")
    return im


def pix_virtual_review(t):
    im=shot("ChromoPix / Fig. 3D", "Virtual FISH flags a discordant instance")
    p=ramp(t,1.45,2.55)
    plate(im,SOURCE/"fish-reference.webp",(259,379),(343,343))
    plate(im,SOURCE/"fish-prediction.webp",(643,379),(343,343),p)
    label(im,259,568,"FISH annotation",size=24)
    label(im,643,568,"ChromoPix prediction",p,WHITE,24)
    q=ramp(t,3.1,4.1)
    if q:
        # Center of the discordant FISH polygon in the original 1024-pixel field.
        for x,col in ((259,PALETTE[0]),(643,PALETTE[16])):
            dot(ImageDraw.Draw(im),x+(.856469-.5)*343,379+(.530997-.5)*343,18,None,tone(col,q),2)
    plate(im,STORY/"pix-review-detail.webp",(1019,343),(175,172),q)
    text(im,(1019,442),"FISH annotation  Chr1",23,tone(MUTED,q),anchor="ma")
    text(im,(1019,486),"ChromoPix  Chr17",25,tone(PALETTE[16],q),True,"ma")
    caption(im,"A morphology-based discrepancy calls for review of the assignment.")
    return im


def pix_to_morphology(t):
    im=shot("ChromoPix / morphology", "Follow one chromosome into the shape map")
    p=camera(t,.25,1.65)
    reference=Image.new("RGB",(W,H),BLACK)
    field_view(reference,0.,0.,(640,382),452,ramp(t,1.35,3.05))
    plate(im,SOURCE/"fish-prediction.webp",(640,382),(452,452))
    if p:
        im.paste(blend(im.crop((414,156,866,608)),reference.crop((414,156,866,608)),p),(414,156))
    caption(im,"The labeled mask provides a boundary for measuring shape.")
    return im


def projection_failure(t):
    im = shot("Fig. 5A–B / cross-dimensional test", "Flattening the cell can collapse identities")
    for i, (filename, name) in enumerate((("mip-raw","Live DNA projection"),
                                         ("mip-naive","Direct projection IDs"),
                                         ("mip-optimal","Isolated, optimal projection IDs"))):
        p = ramp(t, i*.75+.1, i*.75+.7)
        x = 248 + i*392
        plate(im, STORY/(filename+".webp"), (x,374),(320,357),p)
        label(im,x,586,name,p,WHITE,23)
    caption(im, "Optimizing the projection partially restores identity structure; the remaining gap is in 3D.")
    return im


@lru_cache(maxsize=1)
def paper_centroids():
    return json.loads((STORY/"identity-centers.json").read_text())


def retrieval_comparison(t):
    im=shot("Fig. 5A–E / live-cell morphology", "How identity order returns in 3D")
    types=(("direct","Direct projection","Centroids collapse"),
           ("optimal","Optimal projection","Partial separation"),
           ("chromoid","ChromoLift + ChromoID","Clearer class order"))
    classes=(*range(12),22,*range(12,22),23)
    for i,(name,title,summary) in enumerate(types):
        x=252+388*i
        p=ramp(t,.15+.35*i,.82+.35*i)
        flow=ramp(t,2.5+i*.78,3.65+i*.78)
        cloud=fitted(STORY/f"identity-{name}.webp",354,354)
        top=np.array((x-cloud.width/2,391-cloud.height/2))
        paste(im,light(cloud,p*(1-.52*flow)),(x,391))
        label(im,x,167,title,p,"#d5c2ff" if i==2 else WHITE,26)
        label(im,x,584,summary,flow,MUTED,22)
        if not flow:
            continue
        points=np.array(paper_centroids()[name])*cloud.size+top
        if i:
            for j,(a,b) in enumerate(zip(points,points[1:])):
                line(im,tuple(a),tuple(b),tone("#08090d",flow),5)
                line(im,tuple(a),tuple(b),tone(PALETTE[classes[j]],flow),2)
        d=ImageDraw.Draw(im)
        for pos,index in zip(points,classes):
            dot(d,*pos,5,tone(PALETTE[index],flow),tone(WHITE,flow),1)
        if i==2:
            for index,dx,dy in ((0,-15,15),(2,14,12),(7,-10,-19),
                                (12,-4,-18),(17,1,-19),(21,3,-17),(23,18,-12)):
                px,py=points[classes.index(index)]
                text(im,(px+dx,py+dy),CLASS_NAMES[index],17,tone(WHITE,flow),True,"mm")
    caption(im,"Dots are the paper’s chromosome-class centroids. Lines connect them in identity order.")
    return im


def batch_lift(t):
    im = shot("ChromoLift / flow matching", "Lift the descriptors, then build the atlas")
    reference, projection = isolated_views(170,180)
    paste(im,projection,(224,303))
    paste(im,reference,(1056,303))
    label(im,224,447,"2D projections")
    label(im,1056,447,"3D shape descriptors")
    p = camera(t,.8,5.6)
    d=ImageDraw.Draw(im)
    for i in range(11):
        x=400+(i%4)*34
        y=240+(i//4)*55
        line(im,(x,y),(x+380,y+17*np.sin(i)),tone("#6f5a9d",.35),1)
        blend_pos=clamp(p-(i%4)*.047)
        dot(d,mix(x,x+380,blend_pos),mix(y,y+17*np.sin(i),blend_pos),4,PALETTE[i*2])
    label(im,638,447,"Flow matching",1,WHITE,28)
    caption(im,"Self-supervised projection pairs teach the mapping. Compact feature vectors can be lifted in batches.")
    return im


# A short dip separates changes of specimen or scale. Aligned views and
# object-to-point handoffs cut in place so no pair of headlines overlaps.
SCENES = [
    (0,karyotype_open,False),
    (12,living_question,False),
    (22,architecture,True),
    (43,preparation_states,True),
    (51,cross_domain,True),
    (65,lambda t: virtual_fish(t*.92),True),
    (77,lambda t: invariant_learning(t*.95),False),
    (94,pix_cell_lines,True),
    (102,pix_scale,False),
    (110,pix_perturbations,False),
    (114,lambda t: pix_perturbation_charts(),False),
    (120,pix_virtual_review,True),
    (129,pix_to_morphology,False),
    (134,lambda t: morphology(t*1.22),False),
    (141,to_one_point,False),
    (146,lambda t: population(t*1.15),False),
    (154,lambda t: native_comparison(t+4.2),False),
    (157,centroid_comparison,False),
    (163,projection_failure,True),
    (170,lambda t: architecture(t,True),False),
    (173,pose_comparison,True),
    (185,pose_design,True),
    (201,segmentation,False),
    (211,lambda t: lift_training(t*.91),False),
    (222,batch_lift,False),
    (233,atlas_and_match,True),
    (244,retrieval_comparison,False),
    (255,lambda t: live_identity(t,"chr3"),True),
    (262,live_identity,False),
    (269,patient_division,True),
    (282,micromanipulation,True),
    (288,qpcr_chapter,False),
    (294,assay_evidence,False),
    (302,lambda t: treatment_followup(t*.89),True),
    (315,broader_recovery,False),
    (325,closing,True),
]


def render(t):
    for i in range(len(SCENES)-1,-1,-1):
        start,scene,dip=SCENES[i]
        if t<start:
            continue
        local=t-start
        visibility=ramp(local,0,.23) if dip else 1.
        if i+1<len(SCENES) and SCENES[i+1][2]:
            following=SCENES[i+1][0]
            visibility=min(visibility,1-ramp(t,following-.20,following))
        if i == len(SCENES)-1:
            visibility = min(visibility,1-ramp(t,DURATION-.75,DURATION))
        return light(scene(local),visibility)
    raise ValueError(f"Negative video time {t}")


def write_video(path, times, preview=False, credits=None, audio=None):
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    frame_size=(640,360) if preview else (W,H)
    cmd=[ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo",
         "-pixel_format","rgb24","-video_size",f"{frame_size[0]}x{frame_size[1]}",
         "-framerate",str(FPS),"-i","pipe:0"]
    if credits:
        cmd += ["-i",str(credits),"-i",str(audio),"-filter_complex",
                "[0:v]setsar=1,format=yuv420p,setpts=PTS-STARTPTS[film];"
                "[1:v]setsar=1,format=yuv420p,setpts=PTS-STARTPTS[credit];"
                "[film][credit]concat=n=2:v=1:a=0,pad=iw:ih+112:0:0:black[v]",
                "-map","[v]","-map","2:a","-t",str(DURATION+5),"-c:a","aac","-b:a","192k",
                "-metadata","comment=Music: A Kind Of Hope by Scott Buckley (CC BY 4.0)."]
    else:
        cmd += ["-an"]
    cmd += ["-c:v","libx264","-preset","medium" if credits else "fast","-threads","4",
            "-crf","14" if credits else "19","-pix_fmt","yuv420p","-movflags","+faststart",str(path)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for i,t in enumerate(times):
            if i%(10*FPS)==0:
                print(f"Rendering at {t:.1f}s",flush=True)
            im=render(t)
            if preview:
                im=im.resize(frame_size,Image.Resampling.LANCZOS)
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Introduction video encoding failed")
    subprocess.run([ffmpeg,"-v","error","-xerror","-i",str(path),"-f","null","-"],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",type=Path,help="Render a short review of the revised transitions")
    parser.add_argument("--output-dir",type=Path,default=ASSETS,help="Destination for the silent movie and poster")
    parser.add_argument("--credits",type=Path,help="Five-second, 1280×720 credits composition")
    parser.add_argument("--audio-master",type=Path,help="Mastered narration and music WAV")
    args=parser.parse_args()
    if bool(args.credits) != bool(args.audio_master):
        parser.error("Pass both --credits and --audio-master for the final film.")
    if args.preview:
        ranges=((21.,43.),(93.,164.),(172.,211.),(221.,256.))
        times=(i/FPS for a,b in ranges for i in range(round(a*FPS),round(b*FPS)))
        write_video(args.preview,times,preview=True)
        return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as staging:
        poster=Path(staging)/"intro-poster.webp"
        image=Image.new("RGB",(W,H),BLACK)
        plate(image,SOURCE/"fish-prediction.webp",(322,353),(550,550))
        plate(image,SOURCE/"patient-6.webp",(950,352),(552,552))
        label(image,322,598,"Fixed cell",color=WHITE,size=34)
        label(image,950,598,"Dividing cell",color=WHITE,size=34)
        if args.credits:
            padded=Image.new("RGB",(W,H+112),BLACK)
            padded.paste(image,(0,0))
            image=padded
        image.save(poster,"WEBP",lossless=True)
        video=Path(staging)/"intro.mp4"
        write_video(video,(i/FPS for i in range(DURATION*FPS)),credits=args.credits,audio=args.audio_master)
        if args.credits:
            stream = Path(staging)/"intro-stream.webm"
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y",
                            "-i",str(video),"-map","0:v:0","-map","0:a:0","-vf","scale=960:624:flags=lanczos",
                            "-c:v","libvpx-vp9","-row-mt","1","-tile-columns","1","-cpu-used","5","-threads","6",
                            "-crf","33","-b:v","0","-g","72","-pix_fmt","yuv420p","-c:a","libopus","-b:a","72k",
                            "-reserve_index_space","8192",str(stream)],check=True)
            stream.replace(args.output_dir/stream.name)
        video.replace(args.output_dir/video.name)
        poster.replace(args.output_dir/poster.name)


if __name__ == "__main__":
    main()
