"""Render the homepage explainer from microscopy and archived morphology embeddings."""
from functools import lru_cache
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SITE = Path(__file__).resolve().parents[1]
ASSETS = SITE / "public/assets"
SOURCE = SITE / "src/data/overview"
FIG4 = SITE.parent / "release/Artifacts/fig4"
W, H, FPS, DURATION = 1280, 720, 24, 168
BLACK, WHITE, MUTED = "#000000", "#f8faff", "#c1c6d4"
GRID, AXIS, HIRULE = "#151a20", "#626b77", "#262a32"
FONT = Path("/System/Library/Fonts/Supplemental")
CLASS_NAMES = [str(i) for i in range(1, 23)] + ["X", "Y"]
PAPER_COLORS = json.loads((SOURCE.parent / "palette.json").read_text())
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
    return picture(ASSETS / f"live/{case}/{t}-{layer}.webp", width)


def fish(layer, width):
    return picture(ASSETS / f"fish/hff-{layer}.webp", width)


def field_layers():
    dna = picture(SOURCE / "example-dna.webp")
    mask = Image.open(SOURCE / "example-mask.png").convert("L")
    mask = mask.resize(dna.size, Image.Resampling.NEAREST)
    rim = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)),
                              mask.filter(ImageFilter.MinFilter(3)))
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
    side = mix(768, 172, zoom)
    cx = mix(384, (left + right) / 2, zoom)
    cy = mix(384, (top + bottom) / 2, zoom)
    crop = tuple(round(c) for c in (cx-side/2, cy-side/2, cx+side/2, cy+side/2))
    field = blend(blend(dna, marked, highlight), isolated, isolation)
    paste(im, field.crop(crop).resize((width, width), Image.Resampling.BICUBIC), center)
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

def opening(t):
    im = shot("Live-cell imaging", "Which chromosome goes where?")
    move = camera(t,2.6,5.2)
    frame = min(18, round(mix(1,18, clamp(t/8.7))))
    x = mix(641,365,move)
    width = round(mix(480,488,move))
    paste(im,live("mitosis",frame,"dna",width),(x,381))
    for i,(head,detail) in enumerate((
        ("Biology",("No general method labels chromosome",
                    "identities stably through live mitosis.")),
        ("Imaging",("Chromosomes overlap in three dimensions.",)),
        ("Computation",("Shape and position change during division.",)),
    )):
        q = ramp(t,3.2+i*1.02,4.2+i*1.02)
        if q:
            y = 224+i*109
            left = mix(712,680,q)
            line(im,(left,y-12),(1161,y-12),tone(HIRULE,q))
            text(im,(left,y),head,32,tone(WHITE,q),True)
            for j, row in enumerate(detail):
                text(im,(left,y+43+j*26),row,23,tone(MUTED,q))
    caption(im,"A live DNA projection follows one cell into two daughters.")
    return im


def banding(t):
    if t >= 8.1:
        p = camera(t,8.1,9.65)
        im = shot("Fixed cells / G-banding","When the bands disappear")
        plate(im,SOURCE/"banded-chr1.webp",(mix(741,319,p),383),
              (round(mix(244,315,p)),round(mix(294,320,p))))
        if p:
            field_view(im,1,1,(949,379),round(mix(290,459,p)),p)
        label(im,319,583,"G-banded Chr1",p)
        label(im,949,583,"Rehydrated Chr1",p)
        caption(im,"Models trained only on G-banding lose accuracy on rehydrated cells.",p)
        return im
    im = shot("Fixed cells / G-banding","A barcode for each chromosome")
    p = camera(t,1.1,3.8)
    x = mix(640,305,p)
    w = round(mix(478,425,p))
    paste(im,picture(SOURCE/"banding-dna.webp",w),(x,376))
    for i,(chr_number,u,v,dest) in enumerate(((1,.634,.356,741),(2,.919,.483,1033))):
        q = ramp(t,2+i*.6,3.55+i*.6)
        if not q:
            continue
        color = PALETTE[chr_number-1]
        fx,fy = x+(u-.5)*w,376+(v-.5)*w
        dot(ImageDraw.Draw(im),fx,fy,round(.079*w),None,tone(color,q),2)
        plate(im,SOURCE/f"banded-chr{chr_number}.webp",(dest,382),(244,294),q)
        label(im,dest,576,f"Chr{chr_number}",q,color)
    label(im,x,587,"G-banded metaphase",1.,MUTED,23)
    caption(im,"Distinctive light and dark bands provide an identity cue.")
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
        im = shot("Fixed cells / ChromoPix","Identity, across preparations")
        p = camera(t,6.3,8.2)
        w = 444
        paste(im,fish("reference",w),(334,382))
        line(im,(640,179),(640,556),HIRULE)
        paste(im,reveal_registered(picture(SOURCE/"example-dna.webp",w),fish("prediction",w),p),(946,382))
        label(im,334,597,"FISH annotation")
        label(im,946,597,"ChromoPix prediction",p)
        caption(im,"ChromoPix learns from both G-banding and FISH to segment and identify chromosomes.")
        return im
    im = shot("Fixed cells / FISH","Read identity from a DNA probe")
    pullback = camera(t,0,1.55)
    if pullback < 1:
        field_view(im,1-pullback,1-pullback,(mix(949,640,pullback),mix(379,382,pullback)),
                   round(mix(459,490,pullback)),1-pullback)
        caption(im,"A sequence-specific fluorescent probe marks Chr1 in this fixed cell.")
        return im
    layer = ramp(t,1.55,2.72)
    split = camera(t,3.25,5.05)
    x = mix(640,334,split)
    w = round(mix(490,444,split))
    paste(im,blend(picture(SOURCE/"example-dna.webp",w),fish_overlay(w),layer),(x,382))
    if split:
        line(im,(640,179),(640,556),tone(HIRULE,split))
        paste(im,light(fish("reference",444),split),(946,382))
    label(im,x,597,"DNA + Chr1 probe" if layer>.40 else "DNA image")
    label(im,946,597,"FISH annotation",split)
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


def segmentation(t):
    im=shot("ChromoPose / tracking","Follow the same chromosome")
    frames=fig5_frames()
    index=min(len(frames)-1,int(clamp((t-.75)/6.1)*len(frames)))
    t_image=frames[index]
    split=camera(t,.75,2.4)
    focus=ramp(t,2.7,3.72)
    x=mix(640,328,split)
    width=round(mix(478,463,split))
    paste(im,fig5_layer(t_image,"dna",width),(x,377))
    if split:
        mask=fig5_layer(t_image,"mask",463)
        target=fig5_layer(t_image,"track",463)
        paste(im,light(blend(mask,target,focus),split),(952,377))
        line(im,(640,177),(640,549),tone(HIRULE,split))
    label(im,x,590,"Live DNA projection")
    label(im,952,590,"Tracked instance" if focus>.55 else "3D instances",split,
          "#c0ddfa" if focus>.55 else WHITE)
    caption(im,"ChromoPose separates instances in 3D. Tracking links them through time.")
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
        paste(im,light(reveal_registered(surface,mip,project),project*(1-.87*descriptors)),(899,385))
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
        xy=arrays["lifted_combined_embedding"]
        labels=arrays["lifted_combined_y"]
    pr=Projection((0,0,668,401),[xy.min(0)-.5,xy.max(0)+.5])
    return raster_cloud(xy,labels,pr),len(xy)


def atlas_and_match(t):
    moving=camera(t,.38,2.56)
    query=ramp(t,5.1,6.3)
    if t < 5.7:
        im=shot("ChromoLift / reference atlas","Lift the fixed-cell reference")
    else:
        im=shot("ChromoID / identity retrieval","Match live shape to the atlas")
    cloud,count=lifted_atlas()
    if query<1:
        field_view(im,1,1,(mix(640,296,moving),381),round(mix(460,344,moving)),1)
    if query:
        paste(im,light(isolated_views(398,270)[0],query),(296,381))
    if moving:
        chart_x=round(mix(602,558,moving))
        blank=Image.new("RGB",cloud.size)
        paste_cloud(im,reveal_registered(blank,cloud,moving,85),(chart_x,166))
        line(im,(555,175),(555,574),tone(HIRULE,moving))
        label(im,892,564,f"{count:,} reference instances",moving,WHITE,26)
        label(im,892,598,"UMAP of lifted 3D shape descriptors",moving,MUTED,21)
    label(im,296,577,"Live 3D query" if query>.5 else "Known fixed-cell identities",
          moving,"#c0ddfa" if query>.5 else WHITE,25)
    caption(im,"Match on multiple 3D features, including volume, elongation and sphericity."
            if t>=5.7 else "ChromoLift maps identity-annotated 2D shapes into a shared 3D descriptor space.")
    return im


def live_identity(t):
    im=shot("ChromoPhy / independent tag","Check the inferred identity")
    p=ramp(t,1.05,2.58)
    frame=min(7,1+int(t*1.11))
    paste(im,live("chr18",frame,"candidates",451),(329,379))
    dna=live("chr18",frame,"dna",451)
    tag=live("chr18",frame,"dna-tag",451)
    paste(im,blend(dna,tag,p),(951,379))
    line(im,(640,174),(640,547),HIRULE)
    label(im,329,588,"Chr18 candidates")
    label(im,951,588,"DNA + fluorescent tag" if p>.40 else "Live DNA")
    caption(im,"An independent fluorescence channel supports the imaging-based identity check.")
    return im


def patient_division(t):
    im=shot("Patient-derived tumor cells","An unequal split of chromosome 14")
    frame=min(9,int(t*1.13))
    p=ramp(t,.6,2.4)
    crop=(43,45,363,375)
    image=picture(SOURCE/f"patient-{frame}.webp").crop(crop)
    size=round(mix(421,471,camera(t,1.,6.5)))
    image=blend(image.convert("L").convert("RGB"),image,p)
    width=round(size*image.width/image.height)
    paste(im,image.resize((width,size),Image.Resampling.LANCZOS),(640,382))
    reveal=ramp(t,8.4,9.72)
    if reveal:
        green=PALETTE[13]
        for x,a,(u,v),count in ((191,311,(127,109),2),(1089,970,(271,210),4)):
            ink=tone(green,reveal)
            end=(640-width/2+(u-crop[0])*width/image.width,382-size/2+(v-crop[1])*size/image.height)
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
    label(im,936,600,"Passage",p,MUTED,22)
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
    plate(im,SOURCE/"response.webp",(942,386),(498,302),q)
    label(im,961,539,"Drug concentration (log₁₀ mol/L)",q,MUTED,22)
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


# A short dip separates changes of specimen or scale. Aligned views and
# object-to-point handoffs cut in place so no pair of headlines overlaps.
SCENES = [
    (0,opening,False),
    (11,banding,True),
    (25,virtual_fish,False),
    (41,morphology,False),
    (50,to_one_point,False),
    (56,population,False),
    (66,native_comparison,True),
    (72,centroid_comparison,False),
    (78,segmentation,True),
    (86,lift_training,False),
    (94,atlas_and_match,True),
    (103,live_identity,True),
    (109,patient_division,True),
    (122,micromanipulation,True),
    (128,qpcr_chapter,False),
    (132.5,assay_evidence,False),
    (139,treatment_followup,True),
    (151,broader_recovery,False),
    (161,closing,True),
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


def write_video(path, times, preview=False):
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    frame_size=(640,360) if preview else (W,H)
    cmd=[ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo",
         "-pixel_format","rgb24","-video_size",f"{frame_size[0]}x{frame_size[1]}",
         "-framerate",str(FPS),"-i","pipe:0","-an","-c:v","libx264",
         "-preset","fast","-threads","4","-crf","19","-pix_fmt","yuv420p",
         "-movflags","+faststart",str(path)]
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
    args=parser.parse_args()
    if args.preview:
        ranges=((3.5,10.),(131.5,145.5),(151.,156.5),(161.,168.))
        times=(i/FPS for a,b in ranges for i in range(round(a*FPS),round(b*FPS)))
        write_video(args.preview,times,preview=True)
        return
    with tempfile.TemporaryDirectory(dir=ASSETS) as staging:
        poster=Path(staging)/"intro-poster.webp"
        render(119.4).save(poster,"WEBP",quality=94)
        video=Path(staging)/"intro.mp4"
        write_video(video,(i/FPS for i in range(DURATION*FPS)))
        video.replace(ASSETS/video.name)
        poster.replace(ASSETS/poster.name)


if __name__ == "__main__":
    main()
