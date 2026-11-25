"""Rebuild film-only source layers from the paper figures and native images."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter
from scipy import ndimage as ndi
import tifffile as tf

from prepare_intro_story import (OPENING_SPREAD, fig2_panels, fig3_panels, fig3_perturbation_bars,
                                 fig5_identity_panels, fig5_panels, karyotype,
                                 save, white_to_black)
from prepare_live_comparison import palette
from prepare_live_tracks import physical_spacing, read_frame


def flatten(image):
    rgba = image.convert("RGBA")
    base = Image.new("RGB",rgba.size)
    base.paste(rgba,mask=rgba.getchannel("A"))
    return base


def original_panel(path, box=None):
    with Image.open(path) as image:
        return flatten(image if box is None else image.crop(box))


def banding_thumbnail(image, annotations):
    matte = Image.new("L", image.size)
    draw = ImageDraw.Draw(matte)
    for row in annotations.splitlines():
        coords = np.array(row.split()[1:], dtype=float).reshape(-1, 2) * image.size
        draw.polygon(list(map(tuple, coords)), fill=255)
    values = np.asarray(image, dtype=float)
    background = np.median(values[np.asarray(matte) == 0])
    white = np.percentile(values[np.asarray(matte) != 0], 99)
    display = Image.fromarray(np.uint8(np.clip((values-background)/(white-background),0,1)*245))
    return Image.composite(display, Image.new("L",image.size), matte.filter(ImageFilter.GaussianBlur(1)))


def transfer_fields(dapi_path, predictions, out):
    # The archived inference square is an enlargement of the native acquisition.
    # Measured on id01_HFF_FISH_00023; check the match before applying predictions.
    with Image.open(dapi_path) as im:
        dapi = np.array(im.convert("RGB").getchannel("B"),dtype=np.float32)
    sx, sy, x0, y0 = .901379, .901324, 1291.951, 791.973
    baseline = np.asarray(original_panel(predictions / "id01_HFF_FISH_00023_MaskRCNN_banding.png",
                                         (0,0,1024,1024)),dtype=np.float32)
    yy,xx = np.mgrid[16:1008:8,72:952:8]
    sampled = ndi.map_coordinates(dapi,[y0+sy*yy,x0+sx*xx],order=1)
    if np.corrcoef(sampled.ravel(),baseline[yy,xx,0].ravel())[0,1] < .99:
        raise ValueError("Full DAPI field does not match the archived benchmark")
    scale,bias = np.polyfit(sampled.ravel(),baseline[yy,xx,0].ravel(),1)
    # A single viewport around the spread preserves the tips outside the model input.
    yy,xx = np.mgrid[680:1850,1170:2340]
    gray = np.maximum(dapi[yy,xx]*scale+bias-10,0)*1.38
    coords = [(yy-y0)/sy,(xx-x0)/sx]
    for name,model in (("mask-rcnn","MaskRCNN"),("v11","YOLOv11-Seg"),("v12","YOLOv12-Seg")):
        p = predictions / f"id01_HFF_FISH_00023_{model}_banding.png"
        if not np.array_equal(np.asarray(original_panel(p,(0,0,1024,1024))),baseline):
            raise ValueError(f"Archived prediction does not share the evaluated field: {model}")
        pred = np.asarray(original_panel(p,(0,1024,1024,2048)),dtype=np.float32)
        delta = pred-baseline
        rgb = np.stack([gray+ndi.map_coordinates(delta[:,:,c],coords,order=1,mode="constant")
                        for c in range(3)],axis=-1)
        save(Image.fromarray(np.uint8(np.clip(rgb,0,255))),out,f"banding-only-{name}")


def fixed_layers(args):
    out = args.output
    example = json.loads((out/"example.json").read_text())
    name = example["base"]
    with ZipFile(args.dataset) as data:
        def photo(member):
            with Image.open(BytesIO(data.read(member))) as image:
                return image.convert("L")

        for member in (f"2D/test/image/{OPENING_SPREAD}.png",f"2D/test/label/Banding/{OPENING_SPREAD}.txt"):
            data.extract(member,args.figures)
        karyotype(args.figures/"2D/test",OPENING_SPREAD,out/"story")
        banding = photo("2D/test/image/Banding_04301.png")
        annotation = data.read("2D/test/label/Banding/Banding_04301.txt").decode()
        save(banding_thumbnail(banding,annotation),out,"banding-dna")
        dna = photo(f"2D/train/image/{name}.png")
        save(dna,out,"example-dna")
        # Confirm the selected mask against the original annotation before re-rasterizing.
        with Image.open(out/"example-mask.png") as mask:
            old = np.asarray(mask.convert("L").resize(dna.size,Image.Resampling.NEAREST)) > 0
        lines = data.read(f"2D/train/label/FISH/{name}.txt").decode().splitlines()
        best_score, best = 0, None
        for row in lines:
            category,*coords = row.split()
            if int(category) != example["class_id"]:
                continue
            polygon = np.array(coords,dtype=float).reshape(-1,2)*dna.size
            layer = Image.new("L",dna.size)
            ImageDraw.Draw(layer).polygon(list(map(tuple,polygon)),fill=255)
            hit = np.asarray(layer)>0
            score = (hit&old).sum() / (hit|old).sum()
            if score > best_score:
                best_score,best,outline = score,layer,polygon
        if best_score < .9:
            raise ValueError(f"Original FISH polygon does not match the film anchor: IoU {best_score:.3f}")
        best.save(out/"example-mask.png")
        (out/"example-outline.json").write_text(json.dumps(outline.tolist()))
        # Fig. 3D, object C, the rightmost Chr1 annotation in this specimen.
        rightmost = max((np.array(row.split()[1:],dtype=float).reshape(-1,2)
                         for row in lines if row.split()[0] == "0"),key=lambda p:p[:,0].max())
        contour = Image.new("L",dna.size)
        ImageDraw.Draw(contour).polygon(list(map(tuple,rightmost*dna.size)),fill=255)
        detail = Image.composite(dna,Image.new("L",dna.size),contour)
        x0,y0,x1,y1 = contour.getbbox()
        save(detail.crop((x0-8,y0-8,x1+8,y1+8)),out/"story","pix-review-detail")

    sx,sy,bx,by = json.loads((out/"fish-probe-registration.json").read_text())["processed_to_raw"]
    with Image.open(args.probe) as probe:
        sw,sh = probe.size
        registered = probe.convert("RGB").transform(dna.size,Image.Transform.AFFINE,
            (sx*sw/dna.width,0,bx*sw,0,sy*sh/dna.height,by*sh),Image.Resampling.BICUBIC)
    save(registered,out,"fish-probe")

    original = args.archive.parent/"invert/fig3"
    gray = np.asarray(dna.convert("RGB"),dtype=np.float32)
    for layer,suffix in (("reference",""),("prediction","_ChromoPix")):
        source = np.asarray(original_panel(original/f"{name}{suffix}.png",(0,1024,1024,2048)),dtype=np.float32)
        chroma = np.ptp(source,axis=2)
        visible = ndi.binary_opening(chroma>6,structure=np.ones((5,5)))
        opacity = np.clip(chroma/52,0,.72)*visible*np.clip((gray.mean(2)-13)/35,0,1)
        hue = (source-source.min(2,keepdims=True))/np.maximum(chroma[...,None],1)
        rgb = gray*(1-opacity[...,None])+(55+200*hue)*opacity[...,None]
        save(Image.fromarray(np.uint8(np.clip(rgb,0,255))),out,f"fish-{layer}")


def paper_layers(args):
    out = args.output/"story"
    figs = args.figures
    fig2_panels(figs/"fig2.png",out)
    fig3_panels(figs/"fig3.png",out)
    fig3_perturbation_bars(args.archive/"archive/fig3",out)
    fig5_panels(args.ppt_media,out)
    fig5_identity_panels(args.fig5_rasters,args.fig5_vectors,out)

    transfer_fields(args.transfer_dapi,args.archive/"archive/fig2",out)
    for name,specimen in (("line-hff","id01_HFF_FISH_00013"),("line-mcf","id01_MCF-10F_FISH_00070"),
                          ("line-rpe","id01_RPE1_FISH_00038"),("line-mrc","idY_MRC5_FISH_00024"),
                          ("scale-small","id02_MCF-10F_FISH_00067"),("scale-large","id01_MCF-10F_FISH_00139")):
        p = args.archive/"archive/fig3"/f"{specimen}.png"
        save(original_panel(p,(0,1024,1024,2048)),out,f"pix-{name}")
    for name,suffix in (("original",""),("rotate","_RRRF"),("crowd","_CP"),("blur","_PSF")):
        p = args.archive/"archive/fig3"/f"id01_HFF_FISH_00003{suffix}.png"
        save(original_panel(p),out,f"pix-perturb-{name}")

    with ZipFile(args.pptx) as slides:
        for name,index in (("micromanipulator",325),("plate",322),("pakt",300),("akt",293)):
            with Image.open(BytesIO(slides.read(f"ppt/media/image{index}.png"))) as image:
                save(image,args.output,name)
    # Fig. 6 plots exported from their original Illustrator page.
    with Image.open(figs/"fig6.png") as source:
        for name,bounds in (("qpcr",(704,484,898,698)),("response",(353,986,493,1090)),
                            ("response-legend",(370,953,615,985))):
            crop = tuple(round(v*s/r) for v,s,r in zip(bounds,source.size*2,(900,1272)*2))
            save(white_to_black(source.crop(crop)),args.output,name)


def series_by_grid(stack, *paths):
    files = [stack.enter_context(tf.TiffFile(path)) for path in paths]
    series = [file.series[0] for file in files]
    if any(row.axes!="TZYX" or row.shape!=series[0].shape for row in series):
        raise ValueError(f"The native volumes do not share a TZYX grid: {paths}")
    return physical_spacing(files[0]),series


def render_front(volume, spacing, lut, up):
    from rounded_render import build_surfaces, shade
    surfaces,_ = build_surfaces(volume,spacing,up=up,front_only=True)
    bounds = (0.,(volume.shape[0]-1)*spacing[0])
    return shade(surfaces[0],lut,spacing,bounds,up=up),surfaces[0][0]


def gray_lut(max_id, intensity=205):
    lut = np.full((max_id+1,3),intensity,np.uint8)
    lut[0] = 0
    return lut


def dna_image(mip, low, high, gamma=1., ceiling=245):
    gray = np.uint8(ceiling*np.clip((mip.astype(float)-low)/max(1.,high-low),0,1)**gamma)
    return Image.fromarray(gray)


def glow(dna, tag, high):
    field = np.clip((tag.astype(np.float32)/max(high,1)-.10)/.90,0,1)**1.05
    field = np.asarray(Image.fromarray(field).resize(dna.size,Image.Resampling.BILINEAR))
    factor = dna.width/420
    core,body,halo = (ndi.gaussian_filter(field,s*factor) for s in (1.15,4.,9.))
    alpha = 1-np.exp(-(3.1*core+4.2*body+5.6*halo))
    rgb = np.asarray(dna.convert("RGB"),dtype=np.float32)/255
    rgb = rgb*(1-.58*alpha[...,None])+np.array([.08,.78,.23])*(.58*alpha[...,None])
    rgb = 1-(1-rgb)*np.exp(-(2*core+2.1*body+3.6*halo)[...,None]*np.array([.10,.9,.24]))
    white = np.clip((core-.40)*.7,0,.35)[...,None]
    return Image.fromarray(np.uint8(np.clip(rgb*(1-white)+np.array([.76,1.,.82])*white,0,1)*255))


def raw_path(args, group, name):
    return args.release/"image/raw_pre/tiff"/group/f"{name}-DAPI.tif"


def mask_path(args, group, name):
    return args.release/"mask/chrp3d_post_tracked"/group/f"{name}.tif"


def live_layers(args, case):
    sys.path.insert(0,str(args.renderer))
    out = args.output/"live"/case
    out.mkdir(parents=True,exist_ok=True)
    if case=="mitosis":
        with tf.TiffFile(raw_path(args,"others","RPE1_3_F2CGD")) as source:
            for t in range(19):
                mip = read_frame(source.series[0],t).max(0)
                save(dna_image(mip,*np.percentile(mip,[1,99.8])),out,f"{t}-dna")
        return
    name = {"chr3":"RPE-1 chr3_2026-09-08_5 CCDS6","chr18":"RPE-1 chr18_2026-09-09_6 CCDS5"}[case]
    review = args.tags/case
    candidate = (review/"group_2" if case=="chr18" else review)/"candidates.ome.tif"
    times = range(2,8) if case=="chr3" else range(8)
    with ExitStack() as stack:
        spacing,(raw,full,targets) = series_by_grid(stack,raw_path(args,case,name),mask_path(args,case,name),candidate)
        measured = stack.enter_context(tf.TiffFile(review/"image.ome.tif")).series[0]
        depth = raw.shape[1]
        if measured.shape != (raw.shape[0],2,*raw.shape[1:]) or measured.axes!="TCZYX":
            raise ValueError(f"The Tag channel is not matched to the DAPI series: {case}, {measured.axes}, {measured.shape}")
        mips = {t:np.maximum.reduce([p.asarray() for p in measured.pages[(2*t+1)*depth:(2*t+2)*depth]]) for t in times}
        high = np.percentile(np.concatenate([p[p>0] for p in mips.values()]),99)
        for t in times:
            mip = read_frame(raw,t).max(0)
            dna = dna_image(mip,*np.percentile(mip,[1,99.8]))
            full_frame,selected = read_frame(full,t),read_frame(targets,t)
            lut = gray_lut(max(int(full_frame.max()),int(selected.max())))
            rgb,_ = render_front(full_frame,spacing,lut,4)
            lut[1:] = ImageColor.getrgb({"chr3":"#990000","chr18":"#008080"}[case])
            color, ids = render_front(selected,spacing,lut,4)
            rgb[ids>0] = color[ids>0]
            scene = Image.fromarray(rgb)
            save(dna,out,f"{t}-dna")
            save(glow(dna.resize((768,768),Image.Resampling.LANCZOS),mips[t],high),out,f"{t}-dna-tag")
            save(scene,out,f"{t}-candidates")


def patient_layers(args):
    sys.path.insert(0,str(args.renderer))
    out = args.output
    info = json.loads((out/"patient-source.json").read_text())
    name = info["sample"]
    path = name+"_24.tif"
    with ExitStack() as stack:
        spacing,(_,full,candidate) = series_by_grid(stack,raw_path(args,"patients",name),
            args.patient/"tiff/full_context"/path,args.patient/"tiff/candidates"/path)
        for t in range(full.shape[0]):
            context,selected = read_frame(full,t),read_frame(candidate,t)
            if not np.array_equal(context[selected>0],selected[selected>0]):
                raise ValueError(f"Patient candidate/context mismatch at t={t}")
            lut = gray_lut(int(context.max()))
            rgb,_ = render_front(context,spacing,lut,4)
            lut[1:] = (51,255,51)
            color, ids = render_front(selected,spacing,lut,4)
            rgb[ids>0] = color[ids>0]
            save(Image.fromarray(rgb),out,f"patient-{t}")


def fig5_track_regions(volumes, times, info, spacing):
    """Trace the selected target through the merged early masks."""
    by_time = dict(zip(times, volumes))
    regions = {t: volume==info["track_id"] for t,volume in by_time.items()}
    repair = info.get("early_track_repair")
    if repair is None:
        return regions
    neighbor = by_time[repair["anchor_t"]]==repair["neighbor_anchor_label"]
    for frame in reversed(repair["frames"]):
        t = frame["t"]
        merged = by_time[t]==frame["merged_label"]
        if not merged.any() or not regions[t+1].any() or not neighbor.any():
            raise ValueError(f"Missing a foreground region for the Fig. 5f repair at t={t}")
        seeds = [
            ndi.shift(mask,frame[key],order=0,mode="constant",cval=False,prefilter=False)
            for mask,key in ((regions[t+1],"target_shift_zyx"),(neighbor,"neighbor_shift_zyx"))
        ]
        roi = ndi.find_objects(merged | seeds[0] | seeds[1])[0]
        target_dist, neighbor_dist = (
            ndi.distance_transform_edt(~seed[roi],sampling=spacing) for seed in seeds
        )
        target = np.zeros_like(merged)
        target[roi] = merged[roi] & (target_dist < neighbor_dist)
        regions[t] = target
        neighbor = merged & ~target
        if not target.any() or not neighbor.any():
            raise ValueError(f"Could not divide the merged Fig. 5f mask at t={t}")
    return regions


def fig5_live_layers(args):
    sys.path.insert(0,str(args.renderer))
    out = args.output/"fig5-live"
    info = json.loads((out/"source.json").read_text())
    track = info["track_id"]
    times = [frame["t"] for frame in info["frames"]]
    with ExitStack() as stack:
        spacing,(raw,tracked) = series_by_grid(stack,args.release/info["raw_source_relative_to_release"],
                                              args.release/info["tracked_source_relative_to_release"])
        mips = np.stack([read_frame(raw,t).max(0) for t in times])
        volumes = [read_frame(tracked,t) for t in times]
        regions = fig5_track_regions(volumes,times,info,spacing)
        repaired_times = {frame["t"] for frame in info.get("early_track_repair",{}).get("frames",[])}
        y,x = np.nonzero(np.any([v.any(0) for v in volumes],axis=0))
        side = max(np.ptp(x),np.ptp(y))+24
        cx,cy = (x.min()+x.max()+1)/2,(y.min()+y.max()+1)/2
        box = tuple(round(v) for v in (cx-side/2,cy-side/2,cx+side/2,cy+side/2))
        low,high = np.percentile(mips[:,y,x],[9,99.6])
        up = 4
        crop = tuple(v*up for v in box)
        frame_info = []
        for t,volume,mip in zip(times,volumes,mips):
            binary = regions[t]
            if not binary.any():
                raise ValueError(f"Fig. 5f track {track} missing at t={t}")
            active_label = track
            if t in repaired_times:
                active_label = int(volume.max())+1
                volume = volume.astype(np.int32)
                volume[binary] = active_label
            rgb,labels = render_front(volume,spacing,gray_lut(int(volume.max()),207),up)
            target,ids = render_front(binary.astype(np.uint16),spacing,np.array([[0,0,0],[180,219,251]],np.uint8),up)
            visible = labels==active_label
            overlay = np.uint8(rgb*.66)
            overlay[visible] = target[visible]
            dna = dna_image(mip,low,high,.85,247).crop(box)
            save(dna,out,f"{t}-dna")
            # The overlay is the XY outline of the same 3D label in this acquisition.
            silhouette = Image.fromarray(binary.any(axis=0)).crop(box)
            projected = np.asarray(silhouette.resize((dna.width*up,dna.height*up),Image.Resampling.NEAREST))
            edge = ndi.binary_dilation(projected,iterations=3) ^ ndi.binary_erosion(projected,iterations=2)
            annotated = np.array(dna.resize((dna.width*up,dna.height*up),Image.Resampling.LANCZOS).convert("RGB"))
            annotated[edge] = (132,201,253)
            save(Image.fromarray(annotated),out,f"{t}-dna-track")
            save(Image.fromarray(rgb).crop(crop),out,f"{t}-mask")
            save(Image.fromarray(overlay).crop(crop),out,f"{t}-track")
            frame_info.append({
                "t": t, "instance_voxels": int(binary.sum()),
                "front_visible_fraction": round(np.count_nonzero(visible)/(ids>0).sum(),4),
            })
        roi = ndi.find_objects(binary)[0]
        selected = np.pad(binary[roi].astype(np.uint16),5)
        intensity = np.pad(np.where(binary[roi],read_frame(raw,times[-1])[roi],0),5).max(0).astype(float)
        rgb,_ = render_front(selected,spacing,np.array([[0,0,0],[185,219,243]],np.uint8),12)
        low,high = np.percentile(intensity[intensity>0],[1,99.2])
        scaled = np.where(intensity>0,90+156*np.clip((intensity-low)/max(1,high-low),0,1)**.8,0)
        save(Image.fromarray(rgb),args.output,"tracked-chromosome-3d")
        save(Image.fromarray(np.uint8(scaled)).resize(Image.fromarray(rgb).size,Image.Resampling.LANCZOS),
             args.output,"tracked-chromosome-projection")
    info["frames"] = frame_info
    (out/"source.json").write_text(json.dumps(info,indent=2)+"\n")


def pose_live_layers(args):
    sys.path.insert(0,str(args.renderer))
    # The preprocessed DNA MIP in Fig. S2c matches this release frame.
    group,name,time = "chr3","RPE-1 chr3_2026-09-08_5 CCDS6",3
    out = args.output/"story"
    methods = (("fast-chimp","chimp"),("cellpose","cpsam"),("chromopose","chrp3d_post"))
    paths = [args.release/"mask"/dirname/group/f"{name}.tif" for _,dirname in methods]
    with ExitStack() as stack:
        spacing,(_, *masks) = series_by_grid(stack,raw_path(args,group,name),*paths)
        for (method,_),series in zip(methods,masks):
            volume = read_frame(series,time)
            rgb,_ = render_front(volume,spacing,palette(int(volume.max())),4)
            save(Image.fromarray(rgb),out,f"pose-{method}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("output","dataset","probe","transfer-dapi","archive","pptx","ppt-media","figures","fig5-rasters",
                   "fig5-vectors","release","patient","tags","renderer"):
        parser.add_argument("--"+option,type=Path,required=True)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(f,args) for f in (fixed_layers,paper_layers,patient_layers,fig5_live_layers,pose_live_layers)]
        jobs.extend(pool.submit(live_layers,args,case) for case in ("mitosis","chr3","chr18"))
        for job in jobs:
            job.result()


if __name__=="__main__":
    main()
