#!/usr/bin/env python3
"""Register preprocessed 4D DAPI before ChromoPose segmentation.

The default image-only path estimates rigid motion from raw and preprocessed
DAPI, then writes the preprocessed movie on a common padded canvas. It uses no
predicted masks. The former mask-assisted path remains available with
--legacy-mask-assisted for reproducing older outputs.
"""
from __future__ import annotations
import argparse,json,time,hashlib,os,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from skimage.registration import phase_cross_correlation
import tifffile as tf
VERSION="rigid-stabilization-1.1"
IMAGE_VERSION="image-only-rigid-registration-v1"

def read_frame(f, t):
    s = f.series[0]
    if s.axes == "ZYX":
        if t:
            raise IndexError(t)
        return s.asarray()
    if len(s.shape) != 4 or s.axes not in ("TZYX", "QQYX"):
        raise ValueError(f"Unsupported axes: {s.axes} {s.shape}")
    z = s.shape[1]
    if len(s.pages) == s.shape[0] * z:
        return np.stack([p.asarray() for p in s.pages[t * z : (t + 1) * z]])
    return np.asarray(tf.memmap(f.filehandle.path)[t]).copy()

def get_metadata(f):
    m = dict(f.imagej_metadata or {})
    if not m:
        try:
            m = json.loads(f.pages[0].description or "{}")
        except (ValueError, TypeError):
            m = {}
    if m.get("unit") not in ("micron", "um", "µm") or "spacing" not in m:
        raise ValueError("Missing calibrated spacing in raw DAPI")

    def res(tag):
        a, b = f.pages[0].tags[tag].value
        return b / a

    sp = np.array([m["spacing"], res("YResolution"), res("XResolution")], float)
    if not np.all(np.isfinite(sp)) or np.any(sp <= 0):
        raise ValueError(sp)
    return sp, m


def registration_shift(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float, str]:
    """Estimate source-to-target translation from DAPI signal."""
    src_xy = np.sqrt(np.maximum(source.max(axis=0).astype(np.float32), 0))
    tgt_xy = np.sqrt(np.maximum(target.max(axis=0).astype(np.float32), 0))
    src_xy -= np.median(src_xy)
    tgt_xy -= np.median(tgt_xy)
    shift_yx, _, _ = phase_cross_correlation(
        tgt_xy, src_xy, upsample_factor=2, normalization=None
    )
    moved = ndi.shift(src_xy, shift_yx, order=1, mode="constant", prefilter=False)
    use = (tgt_xy > np.percentile(tgt_xy, 70)) | (
        moved > np.percentile(moved, 70)
    )
    corr = (
        float(np.corrcoef(tgt_xy[use], moved[use])[0, 1])
        if use.sum() >= 32 and tgt_xy[use].std() and moved[use].std()
        else 0.0
    )
    method = "phase_correlation"
    max_reasonable = 0.25 * min(source.shape[1:])
    if not np.all(np.isfinite(shift_yx)) or np.linalg.norm(shift_yx) > max_reasonable:
        src_positive = np.maximum(src_xy, 0)
        tgt_positive = np.maximum(tgt_xy, 0)
        src_center = np.asarray(ndi.center_of_mass(src_positive))
        tgt_center = np.asarray(ndi.center_of_mass(tgt_positive))
        shift_yx = np.nan_to_num(tgt_center - src_center)
        shift_yx = np.clip(shift_yx, -max_reasonable, max_reasonable)
        method = "intensity_centroid_fallback"

    src_z = source.sum(axis=(1, 2), dtype=np.float64)
    tgt_z = target.sum(axis=(1, 2), dtype=np.float64)
    z_src = float(np.sum(np.arange(len(src_z)) * src_z) / max(src_z.sum(), 1))
    z_tgt = float(np.sum(np.arange(len(tgt_z)) * tgt_z) / max(tgt_z.sum(), 1))
    shift_z = float(np.clip(z_tgt - z_src, -6, 6))
    return np.array([shift_z, *shift_yx], float), corr, method


def transform(params,spacing,center):
    R=Rotation.from_rotvec(params[3:]).as_matrix()
    D=np.diag(spacing)
    A=np.linalg.solve(D,R.T@D)
    offset=np.linalg.solve(D,center-R.T@(center+params[:3]))
    return A,offset,R

def warp(x,params,spacing,center,order=1):
    A,b,_=transform(params,spacing,center)
    return ndi.affine_transform(x,A,offset=b,output_shape=x.shape,order=order,mode='constant',prefilter=False)

def prepare(raw,labels):
    x=raw.astype(np.float32);bg=np.percentile(x,40)
    v=x[labels>0];hi=np.percentile(v,99.5) if len(v) else bg+1
    x=np.sqrt(np.clip((x-bg)/max(hi-bg,1),0,1))
    # Broad foreground support; no categorical IDs enter the registration metric.
    roi=ndi.binary_dilation(labels>0,iterations=3)
    return ndi.gaussian_filter(x*roi,(.4,.7,.7))

def ncc(a,b):
    mask=(a>.10)|(b>.10)
    if mask.sum()<30:return 0.
    x=a[mask];y=b[mask];x=x-x.mean();y=y-y.mean()
    return float(np.sum(x*y)/max(np.sqrt(np.sum(x*x)*np.sum(y*y)),1e-9))

def small(x,spacing,maxsize):
    stride=np.maximum(1,np.ceil(np.array(x.shape)/maxsize).astype(int))
    return x[tuple(slice(None,None,int(s)) for s in stride)],spacing*stride

def estimate(a,b,spacing,old):
    center=(np.array(a.shape)-1)*spacing/2
    x,sp=small(a,spacing,56);y,_=small(b,spacing,56)
    sh=phase_cross_correlation(y,x,upsample_factor=2,normalization=None)[0]*sp
    candidates=[np.r_[old*spacing,[0.,0.,0.]],np.r_[sh,[0.,0.,0.]],np.zeros(6)]
    candidates=[p for p in candidates if np.isfinite(p).all() and np.linalg.norm(p[:3])<12]
    start=max(candidates,key=lambda p:ncc(warp(x,p,sp,center),y))
    bounds=[(float(v-2),float(v+2)) for v in start[:3]]
    def fit(trans,rigid,maxsize,fev):
        xx,ss=small(a,spacing,maxsize);yy,_=small(b,spacing,maxsize)
        init=trans if rigid else trans[:3]
        def objective(v):
            p=v if rigid else np.r_[v,[0.,0.,0.]]
            return -ncc(warp(xx,p,ss,center),yy)
        bd=bounds+([(-np.deg2rad(15),np.deg2rad(15))]*3 if rigid else [])
        r=minimize(objective,init,method='Powell',bounds=bd,options={'maxfev':fev,'xtol':.015,'ftol':.0005})
        v=r.x if r.fun<=objective(init) else init
        return v if rigid else np.r_[v,[0.,0.,0.]]
    translation=fit(start,False,56,140)
    translation=fit(translation,False,80,100)
    rigid=fit(translation,True,56,240)
    rigid=fit(rigid,True,80,160)
    return [np.r_[old*spacing,[0.,0.,0.]],translation,rigid],center

def physical_matrix(params,center):
    R=Rotation.from_rotvec(params[3:]).as_matrix()
    H=np.eye(4);H[:3,:3]=R;H[:3,3]=center-R@center+params[:3]
    return H

def quality(a,b,sl,tl,p,sp,center):
    moved=warp(a,p,sp,center)
    mask=warp(sl,p,sp,center,0)>0;target=tl>0
    dice=2*np.count_nonzero(mask&target)/max(np.count_nonzero(mask)+np.count_nonzero(target),1)
    corr=ncc(moved,b)
    return dict(ncc=corr,dice=dice,score=.65*corr+.35*dice)

def choose(a,b,sl,tl,sp,old):
    params,center=estimate(a,b,sp,old)
    names=['old_translation','translation_3d','rigid_3d','identity']
    params.append(np.zeros(6))
    # Full-resolution acceptance, separate from downsampled optimization.
    stats=[quality(a,b,sl,tl,p,sp,center) for p in params]
    baseline=max([0,3],key=lambda i:stats[i]['score']);chosen=baseline
    for i in [1,2]:
        if (stats[i]['ncc']>=stats[baseline]['ncc']-.005 and
            stats[i]['dice']>=stats[baseline]['dice']-.01 and
            stats[i]['score']>stats[chosen]['score']+.003):chosen=i
    uncertain=stats[chosen]['ncc']<.15 or stats[chosen]['dice']<.25
    candidates={n:dict(**s,parameters_zyx_um_rotvec_rad=p.tolist()) for n,s,p in zip(names,stats,params)}
    return physical_matrix(params[chosen],center),dict(selected=names[chosen],baseline=names[baseline],
        uncertain=uncertain,candidates=candidates,rotation_norm_deg=float(np.rad2deg(np.linalg.norm(params[chosen][3:]))))

def compose(pairs,anchor):
    C=[None]*(len(pairs)+1);C[anchor]=np.eye(4)
    for t in range(anchor+1,len(C)):C[t]=C[t-1]@np.linalg.inv(pairs[t-1])
    for t in range(anchor-1,-1,-1):C[t]=C[t+1]@pairs[t]
    return C

def canvas(shape,spacing,matrices):
    corners=np.array(list(itertools.product(*[(-.5,n-.5) for n in shape])))*spacing
    points=np.concatenate([corners@H[:3,:3].T+H[:3,3] for H in matrices])
    origin=(np.floor(points.min(0)/spacing)-2)*spacing
    end=np.ceil(points.max(0)/spacing)+2
    outshape=(end-np.rint(origin/spacing)).astype(int)+1
    return origin,tuple(int(n) for n in outshape)

def resample(x,H,spacing,origin,shape,order):
    inv=np.linalg.inv(H);D=np.diag(spacing)
    A=np.linalg.solve(D,inv[:3,:3]@D)
    offset=np.linalg.solve(D,inv[:3,:3]@origin+inv[:3,3])
    # Exact integer translations must not become -epsilon at source boundaries.
    A=np.where(np.isclose(A,np.rint(A),atol=1e-10,rtol=0),np.rint(A),A)
    offset=np.where(np.isclose(offset,np.rint(offset),atol=1e-9,rtol=0),np.rint(offset),offset)
    return ndi.affine_transform(x,A,offset=offset,output_shape=shape,order=order,mode='constant',prefilter=False)

def choose_image_transform(previous_raw,current_raw,previous_pre,current_pre,spacing):
    """Select image-only rigid candidates by preprocessed DAPI correlation."""
    a=prepare(previous_pre,previous_pre>0)
    b=prepare(current_pre,current_pre>0)
    if np.count_nonzero(a>.1)<30 or np.count_nonzero(b>.1)<30:
        return np.eye(4),dict(selected='identity_low_signal',uncertain=True)
    try:
        old,old_corr,old_method=registration_shift(previous_raw,current_raw)
    except (ValueError,RuntimeError) as error:
        old=np.zeros(3,float);old_corr=0.;old_method=f'failed:{type(error).__name__}'
    candidates,center=estimate(a,b,spacing,old)
    candidates.append(np.zeros(6,float))
    names=('old_translation','translation_3d','rigid_3d','identity')
    scores=[float(ncc(warp(a,p,spacing,center),b)) for p in candidates]
    baseline=max((0,3),key=lambda index:scores[index])
    chosen=baseline
    for index in (1,2):
        if scores[index]>=scores[baseline]-.005 and scores[index]>scores[chosen]+.003:
            chosen=index
    return physical_matrix(candidates[chosen],center),dict(
        selected=names[chosen],ncc=scores[chosen],uncertain=scores[chosen]<.15,
        old_corr=old_corr,old_method=old_method,
        candidates={name:dict(ncc=score,parameters_zyx_um_rotvec_rad=p.tolist())
                    for name,score,p in zip(names,scores,candidates)})

def run_image_movie(raw_path,pre_path,out_dir,anchor_t=None,report_path=None):
    """Register one DAPI movie without requiring segmentation masks."""
    raw_path=Path(raw_path);pre_path=Path(pre_path);out_dir=Path(out_dir)
    if not raw_path.name.endswith('-DAPI.tif'):
        raise ValueError(f'Expected a *-DAPI.tif input: {raw_path}')
    if pre_path.name!=raw_path.name:
        raise ValueError('Raw and preprocessed DAPI filenames must match')
    out_dir.mkdir(parents=True,exist_ok=True)
    base=raw_path.name.removesuffix('-DAPI.tif')
    output=out_dir/raw_path.name
    report_path=Path(report_path) if report_path else out_dir/f'{base}_transforms.json'
    temporary=output.with_name(output.name+'.partial')
    for path in (output,report_path,temporary):
        if path.exists():raise FileExistsError(path)
    with tf.TiffFile(raw_path) as raw,tf.TiffFile(pre_path) as pre:
        raw_series=raw.series[0];pre_series=pre.series[0]
        if raw_series.axes not in ('ZYX','TZYX') or pre_series.axes not in ('ZYX','TZYX'):
            raise ValueError('Expected ZYX or TZYX DAPI movies')
        shape=raw_series.shape if raw_series.axes=='TZYX' else (1,*raw_series.shape)
        pre_shape=pre_series.shape if pre_series.axes=='TZYX' else (1,*pre_series.shape)
        if shape!=pre_shape or raw_series.dtype!=np.dtype('uint16') or pre_series.dtype!=np.dtype('uint16'):
            raise ValueError('Raw and preprocessed DAPI must have identical TZYX shape and uint16 dtype')
        spacing,meta=get_metadata(raw)
        pre_spacing,_=get_metadata(pre)
        if not np.allclose(spacing,pre_spacing):
            raise ValueError('Raw and preprocessed voxel spacings differ')
        T=shape[0]
        anchor=T//2 if anchor_t is None else int(anchor_t)
        if not 0<=anchor<T:raise ValueError(f'anchor_t={anchor} outside 0..{T-1}')
        pairs=[];transitions=[]
        for t in range(1,T):
            matrix,detail=choose_image_transform(
                read_frame(raw,t-1),read_frame(raw,t),
                read_frame(pre,t-1),read_frame(pre,t),spacing)
            pairs.append(matrix)
            transitions.append(dict(source_t=t-1,target_t=t,**detail))
            print(json.dumps(dict(sample=base,step='image_affine',t=t,method=detail['selected'])),flush=True)
        matrices=compose(pairs,anchor)
        origin,output_shape=canvas(shape[1:],spacing,matrices)
        metadata={k:meta[k] for k in ('unit','spacing','finterval','fps') if k in meta}
        metadata.update(axes='TZYX',pipeline=IMAGE_VERSION,origin_zyx_um=origin.tolist(),anchor_t=anchor)
        hashes=[]
        def planes():
            for t in range(T):
                frame=read_frame(pre,t)
                moved=resample(frame.astype(np.float32),matrices[t],spacing,origin,output_shape,order=1)
                moved=np.rint(np.clip(moved,0,65535)).astype(np.uint16)
                if t==anchor:
                    corner=np.rint(-origin/spacing).astype(int)
                    crop=tuple(slice(int(k),int(k+n)) for k,n in zip(corner,frame.shape))
                    if not np.array_equal(moved[crop],frame):
                        raise AssertionError('Anchor image must be exact within the padded canvas')
                hashes.append(digest(moved))
                yield from moved
        tf.imwrite(temporary,planes(),shape=(T,*output_shape),dtype='uint16',
                   metadata=metadata,resolution=(1/spacing[2],1/spacing[1]),
                   photometric='minisblack',compression='zlib',bigtiff=True)
    with tf.TiffFile(temporary) as registered:
        if registered.series[0].shape!=(T,*output_shape) or registered.series[0].axes!='TZYX':
            raise AssertionError('Registered TIFF shape or axes mismatch')
        for t,expected in enumerate(hashes):
            if digest(read_frame(registered,t))!=expected:
                raise AssertionError(f'Registered frame t={t} failed verification')
    os.replace(temporary,output)
    report=dict(version=IMAGE_VERSION,mode='image_only_rigid',sample=base,
                raw=str(raw_path),preprocessed=str(pre_path),output=str(output),
                anchor_t=anchor,source_shape_tzyx=list(shape),output_shape_tzyx=[T,*output_shape],
                spacing_zyx_um=spacing.tolist(),origin_zyx_um=origin.tolist(),
                frame_to_anchor_physical_zyx=[matrix.tolist() for matrix in matrices],
                transitions=transitions,verified_sha256=hashes)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    save_json(report_path,report)
    print(json.dumps(dict(sample=base,step='image_affine_complete',output=str(output),shape=report['output_shape_tzyx'])),flush=True)
    return report

def digest(x):return hashlib.sha256(x if isinstance(x,bytes) else x.tobytes()).hexdigest()

def stat(p):
    s=p.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def save_json(p,x):
    temp=p.with_suffix('.json.partial');temp.write_text(json.dumps(x,indent=2));os.replace(temp,p)

def preview(out,base,t,before,after,lbefore,lafter):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import colorsys
    fig,axes=plt.subplots(2,2,figsize=(10,9),facecolor='black')
    high=max(float(np.percentile(before.max(0),99.7)),1)
    colors=np.zeros((max(int(lbefore.max()),int(lafter.max()))+1,3))
    for i in range(1,len(colors)):colors[i]=colorsys.hsv_to_rgb((i*.61803398875)%1,.6,.95)
    for col,(raw,lab,title) in enumerate([(before,lbefore,'Original'),(after,lafter,'Registered / padded')]):
        axes[0,col].imshow(raw.max(0),cmap='gray',vmin=0,vmax=high)
        z=np.where(lab>0,raw.astype(np.float32),-1).argmax(0)
        pr=np.take_along_axis(lab,z[None],axis=0)[0]
        axes[1,col].imshow(colors[pr],interpolation='nearest')
        axes[0,col].set_title(title,color='white')
        for row in range(2):axes[row,col].axis('off')
    fig.suptitle(f'{base} | t={t}\nraw_preprocessed + post CHRP3D | same IDs, NOT tracking',color='white',fontsize=11)
    fig.tight_layout();fig.savefig(out/f't{t:03d}.png',dpi=110,facecolor='black');plt.close(fig)

def run_movie(task):
    root,out,base=task;root=Path(root);out=Path(out);start=time.monotonic()
    sources=dict(raw=root/'raw'/f'{base}-DAPI.tif',pre=root/'raw_preprocessed'/f'{base}-DAPI.tif',
                 mask=root/'chrp3d_postprocessed'/f'{base}_shape_postprocessed.tif')
    reportpath=out/'reports'/f'{base}.json'
    if reportpath.exists():
        old=json.loads(reportpath.read_text())
        assert old['version']==VERSION and old['code_sha256']==digest(Path(__file__).read_bytes())
        assert all(stat(Path(p))==s for p,s in old['source_files'].items())
        return old
    source_stats={str(p):stat(p) for p in sources.values()}
    with tf.TiffFile(sources['raw']) as rf,tf.TiffFile(sources['mask']) as mf,tf.TiffFile(sources['pre']) as pf:
        shape=rf.series[0].shape;assert rf.series[0].axes=='TZYX' and shape==mf.series[0].shape==pf.series[0].shape
        assert pf.series[0].dtype==np.dtype('uint16')
        sp,meta=get_metadata(rf);spp,_=get_metadata(pf);assert np.allclose(sp,spp)
        T=shape[0];ap=root/'identified'/'reports'/f'{base}.json'
        anchor=int(json.loads(ap.read_text())['anchor']['representative_t']) if ap.exists() else T//2
        assert 0<=anchor<T
        pairs=[];transitions=[]
        src=read_frame(rf,0);sl=read_frame(mf,0);a=prepare(src,sl)
        for t in range(1,T):
            tgt=read_frame(rf,t);tl=read_frame(mf,t);b=prepare(tgt,tl)
            old,_,_=registration_shift(src,tgt)
            H,info=choose(a,b,sl,tl,sp,old);pairs.append(H)
            info.update(source_t=t-1,target_t=t,source_to_target_physical_zyx=H.tolist());transitions.append(info)
            print(json.dumps(dict(sample=base,step='estimate',t=t,method=info['selected'],uncertain=info['uncertain'])),flush=True)
            src,sl,a=tgt,tl,b
        C=compose(pairs,anchor);origin,oshape=canvas(shape[1:],sp,C)
        del src,sl,a
        if T>1:del tgt,tl,b
    outputs=dict(pre=out/'raw_preprocessed'/sources['pre'].name,mask=out/'chrp3d'/f'{base}_shape_postprocessed.tif')
    for p in outputs.values():assert not p.exists(),p
    qm=out/'qc'/base;qm.mkdir(parents=True,exist_ok=True)
    result=dict(version=VERSION,code_sha256=digest(Path(__file__).read_bytes()),sample=base,anchor_t=anchor,
        source_files=source_stats,source_shape=list(shape),output_shape=[T,*oshape],spacing_zyx_um=sp.tolist(),
        output_origin_zyx_um=origin.tolist(),frame_to_anchor_physical_zyx=[H.tolist() for H in C],
        transitions=transitions,outputs={k:str(p) for k,p in outputs.items()},frames=[])
    save_json(out/'transforms'/f'{base}.json',result)
    hashes={'pre':[],'mask':[]};frameinfo=[]
    md={k:meta[k] for k in ('unit','spacing','finterval','fps') if k in meta}
    md.update(axes='TZYX',pipeline=VERSION,origin_zyx_um=origin.tolist(),anchor_t=anchor)
    # Write each stack as a stream, one interpolation from its untouched original frame.
    for kind,order,dtype in [('pre',1,np.uint16),('mask',0,np.uint32)]:
        target=outputs[kind];temp=target.with_name(target.name+'.partial');assert not temp.exists(),temp
        def planes():
            with tf.TiffFile(sources[kind]) as f:
                for t in range(T):
                    x=read_frame(f,t)
                    y=resample(x.astype(np.float32) if kind=='pre' else x,C[t],sp,origin,oshape,order)
                    y=np.rint(np.clip(y,0,65535)).astype(dtype) if kind=='pre' else y.astype(dtype)
                    if t==anchor:
                        corner=np.rint(-origin/sp).astype(int)
                        crop=tuple(slice(int(k),int(k+n)) for k,n in zip(corner,x.shape))
                        assert np.array_equal(y[crop],x),'Anchor must be byte-exact, with padding only'
                    hashes[kind].append(digest(y))
                    if kind=='mask':
                        ids=np.unique(x);newids=np.unique(y);assert np.isin(newids,ids).all()
                        missing=np.setdiff1d(ids[ids>0],newids).tolist()
                        frameinfo.append(dict(t=t,input_ids=int(np.count_nonzero(ids)),output_ids=int(np.count_nonzero(newids)),
                            missing_ids_due_to_resampling=missing,foreground_before=int(np.count_nonzero(x)),foreground_after=int(np.count_nonzero(y))))
                        if t in {0,anchor,T-1}:
                            with tf.TiffFile(sources['pre']) as fp,tf.TiffFile(outputs['pre']) as fo:
                                preview(qm,base,t,read_frame(fp,t),read_frame(fo,t),x,y)
                    yield from y
        tf.imwrite(temp,planes(),shape=(T,*oshape),dtype=dtype,metadata=md,resolution=(1/sp[2],1/sp[1]),
                   photometric='minisblack',compression='zlib',bigtiff=True)
        with tf.TiffFile(temp) as f:
            assert f.series[0].shape==(T,*oshape) and f.series[0].dtype==np.dtype(dtype)
            for t in range(T):assert digest(read_frame(f,t))==hashes[kind][t]
        os.replace(temp,target)
    assert all(stat(Path(p))==s for p,s in source_stats.items())
    result.update(frames=frameinfo,verified_sha256=hashes,seconds=time.monotonic()-start,passed=True)
    save_json(reportpath,result)
    print(json.dumps(dict(sample=base,step='complete',frames=T,seconds=round(result['seconds'],1))),flush=True)
    return result

def self_test():
    rng=np.random.default_rng(13);a=np.zeros((32,48,48),np.float32)
    for _ in range(22):a[tuple(rng.integers([6,8,8],[26,40,40]))]=1
    a=ndi.gaussian_filter(a,1);a/=a.max();sp=np.array([.3,.13,.13]);c=(np.array(a.shape)-1)*sp/2
    p=np.array([.3,-.26,.39,.08,-.035,.02]);b=warp(a,p,sp,c);fits,_=estimate(a,b,sp,p[:3]/sp)
    assert ncc(warp(a,fits[-1],sp,c),b)>.98
    H=physical_matrix(p,c);C=compose([H,H],1)
    assert np.allclose(C[0],H) and np.allclose(C[2],np.linalg.inv(H))
    lab=(a>.25).astype(np.uint32)*7;origin,oshape=canvas(a.shape,sp,C)
    for mat in C:
        y=resample(lab,mat,sp,origin,oshape,0);assert set(np.unique(y))=={0,7}
        corners=np.array(list(itertools.product(*[(0,n-1) for n in a.shape])))*sp
        vox=(corners@mat[:3,:3].T+mat[:3,3]-origin)/sp
        assert np.all(vox>=0) and np.all(vox<=np.array(oshape)-1)
    assert np.array_equal(resample(lab,np.eye(4),sp,np.zeros(3),lab.shape,0),lab)
    rng=np.random.default_rng(17);full=rng.integers(1,65535,(9,13,17),dtype=np.uint16)
    sp2=np.array([.2,.0600000024,.0600000024]);o=-np.array([7,9,11])*sp2
    padded=resample(full,np.eye(4),sp2,o,(30,40,50),1)
    assert np.array_equal(padded[7:16,9:22,11:28],full)
    print('PASS synthetic anisotropic rigid recovery, matrix composition, padded bounds, label IDs, identity transform')

def crop_registered_movie(task):
    """Lossless common-index crop of registered image/labels; no interpolation.

    The union of ALL nonzero labels over time defines support, plus a physical
    margin. Unsegmented raw signal outside that region may be removed; its
    intensity fraction is recorded. Original artifacts are moved to backup only
    after the replacement TIFFs and GIF have all passed verification.
    """
    import importlib.util
    import torch
    from PIL import Image
    root, base, backup, margin = task
    root, backup = Path(root), Path(backup)
    torch.set_num_threads(1)
    paths = {'pre':root/'raw_pre_affine/tiff'/f'{base}-DAPI.tif',
             'mask':root/'chrp3d_post_affine'/f'{base}_shape_postprocessed.tif'}
    gif = root/'raw_pre_affine/gif'/f'{base}-DAPI_time_rotate_raw.gif'
    rp = root/'registered/reports'/f'{base}.json'
    prior = json.loads(rp.read_text())
    sp = np.array(prior['spacing_zyx_um'])
    with tf.TiffFile(paths['mask']) as f:
        shape = f.series[0].shape
        assert list(shape) == prior['output_shape']
        low = np.array(shape[1:]); high = np.zeros(3,dtype=int)
        for t in range(shape[0]):
            a = read_frame(f,t)
            for axis in range(3):
                occupied = np.flatnonzero(np.any(a != 0, axis=tuple(i for i in range(3) if i != axis)))
                if occupied.size:
                    low[axis] = min(low[axis],occupied[0])
                    high[axis] = max(high[axis],occupied[-1]+1)
    assert np.all(high>low)
    pad = np.ceil(margin/sp).astype(int)
    low = np.maximum(0,low-pad); high = np.minimum(shape[1:],high+pad)
    slices = tuple(slice(int(a),int(b)) for a,b in zip(low,high))
    newshape = (shape[0],*(high-low).tolist())
    origin = np.array(prior['output_origin_zyx_um'])+low*sp
    hashes={}; totals={'raw_total':0,'raw_removed':0}; staged={}
    for kind,path in paths.items():
        temp=path.with_name(path.name+'.crop.partial'); assert not temp.exists()
        staged[kind]=temp; hashes[kind]=[]
        with tf.TiffFile(path) as f:
            assert f.series[0].shape==shape
            _,md=get_metadata(f); dtype=f.series[0].dtype
        md={k:v for k,v in md.items() if k not in ('shape','axes')}
        md.update(axes='TZYX',origin_zyx_um=origin.tolist(),crop_start_zyx=low.tolist(),crop_margin_um=margin)
        def planes():
            with tf.TiffFile(path) as f:
                for t in range(shape[0]):
                    a=read_frame(f,t); b=np.ascontiguousarray(a[slices])
                    if kind=='mask':
                        assert np.count_nonzero(a)==np.count_nonzero(b)
                        assert np.array_equal(np.unique(a[a>0]),np.unique(b[b>0]))
                    else:
                        total=int(a.sum(dtype=np.uint64)); kept=int(b.sum(dtype=np.uint64))
                        totals['raw_total']+=total; totals['raw_removed']+=total-kept
                    hashes[kind].append(digest(b)); yield from b
        tf.imwrite(temp,planes(),shape=newshape,dtype=dtype,metadata=md,
                   resolution=(1/sp[2],1/sp[1]),photometric='minisblack',compression='zlib',bigtiff=True)
        with tf.TiffFile(temp) as f:
            assert f.series[0].shape==newshape
            for t in range(shape[0]): assert digest(read_frame(f,t))==hashes[kind][t]
    # Same rendering functions and display settings as the previous GIF batch.
    reference=Path(__file__).resolve().parent/'segmentation'/'visualize.py'
    spec=importlib.util.spec_from_file_location('gif_reference',reference)
    ref=importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    frames=[]; last=None
    with tf.TiffFile(staged['pre']) as f, torch.no_grad():
        for i in range(36):
            t=ref._time_index_for_combo_frame(i,36,shape[0])
            if t!=last:
                a=read_frame(f,t); lo,hi=ref.contrast_range_from_img3d(a,'xy',1.,99.)
                v=torch.from_numpy(a.astype(np.float32))[None,None]
                ratio=min(1.,256/max(a.shape)); size=tuple(max(1,round(n*ratio)) for n in a.shape)
                if size!=a.shape:v=torch.nn.functional.interpolate(v,size=size,mode='trilinear',align_corners=False)
                zero=torch.zeros_like(v);last=t
            rotated,_=ref.rotate_volume_around_y(v,zero,i*10.,torch.device('cpu'))
            g=((ref._vol_proj_from_rot(rotated,'xy')-lo)/(hi-lo)).clamp(0,1).mul(255).round().to(torch.uint8).numpy()
            frames.append(Image.fromarray(g))
    giftemp=gif.with_name(gif.name+'.crop.partial'); assert not giftemp.exists()
    frames[0].save(giftemp,format='GIF',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False,disposal=2)
    with Image.open(giftemp) as f:
        assert f.n_frames==36
        for t in range(36):f.seek(t);f.load()
    # Preserve the exact previous files, in their current directory layout.
    for kind,path in {**paths,'gif':gif}.items():
        dest=backup/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists()
        if path.exists():os.replace(path,dest)
        os.replace(giftemp if kind=='gif' else staged[kind],path)
    result=dict(sample=base,old_shape=list(shape),output_shape=list(newshape),
        crop_start_zyx=low.tolist(),crop_stop_zyx=high.tolist(),margin_um=margin,
        spacing_zyx_um=sp.tolist(),output_origin_zyx_um=origin.tolist(),
        retained_volume_fraction=float(np.prod(newshape[1:])/np.prod(shape[1:])),
        raw_removed_intensity_fraction=totals['raw_removed']/max(1,totals['raw_total']),
        mask_foreground_preserved=True,verified_sha256=hashes,
        uncertain_pairs=sum(x['uncertain'] for x in prior['transitions']),
        outputs={k:str(p) for k,p in paths.items()},gif=str(gif),backup=str(backup),passed=True)
    save_json(root/'registered/crop_reports'/f'{base}.json',result)
    print(json.dumps({k:result[k] for k in ['sample','old_shape','output_shape','retained_volume_fraction','raw_removed_intensity_fraction']}),flush=True)
    return result

def crop_existing(root,workers,margin):
    import tempfile
    root=Path(root).resolve()
    reports=root/'registered/crop_reports'
    assert not reports.exists(),'Existing crop reports: refusing accidental repeated crop'
    reports.mkdir()
    backup=Path(tempfile.mkdtemp(prefix='.before_fov_crop_',dir=root))
    bases=sorted(p.name.removesuffix('_shape_postprocessed.tif') for p in (root/'chrp3d_post_affine').glob('*_shape_postprocessed.tif'))
    assert bases
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(crop_registered_movie,[(str(root),b,str(backup),margin) for b in bases]))
    save_json(root/'registered/CROP_SUMMARY.json',dict(samples=len(rows),frames=sum(r['output_shape'][0] for r in rows),
        backup=str(backup),margin_um=margin,mask_foreground_preserved=all(r['mask_foreground_preserved'] for r in rows),
        note='Original reports/transforms/VALIDATION describe pre-crop canvas; crop_reports define current origin, shape and hashes.',
        median_retained_volume_fraction=float(np.median([r['retained_volume_fraction'] for r in rows])),
        max_raw_removed_intensity_fraction=max(r['raw_removed_intensity_fraction'] for r in rows)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--raw',type=Path,help='Raw *-DAPI.tif file or directory for image-first registration')
    p.add_argument('--preprocessed',type=Path,help='Matching preprocessed TIFF file or directory')
    p.add_argument('--anchor-t',type=int,help='Zero-based anchor time; defaults to the middle frame')
    p.add_argument('--report',type=Path,help='Transforms JSON path for a single movie')
    p.add_argument('--legacy-mask-assisted',action='store_true',help='Reproduce old mask-assisted post-segmentation registration')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--sample');p.add_argument('--self-test',action='store_true')
    p.add_argument('--crop-existing',action='store_true',help='Crop current raw_pre_affine/tiff and chrp3d_post_affine in place, with backup, and refresh GIFs')
    p.add_argument('--crop-margin-um',type=float,default=2.)
    args=p.parse_args()
    if args.crop_existing:
        if not args.root or args.crop_margin_um<0:p.error('crop requires --root and nonnegative margin')
        crop_existing(args.root,args.workers,args.crop_margin_um);return
    if args.self_test:self_test();return
    if args.workers<1:p.error('--workers must be positive')
    if not args.output:p.error('--output is required')
    out=args.output.resolve()
    if args.legacy_mask_assisted:
        if not args.root or args.raw or args.preprocessed:
            p.error('--legacy-mask-assisted requires --root and no --raw/--preprocessed')
        root=args.root.resolve()
        if out==root:p.error('--output must differ from --root')
        for name in ['raw_preprocessed','chrp3d','reports','transforms','qc']:(out/name).mkdir(parents=True,exist_ok=True)
        bases=sorted(f.name.removesuffix('_shape_postprocessed.tif') for f in (root/'chrp3d_postprocessed').glob('*_shape_postprocessed.tif'))
        if args.sample:bases=[b for b in bases if b==args.sample]
        if not bases:p.error('No postprocessed masks matched')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_movie,(str(root),str(out),b)) for b in bases]
            rows=[f.result() for f in as_completed(futures)]
        summary=dict(version=VERSION,files=len(rows),frames=sum(len(r['frames']) for r in rows),passed=all(r['passed'] for r in rows),
            missing_label_events=sum(len(f['missing_ids_due_to_resampling']) for r in rows for f in r['frames']),
            uncertain_pairs=sum(t['uncertain'] for r in rows for t in r['transitions']),
            source_mask='chrp3d_postprocessed',transform='rigid, adjacent composition to representative anchor; union padded field',
            methods={m:sum(t['selected']==m for r in rows for t in r['transitions']) for m in ['identity','old_translation','translation_3d','rigid_3d']})
        save_json(out/'SUMMARY.json',summary);print(json.dumps(summary),flush=True)
        return

    if args.root and not args.raw and not args.preprocessed:
        root=args.root.resolve();raw_root=root/'raw';pre_root=root/'raw_preprocessed'
    elif args.raw and args.preprocessed and not args.root:
        raw_root=args.raw.resolve();pre_root=args.preprocessed.resolve()
    else:
        p.error('Provide --raw and --preprocessed, or --root')
    if out in (raw_root,pre_root):p.error('--output must differ from both input locations')
    files=[raw_root] if raw_root.is_file() else sorted(raw_root.glob('*-DAPI.tif'))
    if args.sample:files=[f for f in files if f.name.removesuffix('-DAPI.tif')==args.sample]
    if not files:p.error('No raw *-DAPI.tif files matched')
    if args.report and len(files)!=1:p.error('--report requires a single input movie')
    if pre_root.is_file() and len(files)!=1:p.error('One preprocessed TIFF cannot match multiple raw movies')
    tasks=[(raw,pre_root if pre_root.is_file() else pre_root/raw.name,out,args.anchor_t,args.report)
           for raw in files]
    if args.workers>1 and len(tasks)>1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows=list(pool.map(_run_image_task,tasks))
    else:
        rows=[run_image_movie(*task) for task in tasks]
    summary=dict(version=IMAGE_VERSION,files=len(rows),frames=sum(row['source_shape_tzyx'][0] for row in rows),
                 output=str(out),mode='image_only_rigid')
    save_json(out/'SUMMARY.json',summary)
    print(json.dumps(summary),flush=True)

def _run_image_task(task):
    return run_image_movie(*task)

if __name__=='__main__':main()
