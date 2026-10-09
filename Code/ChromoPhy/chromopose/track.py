#!/usr/bin/env python3
"""Motion/shape-aware, anchor-centred chromosome candidate tracking.

Identity is NOT a per-frame connected component. One identity may have several
fragments; a compound observation may contain several latent identities. Masks
are motion-compensated before matching, with global one-to-one assignment,
local collective motion, shape/volume regularization and two-frame gap memory.
An anchor is a soft identity origin, not a 46/92 count constraint. Compound
partitions are explicitly uncertain hypotheses, NOT recovered ground truth.
Optional DAPI-guided temporal refinement reconnects reliable tracklets, selects
bidirectionally supported compound partitions, absorbs minor fragments and
quarantines low-support specks. No foreground is added; source files are read-only.
Use --raw RAW_TIFF_ROOT --initial-tracks TRACKED_ROOT to refine existing tracking.

Dependencies: numpy, scipy, tifffile, scikit-image (DAPI-guided refinement).
GIF rendering is imported separately. A final regression guard selects unchanged
initial tracking if short-track count increases or mean continuity deteriorates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

BASE_VERSION = "motion-shape-latent-tracking-2.1"
VERSION = "motion-shape-temporal-refinement-3.2"
CATEGORIES = ("chr3", "chr18", "others")
MAX_GAP = 2
POINTS = 384


def _stem(path):
    return path.stem.removesuffix("_shape_postprocessed").removesuffix("_shape")


@dataclass
class Observation:
    label: int
    volume: int
    center: np.ndarray
    cloud: np.ndarray
    shape: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


@dataclass
class State:
    identity: int
    volume: float
    center: np.ndarray
    cloud: np.ndarray
    shape: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    missing: int = 0
    status: str = "single"


def _shape(cloud, scale):
    q = (cloud - cloud.mean(axis=0)) * scale
    covariance = q.T @ q / max(1, len(q))
    eig = np.maximum(np.linalg.eigvalsh(covariance), .25)
    # Physical principal extents encode compact vs elongated/bent observations.
    return np.sqrt(eig)


def _objects(frame, scale):
    result = {}
    for label, sl in enumerate(ndi.find_objects(frame), 1):
        if sl is None:
            continue
        xyz = np.argwhere(frame[sl] == label)
        if not len(xyz):
            continue
        origin = np.array([s.start for s in sl])
        center = xyz.mean(axis=0) + origin
        # Deterministic dispersed sampling, not merely the first Z plane.
        points = xyz[np.linspace(0, len(xyz)-1, min(POINTS, len(xyz)), dtype=int)] + origin
        result[label] = Observation(label, len(xyz), center, points.astype(float),
            _shape(points, scale), origin, np.array([s.stop-1 for s in sl]))
    return result


def _density(frame, scale):
    steps = np.maximum(1, np.rint(4 / scale).astype(int))
    # Max pooling preserves thin foreground that stride sampling would miss.
    work = np.asarray(frame > 0, dtype=np.float32)
    work = ndi.maximum_filter(work, size=tuple(steps), mode="constant")
    work = work[tuple(slice(None, None, int(s)) for s in steps)]
    return ndi.gaussian_filter(work, .8), steps


def _collective_shift(previous, current, scale):
    if not previous.any() or not current.any():
        return np.zeros(3)
    # Cross correlation of pooled foreground, rather than arbitrary source IDs.
    from scipy.signal import fftconvolve
    correlation = fftconvolve(current, previous[::-1, ::-1, ::-1], mode="full")
    origin = np.array(previous.shape)-1
    # Restrict search to 80 equivalent XY voxels; reject wrap-around solutions.
    radius = np.ceil(80 / scale).astype(int)
    lo = np.maximum(0, origin-radius); hi = np.minimum(correlation.shape, origin+radius+1)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    peak = np.array(np.unravel_index(np.argmax(correlation[sl]), correlation[sl].shape)) + lo
    return (peak-origin).astype(float)


def _votes(cloud, frame, scale, radius=3.):
    """Fraction of predicted cloud supported by each nearby observation."""
    xyz = np.rint(cloud).astype(int)
    votes = defaultdict(set)
    offsets = [(0, 0, 0)]
    for axis in range(3):
        step = int(math.floor(radius/scale[axis]))
        if step:
            for sign in (-1, 1):
                offset = [0, 0, 0]; offset[axis] = sign*step; offsets.append(offset)
    bounds = np.array(frame.shape)
    for offset in offsets:
        p = xyz + offset
        valid = np.flatnonzero(np.all((p >= 0) & (p < bounds), axis=1))
        if not len(valid):
            continue
        labels = frame[tuple(p[valid].T)]
        for label in np.unique(labels):
            if label:
                votes[int(label)].update(valid[labels == label].tolist())
    return {k: len(v)/max(1, len(cloud)) for k, v in votes.items()}


def _costs(states, observations, shifts, frame, scale):
    n, m = len(states), len(observations)
    cost = np.full((n, m), 10.)
    support = np.zeros((n, m))
    if not n or not m:
        return cost, support, np.empty((n, m))
    predicted = np.array([s.center+d for s, d in zip(states, shifts)])
    dist = cdist(predicted*scale, np.array([o.center for o in observations])*scale)
    index = {o.label: j for j, o in enumerate(observations)}
    for i, (s, displacement) in enumerate(zip(states, shifts)):
        for label, fraction in _votes(s.cloud+displacement, frame, scale).items():
            support[i, index[label]] = fraction
        gate = min(65., max(22., 12.+2*s.shape[-1])) + 12*s.missing
        for j, o in enumerate(observations):
            ratio = o.volume/max(1., s.volume)
            compound_supported = ratio > 2.5 and support[i, j] >= .45
            if not compound_supported and (dist[i, j] > gate or not .12 <= ratio <= 6):
                continue
            volume_error = min(1.5, abs(math.log(ratio))) / 1.5
            shape_error = min(1.5, float(np.linalg.norm(np.log((o.shape+.5)/(s.shape+.5)))/math.sqrt(3))) / 1.5
            motion_error = min(.4, dist[i, j]/gate) if compound_supported else dist[i, j]/gate
            cost[i, j] = (.34*(1-support[i, j]) + .34*motion_error
                          + .18*volume_error + .14*shape_error + .025*s.missing)
    return cost, support, dist


def _assignment(cost):
    if not cost.size:
        return {}
    n, m = cost.shape
    # A private unmatched column per state avoids forced associations.
    augmented = np.concatenate([cost, np.full((n, n), .70)], axis=1)
    rows, cols = linear_sum_assignment(augmented)
    return {int(j): int(i) for i, j in zip(rows, cols) if j < m and cost[i, j] < .70}


def _local_motion(states, observations, assignment, costs, support, global_shift, scale):
    centers, displacements = [], []
    for j, i in assignment.items():
        s, o = states[i], observations[j]
        if s.missing or s.status != "single" or not .55 < o.volume/s.volume < 1.8:
            continue
        row = np.sort(costs[i]); margin = row[1]-row[0] if len(row)>1 else 1.
        if (support[i, j] >= .25 and margin > .025) or costs[i, j] < .32:
            centers.append(s.center); displacements.append(o.center-s.center)
    if len(centers) < 4:
        return np.array([global_shift+.25*(s.velocity-global_shift)
            if np.any(s.velocity) else global_shift for s in states]), len(centers)
    centers, displacements = np.array(centers), np.array(displacements)
    tree = cKDTree(centers*scale)
    result = []
    for s in states:
        distance, indices = tree.query(s.center*scale, k=min(7, len(centers)))
        local = displacements[np.atleast_1d(indices)]
        median = np.median(local, axis=0)
        residual = np.linalg.norm((local-median)*scale, axis=1)
        keep = residual <= max(10., 2.5*np.median(residual))
        weights = 1/(np.atleast_1d(distance)[keep]+8)
        field_shift = np.average(local[keep], axis=0, weights=weights)
        # History residual is relative to collective motion, not double-counted.
        residual_velocity = s.velocity-global_shift
        length = np.linalg.norm(residual_velocity*scale)
        if length > 18:
            residual_velocity *= 18/length
        result.append(.85*field_shift+.15*(global_shift+residual_velocity))
    return np.asarray(result), len(centers)


def _partition(o, identities, predictions, source, destination, scale):
    sl = tuple(slice(int(a), int(b)+1) for a, b in zip(o.lo, o.hi))
    binary = source[sl] == o.label
    xyz = np.argwhere(binary); absolute = xyz+o.lo
    distances = []
    for identity in identities:
        cloud = predictions[identity]
        distance = cKDTree(cloud*scale).query(absolute*scale)[0]
        field = np.zeros(binary.shape, np.float32)
        field[tuple(xyz.T)] = distance
        # Normalized smoothing avoids background pulling the interface outward.
        sig = np.maximum(.12, .8/scale)
        smooth = ndi.gaussian_filter(field, sig) / np.maximum(.001, ndi.gaussian_filter(binary.astype(float), sig))
        distances.append(smooth[tuple(xyz.T)])
    winner = np.argmin(np.array(distances), axis=0)
    # Every retained hypothesis gets a seed. Otherwise it remains latent only.
    present = set(np.unique(winner).tolist())
    kept = [identities[i] for i in sorted(present)]
    destination[sl][tuple(xyz.T)] = np.array(identities, np.uint32)[winner]
    return kept


def _anchor_ids(objects, source, anchor_t, all_objects, shifts, scale):
    ordered = sorted(objects.values(), key=lambda o: (*o.center, -o.volume))
    identities = {o.label: i+1 for i, o in enumerate(ordered)}
    # Tiny enclosed/adjacent fragments require agreement on BOTH neighbouring
    # frames; two substantial anchor observations are never proactively joined.
    if not (0 < anchor_t < len(all_objects)-1) or not ordered:
        return identities, []
    med = np.median([o.volume for o in ordered]); summaries = {}
    for o in ordered:
        summaries[o.label] = [
            _votes(o.cloud-shifts[anchor_t-1], source[anchor_t-1], scale),
            _votes(o.cloud+shifts[anchor_t], source[anchor_t+1], scale)]
    changes=[]; absorbed=Counter()
    for o in sorted(ordered, key=lambda o:o.volume):
        if o.volume > .30*med:
            continue
        options=[]
        for parent in ordered:
            if parent.volume < 4*o.volume or parent.volume < .6*med:
                continue
            gap = np.maximum(0, np.maximum(parent.lo-o.hi, o.lo-parent.hi)-1)*scale
            if np.linalg.norm(gap)>6 or absorbed[parent.label]+o.volume>.35*parent.volume:
                continue
            score=1.
            for v, p in zip(summaries[o.label], summaries[parent.label]):
                common=[min(v[k],p[k]) for k in v.keys() & p.keys()]
                score=min(score,max(common,default=0))
            if score >= .4:
                options.append((score,parent.label))
        if options:
            score, parent=max(options)
            identities[o.label]=identities[parent]; absorbed[parent]+=o.volume
            changes.append(dict(type="anchor_fragment_absorption",frame=anchor_t,
                source=o.label,parent_source=parent,track=identities[parent],support=score))
    return identities, changes


def _step(states, observations, source, global_shift, scale, next_id, t, direction):
    ss=sorted(states.values(),key=lambda s:s.identity); oo=list(observations.values())
    first=np.array([global_shift+.25*(s.velocity-global_shift)
        if np.any(s.velocity) else global_shift for s in ss])
    costs,support,_=_costs(ss,oo,first,source,scale)
    rough=_assignment(costs)
    displacement, trusted=_local_motion(ss,oo,rough,costs,support,global_shift,scale)
    costs,support,distance=_costs(ss,oo,displacement,source,scale)
    match=_assignment(costs)
    ownership={j:[ss[i].identity] for j,i in match.items()}
    index={s.identity:i for i,s in enumerate(ss)}
    assigned=set(match.values()); events=[]
    # A large observation can preserve unassigned identities, rather than erase
    # all but its best parent. Require cloud support and plausible volume sum.
    for j,i in sorted(match.items(),key=lambda ji:oo[ji[0]].volume,reverse=True):
        o=oo[j]; ids=ownership[j]; total=ss[i].volume
        candidates=sorted((k for k in range(len(ss)) if k not in assigned
            and support[k,j]>=.22 and (support[k,j]>=.45
                or distance[k,j]<min(50.,12+2*ss[k].shape[-1]))),
            key=lambda k:support[k,j],reverse=True)
        proposed=[]; proposed_total=total
        for k in candidates:
            combined=proposed_total+ss[k].volume
            if combined > o.volume/.62 or o.volume < 1.2*max(ss[k].volume,ss[i].volume):
                continue
            proposed.append(k);proposed_total=combined
        if proposed and .62*proposed_total <= o.volume <= 1.65*proposed_total:
            for k in proposed:
                ids.append(ss[k].identity);assigned.add(k)
            total=proposed_total
        if len(ids)>1:
            events.append(dict(type="compound_observation",frame=t,direction=direction,
                source=o.label,identities=ids.copy(),uncertain_partition=True))
    # Recover multiple mask fragments of ONE identity only with spatial support,
    # volume conservation and a margin against competing identities.
    volumes=Counter()
    for j,ids in ownership.items():
        if len(ids)==1:volumes[ids[0]]+=oo[j].volume
    for j,o in sorted(enumerate(oo),key=lambda jo:jo[1].volume):
        if j in ownership:continue
        options=[]
        for identity,total in volumes.items():
            i=index[identity]; s=ss[i]
            if o.volume>.6*s.volume or total+o.volume>1.65*s.volume:
                continue
            if support[i,j]<.20 or distance[i,j]>min(35.,10+1.8*s.shape[-1]):continue
            competitor=max((support[k,j] for k in range(len(ss)) if k!=i),default=0)
            if support[i,j]-competitor<.10:continue
            options.append((support[i,j]-.01*distance[i,j],identity))
        if options:
            _,identity=max(options);ownership[j]=[identity];volumes[identity]+=o.volume
            events.append(dict(type="fragment_absorption",frame=t,direction=direction,
                source=o.label,track=identity,support=float(support[index[identity],j])))
    for j,o in enumerate(oo):
        if j not in ownership:
            ownership[j]=[next_id]
            events.append(dict(type="unresolved_new_candidate",frame=t,direction=direction,
                source=o.label,track=next_id,
                note="Not proof of biological birth; may be residual segmentation error"))
            next_id+=1
    output=np.zeros(source.shape,np.uint32)
    lut=np.zeros(int(source.max(initial=0))+1,np.uint32)
    predictions={s.identity:s.cloud+d for s,d in zip(ss,displacement)}
    for j,ids in ownership.items():lut[oo[j].label]=ids[0]
    output[:]=lut[source]
    compound_ids=set()
    for j,ids in ownership.items():
        if len(ids)>1:
            kept=_partition(oo[j],ids,predictions,source,output,scale)
            compound_ids.update(ids)
            if len(kept)<len(ids):
                events.append(dict(type="latent_identity_not_visible",frame=t,
                    identities=sorted(set(ids)-set(kept)),source=oo[j].label))
    new_objects=_objects(output,scale)
    next_states={}; fragment_ids={identity for identity,n in
        Counter(identity for ids in ownership.values() if len(ids)==1 for identity in ids).items() if n>1}
    for identity,o in new_objects.items():
        if identity in states:
            old=states[identity]; i=index[identity]
            status="compound" if identity in compound_ids else "fragment_group" if identity in fragment_ids else "single"
            if status=="compound":
                # Do not learn falsely narrow shapes from an uncertain partition.
                delta=.8*displacement[i]+.2*(o.center-old.center)
                new=State(identity,old.volume,old.center+delta,old.cloud+delta,
                    old.shape.copy(),.7*old.velocity+.3*delta,status=status)
            else:
                delta=o.center-old.center
                new=State(identity,.75*old.volume+.25*o.volume,o.center,o.cloud,
                    .75*old.shape+.25*o.shape,.55*old.velocity+.45*delta,status=status)
            if old.missing:
                events.append(dict(type="gap_reconnection",frame=t,direction=direction,
                    track=identity,missing_frames=old.missing))
            next_states[identity]=new
        else:
            next_states[identity]=State(identity,o.volume,o.center,o.cloud,o.shape)
    for identity,old in states.items():
        if identity in next_states:continue
        i=index[identity];delta=displacement[i]
        if identity in compound_ids or old.missing<MAX_GAP:
            next_states[identity]=State(identity,old.volume,old.center+delta,old.cloud+delta,
                old.shape,.7*old.velocity+.3*delta,
                0 if identity in compound_ids else old.missing+1,
                "latent_compound" if identity in compound_ids else "missing")
        else:
            events.append(dict(type="unresolved_track_end",frame=t,direction=direction,track=identity))
    relations=[]
    for j,ids in ownership.items():
        for identity in ids:
            i=index.get(identity)
            margin=None
            if i is not None:
                alternatives=[costs[i,k] for k in range(len(oo)) if k!=j]
                alternatives.extend(costs[k,j] for k in range(len(ss)) if k!=i)
                margin=float(min(alternatives,default=1.)-costs[i,j])
            ratio=float(oo[j].volume/ss[i].volume) if i is not None else None
            shape_change=float(np.linalg.norm(np.log((oo[j].shape+.5)/(ss[i].shape+.5)))/math.sqrt(3)) if i is not None else None
            relations.append(dict(frame=t,direction=direction,source_label=oo[j].label,
                track_id=identity,status="compound" if len(ids)>1 else
                "fragment_group" if identity in fragment_ids else "single",
                score=float(1-costs[i,j]) if i is not None else 0.,
                motion_compensated_support=float(support[i,j]) if i is not None else 0.,
                predicted_distance=float(distance[i,j]) if i is not None else None,
                trusted_motion_matches=trusted,partition_is_uncertain=len(ids)>1,
                competing_assignment_margin=margin,
                observed_to_reference_volume_ratio=ratio,shape_deviation=shape_change,
                association_uncertain=(i is None or len(ids)>1 or margin<.06
                    or costs[i,j]>.55 or support[i,j]<.15
                    or not .5 < ratio < 2.2 or shape_change>.9)))
    return output,next_states,next_id,events,relations


def _smooth(output, source_objects, anchor, shifts, scale, anchor_identity_set):
    # A short new ID is corrected only when BOTH temporal neighbours support
    # the SAME existing identity. Anchored identities cannot be overwritten.
    events=[]; snapshots=[_objects(f,scale) for f in output]
    presence=Counter(identity for objs in snapshots for identity in objs)
    volumes=defaultdict(list)
    for objs in snapshots:
        for identity,o in objs.items():volumes[identity].append(o.volume)
    med={k:float(np.median(v)) for k,v in volumes.items()}
    for t in range(1,len(output)-1):
        if t==anchor:continue
        proposals=[]
        for identity,o in snapshots[t].items():
            if identity in anchor_identity_set or presence[identity]>2:continue
            before=_votes(o.cloud-shifts[t-1],output[t-1],scale)
            after=_votes(o.cloud+shifts[t],output[t+1],scale)
            options=[]
            for target in before.keys() & after.keys():
                if target==identity or min(before[target],after[target])<.30:continue
                existing=snapshots[t].get(target)
                if existing and (o.volume>.55*med[target] or existing.volume+o.volume>1.65*med[target]):continue
                if not existing and not .35*med[target]<o.volume<2*med[target]:continue
                options.append((min(before[target],after[target]),target))
            if options:
                score,target=max(options);proposals.append((identity,target,score))
        absorbed=Counter()
        for identity,target,score in sorted(proposals,key=lambda p:p[2],reverse=True):
            o=snapshots[t][identity];existing=snapshots[t].get(target)
            if (existing.volume if existing else 0)+absorbed[target]+o.volume>1.65*med[target]:continue
            sl=tuple(slice(int(a),int(b)+1) for a,b in zip(o.lo,o.hi))
            patch=output[t][sl];patch[patch==identity]=target;absorbed[target]+=o.volume
            events.append(dict(type="bidirectional_identity_repair",frame=t,
                old_track=identity,track=target,agreement=score))
    return events


def _track_movie(arr,name,anchors,axis_scale=(2.,1.,1.)):
    if arr.ndim!=4 or not np.issubdtype(arr.dtype,np.integer):raise ValueError("Expected integer TZYX labels")
    if np.issubdtype(arr.dtype,np.signedinteger) and arr.min()<0:raise ValueError("Negative source label")
    scale=np.asarray(axis_scale,float)
    config=anchors.get(name,{})
    anchor=max(0,min(len(arr)-1,int(config.get("representative_t",config.get("window_start",0)))))
    objects=[_objects(f,scale) for f in arr]
    densities=[];steps=None
    for f in arr:
        density,steps=_density(f,scale);densities.append(density)
    shifts=[_collective_shift(a,b,scale*steps)*steps for a,b in zip(densities,densities[1:])]
    del densities
    anchor_map,events=_anchor_ids(objects[anchor],arr,anchor,objects,shifts,scale)
    next_id=max(anchor_map.values(),default=0)+1
    output=np.empty(arr.shape,np.uint32);lut=np.zeros(int(arr[anchor].max(initial=0))+1,np.uint32)
    for label,identity in anchor_map.items():lut[label]=identity
    output[anchor]=lut[arr[anchor]]
    anchor_objects=_objects(output[anchor],scale);relations=[]
    for direction,indices in [("forward",range(anchor+1,len(arr))),("backward",range(anchor-1,-1,-1))]:
        states={identity:State(identity,o.volume,o.center.copy(),o.cloud.copy(),o.shape.copy())
                for identity,o in anchor_objects.items()}
        for t in indices:
            shift=shifts[t-1] if direction=="forward" else -shifts[t]
            output[t],states,next_id,ev,rel=_step(states,objects[t],arr[t],shift,scale,next_id,t,direction)
            events.extend(ev);relations.extend(rel)
    events.extend(_smooth(output,objects,anchor,shifts,scale,set(anchor_map.values())))
    nodes=[];per_frame=[];lengths=Counter();mapping_rows=[]
    for t,(source,tracked) in enumerate(zip(arr,output)):
        if not np.array_equal(source>0,tracked>0):raise AssertionError(f"Foreground changed at t={t}")
        tracked_objects=_objects(tracked,scale)
        for identity,o in tracked_objects.items():
            lengths[identity]+=1
            nodes.append(dict(movie=name,t=t,track_id=identity,volume_voxels=o.volume,
                centroid_z=float(o.center[0]),centroid_y=float(o.center[1]),centroid_x=float(o.center[2]),
                anchor_frame=t==anchor,candidate_identity=True))
        for label,o in objects[t].items():
            sl=tuple(slice(int(a),int(b)+1) for a,b in zip(o.lo,o.hi))
            ids,counts=np.unique(tracked[sl][source[sl]==label],return_counts=True)
            for identity,volume in zip(ids,counts):
                mapping_rows.append(dict(t=t,source_label=label,track_id=int(identity),
                    voxels=int(volume),fraction=float(volume/o.volume),compound_partition=len(ids)>1))
        per_frame.append(dict(t=t,source_instances=len(objects[t]),tracked_instances=len(tracked_objects),
            foreground_voxels=sum(o.volume for o in objects[t].values())))
    vals=list(lengths.values())
    report=dict(version=BASE_VERSION,movie=name,shape=list(arr.shape),anchor_frame=anchor,
        anchor_source_instances=len(objects[anchor]),anchor_candidate_identities=len(anchor_objects),
        source_instances=sum(map(len,objects)),track_segments=len(lengths),allocated_ids=next_id-1,
        distance_axis_scale=scale.tolist(),per_frame=per_frame,
        foreground_preserved=True,instance_counts_preserved=False,
        source_mask_files_modified=False,partitions_are_uncertain_hypotheses=True,
        relations_stage="initial propagation; source_mapping table is the final smoothed assignment",
        hard_chromosome_count_constraint=False,max_missing_frames=MAX_GAP,
        frame_interval_policy="frame units; acquisition intervals unavailable to this mask-only tracker",
        collective_shifts_zyx=[s.tolist() for s in shifts],
        track_length=dict(mean=float(np.mean(vals)) if vals else 0,median=float(np.median(vals)) if vals else 0,
            one_frame=sum(v==1 for v in vals),one_frame_fraction=sum(v==1 for v in vals)/max(1,len(vals))),
        events=dict(Counter(e["type"] for e in events)))
    return output,report,nodes,relations+events,mapping_rows


def _ownership(source, tracked, objects):
    result={}
    for label,o in objects.items():
        sl=tuple(slice(int(a),int(b)+1) for a,b in zip(o.lo,o.hi))
        ids,counts=np.unique(tracked[sl][source[sl]==label],return_counts=True)
        result[label]={int(i):int(n) for i,n in zip(ids,counts) if i}
    return result


def _signal_stats(raw, foreground):
    step=(max(1,len(raw)//32),4,4)
    sl=tuple(slice(None,None,k) for k in step)
    values=np.asarray(raw[sl],dtype=float);mask=foreground[sl]>0
    fg=values[mask];bg=values[~mask]
    median=float(np.median(fg)) if len(fg) else 1.
    baseline=float(np.median(bg)) if len(bg) else 0.
    noise=max(1.,.03*max(1.,median-baseline),
              float(1.4826*np.median(np.abs(bg-baseline))) if len(bg) else 1.)
    return baseline,noise,max(1.,median-baseline)


def _sample_signal(obj,raw,stats):
    points=np.rint(obj.cloud).astype(int)
    values=np.asarray(raw[tuple(points.T)],dtype=float)
    bg,noise,reference=stats
    median=float(np.median(values))
    return dict(median=median,snr=(median-bg)/noise,relative=(median-bg)/reference)


def _transient_cleanup(initial,source,raw,scale,spacing,base_report,max_frames=3):
    """Whole-track quarantine without size gates or deletion budgets.

    Short IDs alone are not evidence of false foreground. Test motion-compensated
    DAPI persistence and plausible endpoint/parent tracks. Boundary-censored IDs
    are retained. Every discarded voxel is recoverable, including large patches.
    """
    if initial.shape!=source.shape or raw.shape!=source.shape:raise ValueError('Cleanup grids differ')
    snapshots=[_objects(frame,scale) for frame in initial]
    histories=defaultdict(dict)
    for t,objects in enumerate(snapshots):
        for identity,o in objects.items():histories[identity][t]=o
    shifts=np.asarray(base_report['collective_shifts_zyx'],float).reshape(-1,3)
    cumulative=np.vstack([np.zeros(3),np.cumsum(shifts,axis=0)])
    stats=[_signal_stats(raw[t],source[t]) for t in range(len(source))]
    output=initial.copy();events=[];quarantine=[];audit=[];mips={}
    offsets=[np.zeros(3)]
    # Local physical tolerance is independent of voxel anisotropy.
    for axis in range(3):
        for sign in (-1,1):
            delta=np.zeros(3);delta[axis]=sign*.45/spacing[axis];offsets.append(delta)
    frame_shape=np.array(source.shape[1:])
    candidates={identity for identity,poses in histories.items() if len(poses)<=max_frames
                and min(poses)>0 and max(poses)<len(source)-1
                and max(poses)-min(poses)+1==len(poses)}
    parent={identity:identity for identity in candidates}
    def root(identity):
        while parent[identity]!=identity:identity=parent[identity]
        return identity
    # Treat label changes within the same brief patch as one transient episode.
    # Otherwise one doomed short ID can spuriously protect its short-lived neighbour.
    for identity in sorted(candidates):
        poses=histories[identity]
        for t,q in ((min(poses),min(poses)-1),(max(poses),max(poses)+1)):
            cloud=poses[t].cloud+cumulative[q]-cumulative[t]
            for other,support in _votes(cloud,initial[q],scale).items():
                if other not in candidates or support<.50:continue
                left,right=root(identity),root(other)
                if left==right:continue
                ids=[i for i in candidates if root(i) in (left,right)]
                times={t for i in ids for t in histories[i]}
                if max(times)-min(times)+1<=max_frames:parent[right]=left
    episodes=defaultdict(list)
    for identity in candidates:episodes[root(identity)].append(identity)
    for identity,episode in sorted(episodes.items()):
        poses={}
        for t in sorted({t for i in episode for t in histories[i]}):
            observations=[histories[i][t] for i in episode if t in histories[i]]
            volume=sum(o.volume for o in observations)
            cloud=np.concatenate([o.cloud[np.linspace(0,len(o.cloud)-1,max(1,round(POINTS*o.volume/volume)),dtype=int)]
                                  for o in observations])
            poses[t]=Observation(identity,volume,sum(o.center*o.volume for o in observations)/volume,
                                 cloud,_shape(cloud,scale),np.min([o.lo for o in observations],axis=0),
                                 np.max([o.hi for o in observations],axis=0))
        times=sorted(poses)
        if len(times)>max_frames or times[0]==0 or times[-1]==len(source)-1:continue
        # Disjoint appearances are evidence against a purely transient artifact.
        if times[-1]-times[0]+1!=len(times):continue
        a,b=times[0],times[-1]
        residual=(poses[b].center-poses[a].center-cumulative[b]+cumulative[a])/max(1,b-a)
        # Bound extrapolation to observed collective/local displacement, not arbitrary jumps.
        if np.linalg.norm(residual*spacing)>2.:residual*=2./np.linalg.norm(residual*spacing)
        current=[_sample_signal(o,raw[t],stats[t]) for t,o in poses.items()]
        weak=float(np.median([s['relative'] for s in current]))<.30 or float(np.median([s['snr'] for s in current]))<4
        temporal=[];axial=[];continuation=[]
        neighbours=[(q,t) for q,t in ((a-1,a),(a-2,a),(b+1,b),(b+2,b)) if 0<=q<len(source)]
        for q,t in neighbours:
            o=poses[t];cloud=o.cloud+cumulative[q]-cumulative[t]+residual*(q-t)
            movers=[(np.linalg.norm((near.center-o.center)*spacing),
                     snapshots[q][i].center-near.center)
                    for i,near in snapshots[t].items() if i in snapshots[q]
                    and i not in episode and len(histories[i])>max_frames]
            movers=sorted(movers,key=lambda item:item[0])[:8]
            movers=[delta for distance,delta in movers if distance<10.]
            clouds=[cloud]
            if len(movers)>=3:clouds.append(o.cloud+np.median(movers,axis=0))
            # Per-point maximum over a small physical neighbourhood tolerates blur/motion.
            values=np.zeros(len(cloud),float);valid=np.zeros(len(cloud),bool)
            for predicted in clouds:
                for offset in offsets:
                    points=predicted+offset;inside=np.all((points>=0)&(points<=frame_shape-1),axis=1)
                    valid|=inside
                    sampled=ndi.map_coordinates(np.asarray(raw[q]),points.T,order=1,mode='constant',cval=0,prefilter=False)
                    values=np.maximum(values,sampled)
            bg,noise,reference=stats[q];threshold=bg+max(3*noise,.25*reference)
            coverage=float(np.mean(values[valid]>=threshold)) if valid.any() else 1.
            temporal.append(coverage)
            # A Z excursion/weak axial focus is not disappearance of the object.
            if q not in mips:mips[q]=np.asarray(raw[q]).max(axis=0)
            xy_values=np.zeros(len(cloud),float)
            for predicted in clouds:
                for offset in offsets:
                    points=(predicted+offset)[:,1:]
                    sampled=ndi.map_coordinates(mips[q],points.T,order=1,mode='constant',cval=0,prefilter=False)
                    xy_values=np.maximum(xy_values,sampled)
            axial.append(float(np.mean(xy_values>=threshold)))
            # Protect endpoint changes and attachment to a persistent parent, even if DAPI is weak.
            best=0.;target=None
            for other,neighbour in snapshots[q].items():
                if other in episode or len(histories[other])<=max_frames:continue
                distance=min(np.linalg.norm((neighbour.center-predicted.mean(axis=0))*spacing) for predicted in clouds)
                if distance>max(2.,3*neighbour.shape[-1]*math.sqrt(spacing[1]*spacing[2])):continue
                tree=cKDTree(neighbour.cloud*spacing)
                nearest=np.min([tree.query(predicted*spacing)[0] for predicted in clouds],axis=0)
                support=float(np.mean(nearest<.65))
                if support>best:best=support;target=other
            continuation.append(dict(frame=q,support=best,track=target))
        absent=max(temporal)<.20 and max(axial)<.35
        unsupported=max(c['support'] for c in continuation)<.35
        # Strong isolated 1-2 frame patches can be artifacts too; 3-frame strong objects are protected.
        discard=absent and unsupported and (len(times)<=2 or weak)
        reason='transient_no_image_or_track_continuation' if discard else 'retained_image_or_track_support'
        if len(times)==3 and not weak:reason='retained_strong_three_frame_track'
        row=dict(track=identity,episode_tracks=sorted(episode),frames=times,voxels=sum(o.volume for o in poses.values()),
                 weak_dapi=weak,relative_dapi=float(np.median([s['relative'] for s in current])),
                 neighbour_dapi_coverage=temporal,neighbour_axial_mip_coverage=axial,
                 continuation=continuation,removed=discard,reason=reason)
        audit.append(row)
        if not discard:continue
        for t,o in poses.items():
            sl=tuple(slice(int(lo),int(hi)+1) for lo,hi in zip(o.lo,o.hi));binary=np.isin(output[t][sl],episode)
            xyz=np.argwhere(binary)+o.lo
            quarantine.append(np.column_stack([np.full(len(xyz),t),xyz,output[t][tuple(xyz.T)],source[t][tuple(xyz.T)]]))
            output[t][sl][binary]=0
        events.append(dict(type='transient_track_quarantined',**row))
    removed=np.concatenate(quarantine).astype(np.uint32) if quarantine else np.empty((0,6),np.uint32)
    return output,events,removed,audit


def _templates(snapshots,source,ownership,raw,scale,cumulative,clock):
    volumes=[o.volume for objects in snapshots for o in objects.values()]
    vref=float(np.median(volumes)) if volumes else 1.
    histories=defaultdict(dict);signals=[]
    for t,objects in enumerate(snapshots):
        stats=_signal_stats(raw[t],source[t]);frame_signals={}
        for identity,o in objects.items():
            signal=_sample_signal(o,raw[t],stats);frame_signals[identity]=signal
            labels=source[t][tuple(np.rint(o.cloud).astype(int).T)]
            labels,counts=np.unique(labels[labels>0],return_counts=True)
            if not len(labels):continue
            label=int(labels[np.argmax(counts)])
            source_n=sum(ownership[t][label].values())
            if counts.max()/max(1,len(o.cloud))>=.85 and ownership[t][label].get(identity,0)/max(1,source_n)>=.85:
                histories[identity][t]=o
        signals.append(frame_signals)
    reliable={}
    for identity,poses in histories.items():
        reference=float(np.median([o.volume for o in poses.values()]))
        poses={t:o for t,o in poses.items() if .55*reference<=o.volume<=1.8*reference
               and signals[t][identity]['relative']>=.20}
        if len(poses)<2 or reference<.08*vref:continue
        ordered=sorted(poses);stable=0
        for a,b in zip(ordered,ordered[1:]):
            oa,ob=poses[a],poses[b]
            distance=np.linalg.norm((ob.center-oa.center-cumulative[b]+cumulative[a])*scale)
            deviation=np.linalg.norm(np.log((ob.shape+.5)/(oa.shape+.5)))/math.sqrt(3)
            if b-a<=3 and distance<max(20.,2*oa.shape[-1]) and deviation<.65:stable+=1
        if stable:
            reliable[identity]=dict(poses=poses,volume=reference,width=max(1.5,float(np.median([o.shape[0] for o in poses.values()]))))
    return reliable,signals,vref


def _residual_velocity(poses,t,cumulative,clock):
    neighbours=sorted(poses,key=lambda q:abs(q-t))[:3]
    velocities=[]
    for a,b in zip(sorted(neighbours),sorted(neighbours)[1:]):
        dt=clock[b]-clock[a]
        if dt>0:velocities.append((poses[b].center-poses[a].center-cumulative[b]+cumulative[a])/dt)
    return np.median(velocities,axis=0) if velocities else np.zeros(3)


def _predict(template,t,cumulative,clock,window=6):
    poses=template['poses'];before=[q for q in poses if q<=t];after=[q for q in poses if q>=t]
    a=max(before) if before else None;b=min(after) if after else None
    if a==t:
        nearby=sorted((q for q in poses if q!=t and abs(q-t)<=window),key=lambda q:abs(q-t))[:2]
        weights=[(t,.5)]+[(q,.5/len(nearby)) for q in nearby] if nearby else [(t,1.)]
        clouds=[];centers=[]
        for q,w in weights:
            delta=cumulative[t]-cumulative[q]+_residual_velocity(poses,q,cumulative,clock)*(clock[t]-clock[q])
            clouds.append((w,poses[q].cloud+delta));centers.append(w*(poses[q].center+delta))
        return dict(clouds=clouds,center=sum(centers),paired=any(q<t for q in nearby) and any(q>t for q in nearby),
                    multi_pose=bool(nearby),gap=max(abs(t-q) for q,_ in weights),volume=template['volume'],width=template['width'])
    if a is not None and t-a>window:a=None
    if b is not None and b-t>window:b=None
    if a is None and b is None:return None
    if a is None or b is None:
        q=b if a is None else a
        if abs(q-t)>min(3,window):return None
        nearby=sorted((k for k in poses if k!=q and abs(k-t)<=window),key=lambda k:abs(k-t))[:1]
        weights=[(q,.65),(nearby[0],.35)] if nearby else [(q,1.)]
    else:
        span=clock[b]-clock[a]
        weights=[(a,(clock[b]-clock[t])/span),(b,(clock[t]-clock[a])/span)]
    clouds=[];centers=[]
    for q,w in weights:
        velocity=_residual_velocity(poses,q,cumulative,clock)
        delta=cumulative[t]-cumulative[q]+velocity*(clock[t]-clock[q])
        clouds.append((w,poses[q].cloud+delta));centers.append(w*(poses[q].center+delta))
    return dict(clouds=clouds,center=sum(centers),paired=a is not None and b is not None,multi_pose=len(clouds)>1,
                gap=max(abs(t-q) for q,_ in weights),volume=template['volume'],width=template['width'])


def _reconnect(snapshots,templates,cumulative,clock,scale,window,anchor_ids):
    """Whole-movie endpoint assignment, with ambiguity rejection and no overlapping lifetimes."""
    times=defaultdict(list)
    for t,objects in enumerate(snapshots):
        for identity in objects:times[identity].append(t)
    ids=sorted(templates);cost=np.full((len(ids),len(ids)),10.);evidence={}
    for i,left in enumerate(ids):
        end=max(times[left]);a=max(templates[left]['poses'])
        if end-a>2:continue
        oa=templates[left]['poses'][a]
        for j,right in enumerate(ids):
            start=min(times[right]);b=min(templates[right]['poses'])
            if not 0<start-end<=window+1 or b-start>2:continue
            ob=templates[right]['poses'][b]
            ratio=templates[right]['volume']/templates[left]['volume']
            if not .45<ratio<2.2:continue
            va=_residual_velocity(templates[left]['poses'],a,cumulative,clock)
            vb=_residual_velocity(templates[right]['poses'],b,cumulative,clock)
            dt=clock[b]-clock[a];global_delta=cumulative[b]-cumulative[a]
            forward=oa.center+global_delta+va*dt
            backward=ob.center-global_delta-vb*dt
            distance=.5*(np.linalg.norm((forward-ob.center)*scale)+np.linalg.norm((backward-oa.center)*scale))
            gate=min(32.,max(10.,2*oa.shape[-1]))+2*(start-end-1)
            if distance>gate:continue
            deviation=np.linalg.norm(np.log((ob.shape+.5)/(oa.shape+.5)))/math.sqrt(3)
            if deviation>.7:continue
            cloud=oa.cloud+(forward-oa.center)
            nearest=cKDTree(ob.cloud*scale).query(cloud*scale)[0]
            coverage=float(np.mean(nearest<max(3.,2*templates[left]['width'])))
            if coverage<.35:continue
            score=.32*distance/gate+.28*(1-coverage)+.20*min(1.,abs(math.log(ratio)))+.15*deviation+.05*(start-end-1)/max(1,window)
            cost[i,j]=score;evidence[i,j]=(distance,coverage,start-end-1)
    match=_assignment(cost);parent={identity:identity for identity in times};events=[]
    def root(identity):
        while parent[identity]!=identity:identity=parent[identity]
        return identity
    for j,i in sorted(match.items(),key=lambda ji:cost[ji[1],ji[0]]):
        if cost[i,j]>.48:continue
        alternatives=np.concatenate([np.delete(cost[i],j),np.delete(cost[:,j],i)])
        margin=float(alternatives.min(initial=10.)-cost[i,j])
        if margin<.08:continue
        left,right=ids[i],ids[j];a,b=root(left),root(right)
        if a==b:continue
        group_a=[k for k in times if root(k)==a];group_b=[k for k in times if root(k)==b]
        if set(t for k in group_a for t in times[k]) & set(t for k in group_b for t in times[k]):continue
        if b in anchor_ids or (a not in anchor_ids and len(times[b])>len(times[a])):a,b=b,a
        parent[b]=a
        distance,coverage,gap=evidence[i,j]
        events.append(dict(type='tracklet_reconnection',old_track=right,previous_track=left,track=a,
                           score=float(1-cost[i,j]),margin=margin,gap_frames=gap,cloud_coverage=coverage))
    return {identity:root(identity) for identity in times},events


def _tubeness(raw,spacing):
    """Physical-axis Hessian line response for a local ROI; not a binary truth mask."""
    values=np.asarray(raw,np.float32)
    lo,hi=np.percentile(values,[5,99]);values=np.clip((values-lo)/max(1.,hi-lo),0,1)
    response=np.zeros(values.shape,np.float32)
    for radius in (.12,.22):
        sigma=np.maximum(.65,radius/spacing)
        hessian=np.empty((*values.shape,3,3),np.float32)
        for i in range(3):
            for j in range(i,3):
                order=[0,0,0];order[i]+=1;order[j]+=1
                derivative=ndi.gaussian_filter(values,sigma,order=order)*radius**2/(spacing[i]*spacing[j])
                hessian[...,i,j]=derivative;hessian[...,j,i]=derivative
        eig=np.linalg.eigvalsh(hessian)
        eig=np.take_along_axis(eig,np.argsort(np.abs(eig),axis=-1),axis=-1)
        a,b,c=np.moveaxis(eig,-1,0);eps=1e-6
        strength=np.sqrt(a*a+b*b+c*c);contrast=max(1e-4,float(np.percentile(strength,90)))
        line=(1-np.exp(-(np.abs(b)/(np.abs(c)+eps))**2/.5))*np.exp(-(np.abs(a)/np.sqrt(np.abs(b*c)+eps))**2/.5)
        line*=1-np.exp(-strength**2/(2*contrast**2));line[(b>=0)|(c>=0)]=0
        response=np.maximum(response,line.astype(np.float32))
    return response


def _compound_partition(obj,source,tracked,raw,predictions,identities,scale,spacing):
    from skimage.segmentation import watershed
    sl=tuple(slice(int(a),int(b)+1) for a,b in zip(obj.lo,obj.hi))
    binary=source[sl]==obj.label;xyz=np.argwhere(binary);absolute=xyz+obj.lo
    distances=[]
    for identity in identities:
        prediction=predictions[identity]
        distance=sum(weight*cKDTree(cloud*scale).query(absolute*scale)[0]
                     for weight,cloud in prediction['clouds'])
        distances.append(distance)
    distances=np.asarray(distances);winner=np.argmin(distances,axis=0)
    markers=np.zeros(binary.shape,np.int32);used=np.zeros(len(xyz),bool)
    for i,identity in enumerate(identities):
        eligible=(winner==i)&~used
        other=np.min(np.delete(distances,i,axis=0),axis=0)
        core=eligible&(distances[i]<max(2.,1.2*predictions[identity]['width']))&(other-distances[i]>.5)
        if not core.any():
            candidates=np.flatnonzero(eligible)
            if not len(candidates):return None,dict(reason='no_separated_seed')
            k=candidates[np.argmin(distances[i,candidates])]
            if distances[i,k]>max(5.,2.5*predictions[identity]['width']):return None,dict(reason='seed_outside_prediction')
            core[k]=True
        markers[tuple(xyz[core].T)]=i+1;used|=core
    signal=ndi.gaussian_filter(np.asarray(raw[sl],np.float32),np.maximum(.4,.10/spacing))
    lo,hi=np.percentile(signal[binary],[10,95]);signal=np.clip((signal-lo)/max(1.,hi-lo),0,1)
    tube=_tubeness(raw[sl],spacing)
    radius=ndi.distance_transform_edt(binary,sampling=spacing)
    interior=radius/max(.05,float(np.percentile(radius[binary],90)))
    elevation=.45*(1-signal)+.35*(1-tube)-.20*np.minimum(1,interior)
    wlabels=watershed(elevation,markers,mask=binary)
    choices={'cloud_prior':winner,'signal_tubeness_watershed':wlabels[tuple(xyz.T)]-1}
    previous=tracked[sl][binary];lookup={identity:i for i,identity in enumerate(identities)}
    if all(int(k) in lookup for k in np.unique(previous)):
        choices['existing_partition']=np.array([lookup[int(k)] for k in previous])
    evaluated=[]
    for name,assignment in choices.items():
        if (assignment<0).any():continue
        cost=0.;valid=True
        for i,identity in enumerate(identities):
            selected=assignment==i;volume=int(selected.sum());reference=predictions[identity]['volume']
            ratio=volume/max(1,reference)
            if not .20<ratio<2.5:valid=False;break
            center=absolute[selected].mean(0)
            displacement=np.linalg.norm((center-predictions[identity]['center'])*scale)
            point_distance=float(np.mean(distances[i,selected]))/max(2.,predictions[identity]['width'])
            cost+=.50*min(3.,point_distance)+.30*abs(math.log(ratio))+.20*min(3.,displacement/max(4.,predictions[identity]['width']*2))
        if valid:
            partition=np.zeros(binary.shape,np.int16);partition[tuple(xyz.T)]=assignment+1
            boundary=np.zeros(binary.shape,bool)
            for axis in range(3):
                a=[slice(None)]*3;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
                a,b=tuple(a),tuple(b)
                interface=(partition[a]>0)&(partition[b]>0)&(partition[a]!=partition[b])
                boundary[a]|=interface;boundary[b]|=interface
            boundary_penalty=float(np.mean((.6*signal+.4*tube)[boundary])) if boundary.any() else 0.
            evaluated.append((cost/len(identities)+.20*boundary_penalty,name,assignment))
    if not evaluated:return None,dict(reason='volume_or_prediction_inconsistent')
    evaluated.sort(key=lambda q:q[0]);cost,name,assignment=evaluated[0]
    previous_score=next((c for c,n,_ in evaluated if n=='existing_partition'),None)
    if previous_score is not None and name!='existing_partition' and previous_score-cost<.04:
        cost,name,assignment=next(q for q in evaluated if q[1]=='existing_partition')
    confidence='bidirectional_supported' if all(predictions[i]['paired'] for i in identities) and cost<1.2 else 'uncertain'
    if cost>2:return None,dict(reason='high_partition_cost',cost=cost)
    labels=np.array(identities,np.uint32)[assignment]
    return (sl,binary,labels),dict(method=name,cost=float(cost),confidence=confidence,
                                 candidate_scores={n:float(c) for c,n,_ in evaluated},inferred_partition=True)


def _temporal_refine(initial,source,raw,scale,spacing,base_report,window=6,frame_interval=None):
    """Refine observed foreground, preserving an explicit record of every deletion."""
    if initial.shape!=source.shape or raw.shape!=source.shape:raise ValueError('DAPI/source/tracking grids differ')
    if not np.array_equal(initial>0,source>0):raise ValueError('Initial tracking must retain source foreground')
    snapshots=[_objects(frame,scale) for frame in initial]
    original_times=defaultdict(list)
    for t,objects in enumerate(snapshots):
        for identity in objects:original_times[identity].append(t)
    initial_runs=[]
    for times in original_times.values():
        best=current=1
        for a,b in zip(times,times[1:]):current=current+1 if b==a+1 else 1;best=max(best,current)
        initial_runs.append(best)
    source_objects=[_objects(frame,scale) for frame in source]
    ownership=[_ownership(s,i,o) for s,i,o in zip(source,initial,source_objects)]
    shifts=np.asarray(base_report['collective_shifts_zyx'],float).reshape(-1,3)
    cumulative=np.vstack([np.zeros(3),np.cumsum(shifts,axis=0)])
    clock=np.arange(len(source),dtype=float)*(frame_interval if frame_interval else 1.)
    templates,signals,vref=_templates(snapshots,source,ownership,raw,scale,cumulative,clock)
    anchor=int(base_report['anchor_frame']);anchor_ids=set(snapshots[anchor])&set(templates)
    aliases,events=_reconnect(snapshots,templates,cumulative,clock,scale,window,anchor_ids)
    output=initial.copy();lut=np.arange(int(initial.max(initial=0))+1,dtype=np.uint32)
    for identity,target in aliases.items():lut[identity]=target
    if any(k!=v for k,v in aliases.items()):
        for frame in output:frame[:]=lut[frame]
        snapshots=[_objects(frame,scale) for frame in output]
        ownership=[_ownership(s,i,o) for s,i,o in zip(source,output,source_objects)]
        templates,signals,vref=_templates(snapshots,source,ownership,raw,scale,cumulative,clock)
    compounds=[];quarantine=[];removed=0;changed_partitions=0;fragment_voxels=0
    presence=Counter(identity for objects in snapshots for identity in objects)
    for t,objects in enumerate(source_objects):
        predictions={identity:p for identity,template in templates.items()
                     if (p:=_predict(template,t,cumulative,clock,window)) is not None}
        supports={}
        for identity,p in predictions.items():
            votes=defaultdict(float)
            for weight,cloud in p['clouds']:
                for label,support in _votes(cloud,source[t],scale).items():votes[label]+=weight*support
            for label,support in votes.items():supports.setdefault(label,{})[identity]=support
        for label,o in sorted(objects.items(),key=lambda kv:kv[1].volume,reverse=True):
            current=ownership[t][label];existing=list(current)
            options=supports.get(label,{})
            if len(existing)==1 and existing[0] in templates and t in templates[existing[0]]['poses'] and o.volume<1.25*templates[existing[0]]['volume']:
                continue  # Do not split a clear, normal-volume observation merely because another prediction is nearby.
            candidates=[i for i in options if options[i]>=.25 and (i in current or predictions[i]['paired']
                         or predictions[i]['multi_pose'] and options[i]>=.65)]
            candidates.sort(key=lambda i:options[i],reverse=True);candidates=candidates[:8]
            if len(candidates)<2:continue
            reference=sum(predictions[i]['volume'] for i in candidates)
            if not .45*reference<o.volume<2.2*reference:continue
            extra=[i for i in existing if i not in candidates]
            if extra:continue  # Fragment absorption needs its own evidence, not incidental erasure during a partition.
            result,diagnostics=_compound_partition(o,source[t],output[t],raw[t],predictions,candidates,scale,spacing)
            row=dict(t=t,source_label=label,identities=candidates,existing_identities=existing,**diagnostics)
            compounds.append(row)
            if result is not None:
                sl,binary,labels=result;before=output[t][sl][binary]
                if not np.array_equal(before,labels):
                    output[t][sl][binary]=labels;changed_partitions+=1
                    events.append(dict(type='temporal_compound_repartition',frame=t,source=label,identities=candidates,**diagnostics))
        # Minor source observations only: two substantial instances are never joined.
        totals=Counter()
        for frame_id,count in zip(*np.unique(output[t],return_counts=True)):
            if frame_id:totals[int(frame_id)]=int(count)
        for label,o in sorted(_objects(output[t],scale).items(),key=lambda kv:kv[1].volume):
            if o.volume>.20*vref or label in templates:continue
            sl=tuple(slice(int(a),int(b)+1) for a,b in zip(o.lo,o.hi));binary=output[t][sl]==label
            # A small fragment occupies only a tiny fraction of its parent cloud.
            # Measure support on FRAGMENT points, separately from both clear poses.
            options=[]
            for identity,p in predictions.items():
                if not p['multi_pose'] or o.volume>.20*p['volume']:continue
                fractions=[float(np.mean(cKDTree(cloud*scale).query(o.cloud*scale)[0]
                                          <max(3.,1.5*p['width']))) for _,cloud in p['clouds']]
                options.append((identity,min(fractions)))
            options.sort(key=lambda q:q[1],reverse=True)
            if not options:continue
            identity,support=options[0];competitor=options[1][1] if len(options)>1 else 0.
            p=predictions[identity]
            minimum,margin=(.60,.15) if p['paired'] else (.80,.20)
            if support<minimum or support-competitor<margin:continue
            if o.volume>.20*p['volume'] or totals[identity]+o.volume>1.65*p['volume']:continue
            output[t][sl][binary]=identity;totals[identity]+=o.volume;fragment_voxels+=o.volume
            events.append(dict(type='temporal_fragment_absorption',frame=t,fragment_track=label,track=identity,support=support,
                               bidirectional=p['paired'],independent_reference_poses=len(p['clouds']),voxels=o.volume))
    events.extend(_smooth(output,source_objects,anchor,list(shifts),scale,set(templates)))
    # Do not absorb just half of a short track and leave a new one-frame orphan.
    # Complete, evidence-supported absorption remains valid; partial short-track
    # changes are reverted rather than extending a guess into unsupported frames.
    final_presence=Counter(int(i) for frame in output for i in np.unique(frame) if i)
    previous_presence=Counter(i for objects in snapshots for i in objects)
    reverted=set()
    for identity,count in previous_presence.items():
        if identity in templates or count>3 or not 0<final_presence[identity]<count:continue
        restored=0
        for t,objects in enumerate(snapshots):
            if identity not in objects:continue
            o=objects[identity];sl=tuple(slice(int(a),int(b)+1) for a,b in zip(o.lo,o.hi))
            before=lut[initial[t][sl]]==identity
            restored+=int(np.count_nonzero(before&(output[t][sl]!=identity)))
            output[t][sl][before]=identity
        if restored:
            reverted.add(identity)
            events.append(dict(type='partial_short_track_change_reverted',track=identity,voxels=restored))
    if reverted:
        quarantine=[rows[~np.isin(rows[:,4],list(reverted))] for rows in quarantine]
        for event in events:
            old=event.get('fragment_track',event.get('old_track',event.get('track') if event['type']=='speck_quarantined' else None))
            if old in reverted:event['applied']=False
    output,cleanup_events,cleanup_removed,cleanup_audit=_transient_cleanup(
        output,source,raw,scale,spacing,base_report)
    events.extend(cleanup_events)
    quarantine.append(cleanup_removed)
    removed=sum(len(rows) for rows in quarantine)
    fragment_voxels=sum(e.get('voxels',0) for e in events if e['type']=='temporal_fragment_absorption' and e.get('applied',True))
    if np.any((output>0)&(source==0)):raise AssertionError('Refinement added foreground')
    nodes=[];mapping=[];per_frame=[];lengths=Counter();time_lists=defaultdict(list)
    for t,(s,f) in enumerate(zip(source,output)):
        objs=_objects(f,scale)
        for identity,o in objs.items():
            lengths[identity]+=1;time_lists[identity].append(t)
            nodes.append(dict(movie=base_report['movie'],t=t,track_id=identity,volume_voxels=o.volume,
                              centroid_z=float(o.center[0]),centroid_y=float(o.center[1]),centroid_x=float(o.center[2]),
                              reliable_template=identity in templates,candidate_identity=True))
        own=_ownership(s,f,source_objects[t])
        for label,o in source_objects[t].items():
            retained=0
            for identity,count in own[label].items():
                retained+=count;mapping.append(dict(t=t,source_label=label,track_id=identity,voxels=count,
                                   fraction=count/o.volume,compound_partition=len(own[label])>1))
            if retained<o.volume:mapping.append(dict(t=t,source_label=label,track_id=0,voxels=o.volume-retained,fraction=(o.volume-retained)/o.volume,quarantined=True))
        per_frame.append(dict(t=t,source_instances=len(source_objects[t]),tracked_instances=len(objs),
                              foreground_voxels=int(np.count_nonzero(f)),removed_voxels=int(np.count_nonzero(s)-np.count_nonzero(f))))
    runs=[]
    for times in time_lists.values():
        best=current=1
        for a,b in zip(times,times[1:]):current=current+1 if b==a+1 else 1;best=max(best,current)
        runs.append(best)
    values=list(lengths.values());report=dict(base_report)
    report.update(version=VERSION,initial_tracking_version=base_report['version'],
        initial_tracking_metrics=dict(track_segments=base_report['track_segments'],events=base_report['events'],
            track_length=dict(base_report['track_length'],longest_contiguous_mean=float(np.mean(initial_runs)) if initial_runs else 0.,
                              longest_contiguous_median=float(np.median(initial_runs)) if initial_runs else 0.)),
        track_segments=len(lengths),allocated_ids=int(output.max(initial=0)),per_frame=per_frame,
        reliable_multi_frame_templates=len(templates),temporal_window_frames=window,
        foreground_preserved=removed==0,foreground_added_voxels=0,foreground_removed_voxels=removed,
        deletion_cap_per_frame_fraction=None,deletion_volume_gate=None,
        transient_cleanup_audit=cleanup_audit,quarantine_columns=['t','z','y','x','track_id','source_label'],
        source_mask_files_modified=False,raw_files_modified=False,hard_chromosome_count_constraint=False,
        compound_partitions_changed=changed_partitions,compound_hypotheses_evaluated=len(compounds),
        fragment_voxels_absorbed=fragment_voxels,frame_interval_seconds=frame_interval,
        frame_interval_policy='seconds from supplied/acquisition metadata' if frame_interval else 'frame units; interval unknown, not invented',
        track_length=dict(mean=float(np.mean(values)) if values else 0.,median=float(np.median(values)) if values else 0.,
                          one_frame=sum(v==1 for v in values),one_frame_fraction=sum(v==1 for v in values)/max(1,len(values)),
                          longest_contiguous_mean=float(np.mean(runs)) if runs else 0.,longest_contiguous_median=float(np.median(runs)) if runs else 0.),
        refinement_events=dict(Counter(e['type'] for e in events if e.get('applied',True))),
        relations_stage='Final assignments are in nodes/source_mapping; refinement hypotheses are in compounds/refinement events.',
        inferred_partitions_are_not_ground_truth=True,quality_scores_are_heuristic_not_calibrated=True)
    removed_array=np.concatenate(quarantine).astype(np.uint32) if quarantine else np.empty((0,6),np.uint32)
    return output,report,nodes,events,mapping,compounds,removed_array


def _refinement_regressions(report):
    old=report['initial_tracking_metrics']['track_length'];new=report['track_length'];reasons=[]
    if new['one_frame']>old['one_frame']:reasons.append('more_one_frame_tracks')
    if new['longest_contiguous_mean']+.001<.995*old['longest_contiguous_mean']:
        reasons.append('shorter_mean_contiguous_visibility')
    return reasons


def _cleanup_existing(initial,source,raw,scale,spacing,base_report):
    output,events,removed,audit=_transient_cleanup(initial,source,raw,scale,spacing,base_report)
    nodes=[];mapping=[];per_frame=[];histories=defaultdict(list)
    for t,(s,f) in enumerate(zip(source,output)):
        objects=_objects(f,scale);src_objects=_objects(s,scale)
        for identity,o in objects.items():
            histories[identity].append(t)
            nodes.append(dict(movie=base_report['movie'],t=t,track_id=identity,volume_voxels=o.volume,
                              centroid_z=float(o.center[0]),centroid_y=float(o.center[1]),centroid_x=float(o.center[2]),
                              candidate_identity=True))
        for label,owned in _ownership(s,f,src_objects).items():
            total=src_objects[label].volume;retained=sum(owned.values())
            for identity,count in owned.items():
                mapping.append(dict(t=t,source_label=label,track_id=identity,voxels=count,
                                    fraction=count/total,compound_partition=len(owned)>1))
            if retained<total:mapping.append(dict(t=t,source_label=label,track_id=0,voxels=total-retained,
                                                 fraction=(total-retained)/total,quarantined=True))
        per_frame.append(dict(t=t,source_instances=len(src_objects),tracked_instances=len(objects),
                              foreground_voxels=int(np.count_nonzero(f)),removed_voxels=int(np.count_nonzero(s)-np.count_nonzero(f))))
    lengths=[len(times) for times in histories.values()];runs=[]
    for times in histories.values():
        best=current=1
        for a,b in zip(times,times[1:]):current=current+1 if b==a+1 else 1;best=max(best,current)
        runs.append(best)
    report=dict(base_report)
    report.update(version=VERSION,cleanup_initial_version=base_report['version'],cleanup_only=True,
                  cleanup_before_metrics=dict(track_segments=base_report['track_segments'],track_length=base_report['track_length']),
                  track_segments=len(histories),allocated_ids=int(output.max(initial=0)),per_frame=per_frame,
                  track_length=dict(mean=float(np.mean(lengths)) if lengths else 0.,median=float(np.median(lengths)) if lengths else 0.,
                      one_frame=sum(n==1 for n in lengths),one_frame_fraction=sum(n==1 for n in lengths)/max(1,len(lengths)),
                      longest_contiguous_mean=float(np.mean(runs)) if runs else 0.,longest_contiguous_median=float(np.median(runs)) if runs else 0.),
                  transient_cleanup_audit=audit,transient_tracks_quarantined=sum(len(e['episode_tracks']) for e in events),
                  cleanup_removed_voxels=len(removed),foreground_added_voxels=0,
                  foreground_removed_voxels=sum(row['removed_voxels'] for row in per_frame),
                  foreground_preserved=not any(row['removed_voxels'] for row in per_frame),
                  deletion_cap_per_frame_fraction=None,deletion_volume_gate=None,
                  refinement_selection='existing_tracking_with_evidence_based_transient_cleanup',
                  quarantine_columns=['t','z','y','x','track_id','source_label'])
    report['refinement_events']=dict(base_report.get('refinement_events',{}))
    report['refinement_events']['transient_track_quarantined']=len(events)
    return output,report,nodes,events,mapping,removed


def _retain_initial(initial,source,report):
    """Selecting unchanged tracking is a valid alternative, not a forced repair."""
    old=report['initial_tracking_metrics'];result=dict(report)
    result.update(refinement_applied=False,refinement_selection='initial_retained_by_regression_guard',
                  refinement_rejection_reasons=_refinement_regressions(report),proposal_metrics=dict(
                      track_segments=report['track_segments'],track_length=report['track_length'],
                      refinement_events=report['refinement_events'],foreground_removed_voxels=report['foreground_removed_voxels']),
                  track_segments=old['track_segments'],track_length=old['track_length'],
                  allocated_ids=int(initial.max(initial=0)),foreground_preserved=True,foreground_removed_voxels=0,
                  compound_partitions_changed=0,fragment_voxels_absorbed=0,refinement_events={},reliable_multi_frame_templates=0)
    nodes=[];mapping=[];per_frame=[]
    for t,(s,f) in enumerate(zip(source,initial)):
        objects=_objects(f,np.asarray(report['distance_axis_scale']));src_objects=_objects(s,np.asarray(report['distance_axis_scale']))
        for identity,o in objects.items():
            nodes.append(dict(movie=report['movie'],t=t,track_id=identity,volume_voxels=o.volume,
                              centroid_z=float(o.center[0]),centroid_y=float(o.center[1]),centroid_x=float(o.center[2]),
                              reliable_template=False,candidate_identity=True,selection='initial_retained'))
        for label,owned in _ownership(s,f,src_objects).items():
            for identity,count in owned.items():
                mapping.append(dict(t=t,source_label=label,track_id=identity,voxels=count,
                                    fraction=count/src_objects[label].volume,compound_partition=len(owned)>1))
        per_frame.append(dict(t=t,source_instances=len(src_objects),tracked_instances=len(objects),
                              foreground_voxels=int(np.count_nonzero(f)),removed_voxels=0))
    result['per_frame']=per_frame
    return result,nodes,mapping


def _write_csv(path,rows):
    rows=list(rows)
    with path.open("w",newline="") as f:
        if not rows:return
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader();writer.writerows(rows)


def _read_metadata(path):
    with tifffile.TiffFile(path) as tf:
        meta=dict(tf.imagej_metadata or {})
        if not meta:
            try:meta=json.loads(tf.pages[0].description or "{}")
            except (ValueError,TypeError):meta={}
        tags=tf.pages[0].tags;resolution=None
        if "XResolution" in tags and "YResolution" in tags:
            xr,yr=tags["XResolution"].value,tags["YResolution"].value
            resolution=(xr[0]/xr[1],yr[0]/yr[1])
        unit=tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None
    spacing=None
    if resolution and meta.get("unit") in ("micron","um","µm") and "spacing" in meta:
        spacing=np.array([meta["spacing"],1/resolution[1],1/resolution[0]],float)
        if not np.all(np.isfinite(spacing)) or np.any(spacing<=0):raise ValueError("Invalid spacing")
    return meta,resolution,unit,spacing


def _sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--published-output",type=Path,help="Final path when processing into an atomic staging directory")
    p.add_argument("--anchors",type=Path);p.add_argument("--gif-script",type=Path)
    p.add_argument("--raw",type=Path,help="Matched preprocessed DAPI TZYX TIFF root; enables temporal refinement")
    p.add_argument("--initial-tracks",type=Path,help="Reuse v2.1 tracking as the initial solution")
    p.add_argument("--cleanup-only",action="store_true",help="Clean current tracking without rerunning identity inference")
    p.add_argument("--temporal-window",type=int,default=6)
    p.add_argument("--frame-interval-seconds",type=float,help="Only supply a verified acquisition interval")
    p.add_argument("--workers",type=int,default=1);p.add_argument("--worker-index",type=int,default=0)
    p.add_argument("--duration-seconds",type=float,default=6.);p.add_argument("--gif-max-size",type=int,default=300)
    p.add_argument("--limit",type=int,default=0);p.add_argument("--sample",action="append",default=[])
    p.add_argument("--resume",action="store_true");args=p.parse_args()
    if args.input.resolve()==args.output.resolve():p.error("Output must differ from source masks")
    if args.initial_tracks and args.initial_tracks.resolve()==args.output.resolve():p.error("Output must differ from initial tracking")
    if args.initial_tracks and not args.raw:p.error("Initial-track refinement requires matched DAPI")
    if args.cleanup_only and not (args.initial_tracks and args.raw):p.error('Cleanup requires initial tracking and DAPI')
    if args.temporal_window<1 or not 0<=args.worker_index<args.workers:p.error("Invalid temporal window/worker")
    if args.frame_interval_seconds is not None and args.frame_interval_seconds<=0:p.error("Invalid acquisition interval")
    if args.duration_seconds<=0 or args.gif_max_size<0:p.error("Invalid GIF configuration")
    for directory in (args.output,args.output/"reports",args.output/"tables"):
        directory.mkdir(parents=True,exist_ok=True)
    anchors=json.loads(args.anchors.read_text()) if args.anchors else {}
    anchors=anchors.get("anchors",anchors)
    gif=None
    if args.gif_script:
        spec=importlib.util.spec_from_file_location("motion_gif",args.gif_script)
        gif=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gif;spec.loader.exec_module(gif)
    stamp=dict(tracker_sha256=_sha(Path(__file__)),anchors_sha256=_sha(args.anchors) if args.anchors else None,
        gif_renderer_sha256=_sha(args.gif_script) if args.gif_script else None,
        gif_duration_seconds=args.duration_seconds,gif_max_size=args.gif_max_size,
        temporal_window=args.temporal_window,frame_interval_seconds=args.frame_interval_seconds,
        cleanup_only=args.cleanup_only)
    raw_lookup={}
    if args.raw:
        for path in args.raw.rglob('*-DAPI.tif'):
            if any(part.startswith('.') for part in path.relative_to(args.raw).parts):continue
            name=path.stem.removesuffix('-DAPI')
            if name in raw_lookup:p.error(f"Duplicate DAPI name: {name}")
            raw_lookup[name]=path
    files=sorted(path for d in [args.input,*(args.input/c for c in CATEGORIES)] if d.is_dir()
        for path in d.iterdir() if path.is_file() and path.suffix.lower() in (".tif",".tiff"))
    if args.sample:files=[path for path in files if _stem(path) in args.sample]
    if args.limit:files=files[:args.limit]
    files=[path for i,path in enumerate(files) if i%args.workers==args.worker_index]
    if not files:p.error("No matching source TIFFs")
    if len({_stem(path) for path in files})!=len(files):p.error("Duplicate sample basenames")
    reports=[]
    for k,path in enumerate(files,1):
        start=time.monotonic();name=_stem(path);category=path.parent.name if path.parent.name in CATEGORIES else None
        directory=args.output/category if category else args.output;directory.mkdir(exist_ok=True)
        target=directory/f"{name}_tracked.tif";report_path=args.output/"reports"/f"{name}.json"
        gif_path=args.output/"gif"/f"{name}_tracked_time_front_back_mask.gif"
        source_stamp=dict(path=str(path.resolve()),bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,sha256=_sha(path))
        raw_path=raw_lookup.get(name)
        if args.raw and raw_path is None:raise ValueError(f"Missing matched DAPI: {name}")
        initial_path=(args.initial_tracks/category if category else args.initial_tracks)/f"{name}_tracked.tif" if args.initial_tracks else None
        extra_stamps={}
        for key,extra in [('raw',raw_path),('initial_tracking',initial_path)]:
            if extra:extra_stamps[key]=dict(path=str(extra),sha256=_sha(extra),mtime_ns=extra.stat().st_mtime_ns)
        movie_stamp=dict(stamp,**extra_stamps)
        run_version=VERSION if args.raw else BASE_VERSION
        if args.resume and target.exists() and report_path.exists():
            previous=json.loads(report_path.read_text())
            if (previous.get("version")==run_version and previous.get("source_stamp")==source_stamp
                and previous.get("config_stamp")==movie_stamp and previous.get("status")=="complete"
                and (not gif or gif_path.exists())):
                reports.append(previous);print(f"[{k}/{len(files)}] resume {name}",flush=True);continue
        print(f"[{k}/{len(files)}] {name}",flush=True)
        arr=tifffile.imread(path)
        if arr.ndim==3:arr=arr[None]
        meta,resolution,unit,spacing=_read_metadata(path)
        scale=spacing/math.sqrt(spacing[1]*spacing[2]) if spacing is not None else np.array([2.,1.,1.])
        if initial_path:
            tracked=tifffile.imread(initial_path)
            report=json.loads((args.initial_tracks/'reports'/f'{name}.json').read_text())
            if not args.cleanup_only and report['version']!=BASE_VERSION:raise ValueError('Initial tracks must be v2.1, not an already refined output')
        else:
            tracked,report,nodes,relations,mapping=_track_movie(arr,name,anchors,scale)
        if raw_path:
            raw_meta,_,_,raw_spacing=_read_metadata(raw_path)
            if raw_spacing is None:raise ValueError('DAPI calibration is missing')
            if spacing is not None and not np.allclose(spacing,raw_spacing,rtol=1e-5):raise ValueError('DAPI and mask calibration differs')
            with tifffile.TiffFile(raw_path) as raw_file:
                if raw_file.series[0].axes not in ('TZYX','ZYX'):raise ValueError('DAPI axes must be TZYX/ZYX')
            try:raw=tifffile.memmap(raw_path,mode='r')
            except (ValueError,TypeError):raw=tifffile.imread(raw_path)
            if raw.ndim==3:raw=raw[None]
            interval=args.frame_interval_seconds
            if interval is None:
                for key in ('finterval','time_interval_seconds','dt_seconds'):
                    value=raw_meta.get(key)
                    if isinstance(value,(int,float)) and value>0:interval=float(value);break
            if args.cleanup_only:
                tracked,report,nodes,relations,mapping,removed=_cleanup_existing(tracked,arr,raw,scale,raw_spacing,report)
                old_qpath=args.initial_tracks/'quarantine'/f'{name}_removed.npz'
                if old_qpath.exists():
                    with np.load(old_qpath) as old_q:removed=np.concatenate([old_q['removed'],removed])
                old_compounds=args.initial_tracks/'tables'/f'{name}_compounds.csv'
                compounds=list(csv.DictReader(old_compounds.open())) if old_compounds.exists() else []
                old_relations=args.initial_tracks/'tables'/f'{name}_relations.csv'
                if old_relations.exists():relations=list(csv.DictReader(old_relations.open()))+relations
            else:
                tracked,report,nodes,relations,mapping,compounds,removed=_temporal_refine(
                    tracked,arr,raw,scale,raw_spacing,report,args.temporal_window,interval)
            if not args.cleanup_only and _refinement_regressions(report):
                # This path is also used by final selection of staged v3.0 proposals.
                if initial_path:tracked=tifffile.imread(initial_path)
                else:tracked,_,_,_,_=_track_movie(arr,name,anchors,scale)
                report,nodes,mapping=_retain_initial(tracked,arr,report)
                relations=[dict(type='refinement_rejected',reasons=report['refinement_rejection_reasons'])]
                for row in compounds:row['selected']=False
                removed=np.empty((0,6),np.uint32)
            elif not args.cleanup_only:report.update(refinement_applied=True,refinement_selection='temporal_refinement_accepted')
            qdir=args.output/'quarantine';qdir.mkdir(exist_ok=True)
            qpath=qdir/f'{name}_removed.npz'
            np.savez_compressed(qpath,removed=removed,columns=np.array(report['quarantine_columns']))
            _write_csv(args.output/'tables'/f'{name}_compounds.csv',compounds)
            report['quarantine_file']=str((args.published_output or args.output)/qpath.relative_to(args.output))
            del raw,removed
        report.update(source_stamp=source_stamp,config_stamp=movie_stamp,category=category,
            spacing_um=spacing.tolist() if spacing is not None else None,
            anchor_supplied=name in anchors,source_pipeline=meta.get("pipeline"))
        metadata={k:v for k,v in meta.items() if k!="shape"}
        metadata.update(axes="TZYX",tracking_version=run_version,tracking_anchor_t=report["anchor_frame"],
            tracking_input=str(path),tracking_compound_partitions_uncertain=True,
            tracking_raw=str(raw_path) if raw_path else None,tracking_temporal_refined=bool(raw_path),
            tracking_foreground_added=False)
        temp=target.with_suffix(".tmp.tif")
        tifffile.imwrite(temp,tracked,compression="zlib",compressionargs={"level":1},maxworkers=2,
            metadata=metadata,resolution=resolution,resolutionunit=unit,photometric="minisblack",bigtiff=True)
        saved=tifffile.imread(temp)
        if not np.array_equal(saved,tracked):raise AssertionError("TIFF round-trip failure")
        if _sha(path)!=source_stamp["sha256"]:raise AssertionError("Source changed during processing")
        for key,extra in [('raw',raw_path),('initial_tracking',initial_path)]:
            if extra and (_sha(extra)!=extra_stamps[key]['sha256'] or extra.stat().st_mtime_ns!=extra_stamps[key]['mtime_ns']):
                raise AssertionError(f'{key} changed during processing')
        temp.replace(target);del arr,saved,tracked
        _write_csv(args.output/"tables"/f"{name}_nodes.csv",nodes)
        _write_csv(args.output/"tables"/f"{name}_relations.csv",relations)
        _write_csv(args.output/"tables"/f"{name}_source_mapping.csv",mapping)
        if gif:report["gif"]=gif._mask_gif(target,gif_path,
            SimpleNamespace(duration_seconds=args.duration_seconds,fps=None,max_size=args.gif_max_size))
        published=args.published_output or args.output
        report.update(status="complete",output_tiff=str(published/target.relative_to(args.output)),tiff_roundtrip_verified=True,
            seconds=round(time.monotonic()-start,3))
        report_path.write_text(json.dumps(report,indent=2));reports.append(report)
        print(f"  done: {len(report['per_frame'])} t, {report['track_segments']} candidate IDs, "
            f"mean {report['track_length']['mean']:.2f} t, {report['seconds']}s",flush=True)
    summary=dict(version=VERSION if args.raw else BASE_VERSION,input=str(args.input),output=str(args.published_output or args.output),
        staging_output=str(args.output) if args.published_output else None,anchor_file=str(args.anchors),
        raw_input=str(args.raw) if args.raw else None,initial_tracking_input=str(args.initial_tracks) if args.initial_tracks else None,
        movie_count=len(reports),frame_count=sum(r["shape"][0] for r in reports),movies=reports)
    suffix=f'_worker_{args.worker_index}' if args.workers>1 else ''
    (args.output/f"summary{suffix}.json").write_text(json.dumps(summary,indent=2))
    _write_csv(args.output/f"summary{suffix}.csv",[{k:v for k,v in r.items() if k not in ("per_frame","gif","source_stamp")} for r in reports])
    print(json.dumps({"movies":len(reports),"frames":summary["frame_count"]}),flush=True)


if __name__=="__main__":main()
