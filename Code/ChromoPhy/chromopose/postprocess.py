# Standalone post v2.0 readability. Python >=3.10.
# Dependencies: numpy, scipy, scikit-image, tifffile, matplotlib.
# Run: python postprocess.py --masks MASK_DIR --raw RAW_DIR --output NEW_DIR --readability-post
# Default: fill background by nearest original ID, absorb only small fragments,
# then quarantine unresolved fragments. No large-instance merges or splitting.
# Apply once to unprocessed CHRP3D predictions. Legacy modes remain available
# only for reproducing older results and should not be used for readability.
# Four helper modules are inlined below; no companion files are required.
#!/usr/bin/env python3
"""Signal-guided 3D mask completion for manual instance review.

Default readability mode fills small holes/seams with physical nearest-ID
ownership, then absorbs only minor fragments with a clear substantial parent,
then quarantines unresolved small fragments. Substantial source IDs and their
source voxels remain unchanged. No count target, large-instance joining or
instance splitting is applied. Legacy repair modes are explicit opt-ins.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import ndimage as ndi
from skimage.registration import optical_flow_tvl1, phase_cross_correlation
from skimage.transform import warp
import tifffile as tf

VERSION = "signal-guided-instances-v2.0-readability"
CONN = np.ones((3, 3, 3), bool)


@dataclass
class Settings:
    max_gap_width: float = 0.50
    interface_ratio: float = 0.70
    strong_snr: float = 3.0
    group_volume_factor: float = 2.0
    min_direction_cos: float = 0.65
    max_added_fraction: float = 0.05
    max_deleted_fraction: float = 0.005
    boundary_growth_width: float = 0.15
    embedded_contact_fraction: float = 0.55
    embedded_max_volume_ratio: float = 0.30
    smooth_cut_min_contact_fraction: float = 0.012
    smooth_cut_min_alignment: float = 0.58
    smooth_cut_min_axis_agreement: float = 0.55
    smooth_cut_max_axial_spread: float = 0.65
    smooth_cut_min_depth_ratio: float = 0.45
    relaxed_cut_min_contact_fraction: float = 0.025
    relaxed_cut_min_alignment: float = 0.70
    relaxed_cut_min_axis_agreement: float = 0.70
    relaxed_cut_max_axial_spread: float = 0.80
    relaxed_cut_min_depth_ratio: float = 0.50
    relaxed_cut_min_planarity: float = 0.50
    relaxed_cut_min_direct_faces: int = 20
    relaxed_cut_min_volume_ratio: float = 0.35
    relaxed_cut_max_width_factor: float = 1.10
    relaxed_cut_min_signal_ratio: float = 1.15
    shared_fragment_max_vref_fraction: float = 0.08
    shared_fragment_min_total_contact: float = 0.70
    shared_fragment_min_each_contact: float = 0.20
    micro_fragment_max_vref_fraction: float = 0.03
    micro_fragment_min_contact: float = 0.35
    micro_fragment_min_parent_ratio: float = 4.0
    seam_connector_max_vref_fraction: float = 0.03
    seam_connector_min_total_contact: float = 0.70
    seam_connector_min_each_contact: float = 0.20
    seam_connector_min_direct_faces: int = 8
    seam_connector_min_axis_agreement: float = 0.75
    seam_connector_min_depth_ratio: float = 0.45
    seam_connector_min_planarity: float = 0.45
    seam_connector_min_volume_ratio: float = 0.12
    seam_connector_max_width_factor: float = 1.60
    temporal_cobody_min_votes: int = 2
    temporal_cobody_min_contact: float = 0.10
    temporal_cobody_min_direct_faces: int = 20
    temporal_cobody_min_depth_ratio: float = 0.45
    temporal_cobody_min_planarity: float = 0.35
    temporal_cobody_max_width_factor: float = 1.55
    continuity_only: bool = False
    temporal: bool = True
    surface_refine: bool = False
    aggressive_refine: bool = False
    hybrid_refine: bool = False
    readability_refine: bool = True
    readable_fill_radius_um: float = 0.25
    readable_added_fraction: float = 0.08
    readable_instance_added_fraction: float = 0.10
    readable_fragment_vref_fraction: float = 0.05
    readable_fragment_parent_fraction: float = 0.12
    readable_deleted_fraction: float = 0.01


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


def shape4(f):
    s = f.series[0]
    if s.axes == "ZYX":
        return (1,) + s.shape
    if s.axes == "TZYX":
        return s.shape
    raise ValueError(f"Expected TZYX/ZYX, got {s.axes}")


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


def signal_score(raw):
    sample = raw[:: max(1, len(raw) // 40), ::2, ::2].astype(np.float32)
    positive = sample[sample > 0]
    if not positive.size:
        return np.zeros(raw.shape, np.float32), 0.0, 1.0
    bg, ce = np.percentile(positive, [40, 65])
    estimates = []
    for axis in (1, 2):
        pair1 = [slice(None)] * 3
        pair2 = pair1.copy()
        pair1[axis] = slice(1, None)
        pair2[axis] = slice(None, -1)
        x = sample[tuple(pair1)]
        y = sample[tuple(pair2)]
        mask = (x > 0) & (y > 0) & (x <= ce) & (y <= ce)
        if mask.any():
            estimates.append(
                float(np.median(np.abs(x[mask] - y[mask])) / 0.67448975 / np.sqrt(2))
            )
    noise = max(0.5, float(np.median(estimates)) if estimates else 1.0)
    score = (raw.astype(np.float32) - bg) / noise
    return score, float(bg), noise


def features(labels, snr, spacing, keep_coords=False):
    props = {}
    for i, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        local = labels[sl] == i
        coords = np.argwhere(local).astype(float)
        coords += np.array([s.start for s in sl])
        physical = coords * spacing
        center = physical.mean(0)
        covariance = (physical - center).T @ (physical - center) / max(1, len(coords))
        # Include finite voxel extent; singleton/slab fragments have nonzero width.
        covariance += np.diag(spacing**2 / 12)
        eig, vec = np.linalg.eigh(covariance)
        props[i] = dict(
            n=len(coords),
            center=center,
            cov=covariance,
            eig=eig,
            axis=vec[:, -1],
            snr=float(np.median(snr[sl][local])),
            members=[i],
            reserved=0,
        )
        if keep_coords:
            # Used only while measuring the local tangent at a candidate seam.
            # float32 bounds the extra per-frame memory without affecting a
            # voxel-scale orientation decision.
            props[i]["coords"] = physical.astype(np.float32, copy=False)
    return props


def combine(a, b):
    n = a["n"] + b["n"]
    center = (a["n"] * a["center"] + b["n"] * b["center"]) / n
    cov = (
        sum(
            p["n"] * (p["cov"] + np.outer(p["center"] - center, p["center"] - center))
            for p in (a, b)
        )
        / n
    )
    eig, vec = np.linalg.eigh(cov)
    return dict(
        n=n,
        center=center,
        cov=cov,
        eig=eig,
        axis=vec[:, -1],
        snr=(a["snr"] * a["n"] + b["snr"] * b["n"]) / n,
        members=a["members"] + b["members"],
        reserved=a["reserved"] + b["reserved"],
    )


def temporal_mapping(current, raw, neighbor, neighbor_raw):
    """Map labels to one neighbor without treating registration failure as absence.

    Rigid XY translation remains the cheap first choice.  Mitotic chromosome
    fields often deform or rotate too much for one translation, so an
    independently quality-gated dense XY flow is used as a fallback.  The
    fallback is deliberately only temporal *support*: same-Z label votes must
    still be dominant, and geometry remains responsible for accepting a merge.
    """
    projection_a = np.sqrt(np.maximum(raw.max(0).astype(float), 0))
    projection_b = np.sqrt(np.maximum(neighbor_raw.max(0).astype(float), 0))
    a = projection_a.astype(float, copy=True)
    b = projection_b.astype(float, copy=True)
    a -= np.median(a)
    b -= np.median(b)
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=1, normalization=None)
    shifted = ndi.shift(b, shift, order=1, mode="constant", prefilter=False)
    use = (a > np.percentile(a, 65)) | (shifted > np.percentile(shifted, 65))
    if use.sum() < 32 or a[use].std() == 0 or shifted[use].std() == 0:
        return {}, dict(reliable=False, reason="insufficient_signal")
    correlation = float(np.corrcoef(a[use], shifted[use])[0, 1])
    az = raw.sum((1, 2), dtype=float)
    bz = neighbor_raw.sum((1, 2), dtype=float)
    zcorr = float(np.corrcoef(az, bz)[0, 1]) if az.std() and bz.std() else 0.0

    def dominant_mapping(other, valid, fraction, minimum):
        coords = np.argwhere(current > 0)
        own = current[tuple(coords[valid].T)].astype(np.int64)
        target = neighbor[tuple(other[valid].T)].astype(np.int64)
        keep = target > 0
        own, target = own[keep], target[keep]
        if not len(target):
            return {}
        base = int(neighbor.max()) + 1
        keys, counts = np.unique(own * base + target, return_counts=True)
        totals = np.bincount(current.ravel())
        by_source = {}
        for k, n in zip(keys, counts):
            i, j = int(k // base), int(k % base)
            by_source.setdefault(i, []).append((int(n), j))
        mapping = {}
        for i, choices in by_source.items():
            choices.sort(reverse=True)
            n, j = choices[0]
            second = choices[1][0] if len(choices) > 1 else 0
            if (
                n >= max(minimum, fraction * totals[i])
                and (second == 0 or n >= 1.5 * second)
            ):
                mapping[i] = j
        return mapping

    rigid_ok = (
        np.isfinite(correlation)
        and correlation >= 0.80
        and np.max(np.abs(shift)) <= min(a.shape) * 0.12
        and np.isfinite(zcorr)
        and zcorr >= 0.85
    )
    if rigid_ok:
        coords = np.argwhere(current > 0)
        other = coords.copy()
        other[:, 1:] -= np.rint(shift).astype(int)
        valid = np.all((other >= 0) & (other < np.array(neighbor.shape)), axis=1)
        mapping = dominant_mapping(other, valid, fraction=0.60, minimum=5)
        return mapping, dict(
            reliable=True,
            method="rigid_xy",
            correlation=correlation,
            axial_correlation=zcorr,
            shift_yx=shift.tolist(),
            mapped=len(mapping),
        )

    # Dense 2D flow is substantially safer here than lowering the rigid
    # correlation threshold: it can follow local chromosome motion while its
    # warped-image correlation supplies an independent acceptance check.
    def normalize_projection(x):
        lo, hi = np.percentile(x, [5, 99])
        return np.clip((x - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(
            np.float32
        )

    reference = normalize_projection(projection_a)
    moving = normalize_projection(projection_b)
    try:
        flow = optical_flow_tvl1(
            reference,
            moving,
            attachment=10,
            tightness=0.3,
            num_warp=5,
            num_iter=10,
            prefilter=True,
        )
    except (FloatingPointError, ValueError):
        return {}, dict(
            reliable=False,
            reason="deformable_registration_failed",
            rigid_correlation=correlation,
            axial_correlation=zcorr,
            shift_yx=shift.tolist(),
        )
    yy, xx = np.indices(reference.shape)
    warped = warp(
        moving,
        np.array([yy + flow[0], xx + flow[1]]),
        mode="edge",
        preserve_range=True,
    )
    flow_use = (reference > np.percentile(reference, 65)) | (
        warped > np.percentile(warped, 65)
    )
    flow_correlation = (
        float(np.corrcoef(reference[flow_use], warped[flow_use])[0, 1])
        if flow_use.sum() >= 32
        and reference[flow_use].std()
        and warped[flow_use].std()
        else 0.0
    )
    magnitude = np.hypot(flow[0], flow[1])
    flow_p95 = float(np.percentile(magnitude, 95))
    if (
        not np.isfinite(flow_correlation)
        or flow_correlation < 0.75
        or flow_p95 > min(reference.shape) * 0.20
    ):
        return {}, dict(
            reliable=False,
            reason="deformable_registration_unreliable",
            rigid_correlation=correlation,
            flow_correlation=flow_correlation,
            flow_p95_pixels=flow_p95,
            shift_yx=shift.tolist(),
        )
    coords = np.argwhere(current > 0)
    other = coords.copy()
    y = coords[:, 1]
    x = coords[:, 2]
    other[:, 1] = np.rint(y + flow[0, y, x]).astype(int)
    other[:, 2] = np.rint(x + flow[1, y, x]).astype(int)
    valid = np.all((other >= 0) & (other < np.array(neighbor.shape)), axis=1)
    mapping = dominant_mapping(other, valid, fraction=0.45, minimum=5)
    return mapping, dict(
        reliable=True,
        method="tvl1_xy",
        correlation=flow_correlation,
        rigid_correlation=correlation,
        axial_correlation=zcorr,
        flow_p95_pixels=flow_p95,
        mapped=len(mapping),
    )


def collect_edges(labels, snr, spacing, reach, inside=None):
    if reach <= 0:
        dist = np.where(labels > 0, 0., np.inf).astype(np.float32)
        indices = None
        nearest = labels
    else:
        dist, indices = ndi.distance_transform_edt(
            labels == 0, sampling=spacing, return_indices=True
        )
        nearest = labels[tuple(indices)]
    expanded = np.where(dist <= reach, nearest, 0)
    chunks = []
    base = int(labels.max()) + 1
    surface_area = np.zeros(base, np.float64)
    for ax in range(3):
        left = [slice(None)] * 3
        right = left.copy()
        left[ax] = slice(None, -1)
        right[ax] = slice(1, None)
        left = tuple(left)
        right = tuple(right)
        # Physical exposed area for each original label, including contact with
        # background and with another label. This is later used to distinguish
        # an enclosed fragment from a small object merely touching one face.
        face_area = float(np.prod(np.delete(spacing, ax)))
        lx, ly = labels[left], labels[right]
        changed = lx != ly
        if changed.any():
            xv, yv = lx[changed], ly[changed]
            if np.any(xv > 0):
                surface_area += (
                    np.bincount(xv[xv > 0], minlength=base) * face_area
                )
            if np.any(yv > 0):
                surface_area += (
                    np.bincount(yv[yv > 0], minlength=base) * face_area
                )
        for side in (0, -1):
            border = np.take(labels, side, axis=ax)
            border = border[border > 0]
            if border.size:
                surface_area += np.bincount(border, minlength=base) * face_area

        x, y = expanded[left], expanded[right]
        keep = (x > 0) & (y > 0) & (x != y)
        pos = np.argwhere(keep)
        if not len(pos):
            continue
        p = pos.copy()
        q = pos.copy()
        q[:, ax] += 1
        i = x[keep].astype(np.int64)
        j = y[keep].astype(np.int64)
        low, high = np.minimum(i, j), np.maximum(i, j)
        k = low * base + high
        value = np.minimum(snr[tuple(p.T)], snr[tuple(q.T)])
        gap = dist[tuple(p.T)] + dist[tuple(q.T)]
        isdirect = (labels[tuple(p.T)] > 0) & (labels[tuple(q.T)] > 0)
        # Store endpoint coordinates for narrowly constrained straight bridges.
        ep = p if indices is None else indices[:, tuple(p.T)[0], tuple(p.T)[1], tuple(p.T)[2]].T
        eq = q if indices is None else indices[:, tuple(q.T)[0], tuple(q.T)[1], tuple(q.T)[2]].T
        flip = i > j
        endpoint_low = np.where(flip[:, None], eq, ep)
        endpoint_high = np.where(flip[:, None], ep, eq)
        physical_contact = np.where(isdirect, face_area, 0.0)
        chunks.append(
            (
                k,
                value,
                gap,
                isdirect,
                ep,
                eq,
                endpoint_low,
                endpoint_high,
                physical_contact,
            )
        )
    edges = []
    if chunks:
        keys = np.concatenate([x[0] for x in chunks])
        order = np.argsort(keys)
        value = np.concatenate([x[1] for x in chunks])[order]
        gap = np.concatenate([x[2] for x in chunks])[order]
        direct = np.concatenate([x[3] for x in chunks])[order]
        ep = np.concatenate([x[4] for x in chunks])[order]
        eq = np.concatenate([x[5] for x in chunks])[order]
        endpoint_low = np.concatenate([x[6] for x in chunks])[order]
        endpoint_high = np.concatenate([x[7] for x in chunks])[order]
        physical_contact = np.concatenate([x[8] for x in chunks])[order]
        keys = keys[order]
        u, start, count = np.unique(keys, return_index=True, return_counts=True)
        for k, s, n in zip(u, start, count):
            sl = slice(s, s + n)
            near = gap[sl] <= float(gap[sl].min()) + min(spacing) * 1.01
            candidates = np.flatnonzero(near) + s
            best = int(candidates[np.argmax(value[candidates])])
            direct_here = direct[sl]
            use = direct_here if direct_here.any() else near
            pi = endpoint_low[sl][use]
            pj = endpoint_high[sl][use]
            midpoint = (pi + pj).astype(np.float64) * spacing / 2
            interface_center = midpoint.mean(0)
            centered = midpoint - interface_center
            interface_cov = centered.T @ centered / max(1, len(midpoint))
            vectors = (pj - pi).astype(np.float64) * spacing
            norms = np.linalg.norm(vectors, axis=1)
            vectors = vectors[norms > 0] / norms[norms > 0, None]
            mean_normal = vectors.mean(0) if len(vectors) else np.zeros(3)
            normal_strength = float(np.linalg.norm(mean_normal))
            if normal_strength > 1e-8:
                mean_normal /= normal_strength
            else:
                mean_normal = np.zeros(3)
            depth = 0.0
            if inside is not None and len(pi):
                depths = np.minimum(
                    inside[tuple(pi.T)], inside[tuple(pj.T)]
                )
                depth = float(np.percentile(depths, 75))
            ii, jj = int(k // base), int(k % base)
            direct_area = float(physical_contact[sl].sum())
            edges.append(
                dict(
                    i=ii,
                    j=jj,
                    signal=float(np.percentile(value[sl][near], 35)),
                    gap=float(gap[sl].min()),
                    contacts=int(n),
                    direct=int(direct[sl].sum()),
                    direct_area=direct_area,
                    contact_fraction_i=direct_area / max(surface_area[ii], 1e-12),
                    contact_fraction_j=direct_area / max(surface_area[jj], 1e-12),
                    interface_center=interface_center,
                    interface_cov=interface_cov,
                    interface_normal=mean_normal,
                    interface_planarity=normal_strength,
                    interface_depth=depth,
                    p=ep[best],
                    q=eq[best],
                )
            )
    return edges, dist, nearest


def add_shape_priors(edges, props, spacing, width):
    """Measure whether a label seam looks enclosed or transverse to a local rod.

    A chromosome may be curved, so whole-object PCA is not sufficient. At a
    direct interface we estimate each side's local tangent inside a physical
    neighborhood. A false cut should cross both tangents; a contact between two
    parallel chromosomes normally runs along them. The interface covariance
    supplies the complementary seam-spread check.
    """
    radius = max(3.0 * width, 4.0 * float(max(spacing)))
    stable_elongation = 1.18
    for e in edges:
        i, j = e["i"], e["j"]
        ni, nj = props[i]["n"], props[j]["n"]
        small = i if ni <= nj else j
        small_contact = (
            e["contact_fraction_i"] if small == i else e["contact_fraction_j"]
        )
        prior = dict(
            small_contact_fraction=float(small_contact),
            local_alignment=0.0,
            local_axis_agreement=0.0,
            axial_seam_spread=1.0e6,
            depth_ratio=e["interface_depth"] / max(width / 2, min(spacing)),
            interface_planarity=e["interface_planarity"],
        )
        if e["direct"] < 3 or not np.any(e["interface_normal"]):
            e["shape_prior"] = prior
            continue
        # Do the costlier local PCA only for a seam that occupies a measurable
        # part of at least one surface, or for a possible enclosed fragment.
        if (
            max(e["contact_fraction_i"], e["contact_fraction_j"]) < 0.01
            and small_contact < 0.35
        ):
            e["shape_prior"] = prior
            continue
        center = e["interface_center"]
        normal = e["interface_normal"]
        local_axes = []
        local_elongations = []
        alignments = []
        spreads = []
        for label_id in (i, j):
            p = props[label_id]
            coords = p["coords"].astype(np.float64, copy=False)
            nearby = coords[np.linalg.norm(coords - center, axis=1) <= radius]
            if len(nearby) < 8:
                nearby = coords
            local_center = nearby.mean(0)
            covariance = (
                (nearby - local_center).T @ (nearby - local_center)
                / max(1, len(nearby))
            )
            covariance += np.diag(spacing**2 / 12)
            eig, vec = np.linalg.eigh(covariance)
            axis = vec[:, -1]
            elongation = float(np.sqrt(eig[-1] / max(eig[-2], 1e-12)))
            tangent_alignment = abs(float(axis @ normal))
            # A very short/isotropic fragment has no reliable PCA tangent. Its
            # vector from the seam to its mass center is the safer endpoint cue.
            center_vector = p["center"] - center
            center_norm = np.linalg.norm(center_vector)
            center_alignment = (
                abs(float(center_vector @ normal)) / center_norm
                if center_norm > 1e-9
                else 0.0
            )
            alignment = (
                tangent_alignment
                if elongation >= stable_elongation
                else max(tangent_alignment, center_alignment)
            )
            spread = float(
                np.sqrt(max(0.0, axis @ e["interface_cov"] @ axis))
                / max(width, min(spacing))
            )
            local_axes.append(axis)
            local_elongations.append(elongation)
            alignments.append(alignment)
            if elongation >= stable_elongation:
                spreads.append(spread)
        if len(spreads) == 0:
            spreads.append(0.0)
        prior.update(
            local_alignment=float(min(alignments)),
            local_axis_agreement=float(abs(local_axes[0] @ local_axes[1])),
            axial_seam_spread=float(max(spreads)),
            local_elongation_i=float(local_elongations[0]),
            local_elongation_j=float(local_elongations[1]),
        )
        e["shape_prior"] = prior

    # Coordinates are no longer needed after edge-local measurements. Removing
    # them before iterative group merging keeps peak memory bounded.
    for p in props.values():
        p.pop("coords", None)


def _repair_frame_once(source, raw, spacing, cfg, temporal_maps=()):
    if source.shape != raw.shape:
        raise ValueError("Raw/mask shape mismatch")
    if not np.issubdtype(source.dtype, np.integer) or source.min() < 0:
        raise ValueError("Labels must be nonnegative integers")
    original_ids = np.unique(source)
    original_ids = original_ids[original_ids > 0]
    empty_stats = dict(
        before=0,
        after=0,
        merged=0,
        deleted_voxels=0,
        added_voxels=0,
        foreground_before=0,
        foreground_after=0,
        merge_reasons={},
    )
    if not len(original_ids):
        return source.astype(np.uint16), empty_stats, [], {}, []
    # Compact IDs prevent max-label-sized arrays for sparse identifiers.
    fg = source > 0
    pts = np.argwhere(fg)
    lo = np.maximum(0, pts.min(0) - np.ceil(0.7 / spacing).astype(int) - 1)
    hi = np.minimum(source.shape, pts.max(0) + np.ceil(0.7 / spacing).astype(int) + 2)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local = source[crop]
    labels = np.where(local > 0, np.searchsorted(original_ids, local) + 1, 0).astype(
        np.int32
    )
    raw_snr, bg, noise = signal_score(raw)
    snr = ndi.gaussian_filter(raw_snr[crop], (0.35, 0.65, 0.65))
    props = features(labels, snr, spacing, keep_coords=True)
    sizes = np.array([p["n"] for p in props.values()])
    # Volume is a broad upper bound, NOT a learned normal-chromosome reference.
    vref = float(np.percentile(sizes, 85))
    inside = ndi.distance_transform_edt(labels > 0, sampling=spacing)
    width = max(
        2 * float(np.percentile(inside[labels > 0], 80)), 2 * float(min(spacing))
    )
    reach = max(float(min(spacing)), cfg.max_gap_width * width / 2)
    edges, dist, nearest = collect_edges(labels, snr, spacing, reach, inside)
    add_shape_priors(edges, props, spacing, width)
    parent = np.arange(len(original_ids) + 1)
    groups = dict(props)
    operations = []
    uncertain = []
    for e in edges:
        p, q = e["p"], e["q"]
        steps = int(np.max(np.abs(p - q))) * 2 + 1
        coords = np.unique(
            np.rint(np.linspace(p, q, max(steps, 2))).astype(int), axis=0
        )
        e["path"] = coords
    edges_by_label = {i: [] for i in props}
    for e in edges:
        edges_by_label[e["i"]].append(e)
        edges_by_label[e["j"]].append(e)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def count_temporal_votes(a, b):
        votes = 0
        for mapping in temporal_maps:
            aa = {mapping.get(int(original_ids[i - 1])) for i in a["members"]}
            bb = {mapping.get(int(original_ids[i - 1])) for i in b["members"]}
            if None not in aa | bb and len(aa | bb) == 1:
                votes += 1
        return votes

    def shared_fragment_evidence(subject, destination_root):
        """Aggregate one chip's contact with members of an assembled parent.

        Pairwise containment misses a seam chip when roughly half of its surface
        touches each half of the same chromosome.  Only a still-singleton,
        very small label can use this rule, and the destination must already
        contain at least two independently joined labels.
        """
        if groups[root(subject)]["members"] != [subject]:
            return None
        if props[subject]["n"] > cfg.shared_fragment_max_vref_fraction * vref:
            return None
        destination = groups[destination_root]
        if len(destination["members"]) < 2:
            return None
        contacts = []
        for candidate in edges_by_label[subject]:
            if candidate["direct"] <= 0:
                continue
            if candidate["i"] == subject:
                other = candidate["j"]
                fraction = candidate["contact_fraction_i"]
            else:
                other = candidate["i"]
                fraction = candidate["contact_fraction_j"]
            if root(other) != destination_root:
                continue
            signal_floor = 0.75 * min(props[subject]["snr"], props[other]["snr"])
            if candidate["signal"] < signal_floor:
                continue
            contacts.append(
                dict(
                    member=other,
                    fraction=float(fraction),
                    direct_faces=int(candidate["direct"]),
                    interface_snr=float(candidate["signal"]),
                )
            )
        contacts.sort(key=lambda x: x["fraction"], reverse=True)
        if (
            len(contacts) < 2
            or contacts[1]["fraction"] < cfg.shared_fragment_min_each_contact
            or sum(x["fraction"] for x in contacts)
            < cfg.shared_fragment_min_total_contact
        ):
            return None
        return dict(
            total_contact_fraction=float(
                min(1.0, sum(x["fraction"] for x in contacts))
            ),
            contacts=contacts,
        )

    def micro_fragment_evidence(subject, destination_root, edge):
        """Recognize a minute chip attached substantially to a much larger body.

        A strict 55% surface-containment threshold is appropriate for larger
        labels, but a one-voxel-thick seam chip can expose much of its surface to
        background while still occupying an internal segmentation break.  This
        relaxed branch is limited to <=3% of the frame reference volume and a
        parent at least four times larger; a merely touching external speck does
        not have enough physical contact to pass.
        """
        subject_root = root(subject)
        if subject_root == destination_root:
            return None
        if groups[subject_root]["members"] != [subject]:
            return None
        subject_props = props[subject]
        destination = groups[destination_root]
        if subject_props["n"] > cfg.micro_fragment_max_vref_fraction * vref:
            return None
        if destination["n"] < cfg.micro_fragment_min_parent_ratio * subject_props["n"]:
            return None
        fraction = (
            edge["contact_fraction_i"]
            if edge["i"] == subject
            else edge["contact_fraction_j"]
        )
        if edge["direct"] < 3 or fraction < cfg.micro_fragment_min_contact:
            return None
        other = edge["j"] if edge["i"] == subject else edge["i"]
        signal_floor = 0.75 * min(subject_props["snr"], props[other]["snr"])
        if edge["signal"] < signal_floor:
            return None
        return dict(
            subject=int(subject),
            contact_fraction=float(fraction),
            parent_volume_ratio=float(destination["n"] / subject_props["n"]),
            direct_faces=int(edge["direct"]),
            interface_snr=float(edge["signal"]),
        )

    def seam_connector_evidence(left_root, right_root):
        """Find a tiny seam label that independently contacts both candidate roots.

        This reverses the old ordering dependency: previously a chip shared by
        two halves could be absorbed only *after* the halves had joined.  Here a
        strongly enclosed chip may first prove that the two roots are parts of
        one cut body; the existing shared-fragment rule then absorbs the chip.
        """
        if left_root == right_root:
            return None
        left_members = set(groups[left_root]["members"])
        right_members = set(groups[right_root]["members"])

        def neighbors(members):
            found = set()
            for member in members:
                for candidate in edges_by_label[member]:
                    if candidate["direct"] <= 0:
                        continue
                    found.add(
                        candidate["j"]
                        if candidate["i"] == member
                        else candidate["i"]
                    )
            return found

        possible = neighbors(left_members) & neighbors(right_members)
        evidence = []
        for subject in possible:
            subject_root = root(subject)
            if subject_root in {left_root, right_root}:
                continue
            if groups[subject_root]["members"] != [subject]:
                continue
            if props[subject]["n"] > cfg.seam_connector_max_vref_fraction * vref:
                continue
            by_root = {left_root: [], right_root: []}
            for candidate in edges_by_label[subject]:
                if candidate["direct"] <= 0:
                    continue
                other = (
                    candidate["j"]
                    if candidate["i"] == subject
                    else candidate["i"]
                )
                destination_root = root(other)
                if destination_root not in by_root:
                    continue
                fraction = (
                    candidate["contact_fraction_i"]
                    if candidate["i"] == subject
                    else candidate["contact_fraction_j"]
                )
                signal_floor = 0.75 * min(
                    props[subject]["snr"], props[other]["snr"]
                )
                if candidate["signal"] < signal_floor:
                    continue
                by_root[destination_root].append(
                    dict(
                        member=int(other),
                        fraction=float(fraction),
                        direct_faces=int(candidate["direct"]),
                        interface_snr=float(candidate["signal"]),
                    )
                )
            left_fraction = sum(x["fraction"] for x in by_root[left_root])
            right_fraction = sum(x["fraction"] for x in by_root[right_root])
            total = left_fraction + right_fraction
            if (
                left_fraction < cfg.seam_connector_min_each_contact
                or right_fraction < cfg.seam_connector_min_each_contact
                or total < cfg.seam_connector_min_total_contact
            ):
                continue
            evidence.append(
                dict(
                    subject=int(subject),
                    total_contact_fraction=float(min(1.0, total)),
                    left_contact_fraction=float(left_fraction),
                    right_contact_fraction=float(right_fraction),
                    contacts=by_root[left_root] + by_root[right_root],
                )
            )
        if not evidence:
            return None
        return max(evidence, key=lambda item: item["total_contact_fraction"])

    def evaluate(e):
        i, j = root(e["i"]), root(e["j"])
        if i == j:
            return None
        a, b = groups[i], groups[j]
        c = combine(a, b)
        over_volume_limit = c["n"] > cfg.group_volume_factor * vref
        min_signal = max(cfg.strong_snr, cfg.interface_ratio * min(a["snr"], b["snr"]))
        if e["signal"] < min_signal:
            return None
        if e["gap"] > cfg.max_gap_width * width:
            return None
        if e["gap"] > 0:
            path = e["path"]
            values = labels[tuple(path.T)]
            pathroots = np.array([root(int(v)) if v else 0 for v in values])
            if np.any((pathroots != 0) & (pathroots != i) & (pathroots != j)):
                return None
            new = values == 0
            if new.any():
                c["reserved"] += int(new.sum())
                if c["reserved"] > cfg.max_added_fraction * c["n"]:
                    return None
                if np.min(snr[tuple(path[new].T)]) < cfg.strong_snr:
                    return None
                if np.percentile(raw_snr[crop][tuple(path[new].T)], 25) < 2:
                    return None
        delta = b["center"] - a["center"]
        direction = delta / max(np.linalg.norm(delta), 1e-9)
        align_a = abs(float(direction @ a["axis"]))
        align_b = abs(float(direction @ b["axis"]))
        axis_agree = abs(float(a["axis"] @ b["axis"]))
        elong_a = np.sqrt(a["eig"][-1] / a["eig"][-2])
        elong_b = np.sqrt(b["eig"][-1] / b["eig"][-2])
        transverse = 4 * np.sqrt(c["eig"][-2])
        original_width = max(
            4 * np.sqrt(a["eig"][-2]), 4 * np.sqrt(b["eig"][-2]), width
        )
        ratio = min(a["n"], b["n"]) / max(a["n"], b["n"])
        tiny = ratio < 0.18
        prior = e["shape_prior"]
        initial_small = (
            e["i"]
            if props[e["i"]]["n"] <= props[e["j"]]["n"]
            else e["j"]
        )
        initial_ratio = min(props[e["i"]]["n"], props[e["j"]]["n"]) / max(
            props[e["i"]]["n"], props[e["j"]]["n"]
        )
        small_not_yet_absorbed = groups[root(initial_small)]["members"] == [
            initial_small
        ]
        pair_embedded = (
            e["direct"] >= 3
            and small_not_yet_absorbed
            and initial_ratio <= cfg.embedded_max_volume_ratio
            and prior["small_contact_fraction"] >= cfg.embedded_contact_fraction
            and e["signal"] >= 0.75 * min(a["snr"], b["snr"])
        )
        micro_subject = None
        micro_info = None
        if cfg.continuity_only:
            for subject, destination_root in (
                (e["i"], root(e["j"])),
                (e["j"], root(e["i"])),
            ):
                evidence = micro_fragment_evidence(
                    subject, destination_root, e
                )
                if evidence is not None and (
                    micro_info is None
                    or evidence["contact_fraction"]
                    > micro_info["contact_fraction"]
                ):
                    micro_subject = subject
                    micro_info = evidence
        micro_embedded = micro_subject is not None
        smooth_cut = (
            e["direct"] >= 6
            and min(e["contact_fraction_i"], e["contact_fraction_j"])
            >= cfg.smooth_cut_min_contact_fraction
            and prior["local_alignment"] >= cfg.smooth_cut_min_alignment
            and prior["local_axis_agreement"]
            >= cfg.smooth_cut_min_axis_agreement
            and prior["axial_seam_spread"]
            <= cfg.smooth_cut_max_axial_spread
            and prior["depth_ratio"] >= cfg.smooth_cut_min_depth_ratio
            and prior["interface_planarity"] >= 0.30
        )
        relaxed_cut = (
            cfg.continuity_only
            and not smooth_cut
            and e["direct"] >= cfg.relaxed_cut_min_direct_faces
            and initial_ratio >= cfg.relaxed_cut_min_volume_ratio
            and min(e["contact_fraction_i"], e["contact_fraction_j"])
            >= cfg.relaxed_cut_min_contact_fraction
            and prior["local_alignment"] >= cfg.relaxed_cut_min_alignment
            and prior["local_axis_agreement"]
            >= cfg.relaxed_cut_min_axis_agreement
            and prior["axial_seam_spread"]
            <= cfg.relaxed_cut_max_axial_spread
            and prior["depth_ratio"] >= cfg.relaxed_cut_min_depth_ratio
            and prior["interface_planarity"] >= cfg.relaxed_cut_min_planarity
            and e["signal"] >= cfg.relaxed_cut_min_signal_ratio * min_signal
            and transverse <= cfg.relaxed_cut_max_width_factor * original_width
        )
        temporal_vote_count = count_temporal_votes(a, b)
        temporal = temporal_vote_count > 0
        temporal_cobody = (
            cfg.continuity_only
            and temporal_vote_count >= cfg.temporal_cobody_min_votes
            and e["direct"] >= cfg.temporal_cobody_min_direct_faces
            and min(e["contact_fraction_i"], e["contact_fraction_j"])
            >= cfg.temporal_cobody_min_contact
            and prior["depth_ratio"] >= cfg.temporal_cobody_min_depth_ratio
            and prior["interface_planarity"]
            >= cfg.temporal_cobody_min_planarity
            and transverse
            <= cfg.temporal_cobody_max_width_factor * original_width
            and c["n"] <= cfg.group_volume_factor * vref
        )
        connector_info = (
            seam_connector_evidence(i, j) if cfg.continuity_only else None
        )
        connector_supported = (
            connector_info is not None
            and e["direct"] >= cfg.seam_connector_min_direct_faces
            and ratio >= cfg.seam_connector_min_volume_ratio
            and prior["local_axis_agreement"]
            >= cfg.seam_connector_min_axis_agreement
            and prior["depth_ratio"] >= cfg.seam_connector_min_depth_ratio
            and prior["interface_planarity"]
            >= cfg.seam_connector_min_planarity
            and transverse
            <= cfg.seam_connector_max_width_factor * original_width
        )
        shared_subject = None
        shared_info = None
        if cfg.continuity_only:
            for subject, destination_root in (
                (e["i"], root(e["j"])),
                (e["j"], root(e["i"])),
            ):
                if root(subject) == destination_root:
                    continue
                evidence = shared_fragment_evidence(subject, destination_root)
                if evidence is not None and (
                    shared_info is None
                    or evidence["total_contact_fraction"]
                    > shared_info["total_contact_fraction"]
                ):
                    shared_subject = subject
                    shared_info = evidence
        joint_embedded = shared_subject is not None
        embedded = pair_embedded or joint_embedded or micro_embedded
        if over_volume_limit:
            extra_volume = max(
                cfg.shared_fragment_max_vref_fraction,
                cfg.micro_fragment_max_vref_fraction,
                cfg.seam_connector_max_vref_fraction,
            )
            if not (
                (joint_embedded or micro_embedded or connector_supported)
                and c["n"]
                <= (cfg.group_volume_factor + extra_volume) * vref
            ):
                return None
        if cfg.continuity_only and not (
            embedded
            or smooth_cut
            or relaxed_cut
            or temporal_cobody
            or connector_supported
        ):
            return None
        # Strong local evidence can accommodate a curved chromosome whose union
        # is wider in global PCA. All other candidates retain the old bound.
        max_width_factor = (
            cfg.seam_connector_max_width_factor
            if connector_supported
            else cfg.temporal_cobody_max_width_factor
            if temporal_cobody
            else 1.65
            if smooth_cut
            else cfg.relaxed_cut_max_width_factor
            if relaxed_cut
            else 1.18
        )
        if transverse > max_width_factor * original_width:
            return None
        # Parallel side-by-side complete rods are not end-to-end fragments.
        end_to_end = (align_a >= cfg.min_direction_cos or elong_a < 1.4) and (
            align_b >= cfg.min_direction_cos or elong_b < 1.4
        )
        end_to_end &= axis_agree >= 0.65 or min(elong_a, elong_b) < 1.4
        side_fragment = (
            tiny
            and e["direct"] >= 4
            and e["signal"] >= 0.9 * min(a["snr"], b["snr"])
        )
        if not (
            end_to_end
            or side_fragment
            or embedded
            or smooth_cut
            or relaxed_cut
            or temporal_cobody
            or connector_supported
        ):
            return None
        # Isotropic, untracked fragments need stronger interface evidence.
        if (
            min(elong_a, elong_b) < 1.4
            and not tiny
            and not temporal
            and not embedded
            and not smooth_cut
            and not relaxed_cut
            and not temporal_cobody
            and not connector_supported
            and e["signal"] < 0.9 * min(a["snr"], b["snr"])
        ):
            return None
        if e["contacts"] < 3:
            return None
        strength = e["signal"] / max(1.0, min(a["snr"], b["snr"]))
        reason = "geometric_continuation"
        prior_bonus = 0.0
        merge_prior = dict(prior)
        if joint_embedded:
            reason = "shared_seam_fragment"
            merge_prior["shared_fragment"] = shared_info
            prior_bonus = 0.30 * shared_info["total_contact_fraction"]
        elif micro_embedded:
            reason = "micro_fragment"
            merge_prior["micro_fragment"] = micro_info
            prior_bonus = (
                0.30 * micro_info["contact_fraction"]
                + 0.03
                * min(10.0, np.log2(micro_info["parent_volume_ratio"]))
            )
        elif connector_supported:
            reason = "connector_supported_split"
            merge_prior["seam_connector"] = connector_info
            prior_bonus = (
                0.35 * connector_info["total_contact_fraction"]
                + 0.10 * prior["local_axis_agreement"]
                + 0.05 * prior["interface_planarity"]
            )
        elif temporal_cobody:
            reason = "temporal_cobody_split"
            merge_prior["temporal_cobody_votes"] = int(
                temporal_vote_count
            )
            prior_bonus = (
                0.12 * temporal_vote_count
                + 0.12
                * min(e["contact_fraction_i"], e["contact_fraction_j"])
                + 0.05 * prior["interface_planarity"]
            )
        elif pair_embedded:
            reason = "embedded_fragment"
            prior_bonus = 0.30 * prior["small_contact_fraction"]
        elif relaxed_cut:
            reason = "high_confidence_transverse_cut"
            prior_bonus = (
                0.20 * prior["local_alignment"]
                + 0.10 * prior["local_axis_agreement"]
                - 0.05 * prior["axial_seam_spread"]
                + 0.05 * min(2.0, e["signal"] / max(min_signal, 1e-9))
            )
        elif smooth_cut:
            reason = "smooth_transverse_cut"
            prior_bonus = (
                0.18 * prior["local_alignment"]
                + 0.08 * prior["local_axis_agreement"]
                - 0.06 * prior["axial_seam_spread"]
            )
        score = float(
            strength
            + 0.15 * min(align_a, align_b)
            + 0.15 * temporal
            + prior_bonus
            - 0.15 * e["gap"] / max(width, 1e-9)
        )
        embedded_subject = (
            root(shared_subject)
            if joint_embedded
            else root(micro_subject)
            if micro_embedded
            else root(initial_small)
            if pair_embedded
            else None
        )
        return score, i, j, c, temporal, reason, embedded_subject, merge_prior

    merges = 0
    accepted_edges = []
    # Recompute complete-group geometry after every accepted edge. No transitive
    # connected-components merging of an initially accepted graph.
    while True:
        eligible = [(evaluate(e), idx) for idx, e in enumerate(edges)]
        eligible = [(v, idx) for v, idx in eligible if v is not None]
        if not eligible:
            break
        # A connector-backed parent merge must precede absorption of the
        # connector itself; otherwise the evidence disappears and the result
        # depends on score order. The connector conjunction is deliberately
        # stricter than the ordinary micro-fragment rule.
        eligible.sort(
            key=lambda x: (
                x[0][5] == "connector_supported_split",
                x[0][0],
            ),
            reverse=True,
        )
        chosen = None
        for value, idx in eligible:
            score, i, j, c, temporal, reason, embedded_subject, merge_prior = value
            competitors = []
            for v, k in eligible:
                if k == idx or {v[1], v[2]} == {i, j}:
                    continue
                if embedded_subject is not None:
                    conflict = (
                        v[6] == embedded_subject
                        if v[6] is not None
                        else embedded_subject in {v[1], v[2]}
                    )
                elif v[6] is not None:
                    conflict = v[6] in {i, j}
                else:
                    conflict = bool({v[1], v[2]} & {i, j})
                if conflict:
                    competitors.append(v[0])
            decisive = temporal or reason == "connector_supported_split"
            if competitors and score - max(competitors) < 0.08 and not decisive:
                continue
            chosen = (value, idx)
            break
        if chosen is None:
            uncertain.extend(
                dict(
                    i=int(original_ids[edges[idx]["i"] - 1]),
                    j=int(original_ids[edges[idx]["j"] - 1]),
                    reason="competing_candidates",
                    score=float(v[0]),
                )
                for v, idx in eligible
            )
            break
        (
            score,
            i,
            j,
            c,
            temporal,
            reason,
            embedded_subject,
            merge_prior,
        ), idx = chosen
        e = edges[idx]
        parent[j] = i
        groups[i] = c
        del groups[j]
        merges += 1
        accepted_edges.append(e)
        operations.append(
            dict(
                action="merge",
                members=[int(original_ids[k - 1]) for k in c["members"]],
                score=score,
                interface_snr=e["signal"],
                gap_um=e["gap"],
                temporal_support=temporal,
                reason=reason,
                shape_prior=merge_prior,
            )
        )
    lut = np.array([root(i) for i in range(len(parent))], np.int32)
    output = lut[labels]
    original_total = int((labels > 0).sum())
    added = 0
    deleted = 0
    per_group_added = {i: 0 for i in groups}
    # Delete only extreme low-signal residual specks. Signal-bearing small labels
    # remain independent or are merged above; no size-only pruning.
    max_delete = int(cfg.max_deleted_fraction * original_total)
    deletion_groups = [] if cfg.continuity_only else list(groups.items())
    for i, p in deletion_groups:
        limit = min(8, max(1, int(0.01 * vref)))
        if p["n"] <= limit and p["snr"] < 1.5 and deleted + p["n"] <= max_delete:
            where = output == i
            # A bright single voxel in the unsmoothed raw image is protected.
            if np.max(raw_snr[crop][where]) >= 3:
                continue
            if any(
                any(int(original_ids[k - 1]) in m for k in p["members"])
                for m in temporal_maps
            ):
                continue
            output[where] = 0
            deleted += p["n"]
            operations.append(
                dict(
                    action="delete_noise",
                    members=[int(original_ids[k - 1]) for k in p["members"]],
                    voxels=p["n"],
                )
            )
            del groups[i]
    # Fill small enclosed 3D cavities, never 2D holes or other labels.
    hole_objects = [] if cfg.continuity_only else ndi.find_objects(output)
    for i, sl in enumerate(hole_objects, 1):
        if sl is None or i not in groups:
            continue
        pad = tuple(
            slice(max(0, s.start - 1), min(output.shape[k], s.stop + 1))
            for k, s in enumerate(sl)
        )
        sub = output[pad]
        mask = sub == i
        holes = ndi.binary_fill_holes(mask) & ~mask
        cc, n = ndi.label(holes)
        for j, hs in enumerate(ndi.find_objects(cc), 1):
            if hs is None:
                continue
            h = cc[hs] == j
            size = int(h.sum())
            if size > max(1, int(0.005 * groups[i]["n"])):
                continue
            hole_budget = min(
                0.02 * groups[i]["n"],
                cfg.max_added_fraction * groups[i]["n"] - groups[i]["reserved"],
            )
            if per_group_added[i] + size > hole_budget:
                continue
            if np.any(sub[hs][h] != 0) or np.median(snr[pad][hs][h]) < 2:
                continue
            sub[hs][h] = i
            added += size
            per_group_added[i] += size
            positions = (
                np.argwhere(h)
                + np.array([s.start for s in pad])
                + np.array([s.start for s in hs])
                + lo
            )
            operations.append(
                dict(
                    action="fill_hole",
                    members=[int(original_ids[k - 1]) for k in groups[i]["members"]],
                    voxels=size,
                    coordinates_zyx=positions.tolist(),
                )
            )
    # Connect only the specific accepted short gap; verify all path voxels.
    for e in ([] if cfg.continuity_only else accepted_edges):
        i = root(e["i"])
        if i not in groups or e["gap"] == 0:
            continue
        coords = e["path"]
        vox = tuple(coords.T)
        existing = output[vox]
        if (
            np.any((existing != 0) & (existing != i))
            or np.min(snr[vox]) < cfg.strong_snr
        ):
            raise RuntimeError("Accepted bridge became invalid; output not committed")
        new = existing == 0
        n = int(new.sum())
        if per_group_added[i] + n > cfg.max_added_fraction * groups[i]["n"]:
            raise RuntimeError("Accepted bridge budget invalid; output not committed")
        output[vox] = i
        added += n
        per_group_added[i] += n
        if n:
            operations.append(
                dict(
                    action="bridge",
                    members=[int(original_ids[k - 1]) for k in groups[i]["members"]],
                    voxels=n,
                    coordinates_zyx=(coords[new] + lo).tolist(),
                )
            )
    # Very local boundary recovery, constrained by raw signal, competition and a
    # per-instance total budget. Do not fill the midline between nearby instances.
    nearest_root = lut[nearest]
    competition = ndi.maximum_filter(nearest_root, size=3) != ndi.minimum_filter(
        nearest_root, size=3
    )
    can_grow = (
        (not cfg.continuity_only)
        & (output == 0)
        & (dist <= cfg.boundary_growth_width * width)
        & ~competition
        & (snr >= cfg.strong_snr)
        & (raw_snr[crop] >= 2)
    )
    all_candidates = np.argwhere(can_grow)
    by_group = {}
    if len(all_candidates):
        ids = nearest_root[tuple(all_candidates.T)]
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        all_candidates = all_candidates[order]
        keys, starts, counts = np.unique(ids, return_index=True, return_counts=True)
        by_group = {
            int(k): all_candidates[start : start + count]
            for k, start, count in zip(keys, starts, counts)
        }
    for i, p in groups.items():
        budget = int(cfg.max_added_fraction * p["n"]) - per_group_added[i]
        if budget <= 0 or i not in by_group:
            continue
        candidates = by_group[i]
        candidates = candidates[snr[tuple(candidates.T)] >= 0.6 * p["snr"]]
        if not len(candidates):
            continue
        strengths = snr[tuple(candidates.T)] / (
            1 + dist[tuple(candidates.T)] / min(spacing)
        )
        positions = candidates[np.argsort(strengths)[-budget:]]
        output[tuple(positions.T)] = i
        added += len(positions)
        per_group_added[i] += len(positions)
        operations.append(
            dict(
                action="boundary_fill",
                members=[int(original_ids[k - 1]) for k in p["members"]],
                voxels=len(positions),
                coordinates_zyx=(positions + lo).tolist(),
            )
        )
    remaining = np.unique(output)
    remaining = remaining[remaining > 0]
    mapping = {}
    reindex = np.zeros(len(parent), np.uint32)
    for new, i in enumerate(remaining, 1):
        reindex[i] = new
        for old in groups[int(i)]["members"]:
            mapping[str(int(original_ids[old - 1]))] = new
    for i in original_ids:
        mapping.setdefault(str(int(i)), 0)
    dtype = np.uint16 if len(remaining) <= 65535 else np.uint32
    full = np.zeros(source.shape, dtype)
    full[crop] = reindex[output].astype(dtype)
    after = int((full > 0).sum())
    assert after == original_total + added - deleted
    assert np.all(
        (full[source > 0] > 0)
        | np.isin(source[source > 0], [int(k) for k, v in mapping.items() if v == 0])
    )
    stats = dict(
        before=len(original_ids),
        after=len(remaining),
        merged=merges,
        deleted_voxels=deleted,
        added_voxels=added,
        foreground_before=original_total,
        foreground_after=after,
        background=bg,
        noise=noise,
        width_um=width,
        volume_reference_voxels=vref,
        candidate_edges=len(edges),
        uncertain=len(uncertain),
        merge_reasons={
            reason: sum(
                op.get("reason") == reason
                for op in operations
                if op.get("action") == "merge"
            )
            for reason in sorted(
                {
                    op.get("reason")
                    for op in operations
                    if op.get("action") == "merge"
                }
            )
        },
        crop=[lo.tolist(), hi.tolist()],
    )
    return full, stats, operations, mapping, uncertain


def _reverse_mapping(original_to_current):
    """Return current-label -> sorted original-label members."""
    reverse = {}
    for old, current in original_to_current.items():
        current = int(current)
        if current > 0:
            reverse.setdefault(current, []).append(int(old))
    for members in reverse.values():
        members.sort()
    return reverse


def _translate_operations(operations, original_to_current, pass_index):
    """Express later fixed-point operations in the frame's original IDs."""
    reverse = _reverse_mapping(original_to_current)
    translated = []
    for operation in operations:
        item = dict(operation)
        if "members" in item:
            members = {
                old
                for current in item["members"]
                for old in reverse.get(int(current), [int(current)])
            }
            item["members"] = sorted(members)
        item["continuity_pass"] = int(pass_index)
        translated.append(item)
    return translated


def _translate_uncertain(uncertain, original_to_current, pass_index):
    """Keep scalar compatibility and retain every original member explicitly."""
    reverse = _reverse_mapping(original_to_current)
    translated = []
    for candidate in uncertain:
        item = dict(candidate)
        for key in ("i", "j"):
            if key not in item:
                continue
            members = reverse.get(int(item[key]), [int(item[key])])
            item[key] = int(members[0])
            if len(members) > 1:
                item[key + "_members"] = members
        item["continuity_pass"] = int(pass_index)
        translated.append(item)
    return translated


def _compose_mapping(original_to_current, current_to_next):
    return {
        int(old): int(current_to_next.get(str(int(current)), 0)) if current else 0
        for old, current in original_to_current.items()
    }


def _remap_temporal_evidence(original_to_current, temporal_maps):
    """Project input-ID temporal correspondences onto a later continuity pass."""
    reverse = _reverse_mapping(original_to_current)
    projected = []
    for mapping in temporal_maps:
        current_mapping = {}
        for current, originals in reverse.items():
            targets = {mapping.get(int(old)) for old in originals}
            if None not in targets and len(targets) == 1:
                current_mapping[int(current)] = int(next(iter(targets)))
        projected.append(current_mapping)
    return projected


def _repair_frame_base(source, raw, spacing, cfg, temporal_maps=()):
    """Repair one frame, iterating continuity-only graph rebuilding to stability.

    A merge can expose a new aggregate seam that is not represented by any one
    edge of the initial label graph.  Rebuilding the graph after a successful
    continuity pass prevents results from depending on how many times the user
    happens to invoke the program.  Every pass uses the same conservative
    signal, shape, volume and temporal gates; no target instance count is used.
    """
    if not cfg.continuity_only:
        return _repair_frame_once(source, raw, spacing, cfg, temporal_maps)

    source_ids = np.unique(source)
    source_ids = source_ids[source_ids > 0]
    original_to_current = {int(label): int(label) for label in source_ids}
    current = source
    all_operations = []
    pass_stats = []
    final_uncertain = []
    # Each productive pass removes at least one label, so this is a strict
    # termination bound rather than an arbitrary tuning parameter.
    max_passes = max(1, len(source_ids))

    for pass_index in range(1, max_passes + 1):
        pass_temporal = _remap_temporal_evidence(
            original_to_current, temporal_maps
        )
        result, stats, operations, mapping, uncertain = _repair_frame_once(
            current, raw, spacing, cfg, pass_temporal
        )
        pass_stats.append(stats)
        all_operations.extend(
            _translate_operations(operations, original_to_current, pass_index)
        )
        final_uncertain = _translate_uncertain(
            uncertain, original_to_current, pass_index
        )
        original_to_current = _compose_mapping(original_to_current, mapping)

        if stats["merged"] == 0:
            current = result
            break
        if stats["after"] >= stats["before"]:
            raise RuntimeError("Productive continuity pass did not reduce labels")
        current = result
    else:
        raise RuntimeError("Continuity fixed-point iteration failed to converge")

    first = pass_stats[0]
    last = pass_stats[-1]
    merged = sum(int(item["merged"]) for item in pass_stats)
    added = sum(int(item["added_voxels"]) for item in pass_stats)
    deleted = sum(int(item["deleted_voxels"]) for item in pass_stats)
    reasons = {}
    for item in pass_stats:
        for reason, count in item.get("merge_reasons", {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
    aggregate = dict(last)
    aggregate.update(
        before=int(first["before"]),
        after=int(last["after"]),
        merged=merged,
        added_voxels=added,
        deleted_voxels=deleted,
        foreground_before=int(first["foreground_before"]),
        foreground_after=int(last["foreground_after"]),
        candidate_edges=sum(int(item.get("candidate_edges", 0)) for item in pass_stats),
        uncertain=len(final_uncertain),
        merge_reasons=dict(sorted(reasons.items())),
        continuity_passes=len(pass_stats),
        productive_continuity_passes=sum(
            int(item["merged"] > 0) for item in pass_stats
        ),
    )
    if aggregate["foreground_after"] != (
        aggregate["foreground_before"] + added - deleted
    ):
        raise RuntimeError("Fixed-point foreground accounting mismatch")
    return (
        current,
        aggregate,
        all_operations,
        {str(old): current for old, current in original_to_current.items()},
        final_uncertain,
    )


def repair_frame(source, raw, spacing, cfg, temporal_maps=()):
    if cfg.readability_refine:
        return readability_repair(source, raw, spacing, cfg)
    if cfg.hybrid_refine:
        return hybrid_repair(source,raw,spacing)[:5]
    result, stats, ops, mapping, uncertain = _repair_frame_base(
        source, raw, spacing, cfg, temporal_maps
    )
    if not cfg.surface_refine:
        return result, stats, ops, mapping, uncertain
    remapped = _remap_temporal_evidence({int(k): v for k,v in mapping.items()}, temporal_maps)
    polished, changes, identity, detail = refine_surface(result, raw, spacing, remapped,
                                                        aggressive=cfg.aggressive_refine)
    ops.extend(_translate_operations(changes, {int(k):v for k,v in mapping.items()}, 0))
    mapping = {k: identity.get(str(v), 0) for k,v in mapping.items()}
    stats = dict(stats)
    stats['after'] = detail['after']
    stats['merged'] += detail['merged']
    stats['added_voxels'] += detail['added_voxels']
    stats['deleted_voxels'] += detail['deleted_voxels']
    stats['foreground_after'] = int((polished > 0).sum())
    stats['surface_refinement'] = detail
    for op in changes:
        if op['action'] == 'merge':
            reason = op['reason']
            stats['merge_reasons'][reason] = stats['merge_reasons'].get(reason, 0) + 1
    assert stats['foreground_after'] == stats['foreground_before'] + stats['added_voxels'] - stats['deleted_voxels']
    return polished, stats, ops, mapping, uncertain


def make_preview(path, raw, before, after, t, spacing):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(11)
    colors = rng.uniform(0.2, 1, (max(int(before.max()), int(after.max())) + 1, 3))
    colors[0] = 0
    after_colors = colors.copy()
    fg = (after > 0) & (before > 0)
    base = int(before.max()) + 1
    pairs, counts = np.unique(
        after[fg].astype(np.int64) * base + before[fg], return_counts=True
    )
    seen = set()
    for ix in np.argsort(counts)[::-1]:
        new, old = divmod(int(pairs[ix]), base)
        if new not in seen:
            after_colors[new] = colors[old]
            seen.add(new)
    z = int(np.argmax((before > 0).sum((1, 2))))
    coords = np.argwhere(before > 0)
    lo = np.maximum(0, coords.min(0) - 5)
    hi = np.minimum(before.shape, coords.max(0) + 6)
    y = int(np.median(coords[:, 1]))
    crop = (slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    vmin, vmax = np.percentile(raw[before > 0], [1, 99.5])
    ax[0, 0].imshow(raw[z][crop], cmap="gray", vmin=vmin, vmax=vmax)
    ax[0, 1].imshow(colors[before[z][crop]])
    ax[0, 2].imshow(after_colors[after[z][crop]])
    sc = (slice(lo[0], hi[0]), slice(lo[2], hi[2]))
    ax[1, 0].imshow(
        raw[:, y, :][sc],
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        aspect=spacing[0] / spacing[2],
    )
    ax[1, 1].imshow(colors[before[:, y, :][sc]], aspect=spacing[0] / spacing[2])
    ax[1, 2].imshow(after_colors[after[:, y, :][sc]], aspect=spacing[0] / spacing[2])
    for a, title in zip(
        ax[0],
        [
            "Raw DAPI",
            f"Before: {len(np.unique(before[before>0]))}",
            f"After: {len(np.unique(after[after>0]))}",
        ],
    ):
        a.set_title(title)
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"{path.stem}\nt={t}, XY at z={z}; XZ at y={y}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def mask_basename(mask_path):
    stem = mask_path.stem
    for suffix in ("_shape_postprocessed", "_shape_continuity", "_shape"):
        if stem.endswith(suffix):
            base = stem.removesuffix(suffix)
            # CHRP3D exports are named {sample}-DAPI_shape.tif while the
            # paired raw file is {sample}-DAPI.tif.  Internally the sample
            # basename must therefore not retain the channel suffix.
            if base.endswith("-DAPI"):
                base = base[:-5]
            return base
    raise ValueError(f"Unrecognized mask suffix: {mask_path.name}")


def process_file(mask_path, raw_dir, out_dir, cfg, frame_indices=None, preview=True):
    base = mask_basename(mask_path)
    raw_path = raw_dir / (base + "-DAPI.tif")
    if not raw_path.exists():
        matches = list(raw_dir.rglob(base + "-DAPI.tif"))
        if len(matches) != 1:
            raise ValueError(f"Expected one matching DAPI for {base}, got {len(matches)}")
        raw_path = matches[0]
    output_suffix = "_shape_continuity.tif" if cfg.continuity_only else "_shape_postprocessed.tif"
    if cfg.aggressive_refine:
        output_suffix = '_shape_aggressive.tif'
    if cfg.hybrid_refine:
        output_suffix = '_shape_postprocessed.tif'
    out = out_dir / (base + output_suffix)
    report = out_dir / "reports" / (base + ".json")
    oplog = out_dir / "operations" / (base + ".jsonl")
    temp = out.with_name(out.name + ".partial")
    for p in (out, report, oplog, temp):
        if p.exists():
            raise FileExistsError(p)
    for p in (out_dir, report.parent, oplog.parent, out_dir / "qc"):
        p.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    mappings = {}
    digest = []
    source_stats = {
        str(p): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        for p in (mask_path, raw_path)
    }
    with tf.TiffFile(mask_path) as mf, tf.TiffFile(raw_path) as rf, oplog.open(
        "x"
    ) as log:
        shape = shape4(mf)
        raw_shape = shape4(rf)
        if shape[0] != raw_shape[0]:
            raise ValueError(
                f"Time axis mismatch: {mask_path.name}: mask={shape}, raw={raw_shape}"
            )
        raw_spacing, meta = get_metadata(rf)
        # ChromoPose3D infers on a fixed canonical grid (typically
        # 64x256x256) and writes labels on that grid.  The source raw can have
        # a larger, non-square FOV.  Bring only the raw intensity into the
        # label grid; the label geometry is not interpolated.  Spacing is
        # scaled so all gap/width priors remain physical.
        spatial_scale = np.asarray(raw_shape[1:], float) / np.asarray(shape[1:], float)
        spacing = raw_spacing * spatial_scale
        ts = list(range(shape[0])) if frame_indices is None else frame_indices
        if any(t < 0 or t >= shape[0] for t in ts):
            raise ValueError("Frame outside stack")
        cache = {}

        def get(t):
            if t not in cache:
                source = read_frame(mf, t)
                raw = read_frame(rf, t)
                if raw.shape != source.shape:
                    zoom = np.asarray(source.shape, float) / np.asarray(raw.shape, float)
                    raw = ndi.zoom(raw.astype(np.float32), zoom, order=1, prefilter=False)
                    if raw.shape != source.shape:
                        raise ValueError(
                            f"Raw resampling failed for {mask_path.name} t={t}: "
                            f"{raw.shape} != {source.shape}"
                        )
                cache[t] = (source, raw)
            return cache[t]

        def planes():
            for t in ts:
                source, raw = get(t)
                maps = []
                temporal = []
                if cfg.temporal and not cfg.hybrid_refine and not cfg.readability_refine:
                    for nt in (t - 1, t + 1):
                        if 0 <= nt < shape[0]:
                            nl, nr = get(nt)
                            mp, info = temporal_mapping(source, raw, nl, nr)
                            maps.append(mp)
                            temporal.append(dict(neighbor=nt, **info))
                result, stats, ops, mapping, uncertain = repair_frame(
                    source, raw, spacing, cfg, maps
                )
                # uint32 avoids per-movie overflow if frame count varies greatly.
                result = result.astype(np.uint32)
                if (cfg.surface_refine or cfg.hybrid_refine or cfg.readability_refine) and np.any((source>0) & (result==0)):
                    # Bright but structurally sub-resolution components are
                    # quarantined, not irreversibly discarded. Preserve IDs.
                    tf.imwrite(out_dir / 'qc' / (base + f'_t{t:03d}_removed_labels.tif'),
                               np.where(result==0,source,0).astype('uint32'),
                               compression='zlib', metadata=dict(axes='ZYX',spacing_um=spacing.tolist()))
                row = dict(t=t, **stats, temporal=temporal)
                rows.append(row)
                mappings[str(t)] = mapping
                for op in ops:
                    log.write(json.dumps(dict(t=t, **op)) + "\n")
                for item in uncertain:
                    log.write(json.dumps(dict(t=t, action="uncertain", **item)) + "\n")
                log.flush()
                digest.append(hashlib.sha256(result.tobytes()).hexdigest())
                print(
                    json.dumps(
                        dict(
                            file=mask_path.name,
                            t=t,
                            before=stats["before"],
                            after=stats["after"],
                            added=stats["added_voxels"],
                            deleted=stats["deleted_voxels"],
                            elapsed=round(time.monotonic() - started, 1),
                        )
                    ),
                    flush=True,
                )
                if (
                    preview
                    and stats["before"]
                    and t in {ts[0], ts[len(ts) // 2], ts[-1]}
                ):
                    make_preview(
                        out_dir / "qc" / (base + f"_t{t:03d}.png"),
                        raw,
                        source,
                        result,
                        t,
                        spacing,
                    )
                # Source IDs must not be presented as stable cross-frame track IDs.
                for key in list(cache):
                    if key < t:
                        del cache[key]
                yield from result

        outmeta = {k: meta[k] for k in ("unit", "spacing", "finterval") if k in meta}
        outmeta.update(axes="TZYX", pipeline=VERSION, source_timepoints=ts)
        tf.imwrite(
            temp,
            planes(),
            shape=(len(ts),) + shape[1:],
            dtype="uint32",
            compression="zlib",
            metadata=outmeta,
            resolution=(1 / spacing[2], 1 / spacing[1]),
            photometric="minisblack",
        )
    with tf.TiffFile(temp) as f:
        assert f.series[0].shape == (len(ts),) + shape[1:]
        for j, h in enumerate(digest):
            assert hashlib.sha256(read_frame(f, j).tobytes()).hexdigest() == h
    for path, st in source_stats.items():
        p = Path(path)
        assert (
            p.stat().st_size == st["bytes"] and p.stat().st_mtime_ns == st["mtime_ns"]
        )
    os.replace(temp, out)
    result = dict(
        version=VERSION,
        config=asdict(cfg),
        mask=str(mask_path),
        raw=str(raw_path),
        output=str(out),
        shape=[len(ts), *shape[1:]],
        spacing_zyx_um=spacing.tolist(),
        source_files=source_stats,
        frames=rows,
        id_mapping=mappings,
        verified_sha256=digest,
        seconds=round(time.monotonic() - started, 2),
        limitations=([
            "Readability mode fills background before absorbing/deleting small fragments",
            "Original substantial instance IDs and all their source voxels are protected",
            "Background ownership is nearest-source distance in physical ZYX coordinates",
            "No large-instance joins, topology splits, target chromosome count or temporal tracking",
            "Fragments and missing signal remain heuristic; removed labels are saved for review",
            "Apply once to model predictions; do not feed previous postprocessed outputs back in",
        ] if cfg.readability_refine else [
            "No learned stage classifier or full lineage tracking",
            "Temporal evidence uses gated rigid/dense XY registration, not full 3D deformation",
            "Conservative rule scores are not calibrated probabilities",
            "No target count; v1.8 can quarantine small components despite brightness or slenderness",
            "Hybrid refinement is per-frame; temporal correspondences do not exempt small fragments",
            "Tubeness and watershed ownership are heuristic proposals, not calibrated probabilities",
        ]),
    )
    report.write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--pattern",
        help="Input glob (default: *_shape.tif, or *_shape_postprocessed.tif in continuity-only mode)",
    )
    p.add_argument(
        "--frames", help="Comma-separated zero-based input timepoints for pilot"
    )
    p.add_argument("--no-temporal", action="store_true")
    p.add_argument(
        "--continuity-only",
        action="store_true",
        help=(
            "Second pass for existing postprocessed masks: only merge enclosed "
            "fragments, smooth cuts and temporally confirmed co-body splits; "
            "preserve foreground exactly"
        ),
    )
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--surface-refine", action="store_true",
                   help="Opt-in experimental local seam and signal-supported boundary repair; allows foreground changes even with --continuity-only")
    p.add_argument('--aggressive-refine',action='store_true',
                   help='High-recall fragment absorption/quarantine and curved skeleton endpoint joins; implies surface refinement')
    p.add_argument('--hybrid-post',action='store_true',help='Legacy v1.9 CHIMP-inspired completion and instance merging')
    p.add_argument('--readability-post', action='store_true',
                   help='Default: nearest-ID completion, small-fragment absorption, then quarantine; never merge large IDs')
    p.add_argument('--legacy-post',action='store_true',help='Use pre-v1.8 behavior')
    p.add_argument("--config", type=Path)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent movies in parallel; 2 recommended for 24 GB RAM",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete outputs with identical config/version/source stat",
    )
    args = p.parse_args()
    cfg = Settings(**json.loads(args.config.read_text())) if args.config else Settings()
    if args.legacy_post or args.surface_refine or args.aggressive_refine:
        cfg.hybrid_refine=False
        cfg.readability_refine=False
    if args.hybrid_post:
        cfg.hybrid_refine=True
        cfg.readability_refine=False
    if args.readability_post:
        if args.hybrid_post or args.legacy_post or args.surface_refine or args.aggressive_refine:
            p.error('Readability mode cannot be combined with legacy/hybrid/aggressive modes')
        cfg.readability_refine=True
        cfg.hybrid_refine=False
    if args.hybrid_post and args.legacy_post:
        p.error('Choose hybrid or legacy, not both')
    if args.no_temporal:
        cfg.temporal = False
    if args.continuity_only:
        cfg.continuity_only = True
    if args.surface_refine:
        cfg.surface_refine = True
    if args.aggressive_refine:
        cfg.surface_refine = True
        cfg.aggressive_refine = True
    if cfg.readability_refine:
        if cfg.continuity_only:
            p.error('Readability mode changes foreground; do not combine with continuity-only')
        if (cfg.readable_fill_radius_um <= 0 or any(
            not 0 < x < 1 for x in (
                cfg.readable_added_fraction, cfg.readable_instance_added_fraction,
                cfg.readable_fragment_vref_fraction, cfg.readable_fragment_parent_fraction,
                cfg.readable_deleted_fraction))):
            p.error('Readability radius must be positive and its fractions must lie between 0 and 1')
    if any(
        not 0 < v <= 1
        for v in (
            cfg.max_gap_width,
            cfg.interface_ratio,
            cfg.min_direction_cos,
            cfg.max_added_fraction,
            cfg.max_deleted_fraction,
            cfg.embedded_contact_fraction,
            cfg.embedded_max_volume_ratio,
            cfg.smooth_cut_min_contact_fraction,
            cfg.smooth_cut_min_alignment,
            cfg.smooth_cut_min_axis_agreement,
            cfg.smooth_cut_max_axial_spread,
            cfg.smooth_cut_min_depth_ratio,
            cfg.relaxed_cut_min_contact_fraction,
            cfg.relaxed_cut_min_alignment,
            cfg.relaxed_cut_min_axis_agreement,
            cfg.relaxed_cut_max_axial_spread,
            cfg.relaxed_cut_min_depth_ratio,
            cfg.relaxed_cut_min_planarity,
            cfg.relaxed_cut_min_volume_ratio,
            cfg.shared_fragment_max_vref_fraction,
            cfg.shared_fragment_min_total_contact,
            cfg.shared_fragment_min_each_contact,
            cfg.micro_fragment_max_vref_fraction,
            cfg.micro_fragment_min_contact,
            cfg.seam_connector_max_vref_fraction,
            cfg.seam_connector_min_total_contact,
            cfg.seam_connector_min_each_contact,
            cfg.seam_connector_min_axis_agreement,
            cfg.seam_connector_min_depth_ratio,
            cfg.seam_connector_min_planarity,
            cfg.seam_connector_min_volume_ratio,
            cfg.temporal_cobody_min_contact,
            cfg.temporal_cobody_min_depth_ratio,
            cfg.temporal_cobody_min_planarity,
        )
    ):
        p.error("Fraction settings must be in (0,1]")
    if not 0 <= cfg.boundary_growth_width <= 1:
        p.error("boundary_growth_width must be in [0,1]")
    if (
        not np.isfinite(cfg.strong_snr)
        or cfg.strong_snr <= 0
        or not np.isfinite(cfg.group_volume_factor)
        or cfg.group_volume_factor <= 0
        or not np.isfinite(cfg.relaxed_cut_max_width_factor)
        or cfg.relaxed_cut_max_width_factor <= 0
        or not np.isfinite(cfg.relaxed_cut_min_signal_ratio)
        or cfg.relaxed_cut_min_signal_ratio <= 0
        or cfg.relaxed_cut_min_direct_faces < 1
        or cfg.micro_fragment_min_parent_ratio <= 1
        or cfg.seam_connector_min_direct_faces < 1
        or cfg.temporal_cobody_min_votes < 1
        or cfg.temporal_cobody_min_direct_faces < 1
        or cfg.seam_connector_max_width_factor <= 0
        or cfg.temporal_cobody_max_width_factor <= 0
    ):
        p.error("Signal, volume, shape and vote settings must be positive")
    pattern = args.pattern or (
        "*_shape_postprocessed.tif" if cfg.continuity_only else "*_shape.tif"
    )
    files = [args.masks] if args.masks.is_file() else sorted(args.masks.rglob(pattern))
    if not files:
        p.error("No masks matched")
    if args.output.resolve() in {args.raw.resolve(), args.masks.resolve()} or any(
        f.resolve().parent == args.output.resolve() for f in files
    ):
        p.error("Output must be a separate directory")
    frames = [int(t) for t in args.frames.split(",")] if args.frames else None
    if frames is not None and (frames != sorted(set(frames)) or min(frames) < 0):
        p.error("--frames must contain strictly increasing nonnegative indices")
    if args.workers < 1:
        p.error("--workers must be positive")
    summaries = []
    pending = []

    def summarize(f, r):
        summaries.append(
            dict(
                file=f.name,
                frames=len(r["frames"]),
                median_before=float(np.median([x["before"] for x in r["frames"]])),
                median_after=float(np.median([x["after"] for x in r["frames"]])),
                added=sum(x["added_voxels"] for x in r["frames"]),
                deleted=sum(x["deleted_voxels"] for x in r["frames"]),
                seconds=r["seconds"],
            )
        )
        (args.output / "summary.json").write_text(
            json.dumps(sorted(summaries, key=lambda x: x["file"]), indent=2)
        )

    for f in files:
        report = args.output / "reports" / (mask_basename(f) + ".json")
        if args.resume and report.exists():
            r = json.loads(report.read_text())
            expected_raw = args.raw / (mask_basename(f) + "-DAPI.tif")
            if not expected_raw.exists():
                matches = list(args.raw.rglob(expected_raw.name))
                if len(matches) != 1:
                    raise ValueError('Resume DAPI pairing is ambiguous')
                expected_raw = matches[0]
            if (
                Path(r["mask"]).resolve() != f.resolve()
                or Path(r["raw"]).resolve() != expected_raw.resolve()
            ):
                raise ValueError("Resume input paths changed")
            if (
                r["version"] != VERSION
                or r["config"] != asdict(cfg)
                or not Path(r["output"]).is_file()
            ):
                raise ValueError("Resume config/version/output mismatch")
            for src, stat in r["source_files"].items():
                if (
                    Path(src).stat().st_size != stat["bytes"]
                    or Path(src).stat().st_mtime_ns != stat["mtime_ns"]
                ):
                    raise ValueError("Resume source changed")
            with tf.TiffFile(f) as source_file:
                expected_ts = (
                    list(range(shape4(source_file)[0])) if frames is None else frames
                )
            if [v["t"] for v in r["frames"]] != expected_ts:
                raise ValueError("Resume frame selection changed")
            summarize(f, r)
        else:
            pending.append(f)
    if args.workers == 1:
        for f in pending:
            summarize(
                f,
                process_file(
                    f, args.raw, args.output, cfg, frames, not args.no_preview
                ),
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {
                pool.submit(
                    process_file,
                    f,
                    args.raw,
                    args.output,
                    cfg,
                    frames,
                    not args.no_preview,
                ): f
                for f in pending
            }
            for job in as_completed(jobs):
                summarize(jobs[job], job.result())

# ---- surface_refinement ----
"""Experimental, signal-constrained mask polishing after identity repair.

No count target, convex hull, 2D hole filling, or replacement of another label.
All thresholds are physical or relative to this frame. Defaults are opt-in.
"""
import numpy as np
from scipy import ndimage as ndi


def refine_surface(source, raw, spacing, temporal_maps=(), aggressive=False, protected_background=None, relabel_only=False):

    if aggressive:
        stage, ops, first_map, info = aggressive_repair(source,raw,spacing,temporal_maps)
        int_map = {int(k):v for k,v in first_map.items()}
        result, polish_ops, second_map, stats = refine_surface(
            stage,raw,spacing,_remap_temporal_evidence(int_map,temporal_maps),
            protected_background=(source>0)&(stage==0))
        ops.extend(_translate_operations(polish_ops,int_map,0))
        mapping = {k:second_map.get(str(v),0) if v else 0 for k,v in first_map.items()}
        stats['merged'] += info['merged']
        stats['deleted_voxels'] += info['deleted_voxels']
        stats['aggressive'] = info
        return result,ops,mapping,stats

    spacing = np.asarray(spacing, float)
    ids = np.unique(source)
    ids = ids[ids > 0]
    if not len(ids):
        return source.copy(), [], {}, dict(merged=0,added_voxels=0,deleted_voxels=0,after=0)
    labels = np.where(source > 0, np.searchsorted(ids, source) + 1, 0).astype('int32')
    snr, _, _ = signal_score(raw)
    smooth_snr = ndi.gaussian_filter(snr, np.maximum(.06 / spacing, .3))
    props = features(labels, smooth_snr, spacing, keep_coords=True)
    vref = float(np.percentile([p['n'] for p in props.values()], 85))
    depth = ndi.distance_transform_edt(labels > 0, sampling=spacing)
    width = max(2 * np.percentile(depth[labels > 0], 80), 2 * min(spacing))
    edges, _, _ = collect_edges(labels, smooth_snr, spacing, 0, depth)
    objects = ndi.find_objects(labels)
    candidates, rejected = [], []
    chip_pairs = set()
    cross = ndi.generate_binary_structure(3, 1)
    # Unlike a whole-object axis, a patch-local tangent accommodates curved rods
    # and separate contacts at the two arms of a U. Require EVERY sizable patch
    # to agree; a crossing or side-by-side contact must not pass on one good patch.
    for e in edges:
        a, b = e['i'], e['j']
        if e['direct'] < 8 or props[a]['n'] + props[b]['n'] > 2 * vref:
            continue
        if e['signal'] < max(3., .85 * min(props[a]['snr'], props[b]['snr'])):
            continue
        small, large = sorted((a,b), key=lambda i: props[i]['n'])
        fraction = e['contact_fraction_i'] if small == a else e['contact_fraction_j']
        competing_contact = sum((q['contact_fraction_i'] if q['i']==small else q['contact_fraction_j'])
                                for q in edges if small in (q['i'],q['j']) and q is not e)
        if (props[small]['n'] <= .10*vref and props[small]['n'] <= .15*props[large]['n']
                and fraction >= .35 and competing_contact <= .10
                and e['interface_depth'] >= .30*width):
            # A curved tip/seam chip can have two opposed interfaces and no
            # meaningful PCA axis. Its sole substantial parent is the evidence.
            candidates.append((1.2+fraction, a,b,[dict(parent_contact=float(fraction),
                                                     competing_contact=float(competing_contact))]))
            chip_pairs.add((a,b))
            continue
        sa, sb = objects[a-1], objects[b-1]
        sl = tuple(slice(max(0, min(x.start, y.start)-1),
                         min(labels.shape[k], max(x.stop, y.stop)+1))
                   for k, (x, y) in enumerate(zip(sa, sb)))
        sub = labels[sl]
        ma, mb = sub == a, sub == b
        seam = (ma & ndi.binary_dilation(mb, cross)) | (mb & ndi.binary_dilation(ma, cross))
        patches, n = ndi.label(seam, np.ones((3,3,3)))
        patch_sizes = np.bincount(patches.ravel())
        good, diagnostics = True, []
        for p in range(1, n+1):
            if patch_sizes[p] < max(6, .08 * seam.sum()):
                continue
            patch = patches == p
            origin = np.array([s.start for s in sl])
            cp = (np.argwhere(patch) + origin) * spacing
            center = cp.mean(0)
            local_radius = max(1.8 * width, 2.1 * max(spacing))
            axes, centers, elongations = [], [], []
            for lab in (a, b):
                pts = props[lab]['coords']
                pts = pts[np.linalg.norm(pts-center, axis=1) < local_radius]
                if len(pts) < 8:
                    good = False
                    break
                ctr = pts.mean(0)
                # Reflect the one-sided neighborhood about the seam center:
                # ordinary centered PCA mistakes a short half-cylinder's wide
                # cut face for its longitudinal axis.
                cov = (pts-center).T @ (pts-center) / len(pts) + np.diag(spacing**2/12)
                eig, vec = np.linalg.eigh(cov)
                axes.append(vec[:,-1]); centers.append(ctr)
                elongations.append(float(np.sqrt(eig[-1]/eig[-2])))
            if len(axes) != 2:
                break
            ca = (np.argwhere(patch & ma)+origin).mean(0) * spacing
            cb = (np.argwhere(patch & mb)+origin).mean(0) * spacing
            normal = cb-ca
            normal /= max(np.linalg.norm(normal), 1e-10)
            alignment = min(abs(ax @ normal) for ax in axes)
            agreement = abs(axes[0] @ axes[1])
            opposite = (centers[0]-center) @ (centers[1]-center) < 0
            local_depth = float(np.percentile(depth[sl][patch], 75))
            ok = (alignment >= .70 and agreement >= .75 and opposite
                  and min(elongations) >= 1.12 and local_depth >= .35 * width)
            diagnostics.append(dict(alignment=float(alignment), agreement=float(agreement),
                                    depth_um=local_depth, passed=bool(ok)))
            good &= ok
        if good and diagnostics:
            score = min(d['alignment'] * d['agreement'] for d in diagnostics)
            candidates.append((score, a, b, diagnostics))
        elif diagnostics:
            rejected.append(dict(i=int(ids[a-1]), j=int(ids[b-1]), patches=diagnostics))
    parent = np.arange(len(ids)+1)
    def root(i):
        while parent[i] != i:
            i = int(parent[i])
        return i
    ops = []
    # At most one new seam per original label per invocation. No transitive
    # avalanche through a dense cluster; competitors remain for review.
    used = set()
    for score, a, b, diagnostics in sorted(candidates, reverse=True):
        competitors = [s for s,x,y,_ in candidates if {x,y} != {a,b} and {x,y}&{a,b}]
        if a in used or b in used or (competitors and score-max(competitors) < .10):
            continue
        parent[b] = a
        used.update((a,b))
        ops.append(dict(action='merge', reason='single_parent_seam_chip' if (a,b) in chip_pairs else 'patch_local_transverse_cut',
                        members=[int(ids[a-1]),int(ids[b-1])], patches=diagnostics))
    lut = np.array([root(i) for i in range(len(parent))])
    output = lut[labels].astype('uint32')
    if relabel_only:
        mapping={str(int(old)):int(root(k)) for k,old in enumerate(ids,1)}
        return output,ops,mapping,dict(added_voxels=0,deleted_voxels=0,
              merged=sum(op['action']=='merge' for op in ops),after=int(len(np.unique(output[output>0]))))
    baseline = output.copy()
    foreground = int((source > 0).sum())
    delete_budget = int(.005 * foreground)
    deleted = 0
    # Remove small disconnected components only when also weak in the actual
    # DAPI, isolated, and not supported by either reliable temporal map.
    for lab, sl0 in enumerate(ndi.find_objects(output), 1):
        if sl0 is None:
            continue
        pad = np.ceil(width / spacing).astype(int) + 2
        sl = tuple(slice(max(0,s.start-int(pad[k])), min(output.shape[k],s.stop+int(pad[k])))
                   for k,s in enumerate(sl0))
        sub = output[sl]
        mask = sub == lab
        cc, n = ndi.label(mask, np.ones((3,3,3)))
        sizes = np.bincount(cc.ravel())
        member_ids = ids[np.array([root(i) == lab for i in range(1,len(ids)+1)])]
        supported = any(any(int(i) in mp for i in member_ids) for mp in temporal_maps)
        other = (sub > 0) & ~mask
        distance = ndi.distance_transform_edt(~other, sampling=spacing) if other.any() else np.full(sub.shape, np.inf)
        for component in range(1,n+1):
            region = cc == component
            count = int(sizes[component])
            if count > max(8, .02*vref) or count + deleted > delete_budget or supported:
                continue
            values = snr[sl][region]
            if np.min(distance[region]) < .5 * width:
                continue
            weak = np.median(values) < 2 and np.percentile(values,90) < 3
            extent_component = np.ptp(np.argwhere(region),axis=0)+1
            unresolved = (count <= .01*vref and extent_component[0] <= 2
                          and bool(temporal_maps))
            if not weak and not unresolved:
                continue
            sub[region] = 0
            deleted += count
            ops.append(dict(action='delete_isolated_weak_component' if weak else 'quarantine_tiny_isolated_component',
                            members=member_ids.astype(int).tolist(), voxels=count))
    # Closing fills only sub-width notches; Gaussian occupancy regularizes the
    # boundary. Never replace foreground of another label, and never fill a dark
    # loop merely to make the chromosome look solid. Keep existing bright tips.
    radius = max(min(spacing), .18 * width)
    extent = np.ceil(radius/spacing).astype(int)
    zz,yy,xx = np.ogrid[tuple(slice(-int(k),int(k)+1) for k in extent)]
    footprint = (zz*spacing[0])**2+(yy*spacing[1])**2+(xx*spacing[2])**2 <= radius**2+1e-12
    proposals = np.zeros(output.shape, np.uint32)
    contested = np.zeros(output.shape, bool)
    for lab, sl0 in enumerate(ndi.find_objects(output),1):
        if sl0 is None:
            continue
        pad = extent*3+2
        sl = tuple(slice(max(0,s.start-int(pad[k])), min(output.shape[k],s.stop+int(pad[k]))) for k,s in enumerate(sl0))
        sub = output[sl]
        mask = sub == lab
        closed = ndi.binary_closing(mask, footprint)
        density = ndi.gaussian_filter(mask.astype('float32'), np.maximum(.10*width/spacing,.25))
        candidate = (closed | (density >= .45)) & (sub == 0)
        if protected_background is not None:
            candidate &= ~protected_background[sl]
        distance = ndi.distance_transform_edt(~mask, sampling=spacing)
        own_snr = float(np.median(smooth_snr[sl][mask]))
        candidate &= (distance <= radius) & (snr[sl] >= 2) & (smooth_snr[sl] >= max(3., .35*own_snr))
        # Keep a one-face clearance to other instances to preserve separability.
        candidate &= ~ndi.binary_dilation((sub>0)&~mask, cross)
        coords = np.argwhere(candidate)
        budget = int(.05 * mask.sum())
        if len(coords) > budget:
            coords = coords[np.argsort(density[tuple(coords.T)])[-budget:]] if budget else coords[:0]
        key = tuple(coords.T)
        proposed = proposals[sl]
        contested[sl][key] |= proposed[key] != 0
        proposed[key] = lab
    add = (proposals>0) & ~contested & (output == 0)
    output[add] = proposals[add]
    for lab, count in zip(*np.unique(output[add],return_counts=True)):
        ops.append(dict(action='signal_supported_surface_fill', members=ids[np.array([root(i)==lab for i in range(1,len(ids)+1)])].astype(int).tolist(), voxels=int(count)))
    mapping = {str(int(old)): int(root(k)) if np.any(output == root(k)) else 0 for k,old in enumerate(ids,1)}
    stats = dict(added_voxels=int(add.sum()), deleted_voxels=deleted,
                 merged=sum(op['action']=='merge' for op in ops),
                 after=int(len(np.unique(output[output>0]))),
                 rejected_seams=rejected, width_um=float(width))
    assert np.all(output[baseline>0][output[baseline>0]>0] == baseline[baseline>0][output[baseline>0]>0])
    return output, ops, mapping, stats

# ---- aggressive_refinement ----
"""Opt-in high-recall fragment absorption and curved-endpoint repair.

No chromosome-count target. Small bright objects may be quarantined, so callers
must retain a removed-label volume and use new output paths.
"""
import heapq
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def endpoint_geometry(mask, spacing, origin, width):
    """Physical endpoint-to-inward tangent from a voxel skeleton graph."""
    sk = skeletonize(np.pad(mask, 1), method='lee') > 0
    xyz = np.argwhere(sk)
    if len(xyz) < 2:
        return [], 0.
    tree = cKDTree(xyz)
    neighbors = tree.query_ball_point(xyz, 1.75)
    ends = [i for i, ns in enumerate(neighbors) if len(ns) == 2]
    physical = (xyz - 1 + origin) * spacing
    result = []
    for start in ends:
        distances = {start: 0.}
        queue = [(0., start)]
        reached = start
        while queue:
            distance, current = heapq.heappop(queue)
            if distance != distances[current]:
                continue
            reached = current
            if distance >= 1.2*width:
                break
            for nxt in neighbors[current]:
                if nxt == current:
                    continue
                value = distance + np.linalg.norm(physical[nxt]-physical[current])
                if value < distances.get(nxt, np.inf):
                    distances[nxt] = value
                    heapq.heappush(queue, (value,nxt))
        tangent = physical[reached]-physical[start]
        length = np.linalg.norm(tangent)
        if length >= .35*width:
            result.append((physical[start], tangent/length))
    span = float(np.linalg.norm(np.ptp(physical,axis=0)))
    return result, span


def aggressive_repair(source, raw, spacing, temporal_maps=(), strict_fragments=False):
    spacing = np.asarray(spacing,float)
    current = source.astype('uint32',copy=True)
    snr,_,_ = signal_score(raw)
    snr = ndi.gaussian_filter(snr, np.maximum(.06/spacing,.3))
    original_ids, sizes0 = np.unique(source[source>0],return_counts=True)
    if not len(sizes0):
        return current, [], {}, dict(merged=0,deleted_voxels=0)
    # Fixed input reference prevents successive passes lowering the fragment bar.
    vref = float(np.percentile(sizes0,85))
    depth = ndi.distance_transform_edt(source>0,sampling=spacing)
    width = max(2*float(np.percentile(depth[source>0],80)),2*min(spacing))
    mapping = {str(int(i)):int(i) for i in original_ids}
    operations = []
    for iteration in range(3):
        props = features(current,snr,spacing)
        objects = ndi.find_objects(current)
        edges,_,_ = collect_edges(current,snr,spacing,0,depth)
        geometry = {}
        def ends(lab):
            if lab not in geometry:
                sl = objects[lab-1]
                geometry[lab] = endpoint_geometry(current[sl]==lab, spacing,
                                                  np.array([s.start for s in sl]),width)
            return geometry[lab][0]
        candidates = []
        for edge in edges:
            a,b = edge['i'],edge['j']
            small,large = sorted((a,b),key=lambda k:props[k]['n'])
            smalln,largen = props[small]['n'],props[large]['n']
            if smalln+largen > 2.5*vref or edge['direct'] < 4:
                continue
            if edge['signal'] < max(3.,.55*min(props[a]['snr'],props[b]['snr'])):
                continue
            contact = edge['contact_fraction_i'] if small==a else edge['contact_fraction_j']
            rivals = [q['contact_fraction_i'] if q['i']==small else q['contact_fraction_j']
                      for q in edges if q is not edge and small in (q['i'],q['j'])]
            dominant = contact >= 1.5*max(rivals,default=0.)
            if (smalln <= .25*vref and smalln <= .45*largen and contact >= .16
                    and dominant and edge['interface_depth'] >= .20*width):
                candidates.append((2.+contact,a,b,'aggressive_parent_absorption',
                                   dict(contact_fraction=float(contact),fraction_of_reference=smalln/vref)))
                continue
            if edge['interface_depth'] < .30*width or min(edge['contact_fraction_i'],edge['contact_fraction_j']) < .02:
                continue
            center = edge['interface_center']
            ea = [x for x in ends(a) if np.linalg.norm(x[0]-center) <= 1.1*width]
            eb = [x for x in ends(b) if np.linalg.norm(x[0]-center) <= 1.1*width]
            valid = []
            for pa,ta in ea:
                for pb,tb in eb:
                    distance = np.linalg.norm(pa-pb)
                    turn = float(ta@tb)
                    # Inward tangents may turn at a bend. They must NOT run in
                    # the same direction as two parallel, adjacent complete rods.
                    if distance <= 1.25*width and turn <= .35:
                        valid.append((1.-.25*turn-.1*distance/width,turn,distance))
            if valid:
                score,turn,distance = max(valid)
                candidates.append((score,a,b,'curved_endpoint_join',
                                   dict(inward_tangent_cosine=turn,endpoint_distance_um=float(distance))))
        parent = {i:i for i in props}
        volumes = {i:p['n'] for i,p in props.items()}
        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        used_endpoint = set()
        merges = 0
        for score,a,b,reason,evidence in sorted(candidates,reverse=True):
            i,j = root(a),root(b)
            if i==j or volumes[i]+volumes[j] > 2.5*vref:
                continue
            if reason == 'curved_endpoint_join':
                rivals = [s for s,x,y,r,_ in candidates if r==reason and {x,y}!={a,b} and {x,y}&{a,b}]
                if a in used_endpoint or b in used_endpoint or (rivals and score-max(rivals)<.04):
                    continue
                used_endpoint.update((a,b))
            members = [int(k) for k,v in mapping.items() if root(v) in (i,j)]
            parent[j] = i
            volumes[i] += volumes[j]
            merges += 1
            operations.append(dict(action='merge',reason=reason,members=members,
                                   aggressive_pass=iteration+1,**evidence))
        if not merges:
            break
        lut = np.arange(int(current.max())+1,dtype='uint32')
        for i in props:
            lut[i] = root(i)
        current = lut[current]
        mapping = {k:root(v) for k,v in mapping.items()}
    # Quarantine unresolved small COMPONENTS, not just small label IDs. Bright
    # fluorescence alone is not evidence of a complete chromosome. Keep elongated
    # small rods and moderately small objects matched in both adjacent frames.
    deleted = 0
    budget = int(.06*(source>0).sum())
    candidates = []
    for lab,sl in enumerate(ndi.find_objects(current),1):
        if sl is None:
            continue
        cc,n = ndi.label(current[sl]==lab,np.ones((3,3,3)))
        members = [int(k) for k,v in mapping.items() if v==lab]
        votes = sum(any(k in mp for k in members) for mp in temporal_maps)
        for component,cs in enumerate(ndi.find_objects(cc),1):
            mask = cc[cs]==component
            count = int(mask.sum())
            if count > .12*vref:
                continue
            coords = np.argwhere(mask)*spacing
            span = float(np.linalg.norm(np.ptp(coords,axis=0)))
            cov = np.cov(coords.T) if len(coords)>3 else np.zeros((3,3))
            eig = np.linalg.eigvalsh(cov+np.diag(spacing**2/12))
            elongation = float(np.sqrt(eig[-1]/eig[-2]))
            if not strict_fragments and ((count > .04*vref and span > 2.5*width and elongation > 2.) or (count>.06*vref and votes>=2)):
                continue
            target = tuple(slice(s.start+c.start,s.start+c.stop) for s,c in zip(sl,cs))
            candidates.append((count,lab,target,mask,members,span,votes))
    for count,lab,target,mask,members,span,votes in sorted(candidates,key=lambda x:x[0]):
        if deleted+count > budget:
            continue
        current[target][mask] = 0
        deleted += count
        operations.append(dict(action='quarantine_aggressive_fragment',members=members,voxels=count,
                               span_um=span,temporal_votes=votes,fraction_of_reference=count/vref))
    remaining = set(np.unique(current))
    mapping = {k:v if v in remaining else 0 for k,v in mapping.items()}
    return current,operations,mapping,dict(merged=sum(o['action']=='merge' for o in operations),
                                          deleted_voxels=deleted,volume_reference_voxels=vref,
                                          width_um=width,delete_budget_voxels=budget)

# ---- tubeness_postprocess ----
"""CHIMP-inspired, seed-owned 3D completion with strict fragment quarantine.

Unlike CHIMP refinement, unowned watershed pieces NEVER become new IDs and
tubeness is NOT used to cut away existing foreground. Physical Hessian scales
are applied to raw DAPI; no learned image synthesis or convex-hull filling.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def physical_tubeness(raw, foreground, spacing):
    """Scale-normalized 3D Hessian ridge response on a bounded working grid."""
    spacing = np.asarray(spacing,float)
    stride = np.array([1, 2 if spacing[1]<.11 else 1, 2 if spacing[2]<.11 else 1])
    ds = tuple(slice(None,None,int(k)) for k in stride)
    image = ndi.gaussian_filter(raw.astype('float32'),(.3,.6,.6))[ds]
    bg = float(np.percentile(image,40))
    fg_values = raw[foreground]
    upper = float(np.percentile(fg_values,99)) if len(fg_values) else bg+1
    image = np.clip((image-bg)/max(upper-bg,1.),0,2)
    step = spacing*stride
    scales = sorted(set(max(s,.75*max(step)) for s in (.16,.24,.34)))
    response = np.zeros(image.shape,'float32')
    for sigma in scales:
        derivatives = {}
        for i in range(3):
            for j in range(i,3):
                order = np.zeros(3,int);order[i]+=1;order[j]+=1
                derivatives[i,j] = ndi.gaussian_filter(image,sigma/step,order=order,
                                                       mode='reflect')*(sigma**2/(step[i]*step[j]))
        # Slabs bound eigensolver memory, including large ZYX source movies.
        for z in range(0,len(image),12):
            sl=slice(z,min(z+12,len(image)))
            matrix=np.empty(image[sl].shape+(3,3),'float32')
            for (i,j),array in derivatives.items():
                matrix[...,i,j]=array[sl];matrix[...,j,i]=array[sl]
            eig=np.linalg.eigvalsh(matrix)
            eig=np.take_along_axis(eig,np.argsort(abs(eig),axis=-1),axis=-1)
            a,b,c=np.moveaxis(eig,-1,0)
            radial=np.sqrt(np.maximum(-b,0)*np.maximum(-c,0))
            ridge=radial*np.exp(-.5*(a/(radial+1e-6))**2)
            response[sl]=np.maximum(response[sl],ridge)
    scale=float(np.percentile(response[foreground[ds]],99)) if foreground[ds].any() else 1.
    response=np.clip(response/max(scale,1e-6),0,1)
    restored=ndi.zoom(response,np.array(raw.shape)/np.array(response.shape),order=1,prefilter=False)
    assert restored.shape==raw.shape
    return restored.astype('float32'),dict(scales_um=scales,working_spacing_um=step.tolist(),
                                          response_normalizer=scale)


def split_underseg_topology(labels, raw, spacing):
    """Conservative second direction of repair: split suspicious blobs.

    A normal I/Y/X-like chromosome has approximately 2/3/4 terminal tips.
    Only unusually large labels or labels with more than four reliable
    skeleton terminals are considered.  Distance-transform peaks must yield
    2--4 substantial, signal-supported pieces; otherwise the label is kept
    intact.  This deliberately avoids using a target count as a splitter.
    """
    out = labels.astype("uint32", copy=True)
    ids, sizes = np.unique(out[out > 0], return_counts=True)
    if not len(ids):
        return out, [], { }, dict(split=0, rejected=0)
    vref = float(np.percentile(sizes, 85))
    frame_count = int(len(ids))
    score, _, _ = signal_score(raw)
    smooth = ndi.gaussian_filter(score, np.maximum(.06 / np.asarray(spacing), .3))
    next_id = int(out.max())
    mapping = {str(int(i)): int(i) for i in ids}
    ops = []
    rejected = 0
    objects = ndi.find_objects(out)
    for lab0, size0 in zip(ids.astype(int), sizes.astype(int)):
        lab, size = int(lab0), int(size0)
        sl = objects[lab - 1]
        if sl is None:
            continue
        local = out[sl] == lab
        if int(size) < max(120, .90 * vref):
            continue
        depth = ndi.distance_transform_edt(local, sampling=spacing)
        width = max(2.0 * float(np.percentile(depth[local], 80)), 2.0 * min(spacing))
        ends, _ = endpoint_geometry(local, np.asarray(spacing, float), np.zeros(3), width)
        # Count is only a soft pressure signal: it never specifies an exact
        # target, but prevents aggressive splitting in clearly unfragmented
        # frames.  A very extreme topology/volume anomaly can still bypass it.
        if frame_count <= 105 and not (len(ends) >= 7 and size > 3.2 * vref):
            continue
        peak_size = tuple(int(2 * np.ceil(max(.16, .55 * width) / s) + 1) for s in spacing)
        maxima = (depth == ndi.maximum_filter(depth, size=peak_size))
        maxima &= local & (depth >= max(.12, .42 * float(depth[local].max())))
        markers, n = ndi.label(maxima, structure=CONN)
        if n > 4:
            # Crowded underseg blobs often produce several small plateau
            # peaks.  Keep the strongest four rather than discarding the
            # whole split proposal; weak peaks are usually ridge noise.
            strengths = [(float(depth[markers == k].max()), k) for k in range(1, n + 1)]
            keep = {k for _, k in sorted(strengths, reverse=True)[:4]}
            markers = np.where(np.isin(markers, list(keep)), markers, 0)
            relabel = {old: new for new, old in enumerate(sorted(keep), 1)}
            lut = np.zeros(int(markers.max()) + 1, dtype=np.int32)
            for old, new in relabel.items():
                lut[old] = new
            markers = lut[markers]
            n = len(keep)
        if n < 2:
            rejected += 1
            continue
        # Weighted evidence rather than an all-or-nothing gate.  A valid
        # I/Y/X chromosome may have 2--4 terminals, so topology is supportive
        # evidence, not a mandatory prerequisite for every split.
        terminal_score = 0 if len(ends) <= 4 else (2 if len(ends) <= 6 else 3)
        volume_ratio = float(size / max(vref, 1.0))
        volume_score = 0 if volume_ratio < 2.2 else (1 if volume_ratio < 2.65 else (2 if volume_ratio < 3.2 else 3))
        coords = np.argwhere(markers > 0)
        peak_centers = []
        for marker in range(1, n + 1):
            pts = np.argwhere(markers == marker)
            if len(pts):
                peak_centers.append(pts.mean(0) * np.asarray(spacing, float))
        separated = False
        if len(peak_centers) >= 2:
            distances = [np.linalg.norm(a - b) for i, a in enumerate(peak_centers) for b in peak_centers[i + 1:]]
            separated = min(distances) >= .8 * width
        separation_score = 1 if separated else 0
        pieces = watershed(-depth, markers, mask=local, connectivity=CONN)
        pids, pcounts = np.unique(pieces[pieces > 0], return_counts=True)
        min_piece = max(18, int(.10 * vref))
        if len(pids) < 2 or len(pids) > 4 or np.any(pcounts < min_piece):
            rejected += 1
            continue
        med_signal = np.array([
            float(np.median(smooth[sl][pieces == p])) for p in pids
        ])
        if np.any(med_signal < 2.0):
            rejected += 1
            continue
        signal_score_part = 1 if float(np.min(med_signal)) >= 3.0 else 0
        evidence_score = int(2 + terminal_score + volume_score + separation_score + signal_score_part)
        # At least one primary underseg signal is required.  This prevents a
        # normal early-stage chromosome with several thickness peaks from
        # being split merely because its texture is heterogeneous.
        if evidence_score < 6 or (terminal_score == 0 and volume_score < 2):
            rejected += 1
            continue
        # Keep the old id on the largest child; assign fresh ids to the rest.
        order = pids[np.argsort(pcounts)[::-1]]
        local_new = np.zeros_like(pieces, dtype="uint32")
        local_new[pieces == order[0]] = lab
        children = [int(lab)]
        for p in order[1:]:
            next_id += 1
            local_new[pieces == p] = next_id
            children.append(int(next_id))
        if np.any(local_new[local] == 0):
            rejected += 1
            continue
        out[sl][local] = local_new[local]
        ops.append(dict(action="split", reason="topology_underseg", members=[lab],
                        children=children, terminal_count=int(len(ends)),
                        parent_voxels=int(size), child_voxels=[int(x) for x in pcounts[np.argsort(pcounts)[::-1]]],
                        evidence_score=evidence_score, volume_ratio=volume_ratio,
                        separated_peaks=bool(separated)))
        mapping[str(lab)] = lab
    return out, ops, mapping, dict(split=len(ops), rejected=rejected)


def hybrid_repair(source, raw, spacing, diagnostics=False):
    if source.shape != raw.shape or source.ndim != 3:
        raise ValueError('Expected paired ZYX DAPI and labels')
    if not np.issubdtype(source.dtype,np.integer) or source.min()<0:
        raise ValueError('Nonnegative integer labels required')
    spacing=np.asarray(spacing,float)
    ids,counts=np.unique(source[source>0],return_counts=True)
    if not len(ids):
        return source.astype('uint32'),dict(before=0,after=0,merged=0,added_voxels=0,
             deleted_voxels=0,foreground_before=0,foreground_after=0,merge_reasons={}),[],{},[],{}
    locations=ndi.find_objects((source>0).astype('uint8'))[0]
    margin=np.ceil(.8/spacing).astype(int)+2
    crop=tuple(slice(max(0,s.start-int(margin[k])),min(source.shape[k],s.stop+int(margin[k])))
               for k,s in enumerate(locations))
    original=source[crop];image=raw[crop]
    stage,ops,mapping,aggressive=aggressive_repair(original,image,spacing,strict_fragments=True)
    quarantined=(original>0)&(stage==0)
    if not stage.any():
        raise RuntimeError('Quarantine removed all foreground; refuse output')
    stage,cut_ops,cut_map,cut_stats=refine_surface(stage,image,spacing,relabel_only=True)
    ops.extend(_translate_operations(cut_ops,{int(k):v for k,v in mapping.items()},0))
    mapping={k:cut_map.get(str(v),0) if v else 0 for k,v in mapping.items()}
    aggressive['merged']+=cut_stats['merged']
    width=aggressive['width_um']
    tube,tube_info=physical_tubeness(image,stage>0,spacing)
    score,bg,noise=signal_score(image)
    smooth=ndi.gaussian_filter(score,np.maximum(.07/spacing,.3))
    positive=smooth[stage>0]
    reference=max(3.,float(np.median(positive)))
    strength=np.clip(smooth/max(float(np.percentile(positive,95)),3.),0,1)
    distance,indices=ndi.distance_transform_edt(stage==0,sampling=spacing,return_indices=True)
    nearest=stage[tuple(indices)];del indices
    reach=min(.50,max(.16,.8*width))
    # Tubeness is a soft structural proposal, not a hard foreground template.
    # A small physical dilation covers the outer shoulder of a thick ridge.
    radius=max(.10,.25*width)
    dilsize=tuple(int(2*np.ceil(radius/s)+1) for s in spacing)
    tube_near=ndi.maximum_filter(tube,size=dilsize)
    support=(stage>0)|((distance<=reach)&(smooth>=max(3.,.22*reference))&
                     (score>=2)&(tube_near>=.045)&~quarantined)
    feature=(.7*strength+.3*tube).astype('float32')
    peaksize=tuple(int(2*np.ceil(max(.12,.45*width)/s)+1) for s in spacing)
    maxima=(feature==ndi.maximum_filter(feature,size=peaksize))&support&(feature>.05)
    markers,n=ndi.label(maxima)
    # Every retained label has a seed, including weak broad chromosome tips.
    for lab,sl in enumerate(ndi.find_objects(stage),1):
        if sl is None:continue
        inside=stage[sl]==lab
        local=feature[sl]
        pos=np.unravel_index(np.argmax(np.where(inside,local,-1)),local.shape)
        if markers[sl][pos]==0:
            n+=1;markers[sl][pos]=n
    regions=watershed(-feature,markers,mask=support).astype('uint32')
    seed_mask=(stage>0)&(regions>0)
    base=int(stage.max())+1
    pairs,overlap=np.unique(regions[seed_mask].astype('int64')*base+stage[seed_mask],return_counts=True)
    total=np.bincount(regions[seed_mask],minlength=int(regions.max())+1)
    owners=np.zeros(len(total),'uint32')
    confidence=np.zeros(len(total),'float32')
    for ix in np.argsort(overlap)[::-1]:
        region,lab=divmod(int(pairs[ix]),base)
        if owners[region]:continue
        fraction=float(overlap[ix]/total[region])
        if overlap[ix]>=3 and fraction>=.85:
            owners[region]=lab;confidence[region]=fraction
    proposed=owners[regions]
    neighbor_max=ndi.maximum_filter(stage,size=3)
    neighbor_min=ndi.minimum_filter(np.where(stage>0,stage,base),size=3)
    competition=(neighbor_max>0)&((neighbor_max!=proposed)|(neighbor_min!=proposed))
    candidates=(stage==0)&(proposed>0)&(nearest==proposed)&~competition&~quarantined
    output=stage.copy()
    stage_sizes=np.bincount(stage.ravel())
    stage_objects=ndi.find_objects(stage)
    # Per-instance and per-frame bounds prevent a bright halo swallowing gaps.
    global_budget=int(.12*(original>0).sum())
    added=0
    for lab,sl in enumerate(ndi.find_objects(np.where(candidates,proposed,0)),1):
        if sl is None:continue
        selected=candidates[sl]&(proposed[sl]==lab)
        positions=np.argwhere(selected)
        own_slice=stage_objects[lab-1]
        own_signal=float(np.median(smooth[own_slice][stage[own_slice]==lab]))
        positions=positions[smooth[sl][tuple(positions.T)]>=max(3.,.25*own_signal)]
        budget=min(int(.12*stage_sizes[lab]),global_budget-added)
        if budget<=0:continue
        priority=feature[sl][tuple(positions.T)]/(1+distance[sl][tuple(positions.T)]/min(spacing))
        if len(positions)>budget:positions=positions[np.argsort(priority)[-budget:]]
        union=tuple(slice(min(a.start,b.start),max(a.stop,b.stop)) for a,b in zip(sl,own_slice))
        seeds=stage[union]==lab
        allowed=seeds.copy()
        shifted=positions+np.array([a.start-b.start for a,b in zip(sl,union)])
        allowed[tuple(shifted.T)]=True
        connected=ndi.binary_propagation(seeds,mask=allowed,structure=np.ones((3,3,3)))
        positions=positions[connected[tuple(shifted.T)]]
        output[sl][tuple(positions.T)]=lab
        added+=len(positions)
        if len(positions):
            ops.append(dict(action='tubeness_watershed_completion',members=[int(k) for k,v in mapping.items() if v==lab],
                            voxels=len(positions),minimum_overlap_dominance=.85))
    # Reconsider cross-ID connectivity AFTER shoulder completion; otherwise
    # the competition guard above permanently preserves true fracture seams.
    output,gap_ops,gap_map,gap_stats=repair_gaps(
        output,image,spacing,tube,forbidden=quarantined,budget=global_budget-added)
    ops.extend(_translate_operations(gap_ops,{int(k):v for k,v in mapping.items()},0))
    mapping={k:gap_map.get(str(v),0) if v else 0 for k,v in mapping.items()}
    added+=gap_stats['added_voxels']
    aggressive['merged']+=gap_stats['merged']
    # Repair the opposite failure mode after all bridge proposals: a large
    # multi-terminal blob can be an undersegmented group, not a chromosome
    # that should receive further bridge filling.
    output,split_ops,split_map,split_stats=split_underseg_topology(output,image,spacing)
    ops.extend(_translate_operations(split_ops,{int(k):v for k,v in mapping.items()},0))
    mapping={k:split_map.get(str(v),0) if v else 0 for k,v in mapping.items()}
    full=np.zeros(source.shape,'uint32');full[crop]=output
    removed=int(((source>0)&(full==0)).sum())
    actual_added=int(((source==0)&(full>0)).sum())
    assert actual_added==added
    assert not np.any(output[quarantined])
    reasons={}
    for op in ops:
        if op['action']=='merge':reasons[op['reason']]=reasons.get(op['reason'],0)+1
    stats=dict(before=len(ids),after=int(len(np.unique(full[full>0]))),merged=aggressive['merged'],
               added_voxels=added,deleted_voxels=removed,foreground_before=int((source>0).sum()),
               foreground_after=int((full>0).sum()),merge_reasons=reasons,
               hybrid=dict(**tube_info,background=bg,noise=noise,width_um=width,
                           watershed_regions=int(regions.max()),owned_regions=int((owners>0).sum()),
                           unowned_regions=int(((total==0)&(np.arange(len(total))>0)).sum()),
                           crop_zyx=[[s.start,s.stop] for s in crop],strict_fragment_quarantine=True,
                           added_budget_voxels=global_budget,aggressive=aggressive,
                           gap_bend=gap_stats,topology_split=split_stats))
    assert stats['foreground_after']==stats['foreground_before']+added-removed
    extra=dict(tubeness=tube,overseg=regions,owners=owners,stage=stage,crop=crop) if diagnostics else {}
    return full,stats,ops,mapping,[],extra

# ---- gap_bend_refinement ----
"""Physical, raw-supported endpoint bridges, including sharply bent joins.

Runs AFTER completion so that protected seams get a second chance. It never
removes foreground, crosses another label, or uses a target instance count.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric


def repair_gaps(labels, raw, spacing, tube, forbidden=None, budget=None):
    spacing=np.asarray(spacing,float)
    out=labels.astype('uint32',copy=True)
    ids,sizes=np.unique(out[out>0],return_counts=True)
    mapping={str(int(i)):int(i) for i in ids}
    if not len(ids):return out,[],mapping,dict(merged=0,added_voxels=0,rejected=[])
    forbidden=np.zeros(out.shape,bool) if forbidden is None else forbidden
    budget=int(.02*(out>0).sum()) if budget is None else int(budget)
    score,_,_=signal_score(raw)
    smooth=ndi.gaussian_filter(score,np.maximum(.06/spacing,.3))
    depth=ndi.distance_transform_edt(out>0,sampling=spacing)
    width=max(2*float(np.percentile(depth[out>0],80)),2*min(spacing))
    # Max physical *background* gap; include Z sampling without making XY
    # search unbounded. Endpoint geometry is still evaluated in microns.
    maxgap=min(.50,max(1.1*max(spacing),.55*width))
    edges,_,_=collect_edges(out,smooth,spacing,maxgap,depth)
    objects=ndi.find_objects(out);ends={};signals={}
    vref=float(np.percentile(sizes,85));counts=dict(zip(ids,sizes))
    ops=[];rejected=[];candidates=[]
    def geometry(lab):
        if lab not in ends:
            sl=objects[lab-1];m=out[sl]==lab
            # Voxel-space skeletonization retracts endpoints excessively along
            # coarse Z. Skeletonize on an isotropic physical working grid.
            step=max(float(min(spacing)),width/8)
            iso=ndi.zoom(m,spacing/step,order=0,prefilter=False)
            iso_ends=endpoint_geometry(iso,np.full(3,step),np.zeros(3),width)[0]
            coords=np.argwhere(m);physical=coords*spacing
            tree=cKDTree(physical);origin=np.array([s.start for s in sl])*spacing
            ends[lab]=[(physical[tree.query(p)[1]]+origin,t) for p,t in iso_ends]
            signals[lab]=float(np.median(smooth[sl][m]))
        return ends[lab]
    for e in edges:
        a,b=e['i'],e['j'];reason=None
        if counts[a]+counts[b]>3*vref:reason='combined_volume_guard'
        ea,eb=geometry(a),geometry(b)
        pairs=[]
        if reason is None:
            for ia,(pa,ta) in enumerate(ea):
                for ib,(pb,tb) in enumerate(eb):
                    delta=pb-pa;distance=float(np.linalg.norm(delta))
                    if distance<=1e-8 or distance>maxgap+2.*width:continue
                    direction=delta/distance
                    # Both ends must face the junction. Unlike the old straight
                    # join, the inward tangents may meet at an acute bend.
                    facing_a=float(-ta@direction);facing_b=float(tb@direction)
                    turn=float(ta@tb)
                    if min(facing_a,facing_b)<.12 or turn>.85:continue
                    pairs.append((min(facing_a,facing_b)-.15*distance/width,ia,ib,pa,pb,turn))
            if not pairs:reason='no_facing_terminal_pair'
        if reason:
            rejected.append(dict(i=a,j=b,reason=reason));continue
        confidence,ia,ib,pa,pb,turn=max(pairs,key=lambda p:p[0])
        p=np.rint(pa/spacing).astype(int);q=np.rint(pb/spacing).astype(int)
        # Endpoints are skeleton voxels. Their local corridor includes a bend,
        # rather than assuming the missing segment is a straight voxel line.
        pad=np.ceil(.7*width/spacing).astype(int)+1
        lo=np.maximum(0,np.minimum(p,q)-pad);hi=np.minimum(out.shape,np.maximum(p,q)+pad+1)
        sl=tuple(slice(int(l),int(h)) for l,h in zip(lo,hi))
        sub=out[sl];own=(sub==a)|(sub==b)
        signal_floor=max(3.,.22*min(signals[a],signals[b]))
        allowed=own|((sub==0)&~forbidden[sl]&(score[sl]>=signal_floor)&
                     ((tube[sl]>=.02)|(smooth[sl]>=.55*min(signals[a],signals[b]))))
        other=(sub>0)&~own
        allowed &= own|~ndi.binary_dilation(other,structure=np.ones((3,3,3)))
        costs=np.where(allowed,1.+2./(1.+np.maximum(smooth[sl],0))+1.-tube[sl],np.inf)
        solver=MCP_Geometric(costs,fully_connected=True,sampling=tuple(spacing))
        start=tuple(p-lo);finish=tuple(q-lo)
        distance_map,_=solver.find_costs([start],[finish])
        if not np.isfinite(distance_map[finish]):
            rejected.append(dict(i=a,j=b,reason='no_raw_supported_path'));continue
        path=np.asarray(solver.traceback(finish),int)
        if np.percentile(score[sl][tuple(path.T)],20)<signal_floor:
            rejected.append(dict(i=a,j=b,reason='weak_seam_signal'));continue
        length=float(np.linalg.norm(np.diff(path,axis=0)*spacing,axis=1).sum())
        if length>1.65*np.linalg.norm(pa-pb)+min(spacing):
            rejected.append(dict(i=a,j=b,reason='path_detour'));continue
        # Bridge must be short; long travels through background are not repairs.
        is_bg=sub[tuple(path.T)]==0
        gaplength=float(sum(np.linalg.norm((path[k]-path[k-1])*spacing)
                            for k in range(1,len(path)) if is_bg[k] or is_bg[k-1]))
        if gaplength>maxgap+2*max(spacing):
            rejected.append(dict(i=a,j=b,reason='gap_too_wide'));continue
        proposed=np.zeros(sub.shape,bool);proposed[tuple(path.T)]=True
        # Modest physical shoulder around the path; never close entire objects.
        dist=ndi.distance_transform_edt(~proposed,sampling=spacing)
        fill=(dist<=max(min(spacing),.16*width))&allowed&(sub==0)
        bgcoords=np.argwhere(fill)+lo
        candidates.append(dict(i=a,j=b,ea=ia,eb=ib,score=confidence,fill=bgcoords,
                               path=path+lo,path_length_um=length,gap_um=gaplength,
                               inward_cosine=turn))
    parent={int(i):int(i) for i in ids};vol=dict(counts);used=set();added=0
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for c in sorted(candidates,key=lambda c:c['score'],reverse=True):
        a,b=c['i'],c['j'];tokens={(a,c['ea']),(b,c['eb'])}
        rivals=[r['score'] for r in candidates if r is not c and
                tokens&{(r['i'],r['ea']),(r['j'],r['eb'])}]
        reason=None
        if tokens&used or (rivals and c['score']-max(rivals)<.06):reason='ambiguous_endpoint'
        ra,rb=root(a),root(b)
        if ra==rb:continue
        if vol[ra]+vol[rb]>3*vref:reason='combined_volume_guard'
        coords=c['fill'];coords=coords[out[tuple(coords.T)]==0]
        path_labels=np.unique(out[tuple(c['path'].T)])
        if any(v and root(int(v)) not in (ra,rb) for v in path_labels):reason='competing_bridge'
        if added+len(coords)>budget:reason='addition_budget'
        if reason:
            rejected.append(dict(i=a,j=b,reason=reason));continue
        members=[i for i in parent if root(i) in (ra,rb)]
        parent[rb]=ra;vol[ra]+=vol[rb];used|=tokens
        out[tuple(coords.T)]=ra;added+=len(coords)
        ops.append(dict(action='merge',reason='raw_supported_gap_or_sharp_bend',members=members,
                        added_voxels=len(coords),path_zyx=c['path'].tolist(),
                        path_length_um=c['path_length_um'],gap_um=c['gap_um'],
                        inward_cosine=c['inward_cosine']))
    lut=np.arange(int(out.max())+1,dtype='uint32')
    for i in parent:lut[i]=root(i)
    out=lut[out];mapping={str(i):root(i) for i in parent}
    return out,ops,mapping,dict(merged=len(ops),added_voxels=added,rejected=rejected,
                                max_gap_um=maxgap,width_um=width)

def readability_repair(source, raw, spacing, cfg):
    """Fill -> absorb minor fragments -> quarantine, preserving substantial IDs.

    All proposals use the untouched input mask as their ownership reference.
    Watershed determines signal-supported fill regions, never instance identity.
    There is no endpoint-join, transitive union-find, or topology-split step.
    """
    if source.shape != raw.shape or source.ndim != 3:
        raise ValueError('Expected matching ZYX labels and DAPI')
    if not np.issubdtype(source.dtype, np.integer) or source.min() < 0:
        raise ValueError('Expected nonnegative integer source labels')
    spacing = np.asarray(spacing, float)
    original = source.astype('uint32', copy=False)
    ids, sizes = np.unique(original[original > 0], return_counts=True)
    foreground = int(sizes.sum())
    initial_sizes = {int(i): int(n) for i, n in zip(ids, sizes)}
    mapping = {str(int(i)): int(i) for i in ids}
    if not foreground:
        return original.copy(), dict(before=0, after=0, merged=0,
            added_voxels=0, deleted_voxels=0, foreground_before=0,
            foreground_after=0, merge_reasons={}), [], mapping, []

    vref = max(8., float(np.percentile(sizes, 85)))
    small_limit = cfg.readable_fragment_vref_fraction * vref
    protected_ids = ids[sizes > small_limit]
    extent = ndi.find_objects((original > 0).astype('uint8'))[0]
    pad = np.ceil(max(.8, 2 * cfg.readable_fill_radius_um) / spacing).astype(int) + 2
    crop = tuple(slice(max(0, s.start - int(pad[k])),
                       min(original.shape[k], s.stop + int(pad[k])))
                 for k, s in enumerate(extent))
    seed = original[crop]
    image = raw[crop]
    fg = seed > 0
    score, bg, noise = signal_score(image)
    smooth = ndi.gaussian_filter(score, np.maximum(.06 / spacing, .3))
    depth = ndi.distance_transform_edt(fg, sampling=spacing)
    width = max(2 * float(np.percentile(depth[fg], 80)), 2 * min(spacing))
    distance, indices = ndi.distance_transform_edt(~fg, sampling=spacing, return_indices=True)
    nearest = seed[tuple(indices)]
    del indices, depth
    reach = min(cfg.readable_fill_radius_um, max(1.05 * min(spacing), .4 * width))
    nlabels = int(ids.max()) + 1
    seed_objects = ndi.find_objects(seed)
    signal_by_id = np.zeros(nlabels, dtype='float32')
    for lab, sl in enumerate(seed_objects, 1):
        if sl is not None:
            signal_by_id[lab] = max(2., float(np.median(smooth[sl][seed[sl] == lab])))
    signal_floor = np.maximum(2.5, .25 * signal_by_id[nearest])
    valid = (~fg) & (distance <= reach) & (score >= 1.5) & (smooth >= signal_floor)

    # True bounded 3D cavities only. Open loops/valleys are not blanket-filled.
    holes = np.zeros(seed.shape, bool)
    for lab, sl0 in enumerate(seed_objects, 1):
        if sl0 is None:
            continue
        sl = tuple(slice(max(0, s.start - 1), min(seed.shape[k], s.stop + 1))
                   for k, s in enumerate(sl0))
        own = seed[sl] == lab
        cavity = ndi.binary_fill_holes(own) & ~own & (seed[sl] == 0)
        cc, _ = ndi.label(cavity, structure=CONN)
        counts = np.bincount(cc.ravel())
        allowed = np.flatnonzero((counts <= max(2, .03 * initial_sizes[lab])) & (counts > 0))
        allowed = allowed[allowed > 0]
        holes[sl] |= np.isin(cc, allowed)

    # Local closing is a geometric proposal on the foreground, not a label
    # merge. The footprint is small in physical units and cannot exceed reach.
    radius = min(reach, max(min(spacing), .20 * width))
    half = np.ceil(radius / spacing).astype(int)
    grid = np.ogrid[tuple(slice(-int(h), int(h) + 1) for h in half)]
    footprint = sum((g * s) ** 2 for g, s in zip(grid, spacing)) <= radius ** 2 * 1.001
    closed = ndi.binary_closing(fg, structure=footprint)
    seams = closed & ~fg

    tube, tube_info = physical_tubeness(image, fg, spacing)
    tube_near = ndi.maximum_filter(tube, size=tuple(2 * half + 1))
    proposed = valid & (holes | seams | (tube_near >= .06))
    support = fg | proposed
    normalizer = max(3., float(np.percentile(smooth[fg], 95)))
    feature = (.7 * np.clip(smooth / normalizer, 0, 1) + .3 * tube).astype('float32')
    peaksize = tuple(int(2 * np.ceil(max(.12, .45 * width) / s) + 1) for s in spacing)
    maxima = support & (feature >= .05) & (feature == ndi.maximum_filter(feature, size=peaksize))
    markers, marker_count = ndi.label(maxima, structure=CONN)
    for lab, sl in enumerate(seed_objects, 1):
        if sl is None:
            continue
        own = seed[sl] == lab
        pos = np.unravel_index(np.argmax(np.where(own, feature[sl], -1)), own.shape)
        if not markers[sl][pos]:
            marker_count += 1
            markers[sl][pos] = marker_count
    regions = watershed(-feature, markers, mask=support, connectivity=CONN)
    anchored_regions = np.bincount(regions[fg], minlength=int(regions.max()) + 1) >= 3
    anchored_regions[0] = False
    candidates = proposed & (holes | seams | anchored_regions[regions])
    # Assign all accepted background to its nearest ORIGINAL label, even where
    # watershed might favour a brighter or larger neighboring instance.
    output = seed.copy()
    global_budget = int(cfg.readable_added_fraction * foreground)
    added = 0
    operations = []
    rejected = []
    proposal_labels = np.where(candidates, nearest, 0)
    for lab, sl0 in enumerate(ndi.find_objects(proposal_labels), 1):
        if sl0 is None:
            continue
        positions = np.argwhere(proposal_labels[sl0] == lab)
        original_sl = seed_objects[lab - 1]
        sl = tuple(slice(min(a.start, b.start), max(a.stop, b.stop))
                   for a, b in zip(sl0, original_sl))
        absolute = positions + np.array([s.start for s in sl0])
        local_positions = absolute - np.array([s.start for s in sl])
        own = seed[sl] == lab
        allowed = own.copy()
        allowed[tuple(local_positions.T)] = True
        connected = ndi.binary_propagation(own, mask=allowed, structure=CONN)
        connected_mask = connected[tuple(local_positions.T)]
        absolute = absolute[connected_mask]
        if not len(absolute):
            continue
        coordinates = tuple(absolute.T)
        priority = (4 * holes[coordinates] + 2 * seams[coordinates] + feature[coordinates]
                    - distance[coordinates] / max(reach, 1e-8))
        # Allocate proportionally before iterating, so a low numeric ID cannot
        # consume the frame budget and starve later IDs of completion.
        share = min(cfg.readable_instance_added_fraction, cfg.readable_added_fraction)
        limit = min(int(share * initial_sizes[lab]), global_budget - added)
        if limit <= 0:
            rejected.append(dict(label=lab, reason='fill_budget'))
            continue
        if len(absolute) > limit:
            absolute = absolute[np.argsort(priority)[-limit:]]
        # Check connectivity again after truncating to the volume budget.
        local_positions = absolute - np.array([s.start for s in sl])
        allowed = own.copy()
        allowed[tuple(local_positions.T)] = True
        connected = ndi.binary_propagation(own, mask=allowed, structure=CONN)
        absolute = absolute[connected[tuple(local_positions.T)]]
        coordinates = tuple(absolute.T)
        output[coordinates] = lab
        added += len(absolute)
        operations.append(dict(action='nearest_id_completion', members=[lab], voxels=len(absolute),
            holes=int(holes[coordinates].sum()),
            seams=int((seams[coordinates] & ~holes[coordinates]).sum()),
            tubeness=int((~seams[coordinates] & ~holes[coordinates]).sum())))

    # Frozen minor-fragment list and frozen substantial-parent list prevent
    # repeated absorption or growth from turning two major labels into one ID.
    minor_ids = ids[sizes <= small_limit]
    parent_ids = set(int(i) for i, n in initial_sizes.items() if n >= .20 * vref and n > small_limit)
    absorption_candidates = []
    cross = ndi.generate_binary_structure(3, 1)
    output_objects = ndi.find_objects(output)
    for lab0 in minor_ids:
        lab = int(lab0)
        sl0 = output_objects[lab - 1]
        if sl0 is None:
            continue
        sl = tuple(slice(max(0, s.start - 1), min(output.shape[k], s.stop + 1))
                   for k, s in enumerate(sl0))
        sub = output[sl]
        own = sub == lab
        surface = own & ~ndi.binary_erosion(own, structure=cross, border_value=0)
        surface_count = max(1, int(surface.sum()))
        contact_labels = np.unique(sub[ndi.binary_dilation(own, structure=CONN) & ~own])
        contacts = []
        for dest0 in contact_labels:
            dest = int(dest0)
            if dest not in parent_ids:
                continue
            if initial_sizes[lab] > cfg.readable_fragment_parent_fraction * initial_sizes[dest]:
                continue
            touching = surface & ndi.binary_dilation(sub == dest, structure=CONN)
            fraction = float(touching.sum() / surface_count)
            if touching.sum() < 3 or fraction < .25:
                continue
            if float(np.median(smooth[sl][touching])) < 2.5:
                continue
            contacts.append((fraction, dest, int(touching.sum())))
        contacts.sort(reverse=True)
        if contacts:
            fraction, dest, contact_voxels = contacts[0]
            runner_up = contacts[1][0] if len(contacts) > 1 else 0.
            if fraction >= 2 * runner_up and fraction - runner_up >= .15:
                absorption_candidates.append((lab, dest, fraction, contact_voxels))
            else:
                rejected.append(dict(label=lab, reason='ambiguous_fragment_parent'))
    for lab, dest, fraction, contact_voxels in absorption_candidates:
        output[output == lab] = dest
        mapping[str(lab)] = dest
        operations.append(dict(action='merge', reason='readable_minor_fragment_absorption',
            members=[lab, dest], fragment_label=lab, parent_label=dest,
            source_fragment_voxels=initial_sizes[lab], contact_fraction=fraction,
            contact_voxels=contact_voxels))

    delete_budget = int(cfg.readable_deleted_fraction * foreground)
    deleted = 0
    # A small coherent rod can be a complete small chromosome. Preserve such
    # objects; quarantine sub-resolution/unstructured leftovers within budget.
    deletion_candidates = []
    objects = ndi.find_objects(output)
    for lab0 in minor_ids:
        lab = int(lab0)
        if mapping[str(lab)] != lab:
            continue
        sl = objects[lab - 1]
        if sl is None:
            continue
        own = output[sl] == lab
        cc, components = ndi.label(own, structure=CONN)
        coords = np.argwhere(own).astype(float) * spacing
        covariance = np.cov(coords.T) if len(coords) > 3 else np.zeros((3, 3))
        eigenvalues = np.linalg.eigvalsh(covariance + np.diag(spacing ** 2 / 12))
        elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[-2]))
        span = float(np.linalg.norm(np.ptp(coords, axis=0)))
        coherent_rod = (components == 1 and initial_sizes[lab] >= .02 * vref
                        and elongation >= 2.2 and span >= 2 * width)
        if coherent_rod:
            rejected.append(dict(label=lab, reason='preserved_small_coherent_rod'))
            continue
        deletion_candidates.append((initial_sizes[lab], lab, sl, own))
    for count, lab, sl, own in sorted(deletion_candidates, key=lambda x: (x[0], x[1])):
        if deleted + count > delete_budget:
            rejected.append(dict(label=lab, reason='fragment_delete_budget'))
            continue
        removed_added = int((own & (seed[sl] == 0)).sum())
        output[sl][own] = 0
        deleted += count
        added -= removed_added
        mapping[str(lab)] = 0
        operations.append(dict(action='quarantine_readable_fragment', members=[lab],
            voxels=count, removed_completion_voxels=removed_added,
            fraction_of_reference=count / vref))

    full = np.zeros(original.shape, dtype='uint32')
    full[crop] = output
    lut = np.arange(nlabels, dtype='uint32')
    for old, new in mapping.items():
        lut[int(old)] = new
    # These assertions enforce the user's intent on every frame, including
    # labels that touch after filling. No substantial ID can be merged/deleted.
    assert np.array_equal(full[original > 0], lut[original[original > 0]])
    assert all(mapping[str(int(i))] == int(i) for i in protected_ids)
    background_added = (seed == 0) & (output > 0)
    assert np.array_equal(output[background_added], lut[nearest[background_added]])
    assert set(np.unique(full)) <= set(np.unique(original)) | {0}
    assert int(((original == 0) & (full > 0)).sum()) == added
    assert int(((original > 0) & (full == 0)).sum()) == deleted
    final_ids = np.unique(full[full > 0])
    stats = dict(before=len(ids), after=len(final_ids), merged=len(absorption_candidates),
        added_voxels=added, deleted_voxels=deleted, foreground_before=foreground,
        foreground_after=int((full > 0).sum()),
        merge_reasons={'readable_minor_fragment_absorption': len(absorption_candidates)},
        readable=dict(volume_reference_voxels=vref, fragment_limit_voxels=small_limit,
            width_um=width, fill_radius_um=reach, protected_instances=len(protected_ids),
            protected_ids_preserved=True, nearest_id_ownership_verified=True,
            large_instance_merges=0, topology_splits=0, temporal_used=False,
            deleted_instances=sum(v == 0 for v in mapping.values()),
            added_budget_voxels=global_budget, deleted_budget_voxels=delete_budget,
            tubeness=tube_info, rejected=rejected))
    assert stats['foreground_after'] == foreground + added - deleted
    return full, stats, operations, mapping, []


if __name__ == "__main__":
    main()
