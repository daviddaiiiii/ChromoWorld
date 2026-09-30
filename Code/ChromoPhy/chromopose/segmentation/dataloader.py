"""Dataloader for ChromoPose per-T volumes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

@dataclass
class IndexRecord:
    sid: str
    t: int
    raw: str
    mask: str
    target: str
    nz: int
    ny: int
    nx: int
    n_inst: int
    split: str = ""
    target_vis: str = ""
    celltype: str = ""
    sample_num: str = ""

def load_index(index_path: Path) -> List[IndexRecord]:
    records: List[IndexRecord] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(IndexRecord(**{k: d[k] for k in IndexRecord.__dataclass_fields__ if k in d}))
    return records

def filter_by_split(records: Sequence[IndexRecord], split: str) -> List[IndexRecord]:
    if not str(split).strip():
        return list(records)
    return [r for r in records if r.split == str(split).strip()]

def split_records(
    records: Sequence[IndexRecord],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[IndexRecord], List[IndexRecord]]:
    sids = sorted({r.sid for r in records})
    rng = random.Random(seed)
    rng.shuffle(sids)
    n_val = max(1, int(round(len(sids) * float(val_ratio)))) if len(sids) > 1 else 0
    val_set = set(sids[:n_val])
    train = [r for r in records if r.sid not in val_set]
    val = [r for r in records if r.sid in val_set]
    return train, val

def normalize_volume(raw: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(raw, pmin))
    hi = float(np.percentile(raw, pmax))
    if hi <= lo:
        hi = lo + 1e-6
    out = (raw.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

def augment_flip_rot90(
    raw: np.ndarray,
    target: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random flips / 90° rotations; sync flow component permutation."""
    raw = raw.copy()
    target = target.copy()
    # raw: (Z,Y,X), target: (C,Z,Y,X)
    if rng.random() < 0.5:
        raw = raw[:, :, ::-1].copy()
        target = target[:, :, :, ::-1].copy()
        target[CH_FLOW_X] *= -1
        target[CH_EMB_0] *= -1
    if rng.random() < 0.5:
        raw = raw[:, ::-1, :].copy()
        target = target[:, :, ::-1, :].copy()
        target[CH_FLOW_Y] *= -1
        if EMBEDDING_DIM >= 2:
            target[CH_EMB_0 + 1] *= -1
    if rng.random() < 0.5:
        raw = raw[::-1, :, :].copy()
        target = target[:, ::-1, :, :].copy()
        target[CH_FLOW_Z] *= -1
        if EMBEDDING_DIM >= 3:
            target[CH_EMB_0 + 2] *= -1
    k = rng.randint(0, 3)
    if k:
        raw = np.rot90(raw, k=k, axes=(1, 2)).copy()
        target = np.rot90(target, k=k, axes=(2, 3)).copy()
        for _ in range(k):
            fx, fy = target[CH_FLOW_X].copy(), target[CH_FLOW_Y].copy()
            target[CH_FLOW_X] = fy
            target[CH_FLOW_Y] = -fx
            if EMBEDDING_DIM >= 2:
                ex, ey = target[CH_EMB_0].copy(), target[CH_EMB_0 + 1].copy()
                target[CH_EMB_0] = ey
                target[CH_EMB_0 + 1] = -ex
    if rng.random() < 0.3:
        scale = 0.9 + 0.2 * rng.random()
        raw = np.clip(raw * scale, 0.0, 1.0)
    return raw, target

def _random_crop_slices(
    shape: Tuple[int, int, int],
    crop: Tuple[int, int, int],
    center: Optional[Tuple[int, int, int]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[slice, slice, slice]:
    nz, ny, nx = shape
    cz, cy, cx = crop
    rng = rng or random.Random()
    if center is None:
        z0 = rng.randint(0, max(nz - cz, 0))
        y0 = rng.randint(0, max(ny - cy, 0))
        x0 = rng.randint(0, max(nx - cx, 0))
    else:
        z0 = int(np.clip(center[0] - cz // 2, 0, max(nz - cz, 0)))
        y0 = int(np.clip(center[1] - cy // 2, 0, max(ny - cy, 0)))
        x0 = int(np.clip(center[2] - cx // 2, 0, max(nx - cx, 0)))
    return slice(z0, z0 + cz), slice(y0, y0 + cy), slice(x0, x0 + cx)

class ChromoPoseVolumeDataset(Dataset):
    def __init__(
        self,
        records: Sequence[IndexRecord],
        data_root: Path,
        *,
        crop_zyx: Tuple[int, int, int] = (64, 256, 256),
        crop_mode_probs: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        augment: bool = True,
        seed: int = 0,
    ):
        if tifffile is None:
            raise RuntimeError("tifffile is required")
        self.records = list(records)
        self.data_root = Path(data_root)
        self.crop_zyx = crop_zyx
        self.crop_mode_probs = crop_mode_probs
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, rec: IndexRecord) -> Tuple[np.ndarray, np.ndarray]:
        raw = tifffile.imread(str(self.data_root / rec.raw)).astype(np.float32)
        target = tifffile.imread(str(self.data_root / rec.target)).astype(np.float32)
        if target.ndim == 3:
            raise ValueError(f"expected (C,Z,Y,X) target, got {target.shape}")
        return normalize_volume(raw), target

    def _pick_crop_center(self, target: np.ndarray, mode: str) -> Optional[Tuple[int, int, int]]:
        fg = target[CH_FOREGROUND] > 0.5
        if not bool(fg.any()):
            return None
        if mode == "full":
            return None
        coords = np.column_stack(np.where(fg))
        if mode == "instance":
            skel = target[CH_SKELETON] > 0.5
            if bool(skel.any()):
                coords = np.column_stack(np.where(skel))
        idx = self.rng.randint(0, len(coords) - 1)
        z, y, x = coords[idx]
        return int(z), int(y), int(x)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        raw, target = self._load(rec)
        cz, cy, cx = self.crop_zyx
        if raw.shape != target.shape[1:]:
            raise ValueError(f"raw/target shape mismatch for {rec.sid}")

        modes = ("full", "instance", "random")
        probs = self.crop_mode_probs
        r = self.rng.random()
        mode = modes[0]
        acc = 0.0
        for m, p in zip(modes, probs):
            acc += p
            if r < acc:
                mode = m
                break

        zsl, ysl, xsl = _random_crop_slices(
            raw.shape, self.crop_zyx, center=self._pick_crop_center(target, mode), rng=self.rng
        )
        raw = raw[zsl, ysl, xsl]
        target = target[:, zsl, ysl, xsl]

        if self.augment:
            raw, target = augment_flip_rot90(raw, target, self.rng)

        return {
            "raw": torch.from_numpy(raw[None].astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "sid": rec.sid,
            "t": rec.t,
        }

def chromopose_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "raw": torch.stack([b["raw"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "sid": [b["sid"] for b in batch],
        "t": [b["t"] for b in batch],
    }

def build_dataloader(
    records: Sequence[IndexRecord],
    data_root: Path,
    *,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    crop_zyx: Tuple[int, int, int] = (64, 256, 256),
    crop_mode_probs: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    augment: bool = True,
    seed: int = 0,
) -> DataLoader:
    ds = ChromoPoseVolumeDataset(
        records,
        data_root,
        crop_zyx=crop_zyx,
        crop_mode_probs=crop_mode_probs,
        augment=augment,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=chromopose_collate,
        pin_memory=True,
    )


# ---- Embedded helpers: data_io.py ----

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

@dataclass
class SamplePair:
    folder: Path
    celltype: str
    sample_num: str
    raw_path: Path
    mask_path: Path

def read_tiff(path: Path) -> np.ndarray:
    if tifffile is not None:
        return tifffile.imread(path)
    try:
        import imageio.v3 as imageio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install tifffile or imageio to read training TIFFs") from exc
    return imageio.imread(path)

def parse_folder_fields(folder_name: str) -> tuple[str, str]:
    parts = folder_name.split(None, 2)
    if len(parts) < 3:
        raise ValueError(
            f"Expected a sample folder with prefix, sample number, and cell type: {folder_name!r}"
        )
    return parts[1], parts[2]

def collect_pairs(source_path: Path) -> list[SamplePair]:
    pairs: list[SamplePair] = []
    for subdir in sorted(
        (path for path in Path(source_path).iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        sample_num, celltype = parse_folder_fields(subdir.name)
        pairs.append(
            SamplePair(
                folder=subdir,
                celltype=celltype,
                sample_num=sample_num,
                raw_path=subdir / "raw.tif",
                mask_path=subdir / "mask.tif",
            )
        )
    return pairs

def infer_xyzt_axis_order(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected 4D TIFF, got shape={shape}")

    dims = list(shape)
    largest_axis = max(range(4), key=lambda index: dims[index])
    xy_pair = next(
        (
            (i, j)
            for i in range(4)
            for j in range(i + 1, 4)
            if dims[i] == dims[j]
        ),
        None,
    )
    if xy_pair is None:
        xy_pair = min(
            ((i, j) for i in range(4) for j in range(i + 1, 4)),
            key=lambda pair: (abs(dims[pair[0]] - dims[pair[1]]), -min(dims[pair[0]], dims[pair[1]]), *pair),
        )

    remaining = [index for index in range(4) if index not in xy_pair]
    z_axis, t_axis = sorted(remaining, key=lambda index: dims[index])
    if z_axis == largest_axis:
        z_axis, t_axis = t_axis, z_axis
    x_axis, y_axis = xy_pair
    return x_axis, y_axis, z_axis, t_axis

@dataclass
class VolumeRecord:
    sid: str
    t: int
    raw: str
    mask: str
    target: str
    target_vis: str
    nz: int
    ny: int
    nx: int
    n_inst: int
    split: str = ""
    celltype: str = ""
    sample_num: str = ""

def to_tzyx(arr: np.ndarray) -> np.ndarray:
    shape = tuple(arr.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected 4D array, got {shape}")
    d0, d1, d2, d3 = shape
    if d0 == max(shape) and d1 == min(shape) and d2 == d3:
        return np.ascontiguousarray(arr)
    x_idx, y_idx, z_idx, t_idx = infer_xyzt_axis_order(shape)
    return np.transpose(arr, (t_idx, z_idx, x_idx, y_idx))

def resize_volume_zyx(
    raw: np.ndarray,
    mask: np.ndarray,
    size_zyx: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    zt, yt, xt = size_zyx
    if raw.shape == (zt, yt, xt):
        return raw, mask.astype(mask.dtype, copy=False)
    if torch is None:
        raise RuntimeError("torch is required for 3D resize")

    def _resize(vol: np.ndarray, *, mode: str) -> np.ndarray:
        t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        kwargs = {"size": (zt, yt, xt), "mode": mode}
        if mode == "trilinear":
            kwargs["align_corners"] = False
        out = torch.nn.functional.interpolate(t, **kwargs)
        return out.squeeze().numpy()

    raw_r = _resize(raw, mode="trilinear")
    mask_r = _resize(mask.astype(np.float32), mode="nearest").astype(mask.dtype)
    return raw_r, mask_r

def collect_chimp_pairs(chimp_root: Path) -> List[SamplePair]:
    pairs: List[SamplePair] = []
    chimp_root = Path(chimp_root)
    for split in ("train", "test"):
        split_root = chimp_root / split
        if not split_root.is_dir():
            continue
        for subdir in sorted([p for p in split_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            raw_path = subdir / "raw.tif"
            mask_path = subdir / "mask.tif"
            if not raw_path.is_file():
                continue
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {subdir.name}: {mask_path}")
            try:
                sample_num, celltype = parse_folder_fields(subdir.name)
            except ValueError:
                sample_num, celltype = subdir.name, ""
            pairs.append(
                SamplePair(
                    folder=subdir,
                    celltype=celltype,
                    sample_num=sample_num,
                    raw_path=raw_path,
                    mask_path=mask_path,
                )
            )
    return pairs

def _sid_from_pair(pair: SamplePair) -> str:
    split = pair.folder.parent.name
    folder = pair.folder.name.replace(" ", "_")
    if split in {"train", "test"}:
        return f"{split}_{folder}"
    return folder

def split_t_volumes(raw_4d: np.ndarray, mask_4d: np.ndarray) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    tzyx_raw = to_tzyx(raw_4d)
    tzyx_mask = to_tzyx(mask_4d)
    if tzyx_raw.shape != tzyx_mask.shape:
        raise ValueError(f"raw/mask shape mismatch: {tzyx_raw.shape} vs {tzyx_mask.shape}")
    for t in range(int(tzyx_raw.shape[0])):
        yield t, tzyx_raw[t], tzyx_mask[t]

def load_volume_index(index_path: Path) -> List[VolumeRecord]:
    if not index_path.is_file():
        return []
    records: List[VolumeRecord] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            fields = VolumeRecord.__dataclass_fields__
            records.append(VolumeRecord(**{k: d[k] for k in fields if k in d}))
    return records

def write_index(records: List[VolumeRecord], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
