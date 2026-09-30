#!/usr/bin/env python3
"""Train ChromoPose 3D U-Net (skeleton flow + multi-head supervision)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from dataloader import (  # noqa: E402
    IndexRecord,
    build_dataloader,
    chromopose_collate,
    filter_by_split,
    load_index,
    split_records,
)
from model import build_model  # noqa: E402
from evaluate import (  # noqa: E402
    compute_instance_mask_from_pred,
    eval_test_set,
    instance_count_metrics,
    mean_iou_instances,
)
try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

@dataclass
class Meter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    def mean(self) -> float:
        return self.total / max(1, self.count)

def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v!r}")

def parse_crop_probs(spec: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in str(spec).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("crop-mode-probs must be 'full,instance,random'")
    s = sum(parts)
    if s <= 0:
        raise argparse.ArgumentTypeError("crop-mode-probs must sum > 0")
    return (parts[0] / s, parts[1] / s, parts[2] / s)

def is_main_process(args) -> bool:
    if not getattr(args, "distributed", False):
        return True
    return int(getattr(args, "rank", 0)) == 0

def setup_distributed(args) -> None:
    if not getattr(args, "distributed", False):
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.rank))
    torch.cuda.set_device(args.local_rank)

def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    k0 = next(iter(state))
    if not str(k0).startswith("module."):
        return state
    return OrderedDict((str(k)[7:], v) for k, v in state.items())

def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model

def setup_output_dir(args) -> Path:
    out = Path(args.output_dir)
    if is_main_process(args):
        out.mkdir(parents=True, exist_ok=True)
        (out / "ckpt").mkdir(exist_ok=True)
        with (out / "args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)
    if getattr(args, "distributed", False) and dist.is_available() and dist.is_initialized():
        dist.barrier()
    return out

def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _PKG_ROOT / "outputs" / f"chromopose_{stamp}"

def init_csv(path: Path, header: List[str]) -> None:
    if path.is_file():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

def append_csv(path: Path, row: List[Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def build_loaders(args) -> Tuple[DataLoader, DataLoader]:
    index_path = Path(args.index_path)
    root = Path(args.data_root)
    records = load_index(index_path)
    pool = filter_by_split(records, str(getattr(args, "train_split", "train")))
    if not pool:
        pool = records
    train_recs, val_recs = split_records(pool, val_ratio=float(args.val_ratio), seed=int(args.seed))
    crop = (int(args.crop_z), int(args.crop_y), int(args.crop_x))
    probs = parse_crop_probs(args.crop_mode_probs)
    train_loader = build_dataloader(
        train_recs,
        root,
        batch_size=int(args.batch_size),
        shuffle=not getattr(args, "distributed", False),
        num_workers=int(args.num_workers),
        crop_zyx=crop,
        crop_mode_probs=probs,
        augment=parse_bool(args.if_aug),
        seed=int(args.seed),
    )
    val_loader = build_dataloader(
        val_recs,
        root,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        crop_zyx=crop,
        crop_mode_probs=(1.0, 0.0, 0.0),
        augment=False,
        seed=int(args.seed) + 1,
    )
    if getattr(args, "distributed", False):
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=int(args.batch_size),
            sampler=DistributedSampler(train_loader.dataset, shuffle=True),
            num_workers=int(args.num_workers),
            collate_fn=chromopose_collate,
            pin_memory=True,
            drop_last=True,
        )
    return train_loader, val_loader

def build_scheduler(optimizer, args, steps_per_epoch: int):
    warmup_steps = int(args.warmup_epochs) * steps_per_epoch
    total_steps = int(args.epochs) * steps_per_epoch
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=float(args.lr_min))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[max(1, warmup_steps)])

def save_checkpoint(path, model, optimizer, scheduler, *, epoch, global_step, best_miou, args) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_miou": best_miou,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
        },
        path,
    )

def load_checkpoint(path, model, optimizer=None, scheduler=None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(strip_module_prefix(ckpt["model_state_dict"]), strict=False)
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt

@torch.no_grad()
def quick_val_instance_eval(model, val_loader, device, args) -> Dict[str, float]:
    """Run ChromoPose postprocess on one val batch (expensive; optional)."""
    if tifffile is None:
        return {}
    model.eval()
    batch = next(iter(val_loader))
    raw = batch["raw"].to(device)
    pred = model(raw)[0]
    rec = val_loader.dataset.records[0]
    gt_mask = tifffile.imread(str(Path(args.data_root) / rec.mask))
    pred_mask = compute_instance_mask_from_pred(
        pred,
        fg_threshold=float(args.eval_fg_threshold),
        skel_threshold=float(args.eval_skel_threshold),
        max_merge_dist=float(args.eval_max_merge_dist),
        min_merge_score=float(args.eval_min_merge_score),
        min_size=int(args.eval_min_size),
    )
    # crop alignment for quick metric
    cz, cy, cx = int(args.crop_z), int(args.crop_y), int(args.crop_x)
    gt_crop = gt_mask[:cz, :cy, :cx]
    if pred_mask.shape != gt_crop.shape:
        return {}
    miou = mean_iou_instances(pred_mask, gt_crop)
    cm = instance_count_metrics(pred_mask, gt_crop)
    return {"miou": miou, **cm}

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    args,
    *,
    epoch,
    global_step,
    train_csv: Optional[Path] = None,
):
    model.train()
    loss_m = Meter()
    part_m = {k: Meter() for k in ("flow", "foreground", "distance", "skeleton", "boundary", "embedding")}
    log_every = max(1, int(getattr(args, "log_every", 50)))
    pbar = tqdm(loader, desc=f"train e{epoch}", disable=not is_main_process(args))
    for batch in pbar:
        raw = batch["raw"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(raw)
        total, parts = chromopose_loss(
            pred,
            target,
            flow_scale=float(args.flow_scale),
            fg_weight=float(args.fg_weight),
            flow_weight=float(args.flow_weight),
            dist_weight=float(args.dist_weight),
            skel_weight=float(args.skel_weight),
            boundary_weight=float(args.boundary_weight),
            emb_weight=float(args.emb_weight),
            use_embedding=parse_bool(args.use_embedding),
        )
        total.backward()
        if float(args.grad_clip) > 0:
            nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), float(args.grad_clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = int(raw.shape[0])
        loss_m.update(float(total.item()), bs)
        for k, v in parts.items():
            part_m[k].update(float(v.item()), bs)
        global_step += 1
        if (
            is_main_process(args)
            and train_csv is not None
            and global_step % log_every == 0
        ):
            append_csv(
                train_csv,
                [
                    epoch,
                    global_step,
                    loss_m.mean(),
                    part_m["flow"].mean(),
                    part_m["foreground"].mean(),
                    part_m["distance"].mean(),
                    part_m["skeleton"].mean(),
                    part_m["boundary"].mean(),
                    part_m["embedding"].mean(),
                    optimizer.param_groups[0]["lr"],
                ],
            )
            loss_m = Meter()
            part_m = {k: Meter() for k in part_m}
        if is_main_process(args):
            pbar.set_postfix(loss=f"{loss_m.mean():.4f}", flow=f"{part_m['flow'].mean():.4f}")
    out = {k: m.mean() for k, m in part_m.items()}
    out["loss"] = loss_m.mean()
    return out, global_step

@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    loss_m = Meter()
    for batch in loader:
        raw = batch["raw"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        pred = model(raw)
        total, _ = chromopose_loss(pred, target, flow_scale=float(args.flow_scale))
        loss_m.update(float(total.item()), int(raw.shape[0]))
    return {"val_loss": loss_m.mean()}

def run_test_eval(model, args, device, vis_root, *, epoch, epoch_dir_name):
    test_epoch_dir = vis_root / epoch_dir_name
    summary = eval_test_set(
        model,
        Path(args.index_path),
        Path(args.data_root),
        test_epoch_dir,
        device,
        epoch=epoch,
        size_zyx=(int(args.crop_z), int(args.crop_y), int(args.crop_x)),
        distance_clip=float(args.distance_clip),
        fg_threshold=float(args.eval_fg_threshold),
        skel_threshold=float(args.eval_skel_threshold),
        max_dist=float(args.eval_max_dist),
        max_merge_dist=float(args.eval_max_merge_dist),
        min_merge_score=float(args.eval_min_merge_score),
        min_size=int(args.eval_min_size),
        flow_niter=int(args.eval_flow_niter),
        rank=int(getattr(args, "rank", 0)),
        world_size=int(getattr(args, "world_size", 1)),
        show_tqdm=is_main_process(args),
    )
    return summary, str(test_epoch_dir)

def log_test_eval(
    test_summary,
    test_csv,
    test_vis_dir,
    epoch,
    *,
    best_miou,
    out_dir,
    model,
    optimizer,
    scheduler,
    global_step,
    args,
    update_best=True,
):
    if not test_summary:
        return best_miou
    miou_t0 = float(np.mean([m["miou_t0"] for m in test_summary]))
    miou_mean = float(np.mean([m.get("miou_mean", m["miou_t0"]) for m in test_summary]))
    count_err_t0 = float(np.mean([m["count_err_t0"] for m in test_summary]))
    count_err_mean = float(np.mean([m.get("count_err_mean", m["count_err_t0"]) for m in test_summary]))
    if update_best and miou_t0 > best_miou:
        best_miou = miou_t0
        save_checkpoint(
            out_dir / "ckpt" / "best.pth",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            global_step=global_step,
            best_miou=best_miou,
            args=args,
        )
    append_csv(
        test_csv,
        [epoch, len(test_summary), miou_t0, miou_mean, count_err_t0, count_err_mean, test_vis_dir],
    )
    label = f"epoch {epoch}"
    print(
        f"test {label}: n={len(test_summary)} miou_t0={miou_t0:.4f} miou_mean={miou_mean:.4f} "
        f"count_err_t0={count_err_t0:.4f} → {test_vis_dir}"
    )
    return best_miou

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ChromoPose training")
    p.add_argument("--index-path", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--crop-z", type=int, default=64)
    p.add_argument("--crop-y", type=int, default=256)
    p.add_argument("--crop-x", type=int, default=256)
    p.add_argument("--crop-mode-probs", type=str, default="0.5,0.3,0.2")
    p.add_argument("--backbone", type=str, default="unet3d")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--use-embedding", type=parse_bool, default=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--flow-scale", type=float, default=5.0)
    p.add_argument("--fg-weight", type=float, default=1.0)
    p.add_argument("--flow-weight", type=float, default=1.0)
    p.add_argument("--dist-weight", type=float, default=0.5)
    p.add_argument("--skel-weight", type=float, default=1.0)
    p.add_argument("--boundary-weight", type=float, default=0.5)
    p.add_argument("--emb-weight", type=float, default=0.25)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--if-aug", type=parse_bool, default=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-fg-threshold", type=float, default=0.5)
    p.add_argument("--eval-skel-threshold", type=float, default=0.35)
    p.add_argument("--eval-max-merge-dist", type=float, default=8.0)
    p.add_argument("--eval-min-merge-score", type=float, default=0.75)
    p.add_argument("--eval-min-size", type=int, default=15)
    p.add_argument("--eval-max-dist", type=float, default=24.0)
    p.add_argument("--eval-flow-niter", type=int, default=40)
    p.add_argument("--distance-clip", type=float, default=32.0)
    p.add_argument("--test-every", type=int, default=5, help="run test eval every N epochs (skip epoch 0)")
    p.add_argument("--log-every", type=int, default=50, help="log train losses to train.csv every N steps")
    p.add_argument("--resume", type=str, default="")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    setup_distributed(args)
    if not str(args.output_dir).strip():
        args.output_dir = str(default_output_dir())
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = setup_output_dir(args)
    if is_main_process(args):
        print(f"Output: {out_dir} | device={device}")
    train_loader, val_loader = build_loaders(args)
    model = build_model(
        backbone=args.backbone,
        base_channels=int(args.base_ch),
        levels=int(args.levels),
        use_embedding=parse_bool(args.use_embedding),
    ).to(device)
    if getattr(args, "distributed", False):
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    optimizer = AdamW(unwrap_model(model).parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=max(1, len(train_loader)))
    start_epoch, global_step, best_miou = 0, 0, 0.0
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, optimizer, scheduler)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_miou = float(ckpt.get("best_miou", 0.0))
    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    test_vis_root = out_dir / "test_vis"
    if is_main_process(args):
        init_csv(
            train_csv,
            [
                "epoch",
                "step",
                "loss",
                "flow",
                "foreground",
                "distance",
                "skeleton",
                "boundary",
                "embedding",
                "lr",
            ],
        )
        init_csv(
            test_csv,
            ["epoch", "n_test_samples", "miou_t0", "miou_mean", "count_err_t0", "count_err_mean", "test_vis_dir"],
        )
    for epoch in range(start_epoch, int(args.epochs)):
        if getattr(args, "distributed", False) and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            args,
            epoch=epoch,
            global_step=global_step,
            train_csv=train_csv if is_main_process(args) else None,
        )
        val_metrics = validate(model, val_loader, device, args) if is_main_process(args) else {"val_loss": 0.0}
        test_summary, test_vis_dir = [], ""
        run_test = int(args.test_every) > 0 and epoch > 0 and (epoch % int(args.test_every) == 0)
        if run_test:
            test_summary, test_vis_dir = run_test_eval(
                unwrap_model(model),
                args,
                device,
                test_vis_root,
                epoch=epoch,
                epoch_dir_name=f"epoch_{epoch:04d}",
            )
        if getattr(args, "distributed", False) and dist.is_available() and dist.is_initialized():
            dist.barrier()
        if is_main_process(args):
            if test_summary:
                best_miou = log_test_eval(
                    test_summary,
                    test_csv,
                    test_vis_dir,
                    epoch,
                    best_miou=best_miou,
                    out_dir=out_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    args=args,
                )
            save_checkpoint(
                out_dir / "ckpt" / "last.pth", model, optimizer, scheduler,
                epoch=epoch, global_step=global_step, best_miou=best_miou, args=args,
            )
            print(
                f"epoch {epoch}: loss={train_metrics['loss']:.4f} "
                f"val={val_metrics['val_loss']:.4f} step={global_step}"
            )
    cleanup_distributed()

# ---- Embedded helpers: losses.py ----

import torch
import torch.nn.functional as F

from model import (
    BASE_TARGET_CHANNELS,
    CH_BOUNDARY,
    CH_DISTANCE,
    CH_EMB_0,
    CH_FLOW_X,
    CH_FLOW_Y,
    CH_FLOW_Z,
    CH_FOREGROUND,
    CH_SKELETON,
    EMBEDDING_DIM,
)

def _flow_loss(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    fg_mask: torch.Tensor,
    *,
    flow_scale: float,
    cosine_weight: float = 1.0,
    smooth_l1_weight: float = 1.0,
) -> torch.Tensor:
    """Cosine + Smooth L1 on foreground voxels only."""
    fs = float(flow_scale)
    gt = fs * gt_flow
    pred = pred_flow

    mask = fg_mask > 0.5
    if not bool(mask.any()):
        return pred_flow.new_zeros(())

    pred_fg = pred_flow.permute(0, 2, 3, 4, 1)[mask]
    gt_fg = gt.permute(0, 2, 3, 4, 1)[mask]

    pred_n = F.normalize(pred_fg, dim=1, eps=1e-6)
    gt_n = F.normalize(gt_fg, dim=1, eps=1e-6)
    cosine = 1.0 - (pred_n * gt_n).sum(dim=1)
    smooth = F.smooth_l1_loss(pred_fg, gt_fg)
    return float(cosine_weight) * cosine.mean() + float(smooth_l1_weight) * smooth

def chromopose_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    flow_scale: float = 5.0,
    fg_weight: float = 1.0,
    flow_weight: float = 1.0,
    dist_weight: float = 0.5,
    skel_weight: float = 1.0,
    boundary_weight: float = 0.5,
    emb_weight: float = 0.25,
    use_embedding: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        pred: (B, C, Z, Y, X) — flow xyz, fg logit, distance, skel logit, boundary logit, [emb]
        gt:   (B, C, Z, Y, X) — matching regression / probability targets
    """
    pred_flow = torch.stack([pred[:, CH_FLOW_X], pred[:, CH_FLOW_Y], pred[:, CH_FLOW_Z]], dim=1)
    gt_flow = torch.stack([gt[:, CH_FLOW_X], gt[:, CH_FLOW_Y], gt[:, CH_FLOW_Z]], dim=1)
    fg_mask = gt[:, CH_FOREGROUND]

    flow_l = _flow_loss(pred_flow, gt_flow, fg_mask, flow_scale=flow_scale)
    fg_bce = F.binary_cross_entropy_with_logits(pred[:, CH_FOREGROUND], gt[:, CH_FOREGROUND])
    dist_l = F.smooth_l1_loss(pred[:, CH_DISTANCE], gt[:, CH_DISTANCE], reduction="none")
    dist_l = (dist_l * fg_mask).sum() / fg_mask.sum().clamp_min(1.0)
    skel_bce = F.binary_cross_entropy_with_logits(pred[:, CH_SKELETON], gt[:, CH_SKELETON])
    boundary_bce = F.binary_cross_entropy_with_logits(pred[:, CH_BOUNDARY], gt[:, CH_BOUNDARY])

    emb_l = pred_flow.new_zeros(())
    if use_embedding and pred.shape[1] > BASE_TARGET_CHANNELS:
        skel_mask = gt[:, CH_SKELETON] > 0.5
        if bool(skel_mask.any()):
            pred_emb = pred[:, CH_EMB_0 : CH_EMB_0 + EMBEDDING_DIM].permute(0, 2, 3, 4, 1)
            gt_emb = gt[:, CH_EMB_0 : CH_EMB_0 + EMBEDDING_DIM].permute(0, 2, 3, 4, 1)
            emb_l = F.mse_loss(pred_emb[skel_mask], gt_emb[skel_mask])

    total = (
        float(fg_weight) * fg_bce
        + float(flow_weight) * flow_l
        + float(dist_weight) * dist_l
        + float(skel_weight) * skel_bce
        + float(boundary_weight) * boundary_bce
        + float(emb_weight) * emb_l
    )
    parts = {
        "flow": flow_l.detach(),
        "foreground": fg_bce.detach(),
        "distance": dist_l.detach(),
        "skeleton": skel_bce.detach(),
        "boundary": boundary_bce.detach(),
        "embedding": emb_l.detach(),
    }
    return total, parts

if __name__ == "__main__":
    main()
