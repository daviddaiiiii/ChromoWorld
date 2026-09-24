#!/usr/bin/env python
"""
YOLO Label Extraction Script for Chromosome Segmentation

Reads images from image_path, runs 24-class segmentation with ChromoPix model,
extracts instance masks via connected components, converts to normalized polygon
coordinates (YOLO segmentation format), and writes class_id, confidence, polygon
to txt files in out_path.
"""

import os
import sys
import argparse
import glob
import multiprocessing as mp
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm


def _find_project_root():
    """Find the ChromoWorld root so defaults work from any current directory."""
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / 'release').is_dir():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def _resolve_path(value):
    """Resolve a user-supplied path relative to the Chromo World project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _import_chromopix_model(code_root):
    """Import the ChromoPix model without requiring a fixed checkout path."""
    code_root = Path(code_root).expanduser().resolve()
    if code_root.name == "chromopix":
        package_root = code_root
        import_root = code_root.parent
    else:
        package_root = code_root / "chromopix"
        import_root = code_root

    if package_root.is_dir():
        sys.path.insert(0, str(package_root))
    if import_root.is_dir():
        sys.path.insert(0, str(import_root))
    dino_hub_root = import_root / "dinov3"
    if (dino_hub_root / "hubconf.py").is_file():
        sys.path.insert(0, str(dino_hub_root))

    try:
        from model import Model
    except ImportError:
        from chromopix.model import Model
    return Model


def load_model_from_checkpoint(checkpoint_path, args):
    """Load ChromoPix model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    if hasattr(args, 'gpu_id') and args.gpu_id is not None:
        gid = args.gpu_id
        if isinstance(gid, str) and gid.strip().isdigit():
            gid = int(gid.strip())
        device = torch.device(f'cuda:{int(gid)}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("Loading DINOv3 backbone...")
    backbone = torch.hub.load(
        repo_or_dir=args.dinov3_repo,
        model=args.model_name,
        source="local",
        weights=args.model_weights,
    )
    backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad = False

    model_class = _import_chromopix_model(args.chromopix_code)
    model = model_class(backbone, args)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    return model, device


def predict_all_classes_batch(model, image, num_classes=24, device=None):
    """Predict all class masks for one image. Returns (num_classes, H, W) float tensor."""
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        images_batch = image.unsqueeze(0).repeat(num_classes, 1, 1, 1).to(device)
        class_ids = torch.arange(num_classes).to(device)
        logits = model(images_batch, class_ids=class_ids)['mask_logits']
        pred_probs = torch.sigmoid(logits)[:, 0].cpu()
    return pred_probs


def predict_all_classes_multi_image(model, images_batch, num_classes=24, device=None):
    """
    Predict all class masks for a batch of images in one forward pass.
    images_batch: (B, 3, H, W)
    Returns: (B, num_classes, H, W) float tensor.
    """
    if device is None:
        device = next(model.parameters()).device
    B = images_batch.shape[0]
    # Repeat each image once for every queried class.
    images_repeat = images_batch.unsqueeze(1).repeat(1, num_classes, 1, 1, 1).reshape(
        B * num_classes, images_batch.shape[1], images_batch.shape[2], images_batch.shape[3]
    ).to(device)
    class_ids = torch.arange(num_classes).to(device).unsqueeze(0).repeat(B, 1).reshape(B * num_classes)
    with torch.no_grad():
        logits = model(images_repeat, class_ids=class_ids)['mask_logits']
    pred_probs = torch.sigmoid(logits)[:, 0].cpu().reshape(B, num_classes, images_batch.shape[2], images_batch.shape[3])
    return pred_probs


def instance_masks_to_yolo_polygons(pred_masks_all_classes, orig_h, orig_w, num_classes=24,
                                    min_area=20):
    """
    From per-class probability masks (at model resolution), resize to original size,
    extract instances via connected components, and convert each instance to
    YOLO segmentation line: class_id confidence x1 y1 x2 y2 ... (normalized 0-1).

    pred_masks_all_classes: (NUM_CLASSES, H_model, W_model)
    Returns: list of strings, each one line for one instance.
    """
    lines = []
    for class_id in range(min(num_classes, pred_masks_all_classes.shape[0])):
        pred_mask = pred_masks_all_classes[class_id]
        if torch.is_tensor(pred_mask):
            pred_mask_np = pred_mask.cpu().numpy()
        else:
            pred_mask_np = np.asarray(pred_mask, dtype=np.float32)
        # Resize to the original image size before normalizing coordinates.
        if pred_mask_np.shape[0] != orig_h or pred_mask_np.shape[1] != orig_w:
            pred_mask_np = cv2.resize(
                pred_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
        binary = (pred_mask_np > 0.5).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary)
        for label_id in range(1, num_labels):
            instance_mask = (labels == label_id).astype(np.uint8)
            if instance_mask.sum() < min_area:
                continue
            conf = float(np.mean(pred_mask_np[instance_mask > 0])) if np.any(instance_mask > 0) else 0.0
            contours, _ = cv2.findContours(
                instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            # Flatten the OpenCV contour and normalize its coordinates.
            pts = contour.reshape(-1, 2)
            x_norm = (pts[:, 0].astype(np.float64) / orig_w).tolist()
            y_norm = (pts[:, 1].astype(np.float64) / orig_h).tolist()
            polygon_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(x_norm, y_norm))
            line = f"{class_id} {conf:.6f} {polygon_str}"
            lines.append(line)
    return lines


def _parse_gpu_ids(gpu_arg):
    """Parse a GPU list; return None for CPU mode."""
    if gpu_arg is None:
        return None
    s = str(gpu_arg).strip().lower()
    if s in ("", "cpu", "-1", "none"):
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def _args_to_worker_dict(args):
    """Return a picklable configuration for multiprocessing workers."""
    d = vars(args).copy()
    d.pop("image_path", None)
    d.pop("ext", None)
    return d


def _gpu_worker(gpu_id: int, image_paths, cfg: dict):
    """Load the model on one GPU and process its assigned image list."""
    if not image_paths:
        return
    wargs = SimpleNamespace(**cfg)
    wargs.gpu_id = int(gpu_id)
    print(f"[GPU {gpu_id}] loading model, {len(image_paths)} image(s)...", flush=True)
    model, device = load_model_from_checkpoint(wargs.checkpoint, wargs)
    run_extraction_loop(model, device, image_paths, wargs)
    print(f"[GPU {gpu_id}] done.", flush=True)


def run_extraction_loop(model, device, image_files, args):
    """Run batched inference for a list of images and write TXT labels."""
    transform = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(image_files), batch_size), desc="Extracting labels"):
        end = min(start + batch_size, len(image_files))
        paths_batch = image_files[start:end]
        batch_tensors = []
        batch_meta = []
        for path in paths_batch:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Skip {path}: {e}")
                continue
            orig_w, orig_h = img.size
            if args.invert_colormap:
                img = ImageOps.invert(img)
            tensor = transform(img)
            batch_tensors.append(tensor)
            batch_meta.append((orig_h, orig_w, os.path.splitext(os.path.basename(path))[0]))
        if not batch_tensors:
            continue
        stack = torch.stack(batch_tensors, dim=0)
        pred_probs_batch = predict_all_classes_multi_image(
            model, stack, num_classes=args.num_classes, device=device
        )
        for i, (orig_h, orig_w, base_name) in enumerate(batch_meta):
            pred_probs = pred_probs_batch[i]
            lines = instance_masks_to_yolo_polygons(
                pred_probs,
                orig_h,
                orig_w,
                num_classes=args.num_classes,
                min_area=args.min_area,
            )
            out_file = os.path.join(args.out_path, f"{base_name}.txt")
            with open(out_file, "w") as f:
                f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='Extract YOLO segmentation labels from ChromoPix 24-class predictions.'
    )
    parser.add_argument('--image_path', type=str,
                        default='release/Dataset/2D/test/image',
                        help='Directory containing input images')
    parser.add_argument('--out_path', type=str,
                        default='Analyze/results/predictions/ChromoPix',
                        help='Directory for YOLO TXT predictions')
    parser.add_argument('--chromopix_ckpt', '--checkpoint', dest='checkpoint', type=str, 
                        default='release/Checkpoint/ChromoPix/chromopix.pth',
                        help='Path to ChromoPix model checkpoint')
    parser.add_argument('--chromopix_code', default='release/Code/ChromoPix', type=str,
                        help='ChromoPix code root or its chromopix package directory')
    parser.add_argument('--dinov3_repo', default='release/Code/ChromoPix/dinov3', type=str,
                        help='Bundled local DINOv3 source used by torch.hub')
    parser.add_argument('--model_name', default='dinov3_vitb16', type=str)
    parser.add_argument('--model_weights',
                        default='release/Checkpoint/ChromoPix/dinov3_vitb16.pth',
                        type=str)
    parser.add_argument('--num_layers', default=12, type=int)
    parser.add_argument('--num_classes', default=24, type=int)
    parser.add_argument('--num_decoder_layers', default=4, type=int)
    parser.add_argument('--prompt_type', default=['semantic'], type=list)
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument(
        '--gpu_id',
        type=str,
        default='0,1,2,3',
        help='GPU IDs. Use one ID, a comma-separated list, cpu, or -1.',
    )
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of images to run in parallel per forward pass (>=1).')
    parser.add_argument('--img_h', default=512, type=int)
    parser.add_argument('--img_w', default=512, type=int)
    parser.add_argument('--invert_colormap', action='store_true',
                        help='Invert image colormap before inference (match training)')
    parser.add_argument('--min_area', default=20, type=int,
                        help='Minimum instance area in pixels to export')
    parser.add_argument('--ext', type=str, default='png', nargs='+',
                        help='Image extensions to glob (default: png). e.g. --ext png jpg')
    args = parser.parse_args()

    if isinstance(args.ext, str):
        args.ext = [args.ext]

    args.image_path = _resolve_path(args.image_path)
    args.out_path = _resolve_path(args.out_path)
    args.checkpoint = _resolve_path(args.checkpoint)
    args.chromopix_code = _resolve_path(args.chromopix_code)
    args.dinov3_repo = _resolve_path(args.dinov3_repo)
    args.model_weights = _resolve_path(args.model_weights)
    os.makedirs(args.out_path, exist_ok=True)

    # Collect image paths
    image_files = []
    for e in args.ext:
        image_files.extend(glob.glob(os.path.join(args.image_path, f'*.{e}')))
    image_files = sorted(set(image_files))
    if not image_files:
        print(f"No images found under {args.image_path} with extensions {args.ext}")
        return

    gpu_list = _parse_gpu_ids(args.gpu_id)

    if gpu_list is None:
        print(f"Found {len(image_files)} images (batch_size={args.batch_size}). CPU mode...")
        args.gpu_id = None
        model, device = load_model_from_checkpoint(args.checkpoint, args)
        run_extraction_loop(model, device, image_files, args)
    elif len(gpu_list) == 1:
        args.gpu_id = gpu_list[0]
        print(
            f"Found {len(image_files)} images (batch_size={args.batch_size}, GPU {args.gpu_id}). "
            f"Loading model..."
        )
        model, device = load_model_from_checkpoint(args.checkpoint, args)
        run_extraction_loop(model, device, image_files, args)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Multiple GPUs were requested, but CUDA is unavailable.")
        for g in gpu_list:
            if g < 0 or g >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Invalid gpu_id={g}; {torch.cuda.device_count()} visible GPU(s) are available."
                )
        n = len(gpu_list)
        chunks = [[] for _ in range(n)]
        for i, p in enumerate(image_files):
            chunks[i % n].append(p)
        cfg = _args_to_worker_dict(args)
        print(
            f"Found {len(image_files)} images on {n} GPUs (batch_size={args.batch_size} per GPU). "
            f"Spawning workers..."
        )
        ctx = mp.get_context("spawn")
        procs = []
        for gid, chunk in zip(gpu_list, chunks):
            if not chunk:
                continue
            proc = ctx.Process(target=_gpu_worker, args=(gid, chunk, cfg))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
        for proc in procs:
            if proc.exitcode != 0:
                raise RuntimeError(f"A worker exited with code {proc.exitcode}.")

    print(f"Labels written to {args.out_path}")


if __name__ == '__main__':
    main()
