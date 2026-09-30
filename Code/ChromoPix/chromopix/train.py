import io
import os
import pickle
import tarfile
import urllib
import logging
import sys
from pathlib import Path as _PathBoot
# Ensure ChromoWorld root is importable and is default CWD for relative paths
_CW_ROOT = _PathBoot(__file__).resolve().parents[1]
if str(_CW_ROOT) not in sys.path:
    sys.path.insert(0, str(_CW_ROOT))
if str(_PathBoot(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(_PathBoot(__file__).resolve().parent))
_DINOV3_HUB_ROOT = _CW_ROOT / "dinov3"
if (_DINOV3_HUB_ROOT / "hubconf.py").is_file() and str(_DINOV3_HUB_ROOT) not in sys.path:
    sys.path.insert(0, str(_DINOV3_HUB_ROOT))
os.chdir(_CW_ROOT)

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloader import create_chromoworld_2d_dataloader, custom_collate_fn
import argparse
from torch.utils.data.distributed import DistributedSampler
import misc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torchvision
from model import Model
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb
import time
from datetime import datetime
from pathlib import Path


def setup_logging(args):
    """Setup comprehensive logging system"""
    # Create logs directory
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup different loggers for different purposes
    loggers = {}
    
    # Main training logger (detailed logs)
    main_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create main logger for current rank
    main_logger = logging.getLogger(f'training_rank_{misc.get_rank()}')
    main_logger.setLevel(logging.INFO)
    
    # File handler for this rank
    main_handler = logging.FileHandler(
        os.path.join(log_dir, f'training_rank_{misc.get_rank()}.log')
    )
    main_handler.setFormatter(main_formatter)
    main_logger.addHandler(main_handler)
    
    # Console handler only for main process
    if misc.is_main_process():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(main_formatter)
        main_logger.addHandler(console_handler)
        
        # Detailed training log (main process only)
        detailed_handler = logging.FileHandler(
            os.path.join(log_dir, 'detailed_training.log')
        )
        detailed_handler.setFormatter(main_formatter)
        main_logger.addHandler(detailed_handler)
    
    loggers['main'] = main_logger
    
    # CSV loggers for metrics (main process only)
    if misc.is_main_process():
        # Training metrics CSV
        train_csv_path = os.path.join(log_dir, 'train_metrics.csv')
        with open(train_csv_path, 'w') as f:
            f.write('epoch,batch_idx,loss,avg_loss,learning_rate,time\n')
        
        # Validation metrics CSV  
        val_csv_path = os.path.join(log_dir, 'val_metrics.csv')
        with open(val_csv_path, 'w') as f:
            f.write('epoch,val_loss,val_dice,time\n')
            
        # Epoch summary CSV
        epoch_csv_path = os.path.join(log_dir, 'epoch_metrics.csv')
        with open(epoch_csv_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_dice,best_dice,epoch_time,lr\n')
            
        loggers['train_csv'] = train_csv_path
        loggers['val_csv'] = val_csv_path
        loggers['epoch_csv'] = epoch_csv_path
        
        # Save config
        config_path = os.path.join(log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
    
    return loggers


def log_metrics(loggers, metric_type, **kwargs):
    """Log metrics to appropriate files"""
    if not misc.is_main_process():
        return
        
    timestamp = datetime.now().isoformat()
    
    if metric_type == 'train' and 'train_csv' in loggers:
        with open(loggers['train_csv'], 'a') as f:
            f.write(f"{kwargs.get('epoch', '')},{kwargs.get('batch_idx', '')},"
                   f"{kwargs.get('loss', '')},{kwargs.get('avg_loss', '')},"
                   f"{kwargs.get('lr', '')},{timestamp}\n")
    
    elif metric_type == 'val' and 'val_csv' in loggers:
        with open(loggers['val_csv'], 'a') as f:
            f.write(f"{kwargs.get('epoch', '')},{kwargs.get('val_loss', '')},"
                   f"{kwargs.get('val_dice', '')},{timestamp}\n")
    
    elif metric_type == 'epoch' and 'epoch_csv' in loggers:
        with open(loggers['epoch_csv'], 'a') as f:
            f.write(f"{kwargs.get('epoch', '')},{kwargs.get('train_loss', '')},"
                   f"{kwargs.get('val_loss', '')},{kwargs.get('val_dice', '')},"
                   f"{kwargs.get('best_dice', '')},{kwargs.get('epoch_time', '')},"
                   f"{kwargs.get('lr', '')}\n")


def load_model(args):
    """Load DINOv3 backbone model"""
    if misc.is_main_process():
        print("Loading DINOv3 model...")
    
    model = torch.hub.load(
        repo_or_dir=args.dinov3_repo,
        model=args.model_name,
        source="local",
        weights=args.model_weights,
    )
    
    # Set device
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu).eval()
    
    return model


def create_finetune_model(backbone, args):
    """Create the finetune model with decoder and deconv head"""
    if misc.is_main_process():
        print("Creating finetune model...")
    config = args
    
    # Create model
    model = Model(backbone, config)
    model.cuda(args.gpu)
    
    # Wrap with DDP
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
        unwrap_model = model.module
    else:
        unwrap_model = model
    
    # Freeze backbone parameters if specified
    if getattr(args, 'freeze_backbone', True):  # Default to freeze
        if misc.is_main_process():
            print("Freezing DINOv3 backbone parameters...")
        for param in unwrap_model.backbone.parameters():
            param.requires_grad = False
        unwrap_model.backbone.eval()  # Set to eval mode
    else:
        if misc.is_main_process():
            print("Not freezing DINOv3 backbone parameters...")
        for param in unwrap_model.backbone.parameters():
            param.requires_grad = True
        unwrap_model.backbone.train()  # Set to train mode
    
    if misc.is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Model: {model}")
        print(f"Model parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {frozen_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
        
    
    return model


def create_ema_model(model, args):
    """Create EMA (Exponential Moving Average) model for stable feature targets"""
    if misc.is_main_process():
        print("Creating EMA model...")
    
    # Deep copy the model
    ema_model = deepcopy(model)
    
    # Disable gradients for EMA model
    for param in ema_model.parameters():
        param.requires_grad = False
    
    # Set to eval mode
    ema_model.eval()
    
    if misc.is_main_process():
        print("EMA model created and set to eval mode")
    
    return ema_model


def update_ema_model(ema_model, model, momentum):
    """Update EMA model parameters"""
    # Get unwrapped models for parameter access
    model_unwrapped = model.module if hasattr(model, 'module') else model
    ema_model_unwrapped = ema_model.module if hasattr(ema_model, 'module') else ema_model
    
    with torch.no_grad():
        for ema_param, param in zip(ema_model_unwrapped.parameters(), model_unwrapped.parameters()):
            ema_param.data.mul_(momentum).add_(param.data, alpha=1 - momentum)


def load_data(args):
    """Load training and validation data with distributed sampling"""
    if misc.is_main_process():
        print("Loading data...")
    
    transform = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Create datasets
    train_dataset = create_chromoworld_2d_dataloader(
        args.train_label_dir,
        args.train_image_dir,
        args.train_visualize_dir,
        transform=transform,
        batch_size=args.batch_size,
        shuffle=False,  # Handled by sampler
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        return_dataset=True,  # Return dataset instead of dataloader
        data_augmentation=args.data_augmentation,
        invert_colormap=args.invert_colormap
    )
    
    test_dataset = create_chromoworld_2d_dataloader(
        args.test_label_dir,
        args.test_image_dir,
        args.test_visualize_dir,
        transform=transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        return_dataset=True,
        data_augmentation=False,
        invert_colormap=args.invert_colormap
    )
    
    # Create distributed samplers
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=(train_sampler is None),
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, test_loader, train_sampler, test_sampler


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation"""
    pred = torch.sigmoid(pred)
    
    # Handle different target formats
    if target.dim() == 3:  # (B, H, W) - single channel
        target = target.unsqueeze(1)  # (B, 1, H, W)
    
    # Convert target to float
    target = target.float()
    
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    pred = torch.sigmoid(pred)
    
    # Handle different target formats
    if target.dim() == 3:  # (B, H, W) - single channel
        target = target.unsqueeze(1)  # (B, 1, H, W)
    
    # Convert target to float
    target = target.float()
    
    ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    pt = pred * target + (1 - pred) * (1 - target)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def combined_loss(pred, target, alpha=0.5, beta=0.5):
    """Combined dice and focal loss"""
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    return alpha * dice + beta * focal


def invariant_loss(clean_features, blur_features, loss_type='cosine'):
    """
    Compute invariance loss between clean and blur features.
    
    Args:
        clean_features: (B, C, H, W) - features from clean image (detached)
        blur_features: (B, C, H, W) - features from PSF blurred image
        loss_type: 'cosine' or 'l2'
    
    Returns:
        loss: scalar tensor
    """
    # Flatten spatial dimensions: (B, C, H, W) -> (B, C*H*W)
    clean_flat = clean_features.flatten(2)  # (B, C, H*W)
    blur_flat = blur_features.flatten(2)    # (B, C, H*W)
    
    if loss_type == 'cosine':
        # Cosine similarity loss: 1 - cosine_similarity
        # Normalize along feature dimension
        clean_norm = F.normalize(clean_flat, dim=1)  # (B, C, H*W)
        blur_norm = F.normalize(blur_flat, dim=1)    # (B, C, H*W)
        
        # Compute cosine similarity per spatial location
        cos_sim = (clean_norm * blur_norm).sum(dim=1)  # (B, H*W)
        loss = 1 - cos_sim.mean()
    
    elif loss_type == 'l2':
        # L2 loss (MSE)
        loss = F.mse_loss(blur_flat, clean_flat)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss


def train_one_epoch(model, ema_model, train_loader, optimizer, scheduler, epoch, args, loggers):
    """
    Train for one epoch with invariant feature learning using EMA.
    
    Process:
    1. EMA model provides stable clean features (no grad)
    2. Training model forward on clean image for segmentation (with grad)
    3. Training model forward on PSF blurred image for segmentation (with grad)
    4. Compute invariant loss between EMA clean features and training blur features
    5. One backward pass with combined loss
    6. Update EMA model parameters
    """
    model.train()
    ema_model.eval()  # EMA model always in eval mode
    
    # Get unwrapped model for accessing backbone
    unwrap_model = model.module if hasattr(model, 'module') else model
    ema_unwrap_model = ema_model.module if hasattr(ema_model, 'module') else ema_model
    
    # If backbone is frozen, keep it in eval mode
    if getattr(args, 'freeze_backbone', True):
        unwrap_model.backbone.eval()
    
    total_loss = 0
    total_blur_seg_loss = 0
    total_clean_seg_loss = 0
    total_inv_loss = 0
    num_batches = len(train_loader)
    
    # Setup metric logger for distributed training
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('blur_seg_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('clean_seg_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('inv_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = f'Epoch: [{epoch}]'
    
    if misc.is_main_process():
        print(f"\nTraining Epoch {epoch}/{args.epochs}")
    
    for batch_idx, batch in enumerate(metric_logger.log_every(train_loader, args.log_freq, header)):
        # Get clean image and SSL branch image (psf_image or shape_image when --shape_ssl)
        clean_images = batch['image'].cuda(args.gpu, non_blocking=True)
        ssl_images = batch['shape_image'].cuda(args.gpu, non_blocking=True) if getattr(args, 'shape_ssl', False) else batch['psf_image'].cuda(args.gpu, non_blocking=True)
        class_ids = torch.tensor(batch['class_id']).cuda(args.gpu, non_blocking=True)
        masks = batch['mask_image'].cuda(args.gpu, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        batch_size = clean_images.size(0)

        # Step 1: EMA model provides stable clean features (no grad)
        with torch.no_grad():
            ema_clean_features = ema_unwrap_model.forward_backbone(clean_images)[-1]  # (B, C, H, W)

        # Step 2: Merge clean and SSL branch into one batch for single forward (avoid in-place errors)
        combined_images = torch.cat([clean_images, ssl_images], dim=0)  # (2B, 3, H, W)
        combined_class_ids = torch.cat([class_ids, class_ids], dim=0)  # (2B,)

        # Step 3: Single forward pass for both clean and blur
        combined_output = model(combined_images, class_ids=combined_class_ids)
        combined_features = combined_output['features']  # (2B, C, H, W)
        combined_logits = combined_output['mask_logits']  # (2B, 1, H, W)

        # Step 4: Split results back to clean and blur
        clean_features = combined_features[:batch_size]
        blur_features = combined_features[batch_size:]
        clean_logits = combined_logits[:batch_size]
        blur_logits = combined_logits[batch_size:]

        # Step 5: Compute losses
        clean_seg_loss = combined_loss(clean_logits, masks)
        blur_seg_loss = combined_loss(blur_logits, masks)
        inv_loss = invariant_loss(ema_clean_features, blur_features, loss_type=args.inv_loss_type)
        
        # Step 6: Combined loss and single backward
        loss = clean_seg_loss + blur_seg_loss + args.inv_loss_weight * inv_loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        # Step 7: Update EMA model with dynamic momentum
        # Dynamic momentum: starts faster, becomes more stable
        total_steps = epoch * len(train_loader) + batch_idx + 1
        ema_momentum = min(1 - 1 / total_steps, 0.999)
        update_ema_model(ema_model, model, ema_momentum)

        # For logging
        scaled_inv_loss = args.inv_loss_weight * inv_loss
        
        # Update metrics
        metric_logger.update(loss=loss.item())
        metric_logger.update(blur_seg_loss=blur_seg_loss.item())
        metric_logger.update(clean_seg_loss=clean_seg_loss.item())
        metric_logger.update(inv_loss=inv_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        total_loss += loss.item()
        total_blur_seg_loss += blur_seg_loss.item()
        total_clean_seg_loss += clean_seg_loss.item()
        total_inv_loss += inv_loss.item()
        
        # Log metrics
        if misc.is_main_process() and batch_idx % args.log_freq == 0:
            log_metrics(loggers, 'train',
                       epoch=epoch,
                       batch_idx=batch_idx,
                       loss=loss.item(),
                       avg_loss=total_loss / (batch_idx + 1),
                       lr=optimizer.param_groups[0]["lr"])
            
            loggers['main'].info(
                f'Epoch {epoch} Batch {batch_idx}/{num_batches} - '
                f'Loss: {loss.item():.4f} (Blur Seg: {blur_seg_loss.item():.4f}, Clean Seg: {clean_seg_loss.item():.4f}, Inv: {inv_loss.item():.4f}), '
                f'Avg Loss: {total_loss / (batch_idx + 1):.4f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )
        
        # Log to wandb
        if args.use_wandb and misc.is_main_process() and batch_idx % args.log_freq == 0:
            wandb.log({
                'train_loss': loss.item(),
                'train_blur_seg_loss': blur_seg_loss.item(),
                'train_clean_seg_loss': clean_seg_loss.item(),
                'train_inv_loss': inv_loss.item(),
                'train_avg_loss': total_loss / (batch_idx + 1),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch + batch_idx / num_batches
            })
    
    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    avg_loss = total_loss / num_batches
    avg_blur_seg_loss = total_blur_seg_loss / num_batches
    avg_clean_seg_loss = total_clean_seg_loss / num_batches
    avg_inv_loss = total_inv_loss / num_batches
    
    if misc.is_main_process():
        loggers['main'].info(
            f'Training epoch {epoch} completed - '
            f'Avg Loss: {avg_loss:.4f} (Blur Seg: {avg_blur_seg_loss:.4f}, Clean Seg: {avg_clean_seg_loss:.4f}, Inv: {avg_inv_loss:.4f})'
        )
    
    return avg_loss


def compute_cluster_images(features, visualize_images_denorm, num_classes):
    """
    Compute clustering images for a batch of features.
    Optimized: uses MiniBatchKMeans with n_init=1 for speed.
    
    Args:
        features: (B, C, H, W) - encoder output features (patch-level, e.g., 32x32)
        visualize_images_denorm: (B, 3, H', W') - denormalized input images
        num_classes: number of clusters
    
    Returns:
        cluster_images: list of (3, H', W') tensors - cluster visualizations
        cluster_overlays: list of (3, H', W') tensors - cluster overlays on input
    """
    from sklearn.cluster import MiniBatchKMeans
    
    B, C, H, W = features.shape
    cluster_images = []
    cluster_overlays = []
    
    # Pre-compute colormap
    cmap = plt.colormaps['tab20']
    
    for i in range(B):
        # Get feature map for this sample
        feat = features[i]  # (C, H, W)
        
        # Reshape to (H*W, C) for clustering
        feat_flat = feat.flatten(1).transpose(0, 1).cpu().numpy()  # (H*W, C)
        
        # Apply MiniBatchKMeans (much faster than KMeans)
        n_clusters = min(num_classes, feat_flat.shape[0])
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=1, batch_size=256)
        cluster_labels = kmeans.fit_predict(feat_flat)  # (H*W,)
        
        # Reshape to spatial format
        cluster_map = cluster_labels.reshape(H, W)
        
        # Create colorful visualization
        cluster_norm = cluster_map.astype(np.float32) / max(1, n_clusters - 1)
        cluster_colored = cmap(cluster_norm)[:, :, :3]  # (H, W, 3), remove alpha
        
        # Get input image size (H' x W')
        input_img = visualize_images_denorm[i].cpu()  # (3, H', W')
        img_h, img_w = input_img.shape[1], input_img.shape[2]
        
        # Resize cluster map to match original image size using nearest neighbor
        cluster_colored_tensor = torch.from_numpy(cluster_colored).permute(2, 0, 1).float()  # (3, H, W)
        cluster_colored_tensor = F.interpolate(
            cluster_colored_tensor.unsqueeze(0), 
            size=(img_h, img_w),
            mode='nearest'
        ).squeeze(0)  # (3, H', W')
        
        # Create cluster overlay on input image (semi-transparent)
        alpha = 0.5
        cluster_overlay = alpha * cluster_colored_tensor + (1 - alpha) * input_img
        cluster_overlay = torch.clamp(cluster_overlay, 0, 1)
        
        cluster_images.append(cluster_colored_tensor)
        cluster_overlays.append(cluster_overlay)
    
    return cluster_images, cluster_overlays


def save_chromoworld3d_visualizations(model, images, visualize_images, name, vis_dir, num_classes, args):
    """
    Save visualization images as a 2x2 grid for ChromoWorld 3D images (no ground truth).
    
    Grid layout:
        Top-left: Original image
        Top-right: Clustering visualization
        Bottom-left: Clustering overlay on original image
        Bottom-right: Multiclass mask overlay on original image
    
    Args:
        model: the trained model
        images: (B, 3, H, W) - input images (normalized for model)
        visualize_images: (B, 3, H, W) - input images (normalized)
        name: list of sample names
        vis_dir: visualization directory
        num_classes: number of classes for multiclass mask
        args: arguments
    """
    # Get unwrapped model
    unwrap_model = model.module if hasattr(model, 'module') else model
    
    B = images.size(0)
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    visualize_images_denorm = visualize_images * std + mean
    visualize_images_denorm = torch.clamp(visualize_images_denorm, 0, 1)
    
    # Get features for clustering
    with torch.no_grad():
        features = unwrap_model.forward_backbone(images)[-1]  # (B, C, H, W)
    
    # Compute cluster images
    cluster_images, cluster_overlays = compute_cluster_images(
        features, visualize_images_denorm, num_classes
    )
    
    # Get multiclass masks: batch all class_ids together for efficiency
    # For each image, we need masks for all classes
    with torch.no_grad():
        # Repeat images for all classes: (B, 3, H, W) -> (B * num_classes, 3, H, W)
        images_repeated = images.unsqueeze(1).repeat(1, num_classes, 1, 1, 1)  # (B, num_classes, 3, H, W)
        images_repeated = images_repeated.view(B * num_classes, 3, images.size(2), images.size(3))
        
        # Create class_ids for all: (B * num_classes,)
        class_ids_all = torch.arange(num_classes, device=images.device).unsqueeze(0).repeat(B, 1)  # (B, num_classes)
        class_ids_all = class_ids_all.view(-1)  # (B * num_classes,)
        
        # Forward through model in batches to avoid OOM
        max_batch_size = 32
        all_masks = []
        for start_idx in range(0, B * num_classes, max_batch_size):
            end_idx = min(start_idx + max_batch_size, B * num_classes)
            batch_images = images_repeated[start_idx:end_idx]
            batch_class_ids = class_ids_all[start_idx:end_idx]
            
            output = model(batch_images, class_ids=batch_class_ids)
            batch_masks = torch.sigmoid(output['mask_logits'])  # (batch, 1, H, W)
            all_masks.append(batch_masks)
        
        # Concatenate all masks
        all_masks = torch.cat(all_masks, dim=0)  # (B * num_classes, 1, H, W)
        all_masks = all_masks.view(B, num_classes, all_masks.size(2), all_masks.size(3))  # (B, num_classes, H, W)
    
    # Create colormap for multiclass visualization
    cmap = plt.colormaps['tab20']
    
    # Save each sample
    for i in range(B):
        vis_img = visualize_images_denorm[i].cpu()  # (3, H, W)
        
        # ========== Top-left: Original image ==========
        original_img = vis_img  # (3, H, W)
        
        # ========== Top-right: Clustering visualization ==========
        cluster_img = cluster_images[i]  # (3, H, W)
        
        # ========== Bottom-left: Clustering overlay on original ==========
        cluster_overlay = cluster_overlays[i]  # (3, H, W)
        
        # ========== Bottom-right: Multiclass mask overlay on original ==========
        # Get masks for this sample: (num_classes, H, W)
        sample_masks = all_masks[i]  # (num_classes, H, W)
        
        # Binarize masks
        sample_masks_binary = (sample_masks > 0.5).float()
        
        # Create multiclass overlay: each class gets a different color
        multiclass_overlay = vis_img.clone()  # (3, H, W)
        
        # Create a colored mask image
        H_mask, W_mask = sample_masks_binary.shape[1], sample_masks_binary.shape[2]
        colored_mask = torch.zeros(3, H_mask, W_mask)  # (3, H, W)
        
        alpha = 0.5
        for class_idx in range(num_classes):
            mask = sample_masks_binary[class_idx].cpu()  # (H, W)
            if mask.sum() > 0:
                # Get color for this class
                color = cmap(class_idx / max(1, num_classes - 1))[:3]
                color_tensor = torch.tensor(color).float().view(3, 1, 1)
                
                # Apply color to mask region
                mask_3d = mask.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
                colored_mask = torch.where(mask_3d > 0, color_tensor, colored_mask)
        
        # Overlay on original image
        mask_region = colored_mask.sum(dim=0) > 0  # (H, W)
        multiclass_overlay[:, mask_region] = (
            alpha * colored_mask[:, mask_region] + (1 - alpha) * vis_img[:, mask_region]
        )
        multiclass_overlay = torch.clamp(multiclass_overlay, 0, 1)
        
        # ========== Create 2x2 grid ==========
        # Order: [original, cluster, cluster_overlay, multiclass_overlay]
        grid = torchvision.utils.make_grid(
            [original_img, cluster_img, cluster_overlay, multiclass_overlay],
            nrow=2, padding=2, pad_value=1.0
        )
        
        # Save the grid image
        save_path = os.path.join(vis_dir, f'{name[i]}_quad.png')
        torchvision.utils.save_image(grid, save_path, normalize=False)


def save_visualizations(masks, visualize_images, logits, features, name, vis_dir, num_classes):
    """
    Save visualization images as a single 3x2 grid (only called on main process).
    
    Grid layout:
        Row 1: GT mask, Visualize image
        Row 2: Pred mask, Pred mask overlay on input (semi-transparent)
        Row 3: Cluster image, Cluster overlay on input (16x16 patches, semi-transparent)
    
    Args:
        masks: (B, H, W) - ground truth masks
        visualize_images: (B, 3, H, W) - input images (normalized)
        logits: (B, 1, H, W) or (B, 2, H, W) - prediction logits
        features: (B, C, H, W) - encoder output features for clustering
        name: list of sample names
        vis_dir: visualization directory
        num_classes: number of clusters for feature clustering
    """
    # Convert logits to predictions
    pred_masks = torch.sigmoid(logits)
    
    # For binary segmentation, take the foreground channel (index 1)
    if pred_masks.size(1) == 2:
        pred_masks = pred_masks[:, 1:2, :, :]  # (B, 1, H, W)
    
    # Convert to binary masks
    pred_binary = (pred_masks > 0.5).float()
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    visualize_images_denorm = visualize_images * std + mean
    visualize_images_denorm = torch.clamp(visualize_images_denorm, 0, 1)
    
    # Compute cluster images for all samples
    cluster_images, cluster_overlays = compute_cluster_images(
        features, visualize_images_denorm, num_classes
    )
    
    # Save each sample in the batch
    for i in range(visualize_images.size(0)):
        vis_img = visualize_images_denorm[i].cpu()  # (3, H, W)
        
        # ========== Row 1: GT mask, Visualize image ==========
        if masks[i].dim() == 2:
            gt_mask = masks[i].unsqueeze(0).float()  # (1, H, W)
        else:
            gt_mask = masks[i].float()  # (1, H, W)
        gt_mask_rgb = gt_mask.repeat(3, 1, 1).cpu()  # (3, H, W)
        
        # ========== Row 2: Pred mask, Pred mask overlay on input ==========
        pred_mask_rgb = pred_binary[i].repeat(3, 1, 1).cpu()  # (3, H, W)
        
        # Create overlay: predicted mask semi-transparent on input image
        alpha = 0.5
        mask_colored = torch.zeros_like(vis_img)  # (3, H, W)
        mask_colored[1, :, :] = pred_binary[i].squeeze(0).cpu()  # Green channel for mask
        
        pred_overlay = vis_img.clone()
        mask_region = pred_binary[i].squeeze(0).cpu() > 0  # (H, W)
        pred_overlay[:, mask_region] = alpha * mask_colored[:, mask_region] + (1 - alpha) * vis_img[:, mask_region]
        pred_overlay = torch.clamp(pred_overlay, 0, 1)
        
        # ========== Row 3: Cluster image, Cluster overlay on input ==========
        cluster_img = cluster_images[i]  # (3, H, W)
        cluster_overlay = cluster_overlays[i]  # (3, H, W)
        
        # ========== Create 3x2 grid ==========
        # Order: [gt_mask, vis_img, pred_mask, pred_overlay, cluster_img, cluster_overlay]
        # nrow=2 means 2 columns -> 3 rows
        grid = torchvision.utils.make_grid(
            [gt_mask_rgb, vis_img, pred_mask_rgb, pred_overlay, cluster_img, cluster_overlay],
            nrow=2, padding=2, pad_value=1.0
        )
        
        # Save the grid image
        save_path = os.path.join(vis_dir, f'{name[i]}.png')
        torchvision.utils.save_image(grid, save_path, normalize=False)


def evaluate_chromoworld_3d(model, epoch, args, loggers):
    """
    Evaluate and visualize on ChromoWorld 3D MIP dapi images (no ground truth).
    Creates 2x2 grid visualizations: original, cluster, cluster overlay, multiclass mask overlay.
    """
    if not misc.is_main_process():
        return
    
    if not args.is_visualize:
        return
    
    model.eval()
    
    # ChromoWorld 3D MIP dapi directory
    chromoworld_3d_dir = './data/3D/misc/naive_mip'
    
    if not os.path.exists(chromoworld_3d_dir):
        loggers['main'].warning(f"ChromoWorld 3D directory not found: {chromoworld_3d_dir}")
        return
    
    # Create visualization directory
    vis_dir = os.path.join(args.output_dir, f'eval_epoch_{epoch}_chromoworld3d')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(chromoworld_3d_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) == 0:
        loggers['main'].warning(f"No images found in {chromoworld_3d_dir}")
        return
    
    loggers['main'].info(f"Evaluating on {len(image_files)} ChromoWorld 3D images...")
    
    # Process images in batches
    batch_size = min(4, args.batch_size)  # Smaller batch for memory efficiency
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_files))
            batch_files = image_files[start_idx:end_idx]
            
            # Load images
            images_list = []
            visualize_images_list = []
            names = []
            
            for img_file in batch_files:
                img_path = os.path.join(chromoworld_3d_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                
                # Apply transform for model input
                img_tensor = transform(img)
                images_list.append(img_tensor)
                
                # Also keep for visualization (same transform)
                visualize_images_list.append(img_tensor)
                
                # Get base name without extension
                base_name = os.path.splitext(img_file)[0]
                names.append(base_name)
            
            # Stack into batch
            images = torch.stack(images_list).cuda(args.gpu, non_blocking=True)
            visualize_images = torch.stack(visualize_images_list).cuda(args.gpu, non_blocking=True)
            
            # Save visualizations
            save_chromoworld3d_visualizations(
                model, images, visualize_images, names, vis_dir,
                num_classes=args.num_classes, args=args
            )
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                loggers['main'].info(f"ChromoWorld 3D visualization: {batch_idx + 1}/{num_batches} batches processed")
    
    loggers['main'].info(f"ChromoWorld 3D visualization completed. Saved to {vis_dir}")


def evaluate(model, test_loader, epoch, args, loggers):
    """Evaluate the model with distributed support and feature clustering visualization"""
    model.eval()
    
    # Get unwrapped model
    unwrap_model = model.module if hasattr(model, 'module') else model
    
    total_loss = 0
    total_dice = 0
    num_batches = len(test_loader)
    
    # Setup metric logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # Create visualization directory
    if args.is_visualize:
        vis_dir = os.path.join(args.output_dir, f'eval_epoch_{epoch}')
        os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_logger.log_every(test_loader, 50, header)):
            images = batch['image'].cuda(args.gpu, non_blocking=True)
            masks = batch['mask_image'].cuda(args.gpu, non_blocking=True)
            class_ids = torch.tensor(batch['class_id']).cuda(args.gpu, non_blocking=True)
            visualize_images = batch['visualize_image'].cuda(args.gpu, non_blocking=True)
            name = batch['base_name']
            
            # Forward pass - get both features and logits
            output = model(images, class_ids=class_ids)
            features = output['features']  # (B, C, H, W) - encoder output
            logits = output['mask_logits']
            
            # Calculate loss
            loss = combined_loss(logits, masks)
            
            # Calculate dice score
            dice = 1 - dice_loss(logits, masks)
            
            # Update metrics
            metric_logger.update(loss=loss.item())
            metric_logger.update(dice=dice.item())
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            # Save visualizations if enabled (3x2 grid with GT, pred, cluster)
            # Only save for first 5 batches to speed up evaluation
            if args.is_visualize and batch_idx < 5:
                save_visualizations(
                    masks, visualize_images, logits, features, name, vis_dir,
                    num_classes=args.num_classes
                )
    
    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    
    # Calculate averages
    avg_loss = misc.all_reduce_mean(total_loss / num_batches)
    avg_dice = misc.all_reduce_mean(total_dice / num_batches)
    
    if misc.is_main_process():
        print(f'Validation - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}')
        loggers['main'].info(f'Validation epoch {epoch} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}')
        
        # Log metrics
        log_metrics(loggers, 'val', epoch=epoch, val_loss=avg_loss, val_dice=avg_dice)
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                'val_dice': avg_dice,
                'epoch': epoch
            })
    
    # Also evaluate on ChromoWorld 3D images (no ground truth, visualization only)
    # evaluate_chromoworld_3d(model, epoch, args, loggers)
    
    return avg_loss, avg_dice


def save_checkpoint(model, ema_model, optimizer, scheduler, epoch, best_dice, args, loggers):
    """Save model checkpoint (only on main process)"""
    if not misc.is_main_process():
        return
    
    # Get model without DDP wrapper
    model_to_save = model.module if hasattr(model, 'module') else model
    ema_model_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'ema_model_state_dict': ema_model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_dice': best_dice,
        'args': args
    }
    
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    loggers['main'].info(f'Checkpoint saved to {checkpoint_path}')


def get_args_parser():
    parser = argparse.ArgumentParser('DDP training', add_help=False)
    
    # DINOv3 Model
    parser.add_argument('--dinov3_repo', default=str(_CW_ROOT / 'dinov3'), type=str,
                help='dinov3 repo location')
    parser.add_argument('--model_name', default='dinov3_vitb16', type=str,
                      help='model name')
    parser.add_argument('--model_weights', default=str(_CW_ROOT.parents[1] / 'Checkpoint' / 'Outsource' / 'DINOv3' / 'dinov3_vitb16.pth'), type=str,
                      help='model weights')
    parser.add_argument('--num_layers', default=12, type=int, # vits/sp-12, vitb-12, vitl-24, vith/hp-32, vit7b-40
                      help='number of layers')
    
    # Model Architecture
    parser.add_argument('--num_classes', default=24, type=int,
                      help='number of classes')
    parser.add_argument('--num_decoder_layers', default=4, type=int,
                      help='number of decoder layers')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                      help='freeze backbone parameters (default: False)')
    parser.add_argument('--prompt_type', default=['semantic'], type=list,
                      help='prompt type')
    parser.add_argument('--drop_path', type=float, default=0.0,
                      help='drop path rate')
    
    # Invariant Learning
    parser.add_argument('--inv_loss_weight', type=float, default=0.1,
                      help='weight for invariant loss (default: 0.1)')
    parser.add_argument('--shape_ssl', action='store_true',
                        help='Use shape_image (pseudo-binary) instead of psf_image for SSL alignment')
    parser.add_argument('--inv_loss_type', type=str, default='cosine',
                      choices=['cosine', 'l2'],
                      help='type of invariant loss (default: cosine)')

    # Dataset
    parser.add_argument('--train_label_dir', default=str(_CW_ROOT.parents[1] / 'Dataset/2D/train/label'),
                      help='train label folder path')
    parser.add_argument('--train_image_dir', default=str(_CW_ROOT.parents[1] / 'Dataset/2D/train/image'),
                      help='train image folder path')
    parser.add_argument('--train_visualize_dir', default=str(_CW_ROOT / 'outputs/train_visualization'),
                        help='train visualize folder path')
    parser.add_argument('--test_label_dir', default=str(_CW_ROOT.parents[1] / 'Dataset/2D/test/label'),
                        help='test label folder path')
    parser.add_argument('--test_image_dir', default=str(_CW_ROOT.parents[1] / 'Dataset/2D/test/image'),
                        help='test image folder path')
    parser.add_argument('--test_visualize_dir', default=str(_CW_ROOT / 'outputs/test_visualization'),
                        help='test visualize folder path')
    parser.add_argument('--img_h', default=512, type=int,
                      help='image height')
    parser.add_argument('--img_w', default=512, type=int,
                      help='image width')
    parser.add_argument('--batch_size', default=32, type=int,
                      help='batch size per GPU')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool,
                      help='pin memory')
    parser.add_argument('--data_augmentation', default=True, type=bool,
                      help='data augmentation')
    parser.add_argument('--invert_colormap', action='store_true',
                      help='Invert image colormap before model input (default: False).')
    # Optimizer
    parser.add_argument('--weight_decay', type=float, default=0.02,
                      help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='minimum learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                      help='start lr ratio')
    
    # Basic Training
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--output_dir', default='./outputs',
                      help='save path')
    parser.add_argument('--device', default='cuda',
                      help='device')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                      help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                      help='start epoch')
    parser.add_argument('--if_load_opt', default=False, type=bool,
                      help='whether to load optimizer and scheduler state from checkpoint')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                      help='use wandb for logging')
    parser.add_argument('--log_freq', default=10, type=int,
                      help='logging frequency')
    parser.add_argument('--experiment_name', default='dinov3_chromosome_seg', type=str,
                      help='experiment name for logging')
    
    # Distributed Training
    parser.add_argument('--world_size', default=1, type=int,
                      help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                      help='distributed training url')
    
    # Evaluate&Save
    parser.add_argument('--eval_freq', type=int, default=10,
                      help='evaluation frequency')
    parser.add_argument('--is_visualize', default=True, type=bool,
                      help='visualize')
    parser.add_argument('--save_freq', type=int, default=10,
                      help='save last frequency')
    
    return parser


def main():
    args = get_args_parser().parse_args()
    
    # Initialize distributed training
    misc.init_distributed_mode(args)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory with timestamp
    if misc.is_main_process():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    
    # Synchronize output_dir across processes
    if args.distributed:
        # Broadcast output_dir from rank 0 to all other ranks
        if misc.is_main_process():
            output_dir_tensor = torch.tensor([ord(c) for c in args.output_dir], dtype=torch.long).cuda()
            dir_length = torch.tensor([len(args.output_dir)], dtype=torch.long).cuda()
        else:
            dir_length = torch.tensor([0], dtype=torch.long).cuda()
            output_dir_tensor = torch.zeros(500, dtype=torch.long).cuda()  # Max length buffer
        
        dist.broadcast(dir_length, 0)
        output_dir_tensor = output_dir_tensor[:dir_length.item()]
        dist.broadcast(output_dir_tensor, 0)
        
        if not misc.is_main_process():
            args.output_dir = ''.join([chr(c.item()) for c in output_dir_tensor])
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    loggers = setup_logging(args)
    
    if misc.is_main_process():
        loggers['main'].info(f"Starting distributed training with {misc.get_world_size()} GPUs")
        loggers['main'].info(f"Arguments: {vars(args)}")
    
    # Initialize wandb
    if args.use_wandb and misc.is_main_process():
        wandb.init(
            project="dinov3-chromosome-segmentation-ddp",
            config=vars(args),
            name=f"ddp_{args.model_name}_{misc.get_world_size()}gpus"
        )
    
    # Load model and data
    backbone = load_model(args)
    model = create_finetune_model(backbone, args)
    
    # Create EMA model for stable feature targets
    ema_model = create_ema_model(model, args)
    
    train_loader, test_loader, train_sampler, test_sampler = load_data(args)
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Calculate total steps for scheduler
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    if warmup_steps > 0:
        # Create warmup scheduler: linear from min_lr to base_lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=args.min_lr / args.lr,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create cosine annealing scheduler: from base_lr to min_lr
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=args.min_lr
        )
        
        # Combine warmup and cosine annealing
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.min_lr
        )
    
    # Resume from checkpoint if specified
    best_dice = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if misc.is_main_process():
                print(f"Loading checkpoint from {args.resume}")
            
            # Map to current device
            checkpoint = torch.load(args.resume,
                                    map_location=f'cuda:{args.gpu}',
                                    weights_only=False)
            
            # Load model state
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load EMA model state if available
            if 'ema_model_state_dict' in checkpoint:
                if hasattr(ema_model, 'module'):
                    ema_model.module.load_state_dict(checkpoint['ema_model_state_dict'])
                else:
                    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                if misc.is_main_process():
                    loggers['main'].info("Loaded EMA model state from checkpoint")
            
            # Load optimizer and scheduler state if specified
            if args.if_load_opt:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # If loading optimizer state, resume from checkpoint epoch
                args.start_epoch = checkpoint['epoch'] + 1
                if misc.is_main_process():
                    loggers['main'].info("Loaded optimizer and scheduler state from checkpoint")
                    loggers['main'].info(f"Resuming from epoch {args.start_epoch}")
            else:
                # If not loading optimizer state, start from epoch 0
                args.start_epoch = 0
                if misc.is_main_process():
                    loggers['main'].info("Skipped loading optimizer and scheduler state (if_load_opt=False)")
                    loggers['main'].info("Starting training from epoch 0")
            
            # Always load best_score for model saving comparison
            best_dice = checkpoint['best_dice']
            
            if misc.is_main_process():
                loggers['main'].info(f"Loaded model weights, best dice: {best_dice:.4f}, starting from epoch {args.start_epoch}")
        else:
            if misc.is_main_process():
                loggers['main'].warning(f"No checkpoint found at {args.resume}")
    
    # Training loop
    if misc.is_main_process():
        loggers['main'].info("Starting training...")
    
    val_loss, val_dice = evaluate(model, test_loader, args.start_epoch-1, args, loggers)
    
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Set epoch for distributed sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        if misc.is_main_process():
            loggers['main'].info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, ema_model, train_loader, optimizer, scheduler, epoch, args, loggers)
        
        # Evaluate
        val_loss, val_dice = 0, 0
        if epoch % args.eval_freq == 0:
            val_loss, val_dice = evaluate(model, test_loader, epoch, args, loggers)
        
        epoch_time = time.time() - epoch_start_time
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, ema_model, optimizer, scheduler, epoch, best_dice, args, loggers)
            if misc.is_main_process():
                loggers['main'].info(f"New best dice score: {best_dice:.4f}")
        
        # Save checkpoint periodically
        if epoch % args.save_freq == 0:
            save_checkpoint(model, ema_model, optimizer, scheduler, epoch, best_dice, args, loggers)
        
        # Log epoch summary
        if misc.is_main_process():
            log_metrics(loggers, 'epoch',
                       epoch=epoch,
                       train_loss=train_loss,
                       val_loss=val_loss,
                       val_dice=val_dice,
                       best_dice=best_dice,
                       epoch_time=epoch_time,
                       lr=optimizer.param_groups[0]["lr"])
            
            loggers['main'].info(
                f'Epoch {epoch} completed in {epoch_time:.2f}s - '
                f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Val Dice: {val_dice:.4f}, Best Dice: {best_dice:.4f}'
            )
    
    if misc.is_main_process():
        loggers['main'].info(f"Training completed! Best dice score: {best_dice:.4f}")
        print(f"Training completed! Best dice score: {best_dice:.4f}")
    
    if args.use_wandb and misc.is_main_process():
        wandb.finish()
    
    # Cleanup
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
