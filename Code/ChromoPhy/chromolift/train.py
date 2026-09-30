#!/usr/bin/env python
"""
Train Lifter (2D->3D flow) for chromosome classification.
Lifter: flow matching with image condition (phy + img). No IDER.
Eval every eval_epochs: save lifted_2d.json, lifted_mip.json, pred_3d.json (k-NN retrieval from 2D gallery), UMAP colored by pred_3d ID.
UMAP is initialized from original 3D JSON before lifter training.
"""

import os
import sys
import json
import argparse
import shutil
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from datetime import datetime
from pathlib import Path
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not available. UMAP visualization will be skipped.")
try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Correlation analysis will be skipped.")
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Flow matching imports for lifter
try:
    from flow_matching.path import CondOTProbPath
    FLOW_MATCHING_AVAILABLE = True
except ImportError:
    FLOW_MATCHING_AVAILABLE = False
    print("Warning: flow_matching not available. Lifter training will be skipped.")
    CondOTProbPath = None

# Color map for classes (from test_semantic_inv.py)
COLOR_MAP = {
    0: (255, 178, 178),  # Chr 1: Light Coral
    1: (255, 51, 51),    # Chr 2: Vivid Red
    2: (153, 0, 0),      # Chr 3: Deep Maroon
    3: (255, 204, 153),  # Chr 4: Pale Orange
    4: (255, 102, 0),    # Chr 5: Bright Orange
    5: (255, 255, 204),  # Chr 6: Cream Yellow
    6: (255, 255, 0),    # Chr 7: Soft Yellow
    7: (255, 255, 102),  # Chr 8: Pure Yellow
    8: (255, 239, 153),  # Chr 9: Canary Yellow
    9: (255, 204, 0),    # Chr 10: Golden
    10: (204, 153, 0),   # Chr 11: Dark Gold
    11: (153, 102, 0),   # Chr 12: Harvest Gold
    12: (204, 255, 204), # Chr 13: Mint Green
    13: (51, 255, 51),   # Chr 14: Lime Green
    14: (0, 102, 0),     # Chr 15: Forest Green
    15: (153, 255, 255), # Chr 16: Pale Cyan
    16: (0, 255, 255),   # Chr 17: Cyan
    17: (0, 128, 128),   # Chr 18: Teal
    18: (153, 153, 255), # Chr 19: Sky Blue
    19: (0, 0, 255),     # Chr 20: Royal Blue
    20: (216, 191, 216), # Chr 21: Thistle/Light Purple
    21: (153, 51, 255),  # Chr 22: Medium Purple
    22: (255, 153, 51),  # Chr X: Amber
    23: (75, 0, 130),    # Chr Y: Indigo
    24: (220, 220, 220), # Unknown: Light Gray
}

# Group definitions
GROUP_A = [0, 1, 2]  # Red Series
GROUP_B = [3, 4]      # Orange Series
GROUP_C = [5, 6, 7, 8, 9, 10, 11, 22]  # Yellow/Gold Series + X
GROUP_D = [12, 13, 14]  # Green Series
GROUP_E = [15, 16, 17]  # Cyan Series
GROUP_F = [18, 19]      # Blue Series
GROUP_G = [20, 21, 23]  # Purple Series + Y
UNKNOWN = [24]  # Not used in training

# Major groups (for inter-group loss)
MAJOR_GROUPS = {
    'A': GROUP_A,
    'B': GROUP_B,
    'C': GROUP_C,
    'D': GROUP_D,
    'E': GROUP_E,
    'F': GROUP_F,
    'G': GROUP_G,
}

# Map class_id to major group (A, B, C, D, E, F, G)
CLASS_TO_MAJOR_GROUP = {}
for group_name, class_ids in MAJOR_GROUPS.items():
    for class_id in class_ids:
        CLASS_TO_MAJOR_GROUP[class_id] = group_name

# Map major group name to index (A=0, B=1, C=2, D=3, E=4, F=5, G=6)
MAJOR_GROUP_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
INDEX_TO_MAJOR_GROUP = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

# Class ID to chromosome name (for centroid labels: 22->'X', 23->'Y')
CLASS_TO_NAME = {
    0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8", 8: "9",
    9: "10", 10: "11", 11: "12", 22: "X", 12: "13", 13: "14", 14: "15",
    15: "16", 16: "17", 17: "18", 18: "19", 19: "20", 20: "21", 21: "22",
    23: "Y", 24: "Unknown",
}


def class_id_to_name(class_id):
    """Return display name for class_id (e.g. 22 -> 'X', 23 -> 'Y')."""
    return CLASS_TO_NAME.get(int(class_id), str(class_id))


def get_ordered_class_ids():
    """Order: 0..11, 22, 12..21, 23, 24 (Chr 12->X->13, Chr 22->Y)."""
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]


# Major group colors (RGB 0-255) for centroid plots
MAJOR_GROUP_COLOR_MAP = {
    'A': (255, 0, 0), 'B': (255, 165, 0), 'C': (255, 255, 0),
    'D': (0, 255, 0), 'E': (0, 255, 255), 'F': (0, 0, 255), 'G': (128, 0, 128),
}
MAJOR_GROUP_INDEX_COLOR_MAP = {
    i: tuple(c / 255.0 for c in MAJOR_GROUP_COLOR_MAP[g]) for i, g in INDEX_TO_MAJOR_GROUP.items()
}


def class_id_to_major_group_index(class_id):
    """
    Convert class_id to major group index (0-6)
    
    Args:
        class_id: class ID (0-23)
    
    Returns:
        major_group_index: index (0-6) corresponding to A-G, or None if invalid
    """
    if class_id not in CLASS_TO_MAJOR_GROUP:
        return None
    major_group_name = CLASS_TO_MAJOR_GROUP[class_id]
    return MAJOR_GROUP_TO_INDEX.get(major_group_name, None)

def labels_to_major_group_labels(labels):
    """
    Convert class_id labels to major group labels
    
    Args:
        labels: numpy array or torch tensor of class_ids (0-23)
    
    Returns:
        major_group_labels: numpy array or torch tensor of major group indices (0-6)
    """
    if isinstance(labels, torch.Tensor):
        major_group_labels = torch.zeros_like(labels)
        for class_id in range(24):
            mask = labels == class_id
            if mask.any():
                major_group_idx = class_id_to_major_group_index(class_id)
                if major_group_idx is not None:
                    major_group_labels[mask] = major_group_idx
        return major_group_labels
    else:  # numpy array
        major_group_labels = np.zeros_like(labels)
        for class_id in range(24):
            mask = labels == class_id
            if mask.any():
                major_group_idx = class_id_to_major_group_index(class_id)
                if major_group_idx is not None:
                    major_group_labels[mask] = major_group_idx
        return major_group_labels

# Minor group is the class_id itself (0, 1, 2, ..., 23)
# This function just returns the class_id as minor group
def get_minor_group(class_id):
    """Get minor group (class_id) for a given class_id"""
    return int(class_id) if 0 <= int(class_id) < 24 else None


def minor_predictions_to_major_predictions(minor_predictions):
    """
    Convert minor class predictions (0-23) to major group predictions (0-6)
    
    Args:
        minor_predictions: numpy array or torch tensor of minor class predictions (0-23)
    
    Returns:
        major_predictions: numpy array or torch tensor of major group indices (0-6)
    """
    if isinstance(minor_predictions, torch.Tensor):
        major_predictions = torch.zeros_like(minor_predictions)
        for class_id in range(24):
            mask = minor_predictions == class_id
            if mask.any():
                major_group_idx = class_id_to_major_group_index(class_id)
                if major_group_idx is not None:
                    major_predictions[mask] = major_group_idx
        return major_predictions
    else:  # numpy array
        major_predictions = np.zeros_like(minor_predictions)
        for class_id in range(24):
            mask = minor_predictions == class_id
            if mask.any():
                major_group_idx = class_id_to_major_group_index(class_id)
                if major_group_idx is not None:
                    major_predictions[mask] = major_group_idx
        return major_predictions


def build_minor_to_major_transformation_matrix(device='cpu'):
    """
    Build transformation matrix M ∈ {0,1}^{7×24} that maps minor classes to major groups
    
    M[i, j] = 1 if minor class j belongs to major group i, else 0
    
    Args:
        device: torch device
    
    Returns:
        M: (7, 24) tensor, transformation matrix
    """
    M = torch.zeros(7, 24, device=device, dtype=torch.float32)
    
    for minor_class_id in range(24):
        major_group_name = CLASS_TO_MAJOR_GROUP.get(minor_class_id, None)
        if major_group_name is not None:
            major_group_idx = MAJOR_GROUP_TO_INDEX.get(major_group_name, None)
            if major_group_idx is not None:
                M[major_group_idx, minor_class_id] = 1.0
    
    return M


# UNUSED FUNCTION - Commented out (consistency loss is computed directly using KL divergence in train_ider_adv)
# def consistency_loss(pminor_probs, pmajor_probs, transformation_matrix, eps=1e-8):
#     """
#     Compute consistency loss using KL divergence
#     
#     P̂major = M · Pminor (derived major distribution from minor probabilities)
#     Linconsistent = DKL(P̂major || Pmajor)
#     
#     Args:
#         pminor_probs: (batch_size, 24) tensor, softmax probabilities of 24 minor classes
#         pmajor_probs: (batch_size, 7) tensor, softmax probabilities of 7 major groups
#         transformation_matrix: (7, 24) tensor, transformation matrix M
#         eps: small value to avoid numerical issues
#     
#     Returns:
#         loss: scalar tensor, KL divergence loss
#     """
#     # Compute derived major distribution: P̂major = M · Pminor
#     # transformation_matrix: (7, 24)
#     # pminor_probs: (batch_size, 24)
#     # Result: (batch_size, 7)
#     p_derived_major = torch.matmul(pminor_probs, transformation_matrix.t())
#     
#     # Add small epsilon to avoid log(0)
#     p_derived_major = p_derived_major + eps
#     pmajor_probs = pmajor_probs + eps
#     
#     # Normalize to ensure they sum to 1
#     p_derived_major = p_derived_major / p_derived_major.sum(dim=1, keepdim=True)
#     pmajor_probs = pmajor_probs / pmajor_probs.sum(dim=1, keepdim=True)
#     
#     # Compute KL divergence: DKL(P̂major || Pmajor) = sum(P̂major * log(P̂major / Pmajor))
#     # = sum(P̂major * log(P̂major)) - sum(P̂major * log(Pmajor))
#     kl_div = p_derived_major * (torch.log(p_derived_major) - torch.log(pmajor_probs))
#     kl_loss = kl_div.sum(dim=1).mean()
#     
#     return kl_loss


def check_feature_range(feat_name, feat_value, is_3d=False):
    """
    Check if feature value is in expected range.
    
    Args:
        feat_name: name of the feature
        feat_value: feature value (float)
        is_3d: if True, check 3D ranges (fractal_dimension: 2-3), else 2D ranges (fractal_dimension: 1-2)
    
    Returns:
        (is_valid, reason): (bool, str) - whether value is valid and reason if invalid
    """
    if not np.isfinite(feat_value):
        return False, "not_finite"
    
    if feat_name == 'fractal_dimension':
        if is_3d:
            # 3D: fractal_dimension should be in [2, 3]
            if feat_value < 2.0 or feat_value > 3.0:
                return False, "out_of_range_3d_[2,3]"
        else:
            # 2D: fractal_dimension should be in [1, 2]
            if feat_value < 1.0 or feat_value > 2.0:
                return False, "out_of_range_2d_[1,2]"
    else:
        # Other features should be in [0, 1]
        if feat_value < 0.0 or feat_value > 1.0:
            return False, "out_of_range_[0,1]"
    
    return True, "valid"

def _get_instance_feature_value(inst: dict, feat_name: str):
    """
    Fetch a feature from either flat or nested storage.
    Also supports renaming: norm_area/norm_volume -> norm_size.
    """
    src = inst.get("features", inst) if isinstance(inst, dict) else {}
    if not isinstance(src, dict):
        src = {}

    v = src.get(feat_name, None)
    if v is not None:
        return v

    # Backward/forward compatibility for unified naming.
    if feat_name == "norm_size":
        v2 = src.get("norm_area", None)
        if v2 is not None:
            return v2
        return src.get("norm_volume", None)
    if feat_name in ("norm_area", "norm_volume"):
        return src.get("norm_size", None)
    return None


class ChromosomeDataset(Dataset):
    """Dataset for chromosome features (2D static cell data)"""
    def __init__(self, json_paths, feature_names, exclude_class=24, scaler=None):
        """
        Args:
            json_paths: list of paths to JSON files
            feature_names: list of feature names to extract
            exclude_class: class_id to exclude (default: 24 for Unknown)
            scaler: StandardScaler or MinMaxScaler for feature normalization (optional)
        """
        self.feature_names = feature_names
        self.exclude_class = exclude_class
        self.scaler = scaler
        
        # Load data from all JSON files
        self.data = []
        self.labels = []
        
        # Statistics for filtering
        total_samples = 0
        skipped_invalid = 0
        skipped_missing = 0
        skipped_out_of_range = 0
        feature_filter_stats = {name: {'count': 0, 'reasons': {}} for name in feature_names}
        
        for json_path in json_paths:
            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found, skipping...")
                continue
            
            with open(json_path, 'r') as f:
                all_instances_data = json.load(f)
            
            for image_data in all_instances_data:
                for inst in image_data['instances']:
                    class_id = int(inst['class_id'])
                    total_samples += 1
                    
                    # Skip unknown class
                    if class_id == exclude_class:
                        continue
                    
                    # Extract features and check ranges
                    features = []
                    has_invalid = False
                    invalid_reason = None
                    
                    for feat_idx, feat_name in enumerate(feature_names):
                        feat_value = _get_instance_feature_value(inst, feat_name)
                        
                        # Check if value is missing
                        if feat_value is None:
                            feature_filter_stats[feat_name]['count'] += 1
                            feature_filter_stats[feat_name]['reasons']['missing'] = \
                                feature_filter_stats[feat_name]['reasons'].get('missing', 0) + 1
                            has_invalid = True
                            invalid_reason = f"missing_{feat_name}"
                            break
                        
                        # Convert to float
                        try:
                            feat_value = float(feat_value)
                        except (ValueError, TypeError):
                            feature_filter_stats[feat_name]['count'] += 1
                            feature_filter_stats[feat_name]['reasons']['conversion_failed'] = \
                                feature_filter_stats[feat_name]['reasons'].get('conversion_failed', 0) + 1
                            has_invalid = True
                            invalid_reason = f"conversion_failed_{feat_name}"
                            break
                        
                        # Check range (2D data: fractal_dimension [1,2], others [0,1])
                        is_valid, reason = check_feature_range(feat_name, feat_value, is_3d=False)
                        if not is_valid:
                            feature_filter_stats[feat_name]['count'] += 1
                            feature_filter_stats[feat_name]['reasons'][reason] = \
                                feature_filter_stats[feat_name]['reasons'].get(reason, 0) + 1
                            has_invalid = True
                            invalid_reason = f"{reason}_{feat_name}"
                            break
                        
                        features.append(feat_value)
                    
                    # Skip this sample if any feature is invalid
                    if has_invalid or len(features) != len(feature_names):
                        if 'missing' in invalid_reason or 'conversion_failed' in invalid_reason:
                            skipped_missing += 1
                        else:
                            skipped_out_of_range += 1
                        skipped_invalid += 1
                        continue
                    
                    self.data.append(features)
                    self.labels.append(class_id)
        
        # Print filtering statistics
        print(f"\n{'='*80}")
        print(f"ChromosomeDataset (2D Static Cell) Filtering Statistics")
        print(f"{'='*80}")
        print(f"Total samples processed: {total_samples}")
        print(f"Valid samples loaded: {len(self.data)}")
        print(f"Skipped samples: {skipped_invalid} ({100.0*skipped_invalid/max(1,total_samples):.2f}%)")
        print(f"  - Missing/invalid values: {skipped_missing}")
        print(f"  - Out of range: {skipped_out_of_range}")
        print(f"\nPer-feature filtering statistics:")
        for feat_name in feature_names:
            stats = feature_filter_stats[feat_name]
            if stats['count'] > 0:
                print(f"  {feat_name}:")
                print(f"    - Filtered samples: {stats['count']} ({100.0*stats['count']/max(1,total_samples):.2f}%)")
                for reason, count in stats['reasons'].items():
                    print(f"      - {reason}: {count}")
        print(f"{'='*80}\n")
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Apply scaling if scaler is provided
        if self.scaler is not None:
            self.data = self.scaler.transform(self.data)
        
        print(f"Loaded {len(self.data)} samples from {len(json_paths)} JSON files")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx]


class MLPFeatureExtractor(nn.Module):
    """Legacy MLP: 8D physical -> feature_dim (no condition). Kept for compatibility."""
    def __init__(self, input_dim=8, hidden_dims=[64, 32], output_dim=32):
        super(MLPFeatureExtractor, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        hidden_layers = []
        for i in range(len(dims) - 2):
            hidden_layers.append(nn.Linear(dims[i], dims[i+1]))
            hidden_layers.append(nn.GELU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(dims[-2], dims[-1])
        self.bn = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x, condition=None, return_hidden=False):
        h = self.hidden_layers(x)
        out = self.bn(self.output_layer(h))
        if return_hidden:
            return h, out
        return out


class DomainDiscriminator(nn.Module):
    """Domain Discriminator for adversarial training
    
    Distinguishes between three domains:
    - Domain 0: Features from 3D MIP (live cell) lifted to 3D
    - Domain 1: Features from 2D (static cell) lifted to 3D
    - Domain 2: Features from real 3D volume (target 3D physical quantities)
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], num_domains: int = 3):
        super(DomainDiscriminator, self).__init__()
        self.num_domains = num_domains
        dims = [input_dim] + hidden_dims + [num_domains]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: input features [B, feature_dim]
        
        Returns:
            logits: [B, num_domains] domain prediction over 3 domains
        """
        return self.network(x)


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x, lambda_grl=1.0):
    return GradientReversalFn.apply(x, lambda_grl)


# ============================================================================
# Flow Matching Lifter: 2D physical -> 3D physical, condition = image [b,2,c]
# Per-layer AdaLN-zero via 3-layer MLP (expand then reduce).
# ============================================================================

from model import FeatureFlowMatchingLifter

def prepare_data_from_folder(data_path, feature_keys):
    """
    Scan data_path for instances_3d.jsonl, instances_3dmip.jsonl, and instances_2d.jsonl (required for IDER).
    Load corresponding npy files, align by instance. Return structured data for lifter/ider.
    
    New normalization scheme:
    - 3D, 2D MIP, 2D static use independent StandardScaler (no special fractal handling)
    - Physical features min/max are stored for boundary penalty loss and post-inference clamping
    
    Returns:
        data: dict with
            'mip_samples', 'static_samples', 'volume_records', 'feature_keys', 'n_phys', 'image_dim',
            'scaler_mip', 'scaler_volume', 'scaler_2d', 'scaler_image',
            'phys_min_3d', 'phys_max_3d' (for boundary penalty and clamping)
    """
    data_path = os.path.abspath(data_path)
    jsonl_3d = os.path.join(data_path, 'instances_3d.jsonl')
    jsonl_3dmip = os.path.join(data_path, 'instances_3dmip.jsonl')
    jsonl_2d = os.path.join(data_path, 'instances_2d.jsonl')
    if not os.path.isfile(jsonl_3d) or not os.path.isfile(jsonl_3dmip):
        raise FileNotFoundError(f"Need instances_3d.jsonl and instances_3dmip.jsonl in {data_path}")
    if not os.path.isfile(jsonl_2d):
        raise FileNotFoundError(f"instances_2d.jsonl is required (2D gallery + domain adversarial). Not found in {data_path}")
    feature_keys = list(feature_keys)

    mip_samples = []
    volume_records = []
    lines_3d = []
    lines_3dmip = []
    with open(jsonl_3d, 'r') as f3, open(jsonl_3dmip, 'r') as fm:
        for line_3d, line_mip in zip(f3, fm):
            line_3d = line_3d.strip()
            line_mip = line_mip.strip()
            if not line_3d or not line_mip:
                continue
            d3 = json.loads(line_3d)
            dm = json.loads(line_mip)
            if d3.get('base_name') != dm.get('base_name') or d3.get('npy_file') != dm.get('npy_file'):
                continue
            npy_path = os.path.join(data_path, d3['npy_file'])
            if not os.path.isfile(npy_path):
                continue
            npy = np.load(npy_path)  # (N, 2, C)
            insts_3d = d3.get('instances', [])
            insts_mip = dm.get('instances', [])
            if len(insts_3d) != len(insts_mip) or npy.shape[0] != len(insts_3d):
                continue
            for i in range(len(insts_3d)):
                inst_3d = insts_3d[i]
                inst_mip = insts_mip[i]
                vec_2d = []
                vec_3d = []
                valid = True
                for k in feature_keys:
                    v2 = inst_mip.get(k)
                    v3 = inst_3d.get(k)
                    if v2 is None or v3 is None:
                        valid = False
                        break
                    try:
                        vec_2d.append(float(v2))
                        vec_3d.append(float(v3))
                    except (TypeError, ValueError):
                        valid = False
                        break
                if not valid or len(vec_2d) != len(feature_keys):
                    continue
                img_feat = npy[i]  # (2, C)
                img_flat = img_feat.flatten().astype(np.float32)
                mip_samples.append({
                    'input_2d': np.array(vec_2d, dtype=np.float32),
                    'target_3d': np.array(vec_3d, dtype=np.float32),
                    'condition': img_flat,
                    'base_name': d3['base_name'],
                    'instance_id': inst_3d.get('instance_id', i),
                    'npy_file': d3['npy_file'],
                    'class_id': int(inst_3d.get('pred_class_id', inst_3d.get('class_id', 24))),
                })
            volume_records.append(d3)
            lines_3d.append(d3)
            lines_3dmip.append(dm)

    # Load static 2D samples first (needed for joint scaler fitting)
    static_samples = []
    with open(jsonl_2d, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            npy_path = os.path.join(data_path, d['npy_file'])
            if not os.path.isfile(npy_path):
                continue
            npy = np.load(npy_path)
            for idx, inst in enumerate(d.get('instances', [])):
                if int(inst.get('pred_class_id', inst.get('class_id', 24))) == 24:
                    continue
                vec = []
                valid = True
                for k in feature_keys:
                    v = inst.get(k)
                    if v is None:
                        valid = False
                        break
                    try:
                        vec.append(float(v))
                    except (TypeError, ValueError):
                        valid = False
                        break
                if not valid or len(vec) != len(feature_keys) or idx >= npy.shape[0]:
                    continue
                img_flat = npy[idx].flatten().astype(np.float32)
                static_samples.append({
                    'input_2d': np.array(vec, dtype=np.float32),
                    'condition': img_flat,
                    'class_id': int(inst.get('pred_class_id', inst.get('class_id', 24))),
                    'base_name': d['base_name'],
                    'npy_file': d['npy_file'],
                    'instance_id': inst.get('instance_id', idx),
                })
    if not static_samples:
        raise ValueError(f"No valid static (2D) samples in {data_path}/instances_2d.jsonl; lifter domain adversarial requires 2D data.")

    if not mip_samples:
        raise ValueError(f"No valid MIP samples in {data_path}")
    
    # Extract raw arrays (no preprocessing, direct from JSON)
    mip_2d_raw = np.array([s['input_2d'] for s in mip_samples], dtype=np.float32)
    target_3d_raw = np.array([s['target_3d'] for s in mip_samples], dtype=np.float32)
    static_2d_raw = np.array([s['input_2d'] for s in static_samples], dtype=np.float32)
    conditions = np.array([s['condition'] for s in mip_samples], dtype=np.float32)
    static_cond = np.array([s['condition'] for s in static_samples], dtype=np.float32)
    
    # Compute 3D min/max for boundary penalty (from training 3D data)
    phys_min_3d = target_3d_raw.min(axis=0)
    phys_max_3d = target_3d_raw.max(axis=0)
    
    # Independent StandardScalers for 3D, 2D MIP, 2D static
    scaler_volume = StandardScaler()
    scaler_volume.fit(target_3d_raw)
    
    scaler_mip = StandardScaler()
    scaler_mip.fit(mip_2d_raw)
    
    scaler_2d = StandardScaler()
    scaler_2d.fit(static_2d_raw)
    
    scaler_image = StandardScaler()
    all_img_cond = np.vstack([conditions, static_cond])
    scaler_image.fit(all_img_cond)
    
    # Normalize and store back
    mip_2d_norm = scaler_mip.transform(mip_2d_raw)
    target_3d_norm = scaler_volume.transform(target_3d_raw)
    conditions_norm = scaler_image.transform(conditions)
    for i, s in enumerate(mip_samples):
        s['input_2d'] = mip_2d_norm[i]
        s['target_3d'] = target_3d_norm[i]
        s['condition'] = conditions_norm[i]
    
    static_2d_norm = scaler_2d.transform(static_2d_raw)
    static_cond_norm = scaler_image.transform(static_cond)
    for i, s in enumerate(static_samples):
        s['input_2d_raw'] = static_2d_raw[i].copy()  # pre-lift raw values for UMAP
        s['input_2d'] = static_2d_norm[i]
        s['condition'] = static_cond_norm[i]

    # Print normalized data statistics (actual training ranges)
    print("\n" + "="*80)
    print("NORMALIZED DATA STATISTICS (used for training)")
    print("="*80)
    for i, feat_name in enumerate(feature_keys):
        print(f"\n[{feat_name}]")
        print(f"  3D target   (norm): min={target_3d_norm[:, i].min():.4f}, max={target_3d_norm[:, i].max():.4f}, mean={target_3d_norm[:, i].mean():.4f}, std={target_3d_norm[:, i].std():.4f}")
        print(f"  2D MIP      (norm): min={mip_2d_norm[:, i].min():.4f}, max={mip_2d_norm[:, i].max():.4f}, mean={mip_2d_norm[:, i].mean():.4f}, std={mip_2d_norm[:, i].std():.4f}")
        print(f"  2D static   (norm): min={static_2d_norm[:, i].min():.4f}, max={static_2d_norm[:, i].max():.4f}, mean={static_2d_norm[:, i].mean():.4f}, std={static_2d_norm[:, i].std():.4f}")
    print("="*80 + "\n")

    image_dim = conditions.shape[1]
    n_phys = len(feature_keys)
    print(f"prepare_data_from_folder: {data_path}")
    print(f"  MIP samples: {len(mip_samples)}, static (2D raw) samples: {len(static_samples)}, volume records: {len(volume_records)}")
    print(f"  n_phys={n_phys}, image_dim={image_dim}")
    print(f"  3D physical feature ranges (min): {phys_min_3d}")
    print(f"  3D physical feature ranges (max): {phys_max_3d}")
    return {
        'mip_samples': mip_samples,
        'static_samples': static_samples,
        'volume_records': volume_records,
        'feature_keys': feature_keys,
        'n_phys': n_phys,
        'image_dim': image_dim,
        'scaler_mip': scaler_mip,
        'scaler_volume': scaler_volume,
        'scaler_2d': scaler_2d,
        'scaler_image': scaler_image,
        'phys_min_3d': phys_min_3d,
        'phys_max_3d': phys_max_3d,
    }


def _load_jsonl_records(jsonl_path: str):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _resolve_npy_by_basename(npy_dir: str, base_name: str) -> str | None:
    """
    Resolve a .npy path in npy_dir using base_name only.

    Strategy:
    - Try a few common explicit candidates first
    - Fallback to a directory glob: f"{base_name}*.npy"
      Prefer filenames containing "optimal" if multiple exist.
    """
    if not base_name:
        return None
    npy_dir = os.path.abspath(npy_dir)
    candidates = [
        f"{base_name}.npy",
        f"{base_name}_2D.npy",
        f"{base_name}_2d.npy",
        f"{base_name}_optimal.npy",
        f"{base_name}_naive.npy",
        f"{base_name}_mip.npy",
        f"{base_name}_3DMIP.npy",
        f"{base_name}_3dmip.npy",
    ]
    for fn in candidates:
        p = os.path.join(npy_dir, fn)
        if os.path.isfile(p):
            return p

    try:
        import glob as _glob

        matches = sorted(_glob.glob(os.path.join(npy_dir, f"{base_name}*.npy")))
    except Exception:
        matches = []

    if not matches:
        return None
    # Prefer "optimal" if present, else the first stable match.
    for p in matches:
        if "optimal" in os.path.basename(p):
            return p
    return matches[0]


def _get_phy_value(inst: dict, key: str):
    """
    Read physical feature value with compatibility:
    - norm_size == norm_area (2D) == norm_volume (3D)
    """
    if key in inst:
        return inst.get(key)
    if key == "norm_size":
        if "norm_area" in inst:
            return inst.get("norm_area")
        if "norm_volume" in inst:
            return inst.get("norm_volume")
    return inst.get(key)


def prepare_data_from_paths(
    phy_3d: str,
    phy_3d_mip: str,
    phy_2d: str,
    img_2d: str,
    img_3d_mip: str,
    feature_keys,
):
    """
    Same output schema as prepare_data_from_folder(), but with separately specified paths.

    Alignment:
    - Use base_name to align between JSONLs
    - Use per-record instance order to align phy features with npy rows
    """
    feature_keys = list(feature_keys)

    phy_3d = os.path.abspath(phy_3d)
    phy_3d_mip = os.path.abspath(phy_3d_mip)
    phy_2d = os.path.abspath(phy_2d)
    img_2d = os.path.abspath(img_2d)
    img_3d_mip = os.path.abspath(img_3d_mip)

    if not os.path.isfile(phy_3d):
        raise FileNotFoundError(f"phy_3d not found: {phy_3d}")
    if not os.path.isfile(phy_3d_mip):
        raise FileNotFoundError(f"phy_3d_mip not found: {phy_3d_mip}")
    if not os.path.isfile(phy_2d):
        raise FileNotFoundError(f"phy_2d not found: {phy_2d}")
    if not os.path.isdir(img_2d):
        raise FileNotFoundError(f"img_2d not found (dir): {img_2d}")
    if not os.path.isdir(img_3d_mip):
        raise FileNotFoundError(f"img_3d_mip not found (dir): {img_3d_mip}")

    # ---------------------------
    # MIP + 3D (noid) samples
    # ---------------------------
    lines_3d = _load_jsonl_records(phy_3d)
    lines_3dmip = _load_jsonl_records(phy_3d_mip)
    map_3d = {d.get("base_name"): d for d in lines_3d if d.get("base_name") is not None}
    map_mip = {d.get("base_name"): d for d in lines_3dmip if d.get("base_name") is not None}
    keys_common = sorted(set(map_3d.keys()) & set(map_mip.keys()))

    mip_samples = []
    volume_records = []

    for base_name in keys_common:
        d3 = map_3d[base_name]
        dm = map_mip[base_name]
        # Keep 3D/3D-MIP alignment by base_name regardless of NPY/instance alignment.
        volume_records.append(d3)

        npy_path = _resolve_npy_by_basename(img_3d_mip, base_name)
        if npy_path is None or (not os.path.isfile(npy_path)):
            continue
        npy = np.load(npy_path)  # (N, 2, C) expected
        insts_3d = d3.get("instances", [])
        insts_mip = dm.get("instances", [])
        # Remove strict instance↔npy alignment. Use the common prefix length.
        n_common = min(len(insts_3d), len(insts_mip), int(npy.shape[0]))
        if n_common <= 0:
            continue

        for i in range(n_common):
            inst_3d = insts_3d[i]
            inst_mip = insts_mip[i]
            vec_2d = []
            vec_3d = []
            valid = True
            for k in feature_keys:
                v2 = _get_phy_value(inst_mip, k)
                v3 = _get_phy_value(inst_3d, k)
                if v2 is None or v3 is None:
                    valid = False
                    break
                try:
                    vec_2d.append(float(v2))
                    vec_3d.append(float(v3))
                except (TypeError, ValueError):
                    valid = False
                    break
            if not valid or len(vec_2d) != len(feature_keys):
                continue
            img_feat = npy[i]  # (2, C)
            img_flat = img_feat.flatten().astype(np.float32)
            mip_samples.append(
                {
                    "input_2d": np.array(vec_2d, dtype=np.float32),
                    "target_3d": np.array(vec_3d, dtype=np.float32),
                    "condition": img_flat,
                    "base_name": d3.get("base_name"),
                    "instance_id": inst_3d.get("instance_id", i),
                    "npy_file": os.path.basename(npy_path),
                    "class_id": int(inst_3d.get("pred_class_id", inst_3d.get("class_id", 24))),
                }
            )

    # ---------------------------
    # Static 2D (id) samples
    # ---------------------------
    static_samples = []
    lines_2d = _load_jsonl_records(phy_2d)
    for d in lines_2d:
        base_name = d.get("base_name")
        npy_path = _resolve_npy_by_basename(img_2d, base_name)
        if npy_path is None or (not os.path.isfile(npy_path)):
            continue
        npy = np.load(npy_path)
        for idx, inst in enumerate(d.get("instances", [])):
            if int(inst.get("pred_class_id", inst.get("class_id", 24))) == 24:
                continue
            vec = []
            valid = True
            for k in feature_keys:
                v = _get_phy_value(inst, k)
                if v is None:
                    valid = False
                    break
                try:
                    vec.append(float(v))
                except (TypeError, ValueError):
                    valid = False
                    break
            if not valid or len(vec) != len(feature_keys) or idx >= npy.shape[0]:
                continue
            img_flat = npy[idx].flatten().astype(np.float32)
            static_samples.append(
                {
                    "input_2d": np.array(vec, dtype=np.float32),
                    "condition": img_flat,
                    "class_id": int(inst.get("pred_class_id", inst.get("class_id", 24))),
                    "base_name": d.get("base_name"),
                    "npy_file": os.path.basename(npy_path),
                    "instance_id": inst.get("instance_id", idx),
                }
            )

    if not static_samples:
        raise ValueError("No valid static (2D) samples; lifter domain adversarial requires 2D data.")
    if not mip_samples:
        raise ValueError("No valid MIP samples.")

    # ---------------------------
    # Normalization (same scheme)
    # ---------------------------
    mip_2d_raw = np.array([s["input_2d"] for s in mip_samples], dtype=np.float32)
    target_3d_raw = np.array([s["target_3d"] for s in mip_samples], dtype=np.float32)
    static_2d_raw = np.array([s["input_2d"] for s in static_samples], dtype=np.float32)
    conditions = np.array([s["condition"] for s in mip_samples], dtype=np.float32)
    static_cond = np.array([s["condition"] for s in static_samples], dtype=np.float32)

    phys_min_3d = target_3d_raw.min(axis=0)
    phys_max_3d = target_3d_raw.max(axis=0)

    scaler_volume = StandardScaler()
    scaler_volume.fit(target_3d_raw)
    scaler_mip = StandardScaler()
    scaler_mip.fit(mip_2d_raw)
    scaler_2d = StandardScaler()
    scaler_2d.fit(static_2d_raw)
    scaler_image = StandardScaler()
    scaler_image.fit(np.vstack([conditions, static_cond]))

    mip_2d_norm = scaler_mip.transform(mip_2d_raw)
    target_3d_norm = scaler_volume.transform(target_3d_raw)
    conditions_norm = scaler_image.transform(conditions)
    for i, s in enumerate(mip_samples):
        s["input_2d"] = mip_2d_norm[i]
        s["target_3d"] = target_3d_norm[i]
        s["condition"] = conditions_norm[i]

    static_2d_norm = scaler_2d.transform(static_2d_raw)
    static_cond_norm = scaler_image.transform(static_cond)
    for i, s in enumerate(static_samples):
        s["input_2d_raw"] = static_2d_raw[i].copy()
        s["input_2d"] = static_2d_norm[i]
        s["condition"] = static_cond_norm[i]

    image_dim = conditions.shape[1]
    n_phys = len(feature_keys)
    print("prepare_data_from_paths:")
    print(f"  mip_samples={len(mip_samples)}, static_samples={len(static_samples)}, volume_records={len(volume_records)}")
    print(f"  n_phys={n_phys}, image_dim={image_dim}")
    return {
        "mip_samples": mip_samples,
        "static_samples": static_samples,
        "volume_records": volume_records,
        "feature_keys": feature_keys,
        "n_phys": n_phys,
        "image_dim": image_dim,
        "scaler_mip": scaler_mip,
        "scaler_volume": scaler_volume,
        "scaler_2d": scaler_2d,
        "scaler_image": scaler_image,
        "phys_min_3d": phys_min_3d,
        "phys_max_3d": phys_max_3d,
    }


class LifterDatasetFromFolder(Dataset):
    """Dataset for lifter from prepare_data_from_folder output. MIP only (input_2d, target_3d, condition)."""
    def __init__(self, mip_samples, static_samples=None):
        self.mip_samples = mip_samples
        self.static_samples = static_samples or []

    def __len__(self):
        return len(self.mip_samples) + len(self.static_samples)

    def __getitem__(self, idx):
        if idx < len(self.mip_samples):
            s = self.mip_samples[idx]
            return {
                'input_2d': torch.tensor(s['input_2d'], dtype=torch.float32),
                'target_3d': torch.tensor(s['target_3d'], dtype=torch.float32),
                'condition': torch.tensor(s['condition'], dtype=torch.float32),
                'has_target': True,
            }
        s = self.static_samples[idx - len(self.mip_samples)]
        return {
            'input_2d': torch.tensor(s['input_2d'], dtype=torch.float32),
            'target_3d': torch.zeros_like(torch.tensor(s['input_2d'])),  # dummy
            'condition': torch.tensor(s['condition'], dtype=torch.float32),
            'has_target': False,
        }


class MIPVolumeLifterDataset(Dataset):
    """
    Dataset for training lifter from folder data: yields (input_2d, target_3d, condition) for MIP;
    optional static (2D raw) yields (input_2d, condition, label).
    Built from prepare_data_from_folder() result.
    """
    def __init__(self, data_dict, mode='mip'):
        """
        data_dict: from prepare_data_from_folder()
        mode: 'mip' = only MIP samples, 'static' = only 2D raw samples, 'all' = MIP + static
        """
        self.mip_samples = data_dict['mip_samples']
        self.static_samples = data_dict.get('static_samples', [])
        self.mode = mode
        if mode == 'mip':
            self.samples = self.mip_samples
        elif mode == 'static':
            self.samples = self.static_samples
        else:
            self.samples = self.mip_samples + self.static_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        has_target = 'target_3d' in s
        out = {
            'input_2d': torch.tensor(s['input_2d'], dtype=torch.float32),
            'target_3d': torch.tensor(s['target_3d'], dtype=torch.float32) if has_target else torch.zeros(len(s['input_2d']), dtype=torch.float32),
            'condition': torch.tensor(s['condition'], dtype=torch.float32),
            'has_target': has_target,
        }
        if 'class_id' in s:
            out['label'] = int(s['class_id'])
        return out


def self_distill_align_loss(student_out, teacher_out, loss_type='l2'):
    """
    Alignment loss for self-distillation: student (no cond) chases teacher (with cond, detached).
    student_out, teacher_out: same shape (B, D) or (B, C, H, W). Teacher is assumed detached by caller.
    loss_type: 'l2' (MSE) or 'cosine'.
    """
    if loss_type == 'l2':
        return nn.functional.mse_loss(student_out, teacher_out)
    if loss_type == 'cosine':
        flat_s = student_out.flatten(1)
        flat_t = teacher_out.flatten(1)
        flat_s = nn.functional.normalize(flat_s, dim=1)
        flat_t = nn.functional.normalize(flat_t, dim=1)
        cos_sim = (flat_s * flat_t).sum(dim=1).mean()
        return 1.0 - cos_sim
    raise ValueError(f"Unknown loss_type: {loss_type}")


def train_lifter_epoch(model, dataloader, optimizer, device, epoch, 
                       lambda_self_distill_lifter=0.0,
                       self_distill_lifter_type='l2'):
    """
    Train one epoch of the lifter model using flow matching.
    
    Args:
        model: FeatureFlowMatchingLifter model
        dataloader: DataLoader
        optimizer: Optimizer
        device: torch device
        epoch: current epoch number
        lambda_self_distill_lifter: weight for self-distill (no-cond chases cond). 0 to disable.
        self_distill_lifter_type: 'l2' or 'cosine' for align loss
    """
    if not FLOW_MATCHING_AVAILABLE:
        raise ImportError("flow_matching library is not available")
    
    model.train()
    path = CondOTProbPath()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Lifter Epoch {epoch+1}")
    for batch in pbar:
        optimizer.zero_grad()
        input_2d = batch['input_2d'].to(device)
        target_3d = batch['target_3d'].to(device)
        condition = batch['condition'].to(device)  # image [B, 2*C]
        has_target = batch.get('has_target', torch.ones(input_2d.size(0), dtype=torch.bool))
        if isinstance(has_target, torch.Tensor):
            has_target = has_target.to(device)
        else:
            has_target = torch.tensor(has_target, device=device, dtype=torch.bool)
        B = input_2d.size(0)
        if not has_target.any():
            num_batches += 1
            continue
        conditioning = {"condition": condition}
        # Flow: x_0 = input_2d, x_1 = target_3d, x_t = (1-t)*x_0 + t*x_1, u_t = x_1 - x_0
        t = torch.rand(B, device=device)
        x_t = (1 - t).unsqueeze(1) * input_2d + t.unsqueeze(1) * target_3d
        u_t = target_3d - input_2d
        with torch.cuda.amp.autocast():
            u_pred = model(x_t, t, extra=conditioning)
            loss_all = torch.pow(u_pred - u_t, 2).mean(dim=1)
            loss = loss_all[has_target].mean()
            align_loss_item = 0.0
            # Self-distill: no-cond output chases cond output (teacher detached)
            if lambda_self_distill_lifter > 0:
                u_pred_no_cond = model(x_t, t, extra={"condition": None})
                align_loss = self_distill_align_loss(u_pred_no_cond, u_pred.detach(), self_distill_lifter_type)
                loss = loss + lambda_self_distill_lifter * align_loss
                align_loss_item = align_loss.item()
        if loss.numel() == 0 or not torch.isfinite(loss):
            num_batches += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        postfix = {'loss': f'{loss.item():.4f}'}
        if lambda_self_distill_lifter > 0:
            postfix['sd'] = f'{align_loss_item:.4f}'
        pbar.set_postfix(postfix)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_lifter_epoch_with_domain(
    model,
    live_dataloader,
    static_dataloader,
    domain_disc,
    optimizer,
    optimizer_domain,
    device,
    epoch,
    lifter_infer_num_steps=50,
    lambda_domain=1.0,
    lambda_boundary=1.0,
    grl_lambda=1.0,
    lambda_self_distill_lifter=0.0,
    self_distill_lifter_type='l2',
    phys_min_3d=None,
    phys_max_3d=None,
    scaler_mip=None,
    scaler_volume=None,
):
    """
    Train one lifter epoch: flow (x_0=2d, x_1=3d, condition=image) + three-way domain adversarial
    + boundary penalty loss.
    Live batch: input_2d, target_3d, condition (image). Static batch: input_2d, condition, label.
    
    New: Boundary penalty loss penalizes lifted 2D/MIP features exceeding 3D training data min/max
    (computed in denormalized space for physical meaning).
    """
    if not FLOW_MATCHING_AVAILABLE:
        raise ImportError("flow_matching library is not available")
    model.train()
    domain_disc.train()
    ce_domain_fn = nn.CrossEntropyLoss()
    total_flow = 0.0
    total_domain = 0.0
    total_boundary = 0.0
    num_batches = 0
    
    # Prepare boundary penalty tensors (3D min/max in denormalized space)
    use_boundary_penalty = (phys_min_3d is not None and phys_max_3d is not None and 
                           scaler_volume is not None)
    if use_boundary_penalty:
        phys_min_tensor = torch.tensor(phys_min_3d, dtype=torch.float32, device=device)
        phys_max_tensor = torch.tensor(phys_max_3d, dtype=torch.float32, device=device)
    
    static_iter = iter(static_dataloader)
    pbar = tqdm(live_dataloader, desc=f"Lifter+Domain Epoch {epoch+1}")
    for batch in pbar:
        try:
            static_batch = next(static_iter)
        except StopIteration:
            static_iter = iter(static_dataloader)
            static_batch = next(static_iter)
        input_2d = batch['input_2d'].to(device)
        target_3d = batch['target_3d'].to(device)
        condition = batch['condition'].to(device)
        has_target = batch.get('has_target', torch.ones(input_2d.size(0), dtype=torch.bool))
        if isinstance(has_target, torch.Tensor):
            has_target = has_target.to(device)
        else:
            has_target = torch.tensor(has_target, device=device, dtype=torch.bool)
        B_live = input_2d.size(0)
        conditioning = {"condition": condition}
        # Flow loss: x_0 = input_2d, x_1 = target_3d
        t = torch.rand(B_live, device=device)
        x_t = (1 - t).unsqueeze(1) * input_2d + t.unsqueeze(1) * target_3d
        u_t = target_3d - input_2d
        u_pred = model(x_t, t, extra=conditioning)
        flow_loss_all = torch.pow(u_pred - u_t, 2).mean(dim=1)
        flow_loss = flow_loss_all[has_target].mean() if has_target.any() else flow_loss_all.mean()
        align_loss_item = 0.0
        if lambda_self_distill_lifter > 0:
            u_pred_no_cond = model(x_t, t, extra={"condition": None})
            align_loss = self_distill_align_loss(u_pred_no_cond, u_pred.detach(), self_distill_lifter_type)
            flow_loss = flow_loss + lambda_self_distill_lifter * align_loss
            align_loss_item = align_loss.item()
        lifted_mip = lifter_lift_torch(
            model, input_2d, condition, num_steps=lifter_infer_num_steps
        )
        static_input_2d = static_batch['input_2d'].to(device)
        static_condition = static_batch['condition'].to(device)
        static_labels = static_batch['label']
        if isinstance(static_labels, torch.Tensor):
            static_labels = static_labels.to(device).long()
        else:
            static_labels = torch.tensor(static_labels, device=device, dtype=torch.long)
        lifted_2d = lifter_lift_torch(
            model, static_input_2d, static_condition, num_steps=lifter_infer_num_steps
        )
        B_static = lifted_2d.size(0)
        
        # Boundary penalty: penalize lifted features exceeding 3D training data min/max
        boundary_loss = torch.tensor(0.0, device=device)
        if use_boundary_penalty:
            # Denormalize lifted_mip and lifted_2d to original space
            lifted_mip_np = lifted_mip.detach().cpu().numpy()
            lifted_2d_np = lifted_2d.detach().cpu().numpy()
            lifted_mip_denorm = torch.tensor(
                scaler_volume.inverse_transform(lifted_mip_np), dtype=torch.float32, device=device
            )
            lifted_2d_denorm = torch.tensor(
                scaler_volume.inverse_transform(lifted_2d_np), dtype=torch.float32, device=device
            )
            # Linear penalty for exceeding 3D min/max (mild, stable; avoids training collapse)
            lower_violation_mip = torch.relu(phys_min_tensor - lifted_mip_denorm)
            upper_violation_mip = torch.relu(lifted_mip_denorm - phys_max_tensor)
            lower_violation_2d = torch.relu(phys_min_tensor - lifted_2d_denorm)
            upper_violation_2d = torch.relu(lifted_2d_denorm - phys_max_tensor)
            
            penalty_mip = lower_violation_mip + upper_violation_mip
            penalty_2d = lower_violation_2d + upper_violation_2d
            boundary_loss = penalty_mip.mean() + penalty_2d.mean()
        
        feat_domain = torch.cat([lifted_mip, lifted_2d, target_3d], dim=0)
        domain_labels = torch.cat([
            torch.zeros(B_live, dtype=torch.long, device=device),
            torch.ones(B_static, dtype=torch.long, device=device),
            2 * torch.ones(B_live, dtype=torch.long, device=device),
        ], dim=0)
        domain_logits = domain_disc(grad_reverse(feat_domain, lambda_grl=grl_lambda))
        domain_loss = ce_domain_fn(domain_logits, domain_labels)
        loss = flow_loss + lambda_domain * domain_loss + lambda_boundary * boundary_loss
        optimizer.zero_grad()
        optimizer_domain.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer_domain.step()
        total_flow += flow_loss.item()
        total_domain += domain_loss.item()
        total_boundary += boundary_loss.item()
        num_batches += 1
        postfix = {'flow': f'{flow_loss.item():.4f}', 'dom': f'{domain_loss.item():.4f}', 'bound': f'{boundary_loss.item():.4f}'}
        if lambda_self_distill_lifter > 0:
            postfix['sd'] = f'{align_loss_item:.4f}'
        pbar.set_postfix(postfix)
    return (
        total_flow / num_batches if num_batches else 0.0,
        total_domain / num_batches if num_batches else 0.0,
        total_boundary / num_batches if num_batches else 0.0,
    )


def train_lifter(data_dict, output_dir, epochs=100, batch_size=512, lr=1e-4,
                 gpu_id=None, samples_per_group=32,
                 lifter_infer_num_steps=50,
                 lambda_domain=0.1, lambda_boundary=1.0,
                 lambda_self_distill_lifter=0.0, self_distill_lifter_type='l2',
                 grl_lambda=1.0,
                 eval_epochs=0, feature_keys=None,
                 umap_scaler_path=None, umap_reducer_path=None):
    """
    Train lifter from data_dict (from prepare_data_from_folder).
    If static_samples present, uses three-way domain adversarial + boundary penalty.
    """
    if not FLOW_MATCHING_AVAILABLE:
        raise ImportError("flow_matching library is not available")
    n_phys = data_dict['n_phys']
    image_dim = data_dict['image_dim']
    scaler_mip = data_dict['scaler_mip']
    scaler_volume = data_dict['scaler_volume']
    scaler_image = data_dict['scaler_image']
    phys_min_3d = data_dict.get('phys_min_3d')
    phys_max_3d = data_dict.get('phys_max_3d')
    print("="*80)
    use_domain = len(data_dict.get('static_samples', [])) > 0
    print("TRAINING LIFTER (2D->3D, condition=image)" + (" + 3-WAY DOMAIN + TRIPLET" if use_domain else ""))
    print("="*80)
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}' if gpu_id < torch.cuda.device_count() else 'cuda:0')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    live_dataset = MIPVolumeLifterDataset(data_dict, mode='mip')
    live_dataloader = DataLoader(
        live_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    model = FeatureFlowMatchingLifter(
        input_dim=n_phys,
        target_dim=n_phys,
        condition_dim=image_dim,
        hidden_dim=256,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    static_dataloader = None
    domain_disc = None
    optimizer_domain = None
    if use_domain:
        static_dataset = MIPVolumeLifterDataset(data_dict, mode='static')
        batch_indices = create_grouped_batch_sampler_from_labels(
            [s['class_id'] for s in data_dict['static_samples']], samples_per_group
        )
        static_dataloader = DataLoader(
            static_dataset,
            batch_sampler=batch_indices,
            pin_memory=True,
        )
        domain_disc = DomainDiscriminator(input_dim=n_phys, num_domains=3).to(device)
        optimizer_domain = optim.AdamW(domain_disc.parameters(), lr=lr, betas=(0.9, 0.999))
        print(f"Three-way domain adversarial: static batches={len(batch_indices)}")
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        if use_domain:
            flow_loss, dom_loss, bound_loss = train_lifter_epoch_with_domain(
                model, live_dataloader, static_dataloader, domain_disc,
                optimizer, optimizer_domain, device, epoch,
                lifter_infer_num_steps=lifter_infer_num_steps,
                lambda_domain=lambda_domain, lambda_boundary=lambda_boundary,
                grl_lambda=grl_lambda,
                lambda_self_distill_lifter=lambda_self_distill_lifter,
                self_distill_lifter_type=self_distill_lifter_type,
                phys_min_3d=phys_min_3d,
                phys_max_3d=phys_max_3d,
                scaler_mip=scaler_mip,
                scaler_volume=scaler_volume,
            )
            print(f"Epoch {epoch+1}/{epochs}: Flow = {flow_loss:.6f}, Domain = {dom_loss:.6f}, Boundary = {bound_loss:.6f}")
        else:
            loss = train_lifter_epoch(model, live_dataloader, optimizer, device, epoch,
                                     lambda_self_distill_lifter=lambda_self_distill_lifter,
                                     self_distill_lifter_type=self_distill_lifter_type)
            print(f"Epoch {epoch+1}/{epochs}: Loss = {loss:.6f}")
        if eval_epochs > 0 and feature_keys is not None and (epoch + 1) % eval_epochs == 0:
            epoch_dir = os.path.join(output_dir, f'eval_epoch_{epoch+1}')
            run_eval_lifter_epoch(
                model, data_dict, epoch_dir, device, feature_keys,
                lifter_infer_num_steps=lifter_infer_num_steps, lifter_batch_size=batch_size,
                umap_scaler_path=umap_scaler_path, umap_reducer_path=umap_reducer_path,
            )
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_mip': scaler_mip,
                'scaler_volume': scaler_volume,
                'scaler_image': scaler_image,
            }
            torch.save(checkpoint, os.path.join(output_dir, f'lifter_checkpoint_epoch_{epoch+1}.pth'))
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_mip': scaler_mip,
        'scaler_volume': scaler_volume,
        'scaler_image': scaler_image,
    }
    torch.save(final_checkpoint, os.path.join(output_dir, 'lifter_final.pth'))
    print(f"\nFinal lifter model saved to {os.path.join(output_dir, 'lifter_final.pth')}")
    return model, scaler_mip, scaler_volume, data_dict


def lift_features_with_lifter(
    lifter_model,
    scaler_mip,
    scaler_volume,
    mip_features,
    device,
    num_steps=50,
    batch_size=8192,
    show_progress=False,
    phys_min_3d=None,
    phys_max_3d=None,
):
    """
    Use trained lifter to convert MIP features to 3D features.
    
    New behavior:
    - No fractal dimension preprocessing
    - No ODE constraints during integration
    - Clamp to [phys_min_3d, phys_max_3d] after denormalization
    
    Args:
        lifter_model: trained FeatureFlowMatchingLifter model
        scaler_mip: MIP feature scaler (StandardScaler)
        scaler_volume: Volume feature scaler (StandardScaler)
        mip_features: numpy array of MIP features [N, D] (unnormalized)
        device: torch device
        num_steps: number of ODE integration steps
        batch_size: batch size for inference
        show_progress: whether to show progress bar
        phys_min_3d: physical lower bounds for clamping [D] (optional)
        phys_max_3d: physical upper bounds for clamping [D] (optional)
    
    Returns:
        lifted_features: numpy array of lifted 3D features [N, D] (unnormalized)
    """
    if not FLOW_MATCHING_AVAILABLE:
        raise ImportError("flow_matching library is not available")
    
    lifter_model.eval()
    
    # Direct normalization (no fractal preprocessing)
    mip_features_norm = scaler_mip.transform(mip_features)

    n = mip_features_norm.shape[0]
    if n == 0:
        return np.zeros((0, mip_features_norm.shape[1]), dtype=np.float32)

    lifted_out = np.zeros((n, mip_features_norm.shape[1]), dtype=np.float32)

    total_batches = (n + batch_size - 1) // batch_size
    batch_iter = range(total_batches)
    if show_progress:
        batch_iter = tqdm(batch_iter, desc="Lifting batches", leave=False)

    dt = 1.0 / num_steps

    with torch.no_grad():
        for b in batch_iter:
            start = b * batch_size
            end = min((b + 1) * batch_size, n)

            mip_tensor = torch.tensor(mip_features_norm[start:end], dtype=torch.float32, device=device)

            # Initialize from noise in target space
            x = torch.randn_like(mip_tensor)

            # ODE integration (Euler method) without constraints
            for i in range(num_steps):
                t = torch.full((mip_tensor.shape[0],), i * dt, device=device, dtype=torch.float32)
                conditioning = {"condition": mip_tensor}
                u = lifter_model(x, t, extra=conditioning)
                x = x + u * dt

            # Denormalize directly (no intermediate constraints)
            lifted_features_norm = x.cpu().numpy()
            lifted_features = scaler_volume.inverse_transform(lifted_features_norm)

            # Post-inference clamping to physical bounds
            if phys_min_3d is not None and phys_max_3d is not None:
                lifted_features = np.clip(lifted_features, phys_min_3d, phys_max_3d)

            lifted_out[start:end] = lifted_features.astype(np.float32)

    return lifted_out


def lift_batch_with_condition(lifter_model, input_2d_norm, condition_norm, scaler_volume, device,
                              num_steps=50, phys_min_3d=None, phys_max_3d=None):
    """
    Lift a batch (input_2d_norm [N,D], condition_norm [N,2*C]) to 3D, return [N,D] in original scale.
    No fractal processing, no ODE constraints, clamp to physical bounds after denormalization.
    """
    lifter_model.eval()
    with torch.no_grad():
        x = lifter_lift_torch(
            lifter_model,
            torch.tensor(input_2d_norm, dtype=torch.float32, device=device),
            torch.tensor(condition_norm, dtype=torch.float32, device=device),
            num_steps=num_steps,
        )
    lifted_norm = x.cpu().numpy()
    lifted = scaler_volume.inverse_transform(lifted_norm)
    if phys_min_3d is not None and phys_max_3d is not None:
        lifted = np.clip(lifted, phys_min_3d, phys_max_3d)
    return lifted.astype(np.float32)


def run_full_lift_and_save_from_data_dict(
    lifter_model,
    data_dict,
    output_dir,
    device,
    num_steps=50,
    batch_size=8192,
):
    """
    Lift all MIP and static samples from data_dict (with condition), save lifted_2d.json,
    lifted_mip.json, volume_3d.json. Format: list of {base_name, npy_file, instances: [{instance_id, class_id?, feat1, ...}]}.
    No fractal processing, no ODE constraints, clamp to physical bounds after inference.
    """
    lifter_model.eval()
    feature_keys = data_dict['feature_keys']
    scaler_volume = data_dict['scaler_volume']
    phys_min_3d = data_dict.get('phys_min_3d')
    phys_max_3d = data_dict.get('phys_max_3d')

    # ----- (a) Lift MIP -> lifted_mip.json -----
    mip_samples = data_dict['mip_samples']
    lifted_mip_by_key = {}  # (base_name, npy_file) -> list of {instance_id, feat_dict}
    
    # Batch processing for MIP samples
    n_mip = len(mip_samples)
    for batch_start in tqdm(range(0, n_mip, batch_size), desc="Lift MIP"):
        batch_end = min(batch_start + batch_size, n_mip)
        batch_samples = mip_samples[batch_start:batch_end]
        
        # Prepare batch tensors
        batch_input_2d = np.array([s['input_2d'] for s in batch_samples], dtype=np.float32)
        batch_cond = np.array([s['condition'] for s in batch_samples], dtype=np.float32)
        
        # Lift batch
        lifted_batch = lift_batch_with_condition(
            lifter_model, batch_input_2d, batch_cond, scaler_volume, device,
            num_steps=num_steps, phys_min_3d=phys_min_3d, phys_max_3d=phys_max_3d,
        )
        
        # Store results
        for idx, s in enumerate(batch_samples):
            key = (s['base_name'], s['npy_file'])
            if key not in lifted_mip_by_key:
                lifted_mip_by_key[key] = []
            inst = {'instance_id': s['instance_id'], 'class_id': s.get('class_id', 24)}
            for i, fn in enumerate(feature_keys):
                inst[fn] = float(lifted_batch[idx, i])
            lifted_mip_by_key[key].append(inst)
    
    lifted_mip_list = [{'base_name': k[0], 'npy_file': k[1], 'instances': v} for k, v in lifted_mip_by_key.items()]
    lifted_mip_path = os.path.join(output_dir, 'lifted_mip.json')
    with open(lifted_mip_path, 'w') as f:
        json.dump(lifted_mip_list, f, indent=2)
    print(f"Saved lifted_mip.json: {len(lifted_mip_list)} samples, {sum(len(x['instances']) for x in lifted_mip_list)} instances")

    # ----- (b) Lift static (2D raw) -> lifted_2d.json -----
    static_samples = data_dict.get('static_samples', [])
    if static_samples:
        lifted_2d_by_key = {}
        
        # Batch processing for static samples
        n_static = len(static_samples)
        for batch_start in tqdm(range(0, n_static, batch_size), desc="Lift 2D"):
            batch_end = min(batch_start + batch_size, n_static)
            batch_samples = static_samples[batch_start:batch_end]
            
            # Prepare batch tensors
            batch_input_2d = np.array([s['input_2d'] for s in batch_samples], dtype=np.float32)
            batch_cond = np.array([s['condition'] for s in batch_samples], dtype=np.float32)
            
            # Lift batch
            lifted_batch = lift_batch_with_condition(
                lifter_model, batch_input_2d, batch_cond, scaler_volume, device,
                num_steps=num_steps, phys_min_3d=phys_min_3d, phys_max_3d=phys_max_3d,
            )
            
            # Store results
            for idx, s in enumerate(batch_samples):
                key = (s['base_name'], s['npy_file'])
                if key not in lifted_2d_by_key:
                    lifted_2d_by_key[key] = []
                inst = {'instance_id': s['instance_id'], 'class_id': s['class_id']}
                for i, fn in enumerate(feature_keys):
                    inst[fn] = float(lifted_batch[idx, i])
                lifted_2d_by_key[key].append(inst)
        
        lifted_2d_list = [{'base_name': k[0], 'npy_file': k[1], 'instances': v} for k, v in lifted_2d_by_key.items()]
    else:
        lifted_2d_list = []
    lifted_2d_path = os.path.join(output_dir, 'lifted_2d.json')
    with open(lifted_2d_path, 'w') as f:
        json.dump(lifted_2d_list, f, indent=2)
    print(f"Saved lifted_2d.json: {len(lifted_2d_list)} samples, {sum(len(x['instances']) for x in lifted_2d_list)} instances")

    # ----- (c) volume_3d.json from volume_records -----
    volume_3d_path = os.path.join(output_dir, 'volume_3d.json')
    with open(volume_3d_path, 'w') as f:
        json.dump(data_dict['volume_records'], f, indent=2)
    print(f"Saved volume_3d.json: {len(data_dict['volume_records'])} records")


def _set_square_axes_box(ax):
    """Force the axes box (inner plotting area) to be 1:1 square."""
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect(1)
    else:
        ax.set_aspect('equal', adjustable='box')


def _save_figure_constrained(fig, out_path, dpi=150, max_longest_side=2048, max_file_size_mb=3.0):
    """Save figure so longest side <= max_longest_side (pixels) and file size <= max_file_size_mb."""
    w_inch, h_inch = fig.get_size_inches()
    max_inch = max(w_inch, h_inch)
    dpi_use = min(dpi, int(max_longest_side / max_inch)) if max_inch > 0 else dpi
    dpi_use = max(1, dpi_use)
    fig.savefig(out_path, dpi=dpi_use, bbox_inches='tight', pil_kwargs={'compress_level': 6})
    max_bytes = int(max_file_size_mb * 1024 * 1024)
    if os.path.getsize(out_path) <= max_bytes:
        return
    if not PIL_AVAILABLE:
        return
    resampling = getattr(Image, 'Resampling', Image).LANCZOS
    for _ in range(25):
        current_bytes = os.path.getsize(out_path)
        if current_bytes <= max_bytes:
            break
        with Image.open(out_path) as im:
            im = im.convert('RGB')
            w, h = im.size
        if w * h <= 0:
            break
        scale = (max_bytes / current_bytes) ** 0.45
        scale = min(scale, 0.85)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        with Image.open(out_path) as im:
            im = im.convert('RGB')
            im = im.resize((nw, nh), resampling)
            im.save(out_path, 'PNG', optimize=True)


def _plot_umap_centroids(embedding, labels, output_dir, prefix='umap_initial',
                        title_minor='Initial UMAP: Minor Group (Chromosome) Centroids',
                        title_major='Initial UMAP: Major Group Centroids',
                        point_size=8):
    """
    Plot UMAP with minor and major group centroids (same style as initial/eval).
    figsize=(12, 12), 1:1 aspect, _save_figure_constrained.
    labels: class_id or pred_id per point.
    """
    if embedding.shape[1] != 2:
        return  # only 2D UMAP for centroid plots
    all_class_ids = np.asarray(labels, dtype=np.int64)
    class_ids_ordered = get_ordered_class_ids()
    centroids = {}
    for cid in class_ids_ordered:
        if cid == 24:
            continue
        mask = all_class_ids == cid
        if mask.sum() > 0:
            centroids[cid] = np.mean(embedding[mask], axis=0)

    def to_rgb01(cid):
        c = COLOR_MAP.get(int(cid), (220, 220, 220))
        return tuple(x / 255.0 for x in c)

    # ----- Minor group centroids (figsize 12x12, 1:1 aspect) -----
    fig, ax = plt.subplots(figsize=(12, 12))
    unique_cids = np.unique(all_class_ids)
    for cid in sorted(unique_cids.tolist()):
        mask = all_class_ids == cid
        if mask.sum() <= 0:
            continue
        color = to_rgb01(cid)
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[color], label=class_id_to_name(int(cid)),
            alpha=0.3, s=point_size, edgecolors='black', linewidths=0.3, zorder=0,
        )
    y_range = float(np.ptp(embedding[:, 1])) or 1.0
    label_offset = 0.04 * y_range
    for cid, centroid in centroids.items():
        color = to_rgb01(cid)
        ax.scatter(
            centroid[0], centroid[1], c=[color], s=400, marker='*',
            edgecolors='black', linewidths=2.5, zorder=10,
        )
        ax.text(
            centroid[0], centroid[1] + label_offset, class_id_to_name(int(cid)),
            fontsize=12, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.8),
            zorder=11,
        )
    ax.set_xlabel('UMAP Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title(title_minor, fontsize=16, fontweight='bold')
    ax.grid(False)
    _set_square_axes_box(ax)
    plt.tight_layout()
    out_minor = os.path.join(output_dir, f'{prefix}_minor_centroids.png')
    _save_figure_constrained(fig, out_minor, dpi=150, max_longest_side=2048, max_file_size_mb=3.0)
    plt.close(fig)
    print(f"  Saved: {out_minor}")

    # ----- Major group centroids -----
    major_group_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    major_group_centroids = {}
    for gname in major_group_names:
        group_cids = MAJOR_GROUPS.get(gname, [])
        pts = [centroids[c] for c in group_cids if c in centroids]
        if pts:
            major_group_centroids[gname] = np.mean(pts, axis=0)

    major_labels = np.full(len(all_class_ids), -1, dtype=np.int64)
    for i in range(len(all_class_ids)):
        idx = class_id_to_major_group_index(int(all_class_ids[i]))
        if idx is not None:
            major_labels[i] = idx

    fig, ax = plt.subplots(figsize=(12, 12))
    for major_idx in sorted(np.unique(major_labels[major_labels >= 0]).tolist()):
        mask = major_labels == major_idx
        if mask.sum() <= 0:
            continue
        color = MAJOR_GROUP_INDEX_COLOR_MAP.get(int(major_idx), (0.5, 0.5, 0.5))
        gname = INDEX_TO_MAJOR_GROUP.get(int(major_idx), str(major_idx))
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[color], label=f'Group {gname}',
            alpha=0.3, s=point_size, edgecolors='black', linewidths=0.3, zorder=0,
        )
    y_range_major = float(np.ptp(embedding[:, 1])) or 1.0
    label_offset_major = 0.04 * y_range_major
    for gname, centroid in major_group_centroids.items():
        major_idx = MAJOR_GROUP_TO_INDEX.get(gname, 0)
        color = MAJOR_GROUP_INDEX_COLOR_MAP.get(major_idx, (0.5, 0.5, 0.5))
        ax.scatter(
            centroid[0], centroid[1], c=[color], s=500, marker='*',
            edgecolors='black', linewidths=3, zorder=10,
        )
        ax.text(
            centroid[0], centroid[1] + label_offset_major, f'Group {gname}',
            fontsize=14, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9),
            zorder=11,
        )
    ax.set_xlabel('UMAP Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title(title_major, fontsize=16, fontweight='bold')
    ax.grid(False)
    _set_square_axes_box(ax)
    plt.tight_layout()
    out_major = os.path.join(output_dir, f'{prefix}_major_centroids.png')
    _save_figure_constrained(fig, out_major, dpi=150, max_longest_side=2048, max_file_size_mb=2.0)
    plt.close(fig)
    print(f"  Saved: {out_major}")


def _collect_3d_vectors_from_volume_records(volume_records, feature_keys, return_class_ids=False):
    """Collect (N, n_phys) matrix and list of (base_name, npy_file, instance_id) from volume_records.
    If return_class_ids=True, also return list of class_id per row."""
    rows = []
    meta = []
    class_ids = []
    for rec in volume_records:
        for inst in rec.get('instances', []):
            vec = []
            valid = True
            for k in feature_keys:
                v = _get_phy_value(inst, k)
                if v is None:
                    valid = False
                    break
                vec.append(float(v))
            if not valid or len(vec) != len(feature_keys):
                continue
            rows.append(vec)
            meta.append((rec.get('base_name', ''), rec.get('npy_file', ''), inst.get('instance_id', len(meta))))
            class_ids.append(int(inst.get('pred_class_id', inst.get('class_id', 24))))
    arr = np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(feature_keys)), dtype=np.float32)
    if return_class_ids:
        return arr, meta, class_ids
    return arr, meta


def init_umap_from_volume_records(data_dict, output_dir, feature_keys, n_components=2, n_neighbors=30, min_dist=0.1):
    """
    Fit UMAP on original 3D vectors from volume_records; save scaler and reducer to output_dir.
    Plot initial UMAP with minor/major centroids (figsize 12x12, 1:1 aspect, constrained save).
    Used before lifter training so eval can transform lifted/pred vectors with the same embedding.
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping init_umap_from_volume_records")
        return None, None
    volume_records = data_dict.get('volume_records', [])
    if not volume_records:
        print("No volume_records for UMAP init, skipping")
        return None, None
    features_3d, _, class_ids = _collect_3d_vectors_from_volume_records(
        volume_records, feature_keys, return_class_ids=True,
    )
    if len(features_3d) == 0:
        print("No 3D vectors from volume_records, skipping UMAP init")
        return None, None
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_3d)
    reducer = umap.UMAP(n_components=n_components, metric='euclidean', random_state=42,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    reducer.fit(features_scaled)
    scaler_path = os.path.join(output_dir, 'umap_scaler.pkl')
    reducer_path = os.path.join(output_dir, 'umap_reducer.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(reducer_path, 'wb') as f:
        pickle.dump(reducer, f)
    print(f"UMAP init: fitted on {len(features_3d)} 3D vectors; saved {scaler_path}, {reducer_path}")

    # Initial UMAP visualization: centroids (same style as umap_visualization_from_json)
    embedding = reducer.transform(features_scaled)
    if embedding.shape[1] == 2:
        print("Plotting initial UMAP (minor & major centroids)...")
        _plot_umap_centroids(embedding, class_ids, output_dir, prefix='umap_initial',
                            title_minor='Initial UMAP: Minor Group (Chromosome) Centroids',
                            title_major='Initial UMAP: Major Group Centroids')
    return scaler_path, reducer_path


def _build_gallery_from_lifted_json(lifted_path, feature_keys):
    """From lifted_2d.json build (vectors, class_ids) for k-NN gallery."""
    with open(lifted_path, 'r') as f:
        data = json.load(f)
    vecs = []
    labels = []
    for rec in data:
        for inst in rec.get('instances', []):
            vec = [float(inst.get(k, 0)) for k in feature_keys]
            vecs.append(vec)
            labels.append(int(inst.get('class_id', 24)))
    return np.array(vecs, dtype=np.float32) if vecs else np.zeros((0, len(feature_keys)), dtype=np.float32), labels


def _write_pred_3d_from_knn(volume_records, feature_keys, gallery_vecs, gallery_ids, pred_3d_path, k=5,
                            scaler_volume=None):
    """
    For each instance in volume_records, get 3D vec, k-NN in gallery_vecs -> distance-weighted vote class_id.
    If scaler_volume is provided, standardize both gallery and query with it (k-NN in lifter's 3D space).
    Write pred_3d.json: same structure as volume_3d, with class_id replaced by k-NN result.
    """
    if len(gallery_vecs) == 0:
        print("Gallery empty, writing pred_3d with class_id=24 for all instances")
        pred_records = []
        for rec in volume_records:
            new_instances = [{**inst, 'class_id': 24} for inst in rec.get('instances', [])]
            pred_records.append({**rec, 'instances': new_instances})
        with open(pred_3d_path, 'w') as f:
            json.dump(pred_records, f, indent=2)
        return
    if scaler_volume is not None:
        gallery_vecs = scaler_volume.transform(gallery_vecs)
    nn = NearestNeighbors(n_neighbors=min(k, len(gallery_vecs)), metric='euclidean')
    nn.fit(gallery_vecs)
    pred_records = []
    eps = 1e-6
    for rec in volume_records:
        new_instances = []
        for inst in rec.get('instances', []):
            vec = np.array([[float(inst.get(key, 0)) for key in feature_keys]], dtype=np.float32)
            if vec.shape[1] != len(feature_keys):
                new_instances.append({**inst, 'class_id': 24})
                continue
            if scaler_volume is not None:
                vec = scaler_volume.transform(vec)
            dists, inds = nn.kneighbors(vec)
            dists = dists[0]
            inds = inds[0]
            # Distance-weighted vote: weight = 1/(d+eps), sum per class
            class_weight = {}
            for i, idx in enumerate(inds):
                cid = gallery_ids[idx]
                w = 1.0 / (dists[i] + eps)
                class_weight[cid] = class_weight.get(cid, 0.0) + w
            if not class_weight:
                pred_id = 24
            else:
                best_weight = max(class_weight.values())
                winners = [c for c, w in class_weight.items() if w == best_weight]
                if len(winners) == 1:
                    pred_id = int(winners[0])
                else:
                    pred_id = int(gallery_ids[inds[0]])
            new_instances.append({**inst, 'class_id': pred_id})
        pred_records.append({**rec, 'instances': new_instances})
    with open(pred_3d_path, 'w') as f:
        json.dump(pred_records, f, indent=2)
    print(f"Saved pred_3d.json: {len(pred_records)} records ({sum(len(r['instances']) for r in pred_records)} instances)")


def run_eval_lifter_epoch(lifter_model, data_dict, epoch_dir, device, feature_keys,
                          lifter_infer_num_steps=50, lifter_batch_size=512,
                          umap_scaler_path=None, umap_reducer_path=None):
    """
    Eval at one epoch: write lifted_2d.json, lifted_mip.json, pred_3d.json to epoch_dir;
    then UMAP: same volume 3D points as init, re-colored by KNN 寻证 pred_class_id only (no 2D/MIP on UMAP).
    """
    os.makedirs(epoch_dir, exist_ok=True)
    run_full_lift_and_save_from_data_dict(
        lifter_model, data_dict, epoch_dir, device,
        num_steps=lifter_infer_num_steps, batch_size=lifter_batch_size,
    )
    lifted_2d_path = os.path.join(epoch_dir, 'lifted_2d.json')
    lifted_mip_path = os.path.join(epoch_dir, 'lifted_mip.json')
    volume_3d_path = os.path.join(epoch_dir, 'volume_3d.json')
    pred_3d_path = os.path.join(epoch_dir, 'pred_3d.json')
    gallery_vecs, gallery_ids = _build_gallery_from_lifted_json(lifted_2d_path, feature_keys)
    scaler_volume = data_dict.get('scaler_volume')
    _write_pred_3d_from_knn(data_dict['volume_records'], feature_keys, gallery_vecs, gallery_ids, pred_3d_path,
                            scaler_volume=scaler_volume)

    if not UMAP_AVAILABLE or umap_scaler_path is None or umap_reducer_path is None:
        return
    if not os.path.isfile(umap_scaler_path) or not os.path.isfile(umap_reducer_path):
        return
    with open(umap_scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(umap_reducer_path, 'rb') as f:
        reducer = pickle.load(f)

    # UMAP: only volume 3D (same as init), colored by k-NN re-ID (class_id in pred_3d.json)
    vol_vecs, meta = _collect_3d_vectors_from_volume_records(data_dict['volume_records'], feature_keys)
    if len(vol_vecs) == 0:
        return
    with open(pred_3d_path, 'r') as f:
        pred_records = json.load(f)
    key_to_pred = {}
    for rec in pred_records:
        base = rec['base_name']
        npy = rec.get('npy_file', '')
        for inst in rec.get('instances', []):
            iid = inst.get('instance_id', -1)
            key_to_pred[(base, npy, iid)] = int(inst.get('class_id', 24))
    labels = np.array([key_to_pred.get(m, 24) for m in meta], dtype=np.int64)
    X_scaled = scaler.transform(vol_vecs)
    embedding = reducer.transform(X_scaled)
    if embedding.shape[1] == 2:
        _plot_umap_centroids(embedding, labels, epoch_dir, prefix='umap_by_pred_id',
                            title_minor='UMAP by pred id (Minor)',
                            title_major='UMAP by pred id (Major)')


def compute_group_distance(group1, group2):
    """Compute distance between two major groups (for elastic margin)"""
    # Simple distance: number of groups apart
    group_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    idx1 = group_order.index(group1) if group1 in group_order else 0
    idx2 = group_order.index(group2) if group2 in group_order else 0
    return abs(idx1 - idx2)


def create_grouped_batch_sampler(dataset, samples_per_group=4):
    """
    Create batches by sampling n samples from each minor group (class_id).
    When a class is exhausted (all its samples used in previous batches),
    randomly sample (with replacement if needed) from that class to fill the slot,
    so every batch has exactly num_classes * samples_per_group samples and all
    classes appear in every batch (training more stable).

    Args:
        dataset: ChromosomeDataset
        samples_per_group: number of samples to sample from each minor group (class_id)

    Returns:
        batches: list of batch indices (each batch contains samples from all groups)
    """
    # Group samples by minor group (class_id)
    groups_dict = {}
    for idx, label in enumerate(dataset.labels):
        minor_group = int(label)  # minor_group is the class_id itself
        if minor_group < 24:  # Exclude unknown class (24)
            if minor_group not in groups_dict:
                groups_dict[minor_group] = []
            groups_dict[minor_group].append(idx)

    # Shuffle each group
    for group_name in groups_dict:
        random.shuffle(groups_dict[group_name])

    # Create batches: every batch has all classes; exhausted classes use random fill
    batches = []
    max_iterations = max(len(indices) for indices in groups_dict.values()) // samples_per_group + 1

    for iteration in range(max_iterations):
        batch_indices = []
        for group_name, indices in groups_dict.items():
            start_idx = iteration * samples_per_group
            end_idx = start_idx + samples_per_group
            if start_idx < len(indices):
                end_idx = min(end_idx, len(indices))
                batch_indices.extend(indices[start_idx:end_idx])
            else:
                # Class exhausted: randomly sample (with replacement if needed) to fill
                n_fill = samples_per_group
                batch_indices.extend(random.choices(indices, k=n_fill))

        if len(batch_indices) > 0:
            batches.append(batch_indices)

    return batches


def create_grouped_batch_sampler_from_labels(labels, samples_per_group=4):
    """Create batches by sampling n per class from a list of labels (class_id)."""
    labels = np.asarray(labels)
    groups_dict = {}
    for idx, label in enumerate(labels):
        c = int(label)
        if c < 24:
            if c not in groups_dict:
                groups_dict[c] = []
            groups_dict[c].append(idx)
    for c in groups_dict:
        random.shuffle(groups_dict[c])
    batches = []
    max_it = max((len(v) for v in groups_dict.values()), default=0) // samples_per_group + 1
    for it in range(max_it):
        batch_indices = []
        for c, indices in groups_dict.items():
            start = it * samples_per_group
            end = min(start + samples_per_group, len(indices))
            if start < len(indices):
                batch_indices.extend(indices[start:end])
            else:
                batch_indices.extend(random.choices(indices, k=samples_per_group))
        if batch_indices:
            batches.append(batch_indices)
    return batches


def load_data_for_umap(json_paths, feature_names, max_samples=None, return_full_features=False, full_feature_names=None, sample_per_class_ratio=None):
    """
    Load data for UMAP visualization
    
    Args:
        json_paths: list of JSON file paths
        feature_names: list of feature names to extract (for UMAP)
        max_samples: maximum number of samples to load
        return_full_features: if True, also return full 8D features for prediction
        full_feature_names: list of all feature names (8D) if return_full_features=True
        sample_per_class_ratio: if provided (0.0-1.0), sample this ratio of data per class ID
    """
    all_features = []
    all_labels = []
    all_base_names = []
    all_full_features = [] if return_full_features else None
    
    for json_path in json_paths:
        if not os.path.exists(json_path):
            continue
        
        with open(json_path, 'r') as f:
            all_instances_data = json.load(f)
        
        for image_data in all_instances_data:
            base_name = image_data['base_name']
            for inst in image_data['instances']:
                class_id = int(inst['class_id'])
                if class_id == 24:  # Skip unknown
                    continue
                
                features = []
                has_invalid = False
                
                for feat_name in feature_names:
                    feat_value = _get_instance_feature_value(inst, feat_name)
                    
                    # Check if value is missing or invalid
                    if feat_value is None:
                        has_invalid = True
                        break
                    
                    # Convert to float, skip if conversion fails
                    try:
                        feat_value = float(feat_value)
                        # Check if value is finite (not inf or nan)
                        if not np.isfinite(feat_value):
                            has_invalid = True
                            break
                    except (ValueError, TypeError):
                        has_invalid = True
                        break
                    
                    features.append(feat_value)
                
                # Skip this sample if any feature is invalid
                if has_invalid or len(features) != len(feature_names):
                    continue
                
                # Also load full features if requested
                if return_full_features and full_feature_names is not None:
                    full_features = []
                    has_invalid_full = False
                    
                    for feat_name in full_feature_names:
                        feat_value = _get_instance_feature_value(inst, feat_name)
                        
                        # Check if value is missing or invalid
                        if feat_value is None:
                            has_invalid_full = True
                            break
                        
                        # Convert to float, skip if conversion fails
                        try:
                            feat_value = float(feat_value)
                            # Check if value is finite (not inf or nan)
                            if not np.isfinite(feat_value):
                                has_invalid_full = True
                                break
                        except (ValueError, TypeError):
                            has_invalid_full = True
                            break
                        
                        full_features.append(feat_value)
                    
                    # Skip this sample if any full feature is invalid
                    if has_invalid_full or len(full_features) != len(full_feature_names):
                        continue
                    
                    all_full_features.append(full_features)
                
                all_features.append(features)
                all_labels.append(class_id)
                all_base_names.append(base_name)
    
    # Sample per class if requested
    if sample_per_class_ratio is not None and 0 < sample_per_class_ratio < 1:
        # Group samples by class_id
        class_to_indices = {}
        for idx, label in enumerate(all_labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # Sample from each class
        sampled_indices = []
        for class_id, indices in class_to_indices.items():
            n_samples = max(1, int(len(indices) * sample_per_class_ratio))
            sampled_class_indices = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)
            sampled_indices.extend(sampled_class_indices.tolist())
        
        # Reorder to maintain original order (optional, but cleaner)
        sampled_indices = sorted(sampled_indices)
        
        # Apply sampling
        all_features = [all_features[i] for i in sampled_indices]
        all_labels = [all_labels[i] for i in sampled_indices]
        all_base_names = [all_base_names[i] for i in sampled_indices]
        if return_full_features and all_full_features is not None:
            all_full_features = [all_full_features[i] for i in sampled_indices]
        
        print(f"Sampled {len(sampled_indices)} samples ({sample_per_class_ratio*100:.1f}% per class) from {sum(len(v) for v in class_to_indices.values())} total samples")
    
    all_features = np.array(all_features, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    
    # Subsample if needed (after per-class sampling)
    if max_samples is not None and len(all_features) > max_samples:
        indices = np.random.choice(len(all_features), max_samples, replace=False)
        all_features = all_features[indices]
        all_labels = all_labels[indices]
        all_base_names = [all_base_names[i] for i in indices]
        if return_full_features and all_full_features is not None:
            all_full_features = [all_full_features[i] for i in indices]
    
    if return_full_features and all_full_features is not None:
        all_full_features = np.array(all_full_features, dtype=np.float32)
        return all_features, all_labels, all_base_names, all_full_features
    else:
        return all_features, all_labels, all_base_names


def fit_umap_embedding(features, n_components=2, random_state=42, n_neighbors=30, min_dist=0.1):
    """Fit UMAP and return 2D embedding. Returns None if UMAP not available or too few samples."""
    if not UMAP_AVAILABLE or len(features) < 30:
        return None
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    reducer = umap.UMAP(n_components=n_components, metric='euclidean', random_state=random_state,
                        n_neighbors=min(n_neighbors, len(features) - 1), min_dist=min_dist)
    embedding = reducer.fit_transform(features_scaled)
    return embedding.astype(np.float32)


def create_umap_visualization(features, labels, output_path, title="UMAP Visualization", 
                               feature_names=None, use_3d=False, output_dir=None, prefix=None,
                               show_feature_loading=True, save_heatmap=False,
                               original_features=None, original_feature_names=None):
    """
    Create UMAP visualization colored by class labels
    Also computes feature loading (correlation with UMAP axes) and optionally saves Pearson heatmap
    
    Args:
        features: features used for UMAP (e.g., 3D MLP output features)
        labels: class labels for coloring
        output_path: path to save the visualization
        title: title for the plot
        feature_names: names of features used for UMAP (e.g., ['dim1', 'dim2', 'dim3'])
        use_3d: whether to create 3D UMAP
        output_dir: output directory for saving additional files
        prefix: prefix for saved files
        show_feature_loading: If True, overlay feature loading arrows on UMAP plot
        save_heatmap: If True, save Pearson correlation heatmap (only once, not duplicated)
        original_features: original input features (e.g., 8D features) for additional feature loading analysis
        original_feature_names: names of original features (e.g., ['norm_size', 'eccentricity', ...])
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping visualization")
        return None, None, None
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(features.shape[1])]
    
    print(f"Computing UMAP for {len(features)} samples...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute UMAP
    n_components = 3 if use_3d else 2
    reducer = umap.UMAP(n_components=n_components, metric='euclidean', 
                       random_state=42, n_neighbors=30, min_dist=0.1)
    embedding = reducer.fit_transform(features_scaled)
    
    # Feature loading analysis: correlation between features and UMAP axes
    correlation_analysis = {}
    original_correlation_analysis = {}
    
    if SCIPY_AVAILABLE:
        print("\nAnalyzing feature correlations with UMAP axes...")
        
        # First analyze original features if provided
        if original_features is not None and original_feature_names is not None:
            print("\nAnalyzing original input features...")
            for i, feat_name in enumerate(original_feature_names):
                if i >= original_features.shape[1]:
                    continue
                feature_values = original_features[:, i]
                # Remove invalid values for correlation
                if use_3d:
                    valid_mask = (np.isfinite(feature_values) & 
                                 np.isfinite(embedding[:, 0]) & 
                                 np.isfinite(embedding[:, 1]) & 
                                 np.isfinite(embedding[:, 2]))
                else:
                    valid_mask = (np.isfinite(feature_values) & 
                                 np.isfinite(embedding[:, 0]) & 
                                 np.isfinite(embedding[:, 1]))
                
                if valid_mask.sum() < 3:
                    continue
                
                feat_vals = feature_values[valid_mask]
                emb_x = embedding[valid_mask, 0]
                emb_y = embedding[valid_mask, 1]
                
                corr_x, pval_x = pearsonr(feat_vals, emb_x)
                corr_y, pval_y = pearsonr(feat_vals, emb_y)
                
                corr_data = {
                    'correlation_with_umap_x': float(corr_x),
                    'pvalue_with_umap_x': float(pval_x),
                    'correlation_with_umap_y': float(corr_y),
                    'pvalue_with_umap_y': float(pval_y)
                }
                
                # Add Z-axis correlation for 3D UMAP
                if use_3d:
                    emb_z = embedding[valid_mask, 2]
                    corr_z, pval_z = pearsonr(feat_vals, emb_z)
                    corr_data['correlation_with_umap_z'] = float(corr_z)
                    corr_data['pvalue_with_umap_z'] = float(pval_z)
                    print(f"  [Original] {feat_name:15s}: UMAP-X corr={corr_x:7.4f} (p={pval_x:.2e}), "
                          f"UMAP-Y corr={corr_y:7.4f} (p={pval_y:.2e}), "
                          f"UMAP-Z corr={corr_z:7.4f} (p={pval_z:.2e})")
                else:
                    print(f"  [Original] {feat_name:15s}: UMAP-X corr={corr_x:7.4f} (p={pval_x:.2e}), "
                          f"UMAP-Y corr={corr_y:7.4f} (p={pval_y:.2e})")
                
                original_correlation_analysis[feat_name] = corr_data
        
        # Then analyze current features (e.g., 3D MLP output)
        print("\nAnalyzing current features (MLP output)...")
        for i, feat_name in enumerate(feature_names):
            feature_values = features[:, i]
            # Remove invalid values for correlation
            if use_3d:
                valid_mask = (np.isfinite(feature_values) & 
                             np.isfinite(embedding[:, 0]) & 
                             np.isfinite(embedding[:, 1]) & 
                             np.isfinite(embedding[:, 2]))
            else:
                valid_mask = (np.isfinite(feature_values) & 
                             np.isfinite(embedding[:, 0]) & 
                             np.isfinite(embedding[:, 1]))
            
            if valid_mask.sum() < 3:
                continue
            
            feat_vals = feature_values[valid_mask]
            emb_x = embedding[valid_mask, 0]
            emb_y = embedding[valid_mask, 1]
            
            corr_x, pval_x = pearsonr(feat_vals, emb_x)
            corr_y, pval_y = pearsonr(feat_vals, emb_y)
            
            corr_data = {
                'correlation_with_umap_x': float(corr_x),
                'pvalue_with_umap_x': float(pval_x),
                'correlation_with_umap_y': float(corr_y),
                'pvalue_with_umap_y': float(pval_y)
            }
            
            # Add Z-axis correlation for 3D UMAP
            if use_3d:
                emb_z = embedding[valid_mask, 2]
                corr_z, pval_z = pearsonr(feat_vals, emb_z)
                corr_data['correlation_with_umap_z'] = float(corr_z)
                corr_data['pvalue_with_umap_z'] = float(pval_z)
                print(f"  [MLP Output] {feat_name:15s}: UMAP-X corr={corr_x:7.4f} (p={pval_x:.2e}), "
                      f"UMAP-Y corr={corr_y:7.4f} (p={pval_y:.2e}), "
                      f"UMAP-Z corr={corr_z:7.4f} (p={pval_z:.2e})")
            else:
                print(f"  [MLP Output] {feat_name:15s}: UMAP-X corr={corr_x:7.4f} (p={pval_x:.2e}), "
                      f"UMAP-Y corr={corr_y:7.4f} (p={pval_y:.2e})")
            
            correlation_analysis[feat_name] = corr_data
        
        # Determine axis meanings: prefer original features, fallback to current features
        combined_analysis = {}
        if original_correlation_analysis:
            combined_analysis.update(original_correlation_analysis)
        if correlation_analysis:
            combined_analysis.update(correlation_analysis)
        
        if combined_analysis:
            print("\nInterpreting UMAP axes:")
            # Prefer original features for axis meaning
            if original_correlation_analysis:
                analysis_for_meaning = original_correlation_analysis
                print("  (Using original features for axis interpretation)")
            else:
                analysis_for_meaning = correlation_analysis
                print("  (Using MLP output features for axis interpretation)")
            
            max_corr_x_feat = max(analysis_for_meaning.items(), 
                                 key=lambda x: abs(x[1]['correlation_with_umap_x']))
            max_corr_y_feat = max(analysis_for_meaning.items(), 
                                 key=lambda x: abs(x[1]['correlation_with_umap_y']))
            
            x_axis_meaning = f"Primary: {max_corr_x_feat[0]} (r={max_corr_x_feat[1]['correlation_with_umap_x']:.3f})"
            y_axis_meaning = f"Primary: {max_corr_y_feat[0]} (r={max_corr_y_feat[1]['correlation_with_umap_y']:.3f})"
            
            if use_3d:
                max_corr_z_feat = max(analysis_for_meaning.items(), 
                                     key=lambda x: abs(x[1]['correlation_with_umap_z']))
                z_axis_meaning = f"Primary: {max_corr_z_feat[0]} (r={max_corr_z_feat[1]['correlation_with_umap_z']:.3f})"
                print(f"  UMAP Dimension 1: {x_axis_meaning}")
                print(f"  UMAP Dimension 2: {y_axis_meaning}")
                print(f"  UMAP Dimension 3: {z_axis_meaning}")
            else:
                print(f"  UMAP Dimension 1: {x_axis_meaning}")
                print(f"  UMAP Dimension 2: {y_axis_meaning}")
        else:
            x_axis_meaning = "UMAP Dimension 1"
            y_axis_meaning = "UMAP Dimension 2"
            if use_3d:
                z_axis_meaning = "UMAP Dimension 3"
    else:
        x_axis_meaning = "UMAP Dimension 1"
        y_axis_meaning = "UMAP Dimension 2"
        if use_3d:
            z_axis_meaning = "UMAP Dimension 3"
        print("Warning: scipy not available, skipping correlation analysis")
    
    # Create visualization
    if use_3d:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points colored by class
        unique_labels = np.unique(labels)
        for label in sorted(unique_labels):
            mask = labels == label
            if mask.sum() > 0:
                color = COLOR_MAP.get(label, (0.5, 0.5, 0.5))
                color_norm = tuple(c / 255.0 for c in color)
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                          c=[color_norm], label=f'Class {label}', alpha=0.6, s=20)
        
        # Set axis labels with primary component information
        if SCIPY_AVAILABLE and correlation_analysis:
            ax.set_xlabel(f'UMAP Dimension 1\n({x_axis_meaning})', fontsize=12)
            ax.set_ylabel(f'UMAP Dimension 2\n({y_axis_meaning})', fontsize=12)
            ax.set_zlabel(f'UMAP Dimension 3\n({z_axis_meaning})', fontsize=12)
        else:
            ax.set_xlabel('UMAP Dimension 1', fontsize=12)
            ax.set_ylabel('UMAP Dimension 2', fontsize=12)
            ax.set_zlabel('UMAP Dimension 3', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    else:
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot points colored by class
        unique_labels = np.unique(labels)
        for label in sorted(unique_labels):
            mask = labels == label
            if mask.sum() > 0:
                color = COLOR_MAP.get(label, (0.5, 0.5, 0.5))
                color_norm = tuple(c / 255.0 for c in color)
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=[color_norm], label=f'Class {label}', alpha=0.6, s=20,
                          edgecolors='black', linewidths=0.3)
        
        # Add feature loading arrows on UMAP plot (if requested)
        if show_feature_loading and SCIPY_AVAILABLE and correlation_analysis:
            # Calculate center point (mean of UMAP embedding)
            center_x = np.mean(embedding[:, 0])
            center_y = np.mean(embedding[:, 1])
            
            # Scale factor for arrow length (adjust based on UMAP range)
            umap_x_range = np.max(embedding[:, 0]) - np.min(embedding[:, 0])
            umap_y_range = np.max(embedding[:, 1]) - np.min(embedding[:, 1])
            scale_factor = min(umap_x_range, umap_y_range) * 0.3  # Scale to 30% of range
            
            # Draw arrows for each feature
            for feat_name, corr_data in correlation_analysis.items():
                corr_x = corr_data['correlation_with_umap_x']
                corr_y = corr_data['correlation_with_umap_y']
                
                # Arrow length represents correlation strength
                arrow_length = np.sqrt(corr_x**2 + corr_y**2) * scale_factor
                
                # Arrow direction
                dx = corr_x * scale_factor
                dy = corr_y * scale_factor
                
                # Draw arrow
                ax.arrow(center_x, center_y, dx, dy,
                        head_width=arrow_length*0.1, head_length=arrow_length*0.15,
                        fc='black', ec='black', linewidth=2, alpha=0.8)
                
                # Add label at arrow end
                label_x = center_x + dx * 1.15
                label_y = center_y + dy * 1.15
                ax.text(label_x, label_y, feat_name,
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor='black', alpha=0.8),
                       ha='center', va='center')
        
        # Update axis labels based on correlation analysis
        if show_feature_loading and SCIPY_AVAILABLE and correlation_analysis:
            ax.set_xlabel(f'UMAP Dimension 1\n({x_axis_meaning})', fontsize=14)
            ax.set_ylabel(f'UMAP Dimension 2\n({y_axis_meaning})', fontsize=14)
            # Add subtitle about arrow length
            ax.set_title(f'{title}\nArrow length represents correlation strength', 
                        fontsize=16, fontweight='bold')
        else:
            ax.set_xlabel('UMAP Dimension 1', fontsize=14)
            ax.set_ylabel('UMAP Dimension 2', fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"UMAP visualization saved to {output_path}")
    
    # Save a version without feature loading if feature loading was shown
    if not use_3d and show_feature_loading and SCIPY_AVAILABLE and correlation_analysis and output_dir is not None and prefix is not None:
        # Create pure UMAP without feature loading arrows
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot points colored by class (same as before)
        unique_labels = np.unique(labels)
        for label in sorted(unique_labels):
            mask = labels == label
            if mask.sum() > 0:
                color = COLOR_MAP.get(label, (0.5, 0.5, 0.5))
                color_norm = tuple(c / 255.0 for c in color)
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=[color_norm], label=f'Class {label}', alpha=0.6, s=20,
                          edgecolors='black', linewidths=0.3)
        
        # No feature loading arrows
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Save with "_no_loading" suffix
        pure_umap_path = output_path.replace('.png', '_no_loading.png')
        plt.savefig(pure_umap_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Pure UMAP visualization (without feature loading) saved to {pure_umap_path}")
    
    # Create Pearson heatmap only if explicitly requested (to avoid duplication)
    combined_analysis_for_heatmap = {}
    if original_correlation_analysis:
        combined_analysis_for_heatmap.update(original_correlation_analysis)
    if correlation_analysis:
        combined_analysis_for_heatmap.update(correlation_analysis)
    
    if save_heatmap and output_dir is not None and prefix is not None and combined_analysis_for_heatmap:
        # Pearson correlation heatmap
        feat_names = list(combined_analysis_for_heatmap.keys())
        # Add prefix to feature names to distinguish original vs MLP output
        feat_names_display = []
        for fname in feat_names:
            if fname in original_correlation_analysis:
                feat_names_display.append(f'[Orig] {fname}')
            else:
                feat_names_display.append(f'[MLP] {fname}')
        
        # Prepare correlation data (include Z-axis for 3D UMAP)
        if use_3d:
            corr_data = np.array([[combined_analysis_for_heatmap[f]['correlation_with_umap_x'], 
                                  combined_analysis_for_heatmap[f]['correlation_with_umap_y'],
                                  combined_analysis_for_heatmap[f]['correlation_with_umap_z']] 
                                 for f in feat_names])
            yticklabels = ['UMAP-X', 'UMAP-Y', 'UMAP-Z']
            figsize = (max(12, len(feat_names) * 0.8), 8)
        else:
            corr_data = np.array([[combined_analysis_for_heatmap[f]['correlation_with_umap_x'], 
                                  combined_analysis_for_heatmap[f]['correlation_with_umap_y']] 
                                 for f in feat_names])
            yticklabels = ['UMAP-X', 'UMAP-Y']
            figsize = (max(12, len(feat_names) * 0.8), 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        try:
            import seaborn as sns
            sns.heatmap(corr_data.T, 
                       annot=True, 
                       fmt='.3f', 
                       cmap='coolwarm', 
                       center=0,
                       xticklabels=feat_names_display,
                       yticklabels=yticklabels,
                       cbar_kws={"shrink": 0.8},
                       ax=ax)
        except ImportError:
            im = ax.imshow(corr_data.T, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(feat_names_display)))
            ax.set_yticks(np.arange(len(yticklabels)))
            ax.set_xticklabels(feat_names_display, rotation=45, ha='right')
            ax.set_yticklabels(yticklabels)
            
            for i in range(len(yticklabels)):
                for j in range(len(feat_names)):
                    text = ax.text(j, i, f'{corr_data[j, i]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        ax.set_title('Pearson Correlation Heatmap\n(Features vs UMAP Axes)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f'{prefix}_pearson_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Pearson correlation heatmap saved to {heatmap_path}")
        
        # Save correlation data to CSV
        csv_data = {
            'Feature': feat_names_display,
            'Feature_Type': ['Original' if f in original_correlation_analysis else 'MLP_Output' for f in feat_names],
            'Correlation_UMAP_X': [combined_analysis_for_heatmap[f]['correlation_with_umap_x'] for f in feat_names],
            'Pvalue_UMAP_X': [combined_analysis_for_heatmap[f]['pvalue_with_umap_x'] for f in feat_names],
            'Correlation_UMAP_Y': [combined_analysis_for_heatmap[f]['correlation_with_umap_y'] for f in feat_names],
            'Pvalue_UMAP_Y': [combined_analysis_for_heatmap[f]['pvalue_with_umap_y'] for f in feat_names]
        }
        
        # Add Z-axis data for 3D UMAP
        if use_3d:
            csv_data['Correlation_UMAP_Z'] = [combined_analysis_for_heatmap[f]['correlation_with_umap_z'] for f in feat_names]
            csv_data['Pvalue_UMAP_Z'] = [combined_analysis_for_heatmap[f]['pvalue_with_umap_z'] for f in feat_names]
        
        corr_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f'{prefix}_feature_loading_pearson.csv')
        corr_df.to_csv(csv_path, index=False)
        print(f"Feature loading data saved to {csv_path}")
    
    # Create 3D feature loading visualization if 3D UMAP and correlation analysis available
    combined_analysis_for_viz = {}
    if original_correlation_analysis:
        combined_analysis_for_viz.update(original_correlation_analysis)
    if correlation_analysis:
        combined_analysis_for_viz.update(correlation_analysis)
    
    if use_3d and SCIPY_AVAILABLE and combined_analysis_for_viz and output_path is not None:
        print("Creating 3D feature loading visualization...")
        # Create 3 subplots for XY, XZ, YZ projections
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Calculate center point (mean of UMAP embedding)
        center_x = np.mean(embedding[:, 0])
        center_y = np.mean(embedding[:, 1])
        center_z = np.mean(embedding[:, 2])
        
        # Scale factors for arrow length
        umap_x_range = np.max(embedding[:, 0]) - np.min(embedding[:, 0])
        umap_y_range = np.max(embedding[:, 1]) - np.min(embedding[:, 1])
        umap_z_range = np.max(embedding[:, 2]) - np.min(embedding[:, 2])
        scale_factor_xy = min(umap_x_range, umap_y_range) * 0.3
        scale_factor_xz = min(umap_x_range, umap_z_range) * 0.3
        scale_factor_yz = min(umap_y_range, umap_z_range) * 0.3
        
        # XY projection
        ax_xy = axes[0]
        unique_labels = np.unique(labels)
        for label in sorted(unique_labels):
            mask = labels == label
            if mask.sum() > 0:
                color = COLOR_MAP.get(label, (0.5, 0.5, 0.5))
                color_norm = tuple(c / 255.0 for c in color)
                ax_xy.scatter(embedding[mask, 0], embedding[mask, 1],
                            c=[color_norm], label=f'Class {label}', alpha=0.6, s=20,
                            edgecolors='black', linewidths=0.3)
        
        # Draw arrows for XY projection (original features in blue, MLP output in red)
        for feat_name, corr_data in combined_analysis_for_viz.items():
            corr_x = corr_data['correlation_with_umap_x']
            corr_y = corr_data['correlation_with_umap_y']
            arrow_length = np.sqrt(corr_x**2 + corr_y**2) * scale_factor_xy
            dx = corr_x * scale_factor_xy
            dy = corr_y * scale_factor_xy
            
            # Use different colors for original vs MLP output features
            if feat_name in original_correlation_analysis:
                arrow_color = 'blue'
                label_prefix = '[Orig]'
            else:
                arrow_color = 'red'
                label_prefix = '[MLP]'
            
            ax_xy.arrow(center_x, center_y, dx, dy,
                       head_width=arrow_length*0.1, head_length=arrow_length*0.15,
                       fc=arrow_color, ec=arrow_color, linewidth=2, alpha=0.8)
            label_x = center_x + dx * 1.15
            label_y = center_y + dy * 1.15
            ax_xy.text(label_x, label_y, f'{label_prefix} {feat_name}',
                      fontsize=8, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                              edgecolor=arrow_color, alpha=0.8),
                      ha='center', va='center', color=arrow_color)
        
        ax_xy.set_xlabel(f'UMAP Dimension 1\n({x_axis_meaning})', fontsize=11)
        ax_xy.set_ylabel(f'UMAP Dimension 2\n({y_axis_meaning})', fontsize=11)
        ax_xy.set_title('XY Projection', fontsize=12, fontweight='bold')
        ax_xy.grid(True, alpha=0.3)
        
        # XZ projection
        ax_xz = axes[1]
        for label in sorted(unique_labels):
            mask = labels == label
            if mask.sum() > 0:
                color = COLOR_MAP.get(label, (0.5, 0.5, 0.5))
                color_norm = tuple(c / 255.0 for c in color)
                ax_xz.scatter(embedding[mask, 0], embedding[mask, 2],
                            c=[color_norm], label=f'Class {label}', alpha=0.6, s=20,
                            edgecolors='black', linewidths=0.3)
        
        # Draw arrows for XZ projection (original features in blue, MLP output in red)
        for feat_name, corr_data in combined_analysis_for_viz.items():
            corr_x = corr_data['correlation_with_umap_x']
            corr_z = corr_data['correlation_with_umap_z']
            arrow_length = np.sqrt(corr_x**2 + corr_z**2) * scale_factor_xz
            dx = corr_x * scale_factor_xz
            dz = corr_z * scale_factor_xz
            
            # Use different colors for original vs MLP output features
            if feat_name in original_correlation_analysis:
                arrow_color = 'blue'
                label_prefix = '[Orig]'
            else:
                arrow_color = 'red'
                label_prefix = '[MLP]'
            
            ax_xz.arrow(center_x, center_z, dx, dz,
                       head_width=arrow_length*0.1, head_length=arrow_length*0.15,
                       fc=arrow_color, ec=arrow_color, linewidth=2, alpha=0.8)
            label_x = center_x + dx * 1.15
            label_z = center_z + dz * 1.15
            ax_xz.text(label_x, label_z, f'{label_prefix} {feat_name}',
                      fontsize=8, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                              edgecolor=arrow_color, alpha=0.8),
                      ha='center', va='center', color=arrow_color)
        
        ax_xz.set_xlabel(f'UMAP Dimension 1\n({x_axis_meaning})', fontsize=11)
        ax_xz.set_ylabel(f'UMAP Dimension 3\n({z_axis_meaning})', fontsize=11)
        ax_xz.set_title('XZ Projection', fontsize=12, fontweight='bold')
        ax_xz.grid(True, alpha=0.3)
        
        # YZ projection
        ax_yz = axes[2]
        for label in sorted(unique_labels):
            mask = labels == label
            if mask.sum() > 0:
                color = COLOR_MAP.get(label, (0.5, 0.5, 0.5))
                color_norm = tuple(c / 255.0 for c in color)
                ax_yz.scatter(embedding[mask, 1], embedding[mask, 2],
                            c=[color_norm], label=f'Class {label}', alpha=0.6, s=20,
                            edgecolors='black', linewidths=0.3)
        
        # Draw arrows for YZ projection (original features in blue, MLP output in red)
        for feat_name, corr_data in combined_analysis_for_viz.items():
            corr_y = corr_data['correlation_with_umap_y']
            corr_z = corr_data['correlation_with_umap_z']
            arrow_length = np.sqrt(corr_y**2 + corr_z**2) * scale_factor_yz
            dy = corr_y * scale_factor_yz
            dz = corr_z * scale_factor_yz
            
            # Use different colors for original vs MLP output features
            if feat_name in original_correlation_analysis:
                arrow_color = 'blue'
                label_prefix = '[Orig]'
            else:
                arrow_color = 'red'
                label_prefix = '[MLP]'
            
            ax_yz.arrow(center_y, center_z, dy, dz,
                       head_width=arrow_length*0.1, head_length=arrow_length*0.15,
                       fc=arrow_color, ec=arrow_color, linewidth=2, alpha=0.8)
            label_y = center_y + dy * 1.15
            label_z = center_z + dz * 1.15
            ax_yz.text(label_y, label_z, f'{label_prefix} {feat_name}',
                      fontsize=8, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                              edgecolor=arrow_color, alpha=0.8),
                      ha='center', va='center', color=arrow_color)
        
        ax_yz.set_xlabel(f'UMAP Dimension 2\n({y_axis_meaning})', fontsize=11)
        ax_yz.set_ylabel(f'UMAP Dimension 3\n({z_axis_meaning})', fontsize=11)
        ax_yz.set_title('YZ Projection', fontsize=12, fontweight='bold')
        ax_yz.grid(True, alpha=0.3)
        
        # Add legend for feature types
        from matplotlib.lines import Line2D
        legend_elements = []
        if original_correlation_analysis:
            legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='Original Features'))
        if correlation_analysis:
            legend_elements.append(Line2D([0], [0], color='red', lw=2, label='MLP Output Features'))
        
        if legend_elements:
            fig.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.suptitle(f'{title} - Feature Loading\n(Blue: Original Features, Red: MLP Output Features)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save feature loading visualization
        feature_loading_path = output_path.replace('.png', '_feature_loading.png')
        plt.savefig(feature_loading_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"3D feature loading visualization saved to {feature_loading_path}")
    
    return embedding, reducer, scaler


def visualize_umap_with_predictions(embedding, predictions, output_path, title="UMAP Visualization",
                                     sample_mask=None, use_3d=False):
    """
    Visualize existing UMAP embedding colored by predictions
    
    Args:
        embedding: existing UMAP embedding (N, 2) or (N, 3)
        predictions: predicted class IDs (N,)
        output_path: path to save the visualization
        title: title for the plot
        sample_mask: boolean mask indicating which samples were used for prediction
                     (unused samples will be colored as unknown/gray)
        use_3d: whether to create 3D plot
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping visualization")
        return
    
    # Create predictions array with unknown for unused samples
    if sample_mask is not None:
        pred_colors = np.full(len(embedding), 24, dtype=np.int64)  # 24 = unknown
        pred_colors[sample_mask] = predictions
    else:
        pred_colors = predictions
    
    if use_3d:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points colored by predicted class
        unique_labels = np.unique(pred_colors)
        for label in sorted(unique_labels):
            mask = pred_colors == label
            if mask.sum() > 0:
                color = COLOR_MAP.get(int(label), (0.5, 0.5, 0.5))
                color_norm = tuple(c / 255.0 for c in color)
                label_name = f'Class {label}' if label < 24 else 'Unknown'
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                          c=[color_norm], label=label_name, alpha=0.6, s=20)
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_zlabel('UMAP Dimension 3', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    else:
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot points colored by predicted class
        unique_labels = np.unique(pred_colors)
        for label in sorted(unique_labels):
            mask = pred_colors == label
            if mask.sum() > 0:
                color = COLOR_MAP.get(int(label), (0.5, 0.5, 0.5))
                color_norm = tuple(c / 255.0 for c in color)
                label_name = f'Class {label}' if label < 24 else 'Unknown'
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=[color_norm], label=label_name, alpha=0.6, s=20,
                          edgecolors='black', linewidths=0.3)
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"UMAP visualization with predictions saved to {output_path}")


def visualize_3d_scatter(features_3d, labels, output_path, title="3D PCA (Logits)"):
    """
    3D scatter plot colored by class labels (e.g. after PCA on logits).
    features_3d: (N, 3) numpy array; labels: (N,) class IDs.
    """
    if features_3d is None or len(features_3d) == 0:
        return
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    for label in sorted(unique_labels):
        mask = labels == label
        if mask.sum() > 0:
            color = COLOR_MAP.get(int(label), (0.5, 0.5, 0.5))
            color_norm = tuple(c / 255.0 for c in color)
            label_name = f'Class {label}' if label < 24 else 'Unknown'
            ax.scatter(
                features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                c=[color_norm], label=label_name, alpha=0.6, s=20
            )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"3D scatter saved to {output_path}")


def pca_to_3d(features: np.ndarray, random_state: int = 0) -> np.ndarray:
    """If feature dim > 3, reduce to 3 with PCA for visualization."""
    if features is None:
        return None
    if features.ndim != 2:
        return features
    if features.shape[1] <= 3:
        return features
    pca = PCA(n_components=3, random_state=random_state)
    return pca.fit_transform(features)


def lifter_lift_torch(
    lifter_model: nn.Module,
    input_2d_norm: torch.Tensor,
    condition_norm: torch.Tensor,
    num_steps: int = 50,
):
    """
    Differentiable lifting: x_0 = input_2d_norm, integrate ODE to t=1.
    No constraints during ODE integration (removed all tanh/clamp).
    
    Args:
        input_2d_norm: [B, D] 2D physical already normalized (StandardScaler)
        condition_norm: [B, 2*C] image condition (StandardScaler normalized)
    Returns:
        x: [B, D] lifted features in normalized space
    """
    dt = 1.0 / float(num_steps)
    x = input_2d_norm.clone()
    conditioning = {"condition": condition_norm}
    for i in range(num_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=torch.float32)
        u = lifter_model(x, t, extra=conditioning)
        x = x + u * dt
    return x



def compute_major_group_accuracy(labels, predictions):
    """
    Compute accuracy for each major group from minor class predictions
    Accuracy = (samples with true label in group AND predicted label in same group) / (total samples with true label in group)
    
    Args:
        labels: ground truth minor class labels (N,) - 24 classes (0-23)
        predictions: predicted minor class labels (N,) - 24 classes (0-23)
    
    Returns:
        group_accuracies: dictionary mapping group name to accuracy
    """
    group_accuracies = {}
    
    for group_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        group_classes = MAJOR_GROUPS[group_name]
        
        # Find samples belonging to this major group (by true label)
        mask = np.isin(labels, group_classes)
        
        if mask.sum() > 0:
            # For samples with true label in this group, check if predicted label is also in this group
            group_pred_labels = predictions[mask]
            
            # Count correct: predicted label is also in the same major group
            correct_mask = np.isin(group_pred_labels, group_classes)
            group_acc = correct_mask.sum() / len(group_pred_labels)
            
            group_accuracies[group_name] = {
                'accuracy': float(group_acc),
                'num_samples': int(mask.sum())
            }
        else:
            group_accuracies[group_name] = {
                'accuracy': 0.0,
                'num_samples': 0
            }
    
    return group_accuracies


def compute_major_group_accuracy_from_major_predictions(major_labels, major_predictions):
    """
    Compute accuracy for each major group from major group predictions (using last layer classifier)
    Accuracy = (samples with true major group AND predicted major group match) / (total samples with true major group)
    
    Args:
        major_labels: ground truth major group labels (N,) - 7 groups (0-6, corresponding to A-G)
        major_predictions: predicted major group labels (N,) - 7 groups (0-6, corresponding to A-G)
    
    Returns:
        group_accuracies: dictionary mapping group name to accuracy
    """
    group_accuracies = {}
    
    for group_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        major_group_idx = MAJOR_GROUP_TO_INDEX[group_name]  # 0-6
        
        # Find samples belonging to this major group (by true major group label)
        mask = major_labels == major_group_idx
        
        if mask.sum() > 0:
            # For samples with true major group, check if predicted major group matches
            group_pred_labels = major_predictions[mask]
            
            # Count correct: predicted major group matches true major group
            correct_mask = group_pred_labels == major_group_idx
            group_acc = correct_mask.sum() / len(group_pred_labels)
            
            group_accuracies[group_name] = {
                'accuracy': float(group_acc),
                'num_samples': int(mask.sum())
            }
        else:
            group_accuracies[group_name] = {
                'accuracy': 0.0,
                'num_samples': 0
            }
    
    return group_accuracies


def create_confusion_matrix_visualization(labels, predictions, output_path, title="Confusion Matrix", num_classes=24):
    """
    Create confusion matrix visualization with magenta-cyan colormap
    Reordered by major groups (A, B, C, D, E, F, G)
    
    Args:
        labels: ground truth labels (N,)
        predictions: predicted labels (N,)
        output_path: path to save the visualization
        title: title for the plot
        num_classes: number of classes (default: 24)
    """
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions, labels=np.arange(num_classes))
    
    # Create reordering index based on major groups
    # Order: A, B, C, D, E, F, G
    reorder_indices = []
    for group_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        reorder_indices.extend(MAJOR_GROUPS[group_name])
    
    # Verify that all classes are included
    if len(reorder_indices) != num_classes or set(reorder_indices) != set(range(num_classes)):
        raise ValueError(f"Reordering indices incomplete: expected {num_classes} classes, got {len(reorder_indices)}. "
                         f"Missing: {set(range(num_classes)) - set(reorder_indices)}")
    
    # Create mapping from old index to new index
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(reorder_indices)}
    
    # Reorder confusion matrix
    cm_reordered = np.zeros_like(cm)
    for i in range(num_classes):
        for j in range(num_classes):
            if i in old_to_new and j in old_to_new:
                new_i = old_to_new[i]
                new_j = old_to_new[j]
                cm_reordered[new_i, new_j] = cm[i, j]
    
    # Normalize confusion matrix (percentage)
    cm_normalized = cm_reordered.astype('float') / (cm_reordered.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create magenta-cyan colormap (same as test_3d_volume_semantic.py)
    colors_magenta_cyan = [(0, 1, 1), (1, 0, 1)]  # Cyan to Magenta
    cmap = LinearSegmentedColormap.from_list('magenta_cyan', colors_magenta_cyan, N=100)
    
    # Create tick labels with major group information
    tick_labels = []
    for group_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        for class_id in MAJOR_GROUPS[group_name]:
            tick_labels.append(f'{class_id}\n({group_name})')
    
    # Plot raw confusion matrix
    im1 = ax1.imshow(cm_reordered, cmap=cmap, aspect='auto', interpolation='nearest')
    ax1.set_title(f'{title} (Raw Counts, Ordered by Major Groups)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Class (Major Group)', fontsize=12)
    ax1.set_ylabel('True Class (Major Group)', fontsize=12)
    ax1.set_xticks(np.arange(num_classes))
    ax1.set_yticks(np.arange(num_classes))
    ax1.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha='right')
    ax1.set_yticklabels(tick_labels, fontsize=7)
    
    # Add text annotations for raw counts
    thresh = cm_reordered.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax1.text(j, i, f'{cm_reordered[i, j]}',
                          ha="center", va="center",
                          color="white" if cm_reordered[i, j] > thresh else "black",
                          fontsize=7)
    
    # Add colorbar for raw counts
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Count (Magenta=High, Cyan=Low)', rotation=270, labelpad=20)
    
    # Plot normalized confusion matrix
    im2 = ax2.imshow(cm_normalized, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=100)
    ax2.set_title(f'{title} (Normalized %, Ordered by Major Groups)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Class (Major Group)', fontsize=12)
    ax2.set_ylabel('True Class (Major Group)', fontsize=12)
    ax2.set_xticks(np.arange(num_classes))
    ax2.set_yticks(np.arange(num_classes))
    ax2.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha='right')
    ax2.set_yticklabels(tick_labels, fontsize=7)
    
    # Add text annotations for normalized percentages
    thresh_norm = 50.0
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax2.text(j, i, f'{cm_normalized[i, j]:.1f}%',
                          ha="center", va="center",
                          color="white" if cm_normalized[i, j] > thresh_norm else "black",
                          fontsize=7)
    
    # Add colorbar for normalized
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Percentage (Magenta=High, Cyan=Low)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Confusion matrix visualization saved to {output_path}")


def create_phys_to_feat_heatmap(phys_input, feat_output, labels, feature_keys, output_path, title=None, num_classes=24, all_layers=None):
    """
    Visualize physical feature transformation from input to output by class centroid.
    Each class = one block. Within block: rows = physical features, cols = layers (Input, Block0, ..., Output).
    Input normalized to 1, other layers = layer/input ratio.
    Colormap: same as confusion matrix (cyan to magenta).
    
    Args:
        phys_input: [N, n_phys] input physical features
        feat_output: [N, n_phys] output features from IDER
        labels: [N] class labels (0-23)
        feature_keys: list of feature names
        output_path: path to save the figure
        title: optional title
        num_classes: 24
        all_layers: optional list of [N, n_phys] arrays from IDER return_all_layers [input, block0, ..., output]
    """
    n_phys = phys_input.shape[1]
    n_feat = min(len(feature_keys), n_phys)
    feat_names = feature_keys[:n_feat] if feature_keys else [f'F{i}' for i in range(n_feat)]

    # Resolve layers: use all_layers if provided, else [input, output]
    if all_layers is not None and len(all_layers) >= 2:
        layer_arrays = all_layers  # [input, block0, block1, ..., output]
        layer_names = ['Input'] + [f'Block{i}' for i in range(len(all_layers) - 2)] + ['Output']
    else:
        layer_arrays = [phys_input, feat_output]
        layer_names = ['Input', 'Output']

    n_layers = len(layer_arrays)
    eps = 1e-8

    # Per-class centroids for each layer
    centroid_per_layer = []
    for ly in layer_arrays:
        cent = np.full((num_classes, n_phys), np.nan)
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            cent[c] = np.mean(ly[mask], axis=0)
        centroid_per_layer.append(cent)

    centroid_input = centroid_per_layer[0]

    # Same colormap as confusion matrix: Cyan to Magenta
    colors_magenta_cyan = [(0, 1, 1), (1, 0, 1)]  # Cyan to Magenta
    cmap = LinearSegmentedColormap.from_list('magenta_cyan', colors_magenta_cyan, N=100)

    # Reorder classes by major groups (A, B, C, D, E, F, G)
    reorder_indices = []
    for group_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        reorder_indices.extend(MAJOR_GROUPS[group_name])

    n_rows, n_cols_grid = 6, 4
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(max(3, n_layers * 0.8) * n_cols_grid, 3 * n_rows))
    axes = axes.flatten()

    # Global vmin/vmax for consistent colorbar
    all_vals = []
    for c in reorder_indices:
        inp = centroid_input[c]
        for cent in centroid_per_layer:
            r = np.where(np.abs(inp) > eps, cent[c] / (inp + eps), 1.0)
            r = np.nan_to_num(r, nan=1.0, posinf=1.0, neginf=1.0)
            all_vals.extend(r)
    all_vals = np.array(all_vals)
    vmin = max(0.5, np.nanpercentile(all_vals, 2)) if np.isfinite(all_vals).any() else 0.5
    vmax = min(1.5, np.nanpercentile(all_vals, 98)) if np.isfinite(all_vals).any() else 1.5
    vmin, vmax = min(vmin, 1.0), max(vmax, 1.0)

    for idx, c in enumerate(reorder_indices):
        ax = axes[idx]
        inp = centroid_input[c]
        # Block: rows = features, cols = layers; input normalized to 1
        block = np.zeros((n_feat, n_layers))
        for j, cent in enumerate(centroid_per_layer):
            ratio = np.where(np.abs(inp) > eps, cent[c] / (inp + eps), 1.0)
            ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
            block[:, j] = ratio[:n_feat]

        im = ax.imshow(block, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_yticks(np.arange(n_feat))
        ax.set_yticklabels(feat_names, fontsize=7)
        ax.set_xticks(np.arange(n_layers))
        ax.set_xticklabels(layer_names, fontsize=7, rotation=45, ha='right')
        g = CLASS_TO_MAJOR_GROUP.get(c, '?')
        ax.set_title(f'Class {c} ({g})', fontsize=9)

        for i in range(n_feat):
            for j in range(n_layers):
                val = block[i, j]
                thresh = (vmin + vmax) / 2
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if val > thresh else 'black')

    for idx in range(len(reorder_indices), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title or 'Input→Output (Input=1 ref)', fontsize=12)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Value (Cyan=Low, Magenta=High)', rotation=270, labelpad=20)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Phys-to-feat heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train the 7D physical-feature ChromoLift flow model')

    # Split-path mode (required)
    parser.add_argument("--phy_3d", type=str, required=True, help="3D physical descriptor JSONL (historically instance_3d_optimal.jsonl).")
    parser.add_argument("--phy_3d_mip", type=str, required=True, help="Paired 2D MIP descriptor JSONL (historically instance_3d_naive.jsonl).")
    parser.add_argument("--phy_2d", type=str, required=True, help="Independent labeled 2D descriptor JSONL (historically instance_2d.jsonl).")
    parser.add_argument("--img_2d", type=str, required=True, help="Directory containing the 2D image-feature NPY files.")
    parser.add_argument("--img_3d_mip", type=str, required=True, help="Directory containing the paired 3D MIP image-feature NPY files.")
    # Generate timestamp for default output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_output_dir = str(Path(__file__).resolve().parents[3] / 'Checkpoint' / 'ChromoPhy' / 'ChromoLift' / f'training_{timestamp}')
    
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                       help='Output directory for results')
    
    # GPU selection
    parser.add_argument('--gpu_id', type=int, default=1,
                       help='GPU ID to use for training. If None, use default GPU or CPU.')
    
    parser.add_argument('--lifter_ckpt', type=str, 
                        default=None,
                       help='Path to existing lifter checkpoint. If set, load lifter and skip lifter training.')
    parser.add_argument('--lifter_epochs', type=int, default=50,
                       help='Number of epochs to train the lifter model. If 0 and no --lifter_ckpt, skip lifter training.')
    parser.add_argument('--lifter_batch_size', type=int, default=2048,
                       help='Batch size for lifter training.')
    parser.add_argument('--lifter_infer_num_steps', type=int, default=50,
                       help='Number of ODE Euler steps for lifter inference during joint training.')
    parser.add_argument('--lambda_self_distill_lifter', type=float, default=0.0,
                       help='Weight for lifter self-distill (no-cond output chases cond output). 0 to disable.')
    parser.add_argument('--self_distill_lifter_type', type=str, default='l2', choices=['l2', 'cosine'],
                       help='Lifter self-distill alignment loss: l2 (MSE) or cosine.')

    # Lifter training and eval
    parser.add_argument('--lambda_domain', type=float, default=0.01,
                       help='Weight for domain adversarial loss (lifter).')
    parser.add_argument('--lambda_boundary', type=float, default=1.0,
                       help='Weight for boundary penalty loss in lifter training (penalizes exceeding 3D data min/max).')
    parser.add_argument('--grl_lambda', type=float, default=1.0,
                       help='Gradient reversal strength for domain adversarial training.')
    
    # Feature names: training vs visualization (some physical quantities may be excluded)
    parser.add_argument('--feature_names', type=str, nargs='+',
                       default=['norm_size',
                                'eccentricity',
                                'solidity',
                                'circularity',
                                'bulkiness',
                                'tortuosity',
                                'inertia_ratio',
                                ],
                       help='Physical feature names for training (lifter input/target). Must match jsonl keys.')
    parser.add_argument('--umap_feature_names', type=str, nargs='+', 
                        default=['norm_size',
                                'eccentricity',
                                'solidity',
                                'circularity',
                                'tortuosity',
                                ],
                       help='Physical feature names for visualization (UMAP loading plot, labels). If not set, defaults to feature_names. Can be a subset (e.g. 5 features for clearer plots).')
    
    # Training
    parser.add_argument('--samples_per_group', type=int, default=32,
                       help='Number of samples per minor group in training batch. '
                            'Actual training batch_size = num_minor_groups * samples_per_group '
                            '(typically 24 classes * 4 = 96 samples per batch)')
    # Visualization / Eval
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                       help='Ratio of data per class ID to use for visualization and evaluation (0.0-1.0). '
                            'Default: 0.1 (10%%). When set to 1.0, use full data for UMAP initialization and evaluation.')
    parser.add_argument('--eval_epochs', type=int, default=5,
                       help='Run evaluation and visualization every N lifter epochs (save lifted_2d/mip, pred_3d, UMAP by pred id).')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    feature_names = args.feature_names
    umap_feature_names = (args.umap_feature_names if args.umap_feature_names is not None else feature_names)
    print(f"\nFeature names (training): {feature_names}")
    print(f"Feature names (UMAP/visualization): {umap_feature_names}")
    
    # ============================================================================
    # Initialize device
    # ============================================================================
    if args.gpu_id is not None and torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu_id} not available. Using GPU 0 or CPU.")
            device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ============================================================================
    # Prepare data from split paths (phy jsonl + img npy dirs)
    # ============================================================================
    print("\n" + "="*80)
    print("LOADING DATA FROM PATHS")
    print("="*80)
    data_dict = prepare_data_from_paths(
        phy_3d=args.phy_3d,
        phy_3d_mip=args.phy_3d_mip,
        phy_2d=args.phy_2d,
        img_2d=args.img_2d,
        img_3d_mip=args.img_3d_mip,
        feature_keys=feature_names,
    )

    # UMAP init from original 3D (volume_records) before lifter training
    umap_scaler_path, umap_reducer_path = init_umap_from_volume_records(
        data_dict, args.output_dir, feature_names,
    )

    # ============================================================================
    # PIPELINE: Lifter -> Full Lift (eval every eval_epochs: lifted_2d/mip, pred_3d, UMAP by pred id)
    # ============================================================================
    lifter_model = None
    lifter_scaler_mip = None
    lifter_scaler_volume = None

    # Step 1: Load lifter from checkpoint or train lifter
    if args.lifter_ckpt is not None and os.path.isfile(args.lifter_ckpt):
        if not FLOW_MATCHING_AVAILABLE:
            print("ERROR: flow_matching library is not available. Cannot load lifter.")
            sys.exit(1)
        print("\n" + "="*80)
        print("STEP 1: LOADING LIFTER FROM CHECKPOINT")
        print("="*80)
        ckpt = torch.load(args.lifter_ckpt, map_location=device, weights_only=False)
        n_phys = data_dict['n_phys']
        image_dim = data_dict['image_dim']
        lifter_model = FeatureFlowMatchingLifter(
            input_dim=n_phys,
            target_dim=n_phys,
            condition_dim=image_dim,
            hidden_dim=256,
        ).to(device)
        lifter_model.load_state_dict(ckpt['model_state_dict'])
        lifter_scaler_mip = ckpt.get('scaler_mip')
        lifter_scaler_volume = ckpt.get('scaler_volume')
        if lifter_scaler_mip is None or lifter_scaler_volume is None:
            lifter_scaler_mip = data_dict['scaler_mip']
            lifter_scaler_volume = data_dict['scaler_volume']
        lifter_model.eval()
        print(f"Loaded lifter from {args.lifter_ckpt}")
    elif args.lifter_epochs > 0:
        if not FLOW_MATCHING_AVAILABLE:
            print("ERROR: flow_matching library is not available. Cannot train lifter.")
            sys.exit(1)
        print("\n" + "="*80)
        print("STEP 1: TRAINING LIFTER (2D->3D, condition=image)" + (" + 3-WAY DOMAIN + TRIPLET" if data_dict.get('static_samples') else ""))
        print("="*80)
        lifter_model, lifter_scaler_mip, lifter_scaler_volume, data_dict = train_lifter(
            data_dict=data_dict,
            output_dir=args.output_dir,
            epochs=args.lifter_epochs,
            batch_size=args.lifter_batch_size,
            lr=1e-4,
            gpu_id=args.gpu_id,
            samples_per_group=args.samples_per_group,
            lifter_infer_num_steps=args.lifter_infer_num_steps,
            lambda_domain=args.lambda_domain,
            lambda_boundary=args.lambda_boundary,
            lambda_self_distill_lifter=args.lambda_self_distill_lifter,
            self_distill_lifter_type=args.self_distill_lifter_type,
            grl_lambda=args.grl_lambda,
            eval_epochs=args.eval_epochs,
            feature_keys=feature_names,
            umap_scaler_path=umap_scaler_path,
            umap_reducer_path=umap_reducer_path,
        )
        lifter_model = lifter_model.to(device)
        lifter_model.eval()
        print("Lifter training completed!")

    # Step 2: Full lift and save (lifted_2d, lifted_mip, volume_3d)
    lifted_2d_path = os.path.join(args.output_dir, 'lifted_2d.json')
    lifted_mip_path = os.path.join(args.output_dir, 'lifted_mip.json')
    volume_3d_path = os.path.join(args.output_dir, 'volume_3d.json')
    if lifter_model is not None and lifter_scaler_mip is not None and lifter_scaler_volume is not None:
        print("\n" + "="*80)
        print("STEP 2: FULL LIFT AND SAVE (lifted_2d, lifted_mip, volume_3d)")
        print("="*80)
        run_full_lift_and_save_from_data_dict(
            lifter_model, data_dict, args.output_dir, device,
            num_steps=args.lifter_infer_num_steps,
            batch_size=args.lifter_batch_size,
        )
    else:
        if not os.path.isfile(lifted_mip_path):
            print("ERROR: lifter_epochs=0 and no checkpoint; missing lifted_mip.json.")
            sys.exit(1)
        if not os.path.isfile(volume_3d_path):
            print("ERROR: volume_3d.json missing.")
            sys.exit(1)
        print("\nUsing existing lifted JSONs in output_dir.")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE (Lifter only; no IDER)")
    print(f"  Results: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
