# ChromoLift: 2D-to-3D flow matching

[Download the released ChromoLift checkpoint](../../../docs/downloads.md#chromoworld-weights).

Run commands from the release root. model.py defines the checkpoint-compatible
network; train.py trains and validates it; infer.py performs the maintained
50-step Euler lift; evaluate.py scores paired held-out descriptor rows.

```bash
python -m pip install -r Code/ChromoPhy/chromolift/requirements.txt
python Code/ChromoPhy/chromolift/train.py \
  --phy_3d /data/instance_3d_optimal.jsonl \
  --phy_3d_mip /data/instance_3d_naive.jsonl \
  --phy_2d /data/instance_2d.jsonl \
  --img_3d_mip /data/3D_features --img_2d /data/2D_features \
  --output_dir /output/chromolift_training
```

phy_3d contains 3D targets, phy_3d_mip their paired MIP descriptors, and phy_2d
labeled fixed-cell descriptors for domain alignment. The image directories
contain matching per-sample NPY features. These training inputs are not
included in Dataset. Exact historic retraining also requires the original
training snapshots and arguments: the source server's JSONL files were
modified after the published checkpoint.

The released Checkpoint/ChromoPhy/ChromoLift/lifter_final.pth contains network
state and scaler_mip/scaler_volume/scaler_image. Inference NPZ arrays are
features [N,7] in order:
norm_size, eccentricity, solidity, circularity, bulkiness, tortuosity,
inertia_ratio; optional condition is [N,1536].

Its serialized StandardScaler objects were fitted with scikit-learn 1.7.1.
The requirements pin that version for checkpoint compatibility; use a separate
Python environment if another project needs a different version.

```bash
python Code/ChromoPhy/chromolift/infer.py \
  --input /data/paired_input.npz --output /output/lifted_features.npz
python Code/ChromoPhy/chromolift/evaluate.py \
  --input /data/heldout_pairs.npz --output /output/lift_metrics.json
```

Evaluation expects features and targets with aligned [N,7] rows and optional
condition. It reports physical-feature MAE/RMSE and standardized RMSE.
Evaluation splits must be prepared without training leakage by the caller.
Without condition, inference evaluates the unconditioned variant; it does
not reconstruct the conditioned fixed-cell gallery used by the historical
identity retrieval.

Artifacts/fig5/LiftedMIP_umap_embed.npz and Lifted2D_umap_embed.npz contain
five-feature plotting assets plus saved 2D coordinates. They are sufficient
to redraw those plots, but are not the paired seven-feature training set.
