# Figure reproduction

Run commands from the release root. Install `Code/Analyze/requirements.txt`
for descriptors, evaluation, and plotting, and install the relevant model's
requirements for checkpoint inference. Relative paths in the Fig2-3 and UMAP
entry points resolve from the release root; absolute paths are also accepted.

## Fig2-3: processed images to predictions, metrics, and panels

```bash
python Code/Analyze/fig2-3/get-polygon_PNG2TXT.py --gpu_id 0
python Code/Analyze/fig2-3/get-metrics_TXT2CSV.py
```

The input is Dataset/2D/test/image and the reference labels are
Dataset/2D/test/label. The default weights are
Checkpoint/ChromoPix/chromopix.pth and
Checkpoint/Outsource/DINOv3/dinov3_vitb16.pth. ChromoPix uses 512×512 inference
and returns polygons on each input image's native pixel grid.

Prediction TXT lines are `class confidence x1 y1 ...`; GT lines are
`class x1 y1 ...`, with normalized polygon coordinates. The evaluator writes
overall_stats.csv, tag_stats.csv, and PR plots under
Code/Analyze/results/reports/Fig2. Its historical default filters predicted
classes by their presence in that image's GT; this setting affects metrics.
Additional methods and their class-filter policies are explicit CLI inputs:

```bash
python Code/Analyze/fig2-3/get-metrics_TXT2CSV.py \
  --baseline_path '{"ChromoPix":"Code/Analyze/results/predictions/ChromoPix","MaskRCNN":"/data/maskrcnn_txt"}' \
  --class_configurable '{"ChromoPix":true,"MaskRCNN":false}'
```

The comparison models' inference code and weights are not bundled.
test-robustness_TXT2PNG.py creates synchronized image/label perturbations;
rerun inference and evaluation on each generated condition for Fig3.
See [Fig2-3](fig2-3/README.md) for the retained five-row Banding/FISH panels
and the per-image polygon visualizer.

## Fig4: morphology and saved fixed-cell embeddings

get-2ddescriptors_TXT2JSON.py pairs processed images and polygon TXT files.
qc-sample_JSON2JSON.py filters and normalizes the resulting JSONL.
Their enriched feature export is separate from the five historical features
used to fit the published UMAPs.

```bash
python Code/Analyze/fig4/visualize-umap_NPZ2PNG.py \
  --embedding-npz Artifacts/fig4/Combine_umap_embed.npz --space Combine
```

Banding/FISH/Combine NPZs contain fitted coordinates, labels, split tags, and
row indices. This command redraws those coordinates without rerunning UMAP,
ChromoPix, or descriptor extraction. See [Fig4](fig4/README.md).

## Fig5: volume/movie masks, descriptors, and saved embeddings

get-3dmask_TIFF2TIFF.py uses the outsourced CPSAM checkpoint on 3D TIFFs.
get-3ddescriptor_TIFF2JSON.py pairs volumes with instance masks.
get-4dmask_TIFF2TIFF.py runs raw → preprocess → image-only affine →
ChromoPose → postprocess → track.
get-visualize-image_TIFF2GIF.py and get-visualize-mask_TIFF2GIF.py render
six-second, fixed front/back views with physical scale bars and independent
instance colors. See the Fig5 README for directory layout and calibration.

```bash
python Code/Analyze/fig5/get-4dmask_TIFF2TIFF.py \
  --input /data/raw_movies --output-dir /output/4d --gpu-ids 0
python Code/Analyze/fig4/visualize-umap_NPZ2PNG.py \
  --embedding-npz Artifacts/fig5/3D_umap_embed.npz --space 3D
```

Artifacts/fig5 also contains MIP, LiftedMIP, and Lifted2D embeddings.
Rendering them does not repeat the lift or identity retrieval. The retained
five-feature UMAP arrays cannot substitute for the seven-feature ChromoID
gallery. See [Fig5](fig5/README.md) for resource requirements and limitations.
