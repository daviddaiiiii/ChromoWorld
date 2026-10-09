# ChromoWorld code

Run the commands below from the release root, which contains `Code/`,
`Dataset/`, `Checkpoint/`, and `Artifacts/`. The root directory may be
named `release` or `ChromoWorld`; entry points locate resources from their
own file location. Analyze relative paths are resolved against that root.
Historical `release/...` and `Analyze/...` CLI paths remain accepted aliases.

## Model development

`ChromoPix/` contains the 2D model, dataloader, training, inference, and
standalone evaluation entry points, plus the required DINOv3 source subset.
`ChromoPhy/` contains ChromoPose movie segmentation, ChromoLift descriptor
lifting, and ChromoID gallery construction and retrieval. Each model documents
its input contract and dependencies in its own README.

For example:

```bash
python -m pip install -r Code/ChromoPix/requirements.txt
bash Code/ChromoPix/chromopix/train.sh 0
python Code/ChromoPix/chromopix/evaluate.py --gpu-ids 0
```

ChromoPix's released train/test images and polygon labels are in Dataset/2D.
ChromoPose training additionally needs paired Chimps images and GT masks.
ChromoLift training needs historical paired seven-feature JSONL records and
image-feature NPYs. Those extra training inputs are not currently bundled.
ChromoID builds a gallery rather than training another neural network.

## Figure analysis with released weights

`Analyze/` contains the figure entry points. Fig2-3 uses ChromoPix to export
polygon predictions, score them, and render panels. Fig4 extracts and filters
2D descriptors and renders saved fixed-cell UMAPs. Fig5 extracts 3D masks and
descriptors, runs the ChromoPose 4D pipeline, and renders live/lifted UMAPs.

```bash
python -m pip install -r Code/Analyze/requirements.txt
python Code/Analyze/fig2-3/get-polygon_PNG2TXT.py --gpu_id 0
python Code/Analyze/fig2-3/get-metrics_TXT2CSV.py
python Code/Analyze/fig4/visualize-umap_NPZ2PNG.py \
  --embedding-npz Artifacts/fig4/Combine_umap_embed.npz --space Combine
```

Saved UMAP rendering requires NumPy and Matplotlib, without checkpoints or
GPU inference. Recomputing segmentation needs the corresponding checkpoint
and model dependencies. Baseline predictions are supplied explicitly to the
common evaluator; its default component is ChromoPix.

See [Analyze](Analyze/README.md), [ChromoPix](ChromoPix/README.md), and
[ChromoPhy](ChromoPhy/README.md). Downloaded weights live outside Code,
and fitted NPZ assets live exclusively in Artifacts.
