# ChromoPose3D segmentation

The segmentation package has seven focused Python modules:

- `dataloader.py` — Chimps sample indexing, TIFF I/O, normalization, resizing,
  augmentation, and training batches.
- `model.py` — 3D U-Net, output-channel definitions, and checkpoint/model helpers.
- `preprocess.py` — pairs Chimps images with instance-mask TIFFs, resizes each
  time point, and creates the training targets and dataset index.
- `train.py` — optimization, losses, checkpointing, and validation during
  training.
- `evaluate.py` — evaluates the held-out Chimps test split against its GT masks,
  writes per-frame comparison panels, and reports instance metrics. It can run
  independently or be called by the training loop.
- `infer.py` — real-use TIFF inference: accepts an arbitrary 3D or 4D image,
  predicts instance masks, and writes mask TIFFs. It does not prepare or score
  the Chimps train/test dataset.
- `visualize.py` — prediction, target, and volume visualization utilities.

The old helper implementations have been folded into the module matching their
responsibility. `train.sh` and `preprocess.sh` are launchers, not extra Python
modules. Target generation can use Cellpose's mask-boundary utility; standalone
inference uses Cellpose's `follow_flows` dynamics for phase two, but does not
load a Cellpose3D model or apply watershed refinement.

The Chimps evaluator reconstructs instances with the training/evaluation
skeleton-graph postprocessor to score against GT masks. Standalone TIFF
inference uses the maintained two-phase flow decoder. Keep these entry points
distinct when interpreting test metrics versus deployment masks.

Run commands from `release/Code/ChromoPhy/` or use the shell launchers from this
directory.

## Prepare Chimps training data

The preprocessor reads paired Chimps image and instance-mask TIFFs and produces
per-timepoint `raw/`, `mask/`, `target/`, `target_vis/`, and `index.jsonl` files.
It accepts either the Chimps folder layout or an explicit paired-data source:

```bash
python chromopose/segmentation/preprocess.py \
  --chimp-root /path/to/chimps \
  --out-root /path/to/chromopose3d
```

See `python chromopose/segmentation/preprocess.py --help` for the input layout
and target-generation options. `preprocess.sh` defaults to the release's
`Dataset/4D/Chimps` and `Dataset/4D/ChromoPoseTargets` directories; these inputs
are not bundled. Configure `CHIMP_ROOT`, `DATA_ROOT`, and `WORKERS` as needed.
Neither launcher requires a server-specific environment: set `CONDA_ENV`
only if you want it to activate an already installed Conda environment.

## Train and evaluate

Train on the preprocessed dataset using `train.sh`, or invoke the CLI directly:

```bash
python chromopose/segmentation/train.py \
  --index-path /path/to/chromopose3d/index.jsonl \
  --data-root /path/to/chromopose3d
```

Run test-set evaluation separately against the dataset's `test` split and a
trained checkpoint:

```bash
python chromopose/segmentation/evaluate.py \
  --index-path /path/to/chromopose3d/index.jsonl \
  --data-root /path/to/chromopose3d \
  --checkpoint /path/to/checkpoint.pth \
  --output-dir /path/to/evaluation
```

The checkpoint's saved model configuration and crop dimensions are used by
default. Evaluation writes an `eval_summary.json` and per-frame PNG comparisons.

## Infer on new TIFF data

For deployment or new experiments, use `infer.py` with a 3D or 4D TIFF input;
the model resizes internally to its training dimensions and restores output
masks to the source resolution:

```bash
python chromopose/segmentation/infer.py \
  --in_path /path/to/input.tif \
  --out_path /path/to/masks \
  --ckpt /path/to/checkpoint.pth \
  --gpu_id 0
```

The default checkpoint is resolved from
`release/Checkpoint/ChromoPhy/ChromoPose/` when that release folder is present.
