# ChromoPix: 24-class 2D segmentation

Run commands from the release root. chromopix/ contains dataloader.py,
model.py, train.py, train.sh, infer.py, evaluate.py, and shared training
utilities. The required DINOv3 source subset and upstream license are
included under dinov3/. Neural weights live outside Code.

```bash
python -m pip install -r Code/ChromoPix/requirements.txt
bash Code/ChromoPix/chromopix/train.sh 0
python Code/ChromoPix/chromopix/evaluate.py --gpu-ids 0
```

The training launcher uses Dataset/2D/train and test image/label directories,
512×512 inputs, and the outsourced DINOv3 ViT-B/16 weights at
Checkpoint/Outsource/DINOv3/dinov3_vitb16.pth.
Visualization outputs are written under Code/ChromoPix/outputs, rather than
requiring Dataset visualization folders. Extra CLI flags follow the GPU list.

The same maintained inference implementation is exposed both as model code
and the Fig2-3 analysis entry:

```bash
python Code/ChromoPix/chromopix/infer.py --gpu_id 0
```

Default model weights are Checkpoint/ChromoPix/chromopix.pth. Inference queries
24 classes and exports class/confidence/polygon TXT files under
Code/Analyze/results/predictions/ChromoPix. Polygons are normalized on the
original input image grid after 512×512 inference.

evaluate.py runs inference and the shared TXT evaluator on held-out pairs.
It writes predictions and overall/per-class CSV/PR reports under
Code/ChromoPix/outputs/evaluation. To evaluate existing predictions:

```bash
python Code/ChromoPix/chromopix/evaluate.py \
  --prediction-dir Code/Analyze/results/predictions/ChromoPix
```

The default --class-filter present-in-gt preserves the historical figure
metric protocol; --class-filter all evaluates every predicted class.
This choice changes false-positive accounting and is recorded by the common
evaluator. The default test split is separate from training; caller-supplied
data must preserve that separation.

A CUDA-compatible PyTorch/torchvision installation is needed for the maintained
inference entry, which selects GPUs with --gpu_id. GPU selection in the trainer
is controlled by the launcher. Dependencies are listed without claiming a
fully locked environment; the source model computation has not been changed
by the release-directory reorganization.
