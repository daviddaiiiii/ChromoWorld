# ChromoPix code and bundled DINOv3 runtime

This directory contains the ChromoPix model, its training code, and the small
DINOv3 source subset needed to construct the pretrained ViT-B/16 backbone and
the DINOv3 attention layers used by ChromoPix.

## Bundled DINOv3 source

The `dinov3/` directory is a dependency-closed source subset copied from the
DINOv3 checkout used on H100. It contains the ViT-B/16 Torch Hub factory, the
ViT implementation, its layers and utility functions, and a minimal
`hubconf.py` that exposes `dinov3_vitb16`.
This makes the source code available locally; it does not duplicate the model
weights.

The associated weights remain in:

```text
release/Checkpoint/ChromoPix/chromopix.pth
release/Checkpoint/ChromoPix/dinov3_vitb16.pth
```

The DINOv3 license is included at `dinov3/LICENSE.md`. DINOv3 code and weights
are governed by that agreement; retain it when redistributing these materials.

## Reference runtime

The H100 `dinov3` environment was inspected while preparing this bundle:

```text
Python       3.11.13
PyTorch      2.8.0+cu128
CUDA runtime 12.8
torchvision  0.23.0+cu128
timm         1.0.19
NumPy        2.2.6
```

The ChromoPix model imports DINOv3 classes directly from this local source and
loads the backbone through the bundled Torch Hub entry point. The inference
script defaults to this directory for `--dinov3_repo`. Both the training and
inference entry points add this Torch Hub root to `sys.path` before importing
the ChromoPix model. Training runs from this directory's parent and uses
`./dinov3`.

The model file had an unused `timm` `Block` import; it has been removed, so
`timm` is not needed by the ChromoPix architecture itself. The H100 reference
environment includes it, but the ChromoPix code does not instantiate it.

Install a PyTorch/torchvision pair built for the target machine's CUDA runtime,
then install the Python packages listed in `requirements.txt`. Training also
imports Matplotlib and Weights & Biases; both are listed there.

The bundled launcher assumes the released dataset and checkpoints are in their
standard project locations. From the ChromoWorld root, start training with:

```bash
bash release/Code/ChromoPix/chromopix/train.sh 0,1,2,3
```

The launcher resolves its paths from `release/Code/ChromoPix`, points to the
local DINOv3 source, the released ViT-B/16 weights, and `release/Dataset/2D`.
