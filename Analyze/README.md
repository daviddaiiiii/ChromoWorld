# Chromo World analysis utilities

This directory contains the analysis pipeline used for the 2D Chromo World experiments in Figures 2 and 3.

The project is **Chromo World**. Its primary 2D segmentation component is **ChromoPix**. The default comparison components are:

- `ChromoPix`
- `MaskRCNN`
- `MaskRCNN_Banding`
- `YOLOv11-Seg`
- `YOLOv12-Seg`
- `YOLOv11-Seg_Banding`
- `YOLOv12-Seg_Banding`

The names above are stable report identifiers. `MaskRCNN` means Mask R-CNN, and the `_Banding` suffix identifies a component evaluated on the banding subset or trained with the corresponding banding setup.

## Pipeline

The analysis is deliberately separated into four stages:

```text
PNG image
  -> ChromoPix inference
  -> YOLO-segmentation TXT predictions
  -> TXT-based component evaluation
  -> CSV reports and PR curves
```

The scripts are:

| Script | Purpose | Main output |
| --- | --- | --- |
| `test_extraction_PNG2TXT.py` | Run ChromoPix on PNG images and export 24-class polygon predictions | One YOLO-style TXT file per image |
| `test_robustness_TXTPNG.py` | Create synchronized robustness images and labels | Transformed PNG and TXT pairs |
| `test_baseline_TXT2CSV.py` | Compare multiple component TXT directories against GT TXT labels | `overall_stats.csv`, `tag_stats.csv`, PR curves |
| `test_visualize_TXT2PNG.py` | Render TXT polygons on their matching images | PNG overlays and panel visualizations |

`test_extraction_PNG2TXT.py` is a ChromoPix inference script. It does not read labels embedded in an image. It loads the ChromoPix checkpoint, queries all 24 chromosome classes, extracts connected components from the predicted masks, and writes normalized polygons with confidence scores.

The Mask R-CNN and YOLO prediction TXT files must be produced by their own inference pipelines and placed in the component prediction directories before running the common evaluator.

## Expected project layout

All default paths are relative to the Chromo World project root, detected from the location of this README's sibling scripts. No H100-specific or user-specific absolute paths are required.

```text
ChromoWorld/
├── Analyze/
│   ├── README.md
│   ├── test_extraction_PNG2TXT.py
│   ├── test_robustness_TXTPNG.py
│   ├── test_baseline_TXT2CSV.py
│   ├── test_visualize_TXT2PNG.py
│   └── results/
│       ├── predictions/
│       ├── robustness/
│       ├── reports/
│       └── visualizations/
├── release/
│   ├── Dataset/2D/
│   │   ├── train/image/
│   │   ├── train/label/
│   │   ├── test/image/
│   │   └── test/label/
│   ├── Code/ChromoPix/
│   └── Checkpoint/ChromoPix/
└── ...
```

The default ChromoPix inference configuration expects:

```text
release/Code/ChromoPix/
release/Code/ChromoPix/dinov3/
release/Checkpoint/ChromoPix/chromopix.pth
release/Checkpoint/ChromoPix/dinov3_vitb16.pth
release/Dataset/2D/test/image/
release/Dataset/2D/test/label/
```

The required DINOv3 source subset is bundled at `release/Code/ChromoPix/dinov3/`. If using another DINOv3 checkout or weight location, pass `--dinov3_repo` and `--model_weights` explicitly. If ChromoPix is stored elsewhere, pass `--chromopix_code` and `--checkpoint` explicitly.

## Dependencies

The evaluator only needs NumPy, Pillow, and tqdm. The robustness and visualization scripts additionally need OpenCV. ChromoPix extraction needs PyTorch, torchvision, NumPy, OpenCV, Pillow, tqdm, the bundled DINOv3 source, and the ChromoPix code. The DINOv3 license is included under `release/Code/ChromoPix/dinov3/`.

For example:

```bash
python -m pip install numpy pillow tqdm opencv-python
```

Install a PyTorch/torchvision pair built for the target machine's CUDA version, then install the ChromoPix dependencies. A separate DINOv3 checkout is not needed for the default workflow.

## Figure 2 reproduction

### 1. Generate ChromoPix predictions

Run this command from the Chromo World project root:

```bash
python Analyze/test_extraction_PNG2TXT.py
```

The default output is:

```text
Analyze/results/predictions/ChromoPix/
```

For a custom checkpoint or dependency location:

```bash
python Analyze/test_extraction_PNG2TXT.py \
  --image_path release/Dataset/2D/test/image \
  --out_path Analyze/results/predictions/ChromoPix \
  --checkpoint release/Checkpoint/ChromoPix/chromopix.pth \
  --chromopix_code release/Code/ChromoPix \
  --dinov3_repo release/Code/ChromoPix/dinov3 \
  --model_weights release/Checkpoint/ChromoPix/dinov3_vitb16.pth \
  --gpu_id 0,1,2,3
```

The model uses 24 class queries. The default input resolution is 512 x 512, and connected components smaller than 20 pixels are omitted.

### 2. Add comparison component predictions

Place the predictions from the other component inference pipelines in these directories:

```text
Analyze/results/predictions/MaskRCNN/
Analyze/results/predictions/MaskRCNN_Banding/
Analyze/results/predictions/YOLOv11-Seg/
Analyze/results/predictions/YOLOv12-Seg/
Analyze/results/predictions/YOLOv11-Seg_Banding/
Analyze/results/predictions/YOLOv12-Seg_Banding/
```

Each directory should contain one TXT file per image. The file stem must match the ground-truth file stem. The evaluator accepts YOLO polygon TXT files and YOLO bounding-box TXT files.

### 3. Generate the Figure 2 report

```bash
python Analyze/test_baseline_TXT2CSV.py
```

The default evaluator compares all seven components above against:

```text
release/Dataset/2D/test/label/
```

The default report directory is:

```text
Analyze/results/reports/Fig2/
```

It contains:

```text
overall_stats.csv
tag_stats.csv
pr_curves/
```

To evaluate only ChromoPix:

```bash
python Analyze/test_baseline_TXT2CSV.py \
  --baseline_path '{"ChromoPix":"Analyze/results/predictions/ChromoPix"}' \
  --class_configurable '{"ChromoPix":true}' \
  --output_path Analyze/results/reports/Fig2/ChromoPix
```

## Figure 3 reproduction

Figure 3 uses synchronized transformations of the test images and polygon labels, followed by ChromoPix inference and the same TXT evaluator.

### 1. Create robustness inputs

```bash
python Analyze/test_robustness_TXTPNG.py
```

The default output is:

```text
Analyze/results/robustness/
├── image/
└── label/
    ├── Banding/
    ├── FISH/
    └── Other/
```

For every input image, the script creates:

- `RRRF`: random rotation and flipping
- `RS50`, `RS75`, `RS125`, `RS150`: centered scaling
- `CP25`, `CP50`, `CP75`, `CP100`: instance copy-paste
- `PSF15`, `PSF30`, `PSF45`, `PSF60`: PSF blur

The random seed defaults to `0` and should remain fixed for reproducibility.

### 2. Run ChromoPix on robustness images

```bash
python Analyze/test_extraction_PNG2TXT.py \
  --image_path Analyze/results/robustness/image \
  --out_path Analyze/results/predictions/ChromoPix_Robustness
```

Use the same checkpoint and dependency overrides shown in the Figure 2 section when the defaults are not available.

### 3. Generate the Figure 3 report

```bash
python Analyze/test_baseline_TXT2CSV.py \
  --gt_path Analyze/results/robustness/label \
  --baseline_path '{"ChromoPix_Robustness":"Analyze/results/predictions/ChromoPix_Robustness"}' \
  --class_configurable '{"ChromoPix_Robustness":true}' \
  --output_path Analyze/results/reports/Fig3
```

The resulting `tag_stats.csv` contains the condition tags used for the scale, copy-paste, PSF, rotation/flip, and cell-line analyses.

## Qualitative visualizations

Render ChromoPix predictions as PNG overlays with:

```bash
python Analyze/test_visualize_TXT2PNG.py
```

The default output is:

```text
Analyze/results/visualizations/ChromoPix/
```

To render ground-truth labels or another component, override `--label_path`, `--image_path`, and `--output_path`:

```bash
python Analyze/test_visualize_TXT2PNG.py \
  --label_path release/Dataset/2D/test/label \
  --image_path release/Dataset/2D/test/image \
  --output_path Analyze/results/visualizations/GT
```

The default visualization contains the original image, class text, masks, bounding boxes, and contours. Use `--subpic_type` to customize the panels and `--invert_colormap` when the image convention requires inversion.

## Evaluation conventions

The common evaluator uses the following conventions:

- Predictions and ground truth are matched by image basename.
- Classes must match during greedy one-to-one matching.
- Bounding-box matching uses IoU >= 0.5.
- Precision and recall are reported at confidence thresholds 0.5 and 0.75.
- `mAP` is AP at IoU 0.5.
- Polygon Dice and IoU are rasterized at 1024 x 1024.
- `tag_stats.csv` derives tags from underscore-separated filename stems.
- For ChromoPix, `class_configurable=true` removes predictions whose class is absent from that image's ground truth, matching the category-conditioned query protocol.
- For detector-style components, `class_configurable=false` keeps extra predicted classes and counts them as false positives.

Keep the same image-label stems and the same transformation seed when reproducing the manuscript figures.

## Path portability

All default paths are project-relative strings. The scripts resolve them against the project root inferred from `Analyze/`. Absolute paths are accepted as explicit command-line overrides, but they are not required and should not be committed to this repository.
