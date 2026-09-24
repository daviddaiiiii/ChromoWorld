# Fig4 instance descriptor pipeline

The Fig4 workflow stores one structured JSONL record per image or volume.
Each record keeps metadata together with a list of instance-level masks and
features. Flat CSV files remain optional exports for legacy plotting code.

```text
2D processed image + polygon TXT
        |
        v
get-2ddescriptors_TXT2JSON.py
        |
        v
qc-sample_JSON2JSON.py
        |
        v
QC-passed 2D JSONL

3D volume TIFF
        |
        v
get-3dmask_TIFF2TIFF.py
        |
        v
3D instance-label TIFF
        |
        v
get-3ddescriptor_TIFF2JSON.py
        |
        v
3D instance-feature JSONL
```

## 2D raw descriptors

The formal image is the processed image used by the Fig2-3 PNG-to-TXT
pipeline, not raw DAPI. Polygon TXT files can be GT
(`class x1 y1 ...`) or prediction
(`class confidence x1 y1 ...`) labels.

```bash
python get-2ddescriptors_TXT2JSON.py \
  --image-dir /path/to/processed/image \
  --label-dir /path/to/prediction_txt \
  --label-role pred \
  --output-jsonl /path/to/fig4/raw_features.jsonl \
  --output-csv /path/to/fig4/raw_features.csv \
  --with-texture
```

The JSONL record contains `image_basename`, image and label paths,
`label_role`, `label_set_id`, and a nested `instances` list. Each instance
contains class/confidence metadata, the raw instance mask path, morphology,
intensity, and optional texture descriptors. Raw `uint16` instance-ID masks
and `descriptor_report.json` are written beside the output.

## 2D QC and robust normalization

QC reads only the raw JSONL; it never rereads TXT files.

```bash
python qc-sample_JSON2JSON.py \
  --input-jsonl /path/to/fig4/raw_features.jsonl \
  --output-jsonl /path/to/fig4/features_qc.jsonl \
  --output-csv /path/to/fig4/features_qc.csv \
  --min-instances-per-image 20 \
  --max-instances-per-image 60
```

It applies hard validity bounds, optional confidence/overlap/border rules,
optional robust MAD outlier filtering, and optional image-level instance-count
filtering. The default normalization group is each image and label role. Raw
features are retained and normalized columns use `__rel_pct` and
`__robust_z`. An audit JSONL, optional audit CSV, QC report, and plots are
also produced.

## 3D Cellpose-SAM masks

The 3D segmenter consumes native 3D TIFF stacks. The stack is treated as ZYX;
the OpenChromo TIFF files whose metadata reports `QYX` are handled as the same
three-dimensional ZYX array. Native resolution is used by default. Optional
inference resizing is returned to the native voxel grid with nearest-neighbour
interpolation.

```bash
python get-3dmask_TIFF2TIFF.py \
  --input /path/to/3d/raw/dapi \
  --output-dir /path/to/3d/mask/cpsam \
  --model /path/to/checkpoints/cpsam \
  --gpu-id 0
```

The output is one compressed integer label TIFF per input, named
`<basename>_mask.tif`, plus `segmentation_report.json`. Labels are reindexed
from 1 in descending instance volume order. By default, objects smaller than
2000 voxels or with volume-to-bounding-box ratio below 0.1 are removed; use
`--no-filter` to keep every Cellpose label.

The default Cellpose-SAM settings follow the existing ChromoWorld 3D path:
`do_3D=True`, flow threshold 0.1, cell-probability threshold 1.0,
`min_size=2000`, and `flow3d_smooth=3`. All are command-line parameters.

## 3D descriptors

The descriptor stage pairs each volume TIFF with a label TIFF having the same
basename, `<basename>_mask`, or `<basename>_shape` suffix.

```bash
python get-3ddescriptor_TIFF2JSON.py \
  --volume /path/to/3d/raw/dapi \
  --mask /path/to/3d/mask/cpsam \
  --output-jsonl /path/to/fig4/features_3d.jsonl \
  --voxel-size-zyx 0.5 0.1 0.1
```

The spacing order is Z, Y, X. If spacing is not supplied, the default is
`1 1 1`, so physical fields are in voxel units rather than calibrated units.
The JSONL record stores the volume shape, axes, spacing, mask metadata, and
one instance object per positive mask label.

3D descriptors include normalized and calibrated volume, centroid, bounding
box, equivalent diameter, surface area, sphericity, convex-hull solidity,
inertia-based elongation, principal-axis spread, 3D skeleton length and
topology, tortuosity, border contact, and intensity statistics from the paired
volume. `descriptor_3d_report.json` records skipped pairs and instance totals.

The 3D descriptors operate on the saved instance mask only; they do not
rerun segmentation or infer chromosome identity.

## 24-class UMAP visualization demo

The UMAP plotting stage reads the saved embedding rather than refitting UMAP.
The H100 filtered UMAP output contains `combined_embedding`,
`combined_class_id`, and `combined_split` arrays in one NPZ file.  Render a
balanced 24-class overview with:

```bash
python visualize-umap_24class_NPZ2PNG.py \
  --embedding-npz /path/to/umap_embedding.npz \
  --space combined \
  --split both \
  --output-dir /path/to/fig4/umap_24class_demo
```

Use `--space banding` or `--space fish` for the modality-specific embedding,
and `--split pred` or `--split gt` to show one label source.  By default,
`--max-points 100000` samples approximately equally within each chromosome
class while retaining the full 24-class legend; use `--max-points 0` to plot
all points.

For a compact marker-chromosome view, highlight chr1, chr5, chr13, X, and Y
while leaving the other classes as a gray context:

```bash
python visualize-umap_24class_NPZ2PNG.py \
  --embedding-npz /path/to/umap_embedding.npz \
  --space combined \
  --split both \
  --highlight-classes 1,5,13,X,Y \
  --output-dir /path/to/fig4/umap_marker_demo
```

The demo writes both PNG and vector SVG figures plus a JSON manifest:

```text
umap_<space>_<split>_24class.png
umap_<space>_<split>_24class.svg
umap_<space>_<split>_24class.json
```
