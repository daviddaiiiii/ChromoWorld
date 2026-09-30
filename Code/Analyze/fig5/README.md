# Fig5: volume and movie analysis

Run commands from the release root. Install the Analyze dependencies for
descriptors/plots and ChromoPose dependencies for CPSAM and movie inference.

## 3D masks and descriptors

```bash
python Code/Analyze/fig5/get-3dmask_TIFF2TIFF.py \
  --input /data/3D/DAPI --output-dir /output/3D/mask --gpu-id 0
python Code/Analyze/fig5/get-3ddescriptor_TIFF2JSON.py \
  --volume /data/3D/DAPI --mask /output/3D/mask \
  --output-jsonl /output/3D/features.jsonl --voxel-size-zyx 0.5 0.1 0.1
```

The examples' 0.5/0.1/0.1 spacing is illustrative; supply the actual calibrated
Z/Y/X spacing. Without it, the default unit grid yields voxel-unit physical
measurements. The segmenter accepts native ZYX (or equivalent QYX) TIFFs,
uses Checkpoint/Outsource/CPSAM/cpsam, and restores optional inference resizing
to native shape. Defaults are do_3D, flow threshold 0.1, cell probability 1.0,
min size 2000, flow3d_smooth 3, and the historical extent filter.

The descriptor exporter pairs each volume with its integer instance mask;
it retains every label and records historical_feature_eligible for QC.
The historical 3d_semanticv7 equivalents are volume_fraction (norm_area),
elongation (eccentricity), solidity, sphericity (circularity), and tortuosity.
The normalization denominator excludes labels below 20 voxels; historical
eligibility is the 0.1%–20% fraction range. Historical dimensionless shape
calculations use voxel coordinates, while calibrated measurements use spacing.

Additional 2D-compatible descriptors cover inertia/axes, extent, compactness,
convexity, topology, skeleton/radii, border contact, and intensity.
--with-texture adds five masked GLCM summaries over 13 3D directions.
Historical MIP features require a separate 2D projection-extraction stage;
the 3D exporter does not substitute for that stage.

Dataset/3D currently contains the matched DAPI/Tag archive. Its name records
the 1,333-pair selection; unpack and inspect its internal paths before choosing
--input. No separate 3D GT-mask export is present in the release folder.

## 4D ChromoPose pipeline

```bash
python Code/Analyze/fig5/get-4dmask_TIFF2TIFF.py \
  --input /data/raw_movies --output-dir /output/Fig5/4D --gpu-ids 0
```

Each sample receives a basename directory with raw, preprocess, affine,
segment, post, and track TIFFs plus pipeline/stage reports.
Affine estimates rigid motion from DAPI before segmentation. Segment/post/track
share the affine image's padded/cropped TZYX grid and calibrated spacing.
ChromoPose resizes each frame to 64×256×256 internally and restores instance
labels to the affine grid with nearest-neighbor interpolation. It uses
Checkpoint/ChromoPhy/ChromoPose/last.pth and training_run/args.json.

--max-files 1 and --stop-after affine support a small check. --anchors supplies
reviewed representative_t values. Completed stages are checked before reuse;
changed sources or settings are rejected. The legacy mask-assisted affine
route remains an explicit historical option in chromopose/affine.py.

Dataset-specific keyframe renderers require an external movie root and
representative-frame manifest. Those inputs and Chimps training movies are
not currently in Dataset; the retained demo_outputs illustrate past outputs.

## Redraw saved 3D/MIP/lifted UMAPs

```bash
for space in 3D MIP LiftedMIP Lifted2D; do
  python Code/Analyze/fig5/visualize-umap_NPZ2PNG.py \
    --embedding-npz "Artifacts/fig5/${space}_umap_embed.npz" --space "$space"
done
```

Artifacts/fig5 contains 97,305 points in each 3D/MIP space, 113,423 in
LiftedMIP, and 711,541 in Lifted2D. live_umap_meta.json records provenance.
PNG, SVG, and a rendering manifest go to Code/Analyze/fig5/demo_outputs/chromophy.
For 3D/MIP, --label-source canvas is the default and mip selects the other
stored reference labels. The seeded default samples up to 100,000 points;
--max-points 0 draws all coordinates.

These five-feature plotting assets reproduce saved UMAP views without
rerunning the models. They do not contain the seven-feature conditioned
gallery needed for ChromoID inference or the original ChromoLift training
inputs. See ../../ChromoPhy/README.md for building those stages.
