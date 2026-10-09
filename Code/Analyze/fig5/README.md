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

Chimps training movies are not currently in Dataset; the retained
demo_outputs illustrate past outputs.

## Fixed front/back TIFF-to-GIF visualization

```bash
python Code/Analyze/fig5/get-visualize-image_TIFF2GIF.py \
  --input-root /data/release/image --output-root /output/image \
  --folders raw raw_pre raw_pre_affine
python Code/Analyze/fig5/get-visualize-mask_TIFF2GIF.py \
  --input-root /data/release/mask --output-root /output/mask \
  --folders chrp3d_post chrp3d_affine_post chrp3d_post_tracked chrp3d_affine_post_tracked
```

Keep these two scripts together: the image renderer imports display/calibration
helpers from the sibling mask renderer. Both accept calibrated ZYX/TZYX TIFFs,
including singleton-channel movies. A group can contain TIFFs directly, under
chr3/chr18/others, or under tiff/chr3/chr18/others. Outputs go to
OUTPUT_ROOT/GROUP/gif/CATEGORY (flat inputs produce flat gif folders).
Hidden backups are excluded and the image CLI skips Tag-named TIFFs.

Default playback is six seconds across all source timepoints, with a fixed +Z
front view on the left and a -Z back view on the right (180 degrees around Y).
No rotating views or titles are used. Mask rendering selects visible surfaces
by depth, uses ordinary per-instance colors (not 24-class buckets), and adds
bounded surface shading. Equal IDs retain a base color but do not imply a
tracking identity unless the input itself is tracked.

Images use illustrative emission/absorption volume rendering and one shared
movie-wide contrast/gamma mapping; --projection mip selects MIP instead. These
are display effects, not quantitative fluorescence or changes to source data.
All Z planes participate, and source TIFFs are read-only.

Scale bars require calibrated TIFF length units, Z spacing, and X/Y resolution.
For an uncalibrated mask, --calibration-reference-root can supply a uniquely
matching *-DAPI TIFF on the exact same TZYX grid; matching shape alone does not
prove registration, so use only the corresponding image. Missing calibration
is an error, not an invented scale bar. Dependencies are in ../requirements.txt.
No JSON or debugging assets are produced by default; --report-json /output/audit.json
opts into a report. Concurrent workers should use separate report filenames.

## Redraw saved 3D/MIP/lifted UMAPs

```bash
for space in 3D MIP LiftedMIP Lifted2D; do
  python Code/Analyze/fig4/visualize-umap_NPZ2PNG.py \
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
