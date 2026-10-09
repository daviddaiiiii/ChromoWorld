# Fig4: fixed-cell morphology

Run commands from the release root. Descriptor extraction reads the processed
images used by ChromoPix and paired GT/prediction polygon TXT files. Saved UMAP
assets are in Artifacts/fig4, outside the Code tree.

## Extract and filter descriptors

```bash
python Code/Analyze/fig4/get-2ddescriptors_TXT2JSON.py \
  --image-dir Dataset/2D/test/image \
  --label-dir Code/Analyze/results/predictions/ChromoPix \
  --label-role pred --with-texture \
  --output-jsonl Code/Analyze/results/Fig4/raw_features.jsonl
python Code/Analyze/fig4/qc-sample_JSON2JSON.py \
  --input-jsonl Code/Analyze/results/Fig4/raw_features.jsonl \
  --output-jsonl Code/Analyze/results/Fig4/features_qc.jsonl \
  --min-instances-per-image 20 --max-instances-per-image 60
```

Each JSONL record corresponds to an image and contains its basename, input
paths, label role, and nested instances. Instances contain mask/class metadata,
morphology, intensity, and optional texture features. Integer instance-mask
TIFFs and a descriptor report are written with the export. Optional CSVs are
requested with --output-csv.

QC reads the JSONL. It applies validity bounds and optional confidence,
overlap, border, image-count, and MAD filters. Features are normalized within
each image/label-role group with relative percentages and robust z scores;
raw feature values remain in the records. Audit records describe removals.

The current enriched descriptors do not themselves recreate the published
five-feature embeddings. Historical fitting used normalized size,
eccentricity, solidity, circularity, and tortuosity, with the historical QC
and fitting settings recorded in Artifacts/fig4/fixed_umap_meta.json.

## Redraw the published UMAPs

```bash
python Code/Analyze/fig4/visualize-umap_NPZ2PNG.py \
  --embedding-npz Artifacts/fig4/Banding_umap_embed.npz --space Banding
python Code/Analyze/fig4/visualize-umap_NPZ2PNG.py \
  --embedding-npz Artifacts/fig4/FISH_umap_embed.npz --space FISH
python Code/Analyze/fig4/visualize-umap_NPZ2PNG.py \
  --embedding-npz Artifacts/fig4/Combine_umap_embed.npz --space Combine
```

One shared renderer serves Fig4 and Fig5. The files contain 483,738 Banding,
227,803 FISH, and 711,541 combined points, including GT/Pred split labels.
Use --split gt or pred, --highlight-classes 1,5,13,X,Y, or --max-points 0
to plot all retained points. The default is a seeded class-stratified plot
of at most 100,000 points. PNG, SVG, and a rendering manifest are saved in
Code/Analyze/fig4/demo_outputs; --output-dir overrides the destination.

This route uses stored coordinates without refitting UMAP. Metadata alone
cannot generate coordinates. Historical filtered descriptor JSONL and image
feature NPY files are not bundled; they are additional inputs for rebuilding
the conditioned ChromoLift/ChromoID workflow.
