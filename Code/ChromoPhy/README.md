# ChromoPhy model development

Run examples from the release root. Model code lives here; pretrained neural
weights are under Checkpoint/ChromoPhy, fitted NPZ assets are under Artifacts,
and figure orchestration is under Code/Analyze.

## ChromoPose

chromopose/ provides movie preprocessing, image-only affine registration,
signal-guided mask postprocessing, and tracking. segmentation/ contains seven
core modules: dataloader, model, preprocess, train, evaluate, infer, visualize.
The preprocessor/evaluator consume paired Chimps images and GT masks;
deployment infer.py consumes an arbitrary calibrated 3D/4D TIFF.

The released checkpoint is Checkpoint/ChromoPhy/ChromoPose/last.pth, with
training_run/args.json. Inference uses 64×256×256, flow_scale 5, foreground
threshold 0.5, phase-two iterations 200, flow step 0.75, Cellpose follow_flows,
and no watershed refinement. Labels return to the input's native grid.
See [segmentation](chromopose/segmentation/README.md).

`chromopose/auto_crop.py` independently trims oversized original DAPI fields
or redundant registered canvases using a shared all-time XYZ box. It preserves
native pixels, every timepoint, calibrated spacing, and existing instance IDs;
aligned masks/Tag files can receive the same box with `--apply-to`. This is not
denoising, registration, segmentation or tracking repair. See the
[crop CLI and safety limits](../Analyze/fig5/README.md#standalone-automatic-crop).

```bash
python -m pip install -r Code/ChromoPhy/chromopose/requirements.txt
python Code/ChromoPhy/chromopose/segmentation/infer.py \
  --in_path /data/movie-DAPI.tif --out_path /output/masks --gpu_id 0
```

For the full raw-to-tracked workflow, use
Code/Analyze/fig5/get-4dmask_TIFF2TIFF.py. Legacy mask-assisted affine remains
an explicit historical route. Tag-specific maintenance utilities are not
included in the released DAPI movie pipeline.
Add `--auto-crop` to insert `crop` and `affine_crop` stages; without it the
existing six core stages remain unchanged. Use a new output directory when
switching pipeline settings/versions.

## ChromoLift

chromolift/ contains the checkpoint-compatible 7D model, training, inference,
and paired-descriptor evaluation. The released weights are
Checkpoint/ChromoPhy/ChromoLift/lifter_final.pth.
Training additionally requires historical JSONL and image-feature NPY files
that are not bundled. See [ChromoLift](chromolift/README.md).

## ChromoID

chromoid/train.py builds the lifted, labeled gallery; infer.py performs scaled
Euclidean weighted kNN retrieval; evaluate.py scores held-out reference labels.
It has no separate neural checkpoint. Its optional gallery belongs in
Artifacts/ChromoID/gallery_7d_cache.npz and is currently absent.
The five-feature Lifted2D UMAP asset cannot replace this 7D gallery.
See [ChromoID](chromoid/README.md).

The saved seven figure embeddings can be rendered independently with
Code/Analyze/fig4/visualize-umap_NPZ2PNG.py. Model training, gallery rebuilding,
and saved-coordinate plotting have different resource requirements.
