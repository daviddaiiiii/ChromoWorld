# Pretrained checkpoints

```
ChromoPix/chromopix.pth
ChromoPhy/ChromoPose/last.pth
ChromoPhy/ChromoPose/training_run/{args.json,train.csv,test.csv}
ChromoPhy/ChromoLift/lifter_final.pth
Outsource/DINOv3/dinov3_vitb16.pth
Outsource/CPSAM/cpsam
Outsource/FastCHIMP/CARE/{config.json,weights_best.h5}
Outsource/FastCHIMP/EmbedSeg/best_iou_model.pth
```

ChromoID is a gallery-based retrieval component, not a separate neural model.
Its optional gallery belongs in `Artifacts/ChromoID/gallery_7d_cache.npz` and
is currently absent; rebuild it with the documented ChromoID training command.
The figure UMAP assets are not replacements for that gallery.

Large weights stay outside GitHub. Publish project weights with versioned
download links/checksums separately; check redistribution permissions for
third-party weights, otherwise link to their upstream sources. Maintain this
layout after downloading so the Analyze/model defaults resolve correctly.
See the subfolder READMEs for provenance and model-specific information.
