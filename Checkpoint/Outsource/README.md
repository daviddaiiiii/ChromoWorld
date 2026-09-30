# External model weights

These are third-party pretrained or baseline checkpoints, kept separate from
the project's ChromoPix and ChromoPose weights. Each model is stored once in
this release. Do not rename the files without updating the corresponding
inference defaults.

| Model | Release file | Source used here | SHA-256 |
| --- | --- | --- | --- |
| DINOv3 ViT-B/16 | `DINOv3/dinov3_vitb16.pth` | Previously `Checkpoint/ChromoPix/dinov3_vitb16.pth` | `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c` |
| Cellpose-SAM | `CPSAM/cpsam` | H100 `/share/project/daigaole/checkpoints/cpsam` | `e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2` |
| Fast-CHIMP CARE | `FastCHIMP/CARE/weights_best.h5` | `FAST-CHIMP-main/denoising models/my_model/weights_best.h5` | `3572ceac0226f25be3750fbb68efb9dd8f6a99c27bf13527fd9c1bbb02fe0dd2` |
| Fast-CHIMP EmbedSeg | `FastCHIMP/EmbedSeg/best_iou_model.pth` | `FAST-CHIMP-main/segmentation models/full_model/best_iou_model.pth` | `2ed23451e1b792682577dc7f77c97c49b5181cb5698a2c55a24e0b4a6e77a9b9` |

CARE also requires its adjacent `config.json` (SHA-256
`8ad074f97a4524ce478f25fd6a23fbb39abc64582ef555c82302c08f0ad4f60c`).
The Fast-CHIMP best weights were copied from the original external checkout;
that checkout was not modified. Other training snapshots such as
`weights_now.h5` and `checkpoint.pth` are different, not duplicate files, and
are intentionally not included in this release.

Check each upstream model's license and distribution terms before making a
public release of these weights.
