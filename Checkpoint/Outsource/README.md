# External checkpoints: official sources only

No external weight binaries are bundled, and no ChromoWorld Drive mirror is
provided. Obtain weights directly from the original model or application authors.
See [download and usage instructions](../../docs/downloads.md) for commands.

- **CPSAM:** [official Cellpose code](https://github.com/MouseLand/cellpose),
  [original cpsam download](https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam),
  [model selection guide](https://cellpose.readthedocs.io/en/latest/models.html).
  Save locally as `Checkpoint/Outsource/CPSAM/cpsam`; select `cpsam`, not newer
  `cpsam_v2`/DINO variants, for the baseline used here.
- **CARE:** [CSBDeep implementation](https://github.com/CSBDeep/CSBDeep).
  For chromosome images use the FAST-CHIMP authors' [weights](https://raw.githubusercontent.com/rstamatov/FAST-CHIMP/main/denoising%20models/my_model/weights_best.h5)
  and [matching configuration](https://raw.githubusercontent.com/rstamatov/FAST-CHIMP/main/denoising%20models/my_model/config.json).
- **EmbedSeg:** [official implementation](https://github.com/juglab/EmbedSeg)
  and FAST-CHIMP authors' [chromosome checkpoint](https://raw.githubusercontent.com/rstamatov/FAST-CHIMP/main/segmentation%20models/full_model/best_iou_model.pth).
- **VoxelMorph:** [official implementation](https://github.com/voxelmorph/voxelmorph)
  and FAST-CHIMP authors' [chromosome registration checkpoint](https://raw.githubusercontent.com/rstamatov/FAST-CHIMP/main/registration%20models/model_0.02_adjusted.h5).
  This is a legacy TensorFlow model, not a drop-in checkpoint for the newest
  PyTorch interface. The released ChromoWorld rigid affine stage needs no model.
- **DINOv3:** [official source](https://github.com/facebookresearch/dinov3),
  [Meta download-access form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
  Select the native ViT-B/16 LVD-1689M `.pth` and save locally as
  `Checkpoint/Outsource/DINOv3/dinov3_vitb16.pth` for ChromoPix.

For CARE, EmbedSeg, and VoxelMorph, follow the original
[FAST-CHIMP workflow](https://github.com/rstamatov/FAST-CHIMP) in a separate
checkout/environment; preserve its model/configuration paths. These three
external models are not loaded by the released ChromoPose pipeline.

Historical reference SHA-256 values (check downloaded bytes rather than
assuming that upstream `main` never changes):

```text
73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c  dinov3_vitb16.pth
e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2  cpsam
3572ceac0226f25be3750fbb68efb9dd8f6a99c27bf13527fd9c1bbb02fe0dd2  CARE/weights_best.h5
8ad074f97a4524ce478f25fd6a23fbb39abc64582ef555c82302c08f0ad4f60c  CARE/config.json
2ed23451e1b792682577dc7f77c97c49b5181cb5698a2c55a24e0b4a6e77a9b9  EmbedSeg/best_iou_model.pth
```
