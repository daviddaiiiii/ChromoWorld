# ChromoPhy checkpoints

These are local runtime destinations, not embedded weight files. Download
the internal models from the flat [Drive ckpt folder](https://drive.google.com/drive/folders/1B4LLQiesGVCVG3aq-7846YpRV52jHJg3)
using the [filename mapping and commands](../../docs/downloads.md#chromoworld-weights).

- `ChromoPose/last.pth`: July 2026 ChromoPose3D 4D segmentation checkpoint.
  `ChromoPose/training_run/args.json` contains its inference configuration.
  Source: H100 `chromopose3D/outputs/chromopose_20260703_203720/ckpt/last.pth`.
  SHA-256: `ff676b53099068ea003fbe4437dbe4aa701112bd07e3652f844418ecb243c9c4`.

- `ChromoLift/lifter_final.pth`: 7D physical-feature 2D→3D flow model from
  H100 `outputs/chromophy_training_20260330_194834/`. Includes feature and
  image-condition scalers; used by the live UMAP and ChromoID gallery builder.
  SHA-256: `d12cc0c92a7b37f22fe2727d59a5b559692eaa7e4b149f7dc2f962d05b847f6e`.

ChromoID has no independent neural-network checkpoint. Its labeled 7D gallery
is not retained in this release. Rebuild it with the ChromoID training entry
point and place it at `Artifacts/ChromoID/gallery_7d_cache.npz`; it uses
ChromoLift's 3D scaler. The saved figure embeddings are not this gallery.

The third-party Cellpose-SAM model used for the 3D baseline must be downloaded
from its [official source](../Outsource/README.md) to `../Outsource/CPSAM/cpsam`.

The unrelated May 2026 flow-matching ChromoPose checkpoints are not part of
this current release or inference pipeline.
