# Checkpoint download destinations

**No weight binaries are bundled in this Git repository.** This directory
contains documentation and the small ChromoPose training/inference configuration.

## Internal models

The Drive [ckpt folder](https://drive.google.com/drive/folders/1B4LLQiesGVCVG3aq-7846YpRV52jHJg3)
contains only these three files:

- [chromopix.pth](https://drive.google.com/file/d/1LApL64GfdYvF2Bwt-9lQ79HCJSaLrCy6/view)
  → `Checkpoint/ChromoPix/chromopix.pth`.
- [chromopose.pth](https://drive.google.com/file/d/1jWKZ6M9lUZIjxc5bAWG1_7lMc4nFysio/view)
  → `Checkpoint/ChromoPhy/ChromoPose/last.pth`.
- [chromolift.pth](https://drive.google.com/file/d/1yDoa_QtB-FenOwULfTfg01frWOg9FMQ-/view)
  → `Checkpoint/ChromoPhy/ChromoLift/lifter_final.pth`.

The right-hand paths are existing CLI defaults, not bundled files. The
[download commands](../docs/downloads.md#chromoworld-weights) map the standardized
Drive filenames to these runtime paths. Explicit checkpoint CLI paths are also
supported. Access was checked while authenticated; public access is not assumed.

## External models

We do not redistribute or link project-hosted mirrors of CPSAM, CARE, EmbedSeg,
VoxelMorph, or DINOv3. Use the [official model links and instructions](Outsource/README.md)
and retain upstream terms. Optional local downloads remain ignored by Git.

ChromoID uses a fitted gallery, not an independent neural-network checkpoint.
Its optional `Artifacts/ChromoID/gallery_7d_cache.npz` is not bundled; rebuild it
with the ChromoID training entry point. Saved figure UMAP files are not a gallery.
