# Fig5 CPSAM visualization demo

`visualize-cpsam_3D_TIFF2PNG.py` shows the output of the Fig5 3D CPSAM stage.
It pairs a processed 3D DAPI TIFF with the integer-label TIFF written by
`get-3dmask_TIFF2TIFF.py`.  The output has four panels: DAPI maximum-intensity
projection, CPSAM instances over the DAPI projection, one DAPI z-slice, and
the corresponding CPSAM overlay.

The stack is interpreted as ZYX.  The MIP label at each XY pixel is selected
from the z-plane containing the brightest DAPI voxel, which keeps the overlay
linked to the visible signal.  CPSAM IDs are instance IDs, not chromosome
classes, so colours identify instances only.

Run it after generating a CPSAM mask:

```bash
python visualize-cpsam_3D_TIFF2PNG.py \
  --volume /path/to/processed_dapi.tif \
  --mask /path/to/cpsam_mask/processed_dapi_mask.tif \
  --output-dir /path/to/fig5/cpsam_demo \
  --alpha 0.45
```

The `--mask` file must have the same 3D shape as `--volume`.  Outputs are a
PNG panel and a JSON manifest containing shape, axes, selected slice, number
of instances, and mask voxel count.
