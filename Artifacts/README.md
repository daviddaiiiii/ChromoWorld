# Saved figure artifacts

Only saved UMAP NPZ assets and their fitting provenance are retained here.
Raw descriptor tables can be regenerated with the Analyze extraction/QC code.
These seven NPZ files total 39,732,462 bytes (39.73 MB).

| Figure | NPZ | Rows |
| --- | --- | ---: |
| Fig4 | Banding_umap_embed.npz | 483,738 |
| Fig4 | FISH_umap_embed.npz | 227,803 |
| Fig4 | Combine_umap_embed.npz | 711,541 |
| Fig5 | 3D_umap_embed.npz | 97,305 |
| Fig5 | MIP_umap_embed.npz | 97,305 |
| Fig5 | LiftedMIP_umap_embed.npz | 113,423 |
| Fig5 | Lifted2D_umap_embed.npz | 711,541 |

`fig4/fixed_umap_meta.json` and `fig5/live_umap_meta.json` record historical
fitting provenance. Preserve these with the NPZ files. The NPZs include saved
coordinates and row-aligned labels/indexes; some also contain historical 5D
features. They do not contain the 7D training/gallery inputs required by
ChromoLift or ChromoID. In particular, `Lifted2D` is not a ChromoID gallery.

From the release root:

```bash
python Code/Analyze/fig5/visualize-umap_NPZ2PNG.py \
  --embedding-npz Artifacts/fig4/Combine_umap_embed.npz
python Code/Analyze/fig5/visualize-umap_NPZ2PNG.py \
  --embedding-npz Artifacts/fig5/LiftedMIP_umap_embed.npz
```

The single renderer handles all seven assets, writes PNG/SVG/plot metadata
directly under `Code/Analyze/fig4/demo_outputs/` or
`Code/Analyze/fig5/demo_outputs/chromophy/`, and defaults
to deterministic stratified sampling of at most 100,000 points. Use
`--max-points 0` to render every point. It does not refit or rerun ChromoLift.
