# ChromoWorld site

The public project site includes the overview film, a fixed-cell atlas,
morphology maps and a live-cell movie explorer. The tagged examples show
manually reviewed chromosome candidates alongside the fluorescence measurement.

```sh
npm ci
npm run dev
npm run build
```

`public/assets` contains the browser-ready subsets. The preparation scripts
regenerate them from the study's local microscopy, candidate-mask and
embedding files. Full-resolution archives and model weights are separate.

## Fixed-cell atlas

`scripts/prepare_class_atlas.py` exports the fields, crops and reference contours.
`scripts/prepare_atlas_predictions.py PREDICTION_DIR public/assets/atlas/prediction --index public/assets/atlas/index.json`
packages the existing ChromoPix evaluation contours by matching the exact source
field basename. The divider clips the two contour layers over the same DNA image.
`scripts/curate_class_atlas.py TEST_DIR PREDICTION_DIR PREVIOUS_INDEX OUTPUT --report QUALITY_JSON`
selects up to six test fields per class and preparation by same-class mask IoU,
rejects strongly overlapping competing calls and writes an audit of the
previous and selected target matches. It needs OpenCV, NumPy and Pillow.

## Live 4D previews

`scripts/prepare_live_tracks.py RELEASE_DIR public/assets/compare/index.json public/assets/live/ordinary --renderer RENDERER_DIR`
renders front and back views of the ordinary native tracked movies using
`rounded_render.py`. A track ID maps to the same color across time and views.
The tagged Chr3 and Chr18 views come from the separate reviewed-candidate export
in `scripts/prepare_assets.py`.
Use the same command with `--collection patients` and `public/assets/live/patients`
as the output to render the patient tracking cohort from the matched release
TIFFs. The patient scale bars use calibrated OME-TIFF voxel dimensions.
`src/data/live-feature.json` selects the initial Normal view and the paired
homepage preview produced by `scripts/compose_previews.py`.

## Identity across states

Run `python scripts/prepare_state_maps.py ../Artifacts PATH_TO_FIGURE_EXPORTS`
from the site directory to regenerate the UMAP clouds, normalized centroids and
homepage preview. Fixed-cell maps use every saved Fig. 4 coordinate. The main
live-cell map uses the Fig. 5E cloud and vector centroids exported by
`prepare_intro_story.py`. The supplementary Lifted MIP view plots
`lifted_opt_mip_embedding` from `LiftedMIP_umap_embed.npz` and calculates each
center from the entries with the same `lifted_opt_mip_y` label. The published
ChromoPhy assignments appear in the main live-cell panel at `#chromophy`.

## Homepage film

The player first offers a 960 × 624 VP9/Opus WebM for streaming. The original
1280 × 832 H.264/AAC MP4 is available through the same player and the HD link.
Both use the same chapters and bilingual captions.

The opening karyotype is arranged from a single held-out, reference-annotated
G-banding field. `scripts/prepare_intro_story.py` extracts it and the Fig. 2/5
comparison panels from the original local datasets and figure exports. The
fixed-cell chapter shows the Fig. 2A preparation schematics, the Fig. 2B
metrics and the top row of Fig. 2D, where all three models train on G-banding
only and predict on the same held-out FISH image. The same-frame comparison
of FAST-CHIMP (EmbedSeg), Cellpose-SAM and ChromoPose is rendered from the
three released native masks for the Fig. S2c example.
The comparison is qualitative. The subsequent diagram explains the distance-
guided move toward the skeleton and convergence-based grouping along the
learned 3D flow.
The robustness chapter uses the held-out cell-line, scale, perturbation and
virtual FISH panels in Fig. 3A–D. The transition into the shape map follows
the same FISH-supported Chr1 instance from the virtual FISH field.
The CP and PSF bar values come from the original Fig. 3C chart SVGs, supplied
with `--fig3-chart-archive`. The film plots both charts on the same scale.
The ChromoID comparison uses the original point-cloud objects in Fig. 5A/B/E
and the matching vector centroid markers in Fig. 5C/E. The film connects
the markers in the paper's chromosome order. To regenerate those assets,
extract the original raster layers with `pdfimages -png FIG5 FIG5_RASTERS/fig5`
and the vector markers with `pdftocairo -svg FIG5 FIG5.svg`.

Run `python scripts/make_intro.py` with Pillow, NumPy and imageio-ffmpeg
installed. This writes the video and poster using the microscopy exports
and the saved Fig. 4 and Fig. 5 embeddings in the sibling
`release/Artifacts` directory. The morphology example in `src/data/overview`
records the FISH annotation index, probe-channel registration and matching
global row in the archived UMAP. The comparison aligns class means from
the separately fitted G-banding and FISH maps to the joint map. The lifted
atlas uses the original Fig. 5D object and the instance count from Lifted2D.
The tracking, ChromoLift and ChromoID shots share the Fig. 5f movie and
tracked instance recorded in `src/data/overview/fig5-live/source.json`.

`src/data/overview/patient-source.json` records the patient sequence and
its frame matches to the paper presentation. The patient scene uses the
native-grid context and reviewed candidate TIFFs. The displayed daughter
counts are imaging-based inferences. The micromanipulation sequence uses
original embedded presentation artwork. The blots use the embedded raster
objects and the plots use original Fig. 6 artwork.
The clinical follow-up uses the target-lesion measurements and clinical
observations reported in the v15 manuscript Results. The clinical team
selected treatment independently.
For a short rendering of the main transitions, run
`python scripts/make_intro.py --preview /tmp/chromoworld-transitions.mp4`.

`scripts/prepare_intro_media.py --help` lists the local source paths for
regenerating all film layers. The source folder starts with the specimen and
registration metadata and the reference anchor mask from `src/data/overview`.
Generate Fig. 2, 3 and 6 PNG plates with `pdftoppm -scale-to 4800 -png -singlefile`
on the original figures, and extract `image259.png`, `image264.png` and
`image267.png` directly from the presentation's `ppt/media` folder.
The film layers use lossless storage. Presentation raster objects retain
their native detail. `CHROMOWORLD_FILM_MEDIA` can point the renderer at a
separate source folder for review.

For the G-banding transfer benchmark, pass the native blue-channel acquisition
(`2D/test/miscs/dapi/id01_HFF_FISH_00023.png`) as `--transfer-dapi`. The
renderer checks its registration against the archived evaluation image, then
places the original model overlays on a wider view of the same chromosome spread.
The training-view close-up uses the original Chr1 polygon with space around its
outline. The DNA, probe and reference layers share the same image coordinates.

### Voice, captions and music

The published MP4 contains English narration and a music bed mastered for
the web. The player offers English and Simplified Chinese WebVTT tracks.
The timestamped source script is `scripts/intro_narration.json`.

To rebuild, render the silent movie with
`python scripts/make_intro.py --output-dir /tmp/chromoworld-silent`.
Render `scripts/intro_credits.tsx` as the `ChromoWorldCredits` composition
in Remotion. With Kokoro ONNX, SciPy and imageio-ffmpeg available, run
`python scripts/mix_intro.py SILENT_MP4 CREDITS_MP4 SCORE_MP3 KOKORO_ONNX VOICES_BIN OUTPUT_DIR`.
This writes the final MP4 and both subtitle tracks.

With the existing mastered WAV, `make_intro.py --credits CREDITS_MP4
--audio-master MASTER_WAV --output-dir OUTPUT_DIR` encodes directly from
the composed frames at higher quality in one pass.

Music is an excerpt of [“A Kind Of Hope” by Scott Buckley](https://www.scottbuckley.com.au/library/a-kind-of-hope/),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The excerpt is faded and mixed under narration. Speech was synthesized
locally with [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M).

## GitHub Pages

Publish this folder as the source root of the desired GitHub Pages repository.
Enable GitHub Actions as the Pages source, then run the included **GitHub
Pages** workflow. It sets Astro's base path from the Pages configuration.
The workflow is manual so review builds can be checked before publication.

The release data links may require access. Verify downloadable assets and
publication permissions before a public deployment.
