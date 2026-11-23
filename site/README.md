# ChromoWorld site

First review build of the project website. The introduction and interactive
views use local research exports. The live examples with Tag summarize
manually reviewed candidates.

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

## Homepage film

Run `python scripts/make_intro.py` with Pillow, NumPy and imageio-ffmpeg
installed. This writes the video and poster using the microscopy exports
and the saved Fig. 4 and Fig. 5 embeddings in the sibling
`release/Artifacts` directory. The morphology example in `src/data/overview`
records the FISH annotation index, probe-channel registration and matching
global row in the archived UMAP. The comparison aligns class means from
the separately fitted G-banding and FISH maps to the joint map. The film
uses the saved Lifted2D embedding to visualize the 3D descriptor atlas.
The ChromoPose, ChromoLift and ChromoID shots share the Fig. 5f movie and
tracked instance recorded in `src/data/overview/fig5-live/source.json`.

`src/data/overview/patient-source.json` records the patient clip and
its frame matches to the paper presentation. The displayed daughter
counts are imaging-based inferences. The micromanipulation sequence is
a schematic, followed by cropped experimental evidence from Fig. 6.
For a short rendering of the main transitions, run
`python scripts/make_intro.py --preview /tmp/chromoworld-transitions.mp4`.

## GitHub Pages

Publish this folder as the source root of the desired GitHub Pages repository.
Enable GitHub Actions as the Pages source, then run the included **GitHub
Pages** workflow. It sets Astro's base path from the Pages configuration.
The workflow is manual so review builds can be checked before publication.

The release data links may require access. Verify downloadable assets and
publication permissions before a public deployment.
