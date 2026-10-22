# Datasets: 2D, 3D and 4D

Datasets are hosted in [Drive / Dataset](https://drive.google.com/drive/folders/1ou6-J-qmmIMVm7H0XyIVlY5touqYBcWh), not embedded in Git.
Links and inventories were checked on 2026-10-22. **The current files require
Drive access**: sign in with an authorized account or request access.
Public/anonymous downloads are not enabled. No new data license is implied.

## 2D: chromosome images and polygon annotations

Download [2D.zip — 3,614,934,424 bytes (3.61 GB)](https://drive.google.com/file/d/1uiI7PR-H7DH77jsTPzEdkelMbMvmqB6x/view?usp=drivesdk)
from the [2D folder](https://drive.google.com/drive/folders/1w5QHMFfrtndwA6Tc5hwPjjWYC3A1Lm9_).
From the repository root:

```bash
unzip /path/to/2D.zip -d Dataset
```

Actual extracted layout, excluding macOS metadata:

```text
Dataset/2D/
├── train/
│   ├── image/                  10,735 PNG images
│   └── label/
│       ├── Banding/             4,294 TXT annotations
│       └── FISH/                6,441 TXT annotations
└── test/
    ├── image/                   2,945 PNG images
    └── label/
        ├── Banding/             1,316 TXT annotations
        └── FISH/                1,629 TXT annotations
```

13,680 image/annotation pairs in total. TXT lines contain a class index and
normalized polygon coordinates (`class x1 y1 x2 y2 …`), not raster masks.
Match image basenames against both label subfolders; labels are not directly
under `label/`. Ignore the existing ZIP's `__MACOSX`, AppleDouble and `.DS_Store`
entries. Robustness, miscellaneous and visualization assets are excluded.
See [ChromoPix usage](../Code/ChromoPix/README.md).

## 3D: matched DAPI/Tag volumes

Download [3D.tar.gz — 37,556,932,777 bytes (37.56 GB)](https://drive.google.com/file/d/1EoJxi3dP494kBQNJnXOkgXqhCznqACgo/view?usp=drivesdk)
from the [3D folder](https://drive.google.com/drive/folders/1DOW0HCiG4EysoiMDSwSTegUNwnxmM661).
This is the existing archive formerly named `DAPI_Tag_matched_1333pairs.tar.gz`;
the Drive download name is now `3D.tar.gz`.

The archive was packaged with `dapi` and `tag` as its top-level members
(confirmed from the original packaging command; the 37.56 GB iCloud placeholder
was not fully re-downloaded for this documentation update):

```bash
mkdir -p Dataset/3D
tar -xzf /path/to/3D.tar.gz -C Dataset/3D
```

```text
Dataset/3D/
├── dapi/    1,333 matched DAPI TIFF volumes
└── tag/     1,333 matched Tag TIFF volumes
```

Folder names are lowercase. The historical packaging excluded the unmatched
`dapi/id3_MCF-10F_dCas9_00341.tif` and
`tag/id3_MCF-10F_dCas9_00799.tif`. Pair by corresponding basename, not by
position in a directory listing.

The package contains the matched 1,333-pair DAPI/Tag selection for volumetric
analysis. Tag denotes fluorescence, not a ground-truth instance label. Do not
assume separately exported GT segmentation masks are included. See
[Fig5 volume analysis](../Code/Analyze/fig5/README.md#3d-masks-and-descriptors)
for inference and descriptors; read actual voxel calibration from each TIFF.

## 4D: live-cell movies, predicted masks and tracks

Distribution target: [4D folder](https://drive.google.com/drive/folders/1XMFZ6CrQIcQmX7JdBjPRx-hIvt5u_fDB).
**Upload pending: this folder is not yet a complete dataset download.**
The local archive passed CRC/member-path/SHA-256 checks, but the connector's
500 MiB transfers timed out before reaching Drive. The folder's README and
checksums alone are not the dataset; do not attempt extraction until all
parts (or a complete replacement ZIP) are present and verified.
The prepared 12,167,747,775-byte ZIP can be distributed as **24 binary parts**
(`4D.zip.001` through `4D.zip.024`, at most 500 MiB each), because the upload
connector accepts at most 512 MiB per file. Once uploaded, download every part and
`SHA256SUMS`; do not unzip the individual parts.

The package snapshots the finalized batch release on 2026-10-22, including
integrated track repair: **55 samples** across `chr3` (4), `chr18` (10),
`others` (26), and `patients` (15). It contains **510 TIFFs, 480 GIFs and
56 MP4s**, totaling **29,088,698,386 bytes before ZIP compression**. These are
multiple products per movie, not 510 independent samples. Hidden backups,
debugging artifacts, standalone repaired folders and candidate-selection
experiments are excluded.

```bash
# Run in the directory containing all 24 parts (macOS/Linux):
cat 4D.zip.[0-9][0-9][0-9] > 4D.zip
shasum -a 256 -c SHA256SUMS
unzip /path/to/4D.zip -d Dataset
```

The reconstructed ZIP SHA-256 is
`efa75b0764b4b111d447f00b7a786cac09c96a7500d9f418d3b65e3a384570eb`.
CRC integrity and all 1,046 member paths were checked before splitting.

The ZIP has a single `4D/` root; its local source folder was named `release/`:

```text
Dataset/4D/
├── image/
│   ├── raw/
│   │   ├── {chr3,chr18,others,patients}/*.tif
│   │   └── gif/{chr3,chr18,others,patients}/*.gif
│   ├── raw_pre/
│   │   ├── tiff/{chr3,chr18,others,patients}/*.tif
│   │   └── gif/{chr3,chr18,others,patients}/*.gif
│   └── raw_pre_affine/
│       ├── tiff/{chr3,chr18,others,patients}/*.tif
│       └── gif/{chr3,chr18,others,patients}/*.gif
├── mask/
│   ├── chimp/                       40 non-patient comparison masks
│   ├── cpsam/                       40 non-patient comparison masks
│   ├── chrp3d/                      15 patient raw model predictions
│   ├── chrp3d_post/                 55 native postprocessed masks
│   ├── chrp3d_affine_post/          55 registered postprocessed masks
│   ├── chrp3d_post_tracked/         55 native tracked masks
│   └── chrp3d_affine_post_tracked/  55 registered tracked masks
└── mp4/
    ├── {chr3,chr18,others,patients}/*.mp4
    └── patients/patients_all_samples_overview.mp4
```

Braces denote separate category folders, not literal folder names. Each mask
stage stores TIFFs in category subfolders and previews in `gif/<category>/`.
`chimp`/`cpsam` have no patients; `chrp3d` currently has only patients.
Each image stage has 55 DAPI and 10 available Tag TIFFs; GIF previews cover
the 55 DAPI movies, not Tag channels.

### Processing stages and interpretation

- `raw`: cropped intensity data, not necessarily the microscope's uncropped
  field of view. Patient movies use a shared all-time XYZ bounding box.
- `raw_pre`: restoration and background removal, without artificial
  continuity enhancement.
- `raw_pre_affine`: preprocessed and registered intensity movies.
- `chrp3d_post`: ChromoPose 3D segmentation plus native-grid postprocessing.
- `chrp3d_affine_post`: segmentation on registered images, then postprocessing;
  not simply interpolation of native-coordinate labels.
- `*_tracked`: temporal instance IDs with evidence-based repair. IDs are not
  chromosome-number classes or validated identities, and are not shared
  across samples or native/registered workflows.

Read TIFF axes (movie time/Z/Y/X) and embedded voxel calibration rather than
assuming isotropy. Intensity TIFFs differ from integer instance-label TIFFs;
label zero means background. **Masks and tracks are algorithm predictions,
not ground truth**. Native and registered grids can differ. Some legacy
image/mask basenames differ; do not join modalities by alphabetical order.
DAPI/Tag filename suffixes identify available channels, not chromosome identity.

GIF previews are 6 seconds, fixed front/back, without scale bars or 24-class
color buckets; instance masks use opaque shaded colors. The 55 sample MP4s
are 6-second four-panel comparisons with scale bars: top-left `raw_pre` front
MIP; top-right native tracked mask; bottom-left `raw_pre_affine` front MIP;
bottom-right registered tracked mask. One extra MP4 summarizes 15 patients.
Use TIFFs, not rendered previews, for quantitative analysis.

See the [Fig5 workflow](../Code/Analyze/fig5/README.md#4d-chromopose-pipeline).
The external [Chimps benchmark](../docs/downloads.md#chimps--external-paired-imagemask-benchmark)
is separate from this live-cell release.

## Scope and reproducibility

Weights are distributed separately: [model downloads](../docs/downloads.md).
These packages do not supply historical ChromoLift paired seven-feature
records/image conditions or a fitted ChromoID gallery. A dataset download
does not imply complete historical retraining is reproducible.

`estimate_archive_size.py` and `archive_size_estimate.json` retain the older
2D compression estimate; use the measured download sizes above instead.
No dataset binaries are committed to Git.
