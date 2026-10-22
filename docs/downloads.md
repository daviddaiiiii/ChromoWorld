# Model downloads and first use

Run project commands from the repository root. Use separate environments for
ChromoPix, ChromoPose/Cellpose, ChromoLift, and the older FAST-CHIMP stack.
Install a CUDA-compatible PyTorch build appropriate for your machine first.
No checkpoint binaries are bundled in the Git repository. Internal weights
are distributed only through the flat Drive `ckpt/` folder below. External
weights are obtained from their original authors; we provide no Drive mirrors.
Maintainers can check this policy without loading a model:
`python -m unittest discover -s tests -v`.

Links and Drive directory listings were checked on **2026-10-22**. The Drive
listing was authenticated: anonymous download access is **not confirmed**, and
the returned permission metadata did not show an `anyone` grant. Request access
if necessary; these instructions do not change sharing permissions. Large
Drive files were not downloaded or archive-tested during this documentation
update. A visible filename/size is not a remote checksum verification.

## ChromoWorld weights

[Download folder: ckpt](https://drive.google.com/drive/folders/1B4LLQiesGVCVG3aq-7846YpRV52jHJg3).
It contains exactly three files, directly inside the folder:

- [ChromoPix — chromopix.pth](https://drive.google.com/file/d/1LApL64GfdYvF2Bwt-9lQ79HCJSaLrCy6/view):
  2,115,044,631 bytes; save as `Checkpoint/ChromoPix/chromopix.pth`.
- [ChromoPose — chromopose.pth](https://drive.google.com/file/d/1jWKZ6M9lUZIjxc5bAWG1_7lMc4nFysio/view):
  204,827,333 bytes; save as `Checkpoint/ChromoPhy/ChromoPose/last.pth`.
  Its configuration is already tracked in
  [training_run](../Checkpoint/ChromoPhy/ChromoPose/training_run/args.json).
- [ChromoLift — chromolift.pth](https://drive.google.com/file/d/1yDoa_QtB-FenOwULfTfg01frWOg9FMQ-/view):
  1,014,334,305 bytes; save as `Checkpoint/ChromoPhy/ChromoLift/lifter_final.pth`.

Drive filenames are standardized; local destination names above preserve the
existing CLI defaults. The download commands below perform that mapping, so
no inference algorithms or checkpoint formats change. Alternatively, keep
the downloaded filename and pass an explicit CLI checkpoint path.
Do not unzip `.pth` files even if Drive labels their MIME type as ZIP: PyTorch
uses a ZIP-based checkpoint serialization format.

Browser download is sufficient. Optionally, with Drive access and `gdown`
installed, download into a clean destination (these commands write the named
files):

```bash
python -m pip install gdown
mkdir -p Checkpoint/ChromoPix Checkpoint/ChromoPhy/ChromoPose Checkpoint/ChromoPhy/ChromoLift
gdown 1LApL64GfdYvF2Bwt-9lQ79HCJSaLrCy6 -O Checkpoint/ChromoPix/chromopix.pth
gdown 1jWKZ6M9lUZIjxc5bAWG1_7lMc4nFysio -O Checkpoint/ChromoPhy/ChromoPose/last.pth
gdown 1yDoa_QtB-FenOwULfTfg01frWOg9FMQ- -O Checkpoint/ChromoPhy/ChromoLift/lifter_final.pth
```

`gdown` cannot bypass Drive permissions or download quotas. If it fails, use
the browser while signed into an authorized account. Also obtain DINOv3 for
ChromoPix and install each component's own requirements before inference:

```bash
python Code/ChromoPix/chromopix/infer.py \
  --image_path /data/png_images --out_path /output/polygons --gpu_id 0
python Code/ChromoPhy/chromopose/segmentation/infer.py \
  --in_path /data/movie.tif --out_path /output/masks \
  --ckpt Checkpoint/ChromoPhy/ChromoPose/last.pth --gpu_id 0
python Code/ChromoPhy/chromolift/infer.py \
  --input /data/features.npz --output /output/lifted.npz
```

ChromoLift consumes seven physical features in the documented order, not a
raw TIFF; image conditioning is optional, and omitting it selects the
unconditioned inference variant. See the [ChromoPix](../Code/ChromoPix/README.md),
[ChromoPose](../Code/ChromoPhy/chromopose/segmentation/README.md), and
[ChromoLift](../Code/ChromoPhy/chromolift/README.md) guides for dependencies,
input schemas, training, and evaluation.

Expected SHA-256 values from the **local release reference files**, not a new
verification of the remote downloads:

```text
7a9ecb0744be1207883c04ae7138ec3e7399c297ad23124a518a749d1416cc9b  chromopix.pth
ff676b53099068ea003fbe4437dbe4aa701112bd07e3652f844418ecb243c9c4  chromopose.pth
d12cc0c92a7b37f22fe2727d59a5b559692eaa7e4b149f7dc2f962d05b847f6e  chromolift.pth
```

Project dataset distribution is outside this model-focused update. Weight
availability alone does not establish availability of the historical
ChromoLift training records/image conditions or the fitted ChromoID gallery.
Saved UMAP assets are plotting inputs, not substitutes for those training data.

## DINOv3 — ChromoPix backbone

- [Official repository and usage](https://github.com/facebookresearch/dinov3).
- [Meta weight-access/download form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
- [Official ViT-B/16 model card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).

Use **ViT-B/16, LVD-1689M**, native PyTorch checkpoint
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`. Accept the upstream terms,
obtain the approved URL, and save it under the release's shorter filename:

```bash
mkdir -p Checkpoint/Outsource/DINOv3
wget -O Checkpoint/Outsource/DINOv3/dinov3_vitb16.pth 'PASTE_THE_APPROVED_VITB16_PTH_URL_HERE'
```

This is a backbone, not the ChromoPix identity head. Run ChromoPix with both
checkpoints present. For standalone backbone loading with the included source:

```python
import torch
backbone = torch.hub.load(
    'Code/ChromoPix/dinov3', 'dinov3_vitb16', source='local',
    weights='Checkpoint/Outsource/DINOv3/dinov3_vitb16.pth')
backbone.eval()
```

Do not rename a Hugging Face `model.safetensors` file to `.pth`; that is a
different loading interface. Expected native-checkpoint SHA-256 is recorded in
[external-weight provenance](../Checkpoint/Outsource/README.md).

## CPSAM — Cellpose-SAM baseline

- [Official code](https://github.com/MouseLand/cellpose) and
  [model selection documentation](https://cellpose.readthedocs.io/en/latest/models.html).
- [Original `cpsam` weight download](https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam)
  ([file page and SHA-256](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam)).

Select **`cpsam` explicitly** for this release, not the newer `cpsam_v2` or
`cpdino` models. First use can auto-download weights; this project's wrapper
instead uses an explicit local file:

```bash
python -m pip install 'cellpose>=4,<5' scipy tifffile
mkdir -p Checkpoint/Outsource/CPSAM
curl -fL -o Checkpoint/Outsource/CPSAM/cpsam https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam
python Code/Analyze/fig5/get-3dmask_TIFF2TIFF.py \
  --input /data/zyx_volumes --output-dir /output/cpsam \
  --model Checkpoint/Outsource/CPSAM/cpsam --gpu-id 0
```

This wrapper expects single 3D **ZYX** volumes, not a whole TZYX movie.
See its `--help` for CPU mode, resizing, and filtering. Choosing the same
weights alone does not pin every library/numerical setting for exact replay.

## FAST-CHIMP — external baseline pipeline

[Official code, installation, and workflow](https://github.com/rstamatov/FAST-CHIMP).
The upstream checkout includes pretrained models and demonstration stacks.
The authors' chromosome-trained model downloads are:

- [CARE weights_best.h5](https://raw.githubusercontent.com/rstamatov/FAST-CHIMP/main/denoising%20models/my_model/weights_best.h5)
  and [matching config.json](https://raw.githubusercontent.com/rstamatov/FAST-CHIMP/main/denoising%20models/my_model/config.json).
- [EmbedSeg best_iou_model.pth](https://raw.githubusercontent.com/rstamatov/FAST-CHIMP/main/segmentation%20models/full_model/best_iou_model.pth).
- [VoxelMorph model_0.02_adjusted.h5](https://raw.githubusercontent.com/rstamatov/FAST-CHIMP/main/registration%20models/model_0.02_adjusted.h5).

Run **inside a separate upstream checkout**, with its documented dependencies
and Fiji installed. Upstream `requirements.txt` is a descriptive list, not a
valid `pip install -r` file. Follow the upstream environment instructions.
The following is a workflow outline, not a newly validated installation:

```bash
git clone https://github.com/rstamatov/FAST-CHIMP.git
cd FAST-CHIMP
python TiffToStacks.py -filename /data/movie.tif -pixel_size 0.05 -pixel_size_z 0.18
python CARE_denoising.py
# Run tubeness.ijm in Fiji before continuing.
python overseg_and_resize.py 0.01 0.1
python EMBEDSEG_predict.py
python refine_embedseg.py
python simple_elastix_register_pairs.py
python propagate.py 1000 1010
```

Use the actual last frame, not blindly `1010`; numbering begins at 1000.
Models assume training-like physical sampling (0.05 µm XY / 0.18 µm Z).
The upstream tracking guidance calls for ≤10 s intervals; our 30–60 s movies
are outside that recommendation. These weights are not loaded by our released
ChromoPose pipeline. Keep upstream notices and model terms.

### CARE

Original implementation: [CSBDeep](https://github.com/CSBDeep/CSBDeep) and
[documentation](https://csbdeep.bioimagecomputing.com/doc/).
For the chromosome baseline, use the FAST-CHIMP authors' CARE `weights_best.h5`
and matching `config.json` linked above, not an unrelated restoration model.
Preserve `denoising models/my_model/` inside the upstream checkout and run
`python CARE_denoising.py` there. CARE expects a matching model configuration,
image axes, and sampling; the weights alone are insufficient.

### EmbedSeg

Original implementation and tutorials: [juglab/EmbedSeg](https://github.com/juglab/EmbedSeg).
For this chromosome baseline, obtain the authors' `best_iou_model.pth` above,
preserve `segmentation models/full_model/` and the upstream network code, and
run `python EMBEDSEG_predict.py` after preparing its resized input volumes.
These task-specific weights are not a universal model for arbitrary objects.

### VoxelMorph

Original implementation: [voxelmorph/voxelmorph](https://github.com/voxelmorph/voxelmorph).
The [upstream pretrained-model guidance](https://github.com/voxelmorph/voxelmorph#pre-trained-models)
is for general registration; do not substitute a brain/MRI model for the
chromosome model linked above. FAST-CHIMP's `propagate.py` builds a TensorFlow
`VxmDense` network and loads `registration models/model_0.02_adjusted.h5`.
Keep that path, architecture, and compatible historical TensorFlow/VoxelMorph
environment, then run propagation after upstream affine registration.

The current VoxelMorph project has evolving PyTorch interfaces and a separate
legacy TensorFlow branch. Do not assume its latest package is compatible with
FAST-CHIMP's older `.h5` weights. Our released `affine.py` is image-based rigid
registration and does not need VoxelMorph or any registration checkpoint.

## Chimps — external paired image/mask benchmark

Here “Chimps” refers to the FAST-CHIMP study's microscopy benchmark, not an
additional neural model:

- [Dataset DOI / download record](https://doi.org/10.6084/m9.figshare.25035905).
- [Figshare file listing](https://springernature.figshare.com/articles/dataset/Supra-second_tracking_and_live_cell_karyotyping_reveal_principles_of_mitotic_chromosome_dynamics/25035905).
- [Paper and data-availability statement](https://www.nature.com/articles/s41556-025-01637-6#data-availability).
- [Small single-cell example: Cell 5 RPE.zip](https://ndownloader.figshare.com/files/49176949)
  (537,863,434 bytes in the current listing).

Download selected cell archives from the record, inspect their contents, and
pair the corresponding original image and label TIFFs. Our preparer expects
`Dataset/4D/Chimps/{train,test}/<cell>/raw.tif` and `mask.tif`. Each raw/mask
pair must share the same TZYX axes, frame count, voxel grid, and ordering.
Choose train/test cells explicitly and record the split; the official download
does not itself reproduce our historical split or derived identity assignments.

```bash
python Code/ChromoPhy/chromopose/segmentation/preprocess.py \
  --chimp-root Dataset/4D/Chimps --out-root Dataset/4D/ChromoPoseTargets
```

The paper reports 14 cells / 684 chromosomes / 3,569 stacks; the inspected
Figshare v1 record currently lists 12 ZIP files, so consult its manifest rather
than assuming one ZIP per paper cell. Our locally edited/derived masks are not
claimed to be byte-identical to those official archives. Follow the record's
license and cite its authors when reusing the data.
