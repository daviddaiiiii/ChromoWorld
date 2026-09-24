# Fig2-3 qualitative demo

The demo consists of exactly two composite panels: `Banding_5x3.png` and
`FISH_5x3.png`. Each panel has five held-out examples as rows and three columns:
the processed input image, the ground-truth mask, and the ChromoPix prediction.
GT and prediction masks use the same 24-class Chromo World palette and are
shown as semi-transparent overlays on the processed input. Unlabelled FISH GT
instances are shown in neutral gray.

To regenerate the predictions for these ten examples, stage only the selected
processed test images and run ChromoPix with the released checkpoints:

```bash
mkdir -p .tmp/fig23_demo/images .tmp/fig23_demo/predictions
for stem in \
  Banding_04318 Banding_04856 Banding_05241 Banding_05463 Banding_05605 \
  id01_HFF_FISH_00003 id06_MCF-10F_FISH_00000 id12_MCF-10F_FISH_00002 \
  id18_MCF-10F_FISH_00024 idY_MRC5_FISH_00000; do
  cp "release/Dataset/2D/test/image/${stem}.png" .tmp/fig23_demo/images/
done

python release/Code/Analyze/fig2-3/get-polygon_PNG2TXT.py \
  --image_path .tmp/fig23_demo/images \
  --out_path .tmp/fig23_demo/predictions \
  --checkpoint release/Checkpoint/ChromoPix/chromopix.pth \
  --chromopix_code release/Code/ChromoPix \
  --dinov3_repo release/Code/ChromoPix/dinov3 \
  --model_weights release/Checkpoint/ChromoPix/dinov3_vitb16.pth \
  --gpu_id 0 --batch_size 1 --img_h 512 --img_w 512

python release/Code/Analyze/fig2-3/create_fig2_3_demo.py \
  --pred-dir .tmp/fig23_demo/predictions
```

`create_fig2_3_demo.py` writes only the two composite PNGs to `demo_outputs/`.
The input images are the processed images used by the Fig2-3 pipeline, not the
original DAPI files. Relative paths are resolved from the ChromoWorld project
root.
