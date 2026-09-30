# ChromoID: lifted-gallery identity retrieval

Run commands from the release root. ChromoID uses ChromoLift's 3D scaler and
weighted Euclidean kNN (default k=5, weight 1/(distance + 1e-6)).
train.py constructs a labeled 7D gallery; infer.py assigns IDs; evaluate.py
scores held-out descriptors against caller-provided reference class labels.
There is no independently trained ChromoID neural checkpoint.

The requirements pin scikit-learn 1.7.1 to match the scaler objects serialized
in the released ChromoLift checkpoint.

```bash
python -m pip install -r Code/ChromoPhy/chromoid/requirements.txt
python Code/ChromoPhy/chromoid/train.py \
  --input-jsonl /data/instance_2d_filtered.jsonl \
  --image-feature-dir /data/2D_features \
  --output Artifacts/ChromoID/gallery_7d_cache.npz
python Code/ChromoPhy/chromoid/infer.py \
  --input /data/volume_7d.npz --output /output/chromosome_ids.npz
python Code/ChromoPhy/chromoid/evaluate.py \
  --input /data/heldout_volume_7d.npz --output /output/id_metrics.json
```

The builder needs the historical filtered Banding/FISH records and matching
image-feature NPY rows. These inputs and the fitted gallery are not currently
bundled. It loads Checkpoint/ChromoPhy/ChromoLift/lifter_final.pth.
--no-image-condition builds a different, unconditioned gallery.

The default gallery is Artifacts/ChromoID/gallery_7d_cache.npz, containing
lifted_combined_x [N,7] and lifted_combined_y [N]. Supply --gallery to use
another fit. The historic conditioned gallery had 711,541 rows; rebuilding
identical values requires the same source data, conditions, checkpoint, and
inference settings. Artifacts/fig5/Lifted2D_umap_embed.npz contains five-feature
plotting arrays, not this 7D retrieval gallery.

Inference expects features [N,7] in the ChromoLift feature order. Evaluation
also requires class_id [N] in 0..23 and reports accuracy, macro recall of
present classes, and a true-row/prediction-column confusion matrix.
The caller must keep gallery construction rows separate from evaluation
queries. Predictions are inferred identities, not verified karyotype GT.
