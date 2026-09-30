#!/usr/bin/env bash
# ChromoPose — 64×256×256, multi-GPU DDP, skeleton-flow multi-head loss
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PKG}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
RELEASE_ROOT="$(cd "${PKG}/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${RELEASE_ROOT}/Dataset/4D/ChromoPoseTargets}"
GPU_ID="${GPU_ID:-0,1,2}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PKG}/outputs/chromopose_${TIMESTAMP}}"
RESUME_CKPT="${RESUME_CKPT:-}"

IFS=',' read -r -a _GPUS <<< "${GPU_ID// /}"
NPROC="${#_GPUS[@]}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Output dir: ${OUTPUT_DIR}"
echo "Data root:  ${DATA_ROOT}"
echo "GPUs: ${GPU_ID} | nproc=${NPROC}"

EXTRA_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
  EXTRA_ARGS+=(--resume "${RESUME_CKPT}")
fi

python -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" -m train \
  --distributed \
  --index-path "${DATA_ROOT}/index.jsonl" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --crop-z 64 \
  --crop-y 256 \
  --crop-x 256 \
  --crop-mode-probs 0.5,0.3,0.2 \
  --train-split train \
  --backbone unet3d \
  --base-ch 32 \
  --levels 4 \
  --use-embedding true \
  --batch-size 2 \
  --epochs 50 \
  --lr 1e-3 \
  --warmup-epochs 5 \
  --flow-scale 5.0 \
  --fg-weight 1.0 \
  --flow-weight 1.0 \
  --dist-weight 0.5 \
  --skel-weight 1.0 \
  --boundary-weight 0.5 \
  --emb-weight 0.25 \
  --val-ratio 0.1 \
  --num-workers 4 \
  --eval-every 5 \
  --test-every 5 \
  --log-every 50 \
  "${EXTRA_ARGS[@]}" "$@"
