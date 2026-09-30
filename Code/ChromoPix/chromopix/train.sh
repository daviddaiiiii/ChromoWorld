#!/usr/bin/env bash
# Train ChromoPix from ChromoWorld root (relative paths).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEFAULT_GPU_IDS="0,1,2,3"
DEFAULT_MASTER_PORT=29500
DEFAULT_BATCH_SIZE=24
DEFAULT_EPOCHS=100
DEFAULT_LR=1e-4
IMG_H=512
IMG_W=512

GPU_IDS=${1:-$DEFAULT_GPU_IDS}
shift 2>/dev/null || true

if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Error: Invalid GPU IDs. Use e.g. '0,1,2,3'"
  exit 1
fi

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

find_free_port() {
  local port=$DEFAULT_MASTER_PORT
  while netstat -tuln 2>/dev/null | grep -q ":$port "; do
    ((port++))
  done
  echo $port
}

MASTER_PORT=$(find_free_port)
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$MASTER_PORT

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./outputs/chromopix_training_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

TRAINING_ARGS=(
  --batch_size "$DEFAULT_BATCH_SIZE"
  --epochs "$DEFAULT_EPOCHS"
  --lr "$DEFAULT_LR"
  --output_dir "$OUTPUT_DIR"
  --num_workers 8
  --eval_freq 10
  --save_freq 10
  --log_freq 50
  --experiment_name "chromopix_training_${TIMESTAMP}"
  --img_h "$IMG_H"
  --img_w "$IMG_W"
  --dinov3_repo ./dinov3
  --model_weights ../../Checkpoint/Outsource/DINOv3/dinov3_vitb16.pth
  --train_label_dir ../../Dataset/2D/train/label
  --train_image_dir ../../Dataset/2D/train/image
  --train_visualize_dir ./outputs/train_visualization
  --test_label_dir ../../Dataset/2D/test/label
  --test_image_dir ../../Dataset/2D/test/image
  --test_visualize_dir ./outputs/test_visualization
)
TRAINING_ARGS+=("$@")

echo "Launching ChromoPix training on GPUs=$GPU_IDS -> $OUTPUT_DIR"
python -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  chromopix/train.py \
  "${TRAINING_ARGS[@]}"
