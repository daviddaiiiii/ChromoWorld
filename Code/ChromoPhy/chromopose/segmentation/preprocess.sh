#!/usr/bin/env bash
# ChromoPose — preprocess 4D chimp → skeleton-flow targets (64×256×256), parallel workers
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${PKG}/../../../.." && pwd)"
CHIMP_ROOT="${CHIMP_ROOT:-${RELEASE_ROOT}/Dataset/4D/Chimps}"
DATA_ROOT="${DATA_ROOT:-${RELEASE_ROOT}/Dataset/4D/ChromoPoseTargets}"
WORKERS="${WORKERS:-8}"

if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
cd "${PKG}"

# Keep the optional dataset link local to this package.
if [ ! -e "${PKG}/dataset" ] && [ ! -L "${PKG}/dataset" ]; then
  ln -sfn "${DATA_ROOT}" "${PKG}/dataset"
fi

echo "ChromoPose preprocess"
echo "  chimp:   ${CHIMP_ROOT}"
echo "  out:     ${DATA_ROOT}"
echo "  workers: ${WORKERS}"

python preprocess.py \
  --chimp-root "${CHIMP_ROOT}" \
  --out-root "${DATA_ROOT}" \
  --size-z 64 \
  --size-y 256 \
  --size-x 256 \
  --distance-clip 32.0 \
  --workers "${WORKERS}" \
  "$@"
