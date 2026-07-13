#!/bin/bash
set -euo pipefail
CONTAINER="${1:?container name required}"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mods=(
  drop-caches
  ray-keep-node-nccl-hca
  fix-prometheus-instrumentator-router
  fix-mimo-v2-vllm
  fix-mimo-mtp-patches
  fix-modelopt-mixed-mxfp8
  nvfp4-kv-diffkv
)
for mod in "${mods[@]}"; do
  echo "Applying mod: $mod"
  docker cp "$BASE/mods/$mod" "$CONTAINER:/tmp/$mod"
  docker exec "$CONTAINER" bash "/tmp/$mod/run.sh" || echo "[apply-mods] $mod returned non-zero (continuing)"
done
