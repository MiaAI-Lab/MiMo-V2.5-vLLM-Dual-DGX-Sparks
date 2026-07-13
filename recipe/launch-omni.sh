#!/usr/bin/env bash
# MiMo-V2.5 Omni + MTP1 + NVFP4-KV (HEAD after Ray up + source env.sh).
# Port defaults to 8888 — host :8000 is used by Portainer.
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH}"
: "${SERVED_MODEL_NAME:=MiMo-V2.5-NVFP4}"
: "${API_PORT:=8888}"

if [[ -n "${HEAD_ROCE_IP:-}" ]]; then
    export VLLM_HOST_IP="${HEAD_ROCE_IP}"
fi

# Do not put JSON defaults inside ${VAR:-...} — bash treats the closing }
# of the JSON as the end of the parameter expansion (produces an extra }).
if [[ -z "${SPECULATIVE_CONFIG:-}" ]]; then
    SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1,"use_local_argmax_reduction":false}'
fi

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype auto \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend ray \
  --load-format safetensors \
  --hf-overrides '{"architectures":["MiMoV2OmniForCausalLM"]}' \
  --limit-mm-per-prompt '{"image":4,"video":1,"audio":1}' \
  --mm-encoder-tp-mode data \
  --attention-backend triton_attn_diffkv \
  --moe-backend "${MOE_BACKEND:-${VLLM_FLASHINFER_MOE_BACKEND:-flashinfer_cutlass}}" \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.83}" \
  --max-model-len "${MAX_MODEL_LEN:-1000000}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --max-num-seqs "${MAX_NUM_SEQS:-3}" \
  --block-size "${BLOCK_SIZE:-64}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser mimo \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  --generation-config vllm \
  --enforce-eager \
  --disable-uvicorn-access-log \
  --host 0.0.0.0 \
  --port "${API_PORT}"
