#!/usr/bin/env bash
# =============================================================================
# start.sh — MiMo-V2.5 Omni (MiMoV2OmniForCausalLM) + MTP1 + NVFP4 4-bit KV,
# TP=2 over Ray across two NVIDIA DGX Sparks. SSH-driven from the head.
#
# Full two-node bring-up aligned with:
#   https://github.com/tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark
# ...then VERIFIES with a real OpenAI chat completion. Exit 0 only when chat
# returns non-empty content.
#
# This tree serves Omni ONLY (no DFlash lane). Requires the DEV vLLM build +
# recipe mods (nvfp4 DiffKV, MiMo/Omni registration). Stock pip vLLM will not
# accept --kv-cache-dtype nvfp4 / triton_attn_diffkv / MiMoV2OmniForCausalLM.
#
# Modes:
#   bash start.sh              # full bring-up + verify (default)
#   bash start.sh --check      # print plan + weight completeness (no download/launch)
#   bash start.sh --teardown   # ray stop + container stop on both nodes, exit
#
# Weights (full run only): if the HF hub cache is missing or incomplete, downloads
#   with `hf` on the head (resumable), then rsyncs the Omni target to the worker
#   only when the worker cache is missing/incomplete (skip if already complete).
#   Override: TARGET_REPO / TARGET_REVISION / AUTO_DOWNLOAD_MODELS=0 /
#   SYNC_MODELS_TO_WORKER=0 / FORCE_SYNC_MODELS_TO_WORKER=1.
#
# Exit 0 ONLY after a real chat completion returns non-empty content.
# =============================================================================
set -euo pipefail

###############################################################################
# CONFIG BLOCK — all overridable by env. `set -a` so children (containers, ray,
# vllm) inherit what's relevant.
###############################################################################
HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
HEAD_ROCE_IP="${HEAD_ROCE_IP:-$HEAD_IP}"
WORKER_ROCE_IP="${WORKER_ROCE_IP:-$WORKER_IP}"

SSH_USER="${SSH_USER:-zurih}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8)

RAY_PORT="${RAY_PORT:-6379}"
VLLM_PORT="${VLLM_PORT:-8888}"          # the serving API port

# --- NCCL / interconnect -----------------------------------------------------
# On THIS cluster the active fabric is enp1s0f1np1 (10.0.0.0/24) with HCA
# rocep1s0f1. The repo's defaults (enp1s0f0np0 / rocep1s0f0) are for a different
# port that is DOWN here, BUT rocep1s0f1 has a valid RoCEv2 GID at index 2 & 3
# (IPv4-mapped 10.0.0.x) on BOTH nodes, so we use RoCE/IB (NCCL_IB_DISABLE=0)
# — matching the repo's IB transport. TCP sockets (NCCL_IB_DISABLE=1) work but
# add ms of cross-node all-reduce latency that hurts decode/tok-s.
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"          # RoCEv2 GID idx — both nodes have it at 2 & 3
	WORKER_NCCL_IB_GID_INDEX="${WORKER_NCCL_IB_GID_INDEX:-3}"  # repo says worker historically needed 5; our port1 has GID at 3
NCCL_PROTO="${NCCL_PROTO:-LL}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-2}"
NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-LOC}"

# --- model + image (see detect_* for auto-resolution) ------------------------
# IN-CONTAINER paths (identical on both nodes via bind mounts):
: "${MODEL_PATH:=/model_cache/snapshots/auto}"
: "${SERVED_MODEL_NAME:=MiMo-V2.5-NVFP4}"
REPO_DIR="${REPO_DIR:-/mnt/models/MiMo}"               # this repo's root (on the head)
CONTAINER_NAME="${CONTAINER_NAME:-mimo-nvfp4}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/miaai-lab/mimo-v2.5-vllm-dual-dgx-sparks:base-20260620}"
OVERLAY_TAG="${OVERLAY_TAG:-ghcr.io/miaai-lab/mimo-v2.5-vllm-dual-dgx-sparks:20260704}"
BASE_TAG="${BASE_TAG:-ghcr.io/miaai-lab/mimo-v2.5-vllm-dual-dgx-sparks:base-20260620}"
# If the exact OVERLAY_TAG is already local, we never touch GHCR.
# Optional comma-separated local names to retag → OVERLAY_TAG when the GHCR
# name is missing (air-gapped / private package). Empty = no alias retag.
LOCAL_IMAGE_ALIASES="${LOCAL_IMAGE_ALIASES:-}"
# 1 = never docker pull from registry (local / head→worker only).
SKIP_IMAGE_PULL="${SKIP_IMAGE_PULL:-0}"
WORKER_REPO_DIR="${WORKER_REPO_DIR:-/home/zurih/mimo-nvfp4-recipe}"  # rsync target on worker

# Host-side HF cache model dir (auto-detected if blank):
MODEL_HOST_DIR="${MODEL_HOST_DIR:-}"          # models--<org>--<name> hub cache dir

# Auto-download + worker sync (HF hub layout: <hub>/models--org--name/{blobs,snapshots})
# Omni-validated weights: lukealonso @ a147dd (full multimodal + MTP).
TARGET_REPO="${TARGET_REPO:-lukealonso/MiMo-V2.5-NVFP4}"
TARGET_REVISION="${TARGET_REVISION:-a147dd04d6cf861e43b2d783dcde23b53ab7ee68}"
AUTO_DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}"   # 1 = download if missing/incomplete
SYNC_MODELS_TO_WORKER="${SYNC_MODELS_TO_WORKER:-1}" # 1 = rsync head → worker when incomplete
FORCE_SYNC_MODELS_TO_WORKER="${FORCE_SYNC_MODELS_TO_WORKER:-0}" # 1 = always rsync even if complete
WORKER_HF_HUB="${WORKER_HF_HUB:-}"

# --- Omni MTP1 + NVFP4-KV shape ---------------------------------------------
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton_attn_diffkv}"
export ENABLE_MTP="${ENABLE_MTP:-1}"
export MTP_SPEC_TOKENS="${MTP_SPEC_TOKENS:-1}"
export VLLM_MIMO_MTP1_GREEDY_FAST="${VLLM_MIMO_MTP1_GREEDY_FAST:-1}"

# 1M context, 3 concurrent (KV pool ~2.5M ≈ ~2.5× full-1M; 3rd deep 1M queues).
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"
export BLOCK_SIZE="${BLOCK_SIZE:-64}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.83}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# --- verify knobs + retry budget -------------------------------------------
VERIFY_MODEL_PROMPT="${VERIFY_MODEL_PROMPT:-Reply with just: ok}"
MAX_BOOT_ATTEMPTS="${MAX_BOOT_ATTEMPTS:-900}"        # /v1/models polls (~30 min at 2s)
BOOT_POLL_INTERVAL="${BOOT_POLL_INTERVAL:-2}"
MAX_RELAY_ATTEMPTS="${MAX_RELAY_ATTEMPTS:-5}"
RAY_WAIT_TIMEOUT="${RAY_WAIT_TIMEOUT:-600}"

VLLM_LOG="${VLLM_LOG:-$REPO_DIR/vllm.log}"
RETRY_LOG="${RETRY_LOG:-$REPO_DIR/retry.log}"
HF_HOME_HOST="${HF_HOME_HOST:-$HOME/.cache/huggingface}"

set -a

###############################################################################
# Logging / UI  (TTY-aware colors; plain text if piped)
###############################################################################
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_MAGENTA=$'\033[35m'
  C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BLUE=$'\033[34m'
else
  C_RESET=; C_BOLD=; C_DIM=; C_CYAN=; C_GREEN=; C_MAGENTA=
  C_YELLOW=; C_RED=; C_BLUE=
fi

_ts() { date -u +%H:%M:%S; }

log()      { printf '%s%s[%s]%s %s%sstart%s  %s\n' "$C_DIM" "$C_BOLD" "$(_ts)" "$C_RESET" "$C_CYAN" "$C_BOLD" "$C_RESET" "$*"; }
hlog()     { printf '%s%s[%s]%s %s%shead%s   %s\n' "$C_DIM" "$C_BOLD" "$(_ts)" "$C_RESET" "$C_GREEN" "$C_BOLD" "$C_RESET" "$*"; }
wlog()     { printf '%s%s[%s]%s %s%sworker%s %s\n' "$C_DIM" "$C_BOLD" "$(_ts)" "$C_RESET" "$C_MAGENTA" "$C_BOLD" "$C_RESET" "$*"; }
vlog()     { printf '%s%s[%s]%s %s%sverify%s %s\n' "$C_DIM" "$C_BOLD" "$(_ts)" "$C_RESET" "$C_YELLOW" "$C_BOLD" "$C_RESET" "$*"; }
err()      { printf '%s%s[%s]%s %s%sERROR%s  %s\n' "$C_DIM" "$C_BOLD" "$(_ts)" "$C_RESET" "$C_RED" "$C_BOLD" "$C_RESET" "$*" >&2; }
ok()       { printf '%s%s[%s]%s %s%sok%s     %s\n' "$C_DIM" "$C_BOLD" "$(_ts)" "$C_RESET" "$C_GREEN" "$C_BOLD" "$C_RESET" "$*"; }
retrylog() { printf '%s%s[%s]%s %s%sretry%s  %s\n' "$C_DIM" "$C_BOLD" "$(_ts)" "$C_RESET" "$C_YELLOW" "$C_BOLD" "$C_RESET" "$*" | tee -a "$RETRY_LOG"; }

# Numbered phase banner: step 3 7 "Ray cluster"
step() {
  local n="$1" total="$2"; shift 2
  printf '\n%s%s────────────────────────────────────────────────────────────%s\n' "$C_DIM" "$C_BOLD" "$C_RESET"
  printf '%s%s  step %s/%s%s  %s%s%s\n' "$C_BLUE" "$C_BOLD" "$n" "$total" "$C_RESET" "$C_BOLD" "$*" "$C_RESET"
  printf '%s%s────────────────────────────────────────────────────────────%s\n' "$C_DIM" "$C_BOLD" "$C_RESET"
}

# Fixed-width double-line box rows (inner width = BOX_W).
# Use ASCII-only body text (no · / —): those codepoints are ambiguous-width in
# many terminals (esp. Windows) and pull the right border left even when
# character-count padding is correct.
BOX_W=62
_BOX_BAR="$(printf '%*s' "$BOX_W" '' | sed 's/ /═/g')"
_pad_chars() {
  local text="$1" width="$2" n=${#1} pad
  if (( n > width )); then
    printf '%s' "${text:0:width}"
  else
    pad=$((width - n))
    printf '%s%*s' "$text" "$pad" ''
  fi
}
_box_rule() { # $1=color $2=left-corner $3=right-corner  (╔╗ / ╠╣ / ╚╝)
  local color="$1" lc="$2" rc="$3"
  printf '%s%s%s%s%s%s\n' "$color" "$C_BOLD" "$lc" "$_BOX_BAR" "$rc" "$C_RESET"
}
_box_row() { # $1=frame-color $2=optional-text-color-or-empty $3=inner text
  local fc="$1" tc="${2:-}" text="$3" pad
  pad="$(_pad_chars "$text" "$BOX_W")"
  if [ -n "$tc" ]; then
    printf '%s%s║%s%s%s%s%s%s║%s\n' "$fc" "$C_BOLD" "$C_RESET" "$tc" "$pad" "$C_RESET" "$fc" "$C_BOLD" "$C_RESET"
  else
    printf '%s%s║%s%s%s%s║%s\n' "$fc" "$C_BOLD" "$C_RESET" "$pad" "$fc" "$C_BOLD" "$C_RESET"
  fi
}
# Success-banner field: fixed label column so values share one left edge.
_box_field() { # $1=frame-color $2=label $3=value
  local fc="$1" label="$2" value="$3"
  # label pad to 7 chars → "API    " / "model  " / "shape  " / "reply  "
  _box_row "$fc" '' "  $(printf '%-7s' "$label")${value}"
}

banner() {
  local mode="${1:-bring-up}"
  printf '\n'
  _box_rule "$C_CYAN" '╔' '╗'
  _box_row  "$C_CYAN" '' "  Mia's MiMo-V2.5 Dual DGX Spark Start Script"
  _box_row  "$C_CYAN" '' "  Omni | MTP1 | NVFP4-KV | TP=2"
  _box_row  "$C_CYAN" "$C_DIM" "  ${mode}"
  _box_rule "$C_CYAN" '╚' '╝'
  printf '%s  %s:%s  ->  %s (TP1)  |  %s%s\n\n' \
    "$C_DIM" "$HEAD_IP" "$VLLM_PORT" "$WORKER_IP" "$SERVED_MODEL_NAME" "$C_RESET"
}

success_banner() {
  local reply="$1"
  local api="http://${HEAD_IP}:${VLLM_PORT}/v1"
  local shape="${MAX_MODEL_LEN} ctx | ${MAX_NUM_SEQS} seqs | GMU ${GPU_MEMORY_UTILIZATION} | MTP${MTP_SPEC_TOKENS}"
  printf '\n'
  _box_rule "$C_GREEN" '╔' '╗'
  _box_row   "$C_GREEN" '' "  READY - chat verified"
  _box_rule "$C_GREEN" '╠' '╣'
  _box_field "$C_GREEN" "API"    "$api"
  _box_field "$C_GREEN" "model"  "$SERVED_MODEL_NAME"
  _box_field "$C_GREEN" "shape"  "$shape"
  _box_field "$C_GREEN" "reply"  "${reply:0:48}"
  _box_rule "$C_GREEN" '╚' '╝'
  printf '\n%sSmoke:%s\n' "$C_BOLD" "$C_RESET"
  printf '  curl -s http://%s:%s/v1/models | jq .\n' "$HEAD_IP" "$VLLM_PORT"
  printf '  bash %s/stop.sh\n\n' "$REPO_DIR"
}

# Run a command on the worker over SSH. $1+ = argv.
sshw() { ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" "$*"; }

###############################################################################
# Auto-detection / download / completeness / head→worker sync of HF hub caches.
# HF snapshot entries are symlinks into ../blobs, so we bind the WHOLE model
# cache dir (blobs/ + snapshots/) — not just a snapshot.
# Globals: MODEL_HOST_DIR, MODEL_SNAPSHOT_REV, MODEL_PATH, WORKER_MODEL_HOST_DIR.
###############################################################################
hf_repo_cache_name() { # $1 = org/name → models--org--name
  printf 'models--%s' "${1//\//--}"
}

first_snapshot_rev() { # $1 = model dir ; echoes the single snapshot rev or the newest
  local d="$1"; [ -d "$d/snapshots" ] || return 1
  local r
  r=$(ls -1 "$d/snapshots" 2>/dev/null | head -1)
  [ -n "$r" ] || return 1
  echo "$r"
}

# Returns 0 if cache+revision look fully usable (config + weight shards resolve).
# Orphan blobs/*.incomplete from a different/partial download are ignored when the
# requested snapshot's files all resolve — those leftovers must not force re-download
# or re-rsync of an otherwise complete revision.
# $1=cache_dir  $2=revision  $3=optional subdir (e.g. dflash)
hf_cache_complete() {
  local cache="$1" rev="$2" sub="${3:-}"
  local snap="$cache/snapshots/$rev"
  [ -n "$sub" ] && snap="$snap/$sub"
  [ -d "$cache" ] || return 1
  [ -d "$cache/blobs" ] || return 1
  [ -e "$snap/config.json" ] || return 1
  python3 - "$snap" <<'PY'
import json, sys
from pathlib import Path
snap = Path(sys.argv[1])
idx = snap / "model.safetensors.index.json"
if idx.exists():
    weight_map = json.loads(idx.read_text()).get("weight_map") or {}
    files = sorted(set(weight_map.values()))
    if not files:
        sys.exit(1)
    for name in files:
        p = snap / name
        if not p.exists():
            sys.exit(1)
        if p.is_symlink() and not p.resolve().exists():
            sys.exit(1)
    sys.exit(0)
# Single-file / small drafter layouts
cands = list(snap.glob("*.safetensors")) + list(snap.glob("*.pt"))
sys.exit(0 if cands else 1)
PY
}

# Same check over SSH on the worker. Args identical to hf_cache_complete.
hf_cache_complete_remote() {
  local cache="$1" rev="$2" sub="${3:-}"
  sshw "python3 - $(printf '%q' "$cache") $(printf '%q' "$rev") $(printf '%q' "$sub")" <<'PY'
import json, sys
from pathlib import Path
cache, rev, sub = sys.argv[1], sys.argv[2], sys.argv[3]
snap = Path(cache) / "snapshots" / rev
if sub:
    snap = snap / sub
blobs = Path(cache) / "blobs"
if not blobs.is_dir() or not (snap / "config.json").exists():
    raise SystemExit(1)
# Orphan *.incomplete ignored when snapshot weights resolve (same as local check).
idx = snap / "model.safetensors.index.json"
if idx.exists():
    files = sorted(set((json.loads(idx.read_text()).get("weight_map") or {}).values()))
    if not files:
        raise SystemExit(1)
    for name in files:
        p = snap / name
        if not p.exists() or (p.is_symlink() and not p.resolve().exists()):
            raise SystemExit(1)
    raise SystemExit(0)
cands = list(snap.glob("*.safetensors")) + list(snap.glob("*.pt"))
raise SystemExit(0 if cands else 1)
PY
}

# Prefer pinned rev if complete; else any complete snapshot in cache; else empty.
resolve_complete_snapshot_rev() {
  local cache="$1" prefer="$2"
  local rev
  if [ -n "$prefer" ] && hf_cache_complete "$cache" "$prefer"; then
    echo "$prefer"
    return 0
  fi
  [ -d "$cache/snapshots" ] || return 1
  for rev in $(ls -1 "$cache/snapshots" 2>/dev/null); do
    if hf_cache_complete "$cache" "$rev"; then
      echo "$rev"
      return 0
    fi
  done
  return 1
}

find_hf_cli() {
  if command -v hf >/dev/null 2>&1; then echo hf; return 0; fi
  if command -v huggingface-cli >/dev/null 2>&1; then echo huggingface-cli; return 0; fi
  return 1
}

# Download (resumable) into an HF hub parent so cache_dir = $hub_parent/models--org--name.
# Extra args after rev are forwarded to hf download.
hf_download_into() {
  local cache_dir="$1" repo="$2" rev="$3"
  shift 3
  local hub_parent cli
  hub_parent=$(dirname "$cache_dir")
  mkdir -p "$hub_parent"
  cli=$(find_hf_cli) || {
    err "need 'hf' or 'huggingface-cli' on PATH to download $repo"
    return 1
  }
  log "downloading $repo @$rev → $cache_dir (resumable)"
  if [ "$cli" = hf ]; then
    HF_HUB_CACHE="$hub_parent" HF_HOME="$HF_HOME_HOST" \
      hf download "$repo" --revision "$rev" "$@"
  else
    HF_HUB_CACHE="$hub_parent" HF_HOME="$HF_HOME_HOST" \
      huggingface-cli download "$repo" --revision "$rev" "$@"
  fi
}

pick_or_default_cache() {
  # $1=preferred_repo_id  — echoes chosen models-- dir (may not exist yet)
  local repo="$1" name pref
  name=$(hf_repo_cache_name "$repo")
  pref="$HF_HOME_HOST/hub/$name"
  local cand
  for cand in \
      "$pref" \
      "/mnt/$name" \
      "$REPO_DIR/$name"; do
    if [ -d "$cand" ] || [ -d "$cand/snapshots" ]; then
      echo "$cand"
      return 0
    fi
  done
  # Nothing on disk yet — download into the standard HF hub location.
  echo "$pref"
}

ensure_one_model_head() {
  # $1=label  $2=repo  $3=rev  $4=name of global for cache dir
  # $5=optional subdir for completeness (dflash)  $6+=extra hf download args
  local label="$1" repo="$2" rev="$3" var="$4" sub="${5:-}"
  shift 5
  local cache="${!var}"
  if [ -z "$cache" ]; then
    cache=$(pick_or_default_cache "$repo")
    printf -v "$var" '%s' "$cache"
  fi
  if hf_cache_complete "$cache" "$rev" "$sub"; then
    log "$label complete: $cache @$rev${sub:+/$sub}"
    MODEL_SNAPSHOT_REV="$rev"
    return 0
  fi

  # Target only: reuse any already-complete Omni hub dir before re-pulling ~171G.
  # Also accept another complete snapshot in the same cache when the pin is absent
  # (common: refs/main or an older full rev vs a different TARGET_REVISION pin).
  if [ "$label" = "target" ]; then
    local cand alt_rev found_rev
    if found_rev=$(resolve_complete_snapshot_rev "$cache" "$rev"); then
      if [ "$found_rev" != "$rev" ]; then
        log "$label: pin $rev missing/incomplete; using complete snapshot @$found_rev in $cache (skip download)"
      else
        log "$label complete: $cache @$found_rev"
      fi
      printf -v "$var" '%s' "$cache"
      MODEL_SNAPSHOT_REV="$found_rev"
      return 0
    fi
    for cand in \
      "$HF_HOME_HOST/hub/models--lukealonso--MiMo-V2.5-NVFP4" \
      "/mnt/models--lukealonso--MiMo-V2.5-NVFP4" \
      "$REPO_DIR/models--lukealonso--MiMo-V2.5-NVFP4" \
      "$HF_HOME_HOST/hub/models--mitomtuna--MiMo-V2.5-0703-NVFP4" \
      "/mnt/models--mitomtuna--MiMo-V2.5-0703-NVFP4"; do
      [ "$cand" = "$cache" ] && continue
      [ -d "$cand/snapshots" ] || continue
      if found_rev=$(resolve_complete_snapshot_rev "$cand" "$rev"); then
        log "$label: using existing complete cache $cand @$found_rev (skip download)"
        printf -v "$var" '%s' "$cand"
        MODEL_SNAPSHOT_REV="$found_rev"
        return 0
      fi
    done
  fi

  if [ "${AUTO_DOWNLOAD_MODELS}" != "1" ]; then
    err "$label missing/incomplete at $cache @$rev (set AUTO_DOWNLOAD_MODELS=1 or finish download)"
    return 1
  fi
  if [ -d "$cache" ]; then
    log "$label present but incomplete at $cache — resuming download"
  else
    log "$label not found — downloading to $cache"
  fi
  hf_download_into "$cache" "$repo" "$rev" "$@"
  if ! hf_cache_complete "$cache" "$rev" "$sub"; then
    err "$label still incomplete after download: $cache @$rev"
    return 1
  fi
  printf -v "$var" '%s' "$cache"
  log "$label ready: $cache @$rev"
}

rsync_cache_to_worker() {
  local src="$1" dst="$2" label="$3"
  [ -d "$src" ] || { err "cannot sync $label: missing $src"; return 1; }
  wlog "syncing $label → worker:$dst"
  sshw "mkdir -p $(printf '%q' "$(dirname "$dst")")"
  # -a keeps hub symlink layout; --delete drops stale partials on the worker.
  # Exclude leftover HF download temps so orphan *.incomplete on the head are not
  # copied (and so --delete does not force them onto the worker).
  rsync -aH --delete --info=stats2 \
    --exclude='*.incomplete' \
    -e "ssh ${SSH_OPTS[*]}" \
    "$src/" "${SSH_USER}@${WORKER_IP}:$dst/"
}

ensure_models() {
  # Head: ensure Omni target complete (download/resume if needed).
  # Worker: rsync head hub → worker only when missing/incomplete (or forced), then verify.
  ensure_one_model_head "target" "$TARGET_REPO" "$TARGET_REVISION" MODEL_HOST_DIR ""

  local resolved
  if resolved=$(resolve_complete_snapshot_rev "$MODEL_HOST_DIR" "${MODEL_SNAPSHOT_REV:-$TARGET_REVISION}"); then
    MODEL_SNAPSHOT_REV="$resolved"
  else
    MODEL_SNAPSHOT_REV="${MODEL_SNAPSHOT_REV:-$TARGET_REVISION}"
    MODEL_SNAPSHOT_REV="${MODEL_SNAPSHOT_REV:-$(first_snapshot_rev "$MODEL_HOST_DIR" || true)}"
  fi
  if [ -z "${MODEL_SNAPSHOT_REV:-}" ]; then
    err "could not resolve a model snapshot revision under $MODEL_HOST_DIR"
    return 1
  fi
  log "using snapshot revision $MODEL_SNAPSHOT_REV (pin was $TARGET_REVISION)"

  MODEL_PATH="${MODEL_PATH/snapshots\/auto/snapshots/$MODEL_SNAPSHOT_REV}"
  export MODEL_HOST_DIR MODEL_SNAPSHOT_REV MODEL_PATH

  local whome hub
  whome=$(sshw 'echo $HOME')
  hub="${WORKER_HF_HUB:-$whome/.cache/huggingface/hub}"
  WORKER_MODEL_HOST_DIR="${WORKER_MODEL_HOST_DIR:-$hub/$(basename "$MODEL_HOST_DIR")}"

  if [ "${SYNC_MODELS_TO_WORKER}" = "1" ]; then
    if [ "${FORCE_SYNC_MODELS_TO_WORKER}" = "1" ]; then
      rsync_cache_to_worker "$MODEL_HOST_DIR" "$WORKER_MODEL_HOST_DIR" "target"
    elif hf_cache_complete_remote "$WORKER_MODEL_HOST_DIR" "$MODEL_SNAPSHOT_REV"; then
      wlog "target already complete on worker — skip rsync ($WORKER_MODEL_HOST_DIR @$MODEL_SNAPSHOT_REV)"
    else
      wlog "target missing/incomplete on worker @$MODEL_SNAPSHOT_REV — rsync from head"
      rsync_cache_to_worker "$MODEL_HOST_DIR" "$WORKER_MODEL_HOST_DIR" "target"
    fi
  else
    WORKER_MODEL_HOST_DIR=$(detect_worker_model_dir) || return 1
  fi

  if ! hf_cache_complete_remote "$WORKER_MODEL_HOST_DIR" "$MODEL_SNAPSHOT_REV"; then
    err "worker target incomplete: $WORKER_MODEL_HOST_DIR @$MODEL_SNAPSHOT_REV"
    return 1
  fi
  export WORKER_MODEL_HOST_DIR
  wlog "worker cache complete (Omni target)"
}

detect_head_model_dirs() {
  # Lightweight path resolution for --check / early print (no download).
  if [ -z "$MODEL_HOST_DIR" ]; then
    local cand rev
    for cand in \
      "$HF_HOME_HOST/hub/$(hf_repo_cache_name "$TARGET_REPO")" \
      "/mnt/$(hf_repo_cache_name "$TARGET_REPO")" \
      "$REPO_DIR/$(hf_repo_cache_name "$TARGET_REPO")" \
      "$HF_HOME_HOST/hub/models--lukealonso--MiMo-V2.5-NVFP4" \
      "/mnt/models--lukealonso--MiMo-V2.5-NVFP4" \
      "$REPO_DIR/models--lukealonso--MiMo-V2.5-NVFP4" \
      "$HF_HOME_HOST/hub/models--mitomtuna--MiMo-V2.5-0703-NVFP4" \
      "/mnt/models--mitomtuna--MiMo-V2.5-0703-NVFP4"; do
      [ -d "$cand/snapshots" ] || continue
      if hf_cache_complete "$cand" "$TARGET_REVISION" 2>/dev/null; then
        MODEL_HOST_DIR="$cand"
        MODEL_SNAPSHOT_REV="$TARGET_REVISION"
        break
      fi
      for rev in $(ls -1 "$cand/snapshots" 2>/dev/null); do
        if hf_cache_complete "$cand" "$rev" 2>/dev/null; then
          MODEL_HOST_DIR="$cand"
          MODEL_SNAPSHOT_REV="$rev"
          break 2
        fi
      done
    done
    MODEL_HOST_DIR="${MODEL_HOST_DIR:-$(pick_or_default_cache "$TARGET_REPO")}"
  fi
  if [ -z "${MODEL_SNAPSHOT_REV:-}" ] && [ -d "${MODEL_HOST_DIR:-}/snapshots" ]; then
    if hf_cache_complete "$MODEL_HOST_DIR" "$TARGET_REVISION" 2>/dev/null; then
      MODEL_SNAPSHOT_REV="$TARGET_REVISION"
    else
      MODEL_SNAPSHOT_REV=$(first_snapshot_rev "$MODEL_HOST_DIR" || true)
    fi
  fi
  MODEL_SNAPSHOT_REV="${MODEL_SNAPSHOT_REV:-$TARGET_REVISION}"
  MODEL_PATH="${MODEL_PATH/snapshots\/auto/snapshots/$MODEL_SNAPSHOT_REV}"
  export MODEL_HOST_DIR MODEL_SNAPSHOT_REV MODEL_PATH
}

# On the WORKER prefer the local synced HF hub copy, then NFS mirrors.
detect_worker_model_dir() {
  local m whome hub tname
  whome=$(sshw 'echo $HOME')
  hub="${WORKER_HF_HUB:-$whome/.cache/huggingface/hub}"
  tname=$(basename "${MODEL_HOST_DIR:-$(hf_repo_cache_name "$TARGET_REPO")}")
  for cand in "$hub/$tname" \
              "$whome/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4" \
              "$WORKER_REPO_DIR/models--lukealonso--MiMo-V2.5-NVFP4" \
              "/mnt/spark1/models--lukealonso--MiMo-V2.5-NVFP4" \
              "/mnt/models--lukealonso--MiMo-V2.5-NVFP4" \
              "$whome/.cache/huggingface/hub/models--mitomtuna--MiMo-V2.5-0703-NVFP4" \
              "/mnt/spark1/models--mitomtuna--MiMo-V2.5-0703-NVFP4"; do
    if sshw test -d "$cand/snapshots" 2>/dev/null; then m="$cand"; break; fi
  done
  [ -n "$m" ] || { err "worker: Omni target model dir not found"; return 1; }
  echo "$m"
}

print_config() {
  local rev_short="${MODEL_SNAPSHOT_REV:0:12}"
  printf '%s%s┌─ config ──────────────────────────────────────────────────┐%s\n' "$C_DIM" "$C_BOLD" "$C_RESET"
  printf '  %-14s %s\n' "lane" "Omni / MTP${MTP_SPEC_TOKENS} / MiMoV2OmniForCausalLM"
  printf '  %-14s %s → %s\n' "cluster" "$HEAD_IP:$VLLM_PORT" "$WORKER_IP (TP1)"
  printf '  %-14s %s\n' "fabric" "$NCCL_SOCKET_IFNAME · $NCCL_IB_HCA · GID $NCCL_IB_GID_INDEX"
  printf '  %-14s %s\n' "model" "$SERVED_MODEL_NAME"
  printf '  %-14s %s @ %s…\n' "weights" "$TARGET_REPO" "$rev_short"
  printf '  %-14s %s\n' "head cache" "$MODEL_HOST_DIR"
  printf '  %-14s %s\n' "worker cache" "${WORKER_MODEL_HOST_DIR:-?}"
  printf '  %-14s %s ctx · %s seqs · batch %s · block %s · GMU %s\n' \
    "shape" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" "$BLOCK_SIZE" "$GPU_MEMORY_UTILIZATION"
  printf '  %-14s %s / %s\n' "KV / attn" "$KV_CACHE_DTYPE" "$ATTENTION_BACKEND"
  printf '  %-14s %s\n' "image" "$OVERLAY_TAG"
  printf '  %-14s download=%s  sync=%s  force_sync=%s  retries=%s\n' \
    "flags" "$AUTO_DOWNLOAD_MODELS" "$SYNC_MODELS_TO_WORKER" \
    "$FORCE_SYNC_MODELS_TO_WORKER" "$MAX_RELAY_ATTEMPTS"
  printf '%s%s└────────────────────────────────────────────────────────────┘%s\n' "$C_DIM" "$C_BOLD" "$C_RESET"
}

###############################################################################
# PRECHECK
###############################################################################
precheck() {
  local ensure="${1:-1}"   # 1 = download/sync models; 0 = detect-only
  log "GPU + SSH + weights"
  command -v docker >/dev/null || { err "docker not found on head"; exit 1; }
  command -v nvidia-smi >/dev/null || { err "nvidia-smi not found on head"; exit 1; }
  nvidia-smi -L >/dev/null 2>&1 || { err "no GPU on head"; exit 1; }
  hlog "head GPU: $(nvidia-smi -L | head -1)"

  log "testing SSH to worker ${SSH_USER}@${WORKER_IP} ..."
  if ! sshw 'true' 2>/dev/null; then
    err "cannot SSH to ${SSH_USER}@${WORKER_IP} (set SSH_USER / ensure key authorized)"
    exit 1
  fi
  sshw 'nvidia-smi -L >/dev/null 2>&1 && docker --version >/dev/null' || {
    err "worker missing nvidia-smi or docker"; exit 1; }
  wlog "worker GPU: $(sshw 'nvidia-smi -L | head -1')"

  if [ "$ensure" = "1" ]; then
    log "ensuring model weights (complete on head, synced to worker)"
    ensure_models || exit 1
  else
    detect_head_model_dirs
    WORKER_MODEL_HOST_DIR=$(detect_worker_model_dir 2>/dev/null) || true
    if hf_cache_complete "$MODEL_HOST_DIR" "$MODEL_SNAPSHOT_REV" 2>/dev/null; then
      hlog "Omni target looks complete: $MODEL_HOST_DIR @$MODEL_SNAPSHOT_REV"
    else
      hlog "Omni target missing/incomplete (run without --check to download): $MODEL_HOST_DIR"
    fi
  fi
  local nfiles
  nfiles=$(find "$MODEL_HOST_DIR/snapshots/$MODEL_SNAPSHOT_REV" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)
  hlog "head target: $MODEL_HOST_DIR @$MODEL_SNAPSHOT_REV ($nfiles snapshot entries)"
  wlog "worker target: ${WORKER_MODEL_HOST_DIR:-unset}"
  if [ -n "${WORKER_MODEL_HOST_DIR:-}" ]; then
    sshw "test -r '$WORKER_MODEL_HOST_DIR/snapshots/$MODEL_SNAPSHOT_REV/config.json'" \
      || { [ "$ensure" = "1" ] && { err "worker cannot read target config.json"; exit 1; }; true; }
  fi
}

###############################################################################
# Runtime image. Prefer pulling the published GHCR image (~20GB); fall back to
# building the local patches/Dockerfile overlay from BASE_IMAGE if pull fails.
###############################################################################
# Retag first matching local alias → OVERLAY_TAG. Returns 0 on success.
_retag_local_alias() { # $1 = head|worker
  local node="$1" aliases="$LOCAL_IMAGE_ALIASES" a
  [ -z "$aliases" ] && return 1
  IFS=',' read -ra _aliases <<< "$aliases"
  for a in "${_aliases[@]}"; do
    a="$(echo "$a" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ -z "$a" ] && continue
    if [ "$node" = head ]; then
      if docker image inspect "$a" >/dev/null 2>&1; then
        hlog "retagging local alias $a → $OVERLAY_TAG"
        docker tag "$a" "$OVERLAY_TAG"
        return 0
      fi
    else
      if sshw "docker image inspect '$a' >/dev/null 2>&1"; then
        wlog "retagging local alias $a → $OVERLAY_TAG on worker"
        sshw "docker tag '$a' '$OVERLAY_TAG'"
        return 0
      fi
    fi
  done
  return 1
}

# Copy OVERLAY_TAG from head local store → worker (no GHCR). ~20GB.
_transfer_overlay_head_to_worker() {
  wlog "transferring $OVERLAY_TAG from head → worker (local docker save/load, no GHCR)"
  docker image inspect "$OVERLAY_TAG" >/dev/null 2>&1 \
    || { err "head lacks $OVERLAY_TAG — cannot transfer"; return 1; }
  # Stream save|load; retag in case load only restores digest IDs.
  docker save "$OVERLAY_TAG" | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" \
    "docker load && docker tag '$OVERLAY_TAG' '$OVERLAY_TAG' 2>/dev/null || true"
  sshw "docker image inspect '$OVERLAY_TAG' >/dev/null 2>&1" \
    || { err "worker still missing $OVERLAY_TAG after transfer"; return 1; }
  wlog "worker now has $OVERLAY_TAG"
}

ensure_image_head() {
  hlog "ensuring runtime image on head: $OVERLAY_TAG"
  if docker image inspect "$OVERLAY_TAG" >/dev/null 2>&1; then
    hlog "image already present locally — skipping GHCR"
    return 0
  fi
  if _retag_local_alias head; then
    return 0
  fi
  if [ "$SKIP_IMAGE_PULL" = "1" ]; then
    err "head missing $OVERLAY_TAG and SKIP_IMAGE_PULL=1 (no GHCR)"
    return 1
  fi
  hlog "image not local — pulling $OVERLAY_TAG (~20GB) ..."
  if docker pull "$OVERLAY_TAG"; then
    hlog "image ready: $OVERLAY_TAG"
    return 0
  fi
  hlog "pull failed — building overlay from $BASE_IMAGE"
  if ! docker image inspect "$BASE_TAG" >/dev/null 2>&1; then
    [ "$SKIP_IMAGE_PULL" = "1" ] && { err "missing base $BASE_TAG"; return 1; }
    docker pull "$BASE_IMAGE" || return 1
    docker tag "$BASE_IMAGE" "$BASE_TAG"
  fi
  docker build -t "$OVERLAY_TAG" -f "$REPO_DIR/patches/Dockerfile" "$REPO_DIR/patches" || return 1
  hlog "overlay image ready: $OVERLAY_TAG"
}

ensure_image_worker() {
  wlog "ensuring runtime image on worker: $OVERLAY_TAG"
  local ctx="$WORKER_REPO_DIR/patches"
  if sshw "docker image inspect '$OVERLAY_TAG' >/dev/null 2>&1"; then
    wlog "image already present locally — skipping GHCR"
    return 0
  fi
  if _retag_local_alias worker; then
    return 0
  fi
  # Prefer head's local copy over a GHCR pull (private package / no worker login).
  if docker image inspect "$OVERLAY_TAG" >/dev/null 2>&1; then
    _transfer_overlay_head_to_worker || return 1
    return 0
  fi
  if [ "$SKIP_IMAGE_PULL" = "1" ]; then
    err "worker missing $OVERLAY_TAG and SKIP_IMAGE_PULL=1 (no GHCR)"
    return 1
  fi
  wlog "image not local — pulling $OVERLAY_TAG on worker (~20GB) ..."
  if sshw "docker pull '$OVERLAY_TAG'"; then
    wlog "image ready on worker"
    return 0
  fi
  wlog "pull failed — building overlay from $BASE_IMAGE on worker"
  sshw "
    set -e
    if ! docker image inspect '$BASE_TAG' >/dev/null 2>&1; then
      docker pull '$BASE_IMAGE'
      docker tag '$BASE_IMAGE' '$BASE_TAG'
    fi
    docker build -t '$OVERLAY_TAG' -f '$ctx/Dockerfile' '$ctx'
  " || { err "worker could not obtain $OVERLAY_TAG (local/GHCR/build all failed)"; return 1; }
  wlog "overlay ready on worker"
}

_container_running() { # $1 = head|worker
  if [ "$1" = head ]; then
    docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true
  else
    sshw "docker inspect -f '{{.State.Running}}' '$CONTAINER_NAME' 2>/dev/null" | grep -qx true
  fi
}

run_container_one() { # $1 = node tag head|worker, $2 = host MODEL dir, $3 = recipe dir
  local node="$1" mdir="$2" rdir="$3"
  local tag
  [ "$node" = head ] && tag=hlog || tag=wlog
  $tag "starting container $CONTAINER_NAME from $OVERLAY_TAG"
  if [ "$node" = head ]; then
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker run -d \
      --name "$CONTAINER_NAME" \
      --gpus all \
      --network host \
      --ipc host \
      --shm-size 16g \
      --workdir /workspace \
      --device /dev/infiniband:/dev/infiniband \
      --ulimit memlock=-1:-1 \
      --ulimit stack=67108864 \
      -v "$mdir:/model_cache:ro" \
      -v "$rdir:/workspace" \
      "$OVERLAY_TAG" sleep infinity \
      || { err "docker run failed on head"; return 1; }
  else
    sshw "
      docker rm -f '$CONTAINER_NAME' 2>/dev/null || true
      docker run -d \
        --name '$CONTAINER_NAME' \
        --gpus all \
        --network host \
        --ipc host \
        --shm-size 16g \
        --workdir /workspace \
        --device /dev/infiniband:/dev/infiniband \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864 \
        -v '$mdir:/model_cache:ro' \
        -v '$rdir:/workspace' \
        '$OVERLAY_TAG' sleep infinity
    " || { err "docker run failed on worker — image missing? last docker images:"; \
          sshw "docker images | head -20" >&2; return 1; }
  fi
  if ! _container_running "$node"; then
    err "$node container $CONTAINER_NAME is not running"
    if [ "$node" = head ]; then docker ps -a --filter "name=$CONTAINER_NAME" >&2
    else sshw "docker ps -a --filter name=$CONTAINER_NAME" >&2; fi
    return 1
  fi
  $tag "container up"
}

run_containers() {
  log "RUN CONTAINER on both nodes"
  ensure_image_head
  # rsync this repo (recipe + patches) to the worker so in-container /workspace/recipe is identical.
  wlog "syncing repo to worker:$WORKER_REPO_DIR"
  sshw "mkdir -p '$WORKER_REPO_DIR'"
  rsync -a --delete -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.git' --exclude 'vllm.log' --exclude 'retry.log' \
    "$REPO_DIR/" "${SSH_USER}@${WORKER_IP}:$WORKER_REPO_DIR/"
  ensure_image_worker
  run_container_one head "$MODEL_HOST_DIR" "$REPO_DIR"
  run_container_one worker "$WORKER_MODEL_HOST_DIR" "$WORKER_REPO_DIR"
}

###############################################################################
# MODS + PATCHES (inside the container). recipe/mods/ is absent in this repo
# (mods are baked into the base image), so apply-mods is attempted only if present;
# otherwise we verify registration at launch. The 7 engine patches (idempotent,
# each prints "already patched") are applied on BOTH nodes.
###############################################################################
ALL_PATCHES=(patch_mimo_v2_eagle3 patch_triton_noncausal patch_nc_fix patch_kv_page_lcm
             patch_aux_layer_off_by_one patch_diffkv_noncausal
             patch_draft_cache_auto patch_spec_dtype_guard)

apply_mods_patches_one() { # $1 = head|worker
  local node="$1" tag rdir
  [ "$node" = head ] && { tag=hlog; rdir="$REPO_DIR"; }
  [ "$node" = worker ] && { tag=wlog; rdir="$WORKER_REPO_DIR"; }
  $tag "applying mods + engine patches"

  # mods (only if recipe/mods shipped — here it doesn't, mods are in the image)
  if [ "$node" = head ]; then has_mods=$([ -d "$REPO_DIR/recipe/mods" ] && echo y || echo n)
  else has_mods=$(sshw "test -d '$WORKER_REPO_DIR/recipe/mods' && echo y || echo n"); fi
  if [ "$has_mods" = y ]; then
    $tag "recipe/mods present — running apply-mods.sh (6 base mods incl nvfp4-kv-diffkv)"
    if [ "$node" = head ]; then
      bash "$REPO_DIR/recipe/apply-mods.sh" "$CONTAINER_NAME"
    else
      sshw "bash '$WORKER_REPO_DIR/recipe/apply-mods.sh' '$CONTAINER_NAME'"
    fi
  else
    $tag "recipe/mods absent — assuming mods baked into image (will verify at launch)"
  fi

  # engine patches — tolerant of "already-evolved" anchors. The base image may
  # have partial ports baked in; a patch's edit() assertion can fire if the file
  # is already past an anchor. We capture failures and verify the FUNCTIONAL
  # end-state afterward, rather than failing on any single assertion.
  for p in "${ALL_PATCHES[@]}"; do
    $tag "patch: $p"
    if [ "$node" = head ]; then
      docker cp "$rdir/patches/$p.py" "$CONTAINER_NAME:/tmp/$p.py"
      docker exec "$CONTAINER_NAME" python3 "/tmp/$p.py" 2>&1 | tail -2 \
        || $tag "$p: assertion (likely already-evolved anchor) — will verify end-state"
    else
      sshw "docker cp '$rdir/patches/$p.py' '$CONTAINER_NAME:/tmp/$p.py' && \
            docker exec '$CONTAINER_NAME' python3 '/tmp/$p.py' 2>&1 | tail -2" \
        || $tag "$p: assertion (likely already-evolved anchor) — will verify end-state"
    fi
  done

  # CRITICAL end-state verification for the NVFP4-KV lane (both nodes must pass).
  $tag "verifying NVFP4-KV functional end-state"
  local verify_script='
import sys
def _say(m): print(m, flush=True)
ok = True
# 1) triton_attn_diffkv supports nvfp4 KV (nvfp4-kv-diffkv mod)
try:
    from vllm.v1.attention.backends.triton_attn_diffkv import TritonAttentionDiffKVBackend as B
    assert "nvfp4" in B.supported_kv_cache_dtypes, f"diffkv lacks nvfp4: {B.supported_kv_cache_dtypes}"
except Exception as e:
    _say(f"FAIL diffkv nvfp4: {e}"); ok = False
# 2) supports_non_causal flipped to True (patch_diffkv_noncausal)
try:
    assert B.supports_non_causal() is True, f"supports_non_causal={B.supports_non_causal()}"
except Exception as e:
    _say(f"FAIL non_causal: {e}"); ok = False
_say("END_STATE_OK" if ok else "END_STATE_FAIL")
sys.exit(0 if ok else 1)
'
  if [ "$node" = head ]; then
    printf '%s' "$verify_script" | docker exec -i "$CONTAINER_NAME" python3 - 2>&1 | tail -20
    [ "${PIPESTATUS[1]}" = 0 ] || { err "head end-state verification FAILED"; return 1; }
  else
    sshw "printf '%s' '$verify_script' | docker exec -i '$CONTAINER_NAME' python3 - 2>&1 | tail -20"
    [ $? = 0 ] || { err "worker end-state verification FAILED"; return 1; }
  fi
  $tag "mods + patches done (end-state verified)"
}

apply_mods_patches() {
  log "APPLY MODS + PATCHES on both nodes"
  apply_mods_patches_one head || return 1
  apply_mods_patches_one worker || return 1
}

###############################################################################
# RAY CLUSTER (head first, then worker; wait 2/2 GPUs before vLLM)
###############################################################################
ray_up_head() {
  hlog "starting Ray HEAD"
  # The base image bakes in stale RAY_OVERRIDE_NODE_IP_ADDRESS / VLLM_HOST_IP
  # from a different cluster (10.0.0.5/6). Unset them so --node-ip-address wins,
  # and pin RAY_ADDRESS to our head GCS.
  docker exec "$CONTAINER_NAME" bash -lc "
    unset RAY_OVERRIDE_NODE_IP_ADDRESS RAY_NODE_IP_ADDRESS
    export HEAD_ROCE_IP='$HEAD_ROCE_IP'
    export VLLM_HOST_IP='$HEAD_ROCE_IP'
    export RAY_TMPDIR=\${RAY_TMPDIR:-/dev/shm/ray}; mkdir -p \$RAY_TMPDIR
    export RAY_ADDRESS='$HEAD_ROCE_IP:$RAY_PORT'
    export NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
    export NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX NCCL_PROTO=$NCCL_PROTO NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS
    export NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL
	    rm -f /tmp/ray/ray_current_cluster
    ray stop --force || true
    ray start --head --port=$RAY_PORT --node-ip-address=$HEAD_ROCE_IP \
      --dashboard-host=0.0.0.0 --num-gpus=1 --object-store-memory=1073741824
  "
}

ray_up_worker() {
  wlog "worker joining Ray cluster"
  if ! _container_running worker; then
    err "worker has no running container $CONTAINER_NAME — cannot join Ray"
    sshw "docker ps -a --filter name=$CONTAINER_NAME; docker images | head -15" >&2 || true
    return 1
  fi
  sshw "
    docker exec '$CONTAINER_NAME' bash -lc '
      unset RAY_OVERRIDE_NODE_IP_ADDRESS RAY_NODE_IP_ADDRESS
      export HEAD_ROCE_IP=$HEAD_ROCE_IP WORKER_ROCE_IP=$WORKER_ROCE_IP
      export VLLM_HOST_IP=$WORKER_ROCE_IP
      export RAY_TMPDIR=\${RAY_TMPDIR:-/dev/shm/ray}; mkdir -p \$RAY_TMPDIR
      export RAY_ADDRESS=\"$HEAD_ROCE_IP:$RAY_PORT\"
      export NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
	      export NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$WORKER_NCCL_IB_GID_INDEX NCCL_PROTO=$NCCL_PROTO NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS
	      export NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL
	      rm -f /tmp/ray/ray_current_cluster
      ray stop --force || true
      ray start --address=$HEAD_ROCE_IP:$RAY_PORT --node-ip-address=$WORKER_ROCE_IP \
        --num-gpus=1 --object-store-memory=1073741824
    '
  " || { err "worker ray start failed"; sshw "docker logs --tail 40 $CONTAINER_NAME" >&2 || true; return 1; }
}

wait_2_gpus() {
  hlog "waiting for ray status to show 2.0/2.0 GPU (timeout ${RAY_WAIT_TIMEOUT}s)"
  local elapsed=0
  while [ $elapsed -lt $RAY_WAIT_TIMEOUT ]; do
    if ! _container_running worker; then
      echo
      err "worker container $CONTAINER_NAME disappeared while waiting for 2/2 GPU"
      sshw "docker ps -a --filter name=$CONTAINER_NAME; docker logs --tail 40 $CONTAINER_NAME" >&2 || true
      return 1
    fi
    if docker exec -e RAY_ADDRESS="$HEAD_ROCE_IP:$RAY_PORT" "$CONTAINER_NAME" ray status 2>/dev/null | grep -qE '2\.0(/2\.0)? GPU'; then
      hlog "Ray cluster healthy: 2.0/2.0 GPU"
      docker exec -e RAY_ADDRESS="$HEAD_ROCE_IP:$RAY_PORT" "$CONTAINER_NAME" ray status 2>/dev/null | sed 's/^/    /'
      return 0
    fi
    sleep 5; elapsed=$((elapsed+5))
    printf '.'
    if [ $((elapsed % 60)) -eq 0 ]; then
      hlog "still waiting for 2/2 GPU (${elapsed}s)"
      docker exec -e RAY_ADDRESS="$HEAD_ROCE_IP:$RAY_PORT" "$CONTAINER_NAME" ray status 2>/dev/null \
        | sed 's/^/    /' || true
      sshw "docker inspect -f 'worker container={{.State.Status}}' $CONTAINER_NAME 2>/dev/null || echo 'worker container=MISSING'" \
        | sed 's/^/    /' || true
    fi
  done
  echo
  err "Ray did not reach 2.0/2.0 GPU in ${RAY_WAIT_TIMEOUT}s"
  docker exec -e RAY_ADDRESS="$HEAD_ROCE_IP:$RAY_PORT" "$CONTAINER_NAME" ray status 2>&1 | sed 's/^/    /'
  sshw "docker ps -a --filter name=$CONTAINER_NAME; docker logs --tail 40 $CONTAINER_NAME" 2>&1 | sed 's/^/    /' || true
  return 1
}

ray_up() {
  log "RAY CLUSTER bring-up (head first)"
  ray_up_head
  ray_up_worker
  wait_2_gpus
}

###############################################################################
# ENV + LAUNCH Omni MTP1 vLLM on the HEAD only (background).
# recipe/launch-omni.sh uses API_PORT (default 8888) — no sed port hack needed.
###############################################################################
launch_vllm() {
  hlog "ENV + LAUNCH Omni MTP1 on port $VLLM_PORT (NVFP4-KV, 1M ctx)"

  docker exec "$CONTAINER_NAME" bash -lc '
    for pid in $(pgrep -f "vllm serve" 2>/dev/null); do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in $(pgrep -f launch-omni 2>/dev/null); do kill -TERM "$pid" 2>/dev/null || true; done
    sleep 1
    exit 0
  ' || true

  : > "$VLLM_LOG"

  docker exec "$CONTAINER_NAME" bash -lc '
    set -e
    cd /workspace/recipe
    source /workspace/recipe/env.sh

    # Omni shape overrides (env.sh GMU may be 0.84; we pin validated 0.83 @ 1M).
    export MAX_MODEL_LEN='"$MAX_MODEL_LEN"'
    export MAX_NUM_SEQS='"$MAX_NUM_SEQS"'
    export MAX_NUM_BATCHED_TOKENS='"$MAX_NUM_BATCHED_TOKENS"'
    export BLOCK_SIZE='"$BLOCK_SIZE"'
    export GPU_MEMORY_UTILIZATION='"$GPU_MEMORY_UTILIZATION"'
    export TENSOR_PARALLEL_SIZE='"$TENSOR_PARALLEL_SIZE"'
    export ENFORCE_EAGER='"$ENFORCE_EAGER"'
    export KV_CACHE_DTYPE='"$KV_CACHE_DTYPE"'
    export ATTENTION_BACKEND='"$ATTENTION_BACKEND"'
    export ENABLE_MTP='"$ENABLE_MTP"'
    export MTP_SPEC_TOKENS='"$MTP_SPEC_TOKENS"'
    export VLLM_MIMO_MTP1_GREEDY_FAST='"$VLLM_MIMO_MTP1_GREEDY_FAST"'
    export SPECULATIVE_CONFIG='"'"'{"method":"mtp","num_speculative_tokens":'"$MTP_SPEC_TOKENS"',"use_local_argmax_reduction":false}'"'"'

    export MODEL_PATH="'"$MODEL_PATH"'"
    export SERVED_MODEL_NAME="'"$SERVED_MODEL_NAME"'"
    export API_PORT='"$VLLM_PORT"'
    export HEAD_ROCE_IP="'"$HEAD_ROCE_IP"'"
    export VLLM_HOST_IP="'"$HEAD_ROCE_IP"'"

    export NCCL_IB_DISABLE='"$NCCL_IB_DISABLE"'
    export NCCL_SOCKET_IFNAME='"$NCCL_SOCKET_IFNAME"'
    export GLOO_SOCKET_IFNAME='"$NCCL_SOCKET_IFNAME"'
    export NCCL_IB_HCA='"$NCCL_IB_HCA"'
    export NCCL_IB_GID_INDEX='"$NCCL_IB_GID_INDEX"'
    export NCCL_PROTO='"${NCCL_PROTO:-LL}"'
    export NCCL_MAX_NCHANNELS='"${NCCL_MAX_NCHANNELS:-2}"'
    export NCCL_NET_GDR_LEVEL='"${NCCL_NET_GDR_LEVEL:-LOC}"'
    if [ '"$NCCL_IB_DISABLE"' = "1" ]; then
      unset NCCL_NET NCCL_NET_PLUGIN
    fi

    echo "=== launching Omni MTP1 (log -> /workspace/vllm.log) ==="
    nohup bash /workspace/recipe/launch-omni.sh > /workspace/vllm.log 2>&1 &
    echo $! > /tmp/vllm.pid
    echo "vllm launched, pid=$(cat /tmp/vllm.pid)"
  '
  hlog "vLLM launching in background (logs: $VLLM_LOG)"
}

###############################################################################
# VERIFY + SELF-HEAL
###############################################################################
# Check vLLM is alive in the container.
vllm_alive() {
  docker exec "$CONTAINER_NAME" bash -lc '[ -f /tmp/vllm.pid ] && kill -0 "$(cat /tmp/vllm.pid)" 2>/dev/null' 2>/dev/null
}

wait_for_models() {
  local i elapsed=0 max_s=$((MAX_BOOT_ATTEMPTS * BOOT_POLL_INTERVAL))
  local est_min=15 max_min=$(( (max_s + 59) / 60 ))
  vlog "waiting for /v1/models (timeout ~${max_min} min) …"
  # One-shot notice — shard progress is already in vllm.log / docker logs.
  printf '%s%s[%s]%s %s%snotice%s  This could take a while (~%s min typical for weight load + KV profile; timeout ~%s min).\n' \
    "$C_DIM" "$C_BOLD" "$(_ts)" "$C_RESET" "$C_YELLOW" "$C_BOLD" "$C_RESET" "$est_min" "$max_min"
  for i in $(seq 1 "$MAX_BOOT_ATTEMPTS"); do
    local resp
    resp=$(curl -sS --max-time 5 "http://$HEAD_IP:$VLLM_PORT/v1/models" 2>/dev/null || true)
    if echo "$resp" | grep -q "$SERVED_MODEL_NAME"; then
      [[ -t 1 ]] && printf '\n'
      ok "model live after ${elapsed}s · $SERVED_MODEL_NAME"
      return 0
    fi
    if ! vllm_alive; then
      [[ -t 1 ]] && printf '\n'
      err "vLLM process died while booting — last log lines:"
      tail -n 30 "$VLLM_LOG" 2>/dev/null | sed 's/^/    /'
      return 1
    fi
    sleep "$BOOT_POLL_INTERVAL"
    elapsed=$((elapsed + BOOT_POLL_INTERVAL))
    # In-place heartbeat every poll when TTY; fuller status every ~60s
    if [[ -t 1 ]]; then
      printf '\r%s%s[%s]%s %s%sverify%s  booting… %ss / %ss   ' \
        "$C_DIM" "$C_BOLD" "$(_ts)" "$C_RESET" "$C_YELLOW" "$C_BOLD" "$C_RESET" "$elapsed" "$max_s"
    fi
    if [ $((elapsed % 60)) -eq 0 ]; then
      [[ -t 1 ]] && printf '\n'
      vlog "still loading (${elapsed}s) — log:"
      tail -n 2 "$VLLM_LOG" 2>/dev/null | sed 's/^/    /' || true
    fi
  done
  [[ -t 1 ]] && printf '\n'
  err "timed out waiting for /v1/models after ${max_s}s"
  return 1
}

do_chat() {  # echoes the content on success, returns nonzero on empty/5xx
  # NOTE: vlog must go to stderr — callers capture stdout via reply=$(do_chat).
  local body json code content
  # Cold first request after 1M-ctx load can exceed 120s; allow 5 min.
  body=$(curl -sS --max-time 300 -w '\n%{http_code}' \
    "http://$HEAD_IP:$VLLM_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$SERVED_MODEL_NAME"'","messages":[{"role":"user","content":"'"$VERIFY_MODEL_PROMPT"'"}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    2>/dev/null || true)
  code=$(printf '%s\n' "$body" | tail -1)
  json=$(printf '%s\n' "$body" | sed '$d')
  content=$(printf '%s\n' "$json" | python3 -c 'import sys,json,re
try:
    j=json.load(sys.stdin)
    m=j["choices"][0]["message"]
    c=m.get("content") or ""
    if not str(c).strip():
        c=m.get("reasoning_content") or ""
    # Strip leftover think tags if template still emits them.
    c=re.sub(r"<think>.*?</think>", "", str(c), flags=re.S).strip()
    print(c)
except Exception:
    print("")
' 2>/dev/null || true)
  vlog "chat HTTP=$code content_len=${#content}" >&2
  if [ "$code" != "200" ] || [ -z "$content" ]; then
    # Keep a short raw sample in the verify stream for diagnosis (stderr).
    printf '%s\n' "$json" | head -c 600 | sed 's/^/    raw: /' >&2 || true
    echo >&2
    return 1
  fi
  printf '%s\n' "$content"
  return 0
}

teardown_cluster() {
  log "TEARDOWN"
  docker exec "$CONTAINER_NAME" bash -lc 'ray stop --force 2>/dev/null || true; pkill -f "vllm serve" 2>/dev/null || true' 2>/dev/null || true
  sshw "docker exec '$CONTAINER_NAME' bash -lc 'ray stop --force 2>/dev/null || true; pkill -f \"vllm serve\" 2>/dev/null || true'" 2>/dev/null || true
  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
  sshw "docker rm -f '$CONTAINER_NAME' 2>/dev/null || true"
  log "teardown done"
}

# Diagnose a failure: dump rays status, gpu, tail of vllm log.
diagnose() {
  vlog "=== diagnosis ==="
  hlog "ray status (head):"
  docker exec -e RAY_ADDRESS="$HEAD_ROCE_IP:$RAY_PORT" "$CONTAINER_NAME" ray status 2>&1 | sed 's/^/    /' || true
  hlog "GPU (head): "
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/    /' || true
  vlog "vllm.log tail (last 60):"
  tail -n 60 "$VLLM_LOG" 2>/dev/null | sed 's/^/    /' || true
}

# Apply a fix based on observed symptoms. $1 = attempt number.
apply_fix() {
  local n="$1"
  # symptom-driven knobs — each attempt nudges a different common cause.
  case "$n" in
    1) retrylog "fix1: re-apply NVFP4-KV / noncausal path (mods + patches)"
       true ;;
    2) retrylog "fix2: lower GMU to 0.80 + seqs 4 (OOM/quickload headroom at 1M)"
       export GPU_MEMORY_UTILIZATION=0.80 MAX_NUM_SEQS=4 ;;
    3) retrylog "fix3: fall back to NCCL TCP sockets if RoCE stalls"
       export NCCL_IB_DISABLE=1 ;;
    4) retrylog "fix4: reduce MAX_MODEL_LEN to 500000 (1M may not fit on this fabric/ram)"
       export MAX_MODEL_LEN=500000 MAX_NUM_SEQS=4 ;;
    5) retrylog "fix5: conservative Omni profile (GMU 0.82, seqs 2, 500K)"
       export GPU_MEMORY_UTILIZATION=0.82 MAX_NUM_SEQS=2 MAX_MODEL_LEN=500000 ;;
    *) retrylog "fix$n: no further automated fixes" ;;
  esac
}

###############################################################################
# MAIN: full bring-up + verify, with a relay retry loop around the whole stack.
###############################################################################
bringup_and_verify() {
  : > "$RETRY_LOG"
  local attempt=0 total_steps=7
  while [ $attempt -lt "$MAX_RELAY_ATTEMPTS" ]; do
    attempt=$((attempt+1))
    if [ "$attempt" -gt 1 ]; then
      retrylog "relay $attempt/$MAX_RELAY_ATTEMPTS — applying fix then full restack"
    fi

    step 1 "$total_steps" "Containers (head + worker)"
    run_containers || { diagnose; apply_fix "$attempt"; teardown_cluster; continue; }
    ok "containers up"

    step 2 "$total_steps" "Mods + engine patches"
    apply_mods_patches || { diagnose; apply_fix "$attempt"; teardown_cluster; continue; }
    ok "mods + patches applied"

    step 3 "$total_steps" "Ray cluster (2× GPU)"
    if ! ray_up; then
      diagnose; apply_fix "$attempt"; teardown_cluster; continue
    fi
    ok "Ray healthy · 2.0/2.0 GPU"

    step 4 "$total_steps" "Launch Omni MTP1 on :$VLLM_PORT"
    launch_vllm
    ok "vLLM process started (log: $VLLM_LOG)"

    step 5 "$total_steps" "Wait until /v1/models is ready"
    if ! wait_for_models; then
      vlog "API not up — diagnose + relay"
      diagnose; apply_fix "$attempt"; teardown_cluster; continue
    fi

    step 6 "$total_steps" "Chat smoke test"
    vlog "POST /v1/chat/completions …"
    if reply=$(do_chat) && [ -n "$reply" ]; then
      step 7 "$total_steps" "Done"
      success_banner "$reply"
      print_config
      exit 0
    fi
    vlog "chat failed/empty — diagnose + relay"
    diagnose; apply_fix "$attempt"; teardown_cluster
  done

  err "exhausted $MAX_RELAY_ATTEMPTS relay attempts — chat never confirmed"
  printf '\n%sLast log:%s\n' "$C_BOLD" "$C_RESET"
  tail -n 80 "$VLLM_LOG" 2>/dev/null | sed 's/^/    /' || true
  exit 1
}

###############################################################################
# Arg dispatch
###############################################################################
main() {
  local mode=run
  case "${1:-}" in
    --check) mode=check ;;
    --teardown) mode=teardown ;;
    -h|--help)
      banner "help"
      cat <<EOF
  ${C_BOLD}Usage${C_RESET}
    bash start.sh              Full Omni bring-up + chat verify
    bash start.sh --check      Plan + weight check (no launch)
    bash start.sh --teardown   Stop Ray/vLLM/containers on both nodes
    bash stop.sh               Stop serve (lighter than --teardown)

  ${C_BOLD}Useful env${C_RESET}
    MAX_MODEL_LEN  MAX_NUM_SEQS  GPU_MEMORY_UTILIZATION
    TARGET_REPO  TARGET_REVISION  MODEL_HOST_DIR
    AUTO_DOWNLOAD_MODELS=0  SYNC_MODELS_TO_WORKER=0  NO_COLOR=1

EOF
      exit 0 ;;
  esac

  case "$mode" in
    teardown)
      detect_head_model_dirs || true
      teardown_cluster; exit 0 ;;
  esac

  case "$mode" in
    check) banner "dry-run / --check" ;;
    *)     banner "full bring-up + verify" ;;
  esac

  step 0 7 "Precheck + weights"
  if [ "$mode" = check ]; then
    precheck 0
  else
    precheck 1
  fi
  print_config | tee "$REPO_DIR/last-config.txt"
  ok "precheck passed"

  case "$mode" in
    check)
      printf '
%s%sWould run:%s
' "$C_BOLD" "$C_CYAN" "$C_RESET"
      printf '  1. containers from %s
' "$OVERLAY_TAG"
      printf '  2. mods + engine patches (both nodes)
'
      printf '  3. Ray head → worker → 2/2 GPU
'
      printf '  4. launch-omni.sh (MTP1) on :%s
' "$VLLM_PORT"
      printf '  5. poll /v1/models → chat verify (up to %s relays)

' "$MAX_RELAY_ATTEMPTS"
      ok "check complete — no changes made"
      exit 0 ;;
    run) bringup_and_verify ;;
  esac
}

main "$@"
