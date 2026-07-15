#!/usr/bin/env bash
# crash_regression_smoke.sh — reusable OpenAI-compatible crash / stability smoke
#
# Motivated by the dual-Spark MiMo multimodal crash:
#   image + text request with max_tokens ≈ max_model_len killed EngineCore
#   (ValueError: Error during index put operation / cudaErrorNotPermitted).
#
# What this proves:
#   1. API is up (/v1/models)
#   2. Short text completion works
#   3. Extreme max_tokens does NOT kill the server (clamp or graceful fail)
#   4. Multimodal image+text works (or fails the *request* without killing API)
#   5. Optional: bug-report combo (image + extreme max_tokens)
#   6. Final health check — server still answers /v1/models
#
# Usage:
#   bash scripts/crash_regression_smoke.sh
#   BASE_URL=http://host:8888/v1 MODEL=MyModel bash scripts/crash_regression_smoke.sh
#   bash scripts/crash_regression_smoke.sh --image /path/to.png --extreme-mm
#   bash scripts/crash_regression_smoke.sh --skip-mm          # text-only suite
#   bash scripts/crash_regression_smoke.sh --help
#
# Exit codes:
#   0  all enabled cases passed (and server still healthy)
#   1  a required case failed or server died
#   2  bad args / missing deps
set -euo pipefail

###############################################################################
# Defaults (override via env or flags)
###############################################################################
BASE_URL="${BASE_URL:-http://10.0.0.1:8888/v1}"
MODEL="${MODEL:-MiMo-V2.5-NVFP4}"
API_KEY="${API_KEY:-dummy}"
TIMEOUT_S="${TIMEOUT_S:-300}"          # per-request curl timeout
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-15}"
EXTREME_MAX_TOKENS="${EXTREME_MAX_TOKENS:-999078}"  # bug-report value
SAFE_MAX_TOKENS="${SAFE_MAX_TOKENS:-64}"
MM_SAFE_MAX_TOKENS="${MM_SAFE_MAX_TOKENS:-64}"
# Default image: repo hero (small, local). Override with --image /path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_PATH="${IMAGE_PATH:-$REPO_DIR/assets/mimo.jpg}"
RUN_MM=1
RUN_EXTREME_TEXT=1
RUN_EXTREME_MM=0                       # off by default; enable with --extreme-mm
CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
TEMPERATURE="${TEMPERATURE:-0}"
WORKDIR="${WORKDIR:-}"                 # empty → mktemp
VERBOSE="${VERBOSE:-0}"

###############################################################################
# CLI
###############################################################################
usage() {
  sed -n '2,30p' "$0" | sed 's/^# \?//'
  cat <<EOF

Flags:
  --base-url URL          OpenAI base (default: $BASE_URL)
  --model ID              model id (default: $MODEL)
  --image PATH            local image for multimodal cases
  --extreme N             extreme max_tokens value (default: $EXTREME_MAX_TOKENS)
  --extreme-mm            also run image + extreme max_tokens (bug-report combo)
  --skip-mm               skip all multimodal cases
  --skip-extreme-text     skip text extreme max_tokens case
  --timeout S             per-request timeout seconds (default: $TIMEOUT_S)
  --workdir DIR           save request/response artifacts here
  -v, --verbose           print response snippets
  -h, --help              this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --image) IMAGE_PATH="$2"; shift 2 ;;
    --extreme) EXTREME_MAX_TOKENS="$2"; shift 2 ;;
    --extreme-mm) RUN_EXTREME_MM=1; shift ;;
    --skip-mm) RUN_MM=0; RUN_EXTREME_MM=0; shift ;;
    --skip-extreme-text) RUN_EXTREME_TEXT=0; shift ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Strip trailing slash for consistency
BASE_URL="${BASE_URL%/}"

command -v curl >/dev/null || { echo "need curl" >&2; exit 2; }
command -v python3 >/dev/null || { echo "need python3" >&2; exit 2; }
command -v base64 >/dev/null || { echo "need base64" >&2; exit 2; }

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(mktemp -d -t crash-smoke-XXXXXX)"
else
  mkdir -p "$WORKDIR"
fi
echo "workdir: $WORKDIR"

###############################################################################
# Helpers
###############################################################################
C_OK=$'\033[32m'; C_FAIL=$'\033[31m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
if [[ ! -t 1 ]]; then C_OK=; C_FAIL=; C_DIM=; C_BOLD=; C_RESET=; fi

PASS=0
FAIL=0
SKIP=0
RESULTS=()

log()  { printf '%s%s%s %s\n' "$C_DIM" "$(date +%H:%M:%S)" "$C_RESET" "$*"; }
ok()   { PASS=$((PASS+1)); RESULTS+=("PASS  $1"); printf '%sPASS%s  %s\n' "$C_OK" "$C_RESET" "$1"; }
fail() { FAIL=$((FAIL+1)); RESULTS+=("FAIL  $1 — $2"); printf '%sFAIL%s  %s — %s\n' "$C_FAIL" "$C_RESET" "$1" "$2"; }
skip() { SKIP=$((SKIP+1)); RESULTS+=("SKIP  $1 — $2"); printf '%sSKIP%s  %s — %s\n' "$C_DIM" "$C_RESET" "$1" "$2"; }

auth_hdr=()
if [[ -n "$API_KEY" && "$API_KEY" != "none" ]]; then
  auth_hdr=(-H "Authorization: Bearer $API_KEY")
fi

# GET health. Prints body to stdout on success; returns nonzero on failure.
health() {
  local out code
  out=$(curl -sS --max-time "$HEALTH_TIMEOUT_S" -w '\n%{http_code}' \
    "${auth_hdr[@]}" "$BASE_URL/models" 2>"$WORKDIR/health.err" || true)
  code=$(printf '%s\n' "$out" | tail -1)
  body=$(printf '%s\n' "$out" | sed '$d')
  printf '%s' "$body" > "$WORKDIR/health.json"
  [[ "$code" == "200" ]] || return 1
  # Must list the model (or at least return data[])
  python3 - "$MODEL" "$WORKDIR/health.json" <<'PY'
import json,sys
model, path = sys.argv[1], sys.argv[2]
j=json.load(open(path))
ids=[m.get("id") for m in j.get("data",[])]
if model and model not in ids and ids:
    # soft warn only — some servers rename models
    print(f"note: model {model!r} not in {ids}", file=sys.stderr)
if not ids:
    sys.exit(1)
print(",".join(ids))
PY
}

# POST chat completion. Args: name max_tokens content_json_array_or_string
# content is a JSON expression for messages[0].content (string or array).
# Sets globals: LAST_HTTP LAST_BODY LAST_CONTENT LAST_ERR
chat() {
  local name="$1" max_tokens="$2" content_json="$3"
  local req="$WORKDIR/${name}.req.json"
  local resp="$WORKDIR/${name}.resp.json"
  local meta="$WORKDIR/${name}.meta"

  python3 - "$req" "$MODEL" "$max_tokens" "$TEMPERATURE" "$CHAT_TEMPLATE_KWARGS" "$content_json" <<'PY'
import json,sys
path, model, max_tokens, temp, ctk, content_raw = sys.argv[1:7]
content = json.loads(content_raw)
body = {
    "model": model,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": int(max_tokens),
    "temperature": float(temp),
}
try:
    kw = json.loads(ctk) if ctk else None
except Exception:
    kw = None
if kw:
    body["chat_template_kwargs"] = kw
json.dump(body, open(path,"w"), indent=2)
print(path)
PY

  local out code
  out=$(curl -sS --max-time "$TIMEOUT_S" -w '\n%{http_code}' \
    "${auth_hdr[@]}" \
    -H 'Content-Type: application/json' \
    -d @"$req" \
    "$BASE_URL/chat/completions" 2>"$WORKDIR/${name}.curl.err" || true)
  code=$(printf '%s\n' "$out" | tail -1)
  body=$(printf '%s\n' "$out" | sed '$d')
  printf '%s' "$body" > "$resp"
  printf 'http=%s\n' "$code" > "$meta"
  LAST_HTTP="$code"
  LAST_BODY="$body"
  LAST_CONTENT=$(python3 - <<PY
import json,re,sys
try:
    j=json.loads(open("$resp").read() or "{}")
    m=(j.get("choices") or [{}])[0].get("message") or {}
    c=m.get("content") or m.get("reasoning_content") or ""
    c=re.sub(r"<think>.*?</think>", "", str(c), flags=re.S).strip()
    print(c)
except Exception as e:
    print("")
PY
)
  LAST_ERR=$(cat "$WORKDIR/${name}.curl.err" 2>/dev/null || true)
  if [[ "$VERBOSE" == "1" ]]; then
    log "  http=$LAST_HTTP content=${LAST_CONTENT:0:120}"
  fi
}

# After a case: is the API still alive?
assert_alive() {
  local label="$1"
  if health >/dev/null 2>&1; then
    return 0
  fi
  fail "$label" "server DEAD after case (health check failed) — this is the crash class of bug"
  return 1
}

image_data_url() {
  local path="$1"
  [[ -f "$path" ]] || { echo "image not found: $path" >&2; return 1; }
  local mime
  case "${path,,}" in
    *.png) mime=image/png ;;
    *.jpg|*.jpeg) mime=image/jpeg ;;
    *.webp) mime=image/webp ;;
    *.gif) mime=image/gif ;;
    *) mime=image/jpeg ;;
  esac
  # data URL (OpenAI-compatible image_url)
  printf 'data:%s;base64,%s' "$mime" "$(base64 -w0 "$path" 2>/dev/null || base64 "$path" | tr -d '\n')"
}

###############################################################################
# Cases
###############################################################################
log "BASE_URL=$BASE_URL  MODEL=$MODEL  extreme_max_tokens=$EXTREME_MAX_TOKENS"
log "image=$IMAGE_PATH  mm=$RUN_MM extreme_mm=$RUN_EXTREME_MM"

# --- 0. Health ---
if ids=$(health); then
  ok "health /v1/models (ids=$ids)"
else
  fail "health /v1/models" "cannot reach $BASE_URL/models — is the server up?"
  printf '\n%s%sSUMMARY%s  pass=%s fail=%s skip=%s\n' "$C_BOLD" "$C_FAIL" "$C_RESET" "$PASS" "$FAIL" "$SKIP"
  exit 1
fi

# --- 1. Safe text ---
chat "01_text_safe" "$SAFE_MAX_TOKENS" '"Reply with just the word: ok"'
if [[ "$LAST_HTTP" == "200" && -n "$LAST_CONTENT" ]]; then
  ok "text safe max_tokens=$SAFE_MAX_TOKENS (http=200 content=$(printf '%s' "$LAST_CONTENT" | head -c 40))"
else
  fail "text safe" "http=$LAST_HTTP content=${LAST_CONTENT:0:80} err=$LAST_ERR"
fi
assert_alive "after text safe" || true

# --- 2. Extreme max_tokens (text) — bug-adjacent: must NOT kill server ---
if [[ "$RUN_EXTREME_TEXT" == "1" ]]; then
  chat "02_text_extreme" "$EXTREME_MAX_TOKENS" '"Reply with just the word: ok"'
  # Success criteria for this case:
  #   A) request completes (2xx/4xx) AND server still healthy, OR
  #   B) request fails with transport error BUT only if server still healthy
  #      (timeout under huge reserved work can happen; death is the bug)
  if ! assert_alive "after text extreme max_tokens=$EXTREME_MAX_TOKENS"; then
    : # fail already recorded
  elif [[ "$LAST_HTTP" == "200" && -n "$LAST_CONTENT" ]]; then
    ok "text extreme max_tokens=$EXTREME_MAX_TOKENS survived (http=200, content non-empty)"
  elif [[ "$LAST_HTTP" =~ ^[45][0-9][0-9]$ ]]; then
    ok "text extreme max_tokens=$EXTREME_MAX_TOKENS rejected/errored gracefully (http=$LAST_HTTP, server alive)"
  elif [[ -z "$LAST_HTTP" || "$LAST_HTTP" == "000" ]]; then
    # curl timeout / connection issue but health OK → soft pass with note
    ok "text extreme max_tokens=$EXTREME_MAX_TOKENS timed out/transport fail but server ALIVE (http=${LAST_HTTP:-none})"
  else
    ok "text extreme max_tokens=$EXTREME_MAX_TOKENS http=$LAST_HTTP server ALIVE"
  fi
else
  skip "text extreme" "disabled"
fi

# --- 3. Multimodal safe ---
if [[ "$RUN_MM" == "1" ]]; then
  if [[ ! -f "$IMAGE_PATH" ]]; then
    skip "mm safe" "image missing: $IMAGE_PATH"
  else
    DATA_URL=$(image_data_url "$IMAGE_PATH")
    # content as multimodal array
    MM_CONTENT=$(python3 -c 'import json,sys; print(json.dumps([
      {"type":"text","text":"Describe this image in one short sentence. Reply with just the description."},
      {"type":"image_url","image_url":{"url":sys.argv[1]}},
    ]))' "$DATA_URL")
    chat "03_mm_safe" "$MM_SAFE_MAX_TOKENS" "$MM_CONTENT"
    if ! assert_alive "after mm safe"; then
      :
    elif [[ "$LAST_HTTP" == "200" && -n "$LAST_CONTENT" ]]; then
      ok "mm safe max_tokens=$MM_SAFE_MAX_TOKENS (http=200 content=$(printf '%s' "$LAST_CONTENT" | head -c 60 | tr '\n' ' '))"
    elif [[ "$LAST_HTTP" =~ ^4[0-9][0-9]$ ]]; then
      # 4xx: model may not accept this image format — not a crash
      ok "mm safe rejected request (http=$LAST_HTTP) but server ALIVE"
    else
      fail "mm safe" "http=$LAST_HTTP content=${LAST_CONTENT:0:80} err=$LAST_ERR (server still alive)"
    fi
  fi
else
  skip "mm safe" "disabled"
fi

# --- 4. Multimodal + extreme max_tokens (exact bug-report shape) ---
if [[ "$RUN_EXTREME_MM" == "1" ]]; then
  if [[ ! -f "$IMAGE_PATH" ]]; then
    skip "mm extreme" "image missing: $IMAGE_PATH"
  else
    log "running bug-report combo: image + max_tokens=$EXTREME_MAX_TOKENS"
    log "if this kills EngineCore you will need a full restack (~12 min)"
    DATA_URL=$(image_data_url "$IMAGE_PATH")
    MM_CONTENT=$(python3 -c 'import json,sys; print(json.dumps([
      {"type":"text","text":"What do you see? One short sentence."},
      {"type":"image_url","image_url":{"url":sys.argv[1]}},
    ]))' "$DATA_URL")
    chat "04_mm_extreme" "$EXTREME_MAX_TOKENS" "$MM_CONTENT"
    # THE critical assertion: server must still be healthy.
    if ! health >/dev/null 2>&1; then
      fail "mm extreme max_tokens=$EXTREME_MAX_TOKENS" \
        "SERVER DIED — reproduced the EngineCore crash class. Check vllm.log"
    elif [[ "$LAST_HTTP" == "200" ]]; then
      ok "mm extreme max_tokens=$EXTREME_MAX_TOKENS survived (http=200, server alive)"
    elif [[ "$LAST_HTTP" =~ ^[45][0-9][0-9]$ ]]; then
      ok "mm extreme max_tokens=$EXTREME_MAX_TOKENS request failed gracefully (http=$LAST_HTTP, server alive)"
    else
      ok "mm extreme max_tokens=$EXTREME_MAX_TOKENS transport/http=$LAST_HTTP but server ALIVE"
    fi
  fi
else
  skip "mm extreme" "pass --extreme-mm to enable bug-report combo"
fi

# --- 5. Final health ---
if health >/dev/null 2>&1; then
  ok "final health /v1/models still up"
else
  fail "final health" "server not responding at end of suite"
fi

###############################################################################
# Summary
###############################################################################
{
  echo "BASE_URL=$BASE_URL MODEL=$MODEL"
  echo "EXTREME_MAX_TOKENS=$EXTREME_MAX_TOKENS IMAGE=$IMAGE_PATH"
  printf '%s\n' "${RESULTS[@]}"
  echo "pass=$PASS fail=$FAIL skip=$SKIP"
} | tee "$WORKDIR/summary.txt"

printf '\n%sSUMMARY%s  pass=%s fail=%s skip=%s  workdir=%s\n' \
  "$C_BOLD" "$C_RESET" "$PASS" "$FAIL" "$SKIP" "$WORKDIR"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
exit 0
