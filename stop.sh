#!/usr/bin/env bash
# =============================================================================
# stop.sh — Stop the MiMo-V2.5 Docker containers (head + worker).
#
# Stops vLLM + Ray inside the container, then stops the container.
# Does NOT remove the container — only stops it.
# No prompts, no verbose Ray spew.
#
# Usage:
#   bash stop.sh               # full stop (default)
#   bash stop.sh --check       # dry run
#   bash stop.sh --help        # this help
# =============================================================================
set -euo pipefail

###############################################################################
# CONFIG BLOCK
###############################################################################
HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
SSH_USER="${SSH_USER:-zurih}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8)
HEAD_ROCE_IP="${HEAD_ROCE_IP:-$HEAD_IP}"
WORKER_ROCE_IP="${WORKER_ROCE_IP:-$WORKER_IP}"
CONTAINER_NAME="${CONTAINER_NAME:-mimo-nvfp4}"
REPO_DIR="${REPO_DIR:-/mnt/models/MiMo}"

###############################################################################
# Logging
###############################################################################
log()  { printf '\033[1;36m[stop]\033[0m %s\n' "$*"; }
hlog() { printf '\033[1;32m[head]\033[0m %s\n' "$*"; }
wlog() { printf '\033[1;35m[worker]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }

sshw() { ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" "$*" 2>/dev/null || true; }

###############################################################################
# Steps
###############################################################################

stop_processes_head() {
  hlog "stopping vLLM + Ray inside $CONTAINER_NAME"
  if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker exec "$CONTAINER_NAME" bash -lc '
      # Kill vLLM first
      for pid in $(pgrep -f "vllm serve" 2>/dev/null); do kill -TERM "$pid" 2>/dev/null || true; done
      for pid in $(pgrep -f launch-8888 2>/dev/null); do kill -TERM "$pid" 2>/dev/null || true; done
      # Stop Ray (quietly)
      ray stop --force >/dev/null 2>&1 || true
    ' 2>/dev/null || true
    hlog "done"
  else
    hlog "container not found — skipping"
  fi
}

stop_processes_worker() {
  wlog "stopping vLLM + Ray inside $CONTAINER_NAME on worker"
  local result
  result=$(sshw "
    if docker inspect '$CONTAINER_NAME' >/dev/null 2>&1; then
      docker exec '$CONTAINER_NAME' bash -lc '
        for pid in \$(pgrep -f \"vllm serve\" 2>/dev/null); do kill -TERM \"\$pid\" 2>/dev/null || true; done
        for pid in \$(pgrep -f launch-8888 2>/dev/null); do kill -TERM \"\$pid\" 2>/dev/null || true; done
        ray stop --force >/dev/null 2>&1 || true
      ' 2>/dev/null || true
      echo ok
    else
      echo missing
    fi
  ")
  if [ "$result" = "ok" ]; then
    wlog "done"
  elif [ "$result" = "missing" ]; then
    wlog "container not found on worker — skipping"
  fi
}

stop_container_head() {
  hlog "stopping container $CONTAINER_NAME on head"
  if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    hlog "container stopped"
  else
    hlog "container not found — skipping"
  fi
}

stop_container_worker() {
  wlog "stopping container $CONTAINER_NAME on worker"
  local result
  result=$(sshw "
    if docker inspect '$CONTAINER_NAME' >/dev/null 2>&1; then
      docker stop '$CONTAINER_NAME' >/dev/null 2>&1 || true
      echo ok
    else
      echo missing
    fi
  ")
  if [ "$result" = "ok" ]; then
    wlog "container stopped"
  elif [ "$result" = "missing" ]; then
    wlog "container not found on worker — skipping"
  fi
}

###############################################################################
# Main
###############################################################################

main() {
  case "${1:-}" in
    --check)
      log "CHECK MODE — would execute:"
      echo "  1) Kill vLLM + stop Ray in $CONTAINER_NAME on head"
      echo "  2) Kill vLLM + stop Ray in $CONTAINER_NAME on worker"
      echo "  3) docker stop $CONTAINER_NAME on head"
      echo "  4) docker stop $CONTAINER_NAME on worker"
      exit 0
      ;;
    --help|-h)
      sed -n '2,14p' "$0"
      exit 0
      ;;
  esac

  log "stopping cluster: head=$HEAD_IP worker=$WORKER_IP container=$CONTAINER_NAME"

  stop_processes_head
  stop_processes_worker
  stop_container_head
  stop_container_worker

  log "done"
}

main "$@"
