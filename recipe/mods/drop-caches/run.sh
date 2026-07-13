#!/bin/bash
set -euo pipefail
LOG=/tmp/drop_caches.log
pkill -x codex_drop_cach 2>/dev/null || true
cat >/tmp/codex_drop_caches_loop.sh <<'EOF'
#!/bin/bash
while true; do
  sync
  (echo 3 >/proc/sys/vm/drop_caches) 2>/dev/null || true
  sleep "${MIMO_DROP_CACHES_INTERVAL:-5}"
done
EOF
chmod +x /tmp/codex_drop_caches_loop.sh
nohup /tmp/codex_drop_caches_loop.sh >>"$LOG" 2>&1 &
echo "[drop-caches] started loop pid=$! log=$LOG"
