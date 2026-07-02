#!/bin/bash
set -euo pipefail

ROUTER_HOST="10.244.194.248"
ENGINE_PORTS=(15000 15002 15004 15006)

CHECK_INTERVAL=5
LOG_DIR="${LOG_DIR:-/tmp/router_watch}"
mkdir -p "$LOG_DIR"

ROUTER_METRICS_URL="http://${ROUTER_HOST}:8000/metrics"

# -----------------------------
# health check (engine)
# -----------------------------
check_engine_health() {
  local port=$1
  curl -sf "http://${ROUTER_HOST}:${port}/health" >/dev/null 2>&1
}

# -----------------------------
# router metrics check
# -----------------------------
check_router_registered() {
  local port=$1

  curl -s "$ROUTER_METRICS_URL" | grep -q "engine_${port}"
}

# -----------------------------
# restart engine (placeholder)
# -----------------------------
restart_engine() {
  local port=$1

  echo "[router-watch] restarting engine on port ${port}"

  # ⚠️ 这里替换成你真实启动命令
  # 例如：kubectl / ray / ssh / local launch

  ssh root@"${ROUTER_HOST}" \
    "bash -lc '
      pkill -f \"port ${port}\" || true
      sleep 1
      nohup python3 -m sglang.launch_server \
        --host 0.0.0.0 \
        --port ${port} \
        > /tmp/engine_${port}.log 2>&1 &
    '" &
}

# -----------------------------
# monitor loop
# -----------------------------
monitor() {
  while true; do

    echo "[router-watch] checking engines..."

    for port in "${ENGINE_PORTS[@]}"; do

      local ok_http=0
      local ok_metric=0

      if check_engine_health "$port"; then
        ok_http=1
      fi

      if check_router_registered "$port"; then
        ok_metric=1
      fi

      if [[ "$ok_http" -eq 1 && "$ok_metric" -eq 1 ]]; then
        echo "[OK] engine ${port} healthy + registered"
        continue
      fi

      echo "[WARN] engine ${port} failed:"
      echo "       http_health=${ok_http} router_metric=${ok_metric}"

      restart_engine "$port"

    done

    sleep "$CHECK_INTERVAL"
  done
}

# -----------------------------
# graceful shutdown
# -----------------------------
cleanup() {
  echo "[router-watch] stopping..."
  exit 0
}

trap cleanup SIGINT SIGTERM

echo "[router-watch] monitoring 4 engines on ${ROUTER_HOST}"
monitor