#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/health.sh
# Comprehensive health check: GPU, server, model, performance
#
# Usage:
#   ./scripts/health.sh
#   ./scripts/health.sh --json
#   ./scripts/health.sh --watch 5   # refresh every 5s
# =============================================================================
set -euo pipefail

ENV_FILE="${TF_PREFIX:-/opt/tensorforge/llamacpp}/env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

URL="${LLAMACPP_URL:-http://127.0.0.1:8080}"
JSON_MODE=false
WATCH_INTERVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)  JSON_MODE=true;       shift   ;;
    --watch) WATCH_INTERVAL="$2";  shift 2 ;;
    *) shift ;;
  esac
done

GRN='\033[1;32m'; YLW='\033[1;33m'; RED='\033[1;31m'; BLU='\033[1;34m'; RST='\033[0m'
BOLD='\033[1m'

run_check() {
  local ts
  ts=$(date -Iseconds)

  # ── GPU ───────────────────────────────────────────────────────────────────
  GPU_JSON=$(nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{
        printf "{\"index\":%s,\"name\":\"%s\",\"vram_total_mb\":%s,\"vram_used_mb\":%s,\"vram_free_mb\":%s,\"gpu_util\":%s,\"mem_util\":%s,\"temp_c\":%s,\"power_w\":%.0f,\"power_limit_w\":%.0f}\n",
        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10
      }' | jq -s '.' 2>/dev/null || echo "[]")

  # ── Server ────────────────────────────────────────────────────────────────
  SERVER_UP=false
  HEALTH_JSON="{}"
  PROPS_JSON="{}"
  METRICS_SAMPLE="{}"

  if curl -sf "$URL/health" &>/dev/null; then
    SERVER_UP=true
    HEALTH_JSON=$(curl -sf "$URL/health" 2>/dev/null || echo "{}")
    PROPS_JSON=$(curl -sf "$URL/props" 2>/dev/null || echo "{}")
  fi

  # ── Perf benchmark (quick) ────────────────────────────────────────────────
  BENCH_JSON="{}"
  if [[ "$SERVER_UP" == "true" ]]; then
    START_NS=$(date +%s%N)
    RESP=$(curl -sf -X POST "$URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"local","messages":[{"role":"user","content":"1+1="}],"max_tokens":5,"temperature":0}' \
      2>/dev/null || echo "{}")
    END_NS=$(date +%s%N)
    ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
    BENCH_JSON=$(echo "$RESP" | jq --argjson ms "$ELAPSED_MS" '
      {
        latency_ms: $ms,
        prompt_tokens: (.usage.prompt_tokens // 0),
        completion_tokens: (.usage.completion_tokens // 0),
        tokens_per_sec: ((.usage.completion_tokens // 0) * 1000 / ($ms | if . == 0 then 1 else . end) | floor)
      }' 2>/dev/null || echo "{\"latency_ms\":$ELAPSED_MS}")
  fi

  # ── Build JSON report ─────────────────────────────────────────────────────
  if [[ "$JSON_MODE" == "true" ]]; then
    jq -n \
      --arg ts "$ts" \
      --arg url "$URL" \
      --argjson server_up "$SERVER_UP" \
      --argjson health "$HEALTH_JSON" \
      --argjson props "$PROPS_JSON" \
      --argjson bench "$BENCH_JSON" \
      --argjson gpus "$GPU_JSON" \
      '{
        timestamp: $ts,
        endpoint: $url,
        server: {
          up: $server_up,
          status: ($health.status // "unknown"),
          slots_idle: ($health.slots_idle // 0),
          slots_processing: ($health.slots_processing // 0),
          model: ($props.model_alias // $props.model // "unknown"),
          n_ctx: ($props.n_ctx // 0),
          n_gpu_layers: ($props.n_gpu_layers // 0)
        },
        benchmark: $bench,
        gpus: $gpus
      }'
    return
  fi

  # ── Human output ──────────────────────────────────────────────────────────
  clear 2>/dev/null || true
  echo -e "${BOLD}tensorforge health — $ts${RST}"
  echo ""

  # Server
  if [[ "$SERVER_UP" == "true" ]]; then
    STATUS=$(echo "$HEALTH_JSON" | jq -r '.status // "?"')
    SLOTS_I=$(echo "$HEALTH_JSON" | jq -r '.slots_idle // "?"')
    SLOTS_P=$(echo "$HEALTH_JSON" | jq -r '.slots_processing // "?"')
    MODEL=$(echo "$PROPS_JSON" | jq -r '.model_alias // .model // "?"')
    N_CTX=$(echo "$PROPS_JSON" | jq -r '.n_ctx // "?"')
    N_GPU=$(echo "$PROPS_JSON" | jq -r '.n_gpu_layers // "?"')
    LAT=$(echo "$BENCH_JSON" | jq -r '.latency_ms // "?"')
    TPS=$(echo "$BENCH_JSON" | jq -r '.tokens_per_sec // "?"')

    echo -e "  ${GRN}● server${RST}  $URL  [$STATUS]"
    echo    "    model      : $MODEL"
    echo    "    ctx        : $N_CTX tokens  |  gpu layers: $N_GPU"
    echo    "    slots      : $SLOTS_I idle / $SLOTS_P busy"
    echo    "    latency    : ${LAT}ms  |  throughput: ~${TPS} tok/s"
  else
    echo -e "  ${RED}● server${RST}  $URL  [DOWN]"
  fi

  echo ""

  # GPUs
  echo -e "  ${BOLD}GPU(s):${RST}"
  echo "$GPU_JSON" | jq -r '.[] |
    "    [\(.index)] \(.name)\n" +
    "      VRAM   : \(.vram_used_mb)MB / \(.vram_total_mb)MB  (\(.vram_used_mb * 100 / (.vram_total_mb | if . == 0 then 1 else . end) | floor)% used)\n" +
    "      Util   : GPU \(.gpu_util)%  Mem \(.mem_util)%\n" +
    "      Temp   : \(.temp_c)°C\n" +
    "      Power  : \(.power_w)W / \(.power_limit_w)W"
  ' 2>/dev/null || echo "    (nvidia-smi unavailable)"

  echo ""
}

if [[ "$WATCH_INTERVAL" -gt 0 ]]; then
  while true; do
    run_check
    sleep "$WATCH_INTERVAL"
  done
else
  run_check
fi
