#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/server.sh
# Start / stop / restart / status / logs for llama-server
# B200-optimized defaults: -ngl 999, --flash-attn, --cuda-graphs
#
# Usage:
#   ./scripts/server.sh start [--model FILE] [--port 8080] [--ctx auto]
#   ./scripts/server.sh stop
#   ./scripts/server.sh restart [OPTIONS]
#   ./scripts/server.sh status
#   ./scripts/server.sh logs [--follow]
#   ./scripts/server.sh gpu
# =============================================================================
set -euo pipefail

# ── Source env if available ───────────────────────────────────────────────────
_DEFAULT_PREFIX="$([[ "$(id -u)" == "0" ]] && echo "/opt/tensorforge/llamacpp" || echo "$HOME/.tensorforge/llamacpp")"
ENV_FILE="${TF_PREFIX:-$_DEFAULT_PREFIX}/env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

# ── Defaults (can be overridden via env or args) ──────────────────────────────
PREFIX="${TF_PREFIX:-$_DEFAULT_PREFIX}"
LLAMA_BIN="${TF_BIN:-$PREFIX/bin/llama-server}"
MODELS_DIR="${TF_MODELS:-/var/lib/tensorforge/models/gguf}"
PID_FILE="${TF_PID:-$PREFIX/run/server.pid}"
LOG_FILE="${TF_LOG:-$PREFIX/logs/server.log}"
PORT="${TF_PORT:-8080}"
HOST="${TF_HOST:-127.0.0.1}"

# B200-optimized defaults
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
CTX_SIZE="${CTX_SIZE:-auto}"
N_PARALLEL="${N_PARALLEL:-8}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
FLASH_ATTN="${FLASH_ATTN:-true}"
CONT_BATCHING="${CONT_BATCHING:-true}"
# mlock/no-mmap: good on bare metal B200, but require CAP_IPC_LOCK in containers
# Set MLOCK=true / NO_MMAP=true manually on bare-metal NIM nodes
MLOCK="${MLOCK:-false}"
NO_MMAP="${NO_MMAP:-false}"
THREADS_BATCH="${THREADS_BATCH:-$(nproc)}"
MAIN_THREADS="${MAIN_THREADS:-4}"

RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
log()  { echo -e "${BLU}[server]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }

# ── Helpers ───────────────────────────────────────────────────────────────────
is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

wait_ready() {
  local max="${1:-120}" i=0
  log "Waiting for server (max ${max}s)..."
  while ! curl -sf "http://${HOST}:${PORT}/health" &>/dev/null; do
    sleep 1; i=$((i+1))
    [[ $i -ge $max ]] && die "Server not ready after ${max}s\n  Check: $LOG_FILE"
    [[ $((i % 15)) -eq 0 ]] && {
      # Show what the server is doing (model loading progress)
      LAST=$(tail -1 "$LOG_FILE" 2>/dev/null | tr -d '\r' || echo "")
      log "  (${i}s) $LAST"
    }
  done
}

auto_ctx() {
  # B200=192GB → 65536, H100 80GB → 32768, 40GB → 16384, 24GB → 8192
  local vram_gb
  vram_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null \
    | head -1 | awk '{print int($1/1024)}' || echo 0)
  if   [[ $vram_gb -ge 160 ]]; then echo 65536
  elif [[ $vram_gb -ge  80 ]]; then echo 32768
  elif [[ $vram_gb -ge  40 ]]; then echo 16384
  elif [[ $vram_gb -ge  24 ]]; then echo 8192
  else                               echo 4096
  fi
}

find_model() {
  find "$MODELS_DIR" -name "*.gguf" -type f 2>/dev/null | sort | head -1
}

# ── start ─────────────────────────────────────────────────────────────────────
cmd_start() {
  local model_file=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model|-m)     model_file="$2";       shift 2 ;;
      --port|-p)      PORT="$2";             shift 2 ;;
      --host)         HOST="$2";             shift 2 ;;
      --ctx)          CTX_SIZE="$2";         shift 2 ;;
      --parallel)     N_PARALLEL="$2";       shift 2 ;;
      --gpu-layers)   N_GPU_LAYERS="$2";     shift 2 ;;
      --no-flash)     FLASH_ATTN=false;      shift   ;;
      --no-mlock)     MLOCK=false;           shift   ;;
      *) die "Unknown flag: $1" ;;
    esac
  done

  is_running && die "Already running (PID $(cat "$PID_FILE")). Use restart."
  [[ -x "$LLAMA_BIN" ]] || die "Binary not found: $LLAMA_BIN\n  Run: ./scripts/bootstrap.sh"

  # Resolve model
  if [[ -z "$model_file" ]]; then
    model_file=$(find_model)
    [[ -n "$model_file" ]] || die "No .gguf model in $MODELS_DIR\n  Run: ./scripts/model-pull.sh"
    log "Auto model: $(basename "$model_file")"
  else
    [[ "$model_file" != /* ]] && model_file="$MODELS_DIR/$model_file"
  fi
  [[ -f "$model_file" ]] || die "Model not found: $model_file"

  # Auto context
  [[ "$CTX_SIZE" == "auto" ]] && CTX_SIZE=$(auto_ctx) && log "Auto ctx: $CTX_SIZE"

  # Build server args
  ARGS=(
    --model         "$model_file"
    --host          "$HOST"
    --port          "$PORT"
    --ctx-size      "$CTX_SIZE"
    -ngl            "$N_GPU_LAYERS"
    --parallel      "$N_PARALLEL"
    --batch-size    "$BATCH_SIZE"
    --ubatch-size   "$UBATCH_SIZE"
    --threads       "$MAIN_THREADS"
    --threads-batch "$THREADS_BATCH"
  )
  [[ "$FLASH_ATTN"    == "true" ]] && ARGS+=(--flash-attn)
  [[ "$CONT_BATCHING" == "true" ]] && ARGS+=(--cont-batching)
  [[ "$MLOCK"         == "true" ]] && ARGS+=(--mlock)
  [[ "$NO_MMAP"       == "true" ]] && ARGS+=(--no-mmap)

  log "Starting llama-server..."
  log "  model    : $(basename "$model_file")"
  log "  port     : $PORT"
  log "  ctx      : $CTX_SIZE tokens"
  log "  gpu-lyr  : $N_GPU_LAYERS"
  log "  parallel : $N_PARALLEL slots"
  log "  flash    : $FLASH_ATTN"

  mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
  nohup "$LLAMA_BIN" "${ARGS[@]}" >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  ok "PID: $(cat "$PID_FILE")"

  wait_ready 180
  ok "Server ready → http://${HOST}:${PORT}"
  echo ""
  echo "  /health  : http://${HOST}:${PORT}/health"
  echo "  /metrics : http://${HOST}:${PORT}/metrics"
  echo "  /slots   : http://${HOST}:${PORT}/slots"
  echo "  Logs     : $LOG_FILE"
}

# ── stop ──────────────────────────────────────────────────────────────────────
cmd_stop() {
  if ! is_running; then
    warn "Not running"
    rm -f "$PID_FILE"
    return 0
  fi
  local pid
  pid=$(cat "$PID_FILE")
  log "Stopping (PID $pid)..."
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 15); do
    kill -0 "$pid" 2>/dev/null || { rm -f "$PID_FILE"; ok "Stopped"; return 0; }
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  ok "Force-killed"
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
  echo ""
  if is_running; then
    local pid
    pid=$(cat "$PID_FILE")
    ok "RUNNING  (PID $pid)  http://${HOST}:${PORT}"
    echo ""

    HEALTH=$(curl -sf "http://${HOST}:${PORT}/health" 2>/dev/null || echo "{}")
    echo "  status  : $(echo "$HEALTH" | jq -r '.status // "unknown"' 2>/dev/null)"
    echo "  idle    : $(echo "$HEALTH" | jq -r '.slots_idle // "?"' 2>/dev/null) slots"
    echo "  busy    : $(echo "$HEALTH" | jq -r '.slots_processing // "?"' 2>/dev/null) slots"
    echo ""

    # Props
    PROPS=$(curl -sf "http://${HOST}:${PORT}/props" 2>/dev/null || echo "{}")
    echo "  model   : $(echo "$PROPS" | jq -r '.model_alias // .model // "?"' 2>/dev/null)"
    echo "  n_ctx   : $(echo "$PROPS" | jq -r '.n_ctx // "?"' 2>/dev/null)"
    echo "  n_gpu   : $(echo "$PROPS" | jq -r '.n_gpu_layers // "?"' 2>/dev/null)"
  else
    warn "NOT RUNNING"
    [[ -f "$LOG_FILE" ]] && echo "  Last log : $(tail -1 "$LOG_FILE" 2>/dev/null | tr -d '\r')"
  fi

  echo ""
  echo "  GPU:"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>/dev/null \
    | awk -F',' '{
        printf "  %-32s VRAM: %s/%s  util: %s  %s\n",
        $1, $2, $3, $4, $5
      }' || echo "  (nvidia-smi unavailable)"
  echo ""
}

# ── gpu ───────────────────────────────────────────────────────────────────────
cmd_gpu() {
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv 2>/dev/null || die "nvidia-smi not available"
}

# ── logs ──────────────────────────────────────────────────────────────────────
cmd_logs() {
  local n=80 follow=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n|--tail)    n="$2";      shift 2 ;;
      -f|--follow)  follow=true; shift   ;;
      *) shift ;;
    esac
  done
  [[ -f "$LOG_FILE" ]] || die "No log file: $LOG_FILE"
  if [[ "$follow" == "true" ]]; then
    tail -n "$n" -f "$LOG_FILE"
  else
    tail -n "$n" "$LOG_FILE"
  fi
}

# ── entry ─────────────────────────────────────────────────────────────────────
CMD="${1:-status}"; shift || true
case "$CMD" in
  start)   cmd_start "$@"   ;;
  stop)    cmd_stop         ;;
  restart) cmd_stop; sleep 2; cmd_start "$@" ;;
  status)  cmd_status       ;;
  logs)    cmd_logs "$@"    ;;
  gpu)     cmd_gpu          ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|gpu}"
    echo ""
    echo "  start   [--model FILE] [--port N] [--ctx auto|N] [--parallel N]"
    echo "  stop"
    echo "  restart [same as start opts]"
    echo "  status"
    echo "  logs    [-n N] [-f]"
    echo "  gpu"
    exit 1 ;;
esac
