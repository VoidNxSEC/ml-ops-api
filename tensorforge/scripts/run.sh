#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/run.sh
# Full pipeline: start → infer → export → stop
# "entrar, inferir, extrair, exportar, fechar"
#
# Usage:
#   ./scripts/run.sh --prompt "analyze this code..."
#   ./scripts/run.sh --file prompt.txt --output result.json --model qwen2.5-coder-32b
#   ./scripts/run.sh --batch prompts/ --output-dir results/
#   ./scripts/run.sh --keep-alive  # don't stop server after inference
# =============================================================================
set -euo pipefail

ENV_FILE="${TF_PREFIX:-/opt/tensorforge/llamacpp}/env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${TF_PORT:-8080}"
HOST="${TF_HOST:-127.0.0.1}"

# Args
PROMPT=""
PROMPT_FILE=""
BATCH_DIR=""
OUTPUT_FILE="${OUTPUT:-output.json}"
OUTPUT_DIR=""
MODEL_FILE=""
SYSTEM_PROMPT=""
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.1}"
KEEP_ALIVE=false
CTX="auto"

RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
log()  { echo -e "${BLU}[run]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }
hr()   { echo -e "${BLU}────────────────────────────────────────────${RST}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt|-p)      PROMPT="$2";       shift 2 ;;
    --file|-f)        PROMPT_FILE="$2";  shift 2 ;;
    --batch)          BATCH_DIR="$2";    shift 2 ;;
    --output|-o)      OUTPUT_FILE="$2";  shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2";   shift 2 ;;
    --model|-m)       MODEL_FILE="$2";   shift 2 ;;
    --system|-s)      SYSTEM_PROMPT="$2"; shift 2 ;;
    --max-tokens)     MAX_TOKENS="$2";   shift 2 ;;
    --temp)           TEMPERATURE="$2";  shift 2 ;;
    --ctx)            CTX="$2";          shift 2 ;;
    --port)           PORT="$2";         shift 2 ;;
    --keep-alive)     KEEP_ALIVE=true;   shift   ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "  --prompt TEXT       Inline prompt"
      echo "  --file FILE         Read prompt from file"
      echo "  --batch DIR         Process all .txt files in DIR"
      echo "  --output FILE       Output JSON file (single inference)"
      echo "  --output-dir DIR    Output directory (batch mode)"
      echo "  --model FILE        GGUF model file or name"
      echo "  --system TEXT       System prompt"
      echo "  --max-tokens N      Max tokens (default: $MAX_TOKENS)"
      echo "  --temp FLOAT        Temperature (default: $TEMPERATURE)"
      echo "  --ctx N|auto        Context size (default: auto)"
      echo "  --port N            Server port (default: $PORT)"
      echo "  --keep-alive        Don't stop server after inference"
      exit 0 ;;
    *) PROMPT="$1"; shift ;;
  esac
done

SERVER_STARTED=false

# ── Cleanup trap ──────────────────────────────────────────────────────────────
cleanup() {
  if [[ "$SERVER_STARTED" == "true" && "$KEEP_ALIVE" == "false" ]]; then
    log "Stopping server..."
    "$SCRIPT_DIR/server.sh" stop 2>/dev/null || true
    ok "Server stopped"
  elif [[ "$KEEP_ALIVE" == "true" ]]; then
    warn "Server left running on port $PORT (--keep-alive)"
  fi
}
trap cleanup EXIT

# ── 1. Start server if not running ───────────────────────────────────────────
if ! curl -sf "http://${HOST}:${PORT}/health" &>/dev/null; then
  hr
  log "Starting llama-server..."
  START_ARGS=("--port" "$PORT" "--ctx" "$CTX")
  [[ -n "$MODEL_FILE" ]] && START_ARGS+=("--model" "$MODEL_FILE")
  "$SCRIPT_DIR/server.sh" start "${START_ARGS[@]}"
  SERVER_STARTED=true
else
  log "Server already running on port $PORT"
fi

export LLAMACPP_URL="http://${HOST}:${PORT}"

# ── 2. Run inference ──────────────────────────────────────────────────────────

infer_one() {
  local prompt="$1"
  local output="$2"

  INFER_ARGS=(
    --prompt    "$prompt"
    --max-tokens "$MAX_TOKENS"
    --temp      "$TEMPERATURE"
    --output    "$output"
    --raw
  )
  [[ -n "$SYSTEM_PROMPT" ]] && INFER_ARGS+=(--system "$SYSTEM_PROMPT")

  "$SCRIPT_DIR/infer.sh" "${INFER_ARGS[@]}" > /dev/null
}

if [[ -n "$BATCH_DIR" ]]; then
  # ── Batch mode ─────────────────────────────────────────────────────────────
  [[ -d "$BATCH_DIR" ]] || die "Batch dir not found: $BATCH_DIR"
  [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="${BATCH_DIR}/results"
  mkdir -p "$OUTPUT_DIR"

  FILES=("$BATCH_DIR"/*.txt)
  TOTAL=${#FILES[@]}
  log "Batch mode: $TOTAL prompts → $OUTPUT_DIR"
  hr

  DONE=0 FAILED=0
  for f in "${FILES[@]}"; do
    [[ -f "$f" ]] || continue
    NAME=$(basename "$f" .txt)
    OUT="$OUTPUT_DIR/${NAME}.json"

    if [[ -f "$OUT" ]]; then
      warn "Skipping (exists): $NAME"
      continue
    fi

    log "[$((DONE+1))/$TOTAL] $NAME..."
    PROMPT_TEXT=$(cat "$f")

    if infer_one "$PROMPT_TEXT" "$OUT"; then
      CONTENT=$(jq -r '.choices[0].message.content' "$OUT" 2>/dev/null | head -c 120)
      ok "$NAME → $CONTENT..."
      DONE=$((DONE+1))
    else
      warn "FAILED: $NAME"
      FAILED=$((FAILED+1))
    fi
  done

  hr
  ok "Batch complete: $DONE succeeded, $FAILED failed"
  ok "Results: $OUTPUT_DIR"

else
  # ── Single inference ───────────────────────────────────────────────────────
  [[ -n "$PROMPT_FILE" ]] && { [[ -f "$PROMPT_FILE" ]] || die "File not found: $PROMPT_FILE"; PROMPT=$(cat "$PROMPT_FILE"); }
  [[ -n "$PROMPT" ]] || die "No prompt. Use --prompt, --file, or --batch."

  hr
  log "Running inference..."

  infer_one "$PROMPT" "$OUTPUT_FILE"

  # Extract and display
  CONTENT=$(jq -r '.choices[0].message.content // empty' "$OUTPUT_FILE" 2>/dev/null)
  TOKENS_USED=$(jq -r '.usage.total_tokens // "?"' "$OUTPUT_FILE" 2>/dev/null)
  INFER_MS=$(jq -r '.timings.predicted_ms // "?"' "$OUTPUT_FILE" 2>/dev/null)

  hr
  echo ""
  echo "$CONTENT"
  echo ""
  hr
  ok "Output  : $OUTPUT_FILE"
  ok "Tokens  : $TOKENS_USED"
  [[ "$INFER_MS" != "?" ]] && ok "Time    : ${INFER_MS}ms"
fi

# cleanup runs via trap
