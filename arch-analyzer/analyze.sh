#!/usr/bin/env bash
# =============================================================================
# ml-ops-api/arch-analyzer/analyze.sh
# Entry point that wires arch-analyzer to the tensorforge llama.cpp backend
#
# Usage:
#   ./arch-analyzer/analyze.sh platform ~/master/
#   ./arch-analyzer/analyze.sh analyze  ~/master/neoland --template rust
#   ./arch-analyzer/analyze.sh platform ~/master/ --model qwen2.5-coder-32b
#
# If the tensorforge server isn't running, it starts it automatically,
# runs the analysis, and stops it (unless --keep-server).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TF_SCRIPTS="$ROOT_DIR/tensorforge/scripts"

# Source tensorforge env if available
ENV_FILE="${TF_PREFIX:-/opt/tensorforge/llamacpp}/env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

export LLAMACPP_URL="${LLAMACPP_URL:-http://127.0.0.1:8080}"
export LLM_PARALLEL="${LLM_PARALLEL:-8}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-180}"

RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
log()  { echo -e "${BLU}[arch-analyzer]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }

KEEP_SERVER=false
SERVER_STARTED=false

# Parse --keep-server before passing remaining args to analyzer
PASSTHROUGH_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --keep-server) KEEP_SERVER=true ;;
    *)             PASSTHROUGH_ARGS+=("$arg") ;;
  esac
done

cleanup() {
  if [[ "$SERVER_STARTED" == "true" && "$KEEP_SERVER" == "false" ]]; then
    log "Stopping tensorforge server..."
    bash "$TF_SCRIPTS/server.sh" stop 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Auto-start tensorforge server if not running
if ! curl -sf "$LLAMACPP_URL/health" &>/dev/null; then
  if [[ -f "$TF_SCRIPTS/server.sh" ]]; then
    log "Starting tensorforge llama-server..."
    bash "$TF_SCRIPTS/server.sh" start
    SERVER_STARTED=true
  else
    die "llama-server not running and tensorforge scripts not found.\n  Run: ml-ops-api/tensorforge/scripts/bootstrap.sh"
  fi
else
  ok "Using running server: $LLAMACPP_URL"
fi

# Find Python env
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in \
    "$SCRIPT_DIR/.venv/bin/python" \
    "$ROOT_DIR/.venv/bin/python" \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"
  do
    [[ -x "$candidate" ]] && { PYTHON="$candidate"; break; }
  done
fi
[[ -n "$PYTHON" ]] || die "Python not found. Set PYTHON= env var."

# Bootstrap venv if needed
VENV="$SCRIPT_DIR/.venv"
if [[ ! -d "$VENV" ]]; then
  log "Creating Python venv..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q aiohttp aiofiles pydantic rich typer httpx pyyaml psutil
  ok "venv ready: $VENV"
  PYTHON="$VENV/bin/python"
fi

log "Running analyzer → $LLAMACPP_URL"
exec "$PYTHON" "$SCRIPT_DIR/src/analyzer.py" "${PASSTHROUGH_ARGS[@]}"
