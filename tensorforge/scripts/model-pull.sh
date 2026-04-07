#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/model-pull.sh
# Download GGUF models from HuggingFace
#
# Usage:
#   ./scripts/model-pull.sh --list
#   ./scripts/model-pull.sh --model qwen2.5-coder-32b
#   ./scripts/model-pull.sh --url https://... [--name mymodel.gguf]
#   HF_TOKEN=hf_... ./scripts/model-pull.sh --model llama-3.1-70b
# =============================================================================
set -euo pipefail

ENV_FILE="${TF_PREFIX:-/opt/tensorforge/llamacpp}/env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

MODELS_DIR="${TF_MODELS:-/var/lib/tensorforge/models/gguf}"
HF_TOKEN="${HF_TOKEN:-}"
RESUME=true

RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
log()  { echo -e "${BLU}[model-pull]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }

# ── Registry ──────────────────────────────────────────────────────────────────
# Format: url|file|vram_gb|description
declare -A REGISTRY
REGISTRY["llama-3.1-8b"]="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf|llama-3.1-8b-instruct.Q8_0.gguf|9|Llama 3.1 8B Q8 — fast general"
REGISTRY["llama-3.1-70b"]="https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q8_0.gguf|llama-3.1-70b-instruct.Q8_0.gguf|75|Llama 3.1 70B Q8 — B200 recommended"
REGISTRY["llama-3.3-70b"]="https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q8_0.gguf|llama-3.3-70b-instruct.Q8_0.gguf|75|Llama 3.3 70B Q8 — best instruct following"
REGISTRY["qwen2.5-coder-7b"]="https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf|qwen2.5-coder-7b-instruct.Q8_0.gguf|8|Qwen 2.5 Coder 7B — fast code analysis"
REGISTRY["qwen2.5-coder-32b"]="https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf|qwen2.5-coder-32b-instruct.Q8_0.gguf|35|Qwen 2.5 Coder 32B — arch-analyzer default"
REGISTRY["deepseek-r1-14b"]="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf|deepseek-r1-distill-qwen-14b.Q8_0.gguf|16|DeepSeek R1 14B — reasoning, fits 24GB"
REGISTRY["deepseek-r1-70b"]="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-70B-Q8_0.gguf|deepseek-r1-distill-llama-70b.Q8_0.gguf|75|DeepSeek R1 70B — strong reasoning"
REGISTRY["mistral-24b"]="https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF/resolve/main/Mistral-Small-3.1-24B-Instruct-2503-Q8_0.gguf|mistral-small-24b-instruct.Q8_0.gguf|26|Mistral Small 24B — balanced"

cmd_list() {
  echo ""
  printf "${BLU}%-25s %-8s %s${RST}\n" "NAME" "VRAM" "DESCRIPTION"
  printf "%-25s %-8s %s\n" "─────────────────────────" "────────" "────────────────────────────────────────"
  for key in $(echo "${!REGISTRY[@]}" | tr ' ' '\n' | sort); do
    IFS='|' read -r url file vram desc <<< "${REGISTRY[$key]}"
    printf "%-25s %-8s %s\n" "$key" "${vram}GB" "$desc"
  done
  echo ""
  echo "  Downloaded models in $MODELS_DIR:"
  find "$MODELS_DIR" -name "*.gguf" -type f 2>/dev/null | sort | while read -r f; do
    SIZE=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "    ✓ $(basename "$f")  ($SIZE)"
  done || echo "    (none)"
  echo ""
}

cmd_pull() {
  local key_or_url="$1"
  local custom_name="${2:-}"
  local url file

  if [[ "$key_or_url" =~ ^https?:// ]]; then
    url="$key_or_url"
    file="${custom_name:-$(basename "$url" | sed 's/?.*//')}"
  else
    [[ -n "${REGISTRY[$key_or_url]+x}" ]] || die "Unknown model: $key_or_url\nRun --list to see available models."
    IFS='|' read -r url file vram desc <<< "${REGISTRY[$key_or_url]}"
    log "Model : $key_or_url ($desc, ${vram}GB VRAM)"
  fi

  mkdir -p "$MODELS_DIR"
  local dest="$MODELS_DIR/$file"

  if [[ -f "$dest" ]]; then
    SIZE=$(du -sh "$dest" | cut -f1)
    ok "Already exists: $file ($SIZE)"
    return 0
  fi

  log "Downloading → $dest"
  log "From: $url"

  CURL_ARGS=(--location --progress-bar --output "$dest.tmp")
  [[ "$RESUME" == "true" ]]  && CURL_ARGS+=(--continue-at -)
  [[ -n "$HF_TOKEN" ]]       && CURL_ARGS+=(--header "Authorization: Bearer $HF_TOKEN")

  curl "${CURL_ARGS[@]}" "$url" || { rm -f "$dest.tmp"; die "Download failed"; }
  mv "$dest.tmp" "$dest"

  SIZE=$(du -sh "$dest" | cut -f1)
  ok "Downloaded: $file ($SIZE)"
}

# ── Args ──────────────────────────────────────────────────────────────────────
ACTION="pull"
MODEL_KEY=""
CUSTOM_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--list)     ACTION=list;           shift   ;;
    -m|--model)    MODEL_KEY="$2";        shift 2 ;;
    --url)         MODEL_KEY="$2";        shift 2 ;;
    --name)        CUSTOM_NAME="$2";      shift 2 ;;
    --no-resume)   RESUME=false;          shift   ;;
    --hf-token)    HF_TOKEN="$2";         shift 2 ;;
    --models-dir)  MODELS_DIR="$2";       shift 2 ;;
    *)             MODEL_KEY="$1";        shift   ;;
  esac
done

case "$ACTION" in
  list) cmd_list ;;
  pull)
    [[ -n "$MODEL_KEY" ]] || {
      echo "Usage: $0 --model <name>  |  $0 --url <url>"
      echo "       $0 --list"
      exit 1
    }
    cmd_pull "$MODEL_KEY" "$CUSTOM_NAME"
    ;;
esac
