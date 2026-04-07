#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/infer.sh
# Single-shot inference against running llama-server
#
# Usage:
#   ./scripts/infer.sh "what is 2+2?"
#   ./scripts/infer.sh --system "You are a code reviewer" --prompt "review this: ..."
#   ./scripts/infer.sh --file prompt.txt --output result.json
#   ./scripts/infer.sh --stream "explain this step by step..."
#   LLAMACPP_URL=http://remote:8080 ./scripts/infer.sh "hello"
# =============================================================================
set -euo pipefail

ENV_FILE="${TF_PREFIX:-/opt/tensorforge/llamacpp}/env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

URL="${LLAMACPP_URL:-http://127.0.0.1:8080}"
SYSTEM_PROMPT=""
PROMPT=""
PROMPT_FILE=""
OUTPUT_FILE=""
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
STREAM=false
FORMAT=""          # json_object | null
RAW=false          # print raw JSON response

RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
die() { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }
ok()  { echo -e "${GRN}[✓]${RST} $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url|-u)       URL="$2";           shift 2 ;;
    --system|-s)    SYSTEM_PROMPT="$2"; shift 2 ;;
    --prompt|-p)    PROMPT="$2";        shift 2 ;;
    --file|-f)      PROMPT_FILE="$2";   shift 2 ;;
    --output|-o)    OUTPUT_FILE="$2";   shift 2 ;;
    --temp)         TEMPERATURE="$2";   shift 2 ;;
    --max-tokens)   MAX_TOKENS="$2";    shift 2 ;;
    --stream)       STREAM=true;        shift   ;;
    --json)         FORMAT="json_object"; shift  ;;
    --raw)          RAW=true;           shift   ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS] [PROMPT]"
      echo ""
      echo "  --url URL          Server URL (default: $URL)"
      echo "  --system TEXT      System prompt"
      echo "  --prompt TEXT      User prompt"
      echo "  --file FILE        Read prompt from file"
      echo "  --output FILE      Save raw JSON response to file"
      echo "  --temp FLOAT       Temperature (default: 0.1)"
      echo "  --max-tokens N     Max completion tokens (default: 4096)"
      echo "  --stream           Stream output (SSE)"
      echo "  --json             Request JSON object response format"
      echo "  --raw              Print raw JSON (don't extract content)"
      exit 0 ;;
    *) PROMPT="$1"; shift ;;
  esac
done

# Read from file
[[ -n "$PROMPT_FILE" ]] && { [[ -f "$PROMPT_FILE" ]] || die "Prompt file not found: $PROMPT_FILE"; PROMPT="$(cat "$PROMPT_FILE")"; }
[[ -n "$PROMPT" ]] || die "No prompt. Use --prompt, --file, or pass as arg."

# Health check
curl -sf "$URL/health" &>/dev/null || die "Server not responding at $URL\n  Run: ./scripts/server.sh start"

# Build messages JSON
MESSAGES=$(jq -n \
  --arg system "$SYSTEM_PROMPT" \
  --arg user "$PROMPT" \
  '
  if $system != "" then
    [{role:"system", content:$system}, {role:"user", content:$user}]
  else
    [{role:"user", content:$user}]
  end
  ')

# Build request payload
PAYLOAD=$(jq -n \
  --argjson messages "$MESSAGES" \
  --argjson temp "$TEMPERATURE" \
  --argjson max "$MAX_TOKENS" \
  --argjson stream "$STREAM" \
  --arg fmt "$FORMAT" \
  '
  {
    model: "local",
    messages: $messages,
    temperature: $temp,
    max_tokens: $max,
    stream: $stream
  }
  + if $fmt != "" then {response_format: {type: $fmt}} else {} end
  ')

# ── Stream mode ───────────────────────────────────────────────────────────────
if [[ "$STREAM" == "true" ]]; then
  curl -sf -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    --no-buffer | while IFS= read -r line; do
      [[ "$line" == data:* ]] || continue
      data="${line#data: }"
      [[ "$data" == "[DONE]" ]] && break
      echo "$data" | jq -r '.choices[0].delta.content // empty' 2>/dev/null | printf '%s'
    done
  echo ""
  exit 0
fi

# ── Regular mode ──────────────────────────────────────────────────────────────
RESPONSE=$(curl -sf -X POST "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD") || die "Inference request failed"

# Save raw JSON if requested
if [[ -n "$OUTPUT_FILE" ]]; then
  echo "$RESPONSE" > "$OUTPUT_FILE"
  ok "Response saved → $OUTPUT_FILE" >&2
fi

if [[ "$RAW" == "true" ]]; then
  echo "$RESPONSE" | jq '.'
else
  echo "$RESPONSE" | jq -r '.choices[0].message.content // empty'
fi
