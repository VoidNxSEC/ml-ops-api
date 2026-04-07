#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/export.sh
# Extract and export inference results in multiple formats
#
# Usage:
#   ./scripts/export.sh --input result.json --format text
#   ./scripts/export.sh --input result.json --format markdown
#   ./scripts/export.sh --input result.json --format jsonl
#   ./scripts/export.sh --dir results/ --format text --out-dir exports/
# =============================================================================
set -euo pipefail

FORMAT="${FORMAT:-text}"
INPUT_FILE=""
INPUT_DIR=""
OUTPUT_DIR=""
OUTPUT_FILE=""

RED='\033[1;31m'; GRN='\033[1;32m'; BLU='\033[1;34m'; RST='\033[0m'
log()  { echo -e "${BLU}[export]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input|-i)    INPUT_FILE="$2";   shift 2 ;;
    --dir|-d)      INPUT_DIR="$2";    shift 2 ;;
    --output|-o)   OUTPUT_FILE="$2";  shift 2 ;;
    --out-dir)     OUTPUT_DIR="$2";   shift 2 ;;
    --format|-f)   FORMAT="$2";       shift 2 ;;
    *) shift ;;
  esac
done

extract_content() {
  local file="$1"
  jq -r '.choices[0].message.content // empty' "$file" 2>/dev/null
}

extract_meta() {
  local file="$1"
  jq '{
    model: (.model // "unknown"),
    prompt_tokens: (.usage.prompt_tokens // 0),
    completion_tokens: (.usage.completion_tokens // 0),
    total_tokens: (.usage.total_tokens // 0),
    latency_ms: (.timings.predicted_ms // null)
  }' "$file" 2>/dev/null || echo "{}"
}

convert_one() {
  local input="$1"
  local output="$2"
  local name
  name=$(basename "$input" .json)

  case "$FORMAT" in
    text)
      extract_content "$input" > "$output"
      ;;
    markdown)
      {
        echo "# $name"
        echo ""
        echo "_Generated: $(date -Iseconds)_"
        echo ""
        extract_content "$input"
        echo ""
        echo "---"
        echo ""
        echo "**Metadata:**"
        extract_meta "$input" | jq -r 'to_entries[] | "- \(.key): \(.value)"'
      } > "$output"
      ;;
    jsonl)
      jq -c '{
        id: input_filename,
        content: .choices[0].message.content,
        usage: .usage,
        model: .model
      }' "$input" >> "$output"
      ;;
    csv)
      {
        CONTENT=$(extract_content "$input" | tr '\n' ' ' | tr ',' ';')
        TOKENS=$(jq -r '.usage.total_tokens // 0' "$input")
        echo "\"$name\",\"$CONTENT\",$TOKENS"
      } >> "$output"
      ;;
    *)
      die "Unknown format: $FORMAT. Options: text, markdown, jsonl, csv"
      ;;
  esac
}

if [[ -n "$INPUT_DIR" ]]; then
  [[ -d "$INPUT_DIR" ]] || die "Directory not found: $INPUT_DIR"
  [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="${INPUT_DIR}/exports_${FORMAT}"
  mkdir -p "$OUTPUT_DIR"

  EXT="txt"
  case "$FORMAT" in
    markdown) EXT="md"  ;;
    jsonl)    EXT="jsonl" ;;
    csv)      EXT="csv"  ;;
  esac

  # For JSONL/CSV, use single output file
  if [[ "$FORMAT" == "jsonl" || "$FORMAT" == "csv" ]]; then
    OUT="$OUTPUT_DIR/all.${EXT}"
    [[ "$FORMAT" == "csv" ]] && echo '"name","content","tokens"' > "$OUT"
    for f in "$INPUT_DIR"/*.json; do
      [[ -f "$f" ]] || continue
      convert_one "$f" "$OUT"
      log "Exported: $(basename "$f")"
    done
    ok "Combined: $OUT"
  else
    for f in "$INPUT_DIR"/*.json; do
      [[ -f "$f" ]] || continue
      name=$(basename "$f" .json)
      OUT="$OUTPUT_DIR/${name}.${EXT}"
      convert_one "$f" "$OUT"
      log "Exported: $name → $OUT"
    done
    ok "Done → $OUTPUT_DIR"
  fi

elif [[ -n "$INPUT_FILE" ]]; then
  [[ -f "$INPUT_FILE" ]] || die "File not found: $INPUT_FILE"

  if [[ -z "$OUTPUT_FILE" ]]; then
    EXT="txt"
    case "$FORMAT" in
      markdown) EXT="md"  ;;
      jsonl)    EXT="jsonl" ;;
      csv)      EXT="csv"  ;;
    esac
    OUTPUT_FILE="${INPUT_FILE%.json}.${EXT}"
  fi

  convert_one "$INPUT_FILE" "$OUTPUT_FILE"
  ok "Exported → $OUTPUT_FILE"

else
  # Read from stdin
  TMP=$(mktemp /tmp/tf-export-XXXXXX.json)
  cat > "$TMP"
  OUT="${OUTPUT_FILE:-/dev/stdout}"
  convert_one "$TMP" "$OUT"
  rm -f "$TMP"
fi
