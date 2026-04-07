#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/entrypoint.sh
# Single entrypoint for the full inference pipeline
#
# Orchestrates: install → bootstrap → pull → start → infer → export → stop
#
# COMMANDS:
#   setup     Install system deps + build llama.cpp (once)
#   pull      Download a model
#   run       Full pipeline: start → infer → export → stop
#   analyze   Run arch-analyzer on a project or the full platform
#   server    Manage the server (start/stop/restart/status/logs/gpu)
#   health    GPU + server health check
#   status    Quick status overview
#
# USAGE (NIM / bare Linux):
#   ./entrypoint.sh setup --model qwen2.5-coder-32b
#   ./entrypoint.sh run --prompt "analyze this code" --output result.json
#   ./entrypoint.sh analyze --platform
#   ./entrypoint.sh analyze --project ~/master/neoland --template rust
#   ./entrypoint.sh server start
#   ./entrypoint.sh health --watch 5
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_OPS_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"   # ~/master/ml-ops-api

RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
BOLD='\033[1m'
log()  { echo -e "${BLU}[tensorforge]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }
hr()   { echo -e "${BLU}────────────────────────────────────────────${RST}"; }

# ── Script references ──────────────────────────────────────────────────────────
S_DEPS="$SCRIPT_DIR/install-deps.sh"
S_SHELL="$SCRIPT_DIR/setup-shell.sh"
S_NIX="$SCRIPT_DIR/bootstrap-nix.sh"
S_BOOT="$SCRIPT_DIR/bootstrap.sh"
S_PULL="$SCRIPT_DIR/model-pull.sh"
S_SERV="$SCRIPT_DIR/server.sh"
S_INFER="$SCRIPT_DIR/infer.sh"
S_RUN="$SCRIPT_DIR/run.sh"
S_EXPORT="$SCRIPT_DIR/export.sh"
S_HEALTH="$SCRIPT_DIR/health.sh"
S_ANALYZE="$ML_OPS_ROOT/arch-analyzer/analyze.sh"

require_script() {
  [[ -x "$1" ]] || die "Script not found or not executable: $1"
}

banner() {
  echo ""
  echo -e "  ${BOLD}tensorforge — inference pipeline${RST}"
  echo    "  $(date '+%Y-%m-%d %H:%M:%S')  |  node: $(hostname)"
  echo ""
}

usage() {
  cat <<EOF
${BOLD}Usage:${RST}
  $(basename "$0") <command> [options]

${BOLD}Commands:${RST}

  ${GRN}setup${RST}   [--model NAME] [--skip-deps] [--force-rebuild]
          Install system packages, build llama.cpp from source, optionally
          download a starter model. Run this once per machine/container.

  ${GRN}pull${RST}    --model NAME  |  --list
          Download a GGUF model from the registry.
          Names: llama-3.1-8b, llama-3.3-70b, qwen2.5-coder-7b,
                 qwen2.5-coder-32b, deepseek-r1-14b, deepseek-r1-70b,
                 mistral-24b

  ${GRN}run${RST}     --prompt TEXT  |  --file FILE  |  --batch DIR
          Full pipeline: auto-start server → infer → display → auto-stop.
          Options: --output FILE, --model FILE, --max-tokens N, --keep-alive,
                   --system TEXT, --temp FLOAT, --ctx N|auto, --export FORMAT

  ${GRN}analyze${RST} --platform  |  --project PATH [--template TMPL]
          Run arch-analyzer against one or all projects via tensorforge.
          Templates: rust, python, typescript, nix, auto
          Example: $(basename "$0") analyze --project ~/master/neoland --template rust

  ${GRN}server${RST}  start|stop|restart|status|logs|gpu [options]
          Manage the llama-server directly.
          Example: $(basename "$0") server start --ctx 32768 --parallel 4

  ${GRN}health${RST}  [--json] [--watch N]
          GPU stats, server health, quick latency benchmark.

  ${GRN}shell${RST}   [--no-starship] [--dry-run]
          Install zsh + starship + plugins + GPU/tensorforge aliases.
          Run once per machine to get a productive terminal.

  ${GRN}status${RST}
          Quick overview: GPU, server, available models.

${BOLD}Quick start (NIM / bare Linux):${RST}
  $(basename "$0") setup --model qwen2.5-coder-32b
  $(basename "$0") run --prompt "Analyze this Rust codebase for security issues"
  $(basename "$0") analyze --platform

${BOLD}Batch inference (fleet pattern):${RST}
  mkdir prompts/ && ls prompts/*.txt
  $(basename "$0") run --batch prompts/ --output-dir results/ --keep-alive
  $(basename "$0") server stop

EOF
}

# ── setup ─────────────────────────────────────────────────────────────────────
cmd_setup() {
  local model="" boot_args=() deps_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model|-m)    model="$2";                  shift 2 ;;
      --skip-deps)   boot_args+=(--skip-deps);
                     deps_args+=(--dry-run);      shift   ;;
      --force-rebuild) boot_args+=(--force-rebuild); shift ;;
      --no-python)   deps_args+=(--no-python);    shift   ;;
      *) die "Unknown flag: $1" ;;
    esac
  done

  banner
  hr
  log "STEP 1/3 — Installing system dependencies"
  hr
  require_script "$S_DEPS"
  bash "$S_DEPS" "${deps_args[@]}"

  echo ""
  hr
  log "STEP 2/3 — Building llama.cpp from source"
  hr
  require_script "$S_BOOT"
  bash "$S_BOOT" "${boot_args[@]}"

  if [[ -n "$model" ]]; then
    echo ""
    hr
    log "STEP 3/3 — Downloading model: $model"
    hr
    require_script "$S_PULL"
    bash "$S_PULL" --model "$model"
  else
    echo ""
    hr
    ok "Setup complete. Pull a model to get started:"
    echo "  $(basename "$0") pull --model qwen2.5-coder-32b"
    echo "  $(basename "$0") pull --list"
    hr
  fi
}

# ── pull ──────────────────────────────────────────────────────────────────────
cmd_pull() {
  require_script "$S_PULL"
  bash "$S_PULL" "$@"
}

# ── run ───────────────────────────────────────────────────────────────────────
cmd_run() {
  local export_fmt="" run_args=()

  # Intercept --export so we can pipe through export.sh after run
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --export) export_fmt="$2"; shift 2 ;;
      *)        run_args+=("$1"); shift   ;;
    esac
  done

  require_script "$S_RUN"
  bash "$S_RUN" "${run_args[@]}"

  # Post-run export if requested
  if [[ -n "$export_fmt" ]]; then
    # Find the output file from run_args
    OUTPUT="output.json"
    for i in "${!run_args[@]}"; do
      if [[ "${run_args[$i]}" == "--output" || "${run_args[$i]}" == "-o" ]]; then
        OUTPUT="${run_args[$((i+1))]}"
        break
      fi
    done
    if [[ -f "$OUTPUT" ]]; then
      require_script "$S_EXPORT"
      bash "$S_EXPORT" --input "$OUTPUT" --format "$export_fmt"
    fi
  fi
}

# ── analyze ───────────────────────────────────────────────────────────────────
cmd_analyze() {
  local platform=false project="" template="auto" model=""
  local analyze_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --platform)       platform=true;          shift   ;;
      --project|-p)     project="$2";           shift 2 ;;
      --template|-t)    template="$2";          shift 2 ;;
      --model|-m)       model="$2";             shift 2 ;;
      --keep-server)    analyze_args+=(--keep-server); shift ;;
      *) die "Unknown flag: $1" ;;
    esac
  done

  require_script "$S_ANALYZE"

  if [[ "$platform" == "true" ]]; then
    log "Running platform-wide analysis (~/master/)..."
    ARGS=(platform --path "$(dirname "$ML_OPS_ROOT")" --output "$ML_OPS_ROOT/arch-analyzer")
    [[ -n "$model" ]] && ARGS+=(--model "$model")
    bash "$S_ANALYZE" "${ARGS[@]}" "${analyze_args[@]}"
  elif [[ -n "$project" ]]; then
    log "Analyzing project: $project (template: $template)"
    ARGS=(analyze --path "$project" --output "$ML_OPS_ROOT/arch-analyzer" --template "$template")
    [[ -n "$model" ]] && ARGS+=(--model "$model")
    bash "$S_ANALYZE" "${ARGS[@]}" "${analyze_args[@]}"
  else
    die "Specify --platform or --project PATH"
  fi
}

# ── server ────────────────────────────────────────────────────────────────────
cmd_server() {
  require_script "$S_SERV"
  bash "$S_SERV" "$@"
}

# ── health ────────────────────────────────────────────────────────────────────
cmd_health() {
  require_script "$S_HEALTH"
  bash "$S_HEALTH" "$@"
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
  banner

  # GPU
  echo -e "  ${BOLD}GPU:${RST}"
  nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{
        printf "    [%s] %-32s  VRAM: %s / %s  util: %s  temp: %s\n",
        $1, $2, $3, $4, $5, $6
      }' || echo "    (nvidia-smi unavailable)"

  echo ""

  # Server
  echo -e "  ${BOLD}Server:${RST}"
  require_script "$S_SERV"
  bash "$S_SERV" status

  # Models
  echo -e "  ${BOLD}Models:${RST}"
  _DEFAULT_PREFIX="$([[ "$(id -u)" == "0" ]] && echo "/opt/tensorforge/llamacpp" || echo "$HOME/.tensorforge/llamacpp")"
  MODELS_DIR="${TF_MODELS:-${_DEFAULT_PREFIX}/models/gguf}"
  find "$MODELS_DIR" -name "*.gguf" -type f 2>/dev/null | sort | while read -r f; do
    SIZE=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "    ✓ $(basename "$f")  ($SIZE)"
  done || echo "    (no models downloaded — run: $(basename "$0") pull --list)"
  echo ""
}

# ── main dispatcher ───────────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true

case "$CMD" in
  setup)    cmd_setup   "$@" ;;
  shell)    require_script "$S_SHELL"; bash "$S_SHELL" "$@" ;;
  nix)      require_script "$S_NIX";   bash "$S_NIX"   "$@" ;;
  pull)     cmd_pull    "$@" ;;
  run)      cmd_run     "$@" ;;
  analyze)  cmd_analyze "$@" ;;
  server)   cmd_server  "$@" ;;
  health)   cmd_health  "$@" ;;
  status)   cmd_status        ;;
  help|--help|-h) usage       ;;
  *)
    warn "Unknown command: $CMD"
    echo ""
    usage
    exit 1 ;;
esac
