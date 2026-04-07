#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/bootstrap-nix.sh
# Nix / NixOS bootstrap for the tensorforge inference pipeline
#
# Three modes:
#   quick     — nix shell: llama-server without installing anything permanently
#   dev       — nix develop: full Rust + Python + llama.cpp dev environment
#   nixos     — NixOS module: declarative systemd service for production
#
# Usage:
#   ./scripts/bootstrap-nix.sh quick                  # try llama.cpp right now
#   ./scripts/bootstrap-nix.sh dev                    # full dev shell
#   ./scripts/bootstrap-nix.sh nixos                  # install NixOS module
#   ./scripts/bootstrap-nix.sh install-nix            # install Nix if missing
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_OPS_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # ml-ops-api/

RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
BOLD='\033[1m'
log()  { echo -e "${BLU}[bootstrap-nix]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }
hr()   { echo -e "${BLU}────────────────────────────────────────────${RST}"; }

# ── Detect environment ────────────────────────────────────────────────────────
is_nixos()    { [[ -f /etc/nixos/configuration.nix ]]; }
has_nix()     { command -v nix &>/dev/null; }
has_gpu()     { command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null 2>&1; }
is_root()     { [[ "$(id -u)" == "0" ]]; }

detect_gpu_arch() {
  python3 -c "
import subprocess
try:
    out = subprocess.check_output(
        ['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],
        text=True, stderr=subprocess.DEVNULL
    ).strip().split('\n')[0].replace('.','')
    print(out)
except:
    print('90')
" 2>/dev/null || echo "90"
}

usage() {
  cat <<EOF
${BOLD}Usage:${RST} $(basename "$0") <command>

${BOLD}Commands:${RST}

  ${GRN}quick${RST}        Drop into a nix shell with llama-server available (no install)
               Uses: nix shell nixpkgs#llama-cpp

  ${GRN}dev${RST}          Full dev shell: Rust + Python + llama.cpp + CUDA tools
               Uses: nix develop (from ml-ops-api flake.nix)

  ${GRN}nixos${RST}        Install tensorforge as a NixOS declarative service
               Adds module to /etc/nixos, enables systemd llama-server

  ${GRN}install-nix${RST}  Install Nix (Determinate Systems installer) on non-NixOS Linux

  ${GRN}status${RST}       Show Nix/NixOS/GPU environment summary

${BOLD}Examples:${RST}
  $(basename "$0") quick                         # try it in 30 seconds
  $(basename "$0") dev                           # development environment
  $(basename "$0") nixos --gpu b200              # production B200 config
  $(basename "$0") nixos --gpu l40s --port 8080

EOF
}

# ── install-nix ───────────────────────────────────────────────────────────────
cmd_install_nix() {
  has_nix && { ok "Nix already installed: $(nix --version)"; return 0; }
  is_nixos && { ok "This is NixOS — Nix is always available"; return 0; }

  log "Installing Nix via Determinate Systems installer..."
  log "(multi-user, flakes enabled by default)"
  echo ""

  curl -fsSL https://install.determinate.systems/nix \
    | sh -s -- install --no-confirm

  # Source nix
  if [[ -f /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]]; then
    source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
  fi

  ok "Nix installed: $(nix --version)"
  warn "Restart your shell or run: source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh"
}

# ── quick ─────────────────────────────────────────────────────────────────────
cmd_quick() {
  has_nix || die "Nix not found. Run: $(basename "$0") install-nix"

  hr
  log "Dropping into nix shell with llama.cpp..."
  log "No permanent install — exits cleanly when you leave the shell"
  hr
  echo ""

  NIX_FLAGS="--extra-experimental-features nix-command flakes"

  if has_gpu; then
    ARCH=$(detect_gpu_arch)
    ok "GPU detected (sm_${ARCH})"

    # IMPORTANT: nixpkgs#llama-cpp does NOT include CUDA by default.
    # cudaSupport must be set at nixpkgs import time, not per-package.
    # We use the flake's .#llama-cpp-cuda output which sets cudaSupport=true.
    if [[ -f "$ML_OPS_ROOT/flake.nix" ]]; then
      log "Using flake .#llama-cpp-cuda (CUDA-enabled, first run builds ~10 min)..."
      warn "First run compiles llama.cpp with CUDA from source via Nix — grab a coffee."
      exec nix shell $NIX_FLAGS \
        "$ML_OPS_ROOT#llama-cpp-cuda" \
        "nixpkgs#jq" "nixpkgs#curl" \
        --command ${SHELL:-bash} -c '
          echo ""
          echo "  tensorforge — llama.cpp CUDA shell (via Nix)"
          echo "  $(llama-server --version 2>&1 | head -1)"
          echo ""
          echo "  llama-server --model /path/to/model.gguf --port 8080 -ngl 999 --flash-attn"
          echo ""
          exec '"${SHELL:-bash}"
    else
      warn "flake.nix not found — falling back to nixpkgs#llama-cpp (CPU only)"
      warn "For CUDA support clone the repo first: git clone ... ml-ops-api"
      exec nix shell $NIX_FLAGS nixpkgs#llama-cpp nixpkgs#jq nixpkgs#curl \
        --command ${SHELL:-bash}
    fi
  else
    warn "No GPU detected — CPU-only llama.cpp"
    exec nix shell $NIX_FLAGS nixpkgs#llama-cpp nixpkgs#jq nixpkgs#curl \
      --command ${SHELL:-bash} -c '
        echo "  llama-server (CPU): $(llama-server --version 2>&1 | head -1)"
        exec '"${SHELL:-bash}"
  fi
}

# ── dev ───────────────────────────────────────────────────────────────────────
cmd_dev() {
  has_nix || die "Nix not found. Run: $(basename "$0") install-nix"

  [[ -f "$ML_OPS_ROOT/flake.nix" ]] || die "flake.nix not found at $ML_OPS_ROOT"

  hr
  log "Entering ml-ops-api dev shell..."
  log "Includes: Rust toolchain + Python 3.13 + llama.cpp + CUDA packages"
  hr
  echo ""

  cd "$ML_OPS_ROOT"
  exec nix develop \
    --extra-experimental-features "nix-command flakes" \
    "${@}"
}

# ── nixos ─────────────────────────────────────────────────────────────────────
cmd_nixos() {
  is_nixos || die "Not a NixOS system. Use bootstrap.sh for bare Linux."
  is_root  || die "NixOS module install requires root. Run with sudo."

  local gpu_profile="generic"
  local port=8080
  local model_dir="/var/lib/tensorforge/models/gguf"
  local dry_run=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu)      gpu_profile="$2"; shift 2 ;;
      --port)     port="$2";        shift 2 ;;
      --model-dir) model_dir="$2"; shift 2 ;;
      --dry-run)  dry_run=true;     shift   ;;
      *) die "Unknown flag: $1" ;;
    esac
  done

  hr
  log "Installing tensorforge NixOS module"
  log "GPU profile : $gpu_profile"
  log "Port        : $port"
  log "Models dir  : $model_dir"
  hr
  echo ""

  NIXOS_DIR="/etc/nixos"
  TF_MODULE_SRC="$ML_OPS_ROOT/tensorforge/nix"
  TF_MODULE_DST="$NIXOS_DIR/modules/tensorforge"

  # 1. Copy nix module
  log "Copying tensorforge nix module → $TF_MODULE_DST"
  if [[ "$dry_run" == "false" ]]; then
    mkdir -p "$TF_MODULE_DST"
    cp -r "$TF_MODULE_SRC/"* "$TF_MODULE_DST/"
    ok "Module copied"
  else
    warn "DRY-RUN: cp -r $TF_MODULE_SRC/ $TF_MODULE_DST/"
  fi

  # 2. Generate tensorforge.nix config
  TF_CONFIG="$NIXOS_DIR/tensorforge.nix"
  log "Generating $TF_CONFIG..."

  if [[ "$dry_run" == "false" ]]; then
  cat > "$TF_CONFIG" <<NIXCFG
# tensorforge — llama.cpp inference service
# Generated by bootstrap-nix.sh on $(date -Iseconds)
# GPU profile: $gpu_profile

{ config, lib, pkgs, ... }:

{
  imports = [ ./modules/tensorforge/b200-optimized/default.nix ];

  # ── Model storage ──────────────────────────────────────────────────────────
  systemd.tmpfiles.rules = [
    "d ${model_dir} 0755 root root -"
  ];

  # ── Environment ────────────────────────────────────────────────────────────
  environment.variables = {
    TF_MODELS   = "${model_dir}";
    LLAMACPP_URL = "http://127.0.0.1:${port}";
  };

$(
  case "$gpu_profile" in
    b200|b300)
  cat <<B200
  # ── B200/B300 optimized ──────────────────────────────────────────────────
  services.tensorforge.b200Optimized = {
    enable = true;
    llamacpp.enable = true;   # systemd: llama-server -ngl 999 --flash-attn
    vllm.enable     = false;  # enable if you want vLLM alongside
    systemTuning.enable  = true;
    monitoring.enable    = true;
  };
B200
      ;;
    h100|a100)
  cat <<H100
  # ── H100/A100 config ─────────────────────────────────────────────────────
  services.llamacpp-turbo = {
    enable         = true;
    port           = ${port};
    n_gpu_layers   = 999;
    flashAttention = true;
    n_ctx          = 32768;
    n_parallel     = 8;
    modelsDir      = "${model_dir}";
  };
H100
      ;;
    l40s)
  cat <<L40S
  # ── L40S (48GB) config ───────────────────────────────────────────────────
  services.llamacpp-turbo = {
    enable         = true;
    port           = ${port};
    n_gpu_layers   = 999;
    flashAttention = true;
    n_ctx          = 16384;
    n_parallel     = 4;
    modelsDir      = "${model_dir}";
  };
L40S
      ;;
    *)
  cat <<GENERIC
  # ── Generic GPU config ───────────────────────────────────────────────────
  services.llamacpp-turbo = {
    enable         = true;
    port           = ${port};
    n_gpu_layers   = 80;
    flashAttention = false;
    n_ctx          = 8192;
    modelsDir      = "${model_dir}";
  };
GENERIC
      ;;
  esac
)
}
NIXCFG
  ok "Config written → $TF_CONFIG"
  else
    warn "DRY-RUN: would write $TF_CONFIG"
  fi

  # 3. Check if already imported in configuration.nix
  if ! grep -q "tensorforge.nix" "$NIXOS_DIR/configuration.nix" 2>/dev/null; then
    warn "Add to /etc/nixos/configuration.nix:"
    echo ""
    echo '  imports = [ ./tensorforge.nix ];'
    echo ""
  fi

  # 4. Offer to rebuild
  if [[ "$dry_run" == "false" ]]; then
    echo ""
    log "Ready to rebuild NixOS? This will activate the tensorforge service."
    read -rp "  Run nixos-rebuild switch? [y/N]: " ans
    if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
      nixos-rebuild switch
      ok "NixOS rebuilt"
      echo ""
      echo "  Check service: systemctl status llamacpp-turbo"
      echo "  Follow logs  : journalctl -fu llamacpp-turbo"
      echo "  Health check : ./tensorforge/scripts/health.sh"
    else
      warn "Skipped rebuild. Run manually:"
      echo "  sudo nixos-rebuild switch"
    fi
  fi
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
  hr
  echo -e "  ${BOLD}tensorforge — nix environment status${RST}"
  hr
  echo ""

  # Nix
  if has_nix; then
    ok "nix     : $(nix --version 2>/dev/null)"
    nix config show experimental-features 2>/dev/null | grep -q flakes \
      && ok "flakes  : enabled" \
      || warn "flakes  : not enabled (add to nix.conf: experimental-features = nix-command flakes)"
  else
    warn "nix     : NOT installed"
    echo "  Install: $(basename "$0") install-nix"
  fi

  echo ""

  # NixOS
  if is_nixos; then
    ok "nixos   : YES — $(nixos-version 2>/dev/null || echo 'version unknown')"
    if [[ -f /etc/nixos/tensorforge.nix ]]; then
      ok "module  : tensorforge.nix present"
    else
      warn "module  : tensorforge.nix not found"
      echo "  Install: sudo $(basename "$0") nixos"
    fi
  else
    ok "nixos   : no (bare Linux / container)"
  fi

  echo ""

  # GPU
  if has_gpu; then
    GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    ARCH=$(detect_gpu_arch)
    ok "gpu     : $GPU  (sm_${ARCH})"
  else
    warn "gpu     : nvidia-smi not available or NVML mismatch"
  fi

  echo ""

  # flake.nix
  if [[ -f "$ML_OPS_ROOT/flake.nix" ]]; then
    ok "flake   : $ML_OPS_ROOT/flake.nix"
    echo "  shells : nix develop (default, mlops)"
  else
    warn "flake   : not found at $ML_OPS_ROOT"
  fi

  echo ""
}

# ── main ──────────────────────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true

case "$CMD" in
  quick)       cmd_quick          ;;
  dev)         cmd_dev     "$@"   ;;
  nixos)       cmd_nixos   "$@"   ;;
  install-nix) cmd_install_nix    ;;
  status)      cmd_status         ;;
  help|--help|-h) usage           ;;
  *)
    warn "Unknown command: $CMD"
    echo ""
    usage
    exit 1 ;;
esac
