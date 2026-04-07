#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/install-deps.sh
# Install ALL system dependencies for the tensorforge inference pipeline
#
# Covers: llama.cpp build, Python env, arch-analyzer, health monitoring
#
# Usage:
#   ./scripts/install-deps.sh              # auto-detect, use sudo if needed
#   ./scripts/install-deps.sh --dry-run    # show what would be installed
#   ./scripts/install-deps.sh --no-python  # skip Python packages
#   ./scripts/install-deps.sh --no-cuda    # skip CUDA/nvcc checks
# =============================================================================
set -euo pipefail

DRY_RUN=false
WITH_PYTHON=true
CHECK_CUDA=true

RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
BOLD='\033[1m'
log()  { echo -e "${BLU}[install-deps]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
skip() { echo -e "${YLW}[skip]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }
hr()   { echo -e "${BLU}────────────────────────────────────────────${RST}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=true;      shift ;;
    --no-python)  WITH_PYTHON=false; shift ;;
    --no-cuda)    CHECK_CUDA=false;  shift ;;
    -h|--help)
      echo "Usage: $0 [--dry-run] [--no-python] [--no-cuda]"
      exit 0 ;;
    *) die "Unknown flag: $1" ;;
  esac
done

# ── Privilege setup ────────────────────────────────────────────────────────────
SUDO=""
if [[ "$(id -u)" != "0" ]]; then
  command -v sudo &>/dev/null || die "Not root and sudo not available. Run as root or install sudo."
  SUDO="sudo"
  log "Running as non-root — will use sudo"
fi

apt_install() {
  if [[ "$DRY_RUN" == "true" ]]; then
    skip "DRY-RUN: apt-get install $*"
  else
    $SUDO apt-get install -y --no-install-recommends "$@"
  fi
}

hr
echo -e "  ${BOLD}tensorforge — system dependency installer${RST}"
echo    "  Target: llama.cpp build + arch-analyzer + monitoring"
hr
echo ""

# ── 1. Base system ─────────────────────────────────────────────────────────────
log "Updating package index..."
[[ "$DRY_RUN" == "false" ]] && $SUDO apt-get update -qq

log "[1/6] Base build tools..."
apt_install \
  build-essential \
  cmake \
  ninja-build \
  git \
  curl \
  wget \
  ca-certificates \
  unzip \
  pkg-config
ok "Base build tools"

# ── 2. llama.cpp build deps ────────────────────────────────────────────────────
log "[2/6] llama.cpp build dependencies..."
apt_install \
  libcurl4-openssl-dev \
  libssl-dev \
  libopenblas-dev \
  libgomp1 \
  ccache
ok "llama.cpp build deps"

# ── 3. Python + arch-analyzer ─────────────────────────────────────────────────
log "[3/6] Python runtime..."
apt_install \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev
ok "Python runtime"

# ── 4. Utilities (jq, bc, htop, etc.) ────────────────────────────────────────
log "[4/6] Utilities..."
apt_install \
  jq \
  bc \
  htop \
  procps \
  lsof \
  net-tools \
  psmisc \
  tree \
  less \
  nano \
  vim-tiny \
  tmux
ok "Utilities"

# ── 5. Monitoring + performance ────────────────────────────────────────────────
log "[5/6] Monitoring tools..."
apt_install \
  sysstat \
  iotop \
  numactl \
  linux-tools-common \
  linux-tools-generic 2>/dev/null \
  || warn "linux-tools-generic not available (OK in containers)"
ok "Monitoring tools"

# ── 6. CUDA toolkit check ──────────────────────────────────────────────────────
if [[ "$CHECK_CUDA" == "true" ]]; then
  log "[6/6] Checking CUDA toolkit..."

  if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version 2>&1 | grep -oP 'release \K[\d.]+' || echo "?")
    ok "nvcc found: v${NVCC_VER}"
  else
    # Try to find nvcc in common CUDA paths
    NVCC_FOUND=""
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
      [[ -x "$d/nvcc" ]] && { NVCC_FOUND="$d/nvcc"; break; }
    done

    if [[ -n "$NVCC_FOUND" ]]; then
      warn "nvcc found at $NVCC_FOUND but not in PATH"
      warn "Add to PATH: export PATH=\"$(dirname "$NVCC_FOUND"):\$PATH\""
    else
      warn "nvcc not found — CUDA toolkit not installed"
      warn "For GPU builds, install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
      warn "Or on Ubuntu/Debian:"
      warn "  apt-get install -y cuda-toolkit-12-x  (replace x with your version)"
    fi
  fi
fi

# ── Python packages ────────────────────────────────────────────────────────────
if [[ "$WITH_PYTHON" == "true" ]]; then
  echo ""
  log "Installing Python packages for arch-analyzer..."

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ARCH_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/arch-analyzer"
  VENV="$ARCH_DIR/.venv"

  if [[ ! -d "$VENV" ]]; then
    log "Creating Python venv at $VENV..."
    [[ "$DRY_RUN" == "false" ]] && python3 -m venv "$VENV"
  fi

  PKGS=(
    aiohttp
    aiofiles
    pydantic
    rich
    typer
    httpx
    pyyaml
    psutil
  )

  if [[ "$DRY_RUN" == "true" ]]; then
    skip "DRY-RUN: pip install ${PKGS[*]}"
  else
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q "${PKGS[@]}"
    ok "Python packages installed in $VENV"
  fi
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
hr
echo -e "  ${GRN}${BOLD}All dependencies installed!${RST}"
hr
echo ""
echo "  Next steps:"
echo "    1. ./scripts/bootstrap.sh         # Build llama.cpp from source"
echo "    2. ./scripts/model-pull.sh --list  # Browse models"
echo "    3. ./scripts/model-pull.sh --model qwen2.5-coder-32b"
echo "    4. ./scripts/server.sh start"
echo "    5. ./scripts/infer.sh 'Hello!'"
echo ""
echo "  Full pipeline:"
echo "    ./scripts/run.sh --prompt 'analyze codebase...' --output result.json"
echo ""
