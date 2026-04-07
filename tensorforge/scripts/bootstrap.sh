#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/bootstrap.sh
# Full llama.cpp CUDA bootstrap — build from source on bare Linux / NVIDIA NIM
#
# Detects GPU arch automatically, builds with optimal CUDA flags,
# installs to PREFIX, writes env.sh for subsequent scripts.
#
# Usage:
#   ./scripts/bootstrap.sh
#   ./scripts/bootstrap.sh --prefix /usr/local --skip-deps
#   PREFIX=/opt/llamacpp JOBS=16 ./scripts/bootstrap.sh
# =============================================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PREFIX="${PREFIX:-/opt/tensorforge/llamacpp}"
MODELS_DIR="${MODELS_DIR:-/var/lib/tensorforge/models/gguf}"
BUILD_DIR="${BUILD_DIR:-/tmp/llama-cpp-src}"
REPO="https://github.com/ggerganov/llama.cpp.git"
TAG="${LLAMACPP_TAG:-}"          # empty = latest main
JOBS="${JOBS:-$(nproc)}"

SKIP_DEPS=false
SKIP_BUILD=false
FORCE_REBUILD=false

# ── Output ────────────────────────────────────────────────────────────────────
RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
log()  { echo -e "${BLU}[bootstrap]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }
hr()   { echo -e "${BLU}────────────────────────────────────────────${RST}"; }

# ── Args ──────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)       PREFIX="$2";         shift 2 ;;
    --models-dir)   MODELS_DIR="$2";     shift 2 ;;
    --tag)          TAG="$2";            shift 2 ;;
    --jobs)         JOBS="$2";           shift 2 ;;
    --skip-deps)    SKIP_DEPS=true;      shift   ;;
    --skip-build)   SKIP_BUILD=true;     shift   ;;
    --force-rebuild) FORCE_REBUILD=true; shift   ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  --prefix PATH       Install prefix (default: $PREFIX)"
      echo "  --models-dir PATH   Models directory (default: $MODELS_DIR)"
      echo "  --tag TAG           llama.cpp git tag (default: latest)"
      echo "  --jobs N            Build parallelism (default: nproc)"
      echo "  --skip-deps         Skip apt install"
      echo "  --skip-build        Skip cmake build (use existing binary)"
      echo "  --force-rebuild     Force clean rebuild"
      exit 0 ;;
    *) die "Unknown flag: $1" ;;
  esac
done

hr
echo "  tensorforge — llama.cpp bootstrap"
echo "  PREFIX    : $PREFIX"
echo "  MODELS    : $MODELS_DIR"
echo "  BUILD_DIR : $BUILD_DIR"
echo "  JOBS      : $JOBS"
hr
echo ""

# ── 1. Detect NVIDIA GPU ──────────────────────────────────────────────────────
log "Detecting GPU..."
command -v nvidia-smi &>/dev/null || die "nvidia-smi not found — NVIDIA driver required"

GPU_NAME=$(nvidia-smi --query-gpu=name         --format=csv,noheader | head -1 | xargs)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | xargs)
GPU_DRIV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
GPU_CUDA=$(nvidia-smi | awk '/CUDA Version/{print $NF}')

ok "GPU    : $GPU_NAME"
ok "VRAM   : $GPU_VRAM"
ok "Driver : $GPU_DRIV | CUDA cap: $GPU_CUDA"

# Detect compute capability
CUDA_ARCH=$(python3 -c "
import subprocess, sys
try:
    out = subprocess.check_output(
        ['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],
        text=True, stderr=subprocess.DEVNULL
    ).strip().split('\n')[0].replace('.','')
    print(out)
except:
    print('90')
" 2>/dev/null || echo "90")

ok "Compute: sm_${CUDA_ARCH}"

# ── 2. System deps ────────────────────────────────────────────────────────────
if [[ "$SKIP_DEPS" == "false" ]]; then
  log "Installing build dependencies..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq 2>/dev/null
  apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl wget \
    libcurl4-openssl-dev libopenblas-dev \
    python3 python3-pip jq bc \
    ca-certificates 2>/dev/null
  ok "Build deps installed"
fi

# ── 3. Verify CUDA toolkit ────────────────────────────────────────────────────
log "Checking CUDA toolkit (nvcc)..."
NVCC_PATH=$(command -v nvcc 2>/dev/null \
  || find /usr/local/cuda*/bin -name nvcc 2>/dev/null | sort -V | tail -1 \
  || echo "")

if [[ -n "$NVCC_PATH" ]]; then
  export PATH="$(dirname "$NVCC_PATH"):$PATH"
  NVCC_VER=$(nvcc --version 2>&1 | grep -oP 'release \K[\d.]+' || echo "?")
  ok "nvcc: $NVCC_PATH (v$NVCC_VER)"
  HAS_CUDA=true
else
  warn "nvcc not found — building CPU-only (no CUDA acceleration)"
  HAS_CUDA=false
  CUDA_ARCH=""
fi

# ── 4. Check if already built ────────────────────────────────────────────────
LLAMA_BIN="$PREFIX/bin/llama-server"
if [[ -x "$LLAMA_BIN" && "$SKIP_BUILD" == "false" && "$FORCE_REBUILD" == "false" ]]; then
  CUR_VER=$("$LLAMA_BIN" --version 2>&1 | head -1 || echo "")
  warn "Binary exists: $LLAMA_BIN ($CUR_VER)"
  warn "Use --force-rebuild to rebuild or --skip-build to skip"
  SKIP_BUILD=true
fi

# ── 5. Clone / update repo ────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == "false" ]]; then
  log "Fetching llama.cpp source..."
  if [[ -d "$BUILD_DIR/.git" && "$FORCE_REBUILD" == "false" ]]; then
    git -C "$BUILD_DIR" fetch --depth=1 origin HEAD 2>/dev/null
    git -C "$BUILD_DIR" reset --hard FETCH_HEAD 2>/dev/null
    ok "Source updated"
  else
    [[ -d "$BUILD_DIR" ]] && rm -rf "$BUILD_DIR"
    git clone --depth=1 "$REPO" "$BUILD_DIR" 2>/dev/null
    ok "Source cloned"
  fi

  [[ -n "$TAG" ]] && git -C "$BUILD_DIR" checkout "tags/$TAG" 2>/dev/null && ok "Tag: $TAG"

  # ── 6. CMake build ────────────────────────────────────────────────────────
  log "Configuring build (sm_${CUDA_ARCH:-cpu})..."

  CMAKE_ARGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DLLAMA_CURL=ON
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
  )

  if [[ "$HAS_CUDA" == "true" ]]; then
    CMAKE_ARGS+=(
      -DGGML_CUDA=ON
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
      -DGGML_CUDA_GRAPHS=ON       # CUDA graphs for B200+
      -DGGML_CUDA_FA=ON           # Flash Attention kernel
      -DGGML_CUDA_MMQ=ON          # Matrix multiply kernels
      -DGGML_CUDA_F16=ON          # FP16 operations
    )
  fi

  BUILD_OUT="$BUILD_DIR/build"
  [[ "$FORCE_REBUILD" == "true" ]] && rm -rf "$BUILD_OUT"
  mkdir -p "$BUILD_OUT"

  cmake -S "$BUILD_DIR" -B "$BUILD_OUT" "${CMAKE_ARGS[@]}" > "$BUILD_DIR/cmake.log" 2>&1 \
    || { cat "$BUILD_DIR/cmake.log"; die "CMake configure failed"; }

  log "Building llama-server (${JOBS} jobs)..."
  cmake --build "$BUILD_OUT" \
    --parallel "$JOBS" \
    --target llama-server \
    2>&1 | tail -5

  log "Installing to $PREFIX..."
  cmake --install "$BUILD_OUT" --strip > /dev/null 2>&1

  ok "Build complete"
fi

# ── 7. Validate binary ────────────────────────────────────────────────────────
[[ -x "$LLAMA_BIN" ]] || die "Binary not found after build: $LLAMA_BIN"
BIN_VER=$("$LLAMA_BIN" --version 2>&1 | head -1 || echo "unknown")
ok "llama-server: $BIN_VER"

# ── 8. Create directories ────────────────────────────────────────────────────
log "Creating runtime directories..."
mkdir -p "$MODELS_DIR" "$PREFIX/logs" "$PREFIX/run"
ok "Models  : $MODELS_DIR"
ok "Logs    : $PREFIX/logs"

# ── 9. System symlink ────────────────────────────────────────────────────────
ln -sf "$LLAMA_BIN" /usr/local/bin/llama-server 2>/dev/null \
  && ok "Symlink : /usr/local/bin/llama-server" \
  || warn "Could not symlink to /usr/local/bin (no sudo?)"

# ── 10. Write env.sh ──────────────────────────────────────────────────────────
cat > "$PREFIX/env.sh" <<-ENV
# tensorforge — llama.cpp runtime environment
# Usage:  source $PREFIX/env.sh

export TF_PREFIX="${PREFIX}"
export TF_BIN="${LLAMA_BIN}"
export TF_MODELS="${MODELS_DIR}"
export TF_PORT="\${TF_PORT:-8080}"
export TF_LOG="${PREFIX}/logs/server.log"
export TF_PID="${PREFIX}/run/server.pid"
export LLAMACPP_URL="http://localhost:\${TF_PORT}"
export CUDA_ARCH="${CUDA_ARCH}"
export PATH="${PREFIX}/bin:\$PATH"

# B200 CUDA performance vars
export GGML_CUDA_MAX_STREAMS=32
export GGML_CUDA_F16=1
export GGML_CUDA_MMQ=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
ENV
ok "Env     : $PREFIX/env.sh"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
hr
echo "  Bootstrap complete! GPU: $GPU_NAME"
hr
echo ""
echo "  Quick start:"
echo "    source $PREFIX/env.sh"
echo "    ./scripts/model-pull.sh --model qwen2.5-coder-32b"
echo "    ./scripts/server.sh start"
echo "    ./scripts/infer.sh \"What is 2+2?\""
echo ""
echo "  Full pipeline (NIM / batch):"
echo "    ./scripts/run.sh --prompt \"analyze...\" --output result.json"
echo ""
