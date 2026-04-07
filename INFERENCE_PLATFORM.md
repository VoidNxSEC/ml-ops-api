# ML-Ops API — Inference Platform

**Status**: Active Development  
**Date**: 2026-04-07  
**GPU Target**: NVIDIA B200 / B300 (Blackwell) via NVIDIA Inception

---

## Overview

`ml-ops-api` is the inference backbone of the voidnxlabs platform.  
It routes LLM requests from all services (neoland, cerebro, arch-analyzer) to the
appropriate GPU backend, with full B200 optimization and batch inference support.

---

## Architecture

```
voidnxlabs platform services
    │
    ▼
ml-ops-api
    ├── tensorforge/          ← Rust core (BackendDriver, B200 NixOS module)
    │   ├── backends/
    │   │   ├── llamacpp/     ← llama.cpp Rust client (async, VRAM tracking)
    │   │   └── vllm/         ← vLLM Rust client
    │   ├── core/             ← unified API (axum + inference trait)
    │   ├── nix/b200-optimized/ ← NixOS module for B200 fleet
    │   └── scripts/          ← Linux bootstrap + pipeline scripts
    │
    └── arch-analyzer/        ← AI code analysis (Python, platform-wide)
        └── src/analyzer.py   ← symlink → ~/master/arch-analyzer/src/analyzer.py
```

---

## tensorforge/scripts — Inference Pipeline

The full "entrar, inferir, extrair, exportar, fechar" lifecycle in bash:

| Script | Role |
|--------|------|
| `bootstrap.sh` | Build llama.cpp from source with CUDA, auto-detect GPU arch (sm_XX), write `env.sh` |
| `server.sh` | start / stop / restart / status / logs / gpu — B200 defaults: `-ngl 999 --flash-attn --cont-batching` |
| `model-pull.sh` | Download GGUF models from HuggingFace registry; supports resume, HF token |
| `infer.sh` | Single-shot inference: prompt / file / stream / JSON mode |
| `run.sh` | **Full pipeline**: auto-start server → infer → export → auto-stop |
| `export.sh` | Convert JSON results to text / markdown / jsonl / csv |
| `health.sh` | GPU + server status + latency benchmark; `--watch N` for live monitoring |
| `Makefile` | Shortcuts: `make bootstrap`, `make run PROMPT="..."`, `make health-watch` |

### Quick Start (NIM / bare Linux)

```bash
# 1. Build llama.cpp with CUDA (once)
./tensorforge/scripts/bootstrap.sh

# 2. Download model
./tensorforge/scripts/model-pull.sh --model qwen2.5-coder-32b

# 3. Full pipeline
./tensorforge/scripts/run.sh --prompt "analyze this Rust codebase" --output result.json

# 4. Export result
./tensorforge/scripts/export.sh --input result.json --format markdown
```

### Model Registry

| Name | VRAM | Best For |
|------|------|----------|
| `qwen2.5-coder-7b` | 8GB | Fast code analysis |
| `qwen2.5-coder-32b` | 35GB | arch-analyzer default |
| `llama-3.1-70b` | 75GB | General purpose, B200 |
| `llama-3.3-70b` | 75GB | Best instruction following |
| `deepseek-r1-14b` | 16GB | Reasoning, fits 24GB |
| `deepseek-r1-70b` | 75GB | Strong reasoning, B200 |

---

## arch-analyzer — Platform-Wide Code Analysis

Integrated into `ml-ops-api`, using the tensorforge llama.cpp backend as LLM.

### Language Templates

| Template | Files | Projects |
|----------|-------|---------|
| `nix` | `*.nix` | spider-nix, spooknix, nixos modules |
| `python` | `*.py` (AST) | cerebro, phantom, neoland-agents, neotron |
| `rust` | `*.rs` | neoland, securellm-bridge, adr-ledger, spectre, tensorforge |
| `typescript` | `*.ts/.tsx/.jsx` | neoland-ui, portfolio |
| `auto` | auto-detect | any project |

### Platform Command

Analyzes **every project** under `~/master/` in one run:

```bash
# Full platform analysis — outputs PLATFORM-REPORT.md
make -C arch-analyzer platform

# With specific model
make -C arch-analyzer platform MODEL=llama-3.3-70b

# Single project
make -C arch-analyzer analyze-project TARGET=~/master/neoland TEMPLATE=rust
```

Output: `PLATFORM-REPORT.md` + `PLATFORM-REPORT.json` with:
- Per-project quality scores, security issues, executive summaries
- Cross-project dependency graph (Cargo path deps, flake inputs)
- Platform-level priority actions from LLM

---

## tensorforge — Rust Core

### B200 Optimizations (NixOS Module)

`nix/b200-optimized/default.nix` — declarative config for NVIDIA B200 (192GB HBM3e):

```nix
services.tensorforge.b200Optimized = {
  enable = true;
  vllm.enable = true;       # tensor-parallel-size=4, max-model-len=131072
  llamacpp.enable = true;   # -ngl 999, --flash-attn, ctx=32768
  systemTuning.enable = true;
  monitoring.enable = true; # Prometheus + Grafana
};
```

Key specs:
- **VRAM**: 192GB HBM3e
- **Context**: up to 131k tokens (vLLM), 32768 (llama.cpp)
- **Tensor parallelism**: 4 (B200 4-die architecture)
- **Quantization**: FP8 (2x throughput)

### Backend Trait

```rust
// tensorforge_core::Backend
async fn infer(request: InferenceRequest) -> TensorForgeResult<InferenceResult>
async fn load_model(model_id: &str, options: LoadOptions) -> TensorForgeResult<LoadResult>
async fn health_check() -> TensorForgeResult<BackendHealth>
async fn vram_usage() -> TensorForgeResult<VramUsage>
fn capabilities() -> BackendCapabilities
```

---

## Integration with Platform

```
neoland control plane  ──HTTP──►  ml-ops-api / tensorforge (port 8080)
arch-analyzer          ──HTTP──►  LLAMACPP_URL=http://127.0.0.1:8080
cerebro RAG            ──HTTP──►  ml-ops-api / tensorforge
```

Environment variables:

| Var | Default | Description |
|-----|---------|-------------|
| `LLAMACPP_URL` | `http://127.0.0.1:8080` | llama.cpp server endpoint |
| `LLM_PARALLEL` | `8` | Concurrent inference slots |
| `LLM_TIMEOUT` | `180` | Request timeout (seconds) |
| `TF_PREFIX` | `/opt/tensorforge/llamacpp` | Install prefix |
| `TF_MODELS` | `/var/lib/tensorforge/models/gguf` | GGUF model directory |
| `HF_TOKEN` | — | HuggingFace token (private models) |

---

## Azure Deployment

`azure-ml/azure-nix-template.sh` provisions the VM fleet:

```bash
# Foundation (RG, VNet, NSG, Storage)
./azure-ml/azure-nix-template.sh foundation

# Deploy VM — nix binary cache (B2s ~$35/mo)
./azure-ml/azure-nix-template.sh deploy

# Stream VM — app hosting (D4s_v5 ~$140/mo)
./azure-ml/azure-nix-template.sh stream

# Status
./azure-ml/azure-nix-template.sh status
```

Budget reference (brazilsouth):
- B2s (2 vCPU/4GB): ~$35/mo — CI/nix-cache
- B4ms (4 vCPU/16GB) Spot: ~$20/mo — ephemeral test runs
- D4s_v5 (4 vCPU/16GB): ~$140/mo — stream/app hosting

---

**Maintained by**: VoidNxSEC Team  
**GPU Program**: NVIDIA Inception (B200/B300 fleet)
