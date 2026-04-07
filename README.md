# ml-ops-api

> GPU inference backbone for the voidnxlabs platform

[![CI](https://github.com/VoidNxSEC/ml-ops-api/actions/workflows/ci.yml/badge.svg)](https://github.com/VoidNxSEC/ml-ops-api/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with Nix](https://img.shields.io/badge/Built_With-Nix-5277C3.svg?logo=nixos&logoColor=white)](https://nixos.org)
[![Rust](https://img.shields.io/badge/Rust-stable-orange.svg?logo=rust)](https://www.rust-lang.org/)

Routes LLM requests from platform services (neoland, cerebro, arch-analyzer) to GPU backends.  
Optimized for NVIDIA Blackwell (B200/B300) via NVIDIA Inception, with full batch inference support.

---

## Architecture

```
voidnxlabs services (neoland · cerebro · arch-analyzer)
                │
                ▼
          ml-ops-api
          ├── tensorforge/          Rust core — BackendDriver, VRAM management
          │   ├── backends/
          │   │   ├── llamacpp/     llama.cpp async Rust client
          │   │   └── vllm/         vLLM client
          │   ├── core/             unified axum API + inference trait
          │   ├── nix/              NixOS modules (b200-optimized, profiles)
          │   └── scripts/          inference pipeline scripts
          └── arch-analyzer/        AI code analysis (Python, all projects)
```

---

## Quick Start

### Nix (recommended)

```bash
# Dev shell — Rust + Python + CUDA tools
nix develop

# Build
nix build .#api
nix build .#forge
```

### Bare Linux / Cloud GPU

```bash
# One-shot: deps → build → model → run
./tensorforge/scripts/entrypoint.sh setup --model qwen2.5-coder-32b

# If nvidia-smi fails (toolkit installed, no reboot — spot instance safe)
./tensorforge/scripts/bootstrap.sh --skip-deps --force-rebuild --cuda-arch 89

# Nice shell
./tensorforge/scripts/setup-shell.sh && exec zsh
```

### NixOS (production)

```bash
# Generate service config for your GPU
sudo ./tensorforge/scripts/bootstrap-nix.sh nixos --gpu b200
# or: --gpu h100 / --gpu l40s / --gpu generic

# Then rebuild
sudo nixos-rebuild switch
```

Full guide: **[GETTING-STARTED.md](GETTING-STARTED.md)**

---

## Scripts

```
tensorforge/scripts/
├── entrypoint.sh      single entrypoint — setup · run · analyze · server · health · shell · nix
├── bootstrap.sh       build llama.cpp from source (--cuda-arch bypass for NVML mismatches)
├── bootstrap-nix.sh   Nix/NixOS bootstrap — quick · dev · nixos · install-nix
├── install-deps.sh    all apt packages + Python venv
├── setup-shell.sh     zsh + starship + fzf + bat + eza + gpu aliases
├── server.sh          manage llama-server process
├── model-pull.sh      download GGUF models from HuggingFace
├── infer.sh           single inference call
├── run.sh             full pipeline: start → infer → stop
├── export.sh          JSON → text / markdown / jsonl / csv
└── health.sh          GPU + server health + latency benchmark
```

### GPU arch flags

| GPU | Flag |
|-----|------|
| B200 / B300 | `--cuda-arch 100` |
| H100 / H200 | `--cuda-arch 90` |
| A100 | `--cuda-arch 80` |
| L40S / RTX 4090 | `--cuda-arch 89` |
| RTX 3090 / A6000 | `--cuda-arch 86` |
| T4 | `--cuda-arch 75` |

---

## Inference Pipeline

Batch pattern — "entrar, inferir, exportar, fechar":

```bash
# Prepare prompts
ls prompts/*.txt

# Full batch: server starts once → all prompts → server stops
./tensorforge/scripts/run.sh --batch prompts/ --output-dir results/

# Export
./tensorforge/scripts/export.sh --dir results/ --format markdown
```

---

## arch-analyzer

Platform-wide AI code analysis using the tensorforge backend:

```bash
# Analyze all ~/master/ projects
./tensorforge/scripts/entrypoint.sh analyze --platform

# Single project
./tensorforge/scripts/entrypoint.sh analyze \
  --project ~/master/neoland \
  --template rust
```

Templates: `rust` · `python` · `typescript` · `nix` · `auto`

---

## NixOS — B200 Optimized

```nix
imports = [ ./modules/tensorforge/b200-optimized/default.nix ];

services.tensorforge.b200Optimized = {
  enable = true;
  llamacpp.enable     = true;   # -ngl 999, --flash-attn, ctx=65536
  vllm.enable         = false;
  systemTuning.enable = true;
  monitoring.enable   = true;   # Prometheus + Grafana
};
```

---

## Environment Variables

| Var | Default | Description |
|-----|---------|-------------|
| `LLAMACPP_URL` | `http://127.0.0.1:8080` | Server endpoint |
| `TF_PREFIX` | `~/.tensorforge/llamacpp` | Install prefix (auto: root uses /opt/) |
| `TF_MODELS` | `$TF_PREFIX/models/gguf` | GGUF model directory |
| `HF_TOKEN` | — | HuggingFace token (private models) |
| `N_GPU_LAYERS` | `999` | GPU offload layers |
| `CTX_SIZE` | `auto` | Context window (VRAM-based) |
| `MLOCK` | `false` | Lock model in RAM (bare-metal only) |

---

**Maintained by**: VoidNxSEC Team  
**GPU Program**: NVIDIA Inception — B200/B300 fleet
