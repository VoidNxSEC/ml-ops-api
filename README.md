# ML Offload API

> **Unified Multi-Backend ML Model Orchestration & Inference Platform**

[!\[License: MIT\](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[!\[Built with Nix\](https://img.shields.io/badge/Built\_With-Nix-5277C3.svg?logo=nixos\&logoColor=white)](https://nixos.org)
[!\[Rust\](https://img.shields.io/badge/Rust-1.70+-orange.svg?logo=rust)](https://www.rust-lang.org/)
[!\[Status: Active Development\](https://img.shields.io/badge/Status-Active\_Development-green.svg)](#)

ML Offload API is a high-performance orchestration layer designed to provide unified access to multiple ML inference backends (Ollama, llama.cpp, vLLM, TGI). It features intelligent VRAM management, automatic backend selection, and deep optimization for NVIDIA Blackwell (B200/B300) architectures.

---

## 🏗️ Architecture

```javascript
graph TD
    subgraph "Platform Services"
        NL[neoland] --> MOA[ml-ops-api]
        CR[cerebro] --> MOA
        AA[arch-analyzer] --> MOA
    end

    subgraph "ml-ops-api Core"
        MOA --> TF[tensorforge / Rust Core]
        TF --> B[Backends]
        B --> LC[llama.cpp]
        B --> VL[vLLM]
        TF --> RM[VRAM Monitoring / NVML]
        TF --> MR[Model Registry / SQLite]
    end

    subgraph "Analysis Layer"
        AA --> PY[Python Analyzer]
        PY --> TF
    end

    subgraph "Infrastructure"
        TF --> NX[NixOS / B200 Optimized]
        TF --> AZ[Azure Fleet]
    end
```

---

## 🚀 Key Features

- 🎯 **Multi-Backend Orchestration** - Unified API for Ollama, llama.cpp, vLLM, and TGI.
- 🧠 **Intelligent Routing** - Automatic backend selection based on VRAM availability, load, and model requirements.
- 📊 **Real-time Monitoring** - GPU memory tracking via NVIDIA NVML for precise resource allocation.
- 🏎️ **Blackwell Optimized** - Specialized NixOS modules for NVIDIA B200 (192GB HBM3e) with FP8 quantization and tensor parallelism.
- 🔌 **WebSocket & Streaming** - Real-time streaming inference for low-latency applications.
- 📦 **Nix-First Workflow** - Reproducible builds, development shells, and declarative deployment via Nix Flakes.

---

## 🛠️ Components

### 1. **TensorForge (Rust Core)**

The heart of the platform, built with Axum and Tokio. It defines a unified `Backend` trait and manages the lifecycle of inference processes.

### 2. **Arch-Analyzer (Python)**

An integrated code analysis tool that leverages the platform's LLM capabilities to perform deep structural analysis of codebases across multiple languages (Rust, Python, Nix, TypeScript).

---

## 📋 Inference Pipeline

The platform provides a suite of scripts for the full "bootstrap, pull, run, export" lifecycle:

| Script          | Role                                                               |
| --------------- | ------------------------------------------------------------------ |
| `bootstrap.sh`  | Build llama.cpp with CUDA, auto-detect GPU arch (sm\_XX).          |
| `server.sh`     | Manage server lifecycle (start/stop/restart/status).               |
| `model-pull.sh` | Download GGUF models from HuggingFace.                             |
| `run.sh`        | **Full pipeline**: auto-start server → infer → export → auto-stop. |
| `export.sh`     | Convert JSON results to MD, Text, JSONL, or CSV.                   |
| `health.sh`     | GPU + server status + latency benchmarks.                          |

---

## 🚀 Quick Start

### Nix (Recommended)

```bash
# Enter development shell
nix develop

# Build the project
nix build

# Run the API
nix run
```

### Bare Metal / Pipeline

```bash
# 1. Build backend (llama.cpp)
./tensorforge/scripts/bootstrap.sh

# 2. Pull a model
./tensorforge/scripts/model-pull.sh --model qwen2.5-coder-32b

# 3. Run a one-shot inference pipeline
./tensorforge/scripts/run.sh --prompt "Explain quantum computing" --output result.json
```

### Docker (via master platform)

The API runs as part of the unified `spectre-net` — it does **not** have its own NATS.
Start it with the `intelligence` profile from `~/master/`:

```bash
# From ~/master/
docker compose --profile core --profile intelligence up -d

# Or standalone build/test (requires NATS reachable at localhost:4222):
docker build -t ml-offload-api .
docker run --rm \
  -e NATS_URL=nats://host.docker.internal:4222 \
  -p 8080:8080 \
  ml-offload-api
```

The service is available at `http://localhost:8083` (master compose maps `8083 → 8080`).

---

## ⚙️ Configuration

| Variable                  | Default                           | Description                          |
| ------------------------- | --------------------------------- | ------------------------------------ |
| `ML_OFFLOAD_HOST`         | `0.0.0.0`                         | Bind address                         |
| `ML_OFFLOAD_PORT`         | `8080`                            | HTTP port (master maps → 8083)       |
| `ML_OFFLOAD_DATA_DIR`     | `/var/lib/ml-offload`             | SQLite + runtime data directory      |
| `ML_OFFLOAD_MODELS_PATH`  | `/var/lib/ml-models`              | GGUF model storage root              |
| `ML_OFFLOAD_DB_PATH`      | `/var/lib/ml-offload/registry.db` | SQLite registry path                 |
| `ML_OFFLOAD_CORS_ENABLED` | `false`                           | Enable permissive CORS               |
| `NATS_URL`                | `nats://nats:4222`                | Shared spectre-net NATS bus          |
| `RUST_LOG`                | `ml_offload_api=info`             | Log level filter (tracing-subscriber)|

---

## 💎 B200 Blackwell Optimization

Declarative configuration via NixOS module:

```javascript
services.tensorforge.b200Optimized = {
  enable = true;
  vllm.enable = true;       # tensor-parallel-size=4
  llamacpp.enable = true;   # --flash-attn, ctx=32768
  monitoring.enable = true; # Prometheus + Grafana
};
```

---

**Maintained by**: VoidNxSEC Team**GPU Program**: NVIDIA Inception (B200/B300 fleet)
