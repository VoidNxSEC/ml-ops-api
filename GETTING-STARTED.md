# tensorforge — Getting Started Guide

> GPU inference pipeline: llama.cpp + arch-analyzer + batch inference  
> For NVIDIA Inception (B200/B300), Shadeform, NIM containers, and NixOS

---

## Choose your path

| Environment | Script | Time |
|-------------|--------|------|
| Bare Linux (Ubuntu/Debian) | `entrypoint.sh setup` | ~10 min |
| Cloud GPU (Shadeform, Lambda, RunPod) | `entrypoint.sh setup --cuda-arch XX` | ~10 min |
| NVIDIA NIM / K8s container | `bootstrap.sh --skip-deps --cuda-arch XX` | ~8 min |
| NixOS (production) | `bootstrap-nix.sh nixos --gpu b200` | ~5 min |
| Nix (dev shell, no install) | `bootstrap-nix.sh quick` | ~2 min |

---

## Path 1 — Bare Linux (Ubuntu/Debian)

Full setup in one command:

```bash
git clone git@github.com:VoidNxSEC/ml-ops-api.git
cd ml-ops-api

# Everything: deps + build + model + nice shell
./tensorforge/scripts/entrypoint.sh setup --model qwen2.5-coder-32b
./tensorforge/scripts/setup-shell.sh
exec zsh
```

Then:

```bash
./tensorforge/scripts/entrypoint.sh run \
  --prompt "analyze this Rust codebase for security issues" \
  --output result.json
```

---

## Path 2 — Cloud GPU (Shadeform / Lambda / RunPod)

```bash
git clone git@github.com:VoidNxSEC/ml-ops-api.git
cd ml-ops-api/tensorforge/scripts

# 1. Install system packages
sudo ./install-deps.sh

# 2. Build llama.cpp with CUDA
#    GPU arch reference:
#      L40S / L40 / RTX 4090  → 89
#      H100 / H200 / A100      → 90  (H100=90, A100=80)
#      B200 / B300             → 100
#      RTX 3090 / A6000        → 86
#      RTX 3080 / A40          → 86
#      T4                      → 75
sudo ./bootstrap.sh --cuda-arch 89   # L40S example

# 3. Nice shell
./setup-shell.sh && exec zsh

# 4. Download model
./model-pull.sh --model qwen2.5-coder-32b

# 5. Run
./server.sh start
./infer.sh "Hello from L40S"
```

### If nvidia-smi fails (driver/library mismatch after toolkit install)

```bash
# Try to fix without reboot
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
nvidia-smi

# If still broken — bypass nvidia-smi entirely
sudo ./bootstrap.sh --skip-deps --force-rebuild --cuda-arch 89
```

---

## Path 3 — NVIDIA NIM / K8s Container

NIM containers run as non-root. The scripts handle this automatically:
- `PREFIX` resolves to `~/.tensorforge/llamacpp`
- `sudo` used only when available
- `mlock`/`no-mmap` default to `false` (no `CAP_IPC_LOCK` in containers)

```bash
git clone git@github.com:VoidNxSEC/ml-ops-api.git
cd ml-ops-api/tensorforge/scripts

./bootstrap.sh --cuda-arch 100   # B200 = sm_100
./model-pull.sh --model llama-3.3-70b
./server.sh start
```

Batch inference fleet pattern ("entrar, inferir, exportar, fechar"):

```bash
# Prepare prompts
ls prompts/*.txt

# Run entire batch — server starts once, processes all, stops
./run.sh --batch prompts/ --output-dir results/

# Export all to markdown
./export.sh --dir results/ --format markdown --out-dir reports/
```

---

## Path 4 — NixOS (Production)

```bash
# Dry-run first to preview changes
sudo ./tensorforge/scripts/bootstrap-nix.sh nixos --gpu b200 --dry-run

# Apply
sudo ./tensorforge/scripts/bootstrap-nix.sh nixos --gpu b200

# Or for L40S:
sudo ./tensorforge/scripts/bootstrap-nix.sh nixos --gpu l40s --port 8080
```

This generates `/etc/nixos/tensorforge.nix` with the appropriate service config.
Add to your `configuration.nix`:

```nix
imports = [ ./tensorforge.nix ];
```

Then rebuild:

```bash
sudo nixos-rebuild switch
systemctl status llamacpp-turbo
journalctl -fu llamacpp-turbo
```

### B200 optimized module (direct)

```nix
# /etc/nixos/configuration.nix
{
  imports = [ ./modules/tensorforge/b200-optimized/default.nix ];

  services.tensorforge.b200Optimized = {
    enable = true;
    llamacpp.enable     = true;   # -ngl 999, --flash-attn, ctx=65536
    vllm.enable         = false;
    systemTuning.enable = true;
    monitoring.enable   = true;   # Prometheus + Grafana
  };
}
```

---

## Path 5 — Nix Dev Shell (no install)

Try llama.cpp without installing anything permanently:

```bash
./tensorforge/scripts/bootstrap-nix.sh quick
# → drops into nix shell with llama-server available
```

Full dev environment (Rust + Python + llama.cpp):

```bash
./tensorforge/scripts/bootstrap-nix.sh dev
# → nix develop from ml-ops-api/flake.nix
```

---

## Script Reference

```
tensorforge/scripts/
│
├── entrypoint.sh       ← SINGLE ENTRYPOINT — use this
│   commands:
│     setup             install deps + build + optional model pull
│     run               full pipeline (start → infer → stop)
│     analyze           arch-analyzer against project or platform
│     server            start/stop/restart/status/logs/gpu
│     health            GPU + server + benchmark
│     status            quick overview
│     shell             setup zsh + starship + aliases
│
├── bootstrap.sh        bare Linux — build llama.cpp from source
│   --cuda-arch XX      bypass nvidia-smi (mismatch workaround)
│   --skip-deps         skip apt install
│   --force-rebuild     clean rebuild
│
├── bootstrap-nix.sh    Nix / NixOS bootstrap
│   quick               nix shell with llama-server
│   dev                 nix develop (full dev env)
│   nixos               install NixOS module
│   install-nix         install Nix via Determinate Systems
│
├── install-deps.sh     all apt packages + Python venv
│   --dry-run           preview without installing
│   --no-python         skip Python packages
│
├── setup-shell.sh      zsh + starship + fzf + bat + eza + aliases
│   --no-starship       skip starship prompt
│
├── server.sh           manage llama-server process
├── model-pull.sh       download GGUF models from HuggingFace
├── infer.sh            single inference call
├── run.sh              full pipeline (used by entrypoint)
├── export.sh           JSON → text/markdown/jsonl/csv
└── health.sh           health check + benchmark
```

---

## Model Registry

| Name | VRAM | Best For |
|------|------|----------|
| `qwen2.5-coder-7b` | 8 GB | Fast code analysis |
| `qwen2.5-coder-32b` | 35 GB | arch-analyzer default |
| `mistral-24b` | 26 GB | Balanced |
| `deepseek-r1-14b` | 16 GB | Reasoning, fits 24 GB |
| `llama-3.1-8b` | 9 GB | Fast general purpose |
| `llama-3.3-70b` | 75 GB | Best instruction following |
| `deepseek-r1-70b` | 75 GB | Strong reasoning, B200 |

```bash
./tensorforge/scripts/model-pull.sh --list
./tensorforge/scripts/model-pull.sh --model qwen2.5-coder-32b
HF_TOKEN=hf_... ./tensorforge/scripts/model-pull.sh --model llama-3.3-70b
```

---

## GPU Arch Reference

| GPU | CUDA Arch | Flag |
|-----|-----------|------|
| NVIDIA B200 / B300 | sm_100 | `--cuda-arch 100` |
| NVIDIA H100 / H200 | sm_90 | `--cuda-arch 90` |
| NVIDIA A100 | sm_80 | `--cuda-arch 80` |
| NVIDIA L40S / L40 / RTX 4090 | sm_89 | `--cuda-arch 89` |
| NVIDIA RTX 3090 / A6000 | sm_86 | `--cuda-arch 86` |
| NVIDIA T4 | sm_75 | `--cuda-arch 75` |

---

## Environment Variables

| Var | Default | Description |
|-----|---------|-------------|
| `TF_PREFIX` | `~/.tensorforge/llamacpp` (non-root) | Install prefix |
| `TF_MODELS` | `$TF_PREFIX/models/gguf` | GGUF model directory |
| `LLAMACPP_URL` | `http://127.0.0.1:8080` | Server endpoint |
| `LLM_PARALLEL` | `8` | Concurrent inference slots |
| `LLM_TIMEOUT` | `180` | Request timeout (seconds) |
| `HF_TOKEN` | — | HuggingFace token (private models) |
| `MLOCK` | `false` | Lock model in RAM (bare-metal only) |
| `NO_MMAP` | `false` | Disable mmap (bare-metal only) |
| `N_GPU_LAYERS` | `999` | GPU layers (-1 = all) |
| `CTX_SIZE` | `auto` | Context window (auto = VRAM-based) |

---

## Platform Integration

```
neoland control plane  ──HTTP──►  tensorforge (port 8080)
arch-analyzer          ──HTTP──►  LLAMACPP_URL=http://127.0.0.1:8080
cerebro RAG            ──HTTP──►  tensorforge
```

Run full platform analysis (all `~/master/` projects):

```bash
./tensorforge/scripts/entrypoint.sh analyze --platform
./tensorforge/scripts/entrypoint.sh analyze --platform --model llama-3.3-70b

# Single project
./tensorforge/scripts/entrypoint.sh analyze \
  --project ~/master/neoland \
  --template rust
```

---

## Troubleshooting

### nvidia-smi: Failed to initialize NVML: Driver/library version mismatch

Happens after installing `cuda-toolkit` without rebooting.

```bash
# Option 1: reload module
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
nvidia-smi

# Option 2: bypass detection (spot instance — can't reboot)
./bootstrap.sh --skip-deps --force-rebuild --cuda-arch 89
```

### llama-server binary not found after build

Usually caused by building with `LLAMA_BUILD_EXAMPLES=OFF` (fixed in current version).
Force rebuild:

```bash
./bootstrap.sh --force-rebuild
```

### mlock: Operation not permitted (containers)

```bash
# Already defaulted to false. Explicitly:
MLOCK=false ./server.sh start
```

### Server not responding after start

```bash
./server.sh logs           # check what's happening
./health.sh                # GPU + server + benchmark
./server.sh status         # PID + slots + model info
```

---

**Maintained by**: VoidNxSEC Team  
**GPU Program**: NVIDIA Inception (B200/B300 fleet)  
**Cloud GPU**: Shadeform / Lambda Labs / RunPod
