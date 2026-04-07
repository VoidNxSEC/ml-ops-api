# tensorforge — Nix & NixOS Guide

> For the Nix community: what works, what doesn't, and why.

---

## TL;DR

| Goal | Command |
|------|---------|
| Try llama-server right now (CUDA) | `nix shell .#llama-cpp-cuda` |
| Try llama-server right now (CPU) | `nix shell nixpkgs#llama-cpp` |
| Full dev shell (Rust + CUDA) | `nix develop .#cuda` |
| NixOS production service | `./tensorforge/scripts/bootstrap-nix.sh nixos --gpu <profile>` |
| Install Nix on bare Linux | `./tensorforge/scripts/bootstrap-nix.sh install-nix` |

---

## The CUDA problem in Nix

This is the part that bites everyone:

**`nix shell nixpkgs#llama-cpp` does NOT give you CUDA.**

`cudaSupport` is a nixpkgs-level flag — it must be set when importing nixpkgs, not per-package. This means:

```nix
# This does NOT work
pkgs.llama-cpp.override { cudaSupport = true; }

# This DOES work — set at import time
pkgs = import nixpkgs { config.cudaSupport = true; config.allowUnfree = true; };
pkgs.llama-cpp  # now CUDA-enabled
```

This flake exposes a separate `cudaPkgs` nixpkgs instance with `cudaSupport = true`. The CUDA packages are under `.#llama-cpp-cuda`, `devShells.cuda`, and `apps.llama-server`.

**First build warning**: building llama-cpp with CUDA from source takes ~10 minutes. There's no official Nix binary cache for CUDA packages because CUDA is unfree. Subsequent runs use the Nix store cache.

---

## Flake outputs

```
packages.default         → ml-offload-api (Rust API server)
packages.api             → same as default
packages.forge           → tensorforge (Rust inference engine)
packages.python          → Python env for arch-analyzer
packages.llama-cpp-cuda  → llama.cpp with CUDA (x86_64-linux only, builds from source)

apps.default             → ml-offload-api
apps.forge               → tensorforge
apps.llama-server        → llama-server binary (CUDA-enabled)

devShells.default        → Rust + CUDA headers
devShells.python         → Python 3.13 + arch-analyzer deps
devShells.cuda           → Rust + llama-server + nvcc (full inference dev env)
```

---

## Quick start — Nix (non-NixOS)

```bash
# Install Nix first (Determinate Systems — flakes enabled by default)
curl -fsSL https://install.determinate.systems/nix | sh -s -- install

# Clone
git clone https://github.com/VoidNxSEC/ml-ops-api
cd ml-ops-api

# Drop into CUDA shell (builds llama.cpp with CUDA — ~10 min first run)
nix develop .#cuda

# Or use the bootstrap script
./tensorforge/scripts/bootstrap-nix.sh quick   # auto-detects GPU
```

---

## NixOS — production service

```bash
# Generate /etc/nixos/tensorforge.nix for your GPU
sudo ./tensorforge/scripts/bootstrap-nix.sh nixos --gpu b200
# profiles: b200 · h100 · l40s · generic

# Rebuild
sudo nixos-rebuild switch
systemctl status llamacpp-turbo
```

Or manually, add to your `configuration.nix`:

```nix
{ config, pkgs, ... }:
{
  imports = [ /path/to/ml-ops-api/tensorforge/nix/b200-optimized/default.nix ];

  # Required for CUDA packages
  nixpkgs.config.allowUnfree = true;
  hardware.nvidia.package = config.boot.kernelPackages.nvidiaPackages.stable;
  hardware.opengl.enable = true;

  services.tensorforge.b200Optimized = {
    enable             = true;
    llamacpp.enable    = true;
    systemTuning.enable = true;
    monitoring.enable  = true;
  };
}
```

### NixOS CUDA requirements

```nix
# hardware-configuration.nix or configuration.nix
hardware.opengl = {
  enable          = true;
  driSupport      = true;
  driSupport32Bit = true;
};

hardware.nvidia = {
  modesetting.enable = true;
  powerManagement.enable = false;
  open = false;
  nvidiaSettings = true;
  package = config.boot.kernelPackages.nvidiaPackages.stable;
};

services.xserver.videoDrivers = [ "nvidia" ];  # even on headless
```

---

## Dev workflow

```bash
# Default: Rust + CUDA headers
nix develop

# Full CUDA shell with llama-server
nix develop .#cuda
llama-server --version   # confirms CUDA build

# Python / arch-analyzer
nix develop .#python
python arch-analyzer/src/analyzer.py --help
```

---

## Why bare Linux scripts exist alongside the Nix path

The bare Linux scripts (`bootstrap.sh`, `server.sh`, etc.) exist because:

1. **Cloud GPU instances** (Shadeform, Lambda, RunPod) run Ubuntu, not NixOS. Installing Nix there and waiting for a CUDA build is overkill for a spot instance.
2. **NVIDIA NIM** containers have no Nix at all.
3. **Speed**: `bootstrap.sh` compiles llama.cpp in ~3 min on 12 cores. `nix build .#llama-cpp-cuda` takes ~10 min without a binary cache.

On NixOS workstations and persistent servers, the Nix path is the right choice. On ephemeral GPU nodes, the bash scripts are more practical.

---

## Binary cache note

There's no public binary cache for `.#llama-cpp-cuda` — CUDA is unfree and the package depends on your driver version. You can set up your own cache with:

```bash
# Attic or Cachix
nix build .#llama-cpp-cuda
cachix push your-cache result
```

Then add to `flake.nix`:
```nix
nixConfig.extra-substituters = [ "https://your-cache.cachix.org" ];
nixConfig.extra-trusted-public-keys = [ "your-cache.cachix.org-1:..." ];
```

---

**GPU arch reference** (for `CUDA_ARCH` in bare scripts):

| GPU | sm | `--cuda-arch` |
|-----|----|---------------|
| B200 / B300 | 100 | `--cuda-arch 100` |
| H100 / H200 | 90 | `--cuda-arch 90` |
| A100 | 80 | `--cuda-arch 80` |
| L40S / RTX 4090 | 89 | `--cuda-arch 89` |
| RTX 3090 | 86 | `--cuda-arch 86` |
| T4 | 75 | `--cuda-arch 75` |

---

**Maintained by**: VoidNxSEC — NVIDIA Inception (B200/B300)
