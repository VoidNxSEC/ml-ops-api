# ML Offload API

> **Unified multi-backend ML model orchestration system**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built with Nix](https://img.shields.io/badge/Built_With-Nix-5277C3.svg?logo=nixos&logoColor=white)](https://nixos.org)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg?logo=rust)](https://www.rust-lang.org/)

## Overview

ML Offload API is a sophisticated orchestration layer that provides unified access to multiple ML inference backends (Ollama, llama.cpp, vLLM, TGI) with intelligent VRAM management and automatic backend selection.

### Key Features

- 🎯 **Multi-Backend Orchestration** - Unified API for Ollama, llama.cpp, vLLM, TGI
- 🧠 **Intelligent Routing** - Automatic backend selection based on VRAM, load, and model availability  
- 📊 **VRAM Monitoring** - Real-time GPU memory tracking via NVIDIA NVML
- 🗃️ **Model Registry** - SQLite-based model discovery and management
- 🚀 **High Performance** - Built in Rust with async/await (Tokio + Axum)
- 🔌 **WebSocket Support** - Real-time streaming inference
- 📦 **Nix Flake** - Reproducible builds and development environments

## Quick Start

\`\`\`bash
# Development shell
nix develop

# Build
nix build

# Run
nix run
\`\`\`

See full documentation in the original README.
