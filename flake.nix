{
  description = "tensorforge — GPU inference pipeline for the voidnxlabs platform";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # ── Standard pkgs ──────────────────────────────────────────────────────
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
          config.allowUnfree = true;
        };

        # ── CUDA-enabled pkgs (x86_64-linux only) ──────────────────────────────
        # cudaSupport must be set at import time — it cannot be overridden per-package
        cudaPkgs = if system == "x86_64-linux" then
          import nixpkgs {
            inherit system;
            overlays = [ (import rust-overlay) ];
            config = {
              allowUnfree   = true;
              cudaSupport   = true;
            };
          }
        else pkgs;

        # ── Rust toolchain ─────────────────────────────────────────────────────
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        commonNativeBuildInputs = with pkgs; [ pkg-config rustToolchain ];

        commonBuildInputs = with pkgs; [ openssl sqlite zlib ]
          ++ pkgs.lib.optionals (system == "x86_64-linux") (with pkgs.cudaPackages; [
            cuda_nvcc
            cudatoolkit
          ]);

        # ── Rust packages ──────────────────────────────────────────────────────
        mlOffloadApi = pkgs.rustPlatform.buildRustPackage {
          pname   = "ml-offload-api";
          version = "0.1.0";
          src     = ./api;
          cargoLock.lockFile = ./api/Cargo.lock;
          nativeBuildInputs  = commonNativeBuildInputs;
          buildInputs        = commonBuildInputs;
        };

        tensorForge = pkgs.rustPlatform.buildRustPackage {
          pname   = "tensorforge";
          version = "0.1.0";
          src     = ./tensorforge;
          cargoLock.lockFile = ./tensorforge/Cargo.lock;
          nativeBuildInputs  = commonNativeBuildInputs;
          buildInputs        = commonBuildInputs;
          doCheck = false;  # GPU tests require real hardware
        };

        # ── Python env (arch-analyzer) ─────────────────────────────────────────
        pythonEnv = pkgs.python313.withPackages (ps: with ps; [
          aiohttp aiofiles pydantic rich typer httpx pyyaml psutil
          pytest pytest-asyncio
        ]);

      in {
        # ── Packages ───────────────────────────────────────────────────────────
        packages = {
          default      = mlOffloadApi;
          api          = mlOffloadApi;
          forge        = tensorForge;
          python       = pythonEnv;

          # llama.cpp with CUDA (x86_64-linux only)
          # Build time is significant (~10 min) — uses CUDA pkgs overlay
          llama-cpp-cuda = cudaPkgs.llama-cpp;
        };

        # ── Apps ───────────────────────────────────────────────────────────────
        apps = {
          default = flake-utils.lib.mkApp { drv = mlOffloadApi; };
          forge   = flake-utils.lib.mkApp { drv = tensorForge; };

          # Quick llama-server with CUDA
          llama-server = flake-utils.lib.mkApp {
            drv     = cudaPkgs.llama-cpp;
            exePath = "/bin/llama-server";
          };
        };

        # ── Dev shells ─────────────────────────────────────────────────────────
        devShells = {
          # Main dev shell: Rust + CUDA headers
          default = pkgs.mkShell {
            buildInputs = commonBuildInputs ++ (with pkgs; [
              cargo-watch cargo-edit bacon jq pciutils hwloc just
            ]);
            nativeBuildInputs = commonNativeBuildInputs;
            ML_OFFLOAD_DB_PATH = "./dev.db";
            RUST_LOG           = "info";
            shellHook = ''
              echo "👋 **Welcome to the ML Offload API & TensorForge Workspace!**"
              echo ""
              echo "**🔧 Quick Project Commands:**"
              echo "- /just dev: Enter the Nix development shell (Rust + CUDA)"
              echo "- /just dev-python: Enter the Python Nix shell"
              echo "- /just setup-linux: Bare Linux setup"
              echo "- /just quick-nix: Quick Nix dev shell"
              echo ""
              echo "**🤖 Gemini CLI Commands:**"
              echo "- /help: View all available commands"
              echo "- /memory show: See current session context"
              echo "- /clear: Clear the session"
              echo ""
            '';
          };

          # Python / arch-analyzer shell
          python = pkgs.mkShell {
            buildInputs = [ pythonEnv pkgs.python313 ];
            shellHook = ''
              echo "  arch-analyzer Python shell"
              echo "  python: $(python3 --version)"
              export PYTHONPATH="$PWD/arch-analyzer:$PYTHONPATH"
            '';
          };

          # Full CUDA shell: Rust + llama.cpp + CUDA tools
          # Note: first build downloads/compiles llama.cpp with CUDA (~10 min)
          cuda = cudaPkgs.mkShell {
            buildInputs = [
              cudaPkgs.llama-cpp
              cudaPkgs.cudaPackages.cuda_nvcc
              cudaPkgs.cudaPackages.cudatoolkit
              pkgs.jq pkgs.curl
              rustToolchain
            ];
            shellHook = ''
              echo ""
              echo "  tensorforge CUDA shell"
              echo "  llama-server: $(llama-server --version 2>&1 | head -1)"
              echo "  nvcc:         $(nvcc --version 2>&1 | grep release)"
              echo ""
              echo "  Start server:  llama-server --model /path/to/model.gguf -ngl 999 --flash-attn"
              echo ""
            '';
          };
        };
      }
    );
}
