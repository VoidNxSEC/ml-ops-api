{
  description = "TensorForge - High-performance ML inference orchestration engine for massive-scale pipelines";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true; # For CUDA
          cudaSupport = true;
        };

        # Rust toolchain with nightly for some features if needed
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
            "clippy"
            "rustfmt"
          ];
          targets = [ "x86_64-unknown-linux-gnu" ];
        };

        # Python environment with DSPy and ML dependencies
        pythonEnv = pkgs.python313.withPackages (
          ps: with ps; [
            # Core ML and inference
            torch
            torchvision
            transformers
            accelerate
            vllm
            llama-cpp-python

            # DSPy framework
            dspy

            # Data processing
            pandas
            numpy
            scipy
            scikit-learn

            # Web and API
            fastapi
            uvicorn
            pydantic
            httpx
            websockets

            # Monitoring and metrics
            prometheus-client
            grafana-client
            psutil
            nvidia-ml-py

            # Development and testing
            pytest
            pytest-asyncio
            black
            mypy
            types-requests

            # Utilities
            tqdm
            rich
            python-dotenv
            pyyaml
            toml
          ]
        );

        # GPU support configuration
        enableGpu = true;

        gpuBuildInputs = pkgs.lib.optionals enableGpu (
          if system == "x86_64-linux" then
            [
              pkgs.cudaPackages.cuda_nvcc
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.libcublas
              pkgs.cudaPackages.libcurand
              pkgs.cudaPackages.libcusparse
              pkgs.cudaPackages.libcusolver
              pkgs.cudaPackages.libcufft
              pkgs.cudaPackages.libcudnn
              pkgs.linuxPackages.nvidia_x11
            ]
          else
            [ ]
        );

        # Rust crate build
        tensorforgeRust = pkgs.rustPlatform.buildRustPackage {
          pname = "tensorforge-core";
          version = "0.1.0";
          src = ./core;

          cargoLock = {
            lockFile = ./core/Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [
            pkg-config
            cmake
            rustToolchain
          ];

          buildInputs =
            with pkgs;
            [
              openssl
              sqlite
              zlib
              libiconv
            ]
            ++ gpuBuildInputs;

          # Set CUDA paths for nvml-wrapper
          CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          CUDA_INCLUDE_PATH = "${pkgs.cudaPackages.cudatoolkit}/include";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath gpuBuildInputs;

          buildPhase = ''
            export RUST_BACKTRACE=1
            ${
              if enableGpu then
                "cargo build --release --features gpu,vram-monitoring,backend-vllm,backend-llamacpp"
              else
                "cargo build --release --no-default-features"
            }
          '';

          installPhase = ''
            mkdir -p $out/bin $out/lib

            # Install binary
            cp target/release/tensorforge $out/bin/

            # Install library if needed
            cp target/release/libtensorforge*.so $out/lib/ 2>/dev/null || true
            cp target/release/libtensorforge*.a $out/lib/ 2>/dev/null || true

            # Install assets and configuration examples
            mkdir -p $out/share/tensorforge
            cp -r ../examples $out/share/tensorforge/
            cp -r ../config $out/share/tensorforge/
            cp ../README.md $out/share/tensorforge/

            # Create wrapper script with environment setup
            cat > $out/bin/tensorforge-wrapper <<EOF
            #!/bin/sh
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath gpuBuildInputs}:\$LD_LIBRARY_PATH"
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            exec $out/bin/tensorforge "\$@"
            EOF
            chmod +x $out/bin/tensorforge-wrapper
          '';

          checkPhase = ''
            cargo test --release --no-fail-fast
          '';

          meta = with pkgs.lib; {
            description = "TensorForge core Rust engine for ML inference orchestration";
            license = licenses.mit;
            maintainers = [ "kernelcore" ];
            platforms = platforms.linux;
          };
        };

        # Python DSPy integration package
        tensorforgePython = pkgs.stdenv.mkDerivation {
          pname = "tensorforge-dspy";
          version = "0.1.0";
          src = ./dspy;

          buildInputs = [ pythonEnv ];

          installPhase = ''
            mkdir -p $out/lib/tensorforge-dspy $out/bin

            # Copy Python source
            cp -r . $out/lib/tensorforge-dspy/

            # Create entry points
            cat > $out/bin/tensorforge-dspy <<EOF
            #!/bin/sh
            export PYTHONPATH="$out/lib/tensorforge-dspy:\${PYTHONPATH}"
            exec ${pythonEnv}/bin/python -m tensorforge.dspy "\$@"
            EOF
            chmod +x $out/bin/tensorforge-dspy

            # Create Python module
            cat > $out/lib/tensorforge-dspy/__init__.py <<EOF
            from .backend import TensorForgeLM
            from .compiler import TensorForgeCompiler
            from .metrics import TensorForgeMetrics

            __version__ = "0.1.0"
            __all__ = ["TensorForgeLM", "TensorForgeCompiler", "TensorForgeMetrics"]
            EOF
          '';

          meta = with pkgs.lib; {
            description = "TensorForge DSPy integration for LM pipeline optimization";
            license = licenses.mit;
            maintainers = [ "kernelcore" ];
          };
        };

        # B200-optimized configuration package
        b200Config = pkgs.stdenv.mkDerivation {
          pname = "tensorforge-b200-config";
          version = "0.1.0";
          src = ./nix/b200-optimized;

          installPhase = ''
            mkdir -p $out/etc/tensorforge
            cp *.nix $out/etc/tensorforge/
            cp *.toml $out/etc/tensorforge/ 2>/dev/null || true
            cp *.json $out/etc/tensorforge/ 2>/dev/null || true

            # Create README
            cat > $out/etc/tensorforge/README.md <<EOF
            # TensorForge B200 Optimized Configuration

            This directory contains NVIDIA B200-optimized configurations for TensorForge.

            Files:
            - default.nix: NixOS module configuration
            - hardware.nix: B200-specific hardware optimizations
            - vllm-optimized.nix: vLLM backend optimizations for B200
            - llamacpp-optimized.nix: llama.cpp backend optimizations for B200

            To use these configurations, import them in your NixOS configuration:

            \`\`\`nix
            { config, lib, pkgs, ... }:
            {
              imports = [
                ${"$"}{self}/nix/b200-optimized/default.nix
              ];
            }
            \`\`\`
            EOF
          '';
        };

        # Combined package
        tensorforgeAll = pkgs.symlinkJoin {
          name = "tensorforge-all";
          paths = [
            tensorforgeRust
            tensorforgePython
            b200Config
          ];
          postBuild = ''
            # Create convenience scripts
            cat > $out/bin/tf-start <<EOF
            #!/bin/sh
            echo "Starting TensorForge with B200 optimizations..."
            export TENSORFORGE_CONFIG_DIR="$out/etc/tensorforge"
              $out/bin/tensorforge-wrapper serve \
              --host 0.0.0.0 \
              --port 8080 \
              --metrics \
              --websocket \
              "\$@"
            EOF
            chmod +x $out/bin/tf-start

            cat > $out/bin/tf-batch <<EOF
            #!/bin/sh
            $out/bin/tensorforge-wrapper batch \
              --parallel 8 \
              --checkpoint \
              --checkpoint-interval 1000 \
              "\$@"
            EOF
            chmod +x $out/bin/tf-batch

            cat > $out/bin/tf-monitor <<EOF
            #!/bin/sh
            $out/bin/tensorforge-wrapper stats \
              --refresh 5 \
              --metrics all \
              --format text
            EOF
            chmod +x $out/bin/tf-monitor
          '';
        };

      in
      {
        # Packages
        packages = {
          default = tensorforgeAll;
          rust = tensorforgeRust;
          python = tensorforgePython;
          b200-config = b200Config;
          all = tensorforgeAll;
        };

        # Apps
        apps = {
          default = {
            type = "app";
            program = "${tensorforgeRust}/bin/tensorforge-wrapper";
          };

          serve = {
            type = "app";
            program = "${tensorforgeRust}/bin/tensorforge-wrapper";
          };

          cli = {
            type = "app";
            program = "${tensorforgeRust}/bin/tensorforge-wrapper";
          };

          batch = {
            type = "app";
            program = "${tensorforgeRust}/bin/tensorforge-wrapper";
          };

          dspy = {
            type = "app";
            program = "${tensorforgePython}/bin/tensorforge-dspy";
          };
        };

        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs =
            with pkgs;
            [
              # Rust toolchain
              rustToolchain
              cargo-watch
              cargo-edit
              cargo-audit
              cargo-udeps
              cargo-nextest
              rust-analyzer

              # Python environment
              pythonEnv
              python313Packages.pip
              python313Packages.ipython
              python313Packages.jupyter

              # Build tools
              pkg-config
              cmake
              gcc
              clang
              llvmPackages.libclang

              # Database and storage
              sqlite
              postgresql
              redis

              # Monitoring and debugging
              htop
              nvtop
              nvidia-smi
              prometheus
              grafana
              curl
              jq
              yq

              # Version control and dev tools
              git
              gh
              ripgrep
              fd
              fzf
              tree
              bat
              exa
              du-dust
              hyperfine
            ]
            ++ gpuBuildInputs;

          shellHook = ''
            echo "🚀 TensorForge Development Environment"
            echo "====================================="
            echo "  Rust: $(rustc --version | cut -d' ' -f2)"
            echo "  Python: $(python --version)"
            echo "  CUDA: ${if enableGpu then "✅ Enabled" else "❌ Disabled"}"
            echo ""
            echo "📦 Available Packages:"
            echo "  tensorforge-rust   - Rust core engine"
            echo "  tensorforge-python - DSPy integration"
            echo "  tensorforge-all    - Combined package"
            echo ""
            echo "🛠️  Development Commands:"
            echo "  cargo build --release  - Build Rust core"
            echo "  cargo test --all       - Run all tests"
            echo "  cargo bench           - Run benchmarks"
            echo "  python -m pytest      - Run Python tests"
            echo ""
            echo "🚀 Production Commands:"
            echo "  nix build .#rust      - Build Rust package"
            echo "  nix build .#python    - Build Python package"
            echo "  nix build .#all       - Build everything"
            echo "  nix run .#serve       - Start API server"
            echo "  nix run .#batch       - Run batch processing"
            echo ""

            if ${if enableGpu then "true" else "false"}; then
              echo "🎮 GPU Information:"
              if command -v nvidia-smi &> /dev/null; then
                nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
              else
                echo "  nvidia-smi not available"
              fi
            fi

            # Set environment variables
            export RUST_BACKTRACE=1
            export RUST_LOG=debug
            export TENSORFORGE_DATA_DIR="$PWD/data"
            export TENSORFORGE_MODELS_DIR="$PWD/models"
            export TENSORFORGE_LOG_DIR="$PWD/logs"
            export TENSORFORGE_CONFIG_DIR="$PWD/config"

            # Create directories if they don't exist
            mkdir -p $TENSORFORGE_DATA_DIR $TENSORFORGE_MODELS_DIR $TENSORFORGE_LOG_DIR $TENSORFORGE_CONFIG_DIR

            # Set up Python path
            export PYTHONPATH="$PWD/dspy:$PYTHONPATH"

            # CUDA paths
            ${
              if enableGpu then
                ''
                  export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
                  export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath gpuBuildInputs}:$LD_LIBRARY_PATH"
                ''
              else
                ""
            }

            echo ""
            echo "📁 Project directories created in:"
            echo "  Data: $TENSORFORGE_DATA_DIR"
            echo "  Models: $TENSORFORGE_MODELS_DIR"
            echo "  Logs: $TENSORFORGE_LOG_DIR"
            echo "  Configs: $TENSORFORGE_CONFIG_DIR"
            echo ""
            echo "💡 Tip: Copy example configs from nix/b200-optimized/ to config/"
          '';

          # Environment variables for development
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        };

        # Checks
        checks = {
          rust-build = tensorforgeRust;
          python-build = tensorforgePython;
          b200-config-build = b200Config;
        };
      }
    );
}
