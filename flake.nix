{
  description = "ML Offload Ecosystem - API & TensorForge Orchestrator";

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
          config.allowUnfree = true;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
          ];
        };

        commonNativeBuildInputs = with pkgs; [
          pkg-config
          rustToolchain
        ];

        commonBuildInputs =
          with pkgs;
          [
            openssl
            sqlite
            zlib
          ]
          ++ pkgs.lib.optionals (system == "x86_64-linux") [
            pkgs.cudaPackages.cuda_nvcc
            pkgs.cudaPackages.cudatoolkit
          ];

        # Pacote 1: API Server (Leve)
        mlOffloadApi = pkgs.rustPlatform.buildRustPackage {
          pname = "ml-offload-api";
          version = "0.1.0";
          src = ./api;
          cargoLock.lockFile = ./api/Cargo.lock;

          nativeBuildInputs = commonNativeBuildInputs;
          buildInputs = commonBuildInputs;

          # NVML wrapper requer libnvidia-ml.so em runtime, geralmente provido pelo driver do host
          # NixOS requer configuração especial via hardware.opengl
        };

        # Pacote 2: TensorForge (Engine)
        tensorForge = pkgs.rustPlatform.buildRustPackage {
          pname = "tensorforge";
          version = "0.1.0";
          src = ./tensorforge;

          # TensorForge usa workspace, lockfile na raiz do subprojeto
          cargoLock.lockFile = ./tensorforge/Cargo.lock;

          nativeBuildInputs = commonNativeBuildInputs;
          buildInputs = commonBuildInputs;

          # Ignorar testes que requerem GPU real durante o build
          doCheck = false;
        };

        # Python MLOps environment
        python = pkgs.python313;

        mlopsEnv = python.withPackages (ps: with ps; [
          mlflow
          pyyaml
          grpcio
          # Optional heavy deps — comment out if not needed locally
          # temporalio
          # dspy
          # anthropic
          # openai
          # tenacity
          pytest
          pytest-asyncio
        ]);

      in
      {
        packages = {
          default = mlOffloadApi;
          api = mlOffloadApi;
          forge = tensorForge;
        };

        apps = {
          default = flake-utils.lib.mkApp { drv = mlOffloadApi; };
          forge = flake-utils.lib.mkApp { drv = tensorForge; };
        };

        devShells = {
          default = pkgs.mkShell {
            buildInputs =
              commonBuildInputs
              ++ (with pkgs; [
                # Ferramentas de Dev
                cargo-watch
                cargo-edit
                bacon
                jq

                # Utilitários de Sistema
                pciutils # lspci
                hwloc
              ]);

            nativeBuildInputs = commonNativeBuildInputs;

            # Variáveis de ambiente para desenvolvimento
            ML_OFFLOAD_DB_PATH = "./dev.db";
            RUST_LOG = "info";

            shellHook = ''
              echo "ML Offload Ecosystem Environment"
              echo "   Components: API (axum) + TensorForge (engine)"
              echo "   Rust: $(rustc --version)"
            '';
          };

          # Shell Python MLOps — para desenvolvimento da camada mlops/
          mlops = pkgs.mkShell {
            buildInputs = [ mlopsEnv pkgs.python313 ];

            shellHook = ''
              echo "MLOps Python Environment"
              echo "   Python: $(python3.13 --version)"
              echo "   Pacote: python/mlops/"
              echo "   Testes: cd python && python3.13 -m pytest tests/ -v"
              export PYTHONPATH="$PWD/python:$PYTHONPATH"
            '';
          };
        };
      }
    );
}
