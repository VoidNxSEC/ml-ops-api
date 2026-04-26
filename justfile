set shell := ["bash", "-c"]

# Enter the default Nix development shell (Rust + CUDA)
dev:
	nix develop

# Enter the Python/arch-analyzer Nix development shell
dev-python:
	nix develop .#python

# Setup Bare Linux (Ubuntu/Debian)
setup-linux:
	./scripts/entrypoint.sh setup

# Quick Nix dev shell without installing
quick-nix:
	./scripts/bootstrap-nix.sh quick
