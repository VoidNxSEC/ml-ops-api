# ML-Ops API Roadmap

Active execution tracker for `ml-ops-api`.

`INFERENCE_PLATFORM.md` remains the product and architecture overview. This file is the repo-truth roadmap: what is actually shipped, what is partially shipped, and what we should do next.

**Last Updated**: 2026-04-30
**Mode**: verification-first

## Current validated state

### Checks run this turn

- `nix flake check --no-build` -> passed
- `nix develop --command cargo test --manifest-path api/Cargo.toml` -> passed
- Rust test count validated in this run: 51 passing
- `python3 -m compileall -q python/mlops python/tests` -> passed

### Validation gaps

- `python3 -m pytest -q` is not runnable on the host PATH because `pytest` is not installed globally.
- `nix develop .#python --command pytest -q python/tests` is currently blocked by sandbox access to the Nix daemon, so Python tests are not yet validated end-to-end in this turn.

## What is shipped now

### API and control surface

- OpenAI-compatible inference routes exist in `api/`:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/embeddings`
- Operational routes exist for:
  - `/metrics`
  - `/health`
  - `/api/health`
  - `/api/backend/info`
  - `/api/stats`
  - `/status`
  - `/vram`
  - `/ws`
  - `/backends`
  - `/models`
  - `/load`
  - `/unload`
  - `/switch`
- Auth and per-key rate limiting are implemented in `api/src/auth.rs`.
- A priority queue orchestrator and backend scoring router are implemented in `api/src/orchestrator.rs` and `api/src/router.rs`.
- Prometheus metrics are wired in `api/src/metrics.rs` and installed in `api/src/main.rs`.

### Packaging and runtime

- The root flake exports packages for:
  - `api`
  - `forge`
  - `python`
- Dev shells exist for:
  - default Rust development
  - Python development
  - CUDA-oriented work
- `tensorforge/scripts/` contains a real local pipeline surface:
  - `entrypoint.sh`
  - `bootstrap.sh`
  - `server.sh`
  - `model-pull.sh`
  - `infer.sh`
  - `run.sh`
  - `export.sh`
  - `health.sh`

### Python support

- The Python tree contains a fallback-capable LLM client, provider abstractions, optimization helpers, reasoning helpers, and test files.
- The Python reasoning layer is still explicitly MVP-level in parts:
  - `python/mlops/reasoning/dspy_adapter.py`
  - `python/mlops/reasoning/ensemble_reasoning.py`

## Known truth gaps and drift

### High-priority drift

- `python/mlops/llm/providers/ml_offload.py` is still a stub around a notional gRPC service on `localhost:50051`, while the repo's shipped inference service is the Axum HTTP API with `/v1/*` endpoints.
- `api/src/lib.rs` duplicates handler logic from `api/src/main.rs` and even documents that it was copied "for now". This is a drift risk for tests, routes, and future middleware changes.
- `api/src/api.rs` still describes auth, rate limiting, and metrics as future TODO work even though those features already exist elsewhere in `api/`.

### Documentation drift

- The default runtime config in code is `127.0.0.1:9000` unless overridden by env.
- `README.md` is primarily written from the compose/container point of view and emphasizes `8080` and `8083`.
- `INFERENCE_PLATFORM.md` mixes real repo structure with aspirational platform positioning, so it should not be treated as the delivery tracker anymore.
- `tensorforge/README.md` reads as a technical showcase and contains benchmark/platform claims that are broader than what this repo validates in its current test surface.

## Roadmap

### Phase 1 - Truth alignment
**Status**: in progress

Goal: make docs, defaults, and execution modes unambiguous.

Exit criteria:

- `README.md`, `INFERENCE_PLATFORM.md`, and `ROADMAP.md` clearly distinguish:
  - local defaults
  - compose deployment values
  - aspirational architecture
  - validated shipped features
- Auth, rate-limit, backend, and runtime env vars are documented from the real code paths.

Work items:

1. Document the real default bind and port from `api/src/main.rs` and `api/src/lib.rs`.
2. Separate local runtime instructions from compose runtime instructions.
3. Mark unvalidated benchmark and fleet claims as targets, not current guarantees.

### Phase 2 - API consolidation
**Status**: queued

Goal: converge the Rust API surface into one maintainable routing and handler layer.

Exit criteria:

- `main.rs` and `lib.rs` share one source of truth for handlers and router construction.
- The placeholder `api/src/api.rs` is either removed or turned into the real extension point.
- Route behavior remains covered by integration tests after consolidation.

Work items:

1. Remove duplicated handler definitions between `main.rs` and `lib.rs`.
2. Keep router assembly in one place and reuse it from both runtime and tests.
3. Add or extend tests around `/metrics` and `/ws` behavior once the shared surface is settled.

### Phase 3 - Python integration completion
**Status**: queued

Goal: make the Python layer a real client of this repo instead of a migrated placeholder.

Exit criteria:

- `MLOffloadProvider` speaks to the actual shipped inference API.
- Local-first behavior prefers this repo's inference service before remote providers when configured to do so.
- Python tests are runnable with a single repo-native command.

Work items:

1. Replace the stubbed gRPC-shaped `MLOffloadProvider` with a real HTTP client against `/v1/chat/completions` and `/v1/embeddings`.
2. Align default URLs, models, and provider names with the Rust service surface.
3. Add a documented validation command for Python tests via flake/dev shell or another reproducible repo-native path.

### Phase 4 - Backend and event-path validation
**Status**: queued

Goal: prove real runtime behavior across backends and integrations.

Exit criteria:

- Local smoke tests exist for llama.cpp and optional vLLM backends.
- Router selection logic is validated against live or mocked backend pressure.
- Best-effort NATS publishing behavior is documented and tested at the contract level where practical.

Work items:

1. Add smoke coverage for LLAMACPP and VLLM runtime paths.
2. Validate no-backend and degraded-backend behavior as explicit supported modes.
3. Decide whether WebSocket and NATS streams need dedicated contract tests or documented best-effort semantics only.

### Phase 5 - CI and release discipline
**Status**: queued

Goal: make CI reflect the same truth we use locally.

Exit criteria:

- CI runs the same Rust validation path used in this roadmap.
- Python validation is either enabled in CI or explicitly deferred with a documented reason.
- Docs updates stop drifting from the real runtime defaults and route surface.

Work items:

1. Standardize on repo-native validation commands.
2. Keep CI focused on shipped surfaces before expanding workflow breadth.
3. Add release-facing checks only after Phases 1-4 are stable.

## Immediate next slice

1. Align docs with real ports, env vars, and runtime modes.
2. Refactor `api` so `main.rs` and `lib.rs` stop duplicating handlers.
3. Implement the real Python `MLOffloadProvider` against the shipped HTTP API.
