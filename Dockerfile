# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Pin to same major as the Nix flake (flake.nix: rustVersion = "1.xx").
# Update when the flake version bumps.
FROM rust:1.82-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ── Dependency cache layer ────────────────────────────────────────────────────
# Cargo only re-downloads crates when Cargo.lock changes.
# Both stubs must exist: main.rs (binary) + lib.rs (library target inferred
# by Cargo because api/src/lib.rs exists in the real tree).
COPY api/Cargo.toml api/Cargo.lock ./
RUN mkdir -p src \
    && echo "fn main() {}" > src/main.rs \
    && echo "" > src/lib.rs \
    && cargo build --release \
    && rm -rf src

# ── Real build ────────────────────────────────────────────────────────────────
COPY api/src ./src
# Touch to force re-link (timestamps didn't change for cargo to notice)
RUN touch src/main.rs src/lib.rs \
    && cargo build --release

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libsqlite3-0 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false -d /var/lib/ml-offload mloffload \
    && mkdir -p /var/lib/ml-offload /var/lib/ml-models \
    && chown mloffload:mloffload /var/lib/ml-offload /var/lib/ml-models

COPY --from=builder /build/target/release/ml-offload-api /usr/local/bin/ml-offload-api

USER mloffload

# ── Defaults (all overridable at runtime) ─────────────────────────────────────
ENV ML_OFFLOAD_HOST=0.0.0.0 \
    ML_OFFLOAD_PORT=9000 \
    ML_OFFLOAD_DATA_DIR=/var/lib/ml-offload \
    ML_OFFLOAD_MODELS_PATH=/var/lib/ml-models \
    ML_OFFLOAD_DB_PATH=/var/lib/ml-offload/registry.db \
    ML_OFFLOAD_CORS_ENABLED=false \
    LLAMACPP_URL=http://llama-server:8080 \
    ORCHESTRATOR_WORKERS=4 \
    ORCHESTRATOR_MAX_CONCURRENT=8 \
    ORCHESTRATOR_TIMEOUT_SECS=300 \
    CIRCUIT_BREAKER_THRESHOLD=3 \
    CIRCUIT_BREAKER_COOLDOWN_SECS=30 \
    ML_OFFLOAD_RATE_LIMIT_RPM=60 \
    NATS_URL=nats://nats:4222 \
    RUST_LOG=ml_offload_api=info,axum=info,tower_http=warn

EXPOSE 9000

VOLUME ["/var/lib/ml-offload", "/var/lib/ml-models"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=6 \
    CMD curl -sf http://localhost:9000/health || exit 1

ENTRYPOINT ["/usr/local/bin/ml-offload-api"]
