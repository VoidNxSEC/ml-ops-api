# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM rust:1.82-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache dependencies before copying source
COPY api/Cargo.toml api/Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

# Build the real binary
COPY api/src ./src
RUN touch src/main.rs && cargo build --release

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libsqlite3-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false -d /var/lib/ml-offload mloffload \
    && mkdir -p /var/lib/ml-offload /var/lib/ml-models \
    && chown mloffload:mloffload /var/lib/ml-offload /var/lib/ml-models

COPY --from=builder /build/target/release/ml-offload-api /usr/local/bin/ml-offload-api

USER mloffload

ENV ML_OFFLOAD_HOST=0.0.0.0 \
    ML_OFFLOAD_PORT=8080 \
    ML_OFFLOAD_DATA_DIR=/var/lib/ml-offload \
    ML_OFFLOAD_MODELS_PATH=/var/lib/ml-models \
    ML_OFFLOAD_DB_PATH=/var/lib/ml-offload/registry.db \
    ML_OFFLOAD_CORS_ENABLED=false \
    NATS_URL=nats://nats:4222 \
    RUST_LOG=ml_offload_api=info,axum=info,tower_http=warn

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --retries=6 \
    CMD wget -q --spider http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/ml-offload-api"]
