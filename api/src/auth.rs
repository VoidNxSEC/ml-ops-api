/// Auth + Rate-limiting middleware
///
/// Auth: validates `X-API-Key` header against a set of configured keys.
///   - If ML_OFFLOAD_API_KEYS is unset or empty → dev mode, all requests pass.
///   - Otherwise → 401 on missing/invalid key.
///
/// Rate limiting: per-key GCRA token bucket via `governor`.
///   - Limit: ML_OFFLOAD_RATE_LIMIT_RPM requests/minute (default 60).
///   - Key: the API key string, or "anonymous" in dev mode.
///   - Exceeding quota → 429 with Retry-After header.

use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use governor::{clock::DefaultClock, state::keyed::DefaultKeyedStateStore, Quota, RateLimiter};
use tracing::warn;

// ── Types ─────────────────────────────────────────────────────────────────────

pub type KeyedLimiter = RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock>;

// ── Construction ──────────────────────────────────────────────────────────────

/// Parse `ML_OFFLOAD_API_KEYS` (comma-separated) into a set.
/// Empty set = dev mode (no auth).
pub fn api_keys_from_env() -> Arc<HashSet<String>> {
    let keys: HashSet<String> = std::env::var("ML_OFFLOAD_API_KEYS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    Arc::new(keys)
}

/// Build a keyed GCRA rate limiter from `ML_OFFLOAD_RATE_LIMIT_RPM` (default 60).
pub fn rate_limiter_from_env() -> Arc<KeyedLimiter> {
    let rpm: u32 = std::env::var("ML_OFFLOAD_RATE_LIMIT_RPM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);

    let quota = Quota::per_minute(NonZeroU32::new(rpm.max(1)).unwrap());
    Arc::new(RateLimiter::keyed(quota))
}

// ── Auth middleware ───────────────────────────────────────────────────────────

/// Validates `X-API-Key`. Pass-through when no keys are configured.
pub async fn auth_middleware(
    axum::extract::State(state): axum::extract::State<crate::AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let keys = &state.api_keys;

    // Dev mode — no keys configured
    if keys.is_empty() {
        return next.run(request).await;
    }

    let provided = request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if keys.contains(provided) {
        next.run(request).await
    } else {
        warn!(key_prefix = &provided[..provided.len().min(8)], "rejected: invalid api key");
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "message": "Invalid or missing X-API-Key",
                    "type": "authentication_error"
                }
            })),
        )
            .into_response()
    }
}

// ── Rate-limit middleware ─────────────────────────────────────────────────────

/// Per-key GCRA rate limiter. Key = API key value, or "anonymous" in dev mode.
pub async fn rate_limit_middleware(
    axum::extract::State(state): axum::extract::State<crate::AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let key = request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("anonymous")
        .to_string();

    match state.rate_limiter.check_key(&key) {
        Ok(_) => next.run(request).await,
        Err(not_until) => {
            let wait_secs = not_until.wait_time_from(governor::clock::Clock::now(
                &governor::clock::DefaultClock::default(),
            ));
            warn!(key_prefix = &key[..key.len().min(8)], "rate limited");
            let mut resp = (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "retry_after_secs": wait_secs.as_secs()
                    }
                })),
            )
                .into_response();
            resp.headers_mut().insert(
                "retry-after",
                HeaderValue::from_str(&wait_secs.as_secs().to_string())
                    .unwrap_or(HeaderValue::from_static("60")),
            );
            resp
        }
    }
}
