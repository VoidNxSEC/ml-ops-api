use axum::http::StatusCode;
use axum_test::TestServer;
use governor::{Quota, RateLimiter};
use ml_offload_api::{auth, create_router, AppState, Config};
use pretty_assertions::assert_eq;
use serde_json::Value;
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Dev-mode server: no auth, no rate limit pressure.
async fn dev_server() -> TestServer {
    let config = Arc::new(Config::test_config());
    let state = AppState::new_for_test(config).await;
    TestServer::new(create_router(state)).unwrap()
}

/// Server with a specific set of valid API keys.
async fn auth_server(keys: &[&str]) -> TestServer {
    let config = Arc::new(Config::test_config());
    let mut state = AppState::new_for_test(config).await;
    let set: HashSet<String> = keys.iter().map(|k| k.to_string()).collect();
    state.api_keys = Arc::new(set);
    TestServer::new(create_router(state)).unwrap()
}

/// Server with 1 RPM rate limit and a single valid key.
async fn rate_limit_server() -> TestServer {
    let config = Arc::new(Config::test_config());
    let mut state = AppState::new_for_test(config).await;
    let quota = Quota::per_minute(NonZeroU32::new(1).unwrap());
    state.rate_limiter = Arc::new(RateLimiter::keyed(quota));
    TestServer::new(create_router(state)).unwrap()
}

// ── Public routes ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_root_endpoint() {
    let server = dev_server().await;
    let resp = server.get("/").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert_eq!(json["name"], "ML Offload Manager API");
    assert!(json["endpoints"].is_object());
}

#[tokio::test]
async fn test_health_endpoint() {
    let server = dev_server().await;
    let resp = server.get("/health").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json["status"].is_string());
    assert!(json["services"].is_object());
    assert_eq!(json["version"], "0.1.0");
}

#[tokio::test]
async fn test_status_endpoint() {
    let server = dev_server().await;
    let resp = server.get("/status").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json["timestamp"].is_string());
    assert!(json["vram"].is_object());
    assert!(json["backends"].is_array());
}

#[tokio::test]
async fn test_vram_endpoint() {
    let server = dev_server().await;
    let resp = server.get("/vram").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json["total_gb"].is_f64() || json["total_gb"].is_number());
    assert!(json["used_gb"].is_number());
    assert!(json["free_gb"].is_number());
}

#[tokio::test]
async fn test_list_backends_endpoint() {
    let server = dev_server().await;
    let resp = server.get("/backends").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json.is_array());
}

#[tokio::test]
async fn test_list_models_endpoint() {
    let server = dev_server().await;
    let resp = server.get("/models").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json.is_array());
}

#[tokio::test]
async fn test_unknown_route_returns_404() {
    let server = dev_server().await;
    let resp = server.get("/does-not-exist").await;
    resp.assert_status_not_found();
}

// ── /api/stats ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_stats_shape() {
    let server = dev_server().await;
    let resp = server.get("/api/stats").await;
    resp.assert_status_ok();
    let json: Value = resp.json();

    assert!(json["timestamp"].is_string());

    let orch = &json["orchestrator"];
    assert!(orch["in_flight"].is_number());
    let queue = &orch["queue"];
    assert!(queue["low"].is_number());
    assert!(queue["normal"].is_number());
    assert!(queue["high"].is_number());
    assert!(queue["critical"].is_number());
    assert!(queue["total"].is_number());

    assert!(json["backends"].is_array());
}

#[tokio::test]
async fn test_stats_queue_total_matches_sum() {
    let server = dev_server().await;
    let resp = server.get("/api/stats").await;
    let json: Value = resp.json();
    let q = &json["orchestrator"]["queue"];
    let sum = q["low"].as_u64().unwrap()
        + q["normal"].as_u64().unwrap()
        + q["high"].as_u64().unwrap()
        + q["critical"].as_u64().unwrap();
    assert_eq!(sum, q["total"].as_u64().unwrap());
}

// ── /v1/models (protected) ────────────────────────────────────────────────────

#[tokio::test]
async fn test_v1_models_dev_mode_passthrough() {
    // No keys configured → dev mode → 200 from DB
    let server = dev_server().await;
    let resp = server.get("/v1/models").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert_eq!(json["object"], "list");
    assert!(json["data"].is_array());
}

#[tokio::test]
async fn test_v1_models_missing_key_returns_401() {
    let server = auth_server(&["valid-key-123"]).await;
    let resp = server.get("/v1/models").await;
    resp.assert_status(StatusCode::UNAUTHORIZED);
    let json: Value = resp.json();
    assert_eq!(json["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn test_v1_models_wrong_key_returns_401() {
    let server = auth_server(&["valid-key-123"]).await;
    let resp = server.get("/v1/models").add_header("x-api-key", "wrong-key").await;
    resp.assert_status(StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_v1_models_valid_key_returns_200() {
    let server = auth_server(&["valid-key-123"]).await;
    let resp = server.get("/v1/models").add_header("x-api-key", "valid-key-123").await;
    resp.assert_status_ok();
}

// ── /v1/chat/completions (validation + auth) ──────────────────────────────────

#[tokio::test]
async fn test_chat_completion_empty_messages_returns_400() {
    let server = dev_server().await;
    let resp = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": []
        }))
        .await;
    resp.assert_status(StatusCode::BAD_REQUEST);
    let json: Value = resp.json();
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn test_chat_completion_missing_key_when_auth_enabled_returns_401() {
    let server = auth_server(&["sk-test"]).await;
    let resp = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;
    resp.assert_status(StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_chat_completion_valid_key_no_backend_returns_502() {
    // Auth passes, but no llama-server running → orchestrator returns 502
    let server = auth_server(&["sk-test"]).await;
    let resp = server
        .post("/v1/chat/completions")
        .add_header("x-api-key", "sk-test")
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .await;
    resp.assert_status(StatusCode::BAD_GATEWAY);
    let json: Value = resp.json();
    assert_eq!(json["error"]["type"], "backend_error");
}

#[tokio::test]
async fn test_chat_completion_dev_mode_no_backend_returns_502() {
    let server = dev_server().await;
    let resp = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .await;
    // No backend → 502
    resp.assert_status(StatusCode::BAD_GATEWAY);
}

// ── Rate limiting ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_rate_limit_second_request_returns_429() {
    let server = rate_limit_server().await;

    // First request: passes rate limiter, fails at inference (no backend)
    let first = server
        .get("/v1/models")
        .await;
    assert_ne!(first.status_code(), StatusCode::TOO_MANY_REQUESTS);

    // Second request: same "anonymous" key → over the 1 RPM quota
    let second = server
        .get("/v1/models")
        .await;
    second.assert_status(StatusCode::TOO_MANY_REQUESTS);
    let json: Value = second.json();
    assert_eq!(json["error"]["type"], "rate_limit_error");
    assert!(json["error"]["retry_after_secs"].is_number());
}

#[tokio::test]
async fn test_rate_limit_different_keys_independent_buckets() {
    let server = rate_limit_server().await;

    // Exhaust quota for key-a
    let _ = server.get("/v1/models").add_header("x-api-key", "key-a").await;
    let second_a = server.get("/v1/models").add_header("x-api-key", "key-a").await;
    second_a.assert_status(StatusCode::TOO_MANY_REQUESTS);

    // key-b is a fresh bucket → should not be rate-limited
    let first_b = server.get("/v1/models").add_header("x-api-key", "key-b").await;
    assert_ne!(first_b.status_code(), StatusCode::TOO_MANY_REQUESTS);
}

// ── Embeddings validation ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_embeddings_missing_key_when_auth_enabled_returns_401() {
    let server = auth_server(&["emb-key"]).await;
    let resp = server
        .post("/v1/embeddings")
        .json(&serde_json::json!({
            "model": "text-embed",
            "input": "hello"
        }))
        .await;
    resp.assert_status(StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_embeddings_no_backend_returns_503_or_502() {
    let server = dev_server().await;
    let resp = server
        .post("/v1/embeddings")
        .json(&serde_json::json!({
            "model": "text-embed",
            "input": "hello world"
        }))
        .await;
    // No backend → 503 (no backends available)
    let code = resp.status_code().as_u16();
    assert!(code == 503 || code == 502, "expected 502 or 503, got {code}");
}
