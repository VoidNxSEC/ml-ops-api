use axum_test::TestServer;
use ml_offload_api::{create_router, AppState, Config};
use pretty_assertions::assert_eq;
use serde_json::Value;
use std::sync::Arc;

// ── Test helpers ──────────────────────────────────────────────────────────────

async fn setup_test_server() -> TestServer {
    let config = Arc::new(Config::test_config());
    let state = AppState::new_for_test(config).await;
    TestServer::new(create_router().await.with_state(state)).unwrap()
}

// ── Endpoint tests ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_root_endpoint() {
    let server = setup_test_server().await;
    let resp = server.get("/").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert_eq!(json["name"], "ML Offload Manager API");
    assert!(json["endpoints"].is_object());
}

#[tokio::test]
async fn test_health_endpoint() {
    let server = setup_test_server().await;
    let resp = server.get("/health").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json["status"].is_string());
    assert!(json["services"].is_object());
    assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn test_status_endpoint() {
    let server = setup_test_server().await;
    let resp = server.get("/status").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json["timestamp"].is_string());
    assert!(json["vram"].is_object());
    assert!(json["backends"].is_array());
    assert!(json["loaded_models"].is_array());
}

#[tokio::test]
async fn test_vram_endpoint() {
    let server = setup_test_server().await;
    let resp = server.get("/vram").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json["total_gb"].is_f64());
    assert!(json["used_gb"].is_f64());
    assert!(json["free_gb"].is_f64());
}

#[tokio::test]
async fn test_list_backends_endpoint() {
    let server = setup_test_server().await;
    let resp = server.get("/backends").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json.is_array());
}

#[tokio::test]
async fn test_list_models_endpoint() {
    let server = setup_test_server().await;
    let resp = server.get("/models").await;
    resp.assert_status_ok();
    let json: Value = resp.json();
    assert!(json.is_array());
}

#[tokio::test]
async fn test_unknown_route_returns_404() {
    let server = setup_test_server().await;
    let resp = server.get("/nonexistent").await;
    resp.assert_status_not_found();
}
