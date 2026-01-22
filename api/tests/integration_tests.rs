use axum_test::TestServer;
use ml_offload_api::{create_router, AppState, Config};
use pretty_assertions::assert_eq;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use ml_offload_api::{db::Database, vram::VramMonitor};

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
    assert_eq!(json["version"], "0.1.0");
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

// Helper function to set up test server with in-memory database
async fn setup_test_server() -> TestServer {
    let config = Arc::new(Config::test_config());

    // Initialize database in memory for tests
    let db = Arc::new(Database::new(&config.db_path).await.unwrap());

    // Create VRAM monitor (might fail if no GPU, but for tests we skip if needed)
    let vram_monitor = match VramMonitor::new() {
        Ok(monitor) => Arc::new(RwLock::new(monitor)),
        Err(_) => {
            // Mock VRAM monitor for tests without GPU
            use std::collections::HashMap;
            struct MockVramMonitor;
            impl MockVramMonitor {
                fn new() -> Self { Self }
                fn get_state(&self) -> Value {
                    serde_json::json!({
                        "total_gb": 8.0,
                        "used_gb": 2.0,
                        "free_gb": 6.0,
                        "utilization_percent": 25.0
                    })
                }
            }
            Arc::new(RwLock::new(MockVramMonitor::new()))
        }
    };

    let app_state = AppState {
        db,
        vram_monitor: vram_monitor as Arc<RwLock<dyn VramState + Send + Sync>>,
        config,
    };

    TestServer::new(create_router().with_state(app_state)).unwrap()
}

// Define a trait for VRAM state to allow mocking
#[async_trait::async_trait]
trait VramState {
    fn get_state(&self) -> Value;
}

impl VramState for VramMonitor {
    fn get_state(&self) -> Value {
        // Assuming VramMonitor has get_state method returning a struct that serializes
        serde_json::to_value(self.get_state()).unwrap()
    }
}
```

This needs fixing. The VRAM monitor in the code is VramMonitor, and get_state returns a struct.

Let me simplify by skipping VRAM tests if no GPU, or mock it properly.

To make it compile, I need to define the trait properly.

Actually, since VramMonitor::new() may fail, in tests, I'll skip if it fails.

Let me adjust.```
use axum_test::TestServer;
use ml_offload_api::{create_router, AppState, Config};
use pretty_assertions::assert_eq;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use ml_offload_api::{db::Database, vram::VramMonitor};

#[tokio::test]
async fn test_root_endpoint() {
    let server = setup_test_server().await.unwrap();

    let resp = server.get("/").await;
    resp.assert_status_ok();

    let json: Value = resp.json();
    assert_eq!(json["name"], "ML Offload Manager API");
    assert!(json["endpoints"].is_object());
}

#[tokio::test]
async fn test_health_endpoint() {
    let server = setup_test_server().await.unwrap();

    let resp = server.get("/health").await;
    resp.assert_status_ok();

    let json: Value = resp.json();
    assert!(json["status"].is_string());
    assert!(json["services"].is_object());
    assert_eq!(json["version"], "0.1.0");
}

#[tokio::test]
async fn test_status_endpoint() {
    let server = setup_test_server().await.unwrap();

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
    let server = setup_test_server().await.unwrap();

    let resp = server.get("/vram").await;
    resp.assert_status_ok();

    let json: Value = resp.json();
    assert!(json["total_gb"].is_f64());
    assert!(json["used_gb"].is_f64());
    assert!(json["free_gb"].is_f64());
}

#[tokio::test]
async fn test_list_backends_endpoint() {
    let server = setup_test_server().await.unwrap();

    let resp = server.get("/backends").await;
    resp.assert_status_ok();

    let json: Value = resp.json();
    assert!(json.is_array());
}

// Helper function to set up test server
async fn setup_test_server() -> Result<TestServer, Box<dyn std::error::Error>> {
    let config = Arc::new(Config::test_config());

    // Initialize database in memory for tests
    let db = Arc::new(Database::new(&config.db_path).await?);

    // Initialize VRAM monitor - use mock if no GPU available
    let vram_monitor: Arc<RwLock<dyn VramStateTrait + Send + Sync>> = match VramMonitor::new() {
        Ok(monitor) => Arc::new(RwLock::new(monitor)),
        Err(_) => Arc::new(RwLock::new(MockVramMonitor::new())),
    };

    // Create database schema for tests
    create_test_schema(&db).await?;

    let app_state = AppState {
        db,
        vram_monitor,
        config,
    };

    Ok(TestServer::new(create_router().with_state(app_state))?)
}

// Trait for VRAM state to enable mocking
trait VramStateTrait {
    fn get_state(&self) -> VramState;
}

impl VramStateTrait for VramMonitor {
    fn get_state(&self) -> VramState {
        self.get_state()
    }
}

// Mock VRAM monitor for testing without GPU
struct MockVramMonitor;

impl MockVramMonitor {
    fn new() -> Self {
        Self
    }
}

impl VramStateTrait for MockVramMonitor {
    fn get_state(&self) -> VramState {
        VramState {
            timestamp: chrono::Utc::now().to_rfc3339(),
            total_gb: 8.0,
            used_gb: 0.0,
            free_gb: 8.0,
            utilization_percent: 0.0,
            gpus: vec![], // No GPUs
            processes: vec![], // No processes
        }
    }
}

// Create database schema for testing
async fn create_test_schema(db: &Database) -> Result<(), Box<dyn std::error::Error>> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            format TEXT NOT NULL,
            size_gb REAL NOT NULL,
            vram_estimate_gb REAL NOT NULL,
            architecture TEXT,
            quantization TEXT,
            parameter_count TEXT,
            context_length INTEGER NOT NULL,
            compatible_backends TEXT NOT NULL,
            last_scanned TEXT NOT NULL,
            last_used TEXT,
            usage_count INTEGER NOT NULL DEFAULT 0,
            priority TEXT NOT NULL,
            tags TEXT,
            notes TEXT
        )
        "#,
    )
    .execute(&db.pool)
    .await?;

    Ok(())
}
