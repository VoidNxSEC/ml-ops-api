use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;
use tower_http::{
    cors::CorsLayer,
    trace::{DefaultMakeSpan, TraceLayer},
};
use crate::models::{ApiResponse, LoadRequest, UnloadRequest, SwitchRequest};
use crate::backends::BackendDriver;
use tracing::{info, warn};

mod api;
mod backends;
mod db;
mod health;
mod inference;
mod models;
mod nats;
mod vram;
mod websocket;

use crate::VramMonitorTrait;

use db::Database;
use vram::VramMonitor;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    db: Arc<Database>,
    vram_monitor: Arc<RwLock<dyn VramMonitorTrait>>,
    config: Arc<Config>,
    pub nats_publisher: Arc<nats::NatsPublisher>,
}

/// Configuration from environment variables
#[derive(Clone)]
pub struct Config {
    host: String,
    port: u16,
    data_dir: String,
    models_path: String,
    db_path: String,
    cors_enabled: bool,
}

impl Config {
    fn from_env() -> Self {
        Self {
            host: std::env::var("ML_OFFLOAD_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("ML_OFFLOAD_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(9000),
            data_dir: std::env::var("ML_OFFLOAD_DATA_DIR")
                .unwrap_or_else(|_| "/var/lib/ml-offload".to_string()),
            models_path: std::env::var("ML_OFFLOAD_MODELS_PATH")
                .unwrap_or_else(|_| "/var/lib/ml-models".to_string()),
            db_path: std::env::var("ML_OFFLOAD_DB_PATH")
                .unwrap_or_else(|_| "/var/lib/ml-offload/registry.db".to_string()),
            cors_enabled: std::env::var("ML_OFFLOAD_CORS_ENABLED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(false),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ml_offload_api=info,axum=info,tower_http=info".into()),
        )
        .init();

    // Load configuration
    let config = Arc::new(Config::from_env());
    info!("Starting ML Offload Manager API");
    info!("Host: {}", config.host);
    info!("Port: {}", config.port);
    info!("Data directory: {}", config.data_dir);
    info!("Models path: {}", config.models_path);
    info!("Database: {}", config.db_path);

    // Initialize database
    let db = Arc::new(Database::new(&config.db_path).await?);
    info!("Database initialized");

    // Initialize VRAM monitor
    let vram_monitor = Arc::new(RwLock::new(VramMonitor::new()?));
    info!("VRAM monitor initialized");

    // Initialize NATS publisher (non-fatal if unavailable)
    let nats_url = std::env::var("NATS_URL")
        .unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let nats_publisher = Arc::new(nats::NatsPublisher::connect(&nats_url).await);

    // Create application state
    let app_state = AppState {
        db,
        vram_monitor,
        config: config.clone(),
        nats_publisher,
    };

    // Build router
    let app = Router::new()
        // Root endpoint
        .route("/", get(root_handler))
        // Health check (basic)
        .route("/health", get(health_handler))
        // API health check (detailed backend status)
        .route("/api/health", get(health::health_check_handler))
        .route("/api/backend/info", get(health::backend_info_handler))
        // Backends
        .route("/backends", get(list_backends_handler))
        // Models
        .route("/models", get(list_models_handler))
        .route("/models/:id", get(get_model_handler))
        .route("/models/scan", post(trigger_scan_handler))
        // Status
        .route("/status", get(status_handler))
        // VRAM
        .route("/vram", get(vram_handler))
        // WebSocket
        .route("/ws", get(websocket::websocket_handler))
        // OpenAI-compatible inference endpoints
        .route("/v1/models", get(inference::list_models_openai_handler))
        .route("/v1/chat/completions", post(inference::chat_completions_handler))
        .route("/v1/embeddings", post(inference::embeddings_handler))
        // Load/Unload/Switch
        .route("/load", post(load_model_handler))
        .route("/unload", post(unload_model_handler))
        .route("/switch", post(switch_model_handler))
        // Add state
        .with_state(app_state)
        // Add middleware
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::default().include_headers(false)),
        );

    // Add CORS if enabled
    let app = if config.cors_enabled {
        info!("CORS enabled");
        app.layer(CorsLayer::permissive())
    } else {
        app
    };

    // Start server
    let addr = SocketAddr::from((
        config.host.parse::<std::net::IpAddr>()?,
        config.port,
    ));
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// =============================================================================
// Handlers
// =============================================================================

async fn root_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "ML Offload Manager API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "backends": "/backends",
            "models": "/models",
            "status": "/status",
            "vram": "/vram",
        }
    }))
}

async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    let services = serde_json::json!({
        "registry_db": std::path::Path::new(&state.config.db_path).exists(),
        "models_path": std::path::Path::new(&state.config.models_path).exists(),
        "vram_monitor": true,
    });

    let all_healthy = services
        .as_object()
        .map(|obj| obj.values().all(|v| v.as_bool().unwrap_or(false)))
        .unwrap_or(false);

    Json(serde_json::json!({
        "status": if all_healthy { "healthy" } else { "degraded" },
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "version": "0.1.0",
        "services": services,
    }))
}

async fn list_backends_handler() -> impl IntoResponse {
    tokio::task::spawn_blocking(|| async {
        match BackendDriver::list_backends().await {
            Ok(backends) => Ok::<_, StatusCode>(Json(backends)),
            Err(e) => {
                warn!("Backend list error: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    })
    .await
    .unwrap()
    .unwrap_or_else(|_| {
        Json(ApiResponse::error("Failed to list backends".to_string()))
    })
}

#[derive(Deserialize)]
struct ModelsQuery {
    format: Option<String>,
    backend: Option<String>,
    limit: Option<i64>,
}

async fn list_models_handler(
    State(state): State<AppState>,
    Query(query): Query<ModelsQuery>,
) -> Result<impl IntoResponse, StatusCode> {
    match state
        .db
        .list_models(
            query.format.as_deref(),
            query.backend.as_deref(),
            query.limit.unwrap_or(100),
        )
        .await
    {
        Ok(models) => Ok(Json(models)),
        Err(e) => {
            warn!("Error listing models: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_model_handler(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<impl IntoResponse, StatusCode> {
    match state.db.get_model_by_id(id).await {
        Ok(Some(model)) => Ok(Json(model)),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            warn!("Error getting model {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn trigger_scan_handler() -> impl IntoResponse {
    // Trigger systemd service
    tokio::spawn(async {
        if let Err(e) = tokio::process::Command::new("systemctl")
            .args(["start", "ml-registry-scan.service"])
            .output()
            .await
        {
            warn!("Failed to trigger scan: {}", e);
        }
    });

    Json(serde_json::json!({
        "status": "scan_triggered",
        "message": "Model registry scan started in background"
    }))
}

async fn status_handler(State(state): State<AppState>) -> impl IntoResponse {
    let vram_monitor = state.vram_monitor.read().await;
    let vram_state = vram_monitor.get_state();
    let backends = BackendDriver::list_backends().await.unwrap_or_default();

    Json(serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "vram": {
            "total_gb": vram_state.total_gb,
            "used_gb": vram_state.used_gb,
            "free_gb": vram_state.free_gb,
            "utilization_percent": vram_state.utilization_percent,
        },
        "backends": backends,
        "loaded_models": [],
        "pending_queue": [],
    }))
}

async fn vram_handler(State(state): State<AppState>) -> impl IntoResponse {
    let vram_monitor = state.vram_monitor.read().await;
    let vram_state = vram_monitor.get_state();

    Json(vram_state)
}

async fn load_model_handler(
    State(state): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> impl IntoResponse {
    let model_path = match req.model_id {
        Some(id) => {
            match state.db.get_model_by_id(id).await {
                Ok(Some(model)) => model.path,
                Ok(None) => return Json(ApiResponse::error("Model not found".to_string())),
                Err(e) => {
                    warn!("DB error: {}", e);
                    return Json(ApiResponse::error("Database error".to_string()));
                }
            }
        }
        None => req.model_path.unwrap_or_else(|| return Json(ApiResponse::error("Model ID or path required".to_string()))),
    };

    match BackendDriver::load_model(&req.backend, &model_path, req.gpu_layers).await {
        Ok(()) => Json(ApiResponse::success("Model loaded successfully".to_string())),
        Err(e) => {
            warn!("Load failed for {} on {}: {}", model_path, req.backend, e);
            Json(ApiResponse::error(format!("Load failed: {}", e)))
        }
    }
}

async fn unload_model_handler(
    State(_state): State<AppState>,
    Json(req): Json<UnloadRequest>,
) -> impl IntoResponse {
    match BackendDriver::unload_model(&req.backend).await {
        Ok(()) => Json(ApiResponse::success("Model unloaded successfully".to_string())),
        Err(e) => {
            warn!("Unload failed for {}: {}", req.backend, e);
            Json(ApiResponse::error(format!("Unload failed: {}", e)))
        }
    }
}

async fn switch_model_handler(
    State(state): State<AppState>,
    Json(req): Json<SwitchRequest>,
) -> impl IntoResponse {
    // Unload first, then load new
    if let Err(e) = BackendDriver::unload_model(&req.backend).await {
        warn!("Switch unload failed: {}", e);
    }

    let model_path = match req.model_id {
        Some(id) => {
            match state.db.get_model_by_id(id).await {
                Ok(Some(model)) => model.path,
                Ok(None) => return Json(ApiResponse::error("Model not found".to_string())),
                Err(e) => {
                    warn!("DB error: {}", e);
                    return Json(ApiResponse::error("Database error".to_string()));
                }
            }
        }
        None => req.model_path.unwrap_or_else(|| return Json(ApiResponse::error("Model ID or path required".to_string()))),
    };

    match BackendDriver::load_model(&req.backend, &model_path, req.gpu_layers).await {
        Ok(()) => Json(ApiResponse::success("Model switched successfully".to_string())),
        Err(e) => {
            warn!("Switch load failed for {} on {}: {}", model_path, req.backend, e);
            Json(ApiResponse::error(format!("Switch failed: {}", e)))
        }
    }
}
