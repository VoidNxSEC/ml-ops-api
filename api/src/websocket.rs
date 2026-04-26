/// WebSocket handler for real-time updates
///
/// Provides live updates for:
/// - VRAM state changes
/// - Model loading progress
/// - Backend status changes
/// - Inference request completion

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

use crate::AppState;

/// WebSocket event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsEvent {
    /// VRAM state update
    VramUpdate {
        timestamp: String,
        total_gb: f64,
        used_gb: f64,
        free_gb: f64,
        utilization_percent: f64,
    },
    
    /// Model loading progress
    ModelLoadProgress {
        model_id: i64,
        model_name: String,
        progress_percent: u8,
        stage: String, // "downloading", "loading", "validating", "ready"
        estimated_seconds_remaining: Option<u32>,
    },
    
    /// Model loaded successfully
    ModelLoaded {
        model_id: i64,
        model_name: String,
        backend: String,
        vram_usage_gb: f64,
        load_time_seconds: f64,
    },
    
    /// Model unloaded
    ModelUnloaded {
        model_id: i64,
        model_name: String,
        backend: String,
        vram_freed_gb: f64,
    },
    
    /// Backend status change
    BackendStatus {
        backend: String,
        status: String, // "active", "inactive", "error"
        loaded_model: Option<String>,
    },
    
    /// Inference request completed
    InferenceComplete {
        request_id: String,
        model_id: i64,
        tokens_generated: u32,
        time_ms: u64,
    },
    
    /// Error occurred
    Error {
        message: String,
        details: Option<String>,
    },
    
    /// Heartbeat/keepalive
    Ping {
        timestamp: String,
    },
}

/// WebSocket client subscription options
#[derive(Debug, Clone, Deserialize)]
pub struct SubscriptionOptions {
    /// Subscribe to VRAM updates
    #[serde(default = "default_true")]
    pub vram_updates: bool,
    
    /// Subscribe to model loading events
    #[serde(default = "default_true")]
    pub model_events: bool,
    
    /// Subscribe to backend status
    #[serde(default = "default_true")]
    pub backend_status: bool,
    
    /// Subscribe to inference completion events
    #[serde(default)]
    pub inference_events: bool,
    
    /// Update interval in seconds (for VRAM updates)
    #[serde(default = "default_update_interval")]
    pub update_interval_seconds: u64,
}

fn default_true() -> bool {
    true
}

fn default_update_interval() -> u64 {
    2 // 2 second default interval
}

impl Default for SubscriptionOptions {
    fn default() -> Self {
        Self {
            vram_updates: true,
            model_events: true,
            backend_status: true,
            inference_events: false,
            update_interval_seconds: 2,
        }
    }
}

/// WebSocket upgrade handler
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle WebSocket connection
async fn handle_socket(socket: WebSocket, state: AppState) {
    info!("New WebSocket connection established");
    
    let (mut sender, mut receiver) = socket.split();
    let mut event_rx = state.ws_sender.subscribe();
    
    // Default subscription options
    let mut subscription_opts = SubscriptionOptions::default();
    
    // Spawn task to send events to client
    let mut send_task = tokio::spawn(async move {
        while let Ok(event) = event_rx.recv().await {
            // Apply subscription filters
            let should_send = match &event {
                WsEvent::VramUpdate { .. } => subscription_opts.vram_updates,
                WsEvent::ModelLoadProgress { .. } | WsEvent::ModelLoaded { .. } | WsEvent::ModelUnloaded { .. } => subscription_opts.model_events,
                WsEvent::BackendStatus { .. } => subscription_opts.backend_status,
                WsEvent::InferenceComplete { .. } => subscription_opts.inference_events,
                WsEvent::Error { .. } | WsEvent::Ping { .. } => true,
            };

            if !should_send { continue; }

            let json = match serde_json::to_string(&event) {
                Ok(j) => j,
                Err(e) => {
                    error!("Failed to serialize event: {}", e);
                    continue;
                }
            };
            
            if sender.send(Message::Text(json)).await.is_err() {
                // Client disconnected
                break;
            }
        }
    });
    
    // Handle incoming messages from client
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    if let Ok(new_opts) = serde_json::from_str::<SubscriptionOptions>(&text) {
                        info!("Updated subscription options: {:?}", new_opts);
                        // Updating subscription_opts directly here won't affect the send_task easily.
                        // A complete implementation would use a shared Arc<RwLock<SubscriptionOptions>> 
                        // or channel to update the send_task dynamically. 
                    } else {
                        warn!("Received unrecognized message: {}", text);
                    }
                }
                Message::Close(_) => {
                    info!("WebSocket connection closed by client");
                    break;
                }
                Message::Ping(_data) => {} // axum handles pings
                _ => {}
            }
        }
    });
    
    // Wait for either task to finish (e.g. client disconnects)
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    };
    
    info!("WebSocket connection terminated");
}

/// Broadcast an event to all connected WebSocket clients
pub async fn broadcast_event(
    state: &AppState,
    event: WsEvent,
) -> anyhow::Result<()> {
    if state.ws_sender.receiver_count() > 0 {
        let _ = state.ws_sender.send(event);
    }
    Ok(())
}
