/// Inference endpoints module
///
/// Provides OpenAI-compatible API endpoints for:
/// - Chat completions (POST /v1/chat/completions)
/// - Embeddings (POST /v1/embeddings)
///
/// These endpoints proxy to the active backend (Ollama, llama.cpp, vLLM, TGI)

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    Json,
};
use bytes::Bytes;
use futures::stream::{self, unfold, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::Instant;
use tracing::{info, warn};
use uuid::Uuid;

use metrics::counter;

use crate::backends::llamacpp::{LlamaCppBackend, LlamaCppConfig};
use crate::metrics as m;
use crate::orchestrator::Priority;
use crate::AppState;

// =============================================================================
// Chat Completions API
// =============================================================================

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    /// ID of the model to use
    pub model: String,
    
    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,
    
    /// Temperature (0.0 to 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    
    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<u32>,
    
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    
    /// Top-p sampling
    #[serde(default)]
    pub top_p: Option<f32>,
    
    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    
    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    
    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant"
    pub role: String,
    
    /// Message content
    pub content: String,
    
    /// Optional name of the message author
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoiceDelta>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoiceDelta {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_n() -> u32 {
    1
}

// =============================================================================
// Embeddings API
// =============================================================================

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingsRequest {
    /// ID of the model to use
    pub model: String,
    
    /// Input text(s) to embed
    pub input: EmbeddingInput,
    
    /// Encoding format (default: "float")
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

fn default_encoding_format() -> String {
    "float".to_string()
}

// =============================================================================
// Handlers
// =============================================================================

pub async fn chat_completions_handler(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    info!("Chat completion request for model: {}", request.model);

    if request.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "messages array cannot be empty", "type": "invalid_request_error" }
            })),
        )
            .into_response();
    }

    // X-Priority: critical | high | normal | low  (default: normal)
    let priority = headers
        .get("x-priority")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();

    if request.stream {
        streaming_response(state, request).await
    } else {
        match non_streaming_response(state, request, priority).await {
            Ok(response) => response.into_response(),
            Err(err) => err.into_response(),
        }
    }
}

async fn non_streaming_response(
    state: AppState,
    request: ChatCompletionRequest,
    priority: Priority,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let request_id = Uuid::new_v4();
    let start = Instant::now();
    let model = request.model.clone();
    let messages_count = request.messages.len();

    // Fire-and-forget: publish inference.request.v1
    {
        let publisher = state.nats_publisher.clone();
        let m = model.clone();
        tokio::spawn(async move {
            publisher.publish_inference_request(request_id, &m, messages_count).await;
        });
    }

    info!(request_id = %request_id, model = %model, %priority, "enqueuing");

    let request_json = serde_json::to_value(&request).unwrap_or_default();

    // Submit to orchestrator — priority queue + N workers + concurrency limiter
    let raw = match state.orchestrator.submit(request_json, priority).await {
        Ok(v) => v,
        Err(e) => {
            warn!(request_id = %request_id, err = %e, "orchestrator error");
            return Err((
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({ "error": { "message": e, "type": "backend_error" } })),
            ));
        }
    };

    match serde_json::from_value::<ChatCompletionResponse>(raw) {
        Ok(chat_response) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            let completion_tokens = chat_response.usage.completion_tokens;
            let publisher = state.nats_publisher.clone();
            let m = model.clone();
            tokio::spawn(async move {
                publisher
                    .publish_inference_response(request_id, &m, completion_tokens, duration_ms, "success")
                    .await;
            });
            Ok(Json(chat_response))
        }
        Err(e) => {
            warn!(request_id = %request_id, err = %e, "response parse error");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": { "message": "Failed to parse backend response", "type": "internal_error" }
                })),
            ))
        }
    }
}

async fn streaming_response(state: AppState, request: ChatCompletionRequest) -> Response {
    counter!(m::STREAMING_REQUESTS, "model" => request.model.clone()).increment(1);
    let request_id = Uuid::new_v4();
    let model = request.model.clone();

    // Fire-and-forget: publish inference.request.v1
    {
        let publisher = state.nats_publisher.clone();
        let m = model.clone();
        let msgs = request.messages.len();
        tokio::spawn(async move {
            publisher.publish_inference_request(request_id, &m, msgs).await;
        });
    }

    // Select best available backend via VRAM-aware router
    let selected = match state.router.pick().await {
        Some(b) => b,
        None => {
            warn!("no backends available for streaming");
            return sse_error("No inference backends available".to_string());
        }
    };

    let backend = match LlamaCppBackend::new(LlamaCppConfig {
        base_url: selected.base_url().to_string(),
        timeout_secs: 300,
    }) {
        Ok(b) => b,
        Err(e) => {
            warn!("streaming backend client failed: {}", e);
            return sse_error(format!("backend unavailable: {e}"));
        }
    };

    info!(
        backend = %selected.config.id,
        score = selected.score,
        kind = %selected.kind(),
        model = %model,
        "routing streaming request"
    );

    let mut req_json = serde_json::to_value(&request).unwrap_or_default();
    req_json["stream"] = serde_json::json!(true);

    let response = match backend.stream_chat_completion(req_json).await {
        Ok(r) => r,
        Err(e) => {
            warn!("backend stream error on {}: {}", selected.config.id, e);
            return sse_error(format!("backend error: {e}"));
        }
    };

    info!(
        request_id = %request_id,
        backend    = %selected.config.id,
        score      = selected.score,
        model      = %model,
        "stream open"
    );

    Sse::new(sse_from_byte_stream(response.bytes_stream()))
        .keep_alive(
            KeepAlive::default()
                .interval(std::time::Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

/// Convert a reqwest byte stream (SSE from llama.cpp) to axum SSE events.
///
/// llama.cpp emits `data: <json>\n\n` per token. We buffer across chunk
/// boundaries, split on `\n\n`, strip `data: `, and forward raw JSON.
/// This is zero-copy at the JSON level — no parsing or re-serialization.
fn sse_from_byte_stream<S>(byte_stream: S) -> impl Stream<Item = Result<Event, Infallible>>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
{
    unfold(
        (Box::pin(byte_stream), String::new()),
        |(mut stream, mut buf)| async move {
            loop {
                // Emit one SSE event per complete `\n\n`-delimited block
                if let Some(pos) = buf.find("\n\n") {
                    let block = buf[..pos].to_string();
                    buf = buf[pos + 2..].to_string();

                    for line in block.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            return Some((
                                Ok(Event::default().data(data.to_string())),
                                (stream, buf),
                            ));
                        }
                    }
                    continue; // block had no data line (comment/empty), keep going
                }

                // Need more bytes from backend
                match stream.next().await {
                    Some(Ok(bytes)) => buf.push_str(&String::from_utf8_lossy(&bytes)),
                    _ => return None, // backend closed or error → end stream
                }
            }
        },
    )
}

fn sse_error(msg: String) -> Response {
    let payload = serde_json::json!({
        "error": { "message": msg, "type": "backend_error" }
    })
    .to_string();
    let s = stream::once(async move { Ok::<_, Infallible>(Event::default().data(payload)) });
    Sse::new(s).into_response()
}

pub async fn embeddings_handler(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, (StatusCode, Json<serde_json::Value>)> {
    info!("Embeddings request for model: {}", request.model);

    // Select best available backend via router
    let selected = match state.router.pick().await {
        Some(b) => b,
        None => {
            warn!("no backends available for embeddings");
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "message": "No inference backends available",
                        "type": "service_unavailable"
                    }
                })),
            ));
        }
    };

    let backend = match LlamaCppBackend::new(LlamaCppConfig {
        base_url: selected.base_url().to_string(),
        timeout_secs: 300,
    }) {
        Ok(b) => b,
        Err(e) => {
            warn!("Failed to create backend client: {}", e);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "Failed to initialize backend client",
                        "type": "internal_error"
                    }
                })),
            ));
        }
    };

    // Convert request to JSON for proxying
    let request_json = serde_json::to_value(&request).unwrap();
    
    // Proxy to llama-server
    match backend.proxy_embeddings(request_json).await {
        Ok(response) => {
            let status = response.status();
            let body = response.bytes().await.unwrap_or_default();
            
            if status.is_success() {
                // Parse and return the response
                match serde_json::from_slice::<EmbeddingsResponse>(&body) {
                    Ok(embeddings_response) => Ok(Json(embeddings_response)),
                    Err(e) => {
                        warn!("Failed to parse llama-server embeddings response: {}", e);
                        Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": {
                                    "message": "Failed to parse backend response",
                                    "type": "internal_error"
                                }
                            })),
                        ))
                    }
                }
            } else {
                warn!("llama-server returned error for embeddings: {}", status);
                Err((
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({
                        "error": {
                            "message": "Backend returned error",
                            "type": "backend_error"
                        }
                    })),
                ))
            }
        }
        Err(e) => {
            warn!("Failed to proxy embeddings to llama-server: {}", e);
            Err((
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": "Failed to connect to backend",
                        "type": "connection_error"
                    }
                })),
            ))
        }
    }
}

// =============================================================================
// Models List (OpenAI compatible)
// =============================================================================

#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

pub async fn list_models_openai_handler(
    State(state): State<AppState>,
) -> Result<Json<ModelsListResponse>, (StatusCode, Json<serde_json::Value>)> {
    // Get models from database
    match state.db.list_models(None, None, 100).await {
        Ok(models) => {
            let data = models
                .into_iter()
                .map(|m| ModelInfo {
                    id: m.name,
                    object: "model".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    owned_by: "ml-offload".to_string(),
                })
                .collect();
            
            Ok(Json(ModelsListResponse {
                object: "list".to_string(),
                data,
            }))
        }
        Err(e) => {
            warn!("Error listing models: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "Failed to list models",
                        "type": "internal_error"
                    }
                })),
            ))
        }
    }
}
