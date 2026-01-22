//! llama.cpp HTTP Client
//!
//! High-performance HTTP client for communicating with llama.cpp servers.
//! Supports B200-optimized configurations, streaming, and comprehensive error handling.

use crate::{config::LlamaCppConfig, error::LlamaCppError};
use async_trait::async_trait;
use backoff::ExponentialBackoff;
use bytes::Bytes;
use futures::Stream;
use reqwest::{Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::pin::Pin;
use std::time::Duration;
use tensorforge_core::{BackendHealth, BackendStatus, InferenceRequest, VramUsage};
use tracing::{debug, error, info, trace, warn};

/// llama.cpp HTTP client for communicating with llama.cpp servers
#[derive(Debug, Clone)]
pub struct LlamaCppClient {
    /// HTTP client
    client: Client,
    /// Configuration
    config: LlamaCppConfig,
    /// Base URL for llama.cpp API
    base_url: String,
}

/// Health check response from llama.cpp server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppHealthResponse {
    /// Server status
    pub status: String,
    /// Model information (if loaded)
    #[serde(default)]
    pub model_info: Option<LlamaCppModelInfo>,
    /// Slot information for concurrent requests
    #[serde(default)]
    pub slots: Option<Vec<LlamaCppSlotInfo>>,
    /// Number of idle slots
    #[serde(default)]
    pub slots_idle: Option<u32>,
    /// Number of processing slots
    #[serde(default)]
    pub slots_processing: Option<u32>,
    /// GPU information (if available)
    #[serde(default)]
    pub gpu_info: Option<Vec<LlamaCppGpuInfo>>,
}

/// Model information from llama.cpp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppModelInfo {
    /// Model identifier
    pub model: String,
    /// Context length
    #[serde(default)]
    pub n_ctx: u32,
    /// Number of GPU layers
    #[serde(default)]
    pub n_gpu_layers: u32,
    /// Number of threads
    #[serde(default)]
    pub n_threads: u32,
    /// Batch size
    #[serde(default)]
    pub n_batch: u32,
    /// FlashAttention enabled
    #[serde(default)]
    pub flash_attn: bool,
    /// CUDA graphs enabled
    #[serde(default)]
    pub cuda_graphs: bool,
    /// Rope scaling type
    #[serde(default)]
    pub rope_scaling_type: String,
}

/// GPU information from llama.cpp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppGpuInfo {
    /// GPU index
    pub index: u32,
    /// GPU name
    pub name: String,
    /// Total VRAM in bytes
    pub total_memory: u64,
    /// Used VRAM in bytes
    pub used_memory: u64,
    /// GPU utilization percentage
    #[serde(default)]
    pub utilization: Option<f32>,
    /// GPU temperature in Celsius
    #[serde(default)]
    pub temperature: Option<f32>,
}

/// Slot information for concurrent requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppSlotInfo {
    /// Slot ID
    pub id: u32,
    /// Slot state
    pub state: String,
    /// Task ID (if processing)
    #[serde(default)]
    pub task_id: Option<u32>,
}

/// Model load request for llama.cpp
#[derive(Debug, Clone, Serialize)]
pub struct LoadModelRequest {
    /// Model identifier
    pub model_id: String,
    /// Path to model files
    pub model_path: String,
    /// Number of GPU layers to load (-1 = all layers)
    pub n_gpu_layers: i32,
    /// Context length
    pub n_ctx: u32,
    /// Batch size
    pub n_batch: u32,
    /// Enable FlashAttention
    pub flash_attn: bool,
    /// Enable CUDA graphs
    pub cuda_graphs: bool,
    /// Rope scaling type
    pub rope_scaling_type: String,
    /// Additional backend options
    #[serde(skip_serializing_if = "Value::is_null")]
    pub backend_options: Value,
}

/// Model load response from llama.cpp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelResponse {
    /// Load success status
    pub success: bool,
    /// Model identifier
    pub model_id: String,
    /// VRAM used in MB
    pub vram_used_mb: u64,
    /// Load time in seconds
    #[serde(default)]
    pub load_time_secs: Option<f64>,
    /// Any warnings during load
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Inference request for llama.cpp
#[derive(Debug, Clone, Serialize)]
pub struct LlamaCppInferenceRequest {
    /// Request ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Model identifier
    pub model: String,
    /// Prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p for nucleus sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Whether to stream response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Additional parameters
    #[serde(skip_serializing_if = "Value::is_null")]
    pub parameters: Value,
}

/// Inference response from llama.cpp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppInferenceResponse {
    /// Request ID
    pub request_id: String,
    /// Generated text
    pub text: String,
    /// Prompt token count
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Completion token count
    #[serde(default)]
    pub completion_tokens: u32,
    /// Total token count
    #[serde(default)]
    pub total_tokens: u32,
    /// Finish reason
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: Value,
}

/// Streaming chunk from llama.cpp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppStreamChunk {
    /// Request ID
    pub request_id: String,
    /// Generated text so far
    pub text: String,
    /// Is this the final chunk?
    pub finished: bool,
    /// Token count
    #[serde(default)]
    pub token_count: Option<u32>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: Value,
}

/// VRAM usage response from llama.cpp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppVramUsage {
    /// Total VRAM in MB
    pub total_mb: u64,
    /// Used VRAM in MB
    pub used_mb: u64,
    /// Free VRAM in MB
    pub free_mb: u64,
    /// Utilization percentage
    pub utilization_percent: f32,
    /// Per-GPU breakdown
    #[serde(default)]
    pub gpus: Vec<LlamaCppGpuInfo>,
}

impl LlamaCppClient {
    /// Create a new llama.cpp HTTP client
    ///
    /// # Arguments
    /// * `config` - llama.cpp configuration
    ///
    /// # Returns
    /// * `Result<Self, LlamaCppError>` - New client instance
    pub async fn new(config: &LlamaCppConfig) -> Result<Self, LlamaCppError> {
        info!("Creating llama.cpp client for endpoint: {}", config.endpoint);

        // Validate endpoint URL
        let base_url = config.endpoint.trim_end_matches('/').to_string();

        // Create HTTP client with appropriate timeouts
        let client = Client::builder()
            .timeout(config.request_timeout())
            .connect_timeout(Duration::from_secs(10))
            .tcp_keepalive(Duration::from_secs(60))
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(100)
            .http1_only()
            .build()
            .map_err(|e| LlamaCppError::connection_failed(&base_url, e))?;

        Ok(Self {
            client,
            config: config.clone(),
            base_url,
        })
    }

    /// Perform health check on llama.cpp server
    ///
    /// # Returns
    /// * `Result<BackendHealth, LlamaCppError>` - Health status
    pub async fn health_check(&self) -> Result<BackendHealth, LlamaCppError> {
        let url = format!("{}/health", self.base_url);
        trace!("Performing health check: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LlamaCppError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.ok();
            return Err(LlamaCppError::server_error(
                status.as_u16(),
                &format!("Health check failed with status {}", status),
                body,
            ));
        }

        let health_response: LlamaCppHealthResponse = response
            .json()
            .await
            .map_err(|e| LlamaCppError::InvalidResponse {
                message: format!("Failed to parse health response: {}", e),
                raw_response: None,
                expected: "LlamaCppHealthResponse".to_string(),
            })?;

        // Convert to TensorForge BackendHealth
        let backend_status = match health_response.status.as_str() {
            "ok" | "ready" | "healthy" => BackendStatus::Healthy,
            "busy" | "degraded" => BackendStatus::Degraded,
            _ => BackendStatus::Unhealthy,
        };

        let loaded_models = health_response
            .model_info
            .as_ref()
            .map(|info| vec![info.model.clone()])
            .unwrap_or_default();

        Ok(BackendHealth {
            status: backend_status,
            message: Some(format!("llama.cpp status: {}", health_response.status)),
            timestamp: chrono::Utc::now(),
            loaded_models,
        })
    }

    /// Load a model onto the llama.cpp server
    ///
    /// # Arguments
    /// * `request` - Model load request
    ///
    /// # Returns
    /// * `Result<LoadModelResponse, LlamaCppError>` - Load result
    pub async fn load_model(&self, request: &LoadModelRequest) -> Result<LoadModelResponse, LlamaCppError> {
        let url = format!("{}/v1/models/load", self.base_url);
        info!("Loading model {} to llama.cpp", request.model_id);

        // Use exponential backoff for model loading (can take time)
        let backoff = ExponentialBackoff {
            initial_interval: Duration::from_secs(5),
            max_interval: Duration::from_secs(30),
            max_elapsed_time: Some(self.config.model_load_timeout()),
            ..Default::default()
        };

        let operation = || async {
            let response = self
                .client
                .post(&url)
                .json(request)
                .timeout(self.config.model_load_timeout())
                .send()
                .await
                .map_err(|e| backoff::Error::transient(LlamaCppError::connection_failed(&url, e)))?;

            let status = response.status();

            if !status.is_success() {
                let body = response.text().await.ok();
                let error = LlamaCppError::server_error(
                    status.as_u16(),
                    &format!("Model load failed with status {}", status),
                    body,
                );
                return Err(backoff::Error::transient(error));
            }

            let load_response: LoadModelResponse = response
                .json()
                .await
                .map_err(|e| backoff::Error::permanent(LlamaCppError::InvalidResponse {
                    message: format!("Failed to parse load response: {}", e),
                    raw_response: None,
                    expected: "LoadModelResponse".to_string(),
                }))?;

            if !load_response.success {
                return Err(backoff::Error::permanent(LlamaCppError::ModelLoadFailed {
                    model_id: request.model_id.clone(),
                    reason: "Server reported load failure".to_string(),
                    details: Some(json!({"response": load_response})),
                }));
            }

            Ok(load_response)
        };

        backoff::future::retry(backoff, operation)
            .await
            .map_err(|e| match e {
                backoff::Error::Permanent(err) => err,
                backoff::Error::Transient(err) => LlamaCppError::timeout(
                    self.config.model_load_timeout_secs,
                    &format!("model load for {}", request.model_id),
                ),
            })
    }

    /// Unload a model from llama.cpp server
    ///
    /// # Arguments
    /// * `model_id` - Model identifier to unload
    ///
    /// # Returns
    /// * `Result<(), LlamaCppError>` - Success or error
    pub async fn unload_model(&self, model_id: &str) -> Result<(), LlamaCppError> {
        let url = format!("{}/v1/models/{}", self.base_url, model_id);
        info!("Unloading model {} from llama.cpp", model_id);

        let response = self
            .client
            .delete(&url)
            .send()
            .await
            .map_err(|e| LlamaCppError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() && status != StatusCode::NOT_FOUND {
            let body = response.text().await.ok();
            return Err(LlamaCppError::server_error(
                status.as_u16(),
                &format!("Model unload failed with status {}", status),
                body,
            ));
        }

        Ok(())
    }

    /// Perform inference with llama.cpp server
    ///
    /// # Arguments
    /// * `request` - Inference request
    ///
    /// # Returns
    /// * `Result<LlamaCppInferenceResponse, LlamaCppError>` - Inference result
    pub async fn infer(&self, request: &LlamaCppInferenceRequest) -> Result<LlamaCppInferenceResponse, LlamaCppError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        trace!("Sending inference request for model {}", request.model);

        let response = self
            .client
            .post(&url)
            .json(request)
            .timeout(self.config.request_timeout())
            .send()
            .await
            .map_err(|e| LlamaCppError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.ok();
            return Err(LlamaCppError::server_error(
                status.as_u16(),
                &format!("Inference failed with status {}", status),
                body,
            ));
        }

        let inference_response: LlamaCppInferenceResponse = response
            .json()
            .await
            .map_err(|e| LlamaCppError::InvalidResponse {
                message: format!("Failed to parse inference response: {}", e),
                raw_response: None,
                expected: "LlamaCppInferenceResponse".to_string(),
            })?;

        Ok(inference_response)
    }

    /// Perform streaming inference with llama.cpp server
    ///
    /// # Arguments
    /// * `request` - Inference request with streaming enabled
    ///
    /// # Returns
    /// * `Result<Pin<Box<dyn Stream<Item = Result<LlamaCppStreamChunk, LlamaCppError>> + Send>>, LlamaCppError>` - Stream of chunks
    pub async fn infer_stream(
        &self,
        request: &LlamaCppInferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LlamaCppStreamChunk, LlamaCppError>> + Send>>, LlamaCppError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        trace!("Sending streaming inference request for model {}", request.model);

        // Clone request with streaming enabled
        let mut stream_request = request.clone();
        stream_request.stream = Some(true);

        let response = self
            .client
            .post(&url)
            .json(&stream_request)
            .timeout(self.config.request_timeout())
            .send()
            .await
            .map_err(|e| LlamaCppError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.ok();
            return Err(LlamaCppError::server_error(
                status.as_u16(),
                &format!("Streaming inference failed with status {}", status),
                body,
            ));
        }

        // Create stream from response bytes
        let stream = response
            .bytes_stream()
            .map(|chunk_result| {
                chunk_result
                    .map_err(|e| LlamaCppError::HttpClient { source: e })
                    .and_then(|chunk| {
                        // Parse chunk as JSON
                        let chunk_str = String::from_utf8(chunk.to_vec())
                            .map_err(|e| LlamaCppError::InvalidResponse {
                                message: format!("Invalid UTF-8 in stream chunk: {}", e),
                                raw_response: Some(String::from_utf8_lossy(&chunk).to_string()),
                                expected: "UTF-8 string".to_string(),
                            })?;

                        // Skip empty chunks
                        if chunk_str.trim().is_empty() {
                            return Ok(None);
                        }

                        // Parse as stream chunk
                        let stream_chunk: LlamaCppStreamChunk = serde_json::from_str(&chunk_str)
                            .map_err(|e| LlamaCppError::InvalidResponse {
                                message: format!("Failed to parse stream chunk: {}", e),
                                raw_response: Some(chunk_str),
                                expected: "LlamaCppStreamChunk".to_string(),
                            })?;

                        Ok(Some(stream_chunk))
                    })
                    .transpose()
            })
            .filter_map(|item| async { item });

        Ok(Box::pin(stream))
    }

    /// Get VRAM usage information from llama.cpp server
    ///
    /// # Returns
    /// * `Result<VramUsage, LlamaCppError>` - VRAM usage information
    pub async fn get_vram_usage(&self) -> Result<VramUsage, LlamaCppError> {
        // llama.cpp doesn't have a dedicated VRAM endpoint, try to get from health
        self.get_vram_usage_from_health().await
    }

    /// Get VRAM usage from health endpoint
    async fn get_vram_usage_from_health(&self) -> Result<VramUsage, LlamaCppError> {
        let url = format!("{}/health", self.base_url);
        trace!("Querying VRAM usage from llama.cpp health endpoint");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LlamaCppError::connection_failed(&url, e))?;

        let health_response: LlamaCppHealthResponse = response
            .json()
            .await
            .map_err(|e| LlamaCppError::InvalidResponse {
                message: format!("Failed to parse health response for VRAM: {}", e),
                raw_response: None,
                expected: "LlamaCppHealthResponse".to_string(),
            })?;

        if let Some(gpu_info) = health_response.gpu_info {
            if !gpu_info.is_empty() {
                let total_mb = gpu_info.iter().map(|gpu| gpu.total_memory / 1024 / 1024).sum();
                let used_mb = gpu_info.iter().map(|gpu| gpu.used_memory / 1024 / 1024).sum();
                let utilization_percent = if total_mb > 0 {
                    (used_mb as f32 / total_mb as f32) * 100.0
                } else {
                    0.0
                };

                return Ok(VramUsage {
                    total_mb,
                    used_mb,
                    peak_mb: used_mb,
                    utilization_percent,
                });
            }
        }

        // Return placeholder if no VRAM info available
        warn!("VRAM information not available from llama.cpp");
        Ok(VramUsage {
            total_mb: 0,
            used_mb: 0,
            peak_mb: 0,
            utilization_percent: 0.0,
        })
    }

    /// List loaded models from llama.cpp server
    ///
    /// # Returns
    /// * `Result<Vec<String>, LlamaCppError>` - List of loaded model IDs
    pub async fn list_models(&self) -> Result<Vec<String>, LlamaCppError> {
        // llama.cpp typically only has one model loaded at a time
        // We can check the health endpoint for model info
        let health = self.health_check().await?;

        if !health.loaded_models.is_empty() {
            Ok(health.loaded_models)
        } else {
            // Try to get from /props endpoint as fallback
            let url = format!("{}/props", self.base_url);

            let response = self
                .client
                .get(&url)
                .send()
                .await
                .map_err(|e| LlamaCppError::connection_failed(&url, e))?;

            if response.status().is_success() {
                let model_info: LlamaCppModelInfo = response
                    .json()
                    .await
                    .map_err(|e| LlamaCppError::InvalidResponse {
                        message: format!("Failed to parse model info: {}", e),
                        raw_response: None,
                        expected: "LlamaCppModelInfo".to_string(),
                    })?;

                if !model_info.model.is_empty() {
                    return Ok(vec![model_info.model]);
                }
            }

            Ok(vec![])
        }
    }

    /// Get model properties/information
    ///
    /// # Returns
    /// * `Result<LlamaCppModelInfo, LlamaCppError>` - Model information
    pub async fn get_model_props(&self) -> Result<LlamaCppModelInfo, LlamaCppError> {
        let url = format!("{}/props", self.base_url);
        trace!("Getting model properties from llama.cpp");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LlamaCppError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.ok();
            return Err(LlamaCppError::server_error(
                status.as_u16(),
                &format!("Get model props failed with status {}", status),
                body,
            ));
        }

        let model_info: LlamaCppModelInfo = response
            .json()
            .await
            .map_err(|e| LlamaCppError::InvalidResponse {
                message: format!("Failed to parse model info: {}", e),
                raw_response: None,
                expected: "LlamaCppModelInfo".to_string(),
            })?;

        Ok(model_info)
    }

    /// Get slot information for concurrent request management
    ///
    /// # Returns
    /// * `Result<Vec<LlamaCppSlotInfo>, LlamaCppError>` - Slot information
    pub async fn get_slots(&self) -> Result<Vec<LlamaCppSlotInfo>, LlamaCppError> {
        let url = format!("{}/slots", self.base_url);
        trace!("Getting slot information from llama.cpp");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LlamaCppError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            // Some llama.cpp servers might not support /slots endpoint
            return Ok(vec![]);
        }

        let slots_response: Value = response
            .json()
            .await
            .map_err(|e| LlamaCppError::InvalidResponse {
                message: format!("Failed to parse slots response: {}", e),
                raw_response: None,
                expected: "JSON array of slots".to_string(),
            })?;

        // Extract slots from response
        let slots = slots_response
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| serde_json::from_value(item.clone()).ok())
                    .collect()
            })
            .unwrap_or_else(Vec::new);

        Ok(slots)
    }

    /// Get base URL for this client
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get configuration
    pub fn config(&self) -> &LlamaCppConfig {
        &self.config
    }
}

impl From<InferenceRequest> for LlamaCppInferenceRequest {
    fn from(request: InferenceRequest) -> Self {
        LlamaCppInferenceRequest {
            request_id: Some(request.id),
            model: request.model,
            prompt: request.prompt,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop,
            stream: request.stream,
            n: request.n,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            parameters: request.parameters.unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        http::Method,
        matchers::{method, path},
        Mock, MockServer, ResponseTemplate,
    };

    #[tokio::test]
    async fn test_client_creation() {
        let config = LlamaCppConfig::default();
        let client = LlamaCppClient::new(&config).await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "ok",
                "model_info": {
                    "model": "test-model",
                    "n_ctx": 4096,
                    "n_gpu_layers": 32,
                    "n_threads": 8
                },
                "slots_idle": 4,
                "slots_processing": 2
            })))
            .mount(&mock_server)
            .await;

        let config = LlamaCppConfig {
            endpoint: mock_server.uri(),
            ..Default::default()
        };

        let client = LlamaCppClient::new(&config).await.unwrap();
        let health = client.health_check().await.unwrap();

        assert_eq!(health.status, BackendStatus::Healthy);
        assert_eq!(health.loaded_models, vec!["test-model"]);
    }

    #[tokio::test]
    async fn test_health_check_failure() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let config = LlamaCppConfig {
            endpoint: mock_server.uri(),
            ..Default::default()
        };

        let client = LlamaCppClient::new(&config).await.unwrap();
        let result = client.health_check().await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(error, LlamaCppError::ServerError { status: 500, .. }));
    }

    #[tokio::test]
    async fn test_inference_request_conversion() {
        let tf_request = InferenceRequest {
            id: "test-id".to_string(),
            model: "test-model".to_string(),
            prompt: "Test prompt".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop: Some(vec!["stop".to_string()]),
            stream: Some(false),
            n: Some(1),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.1),
            parameters: Some(json!({"test": "param"})),
            ..Default::default()
        };

        let llamacpp_request = LlamaCppInferenceRequest::from(tf_request);

        assert_eq!(llamacpp_request.request_id, Some("test-id".to_string()));
        assert_eq!(llamacpp_request.model, "test-model");
        assert_eq!(llamacpp_request.prompt, "Test prompt");
        assert_eq!(llamacpp_request.max_tokens, Some(100));
        assert_eq!(llamacpp_request.temperature, Some(0.7));
        assert_eq!(llamacpp_request.top_p, Some(0.9));
        assert_eq!(llamacpp_request.stop, Some(vec!["stop".to_string()]));
        assert_eq!(llamacpp_request.stream, Some(false));
        assert_eq!(llamacpp_request.n, Some(1));
        assert_eq!(llamacpp_request.frequency_penalty, Some(0.1));
        assert_eq!(llamacpp_request.presence_penalty, Some(0.1));
        assert_eq!(llamacpp_request.parameters, json!({"test": "param"}));
    }

    #[tokio::test]
    async fn test_get_model_props() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/props"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "test-model",
                "n_ctx": 8192,
                "n_gpu_layers": 48,
                "n_threads": 12,
                "n_batch": 512,
                "flash_attn": true,
                "cuda_graphs": false,
                "rope_scaling_type": "linear"
            })))
            .mount(&mock_server)
            .await;

        let config = LlamaCppConfig {
            endpoint: mock_server.uri(),
            ..Default::default()
        };

        let client = LlamaCppClient::new(&config).await.unwrap();
        let props = client.get_model_props().await.unwrap();

        assert_eq!(props.model, "test-model");
        assert_eq!(props.n_ctx, 8192);
        assert_eq!(props.n_gpu_layers, 48);
        assert_eq!(props.n_threads, 12);
        assert_eq!(props.n_batch, 512);
        assert!(props.flash_attn);
        assert!(!props.cuda_graphs);
        assert_eq!(props.rope_scaling_type, "linear");
    }

    #[tokio::test]
    async fn test_get_vram_usage_from_health() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "ok",
                "gpu_info": [
                    {
                        "index": 0,
                        "name": "NVIDIA B200",
                        "total_memory": 201326592000, // 192GB in bytes
                        "used_memory": 107374182400,  // 100GB in bytes
                        "utilization": 65.5,
                        "temperature": 68.0
                    }
                ]
            })))
            .mount(&mock_server)
            .await;

        let config = LlamaCppConfig {
            endpoint: mock_server.uri(),
            ..Default::default()
        };

        let client = LlamaCppClient::new(&config).await.unwrap();
        let vram = client.get_vram_usage().await.unwrap();

        // 192GB = 192 * 1024 = 196608 MB
        // 100GB = 100 * 1024 = 102400 MB
        assert_eq!(vram.total_mb, 196608);
        assert_eq!(vram.used_mb, 102400);
        assert_eq!(vram.peak_mb, 102400);
        assert!((vram.utilization_percent - 52.08).abs() < 0.1); // 100/192 ≈ 52.08%
    }

    #[tokio::test]
    async fn test_list_models() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "ok",
                "model_info": {
                    "model": "mixtral-8x7b",
                    "n_ctx": 32768,
                    "n_gpu_layers": 48,
                    "n_threads": 8
                }
            })))
            .mount(&mock_server)
            .await;

        let config = LlamaCppConfig {
            endpoint: mock_server.uri(),
            ..Default::default()
        };

        let client = LlamaCppClient::new(&config).await.unwrap();
        let models = client.list_models().await.unwrap();

        assert_eq!(models, vec!["mixtral-8x7b"]);
    }
}
