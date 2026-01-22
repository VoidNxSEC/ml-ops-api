//! vLLM HTTP Client
//!
//! High-performance HTTP client for communicating with vLLM servers.
//! Supports B200-optimized configurations, streaming, and comprehensive error handling.

use crate::{config::VllmConfig, error::VllmError};
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

/// vLLM HTTP client for communicating with vLLM servers
#[derive(Debug, Clone)]
pub struct VllmClient {
    /// HTTP client
    client: Client,
    /// Configuration
    config: VllmConfig,
    /// Base URL for vLLM API
    base_url: String,
}

/// Health check response from vLLM server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmHealthResponse {
    /// Server status
    pub status: String,
    /// Loaded models (if any)
    #[serde(default)]
    pub loaded_models: Option<Vec<String>>,
    /// GPU information (if available)
    #[serde(default)]
    pub gpu_info: Option<Vec<VllmGpuInfo>>,
    /// Number of active requests
    #[serde(default)]
    pub active_requests: Option<u32>,
    /// vLLM version
    #[serde(default)]
    pub version: Option<String>,
}

/// GPU information from vLLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmGpuInfo {
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

/// Model load request for vLLM
#[derive(Debug, Clone, Serialize)]
pub struct LoadModelRequest {
    /// Model identifier
    pub model_id: String,
    /// Path to model files
    pub model_path: String,
    /// Maximum model context length
    pub max_model_len: u32,
    /// GPU memory utilization (0.0-1.0)
    pub gpu_memory_utilization: f32,
    /// Tensor parallel size
    pub tensor_parallel_size: u32,
    /// Quantization type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Enable prefix caching
    pub enable_prefix_caching: bool,
    /// Enable FlashAttention
    pub enable_flash_attention: bool,
    /// Additional backend options
    #[serde(skip_serializing_if = "Value::is_null")]
    pub backend_options: Value,
}

/// Model load response from vLLM
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

/// Inference request for vLLM
#[derive(Debug, Clone, Serialize)]
pub struct VllmInferenceRequest {
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

/// Inference response from vLLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmInferenceResponse {
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

/// Streaming chunk from vLLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmStreamChunk {
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

/// VRAM usage response from vLLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmVramUsage {
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
    pub gpus: Vec<VllmGpuInfo>,
}

impl VllmClient {
    /// Create a new vLLM HTTP client
    ///
    /// # Arguments
    /// * `config` - vLLM configuration
    ///
    /// # Returns
    /// * `Result<Self, VllmError>` - New client instance
    pub async fn new(config: &VllmConfig) -> Result<Self, VllmError> {
        info!("Creating vLLM client for endpoint: {}", config.endpoint);

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
            .map_err(|e| VllmError::connection_failed(&base_url, e))?;

        Ok(Self {
            client,
            config: config.clone(),
            base_url,
        })
    }

    /// Perform health check on vLLM server
    ///
    /// # Returns
    /// * `Result<BackendHealth, VllmError>` - Health status
    pub async fn health_check(&self) -> Result<BackendHealth, VllmError> {
        let url = format!("{}/health", self.base_url);
        trace!("Performing health check: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| VllmError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.ok();
            return Err(VllmError::server_error(
                status.as_u16(),
                &format!("Health check failed with status {}", status),
                body,
            ));
        }

        let health_response: VllmHealthResponse = response
            .json()
            .await
            .map_err(|e| VllmError::InvalidResponse {
                message: format!("Failed to parse health response: {}", e),
                raw_response: None,
                expected: "VllmHealthResponse".to_string(),
            })?;

        // Convert to TensorForge BackendHealth
        let backend_status = match health_response.status.as_str() {
            "healthy" | "ready" => BackendStatus::Healthy,
            "degraded" | "busy" => BackendStatus::Degraded,
            _ => BackendStatus::Unhealthy,
        };

        let loaded_models = health_response.loaded_models.unwrap_or_default();

        Ok(BackendHealth {
            status: backend_status,
            message: Some(format!("vLLM status: {}", health_response.status)),
            timestamp: chrono::Utc::now(),
            loaded_models,
        })
    }

    /// Load a model onto the vLLM server
    ///
    /// # Arguments
    /// * `request` - Model load request
    ///
    /// # Returns
    /// * `Result<LoadModelResponse, VllmError>` - Load result
    pub async fn load_model(&self, request: &LoadModelRequest) -> Result<LoadModelResponse, VllmError> {
        let url = format!("{}/v1/models/load", self.base_url);
        info!("Loading model {} to vLLM", request.model_id);

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
                .map_err(|e| backoff::Error::transient(VllmError::connection_failed(&url, e)))?;

            let status = response.status();

            if !status.is_success() {
                let body = response.text().await.ok();
                let error = VllmError::server_error(
                    status.as_u16(),
                    &format!("Model load failed with status {}", status),
                    body,
                );
                return Err(backoff::Error::transient(error));
            }

            let load_response: LoadModelResponse = response
                .json()
                .await
                .map_err(|e| backoff::Error::permanent(VllmError::InvalidResponse {
                    message: format!("Failed to parse load response: {}", e),
                    raw_response: None,
                    expected: "LoadModelResponse".to_string(),
                }))?;

            if !load_response.success {
                return Err(backoff::Error::permanent(VllmError::ModelLoadFailed {
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
                backoff::Error::Transient(err) => VllmError::timeout(
                    self.config.model_load_timeout_secs,
                    &format!("model load for {}", request.model_id),
                ),
            })
    }

    /// Unload a model from vLLM server
    ///
    /// # Arguments
    /// * `model_id` - Model identifier to unload
    ///
    /// # Returns
    /// * `Result<(), VllmError>` - Success or error
    pub async fn unload_model(&self, model_id: &str) -> Result<(), VllmError> {
        let url = format!("{}/v1/models/{}", self.base_url, model_id);
        info!("Unloading model {} from vLLM", model_id);

        let response = self
            .client
            .delete(&url)
            .send()
            .await
            .map_err(|e| VllmError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() && status != StatusCode::NOT_FOUND {
            let body = response.text().await.ok();
            return Err(VllmError::server_error(
                status.as_u16(),
                &format!("Model unload failed with status {}", status),
                body,
            ));
        }

        Ok(())
    }

    /// Perform inference with vLLM server
    ///
    /// # Arguments
    /// * `request` - Inference request
    ///
    /// # Returns
    /// * `Result<VllmInferenceResponse, VllmError>` - Inference result
    pub async fn infer(&self, request: &VllmInferenceRequest) -> Result<VllmInferenceResponse, VllmError> {
        let url = format!("{}/v1/completions", self.base_url);
        trace!("Sending inference request for model {}", request.model);

        let response = self
            .client
            .post(&url)
            .json(request)
            .timeout(self.config.request_timeout())
            .send()
            .await
            .map_err(|e| VllmError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.ok();
            return Err(VllmError::server_error(
                status.as_u16(),
                &format!("Inference failed with status {}", status),
                body,
            ));
        }

        let inference_response: VllmInferenceResponse = response
            .json()
            .await
            .map_err(|e| VllmError::InvalidResponse {
                message: format!("Failed to parse inference response: {}", e),
                raw_response: None,
                expected: "VllmInferenceResponse".to_string(),
            })?;

        Ok(inference_response)
    }

    /// Perform streaming inference with vLLM server
    ///
    /// # Arguments
    /// * `request` - Inference request with streaming enabled
    ///
    /// # Returns
    /// * `Result<Pin<Box<dyn Stream<Item = Result<VllmStreamChunk, VllmError>> + Send>>, VllmError>` - Stream of chunks
    pub async fn infer_stream(
        &self,
        request: &VllmInferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<VllmStreamChunk, VllmError>> + Send>>, VllmError> {
        let url = format!("{}/v1/completions", self.base_url);
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
            .map_err(|e| VllmError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.ok();
            return Err(VllmError::server_error(
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
                    .map_err(|e| VllmError::HttpClient { source: e })
                    .and_then(|chunk| {
                        // Parse chunk as JSON
                        let chunk_str = String::from_utf8(chunk.to_vec())
                            .map_err(|e| VllmError::InvalidResponse {
                                message: format!("Invalid UTF-8 in stream chunk: {}", e),
                                raw_response: Some(String::from_utf8_lossy(&chunk).to_string()),
                                expected: "UTF-8 string".to_string(),
                            })?;

                        // Skip empty chunks
                        if chunk_str.trim().is_empty() {
                            return Ok(None);
                        }

                        // Parse as stream chunk
                        let stream_chunk: VllmStreamChunk = serde_json::from_str(&chunk_str)
                            .map_err(|e| VllmError::InvalidResponse {
                                message: format!("Failed to parse stream chunk: {}", e),
                                raw_response: Some(chunk_str),
                                expected: "VllmStreamChunk".to_string(),
                            })?;

                        Ok(Some(stream_chunk))
                    })
                    .transpose()
            })
            .filter_map(|item| async { item });

        Ok(Box::pin(stream))
    }

    /// Get VRAM usage information from vLLM server
    ///
    /// # Returns
    /// * `Result<VramUsage, VllmError>` - VRAM usage information
    pub async fn get_vram_usage(&self) -> Result<VramUsage, VllmError> {
        let url = format!("{}/v1/gpu/memory", self.base_url);
        trace!("Querying VRAM usage from vLLM");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| VllmError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            // vLLM might not support this endpoint, fall back to health endpoint
            return self.get_vram_usage_from_health().await;
        }

        let vram_response: VllmVramUsage = response
            .json()
            .await
            .map_err(|e| VllmError::InvalidResponse {
                message: format!("Failed to parse VRAM response: {}", e),
                raw_response: None,
                expected: "VllmVramUsage".to_string(),
            })?;

        Ok(VramUsage {
            total_mb: vram_response.total_mb,
            used_mb: vram_response.used_mb,
            peak_mb: vram_response.used_mb, // Same as used for now
            utilization_percent: vram_response.utilization_percent,
        })
    }

    /// Get VRAM usage from health endpoint as fallback
    async fn get_vram_usage_from_health(&self) -> Result<VramUsage, VllmError> {
        let health = self.health_check().await?;

        // Try to extract VRAM info from health response
        let url = format!("{}/health", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| VllmError::connection_failed(&url, e))?;

        let health_response: VllmHealthResponse = response
            .json()
            .await
            .map_err(|e| VllmError::InvalidResponse {
                message: format!("Failed to parse health response for VRAM: {}", e),
                raw_response: None,
                expected: "VllmHealthResponse".to_string(),
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
        warn!("VRAM information not available from vLLM");
        Ok(VramUsage {
            total_mb: 0,
            used_mb: 0,
            peak_mb: 0,
            utilization_percent: 0.0,
        })
    }

    /// List loaded models from vLLM server
    ///
    /// # Returns
    /// * `Result<Vec<String>, VllmError>` - List of loaded model IDs
    pub async fn list_models(&self) -> Result<Vec<String>, VllmError> {
        let url = format!("{}/v1/models", self.base_url);
        trace!("Listing models from vLLM");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| VllmError::connection_failed(&url, e))?;

        let status = response.status();

        if !status.is_success() {
            // Fall back to health check for model list
            let health = self.health_check().await?;
            return Ok(health.loaded_models);
        }

        let models_response: Value = response
            .json()
            .await
            .map_err(|e| VllmError::InvalidResponse {
                message: format!("Failed to parse models response: {}", e),
                raw_response: None,
                expected: "JSON array of models".to_string(),
            })?;

        // Extract model IDs from response
        let model_ids = models_response
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_object())
                    .filter_map(|obj| obj.get("id"))
                    .filter_map(|id| id.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_else(Vec::new);

        Ok(model_ids)
    }

    /// Get base URL for this client
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get configuration
    pub fn config(&self) -> &VllmConfig {
        &self.config
    }
}

impl From<InferenceRequest> for VllmInferenceRequest {
    fn from(request: InferenceRequest) -> Self {
        VllmInferenceRequest {
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
        let config = VllmConfig::default();
        let client = VllmClient::new(&config).await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "healthy",
                "loaded_models": ["model-1", "model-2"],
                "version": "0.1.0"
            })))
            .mount(&mock_server)
            .await;

        let config = VllmConfig {
            endpoint: mock_server.uri(),
            ..Default::default()
        };

        let client = VllmClient::new(&config).await.unwrap();
        let health = client.health_check().await.unwrap();

        assert_eq!(health.status, BackendStatus::Healthy);
        assert_eq!(health.loaded_models, vec!["model-1", "model-2"]);
    }

    #[tokio::test]
    async fn test_health_check_failure() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let config = VllmConfig {
            endpoint: mock_server.uri(),
            ..Default::default()
        };

        let client = VllmClient::new(&config).await.unwrap();
        let result = client.health_check().await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(error, VllmError::ServerError { status: 500, .. }));
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

        let vllm_request = VllmInferenceRequest::from(tf_request);

        assert_eq!(vllm_request.request_id, Some("test-id".to_string()));
        assert_eq!(vllm_request.model, "test-model");
        assert_eq!(vllm_request.prompt, "Test prompt");
        assert_eq!(vllm_request.max_tokens, Some(100));
        assert_eq!(vllm_request.temperature, Some(0.7));
        assert_eq!(vllm_request.top_p, Some(0.9));
        assert_eq!(vllm_request.stop, Some(vec!["stop".to_string()]));
        assert_eq!(vllm_request.stream, Some(false));
        assert_eq!(vllm_request.n, Some(1));
        assert_eq!(vllm_request.frequency_penalty, Some(0.1));
        assert_eq!(vllm_request.presence_penalty, Some(0.1));
        assert_eq!(vllm_request.parameters, json!({"test": "param"}));
    }

    #[tokio::test]
    async fn test_list_models() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"id": "model-1", "name": "Test Model 1"},
                {"id": "model-2", "name": "Test Model 2"}
            ])))
            .mount(&mock_server)
            .await;

        let config = VllmConfig {
            endpoint: mock_server.uri(),
            ..Default::default()
        };

        let client = VllmClient::new(&config).await.unwrap();
        let models = client.list_models().await.unwrap();

        assert_eq!(models, vec!["model-1", "model-2"]);
    }

    #[tokio::test]
    async fn test_get_vram_usage() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/gpu/memory"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "total_mb": 8192,
                "used_mb": 4096,
                "free_mb": 4096,
                "utilization_percent": 50.0,
                "gpus": []
            })))
            .mount(&mock_server)
            .await;

        let config = VllmConfig {
            endpoint: mock_server.uri(),
            ..Default::default()
        };

        let client = VllmClient::new(&config).await.unwrap();
        let vram = client.get_vram_usage().await.unwrap();

        assert_eq!(vram.total_mb, 8192);
        assert_eq!(vram.used_mb, 4096);
        assert_eq!(vram.peak_mb, 4096);
        assert_eq!(vram.utilization_percent, 50.0);
    }
}
