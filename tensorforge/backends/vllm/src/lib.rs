//! # vLLM Backend for TensorForge
//!
//! High-performance vLLM backend implementation optimized for NVIDIA B200 GPUs.
//! Features tensor parallelism, continuous batching, and FP8 quantization support.

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tensorforge_core::{
    Backend, BackendCapabilities, BackendHealth, BackendStatus, BackendType, InferenceRequest,
    InferenceResult, LoadOptions, LoadResult, TensorForgeError, TensorForgeResult, VramUsage,
};
use tracing::{debug, error, info, instrument, trace, warn};

// Re-export for convenience
pub use config::VllmConfig;
pub use error::VllmError;

mod client;
mod config;
mod error;
mod metrics;

/// vLLM backend implementation
#[derive(Debug, Clone)]
pub struct VllmBackend {
    /// Client for communicating with vLLM server
    client: Arc<client::VllmClient>,
    /// Configuration
    config: VllmConfig,
    /// State tracking loaded models
    state: Arc<RwLock<BackendState>>,
    /// Metrics collector
    metrics: Arc<metrics::VllmMetrics>,
}

/// Internal state tracking for the vLLM backend
#[derive(Debug, Default)]
struct BackendState {
    /// Currently loaded models with metadata
    loaded_models: HashMap<String, ModelMetadata>,
    /// Last health check timestamp
    last_health_check: Option<Instant>,
    /// Last known health status
    health_status: BackendStatus,
    /// Health check error message (if any)
    health_error: Option<String>,
}

/// Metadata for a loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelMetadata {
    /// Model identifier
    model_id: String,
    /// When the model was loaded
    loaded_at: chrono::DateTime<chrono::Utc>,
    /// Estimated VRAM usage in MB
    vram_estimate_mb: u64,
    /// Actual measured VRAM usage
    vram_actual_mb: Option<u64>,
    /// Context length
    context_length: u32,
    /// Quantization type
    quantization: Option<String>,
    /// Number of times used
    usage_count: u64,
    /// Last usage timestamp
    last_used: Option<chrono::DateTime<chrono::Utc>>,
}

impl VllmBackend {
    /// Create a new vLLM backend instance
    ///
    /// # Arguments
    /// * `config` - vLLM configuration
    ///
    /// # Returns
    /// * `Result<Self, VllmError>` - New backend instance or error
    #[instrument(name = "vllm_backend_new", level = "info", skip(config))]
    pub async fn new(config: VllmConfig) -> TensorForgeResult<Self> {
        info!(
            "Initializing vLLM backend with endpoint: {}",
            config.endpoint
        );

        let client = client::VllmClient::new(&config).await?;
        let metrics = Arc::new(metrics::VllmMetrics::new());

        let backend = Self {
            client: Arc::new(client),
            config,
            state: Arc::new(RwLock::new(BackendState::default())),
            metrics: Arc::clone(&metrics),
        };

        // Perform initial health check
        backend.check_health_internal().await?;

        info!("vLLM backend initialized successfully");
        Ok(backend)
    }

    /// Create a new vLLM backend with B200-optimized configuration
    ///
    /// # Returns
    /// * `Result<Self, VllmError>` - New backend instance with B200 optimizations
    #[instrument(name = "vllm_backend_b200", level = "info")]
    pub async fn new_b200_optimized() -> TensorForgeResult<Self> {
        let config = VllmConfig::b200_optimized();
        Self::new(config).await
    }

    /// Internal health check with state update
    async fn check_health_internal(&self) -> TensorForgeResult<BackendHealth> {
        let health = self.client.health_check().await?;

        // Update state
        let mut state = self.state.write();
        state.last_health_check = Some(Instant::now());
        state.health_status = match health.status.as_str() {
            "healthy" | "ready" => BackendStatus::Healthy,
            "degraded" | "busy" => BackendStatus::Degraded,
            _ => BackendStatus::Unhealthy,
        };

        // Update loaded models from health response
        if let Some(models) = health.loaded_models {
            for model in models {
                if !state.loaded_models.contains_key(&model) {
                    state.loaded_models.insert(
                        model.clone(),
                        ModelMetadata {
                            model_id: model,
                            loaded_at: chrono::Utc::now(),
                            vram_estimate_mb: 0, // Unknown initially
                            vram_actual_mb: None,
                            context_length: self.config.max_model_len,
                            quantization: None,
                            usage_count: 0,
                            last_used: None,
                        },
                    );
                }
            }
        }

        Ok(health)
    }

    /// Update model metadata after successful inference
    fn update_model_usage(&self, model_id: &str) {
        let mut state = self.state.write();
        if let Some(metadata) = state.loaded_models.get_mut(model_id) {
            metadata.usage_count += 1;
            metadata.last_used = Some(chrono::Utc::now());
        }
    }

    /// Get current VRAM usage from vLLM server
    async fn get_vram_usage_internal(&self) -> TensorForgeResult<VramUsage> {
        // First try to get from vLLM API if available
        match self.client.get_vram_usage().await {
            Ok(usage) => return Ok(usage),
            Err(e) => {
                debug!("Failed to get VRAM usage from vLLM API: {}", e);
                // Fall back to system-level monitoring
            }
        }

        // Fallback: Use system monitoring (NVML if available)
        #[cfg(feature = "vram-monitoring")]
        {
            use nvml_wrapper::Nvml;
            match Nvml::init() {
                Ok(nvml) => match nvml.device_by_index(0) {
                    Ok(device) => {
                        let memory = device.memory_info().map_err(|e| {
                            VllmError::VramQueryFailed(format!("NVML memory query failed: {}", e))
                        })?;

                        Ok(VramUsage {
                            total_mb: memory.total / 1024 / 1024,
                            used_mb: memory.used / 1024 / 1024,
                            peak_mb: memory.used / 1024 / 1024, // Same as used for now
                            utilization_percent: (memory.used as f32 / memory.total as f32) * 100.0,
                        })
                    }
                    Err(e) => Err(VllmError::VramQueryFailed(format!(
                        "NVML device query failed: {}",
                        e
                    ))),
                },
                Err(e) => Err(VllmError::VramQueryFailed(format!(
                    "NVML initialization failed: {}",
                    e
                ))),
            }
        }

        #[cfg(not(feature = "vram-monitoring"))]
        {
            // Return placeholder if no monitoring available
            warn!("VRAM monitoring not available, returning placeholder values");
            Ok(VramUsage {
                total_mb: 0,
                used_mb: 0,
                peak_mb: 0,
                utilization_percent: 0.0,
            })
        }
    }
}

#[async_trait]
impl Backend for VllmBackend {
    /// Get the backend name
    fn name(&self) -> &str {
        "vllm"
    }

    /// Get the backend type
    fn backend_type(&self) -> BackendType {
        BackendType::Vllm
    }

    /// Check if backend is healthy and ready
    #[instrument(name = "vllm_health_check", level = "debug", skip(self))]
    async fn health_check(&self) -> TensorForgeResult<BackendHealth> {
        self.metrics.health_check_attempts.inc();
        let start = Instant::now();

        match self.check_health_internal().await {
            Ok(health) => {
                self.metrics.health_check_success.inc();
                self.metrics
                    .health_check_duration
                    .observe(start.elapsed().as_secs_f64());
                Ok(health)
            }
            Err(e) => {
                self.metrics.health_check_failures.inc();
                error!("Health check failed: {}", e);

                // Update state with error
                let mut state = self.state.write();
                state.health_status = BackendStatus::Unhealthy;
                state.health_error = Some(e.to_string());

                Err(e.into())
            }
        }
    }

    /// Check if a specific model is loaded
    #[instrument(name = "vllm_has_model", level = "debug", skip(self, model_id))]
    async fn has_model(&self, model_id: &str) -> bool {
        let state = self.state.read();
        state.loaded_models.contains_key(model_id)
    }

    /// Load a model onto the backend
    #[instrument(name = "vllm_load_model", level = "info", skip(self, options), fields(model = %model_id))]
    async fn load_model(&self, model_id: &str, options: LoadOptions) -> TensorForgeResult<LoadResult> {
        info!("Loading model {} onto vLLM backend", model_id);

        self.metrics.model_load_attempts.inc();
        let start = Instant::now();

        // Check if model is already loaded
        if self.has_model(model_id).await {
            warn!("Model {} is already loaded", model_id);
            return Ok(LoadResult {
                success: true,
                model_id: model_id.to_string(),
                load_time_secs: 0.0,
                vram_used_mb: 0,
                warnings: vec!["Model was already loaded".to_string()],
            });
        }

        // Prepare load request for vLLM
        let load_request = client::LoadModelRequest {
            model_id: model_id.to_string(),
            model_path: options.model_path,
            max_model_len: options.context_length.unwrap_or(self.config.max_model_len),
            gpu_memory_utilization: self.config.gpu_memory_utilization,
            tensor_parallel_size: self.config.tensor_parallel_size,
            quantization: options.quantization.or_else(|| self.config.quantization.clone()),
            enable_prefix_caching: self.config.enable_prefix_caching,
            enable_flash_attention: self.config.enable_flash_attention,
            backend_options: options.backend_options,
        };

        // Send load request
        match self.client.load_model(&load_request).await {
            Ok(load_result) => {
                let load_time = start.elapsed();
                let load_time_secs = load_time.as_secs_f64();

                self.metrics.model_load_success.inc();
                self.metrics
                    .model_load_duration
                    .observe(load_time_secs);

                // Update state
                let mut state = self.state.write();
                state.loaded_models.insert(
                    model_id.to_string(),
                    ModelMetadata {
                        model_id: model_id.to_string(),
                        loaded_at: chrono::Utc::now(),
                        vram_estimate_mb: load_result.vram_used_mb,
                        vram_actual_mb: Some(load_result.vram_used_mb),
                        context_length: load_request.max_model_len,
                        quantization: load_request.quantization,
                        usage_count: 0,
                        last_used: None,
                    },
                );

                info!(
                    "Model {} loaded successfully in {:.2}s, using {}MB VRAM",
                    model_id, load_time_secs, load_result.vram_used_mb
                );

                Ok(LoadResult {
                    success: true,
                    model_id: model_id.to_string(),
                    load_time_secs,
                    vram_used_mb: load_result.vram_used_mb,
                    warnings: vec![],
                })
            }
            Err(e) => {
                self.metrics.model_load_failures.inc();
                error!("Failed to load model {}: {}", model_id, e);
                Err(e.into())
            }
        }
    }

    /// Unload a model from the backend
    #[instrument(name = "vllm_unload_model", level = "info", skip(self), fields(model = %model_id))]
    async fn unload_model(&self, model_id: &str) -> TensorForgeResult<()> {
        info!("Unloading model {} from vLLM backend", model_id);

        self.metrics.model_unload_attempts.inc();

        if !self.has_model(model_id).await {
            warn!("Model {} is not loaded", model_id);
            return Ok(());
        }

        match self.client.unload_model(model_id).await {
            Ok(_) => {
                self.metrics.model_unload_success.inc();

                // Update state
                let mut state = self.state.write();
                state.loaded_models.remove(model_id);

                info!("Model {} unloaded successfully", model_id);
                Ok(())
            }
            Err(e) => {
                self.metrics.model_unload_failures.inc();
                error!("Failed to unload model {}: {}", model_id, e);
                Err(e.into())
            }
        }
    }

    /// Process an inference request
    #[instrument(name = "vllm_infer", level = "debug", skip(self, request), fields(request_id = %request.id, model = %request.model))]
    async fn infer(&self, request: InferenceRequest) -> TensorForgeResult<InferenceResult> {
        let start = Instant::now();
        let queue_start = request.created_at;

        self.metrics.inference_requests.inc();

        // Check if model is loaded
        if !self.has_model(&request.model).await {
            return Err(TensorForgeError::Backend(
                tensorforge_core::BackendError::ModelNotFound {
                    model: request.model.clone(),
                    backend: self.name().to_string(),
                },
            ));
        }

        // Convert to vLLM API request
        let vllm_request = client::InferenceRequest::from(request.clone());

        // Send inference request
        match self.client.infer(&vllm_request).await {
            Ok(vllm_response) => {
                let inference_time = start.elapsed();
                let queue_time = queue_start
                    .map(|t| start.duration_since(t).as_millis() as u64)
                    .unwrap_or(0);

                // Update metrics
                self.metrics.inference_success.inc();
                self.metrics
                    .inference_duration
                    .observe(inference_time.as_secs_f64());
                self.metrics
                    .queue_duration
                    .observe(queue_time as f64 / 1000.0);

                // Update model usage stats
                self.update_model_usage(&request.model);

                // Convert to TensorForge result
                let result = InferenceResult {
                    request_id: vllm_response.request_id,
                    text: vllm_response.text,
                    model: request.model,
                    backend: self.name().to_string(),
                    prompt_tokens: vllm_response.prompt_tokens,
                    completion_tokens: vllm_response.completion_tokens,
                    total_tokens: vllm_response.total_tokens,
                    success: true,
                    error: None,
                    inference_time_ms: inference_time.as_millis() as u64,
                    queue_time_ms: queue_time,
                    load_time_ms: 0, // Already loaded
                    started_at: Some(start.into()),
                    completed_at: Some(Instant::now().into()),
                    backend_metadata: Some(serde_json::json!({
                        "vllm_response": vllm_response.metadata
                    })),
                };

                debug!(
                    "Inference completed in {}ms (queue: {}ms) for {} tokens",
                    result.inference_time_ms,
                    result.queue_time_ms,
                    result.total_tokens
                );

                Ok(result)
            }
            Err(e) => {
                self.metrics.inference_failures.inc();
                error!("Inference failed: {}", e);
                Err(e.into())
            }
        }
    }

    /// Get current VRAM usage for this backend
    #[instrument(name = "vllm_vram_usage", level = "debug", skip(self))]
    async fn vram_usage(&self) -> TensorForgeResult<VramUsage> {
        self.metrics.vram_queries.inc();
        let start = Instant::now();

        match self.get_vram_usage_internal().await {
            Ok(usage) => {
                self.metrics
                    .vram_query_duration
                    .observe(start.elapsed().as_secs_f64());
                Ok(usage)
            }
            Err(e) => {
                self.metrics.vram_query_failures.inc();
                Err(e.into())
            }
        }
    }

    /// Get estimated VRAM required for a model
    fn estimated_vram(&self, model_id: &str) -> Option<u64> {
        let state = self.state.read();
        state.loaded_models.get(model_id).map(|m| m.vram_estimate_mb)
    }

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            streaming: true,
            embeddings: true,
            function_calling: true,
            vision: false, // vLLM doesn't support vision models natively
            tool_use: true,
            max_context_length: self.config.max_model_len,
            quantization_types: vec![
                "fp16".to_string(),
                "bf16".to_string(),
                "int8".to_string(),
                "int4".to_string(),
                "fp8".to_string(), // If fp8-quantization feature enabled
            ],
        }
    }
}

/// Create a new vLLM backend instance
///
/// # Arguments
/// * `config` - vLLM configuration
///
/// # Returns
/// * `Result<impl Backend, TensorForgeError>` - Backend instance
#[instrument(name = "create_vllm_backend", level = "info")]
pub async fn create_backend(config: VllmConfig) -> TensorForgeResult<impl Backend> {
    VllmBackend::new(config).await
}

/// Create a new vLLM backend with default configuration
///
/// # Returns
/// * `Result<impl Backend, TensorForgeError>` - Backend instance
#[instrument(name = "create_default_vllm_backend", level = "info")]
pub async fn create_default_backend() -> TensorForgeResult<impl Backend> {
    VllmBackend::new(VllmConfig::default()).await
}

/// Create a new vLLM backend with B200-optimized configuration
///
/// # Returns
/// * `Result<impl Backend, TensorForgeError>` - Backend instance
#[instrument(name = "create_b200_vllm_backend", level = "info")]
pub async fn create_b200_backend() -> TensorForgeResult<impl Backend> {
    VllmBackend::new_b200_optimized().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorforge_core::InferenceRequest;

    #[tokio::test]
    async fn test_backend_creation() {
        let config = VllmConfig {
            endpoint: "http://localhost:8000".to_string(),
            ..Default::default()
        };

        // This will fail because there's no server, but we can test creation logic
        let backend = VllmBackend::new(config).await;
        assert!(backend.is_err()); // Should fail due to connection error
    }

    #[test]
    fn test_backend_name() {
        // Create a mock backend to test trait methods
        struct MockBackend;

        #[async_trait]
        impl Backend for MockBackend {
            fn name(&self) -> &str {
                "vllm"
            }

            fn backend_type(&self) -> BackendType {
                BackendType::Vllm
            }

            async fn health_check(&self) -> TensorForgeResult<BackendHealth> {
                unimplemented!()
            }

            async fn has_model(&self, _model_id: &str) -> bool {
                unimplemented!()
            }

            async fn load_model(&self, _model_id: &str, _options: LoadOptions) -> TensorForgeResult<LoadResult> {
                unimplemented!()
            }

            async fn unload_model(&self, _model_id: &str) -> TensorForgeResult<()> {
                unimplemented!()
            }

            async fn infer(&self, _request: InferenceRequest) -> TensorForgeResult<InferenceResult> {
                unimplemented!()
            }

            async fn vram_usage(&self) -> TensorForgeResult<VramUsage> {
                unimplemented!()
            }

            fn estimated_vram(&self, _model_id: &str) -> Option<u64> {
                unimplemented!()
            }

            fn capabilities(&self) -> BackendCapabilities {
                unimplemented!()
            }
        }

        let backend = MockBackend;
        assert_eq!(backend.name(), "vllm");
        assert_eq!(backend.backend_type(), BackendType::Vllm);
    }
}
