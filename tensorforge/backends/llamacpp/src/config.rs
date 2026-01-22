//! llama.cpp backend configuration
//!
//! Configuration optimized for NVIDIA B200 GPUs with 192GB HBM3e memory,
//! featuring all-layers-on-GPU, FlashAttention, and CUDA graph optimizations.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use validator::Validate;

/// Configuration for llama.cpp backend
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LlamaCppConfig {
    /// Enable llama.cpp backend
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// llama.cpp API endpoint URL
    #[validate(url)]
    #[serde(default = "default_endpoint")]
    pub endpoint: String,

    /// Number of GPU layers to load (0 = CPU only, -1 = all layers)
    #[validate(range(min = -1, max = 1024))]
    #[serde(default = "default_n_gpu_layers")]
    pub n_gpu_layers: i32,

    /// Context length
    #[validate(range(min = 512, max = 131072))]
    #[serde(default = "default_n_ctx")]
    pub n_ctx: u32,

    /// Batch size for prompt processing
    #[validate(range(min = 1, max = 512))]
    #[serde(default = "default_n_batch")]
    pub n_batch: u32,

    /// Number of threads for processing
    #[validate(range(min = 1, max = 64))]
    #[serde(default = "default_n_threads")]
    pub n_threads: u32,

    /// Number of threads for batch processing
    #[validate(range(min = 1, max = 64))]
    #[serde(default = "default_n_threads_batch")]
    pub n_threads_batch: u32,

    /// Enable FlashAttention for faster attention computation
    #[serde(default = "default_flash_attn")]
    pub flash_attn: bool,

    /// Enable CUDA graphs for faster inference
    #[serde(default = "default_cuda_graphs")]
    pub cuda_graphs: bool,

    /// Rope scaling type (none, linear, yarn)
    #[validate(length(min = 1))]
    #[serde(default = "default_rope_scaling_type")]
    pub rope_scaling_type: String,

    /// Maximum concurrent requests
    #[validate(range(min = 1, max = 1000))]
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds
    #[validate(range(min = 1, max = 3600))]
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,

    /// Enable all-layers-on-GPU optimization (B200-specific)
    #[serde(default = "default_enable_all_layers_on_gpu")]
    pub enable_all_layers_on_gpu: bool,

    /// Enable tensor parallelism (multi-GPU)
    #[serde(default = "default_enable_tensor_parallelism")]
    pub enable_tensor_parallelism: bool,

    /// Tensor parallelism size
    #[validate(range(min = 1, max = 8))]
    #[serde(default = "default_tensor_parallelism_size")]
    pub tensor_parallelism_size: u32,

    /// VRAM allocation strategy
    #[serde(default = "default_vram_strategy")]
    pub vram_strategy: VramStrategy,

    /// Model loading timeout in seconds
    #[validate(range(min = 30, max = 3600))]
    #[serde(default = "default_model_load_timeout_secs")]
    pub model_load_timeout_secs: u64,

    /// Health check interval in seconds
    #[validate(range(min = 1, max = 300))]
    #[serde(default = "default_health_check_interval_secs")]
    pub health_check_interval_secs: u64,

    /// Enable streaming responses
    #[serde(default = "default_enable_streaming")]
    pub enable_streaming: bool,

    /// Enable embeddings support
    #[serde(default = "default_enable_embeddings")]
    pub enable_embeddings: bool,
}

/// VRAM allocation strategy for B200 GPUs
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VramStrategy {
    /// Conservative: Leave 20% VRAM free
    Conservative,
    /// Balanced: Leave 10% VRAM free
    Balanced,
    /// Aggressive: Use all available VRAM
    Aggressive,
    /// Auto: Dynamically adjust based on workload
    Auto,
}

/// Configuration errors
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Invalid endpoint URL: {0}")]
    InvalidEndpoint(String),

    #[error("Invalid GPU layers: {0}. Must be between -1 and 1024")]
    InvalidNGpuLayers(i32),

    #[error("Invalid context length: {0}. Must be between 512-131072")]
    InvalidNCtx(u32),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Unsupported rope scaling type: {0}")]
    UnsupportedRopeScaling(String),
}

// Default values
fn default_enabled() -> bool { true }
fn default_endpoint() -> String { "http://127.0.0.1:8080".to_string() }
fn default_n_gpu_layers() -> i32 { -1 } // All layers on GPU
fn default_n_ctx() -> u32 { 32768 } // 32k context
fn default_n_batch() -> u32 { 512 }
fn default_n_threads() -> u32 { 8 }
fn default_n_threads_batch() -> u32 { 8 }
fn default_flash_attn() -> bool { true }
fn default_cuda_graphs() -> bool { true }
fn default_rope_scaling_type() -> String { "none".to_string() }
fn default_max_concurrent_requests() -> usize { 64 }
fn default_request_timeout_secs() -> u64 { 300 } // 5 minutes
fn default_enable_all_layers_on_gpu() -> bool { true }
fn default_enable_tensor_parallelism() -> bool { false }
fn default_tensor_parallelism_size() -> u32 { 1 }
fn default_vram_strategy() -> VramStrategy { VramStrategy::Balanced }
fn default_model_load_timeout_secs() -> u64 { 600 } // 10 minutes
fn default_health_check_interval_secs() -> u64 { 30 }
fn default_enable_streaming() -> bool { true }
fn default_enable_embeddings() -> bool { true }

impl Default for VramStrategy {
    fn default() -> Self {
        VramStrategy::Balanced
    }
}

impl LlamaCppConfig {
    /// Create default llama.cpp configuration
    pub fn default() -> Self {
        Self {
            enabled: default_enabled(),
            endpoint: default_endpoint(),
            n_gpu_layers: default_n_gpu_layers(),
            n_ctx: default_n_ctx(),
            n_batch: default_n_batch(),
            n_threads: default_n_threads(),
            n_threads_batch: default_n_threads_batch(),
            flash_attn: default_flash_attn(),
            cuda_graphs: default_cuda_graphs(),
            rope_scaling_type: default_rope_scaling_type(),
            max_concurrent_requests: default_max_concurrent_requests(),
            request_timeout_secs: default_request_timeout_secs(),
            enable_all_layers_on_gpu: default_enable_all_layers_on_gpu(),
            enable_tensor_parallelism: default_enable_tensor_parallelism(),
            tensor_parallelism_size: default_tensor_parallelism_size(),
            vram_strategy: default_vram_strategy(),
            model_load_timeout_secs: default_model_load_timeout_secs(),
            health_check_interval_secs: default_health_check_interval_secs(),
            enable_streaming: default_enable_streaming(),
            enable_embeddings: default_enable_embeddings(),
        }
    }

    /// Create B200-optimized configuration
    ///
    /// Optimized for NVIDIA B200 GPUs with 192GB HBM3e memory:
    /// - All layers on GPU (-1)
    /// - 131k context length with rope scaling
    /// - FlashAttention and CUDA graphs enabled
    /// - Tensor parallelism across 4 dies
    /// - Aggressive VRAM strategy
    pub fn b200_optimized() -> Self {
        Self {
            n_gpu_layers: -1, // All layers on GPU for B200
            n_ctx: 131072, // Maximum context for B200
            n_batch: 1024, // Larger batches for B200 throughput
            flash_attn: true, // Enable FlashAttention
            cuda_graphs: true, // Enable CUDA graphs
            rope_scaling_type: "yarn".to_string(), // Use yarn rope scaling for long context
            enable_all_layers_on_gpu: true, // B200 can handle all layers
            enable_tensor_parallelism: true, // Utilize tensor parallelism
            tensor_parallelism_size: 4, // B200 has 4 dies
            vram_strategy: VramStrategy::Aggressive, // B200 has plenty of VRAM
            max_concurrent_requests: 128, // Higher concurrency for B200
            ..Self::default()
        }
    }

    /// Create development configuration
    ///
    /// Lower resource usage for development and testing:
    /// - Smaller batches
    /// - Conservative VRAM usage
    /// - Lower parallelism
    pub fn development() -> Self {
        Self {
            n_gpu_layers: 32, // Limited GPU layers for testing
            n_ctx: 4096, // Smaller context for testing
            n_batch: 32, // Smaller batches
            n_threads: 4, // Fewer threads
            n_threads_batch: 4,
            flash_attn: false, // Disable FlashAttention for compatibility
            cuda_graphs: false, // Disable CUDA graphs
            enable_all_layers_on_gpu: false, // Don't load all layers
            enable_tensor_parallelism: false, // No tensor parallelism
            vram_strategy: VramStrategy::Conservative, // Conservative VRAM usage
            max_concurrent_requests: 16, // Lower concurrency
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Use validator crate
        <Self as Validate>::validate(self)
            .map_err(|e| ConfigError::ValidationFailed(e.to_string()))?;

        // Additional custom validation
        if !self.endpoint.starts_with("http://") && !self.endpoint.starts_with("https://") {
            return Err(ConfigError::InvalidEndpoint(self.endpoint.clone()));
        }

        // Validate n_gpu_layers
        if self.n_gpu_layers < -1 || self.n_gpu_layers > 1024 {
            return Err(ConfigError::InvalidNGpuLayers(self.n_gpu_layers));
        }

        // Validate n_ctx
        if self.n_ctx < 512 || self.n_ctx > 131072 {
            return Err(ConfigError::InvalidNCtx(self.n_ctx));
        }

        // Validate rope scaling type
        let valid_rope_types = ["none", "linear", "yarn"];
        if !valid_rope_types.contains(&self.rope_scaling_type.as_str()) {
            return Err(ConfigError::UnsupportedRopeScaling(self.rope_scaling_type.clone()));
        }

        Ok(())
    }

    /// Get request timeout as Duration
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }

    /// Get model load timeout as Duration
    pub fn model_load_timeout(&self) -> Duration {
        Duration::from_secs(self.model_load_timeout_secs)
    }

    /// Get health check interval as Duration
    pub fn health_check_interval(&self) -> Duration {
        Duration::from_secs(self.health_check_interval_secs)
    }

    /// Check if all layers are configured to be on GPU
    pub fn is_all_layers_on_gpu(&self) -> bool {
        self.enable_all_layers_on_gpu && self.n_gpu_layers == -1
    }

    /// Calculate estimated VRAM usage for a model
    ///
    /// # Arguments
    /// * `parameter_count` - Number of model parameters in billions
    /// * `context_length` - Desired context length
    ///
    /// # Returns
    /// * Estimated VRAM usage in GB
    pub fn estimate_vram_usage(&self, parameter_count: f32, context_length: u32) -> f32 {
        // Base formula: ~2 * parameters (in GB) for FP16
        let base_vram_gb = parameter_count * 2.0;

        // Adjust for context length (kv cache)
        let context_factor = (context_length as f32 / 2048.0).min(64.0); // Cap at 64x

        // Adjust for GPU layers (if not all layers)
        let layers_factor = if self.n_gpu_layers == -1 {
            1.0 // All layers
        } else if self.n_gpu_layers == 0 {
            0.0 // No GPU layers
        } else {
            // Estimate based on typical layer distribution
            (self.n_gpu_layers as f32 / 100.0).min(1.0)
        };

        // Tensor parallelism reduces per-GPU memory
        let parallelism_factor = 1.0 / self.tensor_parallelism_size as f32;

        (base_vram_gb * context_factor * layers_factor * parallelism_factor).ceil()
    }

    /// Get recommended GPU layers based on available VRAM
    pub fn recommended_gpu_layers(&self, available_vram_gb: f32, model_vram_gb: f32) -> i32 {
        if self.enable_all_layers_on_gpu && available_vram_gb >= model_vram_gb * 1.2 {
            -1 // All layers if enough VRAM
        } else {
            // Calculate based on available VRAM
            let layers_per_gb = 10.0; // Rough estimate: 10 layers per GB
            let available_for_layers = available_vram_gb - model_vram_gb;
            let recommended_layers = (available_for_layers * layers_per_gb) as i32;

            recommended_layers.clamp(0, 100) // Cap at reasonable number
        }
    }
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self::default()
    }
}

impl From<tensorforge_core::config::LlamaCppConfig> for LlamaCppConfig {
    fn from(core_config: tensorforge_core::config::LlamaCppConfig) -> Self {
        Self {
            enabled: core_config.enabled,
            endpoint: core_config.endpoint,
            n_gpu_layers: core_config.n_gpu_layers as i32,
            n_ctx: core_config.n_ctx,
            n_batch: core_config.n_batch,
            n_threads: core_config.n_threads,
            n_threads_batch: core_config.n_threads_batch,
            flash_attn: core_config.flash_attn,
            cuda_graphs: core_config.cuda_graphs,
            rope_scaling_type: core_config.rope_scaling_type,
            max_concurrent_requests: core_config.max_concurrent_requests,
            request_timeout_secs: core_config.request_timeout_secs,
            // Default values for fields not present in core config
            enable_all_layers_on_gpu: default_enable_all_layers_on_gpu(),
            enable_tensor_parallelism: default_enable_tensor_parallelism(),
            tensor_parallelism_size: default_tensor_parallelism_size(),
            vram_strategy: default_vram_strategy(),
            model_load_timeout_secs: default_model_load_timeout_secs(),
            health_check_interval_secs: default_health_check_interval_secs(),
            enable_streaming: default_enable_streaming(),
            enable_embeddings: default_enable_embeddings(),
        }
    }
}

/// Builder for LlamaCppConfig with fluent interface
#[derive(Debug, Default)]
pub struct LlamaCppConfigBuilder {
    config: LlamaCppConfig,
}

impl LlamaCppConfigBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: LlamaCppConfig::default(),
        }
    }

    /// Set endpoint URL
    pub fn endpoint(mut self, endpoint: &str) -> Self {
        self.config.endpoint = endpoint.to_string();
        self
    }

    /// Set GPU layers
    pub fn n_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.config.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Set context length
    pub fn n_ctx(mut self, n_ctx: u32) -> Self {
        self.config.n_ctx = n_ctx;
        self
    }

    /// Set batch size
    pub fn n_batch(mut self, n_batch: u32) -> Self {
        self.config.n_batch = n_batch;
        self
    }

    /// Enable/disable FlashAttention
    pub fn flash_attn(mut self, enabled: bool) -> Self {
        self.config.flash_attn = enabled;
        self
    }

    /// Enable/disable CUDA graphs
    pub fn cuda_graphs(mut self, enabled: bool) -> Self {
        self.config.cuda_graphs = enabled;
        self
    }

    /// Set rope scaling type
    pub fn rope_scaling_type(mut self, rope_type: &str) -> Self {
        self.config.rope_scaling_type = rope_type.to_string();
        self
    }

    /// Enable all-layers-on-GPU optimization
    pub fn enable_all_layers_on_gpu(mut self, enabled: bool) -> Self {
        self.config.enable_all_layers_on_gpu = enabled;
        if enabled {
            self.config.n_gpu_layers = -1;
        }
        self
    }

    /// Enable tensor parallelism
    pub fn enable_tensor_parallelism(mut self, enabled: bool, size: u32) -> Self {
        self.config.enable_tensor_parallelism = enabled;
        self.config.tensor_parallelism_size = size;
        self
    }

    /// Set VRAM strategy
    pub fn vram_strategy(mut self, strategy: VramStrategy) -> Self {
        self.config.vram_strategy = strategy;
        self
    }

    /// Build configuration with validation
    pub fn build(self) -> Result<LlamaCppConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlamaCppConfig::default();
        assert!(config.enabled);
        assert_eq!(config.endpoint, "http://127.0.0.1:8080");
        assert_eq!(config.n_gpu_layers, -1);
        assert_eq!(config.n_ctx, 32768);
        assert_eq!(config.n_batch, 512);
        assert_eq!(config.n_threads, 8);
        assert_eq!(config.n_threads_batch, 8);
        assert!(config.flash_attn);
        assert!(config.cuda_graphs);
        assert_eq!(config.rope_scaling_type, "none");
        assert_eq!(config.max_concurrent_requests, 64);
        assert_eq!(config.request_timeout_secs, 300);
        assert!(config.enable_all_layers_on_gpu);
        assert!(!config.enable_tensor_parallelism);
        assert_eq!(config.tensor_parallelism_size, 1);
        assert_eq!(config.vram_strategy, VramStrategy::Balanced);
        assert_eq!(config.model_load_timeout_secs, 600);
        assert_eq!(config.health_check_interval_secs, 30);
        assert!(config.enable_streaming);
        assert!(config.enable_embeddings);
    }

    #[test]
    fn test_b200_optimized_config() {
        let config = LlamaCppConfig::b200_optimized();
        assert_eq!(config.n_gpu_layers, -1);
        assert_eq!(config.n_ctx, 131072);
        assert_eq!(config.n_batch, 1024);
        assert!(config.flash_attn);
        assert!(config.cuda_graphs);
        assert_eq!(config.rope_scaling_type, "yarn");
        assert!(config.enable_all_layers_on_gpu);
        assert!(config.enable_tensor_parallelism);
        assert_eq!(config.tensor_parallelism_size, 4);
        assert_eq!(config.vram_strategy, VramStrategy::Aggressive);
        assert_eq!(config.max_concurrent_requests, 128);
    }

    #[test]
    fn test_development_config() {
        let config = LlamaCppConfig::development();
        assert_eq!(config.n_gpu_layers, 32);
        assert_eq!(config.n_ctx, 4096);
        assert_eq!(config.n_batch, 32);
        assert_eq!(config.n_threads, 4);
        assert_eq!(config.n_threads_batch, 4);
        assert!(!config.flash_attn);
        assert!(!config.cuda_graphs);
        assert!(!config.enable_all_layers_on_gpu);
        assert!(!config.enable_tensor_parallelism);
        assert_eq!(config.vram_strategy, VramStrategy::Conservative);
        assert_eq!(config.max_concurrent_requests, 16);
    }

    #[test]
    fn test_config_validation() {
        let config = LlamaCppConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_failure() {
        let mut config = LlamaCppConfig::default();
        config.endpoint = "invalid-url".to_string();
        assert!(config.validate().is_err());

        let mut config = LlamaCppConfig::default();
        config.n_gpu_layers = 2000;
        assert!(config.validate().is_err());

        let mut config = LlamaCppConfig::default();
        config.rope_scaling_type = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vram_estimation() {
        let config = LlamaCppConfig::default();

        // Test 7B model with 4k context
        let vram_gb = config.estimate_vram_usage(7.0, 4096);
        assert!(vram_gb > 0.0);

        // Test with fewer GPU layers
        let mut config_layers = LlamaCppConfig::default();
        config_layers.n_gpu_layers = 32;
        let vram_layers_gb = config_layers.estimate_vram_usage(7.0, 4096);
        assert!(vram_layers_gb < vram_gb); // Fewer layers should use less VRAM
    }

    #[test]
    fn test_builder_pattern() {
        let config = LlamaCppConfigBuilder::new()
            .endpoint("http://localhost:8888")
            .n_gpu_layers(48)
            .n_ctx(65536)
            .flash_attn(true)
            .cuda_graphs(true)
            .enable_all_layers_on_gpu(false)
            .enable_tensor_parallelism(true, 2)
            .vram_strategy(VramStrategy::Aggressive)
            .build()
            .unwrap();

        assert_eq!(config.endpoint, "http://localhost:8888");
        assert_eq!(config.n_gpu_layers, 48);
        assert_eq!(config.n_ctx, 65536);
        assert!(config.flash_attn);
        assert!(config.cuda_graphs);
        assert!(!config.enable_all_layers_on_gpu);
        assert!(config.enable_tensor_parallelism);
        assert_eq!(config.tensor_parallelism_size, 2);
        assert_eq!(config.vram_strategy, VramStrategy::Aggressive);
    }

    #[test]
    fn test_is_all_layers_on_gpu() {
        let mut config = LlamaCppConfig::default();
        config.n_gpu_layers = -1;
        config.enable_all_layers_on_gpu = true;
        assert!(config.is_all_layers_on_gpu());

        config.n_gpu_layers = 32;
        assert!(!config.is_all_layers_on_gpu());

        config.enable_all_layers_on_gpu = false;
        config.n_gpu_layers = -1;
        assert!(!config.is_all_layers_on_gpu());
    }

    #[test]
    fn test_vram_strategy_default() {
        let strategy = VramStrategy::default();
        assert_eq!(strategy, VramStrategy::Balanced);
    }
}
