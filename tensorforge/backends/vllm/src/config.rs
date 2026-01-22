//! vLLM backend configuration
//!
//! Configuration optimized for NVIDIA B200 GPUs with 192GB HBM3e memory,
//! featuring tensor parallelism, FP8 quantization, and continuous batching.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use validator::Validate;

/// Configuration for vLLM backend
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct VllmConfig {
    /// Enable vLLM backend
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// vLLM API endpoint URL
    #[validate(url)]
    #[serde(default = "default_endpoint")]
    pub endpoint: String,

    /// Tensor parallel size for B200 4-die architecture
    #[validate(range(min = 1, max = 8))]
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: u32,

    /// Pipeline parallel size (for very large models)
    #[validate(range(min = 1, max = 4))]
    #[serde(default = "default_pipeline_parallel_size")]
    pub pipeline_parallel_size: u32,

    /// Maximum model context length
    #[validate(range(min = 1024, max = 131072))]
    #[serde(default = "default_max_model_len")]
    pub max_model_len: u32,

    /// GPU memory utilization (0.0-1.0)
    #[validate(range(min = 0.1, max = 1.0))]
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f32,

    /// Enable prefix caching for repeated prompts
    #[serde(default = "default_enable_prefix_caching")]
    pub enable_prefix_caching: bool,

    /// Enable FlashAttention for faster attention computation
    #[serde(default = "default_enable_flash_attention")]
    pub enable_flash_attention: bool,

    /// Quantization type (fp16, bf16, int8, int4, fp8)
    #[validate(length(min = 1))]
    #[serde(default = "default_quantization")]
    pub quantization: String,

    /// Maximum concurrent requests per GPU
    #[validate(range(min = 1, max = 1000))]
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds
    #[validate(range(min = 1, max = 3600))]
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,

    /// Enable continuous batching
    #[serde(default = "default_enable_continuous_batching")]
    pub enable_continuous_batching: bool,

    /// Batch size for continuous batching
    #[validate(range(min = 1, max = 256))]
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,

    /// Enable speculative decoding
    #[serde(default = "default_enable_speculative_decoding")]
    pub enable_speculative_decoding: bool,

    /// Enable FP8 quantization (B200-specific optimization)
    #[serde(default = "default_enable_fp8")]
    pub enable_fp8: bool,

    /// Enable all-layers-on-GPU optimization
    #[serde(default = "default_enable_all_layers_on_gpu")]
    pub enable_all_layers_on_gpu: bool,

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

    #[error("Invalid tensor parallel size: {0}. Must be between 1-8")]
    InvalidTensorParallelSize(u32),

    #[error("Invalid GPU memory utilization: {0}. Must be between 0.1-1.0")]
    InvalidGpuMemoryUtilization(f32),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Unsupported quantization type: {0}")]
    UnsupportedQuantization(String),
}

// Default values
fn default_enabled() -> bool { true }
fn default_endpoint() -> String { "http://127.0.0.1:8000".to_string() }
fn default_tensor_parallel_size() -> u32 { 4 } // B200 has 4 dies
fn default_pipeline_parallel_size() -> u32 { 1 }
fn default_max_model_len() -> u32 { 131072 } // 128k context
fn default_gpu_memory_utilization() -> f32 { 0.9 } // Use 90% of VRAM
fn default_enable_prefix_caching() -> bool { true }
fn default_enable_flash_attention() -> bool { true }
fn default_quantization() -> String { "fp16".to_string() }
fn default_max_concurrent_requests() -> usize { 256 }
fn default_request_timeout_secs() -> u64 { 300 } // 5 minutes
fn default_enable_continuous_batching() -> bool { true }
fn default_batch_size() -> u32 { 128 }
fn default_enable_speculative_decoding() -> bool { false }
fn default_enable_fp8() -> bool { false }
fn default_enable_all_layers_on_gpu() -> bool { true }
fn default_vram_strategy() -> VramStrategy { VramStrategy::Balanced }
fn default_model_load_timeout_secs() -> u64 { 600 } // 10 minutes
fn default_health_check_interval_secs() -> u64 { 30 }

impl Default for VramStrategy {
    fn default() -> Self {
        VramStrategy::Balanced
    }
}

impl VllmConfig {
    /// Create default vLLM configuration
    pub fn default() -> Self {
        Self {
            enabled: default_enabled(),
            endpoint: default_endpoint(),
            tensor_parallel_size: default_tensor_parallel_size(),
            pipeline_parallel_size: default_pipeline_parallel_size(),
            max_model_len: default_max_model_len(),
            gpu_memory_utilization: default_gpu_memory_utilization(),
            enable_prefix_caching: default_enable_prefix_caching(),
            enable_flash_attention: default_enable_flash_attention(),
            quantization: default_quantization(),
            max_concurrent_requests: default_max_concurrent_requests(),
            request_timeout_secs: default_request_timeout_secs(),
            enable_continuous_batching: default_enable_continuous_batching(),
            batch_size: default_batch_size(),
            enable_speculative_decoding: default_enable_speculative_decoding(),
            enable_fp8: default_enable_fp8(),
            enable_all_layers_on_gpu: default_enable_all_layers_on_gpu(),
            vram_strategy: default_vram_strategy(),
            model_load_timeout_secs: default_model_load_timeout_secs(),
            health_check_interval_secs: default_health_check_interval_secs(),
        }
    }

    /// Create B200-optimized configuration
    ///
    /// Optimized for NVIDIA B200 GPUs with 192GB HBM3e memory:
    /// - Tensor parallelism across 4 dies
    /// - FP8 quantization support
    /// - 131k context length
    /// - Continuous batching with 128 batch size
    /// - All layers on GPU
    pub fn b200_optimized() -> Self {
        Self {
            tensor_parallel_size: 4, // Utilize all 4 B200 dies
            max_model_len: 131072, // Maximum context for B200
            gpu_memory_utilization: 0.95, // Aggressive for B200 large memory
            enable_fp8: true, // Enable FP8 for B200
            enable_all_layers_on_gpu: true, // Keep all layers on GPU
            vram_strategy: VramStrategy::Aggressive, // B200 has plenty of VRAM
            batch_size: 256, // Larger batches for B200 throughput
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
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            max_model_len: 8192,
            gpu_memory_utilization: 0.7,
            enable_prefix_caching: false,
            max_concurrent_requests: 32,
            batch_size: 16,
            enable_speculative_decoding: false,
            enable_fp8: false,
            vram_strategy: VramStrategy::Conservative,
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

        // Validate quantization type
        let valid_quantizations = ["fp16", "bf16", "int8", "int4", "fp8", "none"];
        if !valid_quantizations.contains(&self.quantization.as_str()) {
            return Err(ConfigError::UnsupportedQuantization(self.quantization.clone()));
        }

        // Validate tensor parallel size
        if self.tensor_parallel_size < 1 || self.tensor_parallel_size > 8 {
            return Err(ConfigError::InvalidTensorParallelSize(self.tensor_parallel_size));
        }

        // Validate GPU memory utilization
        if self.gpu_memory_utilization < 0.1 || self.gpu_memory_utilization > 1.0 {
            return Err(ConfigError::InvalidGpuMemoryUtilization(self.gpu_memory_utilization));
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

    /// Check if FP8 quantization is enabled and supported
    pub fn is_fp8_enabled(&self) -> bool {
        self.enable_fp8 && self.quantization == "fp8"
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

        // Adjust for quantization
        let quantization_factor = match self.quantization.as_str() {
            "fp16" | "bf16" => 1.0,
            "int8" => 0.5,
            "int4" => 0.25,
            "fp8" => 0.5,
            _ => 1.0,
        };

        // Adjust for context length (kv cache)
        let context_factor = (context_length as f32 / 2048.0).min(64.0); // Cap at 64x

        // Tensor parallelism reduces per-GPU memory
        let parallelism_factor = 1.0 / self.tensor_parallel_size as f32;

        (base_vram_gb * quantization_factor * context_factor * parallelism_factor).ceil()
    }

    /// Get recommended batch size based on VRAM strategy
    pub fn recommended_batch_size(&self, available_vram_gb: f32, model_vram_gb: f32) -> u32 {
        let usable_vram_gb = match self.vram_strategy {
            VramStrategy::Conservative => available_vram_gb * 0.8,
            VramStrategy::Balanced => available_vram_gb * 0.9,
            VramStrategy::Aggressive => available_vram_gb * 0.95,
            VramStrategy::Auto => available_vram_gb * 0.85, // Slightly conservative for auto
        };

        let remaining_vram_gb = usable_vram_gb - model_vram_gb;

        // Estimate batch size based on remaining VRAM
        // Rough estimate: 0.1 GB per batch item for 7B model at 4k context
        let batch_items = (remaining_vram_gb / 0.1).floor() as u32;

        batch_items.clamp(1, self.batch_size)
    }
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self::default()
    }
}

/// Builder for VllmConfig with fluent interface
#[derive(Debug, Default)]
pub struct VllmConfigBuilder {
    config: VllmConfig,
}

impl VllmConfigBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: VllmConfig::default(),
        }
    }

    /// Set endpoint URL
    pub fn endpoint(mut self, endpoint: &str) -> Self {
        self.config.endpoint = endpoint.to_string();
        self
    }

    /// Set tensor parallel size
    pub fn tensor_parallel_size(mut self, size: u32) -> Self {
        self.config.tensor_parallel_size = size;
        self
    }

    /// Set max model context length
    pub fn max_model_len(mut self, len: u32) -> Self {
        self.config.max_model_len = len;
        self
    }

    /// Set GPU memory utilization
    pub fn gpu_memory_utilization(mut self, utilization: f32) -> Self {
        self.config.gpu_memory_utilization = utilization;
        self
    }

    /// Enable/disable prefix caching
    pub fn enable_prefix_caching(mut self, enabled: bool) -> Self {
        self.config.enable_prefix_caching = enabled;
        self
    }

    /// Set quantization type
    pub fn quantization(mut self, quantization: &str) -> Self {
        self.config.quantization = quantization.to_string();
        self
    }

    /// Enable/disable FP8 quantization
    pub fn enable_fp8(mut self, enabled: bool) -> Self {
        self.config.enable_fp8 = enabled;
        if enabled {
            self.config.quantization = "fp8".to_string();
        }
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: u32) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set VRAM strategy
    pub fn vram_strategy(mut self, strategy: VramStrategy) -> Self {
        self.config.vram_strategy = strategy;
        self
    }

    /// Build configuration with validation
    pub fn build(self) -> Result<VllmConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl From<tensorforge_core::config::VllmConfig> for VllmConfig {
    fn from(core_config: tensorforge_core::config::VllmConfig) -> Self {
        Self {
            enabled: core_config.enabled,
            endpoint: core_config.endpoint,
            tensor_parallel_size: core_config.tensor_parallel_size,
            pipeline_parallel_size: core_config.pipeline_parallel_size,
            max_model_len: core_config.max_model_len,
            gpu_memory_utilization: core_config.gpu_memory_utilization,
            enable_prefix_caching: core_config.enable_prefix_caching,
            enable_flash_attention: core_config.enable_flash_attention,
            quantization: core_config.quantization,
            max_concurrent_requests: core_config.max_concurrent_requests,
            request_timeout_secs: core_config.request_timeout_secs,
            // Default values for fields not present in core config
            enable_continuous_batching: default_enable_continuous_batching(),
            batch_size: default_batch_size(),
            enable_speculative_decoding: default_enable_speculative_decoding(),
            enable_fp8: false, // Not in core config
            enable_all_layers_on_gpu: default_enable_all_layers_on_gpu(),
            vram_strategy: default_vram_strategy(),
            model_load_timeout_secs: default_model_load_timeout_secs(),
            health_check_interval_secs: default_health_check_interval_secs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VllmConfig::default();
        assert!(config.enabled);
        assert_eq!(config.endpoint, "http://127.0.0.1:8000");
        assert_eq!(config.tensor_parallel_size, 4);
        assert_eq!(config.max_model_len, 131072);
        assert_eq!(config.gpu_memory_utilization, 0.9);
        assert!(config.enable_prefix_caching);
        assert!(config.enable_flash_attention);
        assert_eq!(config.quantization, "fp16");
        assert_eq!(config.max_concurrent_requests, 256);
        assert_eq!(config.request_timeout_secs, 300);
        assert!(config.enable_continuous_batching);
        assert_eq!(config.batch_size, 128);
        assert!(!config.enable_speculative_decoding);
        assert!(!config.enable_fp8);
        assert!(config.enable_all_layers_on_gpu);
        assert_eq!(config.vram_strategy, VramStrategy::Balanced);
    }

    #[test]
    fn test_b200_optimized_config() {
        let config = VllmConfig::b200_optimized();
        assert_eq!(config.tensor_parallel_size, 4);
        assert_eq!(config.max_model_len, 131072);
        assert_eq!(config.gpu_memory_utilization, 0.95);
        assert!(config.enable_fp8);
        assert!(config.enable_all_layers_on_gpu);
        assert_eq!(config.vram_strategy, VramStrategy::Aggressive);
        assert_eq!(config.batch_size, 256);
    }

    #[test]
    fn test_development_config() {
        let config = VllmConfig::development();
        assert_eq!(config.tensor_parallel_size, 1);
        assert_eq!(config.max_model_len, 8192);
        assert_eq!(config.gpu_memory_utilization, 0.7);
        assert!(!config.enable_prefix_caching);
        assert_eq!(config.max_concurrent_requests, 32);
        assert_eq!(config.batch_size, 16);
        assert!(!config.enable_speculative_decoding);
        assert!(!config.enable_fp8);
        assert_eq!(config.vram_strategy, VramStrategy::Conservative);
    }

    #[test]
    fn test_config_validation() {
        let config = VllmConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_failure() {
        let mut config = VllmConfig::default();
        config.endpoint = "invalid-url".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vram_estimation() {
        let config = VllmConfig::default();

        // Test 7B model with 4k context
        let vram_gb = config.estimate_vram_usage(7.0, 4096);
        assert!(vram_gb > 0.0);

        // Test with FP8 quantization
        let mut config_fp8 = VllmConfig::default();
        config_fp8.quantization = "fp8".to_string();
        let vram_fp8_gb = config_fp8.estimate_vram_usage(7.0, 4096);
        assert!(vram_fp8_gb < vram_gb); // FP8 should use less VRAM
    }

    #[test]
    fn test_builder_pattern() {
        let config = VllmConfigBuilder::new()
            .endpoint("http://localhost:8080")
            .tensor_parallel_size(2)
            .max_model_len(65536)
            .enable_fp8(true)
            .build()
            .unwrap();

        assert_eq!(config.endpoint, "http://localhost:8080");
        assert_eq!(config.tensor_parallel_size, 2);
        assert_eq!(config.max_model_len, 65536);
        assert!(config.enable_fp8);
        assert_eq!(config.quantization, "fp8");
    }

    #[test]
    fn test_vram_strategy_default() {
        let strategy = VramStrategy::default();
        assert_eq!(strategy, VramStrategy::Balanced);
    }
}
