//! Configuration management for TensorForge
//!
//! This module provides a flexible configuration system with support for:
//! - Builder pattern for easy configuration
//! - Loading from TOML files
//! - Environment variable overrides
//! - Predefined configurations (default, B200-optimized, etc.)
//! - Validation and defaults

use std::path::Path;
use std::str::FromStr;
use std::{fs, path::PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::TensorForgeResult;

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    FileRead(#[from] std::io::Error),

    #[error("Failed to parse config file: {0}")]
    Parse(#[from] toml::de::Error),

    #[error("Invalid configuration: {0}")]
    Validation(String),

    #[error("Configuration file not found: {0}")]
    NotFound(PathBuf),

    #[error("Environment variable error: {0}")]
    EnvVar(#[from] std::env::VarError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] toml::ser::Error),
}

/// Main configuration struct
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Orchestrator configuration
    pub orchestrator: OrchestratorConfig,

    /// Backend configurations
    pub backends: BackendConfigs,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// Pipeline configuration
    pub pipeline: PipelineConfig,

    /// API server configuration
    pub api: ApiConfig,

    /// VRAM monitoring configuration
    pub vram: VramConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Paths configuration
    pub paths: PathsConfig,
}

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OrchestratorConfig {
    /// Maximum batch size for requests
    pub max_batch_size: usize,

    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,

    /// Enable dynamic batching
    pub enable_dynamic_batching: bool,

    /// Maximum queue depth
    pub max_queue_depth: usize,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,

    /// Enable request prioritization
    pub enable_prioritization: bool,

    /// Number of worker threads (0 = auto)
    pub worker_threads: usize,

    /// Health check interval in seconds
    pub health_check_interval_secs: u64,

    /// Model cache size (number of models to keep in memory)
    pub model_cache_size: usize,

    /// Enable automatic model loading
    pub auto_load_models: bool,
}

/// Backend configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackendConfigs {
    /// vLLM backend configuration
    pub vllm: VllmConfig,

    /// llama.cpp backend configuration
    pub llamacpp: LlamaCppConfig,

    /// Default backend to use when not specified
    pub default_backend: BackendType,

    /// Enable backend auto-discovery
    pub auto_discovery: bool,

    /// Backend selection strategy
    pub selection_strategy: SelectionStrategy,
}

/// vLLM backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VllmConfig {
    /// Enable vLLM backend
    pub enabled: bool,

    /// vLLM API endpoint
    pub endpoint: String,

    /// Tensor parallel size
    pub tensor_parallel_size: u32,

    /// Pipeline parallel size
    pub pipeline_parallel_size: u32,

    /// Maximum model length (context)
    pub max_model_len: u32,

    /// GPU memory utilization (0.0-1.0)
    pub gpu_memory_utilization: f32,

    /// Enable prefix caching
    pub enable_prefix_caching: bool,

    /// Enable FlashAttention
    pub enable_flash_attention: bool,

    /// Quantization type (fp16, fp8, int8)
    pub quantization: String,

    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,
}

/// llama.cpp backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaCppConfig {
    /// Enable llama.cpp backend
    pub enabled: bool,

    /// llama.cpp API endpoint
    pub endpoint: String,

    /// Number of GPU layers to load
    pub n_gpu_layers: u32,

    /// Context length
    pub n_ctx: u32,

    /// Batch size
    pub n_batch: u32,

    /// Number of threads
    pub n_threads: u32,

    /// Number of threads for batch processing
    pub n_threads_batch: u32,

    /// Enable FlashAttention
    pub flash_attn: bool,

    /// Enable CUDA graphs
    pub cuda_graphs: bool,

    /// Rope scaling type
    pub rope_scaling_type: String,

    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsConfig {
    /// Enable Prometheus metrics endpoint
    pub prometheus_enabled: bool,

    /// Prometheus metrics port
    pub prometheus_port: u16,

    /// Enable cost tracking
    pub cost_tracking_enabled: bool,

    /// Enable quality metrics
    pub quality_metrics_enabled: bool,

    /// Metrics collection interval in seconds
    pub collection_interval_secs: u64,

    /// Enable detailed request metrics
    pub detailed_request_metrics: bool,

    /// Enable GPU metrics
    pub gpu_metrics_enabled: bool,

    /// Metrics retention period in hours
    pub retention_hours: u64,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineConfig {
    /// Enable checkpointing
    pub checkpoint_enabled: bool,

    /// Checkpoint interval (number of items)
    pub checkpoint_interval: usize,

    /// Maximum retries for failed items
    pub max_retries: u32,

    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,

    /// Maximum parallel streams
    pub max_parallel_streams: usize,

    /// Input batch size
    pub input_batch_size: usize,

    /// Output batch size
    pub output_batch_size: usize,

    /// Enable result validation
    pub enable_validation: bool,

    /// Validation timeout in seconds
    pub validation_timeout_secs: u64,
}

/// API server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiConfig {
    /// Enable API server
    pub enabled: bool,

    /// Host to bind to
    pub host: String,

    /// Port to listen on
    pub port: u16,

    /// Enable CORS
    pub cors_enabled: bool,

    /// CORS allowed origins
    pub cors_allowed_origins: Vec<String>,

    /// Enable rate limiting
    pub rate_limiting_enabled: bool,

    /// Rate limit requests per minute
    pub rate_limit_rpm: u32,

    /// Enable authentication
    pub auth_enabled: bool,

    /// API key (if auth enabled)
    pub api_key: Option<String>,

    /// Enable WebSocket endpoint
    pub websocket_enabled: bool,

    /// WebSocket ping interval in seconds
    pub websocket_ping_interval_secs: u64,

    /// Maximum WebSocket message size in bytes
    pub websocket_max_message_size: usize,
}

/// VRAM monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VramConfig {
    /// Enable VRAM monitoring
    pub enabled: bool,

    /// Monitoring interval in seconds
    pub monitoring_interval_secs: u64,

    /// Low VRAM threshold (percentage)
    pub low_vram_threshold_percent: f32,

    /// Critical VRAM threshold (percentage)
    pub critical_vram_threshold_percent: f32,

    /// Enable automatic model eviction
    pub auto_eviction_enabled: bool,

    /// Eviction policy
    pub eviction_policy: EvictionPolicy,

    /// Minimum free VRAM to maintain (GB)
    pub min_free_vram_gb: f32,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,

    /// Enable JSON logging
    pub json_enabled: bool,

    /// Log file path (optional)
    pub file_path: Option<PathBuf>,

    /// Enable file rotation
    pub file_rotation_enabled: bool,

    /// Maximum log file size in MB
    pub max_file_size_mb: u64,

    /// Number of log files to keep
    pub max_files: usize,

    /// Enable structured logging
    pub structured_logging: bool,

    /// Include thread IDs in logs
    pub include_thread_ids: bool,

    /// Include timestamps in logs
    pub include_timestamps: bool,
}

/// Paths configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PathsConfig {
    /// Base data directory
    pub data_dir: PathBuf,

    /// Models directory
    pub models_dir: PathBuf,

    /// Cache directory
    pub cache_dir: PathBuf,

    /// Logs directory
    pub logs_dir: PathBuf,

    /// Config directory
    pub config_dir: PathBuf,

    /// Checkpoints directory
    pub checkpoints_dir: PathBuf,

    /// Temporary directory
    pub temp_dir: PathBuf,
}

/// Backend type enum
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackendType {
    /// vLLM backend
    Vllm,
    /// llama.cpp backend
    LlamaCpp,
    /// Auto-select based on metrics
    Auto,
}

impl FromStr for BackendType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "vllm" => Ok(BackendType::Vllm),
            "llamacpp" | "llama.cpp" | "llama_cpp" => Ok(BackendType::LlamaCpp),
            "auto" => Ok(BackendType::Auto),
            _ => Err(format!("Invalid backend type: {}", s)),
        }
    }
}

/// Backend selection strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Lowest latency
    Latency,
    /// Highest throughput
    Throughput,
    /// Best VRAM utilization
    VramEfficiency,
    /// Balanced approach
    Balanced,
    /// Round-robin
    RoundRobin,
}

/// Eviction policy for models
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In First Out
    Fifo,
    /// Based on model size
    SizeBased,
    /// Based on priority
    PriorityBased,
}

// Default implementations
impl Default for Config {
    fn default() -> Self {
        Self {
            orchestrator: OrchestratorConfig::default(),
            backends: BackendConfigs::default(),
            metrics: MetricsConfig::default(),
            pipeline: PipelineConfig::default(),
            api: ApiConfig::default(),
            vram: VramConfig::default(),
            logging: LoggingConfig::default(),
            paths: PathsConfig::default(),
        }
    }
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            batch_timeout_ms: 1000,
            enable_dynamic_batching: true,
            max_queue_depth: 1000,
            request_timeout_secs: 300,
            enable_prioritization: true,
            worker_threads: 0, // auto
            health_check_interval_secs: 30,
            model_cache_size: 10,
            auto_load_models: true,
        }
    }
}

impl Default for BackendConfigs {
    fn default() -> Self {
        Self {
            vllm: VllmConfig::default(),
            llamacpp: LlamaCppConfig::default(),
            default_backend: BackendType::Auto,
            auto_discovery: true,
            selection_strategy: SelectionStrategy::Balanced,
        }
    }
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "http://localhost:8000".to_string(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            max_model_len: 16384,
            gpu_memory_utilization: 0.9,
            enable_prefix_caching: true,
            enable_flash_attention: true,
            quantization: "fp16".to_string(),
            max_concurrent_requests: 100,
            request_timeout_secs: 300,
        }
    }
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "http://localhost:8080".to_string(),
            n_gpu_layers: 0,
            n_ctx: 4096,
            n_batch: 512,
            n_threads: 0, // auto
            n_threads_batch: 0, // auto
            flash_attn: false,
            cuda_graphs: false,
            rope_scaling_type: "linear".to_string(),
            max_concurrent_requests: 50,
            request_timeout_secs: 300,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            prometheus_enabled: true,
            prometheus_port: 9091,
            cost_tracking_enabled: true,
            quality_metrics_enabled: true,
            collection_interval_secs: 5,
            detailed_request_metrics: true,
            gpu_metrics_enabled: true,
            retention_hours: 24,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            checkpoint_enabled: true,
            checkpoint_interval: 1000,
            max_retries: 3,
            retry_delay_ms: 1000,
            max_parallel_streams: 4,
            input_batch_size: 100,
            output_batch_size: 100,
            enable_validation: true,
            validation_timeout_secs: 30,
        }
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            host: "127.0.0.1".to_string(),
            port: 8080,
            cors_enabled: false,
            cors_allowed_origins: vec!["http://localhost:3000".to_string()],
            rate_limiting_enabled: true,
            rate_limit_rpm: 60,
            auth_enabled: false,
            api_key: None,
            websocket_enabled: true,
            websocket_ping_interval_secs: 30,
            websocket_max_message_size: 16 * 1024 * 1024, // 16MB
        }
    }
}

impl Default for VramConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval_secs: 2,
            low_vram_threshold_percent: 80.0,
            critical_vram_threshold_percent: 95.0,
            auto_eviction_enabled: true,
            eviction_policy: EvictionPolicy::Lru,
            min_free_vram_gb: 1.0,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json_enabled: false,
            file_path: None,
            file_rotation_enabled: true,
            max_file_size_mb: 100,
            max_files: 10,
            structured_logging: false,
            include_thread_ids: true,
            include_timestamps: true,
        }
    }
}

impl Default for PathsConfig {
    fn default() -> Self {
        let data_dir = PathBuf::from("/var/lib/tensorforge");

        Self {
            data_dir: data_dir.clone(),
            models_dir: data_dir.join("models"),
            cache_dir: data_dir.join("cache"),
            logs_dir: data_dir.join("logs"),
            config_dir: PathBuf::from("/etc/tensorforge"),
            checkpoints_dir: data_dir.join("checkpoints"),
            temp_dir: std::env::temp_dir().join("tensorforge"),
        }
    }
}

impl Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for NVIDIA B200 (192GB HBM3e)
    pub fn b200_optimized() -> Self {
        let mut config = Self::default();

        // Optimize orchestrator for high throughput
        config.orchestrator.max_batch_size = 256;
        config.orchestrator.batch_timeout_ms = 2000;
        config.orchestrator.worker_threads = 8;
        config.orchestrator.model_cache_size = 20;

        // Optimize vLLM for B200
        config.backends.vllm.tensor_parallel_size = 4;
        config.backends.vllm.pipeline_parallel_size = 2;
        config.backends.vllm.max_model_len = 131072;
        config.backends.vllm.gpu_memory_utilization = 0.95;
        config.backends.vllm.quantization = "fp8".to_string();
        config.backends.vllm.max_concurrent_requests = 256;

        // Optimize llama.cpp for B200
        config.backends.llamacpp.n_gpu_layers = 999; // All layers on GPU
        config.backends.llamacpp.n_ctx = 32768;
        config.backends.llamacpp.flash_attn = true;
        config.backends.llamacpp.cuda_graphs = true;
        config.backends.llamacpp.max_concurrent_requests = 128;

        // Optimize VRAM settings for 192GB
        config.vram.min_free_vram_gb = 4.0;

        // Optimize pipeline for massive scale
        config.pipeline.max_parallel_streams = 16;
        config.pipeline.input_batch_size = 1000;
        config.pipeline.output_batch_size = 1000;

        config
    }

    /// Create a configuration optimized for development
    pub fn development() -> Self {
        let mut config = Self::default();

        // Development-friendly settings
        config.orchestrator.max_batch_size = 16;
        config.orchestrator.model_cache_size = 3;
        config.orchestrator.auto_load_models = false;

        config.backends.vllm.enabled = true;
        config.backends.llamacpp.enabled = true;

        config.api.host = "127.0.0.1".to_string();
        config.api.cors_enabled = true;
        config.api.auth_enabled = false;

        config.logging.level = "debug".to_string();
        config.logging.json_enabled = false;

        config.paths.data_dir = PathBuf::from("./data");
        config.paths.models_dir = PathBuf::from("./models");
        config.paths.logs_dir = PathBuf::from("./logs");
        config.paths.config_dir = PathBuf::from("./config");

        config
    }

    /// Load configuration from a TOML file
    pub async fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ConfigError::NotFound(path.to_path_buf()));
        }

        let content = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a TOML string
    pub fn from_toml(toml: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(toml)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub async fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let path = path.as_ref();

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let toml = toml::to_string_pretty(self)?;
        fs::write(path, toml)?;

        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_vars(&mut self) {
        // Helper function to read optional env var
        fn env_var<T: FromStr>(key: &str) -> Option<T> {
            std::env::var(key).ok().and_then(|v| v.parse().ok())
        }

        // Backend configuration
        if let Some(enabled) = env_var("TENSORFORGE_VLLM_ENABLED") {
            self.backends.vllm.enabled = enabled;
        }
        if let Some(endpoint) = std::env::var("TENSORFORGE_VLLM_ENDPOINT").ok() {
            self.backends.vllm.endpoint = endpoint;
        }
        if let Some(enabled) = env_var("TENSORFORGE_LLAMACPP_ENABLED") {
            self.backends.llamacpp.enabled = enabled;
        }
        if let Some(endpoint) = std::env::var("TENSORFORGE_LLAMACPP_ENDPOINT").ok() {
            self.backends.llamacpp.endpoint = endpoint;
        }

        // API configuration
        if let Some(host) = std::env::var("TENSORFORGE_HOST").ok() {
            self.api.host = host;
        }
        if let Some(port) = env_var("TENSORFORGE_PORT") {
            self.api.port = port;
        }
        if let Some(enabled) = env_var("TENSORFORGE_CORS_ENABLED") {
            self.api.cors_enabled = enabled;
        }

        // Paths configuration
        if let Some(data_dir) = std::env::var("TENSORFORGE_DATA_DIR").ok() {
            self.paths.data_dir = PathBuf::from(data_dir);
        }
        if let Some(models_dir) = std::env::var("TENSORFORGE_MODELS_DIR").ok() {
            self.paths.models_dir = PathBuf::from(models_dir);
        }

        // Logging configuration
        if let Some(level) = std::env::var("TENSORFORGE_LOG_LEVEL").ok() {
            self.logging.level = level;
        }
        if let Some(json) = env_var("TENSORFORGE_JSON_LOGS") {
            self.logging.json_enabled = json;
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate orchestrator
        if self.orchestrator.max_batch_size == 0 {
            return Err(ConfigError::Validation("max_batch_size must be greater than 0".into()));
        }

        if self.orchestrator.max_queue_depth == 0 {
            return Err(ConfigError::Validation("max_queue_depth must be greater than 0".into()));
        }

        // Validate vLLM configuration
        if self.backends.vllm.enabled {
            if self.backends.vllm.gpu_memory_utilization <= 0.0 || self.backends.vllm.gpu_memory_utilization > 1.0 {
                return Err(ConfigError::Validation(
                    "vLLM gpu_memory_utilization must be between 0.0 and 1.0".into()
                ));
            }

            if self.backends.vllm.max_concurrent_requests == 0 {
                return Err(ConfigError::Validation(
                    "vLLM max_concurrent_requests must be greater than 0".into()
                ));
            }
        }

        // Validate llama.cpp configuration
        if self.backends.llamacpp.enabled {
            if self.backends.llamacpp.max_concurrent_requests == 0 {
                return Err(ConfigError::Validation(
                    "llama.cpp max_concurrent_requests must be greater than 0".into()
                ));
            }
        }

        // Validate at least one backend is enabled
        if !self.backends.vllm.enabled && !self.backends.llamacpp.enabled {
            return Err(ConfigError::Validation(
                "At least one backend must be enabled".into()
            ));
        }

        // Validate API configuration
        if self.api.enabled && self.api.port == 0 {
            return Err(ConfigError::Validation("API port must be greater than 0".into()));
        }

        // Validate VRAM thresholds
        if self.vram.enabled {
            if self.vram.low_vram_threshold_percent >= self.vram.critical_vram_threshold_percent {
                return Err(ConfigError::Validation(
                    "low_vram_threshold must be less than critical_vram_threshold".into()
                ));
            }

            if self.vram.min_free_vram_gb < 0.0 {
                return Err(ConfigError::Validation(
                    "min_free_vram_gb must be non-negative".into()
                ));
            }
        }

        Ok(())
    }

    /// Get a reference to the orchestrator configuration
    pub fn orchestrator(&self) -> &OrchestratorConfig {
        &self.orchestrator
    }

    /// Get a mutable reference to the orchestrator configuration
    pub fn orchestrator_mut(&mut self) -> &mut OrchestratorConfig {
        &mut self.orchestrator
    }

    /// Get a reference to the backends configuration
    pub fn backends(&self) -> &BackendConfigs {
        &self.backends
    }

    /// Get a mutable reference to the backends configuration
    pub fn backends_mut(&mut self) -> &mut BackendConfigs {
        &mut self.backends
    }

    /// Get a reference to the metrics configuration
    pub fn metrics(&self) -> &MetricsConfig {
        &self.metrics
    }

    /// Get a mutable reference to the metrics configuration
    pub fn metrics_mut(&mut self) -> &mut MetricsConfig {
        &mut self.metrics
    }

    /// Get a reference to the pipeline configuration
    pub fn pipeline(&self) -> &PipelineConfig {
        &self.pipeline
    }

    /// Get a mutable reference to the pipeline configuration
    pub fn pipeline_mut(&mut self) -> &mut PipelineConfig {
        &mut self.pipeline
    }
}

/// Builder for Config
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Create a new ConfigBuilder with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// Set orchestrator configuration
    pub fn orchestrator(mut self, orchestrator: OrchestratorConfig) -> Self {
        self.config.orchestrator = orchestrator;
        self
    }

    /// Set backends configuration
    pub fn backends(mut self, backends: BackendConfigs) -> Self {
        self.config.backends = backends;
        self
    }

    /// Enable vLLM backend
    pub fn with_vllm_backend(mut self) -> Self {
        self.config.backends.vllm.enabled = true;
        self
    }

    /// Disable vLLM backend
    pub fn without_vllm_backend(mut self) -> Self {
        self.config.backends.vllm.enabled = false;
        self
    }

    /// Enable llama.cpp backend
    pub fn with_llamacpp_backend(mut self) -> Self {
        self.config.backends.llamacpp.enabled = true;
        self
    }

    /// Disable llama.cpp backend
    pub fn without_llamacpp_backend(mut self) -> Self {
        self.config.backends.llamacpp.enabled = false;
        self
    }

    /// Set metrics configuration
    pub fn metrics(mut self, metrics: MetricsConfig) -> Self {
        self.config.metrics = metrics;
        self
    }

    /// Set pipeline configuration
    pub fn pipeline(mut self, pipeline: PipelineConfig) -> Self {
        self.config.pipeline = pipeline;
        self
    }

    /// Set API configuration
    pub fn api(mut self, api: ApiConfig) -> Self {
        self.config.api = api;
        self
    }

    /// Set VRAM configuration
    pub fn vram(mut self, vram: VramConfig) -> Self {
        self.config.vram = vram;
        self
    }

    /// Set logging configuration
    pub fn logging(mut self, logging: LoggingConfig) -> Self {
        self.config.logging = logging;
        self
    }

    /// Set paths configuration
    pub fn paths(mut self, paths: PathsConfig) -> Self {
        self.config.paths = paths;
        self
    }

    /// Build the configuration, validating it in the process
    pub fn build(self) -> Result<Config, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.backends.vllm.enabled);
        assert!(config.backends.llamacpp.enabled);
        assert_eq!(config.api.port, 8080);
        assert_eq!(config.backends.default_backend, BackendType::Auto);
    }

    #[test]
    fn test_b200_optimized_config() {
        let config = Config::b200_optimized();
        assert_eq!(config.orchestrator.max_batch_size, 256);
        assert_eq!(config.backends.vllm.tensor_parallel_size, 4);
        assert_eq!(config.backends.vllm.max_model_len, 131072);
        assert_eq!(config.backends.llamacpp.n_gpu_layers, 999);
        assert_eq!(config.pipeline.max_parallel_streams, 16);
    }

    #[test]
    fn test_development_config() {
        let config = Config::development();
        assert_eq!(config.orchestrator.max_batch_size, 16);
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.paths.data_dir, PathBuf::from("./data"));
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        config.backends.vllm.enabled = false;
        config.backends.llamacpp.enabled = false;

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .with_vllm_backend()
            .without_llamacpp_backend()
            .build()
            .unwrap();

        assert!(config.backends.vllm.enabled);
        assert!(!config.backends.llamacpp.enabled);
    }

    #[test]
    fn test_backend_type_parsing() {
        assert_eq!("vllm".parse::<BackendType>(), Ok(BackendType::Vllm));
        assert_eq!("llamacpp".parse::<BackendType>(), Ok(BackendType::LlamaCpp));
        assert_eq!("auto".parse::<BackendType>(), Ok(BackendType::Auto));
        assert!("invalid".parse::<BackendType>().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml = toml::to_string(&config).unwrap();
        assert!(toml.contains("orchestrator"));
        assert!(toml.contains("backends"));
        assert!(toml.contains("api"));
    }

    #[test]
    fn test_config_deserialization() {
        let toml = r#"
            [orchestrator]
            max_batch_size = 128
            batch_timeout_ms = 500

            [backends.vllm]
            enabled = true
            endpoint = "http://localhost:8001"

            [api]
            port = 9090
            cors_enabled = true
        "#;

        let config = Config::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.max_batch_size, 128);
        assert_eq!(config.orchestrator.batch_timeout_ms, 500);
        assert_eq!(config.backends.vllm.endpoint, "http://localhost:8001");
        assert_eq!(config.api.port, 9090);
        assert!(config.api.cors_enabled);
    }
}
