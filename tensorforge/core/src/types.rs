
//! Core data types for TensorForge
//!
//! This module defines the fundamental data structures used throughout
//! the TensorForge ML inference orchestration system. These types
//! represent inference requests, responses, model information, system
//! state, and configuration options.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::config::BackendType;

/// Inference request sent to the orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceRequest {
    /// Unique identifier for the request
    pub id: String,

    /// Model identifier to use for inference
    pub model: String,

    /// Input prompt or text to process
    pub prompt: String,

    /// Maximum number of tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Sampling temperature (0.0 to 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p sampling parameter
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(default = "default_frequency_penalty")]
    pub frequency_penalty: f32,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,

    /// Stop sequences to end generation
    #[serde(default)]
    pub stop: Vec<String>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: u32,

    /// Request priority
    #[serde(default)]
    pub priority: RequestPriority,

    /// Request timeout in seconds
    #[serde(default = "default_request_timeout")]
    pub timeout_seconds: u64,

    /// Additional parameters specific to the backend
    #[serde(default)]
    pub parameters: serde_json::Value,

    /// Metadata for tracking and metrics
    #[serde(default)]
    pub metadata: HashMap<String, String>,

    /// Timestamp when request was created
    #[serde(default = "SystemTime::now")]
    pub created_at: SystemTime,
}

/// Inference result returned from the orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Request ID that generated this result
    pub request_id: String,

    /// Generated text
    pub text: String,

    /// Model used for inference
    pub model: String,

    /// Backend that processed the request
    pub backend: BackendType,

    /// Number of tokens in the prompt
    pub prompt_tokens: u32,

    /// Number of tokens generated
    pub completion_tokens: u32,

    /// Total tokens processed
    pub total_tokens: u32,

    /// Whether the request was successful
    pub success: bool,

    /// Error message if unsuccessful
    pub error: Option<String>,

    /// Time taken for inference in milliseconds
    pub inference_time_ms: u64,

    /// Time taken for queuing in milliseconds
    pub queue_time_ms: u64,

    /// Time taken for model loading in milliseconds (if applicable)
    pub load_time_ms: Option<u64>,

    /// Timestamp when inference started
    pub started_at: SystemTime,

    /// Timestamp when inference completed
    pub completed_at: SystemTime,

    /// Additional metadata from the backend
    #[serde(default)]
    pub backend_metadata: serde_json::Value,
}

/// Model information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier
    pub id: String,

    /// Human-readable model name
    pub name: String,

    /// Path to model files
    pub path: String,

    /// Model format (gguf, safetensors, etc.)
    pub format: String,

    /// Model size in gigabytes
    pub size_gb: f64,

    /// Estimated VRAM usage in gigabytes
    pub vram_estimate_gb: f64,

    /// Model architecture (llama, mistral, etc.)
    pub architecture: Option<String>,

    /// Quantization type (q4_k_m, fp16, int8, etc.)
    pub quantization: Option<String>,

    /// Number of parameters (e.g., "7B", "70B")
    pub parameter_count: Option<String>,

    /// Maximum context length in tokens
    pub context_length: u32,

    /// Compatible backends
    pub compatible_backends: Vec<BackendType>,

    /// Model tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Last time model was scanned
    pub last_scanned: SystemTime,

    /// Last time model was used
    pub last_used: Option<SystemTime>,

    /// Number of times model has been used
    pub usage_count: u64,

    /// Model priority for eviction decisions
    #[serde(default)]
    pub priority: ModelPriority,

    /// Additional notes about the model
    pub notes: Option<String>,
}

/// VRAM state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramState {
    /// Timestamp of the snapshot
    pub timestamp: SystemTime,

    /// Total VRAM in gigabytes
    pub total_gb: f64,

    /// Used VRAM in gigabytes
    pub used_gb: f64,

    /// Free VRAM in gigabytes
    pub free_gb: f64,

    /// VRAM utilization percentage
    pub utilization_percent: f64,

    /// GPU-specific information
    #[serde(default)]
    pub gpus: Vec<GpuInfo>,

    /// Processes using VRAM
    #[serde(default)]
    pub processes: Vec<GpuProcess>,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU identifier
    pub id: u32,

    /// GPU name
    pub name: String,

    /// Total memory in megabytes
    pub total_mb: u64,

    /// Used memory in megabytes
    pub used_mb: u64,

    /// Free memory in megabytes
    pub free_mb: u64,

    /// GPU utilization percentage
    pub utilization_percent: u32,

    /// GPU temperature in Celsius
    pub temperature_c: u32,

    /// GPU power draw in watts
    pub power_draw_w: Option<u32>,

    /// GPU power limit in watts
    pub power_limit_w: Option<u32>,
}

/// GPU process information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProcess {
    /// GPU ID
    pub gpu_id: u32,

    /// Process ID
    pub pid: u32,

    /// Process name
    pub name: String,

    /// Memory usage in megabytes
    pub memory_mb: u64,

    /// Process type (inference, training, etc.)
    pub process_type: Option<String>,
}

/// Request priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RequestPriority {
    /// Lowest priority (background tasks)
    Low = 0,
    /// Normal priority (default)
    Normal = 1,
    /// High priority (interactive requests)
    High = 2,
    /// Highest priority (critical requests)
    Critical = 3,
}

/// Model priority for eviction decisions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ModelPriority {
    /// Low priority (can be evicted first)
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority (keep in memory)
    High = 2,
    /// Pinned (never evict)
    Pinned = 3,
}

/// Pipeline processing stage
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    /// Request queued
    Queued,
    /// Model loading
    Loading,
    /// Inference processing
    Processing,
    /// Result aggregation
    Aggregating,
    /// Completed
    Completed,
    /// Failed
    Failed,
}

/// Batch processing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatus {
    /// Batch identifier
    pub batch_id: String,

    /// Total items to process
    pub total_items: u64,

    /// Items processed successfully
    pub processed_items: u64,

    /// Items failed
    pub failed_items: u64,

    /// Current processing stage
    pub stage: PipelineStage,

    /// Progress percentage (0.0 to 100.0)
    pub progress_percent: f64,

    /// Estimated time remaining
    pub estimated_remaining: Option<Duration>,

    /// Start time
    pub start_time: SystemTime,

    /// Last update time
    pub last_update: SystemTime,

    /// Errors encountered (if any)
    #[serde(default)]
    pub errors: Vec<String>,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Overall system status
    pub status: HealthStatus,

    /// Backend health status
    #[serde(default)]
    pub backends: HashMap<String, BackendHealth>,

    /// VRAM health status
    pub vram: VramHealth,

    /// Model availability status
    #[serde(default)]
    pub models: HashMap<String, ModelHealth>,

    /// System load information
    pub load: SystemLoad,

    /// Timestamp of health check
    pub timestamp: SystemTime,

    /// Error messages if status is unhealthy
    #[serde(default)]
    pub errors: Vec<String>,

    /// Warning messages if status is degraded
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Health status enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// System is fully operational
    Healthy,
    /// System is operational but degraded
    Degraded,
    /// System is unhealthy
    Unhealthy,
}

/// Backend health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendHealth {
    /// Backend type
    pub backend_type: BackendType,

    /// Health status
    pub status: HealthStatus,

    /// Whether backend is ready to accept requests
    pub ready: bool,

    /// Loaded models
    #[serde(default)]
    pub loaded_models: Vec<String>,

    /// Active request count
    pub active_requests: u32,

    /// Queue depth
    pub queue_depth: u32,

    /// Average latency in milliseconds
    pub avg_latency_ms: Option<f64>,

    /// Error rate (0.0 to 1.0)
    pub error_rate: Option<f64>,

    /// Last health check time
    pub last_check: SystemTime,

    /// Error message if unhealthy
    pub error: Option<String>,
}

/// VRAM health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramHealth {
    /// Health status
    pub status: HealthStatus,

    /// Total VRAM in gigabytes
    pub total_gb: f64,

    /// Used VRAM in gigabytes
    pub used_gb: f64,

    /// Free VRAM in gigabytes
    pub free_gb: f64,

    /// Utilization percentage
    pub utilization_percent: f64,

    /// Whether VRAM is critically low
    pub critical: bool,

    /// Whether VRAM is low
    pub low: bool,

    /// Recommended actions
    #[serde(default)]
    pub recommendations: Vec<String>,
}

/// Model health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealth {
    /// Health status
    pub status: HealthStatus,

    /// Whether model is loaded
    pub loaded: bool,

    /// Backend where model is loaded (if loaded)
    pub loaded_backend: Option<BackendType>,

    /// Load time in milliseconds (if loaded)
    pub load_time_ms: Option<u64>,

    /// VRAM usage in gigabytes (if loaded)
    pub vram_usage_gb: Option<f64>,

    /// Last used time
    pub last_used: Option<SystemTime>,

    /// Error message if unhealthy
    pub error: Option<String>,
}

/// System load information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemLoad {
    /// CPU utilization percentage
    pub cpu_percent: f64,

    /// Memory utilization percentage
    pub memory_percent: f64,

    /// Disk I/O utilization percentage
    pub disk_io_percent: f64,

    /// Network I/O utilization percentage
    pub network_io_percent: f64,

    /// System load average (1 minute)
    pub load_avg_1m: f64,

    /// System load average (5 minutes)
    pub load_avg_5m: f64,

    /// System load average (15 minutes)
    pub load_avg_15m: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Timestamp
    pub timestamp: SystemTime,

    /// Throughput in tokens per second
    pub tokens_per_second: f64,

    /// Requests per second
    pub requests_per_second: f64,

    /// Average latency in milliseconds
    pub avg_latency_ms: f64,

    /// P50 latency in milliseconds
    pub p50_latency_ms: f64,

    /// P95 latency in milliseconds
    pub p95_latency_ms: f64,

    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,

    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,

    /// GPU utilization percentage
    pub gpu_utilization_percent: f64,

    /// VRAM utilization percentage
    pub vram_utilization_percent: f64,

    /// Power efficiency in tokens per joule
    pub tokens_per_joule: Option<f64>,

    /// Cost per thousand tokens
    pub cost_per_k_tokens: Option<f64>,
}

/// Cost metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Timestamp
    pub timestamp: SystemTime,

    /// Total tokens processed
    pub total_tokens: u64,

    /// Total requests processed
    pub total_requests: u64,

    /// Total GPU time in hours
    pub total_gpu_hours: f64,

    /// Estimated cost in USD
    pub estimated_cost_usd: f64,

    /// Cost per thousand tokens in USD
    pub cost_per_k_tokens_usd: f64,

    /// Cost per request in USD
    pub cost_per_request_usd: f64,

    /// Cost per GPU hour in USD
    pub cost_per_gpu_hour_usd: f64,

    /// Most expensive models
    #[serde(default)]
    pub expensive_models: Vec<ModelCost>,
}

/// Model cost information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    /// Model identifier
    pub model: String,

    /// Total tokens processed
    pub tokens: u64,

    /// Total requests processed
    pub requests: u64,

    /// Total cost in USD
    pub cost_usd: f64,

    /// Cost per thousand tokens in USD
    pub cost_per_k_tokens_usd: f64,
}

// Default values
fn default_max_tokens() -> u32 { 1024 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.95 }
fn default_frequency_penalty() -> f32 { 0.0 }
fn default_presence_penalty() -> f32 { 0.0 }
fn default_n() -> u32 { 1 }
fn default_request_timeout() -> u64 { 300 }

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            model: String::new(),
            prompt: String::new(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            frequency_penalty: default_frequency_penalty(),
            presence_penalty: default_presence_penalty(),
            stop: Vec::new(),
            stream: false,
            n: default_n(),
            priority: RequestPriority::Normal,
            timeout_seconds: default_request_timeout(),
            parameters: serde_json::Value::Object(serde_json::Map::new()),
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
        }
    }
}

impl Default for RequestPriority {
    fn default() -> Self {
        RequestPriority::Normal
    }
}

impl Default for ModelPriority {
    fn default() -> Self {
        ModelPriority::Normal
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        HealthStatus::Healthy
    }
}

// Builder pattern for InferenceRequest
impl InferenceRequest {
    /// Create a new inference request with default values
    pub fn new<S: Into<String>>(model: S, prompt: S) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            ..Default::default()
        }
    }

    /// Set the maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set whether to stream the response
    pub fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Add metadata
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add a stop sequence
    pub fn with_stop_sequence<S: Into<String>>(mut self, stop: S) -> Self {
        self.stop.push(stop.into());
        self
    }
}

// Implement Display for enums
impl fmt::Display for RequestPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestPriority::Low => write!(f, "low"),
            RequestPriority::Normal => write!(f, "normal"),
            RequestPriority::High => write!(f, "high"),
            RequestPriority::Critical => write!(f, "critical"),
        }
    }
}

impl fmt::Display for ModelPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelPriority::Low => write!(f, "low"),
            ModelPriority::Normal => write!(f, "normal"),
            ModelPriority::High => write!(f, "high"),
            ModelPriority::Pinned => write!(f, "pinned"),
        }
    }
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineStage::Queued => write!(f, "queued"),
            PipelineStage::Loading => write!(f, "loading"),
            PipelineStage::Processing => write!(f, "processing"),
            PipelineStage::Aggregating => write!(f, "aggregating"),
            PipelineStage::Completed => write!(f, "completed"),
            PipelineStage::Failed => write!(f, "failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_request_defaults() {
        let request = InferenceRequest::default();
        assert_eq!(request.max_tokens, 1024);
        assert_eq!(request.temperature, 0.7);
        assert_eq!(request.priority, RequestPriority::Normal);
        assert!(!request.stream);
    }

    #[test]
    fn test_inference_request_builder() {
        let request = InferenceRequest::new("model", "prompt")
            .with_max_tokens(500)
            .with_temperature(0.5)
            .with_priority(RequestPriority::High)
            .with_streaming(true);

        assert_eq!(request.model, "model");
        assert_eq!(request.prompt, "prompt");
        assert_eq!(request.max_tokens, 500);
        assert_eq!(request.temperature, 0.5);
        assert_eq!(request.priority, RequestPriority::High);
        assert!(request.stream);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(RequestPriority::Low < RequestPriority::Normal);
        assert!(RequestPriority::Normal < RequestPriority::High);
        assert!(RequestPriority::High < RequestPriority::Critical);

        assert!(ModelPriority::Low < ModelPriority::Normal);
        assert!(ModelPriority::Normal < ModelPriority::High);
        assert!(ModelPriority::High < ModelPriority::Pinned);
    }

    #[test]
    fn test_enum_display() {
        assert_eq!(RequestPriority::High.to_string(), "high");
        assert_eq!(ModelPriority::Pinned.to_string(), "pinned");
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(PipelineStage::Processing.to_string(), "processing");
    }

    #[test]
    fn test_model_info_creation() {
        let model = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            path: "/models/test".to_string(),
            format: "gguf".to_string(),
            size_gb: 7.0,
            vram_estimate_gb: 4.0,
            architecture: Some("llama".to_string()),
            quantization: Some("q4_k_m".to_string()),
            parameter_count: Some("7B".to_string()),
            context_length: 4096,
            compatible_backends: vec![BackendType::LlamaCpp],
            tags: vec!["llama".to_string(), "quantized".to_string()],
            last_scanned: SystemTime::now(),
            last_used: None,
            usage_count: 0,
            priority: ModelPriority::Normal,
            notes: Some("Test model for development".to_string()),
        };

        assert_eq!(model.id, "test-model");
        assert_eq!(model.format, "gguf");
        assert_eq!(model.compatible_backends.len(), 1);
    }

    #[test]
    fn test_vram_state_creation() {
        let vram = VramState {
            timestamp: SystemTime::now(),
            total_gb: 24.0,
            used_gb: 12.0,
            free_gb: 12.0,
            utilization_percent: 50.0,
            gpus: vec![GpuInfo {
                id: 0,
                name: "RTX 4090".to_string(),
                total_mb: 24576,
                used_mb: 12288,
                free_mb: 12288,
                utilization_percent: 50,
                temperature_c: 65,
                power_draw_w: Some(300),
                power_limit_w: Some(450),
            }],
            processes: vec![],
        };

        assert_eq!(vram.total_gb, 24.0);
        assert_eq!(vram.utilization_percent, 50.0);
        assert_eq!(vram.gpus.len(), 1);
    }

    #[test]
    fn test_performance_metrics_serialization() {
        let metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            tokens_per_second: 5000.0,
            requests_per_second: 10.0,
            avg_latency_ms: 85.0,
            p50_latency_ms: 75.0,
            p95_latency_ms: 120.0,
            p99_latency_ms: 200.0,
            success_rate: 0.992,
            error_rate: 0.008,
            gpu_utilization_percent: 78.5,
            vram_utilization_percent: 65.2,
            tokens_per_joule: Some(1.8),
            cost_per_k_tokens: Some(0.42),
        };

        let json = serde_json::to_string(&metrics).unwrap();
        let decoded: PerformanceMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(metrics.tokens_per_second, decoded.tokens_per_second);
        assert_eq!(metrics.success_rate, decoded.success_rate);
    }
}
