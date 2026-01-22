//! # TensorForge
//!
//! High-performance ML inference orchestration engine for massive-scale pipelines.
//!
//! ## Overview
//!
//! TensorForge is a production-grade orchestration engine designed for massive-scale
//! ML inference workloads. It provides:
//!
//! - **Multi-backend support**: Unified interface for vLLM, llama.cpp, and other inference engines
//! - **Intelligent routing**: Hardware-aware request distribution based on VRAM, latency, and cost
//! - **Advanced metrics**: Real-time performance, cost, and quality tracking
//! - **Pipeline orchestration**: Batch and stream processing with fault tolerance
//! - **DSPy integration**: First-class support for LM pipeline optimization framework
//! - **Nix packaging**: Reproducible builds and deployments
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                 TensorForge Core                     │
//! ├─────────────┬─────────────┬─────────────┬───────────┤
//! │ Orchestrator│   Backends  │  Pipeline   │  Metrics  │
//! │             │             │             │           │
//! │ • Routing   │ • vLLM      │ • Batch     │ • Prometheus│
//! │ • Batching  │ • llama.cpp │ • Stream    │ • Cost    │
//! │ • Scheduling│ • Custom    │ • Checkpoint│ • Quality │
//! └─────────────┴─────────────┴─────────────┴───────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use tensorforge::prelude::*;
//! use tensorforge::{Orchestrator, Config};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), TensorForgeError> {
//!     // Initialize with B200-optimized configuration
//!     let config = Config::b200_optimized()
//!         .with_vllm_backend()
//!         .with_llamacpp_backend();
//!
//!     let orchestrator = Orchestrator::new(config).await?;
//!
//!     // Process inference requests
//!     let request = InferenceRequest::new("Explain quantum computing")
//!         .with_model("mixtral-8x7b")
//!         .with_max_tokens(500);
//!
//!     let result = orchestrator.process(request).await?;
//!
//!     println!("Response: {}", result.text);
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - **`api`**: Enable REST API server (default)
//! - **`metrics`**: Enable Prometheus metrics collection (default)
//! - **`vram-monitoring`**: Enable NVIDIA VRAM monitoring (default)
//! - **`backend-vllm`**: Enable vLLM backend support
//! - **`backend-llamacpp`**: Enable llama.cpp backend support
//! - **`experimental-dspy`**: Enable DSPy Python integration
//! - **`gpu`**: Enable GPU-specific optimizations (requires CUDA)
//!
//! ## License
//!
//! MIT

#![doc(html_logo_url = "https://raw.githubusercontent.com/kernelcore/tensorforge/main/docs/logo.svg")]
#![doc(html_favicon_url = "https://raw.githubusercontent.com/kernelcore/tensorforge/main/docs/favicon.ico")]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

// Core modules
pub mod config;
pub mod error;
pub mod types;
pub mod utils;

// Feature-gated modules
#[cfg(feature = "api")]
pub mod api;

#[cfg(feature = "orchestrator")]
pub mod orchestrator;

#[cfg(any(feature = "backend-vllm", feature = "backend-llamacpp"))]
pub mod backends;

#[cfg(feature = "metrics")]
pub mod metrics;

#[cfg(feature = "pipeline")]
pub mod pipeline;

#[cfg(feature = "vram-monitoring")]
pub mod vram;

// Re-exports for common usage
pub use config::Config;
pub use error::{TensorForgeError, TensorForgeResult};
pub use config::BackendType;
pub use types::{
    InferenceRequest, InferenceResult, ModelInfo, ModelPriority,
    PipelineStage, RequestPriority, VramState,
};

// Conditional re-exports based on features
#[cfg(feature = "orchestrator")]
pub use orchestrator::Orchestrator;

#[cfg(feature = "vram-monitoring")]
pub use vram::VramMonitor;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        Config, TensorForgeError, TensorForgeResult,
        BackendType, InferenceRequest, InferenceResult,
    };

    #[cfg(feature = "orchestrator")]
    pub use crate::Orchestrator;

    #[cfg(feature = "vram-monitoring")]
    pub use crate::VramMonitor;

    // Re-export commonly used traits
    pub use async_trait::async_trait;
    pub use serde::{Deserialize, Serialize};
    pub use tracing::{debug, error, info, trace, warn};
}

// Core traits
use async_trait::async_trait;
use std::sync::Arc;

/// Backend trait defining the interface for all inference engines
#[async_trait]
pub trait Backend: Send + Sync {
    /// Get the backend name
    fn name(&self) -> &str;

    /// Get the backend type
    fn backend_type(&self) -> BackendType;

    /// Check if backend is healthy and ready
    async fn health_check(&self) -> TensorForgeResult<BackendHealth>;

    /// Check if a specific model is loaded
    async fn has_model(&self, model_id: &str) -> bool;

    /// Load a model onto the backend
    async fn load_model(&self, model_id: &str, options: LoadOptions) -> TensorForgeResult<LoadResult>;

    /// Unload a model from the backend
    async fn unload_model(&self, model_id: &str) -> TensorForgeResult<()>;

    /// Process an inference request
    async fn infer(&self, request: InferenceRequest) -> TensorForgeResult<InferenceResult>;

    /// Get current VRAM usage for this backend
    async fn vram_usage(&self) -> TensorForgeResult<VramUsage>;

    /// Get estimated VRAM required for a model
    fn estimated_vram(&self, model_id: &str) -> Option<u64>;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;
}

/// Backend health status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendHealth {
    /// Overall status
    pub status: BackendStatus,
    /// Detailed message if available
    pub message: Option<String>,
    /// Timestamp of health check
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Loaded models
    pub loaded_models: Vec<String>,
}

/// Backend status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackendStatus {
    /// Backend is fully operational
    Healthy,
    /// Backend is operational but degraded
    Degraded,
    /// Backend is not responding
    Unhealthy,
    /// Backend is not configured
    Disabled,
}

/// Options for loading a model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoadOptions {
    /// Path to model files
    pub model_path: String,
    /// GPU layers to load (for llama.cpp)
    pub gpu_layers: Option<u32>,
    /// Context length
    pub context_length: Option<u32>,
    /// Batch size
    pub batch_size: Option<u32>,
    /// Quantization type
    pub quantization: Option<String>,
    /// Additional backend-specific options
    pub backend_options: serde_json::Value,
}

/// Result of loading a model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoadResult {
    /// Whether load was successful
    pub success: bool,
    /// Model identifier
    pub model_id: String,
    /// Time taken to load in seconds
    pub load_time_secs: f64,
    /// Actual VRAM used in MB
    pub vram_used_mb: u64,
    /// Any warnings during load
    pub warnings: Vec<String>,
}

/// VRAM usage information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VramUsage {
    /// Total VRAM available in MB
    pub total_mb: u64,
    /// VRAM used by this backend in MB
    pub used_mb: u64,
    /// Peak VRAM usage since last reset
    pub peak_mb: u64,
    /// Percentage utilization
    pub utilization_percent: f32,
}

/// Backend capabilities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendCapabilities {
    /// Supports streaming responses
    pub streaming: bool,
    /// Supports embeddings
    pub embeddings: bool,
    /// Supports function calling
    pub function_calling: bool,
    /// Supports vision models
    pub vision: bool,
    /// Supports tool use
    pub tool_use: bool,
    /// Maximum context length
    pub max_context_length: u32,
    /// Supported quantization types
    pub quantization_types: Vec<String>,
}

/// Initialize TensorForge with default configuration
///
/// # Returns
/// - `Ok(Orchestrator)` if initialization succeeds
/// - `Err(TensorForgeError)` if initialization fails
///
/// # Example
/// ```rust
/// use tensorforge::initialize;
///
/// #[tokio::main]
/// async fn main() -> Result<(), tensorforge::TensorForgeError> {
///     let orchestrator = initialize().await?;
///     // Use orchestrator...
///     Ok(())
/// }
/// ```
#[cfg(feature = "orchestrator")]
pub async fn initialize() -> TensorForgeResult<Orchestrator> {
    let config = Config::default();
    Orchestrator::new(config).await
}

/// Initialize TensorForge with custom configuration
///
/// # Arguments
/// * `config` - Configuration to use
///
/// # Returns
/// - `Ok(Orchestrator)` if initialization succeeds
/// - `Err(TensorForgeError)` if initialization fails
#[cfg(feature = "orchestrator")]
pub async fn initialize_with_config(config: Config) -> TensorForgeResult<Orchestrator> {
    Orchestrator::new(config).await
}

/// Get TensorForge version
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get TensorForge build information
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        authors: env!("CARGO_PKG_AUTHORS").to_string(),
        description: env!("CARGO_PKG_DESCRIPTION").to_string(),
        repository: env!("CARGO_PKG_REPOSITORY").to_string(),
        rustc_version: "unknown".to_string(), // dependency removed
        build_time: chrono::Utc::now(),
        features: std::env::var("CARGO_PKG_FEATURES").unwrap_or_else(|_| "unknown".to_string()),
    }
}

/// Build information structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BuildInfo {
    /// Package version
    pub version: String,
    /// Package authors
    pub authors: String,
    /// Package description
    pub description: String,
    /// Repository URL
    pub repository: String,
    /// Rust compiler version
    pub rustc_version: String,
    /// Build timestamp
    pub build_time: chrono::DateTime<chrono::Utc>,
    /// Enabled features
    pub features: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = version();
        assert!(!version.is_empty());
        assert!(version.starts_with('0') || version.starts_with('1'));
    }

    #[test]
    fn test_build_info() {
        let info = build_info();
        assert!(!info.version.is_empty());
        assert!(!info.authors.is_empty());
        assert!(!info.description.is_empty());
    }

    #[test]
    fn test_backend_status_serialization() {
        let status = BackendStatus::Healthy;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"Healthy\"");

        let decoded: BackendStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, BackendStatus::Healthy);
    }
}
