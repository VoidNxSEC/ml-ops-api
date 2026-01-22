//! Error types for TensorForge
//!
//! This module defines comprehensive error types for the TensorForge ML inference
//! orchestration engine. It provides a unified error handling system with
//! detailed error categories, automatic error conversion, and utilities for
//! error handling and reporting.

use std::fmt;
use std::io;
use std::num::ParseIntError;
use std::str::ParseBoolError;
use std::sync::Arc;

use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Main error type for TensorForge operations
#[derive(Error, Debug, Clone)]
pub enum TensorForgeError {
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    /// Backend-related errors
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),

    /// API and HTTP errors
    #[error("API error: {0}")]
    Api(#[from] ApiError),

    /// Pipeline processing errors
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// Metrics collection and reporting errors
    #[error("Metrics error: {0}")]
    Metrics(#[from] MetricsError),

    /// VRAM monitoring and management errors
    #[error("VRAM error: {0}")]
    Vram(#[from] VramError),

    /// Model-related errors
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    /// Validation errors
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    /// Resource exhaustion errors
    #[error("Resource error: {0}")]
    Resource(#[from] ResourceError),

    /// Timeout errors
    #[error("Timeout error: {0}")]
    Timeout(#[from] TimeoutError),

    /// Internal errors (bugs, invariant violations)
    #[error("Internal error: {0}")]
    Internal(#[from] InternalError),

    /// External service errors
    #[error("External service error: {0}")]
    External(#[from] ExternalError),

    /// IO errors
    #[error("IO error: {0}")]
    Io(String),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Network errors
    #[error("Network error: {0}")]
    Network(String),

    /// Unknown/unexpected errors
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Configuration errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ConfigError {
    /// File not found
    #[error("Configuration file not found: {0}")]
    FileNotFound(String),

    /// Invalid file format
    #[error("Invalid configuration format: {0}")]
    InvalidFormat(String),

    /// Missing required field
    #[error("Missing required configuration field: {0}")]
    MissingField(String),

    /// Invalid field value
    #[error("Invalid value for field '{field}': {value}. {message}")]
    InvalidValue {
        /// Field name
        field: String,
        /// Invalid value
        value: String,
        /// Error message
        message: String,
    },

    /// Environment variable error
    #[error("Environment variable error: {0}")]
    EnvVar(String),

    /// Validation error
    #[error("Configuration validation failed: {0}")]
    Validation(String),
}

/// Backend-related errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum BackendError {
    /// Backend not available
    #[error("Backend '{backend}' is not available. {reason}")]
    NotAvailable {
        /// Backend name
        backend: String,
        /// Reason for unavailability
        reason: String,
    },

    /// Backend not ready
    #[error("Backend '{backend}' is not ready. {details}")]
    NotReady {
        /// Backend name
        backend: String,
        /// Details
        details: String,
    },

    /// Model not found on backend
    #[error("Model '{model}' not found on backend '{backend}'")]
    ModelNotFound {
        /// Model name
        model: String,
        /// Backend name
        backend: String,
    },

    /// Model loading failed
    #[error("Failed to load model '{model}' on backend '{backend}': {reason}")]
    LoadFailed {
        /// Model name
        model: String,
        /// Backend name
        backend: String,
        /// Failure reason
        reason: String,
    },

    /// Model unloading failed
    #[error("Failed to unload model '{model}' from backend '{backend}': {reason}")]
    UnloadFailed {
        /// Model name
        model: String,
        /// Backend name
        backend: String,
        /// Failure reason
        reason: String,
    },

    /// Backend communication error
    #[error("Communication error with backend '{backend}': {message}")]
    Communication {
        /// Backend name
        backend: String,
        /// Error message
        message: String,
    },

    /// Backend-specific errors
    #[error("Backend '{backend}' error: {message}")]
    BackendSpecific {
        /// Backend name
        backend: String,
        /// Error message
        message: String,
    },

    /// vLLM-specific errors
    #[error("vLLM error: {0}")]
    Vllm(String),

    /// llama.cpp-specific errors
    #[error("llama.cpp error: {0}")]
    LlamaCpp(String),

    /// Resource constraints
    #[error("Backend resource constraint: {0}")]
    ResourceConstraint(String),

    /// Timeout
    #[error("Backend operation timeout: {0}")]
    Timeout(String),

    /// Invalid value
    #[error("Invalid value for field '{field}': {value}. {message}")]
    InvalidValue {
        /// Field name
        field: String,
        /// Invalid value
        value: String,
        /// Error message
        message: String,
    },
}

/// API and HTTP errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ApiError {
    /// Invalid request
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// Authorization failed
    #[error("Authorization failed: {0}")]
    Authorization(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Endpoint not found
    #[error("Endpoint not found: {0}")]
    NotFound(String),

    /// Method not allowed
    #[error("Method not allowed: {0}")]
    MethodNotAllowed(String),

    /// Bad request
    #[error("Bad request: {0}")]
    BadRequest(String),

    /// Request timeout
    #[error("Request timeout: {0}")]
    RequestTimeout(String),

    /// Payload too large
    #[error("Payload too large: {0}")]
    PayloadTooLarge(String),

    /// Internal server error
    #[error("Internal server error: {0}")]
    InternalServerError(String),

    /// Service unavailable
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

/// Pipeline processing errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum PipelineError {
    /// Input processing error
    #[error("Input processing error: {0}")]
    Input(String),

    /// Output processing error
    #[error("Output processing error: {0}")]
    Output(String),

    /// Batch processing error
    #[error("Batch processing error: {0}")]
    Batch(String),

    /// Stream processing error
    #[error("Stream processing error: {0}")]
    Stream(String),

    /// Checkpoint error
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    /// Validation error in pipeline
    #[error("Pipeline validation error: {0}")]
    Validation(String),

    /// Pipeline timeout
    #[error("Pipeline timeout: {0}")]
    Timeout(String),

    /// Pipeline interrupted
    #[error("Pipeline interrupted: {0}")]
    Interrupted(String),
}

/// Metrics collection and reporting errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum MetricsError {
    /// Metrics collection failed
    #[error("Metrics collection failed: {0}")]
    Collection(String),

    /// Metrics export failed
    #[error("Metrics export failed: {0}")]
    Export(String),

    /// Metrics storage error
    #[error("Metrics storage error: {0}")]
    Storage(String),

    /// Invalid metrics format
    #[error("Invalid metrics format: {0}")]
    InvalidFormat(String),

    /// Metrics aggregation error
    #[error("Metrics aggregation error: {0}")]
    Aggregation(String),
}

/// VRAM monitoring and management errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum VramError {
    /// NVIDIA NVML initialization failed
    #[error("NVML initialization failed: {0}")]
    NvmlInit(String),

    /// GPU not found
    #[error("GPU not found: {0}")]
    GpuNotFound(String),

    /// VRAM query failed
    #[error("VRAM query failed: {0}")]
    QueryFailed(String),

    /// Insufficient VRAM
    #[error("Insufficient VRAM: requested {requested_gb:.2}GB, available {available_gb:.2}GB")]
    InsufficientVram {
        /// Requested VRAM in GB
        requested_gb: f64,
        /// Available VRAM in GB
        available_gb: f64,
    },

    /// VRAM allocation failed
    #[error("VRAM allocation failed: {0}")]
    AllocationFailed(String),

    /// VRAM monitoring failed
    #[error("VRAM monitoring failed: {0}")]
    MonitoringFailed(String),
}

/// Model-related errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ModelError {
    /// Model not found
    #[error("Model not found: {0}")]
    NotFound(String),

    /// Model format not supported
    #[error("Model format not supported: {0}")]
    FormatNotSupported(String),

    /// Model corrupted or invalid
    #[error("Model corrupted or invalid: {0}")]
    Corrupted(String),

    /// Model metadata error
    #[error("Model metadata error: {0}")]
    Metadata(String),

    /// Model version mismatch
    #[error("Model version mismatch: {0}")]
    VersionMismatch(String),

    /// Model quantization error
    #[error("Model quantization error: {0}")]
    Quantization(String),

    /// Model compatibility error
    #[error("Model compatibility error: {0}")]
    Compatibility(String),
}

/// Validation errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    /// Input validation failed
    #[error("Input validation failed: {0}")]
    Input(String),

    /// Output validation failed
    #[error("Output validation failed: {0}")]
    Output(String),

    /// Schema validation failed
    #[error("Schema validation failed: {0}")]
    Schema(String),

    /// Constraint violation
    #[error("Constraint violation: {0}")]
    Constraint(String),

    /// Type validation failed
    #[error("Type validation failed: {0}")]
    Type(String),

    /// Range validation failed
    #[error("Range validation failed: {0}")]
    Range(String),
}

/// Resource exhaustion errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ResourceError {
    /// Memory exhausted
    #[error("Memory exhausted: {0}")]
    Memory(String),

    /// CPU exhausted
    #[error("CPU exhausted: {0}")]
    Cpu(String),

    /// Disk space exhausted
    #[error("Disk space exhausted: {0}")]
    Disk(String),

    /// File descriptors exhausted
    #[error("File descriptors exhausted: {0}")]
    FileDescriptors(String),

    /// Connection pool exhausted
    #[error("Connection pool exhausted: {0}")]
    Connections(String),

    /// Thread pool exhausted
    #[error("Thread pool exhausted: {0}")]
    Threads(String),
}

/// Timeout errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum TimeoutError {
    /// Operation timeout
    #[error("Operation timeout after {duration_ms}ms: {operation}")]
    Operation {
        /// Operation name
        operation: String,
        /// Timeout duration in milliseconds
        duration_ms: u64,
    },

    /// Connection timeout
    #[error("Connection timeout: {0}")]
    Connection(String),

    /// Read timeout
    #[error("Read timeout: {0}")]
    Read(String),

    /// Write timeout
    #[error("Write timeout: {0}")]
    Write(String),

    /// Overall timeout
    #[error("Overall timeout: {0}")]
    Overall(String),
}

/// Internal errors (bugs, invariant violations)
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum InternalError {
    /// Invariant violation
    #[error("Invariant violation: {0}")]
    Invariant(String),

    /// Unexpected state
    #[error("Unexpected state: {0}")]
    UnexpectedState(String),

    /// Logic error
    #[error("Logic error: {0}")]
    Logic(String),

    /// Assertion failed
    #[error("Assertion failed: {0}")]
    Assertion(String),

    /// Unimplemented feature
    #[error("Unimplemented feature: {0}")]
    Unimplemented(String),

    /// Data corruption
    #[error("Data corruption: {0}")]
    DataCorruption(String),
}

/// External service errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ExternalError {
    /// Database error
    #[error("Database error: {0}")]
    Database(String),

    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),

    /// Message queue error
    #[error("Message queue error: {0}")]
    MessageQueue(String),

    /// External API error
    #[error("External API error: {0}")]
    Api(String),

    /// File system error
    #[error("File system error: {0}")]
    FileSystem(String),

    /// Network service error
    #[error("Network service error: {0}")]
    NetworkService(String),
}

/// Result type alias for TensorForge operations
pub type TensorForgeResult<T> = std::result::Result<T, TensorForgeError>;

impl TensorForgeError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            TensorForgeError::Timeout(_)
            | TensorForgeError::Resource(_)
            | TensorForgeError::Network(_)
            | TensorForgeError::External(_)
            | TensorForgeError::Backend(BackendError::Communication { .. })
            | TensorForgeError::Backend(BackendError::Timeout(_))
            | TensorForgeError::Api(ApiError::RequestTimeout(_))
            | TensorForgeError::Api(ApiError::ServiceUnavailable(_))
            | TensorForgeError::Pipeline(PipelineError::Timeout(_))
            | TensorForgeError::Pipeline(PipelineError::Interrupted(_)) => true,

            _ => false,
        }
    }

    /// Get HTTP status code for the error
    pub fn status_code(&self) -> StatusCode {
        match self {
            TensorForgeError::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
            TensorForgeError::Backend(BackendError::NotAvailable { .. })
            | TensorForgeError::Backend(BackendError::NotReady { .. })
            | TensorForgeError::Api(ApiError::ServiceUnavailable(_)) => StatusCode::SERVICE_UNAVAILABLE,

            TensorForgeError::Backend(BackendError::ModelNotFound { .. })
            | TensorForgeError::Model(ModelError::NotFound(_))
            | TensorForgeError::Api(ApiError::NotFound(_)) => StatusCode::NOT_FOUND,

            TensorForgeError::Api(ApiError::InvalidRequest(_))
            | TensorForgeError::Api(ApiError::BadRequest(_))
            | TensorForgeError::Validation(_)
            | TensorForgeError::Backend(BackendError::InvalidValue { .. }) => StatusCode::BAD_REQUEST,

            TensorForgeError::Api(ApiError::Authentication(_)) => StatusCode::UNAUTHORIZED,
            TensorForgeError::Api(ApiError::Authorization(_)) => StatusCode::FORBIDDEN,
            TensorForgeError::Api(ApiError::RateLimitExceeded(_)) => StatusCode::TOO_MANY_REQUESTS,
            TensorForgeError::Api(ApiError::MethodNotAllowed(_)) => StatusCode::METHOD_NOT_ALLOWED,
            TensorForgeError::Api(ApiError::RequestTimeout(_)) => StatusCode::REQUEST_TIMEOUT,
            TensorForgeError::Api(ApiError::PayloadTooLarge(_)) => StatusCode::PAYLOAD_TOO_LARGE,

            TensorForgeError::Resource(ResourceError::Memory(_))
            | TensorForgeError::Vram(VramError::InsufficientVram { .. }) => StatusCode::INSUFFICIENT_STORAGE,

            TensorForgeError::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,

            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Get a user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            TensorForgeError::Config(err) => format!("Configuration error: {}", err),
            TensorForgeError::Backend(err) => format!("Backend error: {}", err),
            TensorForgeError::Api(err) => format!("API error: {}", err),
            TensorForgeError::Pipeline(err) => format!("Pipeline error: {}", err),
            TensorForgeError::Metrics(err) => format!("Metrics error: {}", err),
            TensorForgeError::Vram(err) => format!("VRAM error: {}", err),
            TensorForgeError::Model(err) => format!("Model error: {}", err),
            TensorForgeError::Validation(err) => format!("Validation error: {}", err),
            TensorForgeError::Resource(err) => format!("Resource error: {}", err),
            TensorForgeError::Timeout(err) => format!("Timeout error: {}", err),
            TensorForgeError::Internal(err) => format!("Internal error: {}", err),
            TensorForgeError::External(err) => format!("External service error: {}", err),
            TensorForgeError::Io(msg) => format!("IO error: {}", msg),
            TensorForgeError::Serialization(msg) => format!("Serialization error: {}", msg),
            TensorForgeError::Network(msg) => format!("Network error: {}", msg),
            TensorForgeError::Unknown(msg) => format!("Unknown error: {}", msg),
        }
    }

    /// Get error category for logging and monitoring
    pub fn category(&self) -> &'static str {
        match self {
            TensorForgeError::Config(_) => "config",
            TensorForgeError::Backend(_) => "backend",
            TensorForgeError::Api(_) => "api",
            TensorForgeError::Pipeline(_) => "pipeline",
            TensorForgeError::Metrics(_) => "metrics",
            TensorForgeError::Vram(_) => "vram",
            TensorForgeError::Model(_) => "model",
            TensorForgeError::Validation(_) => "validation",
            TensorForgeError::Resource(_) => "resource",
            TensorForgeError::Timeout(_) => "timeout",
            TensorForgeError::Internal(_) => "internal",
            TensorForgeError::External(_) => "external",
            TensorForgeError::Io(_) => "io",
            TensorForgeError::Serialization(_) => "serialization",
            TensorForgeError::Network(_) => "network",
            TensorForgeError::Unknown(_) => "unknown",
        }
    }
}

// Implement From for common error types
impl From<io::Error> for TensorForgeError {
    fn from(err: io::Error) -> Self {
        TensorForgeError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for TensorForgeError {
    fn from(err: serde_json::Error) -> Self {
        TensorForgeError::Serialization(err.to_string())
    }
}

impl From<toml::de::Error> for TensorForgeError {
    fn from(err: toml::de::Error) -> Self {
        TensorForgeError::Config(ConfigError::InvalidFormat(err.to_string()))
    }
}

impl From<toml::ser::Error> for TensorForgeError {
    fn from(err: toml::ser::Error) -> Self {
        TensorForgeError::Serialization(err.to_string())
    }
}

impl From<ParseIntError> for TensorForgeError {
    fn from(err: ParseIntError) -> Self {
        TensorForgeError::Validation(ValidationError::Type(err.to_string()))
    }
}

impl From<ParseBoolError> for TensorForgeError {
    fn from(err: ParseBoolError) -> Self {
        TensorForgeError::Validation(ValidationError::Type(err.to_string()))
    }
}

impl From<reqwest::Error> for TensorForgeError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            TensorForgeError::Timeout(TimeoutError::Connection(err.to_string()))
        } else if err.is_connect() {
            TensorForgeError::Network(err.to_string())
        } else {
            TensorForgeError::External(ExternalError::Api(err.to_string()))
        }
    }
}

impl From<sqlx::Error> for TensorForgeError {
    fn from(err: sqlx::Error) -> Self {
        TensorForgeError::External(ExternalError::Database(err.to_string()))
    }
}

impl From<hyper::Error> for TensorForgeError {
    fn from(err: hyper::Error) -> Self {
        TensorForgeError::Network(err.to_string())
    }
}

impl From<tokio::task::JoinError> for TensorForgeError {
    fn from(err: tokio::task::JoinError) -> Self {
        TensorForgeError::Internal(InternalError::UnexpectedState(err.to_string()))
    }
}

impl From<Arc<dyn std::error::Error + Send + Sync>> for TensorForgeError {
    fn from(err: Arc<dyn std::error::Error + Send + Sync>) -> Self {
        TensorForgeError::Unknown(err.to_string())
    }
}

// Display implementation handled by thiserror::Error derive

// Display implementation handled by thiserror::Error derive

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::StatusCode;

    #[test]
    fn test_error_creation() {
        let config_err = TensorForgeError::Config(ConfigError::FileNotFound("config.toml".to_string()));
        assert_eq!(config_err.category(), "config");
        assert!(!config_err.is_retryable());

        let timeout_err = TensorForgeError::Timeout(TimeoutError::Operation {
            operation: "test".to_string(),
            duration_ms: 1000,
        });
        assert_eq!(timeout_err.category(), "timeout");
        assert!(timeout_err.is_retryable());
    }

    #[test]
    fn test_status_codes() {
        let not_found = TensorForgeError::Backend(BackendError::ModelNotFound {
            model: "test".to_string(),
            backend: "vllm".to_string(),
        });
        assert_eq!(not_found.status_code(), StatusCode::NOT_FOUND);

        let bad_request = TensorForgeError::Api(ApiError::BadRequest("invalid input".to_string()));
        assert_eq!(bad_request.status_code(), StatusCode::BAD_REQUEST);

        let service_unavailable = TensorForgeError::Backend(BackendError::NotAvailable {
            backend: "vllm".to_string(),
            reason: "down for maintenance".to_string(),
        });
        assert_eq!(service_unavailable.status_code(), StatusCode::SERVICE_UNAVAILABLE);

        let internal_error = TensorForgeError::Internal(InternalError::Invariant("bug".to_string()));
        assert_eq!(internal_error.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_user_messages() {
        let err = TensorForgeError::Vram(VramError::InsufficientVram {
            requested_gb: 100.0,
            available_gb: 50.0,
        });
        let msg = err.user_message();
        assert!(msg.contains("VRAM error"));
        assert!(msg.contains("Insufficient VRAM"));
    }

    #[test]
    fn test_from_conversions() {
        let io_err: TensorForgeError = io::Error::new(io::ErrorKind::NotFound, "file not found").into();
        assert_eq!(io_err.category(), "io");

        let json_err: TensorForgeError = serde_json::Error::syntax("invalid json", 1, 1).into();
        assert_eq!(json_err.category(), "serialization");

        let parse_err: TensorForgeError = "not_a_number".parse::<i32>().unwrap_err().into();
        assert_eq!(parse_err.category(), "validation");
    }

    #[test]
    fn test_result_alias() {
        fn success() -> TensorForgeResult<String> {
            Ok("success".to_string())
        }

        fn failure() -> TensorForgeResult<String> {
            Err(TensorForgeError::Unknown("test".to_string()))
        }

        assert!(success().is_ok());
        assert!(failure().is_err());
    }

    #[test]
    fn test_display_implementations() {
        let config_err = ConfigError::FileNotFound("config.toml".to_string());
        let display = format!("{}", config_err);
        assert!(display.contains("Configuration file not found"));

        let backend_err = BackendError::NotAvailable {
            backend: "vllm".to_string(),
            reason: "service down".to_string(),
        };
        let display = format!("{}", backend_err);
        assert!(display.contains("Backend 'vllm' is not available"));
    }
}
