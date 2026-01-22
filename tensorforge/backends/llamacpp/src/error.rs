//! llama.cpp backend error types
//!
//! Defines error types specific to the llama.cpp backend implementation,
//! with conversions to TensorForge's unified error system.

use serde_json::Value;
use std::fmt;
use thiserror::Error;
use tensorforge_core::{BackendError, TensorForgeError};

/// llama.cpp-specific errors
#[derive(Debug, Error)]
pub enum LlamaCppError {
    /// Connection to llama.cpp server failed
    #[error("Failed to connect to llama.cpp server at {endpoint}: {source}")]
    ConnectionFailed {
        /// Server endpoint
        endpoint: String,
        /// Underlying error
        #[source]
        source: reqwest::Error,
    },

    /// llama.cpp server returned an error response
    #[error("llama.cpp server error: {status} - {message}")]
    ServerError {
        /// HTTP status code
        status: u16,
        /// Error message from server
        message: String,
        /// Raw response body (if available)
        response_body: Option<String>,
    },

    /// Invalid response format from llama.cpp server
    #[error("Invalid response format from llama.cpp server: {message}")]
    InvalidResponse {
        /// Error message
        message: String,
        /// Raw response (if available)
        raw_response: Option<String>,
        /// Expected format description
        expected: String,
    },

    /// Model loading failed
    #[error("Failed to load model {model_id}: {reason}")]
    ModelLoadFailed {
        /// Model identifier
        model_id: String,
        /// Failure reason
        reason: String,
        /// Additional error details
        details: Option<Value>,
    },

    /// Model unloading failed
    #[error("Failed to unload model {model_id}: {reason}")]
    ModelUnloadFailed {
        /// Model identifier
        model_id: String,
        /// Failure reason
        reason: String,
    },

    /// Model not found/loaded
    #[error("Model {model_id} is not loaded on llama.cpp backend")]
    ModelNotFound {
        /// Model identifier
        model_id: String,
    },

    /// Inference request failed
    #[error("Inference request failed: {reason}")]
    InferenceFailed {
        /// Failure reason
        reason: String,
        /// Request ID (if available)
        request_id: Option<String>,
        /// Error details from server
        server_error: Option<Value>,
    },

    /// Health check failed
    #[error("llama.cpp health check failed: {reason}")]
    HealthCheckFailed {
        /// Failure reason
        reason: String,
        /// Server status (if available)
        server_status: Option<String>,
    },

    /// VRAM query failed
    #[error("Failed to query VRAM usage: {reason}")]
    VramQueryFailed {
        /// Failure reason
        reason: String,
    },

    /// Invalid configuration
    #[error("Invalid llama.cpp configuration: {reason}")]
    InvalidConfig {
        /// Configuration error reason
        reason: String,
        /// Field that caused the error
        field: Option<String>,
    },

    /// Request timeout
    #[error("Request to llama.cpp server timed out after {timeout_secs} seconds")]
    Timeout {
        /// Timeout duration in seconds
        timeout_secs: u64,
        /// Operation that timed out
        operation: String,
    },

    /// Insufficient VRAM for operation
    #[error("Insufficient VRAM: {reason}")]
    InsufficientVram {
        /// Failure reason
        reason: String,
        /// Required VRAM in MB
        required_mb: u64,
        /// Available VRAM in MB
        available_mb: u64,
    },

    /// Invalid GPU layers configuration
    #[error("Invalid GPU layers configuration: {reason}")]
    InvalidGpuLayers {
        /// Failure reason
        reason: String,
        /// Requested GPU layers
        requested_layers: i32,
        /// Maximum supported layers
        max_layers: i32,
    },

    /// Serialization/deserialization error
    #[error("Serialization error: {source}")]
    Serialization {
        /// Underlying error
        #[from]
        source: serde_json::Error,
    },

    /// HTTP client error
    #[error("HTTP client error: {source}")]
    HttpClient {
        /// Underlying error
        #[from]
        source: reqwest::Error,
    },

    /// I/O error
    #[error("I/O error: {source}")]
    Io {
        /// Underlying error
        #[from]
        source: std::io::Error,
    },

    /// URL parsing error
    #[error("URL parsing error: {source}")]
    UrlParse {
        /// Underlying error
        #[from]
        source: url::ParseError,
    },

    /// Other/uncategorized error
    #[error("llama.cpp backend error: {message}")]
    Other {
        /// Error message
        message: String,
        /// Additional context
        context: Option<String>,
    },
}

impl LlamaCppError {
    /// Check if the error is retryable
    ///
    /// Some llama.cpp errors may be transient and worth retrying,
    /// such as connection failures or timeouts.
    pub fn is_retryable(&self) -> bool {
        match self {
            LlamaCppError::ConnectionFailed { .. }
            | LlamaCppError::Timeout { .. }
            | LlamaCppError::HttpClient { .. } => true,
            LlamaCppError::ServerError { status, .. } => {
                // 5xx errors are server errors and may be retryable
                // 429 (rate limit) might be retryable with backoff
                *status == 429 || (*status >= 500 && *status < 600)
            }
            _ => false,
        }
    }

    /// Get HTTP status code if applicable
    pub fn status_code(&self) -> Option<u16> {
        match self {
            LlamaCppError::ServerError { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// Get user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            LlamaCppError::ConnectionFailed { endpoint, .. } => {
                format!("Cannot connect to llama.cpp server at {}", endpoint)
            }
            LlamaCppError::ServerError { message, .. } => {
                format!("llama.cpp server error: {}", message)
            }
            LlamaCppError::ModelLoadFailed { model_id, reason, .. } => {
                format!("Failed to load model {}: {}", model_id, reason)
            }
            LlamaCppError::InferenceFailed { reason, .. } => {
                format!("Inference failed: {}", reason)
            }
            LlamaCppError::Timeout { timeout_secs, operation, .. } => {
                format!("{} timed out after {} seconds", operation, timeout_secs)
            }
            LlamaCppError::InsufficientVram {
                required_mb,
                available_mb,
                ..
            } => {
                format!(
                    "Not enough GPU memory: need {}MB, only {}MB available",
                    required_mb, available_mb
                )
            }
            LlamaCppError::InvalidGpuLayers {
                requested_layers,
                max_layers,
                ..
            } => {
                format!(
                    "Too many GPU layers requested: {} (max: {})",
                    requested_layers, max_layers
                )
            }
            _ => self.to_string(),
        }
    }

    /// Create a connection failed error
    pub fn connection_failed(endpoint: &str, source: reqwest::Error) -> Self {
        LlamaCppError::ConnectionFailed {
            endpoint: endpoint.to_string(),
            source,
        }
    }

    /// Create a server error
    pub fn server_error(status: u16, message: &str, response_body: Option<String>) -> Self {
        LlamaCppError::ServerError {
            status,
            message: message.to_string(),
            response_body,
        }
    }

    /// Create a model load error
    pub fn model_load_failed(model_id: &str, reason: &str, details: Option<Value>) -> Self {
        LlamaCppError::ModelLoadFailed {
            model_id: model_id.to_string(),
            reason: reason.to_string(),
            details,
        }
    }

    /// Create an inference error
    pub fn inference_failed(reason: &str, request_id: Option<String>, server_error: Option<Value>) -> Self {
        LlamaCppError::InferenceFailed {
            reason: reason.to_string(),
            request_id,
            server_error,
        }
    }

    /// Create a timeout error
    pub fn timeout(timeout_secs: u64, operation: &str) -> Self {
        LlamaCppError::Timeout {
            timeout_secs,
            operation: operation.to_string(),
        }
    }

    /// Create an insufficient VRAM error
    pub fn insufficient_vram(required_mb: u64, available_mb: u64, reason: &str) -> Self {
        LlamaCppError::InsufficientVram {
            reason: reason.to_string(),
            required_mb,
            available_mb,
        }
    }

    /// Create an invalid GPU layers error
    pub fn invalid_gpu_layers(requested_layers: i32, max_layers: i32, reason: &str) -> Self {
        LlamaCppError::InvalidGpuLayers {
            reason: reason.to_string(),
            requested_layers,
            max_layers,
        }
    }
}

impl From<LlamaCppError> for TensorForgeError {
    fn from(error: LlamaCppError) -> Self {
        match error {
            LlamaCppError::ModelLoadFailed {
                model_id,
                reason,
                details: _,
            } => TensorForgeError::Backend(BackendError::LoadFailed {
                model: model_id,
                backend: "llamacpp".to_string(),
                reason,
            }),
            LlamaCppError::ModelUnloadFailed { model_id, reason } => {
                TensorForgeError::Backend(BackendError::UnloadFailed {
                    model: model_id,
                    backend: "llamacpp".to_string(),
                    reason,
                })
            }
            LlamaCppError::ModelNotFound { model_id } => TensorForgeError::Backend(BackendError::ModelNotFound {
                model: model_id,
                backend: "llamacpp".to_string(),
            }),
            LlamaCppError::InferenceFailed {
                reason,
                request_id: _,
                server_error: _,
            } => TensorForgeError::Backend(BackendError::BackendSpecific {
                backend: "llamacpp".to_string(),
                message: reason,
            }),
            LlamaCppError::ConnectionFailed { endpoint, source: _ } => {
                TensorForgeError::Backend(BackendError::Communication {
                    backend: "llamacpp".to_string(),
                    message: format!("Failed to connect to {}", endpoint),
                })
            }
            LlamaCppError::ServerError {
                status,
                message,
                response_body: _,
            } => TensorForgeError::Backend(BackendError::BackendSpecific {
                backend: "llamacpp".to_string(),
                message: format!("HTTP {}: {}", status, message),
            }),
            LlamaCppError::Timeout {
                timeout_secs: _,
                operation: _,
            } => TensorForgeError::Backend(BackendError::Timeout),
            LlamaCppError::InsufficientVram {
                required_mb: _,
                available_mb: _,
                reason: _,
            } => TensorForgeError::Backend(BackendError::ResourceConstraint),
            LlamaCppError::InvalidGpuLayers {
                requested_layers: _,
                max_layers: _,
                reason,
            } => TensorForgeError::Backend(BackendError::BackendSpecific {
                backend: "llamacpp".to_string(),
                message: reason,
            }),
            LlamaCppError::HealthCheckFailed { reason, .. } => TensorForgeError::Backend(BackendError::NotReady {
                backend: "llamacpp".to_string(),
                details: reason,
            }),
            LlamaCppError::InvalidConfig { reason, field } => {
                let msg = if let Some(f) = field {
                    format!("Configuration error in field '{}': {}", f, reason)
                } else {
                    format!("Configuration error: {}", reason)
                };
                TensorForgeError::Config(tensorforge_core::ConfigError::Validation(msg))
            }
            // For other errors, wrap them as backend-specific errors
            error => TensorForgeError::Backend(BackendError::BackendSpecific {
                backend: "llamacpp".to_string(),
                message: error.to_string(),
            }),
        }
    }
}

impl From<reqwest::Error> for LlamaCppError {
    fn from(error: reqwest::Error) -> Self {
        LlamaCppError::HttpClient { source: error }
    }
}

impl From<serde_json::Error> for LlamaCppError {
    fn from(error: serde_json::Error) -> Self {
        LlamaCppError::Serialization { source: error }
    }
}

impl From<std::io::Error> for LlamaCppError {
    fn from(error: std::io::Error) -> Self {
        LlamaCppError::Io { source: error }
    }
}

impl From<url::ParseError> for LlamaCppError {
    fn from(error: url::ParseError) -> Self {
        LlamaCppError::UrlParse { source: error }
    }
}

impl From<config::ConfigError> for LlamaCppError {
    fn from(error: config::ConfigError) -> Self {
        LlamaCppError::InvalidConfig {
            reason: error.to_string(),
            field: None,
        }
    }
}

/// Result type alias for llama.cpp operations
pub type LlamaCppResult<T> = Result<T, LlamaCppError>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_llamacpp_error_creation() {
        let error = LlamaCppError::connection_failed(
            "http://localhost:8080",
            reqwest::Error::new(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "test"),
        );
        assert!(matches!(error, LlamaCppError::ConnectionFailed { .. }));
        assert!(error.is_retryable());

        let error = LlamaCppError::server_error(500, "Internal server error", None);
        assert!(error.is_retryable());

        let error = LlamaCppError::server_error(404, "Not found", None);
        assert!(!error.is_retryable());

        let error = LlamaCppError::model_load_failed("test-model", "out of memory", Some(json!({"details": "vram_full"})));
        assert!(!error.is_retryable());
    }

    #[test]
    fn test_llamacpp_error_conversion_to_tensorforge() {
        let llamacpp_error = LlamaCppError::model_load_failed("test-model", "out of memory", None);
        let tf_error: TensorForgeError = llamacpp_error.into();

        match tf_error {
            TensorForgeError::Backend(BackendError::LoadFailed { model, backend, reason }) => {
                assert_eq!(model, "test-model");
                assert_eq!(backend, "llamacpp");
                assert_eq!(reason, "out of memory");
            }
            _ => panic!("Expected BackendError::LoadFailed"),
        }
    }

    #[test]
    fn test_llamacpp_error_user_message() {
        let error = LlamaCppError::ConnectionFailed {
            endpoint: "http://localhost:8080".to_string(),
            source: reqwest::Error::new(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "test"),
        };
        let message = error.user_message();
        assert!(message.contains("Cannot connect to llama.cpp server at http://localhost:8080"));

        let error = LlamaCppError::InsufficientVram {
            reason: "Model too large".to_string(),
            required_mb: 16000,
            available_mb: 8000,
        };
        let message = error.user_message();
        assert!(message.contains("Not enough GPU memory: need 16000MB, only 8000MB available"));

        let error = LlamaCppError::InvalidGpuLayers {
            reason: "Exceeds maximum".to_string(),
            requested_layers: 120,
            max_layers: 100,
        };
        let message = error.user_message();
        assert!(message.contains("Too many GPU layers requested: 120 (max: 100)"));
    }

    #[test]
    fn test_result_alias() {
        let success: LlamaCppResult<()> = Ok(());
        assert!(success.is_ok());

        let failure: LlamaCppResult<()> = Err(LlamaCppError::Other {
            message: "test".to_string(),
            context: None,
        });
        assert!(failure.is_err());
    }
}
