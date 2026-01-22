//! vLLM backend error types
//!
//! Defines error types specific to the vLLM backend implementation,
//! with conversions to TensorForge's unified error system.

use serde_json::Value;
use std::fmt;
use thiserror::Error;
use tensorforge_core::{BackendError, TensorForgeError};
use tensorforge_core::error::ConfigError as CoreConfigError;

/// vLLM-specific errors
#[derive(Debug, Error)]
pub enum VllmError {
    /// Connection to vLLM server failed
    #[error("Failed to connect to vLLM server at {endpoint}: {source}")]
    ConnectionFailed {
        /// Server endpoint
        endpoint: String,
        /// Underlying error
        #[source]
        source: reqwest::Error,
    },

    /// vLLM server returned an error response
    #[error("vLLM server error: {status} - {message}")]
    ServerError {
        /// HTTP status code
        status: u16,
        /// Error message from server
        message: String,
        /// Raw response body (if available)
        response_body: Option<String>,
    },

    /// Invalid response format from vLLM server
    #[error("Invalid response format from vLLM server: {message}")]
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
    #[error("Model {model_id} is not loaded on vLLM backend")]
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
    #[error("vLLM health check failed: {reason}")]
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
    #[error("Invalid vLLM configuration: {reason}")]
    InvalidConfig {
        /// Configuration error reason
        reason: String,
        /// Field that caused the error
        field: Option<String>,
    },

    /// Request timeout
    #[error("Request to vLLM server timed out after {timeout_secs} seconds")]
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
    #[error("vLLM backend error: {message}")]
    Other {
        /// Error message
        message: String,
        /// Additional context
        context: Option<String>,
    },
}

impl VllmError {
    /// Check if the error is retryable
    ///
    /// Some vLLM errors may be transient and worth retrying,
    /// such as connection failures or timeouts.
    pub fn is_retryable(&self) -> bool {
        match self {
            VllmError::ConnectionFailed { .. }
            | VllmError::Timeout { .. }
            | VllmError::HttpClient { .. } => true,
            VllmError::ServerError { status, .. } => {
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
            VllmError::ServerError { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// Get user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            VllmError::ConnectionFailed { endpoint, .. } => {
                format!("Cannot connect to vLLM server at {}", endpoint)
            }
            VllmError::ServerError { message, .. } => {
                format!("vLLM server error: {}", message)
            }
            VllmError::ModelLoadFailed { model_id, reason, .. } => {
                format!("Failed to load model {}: {}", model_id, reason)
            }
            VllmError::InferenceFailed { reason, .. } => {
                format!("Inference failed: {}", reason)
            }
            VllmError::Timeout { timeout_secs, operation, .. } => {
                format!("{} timed out after {} seconds", operation, timeout_secs)
            }
            VllmError::InsufficientVram {
                required_mb,
                available_mb,
                ..
            } => {
                format!(
                    "Not enough GPU memory: need {}MB, only {}MB available",
                    required_mb, available_mb
                )
            }
            _ => self.to_string(),
        }
    }

    /// Create a connection failed error
    pub fn connection_failed(endpoint: &str, source: reqwest::Error) -> Self {
        VllmError::ConnectionFailed {
            endpoint: endpoint.to_string(),
            source,
        }
    }

    /// Create a server error
    pub fn server_error(status: u16, message: &str, response_body: Option<String>) -> Self {
        VllmError::ServerError {
            status,
            message: message.to_string(),
            response_body,
        }
    }

    /// Create a model load error
    pub fn model_load_failed(model_id: &str, reason: &str, details: Option<Value>) -> Self {
        VllmError::ModelLoadFailed {
            model_id: model_id.to_string(),
            reason: reason.to_string(),
            details,
        }
    }

    /// Create an inference error
    pub fn inference_failed(reason: &str, request_id: Option<String>, server_error: Option<Value>) -> Self {
        VllmError::InferenceFailed {
            reason: reason.to_string(),
            request_id,
            server_error,
        }
    }

    /// Create a timeout error
    pub fn timeout(timeout_secs: u64, operation: &str) -> Self {
        VllmError::Timeout {
            timeout_secs,
            operation: operation.to_string(),
        }
    }
}

impl From<VllmError> for TensorForgeError {
    fn from(error: VllmError) -> Self {
        match error {
            VllmError::ModelLoadFailed {
                model_id,
                reason,
                details,
            } => TensorForgeError::Backend(BackendError::LoadFailed {
                model: model_id,
                backend: "vllm".to_string(),
                reason,
            }),
            VllmError::ModelUnloadFailed { model_id, reason } => {
                TensorForgeError::Backend(BackendError::UnloadFailed {
                    model: model_id,
                    backend: "vllm".to_string(),
                    reason,
                })
            }
            VllmError::ModelNotFound { model_id } => TensorForgeError::Backend(BackendError::ModelNotFound {
                model: model_id,
                backend: "vllm".to_string(),
            }),
            VllmError::InferenceFailed {
                reason,
                request_id: _,
                server_error: _,
            } => TensorForgeError::Backend(BackendError::BackendSpecific {
                backend: "vllm".to_string(),
                message: reason,
            }),
            VllmError::ConnectionFailed { endpoint, source: _ } => {
                TensorForgeError::Backend(BackendError::Communication {
                    backend: "vllm".to_string(),
                    message: format!("Failed to connect to {}", endpoint),
                })
            }
            VllmError::ServerError {
                status,
                message,
                response_body: _,
            } => TensorForgeError::Backend(BackendError::BackendSpecific {
                backend: "vllm".to_string(),
                message: format!("HTTP {}: {}", status, message),
            }),
            VllmError::Timeout {
                timeout_secs,
                operation,
            } => TensorForgeError::Backend(BackendError::Timeout),
            VllmError::InsufficientVram {
                required_mb,
                available_mb,
                reason: _,
            } => TensorForgeError::Backend(BackendError::ResourceConstraint),
            VllmError::HealthCheckFailed { reason, .. } => TensorForgeError::Backend(BackendError::NotReady {
                backend: "vllm".to_string(),
                details: reason,
            }),
            VllmError::InvalidConfig { reason, field } => {
                let msg = if let Some(f) = field {
                    format!("Configuration error in field '{}': {}", f, reason)
                } else {
                    format!("Configuration error: {}", reason)
                };
                TensorForgeError::Config(tensorforge_core::ConfigError::Validation(msg))
            }
            // For other errors, wrap them as backend-specific errors
            error => TensorForgeError::Backend(BackendError::BackendSpecific {
                backend: "vllm".to_string(),
                message: error.to_string(),
            }),
        }
    }
}









impl From<CoreConfigError> for VllmError {
    fn from(error: CoreConfigError) -> Self {
        VllmError::InvalidConfig {
            reason: error.to_string(),
            field: None,
        }
    }
}

/// Result type alias for vLLM operations
pub type VllmResult<T> = Result<T, VllmError>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_vllm_error_creation() {
        let error = VllmError::connection_failed(
            "http://localhost:8000",
            reqwest::Error::new(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "test"),
        );
        assert!(matches!(error, VllmError::ConnectionFailed { .. }));
        assert!(error.is_retryable());

        let error = VllmError::server_error(500, "Internal server error", None);
        assert!(error.is_retryable());

        let error = VllmError::server_error(404, "Not found", None);
        assert!(!error.is_retryable());

        let error = VllmError::model_load_failed("test-model", "out of memory", Some(json!({"details": "vram_full"})));
        assert!(!error.is_retryable());
    }

    #[test]
    fn test_vllm_error_conversion_to_tensorforge() {
        let vllm_error = VllmError::model_load_failed("test-model", "out of memory", None);
        let tf_error: TensorForgeError = vllm_error.into();

        match tf_error {
            TensorForgeError::Backend(BackendError::LoadFailed { model, backend, reason }) => {
                assert_eq!(model, "test-model");
                assert_eq!(backend, "vllm");
                assert_eq!(reason, "out of memory");
            }
            _ => panic!("Expected BackendError::LoadFailed"),
        }
    }

    #[test]
    fn test_vllm_error_user_message() {
        let error = VllmError::ConnectionFailed {
            endpoint: "http://localhost:8000".to_string(),
            source: reqwest::Error::new(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "test"),
        };
        let message = error.user_message();
        assert!(message.contains("Cannot connect to vLLM server at http://localhost:8000"));

        let error = VllmError::InsufficientVram {
            reason: "Model too large".to_string(),
            required_mb: 16000,
            available_mb: 8000,
        };
        let message = error.user_message();
        assert!(message.contains("Not enough GPU memory: need 16000MB, only 8000MB available"));
    }

    #[test]
    fn test_result_alias() {
        let success: VllmResult<()> = Ok(());
        assert!(success.is_ok());

        let failure: VllmResult<()> = Err(VllmError::Other {
            message: "test".to_string(),
            context: None,
        });
        assert!(failure.is_err());
    }
}
