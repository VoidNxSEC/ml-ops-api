/// vLLM backend client
///
/// vLLM exposes an OpenAI-compatible API at /v1/chat/completions and
/// /v1/embeddings — identical wire format to llama.cpp. Inference proxying
/// is handled generically via the router-selected base_url; this module
/// exists for vLLM-specific operations (model introspection, metrics).

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct VllmConfig {
    pub base_url: String,
    pub timeout_secs: u64,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            base_url: std::env::var("VLLM_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string()),
            timeout_secs: 300,
        }
    }
}

pub struct VllmBackend {
    client: Client,
    config: VllmConfig,
}

/// vLLM /health response
#[derive(Debug, Deserialize, Serialize)]
pub struct VllmHealthResponse {
    #[serde(default)]
    pub status: Option<String>,
}

/// vLLM /v1/models response (subset)
#[derive(Debug, Deserialize, Serialize)]
pub struct VllmModelsResponse {
    pub data: Vec<VllmModelEntry>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VllmModelEntry {
    pub id: String,
    pub object: String,
    #[serde(default)]
    pub owned_by: Option<String>,
}

impl VllmBackend {
    pub fn new(config: VllmConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .context("Failed to build vLLM HTTP client")?;
        Ok(Self { client, config })
    }

    pub fn with_defaults() -> Result<Self> {
        Self::new(VllmConfig::default())
    }

    /// GET /health — returns true if vLLM is ready to serve requests.
    pub async fn is_ready(&self) -> bool {
        self.client
            .get(format!("{}/health", self.config.base_url))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// GET /v1/models — list models currently loaded in vLLM.
    pub async fn list_models(&self) -> Result<Vec<VllmModelEntry>> {
        let resp = self
            .client
            .get(format!("{}/v1/models", self.config.base_url))
            .send()
            .await
            .context("Failed to reach vLLM /v1/models")?;

        let body: VllmModelsResponse = resp.json().await.context("Failed to parse vLLM models")?;
        Ok(body.data)
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vllm_backend_creation() {
        assert!(VllmBackend::with_defaults().is_ok());
    }
}
