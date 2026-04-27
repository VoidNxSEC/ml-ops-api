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

    /// GET /metrics — scrape Prometheus exposition and extract request pressure.
    /// Returns `(running, waiting, total_capacity)` if parseable, else None.
    ///
    /// Key metrics used:
    ///   vllm:num_requests_running   — slots actively generating
    ///   vllm:num_requests_waiting   — requests queued (scheduler backpressure)
    ///   vllm:gpu_cache_usage_perc   — KV cache fill %
    pub async fn get_capacity(&self) -> Option<(u32, u32)> {
        let text = self
            .client
            .get(format!("{}/metrics", self.config.base_url))
            .send()
            .await
            .ok()?
            .text()
            .await
            .ok()?;

        let mut running: Option<f64> = None;
        let mut waiting: Option<f64> = None;

        for line in text.lines() {
            if line.starts_with('#') {
                continue;
            }
            if let Some(val) = extract_metric(line, "vllm:num_requests_running") {
                running = Some(val);
            }
            if let Some(val) = extract_metric(line, "vllm:num_requests_waiting") {
                waiting = Some(val);
            }
        }

        // slots_free = inverse of running; we approximate max_concurrent from the
        // gpu_cache_usage metric (0% usage ≈ all slots free).
        // Since vLLM doesn't expose a fixed slot count, treat running + waiting
        // as load: (0, running+waiting) → caller uses this for load scoring.
        let r = running.unwrap_or(0.0) as u32;
        let w = waiting.unwrap_or(0.0) as u32;

        // Return (idle_estimate, total_estimate): idle = 0 when under load
        // total is arbitrary; router normalises via ratio — use 16 as baseline
        let total: u32 = 16;
        let busy = (r + w).min(total);
        Some((total - busy, total))
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }
}

/// Parse `<metric_name>{...} <value>` or `<metric_name> <value>` lines.
fn extract_metric(line: &str, name: &str) -> Option<f64> {
    if !line.starts_with(name) {
        return None;
    }
    line.split_whitespace().last()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vllm_backend_creation() {
        assert!(VllmBackend::with_defaults().is_ok());
    }
}
