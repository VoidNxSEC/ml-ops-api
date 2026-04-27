pub mod llamacpp;
pub mod vllm;

use crate::models::BackendInfo;
use llamacpp::LlamaCppBackend;

pub struct BackendDriver;

impl BackendDriver {
    /// Detect and return all configured backends by probing their health endpoints.
    pub async fn list_backends() -> anyhow::Result<Vec<BackendInfo>> {
        let mut backends = vec![];

        // llama.cpp — always included
        let llamacpp_url = std::env::var("LLAMACPP_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());

        let (lc_status, lc_model) = match LlamaCppBackend::with_defaults() {
            Ok(b) => {
                if b.is_ready().await {
                    let model = b
                        .get_model_info()
                        .await
                        .ok()
                        .map(|m| m.model)
                        .filter(|s| !s.is_empty());
                    ("active".to_string(), model)
                } else {
                    ("inactive".to_string(), None)
                }
            }
            Err(_) => ("error".to_string(), None),
        };

        let lc_port = llamacpp_url
            .rsplit(':')
            .next()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8080u16);

        backends.push(BackendInfo {
            name: "llamacpp".to_string(),
            status: lc_status,
            backend_type: "api".to_string(),
            host: "127.0.0.1".to_string(),
            port: lc_port,
            loaded_model: lc_model,
            vram_usage_mb: None,
        });

        // vLLM — only if VLLM_URL is configured
        if let Ok(vllm_url) = std::env::var("VLLM_URL") {
            let vllm_port = vllm_url
                .rsplit(':')
                .next()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8000u16);

            let (vl_status, vl_model) = match vllm::VllmBackend::with_defaults() {
                Ok(b) => {
                    if b.is_ready().await {
                        let model = b
                            .list_models()
                            .await
                            .ok()
                            .and_then(|m| m.into_iter().next())
                            .map(|e| e.id);
                        ("active".to_string(), model)
                    } else {
                        ("inactive".to_string(), None)
                    }
                }
                Err(_) => ("error".to_string(), None),
            };

            backends.push(BackendInfo {
                name: "vllm".to_string(),
                status: vl_status,
                backend_type: "api".to_string(),
                host: "127.0.0.1".to_string(),
                port: vllm_port,
                loaded_model: vl_model,
                vram_usage_mb: None,
            });
        }

        Ok(backends)
    }

    /// Load model on specified backend.
    pub async fn load_model(
        backend: &str,
        _model_path: &str,
        _gpu_layers: Option<u32>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("Backend loading not implemented for: {}", backend)
    }

    /// Unload model from backend.
    pub async fn unload_model(_backend: &str) -> anyhow::Result<()> {
        anyhow::bail!("Backend unloading not implemented")
    }

    /// Switch model on backend (hot-reload).
    pub async fn switch_model(
        backend: &str,
        _model_path: &str,
        _gpu_layers: Option<u32>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("Backend switching not implemented for: {}", backend)
    }

    /// Check if a named backend is alive.
    pub async fn health_check(backend: &str) -> bool {
        match backend {
            "llamacpp" => LlamaCppBackend::with_defaults()
                .map(|b| futures::executor::block_on(b.is_ready()))
                .unwrap_or(false),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_list_backends_active_llamacpp() {
        // Returns at least the llamacpp entry (status may be inactive if server isn't running)
        let backends = BackendDriver::list_backends().await.unwrap();
        assert!(!backends.is_empty());
        assert_eq!(backends[0].name, "llamacpp");
    }
}
