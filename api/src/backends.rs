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

    /// Load (or verify) a model on the specified backend.
    ///
    /// For llama.cpp: validates the model file exists, checks what's currently
    /// loaded via /props, and returns success if it matches. If a different model
    /// is loaded, triggers a systemd restart of llama-server with the new path.
    /// If systemd is unavailable, returns an informative error.
    pub async fn load_model(
        backend: &str,
        model_path: &str,
        gpu_layers: Option<u32>,
    ) -> anyhow::Result<()> {
        // Step 1: model file must exist and be readable
        if !std::path::Path::new(model_path).exists() {
            anyhow::bail!("Model file not found: {}", model_path);
        }

        match backend {
            "llamacpp" => {
                let client = LlamaCppBackend::with_defaults()?;

                // Step 2: if the right model is already loaded → nothing to do
                if client.is_ready().await && client.current_model_matches(model_path).await {
                    tracing::info!(model = model_path, "model already loaded on llamacpp");
                    return Ok(());
                }

                // Step 3: trigger systemd restart with model path + gpu layers via env override
                let gpu_arg = gpu_layers
                    .map(|n| format!(" --n-gpu-layers {n}"))
                    .unwrap_or_default();
                tracing::info!(
                    model = model_path,
                    gpu_arg = gpu_arg.trim(),
                    "restarting llama-server with new model"
                );

                // Write a drop-in override that sets the model env var
                let override_dir = "/run/systemd/system/llama-server.service.d";
                let override_content = format!(
                    "[Service]\nEnvironment=LLAMA_MODEL={}{}\n",
                    model_path, gpu_arg
                );

                if let Err(e) = std::fs::create_dir_all(override_dir) {
                    anyhow::bail!(
                        "Cannot write systemd override (are you root?): {e}\n\
                         Restart llama-server manually with --model {model_path}"
                    );
                }
                std::fs::write(
                    format!("{override_dir}/model-override.conf"),
                    override_content,
                )?;

                let status = tokio::process::Command::new("systemctl")
                    .args(["daemon-reload"])
                    .status()
                    .await?;
                if !status.success() {
                    anyhow::bail!("systemctl daemon-reload failed");
                }

                let status = tokio::process::Command::new("systemctl")
                    .args(["restart", "llama-server.service"])
                    .status()
                    .await?;
                if !status.success() {
                    anyhow::bail!("systemctl restart llama-server.service failed");
                }

                Ok(())
            }
            other => anyhow::bail!("load_model not supported for backend: {}", other),
        }
    }

    /// Unload model from backend (graceful stop of llama-server).
    pub async fn unload_model(backend: &str) -> anyhow::Result<()> {
        match backend {
            "llamacpp" => {
                let status = tokio::process::Command::new("systemctl")
                    .args(["stop", "llama-server.service"])
                    .status()
                    .await?;
                if !status.success() {
                    anyhow::bail!("systemctl stop llama-server.service failed");
                }
                Ok(())
            }
            other => anyhow::bail!("unload_model not supported for backend: {}", other),
        }
    }

    /// Switch model: stop current, start with new model.
    pub async fn switch_model(
        backend: &str,
        model_path: &str,
        gpu_layers: Option<u32>,
    ) -> anyhow::Result<()> {
        Self::load_model(backend, model_path, gpu_layers).await
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
