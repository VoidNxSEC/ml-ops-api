//! NATS event publisher for ml-ops-api.
//!
//! Publishes `inference.request.v1` and `inference.response.v1` events.
//! Non-fatal when NATS is unavailable.

use chrono::Utc;
use serde_json::json;
use tracing::{debug, warn};
use uuid::Uuid;

pub struct NatsPublisher {
    client: Option<async_nats::Client>,
}

impl NatsPublisher {
    pub async fn connect(nats_url: &str) -> Self {
        match async_nats::connect(nats_url).await {
            Ok(client) => {
                tracing::info!("NATS publisher connected: {}", nats_url);
                Self {
                    client: Some(client),
                }
            }
            Err(e) => {
                warn!("NATS connection failed (non-fatal): {}", e);
                Self { client: None }
            }
        }
    }

    pub async fn publish_inference_request(&self, request_id: Uuid, model: &str, messages_count: usize) {
        let payload = json!({
            "event_id": Uuid::new_v4().to_string(),
            "source_service": "ml-ops-api",
            "request_id": request_id.to_string(),
            "model": model,
            "messages_count": messages_count,
            "timestamp": Utc::now().to_rfc3339(),
        });
        self.publish("inference.request.v1", &payload).await;
    }

    pub async fn publish_inference_response(
        &self,
        request_id: Uuid,
        model: &str,
        completion_tokens: u32,
        duration_ms: u64,
        status: &str,
    ) {
        let payload = json!({
            "event_id": Uuid::new_v4().to_string(),
            "source_service": "ml-ops-api",
            "request_id": request_id.to_string(),
            "model": model,
            "completion_tokens": completion_tokens,
            "duration_ms": duration_ms,
            "status": status,
            "timestamp": Utc::now().to_rfc3339(),
        });
        self.publish("inference.response.v1", &payload).await;
    }

    async fn publish(&self, subject: &str, payload: &serde_json::Value) {
        let Some(ref client) = self.client else {
            return;
        };
        let data = payload.to_string().into_bytes();
        match client.publish(subject.to_string(), data.into()).await {
            Ok(_) => debug!("Published NATS: {}", subject),
            Err(e) => warn!("NATS publish failed (non-fatal) on {}: {}", subject, e),
        }
    }
}
