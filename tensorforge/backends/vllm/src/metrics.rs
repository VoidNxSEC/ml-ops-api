//! vLLM backend metrics collection
//!
//! Provides Prometheus metrics for monitoring vLLM backend performance,
//! health, and resource usage.

#[cfg(feature = "prometheus")]
use prometheus::{
    register_counter, register_histogram, register_int_counter, Counter, Histogram, IntCounter,
};
use std::sync::Arc;
use tracing::debug;

/// vLLM backend metrics collector
#[derive(Debug, Clone)]
pub struct VllmMetrics {
    /// Health check attempts
    pub health_check_attempts: MetricCounter,
    /// Health check successes
    pub health_check_success: MetricCounter,
    /// Health check failures
    pub health_check_failures: MetricCounter,
    /// Health check duration in seconds
    pub health_check_duration: MetricHistogram,
    /// Model load attempts
    pub model_load_attempts: MetricCounter,
    /// Model load successes
    pub model_load_success: MetricCounter,
    /// Model load failures
    pub model_load_failures: MetricCounter,
    /// Model load duration in seconds
    pub model_load_duration: MetricHistogram,
    /// Model unload attempts
    pub model_unload_attempts: MetricCounter,
    /// Model unload successes
    pub model_unload_success: MetricCounter,
    /// Model unload failures
    pub model_unload_failures: MetricCounter,
    /// Inference requests
    pub inference_requests: MetricCounter,
    /// Inference successes
    pub inference_success: MetricCounter,
    /// Inference failures
    pub inference_failures: MetricCounter,
    /// Inference duration in seconds
    pub inference_duration: MetricHistogram,
    /// Queue duration in seconds
    pub queue_duration: MetricHistogram,
    /// VRAM queries
    pub vram_queries: MetricCounter,
    /// VRAM query failures
    pub vram_query_failures: MetricCounter,
    /// VRAM query duration in seconds
    pub vram_query_duration: MetricHistogram,
    /// Total tokens processed
    pub total_tokens: MetricCounter,
    /// Total requests processed
    pub total_requests: MetricCounter,
    /// Active connections
    pub active_connections: MetricGauge,
    /// Active models loaded
    pub active_models: MetricGauge,
}

/// Metric counter abstraction (prometheus or no-op)
#[derive(Debug, Clone)]
pub enum MetricCounter {
    #[cfg(feature = "prometheus")]
    Prometheus(Counter),
    Noop,
}

/// Metric histogram abstraction
#[derive(Debug, Clone)]
pub enum MetricHistogram {
    #[cfg(feature = "prometheus")]
    Prometheus(Histogram),
    Noop,
}

/// Metric gauge abstraction
#[derive(Debug, Clone)]
pub enum MetricGauge {
    #[cfg(feature = "prometheus")]
    Prometheus(prometheus::Gauge),
    Noop,
}

impl VllmMetrics {
    /// Create new metrics collector
    pub fn new() -> Arc<Self> {
        debug!("Initializing vLLM metrics collector");

        #[cfg(feature = "prometheus")]
        let metrics = {
            // Register Prometheus metrics
            let health_check_attempts = register_counter!(
                "vllm_health_check_attempts_total",
                "Total number of health check attempts"
            )
            .unwrap();

            let health_check_success = register_counter!(
                "vllm_health_check_success_total",
                "Total number of successful health checks"
            )
            .unwrap();

            let health_check_failures = register_counter!(
                "vllm_health_check_failures_total",
                "Total number of failed health checks"
            )
            .unwrap();

            let health_check_duration = register_histogram!(
                "vllm_health_check_duration_seconds",
                "Health check duration in seconds",
                vec![0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
            )
            .unwrap();

            let model_load_attempts = register_counter!(
                "vllm_model_load_attempts_total",
                "Total number of model load attempts"
            )
            .unwrap();

            let model_load_success = register_counter!(
                "vllm_model_load_success_total",
                "Total number of successful model loads"
            )
            .unwrap();

            let model_load_failures = register_counter!(
                "vllm_model_load_failures_total",
                "Total number of failed model loads"
            )
            .unwrap();

            let model_load_duration = register_histogram!(
                "vllm_model_load_duration_seconds",
                "Model load duration in seconds",
                vec![1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
            )
            .unwrap();

            let model_unload_attempts = register_counter!(
                "vllm_model_unload_attempts_total",
                "Total number of model unload attempts"
            )
            .unwrap();

            let model_unload_success = register_counter!(
                "vllm_model_unload_success_total",
                "Total number of successful model unloads"
            )
            .unwrap();

            let model_unload_failures = register_counter!(
                "vllm_model_unload_failures_total",
                "Total number of failed model unloads"
            )
            .unwrap();

            let inference_requests = register_counter!(
                "vllm_inference_requests_total",
                "Total number of inference requests"
            )
            .unwrap();

            let inference_success = register_counter!(
                "vllm_inference_success_total",
                "Total number of successful inferences"
            )
            .unwrap();

            let inference_failures = register_counter!(
                "vllm_inference_failures_total",
                "Total number of failed inferences"
            )
            .unwrap();

            let inference_duration = register_histogram!(
                "vllm_inference_duration_seconds",
                "Inference duration in seconds",
                vec![0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
            )
            .unwrap();

            let queue_duration = register_histogram!(
                "vllm_queue_duration_seconds",
                "Queue wait duration in seconds",
                vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
            )
            .unwrap();

            let vram_queries = register_counter!(
                "vllm_vram_queries_total",
                "Total number of VRAM queries"
            )
            .unwrap();

            let vram_query_failures = register_counter!(
                "vllm_vram_query_failures_total",
                "Total number of failed VRAM queries"
            )
            .unwrap();

            let vram_query_duration = register_histogram!(
                "vllm_vram_query_duration_seconds",
                "VRAM query duration in seconds",
                vec![0.001, 0.005, 0.01, 0.05, 0.1]
            )
            .unwrap();

            let total_tokens = register_counter!(
                "vllm_total_tokens_total",
                "Total number of tokens processed"
            )
            .unwrap();

            let total_requests = register_counter!(
                "vllm_total_requests_total",
                "Total number of requests processed"
            )
            .unwrap();

            let active_connections = prometheus::register_gauge!(
                "vllm_active_connections",
                "Number of active connections to vLLM"
            )
            .unwrap();

            let active_models = prometheus::register_gauge!(
                "vllm_active_models",
                "Number of active models loaded"
            )
            .unwrap();

            Arc::new(Self {
                health_check_attempts: MetricCounter::Prometheus(health_check_attempts),
                health_check_success: MetricCounter::Prometheus(health_check_success),
                health_check_failures: MetricCounter::Prometheus(health_check_failures),
                health_check_duration: MetricHistogram::Prometheus(health_check_duration),
                model_load_attempts: MetricCounter::Prometheus(model_load_attempts),
                model_load_success: MetricCounter::Prometheus(model_load_success),
                model_load_failures: MetricCounter::Prometheus(model_load_failures),
                model_load_duration: MetricHistogram::Prometheus(model_load_duration),
                model_unload_attempts: MetricCounter::Prometheus(model_unload_attempts),
                model_unload_success: MetricCounter::Prometheus(model_unload_success),
                model_unload_failures: MetricCounter::Prometheus(model_unload_failures),
                inference_requests: MetricCounter::Prometheus(inference_requests),
                inference_success: MetricCounter::Prometheus(inference_success),
                inference_failures: MetricCounter::Prometheus(inference_failures),
                inference_duration: MetricHistogram::Prometheus(inference_duration),
                queue_duration: MetricHistogram::Prometheus(queue_duration),
                vram_queries: MetricCounter::Prometheus(vram_queries),
                vram_query_failures: MetricCounter::Prometheus(vram_query_failures),
                vram_query_duration: MetricHistogram::Prometheus(vram_query_duration),
                total_tokens: MetricCounter::Prometheus(total_tokens),
                total_requests: MetricCounter::Prometheus(total_requests),
                active_connections: MetricGauge::Prometheus(active_connections),
                active_models: MetricGauge::Prometheus(active_models),
            })
        };

        #[cfg(not(feature = "prometheus"))]
        let metrics = {
            Arc::new(Self {
                health_check_attempts: MetricCounter::Noop,
                health_check_success: MetricCounter::Noop,
                health_check_failures: MetricCounter::Noop,
                health_check_duration: MetricHistogram::Noop,
                model_load_attempts: MetricCounter::Noop,
                model_load_success: MetricCounter::Noop,
                model_load_failures: MetricCounter::Noop,
                model_load_duration: MetricHistogram::Noop,
                model_unload_attempts: MetricCounter::Noop,
                model_unload_success: MetricCounter::Noop,
                model_unload_failures: MetricCounter::Noop,
                inference_requests: MetricCounter::Noop,
                inference_success: MetricCounter::Noop,
                inference_failures: MetricCounter::Noop,
                inference_duration: MetricHistogram::Noop,
                queue_duration: MetricHistogram::Noop,
                vram_queries: MetricCounter::Noop,
                vram_query_failures: MetricCounter::Noop,
                vram_query_duration: MetricHistogram::Noop,
                total_tokens: MetricCounter::Noop,
                total_requests: MetricCounter::Noop,
                active_connections: MetricGauge::Noop,
                active_models: MetricGauge::Noop,
            })
        };

        debug!("vLLM metrics collector initialized");
        metrics
    }

    /// Increment health check attempts
    pub fn inc_health_check_attempts(&self) {
        match &self.health_check_attempts {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment health check successes
    pub fn inc_health_check_success(&self) {
        match &self.health_check_success {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment health check failures
    pub fn inc_health_check_failures(&self) {
        match &self.health_check_failures {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Observe health check duration
    pub fn observe_health_check_duration(&self, duration_secs: f64) {
        match &self.health_check_duration {
            #[cfg(feature = "prometheus")]
            MetricHistogram::Prometheus(histogram) => histogram.observe(duration_secs),
            MetricHistogram::Noop => {}
        }
    }

    /// Increment model load attempts
    pub fn inc_model_load_attempts(&self) {
        match &self.model_load_attempts {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment model load successes
    pub fn inc_model_load_success(&self) {
        match &self.model_load_success {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment model load failures
    pub fn inc_model_load_failures(&self) {
        match &self.model_load_failures {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Observe model load duration
    pub fn observe_model_load_duration(&self, duration_secs: f64) {
        match &self.model_load_duration {
            #[cfg(feature = "prometheus")]
            MetricHistogram::Prometheus(histogram) => histogram.observe(duration_secs),
            MetricHistogram::Noop => {}
        }
    }

    /// Increment model unload attempts
    pub fn inc_model_unload_attempts(&self) {
        match &self.model_unload_attempts {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment model unload successes
    pub fn inc_model_unload_success(&self) {
        match &self.model_unload_success {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment model unload failures
    pub fn inc_model_unload_failures(&self) {
        match &self.model_unload_failures {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment inference requests
    pub fn inc_inference_requests(&self) {
        match &self.inference_requests {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment inference successes
    pub fn inc_inference_success(&self) {
        match &self.inference_success {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment inference failures
    pub fn inc_inference_failures(&self) {
        match &self.inference_failures {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Observe inference duration
    pub fn observe_inference_duration(&self, duration_secs: f64) {
        match &self.inference_duration {
            #[cfg(feature = "prometheus")]
            MetricHistogram::Prometheus(histogram) => histogram.observe(duration_secs),
            MetricHistogram::Noop => {}
        }
    }

    /// Observe queue duration
    pub fn observe_queue_duration(&self, duration_secs: f64) {
        match &self.queue_duration {
            #[cfg(feature = "prometheus")]
            MetricHistogram::Prometheus(histogram) => histogram.observe(duration_secs),
            MetricHistogram::Noop => {}
        }
    }

    /// Increment VRAM queries
    pub fn inc_vram_queries(&self) {
        match &self.vram_queries {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Increment VRAM query failures
    pub fn inc_vram_query_failures(&self) {
        match &self.vram_query_failures {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc(),
            MetricCounter::Noop => {}
        }
    }

    /// Observe VRAM query duration
    pub fn observe_vram_query_duration(&self, duration_secs: f64) {
        match &self.vram_query_duration {
            #[cfg(feature = "prometheus")]
            MetricHistogram::Prometheus(histogram) => histogram.observe(duration_secs),
            MetricHistogram::Noop => {}
        }
    }

    /// Add token count
    pub fn add_tokens(&self, count: u64) {
        match &self.total_tokens {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc_by(count as f64),
            MetricCounter::Noop => {}
        }
    }

    /// Add request count
    pub fn add_requests(&self, count: u64) {
        match &self.total_requests {
            #[cfg(feature = "prometheus")]
            MetricCounter::Prometheus(counter) => counter.inc_by(count as f64),
            MetricCounter::Noop => {}
        }
    }

    /// Set active connections count
    pub fn set_active_connections(&self, count: i64) {
        match &self.active_connections {
            #[cfg(feature = "prometheus")]
            MetricGauge::Prometheus(gauge) => gauge.set(count as f64),
            MetricGauge::Noop => {}
        }
    }

    /// Set active models count
    pub fn set_active_models(&self, count: i64) {
        match &self.active_models {
            #[cfg(feature = "prometheus")]
            MetricGauge::Prometheus(gauge) => gauge.set(count as f64),
            MetricGauge::Noop => {}
        }
    }
}

/// Default implementation for VllmMetrics
impl Default for VllmMetrics {
    fn default() -> Self {
        Self::new().as_ref().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = VllmMetrics::new();

        // Test that metrics can be created without panicking
        // (actual functionality depends on prometheus feature)

        // Test that methods don't panic
        metrics.inc_health_check_attempts();
        metrics.inc_health_check_success();
        metrics.inc_health_check_failures();
        metrics.observe_health_check_duration(0.5);

        metrics.inc_model_load_attempts();
        metrics.inc_model_load_success();
        metrics.inc_model_load_failures();
        metrics.observe_model_load_duration(5.0);

        metrics.inc_inference_requests();
        metrics.inc_inference_success();
        metrics.inc_inference_failures();
        metrics.observe_inference_duration(0.1);
        metrics.observe_queue_duration(0.01);

        metrics.inc_vram_queries();
        metrics.inc_vram_query_failures();
        metrics.observe_vram_query_duration(0.001);

        metrics.add_tokens(100);
        metrics.add_requests(10);
        metrics.set_active_connections(5);
        metrics.set_active_models(2);
    }

    #[test]
    fn test_metrics_clone() {
        let metrics1 = VllmMetrics::new();
        let metrics2 = metrics1.clone();

        // Both should be valid (test no panic)
        metrics1.inc_health_check_attempts();
        metrics2.inc_health_check_attempts();
    }
}
