use metrics::{describe_counter, describe_gauge, describe_histogram, Unit};

pub const QUEUE_DEPTH: &str = "ml_orchestrator_queue_depth";
pub const QUEUE_WAIT_SECS: &str = "ml_orchestrator_queue_wait_seconds";
pub const REQUEST_LATENCY_SECS: &str = "ml_orchestrator_request_latency_seconds";
pub const IN_FLIGHT: &str = "ml_orchestrator_in_flight";
pub const REQUESTS_FAILED: &str = "ml_orchestrator_requests_failed_total";
pub const BACKEND_ALIVE: &str = "ml_backend_alive";
pub const CIRCUIT_OPEN: &str = "ml_circuit_breaker_open";
pub const ROUTER_PICKS: &str = "ml_router_picks_total";
pub const STREAMING_REQUESTS: &str = "ml_streaming_requests_total";

pub fn describe_all() {
    describe_gauge!(QUEUE_DEPTH, Unit::Count, "Pending requests per priority level");
    describe_histogram!(QUEUE_WAIT_SECS, Unit::Seconds, "Time a request spent waiting in queue");
    describe_histogram!(REQUEST_LATENCY_SECS, Unit::Seconds, "End-to-end inference latency");
    describe_gauge!(IN_FLIGHT, Unit::Count, "Requests currently being processed by workers");
    describe_counter!(REQUESTS_FAILED, Unit::Count, "Total failed inference requests");
    describe_gauge!(BACKEND_ALIVE, Unit::Count, "1 if backend is alive, 0 otherwise");
    describe_gauge!(CIRCUIT_OPEN, Unit::Count, "1 if circuit breaker is open for this backend");
    describe_counter!(ROUTER_PICKS, Unit::Count, "Total backend selections by router");
    describe_counter!(STREAMING_REQUESTS, Unit::Count, "Total streaming inference requests");
}
