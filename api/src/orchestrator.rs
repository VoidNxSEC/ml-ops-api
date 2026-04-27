/// Inference Orchestrator — Phase 3
///
/// Priority queue + worker pool for non-streaming requests.
/// Streaming requests bypass the queue (TTFT must be minimal).
///
/// Architecture:
///   Handler → submit(json, priority) → SharedQueue
///                                           │
///                               N worker tasks (tokio::spawn)
///                                           │ (Semaphore: max_concurrent)
///                                       router.pick()
///                                           │
///                                    backend.proxy()
///                                           │
///                                    oneshot reply
///
/// Priority levels (served highest first):
///   Critical = 3  |  High = 2  |  Normal = 1  |  Low = 0
///
/// Configuration via env:
///   ORCHESTRATOR_WORKERS         (default: 4)
///   ORCHESTRATOR_MAX_CONCURRENT  (default: 8)
///   ORCHESTRATOR_TIMEOUT_SECS   (default: 300)

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::{oneshot, Notify, Semaphore};
use tracing::{debug, info, warn};

use metrics::{counter, gauge, histogram};

use crate::backends::llamacpp::{LlamaCppBackend, LlamaCppConfig};
use crate::metrics as m;
use crate::router::BackendRouter;

// ── Priority ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Priority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl std::str::FromStr for Priority {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "critical" => Priority::Critical,
            "high" => Priority::High,
            "low" => Priority::Low,
            _ => Priority::Normal,
        })
    }
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Priority::Critical => "critical",
            Priority::High => "high",
            Priority::Normal => "normal",
            Priority::Low => "low",
        };
        write!(f, "{s}")
    }
}

// ── Pending request ───────────────────────────────────────────────────────────

pub struct PendingRequest {
    pub request_json: Value,
    pub reply: oneshot::Sender<Result<Value, String>>,
    pub enqueued_at: Instant,
}

// ── Shared priority queue ─────────────────────────────────────────────────────

/// Four-level priority queue with Tokio Notify for zero-lost-wakeup semantics.
///
/// Uses std::sync::Mutex (not tokio) because the critical section never awaits.
/// The Notify stored-permit guarantee means notify_one() before notified().await
/// is safe: the next await returns immediately.
struct SharedQueue {
    queues: Mutex<[VecDeque<PendingRequest>; 4]>,
    notify: Notify,
}

impl SharedQueue {
    fn new() -> Self {
        Self {
            queues: Mutex::new([
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
            ]),
            notify: Notify::new(),
        }
    }

    fn push(&self, req: PendingRequest, priority: Priority) {
        let depth = {
            let mut qs = self.queues.lock().unwrap();
            qs[priority as usize].push_back(req);
            qs[priority as usize].len()
        };
        let prio_label = match priority {
            Priority::Low => "low",
            Priority::Normal => "normal",
            Priority::High => "high",
            Priority::Critical => "critical",
        };
        gauge!(m::QUEUE_DEPTH, "priority" => prio_label).set(depth as f64);
        self.notify.notify_one();
    }

    async fn pop(&self) -> PendingRequest {
        loop {
            // Release the lock BEFORE awaiting — never hold std::Mutex across await.
            let maybe = {
                let mut qs = self.queues.lock().unwrap();
                (0..4usize)
                    .rev() // Critical(3) first
                    .find_map(|p| qs[p].pop_front())
            };

            if let Some(req) = maybe {
                return req;
            }

            // Queue empty. Tokio's Notify stored-permit ensures we won't miss
            // a notify_one() that fired between releasing the lock and this await.
            self.notify.notified().await;
        }
    }

    fn depths(&self) -> [usize; 4] {
        let qs = self.queues.lock().unwrap();
        [qs[0].len(), qs[1].len(), qs[2].len(), qs[3].len()]
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub workers: usize,
    pub max_concurrent: usize,
    pub request_timeout: Duration,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            workers: 4,
            max_concurrent: 8,
            request_timeout: Duration::from_secs(300),
        }
    }
}

impl OrchestratorConfig {
    pub fn from_env() -> Self {
        Self {
            workers: std::env::var("ORCHESTRATOR_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            max_concurrent: std::env::var("ORCHESTRATOR_MAX_CONCURRENT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            request_timeout: Duration::from_secs(
                std::env::var("ORCHESTRATOR_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(300),
            ),
        }
    }
}

// ── OrchestratorHandle ────────────────────────────────────────────────────────

/// Cheap-to-clone handle for submitting non-streaming requests.
#[derive(Clone)]
pub struct OrchestratorHandle {
    queue: Arc<SharedQueue>,
    semaphore: Arc<Semaphore>,
    config: OrchestratorConfig,
}

impl OrchestratorHandle {
    /// Submit a non-streaming inference request.
    ///
    /// Blocks the caller until a worker completes it (or timeout/error).
    /// Returns raw JSON — the caller is responsible for parsing.
    pub async fn submit(&self, request_json: Value, priority: Priority) -> Result<Value, String> {
        let (tx, rx) = oneshot::channel();

        self.queue.push(
            PendingRequest {
                request_json,
                reply: tx,
                enqueued_at: Instant::now(),
            },
            priority,
        );

        rx.await
            .map_err(|_| "orchestrator worker dropped without reply".to_string())?
    }

    /// Queue depths per priority level [low, normal, high, critical].
    pub fn queue_depths(&self) -> [usize; 4] {
        self.queue.depths()
    }

    /// Number of in-flight requests right now.
    pub fn in_flight(&self) -> usize {
        self.config.max_concurrent - self.semaphore.available_permits()
    }
}

// ── Spawn ─────────────────────────────────────────────────────────────────────

/// Spawn N worker tasks and return a handle for submission.
pub fn spawn(router: Arc<BackendRouter>, config: OrchestratorConfig) -> OrchestratorHandle {
    let queue = Arc::new(SharedQueue::new());
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));

    let handle = OrchestratorHandle {
        queue: queue.clone(),
        semaphore: semaphore.clone(),
        config: config.clone(),
    };

    for worker_id in 0..config.workers {
        let q = queue.clone();
        let sem = semaphore.clone();
        let r = router.clone();
        let timeout = config.request_timeout;

        tokio::spawn(async move {
            info!(worker = worker_id, "orchestrator worker started");

            loop {
                let req = q.pop().await;
                let wait_secs = req.enqueued_at.elapsed().as_secs_f64();

                histogram!(m::QUEUE_WAIT_SECS).record(wait_secs);

                // Update queue depth gauges after dequeue
                {
                    let depths = q.depths();
                    for (i, &depth) in depths.iter().enumerate() {
                        let prio = match i { 0 => "low", 1 => "normal", 2 => "high", _ => "critical" };
                        gauge!(m::QUEUE_DEPTH, "priority" => prio).set(depth as f64);
                    }
                }

                // Limit in-flight requests across all workers
                let permit = match sem.acquire().await {
                    Ok(p) => p,
                    Err(_) => {
                        let _ = req.reply.send(Err("orchestrator shutting down".to_string()));
                        break;
                    }
                };

                gauge!(m::IN_FLIGHT).increment(1.0);
                debug!(worker = worker_id, wait_secs, "dispatching");

                let start = std::time::Instant::now();
                let result =
                    tokio::time::timeout(timeout, execute(&r, req.request_json)).await;

                let latency = start.elapsed().as_secs_f64();
                gauge!(m::IN_FLIGHT).decrement(1.0);
                drop(permit); // Release before sending reply — free slot ASAP

                let outcome = match result {
                    Ok(r) => r,
                    Err(_) => Err(format!("timed out after {}s", timeout.as_secs())),
                };

                histogram!(m::REQUEST_LATENCY_SECS).record(latency);

                if let Err(ref e) = outcome {
                    warn!(worker = worker_id, err = %e, "request failed");
                    counter!(m::REQUESTS_FAILED).increment(1);
                }

                // rx may be dropped if caller cancelled — that's fine
                let _ = req.reply.send(outcome);
            }
        });
    }

    info!(
        workers = config.workers,
        max_concurrent = config.max_concurrent,
        timeout_secs = config.request_timeout.as_secs(),
        "orchestrator spawned"
    );

    handle
}

// ── Execution ─────────────────────────────────────────────────────────────────

async fn execute(router: &BackendRouter, request_json: Value) -> Result<Value, String> {
    let selected = router
        .pick()
        .await
        .ok_or_else(|| "no backends available".to_string())?;

    let backend_id = selected.config.id.clone();

    let backend = LlamaCppBackend::new(LlamaCppConfig {
        base_url: selected.base_url().to_string(),
        timeout_secs: 300,
    })
    .map_err(|e| e.to_string())?;

    let model = request_json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let start = std::time::Instant::now();

    debug!(backend = %backend_id, score = selected.score, model = %model, "executing");

    let result = async {
        let resp = backend
            .proxy_chat_completion(request_json)
            .await
            .map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            return Err(format!("backend {} returned {}", backend_id, resp.status()));
        }

        let body = resp.bytes().await.map_err(|e| e.to_string())?;
        serde_json::from_slice::<Value>(&body).map_err(|e| format!("parse error: {e}"))
    }
    .await;

    let latency_ms = start.elapsed().as_millis();

    match &result {
        Ok(v) => {
            let usage = &v["usage"];
            info!(
                backend = %backend_id,
                model = %model,
                latency_ms,
                prompt_tokens     = usage["prompt_tokens"].as_u64().unwrap_or(0),
                completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0),
                total_tokens      = usage["total_tokens"].as_u64().unwrap_or(0),
                "inference ok"
            );
        }
        Err(e) => {
            warn!(backend = %backend_id, model = %model, latency_ms, err = %e, "inference failed");
            router.report_failure(&backend_id).await;
        }
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pending(priority: Priority) -> (PendingRequest, oneshot::Receiver<Result<Value, String>>) {
        let (tx, rx) = oneshot::channel();
        let req = PendingRequest {
            request_json: serde_json::json!({"priority": priority.to_string()}),
            reply: tx,
            enqueued_at: Instant::now(),
        };
        (req, rx)
    }

    #[test]
    fn priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn priority_from_str() {
        assert_eq!("critical".parse::<Priority>().unwrap(), Priority::Critical);
        assert_eq!("high".parse::<Priority>().unwrap(), Priority::High);
        assert_eq!("low".parse::<Priority>().unwrap(), Priority::Low);
        assert_eq!("unknown".parse::<Priority>().unwrap(), Priority::Normal);
    }

    #[tokio::test]
    async fn queue_push_pop_priority_order() {
        let q = SharedQueue::new();

        let (low, _rx_low) = make_pending(Priority::Low);
        let (normal, _rx_normal) = make_pending(Priority::Normal);
        let (high, _rx_high) = make_pending(Priority::High);
        let (critical, _rx_critical) = make_pending(Priority::Critical);

        q.push(low, Priority::Low);
        q.push(normal, Priority::Normal);
        q.push(high, Priority::High);
        q.push(critical, Priority::Critical);

        // Should pop highest priority first
        let first = q.pop().await;
        let first_prio = first.request_json["priority"].as_str().unwrap().to_string();
        assert_eq!(first_prio, "critical");

        let second = q.pop().await;
        let second_prio = second.request_json["priority"].as_str().unwrap().to_string();
        assert_eq!(second_prio, "high");
    }

    #[tokio::test]
    async fn queue_depths_reflect_contents() {
        let q = SharedQueue::new();

        let (r1, _) = make_pending(Priority::Normal);
        let (r2, _) = make_pending(Priority::Normal);
        let (r3, _) = make_pending(Priority::High);

        q.push(r1, Priority::Normal);
        q.push(r2, Priority::Normal);
        q.push(r3, Priority::High);

        let depths = q.depths();
        assert_eq!(depths[Priority::Normal as usize], 2);
        assert_eq!(depths[Priority::High as usize], 1);
        assert_eq!(depths[Priority::Low as usize], 0);
        assert_eq!(depths[Priority::Critical as usize], 0);
    }

    #[tokio::test]
    async fn handle_in_flight_counter() {
        let router = Arc::new(crate::router::BackendRouter::from_env());
        let config = OrchestratorConfig {
            workers: 1,
            max_concurrent: 4,
            request_timeout: Duration::from_secs(5),
        };
        let handle = spawn(router, config);

        // No requests in flight at start
        assert_eq!(handle.in_flight(), 0);
        assert_eq!(handle.queue_depths(), [0, 0, 0, 0]);
    }
}
