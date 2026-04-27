/// Backend Router
///
/// VRAM-aware, latency-aware routing across all configured backends.
/// Probes all backends concurrently (500ms timeout), caches health for 2s,
/// and scores each live backend to pick the best one per request.
///
/// Score formula per backend:
///   base 1.0
///   + 0.5 × (idle_slots / total_slots)   — load pressure (llama.cpp / vLLM)
///   − 0.01 × priority                    — tiebreak (lower priority = preferred)
///
/// Circuit breaker (per backend):
///   After CIRCUIT_BREAKER_THRESHOLD consecutive request failures, the backend
///   is excluded from picks for CIRCUIT_BREAKER_COOLDOWN_SECS seconds.
///   A successful health probe during or after the cooldown resets the counter.
///
/// Configuration via environment:
///   LLAMACPP_URL                  (default: http://127.0.0.1:8080)
///   VLLM_URL                      (optional — backend omitted if not set)
///   CIRCUIT_BREAKER_THRESHOLD     (default: 3)
///   CIRCUIT_BREAKER_COOLDOWN_SECS (default: 30)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use metrics::{counter, gauge};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::metrics as m;

// ── Backend kinds ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BackendKind {
    LlamaCpp,
    Vllm,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::LlamaCpp => write!(f, "llamacpp"),
            BackendKind::Vllm => write!(f, "vllm"),
        }
    }
}

// ── Static backend configuration ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub id: String,
    pub kind: BackendKind,
    pub base_url: String,
    /// Lower value = higher priority when scores tie.
    pub priority: u8,
}

// ── Health snapshot (cached per TTL) ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct HealthSnapshot {
    alive: bool,
    slots_free: Option<u32>,
    slots_total: Option<u32>,
    checked_at: Instant,
    // Circuit breaker
    consecutive_failures: u32,
    open_until: Option<Instant>,
}

impl HealthSnapshot {
    fn dead() -> Self {
        Self {
            alive: false,
            slots_free: None,
            slots_total: None,
            checked_at: Instant::now(),
            consecutive_failures: 0,
            open_until: None,
        }
    }
}

// ── Selected backend returned to callers ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SelectedBackend {
    pub config: BackendConfig,
    pub score: f64,
}

impl SelectedBackend {
    pub fn kind(&self) -> &BackendKind {
        &self.config.kind
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct BackendRouter {
    backends: Vec<BackendConfig>,
    cache: Arc<RwLock<HashMap<String, HealthSnapshot>>>,
    probe_ttl: Duration,
    probe_timeout: Duration,
    failure_threshold: u32,
    cooldown: Duration,
}

impl BackendRouter {
    pub fn new(backends: Vec<BackendConfig>) -> Self {
        Self {
            backends,
            cache: Arc::new(RwLock::new(HashMap::new())),
            probe_ttl: Duration::from_secs(2),
            probe_timeout: Duration::from_millis(500),
            failure_threshold: 3,
            cooldown: Duration::from_secs(30),
        }
    }

    /// Build from environment variables.
    pub fn from_env() -> Self {
        let failure_threshold = std::env::var("CIRCUIT_BREAKER_THRESHOLD")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let cooldown = Duration::from_secs(
            std::env::var("CIRCUIT_BREAKER_COOLDOWN_SECS")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(30),
        );

        let mut backends = vec![];
        backends.push(BackendConfig {
            id: "llamacpp-0".to_string(),
            kind: BackendKind::LlamaCpp,
            base_url: std::env::var("LLAMACPP_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string()),
            priority: 0,
        });

        if let Ok(url) = std::env::var("VLLM_URL") {
            backends.push(BackendConfig {
                id: "vllm-0".to_string(),
                kind: BackendKind::Vllm,
                base_url: url,
                priority: 1,
            });
        }

        let mut router = Self::new(backends);
        router.failure_threshold = failure_threshold;
        router.cooldown = cooldown;
        router
    }

    /// Record a failed inference request for a backend.
    /// Opens the circuit after `failure_threshold` consecutive failures.
    pub async fn report_failure(&self, backend_id: &str) {
        let mut cache = self.cache.write().await;
        let entry = cache
            .entry(backend_id.to_string())
            .or_insert_with(HealthSnapshot::dead);
        entry.consecutive_failures += 1;

        if entry.consecutive_failures >= self.failure_threshold && entry.open_until.is_none() {
            let until = Instant::now() + self.cooldown;
            entry.open_until = Some(until);
            entry.alive = false;
            warn!(
                backend = %backend_id,
                failures = entry.consecutive_failures,
                cooldown_secs = self.cooldown.as_secs(),
                "circuit breaker opened"
            );
            gauge!(m::CIRCUIT_OPEN, "backend" => backend_id.to_string()).set(1.0);
        }
    }

    /// Pick the best available backend for the next request.
    /// Returns `None` only if ALL backends are unreachable or circuit-open.
    pub async fn pick(&self) -> Option<SelectedBackend> {
        self.refresh_stale().await;

        let now = Instant::now();
        let cache = self.cache.read().await;

        self.backends
            .iter()
            .filter_map(|b| {
                let h = cache.get(&b.id)?;
                // Circuit open and cooldown not yet elapsed → skip
                if let Some(open_until) = h.open_until {
                    if open_until > now {
                        return None;
                    }
                    // Past cooldown → half-open: let one request through
                }
                if !h.alive {
                    return None;
                }
                let score = Self::score(b, h);
                Some((score, b))
            })
            .max_by(|(sa, _), (sb, _)| {
                sa.partial_cmp(sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(score, b)| {
                counter!(m::ROUTER_PICKS, "backend" => b.id.clone()).increment(1);
                SelectedBackend {
                    config: b.clone(),
                    score,
                }
            })
    }

    /// List all backends with their current liveness (for /backends endpoint).
    pub async fn list(&self) -> Vec<(BackendConfig, bool, f64)> {
        self.refresh_stale().await;
        let cache = self.cache.read().await;
        self.backends
            .iter()
            .map(|b| {
                let h = cache.get(&b.id);
                let alive = h.map(|h| h.alive).unwrap_or(false);
                let score = h.filter(|h| h.alive).map(|h| Self::score(b, h)).unwrap_or(0.0);
                (b.clone(), alive, score)
            })
            .collect()
    }

    // ── Scoring ───────────────────────────────────────────────────────────────

    fn score(config: &BackendConfig, health: &HealthSnapshot) -> f64 {
        let mut score = 1.0_f64;

        // Idle slot ratio — more free slots = less pressure = better
        if let (Some(free), Some(total)) = (health.slots_free, health.slots_total) {
            if total > 0 {
                score += 0.5 * (free as f64 / total as f64);
            }
        }

        // Priority tiebreak: lower priority value = tiny bonus
        score -= config.priority as f64 * 0.01;

        score
    }

    // ── Probing ───────────────────────────────────────────────────────────────

    async fn refresh_stale(&self) {
        let stale: Vec<BackendConfig> = {
            let cache = self.cache.read().await;
            self.backends
                .iter()
                .filter(|b| {
                    cache
                        .get(&b.id)
                        .map(|h| h.checked_at.elapsed() >= self.probe_ttl)
                        .unwrap_or(true)
                })
                .cloned()
                .collect()
        };

        if stale.is_empty() {
            return;
        }

        let timeout = self.probe_timeout;
        let results = join_all(stale.iter().map(|b| {
            let b = b.clone();
            async move {
                let snapshot = tokio::time::timeout(timeout, probe_backend(&b))
                    .await
                    .unwrap_or_else(|_| {
                        warn!(backend = %b.id, "probe timed out");
                        HealthSnapshot::dead()
                    });
                (b.id.clone(), snapshot)
            }
        }))
        .await;

        let mut cache = self.cache.write().await;
        for (id, mut snapshot) in results {
            let prev = cache.get(&id);

            // Probe success resets circuit breaker
            if snapshot.alive {
                let was_open = prev.map(|p| p.open_until.is_some()).unwrap_or(false);
                if was_open {
                    info!(backend = %id, "circuit breaker closed — probe recovered");
                    gauge!(m::CIRCUIT_OPEN, "backend" => id.clone()).set(0.0);
                }
                snapshot.consecutive_failures = 0;
                snapshot.open_until = None;
            } else {
                // Preserve circuit breaker state across probes
                if let Some(prev) = prev {
                    snapshot.consecutive_failures = prev.consecutive_failures;
                    snapshot.open_until = prev.open_until;
                }
            }

            debug!(
                backend = %id,
                alive = snapshot.alive,
                slots_free = ?snapshot.slots_free,
                circuit_open = snapshot.open_until.is_some(),
                "probe result"
            );
            gauge!(m::BACKEND_ALIVE, "backend" => id.clone())
                .set(if snapshot.alive { 1.0 } else { 0.0 });
            cache.insert(id, snapshot);
        }
    }
}

// ── Per-backend probes ────────────────────────────────────────────────────────

async fn probe_backend(config: &BackendConfig) -> HealthSnapshot {
    match config.kind {
        BackendKind::LlamaCpp => probe_llamacpp(config).await,
        BackendKind::Vllm => probe_vllm(config).await,
    }
}

async fn probe_llamacpp(config: &BackendConfig) -> HealthSnapshot {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(400))
        .build()
        .unwrap_or_default();

    match client
        .get(format!("{}/health", config.base_url))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => {
            // Extract slot info for load-pressure scoring
            let (slots_free, slots_total) = r
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|v| {
                    let idle = v["slots_idle"].as_u64()? as u32;
                    let processing = v["slots_processing"].as_u64().unwrap_or(0) as u32;
                    Some((idle, idle + processing))
                })
                .map(|(f, t)| (Some(f), Some(t)))
                .unwrap_or((None, None));

            HealthSnapshot {
                alive: true,
                slots_free,
                slots_total,
                checked_at: Instant::now(),
                consecutive_failures: 0,
                open_until: None,
            }
        }
        Ok(r) => {
            warn!(backend = %config.id, status = %r.status(), "llamacpp unhealthy");
            HealthSnapshot::dead()
        }
        Err(e) => {
            warn!(backend = %config.id, err = %e, "llamacpp unreachable");
            HealthSnapshot::dead()
        }
    }
}

async fn probe_vllm(config: &BackendConfig) -> HealthSnapshot {
    use crate::backends::vllm::{VllmBackend, VllmConfig};

    let vllm = match VllmBackend::new(VllmConfig {
        base_url: config.base_url.clone(),
        timeout_secs: 1,
    }) {
        Ok(b) => b,
        Err(_) => return HealthSnapshot::dead(),
    };

    if !vllm.is_ready().await {
        warn!(backend = %config.id, "vllm unhealthy");
        return HealthSnapshot::dead();
    }

    // Pull slot capacity from Prometheus metrics (best-effort)
    let (slots_free, slots_total) = vllm
        .get_capacity()
        .await
        .map(|(f, t)| (Some(f), Some(t)))
        .unwrap_or((None, None));

    HealthSnapshot {
        alive: true,
        slots_free,
        slots_total,
        checked_at: Instant::now(),
        consecutive_failures: 0,
        open_until: None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_router(backends: Vec<(&str, BackendKind, &str, u8)>) -> BackendRouter {
        BackendRouter::new(
            backends
                .into_iter()
                .map(|(id, kind, url, prio)| BackendConfig {
                    id: id.to_string(),
                    kind,
                    base_url: url.to_string(),
                    priority: prio,
                })
                .collect(),
        )
    }

    #[test]
    fn score_prefers_more_free_slots() {
        let config = BackendConfig {
            id: "a".to_string(),
            kind: BackendKind::LlamaCpp,
            base_url: String::new(),
            priority: 0,
        };

        let full = HealthSnapshot {
            alive: true, slots_free: Some(0), slots_total: Some(4),
            checked_at: Instant::now(), consecutive_failures: 0, open_until: None,
        };
        let half = HealthSnapshot {
            alive: true, slots_free: Some(2), slots_total: Some(4),
            checked_at: Instant::now(), consecutive_failures: 0, open_until: None,
        };
        let empty = HealthSnapshot {
            alive: true, slots_free: Some(4), slots_total: Some(4),
            checked_at: Instant::now(), consecutive_failures: 0, open_until: None,
        };

        let s_full = BackendRouter::score(&config, &full);
        let s_half = BackendRouter::score(&config, &half);
        let s_empty = BackendRouter::score(&config, &empty);

        assert!(s_full < s_half, "full slots should score less than half");
        assert!(s_half < s_empty, "half slots should score less than empty");
    }

    #[test]
    fn score_priority_tiebreak() {
        let hi = BackendConfig {
            id: "hi".to_string(),
            kind: BackendKind::LlamaCpp,
            base_url: String::new(),
            priority: 0,
        };
        let lo = BackendConfig {
            id: "lo".to_string(),
            kind: BackendKind::LlamaCpp,
            base_url: String::new(),
            priority: 5,
        };
        let h = HealthSnapshot {
            alive: true,
            slots_free: None,
            slots_total: None,
            checked_at: Instant::now(),
            consecutive_failures: 0,
            open_until: None,
        };

        assert!(
            BackendRouter::score(&hi, &h) > BackendRouter::score(&lo, &h),
            "priority 0 should beat priority 5"
        );
    }

    #[tokio::test]
    async fn pick_returns_none_when_all_down() {
        let router = make_router(vec![
            ("a", BackendKind::LlamaCpp, "http://127.0.0.1:1", 0),
            ("b", BackendKind::Vllm, "http://127.0.0.1:2", 1),
        ]);
        // Seed cache with dead snapshots so we skip real probes
        {
            let mut cache = router.cache.write().await;
            cache.insert("a".to_string(), HealthSnapshot::dead());
            cache.insert("b".to_string(), HealthSnapshot::dead());
        }
        assert!(router.pick().await.is_none());
    }

    #[tokio::test]
    async fn pick_selects_highest_score() {
        let router = make_router(vec![
            ("a", BackendKind::LlamaCpp, "http://127.0.0.1:1", 0),
            ("b", BackendKind::LlamaCpp, "http://127.0.0.1:2", 1),
        ]);
        {
            let mut cache = router.cache.write().await;
            // 'a' has all slots free → higher score
            cache.insert(
                "a".to_string(),
                HealthSnapshot {
                    alive: true,
                    slots_free: Some(4),
                    slots_total: Some(4),
                    checked_at: Instant::now(),
                    consecutive_failures: 0,
                    open_until: None,
                },
            );
            // 'b' has no slots free
            cache.insert(
                "b".to_string(),
                HealthSnapshot {
                    alive: true,
                    slots_free: Some(0),
                    slots_total: Some(4),
                    checked_at: Instant::now(),
                    consecutive_failures: 0,
                    open_until: None,
                },
            );
        }
        let selected = router.pick().await.expect("should pick 'a'");
        assert_eq!(selected.config.id, "a");
    }
}
