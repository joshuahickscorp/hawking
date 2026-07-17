//! The runtime supervisor — the host's process-lifecycle owner (bible ch.01
//! §4.3).
//!
//! The scaffold composed every sibling crate but never *booted the runtime*: the
//! kernel's `Act` step has a clean HTTP seam to `hawking serve`, but nothing
//! spawned or supervised that process. [`RuntimeSupervisor`] closes that gap. It
//! spawns the `hawking serve` child, polls its `/healthz` endpoint, drives a
//! `Down → Booting → Ready → Degraded → Failed` state machine, restarts with a
//! backoff ladder (the [`BackoffPolicy`] already in `hide-core::supervision`),
//! and writes a `runtime.lock` file so a second host can't double-boot the
//! runtime over the same workspace.
//!
//! ## Testability
//!
//! The supervisor is **generic over how the child is launched and where health
//! is polled** via the [`RuntimeLauncher`] trait. Production wires
//! [`ProcessLauncher`] (spawns the `hawking` binary by name — HTTP-only, T5: we
//! never link the engine crates). Tests wire a fake launcher that spins up a
//! tiny in-process axum-free health server (a `tokio` `TcpListener` answering
//! `200 OK` on `/healthz`), so the full Down→Ready→Degraded transition + backoff
//! are exercised without a model.

use hide_core::ids::now_ms;
use hide_core::runtime::RuntimeSupervisorState;
use hide_core::supervision::{BackoffPolicy, ProcessSpec, ProcessStatus};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// A handle to a launched runtime child. The supervisor only needs to know its
/// pid (for status/`runtime.lock`), how to ask whether it is still alive, and
/// how to terminate it — it never speaks the engine protocol directly.
#[async_trait::async_trait]
pub trait RuntimeChild: Send + Sync {
    /// OS pid if the child is a real process (None for in-process fakes).
    fn pid(&self) -> Option<u32>;
    /// True if the child is still running (has not exited / crashed).
    async fn is_alive(&self) -> bool;
    /// Terminate the child (best-effort SIGTERM→wait). Idempotent.
    async fn terminate(&self);
}

/// How the supervisor launches a runtime child and where it polls health. Making
/// this a trait is what lets tests substitute a fake in-process health server for
/// the real `hawking serve` binary.
#[async_trait::async_trait]
pub trait RuntimeLauncher: Send + Sync {
    /// Spawn the runtime child. Returns the child handle + the health URL to
    /// poll (so a fake can bind an ephemeral port and report it back).
    async fn launch(&self, spec: &ProcessSpec) -> Result<(Box<dyn RuntimeChild>, String), String>;
    /// Poll a health URL. `Ok(true)` = healthy, `Ok(false)` = reachable but
    /// unhealthy, `Err` = unreachable.
    async fn poll_health(&self, url: &str) -> Result<bool, String>;
}

/// The supervisor's tunables.
#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    /// How the `hawking serve` process is described (argv/cwd/env/health_url).
    pub spec: ProcessSpec,
    /// Restart backoff ladder + per-window cap (the bible's default ladder).
    pub backoff: BackoffPolicy,
    /// Interval between `/healthz` polls once Ready.
    pub health_interval: Duration,
    /// How long to wait for the first healthy poll before declaring boot failed.
    pub boot_timeout: Duration,
    /// `runtime.lock` path (workspace-scoped). `None` disables the lock (tests).
    pub lock_path: Option<PathBuf>,
}

impl SupervisorConfig {
    /// Production config: spawn the `hawking` binary's `serve` subcommand and
    /// poll its `/healthz`. The caller supplies the bind host:port so the health
    /// URL and the serve `--addr` agree, plus the model `weights` the serve
    /// loads for generation (the `serve` subcommand requires `--weights`).
    ///
    /// When a `.tq` sidecar sits next to the weights (same stem, `.tq`
    /// extension), the runtime is asked to serve it: `HAWKING_QWEN_TQ=1` is set
    /// in the child's env so the engine builds the TQ side map and serves the
    /// FFN/all-linear projections from the `.tq` artifact (the `hawking serve`
    /// binary must have been built `--features tq` for this to engage; without
    /// that feature the env var is a no-op and serve falls back to Q4_K).
    pub fn for_hawking_serve(
        bind_addr: impl Into<String>,
        workspace_root: impl AsRef<Path>,
        weights: impl AsRef<Path>,
        lock_path: impl Into<PathBuf>,
    ) -> Self {
        let bind = bind_addr.into();
        let weights = weights.as_ref();
        let mut argv = vec![
            "hawking".to_string(),
            "serve".to_string(),
            "--addr".to_string(),
            bind.clone(),
            "--weights".to_string(),
            weights.display().to_string(),
        ];
        // The serve `--addr` default is 0.0.0.0:8080; we always pass the bind
        // explicitly above so health URL and serve addr agree. (argv kept as a
        // Vec so a caller can extend it before constructing the supervisor.)
        let _ = &mut argv;

        let mut env: std::collections::BTreeMap<String, String> = Default::default();
        // A `.tq` sidecar (same stem, `.tq` extension) flips on native TQ serving.
        let tq_path = weights.with_extension("tq");
        if tq_path.exists() {
            env.insert("HAWKING_QWEN_TQ".to_string(), "1".to_string());
            // Spine A: read the artifact's REAL measured compression and pass the
            // derived (estimated) effective-context multiplier to the serve process,
            // which surfaces it on GET /v1/hawking/context. Never a hardcoded number.
            if let Some(info) = crate::tq_metadata::read_tq_context(&tq_path) {
                env.insert(
                    "HAWKING_QWEN_TQ_MULTIPLIER".to_string(),
                    format!("{:.3}", info.multiplier),
                );
            }
        }

        let spec = ProcessSpec {
            name: "hawking-serve".to_string(),
            argv,
            cwd: Some(workspace_root.as_ref().display().to_string()),
            env,
            health_url: Some(format!("http://{bind}/healthz")),
        };
        Self {
            spec,
            backoff: BackoffPolicy::default(),
            health_interval: Duration::from_secs(5),
            boot_timeout: Duration::from_secs(30),
            lock_path: Some(lock_path.into()),
        }
    }
}

/// Mutable supervisor state behind a lock (the state machine + restart bookkeeping).
#[derive(Debug, Clone)]
struct Inner {
    state: RuntimeSupervisorState,
    pid: Option<u32>,
    started_at_ms: Option<u64>,
    restarts: u32,
    /// Restart timestamps in the current window (for the per-window cap).
    restart_window: Vec<u64>,
    last_error: Option<String>,
    health_url: Option<String>,
}

/// The runtime supervisor. Owns the child handle + the state machine; cheap to
/// clone-status. Drive it with [`RuntimeSupervisor::boot`] then
/// [`RuntimeSupervisor::supervise_once`] (or [`RuntimeSupervisor::tick`]).
pub struct RuntimeSupervisor {
    config: SupervisorConfig,
    launcher: Arc<dyn RuntimeLauncher>,
    inner: Mutex<Inner>,
    child: Mutex<Option<Box<dyn RuntimeChild>>>,
}

impl RuntimeSupervisor {
    pub fn new(config: SupervisorConfig, launcher: Arc<dyn RuntimeLauncher>) -> Self {
        Self {
            config,
            launcher,
            inner: Mutex::new(Inner {
                state: RuntimeSupervisorState::Down,
                pid: None,
                started_at_ms: None,
                restarts: 0,
                restart_window: Vec::new(),
                last_error: None,
                health_url: None,
            }),
            child: Mutex::new(None),
        }
    }

    /// Production constructor: a [`ProcessLauncher`] spawning `hawking serve`.
    pub fn for_hawking_serve(config: SupervisorConfig) -> Self {
        Self::new(config, Arc::new(ProcessLauncher::default()))
    }

    pub fn state(&self) -> RuntimeSupervisorState {
        self.inner.lock().state.clone()
    }

    pub fn status(&self) -> ProcessStatus {
        let inner = self.inner.lock();
        ProcessStatus {
            name: self.config.spec.name.clone(),
            pid: inner.pid,
            state: inner.state.clone(),
            started_at_ms: inner.started_at_ms,
            restarts: inner.restarts,
            last_error: inner.last_error.clone(),
        }
    }

    /// The base URL of the supervised runtime (`http://host:port`), derived from
    /// the resolved health URL. `None` until booted. The host hands this to the
    /// [`crate::model_provider::HttpModelProvider`] so the kernel generates
    /// against the live child.
    pub fn base_url(&self) -> Option<String> {
        self.inner.lock().health_url.as_ref().map(|h| {
            h.trim_end_matches("/healthz")
                .trim_end_matches('/')
                .to_string()
        })
    }

    /// Acquire the `runtime.lock` (fail-closed if another live host holds it).
    /// The lock stores the host's pid + boot time. On acquire, a pre-existing
    /// lock is inspected: if it names a pid that is **still alive**, we refuse
    /// (`Err`) rather than steal a live lock; only a stale lock — dead pid,
    /// unparseable, or no pid — is reclaimed (with a warning). No-op when
    /// `lock_path` is `None`.
    fn acquire_lock(&self) -> Result<(), String> {
        let Some(path) = &self.config.lock_path else {
            return Ok(());
        };
        if path.exists() {
            // Inspect the existing lock before touching it. Read the holder's pid
            // and probe liveness; only reclaim genuinely stale locks.
            match Self::read_lock_holder(path) {
                Some(pid) if pid_is_alive(pid) => {
                    return Err(format!(
                        "runtime.lock held by live process pid={pid} ({}); refusing to steal it",
                        path.display()
                    ));
                }
                Some(pid) => {
                    // Recorded a pid but it is no longer alive — stale, reclaim.
                    eprintln!(
                        "warning: reclaiming stale runtime.lock at {} (holder pid={pid} is gone)",
                        path.display()
                    );
                }
                None => {
                    // No parseable pid (legacy/corrupt lock): conservatively
                    // reclaim — there is no live holder we can attribute it to.
                    eprintln!(
                        "warning: reclaiming runtime.lock at {} (no readable holder pid)",
                        path.display()
                    );
                }
            }
            let _ = std::fs::remove_file(path);
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(
            path,
            serde_json::json!({
                "name": self.config.spec.name,
                "pid": std::process::id(),
                "acquired_ms": now_ms(),
            })
            .to_string(),
        )
        .map_err(|e| format!("runtime.lock write failed: {e}"))
    }

    /// Read the holder pid out of a `runtime.lock`, if present and parseable.
    /// Returns `None` when the file is missing, unreadable, not JSON, or has no
    /// numeric `pid` field (a legacy lock predating pid-stamping).
    fn read_lock_holder(path: &Path) -> Option<u32> {
        let body = std::fs::read_to_string(path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&body).ok()?;
        json.get("pid")
            .and_then(|p| p.as_u64())
            .and_then(|p| u32::try_from(p).ok())
    }

    fn release_lock(&self) {
        if let Some(path) = &self.config.lock_path {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Boot the runtime: acquire the lock, spawn the child, and poll `/healthz`
    /// until healthy (Booting→Ready) or the boot timeout elapses (→Failed). On
    /// success the base URL is resolved and stored.
    pub async fn boot(&self) -> Result<(), String> {
        self.acquire_lock()?;
        self.transition(RuntimeSupervisorState::Booting, None);

        let (child, health_url) = match self.launcher.launch(&self.config.spec).await {
            Ok(v) => v,
            Err(e) => {
                self.transition(RuntimeSupervisorState::Failed, Some(e.clone()));
                self.release_lock();
                return Err(e);
            }
        };
        {
            let mut inner = self.inner.lock();
            inner.pid = child.pid();
            inner.started_at_ms = Some(now_ms());
            inner.health_url = Some(health_url.clone());
        }
        *self.child.lock() = Some(child);

        // Poll until healthy or boot timeout. The launcher's `poll_health` is the
        // sole I/O; the loop is deterministic given a fake launcher.
        let deadline = std::time::Instant::now() + self.config.boot_timeout;
        loop {
            match self.launcher.poll_health(&health_url).await {
                Ok(true) => {
                    self.transition(RuntimeSupervisorState::Ready, None);
                    return Ok(());
                }
                Ok(false) | Err(_) if std::time::Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                other => {
                    let reason = match other {
                        Err(e) => format!("boot health unreachable: {e}"),
                        _ => "boot health never went green".to_string(),
                    };
                    self.transition(RuntimeSupervisorState::Failed, Some(reason.clone()));
                    self.terminate_child().await;
                    self.release_lock();
                    return Err(reason);
                }
            }
        }
    }

    /// One supervision step: poll health + reconcile the state machine. Ready
    /// stays Ready while healthy; an unhealthy poll degrades (Ready→Degraded);
    /// a dead child or a degraded child past tolerance triggers a backoff
    /// restart (Degraded→Booting→Ready) until the per-window cap trips
    /// (→Failed). Returns the post-step state.
    pub async fn supervise_once(&self) -> RuntimeSupervisorState {
        let health_url = self.inner.lock().health_url.clone();
        let Some(url) = health_url else {
            return self.state();
        };

        // Liveness probe (never holds the child lock across the await).
        let alive = self.child_is_alive().await;

        match self.launcher.poll_health(&url).await {
            Ok(true) if alive => {
                self.transition(RuntimeSupervisorState::Ready, None);
                RuntimeSupervisorState::Ready
            }
            Ok(true) => {
                // Health green but child handle gone — treat as needing restart.
                self.attempt_restart("child handle lost while healthy")
                    .await
            }
            Ok(false) => {
                self.transition(
                    RuntimeSupervisorState::Degraded,
                    Some("healthz reported unhealthy".to_string()),
                );
                self.attempt_restart("runtime unhealthy").await
            }
            Err(e) => {
                self.transition(RuntimeSupervisorState::Degraded, Some(e.clone()));
                self.attempt_restart(&format!("healthz unreachable: {e}"))
                    .await
            }
        }
    }

    /// Probe child liveness without holding the child lock across the await.
    async fn child_is_alive(&self) -> bool {
        // Take the child out, probe, put it back. The supervisor is single-driver
        // (one tick loop), so this never races a concurrent terminate.
        let taken = self.child.lock().take();
        let (alive, back) = match taken {
            Some(child) => {
                let a = child.is_alive().await;
                (a, Some(child))
            }
            None => (false, None),
        };
        *self.child.lock() = back;
        alive
    }

    /// Restart with backoff, respecting the per-window cap. On cap → Failed.
    async fn attempt_restart(&self, reason: &str) -> RuntimeSupervisorState {
        let now = now_ms();
        let (restarts, capped) = {
            let mut inner = self.inner.lock();
            inner
                .restart_window
                .retain(|t| now.saturating_sub(*t) < self.config.backoff.window_ms);
            if inner.restart_window.len() as u32 >= self.config.backoff.max_restarts_per_window {
                inner.last_error = Some(format!(
                    "restart cap reached ({} in window): {reason}",
                    inner.restart_window.len()
                ));
                (inner.restarts, true)
            } else {
                inner.restart_window.push(now);
                inner.restarts += 1;
                (inner.restarts, false)
            }
        };
        if capped {
            // Clone the error out of the guard *before* calling `transition`
            // (which re-locks `inner`): a `self.inner.lock()...` temporary passed
            // as an argument is dropped only at the end of the statement, so it
            // would still be held when `transition` re-locks — a self-deadlock on
            // the non-reentrant parking_lot mutex.
            let last_error = self.inner.lock().last_error.clone();
            self.transition(RuntimeSupervisorState::Failed, last_error);
            self.terminate_child().await;
            self.release_lock();
            return RuntimeSupervisorState::Failed;
        }

        // Backoff delay from the ladder (clamp to the last rung).
        let idx = (restarts as usize).saturating_sub(1);
        let delay_ms = self
            .config
            .backoff
            .delays_ms
            .get(idx)
            .or_else(|| self.config.backoff.delays_ms.last())
            .copied()
            .unwrap_or(1000);
        self.transition(
            RuntimeSupervisorState::Booting,
            Some(format!("restart #{restarts} after {delay_ms}ms: {reason}")),
        );
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;

        self.terminate_child().await;
        match self.launcher.launch(&self.config.spec).await {
            Ok((child, health_url)) => {
                {
                    let mut inner = self.inner.lock();
                    inner.pid = child.pid();
                    inner.started_at_ms = Some(now_ms());
                    inner.health_url = Some(health_url.clone());
                }
                *self.child.lock() = Some(child);
                // One immediate health probe to flip to Ready if it came up fast.
                match self.launcher.poll_health(&health_url).await {
                    Ok(true) => {
                        self.transition(RuntimeSupervisorState::Ready, None);
                        RuntimeSupervisorState::Ready
                    }
                    _ => RuntimeSupervisorState::Booting,
                }
            }
            Err(e) => {
                self.transition(RuntimeSupervisorState::Failed, Some(e));
                self.release_lock();
                RuntimeSupervisorState::Failed
            }
        }
    }

    async fn terminate_child(&self) {
        let child = self.child.lock().take();
        if let Some(child) = child {
            child.terminate().await;
        }
    }

    /// Shut the runtime down: terminate the child + release the lock + go Down.
    pub async fn shutdown(&self) {
        self.terminate_child().await;
        self.release_lock();
        self.transition(RuntimeSupervisorState::Down, None);
    }

    fn transition(&self, state: RuntimeSupervisorState, error: Option<String>) {
        let mut inner = self.inner.lock();
        inner.state = state;
        if error.is_some() {
            inner.last_error = error;
        }
    }
}

// ── Production launcher: spawn the `hawking` binary, poll over reqwest ────────

/// The production launcher: spawns the `hawking serve` binary with `tokio` and
/// polls `/healthz` with `reqwest`. HTTP-only — the engine crates are never
/// linked (T5).
pub struct ProcessLauncher {
    client: reqwest::Client,
}

impl Default for ProcessLauncher {
    fn default() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap_or_default(),
        }
    }
}

/// A real OS child process wrapping `tokio::process::Child`.
struct ProcessChild {
    pid: Option<u32>,
    child: tokio::sync::Mutex<Option<tokio::process::Child>>,
}

#[async_trait::async_trait]
impl RuntimeChild for ProcessChild {
    fn pid(&self) -> Option<u32> {
        self.pid
    }

    async fn is_alive(&self) -> bool {
        let mut guard = self.child.lock().await;
        match guard.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }

    async fn terminate(&self) {
        let mut guard = self.child.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

#[async_trait::async_trait]
impl RuntimeLauncher for ProcessLauncher {
    async fn launch(&self, spec: &ProcessSpec) -> Result<(Box<dyn RuntimeChild>, String), String> {
        let mut argv = spec.argv.iter();
        let program = argv.next().ok_or_else(|| "empty argv".to_string())?;
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(argv);
        if let Some(cwd) = &spec.cwd {
            cmd.current_dir(cwd);
        }
        for (k, v) in &spec.env {
            cmd.env(k, v);
        }
        cmd.stdin(std::process::Stdio::null());
        let child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn {program}: {e}"))?;
        let pid = child.id();
        let health = spec
            .health_url
            .clone()
            .ok_or_else(|| "spec has no health_url".to_string())?;
        Ok((
            Box::new(ProcessChild {
                pid,
                child: tokio::sync::Mutex::new(Some(child)),
            }),
            health,
        ))
    }

    async fn poll_health(&self, url: &str) -> Result<bool, String> {
        match self.client.get(url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// Is the process with the given pid still alive?
///
/// On unix this is the canonical "signal 0" probe: `kill(pid, 0)` delivers no
/// signal but performs the existence + permission checks. A return of `0` means
/// the process exists; `EPERM` means it exists but we lack permission to signal
/// it (still alive); `ESRCH` means no such process (dead). On non-unix targets
/// we have no cheap probe, so we fail **closed** (assume alive) — better to
/// refuse a possibly-live lock than to steal one.
fn pid_is_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // pid 0 means "the calling process group" to kill(2) — never a real
        // lock holder; treat as not-alive so a bogus 0 lock is reclaimed.
        if pid == 0 {
            return false;
        }
        // SAFETY: kill with signal 0 only inspects; it never mutates our state.
        let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
        if rc == 0 {
            return true;
        }
        // rc == -1: distinguish "no such process" (dead) from "exists but EPERM".
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        true
    }
}

#[cfg(test)]
pub(crate) mod testkit {
    //! A fake in-process health server + launcher for supervisor tests. The
    //! "runtime" is a `tokio` `TcpListener` answering `200 OK` on `/healthz`
    //! (and a generate/embed stub the ModelProvider tests reuse) — no model, no
    //! binary.
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A controllable fake runtime: a TCP listener answering minimal HTTP. The
    /// `healthy` flag flips Ready↔Degraded; `crashed` makes `is_alive` false.
    pub struct FakeRuntime {
        pub addr: String,
        pub healthy: Arc<AtomicBool>,
        pub crashed: Arc<AtomicBool>,
        shutdown: Arc<AtomicBool>,
    }

    impl FakeRuntime {
        /// Bind an ephemeral port and serve until `shutdown` is set.
        pub async fn spawn() -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap().to_string();
            let healthy = Arc::new(AtomicBool::new(true));
            let crashed = Arc::new(AtomicBool::new(false));
            let shutdown = Arc::new(AtomicBool::new(false));
            let h = healthy.clone();
            let sd = shutdown.clone();
            tokio::spawn(async move {
                loop {
                    if sd.load(Ordering::SeqCst) {
                        break;
                    }
                    let accept =
                        tokio::time::timeout(Duration::from_millis(100), listener.accept()).await;
                    let Ok(Ok((mut stream, _))) = accept else {
                        continue;
                    };
                    let healthy_now = h.load(Ordering::SeqCst);
                    tokio::spawn(async move {
                        use tokio::io::{AsyncReadExt, AsyncWriteExt};
                        let mut buf = [0u8; 2048];
                        let n = stream.read(&mut buf).await.unwrap_or(0);
                        let req = String::from_utf8_lossy(&buf[..n]);
                        let body = serve_fake(&req, healthy_now);
                        let status = if healthy_now {
                            "200 OK"
                        } else {
                            "503 Service Unavailable"
                        };
                        let resp = format!(
                            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        );
                        let _ = stream.write_all(resp.as_bytes()).await;
                        let _ = stream.flush().await;
                    });
                }
            });
            Self {
                addr,
                healthy,
                crashed,
                shutdown,
            }
        }

        pub fn base_url(&self) -> String {
            format!("http://{}", self.addr)
        }

        pub fn set_healthy(&self, v: bool) {
            self.healthy.store(v, Ordering::SeqCst);
        }

        pub fn set_crashed(&self, v: bool) {
            self.crashed.store(v, Ordering::SeqCst);
        }

        pub fn stop(&self) {
            self.shutdown.store(true, Ordering::SeqCst);
        }
    }

    /// Minimal request router for the fake: `/healthz`, `/v1/chat/completions`,
    /// `/v1/embeddings`, `/v1/hawking/generate`.
    fn serve_fake(req: &str, healthy: bool) -> String {
        let first = req.lines().next().unwrap_or_default();
        if first.contains("/healthz") {
            return if healthy {
                "ok".to_string()
            } else {
                "unhealthy".to_string()
            };
        }
        if first.contains("/v1/embeddings") {
            return serde_json::json!({
                "data": [{ "embedding": [0.1f32, 0.2, 0.3] }]
            })
            .to_string();
        }
        if first.contains("/v1/chat/completions") {
            return serde_json::json!({
                "choices": [{ "message": { "content": "fake completion" } }]
            })
            .to_string();
        }
        if first.contains("/v1/hawking/generate") {
            // Native non-stream JSON-full shape.
            return serde_json::json!({
                "text": "fake generate",
                "stats": { "input_tokens": 1, "output_tokens": 2, "dec_tps": 42.0 }
            })
            .to_string();
        }
        "{}".to_string()
    }

    /// A launcher backed by a [`FakeRuntime`]. `launch` returns a child wired to
    /// the fake's `crashed` flag and the fake's `/healthz` URL.
    pub struct FakeLauncher {
        pub runtime: Arc<FakeRuntime>,
        pub fail_launch: Arc<AtomicBool>,
        client: reqwest::Client,
    }

    impl FakeLauncher {
        pub fn new(runtime: Arc<FakeRuntime>) -> Self {
            Self {
                runtime,
                fail_launch: Arc::new(AtomicBool::new(false)),
                client: reqwest::Client::builder()
                    .timeout(Duration::from_millis(500))
                    .build()
                    .unwrap(),
            }
        }
    }

    struct FakeChild {
        crashed: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl RuntimeChild for FakeChild {
        fn pid(&self) -> Option<u32> {
            Some(4242)
        }
        async fn is_alive(&self) -> bool {
            !self.crashed.load(Ordering::SeqCst)
        }
        async fn terminate(&self) {}
    }

    #[async_trait::async_trait]
    impl RuntimeLauncher for FakeLauncher {
        async fn launch(
            &self,
            _spec: &ProcessSpec,
        ) -> Result<(Box<dyn RuntimeChild>, String), String> {
            if self.fail_launch.load(Ordering::SeqCst) {
                return Err("fake launch refused".to_string());
            }
            Ok((
                Box::new(FakeChild {
                    crashed: self.runtime.crashed.clone(),
                }),
                format!("{}/healthz", self.runtime.base_url()),
            ))
        }

        async fn poll_health(&self, url: &str) -> Result<bool, String> {
            match self.client.get(url).send().await {
                Ok(resp) => Ok(resp.status().is_success()),
                Err(e) => Err(e.to_string()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::{FakeLauncher, FakeRuntime};
    use super::*;

    fn test_config() -> SupervisorConfig {
        SupervisorConfig {
            spec: ProcessSpec {
                name: "fake-serve".to_string(),
                argv: vec!["fake".to_string()],
                cwd: None,
                env: Default::default(),
                health_url: None,
            },
            backoff: BackoffPolicy {
                delays_ms: vec![1, 1, 1],
                max_restarts_per_window: 2,
                window_ms: 60_000,
            },
            health_interval: Duration::from_millis(10),
            boot_timeout: Duration::from_secs(2),
            lock_path: None,
        }
    }

    #[tokio::test]
    async fn boot_reaches_ready_against_fake() {
        let rt = Arc::new(FakeRuntime::spawn().await);
        let sup = RuntimeSupervisor::new(test_config(), Arc::new(FakeLauncher::new(rt.clone())));
        assert_eq!(sup.state(), RuntimeSupervisorState::Down);
        sup.boot().await.unwrap();
        assert_eq!(sup.state(), RuntimeSupervisorState::Ready);
        // base_url derives cleanly from the health URL.
        assert_eq!(sup.base_url(), Some(rt.base_url()));
        sup.shutdown().await;
        assert_eq!(sup.state(), RuntimeSupervisorState::Down);
        rt.stop();
    }

    #[tokio::test]
    async fn unhealthy_poll_degrades_then_restarts_to_ready() {
        let rt = Arc::new(FakeRuntime::spawn().await);
        let sup = RuntimeSupervisor::new(test_config(), Arc::new(FakeLauncher::new(rt.clone())));
        sup.boot().await.unwrap();
        // Flip unhealthy: supervise_once degrades + restarts. The relaunch's
        // immediate probe sees the (still-unhealthy) server, so it stays Booting.
        rt.set_healthy(false);
        let state = sup.supervise_once().await;
        assert!(matches!(
            state,
            RuntimeSupervisorState::Booting | RuntimeSupervisorState::Degraded
        ));
        // Recover: a healthy poll returns to Ready.
        rt.set_healthy(true);
        let state = sup.supervise_once().await;
        assert_eq!(state, RuntimeSupervisorState::Ready);
        rt.stop();
    }

    #[tokio::test]
    async fn restart_cap_drives_to_failed() {
        let rt = Arc::new(FakeRuntime::spawn().await);
        let sup = RuntimeSupervisor::new(test_config(), Arc::new(FakeLauncher::new(rt.clone())));
        sup.boot().await.unwrap();
        rt.set_healthy(false);
        // window cap = 2: two restarts allowed, the third trips Failed.
        let _ = sup.supervise_once().await; // restart #1
        let _ = sup.supervise_once().await; // restart #2
        let state = sup.supervise_once().await; // cap → Failed
        assert_eq!(state, RuntimeSupervisorState::Failed);
        assert!(sup.status().last_error.unwrap().contains("restart cap"));
        rt.stop();
    }

    #[tokio::test]
    async fn crashed_child_while_healthy_triggers_restart() {
        let rt = Arc::new(FakeRuntime::spawn().await);
        let sup = RuntimeSupervisor::new(test_config(), Arc::new(FakeLauncher::new(rt.clone())));
        sup.boot().await.unwrap();
        // Health stays green but the child handle reports dead → restart path.
        rt.set_crashed(true);
        let state = sup.supervise_once().await;
        // The relaunched child is alive again (a fresh FakeChild), so the
        // post-restart immediate probe flips it back to Ready.
        assert!(matches!(
            state,
            RuntimeSupervisorState::Ready | RuntimeSupervisorState::Booting
        ));
        assert!(sup.status().restarts >= 1);
        rt.stop();
    }

    #[tokio::test]
    async fn launch_failure_is_failed_state() {
        let rt = Arc::new(FakeRuntime::spawn().await);
        let launcher = FakeLauncher::new(rt.clone());
        launcher
            .fail_launch
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let sup = RuntimeSupervisor::new(test_config(), Arc::new(launcher));
        let err = sup.boot().await.unwrap_err();
        assert!(err.contains("refused"));
        assert_eq!(sup.state(), RuntimeSupervisorState::Failed);
        rt.stop();
    }

    /// Pick a pid that is (almost certainly) not alive. We probe upward until
    /// `pid_is_alive` reports false, so the test is robust regardless of which
    /// pids happen to be running.
    fn dead_pid() -> u32 {
        for pid in (90_000u32..=100_000).rev() {
            if !pid_is_alive(pid) {
                return pid;
            }
        }
        // Fallback: pid 0 is reclaimable by construction.
        0
    }

    #[test]
    fn pid_is_alive_for_self_dead_for_bogus() {
        // Our own pid is unmistakably alive.
        assert!(pid_is_alive(std::process::id()));
        // A pid we just confirmed has no process must read dead.
        assert!(!pid_is_alive(dead_pid()));
    }

    #[tokio::test]
    async fn live_lock_is_not_stolen_but_stale_lock_is_reclaimed() {
        let dir = std::env::temp_dir().join(format!("hide_sup_steal_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let lock = dir.join("runtime.lock");

        let rt = Arc::new(FakeRuntime::spawn().await);
        let mut cfg = test_config();
        cfg.lock_path = Some(lock.clone());

        // 1) A lock stamped with the *current* (alive) pid must NOT be stolen.
        std::fs::write(
            &lock,
            serde_json::json!({
                "name": "other-host",
                "pid": std::process::id(),
                "acquired_ms": now_ms(),
            })
            .to_string(),
        )
        .unwrap();
        let sup = RuntimeSupervisor::new(cfg.clone(), Arc::new(FakeLauncher::new(rt.clone())));
        let err = sup.boot().await.unwrap_err();
        assert!(
            err.contains("held by live process"),
            "boot should refuse a live lock, got: {err}"
        );
        assert_eq!(sup.state(), RuntimeSupervisorState::Down);
        // The live host's lock was left untouched (still names the live pid).
        let body = std::fs::read_to_string(&lock).unwrap();
        assert!(body.contains(&std::process::id().to_string()));

        // 2) A lock stamped with a dead/bogus pid MUST be reclaimed and booted.
        std::fs::write(
            &lock,
            serde_json::json!({
                "name": "ghost-host",
                "pid": dead_pid(),
                "acquired_ms": now_ms(),
            })
            .to_string(),
        )
        .unwrap();
        let sup2 = RuntimeSupervisor::new(cfg, Arc::new(FakeLauncher::new(rt.clone())));
        sup2.boot().await.unwrap();
        assert_eq!(sup2.state(), RuntimeSupervisorState::Ready);
        // The reclaimed lock is now ours — stamped with our own pid.
        let body = std::fs::read_to_string(&lock).unwrap();
        assert!(body.contains(&std::process::id().to_string()));

        sup2.shutdown().await;
        let _ = std::fs::remove_dir_all(dir);
        rt.stop();
    }

    #[tokio::test]
    async fn runtime_lock_is_written_and_released() {
        let dir = std::env::temp_dir().join(format!("hide_sup_lock_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let lock = dir.join("runtime.lock");
        let rt = Arc::new(FakeRuntime::spawn().await);
        let mut cfg = test_config();
        cfg.lock_path = Some(lock.clone());
        let sup = RuntimeSupervisor::new(cfg, Arc::new(FakeLauncher::new(rt.clone())));
        sup.boot().await.unwrap();
        assert!(lock.exists(), "runtime.lock should exist while Ready");
        sup.shutdown().await;
        assert!(
            !lock.exists(),
            "runtime.lock should be released on shutdown"
        );
        let _ = std::fs::remove_dir_all(dir);
        rt.stop();
    }
}
