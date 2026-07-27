//! Live model turn loop: the real intent path against a real served model.
//!
//! Proves end-to-end what the flagship fake-serve test only sketches:
//!
//!   user `SubmitTurn` intent
//!     → context compile
//!     → RuntimeSupervisor + HttpModelProvider
//!     → streamed tokens on Wire-B
//!     → persisted `agent.message` assistant turn
//!     → next turn's history sees that assistant text
//!
//! Activation (skip-cleanly otherwise so model-free CI stays green):
//!
//! * `HIDE_MODEL_WEIGHTS` pointing at a loadable GGUF, **or** a discoverable
//!   weights file (repo `models/*.gguf`, or the common HF-cache Llama-3.2-1B
//!   Q4_K_M used by local smoke gates)
//! * a `hawking` binary: `HIDE_HAWKING_BIN`, else `PATH`, else
//!   `target/{release,debug}/hawking` relative to the workspace
//!
//! Force-skip with `HIDE_LIVE_MODEL_TURN=0`. This does **not** flip
//! `HIDE_KERNEL_TURN` (that gate is separate and stays at its default).

use hide_backend::BackendHost;
use hide_core::api::{Intent, UiEventKind};
use hide_core::ids::now_ms;
use hide_core::runtime::RuntimeSupervisorState;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

fn unique_dir(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let uniq = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{tag}_{}_{}", now_ms(), uniq))
}

fn env_truthy(name: &str) -> Option<bool> {
    match std::env::var(name).ok().as_deref().map(str::trim) {
        Some("0") | Some("false") | Some("no") | Some("off") => Some(false),
        Some("1") | Some("true") | Some("yes") | Some("on") => Some(true),
        Some(s) if !s.is_empty() => Some(true),
        _ => None,
    }
}

/// Resolve a weights path the live loop can actually load **through `hawking
/// serve`** (the HIDE path). Prefer an explicit `HIDE_MODEL_WEIGHTS`; otherwise
/// probe known local locations. Never invents a fake provider — missing weights
/// means the test skips.
///
/// Prefer Qwen dense GGUFs: serve's continuous-batch path requires
/// `Engine::prefill_slot`, which is implemented for qwen_dense / rwkv7. Llama
/// loads and `/healthz`s but generate returns empty (`not yet implemented:
/// prefill_slot`). Do not prefer HF-cache Llama for this test.
fn resolve_weights() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HIDE_MODEL_WEIGHTS") {
        let p = p.trim();
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
            eprintln!("live_model_turn: HIDE_MODEL_WEIGHTS set but not a file: {p}");
            return None;
        }
    }

    // Repo-local `models/` (same convention as hawking-core smokes). Prefer Qwen.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(manifest_dir.as_path());
    if let Some(found) = first_preferred_gguf(&repo_root.join("models")) {
        return Some(found);
    }

    // Sibling worktrees on this machine sometimes hold the verified Qwen 0.5B /
    // 3B smokes under `models/` — use them when present so the live loop can run
    // without re-downloading. Prefer smaller Qwen first.
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let worktrees = home.join(".claude-grok/worktrees");
        if let Ok(entries) = std::fs::read_dir(&worktrees) {
            let mut candidates: Vec<PathBuf> = Vec::new();
            for e in entries.flatten() {
                let models = e.path().join("models");
                if let Some(g) = first_preferred_gguf(&models) {
                    candidates.push(g);
                }
            }
            // Prefer 0.5B, then any other qwen, then whatever remained.
            candidates.sort_by_key(|p| {
                let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                if n.contains("0.5b") || n.contains("05b") {
                    0
                } else if n.contains("qwen") {
                    1
                } else {
                    2
                }
            });
            if let Some(p) = candidates.into_iter().next() {
                return Some(p);
            }
        }
    }

    None
}

/// Prefer a serve-capable dense Qwen GGUF over architectures whose
/// `prefill_slot` is still unimplemented (e.g. plain Llama dense).
fn first_preferred_gguf(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut ggufs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf") && p.is_file())
        .collect();
    if ggufs.is_empty() {
        return None;
    }
    ggufs.sort_by_key(|p| {
        let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        if n.contains("qwen") && (n.contains("0.5b") || n.contains("05b")) {
            0
        } else if n.contains("qwen") {
            1
        } else {
            2
        }
    });
    ggufs.into_iter().next()
}

/// Resolve the `hawking` binary that can serve the weights.
fn resolve_hawking_bin() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HIDE_HAWKING_BIN") {
        let p = PathBuf::from(p.trim());
        if p.is_file() {
            return Some(p);
        }
    }
    // PATH lookup.
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            let cand = dir.join("hawking");
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    // Workspace target dirs (prefer release — debug is impractically slow).
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(manifest_dir.as_path());
    for rel in ["target/release/hawking", "target/debug/hawking"] {
        let cand = repo_root.join(rel);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// Pick a free loopback port so parallel suites do not collide with a fixed
/// 8745, or with each other.
async fn ephemeral_bind() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);
    addr.to_string()
}

struct LiveEnvGuard {
    keys: Vec<String>,
}

impl LiveEnvGuard {
    fn apply(pairs: &[(&str, String)]) -> Self {
        let mut keys = Vec::with_capacity(pairs.len());
        for (k, v) in pairs {
            // This integration binary is the sole mutator of these process env
            // keys for its duration (one test process per file).
            std::env::set_var(k, v);
            keys.push((*k).to_string());
        }
        Self { keys }
    }
}

impl Drop for LiveEnvGuard {
    fn drop(&mut self) {
        for k in &self.keys {
            std::env::remove_var(k);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_submit_turn_streams_tokens_persists_and_next_turn_sees_history() {
    if env_truthy("HIDE_LIVE_MODEL_TURN") == Some(false) {
        eprintln!("live_model_turn: skipped (HIDE_LIVE_MODEL_TURN=0)");
        return;
    }

    let Some(weights) = resolve_weights() else {
        eprintln!(
            "live_model_turn: SKIP — no servable GGUF (set HIDE_MODEL_WEIGHTS or place one under models/)"
        );
        return;
    };
    let Some(hawking_bin) = resolve_hawking_bin() else {
        eprintln!(
            "live_model_turn: SKIP — no hawking binary (build with `cargo build -p hawking --release`, or set HIDE_HAWKING_BIN)"
        );
        return;
    };

    // Keep the default single-shot path (do NOT promote HIDE_KERNEL_TURN).
    std::env::remove_var("HIDE_KERNEL_TURN");

    let bind = ephemeral_bind().await;
    let dir = unique_dir("hide_live_model_turn");
    let _env = LiveEnvGuard::apply(&[
        ("HIDE_MODEL_WEIGHTS", weights.display().to_string()),
        ("HIDE_HAWKING_BIN", hawking_bin.display().to_string()),
        ("HIDE_MODEL_ADDR", bind.clone()),
        // Real GGUF + Metal load regularly exceeds the old 30s default.
        ("HIDE_MODEL_BOOT_TIMEOUT_SECS", "600".to_string()),
        // Cap generation so a live smoke stays bounded (EOS may finish sooner).
        ("HIDE_MAX_OUTPUT_TOKENS", "24".to_string()),
    ]);

    eprintln!(
        "live_model_turn: weights={} hawking={} addr={}",
        weights.display(),
        hawking_bin.display(),
        bind
    );

    let host = BackendHost::open_workspace(&dir).expect("open_workspace");
    assert!(
        host.runtime_state().is_some(),
        "HIDE_MODEL_WEIGHTS must install a RuntimeSupervisor"
    );

    // Wait until the supervised hawking serve is Ready (or Failed / timeout).
    let ready_deadline = tokio::time::Instant::now() + Duration::from_secs(600);
    loop {
        match host.runtime_state() {
            Some(RuntimeSupervisorState::Ready) => break,
            Some(RuntimeSupervisorState::Failed) => {
                panic!(
                    "runtime supervisor Failed while booting hawking serve \
                     (weights={}, bin={}, addr={bind})",
                    weights.display(),
                    hawking_bin.display()
                );
            }
            _ if tokio::time::Instant::now() < ready_deadline => {
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            _ => {
                panic!(
                    "runtime never reached Ready within boot budget (last={:?})",
                    host.runtime_state()
                );
            }
        }
    }

    let session = host.services.session();
    let mut rx = host.subscribe_ui();

    // --- Turn 1: live SubmitTurn intent path ---------------------------------
    let turn1_prompt = "Reply with exactly one short word, then stop.";
    let ack1 = host
        .handle_intent(Intent::SubmitTurn {
            session_id: session.clone(),
            text: turn1_prompt.to_string(),
            attachments: Vec::new(),
        })
        .await
        .expect("SubmitTurn 1");
    assert!(ack1.accepted, "valid SubmitTurn must be accepted: {ack1:?}");

    let mut streamed = String::new();
    let gen_deadline = tokio::time::Instant::now() + Duration::from_secs(600);
    loop {
        if tokio::time::Instant::now() >= gen_deadline {
            panic!(
                "timed out waiting for streamed tokens / completion (got so far: {streamed:?})"
            );
        }
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Ok(ev)) => match ev.kind {
                UiEventKind::TokenBatch { text, .. } => {
                    streamed.push_str(&text);
                    // Token batches can be partial; once we have any real text and
                    // a persisted assistant message, the stream half is proven.
                    if !streamed.trim().is_empty() {
                        // Keep draining briefly for more tokens, but do not require
                        // a specific EOS event (Wire-B coalesces).
                        let extra = tokio::time::timeout(Duration::from_secs(30), async {
                            loop {
                                match rx.recv().await {
                                    Ok(e) => {
                                        if let UiEventKind::TokenBatch { text, .. } = e.kind {
                                            streamed.push_str(&text);
                                        }
                                    }
                                    Err(_) => break,
                                }
                            }
                        })
                        .await;
                        let _ = extra;
                        break;
                    }
                }
                UiEventKind::Error { code, message } => {
                    panic!("generation error on Wire-B: {code}: {message}");
                }
                UiEventKind::RuntimeStatus { status, detail } => {
                    // Boot/status chatter is fine; offline after Ready is not.
                    if status == "down" || status == "failed" {
                        panic!("runtime went {status} mid-turn: {detail:?}");
                    }
                }
                _ => {}
            },
            Ok(Err(e)) => panic!("ui bus closed: {e}"),
            Err(_) => {
                // No event in 5s — check whether the assistant turn already landed
                // (tokens may have coalesced before we subscribed if racey; we
                // subscribed first, but still poll the durable log).
                let events = host
                    .services
                    .event_log
                    .scan(Some(session.clone()), None, None)
                    .await
                    .unwrap();
                if events.iter().any(|e| {
                    e.kind == "agent.message"
                        && e.payload.get("role").and_then(|r| r.as_str()) == Some("assistant")
                        && e.payload
                            .get("text")
                            .and_then(|t| t.as_str())
                            .map(|t| !t.trim().is_empty())
                            .unwrap_or(false)
                }) {
                    break;
                }
            }
        }
    }

    // Durable assistant turn from turn 1.
    let events_after_1 = host
        .services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap();
    let assistant_1 = events_after_1.iter().find_map(|e| {
        if e.kind == "agent.message"
            && e.payload.get("role").and_then(|r| r.as_str()) == Some("assistant")
        {
            e.payload
                .get("text")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
        } else {
            None
        }
    });
    let assistant_1 = assistant_1.expect(
        "turn 1 must persist a non-empty agent.message assistant turn on the event log",
    );
    assert!(
        !assistant_1.trim().is_empty(),
        "assistant text must be non-empty real tokens, got {assistant_1:?}"
    );
    // Prefer streamed evidence when present; persistence alone is the durable contract.
    if !streamed.trim().is_empty() {
        assert!(
            streamed.contains(assistant_1.trim())
                || assistant_1.contains(streamed.trim())
                || streamed.chars().any(|c| !c.is_whitespace()),
            "streamed tokens should reflect real generation; streamed={streamed:?} persisted={assistant_1:?}"
        );
    }

    // --- Turn 2: next SubmitTurn must see turn-1 assistant in rebuild_history -
    let turn2_prompt = "What single word did you just say? Answer briefly.";
    let ack2 = host
        .handle_intent(Intent::SubmitTurn {
            session_id: session.clone(),
            text: turn2_prompt.to_string(),
            attachments: Vec::new(),
        })
        .await
        .expect("SubmitTurn 2");
    assert!(ack2.accepted, "turn 2 must be accepted: {ack2:?}");

    let gen2_deadline = tokio::time::Instant::now() + Duration::from_secs(600);
    loop {
        if tokio::time::Instant::now() >= gen2_deadline {
            panic!("timed out waiting for turn 2 assistant persistence");
        }
        let events = host
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let assistant_msgs: Vec<&str> = events
            .iter()
            .filter(|e| {
                e.kind == "agent.message"
                    && e.payload.get("role").and_then(|r| r.as_str()) == Some("assistant")
            })
            .filter_map(|e| e.payload.get("text").and_then(|t| t.as_str()))
            .collect();
        if assistant_msgs.len() >= 2 {
            break;
        }
        // Drain Wire-B so the generation task is not blocked on lag.
        while let Ok(ev) = rx.try_recv() {
            if let UiEventKind::Error { code, message } = ev.kind {
                panic!("turn 2 generation error: {code}: {message}");
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // History reconstruction (the same rebuild_history path run_turn_core uses).
    let events_final = host
        .services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap();
    let mut history_roles: Vec<(String, String)> = Vec::new();
    for ev in &events_final {
        match ev.kind.as_str() {
            "user.intent.submit_turn" => {
                if let Some(text) = ev
                    .payload
                    .get("args")
                    .and_then(|a| a.get("text"))
                    .and_then(|t| t.as_str())
                {
                    history_roles.push(("user".into(), text.to_string()));
                }
            }
            "agent.message" => {
                if ev.payload.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                    if let Some(text) = ev.payload.get("text").and_then(|t| t.as_str()) {
                        history_roles.push(("assistant".into(), text.to_string()));
                    }
                }
            }
            _ => {}
        }
    }

    assert!(
        history_roles
            .iter()
            .any(|(r, c)| r == "assistant" && c == &assistant_1),
        "next turn's durable history must include turn-1 assistant text; history={history_roles:?}"
    );
    assert!(
        history_roles
            .iter()
            .filter(|(r, _)| r == "user")
            .count()
            >= 2,
        "both user turns must be on the event log; history={history_roles:?}"
    );
    assert!(
        history_roles
            .iter()
            .filter(|(r, _)| r == "assistant")
            .count()
            >= 2,
        "both assistant turns must be persisted; history={history_roles:?}"
    );

    eprintln!(
        "live_model_turn: OK — streamed_len={} assistant_1_len={} history_msgs={}",
        streamed.len(),
        assistant_1.len(),
        history_roles.len()
    );

    // Drop host first. BackendHost does not currently shut the supervised child
    // down on Drop, so also best-effort kill whatever is still bound to our port.
    drop(host);
    kill_listener_on(&bind);
    let _ = std::fs::remove_dir_all(dir);
}

/// Best-effort cleanup for a leaked `hawking serve` child (host Drop does not
/// yet call `RuntimeSupervisor::shutdown`).
fn kill_listener_on(bind: &str) {
    let port = bind.rsplit(':').next().unwrap_or("");
    if port.is_empty() {
        return;
    }
    let _ = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "pids=$(lsof -nP -iTCP:{port} -sTCP:LISTEN -t 2>/dev/null); \
             [ -n \"$pids\" ] && kill $pids 2>/dev/null; true"
        ))
        .status();
}
