//! Stage 4 integration (campaign Trace G, model-free): background promotion of a
//! live interactive run, the durable-thread four-verb lifecycle plus the
//! init-guard discard, and Initialize per-connection notification suppression.
//!
//! The four-verb lifecycle and the init-guard are unit-tested in
//! `src/live_thread.rs`; the capability registry in `src/initialize.rs`. This file
//! exercises them (and promotion) end to end through the real [`BackendHost`]:
//! the durable JSONL event log, the durable job store, the interrupt hub, and the
//! replay projection are all the real thing.
//!
//! The one model decision -- an interactive run's streamed output -- is
//! DEFERRED_MODEL_REQUIRED (no runtime in the harness), so it is substituted by a
//! scripted run event stream tagged with the run id. Everything downstream
//! (promotion, durability, disconnect survival, control, resume) is the real path.

use hide_backend::live_thread::{LiveThread, LiveThreadInitGuard, THREAD_PERSISTED_KIND};
use hide_backend::services::{Budget, JobStatus};
use hide_backend::{BackendHost, ClientCapabilities, ClientInfo};
use hide_core::api::{Intent, UiEvent, UiEventKind};
use hide_core::event::{Event, NewEvent};
use hide_core::ids::{now_ms, RunId, SessionId};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

// --- scaffolding -------------------------------------------------------------

fn unique_dir(tag: &str) -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let uniq = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{tag}_{}_{}", now_ms(), uniq))
}

async fn append_run_event(
    host: &BackendHost,
    session: &SessionId,
    run: &RunId,
    kind: &str,
    payload: serde_json::Value,
) {
    host.services
        .event_log
        .append(NewEvent::system(session.clone(), kind, payload).with_run(run.clone()))
        .await
        .unwrap();
}

async fn scan(host: &BackendHost, session: &SessionId) -> Vec<Event> {
    host.services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap()
}

// --- (1) Trace G: promote -> disconnect -> reconnect -> inspect/steer/resume --

#[tokio::test]
async fn trace_g_promote_disconnect_reconnect_inspect_steer_resume() {
    let dir = unique_dir("hide_trace_g");
    let session;
    let run = RunId::new();
    let job_id;
    let pre_restart_kinds: Vec<String>;

    // --- Phase 1: an interactive run streams, then is promoted to background ---
    {
        let host = BackendHost::open_workspace(&dir).unwrap();
        session = host.services.session();

        // Start an "interactive task": the model's streamed output is
        // DEFERRED_MODEL_REQUIRED, so a scripted run event stream (tagged with the
        // run id) stands in for it. The promotion + lifecycle below are REAL.
        append_run_event(&host, &session, &run, "agent.observation", json!({ "text": "read file A" })).await;
        append_run_event(&host, &session, &run, "agent.observation", json!({ "text": "read file B" })).await;

        // Promote the LIVE run to a durable background job WITHOUT restarting it.
        let budget = Budget {
            max_wall_secs: Some(1800),
            ..Budget::default()
        };
        let job = host
            .promote_run_to_background(run.clone(), session.clone(), Some("goal_bg".to_string()), budget)
            .await
            .unwrap();
        job_id = job.job_id.clone();
        assert_eq!(job.status, JobStatus::Running, "a promoted live run is Running, not Pending");
        assert_eq!(
            job.run_id.as_deref(),
            Some(run.as_str()),
            "the job reuses the running run id (no restart)"
        );

        // The run.promoted + job.created markers are on the durable log.
        let events = scan(&host, &session).await;
        assert!(events.iter().any(|e| e.kind == "run.promoted"), "a run.promoted marker ties run->job");
        assert!(events.iter().any(|e| e.kind == "job.created"), "job_create wrote a durable job.created");

        // Same host, still connected: the promotion is findable by run id.
        assert_eq!(host.background_job_for_run(&run).unwrap().job_id, job_id);

        pre_restart_kinds = events.iter().map(|e| e.kind.clone()).collect();
        // --- CLIENT CLOSE: the host handle drops here (client disconnects). The
        // durable record + event log persist to disk.
    }

    // --- Phase 2: RECONNECT as a fresh host over the same workspace (restart) ---
    let reopened = BackendHost::open_workspace(&dir).unwrap();

    // The background job survived the disconnect + restart, recovered ACTIVE.
    let recovered = reopened.jobs_recover();
    assert!(
        recovered.iter().any(|j| j.job_id == job_id && j.status == JobStatus::Running),
        "the promoted background job survives client close + restart"
    );
    let found = reopened
        .background_job_for_run(&run)
        .expect("the run->job binding survives restart");
    assert_eq!(found.job_id, job_id);

    // The event log is CONSISTENT across the restart: every pre-restart event is
    // still present, in the same order (the durable prefix is unchanged).
    let post_kinds: Vec<String> = scan(&reopened, &session).await.iter().map(|e| e.kind.clone()).collect();
    assert!(
        post_kinds.starts_with(&pre_restart_kinds),
        "the reopened log replays the pre-restart events unchanged and in order"
    );

    // --- Inspect state: the reconnecting client reads the run's artifacts ---
    let artifacts = reopened.background_job_artifacts(&job_id).await.unwrap();
    assert_eq!(artifacts["job"]["job_id"], json!(job_id));
    let run_events = artifacts["run_events"].as_array().unwrap();
    assert!(run_events.len() >= 2, "inspect surfaces the run's own tagged events");
    assert!(
        run_events.iter().all(|e| e["run_id"] == json!(run.as_str())),
        "inspect scopes strictly to the promoted run id"
    );

    // --- Steer the background run (reuses the shipped redirect_run control path) ---
    const STEER: &str = "switch to the auth module";
    let ack = reopened
        .handle_intent(Intent::Custom {
            name: "redirect_run".to_string(),
            payload: json!({ "run_id": run.as_str(), "text": STEER, "session_id": session.as_str() }),
        })
        .await
        .unwrap();
    assert!(ack.accepted, "steering a promoted background run is accepted");
    let steer = scan(&reopened, &session)
        .await
        .into_iter()
        .find(|e| e.kind == "turn.steer")
        .expect("a durable turn.steer event landed for the promoted run");
    assert_eq!(steer.payload.get("instruction").and_then(|v| v.as_str()), Some(STEER));
    assert_eq!(steer.run_id.as_ref().map(|r| r.as_str()), Some(run.as_str()));
    assert!(reopened.interrupts().is_pending(&run), "the steer reached the interrupt hub for the run");

    // --- Resume in foreground: reattach + continue ---
    let (resumed, projection) = reopened.resume_background_job_in_foreground(&job_id).await.unwrap();
    assert_eq!(resumed.status, JobStatus::Running, "the job is foregrounded and running");
    assert!(
        scan(&reopened, &session).await.iter().any(|e| e.kind == "run.resumed_foreground"),
        "a durable run.resumed_foreground marker landed"
    );
    assert_eq!(projection.session_id, session, "the replay projection is the run's own session");

    let _ = std::fs::remove_dir_all(dir);
}

// --- (2) The same promotion + resume reachable over the intent surface -------

#[tokio::test]
async fn background_promotion_intents_are_reachable_over_intent() {
    let dir = unique_dir("hide_bg_intent");
    let host = BackendHost::open_workspace(&dir).unwrap();
    let session = host.services.session();
    let run = RunId::new();

    // Promote via the `promote_run` custom intent (the wired FE gesture).
    let ack = host
        .handle_intent(Intent::Custom {
            name: "promote_run".to_string(),
            payload: json!({ "run_id": run.as_str(), "session_id": session.as_str(), "goal_id": "goal_x" }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    let job = host.background_job_for_run(&run).expect("promote_run created the background job");
    assert_eq!(job.status, JobStatus::Running);
    assert_eq!(job.goal_id.as_deref(), Some("goal_x"));

    // Resume-in-foreground via the `resume_run_foreground` custom intent.
    let ack = host
        .handle_intent(Intent::Custom {
            name: "resume_run_foreground".to_string(),
            payload: json!({ "job_id": job.job_id }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    assert!(
        scan(&host, &session).await.iter().any(|e| e.kind == "run.resumed_foreground"),
        "the resume_run_foreground intent drove the durable resume"
    );

    // A missing run_id is rejected by the host method (surfaced as an Error UiEvent, not a
    // panic). The intent is still LOGGED, but the ack is refused with the reason: a caller
    // that reads `accepted` alone would otherwise render a promotion that never happened.
    let ack = host
        .handle_intent(Intent::Custom {
            name: "promote_run".to_string(),
            payload: json!({ "session_id": session.as_str() }),
        })
        .await
        .unwrap();
    assert!(
        ack.event_seq.is_some(),
        "the intent is logged even when the side effect fails"
    );
    assert!(!ack.accepted, "a failed side effect refuses the ack: {ack:?}");
    assert!(ack.message.is_some(), "the refusal carries the host's reason");

    let _ = std::fs::remove_dir_all(dir);
}

// --- (3) The four-verb lifecycle + init-guard over the host's event log ------

#[tokio::test]
async fn live_thread_four_verbs_and_init_guard_over_the_host_log() {
    let dir = unique_dir("hide_live_thread");
    let host = BackendHost::open_workspace(&dir).unwrap();
    let session = host.services.session();

    // A committed live thread: persist materializes the lazy marker AND flushes.
    let mut thread: LiveThread = {
        let mut guard = LiveThreadInitGuard::new(host.open_live_thread(session.clone()));
        guard
            .thread_mut()
            .unwrap()
            .append_item(NewEvent::system(session.clone(), "thread.item.kept", json!({})))
            .unwrap();
        guard.commit()
    };
    thread.persist().await.unwrap();
    let kinds: Vec<String> = scan(&host, &session).await.iter().map(|e| e.kind.clone()).collect();
    assert!(kinds.iter().any(|k| k == "thread.item.kept"), "committed + persisted item is durable");
    assert!(kinds.iter().any(|k| k == THREAD_PERSISTED_KIND), "persist wrote the lazy-state marker");

    // A failed init: the guard drops without commit, so the partial stream is
    // discarded (nothing durable), and the earlier committed items are untouched.
    let before = scan(&host, &session).await.len();
    {
        let mut guard = LiveThreadInitGuard::new(host.open_live_thread(session.clone()));
        guard
            .thread_mut()
            .unwrap()
            .append_item(NewEvent::system(session.clone(), "thread.item.partial", json!({})))
            .unwrap();
        // early return from a failing init -> guard drops here without commit
    }
    let after = scan(&host, &session).await;
    assert_eq!(after.len(), before, "a discarded (failed) init writes nothing to the durable log");
    assert!(!after.iter().any(|e| e.kind == "thread.item.partial"), "the partial item never became durable");

    let _ = std::fs::remove_dir_all(dir);
}

// --- (4) Initialize per-connection notification suppression in the emit path -

#[tokio::test]
async fn initialize_suppresses_opted_out_notifications_in_the_emit_path() {
    let dir = unique_dir("hide_init");
    let host = BackendHost::open_workspace(&dir).unwrap();

    // Connection A opts out of `runtime/status`; connection B initializes with no
    // opt-out; connection C never initializes at all.
    let resp = host.initialize(
        "conn-A",
        ClientInfo {
            name: "hide-desktop".to_string(),
            title: Some("HIDE".to_string()),
            version: "1.0.0".to_string(),
        },
        ClientCapabilities {
            experimental_api: true,
            opt_out_notification_methods: vec!["runtime/status".to_string()],
        },
    );
    assert!(resp.user_agent.starts_with("hide-backend/"), "the server-info reply names the server");
    assert!(!resp.platform_os.is_empty());
    assert!(host.connections().experimental_api("conn-A"), "the experimental_api gate was stored");

    host.initialize(
        "conn-B",
        ClientInfo { name: "cli".to_string(), title: None, version: "0.1".to_string() },
        ClientCapabilities::default(),
    );

    // A runtime/status UiEvent: suppressed for A, delivered for B and (never
    // initialized) C.
    let runtime_event = UiEvent {
        seq: 1,
        session_id: None,
        kind: UiEventKind::RuntimeStatus {
            status: "ready".to_string(),
            detail: Some("model online".to_string()),
        },
    };
    assert!(
        host.notification_for_connection("conn-A", &runtime_event).is_none(),
        "connection A opted out of runtime/status -> the emit path skips it"
    );
    assert!(
        host.notification_for_connection("conn-B", &runtime_event).is_some(),
        "connection B did not opt out -> it still receives runtime/status"
    );
    assert!(
        host.notification_for_connection("conn-C", &runtime_event).is_some(),
        "an uninitialized connection suppresses nothing"
    );

    // A DIFFERENT method (error) is NOT suppressed for A -- opt-out is per method.
    let error_event = UiEvent {
        seq: 2,
        session_id: None,
        kind: UiEventKind::Error {
            code: "boom".to_string(),
            message: "kaboom".to_string(),
        },
    };
    assert!(
        host.notification_for_connection("conn-A", &error_event).is_some(),
        "A opted out of runtime/status only, so error notifications still reach it"
    );

    let _ = std::fs::remove_dir_all(dir);
}
