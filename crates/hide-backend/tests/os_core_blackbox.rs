//! Black-box capability contract for the HIDE OS core recomposition (Track S).
//!
//! These tests pin external behaviour that must survive crate merges and
//! architecture replacement. They call public APIs only. A failure means a
//! user-facing capability was lost quietly.

use hide_backend::lenses::{Claim, EvidenceTier, HandoffKind, Surface, SurfaceGraph};
use hide_backend::surfaces::SurfaceGraphService;
use hide_backend::{
    BackendHost, BackendServices, ClientCapabilities, ClientInfo, ConnectorRegistry,
};
use hide_core::api::{Intent, UiEvent, UiEventKind};
use hide_core::automation::{
    standard_fixture_registry, Automation, AutomationEngine, AutomationKind, Clock, InjectedClock,
    JobCapability, JobPlan, NotificationPolicy, PermissionSet, ResourceBudget, StopCondition,
    TriggerSpec,
};
use hide_core::config::HideConfig;
use hide_core::event::{EventLog, InMemoryEventLog, NewEvent};
use hide_core::ids::{now_ms, SessionId};
use hide_core::permission::{
    PermissionEngine, PermissionPolicy, PermissionRequest, PermissionRule, StaticPermissionEngine,
};
use hide_core::tool::ToolRegistry;
use hide_core::types::{Decision, RiskLevel};
use hide_protocol::{command_catalog, Method, PROTOCOL_VERSION};
use serde_json::json;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Dual event models (must_not_delete until Bridge): Event log + UiEvent wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blackbox_dual_event_models_both_exist_and_round_trip() {
    let log = InMemoryEventLog::default();
    let session = SessionId::from("s-bb");
    let stored = log
        .append(NewEvent::system(
            session.clone(),
            "test.ping",
            json!({"n": 1}),
        ))
        .await
        .expect("append");
    assert_eq!(stored.payload["n"], 1);
    let _ = stored.seq; // ordered seq assigned by the log

    // Wire UiEvent is a distinct type from the durable Event — dual until Bridge.
    let ui = UiEvent {
        seq: 1,
        session_id: Some(session),
        kind: UiEventKind::RuntimeStatus {
            status: "ok".into(),
            detail: None,
        },
    };
    let s = serde_json::to_string(&ui).unwrap();
    let back: UiEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.seq, 1);
    assert!(matches!(
        back.kind,
        UiEventKind::RuntimeStatus { status, .. } if status == "ok"
    ));
}

// ---------------------------------------------------------------------------
// Permission engine: deny beats allow
// ---------------------------------------------------------------------------

#[test]
fn blackbox_permission_deny_beats_allow() {
    let policy = PermissionPolicy {
        default_decision: Decision::Allow,
        rules: vec![
            PermissionRule {
                id: "allow-shell".into(),
                capability_kind: "shell".into(),
                scope_pattern: "*".into(),
                decision: Decision::Allow,
                max_risk: RiskLevel::Critical,
                reason: "tmp allowed".into(),
            },
            PermissionRule {
                id: "deny-shell".into(),
                capability_kind: "shell".into(),
                scope_pattern: "*".into(),
                decision: Decision::Deny,
                max_risk: RiskLevel::Critical,
                reason: "deny wins".into(),
            },
        ],
        risk_gates: vec![],
    };
    let eng = StaticPermissionEngine::new(policy);
    let v = eng.evaluate(&PermissionRequest {
        capability_kind: "shell".into(),
        target: "/bin/ls".into(),
        risk: RiskLevel::Low,
        effects: vec![],
        grant: None,
    });
    assert_eq!(v.decision, Decision::Deny, "deny must beat allow");
}

// ---------------------------------------------------------------------------
// Automation / YOU capability non-widening (JobCapability seal)
// ---------------------------------------------------------------------------

#[test]
fn blackbox_job_capability_cannot_widen_via_serde() {
    let forged: JobCapability = serde_json::from_value(json!({
        "tools": ["email.send", "shell.run"],
        "connectors": ["gmail"],
        "live": true
    }))
    .expect("shape deserializes");
    assert!(
        !forged.is_live(),
        "serde must not mint a live JobCapability"
    );
    assert!(!forged.allows_tool("email.send"));
    assert!(forged.require_tool("email.send").is_err());

    let live = PermissionSet::new(["notify.send"], None::<&str>).derive_capability();
    assert!(live.is_live());
    assert!(live.allows_tool("notify.send"));
    assert!(!live.allows_tool("shell.run"));
}

#[test]
fn blackbox_automation_engine_runs_fixture_tool_under_budget() {
    use hide_core::persistence::InMemoryKeyValueStore;

    let clock = Arc::new(InjectedClock::new(1_000_000));
    let kv: hide_core::persistence::DynKeyValueStore = Arc::new(InMemoryKeyValueStore::default());
    let engine = AutomationEngine::new(kv, clock.clone(), standard_fixture_registry());
    let a = Automation::declare(
        AutomationKind::AgentJob,
        "bb run",
        TriggerSpec::Manual,
        ["notify.send"],
        None::<&str>,
        ResourceBudget {
            max_runs: Some(1),
            max_tool_calls: None,
            max_wall_ms: None,
            max_tokens: None,
        },
        NotificationPolicy::Silent,
        StopCondition::AfterRuns { count: 1 },
        clock.now_ms(),
    );
    let id = a.id.as_str().to_string();
    engine.create(a).expect("create");
    let result = engine
        .run_manual(
            &id,
            JobPlan {
                tool_calls: vec![("notify.send".into(), json!({}))],
            },
        )
        .expect("run");
    assert!(result.ok);
}

// ---------------------------------------------------------------------------
// Protocol / IPC contract
// ---------------------------------------------------------------------------

#[test]
fn blackbox_protocol_version_and_catalog_stable_floor() {
    assert_eq!(PROTOCOL_VERSION, "hide.agent.v1");
    let catalog = command_catalog();
    assert!(
        catalog.len() >= 40,
        "command catalog must not shrink below the consolidation floor (got {})",
        catalog.len()
    );
    assert_eq!(Method::ThreadFork.as_str(), "thread/fork");
    assert!(catalog.iter().all(|c| !c.id.is_empty()));
}

// ---------------------------------------------------------------------------
// YOU / CHAT / IDE lenses: claim-only handoff, one session graph
// ---------------------------------------------------------------------------

#[test]
fn blackbox_surface_graph_claim_not_capability_handoff() {
    let mut g = SurfaceGraph::open("session-bb-1");
    assert_eq!(g.session_id(), "session-bb-1");
    // Front door is Chat.
    assert_eq!(g.active(), Surface::Chat);
    assert_eq!(g.switch(Surface::Ide), Surface::Ide);
    assert_eq!(g.switch(Surface::You), Surface::You);

    let claim = Claim {
        id: "c1".into(),
        text: "bb claim body".into(),
        evidence_tier: EvidenceTier::Asserted,
        payload: json!({}),
    };
    // Active must be YOU to seal YouToChat.
    let sealed = g
        .create_handoff(
            HandoffKind::YouToChat,
            1,
            vec![claim],
            vec![],
            json!({"note": "bb"}),
            "bb-actor",
        )
        .expect("seal");
    assert!(!sealed.claims.is_empty());
    // Capsule must refuse capability extraction.
    assert!(sealed.try_extract_capability().is_err());

    let opened = g.receive_handoff(&sealed.id).expect("receive");
    assert!(opened.capability_unchanged());
}

// ---------------------------------------------------------------------------
// Builtin tools catalog registers without panic
// ---------------------------------------------------------------------------

#[test]
fn blackbox_builtin_tools_register() {
    let registry = ToolRegistry::default();
    hide_kernel::tooling::register_builtin_tools(&registry);
    let n = registry.specs().len();
    assert!(n >= 15, "builtin tool catalog too small: {n}");
}

// ---------------------------------------------------------------------------
// Six memory classes exist on ClassedMemorySystem
// ---------------------------------------------------------------------------

#[test]
fn blackbox_six_memory_classes_open() {
    use hawking_context::memory_classes::{ClassedMemorySystem, MemoryClass};
    let mem = ClassedMemorySystem::open_in_memory("bb-ws").expect("open");
    for c in MemoryClass::all() {
        let n = mem.count(c).expect("count");
        assert_eq!(n, 0);
    }
    assert_eq!(MemoryClass::all().len(), 6);
}

// ---------------------------------------------------------------------------
// Host composition: initialize + surface graph service bind
// ---------------------------------------------------------------------------

#[test]
fn blackbox_backend_host_initialize_and_surface_lenses() {
    let dir = std::env::temp_dir().join(format!("hide_bb_host_{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = HideConfig::for_workspace(&dir);
    let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
    let resp = host.initialize(
        "conn-bb",
        ClientInfo {
            name: "bb".into(),
            title: None,
            version: "0".into(),
        },
        ClientCapabilities::default(),
    );
    assert!(resp.user_agent.contains("hide-backend"));

    let session = host.services.session();
    let surfaces = SurfaceGraphService::for_session(
        &session,
        host.services.event_log.clone(),
        host.ui_bus().clone(),
    );
    // Doctrine: Chat is the front door; YOU is a lens, not the default silo.
    assert_eq!(surfaces.active(), Surface::Chat);
    surfaces.switch_surface(Surface::You).expect("switch you");
    assert_eq!(surfaces.active(), Surface::You);
    surfaces.switch_surface(Surface::Ide).expect("switch ide");
    assert_eq!(surfaces.active(), Surface::Ide);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Intent wire + kernel type presence (agent authority still linked)
// ---------------------------------------------------------------------------

#[test]
fn blackbox_intent_submit_turn_and_kernel_linked() {
    let _ = std::any::type_name::<hide_kernel::AgentKernel>();
    let intent = Intent::SubmitTurn {
        session_id: SessionId::from("s"),
        text: "ping".into(),
        attachments: vec![],
    };
    let v = serde_json::to_value(&intent).unwrap();
    assert_eq!(v["type"], "submit_turn");
    assert_eq!(v["data"]["text"], "ping");
}

// ---------------------------------------------------------------------------
// Connector registry (host path) remains usable empty
// ---------------------------------------------------------------------------

#[test]
fn blackbox_connector_registry_empty() {
    let reg = ConnectorRegistry::default();
    assert!(reg.ids().is_empty());
    assert!(reg.get("nope").is_none());
}
