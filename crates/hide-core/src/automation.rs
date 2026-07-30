//! HIDE YOU automations: durable, permission-bounded background jobs.
//!
//! An [`Automation`] is a declared standing goal (reminder, recurring brief,
//! connector summary, calendar prep, email triage, project status check, watch
//! condition, research monitor, file ingestion pipeline, or agent job). Every
//! automation carries a closed [`PermissionSet`]. The job it spawns receives a
//! [`JobCapability`] *derived from* that set and structurally cannot widen it.
//!
//! # The property that matters most
//!
//! **A background agent cannot inherit broader authority than the automation
//! grants.** Tool use is gated by the job capability; an attempt to call a tool
//! the automation did not grant fails closed and is recorded on the result.
//!
//! # What this module is (and is not)
//!
//! * **Is:** declaration model, capability derivation, durable store, injected
//!   clock, stop-condition enforcement, schedule-slot idempotency, fixture tool
//!   registry, inspectable result history.
//! * **Is not:** a wall-clock daemon, launchd/cron installer, real connector or
//!   model execution, fleets, Fabric, or Metal. Real tool bodies are fixture
//!   stubs; wall-clock wiring is a later step.
//!
//! Model-free throughout. Deterministic under an injected [`Clock`].


#[path = "automation_types.rs"]
mod automation_types;
pub use automation_types::*;

#[path = "automation_fixture.rs"]
mod automation_fixture;
pub use automation_fixture::*;

#[path = "automation_decl.rs"]
mod automation_decl;
pub use automation_decl::*;

#[path = "automation_engine.rs"]
mod automation_engine;
pub use automation_engine::*;

#[cfg(test)]
mod tests {
    use super::*;
    // Scoped to the tests: a lib-only `cargo fix` cannot see these uses.
    use crate::persistence::DynKeyValueStore;
    use crate::error::HideError;
    use serde_json::json;
    use std::sync::Arc;
    use crate::persistence::InMemoryKeyValueStore;
    fn engine_at(ms: u64) -> (Arc<InjectedClock>, AutomationEngine) {
        let clock = Arc::new(InjectedClock::new(ms));
        let kv: DynKeyValueStore = Arc::new(InMemoryKeyValueStore::default());
        let engine = AutomationEngine::new(kv, clock.clone(), standard_fixture_registry());
        (clock, engine)
    }
    fn sample_automation(now: u64) -> Automation {
        Automation::declare(
            AutomationKind::EmailTriage,
            "triage unread mail",
            TriggerSpec::Interval {
                every_ms: 60_000,
                anchor_ms: now,
            },
            ["email.list", "email.summarize"],
            ["gmail"],
            ResourceBudget {
                max_runs: Some(10),
                max_tool_calls: Some(50),
                max_wall_ms: None,
                max_tokens: Some(1_000),
            },
            NotificationPolicy::OnFailure,
            StopCondition::Never,
            now,
        )
    }
    #[test]
    fn capability_is_derived_and_cannot_be_widened() {
        let perms = PermissionSet::new(["email.list", "email.summarize"], ["gmail"]);
        let cap = perms.derive_capability();
        assert!(cap.allows_tool("email.list"));
        assert!(!cap.allows_tool("shell.run"));
        assert!(cap.is_within(&perms));
        let sub = perms
            .derive_capability_subset(["email.list"], None::<&str>)
            .unwrap();
        assert!(sub.allows_tool("email.list"));
        assert!(!sub.allows_tool("email.summarize"));
        let err = perms
            .derive_capability_subset(["shell.run"], None::<&str>)
            .unwrap_err();
        assert!(matches!(err, HideError::CapabilityMissing(_)));
    }
    #[test]
    fn authority_containment_fail_closed_and_recorded() {
        let (clock, engine) = engine_at(1_000);
        let a = sample_automation(clock.now_ms());
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();
        let plan = JobPlan {
            tool_calls: vec![
                ("email.list".into(), json!({})),
                ("shell.run".into(), json!({"cmd": "rm -rf /"})),
            ],
        };
        let result = engine.run_manual(&id, plan).unwrap();
        assert!(!result.ok, "job must fail closed on ungranted tool");
        assert!(result.tool_attempts.iter().any(|t| { t.tool == "shell.run" && !t.authorized && !t.ok }));
        assert!(matches!( result.stop_reason, Some(StopReason::AuthorityDenied { ref tool }) if tool == "shell.run" ));
        let a = engine.get(&id).unwrap();
        assert_eq!(a.status, AutomationStatus::Active);
        assert!(a.last_result.as_ref().is_some_and(|r| !r.ok));
        let inspected = engine.inspect(&id, 5).unwrap();
        let results = inspected["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["ok"], false);
    }
    #[test]
    fn authorized_tools_succeed_under_capability() {
        let (clock, engine) = engine_at(1_000);
        let a = sample_automation(clock.now_ms());
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();
        let plan = JobPlan {
            tool_calls: vec![
                ("email.list".into(), json!({})),
                ("email.summarize".into(), json!({})),
            ],
        };
        let result = engine.run_manual(&id, plan).unwrap();
        assert!(result.ok);
        assert_eq!(result.tool_attempts.len(), 2);
        assert!(result.tool_attempts.iter().all(|t| t.authorized && t.ok));
    }
    #[test]
    fn durable_across_restart_next_run_survives() {
        let clock = Arc::new(InjectedClock::new(10_000));
        let kv: DynKeyValueStore = Arc::new(InMemoryKeyValueStore::default());
        let registry = standard_fixture_registry();
        let next_run = {
            let engine = AutomationEngine::new(kv.clone(), clock.clone(), registry.clone());
            let mut a = sample_automation(clock.now_ms());
            a.next_run_ms = Some(99_000);
            let created = engine.create(a).unwrap();
            let id = created.id.as_str().to_string();
            drop(engine);
            (id, 99_000u64)
        };
        let engine2 = AutomationEngine::new(kv, clock, registry);
        let recovered = engine2.recover().unwrap();
        assert_eq!(recovered.len(), 1);
        let a = engine2.get(next_run.0.as_str()).unwrap();
        assert_eq!(a.next_run_ms, Some(next_run.1));
        assert_eq!(a.goal, "triage unread mail");
        assert!(a.permissions.grants_tool("email.list"));
    }
    #[test]
    fn stop_condition_after_runs_is_enforced() {
        let (clock, engine) = engine_at(5_000);
        let a = Automation::declare(
            AutomationKind::Reminder,
            "nudge me twice",
            TriggerSpec::Manual,
            ["notify.send"],
            None::<&str>,
            ResourceBudget::default(),
            NotificationPolicy::Silent,
            StopCondition::AfterRuns { count: 2 },
            clock.now_ms(),
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();
        let plan = JobPlan {
            tool_calls: vec![("notify.send".into(), json!({}))],
        };
        let r1 = engine.run_manual(&id, plan.clone()).unwrap();
        assert!(r1.ok);
        assert!(engine.get(&id).unwrap().status.may_run());
        let r2 = engine.run_manual(&id, plan).unwrap();
        assert!(r2.ok);
        assert!(matches!( r2.stop_reason, Some(StopReason::AfterRuns { count: 2 }) ));
        let a = engine.get(&id).unwrap();
        assert_eq!(a.status, AutomationStatus::Stopped);
 assert!(matches!( a.stop_reason, Some(StopReason::AfterRuns { count: 2 }) ));
        let err = engine
            .run_manual(
                &id,
                JobPlan {
                    tool_calls: vec![("notify.send".into(), json!({}))],
                },
            )
            .unwrap_err();
        assert!(matches!(err, HideError::InvalidState(_)));
    }
    #[test]
    fn budget_exhaustion_halts_and_records_why() {
        let (clock, engine) = engine_at(0);
        let a = Automation::declare(
            AutomationKind::ResearchMonitor,
            "watch once",
            TriggerSpec::Manual,
            ["web.search"],
            None::<&str>,
            ResourceBudget {
                max_runs: Some(1),
                max_tool_calls: None,
                max_wall_ms: None,
                max_tokens: None,
            },
            NotificationPolicy::Always,
            StopCondition::Never,
            clock.now_ms(),
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();
        let r = engine
            .run_manual(
                &id,
                JobPlan {
                    tool_calls: vec![("web.search".into(), json!({}))],
                },
            )
            .unwrap();
        assert!(r.ok);
        assert!(matches!( r.stop_reason, Some(StopReason::BudgetExhausted { ref axis }) if axis == "max_runs" ));
        let a = engine.get(&id).unwrap();
        assert_eq!(a.status, AutomationStatus::Stopped);
        assert!(!a.results[0].notifications.is_empty());
    }
    #[test]
    fn condition_met_stop_is_enforced() {
        let (clock, engine) = engine_at(0);
        let a = Automation::declare(
            AutomationKind::WatchCondition,
            "watch deploy",
            TriggerSpec::Watch {
                condition: "deploy_green".into(),
            },
            ["notify.send"],
            None::<&str>,
            ResourceBudget::default(),
            NotificationPolicy::Silent,
            StopCondition::ConditionMet {
                name: "deploy_green".into(),
            },
            clock.now_ms(),
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();
        let result = engine.signal_condition(&id, "deploy_green").unwrap();
        assert!(result.is_none());
        let a = engine.get(&id).unwrap();
        assert_eq!(a.status, AutomationStatus::Stopped);
        assert!(matches!(
            a.stop_reason,
            Some(StopReason::ConditionMet { ref name }) if name == "deploy_green"
        ));
    }
    #[test]
    fn schedule_slot_is_idempotent() {
        let (clock, engine) = engine_at(100_000);
        let a = Automation::declare(
            AutomationKind::RecurringBrief,
            "morning brief",
            TriggerSpec::CronSlot {
                slot_key: "2026-07-27T09:00".into(),
                at_ms: 100_000,
            },
            ["calendar.list", "calendar.prepare"],
            ["calendar"],
            ResourceBudget::default(),
            NotificationPolicy::Silent,
            StopCondition::Never,
            clock.now_ms(),
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();
        let plan = JobPlan {
            tool_calls: vec![
                ("calendar.list".into(), json!({})),
                ("calendar.prepare".into(), json!({})),
            ],
        };
        let r1 = engine.tick(&plan).unwrap();
        assert_eq!(r1.len(), 1, "first tick should run once");
        assert!(r1[0].ok);
        let r2 = engine.tick(&plan).unwrap();
 assert!( r2.is_empty(), "idempotent: same slot must not run twice, got {:?}", r2 );
        let slot = "cron:2026-07-27T09:00";
        let r3 = engine.fire_slot(&id, slot, plan).unwrap();
        assert!(r3.is_none());
        let a = engine.get(&id).unwrap();
        assert_eq!(a.usage.runs, 1);
        assert_eq!(a.results.len(), 1);
    }
    #[test]
    fn interval_slots_advance_and_do_not_double_fire() {
        let (clock, engine) = engine_at(0);
        let a = Automation::declare(
            AutomationKind::ProjectStatusCheck,
            "hourly status",
            TriggerSpec::Interval {
                every_ms: 3_600_000,
                anchor_ms: 0,
            },
            ["fs.read"],
            None::<&str>,
            ResourceBudget::default(),
            NotificationPolicy::Silent,
            StopCondition::Never,
            0,
        );
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();
        let plan = JobPlan {
            tool_calls: vec![("fs.read".into(), json!({}))],
        };
        let r0 = engine.tick(&plan).unwrap();
        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].schedule_slot.as_deref(), Some("interval:3600000:0"));
        clock.advance(1_000);
        let r_same = engine.tick(&plan).unwrap();
        assert!(r_same.is_empty());
        clock.set(3_600_000);
        let r1 = engine.tick(&plan).unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].schedule_slot.as_deref(), Some("interval:3600000:1"));
        let a = engine.get(&id).unwrap();
        assert_eq!(a.usage.runs, 2);
        assert_eq!(a.next_run_ms, Some(7_200_000));
    }
    #[test]
    fn inspect_exposes_full_declaration_and_history() {
        let (clock, engine) = engine_at(0);
        let a = sample_automation(clock.now_ms());
        let id = a.id.as_str().to_string();
        engine.create(a).unwrap();
        engine
            .run_manual(
                &id,
                JobPlan {
                    tool_calls: vec![("email.list".into(), json!({}))],
                },
            )
            .unwrap();
        let view = engine.inspect(&id, 10).unwrap();
        assert_eq!(view["declaration"]["goal"], "triage unread mail");
        assert!(view["declaration"]["permissions"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "email.list"));
        assert_eq!(view["results"].as_array().unwrap().len(), 1);
    }
    #[test]
    fn job_capability_has_no_public_widen_path() {
        let parent = PermissionSet::new(["fs.read"], None::<&str>);
        let cap = parent.derive_capability();
        assert!(cap.is_live());
        assert!(!cap.allows_tool("fs.write"));
        assert!(cap.require_tool("fs.write").is_err());
    }
    #[test]
    fn adversarial_forged_job_capability_via_serde_is_dead() {
        let forged: JobCapability = serde_json::from_value(json!({
            "tools": ["email.send", "shell.exec"],
            "connectors": ["gmail"],
            "live": true
        }))
        .expect("shape deserializes");
 assert!( !forged.is_live(), "serde must not mint a live JobCapability" );
        assert!(!forged.allows_tool("email.send"));
        assert!(forged.require_tool("email.send").is_err());
        assert!(!forged.allows_connector("gmail"));
    }
}
