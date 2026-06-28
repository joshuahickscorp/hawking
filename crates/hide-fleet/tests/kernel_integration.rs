//! Integration test (bible ch.09 §4.1): the FleetManager schedules and drives a
//! REAL `hide-kernel` run via `KernelRunLauncher` — proving the declared
//! `hide-kernel` dep is load-bearing (scheduling kernel runs is the whole point).
//!
//! The kernel is backed by a `hawking-orch::StubInferenceClient` + `SimpleRouter`
//! so no live model is needed. The fleet admits the job under the Governor's
//! envelope, isolates it (fake git so no real repo is touched), launches the
//! kernel run, drives the FSM to a terminal phase, and folds the completion.

use std::sync::Arc;

use hawking_orch::inference::StubInferenceClient;
use hawking_orch::registry::RoleRegistry;
use hawking_orch::router::SimpleRouter;
use hide_core::event::InMemoryEventLog;
use hide_core::persistence::DynEventLog;
use hide_fleet::manager::{FleetConfig, FleetManager, KernelRunLauncher, RunLauncher};
use hide_fleet::queue::{AgentJob, JobStatus, PriorityClass};
use hide_fleet::resources::{FixedResourceProbe, ResourceSnapshot, ThermalState};
use hide_fleet::scheduler::{FleetGovernor, ResourceEnvelope};

fn nominal_snapshot(slots: u32) -> ResourceSnapshot {
    ResourceSnapshot {
        free_memory_mb: 32_000,
        max_generation_slots: slots,
        active_generation_slots: 0,
        thermal: ThermalState::Nominal,
        dec_tps_now: 40.0,
        dec_tps_baseline: 40.0,
        battery_percent: None,
        on_ac_power: true,
        idle: true,
    }
}

/// The minimal real `hide-kernel::AgentKernel` (stub planner, empty oracle
/// suite). The kernel's own tests prove this drives the FSM to `Done` with no
/// model — exactly what we want for a deterministic fleet integration test that
/// nonetheless exercises a REAL kernel run end-to-end.
fn minimal_kernel_launcher(log: DynEventLog) -> Arc<dyn RunLauncher> {
    let kernel = Arc::new(hide_kernel::AgentKernel::new(log));
    Arc::new(KernelRunLauncher::new(kernel).with_max_steps(128))
}

/// A fully-wired kernel backed by a stub inference client + router (no live
/// model). Used to prove the orch runtime seam compiles and runs through the
/// fleet; the scaffolded loop with model-authored plans reaches a terminal phase.
fn stub_runtime_kernel_launcher(log: DynEventLog) -> Arc<dyn RunLauncher> {
    let registry = Arc::new(RoleRegistry::with_default_local_roles());
    let router = Arc::new(SimpleRouter::new(registry));
    let inference = Arc::new(StubInferenceClient::new("investigate the module"));
    let runtime = Arc::new(hide_kernel::runtime_client::KernelRuntimeClient::new(
        router, inference,
    ));
    let kernel = Arc::new(hide_kernel::AgentKernel::builder(log).runtime(runtime).build());
    Arc::new(KernelRunLauncher::new(kernel).with_max_steps(256))
}

#[tokio::test]
async fn fleet_drives_a_real_kernel_run_to_done() {
    let log: DynEventLog = Arc::new(InMemoryEventLog::new());
    let launcher = minimal_kernel_launcher(log.clone());

    let dir = std::env::temp_dir().join(format!("hide_fleet_it_{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&dir).unwrap();
    let manager = FleetManager::new(
        log.clone(),
        FleetGovernor::new(ResourceEnvelope {
            max_model_runs: 2,
            ram_headroom_mb_min: 256,
            ..Default::default()
        }),
        Arc::new(FixedResourceProbe {
            snapshot: nominal_snapshot(2),
        }),
        launcher,
        FleetConfig {
            repo_root: dir.display().to_string(),
            ..Default::default()
        },
    )
    .with_fake_worktrees();

    let job = AgentJob::new("scaffold the parser module", PriorityClass::Normal);
    let job_id = job.id.clone();
    manager.enqueue(job).await.unwrap();

    // One tick admits + launches the kernel run.
    let (plan, _launched) = manager.schedule_tick(2, 40.0, 40.0).await.unwrap();
    assert_eq!(plan.admit, vec![job_id.clone()]);

    // The kernel run drives itself to terminal and reports back; fold it.
    manager.await_completions(1).await.unwrap();

    let folded = manager.queue().get(&job_id).unwrap();
    assert_eq!(
        folded.status,
        JobStatus::Done,
        "the real kernel run should reach Done"
    );
    assert!(folded.run_id.is_some(), "the kernel minted a run id");

    // The kernel emitted its own run events (user.intent + agent.phase) under the
    // same log — proof the fleet drove a real run, not a scripted stub.
    let events = log.scan(None, None, None).await.unwrap();
    assert!(
        events.iter().any(|e| e.kind == "user.intent"),
        "kernel start_run emitted the user intent event"
    );
    assert!(
        events.iter().any(|e| e.kind == "agent.phase"),
        "kernel drove FSM phases"
    );
    // And the fleet's own lifecycle events are interleaved.
    assert!(events.iter().any(|e| e.kind == "job.completed"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn fleet_runs_a_fanout_of_real_kernel_runs() {
    // A small fan-out: 3 independent kernel runs admitted under a model ceiling of
    // 3, all driven to Done by the fleet.
    let log: DynEventLog = Arc::new(InMemoryEventLog::new());
    let launcher = minimal_kernel_launcher(log.clone());

    let dir = std::env::temp_dir().join(format!("hide_fleet_fan_{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&dir).unwrap();
    let manager = FleetManager::new(
        log.clone(),
        FleetGovernor::new(ResourceEnvelope {
            max_model_runs: 3,
            ram_headroom_mb_min: 256,
            ..Default::default()
        }),
        Arc::new(FixedResourceProbe {
            snapshot: nominal_snapshot(3),
        }),
        launcher,
        FleetConfig {
            repo_root: dir.display().to_string(),
            ..Default::default()
        },
    )
    .with_fake_worktrees();

    for i in 0..3 {
        manager
            .enqueue(AgentJob::new(format!("port endpoint {i}"), PriorityClass::Normal))
            .await
            .unwrap();
    }

    // Drive to quiescence: ticks + completion folds until everything is terminal.
    manager.run_to_quiescence(3, 8).await.unwrap();

    let done = manager
        .queue()
        .all()
        .iter()
        .filter(|j| j.status == JobStatus::Done)
        .count();
    assert_eq!(done, 3, "all three real kernel runs reached Done");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn fleet_drives_a_stub_runtime_backed_kernel_to_terminal() {
    // Proves the orch runtime seam: a kernel wired with a StubInferenceClient +
    // SimpleRouter (no live model) runs through the fleet to a terminal job
    // status. The model-authored plan exercises the runtime generate path.
    let log: DynEventLog = Arc::new(InMemoryEventLog::new());
    let launcher = stub_runtime_kernel_launcher(log.clone());

    let dir = std::env::temp_dir().join(format!("hide_fleet_stub_{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&dir).unwrap();
    let manager = FleetManager::new(
        log.clone(),
        FleetGovernor::new(ResourceEnvelope {
            max_model_runs: 1,
            ram_headroom_mb_min: 256,
            ..Default::default()
        }),
        Arc::new(FixedResourceProbe {
            snapshot: nominal_snapshot(1),
        }),
        launcher,
        FleetConfig {
            repo_root: dir.display().to_string(),
            ..Default::default()
        },
    )
    .with_fake_worktrees();

    let job = AgentJob::new("investigate the parser", PriorityClass::Normal);
    let job_id = job.id.clone();
    manager.enqueue(job).await.unwrap();
    let _ = manager.schedule_tick(1, 40.0, 40.0).await.unwrap();
    manager.await_completions(1).await.unwrap();

    let folded = manager.queue().get(&job_id).unwrap();
    assert!(
        folded.status.is_terminal(),
        "stub-runtime kernel run reached a terminal status, got {:?}",
        folded.status
    );
    // The runtime generate path ran: the kernel emitted plan + action events.
    let events = log.scan(None, None, None).await.unwrap();
    assert!(events.iter().any(|e| e.kind == "plan.created"));

    let _ = std::fs::remove_dir_all(&dir);
}
