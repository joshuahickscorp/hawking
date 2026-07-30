use hawking_orch::inference::StubInferenceClient;
use hawking_orch::registry::RoleRegistry;
use hawking_orch::router::SimpleRouter;
use hide_core::event::InMemoryEventLog;
use hide_core::persistence::DynEventLog;
use hide_fleet::manager::{FleetConfig, FleetManager, KernelRunLauncher, RunLauncher};
use hide_fleet::queue::{AgentJob, JobStatus, PriorityClass};
use hide_fleet::resources::{FixedResourceProbe, ResourceSnapshot, ThermalState};
use hide_fleet::scheduler::{FleetGovernor, ResourceEnvelope};
use std::sync::Arc;
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
fn minimal_kernel_launcher(log: DynEventLog) -> Arc<dyn RunLauncher> {
    let kernel = Arc::new(hide_kernel::AgentKernel::new(log));
    Arc::new(KernelRunLauncher::new(kernel).with_max_steps(128))
}
fn stub_runtime_kernel_launcher(log: DynEventLog) -> Arc<dyn RunLauncher> {
    let registry = Arc::new(RoleRegistry::with_default_local_roles());
    let router = Arc::new(SimpleRouter::new(registry));
    let inference = Arc::new(StubInferenceClient::new("investigate the module"));
    let runtime = Arc::new(hide_kernel::runtime_client::KernelRuntimeClient::new(
        router, inference,
    ));
    let kernel = Arc::new(
        hide_kernel::AgentKernel::builder(log)
            .runtime(runtime)
            .build(),
    );
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
    let (plan, _launched) = manager.schedule_tick(2, 40.0, 40.0).await.unwrap();
    assert_eq!(plan.admit, vec![job_id.clone()]);
    manager.await_completions(1).await.unwrap();
    let folded = manager.queue().get(&job_id).unwrap();
    assert_eq!(
        folded.status,
        JobStatus::Done,
        "the real kernel run should reach Done"
    );
    assert!(folded.run_id.is_some(), "the kernel minted a run id");
    let events = log.scan(None, None, None).await.unwrap();
    assert!(events.iter().any(|e| e.kind == "user.intent"));
    assert!(
        events.iter().any(|e| e.kind == "agent.phase"),
        "kernel drove FSM phases"
    );
    assert!(events.iter().any(|e| e.kind == "job.completed"));
    let _ = std::fs::remove_dir_all(&dir);
}
#[tokio::test]
async fn fleet_runs_a_fanout_of_real_kernel_runs() {
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
            .enqueue(AgentJob::new(
                format!("port endpoint {i}"),
                PriorityClass::Normal,
            ))
            .await
            .unwrap();
    }
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
    assert!(folded.status.is_terminal());
    let events = log.scan(None, None, None).await.unwrap();
    assert!(events.iter().any(|e| e.kind == "plan.created"));
    let _ = std::fs::remove_dir_all(&dir);
}
