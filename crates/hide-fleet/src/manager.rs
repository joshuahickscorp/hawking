//! The FleetManager — the ch.09 orchestrator entry (§4.1).
//!
//! `FleetManager` composes the parts: it owns the [`JobGraph`] (queue), the
//! [`FleetGovernor`] (machine-wide admission + circuit-breaker), the
//! [`WorktreeManager`] (isolation), a [`ResourceProbe`], and a [`RunLauncher`]
//! that actually launches a `hide-kernel` run. `schedule_tick` is the ~1 Hz loop:
//! probe → backoff → preempt → admit → `spawn_run`. Admission is bounded;
//! completion-reporting is unbounded (§3.5) via a tokio mpsc — a finished run
//! always reports back, guaranteeing forward progress.
//!
//! This is where the declared-but-unused `hide-kernel` dep becomes load-bearing:
//! [`KernelRunLauncher`] drives `AgentKernel::start_run` + `step` to a terminal
//! phase on an admitted job. Tests use a `StubInferenceClient`-backed kernel (or
//! a scripted launcher) so no live model is needed.

use crate::isolate::{PortAllocator, RunOutcome, RunWorkspace, WorktreeManager};
use crate::queue::{AgentJob, ConcurrencyClass, JobGraph, JobStatus, PriorityClass};
use crate::resources::ResourceProbe;
use crate::scheduler::{FleetGovernor, FleetScheduler, PoolOccupancy, ReadyJob, TickPlan};
use async_trait::async_trait;
use hide_core::event::{EventClass, EventSource, NewEvent};
use hide_core::ids::{now_ms, RunId};
use hide_core::persistence::DynEventLog;
use hide_core::Result;
use serde_json::json;
use std::collections::BTreeSet;
use std::sync::Arc;
use tokio::sync::mpsc;

/// The terminal outcome a launcher reports back for a job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunReport {
    pub job_id: String,
    pub run_id: RunId,
    pub status: JobStatus,
    pub summary: String,
}

/// Launches and drives a single agent run to a terminal phase. The real impl
/// drives `hide-kernel`; tests inject a scripted double.
#[async_trait]
pub trait RunLauncher: Send + Sync {
    /// Run `job` in `workspace` to completion, returning its terminal report.
    /// Implementations MUST be cancellation-cooperative (checkpoint on a flipped
    /// abort) for preemption; the scaffolded kernel runs short, so this is a
    /// drive-to-terminal loop with a step cap.
    async fn launch(&self, job: &AgentJob, workspace: &RunWorkspace) -> RunReport;
}

/// The real launcher: drives a `hide_kernel::AgentKernel` run to a terminal
/// phase. The kernel is constructed by the caller (so tests can inject a
/// `StubInferenceClient`-backed kernel and production the fully-wired one); we
/// only need `start_run` + `step` + the terminal check.
pub struct KernelRunLauncher {
    kernel: Arc<hide_kernel::AgentKernel>,
    max_steps: u32,
}

impl KernelRunLauncher {
    pub fn new(kernel: Arc<hide_kernel::AgentKernel>) -> Self {
        Self {
            kernel,
            max_steps: 256,
        }
    }

    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }
}

#[async_trait]
impl RunLauncher for KernelRunLauncher {
    async fn launch(&self, job: &AgentJob, _workspace: &RunWorkspace) -> RunReport {
        // Drive a real kernel run: start_run emits the intent event + builds the
        // AgentState, then step() advances the FSM one transition at a time until
        // a terminal phase (Done/Aborted) or the step cap.
        match self
            .kernel
            .start_run(job.session_id.clone(), job.spec.objective.clone())
            .await
        {
            Ok(mut state) => {
                let run_id = state.run_id.clone();
                let mut steps = 0;
                while !state.phase.is_terminal() && steps < self.max_steps {
                    if self.kernel.step(&mut state).await.is_err() {
                        return RunReport {
                            job_id: job.id.clone(),
                            run_id,
                            status: JobStatus::Failed,
                            summary: "kernel step error".to_string(),
                        };
                    }
                    steps += 1;
                }
                let status = match state.phase {
                    hide_kernel::machine::state::Phase::Done => JobStatus::Done,
                    hide_kernel::machine::state::Phase::Aborted => JobStatus::Failed,
                    _ => JobStatus::Failed, // hit the step cap without terminating
                };
                RunReport {
                    job_id: job.id.clone(),
                    run_id,
                    status,
                    summary: format!("kernel run reached {:?} in {steps} steps", state.phase),
                }
            }
            Err(e) => RunReport {
                job_id: job.id.clone(),
                run_id: RunId::new(),
                status: JobStatus::Failed,
                summary: format!("start_run failed: {e}"),
            },
        }
    }
}

/// FleetManager configuration.
#[derive(Debug, Clone)]
pub struct FleetConfig {
    pub repo_root: String,
    pub port_pool_start: u16,
    pub port_pool_end: u16,
    /// Ports leased per run (dev-server + DB, typically 2).
    pub ports_per_run: u16,
}

impl Default for FleetConfig {
    fn default() -> Self {
        Self {
            repo_root: ".".to_string(),
            port_pool_start: 4000,
            port_pool_end: 4100,
            ports_per_run: 2,
        }
    }
}

/// A run launched this tick (the job + its isolated workspace). Surfaced by
/// [`FleetManager::schedule_tick`] for observability + tests.
#[derive(Debug, Clone)]
pub struct LaunchedRun {
    pub job_id: String,
    pub workspace: RunWorkspace,
}

/// The orchestrator. Holds shared state behind `Arc` so the tick loop + the
/// completion drain can run concurrently.
pub struct FleetManager {
    log: DynEventLog,
    queue: Arc<JobGraph>,
    governor: parking_lot::Mutex<FleetGovernor>,
    worktrees: Arc<WorktreeManager>,
    probe: Arc<dyn ResourceProbe>,
    launcher: Arc<dyn RunLauncher>,
    config: FleetConfig,
    completions_tx: mpsc::UnboundedSender<RunReport>,
    completions_rx: parking_lot::Mutex<Option<mpsc::UnboundedReceiver<RunReport>>>,
}

impl FleetManager {
    pub fn new(
        log: DynEventLog,
        governor: FleetGovernor,
        probe: Arc<dyn ResourceProbe>,
        launcher: Arc<dyn RunLauncher>,
        config: FleetConfig,
    ) -> Self {
        let worktrees = Arc::new(WorktreeManager::new(
            config.repo_root.clone(),
            PortAllocator::new(config.port_pool_start, config.port_pool_end),
        ));
        // Unbounded completion channel: a finished run always reports back (P7 /
        // §3.5 "unbounded completion-reporting"); only admission is bounded.
        let (completions_tx, completions_rx) = mpsc::unbounded_channel();
        Self {
            log,
            queue: Arc::new(JobGraph::new()),
            governor: parking_lot::Mutex::new(governor),
            worktrees,
            probe,
            launcher,
            config,
            completions_tx,
            completions_rx: parking_lot::Mutex::new(Some(completions_rx)),
        }
    }

    /// Install a worktree manager with a fake git runner (tests) so `schedule_tick`
    /// can launch runs without a real repository.
    pub fn with_fake_worktrees(mut self) -> Self {
        let (mgr, _log) = WorktreeManager::new(
            self.config.repo_root.clone(),
            PortAllocator::new(self.config.port_pool_start, self.config.port_pool_end),
        )
        .with_fake_git();
        self.worktrees = Arc::new(mgr);
        self
    }

    pub fn queue(&self) -> &Arc<JobGraph> {
        &self.queue
    }

    pub fn worktrees(&self) -> &Arc<WorktreeManager> {
        &self.worktrees
    }

    /// Enqueue a job durably (the §4.5 "queue is a projection" path). Emits
    /// `job.enqueued`. Rejects a cyclic graph.
    pub async fn enqueue(&self, job: AgentJob) -> Result<()> {
        self.queue.enqueue_logged(&self.log, job).await
    }

    /// Current pool occupancy across the two pools + isolation leases.
    pub fn occupancy(&self) -> PoolOccupancy {
        PoolOccupancy {
            model_runs: self.queue.live_in_class(ConcurrencyClass::Model),
            cpu_runs: self.queue.live_in_class(ConcurrencyClass::CpuOnly),
            worktrees: self
                .queue
                .all()
                .iter()
                .filter(|j| j.status.is_live())
                .count() as u32,
            ports_leased: self.worktrees_ports_leased(),
        }
    }

    fn worktrees_ports_leased(&self) -> u32 {
        // The worktree manager owns the port allocator; live model runs each hold
        // `ports_per_run`. We approximate from live jobs (the allocator is the
        // source of truth but lives behind the manager's internal lock).
        self.queue
            .all()
            .iter()
            .filter(|j| j.status.is_live() && j.concurrency_class() == ConcurrencyClass::Model)
            .count() as u32
            * self.config.ports_per_run as u32
    }

    /// One scheduler tick (§4.6.2). Probes the machine, applies thermal/RAM
    /// backoff, preempts for a waiting interactive job, admits in priority order,
    /// and launches the admitted runs (spawning a `hide-kernel` run per job). It
    /// also drains any completions that arrived since the last tick.
    ///
    /// Returns the tick plan + the runs launched this tick (for observability /
    /// tests). The launched runs report back over the unbounded completion
    /// channel; call [`FleetManager::drain_completions`] (or `run_to_quiescence`)
    /// to fold them.
    pub async fn schedule_tick(
        &self,
        max_slots: u32,
        runtime_dec_tps_now: f32,
        runtime_dec_tps_baseline: f32,
    ) -> Result<(TickPlan, Vec<LaunchedRun>)> {
        // First, fold any completions that arrived (frees slots before admission).
        self.drain_completions().await?;

        // 0. probe.
        let active = self.queue.live_in_class(ConcurrencyClass::Model);
        let mut snapshot = self.probe.snapshot(max_slots, active).await;
        snapshot.dec_tps_now = runtime_dec_tps_now;
        snapshot.dec_tps_baseline = runtime_dec_tps_baseline;
        {
            let mut gov = self.governor.lock();
            gov.refresh(snapshot);
        }

        // Build the ready set + preemption inputs.
        let ready_jobs = self.queue.ready_jobs();
        let occupancy = self.occupancy();
        let interactive_waiting = self.queue.has_waiting(PriorityClass::Interactive);
        let preempt_floor = self.governor.lock().preempt_floor();
        let victim = self.queue.lowest_priority_live_below(preempt_floor);
        let victim_id = victim.as_ref().map(|j| j.id.clone());

        let ready: Vec<ReadyJob> = ready_jobs
            .iter()
            .map(|j| ReadyJob {
                id: j.id.clone(),
                priority: j.priority,
                request: j.spec.required_resources.clone(),
                status: j.status,
            })
            .collect();

        // Compute the plan (pure).
        let plan = {
            let gov = self.governor.lock();
            FleetScheduler::plan_tick(
                &gov,
                &ready,
                occupancy,
                interactive_waiting,
                victim_id.as_deref(),
            )
        };

        // Surface a breaker banner event if open.
        if plan.breaker_open {
            let reason = self
                .governor
                .lock()
                .breaker()
                .reason
                .clone()
                .unwrap_or_default();
            self.emit_governor_breaker("spawn_rate", &reason).await?;
        }

        // 2. preemption — checkpoint-and-yield (never kill, P8). The scaffolded
        // kernel run is short, so here we mark the victim Preempted + emit the
        // event; a fuller impl flips GenerateRequest.abort at a token boundary.
        for victim_id in &plan.preempt {
            self.queue
                .set_status_logged(
                    &self.log,
                    victim_id,
                    JobStatus::Preempted,
                    json!({ "for": "interactive" }),
                )
                .await?;
        }

        // 3. admit + launch.
        let mut launched = Vec::new();
        for job_id in &plan.admit {
            if let Some(job) = self.queue.get(job_id) {
                match self.admit_and_launch(&job).await {
                    Ok(ws) => launched.push(LaunchedRun {
                        job_id: job_id.clone(),
                        workspace: ws,
                    }),
                    Err(e) => {
                        // Admission/isolation failure → back to queued, log the reason.
                        let _ = self
                            .queue
                            .set_status_logged(
                                &self.log,
                                job_id,
                                JobStatus::Queued,
                                json!({ "admission_error": e.to_string() }),
                            )
                            .await;
                    }
                }
            }
        }
        // launched runs report back over the unbounded completion channel.
        Ok((plan, launched))
    }

    /// Admit a single job: emit `job.admitted`, create its isolated workspace
    /// (`workspace.created`), mark it Running (`job.started`), note the spawn for
    /// the breaker, and launch the run on the executor. The run reports back over
    /// the completion channel.
    async fn admit_and_launch(&self, job: &AgentJob) -> Result<RunWorkspace> {
        self.governor.lock().note_spawn(now_ms());

        self.queue
            .set_status_logged(&self.log, &job.id, JobStatus::Admitted, json!({}))
            .await?;

        // Isolate (worktree + ports + env). Model runs get a worktree; CpuOnly
        // test shards also get one (so a flaky test can't poison siblings, §4.3).
        let n_ports = match job.concurrency_class() {
            ConcurrencyClass::Model => self.config.ports_per_run,
            ConcurrencyClass::CpuOnly => self.config.ports_per_run,
        };
        let workspace = self
            .worktrees
            .isolate_run(&job.id, &job.base_ref, n_ports)
            .await
            .map_err(|e| hide_core::HideError::msg(format!("isolate_run: {e}")))?;
        self.emit_workspace_created(job, &workspace).await?;

        self.queue
            .set_status_logged(&self.log, &job.id, JobStatus::Running, json!({}))
            .await?;

        // Launch the run on a detached task; it reports back over the channel.
        let launcher = self.launcher.clone();
        let tx = self.completions_tx.clone();
        let job_clone = job.clone();
        let ws_clone = workspace.clone();
        tokio::spawn(async move {
            let report = launcher.launch(&job_clone, &ws_clone).await;
            // Send is infallible while the manager lives; drop on shutdown is fine.
            let _ = tx.send(report);
        });

        Ok(workspace)
    }

    /// Fold all pending completion reports: mark jobs terminal (`job.completed`),
    /// release their worktrees (`workspace.released`). Non-blocking — drains what
    /// is currently in the channel. Unbounded so a finished run never blocks (P7).
    pub async fn drain_completions(&self) -> Result<usize> {
        let mut rx = self.completions_rx.lock().take();
        let mut folded = 0;
        if let Some(rx) = rx.as_mut() {
            while let Ok(report) = rx.try_recv() {
                self.fold_completion(report).await?;
                folded += 1;
            }
        }
        *self.completions_rx.lock() = rx;
        Ok(folded)
    }

    /// Block until `pending` more completions have been folded (tests / batch
    /// drains). Returns the number folded.
    pub async fn await_completions(&self, pending: usize) -> Result<usize> {
        let mut rx = self.completions_rx.lock().take();
        let mut folded = 0;
        if let Some(rx) = rx.as_mut() {
            while folded < pending {
                match rx.recv().await {
                    Some(report) => {
                        self.fold_completion(report).await?;
                        folded += 1;
                    }
                    None => break,
                }
            }
        }
        *self.completions_rx.lock() = rx;
        Ok(folded)
    }

    async fn fold_completion(&self, report: RunReport) -> Result<()> {
        self.queue.set_run_id(&report.job_id, report.run_id.clone());
        self.queue
            .set_status_logged(
                &self.log,
                &report.job_id,
                report.status,
                json!({ "run_id": report.run_id, "summary": report.summary }),
            )
            .await?;
        // Release the worktree (adopted vs discarded: a failed run is discarded).
        if let Some(job) = self.queue.get(&report.job_id) {
            let discarded = report.status != JobStatus::Done;
            // Reconstruct a minimal workspace handle for release (run_id keyed).
            let ws = RunWorkspace {
                run_id: job.id.clone(),
                worktree: crate::isolate::WorktreeLease {
                    run_id: job.id.clone(),
                    branch: format!("hide/{}", job.id),
                    path: self.worktrees.worktree_root().join(&job.id),
                    base_ref: job.base_ref.clone(),
                    sandbox: crate::isolate::workspace_sandbox(
                        &self.worktrees.worktree_root().join(&job.id),
                    ),
                },
                ports: crate::isolate::PortLease {
                    run_id: job.id.clone(),
                    ports: Vec::new(),
                },
                env: Default::default(),
            };
            let _ = self
                .worktrees
                .release_run(&ws, RunOutcome { discarded })
                .await;
            self.emit_workspace_released(&job.id, &job.session_id, !discarded)
                .await?;
        }
        Ok(())
    }

    /// Drive ticks until the queue reaches quiescence: no ready jobs and no live
    /// jobs (everything terminal). Bounded by `max_ticks` to avoid a runaway loop
    /// in a misbehaving test. Used by batch drains + tests.
    pub async fn run_to_quiescence(
        &self,
        max_slots: u32,
        max_ticks: usize,
    ) -> Result<()> {
        for _ in 0..max_ticks {
            let _ = self.schedule_tick(max_slots, 40.0, 40.0).await?;
            // Wait for in-flight runs to report, then fold.
            let live = self
                .queue
                .all()
                .iter()
                .filter(|j| j.status.is_live())
                .count();
            if live > 0 {
                self.await_completions(live).await?;
            }
            self.drain_completions().await?;
            let pending = self
                .queue
                .all()
                .iter()
                .any(|j| !j.status.is_terminal());
            if !pending {
                return Ok(());
            }
        }
        Ok(())
    }

    /// GC orphaned worktrees on boot (F9): reap any worktree whose run isn't live.
    pub async fn gc_orphans_on_boot(&self) -> Result<usize> {
        let live: BTreeSet<String> = self
            .queue
            .all()
            .iter()
            .filter(|j| j.status.is_live())
            .map(|j| j.id.clone())
            .collect();
        self.worktrees
            .gc_orphans(&live)
            .await
            .map_err(|e| hide_core::HideError::msg(format!("gc_orphans: {e}")))
    }

    // --- A.5 event emitters ---

    async fn emit_workspace_created(&self, job: &AgentJob, ws: &RunWorkspace) -> Result<()> {
        self.log
            .append(
                NewEvent::of(
                    job.session_id.clone(),
                    EventSource::System,
                    "workspace.created",
                    json!({
                        "run_id": job.id,
                        "path": ws.worktree.path,
                        "ports": ws.ports.ports,
                        "base_ref": ws.worktree.base_ref,
                    }),
                )
                .with_class(EventClass::Action),
            )
            .await?;
        Ok(())
    }

    async fn emit_workspace_released(
        &self,
        run_id: &str,
        session: &hide_core::ids::SessionId,
        kept: bool,
    ) -> Result<()> {
        self.log
            .append(
                NewEvent::of(
                    session.clone(),
                    EventSource::System,
                    "workspace.released",
                    json!({ "run_id": run_id, "kept": kept }),
                )
                .with_class(EventClass::Observation),
            )
            .await?;
        Ok(())
    }

    async fn emit_governor_breaker(&self, trigger: &str, reason: &str) -> Result<()> {
        // Use the first live job's session, else a fresh one (system-scoped).
        let session = self
            .queue
            .all()
            .first()
            .map(|j| j.session_id.clone())
            .unwrap_or_default();
        self.log
            .append(NewEvent::of(
                session,
                EventSource::System,
                "governor.breaker",
                json!({ "trigger": trigger, "action": "pause_new_admissions", "reason": reason }),
            ))
            .await?;
        Ok(())
    }
}

/// A scripted launcher for tests: returns a preset terminal status without a
/// kernel. Lets the manager's scheduling/isolation/event path be exercised
/// without driving a real run.
pub struct ScriptedLauncher {
    pub status: JobStatus,
    pub summary: String,
}

impl Default for ScriptedLauncher {
    fn default() -> Self {
        Self {
            status: JobStatus::Done,
            summary: "scripted ok".to_string(),
        }
    }
}

#[async_trait]
impl RunLauncher for ScriptedLauncher {
    async fn launch(&self, job: &AgentJob, _workspace: &RunWorkspace) -> RunReport {
        RunReport {
            job_id: job.id.clone(),
            run_id: RunId::new(),
            status: self.status,
            summary: self.summary.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::ResourceRequest;
    use crate::resources::{FixedResourceProbe, ResourceSnapshot, ThermalState};
    use crate::scheduler::ResourceEnvelope;
    use hide_core::event::InMemoryEventLog;

    fn manager_with(launcher: Arc<dyn RunLauncher>, max_model: u32) -> FleetManager {
        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        let envelope = ResourceEnvelope {
            max_model_runs: max_model,
            ram_headroom_mb_min: 256,
            max_cpu_runs: 8,
            ..Default::default()
        };
        let snap = ResourceSnapshot {
            free_memory_mb: 32_000,
            max_generation_slots: max_model,
            active_generation_slots: 0,
            thermal: ThermalState::Nominal,
            dec_tps_now: 40.0,
            dec_tps_baseline: 40.0,
            battery_percent: None,
            on_ac_power: true,
            idle: true,
        };
        let probe = Arc::new(FixedResourceProbe { snapshot: snap });
        let dir = std::env::temp_dir().join(format!("hide_fleet_mgr_{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = FleetConfig {
            repo_root: dir.display().to_string(),
            ..Default::default()
        };
        FleetManager::new(log, FleetGovernor::new(envelope), probe, launcher, config)
            .with_fake_worktrees()
    }

    #[tokio::test]
    async fn schedule_tick_admits_and_completes_a_run() {
        let mgr = manager_with(Arc::new(ScriptedLauncher::default()), 2);
        let job = AgentJob::new("do the thing", PriorityClass::Normal);
        let id = job.id.clone();
        mgr.enqueue(job).await.unwrap();

        let (plan, _launched) = mgr.schedule_tick(2, 40.0, 40.0).await.unwrap();
        assert_eq!(plan.admit, vec![id.clone()]);
        // The run reports back; fold it.
        mgr.await_completions(1).await.unwrap();
        let folded = mgr.queue().get(&id).unwrap();
        assert_eq!(folded.status, JobStatus::Done);
        assert!(folded.run_id.is_some());
    }

    #[tokio::test]
    async fn emits_a5_events_through_the_lifecycle() {
        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        let envelope = ResourceEnvelope {
            max_model_runs: 1,
            ram_headroom_mb_min: 128,
            ..Default::default()
        };
        let snap = ResourceSnapshot {
            free_memory_mb: 16_000,
            max_generation_slots: 1,
            active_generation_slots: 0,
            thermal: ThermalState::Nominal,
            dec_tps_now: 40.0,
            dec_tps_baseline: 40.0,
            battery_percent: None,
            on_ac_power: true,
            idle: true,
        };
        let dir = std::env::temp_dir().join(format!("hide_fleet_ev_{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let mgr = FleetManager::new(
            log.clone(),
            FleetGovernor::new(envelope),
            Arc::new(FixedResourceProbe { snapshot: snap }),
            Arc::new(ScriptedLauncher::default()),
            FleetConfig {
                repo_root: dir.display().to_string(),
                ..Default::default()
            },
        )
        .with_fake_worktrees();

        let job = AgentJob::new("emit events", PriorityClass::Normal);
        mgr.enqueue(job).await.unwrap();
        mgr.schedule_tick(1, 40.0, 40.0).await.unwrap();
        mgr.await_completions(1).await.unwrap();

        let kinds: Vec<String> = log
            .scan(None, None, None)
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.kind)
            .collect();
        for expected in [
            "job.enqueued",
            "job.admitted",
            "workspace.created",
            "job.started",
            "job.completed",
            "workspace.released",
        ] {
            assert!(
                kinds.iter().any(|k| k == expected),
                "missing event {expected}; got {kinds:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn model_pool_ceiling_bounds_concurrent_admissions() {
        // max_model_runs = 1: enqueue 2 jobs; only one is admitted per tick.
        let mgr = manager_with(Arc::new(ScriptedLauncher::default()), 1);
        for i in 0..2 {
            mgr.enqueue(AgentJob::new(format!("job {i}"), PriorityClass::Normal))
                .await
                .unwrap();
        }
        let (plan, _launched) = mgr.schedule_tick(1, 40.0, 40.0).await.unwrap();
        assert_eq!(plan.admit.len(), 1, "model pool ceiling = 1");
        assert_eq!(plan.deferred.len(), 1);
    }

    #[tokio::test]
    async fn cpu_only_jobs_run_wider_than_model_pool() {
        // max_model_runs = 1 but max_cpu_runs = 8: 3 CpuOnly shards all admit.
        let mgr = manager_with(Arc::new(ScriptedLauncher::default()), 1);
        for i in 0..3 {
            let job = AgentJob::new(format!("shard {i}"), PriorityClass::Normal)
                .with_resources(ResourceRequest::cpu_only(256));
            mgr.enqueue(job).await.unwrap();
        }
        let (plan, _launched) = mgr.schedule_tick(1, 40.0, 40.0).await.unwrap();
        assert_eq!(plan.admit.len(), 3, "cpu pool admits all three shards");
    }
}
