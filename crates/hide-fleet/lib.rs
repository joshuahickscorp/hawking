//! Parallel-agent and workstation backend.
//!
//! This crate implements the headless fabric described in HIDE bible chapter
//! 09: job queues, machine-wide resource admission, isolation leases, merge
//! selection, batch reports, and remote protocol contracts.

#[rustfmt::skip]
pub mod batch {
    //! Overnight / batch jobs (bible ch.09 §4.7).
    //!
    //! A `BatchJob` is a job DAG with a `schedule` gate and a wake report. The
    //! flagship local superpower: start a swarm at midnight, wake to a report (P1/P8/
    //! P12). The batch fires only when ALL its gate conditions hold (idle, AC power,
    //! thermal_ok, cron window) — designed around the Apple-Silicon thermal reality
    //! so a laptop runs batches only plugged in, idle, and cool (§4.7.2).
    //!
    //! The wake report (A.3) is a **projection over the batch's events** — every line
    //! is reconstructable from the log (P12). `assemble_wake_report` folds the
    //! `job.*` events of a batch's members into the structured report.

    use crate::queue::ScheduleGate;
    use crate::resources::ResourceSnapshot;
    use hide_core::event::Event;
    use hide_core::ids::now_ms;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    /// A scheduled batch of goals (a job DAG + a gate).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BatchJob {
        pub id: String,
        /// Member job ids (the batch's DAG).
        pub job_ids: Vec<String>,
        pub schedule: BatchSchedule,
        pub report_on_wake: bool,
    }

    impl BatchJob {
        pub fn new(job_ids: Vec<String>, schedule: BatchSchedule) -> Self {
            Self { id: format!("batch_{}", ulid::Ulid::new()), job_ids, schedule, report_on_wake: true }
        }
    }

    /// The schedule gate (§4.7.2). The batch fires only when all conditions hold.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BatchSchedule {
        /// Wall-clock ms since epoch; the batch can't start before this.
        pub earliest_start_ms: Option<u64>,
        /// Latest start of a cron window (e.g. 06:00). `None` = no upper bound.
        pub window_end_ms: Option<u64>,
        pub gates: Vec<ScheduleGate>,
    }

    impl Default for BatchSchedule {
        fn default() -> Self {
            Self {
                earliest_start_ms: None,
                window_end_ms: None,
                gates: vec![ScheduleGate::Idle, ScheduleGate::AcPower],
            }
        }
    }

    /// Why a batch is not firing yet (so the UI can show "waiting on AC power").
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum GateStatus {
        Ready,
        Blocked { reasons: Vec<String> },
    }

    impl GateStatus {
        pub fn is_ready(&self) -> bool {
            matches!(self, GateStatus::Ready)
        }
    }

    /// Evaluate a batch's schedule gate against the clock + a resource snapshot
    /// (§4.7.2). All gates must pass. Real logic: time window, idle, AC power, and a
    /// thermal_ok check against the thermal proxy.
    pub fn evaluate_gate(schedule: &BatchSchedule, now: u64, snapshot: &ResourceSnapshot) -> GateStatus {
        let mut reasons = Vec::new();
        if let Some(earliest) = schedule.earliest_start_ms {
            if now < earliest {
                reasons.push(format!("before earliest start ({} > now {})", earliest, now));
            }
        }
        if let Some(end) = schedule.window_end_ms {
            if now > end {
                reasons.push("past the cron window".to_string());
            }
        }
        for gate in &schedule.gates {
            match gate {
                ScheduleGate::Idle => {
                    if !snapshot.idle {
                        reasons.push("machine not idle (a session is active)".to_string());
                    }
                }
                ScheduleGate::AcPower => {
                    if !snapshot.on_ac_power {
                        reasons.push("not on AC power".to_string());
                    }
                }
                ScheduleGate::ThermalOk => {
                    use crate::resources::ThermalState;
                    if snapshot.thermal >= ThermalState::Serious {
                        reasons.push(format!("thermal not nominal ({:?})", snapshot.thermal));
                    }
                }
                ScheduleGate::Cron => { /* the time-window checks above cover cron */ }
            }
        }
        if reasons.is_empty() {
            GateStatus::Ready
        } else {
            GateStatus::Blocked { reasons }
        }
    }

    /// The wake report (A.3) — a projection over the batch's events.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct WakeReport {
        pub batch_id: String,
        pub ran_from_ms: Option<u64>,
        pub ran_to_ms: Option<u64>,
        pub summary: WakeSummary,
        pub results: Vec<WakeResult>,
        pub needs_review: Vec<String>,
        pub thermal_events: u32,
        pub total_runs: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    pub struct WakeSummary {
        pub goals: u32,
        pub succeeded: u32,
        pub partial: u32,
        pub failed: u32,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct WakeResult {
        pub job_id: String,
        pub status: String,
        pub outcome: String,
    }

    /// Assemble the wake report by folding the batch's events (P12). Counts member
    /// outcomes from `job.completed`, tracks the run window from event timestamps,
    /// and counts `governor.backoff{reason:thermal}` as thermal events.
    pub fn assemble_wake_report(batch: &BatchJob, events: &[Event]) -> WakeReport {
        let member_set: std::collections::BTreeSet<&str> = batch.job_ids.iter().map(String::as_str).collect();

        let mut results: BTreeMap<String, WakeResult> = BTreeMap::new();
        let mut summary = WakeSummary::default();
        let mut thermal_events = 0u32;
        let mut total_runs = 0u32;
        let mut ran_from: Option<u64> = None;
        let mut ran_to: Option<u64> = None;

        for event in events {
            match event.kind.as_str() {
                "job.started" => {
                    if let Some(job_id) = event.payload.get("job_id").and_then(|v| v.as_str()) {
                        if member_set.contains(job_id) {
                            total_runs += 1;
                            ran_from = Some(ran_from.map_or(event.ts, |t| t.min(event.ts)));
                        }
                    }
                }
                "job.completed" => {
                    if let Some(job_id) = event.payload.get("job_id").and_then(|v| v.as_str()) {
                        if member_set.contains(job_id) {
                            ran_to = Some(ran_to.map_or(event.ts, |t| t.max(event.ts)));
                            let status = event.payload.get("status").and_then(|v| v.as_str()).unwrap_or("failed");
                            let outcome =
                                event.payload.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            match status {
                                "done" => summary.succeeded += 1,
                                "failed" | "cancelled" => summary.failed += 1,
                                _ => summary.partial += 1,
                            }
                            results.insert(
                                job_id.to_string(),
                                WakeResult { job_id: job_id.to_string(), status: status.to_string(), outcome },
                            );
                        }
                    }
                }
                "governor.backoff"
                    if event.payload.get("reason").and_then(|v| v.as_str()) == Some("thermal") =>
                {
                    thermal_events += 1;
                }
                _ => {}
            }
        }

        summary.goals = batch.job_ids.len() as u32;
        // Members that completed-but-not-done are queued for review.
        let needs_review: Vec<String> =
            results.values().filter(|r| r.status != "done").map(|r| r.job_id.clone()).collect();

        WakeReport {
            batch_id: batch.id.clone(),
            ran_from_ms: ran_from,
            ran_to_ms: ran_to.or_else(|| Some(now_ms())),
            summary,
            results: results.into_values().collect(),
            needs_review,
            thermal_events,
            total_runs,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::resources::{ResourceSnapshot, ThermalState};
        use hide_core::event::{EventLog, InMemoryEventLog, NewEvent};
        use hide_core::ids::SessionId;
        use serde_json::json;

        fn snap(idle: bool, ac: bool, thermal: ThermalState) -> ResourceSnapshot {
            ResourceSnapshot {
                free_memory_mb: 16_000,
                max_generation_slots: 4,
                active_generation_slots: 0,
                thermal,
                dec_tps_now: 40.0,
                dec_tps_baseline: 40.0,
                battery_percent: None,
                on_ac_power: ac,
                idle,
            }
        }

        #[test]
        fn gate_blocks_on_battery_and_active_session() {
            let sched = BatchSchedule::default();
            let g = evaluate_gate(&sched, now_ms(), &snap(false, false, ThermalState::Nominal));
            match g {
                GateStatus::Blocked { reasons } => {
                    assert!(reasons.iter().any(|r| r.contains("not idle")));
                    assert!(reasons.iter().any(|r| r.contains("AC power")));
                }
                _ => panic!("expected blocked"),
            }
        }

        #[test]
        fn gate_ready_when_idle_and_plugged_in() {
            let sched = BatchSchedule::default();
            let g = evaluate_gate(&sched, now_ms(), &snap(true, true, ThermalState::Nominal));
            assert!(g.is_ready());
        }

        #[test]
        fn gate_respects_thermal_ok() {
            let sched = BatchSchedule { gates: vec![ScheduleGate::ThermalOk], ..Default::default() };
            let blocked = evaluate_gate(&sched, now_ms(), &snap(true, true, ThermalState::Serious));
            assert!(!blocked.is_ready());
            let ready = evaluate_gate(&sched, now_ms(), &snap(true, true, ThermalState::Fair));
            assert!(ready.is_ready());
        }

        #[tokio::test]
        async fn wake_report_is_a_projection_of_batch_events() {
            let log = InMemoryEventLog::new();
            let session = SessionId::new();
            // Two member jobs: one done, one failed.
            for (job_id, status) in [("job_a", "done"), ("job_b", "failed")] {
                log.append(NewEvent::system(session.clone(), "job.started", json!({ "job_id": job_id })))
                    .await
                    .unwrap();
                log.append(NewEvent::system(
                    session.clone(),
                    "job.completed",
                    json!({ "job_id": job_id, "status": status, "summary": "ran" }),
                ))
                .await
                .unwrap();
            }
            log.append(NewEvent::system(session.clone(), "governor.backoff", json!({ "reason": "thermal" })))
                .await
                .unwrap();

            let batch = BatchJob::new(vec!["job_a".to_string(), "job_b".to_string()], BatchSchedule::default());
            let events = log.scan(None, None, None).await.unwrap();
            let report = assemble_wake_report(&batch, &events);
            assert_eq!(report.summary.goals, 2);
            assert_eq!(report.summary.succeeded, 1);
            assert_eq!(report.summary.failed, 1);
            assert_eq!(report.total_runs, 2);
            assert_eq!(report.thermal_events, 1);
            assert_eq!(report.needs_review, vec!["job_b".to_string()]);
        }
    }
}
#[rustfmt::skip]
pub mod fleetview {
    //! The live fleet view (bible ch.09 §5 / §4.1 `fleetview.rs`).
    //!
    //! Observability is mandatory for swarms (P12): a 30-agent overnight run is
    //! unmanageable without a live fleet view + per-run resource/outcome accounting.
    //! The `FleetView` is a **projection** — built purely by folding the event log
    //! (`job.*`/`workspace.*`/`governor.*`), so it is reconstructable, replay-safe,
    //! and never an authoritative store. ch.03 renders it as a panel.

    use crate::scheduler::FleetGovernor;
    use hide_core::event::Event;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    /// A single run's row in the fleet view.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RunRow {
        pub job_id: String,
        pub status: String,
        pub run_id: Option<String>,
        /// Worktree path (once isolated).
        pub workspace: Option<String>,
        /// Ports leased to this run.
        pub ports: Vec<u16>,
        pub started_ts: Option<u64>,
        pub finished_ts: Option<u64>,
        pub outcome: Option<String>,
    }

    /// Machine-wide live counters + governor banner.
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct FleetView {
        pub rows: Vec<RunRow>,
        pub admitted: u32,
        pub running: u32,
        pub merging: u32,
        pub preempted: u32,
        pub done: u32,
        pub failed: u32,
        /// Open-circuit-breaker banner (None when nominal).
        pub breaker_banner: Option<String>,
        /// Governor backoff banner (None when nominal).
        pub backoff_banner: Option<String>,
        pub spawn_ewma_per_min: f32,
    }

    impl FleetView {
        /// Project the fleet view from the event log (P12). A pure fold over the
        /// `job.*`/`workspace.*`/`governor.*` events.
        pub fn project(events: &[Event]) -> Self {
            let mut rows: BTreeMap<String, RunRow> = BTreeMap::new();
            let mut view = FleetView::default();

            for event in events {
                match event.kind.as_str() {
                    "job.enqueued" => {
                        if let Some(job_id) = str_field(event, "job_id") {
                            rows.entry(job_id.clone()).or_insert_with(|| RunRow {
                                job_id,
                                status: "queued".to_string(),
                                run_id: None,
                                workspace: None,
                                ports: Vec::new(),
                                started_ts: None,
                                finished_ts: None,
                                outcome: None,
                            });
                        }
                    }
                    "job.admitted" | "job.started" | "job.completed" | "job.preempted" | "job.merging"
                    | "job.paused" | "job.requeued" => {
                        if let Some(job_id) = str_field(event, "job_id") {
                            let row = rows.entry(job_id.clone()).or_insert_with(|| RunRow {
                                job_id: job_id.clone(),
                                status: "queued".to_string(),
                                run_id: None,
                                workspace: None,
                                ports: Vec::new(),
                                started_ts: None,
                                finished_ts: None,
                                outcome: None,
                            });
                            if let Some(status) = str_field(event, "status") {
                                row.status = status;
                            }
                            if event.kind == "job.started" {
                                row.started_ts = Some(event.ts);
                            }
                            if event.kind == "job.completed" {
                                row.finished_ts = Some(event.ts);
                                row.outcome = str_field(event, "summary");
                                row.run_id = str_field(event, "run_id");
                            }
                        }
                    }
                    "workspace.created" => {
                        if let Some(run_id) = str_field(event, "run_id") {
                            let row = rows.entry(run_id.clone()).or_insert_with(|| RunRow {
                                job_id: run_id.clone(),
                                status: "running".to_string(),
                                run_id: None,
                                workspace: None,
                                ports: Vec::new(),
                                started_ts: None,
                                finished_ts: None,
                                outcome: None,
                            });
                            row.workspace = event.payload.get("path").and_then(|v| v.as_str()).map(String::from);
                            row.ports = event
                                .payload
                                .get("ports")
                                .and_then(|v| v.as_array())
                                .map(|a| a.iter().filter_map(|p| p.as_u64().map(|n| n as u16)).collect())
                                .unwrap_or_default();
                        }
                    }
                    "governor.breaker" => {
                        view.breaker_banner = str_field(event, "reason").or_else(|| {
                            Some(format!("circuit breaker: {}", str_field(event, "trigger").unwrap_or_default()))
                        });
                    }
                    "governor.backoff" => {
                        view.backoff_banner = Some(format!(
                            "backoff ({}): ceiling {}",
                            str_field(event, "reason").unwrap_or_default(),
                            event.payload.get("new_ceiling").map(|v| v.to_string()).unwrap_or_default()
                        ));
                    }
                    _ => {}
                }
            }

            view.rows = rows.into_values().collect();
            for row in &view.rows {
                match row.status.as_str() {
                    "admitted" => view.admitted += 1,
                    "running" => view.running += 1,
                    "merging" => view.merging += 1,
                    "preempted" => view.preempted += 1,
                    "done" => view.done += 1,
                    "failed" | "cancelled" => view.failed += 1,
                    _ => {}
                }
            }
            view
        }

        /// Fold live governor telemetry (spawn-rate EWMA, breaker) onto a projected
        /// view — the parts that aren't in the log but are live machine state.
        pub fn with_governor(mut self, gov: &FleetGovernor) -> Self {
            self.spawn_ewma_per_min = gov.breaker().spawn_ewma_per_min;
            if gov.breaker().tripped {
                self.breaker_banner = gov.breaker().reason.clone();
            }
            self
        }
    }

    fn str_field(event: &Event, key: &str) -> Option<String> {
        event.payload.get(key).and_then(|v| v.as_str()).map(String::from)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::event::{EventLog, InMemoryEventLog, NewEvent};
        use hide_core::ids::SessionId;
        use serde_json::json;

        #[tokio::test]
        async fn fleet_view_projects_run_rows_and_counters() {
            let log = InMemoryEventLog::new();
            let s = SessionId::new();
            log.append(NewEvent::system(s.clone(), "job.enqueued", json!({ "job_id": "j1" }))).await.unwrap();
            log.append(NewEvent::system(s.clone(), "job.started", json!({ "job_id": "j1", "status": "running" })))
                .await
                .unwrap();
            log.append(NewEvent::system(
                s.clone(),
                "workspace.created",
                json!({ "run_id": "j1", "path": "/wt/j1", "ports": [4000, 4001] }),
            ))
            .await
            .unwrap();
            log.append(NewEvent::system(
                s.clone(),
                "job.completed",
                json!({ "job_id": "j1", "status": "done", "summary": "ok", "run_id": "run_x" }),
            ))
            .await
            .unwrap();

            let events = log.scan(None, None, None).await.unwrap();
            let view = FleetView::project(&events);
            assert_eq!(view.rows.len(), 1);
            let row = &view.rows[0];
            assert_eq!(row.status, "done");
            assert_eq!(row.workspace.as_deref(), Some("/wt/j1"));
            assert_eq!(row.ports, vec![4000, 4001]);
            assert_eq!(row.run_id.as_deref(), Some("run_x"));
            assert_eq!(view.done, 1);
        }

        #[tokio::test]
        async fn breaker_event_surfaces_a_banner() {
            let log = InMemoryEventLog::new();
            let s = SessionId::new();
            log.append(NewEvent::system(
                s,
                "governor.breaker",
                json!({ "trigger": "spawn_rate", "reason": "spawn rate 99/min exceeds 30/min" }),
            ))
            .await
            .unwrap();
            let events = log.scan(None, None, None).await.unwrap();
            let view = FleetView::project(&events);
            assert!(view.breaker_banner.as_deref().unwrap().contains("spawn rate"));
        }
    }
}
#[rustfmt::skip]
pub mod isolate {
    //! The isolation model (bible ch.09 §4.3).
    //!
    //! Parallel work is isolated at four levels; worktrees alone are insufficient
    //! (the §3.4 runtime-isolation gap). This module owns the *orchestration*
    //! lifecycle — `isolate_run` creates a git worktree off the shared `.git`, leases
    //! a disjoint port range, and seeds a per-run env namespace; `release_run`
    //! removes/prunes the worktree and returns the ports to the pool. ch.10 owns the
    //! sandbox *enforcement* (the `SandboxProfile` boundary); we reference it.
    //!
    //! Worktrees over full clones or containers (§4.3 rationale): worktrees share the
    //! object store (cheap), give true file isolation, and leave unified RAM for the
    //! model — containers would compete with the runtime for RAM, the worst trade on
    //! an Apple-Silicon box.

    use hide_core::security::{NetworkPolicy, SandboxProfile, SandboxTier};
    use parking_lot::Mutex;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tokio::process::Command;

    /// A leased git worktree for one run (§4.3 filesystem level).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct WorktreeLease {
        pub run_id: String,
        pub branch: String,
        pub path: PathBuf,
        pub base_ref: String,
        pub sandbox: SandboxProfile,
    }

    /// A disjoint port range leased to a run so dev-servers/test-DBs in different
    /// runs never collide on 3000/5432/8080 (the named §3.4 gap).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PortLease {
        pub run_id: String,
        pub ports: Vec<u16>,
    }

    /// The full workspace handed to a run: tree + ports + env namespace + sandbox.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RunWorkspace {
        pub run_id: String,
        pub worktree: WorktreeLease,
        pub ports: PortLease,
        /// Per-run env (TMPDIR, build caches, DB/schema names, PORT, DATABASE_URL).
        pub env: BTreeMap<String, String>,
    }

    /// Outcome of a run, deciding whether its worktree is kept (merged) or discarded
    /// (tournament loser / speculative discard).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RunOutcome {
        pub discarded: bool,
    }

    /// Disjoint port-range allocator (§4.3 ports level). Hands each run `count`
    /// contiguous-from-pool ports; releases them on run end. Tracks leases by
    /// `run_id` so occupancy reads are the allocator's truth, not an estimate.
    #[derive(Debug, Clone, Default)]
    pub struct PortAllocator {
        start: u16,
        end: u16,
        leased: BTreeSet<u16>,
        /// run_id → the ports that run currently holds. The authority for both
        /// `leased_count()` (ports) and `leased_runs()` (distinct runs).
        run_ports: BTreeMap<String, Vec<u16>>,
    }

    impl PortAllocator {
        pub fn new(start: u16, end: u16) -> Self {
            Self { start, end, leased: BTreeSet::new(), run_ports: BTreeMap::new() }
        }

        pub fn lease(&mut self, run_id: impl Into<String>, count: u16) -> Option<PortLease> {
            let mut ports = Vec::new();
            for port in self.start..=self.end {
                if !self.leased.contains(&port) {
                    ports.push(port);
                    if ports.len() == count as usize {
                        break;
                    }
                }
            }
            if ports.len() != count as usize {
                return None;
            }
            for port in &ports {
                self.leased.insert(*port);
            }
            let run_id = run_id.into();
            self.run_ports.entry(run_id.clone()).or_default().extend(ports.iter().copied());
            Some(PortLease { run_id, ports })
        }

        pub fn release(&mut self, lease: &PortLease) {
            for port in &lease.ports {
                self.leased.remove(port);
            }
            // Drop the released ports from the run's ledger; forget the run once it
            // holds none. Tolerant of a partial/empty lease (idempotent release).
            if let Some(held) = self.run_ports.get_mut(&lease.run_id) {
                held.retain(|p| !lease.ports.contains(p));
                if held.is_empty() {
                    self.run_ports.remove(&lease.run_id);
                }
            }
        }

        /// The number of ports currently leased (the pool's true occupancy).
        pub fn leased_count(&self) -> usize {
            self.leased.len()
        }

        /// The number of distinct runs currently holding any leased port.
        pub fn leased_runs(&self) -> usize {
            self.run_ports.len()
        }
    }

    /// Errors from the isolation lifecycle.
    #[derive(Debug, thiserror::Error)]
    pub enum IsolateError {
        #[error("git worktree {op} failed (status {code:?}): {stderr}")]
        Git { op: &'static str, code: Option<i32>, stderr: String },
        #[error("port pool exhausted: could not lease {requested} ports")]
        PortsExhausted { requested: u16 },
        #[error("io error: {0}")]
        Io(#[from] std::io::Error),
    }

    /// The worktree manager: owns the repo root, the `.hide/wt/` root, and the port
    /// pool. Creating one does no git work; `isolate_run` does.
    pub struct WorktreeManager {
        repo_root: PathBuf,
        worktree_root: PathBuf,
        ports: Arc<Mutex<PortAllocator>>,
        /// Pluggable git runner (real `git` by default; a closure double in tests).
        git: GitRunner,
    }

    /// How git commands are executed. The default shells out to the system `git`;
    /// tests inject a fake that records invocations without touching a real repo.
    #[derive(Clone)]
    pub enum GitRunner {
        /// Real `git` via `tokio::process::Command` in `repo_root`.
        System,
        /// Test double: records args, returns success. The recorded log lets tests
        /// assert the exact worktree commands without a real repository.
        Fake(Arc<Mutex<Vec<Vec<String>>>>),
    }

    impl WorktreeManager {
        pub fn new(repo_root: impl Into<PathBuf>, ports: PortAllocator) -> Self {
            let repo_root = repo_root.into();
            let worktree_root = repo_root.join(".hide").join("wt");
            Self { repo_root, worktree_root, ports: Arc::new(Mutex::new(ports)), git: GitRunner::System }
        }

        /// Install a fake git runner (tests). Returns the shared invocation log.
        pub fn with_fake_git(mut self) -> (Self, Arc<Mutex<Vec<Vec<String>>>>) {
            let log = Arc::new(Mutex::new(Vec::new()));
            self.git = GitRunner::Fake(log.clone());
            (self, log)
        }

        pub fn worktree_root(&self) -> &Path {
            &self.worktree_root
        }

        /// The number of ports currently leased from the pool (the allocator's
        /// truth). The Governor reconciles its occupancy estimate against this so a
        /// release leak can't silently shrink the pool (§4.3 ports level).
        pub fn ports_leased_count(&self) -> usize {
            self.ports.lock().leased_count()
        }

        /// The number of distinct runs holding live worktree/port leases right now
        /// (the allocator's truth, keyed by `run_id`). Used to reconcile the
        /// Governor's worktree occupancy against reality rather than estimating from
        /// the job projection.
        pub fn live_worktree_count(&self) -> usize {
            self.ports.lock().leased_runs()
        }

        /// Create an isolated workspace for `run_id` (§4.3 lifecycle). Steps:
        /// 1. `git worktree add -b hide/<run> .hide/wt/<run> <base_ref>` (own dir, shared .git)
        /// 2. lease a disjoint port range
        /// 3. seed the env namespace (TMPDIR, caches, DB names, PORT, DATABASE_URL)
        /// 4. build the ch.10 sandbox profile scoped to the worktree
        ///
        /// The caller is responsible for emitting `workspace.created` (the manager is
        /// event-log-agnostic so it stays unit-testable; `FleetManager` wires events).
        pub async fn isolate_run(
            &self,
            run_id: &str,
            base_ref: &str,
            n_ports: u16,
        ) -> Result<RunWorkspace, IsolateError> {
            let rel = format!(".hide/wt/{run_id}");
            let path = self.repo_root.join(&rel);
            let branch = format!("hide/{run_id}");

            // 1. worktree add (creates the directory + a fresh branch off base_ref).
            std::fs::create_dir_all(&self.worktree_root)?;
            self.run_git("add", &["worktree", "add", "-b", &branch, &rel, base_ref]).await?;

            // 2. ports.
            let ports =
                self.ports.lock().lease(run_id, n_ports).ok_or(IsolateError::PortsExhausted { requested: n_ports })?;

            // 3. env namespace.
            let env = env_seed(run_id, &path, &ports);

            // 4. sandbox scoped to the worktree (ch.10 enforces; we shape).
            let sandbox = workspace_sandbox(&path);

            Ok(RunWorkspace {
                run_id: run_id.to_string(),
                worktree: WorktreeLease {
                    run_id: run_id.to_string(),
                    branch,
                    path,
                    base_ref: base_ref.to_string(),
                    sandbox,
                },
                ports,
                env,
            })
        }

        /// Release a workspace (§4.3 `release_run`): return its ports and remove or
        /// prune its worktree. Discarded runs (tournament losers / speculative
        /// discards) are force-removed; adopted runs are pruned after merge.
        pub async fn release_run(&self, ws: &RunWorkspace, outcome: RunOutcome) -> Result<(), IsolateError> {
            self.ports.lock().release(&ws.ports);
            let rel = format!(".hide/wt/{}", ws.run_id);
            if outcome.discarded {
                // Force-remove the loser's tree (it has uncommitted work we discard).
                self.run_git("remove", &["worktree", "remove", "--force", &rel]).await?;
            } else {
                // Adopted: the branch was merged; remove the now-redundant tree, then
                // prune dangling administrative files.
                let _ = self.run_git("remove", &["worktree", "remove", "--force", &rel]).await;
            }
            self.run_git("prune", &["worktree", "prune"]).await?;
            Ok(())
        }

        /// List live worktrees under management (`git worktree list --porcelain`,
        /// filtered to our `.hide/wt/` root). Used for GC of orphans (F9).
        pub async fn list(&self) -> Result<Vec<PathBuf>, IsolateError> {
            let out = self.run_git_capture("list", &["worktree", "list", "--porcelain"]).await?;
            let mut paths = Vec::new();
            for line in out.lines() {
                if let Some(p) = line.strip_prefix("worktree ") {
                    let pb = PathBuf::from(p);
                    if pb.starts_with(&self.worktree_root) {
                        paths.push(pb);
                    }
                }
            }
            Ok(paths)
        }

        /// GC orphaned worktrees left by a crash (F9): prune, then remove any tree
        /// under our root whose run is no longer live. The caller supplies the set of
        /// live run ids; everything else is reaped.
        pub async fn gc_orphans(&self, live_run_ids: &BTreeSet<String>) -> Result<usize, IsolateError> {
            self.run_git("prune", &["worktree", "prune"]).await?;
            let mut reaped = 0;
            for path in self.list().await? {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
                if !live_run_ids.contains(&name) {
                    let rel = format!(".hide/wt/{name}");
                    if self.run_git("remove", &["worktree", "remove", "--force", &rel]).await.is_ok() {
                        reaped += 1;
                    }
                }
            }
            Ok(reaped)
        }

        async fn run_git(&self, op: &'static str, args: &[&str]) -> Result<(), IsolateError> {
            self.run_git_capture(op, args).await.map(|_| ())
        }

        async fn run_git_capture(&self, op: &'static str, args: &[&str]) -> Result<String, IsolateError> {
            match &self.git {
                GitRunner::Fake(log) => {
                    log.lock().push(args.iter().map(|s| s.to_string()).collect());
                    Ok(String::new())
                }
                GitRunner::System => {
                    let output = Command::new("git")
                        .args(args)
                        .current_dir(&self.repo_root)
                        .stdin(std::process::Stdio::null())
                        .output()
                        .await?;
                    if output.status.success() {
                        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
                    } else {
                        Err(IsolateError::Git {
                            op,
                            code: output.status.code(),
                            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                        })
                    }
                }
            }
        }
    }

    /// Seed the per-run env namespace (§4.3 process/env level): a private TMPDIR,
    /// build cache, unique DB/schema names, and PORT/DATABASE_URL pointing at the
    /// run's leased port range. Injected into the run's shell tools so migrations and
    /// dev-servers in different runs never clobber each other.
    pub fn env_seed(run_id: &str, worktree: &Path, ports: &PortLease) -> BTreeMap<String, String> {
        let mut env = BTreeMap::new();
        let tmp = worktree.join(".hide-tmp");
        env.insert("TMPDIR".to_string(), tmp.display().to_string());
        env.insert("HIDE_RUN_CACHE".to_string(), worktree.join(".hide-cache").display().to_string());
        // Unique DB/schema names per run (§4.3 "unique DB/schema names hide_run_<id>").
        let db_name = format!("hide_run_{}", sanitize(run_id));
        env.insert("HIDE_DB_NAME".to_string(), db_name.clone());
        env.insert("HIDE_DB_SCHEMA".to_string(), db_name.clone());
        if let Some(&primary) = ports.ports.first() {
            env.insert("PORT".to_string(), primary.to_string());
            env.insert("DATABASE_URL".to_string(), format!("postgres://localhost:{primary}/{db_name}"));
        }
        env.insert("HIDE_RUN_ID".to_string(), run_id.to_string());
        env
    }

    /// A workspace-write sandbox scoped to the run's worktree (ch.10 enforces). Reads
    /// + writes confined to the tree; default-deny network.
    pub fn workspace_sandbox(worktree: &Path) -> SandboxProfile {
        let root = worktree.display().to_string();
        SandboxProfile {
            tier: SandboxTier::WorkspaceWrite,
            read_roots: vec![root.clone()],
            write_roots: vec![root],
            allowed_commands: Vec::new(),
            network: NetworkPolicy::default(),
        }
    }

    fn sanitize(id: &str) -> String {
        id.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn port_allocator_leases_disjoint_ranges() {
            let mut alloc = PortAllocator::new(4000, 4005);
            let a = alloc.lease("run_a", 2).unwrap();
            let b = alloc.lease("run_b", 2).unwrap();
            assert_eq!(a.ports.len(), 2);
            assert_eq!(b.ports.len(), 2);
            // Disjoint.
            for p in &a.ports {
                assert!(!b.ports.contains(p));
            }
            // Exhaustion is honest (only 2 left, ask for 3).
            assert!(alloc.lease("run_c", 3).is_none());
            alloc.release(&a);
            // Released ports are reusable.
            assert!(alloc.lease("run_c", 2).is_some());
        }

        #[test]
        fn allocator_tracks_leased_count_and_runs_as_truth() {
            let mut alloc = PortAllocator::new(4000, 4010);
            assert_eq!(alloc.leased_count(), 0);
            assert_eq!(alloc.leased_runs(), 0);

            let a = alloc.lease("run_a", 2).unwrap();
            let b = alloc.lease("run_b", 3).unwrap();
            assert_eq!(alloc.leased_count(), 5, "2 + 3 ports leased");
            assert_eq!(alloc.leased_runs(), 2, "two distinct runs hold leases");

            // Releasing one run returns exactly its ports + forgets the run.
            alloc.release(&a);
            assert_eq!(alloc.leased_count(), 3);
            assert_eq!(alloc.leased_runs(), 1);

            alloc.release(&b);
            assert_eq!(alloc.leased_count(), 0, "pool fully returned to baseline");
            assert_eq!(alloc.leased_runs(), 0);

            // Releasing an empty/synthetic lease is a no-op (idempotent, can't leak).
            alloc.release(&PortLease { run_id: "run_a".to_string(), ports: Vec::new() });
            assert_eq!(alloc.leased_count(), 0);
            assert_eq!(alloc.leased_runs(), 0);
        }

        #[test]
        fn env_seed_namespaces_db_and_ports() {
            let lease = PortLease { run_id: "run_x".to_string(), ports: vec![5100, 5101] };
            let env = env_seed("run_x", Path::new("/tmp/wt/run_x"), &lease);
            assert_eq!(env.get("PORT").map(String::as_str), Some("5100"));
            assert_eq!(env.get("HIDE_DB_NAME").map(String::as_str), Some("hide_run_run_x"));
            assert!(env.get("DATABASE_URL").unwrap().contains(":5100/"));
            assert!(env.get("TMPDIR").unwrap().contains("run_x"));
        }

        #[tokio::test]
        async fn isolate_run_issues_worktree_add_and_leases_ports() {
            let dir = std::env::temp_dir().join(format!("hide_fleet_iso_{}", ulid::Ulid::new()));
            std::fs::create_dir_all(&dir).unwrap();
            let (mgr, log) = WorktreeManager::new(&dir, PortAllocator::new(4100, 4110)).with_fake_git();

            let ws = mgr.isolate_run("run_42", "main", 2).await.unwrap();
            assert_eq!(ws.ports.ports.len(), 2);
            assert_eq!(ws.worktree.branch, "hide/run_42");
            assert!(ws.worktree.path.ends_with(".hide/wt/run_42"));
            assert_eq!(ws.worktree.sandbox.tier, SandboxTier::WorkspaceWrite);

            // The exact `git worktree add` invocation was issued.
            {
                let calls = log.lock();
                assert!(calls.iter().any(|c| {
                    c.first().map(String::as_str) == Some("worktree")
                        && c.get(1).map(String::as_str) == Some("add")
                        && c.contains(&"main".to_string())
                        && c.contains(&"hide/run_42".to_string())
                }));
            }

            // Release a discarded run → force-remove + prune issued, ports returned.
            mgr.release_run(&ws, RunOutcome { discarded: true }).await.unwrap();
            {
                let calls = log.lock();
                assert!(calls.iter().any(|c| c.contains(&"remove".to_string()) && c.contains(&"--force".to_string())));
                assert!(calls.iter().any(|c| c == &vec!["worktree", "prune"]));
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
#[rustfmt::skip]
pub mod manager {
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
    use std::collections::{BTreeMap, BTreeSet};
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
            Self { kernel, max_steps: 256 }
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
            match self.kernel.start_run(job.session_id.clone(), job.spec.objective.clone()).await {
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
            Self { repo_root: ".".to_string(), port_pool_start: 4000, port_pool_end: 4100, ports_per_run: 2 }
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
        /// The real `RunWorkspace` produced by `admit_and_launch`, keyed by job id.
        /// `fold_completion` reuses the actual handle (its leased ports) so
        /// `release_run` returns the run's ports to the pool — rebuilding a synthetic
        /// workspace with empty ports leaked the pool under sustained scheduling.
        active_workspaces: parking_lot::Mutex<BTreeMap<String, RunWorkspace>>,
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
                active_workspaces: parking_lot::Mutex::new(BTreeMap::new()),
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
        ///
        /// The model/cpu pool counts come from the job projection (the queue is the
        /// source of truth for *which* jobs are live in each pool). The isolation
        /// leases (`worktrees`, `ports_leased`) are read from the allocator's TRUTH
        /// via the [`WorktreeManager`], not estimated from the projection — so a
        /// release leak (or a launch that out-/under-paced the projection) can't
        /// silently desync the Governor's admission ceilings (reviewer finding #2).
        pub fn occupancy(&self) -> PoolOccupancy {
            PoolOccupancy {
                model_runs: self.queue.live_in_class(ConcurrencyClass::Model),
                cpu_runs: self.queue.live_in_class(ConcurrencyClass::CpuOnly),
                worktrees: self.worktrees.live_worktree_count() as u32,
                ports_leased: self.worktrees.ports_leased_count() as u32,
            }
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
                FleetScheduler::plan_tick(&gov, &ready, occupancy, interactive_waiting, victim_id.as_deref())
            };

            // Surface a breaker banner event if open.
            if plan.breaker_open {
                let reason = self.governor.lock().breaker().reason.clone().unwrap_or_default();
                self.emit_governor_breaker("spawn_rate", &reason).await?;
            }

            // 2. preemption — checkpoint-and-yield (never kill, P8). The scaffolded
            // kernel run is short, so here we mark the victim Preempted + emit the
            // event; a fuller impl flips GenerateRequest.abort at a token boundary.
            for victim_id in &plan.preempt {
                self.queue
                    .set_status_logged(&self.log, victim_id, JobStatus::Preempted, json!({ "for": "interactive" }))
                    .await?;
            }

            // 3. admit + launch.
            let mut launched = Vec::new();
            for job_id in &plan.admit {
                if let Some(job) = self.queue.get(job_id) {
                    match self.admit_and_launch(&job).await {
                        Ok(ws) => launched.push(LaunchedRun { job_id: job_id.clone(), workspace: ws }),
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

            self.queue.set_status_logged(&self.log, &job.id, JobStatus::Admitted, json!({})).await?;

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

            self.queue.set_status_logged(&self.log, &job.id, JobStatus::Running, json!({})).await?;

            // Remember the real workspace (with its leased ports) so completion can
            // release exactly what was leased (P-leak fix: see `active_workspaces`).
            self.active_workspaces.lock().insert(job.id.clone(), workspace.clone());

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
                // Reuse the REAL workspace produced by admit_and_launch (its leased
                // ports), so release_run actually returns this run's ports to the
                // pool. Falling back to a synthetic empty-port handle would re-leak.
                let ws = self
                    .active_workspaces
                    .lock()
                    .remove(&report.job_id)
                    .unwrap_or_else(|| self.synthetic_release_handle(&job));
                let _ = self.worktrees.release_run(&ws, RunOutcome { discarded }).await;
                self.emit_workspace_released(&job.id, &job.session_id, !discarded).await?;
            }
            Ok(())
        }

        /// Fallback release handle when the real workspace is absent from
        /// `active_workspaces` (e.g. a completion folded after a manager restart, or
        /// a boot-time orphan). Carries empty ports — the allocator wasn't holding a
        /// lease for this run in the current process, so there is nothing to return —
        /// but still drives the worktree remove/prune so the tree is reaped.
        fn synthetic_release_handle(&self, job: &AgentJob) -> RunWorkspace {
            let path = self.worktrees.worktree_root().join(&job.id);
            RunWorkspace {
                run_id: job.id.clone(),
                worktree: crate::isolate::WorktreeLease {
                    run_id: job.id.clone(),
                    branch: format!("hide/{}", job.id),
                    path: path.clone(),
                    base_ref: job.base_ref.clone(),
                    sandbox: crate::isolate::workspace_sandbox(&path),
                },
                ports: crate::isolate::PortLease { run_id: job.id.clone(), ports: Vec::new() },
                env: Default::default(),
            }
        }

        /// Drive ticks until the queue reaches quiescence: no ready jobs and no live
        /// jobs (everything terminal). Bounded by `max_ticks` to avoid a runaway loop
        /// in a misbehaving test. Used by batch drains + tests.
        pub async fn run_to_quiescence(&self, max_slots: u32, max_ticks: usize) -> Result<()> {
            for _ in 0..max_ticks {
                let _ = self.schedule_tick(max_slots, 40.0, 40.0).await?;
                // Wait for in-flight runs to report, then fold.
                let live = self.queue.all().iter().filter(|j| j.status.is_live()).count();
                if live > 0 {
                    self.await_completions(live).await?;
                }
                self.drain_completions().await?;
                let pending = self.queue.all().iter().any(|j| !j.status.is_terminal());
                if !pending {
                    return Ok(());
                }
            }
            Ok(())
        }

        /// GC orphaned worktrees on boot (F9): reap any worktree whose run isn't live.
        pub async fn gc_orphans_on_boot(&self) -> Result<usize> {
            let live: BTreeSet<String> =
                self.queue.all().iter().filter(|j| j.status.is_live()).map(|j| j.id.clone()).collect();
            self.worktrees.gc_orphans(&live).await.map_err(|e| hide_core::HideError::msg(format!("gc_orphans: {e}")))
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
            let session = self.queue.all().first().map(|j| j.session_id.clone()).unwrap_or_default();
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
            Self { status: JobStatus::Done, summary: "scripted ok".to_string() }
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
                // Tests drive ticks synchronously (microseconds apart), not at the ~1 Hz
                // production cadence, so the spawn-rate EWMA would trip the breaker and
                // stall admission. Lift the ceiling for deterministic scheduling tests.
                max_spawns_per_min: 1_000_000.0,
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
            let config = FleetConfig { repo_root: dir.display().to_string(), ..Default::default() };
            FleetManager::new(log, FleetGovernor::new(envelope), probe, launcher, config).with_fake_worktrees()
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
            let envelope = ResourceEnvelope { max_model_runs: 1, ram_headroom_mb_min: 128, ..Default::default() };
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
                FleetConfig { repo_root: dir.display().to_string(), ..Default::default() },
            )
            .with_fake_worktrees();

            let job = AgentJob::new("emit events", PriorityClass::Normal);
            mgr.enqueue(job).await.unwrap();
            mgr.schedule_tick(1, 40.0, 40.0).await.unwrap();
            mgr.await_completions(1).await.unwrap();

            let kinds: Vec<String> = log.scan(None, None, None).await.unwrap().into_iter().map(|e| e.kind).collect();
            for expected in [
                "job.enqueued",
                "job.admitted",
                "workspace.created",
                "job.started",
                "job.completed",
                "workspace.released",
            ] {
                assert!(kinds.iter().any(|k| k == expected), "missing event {expected}; got {kinds:?}");
            }
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[tokio::test]
        async fn model_pool_ceiling_bounds_concurrent_admissions() {
            // max_model_runs = 1: enqueue 2 jobs; only one is admitted per tick.
            let mgr = manager_with(Arc::new(ScriptedLauncher::default()), 1);
            for i in 0..2 {
                mgr.enqueue(AgentJob::new(format!("job {i}"), PriorityClass::Normal)).await.unwrap();
            }
            let (plan, _launched) = mgr.schedule_tick(1, 40.0, 40.0).await.unwrap();
            assert_eq!(plan.admit.len(), 1, "model pool ceiling = 1");
            assert_eq!(plan.deferred.len(), 1);
        }

        #[tokio::test]
        async fn ports_round_trip_to_baseline_after_a_run_completes() {
            // Finding #1: a completed run must return its leased ports to the pool.
            let mgr = manager_with(Arc::new(ScriptedLauncher::default()), 2);
            // Baseline: nothing leased.
            assert_eq!(mgr.worktrees().ports_leased_count(), 0);
            assert_eq!(mgr.occupancy().ports_leased, 0);

            let job = AgentJob::new("lease then release", PriorityClass::Normal);
            let id = job.id.clone();
            mgr.enqueue(job).await.unwrap();

            mgr.schedule_tick(2, 40.0, 40.0).await.unwrap();
            // The run is live and holds `ports_per_run` ports (allocator truth).
            assert_eq!(
                mgr.worktrees().ports_leased_count(),
                mgr.config.ports_per_run as usize,
                "ports leased while the run is live"
            );
            assert_eq!(mgr.occupancy().ports_leased, mgr.config.ports_per_run as u32);

            // Fold the completion → release_run must return the ports.
            mgr.await_completions(1).await.unwrap();
            assert_eq!(mgr.queue().get(&id).unwrap().status, JobStatus::Done);
            assert_eq!(
                mgr.worktrees().ports_leased_count(),
                0,
                "ports returned to the pool — no leak after completion"
            );
            assert_eq!(mgr.occupancy().ports_leased, 0);
            assert_eq!(mgr.occupancy().worktrees, 0, "no live leases remain");
        }

        #[tokio::test]
        async fn ports_do_not_leak_under_sustained_scheduling() {
            // Finding #1 under load: many sequential runs through a tiny pool must not
            // exhaust it. With max_ports_leased ceilings unchanged, a leak would make
            // later admissions fail; instead every cycle round-trips to baseline.
            let mgr = manager_with(Arc::new(ScriptedLauncher::default()), 1);
            for i in 0..12 {
                let job = AgentJob::new(format!("run {i}"), PriorityClass::Normal);
                let id = job.id.clone();
                mgr.enqueue(job).await.unwrap();
                mgr.schedule_tick(1, 40.0, 40.0).await.unwrap();
                mgr.await_completions(1).await.unwrap();
                assert_eq!(
                    mgr.queue().get(&id).unwrap().status,
                    JobStatus::Done,
                    "cycle {i} admitted + completed (pool not exhausted)"
                );
                assert_eq!(mgr.worktrees().ports_leased_count(), 0, "cycle {i}: pool back to baseline");
            }
        }

        #[tokio::test]
        async fn occupancy_reflects_allocator_truth_not_an_estimate() {
            // Finding #2: occupancy.ports_leased / worktrees come from the allocator,
            // so they exactly track real leases through the lifecycle.
            let mgr = manager_with(Arc::new(ScriptedLauncher::default()), 2);
            for i in 0..2 {
                mgr.enqueue(AgentJob::new(format!("job {i}"), PriorityClass::Normal)).await.unwrap();
            }
            mgr.schedule_tick(2, 40.0, 40.0).await.unwrap();

            let occ = mgr.occupancy();
            // Two live model runs each hold ports_per_run ports → allocator truth.
            assert_eq!(occ.ports_leased, 2 * mgr.config.ports_per_run as u32);
            assert_eq!(occ.ports_leased, mgr.worktrees().ports_leased_count() as u32);
            assert_eq!(occ.worktrees, 2);
            assert_eq!(occ.worktrees, mgr.worktrees().live_worktree_count() as u32);

            mgr.await_completions(2).await.unwrap();
            let occ = mgr.occupancy();
            assert_eq!(occ.ports_leased, 0);
            assert_eq!(occ.worktrees, 0);
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
}
#[rustfmt::skip]
pub mod merge {
    //! Merge & conflict resolution (bible ch.09 §4.4).
    //!
    //! Two distinct flows funnel through an integration branch:
    //! - **Tournament** (same goal → select one winner): regression-filter →
    //!   oracle-rank → judge tie-break (§4.4.2, P4). The selector is oracle-first by
    //!   construction; the judge only breaks ties among oracle-equivalent leaders.
    //! - **Fan-out / map-reduce** (disjoint footprints → combine all): the
    //!   integration funnel merges each run's changes, with the conflict ladder
    //!   (structured → 3-way → escalate, §4.4.3).
    //!
    //! **Footprint-disjoint scheduling** (§4.2.3) is the single biggest lever against
    //! "merge is the hard part": subtasks with disjoint file footprints parallelize
    //! freely (no conflict possible by construction); overlapping ones serialize or
    //! race under tournament semantics.
    //!
    //! The 3-way merge is real (the `similar` crate's diff over the common ancestor);
    //! it is *content* merge, not a git-process invocation, so it is unit-testable
    //! and runs in-memory on candidate file blobs.

    use serde::{Deserialize, Serialize};
    use std::collections::BTreeSet;

    // ---------------------------------------------------------------------------
    // Footprint analysis (§4.2.3)
    // ---------------------------------------------------------------------------

    /// A subtask's predicted file footprint (from the plan's target paths + a cheap
    /// static touch-set). Disjointness is decided by set intersection.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Footprint {
        pub job_id: String,
        pub files: BTreeSet<String>,
    }

    impl Footprint {
        pub fn new(job_id: impl Into<String>, files: impl IntoIterator<Item = String>) -> Self {
            Self { job_id: job_id.into(), files: files.into_iter().collect() }
        }

        pub fn overlaps(&self, other: &Footprint) -> bool {
            !self.files.is_disjoint(&other.files)
        }
    }

    /// How a set of subtasks should be scheduled relative to each other.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct FootprintPlan {
        /// Groups that may run fully in parallel (mutually disjoint footprints).
        pub parallel_groups: Vec<Vec<String>>,
        /// Pairs that overlap and must serialize (a dependency edge is added) or race
        /// under tournament semantics.
        pub overlaps: Vec<(String, String)>,
    }

    /// Partition subtasks into parallel-safe groups + overlap edges. Greedy
    /// graph-coloring: a job joins an existing group iff it is disjoint from every
    /// member; otherwise it opens a new group, and the conflicting pair is recorded.
    pub fn plan_footprints(footprints: &[Footprint]) -> FootprintPlan {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut overlaps = Vec::new();
        for (i, fp) in footprints.iter().enumerate() {
            // Record overlap edges against all prior jobs.
            for prior in footprints.iter().take(i) {
                if fp.overlaps(prior) {
                    overlaps.push((prior.job_id.clone(), fp.job_id.clone()));
                }
            }
            // Place into the first group with no conflicting member.
            let mut placed = false;
            for group in &mut groups {
                let conflict = group.iter().any(|&j| footprints[j].overlaps(fp));
                if !conflict {
                    group.push(i);
                    placed = true;
                    break;
                }
            }
            if !placed {
                groups.push(vec![i]);
            }
        }
        FootprintPlan {
            parallel_groups: groups
                .into_iter()
                .map(|g| g.into_iter().map(|i| footprints[i].job_id.clone()).collect())
                .collect(),
            overlaps,
        }
    }

    // ---------------------------------------------------------------------------
    // Tournament selection (§4.4.2)
    // ---------------------------------------------------------------------------

    /// One candidate run's outcome (a patch attempt at the same goal).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CandidatePatch {
        pub job_id: String,
        pub diff_hash: String,
        pub changed_files: Vec<String>,
        /// Passed its own acceptance oracle (build+test).
        pub oracle_passed: bool,
        /// Passed the existing regression suite (the §3.2 regression filter).
        pub regression_clean: bool,
        /// Number of deterministic acceptance oracles passed (rank signal).
        pub oracles_passed: u32,
        /// Diff size in changed lines (smaller is better — a rank tie-break signal).
        pub diff_lines: u32,
        /// Warnings emitted (fewer is better).
        pub warnings: u32,
        pub summary: String,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum MergeStrategy {
        SelectWinner,
        ThreeWay,
        Structured,
        ManualReview,
        RejectAll,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct MergeDecision {
        pub winner_job_id: Option<String>,
        /// The candidates the winner beat (for the `merge.selected` event).
        pub beaten: Vec<String>,
        pub strategy: MergeStrategy,
        pub conflicts: Vec<String>,
        /// Human-readable basis (the §4.4.2 `selection_basis`).
        pub basis: String,
        /// True when the oracle signals couldn't separate the leaders → a judge
        /// tie-break is needed (the host runs the LLM judge; the selector flags it).
        pub needs_judge: bool,
        pub judge_leaders: Vec<String>,
    }

    pub struct TournamentSelector;

    impl TournamentSelector {
        /// Oracle-first selection (P4). 1) regression filter, 2) rank by oracle
        /// signals (oracles passed ↓, warnings ↑, diff size ↑ — all deterministic),
        /// 3) flag a judge tie-break only among oracle-equivalent leaders.
        pub fn select(candidates: &[CandidatePatch]) -> MergeDecision {
            // 1. Regression filter: a fix that breaks the existing suite is out.
            let mut viable: Vec<&CandidatePatch> =
                candidates.iter().filter(|c| c.oracle_passed && c.regression_clean).collect();
            if viable.is_empty() {
                return MergeDecision {
                    winner_job_id: None,
                    beaten: Vec::new(),
                    strategy: MergeStrategy::RejectAll,
                    conflicts: Vec::new(),
                    basis: "no candidate passed deterministic oracles + regression".to_string(),
                    needs_judge: false,
                    judge_leaders: Vec::new(),
                };
            }
            // 2. Oracle rank (lexicographic, all deterministic): more oracles, fewer
            // warnings, smaller diff, then stable job_id.
            viable.sort_by(|a, b| {
                b.oracles_passed
                    .cmp(&a.oracles_passed)
                    .then_with(|| a.warnings.cmp(&b.warnings))
                    .then_with(|| a.diff_lines.cmp(&b.diff_lines))
                    .then_with(|| a.job_id.cmp(&b.job_id))
            });
            let leader = viable[0];
            // 3. Leaders the oracle signals cannot separate (equal on all signals).
            let leaders: Vec<&CandidatePatch> = viable
                .iter()
                .copied()
                .filter(|c| {
                    c.oracles_passed == leader.oracles_passed
                        && c.warnings == leader.warnings
                        && c.diff_lines == leader.diff_lines
                })
                .collect();
            let needs_judge = leaders.len() > 1;
            let beaten: Vec<String> = viable.iter().skip(1).map(|c| c.job_id.clone()).collect();
            MergeDecision {
                winner_job_id: Some(leader.job_id.clone()),
                beaten,
                strategy: MergeStrategy::SelectWinner,
                conflicts: Vec::new(),
                basis: format!(
                    "oracle-first: {} oracles, {} warnings, {} diff lines",
                    leader.oracles_passed, leader.warnings, leader.diff_lines
                ),
                needs_judge,
                judge_leaders: leaders.iter().map(|c| c.job_id.clone()).collect(),
            }
        }
    }

    // ---------------------------------------------------------------------------
    // The conflict ladder (§4.4.3) — real 3-way text merge via `similar`
    // ---------------------------------------------------------------------------

    /// How a file merge was resolved (the `merge.resolved{by}` event basis).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ResolvedBy {
        /// One side was unchanged from base → take the other.
        FastForward,
        /// Both sides changed disjoint regions → clean 3-way merge.
        ThreeWay,
        /// Both sides changed the same region differently → conflict.
        Conflict,
    }

    /// The result of a 3-way merge of one file.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct FileMerge {
        pub path: String,
        pub by: ResolvedBy,
        /// The merged text (with conflict markers if `by == Conflict`).
        pub merged: String,
        pub conflicted: bool,
    }

    /// A real line-level 3-way merge of one file against a common ancestor, using the
    /// `similar` crate to compute each side's hunks. Clean when the two sides touch
    /// disjoint line regions; conflicting when they edit the same region differently.
    /// This is step 2 of the ladder (step 1, structured/AST merge, is a host concern
    /// that uses tree-sitter; step 3 is the LLM resolver run; step 4 is escalate).
    pub fn three_way_merge(path: &str, base: &str, ours: &str, theirs: &str) -> FileMerge {
        // Fast paths.
        if ours == theirs {
            return FileMerge {
                path: path.to_string(),
                by: ResolvedBy::FastForward,
                merged: ours.to_string(),
                conflicted: false,
            };
        }
        if ours == base {
            return FileMerge {
                path: path.to_string(),
                by: ResolvedBy::FastForward,
                merged: theirs.to_string(),
                conflicted: false,
            };
        }
        if theirs == base {
            return FileMerge {
                path: path.to_string(),
                by: ResolvedBy::FastForward,
                merged: ours.to_string(),
                conflicted: false,
            };
        }

        let base_lines: Vec<&str> = base.lines().collect();
        let ours_lines: Vec<&str> = ours.lines().collect();
        let theirs_lines: Vec<&str> = theirs.lines().collect();

        // Compute, for each base line index, whether "ours" and "theirs" modified the
        // region around it. We walk base and emit merged output, detecting same-region
        // double edits as conflicts.
        let ours_changes = changed_base_regions(&base_lines, &ours_lines);
        let theirs_changes = changed_base_regions(&base_lines, &theirs_lines);

        // If the changed base-line sets are disjoint, the merge is clean: apply each
        // side's version. Otherwise it conflicts.
        let conflict = !ours_changes.is_disjoint(&theirs_changes);

        if conflict {
            let merged = format!("<<<<<<< ours\n{}\n=======\n{}\n>>>>>>> theirs\n", ours.trim_end(), theirs.trim_end());
            FileMerge { path: path.to_string(), by: ResolvedBy::Conflict, merged, conflicted: true }
        } else {
            // Disjoint edits: a real 3-way reconstruction. Apply theirs' changes onto
            // ours (ours already differs from base only in ours_changes; theirs'
            // changes are in disjoint base regions, so layering is well-defined). We
            // reconstruct by taking, per base region, whichever side changed it.
            let merged = reconstruct_disjoint(&base_lines, ours, theirs, &ours_changes);
            FileMerge { path: path.to_string(), by: ResolvedBy::ThreeWay, merged, conflicted: false }
        }
    }

    /// The set of base line indices that `side` modified or deleted relative to base
    /// (an "anchor" set used for disjointness). Computed from the `similar` line diff.
    fn changed_base_regions(base: &[&str], side: &[&str]) -> BTreeSet<usize> {
        use similar::{capture_diff_slices, Algorithm, DiffOp};
        let ops = capture_diff_slices(Algorithm::Myers, base, side);
        let mut changed = BTreeSet::new();
        for op in ops {
            match op {
                DiffOp::Equal { .. } => {}
                DiffOp::Delete { old_index, old_len, .. } => {
                    for i in old_index..old_index + old_len {
                        changed.insert(i);
                    }
                }
                DiffOp::Replace { old_index, old_len, .. } => {
                    for i in old_index..old_index + old_len {
                        changed.insert(i);
                    }
                }
                DiffOp::Insert { old_index, .. } => {
                    // Insertion anchors at the base position it precedes.
                    changed.insert(old_index);
                }
            }
        }
        changed
    }

    /// Reconstruct a merged file when the two sides edited disjoint base regions. We
    /// rebuild by walking base and, for each region, choosing the side that changed
    /// it. Insertions from both sides are preserved in base order.
    fn reconstruct_disjoint(base: &[&str], ours: &str, theirs: &str, ours_changes: &BTreeSet<usize>) -> String {
        use similar::{capture_diff_slices, Algorithm};
        let ours_lines: Vec<&str> = ours.lines().collect();
        let theirs_lines: Vec<&str> = theirs.lines().collect();

        // Build a map base_index -> replacement text from each side.
        // Strategy: take `theirs` as the canvas, then overlay `ours`' changes onto the
        // base regions ours owns. Because the change sets are disjoint, every base
        // region is owned by at most one side; theirs already reflects its own edits,
        // so we only need to splice ours' edits back in. We do this by diffing
        // theirs-vs-base and ours-vs-base in lockstep over base.
        let ours_ops = capture_diff_slices(Algorithm::Myers, base, &ours_lines);
        let theirs_ops = capture_diff_slices(Algorithm::Myers, base, &theirs_lines);

        // Map each base line to its "ours" output (the lines ours produces for it).
        let ours_out = side_output_per_base(base, &ours_lines, &ours_ops);
        let theirs_out = side_output_per_base(base, &theirs_lines, &theirs_ops);

        let mut merged_lines: Vec<String> = Vec::new();
        for (i, _) in base.iter().enumerate() {
            if ours_changes.contains(&i) {
                merged_lines.extend(ours_out.get(&i).cloned().unwrap_or_default());
            } else {
                merged_lines.extend(theirs_out.get(&i).cloned().unwrap_or_default());
            }
        }
        // Trailing insertions anchored past the last base line.
        let end = base.len();
        if ours_changes.contains(&end) {
            merged_lines.extend(ours_out.get(&end).cloned().unwrap_or_default());
        } else {
            merged_lines.extend(theirs_out.get(&end).cloned().unwrap_or_default());
        }
        let mut s = merged_lines.join("\n");
        if !s.is_empty() {
            s.push('\n');
        }
        s
    }

    /// For each base line index (and a synthetic `base.len()` slot for trailing
    /// inserts), the lines a side emits there.
    fn side_output_per_base(
        base: &[&str],
        side: &[&str],
        ops: &[similar::DiffOp],
    ) -> std::collections::BTreeMap<usize, Vec<String>> {
        use similar::DiffOp;
        let mut out: std::collections::BTreeMap<usize, Vec<String>> = std::collections::BTreeMap::new();
        for op in ops {
            match *op {
                DiffOp::Equal { old_index, new_index, len } => {
                    for k in 0..len {
                        out.entry(old_index + k).or_default().push(side[new_index + k].to_string());
                    }
                }
                DiffOp::Replace { old_index, old_len, new_index, new_len } => {
                    let repl: Vec<String> = (0..new_len).map(|k| side[new_index + k].to_string()).collect();
                    out.entry(old_index).or_default().extend(repl);
                    // Mark the rest of the replaced base region as consumed (no output).
                    for k in 1..old_len {
                        out.entry(old_index + k).or_default();
                    }
                }
                DiffOp::Delete { old_index, old_len, .. } => {
                    for k in 0..old_len {
                        out.entry(old_index + k).or_default();
                    }
                }
                DiffOp::Insert { old_index, new_index, new_len } => {
                    let ins: Vec<String> = (0..new_len).map(|k| side[new_index + k].to_string()).collect();
                    out.entry(old_index).or_default().extend(ins);
                    let _ = base; // base only used for bounds reasoning above.
                }
            }
        }
        out
    }

    // ---------------------------------------------------------------------------
    // The integration funnel (§4.4.1)
    // ---------------------------------------------------------------------------

    /// The outcome of integrating N fan-out runs onto an integration branch.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct IntegrationResult {
        pub adopted: Vec<String>,
        pub dropped: Vec<String>,
        pub conflicts: Vec<String>,
        /// Whether the full suite was green after integration (the promote gate).
        pub suite_green: bool,
    }

    /// Integrate a set of fan-out candidates whose footprints have been classified.
    /// Disjoint-footprint candidates are adopted in order; any pair that conflicts on
    /// a file is recorded as a conflict for the ladder/escalation. This is the
    /// *decision* funnel — the host performs the actual git merges + suite run and
    /// supplies `suite_green`.
    pub fn integrate(candidates: &[CandidatePatch], footprints: &[Footprint], suite_green: bool) -> IntegrationResult {
        let plan = plan_footprints(footprints);
        let conflict_files: BTreeSet<String> = plan
            .overlaps
            .iter()
            .flat_map(|(a, b)| {
                let fa = footprints.iter().find(|f| &f.job_id == a);
                let fb = footprints.iter().find(|f| &f.job_id == b);
                match (fa, fb) {
                    (Some(fa), Some(fb)) => fa.files.intersection(&fb.files).cloned().collect(),
                    _ => Vec::new(),
                }
            })
            .collect();

        let mut adopted = Vec::new();
        let mut dropped = Vec::new();
        let conflicted_jobs: BTreeSet<String> =
            plan.overlaps.iter().flat_map(|(a, b)| [a.clone(), b.clone()]).collect();

        for c in candidates {
            if !c.oracle_passed || !c.regression_clean {
                dropped.push(c.job_id.clone());
            } else if conflicted_jobs.contains(&c.job_id) {
                // Overlapping footprint: held for the conflict ladder, not silently
                // adopted (no silent wrong merge, P3).
                dropped.push(c.job_id.clone());
            } else {
                adopted.push(c.job_id.clone());
            }
        }

        IntegrationResult { adopted, dropped, conflicts: conflict_files.into_iter().collect(), suite_green }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn cand(id: &str, ok: bool, oracles: u32, warns: u32, diff: u32) -> CandidatePatch {
            CandidatePatch {
                job_id: id.to_string(),
                diff_hash: format!("h_{id}"),
                changed_files: vec![format!("{id}.rs")],
                oracle_passed: ok,
                regression_clean: ok,
                oracles_passed: oracles,
                diff_lines: diff,
                warnings: warns,
                summary: String::new(),
            }
        }

        #[test]
        fn selector_is_oracle_first_and_filters_regressions() {
            let cands = vec![
                cand("a", true, 3, 0, 50),
                cand("b", false, 5, 0, 10), // more oracles but fails regression → out
                cand("c", true, 4, 1, 40),
            ];
            let d = TournamentSelector::select(&cands);
            // c has more oracles passed than a (4 > 3) → c wins; b filtered out.
            assert_eq!(d.winner_job_id, Some("c".to_string()));
            assert!(d.beaten.contains(&"a".to_string()));
            assert!(!d.needs_judge);
        }

        #[test]
        fn selector_flags_judge_on_oracle_equivalent_leaders() {
            let cands = vec![cand("a", true, 3, 0, 20), cand("b", true, 3, 0, 20)];
            let d = TournamentSelector::select(&cands);
            assert!(d.needs_judge);
            assert_eq!(d.judge_leaders.len(), 2);
        }

        #[test]
        fn selector_rejects_when_nothing_viable() {
            let cands = vec![cand("a", false, 1, 0, 5)];
            let d = TournamentSelector::select(&cands);
            assert_eq!(d.strategy, MergeStrategy::RejectAll);
            assert!(d.winner_job_id.is_none());
        }

        #[test]
        fn footprints_partition_disjoint_and_record_overlaps() {
            let fps = vec![
                Footprint::new("a", ["src/x.rs".to_string()]),
                Footprint::new("b", ["src/y.rs".to_string()]),
                Footprint::new("c", ["src/x.rs".to_string()]), // overlaps a
            ];
            let plan = plan_footprints(&fps);
            assert!(plan.overlaps.contains(&("a".to_string(), "c".to_string())));
            // a and b are disjoint → same parallel group; c opens a new one.
            assert!(plan.parallel_groups.iter().any(|g| g.contains(&"a".to_string()) && g.contains(&"b".to_string())));
        }

        #[test]
        fn three_way_merge_fast_forwards_when_one_side_unchanged() {
            let base = "line1\nline2\n";
            let ours = "line1\nline2\n";
            let theirs = "line1\nCHANGED\n";
            let m = three_way_merge("f.txt", base, ours, theirs);
            assert!(!m.conflicted);
            assert_eq!(m.by, ResolvedBy::FastForward);
            assert_eq!(m.merged, theirs);
        }

        #[test]
        fn three_way_merge_combines_disjoint_edits() {
            let base = "a\nb\nc\nd\n";
            let ours = "AA\nb\nc\nd\n"; // edited line 0
            let theirs = "a\nb\nc\nDD\n"; // edited line 3
            let m = three_way_merge("f.txt", base, ours, theirs);
            assert!(!m.conflicted, "disjoint edits must merge cleanly");
            assert!(m.merged.contains("AA"));
            assert!(m.merged.contains("DD"));
        }

        #[test]
        fn three_way_merge_conflicts_on_same_region() {
            let base = "a\nb\nc\n";
            let ours = "a\nOURS\nc\n";
            let theirs = "a\nTHEIRS\nc\n";
            let m = three_way_merge("f.txt", base, ours, theirs);
            assert!(m.conflicted);
            assert_eq!(m.by, ResolvedBy::Conflict);
            assert!(m.merged.contains("<<<<<<<"));
        }

        #[test]
        fn integrate_holds_overlapping_footprints_for_the_ladder() {
            let cands = vec![cand("a", true, 2, 0, 5), cand("c", true, 2, 0, 5)];
            let fps =
                vec![Footprint::new("a", ["shared.rs".to_string()]), Footprint::new("c", ["shared.rs".to_string()])];
            let r = integrate(&cands, &fps, true);
            // Both touch shared.rs → neither silently adopted.
            assert!(r.adopted.is_empty());
            assert!(r.conflicts.contains(&"shared.rs".to_string()));
        }
    }
}
#[rustfmt::skip]
pub mod patterns {
    //! Orchestration patterns & the selection rule (bible ch.09 §4.2).
    //!
    //! Seven patterns compose ch.02 runs. The selection rule (P5, §4.2.2) gates them
    //! all; the default for an ambiguous task is a *single* run. The decisive
    //! heuristic: **the presence of a deterministic oracle flips the strategy from
    //! coordinate to verify-and-select** — when you can write an acceptance oracle,
    //! generate many and let the oracle pick; debate is reserved for the genuinely
    //! oracle-less case.
    //!
    //! A [`Pattern`] is the executable shape (`fan_out → runs → reduce`). The crate
    //! ships the decision (`choose_pattern`) + the pattern descriptors; the
    //! `FleetManager` materialises a chosen pattern into a set of jobs with the right
    //! footprint/dependency wiring (e.g. tournament = N footprint-overlapping jobs
    //! that race; fan-out = N footprint-disjoint jobs + a reduce job depending on
    //! them).

    use crate::queue::{AgentJob, ConcurrencyClass, PriorityClass};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum OrchestrationPattern {
        SingleAgent,
        FanOutMapReduce,
        Pipeline,
        Tournament,
        PlannerWorkersMerger,
        Debate,
        SpeculativeExploration,
    }

    impl OrchestrationPattern {
        /// The wire/config name (matches the A.1 `run_spec.orchestration` field).
        pub fn name(self) -> &'static str {
            match self {
                OrchestrationPattern::SingleAgent => "single",
                OrchestrationPattern::FanOutMapReduce => "map_reduce",
                OrchestrationPattern::Pipeline => "pipeline",
                OrchestrationPattern::Tournament => "tournament",
                OrchestrationPattern::PlannerWorkersMerger => "planner_workers",
                OrchestrationPattern::Debate => "debate",
                OrchestrationPattern::SpeculativeExploration => "speculative",
            }
        }

        pub fn from_name(name: &str) -> Option<Self> {
            Some(match name {
                "single" => OrchestrationPattern::SingleAgent,
                "map_reduce" | "fanout" => OrchestrationPattern::FanOutMapReduce,
                "pipeline" => OrchestrationPattern::Pipeline,
                "tournament" => OrchestrationPattern::Tournament,
                "planner_workers" => OrchestrationPattern::PlannerWorkersMerger,
                "debate" => OrchestrationPattern::Debate,
                "speculative" => OrchestrationPattern::SpeculativeExploration,
                _ => return None,
            })
        }
    }

    /// The characteristics of a task that drive pattern selection (§4.2.2 inputs).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct TaskShape {
        /// Can we write an acceptance oracle (build+test+grep) up front?
        pub has_acceptance_oracle: bool,
        /// Does the work partition into footprint-disjoint subtasks?
        pub partitions_disjoint: bool,
        /// Clean staged handoffs (design → implement → test)?
        pub staged_handoffs: bool,
        /// One hard goal with a high first-attempt failure rate?
        pub one_hard_goal: bool,
        /// Exploratory divergent approaches worth racing?
        pub exploratory: bool,
        /// Breadth task needing isolated context windows (research/investigation)?
        pub needs_breadth_isolation: bool,
        /// Subjective synthesis with no oracle?
        pub subjective_synthesis: bool,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PatternDecision {
        pub pattern: OrchestrationPattern,
        pub reason: String,
    }

    /// The normative selection rule (§4.2.2). The presence of a deterministic oracle
    /// flips the whole strategy.
    pub fn choose_pattern(shape: TaskShape) -> PatternDecision {
        let (pattern, reason) = if !shape.has_acceptance_oracle {
            if shape.needs_breadth_isolation {
                (
                    OrchestrationPattern::PlannerWorkersMerger,
                    "no oracle; breadth task → isolated-context workers (Anthropic 90.2% shape)",
                )
            } else if shape.subjective_synthesis {
                (OrchestrationPattern::Debate, "no oracle; subjective synthesis → debate (documented fallback, P5)")
            } else {
                (
                    OrchestrationPattern::SingleAgent,
                    "no oracle and not separable → single-agent (the safe default, K12)",
                )
            }
        } else if shape.partitions_disjoint {
            (OrchestrationPattern::FanOutMapReduce, "oracle + disjoint footprints → fan-out/map-reduce")
        } else if shape.staged_handoffs {
            (OrchestrationPattern::Pipeline, "oracle + clean staged handoffs → pipeline")
        } else if shape.one_hard_goal {
            (OrchestrationPattern::Tournament, "oracle + one hard high-failure goal → tournament/best-of-N (P4)")
        } else if shape.exploratory {
            (
                OrchestrationPattern::SpeculativeExploration,
                "oracle + divergent approaches → speculative exploration (free locally, P1)",
            )
        } else {
            (OrchestrationPattern::SingleAgent, "oracle but no parallel structure → single-agent")
        };
        PatternDecision { pattern, reason: reason.to_string() }
    }

    /// Materialise a chosen pattern into a set of jobs (the executable shape). Each
    /// pattern wires footprint/dependency structure so the scheduler + merge funnel
    /// behave correctly:
    /// - **Tournament**: `width` jobs racing the same goal (oracle selects one).
    /// - **Fan-out**: `width` disjoint jobs + a `reduce` job depending on all of them.
    /// - **Single**: one job.
    pub fn materialise(
        pattern: OrchestrationPattern,
        objective: &str,
        width: u8,
        priority: PriorityClass,
    ) -> Vec<AgentJob> {
        let width = width.max(1);
        match pattern {
            OrchestrationPattern::SingleAgent | OrchestrationPattern::Pipeline => {
                vec![tagged(AgentJob::new(objective, priority), pattern)]
            }
            OrchestrationPattern::Tournament
            | OrchestrationPattern::SpeculativeExploration
            | OrchestrationPattern::Debate => (0..width)
                .map(|i| tagged(AgentJob::new(format!("{objective} (attempt {})", i + 1), priority), pattern))
                .collect(),
            OrchestrationPattern::FanOutMapReduce | OrchestrationPattern::PlannerWorkersMerger => {
                let mut jobs: Vec<AgentJob> = (0..width)
                    .map(|i| tagged(AgentJob::new(format!("{objective} (part {})", i + 1), priority), pattern))
                    .collect();
                let child_ids: Vec<String> = jobs.iter().map(|j| j.id.clone()).collect();
                // The reduce/merger job integrates all children and runs the full
                // suite (§4.4.1). It depends on every child.
                let reduce = tagged(
                    AgentJob::new(format!("{objective} (reduce)"), priority)
                        .depends_on(child_ids)
                        .with_concurrency_class(ConcurrencyClass::Model),
                    pattern,
                );
                jobs.push(reduce);
                jobs
            }
        }
    }

    fn tagged(mut job: AgentJob, pattern: OrchestrationPattern) -> AgentJob {
        job.spec.pattern = Some(pattern.name().to_string());
        job
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn oracle_plus_disjoint_picks_fanout() {
            let d = choose_pattern(TaskShape {
                has_acceptance_oracle: true,
                partitions_disjoint: true,
                ..TaskShape::default()
            });
            assert_eq!(d.pattern, OrchestrationPattern::FanOutMapReduce);
        }

        #[test]
        fn oracle_plus_hard_goal_picks_tournament() {
            let d =
                choose_pattern(TaskShape { has_acceptance_oracle: true, one_hard_goal: true, ..TaskShape::default() });
            assert_eq!(d.pattern, OrchestrationPattern::Tournament);
        }

        #[test]
        fn no_oracle_ambiguous_defaults_to_single_agent() {
            let d = choose_pattern(TaskShape::default());
            assert_eq!(d.pattern, OrchestrationPattern::SingleAgent);
        }

        #[test]
        fn no_oracle_breadth_picks_planner_workers() {
            let d = choose_pattern(TaskShape { needs_breadth_isolation: true, ..TaskShape::default() });
            assert_eq!(d.pattern, OrchestrationPattern::PlannerWorkersMerger);
        }

        #[test]
        fn materialise_fanout_adds_reduce_depending_on_children() {
            let jobs = materialise(OrchestrationPattern::FanOutMapReduce, "port endpoints", 3, PriorityClass::Normal);
            assert_eq!(jobs.len(), 4); // 3 parts + 1 reduce
            let reduce = jobs.last().unwrap();
            assert_eq!(reduce.dependencies.len(), 3);
        }

        #[test]
        fn materialise_tournament_races_n_jobs_same_goal() {
            let jobs = materialise(OrchestrationPattern::Tournament, "fix the bug", 4, PriorityClass::High);
            assert_eq!(jobs.len(), 4);
            assert!(jobs.iter().all(|j| j.dependencies.is_empty()));
            assert!(jobs.iter().all(|j| j.spec.pattern.as_deref() == Some("tournament")));
        }
    }
}
#[rustfmt::skip]
pub mod queue {
    //! The task queue & job schema (bible ch.09 §4.5).
    //!
    //! The queue is **a projection of the event log** (P8): jobs are created and
    //! mutated by appending `job.*` events; [`JobGraph::project_from`] rebuilds the
    //! in-memory graph by folding those events. The `JobGraph` itself is a fast
    //! in-memory index over that durable truth — never a second authoritative store.
    //!
    //! The schema is reconciled to Appendix A.1: `kind`, `parent_job`, `base_ref`,
    //! `isolation`, `concurrency_class`, `attempts`/`max_attempts`, `result_ref`,
    //! `schedule`, `schema_version`. The status set restores `Admitted`,
    //! `Preempted`, and `Merging`. The `concurrency_class` Model-vs-CpuOnly split is
    //! load-bearing for the scheduler's two-pool admission (§4.5.1).

    use hide_core::event::{Event, EventClass, EventSource, NewEvent};
    use hide_core::ids::{now_ms, RunId, SessionId};
    use hide_core::persistence::DynEventLog;
    use hide_core::types::BlobRef;
    use hide_core::Result;
    use parking_lot::RwLock;
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::collections::{BTreeMap, BTreeSet};

    pub const JOB_SCHEMA_VERSION: u16 = 1;

    /// What kind of work this job represents (A.1 `kind`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum JobKind {
        #[default]
        AgentRun,
        Batch,
        TestShard,
        Research,
        Merge,
        Custom,
    }

    /// Who created this job (A.1 `created_by`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum CreatedBy {
        User,
        Agent,
        Schedule,
    }

    /// Isolation backend for the job's workspace (A.1 `isolation`, §4.3).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Isolation {
        #[default]
        Worktree,
        Overlay,
        Container,
        None,
    }

    /// The load-bearing two-pool split (§4.5.1). A `Model`-class job holds one of the
    /// runtime's `max_batch_size` generation slots — the scarce resource. A `CpuOnly`
    /// job (test shard, build, grep) does not compete for the model and runs far
    /// wider, bounded only by cores/RAM.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ConcurrencyClass {
        #[default]
        Model,
        CpuOnly,
    }

    /// Priority classes (strict order, §4.6.3). `Interactive` always wins; it
    /// preempts lower classes for a model slot. Fair-share applies *within* a class.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum PriorityClass {
        Interactive,
        High,
        Normal,
        Batch,
        Idle,
    }

    /// Job lifecycle status (A.1 `status`). Restores `Admitted`/`Preempted`/`Merging`
    /// over the original scaffold's reduced set.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum JobStatus {
        Queued,
        Admitted,
        Running,
        Paused,
        Preempted,
        Merging,
        Done,
        Failed,
        Cancelled,
    }

    impl JobStatus {
        pub fn is_terminal(self) -> bool {
            matches!(self, JobStatus::Done | JobStatus::Failed | JobStatus::Cancelled)
        }

        /// Whether this status holds a live resource grant (counts against ceilings).
        pub fn is_live(self) -> bool {
            matches!(self, JobStatus::Admitted | JobStatus::Running | JobStatus::Merging)
        }
    }

    /// Resource admission inputs (A.1 `resource_hint`; advisory — the Governor
    /// decides). The `concurrency_class` routes the job to the Model or CpuOnly pool.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ResourceRequest {
        pub memory_mb: u64,
        pub needs_gpu: bool,
        pub est_wallclock_ms: u64,
        pub generation_slots: u32,
        pub ports: u16,
        pub concurrency_class: ConcurrencyClass,
    }

    impl Default for ResourceRequest {
        fn default() -> Self {
            Self {
                memory_mb: 1024,
                needs_gpu: true,
                est_wallclock_ms: 600_000,
                generation_slots: 1,
                ports: 0,
                concurrency_class: ConcurrencyClass::Model,
            }
        }
    }

    impl ResourceRequest {
        /// A CPU-only request (test shard / build) — no model slot, no GPU.
        pub fn cpu_only(memory_mb: u64) -> Self {
            Self {
                memory_mb,
                needs_gpu: false,
                est_wallclock_ms: 120_000,
                generation_slots: 0,
                ports: 0,
                concurrency_class: ConcurrencyClass::CpuOnly,
            }
        }
    }

    /// What ch.02 run to launch (the black box, A.1 `run_spec`).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct JobSpec {
        pub objective: String,
        /// §4.2 pattern hint (single|fanout|tournament|…) — see `crate::patterns`.
        pub pattern: Option<String>,
        pub profile: Option<String>,
        pub max_steps: u32,
        pub required_resources: ResourceRequest,
    }

    impl Default for JobSpec {
        fn default() -> Self {
            Self {
                objective: String::new(),
                pattern: None,
                profile: None,
                max_steps: 128,
                required_resources: ResourceRequest::default(),
            }
        }
    }

    /// Schedule gate for batch jobs (A.1 `schedule`, §4.7.2).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct JobSchedule {
        /// Earliest start (wall-clock ms since epoch).
        pub earliest_start_ms: Option<u64>,
        /// Gate conditions that must ALL hold before the batch fires.
        pub gates: Vec<ScheduleGate>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ScheduleGate {
        Idle,
        AcPower,
        ThermalOk,
        Cron,
    }

    /// The queue unit (A.1 `Job`). Reconciled to the binding contract.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct AgentJob {
        pub id: String,
        pub kind: JobKind,
        pub title: String,
        pub session_id: SessionId,
        pub run_id: Option<RunId>,
        pub spec: JobSpec,
        pub status: JobStatus,
        pub priority: PriorityClass,
        pub created_by: CreatedBy,
        /// Fan-out children / batch members point at their parent (the job DAG).
        pub parent_job: Option<String>,
        /// Job-level DAG edges (this runs after these complete).
        pub dependencies: Vec<String>,
        pub isolation: Isolation,
        /// The commit each worktree forks from.
        pub base_ref: String,
        pub schedule: Option<JobSchedule>,
        pub attempts: u32,
        pub max_attempts: u32,
        /// Blob ref to the run's outcome summary (set on completion).
        pub result_ref: Option<BlobRef>,
        pub created_at_ms: u64,
        pub admitted_at_ms: Option<u64>,
        pub finished_at_ms: Option<u64>,
        pub schema_version: u16,
    }

    impl AgentJob {
        pub fn new(objective: impl Into<String>, priority: PriorityClass) -> Self {
            let objective = objective.into();
            Self {
                id: new_job_id(),
                kind: JobKind::AgentRun,
                title: truncate_title(&objective),
                session_id: SessionId::new(),
                run_id: None,
                spec: JobSpec { objective, ..JobSpec::default() },
                status: JobStatus::Queued,
                priority,
                created_by: CreatedBy::User,
                parent_job: None,
                dependencies: Vec::new(),
                isolation: Isolation::Worktree,
                base_ref: "HEAD".to_string(),
                schedule: None,
                attempts: 0,
                max_attempts: 1,
                result_ref: None,
                created_at_ms: now_ms(),
                admitted_at_ms: None,
                finished_at_ms: None,
                schema_version: JOB_SCHEMA_VERSION,
            }
        }

        pub fn with_kind(mut self, kind: JobKind) -> Self {
            self.kind = kind;
            self
        }

        pub fn with_session(mut self, session: SessionId) -> Self {
            self.session_id = session;
            self
        }

        pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
            self.parent_job = Some(parent.into());
            self
        }

        pub fn depends_on(mut self, deps: impl IntoIterator<Item = String>) -> Self {
            self.dependencies = deps.into_iter().collect();
            self
        }

        pub fn with_resources(mut self, resources: ResourceRequest) -> Self {
            self.spec.required_resources = resources;
            self
        }

        pub fn with_concurrency_class(mut self, class: ConcurrencyClass) -> Self {
            self.spec.required_resources.concurrency_class = class;
            self
        }

        pub fn with_base_ref(mut self, base_ref: impl Into<String>) -> Self {
            self.base_ref = base_ref.into();
            self
        }

        pub fn concurrency_class(&self) -> ConcurrencyClass {
            self.spec.required_resources.concurrency_class
        }
    }

    fn new_job_id() -> String {
        format!("job_{}", ulid::Ulid::new())
    }

    fn truncate_title(objective: &str) -> String {
        let first_line = objective.lines().next().unwrap_or(objective);
        if first_line.chars().count() <= 80 {
            first_line.to_string()
        } else {
            let mut t: String = first_line.chars().take(77).collect();
            t.push_str("...");
            t
        }
    }

    /// The in-memory job graph: a fast index that is also a projection of the event
    /// log. Mutations go through `enqueue_logged`/`set_status_logged`, which append
    /// the durable `job.*` event AND update the index; `project_from` rebuilds the
    /// whole index from the log on startup (crash recovery, P8).
    #[derive(Debug, Default)]
    pub struct JobGraph {
        jobs: RwLock<BTreeMap<String, AgentJob>>,
    }

    impl JobGraph {
        pub fn new() -> Self {
            Self::default()
        }

        // --- pure in-memory index ops (used by the projection fold + tests) ---

        pub fn enqueue(&self, job: AgentJob) {
            self.jobs.write().insert(job.id.clone(), job);
        }

        pub fn get(&self, id: &str) -> Option<AgentJob> {
            self.jobs.read().get(id).cloned()
        }

        pub fn all(&self) -> Vec<AgentJob> {
            self.jobs.read().values().cloned().collect()
        }

        pub fn len(&self) -> usize {
            self.jobs.read().len()
        }

        pub fn is_empty(&self) -> bool {
            self.jobs.read().is_empty()
        }

        pub fn set_status(&self, id: &str, status: JobStatus) {
            if let Some(job) = self.jobs.write().get_mut(id) {
                apply_status_side_effects(job, status);
            }
        }

        pub fn set_run_id(&self, id: &str, run_id: RunId) {
            if let Some(job) = self.jobs.write().get_mut(id) {
                job.run_id = Some(run_id);
            }
        }

        pub fn set_result_ref(&self, id: &str, result_ref: BlobRef) {
            if let Some(job) = self.jobs.write().get_mut(id) {
                job.result_ref = Some(result_ref);
            }
        }

        /// Count live jobs (admitted/running/merging) in a concurrency class — the
        /// scheduler's two-pool occupancy read (§4.5.1).
        pub fn live_in_class(&self, class: ConcurrencyClass) -> u32 {
            self.jobs.read().values().filter(|j| j.status.is_live() && j.concurrency_class() == class).count() as u32
        }

        /// Ready jobs: queued/preempted with all deps `Done`, ordered by priority
        /// then least-recently-created (fair-share-within-class proxy). Kahn-style
        /// ready-set (§4.5.2).
        pub fn ready_jobs(&self) -> Vec<AgentJob> {
            let jobs = self.jobs.read();
            let completed: BTreeSet<_> =
                jobs.values().filter(|job| job.status == JobStatus::Done).map(|job| job.id.clone()).collect();
            let mut ready: Vec<_> = jobs
                .values()
                .filter(|job| matches!(job.status, JobStatus::Queued | JobStatus::Preempted))
                .filter(|job| job.dependencies.iter().all(|dep| completed.contains(dep)))
                .cloned()
                .collect();
            ready.sort_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| a.created_at_ms.cmp(&b.created_at_ms))
                    .then_with(|| a.id.cmp(&b.id))
            });
            ready
        }

        /// The lowest-priority live job at or below `floor` priority — the preemption
        /// victim selector (§4.6.2 step 2). Returns the *weakest* candidate.
        pub fn lowest_priority_live_below(&self, floor: PriorityClass) -> Option<AgentJob> {
            self.jobs
                .read()
                .values()
                .filter(|j| j.status.is_live() && j.priority > floor)
                .filter(|j| j.concurrency_class() == ConcurrencyClass::Model)
                .max_by(|a, b| a.priority.cmp(&b.priority).then_with(|| b.created_at_ms.cmp(&a.created_at_ms)))
                .cloned()
        }

        pub fn has_waiting(&self, priority: PriorityClass) -> bool {
            self.jobs.read().values().any(|j| j.priority == priority && matches!(j.status, JobStatus::Queued))
        }

        /// Cycle detection over the dependency DAG (rejects cyclic graphs at enqueue,
        /// §4.5.2 / F1). Real DFS with a recursion stack.
        pub fn has_cycle(&self) -> bool {
            let jobs = self.jobs.read();
            let mut color: BTreeMap<&str, u8> = BTreeMap::new(); // 0=white 1=gray 2=black
            for id in jobs.keys() {
                if dfs_cycle(id.as_str(), &jobs, &mut color) {
                    return true;
                }
            }
            false
        }

        // --- durable, event-logged ops (the queue-as-projection path, P8) ---

        /// Enqueue a job AND append `job.enqueued` carrying the FULL job record (so
        /// `project_from` can rehydrate it verbatim). Rejects a cyclic graph (§4.5.2).
        pub async fn enqueue_logged(&self, log: &DynEventLog, job: AgentJob) -> Result<()> {
            {
                let mut guard = self.jobs.write();
                guard.insert(job.id.clone(), job.clone());
            }
            if self.has_cycle() {
                self.jobs.write().remove(&job.id);
                return Err(hide_core::HideError::InvalidState(format!(
                    "job {} would create a dependency cycle",
                    job.id
                )));
            }
            log.append(job_event(&job, "job.enqueued", EventClass::Neither, json!({ "job_id": job.id, "job": job })))
                .await?;
            Ok(())
        }

        /// Transition status AND append the matching `job.*` event (A.5). The index
        /// and the log move together, so a replay reconstructs the same state.
        pub async fn set_status_logged(
            &self,
            log: &DynEventLog,
            id: &str,
            status: JobStatus,
            extra: Value,
        ) -> Result<()> {
            let job = {
                let mut guard = self.jobs.write();
                match guard.get_mut(id) {
                    Some(job) => {
                        apply_status_side_effects(job, status);
                        job.clone()
                    }
                    None => {
                        return Err(hide_core::HideError::NotFound(format!("job {id}")));
                    }
                }
            };
            let (kind, class) = status_event_kind(status);
            let mut payload = json!({ "job_id": id, "status": status });
            merge_payload(&mut payload, extra);
            log.append(job_event(&job, kind, class, payload)).await?;
            Ok(())
        }

        /// Rebuild the entire graph from the event log (crash recovery / replay, P8).
        /// Folds `job.enqueued` → insert, `job.*` status events → transition. Folding
        /// records data; it never re-launches a run (T3).
        pub fn project_from(&self, events: &[Event]) {
            let mut guard = self.jobs.write();
            guard.clear();
            for event in events {
                match event.kind.as_str() {
                    "job.enqueued" => {
                        if let Some(job) = event.payload_as::<EnqueuedSnapshot>() {
                            if let Some(full) = job.job {
                                guard.insert(full.id.clone(), full);
                            }
                        }
                    }
                    kind if kind.starts_with("job.") => {
                        if let Some(p) = event.payload_as::<StatusPatch>() {
                            if let (Some(job_id), Some(status)) = (p.job_id, p.status) {
                                if let Some(job) = guard.get_mut(&job_id) {
                                    apply_status_side_effects(job, status);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Apply the bookkeeping side effects of a status transition (timestamps, attempt
    /// counter) consistently whether the transition came from the index path or the
    /// projection fold.
    fn apply_status_side_effects(job: &mut AgentJob, status: JobStatus) {
        match status {
            JobStatus::Admitted if job.admitted_at_ms.is_none() => {
                job.admitted_at_ms = Some(now_ms());
            }
            JobStatus::Running => {
                // First entry into Running counts as an attempt.
                if job.status != JobStatus::Running {
                    job.attempts += 1;
                }
            }
            s if s.is_terminal() && job.finished_at_ms.is_none() => {
                job.finished_at_ms = Some(now_ms());
            }
            _ => {}
        }
        job.status = status;
    }

    /// Map a status to the A.5 event kind + class.
    fn status_event_kind(status: JobStatus) -> (&'static str, EventClass) {
        match status {
            JobStatus::Admitted => ("job.admitted", EventClass::Neither),
            JobStatus::Running => ("job.started", EventClass::Neither),
            JobStatus::Preempted => ("job.preempted", EventClass::Action),
            JobStatus::Merging => ("job.merging", EventClass::Neither),
            JobStatus::Done | JobStatus::Failed | JobStatus::Cancelled => ("job.completed", EventClass::Observation),
            JobStatus::Paused => ("job.paused", EventClass::Neither),
            JobStatus::Queued => ("job.requeued", EventClass::Neither),
        }
    }

    fn job_event(job: &AgentJob, kind: &str, class: EventClass, payload: Value) -> NewEvent {
        let mut ev = NewEvent::of(job.session_id.clone(), EventSource::System, kind, payload).with_class(class);
        if let Some(run) = &job.run_id {
            ev = ev.with_run(run.clone());
        }
        ev
    }

    fn merge_payload(base: &mut Value, extra: Value) {
        if let (Some(base_obj), Value::Object(extra_obj)) = (base.as_object_mut(), extra) {
            for (k, v) in extra_obj {
                base_obj.insert(k, v);
            }
        }
    }

    /// The full job is embedded in the `job.enqueued` event so the projection can
    /// rehydrate it verbatim.
    #[derive(Debug, Serialize, Deserialize)]
    struct EnqueuedSnapshot {
        #[serde(default)]
        job: Option<AgentJob>,
    }

    #[derive(Debug, Deserialize)]
    struct StatusPatch {
        job_id: Option<String>,
        status: Option<JobStatus>,
    }

    fn dfs_cycle<'a>(id: &'a str, jobs: &'a BTreeMap<String, AgentJob>, color: &mut BTreeMap<&'a str, u8>) -> bool {
        match color.get(id) {
            Some(2) => return false, // fully explored
            Some(1) => return true,  // back-edge → cycle
            _ => {}
        }
        color.insert(id, 1);
        if let Some(job) = jobs.get(id) {
            for dep in &job.dependencies {
                // Resolve the dep against the keyset so dangling deps don't recurse.
                if let Some((dep_key, _)) = jobs.get_key_value(dep) {
                    if dfs_cycle(dep_key.as_str(), jobs, color) {
                        return true;
                    }
                }
            }
        }
        color.insert(id, 2);
        false
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::event::InMemoryEventLog;
        use std::sync::Arc;

        #[test]
        fn ready_jobs_respect_dependencies_and_priority() {
            let g = JobGraph::new();
            let a = AgentJob::new("a", PriorityClass::Normal);
            let a_id = a.id.clone();
            let b = AgentJob::new("b", PriorityClass::Interactive).depends_on([a_id.clone()]);
            g.enqueue(a);
            g.enqueue(b.clone());
            // b depends on a (not done) → only a is ready.
            let ready = g.ready_jobs();
            assert_eq!(ready.len(), 1);
            assert_eq!(ready[0].id, a_id);
            // Complete a → b becomes ready.
            g.set_status(&a_id, JobStatus::Done);
            let ready = g.ready_jobs();
            assert_eq!(ready.len(), 1);
            assert_eq!(ready[0].id, b.id);
        }

        #[test]
        fn cycle_detection_rejects_back_edge() {
            let g = JobGraph::new();
            let mut a = AgentJob::new("a", PriorityClass::Normal);
            let mut b = AgentJob::new("b", PriorityClass::Normal);
            a.dependencies = vec![b.id.clone()];
            b.dependencies = vec![a.id.clone()];
            g.enqueue(a);
            g.enqueue(b);
            assert!(g.has_cycle());
        }

        #[test]
        fn two_pool_occupancy_counts_per_class() {
            let g = JobGraph::new();
            let mut m = AgentJob::new("model", PriorityClass::Normal);
            m.status = JobStatus::Running;
            let mut c = AgentJob::new("cpu", PriorityClass::Normal).with_concurrency_class(ConcurrencyClass::CpuOnly);
            c.status = JobStatus::Running;
            g.enqueue(m);
            g.enqueue(c);
            assert_eq!(g.live_in_class(ConcurrencyClass::Model), 1);
            assert_eq!(g.live_in_class(ConcurrencyClass::CpuOnly), 1);
        }

        #[tokio::test]
        async fn queue_is_a_projection_of_the_event_log() {
            let log: DynEventLog = Arc::new(InMemoryEventLog::new());
            let g = JobGraph::new();
            let job = AgentJob::new("durable goal", PriorityClass::Normal);
            let id = job.id.clone();
            g.enqueue_logged(&log, job).await.unwrap();
            g.set_status_logged(&log, &id, JobStatus::Admitted, json!({})).await.unwrap();
            g.set_status_logged(&log, &id, JobStatus::Running, json!({})).await.unwrap();
            g.set_status_logged(&log, &id, JobStatus::Done, json!({ "result": "ok" })).await.unwrap();

            // Rebuild a fresh graph purely from the log → identical terminal state.
            let events = log.scan(None, None, None).await.unwrap();
            let rebuilt = JobGraph::new();
            rebuilt.project_from(&events);
            let job = rebuilt.get(&id).expect("job rehydrated from log");
            assert_eq!(job.status, JobStatus::Done);
            assert!(job.admitted_at_ms.is_some());
            assert!(job.finished_at_ms.is_some());
            assert_eq!(job.attempts, 1);
        }

        #[tokio::test]
        async fn enqueue_logged_rejects_cycle_and_emits_event() {
            let log: DynEventLog = Arc::new(InMemoryEventLog::new());
            let g = JobGraph::new();
            let a = AgentJob::new("a", PriorityClass::Normal);
            let a_id = a.id.clone();
            g.enqueue_logged(&log, a).await.unwrap();
            // A self-cycle via dependency on itself is rejected.
            let mut bad = AgentJob::new("bad", PriorityClass::Normal);
            bad.dependencies = vec![bad.id.clone()];
            assert!(g.enqueue_logged(&log, bad).await.is_err());
            // The valid job's enqueue event is in the log.
            let events = log.scan(None, None, None).await.unwrap();
            assert!(events.iter().any(|e| e.kind == "job.enqueued" && e.payload["job_id"] == a_id.as_str()));
        }
    }
}
#[rustfmt::skip]
pub mod remote {
    //! Workstation / remote mode — the wire protocol (bible ch.09 §4.9).
    //!
    //! A laptop thin-client drives a Mac-Studio agent server. The protocol is
    //! **ACP-shaped**: JSON-RPC 2.0 over a persistent WebSocket, session-centric,
    //! resumable — but the update payload is the **ch.01 Event envelope** (richer than
    //! ACP notifications). The server is authoritative; the client is a disposable
    //! view (P10).
    //!
    //! Reliability core (§4.9.4):
    //! - **Server-authoritative**: all state lives in the event log; the client holds
    //!   only a rebuildable projection.
    //! - **Durable sessions**: sessions persist server-side independent of the socket
    //!   (a batch survives client sleep/disconnect).
    //! - **Reconnect = `session/resume{from_seq}`**: the server replays `(from_seq,
    //!   head]` from the log — exactly-once by construction (events are immutable,
    //!   `seq`-ordered; replay re-applies recorded data, never re-fires effects, T3).
    //! - **Deny-first auth (P11)**: loopback-only by default; tokens only over
    //!   wss/loopback; a ch.10 capability grant rides on the token (we transport +
    //!   check presence; ch.10 defines the grant).
    //!
    //! The JSON-RPC framing + session/replay logic is unit-testable without a socket
    //! ([`RemoteSession`], [`dispatch`]); [`serve`] runs the real `tokio-tungstenite`
    //! loop, and an integration test binds a loopback WS to exercise the handshake +
    //! resume end-to-end.

    use hide_core::api::Intent;
    use hide_core::event::Event;
    use hide_core::ids::SessionId;
    use hide_core::persistence::DynEventLog;
    use parking_lot::Mutex;
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::net::SocketAddr;
    use std::sync::Arc;

    // ---------------------------------------------------------------------------
    // JSON-RPC 2.0 envelope (§4.9.2 / A.4)
    // ---------------------------------------------------------------------------

    /// A JSON-RPC 2.0 request from the client.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct JsonRpcRequest {
        pub jsonrpc: String,
        /// Request id (number or string); absent for notifications.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub id: Option<Value>,
        pub method: String,
        #[serde(default)]
        pub params: Value,
    }

    /// A JSON-RPC 2.0 response/notification from the server.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct JsonRpcResponse {
        pub jsonrpc: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub id: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub result: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub error: Option<JsonRpcError>,
        /// For server→client notifications (`hide/event`), the method is set and id
        /// is absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub method: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub params: Option<Value>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct JsonRpcError {
        pub code: i32,
        pub message: String,
    }

    impl JsonRpcResponse {
        pub fn ok(id: Option<Value>, result: Value) -> Self {
            Self { jsonrpc: "2.0".to_string(), id, result: Some(result), error: None, method: None, params: None }
        }

        pub fn err(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
            Self {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(JsonRpcError { code, message: message.into() }),
                method: None,
                params: None,
            }
        }

        /// A server→client `hide/event` notification carrying a ch.01 Event.
        pub fn event_notification(event: &Event) -> Self {
            Self {
                jsonrpc: "2.0".to_string(),
                id: None,
                result: None,
                error: None,
                method: Some("hide/event".to_string()),
                params: Some(serde_json::to_value(event).unwrap_or(Value::Null)),
            }
        }
    }

    // JSON-RPC standard error codes + our extensions.
    pub const ERR_PARSE: i32 = -32700;
    pub const ERR_INVALID_REQUEST: i32 = -32600;
    pub const ERR_METHOD_NOT_FOUND: i32 = -32601;
    pub const ERR_INVALID_PARAMS: i32 = -32602;
    pub const ERR_UNAUTHORIZED: i32 = -32001;
    pub const ERR_CAPABILITY: i32 = -32002;

    // ---------------------------------------------------------------------------
    // Auth posture (§4.9.3, references ch.10)
    // ---------------------------------------------------------------------------

    /// Deny-first remote auth policy (P11). Default: loopback only, token required.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RemoteAuthPolicy {
        pub loopback_only: bool,
        pub allow_ssh_tunnel: bool,
        pub token_required: bool,
        /// Accepted bearer tokens (device-paired). Empty = reject all when
        /// `token_required` (deny-first).
        #[serde(default)]
        pub accepted_tokens: Vec<String>,
    }

    impl Default for RemoteAuthPolicy {
        fn default() -> Self {
            Self { loopback_only: true, allow_ssh_tunnel: true, token_required: true, accepted_tokens: Vec::new() }
        }
    }

    impl RemoteAuthPolicy {
        /// Whether a peer at `addr` presenting `token` may connect. Loopback peers
        /// over an SSH tunnel may skip the token (the §4.9.3 "loopback + SSH" rule);
        /// non-loopback peers always need a valid token.
        pub fn authorize(&self, addr: &SocketAddr, token: Option<&str>) -> bool {
            let is_loopback = addr.ip().is_loopback();
            if self.loopback_only && !is_loopback {
                return false;
            }
            if !self.token_required {
                return true;
            }
            if is_loopback && self.allow_ssh_tunnel && token.is_none() {
                // Loopback (SSH-forwarded) is trusted without a token by policy.
                return true;
            }
            match token {
                Some(t) => self.accepted_tokens.iter().any(|a| a == t),
                None => false,
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Server-authoritative sessions (§4.9.4)
    // ---------------------------------------------------------------------------

    /// A server-side session. Persists independent of any connection (a batch keeps
    /// running while the laptop sleeps). The session id maps to a ch.01 `SessionId`
    /// in the event log; the client resumes by `from_seq`.
    #[derive(Debug, Clone)]
    pub struct RemoteSession {
        pub session_id: SessionId,
        /// The ch.10 capability grant carried by the client's token (opaque here;
        /// ch.10 validates it). Presence is enforced; semantics are ch.10's.
        pub capability_grant: Option<String>,
        pub transport: String,
    }

    /// The server's session registry (server-authoritative, P10).
    #[derive(Default)]
    pub struct SessionRegistry {
        sessions: Mutex<BTreeMap<String, RemoteSession>>,
    }

    impl SessionRegistry {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn open(&self, grant: Option<String>, transport: impl Into<String>) -> RemoteSession {
            let session =
                RemoteSession { session_id: SessionId::new(), capability_grant: grant, transport: transport.into() };
            self.sessions.lock().insert(session.session_id.0.clone(), session.clone());
            session
        }

        pub fn get(&self, session_id: &str) -> Option<RemoteSession> {
            self.sessions.lock().get(session_id).cloned()
        }

        pub fn len(&self) -> usize {
            self.sessions.lock().len()
        }

        pub fn is_empty(&self) -> bool {
            self.sessions.lock().is_empty()
        }
    }

    // ---------------------------------------------------------------------------
    // The intent sink (how the server applies a client intent)
    // ---------------------------------------------------------------------------

    /// The server forwards client intents to this sink (the host wires it to the
    /// kernel/fleet). Decoupled so the protocol is testable without a backend.
    pub trait IntentSink: Send + Sync {
        /// Apply an intent in a session; return the event seq it produced (for the
        /// ack). The default test sink just records intents.
        fn submit(&self, session: &RemoteSession, intent: Intent) -> u64;
    }

    /// A recording sink for tests: stores received intents, returns increasing seqs.
    #[derive(Default)]
    pub struct RecordingSink {
        pub received: Mutex<Vec<Intent>>,
        next_seq: Mutex<u64>,
    }

    impl IntentSink for RecordingSink {
        fn submit(&self, _session: &RemoteSession, intent: Intent) -> u64 {
            self.received.lock().push(intent);
            let mut s = self.next_seq.lock();
            *s += 1;
            *s
        }
    }

    /// The dependencies a dispatch needs: the event log (for `from_seq` replay), the
    /// session registry, the intent sink, and the auth-derived capability grant.
    pub struct RemoteContext {
        pub log: DynEventLog,
        pub sessions: Arc<SessionRegistry>,
        pub sink: Arc<dyn IntentSink>,
        pub transport: String,
        pub grant: Option<String>,
    }

    /// Dispatch one JSON-RPC request against the server state, returning the response
    /// plus any backlog events to stream (for `session/resume`). This is the pure
    /// protocol core — `serve` wraps it with the socket.
    pub async fn dispatch(ctx: &RemoteContext, req: JsonRpcRequest) -> (JsonRpcResponse, Vec<Event>) {
        if req.jsonrpc != "2.0" {
            return (JsonRpcResponse::err(req.id, ERR_INVALID_REQUEST, "jsonrpc must be \"2.0\""), Vec::new());
        }
        match req.method.as_str() {
            // session/new → open a server-side session (ACP-style handshake).
            "session/new" => {
                let session = ctx.sessions.open(ctx.grant.clone(), ctx.transport.clone());
                (
                    JsonRpcResponse::ok(
                        req.id,
                        json!({ "session": session.session_id.0, "head_seq": head_seq(ctx).await }),
                    ),
                    Vec::new(),
                )
            }
            // session/resume{from_seq} → replay (from_seq, head] (§4.9.4).
            "session/resume" => {
                let session_id = req.params.get("session").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let from_seq = req.params.get("from_seq").and_then(|v| v.as_u64());
                let Some(session) = ctx.sessions.get(&session_id) else {
                    return (JsonRpcResponse::err(req.id, ERR_INVALID_PARAMS, "unknown session"), Vec::new());
                };
                let backlog = ctx.log.scan(Some(session.session_id.clone()), from_seq, None).await.unwrap_or_default();
                let replayed = backlog.len();
                (
                    JsonRpcResponse::ok(
                        req.id,
                        json!({ "session": session_id, "from_seq": from_seq, "replayed_n": replayed }),
                    ),
                    backlog,
                )
            }
            // hide/intent → apply a client intent (ch.01 Wire A over the wire).
            "hide/intent" => {
                let session_id = req.params.get("session").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let Some(session) = ctx.sessions.get(&session_id) else {
                    return (JsonRpcResponse::err(req.id, ERR_INVALID_PARAMS, "unknown session"), Vec::new());
                };
                // Every remote intent is checked for a capability grant before it acts
                // (ch.01 T4; ch.10 validates the grant itself).
                if session.capability_grant.is_none() && ctx.grant.is_none() {
                    // No grant at all → reject (deny ambient authority, P11).
                    return (
                        JsonRpcResponse::err(req.id, ERR_CAPABILITY, "no capability grant on session"),
                        Vec::new(),
                    );
                }
                let intent: Result<Intent, _> =
                    serde_json::from_value(req.params.get("intent").cloned().unwrap_or(Value::Null));
                match intent {
                    Ok(intent) => {
                        let seq = ctx.sink.submit(&session, intent);
                        (JsonRpcResponse::ok(req.id, json!({ "accepted": true, "event_seq": seq })), Vec::new())
                    }
                    Err(e) => {
                        (JsonRpcResponse::err(req.id, ERR_INVALID_PARAMS, format!("bad intent: {e}")), Vec::new())
                    }
                }
            }
            "ping" => (JsonRpcResponse::ok(req.id, json!({ "pong": true })), Vec::new()),
            other => {
                (JsonRpcResponse::err(req.id, ERR_METHOD_NOT_FOUND, format!("unknown method: {other}")), Vec::new())
            }
        }
    }

    async fn head_seq(ctx: &RemoteContext) -> u64 {
        ctx.log.scan(None, None, None).await.ok().and_then(|e| e.last().map(|ev| ev.seq)).unwrap_or(0)
    }

    // ---------------------------------------------------------------------------
    // The WebSocket server (§4.9.2) — real tokio-tungstenite loop
    // ---------------------------------------------------------------------------

    /// Configuration for the agent server.
    #[derive(Debug, Clone)]
    pub struct ServerConfig {
        pub bind: SocketAddr,
        pub auth: RemoteAuthPolicy,
    }

    impl Default for ServerConfig {
        fn default() -> Self {
            Self {
                // Loopback-default (P11). 0 = OS-assigned port (test-friendly).
                bind: "127.0.0.1:0".parse().unwrap(),
                auth: RemoteAuthPolicy::default(),
            }
        }
    }

    /// A handle to a running server: its bound address + a shutdown signal.
    pub struct ServerHandle {
        pub local_addr: SocketAddr,
        shutdown: tokio::sync::watch::Sender<bool>,
    }

    impl ServerHandle {
        pub fn shutdown(&self) {
            let _ = self.shutdown.send(true);
        }
    }

    /// Bind the agent server and accept WebSocket connections, each speaking JSON-RPC
    /// 2.0. Returns once bound (the accept loop runs as a detached task). The server
    /// is authoritative and keeps running when a client disconnects (§4.9.4).
    pub async fn serve(
        config: ServerConfig,
        log: DynEventLog,
        sessions: Arc<SessionRegistry>,
        sink: Arc<dyn IntentSink>,
    ) -> std::io::Result<ServerHandle> {
        let listener = tokio::net::TcpListener::bind(config.bind).await?;
        let local_addr = listener.local_addr()?;
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);

        let auth = config.auth.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() { break; }
                    }
                    accepted = listener.accept() => {
                        let Ok((stream, peer)) = accepted else { continue; };
                        // Loopback/auth gate at the TCP layer (P11). Token auth would
                        // ride the WS handshake headers; loopback-default trusts the
                        // SSH-forwarded peer per policy.
                        if !auth.authorize(&peer, None) {
                            continue;
                        }
                        let log = log.clone();
                        let sessions = sessions.clone();
                        let sink = sink.clone();
                        tokio::spawn(async move {
                            let _ = handle_connection(stream, peer, log, sessions, sink).await;
                        });
                    }
                }
            }
        });

        Ok(ServerHandle { local_addr, shutdown: shutdown_tx })
    }

    async fn handle_connection(
        stream: tokio::net::TcpStream,
        peer: SocketAddr,
        log: DynEventLog,
        sessions: Arc<SessionRegistry>,
        sink: Arc<dyn IntentSink>,
    ) -> Result<(), tokio_tungstenite::tungstenite::Error> {
        use futures::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message;

        let ws = tokio_tungstenite::accept_async(stream).await?;
        let (mut write, mut read) = ws.split();

        let ctx = RemoteContext {
            log,
            sessions,
            sink,
            // Loopback peers are treated as SSH-tunnel transport; a grant is implied
            // by the trusted-loopback policy so intents are accepted in this mode.
            transport: if peer.ip().is_loopback() { "ssh-loopback".to_string() } else { "wss".to_string() },
            grant: Some("loopback-implicit".to_string()),
        };

        while let Some(msg) = read.next().await {
            let msg = msg?;
            match msg {
                Message::Text(text) => {
                    let req: JsonRpcRequest = match serde_json::from_str(&text) {
                        Ok(r) => r,
                        Err(e) => {
                            let resp = JsonRpcResponse::err(None, ERR_PARSE, format!("parse: {e}"));
                            write.send(Message::Text(serde_json::to_string(&resp).unwrap())).await?;
                            continue;
                        }
                    };
                    let (resp, backlog) = dispatch(&ctx, req).await;
                    write.send(Message::Text(serde_json::to_string(&resp).unwrap())).await?;
                    // Stream any replay backlog as `hide/event` notifications.
                    for event in backlog {
                        let note = JsonRpcResponse::event_notification(&event);
                        write.send(Message::Text(serde_json::to_string(&note).unwrap())).await?;
                    }
                }
                Message::Close(_) => break,
                Message::Ping(p) => write.send(Message::Pong(p)).await?,
                _ => {}
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::event::{InMemoryEventLog, NewEvent};
        use hide_core::ids::SessionId;

        fn ctx(log: DynEventLog) -> RemoteContext {
            RemoteContext {
                log,
                sessions: Arc::new(SessionRegistry::new()),
                sink: Arc::new(RecordingSink::default()),
                transport: "test".to_string(),
                grant: Some("cap-test".to_string()),
            }
        }

        #[test]
        fn auth_denies_non_loopback_without_token() {
            let policy = RemoteAuthPolicy::default();
            let lan: SocketAddr = "192.168.1.5:9000".parse().unwrap();
            assert!(!policy.authorize(&lan, None));
            let loop_addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
            // Loopback + SSH-tunnel trusted without a token by default.
            assert!(policy.authorize(&loop_addr, None));
            // LAN with a valid paired token is allowed only if loopback_only is off.
            let mut lan_policy = RemoteAuthPolicy {
                loopback_only: false,
                accepted_tokens: vec!["good".to_string()],
                ..Default::default()
            };
            assert!(lan_policy.authorize(&lan, Some("good")));
            assert!(!lan_policy.authorize(&lan, Some("bad")));
            lan_policy.accepted_tokens.clear();
            assert!(!lan_policy.authorize(&lan, Some("good")));
        }

        #[tokio::test]
        async fn session_new_then_intent_acks_with_seq() {
            let log: DynEventLog = Arc::new(InMemoryEventLog::new());
            let ctx = ctx(log);
            // open
            let (resp, _) = dispatch(
                &ctx,
                JsonRpcRequest {
                    jsonrpc: "2.0".to_string(),
                    id: Some(json!(1)),
                    method: "session/new".to_string(),
                    params: json!({}),
                },
            )
            .await;
            let session = resp.result.unwrap()["session"].as_str().unwrap().to_string();
            // intent
            let intent = Intent::SubmitTurn {
                session_id: SessionId::from(session.as_str()),
                text: "add JWT refresh".to_string(),
                attachments: vec![],
            };
            let (resp, _) = dispatch(
                &ctx,
                JsonRpcRequest {
                    jsonrpc: "2.0".to_string(),
                    id: Some(json!(2)),
                    method: "hide/intent".to_string(),
                    params: json!({ "session": session, "intent": intent }),
                },
            )
            .await;
            assert_eq!(resp.result.unwrap()["accepted"], true);
        }

        #[tokio::test]
        async fn session_resume_replays_from_seq() {
            let log: DynEventLog = Arc::new(InMemoryEventLog::new());
            let ctx = ctx(log.clone());
            // Open a session, then append 3 events to its session id.
            let (resp, _) = dispatch(
                &ctx,
                JsonRpcRequest {
                    jsonrpc: "2.0".to_string(),
                    id: Some(json!(1)),
                    method: "session/new".to_string(),
                    params: json!({}),
                },
            )
            .await;
            let session_str = resp.result.unwrap()["session"].as_str().unwrap().to_string();
            let session = ctx.sessions.get(&session_str).unwrap().session_id;
            for i in 0..3 {
                log.append(NewEvent::system(session.clone(), "agent.phase", json!({ "n": i }))).await.unwrap();
            }
            // Resume from seq 1 → replays seqs 2,3 (after_seq is exclusive).
            let (resp, backlog) = dispatch(
                &ctx,
                JsonRpcRequest {
                    jsonrpc: "2.0".to_string(),
                    id: Some(json!(9)),
                    method: "session/resume".to_string(),
                    params: json!({ "session": session_str, "from_seq": 1 }),
                },
            )
            .await;
            assert_eq!(resp.result.unwrap()["replayed_n"], 2);
            assert_eq!(backlog.len(), 2);
            assert!(backlog.iter().all(|e| e.seq > 1));
        }

        #[tokio::test]
        async fn unknown_method_is_method_not_found() {
            let log: DynEventLog = Arc::new(InMemoryEventLog::new());
            let ctx = ctx(log);
            let (resp, _) = dispatch(
                &ctx,
                JsonRpcRequest {
                    jsonrpc: "2.0".to_string(),
                    id: Some(json!(1)),
                    method: "bogus/method".to_string(),
                    params: json!({}),
                },
            )
            .await;
            assert_eq!(resp.error.unwrap().code, ERR_METHOD_NOT_FOUND);
        }

        #[tokio::test]
        async fn end_to_end_websocket_handshake_intent_and_resume() {
            use futures::{SinkExt, StreamExt};
            use tokio_tungstenite::tungstenite::Message;

            let log: DynEventLog = Arc::new(InMemoryEventLog::new());
            let sessions = Arc::new(SessionRegistry::new());
            let sink = Arc::new(RecordingSink::default());
            let handle = serve(ServerConfig::default(), log.clone(), sessions.clone(), sink.clone()).await.unwrap();

            let url = format!("ws://{}", handle.local_addr);
            let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

            // session/new
            ws.send(Message::Text(
                json!({ "jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {} }).to_string(),
            ))
            .await
            .unwrap();
            let reply = ws.next().await.unwrap().unwrap();
            let v: Value = serde_json::from_str(reply.to_text().unwrap()).unwrap();
            let session = v["result"]["session"].as_str().unwrap().to_string();

            // hide/intent
            let intent = Intent::SubmitTurn {
                session_id: SessionId::from(session.as_str()),
                text: "do it".to_string(),
                attachments: vec![],
            };
            ws.send(Message::Text(
                json!({ "jsonrpc": "2.0", "id": 2, "method": "hide/intent",
                        "params": { "session": session, "intent": intent } })
                .to_string(),
            ))
            .await
            .unwrap();
            let reply = ws.next().await.unwrap().unwrap();
            let v: Value = serde_json::from_str(reply.to_text().unwrap()).unwrap();
            assert_eq!(v["result"]["accepted"], true);
            assert_eq!(sink.received.lock().len(), 1);

            // Append an event to the session, then resume → server streams it back.
            let sid = sessions.get(&session).unwrap().session_id;
            log.append(NewEvent::system(sid, "agent.phase", json!({ "x": 1 }))).await.unwrap();
            ws.send(Message::Text(
                json!({ "jsonrpc": "2.0", "id": 3, "method": "session/resume",
                        "params": { "session": session, "from_seq": 0 } })
                .to_string(),
            ))
            .await
            .unwrap();
            // First the resume result...
            let reply = ws.next().await.unwrap().unwrap();
            let v: Value = serde_json::from_str(reply.to_text().unwrap()).unwrap();
            assert!(v["result"]["replayed_n"].as_u64().unwrap() >= 1);
            // ...then a hide/event notification carrying the replayed event.
            let note = ws.next().await.unwrap().unwrap();
            let v: Value = serde_json::from_str(note.to_text().unwrap()).unwrap();
            assert_eq!(v["method"], "hide/event");

            handle.shutdown();
        }
    }
}
#[rustfmt::skip]
pub mod resources {
    //! The resource probe (bible ch.09 §4.6.1 `resources.rs`).
    //!
    //! The Governor admits on *physical* signals (P1). This module reads them:
    //! - **Free RAM** via a light OS read (`vm_stat` on macOS, `/proc/meminfo` on
    //!   Linux) — no privilege, no heavy `sysinfo` dep.
    //! - **The thermal proxy**: macOS hands no clean "you are throttling" bit, so the
    //!   Governor derives it from the runtime's own throughput — a sustained
    //!   `dec_tps` drop vs a per-model baseline is read as throttle (§4.6.1). The
    //!   runtime's throughput number is the thermometer.
    //! - **`max_batch_size`/active slots**: supplied by the runtime (the host reads
    //!   `/metrics`); the probe takes them as inputs and returns a full snapshot.

    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};

    /// A coarse thermal classification derived from the throughput proxy + OS hints.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ThermalState {
        Nominal,
        Fair,
        Serious,
        Critical,
    }

    /// Live machine state sampled ~1 Hz (A.2 `GovernorState`).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ResourceSnapshot {
        pub free_memory_mb: u64,
        /// The runtime's `max_batch_size` (the hard model-concurrency ceiling).
        pub max_generation_slots: u32,
        pub active_generation_slots: u32,
        pub thermal: ThermalState,
        /// Current decode throughput (tok/s) from `/metrics`.
        pub dec_tps_now: f32,
        /// Per-model baseline throughput (cool-machine reference).
        pub dec_tps_baseline: f32,
        pub battery_percent: Option<u8>,
        pub on_ac_power: bool,
        pub idle: bool,
    }

    impl ResourceSnapshot {
        /// A conservative idle default (used before the first probe).
        pub fn idle() -> Self {
            Self {
                free_memory_mb: 0,
                max_generation_slots: 1,
                active_generation_slots: 0,
                thermal: ThermalState::Nominal,
                dec_tps_now: 0.0,
                dec_tps_baseline: 0.0,
                battery_percent: None,
                on_ac_power: true,
                idle: true,
            }
        }

        /// The thermal proxy: the fractional drop of current dec_tps vs baseline. A
        /// drop ≥ the envelope's throttle threshold signals throttling/contention.
        pub fn thermal_drop_pct(&self) -> f32 {
            if self.dec_tps_baseline <= 0.0 || self.dec_tps_now <= 0.0 {
                return 0.0;
            }
            let drop = (self.dec_tps_baseline - self.dec_tps_now) / self.dec_tps_baseline;
            drop.clamp(0.0, 1.0)
        }
    }

    /// Reads physical machine state. The default [`OsResourceProbe`] reads the real
    /// OS; tests use [`FixedResourceProbe`].
    #[async_trait]
    pub trait ResourceProbe: Send + Sync {
        /// Sample free RAM (+ derive thermal from the supplied throughput) and fold
        /// in the runtime-supplied slot counts. `max_slots`/`active` come from the
        /// runtime's `/metrics`; `dec_tps_now`/`baseline` likewise — the probe owns
        /// RAM + power, the host owns runtime telemetry.
        async fn snapshot(&self, max_slots: u32, active: u32) -> ResourceSnapshot;
    }

    /// The real probe: reads free RAM from the OS and power state where available.
    /// Throughput/slots are passed through (the host reads `/metrics`).
    #[derive(Debug, Clone, Default)]
    pub struct OsResourceProbe {
        /// Per-model baseline dec_tps for the thermal proxy (set by the host once a
        /// cool-machine baseline is known).
        pub dec_tps_baseline: f32,
        /// Latest observed dec_tps (the host updates this from `/metrics`).
        pub dec_tps_now: f32,
    }

    #[async_trait]
    impl ResourceProbe for OsResourceProbe {
        async fn snapshot(&self, max_slots: u32, active: u32) -> ResourceSnapshot {
            let free_memory_mb = read_free_memory_mb().unwrap_or(0);
            let mut snap = ResourceSnapshot {
                free_memory_mb,
                max_generation_slots: max_slots,
                active_generation_slots: active,
                thermal: ThermalState::Nominal,
                dec_tps_now: self.dec_tps_now,
                dec_tps_baseline: self.dec_tps_baseline,
                battery_percent: None,
                on_ac_power: true,
                idle: active == 0,
            };
            // Classify thermal from the throughput-derived proxy.
            let drop = snap.thermal_drop_pct();
            snap.thermal = if drop >= 0.40 {
                ThermalState::Critical
            } else if drop >= 0.25 {
                ThermalState::Serious
            } else if drop >= 0.15 {
                ThermalState::Fair
            } else {
                ThermalState::Nominal
            };
            snap
        }
    }

    /// A fixed probe for tests / deterministic scheduling.
    #[derive(Debug, Clone)]
    pub struct FixedResourceProbe {
        pub snapshot: ResourceSnapshot,
    }

    #[async_trait]
    impl ResourceProbe for FixedResourceProbe {
        async fn snapshot(&self, max_slots: u32, active: u32) -> ResourceSnapshot {
            let mut s = self.snapshot.clone();
            s.max_generation_slots = max_slots;
            s.active_generation_slots = active;
            s
        }
    }

    /// Read free physical memory in MB without a heavy dependency. Returns `None` on
    /// an unsupported OS or a parse failure (the Governor then treats RAM as
    /// unknown-but-present; the host can supply a fixed probe instead).
    pub fn read_free_memory_mb() -> Option<u64> {
        #[cfg(target_os = "macos")]
        {
            read_free_memory_mb_macos()
        }
        #[cfg(target_os = "linux")]
        {
            read_free_memory_mb_linux()
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            None
        }
    }

    #[cfg(target_os = "macos")]
    fn read_free_memory_mb_macos() -> Option<u64> {
        // `vm_stat` reports page counts; multiply free+inactive+speculative by page
        // size. Inactive pages are reclaimable, so they count as effectively free for
        // admission headroom (matching Activity Monitor's "available" notion).
        use std::process::Command;
        let out = Command::new("vm_stat").output().ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut page_size: u64 = 4096;
        let mut free_pages: u64 = 0;
        let mut inactive_pages: u64 = 0;
        let mut speculative_pages: u64 = 0;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics:") {
                // The header sometimes carries "(page size of N bytes)".
                if let Some(idx) = rest.find("page size of ") {
                    let tail = &rest[idx + "page size of ".len()..];
                    if let Some(num) = tail.split_whitespace().next() {
                        if let Ok(n) = num.parse::<u64>() {
                            page_size = n;
                        }
                    }
                }
            } else if let Some(v) = parse_vm_stat_line(line, "Pages free:") {
                free_pages = v;
            } else if let Some(v) = parse_vm_stat_line(line, "Pages inactive:") {
                inactive_pages = v;
            } else if let Some(v) = parse_vm_stat_line(line, "Pages speculative:") {
                speculative_pages = v;
            }
        }
        let total_free_pages = free_pages + inactive_pages + speculative_pages;
        Some(total_free_pages.saturating_mul(page_size) / (1024 * 1024))
    }

    #[cfg(target_os = "macos")]
    fn parse_vm_stat_line(line: &str, prefix: &str) -> Option<u64> {
        let rest = line.trim().strip_prefix(prefix)?;
        let digits: String = rest.trim().chars().filter(|c| c.is_ascii_digit()).collect();
        digits.parse::<u64>().ok()
    }

    #[cfg(target_os = "linux")]
    fn read_free_memory_mb_linux() -> Option<u64> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        // Prefer MemAvailable (kernel's reclaimable estimate); fall back to MemFree.
        let mut available_kb: Option<u64> = None;
        let mut free_kb: Option<u64> = None;
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("MemAvailable:") {
                available_kb = v.split_whitespace().next().and_then(|n| n.parse().ok());
            } else if let Some(v) = line.strip_prefix("MemFree:") {
                free_kb = v.split_whitespace().next().and_then(|n| n.parse().ok());
            }
        }
        available_kb.or(free_kb).map(|kb| kb / 1024)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn thermal_drop_is_fractional_and_clamped() {
            let mut s = ResourceSnapshot::idle();
            s.dec_tps_baseline = 100.0;
            s.dec_tps_now = 75.0;
            assert!((s.thermal_drop_pct() - 0.25).abs() < 1e-6);
            // No baseline → 0 (don't false-throttle on cold start).
            s.dec_tps_baseline = 0.0;
            assert_eq!(s.thermal_drop_pct(), 0.0);
        }

        #[tokio::test]
        async fn os_probe_classifies_thermal_from_proxy() {
            let probe = OsResourceProbe {
                dec_tps_baseline: 40.0,
                dec_tps_now: 22.0, // 45% drop → Critical.
            };
            let snap = probe.snapshot(4, 1).await;
            assert_eq!(snap.thermal, ThermalState::Critical);
            assert_eq!(snap.max_generation_slots, 4);
        }

        #[tokio::test]
        async fn os_probe_reads_some_memory_on_this_platform() {
            // On macOS/Linux this returns a real number; on others it's 0 (None path).
            let probe = OsResourceProbe::default();
            let snap = probe.snapshot(1, 0).await;
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            assert!(snap.free_memory_mb > 0, "expected a real free-RAM read");
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            let _ = snap;
        }
    }
}
#[rustfmt::skip]
pub mod scheduler {
    //! The scheduler & the machine-wide resource Governor (bible ch.09 §4.6).
    //!
    //! This is the chapter's systems core. The [`FleetGovernor`] extends ch.02's
    //! per-run Governor to the whole box's physical envelope (P1/P6): it admits on
    //! **RAM headroom**, the runtime's **model-batch ceiling**, and a **thermal
    //! proxy** (a dec_tps drop), and it *is* the runaway-swarm circuit-breaker
    //! (§4.6.4) via an EWMA on spawn rate.
    //!
    //! [`FleetScheduler::schedule_tick`] is the ~1 Hz loop: refresh the probe, shrink
    //! the ceiling under thermal/RAM pressure, preempt a batch run for a waiting
    //! interactive job (checkpoint-and-yield, never kill), then admit in priority
    //! order under the two-pool (Model vs CpuOnly) ceiling and `spawn_run` the
    //! admitted jobs — actually launching a [`hide_kernel::AgentKernel`] run.

    use crate::queue::{ConcurrencyClass, JobStatus, PriorityClass, ResourceRequest};
    use crate::resources::{ResourceProbe, ResourceSnapshot, ThermalState};
    use serde::{Deserialize, Serialize};

    /// The machine-wide ceilings (A.2 `ResourceEnvelope`).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ResourceEnvelope {
        /// Never admit below this much free unified RAM (no swap, P1).
        pub ram_headroom_mb_min: u64,
        /// Concurrent Model-class runs ≤ runtime `max_batch_size`.
        pub max_model_runs: u32,
        /// Concurrent CpuOnly jobs (default physical_cores − 1).
        pub max_cpu_runs: u32,
        pub max_worktrees: u32,
        pub max_ports_leased: u32,
        /// Thermal backoff thresholds (dec_tps-drop fractions, 0..1).
        pub thermal_warn_pct: f32,
        pub thermal_throttle_pct: f32,
        /// Spawn-rate circuit-breaker: trip if EWMA jobs/min exceeds this (§4.6.4).
        pub max_spawns_per_min: f32,
        /// Preemption enabled, and the weakest priority that may *not* be preempted.
        pub preempt_enabled: bool,
        pub preempt_floor: PriorityClass,
    }

    impl Default for ResourceEnvelope {
        fn default() -> Self {
            let cpus = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(8);
            Self {
                ram_headroom_mb_min: 2048,
                max_model_runs: 1, // = runtime max_batch_size; raised by the host.
                max_cpu_runs: cpus.saturating_sub(1).max(1),
                max_worktrees: 32,
                max_ports_leased: 64,
                thermal_warn_pct: 0.15,
                thermal_throttle_pct: 0.25,
                max_spawns_per_min: 30.0,
                preempt_enabled: true,
                // Interactive(0)/High(1) are protected; Normal+ may be preempted.
                preempt_floor: PriorityClass::High,
            }
        }
    }

    /// Admission verdict (A.2). `Defer` means "skip this one, a cheaper job may fit".
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(tag = "verdict", rename_all = "snake_case")]
    pub enum Admission {
        Yes,
        No { reason: String },
        Defer { reason: String },
    }

    impl Admission {
        pub fn allowed(&self) -> bool {
            matches!(self, Admission::Yes)
        }
    }

    /// Back-compat alias: the original scaffold exposed `AdmissionDecision`. Callers
    /// can keep using `.allowed`/`.reason`; new code uses [`Admission`].
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct AdmissionDecision {
        pub allowed: bool,
        pub reason: String,
    }

    impl From<Admission> for AdmissionDecision {
        fn from(a: Admission) -> Self {
            match a {
                Admission::Yes => AdmissionDecision { allowed: true, reason: "admitted".to_string() },
                Admission::No { reason } | Admission::Defer { reason } => AdmissionDecision { allowed: false, reason },
            }
        }
    }

    /// Why the breaker is open (a banner reason, §4.6.4).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct BreakerState {
        pub tripped: bool,
        pub reason: Option<String>,
        pub spawn_ewma_per_min: f32,
    }

    /// The machine-wide Governor. Holds the envelope + live spawn-rate EWMA + the
    /// latest probe snapshot.
    #[derive(Debug, Clone)]
    pub struct FleetGovernor {
        pub envelope: ResourceEnvelope,
        snapshot: ResourceSnapshot,
        spawn_ewma_per_min: f32,
        last_spawn_ms: Option<u64>,
        breaker: BreakerState,
    }

    impl Default for FleetGovernor {
        fn default() -> Self {
            Self::new(ResourceEnvelope::default())
        }
    }

    impl FleetGovernor {
        pub fn new(envelope: ResourceEnvelope) -> Self {
            Self {
                envelope,
                snapshot: ResourceSnapshot::idle(),
                spawn_ewma_per_min: 0.0,
                last_spawn_ms: None,
                breaker: BreakerState { tripped: false, reason: None, spawn_ewma_per_min: 0.0 },
            }
        }

        pub fn snapshot(&self) -> &ResourceSnapshot {
            &self.snapshot
        }

        pub fn breaker(&self) -> &BreakerState {
            &self.breaker
        }

        /// Sample the machine (§4.6.2 step 0). Called ~1 Hz by the tick.
        pub fn refresh(&mut self, snapshot: ResourceSnapshot) {
            self.snapshot = snapshot;
        }

        /// The effective ceiling after thermal/RAM backoff (§4.6.2 step 1). When the
        /// thermal proxy shows a sustained dec_tps drop past the throttle threshold,
        /// the Model-pool ceiling shrinks (don't kill running work — just admit
        /// fewer). Returns the effective `max_model_runs`.
        pub fn effective_model_ceiling(&self) -> u32 {
            let base = self.envelope.max_model_runs.min(self.snapshot.max_generation_slots);
            let drop = self.snapshot.thermal_drop_pct();
            if drop >= self.envelope.thermal_throttle_pct {
                // Hard backoff: halve (at least 1 if anything is allowed at all).
                (base / 2).max(if base == 0 { 0 } else { 1 })
            } else if drop >= self.envelope.thermal_warn_pct {
                // Soft backoff: shave one slot.
                base.saturating_sub(1).max(if base == 0 { 0 } else { 1 })
            } else {
                base
            }
        }

        /// Admit a job's resource request under the (possibly-shrunk) envelope. The
        /// two-pool split is enforced here: a `Model` job checks the model ceiling;
        /// a `CpuOnly` job checks the CPU ceiling and never the model batch (§4.5.1).
        pub fn can_admit(&self, req: &ResourceRequest, live: &PoolOccupancy) -> Admission {
            if self.breaker.tripped {
                return Admission::No {
                    reason: format!("circuit breaker open: {}", self.breaker.reason.clone().unwrap_or_default()),
                };
            }
            // Critical thermal blocks Model-class admission outright; CpuOnly still ok.
            if self.snapshot.thermal == ThermalState::Critical && req.concurrency_class == ConcurrencyClass::Model {
                return Admission::No { reason: "thermal critical: refusing new model-class runs".to_string() };
            }
            // RAM headroom is a hard floor for everyone (no swap, P1).
            if self.snapshot.free_memory_mb < self.envelope.ram_headroom_mb_min + req.memory_mb {
                return Admission::Defer {
                    reason: format!(
                        "insufficient RAM headroom: {} free, need {} + {} floor",
                        self.snapshot.free_memory_mb, req.memory_mb, self.envelope.ram_headroom_mb_min
                    ),
                };
            }
            if live.worktrees >= self.envelope.max_worktrees {
                return Admission::Defer { reason: "worktree ceiling reached".to_string() };
            }
            if live.ports_leased.saturating_add(req.ports as u32) > self.envelope.max_ports_leased {
                return Admission::Defer { reason: "port pool ceiling reached".to_string() };
            }
            match req.concurrency_class {
                ConcurrencyClass::Model => {
                    let ceiling = self.effective_model_ceiling();
                    if live.model_runs + req.generation_slots > ceiling {
                        return Admission::Defer {
                            reason: format!("model pool full: {}/{} slots", live.model_runs, ceiling),
                        };
                    }
                }
                ConcurrencyClass::CpuOnly => {
                    if live.cpu_runs + 1 > self.envelope.max_cpu_runs {
                        return Admission::Defer {
                            reason: format!("cpu pool full: {}/{}", live.cpu_runs, self.envelope.max_cpu_runs),
                        };
                    }
                }
            }
            Admission::Yes
        }

        /// Record a spawn and update the spawn-rate EWMA; trip the breaker if the
        /// rate exceeds the envelope (the §3.8 "rapid increase in spawn frequency"
        /// trigger). Bounds *spawning*, never starves *running* work.
        pub fn note_spawn(&mut self, now_ms: u64) {
            if let Some(prev) = self.last_spawn_ms {
                let gap_ms = now_ms.saturating_sub(prev).max(1) as f32;
                let inst_per_min = 60_000.0 / gap_ms;
                // EWMA with alpha=0.3 over the instantaneous spawn rate.
                self.spawn_ewma_per_min = 0.3 * inst_per_min + 0.7 * self.spawn_ewma_per_min;
            } else {
                self.spawn_ewma_per_min = 0.0;
            }
            self.last_spawn_ms = Some(now_ms);
            self.breaker.spawn_ewma_per_min = self.spawn_ewma_per_min;
            if self.spawn_ewma_per_min > self.envelope.max_spawns_per_min {
                self.breaker.tripped = true;
                self.breaker.reason = Some(format!(
                    "spawn rate {:.1}/min exceeds {:.1}/min ceiling",
                    self.spawn_ewma_per_min, self.envelope.max_spawns_per_min
                ));
            }
        }

        /// Manually reset the breaker (host action after surfacing the banner).
        pub fn reset_breaker(&mut self) {
            self.breaker.tripped = false;
            self.breaker.reason = None;
            self.spawn_ewma_per_min = 0.0;
        }

        /// Should an interactive job preempt a lower-priority running one? True when
        /// preemption is enabled, the model pool is full at the effective ceiling, and
        /// an interactive job is waiting.
        pub fn should_preempt(&self, live: &PoolOccupancy, interactive_waiting: bool) -> bool {
            self.envelope.preempt_enabled && interactive_waiting && live.model_runs >= self.effective_model_ceiling()
        }

        pub fn preempt_floor(&self) -> PriorityClass {
            self.envelope.preempt_floor
        }
    }

    /// Live occupancy across the two pools + isolation leases (the admission input).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct PoolOccupancy {
        pub model_runs: u32,
        pub cpu_runs: u32,
        pub worktrees: u32,
        pub ports_leased: u32,
    }

    /// The result of one `schedule_tick`: what the loop decided to do, for events +
    /// tests. Returned so the caller (FleetManager) can emit the A.5 events and the
    /// scheduler stays I/O-free and unit-testable.
    #[derive(Debug, Clone, Default, PartialEq)]
    pub struct TickPlan {
        /// Job ids to admit + launch (already passed `can_admit`).
        pub admit: Vec<String>,
        /// Job ids to preempt (checkpoint-and-yield) for a waiting interactive job.
        pub preempt: Vec<String>,
        /// Job ids that were deferred (backpressure telemetry).
        pub deferred: Vec<String>,
        /// True if the breaker is open this tick (surface a banner).
        pub breaker_open: bool,
        /// The effective model ceiling this tick (after backoff).
        pub effective_model_ceiling: u32,
    }

    /// The pure scheduling decision (no I/O). `FleetManager::tick` calls this, then
    /// performs the I/O (preempt checkpoints, kernel launches) for the returned plan.
    pub struct FleetScheduler;

    impl FleetScheduler {
        /// Compute the tick plan (§4.6.2): preempt → admit under the ceiling. `ready`
        /// is the priority-ordered ready-set; `occupancy` is current pool usage;
        /// `lowest_preemptible` is the victim candidate (if any).
        pub fn plan_tick(
            gov: &FleetGovernor,
            ready: &[ReadyJob],
            mut occupancy: PoolOccupancy,
            interactive_waiting: bool,
            lowest_preemptible: Option<&str>,
        ) -> TickPlan {
            let mut plan = TickPlan {
                breaker_open: gov.breaker().tripped,
                effective_model_ceiling: gov.effective_model_ceiling(),
                ..Default::default()
            };

            // Step 2: preemption. If an interactive job waits and the model pool is
            // saturated, free a slot by preempting the weakest batch run.
            if gov.should_preempt(&occupancy, interactive_waiting) {
                if let Some(victim) = lowest_preemptible {
                    plan.preempt.push(victim.to_string());
                    occupancy.model_runs = occupancy.model_runs.saturating_sub(1);
                }
            }

            // Step 3: admit in priority order under the (possibly-shrunk) ceiling.
            for job in ready {
                match gov.can_admit(&job.request, &occupancy) {
                    Admission::Yes => {
                        plan.admit.push(job.id.clone());
                        match job.request.concurrency_class {
                            ConcurrencyClass::Model => occupancy.model_runs += job.request.generation_slots,
                            ConcurrencyClass::CpuOnly => occupancy.cpu_runs += 1,
                        }
                        occupancy.ports_leased += job.request.ports as u32;
                        occupancy.worktrees += 1;
                    }
                    Admission::No { .. } => {
                        // Hard no (thermal/breaker): stop admitting model work, but a
                        // cheaper CpuOnly job later in the list may still fit.
                        plan.deferred.push(job.id.clone());
                    }
                    Admission::Defer { .. } => {
                        plan.deferred.push(job.id.clone());
                    }
                }
            }
            plan
        }
    }

    /// A ready job's scheduling inputs (decoupled from the full `AgentJob` so the
    /// scheduler is pure).
    #[derive(Debug, Clone, PartialEq)]
    pub struct ReadyJob {
        pub id: String,
        pub priority: PriorityClass,
        pub request: ResourceRequest,
        pub status: JobStatus,
    }

    /// Build a snapshot from a live probe (helper so the host doesn't construct it by
    /// hand). Re-exported convenience.
    pub async fn probe_snapshot(probe: &dyn ResourceProbe, max_slots: u32, active: u32) -> ResourceSnapshot {
        probe.snapshot(max_slots, active).await
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::resources::ThermalState;

        fn model_req(mem: u64) -> ResourceRequest {
            ResourceRequest { memory_mb: mem, ..ResourceRequest::default() }
        }

        fn snapshot(free_mb: u64, slots: u32, active: u32, thermal: ThermalState) -> ResourceSnapshot {
            ResourceSnapshot {
                free_memory_mb: free_mb,
                max_generation_slots: slots,
                active_generation_slots: active,
                thermal,
                dec_tps_now: 40.0,
                dec_tps_baseline: 40.0,
                battery_percent: None,
                on_ac_power: true,
                idle: true,
            }
        }

        #[test]
        fn admits_when_ram_and_slots_available() {
            let mut gov = FleetGovernor::new(ResourceEnvelope {
                max_model_runs: 2,
                ram_headroom_mb_min: 1000,
                ..Default::default()
            });
            gov.refresh(snapshot(8192, 2, 0, ThermalState::Nominal));
            let occ = PoolOccupancy::default();
            assert!(gov.can_admit(&model_req(1024), &occ).allowed());
        }

        #[test]
        fn defers_when_ram_below_floor() {
            let mut gov = FleetGovernor::new(ResourceEnvelope { ram_headroom_mb_min: 4000, ..Default::default() });
            gov.refresh(snapshot(4500, 4, 0, ThermalState::Nominal));
            // 4500 free, need 1024 + 4000 floor = 5024 → defer.
            assert!(matches!(gov.can_admit(&model_req(1024), &PoolOccupancy::default()), Admission::Defer { .. }));
        }

        #[test]
        fn two_pools_have_independent_ceilings() {
            let mut gov = FleetGovernor::new(ResourceEnvelope {
                max_model_runs: 1,
                max_cpu_runs: 4,
                ram_headroom_mb_min: 500,
                ..Default::default()
            });
            gov.refresh(snapshot(16000, 1, 0, ThermalState::Nominal));
            // Model pool full at 1...
            let occ = PoolOccupancy { model_runs: 1, ..Default::default() };
            assert!(!gov.can_admit(&model_req(512), &occ).allowed());
            // ...but a CpuOnly job still admits (separate pool).
            let cpu = ResourceRequest::cpu_only(512);
            assert!(gov.can_admit(&cpu, &occ).allowed());
        }

        #[test]
        fn thermal_drop_shrinks_model_ceiling() {
            let mut gov = FleetGovernor::new(ResourceEnvelope {
                max_model_runs: 4,
                thermal_throttle_pct: 0.25,
                ..Default::default()
            });
            // dec_tps dropped 40 → 28 = 30% drop ≥ 25% throttle → halve 4 → 2.
            let mut snap = snapshot(32000, 4, 0, ThermalState::Fair);
            snap.dec_tps_now = 28.0;
            snap.dec_tps_baseline = 40.0;
            gov.refresh(snap);
            assert_eq!(gov.effective_model_ceiling(), 2);
        }

        #[test]
        fn spawn_rate_trips_the_breaker() {
            let mut gov = FleetGovernor::new(ResourceEnvelope { max_spawns_per_min: 10.0, ..Default::default() });
            // 5 spawns 100 ms apart → 600/min instantaneous → EWMA climbs past 10.
            let mut t = 1_000_000u64;
            for _ in 0..6 {
                gov.note_spawn(t);
                t += 100;
            }
            assert!(gov.breaker().tripped);
            // A tripped breaker refuses admission.
            gov.refresh(snapshot(32000, 4, 0, ThermalState::Nominal));
            assert!(matches!(gov.can_admit(&model_req(512), &PoolOccupancy::default()), Admission::No { .. }));
        }

        #[test]
        fn plan_tick_preempts_for_interactive_then_admits() {
            let mut gov = FleetGovernor::new(ResourceEnvelope {
                max_model_runs: 1,
                ram_headroom_mb_min: 500,
                ..Default::default()
            });
            gov.refresh(snapshot(32000, 1, 1, ThermalState::Nominal));
            let occ = PoolOccupancy { model_runs: 1, ..Default::default() };
            let ready = vec![ReadyJob {
                id: "interactive".to_string(),
                priority: PriorityClass::Interactive,
                request: model_req(1024),
                status: JobStatus::Queued,
            }];
            let plan = FleetScheduler::plan_tick(&gov, &ready, occ, true, Some("batch_victim"));
            assert_eq!(plan.preempt, vec!["batch_victim".to_string()]);
            // After preemption frees the slot, the interactive job is admitted.
            assert_eq!(plan.admit, vec!["interactive".to_string()]);
        }
    }
}

pub use manager::{FleetConfig, FleetManager, LaunchedRun};
pub use queue::{
    AgentJob, ConcurrencyClass, Isolation, JobGraph, JobKind, JobStatus, PriorityClass,
    ResourceRequest,
};
pub use resources::{
    FixedResourceProbe, OsResourceProbe, ResourceProbe, ResourceSnapshot, ThermalState,
};
pub use scheduler::{
    AdmissionDecision, FleetGovernor, FleetScheduler, PoolOccupancy, ReadyJob, ResourceEnvelope,
    TickPlan,
};
