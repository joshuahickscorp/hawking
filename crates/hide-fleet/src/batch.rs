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
        Self {
            id: format!("batch_{}", ulid::Ulid::new()),
            job_ids,
            schedule,
            report_on_wake: true,
        }
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
pub fn evaluate_gate(
    schedule: &BatchSchedule,
    now: u64,
    snapshot: &ResourceSnapshot,
) -> GateStatus {
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
    let member_set: std::collections::BTreeSet<&str> =
        batch.job_ids.iter().map(String::as_str).collect();

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
                        let status = event
                            .payload
                            .get("status")
                            .and_then(|v| v.as_str())
                            .unwrap_or("failed");
                        let outcome = event
                            .payload
                            .get("summary")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        match status {
                            "done" => summary.succeeded += 1,
                            "failed" | "cancelled" => summary.failed += 1,
                            _ => summary.partial += 1,
                        }
                        results.insert(
                            job_id.to_string(),
                            WakeResult {
                                job_id: job_id.to_string(),
                                status: status.to_string(),
                                outcome,
                            },
                        );
                    }
                }
            }
            "governor.backoff" => {
                if event.payload.get("reason").and_then(|v| v.as_str()) == Some("thermal") {
                    thermal_events += 1;
                }
            }
            _ => {}
        }
    }

    summary.goals = batch.job_ids.len() as u32;
    // Members that completed-but-not-done are queued for review.
    let needs_review: Vec<String> = results
        .values()
        .filter(|r| r.status != "done")
        .map(|r| r.job_id.clone())
        .collect();

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
        let sched = BatchSchedule {
            gates: vec![ScheduleGate::ThermalOk],
            ..Default::default()
        };
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
            log.append(NewEvent::system(
                session.clone(),
                "job.started",
                json!({ "job_id": job_id }),
            ))
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
        log.append(NewEvent::system(
            session.clone(),
            "governor.backoff",
            json!({ "reason": "thermal" }),
        ))
        .await
        .unwrap();

        let batch = BatchJob::new(
            vec!["job_a".to_string(), "job_b".to_string()],
            BatchSchedule::default(),
        );
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
