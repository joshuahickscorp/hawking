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
                "job.admitted" | "job.started" | "job.completed" | "job.preempted"
                | "job.merging" | "job.paused" | "job.requeued" => {
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
                        row.workspace = event
                            .payload
                            .get("path")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        row.ports = event
                            .payload
                            .get("ports")
                            .and_then(|v| v.as_array())
                            .map(|a| {
                                a.iter()
                                    .filter_map(|p| p.as_u64().map(|n| n as u16))
                                    .collect()
                            })
                            .unwrap_or_default();
                    }
                }
                "governor.breaker" => {
                    view.breaker_banner = str_field(event, "reason").or_else(|| {
                        Some(format!(
                            "circuit breaker: {}",
                            str_field(event, "trigger").unwrap_or_default()
                        ))
                    });
                }
                "governor.backoff" => {
                    view.backoff_banner = Some(format!(
                        "backoff ({}): ceiling {}",
                        str_field(event, "reason").unwrap_or_default(),
                        event
                            .payload
                            .get("new_ceiling")
                            .map(|v| v.to_string())
                            .unwrap_or_default()
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
        log.append(NewEvent::system(
            s.clone(),
            "job.enqueued",
            json!({ "job_id": "j1" }),
        ))
        .await
        .unwrap();
        log.append(NewEvent::system(
            s.clone(),
            "job.started",
            json!({ "job_id": "j1", "status": "running" }),
        ))
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
