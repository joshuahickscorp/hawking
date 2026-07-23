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
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8);
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
            Admission::Yes => AdmissionDecision {
                allowed: true,
                reason: "admitted".to_string(),
            },
            Admission::No { reason } | Admission::Defer { reason } => AdmissionDecision {
                allowed: false,
                reason,
            },
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
            breaker: BreakerState {
                tripped: false,
                reason: None,
                spawn_ewma_per_min: 0.0,
            },
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
        let base = self
            .envelope
            .max_model_runs
            .min(self.snapshot.max_generation_slots);
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
                reason: format!(
                    "circuit breaker open: {}",
                    self.breaker.reason.clone().unwrap_or_default()
                ),
            };
        }
        // Critical thermal blocks Model-class admission outright; CpuOnly still ok.
        if self.snapshot.thermal == ThermalState::Critical
            && req.concurrency_class == ConcurrencyClass::Model
        {
            return Admission::No {
                reason: "thermal critical: refusing new model-class runs".to_string(),
            };
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
            return Admission::Defer {
                reason: "worktree ceiling reached".to_string(),
            };
        }
        if live.ports_leased.saturating_add(req.ports as u32) > self.envelope.max_ports_leased {
            return Admission::Defer {
                reason: "port pool ceiling reached".to_string(),
            };
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
                        reason: format!(
                            "cpu pool full: {}/{}",
                            live.cpu_runs, self.envelope.max_cpu_runs
                        ),
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
        self.envelope.preempt_enabled
            && interactive_waiting
            && live.model_runs >= self.effective_model_ceiling()
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
                        ConcurrencyClass::Model => {
                            occupancy.model_runs += job.request.generation_slots
                        }
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
pub async fn probe_snapshot(
    probe: &dyn ResourceProbe,
    max_slots: u32,
    active: u32,
) -> ResourceSnapshot {
    probe.snapshot(max_slots, active).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::ThermalState;

    fn model_req(mem: u64) -> ResourceRequest {
        ResourceRequest {
            memory_mb: mem,
            ..ResourceRequest::default()
        }
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
        let mut gov = FleetGovernor::new(ResourceEnvelope {
            ram_headroom_mb_min: 4000,
            ..Default::default()
        });
        gov.refresh(snapshot(4500, 4, 0, ThermalState::Nominal));
        // 4500 free, need 1024 + 4000 floor = 5024 → defer.
        assert!(matches!(
            gov.can_admit(&model_req(1024), &PoolOccupancy::default()),
            Admission::Defer { .. }
        ));
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
        let occ = PoolOccupancy {
            model_runs: 1,
            ..Default::default()
        };
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
        let mut gov = FleetGovernor::new(ResourceEnvelope {
            max_spawns_per_min: 10.0,
            ..Default::default()
        });
        // 5 spawns 100 ms apart → 600/min instantaneous → EWMA climbs past 10.
        let mut t = 1_000_000u64;
        for _ in 0..6 {
            gov.note_spawn(t);
            t += 100;
        }
        assert!(gov.breaker().tripped);
        // A tripped breaker refuses admission.
        gov.refresh(snapshot(32000, 4, 0, ThermalState::Nominal));
        assert!(matches!(
            gov.can_admit(&model_req(512), &PoolOccupancy::default()),
            Admission::No { .. }
        ));
    }

    #[test]
    fn plan_tick_preempts_for_interactive_then_admits() {
        let mut gov = FleetGovernor::new(ResourceEnvelope {
            max_model_runs: 1,
            ram_headroom_mb_min: 500,
            ..Default::default()
        });
        gov.refresh(snapshot(32000, 1, 1, ThermalState::Nominal));
        let occ = PoolOccupancy {
            model_runs: 1,
            ..Default::default()
        };
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
