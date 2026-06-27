use crate::queue::{AgentJob, JobGraph, JobStatus, ResourceRequest};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub free_memory_mb: u64,
    pub max_generation_slots: u32,
    pub active_generation_slots: u32,
    pub thermal: ThermalState,
    pub battery_percent: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionDecision {
    pub allowed: bool,
    pub reason: String,
}

pub struct FleetGovernor {
    pub min_free_memory_mb: u64,
    pub allow_on_thermal: ThermalState,
}

impl Default for FleetGovernor {
    fn default() -> Self {
        Self {
            min_free_memory_mb: 2048,
            allow_on_thermal: ThermalState::Fair,
        }
    }
}

impl FleetGovernor {
    pub fn admit(
        &self,
        request: &ResourceRequest,
        snapshot: &ResourceSnapshot,
    ) -> AdmissionDecision {
        if snapshot.thermal > self.allow_on_thermal {
            return AdmissionDecision {
                allowed: false,
                reason: format!("thermal state {:?} exceeds policy", snapshot.thermal),
            };
        }
        if snapshot.free_memory_mb < self.min_free_memory_mb + request.memory_mb {
            return AdmissionDecision {
                allowed: false,
                reason: "insufficient memory headroom".to_string(),
            };
        }
        if snapshot.active_generation_slots + request.generation_slots
            > snapshot.max_generation_slots
        {
            return AdmissionDecision {
                allowed: false,
                reason: "runtime generation slot limit reached".to_string(),
            };
        }
        AdmissionDecision {
            allowed: true,
            reason: "admitted".to_string(),
        }
    }
}

pub struct Scheduler {
    pub governor: FleetGovernor,
}

impl Scheduler {
    pub fn next(&self, graph: &JobGraph, snapshot: &ResourceSnapshot) -> Option<AgentJob> {
        for job in graph.ready_jobs() {
            if self
                .governor
                .admit(&job.spec.required_resources, snapshot)
                .allowed
            {
                graph.set_status(&job.id, JobStatus::Ready);
                return Some(job);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn governor_rejects_when_slots_are_full() {
        let governor = FleetGovernor::default();
        let decision = governor.admit(
            &ResourceRequest {
                memory_mb: 512,
                generation_slots: 1,
                ports: 0,
            },
            &ResourceSnapshot {
                free_memory_mb: 8192,
                max_generation_slots: 1,
                active_generation_slots: 1,
                thermal: ThermalState::Nominal,
                battery_percent: None,
            },
        );
        assert!(!decision.allowed);
    }
}
