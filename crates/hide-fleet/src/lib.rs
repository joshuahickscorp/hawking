//! Parallel-agent and workstation backend.
//!
//! This crate implements the headless fabric described in HIDE bible chapter
//! 09: job queues, machine-wide resource admission, isolation leases, merge
//! selection, batch reports, and remote protocol contracts.

pub mod batch;
pub mod fleetview;
pub mod isolate;
pub mod manager;
pub mod merge;
pub mod patterns;
pub mod queue;
pub mod remote;
pub mod resources;
pub mod scheduler;

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
