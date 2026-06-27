//! Parallel-agent and workstation backend.
//!
//! This crate implements the headless fabric described in HIDE bible chapter
//! 09: job queues, machine-wide resource admission, isolation leases, merge
//! selection, batch reports, and remote protocol contracts.

pub mod batch;
pub mod isolate;
pub mod merge;
pub mod patterns;
pub mod queue;
pub mod remote;
pub mod scheduler;

pub use queue::{AgentJob, JobGraph, JobStatus};
pub use scheduler::{AdmissionDecision, FleetGovernor, ResourceSnapshot};
