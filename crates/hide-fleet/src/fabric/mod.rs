//! Fabric software plane: node discovery, placement simulation, KV ownership,
//! pipeline scheduling, heartbeats, and failure/replay receipts.
//!
//! This module coordinates distributed *inference placement* software. It does
//! **not** execute model weights or touch the inference hot path (Metal /
//! gravity / kernels). Another lane owns execution.
//!
//! ## Qualification law
//!
//! Simulated or fixture results are **never** physical hardware qualification.
//! Every plan / receipt that is not physical hardware must set
//! `not_physical_qualification: true` and a non-physical
//! [`qualification::QualificationKind`]. The schema validator rejects
//! unlabelled simulated results.
//!
//! Terminal hardware state for this single-machine session:
//! `FABRIC_HARDWARE_QUALIFICATION_PENDING`.

pub mod agent;
pub mod failure;
pub mod fixture;
pub mod node;
pub mod placement;
pub mod pipeline;
pub mod protocol;
pub mod qualification;

pub use agent::{AgentConfig, AgentState, FabricAgent, FabricAgentHandle};
pub use failure::{
    CheckpointId, FailureDetector, FailureReplayReceipt, HeartbeatMonitor, LostWorkSummary,
};
pub use fixture::{
    run_inprocess_software_fixture, run_two_process_fixture, TwoProcessFixtureResult,
};
pub use node::{
    AcceleratorClass, BandwidthClass, DiscoverySource, NodeCapabilities, NodeDiscovery, NodeId,
    OsNodeProbe, SimulatedNodeSet, FIXED_FAKE_MEMORY_BYTES,
};
pub use placement::{
    reject_unlabelled_simulated, validate_placement_plan_schema, ContentHash,
    KvOwnershipInvariant, KvRangeOwnership, ModelSection, PlacementPlan, PlacementRequest,
    PlacementSimulator, PredictedCost, SectionPlacement, StageAssignment, WorkloadClass,
    PLACEMENT_SCHEMA,
};
pub use pipeline::{
    MicrobatchState, PipelineScheduler, PipelineStatus, StageGraph, StageId, StageState,
};
pub use protocol::{AgentRequest, AgentResponse, PlacementAssignment};
pub use qualification::{QualificationKind, HARDWARE_QUALIFICATION_PENDING, QUALIFICATION_SCHEMA};
