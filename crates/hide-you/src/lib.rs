//! # hide-you — YOU surface swarms, projects, and typed handoffs
//!
//! HIDE has three surfaces (YOU / CHAT / IDE) as lenses over one session. This
//! crate implements the YOU-side product layer that the surface-authority
//! contract requires:
//!
//! * **Swarms** — governed teams on the fleet substrate concept (roles, modes,
//!   per-agent capsules, budgets), not prompt multiplication.
//! * **Projects** — unified containers for conversations, documents, objects,
//!   connectors, plans, tasks, memory, automations, agents, and artifacts.
//! * **Typed handoffs** — YOU→CHAT / CHAT→IDE / IDE→YOU capsules that carry
//!   **claims**, never **capabilities**.
//!
//! ## The properties that matter
//!
//! 1. A capsule carries the CLAIM, never the CAPABILITY. Receiving a YOU→CHAT
//!    capsule must not grant CHAT the connector access YOU held.
//! 2. No agent promotes its own high-risk conclusion; promotion needs
//!    independent verification. Consensus is weak; a reproduced defect outranks
//!    votes.
//! 3. Resource economics are enforced (CPU/RAM, tokens/steps, wall time, stop
//!    condition). A swarm that exceeds its budget halts and records why.
//! 4. Every capsule carries provenance, evidence tier of claims, permissions it
//!    was created under, and what it deliberately excludes.
//!
//! ## Why a new crate (not hide-fleet)
//!
//! YOU-surface domain (roles, modes, projects, typed surface handoffs) is
//! orthogonal to hide-fleet's machine-wide job scheduler and GPU admission;
//! this crate reuses the capability-derivation pattern from
//! `hide_core::automation` and does not pull hide-kernel or Metal.
//!
//! ## What is fake
//!
//! All "inference" goes through [`fixture::FixtureProvider`]. No model loads,
//! no network, no Metal, no Odyssey fence, no adapter-grade changes.

pub mod agent;
pub mod budget;
pub mod capsule;
pub mod capability;
pub mod error;
pub mod evidence;
pub mod fixture;
pub mod modes;
pub mod project;
pub mod promotion;
pub mod roles;
pub mod surface;
pub mod swarm;

pub use agent::{AgentId, AgentReceipt, AgentSpec, OutputSchema, VerificationContract};
pub use budget::{
    BudgetAxis, BudgetUsage, ResourceBudget, StopCondition, StopReason, SwarmBudget,
};
pub use capsule::{
    Claim, DeliberateExclusion, HandoffCapsule, HandoffKind, OpenedCapsule, PermissionSnapshot,
    ProvenanceEntry, ReceivedHandoff, SurfaceSession,
};
pub use capability::{CapabilitySnapshot, SurfaceCapability, SurfacePermissionSet};
pub use error::{Result, YouError};
pub use evidence::EvidenceTier;
pub use fixture::{FixtureProvider, FixtureReply};
pub use modes::SwarmMode;
pub use project::{Project, ProjectId, ProjectMemberKind, ProjectState};
pub use promotion::{
    Conclusion, ConclusionRisk, PromotionBoard, PromotionDecision, PromotionEvidence, VoteTally,
};
pub use roles::AgentRole;
pub use surface::{Surface, SurfaceDefaults};
pub use swarm::{Swarm, SwarmId, SwarmStatus};
