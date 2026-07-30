//! # One model-family adapter ABI + honest support-level registry
//!
//! This is **not** the LoRA selection registry in `hawking-orch::adapters`
//! (language/task LoRAs). This is the **architecture-family** registry that
//! answers: for Llama / Qwen / GLM / …, what is the true support level, does
//! anything execute, is the family serve-registered, and what is the full ABI?
//!
//! ## Support grades (exactly these, never inflate)
//!
//! ```text
//! DECLARED                 described; nothing parsed, nothing executes
//! SOURCE_HEADER_VALIDATED  real official config/tokenizer/safetensors header parsed and mapped
//! SYNTHETIC_PARITY         matches a deterministic reference on a synthetic twin
//! REAL_TENSOR_DECODE       at least one real tensor decoded from a real checkpoint
//! SMALL_REAL_CHECKPOINT    a real small checkpoint of the family runs end to end
//! FULL_PARENT_VALIDATED    a real full-size parent validated
//! PRODUCTION               served, under test, with a standing parity receipt
//! ```
//!
//! **No family is PRODUCTION today.** Promoting a level requires evidence of
//! the kind the grade names, enforced by the registry test.
//!
//! ## Layout
//!
//! Family rows live in the counted table in [`families`]; [`registry`] indexes
//! them. [`generate`] emits docs, JSON schemas (adapters/artifacts/profiles/runtime
//! capabilities/events/Fabric/tool effects), CLI surface + shell completion,
//! SDK types, HIDE capability declarations, Fabric declarations, schema
//! migrations, and the root JSON deliverables — same deterministic golden-file
//! pattern as `hide-sdk-codegen`. Do not add a second codegen system.

pub mod bridge_surface;
pub mod abi;
pub mod evidence;
pub mod export;
pub mod families;
#[cfg(feature = "events")]
pub mod generate;
pub mod registry;
pub mod support_level;

pub use abi::{
    required_evidence_kind, AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind,
    FamilyAbi, FamilyDescriptor, ProviderAvailability, ABI_FIELD_NAMES,
};
pub use bridge_surface::{
    bridge_surface_document, bridge_surface_json, not_implemented_body, EndpointStatus,
};
pub use export::{
    adapter_abi_json, adapter_registry_document, adapter_registry_json, capability_matrix_json,
    migration_map_json, test_matrix_json,
};
#[cfg(feature = "events")]
pub use generate::{generate_all, repo_root_artifacts, write_all, GeneratedArtifact};
pub use registry::{builtin_registry, FamilyRegistry};
pub use support_level::SupportLevel;

/// Schema id for `HAWKING_ADAPTER_REGISTRY.json`.
pub const REGISTRY_SCHEMA: &str = "hawking.adapters.registry.v2";

/// Schema id for `HAWKING_ADAPTER_ABI.json`.
pub const ABI_SCHEMA: &str = "hawking.adapters.abi.v1";
