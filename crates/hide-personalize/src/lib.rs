//! Personalization and self-improvement backend.
//!
//! This crate covers HIDE bible chapter 11 backend hooks: accepted-edit capture,
//! curation, local eval generation, prompt optimization records, RLEF data,
//! learned retrieval signals, KV handoff descriptors, and world-model stubs.

pub mod curate;
pub mod eval;
pub mod kv_handoff;
pub mod prompts;
pub mod records;
pub mod retrieval;
pub mod rlef;
pub mod store;
pub mod world;

pub use records::{Outcome, PersonalizationRecord, TaskClass};
pub use store::{
    DynPersonalizationStore, InMemoryPersonalizationStore, JsonlPersonalizationStore,
    PersonalizationStore,
};
