//! hide-state: state-capsule schemas, integrity, and compatibility binding.
//!
//! HIDE carries agent state as capsules: opaque runtime bytes wrapped in a
//! header that identifies them and a digest that lets a reader prove they are
//! intact (Bible sec 23, sec 56). This crate defines those schemas and the
//! deterministic logic around them: what a capsule is, how it serializes to a
//! self-describing byte stream, how a reader verifies it, when it is allowed to
//! bind to a runtime, and how a store saves, loads, forks, compares, releases,
//! and inspects capsules.
//!
//! Scope: this crate is schema-only and entirely model-free. It describes and
//! verifies capsule bytes over synthetic fixtures. It never runs a model, never
//! produces or consumes live runtime state, and makes no assertion about
//! runtime performance or output quality (Bible law 17, sec 56 gate). The
//! runtime that actually produces these bytes from a live engine and rebinds
//! them is DEFERRED_MODEL_REQUIRED: it connects later, against these schemas,
//! and nothing here should be read as a claim that it exists yet.
//!
//! The invariants this crate holds:
//!
//! - Self-describing bytes. A serialized capsule carries its own magic tag,
//!   format version, metadata, and integrity digest, so a reader needs no
//!   out-of-band convention to parse and check it.
//!
//! - Integrity on every load. [`Capsule::from_bytes`] recomputes the payload
//!   digest and rejects any stream whose payload was altered.
//!
//! - Honest ancestry. [`Capsule::fork`] copies the payload byte for byte under
//!   a fresh id and records the parent, so lineage is always recoverable.
//!
//! - Strict binding. A capsule refuses to load into a runtime whose identity
//!   disagrees on any field, and says exactly which field via a typed
//!   [`IncompatibleReason`] rather than a scraped string.
//!
//! ```
//! use hide_state::{
//!     CapsuleBuilder, CapsuleType, CapsuleStore, IdentityBinding, MemoryStore,
//! };
//!
//! let identity = IdentityBinding {
//!     model_weights_id: "w".into(),
//!     arch_id: "a".into(),
//!     tokenizer_id: "t".into(),
//!     prompt_abi_version: "1".into(),
//!     tool_registry_id: "r".into(),
//!     engine_build_id: "b".into(),
//!     security_domain: "d".into(),
//! };
//!
//! let capsule = CapsuleBuilder::new(CapsuleType::Recurrent, "model-x", identity.clone())
//!     .runtime_version("rt-1")
//!     .seal(vec![1, 2, 3, 4]);
//!
//! let mut store = MemoryStore::new();
//! let id = store.save(&capsule).unwrap();
//! let loaded = store.load(&id).unwrap();
//! assert_eq!(loaded.payload(), &[1, 2, 3, 4]);
//! assert!(loaded.is_loadable(&identity).is_ok());
//! ```

pub mod capsule;
pub mod error;
pub mod header;
pub mod identity;
pub mod integrity;
pub mod store;

pub use capsule::{Capsule, CapsuleBuilder, CapsuleInspect};
pub use error::{CapsuleError, IncompatibleReason, Result};
pub use header::{now_ms, CapsuleHeader, CapsuleId, CapsuleType};
pub use identity::IdentityBinding;
pub use integrity::{Integrity, IntegrityAlgo};
pub use store::{Ancestry, CapsuleComparison, CapsuleStore, DiskStore, MemoryStore};
