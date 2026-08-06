//! # hide-objects — YOU surface object + attachment store
//!
//! Content-addressed object system for HIDE's **YOU | CHAT | IDE** surfaces.
//! All three surfaces share one session, Context OS, memory graph, object store,
//! tools, permissions and canonical events; this crate is the object store.
//!
//! ## Laws
//!
//! 1. **Content hash is identity.** Same bytes → one object, many refs.
//! 2. **Incremental processing.** Stages are independently resumable; working
//!    buffer is bounded by [`hash::CHUNK_SIZE`] (256 KiB).
//! 3. **Model sees derivatives only.** [`derivatives::CompileObjectView`] has
//!    no raw-bytes path; raw access requires [`derivatives::RawBytesCap`].
//! 4. **Queue never silent-drops.** Failures retry or land in a dead-letter
//!    log as [`queue::JobStatus::FailedVisible`].
//! 5. **Retention + permissions at read time.**
//! 6. **Storage is finite.** See [`budget::BOUND_STATEMENT`].
//!
//! ## What is fake
//!
//! OCR, ASR, and thumbnail codecs are intentionally [`processors::FakeOcrEngine`],
//! [`processors::FakeAsrEngine`], and [`processors::FakeThumbnailer`]. The
//! pipeline is real; the engines are labelled fake so nothing pretends to be a
//! model. No embeddings, no large index builds, no Metal.
//!
//! ```
//! use hide_core::objects::{
//!     ObjectStore, StorageBudget, ObjectSource, ObjectPermissions, Surface,
//!     RetentionPolicy, Priority, Reader, DerivativeSelection,
//! };
//! use tempfile::tempdir;
//!
//! let dir = tempdir().unwrap();
//! let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
//! let job = store.enqueue_bytes(
//!     b"hello from YOU",
//!     "text/plain",
//!     ObjectSource::Synthetic { label: "demo".into() },
//!     ObjectPermissions::owner_only("alice", vec![Surface::You]),
//!     RetentionPolicy::durable(),
//!     Some("note.txt".into()),
//!     "alice",
//!     Priority::NORMAL,
//! ).unwrap();
//! let (_id, status) = store.process_one().unwrap();
//! assert_eq!(status, hide_core::objects::JobStatus::Succeeded);
//! let hash = store.hash_for_job(&job).unwrap();
//! let reader = Reader { principal: "alice".into(), surface: Surface::You };
//! let view = store.compile_view(&hash, &reader, &DerivativeSelection::default(), None).unwrap();
//! assert!(!hide_core::objects::CompileObjectView::exposes_raw_bytes());
//! assert!(view.try_raw_bytes().is_err());
//! let _ = view;
//! ```

pub use budget::{StorageBudget, BOUND_STATEMENT};
pub use derivatives::{CompileObjectView, DerivativeSelection, ModelFacingDerivative, RawBytesCap};
pub use error::{ObjectError, Result};
pub use hash::{ContentHash, CHUNK_SIZE};
pub use kinds::{mime_from_filename, ObjectKind};
pub use permissions::{ObjectPermissions, Reader, Surface};
pub use processors::{
    AsrEngine, FakeAsrEngine, FakeOcrEngine, FakeThumbnailer, OcrEngine, ProcessorSet,
    TextExtractor, Thumbnailer, Utf8TextExtractor,
};
pub use queue::{IngestJob, IngestQueue, JobStatus, Priority};
pub use retention::RetentionPolicy;
pub use schema::{
    Derivative, DerivativeKind, ObjectLocation, ObjectRecord, ObjectRef, ObjectSource,
    ObjectStatus, RefId, StageName, StageRecord, StageStatus,
};
pub use store::ObjectStore;

#[path = "objects_budget.rs"]
pub mod budget;
#[path = "objects_derivatives.rs"]
pub mod derivatives;
#[path = "objects_error.rs"]
pub mod error;
#[path = "objects_hash.rs"]
pub mod hash;
#[path = "objects_kinds.rs"]
pub mod kinds;
#[path = "objects_permissions.rs"]
pub mod permissions;
#[path = "objects_pipeline.rs"]
pub mod pipeline;
#[path = "objects_processors.rs"]
pub mod processors;
#[path = "objects_queue.rs"]
pub mod queue;
#[path = "objects_retention.rs"]
pub mod retention;
#[path = "objects_schema.rs"]
pub mod schema;
#[path = "objects_store.rs"]
pub mod store;
