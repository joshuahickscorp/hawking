//! Security infrastructure for HIDE.
//!
//! Backend-only pieces from bible chapter 10: redaction before durability,
//! hash-chain audit helpers, sandbox profile rendering, and at-rest policy
//! records. Actual OS enforcement is still host/platform-specific.

pub mod audit;
pub mod redaction;
pub mod sandbox;
pub mod storage;

pub use audit::{compute_event_chain, verify_event_chain, EventChainAuditor};
pub use redaction::{Redaction, Redactor};
