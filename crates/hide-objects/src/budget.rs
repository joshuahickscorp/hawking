//! Storage bounds. The system is *effectively unbounded* relative to a single
//! user turn, but never literally unlimited.
//!
//! Bounds are configuration + local/cloud capacity + model capability + user
//! policy. The schema and runtime both refuse to claim otherwise.

use serde::{Deserialize, Serialize};

use crate::error::{ObjectError, Result};

/// Configured ceilings for the object store.
///
/// Defaults are intentionally modest for LIGHT_ONLY tests; production sets
/// these from user policy and measured free space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageBudget {
    /// Max total local blob bytes retained by this store instance.
    pub max_local_bytes: u64,
    /// Max total cloud-resident blob bytes (accounting only; this crate does
    /// not implement cloud I/O).
    pub max_cloud_bytes: u64,
    /// Hard cap on a single object body.
    pub max_object_bytes: u64,
    /// Human-readable policy note recorded on every rejection.
    pub policy_note: String,
}

impl Default for StorageBudget {
    fn default() -> Self {
        Self {
            // 64 GiB local default — "effectively unbounded" for personal use,
            // still a hard number the runtime enforces.
            max_local_bytes: 64 * 1024 * 1024 * 1024,
            max_cloud_bytes: 256 * 1024 * 1024 * 1024,
            // 32 GiB single-object cap (large video), still finite.
            max_object_bytes: 32 * 1024 * 1024 * 1024,
            policy_note: "bounded by configured local/cloud storage, model capability, and user policy — not unlimited".into(),
        }
    }
}

impl StorageBudget {
    /// Tight budget for unit tests.
    pub fn test_small() -> Self {
        Self {
            max_local_bytes: 64 * 1024 * 1024, // 64 MiB
            max_cloud_bytes: 64 * 1024 * 1024,
            max_object_bytes: 32 * 1024 * 1024, // 32 MiB
            policy_note: "test budget — deliberately small".into(),
        }
    }

    pub fn check_object_size(&self, size: u64) -> Result<()> {
        if size > self.max_object_bytes {
            return Err(ObjectError::ObjectTooLarge {
                size,
                max: self.max_object_bytes,
            });
        }
        Ok(())
    }

    pub fn check_local_admission(&self, used: u64, additional: u64) -> Result<()> {
        self.check_object_size(additional)?;
        let need = used.saturating_add(additional);
        if need > self.max_local_bytes {
            let available = self.max_local_bytes.saturating_sub(used);
            return Err(ObjectError::BudgetExceeded {
                need: additional,
                available,
                budget: format!(
                    "max_local_bytes={} ({})",
                    self.max_local_bytes, self.policy_note
                ),
            });
        }
        Ok(())
    }
}

/// Honest bound statement for contracts and docs.
pub const BOUND_STATEMENT: &str = "Storage is effectively unbounded relative to a single turn or attachment, but is always finite: bounded by configured local/cloud storage (StorageBudget), free disk, model context capability for derivatives, and user policy. Never claim literal unlimited storage.";
