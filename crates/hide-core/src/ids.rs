use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};
use ulid::Ulid;

thread_local! {
    /// Optional deterministic id source for tests. When set, ids are minted as
    /// `{prefix}_{seed:026X}` with a monotonically increasing seed instead of a
    /// random ULID, so replay/integrity tests are reproducible (T6).
    static DETERMINISTIC_SEED: Cell<Option<u128>> = const { Cell::new(None) };
}

/// Scope guard installing a deterministic, monotonically increasing id source on
/// the current thread for the duration of the closure. Restores the previous
/// source on exit. Intended for tests that assert id monotonicity / replay.
pub fn with_deterministic_ids<T>(start: u128, f: impl FnOnce() -> T) -> T {
    let previous = DETERMINISTIC_SEED.with(|cell| cell.replace(Some(start)));
    let out = f();
    DETERMINISTIC_SEED.with(|cell| cell.set(previous));
    out
}

fn next_ulid_body() -> String {
    if let Some(seed) = DETERMINISTIC_SEED.with(|cell| cell.get()) {
        DETERMINISTIC_SEED.with(|cell| cell.set(Some(seed + 1)));
        // Encode the seed as a ULID so it stays lexicographically sortable and
        // 26 chars wide, matching the random path's format.
        return Ulid::from(seed).to_string();
    }
    Ulid::new().to_string()
}

fn next_id(prefix: &str) -> String {
    format!("{prefix}_{}", next_ulid_body())
}

macro_rules! id_newtype {
    ($name:ident, $prefix:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name(pub String);

        impl $name {
            pub fn new() -> Self {
                Self(next_id($prefix))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

id_newtype!(EventId, "evt");
id_newtype!(SessionId, "ses");
id_newtype!(RunId, "run");
id_newtype!(PlanId, "pln");
id_newtype!(StepId, "stp");
id_newtype!(ToolCallId, "tcl");
id_newtype!(ToolResultId, "trs");
id_newtype!(GrantId, "gnt");
id_newtype!(PluginId, "plg");
id_newtype!(WorkspaceId, "wsp");
id_newtype!(BlobId, "blb");
id_newtype!(ValueId, "val");
id_newtype!(ModelId, "mdl");
id_newtype!(RoleId, "rol");

pub type TimestampMs = u64;

pub fn now_ms() -> TimestampMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Wall-clock microseconds since the Unix epoch — the `ts` field on `Event`
/// (informational; `seq` is the authoritative order, ch.01 §4.6).
pub fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulid_ids_are_sortable_and_unique() {
        // Distinctness and prefix hold for the random path.
        let a = EventId::new();
        let b = EventId::new();
        assert_ne!(a, b);
        assert!(a.as_str().starts_with("evt_"));
        // Lexicographic sortability is a property of TIME-ORDERED ulids. Two random ulids
        // minted in the same millisecond carry random relative order, so assert sortability
        // on the deterministic monotonic source instead (otherwise the test is flaky).
        with_deterministic_ids(0, || {
            let x = EventId::new();
            let y = EventId::new();
            assert!(y.as_str() > x.as_str(), "a later id sorts after an earlier id");
        });
    }

    #[test]
    fn deterministic_ids_are_monotonic_and_reproducible() {
        let first =
            with_deterministic_ids(0, || (0..4).map(|_| EventId::new().0).collect::<Vec<_>>());
        let second =
            with_deterministic_ids(0, || (0..4).map(|_| EventId::new().0).collect::<Vec<_>>());
        assert_eq!(first, second, "same seed yields identical id sequence");
        for pair in first.windows(2) {
            assert!(
                pair[1] > pair[0],
                "deterministic ids are strictly increasing"
            );
        }
    }
}
