use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id(prefix: &str) -> String {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{ms:013x}_{n:08x}")
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
