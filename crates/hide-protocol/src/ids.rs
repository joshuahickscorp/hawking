//! Typed identifiers for the semantic object model.
//!
//! Every object in the model (Bible sec 14) is addressed by a transparent
//! string newtype. They serialize as bare strings (via `#[serde(transparent)]`)
//! so the wire carries `"ses_..."` rather than `{ "0": "ses_..." }`, and they
//! generate a plain string JSON Schema. These are pure address types: they mint
//! nothing and carry no minting policy, so this crate stays model-free and
//! deterministic. A live runtime assigns the actual id values.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash,
            Serialize, Deserialize, JsonSchema,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            /// Wrap an existing id value. This crate never generates ids; a
            /// caller (the runtime, a test fixture, or the compat bridge) owns
            /// the value.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
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

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

id_newtype!(
    /// A workspace: the outermost container binding repositories, environments,
    /// and sessions together.
    WorkspaceId
);
id_newtype!(RepositoryId);
id_newtype!(EnvironmentId);
id_newtype!(SessionId);
id_newtype!(ThreadId);
id_newtype!(TurnId);
id_newtype!(ItemId);
id_newtype!(GoalId);
id_newtype!(PlanId);
id_newtype!(StepId);
id_newtype!(ArtifactId);
id_newtype!(CheckpointId);
id_newtype!(StateCapsuleId);
id_newtype!(AgentId);
id_newtype!(ToolId);
id_newtype!(ToolCallId);
id_newtype!(OracleId);
id_newtype!(ApprovalId);
id_newtype!(VerificationId);
id_newtype!(
    /// Correlation id on a protocol request/response pair (Bible sec 15).
    RequestId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_serialize_transparently_as_bare_strings() {
        let id = SessionId::from("ses_abc");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"ses_abc\"");
        let back: SessionId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn ids_display_and_as_str_agree() {
        let id = TurnId::new("trn_1");
        assert_eq!(id.as_str(), "trn_1");
        assert_eq!(id.to_string(), "trn_1");
    }
}
