use thiserror::Error;

use crate::manifest::Effect;

/// Errors surfaced by the capability registry. Registration is strict on the
/// safety-relevant paths (duplicate ids, undeclared effects, pin violations)
/// because those are the invariants the registry exists to hold.
#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("capability id {0:?} is already registered")]
    DuplicateId(String),

    #[error("no capability registered with id {0:?}")]
    NotFound(String),

    #[error("capability {0:?} has been revoked")]
    Revoked(String),

    #[error("capability {id:?} declares effects {missing} that its policies and scopes require but the effects list omits")]
    UndeclaredEffects { id: String, missing: EffectList },

    #[error("capability {id:?} violates pin: {detail}")]
    PinViolation { id: String, detail: String },

    #[error("capability {id:?} is invalid: {detail}")]
    InvalidManifest { id: String, detail: String },

    #[error("schema parse error for capability {id:?} ({which}): {source}")]
    Schema {
        id: String,
        which: &'static str,
        #[source]
        source: serde_json::Error,
    },
}

pub type Result<T> = std::result::Result<T, RegistryError>;

/// A comma-joined list of effect names, used in error messages so the missing
/// declarations are legible without a custom Display on `Vec<Effect>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectList(pub Vec<Effect>);

impl std::fmt::Display for EffectList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.0.iter().map(|e| e.as_str()).collect();
        write!(f, "[{}]", names.join(", "))
    }
}
