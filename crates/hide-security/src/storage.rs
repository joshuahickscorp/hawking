use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtRestPolicy {
    pub enabled: bool,
    pub key_ref: Option<String>,
    pub encrypt_event_log: bool,
    pub encrypt_blobs: bool,
    pub encrypt_metadata: bool,
    pub plaintext_cache_allowed: bool,
}

impl Default for AtRestPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            key_ref: None,
            encrypt_event_log: false,
            encrypt_blobs: false,
            encrypt_metadata: false,
            plaintext_cache_allowed: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutValidation {
    pub ok: bool,
    pub root_mode_owner_only: bool,
    pub hide_log_agent_writable: bool,
    pub warnings: Vec<String>,
}
