use hide_core::ids::{RunId, SessionId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentHandoff {
    pub from_session: SessionId,
    pub to_session: SessionId,
    pub from_run: Option<RunId>,
    pub summary: String,
    pub context_manifest_hash: Option<String>,
    pub kv: Option<KvHandoff>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandoff {
    pub provider: String,
    pub handle: String,
    pub token_count: usize,
    pub lossless: bool,
}
