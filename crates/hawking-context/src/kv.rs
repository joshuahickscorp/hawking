use hide_core::ids::{RunId, SessionId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandle {
    pub provider: String,
    pub key: String,
    pub tokens: usize,
    pub tier: KvTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvTier {
    Gpu,
    Ram,
    Disk,
    Remote,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCheckpoint {
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub handles: Vec<KvHandle>,
    pub tokenizer_signature: String,
}

pub trait KvStoreClient: Send + Sync {
    fn lookup_prefix(&self, token_hash: &str) -> Option<KvHandle>;
    fn checkpoint(&self, session_id: &SessionId, run_id: Option<&RunId>) -> Option<KvCheckpoint>;
}
