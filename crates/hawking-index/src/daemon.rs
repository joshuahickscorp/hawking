use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDaemonConfig {
    pub debounce_ms: u64,
    pub idle_reindex_after_ms: u64,
    pub max_concurrent_lsp: usize,
}

impl Default for IndexDaemonConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 500,
            idle_reindex_after_ms: 15_000,
            max_concurrent_lsp: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDaemonState {
    pub generation: u64,
    pub queue_depth: usize,
    pub is_idle: bool,
    pub last_error: Option<String>,
}
