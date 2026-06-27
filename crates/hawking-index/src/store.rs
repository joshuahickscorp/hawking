use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexStoreConfig {
    pub sqlite_path: String,
    pub vector_path: String,
    pub cas_path: String,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreGeneration {
    pub generation: u64,
    pub manifest_hash: String,
    pub sealed: bool,
}
