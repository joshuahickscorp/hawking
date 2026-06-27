use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleNode {
    pub path: PathBuf,
    pub hash: String,
    pub kind: MerkleKind,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MerkleKind {
    File,
    Directory,
    Symlink,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChangeSet {
    pub added: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub removed: Vec<PathBuf>,
    pub renamed: Vec<(PathBuf, PathBuf)>,
}

pub trait MerkleScanner: Send + Sync {
    fn scan_workspace(&self) -> hide_core::Result<MerkleNode>;
    fn diff(&self, old: &MerkleNode, new: &MerkleNode) -> hide_core::Result<ChangeSet>;
}
