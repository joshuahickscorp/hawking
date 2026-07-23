use crate::ids::WorkspaceId;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workspace {
    pub id: WorkspaceId,
    pub root: PathBuf,
    pub hide_dir: PathBuf,
}

impl Workspace {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let hide_dir = root.join(".hide");
        Self {
            id: WorkspaceId::new(),
            root,
            hide_dir,
        }
    }

    pub fn layout(&self) -> WorkspaceLayout {
        WorkspaceLayout::new(&self.root)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceLayout {
    pub root: PathBuf,
    pub hide_dir: PathBuf,
    pub event_log: PathBuf,
    pub snapshots: PathBuf,
    pub projections: PathBuf,
    pub metadata_db: PathBuf,
    pub kv: PathBuf,
    pub vectors_db: PathBuf,
    pub blobs: PathBuf,
    pub taint: PathBuf,
    pub cache: PathBuf,
    pub sandbox: PathBuf,
    pub tmp: PathBuf,
}

impl WorkspaceLayout {
    pub fn new(root: &Path) -> Self {
        let hide_dir = root.join(".hide");
        Self {
            root: root.to_path_buf(),
            event_log: hide_dir.join("log"),
            snapshots: hide_dir.join("snapshots"),
            projections: hide_dir.join("projections"),
            metadata_db: hide_dir.join("meta.sqlite"),
            kv: hide_dir.join("kv"),
            vectors_db: hide_dir.join("vectors.sqlite"),
            blobs: hide_dir.join("blobs"),
            taint: hide_dir.join("taint"),
            cache: hide_dir.join("cache"),
            sandbox: hide_dir.join("sandbox"),
            tmp: hide_dir.join("tmp"),
            hide_dir,
        }
    }
}
