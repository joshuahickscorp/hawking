//! `local_folder` connector: read a directory under an account-bound root.
//!
//! Read-only. Does **not** implement [`crate::connector_abi::connector::ConnectorWrite`] — that
//! is the type boundary for safety property 1.

use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::connector_abi::abi::{ConnectorAbi, FamilyId};
use crate::connector_abi::account::{AccountHandle, AccountStore, InFlightGuard};
use crate::connector_abi::connector::{
    BTreeMapStr, Connector, ConnectorObject, ConnectorRead, ListRequest, ReadRequest,
};
use crate::connector_abi::error::{ConnectorError, Result};
use crate::connector_abi::families;

/// Live local-folder connector. Account credential material is the absolute
/// root path the account may read.
#[derive(Debug)]
pub struct LocalFolderConnector {
    abi: ConnectorAbi,
}

impl LocalFolderConnector {
    pub fn new() -> Self {
        Self {
            abi: families::local_folder(),
        }
    }

    fn root_from_handle(&self, handle: &AccountHandle) -> Result<PathBuf> {
        let raw = handle.credential_material();
        if raw.is_empty() {
            return Err(ConnectorError::InvalidRequest(
                "local_folder account root path is empty".into(),
            ));
        }
        let p = PathBuf::from(raw);
        if !p.is_absolute() {
            return Err(ConnectorError::InvalidRequest(format!(
                "local_folder root must be absolute, got {raw}"
            )));
        }
        Ok(p)
    }

    /// Resolve a locator under root without allowing `..` escape.
    fn resolve(&self, root: &Path, locator: &str) -> Result<PathBuf> {
        let rel = Path::new(locator.trim_start_matches('/'));
        for c in rel.components() {
            match c {
                Component::Normal(_) | Component::CurDir => {}
                Component::ParentDir => {
                    return Err(ConnectorError::InvalidRequest(
                        "path escape (..) forbidden".into(),
                    ));
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(ConnectorError::InvalidRequest(
                        "absolute locator forbidden; use root-relative paths".into(),
                    ));
                }
            }
        }
        let joined = root.join(rel);
        // Canonicalize when the path exists; otherwise keep joined for not-found.
        if joined.exists() {
            let canon = joined.canonicalize().map_err(ConnectorError::from)?;
            let root_canon = root.canonicalize().map_err(ConnectorError::from)?;
            if !canon.starts_with(&root_canon) {
                return Err(ConnectorError::InvalidRequest(
                    "resolved path escapes account root".into(),
                ));
            }
            Ok(canon)
        } else {
            Ok(joined)
        }
    }

    fn entry_to_object(&self, root: &Path, path: &Path) -> Result<ConnectorObject> {
        let meta = fs::metadata(path).map_err(ConnectorError::from)?;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let id = if rel.is_empty() { ".".to_string() } else { rel };
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| id.clone());
        let object_type = if meta.is_dir() { "directory" } else { "file" };
        let mut metadata = BTreeMapStr::new();
        metadata.insert("size", meta.len().to_string());
        metadata.insert("is_dir", meta.is_dir().to_string());
        Ok(ConnectorObject {
            id,
            object_type: object_type.into(),
            title: name,
            content: None,
            metadata,
        })
    }
}

impl Default for LocalFolderConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for LocalFolderConnector {
    fn family_id(&self) -> &FamilyId {
        &self.abi.family_id
    }
    fn abi(&self) -> &ConnectorAbi {
        &self.abi
    }
}

impl ConnectorRead for LocalFolderConnector {
    fn list(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ListRequest,
    ) -> Result<Vec<ConnectorObject>> {
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let root = self.root_from_handle(handle)?;
        let dir = match &request.prefix {
            Some(p) if !p.is_empty() && p != "." => self.resolve(&root, p)?,
            _ => root.canonicalize().map_err(ConnectorError::from)?,
        };
        if !dir.is_dir() {
            return Err(ConnectorError::NotFound(format!(
                "not a directory: {}",
                dir.display()
            )));
        }
        let root_canon = root.canonicalize().map_err(ConnectorError::from)?;
        let mut out = Vec::new();
        let rd = fs::read_dir(&dir).map_err(ConnectorError::from)?;
        for ent in rd {
            let ent = ent.map_err(ConnectorError::from)?;
            let path = ent.path();
            out.push(self.entry_to_object(&root_canon, &path)?);
            if out.len() >= request.limit {
                break;
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        // Fail closed if revoked mid-flight before returning results.
        guard.complete(store)?;
        Ok(out)
    }

    fn fetch(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ReadRequest,
    ) -> Result<ConnectorObject> {
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let root = self.root_from_handle(handle)?;
        let root_canon = root.canonicalize().map_err(ConnectorError::from)?;
        let path = self.resolve(&root, &request.locator)?;
        if !path.exists() {
            return Err(ConnectorError::NotFound(request.locator.clone()));
        }
        let mut obj = self.entry_to_object(&root_canon, &path)?;
        if path.is_file() {
            // Read text when UTF-8; otherwise leave content empty and note binary.
            match fs::read_to_string(&path) {
                Ok(s) => obj.content = Some(s),
                Err(_) => {
                    obj.metadata.insert("binary", "true");
                }
            }
        }
        guard.complete(store)?;
        Ok(obj)
    }
}

// Deliberately no `impl ConnectorWrite for LocalFolderConnector`.
// That absence is the type boundary for default read-only / least privilege.
