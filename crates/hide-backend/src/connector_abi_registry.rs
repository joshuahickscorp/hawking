//! Connector registry: every family ABI, construct only the implemented ones.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::connector_abi::abi::{ConnectorAbi, FamilyId, ImplementationStatus};
use crate::connector_abi::connector::{Connector, ConnectorRead, DeclaredConnector};
use crate::connector_abi::error::{ConnectorError, Result};
use crate::connector_abi::families;
use crate::connector_abi::impls::{LocalFolderConnector, RssConnector};

/// What the registry can hand back as a live connector.
#[derive(Debug)]
pub enum LiveConnector {
    LocalFolder(LocalFolderConnector),
    Rss(RssConnector),
}

impl LiveConnector {
    pub fn family_id(&self) -> &FamilyId {
        match self {
            Self::LocalFolder(c) => c.family_id(),
            Self::Rss(c) => c.family_id(),
        }
    }

    pub fn as_read(&self) -> &dyn ConnectorRead {
        match self {
            Self::LocalFolder(c) => c,
            Self::Rss(c) => c,
        }
    }
}

/// The YOU connector registry.
pub struct ConnectorRegistry {
    by_id: BTreeMap<String, ConnectorAbi>,
}

impl ConnectorRegistry {
    /// Built-in registry with every family declaration.
    pub fn builtin() -> Self {
        let mut by_id = BTreeMap::new();
        for abi in families::all_families() {
            by_id.insert(abi.family_id.as_str().to_string(), abi);
        }
        Self { by_id }
    }

    pub fn get(&self, id: &str) -> Option<&ConnectorAbi> {
        self.by_id.get(id)
    }

    pub fn families(&self) -> impl Iterator<Item = &ConnectorAbi> {
        self.by_id.values()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn implemented(&self) -> Vec<&ConnectorAbi> {
        self.families()
            .filter(|a| a.status == ImplementationStatus::Implemented)
            .collect()
    }

    pub fn declared(&self) -> Vec<&ConnectorAbi> {
        self.families()
            .filter(|a| a.status == ImplementationStatus::Declared)
            .collect()
    }

    /// Validate every ABI for internal consistency.
    pub fn validate_all(&self) -> std::result::Result<(), Vec<String>> {
        let mut errs = Vec::new();
        for a in self.families() {
            if let Err(e) = a.validate() {
                errs.extend(e);
            }
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }

    /// Construct a live connector. Declared families return
    /// [`ConnectorError::DeclaredNotConstructible`].
    pub fn construct(&self, family_id: &str) -> Result<LiveConnector> {
        let abi = self
            .by_id
            .get(family_id)
            .ok_or_else(|| ConnectorError::UnknownFamily(FamilyId::new(family_id)))?;
        match abi.status {
            ImplementationStatus::Declared => {
                // Explicit: declared connectors are not constructible.
                let _ = DeclaredConnector::try_construct(abi.family_id.clone());
                Err(ConnectorError::DeclaredNotConstructible(
                    abi.family_id.clone(),
                ))
            }
            ImplementationStatus::Implemented => match family_id {
                "local_folder" => Ok(LiveConnector::LocalFolder(LocalFolderConnector::new())),
                "rss" => Ok(LiveConnector::Rss(RssConnector::new())),
                other => Err(ConnectorError::InvalidRequest(format!(
                    "family {other} marked implemented but has no constructor"
                ))),
            },
        }
    }

    /// Export the registry document for `evidence/hide/HIDE_YOU_CONNECTOR_REGISTRY.json`.
    pub fn export_document(&self) -> RegistryDocument {
        let mut families: Vec<ConnectorAbi> = self.families().cloned().collect();
        families.sort_by(|a, b| a.family_id.as_str().cmp(b.family_id.as_str()));
        let implemented: Vec<String> = families
            .iter()
            .filter(|a| a.status == ImplementationStatus::Implemented)
            .map(|a| a.family_id.as_str().to_string())
            .collect();
        let declared: Vec<String> = families
            .iter()
            .filter(|a| a.status == ImplementationStatus::Declared)
            .map(|a| a.family_id.as_str().to_string())
            .collect();
        RegistryDocument {
            schema: "hide.you.connector_registry.v1".into(),
            surface: "YOU".into(),
            crate_name: "hide-connectors".into(),
            safety_properties: vec![
                "default_read_only_type_boundary".into(),
                "no_ambient_credentials".into(),
                "every_write_is_effect_with_receipt".into(),
                "connector_data_not_silent_global_memory".into(),
                "revocation_fail_closed".into(),
            ],
            implemented,
            declared,
            families,
            notes: vec![
                "Only local_folder and rss are constructible; all others refuse construction.".into(),
                "No real credentials, OAuth flows, or network calls in this crate.".into(),
                "Write-capable families declare write in ABI but have no ConnectorWrite impl until implemented.".into(),
            ],
        }
    }

    /// Write the registry JSON to a path (deterministic pretty JSON).
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let doc = self.export_document();
        let text =
            serde_json::to_string_pretty(&doc).map_err(|e| ConnectorError::Parse(e.to_string()))?;
        std::fs::write(path, text + "\n").map_err(ConnectorError::from)
    }
}

impl Default for ConnectorRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

/// Top-level document for `evidence/hide/HIDE_YOU_CONNECTOR_REGISTRY.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryDocument {
    pub schema: String,
    pub surface: String,
    pub crate_name: String,
    pub safety_properties: Vec<String>,
    pub implemented: Vec<String>,
    pub declared: Vec<String>,
    pub families: Vec<ConnectorAbi>,
    pub notes: Vec<String>,
}
