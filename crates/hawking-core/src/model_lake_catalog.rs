//! Read-only, deterministic ModelLake catalog access for Gravity.
//!
//! The catalog is deliberately outside sealed specimens.  This module never
//! walks a lake, opens weights, or mutates an index: it turns an already-built
//! Python-era catalog into a compact native query owner so admission, routing,
//! and retention planning do not repeatedly import the full research stack.

use crate::{Error, Result};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const CATALOG_SCHEMA: &str = "hawking.modellake.index.catalog.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLakeCatalog {
    path: PathBuf,
    specimens: Vec<ModelLakeSpecimen>,
    pub specimen_count: u64,
    pub partial_count: u64,
    pub used_bytes: u64,
    pub budget_bytes: u64,
    pub over_budget: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLakeSpecimen {
    pub slug: String,
    pub repository: Option<String>,
    pub revision: Option<String>,
    pub bytes: u64,
    pub architecture_family: String,
    pub lifecycle: String,
    pub seal_status: String,
    pub location: String,
}

impl ModelLakeCatalog {
    /// Load a catalog with a stable schema and canonical lexical specimen order.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bytes = fs::read(&path).map_err(|error| {
            Error::Gravity(format!(
                "cannot read ModelLake catalog {}: {error}",
                path.display()
            ))
        })?;
        let document: Value = serde_json::from_slice(&bytes).map_err(|error| {
            Error::Gravity(format!(
                "invalid ModelLake catalog {}: {error}",
                path.display()
            ))
        })?;
        if document.get("schema").and_then(Value::as_str) != Some(CATALOG_SCHEMA) {
            return Err(Error::Gravity(format!(
                "ModelLake catalog {} does not declare {CATALOG_SCHEMA}",
                path.display()
            )));
        }
        let rows = document
            .get("specimens")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                Error::Gravity(format!(
                    "ModelLake catalog {} has no specimens array",
                    path.display()
                ))
            })?;
        let mut specimens = rows
            .iter()
            .map(parse_specimen)
            .collect::<Result<Vec<_>>>()?;
        specimens.sort_by(|left, right| left.slug.cmp(&right.slug));
        for pair in specimens.windows(2) {
            if pair[0].slug == pair[1].slug {
                return Err(Error::Gravity(format!(
                    "ModelLake catalog {} has duplicate slug {}",
                    path.display(),
                    pair[0].slug
                )));
            }
        }
        Ok(Self {
            path,
            specimen_count: required_u64(&document, "n_specimens")?,
            partial_count: required_u64(&document, "n_partial")?,
            used_bytes: required_u64(&document, "tier2_used_bytes")?,
            budget_bytes: required_u64(&document, "tier2_budget")?,
            over_budget: document
                .get("over_budget")
                .and_then(Value::as_bool)
                .ok_or_else(|| {
                    Error::Gravity("ModelLake catalog over_budget must be boolean".into())
                })?,
            specimens,
        })
    }

    /// O(log n) lookup.  Slugs are deliberately path-safe identifiers, never paths.
    pub fn specimen(&self, slug: &str) -> Result<Option<&ModelLakeSpecimen>> {
        validate_slug(slug)?;
        Ok(self
            .specimens
            .binary_search_by(|item| item.slug.as_str().cmp(slug))
            .ok()
            .map(|index| &self.specimens[index]))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn specimens(&self) -> &[ModelLakeSpecimen] {
        &self.specimens
    }
}

fn parse_specimen(row: &Value) -> Result<ModelLakeSpecimen> {
    let object = row
        .as_object()
        .ok_or_else(|| Error::Gravity("ModelLake specimen row must be object".into()))?;
    let slug = object
        .get("slug")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Gravity("ModelLake specimen slug missing".into()))?
        .to_owned();
    validate_slug(&slug)?;
    Ok(ModelLakeSpecimen {
        slug,
        repository: optional_string(object.get("repo")),
        revision: optional_string(object.get("revision")),
        bytes: required_u64(row, "bytes")?,
        architecture_family: required_string(row, "architecture_family")?,
        lifecycle: required_string(row, "lifecycle")?,
        seal_status: required_string(row, "seal_status")?,
        location: required_string(row, "location")?,
    })
}

fn validate_slug(slug: &str) -> Result<()> {
    if slug.is_empty()
        || slug.starts_with('.')
        || slug.contains('/')
        || slug.contains('\\')
        || slug.contains("..")
    {
        return Err(Error::Gravity(format!("unsafe ModelLake slug {slug:?}")));
    }
    Ok(())
}

fn required_u64(value: &Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Gravity(format!("ModelLake catalog {key} must be unsigned integer")))
}

fn required_string(value: &Value, key: &str) -> Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| Error::Gravity(format!("ModelLake catalog {key} must be non-empty string")))
}

fn optional_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn fixture() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hawking-catalog-{}-{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&path, r#"{"schema":"hawking.modellake.index.catalog.v1","n_specimens":1,"n_partial":0,"tier2_used_bytes":12,"tier2_budget":99,"over_budget":false,"specimens":[{"slug":"Qwen--Tiny@abc","repo":"Qwen/Tiny","revision":"abc","bytes":12,"architecture_family":"qwen","lifecycle":"SEALED","seal_status":"SEALED","location":"specimens"}]}"#).unwrap();
        path
    }

    #[test]
    fn loads_and_looks_up_without_scanning_a_lake() {
        let path = fixture();
        let catalog = ModelLakeCatalog::load(&path).unwrap();
        assert_eq!(
            catalog
                .specimen("Qwen--Tiny@abc")
                .unwrap()
                .unwrap()
                .architecture_family,
            "qwen"
        );
        assert!(catalog.specimen("missing").unwrap().is_none());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_unsafe_slugs_and_wrong_schema() {
        assert!(validate_slug("../escape").is_err());
        let path =
            std::env::temp_dir().join(format!("hawking-catalog-bad-{}.json", std::process::id()));
        fs::write(&path, "{}").unwrap();
        assert!(ModelLakeCatalog::load(&path).is_err());
        fs::remove_file(path).unwrap();
    }
}
