//! Native admission fence for the Python-owned HCLI tool dispatcher.
//!
//! This is intentionally below execution: it checks that a requested name is
//! in the immutable catalog, its mutation class has not been substituted, and
//! the caller holds that class. Python remains the handler/schema authority
//! until each executor earns an independent native replacement.

use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDispatchEntry {
    pub name: String,
    pub mutation: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolDispatchCatalog {
    mutations: BTreeMap<String, String>,
}

impl ToolDispatchCatalog {
    pub fn from_entries(entries: Vec<ToolDispatchEntry>) -> Result<Self> {
        let mut mutations = BTreeMap::new();
        for entry in entries {
            if entry.name.is_empty() || entry.name.chars().any(char::is_whitespace) {
                return Err(Error::Gravity(format!(
                    "invalid dispatch tool name {:?}",
                    entry.name
                )));
            }
            if entry.mutation.is_empty() || entry.mutation.chars().any(char::is_whitespace) {
                return Err(Error::Gravity(format!(
                    "invalid mutation for {}",
                    entry.name
                )));
            }
            if mutations
                .insert(entry.name.clone(), entry.mutation)
                .is_some()
            {
                return Err(Error::Gravity(format!(
                    "duplicate dispatch tool {}",
                    entry.name
                )));
            }
        }
        Ok(Self { mutations })
    }

    /// Return a refusal reason rather than silently falling through to a
    /// different caller-owned policy path.
    pub fn admit(
        &self,
        name: &str,
        claimed_mutation: &str,
        permissions: &[String],
    ) -> DispatchAdmission {
        let Some(declared) = self.mutations.get(name) else {
            return DispatchAdmission::refuse("tool is absent from native dispatch catalog");
        };
        if declared != claimed_mutation {
            return DispatchAdmission::refuse("tool mutation does not match native catalog");
        }
        let allowed = permissions.iter().collect::<BTreeSet<_>>();
        if !allowed.contains(declared) {
            return DispatchAdmission::refuse("permission denied for declared mutation class");
        }
        DispatchAdmission {
            admitted: true,
            mutation: Some(declared.clone()),
            reason: None,
        }
    }

    pub fn len(&self) -> usize {
        self.mutations.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchAdmission {
    pub admitted: bool,
    pub mutation: Option<String>,
    pub reason: Option<String>,
}

impl DispatchAdmission {
    fn refuse(reason: &str) -> Self {
        Self {
            admitted: false,
            mutation: None,
            reason: Some(reason.to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> ToolDispatchCatalog {
        ToolDispatchCatalog::from_entries(vec![ToolDispatchEntry {
            name: "fs.read".into(),
            mutation: "read_only".into(),
        }])
        .unwrap()
    }

    #[test]
    fn declared_mutation_and_permission_must_both_match() {
        assert!(
            catalog()
                .admit("fs.read", "read_only", &["read_only".into()])
                .admitted
        );
        assert!(
            !catalog()
                .admit("fs.read", "repo_write", &["repo_write".into()])
                .admitted
        );
        assert!(
            !catalog()
                .admit("absent", "read_only", &["read_only".into()])
                .admitted
        );
    }

    #[test]
    fn duplicate_or_unsafe_entries_fail_closed() {
        assert!(ToolDispatchCatalog::from_entries(vec![ToolDispatchEntry {
            name: "bad name".into(),
            mutation: "read_only".into()
        }])
        .is_err());
    }
}
