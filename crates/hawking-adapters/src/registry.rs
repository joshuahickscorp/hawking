//! One registry indexing all family descriptors.

use std::collections::BTreeMap;

use crate::abi::FamilyDescriptor;
use crate::evidence::{validate_family_evidence, workspace_root};
use crate::families;
use crate::support_level::SupportLevel;

/// The sole architecture-family registry.
pub struct FamilyRegistry {
    by_id: BTreeMap<String, FamilyDescriptor>,
}

impl FamilyRegistry {
    pub fn new() -> Self {
        Self {
            by_id: BTreeMap::new(),
        }
    }

    pub fn insert_descriptor(&mut self, d: FamilyDescriptor) {
        self.by_id.insert(d.id.to_string(), d);
    }

    pub fn get(&self, id: &str) -> Option<&FamilyDescriptor> {
        self.by_id
            .get(id)
            .or_else(|| self.by_id.values().find(|d| d.aliases.contains(&id)))
    }

    pub fn families(&self) -> impl Iterator<Item = &FamilyDescriptor> {
        self.by_id.values()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Every declared family's support level must be backed by evidence, and
    /// every ABI field must be complete (value or null+reason).
    pub fn validate_all_evidence(&self) -> Result<(), Vec<String>> {
        let root = workspace_root();
        let mut errs = Vec::new();
        for d in self.families() {
            if let Err(e) = validate_family_evidence(&root, d) {
                errs.push(e);
            }
            if d.level == SupportLevel::Production {
                errs.push(format!(
                    "family {}: PRODUCTION forbidden (no family is PRODUCTION today)",
                    d.id
                ));
            }
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }

    /// ABI completeness for every family.
    pub fn validate_all_abi(&self) -> Result<(), Vec<String>> {
        let mut errs = Vec::new();
        for d in self.families() {
            if let Err(e) = d.abi.validate_complete(d.id) {
                errs.extend(e);
            }
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}

impl Default for FamilyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in registry with every in-tree family descriptor.
pub fn builtin_registry() -> FamilyRegistry {
    let mut r = FamilyRegistry::new();
    for d in families::FAMILY_TABLE {
        r.insert_descriptor(*d);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn builtin_has_expected_families() {
        let r = builtin_registry();
        assert_eq!(r.len(), 10);
        for id in [
            "llama",
            "mistral_mixtral",
            "qwen",
            "glm",
            "deepseek",
            "kimi",
            "minimax",
            "gemma",
            "phi",
            "state_space",
        ] {
            assert!(r.get(id).is_some(), "missing {id}");
        }
    }
    #[test]
    fn aliases_resolve() {
        let r = builtin_registry();
        assert_eq!(r.get("llama3").unwrap().id, "llama");
        assert_eq!(r.get("gemma2").unwrap().id, "gemma");
        assert_eq!(r.get("rwkv7").unwrap().id, "state_space");
    }
    #[test]
    fn no_family_is_production() {
        let r = builtin_registry();
        for d in r.families() {
            assert_ne!(
                d.level,
                SupportLevel::Production,
                "{} must not be PRODUCTION",
                d.id
            );
        }
    }
    #[test]
    fn qwen_is_source_header_not_higher_without_live_checkpoint() {
        let r = builtin_registry();
        let q = r.get("qwen").unwrap();
        assert_eq!(q.level, SupportLevel::SourceHeaderValidated);
        assert!(q.executes);
        assert!(q.serve_registered);
        assert!(q
            .evidence
            .iter()
            .any(|e| e.path.contains("ADAPTER_QWEN_RECEIPT")));
    }
    #[test]
    fn glm_is_small_real_checkpoint() {
        let r = builtin_registry();
        let g = r.get("glm").unwrap();
        assert_eq!(g.level, SupportLevel::SmallRealCheckpoint);
        assert!(g.evidence.iter().any(|e| e.path.contains("GLM52_FLAGSHIP")));
    }
    #[test]
    fn kimi_is_synthetic_not_serve_registered() {
        let r = builtin_registry();
        let k = r.get("kimi").unwrap();
        assert_eq!(k.level, SupportLevel::SyntheticParity);
        assert!(!k.serve_registered);
        assert!(!k.executes);
    }
    #[test]
    fn every_family_abi_complete() {
        let r = builtin_registry();
        r.validate_all_abi()
            .unwrap_or_else(|e| panic!("ABI incomplete:\n{}", e.join("\n")));
    }
}
