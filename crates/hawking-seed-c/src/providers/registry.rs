//! The **one capability registry**. There is not a registry per category — one registry answers, for
//! every capability kind, the required questions:
//!
//! - which pack provides a capability?      → [`Registry::providers`]
//! - which implementation is active?        → [`Registry::active`]
//! - why was it selected?                   → [`Selection::reason`]
//! - what Seed ABI does it require?         → [`Selection::hawking_compat`]
//! - what code and bytes does it add?       → [`Selection::loc`] / [`Selection::bytes`]
//! - what source commit produced it?        → [`Selection::source_commit`]
//! - what tests validate it?                → [`Selection::tests`]
//! - what is the rollback?                  → [`Selection::rollback`]
//!
//! Category-specific metadata rides as a typed payload ([`CapabilityKind`]) inside this one registry.
//!
//! ## One activation authority
//!
//! The registry is a **pure selection index**: `activate`/`activate_sole` record which implementation wins
//! a capability and why, but they hold no `&mut Machine`, own no run state, and cannot admit anything. The
//! Seed's [`crate::state::Machine`] is the sole controller. Activation is witnessed through the Seed's ONE
//! evidence engine by exactly one record — a sealed `admission` receipt ([`Selection::admission_receipt`])
//! that the controller records via the Machine. There is no second activation path.

use crate::evidence::receipt;
use crate::pack::{CapabilityKind, PackManifest};
use crate::record::Record;
use crate::{Error, Result};
use serde::Serialize;
use std::collections::BTreeMap;

/// A concrete provider of a capability, discovered from a verified manifest.
#[derive(Debug, Clone, Serialize)]
pub struct ProviderEntry {
    pub capability: String,
    pub kind: CapabilityKind,
    pub pack: String,
    pub implementation: String,
    pub entry: String,
    pub loc: usize,
    pub bytes: usize,
    pub hawking_compat: String,
    pub source_commit: String,
    pub tests: Vec<String>,
    pub rollback: String,
}

/// The active selection for a capability, with the reason it won.
#[derive(Debug, Clone, Serialize)]
pub struct Selection {
    pub capability: String,
    pub kind: CapabilityKind,
    pub pack: String,
    pub implementation: String,
    pub entry: String,
    pub reason: String,
    pub hawking_compat: String,
    pub loc: usize,
    pub bytes: usize,
    pub source_commit: String,
    pub tests: Vec<String>,
    pub rollback: String,
}

impl Selection {
    /// The single activation record: a sealed `admission` receipt witnessing this selection through the
    /// Seed's ONE evidence engine. The registry never mutates run state; the controller records this
    /// receipt via [`crate::state::Machine::record`] — that (and only that) is the activation authority.
    pub fn admission_receipt(&self) -> Record {
        receipt("admission", serde_json::to_value(self).unwrap_or_else(|_| serde_json::json!({})))
    }
}

#[derive(Debug, Default)]
pub struct Registry {
    providers: Vec<ProviderEntry>,
    active: BTreeMap<String, Selection>,
}

impl Registry {
    pub fn new() -> Self {
        Registry::default()
    }

    /// Ingest a verified manifest, indexing every capability→implementation it declares. Only call after
    /// the ONE verifier has accepted the manifest.
    pub fn insert(&mut self, man: &PackManifest) -> Result<()> {
        man.validate_capabilities()?;
        for cap in &man.capabilities {
            let imp = man
                .implementations
                .iter()
                .find(|i| i.id == cap.implementation)
                .ok_or_else(|| Error::Registry(format!("missing impl {}", cap.implementation)))?;
            self.providers.push(ProviderEntry {
                capability: cap.capability.clone(),
                kind: cap.kind,
                pack: man.pack.clone(),
                implementation: imp.id.clone(),
                entry: imp.entry.clone(),
                loc: imp.loc,
                bytes: imp.bytes,
                hawking_compat: man.compatibility.clone(),
                source_commit: man.source_commit.clone(),
                tests: imp.tests.clone(),
                rollback: man.rollback.clone(),
            });
        }
        Ok(())
    }

    /// All providers of a capability.
    pub fn providers(&self, capability: &str) -> Vec<&ProviderEntry> {
        self.providers.iter().filter(|p| p.capability == capability).collect()
    }

    /// All providers of a given kind (e.g. every model adapter).
    pub fn providers_of_kind(&self, kind: CapabilityKind) -> Vec<&ProviderEntry> {
        self.providers.iter().filter(|p| p.kind == kind).collect()
    }

    /// Activate a specific pack's implementation for a capability, recording WHY.
    pub fn activate(&mut self, capability: &str, pack: &str, reason: &str) -> Result<&Selection> {
        let p = self
            .providers
            .iter()
            .find(|p| p.capability == capability && p.pack == pack)
            .ok_or_else(|| Error::Registry(format!("no provider {capability} from {pack}")))?
            .clone();
        let sel = Selection {
            capability: p.capability.clone(),
            kind: p.kind,
            pack: p.pack.clone(),
            implementation: p.implementation.clone(),
            entry: p.entry.clone(),
            reason: reason.into(),
            hawking_compat: p.hawking_compat.clone(),
            loc: p.loc,
            bytes: p.bytes,
            source_commit: p.source_commit.clone(),
            tests: p.tests.clone(),
            rollback: p.rollback.clone(),
        };
        self.active.insert(capability.into(), sel);
        Ok(self.active.get(capability).unwrap())
    }

    /// Activate the single provider of a capability when exactly one exists (the common case).
    pub fn activate_sole(&mut self, capability: &str) -> Result<&Selection> {
        let ps = self.providers(capability);
        match ps.as_slice() {
            [only] => {
                let pack = only.pack.clone();
                self.activate(capability, &pack, "sole verified provider")
            }
            [] => Err(Error::Registry(format!("no provider for {capability}"))),
            _ => Err(Error::Registry(format!(
                "{capability} has {} providers; choose explicitly",
                ps.len()
            ))),
        }
    }

    pub fn active(&self, capability: &str) -> Option<&Selection> {
        self.active.get(capability)
    }

    pub fn active_selections(&self) -> Vec<&Selection> {
        self.active.values().collect()
    }

    /// Total honest owned LOC / bytes across all ACTIVE selections.
    pub fn active_loc(&self) -> usize {
        self.active.values().map(|s| s.loc).sum()
    }
    pub fn active_bytes(&self) -> usize {
        self.active.values().map(|s| s.bytes).sum()
    }

    /// The full registry answer as a canonical JSON value (for artifacts / CLI).
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": "hawking.packs.capability_registry.v1",
            "providers": self.providers,
            "active": self.active.values().collect::<Vec<_>>(),
            "active_loc": self.active_loc(),
            "active_bytes": self.active_bytes(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{Implementation, PackManifest, Profile, SEED_COMPAT};

    fn forge_pack() -> PackManifest {
        PackManifest::capability_pack("packs-nucleus-forge", "1.0.0", Profile::Default)
            .with_source_commit("f0f0")
            .with_rollback("git checkout <c> -- forge")
            .add_implementation(Implementation {
                id: "ternary-latent".into(),
                kind: CapabilityKind::ForgeFamily,
                loc: 120,
                bytes: 4096,
                entry: "forge::TernaryLatentFamily".into(),
                tests: vec!["subbit-under-one-bpw".into()],
            })
            .add_capability("forge.ternary_latent", CapabilityKind::ForgeFamily, "ternary-latent")
    }

    #[test]
    fn one_registry_answers_all_questions() {
        let mut reg = Registry::new();
        reg.insert(&forge_pack()).unwrap();
        assert_eq!(reg.providers("forge.ternary_latent").len(), 1);
        assert_eq!(reg.providers_of_kind(CapabilityKind::ForgeFamily).len(), 1);
        let sel = reg.activate_sole("forge.ternary_latent").unwrap().clone();
        assert_eq!(sel.pack, "packs-nucleus-forge");
        assert_eq!(sel.reason, "sole verified provider");
        assert_eq!(sel.hawking_compat, SEED_COMPAT);
        assert_eq!(sel.loc, 120);
        assert_eq!(sel.source_commit, "f0f0");
        assert_eq!(sel.tests, vec!["subbit-under-one-bpw".to_string()]);
        assert!(sel.rollback.contains("git checkout"));
        assert_eq!(reg.active_loc(), 120);
    }

    #[test]
    fn ambiguous_activation_requires_explicit_choice() {
        let mut reg = Registry::new();
        reg.insert(&forge_pack()).unwrap();
        let mut second = forge_pack();
        second.pack = "packs-nucleus-forge-alt".into();
        reg.insert(&second).unwrap();
        assert!(reg.activate_sole("forge.ternary_latent").is_err());
        assert!(reg.activate("forge.ternary_latent", "packs-nucleus-forge-alt", "measured winner").is_ok());
        assert_eq!(reg.active("forge.ternary_latent").unwrap().pack, "packs-nucleus-forge-alt");
    }

    #[test]
    fn activation_is_witnessed_by_one_sealed_admission_receipt() {
        // The single activation authority: a sealed `admission` record through the Seed's ONE evidence
        // engine (the registry never mutates run state).
        let mut reg = Registry::new();
        reg.insert(&forge_pack()).unwrap();
        let sel = reg.activate_sole("forge.ternary_latent").unwrap();
        let rec = sel.admission_receipt();
        assert!(rec.verify().is_ok());
        assert_eq!(rec.kind, "admission");
        assert!(crate::evidence::is_known_kind(&rec.kind));
    }
}
