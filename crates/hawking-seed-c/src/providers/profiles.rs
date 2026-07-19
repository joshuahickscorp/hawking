//! **Profile design.** Real buildable profiles, each reporting LOC / bytes / packs / capabilities / tests
//! / dependencies / offline requirements. Every pack has exactly one home profile
//! ([`crate::pack::Profile`]); the integration nucleus contains only code intended to fold into the Seed.

use crate::pack::{PackManifest, Profile};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize)]
pub struct ProfileReport {
    pub profile: String,
    pub packs: Vec<String>,
    pub capabilities: Vec<String>,
    pub loc: usize,
    pub bytes: usize,
    pub tests: usize,
    pub dependencies: Vec<String>,
    /// Offline requirements: source assets the profile's packs declare they need.
    pub offline_requirements: Vec<String>,
}

/// Summarize a set of manifests grouped by their home profile.
pub fn report(manifests: &[PackManifest]) -> Vec<ProfileReport> {
    let order = [
        Profile::IntegrationNucleus,
        Profile::Default,
        Profile::Optional,
        Profile::Performance,
        Profile::Development,
        Profile::Historical,
    ];
    let mut out = Vec::new();
    for prof in order {
        let members: Vec<&PackManifest> = manifests.iter().filter(|m| m.profile() == prof).collect();
        if members.is_empty() {
            continue;
        }
        let mut caps = BTreeSet::new();
        let mut deps = BTreeSet::new();
        let mut assets = BTreeSet::new();
        let (mut loc, mut bytes, mut tests) = (0usize, 0usize, 0usize);
        for m in &members {
            for c in &m.capabilities {
                caps.insert(c.capability.clone());
            }
            for d in &m.dependencies {
                deps.insert(d.clone());
            }
            for a in &m.assets {
                assets.insert(format!("{}:{}", a.format, a.role));
            }
            loc += m.owned_loc();
            bytes += m.implementations.iter().map(|i| i.bytes).sum::<usize>();
            tests += m.tests.len() + m.implementations.iter().map(|i| i.tests.len()).sum::<usize>();
        }
        out.push(ProfileReport {
            profile: prof.as_str().into(),
            packs: members.iter().map(|m| m.pack.clone()).collect(),
            capabilities: caps.into_iter().collect(),
            loc,
            bytes,
            tests,
            dependencies: deps.into_iter().collect(),
            offline_requirements: assets.into_iter().collect(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{CapabilityKind, Implementation};

    #[test]
    fn profiles_group_and_account_honestly() {
        let a = PackManifest::capability_pack("packs-nucleus-forge", "1.0.0", Profile::Default)
            .add_implementation(Implementation { id: "t".into(), kind: CapabilityKind::ForgeFamily, loc: 120, bytes: 4096, entry: "forge::T".into(), tests: vec!["x".into()] })
            .add_capability("forge.ternary_latent", CapabilityKind::ForgeFamily, "t");
        let b = PackManifest::capability_pack("hide-desktop", "1.0.0", Profile::Historical);
        let rep = report(&[a, b]);
        assert_eq!(rep.len(), 2);
        let def = rep.iter().find(|r| r.profile == "default").unwrap();
        assert_eq!(def.loc, 120);
        assert_eq!(def.tests, 1);
        assert_eq!(def.capabilities, vec!["forge.ternary_latent".to_string()]);
    }
}
