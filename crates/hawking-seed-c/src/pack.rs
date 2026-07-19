//! Pack ABI + verification. A pack declares identity, version, compatibility, hashes, and contents.
//! Seed verifies content hashes offline, rejects tampering, and supports rollback. No arbitrary
//! plugin execution: content-addressed verified packs only. Reused from Candidate A.

use crate::record::{sha256_hex, Record};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The one Pack ABI compatibility string the Seed authority targets. A pack declaring this (or an empty
/// string) is compatible. Single source of truth for the CLI and every absorbed provider.
pub const SEED_COMPAT: &str = "seed-c-1";

// --- capability schema (absorbed from the packs nucleus manifest). This is a strictly ADDITIVE extension
// of the Pack ABI: every field below is `serde(default)` + skipped when empty, and is EXCLUDED from the
// content-identity block (`content_identity`), so every previously sealed pack manifest still serializes
// and canonicalizes byte-for-byte identically. `PackManifest` is the ONE manifest; there is no parallel. ---

/// The kinds of capability the one registry can hold. Category-specific metadata rides as a typed payload
/// inside the one registry; there is no separate registry per category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKind {
    ModelAdapter,
    ForgeFamily,
    DoctorTreatment,
    CompactOperator,
    MetalImpl,
    SpeculationProvider,
    ValidationSuite,
    Laboratory,
    ClientExtension,
    CompatibilityReader,
}

impl CapabilityKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            CapabilityKind::ModelAdapter => "model_adapter",
            CapabilityKind::ForgeFamily => "forge_family",
            CapabilityKind::DoctorTreatment => "doctor_treatment",
            CapabilityKind::CompactOperator => "compact_operator",
            CapabilityKind::MetalImpl => "metal_impl",
            CapabilityKind::SpeculationProvider => "speculation_provider",
            CapabilityKind::ValidationSuite => "validation_suite",
            CapabilityKind::Laboratory => "laboratory",
            CapabilityKind::ClientExtension => "client_extension",
            CapabilityKind::CompatibilityReader => "compatibility_reader",
        }
    }
}

/// Which buildable profile a pack belongs to. Every pack has exactly one home profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    IntegrationNucleus,
    #[default]
    Default,
    Optional,
    Performance,
    Development,
    Historical,
}

impl Profile {
    pub fn as_str(&self) -> &'static str {
        match self {
            Profile::IntegrationNucleus => "integration_nucleus",
            Profile::Default => "default",
            Profile::Optional => "optional",
            Profile::Performance => "performance",
            Profile::Development => "development",
            Profile::Historical => "historical",
        }
    }
}

/// A capability the pack provides, bound to the implementation that realizes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityDecl {
    pub capability: String,
    pub kind: CapabilityKind,
    pub implementation: String,
}

/// One implementation inside the pack, with honest owned-code accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Implementation {
    pub id: String,
    pub kind: CapabilityKind,
    /// Owned lines of code for THIS implementation (declarative descriptors are small by construction).
    pub loc: usize,
    /// Owned bytes for THIS implementation.
    pub bytes: usize,
    /// The provider entrypoint symbol/type that the Seed activates.
    pub entry: String,
    /// Validation manifest ids that prove this implementation.
    #[serde(default)]
    pub tests: Vec<String>,
}

/// A required source or protocol asset. Packs DECLARE assets; the Seed owns their identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub role: String,
    pub format: String,
    pub identity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackEntry {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PackManifest {
    pub pack: String,
    pub version: String,
    #[serde(default)]
    pub compatibility: String,
    #[serde(default)]
    pub source_commit: String,
    #[serde(default)]
    pub contents: Vec<PackEntry>,
    #[serde(default)]
    pub offline_cache: String,

    // --- ADDITIVE capability schema (excluded from `content_identity`; skipped when empty so existing
    // manifests serialize identically). ---
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<Profile>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<CapabilityDecl>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub implementations: Vec<Implementation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<Asset>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tests: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub rollback: String,
}

impl PackManifest {
    fn base_dir(&self, manifest_path: &Path) -> PathBuf {
        if !self.offline_cache.is_empty() {
            PathBuf::from(&self.offline_cache)
        } else {
            manifest_path.parent().unwrap_or(Path::new(".")).to_path_buf()
        }
    }

    /// Verify every declared content hash offline. Returns Err on the first tamper/missing file.
    pub fn verify(&self, manifest_path: impl AsRef<Path>) -> Result<usize> {
        let base = self.base_dir(manifest_path.as_ref());
        let mut ok = 0usize;
        for e in &self.contents {
            let p = if Path::new(&e.path).is_absolute() {
                PathBuf::from(&e.path)
            } else {
                base.join(&e.path)
            };
            let bytes = std::fs::read(&p)
                .map_err(|_| Error::Pack(format!("pack {}: missing content {}", self.pack, e.path)))?;
            let got = sha256_hex(&bytes);
            if got != e.sha256 {
                // Typed tamper signal (Display keeps the literal "TAMPERED" contract); callers match the
                // variant instead of scraping the message string.
                return Err(Error::Tamper {
                    pack: self.pack.clone(),
                    path: e.path.clone(),
                    got,
                    declared: e.sha256.clone(),
                });
            }
            ok += 1;
        }
        Ok(ok)
    }

    pub fn compatible_with(&self, seed_compat: &str) -> bool {
        self.compatibility.is_empty() || self.compatibility == seed_compat
    }

    /// The home profile (defaults to `Profile::Default` for legacy manifests that never declared one).
    pub fn profile(&self) -> Profile {
        self.profile.unwrap_or(Profile::Default)
    }

    /// Total honest owned LOC contributed by this pack's implementations.
    pub fn owned_loc(&self) -> usize {
        self.implementations.iter().map(|i| i.loc).sum()
    }

    /// Content identity over ONLY the Seed-verifiable core block. The additive capability schema is
    /// EXCLUDED, so declaring capabilities never changes a pack's identity and never disturbs a previously
    /// sealed receipt over this manifest. Reuses the one canonical-JSON + sha256 engine (`Record::new`).
    pub fn content_identity(&self) -> String {
        let core = serde_json::json!({
            "pack": self.pack,
            "version": self.version,
            "compatibility": self.compatibility,
            "source_commit": self.source_commit,
            "contents": self.contents,
            "offline_cache": self.offline_cache,
        });
        Record::new("pack.manifest", core).identity
    }

    /// Structural well-formedness of the capability schema: every declared capability names a real
    /// implementation, and every implementation kind matches its capability declaration.
    pub fn validate_capabilities(&self) -> Result<()> {
        if self.pack.is_empty() || self.version.is_empty() {
            return Err(Error::Pack("pack/version required".into()));
        }
        for cap in &self.capabilities {
            let imp = self
                .implementations
                .iter()
                .find(|i| i.id == cap.implementation)
                .ok_or_else(|| {
                    Error::Pack(format!(
                        "capability {} names missing impl {}",
                        cap.capability, cap.implementation
                    ))
                })?;
            if imp.kind != cap.kind {
                return Err(Error::Pack(format!(
                    "capability {} kind {:?} != impl {} kind {:?}",
                    cap.capability, cap.kind, imp.id, imp.kind
                )));
            }
        }
        Ok(())
    }

    // ---- Capability-pack builders (ergonomic construction for absorbed provider packs). ----

    /// A minimal well-formed capability pack, compatible with the Seed ABI by default.
    pub fn capability_pack(pack: &str, version: &str, profile: Profile) -> Self {
        PackManifest {
            pack: pack.into(),
            version: version.into(),
            compatibility: SEED_COMPAT.into(),
            profile: Some(profile),
            ..Default::default()
        }
    }

    pub fn with_source_commit(mut self, c: &str) -> Self {
        self.source_commit = c.into();
        self
    }
    pub fn with_offline_cache(mut self, c: &str) -> Self {
        self.offline_cache = c.into();
        self
    }
    pub fn with_rollback(mut self, r: &str) -> Self {
        self.rollback = r.into();
        self
    }
    pub fn add_content(mut self, path: &str, sha256: &str) -> Self {
        self.contents.push(PackEntry { path: path.into(), sha256: sha256.into() });
        self
    }
    pub fn add_capability(mut self, capability: &str, kind: CapabilityKind, implementation: &str) -> Self {
        self.capabilities.push(CapabilityDecl {
            capability: capability.into(),
            kind,
            implementation: implementation.into(),
        });
        self
    }
    pub fn add_implementation(mut self, imp: Implementation) -> Self {
        self.implementations.push(imp);
        self
    }
    pub fn add_asset(mut self, role: &str, format: &str, identity: &str) -> Self {
        self.assets.push(Asset { role: role.into(), format: format.into(), identity: identity.into() });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch() -> PathBuf {
        let d = std::env::temp_dir().join(format!("seedb-pack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn write(dir: &Path, name: &str, content: &[u8]) -> String {
        let p = dir.join(name);
        std::fs::File::create(&p).unwrap().write_all(content).unwrap();
        sha256_hex(content)
    }

    #[test]
    fn verifies_offline_and_rejects_tamper_and_rollback() {
        let d = scratch();
        let sha = write(&d, "impl.txt", b"runtime pack v1\n");
        let man = PackManifest {
            pack: "hawking-seed-b-runtime".into(),
            version: "1.0.0".into(),
            compatibility: "seed-b-1".into(),
            source_commit: "seed-b".into(),
            offline_cache: d.to_string_lossy().into(),
            contents: vec![PackEntry { path: "impl.txt".into(), sha256: sha.clone() }],
            ..Default::default()
        };
        assert_eq!(man.verify(d.join("manifest.json")).unwrap(), 1);
        assert!(man.compatible_with("seed-b-1"));
        write(&d, "impl.txt", b"tampered\n");
        assert!(man.verify(d.join("manifest.json")).is_err());
        write(&d, "impl.txt", b"runtime pack v1\n"); // rollback
        assert!(man.verify(d.join("manifest.json")).is_ok());
    }

    fn sample_capability_pack() -> PackManifest {
        PackManifest::capability_pack("packs-nucleus-forge", "1.0.0", Profile::Default)
            .with_source_commit("deadbeef")
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
    fn capability_schema_validates_and_stays_seed_compatible() {
        let m = sample_capability_pack();
        m.validate_capabilities().unwrap();
        assert_eq!(m.compatibility, SEED_COMPAT);
        assert!(m.compatible_with(SEED_COMPAT));
        assert_eq!(m.owned_loc(), 120);
        assert_eq!(m.profile(), Profile::Default);
    }

    #[test]
    fn content_identity_is_canonical_and_excludes_capability_schema() {
        let base = sample_capability_pack();
        let id = base.content_identity();
        // identity is stable across a serde round-trip
        let m2: PackManifest =
            serde_json::from_str(&serde_json::to_string(&base).unwrap()).unwrap();
        assert_eq!(id, m2.content_identity());
        // adding capability metadata does NOT change the content identity (additive, excluded from the block)
        let mut enriched = base.clone();
        enriched = enriched.add_asset("weights", "safetensors", "openai/gpt-oss-120b@b5c939de");
        enriched.dependencies.push("packs-nucleus-source".into());
        assert_eq!(id, enriched.content_identity(), "capability schema is excluded from content identity");
    }

    #[test]
    fn additive_schema_preserves_legacy_serialization() {
        // A legacy manifest with none of the additive fields serializes byte-for-byte as before: the new
        // fields are skipped when empty, so any previously sealed receipt over such a manifest is preserved.
        let legacy = PackManifest {
            pack: "hawking-seed-c-runtime".into(),
            version: "1.0.0".into(),
            compatibility: SEED_COMPAT.into(),
            source_commit: "seed-c".into(),
            offline_cache: "/tmp/x".into(),
            contents: vec![PackEntry { path: "runtime.txt".into(), sha256: "abc".into() }],
            ..Default::default()
        };
        let json = serde_json::to_string(&legacy).unwrap();
        assert_eq!(
            json,
            r#"{"pack":"hawking-seed-c-runtime","version":"1.0.0","compatibility":"seed-c-1","source_commit":"seed-c","contents":[{"path":"runtime.txt","sha256":"abc"}],"offline_cache":"/tmp/x"}"#
        );
    }

    #[test]
    fn rejects_capability_without_impl() {
        let mut m = PackManifest::capability_pack("x", "1.0.0", Profile::Default);
        m.capabilities.push(CapabilityDecl {
            capability: "nope".into(),
            kind: CapabilityKind::ModelAdapter,
            implementation: "ghost".into(),
        });
        assert!(m.validate_capabilities().is_err());
    }
}
