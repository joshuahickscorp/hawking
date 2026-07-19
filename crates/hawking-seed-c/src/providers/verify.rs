//! The **one verifier**. It owns canonical manifest encoding, content identity, compatibility, dependency
//! closure, tamper refusal, offline hydration, and active-set reporting — for every pack. There is no
//! second verification system: content hashing and rollback are delegated to the Seed's
//! [`crate::pack::PackManifest::verify`], and the verification result is sealed as a Seed `compatibility`
//! receipt via [`crate::evidence`].

use crate::evidence::receipt;
use crate::pack::PackManifest;
use crate::record::Record;
use crate::{Error, Result};
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub struct VerifyReport {
    pub pack: String,
    pub manifest_identity: String,
    pub content_entries_ok: usize,
    pub compatible: bool,
    pub tamper_detected: bool,
    pub missing_dependencies: Vec<String>,
    pub reason: String,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        !self.tamper_detected && self.compatible && self.missing_dependencies.is_empty()
    }
}

/// Verify one pack against a set of available pack ids (its dependency closure) and the target Seed ABI.
/// Content hashes are checked offline through the Seed engine; the manifest itself is content addressed.
/// Never executes pack code — content-addressed verified packs only.
pub fn verify_pack(
    man: &PackManifest,
    manifest_path: impl AsRef<Path>,
    seed_compat: &str,
    available: &[String],
) -> Result<VerifyReport> {
    man.validate_capabilities()?;
    let compatible = man.compatible_with(seed_compat);
    let missing: Vec<String> = man
        .dependencies
        .iter()
        .filter(|d| !available.contains(*d))
        .cloned()
        .collect();

    // Tamper is a TYPED signal now (Error::Tamper), not a string scrape. Missing-file / IO errors still
    // propagate. Exact tamper/missing semantics preserved.
    let (content_entries_ok, tamper_detected, reason) = match man.verify(manifest_path.as_ref()) {
        Ok(n) => (n, false, "content verified offline".to_string()),
        Err(e @ Error::Tamper { .. }) => (0, true, e.to_string()),
        Err(e) => return Err(e),
    };

    Ok(VerifyReport {
        pack: man.pack.clone(),
        manifest_identity: man.content_identity(),
        content_entries_ok,
        compatible,
        tamper_detected,
        missing_dependencies: missing,
        reason: if !compatible {
            format!("incompatible: pack wants {}, host is {}", man.compatibility, seed_compat)
        } else {
            reason
        },
    })
}

/// Seal the verification outcome as a Seed `compatibility` receipt — the ONE evidence engine records pack
/// verification; the verifier does not own a receipt writer.
pub fn seal_receipt(report: &VerifyReport) -> Result<Record> {
    Ok(receipt("compatibility", serde_json::to_value(report)?))
}

/// Offline hydration: prove the pack can be reconstituted from its offline cache with no network. The
/// content check IS the hydration proof (every declared file present + hash-matching under the cache).
pub fn offline_hydrate(man: &PackManifest, manifest_path: impl AsRef<Path>) -> Result<usize> {
    man.verify(manifest_path.as_ref())
        .map_err(|e| Error::Pack(format!("hydration failed: {e}")))
}

/// The active-set report: which packs verified, which are compatible, and the total honest owned LOC.
#[derive(Debug, Clone, Serialize)]
pub struct ActiveSet {
    pub packs: Vec<VerifyReport>,
    pub compatible_packs: usize,
    pub owned_loc: usize,
}

pub fn active_set(reports: Vec<(VerifyReport, usize)>) -> ActiveSet {
    let compatible_packs = reports.iter().filter(|(r, _)| r.ok()).count();
    let owned_loc = reports.iter().filter(|(r, _)| r.ok()).map(|(_, loc)| loc).sum();
    ActiveSet {
        packs: reports.into_iter().map(|(r, _)| r).collect(),
        compatible_packs,
        owned_loc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{CapabilityKind, Implementation, PackManifest, Profile, SEED_COMPAT};
    use crate::record::sha256_hex;
    use std::io::Write;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("nucleus-verify-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn write(dir: &Path, name: &str, content: &[u8]) -> String {
        std::fs::File::create(dir.join(name)).unwrap().write_all(content).unwrap();
        sha256_hex(content)
    }

    fn manifest(dir: &Path, sha: String) -> PackManifest {
        PackManifest::capability_pack("packs-nucleus-metal", "1.0.0", Profile::Performance)
            .with_offline_cache(&dir.to_string_lossy())
            .add_content("impl.txt", &sha)
            .add_implementation(Implementation {
                id: "q8-gemv".into(),
                kind: CapabilityKind::MetalImpl,
                loc: 90,
                bytes: 3000,
                entry: "metal::TiedLogitsOp".into(),
                tests: vec!["metal-cpu-parity".into()],
            })
            .add_capability("metal.tied_logits", CapabilityKind::MetalImpl, "q8-gemv")
    }

    #[test]
    fn verify_hydrate_tamper_rollback_and_receipt() {
        let d = scratch("main");
        let sha = write(&d, "impl.txt", b"metal provider v1\n");
        let man = manifest(&d, sha.clone());
        let mpath = d.join("manifest.json");

        let r = verify_pack(&man, &mpath, SEED_COMPAT, &[]).unwrap();
        assert!(r.ok() && r.content_entries_ok == 1 && !r.tamper_detected);
        // offline hydration proves reconstitution
        assert_eq!(offline_hydrate(&man, &mpath).unwrap(), 1);
        // the verification is sealed as a Seed compatibility receipt (tamper-evident)
        let rec = seal_receipt(&r).unwrap();
        assert!(rec.verify().is_ok() && rec.kind == "compatibility");

        // tamper -> refusal
        write(&d, "impl.txt", b"tampered\n");
        let r2 = verify_pack(&man, &mpath, SEED_COMPAT, &[]).unwrap();
        assert!(r2.tamper_detected && !r2.ok());

        // rollback -> green again
        write(&d, "impl.txt", b"metal provider v1\n");
        assert!(verify_pack(&man, &mpath, SEED_COMPAT, &[]).unwrap().ok());
    }

    #[test]
    fn incompatible_abi_is_refused() {
        let d = scratch("abi");
        let sha = write(&d, "impl.txt", b"x\n");
        let mut man = manifest(&d, sha);
        man.compatibility = "seed-b-1".into();
        let r = verify_pack(&man, d.join("manifest.json"), SEED_COMPAT, &[]).unwrap();
        assert!(!r.compatible && !r.ok());
    }

    #[test]
    fn missing_dependency_is_reported() {
        let d = scratch("deps");
        let sha = write(&d, "impl.txt", b"x\n");
        let mut man = manifest(&d, sha);
        man.dependencies.push("packs-nucleus-source".into());
        let r = verify_pack(&man, d.join("manifest.json"), SEED_COMPAT, &[]).unwrap();
        assert_eq!(r.missing_dependencies, vec!["packs-nucleus-source".to_string()]);
        assert!(!r.ok());
        // once the dependency is available, it verifies
        let r2 = verify_pack(&man, d.join("manifest.json"), SEED_COMPAT, &["packs-nucleus-source".into()]).unwrap();
        assert!(r2.ok());
    }
}
