//! Evidence path + grade-kind checks for the registry honesty test.

use std::path::{Path, PathBuf};

use crate::abi::{required_evidence_kind, Evidence, FamilyDescriptor};
use crate::support_level::SupportLevel;

/// Paths that are allowed as evidence even when the binary artifact itself is
/// off-tree / sealed (we still require the *receipt* file in-repo).
const RECEIPT_ALLOWLIST: &[&str] = &[
    "evidence/glm52/GLM52_FLAGSHIP_ADAPTER_PARITY.json",
    "evidence/kimi-k26/KIMI_K26_ADAPTER_TWIN.json",
    "packs/hawking-adapters-extra.json",
    "evidence/fabric/FABRIC_BRIDGE_ARCHAEOLOGY.md",
];

/// Validate that a family's declared level is backed by named evidence of the
/// kind the grade names, and that every evidence path exists on disk.
///
/// Rules:
/// - `DECLARED`: evidence optional (family description is the claim).
/// - Any higher level: at least one evidence entry of the *required kind*;
///   every `path` must resolve under `workspace_root`.
/// - `PRODUCTION`: always fails today (no family may claim it).
pub fn validate_family_evidence(
    workspace_root: &Path,
    desc: &FamilyDescriptor,
) -> Result<(), String> {
    if desc.level == SupportLevel::Production {
        return Err(format!(
            "family {}: PRODUCTION is forbidden until a standing parity receipt \
             is added and this gate is deliberately lifted; current level must not be PRODUCTION",
            desc.id
        ));
    }

    // ABI completeness (value or null+reason for every field).
    if let Err(errs) = desc.abi.validate_complete(desc.id) {
        return Err(errs.join("; "));
    }

    if desc.level == SupportLevel::Declared {
        // Paths that *are* listed must still exist.
        for ev in desc.evidence {
            check_evidence_path(workspace_root, desc.id, ev)?;
        }
        return Ok(());
    }

    if desc.evidence.is_empty() {
        return Err(format!(
            "family {}: level {} requires at least one named evidence path",
            desc.id,
            desc.level.as_str()
        ));
    }

    let required = required_evidence_kind(desc.level).ok_or_else(|| {
        format!(
            "family {}: internal error — no required kind for {}",
            desc.id,
            desc.level.as_str()
        )
    })?;

    let has_kind = desc.evidence.iter().any(|e| e.kind == required);
    if !has_kind {
        return Err(format!(
            "family {}: level {} requires evidence kind `{}` (the grade names the evidence); \
             found kinds: [{}]",
            desc.id,
            desc.level.as_str(),
            required.as_str(),
            desc.evidence
                .iter()
                .map(|e| e.kind.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    for ev in desc.evidence {
        check_evidence_path(workspace_root, desc.id, ev)?;
    }
    Ok(())
}

fn check_evidence_path(root: &Path, family: &str, ev: &Evidence) -> Result<(), String> {
    let p = PathBuf::from(ev.path);
    let full = if p.is_absolute() {
        p
    } else {
        root.join(ev.path)
    };
    if full.exists() {
        return Ok(());
    }
    if RECEIPT_ALLOWLIST.contains(&ev.path) {
        return Err(format!(
            "family {family}: allowlisted evidence {} is missing on disk at {}",
            ev.path,
            full.display()
        ));
    }
    Err(format!(
        "family {family}: evidence path {} does not exist (claim: {})",
        ev.path, ev.claim
    ))
}

/// Workspace root: walk up from CARGO_MANIFEST_DIR to the dir that contains
/// `Cargo.toml` workspace + `crates/`.
pub fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crates/hawking-adapters -> repo root
    if dir.ends_with("hawking-adapters") {
        dir.pop(); // crates
        dir.pop(); // root
    }
    dir
}
