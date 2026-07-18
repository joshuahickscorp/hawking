//! Pack ABI + verification. A pack declares identity, version, compatibility, hashes, and contents.
//! Seed verifies content hashes offline, rejects tampering, and supports rollback. No arbitrary
//! plugin execution: content-addressed verified packs only. Reused from Candidate A.

use crate::record::sha256_hex;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackEntry {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
                return Err(Error::Pack(format!(
                    "pack {} TAMPERED: {} sha {} != declared {}",
                    self.pack, e.path, got, e.sha256
                )));
            }
            ok += 1;
        }
        Ok(ok)
    }

    pub fn compatible_with(&self, seed_compat: &str) -> bool {
        self.compatibility.is_empty() || self.compatibility == seed_compat
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
        };
        assert_eq!(man.verify(d.join("manifest.json")).unwrap(), 1);
        assert!(man.compatible_with("seed-b-1"));
        write(&d, "impl.txt", b"tampered\n");
        assert!(man.verify(d.join("manifest.json")).is_err());
        write(&d, "impl.txt", b"runtime pack v1\n"); // rollback
        assert!(man.verify(d.join("manifest.json")).is_ok());
    }
}
