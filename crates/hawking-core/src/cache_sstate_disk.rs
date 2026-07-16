//! Disk cache for serialized RWKV-7 recurrent states (`.sstate`) -- the M1
//! "instant resume / no re-prefill" store. A state captured after prefilling a
//! token sequence is keyed by `sha256(model_id || tokens)` and written
//! atomically (tmp + rename); reloading replays it with zero recompute. Mirrors
//! the `prefill_disk` KV-cache discipline, but for the constant-size recurrent
//! state (so the cached artifact is a flat ~6-16 MiB blob, not a growing cache).

use crate::model::rwkv7::RwkvState;
use sha2::{Digest, Sha256};
use std::io;
use std::path::PathBuf;

/// Content-addressed on-disk cache of recurrent states.
pub struct SstateDiskCache {
    dir: PathBuf,
}

impl SstateDiskCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Content-address key for a `(model_id, token-sequence)` prefix. Two runs
    /// over the same model + tokens hit the same entry; any divergence misses.
    pub fn key(model_id: &str, token_ids: &[u32]) -> String {
        let mut h = Sha256::new();
        h.update(model_id.as_bytes());
        h.update([0u8]);
        for &t in token_ids {
            h.update(t.to_le_bytes());
        }
        hex_str(h.finalize().as_slice())
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.sstate"))
    }

    /// Store a state atomically (tmp + rename) so a crash never leaves a torn
    /// file in the cache.
    pub fn store(&self, key: &str, state: &RwkvState) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.path_for(key);
        let tmp = path.with_extension("sstate.tmp");
        std::fs::write(&tmp, state.to_bytes())?;
        std::fs::rename(&tmp, &path)
    }

    /// Load a previously-stored state, or `None` if absent / corrupt.
    pub fn load(&self, key: &str) -> Option<RwkvState> {
        let bytes = std::fs::read(self.path_for(key)).ok()?;
        RwkvState::from_bytes(&bytes).ok()
    }

    /// Whether a key is present on disk.
    pub fn contains(&self, key: &str) -> bool {
        self.path_for(key).exists()
    }
}

fn hex_str(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RwkvState {
        RwkvState { wkv: vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]], att_shift: vec![vec![0.1, 0.2], vec![0.3, 0.4]], ffn_shift: vec![vec![-0.1, -0.2], vec![-0.3, -0.4]], fresh: false }
    }

    #[test]
    fn store_then_load_roundtrips_bit_identical() {
        let dir = std::env::temp_dir().join(format!("hawking_sstate_{}", std::process::id()));
        let cache = SstateDiskCache::new(&dir);
        let s = sample();
        let k = SstateDiskCache::key("rwkv7-0.4b", &[10, 20, 30]);
        assert!(!cache.contains(&k));
        cache.store(&k, &s).expect("store");
        assert!(cache.contains(&k));
        let back = cache.load(&k).expect("load");
        assert_eq!(back.to_bytes(), s.to_bytes(), "instant-resume must be exact");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn key_is_deterministic_and_sequence_sensitive() {
        let a = SstateDiskCache::key("m", &[1, 2, 3]);
        let b = SstateDiskCache::key("m", &[1, 2, 3]);
        let c = SstateDiskCache::key("m", &[1, 2, 4]);
        assert_eq!(a, b, "same inputs -> same key");
        assert_ne!(a, c, "different tokens -> different key");
        assert_eq!(a.len(), 64, "sha256 hex");
    }

    #[test]
    fn missing_key_loads_none() {
        let dir = std::env::temp_dir().join(format!("hawking_sstate_none_{}", std::process::id()));
        let cache = SstateDiskCache::new(&dir);
        assert!(cache.load("deadbeef").is_none());
    }
}
