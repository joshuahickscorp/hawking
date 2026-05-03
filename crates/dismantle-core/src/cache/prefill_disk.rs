//! mmap-backed cross-session prefill cache — wedge 5.
//!
//! Layout on disk:
//!
//!   <cache_dir>/<model_hash>/<prompt_hash>.kv
//!
//! File header: magic, version, model_hash, tokenizer_hash, n_layers,
//! n_kv_heads, head_dim, prompt_token_count, prompt_token_ids,
//! created_at_unix_ms. Body: KV tensors per layer, mmap-aligned.

use crate::cache::KvCache;
use crate::Result;
use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const PREFILL_MAGIC: &[u8; 8] = b"DSPRFKV1";
const PREFILL_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct PrefillKey {
    pub model_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub prompt_hash: [u8; 32],
    pub prompt_tokens: Vec<u32>,
}

impl PrefillKey {
    pub fn from_model_and_prompt(
        model_id: &str,
        tokenizer_signature: &[u8],
        prompt_tokens: &[u32],
    ) -> Self {
        let mut h = Sha256::new();
        h.update(model_id.as_bytes());
        let model_hash: [u8; 32] = h.finalize().into();

        let mut h = Sha256::new();
        h.update(tokenizer_signature);
        let tokenizer_hash: [u8; 32] = h.finalize().into();

        let mut h = Sha256::new();
        for &t in prompt_tokens {
            h.update(t.to_le_bytes());
        }
        let prompt_hash: [u8; 32] = h.finalize().into();

        Self {
            model_hash,
            tokenizer_hash,
            prompt_hash,
            prompt_tokens: prompt_tokens.to_vec(),
        }
    }

    fn path(&self, root: &Path) -> PathBuf {
        let model_hex = hex32(&self.model_hash);
        let prompt_hex = hex32(&self.prompt_hash);
        root.join(model_hex).join(format!("{prompt_hex}.kv"))
    }
}

#[derive(Debug)]
pub struct PrefillDiskCache {
    root: PathBuf,
}

impl PrefillDiskCache {
    pub fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
        std::fs::create_dir_all(root.as_ref())?;
        Ok(Self {
            root: root.as_ref().to_owned(),
        })
    }

    /// Look up a prefill cache file. Returns `None` on miss.
    pub fn lookup(&self, key: &PrefillKey) -> Result<Option<Mmap>> {
        let path = key.path(&self.root);
        if !path.exists() {
            return Ok(None);
        }
        let f = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&f)? };
        // Validate header.
        if mmap.len() < 80 || &mmap[..8] != PREFILL_MAGIC {
            return Ok(None);
        }
        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version != PREFILL_VERSION {
            return Ok(None);
        }
        if &mmap[12..44] != key.model_hash || &mmap[44..76] != key.tokenizer_hash {
            return Ok(None);
        }
        Ok(Some(mmap))
    }

    /// Persist a freshly computed KV cache for `key`.
    pub fn store(&self, key: &PrefillKey, kv: &KvCache) -> Result<()> {
        let path = key.path(&self.root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)?;
        f.write_all(PREFILL_MAGIC)?;
        f.write_all(&PREFILL_VERSION.to_le_bytes())?;
        f.write_all(&key.model_hash)?;
        f.write_all(&key.tokenizer_hash)?;
        f.write_all(&(kv.n_layers as u32).to_le_bytes())?;
        f.write_all(&(kv.n_kv_heads as u32).to_le_bytes())?;
        f.write_all(&(kv.head_dim as u32).to_le_bytes())?;
        f.write_all(&(kv.seq_len as u32).to_le_bytes())?;
        f.write_all(&(key.prompt_tokens.len() as u32).to_le_bytes())?;
        for t in &key.prompt_tokens {
            f.write_all(&t.to_le_bytes())?;
        }
        // Body: keys then values for each layer.
        for layer in 0..kv.n_layers {
            for &v in kv.keys_for(layer) {
                f.write_all(&v.to_le_bytes())?;
            }
            for &v in kv.values_for(layer) {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

fn hex32(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_empty_kv() {
        let tmp = TempDir::new().unwrap();
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();
        let key = PrefillKey::from_model_and_prompt("test", b"tok-sig", &[1, 2, 3]);
        let kv = KvCache::new(2, 16, 4, 64);
        cache.store(&key, &kv).unwrap();
        let mmap = cache.lookup(&key).unwrap();
        assert!(mmap.is_some());
    }
}
