//! On-disk prefix KV cache.
//!
//! Caches the post-prefill KV state of a prompt to disk so chat-style
//! workloads (constant system prompt + chat history + new user message)
//! can skip prefill for the shared prefix on subsequent turns.
//!
//! ## Wire format (v2)
//!
//! Each cache entry is a single file under
//! `<root>/<model_hex>/<prefix_hex>.kv`. `prefix_hex` is the rolling
//! sha256 of the token sequence (see [`PrefillKey::rolling_prefix_hash`])
//! so longest-prefix lookup is a constant-size probe per length without
//! reading file bodies.
//!
//! ```text
//! magic       u8[8]   "DSPRFKV2"
//! version     u32     2
//! model_hash  u8[32]
//! tok_hash    u8[32]
//! prefix_hash u8[32]    rolling sha256 over the prompt tokens
//! n_layers    u32
//! n_kv_heads  u32
//! head_dim    u32
//! seq_len     u32       == n_tokens
//! n_tokens    u32
//! tokens      u32[n_tokens]
//! body        f32 keys-then-values per layer, packed [seq_len, n_kv_heads, head_dim]
//! ```
//!
//! The body matches the in-memory layout used by [`KvCache`] (one f32
//! element per KV scalar, layer-major).
//!
//! ## Lookup
//!
//! Longest-prefix lookup is O(N) hash-update ops against an in-memory
//! `HashMap<[u8;32], EntryMeta>` populated at `open()` from the on-disk
//! file inventory. No directory scans on the hot path beyond a single
//! `metadata()` for LRU touch.
//!
//! ## Eviction
//!
//! Optional disk-size budget (`with_budget_bytes`). Eviction is LRU by
//! file mtime, applied after each `store()`. Default = unlimited.

use crate::cache::KvCache;
use crate::Result;
use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

const PREFILL_MAGIC: &[u8; 8] = b"DSPRFKV2";
const PREFILL_VERSION: u32 = 2;
/// Byte offset where the file body (KV bytes) begins, relative to the
/// `tokens[]` array end. The fixed-size header is everything up to and
/// including `n_tokens`.
const HEADER_FIXED_LEN: usize = 8 + 4 + 32 + 32 + 32 + 4 + 4 + 4 + 4 + 4;

/// Identifies a cached prefill. `prompt_tokens` is the full token list
/// the cache entry covers; `prefix_hash` is the rolling sha256 over
/// those tokens (see [`PrefillKey::rolling_prefix_hash`]).
#[derive(Debug, Clone)]
pub struct PrefillKey {
    pub model_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub prefix_hash: [u8; 32],
    pub prompt_tokens: Vec<u32>,
}

impl PrefillKey {
    /// Build a key for `prompt_tokens` under `model_id` +
    /// `tokenizer_signature`. The prefix hash is the rolling SHA-256
    /// described by [`Self::rolling_prefix_hash`].
    pub fn from_model_and_prompt(
        model_id: &str,
        tokenizer_signature: &[u8],
        prompt_tokens: &[u32],
    ) -> Self {
        let model_hash = sha256_of(&[model_id.as_bytes()]);
        let tokenizer_hash = sha256_of(&[tokenizer_signature]);
        let prefix_hash =
            Self::rolling_prefix_hash(&model_hash, &tokenizer_hash, prompt_tokens);
        Self {
            model_hash,
            tokenizer_hash,
            prefix_hash,
            prompt_tokens: prompt_tokens.to_vec(),
        }
    }

    /// Rolling SHA-256 over the token sequence, seeded by the model and
    /// tokenizer hashes. Defined so that the hash of the first `i`
    /// tokens is recoverable by feeding `tokens[0..i]` through the
    /// hasher in order — this is what makes longest-prefix lookup cheap.
    pub fn rolling_prefix_hash(
        model_hash: &[u8; 32],
        tokenizer_hash: &[u8; 32],
        tokens: &[u32],
    ) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(model_hash);
        h.update(tokenizer_hash);
        for &t in tokens {
            h.update(t.to_le_bytes());
        }
        h.finalize().into()
    }

    fn path(&self, root: &Path) -> PathBuf {
        let model_hex = hex32(&self.model_hash);
        let prefix_hex = hex32(&self.prefix_hash);
        root.join(model_hex).join(format!("{prefix_hex}.kv"))
    }
}

/// Per-entry metadata held in RAM for fast lookup. The full token list
/// is reloaded from disk on a hit; the in-RAM `n_tokens` lets us pick
/// the longest-prefix candidate without touching disk.
#[derive(Debug, Clone)]
struct EntryMeta {
    n_tokens: usize,
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
}

#[derive(Debug)]
pub struct PrefillDiskCache {
    root: PathBuf,
    /// budget on the total byte footprint of cache entries; `None` ⇒ unbounded.
    budget_bytes: Option<u64>,
    /// model-scoped index: `model_hash → prefix_hash → meta`.
    index: Mutex<HashMap<[u8; 32], HashMap<[u8; 32], EntryMeta>>>,
}

/// Result of a successful prefix lookup.
pub struct PrefillHit {
    pub n_tokens: usize,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// mmap of the full file. Bytes start at `body_offset`.
    mmap: Mmap,
    body_offset: usize,
    tokens_offset: usize,
}

impl PrefillHit {
    /// Tokens this entry was prefilled with. Returned as a freshly
    /// decoded `Vec<u32>` so callers don't have to juggle endianness.
    pub fn tokens(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.n_tokens);
        let mut off = self.tokens_offset;
        for _ in 0..self.n_tokens {
            let t = u32::from_le_bytes(self.mmap[off..off + 4].try_into().unwrap());
            out.push(t);
            off += 4;
        }
        out
    }

    /// Slice for `layer`'s keys (f32, native endian, length =
    /// `seq_len × n_kv_heads × head_dim`).
    pub fn keys_for(&self, layer: usize) -> &[f32] {
        let per_layer_elems = self.n_tokens * self.n_kv_heads * self.head_dim;
        let per_layer_bytes = per_layer_elems * 4;
        // body layout: [layer0_keys, layer0_values, layer1_keys, layer1_values, ...]
        let off = self.body_offset + layer * 2 * per_layer_bytes;
        f32_slice_from_bytes(&self.mmap[off..off + per_layer_bytes])
    }

    pub fn values_for(&self, layer: usize) -> &[f32] {
        let per_layer_elems = self.n_tokens * self.n_kv_heads * self.head_dim;
        let per_layer_bytes = per_layer_elems * 4;
        let off = self.body_offset + layer * 2 * per_layer_bytes + per_layer_bytes;
        f32_slice_from_bytes(&self.mmap[off..off + per_layer_bytes])
    }
}

impl PrefillDiskCache {
    pub fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
        std::fs::create_dir_all(root.as_ref())?;
        let mut index: HashMap<[u8; 32], HashMap<[u8; 32], EntryMeta>> = HashMap::new();
        // Walk one level deep: <root>/<model_hex>/*.kv
        if let Ok(top) = std::fs::read_dir(root.as_ref()) {
            for ent in top.flatten() {
                if !ent.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                let model_dir = ent.path();
                let model_hash = match parse_hex32(ent.file_name().to_str().unwrap_or("")) {
                    Some(h) => h,
                    None => continue,
                };
                let bucket = index.entry(model_hash).or_default();
                if let Ok(files) = std::fs::read_dir(&model_dir) {
                    for f in files.flatten() {
                        let name = f.file_name();
                        let s = name.to_str().unwrap_or("");
                        let stem = s.strip_suffix(".kv").unwrap_or(s);
                        let prefix_hash = match parse_hex32(stem) {
                            Some(h) => h,
                            None => continue,
                        };
                        if let Some(meta) = read_header_meta(&f.path()) {
                            bucket.insert(prefix_hash, meta);
                        }
                    }
                }
            }
        }
        Ok(Self {
            root: root.as_ref().to_owned(),
            budget_bytes: None,
            index: Mutex::new(index),
        })
    }

    /// Open from `DISMANTLE_PREFIX_CACHE_DIR` if set. Returns `Ok(None)`
    /// if the env var is unset (which signals "feature disabled").
    pub fn open_from_env() -> Result<Option<Self>> {
        let Ok(dir) = std::env::var("DISMANTLE_PREFIX_CACHE_DIR") else {
            return Ok(None);
        };
        if dir.is_empty() {
            return Ok(None);
        }
        let mut cache = Self::open(&dir)?;
        if let Ok(bytes_str) = std::env::var("DISMANTLE_PREFIX_CACHE_BUDGET_MB") {
            if let Ok(mb) = bytes_str.parse::<u64>() {
                cache.budget_bytes = Some(mb * 1024 * 1024);
            }
        }
        Ok(Some(cache))
    }

    /// Cap total on-disk size at `bytes`. Eviction runs LRU (oldest
    /// mtime first) at the end of every `store()` call.
    pub fn with_budget_bytes(mut self, bytes: u64) -> Self {
        self.budget_bytes = Some(bytes);
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Find the longest cached prefix of `prompt_tokens` for this
    /// `(model, tokenizer)` pair. Returns the hit (mmap'd) and the
    /// number of tokens it covers, or `Ok(None)` on miss.
    pub fn lookup_longest_prefix(
        &self,
        model_hash: &[u8; 32],
        tokenizer_hash: &[u8; 32],
        prompt_tokens: &[u32],
    ) -> Result<Option<PrefillHit>> {
        let guard = self.index.lock().unwrap();
        let bucket = match guard.get(model_hash) {
            Some(b) if !b.is_empty() => b,
            _ => return Ok(None),
        };
        // Rolling hash forward; record (n_tokens, hash) candidates.
        let mut h = Sha256::new();
        h.update(model_hash);
        h.update(tokenizer_hash);
        let max_len = prompt_tokens.len();
        // We want the longest match. Walk forward, remember the last
        // matching candidate.
        let mut best: Option<([u8; 32], usize, EntryMeta)> = None;
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            h.update(tok.to_le_bytes());
            let n_so_far = i + 1;
            if n_so_far == max_len {
                // Don't cache-load the *entire* prompt — leaves nothing
                // for the engine to run prefill on so the decode loop
                // has a real last_id to feed forward.
                break;
            }
            let candidate: [u8; 32] = h.clone().finalize().into();
            if let Some(meta) = bucket.get(&candidate) {
                debug_assert_eq!(meta.n_tokens, n_so_far);
                best = Some((candidate, n_so_far, meta.clone()));
            }
        }
        drop(guard);
        let (prefix_hash, n_tokens, meta) = match best {
            Some(b) => b,
            None => return Ok(None),
        };
        let path = self
            .root
            .join(hex32(model_hash))
            .join(format!("{}.kv", hex32(&prefix_hash)));
        let f = match File::open(&path) {
            Ok(f) => f,
            Err(_) => {
                self.forget(model_hash, &prefix_hash);
                return Ok(None);
            }
        };
        // LRU touch — bump mtime so eviction prefers older entries.
        let _ = filetime_touch(&path);
        let mmap = unsafe { MmapOptions::new().map(&f)? };
        if mmap.len() < HEADER_FIXED_LEN
            || &mmap[..8] != PREFILL_MAGIC
            || u32::from_le_bytes(mmap[8..12].try_into().unwrap()) != PREFILL_VERSION
            || &mmap[12..44] != model_hash
            || &mmap[44..76] != tokenizer_hash
            || &mmap[76..108] != prefix_hash
        {
            self.forget(model_hash, &prefix_hash);
            return Ok(None);
        }
        let tokens_offset = HEADER_FIXED_LEN;
        let body_offset = tokens_offset + n_tokens * 4;
        let expected_body = meta.n_layers as usize
            * 2
            * n_tokens
            * meta.n_kv_heads as usize
            * meta.head_dim as usize
            * 4;
        if mmap.len() < body_offset + expected_body {
            self.forget(model_hash, &prefix_hash);
            return Ok(None);
        }
        Ok(Some(PrefillHit {
            n_tokens,
            n_layers: meta.n_layers as usize,
            n_kv_heads: meta.n_kv_heads as usize,
            head_dim: meta.head_dim as usize,
            mmap,
            body_offset,
            tokens_offset,
        }))
    }

    /// Look up a cache entry by exact key (used by tests).
    pub fn lookup(&self, key: &PrefillKey) -> Result<Option<Mmap>> {
        let path = key.path(&self.root);
        if !path.exists() {
            return Ok(None);
        }
        let f = File::open(&path)?;
        let _ = filetime_touch(&path);
        let mmap = unsafe { MmapOptions::new().map(&f)? };
        if mmap.len() < HEADER_FIXED_LEN || &mmap[..8] != PREFILL_MAGIC {
            return Ok(None);
        }
        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version != PREFILL_VERSION {
            return Ok(None);
        }
        if &mmap[12..44] != key.model_hash
            || &mmap[44..76] != key.tokenizer_hash
            || &mmap[76..108] != key.prefix_hash
        {
            return Ok(None);
        }
        Ok(Some(mmap))
    }

    /// Persist `key`'s post-prefill KV state. The `kv` buffers are
    /// expected to be filled at least up to `kv.seq_len = key.prompt_tokens.len()`.
    pub fn store(&self, key: &PrefillKey, kv: &KvCache) -> Result<()> {
        if kv.seq_len < key.prompt_tokens.len() {
            return Err(crate::Error::Model(format!(
                "prefill_disk store: kv.seq_len={} < key.prompt_tokens.len()={}",
                kv.seq_len,
                key.prompt_tokens.len()
            )));
        }
        let n_tokens = key.prompt_tokens.len();
        let stride = kv.n_kv_heads * kv.head_dim;
        let want = n_tokens * stride;
        let mut keys_refs: Vec<&[f32]> = Vec::with_capacity(kv.n_layers);
        let mut vals_refs: Vec<&[f32]> = Vec::with_capacity(kv.n_layers);
        for layer in 0..kv.n_layers {
            keys_refs.push(&kv.keys[layer][..want]);
            vals_refs.push(&kv.values[layer][..want]);
        }
        self.store_raw(
            key,
            kv.n_layers,
            kv.n_kv_heads,
            kv.head_dim,
            &keys_refs,
            &vals_refs,
        )
    }

    /// Persist KV state assembled directly from raw f32 layer slices.
    /// Use this when the live KV mirror lives in GPU buffers and you
    /// don't want to materialize a `KvCache` just to call `store`.
    pub fn store_raw(
        &self,
        key: &PrefillKey,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        keys_per_layer: &[&[f32]],
        values_per_layer: &[&[f32]],
    ) -> Result<()> {
        let n_tokens = key.prompt_tokens.len();
        let want = n_tokens * n_kv_heads * head_dim;
        if keys_per_layer.len() != n_layers || values_per_layer.len() != n_layers {
            return Err(crate::Error::Model(format!(
                "store_raw: layer mismatch (got K={}, V={}, want {})",
                keys_per_layer.len(),
                values_per_layer.len(),
                n_layers
            )));
        }
        for (li, (k, v)) in keys_per_layer.iter().zip(values_per_layer.iter()).enumerate() {
            if k.len() < want || v.len() < want {
                return Err(crate::Error::Model(format!(
                    "store_raw: layer {} too short (k={}, v={}, want {})",
                    li,
                    k.len(),
                    v.len(),
                    want
                )));
            }
        }
        let path = key.path(&self.root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Write to a temp file and atomically rename — partial writes
        // would corrupt the index.
        let tmp_path = path.with_extension("kv.tmp");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp_path)?;
            f.write_all(PREFILL_MAGIC)?;
            f.write_all(&PREFILL_VERSION.to_le_bytes())?;
            f.write_all(&key.model_hash)?;
            f.write_all(&key.tokenizer_hash)?;
            f.write_all(&key.prefix_hash)?;
            f.write_all(&(n_layers as u32).to_le_bytes())?;
            f.write_all(&(n_kv_heads as u32).to_le_bytes())?;
            f.write_all(&(head_dim as u32).to_le_bytes())?;
            f.write_all(&(n_tokens as u32).to_le_bytes())?;
            f.write_all(&(n_tokens as u32).to_le_bytes())?;
            for t in &key.prompt_tokens {
                f.write_all(&t.to_le_bytes())?;
            }
            for li in 0..n_layers {
                f.write_all(f32_slice_as_bytes(&keys_per_layer[li][..want]))?;
                f.write_all(f32_slice_as_bytes(&values_per_layer[li][..want]))?;
            }
            f.sync_data().ok();
        }
        std::fs::rename(&tmp_path, &path)?;
        {
            let mut guard = self.index.lock().unwrap();
            let bucket = guard.entry(key.model_hash).or_default();
            bucket.insert(
                key.prefix_hash,
                EntryMeta {
                    n_tokens,
                    n_layers: n_layers as u32,
                    n_kv_heads: n_kv_heads as u32,
                    head_dim: head_dim as u32,
                },
            );
        }
        self.evict_if_over_budget();
        Ok(())
    }

    /// Drop a known-bad entry from both the index and disk.
    fn forget(&self, model_hash: &[u8; 32], prefix_hash: &[u8; 32]) {
        let path = self
            .root
            .join(hex32(model_hash))
            .join(format!("{}.kv", hex32(prefix_hash)));
        let _ = std::fs::remove_file(&path);
        let mut guard = self.index.lock().unwrap();
        if let Some(bucket) = guard.get_mut(model_hash) {
            bucket.remove(prefix_hash);
        }
    }

    fn evict_if_over_budget(&self) {
        let Some(budget) = self.budget_bytes else {
            return;
        };
        // Gather (path, mtime, len) for everything under root.
        let mut entries: Vec<(PathBuf, SystemTime, u64)> = Vec::new();
        if let Ok(top) = std::fs::read_dir(&self.root) {
            for d in top.flatten() {
                if !d.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                if let Ok(files) = std::fs::read_dir(d.path()) {
                    for f in files.flatten() {
                        if !f.file_name().to_string_lossy().ends_with(".kv") {
                            continue;
                        }
                        let meta = match f.metadata() {
                            Ok(m) => m,
                            Err(_) => continue,
                        };
                        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                        entries.push((f.path(), mtime, meta.len()));
                    }
                }
            }
        }
        let total: u64 = entries.iter().map(|(_, _, l)| *l).sum();
        if total <= budget {
            return;
        }
        // Evict LRU first (oldest mtime).
        entries.sort_by_key(|(_, mtime, _)| *mtime);
        let mut to_free = total - budget;
        for (path, _, len) in entries {
            if to_free == 0 {
                break;
            }
            let prefix_hash_opt = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(parse_hex32);
            let model_hash_opt = path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .and_then(parse_hex32);
            if let (Some(mh), Some(ph)) = (model_hash_opt, prefix_hash_opt) {
                self.forget(&mh, &ph);
            } else {
                let _ = std::fs::remove_file(&path);
            }
            to_free = to_free.saturating_sub(len);
        }
    }

    /// Diagnostic — number of indexed entries for this `(model_hash)`.
    pub fn len_for_model(&self, model_hash: &[u8; 32]) -> usize {
        self.index
            .lock()
            .unwrap()
            .get(model_hash)
            .map(|b| b.len())
            .unwrap_or(0)
    }
}

/// Restore a [`PrefillHit`] into a [`KvCache`], setting `kv.seq_len`
/// to `hit.n_tokens`. The cache's `n_layers/n_kv_heads/head_dim` must
/// match the hit's shape.
pub fn restore_hit_into_kv(hit: &PrefillHit, kv: &mut KvCache) -> Result<()> {
    if hit.n_layers != kv.n_layers
        || hit.n_kv_heads != kv.n_kv_heads
        || hit.head_dim != kv.head_dim
    {
        return Err(crate::Error::Model(format!(
            "restore_hit_into_kv: shape mismatch (hit={}x{}x{}, kv={}x{}x{})",
            hit.n_layers,
            hit.n_kv_heads,
            hit.head_dim,
            kv.n_layers,
            kv.n_kv_heads,
            kv.head_dim,
        )));
    }
    if hit.n_tokens > kv.max_seq {
        return Err(crate::Error::Model(format!(
            "restore_hit_into_kv: hit n_tokens {} > kv.max_seq {}",
            hit.n_tokens, kv.max_seq
        )));
    }
    let stride = kv.n_kv_heads * kv.head_dim;
    let want = hit.n_tokens * stride;
    for li in 0..kv.n_layers {
        kv.keys[li][..want].copy_from_slice(hit.keys_for(li));
        kv.values[li][..want].copy_from_slice(hit.values_for(li));
    }
    kv.seq_len = hit.n_tokens;
    Ok(())
}

// --- helpers ---

fn sha256_of(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

fn read_header_meta(path: &Path) -> Option<EntryMeta> {
    let mut f = File::open(path).ok()?;
    let mut hdr = [0u8; HEADER_FIXED_LEN];
    f.read_exact(&mut hdr).ok()?;
    if &hdr[..8] != PREFILL_MAGIC {
        return None;
    }
    if u32::from_le_bytes(hdr[8..12].try_into().ok()?) != PREFILL_VERSION {
        return None;
    }
    let n_layers = u32::from_le_bytes(hdr[108..112].try_into().ok()?);
    let n_kv_heads = u32::from_le_bytes(hdr[112..116].try_into().ok()?);
    let head_dim = u32::from_le_bytes(hdr[116..120].try_into().ok()?);
    // hdr[120..124] = seq_len (== n_tokens)
    let n_tokens = u32::from_le_bytes(hdr[124..128].try_into().ok()?) as usize;
    Some(EntryMeta {
        n_tokens,
        n_layers,
        n_kv_heads,
        head_dim,
    })
}

fn hex32(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// LRU touch — bump mtime to now.
fn filetime_touch(path: &Path) -> std::io::Result<()> {
    let now = SystemTime::now();
    let f = OpenOptions::new().write(true).open(path)?;
    f.set_modified(now)?;
    Ok(())
}

fn f32_slice_as_bytes(src: &[f32]) -> &[u8] {
    // Safety: f32 is plain old data.
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) }
}

fn f32_slice_from_bytes(src: &[u8]) -> &[f32] {
    // Safety: caller (PrefillHit::{keys,values}_for) only ever passes
    // regions whose start address is 4-aligned within the mmap. mmap
    // base is page-aligned and body_offset is a multiple of 4 by
    // construction (fixed header is 132 bytes + tokens area is u32-aligned).
    assert!(src.len() % 4 == 0);
    debug_assert_eq!(
        (src.as_ptr() as usize) % std::mem::align_of::<f32>(),
        0,
        "f32 slice misaligned"
    );
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn deterministic_kv(
        n_layers: usize,
        max_seq: usize,
        n_kv: usize,
        head_dim: usize,
        seq: usize,
    ) -> KvCache {
        let mut kv = KvCache::new(n_layers, max_seq, n_kv, head_dim);
        let stride = n_kv * head_dim;
        for li in 0..n_layers {
            for p in 0..seq {
                for d in 0..stride {
                    let val = ((li * 7919 + p * 31 + d) as i32 % 1024) as f32 * 0.001;
                    kv.keys[li][p * stride + d] = val;
                    kv.values[li][p * stride + d] = -val;
                }
            }
        }
        kv.seq_len = seq;
        kv
    }

    #[test]
    fn round_trip_basic() {
        let tmp = TempDir::new().unwrap();
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();
        let key = PrefillKey::from_model_and_prompt("test-model", b"tok-sig", &[1, 2, 3, 4]);
        let kv = deterministic_kv(2, 16, 4, 8, 4);
        cache.store(&key, &kv).unwrap();
        let hit = cache
            .lookup_longest_prefix(&key.model_hash, &key.tokenizer_hash, &[1, 2, 3, 4, 5])
            .unwrap()
            .expect("expected hit on prefix");
        assert_eq!(hit.n_tokens, 4);
        assert_eq!(hit.tokens(), vec![1, 2, 3, 4]);
        for li in 0..kv.n_layers {
            assert_eq!(hit.keys_for(li), kv.keys_for(li));
            assert_eq!(hit.values_for(li), kv.values_for(li));
        }
    }

    #[test]
    fn restore_round_trip() {
        let tmp = TempDir::new().unwrap();
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();
        let key = PrefillKey::from_model_and_prompt("m", b"t", &[1, 2, 3, 4]);
        let kv = deterministic_kv(3, 32, 2, 8, 4);
        cache.store(&key, &kv).unwrap();
        let hit = cache
            .lookup_longest_prefix(&key.model_hash, &key.tokenizer_hash, &[1, 2, 3, 4, 9])
            .unwrap()
            .unwrap();
        let mut restored = KvCache::new(3, 32, 2, 8);
        restore_hit_into_kv(&hit, &mut restored).unwrap();
        assert_eq!(restored.seq_len, 4);
        for li in 0..kv.n_layers {
            assert_eq!(restored.keys_for(li), kv.keys_for(li));
            assert_eq!(restored.values_for(li), kv.values_for(li));
        }
    }

    #[test]
    fn longest_prefix_picks_longest() {
        let tmp = TempDir::new().unwrap();
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();
        let kv_a = deterministic_kv(1, 16, 2, 4, 2);
        let kv_b = deterministic_kv(1, 16, 2, 4, 4);
        let key_a = PrefillKey::from_model_and_prompt("m", b"t", &[10, 11]);
        let key_b = PrefillKey::from_model_and_prompt("m", b"t", &[10, 11, 12, 13]);
        cache.store(&key_a, &kv_a).unwrap();
        cache.store(&key_b, &kv_b).unwrap();
        let hit = cache
            .lookup_longest_prefix(&key_a.model_hash, &key_a.tokenizer_hash, &[10, 11, 12, 13, 14])
            .unwrap()
            .unwrap();
        assert_eq!(hit.n_tokens, 4);
    }

    #[test]
    fn lookup_does_not_return_full_prompt() {
        // When the cached entry covers EXACTLY the lookup prompt, miss —
        // leave at least one token for the engine to run prefill on so
        // the decode loop has a real last_id.
        let tmp = TempDir::new().unwrap();
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();
        let kv = deterministic_kv(1, 8, 1, 4, 3);
        let key = PrefillKey::from_model_and_prompt("m", b"t", &[1, 2, 3]);
        cache.store(&key, &kv).unwrap();
        assert!(cache
            .lookup_longest_prefix(&key.model_hash, &key.tokenizer_hash, &[1, 2, 3])
            .unwrap()
            .is_none());
        let hit = cache
            .lookup_longest_prefix(&key.model_hash, &key.tokenizer_hash, &[1, 2, 3, 99])
            .unwrap()
            .unwrap();
        assert_eq!(hit.n_tokens, 3);
    }

    #[test]
    fn tokenizer_change_invalidates() {
        let tmp = TempDir::new().unwrap();
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();
        let kv = deterministic_kv(1, 8, 1, 4, 2);
        let key = PrefillKey::from_model_and_prompt("m", b"tok-v1", &[5, 6]);
        cache.store(&key, &kv).unwrap();
        let key_v2 = PrefillKey::from_model_and_prompt("m", b"tok-v2", &[5, 6, 7]);
        let hit = cache
            .lookup_longest_prefix(&key_v2.model_hash, &key_v2.tokenizer_hash, &[5, 6, 7])
            .unwrap();
        assert!(hit.is_none(), "tokenizer change should invalidate");
    }

    #[test]
    fn lru_eviction_respects_budget() {
        let tmp = TempDir::new().unwrap();
        let kv = deterministic_kv(1, 16, 2, 4, 4);
        let cache = PrefillDiskCache::open(tmp.path())
            .unwrap()
            .with_budget_bytes(900);
        for i in 0..5u32 {
            let key = PrefillKey::from_model_and_prompt("m", b"t", &[i, i + 1, i + 2, i + 3]);
            cache.store(&key, &kv).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let total: u64 = walkdir_size(tmp.path());
        assert!(
            total <= 900,
            "expected ≤900 bytes on disk after eviction, got {}",
            total
        );
        assert!(total > 0);
    }

    fn walkdir_size(p: &Path) -> u64 {
        let mut total = 0u64;
        if let Ok(top) = std::fs::read_dir(p) {
            for d in top.flatten() {
                if d.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    if let Ok(files) = std::fs::read_dir(d.path()) {
                        for f in files.flatten() {
                            if let Ok(m) = f.metadata() {
                                total += m.len();
                            }
                        }
                    }
                }
            }
        }
        total
    }

    #[test]
    fn index_repopulates_on_reopen() {
        let tmp = TempDir::new().unwrap();
        let kv = deterministic_kv(1, 8, 1, 4, 2);
        let key = PrefillKey::from_model_and_prompt("m", b"t", &[1, 2]);
        {
            let cache = PrefillDiskCache::open(tmp.path()).unwrap();
            cache.store(&key, &kv).unwrap();
        }
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();
        assert_eq!(cache.len_for_model(&key.model_hash), 1);
        let hit = cache
            .lookup_longest_prefix(&key.model_hash, &key.tokenizer_hash, &[1, 2, 3])
            .unwrap();
        assert!(hit.is_some());
    }
}
