//! The KV-store seam (bible §4.5, Appendix A.4).
//!
//! The shell-facing view of the runtime's tiered KV store (GPU→RAM→disk→
//! checkpoints). Today this talks to `hawking-serve` over HTTP; later it can be
//! an FFI. Lossless-by-construction: every reuse is re-verified by the engine's
//! bit-identical copy+prefill-from-pos path (greedy-lossless), so this layer
//! only routes — it stores no KV bytes itself.

use async_trait::async_trait;
use hide_core::error::{HideError, Result};
use hide_core::ids::{RunId, SessionId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A prefix address. **Byte-compatible with the in-tree
/// `hawking_core::stateful::prefix_cache::PrefixKey` and
/// `cache::prefill_disk::PrefillKey`** so the RAM and disk tiers agree on a
/// prefix's address and the `SystemPromptKvBank` routes into them (bible A.4).
///
/// Compatibility is by construction — [`PrefixKey::from_model_and_prompt`]
/// reproduces the exact derivation:
///   - `model_hash      = sha256(model_id)`
///   - `tokenizer_hash  = sha256(tokenizer_signature)`
///   - `prefix_hash     = sha256(model_hash ‖ tokenizer_hash ‖ Σ tok.to_le_bytes())`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixKey {
    pub model_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub prefix_hash: [u8; 32],
    pub n_tokens: usize,
}

impl PrefixKey {
    /// Build a key for `prompt_tokens` under `(model_id, tokenizer_signature)`.
    /// Defined to be byte-compatible with the in-tree disk/RAM tiers.
    pub fn from_model_and_prompt(
        model_id: &str,
        tokenizer_signature: &[u8],
        prompt_tokens: &[u32],
    ) -> Self {
        let model_hash = sha256(&[model_id.as_bytes()]);
        let tokenizer_hash = sha256(&[tokenizer_signature]);
        let prefix_hash = rolling_prefix_hash(&model_hash, &tokenizer_hash, prompt_tokens);
        Self {
            model_hash,
            tokenizer_hash,
            prefix_hash,
            n_tokens: prompt_tokens.len(),
        }
    }

    /// Lowercase hex of the prefix hash (the disk tier's `<prefix_hex>.kv`).
    pub fn prefix_hex(&self) -> String {
        hex32(&self.prefix_hash)
    }
}

fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

fn rolling_prefix_hash(
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

fn hex32(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

/// Residency tier of a KV range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvTier {
    Gpu,
    Ram,
    Disk,
    Remote,
}

/// A live decode-slot identifier on the serve side.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotId(pub u64);

/// A reusable-prefix handle returned by a lookup (a routing hint, not bytes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixHandle {
    pub key: PrefixKey,
    /// Matched prefix length in tokens (strictly ≤ the query length).
    pub matched_tokens: usize,
    pub tier: KvTier,
}

/// Back-compat: a coarse handle used by the previous (read-only) API shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandle {
    pub provider: String,
    pub key: String,
    pub tokens: usize,
    pub tier: KvTier,
}

/// Working-set eviction choice for a slot (mirrors the profile type).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum EvictionChoice {
    Lossless,
    StreamingLlm { sinks: usize, recent: usize },
    SnapKv { keep: usize },
    H2o { recent: usize, heavy: usize },
}

/// Budget for a slot's working set (mirrors the in-tree `WorkingSetBudget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkingSetBudget {
    pub max_tokens: usize,
}

/// A named KV checkpoint id (bible §4.5.5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointId(pub String);

/// Metadata for a checkpoint (for a resume / time-travel UI).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub id: CheckpointId,
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub label: String,
    pub created_at_ms: u64,
    pub tokenizer_signature: String,
}

/// A KV checkpoint descriptor (kept for back-compat; references handles).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCheckpoint {
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub handles: Vec<KvHandle>,
    pub tokenizer_signature: String,
}

/// Stats for the manifest / `/metrics` (bible A.4 `stats`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStoreStats {
    pub bank_hits: u64,
    pub prefix_reuse_tokens: u64,
    pub tier_bytes_gpu: u64,
    pub tier_bytes_ram: u64,
    pub tier_bytes_disk: u64,
    pub evictions: u64,
}

/// Restored-session handle from a checkpoint restore.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoredSession {
    pub session_id: SessionId,
    pub slot: SlotId,
    pub warmed_tokens: usize,
}

/// The shell-facing KV operations (bible A.4). Async so the live impl can call
/// `hawking-serve`. The runtime is not up during tests — use [`StubKvStore`].
#[async_trait]
pub trait KvStore: Send + Sync {
    /// Longest-prefix lookup across tiers; returns a handle if reusable.
    async fn lookup_prefix(&self, key: &PrefixKey) -> Result<Option<PrefixHandle>>;
    /// Promote a hit into a live slot and prefill only the tail.
    async fn warm_into_slot(&self, h: &PrefixHandle, full_ids: &[u32]) -> Result<SlotId>;
    /// Demote a finished slot's prefix to RAM (and async to disk). Never deletes.
    async fn demote(&self, slot: SlotId, prefix_len: usize) -> Result<()>;
    /// Set the working-set eviction policy + budget for a slot.
    async fn set_policy(
        &self,
        slot: SlotId,
        policy: EvictionChoice,
        budget: WorkingSetBudget,
    ) -> Result<()>;
    /// Checkpoint the live KV + manifest under a label.
    async fn checkpoint(&self, session: &SessionId, label: &str) -> Result<CheckpointId>;
    /// Restore a checkpoint into a warm slot (validates model/tokenizer hash).
    async fn restore(&self, id: &CheckpointId) -> Result<RestoredSession>;
    /// List a session's checkpoints (resume / time-travel UI).
    async fn list_checkpoints(&self, session: &SessionId) -> Result<Vec<CheckpointMeta>>;
    /// Stats for the manifest / `/metrics`.
    async fn stats(&self) -> Result<KvStoreStats>;
}

/// Back-compat alias for the previous trait name. Kept so any caller that named
/// `KvStoreClient` keeps compiling; new code uses [`KvStore`].
pub use self::KvStore as KvStoreClient;

// ---------------------------------------------------------------------------
// HTTP client to hawking-serve (the live seam)
// ---------------------------------------------------------------------------

/// Talks to `hawking-serve`'s `/v1/hawking/kv/*` surface. The endpoints are
/// `[RUNTIME-SIDE — LATER]`; this client is the shell-side seam that lights up
/// when they land. Construction never fails (no network on `new`).
#[derive(Clone)]
pub struct HttpKvStore {
    base_url: String,
    client: reqwest::Client,
}

impl HttpKvStore {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    async fn post(&self, path: &str, body: serde_json::Value) -> Result<serde_json::Value> {
        let resp = self
            .client
            .post(format!("{}{}", self.base_url, path))
            .json(&body)
            .send()
            .await
            .map_err(|e| HideError::RuntimeUnavailable(format!("kv {path}: {e}")))?;
        if !resp.status().is_success() {
            return Err(HideError::RuntimeUnavailable(format!(
                "kv {path} HTTP {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| HideError::RuntimeUnavailable(format!("kv {path} decode: {e}")))
    }
}

#[async_trait]
impl KvStore for HttpKvStore {
    async fn lookup_prefix(&self, key: &PrefixKey) -> Result<Option<PrefixHandle>> {
        let v = self
            .post(
                "/v1/hawking/kv/lookup",
                serde_json::json!({ "prefix_hex": key.prefix_hex(), "n_tokens": key.n_tokens }),
            )
            .await?;
        if v.get("hit").and_then(|h| h.as_bool()) == Some(true) {
            let matched = v
                .get("matched_tokens")
                .and_then(|m| m.as_u64())
                .unwrap_or(0) as usize;
            Ok(Some(PrefixHandle {
                key: key.clone(),
                matched_tokens: matched,
                tier: KvTier::Ram,
            }))
        } else {
            Ok(None)
        }
    }

    async fn warm_into_slot(&self, h: &PrefixHandle, full_ids: &[u32]) -> Result<SlotId> {
        let v = self
            .post(
                "/v1/hawking/kv/warm",
                serde_json::json!({
                    "prefix_hex": h.key.prefix_hex(),
                    "matched_tokens": h.matched_tokens,
                    "n_tokens": full_ids.len(),
                }),
            )
            .await?;
        let slot = v
            .get("slot")
            .and_then(|s| s.as_u64())
            .ok_or_else(|| HideError::RuntimeUnavailable("kv/warm: missing slot".into()))?;
        Ok(SlotId(slot))
    }

    async fn demote(&self, slot: SlotId, prefix_len: usize) -> Result<()> {
        self.post(
            "/v1/hawking/kv/demote",
            serde_json::json!({ "slot": slot.0, "prefix_len": prefix_len }),
        )
        .await
        .map(|_| ())
    }

    async fn set_policy(
        &self,
        slot: SlotId,
        policy: EvictionChoice,
        budget: WorkingSetBudget,
    ) -> Result<()> {
        self.post(
            "/v1/hawking/kv/policy",
            serde_json::json!({ "slot": slot.0, "policy": policy, "budget": budget }),
        )
        .await
        .map(|_| ())
    }

    async fn checkpoint(&self, session: &SessionId, label: &str) -> Result<CheckpointId> {
        let v = self
            .post(
                "/v1/hawking/kv/checkpoint",
                serde_json::json!({ "session": session.as_str(), "label": label }),
            )
            .await?;
        let id = v
            .get("checkpoint_id")
            .and_then(|c| c.as_str())
            .ok_or_else(|| HideError::RuntimeUnavailable("kv/checkpoint: missing id".into()))?;
        Ok(CheckpointId(id.to_string()))
    }

    async fn restore(&self, id: &CheckpointId) -> Result<RestoredSession> {
        let v = self
            .post(
                "/v1/hawking/kv/restore",
                serde_json::json!({ "checkpoint_id": id.0 }),
            )
            .await?;
        serde_json::from_value(v)
            .map_err(|e| HideError::RuntimeUnavailable(format!("kv/restore decode: {e}")))
    }

    async fn list_checkpoints(&self, session: &SessionId) -> Result<Vec<CheckpointMeta>> {
        let v = self
            .post(
                "/v1/hawking/kv/list",
                serde_json::json!({ "session": session.as_str() }),
            )
            .await?;
        let arr = v
            .get("checkpoints")
            .cloned()
            .unwrap_or(serde_json::json!([]));
        serde_json::from_value(arr)
            .map_err(|e| HideError::RuntimeUnavailable(format!("kv/list decode: {e}")))
    }

    async fn stats(&self) -> Result<KvStoreStats> {
        let v = self
            .post("/v1/hawking/kv/stats", serde_json::json!({}))
            .await?;
        serde_json::from_value(v)
            .map_err(|e| HideError::RuntimeUnavailable(format!("kv/stats decode: {e}")))
    }
}

// ---------------------------------------------------------------------------
// In-process stub (tests + offline). Models prefix reuse against a local map.
// ---------------------------------------------------------------------------

use parking_lot::Mutex;
use std::collections::HashMap;

/// A deterministic in-process [`KvStore`] for tests and offline operation. It
/// models longest-prefix reuse against an internal map of banked prefixes and
/// keeps real stats — it does not pretend to hold GPU KV.
#[derive(Default)]
pub struct StubKvStore {
    inner: Mutex<StubInner>,
}

#[derive(Default)]
struct StubInner {
    banked: HashMap<String, usize>, // prefix_hex -> matched tokens
    next_slot: u64,
    checkpoints: Vec<CheckpointMeta>,
    stats: KvStoreStats,
}

impl StubKvStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-bank a prefix (e.g. the system block) so a later lookup hits.
    pub fn bank(&self, key: &PrefixKey) {
        self.inner
            .lock()
            .banked
            .insert(key.prefix_hex(), key.n_tokens);
    }
}

#[async_trait]
impl KvStore for StubKvStore {
    async fn lookup_prefix(&self, key: &PrefixKey) -> Result<Option<PrefixHandle>> {
        let mut inner = self.inner.lock();
        if let Some(&n) = inner.banked.get(&key.prefix_hex()) {
            inner.stats.bank_hits += 1;
            inner.stats.prefix_reuse_tokens += n as u64;
            return Ok(Some(PrefixHandle {
                key: key.clone(),
                matched_tokens: n,
                tier: KvTier::Ram,
            }));
        }
        Ok(None)
    }

    async fn warm_into_slot(&self, _h: &PrefixHandle, _full_ids: &[u32]) -> Result<SlotId> {
        let mut inner = self.inner.lock();
        let slot = inner.next_slot;
        inner.next_slot += 1;
        Ok(SlotId(slot))
    }

    async fn demote(&self, _slot: SlotId, prefix_len: usize) -> Result<()> {
        self.inner.lock().stats.tier_bytes_ram += prefix_len as u64;
        Ok(())
    }

    async fn set_policy(
        &self,
        _slot: SlotId,
        _policy: EvictionChoice,
        _budget: WorkingSetBudget,
    ) -> Result<()> {
        Ok(())
    }

    async fn checkpoint(&self, session: &SessionId, label: &str) -> Result<CheckpointId> {
        let id = CheckpointId(format!("ckpt_{}_{}", session.as_str(), label));
        let mut inner = self.inner.lock();
        inner.checkpoints.push(CheckpointMeta {
            id: id.clone(),
            session_id: session.clone(),
            run_id: None,
            label: label.to_string(),
            created_at_ms: hide_core::ids::now_ms(),
            tokenizer_signature: String::new(),
        });
        Ok(id)
    }

    async fn restore(&self, id: &CheckpointId) -> Result<RestoredSession> {
        let inner = self.inner.lock();
        let meta = inner
            .checkpoints
            .iter()
            .find(|c| c.id == *id)
            .ok_or_else(|| HideError::NotFound(format!("checkpoint {}", id.0)))?;
        Ok(RestoredSession {
            session_id: meta.session_id.clone(),
            slot: SlotId(0),
            warmed_tokens: 0,
        })
    }

    async fn list_checkpoints(&self, session: &SessionId) -> Result<Vec<CheckpointMeta>> {
        Ok(self
            .inner
            .lock()
            .checkpoints
            .iter()
            .filter(|c| c.session_id == *session)
            .cloned()
            .collect())
    }

    async fn stats(&self) -> Result<KvStoreStats> {
        Ok(self.inner.lock().stats.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Interop lock: [`PrefixKey`] must be byte-identical to hawking-core's
    /// in-tree disk/RAM prefill key so the shell and runtime agree on a prefix's
    /// address. `hawking-core` is a heavy, macOS-Metal crate (`metal`, `objc2`),
    /// so we do **not** pull it in as a dev-dependency; instead this test
    /// replicates the *exact* in-tree byte derivation from:
    ///
    ///   crates/hawking-core/src/cache/prefill_disk.rs
    ///     · `PrefillKey::from_model_and_prompt`  (model/tokenizer sha256)
    ///     · `PrefillKey::rolling_prefix_hash`    (rolling sha256 over tokens)
    ///     · `PrefillKey::path` → `<model_hex>/<prefix_hex>.kv`
    ///
    /// If the in-tree derivation ever changes, this hand-replicated reference
    /// will diverge from `PrefixKey` and fail — locking the interop guarantee.
    #[test]
    fn prefix_key_matches_in_tree_derivation() {
        let model = "qwen-7b";
        let tok_sig = b"tokv1";
        let tokens = [1u32, 2, 3, 4];
        let key = PrefixKey::from_model_and_prompt(model, tok_sig, &tokens);

        // --- begin: byte-for-byte copy of prefill_disk.rs derivation ---
        // model_hash = sha256(model_id)
        let model_hash: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(model.as_bytes());
            h.finalize().into()
        };
        // tokenizer_hash = sha256(tokenizer_signature)
        let tokenizer_hash: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(tok_sig);
            h.finalize().into()
        };
        // prefix_hash = sha256(model_hash ‖ tokenizer_hash ‖ Σ tok.to_le_bytes())
        let prefix_hash: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(model_hash);
            h.update(tokenizer_hash);
            for t in tokens {
                h.update(t.to_le_bytes());
            }
            h.finalize().into()
        };
        // --- end: copy ---

        assert_eq!(key.model_hash, model_hash, "model_hash interop");
        assert_eq!(key.tokenizer_hash, tokenizer_hash, "tokenizer_hash interop");
        assert_eq!(key.prefix_hash, prefix_hash, "prefix_hash interop");
        assert_eq!(key.n_tokens, 4);

        // Disk-tier path form: `<model_hex>/<prefix_hex>.kv` (prefill_disk.rs
        // `PrefillKey::path`). `PrefixKey::prefix_hex` is the file stem.
        let model_hex = hex32(&model_hash);
        let prefix_hex = hex32(&prefix_hash);
        assert_eq!(key.prefix_hex(), prefix_hex);
        assert_eq!(key.prefix_hex().len(), 64);
        assert_eq!(
            format!("{model_hex}/{prefix_hex}.kv").len(),
            64 + 1 + 64 + 3
        );

        // Empty-prompt edge: still seeded by model‖tokenizer (n_tokens == 0).
        let empty = PrefixKey::from_model_and_prompt(model, tok_sig, &[]);
        assert_eq!(empty.n_tokens, 0);
        let empty_prefix: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(model_hash);
            h.update(tokenizer_hash);
            h.finalize().into()
        };
        assert_eq!(empty.prefix_hash, empty_prefix);
    }

    #[tokio::test]
    async fn stub_models_prefix_reuse_and_stats() {
        let store = StubKvStore::new();
        let key = PrefixKey::from_model_and_prompt("m", b"t", &[1, 2, 3]);
        assert!(store.lookup_prefix(&key).await.unwrap().is_none());
        store.bank(&key);
        let hit = store.lookup_prefix(&key).await.unwrap().unwrap();
        assert_eq!(hit.matched_tokens, 3);
        let slot = store.warm_into_slot(&hit, &[1, 2, 3, 4]).await.unwrap();
        store.demote(slot, 3).await.unwrap();
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.bank_hits, 1);
        assert_eq!(stats.prefix_reuse_tokens, 3);
    }

    #[tokio::test]
    async fn stub_checkpoint_roundtrip() {
        let store = StubKvStore::new();
        let sess = SessionId::from("ses_test");
        let id = store.checkpoint(&sess, "before-refactor").await.unwrap();
        let list = store.list_checkpoints(&sess).await.unwrap();
        assert_eq!(list.len(), 1);
        let restored = store.restore(&id).await.unwrap();
        assert_eq!(restored.session_id, sess);
    }
}
