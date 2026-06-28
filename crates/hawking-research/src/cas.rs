//! Content-addressing for the Research Lab (bible ch.08 §4.2.1, §4.3, Tenet 7).
//!
//! Two responsibilities:
//!
//! * **Stable, content-derived ids.** Re-ingesting identical bytes must yield
//!   the same node ids so ingestion is idempotent. We hash *normalized* content
//!   with blake3 (the bible's chosen hash) and mint ids like
//!   `chunk:<hex>` / `claim:<hex>` / `doc:<hex>`.
//! * **Immutable evidence receipts.** The raw bytes behind a citation are pinned
//!   in a [`hide_core::persistence::BlobStore`] so a synthesized sentence can be
//!   re-verified against the exact bytes it was derived from (§4.7.3, §6).
//!
//! Note on stores: `hide_core::FileBlobStore` content-addresses with sha256, but
//! the *graph node id* uses blake3 of the normalized form, which is what makes
//! re-ingest idempotent regardless of the blob backend.

use hide_core::error::Result;
use hide_core::persistence::DynBlobStore;
use hide_core::types::BlobRef;

/// Lowercase-hex blake3 digest of `bytes`.
pub fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Normalize free text before hashing so trivially-different encodings of the
/// same content collapse to one id: trim, collapse internal whitespace runs to a
/// single space, drop a trailing newline. Deterministic and cheap.
pub fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_ws = false;
    for ch in text.trim().chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    out
}

/// Content-addressed id for normalized text under a node-kind prefix, e.g.
/// `content_id("chunk", text)` → `chunk:9f86d0...`.
pub fn content_id(prefix: &str, text: &str) -> String {
    let norm = normalize_text(text);
    format!("{prefix}:{}", blake3_hex(norm.as_bytes()))
}

/// Content-addressed id derived from several fields (order-significant). Used for
/// claims (`claim:hash(text|paper_id)`) and docs (`doc:hash(title|body)`).
pub fn composite_id(prefix: &str, fields: &[&str]) -> String {
    let joined = fields
        .iter()
        .map(|f| normalize_text(f))
        .collect::<Vec<_>>()
        .join("\u{1f}"); // unit separator — unlikely to occur in source text
    format!("{prefix}:{}", blake3_hex(joined.as_bytes()))
}

/// The canonical evidence bytes for a piece of free text: the *normalized* form,
/// UTF-8 encoded. This is the single byte source that both content-addressing
/// (claim/citation ids) and evidence pinning/re-verification must agree on, so
/// that an id, its pinned blob, and its re-check hash never diverge (§4.7.3).
pub fn canonical_evidence_bytes(text: &str) -> Vec<u8> {
    normalize_text(text).into_bytes()
}

/// Pin raw evidence bytes in the CAS and return both the [`BlobRef`] and the
/// blake3 content hash used as the immutable receipt. The blake3 hash is the
/// one we record on provenance (so re-verification is backend-independent); the
/// `BlobRef` is how the bytes are fetched back.
///
/// Invariant: the recorded hash is `blake3_hex` of *exactly* the bytes pinned,
/// so [`verify_evidence`] (which re-hashes the fetched blob bytes) is sound by
/// construction.
pub fn pin_evidence(
    cas: &DynBlobStore,
    bytes: Vec<u8>,
    media_type: Option<String>,
) -> Result<(BlobRef, String)> {
    let hash = blake3_hex(&bytes);
    let blob = cas.put(bytes, media_type)?;
    Ok((blob, hash))
}

/// Pin a section's *canonical* evidence bytes (normalized text) and return the
/// receipt. The pinned bytes match what [`content_id`]/[`composite_id`] hash for
/// the same text, so the claim id and its evidence receipt agree on one source.
pub fn pin_canonical_evidence(cas: &DynBlobStore, text: &str) -> Result<(BlobRef, String)> {
    pin_evidence(cas, canonical_evidence_bytes(text), Some("text/plain".to_string()))
}

/// Re-open evidence bytes from the CAS and confirm they still blake3-hash to the
/// recorded receipt. `None` blob → bytes were never pinned (cannot verify).
pub fn verify_evidence(
    cas: &DynBlobStore,
    blob: &BlobRef,
    expected_hash: &str,
) -> Result<EvidenceCheck> {
    let Some(bytes) = cas.get(blob)? else {
        return Ok(EvidenceCheck::Missing);
    };
    let actual = blake3_hex(&bytes);
    if actual == expected_hash {
        Ok(EvidenceCheck::Intact { bytes })
    } else {
        Ok(EvidenceCheck::Tampered {
            expected: expected_hash.to_string(),
            actual,
        })
    }
}

/// Outcome of re-checking a citation's evidence against the CAS (§4.7.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceCheck {
    /// Bytes present and hash-matched. Carries the bytes for phrase re-checks.
    Intact { bytes: Vec<u8> },
    /// Bytes present but the hash changed — the evidence was mutated.
    Tampered { expected: String, actual: String },
    /// No bytes pinned for this blob — the claim cannot be re-verified.
    Missing,
}

impl EvidenceCheck {
    pub fn is_intact(&self) -> bool {
        matches!(self, EvidenceCheck::Intact { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::persistence::{DynBlobStore, InMemoryBlobStore};
    use std::sync::Arc;

    #[test]
    fn normalization_collapses_whitespace() {
        assert_eq!(normalize_text("  a\n\t b  c \n"), "a b c");
    }

    #[test]
    fn content_ids_are_idempotent_and_normalization_stable() {
        let a = content_id("chunk", "paged   attention\n improves reuse");
        let b = content_id("chunk", "paged attention improves reuse");
        assert_eq!(a, b);
        assert!(a.starts_with("chunk:"));
    }

    #[test]
    fn composite_id_is_order_sensitive() {
        let a = composite_id("claim", &["text", "paper1"]);
        let b = composite_id("claim", &["paper1", "text"]);
        assert_ne!(a, b);
    }

    #[test]
    fn pin_and_verify_roundtrips_and_detects_tamper() {
        let cas: DynBlobStore = Arc::new(InMemoryBlobStore::default());
        let (blob, hash) = pin_evidence(&cas, b"73% accuracy".to_vec(), None).unwrap();
        assert!(verify_evidence(&cas, &blob, &hash).unwrap().is_intact());
        // A wrong expected hash is reported as tampering, not a silent pass.
        match verify_evidence(&cas, &blob, "deadbeef").unwrap() {
            EvidenceCheck::Tampered { .. } => {}
            other => panic!("expected tamper, got {other:?}"),
        }
    }
}
