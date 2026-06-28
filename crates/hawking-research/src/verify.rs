//! Adversarial verification (bible ch.08 §4.7).
//!
//! Three real checks replace the prior 2-antonym toy:
//!
//! 1. **Independence + corroboration.** A claim is only `Supported` once it is
//!    backed by at least `MIN_CORROBORATION` *independent* sources — counted by
//!    distinct origin (doc id / provenance source), not by raw peer count, so a
//!    paper that repeats itself across sections cannot self-corroborate.
//! 2. **Refutation detection.** Negation/antonym signals between overlapping
//!    claims surface a `Contradicted`/`Refuted` status (first-class, §Tenet 3),
//!    using a small but extensible polarity lexicon plus explicit negation.
//! 3. **Citation re-verification** (§4.7.3, the #1 anti-hallucination guard):
//!    every claim's evidence is re-opened from the CAS and re-hashed against the
//!    recorded receipt; a claim whose bytes are missing or tampered is flagged.

use crate::cas::{self, EvidenceCheck};
use crate::kg::{Claim, ProvenanceSpan};
use hide_core::error::Result;
use hide_core::persistence::DynBlobStore;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Default minimum independent sources required to call a claim corroborated.
pub const MIN_CORROBORATION: usize = 2;
/// Lexical overlap above which two claims are treated as "about the same thing".
pub const OVERLAP_THRESHOLD: f32 = 0.4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimVerification {
    pub claim_id: String,
    pub status: ClaimStatus,
    pub supporting_sources: usize,
    pub refuting_sources: usize,
    /// Distinct origin count (independence), ≤ supporting_sources.
    pub independent_sources: usize,
    /// Outcome of re-checking the cited evidence against the CAS.
    pub citation_check: CitationCheck,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    /// ≥ MIN_CORROBORATION independent supporting sources, no refutation.
    Supported,
    /// Refuting evidence present, no support.
    Refuted,
    /// Both support and refutation present (tension).
    Contradicted,
    /// Some support but below the independence/corroboration bar.
    SingleSource,
    /// No support, no refutation found.
    Unverified,
}

/// Whether the claim's cited evidence still hashes to its recorded receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CitationCheck {
    /// Evidence bytes present and hash-matched.
    Intact,
    /// Evidence bytes missing from the CAS — cannot verify.
    Missing,
    /// Evidence hash changed — the source was mutated after extraction.
    Tampered,
    /// No CAS available / no receipt recorded — not checked.
    NotChecked,
}

pub struct AdversarialVerifier;

impl AdversarialVerifier {
    /// Verify a claim against its peers using independence + corroboration. Does
    /// NOT touch the CAS (pure, fast). Use [`Self::verify_with_cas`] to also
    /// re-verify the cited evidence bytes.
    pub fn verify(claim: &Claim, peer_claims: &[Claim]) -> ClaimVerification {
        let a = claim.text.to_lowercase();
        let mut support_origins: HashSet<String> = HashSet::new();
        let mut supporting = 0usize;
        let mut refuting = 0usize;
        let mut notes = Vec::new();

        for peer in peer_claims {
            if peer.id == claim.id {
                continue;
            }
            let b = peer.text.to_lowercase();
            let overlap = lexical_overlap(&a, &b);
            if overlap < OVERLAP_THRESHOLD {
                continue; // unrelated — neither supports nor refutes
            }
            if polarity_conflict(&a, &b) {
                refuting += 1;
                notes.push(format!("refuted by {}", peer.id));
            } else {
                supporting += 1;
                support_origins.insert(origin_key(&peer.provenance));
            }
        }

        // Independence: the claim's own origin does not count toward its support.
        support_origins.remove(&origin_key(&claim.provenance));
        let independent = support_origins.len();

        let status = if refuting > 0 && supporting > 0 {
            ClaimStatus::Contradicted
        } else if refuting > 0 {
            ClaimStatus::Refuted
        } else if independent >= MIN_CORROBORATION {
            ClaimStatus::Supported
        } else if supporting > 0 {
            ClaimStatus::SingleSource
        } else {
            ClaimStatus::Unverified
        };

        ClaimVerification {
            claim_id: claim.id.clone(),
            status,
            supporting_sources: supporting,
            refuting_sources: refuting,
            independent_sources: independent,
            citation_check: CitationCheck::NotChecked,
            notes,
        }
    }

    /// As [`Self::verify`], plus re-open and re-hash the claim's cited evidence
    /// against the CAS (§4.7.3). A failed citation check is recorded and noted.
    pub fn verify_with_cas(
        claim: &Claim,
        peer_claims: &[Claim],
        cas: &DynBlobStore,
    ) -> Result<ClaimVerification> {
        let mut v = Self::verify(claim, peer_claims);
        v.citation_check = check_citation(&claim.provenance, cas)?;
        match v.citation_check {
            CitationCheck::Tampered => v
                .notes
                .push("citation evidence tampered (hash mismatch)".to_string()),
            CitationCheck::Missing => v
                .notes
                .push("citation evidence missing from CAS".to_string()),
            _ => {}
        }
        Ok(v)
    }
}

/// Re-verify one provenance span's evidence against the CAS.
fn check_citation(span: &ProvenanceSpan, cas: &DynBlobStore) -> Result<CitationCheck> {
    let (Some(blob), Some(hash)) = (&span.evidence_blob, &span.content_hash) else {
        return Ok(CitationCheck::NotChecked);
    };
    Ok(match cas::verify_evidence(cas, blob, hash)? {
        EvidenceCheck::Intact { .. } => CitationCheck::Intact,
        EvidenceCheck::Missing => CitationCheck::Missing,
        EvidenceCheck::Tampered { .. } => CitationCheck::Tampered,
    })
}

/// Independence key: the distinct *origin* of a claim — its doc id, falling back
/// to the provenance source string.
fn origin_key(span: &ProvenanceSpan) -> String {
    if !span.doc_id.is_empty() {
        span.doc_id.clone()
    } else {
        span.provenance.source.clone()
    }
}

/// A small, extensible polarity lexicon. A conflict is signalled when one claim
/// asserts a direction and the peer asserts the opposite, OR when one negates a
/// key term the other asserts.
fn polarity_conflict(a: &str, b: &str) -> bool {
    const ANTONYMS: &[(&str, &str)] = &[
        ("increase", "decrease"),
        ("faster", "slower"),
        ("higher", "lower"),
        ("improves", "degrades"),
        ("reduces", "increases"),
        ("outperforms", "underperforms"),
        ("better", "worse"),
        ("gains", "loses"),
    ];
    for (x, y) in ANTONYMS {
        if (a.contains(x) && b.contains(y)) || (a.contains(y) && b.contains(x)) {
            return true;
        }
    }
    // Explicit negation: one side asserts a salient token, the other negates it.
    const NEG: &[&str] = &["not ", "no ", "without ", "fails to ", "does not "];
    let a_neg = NEG.iter().any(|n| a.contains(n));
    let b_neg = NEG.iter().any(|n| b.contains(n));
    if a_neg ^ b_neg {
        // Opposite negation polarity on overlapping topics is a soft conflict.
        return lexical_overlap(a, b) > 0.6;
    }
    false
}

fn lexical_overlap(a: &str, b: &str) -> f32 {
    let words: Vec<_> = a
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 3)
        .collect();
    if words.is_empty() {
        return 0.0;
    }
    let hits = words.iter().filter(|w| b.contains(**w)).count();
    hits as f32 / words.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::ConfidenceTier;
    use hide_core::persistence::InMemoryBlobStore;
    use hide_core::types::Provenance;
    use std::sync::Arc;

    fn claim(id: &str, doc: &str, text: &str) -> Claim {
        Claim {
            id: id.to_string(),
            text: text.to_string(),
            provenance: ProvenanceSpan {
                doc_id: doc.to_string(),
                span_id: None,
                char_range: None,
                citation: None,
                content_hash: None,
                evidence_blob: None,
                provenance: Provenance::trusted("test"),
            },
            confidence: ConfidenceTier::Extracted,
        }
    }

    #[test]
    fn corroboration_requires_independent_origins() {
        let target = claim("c0", "docA", "paged attention reduces kv cache memory waste");
        // Two peers from the SAME doc → only one independent origin.
        let peers = vec![
            target.clone(),
            claim("c1", "docA", "paged attention reduces kv cache memory waste greatly"),
        ];
        let v = AdversarialVerifier::verify(&target, &peers);
        assert_eq!(v.independent_sources, 0); // same origin as target excluded
        assert_eq!(v.status, ClaimStatus::SingleSource);

        // Add a genuinely independent doc.
        let mut peers2 = peers;
        peers2.push(claim(
            "c2",
            "docB",
            "paged attention reduces kv cache memory waste in serving",
        ));
        let v2 = AdversarialVerifier::verify(&target, &peers2);
        assert_eq!(v2.independent_sources, 1);
    }

    #[test]
    fn antonym_pair_triggers_refutation() {
        let target = claim("c0", "docA", "the method improves throughput substantially");
        let peers = vec![claim(
            "c1",
            "docB",
            "the method degrades throughput substantially",
        )];
        let v = AdversarialVerifier::verify(&target, &peers);
        assert_eq!(v.status, ClaimStatus::Refuted);
        assert_eq!(v.refuting_sources, 1);
    }

    #[test]
    fn citation_recheck_detects_tamper() {
        let cas: DynBlobStore = Arc::new(InMemoryBlobStore::default());
        let bytes = b"reports 73% accuracy".to_vec();
        let (blob, hash) = cas::pin_evidence(&cas, bytes, None).unwrap();

        let mut good = claim("c0", "docA", "reports 73% accuracy on the benchmark");
        good.provenance.evidence_blob = Some(blob.clone());
        good.provenance.content_hash = Some(hash);
        let v = AdversarialVerifier::verify_with_cas(&good, &[], &cas).unwrap();
        assert_eq!(v.citation_check, CitationCheck::Intact);

        let mut bad = good.clone();
        bad.provenance.content_hash = Some("deadbeef".to_string());
        let vb = AdversarialVerifier::verify_with_cas(&bad, &[], &cas).unwrap();
        assert_eq!(vb.citation_check, CitationCheck::Tampered);
    }
}
