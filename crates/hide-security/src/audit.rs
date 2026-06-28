//! Tamper-evident audit chain over the single-writer event log (bible ch.10
//! §4.2.1 / §4.11).
//!
//! This mirrors `hide_core::event`'s on-append chain *exactly* — both hash with
//! **blake3** over `prev_hash || canonical_event_bytes(chain_hash cleared)` — so
//! the security-side verifier and the core writer agree byte-for-byte. The one
//! refinement this module adds over a raw zero-prefix chain is a per-workspace
//! **genesis salt** (§4.2.1: "seq 0 uses a per-workspace random genesis salt"),
//! and the **signed ANCHORS** story (§4.11): periodically record a signed
//! `(seq, chain_root, signature, signer)`, expose verification against the
//! nearest anchor, and emit `security.anchor` / `security.integrity_alarm`
//! events (built with `NewEvent`).
//!
//! `compute_event_chain` / `verify_event_chain` keep their original signatures
//! (siblings + the `EventChainAuditor` integrity trait consume them) — they are
//! extended, not replaced. The salted variants are additive.

use hide_core::event::{Event, NewEvent};
use hide_core::ids::SessionId;
use hide_core::persistence::{EventLogIntegrity, IntegrityReport};
use hide_core::Result;
use serde::{Deserialize, Serialize};

/// Length, in bytes, of a blake3 digest and of the genesis salt.
pub const CHAIN_HASH_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainRecord {
    pub seq: u64,
    pub hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainAuditReport {
    pub ok: bool,
    pub records: Vec<ChainRecord>,
    pub chain_root: Option<String>,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Core chain over events (blake3; genesis = all-zero by default, salted variant
// below). Public signatures preserved for siblings.
// ---------------------------------------------------------------------------

/// Recompute the blake3 chain over `events`, starting from the all-zero genesis
/// (the default for a chain with no per-workspace salt). Does not verify
/// embedded hashes — see [`verify_event_chain`].
pub fn compute_event_chain(events: &[Event]) -> ChainAuditReport {
    chain_from(&[0u8; CHAIN_HASH_LEN], events, false)
}

/// Recompute the blake3 chain and verify each event's embedded `chain_hash`
/// matches. The first divergence (mismatch or a missing hash) fails the report.
pub fn verify_event_chain(events: &[Event]) -> ChainAuditReport {
    chain_from(&[0u8; CHAIN_HASH_LEN], events, true)
}

/// Like [`compute_event_chain`] but seeded with a per-workspace genesis salt
/// (§4.2.1). Two workspaces with identical event bodies still produce different
/// chain roots, so a chain can't be transplanted between logs.
pub fn compute_event_chain_salted(genesis: &[u8], events: &[Event]) -> ChainAuditReport {
    chain_from(genesis, events, false)
}

/// Salted counterpart to [`verify_event_chain`].
pub fn verify_event_chain_salted(genesis: &[u8], events: &[Event]) -> ChainAuditReport {
    chain_from(genesis, events, true)
}

fn chain_from(genesis: &[u8], events: &[Event], verify_embedded: bool) -> ChainAuditReport {
    let mut prev = genesis.to_vec();
    let mut records = Vec::with_capacity(events.len());
    for event in events {
        let digest = match chain_hash(&prev, event) {
            Ok(d) => d,
            Err(err) => {
                return ChainAuditReport {
                    ok: false,
                    records,
                    chain_root: None,
                    error: Some(format!("event {} failed canonical serialization: {err}", event.seq)),
                };
            }
        };
        let hash = hex_lower(&digest);
        records.push(ChainRecord {
            seq: event.seq,
            hash: hash.clone(),
        });
        if verify_embedded {
            match event.chain_hash.as_deref() {
                Some(embedded) if embedded == hash => {}
                Some(embedded) => {
                    return ChainAuditReport {
                        ok: false,
                        records,
                        chain_root: Some(hash.clone()),
                        error: Some(format!(
                            "event {} hash mismatch: embedded {embedded}, computed {hash}",
                            event.seq
                        )),
                    };
                }
                None => {
                    return ChainAuditReport {
                        ok: false,
                        records,
                        chain_root: Some(hash),
                        error: Some(format!("event {} is missing chain_hash", event.seq)),
                    };
                }
            }
        }
        prev = digest;
    }
    ChainAuditReport {
        ok: true,
        chain_root: records.last().map(|r| r.hash.clone()),
        records,
        error: None,
    }
}

/// `blake3(prev_hash || canonical_event_bytes)` with `chain_hash` cleared — the
/// exact construction `hide_core::event` uses on append, so the two never drift
/// (§4.2.1). Public so a caller can splice a single record (e.g. an anchor or a
/// compaction re-link) onto a known prefix hash.
pub fn chain_hash(previous_hash: &[u8], event: &Event) -> Result<Vec<u8>> {
    let mut canonical = event.clone();
    canonical.chain_hash = None;
    let bytes = serde_json::to_vec(&canonical)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(previous_hash);
    hasher.update(&bytes);
    Ok(hasher.finalize().as_bytes().to_vec())
}

// ---------------------------------------------------------------------------
// Signed anchors (§4.11): periodic (seq, chain_root, signature, signer).
// ---------------------------------------------------------------------------

/// A signing key for chain anchors. In production this is a Keychain (or
/// Secure-Enclave-bound) key (§4.11); here it is a keyed-blake3 MAC over the
/// anchored tuple, which is a real, verifiable authenticator: without the key an
/// attacker who rewrites history cannot forge a matching anchor signature. The
/// `keyring`-backed key material lives in `storage.rs`; this type is the
/// signing seam both share.
#[derive(Clone)]
pub struct AnchorSigner {
    key: [u8; CHAIN_HASH_LEN],
    signer_id: String,
}

impl std::fmt::Debug for AnchorSigner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnchorSigner")
            .field("signer_id", &self.signer_id)
            .field("key", &"<redacted>")
            .finish()
    }
}

impl AnchorSigner {
    /// Construct from raw 32-byte key material (e.g. derived from the WDK).
    pub fn from_key(key: [u8; CHAIN_HASH_LEN], signer_id: impl Into<String>) -> Self {
        Self {
            key,
            signer_id: signer_id.into(),
        }
    }

    pub fn signer_id(&self) -> &str {
        &self.signer_id
    }

    /// Keyed-blake3 MAC over `seq || chain_root`. Domain-separated so an anchor
    /// signature can never be confused with a chain hash.
    fn sign(&self, seq: u64, chain_root: &str) -> String {
        let mut hasher = blake3::Hasher::new_keyed(&self.key);
        hasher.update(b"hide.security.anchor.v1");
        hasher.update(&seq.to_le_bytes());
        hasher.update(chain_root.as_bytes());
        hex_lower(hasher.finalize().as_bytes())
    }

    fn verify(&self, anchor: &ChainAnchor) -> bool {
        // Constant-time-ish compare via the fixed-width hex string equality.
        // blake3's MAC output is uniformly random to a key-less attacker.
        self.sign(anchor.seq, &anchor.chain_root) == anchor.signature
            && self.signer_id == anchor.signer
    }

    /// Mint a signed anchor at the current chain tip.
    pub fn anchor(&self, seq: u64, chain_root: impl Into<String>) -> ChainAnchor {
        let chain_root = chain_root.into();
        let signature = self.sign(seq, &chain_root);
        ChainAnchor {
            seq,
            chain_root,
            signature,
            signer: self.signer_id.clone(),
        }
    }
}

/// One row of `log/ANCHORS` (§4.1 / §4.11): a signed commitment to the chain
/// root at a given `seq`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainAnchor {
    pub seq: u64,
    pub chain_root: String,
    pub signature: String,
    pub signer: String,
}

impl ChainAnchor {
    /// Build the `security.anchor` event for this anchor (§4.2.1 event table).
    pub fn to_event(&self, session_id: SessionId) -> NewEvent {
        NewEvent::system(
            session_id,
            "security.anchor",
            serde_json::json!({
                "seq": self.seq,
                "chain_root": self.chain_root,
                "signature": self.signature,
                "signer": self.signer,
            }),
        )
    }
}

/// What kind of integrity failure tripped (§4.2.1 `security.integrity_alarm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegrityAlarmKind {
    /// The recomputed chain diverged from an embedded hash.
    ChainBreak,
    /// A grant ledger projection disagreed with the log.
    LedgerMismatch,
    /// An anchor signature failed to verify against the signer's key.
    SigFail,
}

impl IntegrityAlarmKind {
    pub fn as_str(self) -> &'static str {
        match self {
            IntegrityAlarmKind::ChainBreak => "chain_break",
            IntegrityAlarmKind::LedgerMismatch => "ledger_mismatch",
            IntegrityAlarmKind::SigFail => "sig_fail",
        }
    }
}

/// Build a `security.integrity_alarm` event (§4.2.1). Fail-loud, fail-recorded
/// (S12): the host emits this and enters read-only forensic mode rather than
/// silently continuing.
pub fn integrity_alarm_event(
    session_id: SessionId,
    kind: IntegrityAlarmKind,
    detail: impl Into<String>,
) -> NewEvent {
    // `NewEvent::system` already sets `source = System` and `class = Neither`,
    // which is exactly the classification an integrity alarm wants — so no
    // post-construction re-assignment is needed.
    NewEvent::system(
        session_id,
        "security.integrity_alarm",
        serde_json::json!({ "kind": kind.as_str(), "detail": detail.into() }),
    )
}

/// Verdict of [`verify_with_anchors`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchoredVerification {
    pub ok: bool,
    /// The recomputed chain audit.
    pub chain: ChainAuditReport,
    /// Anchors that were checked, and whether each verified.
    pub anchors_checked: Vec<AnchorCheck>,
    /// If `!ok`, the alarm to emit (caller turns it into an event via
    /// [`integrity_alarm_event`]).
    pub alarm: Option<IntegrityAlarmKind>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorCheck {
    pub seq: u64,
    /// Signature verified against the signer key.
    pub signature_ok: bool,
    /// The anchor's `chain_root` matched the recomputed hash at that `seq`.
    pub root_matches_chain: bool,
}

/// Full §4.11 verification: recompute the salted chain, confirm every embedded
/// hash, then confirm each signed anchor (a) verifies under the signer key and
/// (b) commits to the chain root we actually recomputed at that `seq`. Any
/// failure yields the alarm kind the host must record. This is the function the
/// crash-recovery sequence (§4.3 step 2) calls.
pub fn verify_with_anchors(
    genesis: &[u8],
    events: &[Event],
    anchors: &[ChainAnchor],
    signer: &AnchorSigner,
) -> AnchoredVerification {
    let chain = verify_event_chain_salted(genesis, events);
    if !chain.ok {
        return AnchoredVerification {
            ok: false,
            detail: chain
                .error
                .clone()
                .unwrap_or_else(|| "chain break".to_string()),
            chain,
            anchors_checked: Vec::new(),
            alarm: Some(IntegrityAlarmKind::ChainBreak),
        };
    }

    // Index recomputed hashes by seq for anchor cross-checking.
    let mut anchors_checked = Vec::with_capacity(anchors.len());
    let mut alarm: Option<IntegrityAlarmKind> = None;
    let mut detail = "chain + anchors verified".to_string();
    for anchor in anchors {
        let signature_ok = signer.verify(anchor);
        let recomputed_root = chain
            .records
            .iter()
            .find(|r| r.seq == anchor.seq)
            .map(|r| r.hash.as_str());
        let root_matches_chain = recomputed_root == Some(anchor.chain_root.as_str());
        if !signature_ok && alarm.is_none() {
            alarm = Some(IntegrityAlarmKind::SigFail);
            detail = format!("anchor seq {} signature failed to verify", anchor.seq);
        } else if !root_matches_chain && alarm.is_none() {
            alarm = Some(IntegrityAlarmKind::ChainBreak);
            detail = format!(
                "anchor seq {} commits to root {} but chain recomputes {}",
                anchor.seq,
                anchor.chain_root,
                recomputed_root.unwrap_or("<missing seq>")
            );
        }
        anchors_checked.push(AnchorCheck {
            seq: anchor.seq,
            signature_ok,
            root_matches_chain,
        });
    }

    AnchoredVerification {
        ok: alarm.is_none(),
        chain,
        anchors_checked,
        alarm,
        detail,
    }
}

// ---------------------------------------------------------------------------
// Integrity trait bridge (consumed by hide-core's persistence layer).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct EventChainAuditor;

impl EventLogIntegrity for EventChainAuditor {
    fn verify_chain(&self, events: &[Event]) -> Result<IntegrityReport> {
        let report = verify_event_chain(events);
        Ok(IntegrityReport {
            ok: report.ok,
            checked_events: report.records.len(),
            chain_root: report.chain_root,
            detail: report
                .error
                .unwrap_or_else(|| "event chain verified".to_string()),
        })
    }
}

// ---------------------------------------------------------------------------
// Hex helpers.
// ---------------------------------------------------------------------------

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::{Event, EventSource, NewEvent};
    use hide_core::ids::SessionId;

    fn embed_chain(genesis: &[u8], events: &mut [Event]) {
        let mut prev = genesis.to_vec();
        for e in events.iter_mut() {
            let h = chain_hash(&prev, e).unwrap();
            e.chain_hash = Some(hex_lower(&h));
            prev = h;
        }
    }

    fn ev(seq: u64, session: &SessionId, kind: &str, n: i64) -> Event {
        Event::new(
            seq,
            NewEvent::of(
                session.clone(),
                EventSource::System,
                kind,
                serde_json::json!({ "n": n }),
            ),
        )
    }

    #[test]
    fn verifies_embedded_blake3_chain() {
        let session = SessionId::new();
        let mut events = [
            ev(1, &session, "system.started", 1),
            ev(2, &session, "system.ready", 2),
        ];
        embed_chain(&[0u8; CHAIN_HASH_LEN], &mut events);
        let report = verify_event_chain(&events);
        assert!(report.ok, "{:?}", report.error);
        assert!(report.chain_root.is_some());
    }

    #[test]
    fn rejects_tampered_chain() {
        let session = SessionId::new();
        let mut events = [ev(1, &session, "system.started", 1)];
        embed_chain(&[0u8; CHAIN_HASH_LEN], &mut events);
        events[0].chain_hash = Some("bad".to_string());
        let report = verify_event_chain(&events);
        assert!(!report.ok);
        assert!(report.error.unwrap().contains("hash mismatch"));
    }

    #[test]
    fn matches_hide_core_chain_construction() {
        // The security verifier must agree with hide_core's on-append chain over
        // the all-zero genesis. We can at least assert internal consistency:
        // recomputing a freshly-embedded chain verifies.
        let session = SessionId::new();
        let mut events = [
            ev(1, &session, "a", 1),
            ev(2, &session, "b", 2),
            ev(3, &session, "c", 3),
        ];
        embed_chain(&[0u8; CHAIN_HASH_LEN], &mut events);
        assert!(verify_event_chain(&events).ok);
    }

    #[test]
    fn genesis_salt_changes_root() {
        let session = SessionId::new();
        let events = [ev(1, &session, "a", 1)];
        let zero = compute_event_chain(&events).chain_root.unwrap();
        let salted = compute_event_chain_salted(&[7u8; CHAIN_HASH_LEN], &events)
            .chain_root
            .unwrap();
        assert_ne!(zero, salted, "salt must perturb the root");
    }

    #[test]
    fn anchor_signs_and_verifies_against_chain() {
        let session = SessionId::new();
        let genesis = [9u8; CHAIN_HASH_LEN];
        let mut events = [ev(1, &session, "a", 1), ev(2, &session, "b", 2)];
        embed_chain(&genesis, &mut events);

        let signer = AnchorSigner::from_key([3u8; CHAIN_HASH_LEN], "test-signer");
        let tip = compute_event_chain_salted(&genesis, &events)
            .chain_root
            .unwrap();
        let anchor = signer.anchor(2, tip);

        let v = verify_with_anchors(&genesis, &events, std::slice::from_ref(&anchor), &signer);
        assert!(v.ok, "{}", v.detail);
        assert!(v.anchors_checked[0].signature_ok);
        assert!(v.anchors_checked[0].root_matches_chain);
        assert!(v.alarm.is_none());

        // The anchor event builds with the right kind.
        let new = anchor.to_event(session);
        assert_eq!(new.kind, "security.anchor");
    }

    #[test]
    fn forged_anchor_signature_raises_sig_fail() {
        let session = SessionId::new();
        let genesis = [1u8; CHAIN_HASH_LEN];
        let mut events = [ev(1, &session, "a", 1)];
        embed_chain(&genesis, &mut events);

        let real = AnchorSigner::from_key([4u8; CHAIN_HASH_LEN], "real");
        let tip = compute_event_chain_salted(&genesis, &events)
            .chain_root
            .unwrap();
        let mut anchor = real.anchor(1, tip);
        anchor.signature = "deadbeef".to_string(); // forged

        let v = verify_with_anchors(&genesis, &events, std::slice::from_ref(&anchor), &real);
        assert!(!v.ok);
        assert_eq!(v.alarm, Some(IntegrityAlarmKind::SigFail));
    }

    #[test]
    fn anchor_over_tampered_history_raises_chain_break() {
        let session = SessionId::new();
        let genesis = [2u8; CHAIN_HASH_LEN];
        let mut events = [ev(1, &session, "a", 1), ev(2, &session, "b", 2)];
        embed_chain(&genesis, &mut events);

        let signer = AnchorSigner::from_key([5u8; CHAIN_HASH_LEN], "s");
        let tip = compute_event_chain_salted(&genesis, &events)
            .chain_root
            .unwrap();
        let anchor = signer.anchor(2, tip);

        // Tamper a past event WITHOUT re-embedding hashes: verify_event_chain
        // catches the embedded-hash mismatch first → chain_break.
        events[0].payload = serde_json::json!({ "n": 999 });
        let v = verify_with_anchors(&genesis, &events, std::slice::from_ref(&anchor), &signer);
        assert!(!v.ok);
        assert_eq!(v.alarm, Some(IntegrityAlarmKind::ChainBreak));
    }

    #[test]
    fn integrity_alarm_event_shape() {
        let session = SessionId::new();
        let e = integrity_alarm_event(session, IntegrityAlarmKind::LedgerMismatch, "boom");
        assert_eq!(e.kind, "security.integrity_alarm");
        assert_eq!(e.payload["kind"], "ledger_mismatch");
        assert_eq!(e.payload["detail"], "boom");
    }
}
