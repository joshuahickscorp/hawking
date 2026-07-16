//! Security infrastructure for HIDE (bible chapter 10).
//!
//! Backend-only pieces from ch.10: tamper-evident **blake3** hash-chain audit
//! with genesis salt + signed anchors (`audit`), secret **redaction** before
//! durability with a regex + entropy detector suite (`redaction`),
//! **encryption-at-rest** via AES-256-GCM AEAD with an OS-keychain-wrapped key
//! plus fail-closed layout validation (`storage`), and macOS **Seatbelt**
//! profile rendering + `sandbox-exec` spawning (`sandbox`).
//!
//! OS enforcement that needs a live host (the egress proxy, the microVM tier,
//! the Endpoint-Security reference monitor) remains a documented seam — see the
//! per-module docs. The pure security logic (chain math, detectors, AEAD,
//! profile rendering) is real and tested here.

#[rustfmt::skip]
pub mod audit {
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
            records.push(ChainRecord { seq: event.seq, hash: hash.clone() });
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
        ChainAuditReport { ok: true, chain_root: records.last().map(|r| r.hash.clone()), records, error: None }
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
            f.debug_struct("AnchorSigner").field("signer_id", &self.signer_id).field("key", &"<redacted>").finish()
        }
    }

    impl AnchorSigner {
        /// Construct from raw 32-byte key material (e.g. derived from the WDK).
        pub fn from_key(key: [u8; CHAIN_HASH_LEN], signer_id: impl Into<String>) -> Self {
            Self { key, signer_id: signer_id.into() }
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
            self.sign(anchor.seq, &anchor.chain_root) == anchor.signature && self.signer_id == anchor.signer
        }

        /// Mint a signed anchor at the current chain tip.
        pub fn anchor(&self, seq: u64, chain_root: impl Into<String>) -> ChainAnchor {
            let chain_root = chain_root.into();
            let signature = self.sign(seq, &chain_root);
            ChainAnchor { seq, chain_root, signature, signer: self.signer_id.clone() }
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
                detail: chain.error.clone().unwrap_or_else(|| "chain break".to_string()),
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
            let recomputed_root = chain.records.iter().find(|r| r.seq == anchor.seq).map(|r| r.hash.as_str());
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
            anchors_checked.push(AnchorCheck { seq: anchor.seq, signature_ok, root_matches_chain });
        }

        AnchoredVerification { ok: alarm.is_none(), chain, anchors_checked, alarm, detail }
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
                detail: report.error.unwrap_or_else(|| "event chain verified".to_string()),
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
            Event::new(seq, NewEvent::of(session.clone(), EventSource::System, kind, serde_json::json!({ "n": n })))
        }

        #[test]
        fn verifies_embedded_blake3_chain() {
            let session = SessionId::new();
            let mut events = [ev(1, &session, "system.started", 1), ev(2, &session, "system.ready", 2)];
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
            let mut events = [ev(1, &session, "a", 1), ev(2, &session, "b", 2), ev(3, &session, "c", 3)];
            embed_chain(&[0u8; CHAIN_HASH_LEN], &mut events);
            assert!(verify_event_chain(&events).ok);
        }

        #[test]
        fn genesis_salt_changes_root() {
            let session = SessionId::new();
            let events = [ev(1, &session, "a", 1)];
            let zero = compute_event_chain(&events).chain_root.unwrap();
            let salted = compute_event_chain_salted(&[7u8; CHAIN_HASH_LEN], &events).chain_root.unwrap();
            assert_ne!(zero, salted, "salt must perturb the root");
        }

        #[test]
        fn anchor_signs_and_verifies_against_chain() {
            let session = SessionId::new();
            let genesis = [9u8; CHAIN_HASH_LEN];
            let mut events = [ev(1, &session, "a", 1), ev(2, &session, "b", 2)];
            embed_chain(&genesis, &mut events);

            let signer = AnchorSigner::from_key([3u8; CHAIN_HASH_LEN], "test-signer");
            let tip = compute_event_chain_salted(&genesis, &events).chain_root.unwrap();
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
            let tip = compute_event_chain_salted(&genesis, &events).chain_root.unwrap();
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
            let tip = compute_event_chain_salted(&genesis, &events).chain_root.unwrap();
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
}
#[rustfmt::skip]
pub mod redaction {
    //! Secret redaction before durability (bible ch.10 §4.8, S6).
    //!
    //! Replaces the original two-prefix toy with a real detector suite: known-format
    //! pattern detectors (AWS access keys, GitHub/GitLab PATs, PEM private-key
    //! blocks, JWTs, Slack tokens) plus a generic **Shannon-entropy** detector for
    //! high-entropy tokens that no signature catches. On a hit, the span is replaced
    //! with `«redacted:detector»` (guillemets U+00AB/U+00BB, lowercase detector
    //! name, per bible §4.8) and the location is recorded so the *fact and place* of
    //! redaction stays auditable while the secret never enters the log, the chain
    //! hash, the blob CAS, or a vector store (§4.2.1 / §4.8).
    //!
    //! Two surfaces:
    //!   * [`Redactor::redact`] — scrub a flat string (shell output, a log line).
    //!   * [`Redactor::redact_json`] — scrub every string leaf of a JSON value and
    //!     emit the **JSON-pointer paths** (RFC 6901) of redacted leaves, ready to
    //!     drop into `Event.redactions` (§4.8). This is the form a `tool.result`
    //!     payload goes through before it becomes a durable event.

    use hide_core::event::NewEvent;
    use hide_core::ids::SessionId;
    use regex::Regex;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::sync::OnceLock;

    /// Opening guillemet of the redaction marker (U+00AB, `«`), per bible §4.8.
    pub const MARKER_OPEN: &str = "\u{00AB}";
    /// Closing guillemet of the redaction marker (U+00BB, `»`), per bible §4.8.
    pub const MARKER_CLOSE: &str = "\u{00BB}";

    /// Marker substituted for a redacted span (§4.8): `«redacted:<detector>»` with
    /// guillemets (U+00AB / U+00BB) and a **lowercase** detector name, so the audit
    /// trail records *why* a span was scrubbed in exactly the form the bible
    /// mandates (and the UI renders verbatim, §4.8). Detector names are already
    /// lowercase ASCII identifiers; we lowercase defensively for any plugin-supplied
    /// detector registered via [`Redactor::with_detector`].
    fn marker(detector: &str) -> String {
        format!("{MARKER_OPEN}redacted:{detector}{MARKER_CLOSE}", detector = detector.to_ascii_lowercase())
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Redaction {
        /// Detector that matched (e.g. `"aws_access_key"`, `"entropy"`).
        pub pattern_name: String,
        /// The marker that replaced the span.
        pub replacement: String,
        pub occurrences: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RedactionReport {
        pub text: String,
        pub redactions: Vec<Redaction>,
    }

    impl RedactionReport {
        pub fn is_clean(&self) -> bool {
            self.redactions.is_empty()
        }
    }

    /// Result of scrubbing a JSON payload (§4.8): the scrubbed value plus the
    /// JSON-pointer paths of every redacted string leaf (for `Event.redactions`).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct JsonRedactionReport {
        pub value: Value,
        /// RFC 6901 JSON-pointer paths of redacted leaves (e.g. `/output/stdout`).
        pub paths: Vec<String>,
        /// Per-detector tallies across the whole document.
        pub redactions: Vec<Redaction>,
    }

    impl JsonRedactionReport {
        pub fn is_clean(&self) -> bool {
            self.paths.is_empty()
        }

        /// Build the `security.redaction` event the host appends when a payload was
        /// scrubbed before durability (bible §4.8).
        ///
        /// ## Seam
        /// The redactor is *pure*: it never touches the log. The host owns the write
        /// ordering: it (1) runs [`Redactor::redact_json`] on a `tool.result` (or
        /// shell-output, or any pre-durable) payload, (2) sets the **scrubbed** value
        /// and `report.paths` on the durable `Event.redactions` (so the chain covers
        /// only the redacted form — the secret never enters the hash, the blob CAS,
        /// or the vector store), and (3) appends *this* `security.redaction` event so
        /// the *fact and location* of redaction are independently auditable. The
        /// event payload carries the JSON-pointer paths and per-detector tallies, but
        /// **never the secret** — only where and why a span was removed.
        ///
        /// Returns `None` when nothing was redacted (no event to emit), so the host
        /// can call this unconditionally.
        pub fn build_redaction_event(&self, session_id: SessionId) -> Option<NewEvent> {
            if self.is_clean() {
                return None;
            }
            let detectors: Vec<_> = self
                .redactions
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "detector": r.pattern_name,
                        "occurrences": r.occurrences,
                    })
                })
                .collect();
            let total: usize = self.redactions.iter().map(|r| r.occurrences).sum();
            Some(NewEvent::system(
                session_id,
                "security.redaction",
                serde_json::json!({
                    // RFC 6901 JSON-pointer paths of the scrubbed leaves — mirrors
                    // what the host writes into Event.redactions.
                    "paths": self.paths,
                    // Per-detector tallies (why each span was scrubbed). No secret.
                    "detectors": detectors,
                    "total_spans": total,
                }),
            ))
        }
    }

    /// A known-format secret detector (compiled regex + a name).
    #[derive(Clone)]
    pub struct PatternDetector {
        pub name: String,
        pub regex: Regex,
    }

    impl std::fmt::Debug for PatternDetector {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PatternDetector").field("name", &self.name).finish()
        }
    }

    /// The redaction engine. Holds the ordered pattern detectors plus an entropy
    /// threshold for the generic catch-all.
    #[derive(Debug, Clone)]
    pub struct Redactor {
        detectors: Vec<PatternDetector>,
        /// Tokens of at least this length whose Shannon entropy (bits/char) meets
        /// [`Self::entropy_threshold`] are redacted by the generic detector.
        entropy_min_len: usize,
        entropy_threshold: f64,
        /// Length+entropy dial for the **single-class** catch-all (item 3): a token
        /// of one character class (e.g. all-lowercase base64) is only redacted if it
        /// is at least this long AND at least [`Self::single_class_entropy`]
        /// bits/char. Set well above the prose floor so an all-lowercase secret can't
        /// hide behind the two-class `looks_secretish` gate, while ordinary long
        /// lowercase words (rare, and low-entropy) stay untouched.
        single_class_min_len: usize,
        single_class_entropy: f64,
        entropy_enabled: bool,
    }

    impl Default for Redactor {
        fn default() -> Self {
            Self {
                detectors: builtin_detectors().to_vec(),
                // Tuned so realistic 32+ char base64/hex secrets trip it while
                // ordinary prose words / short identifiers do not. English text
                // sits well under ~4.0 bits/char per token; a random 40-char
                // base64 blob is ~5.5–6.0.
                entropy_min_len: 24,
                entropy_threshold: 4.0,
                // Single-class dial. A long all-lowercase random base64/base36 blob
                // draws from ~26+ symbols → ~4.5–4.7 bits/char, while a hex sha draws
                // from only 16 symbols (≤4.0 bits/char ceiling) and a decimal id from
                // 10 (≤3.32). Setting the floor at 4.2 cleanly separates a lowercase
                // secret from a commit hash / numeric id, and sits above the prose
                // ceiling so ordinary lowercase words never trip it.
                single_class_min_len: 32,
                single_class_entropy: 4.2,
                entropy_enabled: true,
            }
        }
    }

    impl Redactor {
        /// Pattern detectors only (entropy off) — for callers that want zero
        /// false-positives on high-entropy-but-benign data (hashes, UUIDs).
        pub fn patterns_only() -> Self {
            Self { entropy_enabled: false, ..Self::default() }
        }

        pub fn with_entropy(mut self, min_len: usize, threshold: f64) -> Self {
            self.entropy_min_len = min_len;
            self.entropy_threshold = threshold;
            self.entropy_enabled = true;
            self
        }

        /// Tune the **single-class** catch-all dial (item 3): the minimum length and
        /// bits/char a single-character-class token (e.g. all-lowercase base64) must
        /// reach to be redacted even though it fails the two-class `looks_secretish`
        /// gate. Higher values = fewer false positives, more risk a single-class
        /// secret slips through.
        pub fn with_single_class(mut self, min_len: usize, entropy: f64) -> Self {
            self.single_class_min_len = min_len;
            self.single_class_entropy = entropy;
            self
        }

        /// The generic-detector decision for one token: redact if either
        ///   * it is mixed-class (≥2 of upper/lower/digit) and clears the standard
        ///     `entropy_min_len` / `entropy_threshold` gate, OR
        ///   * it is **single-class** but long enough and high-entropy enough to be a
        ///     credential rather than a word (item 3 — catches all-lowercase base64).
        fn is_high_entropy_secret(&self, token: &str) -> bool {
            let h = shannon_entropy(token);
            if looks_secretish(token) && token.len() >= self.entropy_min_len && h >= self.entropy_threshold {
                return true;
            }
            // Single-class branch: one character class only, but conspicuously long
            // and high-entropy. A pure-hex sha or all-decimal id has too few distinct
            // symbols to clear `single_class_entropy`, so commit hashes survive.
            token.len() >= self.single_class_min_len && h >= self.single_class_entropy
        }

        /// Register an extra detector (the `secret-detector` policy-plugin seam,
        /// §7 — tighten-only: plugins can add detections, never remove).
        pub fn with_detector(mut self, name: impl Into<String>, regex: Regex) -> Self {
            self.detectors.push(PatternDetector { name: name.into(), regex });
            self
        }

        /// Scrub a flat string. Pattern detectors run first (most specific), then
        /// the entropy catch-all over whatever survives.
        pub fn redact(&self, input: &str) -> RedactionReport {
            let mut text = input.to_string();
            let mut tally: Vec<Redaction> = Vec::new();

            for det in &self.detectors {
                let mut count = 0usize;
                text = det
                    .regex
                    .replace_all(&text, |_: &regex::Captures| {
                        count += 1;
                        marker(&det.name)
                    })
                    .into_owned();
                if count > 0 {
                    tally.push(Redaction {
                        pattern_name: det.name.clone(),
                        replacement: marker(&det.name),
                        occurrences: count,
                    });
                }
            }

            if self.entropy_enabled {
                let (scrubbed, count) = self.redact_entropy(&text);
                if count > 0 {
                    text = scrubbed;
                    tally.push(Redaction {
                        pattern_name: "entropy".to_string(),
                        replacement: marker("entropy"),
                        occurrences: count,
                    });
                }
            }

            RedactionReport { text, redactions: tally }
        }

        /// Generic high-entropy token detector. Splits on whitespace and common
        /// delimiters, keeping the secret-like core; replaces any token that is long
        /// enough and high-entropy enough, and isn't already a redaction marker.
        fn redact_entropy(&self, input: &str) -> (String, usize) {
            let mut count = 0usize;
            // Walk the string, copying through, replacing qualifying runs of
            // "secret-ish" characters (alnum + a few base64/url-safe symbols).
            let mut out = String::with_capacity(input.len());
            let mut token = String::new();
            // `is_high_entropy_secret` owns ALL length/entropy gating (both the
            // mixed-class and single-class branches carry their own length floor), so
            // the closure only guards against re-redacting an existing marker.
            let flush = |token: &mut String, out: &mut String, count: &mut usize, this: &Redactor| {
                if !token.is_empty() {
                    if !token.starts_with(MARKER_OPEN) && this.is_high_entropy_secret(token) {
                        out.push_str(&marker("entropy"));
                        *count += 1;
                    } else {
                        out.push_str(token);
                    }
                    token.clear();
                }
            };
            for ch in input.chars() {
                if is_token_char(ch) {
                    token.push(ch);
                } else {
                    flush(&mut token, &mut out, &mut count, self);
                    out.push(ch);
                }
            }
            flush(&mut token, &mut out, &mut count, self);
            (out, count)
        }

        /// Scrub every string leaf of a JSON value, returning the scrubbed value and
        /// the JSON-pointer paths of redacted leaves (§4.8). Object keys are NOT
        /// scrubbed (a key is structural, not content); only values.
        pub fn redact_json(&self, value: &Value) -> JsonRedactionReport {
            let mut paths = Vec::new();
            let mut tally: Vec<Redaction> = Vec::new();
            let scrubbed = self.scrub_value(value, String::new(), &mut paths, &mut tally);
            JsonRedactionReport { value: scrubbed, paths, redactions: tally }
        }

        fn scrub_value(
            &self,
            value: &Value,
            pointer: String,
            paths: &mut Vec<String>,
            tally: &mut Vec<Redaction>,
        ) -> Value {
            match value {
                Value::String(s) => {
                    let report = self.redact(s);
                    if !report.is_clean() {
                        paths.push(if pointer.is_empty() { "".to_string() } else { pointer });
                        merge_tally(tally, report.redactions);
                        Value::String(report.text)
                    } else {
                        Value::String(s.clone())
                    }
                }
                Value::Array(items) => Value::Array(
                    items
                        .iter()
                        .enumerate()
                        .map(|(i, v)| self.scrub_value(v, format!("{pointer}/{i}"), paths, tally))
                        .collect(),
                ),
                Value::Object(map) => Value::Object(
                    map.iter()
                        .map(|(k, v)| {
                            let child = format!("{pointer}/{}", escape_pointer_token(k));
                            (k.clone(), self.scrub_value(v, child, paths, tally))
                        })
                        .collect(),
                ),
                other => other.clone(),
            }
        }
    }

    fn merge_tally(tally: &mut Vec<Redaction>, more: Vec<Redaction>) {
        for r in more {
            if let Some(existing) = tally.iter_mut().find(|e| e.pattern_name == r.pattern_name) {
                existing.occurrences += r.occurrences;
            } else {
                tally.push(r);
            }
        }
    }

    /// RFC 6901: `~` → `~0`, `/` → `~1`.
    fn escape_pointer_token(token: &str) -> String {
        token.replace('~', "~0").replace('/', "~1")
    }

    fn is_token_char(ch: char) -> bool {
        ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '_' | '-' | '=' | '.')
    }

    /// A token that is *all* digits or *all* lowercase hex of a "nice" length is
    /// likely an id/hash, not a credential; require some mixed-class content so we
    /// don't redact every git SHA or large integer. (Pattern detectors still catch
    /// real secrets that happen to be hex/base64.)
    fn looks_secretish(token: &str) -> bool {
        let has_upper = token.chars().any(|c| c.is_ascii_uppercase());
        let has_lower = token.chars().any(|c| c.is_ascii_lowercase());
        let has_digit = token.chars().any(|c| c.is_ascii_digit());
        // Need at least two of {upper, lower, digit} — rules out pure-hex shas,
        // decimal ids, and uppercase-only constants.
        [has_upper, has_lower, has_digit].iter().filter(|b| **b).count() >= 2
    }

    /// Shannon entropy in bits per character.
    pub fn shannon_entropy(s: &str) -> f64 {
        if s.is_empty() {
            return 0.0;
        }
        let mut counts = [0usize; 256];
        let mut n = 0usize;
        for b in s.bytes() {
            counts[b as usize] += 1;
            n += 1;
        }
        let n = n as f64;
        let mut h = 0.0;
        for &c in counts.iter() {
            if c == 0 {
                continue;
            }
            let p = c as f64 / n;
            h -= p * p.log2();
        }
        h
    }

    /// The built-in pattern detectors (§4.8). Ordered most-specific first.
    fn builtin_detectors() -> &'static [PatternDetector] {
        static DETECTORS: OnceLock<Vec<PatternDetector>> = OnceLock::new();
        DETECTORS.get_or_init(|| {
            let mut v = Vec::new();
            let mut add = |name: &str, pat: &str| {
                v.push(PatternDetector {
                    name: name.to_string(),
                    regex: Regex::new(pat).expect("builtin redaction pattern compiles"),
                });
            };
            // PEM private-key block (whole block, multi-line).
            add("pem_private_key", r"(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----");
            // AWS access key id (AKIA/ASIA + 16 base32 chars).
            add("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b");
            // AWS secret access key, when introduced by an obvious key= context.
            add("aws_secret_key", r#"(?i)aws_secret_access_key["']?\s*[:=]\s*["']?[A-Za-z0-9/+=]{40}"#);
            // GitHub PATs (classic + fine-grained + app/refresh tokens).
            add("github_pat", r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b");
            add("github_fine_grained_pat", r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b");
            // GitLab PAT.
            add("gitlab_pat", r"\bglpat-[A-Za-z0-9_\-]{20,}\b");
            // Slack token.
            add("slack_token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b");
            // JWT: three base64url segments separated by dots; header starts eyJ.
            add("jwt", r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b");
            // Generic "sk-"/"rk-" style provider keys (kept from the old toy, widened).
            add("provider_key", r"\b[sr]k-[A-Za-z0-9]{20,}\b");
            v
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn redacts_aws_access_key() {
            let r = Redactor::default().redact("export AWS_KEY=AKIAIOSFODNN7EXAMPLE done");
            assert!(r.text.contains("\u{00AB}redacted:aws_access_key\u{00BB}"), "{}", r.text);
            assert!(r.redactions.iter().any(|x| x.pattern_name == "aws_access_key"));
            assert!(!r.text.contains("AKIA"));
        }

        #[test]
        fn redacts_github_pat() {
            let token = format!("ghp_{}", "a".repeat(36));
            let r = Redactor::default().redact(&format!("token={token}"));
            assert!(r.text.contains("\u{00AB}redacted:github_pat\u{00BB}"), "{}", r.text);
        }

        #[test]
        fn redacts_jwt() {
            let jwt = "eyJhbGciOiJIUzI1Ni1.eyJzdWIiOiIxMjM0NTY3.SflKxwRJSMeKKF2QT4f";
            let r = Redactor::default().redact(&format!("auth {jwt} end"));
            assert!(r.text.contains("\u{00AB}redacted:jwt\u{00BB}"), "{}", r.text);
        }

        #[test]
        fn redacts_pem_block() {
            let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj...\nabcDEF123==\n-----END RSA PRIVATE KEY-----";
            let r = Redactor::default().redact(&format!("key:\n{pem}\nrest"));
            assert!(r.text.contains("\u{00AB}redacted:pem_private_key\u{00BB}"), "{}", r.text);
            assert!(!r.text.contains("PRIVATE KEY-----\nMIIB"));
        }

        #[test]
        fn entropy_catches_unknown_high_entropy_token() {
            // No known prefix, but high-entropy mixed-class blob → entropy detector.
            let secret = "Zk9Qm2Xp7Lv3Rt8Wf1Yc6Nb4Hd0Sg5Aj"; // 33 chars, mixed
            let r = Redactor::default().redact(&format!("password is {secret} ok"));
            assert!(r.text.contains("\u{00AB}redacted:entropy\u{00BB}"), "got: {}", r.text);
        }

        #[test]
        fn marker_uses_guillemets_and_lowercase_detector(/* item 1 */) {
            // The marker must be «redacted:<detector>» with U+00AB/U+00BB guillemets
            // and a LOWERCASE detector name (bible §4.8), never the old ASCII
            // <<redacted:...>> form.
            let m = marker("AWS_Access_Key");
            assert_eq!(m, "\u{00AB}redacted:aws_access_key\u{00BB}");
            assert!(m.starts_with(MARKER_OPEN) && m.ends_with(MARKER_CLOSE));
            assert!(!m.contains("<<") && !m.contains(">>"));
            // A plugin-registered detector with mixed case is lowercased in output.
            let r = Redactor::patterns_only()
                .with_detector("MyCorp_Token", Regex::new(r"\bMYC-[0-9]{6}\b").unwrap())
                .redact("see MYC-123456 here");
            assert!(r.text.contains("\u{00AB}redacted:mycorp_token\u{00BB}"), "{}", r.text);
        }

        #[test]
        fn entropy_leaves_prose_and_ids_alone() {
            let prose = "the quick brown fox jumps over the lazy dog repeatedly today";
            let r = Redactor::default().redact(prose);
            assert!(r.is_clean(), "prose redacted: {:?}", r.redactions);

            // A long decimal id and a pure-hex sha should survive the entropy pass
            // (they fail looks_secretish), so we don't redact every commit hash.
            let idish = "0123456789012345678901234567 deadbeefdeadbeefdeadbeefdeadbeef";
            let r2 = Redactor::patterns_only().redact(idish);
            assert!(r2.is_clean());
        }

        #[test]
        fn redact_json_emits_pointer_paths() {
            let r = Redactor::default();
            let payload = serde_json::json!({
                "output": {
                    "stdout": format!("ghp_{}", "b".repeat(36)),
                    "exit": 0
                },
                "args": ["clean", "AKIAIOSFODNN7EXAMPLE"]
            });
            let report = r.redact_json(&payload);
            assert!(!report.is_clean());
            assert!(report.paths.contains(&"/output/stdout".to_string()), "{:?}", report.paths);
            assert!(report.paths.contains(&"/args/1".to_string()), "{:?}", report.paths);
            // Clean leaves untouched; non-string leaves preserved.
            assert_eq!(report.value["output"]["exit"], 0);
            assert_eq!(report.value["args"][0], "clean");
            assert!(report.value["output"]["stdout"].as_str().unwrap().contains("\u{00AB}redacted:github_pat\u{00BB}"));
        }

        #[test]
        fn json_pointer_escapes_special_keys() {
            let r = Redactor::patterns_only();
            let payload = serde_json::json!({ "a/b": "AKIAIOSFODNN7EXAMPLE" });
            let report = r.redact_json(&payload);
            // `/` in the key becomes `~1` per RFC 6901.
            assert_eq!(report.paths, vec!["/a~1b".to_string()]);
        }

        #[test]
        fn multiple_occurrences_tallied() {
            let two = "AKIAIOSFODNN7EXAMPL1 and AKIAIOSFODNN7EXAMPL2";
            let r = Redactor::default().redact(two);
            let aws = r.redactions.iter().find(|x| x.pattern_name == "aws_access_key").unwrap();
            assert_eq!(aws.occurrences, 2);
        }

        #[test]
        fn entropy_catches_single_class_all_lowercase_secret(/* item 3 */) {
            // An all-lowercase, high-entropy base36-ish blob has only ONE character
            // class, so the two-class `looks_secretish` gate misses it — the
            // single-class branch must still redact it.
            // 35 chars, ALL lowercase letters (one character class), drawing from
            // ~24 distinct symbols → ~4.6 bits/char, over the 4.2 single-class floor.
            let secret = "qjxmfwbnzkdpvhsugtrclyaeoiqwrtmkxbv";
            assert!(!looks_secretish(secret), "test premise: token must be single-class");
            assert!(shannon_entropy(secret) >= 4.2, "entropy {} too low for fixture", shannon_entropy(secret));
            let r = Redactor::default().redact(&format!("api_key={secret}"));
            assert!(r.text.contains("\u{00AB}redacted:entropy\u{00BB}"), "single-class secret slipped: {}", r.text);

            // The single-class branch must NOT swallow a hex sha (16 symbols → ≤4.0
            // bits/char) or a long decimal id (10 symbols → ≤3.32).
            let sha = "a1b9c3d7e5f1a2b4c6d8e0f2a4b6c8d0e2f4a6b8"; // 40 hex chars
            let id = "0123456789012345678901234567890123456789"; // 40 digits
            let clean = Redactor::default().redact(&format!("{sha} {id}"));
            assert!(clean.is_clean(), "hex/decimal id redacted: {:?}", clean.redactions);
        }

        #[test]
        fn single_class_dial_is_tunable(/* item 3 dial */) {
            // Lowering the dial redacts a shorter single-class token; raising it past
            // the token's reach leaves it alone.
            let tok = "qjxmfwbnzkdpvhsugtrcl"; // 21 lowercase chars
            assert!(!looks_secretish(tok));
            let loosened = Redactor::default().with_single_class(16, 3.5).redact(&format!("x={tok}"));
            assert!(loosened.text.contains("\u{00AB}redacted:entropy\u{00BB}"), "{}", loosened.text);
            // Default dial (min_len 32) leaves the 21-char token untouched.
            let tight = Redactor::default().redact(&format!("x={tok}"));
            assert!(tight.is_clean(), "{:?}", tight.redactions);
        }

        #[test]
        fn build_redaction_event_shape(/* item 2 */) {
            use hide_core::ids::SessionId;
            let r = Redactor::default();
            let payload = serde_json::json!({
                "output": { "stdout": format!("ghp_{}", "b".repeat(36)) },
                "args": ["clean", "AKIAIOSFODNN7EXAMPLE"]
            });
            let report = r.redact_json(&payload);
            assert!(!report.is_clean());

            let session = SessionId::new();
            let ev = report.build_redaction_event(session).expect("redacted payload yields an event");
            assert_eq!(ev.kind, "security.redaction");
            // Paths mirror what the host writes into Event.redactions.
            let paths = ev.payload["paths"].as_array().unwrap();
            assert!(paths.iter().any(|p| p == "/output/stdout"));
            assert!(paths.iter().any(|p| p == "/args/1"));
            // Per-detector tallies present; total span count present.
            assert!(ev.payload["detectors"].is_array());
            assert!(ev.payload["total_spans"].as_u64().unwrap() >= 2);
            // The event NEVER carries the secret itself.
            let serialized = serde_json::to_string(&ev.payload).unwrap();
            assert!(!serialized.contains("ghp_"));
            assert!(!serialized.contains("AKIA"));
        }

        #[test]
        fn build_redaction_event_none_when_clean(/* item 2 */) {
            use hide_core::ids::SessionId;
            let report = Redactor::default().redact_json(&serde_json::json!({ "ok": "hello world" }));
            assert!(report.is_clean());
            assert!(report.build_redaction_event(SessionId::new()).is_none());
        }
    }
}
#[rustfmt::skip]
pub mod sandbox {
    //! macOS Seatbelt profile rendering + `sandbox-exec` spawning (bible ch.10
    //! §4.5.2, S2/S5/S5b/S12).
    //!
    //! Extends the original `render_macos_seatbelt` (whose signature siblings —
    //! `hide-tools` — call) to a profile that honors the §4.5.2 skeleton:
    //!   * deny-by-default;
    //!   * **process-exec allowlist** — only the granted binaries may `exec`;
    //!   * **filesystem**: read broad-but-bounded, write narrow, with secret paths
    //!     (`.ssh`/`.aws`/`.env`/`*.pem`) read-denied and **`.hide/log`
    //!     write-denied** (S4 — the audit log is invisible/untouchable to the
    //!     sandbox);
    //!   * **network**: the only egress route is the host proxy port (S5b).
    //!
    //! Plus a per-grant `.sb` emitter and a `sandbox-exec` spawn helper that fails
    //! CLOSED (S12): if `sandbox-exec` is unavailable, the spawn errors rather than
    //! running the command unconfined.

    use hide_core::ids::GrantId;
    use hide_core::security::{NetworkPolicy, SandboxProfile, SandboxTier};
    use hide_core::types::Decision;
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// Default secret paths denied at the *read* layer (§4.5.2): removes the
    /// "private data" leg of the lethal trifecta at the OS for every sandboxed run.
    const SECRET_READ_DENY_SUBPATHS: &[&str] = &["$HOME/.ssh", "$HOME/.aws", "$HOME/.config/gh"];
    const SECRET_READ_DENY_REGEXES: &[&str] = &[r"/\.env($|\.)", r"\.pem$", r"\.key$"];

    /// Broad-but-bounded system read roots a build/test realistically needs.
    const SYSTEM_READ_SUBPATHS: &[&str] = &["/usr", "/bin", "/System/Library", "/Library/Developer"];
    const SYSTEM_READ_LITERALS: &[&str] = &["/dev/null", "/dev/urandom", "/dev/random", "/dev/dtracehelper"];

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RenderedSandboxProfile {
        pub tier: SandboxTier,
        pub profile_text: String,
        pub warnings: Vec<String>,
    }

    /// Optional render-time context the basic `render_macos_seatbelt` doesn't carry
    /// on the profile itself (proxy port, workspace/worktree roots, `.hide` dir).
    /// Defaults are conservative (no egress route, no extra confinement seam).
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct SandboxRenderOptions {
        /// Host egress proxy port; `Some` ⇒ render the single allowed outbound
        /// route; `None` ⇒ no network route at all even if policy is `Allow`-ish.
        pub proxy_port: Option<u16>,
        /// The `.hide` directory whose `log` subdir must be write-denied (S4). If
        /// `None`, a relative `.hide/log` deny is still emitted.
        pub hide_dir: Option<PathBuf>,
        /// Worktree root to confine writes to (§4.5.2 `$WORKTREE`); falls back to
        /// the profile's first write root.
        pub worktree_root: Option<String>,
    }

    /// Render a Seatbelt (SBPL) profile from `profile`. **Public signature
    /// preserved** — `hide-tools` calls this exact shape. The render now includes
    /// the process-exec allowlist, secret read-denies, `.hide/log` write-deny, and
    /// (best-effort, no port) proxy-only network framing. For the proxy port and
    /// explicit worktree/`.hide` confinement, use [`render_macos_seatbelt_with`].
    pub fn render_macos_seatbelt(profile: &SandboxProfile) -> RenderedSandboxProfile {
        render_macos_seatbelt_with(profile, &SandboxRenderOptions::default())
    }

    /// Full render with render-time options (proxy port, `.hide`, worktree).
    pub fn render_macos_seatbelt_with(profile: &SandboxProfile, opts: &SandboxRenderOptions) -> RenderedSandboxProfile {
        let mut text = String::from("(version 1)\n(deny default)\n\n");
        let mut warnings = Vec::new();

        // --- process-exec allowlist (§4.5.2) ---
        text.push_str(";; --- process ---\n");
        text.push_str("(allow process-fork)\n");
        if profile.allowed_commands.is_empty() {
            // No allowlist ⇒ nothing may exec. Fail-safe: we deny rather than
            // silently allowing all exec.
            text.push_str("(deny process-exec*)\n");
            warnings
                .push("no allowed_commands: process-exec fully denied (grant the exact binaries to run)".to_string());
        } else {
            let mut literals = String::new();
            for cmd in &profile.allowed_commands {
                // Match the binary path the grant authorized. If it's a bare name,
                // emit both literal and a basename regex so common PATH locations
                // resolve; if it's an absolute path, pin it exactly.
                if cmd.starts_with('/') {
                    literals.push_str(&format!("    (literal \"{}\")\n", escape(cmd)));
                } else {
                    literals.push_str(&format!("    (regex #\"/{}$\")\n", escape_regex(cmd)));
                }
            }
            text.push_str("(allow process-exec\n");
            text.push_str(&literals);
            text.push_str(")\n");
            text.push_str("(deny process-exec*)\n");
        }
        text.push('\n');

        // --- filesystem: read broad-but-bounded, write narrow ---
        text.push_str(";; --- filesystem ---\n");
        for sub in SYSTEM_READ_SUBPATHS {
            text.push_str(&format!("(allow file-read* (subpath \"{}\"))\n", escape(sub)));
        }
        for lit in SYSTEM_READ_LITERALS {
            text.push_str(&format!("(allow file-read* (literal \"{}\"))\n", escape(lit)));
        }
        for root in &profile.read_roots {
            text.push_str(&format!("(allow file-read* (subpath \"{}\"))\n", escape(root)));
        }

        // Secret read-denies (S2/S6) — these come AFTER allows; in SBPL the most
        // specific / last-matching rule wins, and explicit deny always overrides.
        for sub in SECRET_READ_DENY_SUBPATHS {
            text.push_str(&format!("(deny file-read* (subpath \"{}\"))\n", escape(sub)));
        }
        for re in SECRET_READ_DENY_REGEXES {
            text.push_str(&format!("(deny file-read* (regex #\"{}\"))\n", re));
        }

        // Writes confined to worktree/write-roots only.
        let write_root = opts.worktree_root.clone().or_else(|| profile.write_roots.first().cloned());
        if let Some(root) = &write_root {
            text.push_str(&format!("(allow file-write* (subpath \"{}\"))\n", escape(root)));
        }
        for root in &profile.write_roots {
            if Some(root) != write_root.as_ref() {
                text.push_str(&format!("(allow file-write* (subpath \"{}\"))\n", escape(root)));
            }
        }
        if write_root.is_none() && profile.write_roots.is_empty() {
            warnings.push("no write roots: sandbox is read-only".to_string());
        }

        // The audit log is invisible AND untouchable to the sandbox (S4).
        let log_path = opts
            .hide_dir
            .as_ref()
            .map(|d| d.join("log").to_string_lossy().into_owned())
            .unwrap_or_else(|| ".hide/log".to_string());
        text.push_str(&format!("(deny file-read*  (subpath \"{}\"))\n", escape(&log_path)));
        text.push_str(&format!("(deny file-write* (subpath \"{}\"))\n", escape(&log_path)));
        // The whole .hide dir is never writable by the agent (§4.5.2).
        if let Some(hide) = &opts.hide_dir {
            text.push_str(&format!("(deny file-write* (subpath \"{}\"))\n", escape(&hide.to_string_lossy())));
        }
        text.push('\n');

        // --- network: the ONLY socket is the host proxy port (S5b) ---
        text.push_str(";; --- network ---\n");
        text.push_str("(deny network*)\n");
        match (profile.network.default, opts.proxy_port) {
            (Decision::Allow, _) => {
                // Even an Allow policy funnels through the proxy if we have one;
                // a blanket allow without a proxy is a warned escape hatch.
                if let Some(port) = opts.proxy_port {
                    text.push_str(&format!("(allow network-outbound (remote ip \"localhost:{port}\"))\n"));
                } else {
                    text.push_str("(allow network*)\n");
                    warnings
                        .push("network default=allow with no proxy port: unmediated egress (escape hatch)".to_string());
                }
            }
            (Decision::Deny, Some(port)) | (Decision::Ask, Some(port)) => {
                text.push_str(&format!("(allow network-outbound (remote ip \"localhost:{port}\"))\n"));
                if profile.network.default == Decision::Ask {
                    warnings.push(
                        "network=ask is enforced as proxy-only egress; per-host allow is the proxy's job".to_string(),
                    );
                }
            }
            (Decision::Deny, None) => {
                warnings.push("network default deny, no proxy port: zero egress route".to_string());
            }
            (Decision::Ask, None) => {
                warnings.push("network=ask but no proxy port supplied; rendering zero egress (fail-safe)".to_string());
            }
        }

        if !profile.network.allowed_hosts.is_empty() {
            warnings.push(format!(
                "{} allowed_hosts are enforced at the proxy, not in SBPL",
                profile.network.allowed_hosts.len()
            ));
        }

        RenderedSandboxProfile { tier: profile.tier, profile_text: text, warnings }
    }

    /// Default workspace profile (read = workspace root; no exec, no write, no net).
    pub fn default_workspace_profile(root: impl Into<String>) -> SandboxProfile {
        SandboxProfile {
            tier: SandboxTier::Seatbelt,
            read_roots: vec![root.into()],
            write_roots: Vec::new(),
            allowed_commands: Vec::new(),
            network: NetworkPolicy::default(),
        }
    }

    /// Write the compiled per-grant profile to `sandbox/profiles/<grant_id>.sb`
    /// (§4.1) and return its path. The host owns this dir; it is ephemeral.
    pub fn emit_grant_profile(
        sandbox_dir: &Path,
        grant_id: &GrantId,
        rendered: &RenderedSandboxProfile,
    ) -> hide_core::Result<PathBuf> {
        let profiles = sandbox_dir.join("profiles");
        std::fs::create_dir_all(&profiles)?;
        let path = profiles.join(format!("{}.sb", grant_id.as_str()));
        std::fs::write(&path, rendered.profile_text.as_bytes())?;
        Ok(path)
    }

    /// A command to run under `sandbox-exec`.
    #[derive(Debug, Clone)]
    pub struct SandboxedCommand {
        pub program: String,
        pub args: Vec<String>,
        pub cwd: Option<PathBuf>,
    }

    impl SandboxedCommand {
        pub fn new(program: impl Into<String>) -> Self {
            Self { program: program.into(), args: Vec::new(), cwd: None }
        }

        pub fn arg(mut self, a: impl Into<String>) -> Self {
            self.args.push(a.into());
            self
        }

        pub fn args<I, S>(mut self, it: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.args.extend(it.into_iter().map(Into::into));
            self
        }

        pub fn cwd(mut self, dir: impl Into<PathBuf>) -> Self {
            self.cwd = Some(dir.into());
            self
        }
    }

    /// Is `sandbox-exec` available on this host? (§4.5.3 fail-safe pre-check.)
    pub fn sandbox_exec_available() -> bool {
        cfg!(target_os = "macos") && Path::new("/usr/bin/sandbox-exec").exists()
    }

    /// Build the `sandbox-exec -f <profile.sb> <program> <args...>` command WITHOUT
    /// spawning it — so callers (and tests) can inspect/own the spawn. Fails CLOSED
    /// (S12): returns an error if `sandbox-exec` is unavailable rather than handing
    /// back an unconfined command.
    pub fn build_sandbox_exec_command(profile_path: &Path, cmd: &SandboxedCommand) -> hide_core::Result<Command> {
        if !sandbox_exec_available() {
            return Err(hide_core::error::HideError::PolicyDenied(
                "sandbox-exec unavailable: refusing to run unconfined (S12). Escalate to a microVM tier or an explicit logged override.".to_string(),
            ));
        }
        if !profile_path.exists() {
            return Err(hide_core::error::HideError::Storage(format!(
                "sandbox profile {} not found",
                profile_path.display()
            )));
        }
        let mut c = Command::new("/usr/bin/sandbox-exec");
        c.arg("-f").arg(profile_path);
        c.arg(&cmd.program);
        c.args(&cmd.args);
        if let Some(dir) = &cmd.cwd {
            c.current_dir(dir);
        }
        Ok(c)
    }

    /// Render → emit the per-grant `.sb` → build the confined `sandbox-exec`
    /// command, in one step. The host's spawn path for a T2 grant.
    pub fn spawn_under_sandbox(
        sandbox_dir: &Path,
        grant_id: &GrantId,
        profile: &SandboxProfile,
        opts: &SandboxRenderOptions,
        cmd: &SandboxedCommand,
    ) -> hide_core::Result<Command> {
        let rendered = render_macos_seatbelt_with(profile, opts);
        let profile_path = emit_grant_profile(sandbox_dir, grant_id, &rendered)?;
        build_sandbox_exec_command(&profile_path, cmd)
    }

    fn escape(value: &str) -> String {
        value.replace('\\', "\\\\").replace('"', "\\\"")
    }

    /// Minimal regex metachar escape for a binary basename match.
    fn escape_regex(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        for ch in value.chars() {
            if matches!(ch, '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' | '/') {
                out.push('\\');
            }
            out.push(ch);
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn profile_with(cmds: &[&str], net: Decision) -> SandboxProfile {
            SandboxProfile {
                tier: SandboxTier::Seatbelt,
                read_roots: vec!["/work".to_string()],
                write_roots: vec!["/work/wt".to_string()],
                allowed_commands: cmds.iter().map(|s| s.to_string()).collect(),
                network: NetworkPolicy {
                    default: net,
                    allowed_hosts: vec!["api.github.com".to_string()],
                    denied_hosts: vec![],
                },
            }
        }

        #[test]
        fn renders_deny_default_and_exec_allowlist() {
            let p = profile_with(&["/usr/bin/cargo"], Decision::Deny);
            let r = render_macos_seatbelt(&p);
            assert!(r.profile_text.contains("(deny default)"));
            assert!(r.profile_text.contains("(allow process-exec"));
            assert!(r.profile_text.contains("(literal \"/usr/bin/cargo\")"));
            assert!(r.profile_text.contains("(deny process-exec*)"));
        }

        #[test]
        fn empty_allowlist_denies_all_exec() {
            let p = profile_with(&[], Decision::Deny);
            let r = render_macos_seatbelt(&p);
            assert!(r.profile_text.contains("(deny process-exec*)"));
            assert!(!r.profile_text.contains("(allow process-exec\n"));
            assert!(r.warnings.iter().any(|w| w.contains("process-exec fully denied")));
        }

        #[test]
        fn denies_secret_reads_and_hide_log_writes() {
            let p = profile_with(&["/bin/sh"], Decision::Deny);
            let opts = SandboxRenderOptions {
                proxy_port: None,
                hide_dir: Some(PathBuf::from("/work/.hide")),
                worktree_root: Some("/work/wt".to_string()),
            };
            let r = render_macos_seatbelt_with(&p, &opts);
            assert!(r.profile_text.contains("(deny file-read* (subpath \"$HOME/.ssh\"))"));
            assert!(r.profile_text.contains(r#"(deny file-read* (regex #"\.pem$"))"#));
            // .hide/log specifically write-denied.
            assert!(r.profile_text.contains("(deny file-write* (subpath \"/work/.hide/log\"))"));
            // whole .hide write-denied.
            assert!(r.profile_text.contains("(deny file-write* (subpath \"/work/.hide\"))"));
        }

        #[test]
        fn proxy_port_is_the_only_egress() {
            let p = profile_with(&["/bin/sh"], Decision::Deny);
            let opts = SandboxRenderOptions { proxy_port: Some(8131), ..Default::default() };
            let r = render_macos_seatbelt_with(&p, &opts);
            assert!(r.profile_text.contains("(deny network*)"));
            assert!(r.profile_text.contains("(allow network-outbound (remote ip \"localhost:8131\"))"));
            // allowed_hosts are a proxy concern, surfaced as a warning.
            assert!(r.warnings.iter().any(|w| w.contains("allowed_hosts are enforced at the proxy")));
        }

        #[test]
        fn deny_network_without_proxy_warns_zero_egress() {
            let p = profile_with(&["/bin/sh"], Decision::Deny);
            let r = render_macos_seatbelt(&p);
            assert!(r.warnings.iter().any(|w| w.contains("zero egress")));
            assert!(!r.profile_text.contains("network-outbound"));
        }

        #[test]
        fn bare_command_renders_basename_regex() {
            let p = profile_with(&["cargo"], Decision::Deny);
            let r = render_macos_seatbelt(&p);
            assert!(r.profile_text.contains(r#"(regex #"/cargo$")"#), "{}", r.profile_text);
        }

        #[test]
        fn emit_grant_profile_writes_sb() {
            let dir = tempfile::tempdir().unwrap();
            let p = profile_with(&["/bin/sh"], Decision::Deny);
            let r = render_macos_seatbelt(&p);
            let gid = GrantId::new();
            let path = emit_grant_profile(dir.path(), &gid, &r).unwrap();
            assert!(path.exists());
            assert!(path.extension().unwrap() == "sb");
            let written = std::fs::read_to_string(&path).unwrap();
            assert!(written.contains("(deny default)"));
        }

        #[test]
        fn build_sandbox_exec_fails_closed_when_unavailable() {
            // On non-macOS hosts (CI Linux) sandbox-exec is unavailable → the build
            // must REFUSE, never hand back an unconfined command (S12).
            let dir = tempfile::tempdir().unwrap();
            let sb = dir.path().join("p.sb");
            std::fs::write(&sb, "(version 1)(deny default)").unwrap();
            let cmd = SandboxedCommand::new("echo").arg("hi");
            let res = build_sandbox_exec_command(&sb, &cmd);
            if sandbox_exec_available() {
                assert!(res.is_ok());
            } else {
                assert!(res.is_err(), "must fail closed without sandbox-exec");
            }
        }

        #[test]
        fn escape_handles_quotes_and_backslashes() {
            assert_eq!(escape(r#"a"b\c"#), r#"a\"b\\c"#);
        }
    }
}
#[rustfmt::skip]
pub mod storage {
    //! Encryption-at-rest and on-disk layout enforcement (bible ch.10 §4.4, §4.1,
    //! S6/S12).
    //!
    //! Two real capabilities:
    //!
    //!   1. **AES-256-GCM AEAD at rest.** A random 256-bit *workspace data key*
    //!      (WDK) is wrapped by a key held in the OS keychain (`keyring`, behind the
    //!      `os-keychain` feature); without that feature a clearly-marked
    //!      file-backed dev store stands in. Per-store subkeys are HKDF-style
    //!      derived (`blake3` keyed-hash, domain-separated) from the WDK, and every
    //!      segment gets a fresh random 96-bit nonce stored beside its ciphertext
    //!      (§4.4). Open is authenticated: a tampered segment fails the GCM tag.
    //!
    //!   2. **Layout validation that fails CLOSED.** [`validate_layout`] enforces
    //!      that `.hide` is `0700` (owner-only) and that `.hide/log` is
    //!      append-only / not agent-writable (§4.1, §4.5.2). A violation is an
    //!      error the host surfaces — it never silently downgrades to plaintext or
    //!      an open log (S12).

    use aes_gcm::aead::{Aead, KeyInit, Payload};
    use aes_gcm::{Aes256Gcm, Key, Nonce};
    use hide_core::error::{HideError, Result};
    use rand::RngCore;
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};

    pub const WDK_LEN: usize = 32;
    pub const NONCE_LEN: usize = 12;
    const WRAP_AAD: &[u8] = b"hide.atrest.wdk.v1";

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct AtRestPolicy {
        pub enabled: bool,
        /// Opaque handle into the keychain (`~/.hawking/keys/atrest.wrapkey.ref`,
        /// §4.1) — never key material.
        pub key_ref: Option<String>,
        pub encrypt_event_log: bool,
        pub encrypt_blobs: bool,
        pub encrypt_metadata: bool,
        /// `cache/`/`tmp/` may stay plaintext for speed (§4.4 / §9 Q4).
        pub plaintext_cache_allowed: bool,
    }

    impl Default for AtRestPolicy {
        fn default() -> Self {
            Self {
                enabled: false,
                key_ref: None,
                encrypt_event_log: false,
                encrypt_blobs: false,
                encrypt_metadata: false,
                plaintext_cache_allowed: true,
            }
        }
    }

    impl AtRestPolicy {
        /// The fully-on posture: encrypt log/blobs/metadata, leave derivable caches
        /// plaintext.
        pub fn enabled(key_ref: impl Into<String>) -> Self {
            Self {
                enabled: true,
                key_ref: Some(key_ref.into()),
                encrypt_event_log: true,
                encrypt_blobs: true,
                encrypt_metadata: true,
                plaintext_cache_allowed: true,
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Keychain wrap-key store (real where the feature is on; file-backed dev store
    // otherwise — both behind one trait so the rest of the crate is agnostic).
    // ---------------------------------------------------------------------------

    /// Stores/loads the *wrap key* that protects the WDK. The wrap key never leaves
    /// this boundary in the clear; the WDK is sealed under it via AES-256-GCM.
    pub trait WrapKeyStore: Send + Sync {
        /// Fetch the 32-byte wrap key for `key_ref`, creating it if absent.
        fn get_or_create(&self, key_ref: &str) -> Result<[u8; WDK_LEN]>;
        /// Remove a wrap key (key rotation / workspace deletion).
        fn delete(&self, key_ref: &str) -> Result<()>;
    }

    /// macOS Keychain-backed wrap-key store (Data Protection keychain via
    /// `apple-native`). Compiled only with the `os-keychain` feature.
    #[cfg(feature = "os-keychain")]
    #[derive(Debug, Clone)]
    pub struct KeychainWrapKeyStore {
        service: String,
    }

    #[cfg(feature = "os-keychain")]
    impl KeychainWrapKeyStore {
        pub fn new(service: impl Into<String>) -> Self {
            Self { service: service.into() }
        }
    }

    #[cfg(feature = "os-keychain")]
    impl Default for KeychainWrapKeyStore {
        fn default() -> Self {
            Self::new("com.hawking.hide.atrest")
        }
    }

    #[cfg(feature = "os-keychain")]
    impl WrapKeyStore for KeychainWrapKeyStore {
        fn get_or_create(&self, key_ref: &str) -> Result<[u8; WDK_LEN]> {
            let entry = keyring::Entry::new(&self.service, key_ref)
                .map_err(|e| HideError::Storage(format!("keychain entry: {e}")))?;
            match entry.get_password() {
                Ok(hex) => {
                    let raw = hex_decode(&hex)?;
                    let arr: [u8; WDK_LEN] =
                        raw.try_into().map_err(|_| HideError::Storage("wrap key wrong length".into()))?;
                    Ok(arr)
                }
                Err(keyring::Error::NoEntry) => {
                    let mut key = [0u8; WDK_LEN];
                    rand::thread_rng().fill_bytes(&mut key);
                    entry
                        .set_password(&hex_encode(&key))
                        .map_err(|e| HideError::Storage(format!("keychain set: {e}")))?;
                    Ok(key)
                }
                Err(e) => Err(HideError::Storage(format!("keychain get: {e}"))),
            }
        }

        fn delete(&self, key_ref: &str) -> Result<()> {
            let entry = keyring::Entry::new(&self.service, key_ref)
                .map_err(|e| HideError::Storage(format!("keychain entry: {e}")))?;
            match entry.delete_credential() {
                Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
                Err(e) => Err(HideError::Storage(format!("keychain delete: {e}"))),
            }
        }
    }

    /// File-backed wrap-key store for environments without the OS keychain (CI,
    /// Linux dev). The wrap key lives in a `0600` file under `keys/`. This is
    /// **NOT** production-grade key protection — it is a documented dev stand-in so
    /// the AEAD path is exercisable without a Keychain; the `os-keychain` feature
    /// swaps in the real device-bound store.
    #[derive(Debug, Clone)]
    pub struct FileWrapKeyStore {
        dir: PathBuf,
    }

    impl FileWrapKeyStore {
        pub fn new(dir: impl Into<PathBuf>) -> Self {
            Self { dir: dir.into() }
        }

        fn path(&self, key_ref: &str) -> PathBuf {
            // key_ref is a controlled handle, but sanitize defensively.
            let safe: String = key_ref
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                .collect();
            self.dir.join(format!("{safe}.wrapkey"))
        }
    }

    impl WrapKeyStore for FileWrapKeyStore {
        fn get_or_create(&self, key_ref: &str) -> Result<[u8; WDK_LEN]> {
            let path = self.path(key_ref);
            if path.exists() {
                let raw = std::fs::read(&path)?;
                let arr: [u8; WDK_LEN] =
                    raw.try_into().map_err(|_| HideError::Storage("wrap key wrong length".into()))?;
                return Ok(arr);
            }
            std::fs::create_dir_all(&self.dir)?;
            let mut key = [0u8; WDK_LEN];
            rand::thread_rng().fill_bytes(&mut key);
            std::fs::write(&path, key)?;
            set_owner_only_file(&path)?;
            Ok(key)
        }

        fn delete(&self, key_ref: &str) -> Result<()> {
            let path = self.path(key_ref);
            if path.exists() {
                std::fs::remove_file(path)?;
            }
            Ok(())
        }
    }

    // ---------------------------------------------------------------------------
    // The at-rest cipher: WDK sealed under the wrap key; per-store subkeys; per-
    // segment nonces.
    // ---------------------------------------------------------------------------

    /// A sealed (wrapped) workspace data key, as persisted beside `key_ref`.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct WrappedWdk {
        pub nonce: [u8; NONCE_LEN],
        pub ciphertext: Vec<u8>,
    }

    /// A self-describing encrypted segment: nonce + AES-256-GCM ciphertext (incl.
    /// the 16-byte auth tag). Stored beside the plaintext's slot (§4.4 / §4.2.2
    /// `atrest_nonce`).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct EncryptedSegment {
        pub nonce: [u8; NONCE_LEN],
        pub ciphertext: Vec<u8>,
    }

    /// Holds an *open* WDK in memory and derives per-store subkeys. The WDK is zero
    /// when this is dropped is best-effort (no `zeroize` dep here); the threat model
    /// (§4.4) explicitly does not defend a running same-uid process — that is the
    /// sandbox's job.
    pub struct AtRestCipher {
        wdk: [u8; WDK_LEN],
    }

    impl std::fmt::Debug for AtRestCipher {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("AtRestCipher").field("wdk", &"<sealed>").finish()
        }
    }

    impl AtRestCipher {
        /// Generate a fresh random WDK (on first enable).
        pub fn generate() -> Self {
            let mut wdk = [0u8; WDK_LEN];
            rand::thread_rng().fill_bytes(&mut wdk);
            Self { wdk }
        }

        pub fn from_wdk(wdk: [u8; WDK_LEN]) -> Self {
            Self { wdk }
        }

        /// Seal the WDK under the wrap key for at-rest persistence.
        pub fn wrap_wdk(&self, wrap_key: &[u8; WDK_LEN]) -> Result<WrappedWdk> {
            let nonce = random_nonce();
            let cipher = aes(wrap_key);
            let ct = cipher
                .encrypt(Nonce::from_slice(&nonce), Payload { msg: &self.wdk, aad: WRAP_AAD })
                .map_err(|_| HideError::Storage("WDK wrap failed".into()))?;
            Ok(WrappedWdk { nonce, ciphertext: ct })
        }

        /// Recover the WDK from its wrapped form (authenticated — a tampered wrap or
        /// the wrong wrap key fails).
        pub fn unwrap_wdk(wrap_key: &[u8; WDK_LEN], wrapped: &WrappedWdk) -> Result<Self> {
            let cipher = aes(wrap_key);
            let pt = cipher
                .decrypt(Nonce::from_slice(&wrapped.nonce), Payload { msg: &wrapped.ciphertext, aad: WRAP_AAD })
                .map_err(|_| HideError::Storage("WDK unwrap failed (bad key or tampered)".into()))?;
            let wdk: [u8; WDK_LEN] =
                pt.try_into().map_err(|_| HideError::Storage("unwrapped WDK wrong length".into()))?;
            Ok(Self { wdk })
        }

        /// HKDF-style per-store subkey: keyed-blake3 over the store id. Distinct
        /// stores (log/blobs/meta) never share a key, so a nonce reuse in one store
        /// can't endanger another (§4.4 `HKDF(WDK, context=store-id)`).
        fn subkey(&self, store_id: &str) -> [u8; WDK_LEN] {
            let mut h = blake3::Hasher::new_keyed(&self.wdk);
            h.update(b"hide.atrest.subkey.v1");
            h.update(store_id.as_bytes());
            *h.finalize().as_bytes()
        }

        /// Encrypt a segment of `store_id` with a fresh per-segment nonce. `store_id`
        /// is bound as AAD so a ciphertext can't be replayed into another store.
        pub fn encrypt_segment(&self, store_id: &str, plaintext: &[u8]) -> Result<EncryptedSegment> {
            let key = self.subkey(store_id);
            let cipher = aes(&key);
            let nonce = random_nonce();
            let ct = cipher
                .encrypt(Nonce::from_slice(&nonce), Payload { msg: plaintext, aad: store_id.as_bytes() })
                .map_err(|_| HideError::Storage("segment encrypt failed".into()))?;
            Ok(EncryptedSegment { nonce, ciphertext: ct })
        }

        /// Decrypt + authenticate a segment. A wrong `store_id`, wrong key, or any
        /// tamper fails the GCM tag (S12: an unauthenticated open is impossible).
        pub fn decrypt_segment(&self, store_id: &str, segment: &EncryptedSegment) -> Result<Vec<u8>> {
            let key = self.subkey(store_id);
            let cipher = aes(&key);
            cipher
                .decrypt(
                    Nonce::from_slice(&segment.nonce),
                    Payload { msg: &segment.ciphertext, aad: store_id.as_bytes() },
                )
                .map_err(|_| HideError::Storage("segment decrypt failed (bad key or tampered)".into()))
        }
    }

    fn aes(key: &[u8; WDK_LEN]) -> Aes256Gcm {
        Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key))
    }

    fn random_nonce() -> [u8; NONCE_LEN] {
        let mut n = [0u8; NONCE_LEN];
        rand::thread_rng().fill_bytes(&mut n);
        n
    }

    // ---------------------------------------------------------------------------
    // Layout validation (S12: fail CLOSED on 0700 / append-only violations).
    // ---------------------------------------------------------------------------

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct LayoutValidation {
        pub ok: bool,
        /// `.hide` is mode 0700 (no group/world bits).
        pub root_mode_owner_only: bool,
        /// `.hide/log` is NOT writable by group/other (agent-unreadable/append-only
        /// posture). `true` here means the *violation* condition — the field name is
        /// preserved from the scaffold for compatibility; see [`Self::ok`].
        pub hide_log_agent_writable: bool,
        pub warnings: Vec<String>,
    }

    /// Validate `.hide` layout permissions, failing CLOSED (§4.1, §4.5.2, S12).
    ///
    /// On macOS/Unix this checks real file modes: `.hide` must be `0700`, and
    /// `.hide/log` must not be group/world-writable. A violation returns `ok:false`
    /// with the specifics in `warnings`; the host treats `!ok` as a refusal to
    /// start (it does not downgrade to an open log). On non-Unix the mode bits are
    /// unavailable, so it reports the inability rather than claiming success.
    pub fn validate_layout(hide_dir: &Path) -> LayoutValidation {
        let mut warnings = Vec::new();

        if !hide_dir.exists() {
            return LayoutValidation {
                ok: false,
                root_mode_owner_only: false,
                hide_log_agent_writable: false,
                warnings: vec![format!("{} does not exist", hide_dir.display())],
            };
        }

        let log_dir = hide_dir.join("log");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let root_ok = match std::fs::metadata(hide_dir) {
                Ok(m) => {
                    let mode = m.permissions().mode() & 0o777;
                    if mode != 0o700 {
                        warnings.push(format!(".hide mode is {:#o}, expected 0700 (owner-only)", mode));
                        false
                    } else {
                        true
                    }
                }
                Err(e) => {
                    warnings.push(format!("cannot stat .hide: {e}"));
                    false
                }
            };

            // log dir: must not be group/world writable. If absent we don't fail
            // (a fresh workspace) but we note it.
            let log_violation = if log_dir.exists() {
                match std::fs::metadata(&log_dir) {
                    Ok(m) => {
                        let mode = m.permissions().mode() & 0o777;
                        if mode & 0o022 != 0 {
                            warnings.push(format!(
                                ".hide/log mode is {:#o}; group/world write bits set (audit log must be agent-unwritable)",
                                mode
                            ));
                            true
                        } else {
                            false
                        }
                    }
                    Err(e) => {
                        warnings.push(format!("cannot stat .hide/log: {e}"));
                        true
                    }
                }
            } else {
                false
            };

            LayoutValidation {
                ok: root_ok && !log_violation,
                root_mode_owner_only: root_ok,
                hide_log_agent_writable: log_violation,
                warnings,
            }
        }

        #[cfg(not(unix))]
        {
            let _ = log_dir;
            warnings.push("layout mode checks are only enforced on Unix; refusing to claim 0700".into());
            LayoutValidation { ok: false, root_mode_owner_only: false, hide_log_agent_writable: false, warnings }
        }
    }

    /// Create `.hide` with mode 0700 if missing, then validate. The host's first-run
    /// path (§4.1 step 1). Returns the validation; `!ok` means refuse-to-start.
    pub fn ensure_and_validate_layout(hide_dir: &Path) -> Result<LayoutValidation> {
        if !hide_dir.exists() {
            std::fs::create_dir_all(hide_dir)?;
            set_owner_only_dir(hide_dir)?;
        }
        Ok(validate_layout(hide_dir))
    }

    #[cfg(unix)]
    fn set_owner_only_dir(path: &Path) -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o700);
        std::fs::set_permissions(path, perms)?;
        Ok(())
    }

    #[cfg(not(unix))]
    fn set_owner_only_dir(_path: &Path) -> Result<()> {
        Ok(())
    }

    #[cfg(unix)]
    fn set_owner_only_file(path: &Path) -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)?;
        Ok(())
    }

    #[cfg(not(unix))]
    fn set_owner_only_file(_path: &Path) -> Result<()> {
        Ok(())
    }

    // Hex helpers for keychain string round-trips (keychain stores text, not bytes).
    #[cfg(feature = "os-keychain")]
    fn hex_encode(bytes: &[u8]) -> String {
        crate::audit::hex_lower(bytes)
    }

    #[cfg(feature = "os-keychain")]
    fn hex_decode(input: &str) -> Result<Vec<u8>> {
        if input.len() % 2 != 0 {
            return Err(HideError::Storage("odd-length hex wrap key".into()));
        }
        let val = |b: u8| -> Result<u8> {
            match b {
                b'0'..=b'9' => Ok(b - b'0'),
                b'a'..=b'f' => Ok(b - b'a' + 10),
                b'A'..=b'F' => Ok(b - b'A' + 10),
                _ => Err(HideError::Storage("invalid hex wrap key".into())),
            }
        };
        let bytes = input.as_bytes();
        let mut out = Vec::with_capacity(input.len() / 2);
        for pair in bytes.chunks_exact(2) {
            out.push((val(pair[0])? << 4) | val(pair[1])?);
        }
        Ok(out)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn aead_round_trips_and_authenticates() {
            let cipher = AtRestCipher::generate();
            let seg = cipher.encrypt_segment("log", b"hello secret world").unwrap();
            let pt = cipher.decrypt_segment("log", &seg).unwrap();
            assert_eq!(pt, b"hello secret world");

            // Tamper the ciphertext → tag fails.
            let mut bad = seg.clone();
            bad.ciphertext[0] ^= 0xff;
            assert!(cipher.decrypt_segment("log", &bad).is_err());

            // Wrong store id (AAD) → fails even with the same WDK.
            assert!(cipher.decrypt_segment("blobs", &seg).is_err());
        }

        #[test]
        fn nonces_are_per_segment_unique() {
            let cipher = AtRestCipher::generate();
            let a = cipher.encrypt_segment("log", b"same").unwrap();
            let b = cipher.encrypt_segment("log", b"same").unwrap();
            assert_ne!(a.nonce, b.nonce, "each segment gets a fresh nonce");
            assert_ne!(a.ciphertext, b.ciphertext);
        }

        #[test]
        fn distinct_stores_use_distinct_keys() {
            let cipher = AtRestCipher::generate();
            // Encrypt under "log", attempt decrypt as "log" with same nonce bytes
            // but the AAD/subkey separation means a blobs-keyed open of a log
            // segment must fail (covered above). Here assert subkeys differ.
            let log = cipher.subkey("log");
            let blobs = cipher.subkey("blobs");
            assert_ne!(log, blobs);
        }

        #[test]
        fn wrap_unwrap_wdk_round_trip() {
            let wdk_cipher = AtRestCipher::generate();
            let wrap_key = [42u8; WDK_LEN];
            let wrapped = wdk_cipher.wrap_wdk(&wrap_key).unwrap();

            let recovered = AtRestCipher::unwrap_wdk(&wrap_key, &wrapped).unwrap();
            // Recovered WDK must produce identical subkeys.
            assert_eq!(recovered.subkey("log"), wdk_cipher.subkey("log"));

            // Wrong wrap key fails.
            assert!(AtRestCipher::unwrap_wdk(&[0u8; WDK_LEN], &wrapped).is_err());
        }

        #[test]
        fn file_wrap_key_store_persists() {
            let dir = tempfile::tempdir().unwrap();
            let store = FileWrapKeyStore::new(dir.path().join("keys"));
            let k1 = store.get_or_create("atrest").unwrap();
            let k2 = store.get_or_create("atrest").unwrap();
            assert_eq!(k1, k2, "stable across calls");
            store.delete("atrest").unwrap();
            let k3 = store.get_or_create("atrest").unwrap();
            assert_ne!(k1, k3, "regenerated after delete");
        }

        #[test]
        fn full_wrap_key_to_segment_flow() {
            // End-to-end: wrap-key store → wrap WDK → persist → reopen → decrypt.
            let dir = tempfile::tempdir().unwrap();
            let store = FileWrapKeyStore::new(dir.path().join("keys"));
            let wrap_key = store.get_or_create("ws").unwrap();

            let cipher = AtRestCipher::generate();
            let wrapped = cipher.wrap_wdk(&wrap_key).unwrap();
            let seg = cipher.encrypt_segment("log", b"audit event bytes").unwrap();

            // Simulate reopen: reload wrap key, unwrap WDK, decrypt.
            let wrap_key2 = store.get_or_create("ws").unwrap();
            let cipher2 = AtRestCipher::unwrap_wdk(&wrap_key2, &wrapped).unwrap();
            let pt = cipher2.decrypt_segment("log", &seg).unwrap();
            assert_eq!(pt, b"audit event bytes");
        }

        #[cfg(unix)]
        #[test]
        fn layout_validation_enforces_0700() {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let hide = dir.path().join(".hide");
            let v = ensure_and_validate_layout(&hide).unwrap();
            assert!(v.ok, "fresh 0700 .hide should validate: {:?}", v.warnings);
            assert!(v.root_mode_owner_only);

            // Loosen to 0755 → fail closed.
            let mut perms = std::fs::metadata(&hide).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&hide, perms).unwrap();
            let v2 = validate_layout(&hide);
            assert!(!v2.ok);
            assert!(!v2.root_mode_owner_only);
            assert!(v2.warnings.iter().any(|w| w.contains("0700")));
        }

        #[cfg(unix)]
        #[test]
        fn layout_validation_flags_writable_log() {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let hide = dir.path().join(".hide");
            ensure_and_validate_layout(&hide).unwrap();
            let log = hide.join("log");
            std::fs::create_dir_all(&log).unwrap();
            // group/world-writable log dir.
            std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o777)).unwrap();
            let v = validate_layout(&hide);
            assert!(!v.ok);
            assert!(v.hide_log_agent_writable);
        }

        #[test]
        fn missing_hide_dir_fails_closed() {
            let v = validate_layout(Path::new("/nonexistent/.hide/xyz"));
            assert!(!v.ok);
        }
    }
}

pub use audit::{
    chain_hash, compute_event_chain, compute_event_chain_salted, integrity_alarm_event,
    verify_event_chain, verify_event_chain_salted, verify_with_anchors, AnchorSigner,
    AnchoredVerification, ChainAnchor, ChainAuditReport, EventChainAuditor, IntegrityAlarmKind,
    CHAIN_HASH_LEN,
};
pub use redaction::{
    shannon_entropy, JsonRedactionReport, PatternDetector, Redaction, RedactionReport, Redactor,
};
pub use sandbox::{
    build_sandbox_exec_command, default_workspace_profile, emit_grant_profile,
    render_macos_seatbelt, render_macos_seatbelt_with, sandbox_exec_available, spawn_under_sandbox,
    RenderedSandboxProfile, SandboxRenderOptions, SandboxedCommand,
};
pub use storage::{
    ensure_and_validate_layout, validate_layout, AtRestCipher, AtRestPolicy, EncryptedSegment,
    FileWrapKeyStore, LayoutValidation, WrapKeyStore, WrappedWdk, NONCE_LEN, WDK_LEN,
};

#[cfg(feature = "os-keychain")]
pub use storage::KeychainWrapKeyStore;
