use hide_core::event::Event;
use hide_core::persistence::{EventLogIntegrity, IntegrityReport};
use hide_core::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

pub fn compute_event_chain(events: &[Event]) -> ChainAuditReport {
    compute_event_chain_with_verification(events, false)
}

pub fn verify_event_chain(events: &[Event]) -> ChainAuditReport {
    compute_event_chain_with_verification(events, true)
}

fn compute_event_chain_with_verification(
    events: &[Event],
    verify_embedded: bool,
) -> ChainAuditReport {
    let mut prev = vec![0u8; 32];
    let mut records = Vec::new();
    for event in events {
        let mut canonical = event.clone();
        canonical.chain_hash = None;
        let Ok(bytes) = serde_json::to_vec(&canonical) else {
            return ChainAuditReport {
                ok: false,
                records,
                chain_root: None,
                error: Some(format!(
                    "event {} failed canonical serialization",
                    event.seq
                )),
            };
        };
        let mut hasher = Sha256::new();
        hasher.update(&prev);
        hasher.update(bytes);
        let digest = hasher.finalize().to_vec();
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
                            "event {} hash mismatch: embedded {embedded}, computed {}",
                            event.seq, hash
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

fn hex_lower(bytes: &[u8]) -> String {
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
    use hide_core::event::{EventPayload, EventSource, NewEvent};
    use hide_core::ids::SessionId;

    #[test]
    fn verifies_embedded_event_hash_chain() {
        let session = SessionId::new();
        let mut first = hide_core::event::Event::new(
            1,
            NewEvent {
                session_id: session.clone(),
                run_id: None,
                parent: None,
                source: EventSource::System,
                kind: "system.started".into(),
                payload: EventPayload::Custom(serde_json::json!({ "n": 1 })),
                redactions: Vec::new(),
            },
        );
        let mut second = hide_core::event::Event::new(
            2,
            NewEvent {
                session_id: session,
                run_id: None,
                parent: None,
                source: EventSource::System,
                kind: "system.ready".into(),
                payload: EventPayload::Custom(serde_json::json!({ "n": 2 })),
                redactions: Vec::new(),
            },
        );
        first.chain_hash = Some(
            compute_event_chain(&[first.clone()]).records[0]
                .hash
                .clone(),
        );
        second.chain_hash = Some(
            compute_event_chain(&[first.clone(), second.clone()]).records[1]
                .hash
                .clone(),
        );

        let report = verify_event_chain(&[first, second]);
        assert!(report.ok);
        assert!(report.chain_root.is_some());
    }

    #[test]
    fn rejects_tampered_event_hash_chain() {
        let session = SessionId::new();
        let mut event = hide_core::event::Event::new(
            1,
            NewEvent {
                session_id: session,
                run_id: None,
                parent: None,
                source: EventSource::System,
                kind: "system.started".into(),
                payload: EventPayload::Custom(serde_json::json!({ "n": 1 })),
                redactions: Vec::new(),
            },
        );
        event.chain_hash = Some("bad".to_string());

        let report = verify_event_chain(&[event]);
        assert!(!report.ok);
        assert!(report.error.unwrap().contains("hash mismatch"));
    }
}
