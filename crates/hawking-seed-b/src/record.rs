//! The one typed envelope for everything: events, state, receipts, source and artifact identities.
//! Canonical serialization + identity + seal + parent linkage in one engine. Reused from Candidate A
//! verbatim — it is already the single canonical-JSON + sha256-seal primitive the whole system uses.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One record. `payload` carries the kind-specific typed data as canonical JSON so a single engine
/// serves every domain; `seal` is the sha256 of the canonical record with the seal field cleared.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub kind: String,
    pub version: u32,
    pub identity: String,
    pub parent: Option<String>,
    pub state: String,
    pub payload: serde_json::Value,
    pub evidence: serde_json::Value,
    #[serde(default)]
    pub seal: String,
}

/// Canonical JSON: sorted keys, compact separators — the single serialization the whole system uses.
pub fn canonical(value: &serde_json::Value) -> String {
    fn sort(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(m) => {
                let mut b = std::collections::BTreeMap::new();
                for (k, val) in m {
                    b.insert(k.clone(), sort(val));
                }
                serde_json::to_value(b).unwrap()
            }
            serde_json::Value::Array(a) => serde_json::Value::Array(a.iter().map(sort).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&sort(value)).unwrap()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

impl Record {
    pub fn new(kind: &str, payload: serde_json::Value) -> Self {
        let identity = sha256_hex(format!("{kind}:{}", canonical(&payload)).as_bytes());
        Record {
            kind: kind.into(),
            version: 1,
            identity,
            parent: None,
            state: "idle".into(),
            payload,
            evidence: serde_json::json!({}),
            seal: String::new(),
        }
    }

    pub fn with_parent(mut self, parent: &str) -> Self {
        self.parent = Some(parent.into());
        self
    }
    pub fn with_state(mut self, s: &str) -> Self {
        self.state = s.into();
        self
    }
    pub fn with_evidence(mut self, e: serde_json::Value) -> Self {
        self.evidence = e;
        self
    }

    /// Compute the seal over the canonical record with `seal` cleared, and store it.
    pub fn sealed(mut self) -> Self {
        self.seal.clear();
        let v = serde_json::to_value(&self).unwrap();
        self.seal = sha256_hex(canonical(&v).as_bytes());
        self
    }

    /// Recompute the seal and compare — tamper detection.
    pub fn verify(&self) -> Result<()> {
        let mut probe = self.clone();
        probe.seal.clear();
        let v = serde_json::to_value(&probe).unwrap();
        let want = sha256_hex(canonical(&v).as_bytes());
        if want == self.seal {
            Ok(())
        } else {
            Err(Error::Seal(format!(
                "seal mismatch for {} ({} != {})",
                self.identity, want, self.seal
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_roundtrip_and_tamper() {
        let r = Record::new("evaluation", serde_json::json!({"parity": true})).sealed();
        assert!(r.verify().is_ok());
        let mut t = r.clone();
        t.payload = serde_json::json!({"parity": false});
        assert!(t.verify().is_err(), "tamper must break the seal");
    }

    #[test]
    fn identity_is_content_addressed_and_canonical() {
        let a = Record::new("x", serde_json::json!({"a": 1, "b": 2}));
        let b = Record::new("x", serde_json::json!({"b": 2, "a": 1}));
        assert_eq!(a.identity, b.identity, "identity is order-independent (canonical)");
    }
}
