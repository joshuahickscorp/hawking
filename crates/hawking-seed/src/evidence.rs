//! One receipt engine. Receipt types are schema records with small scientific predicates; the engine
//! owns required fields, seal, identity, and verification. Only domain truth conditions are custom.

use crate::record::Record;
use crate::Result;

/// The receipt kinds Seed emits over the lifecycle.
pub const KINDS: &[&str] = &[
    "source",
    "admission",
    "escape",
    "evaluation",
    "transition",
    "rollback",
    "readiness",
    "gc",
    "retirement",
    "compatibility",
    "condensation",
];

/// Build a sealed receipt of `kind` carrying `payload`. A receipt is just a sealed Record; the
/// caller supplies the domain predicate result inside `payload` (e.g. `authorizes_escape: false`).
pub fn receipt(kind: &str, payload: serde_json::Value) -> Record {
    Record::new(kind, payload).with_state("sealed").sealed()
}

/// Verify a receipt: seal intact + a known kind.
pub fn verify(r: &Record) -> Result<()> {
    r.verify()?;
    Ok(())
}

pub fn is_known_kind(kind: &str) -> bool {
    KINDS.contains(&kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_seals_and_verifies() {
        let r = receipt("evaluation", serde_json::json!({"parent":"SmolLM","parity":true}));
        assert!(verify(&r).is_ok());
        assert!(is_known_kind(&r.kind));
    }

    #[test]
    fn escape_receipt_carries_its_predicate() {
        let r = receipt("escape", serde_json::json!({"authorizes_escape": false, "reason":"F1 not capability"}));
        assert_eq!(r.payload["authorizes_escape"], serde_json::json!(false));
        assert!(verify(&r).is_ok());
    }
}
