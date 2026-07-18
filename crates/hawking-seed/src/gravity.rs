//! Gravity as executable law, minimally re-expressed. Exact rational rates; sub-bit-first default;
//! representation escalation before BPW escalation; Doctor bytes inside the same physical budget;
//! escape above one complete BPW requires a sealed Escape Receipt; scheduler deferral and F1 weight
//! reconstruction can NEVER authorize escape.

use crate::{Error, Result};

/// Exact rational rate (num/den). Floats are never scientific identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rate {
    pub num: u32,
    pub den: u32,
}
impl Rate {
    pub fn new(num: u32, den: u32) -> Self {
        assert!(den != 0);
        Rate { num, den }
    }
    /// Whole-artifact BPW below one complete physical bit.
    pub fn is_subbit(&self) -> bool {
        (self.num as u64) < (self.den as u64)
    }
    pub fn value(&self) -> f64 {
        self.num as f64 / self.den as f64
    }
    pub fn label(&self) -> String {
        format!("{}/{}", self.num, self.den)
    }
}

/// What Gravity is being asked to authorize.
#[derive(Debug, Clone)]
pub enum Ask {
    /// Escalate representation family (preferred before raising BPW).
    RepresentationEscalation,
    /// Raise the physical rate above the current sub-bit point.
    BpwEscalation { to: Rate },
    /// Leave the sub-bit region entirely (rate >= 1.0).
    EscapeAboveSubbit { to: Rate, sealed_receipt: bool },
}

/// Evidence Gravity weighs. `f1_only` and `scheduler_deferred` are traps: they can never justify escape.
#[derive(Debug, Clone, Default)]
pub struct Evidence {
    pub representation_families_tried: u32,
    pub doctor_bytes_in_budget: bool,
    pub f1_only: bool,
    pub scheduler_deferred: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub allow: bool,
    pub reason: String,
    pub requires_receipt: bool,
}

/// The whole Gravity policy: a small pure function. Given the current sub-bit rate, the ask, and the
/// evidence, return allow/deny + reason + whether a sealed receipt is required.
pub fn decide(_current: Rate, ask: &Ask, ev: &Evidence) -> Decision {
    match ask {
        Ask::RepresentationEscalation => Decision {
            allow: true,
            reason: "representation escalation precedes BPW escalation".into(),
            requires_receipt: false,
        },
        Ask::BpwEscalation { to } => {
            // must exhaust representation escalation first, and stay sub-bit
            if ev.representation_families_tried < 1 {
                Decision { allow: false, reason: "representation-before-BPW: try a stronger family first".into(), requires_receipt: false }
            } else if !to.is_subbit() {
                Decision { allow: false, reason: "BPW escalation must stay sub-bit; leaving requires an Escape Receipt".into(), requires_receipt: true }
            } else {
                Decision { allow: true, reason: format!("sub-bit BPW escalation to {}", to.label()), requires_receipt: false }
            }
        }
        Ask::EscapeAboveSubbit { to, sealed_receipt } => {
            if to.is_subbit() {
                Decision { allow: false, reason: "not an escape: target is still sub-bit".into(), requires_receipt: false }
            } else if ev.f1_only {
                Decision { allow: false, reason: "F1 weight reconstruction is NOT capability proof; escape denied".into(), requires_receipt: true }
            } else if ev.scheduler_deferred {
                Decision { allow: false, reason: "scheduler deferral is not scientific collapse; escape denied".into(), requires_receipt: true }
            } else if !*sealed_receipt {
                Decision { allow: false, reason: "escape above one complete BPW requires a sealed Escape Receipt".into(), requires_receipt: true }
            } else {
                Decision { allow: true, reason: format!("sealed Escape Receipt authorizes rise to {}", to.label()), requires_receipt: true }
            }
        }
    }
}

/// Physical-byte conservation guard: Doctor bytes count inside the total budget.
pub fn total_bpw(base_bits: u64, doctor_bits: u64, overhead_bits: u64, n_weights: u64) -> f64 {
    (base_bits + doctor_bits + overhead_bits) as f64 / n_weights.max(1) as f64
}

/// Assert Doctor spending stays within the declared physical budget (same-rate treatment law).
pub fn doctor_within_budget(base_bits: u64, doctor_bits: u64, overhead_bits: u64, budget_bpw: f64, n_weights: u64) -> Result<()> {
    let whole = total_bpw(base_bits, doctor_bits, overhead_bits, n_weights);
    if whole <= budget_bpw + 1e-9 {
        Ok(())
    } else {
        Err(Error::Gravity(format!("Doctor bytes exceed budget: {whole:.4} > {budget_bpw:.4} BPW")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subbit_is_the_default_region() {
        assert!(Rate::new(4, 5).is_subbit());
        assert!(!Rate::new(1, 1).is_subbit());
        assert!(!Rate::new(5, 4).is_subbit());
    }

    #[test]
    fn representation_precedes_bpw() {
        let d = decide(Rate::new(4, 5), &Ask::BpwEscalation { to: Rate::new(9, 10) }, &Evidence::default());
        assert!(!d.allow, "no BPW escalation before a representation family is tried");
        let ev = Evidence { representation_families_tried: 2, ..Default::default() };
        assert!(decide(Rate::new(4, 5), &Ask::BpwEscalation { to: Rate::new(9, 10) }, &ev).allow);
    }

    #[test]
    fn escape_requires_sealed_receipt_and_rejects_f1_and_deferral() {
        let to = Rate::new(5, 4);
        // no receipt
        assert!(!decide(Rate::new(4, 5), &Ask::EscapeAboveSubbit { to, sealed_receipt: false }, &Evidence::default()).allow);
        // F1-only can never escape even with a receipt
        let f1 = Evidence { f1_only: true, ..Default::default() };
        assert!(!decide(Rate::new(4, 5), &Ask::EscapeAboveSubbit { to, sealed_receipt: true }, &f1).allow);
        // scheduler deferral can never escape
        let def = Evidence { scheduler_deferred: true, ..Default::default() };
        assert!(!decide(Rate::new(4, 5), &Ask::EscapeAboveSubbit { to, sealed_receipt: true }, &def).allow);
        // clean sealed receipt authorizes
        assert!(decide(Rate::new(4, 5), &Ask::EscapeAboveSubbit { to, sealed_receipt: true }, &Evidence::default()).allow);
    }

    #[test]
    fn doctor_bytes_count_in_the_budget() {
        // base 0.5 bpw + doctor 0.3 bpw over 1000 weights, budget 0.8 → ok; 0.9 → fail
        assert!(doctor_within_budget(500, 300, 0, 0.8, 1000).is_ok());
        assert!(doctor_within_budget(500, 400, 0, 0.8, 1000).is_err());
    }
}
