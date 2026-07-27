//! Adapter: seed-c campaign state-machine `Event` → canonical Event.
//!
//! # Deprecation note
//!
//! `hawking_seed_c::state::Event` (`Prepare`/`Admit`/`Run`/…) is a campaign
//! FSM, **not** a product event bus. It stays in place for seed-c; do not grow
//! new product features on it. This adapter projects transitions into
//! `seed.transition` under the model-lifecycle category.
//!
//! This crate does not depend on hawking-seed-c; the enum is mirrored here so
//! the mapping is hermetic. Keep it in lockstep with
//! `crates/hawking-seed-c/src/state.rs:52`.

use hide_core::event::EventClass;
use hide_core::ids::SessionId;
use serde_json::json;

use crate::categories::Category;
use crate::envelope::{
    CanonicalEvent, ContentVerification, NewCanonical, Subsystem,
};

/// Mirror of `hawking_seed_c::state::Event` (file:line anchor in models.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedFsmEvent {
    Prepare,
    Admit,
    Run,
    Evaluate,
    Seal,
    Drain,
    Pause,
    Resume,
    Fail,
}

impl SeedFsmEvent {
    pub fn as_str(self) -> &'static str {
        match self {
            SeedFsmEvent::Prepare => "prepare",
            SeedFsmEvent::Admit => "admit",
            SeedFsmEvent::Run => "run",
            SeedFsmEvent::Evaluate => "evaluate",
            SeedFsmEvent::Seal => "seal",
            SeedFsmEvent::Drain => "drain",
            SeedFsmEvent::Pause => "pause",
            SeedFsmEvent::Resume => "resume",
            SeedFsmEvent::Fail => "fail",
        }
    }
}

/// Project a seed-c FSM transition into a provisional canonical event.
pub fn seed_event_to_canonical(
    session_id: SessionId,
    seq: u64,
    from_state: &str,
    event: SeedFsmEvent,
    to_state: &str,
) -> CanonicalEvent {
    CanonicalEvent::sequence(
        seq,
        NewCanonical::new(
            session_id,
            Subsystem::SeedC,
            ContentVerification::Provisional,
            Category::ModelLifecycle,
            json!({
                "from": from_state,
                "event": event.as_str(),
                "to": to_state,
            }),
        )
        .with_class(EventClass::Neither)
        .with_kind("seed.transition"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::with_deterministic_ids;

    #[test]
    fn seed_transition_is_model_lifecycle() {
        with_deterministic_ids(40, || {
            let c = seed_event_to_canonical(
                SessionId::from("ses_seed"),
                5,
                "idle",
                SeedFsmEvent::Prepare,
                "prepared",
            );
            assert_eq!(c.category, Category::ModelLifecycle);
            assert_eq!(c.kind(), "seed.transition");
            assert_eq!(c.event.payload["event"], "prepare");
            assert_eq!(c.subsystem, Subsystem::SeedC);
        });
    }
}
