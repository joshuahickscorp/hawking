//! Adapter: historical seed-c campaign state-machine `Event` → canonical Event.
//!
//! # Historical note (BC-BRIDGE-012 / B-RT5 product release)
//!
//! The former `hawking_seed_c::state::Event` (`Prepare`/`Admit`/`Run`/…) was a
//! campaign FSM, **not** a product event bus. The `hawking-seed-c` binary and
//! crate were product-released under BC-BRIDGE-012 (B-RT5) and are absent from
//! the live workspace. This adapter retains a hermetic mirror enum only so
//! historical transition projections can still map into `seed.transition`
//! under the model-lifecycle category. Do not grow new product features on it.

use hide_core::event::EventClass;
use hide_core::ids::SessionId;
use serde_json::json;

use crate::categories::Category;
use crate::envelope::{CanonicalEvent, ContentVerification, NewCanonical, Subsystem};

/// Hermetic historical mirror of released `hawking_seed_c::state::Event`.
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

/// Project a historical seed-c FSM transition into a provisional canonical event.
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
