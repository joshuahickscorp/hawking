//! Adapter: `hawking_core::engine::StreamEvent` → canonical Event.
//!
//! `StreamEvent` is defined in hawking-core and is the inference hot path.
//! This crate intentionally does **not** depend on hawking-core (Metal, kernels).
//! Instead we accept a minimal view so the mapping is tested without pulling
//! the runtime graph.
//!
//! # Deprecation / authority note
//!
//! `StreamEvent` remains live for token streaming. It is **not** a durable
//! product-event authority. Emitters that need a durable log must project
//! through this adapter (or emit canonical kinds directly).

use hide_core::event::EventClass;
use hide_core::ids::SessionId;
use serde_json::json;

use crate::categories::Category;
use crate::envelope::{CanonicalEvent, ContentVerification, NewCanonical, Subsystem};

/// Minimal view of hawking-core's `StreamEvent` (Token | Done).
///
/// Mirrors `crates/hawking-core/src/engine.rs:188` without importing it.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEventView {
    Token {
        id: u32,
        text: String,
    },
    Done {
        reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        prefill_ms: f64,
        decode_ms: f64,
    },
}

/// Project a token event into a provisional `model.token` canonical event.
pub fn stream_token_to_canonical(
    session_id: SessionId,
    seq: u64,
    stream_id: &str,
    token_id: u32,
    text: &str,
) -> CanonicalEvent {
    CanonicalEvent::sequence(
        seq,
        NewCanonical::new(
            session_id,
            Subsystem::CoreEngine,
            ContentVerification::Provisional,
            Category::Text,
            json!({
                "stream_id": stream_id,
                "token_id": token_id,
                "text": text,
            }),
        )
        .with_class(EventClass::Neither),
    )
}

/// Project a Done event into a provisional `model.usage` canonical event.
pub fn stream_done_to_canonical(
    session_id: SessionId,
    seq: u64,
    view: &StreamEventView,
) -> CanonicalEvent {
    let (reason, prompt_tokens, completion_tokens, prefill_ms, decode_ms) = match view {
        StreamEventView::Done {
            reason,
            prompt_tokens,
            completion_tokens,
            prefill_ms,
            decode_ms,
        } => (
            reason.as_str(),
            *prompt_tokens,
            *completion_tokens,
            *prefill_ms,
            *decode_ms,
        ),
        StreamEventView::Token { .. } => {
            panic!("stream_done_to_canonical requires StreamEventView::Done")
        }
    };
    CanonicalEvent::sequence(
        seq,
        NewCanonical::new(
            session_id,
            Subsystem::CoreEngine,
            ContentVerification::Provisional,
            Category::Usage,
            json!({
                "finish_reason": reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
            }),
        ),
    )
}

/// Project any StreamEventView into one or more canonical events.
pub fn stream_event_to_canonical(
    session_id: SessionId,
    seq: u64,
    stream_id: &str,
    view: &StreamEventView,
) -> CanonicalEvent {
    match view {
        StreamEventView::Token { id, text } => {
            stream_token_to_canonical(session_id, seq, stream_id, *id, text)
        }
        StreamEventView::Done { .. } => stream_done_to_canonical(session_id, seq, view),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::with_deterministic_ids;
    #[test]
    fn token_and_done_project_to_distinct_categories() {
        with_deterministic_ids(10, || {
            let ses = SessionId::from("ses_stream");
            let tok = stream_event_to_canonical(
                ses.clone(),
                1,
                "s0",
                &StreamEventView::Token {
                    id: 42,
                    text: "hi".into(),
                },
            );
            assert_eq!(tok.category, Category::Text);
            assert_eq!(tok.kind(), "model.token");
            assert_eq!(tok.event.payload["text"], "hi");
            let done = stream_event_to_canonical(
                ses,
                2,
                "s0",
                &StreamEventView::Done {
                    reason: "eos".into(),
                    prompt_tokens: 3,
                    completion_tokens: 1,
                    prefill_ms: 1.0,
                    decode_ms: 2.0,
                },
            );
            assert_eq!(done.category, Category::Usage);
            assert_eq!(done.kind(), "model.usage");
            assert_eq!(done.event.payload["completion_tokens"], 1);
        });
    }
}
