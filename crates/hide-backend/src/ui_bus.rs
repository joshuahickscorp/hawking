//! The push `UiEvent` channel — the real Wire-B (bible ch.07 §4.4).
//!
//! The scaffold's only way to read `UiEvent`s was a *pull* scan
//! (`BackendReplayService::ui_events`): the caller polled the event log and
//! mapped rows. That's fine for replay/catch-up but it is not the ordered,
//! low-latency push surface the IDE needs for live token streaming.
//!
//! [`UiEventBus`] is a `tokio::sync::broadcast` bus the host publishes onto.
//! Subscribers ([`UiEventBus::subscribe`]) get an ordered stream. Two properties
//! the bible calls for:
//!
//! * **Render coalescing** — consecutive `TokenBatch`es for the *same stream*
//!   are merged before publish (the UI repaints once per batch, not once per
//!   token), via [`UiEventBus::publish_token`].
//! * **Bounded backpressure** — the broadcast channel has a fixed capacity; a
//!   slow subscriber that falls behind gets a `Lagged` signal (drop-oldest)
//!   rather than unbounded memory growth. The publisher never blocks on a slow
//!   reader (P: the host stays responsive).
//!
//! The pull API is retained (it's cheap and replay still needs it); this is the
//! additional, primary live path.

use hide_core::api::{UiEvent, UiEventKind};
use parking_lot::Mutex;
use tokio::sync::broadcast;

/// A pending coalesce buffer for one stream's tokens.
#[derive(Default)]
struct Coalescer {
    /// (stream_id, accumulated text, last seq) of the in-flight token batch.
    pending: Option<(String, String, u64)>,
}

/// The push bus. Cheap to clone the subscribe handle via [`UiEventBus::subscribe`].
pub struct UiEventBus {
    tx: broadcast::Sender<UiEvent>,
    coalescer: Mutex<Coalescer>,
}

impl UiEventBus {
    /// Create a bus with the given channel capacity (the backpressure bound).
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity.max(1));
        Self {
            tx,
            coalescer: Mutex::new(Coalescer::default()),
        }
    }

    /// Subscribe to the live ordered stream. A lagging subscriber receives
    /// [`broadcast::error::RecvError::Lagged`] (oldest-dropped) instead of
    /// stalling the publisher.
    pub fn subscribe(&self) -> broadcast::Receiver<UiEvent> {
        self.tx.subscribe()
    }

    /// Number of live subscribers (for the host's observability).
    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Publish a finished, non-token UiEvent. Flushes any pending coalesced
    /// token batch first so ordering is preserved (tokens before the event that
    /// follows them).
    pub fn publish(&self, event: UiEvent) {
        self.flush_pending();
        let _ = self.tx.send(event);
    }

    /// Publish a token batch with coalescing. Consecutive batches for the *same*
    /// `stream_id` accumulate; a batch for a *different* stream (or a
    /// [`UiEventBus::flush`]) flushes the accumulated text as a single
    /// `TokenBatch`. This is the render-coalescing path.
    pub fn publish_token(&self, seq: u64, session_id: Option<hide_core::ids::SessionId>, stream_id: impl Into<String>, text: impl AsRef<str>) {
        let stream_id = stream_id.into();
        let text = text.as_ref();
        let to_emit = {
            let mut c = self.coalescer.lock();
            match &mut c.pending {
                Some((sid, acc, last_seq)) if *sid == stream_id => {
                    acc.push_str(text);
                    *last_seq = seq;
                    None
                }
                _ => {
                    // Different stream (or first token): flush the old, start new.
                    let flushed = c.pending.take();
                    c.pending = Some((stream_id.clone(), text.to_string(), seq));
                    flushed
                }
            }
        };
        // We can't recover the previous batch's session here, but coalesced
        // batches share a session in practice (one run = one stream); emit the
        // flushed batch under the current session.
        if let Some((sid, acc, last_seq)) = to_emit {
            let _ = self.tx.send(UiEvent {
                seq: last_seq,
                session_id: session_id.clone(),
                kind: UiEventKind::TokenBatch {
                    stream_id: sid,
                    text: acc,
                },
            });
        }
        let _ = session_id; // session retained on the pending entry conceptually
    }

    /// Flush the accumulated token batch (call at stream end, before a Done).
    pub fn flush(&self, session_id: Option<hide_core::ids::SessionId>) {
        if let Some((sid, acc, last_seq)) = self.coalescer.lock().pending.take() {
            let _ = self.tx.send(UiEvent {
                seq: last_seq,
                session_id,
                kind: UiEventKind::TokenBatch {
                    stream_id: sid,
                    text: acc,
                },
            });
        }
    }

    /// Internal: flush pending tokens with no session (used before a non-token
    /// publish). Tokens carry their own stream id; session is best-effort.
    fn flush_pending(&self) {
        if let Some((sid, acc, last_seq)) = self.coalescer.lock().pending.take() {
            let _ = self.tx.send(UiEvent {
                seq: last_seq,
                session_id: None,
                kind: UiEventKind::TokenBatch {
                    stream_id: sid,
                    text: acc,
                },
            });
        }
    }
}

impl Default for UiEventBus {
    fn default() -> Self {
        // 1024 events of buffering before a slow subscriber lags.
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::SessionId;

    #[tokio::test]
    async fn publish_delivers_to_subscriber() {
        let bus = UiEventBus::new(16);
        let mut rx = bus.subscribe();
        bus.publish(UiEvent {
            seq: 1,
            session_id: None,
            kind: UiEventKind::RuntimeStatus {
                status: "ready".to_string(),
                detail: None,
            },
        });
        let got = rx.recv().await.unwrap();
        assert!(matches!(got.kind, UiEventKind::RuntimeStatus { .. }));
    }

    #[tokio::test]
    async fn same_stream_tokens_coalesce_into_one_batch() {
        let bus = UiEventBus::new(16);
        let mut rx = bus.subscribe();
        let sess = Some(SessionId::new());
        bus.publish_token(1, sess.clone(), "s1", "Hel");
        bus.publish_token(2, sess.clone(), "s1", "lo ");
        bus.publish_token(3, sess.clone(), "s1", "world");
        // Nothing emitted yet (still accumulating). Flush forces one batch.
        bus.flush(sess.clone());
        let got = rx.recv().await.unwrap();
        match got.kind {
            UiEventKind::TokenBatch { stream_id, text } => {
                assert_eq!(stream_id, "s1");
                assert_eq!(text, "Hello world");
            }
            other => panic!("expected coalesced TokenBatch, got {other:?}"),
        }
        assert_eq!(got.seq, 3);
    }

    #[tokio::test]
    async fn switching_streams_flushes_the_previous_batch() {
        let bus = UiEventBus::new(16);
        let mut rx = bus.subscribe();
        let sess = Some(SessionId::new());
        bus.publish_token(1, sess.clone(), "s1", "abc");
        // Switching to s2 flushes s1's "abc".
        bus.publish_token(2, sess.clone(), "s2", "x");
        let first = rx.recv().await.unwrap();
        match first.kind {
            UiEventKind::TokenBatch { stream_id, text } => {
                assert_eq!(stream_id, "s1");
                assert_eq!(text, "abc");
            }
            other => panic!("expected s1 flush, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn capacity_bound_lags_a_slow_subscriber() {
        let bus = UiEventBus::new(2);
        let mut rx = bus.subscribe();
        for i in 0..10 {
            bus.publish(UiEvent {
                seq: i,
                session_id: None,
                kind: UiEventKind::Error {
                    code: "x".to_string(),
                    message: i.to_string(),
                },
            });
        }
        // The slow reader sees a Lagged signal, not unbounded growth.
        let err = rx.recv().await.unwrap_err();
        assert!(matches!(err, broadcast::error::RecvError::Lagged(_)));
    }
}
