//! Durable-thread lifecycle (Stage 4, Group B "Durable thread behavior"; Codex
//! adapted-port of `thread-store/src/store.rs` + `live_thread.rs`).
//!
//! HIDE already owns the durable substrate (the `hide-core` event log + the
//! projection store), so the donor's 20-method `ThreadStore` trait is redundant.
//! What ports is the SMALLEST portable unit the depth map names: the four-verb
//! durability contract over the event-log append path, plus the init-guard RAII.
//!
//! ## The four verbs (over the event-log append path)
//!
//! A [`LiveThread`] is a thin writer that buffers appended items IN MEMORY (they
//! are the "lazy in-memory items") and only makes them durable on an explicit
//! verb:
//!
//! * `flush` - make the buffered items durable and readable (append them to the
//!   event log), keep the writer open.
//! * `persist` - materialize lazy DERIVED state (a durable persist marker), THEN
//!   flush. This is `flush` plus the one thing `flush` does not do.
//! * `shutdown` - flush, then close the writer (no more appends).
//! * `discard` - drop the writer WITHOUT flushing the lazy items. This is the
//!   crucial distinction: a discarded thread leaves NOTHING in the durable log.
//!
//! ## The init guard
//!
//! [`LiveThreadInitGuard`] owns the live thread only while session init is still
//! fallible. If init returns early (the guard is dropped without `commit`), `Drop`
//! discards the writer so a half-initialized session leaves no durable turd.
//! `commit` hands ownership to the running session and neutralizes the drop.
//!
//! Model-free: nothing here runs a model. It is pure event-log plumbing.

use hide_core::event::NewEvent;
use hide_core::ids::SessionId;
use hide_core::persistence::DynEventLog;
use hide_core::Result;
use serde_json::json;

/// The durable event kind a `persist` writes to materialize lazy derived state.
/// A `flush` never writes this; only `persist` does, which is what separates the
/// two verbs on the wire.
pub const THREAD_PERSISTED_KIND: &str = "thread.persisted";

/// A caller-facing durable-thread writer bound to one session's event log.
///
/// Appended items are buffered in memory (lazy) until a durability verb runs.
/// Keeps lifecycle decisions above the store while delegating the actual durable
/// write to the shared [`DynEventLog`].
pub struct LiveThread {
    session: SessionId,
    log: DynEventLog,
    /// Appended-but-not-yet-durable items - the "lazy in-memory items". `discard`
    /// drops these without writing; every other verb writes them.
    pending: Vec<NewEvent>,
    /// Whether there is lazy DERIVED state to materialize on the next `persist`.
    /// Set by `append_item`; cleared by `persist` (which writes the marker).
    lazy_dirty: bool,
    /// Once closed (`shutdown`/`discard`) no further items may be appended.
    closed: bool,
}

impl LiveThread {
    /// Open a live thread over `session`'s durable event log. No durable write
    /// happens until a verb runs.
    pub fn open(session: SessionId, log: DynEventLog) -> Self {
        Self {
            session,
            log,
            pending: Vec::new(),
            lazy_dirty: false,
            closed: false,
        }
    }

    /// The session this thread writes to.
    pub fn session(&self) -> &SessionId {
        &self.session
    }

    /// How many lazy items are buffered (not yet durable).
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Whether the writer is closed (a shutdown or discard has run).
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Buffer an item lazily. It is NOT durable until a `flush`/`persist`/
    /// `shutdown`. Rejected once the writer is closed.
    pub fn append_item(&mut self, event: NewEvent) -> Result<()> {
        if self.closed {
            return Err(hide_core::error::HideError::Message(
                "live thread: append after the writer was closed".to_string(),
            ));
        }
        self.pending.push(event);
        self.lazy_dirty = true;
        Ok(())
    }

    /// Make the buffered items durable and readable: append every pending item to
    /// the event log, in order, and clear the buffer. Returns how many items were
    /// written. Does NOT materialize lazy derived state (that is `persist`).
    pub async fn flush(&mut self) -> Result<usize> {
        let drained: Vec<NewEvent> = std::mem::take(&mut self.pending);
        let count = drained.len();
        for event in drained {
            self.log.append(event).await?;
        }
        Ok(count)
    }

    /// Materialize lazy derived state (write a durable [`THREAD_PERSISTED_KIND`]
    /// marker) and THEN flush the buffered items. This is the one verb that makes
    /// the thread's derived state durable; `flush` alone does not. Returns the
    /// number of buffered items flushed (the marker is not counted).
    pub async fn persist(&mut self) -> Result<usize> {
        if self.closed {
            return Err(hide_core::error::HideError::Message(
                "live thread: persist after the writer was closed".to_string(),
            ));
        }
        // Materialize the lazy derived state as a durable marker BEFORE flushing
        // the items, so a reader that sees the marker also sees the items.
        self.log
            .append(NewEvent::system(
                self.session.clone(),
                THREAD_PERSISTED_KIND,
                json!({ "pending_items": self.pending.len() }),
            ))
            .await?;
        self.lazy_dirty = false;
        self.flush().await
    }

    /// Flush the buffered items, then close the writer. After shutdown no further
    /// items may be appended. Returns the number of items flushed.
    pub async fn shutdown(&mut self) -> Result<usize> {
        let flushed = self.flush().await?;
        self.closed = true;
        Ok(flushed)
    }

    /// Drop the writer WITHOUT flushing the lazy items: the buffered items never
    /// reach the durable log. Idempotent and infallible (there is no durable work
    /// to do, which is the whole point). This is what a failed init calls so a
    /// partial event stream leaves nothing behind.
    pub fn discard(&mut self) {
        self.pending.clear();
        self.lazy_dirty = false;
        self.closed = true;
    }
}

/// RAII owner of a [`LiveThread`] during fallible session init.
///
/// While init can still fail, the guard owns the thread. If the guard is dropped
/// without [`commit`](Self::commit) - the natural consequence of an early return
/// on an init error - `Drop` calls `discard`, so a half-initialized session
/// leaves no durable event stream. `commit` takes the thread out and neutralizes
/// the drop-discard, handing ownership to the now-running session.
pub struct LiveThreadInitGuard {
    thread: Option<LiveThread>,
}

impl LiveThreadInitGuard {
    /// Guard a freshly opened live thread through session bring-up.
    pub fn new(thread: LiveThread) -> Self {
        Self {
            thread: Some(thread),
        }
    }

    /// Borrow the guarded thread during init (append the initial items, etc.).
    /// `None` only after `commit`/`discard` already took it.
    pub fn thread_mut(&mut self) -> Option<&mut LiveThread> {
        self.thread.as_mut()
    }

    /// Init succeeded: take the live thread and neutralize the drop-discard so
    /// the running session owns it. After this the guard's `Drop` is a no-op.
    pub fn commit(mut self) -> LiveThread {
        self.thread
            .take()
            .expect("LiveThreadInitGuard::commit called after the thread was already taken")
    }

    /// Explicitly discard now (identical to letting the guard drop): the partial
    /// event stream is dropped without a flush.
    pub fn discard(mut self) {
        if let Some(mut thread) = self.thread.take() {
            thread.discard();
        }
    }
}

impl Drop for LiveThreadInitGuard {
    fn drop(&mut self) {
        // A guard that reaches Drop still holding the thread means init returned
        // early: discard the partial stream (never flush it).
        if let Some(thread) = self.thread.as_mut() {
            thread.discard();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::{Event, InMemoryEventLog};
    use std::sync::Arc;

    fn log_and_session() -> (DynEventLog, SessionId) {
        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        (log, SessionId::new())
    }

    async fn kinds(log: &DynEventLog, session: &SessionId) -> Vec<String> {
        log.scan(Some(session.clone()), None, None)
            .await
            .unwrap()
            .into_iter()
            .map(|e: Event| e.kind)
            .collect()
    }

    #[tokio::test]
    async fn flush_makes_pending_items_durable() {
        let (log, session) = log_and_session();
        let mut thread = LiveThread::open(session.clone(), log.clone());
        thread
            .append_item(NewEvent::system(session.clone(), "item.a", json!({})))
            .unwrap();
        thread
            .append_item(NewEvent::system(session.clone(), "item.b", json!({})))
            .unwrap();
        assert_eq!(thread.pending_len(), 2);
        let n = thread.flush().await.unwrap();
        assert_eq!(n, 2, "both buffered items were flushed");
        assert_eq!(thread.pending_len(), 0, "the buffer is drained");
        let k = kinds(&log, &session).await;
        assert_eq!(k, vec!["item.a", "item.b"], "items are durable and ordered");
    }

    #[tokio::test]
    async fn discard_does_not_flush_lazy_items() {
        let (log, session) = log_and_session();
        let mut thread = LiveThread::open(session.clone(), log.clone());
        thread
            .append_item(NewEvent::system(session.clone(), "item.a", json!({})))
            .unwrap();
        thread.discard();
        assert!(thread.is_closed(), "discard closes the writer");
        let k = kinds(&log, &session).await;
        assert!(k.is_empty(), "a discarded thread writes NOTHING to the log");
        // Append after close is rejected.
        assert!(thread
            .append_item(NewEvent::system(session.clone(), "item.b", json!({})))
            .is_err());
    }

    #[tokio::test]
    async fn persist_writes_a_marker_that_flush_does_not() {
        let (log, session) = log_and_session();
        // flush: items only, no persist marker.
        let mut a = LiveThread::open(session.clone(), log.clone());
        a.append_item(NewEvent::system(session.clone(), "item.a", json!({})))
            .unwrap();
        a.flush().await.unwrap();
        let after_flush = kinds(&log, &session).await;
        assert!(!after_flush.iter().any(|k| k == THREAD_PERSISTED_KIND));

        // persist: materializes the lazy marker AND flushes.
        let mut b = LiveThread::open(session.clone(), log.clone());
        b.append_item(NewEvent::system(session.clone(), "item.b", json!({})))
            .unwrap();
        b.persist().await.unwrap();
        let after_persist = kinds(&log, &session).await;
        assert!(
            after_persist.iter().any(|k| k == THREAD_PERSISTED_KIND),
            "persist materializes the lazy derived state as a durable marker"
        );
        assert!(after_persist.iter().any(|k| k == "item.b"));
    }

    #[tokio::test]
    async fn shutdown_flushes_then_closes() {
        let (log, session) = log_and_session();
        let mut thread = LiveThread::open(session.clone(), log.clone());
        thread
            .append_item(NewEvent::system(session.clone(), "item.a", json!({})))
            .unwrap();
        let n = thread.shutdown().await.unwrap();
        assert_eq!(n, 1);
        assert!(thread.is_closed());
        assert_eq!(kinds(&log, &session).await, vec!["item.a"]);
    }

    #[tokio::test]
    async fn init_guard_discards_on_early_drop() {
        let (log, session) = log_and_session();
        {
            let mut guard = LiveThreadInitGuard::new(LiveThread::open(session.clone(), log.clone()));
            guard
                .thread_mut()
                .unwrap()
                .append_item(NewEvent::system(session.clone(), "partial.item", json!({})))
                .unwrap();
            // Simulate an early return from a failing init: the guard drops here
            // WITHOUT commit.
        }
        let k = kinds(&log, &session).await;
        assert!(
            k.is_empty(),
            "a failed init discards the partial event stream (nothing durable)"
        );
    }

    #[tokio::test]
    async fn init_guard_commit_hands_off_ownership_and_neutralizes_drop() {
        let (log, session) = log_and_session();
        let mut thread = {
            let mut guard = LiveThreadInitGuard::new(LiveThread::open(session.clone(), log.clone()));
            guard
                .thread_mut()
                .unwrap()
                .append_item(NewEvent::system(session.clone(), "kept.item", json!({})))
                .unwrap();
            // Init succeeded: commit hands the thread to the running session.
            guard.commit()
        };
        // The dropped guard did NOT discard (commit neutralized it): the buffered
        // item is still there to flush.
        assert_eq!(thread.pending_len(), 1);
        thread.flush().await.unwrap();
        assert_eq!(kinds(&log, &session).await, vec!["kept.item"]);
    }
}
