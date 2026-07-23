//! The ACP-session <-> HIDE-(session, thread) binding.
//!
//! ACP has one flat `sessionId`. HIDE's spine is Session -> Thread -> Turn
//! (hide-protocol sec 14). One ACP session binds to one HIDE session and its
//! active thread; `session/new` creates the binding and `session/load` looks it
//! up. This is a pure in-memory registry: it holds no model and mints no ids
//! (the caller supplies HIDE ids from the runtime or a fixture).

use std::collections::HashMap;

use hide_protocol::ids::{SessionId, ThreadId};

use crate::ids::AcpSessionId;

/// The HIDE end of one ACP session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HideBinding {
    pub session: SessionId,
    /// The active thread the ACP session prompts into.
    pub thread: ThreadId,
}

/// A bidirectional registry of ACP-session bindings.
#[derive(Debug, Default, Clone)]
pub struct SessionThreadMap {
    forward: HashMap<AcpSessionId, HideBinding>,
    reverse: HashMap<SessionId, AcpSessionId>,
}

impl SessionThreadMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind an ACP session id to a HIDE (session, thread). Overwrites any prior
    /// binding for the same ACP id.
    pub fn bind(&mut self, acp: AcpSessionId, session: SessionId, thread: ThreadId) {
        self.reverse.insert(session.clone(), acp.clone());
        self.forward.insert(acp, HideBinding { session, thread });
    }

    /// The HIDE binding for an ACP session, if bound.
    pub fn hide_for(&self, acp: &AcpSessionId) -> Option<&HideBinding> {
        self.forward.get(acp)
    }

    /// The ACP session id for a HIDE session, if bound.
    pub fn acp_for(&self, session: &SessionId) -> Option<&AcpSessionId> {
        self.reverse.get(session)
    }

    /// Point an already-bound ACP session at a different active thread (for
    /// example after a `thread/fork`). No-op if the ACP session is unbound.
    pub fn rebind_thread(&mut self, acp: &AcpSessionId, thread: ThreadId) {
        if let Some(b) = self.forward.get_mut(acp) {
            b.thread = thread;
        }
    }

    /// Number of bound sessions.
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }
}
