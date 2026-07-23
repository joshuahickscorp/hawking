//! The ACP message envelope and the transport that carries it.
//!
//! The [`AcpServer`](crate::server::AcpServer) run loop reads inbound
//! [`AcpClientMessage`]s and writes outbound [`AcpServerMessage`]s through a
//! [`Transport`]. Both message enums are adjacently tagged on a slash-namespaced
//! `method` (spec-derived, matching the ACP method names) so a line of wire is a
//! single flat JSON object; only the public wire convention is mirrored.
//!
//! Two transports live here:
//!
//! - [`memory_duplex`] returns two connected ends for deterministic tests: the
//!   server holds a [`MemoryTransport`] and the test drives the [`MemoryClient`]
//!   end. No thread, no socket, no model.
//! - [`LineTransport`] frames each message as one newline-delimited JSON object
//!   over any [`BufRead`] reader and [`Write`] writer. It is deterministic and
//!   testable over in-memory buffers.
//!
//! The real editor-facing stdio/socket wiring (spawning under a live editor,
//! streaming real bytes) is DEFERRED_MODEL_REQUIRED; the model-free line
//! transport is the deterministic seam a host reuses over the process stdio.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::handshake::{AcpInitializeRequest, AcpInitializeResponse};
use crate::ids::AcpSessionId;
use crate::permission::RequestPermissionRequest;
use crate::session::{
    AcpLoadSessionRequest, AcpNewSessionRequest, AcpNewSessionResponse, AcpPromptRequest,
    AcpPromptResponse, SessionNotification,
};

/// `session/cancel` params: interrupt the active turn of a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelParams {
    pub session_id: AcpSessionId,
}

/// A boundary-level error message the server sends when it cannot honor an
/// inbound message (unknown session, empty prompt, prompt before initialize).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerError {
    pub code: String,
    pub message: String,
}

/// An inbound ACP message (client -> agent). Adjacently tagged on `method`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum AcpClientMessage {
    #[serde(rename = "initialize")]
    Initialize(AcpInitializeRequest),
    #[serde(rename = "session/new")]
    NewSession(AcpNewSessionRequest),
    #[serde(rename = "session/load")]
    LoadSession(AcpLoadSessionRequest),
    #[serde(rename = "session/prompt")]
    Prompt(AcpPromptRequest),
    #[serde(rename = "session/cancel")]
    Cancel(CancelParams),
    /// End the run loop and disconnect. Not a spec ACP method; the boundary
    /// treats an EOF on the transport the same way.
    #[serde(rename = "shutdown")]
    Shutdown,
}

impl AcpClientMessage {
    /// The wire `method` tag for this inbound message.
    pub fn method(&self) -> &'static str {
        match self {
            AcpClientMessage::Initialize(_) => "initialize",
            AcpClientMessage::NewSession(_) => "session/new",
            AcpClientMessage::LoadSession(_) => "session/load",
            AcpClientMessage::Prompt(_) => "session/prompt",
            AcpClientMessage::Cancel(_) => "session/cancel",
            AcpClientMessage::Shutdown => "shutdown",
        }
    }
}

/// An outbound ACP message (agent -> client). Adjacently tagged on `method`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum AcpServerMessage {
    /// The `initialize` result: negotiated version and advertised capabilities.
    #[serde(rename = "initialize")]
    InitializeResult(AcpInitializeResponse),
    /// The `session/new` result: the minted session id.
    #[serde(rename = "session/new")]
    NewSessionResult(AcpNewSessionResponse),
    /// A `session/update` notification streamed during a turn.
    #[serde(rename = "session/update")]
    Update(SessionNotification),
    /// A `session/request_permission` request for an effectful action.
    #[serde(rename = "session/request_permission")]
    Permission(RequestPermissionRequest),
    /// The `session/prompt` result: the turn stopped, with a stop reason.
    #[serde(rename = "session/prompt")]
    PromptResult(AcpPromptResponse),
    /// A boundary error the client can surface.
    #[serde(rename = "error")]
    Error(ServerError),
}

impl AcpServerMessage {
    /// The wire `method` tag for this outbound message.
    pub fn method(&self) -> &'static str {
        match self {
            AcpServerMessage::InitializeResult(_) => "initialize",
            AcpServerMessage::NewSessionResult(_) => "session/new",
            AcpServerMessage::Update(_) => "session/update",
            AcpServerMessage::Permission(_) => "session/request_permission",
            AcpServerMessage::PromptResult(_) => "session/prompt",
            AcpServerMessage::Error(_) => "error",
        }
    }

    /// Build an error outbound with a stable code and a message.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        AcpServerMessage::Error(ServerError {
            code: code.into(),
            message: message.into(),
        })
    }
}

/// Carries ACP messages in and out of the [`AcpServer`](crate::server::AcpServer)
/// run loop, from the server's point of view: send an outbound message, and
/// receive the next inbound one.
pub trait Transport {
    /// Send one outbound ACP message toward the client.
    fn send(&mut self, msg: AcpServerMessage) -> Result<()>;

    /// Receive the next inbound ACP message. Returns `Ok(None)` at end of
    /// stream (the client disconnected), which the run loop treats as a clean
    /// exit.
    fn recv(&mut self) -> Result<Option<AcpClientMessage>>;
}

// -- in-memory duplex ------------------------------------------------------

/// The shared queues behind one connected pair. `to_server` carries client ->
/// agent; `to_client` carries agent -> client.
#[derive(Default)]
struct Channel {
    to_server: VecDeque<AcpClientMessage>,
    to_client: VecDeque<AcpServerMessage>,
}

/// The server end of an in-memory duplex. Implements [`Transport`].
#[derive(Clone)]
pub struct MemoryTransport {
    chan: Rc<RefCell<Channel>>,
}

/// The client (test-driver) end of an in-memory duplex.
#[derive(Clone)]
pub struct MemoryClient {
    chan: Rc<RefCell<Channel>>,
}

/// Build a connected in-memory pair: the client end the test drives, and the
/// server end the run loop reads and writes.
pub fn memory_duplex() -> (MemoryClient, MemoryTransport) {
    let chan = Rc::new(RefCell::new(Channel::default()));
    (
        MemoryClient { chan: chan.clone() },
        MemoryTransport { chan },
    )
}

impl Transport for MemoryTransport {
    fn send(&mut self, msg: AcpServerMessage) -> Result<()> {
        self.chan.borrow_mut().to_client.push_back(msg);
        Ok(())
    }

    fn recv(&mut self) -> Result<Option<AcpClientMessage>> {
        Ok(self.chan.borrow_mut().to_server.pop_front())
    }
}

impl MemoryClient {
    /// Queue one inbound message toward the server.
    pub fn send(&self, msg: AcpClientMessage) {
        self.chan.borrow_mut().to_server.push_back(msg);
    }

    /// Queue several inbound messages, in order.
    pub fn send_all(&self, msgs: impl IntoIterator<Item = AcpClientMessage>) {
        let mut chan = self.chan.borrow_mut();
        for m in msgs {
            chan.to_server.push_back(m);
        }
    }

    /// Pop the next server message, if any.
    pub fn recv(&self) -> Option<AcpServerMessage> {
        self.chan.borrow_mut().to_client.pop_front()
    }

    /// Drain every server message produced so far, in order.
    pub fn drain(&self) -> Vec<AcpServerMessage> {
        self.chan.borrow_mut().to_client.drain(..).collect()
    }

    /// How many inbound messages the server has not yet consumed.
    pub fn pending_inbound(&self) -> usize {
        self.chan.borrow().to_server.len()
    }
}

// -- newline-delimited line transport --------------------------------------

/// A transport that frames each ACP message as one line of JSON over a reader
/// and a writer. Deterministic and testable over in-memory buffers; the bin
/// runs it over process stdio.
pub struct LineTransport<R: BufRead, W: Write> {
    reader: R,
    writer: W,
}

impl<R: BufRead, W: Write> LineTransport<R, W> {
    pub fn new(reader: R, writer: W) -> Self {
        Self { reader, writer }
    }
}

impl<R: BufRead, W: Write> Transport for LineTransport<R, W> {
    fn send(&mut self, msg: AcpServerMessage) -> Result<()> {
        let line = serde_json::to_string(&msg)?;
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }

    fn recv(&mut self) -> Result<Option<AcpClientMessage>> {
        loop {
            let mut line = String::new();
            let read = self.reader.read_line(&mut line)?;
            if read == 0 {
                return Ok(None); // EOF: the client disconnected.
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue; // tolerate blank framing lines
            }
            let msg = serde_json::from_str(trimmed)?;
            return Ok(Some(msg));
        }
    }
}
