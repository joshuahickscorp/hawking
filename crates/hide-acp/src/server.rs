//! `AcpServer`: a runnable ACP agent server built on the boundary mappings.
//!
//! The server FRAMES and ROUTES an ACP session; it does not run a model. Its run
//! loop reads inbound [`AcpClientMessage`]s from a [`Transport`] and:
//!
//! - on `initialize`, runs the capability [`negotiate`](crate::capability::negotiate)
//!   and replies with the advertised capabilities, recording every honest
//!   [`Degradation`];
//! - on `session/new`, mints a HIDE binding through a [`SessionBinder`] and binds
//!   the ACP session id to it;
//! - on `session/prompt`, maps the prompt to a [`HideTurnIntent`] with
//!   [`AcpToHide`], hands the intent to a pluggable [`TurnHandler`], projects each
//!   yielded HIDE item / notification through [`ProjectHideToAcp`] into ordered
//!   ACP `session/update` and `session/request_permission` outbounds, and closes
//!   the turn with a `session/prompt` result carrying a [`StopReason`];
//! - on `session/cancel`, records the cancellation; on `shutdown` (or transport
//!   EOF), exits the loop cleanly.
//!
//! # What is DEFERRED_MODEL_REQUIRED
//!
//! - The real [`TurnHandler`] that binds to the HIDE backend and executes a turn
//!   with a live model. This crate ships a [`ScriptedTurnHandler`] (fixed item
//!   stream, for tests) and a [`DeferredTurnHandler`] (an honest blocker, for the
//!   bin). Neither runs a model.
//! - Concurrent mid-turn interrupt: because a turn here runs synchronously to
//!   completion, `session/cancel` is recorded but cannot preempt an in-flight
//!   turn. True preemption needs a concurrent model-bearing runtime.
//! - The editor-facing stdio/socket wiring (see [`crate::transport`]).

use hide_protocol::ids::{ItemId, SessionId, ThreadId};
use hide_protocol::item::{Blocker, Completion, Item, ItemKind};
use hide_protocol::model::CompletionStatus;
use hide_protocol::Notification;

use crate::capability::{negotiate, Degradation, EffectiveCapabilities, HideExposure};
use crate::ids::AcpSessionId;
use crate::ingest::{AcpToHide, HideTurnIntent};
use crate::map::SessionThreadMap;
use crate::project::{AcpOutbound, ProjectHideToAcp};
use crate::session::{
    AcpLoadSessionRequest, AcpNewSessionRequest, AcpNewSessionResponse, AcpPromptRequest,
    AcpPromptResponse, StopReason,
};
use crate::transport::{AcpClientMessage, AcpServerMessage, CancelParams, Transport};
use crate::Result;

/// One event a [`TurnHandler`] yields while executing a turn: a HIDE item, or a
/// server notification. Both are projected to ACP by the run loop.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnEvent {
    Item(Item),
    Notification(Notification),
}

impl TurnEvent {
    pub fn item(item: Item) -> Self {
        TurnEvent::Item(item)
    }

    pub fn notification(n: Notification) -> Self {
        TurnEvent::Notification(n)
    }
}

/// Executes one HIDE turn for a mapped prompt, yielding its ordered item stream.
///
/// The real handler binds to the HIDE backend and runs a model; it is
/// DEFERRED_MODEL_REQUIRED. This crate injects the handler so the server stays
/// model-free.
pub trait TurnHandler {
    /// Run the turn for `intent`, returning its ordered events. A streaming
    /// handler would push events as they are produced; the boundary preserves
    /// whatever order it returns.
    fn handle_turn(&mut self, intent: &HideTurnIntent) -> Vec<TurnEvent>;
}

/// A [`TurnHandler`] that replays a fixed, pre-built event stream regardless of
/// the intent. The deterministic test seam: no model, no backend.
#[derive(Debug, Clone, Default)]
pub struct ScriptedTurnHandler {
    events: Vec<TurnEvent>,
}

impl ScriptedTurnHandler {
    /// Build a handler from a pre-ordered list of turn events.
    pub fn new(events: Vec<TurnEvent>) -> Self {
        Self { events }
    }

    /// Build a handler from a list of items (each wrapped as a [`TurnEvent`]).
    pub fn from_items(items: Vec<Item>) -> Self {
        Self {
            events: items.into_iter().map(TurnEvent::Item).collect(),
        }
    }
}

impl TurnHandler for ScriptedTurnHandler {
    fn handle_turn(&mut self, _intent: &HideTurnIntent) -> Vec<TurnEvent> {
        self.events.clone()
    }
}

/// A placeholder [`TurnHandler`] for the bin: it runs no model and yields an
/// honest blocker plus a failed completion, so an editor sees exactly why the
/// turn did not execute. The real backend binding is DEFERRED_MODEL_REQUIRED.
#[derive(Debug, Clone, Default)]
pub struct DeferredTurnHandler;

impl TurnHandler for DeferredTurnHandler {
    fn handle_turn(&mut self, _intent: &HideTurnIntent) -> Vec<TurnEvent> {
        vec![
            TurnEvent::Item(Item::new(
                ItemId::new("blocker_deferred"),
                0,
                ItemKind::Blocker(Blocker {
                    code: "deferred_model_required".to_string(),
                    message: "the HIDE backend is not wired into this build; no turn was run"
                        .to_string(),
                    needs: Some("a model-bearing runtime bound as the TurnHandler".to_string()),
                }),
            )),
            TurnEvent::Item(Item::new(
                ItemId::new("completion_deferred"),
                1,
                ItemKind::Completion(Completion {
                    status: CompletionStatus::Failed,
                    summary: Some("no model runtime bound (DEFERRED_MODEL_REQUIRED)".to_string()),
                }),
            )),
        ]
    }
}

/// The HIDE binding minted for a new ACP session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MintedSession {
    pub acp: AcpSessionId,
    pub session: SessionId,
    pub thread: ThreadId,
}

/// Mints the HIDE (session, thread) binding for a new ACP session. The real
/// binder allocates against the HIDE backend and is DEFERRED_MODEL_REQUIRED;
/// [`CountingBinder`] gives deterministic ids for tests and the bin.
pub trait SessionBinder {
    fn new_session(&mut self, cwd: &str) -> MintedSession;
}

/// A deterministic [`SessionBinder`]: the n-th session is
/// `sess_n` / `ses_n` / `thr_n`.
#[derive(Debug, Clone, Default)]
pub struct CountingBinder {
    n: u64,
}

impl SessionBinder for CountingBinder {
    fn new_session(&mut self, _cwd: &str) -> MintedSession {
        self.n += 1;
        let n = self.n;
        MintedSession {
            acp: AcpSessionId::new(format!("sess_{n}")),
            session: SessionId::new(format!("ses_{n}")),
            thread: ThreadId::new(format!("thr_{n}")),
        }
    }
}

/// A runnable ACP agent server over a transport, a turn handler, and a session
/// binder. Model-free: it frames and routes, and delegates turn execution.
pub struct AcpServer<T: Transport, H: TurnHandler, B: SessionBinder> {
    transport: T,
    handler: H,
    binder: B,
    exposure: HideExposure,
    map: SessionThreadMap,
    effective: Option<EffectiveCapabilities>,
    degradations: Vec<Degradation>,
    cancelled: Vec<AcpSessionId>,
}

impl<T: Transport, H: TurnHandler, B: SessionBinder> AcpServer<T, H, B> {
    /// Build a server. Capabilities are negotiated per client at `initialize`;
    /// `exposure` is HIDE's side of that negotiation.
    pub fn new(transport: T, handler: H, binder: B, exposure: HideExposure) -> Self {
        Self {
            transport,
            handler,
            binder,
            exposure,
            map: SessionThreadMap::new(),
            effective: None,
            degradations: Vec::new(),
            cancelled: Vec::new(),
        }
    }

    /// The effective capabilities negotiated at the last `initialize`, if any.
    pub fn effective(&self) -> Option<EffectiveCapabilities> {
        self.effective
    }

    /// The honest downgrade log from the last `initialize`.
    pub fn degradations(&self) -> &[Degradation] {
        &self.degradations
    }

    /// The ACP sessions a `session/cancel` was received for, in order.
    pub fn cancelled(&self) -> &[AcpSessionId] {
        &self.cancelled
    }

    /// The ACP-session to HIDE-(session, thread) bindings.
    pub fn session_map(&self) -> &SessionThreadMap {
        &self.map
    }

    /// Run the loop until `shutdown` or transport EOF. Returns cleanly either
    /// way; a per-message boundary failure is reported to the client as an
    /// `error` outbound and does not stop the loop.
    pub fn run(&mut self) -> Result<()> {
        while let Some(msg) = self.transport.recv()? {
            match msg {
                AcpClientMessage::Initialize(req) => self.on_initialize(&req)?,
                AcpClientMessage::NewSession(req) => self.on_new_session(&req)?,
                AcpClientMessage::LoadSession(req) => self.on_load_session(&req)?,
                AcpClientMessage::Prompt(req) => self.on_prompt(&req)?,
                AcpClientMessage::Cancel(p) => self.on_cancel(&p)?,
                AcpClientMessage::Shutdown => break,
            }
        }
        Ok(())
    }

    fn on_initialize(&mut self, req: &crate::handshake::AcpInitializeRequest) -> Result<()> {
        match negotiate(req, &self.exposure) {
            Ok(neg) => {
                self.effective = Some(neg.effective);
                self.degradations = neg.degradations;
                self.transport
                    .send(AcpServerMessage::InitializeResult(neg.response))?;
            }
            Err(e) => {
                self.transport
                    .send(AcpServerMessage::error("unsupported_version", e.to_string()))?;
            }
        }
        Ok(())
    }

    fn on_new_session(&mut self, req: &AcpNewSessionRequest) -> Result<()> {
        let minted = self.binder.new_session(&req.cwd);
        self.map
            .bind(minted.acp.clone(), minted.session, minted.thread);
        self.transport
            .send(AcpServerMessage::NewSessionResult(AcpNewSessionResponse {
                session_id: minted.acp,
            }))?;
        Ok(())
    }

    fn on_load_session(&mut self, req: &AcpLoadSessionRequest) -> Result<()> {
        // Bind the resumed ACP session id to a fresh HIDE (session, thread).
        // Replaying the prior thread's items on load is DEFERRED_MODEL_REQUIRED
        // (it needs the persisted turn history from the backend).
        let minted = self.binder.new_session(&req.cwd);
        self.map
            .bind(req.session_id.clone(), minted.session, minted.thread);
        Ok(())
    }

    fn on_prompt(&mut self, req: &AcpPromptRequest) -> Result<()> {
        let effective = match self.effective {
            Some(e) => e,
            None => {
                self.transport.send(AcpServerMessage::error(
                    "not_initialized",
                    "initialize before prompting",
                ))?;
                return Ok(());
            }
        };

        // Map the ACP prompt into a HIDE turn intent. The AcpToHide borrow of the
        // session map ends with this expression, before the handler runs.
        let intent = match AcpToHide::new(&self.map).map_prompt(req) {
            Ok(i) => i,
            Err(e) => {
                self.transport
                    .send(AcpServerMessage::error("prompt_rejected", e.to_string()))?;
                return Ok(());
            }
        };

        let proj = ProjectHideToAcp::new(req.session_id.clone(), effective);
        let events = self.handler.handle_turn(&intent);

        // Project each yielded event, in order, and stream the ACP outbounds.
        for ev in &events {
            for outbound in project_event(&proj, ev) {
                self.transport.send(outbound_to_message(outbound))?;
            }
        }

        // Close the turn with the ACP prompt result (turn-complete).
        let stop_reason = stop_reason_from(&events);
        self.transport
            .send(AcpServerMessage::PromptResult(AcpPromptResponse {
                stop_reason,
            }))?;
        Ok(())
    }

    fn on_cancel(&mut self, p: &CancelParams) -> Result<()> {
        self.cancelled.push(p.session_id.clone());
        Ok(())
    }
}

/// Project one turn event to its ordered ACP outbounds.
fn project_event(proj: &ProjectHideToAcp, ev: &TurnEvent) -> Vec<AcpOutbound> {
    match ev {
        TurnEvent::Item(item) => proj.project_item(item),
        TurnEvent::Notification(n) => proj.project_notification(n),
    }
}

/// Convert a projected outbound into an ACP server message.
fn outbound_to_message(outbound: AcpOutbound) -> AcpServerMessage {
    match outbound {
        AcpOutbound::Update(u) => AcpServerMessage::Update(u),
        AcpOutbound::Permission(p) => AcpServerMessage::Permission(p),
    }
}

/// Derive the ACP stop reason from a turn's events: the last terminal
/// completion (item or notification) decides. A cancelled completion maps to
/// `Cancelled`; every other terminal status ends the turn. With no terminal
/// event the turn is treated as a normal end-of-turn.
fn stop_reason_from(events: &[TurnEvent]) -> StopReason {
    for ev in events.iter().rev() {
        match ev {
            TurnEvent::Item(item) => {
                if let ItemKind::Completion(c) = &item.kind {
                    return stop_reason_for_status(c.status);
                }
            }
            TurnEvent::Notification(Notification::TurnCompleted { status, .. }) => {
                return stop_reason_for_status(*status);
            }
            _ => {}
        }
    }
    StopReason::EndTurn
}

fn stop_reason_for_status(status: CompletionStatus) -> StopReason {
    match status {
        CompletionStatus::Cancelled => StopReason::Cancelled,
        CompletionStatus::Success | CompletionStatus::Partial | CompletionStatus::Failed => {
            StopReason::EndTurn
        }
    }
}
