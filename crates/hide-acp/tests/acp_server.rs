//! Runnable-server tests: drive the ACP [`AcpServer`] end to end over the
//! deterministic in-memory duplex transport. No model, no network, no real
//! editor. Each test queues inbound ACP messages on the client end, runs the
//! server loop to completion, and asserts the ordered outbound ACP messages the
//! server produced. A [`ScriptedTurnHandler`] stands in for the model-bearing
//! turn handler (which is DEFERRED_MODEL_REQUIRED).

use std::io::{BufReader, Cursor};

use hide_acp::handshake::{
    AcpClientCapabilities, AcpInitializeRequest, FsCapabilities,
};
use hide_acp::server::{AcpServer, CountingBinder, ScriptedTurnHandler};
use hide_acp::session::{AcpNewSessionRequest, AcpPromptRequest, StopReason};
use hide_acp::tool_call::{ToolCallStatus, ToolKind};
use hide_acp::transport::{
    memory_duplex, AcpClientMessage, AcpServerMessage, CancelParams, LineTransport, Transport,
};
use hide_acp::{content::ContentBlock, session::SessionUpdate, HideExposure};

use hide_protocol::ids::{ApprovalId, ItemId, ToolCallId, ToolId};
use hide_protocol::item::{
    AgentMessage, ApprovalRequest, Completion, Item, ItemKind, Patch, ToolCall as HideToolCall,
};
use hide_protocol::model::{CompletionStatus, Risk};
use hide_protocol::plan::Effect;
use serde_json::json;

// -- helpers ---------------------------------------------------------------

fn item(seq: u64, kind: ItemKind) -> Item {
    Item::new(ItemId::new(format!("itm_{seq}")), seq, kind)
}

fn full_client() -> AcpClientCapabilities {
    AcpClientCapabilities {
        fs: FsCapabilities {
            read_text_file: true,
            write_text_file: true,
        },
        terminal: true,
    }
}

fn init(caps: AcpClientCapabilities) -> AcpClientMessage {
    AcpClientMessage::Initialize(AcpInitializeRequest {
        protocol_version: 1,
        client_capabilities: caps,
    })
}

fn new_session() -> AcpClientMessage {
    AcpClientMessage::NewSession(AcpNewSessionRequest {
        cwd: "/repo".to_string(),
        mcp_servers: vec![],
    })
}

/// The scripted item stream the mandatory prompt test replays:
/// agent_message, tool_call, patch, approval_request, completion.
fn scripted_turn() -> ScriptedTurnHandler {
    ScriptedTurnHandler::from_items(vec![
        item(
            0,
            ItemKind::AgentMessage(AgentMessage {
                text: "on it".into(),
            }),
        ),
        item(
            1,
            ItemKind::ToolCall(HideToolCall {
                call_id: ToolCallId::new("call_1"),
                tool: ToolId::new("search"),
                arguments: json!({ "q": "retry" }),
            }),
        ),
        item(
            2,
            ItemKind::Patch(Patch {
                patch_id: "pch_1".into(),
                summary: Some("fix retry".into()),
                files: vec!["src/retry.rs".into()],
                unified_diff: concat!(
                    "--- a/src/retry.rs\n",
                    "+++ b/src/retry.rs\n",
                    "@@ -1,3 +1,3 @@\n",
                    " fn retry() {\n",
                    "-    let n = 1;\n",
                    "+    let n = 3;\n",
                    " }\n",
                )
                .into(),
            }),
        ),
        item(
            3,
            ItemKind::ApprovalRequest(ApprovalRequest {
                request_id: ApprovalId::new("apr_1"),
                action: "apply patch".into(),
                risk: Risk::Medium,
                effects: vec![Effect::WriteFs],
                detail: Some("writes src/retry.rs".into()),
            }),
        ),
        item(
            4,
            ItemKind::Completion(Completion {
                status: CompletionStatus::Success,
                summary: Some("done".into()),
            }),
        ),
    ])
}

// -- 1. initialize over the transport replies with capabilities ------------

#[test]
fn initialize_over_transport_replies_with_capabilities() {
    let (client, transport) = memory_duplex();
    client.send_all([init(full_client()), AcpClientMessage::Shutdown]);

    let mut server = AcpServer::new(
        transport,
        ScriptedTurnHandler::default(),
        CountingBinder::default(),
        HideExposure::full_local(),
    );
    server.run().unwrap();

    let out = client.drain();
    assert_eq!(out.len(), 1, "only the initialize result is sent");
    match &out[0] {
        AcpServerMessage::InitializeResult(resp) => {
            assert_eq!(resp.protocol_version, 1);
            assert!(resp.agent_capabilities.load_session);
            assert!(resp.agent_capabilities.prompt_capabilities.embedded_context);
        }
        other => panic!("expected initialize result, got {other:?}"),
    }

    // A fully capable client leaves every surface effective; nothing degraded.
    let eff = server.effective().expect("initialized");
    assert!(eff.terminal && eff.edit_apply && eff.edit_review);
    assert!(server.degradations().is_empty());
}

// -- 2. an unsupported capability degrades honestly ------------------------

#[test]
fn initialize_with_limited_client_degrades_honestly() {
    let limited = AcpClientCapabilities {
        fs: FsCapabilities {
            read_text_file: true,
            write_text_file: false, // cannot apply edits
        },
        terminal: false, // no terminal surface
    };

    let (client, transport) = memory_duplex();
    client.send(init(limited));

    let mut server = AcpServer::new(
        transport,
        ScriptedTurnHandler::default(),
        CountingBinder::default(),
        HideExposure::full_local(),
    );
    server.run().unwrap(); // exits on transport EOF, no explicit shutdown

    // The effective set honestly reflects the missing surfaces.
    let eff = server.effective().expect("initialized");
    assert!(!eff.terminal);
    assert!(!eff.edit_apply);
    assert!(eff.edit_review, "review-only edits still work");

    // Both downgrades are recorded, each with a non-empty reason and a fallback.
    let caps: Vec<&str> = server.degradations().iter().map(|d| d.capability).collect();
    assert!(caps.contains(&"terminal"));
    assert!(caps.contains(&"edit_apply"));
    assert!(server
        .degradations()
        .iter()
        .all(|d| !d.reason.is_empty() && !d.fallback.is_empty()));

    // The client still receives a well-formed initialize result.
    let out = client.drain();
    assert!(matches!(out[0], AcpServerMessage::InitializeResult(_)));
}

// -- 3. a prompt projects the ordered ACP updates, then a turn-complete -----

#[test]
fn prompt_projects_ordered_updates_then_turn_complete() {
    let (client, transport) = memory_duplex();
    // CountingBinder mints the first session as "sess_1".
    client.send_all([
        init(full_client()),
        new_session(),
        AcpClientMessage::Prompt(AcpPromptRequest {
            session_id: "sess_1".into(),
            prompt: vec![ContentBlock::text("fix the retry bug")],
        }),
        AcpClientMessage::Shutdown,
    ]);

    let mut server = AcpServer::new(
        transport,
        scripted_turn(),
        CountingBinder::default(),
        HideExposure::full_local(),
    );
    server.run().unwrap();

    let out = client.drain();
    let methods: Vec<&str> = out.iter().map(|m| m.method()).collect();
    assert_eq!(
        methods,
        vec![
            "initialize",                  // initialize result
            "session/new",                 // new-session result
            "session/update",              // agent_message
            "session/update",              // tool_call
            "session/update",              // patch
            "session/request_permission",  // approval_request
            "session/prompt",              // turn-complete
        ],
        "the prompt yields ordered updates then a turn-complete"
    );

    // The new-session result carries the minted session id the prompt used.
    match &out[1] {
        AcpServerMessage::NewSessionResult(r) => assert_eq!(r.session_id.as_str(), "sess_1"),
        other => panic!("expected new-session result, got {other:?}"),
    }

    // [2] agent_message -> agent_message_chunk "on it".
    match &out[2] {
        AcpServerMessage::Update(n) => match &n.update {
            SessionUpdate::AgentMessageChunk { content } => {
                assert_eq!(content.as_text(), Some("on it"));
            }
            other => panic!("expected agent_message_chunk, got {other:?}"),
        },
        other => panic!("expected update, got {other:?}"),
    }

    // [3] tool_call -> ACP tool_call, kind Search, in_progress, raw_input kept.
    match &out[3] {
        AcpServerMessage::Update(n) => match &n.update {
            SessionUpdate::ToolCall(tc) => {
                assert_eq!(tc.tool_call_id.as_str(), "call_1");
                assert_eq!(tc.kind, ToolKind::Search);
                assert_eq!(tc.status, ToolCallStatus::InProgress);
                assert_eq!(tc.raw_input, Some(json!({ "q": "retry" })));
            }
            other => panic!("expected tool_call, got {other:?}"),
        },
        other => panic!("expected update, got {other:?}"),
    }

    // [4] patch -> ACP tool_call, kind Edit, pending.
    match &out[4] {
        AcpServerMessage::Update(n) => match &n.update {
            SessionUpdate::ToolCall(tc) => {
                assert_eq!(tc.kind, ToolKind::Edit);
                assert_eq!(tc.status, ToolCallStatus::Pending);
            }
            other => panic!("expected tool_call, got {other:?}"),
        },
        other => panic!("expected update, got {other:?}"),
    }

    // [5] approval_request -> session/request_permission with the standard menu,
    // addressed to the same session the prompt named.
    match &out[5] {
        AcpServerMessage::Permission(p) => {
            assert_eq!(p.session_id.as_str(), "sess_1");
            assert_eq!(p.tool_call.tool_call_id.as_str(), "apr_1");
            assert_eq!(p.tool_call.kind, Some(ToolKind::Edit));
            assert_eq!(p.options.len(), 4);
            assert_eq!(p.options[0].option_id, "allow_once");
        }
        other => panic!("expected permission, got {other:?}"),
    }

    // [6] turn-complete carries the projected stop reason.
    match &out[6] {
        AcpServerMessage::PromptResult(r) => assert_eq!(r.stop_reason, StopReason::EndTurn),
        other => panic!("expected prompt result, got {other:?}"),
    }
}

// -- 4. a cancelled completion projects a Cancelled stop reason ------------

#[test]
fn cancelled_completion_maps_to_cancelled_stop_reason() {
    let handler = ScriptedTurnHandler::from_items(vec![
        item(
            0,
            ItemKind::AgentMessage(AgentMessage {
                text: "starting".into(),
            }),
        ),
        item(
            1,
            ItemKind::Completion(Completion {
                status: CompletionStatus::Cancelled,
                summary: None,
            }),
        ),
    ]);

    let (client, transport) = memory_duplex();
    client.send_all([
        init(full_client()),
        new_session(),
        AcpClientMessage::Prompt(AcpPromptRequest {
            session_id: "sess_1".into(),
            prompt: vec![ContentBlock::text("go")],
        }),
        AcpClientMessage::Shutdown,
    ]);

    let mut server = AcpServer::new(
        transport,
        handler,
        CountingBinder::default(),
        HideExposure::full_local(),
    );
    server.run().unwrap();

    let out = client.drain();
    match out.last().unwrap() {
        AcpServerMessage::PromptResult(r) => assert_eq!(r.stop_reason, StopReason::Cancelled),
        other => panic!("expected prompt result, got {other:?}"),
    }
}

// -- 5. an unknown-session prompt is rejected honestly, loop survives -------

#[test]
fn prompt_for_unknown_session_is_rejected_without_stopping_the_loop() {
    let (client, transport) = memory_duplex();
    client.send_all([
        init(full_client()),
        // No session/new, so "ghost" is unbound.
        AcpClientMessage::Prompt(AcpPromptRequest {
            session_id: "ghost".into(),
            prompt: vec![ContentBlock::text("hi")],
        }),
        AcpClientMessage::Shutdown,
    ]);

    let mut server = AcpServer::new(
        transport,
        scripted_turn(),
        CountingBinder::default(),
        HideExposure::full_local(),
    );
    server.run().unwrap();

    let out = client.drain();
    // initialize result, then an error for the unknown session. No prompt result.
    assert_eq!(out.len(), 2);
    match &out[1] {
        AcpServerMessage::Error(e) => assert_eq!(e.code, "prompt_rejected"),
        other => panic!("expected error, got {other:?}"),
    }
    assert!(!out.iter().any(|m| matches!(m, AcpServerMessage::PromptResult(_))));
}

// -- 6. shutdown breaks the loop, leaving later messages unconsumed --------

#[test]
fn shutdown_message_exits_loop_and_leaves_remaining_inbound() {
    let (client, transport) = memory_duplex();
    client.send_all([
        init(full_client()),
        AcpClientMessage::Shutdown,
        // Queued AFTER shutdown: must never be consumed.
        new_session(),
    ]);

    let mut server = AcpServer::new(
        transport,
        ScriptedTurnHandler::default(),
        CountingBinder::default(),
        HideExposure::full_local(),
    );
    server.run().unwrap();

    // Only the initialize result was produced; the post-shutdown message remains.
    let pending_before_drain = client.pending_inbound();
    assert_eq!(pending_before_drain, 1, "the post-shutdown message is untouched");
    let out = client.drain();
    assert_eq!(out.len(), 1);
    assert!(matches!(out[0], AcpServerMessage::InitializeResult(_)));
}

// -- 7. session/cancel is recorded and the loop keeps running --------------

#[test]
fn cancel_is_handled_and_loop_continues() {
    let (client, transport) = memory_duplex();
    client.send_all([
        init(full_client()),
        new_session(),
        AcpClientMessage::Cancel(CancelParams {
            session_id: "sess_1".into(),
        }),
        // The loop must still be alive to serve this prompt after the cancel.
        AcpClientMessage::Prompt(AcpPromptRequest {
            session_id: "sess_1".into(),
            prompt: vec![ContentBlock::text("continue")],
        }),
        AcpClientMessage::Shutdown,
    ]);

    let mut server = AcpServer::new(
        transport,
        scripted_turn(),
        CountingBinder::default(),
        HideExposure::full_local(),
    );
    server.run().unwrap();

    // The cancel was recorded for the right session.
    assert_eq!(server.cancelled().len(), 1);
    assert_eq!(server.cancelled()[0].as_str(), "sess_1");

    // The loop kept running and served the prompt to a turn-complete.
    let out = client.drain();
    assert!(out.iter().any(|m| matches!(m, AcpServerMessage::PromptResult(_))));
}

// -- 8. the line/stdio transport frames messages deterministically ---------

#[test]
fn line_transport_frames_and_parses_a_full_session() {
    // Two newline-delimited inbound messages over an in-memory reader.
    let init_line = serde_json::to_string(&init(full_client())).unwrap();
    let shutdown_line = serde_json::to_string(&AcpClientMessage::Shutdown).unwrap();
    let input = format!("{init_line}\n{shutdown_line}\n");

    let reader = BufReader::new(Cursor::new(input.into_bytes()));
    let output: Vec<u8> = Vec::new();
    let transport = LineTransport::new(reader, output);

    // Run a server over the line transport; capture is via a second construction
    // below, so here just prove it runs clean over framed stdio-style input.
    let mut server = AcpServer::new(
        transport,
        ScriptedTurnHandler::default(),
        CountingBinder::default(),
        HideExposure::full_local(),
    );
    server.run().unwrap();

    // Round-trip the framing directly: a written server message parses back.
    let msg = AcpServerMessage::InitializeResult(
        hide_acp::handshake::AcpInitializeResponse {
            protocol_version: 1,
            agent_capabilities: HideExposure::full_local().agent_capabilities(),
            auth_methods: vec![],
        },
    );
    let mut buf: Vec<u8> = Vec::new();
    {
        let reader = BufReader::new(Cursor::new(Vec::new()));
        let mut t = LineTransport::new(reader, &mut buf);
        t.send(msg.clone()).unwrap();
    }
    let line = String::from_utf8(buf).unwrap();
    assert!(line.ends_with('\n'), "each message is newline-terminated");
    let parsed: AcpServerMessage = serde_json::from_str(line.trim()).unwrap();
    assert_eq!(parsed, msg);
}
