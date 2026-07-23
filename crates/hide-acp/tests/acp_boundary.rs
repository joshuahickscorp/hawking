//! Deterministic fixture tests for the ACP boundary. No model, no network, no
//! real editor: every test drives the mapping over hand-built HIDE items and
//! ACP messages and asserts the projected shapes and the negotiated
//! capabilities. The ACP wire shapes exercised here are spec-derived from the
//! public Apache-2.0 ACP spec (camelCase fields, the `sessionUpdate` /
//! `type` / `outcome` unions); the assertions below pin those shapes.

use hide_acp::content::ContentBlock;
use hide_acp::handshake::{
    negotiate_protocol_version, AcpClientCapabilities, AcpInitializeRequest, FsCapabilities,
};
use hide_acp::ingest::AcpToHide;
use hide_acp::map::SessionThreadMap;
use hide_acp::project::{AcpOutbound, ProjectHideToAcp};
use hide_acp::session::{AcpPromptRequest, SessionUpdate};
use hide_acp::tool_call::{ToolCallContent, ToolCallStatus, ToolKind};
use hide_acp::{negotiate, AcpError, HideExposure};

use hide_protocol::ids::{ApprovalId, ItemId, SessionId, ThreadId, ToolCallId, ToolId};
use hide_protocol::item::{
    AgentMessage, ApprovalRequest, Item, ItemKind, Patch, ShellStream, ShellChannel,
    ToolCall as HideToolCall,
};
use hide_protocol::model::Risk;
use hide_protocol::plan::Effect;
use hide_protocol::protocol::Method;
use serde_json::json;

// -- fixtures --------------------------------------------------------------

fn item(seq: u64, kind: ItemKind) -> Item {
    Item::new(ItemId::new(format!("itm_{seq}")), seq, kind)
}

fn full_effective() -> hide_acp::EffectiveCapabilities {
    let req = AcpInitializeRequest {
        protocol_version: 1,
        client_capabilities: AcpClientCapabilities {
            fs: FsCapabilities {
                read_text_file: true,
                write_text_file: true,
            },
            terminal: true,
        },
    };
    negotiate(&req, &HideExposure::full_local()).unwrap().effective
}

// -- 1. initialize handshake negotiates capabilities -----------------------

#[test]
fn initialize_handshake_negotiates_capabilities() {
    let req = AcpInitializeRequest {
        protocol_version: 1,
        client_capabilities: AcpClientCapabilities {
            fs: FsCapabilities {
                read_text_file: true,
                write_text_file: true,
            },
            terminal: true,
        },
    };
    let out = negotiate(&req, &HideExposure::full_local()).unwrap();

    // The single negotiated version is echoed.
    assert_eq!(out.response.protocol_version, 1);
    // HIDE advertises session load + embedded context.
    assert!(out.response.agent_capabilities.load_session);
    assert!(out.response.agent_capabilities.prompt_capabilities.embedded_context);
    // A fully capable client leaves every surface effective and nothing degraded.
    assert!(out.effective.terminal);
    assert!(out.effective.edit_apply);
    assert!(out.effective.edit_review);
    assert!(out.degradations.is_empty());
}

#[test]
fn version_negotiation_follows_the_acp_rule() {
    // Agent echoes a supported version.
    assert_eq!(negotiate_protocol_version(1, 1), Some(1));
    // A newer client is met at the agent's latest.
    assert_eq!(negotiate_protocol_version(2, 1), Some(1));
    // Version 0 is unsupported.
    assert_eq!(negotiate_protocol_version(0, 1), None);

    let req = AcpInitializeRequest {
        protocol_version: 0,
        client_capabilities: AcpClientCapabilities::default(),
    };
    let err = negotiate(&req, &HideExposure::full_local()).unwrap_err();
    assert!(matches!(err, AcpError::UnsupportedVersion { offered: 0, .. }));
}

// -- 2. an ACP prompt maps to a hide-protocol turn -------------------------

#[test]
fn acp_prompt_maps_to_hide_turn_intent() {
    let mut map = SessionThreadMap::new();
    map.bind(
        "sess_1".into(),
        SessionId::new("ses_a"),
        ThreadId::new("thr_a"),
    );

    let req = AcpPromptRequest {
        session_id: "sess_1".into(),
        prompt: vec![
            ContentBlock::text("fix the retry bug"),
            ContentBlock::ResourceLink {
                uri: "src/retry.rs".into(),
                name: Some("retry.rs".into()),
                mime_type: None,
            },
        ],
    };

    let intent = AcpToHide::new(&map).map_prompt(&req).unwrap();
    assert_eq!(intent.method, Method::TurnCreate);
    assert_eq!(intent.session, SessionId::new("ses_a"));
    assert_eq!(intent.thread, ThreadId::new("thr_a"));
    assert_eq!(intent.message.text, "fix the retry bug");
    assert_eq!(intent.message.attachments.len(), 1);
    assert_eq!(intent.message.attachments[0].id, "retry.rs");
}

#[test]
fn prompt_for_unknown_session_errors_honestly() {
    let map = SessionThreadMap::new();
    let req = AcpPromptRequest {
        session_id: "nope".into(),
        prompt: vec![ContentBlock::text("hi")],
    };
    let err = AcpToHide::new(&map).map_prompt(&req).unwrap_err();
    assert!(matches!(err, AcpError::UnknownSession(_)));
}

#[test]
fn empty_prompt_errors() {
    let mut map = SessionThreadMap::new();
    map.bind("s".into(), SessionId::new("ses"), ThreadId::new("thr"));
    let req = AcpPromptRequest {
        session_id: "s".into(),
        prompt: vec![],
    };
    let err = AcpToHide::new(&map).map_prompt(&req).unwrap_err();
    assert!(matches!(err, AcpError::EmptyPrompt));
}

// -- 3. an item stream projects to the correct ordered ACP updates ---------

#[test]
fn item_stream_projects_to_ordered_acp_updates() {
    let items = vec![
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
    ];

    let proj = ProjectHideToAcp::new("sess_1".into(), full_effective());
    let out = proj.project_items(&items);

    // Four events, in the same order the items arrived.
    let methods: Vec<&str> = out.iter().map(|o| o.method()).collect();
    assert_eq!(
        methods,
        vec![
            "session/update",
            "session/update",
            "session/update",
            "session/request_permission",
        ]
    );

    // [0] agent_message -> agent_message_chunk
    match &out[0] {
        AcpOutbound::Update(n) => {
            assert_eq!(n.update.tag(), "agent_message_chunk");
            match &n.update {
                SessionUpdate::AgentMessageChunk { content } => {
                    assert_eq!(content.as_text(), Some("on it"))
                }
                other => panic!("expected agent_message_chunk, got {other:?}"),
            }
        }
        other => panic!("expected update, got {other:?}"),
    }

    // [1] tool_call -> tool_call, kind Search, in_progress, raw_input preserved
    match &out[1] {
        AcpOutbound::Update(n) => match &n.update {
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

    // [2] patch -> tool_call kind Edit with a reconstructed diff content
    match &out[2] {
        AcpOutbound::Update(n) => match &n.update {
            SessionUpdate::ToolCall(tc) => {
                assert_eq!(tc.kind, ToolKind::Edit);
                assert_eq!(tc.status, ToolCallStatus::Pending);
                assert_eq!(tc.locations.len(), 1);
                assert_eq!(tc.content.len(), 1);
                match &tc.content[0] {
                    ToolCallContent::Diff {
                        path,
                        old_text,
                        new_text,
                    } => {
                        assert_eq!(path, "src/retry.rs");
                        assert_eq!(old_text.as_deref(), Some("fn retry() {\n    let n = 1;\n}\n"));
                        assert_eq!(new_text, "fn retry() {\n    let n = 3;\n}\n");
                    }
                    other => panic!("expected diff content, got {other:?}"),
                }
            }
            other => panic!("expected tool_call, got {other:?}"),
        },
        other => panic!("expected update, got {other:?}"),
    }

    // [3] approval_request -> session/request_permission with the standard menu
    match &out[3] {
        AcpOutbound::Permission(p) => {
            assert_eq!(p.session_id.as_str(), "sess_1");
            assert_eq!(p.tool_call.tool_call_id.as_str(), "apr_1");
            assert_eq!(p.tool_call.kind, Some(ToolKind::Edit));
            assert_eq!(p.options.len(), 4);
            assert_eq!(p.options[0].option_id, "allow_once");
        }
        other => panic!("expected permission, got {other:?}"),
    }
}

// -- 4. an unsupported capability degrades with an honest response ---------

#[test]
fn unsupported_capability_degrades_honestly() {
    let req = AcpInitializeRequest {
        protocol_version: 1,
        client_capabilities: AcpClientCapabilities {
            fs: FsCapabilities {
                read_text_file: true,
                write_text_file: false, // cannot apply edits
            },
            terminal: false, // no terminal surface
        },
    };
    let out = negotiate(&req, &HideExposure::full_local()).unwrap();

    // The effective set honestly reflects the missing surfaces.
    assert!(!out.effective.terminal);
    assert!(!out.effective.edit_apply);
    // ... but review-only edits still work.
    assert!(out.effective.edit_review);

    // Both downgrades are recorded with a reason and a fallback.
    let caps: Vec<&str> = out.degradations.iter().map(|d| d.capability).collect();
    assert!(caps.contains(&"terminal"));
    assert!(caps.contains(&"edit_apply"));
    assert!(out.degradations.iter().all(|d| !d.reason.is_empty()));

    // The projected shell output degrades from a terminal projection to text.
    let shell = item(
        0,
        ItemKind::ShellStream(ShellStream {
            call_id: Some(ToolCallId::new("call_9")),
            channel: ShellChannel::Stdout,
            chunk: "building...\n".into(),
        }),
    );

    let degraded = ProjectHideToAcp::new("s".into(), out.effective);
    let d_out = degraded.project_item(&shell);
    match &d_out[0] {
        AcpOutbound::Update(n) => match &n.update {
            SessionUpdate::ToolCallUpdate(u) => {
                assert!(
                    u.content
                        .iter()
                        .all(|c| !matches!(c, ToolCallContent::Terminal { .. })),
                    "degraded shell must carry no terminal reference"
                );
                assert!(matches!(u.content[0], ToolCallContent::Content { .. }));
            }
            other => panic!("expected tool_call_update, got {other:?}"),
        },
        other => panic!("expected update, got {other:?}"),
    }

    // With a terminal-capable client the SAME item DOES get a terminal projection.
    let full = ProjectHideToAcp::new("s".into(), full_effective());
    let f_out = full.project_item(&shell);
    match &f_out[0] {
        AcpOutbound::Update(n) => match &n.update {
            SessionUpdate::ToolCallUpdate(u) => {
                assert!(matches!(u.content[0], ToolCallContent::Terminal { .. }));
            }
            other => panic!("expected tool_call_update, got {other:?}"),
        },
        other => panic!("expected update, got {other:?}"),
    }
}
