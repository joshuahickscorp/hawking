//! Wire-shape tests: pin the spec-derived ACP JSON conventions so a drift is
//! caught here rather than by a real editor. Spec-derived (ACP): camelCase
//! field names, the `sessionUpdate` union on a session update, the `type` union
//! on content and tool-call content, and the `outcome` union on a permission
//! reply. These mirror only the public ACP wire; no proprietary source is used.

use hide_acp::content::ContentBlock;
use hide_acp::permission::{PermissionOutcome, RequestPermissionResponse};
use hide_acp::session::{SessionNotification, SessionUpdate, StopReason};
use hide_acp::tool_call::ToolCallContent;
use hide_acp::unified_diff::{parse_unified_diff, reconstruct_hunk};
use serde_json::{json, Value};

fn to_value<T: serde::Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap()
}

#[test]
fn session_update_uses_session_update_tag_and_camel_case() {
    let n = SessionNotification::new(
        "sess_1".into(),
        SessionUpdate::AgentMessageChunk {
            content: ContentBlock::text("hello"),
        },
    );
    let v = to_value(&n);
    // sessionId is camelCase; the update is tagged on sessionUpdate.
    assert_eq!(v["sessionId"], json!("sess_1"));
    assert_eq!(v["update"]["sessionUpdate"], json!("agent_message_chunk"));
    assert_eq!(v["update"]["content"]["type"], json!("text"));
    assert_eq!(v["update"]["content"]["text"], json!("hello"));

    // Round trip is lossless.
    let back: SessionNotification = serde_json::from_value(v).unwrap();
    assert_eq!(back, n);
}

#[test]
fn diff_content_uses_type_diff_and_camel_case_text_fields() {
    let c = ToolCallContent::Diff {
        path: "src/a.rs".into(),
        old_text: Some("a\n".into()),
        new_text: "b\n".into(),
    };
    let v = to_value(&c);
    assert_eq!(v["type"], json!("diff"));
    assert_eq!(v["path"], json!("src/a.rs"));
    assert_eq!(v["oldText"], json!("a\n"));
    assert_eq!(v["newText"], json!("b\n"));

    let back: ToolCallContent = serde_json::from_value(v).unwrap();
    assert_eq!(back, c);
}

#[test]
fn terminal_content_uses_camel_case_terminal_id() {
    let c = ToolCallContent::Terminal {
        terminal_id: "term_1".into(),
    };
    let v = to_value(&c);
    assert_eq!(v["type"], json!("terminal"));
    assert_eq!(v["terminalId"], json!("term_1"));
}

#[test]
fn permission_outcome_is_tagged_on_outcome() {
    let selected = RequestPermissionResponse {
        outcome: PermissionOutcome::Selected {
            option_id: "allow_once".into(),
        },
    };
    let v = to_value(&selected);
    assert_eq!(v["outcome"]["outcome"], json!("selected"));
    assert_eq!(v["outcome"]["optionId"], json!("allow_once"));

    let cancelled = RequestPermissionResponse {
        outcome: PermissionOutcome::Cancelled,
    };
    assert_eq!(to_value(&cancelled)["outcome"]["outcome"], json!("cancelled"));

    // Round trips.
    let back: RequestPermissionResponse = serde_json::from_value(v).unwrap();
    assert_eq!(back, selected);
}

#[test]
fn stop_reason_serializes_snake_case() {
    assert_eq!(to_value(&StopReason::EndTurn), json!("end_turn"));
    assert_eq!(to_value(&StopReason::MaxTurnRequests), json!("max_turn_requests"));
}

#[test]
fn unified_diff_reconstructs_single_hunk_exactly() {
    let files = parse_unified_diff(concat!(
        "--- a/x.rs\n",
        "+++ b/x.rs\n",
        "@@ -1,2 +1,2 @@\n",
        " keep\n",
        "-old\n",
        "+new\n",
    ));
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].path, "x.rs");
    assert_eq!(files[0].old_text.as_deref(), Some("keep\nold\n"));
    assert_eq!(files[0].new_text, "keep\nnew\n");
}

#[test]
fn unified_diff_marks_new_file_old_text_none() {
    let files = parse_unified_diff(concat!(
        "--- /dev/null\n",
        "+++ b/new.rs\n",
        "@@ -0,0 +1,1 @@\n",
        "+fn main() {}\n",
    ));
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].path, "new.rs");
    assert_eq!(files[0].old_text, None);
    assert_eq!(files[0].new_text, "fn main() {}\n");
}

#[test]
fn unified_diff_parses_multiple_files() {
    let files = parse_unified_diff(concat!(
        "diff --git a/one.rs b/one.rs\n",
        "--- a/one.rs\n",
        "+++ b/one.rs\n",
        "@@ -1 +1 @@\n",
        "-1\n",
        "+2\n",
        "diff --git a/two.rs b/two.rs\n",
        "--- a/two.rs\n",
        "+++ b/two.rs\n",
        "@@ -1 +1 @@\n",
        "-a\n",
        "+b\n",
    ));
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].path, "one.rs");
    assert_eq!(files[1].path, "two.rs");
    assert_eq!(files[1].new_text, "b\n");
}

#[test]
fn reconstruct_hunk_splits_added_and_removed() {
    let (old, new) = reconstruct_hunk(" ctx\n-gone\n+added\n ctx2\n");
    assert_eq!(old, "ctx\ngone\nctx2\n");
    assert_eq!(new, "ctx\nadded\nctx2\n");
}
