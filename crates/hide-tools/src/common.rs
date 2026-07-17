//! Shared result/error constructors and large-output spill for the builtin catalog.
//!
//! Every tool in this crate produces a [`ToolResult`] through these helpers so the
//! wire-format invariants (ch.03 §4.2) are enforced in one place:
//!
//! * `EXEC_NONZERO` is **data, not failure** — a non-zero process exit is
//!   `ok:true` + `exit_code` (§4.2.3), so the agent can read a failing build.
//! * Over-cap bodies spill to the blob CAS as a `bytes_ref` with a head preview
//!   instead of hard-erroring (TT5 / §4.5).
//! * Errors carry the stable taxonomy code + an optional `fix_hint`/`schema_path`
//!   so the model can self-correct (§4.2.3).

use hide_core::ids::ToolCallId;
use hide_core::persistence::BlobStore;
use hide_core::tool::{ToolContent, ToolError, ToolResult, ToolStats, ToolStatus};
use hide_core::types::{BlobRef, EffectSet};
use serde_json::{json, Value};
use std::sync::Arc;

/// Build a successful result with a structured body and effect set.
pub fn ok(structured: Value, effects: EffectSet) -> ToolResult {
    ToolResult {
        call_id: ToolCallId::new(),
        ok: true,
        status: ToolStatus::Ok,
        content: vec![ToolContent::Json {
            value: structured.clone(),
        }],
        structured_content: Some(structured),
        bytes_ref: None,
        exit_code: None,
        effects,
        provenance: "tool-output".to_string(),
        stats: ToolStats::default(),
        error: None,
    }
}

/// Build a successful result with a text body (used by read/diff/log tools).
pub fn ok_text(text: impl Into<String>, structured: Value, effects: EffectSet) -> ToolResult {
    ToolResult {
        call_id: ToolCallId::new(),
        ok: true,
        status: ToolStatus::Ok,
        content: vec![ToolContent::Text { text: text.into() }],
        structured_content: Some(structured),
        bytes_ref: None,
        exit_code: None,
        effects,
        provenance: "tool-output".to_string(),
        stats: ToolStats::default(),
        error: None,
    }
}

/// The canonical process-shaped result. **A non-zero exit is `ok:true`** — the
/// process *ran*, the diagnostics are data the model must read (§4.2.3). Only a
/// failure to *spawn* the process is `ok:false` ([`spawn_fault`]).
pub fn process_result(exit_code: i32, structured: Value) -> ToolResult {
    ToolResult {
        call_id: ToolCallId::new(),
        ok: true,
        status: ToolStatus::Ok,
        content: vec![ToolContent::Json {
            value: structured.clone(),
        }],
        structured_content: Some(structured),
        bytes_ref: None,
        exit_code: Some(exit_code),
        effects: EffectSet::default(),
        provenance: "tool-output".to_string(),
        stats: ToolStats::default(),
        error: None,
    }
}

/// A process-shaped result whose body is large enough to spill to the blob CAS.
/// `structured` holds the head preview + a `blob_ref` handle; `bytes_ref` carries
/// the CAS reference so the agent can slice the full output later (TT5).
pub fn process_result_spilled(exit_code: i32, structured: Value, bytes_ref: BlobRef) -> ToolResult {
    let mut r = process_result(exit_code, structured);
    r.bytes_ref = Some(bytes_ref);
    r
}

/// Spawn failure (could not even run the process) — this *is* `ok:false`.
pub fn spawn_fault(message: impl Into<String>) -> ToolResult {
    error_result(
        ToolStatus::ToolError,
        ToolError::new("TOOL_FAULT", message, true),
    )
}

/// Argument validation failure (`ARG_INVALID`) with an optional repair hint and
/// JSON-pointer into the offending arg.
pub fn arg_invalid(
    message: impl Into<String>,
    fix_hint: Option<&str>,
    schema_path: Option<&str>,
) -> ToolResult {
    let mut err = ToolError::new("ARG_INVALID", message, true);
    err.fix_hint = fix_hint.map(ToOwned::to_owned);
    err.schema_path = schema_path.map(ToOwned::to_owned);
    error_result(ToolStatus::ProtocolError, err)
}

/// A typed tool error with an explicit taxonomy code (`NOT_FOUND`, `CONFLICT`,
/// `TIMEOUT`, `TOOL_FAULT`, …). `retriable` follows the §4.2.3 taxonomy.
pub fn coded(
    code: &str,
    message: impl Into<String>,
    retriable: bool,
    fix_hint: Option<&str>,
) -> ToolResult {
    let mut err = ToolError::new(code, message, retriable);
    err.fix_hint = fix_hint.map(ToOwned::to_owned);
    let status = match code {
        "TIMEOUT" => ToolStatus::TimedOut,
        "CANCELLED" => ToolStatus::Cancelled,
        _ => ToolStatus::ToolError,
    };
    error_result(status, err)
}

fn error_result(status: ToolStatus, error: ToolError) -> ToolResult {
    ToolResult {
        call_id: ToolCallId::new(),
        ok: false,
        status,
        content: Vec::new(),
        structured_content: None,
        bytes_ref: None,
        exit_code: None,
        effects: EffectSet::default(),
        provenance: "tool-output".to_string(),
        stats: ToolStats::default(),
        error: Some(error),
    }
}

/// How many bytes of a spilled body to keep inline as a preview.
pub const SPILL_PREVIEW_BYTES: usize = 4 * 1024;

/// Decide whether a body fits inline; if not and a blob store is present, write
/// the full body to the CAS and return a `(head_preview, truncated, Some(ref))`.
///
/// Without a blob store the body is hard-truncated to the cap (the legacy
/// behavior), but with one the agent can always recover the whole output.
pub struct Spill {
    pub head: String,
    pub truncated: bool,
    pub total_bytes: usize,
    pub bytes_ref: Option<BlobRef>,
}

pub fn maybe_spill(body: String, cap_bytes: usize, blobs: Option<&Arc<dyn BlobStore>>) -> Spill {
    let total_bytes = body.len();
    if total_bytes <= cap_bytes {
        return Spill {
            head: body,
            truncated: false,
            total_bytes,
            bytes_ref: None,
        };
    }
    // Over cap. Preserve a UTF-8-safe head preview.
    let preview_len = cap_bytes.min(SPILL_PREVIEW_BYTES);
    let head = safe_prefix(&body, preview_len).to_string();
    let bytes_ref = blobs.and_then(|store| {
        store
            .put(body.into_bytes(), Some("text/plain".to_string()))
            .ok()
    });
    Spill {
        head,
        truncated: true,
        total_bytes,
        bytes_ref,
    }
}

/// Largest UTF-8-valid prefix `<= max_bytes`.
pub fn safe_prefix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Project a captured process into the canonical structured body, spilling stdout
/// to the CAS when it exceeds the cap.
pub fn project_process_output(
    exit_code: i32,
    stdout: String,
    stderr: String,
    cap_bytes: usize,
    blobs: Option<&Arc<dyn BlobStore>>,
) -> ToolResult {
    // stderr (diagnostics) is the high-value channel for the agent; keep it whole
    // up to a generous slice, spill stdout which is the bulky channel.
    let stderr_cap = cap_bytes / 4;
    let stderr_view = safe_prefix(&stderr, stderr_cap).to_string();
    let stderr_truncated = stderr.len() > stderr_view.len();
    let spill = maybe_spill(stdout, cap_bytes - stderr_view.len().min(cap_bytes), blobs);

    let mut body = json!({
        "exit_code": exit_code,
        "stdout": spill.head,
        "stderr": stderr_view,
        "stdout_truncated": spill.truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_total_bytes": spill.total_bytes,
    });
    if let Some(ref blob) = spill.bytes_ref {
        body["stdout_blob_ref"] = json!(blob.hash);
    }
    match spill.bytes_ref {
        Some(blob) => process_result_spilled(exit_code, body, blob),
        None => process_result(exit_code, body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::persistence::InMemoryBlobStore;

    #[test]
    fn safe_prefix_respects_char_boundaries() {
        let s = "héllo"; // é is 2 bytes
                         // cutting at byte 2 would split é → must back off to 1
        assert_eq!(safe_prefix(s, 2), "h");
        assert_eq!(safe_prefix(s, 100), s);
    }

    #[test]
    fn maybe_spill_writes_to_cas_over_cap() {
        let store: Arc<dyn BlobStore> = Arc::new(InMemoryBlobStore::default());
        let body = "x".repeat(10_000);
        let spill = maybe_spill(body, 1_000, Some(&store));
        assert!(spill.truncated);
        assert_eq!(spill.total_bytes, 10_000);
        let blob = spill.bytes_ref.expect("spilled to CAS");
        let recovered = store.get(&blob).unwrap().unwrap();
        assert_eq!(recovered.len(), 10_000);
        assert!(spill.head.len() <= SPILL_PREVIEW_BYTES);
    }

    #[test]
    fn under_cap_stays_inline() {
        let spill = maybe_spill("short".to_string(), 1_000, None);
        assert!(!spill.truncated);
        assert!(spill.bytes_ref.is_none());
        assert_eq!(spill.head, "short");
    }

    #[test]
    fn exec_nonzero_is_ok_true() {
        let r = project_process_output(1, "out".into(), "boom".into(), 4096, None);
        assert!(r.ok, "non-zero exit must be ok:true (EXEC_NONZERO is data)");
        assert_eq!(r.exit_code, Some(1));
        assert_eq!(r.status, ToolStatus::Ok);
    }
}
