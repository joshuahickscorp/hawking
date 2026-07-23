//! Tool-call ACI lint + idempotency (bible ch.02 §4.9).
//!
//! Before a tool call is dispatched the kernel lints it (catch hallucinated
//! tools / malformed args early, the SWE-agent ACI lesson) and deduplicates by
//! idempotency key so a replayed/identical call returns the recorded result
//! rather than re-running the effect (A.3 invariant).

pub mod parse;
pub mod runner;

pub use parse::{has_tool_call, parse_tool_calls, ParsedToolCall};
pub use runner::{CallDispatch, ToolLoop, ToolTurn, ToolTurnStatus};

use hide_core::tool::{ToolCall, ToolResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdempotencyRecord {
    pub key: String,
    pub call_hash: String,
    pub result_event_seq: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDispatchRecord {
    pub call: ToolCall,
    pub result: Option<ToolResult>,
    pub replayed: bool,
}

/// ACI lint result — what the call is missing/wrong before it ever runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LintIssue {
    EmptyToolName,
    UnknownTool(String),
    ArgsNotObject,
    /// An `edit`/`fs` call referencing a path that doesn't exist.
    HallucinatedFile(String),
}

/// Lint a tool call against the set of known tool names + (optionally) the
/// workspace root to catch hallucinated files. Returns the issues found.
pub fn lint_tool_call(
    call: &ToolCall,
    known_tools: &[String],
    workspace_root: Option<&str>,
) -> Vec<LintIssue> {
    let mut issues = Vec::new();
    if call.tool.trim().is_empty() {
        issues.push(LintIssue::EmptyToolName);
        return issues;
    }
    if !known_tools.is_empty() && !known_tools.iter().any(|t| t == &call.tool) {
        issues.push(LintIssue::UnknownTool(call.tool.clone()));
    }
    if !call.args.is_object() {
        issues.push(LintIssue::ArgsNotObject);
        return issues;
    }
    // For edit-shaped tools, a referenced `path` that doesn't exist is almost
    // always a hallucination (unless the tool creates it).
    if let (Some(root), true) = (workspace_root, call.tool.starts_with("edit.")) {
        if let Some(path) = call.args.get("path").and_then(|v| v.as_str()) {
            let full = std::path::Path::new(root).join(path);
            let creates = call.tool == "edit.write_file";
            if !creates && !full.exists() {
                issues.push(LintIssue::HallucinatedFile(path.to_string()));
            }
        }
    }
    issues
}

/// A simple idempotency ledger: keyed by the call's `idempotency_key`, it dedups
/// identical calls so a replay returns the recorded result (K5 / A.3).
#[derive(Default)]
pub struct IdempotencyLedger {
    records: BTreeMap<String, IdempotencyRecord>,
}

impl IdempotencyLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the recorded result-event seq if this key was already executed
    /// with the same call hash (a true dedup), else `None`.
    pub fn lookup(&self, call: &ToolCall) -> Option<u64> {
        let key = call.x.idempotency_key.as_ref()?;
        let rec = self.records.get(key)?;
        if rec.call_hash == call_hash(call) {
            rec.result_event_seq
        } else {
            None
        }
    }

    /// Record an executed call so future identical calls dedup.
    pub fn record(&mut self, call: &ToolCall, result_event_seq: u64) {
        if let Some(key) = &call.x.idempotency_key {
            self.records.insert(
                key.clone(),
                IdempotencyRecord {
                    key: key.clone(),
                    call_hash: call_hash(call),
                    result_event_seq: Some(result_event_seq),
                },
            );
        }
    }
}

/// A stable content hash of a call (tool + args) so an idempotency key only
/// dedups when the *call* is genuinely the same.
fn call_hash(call: &ToolCall) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(call.tool.as_bytes());
    hasher.update(b"\0");
    hasher.update(call.args.to_string().as_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lint_catches_unknown_tool_and_bad_args() {
        let known = vec!["fs.read".to_string()];
        let call = ToolCall::new("nope.tool", json!({}));
        let issues = lint_tool_call(&call, &known, None);
        assert!(issues.contains(&LintIssue::UnknownTool("nope.tool".to_string())));

        let mut bad = ToolCall::new("fs.read", json!([1, 2, 3]));
        bad.args = json!([1, 2, 3]);
        let issues = lint_tool_call(&bad, &known, None);
        assert!(issues.contains(&LintIssue::ArgsNotObject));
    }

    #[test]
    fn idempotency_dedups_identical_call() {
        let mut ledger = IdempotencyLedger::new();
        let mut call = ToolCall::new("shell.run", json!({ "argv": ["true"] }));
        call.x.idempotency_key = Some("k1".to_string());
        assert_eq!(ledger.lookup(&call), None);
        ledger.record(&call, 99);
        assert_eq!(ledger.lookup(&call), Some(99));
    }
}
