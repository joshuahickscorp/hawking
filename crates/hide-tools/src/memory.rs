//! `memory.*` — a client-side, cross-session memory tool (Claude-parity, see
//! `docs/plans/agentic_tool_system_2026_07_11.md`, Phase 1b).
//!
//! A single `memory` tool with a `command` discriminator, modeled on Anthropic's
//! memory tool: `view` / `create` / `str_replace` / `insert` / `delete` /
//! `rename`, all rooted at a per-workspace memory directory. It is the durable
//! scratchpad an agent reads at the start of a task and updates as it learns.
//!
//! SECURITY (non-negotiable, the one must-do the research flagged): every path is
//! resolved through [`safe_rel`], which rejects absolute paths, any `..`
//! component, and percent-encoded escapes (`%2e` / `%2f` / `%5c`), so nothing can
//! read or write outside the memory root. There is a dedicated test for each
//! escape vector.

use crate::common;
use crate::spec_helpers::write_spec;
use futures::future::BoxFuture;
use hide_core::tool::{Purity, Tool, ToolCtx, ToolResult, ToolSpec};
use hide_core::types::{Effect, EffectKind, EffectSet, RiskLevel};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

/// Where the memory files live. Defaults to `.hide/memories` under the cwd.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub root: PathBuf,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from(".hide/memories"),
        }
    }
}

#[derive(Clone)]
pub struct MemoryTool {
    spec: ToolSpec,
    config: MemoryConfig,
}

impl Default for MemoryTool {
    fn default() -> Self {
        Self::with_config(MemoryConfig::default())
    }
}

impl MemoryTool {
    pub fn with_config(config: MemoryConfig) -> Self {
        let spec = write_spec(
            "memory",
            "Memory",
            "Durable cross-session memory rooted at a private directory. \
             command=view|create|str_replace|insert|delete|rename. All paths are \
             relative to the memory root; escapes are rejected.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "delete", "rename"]
                    },
                    "path": { "type": "string", "description": "path relative to the memory root" },
                    "content": { "type": "string" },
                    "old_str": { "type": "string" },
                    "new_str": { "type": "string" },
                    "insert_line": { "type": "integer", "minimum": 0 },
                    "new_path": { "type": "string" }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
            Some(json!({
                "type": "object",
                "properties": { "command": {"type":"string"}, "path": {"type":"string"} }
            })),
        );
        Self { spec, config }
    }

    fn run(&self, args: &Value) -> ToolResult {
        let command = match args.get("command").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return common::arg_invalid("missing command", None, Some("/command")),
        };
        match command {
            "view" => self.view(args),
            "create" => self.create(args),
            "str_replace" => self.str_replace(args),
            "insert" => self.insert(args),
            "delete" => self.delete(args),
            "rename" => self.rename(args),
            other => common::arg_invalid(
                format!("unknown command \"{other}\""),
                Some("use one of view|create|str_replace|insert|delete|rename"),
                Some("/command"),
            ),
        }
    }

    /// Resolve `path` (defaulting to the root itself) safely under the root.
    fn resolve(&self, args: &Value, key: &str) -> Result<PathBuf, ToolResult> {
        let rel = args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        match safe_rel(rel) {
            Ok(rel) => Ok(self.config.root.join(rel)),
            Err(msg) => Err(common::coded(
                "ARG_INVALID",
                msg,
                true,
                Some("path escapes rejected"),
            )),
        }
    }

    fn view(&self, args: &Value) -> ToolResult {
        let full = match self.resolve(args, "path") {
            Ok(p) => p,
            Err(r) => return r,
        };
        if full.is_dir() || (!full.exists() && args.get("path").is_none()) {
            // Directory listing (du-style: relative names + a marker for dirs).
            let mut entries: Vec<String> = Vec::new();
            let read = std::fs::read_dir(&full);
            if let Ok(rd) = read {
                for e in rd.flatten() {
                    let name = e.file_name().to_string_lossy().to_string();
                    let suffix = if e.path().is_dir() { "/" } else { "" };
                    entries.push(format!("{name}{suffix}"));
                }
            }
            entries.sort();
            return common::ok(
                json!({ "command": "view", "kind": "dir", "entries": entries }),
                EffectSet::default(),
            );
        }
        match std::fs::read_to_string(&full) {
            Ok(text) => {
                // Numbered contents, cat -n style, so edits can target line numbers.
                let numbered: String = text
                    .lines()
                    .enumerate()
                    .map(|(i, l)| format!("{:>6}\t{}", i + 1, l))
                    .collect::<Vec<_>>()
                    .join("\n");
                common::ok(
                    json!({ "command": "view", "kind": "file", "content": numbered }),
                    EffectSet::default(),
                )
            }
            Err(_) => common::coded("NOT_FOUND", "no such memory path", true, None),
        }
    }

    fn create(&self, args: &Value) -> ToolResult {
        let full = match self.resolve(args, "path") {
            Ok(p) => p,
            Err(r) => return r,
        };
        let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
        if let Some(parent) = full.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(&full, content.as_bytes()) {
            Ok(()) => write_ok(&full, "create", json!({ "bytes": content.len() })),
            Err(e) => common::coded("TOOL_FAULT", e.to_string(), false, None),
        }
    }

    fn str_replace(&self, args: &Value) -> ToolResult {
        let full = match self.resolve(args, "path") {
            Ok(p) => p,
            Err(r) => return r,
        };
        let old = args.get("old_str").and_then(|v| v.as_str()).unwrap_or("");
        let new = args.get("new_str").and_then(|v| v.as_str()).unwrap_or("");
        if old.is_empty() {
            return common::arg_invalid("old_str must not be empty", None, Some("/old_str"));
        }
        let current = match std::fs::read_to_string(&full) {
            Ok(c) => c,
            Err(_) => return common::coded("NOT_FOUND", "no such memory path", true, None),
        };
        let count = current.matches(old).count();
        if count == 0 {
            return common::coded(
                "CONFLICT",
                "old_str not found",
                true,
                Some("re-view the file and copy an exact slice"),
            );
        }
        if count > 1 {
            return common::coded(
                "CONFLICT",
                format!("old_str matched {count} times; must be unique"),
                true,
                Some("include more surrounding context so the match is unique"),
            );
        }
        let next = current.replacen(old, new, 1);
        match std::fs::write(&full, next.as_bytes()) {
            Ok(()) => write_ok(&full, "str_replace", json!({ "replacements": 1 })),
            Err(e) => common::coded("TOOL_FAULT", e.to_string(), false, None),
        }
    }

    fn insert(&self, args: &Value) -> ToolResult {
        let full = match self.resolve(args, "path") {
            Ok(p) => p,
            Err(r) => return r,
        };
        let line = args
            .get("insert_line")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let current = std::fs::read_to_string(&full).unwrap_or_default();
        let mut lines: Vec<&str> = current.lines().collect();
        if line > lines.len() {
            return common::arg_invalid(
                format!("insert_line {line} is past end ({} lines)", lines.len()),
                Some("insert_line must be between 0 and the line count"),
                Some("/insert_line"),
            );
        }
        lines.insert(line, content);
        let next = lines.join("\n");
        match std::fs::write(&full, next.as_bytes()) {
            Ok(()) => write_ok(&full, "insert", json!({ "at_line": line })),
            Err(e) => common::coded("TOOL_FAULT", e.to_string(), false, None),
        }
    }

    fn delete(&self, args: &Value) -> ToolResult {
        let full = match self.resolve(args, "path") {
            Ok(p) => p,
            Err(r) => return r,
        };
        let result = if full.is_dir() {
            std::fs::remove_dir_all(&full)
        } else {
            std::fs::remove_file(&full)
        };
        match result {
            Ok(()) => write_ok(&full, "delete", json!({ "deleted": true })),
            Err(_) => common::coded("NOT_FOUND", "no such memory path", true, None),
        }
    }

    fn rename(&self, args: &Value) -> ToolResult {
        let from = match self.resolve(args, "path") {
            Ok(p) => p,
            Err(r) => return r,
        };
        let to = match self.resolve(args, "new_path") {
            Ok(p) => p,
            Err(r) => return r,
        };
        if args
            .get("new_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .is_empty()
        {
            return common::arg_invalid("missing new_path", None, Some("/new_path"));
        }
        if let Some(parent) = to.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::rename(&from, &to) {
            Ok(()) => write_ok(&to, "rename", json!({ "renamed": true })),
            Err(e) => common::coded("TOOL_FAULT", e.to_string(), false, None),
        }
    }
}

impl Tool for MemoryTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move { self.run(&args) })
    }

    fn purity(&self) -> Purity {
        Purity::Impure
    }
}

fn write_ok(path: &Path, command: &str, mut extra: Value) -> ToolResult {
    if let Some(obj) = extra.as_object_mut() {
        obj.insert("command".into(), json!(command));
        obj.insert("path".into(), json!(path.to_string_lossy()));
    }
    common::ok(
        extra,
        EffectSet {
            effects: vec![Effect {
                kind: EffectKind::Write,
                target: path.to_string_lossy().to_string(),
                bytes_hash: None,
                risk: RiskLevel::Medium,
                metadata: BTreeMap::new(),
            }],
        },
    )
}

/// Resolve a caller-supplied relative path to a safe, root-relative `PathBuf`, or
/// an error string. Rejects absolute paths, `..` traversal, and percent-encoded
/// escapes. This is the security boundary for the whole tool.
fn safe_rel(rel: &str) -> Result<PathBuf, String> {
    let lower = rel.to_ascii_lowercase();
    if lower.contains("%2e") || lower.contains("%2f") || lower.contains("%5c") {
        return Err("percent-encoded path escape rejected".to_string());
    }
    let p = Path::new(rel);
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            Component::ParentDir => return Err("'..' path traversal rejected".to_string()),
            Component::RootDir | Component::Prefix(_) => {
                return Err("absolute path rejected".to_string())
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolCtx {
        ToolCtx {
            grant_id: None,
            deadline_ms: None,
            output_cap_bytes: 1 << 20,
        }
    }

    fn tmp_tool(name: &str) -> (MemoryTool, PathBuf) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "hide_mem_{}_{}_{}_{}",
            name,
            std::process::id(),
            hide_core::ids::now_ms(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&root).unwrap();
        (
            MemoryTool::with_config(MemoryConfig { root: root.clone() }),
            root,
        )
    }

    #[tokio::test]
    async fn create_view_replace_roundtrip() {
        let (tool, root) = tmp_tool("rt");
        let r = tool
            .call(
                json!({ "command": "create", "path": "notes.md", "content": "a\nb\n" }),
                ctx(),
            )
            .await;
        assert!(r.ok, "create failed: {:?}", r.error);
        assert_eq!(
            std::fs::read_to_string(root.join("notes.md")).unwrap(),
            "a\nb\n"
        );

        let v = tool
            .call(json!({ "command": "view", "path": "notes.md" }), ctx())
            .await;
        assert!(v.ok);
        let content = v.structured_content.unwrap()["content"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(content.contains("1\ta"), "numbered view: {content}");

        let rep = tool
            .call(
                json!({ "command": "str_replace", "path": "notes.md", "old_str": "a", "new_str": "A" }),
                ctx(),
            )
            .await;
        assert!(rep.ok, "replace failed: {:?}", rep.error);
        assert_eq!(
            std::fs::read_to_string(root.join("notes.md")).unwrap(),
            "A\nb\n"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn str_replace_requires_unique_match() {
        let (tool, root) = tmp_tool("uniq");
        tool.call(
            json!({ "command": "create", "path": "f", "content": "x x x" }),
            ctx(),
        )
        .await;
        let r = tool
            .call(
                json!({ "command": "str_replace", "path": "f", "old_str": "x", "new_str": "y" }),
                ctx(),
            )
            .await;
        assert!(!r.ok);
        assert_eq!(r.error.unwrap().code, "CONFLICT");
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn insert_at_line_and_delete() {
        let (tool, root) = tmp_tool("ins");
        tool.call(
            json!({ "command": "create", "path": "f", "content": "a\nc" }),
            ctx(),
        )
        .await;
        let r = tool
            .call(
                json!({ "command": "insert", "path": "f", "insert_line": 1, "content": "b" }),
                ctx(),
            )
            .await;
        assert!(r.ok, "insert failed: {:?}", r.error);
        assert_eq!(std::fs::read_to_string(root.join("f")).unwrap(), "a\nb\nc");

        let d = tool
            .call(json!({ "command": "delete", "path": "f" }), ctx())
            .await;
        assert!(d.ok);
        assert!(!root.join("f").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn view_lists_directory() {
        let (tool, root) = tmp_tool("ls");
        tool.call(
            json!({ "command": "create", "path": "a.md", "content": "1" }),
            ctx(),
        )
        .await;
        tool.call(
            json!({ "command": "create", "path": "sub/b.md", "content": "2" }),
            ctx(),
        )
        .await;
        let v = tool.call(json!({ "command": "view" }), ctx()).await;
        assert!(v.ok);
        let entries = v.structured_content.unwrap()["entries"].clone();
        let entries: Vec<String> = serde_json::from_value(entries).unwrap();
        assert!(entries.contains(&"a.md".to_string()));
        assert!(entries.contains(&"sub/".to_string()));
        let _ = std::fs::remove_dir_all(root);
    }

    // --- the security boundary: every escape vector is rejected -------------

    #[tokio::test]
    async fn rejects_parent_traversal() {
        let (tool, root) = tmp_tool("esc1");
        let r = tool
            .call(
                json!({ "command": "create", "path": "../escape.txt", "content": "x" }),
                ctx(),
            )
            .await;
        assert!(!r.ok, "must reject ..");
        // The escape file must not exist outside the root.
        assert!(!root.parent().unwrap().join("escape.txt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn rejects_absolute_path() {
        let (tool, root) = tmp_tool("esc2");
        let r = tool
            .call(
                json!({ "command": "create", "path": "/tmp/hide_mem_escape", "content": "x" }),
                ctx(),
            )
            .await;
        assert!(!r.ok, "must reject absolute path");
        assert!(!Path::new("/tmp/hide_mem_escape").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn rejects_percent_encoded_traversal() {
        let (tool, root) = tmp_tool("esc3");
        let r = tool
            .call(
                json!({ "command": "view", "path": "%2e%2e/%2e%2e/etc/passwd" }),
                ctx(),
            )
            .await;
        assert!(!r.ok, "must reject percent-encoded ..");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn safe_rel_unit() {
        assert!(safe_rel("a/b.md").is_ok());
        assert!(safe_rel("./a").is_ok());
        assert!(safe_rel("../x").is_err());
        assert!(safe_rel("a/../../x").is_err());
        assert!(safe_rel("/abs").is_err());
        assert!(safe_rel("%2e%2e/x").is_err());
    }
}
