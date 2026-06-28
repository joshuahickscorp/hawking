//! Text search (ch.03 §4.6.4 `search.grep`).
//!
//! A real ripgrep-shaped lexical search: walk the workspace with the `ignore`
//! crate (respects `.gitignore`), match each line against a compiled `regex`,
//! and return `{path, line, text}` hits with a bounded result count. The bible
//! treats symbol/semantic search as thin wrappers over Ch.05's Living Index;
//! those are a documented seam (this crate does not re-implement the indexer).

use crate::common;
use crate::spec_helpers::read_spec;
use futures::future::BoxFuture;
use hide_core::tool::{Purity, Tool, ToolCtx, ToolResult, ToolSpec};
use hide_core::types::EffectSet;
use regex::RegexBuilder;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};

#[derive(Clone)]
pub struct SearchTextTool {
    spec: ToolSpec,
}

impl Default for SearchTextTool {
    fn default() -> Self {
        Self {
            spec: {
                let mut s = read_spec(
                    "search.text",
                    "Search text",
                    "Regex search over the workspace, respecting .gitignore. Returns line hits.",
                    json!({
                        "type": "object",
                        "properties": {
                            "pattern": { "type": "string" },
                            "root": { "type": "string", "default": "." },
                            "glob": { "type": "string" },
                            "max": { "type": "integer", "minimum": 1, "default": 200 },
                            "ignore_case": { "type": "boolean", "default": false }
                        },
                        "required": ["pattern"],
                        "additionalProperties": false
                    }),
                    Some(json!({
                        "type":"object",
                        "properties":{"matches":{"type":"array"},"truncated":{"type":"boolean"}},
                        "required":["matches"]
                    })),
                    1024 * 1024,
                );
                s.capabilities_required = vec!["index.read".to_string()];
                s
            },
        }
    }
}

impl Tool for SearchTextTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move {
            let Some(pattern) = args.get("pattern").and_then(|v| v.as_str()) else {
                return common::arg_invalid("missing string arg: pattern", None, Some("/pattern"));
            };
            let root = args.get("root").and_then(|v| v.as_str()).unwrap_or(".");
            let max = args.get("max").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
            let ignore_case = args
                .get("ignore_case")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let glob = args.get("glob").and_then(|v| v.as_str());

            let re = match RegexBuilder::new(pattern)
                .case_insensitive(ignore_case)
                .build()
            {
                Ok(re) => re,
                Err(err) => {
                    return common::arg_invalid(
                        format!("invalid regex: {err}"),
                        Some("provide a valid Rust regex"),
                        Some("/pattern"),
                    )
                }
            };
            let glob_set = glob.and_then(|g| {
                globset::Glob::new(g)
                    .ok()
                    .map(|g| g.compile_matcher())
            });

            let mut matches = Vec::new();
            let mut truncated = false;
            'walk: for entry in ignore::WalkBuilder::new(root)
                .git_ignore(true)
                .require_git(false)
                .hidden(true)
                .build()
                .flatten()
            {
                if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    continue;
                }
                let path = entry.path();
                if let Some(ref gs) = glob_set {
                    let rel = path.strip_prefix(root).unwrap_or(path);
                    if !gs.is_match(rel) && !gs.is_match(path) {
                        continue;
                    }
                }
                let Ok(file) = std::fs::File::open(path) else {
                    continue;
                };
                let reader = BufReader::new(file);
                for (idx, line) in reader.lines().enumerate() {
                    let Ok(line) = line else { break };
                    if re.is_match(&line) {
                        if matches.len() >= max {
                            truncated = true;
                            break 'walk;
                        }
                        matches.push(json!({
                            "path": path.to_string_lossy(),
                            "line": idx + 1,
                            "text": common::safe_prefix(&line, 400),
                        }));
                    }
                }
            }
            common::ok(
                json!({ "pattern": pattern, "matches": matches, "truncated": truncated }),
                EffectSet::default(),
            )
        })
    }

    fn purity(&self) -> Purity {
        Purity::PureFs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn ctx() -> ToolCtx {
        ToolCtx {
            grant_id: None,
            deadline_ms: None,
            output_cap_bytes: 1 << 20,
        }
    }

    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hide_search_{}_{}_{}",
            std::process::id(),
            hide_core::ids::now_ms(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn search_finds_matches_and_respects_gitignore() {
        let dir = tmp();
        std::fs::write(dir.join("a.rs"), "fn target() {}\nother\n").unwrap();
        std::fs::write(dir.join(".gitignore"), "skip.rs\n").unwrap();
        std::fs::write(dir.join("skip.rs"), "fn target() {}\n").unwrap();
        let tool = SearchTextTool::default();
        let r = tool
            .call(
                json!({ "pattern": "fn target", "root": dir.to_string_lossy() }),
                ctx(),
            )
            .await;
        let matches = r.structured_content.unwrap()["matches"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(matches.len(), 1);
        assert!(matches[0]["path"].as_str().unwrap().ends_with("a.rs"));
        assert_eq!(matches[0]["line"], 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn search_invalid_regex_is_arg_invalid() {
        let tool = SearchTextTool::default();
        let r = tool.call(json!({ "pattern": "(" }), ctx()).await;
        assert_eq!(r.error.unwrap().code, "ARG_INVALID");
    }

    #[test]
    fn arc_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<SearchTextTool>>();
    }
}
