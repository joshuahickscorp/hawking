//! The edit family (ch.03 §4.7): the tiered, *verifying* applier.
//!
//! * `edit.search_replace` (Tier 1): a list of `{search, replace, occurrence?}`
//!   blocks. The applier tries **exact** match, then **whitespace-normalized**,
//!   then a **fuzzy** match with a similarity floor; on still-miss it returns
//!   `CONFLICT` with the closest region + a `fix_hint` so the model re-reads and
//!   retries (TT2 fallback). This is the three-stage tolerance Aider learned the
//!   hard way.
//! * `edit.apply_patch` (Tier 2): a standard unified diff, applied hunk-by-hunk
//!   against the current file and verified by re-deriving the post-image.
//! * `edit.write_file` (Tier 0): full-file write with `create_only`.
//!
//! Every tier supports `base_hash` (the blake3 the model based its edit on) for
//! **optimistic concurrency** — a mismatch is `CONFLICT`, no write (§4.7). Each
//! tool's `simulate()` computes the post-image without committing (TT9).

use crate::common;
use crate::spec_helpers::write_spec;
use futures::future::BoxFuture;
use hide_core::tool::{Purity, Tool, ToolCtx, ToolResult, ToolSpec};
use hide_core::types::{Effect, EffectKind, EffectSet, RiskLevel};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// Outcome of computing a post-image without writing it.
enum Plan {
    /// New file content + a per-edit log.
    Ready { content: String, applied: Vec<Value> },
    /// A conflict the model must resolve (no write).
    Conflict { message: String, hint: String },
    /// An argument problem.
    Invalid { message: String, ptr: &'static str },
}

// ---------------------------------------------------------------------------
// edit.search_replace
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SearchReplaceTool {
    spec: ToolSpec,
}

impl Default for SearchReplaceTool {
    fn default() -> Self {
        Self {
            spec: write_spec(
                "edit.search_replace",
                "Search/replace edit",
                "Apply exact/anchored search-replace blocks (Aider/Cline style) with whitespace \
                 and fuzzy fallback; returns CONFLICT with the closest region on no-match.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "edits": { "type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "search": {"type":"string"},
                                "replace": {"type":"string"},
                                "occurrence": {"type":"string", "enum":["first","all"], "default":"first"}
                            },
                            "required": ["search","replace"],
                            "additionalProperties": false
                        }},
                        "base_hash": { "type": "string" }
                    },
                    "required": ["path","edits"],
                    "additionalProperties": false
                }),
                Some(json!({
                    "type":"object",
                    "properties": {"path":{"type":"string"},"applied":{"type":"array"}},
                    "required":["path","applied"]
                })),
            ),
        }
    }
}

impl Tool for SearchReplaceTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move { run_plan(&args, plan_search_replace(&args), true) })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move { simulate_effect(args) })
    }

    fn purity(&self) -> Purity {
        Purity::Impure
    }
}

// ---------------------------------------------------------------------------
// edit.apply_patch
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ApplyPatchTool {
    spec: ToolSpec,
}

impl Default for ApplyPatchTool {
    fn default() -> Self {
        Self {
            spec: write_spec(
                "edit.apply_patch",
                "Apply unified diff",
                "Apply a unified diff to a file, verified by re-deriving the post-image. \
                 base_hash gives optimistic concurrency.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "patch": { "type": "string" },
                        "base_hash": { "type": "string" }
                    },
                    "required": ["path","patch"],
                    "additionalProperties": false
                }),
                None,
            ),
        }
    }
}

impl Tool for ApplyPatchTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move { run_plan(&args, plan_apply_patch(&args), true) })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move { simulate_effect(args) })
    }

    fn purity(&self) -> Purity {
        Purity::Impure
    }
}

// ---------------------------------------------------------------------------
// edit.write_file
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct WriteFileTool {
    spec: ToolSpec,
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self {
            spec: write_spec(
                "edit.write_file",
                "Write whole file",
                "Tier-0 full-file write; create_only guards against clobbering, base_hash gives \
                 optimistic concurrency.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" },
                        "create_only": { "type": "boolean", "default": false },
                        "base_hash": { "type": "string" }
                    },
                    "required": ["path","content"],
                    "additionalProperties": false
                }),
                None,
            ),
        }
    }
}

impl Tool for WriteFileTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move { run_plan(&args, plan_write_file(&args), true) })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move { simulate_effect(args) })
    }

    fn purity(&self) -> Purity {
        Purity::Impure
    }
}

// ---------------------------------------------------------------------------
// shared planning + commit
// ---------------------------------------------------------------------------

fn read_current(path: &str) -> std::io::Result<String> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(s),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(err) => Err(err),
    }
}

fn check_base_hash(args: &Value, current: &str) -> Option<Plan> {
    let want = args.get("base_hash").and_then(|v| v.as_str())?;
    let got = blake3::hash(current.as_bytes()).to_hex().to_string();
    if got != want {
        Some(Plan::Conflict {
            message: format!("base_hash mismatch (on-disk {got}, expected {want})"),
            hint: "the file changed since you read it; re-read and re-plan the edit".to_string(),
        })
    } else {
        None
    }
}

fn plan_write_file(args: &Value) -> Plan {
    let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
        return Plan::Invalid { message: "missing path".into(), ptr: "/path" };
    };
    let Some(content) = args.get("content").and_then(|v| v.as_str()) else {
        return Plan::Invalid { message: "missing content".into(), ptr: "/content" };
    };
    let create_only = args.get("create_only").and_then(|v| v.as_bool()).unwrap_or(false);
    if create_only && Path::new(path).exists() {
        return Plan::Conflict {
            message: format!("create_only set but {path} exists"),
            hint: "set create_only=false to overwrite".into(),
        };
    }
    let current = read_current(path).unwrap_or_default();
    if let Some(conflict) = check_base_hash(args, &current) {
        return conflict;
    }
    Plan::Ready {
        content: content.to_string(),
        applied: vec![json!({ "tier": "write_file", "bytes": content.len() })],
    }
}

fn plan_search_replace(args: &Value) -> Plan {
    let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
        return Plan::Invalid { message: "missing path".into(), ptr: "/path" };
    };
    let Some(edits) = args.get("edits").and_then(|v| v.as_array()) else {
        return Plan::Invalid { message: "missing edits[]".into(), ptr: "/edits" };
    };
    let mut current = match read_current(path) {
        Ok(c) => c,
        Err(err) => {
            return Plan::Invalid { message: err.to_string(), ptr: "/path" };
        }
    };
    if let Some(conflict) = check_base_hash(args, &current) {
        return conflict;
    }
    let mut applied = Vec::new();
    for (i, edit) in edits.iter().enumerate() {
        let search = edit.get("search").and_then(|v| v.as_str()).unwrap_or("");
        let replace = edit.get("replace").and_then(|v| v.as_str()).unwrap_or("");
        let all = edit.get("occurrence").and_then(|v| v.as_str()) == Some("all");
        match apply_one(&current, search, replace, all) {
            Ok((next, strategy, count)) => {
                current = next;
                applied.push(json!({ "index": i, "strategy": strategy, "replacements": count }));
            }
            Err(closest) => {
                return Plan::Conflict {
                    message: format!("edit #{i}: search block not found"),
                    hint: format!(
                        "closest region was:\n{}\nre-read the file and copy an exact slice",
                        common::safe_prefix(&closest, 400)
                    ),
                };
            }
        }
    }
    Plan::Ready { content: current, applied }
}

/// Apply one search/replace with the three-stage tolerance ladder. Returns the
/// new content + the strategy that matched + replacement count, or the closest
/// region on miss.
fn apply_one(
    content: &str,
    search: &str,
    replace: &str,
    all: bool,
) -> Result<(String, &'static str, usize), String> {
    if search.is_empty() {
        return Err(String::new());
    }
    // 1. exact
    if content.contains(search) {
        let count = if all { content.matches(search).count() } else { 1 };
        let next = if all {
            content.replace(search, replace)
        } else {
            content.replacen(search, replace, 1)
        };
        return Ok((next, "exact", count));
    }
    // 2. whitespace-normalized: match a window of the file whose collapsed
    //    whitespace equals the search's collapsed whitespace.
    if let Some((start, end)) = find_ws_normalized(content, search) {
        let mut next = String::with_capacity(content.len());
        next.push_str(&content[..start]);
        next.push_str(replace);
        next.push_str(&content[end..]);
        return Ok((next, "whitespace", 1));
    }
    // 3. fuzzy: best similar window over line-anchored slices.
    if let Some((start, end, score)) = find_fuzzy(content, search) {
        if score >= 0.75 {
            let mut next = String::with_capacity(content.len());
            next.push_str(&content[..start]);
            next.push_str(replace);
            next.push_str(&content[end..]);
            return Ok((next, "fuzzy", 1));
        }
        return Err(content[start..end].to_string());
    }
    Err(closest_region(content, search))
}

/// Collapse runs of ASCII whitespace to a single space and trim.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Find a byte range in `content` whose whitespace-normalized form equals the
/// normalized `search`. Scans line-anchored start positions for robustness.
fn find_ws_normalized(content: &str, search: &str) -> Option<(usize, usize)> {
    let needle = collapse_ws(search);
    if needle.is_empty() {
        return None;
    }
    let search_lines = search.lines().count().max(1);
    let line_starts = line_start_offsets(content);
    for &start in &line_starts {
        // Take up to search_lines + small slack lines from start.
        let mut end = start;
        let mut taken = 0;
        while end < content.len() && taken < search_lines + 2 {
            // advance to next line boundary (or EOF)
            let last = match content[end..].find('\n') {
                Some(rel) => {
                    end += rel + 1;
                    false
                }
                None => {
                    end = content.len();
                    true
                }
            };
            taken += 1;
            let window = content[start..end].trim_end_matches('\n');
            if collapse_ws(window) == needle {
                return Some((start, start + window.len()));
            }
            if last {
                break;
            }
        }
    }
    None
}

/// Find the most similar line-anchored window to `search` using `similar`.
fn find_fuzzy(content: &str, search: &str) -> Option<(usize, usize, f64)> {
    let search_lines = search.lines().count().max(1);
    let line_starts = line_start_offsets(content);
    let mut best: Option<(usize, usize, f64)> = None;
    for &start in &line_starts {
        let mut end = start;
        for _ in 0..search_lines {
            match content[end..].find('\n') {
                Some(rel) => end += rel + 1,
                None => {
                    end = content.len();
                    break;
                }
            }
        }
        let window = content[start..end].trim_end_matches('\n');
        let real_end = start + window.len();
        let score = similar::TextDiff::from_lines(window, search).ratio() as f64;
        if best.map(|(_, _, s)| score > s).unwrap_or(true) {
            best = Some((start, real_end, score));
        }
    }
    best
}

fn closest_region(content: &str, search: &str) -> String {
    find_fuzzy(content, search)
        .map(|(s, e, _)| content[s..e].to_string())
        .unwrap_or_default()
}

fn line_start_offsets(content: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (i, b) in content.bytes().enumerate() {
        if b == b'\n' && i + 1 < content.len() {
            starts.push(i + 1);
        }
    }
    starts
}

/// Apply a unified diff. We parse hunks and reconstruct the post-image by walking
/// the original lines, matching context/removed lines, and emitting added lines.
fn plan_apply_patch(args: &Value) -> Plan {
    let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
        return Plan::Invalid { message: "missing path".into(), ptr: "/path" };
    };
    let Some(patch) = args.get("patch").and_then(|v| v.as_str()) else {
        return Plan::Invalid { message: "missing patch".into(), ptr: "/patch" };
    };
    let current = match read_current(path) {
        Ok(c) => c,
        Err(err) => return Plan::Invalid { message: err.to_string(), ptr: "/path" },
    };
    if let Some(conflict) = check_base_hash(args, &current) {
        return conflict;
    }
    match apply_unified(&current, patch) {
        Ok(content) => Plan::Ready {
            content,
            applied: vec![json!({ "tier": "unified_diff" })],
        },
        Err(msg) => Plan::Conflict {
            message: msg,
            hint: "context lines did not match; re-read the file and regenerate the diff".into(),
        },
    }
}

/// Minimal but real unified-diff applier. Supports multiple `@@` hunks; matches
/// context (` `) and removed (`-`) lines against the original, emits added (`+`).
/// Tolerates absent/incorrect line numbers by searching forward for the hunk's
/// leading context (fuzz).
fn apply_unified(original: &str, patch: &str) -> Result<String, String> {
    let orig_lines: Vec<&str> = original.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize; // index into orig_lines

    let patch_lines: Vec<&str> = patch.lines().collect();
    let mut i = 0;
    let mut saw_hunk = false;
    while i < patch_lines.len() {
        let line = patch_lines[i];
        if line.starts_with("@@") {
            saw_hunk = true;
            i += 1;
            // Collect this hunk's body until the next @@ or EOF.
            let mut hunk: Vec<&str> = Vec::new();
            while i < patch_lines.len() && !patch_lines[i].starts_with("@@") {
                // Stop if we hit a new file header (shouldn't in single-file patch).
                if patch_lines[i].starts_with("--- ") || patch_lines[i].starts_with("+++ ") {
                    i += 1;
                    continue;
                }
                hunk.push(patch_lines[i]);
                i += 1;
            }
            // Drop trailing blank lines: `patch.lines()` yields a spurious "" for a
            // patch string ending in a blank line (and inter-hunk blanks land here
            // too). Those are NOT context lines, so keeping them would force `locate`
            // to demand a phantom blank the file lacks. Interior blanks are kept.
            while hunk.last() == Some(&"") {
                hunk.pop();
            }
            // Determine the hunk's "old" sequence (context + removed) to locate it.
            // A bare interior "" line is a blank context line (a " " context line
            // whose trailing space was stripped), so it IS part of the located
            // sequence; dropping it would desync `locate` from the walk below.
            let old_seq: Vec<&str> = hunk
                .iter()
                .filter(|l| l.is_empty() || l.starts_with(' ') || l.starts_with('-'))
                .map(|l| &l[1.min(l.len())..])
                .collect();
            // Find old_seq in orig_lines starting at cursor (fuzz: scan forward).
            let anchor = locate(&orig_lines, &old_seq, cursor)
                .ok_or_else(|| "could not locate hunk context in file".to_string())?;
            // `locate` can wrap to the file top, so a later hunk whose context only
            // matches earlier returns an anchor BEFORE the cursor. Copying
            // orig_lines[cursor..anchor] would then panic (start > end). Treat a
            // backward hunk as an out-of-order conflict, not a crash.
            if anchor < cursor {
                return Err("hunk context precedes the current position (out-of-order hunk)".to_string());
            }
            // Copy unchanged lines up to the anchor.
            for l in &orig_lines[cursor..anchor] {
                out.push((*l).to_string());
            }
            cursor = anchor;
            // Walk the hunk body.
            for hl in &hunk {
                if hl.is_empty() {
                    // A blank context line must line up with a blank line in the
                    // file at the cursor. Anything else is a desync; emitting the
                    // file's (non-blank) line here would silently corrupt it, so
                    // conflict instead.
                    if cursor >= orig_lines.len() || !orig_lines[cursor].is_empty() {
                        return Err(format!(
                            "patch has a blank context line where the file has \"{}\"",
                            orig_lines.get(cursor).copied().unwrap_or("<eof>")
                        ));
                    }
                    out.push(String::new());
                    cursor += 1;
                    continue;
                }
                let (tag, rest) = hl.split_at(1);
                match tag {
                    " " => {
                        out.push(rest.to_string());
                        cursor += 1;
                    }
                    "-" => {
                        // Verify the line being removed actually matches the file at
                        // the cursor before dropping it. Without this a desynced
                        // cursor (e.g. from a stray blank hunk line) would delete the
                        // WRONG line by count and silently commit a corrupt file.
                        if cursor >= orig_lines.len() || orig_lines[cursor] != rest {
                            return Err(format!(
                                "patch removes a line that does not match the file (expected \"{}\", found \"{}\")",
                                rest,
                                orig_lines.get(cursor).copied().unwrap_or("<eof>")
                            ));
                        }
                        cursor += 1; // drop original line
                    }
                    "+" => {
                        out.push(rest.to_string());
                    }
                    "\\" => { /* "\ No newline at end of file" */ }
                    _ => {}
                }
            }
        } else {
            i += 1;
        }
    }
    if !saw_hunk {
        return Err("no @@ hunks found in patch".to_string());
    }
    // Append any remaining original lines.
    for l in &orig_lines[cursor.min(orig_lines.len())..] {
        out.push((*l).to_string());
    }
    let mut result = out.join("\n");
    if original.ends_with('\n') {
        result.push('\n');
    }
    Ok(result)
}

/// Locate `seq` within `lines` at or after `from`, returning the start index.
fn locate(lines: &[&str], seq: &[&str], from: usize) -> Option<usize> {
    if seq.is_empty() {
        return Some(from);
    }
    let mut start = from;
    while start + seq.len() <= lines.len() {
        if lines[start..start + seq.len()] == *seq {
            return Some(start);
        }
        start += 1;
    }
    // Fuzz: retry from the top (line numbers may have been wrong).
    if from > 0 {
        return locate(lines, seq, 0);
    }
    None
}

fn run_plan(args: &Value, plan: Plan, commit: bool) -> ToolResult {
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
    match plan {
        Plan::Invalid { message, ptr } => common::arg_invalid(message, None, Some(ptr)),
        Plan::Conflict { message, hint } => common::coded("CONFLICT", message, true, Some(&hint)),
        Plan::Ready { content, applied } => {
            if !commit {
                return common::ok(
                    json!({ "path": path, "applied": applied, "committed": false }),
                    EffectSet::default(),
                );
            }
            if let Some(parent) = Path::new(path).parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            let post_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
            match std::fs::write(path, content.as_bytes()) {
                Ok(()) => {
                    let mut metadata = BTreeMap::new();
                    metadata.insert("post_hash".to_string(), post_hash.clone());
                    common::ok(
                        json!({
                            "path": path, "applied": applied,
                            "committed": true, "post_hash": post_hash
                        }),
                        EffectSet {
                            effects: vec![Effect {
                                kind: EffectKind::Write,
                                target: path.to_string(),
                                bytes_hash: Some(post_hash),
                                risk: RiskLevel::High,
                                metadata,
                            }],
                        },
                    )
                }
                Err(err) => common::coded("TOOL_FAULT", err.to_string(), false, None),
            }
        }
    }
}

fn simulate_effect(args: &Value) -> Option<EffectSet> {
    let path = args.get("path").and_then(|v| v.as_str())?;
    Some(EffectSet {
        effects: vec![Effect {
            kind: EffectKind::Write,
            target: path.to_string(),
            bytes_hash: None,
            risk: RiskLevel::High,
            metadata: BTreeMap::new(),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ctx() -> ToolCtx {
        ToolCtx {
            grant_id: None,
            deadline_ms: None,
            output_cap_bytes: 1 << 20,
        }
    }

    fn tmp(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hide_edit_{}_{}_{}_{}",
            name,
            std::process::id(),
            hide_core::ids::now_ms(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn search_replace_exact() {
        let dir = tmp("sr");
        let file = dir.join("f.rs");
        std::fs::write(&file, "fn main() {\n    let x = 1;\n}\n").unwrap();
        let tool = SearchReplaceTool::default();
        let r = tool
            .call(
                json!({
                    "path": file.to_string_lossy(),
                    "edits": [{"search":"let x = 1;","replace":"let x = 2;"}]
                }),
                ctx(),
            )
            .await;
        assert!(r.ok);
        assert!(std::fs::read_to_string(&file).unwrap().contains("let x = 2;"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn search_replace_whitespace_fallback() {
        let dir = tmp("ws");
        let file = dir.join("f.rs");
        // File uses tabs + trailing spaces; search uses plain single spaces.
        std::fs::write(&file, "fn  main() {\n\tlet  x   =  1;   \n}\n").unwrap();
        let tool = SearchReplaceTool::default();
        // search differs only in run-length of whitespace (collapses identically).
        let r = tool
            .call(
                json!({
                    "path": file.to_string_lossy(),
                    "edits": [{"search":"let x = 1;","replace":"let x = 2;"}]
                }),
                ctx(),
            )
            .await;
        assert!(
            r.ok,
            "whitespace-normalized match should succeed: {:?}",
            r.error
        );
        assert!(std::fs::read_to_string(&file).unwrap().contains("let x = 2;"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn search_replace_conflict_on_no_match() {
        let dir = tmp("conf");
        let file = dir.join("f.rs");
        std::fs::write(&file, "totally\nunrelated\ncontent\n").unwrap();
        let tool = SearchReplaceTool::default();
        let r = tool
            .call(
                json!({
                    "path": file.to_string_lossy(),
                    "edits": [{"search":"the quick brown fox jumps","replace":"x"}]
                }),
                ctx(),
            )
            .await;
        assert!(!r.ok);
        assert_eq!(r.error.unwrap().code, "CONFLICT");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn base_hash_mismatch_is_conflict() {
        let dir = tmp("bh");
        let file = dir.join("f.rs");
        std::fs::write(&file, "current\n").unwrap();
        let tool = SearchReplaceTool::default();
        let r = tool
            .call(
                json!({
                    "path": file.to_string_lossy(),
                    "edits": [{"search":"current","replace":"new"}],
                    "base_hash": "deadbeef"
                }),
                ctx(),
            )
            .await;
        assert_eq!(r.error.unwrap().code, "CONFLICT");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn apply_patch_unified_diff() {
        let dir = tmp("patch");
        let file = dir.join("f.txt");
        std::fs::write(&file, "line1\nline2\nline3\n").unwrap();
        let patch = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n line1\n-line2\n+line2-edited\n line3\n";
        let tool = ApplyPatchTool::default();
        let r = tool
            .call(
                json!({ "path": file.to_string_lossy(), "patch": patch }),
                ctx(),
            )
            .await;
        assert!(r.ok, "patch should apply: {:?}", r.error);
        let got = std::fs::read_to_string(&file).unwrap();
        assert_eq!(got, "line1\nline2-edited\nline3\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn apply_patch_out_of_order_hunk_is_conflict_not_panic() {
        // Hunks in the wrong order: edit "b" before "target", but "target" is earlier
        // in the file, so `locate` wraps backward and returns anchor < cursor. That
        // used to panic on orig_lines[cursor..anchor]; now it is an honest CONFLICT.
        let dir = tmp("ooo");
        let file = dir.join("f.txt");
        std::fs::write(&file, "target\na\nb\n").unwrap();
        let patch = "@@\n-b\n+B\n@@\n-target\n+TARGET\n";
        let tool = ApplyPatchTool::default();
        let r = tool
            .call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx())
            .await;
        assert!(!r.ok, "out-of-order hunk must not apply");
        assert_eq!(r.error.unwrap().code, "CONFLICT");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "target\na\nb\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn apply_patch_mismatched_removal_is_conflict_not_corruption() {
        // A stray blank hunk line desyncs the cursor so the '-' removal no longer
        // matches the file. Instead of silently deleting the wrong line (was: writes
        // "a\nA\n"), the applier now returns CONFLICT and writes nothing.
        let dir = tmp("mism");
        let file = dir.join("f.txt");
        std::fs::write(&file, "a\nb\n").unwrap();
        let patch = "@@\n\n-a\n+A\n";
        let tool = ApplyPatchTool::default();
        let r = tool
            .call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx())
            .await;
        assert!(!r.ok, "mismatched removal must not silently corrupt");
        assert_eq!(r.error.unwrap().code, "CONFLICT");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "a\nb\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn apply_patch_blank_context_desync_is_conflict_not_corruption() {
        // Blank context line with no matching blank in the file and no '-' line to
        // trip the removal guard: must CONFLICT, not duplicate a line (was: wrote
        // "a\nb\nX\nb\n").
        let dir = tmp("blankdesync");
        let file = dir.join("f.txt");
        std::fs::write(&file, "a\nb\n").unwrap();
        let patch = "@@\n a\n\n+X\n b\n";
        let tool = ApplyPatchTool::default();
        let r = tool
            .call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx())
            .await;
        assert!(!r.ok, "blank-context desync must conflict, not corrupt");
        assert_eq!(r.error.unwrap().code, "CONFLICT");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "a\nb\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn apply_patch_tolerates_trailing_blank_in_patch_body() {
        // A patch string ending in a blank line ("...\n\n") must still apply: the
        // trailing "" is a patch terminator, not a required blank context line
        // (regression guard for the old_seq empty-line change).
        let dir = tmp("trailblank");
        let file = dir.join("f.txt");
        std::fs::write(&file, "a\nb\nc\n").unwrap();
        let patch = "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n\n";
        let tool = ApplyPatchTool::default();
        let r = tool
            .call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx())
            .await;
        assert!(r.ok, "trailing blank must not block apply: {:?}", r.error);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "a\nB\nc\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn apply_patch_stripped_blank_context_still_applies() {
        // A valid diff whose blank context line was stripped to "" must still apply
        // when the file genuinely has that interior blank line (was over-rejected).
        let dir = tmp("stripblank");
        let file = dir.join("f.txt");
        std::fs::write(&file, "a\n\nc\n").unwrap();
        let patch = "@@ -1,3 +1,3 @@\n a\n\n-c\n+C\n";
        let tool = ApplyPatchTool::default();
        let r = tool
            .call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx())
            .await;
        assert!(r.ok, "stripped-blank context should apply: {:?}", r.error);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "a\n\nC\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn write_file_create_only_conflict() {
        let dir = tmp("wf");
        let file = dir.join("f.txt");
        std::fs::write(&file, "exists").unwrap();
        let tool = WriteFileTool::default();
        let r = tool
            .call(
                json!({ "path": file.to_string_lossy(), "content": "x", "create_only": true }),
                ctx(),
            )
            .await;
        assert_eq!(r.error.unwrap().code, "CONFLICT");
        let _ = std::fs::remove_dir_all(dir);
    }
}
