//! Builtin HIDE tools (bible ch.03).
//!
//! The dispatcher and permission model live in `hide-core`; this crate provides
//! concrete builtin tool implementations and the MCP host/client bridge.
//!
//! Module map:
//! * [`fs`] — read/list/write/stat/glob/watch (§4.6.1), `bytes_ref` spill.
//! * [`edit`] — the tiered verifying applier: search_replace / apply_patch /
//!   write_file (§4.7), with optimistic-concurrency `base_hash`.
//! * [`shell`] — sandboxed `shell.run` with a timeout watchdog (§4.8); `shell.plan`.
//! * [`proc`] — `test.run`/`build.run`/`compile.check`, EXEC_NONZERO-as-data.
//! * [`search`] — `search.text` (ignore-walker + regex).
//! * [`git`] — status/diff/log/commit + the worktree trio (§4.6.6).
//! * [`mcp`] — JSON-RPC 2.0 MCP client over stdio + Streamable HTTP (§4.10).

#[rustfmt::skip]
pub mod common {
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
            content: vec![ToolContent::Json { value: structured.clone() }],
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
            content: vec![ToolContent::Json { value: structured.clone() }],
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
        error_result(ToolStatus::ToolError, ToolError::new("TOOL_FAULT", message, true))
    }

    /// Argument validation failure (`ARG_INVALID`) with an optional repair hint and
    /// JSON-pointer into the offending arg.
    pub fn arg_invalid(message: impl Into<String>, fix_hint: Option<&str>, schema_path: Option<&str>) -> ToolResult {
        let mut err = ToolError::new("ARG_INVALID", message, true);
        err.fix_hint = fix_hint.map(ToOwned::to_owned);
        err.schema_path = schema_path.map(ToOwned::to_owned);
        error_result(ToolStatus::ProtocolError, err)
    }

    /// A typed tool error with an explicit taxonomy code (`NOT_FOUND`, `CONFLICT`,
    /// `TIMEOUT`, `TOOL_FAULT`, …). `retriable` follows the §4.2.3 taxonomy.
    pub fn coded(code: &str, message: impl Into<String>, retriable: bool, fix_hint: Option<&str>) -> ToolResult {
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
            return Spill { head: body, truncated: false, total_bytes, bytes_ref: None };
        }
        // Over cap. Preserve a UTF-8-safe head preview.
        let preview_len = cap_bytes.min(SPILL_PREVIEW_BYTES);
        let head = safe_prefix(&body, preview_len).to_string();
        let bytes_ref = blobs.and_then(|store| store.put(body.into_bytes(), Some("text/plain".to_string())).ok());
        Spill { head, truncated: true, total_bytes, bytes_ref }
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
}
#[rustfmt::skip]
pub mod edit {
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
            let next = if all { content.replace(search, replace) } else { content.replacen(search, replace, 1) };
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
        find_fuzzy(content, search).map(|(s, e, _)| content[s..e].to_string()).unwrap_or_default()
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
            Ok(content) => Plan::Ready { content, applied: vec![json!({ "tier": "unified_diff" })] },
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
                // An all-blank hunk body strips to nothing; an empty located sequence
                // would make `locate` a no-op and silently "apply" a change-free garbage
                // hunk as ok. A real hunk always has at least one context/change line.
                if hunk.is_empty() {
                    return Err("empty hunk body (no context or change lines)".to_string());
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
            ToolCtx { grant_id: None, deadline_ms: None, output_cap_bytes: 1 << 20 }
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
            assert!(r.ok, "whitespace-normalized match should succeed: {:?}", r.error);
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
            let r = tool.call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx()).await;
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
            let r = tool.call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx()).await;
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
            let r = tool.call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx()).await;
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
            let r = tool.call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx()).await;
            assert!(!r.ok, "blank-context desync must conflict, not corrupt");
            assert_eq!(r.error.unwrap().code, "CONFLICT");
            assert_eq!(std::fs::read_to_string(&file).unwrap(), "a\nb\n");
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn apply_patch_all_blank_hunk_is_conflict_not_silent_ok() {
            // A hunk body of only blank lines strips to empty; it must CONFLICT, not
            // report a successful no-op apply (regression guard for the trailing strip).
            let dir = tmp("allblank");
            let file = dir.join("f.txt");
            std::fs::write(&file, "a\nb\nc\n").unwrap();
            let patch = "@@\n\n\n";
            let tool = ApplyPatchTool::default();
            let r = tool.call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx()).await;
            assert!(!r.ok, "all-blank hunk must not report success");
            assert_eq!(std::fs::read_to_string(&file).unwrap(), "a\nb\nc\n");
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
            let r = tool.call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx()).await;
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
            let r = tool.call(json!({ "path": file.to_string_lossy(), "patch": patch }), ctx()).await;
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
            let r =
                tool.call(json!({ "path": file.to_string_lossy(), "content": "x", "create_only": true }), ctx()).await;
            assert_eq!(r.error.unwrap().code, "CONFLICT");
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
#[rustfmt::skip]
pub mod fs {
    //! Filesystem tools (ch.03 §4.6.1): read, list, write, stat, glob, watch.
    //!
    //! `fs.read` gains `range`/`encoding` args and spills over-cap output to the blob
    //! CAS with a head preview (TT5). `fs.glob`/`fs.list` respect `.gitignore` via the
    //! `ignore` crate. `fs.watch` registers a `notify`-style watcher that emits change
    //! events; here it does a bounded synchronous poll (a real watcher daemon is a
    //! documented seam — the agent loop owns the long-lived watch).

    use crate::common;
    use crate::spec_helpers::{read_spec, write_spec};
    use futures::future::BoxFuture;
    use globset::{Glob, GlobSetBuilder};
    use hide_core::persistence::BlobStore;
    use hide_core::tool::{Purity, Tool, ToolCtx, ToolResult, ToolSpec};
    use hide_core::types::{Effect, EffectKind, EffectSet, RiskLevel};
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// Shared FS config: an optional blob store for over-cap read spill.
    #[derive(Clone, Default)]
    pub struct FsConfig {
        pub blobs: Option<Arc<dyn BlobStore>>,
    }

    // ---------------------------------------------------------------------------
    // fs.read
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    pub struct FsReadTool {
        spec: ToolSpec,
        config: FsConfig,
    }

    impl Default for FsReadTool {
        fn default() -> Self {
            Self {
                spec: read_spec(
                    "fs.read",
                    "Read file",
                    "Read a file from an authorized scope. Supports line ranges and base64/utf8 \
                     encoding; spills over-cap output to a bytes_ref with a head preview.",
                    json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" },
                            "range": { "type": "object", "properties": {
                                "start_line": {"type":"integer","minimum":1},
                                "end_line": {"type":"integer","minimum":1}
                            }, "additionalProperties": false },
                            "encoding": { "type": "string", "enum": ["utf8","base64","auto"], "default": "auto" }
                        },
                        "required": ["path"],
                        "additionalProperties": false
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": {
                            "path": {"type":"string"}, "content": {"type":"string"},
                            "truncated": {"type":"boolean"}, "bytes": {"type":"integer"},
                            "encoding": {"type":"string"}
                        },
                        "required": ["path", "content"]
                    })),
                    1024 * 1024,
                ),
                config: FsConfig::default(),
            }
        }
    }

    impl FsReadTool {
        pub fn with_config(config: FsConfig) -> Self {
            Self { config, ..Self::default() }
        }
    }

    impl Tool for FsReadTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing string arg: path", None, Some("/path"));
                };
                let encoding = args.get("encoding").and_then(|v| v.as_str()).unwrap_or("auto");
                let range = parse_range(&args);

                let bytes = match std::fs::read(path) {
                    Ok(b) => b,
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                        return common::coded(
                            "NOT_FOUND",
                            format!("no such file: {path}"),
                            true,
                            Some("check the path; use fs.list to enumerate the directory"),
                        );
                    }
                    Err(err) => return common::coded("TOOL_FAULT", err.to_string(), false, None),
                };

                let is_binary = bytes.iter().take(8000).any(|&b| b == 0);
                let use_base64 = encoding == "base64" || (encoding == "auto" && is_binary);

                if use_base64 {
                    let encoded = base64_encode(&bytes);
                    let spill = common::maybe_spill(encoded, ctx.output_cap_bytes as usize, self.config.blobs.as_ref());
                    return finalize_read(path, "base64", spill, bytes.len());
                }

                // UTF-8 text path (with optional line range).
                let text = match String::from_utf8(bytes) {
                    Ok(t) => t,
                    Err(err) => {
                        // Fall back to base64 rather than hard-erroring on non-UTF-8.
                        let raw = err.into_bytes();
                        let total = raw.len();
                        let encoded = base64_encode(&raw);
                        let spill =
                            common::maybe_spill(encoded, ctx.output_cap_bytes as usize, self.config.blobs.as_ref());
                        return finalize_read(path, "base64", spill, total);
                    }
                };
                let sliced = match range {
                    Some((start, end)) => slice_lines(&text, start, end),
                    None => text,
                };
                let total = sliced.len();
                let spill = common::maybe_spill(sliced, ctx.output_cap_bytes as usize, self.config.blobs.as_ref());
                finalize_read(path, "utf8", spill, total)
            })
        }

        fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
            Box::pin(async move {
                args.get("path").and_then(|v| v.as_str()).map(|path| EffectSet { effects: vec![read_effect(path)] })
            })
        }

        fn purity(&self) -> Purity {
            Purity::PureFs
        }
    }

    fn finalize_read(path: &str, encoding: &str, spill: common::Spill, total: usize) -> ToolResult {
        let mut body = json!({
            "path": path,
            "content": spill.head,
            "truncated": spill.truncated,
            "bytes": total,
            "encoding": encoding,
        });
        if let Some(ref blob) = spill.bytes_ref {
            body["blob_ref"] = json!(blob.hash);
        }
        match spill.bytes_ref {
            Some(blob) => {
                let mut r = common::ok(body, EffectSet::default());
                r.bytes_ref = Some(blob);
                r
            }
            None => common::ok(body, EffectSet::default()),
        }
    }

    // ---------------------------------------------------------------------------
    // fs.list
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    pub struct FsListTool {
        spec: ToolSpec,
    }

    impl Default for FsListTool {
        fn default() -> Self {
            Self {
                spec: read_spec(
                    "fs.list",
                    "List directory",
                    "List a directory, respecting .gitignore by default.",
                    json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" },
                            "depth": { "type": "integer", "minimum": 1, "default": 1 },
                            "include_hidden": { "type": "boolean", "default": false }
                        },
                        "required": ["path"],
                        "additionalProperties": false
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": { "path": {"type":"string"}, "entries": {"type":"array"} },
                        "required": ["path", "entries"]
                    })),
                    1024 * 1024,
                ),
            }
        }
    }

    impl Tool for FsListTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing string arg: path", None, Some("/path"));
                };
                let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
                let include_hidden = args.get("include_hidden").and_then(|v| v.as_bool()).unwrap_or(false);

                if !Path::new(path).exists() {
                    return common::coded("NOT_FOUND", format!("no such path: {path}"), true, None);
                }

                let mut entries = Vec::new();
                let walker = ignore::WalkBuilder::new(path)
                    .max_depth(Some(depth))
                    .hidden(!include_hidden)
                    .git_ignore(true)
                    .require_git(false)
                    .git_global(false)
                    .build();
                for entry in walker.flatten() {
                    if entry.depth() == 0 {
                        continue; // skip the root itself
                    }
                    let p = entry.path();
                    entries.push(json!({
                        "name": p.strip_prefix(path).unwrap_or(p).to_string_lossy(),
                        "is_dir": entry.file_type().map(|t| t.is_dir()).unwrap_or(false),
                    }));
                }
                common::ok(json!({ "path": path, "entries": entries }), EffectSet::default())
            })
        }

        fn purity(&self) -> Purity {
            Purity::PureFs
        }
    }

    // ---------------------------------------------------------------------------
    // fs.write
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    pub struct FsWriteTool {
        spec: ToolSpec,
    }

    impl Default for FsWriteTool {
        fn default() -> Self {
            Self {
                spec: write_spec(
                    "fs.write",
                    "Write file",
                    "Write UTF-8 content to an authorized scope. `create_only` guards against clobbering.",
                    json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" },
                            "content": { "type": "string" },
                            "create_dirs": { "type": "boolean", "default": false },
                            "create_only": { "type": "boolean", "default": false }
                        },
                        "required": ["path", "content"],
                        "additionalProperties": false
                    }),
                    Some(json!({
                        "type": "object",
                        "properties": { "path": {"type":"string"}, "bytes": {"type":"integer"} },
                        "required": ["path", "bytes"]
                    })),
                ),
            }
        }
    }

    impl Tool for FsWriteTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing string arg: path", None, Some("/path"));
                };
                let Some(content) = args.get("content").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing string arg: content", None, Some("/content"));
                };
                let create_dirs = args.get("create_dirs").and_then(|v| v.as_bool()).unwrap_or(false);
                let create_only = args.get("create_only").and_then(|v| v.as_bool()).unwrap_or(false);
                let path_buf = PathBuf::from(path);

                if create_only && path_buf.exists() {
                    return common::coded(
                        "CONFLICT",
                        format!("create_only set but {path} already exists"),
                        true,
                        Some("set create_only=false to overwrite, or pick a new path"),
                    );
                }
                if create_dirs {
                    if let Some(parent) = path_buf.parent() {
                        if !parent.as_os_str().is_empty() {
                            if let Err(err) = std::fs::create_dir_all(parent) {
                                return common::coded("TOOL_FAULT", err.to_string(), false, None);
                            }
                        }
                    }
                }
                match std::fs::write(&path_buf, content.as_bytes()) {
                    Ok(()) => common::ok(
                        json!({ "path": path, "bytes": content.len() }),
                        EffectSet { effects: vec![write_effect(path, content.len())] },
                    ),
                    Err(err) => common::coded("TOOL_FAULT", err.to_string(), false, None),
                }
            })
        }

        fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
            Box::pin(async move {
                let path = args.get("path").and_then(|v| v.as_str())?;
                let content = args.get("content").and_then(|v| v.as_str())?;
                Some(EffectSet { effects: vec![write_effect(path, content.len())] })
            })
        }

        fn purity(&self) -> Purity {
            Purity::Impure
        }
    }

    // ---------------------------------------------------------------------------
    // fs.stat
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    pub struct FsStatTool {
        spec: ToolSpec,
    }

    impl Default for FsStatTool {
        fn default() -> Self {
            Self {
                spec: read_spec(
                    "fs.stat",
                    "Stat path",
                    "Return size, mtime, mode, is_dir, and the blake3 content hash of a file.",
                    json!({
                        "type": "object",
                        "properties": { "path": { "type": "string" } },
                        "required": ["path"],
                        "additionalProperties": false
                    }),
                    None,
                    64 * 1024,
                ),
            }
        }
    }

    impl Tool for FsStatTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing string arg: path", None, Some("/path"));
                };
                let meta = match std::fs::metadata(path) {
                    Ok(m) => m,
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                        return common::coded("NOT_FOUND", format!("no such path: {path}"), true, None);
                    }
                    Err(err) => return common::coded("TOOL_FAULT", err.to_string(), false, None),
                };
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs());
                let blob_hash = if meta.is_file() {
                    std::fs::read(path).ok().map(|b| blake3::hash(&b).to_hex().to_string())
                } else {
                    None
                };
                #[cfg(unix)]
                let mode = {
                    use std::os::unix::fs::PermissionsExt;
                    Some(meta.permissions().mode())
                };
                #[cfg(not(unix))]
                let mode: Option<u32> = None;

                common::ok(
                    json!({
                        "path": path,
                        "size_bytes": meta.len(),
                        "is_dir": meta.is_dir(),
                        "is_file": meta.is_file(),
                        "mtime_secs": mtime,
                        "mode": mode,
                        "blob_hash": blob_hash,
                    }),
                    EffectSet { effects: vec![read_effect(path)] },
                )
            })
        }

        fn purity(&self) -> Purity {
            Purity::PureFs
        }
    }

    // ---------------------------------------------------------------------------
    // fs.glob
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    pub struct FsGlobTool {
        spec: ToolSpec,
    }

    impl Default for FsGlobTool {
        fn default() -> Self {
            Self {
                spec: read_spec(
                    "fs.glob",
                    "Glob files",
                    "Match files by glob pattern under a root, respecting .gitignore.",
                    json!({
                        "type": "object",
                        "properties": {
                            "pattern": { "type": "string" },
                            "root": { "type": "string", "default": "." }
                        },
                        "required": ["pattern"],
                        "additionalProperties": false
                    }),
                    None,
                    512 * 1024,
                ),
            }
        }
    }

    impl Tool for FsGlobTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let Some(pattern) = args.get("pattern").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing string arg: pattern", None, Some("/pattern"));
                };
                let root = args.get("root").and_then(|v| v.as_str()).unwrap_or(".");
                let glob = match Glob::new(pattern) {
                    Ok(g) => g,
                    Err(err) => {
                        return common::arg_invalid(
                            format!("invalid glob: {err}"),
                            Some("use a valid glob, e.g. **/*.rs"),
                            Some("/pattern"),
                        )
                    }
                };
                let mut builder = GlobSetBuilder::new();
                builder.add(glob);
                let set = match builder.build() {
                    Ok(s) => s,
                    Err(err) => return common::coded("TOOL_FAULT", err.to_string(), false, None),
                };
                let mut matches = Vec::new();
                for entry in
                    ignore::WalkBuilder::new(root).git_ignore(true).require_git(false).hidden(true).build().flatten()
                {
                    let p = entry.path();
                    let rel = p.strip_prefix(root).unwrap_or(p);
                    if set.is_match(rel) || set.is_match(p) {
                        matches.push(p.to_string_lossy().to_string());
                    }
                }
                matches.sort();
                common::ok(json!({ "pattern": pattern, "root": root, "matches": matches }), EffectSet::default())
            })
        }

        fn purity(&self) -> Purity {
            Purity::PureFs
        }
    }

    // ---------------------------------------------------------------------------
    // fs.watch (bounded poll; long-lived daemon is a documented seam)
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    pub struct FsWatchTool {
        spec: ToolSpec,
    }

    impl Default for FsWatchTool {
        fn default() -> Self {
            let mut spec = read_spec(
                "fs.watch",
                "Watch path",
                "Poll a path for a single change within a bounded window and report what changed. \
                 A persistent watcher is owned by the run loop (documented seam).",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "timeout_ms": { "type": "integer", "minimum": 1, "default": 2000 }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
                None,
                16 * 1024,
            );
            spec.annotations.idempotent = false;
            Self { spec }
        }
    }

    impl Tool for FsWatchTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing string arg: path", None, Some("/path"));
                };
                let timeout_ms = args.get("timeout_ms").and_then(|v| v.as_u64()).unwrap_or(2000).clamp(1, 60_000);
                let initial = snapshot_mtime(path);
                let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
                loop {
                    if std::time::Instant::now() >= deadline {
                        return common::ok(
                            json!({ "path": path, "changed": false, "reason": "timeout" }),
                            EffectSet::default(),
                        );
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    let now = snapshot_mtime(path);
                    if now != initial {
                        return common::ok(
                            json!({ "path": path, "changed": true,
                                    "exists": Path::new(path).exists() }),
                            EffectSet::default(),
                        );
                    }
                }
            })
        }

        fn purity(&self) -> Purity {
            Purity::Impure
        }
    }

    fn snapshot_mtime(path: &str) -> Option<std::time::SystemTime> {
        std::fs::metadata(path).ok().and_then(|m| m.modified().ok())
    }

    // ---------------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------------

    fn parse_range(args: &Value) -> Option<(usize, usize)> {
        let range = args.get("range")?;
        let start = range.get("start_line").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let end = range.get("end_line").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
        Some((start.max(1), end))
    }

    fn slice_lines(text: &str, start: usize, end: usize) -> String {
        text.lines()
            .skip(start.saturating_sub(1))
            .take(end.saturating_sub(start).saturating_add(1))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn read_effect(path: &str) -> Effect {
        Effect {
            kind: EffectKind::Read,
            target: path.to_string(),
            bytes_hash: None,
            risk: RiskLevel::Low,
            metadata: BTreeMap::new(),
        }
    }

    fn write_effect(path: &str, bytes: usize) -> Effect {
        let mut metadata = BTreeMap::new();
        metadata.insert("bytes".to_string(), bytes.to_string());
        Effect { kind: EffectKind::Write, target: path.to_string(), bytes_hash: None, risk: RiskLevel::High, metadata }
    }

    /// Minimal dependency-free base64 (standard alphabet, padded). Used for binary
    /// reads so we never hard-error on non-UTF-8 content.
    fn base64_encode(input: &[u8]) -> String {
        const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        for chunk in input.chunks(3) {
            let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
            let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
            out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
            out.push(if chunk.len() > 1 { ALPHABET[((n >> 6) & 63) as usize] as char } else { '=' });
            out.push(if chunk.len() > 2 { ALPHABET[(n & 63) as usize] as char } else { '=' });
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::permission::{PermissionPolicy, StaticPermissionEngine};
        use hide_core::persistence::InMemoryBlobStore;
        use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry};

        fn dispatcher(reg: Arc<ToolRegistry>) -> ToolDispatcher {
            ToolDispatcher::new(
                reg,
                Arc::new(StaticPermissionEngine::new(PermissionPolicy {
                    default_decision: hide_core::types::Decision::Allow,
                    rules: Vec::new(),
                    risk_gates: Vec::new(),
                })),
            )
        }

        fn tmp(name: &str) -> PathBuf {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "hide_fs_{}_{}_{}_{}",
                name,
                std::process::id(),
                hide_core::ids::now_ms(),
                N.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        #[tokio::test]
        async fn read_supports_line_range() {
            let dir = tmp("range");
            let file = dir.join("f.txt");
            std::fs::write(&file, "a\nb\nc\nd\ne").unwrap();
            let reg = Arc::new(ToolRegistry::default());
            reg.register(FsReadTool::default());
            let d = dispatcher(reg);
            let r = d
                .dispatch(ToolCall::new(
                    "fs.read",
                    json!({ "path": file.to_string_lossy(), "range": {"start_line":2,"end_line":3} }),
                ))
                .await
                .unwrap();
            assert_eq!(r.structured_content.unwrap()["content"], "b\nc");
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn read_over_cap_spills_to_blob() {
            let dir = tmp("spill");
            let file = dir.join("big.txt");
            std::fs::write(&file, "x".repeat(20_000)).unwrap();
            let blobs: Arc<dyn BlobStore> = Arc::new(InMemoryBlobStore::default());
            let reg = Arc::new(ToolRegistry::default());
            let mut tool = FsReadTool::with_config(FsConfig { blobs: Some(blobs.clone()) });
            // shrink the cap so 20k spills
            tool.spec.output_cap_bytes = 1000;
            reg.register(tool);
            let d = dispatcher(reg);
            let r = d.dispatch(ToolCall::new("fs.read", json!({ "path": file.to_string_lossy() }))).await.unwrap();
            assert!(r.bytes_ref.is_some());
            let sc = r.structured_content.unwrap();
            assert_eq!(sc["truncated"], true);
            assert_eq!(sc["bytes"], 20_000);
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn read_binary_falls_back_to_base64() {
            let dir = tmp("bin");
            let file = dir.join("b.bin");
            std::fs::write(&file, [0u8, 1, 2, 255, 0]).unwrap();
            let reg = Arc::new(ToolRegistry::default());
            reg.register(FsReadTool::default());
            let d = dispatcher(reg);
            let r = d.dispatch(ToolCall::new("fs.read", json!({ "path": file.to_string_lossy() }))).await.unwrap();
            assert_eq!(r.structured_content.unwrap()["encoding"], "base64");
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn stat_returns_blake3_hash() {
            let dir = tmp("stat");
            let file = dir.join("s.txt");
            std::fs::write(&file, "hello").unwrap();
            let reg = Arc::new(ToolRegistry::default());
            reg.register(FsStatTool::default());
            let d = dispatcher(reg);
            let r = d.dispatch(ToolCall::new("fs.stat", json!({ "path": file.to_string_lossy() }))).await.unwrap();
            let sc = r.structured_content.unwrap();
            assert_eq!(sc["size_bytes"], 5);
            let expected = blake3::hash(b"hello").to_hex().to_string();
            assert_eq!(sc["blob_hash"], expected);
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn glob_matches_and_respects_gitignore() {
            let dir = tmp("glob");
            std::fs::write(dir.join("a.rs"), "").unwrap();
            std::fs::write(dir.join("b.txt"), "").unwrap();
            std::fs::write(dir.join(".gitignore"), "ignored.rs\n").unwrap();
            std::fs::write(dir.join("ignored.rs"), "").unwrap();
            let reg = Arc::new(ToolRegistry::default());
            reg.register(FsGlobTool::default());
            let d = dispatcher(reg);
            let r = d
                .dispatch(ToolCall::new("fs.glob", json!({ "pattern": "**/*.rs", "root": dir.to_string_lossy() })))
                .await
                .unwrap();
            let matches = r.structured_content.unwrap()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<Vec<_>>();
            assert!(matches.iter().any(|m| m.ends_with("a.rs")));
            assert!(!matches.iter().any(|m| m.ends_with("ignored.rs")));
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn write_create_only_conflicts() {
            let dir = tmp("conflict");
            let file = dir.join("c.txt");
            std::fs::write(&file, "existing").unwrap();
            let reg = Arc::new(ToolRegistry::default());
            reg.register(FsWriteTool::default());
            let d = dispatcher(reg);
            let r = d
                .dispatch(ToolCall::new(
                    "fs.write",
                    json!({ "path": file.to_string_lossy(), "content": "new", "create_only": true }),
                ))
                .await
                .unwrap();
            assert_eq!(r.error.unwrap().code, "CONFLICT");
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
#[rustfmt::skip]
pub mod git {
    //! Git tools (ch.03 §4.6.6): status, diff, log, commit, and the worktree trio
    //! (`git.worktree.add/remove/list`) — the agent-isolation primitive (§4.9.3).
    //!
    //! All of these shell out to `git` and honor the `EXEC_NONZERO`-is-data discipline:
    //! a non-zero git exit is `ok:true` + `exit_code`, so the agent reads the message
    //! (e.g. "nothing to commit") rather than treating it as a tool fault (§4.2.3).
    //! Only a failure to *spawn* git is `ok:false`.

    use crate::common;
    use crate::spec_helpers::{git_read_spec, git_write_spec};
    use futures::future::BoxFuture;
    use hide_core::persistence::BlobStore;
    use hide_core::tool::{Purity, Tool, ToolCtx, ToolResult, ToolSpec};
    use hide_core::types::{Effect, EffectKind, EffectSet, RiskLevel};
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::process::Stdio;
    use std::sync::Arc;
    use tokio::process::Command;

    #[derive(Clone, Default)]
    pub struct GitConfig {
        pub blobs: Option<Arc<dyn BlobStore>>,
    }

    /// Run a git subcommand in `cwd` and project to the canonical process result.
    /// A non-zero exit is data (EXEC_NONZERO), not a fault.
    async fn run_git(cwd: &str, args: &[String], cap_bytes: usize, blobs: Option<&Arc<dyn BlobStore>>) -> ToolResult {
        let mut command = Command::new("git");
        command
            .args(args)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let output = match command.output().await {
            Ok(o) => o,
            Err(err) => return common::spawn_fault(format!("failed to spawn git: {err}")),
        };
        let exit = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let mut result = common::project_process_output(exit, stdout, stderr, cap_bytes, blobs);
        if let Some(sc) = result.structured_content.as_mut() {
            sc["argv"] = json!(args);
            sc["cwd"] = json!(cwd);
        }
        result
    }

    fn cwd_of(args: &Value) -> String {
        args.get("cwd").and_then(|v| v.as_str()).unwrap_or(".").to_string()
    }

    macro_rules! git_read_tool {
        ($ty:ident, $name:literal, $title:literal, $desc:literal, $schema:expr, $build:expr) => {
            #[derive(Clone)]
            pub struct $ty {
                spec: ToolSpec,
                config: GitConfig,
            }
            impl Default for $ty {
                fn default() -> Self {
                    Self { spec: git_read_spec($name, $title, $desc, $schema), config: GitConfig::default() }
                }
            }
            impl $ty {
                pub fn with_config(config: GitConfig) -> Self {
                    Self { config, ..Self::default() }
                }
            }
            impl Tool for $ty {
                fn spec(&self) -> &ToolSpec {
                    &self.spec
                }
                fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
                    Box::pin(async move {
                        let cwd = cwd_of(&args);
                        let argv: Vec<String> = $build(&args);
                        run_git(&cwd, &argv, ctx.output_cap_bytes as usize, self.config.blobs.as_ref()).await
                    })
                }
                fn purity(&self) -> Purity {
                    Purity::PureFs
                }
            }
        };
    }

    git_read_tool!(
        GitStatusTool,
        "git.status",
        "Git status",
        "Porcelain `git status --short --branch`.",
        json!({"type":"object","properties":{"cwd":{"type":"string"}},"required":[],"additionalProperties":false}),
        |_args: &Value| vec!["status".to_string(), "--short".to_string(), "--branch".to_string()]
    );

    git_read_tool!(
        GitDiffTool,
        "git.diff",
        "Git diff",
        "Unified diff; optional ref, --staged, and path filter.",
        json!({"type":"object","properties":{
            "cwd":{"type":"string"},"ref":{"type":"string"},
            "staged":{"type":"boolean"},"path":{"type":"string"}
        },"required":[],"additionalProperties":false}),
        |args: &Value| {
            let mut v = vec!["diff".to_string()];
            if args.get("staged").and_then(|x| x.as_bool()).unwrap_or(false) {
                v.push("--staged".to_string());
            }
            if let Some(r) = args.get("ref").and_then(|x| x.as_str()) {
                // Option-injection guard: a caller-supplied ref like "--output=FILE"
                // would otherwise be honored by git as an option and WRITE a file.
                // --end-of-options forces git to parse it as a revision (failing safely
                // if it is not one).
                v.push("--end-of-options".to_string());
                v.push(r.to_string());
            }
            if let Some(p) = args.get("path").and_then(|x| x.as_str()) {
                v.push("--".to_string());
                v.push(p.to_string());
            }
            v
        }
    );

    git_read_tool!(
        GitLogTool,
        "git.log",
        "Git log",
        "History as oneline entries; optional ref, max count, path.",
        json!({"type":"object","properties":{
            "cwd":{"type":"string"},"ref":{"type":"string"},
            "max":{"type":"integer","minimum":1},"path":{"type":"string"}
        },"required":[],"additionalProperties":false}),
        |args: &Value| {
            let mut v = vec!["log".to_string(), "--oneline".to_string(), "--decorate".to_string()];
            let max = args.get("max").and_then(|x| x.as_u64()).unwrap_or(20);
            v.push(format!("-n{max}"));
            if let Some(r) = args.get("ref").and_then(|x| x.as_str()) {
                // Option-injection guard (see git.diff): treat the ref as a revision,
                // never an option like "--output=FILE".
                v.push("--end-of-options".to_string());
                v.push(r.to_string());
            }
            if let Some(p) = args.get("path").and_then(|x| x.as_str()) {
                v.push("--".to_string());
                v.push(p.to_string());
            }
            v
        }
    );

    git_read_tool!(
        GitWorktreeListTool,
        "git.worktree.list",
        "List worktrees",
        "Enumerate git worktrees (porcelain).",
        json!({"type":"object","properties":{"cwd":{"type":"string"}},"required":[],"additionalProperties":false}),
        |_args: &Value| vec!["worktree".to_string(), "list".to_string(), "--porcelain".to_string()]
    );

    // ---------------------------------------------------------------------------
    // git.commit (write; ask policy). The message must NOT add AI attribution.
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    pub struct GitCommitTool {
        spec: ToolSpec,
        config: GitConfig,
    }

    impl Default for GitCommitTool {
        fn default() -> Self {
            Self {
                spec: git_write_spec(
                    "git.commit",
                    "Git commit",
                    "Stage given paths (or all) and commit with a message. No AI attribution is added.",
                    json!({
                        "type":"object",
                        "properties":{
                            "cwd":{"type":"string"},
                            "message":{"type":"string"},
                            "paths":{"type":"array","items":{"type":"string"}},
                            "amend":{"type":"boolean","default":false}
                        },
                        "required":["message"],
                        "additionalProperties":false
                    }),
                ),
                config: GitConfig::default(),
            }
        }
    }

    impl GitCommitTool {
        pub fn with_config(config: GitConfig) -> Self {
            Self { config, ..Self::default() }
        }
    }

    impl Tool for GitCommitTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let Some(message) = args.get("message").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing message", None, Some("/message"));
                };
                let cwd = cwd_of(&args);
                let cap = ctx.output_cap_bytes as usize;
                // Stage.
                let paths: Vec<String> = args
                    .get("paths")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let mut add = vec!["add".to_string()];
                if paths.is_empty() {
                    add.push("-A".to_string());
                } else {
                    add.extend(paths.clone());
                }
                let staged = run_git(&cwd, &add, cap, self.config.blobs.as_ref()).await;
                if !staged.ok {
                    return staged; // spawn fault
                }
                // Commit.
                let mut commit = vec!["commit".to_string(), "-m".to_string(), message.to_string()];
                if args.get("amend").and_then(|v| v.as_bool()).unwrap_or(false) {
                    commit.push("--amend".to_string());
                }
                let mut result = run_git(&cwd, &commit, cap, self.config.blobs.as_ref()).await;
                result.effects = EffectSet {
                    effects: vec![Effect {
                        kind: EffectKind::Write,
                        target: format!("git.commit:{cwd}"),
                        bytes_hash: None,
                        risk: RiskLevel::Medium,
                        metadata: {
                            let mut m = BTreeMap::new();
                            m.insert("message".to_string(), message.to_string());
                            m
                        },
                    }],
                };
                result
            })
        }

        fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
            Box::pin(async move {
                Some(EffectSet {
                    effects: vec![Effect {
                        kind: EffectKind::Write,
                        target: format!("git.commit:{}", cwd_of(args)),
                        bytes_hash: None,
                        risk: RiskLevel::Medium,
                        metadata: BTreeMap::new(),
                    }],
                })
            })
        }

        fn purity(&self) -> Purity {
            Purity::Impure
        }
    }

    // ---------------------------------------------------------------------------
    // git.worktree.add / remove (the isolation primitive)
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    pub struct GitWorktreeAddTool {
        spec: ToolSpec,
        config: GitConfig,
    }

    impl Default for GitWorktreeAddTool {
        fn default() -> Self {
            Self {
                spec: git_write_spec(
                    "git.worktree.add",
                    "Add worktree",
                    "Create a git worktree at `path` on a new branch — the agent-isolation primitive.",
                    json!({
                        "type":"object",
                        "properties":{
                            "cwd":{"type":"string"},
                            "path":{"type":"string"},
                            "branch":{"type":"string"},
                            "from":{"type":"string"}
                        },
                        "required":["path","branch"],
                        "additionalProperties":false
                    }),
                ),
                config: GitConfig::default(),
            }
        }
    }

    impl GitWorktreeAddTool {
        pub fn with_config(config: GitConfig) -> Self {
            Self { config, ..Self::default() }
        }
    }

    impl Tool for GitWorktreeAddTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing path", None, Some("/path"));
                };
                let Some(branch) = args.get("branch").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing branch", None, Some("/branch"));
                };
                let cwd = cwd_of(&args);
                let mut argv = vec![
                    "worktree".to_string(),
                    "add".to_string(),
                    "-b".to_string(),
                    branch.to_string(),
                    path.to_string(),
                ];
                if let Some(from) = args.get("from").and_then(|v| v.as_str()) {
                    argv.push(from.to_string());
                }
                let mut result = run_git(&cwd, &argv, ctx.output_cap_bytes as usize, self.config.blobs.as_ref()).await;
                if let Some(sc) = result.structured_content.as_mut() {
                    sc["worktree_id"] = json!(branch);
                    sc["root"] = json!(path);
                }
                result.effects = EffectSet {
                    effects: vec![Effect {
                        kind: EffectKind::Write,
                        target: path.to_string(),
                        bytes_hash: None,
                        risk: RiskLevel::Medium,
                        metadata: BTreeMap::new(),
                    }],
                };
                result
            })
        }

        fn purity(&self) -> Purity {
            Purity::Impure
        }
    }

    #[derive(Clone)]
    pub struct GitWorktreeRemoveTool {
        spec: ToolSpec,
        config: GitConfig,
    }

    impl Default for GitWorktreeRemoveTool {
        fn default() -> Self {
            Self {
                spec: git_write_spec(
                    "git.worktree.remove",
                    "Remove worktree",
                    "Remove a git worktree by path (optionally force).",
                    json!({
                        "type":"object",
                        "properties":{
                            "cwd":{"type":"string"},
                            "path":{"type":"string"},
                            "force":{"type":"boolean","default":false}
                        },
                        "required":["path"],
                        "additionalProperties":false
                    }),
                ),
                config: GitConfig::default(),
            }
        }
    }

    impl GitWorktreeRemoveTool {
        pub fn with_config(config: GitConfig) -> Self {
            Self { config, ..Self::default() }
        }
    }

    impl Tool for GitWorktreeRemoveTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                    return common::arg_invalid("missing path", None, Some("/path"));
                };
                let cwd = cwd_of(&args);
                let mut argv = vec!["worktree".to_string(), "remove".to_string()];
                if args.get("force").and_then(|v| v.as_bool()).unwrap_or(false) {
                    argv.push("--force".to_string());
                }
                argv.push(path.to_string());
                run_git(&cwd, &argv, ctx.output_cap_bytes as usize, self.config.blobs.as_ref()).await
            })
        }

        fn purity(&self) -> Purity {
            Purity::Impure
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::permission::{PermissionPolicy, StaticPermissionEngine};
        use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry};
        use std::path::PathBuf;

        fn dispatcher(reg: Arc<ToolRegistry>) -> ToolDispatcher {
            ToolDispatcher::new(
                reg,
                Arc::new(StaticPermissionEngine::new(PermissionPolicy {
                    default_decision: hide_core::types::Decision::Allow,
                    rules: Vec::new(),
                    risk_gates: Vec::new(),
                })),
            )
        }

        fn unique() -> String {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            format!("{}_{}_{}", std::process::id(), hide_core::ids::now_ms(), N.fetch_add(1, Ordering::SeqCst))
        }

        async fn init_repo() -> PathBuf {
            let dir = std::env::temp_dir().join(format!("hide_git_{}", unique()));
            std::fs::create_dir_all(&dir).unwrap();
            for argv in [vec!["init", "-q"], vec!["config", "user.email", "t@t.t"], vec!["config", "user.name", "t"]] {
                Command::new("git").args(&argv).current_dir(&dir).output().await.unwrap();
            }
            dir
        }

        #[tokio::test]
        async fn git_status_clean_is_ok() {
            let dir = init_repo().await;
            let reg = Arc::new(ToolRegistry::default());
            reg.register(GitStatusTool::default());
            let d = dispatcher(reg);
            let r = d.dispatch(ToolCall::new("git.status", json!({ "cwd": dir.to_string_lossy() }))).await.unwrap();
            assert!(r.ok);
            assert_eq!(r.exit_code, Some(0));
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn git_nonzero_exit_outside_repo_is_data_not_fault() {
            // running git diff outside a repo exits non-zero; that must be ok:true.
            let dir = std::env::temp_dir().join(format!("hide_nogit_{}", unique()));
            std::fs::create_dir_all(&dir).unwrap();
            let reg = Arc::new(ToolRegistry::default());
            reg.register(GitDiffTool::default());
            let d = dispatcher(reg);
            let r = d.dispatch(ToolCall::new("git.diff", json!({ "cwd": dir.to_string_lossy() }))).await.unwrap();
            // git exits 128 outside a repo → EXEC_NONZERO is data, ok stays true.
            assert!(r.ok, "non-zero git exit must be ok:true (EXEC_NONZERO)");
            assert_ne!(r.exit_code, Some(0));
            assert!(r.error.is_none());
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn git_commit_then_log() {
            let dir = init_repo().await;
            std::fs::write(dir.join("a.txt"), "hello").unwrap();
            let reg = Arc::new(ToolRegistry::default());
            reg.register(GitCommitTool::default());
            reg.register(GitLogTool::default());
            let d = dispatcher(reg);
            let commit = d
                .dispatch(ToolCall::new("git.commit", json!({ "cwd": dir.to_string_lossy(), "message": "init" })))
                .await
                .unwrap();
            assert!(commit.ok);
            assert_eq!(commit.exit_code, Some(0));
            let log = d.dispatch(ToolCall::new("git.log", json!({ "cwd": dir.to_string_lossy() }))).await.unwrap();
            assert!(log.structured_content.unwrap()["stdout"].as_str().unwrap().contains("init"));
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn git_diff_ref_cannot_inject_write_option() {
            // A malicious ref like "--output=FILE" must NOT be honored as a git option
            // (which would create/truncate an arbitrary file). Regression guard for the
            // option-injection the read-only auto-dispatch review found.
            let dir = init_repo().await;
            std::fs::write(dir.join("a.txt"), "dirty").unwrap();
            let evil = dir.join("evil_written.txt");
            let reg = Arc::new(ToolRegistry::default());
            reg.register(GitDiffTool::default());
            let d = dispatcher(reg);
            let _ = d
                .dispatch(ToolCall::new(
                    "git.diff",
                    json!({
                        "cwd": dir.to_string_lossy(),
                        "ref": format!("--output={}", evil.to_string_lossy()),
                    }),
                ))
                .await
                .unwrap();
            assert!(!evil.exists(), "git.diff ref must not inject --output and write a file");
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn git_diff_normal_ref_still_works() {
            // --end-of-options must not break a legitimate ref.
            let dir = init_repo().await;
            std::fs::write(dir.join("a.txt"), "x").unwrap();
            let reg = Arc::new(ToolRegistry::default());
            reg.register(GitCommitTool::default());
            reg.register(GitDiffTool::default());
            let d = dispatcher(reg);
            let _ = d
                .dispatch(ToolCall::new("git.commit", json!({ "cwd": dir.to_string_lossy(), "message": "c" })))
                .await
                .unwrap();
            let r = d
                .dispatch(ToolCall::new("git.diff", json!({ "cwd": dir.to_string_lossy(), "ref": "HEAD" })))
                .await
                .unwrap();
            assert!(r.ok, "a normal ref diff must still work: {:?}", r.error);
            let _ = std::fs::remove_dir_all(dir);
        }

        #[tokio::test]
        async fn git_worktree_add_list_remove() {
            let dir = init_repo().await;
            std::fs::write(dir.join("a.txt"), "hello").unwrap();
            Command::new("git").args(["add", "-A"]).current_dir(&dir).output().await.unwrap();
            Command::new("git").args(["commit", "-qm", "init"]).current_dir(&dir).output().await.unwrap();
            let wt = dir.join("wt");
            let reg = Arc::new(ToolRegistry::default());
            reg.register(GitWorktreeAddTool::default());
            reg.register(GitWorktreeListTool::default());
            reg.register(GitWorktreeRemoveTool::default());
            let d = dispatcher(reg);
            let add = d
                .dispatch(ToolCall::new(
                    "git.worktree.add",
                    json!({ "cwd": dir.to_string_lossy(), "path": wt.to_string_lossy(), "branch": "feat" }),
                ))
                .await
                .unwrap();
            assert!(add.ok && add.exit_code == Some(0), "{:?}", add.structured_content);
            let list =
                d.dispatch(ToolCall::new("git.worktree.list", json!({ "cwd": dir.to_string_lossy() }))).await.unwrap();
            assert!(list.structured_content.unwrap()["stdout"].as_str().unwrap().contains("feat"));
            let remove = d
                .dispatch(ToolCall::new(
                    "git.worktree.remove",
                    json!({ "cwd": dir.to_string_lossy(), "path": wt.to_string_lossy(), "force": true }),
                ))
                .await
                .unwrap();
            assert!(remove.ok);
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
#[rustfmt::skip]
pub mod mcp {
    //! MCP host/client (ch.03 §4.10), pinned to protocol revision `2025-11-25`.
    //!
    //! A real JSON-RPC 2.0 client that speaks both standard transports:
    //!
    //! * **stdio** — spawn the server subprocess (`tokio::process`), newline-delimited
    //!   JSON-RPC over its stdin/stdout (stderr is logging).
    //! * **Streamable HTTP** — single endpoint, `reqwest` POST per message, carrying
    //!   the `MCP-Session-Id` + `MCP-Protocol-Version` headers (2025-11-25).
    //!
    //! It performs the `initialize` handshake, `tools/list`, and `tools/call`, and
    //! **maps each MCP `Tool` → a HIDE `ToolSpec` carrying its annotations as
    //! UNTRUSTED provenance** (a server claiming `read_only` does NOT relax HIDE
    //! policy — §4.9.4). Discovered tools become `McpProxyTool`s registered into the
    //! standard registry; calling one runs through the full HIDE dispatcher.

    use crate::common;
    use anyhow::{anyhow, Context, Result};
    use futures::future::BoxFuture;
    use hide_core::tool::{Purity, Tool, ToolAnnotations, ToolContent, ToolCtx, ToolResult, ToolSpec, ToolStatus};
    use hide_core::types::EffectSet;
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::process::Stdio;
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::Arc;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::{Child, ChildStdin, ChildStdout};
    use tokio::sync::Mutex;

    /// MCP protocol revision this client negotiates.
    pub const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

    // ---------------------------------------------------------------------------
    // configuration / descriptors
    // ---------------------------------------------------------------------------

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct McpServerDescriptor {
        pub id: String,
        pub transport: McpTransport,
        #[serde(default = "default_trust")]
        pub trust: String,
    }

    fn default_trust() -> String {
        "third-party".to_string()
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum McpTransport {
        Stdio {
            command: String,
            #[serde(default)]
            args: Vec<String>,
        },
        StreamableHttp {
            endpoint: String,
        },
    }

    /// An MCP `Tool` as returned by `tools/list`.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct McpTool {
        pub name: String,
        #[serde(default)]
        pub title: Option<String>,
        #[serde(default)]
        pub description: Option<String>,
        #[serde(rename = "inputSchema")]
        pub input_schema: Value,
        #[serde(default, rename = "outputSchema")]
        pub output_schema: Option<Value>,
        #[serde(default)]
        pub annotations: Option<McpAnnotations>,
    }

    /// MCP tool annotations — hints with telling defaults (§3.2). HIDE treats these
    /// as UNTRUSTED: they inform the model but never auto-relax policy.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
    pub struct McpAnnotations {
        #[serde(rename = "readOnlyHint", default)]
        pub read_only_hint: Option<bool>,
        #[serde(rename = "destructiveHint", default)]
        pub destructive_hint: Option<bool>,
        #[serde(rename = "idempotentHint", default)]
        pub idempotent_hint: Option<bool>,
        #[serde(rename = "openWorldHint", default)]
        pub open_world_hint: Option<bool>,
    }

    // ---------------------------------------------------------------------------
    // projections (kept stable for siblings)
    // ---------------------------------------------------------------------------

    /// Map an MCP `Tool` → a HIDE `ToolSpec`. Annotations are carried but UNTRUSTED:
    /// the spec is namespaced `mcp:<server>/<name>` and `capabilities_required`
    /// reflects the host's own classification, not the server's claim.
    pub fn mcp_tool_to_hide_spec(server_id: &str, tool: McpTool) -> ToolSpec {
        // The MCP spec's annotation defaults are deliberately pessimistic; we keep
        // them, but we do NOT let a server claim read_only to widen its policy.
        let ann = tool.annotations.unwrap_or_default();
        let annotations = ToolAnnotations {
            read_only: ann.read_only_hint.unwrap_or(false),
            destructive: ann.destructive_hint.unwrap_or(true),
            idempotent: ann.idempotent_hint.unwrap_or(false),
            open_world: ann.open_world_hint.unwrap_or(true),
        };
        ToolSpec {
            name: format!("mcp:{server_id}/{}", tool.name),
            title: tool.title.unwrap_or_else(|| tool.name.clone()),
            version: "0.1.0".to_string(),
            wire_version: 1,
            description: tool.description.unwrap_or_default(),
            input_schema: tool.input_schema,
            output_schema: tool.output_schema,
            annotations,
            // Untrusted external tool: the only capability the host grants by default
            // is the bridged-call capability; real scopes come from the grant ledger.
            capabilities_required: vec!["mcp.call".to_string()],
            output_cap_bytes: 1024 * 1024,
            timeout_ms: 30_000,
        }
    }

    /// Project a HIDE `ToolResult` to an MCP `CallToolResult` (for HIDE-as-server).
    /// `ok` ↔ `!isError`; `structured_content` ↔ `structuredContent`.
    pub fn hide_result_to_mcp(result: &ToolResult) -> Value {
        json!({
            "isError": result.status != ToolStatus::Ok,
            "structuredContent": result.structured_content,
            "content": result.content,
        })
    }

    /// Project an MCP `CallToolResult` → a HIDE `ToolResult`. `isError:true` is a
    /// *tool execution* error surfaced as data (the model can self-correct, §4.10).
    pub fn mcp_result_to_hide(value: &Value) -> ToolResult {
        let is_error = value.get("isError").and_then(|v| v.as_bool()).unwrap_or(false);
        let structured = value.get("structuredContent").cloned();
        let content = value
            .get("content")
            .and_then(|v| v.as_array())
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|b| match b.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            b.get("text").and_then(|t| t.as_str()).map(|t| ToolContent::Text { text: t.to_string() })
                        }
                        _ => Some(ToolContent::Json { value: b.clone() }),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        ToolResult {
            call_id: hide_core::ids::ToolCallId::new(),
            ok: !is_error,
            status: if is_error { ToolStatus::ToolError } else { ToolStatus::Ok },
            content,
            structured_content: structured,
            bytes_ref: None,
            exit_code: None,
            effects: EffectSet::default(),
            provenance: "tool-output".to_string(), // TT8: MCP output is untrusted data
            stats: Default::default(),
            error: if is_error {
                Some(hide_core::tool::ToolError::new("TOOL_FAULT", "mcp tool reported isError", true))
            } else {
                None
            },
        }
    }

    // ---------------------------------------------------------------------------
    // JSON-RPC transport abstraction
    // ---------------------------------------------------------------------------

    /// A live MCP client connection. Owns the transport and the request-id counter.
    pub struct McpClient {
        server_id: String,
        transport: ClientTransport,
        next_id: AtomicI64,
        protocol_version: String,
    }

    enum ClientTransport {
        /// Boxed because the stdio transport carries a `Child` + two large buffered
        /// handles, dwarfing the `Http` variant (`clippy::large_enum_variant`).
        Stdio(Box<StdioTransport>),
        Http {
            client: reqwest::Client,
            endpoint: String,
            session_id: Mutex<Option<String>>,
        },
    }

    /// State for an stdio MCP transport: the live child plus its locked I/O handles.
    struct StdioTransport {
        _child: Child,
        stdin: Mutex<ChildStdin>,
        stdout: Mutex<BufReader<ChildStdout>>,
    }

    impl McpClient {
        /// Connect to a server over the descriptor's transport and run `initialize`.
        pub async fn connect(desc: &McpServerDescriptor) -> Result<Self> {
            let transport = match &desc.transport {
                McpTransport::Stdio { command, args } => {
                    // kill_on_drop so dropping the client tears down the subprocess
                    // instead of leaking it (the registry owns the client via the proxy
                    // tools; without this a discarded server keeps running).
                    let mut child = tokio::process::Command::new(command)
                        .args(args)
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::inherit())
                        .kill_on_drop(true)
                        .spawn()
                        .with_context(|| format!("spawning MCP server {command}"))?;
                    let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
                    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
                    ClientTransport::Stdio(Box::new(StdioTransport {
                        _child: child,
                        stdin: Mutex::new(stdin),
                        stdout: Mutex::new(BufReader::new(stdout)),
                    }))
                }
                McpTransport::StreamableHttp { endpoint } => ClientTransport::Http {
                    // A request timeout so a server that accepts the POST but never
                    // responds cannot hang a call forever.
                    client: reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(30))
                        .build()
                        .unwrap_or_else(|_| reqwest::Client::new()),
                    endpoint: endpoint.clone(),
                    session_id: Mutex::new(None),
                },
            };
            let client = Self {
                server_id: desc.id.clone(),
                transport,
                next_id: AtomicI64::new(1),
                protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            };
            client.initialize().await?;
            Ok(client)
        }

        pub fn server_id(&self) -> &str {
            &self.server_id
        }

        async fn initialize(&self) -> Result<Value> {
            let params = json!({
                "protocolVersion": self.protocol_version,
                "capabilities": { "roots": { "listChanged": true }, "sampling": {}, "elicitation": {} },
                "clientInfo": { "name": "hide", "version": env!("CARGO_PKG_VERSION") }
            });
            let result = self.request("initialize", params).await?;
            // After initialize, the client SHOULD send the `initialized` notification.
            self.notify("notifications/initialized", json!({})).await?;
            Ok(result)
        }

        /// `tools/list` → bridged `ToolSpec`s.
        pub async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
            let result = self.request("tools/list", json!({})).await?;
            let tools =
                result.get("tools").and_then(|v| v.as_array()).ok_or_else(|| anyhow!("tools/list: missing tools[]"))?;
            let mut specs = Vec::new();
            for t in tools {
                let tool: McpTool = serde_json::from_value(t.clone()).context("decoding MCP tool")?;
                specs.push(mcp_tool_to_hide_spec(&self.server_id, tool));
            }
            Ok(specs)
        }

        /// `tools/call` → bridged `ToolResult`. The `tool` arg is the *bare* MCP tool
        /// name (without the `mcp:<server>/` prefix HIDE adds to the spec).
        pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolResult> {
            let bare = name.strip_prefix(&format!("mcp:{}/", self.server_id)).unwrap_or(name);
            let result = self.request("tools/call", json!({ "name": bare, "arguments": arguments })).await?;
            Ok(mcp_result_to_hide(&result))
        }

        /// Send a JSON-RPC request and await its response.
        async fn request(&self, method: &str, params: Value) -> Result<Value> {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            let req = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
            let response = match &self.transport {
                ClientTransport::Stdio(t) => {
                    let StdioTransport { stdin, stdout, .. } = t.as_ref();
                    let mut line = serde_json::to_string(&req)?;
                    line.push('\n');
                    {
                        let mut w = stdin.lock().await;
                        w.write_all(line.as_bytes()).await?;
                        w.flush().await?;
                    }
                    // Read lines until we get one with our id (skip notifications).
                    let mut reader = stdout.lock().await;
                    loop {
                        let mut buf = String::new();
                        let n = reader.read_line(&mut buf).await?;
                        if n == 0 {
                            return Err(anyhow!("MCP stdio closed before response"));
                        }
                        let trimmed = buf.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let v: Value = match serde_json::from_str(trimmed) {
                            Ok(v) => v,
                            Err(_) => continue, // log line on stdout; ignore
                        };
                        if v.get("id").and_then(|x| x.as_i64()) == Some(id) {
                            break v;
                        }
                    }
                }
                ClientTransport::Http { client, endpoint, session_id } => {
                    let mut builder = client
                        .post(endpoint)
                        .header("Content-Type", "application/json")
                        .header("Accept", "application/json, text/event-stream")
                        .header("MCP-Protocol-Version", &self.protocol_version);
                    if let Some(sid) = session_id.lock().await.as_ref() {
                        builder = builder.header("MCP-Session-Id", sid);
                    }
                    let resp = builder.json(&req).send().await?;
                    // Capture a server-assigned session id on the initialize response.
                    if let Some(sid) = resp.headers().get("MCP-Session-Id") {
                        if let Ok(s) = sid.to_str() {
                            *session_id.lock().await = Some(s.to_string());
                        }
                    }
                    let text = resp.text().await?;
                    parse_http_jsonrpc(&text, id)?
                }
            };
            if let Some(err) = response.get("error") {
                return Err(anyhow!("JSON-RPC error: {err}"));
            }
            Ok(response.get("result").cloned().unwrap_or(Value::Null))
        }

        /// Send a JSON-RPC notification (no id, no response expected).
        async fn notify(&self, method: &str, params: Value) -> Result<()> {
            let note = json!({ "jsonrpc": "2.0", "method": method, "params": params });
            match &self.transport {
                ClientTransport::Stdio(t) => {
                    let stdin = &t.stdin;
                    let mut line = serde_json::to_string(&note)?;
                    line.push('\n');
                    let mut w = stdin.lock().await;
                    w.write_all(line.as_bytes()).await?;
                    w.flush().await?;
                }
                ClientTransport::Http { client, endpoint, session_id } => {
                    let mut builder = client
                        .post(endpoint)
                        .header("Content-Type", "application/json")
                        .header("MCP-Protocol-Version", &self.protocol_version);
                    if let Some(sid) = session_id.lock().await.as_ref() {
                        builder = builder.header("MCP-Session-Id", sid);
                    }
                    let _ = builder.json(&note).send().await?;
                }
            }
            Ok(())
        }
    }

    /// Parse an HTTP JSON-RPC response body, which may be either a plain JSON object
    /// or an SSE stream (`data: {…}` lines). Returns the message matching `id`.
    fn parse_http_jsonrpc(text: &str, id: i64) -> Result<Value> {
        let trimmed = text.trim_start();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            let v: Value = serde_json::from_str(trimmed).context("decoding JSON-RPC body")?;
            return Ok(v);
        }
        // SSE: scan `data:` lines for the matching id.
        let mut fallback = None;
        for line in text.lines() {
            let line = line.trim();
            if let Some(payload) = line.strip_prefix("data:") {
                let payload = payload.trim();
                if payload == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(payload) {
                    if v.get("id").and_then(|x| x.as_i64()) == Some(id) {
                        return Ok(v);
                    }
                    fallback.get_or_insert(v);
                }
            }
        }
        fallback.ok_or_else(|| anyhow!("no JSON-RPC message found in HTTP response"))
    }

    // ---------------------------------------------------------------------------
    // proxy tool — a discovered MCP tool registered into the HIDE registry
    // ---------------------------------------------------------------------------

    /// A registered proxy over a bridged MCP tool. Calling it runs `tools/call` on
    /// the live client. Subject to the full HIDE permission model via the dispatcher.
    pub struct McpProxyTool {
        spec: ToolSpec,
        client: Arc<McpClient>,
    }

    impl McpProxyTool {
        pub fn new(spec: ToolSpec, client: Arc<McpClient>) -> Self {
            Self { spec, client }
        }
    }

    impl Tool for McpProxyTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            let name = self.spec.name.clone();
            let client = self.client.clone();
            Box::pin(async move {
                match client.call_tool(&name, args).await {
                    Ok(result) => result,
                    Err(err) => common::coded("TOOL_FAULT", err.to_string(), true, None),
                }
            })
        }

        fn purity(&self) -> Purity {
            Purity::Impure
        }
    }

    /// Connect to a server, discover its tools, and register each as an `McpProxyTool`
    /// in `registry`. Returns the bridged specs (also the live client for shutdown).
    pub async fn discover_and_register(
        desc: &McpServerDescriptor,
        registry: &hide_core::tool::ToolRegistry,
    ) -> Result<(Arc<McpClient>, Vec<ToolSpec>)> {
        let client = Arc::new(McpClient::connect(desc).await?);
        let specs = client.list_tools().await?;
        for spec in &specs {
            registry.register(McpProxyTool::new(spec.clone(), client.clone()));
        }
        Ok((client, specs))
    }

    /// Per-server budget for connect + tools/list, so one hung server cannot stall
    /// the whole catalog.
    pub const MCP_REGISTER_TIMEOUT_SECS: u64 = 30;

    /// The outcome of trying to register one MCP server's tools.
    pub struct McpRegistration {
        pub server_id: String,
        /// `Some` on success. NOTE: the registry itself owns a clone of the client via
        /// each registered proxy tool, so the tools stay callable even if this handle
        /// is dropped. Keep it if you want an explicit handle to the connection (e.g.
        /// to hold the subprocess); dropping the whole registry is what tears the
        /// server down (the client sets `kill_on_drop`).
        pub client: Option<Arc<McpClient>>,
        /// Names of the tools registered from this server (`mcp:<id>/<tool>`).
        pub tools: Vec<String>,
        /// `Some` if this server failed to connect/list, timed out, or was a duplicate
        /// id (the others still ran).
        pub error: Option<String>,
    }

    /// Connect to and register every descriptor's tools into `registry`, resiliently:
    /// each server gets a [`MCP_REGISTER_TIMEOUT_SECS`] budget, and a server that
    /// fails, times out, or has a duplicate id is recorded as an error and does NOT
    /// abort the rest (a single bad or hung MCP server must not disable the whole tool
    /// catalog). Returns one [`McpRegistration`] per descriptor, in order.
    pub async fn register_mcp_servers(
        descriptors: &[McpServerDescriptor],
        registry: &hide_core::tool::ToolRegistry,
    ) -> Vec<McpRegistration> {
        let dur = std::time::Duration::from_secs(MCP_REGISTER_TIMEOUT_SECS);
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            // A duplicate id would silently shadow the first server's tools in the
            // registry (same `mcp:<id>/<tool>` keys), so refuse it explicitly.
            if !seen.insert(desc.id.clone()) {
                out.push(McpRegistration {
                    server_id: desc.id.clone(),
                    client: None,
                    tools: Vec::new(),
                    error: Some(format!("duplicate server id \"{}\" skipped (would shadow the first)", desc.id)),
                });
                continue;
            }
            let reg = match tokio::time::timeout(dur, discover_and_register(desc, registry)).await {
                Ok(Ok((client, specs))) => McpRegistration {
                    server_id: desc.id.clone(),
                    client: Some(client),
                    tools: specs.iter().map(|s| s.name.clone()).collect(),
                    error: None,
                },
                Ok(Err(e)) => McpRegistration {
                    server_id: desc.id.clone(),
                    client: None,
                    tools: Vec::new(),
                    error: Some(e.to_string()),
                },
                Err(_) => McpRegistration {
                    server_id: desc.id.clone(),
                    client: None,
                    tools: Vec::new(),
                    error: Some(format!("timed out after {MCP_REGISTER_TIMEOUT_SECS}s connecting/listing")),
                },
            };
            out.push(reg);
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn bridge_maps_annotations_as_untrusted() {
            let tool = McpTool {
                name: "deploy".into(),
                title: Some("Deploy".into()),
                description: Some("desc".into()),
                input_schema: json!({"type":"object","additionalProperties":false}),
                output_schema: None,
                annotations: Some(McpAnnotations {
                    read_only_hint: Some(true),
                    destructive_hint: Some(false),
                    idempotent_hint: Some(true),
                    open_world_hint: Some(false),
                }),
            };
            let spec = mcp_tool_to_hide_spec("acme", tool);
            assert_eq!(spec.name, "mcp:acme/deploy");
            // server's annotations are carried...
            assert!(spec.annotations.read_only);
            // ...but the capability is only the bridged-call cap (not relaxed).
            assert_eq!(spec.capabilities_required, vec!["mcp.call".to_string()]);
        }

        #[test]
        fn mcp_result_iserror_maps_to_not_ok() {
            let v = json!({ "isError": true, "content": [{"type":"text","text":"boom"}] });
            let r = mcp_result_to_hide(&v);
            assert!(!r.ok);
            assert_eq!(r.status, ToolStatus::ToolError);
            assert!(matches!(&r.content[0], ToolContent::Text { text } if text == "boom"));
        }

        #[test]
        fn mcp_result_success_maps_to_ok() {
            let v = json!({ "structuredContent": {"x":1}, "content": [{"type":"text","text":"ok"}] });
            let r = mcp_result_to_hide(&v);
            assert!(r.ok);
            assert_eq!(r.structured_content.unwrap()["x"], 1);
        }

        #[test]
        fn parse_http_plain_json() {
            let body = r#"{"jsonrpc":"2.0","id":7,"result":{"ok":true}}"#;
            let v = parse_http_jsonrpc(body, 7).unwrap();
            assert_eq!(v["result"]["ok"], true);
        }

        #[test]
        fn parse_http_sse_picks_matching_id() {
            let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"v\":1}}\n\ndata: [DONE]\n";
            let v = parse_http_jsonrpc(body, 3).unwrap();
            assert_eq!(v["result"]["v"], 1);
        }

        #[tokio::test]
        async fn stdio_client_lists_and_calls_tools_against_a_fake_server() {
            // A tiny Python JSON-RPC server that implements initialize/tools/list/
            // tools/call over stdio. Skips if python3 is unavailable.
            if which_python().is_none() {
                eprintln!("python3 not found; skipping stdio MCP integration test");
                return;
            }
            let py = which_python().unwrap();
            let server_src = FAKE_SERVER;
            let desc = McpServerDescriptor {
                id: "fake".into(),
                transport: McpTransport::Stdio { command: py, args: vec!["-c".into(), server_src.into()] },
                trust: "third-party".into(),
            };
            let client = McpClient::connect(&desc).await.expect("connect");
            let specs = client.list_tools().await.expect("list");
            assert_eq!(specs.len(), 1);
            assert_eq!(specs[0].name, "mcp:fake/echo");
            let result = client.call_tool("echo", json!({ "msg": "hi" })).await.expect("call");
            assert!(result.ok);
            assert_eq!(result.structured_content.unwrap()["echoed"], "hi");
        }

        #[tokio::test]
        async fn register_mcp_servers_registers_tools_and_survives_a_bad_server() {
            if which_python().is_none() {
                eprintln!("python3 not found; skipping MCP registration test");
                return;
            }
            let py = which_python().unwrap();
            let good = McpServerDescriptor {
                id: "good".into(),
                transport: McpTransport::Stdio { command: py, args: vec!["-c".into(), FAKE_SERVER.into()] },
                trust: "third-party".into(),
            };
            // A server that cannot even launch: it must be recorded as an error, not
            // panic or abort the good one.
            let bad = McpServerDescriptor {
                id: "bad".into(),
                transport: McpTransport::Stdio { command: "definitely-not-a-real-binary-xyzzy".into(), args: vec![] },
                trust: "third-party".into(),
            };
            let registry = hide_core::tool::ToolRegistry::default();
            let results = register_mcp_servers(&[good, bad], &registry).await;

            assert_eq!(results.len(), 2);
            let good_r = results.iter().find(|r| r.server_id == "good").unwrap();
            assert!(good_r.error.is_none(), "good server errored: {:?}", good_r.error);
            assert!(good_r.tools.contains(&"mcp:good/echo".to_string()));
            let bad_r = results.iter().find(|r| r.server_id == "bad").unwrap();
            assert!(bad_r.error.is_some(), "bad server should have recorded an error");
            // The registry actually holds the good server's proxy tool, dispatchable.
            assert!(registry.get("mcp:good/echo").is_some());
        }

        #[tokio::test]
        async fn register_mcp_servers_rejects_duplicate_ids() {
            if which_python().is_none() {
                eprintln!("python3 not found; skipping MCP dup-id test");
                return;
            }
            let py = which_python().unwrap();
            let mk = |id: &str| McpServerDescriptor {
                id: id.to_string(),
                transport: McpTransport::Stdio { command: py.clone(), args: vec!["-c".into(), FAKE_SERVER.into()] },
                trust: "third-party".into(),
            };
            let registry = hide_core::tool::ToolRegistry::default();
            let results = register_mcp_servers(&[mk("dup"), mk("dup")], &registry).await;
            assert_eq!(results.len(), 2);
            // The first registers; the second is refused as a duplicate, not silently
            // clobbering the first's tools.
            assert!(results[0].error.is_none());
            assert!(results[1].error.as_deref().unwrap_or("").contains("duplicate"));
        }

        #[tokio::test]
        async fn http_client_lists_and_calls_tools_against_an_inprocess_server() {
            use axum::{
                extract::Json as AxumJson,
                http::HeaderMap,
                response::{IntoResponse, Response},
                routing::post,
                Router,
            };
            use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
            use std::sync::Arc as StdArc;

            // Tracks that the client echoed back the server-assigned session id on the
            // second request — the Streamable-HTTP session leg, end to end.
            let saw_session_id = StdArc::new(AtomicBool::new(false));
            let saw = saw_session_id.clone();

            async fn rpc(saw: StdArc<AtomicBool>, headers: HeaderMap, AxumJson(req): AxumJson<Value>) -> Response {
                let id = req.get("id").cloned().unwrap_or(Value::Null);
                let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
                // Any request carrying a session id proves the header round-tripped.
                if headers.contains_key("mcp-session-id") {
                    saw.store(true, AtomicOrdering::SeqCst);
                }
                let body = match method {
                    "initialize" => json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {
                            "protocolVersion": MCP_PROTOCOL_VERSION,
                            "capabilities": {},
                            "serverInfo": { "name": "fake-http", "version": "0" }
                        }
                    }),
                    "notifications/initialized" => {
                        // Notification: no body expected. Return 202-ish empty 200.
                        return AxumJson(json!({})).into_response();
                    }
                    "tools/list" => json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": { "tools": [{
                            "name": "echo",
                            "description": "echo back",
                            "inputSchema": {
                                "type": "object",
                                "properties": { "msg": { "type": "string" } },
                                "required": ["msg"],
                                "additionalProperties": false
                            }
                        }]}
                    }),
                    "tools/call" => {
                        let msg = req["params"]["arguments"]["msg"].clone();
                        json!({
                            "jsonrpc": "2.0", "id": id,
                            "result": {
                                "isError": false,
                                "structuredContent": { "echoed": msg },
                                "content": [{ "type": "text", "text": msg }]
                            }
                        })
                    }
                    _ => json!({
                        "jsonrpc": "2.0", "id": id,
                        "error": { "code": -32601, "message": "method not found" }
                    }),
                };
                // Always assign a session id so the client must echo it back next time.
                ([("MCP-Session-Id", "sess-abc123")], AxumJson(body)).into_response()
            }

            let app = Router::new().route("/mcp", post(move |headers, body| rpc(saw.clone(), headers, body)));

            // Bind an ephemeral port and serve in the background.
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });

            let desc = McpServerDescriptor {
                id: "fakehttp".into(),
                transport: McpTransport::StreamableHttp { endpoint: format!("http://{addr}/mcp") },
                trust: "third-party".into(),
            };

            // connect() runs initialize + the initialized notification.
            let client = McpClient::connect(&desc).await.expect("connect");
            let specs = client.list_tools().await.expect("list");
            assert_eq!(specs.len(), 1);
            assert_eq!(specs[0].name, "mcp:fakehttp/echo");

            let result = client.call_tool("echo", json!({ "msg": "hi-http" })).await.expect("call");
            assert!(result.ok);
            assert_eq!(result.structured_content.unwrap()["echoed"], "hi-http");

            // The session id assigned on the initialize response must have been carried
            // on a subsequent request (the Streamable-HTTP session leg).
            assert!(saw_session_id.load(AtomicOrdering::SeqCst), "client must echo MCP-Session-Id on later requests");

            server.abort();
        }

        fn which_python() -> Option<String> {
            for cand in ["python3", "python"] {
                if std::process::Command::new(cand)
                    .arg("--version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return Some(cand.to_string());
                }
            }
            None
        }

        const FAKE_SERVER: &str = r#"import sys, json
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    m = req.get("method"); i = req.get("id")
    if m == "initialize":
        send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"fake","version":"0"}}})
    elif m == "notifications/initialized":
        pass
    elif m == "tools/list":
        send({"jsonrpc":"2.0","id":i,"result":{"tools":[{"name":"echo","description":"echo","inputSchema":{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"],"additionalProperties":False}}]}})
    elif m == "tools/call":
        msg = req["params"]["arguments"]["msg"]
        send({"jsonrpc":"2.0","id":i,"result":{"isError":False,"structuredContent":{"echoed":msg},"content":[{"type":"text","text":msg}]}})
    else:
        send({"jsonrpc":"2.0","id":i,"error":{"code":-32601,"message":"method not found"}})
"#;
    }
}
#[rustfmt::skip]
pub mod memory {
    //! `memory.*` — a client-side, cross-session memory tool (Claude-parity, see
    //! `docs/RESEARCH.md`, memory/tooling section).
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
            Self { root: PathBuf::from(".hide/memories") }
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
        #[allow(clippy::result_large_err)]
        fn resolve(&self, args: &Value, key: &str) -> Result<PathBuf, ToolResult> {
            let rel = args.get(key).and_then(|v| v.as_str()).unwrap_or("");
            match safe_rel(rel) {
                Ok(rel) => Ok(self.config.root.join(rel)),
                Err(msg) => Err(common::coded("ARG_INVALID", msg, true, Some("path escapes rejected"))),
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
                    common::ok(json!({ "command": "view", "kind": "file", "content": numbered }), EffectSet::default())
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
            let line = args.get("insert_line").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
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
            let result = if full.is_dir() { std::fs::remove_dir_all(&full) } else { std::fs::remove_file(&full) };
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
            if args.get("new_path").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
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
                Component::RootDir | Component::Prefix(_) => return Err("absolute path rejected".to_string()),
            }
        }
        Ok(out)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn ctx() -> ToolCtx {
            ToolCtx { grant_id: None, deadline_ms: None, output_cap_bytes: 1 << 20 }
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
            (MemoryTool::with_config(MemoryConfig { root: root.clone() }), root)
        }

        #[tokio::test]
        async fn create_view_replace_roundtrip() {
            let (tool, root) = tmp_tool("rt");
            let r = tool.call(json!({ "command": "create", "path": "notes.md", "content": "a\nb\n" }), ctx()).await;
            assert!(r.ok, "create failed: {:?}", r.error);
            assert_eq!(std::fs::read_to_string(root.join("notes.md")).unwrap(), "a\nb\n");

            let v = tool.call(json!({ "command": "view", "path": "notes.md" }), ctx()).await;
            assert!(v.ok);
            let content = v.structured_content.unwrap()["content"].as_str().unwrap().to_string();
            assert!(content.contains("1\ta"), "numbered view: {content}");

            let rep = tool
                .call(json!({ "command": "str_replace", "path": "notes.md", "old_str": "a", "new_str": "A" }), ctx())
                .await;
            assert!(rep.ok, "replace failed: {:?}", rep.error);
            assert_eq!(std::fs::read_to_string(root.join("notes.md")).unwrap(), "A\nb\n");
            let _ = std::fs::remove_dir_all(root);
        }

        #[tokio::test]
        async fn str_replace_requires_unique_match() {
            let (tool, root) = tmp_tool("uniq");
            tool.call(json!({ "command": "create", "path": "f", "content": "x x x" }), ctx()).await;
            let r = tool
                .call(json!({ "command": "str_replace", "path": "f", "old_str": "x", "new_str": "y" }), ctx())
                .await;
            assert!(!r.ok);
            assert_eq!(r.error.unwrap().code, "CONFLICT");
            let _ = std::fs::remove_dir_all(root);
        }

        #[tokio::test]
        async fn insert_at_line_and_delete() {
            let (tool, root) = tmp_tool("ins");
            tool.call(json!({ "command": "create", "path": "f", "content": "a\nc" }), ctx()).await;
            let r =
                tool.call(json!({ "command": "insert", "path": "f", "insert_line": 1, "content": "b" }), ctx()).await;
            assert!(r.ok, "insert failed: {:?}", r.error);
            assert_eq!(std::fs::read_to_string(root.join("f")).unwrap(), "a\nb\nc");

            let d = tool.call(json!({ "command": "delete", "path": "f" }), ctx()).await;
            assert!(d.ok);
            assert!(!root.join("f").exists());
            let _ = std::fs::remove_dir_all(root);
        }

        #[tokio::test]
        async fn view_lists_directory() {
            let (tool, root) = tmp_tool("ls");
            tool.call(json!({ "command": "create", "path": "a.md", "content": "1" }), ctx()).await;
            tool.call(json!({ "command": "create", "path": "sub/b.md", "content": "2" }), ctx()).await;
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
            let r = tool.call(json!({ "command": "create", "path": "../escape.txt", "content": "x" }), ctx()).await;
            assert!(!r.ok, "must reject ..");
            // The escape file must not exist outside the root.
            assert!(!root.parent().unwrap().join("escape.txt").exists());
            let _ = std::fs::remove_dir_all(root);
        }

        #[tokio::test]
        async fn rejects_absolute_path() {
            let (tool, root) = tmp_tool("esc2");
            let r =
                tool.call(json!({ "command": "create", "path": "/tmp/hide_mem_escape", "content": "x" }), ctx()).await;
            assert!(!r.ok, "must reject absolute path");
            assert!(!Path::new("/tmp/hide_mem_escape").exists());
            let _ = std::fs::remove_dir_all(root);
        }

        #[tokio::test]
        async fn rejects_percent_encoded_traversal() {
            let (tool, root) = tmp_tool("esc3");
            let r = tool.call(json!({ "command": "view", "path": "%2e%2e/%2e%2e/etc/passwd" }), ctx()).await;
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
}
#[rustfmt::skip]
pub mod proc {
    //! Test / build wrappers (ch.03 §4.6.5).
    //!
    //! These are thin, sandboxed shells over the project's real test/build commands.
    //! Crucially they honor `EXEC_NONZERO`: failing tests and compiler errors are
    //! **data** (`ok:true` + `exit_code`), so the agent reads diagnostics and reacts
    //! (§4.2.3) — the verify-after-edit loop depends on this.

    use crate::shell::{run_command, ShellConfig};
    use crate::spec_helpers::exec_spec;
    use futures::future::BoxFuture;
    use hide_core::tool::{Purity, Tool, ToolCtx, ToolResult, ToolSpec};
    use hide_core::types::{Effect, EffectKind, EffectSet, RiskLevel};
    use serde_json::Value;
    use std::collections::BTreeMap;

    /// Generic argv-running tool for the test/build family. The default argv is the
    /// project command; callers may override `argv` entirely.
    #[derive(Clone)]
    pub struct ProcTool {
        spec: ToolSpec,
        default_argv: Vec<String>,
        config: ShellConfig,
    }

    impl ProcTool {
        fn new(name: &str, title: &str, desc: &str, default_argv: Vec<String>) -> Self {
            let mut spec = exec_spec(name, title, desc, 1024 * 1024, 600_000);
            // these are auto-policy in the catalog (sandboxed + scoped); annotate
            // them as non-open-world so the policy engine treats them gently.
            spec.annotations.open_world = false;
            Self { spec, default_argv, config: ShellConfig::default() }
        }

        pub fn with_config(mut self, config: ShellConfig) -> Self {
            self.config = config;
            self
        }

        /// `test.run` — runs the project test command (default `cargo test`).
        pub fn test_run() -> Self {
            Self::new(
                "test.run",
                "Run tests",
                "Run the project test suite (sandboxed). Failing tests are DATA, not an error.",
                vec!["cargo".into(), "test".into()],
            )
        }

        /// `build.run` — runs the project build (default `cargo build`).
        pub fn build_run() -> Self {
            Self::new(
                "build.run",
                "Build project",
                "Build the project (sandboxed). Compiler errors are DATA, not an error.",
                vec!["cargo".into(), "build".into()],
            )
        }

        /// `compile.check` — type-check only (default `cargo check`).
        pub fn compile_check() -> Self {
            Self::new(
                "compile.check",
                "Type-check",
                "Type-check the project (sandboxed). Diagnostics are DATA, not an error.",
                vec!["cargo".into(), "check".into()],
            )
        }
    }

    impl Tool for ProcTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let argv = args
                    .get("argv")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>())
                    .filter(|v| !v.is_empty())
                    .unwrap_or_else(|| self.default_argv.clone());
                let cwd = args.get("cwd").and_then(|v| v.as_str()).map(String::from);
                let timeout = ctx.deadline_ms.filter(|ms| *ms > 0).unwrap_or(self.spec.timeout_ms);
                let env = BTreeMap::new();
                run_command(&argv, cwd.as_deref(), &env, timeout, ctx.output_cap_bytes as usize, &self.config).await
            })
        }

        fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
            Box::pin(async move {
                let argv = args
                    .get("argv")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(" "))
                    .unwrap_or_else(|| self.default_argv.join(" "));
                Some(EffectSet {
                    effects: vec![Effect {
                        kind: EffectKind::Execute,
                        target: argv,
                        bytes_hash: None,
                        risk: RiskLevel::Medium,
                        metadata: BTreeMap::new(),
                    }],
                })
            })
        }

        fn purity(&self) -> Purity {
            Purity::Impure
        }
    }

    /// `test.run`/`build.run` enrich the structured body with a coarse pass/fail flag.
    pub fn pass_fail(result: &ToolResult) -> Option<bool> {
        result.exit_code.map(|c| c == 0)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;
        use std::sync::Arc;

        fn ctx() -> ToolCtx {
            ToolCtx { grant_id: None, deadline_ms: Some(30_000), output_cap_bytes: 1 << 20 }
        }

        #[tokio::test]
        async fn proc_failing_command_is_ok_data() {
            // override argv with a command that exits non-zero
            let tool = ProcTool::test_run().with_config(ShellConfig { disable_sandbox: true, ..Default::default() });
            let r = tool.call(json!({ "argv": ["sh", "-c", "echo fail 1>&2; exit 1"] }), ctx()).await;
            assert!(r.ok, "failing test run must be ok:true (EXEC_NONZERO)");
            assert_eq!(r.exit_code, Some(1));
            assert_eq!(pass_fail(&r), Some(false));
        }

        #[tokio::test]
        async fn proc_passing_command() {
            let tool =
                ProcTool::compile_check().with_config(ShellConfig { disable_sandbox: true, ..Default::default() });
            let r = tool.call(json!({ "argv": ["true"] }), ctx()).await;
            assert!(r.ok);
            assert_eq!(pass_fail(&r), Some(true));
        }

        #[test]
        fn send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<Arc<ProcTool>>();
        }
    }
}
#[rustfmt::skip]
pub mod registry {
    //! Registration of the builtin catalog into a `hide-core` `ToolRegistry`.

    use crate::edit::{ApplyPatchTool, SearchReplaceTool, WriteFileTool};
    use crate::fs::{FsConfig, FsGlobTool, FsListTool, FsReadTool, FsStatTool, FsWatchTool, FsWriteTool};
    use crate::git::{
        GitCommitTool, GitConfig, GitDiffTool, GitLogTool, GitStatusTool, GitWorktreeAddTool, GitWorktreeListTool,
        GitWorktreeRemoveTool,
    };
    use crate::memory::MemoryTool;
    use crate::proc::ProcTool;
    use crate::search::SearchTextTool;
    use crate::shell::{ShellConfig, ShellPlanTool, ShellRunTool};
    use hide_core::tool::ToolRegistry;

    /// Register the full builtin catalog with default (no-blob, default-sandbox)
    /// configuration.
    pub fn register_builtin_tools(registry: &ToolRegistry) {
        register_builtin_tools_with(registry, ShellConfig::default());
    }

    /// Register the full builtin catalog, threading a [`ShellConfig`] (workspace root,
    /// blob store for large-output spill, sandbox toggle) into the process and FS
    /// tools. The blob store, when present, is shared so over-cap output spills to the
    /// same CAS the event log uses.
    pub fn register_builtin_tools_with(registry: &ToolRegistry, shell_config: ShellConfig) {
        let fs_config = FsConfig { blobs: shell_config.blobs.clone() };
        let git_config = GitConfig { blobs: shell_config.blobs.clone() };

        // Filesystem
        registry.register(FsReadTool::with_config(fs_config.clone()));
        registry.register(FsListTool::default());
        registry.register(FsWriteTool::default());
        registry.register(FsStatTool::default());
        registry.register(FsGlobTool::default());
        registry.register(FsWatchTool::default());

        // Edit family
        registry.register(SearchReplaceTool::default());
        registry.register(ApplyPatchTool::default());
        registry.register(WriteFileTool::default());

        // Shell / process
        registry.register(ShellRunTool::with_config(shell_config.clone()));
        registry.register(ShellPlanTool::with_config(shell_config.clone()));
        registry.register(ProcTool::test_run().with_config(shell_config.clone()));
        registry.register(ProcTool::build_run().with_config(shell_config.clone()));
        registry.register(ProcTool::compile_check().with_config(shell_config.clone()));

        // Search
        registry.register(SearchTextTool::default());

        // Git
        registry.register(GitStatusTool::with_config(git_config.clone()));
        registry.register(GitDiffTool::with_config(git_config.clone()));
        registry.register(GitLogTool::with_config(git_config.clone()));
        registry.register(GitCommitTool::with_config(git_config.clone()));
        registry.register(GitWorktreeAddTool::with_config(git_config.clone()));
        registry.register(GitWorktreeRemoveTool::with_config(git_config.clone()));
        registry.register(GitWorktreeListTool::with_config(git_config));

        // Memory (durable cross-session scratchpad)
        registry.register(MemoryTool::default());
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn registers_full_catalog() {
            let registry = ToolRegistry::default();
            register_builtin_tools(&registry);
            let names: Vec<String> = registry.specs().into_iter().map(|s| s.name).collect();
            for expected in [
                "fs.read",
                "fs.list",
                "fs.write",
                "fs.stat",
                "fs.glob",
                "fs.watch",
                "edit.search_replace",
                "edit.apply_patch",
                "edit.write_file",
                "shell.run",
                "shell.plan",
                "test.run",
                "build.run",
                "compile.check",
                "search.text",
                "git.status",
                "git.diff",
                "git.log",
                "git.commit",
                "git.worktree.add",
                "git.worktree.remove",
                "git.worktree.list",
                "memory",
            ] {
                assert!(names.contains(&expected.to_string()), "missing {expected}");
            }
        }
    }
}
#[rustfmt::skip]
pub mod search {
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
                let ignore_case = args.get("ignore_case").and_then(|v| v.as_bool()).unwrap_or(false);
                let glob = args.get("glob").and_then(|v| v.as_str());

                let re = match RegexBuilder::new(pattern).case_insensitive(ignore_case).build() {
                    Ok(re) => re,
                    Err(err) => {
                        return common::arg_invalid(
                            format!("invalid regex: {err}"),
                            Some("provide a valid Rust regex"),
                            Some("/pattern"),
                        )
                    }
                };
                let glob_set = glob.and_then(|g| globset::Glob::new(g).ok().map(|g| g.compile_matcher()));

                let mut matches = Vec::new();
                let mut truncated = false;
                'walk: for entry in
                    ignore::WalkBuilder::new(root).git_ignore(true).require_git(false).hidden(true).build().flatten()
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
            ToolCtx { grant_id: None, deadline_ms: None, output_cap_bytes: 1 << 20 }
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
            let r = tool.call(json!({ "pattern": "fn target", "root": dir.to_string_lossy() }), ctx()).await;
            let matches = r.structured_content.unwrap()["matches"].as_array().unwrap().clone();
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
}
#[rustfmt::skip]
pub mod shell {
    //! Non-interactive shell execution (ch.03 §4.8).
    //!
    //! Design points honored here:
    //!
    //! * **argv-form preferred** — no shell string is interpolated, so `;`/`&&`/`$()`
    //!   can never smuggle a second command past the capability scope.
    //! * **timeout watchdog** — `timeout_ms` is enforced by a `tokio::time::timeout`
    //!   wrapping the child; on expiry the process is sent SIGTERM, then SIGKILL after
    //!   a short grace, and the result is `TIMEOUT` (§4.8).
    //! * **OS sandbox** — on macOS the command is wrapped in `sandbox-exec` with an
    //!   SBPL profile rendered by `hide_security::sandbox::render_macos_seatbelt_with`
    //!   (network-deny by default; the absolute `.hide/log` write-deny and the
    //!   proxy-egress route are threaded through `SandboxRenderOptions`). On Linux the
    //!   command is wrapped in bubblewrap (`bwrap`) when present. **Fail-closed**: if
    //!   no OS sandbox is available the run is REFUSED (`SANDBOX_UNAVAILABLE`) rather
    //!   than run unconfined — the only opt-outs are `disable_sandbox` (already-confined
    //!   worktree) or the explicit `allow_unconfined` escape hatch, both of which
    //!   record a warning in the result.
    //! * **EXEC_NONZERO is data** — a non-zero exit is `ok:true` + `exit_code`, never a
    //!   tool error (§4.2.3); only a spawn failure is `ok:false`.

    use crate::common;
    use crate::spec_helpers::{exec_spec, plan_spec};
    use futures::future::BoxFuture;
    use hide_core::persistence::BlobStore;
    use hide_core::security::{NetworkPolicy, SandboxProfile, SandboxTier};
    use hide_core::tool::{Purity, Tool, ToolContent, ToolCtx, ToolResult, ToolSpec};
    use hide_core::types::{Effect, EffectKind, EffectSet, RiskLevel};
    use hide_security::sandbox::SandboxRenderOptions;
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::process::Stdio;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::process::Command;

    /// Patterns that are always refused before spawn (defense-in-depth; the canonical
    /// deny policy lives in Ch.10, but a coding agent should never reach these).
    const CATASTROPHIC: &[&str] = &["rm -rf /", ":(){:|:&};:", "mkfs", "dd if="];

    /// Shared configuration for shell execution — the workspace root used to confine
    /// writes, and an optional blob store for large-output spill.
    #[derive(Clone, Default)]
    pub struct ShellConfig {
        pub workspace_root: Option<String>,
        pub blobs: Option<Arc<dyn BlobStore>>,
        /// Force-disable the OS sandbox (e.g. inside an already-confined worktree run).
        pub disable_sandbox: bool,
        /// The `.hide` directory whose `log` subdir must be write-denied (S4). Threaded
        /// into [`SandboxRenderOptions::hide_dir`] so the absolute `.hide/log`
        /// write-deny is rendered rather than the relative fallback.
        pub hide_dir: Option<PathBuf>,
        /// Worktree root writes are confined to (§4.5.2 `$WORKTREE`). Threaded into
        /// [`SandboxRenderOptions::worktree_root`]; falls back to `workspace_root`.
        pub worktree_root: Option<String>,
        /// Host egress proxy port; `Some` ⇒ the only allowed outbound socket is the
        /// proxy (S5b). Threaded into [`SandboxRenderOptions::proxy_port`].
        pub proxy_port: Option<u16>,
        /// Off-macOS escape hatch: explicitly opt out of fail-closed sandboxing. When
        /// `false` (the default) a sandboxed run on a platform with no OS sandbox is
        /// REFUSED rather than run unconfined (fail-closed, item 1).
        pub allow_unconfined: bool,
    }

    #[derive(Clone)]
    pub struct ShellRunTool {
        spec: ToolSpec,
        config: ShellConfig,
    }

    impl Default for ShellRunTool {
        fn default() -> Self {
            Self {
                spec: exec_spec(
                    "shell.run",
                    "Run shell command",
                    "Run an already-authorized non-interactive command (argv form), sandboxed and \
                     deadline-bounded. Non-zero exit is data, not an error.",
                    256 * 1024,
                    30_000,
                ),
                config: ShellConfig::default(),
            }
        }
    }

    impl ShellRunTool {
        pub fn with_config(config: ShellConfig) -> Self {
            Self { config, ..Self::default() }
        }
    }

    impl Tool for ShellRunTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let argv = parse_argv(&args);
                if argv.is_empty() {
                    return common::arg_invalid(
                        "argv must contain at least one element",
                        Some("pass argv as a non-empty array, e.g. [\"cargo\", \"test\"]"),
                        Some("/argv"),
                    );
                }
                if let Some(bad) = catastrophic_hit(&argv) {
                    return common::coded(
                        "CAP_DENIED",
                        format!("refused catastrophic command pattern: {bad}"),
                        false,
                        None,
                    );
                }
                let cwd = args.get("cwd").and_then(|v| v.as_str()).map(str::to_string);
                let env = parse_env(&args);
                let timeout = ctx.deadline_ms.filter(|ms| *ms > 0).unwrap_or(self.spec.timeout_ms);
                run_command(&argv, cwd.as_deref(), &env, timeout, ctx.output_cap_bytes as usize, &self.config).await
            })
        }

        fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
            Box::pin(async move {
                let argv = parse_argv(args);
                Some(EffectSet {
                    effects: vec![Effect {
                        kind: EffectKind::Execute,
                        target: argv.join(" "),
                        bytes_hash: None,
                        risk: RiskLevel::High,
                        metadata: BTreeMap::new(),
                    }],
                })
            })
        }

        fn purity(&self) -> Purity {
            Purity::Impure
        }
    }

    /// `shell.plan` — describe the command + its rendered sandbox profile without
    /// running anything. Powers "show me what this will do" before approval.
    #[derive(Clone)]
    pub struct ShellPlanTool {
        spec: ToolSpec,
        config: ShellConfig,
    }

    impl Default for ShellPlanTool {
        fn default() -> Self {
            Self { spec: plan_spec(), config: ShellConfig::default() }
        }
    }

    impl ShellPlanTool {
        pub fn with_config(config: ShellConfig) -> Self {
            Self { config, ..Self::default() }
        }
    }

    impl Tool for ShellPlanTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn call<'a>(&'a self, args: Value, _ctx: ToolCtx) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let argv = parse_argv(&args);
                let profile = sandbox_profile(&self.config, &argv);
                let opts = sandbox_render_options(&self.config);
                let rendered = hide_security::sandbox::render_macos_seatbelt_with(&profile, &opts);
                let body = json!({
                    "argv": argv,
                    "executed": false,
                    "sandbox_tier": format!("{:?}", profile.tier),
                    "sandbox_warnings": rendered.warnings,
                    "sandbox_profile": runnable_sbpl(&rendered.profile_text),
                    "network": "deny-by-default",
                });
                common::ok_text(format!("planned (sandboxed) command: {argv:?}"), body, EffectSet::default())
            })
        }

        fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
            Box::pin(async move {
                Some(EffectSet {
                    effects: vec![Effect {
                        kind: EffectKind::Execute,
                        target: parse_argv(args).join(" "),
                        bytes_hash: None,
                        risk: RiskLevel::High,
                        metadata: BTreeMap::new(),
                    }],
                })
            })
        }

        fn purity(&self) -> Purity {
            Purity::Pure
        }
    }

    /// Build the sandbox profile for a shell run, honoring Ch.10's model: read broadly
    /// (policy-bounded upstream), write confined to the workspace + temp roots,
    /// network denied by default, and **process-exec allowlisted to exactly the
    /// commands this run needs** (§4.9.3 — exec is granted per binary, not blanket).
    ///
    /// The allowlist is `argv[0]` (resolved to an absolute path where possible) plus
    /// the small set of interpreter/toolchain helpers a real build/test invocation
    /// shells out to. `hide_security::sandbox::render_macos_seatbelt` turns this into
    /// the `(allow process-exec …)` allowlist.
    pub fn sandbox_profile(config: &ShellConfig, argv: &[String]) -> SandboxProfile {
        // Seatbelt `subpath` needs an absolute path; resolve the workspace root.
        let root = config
            .workspace_root
            .clone()
            .and_then(|r| std::fs::canonicalize(&r).ok().map(|p| p.to_string_lossy().into_owned()))
            .or_else(|| std::env::current_dir().ok().map(|p| p.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "/tmp".to_string());
        let tmp = std::env::temp_dir().to_string_lossy().into_owned();

        let mut allowed = essential_exec_allowlist();
        if let Some(bin) = argv.first() {
            allowed.push(resolve_binary(bin));
        }
        allowed.sort();
        allowed.dedup();

        SandboxProfile {
            tier: SandboxTier::Seatbelt,
            read_roots: vec!["/".to_string()],
            write_roots: vec![root, tmp],
            allowed_commands: allowed,
            network: NetworkPolicy::default(), // default = Deny
        }
    }

    /// Build the render-time options that thread the absolute `.hide/log` write-deny
    /// and the proxy-egress route into [`render_macos_seatbelt_with`]. `worktree_root`
    /// falls back to `workspace_root` so writes are confined even when the caller only
    /// set one (§4.5.2).
    pub fn sandbox_render_options(config: &ShellConfig) -> SandboxRenderOptions {
        SandboxRenderOptions {
            proxy_port: config.proxy_port,
            hide_dir: config.hide_dir.clone(),
            worktree_root: config.worktree_root.clone().or_else(|| config.workspace_root.clone()),
        }
    }

    /// The interpreter/toolchain helpers a real command commonly re-execs (git calls
    /// hooks, cargo spawns rustc, shells spawn coreutils). Bare names are matched by
    /// basename regex in the renderer.
    fn essential_exec_allowlist() -> Vec<String> {
        [
            "sh", "bash", "zsh", "env", "git", "cargo", "rustc", "cc", "ld", "clang", "printf", "echo", "true",
            "sleep", "cat", "ls", "node", "python3", "python",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Resolve a binary to an absolute path via `PATH` so the allowlist literal pins
    /// it; fall back to the bare name (matched by basename regex) if not found.
    fn resolve_binary(name: &str) -> String {
        if name.starts_with('/') {
            return name.to_string();
        }
        if let Ok(path) = std::env::var("PATH") {
            for dir in path.split(':') {
                let candidate = std::path::Path::new(dir).join(name);
                if candidate.exists() {
                    return candidate.to_string_lossy().into_owned();
                }
            }
        }
        name.to_string()
    }

    /// Add the universal runtime allowances any process needs under Seatbelt that the
    /// base render (which scopes file/exec/net) does not emit: fork, sysctl-read,
    /// mach-lookup, self-signalling, and `/dev` access for stdio. The renderer already
    /// emits the `(allow process-exec …)` allowlist; we never widen exec here.
    pub fn runnable_sbpl(base: &str) -> String {
        let mut s = String::with_capacity(base.len() + 256);
        s.push_str(base);
        s.push_str("\n;; --- hide-tools runtime allowances ---\n");
        s.push_str("(allow sysctl-read)\n");
        s.push_str("(allow mach-lookup)\n");
        s.push_str("(allow signal (target self))\n");
        s.push_str("(allow file-read* (subpath \"/dev\"))\n");
        s.push_str("(allow file-write* (literal \"/dev/null\"))\n");
        s.push_str("(allow file-write* (literal \"/dev/dtracehelper\"))\n");
        s
    }

    /// Whether `sandbox-exec` exists on this host.
    fn sandbox_exec_available() -> bool {
        cfg!(target_os = "macos") && std::path::Path::new("/usr/bin/sandbox-exec").exists()
    }

    /// Resolve `bwrap` (bubblewrap) on `PATH`. `Some(path)` ⇒ a Linux confinement
    /// route is available.
    fn bubblewrap_path() -> Option<String> {
        if !cfg!(target_os = "linux") {
            return None;
        }
        let resolved = resolve_binary("bwrap");
        if resolved.starts_with('/') && std::path::Path::new(&resolved).exists() {
            Some(resolved)
        } else {
            None
        }
    }

    /// A built (sandbox-wrapped or — explicitly opted-out — bare) command plus an
    /// optional warning to surface in the result.
    struct SandboxedSpawn {
        command: Command,
        warning: Option<String>,
    }

    /// Build the command to spawn, applying OS confinement per the platform.
    ///
    /// Fail-closed (item 1): on a platform with no usable OS sandbox we REFUSE rather
    /// than silently running unconfined. The only ways to run without confinement are
    /// `config.disable_sandbox` (an already-confined worktree) or
    /// `config.allow_unconfined` (an explicit, logged escape hatch). Both surface a
    /// warning in the result.
    ///
    /// * **macOS + `sandbox-exec`** — wrap in `sandbox-exec -p <SBPL>`, where the SBPL
    ///   is rendered by `render_macos_seatbelt_with` (item 2: threads the absolute
    ///   `.hide/log` write-deny + proxy-egress route through `SandboxRenderOptions`).
    /// * **Linux + `bwrap`** — wrap in bubblewrap with a read-only root, a writable
    ///   worktree + tmp, and `--unshare-net` (network denied by default).
    /// * **anything else** — `Err(refusal)` unless an opt-out is set.
    fn build_confined_command(argv: &[String], config: &ShellConfig) -> Result<SandboxedSpawn, Box<ToolResult>> {
        // Explicit, caller-chosen opt-out for an already-confined context.
        if config.disable_sandbox {
            return Ok(SandboxedSpawn { command: bare_command(argv), warning: None });
        }

        if sandbox_exec_available() {
            let profile = sandbox_profile(config, argv);
            let opts = sandbox_render_options(config);
            let rendered = hide_security::sandbox::render_macos_seatbelt_with(&profile, &opts);
            let sbpl = runnable_sbpl(&rendered.profile_text);
            if std::env::var("HIDE_DEBUG_SBPL").is_ok() {
                eprintln!("=== SBPL ===\n{sbpl}\n=== END SBPL ===");
            }
            let mut c = Command::new("/usr/bin/sandbox-exec");
            c.arg("-p").arg(sbpl);
            c.arg("--").args(argv);
            return Ok(SandboxedSpawn { command: c, warning: None });
        }

        if let Some(bwrap) = bubblewrap_path() {
            return Ok(SandboxedSpawn { command: bubblewrap_command(&bwrap, argv, config), warning: None });
        }

        // No OS sandbox available on this platform. Fail closed unless explicitly
        // overridden.
        if config.allow_unconfined {
            return Ok(SandboxedSpawn {
                command: bare_command(argv),
                warning: Some(
                    "OS sandbox unavailable on this platform; running UNCONFINED via explicit \
                     allow_unconfined override (escape hatch)"
                        .to_string(),
                ),
            });
        }

        Err(Box::new(common::coded(
            "SANDBOX_UNAVAILABLE",
            "refusing to run unconfined: no OS sandbox is available on this platform \
             (macOS sandbox-exec / Linux bwrap not found)",
            false,
            Some(
                "install bubblewrap (`bwrap`) on Linux, run under an already-confined worktree \
                 (disable_sandbox), or set ShellConfig.allow_unconfined to opt out explicitly",
            ),
        )))
    }

    /// A bare, unconfined command (`argv[0]` + the rest). Used only when an opt-out
    /// has been chosen.
    fn bare_command(argv: &[String]) -> Command {
        let mut c = Command::new(&argv[0]);
        c.args(&argv[1..]);
        c
    }

    /// Wrap `argv` in bubblewrap: read-only `/`, a writable worktree + tmp, no new
    /// session, and `--unshare-net` so network is denied by default (mirrors the
    /// Seatbelt deny-network posture). The proxy-egress route is the host's job and
    /// is not punched into the net namespace here.
    fn bubblewrap_command(bwrap: &str, argv: &[String], config: &ShellConfig) -> Command {
        let opts = sandbox_render_options(config);
        let tmp = std::env::temp_dir().to_string_lossy().into_owned();
        let write_root = opts.worktree_root.clone();

        let mut c = Command::new(bwrap);
        // Read-only view of the host root so reads work but nothing is mutated...
        c.arg("--ro-bind").arg("/").arg("/");
        // ...then re-bind the writable roots read-write.
        if let Some(root) = &write_root {
            c.arg("--bind").arg(root).arg(root);
        }
        c.arg("--bind").arg(&tmp).arg(&tmp);
        c.arg("--dev").arg("/dev");
        c.arg("--proc").arg("/proc");
        // Network denied by default (no proxy punched in here).
        c.arg("--unshare-net");
        c.arg("--die-with-parent");
        c.arg("--").args(argv);
        c
    }

    /// Run one command with sandbox wrapping + timeout watchdog and project the
    /// captured output to the canonical result.
    pub async fn run_command(
        argv: &[String],
        cwd: Option<&str>,
        env: &BTreeMap<String, String>,
        timeout_ms: u64,
        cap_bytes: usize,
        config: &ShellConfig,
    ) -> ToolResult {
        let (mut command, sandbox_warning) = match build_confined_command(argv, config) {
            Ok(SandboxedSpawn { command, warning }) => (command, warning),
            Err(refusal) => return *refusal,
        };

        if let Some(cwd) = cwd {
            command.current_dir(cwd);
        }
        for (k, v) in env {
            command.env(k, v);
        }
        command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);

        let child = match command.spawn() {
            Ok(child) => child,
            Err(err) => return common::spawn_fault(format!("failed to spawn {}: {err}", argv[0])),
        };

        let pid = child.id();
        let wait = child.wait_with_output();
        let output = match tokio::time::timeout(Duration::from_millis(timeout_ms), wait).await {
            Ok(Ok(output)) => output,
            Ok(Err(err)) => return common::spawn_fault(format!("io error awaiting child: {err}")),
            Err(_elapsed) => {
                // Watchdog fired: SIGTERM, grace, SIGKILL.
                terminate(pid).await;
                return common::coded(
                    "TIMEOUT",
                    format!("command exceeded {timeout_ms}ms deadline"),
                    true,
                    Some("increase timeout_ms or run a smaller/faster command"),
                );
            }
        };

        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let mut result = common::project_process_output(exit_code, stdout, stderr, cap_bytes, config.blobs.as_ref());
        if let Some(warn) = sandbox_warning {
            if let Some(sc) = result.structured_content.as_mut() {
                sc["sandbox_warning"] = json!(warn);
            }
            result.content.push(ToolContent::Text { text: format!("[sandbox] {warn}") });
        }
        result
    }

    /// SIGTERM the process group, wait a short grace, then SIGKILL (§4.8 ladder).
    /// On non-Unix this is a best-effort no-op (the `kill_on_drop` guard still runs).
    #[cfg(unix)]
    async fn terminate(pid: Option<u32>) {
        let Some(pid) = pid else { return };
        let pid = pid as libc::pid_t;
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }

    #[cfg(not(unix))]
    async fn terminate(_pid: Option<u32>) {}

    fn catastrophic_hit(argv: &[String]) -> Option<String> {
        let joined = argv.join(" ");
        CATASTROPHIC.iter().find(|needle| joined.contains(*needle)).map(|s| s.to_string())
    }

    pub(crate) fn parse_argv(args: &Value) -> Vec<String> {
        args.get("argv")
            .and_then(|v| v.as_array())
            .map(|items| items.iter().filter_map(|v| v.as_str().map(ToOwned::to_owned)).collect())
            .unwrap_or_default()
    }

    fn parse_env(args: &Value) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        if let Some(env) = args.get("env").and_then(|v| v.as_object()) {
            for (k, v) in env {
                if let Some(v) = v.as_str() {
                    out.insert(k.clone(), v.to_string());
                }
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::permission::{PermissionPolicy, StaticPermissionEngine};
        use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry, ToolStatus};

        fn allow_all_dispatcher(registry: Arc<ToolRegistry>) -> ToolDispatcher {
            ToolDispatcher::new(
                registry,
                Arc::new(StaticPermissionEngine::new(PermissionPolicy {
                    default_decision: hide_core::types::Decision::Allow,
                    rules: Vec::new(),
                    risk_gates: Vec::new(),
                })),
            )
        }

        #[tokio::test]
        async fn shell_run_executes_and_captures_stdout() {
            let registry = Arc::new(ToolRegistry::default());
            registry.register(ShellRunTool::default());
            let dispatcher = allow_all_dispatcher(registry);
            let result =
                dispatcher.dispatch(ToolCall::new("shell.run", json!({ "argv": ["printf", "hello"] }))).await.unwrap();
            assert_eq!(result.status, ToolStatus::Ok);
            assert_eq!(result.structured_content.unwrap()["stdout"], "hello");
        }

        #[tokio::test]
        async fn shell_run_nonzero_exit_is_ok_data() {
            let registry = Arc::new(ToolRegistry::default());
            registry.register(ShellRunTool::default());
            let dispatcher = allow_all_dispatcher(registry);
            // `sh -c 'exit 3'` — a non-zero exit MUST be ok:true with exit_code.
            let result = dispatcher
                .dispatch(ToolCall::new("shell.run", json!({ "argv": ["sh", "-c", "echo boom 1>&2; exit 3"] })))
                .await
                .unwrap();
            assert!(result.ok, "EXEC_NONZERO must be ok:true");
            assert_eq!(result.exit_code, Some(3));
            assert_eq!(result.status, ToolStatus::Ok);
            let sc = result.structured_content.unwrap();
            assert!(sc["stderr"].as_str().unwrap().contains("boom"));
        }

        #[tokio::test]
        async fn shell_run_times_out() {
            // Direct call with a tiny deadline; the watchdog must kill + return TIMEOUT.
            let config = ShellConfig { disable_sandbox: true, ..Default::default() };
            let env = BTreeMap::new();
            let result = run_command(&["sleep".to_string(), "5".to_string()], None, &env, 150, 4096, &config).await;
            assert!(!result.ok);
            assert_eq!(result.status, ToolStatus::TimedOut);
            assert_eq!(result.error.unwrap().code, "TIMEOUT");
        }

        #[tokio::test]
        async fn shell_run_refuses_catastrophic() {
            let registry = Arc::new(ToolRegistry::default());
            registry.register(ShellRunTool::default());
            let dispatcher = allow_all_dispatcher(registry);
            let result =
                dispatcher.dispatch(ToolCall::new("shell.run", json!({ "argv": ["rm", "-rf", "/"] }))).await.unwrap();
            assert!(!result.ok);
            assert_eq!(result.error.unwrap().code, "CAP_DENIED");
        }

        #[test]
        fn fail_closed_when_no_os_sandbox_available() {
            // Simulate a platform with no usable OS sandbox: not disable_sandbox, not
            // allow_unconfined. On a host where sandbox-exec/bwrap is genuinely
            // unavailable this is the live path; on macOS CI we still assert the
            // decision function refuses (it only ever runs UNCONFINED via an opt-out).
            let config = ShellConfig { allow_unconfined: false, disable_sandbox: false, ..Default::default() };
            let argv = vec!["true".to_string()];
            match build_confined_command(&argv, &config) {
                Ok(_) => {
                    // Only acceptable if this host actually HAS an OS sandbox.
                    assert!(
                        sandbox_exec_available() || bubblewrap_path().is_some(),
                        "got an Ok command with no OS sandbox available — fail-closed breached"
                    );
                }
                Err(refusal) => {
                    let refusal = *refusal;
                    assert!(!refusal.ok);
                    assert_eq!(refusal.error.unwrap().code, "SANDBOX_UNAVAILABLE");
                }
            }
        }

        #[test]
        fn allow_unconfined_opt_out_runs_bare_with_warning() {
            // The explicit escape hatch: a sandboxless host may run UNCONFINED only
            // when allow_unconfined is set, and must surface a warning.
            let config = ShellConfig { allow_unconfined: true, ..Default::default() };
            let argv = vec!["true".to_string()];
            let spawn = build_confined_command(&argv, &config).expect("opt-out must not refuse");
            if sandbox_exec_available() || bubblewrap_path().is_some() {
                // Real sandbox present → confined, no escape-hatch warning.
                assert!(spawn.warning.is_none());
            } else {
                assert!(
                    spawn.warning.as_deref().map(|w| w.contains("UNCONFINED")).unwrap_or(false),
                    "unconfined opt-out must carry a warning"
                );
            }
        }

        #[test]
        fn disable_sandbox_runs_bare_without_refusal() {
            // An already-confined worktree run opts out and is never refused.
            let config = ShellConfig { disable_sandbox: true, ..Default::default() };
            let argv = vec!["true".to_string()];
            let spawn = build_confined_command(&argv, &config).expect("disable_sandbox never refuses");
            assert!(spawn.warning.is_none());
        }

        #[test]
        fn render_options_thread_hide_dir_and_worktree() {
            // Item 2: the absolute .hide/log write-deny and the worktree confinement
            // must reach the rendered SBPL via render_macos_seatbelt_with.
            let config = ShellConfig {
                workspace_root: Some("/tmp".to_string()),
                worktree_root: Some("/tmp/wt".to_string()),
                hide_dir: Some(PathBuf::from("/var/hide-test/.hide")),
                proxy_port: Some(8443),
                ..Default::default()
            };
            let opts = sandbox_render_options(&config);
            assert_eq!(opts.worktree_root.as_deref(), Some("/tmp/wt"));
            assert_eq!(opts.hide_dir, Some(PathBuf::from("/var/hide-test/.hide")));
            assert_eq!(opts.proxy_port, Some(8443));

            let profile = sandbox_profile(&config, &["cargo".to_string(), "test".to_string()]);
            let rendered = hide_security::sandbox::render_macos_seatbelt_with(&profile, &opts);
            // Absolute .hide/log write-deny (S4) — not the relative fallback.
            assert!(
                rendered.profile_text.contains("(deny file-write* (subpath \"/var/hide-test/.hide/log\"))"),
                "absolute .hide/log write-deny must be threaded:\n{}",
                rendered.profile_text
            );
            // Worktree write confinement.
            assert!(rendered.profile_text.contains("(allow file-write* (subpath \"/tmp/wt\"))"));
            // Proxy egress route (S5b).
            assert!(rendered.profile_text.contains("localhost:8443"));
        }

        #[test]
        fn render_options_worktree_falls_back_to_workspace_root() {
            let config = ShellConfig { workspace_root: Some("/tmp/ws".to_string()), ..Default::default() };
            let opts = sandbox_render_options(&config);
            assert_eq!(opts.worktree_root.as_deref(), Some("/tmp/ws"));
        }

        #[tokio::test]
        async fn shell_plan_renders_sandbox_profile() {
            let tool = ShellPlanTool::default();
            let result = tool
                .call(
                    json!({ "argv": ["cargo", "test"] }),
                    ToolCtx { grant_id: None, deadline_ms: None, output_cap_bytes: 65536 },
                )
                .await;
            let sc = result.structured_content.unwrap();
            assert_eq!(sc["executed"], false);
            assert!(sc["sandbox_profile"].as_str().unwrap().contains("(deny default)"));
        }
    }
}
#[rustfmt::skip]
pub mod spec_helpers {
    //! Centralized [`ToolSpec`] builders so the catalog declares schemas concisely
    //! and consistently (ch.03 §4.2.1). Annotations follow the catalog table (§4.6):
    //! read-only tools are `read_only:true` non-destructive; process/edit tools are
    //! destructive.

    use hide_core::tool::{ToolAnnotations, ToolSpec};
    use serde_json::{json, Value};

    /// A read-only filesystem/query tool (auto policy, idempotent).
    pub fn read_spec(
        name: &str,
        title: &str,
        description: &str,
        input_schema: Value,
        output_schema: Option<Value>,
        cap_bytes: u64,
    ) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            title: title.to_string(),
            version: "0.1.0".to_string(),
            wire_version: 1,
            description: description.to_string(),
            input_schema,
            output_schema,
            annotations: ToolAnnotations { read_only: true, destructive: false, idempotent: true, open_world: false },
            capabilities_required: vec!["fs.read".to_string()],
            output_cap_bytes: cap_bytes,
            timeout_ms: 15_000,
        }
    }

    /// A filesystem write/edit tool (ask policy, destructive when overwriting).
    pub fn write_spec(
        name: &str,
        title: &str,
        description: &str,
        input_schema: Value,
        output_schema: Option<Value>,
    ) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            title: title.to_string(),
            version: "0.1.0".to_string(),
            wire_version: 1,
            description: description.to_string(),
            input_schema,
            output_schema,
            annotations: ToolAnnotations { read_only: false, destructive: true, idempotent: false, open_world: false },
            capabilities_required: vec!["fs.write".to_string()],
            output_cap_bytes: 256 * 1024,
            timeout_ms: 15_000,
        }
    }

    /// A process-execution tool (shell/test/build) — destructive, open-world.
    pub fn exec_spec(name: &str, title: &str, description: &str, cap_bytes: u64, timeout_ms: u64) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            title: title.to_string(),
            version: "0.1.0".to_string(),
            wire_version: 1,
            description: description.to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "argv": { "type": "array", "items": {"type":"string"},
                              "description": "command + args; argv form avoids shell injection" },
                    "cwd": { "type": "string" },
                    "env": { "type": "object" }
                },
                "required": ["argv"],
                "additionalProperties": false
            }),
            output_schema: Some(json!({
                "type": "object",
                "properties": {
                    "exit_code": {"type":"integer"},
                    "stdout": {"type":"string"},
                    "stderr": {"type":"string"},
                    "stdout_truncated": {"type":"boolean"}
                },
                "required": ["exit_code", "stdout", "stderr"]
            })),
            annotations: ToolAnnotations { read_only: false, destructive: true, idempotent: false, open_world: true },
            capabilities_required: vec!["shell.exec".to_string()],
            output_cap_bytes: cap_bytes,
            timeout_ms,
        }
    }

    /// The `shell.plan` spec — pure, non-effecting describe-only tool.
    pub fn plan_spec() -> ToolSpec {
        ToolSpec {
            name: "shell.plan".to_string(),
            title: "Plan shell command".to_string(),
            version: "0.1.0".to_string(),
            wire_version: 1,
            description: "Validate a command and render its sandbox profile without executing.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": { "argv": { "type": "array", "items": {"type":"string"} } },
                "required": ["argv"],
                "additionalProperties": false
            }),
            output_schema: None,
            annotations: ToolAnnotations { read_only: true, destructive: false, idempotent: true, open_world: false },
            capabilities_required: vec!["shell.exec".to_string()],
            output_cap_bytes: 64 * 1024,
            timeout_ms: 5_000,
        }
    }

    /// A git read tool.
    pub fn git_read_spec(name: &str, title: &str, description: &str, input_schema: Value) -> ToolSpec {
        let mut s = read_spec(name, title, description, input_schema, None, 256 * 1024);
        s.capabilities_required = vec!["git.read".to_string()];
        s
    }

    /// A git write tool (ask policy).
    pub fn git_write_spec(name: &str, title: &str, description: &str, input_schema: Value) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            title: title.to_string(),
            version: "0.1.0".to_string(),
            wire_version: 1,
            description: description.to_string(),
            input_schema,
            output_schema: None,
            annotations: ToolAnnotations { read_only: false, destructive: false, idempotent: false, open_world: false },
            capabilities_required: vec!["git.write".to_string()],
            output_cap_bytes: 256 * 1024,
            timeout_ms: 30_000,
        }
    }
}

pub use registry::{register_builtin_tools, register_builtin_tools_with};
pub use shell::ShellConfig;
