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
        Self {
            config,
            ..Self::default()
        }
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
            let encoding = args
                .get("encoding")
                .and_then(|v| v.as_str())
                .unwrap_or("auto");
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
                let spill = common::maybe_spill(
                    encoded,
                    ctx.output_cap_bytes as usize,
                    self.config.blobs.as_ref(),
                );
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
                    let spill = common::maybe_spill(
                        encoded,
                        ctx.output_cap_bytes as usize,
                        self.config.blobs.as_ref(),
                    );
                    return finalize_read(path, "base64", spill, total);
                }
            };
            let sliced = match range {
                Some((start, end)) => slice_lines(&text, start, end),
                None => text,
            };
            let total = sliced.len();
            let spill = common::maybe_spill(
                sliced,
                ctx.output_cap_bytes as usize,
                self.config.blobs.as_ref(),
            );
            finalize_read(path, "utf8", spill, total)
        })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move {
            args.get("path")
                .and_then(|v| v.as_str())
                .map(|path| EffectSet {
                    effects: vec![read_effect(path)],
                })
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
            let depth = args
                .get("depth")
                .and_then(|v| v.as_u64())
                .unwrap_or(1)
                .max(1) as usize;
            let include_hidden = args
                .get("include_hidden")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

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
            common::ok(
                json!({ "path": path, "entries": entries }),
                EffectSet::default(),
            )
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
            let create_dirs = args
                .get("create_dirs")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let create_only = args
                .get("create_only")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
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
                    EffectSet {
                        effects: vec![write_effect(path, content.len())],
                    },
                ),
                Err(err) => common::coded("TOOL_FAULT", err.to_string(), false, None),
            }
        })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move {
            let path = args.get("path").and_then(|v| v.as_str())?;
            let content = args.get("content").and_then(|v| v.as_str())?;
            Some(EffectSet {
                effects: vec![write_effect(path, content.len())],
            })
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
                std::fs::read(path)
                    .ok()
                    .map(|b| blake3::hash(&b).to_hex().to_string())
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
                EffectSet {
                    effects: vec![read_effect(path)],
                },
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
            for entry in ignore::WalkBuilder::new(root)
                .git_ignore(true)
                .require_git(false)
                .hidden(true)
                .build()
                .flatten()
            {
                let p = entry.path();
                let rel = p.strip_prefix(root).unwrap_or(p);
                if set.is_match(rel) || set.is_match(p) {
                    matches.push(p.to_string_lossy().to_string());
                }
            }
            matches.sort();
            common::ok(
                json!({ "pattern": pattern, "root": root, "matches": matches }),
                EffectSet::default(),
            )
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
            let timeout_ms = args
                .get("timeout_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(2000)
                .clamp(1, 60_000);
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
    let start = range
        .get("start_line")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let end = range
        .get("end_line")
        .and_then(|v| v.as_u64())
        .unwrap_or(u64::MAX) as usize;
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
    Effect {
        kind: EffectKind::Write,
        target: path.to_string(),
        bytes_hash: None,
        risk: RiskLevel::High,
        metadata,
    }
}

/// Minimal dependency-free base64 (standard alphabet, padded). Used for binary
/// reads so we never hard-error on non-UTF-8 content.
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
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
        let mut tool = FsReadTool::with_config(FsConfig {
            blobs: Some(blobs.clone()),
        });
        // shrink the cap so 20k spills
        tool.spec.output_cap_bytes = 1000;
        reg.register(tool);
        let d = dispatcher(reg);
        let r = d
            .dispatch(ToolCall::new(
                "fs.read",
                json!({ "path": file.to_string_lossy() }),
            ))
            .await
            .unwrap();
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
        let r = d
            .dispatch(ToolCall::new(
                "fs.read",
                json!({ "path": file.to_string_lossy() }),
            ))
            .await
            .unwrap();
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
        let r = d
            .dispatch(ToolCall::new(
                "fs.stat",
                json!({ "path": file.to_string_lossy() }),
            ))
            .await
            .unwrap();
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
            .dispatch(ToolCall::new(
                "fs.glob",
                json!({ "pattern": "**/*.rs", "root": dir.to_string_lossy() }),
            ))
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
