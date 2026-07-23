//! Bounded programmatic-tool capability (Bible Book V, sec 18-19; sec 78.1 #12).
//!
//! This wires the scaffolded [`hide_program_runtime`] sandbox into the backend as
//! a REAL capability: a bounded, deterministic program can coordinate several
//! READ-ONLY tools in a single call and return a filtered, structured result,
//! amplifying a future local model by removing tool round-trips.
//!
//! # What is wired
//!
//! * [`HostProgramHandles`] is the host's implementation of the runtime's
//!   read-only [`HostHandles`] trait. Each granted handle is backed by a REAL
//!   read path:
//!   - `search.text`      -> `hide_tools::search::SearchTextTool`
//!   - `file.read`        -> `hide_tools::fs::FsReadTool` (bounded by an output cap)
//!   - `index.references` -> the backend `code_index` (`InMemoryCodeIndex`)
//!   - `git.diff`         -> `hide_tools::git::GitDiffTool`
//!   - `git.log`          -> `hide_tools::git::GitLogTool`
//!   Every other read handle (`search.symbol`, `diagnostic.list`,
//!   `test.result.read`, `artifact.read`, `mcp.readonly`) returns an honest
//!   DEFERRED error rather than a faked result, and is not granted by default.
//!   There is NO write/exec/network handle to expose: the runtime's
//!   [`HandleName`] enum cannot even name one.
//!
//! * [`BackendHost::run_program`] parses a program (pure data), binds the caller's
//!   `input` as a variable, runs it through the runtime under
//!   [`HostProgramHandles`] + [`Limits`], and returns a [`ProgramRunResult`] with
//!   the structured `output`, the preserved `citations`, and any `write_proposals`
//!   the program prepared. A write proposal is RETURNED, never executed: it must
//!   travel the normal action/approval plane. Each run surfaces a `program_run`
//!   UiEvent and persists a durable `program.run` event.
//!
//! # The async/sync bridge
//!
//! The runtime interpreter is synchronous and calls handles synchronously, but
//! the hide-tools read tools are async (and `git.*` needs a tokio reactor for
//! `tokio::process`). We keep the two worlds cleanly separated: the whole
//! synchronous `run()` is offloaded to `spawn_blocking` so it never blocks the
//! reactor, and each individual tool future is driven to completion on a fresh,
//! dedicated thread that owns its own current-thread tokio runtime. A brand-new
//! thread has no ambient runtime context, so this never nests runtimes and works
//! whether or not the caller is already inside one.
//!
//! Model-free: nothing here runs a model. Authoring a program from a
//! natural-language goal is a model-bearing job, deferred to a higher layer; this
//! layer only EXECUTES programs it is handed, deterministically.

use std::collections::BTreeSet;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use hide_core::api::{UiEvent, UiEventKind};
use hide_core::event::NewEvent;
use hide_core::ids::SessionId;
use hide_core::tool::{Tool, ToolCtx, ToolResult};
use hide_program_runtime::{
    map_of, run, Citation, Expr, HandleError, HandleGrants, HandleName, HostHandles, Limits,
    Program, RunOutput, RuntimeError, Usage, Value, WriteProposal, CITATIONS_KEY,
};
use hide_tools::fs::FsReadTool;
use hide_tools::git::{GitDiffTool, GitLogTool};
use hide_tools::search::SearchTextTool;
use hawking_index::{CodeIndex, InMemoryCodeIndex};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::host::BackendHost;

/// The structured outcome of a program run. The program's `output` is the runtime
/// value it returned; `citations` is the deduplicated union of every citation
/// carried anywhere inside that output (provenance is never silently dropped);
/// `write_proposals` are the mutations the program PREPARED but that the runtime
/// executed none of.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramRunResult {
    pub output: Value,
    pub citations: Vec<Citation>,
    pub write_proposals: Vec<WriteProposal>,
    pub usage: Usage,
}

impl ProgramRunResult {
    fn from_run(run_output: RunOutput) -> Self {
        let citations = collect_citations(&run_output.value);
        Self {
            output: run_output.value,
            citations,
            write_proposals: run_output.proposals,
            usage: run_output.usage,
        }
    }
}

/// Everything that can go wrong running a program through the backend. The
/// runtime's typed failures (a granted-handle denial, a schema mismatch, a
/// resource-limit breach carrying its [`hide_program_runtime::LimitKind`]) are
/// preserved as-is in [`ProgramRunError::Runtime`] so a caller can match on them.
#[derive(Debug, thiserror::Error)]
pub enum ProgramRunError {
    /// The program source or the input value was not valid. A forbidden capability
    /// (write/exec/network) surfaces HERE: `HandleName` only deserializes the ten
    /// read handles, so a program naming `fs.write` fails to parse.
    #[error("program parse error: {0}")]
    Parse(String),
    /// A typed runtime failure (handle-not-granted, type/schema error, or a
    /// resource-limit breach with its `LimitKind`).
    #[error("program runtime error: {0}")]
    Runtime(RuntimeError),
    /// The durable `program.run` event could not be persisted.
    #[error("program run persistence error: {0}")]
    Persist(String),
    /// The blocking runner thread failed to join (should not happen).
    #[error("program run join error: {0}")]
    Join(String),
}

/// The host-supplied bridge from the runtime's read handles to REAL backend read
/// paths. Read-only by construction: there is no write/exec/network handle to
/// name, and this type never mutates the workspace.
pub struct HostProgramHandles {
    workspace_root: PathBuf,
    output_cap_bytes: u64,
    search_tool: SearchTextTool,
    fs_read_tool: FsReadTool,
    git_diff_tool: GitDiffTool,
    git_log_tool: GitLogTool,
    code_index: Arc<InMemoryCodeIndex>,
}

impl HostProgramHandles {
    /// Build the bridge for a workspace. `output_cap_bytes` bounds a single
    /// `file.read` (and the head kept when a read spills), so a program cannot
    /// pull an unbounded blob through one handle.
    pub fn new(
        workspace_root: PathBuf,
        code_index: Arc<InMemoryCodeIndex>,
        output_cap_bytes: u64,
    ) -> Self {
        Self {
            workspace_root,
            output_cap_bytes,
            search_tool: SearchTextTool::default(),
            fs_read_tool: FsReadTool::default(),
            git_diff_tool: GitDiffTool::default(),
            git_log_tool: GitLogTool::default(),
            code_index,
        }
    }

    /// The read handles the host actually backs with a real read path. These are
    /// the only capabilities a program is granted by default. Every entry is
    /// read-oriented; there is intentionally no write/exec/network handle to add.
    pub fn default_grants() -> HandleGrants {
        HandleGrants::of([
            HandleName::SearchText,
            HandleName::FileRead,
            HandleName::IndexReferences,
            HandleName::GitDiff,
            HandleName::GitLog,
        ])
    }

    fn ctx(&self) -> ToolCtx {
        ToolCtx {
            grant_id: None,
            deadline_ms: None,
            output_cap_bytes: self.output_cap_bytes,
        }
    }

    fn call_search_text(&self, args: &Value) -> Result<Value, HandleError> {
        let json = runtime_to_json(args, "search.text")?;
        let tool = self.search_tool.clone();
        let ctx = self.ctx();
        let result = drive_to_completion(async move { tool.call(json, ctx).await });
        let sc = ok_structured(&result, "search.text")?;
        let matches = sc
            .get("matches")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();
        let mut rows = Vec::with_capacity(matches.len());
        for m in matches {
            let path = m.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let line = m.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
            let text = m.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let rec = map_of([
                ("path", Value::from(path)),
                ("line", Value::Int(line as i64)),
                ("text", Value::from(text)),
            ])
            .with_merged_citations(&[
                Citation::new("search.text").with_locator(format!("{path}:{line}"))
            ]);
            rows.push(rec);
        }
        Ok(Value::List(rows))
    }

    fn call_file_read(&self, args: &Value) -> Result<Value, HandleError> {
        let json = runtime_to_json(args, "file.read")?;
        let tool = self.fs_read_tool.clone();
        let ctx = self.ctx();
        let result = drive_to_completion(async move { tool.call(json, ctx).await });
        let sc = ok_structured(&result, "file.read")?;
        let path = sc.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let content = sc.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let bytes = sc.get("bytes").and_then(|v| v.as_u64()).unwrap_or(0);
        let truncated = sc.get("truncated").and_then(|v| v.as_bool()).unwrap_or(false);
        let encoding = sc.get("encoding").and_then(|v| v.as_str()).unwrap_or("utf8");
        let rec = map_of([
            ("path", Value::from(path)),
            ("content", Value::from(content)),
            ("bytes", Value::Int(bytes as i64)),
            ("truncated", Value::Bool(truncated)),
            ("encoding", Value::from(encoding)),
        ])
        .with_merged_citations(&[Citation::new("file.read").with_locator(path)]);
        Ok(rec)
    }

    fn call_index_references(&self, args: &Value) -> Result<Value, HandleError> {
        let symbol = match args {
            Value::Str(s) => s.clone(),
            _ => args
                .as_map()
                .and_then(|m| m.get("symbol"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| {
                    HandleError::new(
                        "index.references",
                        "expected a symbol string or a {symbol} map",
                    )
                })?,
        };
        let index = self.code_index.clone();
        let sym = symbol.clone();
        let occurrences = drive_to_completion(async move { index.references(&sym).await })
            .map_err(|e| HandleError::new("index.references", e.to_string()))?;
        let mut rows = Vec::with_capacity(occurrences.len());
        for occ in occurrences {
            let locator = match &occ.range {
                Some(range) => format!("{}:{}", occ.file, range.start_line),
                None => occ.file.clone(),
            };
            let rec = map_of([
                ("symbol", Value::from(occ.symbol.as_str())),
                ("file", Value::from(occ.file.as_str())),
                ("role", Value::from(occ.role.as_str())),
            ])
            .with_merged_citations(&[Citation::new("index.references").with_locator(locator)]);
            rows.push(rec);
        }
        Ok(Value::List(rows))
    }

    fn call_git(&self, handle: HandleName, args: &Value) -> Result<Value, HandleError> {
        let source = handle.as_str();
        let mut json = runtime_to_json(args, source)?;
        // Confine git reads to the workspace: default `cwd` to the workspace root
        // when the program did not pin one.
        let has_cwd = json.get("cwd").and_then(|v| v.as_str()).is_some();
        if !has_cwd {
            let root = self.workspace_root.to_string_lossy().to_string();
            match json {
                serde_json::Value::Object(ref mut map) => {
                    map.insert("cwd".to_string(), serde_json::Value::String(root));
                }
                _ => json = json!({ "cwd": root }),
            }
        }
        let ctx = self.ctx();
        let result = match handle {
            HandleName::GitDiff => {
                let tool = self.git_diff_tool.clone();
                drive_to_completion(async move { tool.call(json, ctx).await })
            }
            HandleName::GitLog => {
                let tool = self.git_log_tool.clone();
                drive_to_completion(async move { tool.call(json, ctx).await })
            }
            _ => unreachable!("call_git is only reached for git.diff / git.log"),
        };
        // A non-zero git exit is DATA (EXEC_NONZERO), not a failure: `ok` stays
        // true and the program can read `exit_code`/`stderr`. Only a spawn fault is
        // `ok:false` and surfaces as a handle error.
        let sc = ok_structured(&result, source)?;
        let stdout = sc.get("stdout").and_then(|v| v.as_str()).unwrap_or("");
        let stderr = sc.get("stderr").and_then(|v| v.as_str()).unwrap_or("");
        let cwd = sc.get("cwd").and_then(|v| v.as_str()).unwrap_or("");
        let exit = result.exit_code.unwrap_or(-1);
        let rec = map_of([
            ("exit_code", Value::Int(exit as i64)),
            ("stdout", Value::from(stdout)),
            ("stderr", Value::from(stderr)),
            ("cwd", Value::from(cwd)),
        ])
        .with_merged_citations(&[Citation::new(source).with_locator(cwd)]);
        Ok(rec)
    }
}

impl HostHandles for HostProgramHandles {
    fn call(
        &self,
        handle: HandleName,
        args: &Value,
        _attempt: u32,
    ) -> Result<Value, HandleError> {
        match handle {
            HandleName::SearchText => self.call_search_text(args),
            HandleName::FileRead => self.call_file_read(args),
            HandleName::IndexReferences => self.call_index_references(args),
            HandleName::GitDiff | HandleName::GitLog => self.call_git(handle, args),
            // Honest DEFERRED: these read paths are not wired into the program
            // sandbox yet. We return a typed error rather than fake a result (and
            // they are not in `default_grants`, so a default run cannot reach them).
            // Crucially, none of these is a write/exec/network capability.
            HandleName::SearchSymbol
            | HandleName::DiagnosticList
            | HandleName::TestResultRead
            | HandleName::ArtifactRead
            | HandleName::McpReadonly => Err(HandleError::new(
                handle.as_str(),
                "read handle not wired into the program sandbox yet (DEFERRED)",
            )),
        }
    }
}

/// Drive one async tool/index future to completion on a fresh dedicated thread
/// with its own current-thread tokio runtime. A brand-new thread carries no
/// ambient runtime context, so this never nests runtimes and safely drives the
/// `tokio::process`-backed git tools alongside the pure-fs ones.
fn drive_to_completion<T, F>(fut: F) -> T
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("hide-backend: build program-handle tokio runtime");
        rt.block_on(fut)
    })
    .join()
    .expect("hide-backend: program-handle driver thread panicked")
}

/// Encode runtime handle args into JSON for a hide-tools tool call.
fn runtime_to_json(args: &Value, handle: &str) -> Result<serde_json::Value, HandleError> {
    serde_json::to_value(args)
        .map_err(|e| HandleError::new(handle, format!("cannot encode handle args: {e}")))
}

/// Pull the structured body out of a tool result, turning a tool failure
/// (`ok:false`) into a handle error. A non-zero process exit is NOT a failure
/// here (`ok` stays true) and is left for the caller to interpret.
fn ok_structured<'r>(
    result: &'r ToolResult,
    handle: &str,
) -> Result<&'r serde_json::Value, HandleError> {
    if !result.ok {
        let message = result
            .error
            .as_ref()
            .map(|e| format!("{}: {}", e.code, e.message))
            .unwrap_or_else(|| "tool call failed".to_string());
        return Err(HandleError::new(handle, message));
    }
    result
        .structured_content
        .as_ref()
        .ok_or_else(|| HandleError::new(handle, "tool returned no structured content"))
}

/// Recursively gather every citation carried anywhere inside a value, deduped by
/// stable key and in stable (encounter) order. Provenance rides inside records
/// under the reserved citations field; a program's output is usually a
/// list/map of records, so the top-level value alone would miss them.
fn collect_citations(value: &Value) -> Vec<Citation> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    collect_citations_into(value, &mut out, &mut seen);
    out
}

fn collect_citations_into(value: &Value, out: &mut Vec<Citation>, seen: &mut BTreeSet<String>) {
    match value {
        Value::Map(map) => {
            for citation in value.citations() {
                if seen.insert(citation.dedup_key()) {
                    out.push(citation);
                }
            }
            for (key, child) in map {
                if key == CITATIONS_KEY {
                    continue;
                }
                collect_citations_into(child, out, seen);
            }
        }
        Value::List(items) => {
            for item in items {
                collect_citations_into(item, out, seen);
            }
        }
        _ => {}
    }
}

/// Wrap a program so the caller's `input` value is bound as a variable named
/// `input`, in scope for the whole program. Uses only the existing runtime AST.
fn wrap_program_with_input(program: Program, input: Value) -> Program {
    let root = Expr::let_("input", Expr::Lit { value: input }, program.root);
    Program {
        root,
        seed: program.seed,
        clock_start_ms: program.clock_start_ms,
    }
}

/// The default resource envelope a program runs under: the runtime's conservative
/// `strict` bounds (100k instructions, 5s virtual wall time, 128 tool calls,
/// 256 KiB output, ...).
pub fn default_program_limits() -> Limits {
    Limits::strict()
}

/// The per-`file.read` output cap derived from the run's output budget, so a
/// single read cannot pull more than the whole program is allowed to return.
fn handle_output_cap(limits: &Limits) -> u64 {
    limits.output_bytes.max(1)
}

impl BackendHost {
    /// Run a bounded programmatic-tool program: coordinate read-only tools once and
    /// return a filtered, structured result. `program_source` is the JSON-encoded
    /// [`Program`]; `input_json` is bound as the `input` variable. The default
    /// read-only grants + [`default_program_limits`] apply.
    ///
    /// Any [`WriteProposal`] the program prepares is RETURNED in the result, never
    /// executed here; it must travel the normal action/approval plane. A
    /// `program_run` UiEvent is surfaced and a durable `program.run` event is
    /// persisted.
    pub async fn run_program(
        &self,
        session_id: SessionId,
        program_source: &str,
        input_json: serde_json::Value,
    ) -> Result<ProgramRunResult, ProgramRunError> {
        self.run_program_with_limits(
            session_id,
            program_source,
            input_json,
            default_program_limits(),
        )
        .await
    }

    /// [`run_program`](Self::run_program) with an explicit resource envelope. The
    /// grant set stays the default read-only surface; only the [`Limits`] change.
    pub async fn run_program_with_limits(
        &self,
        session_id: SessionId,
        program_source: &str,
        input_json: serde_json::Value,
        limits: Limits,
    ) -> Result<ProgramRunResult, ProgramRunError> {
        // Parse the program (pure data). A forbidden capability cannot be named:
        // `HandleName` only deserializes the ten read handles, so a `fs.write`
        // handle node fails right here.
        let program: Program = serde_json::from_str(program_source)
            .map_err(|e| ProgramRunError::Parse(format!("program: {e}")))?;
        let input: Value = serde_json::from_value(input_json)
            .map_err(|e| ProgramRunError::Parse(format!("input: {e}")))?;
        let program = wrap_program_with_input(program, input);

        let handles = HostProgramHandles::new(
            self.services.config.workspace_root.clone(),
            self.services.code_index.clone(),
            handle_output_cap(&limits),
        );
        let grants = HostProgramHandles::default_grants();

        // The interpreter is synchronous and calls handles synchronously, each of
        // which drives a real async tool to completion on a dedicated thread. Run
        // the whole thing off the reactor via `spawn_blocking`.
        let outcome = tokio::task::spawn_blocking(move || run(&program, &handles, &grants, limits))
            .await
            .map_err(|e| ProgramRunError::Join(e.to_string()))?;

        match outcome {
            Ok(run_output) => {
                let result = ProgramRunResult::from_run(run_output);
                let payload = json!({
                    "status": "ok",
                    "usage": result.usage,
                    "citations": result.citations.len(),
                    "write_proposals": result.write_proposals.len(),
                    "output": result.output,
                });
                let ui = json!({
                    "kind": "program_run",
                    "status": "ok",
                    "tool_calls": result.usage.tool_calls,
                    "instructions": result.usage.instructions,
                    "citations": result.citations.len(),
                    "write_proposals": result.write_proposals.len(),
                });
                self.emit_program_run(&session_id, payload, ui).await?;
                Ok(result)
            }
            Err(err) => {
                let kind = err.limit_kind().map(|k| k.as_str().to_string());
                let payload = json!({
                    "status": "error",
                    "error": err.to_string(),
                    "limit_kind": kind,
                });
                let ui = json!({
                    "kind": "program_run",
                    "status": "error",
                    "error": err.to_string(),
                    "limit_kind": kind,
                });
                self.emit_program_run(&session_id, payload, ui).await?;
                Err(ProgramRunError::Runtime(err))
            }
        }
    }

    /// Persist the durable `program.run` event and surface the live `program_run`
    /// UiEvent on the push bus. Persistence is load-bearing (a failure to record is
    /// surfaced), the UiEvent rides the same seq.
    async fn emit_program_run(
        &self,
        session_id: &SessionId,
        payload: serde_json::Value,
        ui: serde_json::Value,
    ) -> Result<(), ProgramRunError> {
        let event = self
            .services
            .event_log
            .append(NewEvent::system(session_id.clone(), "program.run", payload))
            .await
            .map_err(|e| ProgramRunError::Persist(e.to_string()))?;
        self.ui_bus().publish(UiEvent {
            seq: event.seq,
            session_id: Some(session_id.clone()),
            kind: UiEventKind::Custom(ui),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::BackendServices;
    use hide_core::config::HideConfig;
    use hide_core::ids::now_ms;
    use hide_program_runtime::{BinOp, Lambda, LimitKind, Operator, Order};
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "hide_program_{tag}_{}_{}_{}",
            std::process::id(),
            now_ms(),
            N.fetch_add(1, Ordering::SeqCst)
        ))
    }

    /// A host over a fresh temp workspace with a `work/` subtree of seeded files.
    fn seeded_host(tag: &str) -> (BackendHost, PathBuf) {
        let dir = unique(tag);
        let work = dir.join("work");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::write(
            work.join("alpha.rs"),
            "pub fn compute_needle() -> i32 { 1 }\n",
        )
        .unwrap();
        std::fs::write(
            work.join("beta.rs"),
            "pub fn compute_needle() -> i32 { 2 }\n",
        )
        .unwrap();
        std::fs::write(work.join("gamma.txt"), "compute_needle marker\n").unwrap();
        let config = HideConfig::for_workspace(&dir);
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        (host, work)
    }

    fn source_of(program: &Program) -> String {
        serde_json::to_string(program).expect("program serializes")
    }

    // A program that coordinates TWO read tools once: full-text search + a file
    // read. It filters the hits to `.rs`, projects them while preserving each
    // hit's citation, ranks by path, and also reads one file's content -- all in a
    // single structured result.
    fn coordinate_program(work: &Path) -> Program {
        let work_str = work.to_string_lossy().to_string();
        let alpha_str = work.join("alpha.rs").to_string_lossy().to_string();

        let search = Expr::handle(
            HandleName::SearchText,
            Expr::map_lit([
                ("pattern", Expr::lit("compute_needle")),
                ("root", Expr::lit(work_str.as_str())),
            ]),
        );
        let only_rs = Expr::op(Operator::Filter {
            input: Box::new(search),
            pred: Lambda::new(
                "h",
                Expr::bin(
                    BinOp::Contains,
                    Expr::field(Expr::var("h"), ["path"]),
                    Expr::lit(".rs"),
                ),
            ),
        });
        let projected = Expr::op(Operator::CitationPreservation {
            input: Box::new(only_rs),
            func: Lambda::new(
                "h",
                Expr::map_lit([
                    ("path", Expr::field(Expr::var("h"), ["path"])),
                    ("line", Expr::field(Expr::var("h"), ["line"])),
                ]),
            ),
        });
        let ranked = Expr::op(Operator::Rank {
            input: Box::new(projected),
            key: Lambda::new("h", Expr::field(Expr::var("h"), ["path"])),
            order: Order::Asc,
            limit: Some(10),
        });
        let alpha_content = Expr::field(
            Expr::handle(
                HandleName::FileRead,
                Expr::map_lit([("path", Expr::lit(alpha_str.as_str()))]),
            ),
            ["content"],
        );
        let root = Expr::map_lit([("matches", ranked), ("alpha", alpha_content)]);
        Program::new(root)
    }

    #[tokio::test]
    async fn program_coordinates_read_tools_deterministically_with_citations() {
        let (host, work) = seeded_host("coord");
        let session = host.services.session();
        let source = source_of(&coordinate_program(&work));

        // Subscribe before running so we catch the live program_run UiEvent.
        let mut ui_rx = host.subscribe_ui();

        let first = host
            .run_program(session.clone(), &source, json!(null))
            .await
            .expect("program runs");
        let second = host
            .run_program(session.clone(), &source, json!(null))
            .await
            .expect("program runs again");

        // Deterministic: byte-identical structured output + citations across runs.
        assert_eq!(first.output, second.output);
        assert_eq!(first.citations, second.citations);

        // Structured result: two `.rs` hits, ranked by path (alpha before beta),
        // the `.txt` filtered out; plus the coordinated file read.
        let matches = first
            .output
            .get_path(&["matches".into()])
            .and_then(Value::as_list)
            .expect("matches is a list");
        assert_eq!(matches.len(), 2, "only the two .rs files match");
        let first_path = matches[0]
            .get_path(&["path".into()])
            .and_then(Value::as_str)
            .unwrap();
        let second_path = matches[1]
            .get_path(&["path".into()])
            .and_then(Value::as_str)
            .unwrap();
        assert!(first_path.ends_with("alpha.rs"), "ranked by path: {first_path}");
        assert!(second_path.ends_with("beta.rs"), "ranked by path: {second_path}");

        // Citations survived filter + projection + rank on every row.
        for row in matches {
            let cites = row.citations();
            assert_eq!(cites.len(), 1, "each ranked row keeps its search citation");
            assert_eq!(cites[0].source, "search.text");
        }
        // The coordinated file read landed in the same structured result.
        let alpha = first
            .output
            .get_path(&["alpha".into()])
            .and_then(Value::as_str)
            .expect("alpha content is a string");
        assert_eq!(alpha, "pub fn compute_needle() -> i32 { 1 }\n");

        // Top-level preserved citations: the two search.text hits, deduped.
        assert_eq!(first.citations.len(), 2, "two distinct search.text citations");
        assert!(first.citations.iter().all(|c| c.source == "search.text"));

        // No mutation was prepared.
        assert!(first.write_proposals.is_empty());

        // Durable `program.run` event persisted.
        let events = host
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert!(
            events.iter().filter(|e| e.kind == "program.run").count() >= 2,
            "two runs persist two program.run events"
        );

        // Live `program_run` UiEvent surfaced on the push bus.
        let ui = ui_rx.recv().await.expect("a program_run UiEvent");
        match ui.kind {
            UiEventKind::Custom(v) => {
                assert_eq!(v.get("kind").and_then(|k| k.as_str()), Some("program_run"));
                assert_eq!(v.get("status").and_then(|k| k.as_str()), Some("ok"));
            }
            other => panic!("expected a Custom program_run UiEvent, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn forbidden_capabilities_cannot_be_expressed_or_reached() {
        let (host, _work) = seeded_host("forbidden");
        let session = host.services.session();

        // A write/exec/network handle simply cannot be NAMED: `HandleName` only
        // deserializes the ten read handles, so a program that tries to build a
        // filesystem write (or a shell exec) fails to parse -- there is no such
        // handle to reach.
        for forbidden in ["fs.write", "shell.exec", "net.fetch"] {
            let evil = format!(
                r#"{{"root":{{"expr":"handle","name":"{forbidden}","args":{{"expr":"lit","value":null}}}},"seed":0,"clock_start_ms":0}}"#
            );
            let err = host
                .run_program(session.clone(), &evil, json!(null))
                .await
                .expect_err("a forbidden capability must not parse");
            assert!(
                matches!(err, ProgramRunError::Parse(_)),
                "{forbidden} should be unnameable, got {err:?}"
            );
        }

        // A real read handle that is NOT granted by default is denied at the grant
        // plane with a typed error -- the host, not the program, decides grants.
        let ungranted = Program::new(Expr::handle(HandleName::McpReadonly, Expr::lit("x")));
        let err = host
            .run_program(session.clone(), &source_of(&ungranted), json!(null))
            .await
            .expect_err("an ungranted read handle must be denied");
        assert!(
            matches!(err, ProgramRunError::Runtime(RuntimeError::HandleNotGranted(_))),
            "expected HandleNotGranted, got {err:?}"
        );
    }

    #[tokio::test]
    async fn write_proposal_is_returned_but_never_executed() {
        let (host, work) = seeded_host("proposal");
        let session = host.services.session();
        let target = work.join("SHOULD_NOT_BE_WRITTEN.txt");
        let target_str = target.to_string_lossy().to_string();

        let proposal = Expr::propose_write(Expr::map_lit([
            ("kind", Expr::lit("edit")),
            ("summary", Expr::lit("rename compute_needle across the crate")),
            (
                "payload",
                Expr::map_lit([
                    ("path", Expr::lit(target_str.as_str())),
                    ("content", Expr::lit("this must never be written")),
                ]),
            ),
        ]));
        let program = Program::new(Expr::map_lit([("staged", proposal)]));

        let result = host
            .run_program(session, &source_of(&program), json!(null))
            .await
            .expect("program runs");

        // The proposal is returned...
        assert_eq!(result.write_proposals.len(), 1);
        let wp = &result.write_proposals[0];
        assert_eq!(wp.kind, hide_program_runtime::WriteKind::Edit);
        assert_eq!(wp.summary, "rename compute_needle across the crate");
        assert_eq!(
            wp.payload.get_path(&["path".into()]).and_then(Value::as_str),
            Some(target_str.as_str())
        );
        // ...and NOTHING was written: the mutation never left the sandbox.
        assert!(!target.exists(), "the proposed write must not touch disk");
    }

    #[tokio::test]
    async fn limit_breaches_trip_typed_errors() {
        let (host, work) = seeded_host("limits");
        let session = host.services.session();
        let alpha_str = work.join("alpha.rs").to_string_lossy().to_string();

        // tool-call breach: a single granted read under a zero tool-call budget.
        let read = Program::new(Expr::handle(
            HandleName::FileRead,
            Expr::map_lit([("path", Expr::lit(alpha_str.as_str()))]),
        ));
        let limits = Limits {
            tool_calls: 0,
            ..Limits::strict()
        };
        let err = host
            .run_program_with_limits(session.clone(), &source_of(&read), json!(null), limits)
            .await
            .expect_err("tool-call budget exhausted");
        assert!(
            matches!(
                err,
                ProgramRunError::Runtime(RuntimeError::LimitExceeded {
                    kind: LimitKind::ToolCall,
                    ..
                })
            ),
            "expected a ToolCall limit breach, got {err:?}"
        );

        // instruction breach: a one-instruction budget.
        let trivial = Program::new(Expr::map_lit([
            ("a", Expr::lit(1i64)),
            ("b", Expr::lit(2i64)),
        ]));
        let limits = Limits {
            instructions: 1,
            ..Limits::strict()
        };
        let err = host
            .run_program_with_limits(session.clone(), &source_of(&trivial), json!(null), limits)
            .await
            .expect_err("instruction budget exhausted");
        assert!(
            matches!(
                err,
                ProgramRunError::Runtime(RuntimeError::LimitExceeded {
                    kind: LimitKind::Instruction,
                    ..
                })
            ),
            "expected an Instruction limit breach, got {err:?}"
        );

        // output breach: a one-byte output budget cannot hold the returned value.
        let big = Program::new(Expr::lit("a result larger than one byte"));
        let limits = Limits {
            output_bytes: 1,
            ..Limits::strict()
        };
        let err = host
            .run_program_with_limits(session, &source_of(&big), json!(null), limits)
            .await
            .expect_err("output budget exhausted");
        assert!(
            matches!(
                err,
                ProgramRunError::Runtime(RuntimeError::LimitExceeded {
                    kind: LimitKind::OutputBytes,
                    ..
                })
            ),
            "expected an OutputBytes limit breach, got {err:?}"
        );
    }

    #[tokio::test]
    async fn git_log_handle_bridges_a_real_async_process_tool() {
        // Proves the async->sync bridge drives a `tokio::process`-backed read tool:
        // seed a real git repo, then have a program call the git.log handle and
        // return its stdout.
        let (host, work) = seeded_host("gitlog");
        let session = host.services.session();

        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(&work)
                .output()
        };
        if git(&["init", "-q"]).is_err() {
            eprintln!("skipping git bridge test: git not available");
            return;
        }
        let _ = git(&["config", "user.email", "t@t.t"]);
        let _ = git(&["config", "user.name", "t"]);
        let _ = git(&["add", "-A"]);
        let _ = git(&["commit", "-qm", "seed_commit_marker"]);

        // git.log with cwd defaulted to the workspace root would point at `dir`,
        // not `work`; pin the repo explicitly.
        let work_str = work.to_string_lossy().to_string();
        let program = Program::new(Expr::field(
            Expr::handle(
                HandleName::GitLog,
                Expr::map_lit([("cwd", Expr::lit(work_str.as_str()))]),
            ),
            ["stdout"],
        ));

        let result = host
            .run_program(session, &source_of(&program), json!(null))
            .await
            .expect("git.log program runs");
        let stdout = result.output.as_str().unwrap_or("");
        assert!(
            stdout.contains("seed_commit_marker"),
            "git.log stdout should contain the commit: {stdout:?}"
        );
    }
}
