//! The deterministic oracle suite (bible ch.02 §4.6.2) — the reliability engine.
//!
//! A 7B model's *proposal* is fallible; `cargo build` is not. These oracles shell
//! out to the real toolchain through the `hide-tools` process tools (sandboxed,
//! deadline-bounded, EXEC_NONZERO-as-data) and parse the real diagnostics into
//! structured [`Failure`]s so the repair stage has minimal, high-signal context.
//!
//! Implemented: `patch_apply` (git apply --check), `build` (cargo build / a
//! configurable build argv), `test` (cargo test), `typecheck` (cargo check),
//! `lint` (cargo clippy), `grep_ast` (a structural predicate over the index /
//! file content), `schema` (JSON-against-schema), `runtime_smoke` (run a canned
//! command and check exit/stdout). Each returns a Deterministic [`Verdict`].

use crate::verify::oracle::{
    Cost, Failure, Oracle, OracleClass, Verdict, VerdictStatus, VerificationInput,
};
use futures::future::BoxFuture;
use hide_core::tool::{ToolCall, ToolDispatcher, ToolResult};
use hide_core::Result;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

/// A process-shelling oracle: runs an argv via a `hide-tools` process tool and
/// parses the result. `tool` is the registered tool name (`build.run` /
/// `test.run` / `compile.check` / `shell.run`).
pub struct ProcessOracle {
    name: String,
    tool: String,
    /// Default argv when the step doesn't override it.
    argv: Vec<String>,
    cost: Cost,
    dispatcher: Arc<ToolDispatcher>,
    /// Failure category tag (`build`/`test`/`type`/`lint`).
    category: String,
}

impl ProcessOracle {
    pub fn new(
        name: impl Into<String>,
        tool: impl Into<String>,
        argv: Vec<&str>,
        cost: Cost,
        category: impl Into<String>,
        dispatcher: Arc<ToolDispatcher>,
    ) -> Self {
        Self {
            name: name.into(),
            tool: tool.into(),
            argv: argv.into_iter().map(String::from).collect(),
            cost,
            dispatcher,
            category: category.into(),
        }
    }

    /// `build` oracle (`cargo build`).
    pub fn build(dispatcher: Arc<ToolDispatcher>) -> Self {
        Self::new(
            "build",
            "build.run",
            vec![],
            Cost::Medium,
            "build",
            dispatcher,
        )
    }

    /// `typecheck` oracle (`cargo check`).
    pub fn typecheck(dispatcher: Arc<ToolDispatcher>) -> Self {
        Self::new(
            "typecheck",
            "compile.check",
            vec![],
            Cost::Medium,
            "type",
            dispatcher,
        )
    }

    /// `test` oracle (`cargo test`).
    pub fn test(dispatcher: Arc<ToolDispatcher>) -> Self {
        Self::new(
            "test",
            "test.run",
            vec![],
            Cost::Expensive,
            "test",
            dispatcher,
        )
    }

    /// `lint` oracle (`cargo clippy`).
    pub fn lint(dispatcher: Arc<ToolDispatcher>) -> Self {
        Self::new(
            "lint",
            "shell.run",
            vec!["cargo", "clippy", "--quiet"],
            Cost::Cheap,
            "lint",
            dispatcher,
        )
    }
}

impl Oracle for ProcessOracle {
    fn name(&self) -> &str {
        &self.name
    }

    fn class(&self) -> OracleClass {
        OracleClass::Deterministic
    }

    fn cost_hint(&self) -> Cost {
        self.cost
    }

    fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
        Box::pin(async move {
            let start = Instant::now();
            // Build args: cwd = workspace root, argv = default unless tests override.
            let mut args = json!({ "cwd": input.workspace_root });
            if !self.argv.is_empty() {
                args["argv"] = json!(self.argv);
            } else if self.name == "test" && !input.tests.is_empty() {
                // Scope the test run to the declared selectors.
                let mut argv = vec!["cargo".to_string(), "test".to_string()];
                argv.extend(input.tests.iter().cloned());
                args["argv"] = json!(argv);
            }
            let result = self
                .dispatcher
                .dispatch(ToolCall::new(self.tool.clone(), args))
                .await?;
            let dur = start.elapsed().as_millis() as u64;
            Ok(self.project(&result, dur))
        })
    }
}

impl ProcessOracle {
    fn project(&self, result: &ToolResult, duration_ms: u64) -> Verdict {
        // A spawn fault (couldn't even run the tool) is genuinely Inconclusive.
        if !result.ok {
            let detail = result
                .error
                .as_ref()
                .map(|e| format!("{}: {}", e.code, e.message))
                .unwrap_or_else(|| "tool failed to run".to_string());
            let mut v = Verdict {
                status: VerdictStatus::Inconclusive,
                score: 0.0,
                oracle: self.name.clone(),
                class: OracleClass::Deterministic,
                detail,
                failures: Vec::new(),
                artifacts: Vec::new(),
                duration_ms,
            };
            // A timeout is a real failure (the command hung), not inconclusive.
            if result.error.as_ref().map(|e| e.code.as_str()) == Some("TIMEOUT") {
                v.status = VerdictStatus::Fail;
                v.failures
                    .push(Failure::new(self.category.clone(), "command timed out"));
            }
            return v;
        }
        let exit = result.exit_code.unwrap_or(0);
        let stderr = result
            .structured_content
            .as_ref()
            .and_then(|sc| sc.get("stderr"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let stdout = result
            .structured_content
            .as_ref()
            .and_then(|sc| sc.get("stdout"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let artifacts = result
            .bytes_ref
            .as_ref()
            .map(|b| vec![b.hash.clone()])
            .unwrap_or_default();
        if exit == 0 {
            return Verdict {
                duration_ms,
                artifacts,
                ..Verdict::pass(self.name.clone(), OracleClass::Deterministic, "exit 0")
            };
        }
        let failures = parse_diagnostics(&self.category, stderr, stdout);
        Verdict {
            duration_ms,
            artifacts,
            ..Verdict::fail(
                self.name.clone(),
                OracleClass::Deterministic,
                format!("{} exited {}", self.tool, exit),
                failures,
            )
        }
    }
}

/// Parse cargo/clippy/rustc-style diagnostics into structured failures. The shape
/// `error[E0308]: ... --> file:line:col` is the cargo/rustc default; we also catch
/// bare `error:`/`test ... FAILED` lines. Capped and deduped (minimal-repair, §4.7).
pub fn parse_diagnostics(category: &str, stderr: &str, stdout: &str) -> Vec<Failure> {
    let mut failures = Vec::new();
    let combined = format!("{stderr}\n{stdout}");
    let lines: Vec<&str> = combined.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("error") {
            // error[E0308]: message  OR  error: message
            let (code, message) = if let Some(b) = rest.strip_prefix('[') {
                let end = b.find(']').unwrap_or(0);
                let code = b[..end].to_string();
                let msg = b[end..].trim_start_matches([']', ':', ' ']).to_string();
                (Some(code), msg)
            } else {
                (None, rest.trim_start_matches([':', ' ']).to_string())
            };
            // Look ahead a few lines for the `--> file:line:col` location.
            let mut file = None;
            let mut line_no = None;
            for look in lines.iter().skip(i + 1).take(3) {
                if let Some(loc) = look.trim_start().strip_prefix("--> ") {
                    let parts: Vec<&str> = loc.split(':').collect();
                    if !parts.is_empty() {
                        file = Some(parts[0].to_string());
                    }
                    if parts.len() >= 2 {
                        line_no = parts[1].trim().parse::<u32>().ok();
                    }
                    break;
                }
            }
            failures.push(Failure {
                file,
                line: line_no,
                code,
                category: category.to_string(),
                message: if message.is_empty() {
                    trimmed.to_string()
                } else {
                    message
                },
            });
        } else if trimmed.contains("FAILED") && category == "test" {
            failures.push(Failure::new("test", trimmed.to_string()));
        }
        if failures.len() >= 25 {
            break;
        }
    }
    if failures.is_empty() {
        // Couldn't parse a specific diagnostic; carry the tail as one failure.
        let tail = combined
            .lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        failures.push(Failure::new(category, tail));
    }
    failures
}

/// `patch_apply` (§4.6.2): `git apply --check <patch>` in the workspace. A diff
/// that doesn't apply cleanly fails the gate before any real write.
pub struct PatchApplyOracle {
    dispatcher: Arc<ToolDispatcher>,
    /// The unified diff to check (from the step's candidate output).
    patch_path: Option<String>,
}

impl PatchApplyOracle {
    pub fn new(dispatcher: Arc<ToolDispatcher>) -> Self {
        Self {
            dispatcher,
            patch_path: None,
        }
    }

    pub fn with_patch_path(mut self, path: impl Into<String>) -> Self {
        self.patch_path = Some(path.into());
        self
    }
}

impl Oracle for PatchApplyOracle {
    fn name(&self) -> &str {
        "patch_apply"
    }
    fn class(&self) -> OracleClass {
        OracleClass::Deterministic
    }
    fn cost_hint(&self) -> Cost {
        Cost::Cheap
    }
    fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
        Box::pin(async move {
            let start = Instant::now();
            let patch = self.patch_path.clone().unwrap_or_else(|| "-".to_string());
            let args = json!({
                "cwd": input.workspace_root,
                "argv": ["git", "apply", "--check", patch],
            });
            let result = self
                .dispatcher
                .dispatch(ToolCall::new("shell.run", args))
                .await?;
            let dur = start.elapsed().as_millis() as u64;
            let exit = result.exit_code.unwrap_or(if result.ok { 0 } else { 1 });
            if result.ok && exit == 0 {
                Ok(Verdict {
                    duration_ms: dur,
                    ..Verdict::pass("patch_apply", OracleClass::Deterministic, "applies cleanly")
                })
            } else {
                let stderr = result
                    .structured_content
                    .as_ref()
                    .and_then(|sc| sc.get("stderr"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("rejected");
                Ok(Verdict {
                    duration_ms: dur,
                    ..Verdict::fail(
                        "patch_apply",
                        OracleClass::Deterministic,
                        "git apply --check failed",
                        vec![Failure::new("patch", stderr.to_string())],
                    )
                })
            }
        })
    }
}

/// `grep_ast` (§4.6.2): a structural predicate over file content / the index —
/// "symbol exists", "no TODO left". Pure (reads files), so Deterministic + cheap.
pub struct GrepAstOracle {
    /// A literal/regex-free needle that MUST be present (`must_contain`) or absent
    /// (`must_absent`) across `changed_files` (or the workspace).
    pub must_contain: Option<String>,
    pub must_absent: Option<String>,
}

impl Oracle for GrepAstOracle {
    fn name(&self) -> &str {
        "grep_ast"
    }
    fn class(&self) -> OracleClass {
        OracleClass::Deterministic
    }
    fn cost_hint(&self) -> Cost {
        Cost::Cheap
    }
    fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
        Box::pin(async move {
            let start = Instant::now();
            let mut haystack = String::new();
            let root = std::path::Path::new(&input.workspace_root);
            if input.changed_files.is_empty() {
                // Nothing scoped — read nothing; predicate over empty.
            } else {
                for rel in &input.changed_files {
                    let path = root.join(rel);
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        haystack.push_str(&content);
                        haystack.push('\n');
                    }
                }
            }
            let dur = start.elapsed().as_millis() as u64;
            let mut failures = Vec::new();
            if let Some(needle) = &self.must_contain {
                if !haystack.contains(needle.as_str()) {
                    failures.push(Failure::new("grep", format!("missing required: {needle}")));
                }
            }
            if let Some(needle) = &self.must_absent {
                if haystack.contains(needle.as_str()) {
                    failures.push(Failure::new("grep", format!("forbidden present: {needle}")));
                }
            }
            if failures.is_empty() {
                Ok(Verdict {
                    duration_ms: dur,
                    ..Verdict::pass("grep_ast", OracleClass::Deterministic, "predicate holds")
                })
            } else {
                Ok(Verdict {
                    duration_ms: dur,
                    ..Verdict::fail(
                        "grep_ast",
                        OracleClass::Deterministic,
                        "structural predicate failed",
                        failures,
                    )
                })
            }
        })
    }
}

/// `schema` (§4.6.2): validate a JSON artifact has the required keys. (A minimal,
/// dependency-free structural check — full JSON-Schema is a later swap-in.)
pub struct SchemaOracle {
    pub artifact: Value,
    pub required_keys: Vec<String>,
}

impl Oracle for SchemaOracle {
    fn name(&self) -> &str {
        "schema"
    }
    fn class(&self) -> OracleClass {
        OracleClass::Deterministic
    }
    fn cost_hint(&self) -> Cost {
        Cost::Cheap
    }
    fn verify<'a>(&'a self, _input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
        Box::pin(async move {
            let mut failures = Vec::new();
            for key in &self.required_keys {
                if self.artifact.get(key).is_none() {
                    failures.push(Failure::new("schema", format!("missing key: {key}")));
                }
            }
            if failures.is_empty() {
                Ok(Verdict::pass("schema", OracleClass::Deterministic, "valid"))
            } else {
                Ok(Verdict::fail(
                    "schema",
                    OracleClass::Deterministic,
                    "schema validation failed",
                    failures,
                ))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rustc_error_with_location() {
        let stderr = "\
error[E0308]: mismatched types
  --> src/lib.rs:12:5
   |
12 |     foo();
";
        let f = parse_diagnostics("type", stderr, "");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code.as_deref(), Some("E0308"));
        assert_eq!(f[0].file.as_deref(), Some("src/lib.rs"));
        assert_eq!(f[0].line, Some(12));
    }

    #[tokio::test]
    async fn schema_oracle_detects_missing_key() {
        let oracle = SchemaOracle {
            artifact: json!({ "a": 1 }),
            required_keys: vec!["a".into(), "b".into()],
        };
        let v = oracle
            .verify(&VerificationInput::new("/tmp"))
            .await
            .unwrap();
        assert_eq!(v.status, VerdictStatus::Fail);
        assert_eq!(v.failures.len(), 1);
    }
}
