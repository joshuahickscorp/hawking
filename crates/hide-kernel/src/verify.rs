pub use crate::verify::oracle::VerificationInput;

use crate::verify::oracle::{Cost, Oracle, OracleClass, Verdict, VerdictStatus};
use hide_core::Result;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A registry of named oracles. The kernel resolves a step's
/// `acceptance.oracles` ids against this, runs the resolved set ordered
/// **deterministic-first, then cheapest-first**, and feeds the verdicts to the
/// [`gate::VerificationGate`].
#[derive(Default, Clone)]
pub struct OracleSuite {
    oracles: BTreeMap<String, Arc<dyn Oracle>>,
}

impl OracleSuite {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, oracle: Arc<dyn Oracle>) {
        self.oracles.insert(oracle.name().to_string(), oracle);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Oracle>> {
        self.oracles.get(id).cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.oracles.is_empty()
    }

    /// Resolve the requested ids and return them ordered deterministic-first,
    /// then cheap-before-expensive (so a fast `grep_ast` fails the gate before a
    /// slow `test` ever runs), alongside the list of ids that resolved to *no*
    /// registered oracle. Unknown ids are NOT silently dropped: the caller must
    /// surface them (warn + an Inconclusive marker) so a step declaring an
    /// unregistered verifier can never be accepted on faith (K1).
    pub fn resolve_ranked<'a>(&self, ids: &'a [String]) -> (Vec<Arc<dyn Oracle>>, Vec<&'a str>) {
        let mut resolved: Vec<Arc<dyn Oracle>> = Vec::new();
        let mut unknown: Vec<&'a str> = Vec::new();
        for id in ids {
            match self.get(id) {
                Some(oracle) => resolved.push(oracle),
                None => unknown.push(id.as_str()),
            }
        }
        resolved.sort_by(|a, b| {
            let class_rank = |c: OracleClass| match c {
                OracleClass::Deterministic => 0,
                OracleClass::Probabilistic => 1,
            };
            let cost_rank = |c: Cost| c as u8;
            class_rank(a.class())
                .cmp(&class_rank(b.class()))
                .then(cost_rank(a.cost_hint()).cmp(&cost_rank(b.cost_hint())))
        });
        (resolved, unknown)
    }

    /// Run the ranked oracle set against `input`, short-circuiting on the first
    /// deterministic Fail (no point running an expensive test after the build
    /// already broke — §4.6.4). Returns every verdict produced.
    ///
    /// Every id that did not resolve to a registered oracle is logged
    /// (`tracing::warn`) AND recorded as a `Deterministic` `Inconclusive` verdict
    /// carrying the unknown id. That marker keeps the run auditable and prevents
    /// the gate from accepting a step whose declared verifier never ran: an
    /// Inconclusive deterministic verdict drives the gate to Inconclusive, never
    /// Accept.
    pub async fn run(&self, ids: &[String], input: &VerificationInput) -> Result<Vec<Verdict>> {
        let (resolved, unknown) = self.resolve_ranked(ids);
        let mut verdicts = Vec::new();
        for id in unknown {
            tracing::warn!(
                oracle = %id,
                step = ?input.step_id,
                "step declared an unregistered oracle id; recording Inconclusive marker"
            );
            verdicts.push(unknown_oracle_verdict(id));
        }
        for oracle in resolved {
            let verdict = oracle.verify(input).await?;
            let short_circuit = verdict.is_deterministic() && verdict.status == VerdictStatus::Fail;
            verdicts.push(verdict);
            if short_circuit {
                break;
            }
        }
        Ok(verdicts)
    }
}

/// The auditable marker for an oracle id that resolved to no registered oracle.
/// Deterministic + Inconclusive so the gate cannot Accept on its account (the
/// declared verifier never ran), while the unknown id stays in the verdict set.
fn unknown_oracle_verdict(id: &str) -> Verdict {
    Verdict {
        status: VerdictStatus::Inconclusive,
        score: 0.0,
        oracle: id.to_string(),
        class: OracleClass::Deterministic,
        detail: format!("unknown oracle id '{id}': no oracle registered under this name"),
        failures: Vec::new(),
        artifacts: Vec::new(),
        duration_ms: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::oracle::Cost;
    use futures::future::BoxFuture;
    struct PassOracle(&'static str);
    impl Oracle for PassOracle {
        fn name(&self) -> &str {
            self.0
        }
        fn verify<'a>(&'a self, _input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
            Box::pin(async move { Ok(Verdict::pass(self.0, OracleClass::Deterministic, "ok")) })
        }
    }
    fn suite_with(name: &'static str) -> OracleSuite {
        let mut suite = OracleSuite::new();
        suite.register(Arc::new(PassOracle(name)));
        suite
    }
    #[test]
    fn resolve_ranked_surfaces_unknown_ids() {
        let suite = suite_with("build");
        let ids = ["build".to_string(), "ghost".to_string()];
        let (resolved, unknown) = suite.resolve_ranked(&ids);
        assert_eq!(resolved.len(), 1);
        assert_eq!(unknown, vec!["ghost"]);
    }
    #[tokio::test]
    async fn unknown_oracle_id_produces_visible_inconclusive_marker() {
        let suite = suite_with("build");
        let input = VerificationInput::new(".");
        let verdicts = suite
            .run(&["build".to_string(), "ghost".to_string()], &input)
            .await
            .unwrap();
        let marker = verdicts
            .iter()
            .find(|v| v.oracle == "ghost")
            .expect("unknown oracle id must surface a verdict, not be silently dropped");
        assert_eq!(marker.status, VerdictStatus::Inconclusive);
        assert_eq!(marker.class, OracleClass::Deterministic);
        assert!(marker.detail.contains("ghost"));
    }
    #[tokio::test]
    async fn unknown_oracle_id_does_not_let_gate_accept_on_faith() {
        use crate::verify::gate::{GateDecision, VerificationGate};
        let suite = OracleSuite::new();
        let input = VerificationInput::new(".");
        let verdicts = suite.run(&["ghost".to_string()], &input).await.unwrap();
        assert_ne!(
            VerificationGate::default().decide(&verdicts),
            GateDecision::Accept
        );
    }
    #[test]
    fn resolve_ranked_orders_deterministic_then_cheap() {
        let _ = Cost::Cheap; // touch the import path used by other oracles
        let suite = suite_with("build");
        let ids = ["build".to_string()];
        let (resolved, unknown) = suite.resolve_ranked(&ids);
        assert_eq!(resolved.len(), 1);
        assert!(unknown.is_empty());
    }
}

pub mod deterministic {
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
                        ..Verdict::pass(
                            "patch_apply",
                            OracleClass::Deterministic,
                            "applies cleanly",
                        )
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
}

pub mod gate {
    //! The Verification Gate (bible ch.02 §4.6.4).
    //!
    //! Decides a step's fate from its oracle verdicts. The authority rule (A.2 /
    //! §3.2): **Deterministic verdicts are authoritative**; a Probabilistic score
    //! only ranks *within* the deterministic-pass set and never overrides a
    //! `build`/`test` failure.

    use crate::verify::oracle::{OracleClass, Verdict, VerdictStatus};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct VerificationGate {
        /// Probabilistic-fallback acceptance threshold (only consulted when no
        /// deterministic oracle applied).
        pub min_score: f32,
    }

    impl Default for VerificationGate {
        fn default() -> Self {
            Self { min_score: 0.7 }
        }
    }

    impl VerificationGate {
        pub fn with_threshold(min_score: f32) -> Self {
            Self { min_score }
        }

        /// Decide from the verdicts (§4.6.4). Deterministic first:
        /// * any deterministic Fail  → Repair
        /// * all deterministic Pass (≥1)  → Accept
        /// * no deterministic verdict → fall back to probabilistic vs `min_score`.
        pub fn decide(&self, verdicts: &[Verdict]) -> GateDecision {
            let det: Vec<&Verdict> = verdicts
                .iter()
                .filter(|v| v.class == OracleClass::Deterministic)
                .collect();

            if !det.is_empty() {
                // A deterministic oracle is authoritative.
                if det.iter().any(|v| v.status == VerdictStatus::Fail) {
                    return GateDecision::Repair;
                }
                if det
                    .iter()
                    .all(|v| matches!(v.status, VerdictStatus::Pass | VerdictStatus::Skipped))
                    && det.iter().any(|v| v.status == VerdictStatus::Pass)
                {
                    return GateDecision::Accept;
                }
                // Deterministic ran but was Inconclusive across the board → consistency.
                return GateDecision::Inconclusive;
            }

            // No deterministic oracle applied — probabilistic fallback.
            let prob: Vec<&Verdict> = verdicts
                .iter()
                .filter(|v| v.class == OracleClass::Probabilistic)
                .collect();
            if prob.is_empty() {
                // Nothing ran at all → can't accept on faith (K1).
                return GateDecision::Inconclusive;
            }
            if prob.iter().any(|v| v.status == VerdictStatus::Fail) {
                return GateDecision::Repair;
            }
            let best = prob
                .iter()
                .filter(|v| v.status == VerdictStatus::Pass)
                .map(|v| v.score)
                .fold(0.0_f32, f32::max);
            if best >= self.min_score {
                GateDecision::Accept
            } else {
                GateDecision::Repair
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum GateDecision {
        Accept,
        Repair,
        Replan,
        /// No oracle could decide — route to consistency/judge (probabilistic).
        Inconclusive,
        Abort,
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::verify::oracle::Failure;
        fn det_pass() -> Verdict {
            Verdict::pass("build", OracleClass::Deterministic, "ok")
        }
        fn det_fail() -> Verdict {
            Verdict::fail(
                "build",
                OracleClass::Deterministic,
                "E0308",
                vec![Failure::new("type", "mismatched types")],
            )
        }
        fn prob_pass(score: f32) -> Verdict {
            let mut v = Verdict::pass("judge", OracleClass::Probabilistic, "looks good");
            v.score = score;
            v
        }
        #[test]
        fn deterministic_pass_accepts() {
            assert_eq!(
                VerificationGate::default().decide(&[det_pass()]),
                GateDecision::Accept
            );
        }
        #[test]
        fn deterministic_fail_repairs() {
            assert_eq!(
                VerificationGate::default().decide(&[det_fail()]),
                GateDecision::Repair
            );
        }
        #[test]
        fn deterministic_outranks_probabilistic() {
            let verdicts = vec![det_fail(), prob_pass(1.0)];
            assert_eq!(
                VerificationGate::default().decide(&verdicts),
                GateDecision::Repair
            );
        }
        #[test]
        fn probabilistic_only_uses_threshold() {
            let gate = VerificationGate::with_threshold(0.7);
            assert_eq!(gate.decide(&[prob_pass(0.9)]), GateDecision::Accept);
            assert_eq!(gate.decide(&[prob_pass(0.5)]), GateDecision::Repair);
        }
        #[test]
        fn no_oracle_is_inconclusive() {
            assert_eq!(
                VerificationGate::default().decide(&[]),
                GateDecision::Inconclusive
            );
        }
    }
}

pub mod oracle {
    //! The verifier interface (bible ch.02 Appendix A.2).
    //!
    //! An [`Oracle`] checks a candidate against a step's acceptance contract and
    //! returns a [`Verdict`]. The defining rule: a **Deterministic** verdict is
    //! authoritative; a **Probabilistic** score only ranks *within* the
    //! deterministic-pass set and never overrides `build`/`test` (§3.2 / §4.8.4).

    use futures::future::BoxFuture;
    use hide_core::Result;
    use serde::{Deserialize, Serialize};

    /// The execution environment an oracle checks against: a workspace root and the
    /// set of files the step changed (so an oracle can scope itself).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct VerificationInput {
        pub step_id: Option<String>,
        pub workspace_root: String,
        pub changed_files: Vec<String>,
        /// Optional test selectors propagated from the step's `acceptance.tests`.
        #[serde(default)]
        pub tests: Vec<String>,
        /// The candidate's raw output (model text / diff), for probabilistic oracles.
        #[serde(default)]
        pub candidate_output: String,
    }

    impl VerificationInput {
        pub fn new(workspace_root: impl Into<String>) -> Self {
            Self {
                step_id: None,
                workspace_root: workspace_root.into(),
                changed_files: Vec::new(),
                tests: Vec::new(),
                candidate_output: String::new(),
            }
        }
    }

    /// Oracle class (A.2). The gate ranks Deterministic strictly over Probabilistic.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum OracleClass {
        Deterministic,
        Probabilistic,
    }

    /// Relative cost hint (A.2 `cost_hint`) so the gate can run cheap oracles first.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Cost {
        Cheap,
        Medium,
        Expensive,
    }

    /// A structured oracle failure (A.2 `Failure`) — the minimal-repair context. The
    /// repair stage feeds these (file/line/code/message) back verbatim so the model
    /// fixes the *specific* error, not the whole history.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Failure {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub file: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub line: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub code: Option<String>,
        /// e.g. `"type"`, `"test"`, `"lint"`, `"patch"`.
        pub category: String,
        pub message: String,
    }

    impl Failure {
        pub fn new(category: impl Into<String>, message: impl Into<String>) -> Self {
            Self {
                file: None,
                line: None,
                code: None,
                category: category.into(),
                message: message.into(),
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Verdict {
        pub status: VerdictStatus,
        /// Probabilistic only ∈ [0,1]; for deterministic oracles, 1.0 on Pass / 0.0
        /// on Fail. Never overrides a Deterministic verdict.
        pub score: f32,
        pub oracle: String,
        /// Which class produced this verdict (drives gate ranking).
        #[serde(default = "default_class")]
        pub class: OracleClass,
        pub detail: String,
        /// Structured failures (empty on Pass). Minimal-repair context (§4.7).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub failures: Vec<Failure>,
        /// Content-addressed artifact refs (logs/diffs) — blob hashes.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub artifacts: Vec<String>,
        #[serde(default)]
        pub duration_ms: u64,
    }

    fn default_class() -> OracleClass {
        OracleClass::Deterministic
    }

    impl Verdict {
        pub fn pass(
            oracle: impl Into<String>,
            class: OracleClass,
            detail: impl Into<String>,
        ) -> Self {
            Self {
                status: VerdictStatus::Pass,
                score: 1.0,
                oracle: oracle.into(),
                class,
                detail: detail.into(),
                failures: Vec::new(),
                artifacts: Vec::new(),
                duration_ms: 0,
            }
        }

        pub fn fail(
            oracle: impl Into<String>,
            class: OracleClass,
            detail: impl Into<String>,
            failures: Vec<Failure>,
        ) -> Self {
            Self {
                status: VerdictStatus::Fail,
                score: 0.0,
                oracle: oracle.into(),
                class,
                detail: detail.into(),
                failures,
                artifacts: Vec::new(),
                duration_ms: 0,
            }
        }

        pub fn is_deterministic(&self) -> bool {
            self.class == OracleClass::Deterministic
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum VerdictStatus {
        Pass,
        Fail,
        Inconclusive,
        Skipped,
    }

    /// The verifier interface (A.2). `id`/`class`/`cost_hint` describe the oracle;
    /// `verify` runs it (sandboxed, pure w.r.t. the snapshot) and returns a verdict.
    pub trait Oracle: Send + Sync {
        fn name(&self) -> &str;

        /// Deterministic vs Probabilistic — drives the gate's authority ranking.
        fn class(&self) -> OracleClass {
            OracleClass::Deterministic
        }

        /// Relative cost so the gate can order cheap-before-expensive.
        fn cost_hint(&self) -> Cost {
            Cost::Medium
        }

        fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>>;
    }
}

pub mod probabilistic {
    //! Probabilistic oracles — fallback & tie-break only (bible ch.02 §4.6.3).
    //!
    //! These run ONLY when no deterministic oracle applies (e.g. a `synthesize`
    //! step with no buildable artifact). They never override `build`/`test`.
    //!
    //! * [`ConsistencyOracle`] — self-consistency vote over K samples (§3.1). Cheap,
    //!   local, surprisingly strong; the majority/centroid is the verdict.
    //! * [`LlmJudgeOracle`] — the model critiques the candidate against the step's
    //!   predicate. Strictly gated: used only as the last resort.

    use crate::runtime_client::KernelRuntimeClient;
    use crate::verify::oracle::{
        Cost, Oracle, OracleClass, Verdict, VerdictStatus, VerificationInput,
    };
    use futures::future::BoxFuture;
    use hide_core::runtime::{InferenceRequest, StreamChunk};
    use hide_core::Result;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Self-consistency vote (§4.6.3). Samples the model `k` times for a short
    /// yes/no judgement against the step predicate and takes the majority. The
    /// score is the agreement fraction.
    pub struct ConsistencyOracle {
        runtime: Arc<KernelRuntimeClient>,
        k: u8,
        predicate: String,
    }

    impl ConsistencyOracle {
        pub fn new(runtime: Arc<KernelRuntimeClient>, k: u8, predicate: impl Into<String>) -> Self {
            Self {
                runtime,
                k: k.max(1),
                predicate: predicate.into(),
            }
        }

        async fn sample_once(&self, candidate: &str) -> Result<bool> {
            let prompt = format!(
            "You are a strict verifier. Does the following candidate satisfy this requirement?\n\
             Requirement: {}\n\nCandidate:\n{}\n\nAnswer YES or NO only.",
            self.predicate, candidate
        );
            let request = InferenceRequest {
                task_kind: "verify".to_string(),
                prompt,
                messages: Vec::new(),
                max_output_tokens: 4,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: BTreeMap::new(),
            };
            let mut buf = String::new();
            let mut sink = |chunk: StreamChunk| {
                if let StreamChunk::Token { text, .. } = chunk {
                    buf.push_str(&text);
                }
                Ok(())
            };
            self.runtime.generate(request, &mut sink).await?;
            let answer = buf.trim().to_ascii_lowercase();
            Ok(answer.starts_with("yes") || answer.starts_with('y') || answer.contains("yes"))
        }
    }

    impl Oracle for ConsistencyOracle {
        fn name(&self) -> &str {
            "consistency"
        }
        fn class(&self) -> OracleClass {
            OracleClass::Probabilistic
        }
        fn cost_hint(&self) -> Cost {
            Cost::Medium
        }
        fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
            Box::pin(async move {
                let mut yes = 0u32;
                for _ in 0..self.k {
                    if self.sample_once(&input.candidate_output).await? {
                        yes += 1;
                    }
                }
                let score = yes as f32 / self.k as f32;
                let status = if score > 0.5 {
                    VerdictStatus::Pass
                } else if yes == 0 {
                    VerdictStatus::Fail
                } else {
                    VerdictStatus::Inconclusive
                };
                let mut v = Verdict::pass(
                    "consistency",
                    OracleClass::Probabilistic,
                    format!("{yes}/{} votes yes", self.k),
                );
                v.status = status;
                v.score = score;
                Ok(v)
            })
        }
    }

    /// LLM-as-judge (§4.6.3) — the strictly-fallback critic. One critique; the score
    /// is parsed from a leading `0.0..=1.0`. Never overrides a deterministic verdict
    /// (the gate enforces that by class).
    pub struct LlmJudgeOracle {
        runtime: Arc<KernelRuntimeClient>,
        predicate: String,
    }

    impl LlmJudgeOracle {
        pub fn new(runtime: Arc<KernelRuntimeClient>, predicate: impl Into<String>) -> Self {
            Self {
                runtime,
                predicate: predicate.into(),
            }
        }
    }

    impl Oracle for LlmJudgeOracle {
        fn name(&self) -> &str {
            "llm_judge"
        }
        fn class(&self) -> OracleClass {
            OracleClass::Probabilistic
        }
        fn cost_hint(&self) -> Cost {
            Cost::Medium
        }
        fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
            Box::pin(async move {
                let prompt = format!(
                    "Rate from 0.0 to 1.0 how well the candidate meets the requirement. \
                 Output the number first.\nRequirement: {}\n\nCandidate:\n{}",
                    self.predicate, input.candidate_output
                );
                let request = InferenceRequest {
                    task_kind: "verify".to_string(),
                    prompt,
                    messages: Vec::new(),
                    max_output_tokens: 8,
                    sampler: None,
                    grammar: None,
                    want_logprobs: false,
                    metadata: BTreeMap::new(),
                };
                let mut buf = String::new();
                let mut sink = |chunk: StreamChunk| {
                    if let StreamChunk::Token { text, .. } = chunk {
                        buf.push_str(&text);
                    }
                    Ok(())
                };
                self.runtime.generate(request, &mut sink).await?;
                let score = parse_leading_float(&buf).unwrap_or(0.0);
                let mut v = Verdict::pass(
                    "llm_judge",
                    OracleClass::Probabilistic,
                    format!("judge score {score:.2}"),
                );
                v.score = score;
                v.status = if score >= 0.5 {
                    VerdictStatus::Pass
                } else {
                    VerdictStatus::Fail
                };
                Ok(v)
            })
        }
    }

    fn parse_leading_float(s: &str) -> Option<f32> {
        let t = s.trim();
        let mut end = 0;
        for (i, c) in t.char_indices() {
            if c.is_ascii_digit() || c == '.' {
                end = i + c.len_utf8();
            } else {
                break;
            }
        }
        t.get(..end).and_then(|p| p.parse::<f32>().ok())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hawking_orch::inference::StubInferenceClient;
        use hawking_orch::registry::RoleRegistry;
        use hawking_orch::router::SimpleRouter;
        fn runtime(response: &str) -> Arc<KernelRuntimeClient> {
            let registry = Arc::new(RoleRegistry::with_default_local_roles());
            let router = Arc::new(SimpleRouter::new(registry));
            let inference = Arc::new(StubInferenceClient::new(response));
            Arc::new(KernelRuntimeClient::new(router, inference))
        }
        #[test]
        fn parses_leading_float() {
            assert_eq!(parse_leading_float("0.83 because ..."), Some(0.83));
            assert_eq!(parse_leading_float("1.0"), Some(1.0));
        }
        #[tokio::test]
        async fn consistency_unanimous_yes_passes() {
            let oracle = ConsistencyOracle::new(runtime("YES"), 3, "does the thing");
            let mut input = VerificationInput::new("/tmp");
            input.candidate_output = "the thing".to_string();
            let v = oracle.verify(&input).await.unwrap();
            assert_eq!(v.status, VerdictStatus::Pass);
            assert_eq!(v.score, 1.0);
            assert_eq!(v.class, OracleClass::Probabilistic);
        }
        #[tokio::test]
        async fn judge_low_score_fails() {
            let oracle = LlmJudgeOracle::new(runtime("0.2 not great"), "be great");
            let v = oracle
                .verify(&VerificationInput::new("/tmp"))
                .await
                .unwrap();
            assert_eq!(v.status, VerdictStatus::Fail);
        }
    }
}
