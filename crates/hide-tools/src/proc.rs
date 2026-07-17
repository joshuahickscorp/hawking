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
        Self {
            spec,
            default_argv,
            config: ShellConfig::default(),
        }
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
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect::<Vec<_>>()
                })
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| self.default_argv.clone());
            let cwd = args.get("cwd").and_then(|v| v.as_str()).map(String::from);
            let timeout = ctx
                .deadline_ms
                .filter(|ms| *ms > 0)
                .unwrap_or(self.spec.timeout_ms);
            let env = BTreeMap::new();
            run_command(
                &argv,
                cwd.as_deref(),
                &env,
                timeout,
                ctx.output_cap_bytes as usize,
                &self.config,
            )
            .await
        })
    }

    fn simulate<'a>(&'a self, args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
        Box::pin(async move {
            let argv = args
                .get("argv")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
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
        ToolCtx {
            grant_id: None,
            deadline_ms: Some(30_000),
            output_cap_bytes: 1 << 20,
        }
    }

    #[tokio::test]
    async fn proc_failing_command_is_ok_data() {
        // override argv with a command that exits non-zero
        let tool = ProcTool::test_run().with_config(ShellConfig {
            disable_sandbox: true,
            ..Default::default()
        });
        let r = tool
            .call(
                json!({ "argv": ["sh", "-c", "echo fail 1>&2; exit 1"] }),
                ctx(),
            )
            .await;
        assert!(r.ok, "failing test run must be ok:true (EXEC_NONZERO)");
        assert_eq!(r.exit_code, Some(1));
        assert_eq!(pass_fail(&r), Some(false));
    }

    #[tokio::test]
    async fn proc_passing_command() {
        let tool = ProcTool::compile_check().with_config(ShellConfig {
            disable_sandbox: true,
            ..Default::default()
        });
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
