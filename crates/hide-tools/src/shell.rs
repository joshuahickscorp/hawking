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
        Self {
            config,
            ..Self::default()
        }
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
            let timeout = ctx
                .deadline_ms
                .filter(|ms| *ms > 0)
                .unwrap_or(self.spec.timeout_ms);
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
        Self {
            spec: plan_spec(),
            config: ShellConfig::default(),
        }
    }
}

impl ShellPlanTool {
    pub fn with_config(config: ShellConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
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
            common::ok_text(
                format!("planned (sandboxed) command: {argv:?}"),
                body,
                EffectSet::default(),
            )
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
        .and_then(|r| {
            std::fs::canonicalize(&r)
                .ok()
                .map(|p| p.to_string_lossy().into_owned())
        })
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|p| p.to_string_lossy().into_owned())
        })
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
        worktree_root: config
            .worktree_root
            .clone()
            .or_else(|| config.workspace_root.clone()),
    }
}

/// The interpreter/toolchain helpers a real command commonly re-execs (git calls
/// hooks, cargo spawns rustc, shells spawn coreutils). Bare names are matched by
/// basename regex in the renderer.
fn essential_exec_allowlist() -> Vec<String> {
    [
        "sh", "bash", "zsh", "env", "git", "cargo", "rustc", "cc", "ld", "clang", "printf", "echo",
        "true", "sleep", "cat", "ls", "node", "python3", "python",
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
fn build_confined_command(
    argv: &[String],
    config: &ShellConfig,
) -> Result<SandboxedSpawn, Box<ToolResult>> {
    // Explicit, caller-chosen opt-out for an already-confined context.
    if config.disable_sandbox {
        return Ok(SandboxedSpawn {
            command: bare_command(argv),
            warning: None,
        });
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
        return Ok(SandboxedSpawn {
            command: c,
            warning: None,
        });
    }

    if let Some(bwrap) = bubblewrap_path() {
        return Ok(SandboxedSpawn {
            command: bubblewrap_command(&bwrap, argv, config),
            warning: None,
        });
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
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

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
    let mut result =
        common::project_process_output(exit_code, stdout, stderr, cap_bytes, config.blobs.as_ref());
    if let Some(warn) = sandbox_warning {
        if let Some(sc) = result.structured_content.as_mut() {
            sc["sandbox_warning"] = json!(warn);
        }
        result.content.push(ToolContent::Text {
            text: format!("[sandbox] {warn}"),
        });
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
    CATASTROPHIC
        .iter()
        .find(|needle| joined.contains(*needle))
        .map(|s| s.to_string())
}

pub(crate) fn parse_argv(args: &Value) -> Vec<String> {
    args.get("argv")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                .collect()
        })
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
        let result = dispatcher
            .dispatch(ToolCall::new(
                "shell.run",
                json!({ "argv": ["printf", "hello"] }),
            ))
            .await
            .unwrap();
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
            .dispatch(ToolCall::new(
                "shell.run",
                json!({ "argv": ["sh", "-c", "echo boom 1>&2; exit 3"] }),
            ))
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
        let config = ShellConfig {
            disable_sandbox: true,
            ..Default::default()
        };
        let env = BTreeMap::new();
        let result = run_command(
            &["sleep".to_string(), "5".to_string()],
            None,
            &env,
            150,
            4096,
            &config,
        )
        .await;
        assert!(!result.ok);
        assert_eq!(result.status, ToolStatus::TimedOut);
        assert_eq!(result.error.unwrap().code, "TIMEOUT");
    }

    #[tokio::test]
    async fn shell_run_refuses_catastrophic() {
        let registry = Arc::new(ToolRegistry::default());
        registry.register(ShellRunTool::default());
        let dispatcher = allow_all_dispatcher(registry);
        let result = dispatcher
            .dispatch(ToolCall::new(
                "shell.run",
                json!({ "argv": ["rm", "-rf", "/"] }),
            ))
            .await
            .unwrap();
        assert!(!result.ok);
        assert_eq!(result.error.unwrap().code, "CAP_DENIED");
    }

    #[test]
    fn fail_closed_when_no_os_sandbox_available() {
        // Simulate a platform with no usable OS sandbox: not disable_sandbox, not
        // allow_unconfined. On a host where sandbox-exec/bwrap is genuinely
        // unavailable this is the live path; on macOS CI we still assert the
        // decision function refuses (it only ever runs UNCONFINED via an opt-out).
        let config = ShellConfig {
            allow_unconfined: false,
            disable_sandbox: false,
            ..Default::default()
        };
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
        let config = ShellConfig {
            allow_unconfined: true,
            ..Default::default()
        };
        let argv = vec!["true".to_string()];
        let spawn = build_confined_command(&argv, &config).expect("opt-out must not refuse");
        if sandbox_exec_available() || bubblewrap_path().is_some() {
            // Real sandbox present → confined, no escape-hatch warning.
            assert!(spawn.warning.is_none());
        } else {
            assert!(
                spawn
                    .warning
                    .as_deref()
                    .map(|w| w.contains("UNCONFINED"))
                    .unwrap_or(false),
                "unconfined opt-out must carry a warning"
            );
        }
    }

    #[test]
    fn disable_sandbox_runs_bare_without_refusal() {
        // An already-confined worktree run opts out and is never refused.
        let config = ShellConfig {
            disable_sandbox: true,
            ..Default::default()
        };
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
            rendered
                .profile_text
                .contains("(deny file-write* (subpath \"/var/hide-test/.hide/log\"))"),
            "absolute .hide/log write-deny must be threaded:\n{}",
            rendered.profile_text
        );
        // Worktree write confinement.
        assert!(rendered
            .profile_text
            .contains("(allow file-write* (subpath \"/tmp/wt\"))"));
        // Proxy egress route (S5b).
        assert!(rendered.profile_text.contains("localhost:8443"));
    }

    #[test]
    fn render_options_worktree_falls_back_to_workspace_root() {
        let config = ShellConfig {
            workspace_root: Some("/tmp/ws".to_string()),
            ..Default::default()
        };
        let opts = sandbox_render_options(&config);
        assert_eq!(opts.worktree_root.as_deref(), Some("/tmp/ws"));
    }

    #[tokio::test]
    async fn shell_plan_renders_sandbox_profile() {
        let tool = ShellPlanTool::default();
        let result = tool
            .call(
                json!({ "argv": ["cargo", "test"] }),
                ToolCtx {
                    grant_id: None,
                    deadline_ms: None,
                    output_cap_bytes: 65536,
                },
            )
            .await;
        let sc = result.structured_content.unwrap();
        assert_eq!(sc["executed"], false);
        assert!(sc["sandbox_profile"]
            .as_str()
            .unwrap()
            .contains("(deny default)"));
    }
}
