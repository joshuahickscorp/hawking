//! macOS Seatbelt profile rendering + `sandbox-exec` spawning (bible ch.10
//! §4.5.2, S2/S5/S5b/S12).
//!
//! Extends the original `render_macos_seatbelt` (whose signature siblings —
//! `hide-tools` — call) to a profile that honors the §4.5.2 skeleton:
//!   * deny-by-default;
//!   * **process-exec allowlist** — only the granted binaries may `exec`;
//!   * **filesystem**: read broad-but-bounded, write narrow, with secret paths
//!     (`.ssh`/`.aws`/`.env`/`*.pem`) read-denied and **`.hide/log`
//!     write-denied** (S4 — the audit log is invisible/untouchable to the
//!     sandbox);
//!   * **network**: the only egress route is the host proxy port (S5b).
//!
//! Plus a per-grant `.sb` emitter and a `sandbox-exec` spawn helper that fails
//! CLOSED (S12): if `sandbox-exec` is unavailable, the spawn errors rather than
//! running the command unconfined.

use hide_core::ids::GrantId;
use hide_core::security::{NetworkPolicy, SandboxProfile, SandboxTier};
use hide_core::types::Decision;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default secret paths denied at the *read* layer (§4.5.2): removes the
/// "private data" leg of the lethal trifecta at the OS for every sandboxed run.
const SECRET_READ_DENY_SUBPATHS: &[&str] = &["$HOME/.ssh", "$HOME/.aws", "$HOME/.config/gh"];
const SECRET_READ_DENY_REGEXES: &[&str] = &[r"/\.env($|\.)", r"\.pem$", r"\.key$"];

/// Broad-but-bounded system read roots a build/test realistically needs.
const SYSTEM_READ_SUBPATHS: &[&str] = &["/usr", "/bin", "/System/Library", "/Library/Developer"];
const SYSTEM_READ_LITERALS: &[&str] = &[
    "/dev/null",
    "/dev/urandom",
    "/dev/random",
    "/dev/dtracehelper",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedSandboxProfile {
    pub tier: SandboxTier,
    pub profile_text: String,
    pub warnings: Vec<String>,
}

/// Optional render-time context the basic `render_macos_seatbelt` doesn't carry
/// on the profile itself (proxy port, workspace/worktree roots, `.hide` dir).
/// Defaults are conservative (no egress route, no extra confinement seam).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SandboxRenderOptions {
    /// Host egress proxy port; `Some` ⇒ render the single allowed outbound
    /// route; `None` ⇒ no network route at all even if policy is `Allow`-ish.
    pub proxy_port: Option<u16>,
    /// The `.hide` directory whose `log` subdir must be write-denied (S4). If
    /// `None`, a relative `.hide/log` deny is still emitted.
    pub hide_dir: Option<PathBuf>,
    /// Worktree root to confine writes to (§4.5.2 `$WORKTREE`); falls back to
    /// the profile's first write root.
    pub worktree_root: Option<String>,
}

/// Render a Seatbelt (SBPL) profile from `profile`. **Public signature
/// preserved** — `hide-tools` calls this exact shape. The render now includes
/// the process-exec allowlist, secret read-denies, `.hide/log` write-deny, and
/// (best-effort, no port) proxy-only network framing. For the proxy port and
/// explicit worktree/`.hide` confinement, use [`render_macos_seatbelt_with`].
pub fn render_macos_seatbelt(profile: &SandboxProfile) -> RenderedSandboxProfile {
    render_macos_seatbelt_with(profile, &SandboxRenderOptions::default())
}

/// Full render with render-time options (proxy port, `.hide`, worktree).
pub fn render_macos_seatbelt_with(
    profile: &SandboxProfile,
    opts: &SandboxRenderOptions,
) -> RenderedSandboxProfile {
    let mut text = String::from("(version 1)\n(deny default)\n\n");
    let mut warnings = Vec::new();

    // --- process-exec allowlist (§4.5.2) ---
    text.push_str(";; --- process ---\n");
    text.push_str("(allow process-fork)\n");
    if profile.allowed_commands.is_empty() {
        // No allowlist ⇒ nothing may exec. Fail-safe: we deny rather than
        // silently allowing all exec.
        text.push_str("(deny process-exec*)\n");
        warnings.push(
            "no allowed_commands: process-exec fully denied (grant the exact binaries to run)"
                .to_string(),
        );
    } else {
        let mut literals = String::new();
        for cmd in &profile.allowed_commands {
            // Match the binary path the grant authorized. If it's a bare name,
            // emit both literal and a basename regex so common PATH locations
            // resolve; if it's an absolute path, pin it exactly.
            if cmd.starts_with('/') {
                literals.push_str(&format!("    (literal \"{}\")\n", escape(cmd)));
            } else {
                literals.push_str(&format!("    (regex #\"/{}$\")\n", escape_regex(cmd)));
            }
        }
        text.push_str("(allow process-exec\n");
        text.push_str(&literals);
        text.push_str(")\n");
        text.push_str("(deny process-exec*)\n");
    }
    text.push('\n');

    // --- filesystem: read broad-but-bounded, write narrow ---
    text.push_str(";; --- filesystem ---\n");
    for sub in SYSTEM_READ_SUBPATHS {
        text.push_str(&format!(
            "(allow file-read* (subpath \"{}\"))\n",
            escape(sub)
        ));
    }
    for lit in SYSTEM_READ_LITERALS {
        text.push_str(&format!(
            "(allow file-read* (literal \"{}\"))\n",
            escape(lit)
        ));
    }
    for root in &profile.read_roots {
        text.push_str(&format!(
            "(allow file-read* (subpath \"{}\"))\n",
            escape(root)
        ));
    }

    // Secret read-denies (S2/S6) — these come AFTER allows; in SBPL the most
    // specific / last-matching rule wins, and explicit deny always overrides.
    for sub in SECRET_READ_DENY_SUBPATHS {
        text.push_str(&format!(
            "(deny file-read* (subpath \"{}\"))\n",
            escape(sub)
        ));
    }
    for re in SECRET_READ_DENY_REGEXES {
        text.push_str(&format!("(deny file-read* (regex #\"{}\"))\n", re));
    }

    // Writes confined to worktree/write-roots only.
    let write_root = opts
        .worktree_root
        .clone()
        .or_else(|| profile.write_roots.first().cloned());
    if let Some(root) = &write_root {
        text.push_str(&format!(
            "(allow file-write* (subpath \"{}\"))\n",
            escape(root)
        ));
    }
    for root in &profile.write_roots {
        if Some(root) != write_root.as_ref() {
            text.push_str(&format!(
                "(allow file-write* (subpath \"{}\"))\n",
                escape(root)
            ));
        }
    }
    if write_root.is_none() && profile.write_roots.is_empty() {
        warnings.push("no write roots: sandbox is read-only".to_string());
    }

    // The audit log is invisible AND untouchable to the sandbox (S4).
    let log_path = opts
        .hide_dir
        .as_ref()
        .map(|d| d.join("log").to_string_lossy().into_owned())
        .unwrap_or_else(|| ".hide/log".to_string());
    text.push_str(&format!(
        "(deny file-read*  (subpath \"{}\"))\n",
        escape(&log_path)
    ));
    text.push_str(&format!(
        "(deny file-write* (subpath \"{}\"))\n",
        escape(&log_path)
    ));
    // The whole .hide dir is never writable by the agent (§4.5.2).
    if let Some(hide) = &opts.hide_dir {
        text.push_str(&format!(
            "(deny file-write* (subpath \"{}\"))\n",
            escape(&hide.to_string_lossy())
        ));
    }
    text.push('\n');

    // --- network: the ONLY socket is the host proxy port (S5b) ---
    text.push_str(";; --- network ---\n");
    text.push_str("(deny network*)\n");
    match (profile.network.default, opts.proxy_port) {
        (Decision::Allow, _) => {
            // Even an Allow policy funnels through the proxy if we have one;
            // a blanket allow without a proxy is a warned escape hatch.
            if let Some(port) = opts.proxy_port {
                text.push_str(&format!(
                    "(allow network-outbound (remote ip \"localhost:{port}\"))\n"
                ));
            } else {
                text.push_str("(allow network*)\n");
                warnings.push(
                    "network default=allow with no proxy port: unmediated egress (escape hatch)"
                        .to_string(),
                );
            }
        }
        (Decision::Deny, Some(port)) | (Decision::Ask, Some(port)) => {
            text.push_str(&format!(
                "(allow network-outbound (remote ip \"localhost:{port}\"))\n"
            ));
            if profile.network.default == Decision::Ask {
                warnings.push(
                    "network=ask is enforced as proxy-only egress; per-host allow is the proxy's job"
                        .to_string(),
                );
            }
        }
        (Decision::Deny, None) => {
            warnings.push("network default deny, no proxy port: zero egress route".to_string());
        }
        (Decision::Ask, None) => {
            warnings.push(
                "network=ask but no proxy port supplied; rendering zero egress (fail-safe)"
                    .to_string(),
            );
        }
    }

    if !profile.network.allowed_hosts.is_empty() {
        warnings.push(format!(
            "{} allowed_hosts are enforced at the proxy, not in SBPL",
            profile.network.allowed_hosts.len()
        ));
    }

    RenderedSandboxProfile {
        tier: profile.tier,
        profile_text: text,
        warnings,
    }
}

/// Default workspace profile (read = workspace root; no exec, no write, no net).
pub fn default_workspace_profile(root: impl Into<String>) -> SandboxProfile {
    SandboxProfile {
        tier: SandboxTier::Seatbelt,
        read_roots: vec![root.into()],
        write_roots: Vec::new(),
        allowed_commands: Vec::new(),
        network: NetworkPolicy::default(),
    }
}

/// Write the compiled per-grant profile to `sandbox/profiles/<grant_id>.sb`
/// (§4.1) and return its path. The host owns this dir; it is ephemeral.
pub fn emit_grant_profile(
    sandbox_dir: &Path,
    grant_id: &GrantId,
    rendered: &RenderedSandboxProfile,
) -> hide_core::Result<PathBuf> {
    let profiles = sandbox_dir.join("profiles");
    std::fs::create_dir_all(&profiles)?;
    let path = profiles.join(format!("{}.sb", grant_id.as_str()));
    std::fs::write(&path, rendered.profile_text.as_bytes())?;
    Ok(path)
}

/// A command to run under `sandbox-exec`.
#[derive(Debug, Clone)]
pub struct SandboxedCommand {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
}

impl SandboxedCommand {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            cwd: None,
        }
    }

    pub fn arg(mut self, a: impl Into<String>) -> Self {
        self.args.push(a.into());
        self
    }

    pub fn args<I, S>(mut self, it: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(it.into_iter().map(Into::into));
        self
    }

    pub fn cwd(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cwd = Some(dir.into());
        self
    }
}

/// Is `sandbox-exec` available on this host? (§4.5.3 fail-safe pre-check.)
pub fn sandbox_exec_available() -> bool {
    cfg!(target_os = "macos") && Path::new("/usr/bin/sandbox-exec").exists()
}

/// Build the `sandbox-exec -f <profile.sb> <program> <args...>` command WITHOUT
/// spawning it — so callers (and tests) can inspect/own the spawn. Fails CLOSED
/// (S12): returns an error if `sandbox-exec` is unavailable rather than handing
/// back an unconfined command.
pub fn build_sandbox_exec_command(
    profile_path: &Path,
    cmd: &SandboxedCommand,
) -> hide_core::Result<Command> {
    if !sandbox_exec_available() {
        return Err(hide_core::error::HideError::PolicyDenied(
            "sandbox-exec unavailable: refusing to run unconfined (S12). Escalate to a microVM tier or an explicit logged override.".to_string(),
        ));
    }
    if !profile_path.exists() {
        return Err(hide_core::error::HideError::Storage(format!(
            "sandbox profile {} not found",
            profile_path.display()
        )));
    }
    let mut c = Command::new("/usr/bin/sandbox-exec");
    c.arg("-f").arg(profile_path);
    c.arg(&cmd.program);
    c.args(&cmd.args);
    if let Some(dir) = &cmd.cwd {
        c.current_dir(dir);
    }
    Ok(c)
}

/// Render → emit the per-grant `.sb` → build the confined `sandbox-exec`
/// command, in one step. The host's spawn path for a T2 grant.
pub fn spawn_under_sandbox(
    sandbox_dir: &Path,
    grant_id: &GrantId,
    profile: &SandboxProfile,
    opts: &SandboxRenderOptions,
    cmd: &SandboxedCommand,
) -> hide_core::Result<Command> {
    let rendered = render_macos_seatbelt_with(profile, opts);
    let profile_path = emit_grant_profile(sandbox_dir, grant_id, &rendered)?;
    build_sandbox_exec_command(&profile_path, cmd)
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Minimal regex metachar escape for a binary basename match.
fn escape_regex(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if matches!(
            ch,
            '.' | '+'
                | '*'
                | '?'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '^'
                | '$'
                | '|'
                | '\\'
                | '/'
        ) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile_with(cmds: &[&str], net: Decision) -> SandboxProfile {
        SandboxProfile {
            tier: SandboxTier::Seatbelt,
            read_roots: vec!["/work".to_string()],
            write_roots: vec!["/work/wt".to_string()],
            allowed_commands: cmds.iter().map(|s| s.to_string()).collect(),
            network: NetworkPolicy {
                default: net,
                allowed_hosts: vec!["api.github.com".to_string()],
                denied_hosts: vec![],
            },
        }
    }

    #[test]
    fn renders_deny_default_and_exec_allowlist() {
        let p = profile_with(&["/usr/bin/cargo"], Decision::Deny);
        let r = render_macos_seatbelt(&p);
        assert!(r.profile_text.contains("(deny default)"));
        assert!(r.profile_text.contains("(allow process-exec"));
        assert!(r.profile_text.contains("(literal \"/usr/bin/cargo\")"));
        assert!(r.profile_text.contains("(deny process-exec*)"));
    }

    #[test]
    fn empty_allowlist_denies_all_exec() {
        let p = profile_with(&[], Decision::Deny);
        let r = render_macos_seatbelt(&p);
        assert!(r.profile_text.contains("(deny process-exec*)"));
        assert!(!r.profile_text.contains("(allow process-exec\n"));
        assert!(r
            .warnings
            .iter()
            .any(|w| w.contains("process-exec fully denied")));
    }

    #[test]
    fn denies_secret_reads_and_hide_log_writes() {
        let p = profile_with(&["/bin/sh"], Decision::Deny);
        let opts = SandboxRenderOptions {
            proxy_port: None,
            hide_dir: Some(PathBuf::from("/work/.hide")),
            worktree_root: Some("/work/wt".to_string()),
        };
        let r = render_macos_seatbelt_with(&p, &opts);
        assert!(r
            .profile_text
            .contains("(deny file-read* (subpath \"$HOME/.ssh\"))"));
        assert!(r
            .profile_text
            .contains(r#"(deny file-read* (regex #"\.pem$"))"#));
        // .hide/log specifically write-denied.
        assert!(r
            .profile_text
            .contains("(deny file-write* (subpath \"/work/.hide/log\"))"));
        // whole .hide write-denied.
        assert!(r
            .profile_text
            .contains("(deny file-write* (subpath \"/work/.hide\"))"));
    }

    #[test]
    fn proxy_port_is_the_only_egress() {
        let p = profile_with(&["/bin/sh"], Decision::Deny);
        let opts = SandboxRenderOptions {
            proxy_port: Some(8131),
            ..Default::default()
        };
        let r = render_macos_seatbelt_with(&p, &opts);
        assert!(r.profile_text.contains("(deny network*)"));
        assert!(r
            .profile_text
            .contains("(allow network-outbound (remote ip \"localhost:8131\"))"));
        // allowed_hosts are a proxy concern, surfaced as a warning.
        assert!(r
            .warnings
            .iter()
            .any(|w| w.contains("allowed_hosts are enforced at the proxy")));
    }

    #[test]
    fn deny_network_without_proxy_warns_zero_egress() {
        let p = profile_with(&["/bin/sh"], Decision::Deny);
        let r = render_macos_seatbelt(&p);
        assert!(r.warnings.iter().any(|w| w.contains("zero egress")));
        assert!(!r.profile_text.contains("network-outbound"));
    }

    #[test]
    fn bare_command_renders_basename_regex() {
        let p = profile_with(&["cargo"], Decision::Deny);
        let r = render_macos_seatbelt(&p);
        assert!(
            r.profile_text.contains(r#"(regex #"/cargo$")"#),
            "{}",
            r.profile_text
        );
    }

    #[test]
    fn emit_grant_profile_writes_sb() {
        let dir = tempfile::tempdir().unwrap();
        let p = profile_with(&["/bin/sh"], Decision::Deny);
        let r = render_macos_seatbelt(&p);
        let gid = GrantId::new();
        let path = emit_grant_profile(dir.path(), &gid, &r).unwrap();
        assert!(path.exists());
        assert!(path.extension().unwrap() == "sb");
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("(deny default)"));
    }

    #[test]
    fn build_sandbox_exec_fails_closed_when_unavailable() {
        // On non-macOS hosts (CI Linux) sandbox-exec is unavailable → the build
        // must REFUSE, never hand back an unconfined command (S12).
        let dir = tempfile::tempdir().unwrap();
        let sb = dir.path().join("p.sb");
        std::fs::write(&sb, "(version 1)(deny default)").unwrap();
        let cmd = SandboxedCommand::new("echo").arg("hi");
        let res = build_sandbox_exec_command(&sb, &cmd);
        if sandbox_exec_available() {
            assert!(res.is_ok());
        } else {
            assert!(res.is_err(), "must fail closed without sandbox-exec");
        }
    }

    #[test]
    fn escape_handles_quotes_and_backslashes() {
        assert_eq!(escape(r#"a"b\c"#), r#"a\"b\\c"#);
    }
}
