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
async fn run_git(
    cwd: &str,
    args: &[String],
    cap_bytes: usize,
    blobs: Option<&Arc<dyn BlobStore>>,
) -> ToolResult {
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
    args.get("cwd")
        .and_then(|v| v.as_str())
        .unwrap_or(".")
        .to_string()
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
                Self {
                    spec: git_read_spec($name, $title, $desc, $schema),
                    config: GitConfig::default(),
                }
            }
        }
        impl $ty {
            pub fn with_config(config: GitConfig) -> Self {
                Self {
                    config,
                    ..Self::default()
                }
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
                    run_git(&cwd, &argv, ctx.output_cap_bytes as usize, self.config.blobs.as_ref())
                        .await
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
    |_args: &Value| vec![
        "status".to_string(),
        "--short".to_string(),
        "--branch".to_string()
    ]
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
        let mut v = vec![
            "log".to_string(),
            "--oneline".to_string(),
            "--decorate".to_string(),
        ];
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
    |_args: &Value| vec![
        "worktree".to_string(),
        "list".to_string(),
        "--porcelain".to_string()
    ]
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
        Self {
            config,
            ..Self::default()
        }
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
        Self {
            config,
            ..Self::default()
        }
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
            let mut result =
                run_git(&cwd, &argv, ctx.output_cap_bytes as usize, self.config.blobs.as_ref()).await;
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
        Self {
            config,
            ..Self::default()
        }
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
        format!(
            "{}_{}_{}",
            std::process::id(),
            hide_core::ids::now_ms(),
            N.fetch_add(1, Ordering::SeqCst)
        )
    }

    async fn init_repo() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hide_git_{}", unique()));
        std::fs::create_dir_all(&dir).unwrap();
        for argv in [
            vec!["init", "-q"],
            vec!["config", "user.email", "t@t.t"],
            vec!["config", "user.name", "t"],
        ] {
            Command::new("git")
                .args(&argv)
                .current_dir(&dir)
                .output()
                .await
                .unwrap();
        }
        dir
    }

    #[tokio::test]
    async fn git_status_clean_is_ok() {
        let dir = init_repo().await;
        let reg = Arc::new(ToolRegistry::default());
        reg.register(GitStatusTool::default());
        let d = dispatcher(reg);
        let r = d
            .dispatch(ToolCall::new(
                "git.status",
                json!({ "cwd": dir.to_string_lossy() }),
            ))
            .await
            .unwrap();
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
        let r = d
            .dispatch(ToolCall::new(
                "git.diff",
                json!({ "cwd": dir.to_string_lossy() }),
            ))
            .await
            .unwrap();
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
            .dispatch(ToolCall::new(
                "git.commit",
                json!({ "cwd": dir.to_string_lossy(), "message": "init" }),
            ))
            .await
            .unwrap();
        assert!(commit.ok);
        assert_eq!(commit.exit_code, Some(0));
        let log = d
            .dispatch(ToolCall::new(
                "git.log",
                json!({ "cwd": dir.to_string_lossy() }),
            ))
            .await
            .unwrap();
        assert!(log.structured_content.unwrap()["stdout"]
            .as_str()
            .unwrap()
            .contains("init"));
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
        assert!(
            !evil.exists(),
            "git.diff ref must not inject --output and write a file"
        );
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
            .dispatch(ToolCall::new(
                "git.commit",
                json!({ "cwd": dir.to_string_lossy(), "message": "c" }),
            ))
            .await
            .unwrap();
        let r = d
            .dispatch(ToolCall::new(
                "git.diff",
                json!({ "cwd": dir.to_string_lossy(), "ref": "HEAD" }),
            ))
            .await
            .unwrap();
        assert!(r.ok, "a normal ref diff must still work: {:?}", r.error);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn git_worktree_add_list_remove() {
        let dir = init_repo().await;
        std::fs::write(dir.join("a.txt"), "hello").unwrap();
        Command::new("git")
            .args(["add", "-A"])
            .current_dir(&dir)
            .output()
            .await
            .unwrap();
        Command::new("git")
            .args(["commit", "-qm", "init"])
            .current_dir(&dir)
            .output()
            .await
            .unwrap();
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
        let list = d
            .dispatch(ToolCall::new(
                "git.worktree.list",
                json!({ "cwd": dir.to_string_lossy() }),
            ))
            .await
            .unwrap();
        assert!(list.structured_content.unwrap()["stdout"]
            .as_str()
            .unwrap()
            .contains("feat"));
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
