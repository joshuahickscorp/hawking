//! The isolation model (bible ch.09 §4.3).
//!
//! Parallel work is isolated at four levels; worktrees alone are insufficient
//! (the §3.4 runtime-isolation gap). This module owns the *orchestration*
//! lifecycle — `isolate_run` creates a git worktree off the shared `.git`, leases
//! a disjoint port range, and seeds a per-run env namespace; `release_run`
//! removes/prunes the worktree and returns the ports to the pool. ch.10 owns the
//! sandbox *enforcement* (the `SandboxProfile` boundary); we reference it.
//!
//! Worktrees over full clones or containers (§4.3 rationale): worktrees share the
//! object store (cheap), give true file isolation, and leave unified RAM for the
//! model — containers would compete with the runtime for RAM, the worst trade on
//! an Apple-Silicon box.

use hide_core::security::{NetworkPolicy, SandboxProfile, SandboxTier};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::process::Command;

/// A leased git worktree for one run (§4.3 filesystem level).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorktreeLease {
    pub run_id: String,
    pub branch: String,
    pub path: PathBuf,
    pub base_ref: String,
    pub sandbox: SandboxProfile,
}

/// A disjoint port range leased to a run so dev-servers/test-DBs in different
/// runs never collide on 3000/5432/8080 (the named §3.4 gap).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortLease {
    pub run_id: String,
    pub ports: Vec<u16>,
}

/// The full workspace handed to a run: tree + ports + env namespace + sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunWorkspace {
    pub run_id: String,
    pub worktree: WorktreeLease,
    pub ports: PortLease,
    /// Per-run env (TMPDIR, build caches, DB/schema names, PORT, DATABASE_URL).
    pub env: BTreeMap<String, String>,
}

/// Outcome of a run, deciding whether its worktree is kept (merged) or discarded
/// (tournament loser / speculative discard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOutcome {
    pub discarded: bool,
}

/// Disjoint port-range allocator (§4.3 ports level). Hands each run `count`
/// contiguous-from-pool ports; releases them on run end. Tracks leases by
/// `run_id` so occupancy reads are the allocator's truth, not an estimate.
#[derive(Debug, Clone, Default)]
pub struct PortAllocator {
    start: u16,
    end: u16,
    leased: BTreeSet<u16>,
    /// run_id → the ports that run currently holds. The authority for both
    /// `leased_count()` (ports) and `leased_runs()` (distinct runs).
    run_ports: BTreeMap<String, Vec<u16>>,
}

impl PortAllocator {
    pub fn new(start: u16, end: u16) -> Self {
        Self {
            start,
            end,
            leased: BTreeSet::new(),
            run_ports: BTreeMap::new(),
        }
    }

    pub fn lease(&mut self, run_id: impl Into<String>, count: u16) -> Option<PortLease> {
        let mut ports = Vec::new();
        for port in self.start..=self.end {
            if !self.leased.contains(&port) {
                ports.push(port);
                if ports.len() == count as usize {
                    break;
                }
            }
        }
        if ports.len() != count as usize {
            return None;
        }
        for port in &ports {
            self.leased.insert(*port);
        }
        let run_id = run_id.into();
        self.run_ports
            .entry(run_id.clone())
            .or_default()
            .extend(ports.iter().copied());
        Some(PortLease { run_id, ports })
    }

    pub fn release(&mut self, lease: &PortLease) {
        for port in &lease.ports {
            self.leased.remove(port);
        }
        // Drop the released ports from the run's ledger; forget the run once it
        // holds none. Tolerant of a partial/empty lease (idempotent release).
        if let Some(held) = self.run_ports.get_mut(&lease.run_id) {
            held.retain(|p| !lease.ports.contains(p));
            if held.is_empty() {
                self.run_ports.remove(&lease.run_id);
            }
        }
    }

    /// The number of ports currently leased (the pool's true occupancy).
    pub fn leased_count(&self) -> usize {
        self.leased.len()
    }

    /// The number of distinct runs currently holding any leased port.
    pub fn leased_runs(&self) -> usize {
        self.run_ports.len()
    }
}

/// Errors from the isolation lifecycle.
#[derive(Debug, thiserror::Error)]
pub enum IsolateError {
    #[error("git worktree {op} failed (status {code:?}): {stderr}")]
    Git {
        op: &'static str,
        code: Option<i32>,
        stderr: String,
    },
    #[error("port pool exhausted: could not lease {requested} ports")]
    PortsExhausted { requested: u16 },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// The worktree manager: owns the repo root, the `.hide/wt/` root, and the port
/// pool. Creating one does no git work; `isolate_run` does.
pub struct WorktreeManager {
    repo_root: PathBuf,
    worktree_root: PathBuf,
    ports: Arc<Mutex<PortAllocator>>,
    /// Pluggable git runner (real `git` by default; a closure double in tests).
    git: GitRunner,
}

/// How git commands are executed. The default shells out to the system `git`;
/// tests inject a fake that records invocations without touching a real repo.
#[derive(Clone)]
pub enum GitRunner {
    /// Real `git` via `tokio::process::Command` in `repo_root`.
    System,
    /// Test double: records args, returns success. The recorded log lets tests
    /// assert the exact worktree commands without a real repository.
    Fake(Arc<Mutex<Vec<Vec<String>>>>),
}

impl WorktreeManager {
    pub fn new(repo_root: impl Into<PathBuf>, ports: PortAllocator) -> Self {
        let repo_root = repo_root.into();
        let worktree_root = repo_root.join(".hide").join("wt");
        Self {
            repo_root,
            worktree_root,
            ports: Arc::new(Mutex::new(ports)),
            git: GitRunner::System,
        }
    }

    /// Install a fake git runner (tests). Returns the shared invocation log.
    pub fn with_fake_git(mut self) -> (Self, Arc<Mutex<Vec<Vec<String>>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        self.git = GitRunner::Fake(log.clone());
        (self, log)
    }

    pub fn worktree_root(&self) -> &Path {
        &self.worktree_root
    }

    /// The number of ports currently leased from the pool (the allocator's
    /// truth). The Governor reconciles its occupancy estimate against this so a
    /// release leak can't silently shrink the pool (§4.3 ports level).
    pub fn ports_leased_count(&self) -> usize {
        self.ports.lock().leased_count()
    }

    /// The number of distinct runs holding live worktree/port leases right now
    /// (the allocator's truth, keyed by `run_id`). Used to reconcile the
    /// Governor's worktree occupancy against reality rather than estimating from
    /// the job projection.
    pub fn live_worktree_count(&self) -> usize {
        self.ports.lock().leased_runs()
    }

    /// Create an isolated workspace for `run_id` (§4.3 lifecycle). Steps:
    /// 1. `git worktree add -b hide/<run> .hide/wt/<run> <base_ref>` (own dir, shared .git)
    /// 2. lease a disjoint port range
    /// 3. seed the env namespace (TMPDIR, caches, DB names, PORT, DATABASE_URL)
    /// 4. build the ch.10 sandbox profile scoped to the worktree
    ///
    /// The caller is responsible for emitting `workspace.created` (the manager is
    /// event-log-agnostic so it stays unit-testable; `FleetManager` wires events).
    pub async fn isolate_run(
        &self,
        run_id: &str,
        base_ref: &str,
        n_ports: u16,
    ) -> Result<RunWorkspace, IsolateError> {
        let rel = format!(".hide/wt/{run_id}");
        let path = self.repo_root.join(&rel);
        let branch = format!("hide/{run_id}");

        // 1. worktree add (creates the directory + a fresh branch off base_ref).
        std::fs::create_dir_all(&self.worktree_root)?;
        self.run_git("add", &["worktree", "add", "-b", &branch, &rel, base_ref])
            .await?;

        // 2. ports.
        let ports = self
            .ports
            .lock()
            .lease(run_id, n_ports)
            .ok_or(IsolateError::PortsExhausted { requested: n_ports })?;

        // 3. env namespace.
        let env = env_seed(run_id, &path, &ports);

        // 4. sandbox scoped to the worktree (ch.10 enforces; we shape).
        let sandbox = workspace_sandbox(&path);

        Ok(RunWorkspace {
            run_id: run_id.to_string(),
            worktree: WorktreeLease {
                run_id: run_id.to_string(),
                branch,
                path,
                base_ref: base_ref.to_string(),
                sandbox,
            },
            ports,
            env,
        })
    }

    /// Release a workspace (§4.3 `release_run`): return its ports and remove or
    /// prune its worktree. Discarded runs (tournament losers / speculative
    /// discards) are force-removed; adopted runs are pruned after merge.
    pub async fn release_run(
        &self,
        ws: &RunWorkspace,
        outcome: RunOutcome,
    ) -> Result<(), IsolateError> {
        self.ports.lock().release(&ws.ports);
        let rel = format!(".hide/wt/{}", ws.run_id);
        if outcome.discarded {
            // Force-remove the loser's tree (it has uncommitted work we discard).
            self.run_git("remove", &["worktree", "remove", "--force", &rel])
                .await?;
        } else {
            // Adopted: the branch was merged; remove the now-redundant tree, then
            // prune dangling administrative files.
            let _ = self
                .run_git("remove", &["worktree", "remove", "--force", &rel])
                .await;
        }
        self.run_git("prune", &["worktree", "prune"]).await?;
        Ok(())
    }

    /// List live worktrees under management (`git worktree list --porcelain`,
    /// filtered to our `.hide/wt/` root). Used for GC of orphans (F9).
    pub async fn list(&self) -> Result<Vec<PathBuf>, IsolateError> {
        let out = self
            .run_git_capture("list", &["worktree", "list", "--porcelain"])
            .await?;
        let mut paths = Vec::new();
        for line in out.lines() {
            if let Some(p) = line.strip_prefix("worktree ") {
                let pb = PathBuf::from(p);
                if pb.starts_with(&self.worktree_root) {
                    paths.push(pb);
                }
            }
        }
        Ok(paths)
    }

    /// GC orphaned worktrees left by a crash (F9): prune, then remove any tree
    /// under our root whose run is no longer live. The caller supplies the set of
    /// live run ids; everything else is reaped.
    pub async fn gc_orphans(&self, live_run_ids: &BTreeSet<String>) -> Result<usize, IsolateError> {
        self.run_git("prune", &["worktree", "prune"]).await?;
        let mut reaped = 0;
        for path in self.list().await? {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            if !live_run_ids.contains(&name) {
                let rel = format!(".hide/wt/{name}");
                if self
                    .run_git("remove", &["worktree", "remove", "--force", &rel])
                    .await
                    .is_ok()
                {
                    reaped += 1;
                }
            }
        }
        Ok(reaped)
    }

    async fn run_git(&self, op: &'static str, args: &[&str]) -> Result<(), IsolateError> {
        self.run_git_capture(op, args).await.map(|_| ())
    }

    async fn run_git_capture(
        &self,
        op: &'static str,
        args: &[&str],
    ) -> Result<String, IsolateError> {
        match &self.git {
            GitRunner::Fake(log) => {
                log.lock()
                    .push(args.iter().map(|s| s.to_string()).collect());
                Ok(String::new())
            }
            GitRunner::System => {
                let output = Command::new("git")
                    .args(args)
                    .current_dir(&self.repo_root)
                    .stdin(std::process::Stdio::null())
                    .output()
                    .await?;
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
                } else {
                    Err(IsolateError::Git {
                        op,
                        code: output.status.code(),
                        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                    })
                }
            }
        }
    }
}

/// Seed the per-run env namespace (§4.3 process/env level): a private TMPDIR,
/// build cache, unique DB/schema names, and PORT/DATABASE_URL pointing at the
/// run's leased port range. Injected into the run's shell tools so migrations and
/// dev-servers in different runs never clobber each other.
pub fn env_seed(run_id: &str, worktree: &Path, ports: &PortLease) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    let tmp = worktree.join(".hide-tmp");
    env.insert("TMPDIR".to_string(), tmp.display().to_string());
    env.insert(
        "HIDE_RUN_CACHE".to_string(),
        worktree.join(".hide-cache").display().to_string(),
    );
    // Unique DB/schema names per run (§4.3 "unique DB/schema names hide_run_<id>").
    let db_name = format!("hide_run_{}", sanitize(run_id));
    env.insert("HIDE_DB_NAME".to_string(), db_name.clone());
    env.insert("HIDE_DB_SCHEMA".to_string(), db_name.clone());
    if let Some(&primary) = ports.ports.first() {
        env.insert("PORT".to_string(), primary.to_string());
        env.insert(
            "DATABASE_URL".to_string(),
            format!("postgres://localhost:{primary}/{db_name}"),
        );
    }
    env.insert("HIDE_RUN_ID".to_string(), run_id.to_string());
    env
}

/// A workspace-write sandbox scoped to the run's worktree (ch.10 enforces). Reads
/// + writes confined to the tree; default-deny network.
pub fn workspace_sandbox(worktree: &Path) -> SandboxProfile {
    let root = worktree.display().to_string();
    SandboxProfile {
        tier: SandboxTier::WorkspaceWrite,
        read_roots: vec![root.clone()],
        write_roots: vec![root],
        allowed_commands: Vec::new(),
        network: NetworkPolicy::default(),
    }
}

fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_allocator_leases_disjoint_ranges() {
        let mut alloc = PortAllocator::new(4000, 4005);
        let a = alloc.lease("run_a", 2).unwrap();
        let b = alloc.lease("run_b", 2).unwrap();
        assert_eq!(a.ports.len(), 2);
        assert_eq!(b.ports.len(), 2);
        // Disjoint.
        for p in &a.ports {
            assert!(!b.ports.contains(p));
        }
        // Exhaustion is honest (only 2 left, ask for 3).
        assert!(alloc.lease("run_c", 3).is_none());
        alloc.release(&a);
        // Released ports are reusable.
        assert!(alloc.lease("run_c", 2).is_some());
    }

    #[test]
    fn allocator_tracks_leased_count_and_runs_as_truth() {
        let mut alloc = PortAllocator::new(4000, 4010);
        assert_eq!(alloc.leased_count(), 0);
        assert_eq!(alloc.leased_runs(), 0);

        let a = alloc.lease("run_a", 2).unwrap();
        let b = alloc.lease("run_b", 3).unwrap();
        assert_eq!(alloc.leased_count(), 5, "2 + 3 ports leased");
        assert_eq!(alloc.leased_runs(), 2, "two distinct runs hold leases");

        // Releasing one run returns exactly its ports + forgets the run.
        alloc.release(&a);
        assert_eq!(alloc.leased_count(), 3);
        assert_eq!(alloc.leased_runs(), 1);

        alloc.release(&b);
        assert_eq!(alloc.leased_count(), 0, "pool fully returned to baseline");
        assert_eq!(alloc.leased_runs(), 0);

        // Releasing an empty/synthetic lease is a no-op (idempotent, can't leak).
        alloc.release(&PortLease {
            run_id: "run_a".to_string(),
            ports: Vec::new(),
        });
        assert_eq!(alloc.leased_count(), 0);
        assert_eq!(alloc.leased_runs(), 0);
    }

    #[test]
    fn env_seed_namespaces_db_and_ports() {
        let lease = PortLease {
            run_id: "run_x".to_string(),
            ports: vec![5100, 5101],
        };
        let env = env_seed("run_x", Path::new("/tmp/wt/run_x"), &lease);
        assert_eq!(env.get("PORT").map(String::as_str), Some("5100"));
        assert_eq!(
            env.get("HIDE_DB_NAME").map(String::as_str),
            Some("hide_run_run_x")
        );
        assert!(env.get("DATABASE_URL").unwrap().contains(":5100/"));
        assert!(env.get("TMPDIR").unwrap().contains("run_x"));
    }

    #[tokio::test]
    async fn isolate_run_issues_worktree_add_and_leases_ports() {
        let dir = std::env::temp_dir().join(format!("hide_fleet_iso_{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let (mgr, log) = WorktreeManager::new(&dir, PortAllocator::new(4100, 4110)).with_fake_git();

        let ws = mgr.isolate_run("run_42", "main", 2).await.unwrap();
        assert_eq!(ws.ports.ports.len(), 2);
        assert_eq!(ws.worktree.branch, "hide/run_42");
        assert!(ws.worktree.path.ends_with(".hide/wt/run_42"));
        assert_eq!(ws.worktree.sandbox.tier, SandboxTier::WorkspaceWrite);

        // The exact `git worktree add` invocation was issued.
        {
            let calls = log.lock();
            assert!(calls.iter().any(|c| {
                c.first().map(String::as_str) == Some("worktree")
                    && c.get(1).map(String::as_str) == Some("add")
                    && c.contains(&"main".to_string())
                    && c.contains(&"hide/run_42".to_string())
            }));
        }

        // Release a discarded run → force-remove + prune issued, ports returned.
        mgr.release_run(&ws, RunOutcome { discarded: true })
            .await
            .unwrap();
        {
            let calls = log.lock();
            assert!(calls
                .iter()
                .any(|c| c.contains(&"remove".to_string()) && c.contains(&"--force".to_string())));
            assert!(calls.iter().any(|c| c == &vec!["worktree", "prune"]));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
