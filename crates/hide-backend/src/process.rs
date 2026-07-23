//! Session-aware terminal process surface (Trace D).
//!
//! The terminal's `RunCommand` path used to `exec` argv UNSANDBOXED, bypassing the
//! same OS confinement the agent's `shell.run` tool gets. This module makes the
//! terminal a real, supervised process surface that inherits that safety:
//!
//! * **Sandboxed spawn** ([`confine`]) - every managed process is wrapped in the
//!   SAME confinement `hide_tools::shell::run_command` uses: on macOS a
//!   `sandbox-exec` profile rendered from `hide_tools::shell::sandbox_profile` +
//!   `hide_security::sandbox::render_macos_seatbelt_with` (network-deny by default,
//!   writes confined to the workspace); on Linux a bubblewrap (`bwrap`) jail. If no
//!   OS sandbox is available the spawn is REFUSED (fail-closed) rather than run
//!   unconfined. The dangerous-command `SecurityGate` still sits UPSTREAM in the
//!   host; this is the downstream confinement.
//! * **Incremental streaming** - stdout and stderr are read line by line and both
//!   buffered (for later capture) and published as `ToolProgress` UiEvents tagged
//!   with the process id, so the terminal repaints as output arrives, not only at
//!   the end.
//! * **Persistence + attach/detach/stop** - a process keeps running independent of
//!   any session or turn (the supervisor owns the child), so the user can navigate
//!   away and the service stays alive. `attach` replays the buffered output onto a
//!   turn and resumes live mirroring; `detach` keeps it running but stops the live
//!   UI mirror (output is still buffered); `stop` terminates the whole process
//!   group.
//! * **Capture-as-artifact** - the buffered output is written to the durable blob
//!   store, returning a `BlobRef` the transcript can pin.
//! * **Compact state** ([`ProcessState`]) - argv, cwd, env, status, exit code,
//!   sandboxed flag, and the owning run or job.

use crate::ui_bus::UiEventBus;
use hide_core::api::{UiEvent, UiEventKind};
use hide_core::ids::SessionId;
use hide_core::persistence::DynBlobStore;
use hide_core::types::BlobRef;
use hide_tools::shell::{runnable_sbpl, sandbox_render_options, sandbox_profile};
use hide_tools::ShellConfig;
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::process::{ChildStdin, Command};

/// Buffered-output ceiling per process. ponytail: bounded ring, keep the most
/// recent lines; raise the cap or spill incrementally to the blob store if a
/// long-lived service ever needs its full early history captured.
const MAX_BUFFERED_LINES: usize = 20_000;

/// The lifecycle of a managed process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessStatus {
    /// Alive and (possibly) still producing output.
    Running,
    /// Exited on its own with this code (a non-zero code is data, not an error).
    Exited(i32),
    /// Never got off the ground (spawn fault, or a fail-closed sandbox refusal).
    Failed(String),
    /// Terminated by an explicit `stop`.
    Stopped,
}

/// A compact, serializable snapshot of a managed process (requirement (d)).
#[derive(Debug, Clone, Serialize)]
pub struct ProcessState {
    pub id: String,
    pub argv: Vec<String>,
    pub cwd: Option<String>,
    /// The extra env vars this process was started with (only the explicitly set
    /// ones, not the inherited host environment).
    pub env: BTreeMap<String, String>,
    /// "running" | "exited" | "failed" | "stopped".
    pub status: String,
    /// The exit code, once the process has exited on its own.
    pub exit: Option<i32>,
    /// Detail for the `failed` status (spawn fault or sandbox refusal).
    pub fail_reason: Option<String>,
    /// Whether the process is OS-sandboxed (false only via an explicit opt-out).
    pub sandboxed: bool,
    /// A long-lived service that keeps running across navigation.
    pub persistent: bool,
    /// The owning run or job (session id / run id / job id), if any.
    pub owner: Option<String>,
    /// Whether the live UI mirror is currently attached (detach pauses it).
    pub attached: bool,
    /// Last known terminal geometry from a `pty_resize`, if any.
    pub cols: Option<u16>,
    pub rows: Option<u16>,
    /// Number of buffered output lines.
    pub line_count: usize,
    /// The process-group leader pid, while alive.
    pub pid: Option<u32>,
}

/// One supervised process. Shared behind an `Arc`; the reader and waiter tasks
/// hold clones so the record outlives any session or turn.
struct Proc {
    id: String,
    argv: Vec<String>,
    cwd: Option<String>,
    env: BTreeMap<String, String>,
    sandboxed: bool,
    persistent: bool,
    owner: Option<String>,
    pid: Option<u32>,
    status: Mutex<ProcessStatus>,
    lines: Mutex<Vec<String>>,
    geom: Mutex<(Option<u16>, Option<u16>)>,
    /// While false, live output is mirrored onto the bus; detach flips it true.
    detached: AtomicBool,
    stdin: tokio::sync::Mutex<Option<ChildStdin>>,
}

impl Proc {
    fn set_status(&self, next: ProcessStatus) {
        *self.status.lock() = next;
    }

    /// Set the terminal status only if the process is still Running, so an
    /// explicit `stop` (which sets Stopped first) is not overwritten by the
    /// waiter reaping the killed child.
    fn set_terminal_if_running(&self, next: ProcessStatus) {
        let mut s = self.status.lock();
        if *s == ProcessStatus::Running {
            *s = next;
        }
    }

    fn is_running(&self) -> bool {
        *self.status.lock() == ProcessStatus::Running
    }

    fn push_line(&self, line: String) {
        let mut buf = self.lines.lock();
        if buf.len() >= MAX_BUFFERED_LINES {
            buf.remove(0);
        }
        buf.push(line);
    }

    fn snapshot(&self) -> ProcessState {
        let (status, exit, fail_reason) = match &*self.status.lock() {
            ProcessStatus::Running => ("running", None, None),
            ProcessStatus::Exited(code) => ("exited", Some(*code), None),
            ProcessStatus::Failed(why) => ("failed", None, Some(why.clone())),
            ProcessStatus::Stopped => ("stopped", None, None),
        };
        let (cols, rows) = *self.geom.lock();
        ProcessState {
            id: self.id.clone(),
            argv: self.argv.clone(),
            cwd: self.cwd.clone(),
            env: self.env.clone(),
            status: status.to_string(),
            exit,
            fail_reason,
            sandboxed: self.sandboxed,
            persistent: self.persistent,
            owner: self.owner.clone(),
            attached: !self.detached.load(Ordering::Relaxed),
            cols,
            rows,
            line_count: self.lines.lock().len(),
            pid: self.pid,
        }
    }
}

/// What to start. `interactive` pipes stdin so `pty_input` can write to it (a
/// non-interactive one-shot uses a null stdin, matching `shell.run`).
pub struct StartSpec {
    pub argv: Vec<String>,
    pub cwd: Option<String>,
    pub env: BTreeMap<String, String>,
    pub persistent: bool,
    pub owner: Option<String>,
    pub interactive: bool,
}

impl StartSpec {
    /// A plain non-persistent, non-interactive terminal command.
    pub fn command(argv: Vec<String>, cwd: Option<String>) -> Self {
        Self {
            argv,
            cwd,
            env: BTreeMap::new(),
            persistent: false,
            owner: None,
            interactive: false,
        }
    }
}

/// The registry of managed terminal processes. Owned by the host; shared with the
/// reader/waiter tasks it spawns.
pub struct ProcessSupervisor {
    ui_bus: Arc<UiEventBus>,
    procs: Mutex<Vec<Arc<Proc>>>,
    seq: AtomicU64,
}

impl ProcessSupervisor {
    pub fn new(ui_bus: Arc<UiEventBus>) -> Self {
        Self {
            ui_bus,
            procs: Mutex::new(Vec::new()),
            seq: AtomicU64::new(1),
        }
    }

    fn get(&self, id: &str) -> Option<Arc<Proc>> {
        self.procs.lock().iter().find(|p| p.id == id).cloned()
    }

    /// The most recently started process that is still alive (the default target
    /// for a `pty_input`/`pty_resize` that names no explicit process).
    fn latest_live(&self) -> Option<Arc<Proc>> {
        self.procs
            .lock()
            .iter()
            .rev()
            .find(|p| p.is_running())
            .cloned()
    }

    /// Publish one streamed output line as a `ToolProgress` event tagged with the
    /// process id, under an optional session.
    fn emit_line(&self, id: &str, session: Option<SessionId>, message: String) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: session,
            kind: UiEventKind::ToolProgress {
                call_id: id.to_string(),
                message,
                event_id: None,
            },
        });
    }

    /// Start a managed process, sandbox-confined. Returns its id. A spawn fault or
    /// a fail-closed sandbox refusal is recorded as a `Failed` process (queryable
    /// via [`ProcessSupervisor::state`]) with the reason streamed as one line, so
    /// the terminal always shows why nothing ran; the id is still returned.
    pub fn start(&self, spec: StartSpec, shell_config: &ShellConfig) -> String {
        let id = format!("proc:{}", self.seq.fetch_add(1, Ordering::Relaxed));
        let owner_session = spec.owner.as_deref().map(SessionId::from);

        let confined = confine(&spec.argv, shell_config);
        let (mut command, sandboxed) = match confined {
            Ok(c) => (c.command, c.sandboxed),
            Err(reason) => {
                let proc = Arc::new(Proc {
                    id: id.clone(),
                    argv: spec.argv.clone(),
                    cwd: spec.cwd.clone(),
                    env: spec.env.clone(),
                    sandboxed: false,
                    persistent: spec.persistent,
                    owner: spec.owner.clone(),
                    pid: None,
                    status: Mutex::new(ProcessStatus::Failed(reason.clone())),
                    lines: Mutex::new(vec![reason.clone()]),
                    geom: Mutex::new((None, None)),
                    detached: AtomicBool::new(false),
                    stdin: tokio::sync::Mutex::new(None),
                });
                self.procs.lock().push(proc);
                self.emit_line(&id, owner_session, reason);
                return id;
            }
        };

        if let Some(dir) = spec.cwd.clone().or_else(|| shell_config.workspace_root.clone()) {
            command.current_dir(dir);
        }
        for (k, v) in &spec.env {
            command.env(k, v);
        }
        command
            .stdin(if spec.interactive {
                std::process::Stdio::piped()
            } else {
                std::process::Stdio::null()
            })
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        // Own process group so `stop` can signal the whole tree (the sandbox-exec
        // or bwrap wrapper plus the real command underneath), not just the wrapper.
        #[cfg(unix)]
        command.process_group(0);

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(err) => {
                let msg = format!("{}: {}", spec.argv.first().map(String::as_str).unwrap_or(""), err);
                let proc = Arc::new(Proc {
                    id: id.clone(),
                    argv: spec.argv.clone(),
                    cwd: spec.cwd.clone(),
                    env: spec.env.clone(),
                    sandboxed,
                    persistent: spec.persistent,
                    owner: spec.owner.clone(),
                    pid: None,
                    status: Mutex::new(ProcessStatus::Failed(msg.clone())),
                    lines: Mutex::new(vec![msg.clone()]),
                    geom: Mutex::new((None, None)),
                    detached: AtomicBool::new(false),
                    stdin: tokio::sync::Mutex::new(None),
                });
                self.procs.lock().push(proc);
                self.emit_line(&id, owner_session, msg);
                return id;
            }
        };

        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdin = child.stdin.take();

        let proc = Arc::new(Proc {
            id: id.clone(),
            argv: spec.argv.clone(),
            cwd: spec.cwd.clone(),
            env: spec.env.clone(),
            sandboxed,
            persistent: spec.persistent,
            owner: spec.owner.clone(),
            pid,
            status: Mutex::new(ProcessStatus::Running),
            lines: Mutex::new(Vec::new()),
            geom: Mutex::new((None, None)),
            detached: AtomicBool::new(false),
            stdin: tokio::sync::Mutex::new(stdin),
        });
        self.procs.lock().push(proc.clone());

        let mut readers = Vec::new();
        if let Some(out) = stdout {
            readers.push(self.spill(out, proc.clone(), owner_session.clone()));
        }
        if let Some(err) = stderr {
            readers.push(self.spill(err, proc.clone(), owner_session.clone()));
        }

        // Waiter: reap the child (keeps a persistent process alive until it exits
        // or is stopped), drain the readers, then record the terminal status.
        let waited = proc.clone();
        tokio::spawn(async move {
            let status = child.wait().await;
            for r in readers {
                let _ = r.await;
            }
            match status {
                Ok(s) => waited.set_terminal_if_running(ProcessStatus::Exited(s.code().unwrap_or(-1))),
                Err(e) => waited.set_terminal_if_running(ProcessStatus::Failed(e.to_string())),
            }
        });

        id
    }

    /// Spawn a line reader over one child stream: buffer every line, and (unless
    /// detached) mirror it onto the bus as a `ToolProgress` event.
    fn spill<R>(
        &self,
        reader: R,
        proc: Arc<Proc>,
        session: Option<SessionId>,
    ) -> tokio::task::JoinHandle<()>
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
    {
        let bus = self.ui_bus.clone();
        tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(reader).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                proc.push_line(line.clone());
                if !proc.detached.load(Ordering::Relaxed) {
                    bus.publish(UiEvent {
                        seq: 0,
                        session_id: session.clone(),
                        kind: UiEventKind::ToolProgress {
                            call_id: proc.id.clone(),
                            message: line,
                            event_id: None,
                        },
                    });
                }
            }
        })
    }

    /// Whether the process is still alive.
    pub fn is_alive(&self, id: &str) -> bool {
        self.get(id).map(|p| p.is_running()).unwrap_or(false)
    }

    /// A compact snapshot of the process (requirement (d)).
    pub fn state(&self, id: &str) -> Option<ProcessState> {
        self.get(id).map(|p| p.snapshot())
    }

    /// Every managed process, newest last.
    pub fn list(&self) -> Vec<ProcessState> {
        self.procs.lock().iter().map(|p| p.snapshot()).collect()
    }

    /// Attach a (possibly re-navigated) turn to a running process: replay its
    /// buffered output onto the bus under `session`, and resume live mirroring.
    /// Returns the buffered lines (for a verifier to run over).
    pub fn attach(&self, id: &str, session: SessionId) -> Option<Vec<String>> {
        let proc = self.get(id)?;
        proc.detached.store(false, Ordering::Relaxed);
        let lines = proc.lines.lock().clone();
        for line in &lines {
            self.emit_line(id, Some(session.clone()), line.clone());
        }
        Some(lines)
    }

    /// Detach the live UI mirror (the process keeps running and buffering).
    pub fn detach(&self, id: &str) -> bool {
        match self.get(id) {
            Some(proc) => {
                proc.detached.store(true, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    /// The buffered output captured so far.
    pub fn captured(&self, id: &str) -> Option<Vec<String>> {
        self.get(id).map(|p| p.lines.lock().clone())
    }

    /// Stop a process: mark it Stopped, then SIGTERM the whole process group and
    /// SIGKILL after a short grace. Idempotent.
    pub fn stop(&self, id: &str) -> bool {
        let Some(proc) = self.get(id) else {
            return false;
        };
        proc.set_status(ProcessStatus::Stopped);
        #[cfg(unix)]
        if let Some(pid) = proc.pid {
            let pid = pid as i32;
            unsafe {
                libc::kill(-pid, libc::SIGTERM);
            }
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                unsafe {
                    libc::kill(-pid, libc::SIGKILL);
                }
            });
        }
        true
    }

    /// Preserve the process's buffered output as a durable artifact in the blob
    /// store, returning its `BlobRef`.
    pub fn capture_artifact(&self, id: &str, blobs: &DynBlobStore) -> Option<hide_core::Result<BlobRef>> {
        let proc = self.get(id)?;
        let body = proc.lines.lock().join("\n");
        Some(blobs.put(body.into_bytes(), Some("text/plain".to_string())))
    }

    /// Write bytes to a process's stdin (`pty_input`). `id` = `None` targets the
    /// most recently started live process.
    pub async fn write_stdin(&self, id: Option<&str>, data: &str) -> Result<(), String> {
        let proc = match id {
            Some(id) => self.get(id).ok_or_else(|| format!("unknown process {id}"))?,
            None => self.latest_live().ok_or_else(|| "no live process".to_string())?,
        };
        let mut guard = proc.stdin.lock().await;
        let stdin = guard
            .as_mut()
            .ok_or_else(|| "process has no writable stdin (not interactive)".to_string())?;
        stdin
            .write_all(data.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        stdin.flush().await.map_err(|e| e.to_string())
    }

    /// Record a terminal resize (`pty_resize`). We are not a full PTY, so this is
    /// metadata surfaced in [`ProcessState`]; `id` = `None` targets the latest
    /// live process.
    pub fn resize(&self, id: Option<&str>, cols: u16, rows: u16) -> Result<(), String> {
        let proc = match id {
            Some(id) => self.get(id).ok_or_else(|| format!("unknown process {id}"))?,
            None => self.latest_live().ok_or_else(|| "no live process".to_string())?,
        };
        *proc.geom.lock() = (Some(cols), Some(rows));
        Ok(())
    }
}

/// A confined command ready to spawn, plus whether it is OS-sandboxed.
struct Confined {
    command: Command,
    sandboxed: bool,
}

/// Build the sandbox-confined command, reusing the SAME rendering pipeline
/// `hide_tools::shell` uses (no policy is duplicated: the macOS SBPL comes wholly
/// from `sandbox_profile` + `render_macos_seatbelt_with` + `runnable_sbpl`).
///
/// Fail-closed: with no usable OS sandbox we return `Err` rather than run
/// unconfined, unless the caller set `disable_sandbox` (already-confined worktree)
/// or `allow_unconfined` (the explicit escape hatch).
fn confine(argv: &[String], config: &ShellConfig) -> Result<Confined, String> {
    if config.disable_sandbox {
        return Ok(Confined {
            command: bare(argv),
            sandboxed: false,
        });
    }

    #[cfg(target_os = "macos")]
    {
        if std::path::Path::new("/usr/bin/sandbox-exec").exists() {
            let profile = sandbox_profile(config, argv);
            let opts = sandbox_render_options(config);
            let rendered = hide_security::sandbox::render_macos_seatbelt_with(&profile, &opts);
            let sbpl = runnable_sbpl(&rendered.profile_text);
            let mut c = Command::new("/usr/bin/sandbox-exec");
            c.arg("-p").arg(sbpl).arg("--").args(argv);
            return Ok(Confined {
                command: c,
                sandboxed: true,
            });
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(bwrap) = bwrap_path() {
            return Ok(Confined {
                command: bubblewrap(&bwrap, argv, config),
                sandboxed: true,
            });
        }
    }

    if config.allow_unconfined {
        return Ok(Confined {
            command: bare(argv),
            sandboxed: false,
        });
    }

    Err("SANDBOX_UNAVAILABLE: refusing to run unconfined (no macOS sandbox-exec / Linux bwrap)"
        .to_string())
}

fn bare(argv: &[String]) -> Command {
    let mut c = Command::new(&argv[0]);
    c.args(&argv[1..]);
    c
}

/// Resolve `bwrap` on PATH (Linux confinement route).
#[cfg(target_os = "linux")]
fn bwrap_path() -> Option<String> {
    let path = std::env::var("PATH").ok()?;
    for dir in path.split(':') {
        let cand = std::path::Path::new(dir).join("bwrap");
        if cand.exists() {
            return Some(cand.to_string_lossy().into_owned());
        }
    }
    None
}

/// Wrap argv in bubblewrap: read-only root, a writable worktree + tmp, and
/// `--unshare-net` (network denied by default). ponytail: mirrors
/// `hide_tools::shell::bubblewrap_command`; that builder is private, so the small
/// arg list is replicated here rather than widening its crate API.
#[cfg(target_os = "linux")]
fn bubblewrap(bwrap: &str, argv: &[String], config: &ShellConfig) -> Command {
    let opts = sandbox_render_options(config);
    let tmp = std::env::temp_dir().to_string_lossy().into_owned();
    let mut c = Command::new(bwrap);
    c.arg("--ro-bind").arg("/").arg("/");
    if let Some(root) = &opts.worktree_root {
        c.arg("--bind").arg(root).arg(root);
    }
    c.arg("--bind").arg(&tmp).arg(&tmp);
    c.arg("--dev").arg("/dev");
    c.arg("--proc").arg("/proc");
    c.arg("--unshare-net");
    c.arg("--die-with-parent");
    c.arg("--").args(argv);
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn macos_sandbox_available() -> bool {
        cfg!(target_os = "macos") && std::path::Path::new("/usr/bin/sandbox-exec").exists()
    }

    /// A workspace-rooted config so the sandbox profile canonicalizes a real dir.
    fn config() -> ShellConfig {
        let dir = std::env::temp_dir();
        ShellConfig {
            workspace_root: Some(dir.to_string_lossy().into_owned()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn fail_closed_when_no_sandbox_and_no_optout() {
        // With no OS sandbox and no opt-out, confine refuses; a start records a
        // Failed process rather than running unconfined.
        if macos_sandbox_available() {
            return; // this host has a sandbox; the refusal path is not reachable
        }
        let sup = ProcessSupervisor::new(Arc::new(UiEventBus::default()));
        let id = sup.start(StartSpec::command(vec!["true".to_string()], None), &config());
        let state = sup.state(&id).unwrap();
        assert_eq!(state.status, "failed");
        assert!(!state.sandboxed);
    }

    #[tokio::test]
    async fn disable_sandbox_runs_bare_and_captures_output() {
        // The opt-out path is portable (no sandbox binary needed), so it exercises
        // spawn + streaming + capture on any host.
        let cfg = ShellConfig {
            disable_sandbox: true,
            ..Default::default()
        };
        let sup = ProcessSupervisor::new(Arc::new(UiEventBus::default()));
        let id = sup.start(
            StartSpec::command(vec!["printf".to_string(), "hi".to_string()], None),
            &cfg,
        );
        // Wait for exit.
        for _ in 0..50 {
            if !sup.is_alive(&id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let state = sup.state(&id).unwrap();
        assert_eq!(state.status, "exited");
        assert_eq!(state.exit, Some(0));
        assert!(!state.sandboxed);
        assert!(sup.captured(&id).unwrap().iter().any(|l| l.contains("hi")));
    }

    #[tokio::test]
    async fn interactive_stdin_echo_roundtrips() {
        // pty_input: write a line to an interactive process's stdin and see it
        // echoed back on the buffered output. Uses the portable opt-out path.
        let cfg = ShellConfig {
            disable_sandbox: true,
            ..Default::default()
        };
        let sup = ProcessSupervisor::new(Arc::new(UiEventBus::default()));
        let id = sup.start(
            StartSpec {
                argv: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "while read line; do echo got $line; done".to_string(),
                ],
                cwd: None,
                env: BTreeMap::new(),
                persistent: true,
                owner: None,
                interactive: true,
            },
            &cfg,
        );
        sup.write_stdin(Some(&id), "ping\n").await.unwrap();
        let mut saw = false;
        for _ in 0..50 {
            if sup
                .captured(&id)
                .unwrap()
                .iter()
                .any(|l| l.contains("got ping"))
            {
                saw = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(saw, "stdin write should be echoed back on stdout");
        sup.stop(&id);
    }
}
