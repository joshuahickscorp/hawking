//! Deterministic HAIDER tool bus.
//!
//! Models think. Tools know. Every tool result carries enough provenance to
//! become evidence: command, exit status, stdout/stderr, duration, cwd,
//! timestamp, and a deterministic evidence hash.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRequest {
    pub tool: String,
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool: String,
    pub command: String,
    pub exit_status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
    pub cwd: String,
    pub timestamp_ms: u64,
    pub evidence_hash: String,
    pub ok: bool,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolError {
    #[error("{0}")]
    Message(String),
}

struct WorkerInfo {
    id: String,
    command: Vec<String>,
    started_ms: u64,
    status: String,
    output_path: PathBuf,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

pub struct ToolBus {
    root: PathBuf,
    state_dir: PathBuf,
    workers: HashMap<String, WorkerInfo>,
}

impl ToolBus {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let state_dir = root.join(".haider");
        fs::create_dir_all(&state_dir).ok();
        Self {
            root,
            state_dir,
            workers: HashMap::new(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dispatch(&mut self, req: ToolRequest) -> Result<ToolResult, ToolError> {
        let started = Instant::now();
        let (command, exit, stdout, stderr) = self.run_tool(&req.tool, &req.args)?;
        let duration_ms = started.elapsed().as_millis() as u64;
        let exit_str = exit.map(|e| e.to_string()).unwrap_or_default();
        let evidence_hash = evidence_hash(&[&req.tool, &command, &stdout, &stderr, &exit_str]);
        Ok(ToolResult {
            tool: req.tool,
            command,
            exit_status: exit,
            stdout,
            stderr,
            duration_ms,
            cwd: self.root.display().to_string(),
            timestamp_ms: now_ms(),
            evidence_hash,
            ok: exit.map(|e| e == 0).unwrap_or(true),
        })
    }

    fn run_tool(
        &mut self,
        tool: &str,
        args: &Value,
    ) -> Result<(String, Option<i32>, String, String), ToolError> {
        match tool {
            "repo.search" => {
                let query = Self::arg_str(args, "query")
                    .ok_or_else(|| ToolError::Message("repo.search requires query".into()))?;
                let path = Self::arg_str(args, "path").unwrap_or_default();
                let mut cmd = vec!["grep".to_string(), "-n".to_string(), query];
                if !path.is_empty() {
                    cmd.push("--".to_string());
                    cmd.push(path);
                }
                let (exit, stdout, stderr) = self.run("git", &cmd)?;
                Ok((format!("git grep -n {query}"), exit, stdout, stderr))
            }
            "repo.find" => {
                let pattern = Self::arg_str(args, "pattern")
                    .ok_or_else(|| ToolError::Message("repo.find requires pattern".into()))?;
                let (exit, stdout, stderr) = self.run("git", &["ls-files".to_string()])?;
                let filtered: Vec<String> = stdout
                    .lines()
                    .filter(|l| l.contains(&pattern))
                    .map(|l| l.to_string())
                    .collect();
                Ok((
                    format!("git ls-files | filter {pattern}"),
                    exit,
                    filtered.join("\n"),
                    stderr,
                ))
            }
            "repo.read" => {
                let path = Self::arg_str(args, "path")
                    .ok_or_else(|| ToolError::Message("repo.read requires path".into()))?;
                let (exit, stdout, stderr) = self.read_file(&path)?;
                Ok((format!("fs.read {path}"), exit, stdout, stderr))
            }
            "repo.symbols" => {
                let query = Self::arg_str(args, "query")
                    .ok_or_else(|| ToolError::Message("repo.symbols requires query".into()))?;
                let pattern =
                    format!("(fn|struct|enum|impl|class|def)\\s+{}", regex_escape(&query));
                let (exit, stdout, stderr) = self.run(
                    "git",
                    &["grep".into(), "-nE".into(), pattern, "--".into()],
                )?;
                Ok((format!("git grep -nE {pattern}"), exit, stdout, stderr))
            }
            "repo.callers" => {
                let symbol = Self::arg_str(args, "symbol")
                    .ok_or_else(|| ToolError::Message("repo.callers requires symbol".into()))?;
                let (exit, stdout, stderr) =
                    self.run("git", &["grep".into(), "-n".into(), symbol])?;
                Ok((format!("git grep -n {symbol}"), exit, stdout, stderr))
            }
            "git.status" => {
                let (exit, stdout, stderr) =
                    self.run("git", &["status".into(), "--short".into()])?;
                Ok(("git status --short".into(), exit, stdout, stderr))
            }
            "git.diff" => {
                let (exit, stdout, stderr) = self.run("git", &["diff".into()])?;
                Ok(("git diff".into(), exit, stdout, stderr))
            }
            "git.diff_name_only" => {
                let (exit, stdout, stderr) =
                    self.run("git", &["diff".into(), "--name-only".into()])?;
                Ok(("git diff --name-only".into(), exit, stdout, stderr))
            }
            "git.log" => {
                let (exit, stdout, stderr) =
                    self.run("git", &["log".into(), "--oneline".into(), "-20".into()])?;
                Ok(("git log --oneline -20".into(), exit, stdout, stderr))
            }
            "git.show" => {
                let refspec = Self::arg_str(args, "ref")
                    .ok_or_else(|| ToolError::Message("git.show requires ref".into()))?;
                let (exit, stdout, stderr) = self.run("git", &["show".into(), refspec])?;
                Ok((format!("git show {refspec}"), exit, stdout, stderr))
            }
            "git.branch" => {
                let (exit, stdout, stderr) = self.run("git", &["branch".into(), "-a".into()])?;
                Ok(("git branch -a".into(), exit, stdout, stderr))
            }
            "git.worktree" => {
                let (exit, stdout, stderr) =
                    self.run("git", &["worktree".into(), "list".into()])?;
                Ok(("git worktree list".into(), exit, stdout, stderr))
            }
            "fs.read" => {
                let path = Self::arg_str(args, "path")
                    .ok_or_else(|| ToolError::Message("fs.read requires path".into()))?;
                let (exit, stdout, stderr) = self.read_file(&path)?;
                Ok((format!("fs.read {path}"), exit, stdout, stderr))
            }
            "fs.stat" => {
                let path = Self::arg_str(args, "path")
                    .ok_or_else(|| ToolError::Message("fs.stat requires path".into()))?;
                let p = self.root.join(path);
                match fs::metadata(&p) {
                    Ok(m) => {
                        let v = json!({
                            "path": path,
                            "len": m.len(),
                            "is_dir": m.is_dir(),
                            "is_file": m.is_file(),
                        });
                        Ok((format!("fs.stat {path}"), Some(0), v.to_string(), String::new()))
                    }
                    Err(e) => Ok((format!("fs.stat {path}"), Some(1), String::new(), e.to_string())),
                }
            }
            "fs.list" => {
                let path = Self::arg_str(args, "path").unwrap_or_default();
                let p = if path.is_empty() {
                    self.root.clone()
                } else {
                    self.root.join(path)
                };
                match fs::read_dir(&p) {
                    Ok(entries) => {
                        let mut lines = Vec::new();
                        for e in entries.flatten() {
                            lines.push(e.file_name().to_string_lossy().to_string());
                        }
                        Ok((format!("fs.list {path}"), Some(0), lines.join("\n"), String::new()))
                    }
                    Err(e) => Ok((format!("fs.list {path}"), Some(1), String::new(), e.to_string())),
                }
            }
            "shell.run_safe" => {
                let command = Self::arg_vec(args, "command")
                    .ok_or_else(|| ToolError::Message("shell.run_safe requires command".into()))?;
                if command.is_empty() {
                    return Err(ToolError::Message("shell.run_safe command is empty".into()));
                }
                let timeout = Self::arg_u64(args, "timeout_secs").unwrap_or(60);
                let (exit, stdout, stderr) =
                    self.run_with_timeout(&command[0], &command[1..], Duration::from_secs(timeout))?;
                Ok((format!("shell {}", command.join(" ")), exit, stdout, stderr))
            }
            "test.run" | "build.run" | "lint.run" => {
                let command = Self::arg_vec(args, "command")
                    .ok_or_else(|| ToolError::Message(format!("{tool} requires command")))?;
                if command.is_empty() {
                    return Err(ToolError::Message(format!("{tool} command is empty")));
                }
                let timeout = Self::arg_u64(args, "timeout_secs").unwrap_or(300);
                let (exit, stdout, stderr) =
                    self.run_with_timeout(&command[0], &command[1..], Duration::from_secs(timeout))?;
                Ok((format!("{tool} {}", command.join(" ")), exit, stdout, stderr))
            }
            "worker.spawn" => {
                let command = Self::arg_vec(args, "command")
                    .ok_or_else(|| ToolError::Message("worker.spawn requires command".into()))?;
                if command.is_empty() {
                    return Err(ToolError::Message("worker.spawn command is empty".into()));
                }
                let id = Self::arg_str(args, "id").unwrap_or_else(|| format!("w{}", now_ms()));
                let workers_dir = self.state_dir.join("workers");
                fs::create_dir_all(&workers_dir)
                    .map_err(|e| ToolError::Message(e.to_string()))?;
                let output_path = workers_dir.join(format!("{id}.log"));
                let err_path = workers_dir.join(format!("{id}.err"));
                let cancel = Arc::new(AtomicBool::new(false));
                let root = self.root.clone();
                let cmd = command.clone();
                let handle = thread::spawn(move || {
                    let out_file = fs::File::create(&output_path).ok();
                    let err_file = fs::File::create(&err_path).ok();
                    let stdout = out_file.map(Stdio::from).unwrap_or(Stdio::null());
                    let stderr = err_file.map(Stdio::from).unwrap_or(Stdio::null());
                    let mut child = match Command::new(&cmd[0])
                        .args(&cmd[1..])
                        .current_dir(&root)
                        .stdout(stdout)
                        .stderr(stderr)
                        .spawn()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            if let Some(mut f) = fs::File::create(&err_path).ok() {
                                let _ = f.write_all(e.to_string().as_bytes());
                            }
                            return;
                        }
                    };
                    loop {
                        if cancel.load(Ordering::Relaxed) {
                            let _ = child.kill();
                        }
                        match child.try_wait() {
                            Ok(Some(_)) => break,
                            Ok(None) => thread::sleep(Duration::from_millis(50)),
                            Err(_) => break,
                        }
                    }
                });
                self.workers.insert(
                    id.clone(),
                    WorkerInfo {
                        id: id.clone(),
                        command,
                        started_ms: now_ms(),
                        status: "running".into(),
                        output_path: output_path.clone(),
                        cancel,
                        handle: Some(handle),
                    },
                );
                Ok((format!("worker.spawn {id}"), Some(0), id, String::new()))
            }
            "worker.status" => {
                let arr: Vec<Value> = self
                    .workers
                    .values()
                    .map(|w| {
                        json!({
                            "id": w.id,
                            "command": w.command,
                            "status": w.status,
                            "output": w.output_path.display().to_string(),
                        })
                    })
                    .collect();
                Ok((
                    "worker.status".into(),
                    Some(0),
                    json!(arr).to_string(),
                    String::new(),
                ))
            }
            "worker.cancel" => {
                let id = Self::arg_str(args, "id")
                    .ok_or_else(|| ToolError::Message("worker.cancel requires id".into()))?;
                if let Some(w) = self.workers.get_mut(&id) {
                    w.cancel.store(true, Ordering::Relaxed);
                    w.status = "cancelling".into();
                }
                Ok((format!("worker.cancel {id}"), Some(0), id, String::new()))
            }
            "worker.harvest" => {
                let id = Self::arg_str(args, "id");
                let mut out = String::new();
                if let Some(id) = &id {
                    if let Some(w) = self.workers.get(id) {
                        let log = fs::read_to_string(&w.output_path).unwrap_or_default();
                        let err = fs::read_to_string(w.output_path.with_extension("err"))
                            .unwrap_or_default();
                        out = format!("{log}\n{err}");
                    }
                } else {
                    for w in self.workers.values() {
                        let log = fs::read_to_string(&w.output_path).unwrap_or_default();
                        let err = fs::read_to_string(w.output_path.with_extension("err"))
                            .unwrap_or_default();
                        out.push_str(&format!("[{}]\n{}\n{}\n", w.id, log, err));
                    }
                }
                Ok(("worker.harvest".into(), Some(0), out, String::new()))
            }
            "context.status" => {
                let mission = self.state_dir.join("mission.json");
                let receipts = self.state_dir.join("receipts.jsonl");
                let receipt_count = fs::read_to_string(&receipts)
                    .map(|s| s.lines().count())
                    .unwrap_or(0);
                let v = json!({
                    "state_dir": self.state_dir.display().to_string(),
                    "mission_exists": mission.exists(),
                    "receipt_count": receipt_count,
                });
                Ok(("context.status".into(), Some(0), v.to_string(), String::new()))
            }
            "receipt.write" => {
                let mut receipt = args.clone();
                if !receipt.get("id").and_then(|v| v.as_str()).is_some() {
                    receipt["id"] = json!(format!("R-{}", now_ms()));
                }
                let id = receipt
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let path = self.state_dir.join("receipts.jsonl");
                let mut f = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|e| ToolError::Message(e.to_string()))?;
                writeln!(
                    f,
                    "{}",
                    serde_json::to_string(&receipt).map_err(|e| ToolError::Message(e.to_string()))?
                )
                .map_err(|e| ToolError::Message(e.to_string()))?;
                Ok((format!("receipt.write {id}"), Some(0), id, String::new()))
            }
            "receipt.query" => {
                let id = Self::arg_str(args, "id");
                let path = self.state_dir.join("receipts.jsonl");
                let content = fs::read_to_string(&path).unwrap_or_default();
                let mut out = Vec::new();
                for line in content.lines() {
                    if let Ok(v) = serde_json::from_str::<Value>(line) {
                        let matches = id
                            .as_ref()
                            .map(|i| v.get("id").and_then(|x| x.as_str()) == Some(i.as_str()))
                            .unwrap_or(true);
                        if matches {
                            out.push(v);
                        }
                    }
                }
                Ok((
                    "receipt.query".into(),
                    Some(0),
                    serde_json::to_string_pretty(&out)
                        .map_err(|e| ToolError::Message(e.to_string()))?,
                    String::new(),
                ))
            }
            _ => Err(ToolError::Message(format!("unknown tool {tool}"))),
        }
    }

    fn read_file(&self, path: &str) -> Result<(Option<i32>, String, String), ToolError> {
        let p = self.root.join(path);
        match fs::read_to_string(&p) {
            Ok(s) => {
                let truncated = if s.len() > 200_000 {
                    format!("{}...[truncated]", &s[..200_000])
                } else {
                    s
                };
                Ok((Some(0), truncated, String::new()))
            }
            Err(e) => Ok((Some(1), String::new(), e.to_string())),
        }
    }

    fn run(&self, program: &str, argv: &[String]) -> Result<(Option<i32>, String, String), ToolError> {
        self.run_with_timeout(program, argv, Duration::from_secs(30))
    }

    fn run_with_timeout(
        &self,
        program: &str,
        argv: &[String],
        timeout: Duration,
    ) -> Result<(Option<i32>, String, String), ToolError> {
        let mut child = Command::new(program)
            .args(argv)
            .current_dir(&self.root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ToolError::Message(format!("spawn {program}: {e}")))?;
        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let (stdout, stderr) = read_outputs(&mut child);
                    return Ok((Some(status.code().unwrap_or(-1)), stdout, stderr));
                }
                Ok(None) => {
                    if start.elapsed() > timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        let (stdout, stderr) = read_outputs(&mut child);
                        return Ok((Some(-1), stdout, stderr));
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(ToolError::Message(format!("wait: {e}"))),
            }
        }
    }

    fn arg_str(args: &Value, key: &str) -> Option<String> {
        args.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    }

    fn arg_vec(args: &Value, key: &str) -> Option<Vec<String>> {
        args.get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
    }

    fn arg_u64(args: &Value, key: &str) -> Option<u64> {
        args.get(key).and_then(|v| v.as_u64())
    }
}

fn read_outputs(child: &mut std::process::Child) -> (String, String) {
    let mut out = String::new();
    let mut err = String::new();
    if let Some(mut so) = child.stdout.take() {
        let _ = so.read_to_string(&mut out);
    }
    if let Some(mut se) = child.stderr.take() {
        let _ = se.read_to_string(&mut err);
    }
    (out, err)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn evidence_hash(parts: &[&str]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for part in parts {
        for b in part.bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
}

fn regex_escape(s: &str) -> String {
    s.chars()
        .map(|c| {
            if matches!(
                c,
                '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '\\'
            ) {
                format!("\\{c}")
            } else {
                c.to_string()
            }
        })
        .collect()
}
