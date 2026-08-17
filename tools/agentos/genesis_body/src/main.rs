//! Resident Genesis body.
//!
//! Loads Qwen3.8 weights once, attaches one decode session (the measured
//! concurrent-decode ceiling is 1), and serves propose/health/reload over a
//! Unix socket. This wins LOAD TIME and RESIDENCY, not tokens/s.
//!
//! Liveness is the process itself. This binary never writes a status file.

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, render_qwen38_user_chat, Qwen38HybridDecodeSession,
    Qwen38HybridWeights,
};
#[cfg(target_os = "macos")]
use hawking_core::tokenizer::Tokenizer;

const PROTOCOL: &str = "hawking.genesis.resident.v1";

fn usage() -> &'static str {
    "usage: genesis-resident --artifact-root DIR --tokenizer PATH \
        [--socket PATH] [--stopfile PATH] [--lineage PATH] [--repo DIR] \
        [--max-seq-len N] [--max-new-tokens N]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("genesis-resident: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    socket: PathBuf,
    stopfile: PathBuf,
    lineage: Option<PathBuf>,
    repo: PathBuf,
    max_seq_len: usize,
    max_new_tokens: usize,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut socket = None;
    let mut stopfile = None;
    let mut lineage = None;
    let mut repo = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut max_seq_len = 4096usize;
    let mut max_new_tokens = 900usize;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--socket" => socket = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            "--stopfile" => {
                stopfile = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--lineage" => {
                lineage = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--repo" => repo = PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))),
            "--max-seq-len" => {
                max_seq_len = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-seq-len"));
            }
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-new-tokens"));
            }
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    let artifact_root = artifact_root.unwrap_or_else(|| fail(usage()));
    let tokenizer = tokenizer.unwrap_or_else(|| fail(usage()));
    let socket = socket.unwrap_or_else(|| PathBuf::from("/tmp/hawking-genesis-resident.sock"));
    let stopfile = stopfile.unwrap_or_else(|| repo.join("workspace/ops/GENESIS_STOP"));
    Args {
        artifact_root,
        tokenizer,
        socket,
        stopfile,
        lineage,
        repo,
        max_seq_len,
        max_new_tokens,
    }
}

struct LineageCurrent {
    generation: u64,
    artifact: PathBuf,
    artifact_sha: String,
}

fn resolve_artifact(repo: &Path, raw: &Path) -> PathBuf {
    if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        repo.join(raw)
    }
}

fn read_lineage_current(path: &Path, repo: &Path) -> Option<LineageCurrent> {
    let raw = fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&raw).ok()?;
    let cur = v.get("slots")?.get("CURRENT")?;
    if cur.is_null() {
        return None;
    }
    let generation = cur.get("generation")?.as_u64()?;
    let artifact = cur
        .get("identity")?
        .get("artifact")?
        .as_str()
        .map(PathBuf::from)?;
    let artifact_sha = cur.get("artifact_sha")?.as_str()?.to_string();
    Some(LineageCurrent {
        generation,
        artifact: resolve_artifact(repo, &artifact),
        artifact_sha,
    })
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("genesis-resident is Metal-only");
}

#[cfg(target_os = "macos")]
struct Body {
    weights: Arc<Qwen38HybridWeights>,
    session: Qwen38HybridDecodeSession,
    tokenizer: Tokenizer,
    artifact: PathBuf,
    artifact_sha: String,
    generation: u64,
    max_seq_len: usize,
    default_max_new: usize,
    load_ns: u64,
    load_count: u64,
    serve_count: u64,
    started: Instant,
    started_unix_ns: u64,
    last_reload_error: Option<String>,
}

#[cfg(target_os = "macos")]
impl Body {
    fn load(
        artifact: &Path,
        tokenizer: &Path,
        max_seq_len: usize,
        default_max_new: usize,
        generation: u64,
        artifact_sha: String,
    ) -> Result<Self, String> {
        if max_seq_len == 0 {
            return Err("max_seq_len must be positive".into());
        }
        let tok = load_qwen38_tokenizer(tokenizer).map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        eprintln!("genesis-resident: loading {}", artifact.display());
        let weights = Arc::new(Qwen38HybridWeights::load(artifact).map_err(|e| e.to_string())?);
        let session =
            Qwen38HybridDecodeSession::attach(Arc::clone(&weights), max_seq_len)
                .map_err(|e| e.to_string())?;
        let load_ns = t0.elapsed().as_nanos() as u64;
        eprintln!(
            "genesis-resident: body resident {:.3}s weight_bytes={}",
            load_ns as f64 / 1e9,
            weights.resident_bytes()
        );
        let started_unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Ok(Self {
            weights,
            session,
            tokenizer: tok,
            artifact: artifact.to_path_buf(),
            artifact_sha,
            generation,
            max_seq_len,
            default_max_new,
            load_ns,
            load_count: 1,
            serve_count: 0,
            started: Instant::now(),
            started_unix_ns,
            last_reload_error: None,
        })
    }

    fn reload(
        &mut self,
        artifact: &Path,
        generation: u64,
        artifact_sha: String,
    ) -> Result<(), String> {
        if artifact == self.artifact && artifact_sha == self.artifact_sha {
            return Ok(());
        }
        let t0 = Instant::now();
        eprintln!("genesis-resident: reload {}", artifact.display());
        let weights = Arc::new(Qwen38HybridWeights::load(artifact).map_err(|e| e.to_string())?);
        let session = Qwen38HybridDecodeSession::attach(Arc::clone(&weights), self.max_seq_len)
            .map_err(|e| e.to_string())?;
        let load_ns = t0.elapsed().as_nanos() as u64;
        self.weights = weights;
        self.session = session;
        self.artifact = artifact.to_path_buf();
        self.artifact_sha = artifact_sha;
        self.generation = generation;
        self.load_ns = load_ns;
        self.load_count += 1;
        self.last_reload_error = None;
        eprintln!(
            "genesis-resident: reloaded {:.3}s load_count={}",
            load_ns as f64 / 1e9,
            self.load_count
        );
        Ok(())
    }

    fn maybe_reload_lineage(&mut self, lineage: Option<&Path>, repo: &Path) {
        let Some(path) = lineage else {
            return;
        };
        let Some(cur) = read_lineage_current(path, repo) else {
            return;
        };
        if cur.artifact_sha == self.artifact_sha && cur.artifact == self.artifact {
            return;
        }
        if let Err(err) = self.reload(&cur.artifact, cur.generation, cur.artifact_sha) {
            self.last_reload_error = Some(err);
        }
    }

    fn health(&self) -> Value {
        json!({
            "ok": true,
            "protocol": PROTOCOL,
            "pid": process::id(),
            "body_resident": true,
            "uptime_ns": self.started.elapsed().as_nanos() as u64,
            "started_unix_ns": self.started_unix_ns,
            "load_ns": self.load_ns,
            "load_count": self.load_count,
            "serve_count": self.serve_count,
            "resident_weight_bytes": self.weights.resident_bytes(),
            "workspace_bytes": self.session.workspace_resident_bytes(),
            "artifact": self.artifact,
            "artifact_sha": self.artifact_sha,
            "generation": self.generation,
            "max_seq_len": self.max_seq_len,
            "reload_error": self.last_reload_error,
        })
    }

    fn propose(&mut self, req: &Value) -> Value {
        let prompt = match req.get("prompt").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => p,
            _ => {
                return json!({"ok": false, "error": "prompt required", "protocol": PROTOCOL});
            }
        };
        let max_new = req
            .get("max_new_tokens")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .filter(|n| *n > 0)
            .unwrap_or(self.default_max_new);
        let raw = req.get("raw").and_then(|v| v.as_bool()).unwrap_or(false);
        let rendered = if raw {
            prompt.to_owned()
        } else {
            render_qwen38_user_chat(prompt)
        };
        let prompt_ids = match self.tokenizer.encode(&rendered, false) {
            Ok(ids) => ids,
            Err(err) => {
                return json!({
                    "ok": false,
                    "error": format!("encode: {err}"),
                    "protocol": PROTOCOL
                });
            }
        };
        if prompt_ids.len() + max_new > self.max_seq_len {
            return json!({
                "ok": false,
                "error": format!(
                    "prompt_len {} + max_new {max_new} exceeds max_seq_len {}",
                    prompt_ids.len(),
                    self.max_seq_len
                ),
                "protocol": PROTOCOL,
            });
        }
        let t0 = Instant::now();
        let result = match generate_greedy(&mut self.session, &prompt_ids, max_new) {
            Ok(r) => r,
            Err(err) => {
                return json!({
                    "ok": false,
                    "error": format!("generate: {err}"),
                    "protocol": PROTOCOL
                });
            }
        };
        let text = match result.decode_new(&self.tokenizer) {
            Ok(t) => t,
            Err(err) => {
                return json!({
                    "ok": false,
                    "error": format!("decode: {err}"),
                    "protocol": PROTOCOL
                });
            }
        };
        self.serve_count += 1;
        json!({
            "ok": true,
            "protocol": PROTOCOL,
            "text": text,
            "fallbacks": result.fallbacks,
            "wall_ns": result.wall_ns,
            "prefill_wall_ns": result.prefill_wall_ns,
            "decode_wall_ns": result.decode_wall_ns,
            "new_tokens": result.new_tokens(),
            "prompt_len": result.prompt_len,
            "serve_index": self.serve_count,
            "load_count": self.load_count,
            "load_ns": self.load_ns,
            "body_resident": true,
            "pid": process::id(),
            "rpc_ns": t0.elapsed().as_nanos() as u64,
        })
    }
}

#[cfg(target_os = "macos")]
fn write_json_line(stream: &mut UnixStream, value: &Value) -> std::io::Result<()> {
    let mut line = serde_json::to_string(value).map_err(std::io::Error::other)?;
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    stream.flush()
}

#[cfg(target_os = "macos")]
fn serve_connection(stream: &mut UnixStream, body: &mut Body, stop: &mut bool) {
    let mut reader = match stream.try_clone() {
        Ok(cloned) => BufReader::new(cloned),
        Err(err) => {
            eprintln!("genesis-resident: clone stream: {err}");
            return;
        }
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(1800)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(1800)));
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => return,
            Ok(_) => {}
            Err(err) => {
                eprintln!("genesis-resident: read: {err}");
                return;
            }
        }
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(err) => {
                let _ = write_json_line(
                    stream,
                    &json!({"ok": false, "error": format!("bad json: {err}"), "protocol": PROTOCOL}),
                );
                continue;
            }
        };
        let op = req.get("op").and_then(|v| v.as_str()).unwrap_or("");
        let resp = match op {
            "health" | "ping" => body.health(),
            "propose" => body.propose(&req),
            "reload" => {
                let artifact = req
                    .get("artifact")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .unwrap_or_else(|| body.artifact.clone());
                let generation = req
                    .get("generation")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(body.generation);
                let sha = req
                    .get("artifact_sha")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                match body.reload(&artifact, generation, sha) {
                    Ok(()) => body.health(),
                    Err(err) => {
                        body.last_reload_error = Some(err.clone());
                        json!({"ok": false, "error": err, "protocol": PROTOCOL})
                    }
                }
            }
            "stop" => {
                *stop = true;
                json!({"ok": true, "stopped": true, "protocol": PROTOCOL, "pid": process::id()})
            }
            other => json!({
                "ok": false,
                "error": format!("unknown op {other:?}"),
                "protocol": PROTOCOL
            }),
        };
        if write_json_line(stream, &resp).is_err() {
            return;
        }
        if *stop {
            return;
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
    if args.stopfile.exists() {
        eprintln!("genesis-resident: stopfile present, not starting");
        process::exit(0);
    }
    let (generation, artifact_sha) = match args.lineage.as_ref() {
        Some(path) => match read_lineage_current(path, &args.repo) {
            Some(cur) => (cur.generation, cur.artifact_sha),
            None => (0, String::new()),
        },
        None => (0, String::new()),
    };
    let mut body = Body::load(
        &args.artifact_root,
        &args.tokenizer,
        args.max_seq_len,
        args.max_new_tokens,
        generation,
        artifact_sha,
    )
    .unwrap_or_else(|e| fail(e));

    if let Some(parent) = args.socket.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::remove_file(&args.socket);
    let listener = UnixListener::bind(&args.socket).unwrap_or_else(|e| fail(e));
    listener
        .set_nonblocking(true)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "genesis-resident: listening {} pid={}",
        args.socket.display(),
        process::id()
    );

    let mut stop = false;
    while !stop {
        if args.stopfile.exists() {
            eprintln!("genesis-resident: GENESIS_STOP present, exiting");
            break;
        }
        body.maybe_reload_lineage(args.lineage.as_deref(), &args.repo);
        match listener.accept() {
            Ok((mut stream, _)) => {
                let _ = stream.set_nonblocking(false);
                serve_connection(&mut stream, &mut body, &mut stop);
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(err) => {
                eprintln!("genesis-resident: accept: {err}");
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
    let _ = fs::remove_file(&args.socket);
    eprintln!("genesis-resident: exit 0");
}
