//! Resident Genesis body.
//!
//! Loads Qwen3.8 weights once, attaches four isolated logical sessions, and
//! serves propose/health/reload over a Unix socket. Decode remains serialized
//! because the measured concurrent-decode ceiling is 1. This wins load time,
//! residency, and state isolation; it does not claim a tokens/s win.
//!
//! Liveness is the process itself. This binary never writes a status file.

use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
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
const SESSION_ROLES: [&str; 4] = ["parent", "child_a", "child_b", "protected_test"];
const GPU_LANE_LOCK: &str = "/tmp/hawking-gpu-lane.lock";
const GPU_LOCK_WAIT_SECS: u64 = 1_500;

#[cfg(target_os = "macos")]
struct GpuLaneGuard {
    path: PathBuf,
    pid: u32,
}

#[cfg(target_os = "macos")]
impl GpuLaneGuard {
    fn pid_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|status| status.success())
            .unwrap_or(true)
    }

    fn acquire(session_role: &str) -> Result<Self, String> {
        let path = PathBuf::from(GPU_LANE_LOCK);
        let deadline = Instant::now() + Duration::from_secs(GPU_LOCK_WAIT_SECS);
        loop {
            match fs::create_dir(&path) {
                Ok(()) => {
                    let pid = process::id();
                    if let Err(err) =
                        fs::write(path.join("pid"), format!("{pid}\n")).and_then(|_| {
                            fs::write(
                                path.join("owner"),
                                format!("genesis-resident:{session_role}\n"),
                            )
                        })
                    {
                        let _ = fs::remove_dir_all(&path);
                        return Err(format!("initialize GPU lane lock: {err}"));
                    }
                    return Ok(Self { path, pid });
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale_pid = fs::read_to_string(path.join("pid"))
                        .ok()
                        .and_then(|raw| raw.trim().parse::<u32>().ok());
                    if stale_pid.is_some_and(|pid| !Self::pid_alive(pid)) {
                        let _ = fs::remove_dir_all(&path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        let owner = fs::read_to_string(path.join("owner"))
                            .unwrap_or_else(|_| "unknown".to_string());
                        return Err(format!(
                            "GPU lane lock timeout after {GPU_LOCK_WAIT_SECS}s; held by {}",
                            owner.trim()
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(err) => return Err(format!("acquire GPU lane lock: {err}")),
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for GpuLaneGuard {
    fn drop(&mut self) {
        let still_ours = fs::read_to_string(self.path.join("pid"))
            .ok()
            .and_then(|raw| raw.trim().parse::<u32>().ok())
            == Some(self.pid);
        if still_ours {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn usage() -> &'static str {
    "usage: genesis-resident --artifact-root DIR --tokenizer PATH \
        --genesis-contract-provenance JSON \
        [--socket PATH] [--stopfile PATH] [--lineage PATH] [--follow-lineage] [--repo DIR] \
        [--max-seq-len N] [--max-new-tokens N]"
}

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).saturating_mul(8);
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    let mut out = String::with_capacity(64);
    for word in state {
        out.push_str(&format!("{word:08x}"));
    }
    out
}

fn verify_contract_provenance(repo: &Path, raw: &str) -> Value {
    let parsed: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(err) => fail(format!("unparseable --genesis-contract-provenance: {err}")),
    };
    let Some(obj) = parsed.as_object() else {
        fail("--genesis-contract-provenance must be a JSON object");
    };
    let Some(contracts) = obj.get("contracts").and_then(Value::as_array) else {
        fail("--genesis-contract-provenance missing contracts array");
    };
    if contracts.is_empty() {
        fail("--genesis-contract-provenance contracts array is empty");
    }
    let claimed_binding = obj
        .get("binding_sha256")
        .and_then(Value::as_str)
        .unwrap_or_else(|| fail("--genesis-contract-provenance missing binding_sha256"));
    let mut binding = String::new();
    let mut verified: Vec<Value> = Vec::with_capacity(contracts.len());
    for (idx, item) in contracts.iter().enumerate() {
        let Some(row) = item.as_object() else {
            fail(format!("contract[{idx}] is not an object"));
        };
        let path = row
            .get("canonical_path")
            .and_then(Value::as_str)
            .unwrap_or_else(|| fail(format!("contract[{idx}] missing canonical_path")));
        if path.is_empty() || Path::new(path).is_absolute() {
            fail(format!("contract[{idx}] canonical_path must be a repo-relative path"));
        }
        let claimed_sha = row
            .get("sha256")
            .and_then(Value::as_str)
            .unwrap_or_else(|| fail(format!("contract[{idx}] missing sha256")));
        let claimed_size = match row.get("size_bytes") {
            Some(Value::Number(n)) if n.as_u64().is_some() => n.as_u64().unwrap(),
            Some(_) => fail(format!("contract[{idx}] size_bytes must be an integer")),
            None => fail(format!("contract[{idx}] missing size_bytes")),
        };
        let full = repo.join(path);
        let bytes = fs::read(&full).unwrap_or_else(|err| {
            fail(format!(
                "contract[{idx}] missing file {}: {err}",
                full.display()
            ))
        });
        let measured_size = bytes.len() as u64;
        let measured_sha = sha256_hex(&bytes);
        if measured_sha != claimed_sha || measured_size != claimed_size {
            fail(format!(
                "contract sha256/size mismatch: path={path} \
                 claimed_sha256={claimed_sha} measured_sha256={measured_sha} \
                 claimed_size={claimed_size} measured_size={measured_size}"
            ));
        }
        binding.push_str(path);
        binding.push('\0');
        binding.push_str(&measured_sha);
        binding.push('\0');
        binding.push_str(&measured_size.to_string());
        binding.push('\n');
        let mut verified_row = row.clone();
        verified_row.insert("sha256".into(), Value::String(measured_sha));
        verified_row.insert("size_bytes".into(), json!(measured_size));
        verified_row.insert("integrity_verified".into(), Value::Bool(true));
        if !verified_row.contains_key("source_provenance") {
            fail(format!(
                "contract[{idx}] missing source_provenance (do not invent it)"
            ));
        }
        verified.push(Value::Object(verified_row));
    }
    let measured_binding = sha256_hex(binding.as_bytes());
    if measured_binding != claimed_binding {
        fail(format!(
            "binding_sha256 mismatch: claimed={claimed_binding} measured={measured_binding}"
        ));
    }
    let mut out = Map::new();
    for (key, value) in obj {
        out.insert(key.clone(), value.clone());
    }
    out.insert("contracts".into(), Value::Array(verified));
    out.insert(
        "binding_sha256".into(),
        Value::String(measured_binding),
    );
    out.insert("integrity_verified".into(), Value::Bool(true));
    Value::Object(out)
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
    follow_lineage: bool,
    repo: PathBuf,
    genesis_contract_provenance: String,
    max_seq_len: usize,
    max_new_tokens: usize,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut socket = None;
    let mut stopfile = None;
    let mut lineage = None;
    let mut follow_lineage = false;
    let mut repo = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut genesis_contract_provenance = None;
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
            "--socket" => {
                socket = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))))
            }
            "--stopfile" => {
                stopfile = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--lineage" => {
                lineage = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--follow-lineage" => follow_lineage = true,
            "--repo" => repo = PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))),
            "--genesis-contract-provenance" => {
                genesis_contract_provenance =
                    Some(args.next().unwrap_or_else(|| fail(usage())));
            }
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
    let genesis_contract_provenance =
        genesis_contract_provenance.unwrap_or_else(|| fail(usage()));
    let socket = socket.unwrap_or_else(|| PathBuf::from("/tmp/hawking-genesis-resident.sock"));
    let stopfile = stopfile.unwrap_or_else(|| repo.join("workspace/ops/GENESIS_STOP"));
    Args {
        artifact_root,
        tokenizer,
        socket,
        stopfile,
        lineage,
        follow_lineage,
        repo,
        genesis_contract_provenance,
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

fn artifact_manifest_sha(artifact: &Path) -> Result<String, String> {
    let manifest = artifact.join("manifest.json");
    let mut file = File::open(&manifest)
        .map_err(|e| format!("open artifact manifest {}: {e}", manifest.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("read artifact manifest {}: {e}", manifest.display()))?;
    Ok(sha256_hex(&bytes))
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
    sessions: HashMap<String, Qwen38HybridDecodeSession>,
    session_serve_counts: HashMap<String, u64>,
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
    genesis_system_contract: Value,
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
        genesis_system_contract: Value,
    ) -> Result<Self, String> {
        if max_seq_len == 0 {
            return Err("max_seq_len must be positive".into());
        }
        let tok = load_qwen38_tokenizer(tokenizer).map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        eprintln!("genesis-resident: loading {}", artifact.display());
        let weights = Arc::new(Qwen38HybridWeights::load(artifact).map_err(|e| e.to_string())?);
        let mut sessions = HashMap::new();
        let mut session_serve_counts = HashMap::new();
        for role in SESSION_ROLES {
            let session = Qwen38HybridDecodeSession::attach(Arc::clone(&weights), max_seq_len)
                .map_err(|e| format!("attach session {role}: {e}"))?;
            sessions.insert(role.to_string(), session);
            session_serve_counts.insert(role.to_string(), 0);
        }
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
            sessions,
            session_serve_counts,
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
            genesis_system_contract,
        })
    }

    fn reload(
        &mut self,
        artifact: &Path,
        generation: u64,
        artifact_sha: String,
    ) -> Result<(), String> {
        let measured_sha = artifact_manifest_sha(artifact)?;
        if artifact_sha != measured_sha {
            return Err(format!(
                "lineage artifact SHA mismatch for {}: declared {}, measured {}",
                artifact.display(),
                artifact_sha,
                measured_sha
            ));
        }
        if artifact == self.artifact && artifact_sha == self.artifact_sha {
            return Ok(());
        }
        let t0 = Instant::now();
        eprintln!("genesis-resident: reload {}", artifact.display());
        let weights = Arc::new(Qwen38HybridWeights::load(artifact).map_err(|e| e.to_string())?);
        let mut sessions = HashMap::new();
        let mut session_serve_counts = HashMap::new();
        for role in SESSION_ROLES {
            let session = Qwen38HybridDecodeSession::attach(Arc::clone(&weights), self.max_seq_len)
                .map_err(|e| format!("attach session {role}: {e}"))?;
            sessions.insert(role.to_string(), session);
            session_serve_counts.insert(role.to_string(), 0);
        }
        let load_ns = t0.elapsed().as_nanos() as u64;
        self.weights = weights;
        self.sessions = sessions;
        self.session_serve_counts = session_serve_counts;
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
        let workspace_bytes: u64 = self
            .sessions
            .values()
            .map(Qwen38HybridDecodeSession::workspace_resident_bytes)
            .sum();
        let session_workspace_bytes: serde_json::Map<String, Value> = SESSION_ROLES
            .iter()
            .filter_map(|role| {
                self.sessions.get(*role).map(|session| {
                    (
                        (*role).to_string(),
                        json!(session.workspace_resident_bytes()),
                    )
                })
            })
            .collect();
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
            "workspace_bytes": workspace_bytes,
            "session_count": self.sessions.len(),
            "session_roles": SESSION_ROLES,
            "session_semantics": {
                "parent": {"kind": "lineage_parent", "front": "integration_profiling_synthesis"},
                "child_a": {"kind": "worker_session", "front": "gravity_doctor_representation"},
                "child_b": {"kind": "worker_session", "front": "execution_metal_kernel_runtime"},
                "protected_test": {"kind": "protected_test_slot", "front": "candidate_qualification_ab"},
            },
            "lineage_children": 0,
            "session_workspace_bytes": session_workspace_bytes,
            "session_serve_counts": self.session_serve_counts,
            "decode_concurrency": 1,
            "artifact": self.artifact,
            "artifact_sha": self.artifact_sha,
            "generation": self.generation,
            "max_seq_len": self.max_seq_len,
            "reload_error": self.last_reload_error,
            "genesis_system_contract": self.genesis_system_contract,
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
        let session_role = req
            .get("session")
            .and_then(|v| v.as_str())
            .unwrap_or("parent");
        if !SESSION_ROLES.contains(&session_role) {
            return json!({
                "ok": false,
                "error": format!("unknown session role {session_role:?}"),
                "allowed_sessions": SESSION_ROLES,
                "protocol": PROTOCOL,
            });
        }
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
        let _gpu_guard = match GpuLaneGuard::acquire(session_role) {
            Ok(guard) => guard,
            Err(err) => {
                return json!({
                    "ok": false,
                    "error": err,
                    "protocol": PROTOCOL,
                    "session": session_role,
                });
            }
        };
        let t0 = Instant::now();
        let session = match self.sessions.get_mut(session_role) {
            Some(session) => session,
            None => {
                return json!({
                    "ok": false,
                    "error": format!("session role {session_role:?} is not attached"),
                    "protocol": PROTOCOL,
                });
            }
        };
        let result = match generate_greedy(session, &prompt_ids, max_new) {
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
        let role_serve_index = *self
            .session_serve_counts
            .entry(session_role.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);
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
            "session": session_role,
            "session_serve_index": role_serve_index,
            "load_count": self.load_count,
            "load_ns": self.load_ns,
            "body_resident": true,
            "pid": process::id(),
            "rpc_ns": t0.elapsed().as_nanos() as u64,
            "genesis_system_contract": self.genesis_system_contract,
            "genesis_contract_mode": req.get("genesis_contract_mode"),
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
    let genesis_system_contract =
        verify_contract_provenance(&args.repo, &args.genesis_contract_provenance);
    let measured_artifact_sha =
        artifact_manifest_sha(&args.artifact_root).unwrap_or_else(|e| fail(e));
    let generation = match args.lineage.as_ref() {
        Some(path) => match read_lineage_current(path, &args.repo) {
            Some(cur) => {
                if cur.artifact != args.artifact_root || cur.artifact_sha != measured_artifact_sha {
                    eprintln!(
                        "genesis-resident: lineage identity is stale; loading measured artifact {} sha {}",
                        args.artifact_root.display(),
                        measured_artifact_sha
                    );
                }
                cur.generation
            }
            None => 0,
        },
        None => 0,
    };
    let mut body = Body::load(
        &args.artifact_root,
        &args.tokenizer,
        args.max_seq_len,
        args.max_new_tokens,
        generation,
        measured_artifact_sha,
        genesis_system_contract,
    )
    .unwrap_or_else(|e| fail(e));

    if let Some(parent) = args.socket.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::remove_file(&args.socket);
    let listener = UnixListener::bind(&args.socket).unwrap_or_else(|e| fail(e));
    listener.set_nonblocking(true).unwrap_or_else(|e| fail(e));
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
        // A lineage state write is authority, not an instruction to drop the
        // live parent body.  The external lifecycle controller explicitly
        // reloads only after it has checkpointed and rebound every worker.
        // Legacy/manual callers may request the old polling behavior, but the
        // resident supervisor deliberately does not enable it by default.
        if args.follow_lineage {
            body.maybe_reload_lineage(args.lineage.as_deref(), &args.repo);
        }
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
