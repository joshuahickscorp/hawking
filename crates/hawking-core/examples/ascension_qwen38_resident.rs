//! Persistent Hawking-native Qwen3.8 resident.
//!
//! The process opens the packed artifact once, uploads the weights once, and
//! serves correlated JSONL requests over stdin/stdout.  stdout is protocol
//! only; diagnostics stay on stderr so HCLI can safely parse every line.
//!
//! The wire contract is intentionally small and model-runtime friendly:
//!
//! ```text
//! ready: {"status":"ready", ...}
//! request: {"id":"...", "prompt":"...", "max_new_tokens":N}
//! reply: {"id":"...", "status":"ok", "text":"...", ...}
//! ```
//!
//! Qwen3.8 is the current provider.  HCLI treats this executable as one
//! RuntimeBackend behind the generic native connector; other model families
//! can provide the same JSONL contract without changing AgentOS.

#![recursion_limit = "256"]

use serde_json::{json, Value};
use std::env;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, Qwen38HybridDecodeSession,
    Qwen38HybridWeights,
};

const PROTOCOL: &str = "hawking.qwen38.resident.v1";

fn usage() -> &'static str {
    "usage: ascension_qwen38_resident --artifact-root DIR --tokenizer PATH \
        --max-seq-len N [--resident-identity NAME]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_resident: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    max_seq_len: usize,
    resident_identity: String,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut max_seq_len = None;
    let mut resident_identity = "sealed-3.14".to_owned();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                println!("{}", usage());
                process::exit(0);
            }
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(
                    args.next().unwrap_or_else(|| fail(usage())),
                ));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(
                    args.next().unwrap_or_else(|| fail(usage())),
                ));
            }
            "--max-seq-len" => {
                max_seq_len = Some(
                    args.next()
                        .unwrap_or_else(|| fail(usage()))
                        .parse::<usize>()
                        .unwrap_or_else(|_| fail("--max-seq-len must be a positive integer")),
                );
            }
            "--resident-identity" => {
                resident_identity = args.next().unwrap_or_else(|| fail(usage()));
            }
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    let max_seq_len = max_seq_len.unwrap_or_else(|| fail(usage()));
    if max_seq_len == 0 {
        fail("--max-seq-len must be positive");
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        max_seq_len,
        resident_identity,
    }
}

fn write_record(stdout: &mut impl Write, body: &Value) -> Result<(), String> {
    serde_json::to_writer(&mut *stdout, body).map_err(|e| format!("encode JSONL reply: {e}"))?;
    stdout
        .write_all(b"\n")
        .map_err(|e| format!("write JSONL reply: {e}"))?;
    stdout
        .flush()
        .map_err(|e| format!("flush JSONL reply: {e}"))
}

fn request_id(body: &Value) -> String {
    body.get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned()
}

fn error_reply(id: &str, error: impl std::fmt::Display) -> Value {
    json!({
        "id": id,
        "status": "error",
        "protocol": PROTOCOL,
        "error": error.to_string(),
    })
}

fn sum_optional(values: &[Option<u64>]) -> Option<u64> {
    let mut total = 0u64;
    for value in values {
        total = total.checked_add((*value)?)?;
    }
    Some(total)
}

fn sum_values(values: &[u64]) -> u64 {
    values.iter().copied().fold(0u64, u64::saturating_add)
}

#[cfg(target_os = "macos")]
fn run_resident(args: Args) -> Result<(), String> {
    let weights = Qwen38HybridWeights::load(&args.artifact_root)
        .map_err(|e| format!("load artifact: {e}"))?;
    let dense_w_materialized = weights.dense_w_materialized;
    let weights = std::sync::Arc::new(weights);
    let tokenizer = load_qwen38_tokenizer(&args.tokenizer)
        .map_err(|e| format!("load tokenizer: {e}"))?;
    let mut session = Qwen38HybridDecodeSession::attach(
        std::sync::Arc::clone(&weights),
        args.max_seq_len,
    )
    .map_err(|e| format!("allocate resident session: {e}"))?;

    let pid = process::id();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    write_record(
        &mut stdout,
        &json!({
            "status": "ready",
            "protocol": PROTOCOL,
            "resident_identity": args.resident_identity,
            "resident_pid": pid,
            "max_seq_len": args.max_seq_len,
            "model_open_count": 1,
            "weight_upload_count": 1,
            "dense_w_materialized": dense_w_materialized,
            "resident_weight_bytes": weights.resident_bytes(),
            "workspace_resident_bytes": session.workspace_resident_bytes(),
            "fallbacks": 0,
        }),
    )?;

    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read JSONL request: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed = serde_json::from_str::<Value>(&line);
        let body = match parsed {
            Ok(value) if value.is_object() => value,
            Ok(_) => {
                write_record(&mut stdout, &error_reply("", "request is not a JSON object"))?;
                continue;
            }
            Err(error) => {
                write_record(
                    &mut stdout,
                    &error_reply("", format!("request is not valid JSON: {error}")),
                )?;
                continue;
            }
        };
        let id = request_id(&body);
        let reply = match serve_request(&mut session, &tokenizer, &args, pid, &body) {
            Ok(reply) => reply,
            Err(error) => error_reply(&id, error),
        };
        write_record(&mut stdout, &reply)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn serve_request(
    session: &mut Qwen38HybridDecodeSession,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    args: &Args,
    pid: u32,
    body: &Value,
) -> Result<Value, String> {
    let id = request_id(body);
    if id.is_empty() {
        return Err("request id is required".to_owned());
    }
    let prompt = body
        .get("prompt")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| "request prompt must be a non-empty string".to_owned())?;
    let max_new = body
        .get("max_new_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| "max_new_tokens must be a positive integer".to_owned())?;
    if max_new == 0 {
        return Err("max_new_tokens must be positive".to_owned());
    }
    let max_new = usize::try_from(max_new)
        .map_err(|_| "max_new_tokens does not fit in usize".to_owned())?;
    let requested_seq = body
        .get("max_seq_len")
        .and_then(Value::as_u64)
        .map(|value| usize::try_from(value).map_err(|_| "max_seq_len does not fit in usize"))
        .transpose()?;
    if let Some(requested) = requested_seq {
        if requested > args.max_seq_len {
            return Err(format!(
                "request max_seq_len {requested} exceeds resident max_seq_len {}",
                args.max_seq_len
            ));
        }
    }

    let prompt_ids = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("encode prompt: {e}"))?;
    let required = prompt_ids
        .len()
        .checked_add(max_new)
        .ok_or_else(|| "prompt plus generation length overflowed".to_owned())?;
    if required > args.max_seq_len {
        return Err(format!(
            "prompt has {} tokens and max_new_tokens is {max_new}; resident max_seq_len is {}",
            prompt_ids.len(), args.max_seq_len
        ));
    }

    // The same session is reused, but recurrent and KV state is never shared
    // across HCLI requests. generate_greedy also resets defensively; keeping
    // this explicit here makes the resident isolation contract visible.
    session.reset();
    let started = Instant::now();
    let result = generate_greedy(session, &prompt_ids, max_new)
        .map_err(|e| format!("native generation: {e}"))?;
    let wall_ns = started.elapsed().as_nanos() as u64;
    let generated = result.new_tokens().to_vec();
    let generated_count = generated.len();
    let split = result.prompt_len.min(result.gpu_ns.len());
    let dispatch_split = result.prompt_len.min(result.dispatches.len());
    let prefill_gpu_ns = sum_optional(&result.gpu_ns[..split]);
    let decode_gpu_ns = sum_optional(&result.gpu_ns[split..]);
    let complete_gpu_ns = sum_optional(&result.gpu_ns);
    let prefill_dispatches = sum_values(&result.dispatches[..dispatch_split]);
    let decode_dispatches = sum_values(&result.dispatches[dispatch_split..]);
    let dispatches = sum_values(&result.dispatches);
    let active_weight_bytes_total = sum_values(&result.active_weight_bytes);
    let active_weight_bytes_per_generated_token = if generated_count > 0 {
        Some(active_weight_bytes_total as f64 / generated_count as f64)
    } else {
        None
    };
    let kernel_histogram = session.dispatched_kernel_histogram();
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| format!("decode generated tokens: {e}"))?;
    let complete_wall_ns = started.elapsed().as_nanos() as u64;
    let complete_wall_ns_per_generated_token = if generated_count > 0 {
        Some(complete_wall_ns as f64 / generated_count as f64)
    } else {
        None
    };
    let gpu_ns_per_generated_token = complete_gpu_ns.map(|value| {
        value as f64 / generated_count.max(1) as f64
    });
    let dispatches_per_generated_token = dispatches as f64 / generated_count.max(1) as f64;
    let complete_tps = if wall_ns > 0 {
        Some(generated_count as f64 / (wall_ns as f64 / 1e9))
    } else {
        None
    };
    let decode_tps = if result.decode_wall_ns > 0 {
        Some(result.decode_steps as f64 / (result.decode_wall_ns as f64 / 1e9))
    } else {
        None
    };
    Ok(json!({
        "id": id,
        "status": "ok",
        "protocol": PROTOCOL,
        "resident_identity": args.resident_identity,
        "resident_pid": pid,
        "text": text,
        "generated_text": text,
        "new_token_ids": generated,
        "generated_tokens": generated_count,
        "prompt_tokens": result.prompt_len,
        "prompt_len": result.prompt_len,
        "wall_ns": wall_ns,
        "generation_wall_ns": result.wall_ns,
        "prefill_wall_ns": result.prefill_wall_ns,
        "decode_wall_ns": result.decode_wall_ns,
        "decode_steps": result.decode_steps,
        "complete_tps": complete_tps,
        "decode_tps": decode_tps,
        "fallbacks": result.fallbacks,
        "dense_w_materialized": result.dense_w_materialized,
        "resident_weight_bytes": result.resident_weight_bytes,
        "workspace_resident_bytes": result.workspace_resident_bytes,
        "active_bytes_per_token": active_weight_bytes_per_generated_token,
        "active_weight_bytes_per_generated_token": active_weight_bytes_per_generated_token,
        "active_bytes_scope": "packed_weight_payloads_per_complete_request_generated_token",
        "actual_read_bytes_per_token": null,
        "actual_read_bytes_status": "NOT_MEASURED_NO_METAL_MEMORY_COUNTER",
        "transient_bytes_per_token": null,
        "model_open_count": 1,
        "weight_upload_count": 1,
        "metrics": {
            "generation_wall_ns": wall_ns,
            "complete_wall_ns": complete_wall_ns,
            "complete_wall_ns_per_generated_token": complete_wall_ns_per_generated_token,
            "gpu_ns": complete_gpu_ns,
            "gpu_ns_per_generated_token": gpu_ns_per_generated_token,
            "wall_minus_gpu_ns": complete_gpu_ns.map(|value| complete_wall_ns.saturating_sub(value)),
            "dispatches": dispatches,
            "dispatches_per_generated_token": dispatches_per_generated_token,
            "resident_weight_bytes": result.resident_weight_bytes,
            "workspace_resident_bytes": result.workspace_resident_bytes,
            "active_bytes_per_token": active_weight_bytes_per_generated_token,
            "active_weight_bytes_per_generated_token": active_weight_bytes_per_generated_token,
            "active_weight_bytes_total": active_weight_bytes_total,
            "active_bytes_scope": "packed_weight_payloads_per_complete_request_generated_token",
            "actual_read_bytes_per_token": null,
            "actual_read_bytes_status": "NOT_MEASURED_NO_METAL_MEMORY_COUNTER",
            "transient_bytes_per_token": null,
            "generated_tokens": generated_count,
            "prefill": {
                "steps": result.prompt_len,
                "wall_ns": result.prefill_wall_ns,
                "gpu_ns": prefill_gpu_ns,
                "dispatches": prefill_dispatches,
                "wall_ns_per_step": result.prefill_wall_ns as f64 / result.prompt_len.max(1) as f64,
            },
            "decode": {
                "steps": result.decode_steps,
                "generated_tokens_after_prefill": result.decode_steps,
                "wall_ns": result.decode_wall_ns,
                "gpu_ns": decode_gpu_ns,
                "dispatches": decode_dispatches,
                "wall_ns_per_step": result.decode_wall_ns as f64 / result.decode_steps.max(1) as f64,
            },
            "step_trace": {
                "wall_ns": result.wall_ns_per_step,
                "gpu_ns": result.gpu_ns,
                "wait_ns": result.wait_ns,
                "encode_ns": result.encode_ns,
                "submit_ns": result.submit_ns,
                "dispatches": result.dispatches,
                "active_weight_bytes": result.active_weight_bytes,
            },
            "kernel_genome": {
                "trace_enabled": env::var("HAWKING_TRACE_DISPATCH").ok().as_deref() == Some("1"),
                "histogram": kernel_histogram,
                "exact_when_trace_enabled": true,
            },
            "capability": {
                "resident": true,
                "fallbacks": result.fallbacks,
                "dense_w_materialized": result.dense_w_materialized,
                "weights_loaded_once": true,
                "complete_token_accounting": true,
            },
        },
    }))
}

#[cfg(not(target_os = "macos"))]
fn run_resident(_args: Args) -> Result<(), String> {
    Err("qwen38 native resident is Metal-only".to_owned())
}

fn main() {
    let args = parse_args();
    if let Err(error) = run_resident(args) {
        fail(error);
    }
}
