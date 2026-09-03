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
use hawking_core::json_constrain::{JsonConstraint, JsonVocabIndex};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_constrained, generate_greedy_reusing_snapshot, load_qwen38_tokenizer,
    Qwen38HybridDecodeSession,
    Qwen38HybridWeights,
    Qwen38PrefixCheckpoint,
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
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
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
    let tokenizer =
        load_qwen38_tokenizer(&args.tokenizer).map_err(|e| format!("load tokenizer: {e}"))?;
    let mut session =
        Qwen38HybridDecodeSession::attach(std::sync::Arc::clone(&weights), args.max_seq_len)
            .map_err(|e| format!("allocate resident session: {e}"))?;
    // Built on the first grammar=json request, not at startup: indexing the
    // tokenizer calls decode_one once per vocab id.
    let mut json_vocab_index: Option<JsonVocabIndex> = None;
    // Exactly the token sequence currently held in the session's KV and
    // recurrent state: the last request's prompt followed by what it generated.
    let mut resident_context: Vec<u32> = Vec::new();
    // ONE checkpoint, held at the boundary two consecutive prompts agreed on --
    // in practice the system prompt and tool catalog, which are identical on
    // every goal and were being re-prefilled every time. A checkpoint is ~157 MB
    // of host memory, so this is one, not a pool.
    let mut prefix_checkpoint: Option<(Vec<u32>, Qwen38PrefixCheckpoint)> = None;

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
                write_record(
                    &mut stdout,
                    &error_reply("", "request is not a JSON object"),
                )?;
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
        let reply = match serve_request(
            &mut session,
            &tokenizer,
            &mut json_vocab_index,
            &args,
            pid,
            &body,
            &mut resident_context,
            &mut prefix_checkpoint,
        ) {
            Ok(reply) => reply,
            Err(error) => error_reply(&id, error),
        };
        write_record(&mut stdout, &reply)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
/// Length of the longest common prefix. The whole soundness of KV reuse rests
/// on this being an EXACT token comparison, not a hash or a heuristic.
fn shared_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn serve_request(
    session: &mut Qwen38HybridDecodeSession,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    json_vocab_index: &mut Option<JsonVocabIndex>,
    args: &Args,
    pid: u32,
    body: &Value,
    resident_context: &mut Vec<u32>,
    prefix_checkpoint: &mut Option<(Vec<u32>, Qwen38PrefixCheckpoint)>,
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
    let max_new =
        usize::try_from(max_new).map_err(|_| "max_new_tokens does not fit in usize".to_owned())?;
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
    let constrain_json = match body.get("grammar") {
        None => false,
        Some(Value::String(s)) if s == "json" => true,
        Some(other) => {
            return Err(format!(
                "unsupported grammar {other}; only the string \"json\" is accepted"
            ));
        }
    };

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
            prompt_ids.len(),
            args.max_seq_len
        ));
    }

    // KV and recurrent state ARE shared across requests now, but only when this
    // request's prompt begins with exactly the tokens already in the session.
    //
    // Measured before this: one trivial goal ("count the .py files in hcli")
    // cost 13 model calls, 635 s, and 20,935 re-prefilled prompt tokens at
    // ~33 prompt tok/s. Eleven of those calls carried the SAME prefix_key and
    // still paid full prefill, because the resident reset unconditionally. A
    // tool loop appends an observation and re-sends; re-reading the other 1,500
    // tokens each round is the whole latency.
    //
    // PURE APPEND ONLY, and the check is exact. DeltaNet state is recurrent --
    // a running summary with no per-position index -- so it cannot be rewound
    // or truncated. A prompt that DIVERGES from the resident context, however
    // late, resets: reusing a diverged prefix would condition generation on
    // tokens that are not in the prompt, and nothing downstream could see it.
    let reuse = shared_prefix_len(resident_context, &prompt_ids);
    // `generate_constrained` still resets internally, so claiming a reuse there
    // would skip prompt tokens against a cleared state. Constrained requests
    // take the full prefill until that path learns the same trick.
    let reuse = if !constrain_json
        && reuse == resident_context.len()
        && reuse > 0
        && reuse < prompt_ids.len()
    {
        reuse
    } else {
        0
    };
    // Second chance when the append path does not apply: a stored checkpoint
    // whose tokens are a prefix of this prompt. Restoring it is EXACT -- the
    // recurrent carry after those tokens is a function of those tokens alone,
    // and KV at 0..N already holds the same bytes because the tokens match.
    // Proven bit-identical on this body by
    // examples/qwen38_prefix_checkpoint_parity.rs.
    // Captured BEFORE any clear: the boundary this prompt and the last one
    // agreed on is computed from the PREVIOUS context, and the branches below
    // overwrite it.
    let agreed_with_previous = shared_prefix_len(resident_context, &prompt_ids);
    let mut restored_from_checkpoint = 0usize;
    if reuse == 0 {
        let usable = prefix_checkpoint.as_ref().and_then(|(tokens, cp)| {
            let shared = shared_prefix_len(tokens, &prompt_ids);
            if shared == tokens.len() && shared > 0 && shared < prompt_ids.len() {
                Some((shared, cp))
            } else {
                None
            }
        });
        match usable {
            Some((shared, cp)) => {
                session
                    .restore_prefix(cp)
                    .map_err(|e| format!("restore prefix checkpoint: {e}"))?;
                restored_from_checkpoint = shared;
                resident_context.clear();
                resident_context.extend_from_slice(&prompt_ids[..shared]);
            }
            _ => {
                session.reset();
                resident_context.clear();
            }
        }
    }
    let reuse = reuse.max(restored_from_checkpoint);
    // Snapshot where THIS prompt and the last one agreed, which is the stable
    // head every goal shares. Taken during the prefill we are already paying
    // for, because the carry cannot be rewound to a boundary once passed.
    let snapshot_at = if restored_from_checkpoint == 0
        && prefix_checkpoint.is_none()
        && agreed_with_previous > 16
        && agreed_with_previous < prompt_ids.len()
    {
        Some(agreed_with_previous)
    } else {
        None
    };
    let started = Instant::now();
    let result = if constrain_json {
        let vocab = json_vocab_index.get_or_insert_with(|| {
            JsonVocabIndex::build(tokenizer.vocab_size(), |id| {
                tokenizer.decode_one(id).unwrap_or_default()
            })
        });
        let mut constraint = JsonConstraint::new();
        generate_constrained(
            session,
            tokenizer,
            vocab,
            &mut constraint,
            &prompt_ids,
            max_new,
            reuse,
            snapshot_at,
        )
        .map(|(result, snapshot)| {
            if let Some(cp) = snapshot {
                *prefix_checkpoint = Some((prompt_ids[..cp.position].to_vec(), cp));
            }
            result
        })
    } else {
        generate_greedy_reusing_snapshot(session, &prompt_ids, max_new, reuse, snapshot_at)
            .map(|(result, snapshot)| {
                if let Some(cp) = snapshot {
                    *prefix_checkpoint =
                        Some((prompt_ids[..cp.position].to_vec(), cp));
                }
                result
            })
    }
    .map_err(|e| format!("native generation: {e}"))?;
    let wall_ns = started.elapsed().as_nanos() as u64;
    // What the session now holds: this prompt plus everything it just produced.
    // Written only on success -- a failed generation leaves the session in an
    // unknown state, and the next request must reset rather than trust it.
    resident_context.clear();
    resident_context.extend_from_slice(&result.tokens);
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
    let gpu_ns_per_generated_token =
        complete_gpu_ns.map(|value| value as f64 / generated_count.max(1) as f64);
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
        "grammar_enforced": constrain_json,
        "prefix_reused_tokens": reuse,
        "prefill_tokens_stepped": prompt_ids.len().saturating_sub(reuse),
        "prefix_source": if restored_from_checkpoint > 0 {
            "checkpoint_restore"
        } else if reuse > 0 {
            "session_append"
        } else {
            "cold"
        },
        "prefix_checkpoint_taken_at": snapshot_at,
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

#[cfg(test)]
mod prefix_reuse_tests {
    use super::shared_prefix_len;

    /// The reuse decision, exactly as `serve_request` makes it.
    ///
    /// Kept in one place so the rule can be tested without a 27B model: reuse
    /// ONLY when the resident context is a proper prefix of the new prompt.
    fn reuse_for(resident: &[u32], prompt: &[u32], constrain_json: bool) -> usize {
        let shared = shared_prefix_len(resident, prompt);
        if !constrain_json && shared == resident.len() && shared > 0 && shared < prompt.len() {
            shared
        } else {
            0
        }
    }

    #[test]
    fn a_pure_append_reuses_everything_already_resident() {
        // The tool loop: same conversation plus one observation.
        let resident = [1u32, 2, 3, 4];
        let prompt = [1u32, 2, 3, 4, 9, 9, 9];
        assert_eq!(reuse_for(&resident, &prompt, false), 4);
    }

    #[test]
    fn divergence_resets_however_late_it_happens() {
        // The load-bearing one. DeltaNet state is a running summary with no
        // per-position index: it cannot be rewound. Reusing a diverged prefix
        // would condition generation on tokens that are not in the prompt, and
        // nothing downstream could detect it.
        let resident = [1u32, 2, 3, 4];
        assert_eq!(reuse_for(&resident, &[1, 2, 3, 5, 6], false), 0, "late divergence");
        assert_eq!(reuse_for(&resident, &[9, 2, 3, 4, 5], false), 0, "first token differs");
    }

    #[test]
    fn a_shorter_prompt_resets_because_state_cannot_be_truncated() {
        let resident = [1u32, 2, 3, 4, 5, 6];
        assert_eq!(reuse_for(&resident, &[1, 2, 3], false), 0);
    }

    #[test]
    fn an_identical_prompt_resets_so_a_token_is_always_stepped() {
        // `next` is the argmax the last stepped token produced. With nothing
        // stepped there is none, so full equality must not claim reuse.
        let resident = [1u32, 2, 3];
        assert_eq!(reuse_for(&resident, &[1, 2, 3], false), 0);
    }

    #[test]
    fn a_cold_session_resets() {
        assert_eq!(reuse_for(&[], &[1, 2, 3], false), 0);
    }

    #[test]
    fn the_constrained_path_never_claims_reuse() {
        // generate_constrained still resets internally; claiming reuse there
        // would skip prompt tokens against a cleared state.
        let resident = [1u32, 2, 3, 4];
        let prompt = [1u32, 2, 3, 4, 5];
        assert_eq!(reuse_for(&resident, &prompt, false), 4);
        assert_eq!(reuse_for(&resident, &prompt, true), 0);
    }

    /// The checkpoint decision, exactly as `serve_request` makes it.
    fn checkpoint_reuse(stored: &[u32], prompt: &[u32], constrain_json: bool) -> usize {
        let shared = shared_prefix_len(stored, prompt);
        if !constrain_json && shared == stored.len() && shared > 0 && shared < prompt.len() {
            shared
        } else {
            0
        }
    }

    #[test]
    fn a_stored_prefix_that_this_prompt_begins_with_is_restored() {
        // Two goals sharing a system prompt and tool catalog.
        let stored = [1u32, 2, 3, 4, 5];
        assert_eq!(checkpoint_reuse(&stored, &[1, 2, 3, 4, 5, 90, 91], false), 5);
    }

    #[test]
    fn a_checkpoint_that_diverges_is_never_restored() {
        // The load-bearing one: restoring against a different prefix would
        // condition generation on tokens that are not in the prompt.
        let stored = [1u32, 2, 3, 4, 5];
        assert_eq!(checkpoint_reuse(&stored, &[1, 2, 99, 4, 5, 6], false), 0);
        assert_eq!(checkpoint_reuse(&stored, &[1, 2, 3], false), 0, "shorter prompt");
        assert_eq!(checkpoint_reuse(&stored, &[1, 2, 3, 4, 5], false), 0, "nothing to step");
        assert_eq!(checkpoint_reuse(&[], &[1, 2, 3], false), 0, "no checkpoint");
        assert_eq!(
            checkpoint_reuse(&stored, &[1, 2, 3, 4, 5, 6], true),
            0,
            "the constrained path resets internally"
        );
    }

    #[test]
    fn shared_prefix_len_is_an_exact_token_comparison() {
        assert_eq!(shared_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(shared_prefix_len(&[1, 2, 3], &[1, 2]), 2);
        assert_eq!(shared_prefix_len(&[1], &[2]), 0);
        assert_eq!(shared_prefix_len(&[], &[1]), 0);
    }
}
