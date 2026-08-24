//! One resident Qwen3.8 body, N isolated sessions, real WorkUnit prompts.
//!
//! Loads one catalog (q4 incumbent or sealed parent). Does not load a second
//! 27B. Does not write the artifact. Emits per-WorkUnit generations + walls
//! so the Python verifier can score accepted WUs separately from completed.

use serde::Deserialize;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::time::Instant;

fn usage() -> &'static str {
    "usage: ascension_qwen38_production_bench --artifact-root DIR --tokenizer PATH \
         --workunits FILE --out FILE [--fusion off|parent] [--max-new-tokens N] \
         [--max-seq-len N] [--concurrencies 1,2,4] [--topologies concurrent,sequential] \
         [--probe-c8]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_production_bench: {message}");
    process::exit(2);
}

#[derive(Clone, Debug, Deserialize)]
struct WorkUnitIn {
    id: String,
    prompt: String,
}

#[derive(Clone, Debug, Deserialize)]
struct WorkUnitFile {
    workunits: Vec<WorkUnitIn>,
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    workunits: PathBuf,
    out: PathBuf,
    fusion: String,
    max_new_tokens: usize,
    max_seq_len: usize,
    concurrencies: Vec<usize>,
    topologies: Vec<String>,
    probe_c8: bool,
}

fn parse_csv_usize(s: &str) -> Vec<usize> {
    s.split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<usize>().unwrap_or_else(|_| fail("--concurrencies")))
        .collect()
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut workunits = None;
    let mut out = None;
    let mut fusion = "off".to_owned();
    let mut max_new_tokens = 192usize;
    let mut max_seq_len = 512usize;
    let mut concurrencies = vec![1usize, 2, 4];
    let mut topologies = vec!["sequential".to_owned(), "concurrent".to_owned()];
    let mut probe_c8 = false;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--workunits" => {
                workunits = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            "--fusion" => fusion = args.next().unwrap_or_else(|| fail(usage())),
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-new-tokens"));
            }
            "--max-seq-len" => {
                max_seq_len = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-seq-len"));
            }
            "--concurrencies" => {
                concurrencies = parse_csv_usize(&args.next().unwrap_or_else(|| fail(usage())));
            }
            "--topologies" => {
                topologies = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--probe-c8" => probe_c8 = true,
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    if concurrencies.is_empty() {
        fail("need at least one concurrency");
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail("--artifact-root")),
        tokenizer: tokenizer.unwrap_or_else(|| fail("--tokenizer")),
        workunits: workunits.unwrap_or_else(|| fail("--workunits")),
        out: out.unwrap_or_else(|| fail("--out")),
        fusion,
        max_new_tokens,
        max_seq_len,
        concurrencies,
        topologies,
        probe_c8,
    }
}

fn write_json(path: &Path, body: &Value) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, serde_json::to_vec_pretty(body).expect("json")).unwrap_or_else(|e| fail(e));
    eprintln!("wrote {}", path.display());
}

fn metal_limits() -> Value {
    #[cfg(target_os = "macos")]
    {
        use metal::Device;
        if let Some(d) = Device::system_default() {
            return json!({
                "current_allocated_size": d.current_allocated_size(),
                "recommended_max_working_set_size": d.recommended_max_working_set_size(),
                "has_unified_memory": d.has_unified_memory(),
            });
        }
    }
    json!({"absent": "no Metal default device"})
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("production bench driver is Metal-only");
}

#[cfg(target_os = "macos")]
fn result_json(
    wu_id: &str,
    session_index: usize,
    start_offset_ns: u64,
    result: &hawking_core::model::qwen38_hybrid_decode::Qwen38GenerateResult,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
) -> Value {
    use hawking_core::model::qwen38_hybrid_decode::Qwen38GenerateResult;
    let text = result
        .decode_new(tokenizer)
        .unwrap_or_else(|e| format!("<<decode_new failed: {e}>>"));
    let new_tokens = result.new_tokens().to_vec();
    let prompt_len = result.prompt_len;
    let decode_latencies: Vec<u64> = if result.wall_ns_per_step.len() > prompt_len {
        result.wall_ns_per_step[prompt_len..].to_vec()
    } else {
        Vec::new()
    };
    let last_disp = result.dispatches.last().copied();
    json!({
        "id": wu_id,
        "session_index": session_index,
        "start_offset_ns": start_offset_ns,
        "generated_text": text,
        "new_token_ids": new_tokens,
        "n_new_tokens": new_tokens.len(),
        "prompt_len": prompt_len,
        "wall_ns": result.wall_ns,
        "prefill_wall_ns": result.prefill_wall_ns,
        "decode_wall_ns": result.decode_wall_ns,
        "first_step_wall_ns": result.first_step_wall_ns,
        "decode_steps": result.decode_steps,
        "fallbacks": result.fallbacks,
        "dispatches_last_step": last_disp,
        "median_gpu_ns_per_token": Qwen38GenerateResult::median_gpu_ns_per_token(result),
        "steady_decode_wall_ns_per_token": result.steady_decode_wall_ns_per_token(),
        "decode_step_wall_ns": decode_latencies,
        "ttft_exclusive_ns": result.prefill_wall_ns,
        "ttft_from_batch_start_ns": start_offset_ns.saturating_add(result.prefill_wall_ns),
    })
}

#[cfg(target_os = "macos")]
fn apply_fusion_all(
    sessions: &mut [hawking_core::model::qwen38_hybrid_decode::Qwen38HybridDecodeSession],
    fusion: &str,
) {
    use hawking_core::model::qwen38_hybrid_decode::Qwen38MlpFusion;
    let (mlp, gqa, dn) = match fusion {
        "parent" | "swiglu" | "fused" => (Qwen38MlpFusion::GateUpSwiglu, true, true),
        "off" | "q4" | "incumbent" => (Qwen38MlpFusion::Off, false, false),
        other => fail(format!("unknown --fusion {other}")),
    };
    for session in sessions.iter_mut() {
        session.apply_fusion(mlp, gqa, dn);
    }
}

#[cfg(target_os = "macos")]
fn run_sequential(
    sessions: &mut [hawking_core::model::qwen38_hybrid_decode::Qwen38HybridDecodeSession],
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    wus: &[WorkUnitIn],
    prompt_ids: &[Vec<u32>],
    max_new: usize,
) -> Value {
    use hawking_core::model::qwen38_hybrid_decode::generate_greedy;
    let c = sessions.len().max(1);
    let t0 = Instant::now();
    let mut rows = Vec::new();
    for (i, wu) in wus.iter().enumerate() {
        let slot = i % c;
        let start = t0.elapsed().as_nanos() as u64;
        eprintln!(
            "sequential c={c} wu {}/{} {} -> session {slot}",
            i + 1,
            wus.len(),
            wu.id
        );
        let result = generate_greedy(&mut sessions[slot], &prompt_ids[i], max_new)
            .unwrap_or_else(|e| fail(e));
        rows.push(result_json(&wu.id, slot, start, &result, tokenizer));
    }
    json!({
        "topology": "sequential_per_session",
        "sessions": c,
        "wall_ns": t0.elapsed().as_nanos() as u64,
        "workunits": rows,
    })
}

#[cfg(target_os = "macos")]
fn run_concurrent(
    sessions: &mut [hawking_core::model::qwen38_hybrid_decode::Qwen38HybridDecodeSession],
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    wus: &[WorkUnitIn],
    prompt_ids: &[Vec<u32>],
    max_new: usize,
) -> Value {
    use hawking_core::model::qwen38_hybrid_decode::{generate_greedy, generate_greedy_parallel};
    let c = sessions.len().max(1);
    let t0 = Instant::now();
    let mut rows = Vec::new();
    let mut wave = 0usize;
    let mut i = 0usize;
    while i < wus.len() {
        let n = (wus.len() - i).min(c);
        let start = t0.elapsed().as_nanos() as u64;
        eprintln!(
            "concurrent c={c} wave {wave} wus {}..{}",
            i,
            i + n
        );
        if n == 1 {
            let result = generate_greedy(&mut sessions[0], &prompt_ids[i], max_new)
                .unwrap_or_else(|e| fail(e));
            rows.push(result_json(&wus[i].id, 0, start, &result, tokenizer));
        } else {
            let chunk_ids: Vec<Vec<u32>> = prompt_ids[i..i + n].to_vec();
            let results =
                generate_greedy_parallel(&mut sessions[..n], &chunk_ids, max_new)
                    .unwrap_or_else(|e| fail(e));
            for (k, result) in results.iter().enumerate() {
                rows.push(result_json(
                    &wus[i + k].id,
                    k,
                    start,
                    result,
                    tokenizer,
                ));
            }
        }
        i += n;
        wave += 1;
    }
    json!({
        "topology": "concurrent_independent",
        "sessions": c,
        "wall_ns": t0.elapsed().as_nanos() as u64,
        "waves": wave,
        "workunits": rows,
    })
}

#[cfg(target_os = "macos")]
fn main() {
    use hawking_core::model::qwen38_host_admission::{host_memory_snapshot, process_rss_bytes};
    use hawking_core::model::qwen38_hybrid_decode::{
        load_qwen38_tokenizer, qwen38_fused_dispatches_per_token, qwen38_workspace_bytes,
        render_qwen38_user_chat, Qwen38HybridDecodeSession, Qwen38HybridWeights, Qwen38MlpFusion,
    };

    let args = parse_args();
    if args.max_new_tokens < 8 {
        fail("--max-new-tokens must be >= 8 (WorkUnits need room to close </think>)");
    }
    let spec: WorkUnitFile = serde_json::from_str(
        &fs::read_to_string(&args.workunits).unwrap_or_else(|e| fail(e)),
    )
    .unwrap_or_else(|e| fail(format!("workunits json: {e}")));
    if spec.workunits.is_empty() {
        fail("workunits file is empty");
    }

    let pid = process::id();
    let host_before = host_memory_snapshot().ok();
    let rss_before = process_rss_bytes(pid).ok();
    let metal_before = metal_limits();

    eprintln!(
        "prodbench load {} fusion={}",
        args.artifact_root.display(),
        args.fusion
    );
    let load_started = Instant::now();
    let weights = Arc::new(
        Qwen38HybridWeights::load(&args.artifact_root).unwrap_or_else(|e| fail(e)),
    );
    let load_ns = load_started.elapsed().as_nanos() as u64;
    let rss_after_load = process_rss_bytes(pid).ok();
    let metal_after_load = metal_limits();
    let body_bytes = weights.resident_bytes();
    let workspace = qwen38_workspace_bytes(args.max_seq_len).unwrap_or_else(|e| fail(e));

    let mut want = *args.concurrencies.iter().max().unwrap_or(&1);
    if args.probe_c8 {
        want = want.max(8);
    }
    let mut sessions = Vec::with_capacity(want);
    let mut rss_after_attach = Vec::new();
    let mut metal_after_attach = Vec::new();
    for i in 0..want {
        let session = Qwen38HybridDecodeSession::attach(Arc::clone(&weights), args.max_seq_len)
            .unwrap_or_else(|e| fail(e));
        if !sessions.is_empty() && !session.shares_weights_with(&sessions[0]) {
            fail("attached session does not share the resident weight Arc");
        }
        sessions.push(session);
        rss_after_attach.push(process_rss_bytes(pid).ok());
        metal_after_attach.push(metal_limits());
        eprintln!("attached session {i}/{}", want);
    }
    apply_fusion_all(&mut sessions, &args.fusion);
    let weights_ptr_shared = sessions.len() <= 1
        || sessions
            .windows(2)
            .all(|w| w[1].shares_weights_with(&w[0]));

    let tokenizer = load_qwen38_tokenizer(&args.tokenizer).unwrap_or_else(|e| fail(e));
    let rendered: Vec<String> = spec
        .workunits
        .iter()
        .map(|w| render_qwen38_user_chat(&w.prompt))
        .collect();
    let prompt_ids: Vec<Vec<u32>> = rendered
        .iter()
        .map(|p| tokenizer.encode(p, false).unwrap_or_else(|e| fail(e)))
        .collect();
    for (i, ids) in prompt_ids.iter().enumerate() {
        if ids.len() + args.max_new_tokens > args.max_seq_len {
            fail(format!(
                "workunit {} prompt {} + new {} exceeds max-seq-len {}",
                spec.workunits[i].id,
                ids.len(),
                args.max_new_tokens,
                args.max_seq_len
            ));
        }
    }

    let theoretical = if args.fusion == "parent" || args.fusion == "swiglu" || args.fusion == "fused"
    {
        qwen38_fused_dispatches_per_token(Qwen38MlpFusion::GateUpSwiglu, true, true)
    } else {
        qwen38_fused_dispatches_per_token(Qwen38MlpFusion::Off, false, false)
    };

    let mut cells = Vec::new();
    for topo in &args.topologies {
        for &c in &args.concurrencies {
            if c == 0 || c > sessions.len() {
                fail(format!("concurrency {c} is not attachable"));
            }
            apply_fusion_all(&mut sessions[..c], &args.fusion);
            let cell = match topo.as_str() {
                "sequential" | "sequential_per_session" => run_sequential(
                    &mut sessions[..c],
                    &tokenizer,
                    &spec.workunits,
                    &prompt_ids,
                    args.max_new_tokens,
                ),
                "concurrent" | "concurrent_independent" => run_concurrent(
                    &mut sessions[..c],
                    &tokenizer,
                    &spec.workunits,
                    &prompt_ids,
                    args.max_new_tokens,
                ),
                other => fail(format!("unknown topology {other}")),
            };
            let mut wrapped = cell;
            wrapped["concurrency"] = json!(c);
            wrapped["rss_after_cell_bytes"] = json!(process_rss_bytes(pid).ok());
            wrapped["metal_after_cell"] = metal_limits();
            cells.push(wrapped);
        }
    }

    let c8_probe = if args.probe_c8 && sessions.len() >= 8 {
        json!({
            "attached": true,
            "rss_bytes": rss_after_attach.get(7).cloned(),
            "metal": metal_after_attach.get(7).cloned(),
            "ran_workunits": args.concurrencies.contains(&8),
        })
    } else {
        json!({"attached": false})
    };

    let host_after = host_memory_snapshot().ok();
    let rss_final = process_rss_bytes(pid).ok();
    let metal_final = metal_limits();
    let fusion_label = sessions.first().map(|s| s.mlp_fusion.as_str());
    let body = json!({
        "schema": "hawking.headless.production_bench.raw.v1",
        "pid": pid,
        "artifact_root": args.artifact_root,
        "fusion": args.fusion,
        "mlp_fusion": fusion_label,
        "fuse_gqa_qkv": sessions.first().map(|s| s.fuse_gqa_qkv),
        "fuse_dn_inproj": sessions.first().map(|s| s.fuse_dn_inproj),
        "theoretical_dispatches": theoretical,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "workunit_ids": spec.workunits.iter().map(|w| w.id.clone()).collect::<Vec<_>>(),
        "prompt_token_counts": prompt_ids.iter().map(|p| p.len()).collect::<Vec<_>>(),
        "did_not_load_second_27b": true,
        "weight_loads": 1,
        "process_count": 1,
        "attached_sessions": sessions.len(),
        "weights_ptr_shared": weights_ptr_shared,
        "resident_weight_bytes": body_bytes,
        "weight_tensors": {
            "q4": weights.q4_tensor_count(),
            "f32": weights.f32_tensor_count(),
            "mixed": weights.mixed_tensor_count(),
        },
        "workspace": workspace,
        "load_ns": load_ns,
        "rss_before_bytes": rss_before,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_each_attach_bytes": rss_after_attach,
        "rss_final_bytes": rss_final,
        "metal_before": metal_before,
        "metal_after_load": metal_after_load,
        "metal_after_each_attach": metal_after_attach,
        "metal_final": metal_final,
        "host_before": host_before,
        "host_after": host_after,
        "c8_probe": c8_probe,
        "cells": cells,
    });
    write_json(&args.out, &body);
}
