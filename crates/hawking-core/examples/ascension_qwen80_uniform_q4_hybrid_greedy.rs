#![recursion_limit = "256"]
//! Multi-token greedy decode of Qwen3-Coder-Next through the hybrid token
//! graph, bound to the sealed uniform-Q4 group-64 catalog.
//!
//! Reports prefill, first-token latency, and steady-state tok/s separately.
//! This is a VELOCITY BASELINE at complete_physical_bpw 4.259241 — it is not
//! BASE_TRUE_TPS, HCLI, coherence, or tournament evidence.
//!
//! ```text
//! cargo run --release -p hawking-core --example ascension_qwen80_uniform_q4_hybrid_greedy -- \
//!   --prompt "Say hi." --max-new-tokens 4 \
//!   --out receipts/QWEN80_UNIFORM_Q4_VELOCITY_BASELINE.json
//! ```

use hawking_core::model::device_residency::{
    persistent_address_table_enabled, residency_budget_bytes,
};
use hawking_core::model::qwen80_complete_runtime::qwen80_assert_native_operator_composition_complete;
use hawking_core::model::qwen80_device_expert_table::{
    qwen80_q4_address_geometry, QWEN80_EXPERT_TRIPLET_PAYLOAD_BYTES,
};
use hawking_core::model::qwen80_uniform_q4_hybrid_decode::{
    discover_qwen80_tokenizer, discover_qwen80_uniform_q4_root, generate_greedy,
    load_qwen80_tokenizer, probe_qwen80_expert_first_touch_io, qwen80_default_tokenizer_path,
    qwen80_expert_nocopy_enabled, qwen80_expert_prefetch_enabled, qwen80_payload_rehash_enabled,
    render_qwen80_source_user_chat, Qwen80UniformQ4HybridDecodeSession,
    Qwen80UniformQ4StreamingCatalog, QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW,
    QWEN80_UNIFORM_Q4_EXPECTED_MANIFEST_SEAL, QWEN80_UNIFORM_Q4_EXPECTED_TERMINAL_SEAL,
    QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS,
};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

struct Arguments {
    artifact_root: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    prompt: String,
    raw_prompt: bool,
    max_new_tokens: usize,
    max_seq_len: usize,
    out: Option<PathBuf>,
    ledger: Option<PathBuf>,
    first_touch_probe: Option<usize>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_uniform_q4_hybrid_greedy \
        [--artifact-root DIR] [--tokenizer PATH] \
        [--prompt TEXT] [--raw-prompt] \
        [--max-new-tokens N] [--max-seq-len N] \
        [--out RECEIPT.json] [--ledger LEDGER.json] \
        [--first-touch-probe N]"
}

fn parse_args() -> Result<Arguments, String> {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "Say hi.".to_owned();
    let mut raw_prompt = false;
    let mut max_new_tokens = 4usize;
    let mut max_seq_len = 64usize;
    let mut out = None;
    let mut ledger = None;
    let mut first_touch_probe = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage().to_owned())?,
                ));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage().to_owned())?,
                ));
            }
            "--prompt" => {
                prompt = args.next().ok_or_else(|| usage().to_owned())?;
            }
            "--raw-prompt" => raw_prompt = true,
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .ok_or_else(|| usage().to_owned())?
                    .parse()
                    .map_err(|_| usage().to_owned())?;
            }
            "--max-seq-len" => {
                max_seq_len = args
                    .next()
                    .ok_or_else(|| usage().to_owned())?
                    .parse()
                    .map_err(|_| usage().to_owned())?;
            }
            "--out" => {
                out = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage().to_owned())?,
                ));
            }
            "--ledger" => {
                ledger = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage().to_owned())?,
                ));
            }
            "--first-touch-probe" => {
                first_touch_probe = Some(
                    args.next()
                        .ok_or_else(|| usage().to_owned())?
                        .parse()
                        .map_err(|_| usage().to_owned())?,
                );
            }
            "--help" | "-h" => return Err(usage().to_owned()),
            other => return Err(format!("unknown flag {other}; {}", usage())),
        }
    }
    Ok(Arguments {
        artifact_root,
        tokenizer,
        prompt,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
        out,
        ledger,
        first_touch_probe,
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    qwen80_assert_native_operator_composition_complete()
        .map_err(|error| format!("hybrid operator composition is incomplete: {error}"))?;
    let args = parse_args()?;
    let root = args
        .artifact_root
        .or_else(discover_qwen80_uniform_q4_root)
        .ok_or_else(|| {
            "qwen80 uniform-q4 artifact root not found; pass --artifact-root".to_owned()
        })?;
    let tokenizer_path = args
        .tokenizer
        .or_else(discover_qwen80_tokenizer)
        .unwrap_or_else(qwen80_default_tokenizer_path);
    let rendered = if args.raw_prompt {
        args.prompt.clone()
    } else {
        render_qwen80_source_user_chat(&args.prompt)
    };

    eprintln!("opening streaming catalog at {}", root.display());
    let mut catalog = Qwen80UniformQ4StreamingCatalog::open(&root).map_err(|e| e.to_string())?;
    if catalog.tensor_count() != 74_391 {
        return Err(format!(
            "catalog tensor count {} != 74391",
            catalog.tensor_count()
        ));
    }
    if catalog.manifest_seal_sha256 != QWEN80_UNIFORM_Q4_EXPECTED_MANIFEST_SEAL {
        eprintln!(
            "warning: manifest seal {} differs from sealed d4a140ab…",
            catalog.manifest_seal_sha256
        );
    }
    if catalog.terminal_seal_sha256.as_deref() != Some(QWEN80_UNIFORM_Q4_EXPECTED_TERMINAL_SEAL) {
        eprintln!(
            "warning: terminal seal {:?} differs from sealed b84e2d53…",
            catalog.terminal_seal_sha256
        );
    }
    eprintln!(
        "catalog tensors={} complete_physical_bpw={:.6} claim={}",
        catalog.tensor_count(),
        catalog.complete_physical_bpw,
        QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS
    );

    if let Some(n) = args.first_touch_probe {
        let admit = catalog.admit_session().map_err(|e| e.to_string())?;
        eprintln!(
            "session_admit seal={} stated={} sample_rehashed={} admit_ms={:.1} production_seals={}",
            admit.session_seal_sha256,
            admit.tensors_stated,
            admit.sample_rehashed,
            admit.admit_ns as f64 / 1e6,
            admit.production_seals_checked
        );
        let probe = probe_qwen80_expert_first_touch_io(&catalog, n).map_err(|e| e.to_string())?;
        let per = probe.compared_payloads.max(1) as f64;
        println!(
            "first_touch_probe n_experts={} compared={} bytes={} catalog_read_ms={:.3} sha256_ms={:.3} mmap_ms={:.3}",
            probe.n_experts,
            probe.compared_payloads,
            probe.bytes_read,
            probe.catalog_read_ns as f64 / 1e6,
            probe.sha256_ns as f64 / 1e6,
            probe.mmap_ns as f64 / 1e6
        );
        println!(
            "first_touch_probe_per_payload_ns catalog_read={:.0} sha256={:.0} mmap={:.0}",
            probe.catalog_read_ns as f64 / per,
            probe.sha256_ns as f64 / per,
            probe.mmap_ns as f64 / per
        );
        println!(
            "first_touch_probe_per_expert_ns catalog_read={:.0} sha256={:.0} mmap={:.0} (3 projs)",
            3.0 * probe.catalog_read_ns as f64 / per,
            3.0 * probe.sha256_ns as f64 / per,
            3.0 * probe.mmap_ns as f64 / per
        );
        if let Some(out) = args.out {
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let receipt = json!({
                "schema": "hawking.ascent.q80_expert_first_touch_io_probe.v1",
                "lane": "q80-expert-first-touch",
                "measurement_label": "DIRTY_ENGINEERING",
                "session_admit": {
                    "session_seal_sha256": admit.session_seal_sha256,
                    "tensors_stated": admit.tensors_stated,
                    "sample_rehashed": admit.sample_rehashed,
                    "admit_ns": admit.admit_ns,
                    "production_seals_checked": admit.production_seals_checked,
                },
                "probe": {
                    "n_experts": probe.n_experts,
                    "compared_payloads": probe.compared_payloads,
                    "bytes_read": probe.bytes_read,
                    "catalog_read_ns": probe.catalog_read_ns,
                    "sha256_ns": probe.sha256_ns,
                    "mmap_ns": probe.mmap_ns,
                    "catalog_read_ns_per_payload": probe.catalog_read_ns as f64 / per,
                    "sha256_ns_per_payload": probe.sha256_ns as f64 / per,
                    "mmap_ns_per_payload": probe.mmap_ns as f64 / per,
                    "catalog_read_ns_per_expert": 3.0 * probe.catalog_read_ns as f64 / per,
                    "sha256_ns_per_expert": 3.0 * probe.sha256_ns as f64 / per,
                    "mmap_ns_per_expert": 3.0 * probe.mmap_ns as f64 / per,
                    "bit_identical_mmap_vs_fresh_read": true,
                },
            });
            fs::write(&out, serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            eprintln!("wrote {}", out.display());
        }
        return Ok(());
    }

    let tokenizer = load_qwen80_tokenizer(&tokenizer_path).map_err(|e| e.to_string())?;
    let mut session = Qwen80UniformQ4HybridDecodeSession::new(catalog, args.max_seq_len)
        .map_err(|e| e.to_string())?;
    if args.ledger.is_some() {
        session.token_ns.enable();
    }
    eprintln!(
        "running greedy decode prompt_chars={} max_new_tokens={}",
        rendered.len(),
        args.max_new_tokens
    );
    let result = generate_greedy(&mut session, &tokenizer, &rendered, args.max_new_tokens)
        .map_err(|e| e.to_string())?;

    println!("prompt_token_ids={:?}", result.prompt_token_ids);
    println!("generated_token_ids={:?}", result.generated_token_ids);
    println!("generated_text={:?}", result.generated_text);
    println!("prefill_secs={:.6}", result.prefill_secs);
    println!(
        "first_token_latency_secs={:.6}",
        result.first_token_latency_secs
    );
    println!("decode_secs={:.6}", result.decode_secs);
    println!(
        "steady_state_tokens={} steady_state_decode_secs={:.6} steady_state_tok_s={:.6}",
        result.steady_state_tokens, result.steady_state_decode_secs, result.steady_state_tok_s
    );
    println!("peak_rss_bytes={}", result.peak_rss_bytes);
    println!(
        "fallback_count={} (matvec={} embed={} vec={} act={} expert_bind={} sample={})",
        result.fallbacks.total(),
        result.fallbacks.host_q4_matvec,
        result.fallbacks.host_q4_embedding_gather,
        result.fallbacks.host_q4_vector_decode,
        result.fallbacks.host_activation,
        result.fallbacks.host_expert_payload_bind,
        result.fallbacks.host_sample
    );
    println!(
        "native_q4_dispatches matvec={} embed={} decode_vector={} table_builds={} table_waves={} table_dispatches={} upload_hits={} upload_misses={} evictions={} slot_patches={} resident_slots={} resident_bytes={}",
        result.native.q4_matvec_dispatches,
        result.native.q4_embedding_dispatches,
        result.native.q4_decode_vector_dispatches,
        result.native.expert_table_layer_builds,
        result.native.expert_table_waves,
        result.native.expert_table_matvec_dispatches,
        result.native.expert_upload_hits,
        result.native.expert_upload_misses,
        result.native.expert_residency_evictions,
        result.native.expert_table_slot_patches,
        result.native.expert_resident_slots,
        result.native.expert_resident_bytes
    );
    println!(
        "stage_secs embed={:.4} deltanet={:.4} gqa={:.4} moe_norm_router={:.4} moe_shared={:.4} moe_table_build={:.4} moe_routed={:.4} moe_combine={:.4} terminal={:.4} q4_matvec={:.4} host_expert_bind={:.4}",
        result.stages.embed_secs,
        result.stages.deltanet_secs,
        result.stages.gqa_secs,
        result.stages.moe_norm_router_secs,
        result.stages.moe_shared_secs,
        result.stages.moe_table_build_secs,
        result.stages.moe_routed_secs,
        result.stages.moe_combine_secs,
        result.stages.terminal_secs,
        result.stages.q4_matvec_secs,
        result.stages.host_expert_bind_secs
    );
    let fam = &result.family_gpu;
    println!(
        "family_gpu overlap={} mode={} serial_mixer={} component_matvec={} deltanet_cbs={} deltanet_disp={} deltanet_gpu_ms={:.3} deltanet_wait_ms={:.3} deltanet_mixed={} gqa_cbs={} gqa_disp={} gqa_gpu_ms={:.3} gqa_wait_ms={:.3} gqa_mixed={} prefix_cbs={} prefix_gpu_ms={:.3} suffix_cbs={} suffix_gpu_ms={:.3} suffix_wait_ms={:.3} suffix_combine_disp={} fused_cbs={} fused_gpu_ms={:.3} gpu_gap_ms={:.3} gpu_gap_edges={} weight_bytes={}",
        fam.overlap_cbs,
        fam.overlap_mode,
        fam.serial_mixer,
        fam.component_matvec_kernel,
        fam.deltanet_mixer_cbs,
        fam.deltanet_mixer_dispatches,
        fam.deltanet_mixer_gpu_ns as f64 / 1e6,
        fam.deltanet_mixer_wait_ns as f64 / 1e6,
        fam.deltanet_mixer_mixed_with_moe_prefix,
        fam.gqa_mixer_cbs,
        fam.gqa_mixer_dispatches,
        fam.gqa_mixer_gpu_ns as f64 / 1e6,
        fam.gqa_mixer_wait_ns as f64 / 1e6,
        fam.gqa_mixer_mixed_with_moe_prefix,
        fam.moe_prefix_cbs,
        fam.moe_prefix_gpu_ns as f64 / 1e6,
        fam.moe_suffix_cbs,
        fam.moe_suffix_gpu_ns as f64 / 1e6,
        fam.moe_suffix_wait_ns as f64 / 1e6,
        fam.moe_suffix_combine_dispatches,
        fam.fused_layer_cbs,
        fam.fused_layer_gpu_ns as f64 / 1e6,
        fam.gpu_gap_ns as f64 / 1e6,
        fam.gpu_gap_edges,
        fam.bytes_read_weight
    );
    println!(
        "activation_class_secs conv={:.6} recurrent={:.6} gqa_rms={:.6} gqa_rope={:.6} (device-path values are encode-wall, not GPU)",
        result.stages.activation.deltanet_conv_secs,
        result.stages.activation.deltanet_recurrent_secs,
        result.stages.activation.gqa_input_layernorm_secs,
        result.stages.activation.gqa_norm_rope_secs
    );
    println!(
        "moe_table_split upload_miss={:.6} entries_fill={:.6} buffer_write={:.6} resource_clone={:.6} lease={:.6}",
        result.stages.moe_table_upload_miss_secs,
        result.stages.moe_table_entries_fill_secs,
        result.stages.moe_table_buffer_write_secs,
        result.stages.moe_table_resource_clone_secs,
        result.stages.moe_table_lease_secs
    );
    let split = session.catalog().first_touch_split();
    println!(
        "first_touch catalog_read_ms={:.3} sha256_ms={:.3} metal_copy_ms={:.3} metal_nocopy_ms={:.3} mmap_ms={:.3} hashed={} mmapped={} copy_binds={} nocopy_binds={} prefetch_ms={:.3} prefetch_uploads={}",
        split.catalog_read_ns as f64 / 1e6,
        split.sha256_ns as f64 / 1e6,
        split.metal_copy_ns as f64 / 1e6,
        split.metal_nocopy_ns as f64 / 1e6,
        split.mmap_ns as f64 / 1e6,
        split.payloads_hashed,
        split.payloads_mmapped,
        split.copy_binds,
        split.nocopy_binds,
        split.prefetch_ns as f64 / 1e6,
        split.prefetch_uploads
    );
    println!(
        "session_seal admitted={} seal={:?} admit_ms={:.1} stated={} sample_rehashed={} nocopy={} rehash={} prefetch={}",
        session.catalog().session_seal_sha256.is_some(),
        session.catalog().session_seal_sha256,
        session.catalog().session_admit.admit_ns as f64 / 1e6,
        session.catalog().session_admit.tensors_stated,
        session.catalog().session_admit.sample_rehashed,
        qwen80_expert_nocopy_enabled(),
        qwen80_payload_rehash_enabled(),
        qwen80_expert_prefetch_enabled()
    );
    let geometry = qwen80_q4_address_geometry();
    let limits = session.device_memory_limits();
    println!(
        "residency persistent={} budget_bytes={} table_bytes={} q4_full_payload_bytes={} scaled_1_5_payload_bytes={} triplet_payload_bytes={}",
        persistent_address_table_enabled(),
        session
            .residency_budget_bytes()
            .unwrap_or_else(residency_budget_bytes),
        geometry.table_bytes(),
        geometry.full_payload_bytes(),
        geometry.scaled_payload_bytes(QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW, 1.5),
        QWEN80_EXPERT_TRIPLET_PAYLOAD_BYTES
    );
    if let Some(limits) = limits {
        println!(
            "device_limits max_buffer_length={} recommended_max_working_set_size={} current_allocated_size={} has_unified_memory={}",
            limits.max_buffer_length,
            limits.recommended_max_working_set_size,
            limits.current_allocated_size,
            limits.has_unified_memory
        );
    }
    println!(
        "complete_physical_bpw={:.6} claim={} metal_q4_matvec_used={}",
        result.complete_physical_bpw, result.claim, result.metal_q4_matvec_used
    );
    if let Some(error) = &result.metal_error {
        println!("metal_error={error}");
    }

    if let Some(out) = args.out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let upload_total = result
            .native
            .expert_upload_hits
            .saturating_add(result.native.expert_upload_misses);
        let hit_rate = if upload_total == 0 {
            0.0
        } else {
            result.native.expert_upload_hits as f64 / upload_total as f64
        };
        let receipt = json!({
            "schema": "hawking.ascension.qwen80_uniform_q4_velocity_baseline.v1",
            "status": QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS,
            "claim_boundary": {
                "base_true_tps": false,
                "coherence": false,
                "capability": false,
                "hcli": false,
                "restart": false,
                "tournament": false,
                "complete_physical_bpw": result.complete_physical_bpw,
                "complete_physical_bpw_reported": QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW,
                "density_gate_bpw_max": 1.5,
                "artifact_cannot_satisfy_1_5_bpw_gate": true,
            },
            "artifact": {
                "root": root,
                "tensor_count": 74391,
                "manifest_seal_sha256": session.catalog().manifest_seal_sha256,
                "terminal_seal_sha256": session.catalog().terminal_seal_sha256,
                "complete_physical_bpw": result.complete_physical_bpw,
                "mean_component_cosine": session.catalog().mean_component_cosine,
            },
            "prompt": result.prompt,
            "prompt_token_ids": result.prompt_token_ids,
            "generated_token_ids": result.generated_token_ids,
            "generated_text": result.generated_text,
            "timing": {
                "prefill_secs": result.prefill_secs,
                "first_token_latency_secs": result.first_token_latency_secs,
                "decode_secs": result.decode_secs,
                "steady_state_tokens": result.steady_state_tokens,
                "steady_state_decode_secs": result.steady_state_decode_secs,
                "steady_state_tok_s": result.steady_state_tok_s,
            },
            "resources": {
                "peak_rss_bytes": result.peak_rss_bytes,
                "streamed_rss_cap_bytes": 16u64 * 1024 * 1024 * 1024,
            },
            "execution": {
                "hybrid_token_graph": "embed + 48-layer mixer/MoE + terminal greedy",
                "weight_codec": "uniform_q4_group64",
                "metal_q4_matvec_used": result.metal_q4_matvec_used,
                "metal_error": result.metal_error,
                "native": {
                    "q4_matvec_dispatches": result.native.q4_matvec_dispatches,
                    "q4_embedding_dispatches": result.native.q4_embedding_dispatches,
                    "q4_decode_vector_dispatches": result.native.q4_decode_vector_dispatches,
                    "expert_table_layer_builds": result.native.expert_table_layer_builds,
                    "expert_table_waves": result.native.expert_table_waves,
                    "expert_table_matvec_dispatches": result.native.expert_table_matvec_dispatches,
                    "expert_upload_hits": result.native.expert_upload_hits,
                    "expert_upload_misses": result.native.expert_upload_misses,
                    "expert_residency_evictions": result.native.expert_residency_evictions,
                    "expert_table_slot_patches": result.native.expert_table_slot_patches,
                    "expert_resident_slots": result.native.expert_resident_slots,
                    "expert_resident_bytes": result.native.expert_resident_bytes,
                    "expert_nocopy_binds": result.native.expert_nocopy_binds,
                    "expert_copy_binds": result.native.expert_copy_binds,
                    "expert_prefetch_uploads": result.native.expert_prefetch_uploads,
                },
                "session_seal": {
                    "session_seal_sha256": session.catalog().session_seal_sha256,
                    "admit_ns": session.catalog().session_admit.admit_ns,
                    "tensors_stated": session.catalog().session_admit.tensors_stated,
                    "sample_rehashed": session.catalog().session_admit.sample_rehashed,
                    "production_seals_checked": session.catalog().session_admit.production_seals_checked,
                    "verified_once": [
                        "every catalogued file exists as a regular non-symlink with exact artifact_bytes",
                        "production manifest and terminal seals match the sealed constants",
                        "session seal binds manifest_seal || terminal_seal || ordered (name, bytes, declared_sha256)",
                        "sample payloads rehashed against catalog SHA"
                    ],
                    "still_guards_token_path": [
                        "mmap/read size must equal artifact_bytes",
                        "header geometry must match the catalog row",
                        "no-copy bind is fail-closed (16 KiB window + contents() alias)",
                        "read_payload_rehash and HAWKING_Q80_REHASH_PAYLOADS=1 remain"
                    ],
                    "nocopy": qwen80_expert_nocopy_enabled(),
                    "rehash": qwen80_payload_rehash_enabled(),
                    "prefetch": qwen80_expert_prefetch_enabled(),
                },
                "first_touch": {
                    "catalog_read_ns": split.catalog_read_ns,
                    "sha256_ns": split.sha256_ns,
                    "metal_copy_ns": split.metal_copy_ns,
                    "metal_nocopy_ns": split.metal_nocopy_ns,
                    "header_parse_ns": split.header_parse_ns,
                    "mmap_ns": split.mmap_ns,
                    "payloads_read": split.payloads_read,
                    "payloads_hashed": split.payloads_hashed,
                    "payloads_mmapped": split.payloads_mmapped,
                    "nocopy_binds": split.nocopy_binds,
                    "copy_binds": split.copy_binds,
                    "prefetch_uploads": split.prefetch_uploads,
                    "prefetch_ns": split.prefetch_ns,
                    "prefetch_hits": split.prefetch_hits,
                    "prefetch_misses": split.prefetch_misses,
                },
                "residency": {
                    "persistent_address_table": persistent_address_table_enabled(),
                    "budget_bytes": session.residency_budget_bytes().unwrap_or_else(residency_budget_bytes),
                    "table_bytes": geometry.table_bytes(),
                    "q4_full_payload_bytes": geometry.full_payload_bytes(),
                    "scaled_1_5_payload_bytes": geometry.scaled_payload_bytes(
                        QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW,
                        1.5,
                    ),
                    "triplet_payload_bytes": QWEN80_EXPERT_TRIPLET_PAYLOAD_BYTES,
                    "device_limits": limits.map(|lim| json!({
                        "max_buffer_length": lim.max_buffer_length,
                        "recommended_max_working_set_size": lim.recommended_max_working_set_size,
                        "current_allocated_size": lim.current_allocated_size,
                        "has_unified_memory": lim.has_unified_memory,
                    })),
                    "hit_rate": hit_rate,
                },
                "stages": {
                    "embed_secs": result.stages.embed_secs,
                    "deltanet_secs": result.stages.deltanet_secs,
                    "gqa_secs": result.stages.gqa_secs,
                    "moe_norm_router_secs": result.stages.moe_norm_router_secs,
                    "moe_shared_secs": result.stages.moe_shared_secs,
                    "moe_table_build_secs": result.stages.moe_table_build_secs,
                    "moe_table_upload_miss_secs": result.stages.moe_table_upload_miss_secs,
                    "moe_table_entries_fill_secs": result.stages.moe_table_entries_fill_secs,
                    "moe_table_buffer_write_secs": result.stages.moe_table_buffer_write_secs,
                    "moe_table_resource_clone_secs": result.stages.moe_table_resource_clone_secs,
                    "moe_table_lease_secs": result.stages.moe_table_lease_secs,
                    "first_touch_catalog_read_secs": result.stages.first_touch_catalog_read_secs,
                    "first_touch_sha256_secs": result.stages.first_touch_sha256_secs,
                    "first_touch_metal_copy_secs": result.stages.first_touch_metal_copy_secs,
                    "first_touch_metal_nocopy_secs": result.stages.first_touch_metal_nocopy_secs,
                    "prefetch_secs": result.stages.prefetch_secs,
                    "moe_routed_secs": result.stages.moe_routed_secs,
                    "moe_combine_secs": result.stages.moe_combine_secs,
                    "terminal_secs": result.stages.terminal_secs,
                    "q4_matvec_secs": result.stages.q4_matvec_secs,
                    "host_expert_bind_secs": result.stages.host_expert_bind_secs,
                    "activation": {
                        "deltanet_conv_secs": result.stages.activation.deltanet_conv_secs,
                        "deltanet_recurrent_secs": result.stages.activation.deltanet_recurrent_secs,
                        "gqa_input_layernorm_secs": result.stages.activation.gqa_input_layernorm_secs,
                        "gqa_norm_rope_secs": result.stages.activation.gqa_norm_rope_secs,
                        "note": "On the device-activation path these are host encode-wall of the named kernels, not GPU time. They were previously unwritten (~0) because only the host fallback populated them."
                    },
                },
                "family_gpu": {
                    "overlap_cbs": result.family_gpu.overlap_cbs,
                    "overlap_mode": result.family_gpu.overlap_mode,
                    "serial_mixer": result.family_gpu.serial_mixer,
                    "component_matvec_kernel": result.family_gpu.component_matvec_kernel,
                    "deltanet_mixer_cbs": result.family_gpu.deltanet_mixer_cbs,
                    "deltanet_mixer_dispatches": result.family_gpu.deltanet_mixer_dispatches,
                    "deltanet_mixer_gpu_ns": result.family_gpu.deltanet_mixer_gpu_ns,
                    "deltanet_mixer_wait_ns": result.family_gpu.deltanet_mixer_wait_ns,
                    "deltanet_mixer_mixed_with_moe_prefix": result.family_gpu.deltanet_mixer_mixed_with_moe_prefix,
                    "gqa_mixer_cbs": result.family_gpu.gqa_mixer_cbs,
                    "gqa_mixer_dispatches": result.family_gpu.gqa_mixer_dispatches,
                    "gqa_mixer_gpu_ns": result.family_gpu.gqa_mixer_gpu_ns,
                    "gqa_mixer_wait_ns": result.family_gpu.gqa_mixer_wait_ns,
                    "gqa_mixer_mixed_with_moe_prefix": result.family_gpu.gqa_mixer_mixed_with_moe_prefix,
                    "moe_prefix_cbs": result.family_gpu.moe_prefix_cbs,
                    "moe_prefix_gpu_ns": result.family_gpu.moe_prefix_gpu_ns,
                    "moe_suffix_cbs": result.family_gpu.moe_suffix_cbs,
                    "moe_suffix_gpu_ns": result.family_gpu.moe_suffix_gpu_ns,
                    "moe_suffix_wait_ns": result.family_gpu.moe_suffix_wait_ns,
                    "moe_suffix_combine_dispatches": result.family_gpu.moe_suffix_combine_dispatches,
                    "fused_layer_cbs": result.family_gpu.fused_layer_cbs,
                    "fused_layer_dispatches": result.family_gpu.fused_layer_dispatches,
                    "fused_layer_gpu_ns": result.family_gpu.fused_layer_gpu_ns,
                    "fused_layer_wait_ns": result.family_gpu.fused_layer_wait_ns,
                    "gpu_gap_ns": result.family_gpu.gpu_gap_ns,
                    "gpu_gap_edges": result.family_gpu.gpu_gap_edges,
                    "bytes_read_weight": result.family_gpu.bytes_read_weight,
                    "gpu_timestamp_authority": "MTLCommandBuffer GPUEndTime-GPUStartTime after wait; overlapped CBs resolve after a later same-queue wait",
                },
                "fallbacks": {
                    "total": result.fallbacks.total(),
                    "host_q4_matvec": result.fallbacks.host_q4_matvec,
                    "host_q4_embedding_gather": result.fallbacks.host_q4_embedding_gather,
                    "host_q4_vector_decode": result.fallbacks.host_q4_vector_decode,
                    "host_activation": result.fallbacks.host_activation,
                    "host_expert_payload_bind": result.fallbacks.host_expert_payload_bind,
                    "host_sample": result.fallbacks.host_sample,
                    "note": if result.native.expert_table_waves > 0 {
                        "512-way device expert table is live; host_expert_payload_bind counts remaining host binds".to_string()
                    } else {
                        "expert gather is a host fallback; the composed graph has no 512-way device gather".to_string()
                    },
                },
            },
        });
        let pretty = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
        fs::write(&out, pretty).map_err(|e| e.to_string())?;
        eprintln!("wrote {}", out.display());
    }
    if let Some(ledger_path) = args.ledger {
        if let Some(parent) = ledger_path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let ledger = session.token_ns.finish_report();
        let pretty = serde_json::to_string_pretty(&ledger).map_err(|e| e.to_string())?;
        fs::write(&ledger_path, pretty).map_err(|e| e.to_string())?;
        eprintln!("wrote {}", ledger_path.display());
        let commit = std::env::var("HAWKING_GIT_COMMIT").unwrap_or_else(|_| "unknown".to_owned());
        let unified = hawking_core::token_ns::from_q80_ledger(
            &ledger,
            &hawking_core::token_ns::EmitMeta::new(commit, ledger_path.display().to_string()),
        );
        let token_ns_path = ledger_path.with_file_name("TOKEN_NS.json");
        let unified_pretty = serde_json::to_string_pretty(&unified).map_err(|e| e.to_string())?;
        fs::write(&token_ns_path, unified_pretty).map_err(|e| e.to_string())?;
        eprintln!("wrote {}", token_ns_path.display());
        if let Some(diag) = &ledger.diagnosis {
            eprintln!(
                "token_ns_ledger verdict={} wall_ms={:.1} gpu_ms={:.1} wait_ms={:.1} cbs={:.0} disp={:.0} weight_gib={:.3} implied_gb_s_gpu={:?}",
                diag.verdict,
                diag.wall_ns as f64 / 1e6,
                diag.gpu_execution_ns as f64 / 1e6,
                diag.cpu_wait_ns as f64 / 1e6,
                diag.command_buffers_per_token,
                diag.dispatches_per_token,
                diag.weight_bytes_per_token as f64 / (1024.0 * 1024.0 * 1024.0),
                diag.implied_gb_s_from_gpu
            );
            eprintln!("rationale={}", diag.rationale);
        }
    }
    Ok(())
}
