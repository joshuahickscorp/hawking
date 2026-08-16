//! DRAM-row locality: address-stream model, execution-order pack, isolated A/B.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core --example dram_row_locality
//! Characterise (no GPU):
//!   workspace/ops/build/rust/release-fast/examples/dram_row_locality \
//!     --out receipts/ascent-2026-08-16/dram-row-locality.json
//! Measure (GPU mutex required):
//!   ./tools/gpu_lane_lock.sh dram-row-locality \
//!     workspace/ops/build/rust/release-fast/examples/dram_row_locality \
//!     --measure --out receipts/ascent-2026-08-16/dram-row-locality.json

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("dram_row_locality requires macOS").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::dram_row_locality::{
        analyze_stream, binary_weight_from_interleaved, deinterleave_fp4_blocks,
        deinterleave_q4_groups, dsv4f_expert_colocated_stream, dsv4f_expert_six_chunk_stream,
        dsv4f_fp4_interleaved_stream, dsv4f_fp4_split_stream, greedy_coreoute_order,
        interleave_binary_group, interleave_fp4_blocks, interleave_q4_groups,
        pack_triplet_blob, pair_mean_abs_distance, q4_weight_from_interleaved, q4_weight_from_split,
        q80_binary_interleaved_stream, q80_binary_split_stream, q80_expert_colocated_stream,
        q80_expert_six_file_stream, q80_q4_interleaved_stream, q80_q4_split_stream, rank_streams,
        read_npy_i16, read_npy_i32, scale_stream_to_token, unpack_triplet_blob, StreamStats,
        BINARY_INTERLEAVED_STRIDE_G128, CEILING_GBPS, DRAM_ROW_BYTES_MODEL, DSV4F_FP4_BLOCK,
        DSV4F_HIDDEN, DSV4F_INTER, DSV4F_LAYERS, DSV4F_TOP_K, DSV4F_W1_PACKED, DSV4F_W1_SCALES,
        DSV4F_W2_PACKED, DSV4F_W2_SCALES, FP4_INTERLEAVED_STRIDE, PAGE_16K, Q4_INTERLEAVED_STRIDE,
        Q80_BINARY_GROUP, Q80_EXPERTS, Q80_GATE_COLS, Q80_GATE_ROWS, Q80_LAYERS, Q80_TOP_K,
    };
    use hawking_core::model::qwen_complete_binary::{
        binary_group_matvec_f32, binary_group_weight, deterministic_input, deterministic_matrix,
        pack_binary_group, pack_uniform_q4_group64, parse_uniform_q4_header, UNIFORM_Q4_GROUP_SIZE,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};

    const SCHEMA: &str = "hawking.ascension.dram_row_locality.v1";
    const WARMUP: usize = 2;
    const PAIRS: usize = 3;
    const Q4_TOL: f32 = 0.0; // bit-identical association
    const BINARY_TOL: f32 = 2e-5;
    const DEFAULT_DSV4F: &str = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity";
    const DEFAULT_CAPTURE: &str = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-diagnostics/source-bf16-capture-n192-scale64/capture-index.v1";

    struct Args {
        out: PathBuf,
        measure: bool,
        dsv4f: Option<PathBuf>,
        capture_index: Option<PathBuf>,
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let mut out = PathBuf::from("receipts/ascent-2026-08-16/dram-row-locality.json");
        let mut measure = false;
        let mut dsv4f = Some(PathBuf::from(DEFAULT_DSV4F));
        let mut capture_index = Some(PathBuf::from(DEFAULT_CAPTURE));
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--measure" => measure = true,
                "--out" => {
                    out = PathBuf::from(args.next().ok_or("missing --out value")?);
                }
                "--dsv4f-artifact" => {
                    let v = args.next().ok_or("missing --dsv4f-artifact value")?;
                    dsv4f = if v == "none" {
                        None
                    } else {
                        Some(PathBuf::from(v))
                    };
                }
                "--capture-index" => {
                    let v = args.next().ok_or("missing --capture-index value")?;
                    capture_index = if v == "none" {
                        None
                    } else {
                        Some(PathBuf::from(v))
                    };
                }
                other => return Err(format!("unsupported option {other}").into()),
            }
        }
        Ok(Args {
            out,
            measure,
            dsv4f,
            capture_index,
        })
    }

    fn stats_json(s: &StreamStats) -> Value {
        json!({
            "name": s.name,
            "layout": s.layout,
            "touches": s.touches,
            "bytes": s.bytes,
            "sequential_runs": s.sequential_runs,
            "mean_run_bytes": s.mean_run_bytes,
            "p50_run_bytes": s.p50_run_bytes,
            "p90_run_bytes": s.p90_run_bytes,
            "max_run_bytes": s.max_run_bytes,
            "stride_ge_4k": s.stride_ge_4k,
            "stride_ge_row": s.stride_ge_row,
            "dram_row_transitions": s.dram_row_transitions,
            "unique_pages_4k": s.unique_pages_4k,
            "unique_pages_16k": s.unique_pages_16k,
            "unique_dram_rows": s.unique_dram_rows,
            "min_possible_dram_rows": s.min_possible_dram_rows,
            "scatter_ratio": s.scatter_ratio,
            "note": s.note,
            "derived_model_not_hardware_counter": true,
            "dram_row_bytes_model": DRAM_ROW_BYTES_MODEL,
        })
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        format!("{:x}", h.finalize())
    }

    fn q80_file_scatter() -> Value {
        let mut names = Vec::with_capacity((Q80_EXPERTS as usize) * 3);
        for e in 0..Q80_EXPERTS {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let tensor = format!("model.layers.0.mlp.experts.{e}.{proj}.weight");
                names.push((sha256_hex(tensor.as_bytes()), e, proj));
            }
        }
        names.sort_by(|a, b| a.0.cmp(&b.0));
        let mut pos = vec![[0usize; 3]; Q80_EXPERTS as usize];
        for (i, (_, e, proj)) in names.iter().enumerate() {
            let slot = match *proj {
                "gate_proj" => 0,
                "up_proj" => 1,
                _ => 2,
            };
            pos[*e as usize][slot] = i;
        }
        let mut spans = Vec::new();
        let mut consecutive = 0usize;
        for slots in &pos {
            let mut s = *slots;
            s.sort_unstable();
            spans.push(s[2] - s[0]);
            if s[1] == s[0] + 1 && s[2] == s[1] + 1 {
                consecutive += 1;
            }
        }
        spans.sort_unstable();
        json!({
            "layer": 0,
            "files_per_layer": names.len(),
            "naming": "sha256(tensor_name).hq80uq4 — not expert-number order",
            "experts_with_gate_up_down_consecutive_in_dir": consecutive,
            "triplet_dir_rank_span": {
                "min": spans.first().copied(),
                "median": spans.get(spans.len() / 2).copied(),
                "max": spans.last().copied(),
            },
            "verdict": "0 of 512 layer-0 experts have gate/up/down consecutive in the SHA-named tensors directory. Median span 787 of 1536 slots. Pack-time triplet blob is the locality lever; the live catalog is one file per organ."
        })
    }

    fn dsv4f_source_layout_snapshot() -> Value {
        // Measured 2026-08-16 from full-43-layer-stream.gravity/manifest.json
        // for layers.0.ffn.experts.{0,1,17,255}. Metadata only; no payload read.
        json!({
            "derived_from": "full-43-layer-stream.gravity manifest tensor descriptors",
            "source_shard_layer0_all_experts": "model-00002-of-00046.safetensors",
            "layer0_expert_tensors": 1536,
            "layer0_unique_gravity_chunks": 1536,
            "experts_with_all_six_in_one_chunk": 0,
            "source_safetensor_layout": {
                "scales_region": "early in the shard (expert 0 w1.scale @ 11647016; expert 0 w1.weight @ 345095720). Scales of an expert's three organs are adjacent. Weights of an expert are adjacent as w1,w2,w3 — not execution order w1,w3,w2.",
                "scale_to_weight_gap_bytes_expert0_w1": 345095720u64 - 11647016u64,
                "execution_order": ["w1/gate", "w3/up", "w2/down"],
                "source_weight_order": ["w1", "w2", "w3"],
            },
            "gravity_chunk_layout": {
                "kind": "one content-addressed SHA-named file per tensor",
                "chunk_count": 69837,
                "tensor_count": 69187,
                "effect": "source adjacency of w1/w2/w3 weights is destroyed. No-copy bind pins six unrelated mmap windows per expert."
            },
            "bytes_per_expert": {
                "w1_packed": DSV4F_W1_PACKED,
                "w1_scales": DSV4F_W1_SCALES,
                "w2_packed": DSV4F_W2_PACKED,
                "w2_scales": DSV4F_W2_SCALES,
                "triplet": DSV4F_W1_PACKED * 2 + DSV4F_W2_PACKED + DSV4F_W1_SCALES * 2 + DSV4F_W2_SCALES,
            },
            "streams_claimed": "COLD/streamed remainder and Q80. Hot DSV4F expert-cache hits remove DRAM traffic and are not counted here.",
        })
    }

    fn capture_coreoute(index: &Path) -> Result<Value, Box<dyn Error>> {
        let ids = read_npy_i32(&index.join("expert_ids.npy"))?;
        let layers = read_npy_i16(&index.join("layer.npy"))?;
        if ids.len() != layers.len() * 10 {
            return Err(format!(
                "capture index shape drift: ids={} layers={}",
                ids.len(),
                layers.len()
            )
            .into());
        }
        let n = Q80_EXPERTS as usize;
        let mut freq = vec![0u32; n];
        let mut pair = vec![0u32; n * n];
        let mut pair_list: Vec<(u32, u32, u32)> = Vec::new();
        // Accumulate across all layers into one matrix. Per-layer permutation
        // is permitted at pack time; one global order is the cheap evidence.
        for (row, layer_chunk) in ids.chunks_exact(10).enumerate() {
            let _layer = layers[row];
            let mut picked = [0u32; 10];
            for (i, &id) in layer_chunk.iter().enumerate() {
                if !(0..n as i32).contains(&id) {
                    return Err(format!("expert id {id} out of 0..{n}").into());
                }
                picked[i] = id as u32;
                freq[id as usize] = freq[id as usize].saturating_add(1);
            }
            for i in 0..10 {
                for j in (i + 1)..10 {
                    let a = picked[i].min(picked[j]) as usize;
                    let b = picked[i].max(picked[j]) as usize;
                    pair[a * n + b] = pair[a * n + b].saturating_add(1);
                    pair[b * n + a] = pair[b * n + a].saturating_add(1);
                }
            }
        }
        let mut never = 0usize;
        for &f in &freq {
            if f == 0 {
                never += 1;
            }
        }
        for a in 0..n {
            for b in (a + 1)..n {
                let c = pair[a * n + b];
                if c > 0 {
                    pair_list.push((a as u32, b as u32, c));
                }
            }
        }
        pair_list.sort_by(|x, y| y.2.cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
        let identity: Vec<u32> = (0..n as u32).collect();
        let greedy = greedy_coreoute_order(&freq, &pair, n);
        let mut inv = vec![0u32; n];
        for (pos, &id) in greedy.iter().enumerate() {
            inv[id as usize] = pos as u32;
        }
        let mean_id = pair_mean_abs_distance(&pair_list, &identity);
        let mean_perm = pair_mean_abs_distance(&pair_list, &inv);
        let top: Vec<Value> = pair_list
            .iter()
            .take(8)
            .map(|(a, b, c)| {
                json!({
                    "a": a,
                    "b": b,
                    "co_route_count": c,
                    "id_distance": a.abs_diff(*b),
                    "greedy_distance": inv[*a as usize].abs_diff(inv[*b as usize]),
                })
            })
            .collect();
        Ok(json!({
            "source": index.display().to_string(),
            "n_route_rows": layers.len(),
            "top_k": 10,
            "never_routed_experts_collapsed_across_layers": never,
            "note_never_routed": "Coverage receipt's 221 is per (layer,expert) pair, not collapsed expert ids. This pass collapses layer.",
            "mean_co_routed_id_distance": mean_id,
            "mean_co_routed_greedy_distance": mean_perm,
            "distance_reduction_factor": if mean_perm > 0.0 { mean_id / mean_perm } else { 0.0 },
            "top_pairs": top,
            "verdict": if mean_id > 0.0 && mean_perm + 8.0 < mean_id {
                "CHEAP EVIDENCE SUPPORTS a global co-route permutation inside a packed slab."
            } else if mean_perm + 1.0 < mean_id {
                "WEAK. Greedy order barely cuts mean |id_i-id_j|. Hot pairs can sit together in a slab; do not rewrite the live SHA-named catalog."
            } else {
                "Co-route placement does not beat numeric order on this capture."
            },
            "did_not_run_a_new_capture": true,
        }))
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    fn as_u8_u16(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn as_u8_f32(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn write_f32(buffer: &metal::Buffer, values: &[f32]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                buffer.contents() as *mut f32,
                values.len(),
            );
        }
    }

    fn read_f32(buffer: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, n).to_vec() }
    }

    fn gpu_ns(
        t: hawking_core::metal::MetalDispatchTiming,
        label: &str,
    ) -> Result<u64, Box<dyn Error>> {
        if let (Some(start), Some(end)) = (t.gpu_start_ns, t.gpu_end_ns) {
            if end > start {
                return Ok(end - start);
            }
        }
        let us = t.gpu_duration_us.ok_or_else(|| {
            format!("{label}: MTLCommandBuffer GPUStartTime/GPUEndTime unavailable")
        })?;
        Ok(us.saturating_mul(1000))
    }

    fn median(values: &[u64]) -> u64 {
        let mut s = values.to_vec();
        s.sort_unstable();
        s[s.len() / 2]
    }

    fn gbps(bytes: u64, ns: u64) -> f64 {
        if ns == 0 {
            0.0
        } else {
            bytes as f64 / ns as f64
        }
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn decode_e2m1(nibble: u8) -> f32 {
        let mag = match nibble & 0x07 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        if nibble & 0x08 != 0 {
            -mag
        } else {
            mag
        }
    }

    fn decode_e8m0(bits: u8) -> f32 {
        if bits == 0xff {
            0.0
        } else if bits == 0 {
            f32::from_bits(0x0040_0000)
        } else {
            f32::from_bits(u32::from(bits) << 23)
        }
    }

    fn decode_e4m3(bits: u8) -> f32 {
        let raw = u32::from(bits);
        let exponent = (raw >> 3) & 0x0f;
        let mantissa = raw & 0x07;
        if exponent == 0x0f && mantissa == 0x07 {
            return 0.0;
        }
        let magnitude = if exponent == 0 {
            mantissa as f32 * 0.001953125
        } else {
            f32::from_bits(((exponent + 120) << 23) | (mantissa << 20))
        };
        if raw & 0x80 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    fn fp4_cpu_oracle(
        packed: &[u8],
        scales: &[u8],
        quantized: &[u8],
        act_scales: &[u8],
        rows: usize,
        packed_cols: usize,
        scale_cols: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        for row in 0..rows {
            let mut acc = 0.0f32;
            for block in 0..scale_cols {
                let mut block_acc = 0.0f32;
                let start = block * DSV4F_FP4_BLOCK;
                for offset in 0..DSV4F_FP4_BLOCK {
                    let col = start + offset;
                    let p = packed[row * packed_cols + col / 2];
                    let nibble = if offset & 1 == 0 { p & 0x0f } else { p >> 4 };
                    block_acc += decode_e4m3(quantized[col]) * decode_e2m1(nibble);
                }
                let act_scale = decode_e8m0(act_scales[block / (128 / DSV4F_FP4_BLOCK)]);
                let w_scale = decode_e8m0(scales[row * scale_cols + block]);
                acc += block_acc * (act_scale * w_scale);
            }
            out[row] = acc;
        }
        out
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;

        // ── address-stream model (CPU, no GPU) ───────────────────────────
        let q80_bin_split = analyze_stream(
            "q80_mixed_gate_binary",
            "split_scales_then_signs",
            &q80_binary_split_stream(Q80_GATE_ROWS, Q80_GATE_COLS, Q80_BINARY_GROUP),
            "Current mixed-decode bind: two MTLBuffers. Per group the kernel reads 2B scale then 16B signs from a different allocation.",
        );
        let q80_bin_inter = analyze_stream(
            "q80_mixed_gate_binary",
            "interleaved_per_group",
            &q80_binary_interleaved_stream(Q80_GATE_ROWS, Q80_GATE_COLS, Q80_BINARY_GROUP),
            "Pack/bind-time: [fp16 scale | 16 sign bytes] per group of 128. One sequential run per row.",
        );
        let q80_q4_split = analyze_stream(
            "q80_q4_gate",
            "split_scales_then_codes",
            &q80_q4_split_stream(Q80_GATE_ROWS, Q80_GATE_COLS),
            "Live Q4 vehicle. HQ30UQ4 body is all scales then all codes; upload splits them into two buffers. Kernel alternates 2B + 32B across the pair.",
        );
        let q80_q4_inter = analyze_stream(
            "q80_q4_gate",
            "interleaved_per_group",
            &q80_q4_interleaved_stream(Q80_GATE_ROWS, Q80_GATE_COLS),
            "Execution-order group record [fp16 scale | 32 codes]. Same 34 bytes/group, one sequential run per row.",
        );
        let dsv_w1_split = analyze_stream(
            "dsv4f_fp4_w1",
            "split_packed_then_scales",
            &dsv4f_fp4_split_stream(DSV4F_INTER, DSV4F_HIDDEN / 2, DSV4F_HIDDEN / DSV4F_FP4_BLOCK),
            "dsv4f_worklist_fp4_matvec: per 32-weight block, 16 sequential packed bytes then 1 scale byte from a second buffer.",
        );
        let dsv_w1_inter = analyze_stream(
            "dsv4f_fp4_w1",
            "interleaved_per_block",
            &dsv4f_fp4_interleaved_stream(
                DSV4F_INTER,
                DSV4F_HIDDEN / 2,
                DSV4F_HIDDEN / DSV4F_FP4_BLOCK,
            ),
            "Pack-time [e8m0 | 16 packed] per block. Destroys no-copy if done at bind; belongs in a gravity re-pack.",
        );
        let dsv_six = analyze_stream(
            "dsv4f_expert_triplet",
            "six_sha_chunks",
            &dsv4f_expert_six_chunk_stream(),
            "Gravity content-addressed stream: one SHA file per tensor. Cold miss opens six unrelated mmap windows.",
        );
        let dsv_one = analyze_stream(
            "dsv4f_expert_triplet",
            "colocated_execution_order",
            &dsv4f_expert_colocated_stream(),
            "One blob, w1 then w3 then w2, each organ interleaved. Only the COLD remainder is in scope.",
        );
        let q80_six = analyze_stream(
            "q80_q4_expert_triplet",
            "six_sha_files",
            &q80_expert_six_file_stream(),
            "Live Q4 catalog: sha256(name).hq80uq4, one file per organ, codes and scales then split again at upload.",
        );
        let q80_one = analyze_stream(
            "q80_q4_expert_triplet",
            "colocated_execution_order",
            &q80_expert_colocated_stream(),
            "Pack-time triplet blob in gate/up/down order.",
        );

        let token_q4 = scale_stream_to_token(&q80_q4_split, Q80_LAYERS * Q80_TOP_K * 3);
        let token_dsv = scale_stream_to_token(&dsv_w1_split, DSV4F_LAYERS * DSV4F_TOP_K * 3);
        let ranked = rank_streams(&[
            token_q4.clone(),
            token_dsv.clone(),
            scale_stream_to_token(&q80_bin_split, Q80_LAYERS * Q80_TOP_K),
            scale_stream_to_token(&dsv_six, DSV4F_LAYERS * DSV4F_TOP_K),
            scale_stream_to_token(&q80_six, Q80_LAYERS * Q80_TOP_K),
        ]);

        let file_scatter = q80_file_scatter();
        let dsv_layout = dsv4f_source_layout_snapshot();
        let coreoute = match args.capture_index.as_ref() {
            Some(p) if p.join("expert_ids.npy").is_file() => Some(capture_coreoute(p)?),
            _ => None,
        };

        let mut live_dsv4f = Value::Null;
        if let Some(path) = args.dsv4f.as_ref() {
            if path.join("manifest.json").is_file() {
                match hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader::try_admit_from_artifact_index(
                    path,
                    hawking_core::gravity_deepseek_v4::DeepSeekV4VerifyMode::from_env()?,
                ) {
                    Ok(Some(reader)) => {
                        let key = hawking_core::gravity_deepseek_v4_expert_cache::ExpertBundleKey::new(0, 0);
                        match hawking_core::gravity_deepseek_v4_expert_cache::resolve_expert_bundle(&reader, key) {
                            Ok(desc) => {
                                let chunks: Vec<Value> = desc
                                    .operators
                                    .iter()
                                    .flat_map(|op| {
                                        op.source_chunk_paths.iter().map(|c| {
                                            json!({
                                                "tensor": c.tensor_name,
                                                "role": c.tensor_role,
                                                "chunk": c.chunk_relpath,
                                                "bytes": c.bytes,
                                            })
                                        })
                                    })
                                    .collect();
                                live_dsv4f = json!({
                                    "admitted": true,
                                    "layer": 0,
                                    "expert": 0,
                                    "payload_bytes": desc.payload_bytes,
                                    "source_chunk_read_count_per_fill": desc.source_chunk_read_count_per_fill,
                                    "chunks": chunks,
                                });
                            }
                            Err(e) => {
                                live_dsv4f = json!({"admitted": true, "resolve_error": e.to_string()});
                            }
                        }
                    }
                    Ok(None) => {
                        live_dsv4f = json!({
                            "admitted": false,
                            "reason": "artifact index missing/disabled; snapshot used instead"
                        });
                    }
                    Err(e) => {
                        live_dsv4f = json!({"admitted": false, "error": e.to_string()});
                    }
                }
            }
        }

        // ── value-preserving pack contract (CPU) ─────────────────────────
        let gate_w = deterministic_matrix(Q80_GATE_ROWS, Q80_GATE_COLS, 1);
        let gate_bin = pack_binary_group(&gate_w, Q80_GATE_ROWS, Q80_GATE_COLS, Q80_BINARY_GROUP)?;
        let gate_inter = interleave_binary_group(&gate_bin)?;
        let mut binary_bit_identical = true;
        for row in 0..Q80_GATE_ROWS {
            for col in 0..Q80_GATE_COLS {
                let a = binary_group_weight(&gate_bin, row, col);
                let b = binary_weight_from_interleaved(
                    &gate_inter,
                    Q80_GATE_COLS,
                    Q80_BINARY_GROUP,
                    row,
                    col,
                );
                if a.to_bits() != b.to_bits() {
                    binary_bit_identical = false;
                }
            }
        }
        let (q4_payload, q4_quality) =
            pack_uniform_q4_group64(&gate_w, &[Q80_GATE_ROWS, Q80_GATE_COLS])?;
        let q4_header = parse_uniform_q4_header(&q4_payload)?;
        let mut q4_scales = Vec::with_capacity(q4_header.groups);
        for g in 0..q4_header.groups {
            q4_scales.push(u16::from_le_bytes([
                q4_payload[q4_header.scale_offset + g * 2],
                q4_payload[q4_header.scale_offset + g * 2 + 1],
            ]));
        }
        let q4_codes = q4_payload[q4_header.sign_offset..].to_vec();
        let q4_inter = interleave_q4_groups(&q4_scales, &q4_codes)?;
        let (q4_s2, q4_c2) = deinterleave_q4_groups(&q4_inter)?;
        let mut q4_bit_identical = q4_s2 == q4_scales && q4_c2 == q4_codes;
        for row in 0..Q80_GATE_ROWS {
            for col in 0..Q80_GATE_COLS {
                let a = q4_weight_from_split(&q4_scales, &q4_codes, Q80_GATE_COLS, row, col);
                let b = q4_weight_from_interleaved(&q4_inter, Q80_GATE_COLS, row, col);
                if a.to_bits() != b.to_bits() {
                    q4_bit_identical = false;
                }
            }
        }
        let blob = pack_triplet_blob(&q4_inter, &q4_inter, &q4_inter);
        let (bg, bu, bd) = unpack_triplet_blob(&blob)?;
        let triplet_roundtrip = bg == q4_inter && bu == q4_inter && bd == q4_inter;
        if !binary_bit_identical || !q4_bit_identical || !triplet_roundtrip {
            return Err("CPU layout transform failed bit-identity".into());
        }

        let mut measure = json!({
            "ran": false,
            "reason": "pass --measure under tools/gpu_lane_lock.sh dram-row-locality"
        });
        let mut gpu_correctness = json!({});
        let mut paired = json!({});

        if args.measure {
            let ctx = hawking_core::metal::MetalContext::new()?;
            let x = deterministic_input(Q80_GATE_COLS);
            let x_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&x))?;
            let out_a = ctx.new_buffer_checked(Q80_GATE_ROWS * 4)?;
            let out_b = ctx.new_buffer_checked(Q80_GATE_ROWS * 4)?;
            let zeros = vec![0.0f32; Q80_GATE_ROWS];

            // Q4 A/B
            let q4_oracle: Vec<f32> = (0..Q80_GATE_ROWS)
                .map(|row| {
                    (0..Q80_GATE_COLS)
                        .map(|col| {
                            q4_weight_from_split(&q4_scales, &q4_codes, Q80_GATE_COLS, row, col)
                                * x[col]
                        })
                        .sum()
                })
                .collect();
            let q4_code_buf = ctx.new_buffer_with_bytes_checked(&q4_codes)?;
            let q4_scale_buf = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&q4_scales))?;
            let q4_inter_buf = ctx.new_buffer_with_bytes_checked(&q4_inter)?;
            write_f32(&out_a, &zeros);
            ctx.dispatch_threads(
                "qwen_uniform_q4_group64_matvec",
                (Q80_GATE_ROWS as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&q4_code_buf), 0);
                    enc.set_buffer(1, Some(&q4_scale_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&out_a), 0);
                    set_u32(enc, 4, Q80_GATE_ROWS as u32);
                    set_u32(enc, 5, Q80_GATE_COLS as u32);
                    set_u32(enc, 6, (Q80_GATE_COLS / UNIFORM_Q4_GROUP_SIZE) as u32);
                },
            )?;
            let q4_a_err = max_abs(&q4_oracle, &read_f32(&out_a, Q80_GATE_ROWS));
            write_f32(&out_b, &zeros);
            ctx.dispatch_threads(
                "qwen_uniform_q4_group64_matvec_interleaved",
                (Q80_GATE_ROWS as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&q4_inter_buf), 0);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(&out_b), 0);
                    set_u32(enc, 3, Q80_GATE_ROWS as u32);
                    set_u32(enc, 4, Q80_GATE_COLS as u32);
                    set_u32(enc, 5, (Q80_GATE_COLS / UNIFORM_Q4_GROUP_SIZE) as u32);
                },
            )?;
            let q4_b_err = max_abs(&q4_oracle, &read_f32(&out_b, Q80_GATE_ROWS));
            let q4_ab = max_abs(&read_f32(&out_a, Q80_GATE_ROWS), &read_f32(&out_b, Q80_GATE_ROWS));
            if q4_a_err > BINARY_TOL || q4_b_err > BINARY_TOL {
                return Err(format!("q4 matvec vs oracle a={q4_a_err} b={q4_b_err}").into());
            }
            if q4_ab > 0.0 {
                return Err(format!("q4 split vs interleaved max_abs={q4_ab} (decoded weights are bit-identical; matvec must match)").into());
            }

            // Binary A/B
            let bin_oracle = binary_group_matvec_f32(&gate_bin, &x)?;
            let bin_sign_buf = ctx.new_buffer_with_bytes_checked(&gate_bin.signs)?;
            let bin_scale_buf = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&gate_bin.scales_f16))?;
            let bin_inter_buf = ctx.new_buffer_with_bytes_checked(&gate_inter)?;
            write_f32(&out_a, &zeros);
            ctx.dispatch_threads(
                "q80_binary_group_matvec",
                (Q80_GATE_ROWS as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&bin_sign_buf), 0);
                    enc.set_buffer(1, Some(&bin_scale_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&out_a), 0);
                    set_u32(enc, 4, Q80_GATE_ROWS as u32);
                    set_u32(enc, 5, Q80_GATE_COLS as u32);
                    set_u32(enc, 6, Q80_BINARY_GROUP as u32);
                    set_u32(enc, 7, gate_bin.groups_per_row as u32);
                },
            )?;
            let bin_a_err = max_abs(&bin_oracle, &read_f32(&out_a, Q80_GATE_ROWS));
            write_f32(&out_b, &zeros);
            ctx.dispatch_threads(
                "q80_binary_group_matvec_interleaved",
                (Q80_GATE_ROWS as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&bin_inter_buf), 0);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(&out_b), 0);
                    set_u32(enc, 3, Q80_GATE_ROWS as u32);
                    set_u32(enc, 4, Q80_GATE_COLS as u32);
                    set_u32(enc, 5, Q80_BINARY_GROUP as u32);
                    set_u32(enc, 6, gate_bin.groups_per_row as u32);
                },
            )?;
            let bin_b_err = max_abs(&bin_oracle, &read_f32(&out_b, Q80_GATE_ROWS));
            if bin_a_err > BINARY_TOL || bin_b_err > BINARY_TOL {
                return Err(format!("binary matvec drift a={bin_a_err} b={bin_b_err}").into());
            }

            // FP4 A/B on a reduced but legal geometry: 256 rows × 512 logical K.
            let fp4_rows = 256usize;
            let fp4_logical = 512usize;
            let fp4_packed_cols = fp4_logical / 2;
            let fp4_scale_cols = fp4_logical / DSV4F_FP4_BLOCK;
            let mut fp4_packed = vec![0u8; fp4_rows * fp4_packed_cols];
            let mut fp4_scales = vec![0u8; fp4_rows * fp4_scale_cols];
            for (i, b) in fp4_packed.iter_mut().enumerate() {
                *b = (i.wrapping_mul(41) + 7) as u8;
            }
            for (i, b) in fp4_scales.iter_mut().enumerate() {
                *b = 0x7f_u8.wrapping_add((i as u8) & 3); // near-1.0 E8M0, avoid 0xff
            }
            let mut quant = vec![0u8; fp4_logical];
            let act_s = vec![0x7f_u8; fp4_logical / 128];
            for (i, b) in quant.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(3);
            }
            let fp4_oracle = fp4_cpu_oracle(
                &fp4_packed,
                &fp4_scales,
                &quant,
                &act_s,
                fp4_rows,
                fp4_packed_cols,
                fp4_scale_cols,
            );
            let fp4_inter =
                interleave_fp4_blocks(&fp4_packed, &fp4_scales, fp4_rows, fp4_packed_cols, fp4_scale_cols)?;
            let (fp4_p2, fp4_s2) =
                deinterleave_fp4_blocks(&fp4_inter, fp4_rows, fp4_packed_cols, fp4_scale_cols)?;
            if fp4_p2 != fp4_packed || fp4_s2 != fp4_scales {
                return Err("fp4 interleave lost bytes".into());
            }
            let fp4_p_buf = ctx.new_buffer_with_bytes_checked(&fp4_packed)?;
            let fp4_s_buf = ctx.new_buffer_with_bytes_checked(&fp4_scales)?;
            let fp4_i_buf = ctx.new_buffer_with_bytes_checked(&fp4_inter)?;
            let quant_buf = ctx.new_buffer_with_bytes_checked(&quant)?;
            let act_buf = ctx.new_buffer_with_bytes_checked(&act_s)?;
            let fp4_out_a = ctx.new_buffer_checked(fp4_rows * 4)?;
            let fp4_out_b = ctx.new_buffer_checked(fp4_rows * 4)?;
            let fp4_zeros = vec![0.0f32; fp4_rows];
            write_f32(&fp4_out_a, &fp4_zeros);
            ctx.dispatch_threads(
                "dsv4f_fp4_matvec_split",
                (fp4_rows as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&fp4_p_buf), 0);
                    enc.set_buffer(1, Some(&fp4_s_buf), 0);
                    enc.set_buffer(2, Some(&quant_buf), 0);
                    enc.set_buffer(3, Some(&act_buf), 0);
                    enc.set_buffer(4, Some(&fp4_out_a), 0);
                    set_u32(enc, 5, fp4_rows as u32);
                    set_u32(enc, 6, fp4_packed_cols as u32);
                    set_u32(enc, 7, fp4_scale_cols as u32);
                },
            )?;
            let fp4_a_err = max_abs(&fp4_oracle, &read_f32(&fp4_out_a, fp4_rows));
            write_f32(&fp4_out_b, &fp4_zeros);
            ctx.dispatch_threads(
                "dsv4f_fp4_matvec_interleaved",
                (fp4_rows as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&fp4_i_buf), 0);
                    enc.set_buffer(1, Some(&quant_buf), 0);
                    enc.set_buffer(2, Some(&act_buf), 0);
                    enc.set_buffer(3, Some(&fp4_out_b), 0);
                    set_u32(enc, 4, fp4_rows as u32);
                    set_u32(enc, 5, fp4_packed_cols as u32);
                    set_u32(enc, 6, fp4_scale_cols as u32);
                },
            )?;
            let fp4_b_err = max_abs(&fp4_oracle, &read_f32(&fp4_out_b, fp4_rows));
            let fp4_ab = max_abs(
                &read_f32(&fp4_out_a, fp4_rows),
                &read_f32(&fp4_out_b, fp4_rows),
            );
            if fp4_a_err > 2e-4 || fp4_b_err > 2e-4 {
                return Err(format!("fp4 matvec drift a={fp4_a_err} b={fp4_b_err}").into());
            }

            gpu_correctness = json!({
                "q4_split_max_abs_error": q4_a_err,
                "q4_interleaved_max_abs_error": q4_b_err,
                "q4_split_vs_interleaved": q4_ab,
                "q4_decoded_weights_bit_identical": q4_bit_identical,
                "binary_split_max_abs_error": bin_a_err,
                "binary_interleaved_max_abs_error": bin_b_err,
                "fp4_split_max_abs_error": fp4_a_err,
                "fp4_interleaved_max_abs_error": fp4_b_err,
                "fp4_split_vs_interleaved": fp4_ab,
                "fp4_bytes_bit_identical": true,
                "fallbacks": 0,
                "live_runtime_untouched": true,
            });

            let q4_bytes = (q4_codes.len() + q4_scales.len() * 2) as u64;
            let bin_bytes = (gate_bin.signs.len() + gate_bin.scales_f16.len() * 2) as u64;
            let fp4_bytes = (fp4_packed.len() + fp4_scales.len()) as u64;

            let time_q4 = |arm: &str| -> Result<u64, Box<dyn Error>> {
                write_f32(&out_a, &zeros);
                let t = if arm == "A" {
                    ctx.dispatch_threads_timed(
                        "qwen_uniform_q4_group64_matvec",
                        (Q80_GATE_ROWS as u32, 1, 1),
                        (256, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&q4_code_buf), 0);
                            enc.set_buffer(1, Some(&q4_scale_buf), 0);
                            enc.set_buffer(2, Some(&x_buf), 0);
                            enc.set_buffer(3, Some(&out_a), 0);
                            set_u32(enc, 4, Q80_GATE_ROWS as u32);
                            set_u32(enc, 5, Q80_GATE_COLS as u32);
                            set_u32(enc, 6, (Q80_GATE_COLS / UNIFORM_Q4_GROUP_SIZE) as u32);
                        },
                    )?
                } else {
                    ctx.dispatch_threads_timed(
                        "qwen_uniform_q4_group64_matvec_interleaved",
                        (Q80_GATE_ROWS as u32, 1, 1),
                        (256, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&q4_inter_buf), 0);
                            enc.set_buffer(1, Some(&x_buf), 0);
                            enc.set_buffer(2, Some(&out_a), 0);
                            set_u32(enc, 3, Q80_GATE_ROWS as u32);
                            set_u32(enc, 4, Q80_GATE_COLS as u32);
                            set_u32(enc, 5, (Q80_GATE_COLS / UNIFORM_Q4_GROUP_SIZE) as u32);
                        },
                    )?
                };
                gpu_ns(t, "q4")
            };
            let time_bin = |arm: &str| -> Result<u64, Box<dyn Error>> {
                write_f32(&out_a, &zeros);
                let t = if arm == "A" {
                    ctx.dispatch_threads_timed(
                        "q80_binary_group_matvec",
                        (Q80_GATE_ROWS as u32, 1, 1),
                        (256, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&bin_sign_buf), 0);
                            enc.set_buffer(1, Some(&bin_scale_buf), 0);
                            enc.set_buffer(2, Some(&x_buf), 0);
                            enc.set_buffer(3, Some(&out_a), 0);
                            set_u32(enc, 4, Q80_GATE_ROWS as u32);
                            set_u32(enc, 5, Q80_GATE_COLS as u32);
                            set_u32(enc, 6, Q80_BINARY_GROUP as u32);
                            set_u32(enc, 7, gate_bin.groups_per_row as u32);
                        },
                    )?
                } else {
                    ctx.dispatch_threads_timed(
                        "q80_binary_group_matvec_interleaved",
                        (Q80_GATE_ROWS as u32, 1, 1),
                        (256, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&bin_inter_buf), 0);
                            enc.set_buffer(1, Some(&x_buf), 0);
                            enc.set_buffer(2, Some(&out_a), 0);
                            set_u32(enc, 3, Q80_GATE_ROWS as u32);
                            set_u32(enc, 4, Q80_GATE_COLS as u32);
                            set_u32(enc, 5, Q80_BINARY_GROUP as u32);
                            set_u32(enc, 6, gate_bin.groups_per_row as u32);
                        },
                    )?
                };
                gpu_ns(t, "binary")
            };
            let time_fp4 = |arm: &str| -> Result<u64, Box<dyn Error>> {
                write_f32(&fp4_out_a, &fp4_zeros);
                let t = if arm == "A" {
                    ctx.dispatch_threads_timed(
                        "dsv4f_fp4_matvec_split",
                        (fp4_rows as u32, 1, 1),
                        (256, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&fp4_p_buf), 0);
                            enc.set_buffer(1, Some(&fp4_s_buf), 0);
                            enc.set_buffer(2, Some(&quant_buf), 0);
                            enc.set_buffer(3, Some(&act_buf), 0);
                            enc.set_buffer(4, Some(&fp4_out_a), 0);
                            set_u32(enc, 5, fp4_rows as u32);
                            set_u32(enc, 6, fp4_packed_cols as u32);
                            set_u32(enc, 7, fp4_scale_cols as u32);
                        },
                    )?
                } else {
                    ctx.dispatch_threads_timed(
                        "dsv4f_fp4_matvec_interleaved",
                        (fp4_rows as u32, 1, 1),
                        (256, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&fp4_i_buf), 0);
                            enc.set_buffer(1, Some(&quant_buf), 0);
                            enc.set_buffer(2, Some(&act_buf), 0);
                            enc.set_buffer(3, Some(&fp4_out_a), 0);
                            set_u32(enc, 4, fp4_rows as u32);
                            set_u32(enc, 5, fp4_packed_cols as u32);
                            set_u32(enc, 6, fp4_scale_cols as u32);
                        },
                    )?
                };
                gpu_ns(t, "fp4")
            };

            for _ in 0..WARMUP {
                let _ = time_q4("A")?;
                let _ = time_q4("B")?;
                let _ = time_bin("A")?;
                let _ = time_bin("B")?;
                let _ = time_fp4("A")?;
                let _ = time_fp4("B")?;
            }
            let mut q4_a = Vec::new();
            let mut q4_b = Vec::new();
            let mut bin_a = Vec::new();
            let mut bin_b = Vec::new();
            let mut fp4_a = Vec::new();
            let mut fp4_b = Vec::new();
            for _ in 0..PAIRS {
                q4_a.push(time_q4("A")?);
                q4_b.push(time_q4("B")?);
                bin_a.push(time_bin("A")?);
                bin_b.push(time_bin("B")?);
                fp4_a.push(time_fp4("A")?);
                fp4_b.push(time_fp4("B")?);
            }

            // DRAM sequential vs row-conflict probe, 64 MiB.
            let probe_bytes: usize = 64 * 1024 * 1024;
            let nthreads: u32 = 256 * 60;
            let iters: u32 = 4096;
            let probe = vec![0x5au8; probe_bytes];
            let probe_buf = ctx.new_buffer_with_bytes_checked(&probe)?;
            let probe_out = ctx.new_buffer_checked(nthreads as usize * 4)?;
            let time_probe = |stride: u32| -> Result<u64, Box<dyn Error>> {
                let t = ctx.dispatch_threads_timed(
                    "dram_row_locality_read_reduce",
                    (nthreads, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&probe_buf), 0);
                        enc.set_buffer(1, Some(&probe_out), 0);
                        set_u32(enc, 2, probe_bytes as u32);
                        set_u32(enc, 3, stride);
                        set_u32(enc, 4, iters);
                    },
                )?;
                gpu_ns(t, "probe")
            };
            let seq_stride = nthreads * 16;
            let conflict_stride = (DRAM_ROW_BYTES_MODEL as u32) + 64;
            for _ in 0..WARMUP {
                let _ = time_probe(seq_stride)?;
                let _ = time_probe(conflict_stride)?;
            }
            let mut seq = Vec::new();
            let mut conflict = Vec::new();
            for _ in 0..PAIRS {
                seq.push(time_probe(seq_stride)?);
                conflict.push(time_probe(conflict_stride)?);
            }
            let probe_moved = u64::from(nthreads) * u64::from(iters) * 16;

            fn arm_json(a: &[u64], b: &[u64], bytes: u64) -> Value {
                json!({
                    "A_split_gpu_ns": a,
                    "B_interleaved_gpu_ns": b,
                    "A_median_gpu_ns": median(a),
                    "B_median_gpu_ns": median(b),
                    "A_gbps": gbps(bytes, median(a)),
                    "B_gbps": gbps(bytes, median(b)),
                    "ceiling_gbps": CEILING_GBPS,
                    "A_frac_of_ceiling": gbps(bytes, median(a)) / CEILING_GBPS,
                    "B_frac_of_ceiling": gbps(bytes, median(b)) / CEILING_GBPS,
                    "bytes": bytes,
                    "gpu_time_authority": "MTLCommandBuffer.GPUEndTime-GPUStartTime after wait",
                    "measurement_label": "DIRTY_ENGINEERING",
                })
            }

            paired = json!({
                "q80_q4_gate": arm_json(&q4_a, &q4_b, q4_bytes),
                "q80_mixed_binary_gate": arm_json(&bin_a, &bin_b, bin_bytes),
                "dsv4f_fp4_reduced": arm_json(&fp4_a, &fp4_b, fp4_bytes),
                "dram_row_probe": {
                    "sequential_gpu_ns": seq,
                    "conflict_stride_8192_plus_64_gpu_ns": conflict,
                    "sequential_median_gpu_ns": median(&seq),
                    "conflict_median_gpu_ns": median(&conflict),
                    "sequential_gbps": gbps(probe_moved, median(&seq)),
                    "conflict_gbps": gbps(probe_moved, median(&conflict)),
                    "bytes_moved": probe_moved,
                    "measurement_label": "DIRTY_ENGINEERING",
                    "note": "Isolated read-reduce. If sequential and conflict overlap, this GPU is not row-limited on a 64 MiB stream either.",
                }
            });

            let q4_token_a = median(&q4_a).saturating_mul(Q80_LAYERS * Q80_TOP_K * 3);
            let q4_token_b = median(&q4_b).saturating_mul(Q80_LAYERS * Q80_TOP_K * 3);
            measure = json!({
                "ran": true,
                "measurement_label": "DIRTY_ENGINEERING",
                "whole_token_wall": "NOT RUN. Live Q80/DSV4F kernels were not switched. Isolated organ GPU times only; projected routed-expert ns is 48*10*3 (Q4) or 43*6*3 (FP4) times the organ median. That is not a token wall.",
                "projected_q80_q4_routed_ns_A": q4_token_a,
                "projected_q80_q4_routed_ns_B": q4_token_b,
                "paired": paired,
            });
        }

        let receipt = json!({
            "schema": SCHEMA,
            "lane": "dram-row-locality",
            "measurement_label": if args.measure { "DIRTY_ENGINEERING" } else { "DIRTY_ENGINEERING_MODEL_ONLY" },
            "claim_boundary": {
                "derived_address_model_not_hardware_counter": true,
                "dram_row_bytes_model": DRAM_ROW_BYTES_MODEL,
                "page_4k": 4096,
                "page_16k": PAGE_16K,
                "ceiling_gbps": CEILING_GBPS,
                "did_not_rewrite_live_q80_catalog": true,
                "did_not_rewrite_dsv4f_gravity_chunks": true,
                "did_not_change_default_kernels": true,
                "did_not_weaken_a_gate": true,
                "did_not_run_a_new_capture": true,
                "hot_expert_cache_not_claimed": true,
                "streams_claimed": ["q80_mixed_organs", "q80_q4_organs_and_cold_triplet_files", "dsv4f_cold_streamed_fp4_and_chunks"],
                "energy_figures_are_not_measured": true,
                "cpu_wall_is_not_gpu_time": true,
            },
            "physics": {
                "note": "Row activate/precharge dominates DRAM energy (~10x the column read) and costs tRP+tRCD. Two streams with the same byte count can differ in energy and achieved bandwidth by access order. Peak 819 GB/s assumes near-perfect row locality."
            },
            "address_streams": {
                "q80_mixed_binary_split": stats_json(&q80_bin_split),
                "q80_mixed_binary_interleaved": stats_json(&q80_bin_inter),
                "q80_q4_split": stats_json(&q80_q4_split),
                "q80_q4_interleaved": stats_json(&q80_q4_inter),
                "dsv4f_fp4_w1_split": stats_json(&dsv_w1_split),
                "dsv4f_fp4_w1_interleaved": stats_json(&dsv_w1_inter),
                "dsv4f_expert_six_chunks": stats_json(&dsv_six),
                "dsv4f_expert_colocated": stats_json(&dsv_one),
                "q80_expert_six_files": stats_json(&q80_six),
                "q80_expert_colocated": stats_json(&q80_one),
                "token_scaled": {
                    "q80_q4_48x10x3_split": stats_json(&token_q4),
                    "dsv4f_fp4_43x6x3_split": stats_json(&token_dsv),
                },
                "ranked_by_bytes_x_scatter": ranked.iter().map(|r| json!({
                    "name": r.stats.name,
                    "layout": r.stats.layout,
                    "bytes": r.stats.bytes,
                    "scatter_ratio": r.stats.scatter_ratio,
                    "bytes_x_scatter": r.bytes_x_scatter,
                    "unique_dram_rows": r.stats.unique_dram_rows,
                    "dram_row_transitions": r.stats.dram_row_transitions,
                })).collect::<Vec<_>>(),
            },
            "on_disk": {
                "q80_sha_named_files": file_scatter,
                "dsv4f_source_and_gravity": dsv_layout,
                "dsv4f_live_bundle": live_dsv4f,
            },
            "routing_cooccurrence": coreoute,
            "correctness_cpu": {
                "q4_decoded_weights_bit_identical": q4_bit_identical,
                "binary_decoded_weights_bit_identical": binary_bit_identical,
                "q4_codec_bpw": q4_quality.codec_bpw,
                "q4_interleaved_stride": Q4_INTERLEAVED_STRIDE,
                "binary_interleaved_stride": BINARY_INTERLEAVED_STRIDE_G128,
                "fp4_interleaved_stride": FP4_INTERLEAVED_STRIDE,
                "triplet_blob_roundtrip": triplet_roundtrip,
            },
            "gpu_correctness": gpu_correctness,
            "measure": measure,
            "pack_contract": {
                "q80_mixed_gate": "HGRAVB01 body stays scales||signs. Bind-time interleave into one MTLBuffer is value-preserving and does not invent a second on-disk family.",
                "q80_q4": "HQ30UQ4 body stays scales||codes. Bind-time [scale|codes]*G is the execution-order layout. Same BPW.",
                "dsv4f": "Do not interleave at no-copy bind (that reintroduces the memcpy the expert-nocopy lane deleted). A future gravity pack should emit one chunk per expert: w1_interleaved, w3_interleaved, w2_interleaved.",
                "co_route_permutation": "Permitted by cheap evidence from the existing capture index. Only pays inside a packed slab.",
            },
        });

        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&args.out, format!("{}\n", serde_json::to_string_pretty(&receipt)?))?;
        eprintln!("wrote {}", args.out.display());
        let _ = Q4_TOL;
        Ok(())
    }
}
