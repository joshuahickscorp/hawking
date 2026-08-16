//! Q80 low-rank algebra audit: does the token path reconstruct W or apply factors?
//!
//! CPU-only. Does not implement a reconstruct-then-multiply GPU kernel
//! (that path is forbidden). Grades against the packed two-stage artifact
//! oracle, never the BF16 parent and never dense (L@R)@x.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core --example ascension_qwen80_lowrank_algebra
//! Run (no GPU lock — no Metal):
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen80_lowrank_algebra \
//!     --out receipts/ascent-2026-08-16/q80-lowrank-algebra.json

use half::f16;
use hawking_core::model::qwen_complete_binary::{
    binary_group_matvec_f32, binary_group_weight, binary_rice_q1_matvec_f32, deterministic_input,
    deterministic_matrix, expand_rice_indices, hgravs01_two_stage_matvec_f32, max_abs_error,
    pack_binary_group, pack_binary_rice_q1, pack_uniform_factor, packed_byte_count,
    uniform_factor_value, BinaryGroupPacked, RiceQ1Packed, UniformFactorPacked,
    Q80_BINARY_GROUP_SIZE, Q80_DOWN_COLS, Q80_DOWN_ROWS, Q80_GATE_COLS, Q80_GATE_ROWS,
    Q80_HGRAVS_BITS, Q80_HGRAVS_GROUP_SIZE, Q80_HGRAVS_RANK, Q80_RICE_Q1_OUTLIER_RATIO,
};
use serde_json::{json, Value};
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

const SCHEMA: &str = "hawking.ascension.qwen80_lowrank_algebra.v1";
const SHIPPED_TOL: f32 = 2e-5;
const LAYERS: u64 = 48;
const TOP_K: u64 = 10;
const EXPERT_BUDGET_NS: u64 = 8_000_000;

struct Args {
    out: Option<PathBuf>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--out" => out = Some(PathBuf::from(value)),
            _ => return Err(format!("unsupported option {flag}").into()),
        }
    }
    Ok(Args { out })
}

fn decode_factor(packed: &UniformFactorPacked) -> Vec<f32> {
    let mut out = vec![0.0f32; packed.rows * packed.cols];
    for row in 0..packed.rows {
        for col in 0..packed.cols {
            out[row * packed.cols + col] = uniform_factor_value(packed, row, col);
        }
    }
    out
}

/// Parity-only. Forbidden on the token path.
fn reconstruct_w_then_matvec(
    left: &[f32],
    right: &[f32],
    rows: usize,
    rank: usize,
    cols: usize,
    x: &[f32],
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    for i in 0..rows {
        for j in 0..cols {
            let mut w = 0.0f32;
            for r in 0..rank {
                w += left[i * rank + r] * right[r * cols + j];
            }
            y[i] += w * x[j];
        }
    }
    y
}

fn two_stage_from_decoded(
    left: &[f32],
    right: &[f32],
    rows: usize,
    rank: usize,
    cols: usize,
    x: &[f32],
) -> Vec<f32> {
    let mut mid = vec![0.0f32; rank];
    for r in 0..rank {
        let mut sum = 0.0f32;
        let row = &right[r * cols..(r + 1) * cols];
        for c in 0..cols {
            sum += row[c] * x[c];
        }
        mid[r] = sum;
    }
    let mut y = vec![0.0f32; rows];
    for i in 0..rows {
        let mut sum = 0.0f32;
        let row = &left[i * rank..(i + 1) * rank];
        for r in 0..rank {
            sum += row[r] * mid[r];
        }
        y[i] = sum;
    }
    y
}

fn reconstruct_up_then_matvec(
    packed: &RiceQ1Packed,
    x: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let rows = packed.binary.rows;
    let cols = packed.binary.cols;
    let mut w = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            w[row * cols + col] = binary_group_weight(&packed.binary, row, col);
        }
    }
    let indices = expand_rice_indices(packed)?;
    let scale = f16::from_bits(packed.residual_scale_f16).to_f32();
    for (n, &flat) in indices.iter().enumerate() {
        let flat = flat as usize;
        let positive = ((packed.residual_signs[n >> 3] >> (n & 7)) & 1) != 0;
        w[flat] += if positive { scale } else { -scale };
    }
    let mut y = vec![0.0f32; rows];
    for row in 0..rows {
        let mut sum = 0.0f32;
        let row_w = &w[row * cols..(row + 1) * cols];
        for col in 0..cols {
            sum += row_w[col] * x[col];
        }
        y[row] = sum;
    }
    Ok(y)
}

fn factor_body_bytes(packed: &UniformFactorPacked) -> Result<usize, Box<dyn Error>> {
    let scale_bytes = packed.scales_f16.len() * 2;
    Ok(scale_bytes + packed.codes.len())
}

fn binary_body_bytes(packed: &BinaryGroupPacked) -> usize {
    packed.scales_f16.len() * 2 + packed.signs.len()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let left_w = deterministic_matrix(Q80_DOWN_ROWS, Q80_HGRAVS_RANK, 3);
    let right_w = deterministic_matrix(Q80_HGRAVS_RANK, Q80_DOWN_COLS, 4);
    let gate_w = deterministic_matrix(Q80_GATE_ROWS, Q80_GATE_COLS, 1);
    let up_w = deterministic_matrix(Q80_GATE_ROWS, Q80_GATE_COLS, 2);
    let x_down = deterministic_input(Q80_DOWN_COLS);
    let x_hidden = deterministic_input(Q80_GATE_COLS);

    let left = pack_uniform_factor(
        &left_w,
        Q80_DOWN_ROWS,
        Q80_HGRAVS_RANK,
        Q80_HGRAVS_BITS,
        Q80_HGRAVS_GROUP_SIZE,
    )?;
    let right = pack_uniform_factor(
        &right_w,
        Q80_HGRAVS_RANK,
        Q80_DOWN_COLS,
        Q80_HGRAVS_BITS,
        Q80_HGRAVS_GROUP_SIZE,
    )?;
    let gate = pack_binary_group(&gate_w, Q80_GATE_ROWS, Q80_GATE_COLS, Q80_BINARY_GROUP_SIZE)?;
    let up = pack_binary_rice_q1(
        &up_w,
        Q80_GATE_ROWS,
        Q80_GATE_COLS,
        Q80_RICE_Q1_OUTLIER_RATIO,
    )?;

    let y_two = hgravs01_two_stage_matvec_f32(&left, &right, &x_down)?;
    let l = decode_factor(&left);
    let r = decode_factor(&right);
    let y_fac = two_stage_from_decoded(
        &l,
        &r,
        Q80_DOWN_ROWS,
        Q80_HGRAVS_RANK,
        Q80_DOWN_COLS,
        &x_down,
    );
    let y_recon = reconstruct_w_then_matvec(
        &l,
        &r,
        Q80_DOWN_ROWS,
        Q80_HGRAVS_RANK,
        Q80_DOWN_COLS,
        &x_down,
    );
    let two_stage_self_err = max_abs_error(&y_two, &y_fac);
    let reconstruct_vs_two_stage = max_abs_error(&y_two, &y_recon);
    if two_stage_self_err > SHIPPED_TOL {
        return Err(format!(
            "packed two-stage vs decoded-factor two-stage drifted {two_stage_self_err} > {SHIPPED_TOL}"
        )
        .into());
    }

    let y_up_fused = binary_rice_q1_matvec_f32(&up, &x_hidden)?;
    let y_up_recon = reconstruct_up_then_matvec(&up, &x_hidden)?;
    let up_recon_vs_fused = max_abs_error(&y_up_fused, &y_up_recon);
    let y_gate = binary_group_matvec_f32(&gate, &x_hidden)?;
    if y_gate.iter().any(|v| !v.is_finite()) || y_up_fused.iter().any(|v| !v.is_finite()) {
        return Err("gate/up oracle produced a non-finite value".into());
    }
    let rice_indices = expand_rice_indices(&up)?;
    if rice_indices != up.indices {
        return Err("rice expand is not bit-identical to packed indices".into());
    }

    let dense_macs = (Q80_DOWN_ROWS * Q80_DOWN_COLS) as u64;
    let factored_macs = (Q80_HGRAVS_RANK * (Q80_DOWN_COLS + Q80_DOWN_ROWS)) as u64;
    let reconstruct_form_w_macs = (Q80_DOWN_ROWS * Q80_HGRAVS_RANK * Q80_DOWN_COLS) as u64;
    let dense_w_f32_bytes = dense_macs * 4;
    let mid_bytes = (Q80_HGRAVS_RANK * 4) as u64;
    let left_bytes = factor_body_bytes(&left)?;
    let right_bytes = factor_body_bytes(&right)?;
    let expected_left_codes = packed_byte_count(Q80_DOWN_ROWS * Q80_HGRAVS_RANK, Q80_HGRAVS_BITS)?;
    let expected_right_codes = packed_byte_count(Q80_HGRAVS_RANK * Q80_DOWN_COLS, Q80_HGRAVS_BITS)?;
    if left.codes.len() != expected_left_codes || right.codes.len() != expected_right_codes {
        return Err("packed factor code lengths disagree with 3-bit geometry".into());
    }

    let gate_macs = (Q80_GATE_ROWS * Q80_GATE_COLS) as u64;
    let up_residual_macs = up.outlier_count as u64;
    let up_dense_w_bytes = gate_macs * 4;

    // Inherited quieter-window GPU times from the decode-throughput lane.
    // This example does not re-run Metal and does not implement reconstruct-on-GPU.
    let inherited = inherited_throughput_window();

    let down_b_ns = inherited["down_b_ns"].as_array().unwrap();
    let down_b: Vec<u64> = down_b_ns.iter().map(|v| v.as_u64().unwrap()).collect();
    let organ_b = inherited["median_b_ns"]["organ_sum"].as_u64().unwrap();
    let routed_b = inherited["median_b_ns"]["routed_48x10"].as_u64().unwrap();

    let receipt = json!({
        "schema": SCHEMA,
        "lane": "q80-lowrank-algebra",
        "date": "2026-08-16",
        "status": "SHIPPED_CONFIRMED_NEGATIVE_ALREADY_FACTORED",
        "doctrine": "REPRESENTATION -> ALGEBRA. Never reconstruct a structured representation when the algebra can consume it directly.",
        "verdict": "The shipped Q80 down_proj token path already executes y = L @ (R @ x). It does not reconstruct dense W. Implementing a reconstruct-then-multiply kernel would be a regression against the standing law and is refused. This lane is a confirmed negative, not a kernel rewrite.",
        "claim_boundary": {
            "artifact_packed": false,
            "coherence_generation_tested": false,
            "cpu_wall_is_not_gpu_time": true,
            "decode_kernel_exists": true,
            "dense_w_never_materialized": true,
            "not_bf16_parent_parity": true,
            "not_full_token_runtime": true,
            "this_lane_did_not_run_metal": true,
            "reconstruct_gpu_kernel_not_implemented": true
        },
        "findings": {
            "down_proj": {
                "codec": "hgravs01_r160_b3",
                "already_factored": true,
                "token_path": "y = L @ (R @ x)",
                "cpu_oracle": {
                    "file": "crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs",
                    "fn": "hgravs01_two_stage_matvec_f32",
                    "lines": "566-580",
                    "quote": "let mid = uniform_factor_matvec_f32(right, input)?; uniform_factor_matvec_f32(left, &mid)"
                },
                "metal_two_dispatch": {
                    "file": "crates/hawking-core/shaders/q80_mixed_decode.metal",
                    "kernel": "q80_hgravs01_factor_matvec_simd3 R then L",
                    "lines": "903-964 plus host two-dispatch in ascension_qwen80_mixed_decode_throughput.rs:935+",
                    "temporary": "device mid[rank=160] f32 = 640 B"
                },
                "metal_fused_single_dispatch": {
                    "kernel": "q80_hgravs01_two_stage_matvec",
                    "lines": "594-674",
                    "quote": "Fused y = L @ (R @ x) in one dispatch. Each threadgroup recomputes mid[rank]",
                    "status": "EXISTS but REJECTED as token path: sibling receipt fusion_negative, 5-13x slower because every TG recomputes R[160x512]. Metal has no grid-wide barrier."
                },
                "forbidden_path_absent": "packed -> dense (2048 x 512) W -> W @ x is not implemented on the token path. decode_hgravs01_dense_f32_for_parity is CPU parity only (activation_weighted_svd.rs:677)."
            },
            "up_proj": {
                "codec": "binary + rice_q1_rms sparse residual @ 2%",
                "already_fused": true,
                "token_path": "y = binary_group(x) + sparse residual scatter-add",
                "cpu_oracle": {
                    "file": "crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs",
                    "fn": "binary_rice_q1_matvec_f32",
                    "lines": "459-478",
                    "quote": "y = binary_group(x) + rice_q1 residual corrections"
                },
                "metal_fused": {
                    "kernels": [
                        "q80_binary_group_csr_matvec",
                        "q80_binary_group_csr_matvec_bytes",
                        "q80_binary_group_csr_matvec_tg256"
                    ],
                    "lines": "472-520, 800-895",
                    "quote": "Binary simd + CSR residual in the same dispatch. Residual add order is serial on lane 0 after the binary reduction."
                },
                "not_reconstruct": "Residual is a sparse correction fused into y, not applied to a dense reconstructed W."
            },
            "gate_proj": {
                "codec": "binary_group 128",
                "structure": "none — unstructured sign+scale",
                "token_path": "decode-in-register FMA, no W temporary",
                "algebra_question": "not applicable; there is no low-rank / codebook / residual factor to consume"
            },
            "nonexpert_hgravu01": {
                "codec": "uniform-q8 group-64",
                "covers": "embed, lm_head, attention, DeltaNet, norms, router, shared expert",
                "structure": "none — not low-rank",
                "algebra_question": "decode-in-register vs reconstruct; no U@(V@x) available. Shared expert is 8-bit, not hgravs01."
            },
            "additive_codebooks": {
                "present_on_q80_mixed": false,
                "note": "No PQ / additive-codebook organ is in the mixed-1p5 recipe."
            }
        },
        "flops": {
            "shape": {
                "down": [Q80_DOWN_ROWS, Q80_DOWN_COLS],
                "rank": Q80_HGRAVS_RANK,
                "gate_up": [Q80_GATE_ROWS, Q80_GATE_COLS]
            },
            "down_proj": {
                "dense_w_x_macs": dense_macs,
                "dense_w_x_flops_muladd": dense_macs * 2,
                "factored_macs": factored_macs,
                "factored_flops_muladd": factored_macs * 2,
                "factored_breakdown": {
                    "R_x_macs": Q80_HGRAVS_RANK * Q80_DOWN_COLS,
                    "L_mid_macs": Q80_DOWN_ROWS * Q80_HGRAVS_RANK
                },
                "mac_reduction": (dense_macs as f64) / (factored_macs as f64),
                "reconstruct_form_w_macs": reconstruct_form_w_macs,
                "note": "Shipped path already pays only factored_macs. Reconstruct-then-multiply would pay form_w + dense_w_x every token unless W were cached — and a decoded-W cache is already rejected (12 MiB/expert, 288 GiB if every layer-expert is cached)."
            },
            "up_proj": {
                "binary_macs": gate_macs,
                "residual_macs": up_residual_macs,
                "fused_macs": gate_macs + up_residual_macs,
                "reconstruct_w_then_x_macs": gate_macs,
                "flop_reduction_vs_reconstruct": 1.0,
                "note": "Fusing the 2% residual does not reduce FLOPs versus (B+R)@x. It eliminates the dense W temporary and the write of B+R."
            },
            "gate_proj": {
                "macs": gate_macs,
                "algebra_reduction": 1.0
            }
        },
        "temporaries": {
            "down_mid_f32_bytes": mid_bytes,
            "down_dense_w_f32_bytes_eliminated": dense_w_f32_bytes,
            "down_packed_left_bytes": left_bytes,
            "down_packed_right_bytes": right_bytes,
            "down_packed_factor_bytes": left_bytes + right_bytes,
            "up_dense_w_f32_bytes_eliminated": up_dense_w_bytes,
            "up_binary_body_bytes": binary_body_bytes(&up.binary),
            "up_outlier_count": up.outlier_count,
            "gate_body_bytes": binary_body_bytes(&gate),
            "zero_temporary_principle": "mid[160] f32 is the only down_proj temporary. No (2048 x 512) W. up residual uses bind-time expanded indices, not dense W."
        },
        "correctness": {
            "oracle": "packed two-stage / fused rice artifact, never BF16 parent, never dense (L@R)@x",
            "numeric_equivalence_gate": {
                "statement": "Reassociating L@(R@x) versus (L@R)@x changes f32 add order. Kernels are graded against the two-stage artifact oracle at tolerance 2e-5. Reconstruct-vs-two-stage drift is reported and is NOT a fail.",
                "shipped_tolerance": SHIPPED_TOL,
                "shipped_measured_kernel_drift": {
                    "gate": 1.811981201171875e-5,
                    "up": 1.0967254638671875e-5,
                    "down": 1.1444091796875e-5,
                    "source": "receipts/ascent-2026-08-16/q80-decode-throughput.json"
                },
                "this_lane_cpu": {
                    "packed_two_stage_vs_decoded_factor_two_stage": two_stage_self_err,
                    "reconstruct_LR_then_Wx_vs_two_stage": reconstruct_vs_two_stage,
                    "reconstruct_exceeds_shipped_tol": reconstruct_vs_two_stage > SHIPPED_TOL,
                    "up_reconstruct_W_then_x_vs_fused": up_recon_vs_fused,
                    "rice_indices": "bit-identical"
                }
            },
            "fallbacks": 0,
            "gates_not_weakened": true
        },
        "inherited_gpu_from_q80_decode_throughput": inherited,
        "timing": {
            "label": "DIRTY_ENGINEERING",
            "source": "receipts/ascent-2026-08-16/q80-decode-throughput.json windows.quieter_A_matched_shipped",
            "this_lane_new_gpu_reps": [],
            "why_no_new_gpu": "Algebra question is already answered in shipped kernels. A reconstruct-then-multiply Metal arm is forbidden (faster-by-reconstructing is rejected; never materialize dense W). Re-timing the already-factored kernels would only restate the occupancy lane.",
            "per_organ_us": {
                "gate": {"before_us": 60.541, "after_us": 6.875, "note": "occupancy, not algebra. Already decode-in-register both arms."},
                "up": {"before_us": 80.041, "after_us": 17.25, "note": "both arms already fuse residual into y. Occupancy, not reconstruct-vs-fuse."},
                "down": {"before_us": 261.291, "after_us": 13.959, "note": "both arms already L@(R@x). Serial two-dispatch vs simd3 two-dispatch. Not reconstruct vs factors."}
            },
            "baseline_ns_per_token": 192_899_040u64,
            "result_ns_per_token": 18_280_320u64,
            "reps_down_b_ns": down_b,
            "routed_48x10_ns": routed_b,
            "organ_sum_ns": organ_b
        },
        "budget": {
            "expert_work_budget_ns": EXPERT_BUDGET_NS,
            "inherited_routed_ns_quiet": 18_280_320u64,
            "over_budget": 18_280_320f64 / EXPERT_BUDGET_NS as f64,
            "layers": LAYERS,
            "top_k": TOP_K,
            "honest": "Algebra is already applied. This lane does not move the 18.28 ms routed-expert number. Still ~2.285x over the 8 ms budget."
        },
        "next_bottleneck": {
            "what": "up_proj binary decode / occupancy, ~17.25 us, ~45% of the remaining 38 us per expert. Achieved ~15 GB/s. Still decode/occupancy bound, not DRAM bound, not reconstruct-W bound.",
            "measured_ns": 17_250u64,
            "source": "receipts/ascent-2026-08-16/q80-decode-throughput.json quieter window up_b median",
            "label": "DIRTY_ENGINEERING"
        }
    });

    let text = serde_json::to_string_pretty(&receipt)?;
    if let Some(path) = args.out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, format!("{text}\n"))?;
        eprintln!("wrote {}", path.display());
    } else {
        println!("{text}");
    }
    Ok(())
}

fn inherited_throughput_window() -> Value {
    json!({
        "label": "DIRTY_ENGINEERING",
        "receipt": "receipts/ascent-2026-08-16/q80-decode-throughput.json",
        "window": "quieter_A_matched_shipped",
        "protocol": "warmup=2; 6 alternating A,B pairs; GPUEndTime-GPUStartTime",
        "gate_a_ns": [61041, 60791, 60500, 59750, 60541, 60166],
        "gate_b_ns": [6875, 6959, 6709, 6625, 6625, 6959],
        "up_a_ns": [80709, 79791, 79834, 80041, 79500, 80541],
        "up_b_ns": [17334, 17625, 17250, 17250, 17166, 17125],
        "down_a_ns": [259166, 272250, 261291, 259666, 257125, 265250],
        "down_b_ns": [13750, 13959, 12875, 14000, 12875, 14834],
        "median_a_ns": {"gate": 60541, "up": 80041, "down": 261291, "organ_sum": 401873, "routed_48x10": 192899040},
        "median_b_ns": {"gate": 6875, "up": 17250, "down": 13959, "organ_sum": 38084, "routed_48x10": 18280320},
        "selected_b": {
            "gate": "q80_binary_group_matvec_tg256",
            "up": "q80_binary_group_csr_matvec_bytes",
            "down": "q80_hgravs01_factor_matvec_simd3 R then L"
        }
    })
}
