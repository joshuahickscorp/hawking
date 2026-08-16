//! Q80 mixed-codec switching activity: measure alpha, find a free codebook
//! permutation, prove decoded values stay bit-identical, then time original
//! vs remapped packs on the GPU.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core \
//!     --example ascension_qwen80_codec_switching_activity \
//!     --target-dir workspace/ops/build/rust
//! Run under the GPU mutex:
//!   ./tools/gpu_lane_lock.sh codec-switching-activity \
//!     workspace/ops/build/rust/release-fast/examples/ascension_qwen80_codec_switching_activity \
//!     --out receipts/ascent-2026-08-16/codec-switching-activity.json

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::metal::MetalContext;
    use hawking_core::model::qwen_complete_binary::{
        activity_from_transitions, apply_assignment, assignment_cost, binary_group_matvec_f32,
        binary_kernel_sign_bytes, binary_rice_q1_matvec_f32, binary_sign_bits,
        decode_hgravs01_factors_f32, deterministic_input, deterministic_matrix,
        factor_elements_used, gray_assignment, hgravs01_two_stage_matvec_f32, identity_assignment,
        invert_sign_bytes, json_activity, json_code_activity, kernel_simd_tile_bytes,
        minimize_assignment, msb_pack_activity, pack_binary_group, pack_binary_rice_q1,
        pack_uniform_factor, packed_word_activity, peek_hgravs01_codes, remap_uniform_factor,
        rice_kernel_bytes, scale_bytes_from_f16, sign_magnitude_assignment, transition_matrix,
        twos_complement_assignment, uniform_factor_signed_levels, unpack_unsigned_lsb,
        unpacked_code_activity, Assignment, BinaryGroupPacked, RiceQ1Packed, UniformFactorPacked,
        Q80_BINARY_GROUP_SIZE, Q80_DOWN_COLS, Q80_DOWN_ROWS, Q80_GATE_COLS, Q80_GATE_ROWS,
        Q80_HGRAVS_BITS, Q80_HGRAVS_GROUP_SIZE, Q80_HGRAVS_RANK, Q80_RICE_Q1_OUTLIER_RATIO,
    };
    use metal::{CompileOptions, ComputePipelineState, MTLSize};
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};

    const SCHEMA: &str = "hawking.ascension.q80_codec_switching_activity.v1";
    const SHADER: &str = include_str!("../shaders/q80_codec_activity.metal");
    const WARMUP: usize = 2;
    const PAIRS: usize = 3;
    const LAYERS: u64 = 48;
    const TOP_K: u64 = 10;
    const HGRAVS_TOL: f32 = 2e-5;
    const DEFAULT_HGRAVS: &str = "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/activation-weighted-svd-v1/selected-payloads";

    struct Args {
        out: Option<PathBuf>,
        hgravs_dir: PathBuf,
        max_hgravs: Option<usize>,
        skip_gpu: bool,
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let mut out = None;
        let mut hgravs_dir = PathBuf::from(DEFAULT_HGRAVS);
        let mut max_hgravs = None;
        let mut skip_gpu = false;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--out" => {
                    out = Some(PathBuf::from(args.next().ok_or("missing --out value")?));
                }
                "--hgravs-dir" => {
                    hgravs_dir = PathBuf::from(args.next().ok_or("missing --hgravs-dir value")?);
                }
                "--max-hgravs" => {
                    max_hgravs = Some(
                        args.next()
                            .ok_or("missing --max-hgravs value")?
                            .parse::<usize>()?,
                    );
                }
                "--skip-gpu" => skip_gpu = true,
                other => return Err(format!("unsupported option {other}").into()),
            }
        }
        Ok(Args {
            out,
            hgravs_dir,
            max_hgravs,
            skip_gpu,
        })
    }

    fn as_u8_u16(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn as_u8_f32(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn as_u8_i32(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    fn read_f32(buffer: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, n).to_vec() }
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

    fn tiled_grid(rows: u32, rows_per_tg: u32) -> (u32, u32, u32) {
        let groups = rows.div_ceil(rows_per_tg);
        (groups * 256, 1, 1)
    }

    fn require_gpu_ns(gpu_duration_us: Option<u64>, label: &str) -> Result<u64, Box<dyn Error>> {
        let us = gpu_duration_us.ok_or_else(|| {
            format!("{label}: MTLCommandBuffer GPUStartTime/GPUEndTime unavailable")
        })?;
        Ok(us.saturating_mul(1000))
    }

    fn median(values: &[u64]) -> u64 {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    fn json_assignment(assignment: &Assignment, identity_cost: u64) -> Value {
        json!({
            "method": assignment.method,
            "bits": assignment.bits,
            "symbol_to_code": assignment.symbol_to_code,
            "cost_transitions": assignment.cost,
            "identity_cost_transitions": identity_cost,
            "transition_reduction": identity_cost.saturating_sub(assignment.cost),
            "reduction_ratio": if identity_cost == 0 {
                0.0
            } else {
                1.0 - (assignment.cost as f64 / identity_cost as f64)
            },
        })
    }

    fn named(mut assignment: Assignment, matrix: &[Vec<u64>]) -> Assignment {
        assignment.cost = assignment_cost(matrix, &assignment.symbol_to_code);
        assignment
    }

    fn stream_report(label: &str, bytes: &[u8], codes: Option<(&[u8], u8)>) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("label".into(), json!(label));
        obj.insert("bytes".into(), json!(bytes.len()));
        obj.insert("u8".into(), json_activity(&packed_word_activity(bytes, 8)));
        obj.insert(
            "u32".into(),
            json_activity(&packed_word_activity(bytes, 32)),
        );
        obj.insert(
            "u128".into(),
            json_activity(&packed_word_activity(bytes, 128)),
        );
        if let Some((codes, bits)) = codes {
            obj.insert(
                "code_stream".into(),
                json_code_activity(&unpacked_code_activity(codes, bits)),
            );
        }
        Value::Object(obj)
    }

    fn prove_remap(
        packed: &UniformFactorPacked,
        assignment: &Assignment,
    ) -> Result<Value, Box<dyn Error>> {
        let remapped = remap_uniform_factor(packed, assignment)?;
        let original_q = uniform_factor_signed_levels(packed, None)?;
        let remapped_q = uniform_factor_signed_levels(&remapped.packed, Some(&remapped.lut))?;
        if original_q != remapped_q {
            return Err("signed-level vectors are not bit-identical after remap".into());
        }
        if remapped.packed.codes.len() != packed.codes.len() {
            return Err("remapped pack changed byte count".into());
        }
        if remapped.packed.scales_f16 != packed.scales_f16 {
            return Err("remap mutated scales (weight values must not change)".into());
        }
        Ok(json!({
            "bit_identical_signed_levels": true,
            "packed_bytes_before": packed.codes.len(),
            "packed_bytes_after": remapped.packed.codes.len(),
            "bpw_unchanged": true,
            "scales_unchanged": true,
            "elements": original_q.len(),
        }))
    }

    fn scan_real_hgravs(dir: &Path, max_files: Option<usize>) -> Result<Value, Box<dyn Error>> {
        if !dir.is_dir() {
            return Ok(json!({
                "present": false,
                "path": dir.display().to_string(),
            }));
        }
        let mut files: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("hgravs01"))
            .collect();
        files.sort();
        if let Some(max) = max_files {
            files.truncate(max);
        }
        let mut n4 = 0u64;
        let mut n3 = 0u64;
        let mut matrix4 = vec![vec![0u64; 16]; 16];
        let mut matrix3 = vec![vec![0u64; 8]; 8];
        let mut u32_transitions_4 = 0u64;
        let mut u32_words_4 = 0u64;
        let mut u32_transitions_3 = 0u64;
        let mut u32_words_3 = 0u64;
        let mut code_transitions_4 = 0u64;
        let mut code_pairs_4 = 0u64;
        let mut code_transitions_3 = 0u64;
        let mut code_pairs_3 = 0u64;
        let mut slice_source: Option<PathBuf> = None;
        for path in &files {
            let payload = fs::read(path)?;
            let view = peek_hgravs01_codes(&payload)?;
            if view.matrix_shape == [2048, 768] && slice_source.is_none() {
                slice_source = Some(path.clone());
            }
            for (codes, packed) in [
                (view.left_codes.as_slice(), view.left_code_bytes.as_slice()),
                (
                    view.right_codes.as_slice(),
                    view.right_code_bytes.as_slice(),
                ),
            ] {
                let act = unpacked_code_activity(codes, view.bits);
                let words = packed_word_activity(packed, 32);
                if view.bits == 4 {
                    n4 += 1;
                    accumulate(&mut matrix4, codes, 4);
                    u32_transitions_4 += words.transitions;
                    u32_words_4 += words.words;
                    code_transitions_4 += act.transitions;
                    code_pairs_4 += act.codes.saturating_sub(1);
                } else if view.bits == 3 {
                    n3 += 1;
                    accumulate(&mut matrix3, codes, 3);
                    u32_transitions_3 += words.transitions;
                    u32_words_3 += words.words;
                    code_transitions_3 += act.transitions;
                    code_pairs_3 += act.codes.saturating_sub(1);
                }
            }
        }
        let assign4 = if n4 > 0 {
            Some(minimize_assignment(&matrix4, 4))
        } else {
            None
        };
        let assign3 = if n3 > 0 {
            Some(minimize_assignment(&matrix3, 3))
        } else {
            None
        };
        let id4 = if n4 > 0 {
            assignment_cost(&matrix4, &identity_assignment(4).symbol_to_code)
        } else {
            0
        };
        let id3 = if n3 > 0 {
            assignment_cost(&matrix3, &identity_assignment(3).symbol_to_code)
        } else {
            0
        };

        let mut remapped_u32_transitions_4 = 0u64;
        let mut remapped_u32_words_4 = 0u64;
        let mut remapped_code_transitions_4 = 0u64;
        let mut remapped_code_pairs_4 = 0u64;
        if let Some(assignment) = assign4.as_ref() {
            for path in &files {
                let payload = fs::read(path)?;
                let view = peek_hgravs01_codes(&payload)?;
                if view.bits != 4 {
                    continue;
                }
                for codes in [view.left_codes.as_slice(), view.right_codes.as_slice()] {
                    let remapped = apply_assignment(codes, &assignment.symbol_to_code)?;
                    let act = unpacked_code_activity(&remapped, 4);
                    remapped_code_transitions_4 += act.transitions;
                    remapped_code_pairs_4 += act.codes.saturating_sub(1);
                    let packed =
                        hawking_core::model::qwen_complete_binary::pack_unsigned_lsb(&remapped, 4)?;
                    let words = packed_word_activity(&packed, 32);
                    remapped_u32_transitions_4 += words.transitions;
                    remapped_u32_words_4 += words.words;
                }
            }
        }

        Ok(json!({
            "present": true,
            "path": dir.display().to_string(),
            "files_scanned": files.len(),
            "four_bit_factor_streams": n4,
            "three_bit_factor_streams": n3,
            "four_bit": {
                "code_stream": {
                    "transitions": code_transitions_4,
                    "pairs": code_pairs_4,
                    "mean_hamming": if code_pairs_4 == 0 { 0.0 } else { code_transitions_4 as f64 / code_pairs_4 as f64 },
                    "alpha": if code_pairs_4 == 0 { 0.0 } else { (code_transitions_4 as f64 / code_pairs_4 as f64) / 4.0 },
                },
                "packed_u32": json_activity(&activity_from_transitions(32, u32_words_4, u32_transitions_4)),
                "packed_u32_after_remap": json_activity(&activity_from_transitions(
                    32,
                    remapped_u32_words_4,
                    remapped_u32_transitions_4,
                )),
                "code_stream_after_remap": {
                    "transitions": remapped_code_transitions_4,
                    "pairs": remapped_code_pairs_4,
                    "mean_hamming": if remapped_code_pairs_4 == 0 {
                        0.0
                    } else {
                        remapped_code_transitions_4 as f64 / remapped_code_pairs_4 as f64
                    },
                    "alpha": if remapped_code_pairs_4 == 0 {
                        0.0
                    } else {
                        (remapped_code_transitions_4 as f64 / remapped_code_pairs_4 as f64) / 4.0
                    },
                },
                "identity_cost": id4,
                "assignment": assign4.as_ref().map(|a| json_assignment(a, id4)),
                "gray": assign4.as_ref().map(|_| {
                    let g = named(gray_assignment(4), &matrix4);
                    json_assignment(&g, id4)
                }),
                "twos_complement": assign4.as_ref().map(|_| {
                    let t = named(twos_complement_assignment(4), &matrix4);
                    json_assignment(&t, id4)
                }),
                "sign_magnitude": assign4.as_ref().map(|_| {
                    let s = named(sign_magnitude_assignment(4), &matrix4);
                    json_assignment(&s, id4)
                }),
            },
            "three_bit_real_file": {
                "code_stream": {
                    "transitions": code_transitions_3,
                    "pairs": code_pairs_3,
                    "mean_hamming": if code_pairs_3 == 0 { 0.0 } else { code_transitions_3 as f64 / code_pairs_3 as f64 },
                    "alpha": if code_pairs_3 == 0 { 0.0 } else { (code_transitions_3 as f64 / code_pairs_3 as f64) / 3.0 },
                },
                "packed_u32": json_activity(&activity_from_transitions(32, u32_words_3, u32_transitions_3)),
                "identity_cost": id3,
                "assignment": assign3.as_ref().map(|a| json_assignment(a, id3)),
            },
            "q80_geometry_slice_source": slice_source.map(|p| p.display().to_string()),
        }))
    }

    fn accumulate(matrix: &mut [Vec<u64>], codes: &[u8], bits: u8) {
        let n = 1usize << bits;
        for pair in codes.windows(2) {
            let a = pair[0] as usize;
            let b = pair[1] as usize;
            if a < n && b < n {
                matrix[a][b] += 1;
            }
        }
    }

    fn slice_real_q80_factors(path: &Path) -> Result<(Vec<f32>, Vec<f32>, String), Box<dyn Error>> {
        let payload = fs::read(path)?;
        let (header, left, right) = decode_hgravs01_factors_f32(&payload)?;
        if header.left.shape != [Q80_DOWN_ROWS, header.rank]
            || header.rank < Q80_HGRAVS_RANK
            || header.right.shape[1] < Q80_DOWN_COLS
        {
            return Err(format!(
                "real HGRAVS01 {} cannot be sliced to Q80 down geometry",
                path.display()
            )
            .into());
        }
        let src_rank = header.rank;
        let src_cols = header.right.shape[1];
        let mut left_s = Vec::with_capacity(Q80_DOWN_ROWS * Q80_HGRAVS_RANK);
        for row in 0..Q80_DOWN_ROWS {
            left_s.extend_from_slice(&left[row * src_rank..row * src_rank + Q80_HGRAVS_RANK]);
        }
        let mut right_s = Vec::with_capacity(Q80_HGRAVS_RANK * Q80_DOWN_COLS);
        for row in 0..Q80_HGRAVS_RANK {
            right_s.extend_from_slice(&right[row * src_cols..row * src_cols + Q80_DOWN_COLS]);
        }
        Ok((left_s, right_s, path.display().to_string()))
    }

    fn factor_activity(packed: &UniformFactorPacked, name: &str) -> Result<Value, Box<dyn Error>> {
        let codes = factor_elements_used(packed)?;
        let tile =
            kernel_simd_tile_bytes(&packed.codes, packed.rows, packed.cols, packed.bits, 32, 8);
        let (msb, lsb_u32, msb_u32) = msb_pack_activity(&codes, packed.bits)?;
        let _ = msb;
        Ok(json!({
            "name": name,
            "shape": [packed.rows, packed.cols],
            "bits": packed.bits,
            "group_size": packed.group_size,
            "packed_bytes": packed.codes.len(),
            "scale_bytes": packed.scales_f16.len() * 2,
            "code_stream": json_code_activity(&unpacked_code_activity(&codes, packed.bits)),
            "packed_storage": stream_report("storage_order", &packed.codes, Some((&codes, packed.bits))),
            "kernel_simd_tile": stream_report("simd_tile_fetch", &tile, None),
            "scales": stream_report("fp16_scales", &scale_bytes_from_f16(&packed.scales_f16), None),
            "msb_vs_lsb_u32": {
                "lsb": json_activity(&lsb_u32),
                "msb": json_activity(&msb_u32),
                "note": "MSB-first is free iff extract is also MSB-first. Intra-word bit reversal of whole words preserves Hamming; 3-bit codes straddle words, so MSB vs LSB can change u32 activity."
            },
        }))
    }

    fn solve_factor(packed: &UniformFactorPacked) -> Result<(Assignment, Value), Box<dyn Error>> {
        let codes = factor_elements_used(packed)?;
        let matrix = transition_matrix(&codes, packed.bits);
        let identity = named(identity_assignment(packed.bits), &matrix);
        let gray = named(gray_assignment(packed.bits), &matrix);
        let twos = named(twos_complement_assignment(packed.bits), &matrix);
        let signmag = named(sign_magnitude_assignment(packed.bits), &matrix);
        let best = minimize_assignment(&matrix, packed.bits);
        let remapped_codes = apply_assignment(&codes, &best.symbol_to_code)?;
        let after = unpacked_code_activity(&remapped_codes, packed.bits);
        let before = unpacked_code_activity(&codes, packed.bits);
        let proof = prove_remap(packed, &best)?;
        let remapped = remap_uniform_factor(packed, &best)?;
        let after_u32 = packed_word_activity(&remapped.packed.codes, 32);
        let before_u32 = packed_word_activity(&packed.codes, 32);
        Ok((
            best.clone(),
            json!({
                "identity": json_assignment(&identity, identity.cost),
                "gray": json_assignment(&gray, identity.cost),
                "twos_complement": json_assignment(&twos, identity.cost),
                "sign_magnitude": json_assignment(&signmag, identity.cost),
                "best": json_assignment(&best, identity.cost),
                "code_stream_before": json_code_activity(&before),
                "code_stream_after": json_code_activity(&after),
                "packed_u32_before": json_activity(&before_u32),
                "packed_u32_after": json_activity(&after_u32),
                "proof": proof,
            }),
        ))
    }

    fn binary_activity(packed: &BinaryGroupPacked, name: &str) -> Value {
        let bits = binary_sign_bits(packed);
        let flipped = invert_sign_bytes(&packed.signs, bits.len());
        let mut flipped_bits = Vec::with_capacity(bits.len());
        for flat in 0..bits.len() {
            flipped_bits.push((flipped[flat >> 3] >> (flat & 7)) & 1);
        }
        json!({
            "name": name,
            "codec": "binary_group",
            "shape": [packed.rows, packed.cols],
            "sign_bytes": packed.signs.len(),
            "scale_bytes": packed.scales_f16.len() * 2,
            "kernel_fetch": "row-major sign bytes; serial/simd_bytes/tg256 consume one byte per 8 columns",
            "sign_bits": json_code_activity(&unpacked_code_activity(&bits, 1)),
            "packed": stream_report("sign_bytes", &binary_kernel_sign_bytes(packed), Some((&bits, 1))),
            "scales": stream_report("fp16_scales", &scale_bytes_from_f16(&packed.scales_f16), None),
            "polarity_flip": {
                "free": true,
                "transitions_before": unpacked_code_activity(&bits, 1).transitions,
                "transitions_after": unpacked_code_activity(&flipped_bits, 1).transitions,
                "hamming_unchanged": unpacked_code_activity(&bits, 1).transitions
                    == unpacked_code_activity(&flipped_bits, 1).transitions,
                "note": "The only 1-bit permutation is polarity. Hamming(a,b) = Hamming(~a,~b), so activity is invariant."
            },
        })
    }

    fn rice_activity(packed: &RiceQ1Packed) -> Value {
        json!({
            "name": "up_proj residual",
            "codec": "rice_q1_rms",
            "outlier_count": packed.outlier_count,
            "rice_k": packed.rice_k,
            "rice_bytes": packed.rice_bytes.len(),
            "residual_sign_bytes": packed.residual_signs.len(),
            "kernel_fetch": "serial rice bitstream on bind-time expand (thread 0); residual signs are 1-bit LSB-first",
            "rice_stream": stream_report("rice_bytes", &rice_kernel_bytes(packed), None),
            "residual_signs": stream_report("residual_signs", &packed.residual_signs, None),
            "permutation": {
                "free_codebook": false,
                "reason": "Rice is a variable-length integer code for index diffs, not a centroid codebook. Permuting rice symbols would change bit length (BPW) or break the decoder. Residual signs are 1-bit; polarity flip is Hamming-invariant."
            }
        })
    }

    struct PrivatePipes {
        bound: ComputePipelineState,
        lut: ComputePipelineState,
    }

    fn compile_private(ctx: &MetalContext) -> Result<PrivatePipes, Box<dyn Error>> {
        let opts = CompileOptions::new();
        let lib = ctx
            .device()
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("q80_codec_activity.metal compile failed: {e}"))?;
        let bound_fn = lib
            .get_function("q80_hgravs01_two_stage_matvec_bound", None)
            .map_err(|e| format!("bound kernel: {e}"))?;
        let lut_fn = lib
            .get_function("q80_hgravs01_two_stage_matvec_lut", None)
            .map_err(|e| format!("lut kernel: {e}"))?;
        let bound = ctx
            .device()
            .new_compute_pipeline_state_with_function(&bound_fn)
            .map_err(|e| format!("bound pipeline: {e}"))?;
        let lut = ctx
            .device()
            .new_compute_pipeline_state_with_function(&lut_fn)
            .map_err(|e| format!("lut pipeline: {e}"))?;
        Ok(PrivatePipes { bound, lut })
    }

    fn gpu_ns_private(
        ctx: &MetalContext,
        pipe: &ComputePipelineState,
        grid: (u32, u32, u32),
        encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        label: &str,
    ) -> Result<u64, Box<dyn Error>> {
        let cmd = ctx.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_label(label);
        enc.set_compute_pipeline_state(pipe);
        encode(enc);
        enc.dispatch_threads(
            MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let (start, end) = unsafe {
            use metal::objc::{msg_send, sel, sel_impl};
            let start: f64 = msg_send![cmd, GPUStartTime];
            let end: f64 = msg_send![cmd, GPUEndTime];
            (start, end)
        };
        if start > 0.0 && end > start {
            Ok(((end - start) * 1_000_000_000.0) as u64)
        } else {
            Err(format!("{label}: GPUStartTime/GPUEndTime unavailable").into())
        }
    }

    fn encode_bound(
        enc: &metal::ComputeCommandEncoderRef,
        right_codes: &metal::Buffer,
        right_scales: &metal::Buffer,
        left_codes: &metal::Buffer,
        left_scales: &metal::Buffer,
        input: &metal::Buffer,
        output: &metal::Buffer,
        left: &UniformFactorPacked,
        right: &UniformFactorPacked,
    ) {
        enc.set_buffer(0, Some(right_codes), 0);
        enc.set_buffer(1, Some(right_scales), 0);
        enc.set_buffer(2, Some(left_codes), 0);
        enc.set_buffer(3, Some(left_scales), 0);
        enc.set_buffer(4, Some(input), 0);
        enc.set_buffer(5, Some(output), 0);
        set_u32(enc, 6, right.rows as u32);
        set_u32(enc, 7, right.cols as u32);
        set_u32(enc, 8, left.rows as u32);
        set_u32(enc, 9, left.cols as u32);
        set_u32(enc, 10, left.group_size as u32);
        set_u32(enc, 11, u32::from(left.bits));
        set_u32(enc, 12, u32::from(left.bound));
    }

    fn encode_lut(
        enc: &metal::ComputeCommandEncoderRef,
        right_codes: &metal::Buffer,
        right_scales: &metal::Buffer,
        left_codes: &metal::Buffer,
        left_scales: &metal::Buffer,
        input: &metal::Buffer,
        output: &metal::Buffer,
        right_lut: &metal::Buffer,
        left_lut: &metal::Buffer,
        left: &UniformFactorPacked,
        right: &UniformFactorPacked,
    ) {
        enc.set_buffer(0, Some(right_codes), 0);
        enc.set_buffer(1, Some(right_scales), 0);
        enc.set_buffer(2, Some(left_codes), 0);
        enc.set_buffer(3, Some(left_scales), 0);
        enc.set_buffer(4, Some(input), 0);
        enc.set_buffer(5, Some(output), 0);
        set_u32(enc, 6, right.rows as u32);
        set_u32(enc, 7, right.cols as u32);
        set_u32(enc, 8, left.rows as u32);
        set_u32(enc, 9, left.cols as u32);
        set_u32(enc, 10, left.group_size as u32);
        set_u32(enc, 11, u32::from(left.bits));
        enc.set_buffer(12, Some(right_lut), 0);
        enc.set_buffer(13, Some(left_lut), 0);
    }

    fn run_gpu(
        ctx: &MetalContext,
        left: &UniformFactorPacked,
        right: &UniformFactorPacked,
        left_b: &UniformFactorPacked,
        right_b: &UniformFactorPacked,
        left_lut: &[i32],
        right_lut: &[i32],
        x: &[f32],
        oracle: &[f32],
    ) -> Result<Value, Box<dyn Error>> {
        let pipes = compile_private(ctx)?;
        let grid = tiled_grid(left.rows as u32, 8);
        let right_codes_a = ctx.new_buffer_with_bytes_checked(&right.codes)?;
        let left_codes_a = ctx.new_buffer_with_bytes_checked(&left.codes)?;
        let right_codes_b = ctx.new_buffer_with_bytes_checked(&right_b.codes)?;
        let left_codes_b = ctx.new_buffer_with_bytes_checked(&left_b.codes)?;
        let right_scales = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&right.scales_f16))?;
        let left_scales = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&left.scales_f16))?;
        let x_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(x))?;
        let out = ctx.new_buffer_checked(left.rows * 4)?;
        let left_lut_buf = ctx.new_buffer_with_bytes_checked(&as_u8_i32(left_lut))?;
        let right_lut_buf = ctx.new_buffer_with_bytes_checked(&as_u8_i32(right_lut))?;
        let zeros = vec![0.0f32; left.rows];

        write_f32(&out, &zeros);
        let t = ctx.dispatch_threads_timed(
            "q80_hgravs01_two_stage_matvec",
            grid,
            (256, 1, 1),
            |enc| {
                encode_bound(
                    enc,
                    &right_codes_a,
                    &right_scales,
                    &left_codes_a,
                    &left_scales,
                    &x_buf,
                    &out,
                    left,
                    right,
                )
            },
        )?;
        let _ = require_gpu_ns(t.gpu_duration_us, "prod_warmup_ts")?;
        let prod_err = hawking_core::model::qwen_complete_binary::max_abs_error(
            oracle,
            &read_f32(&out, left.rows),
        );
        if prod_err > HGRAVS_TOL {
            return Err(
                format!("production two_stage max_abs_error={prod_err} > {HGRAVS_TOL}").into(),
            );
        }

        write_f32(&out, &zeros);
        gpu_ns_private(
            ctx,
            &pipes.bound,
            grid,
            |enc| {
                encode_bound(
                    enc,
                    &right_codes_a,
                    &right_scales,
                    &left_codes_a,
                    &left_scales,
                    &x_buf,
                    &out,
                    left,
                    right,
                )
            },
            "bound_correctness",
        )?;
        let bound_gpu = read_f32(&out, left.rows);
        let bound_err =
            hawking_core::model::qwen_complete_binary::max_abs_error(oracle, &bound_gpu);
        if bound_err > HGRAVS_TOL {
            return Err(format!(
                "private bound two_stage max_abs_error={bound_err} > {HGRAVS_TOL}"
            )
            .into());
        }

        write_f32(&out, &zeros);
        gpu_ns_private(
            ctx,
            &pipes.lut,
            grid,
            |enc| {
                encode_lut(
                    enc,
                    &right_codes_b,
                    &right_scales,
                    &left_codes_b,
                    &left_scales,
                    &x_buf,
                    &out,
                    &right_lut_buf,
                    &left_lut_buf,
                    left_b,
                    right_b,
                )
            },
            "lut_correctness",
        )?;
        let lut_gpu = read_f32(&out, left.rows);
        let lut_err = hawking_core::model::qwen_complete_binary::max_abs_error(oracle, &lut_gpu);
        if lut_err > HGRAVS_TOL {
            return Err(
                format!("lut remapped two_stage max_abs_error={lut_err} > {HGRAVS_TOL}").into(),
            );
        }
        let mut n_bit_diff = 0u64;
        let mut gpu_ab_max_abs = 0.0f32;
        for (a, b) in bound_gpu.iter().zip(lut_gpu.iter()) {
            if a.to_bits() != b.to_bits() {
                n_bit_diff += 1;
            }
            gpu_ab_max_abs = gpu_ab_max_abs.max((a - b).abs());
        }
        let bit_identical_private_bound_vs_lut = n_bit_diff == 0;
        if gpu_ab_max_abs > HGRAVS_TOL {
            return Err(format!(
                "private bound vs remapped LUT max_abs={gpu_ab_max_abs} > {HGRAVS_TOL} (n_bit_diff={n_bit_diff})"
            )
            .into());
        }

        for _ in 0..WARMUP {
            write_f32(&out, &zeros);
            let _ = ctx.dispatch_threads_timed(
                "q80_hgravs01_two_stage_matvec",
                grid,
                (256, 1, 1),
                |enc| {
                    encode_bound(
                        enc,
                        &right_codes_a,
                        &right_scales,
                        &left_codes_a,
                        &left_scales,
                        &x_buf,
                        &out,
                        left,
                        right,
                    )
                },
            )?;
            write_f32(&out, &zeros);
            let _ = gpu_ns_private(
                ctx,
                &pipes.lut,
                grid,
                |enc| {
                    encode_lut(
                        enc,
                        &right_codes_b,
                        &right_scales,
                        &left_codes_b,
                        &left_scales,
                        &x_buf,
                        &out,
                        &right_lut_buf,
                        &left_lut_buf,
                        left_b,
                        right_b,
                    )
                },
                "lut_warmup2",
            )?;
        }

        let mut a_ns = Vec::with_capacity(PAIRS);
        let mut b_ns = Vec::with_capacity(PAIRS);
        for _ in 0..PAIRS {
            write_f32(&out, &zeros);
            let t = ctx.dispatch_threads_timed(
                "q80_hgravs01_two_stage_matvec",
                grid,
                (256, 1, 1),
                |enc| {
                    encode_bound(
                        enc,
                        &right_codes_a,
                        &right_scales,
                        &left_codes_a,
                        &left_scales,
                        &x_buf,
                        &out,
                        left,
                        right,
                    )
                },
            )?;
            a_ns.push(require_gpu_ns(t.gpu_duration_us, "A_original")?);
            write_f32(&out, &zeros);
            b_ns.push(gpu_ns_private(
                ctx,
                &pipes.lut,
                grid,
                |enc| {
                    encode_lut(
                        enc,
                        &right_codes_b,
                        &right_scales,
                        &left_codes_b,
                        &left_scales,
                        &x_buf,
                        &out,
                        &right_lut_buf,
                        &left_lut_buf,
                        left_b,
                        right_b,
                    )
                },
                "B_remapped_lut",
            )?);
        }

        let a_med = median(&a_ns);
        let b_med = median(&b_ns);
        Ok(json!({
            "timing_authority": "MTLCommandBuffer.GPUEndTime - GPUStartTime after wait_until_completed",
            "cpu_wall_is_not_gpu_time": true,
            "fallbacks": 0,
            "production_two_stage_max_abs_error": prod_err,
            "private_bound_max_abs_error": bound_err,
            "lut_remapped_max_abs_error": lut_err,
            "gpu_private_bound_vs_lut_bit_identical": bit_identical_private_bound_vs_lut,
            "gpu_private_bound_vs_lut_n_bit_diff": n_bit_diff,
            "gpu_private_bound_vs_lut_max_abs": gpu_ab_max_abs,
            "gpu_vs_cpu_oracle_gate": "numeric_equivalence_two_stage_f32",
            "tolerance": HGRAVS_TOL,
            "pairs": PAIRS,
            "a_original_prod_kernel_ns": a_ns,
            "b_remapped_lut_kernel_ns": b_ns,
            "a_median_ns": a_med,
            "b_median_ns": b_med,
            "token_scale_a_ns": a_med.saturating_mul(LAYERS).saturating_mul(TOP_K),
            "token_scale_b_ns": b_med.saturating_mul(LAYERS).saturating_mul(TOP_K),
            "latency_verdict": if a_med == 0 {
                "no_timing"
            } else if b_med.abs_diff(a_med) * 100 / a_med < 3 {
                "same_speed_within_3pct_lower_switching"
            } else if b_med < a_med {
                "remapped_faster"
            } else {
                "remapped_slower_or_lut_tax"
            },
        }))
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;

        let gate_w = deterministic_matrix(Q80_GATE_ROWS, Q80_GATE_COLS, 1);
        let up_w = deterministic_matrix(Q80_GATE_ROWS, Q80_GATE_COLS, 2);
        let left_w = deterministic_matrix(Q80_DOWN_ROWS, Q80_HGRAVS_RANK, 3);
        let right_w = deterministic_matrix(Q80_HGRAVS_RANK, Q80_DOWN_COLS, 4);
        let x_hidden = deterministic_input(Q80_GATE_COLS);
        let x_swiglu = deterministic_input(Q80_DOWN_COLS);

        let gate = pack_binary_group(&gate_w, Q80_GATE_ROWS, Q80_GATE_COLS, Q80_BINARY_GROUP_SIZE)?;
        let up = pack_binary_rice_q1(
            &up_w,
            Q80_GATE_ROWS,
            Q80_GATE_COLS,
            Q80_RICE_Q1_OUTLIER_RATIO,
        )?;
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

        let _ = binary_group_matvec_f32(&gate, &x_hidden)?;
        let _ = binary_rice_q1_matvec_f32(&up, &x_hidden)?;
        let down_oracle = hgravs01_two_stage_matvec_f32(&left, &right, &x_swiglu)?;

        let (left_assign, left_solve) = solve_factor(&left)?;
        let (right_assign, right_solve) = solve_factor(&right)?;
        let left_remap = remap_uniform_factor(&left, &left_assign)?;
        let right_remap = remap_uniform_factor(&right, &right_assign)?;

        let remapped_oracle = hgravs01_two_stage_lut(
            &left_remap.packed,
            &left_remap.lut,
            &right_remap.packed,
            &right_remap.lut,
            &x_swiglu,
        )?;
        let oracle_bits_match = down_oracle
            .iter()
            .zip(remapped_oracle.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        if !oracle_bits_match {
            return Err("CPU two-stage oracle is not bit-identical after remap+LUT".into());
        }

        let real_scan = scan_real_hgravs(&args.hgravs_dir, args.max_hgravs)?;
        let mut real_slice = Value::Null;
        let mut gpu_pack_kind = "deterministic_q80_geometry";
        let mut gpu_left = left.clone();
        let mut gpu_right = right.clone();
        let mut gpu_left_b = left_remap.packed.clone();
        let mut gpu_right_b = right_remap.packed.clone();
        let mut gpu_left_lut = left_remap.lut.clone();
        let mut gpu_right_lut = right_remap.lut.clone();
        let mut gpu_oracle = down_oracle.clone();
        let mut gpu_x = x_swiglu.clone();

        if let Some(path) = real_scan
            .get("q80_geometry_slice_source")
            .and_then(Value::as_str)
        {
            let path = PathBuf::from(path);
            if path.is_file() {
                let (left_s, right_s, source) = slice_real_q80_factors(&path)?;
                let left_r = pack_uniform_factor(
                    &left_s,
                    Q80_DOWN_ROWS,
                    Q80_HGRAVS_RANK,
                    Q80_HGRAVS_BITS,
                    Q80_HGRAVS_GROUP_SIZE,
                )?;
                let right_r = pack_uniform_factor(
                    &right_s,
                    Q80_HGRAVS_RANK,
                    Q80_DOWN_COLS,
                    Q80_HGRAVS_BITS,
                    Q80_HGRAVS_GROUP_SIZE,
                )?;
                let (la, ls) = solve_factor(&left_r)?;
                let (ra, rs) = solve_factor(&right_r)?;
                let lr = remap_uniform_factor(&left_r, &la)?;
                let rr = remap_uniform_factor(&right_r, &ra)?;
                let oracle = hgravs01_two_stage_matvec_f32(&left_r, &right_r, &x_swiglu)?;
                let remapped =
                    hgravs01_two_stage_lut(&lr.packed, &lr.lut, &rr.packed, &rr.lut, &x_swiglu)?;
                let bits_ok = oracle
                    .iter()
                    .zip(remapped.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                real_slice = json!({
                    "source": source,
                    "note": "Real Q30 down factor sliced to rank 160 / cols 512 and requantized with the Q80 3-bit packer. Values are not the live Q80 pack (that artifact is not on disk). Spatial structure is from a real activation-weighted SVD factor.",
                    "left": ls,
                    "right": rs,
                    "bit_identical_after_remap": bits_ok,
                    "left_activity": factor_activity(&left_r, "real_slice_L")?,
                    "right_activity": factor_activity(&right_r, "real_slice_R")?,
                });
                if bits_ok {
                    gpu_pack_kind = "real_q30_down_sliced_requantized_q80_r160_b3";
                    gpu_left = left_r;
                    gpu_right = right_r;
                    gpu_left_b = lr.packed;
                    gpu_right_b = rr.packed;
                    gpu_left_lut = lr.lut;
                    gpu_right_lut = rr.lut;
                    gpu_oracle = oracle;
                    gpu_x = x_swiglu.clone();
                }
            }
        }

        let mut gpu = json!({
            "ran": false,
            "reason": "skipped",
        });
        if !args.skip_gpu {
            let ctx = MetalContext::new()?;
            gpu = run_gpu(
                &ctx,
                &gpu_left,
                &gpu_right,
                &gpu_left_b,
                &gpu_right_b,
                &gpu_left_lut,
                &gpu_right_lut,
                &gpu_x,
                &gpu_oracle,
            )?;
            gpu.as_object_mut()
                .ok_or("gpu json")?
                .insert("device_name".into(), json!(ctx.device_name()));
            gpu.as_object_mut()
                .unwrap()
                .insert("pack_kind".into(), json!(gpu_pack_kind));
        }

        let a_med = gpu.get("a_median_ns").and_then(Value::as_u64).unwrap_or(0);
        let b_med = gpu.get("b_median_ns").and_then(Value::as_u64).unwrap_or(0);
        let token_a = a_med.saturating_mul(LAYERS).saturating_mul(TOP_K);
        let token_b = b_med.saturating_mul(LAYERS).saturating_mul(TOP_K);

        let receipt = json!({
            "schema": SCHEMA,
            "lane": "codec-switching-activity",
            "status": "ACTIVITY_MEASURED_PERMUTATION_IS_FREE_LATENCY_SEPARATE",
            "measurement_label": "DIRTY_ENGINEERING",
            "baseline_ns_per_token": token_a,
            "baseline_label": "DIRTY_ENGINEERING",
            "result_ns_per_token": token_b,
            "result_label": "DIRTY_ENGINEERING",
            "energy_claim": {
                "kind": "DERIVED_not_measured_joules",
                "primary_metric": "transition_count_reduction",
                "powermetrics": "unavailable_no_root",
                "note": "Energy is inferred from CMOS P = alpha C V^2 f. Only alpha is software-controlled here. No joule number is claimed."
            },
            "fetch_order": {
                "authority": "crates/hawking-core/shaders/q80_mixed_decode.metal",
                "gate_up_signs": "row-major; one byte per 8 consecutive columns after alignment. simd/tg256 coalesce to sequential sign bytes of the row.",
                "hgravs_codes": "element = row*cols+col, 3-bit LSB-first extract. simd 32-wide tiles, 8 rows/TG. Packed bits are sequential in that element order.",
                "rice": "serial bitstream, thread 0, bind-time expand.",
                "not_file_layout": "JSON envelopes are stripped before bind. Activity is on kernel-facing bodies."
            },
            "q80_synthetic": {
                "note": "Same deterministic pack the Q80 mixed-decode kernels already grade against. Not a BF16 parent slice.",
                "gate": binary_activity(&gate, "gate_proj"),
                "up_binary": binary_activity(&up.binary, "up_proj binary base"),
                "up_rice": rice_activity(&up),
                "down_left": factor_activity(&left, "down_proj L")?,
                "down_right": factor_activity(&right, "down_proj R")?,
                "down_left_assignment": left_solve,
                "down_right_assignment": right_solve,
                "cpu_oracle_bit_identical_after_remap": oracle_bits_match,
            },
            "other_free_variables": {
                "binary_polarity": "free and Hamming-invariant — not useful",
                "rice_symbols": "not a free codebook; would change BPW",
                "gray_of_quant_levels": "a specific assignment; reported next to the empirical optimum",
                "sign_magnitude_vs_twos": "also specific assignments of the same 3-bit alphabet; decode formula changes; values stay identical if the LUT matches",
                "bit_plane_msb_vs_lsb": "free if extract matches; can change u32-word Hamming because 3-bit codes straddle words. Intra-word bit reversal of whole words cannot.",
                "group_scale_packing": "kernel fetches scales[element/group_size] in group-index order, which is storage order. Permuting storage and the index together fetches the same scale words in the same order, so activity is unchanged. Changing grouping would change values (not free)."
            },
            "real_q30_hgravs01": real_scan,
            "real_q30_sliced_to_q80": real_slice,
            "gpu": gpu,
            "claim_boundary": {
                "artifact_packed_q80_mixed": false,
                "joules_measured": false,
                "weight_values_changed": false,
                "bpw_changed": false,
                "coherence_generation_tested": false,
                "decode_kernel_exists": true,
                "lut_not_in_production_library": true,
                "note": "The permutation is free on the value/BPW axes. Shipping it on the token path requires a LUT (or Gray) decode in the production kernel. This lane did not replace the shipped q = code - bound extract."
            },
            "next_bottleneck": {
                "what": "down_proj two-stage factor matvec occupancy/dispatch, not switching activity",
                "why": "Q80 mixed is already known dispatch/host bound (~0.79% of 700-800 GB/s). A free alpha reduction does not move that wall unless the bus is contention-limited, which this GPU timing tests.",
                "measured_ns": b_med,
            }
        });

        if let Some(path) = args.out {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, serde_json::to_vec_pretty(&receipt)?)?;
            eprintln!("wrote {}", path.display());
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        Ok(())
    }

    fn hgravs01_two_stage_lut(
        left: &UniformFactorPacked,
        left_lut: &[i32],
        right: &UniformFactorPacked,
        right_lut: &[i32],
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if left.cols != right.rows {
            return Err("rank disagree".into());
        }
        if right.cols != input.len() {
            return Err("input cols disagree".into());
        }
        let mid = matvec_lut(right, right_lut, input)?;
        Ok(matvec_lut(left, left_lut, &mid)?)
    }

    fn matvec_lut(
        packed: &UniformFactorPacked,
        lut: &[i32],
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let count = packed.groups * packed.group_size;
        let codes = unpack_unsigned_lsb(&packed.codes, count, packed.bits)?;
        let mut output = vec![0.0f32; packed.rows];
        for row in 0..packed.rows {
            let mut sum = 0.0f32;
            for col in 0..packed.cols {
                let element = row * packed.cols + col;
                let group = element / packed.group_size;
                let scale = half::f16::from_bits(packed.scales_f16[group]).to_f32();
                let q = lut[codes[element] as usize] as f32;
                sum += q * scale * input[col];
            }
            output[row] = sum;
        }
        Ok(output)
    }
}
