//! Q80 mixed-representation Metal decode-kernel parity + GPU timing.
//!
//! Closes `decode_kernel_exists` for the three mixed organs:
//!   gate_proj  binary_group
//!   up_proj    binary + rice_q1_rms @ 2%
//!   down_proj  hgravs01_r160_b3 two-stage
//!
//! Grade is against the packed-artifact CPU oracle, never the BF16 parent.
//! Dense W is never materialized on the token path.
//!
//! Build (this worktree):
//!   cargo build --profile release-fast -p hawking-core --example ascension_qwen80_mixed_decode_kernel_parity --target-dir workspace/ops/build/rust
//! Run under the GPU mutex:
//!   ./tools/gpu_lane_lock.sh q80-decode-kernels \
//!     workspace/ops/build/rust/release-fast/examples/ascension_qwen80_mixed_decode_kernel_parity \
//!     --out receipts/ascent-2026-08-16/q80-decode-kernels.json

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
        binary_group_matvec_f32, binary_rice_q1_matvec_f32, deterministic_input,
        deterministic_matrix, expand_rice_indices, hgravs01_two_stage_matvec_f32, max_abs_error,
        pack_binary_group, pack_binary_rice_q1, pack_uniform_factor, rice_q1_row_ptr,
        BinaryGroupPacked, RiceQ1Packed, UniformFactorPacked, Q80_BINARY_GROUP_SIZE,
        Q80_DOWN_COLS, Q80_DOWN_ROWS, Q80_GATE_COLS, Q80_GATE_ROWS, Q80_HGRAVS_BITS,
        Q80_HGRAVS_GROUP_SIZE, Q80_HGRAVS_RANK, Q80_RICE_Q1_OUTLIER_RATIO,
    };
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    const SCHEMA: &str = "hawking.ascension.qwen80_mixed_decode_kernel_parity.v1";
    const WARMUP: usize = 2;
    const MEASURED: usize = 6;
    const BINARY_TOL: f32 = 2e-5;
    const RICE_EXPAND_TOL: f32 = 0.0;
    const UP_TOL: f32 = 2e-5;
    const HGRAVS_TOL: f32 = 2e-5;
    const LAYERS: u64 = 48;
    const TOP_K: u64 = 10;

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

    fn as_u8_u16(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn as_u8_f32(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
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

    fn read_u32(buffer: &metal::Buffer, n: usize) -> Vec<u32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, n).to_vec() }
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

    fn encode_binary(
        enc: &metal::ComputeCommandEncoderRef,
        signs: &metal::Buffer,
        scales: &metal::Buffer,
        input: &metal::Buffer,
        output: &metal::Buffer,
        packed: &BinaryGroupPacked,
    ) {
        enc.set_buffer(0, Some(signs), 0);
        enc.set_buffer(1, Some(scales), 0);
        enc.set_buffer(2, Some(input), 0);
        enc.set_buffer(3, Some(output), 0);
        set_u32(enc, 4, packed.rows as u32);
        set_u32(enc, 5, packed.cols as u32);
        set_u32(enc, 6, packed.group_size as u32);
        set_u32(enc, 7, packed.groups_per_row as u32);
    }

    fn encode_csr_apply(
        enc: &metal::ComputeCommandEncoderRef,
        indices: &metal::Buffer,
        row_ptr: &metal::Buffer,
        signs: &metal::Buffer,
        input: &metal::Buffer,
        output: &metal::Buffer,
        packed: &RiceQ1Packed,
    ) {
        enc.set_buffer(0, Some(indices), 0);
        enc.set_buffer(1, Some(row_ptr), 0);
        enc.set_buffer(2, Some(signs), 0);
        enc.set_buffer(3, Some(input), 0);
        enc.set_buffer(4, Some(output), 0);
        set_u32(enc, 5, packed.binary.rows as u32);
        set_u32(enc, 6, packed.binary.cols as u32);
        set_u32(enc, 7, u32::from(packed.residual_scale_f16));
    }

    fn encode_rice_apply(
        enc: &metal::ComputeCommandEncoderRef,
        rice: &metal::Buffer,
        signs: &metal::Buffer,
        input: &metal::Buffer,
        output: &metal::Buffer,
        packed: &RiceQ1Packed,
    ) {
        enc.set_buffer(0, Some(rice), 0);
        enc.set_buffer(1, Some(signs), 0);
        enc.set_buffer(2, Some(input), 0);
        enc.set_buffer(3, Some(output), 0);
        set_u32(enc, 4, packed.first_index);
        set_u32(enc, 5, packed.rice_k);
        set_u32(enc, 6, packed.rice_bytes.len() as u32);
        set_u32(enc, 7, packed.outlier_count as u32);
        set_u32(enc, 8, packed.binary.cols as u32);
        set_u32(enc, 9, u32::from(packed.residual_scale_f16));
    }

    fn encode_factor(
        enc: &metal::ComputeCommandEncoderRef,
        codes: &metal::Buffer,
        scales: &metal::Buffer,
        input: &metal::Buffer,
        output: &metal::Buffer,
        packed: &UniformFactorPacked,
    ) {
        enc.set_buffer(0, Some(codes), 0);
        enc.set_buffer(1, Some(scales), 0);
        enc.set_buffer(2, Some(input), 0);
        enc.set_buffer(3, Some(output), 0);
        set_u32(enc, 4, packed.rows as u32);
        set_u32(enc, 5, packed.cols as u32);
        set_u32(enc, 6, packed.group_size as u32);
        set_u32(enc, 7, u32::from(packed.bits));
        set_u32(enc, 8, u32::from(packed.bound));
    }

    fn require_gpu_ns(gpu_duration_us: Option<u64>, label: &str) -> Result<u64, Box<dyn Error>> {
        let us = gpu_duration_us.ok_or_else(|| {
            format!("{label}: MTLCommandBuffer GPUStartTime/GPUEndTime unavailable")
        })?;
        Ok(us.saturating_mul(1000))
    }

    fn spread(values: &[u64]) -> Value {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        json!({
            "n": n,
            "all_ns": values,
            "min_ns": sorted.first().copied(),
            "max_ns": sorted.last().copied(),
            "median_ns": if n == 0 { None } else { Some(sorted[n / 2]) },
        })
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let ctx = MetalContext::new()?;

        let gate_w = deterministic_matrix(Q80_GATE_ROWS, Q80_GATE_COLS, 1);
        let up_w = deterministic_matrix(Q80_GATE_ROWS, Q80_GATE_COLS, 2);
        let left_w = deterministic_matrix(Q80_DOWN_ROWS, Q80_HGRAVS_RANK, 3);
        let right_w = deterministic_matrix(Q80_HGRAVS_RANK, Q80_DOWN_COLS, 4);
        let x_hidden = deterministic_input(Q80_GATE_COLS);
        let x_swiglu = deterministic_input(Q80_DOWN_COLS);

        let gate = pack_binary_group(&gate_w, Q80_GATE_ROWS, Q80_GATE_COLS, Q80_BINARY_GROUP_SIZE)?;
        let up = pack_binary_rice_q1(&up_w, Q80_GATE_ROWS, Q80_GATE_COLS, Q80_RICE_Q1_OUTLIER_RATIO)?;
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

        let gate_oracle = binary_group_matvec_f32(&gate, &x_hidden)?;
        let up_oracle = binary_rice_q1_matvec_f32(&up, &x_hidden)?;
        let down_oracle = hgravs01_two_stage_matvec_f32(&left, &right, &x_swiglu)?;
        let rice_indices_oracle = expand_rice_indices(&up)?;

        let gate_signs = ctx.new_buffer_with_bytes_checked(&gate.signs)?;
        let gate_scales_bytes = as_u8_u16(&gate.scales_f16);
        let gate_scales = ctx.new_buffer_with_bytes_checked(&gate_scales_bytes)?;
        let up_signs = ctx.new_buffer_with_bytes_checked(&up.binary.signs)?;
        let up_scales_bytes = as_u8_u16(&up.binary.scales_f16);
        let up_scales = ctx.new_buffer_with_bytes_checked(&up_scales_bytes)?;
        let rice_bytes = if up.rice_bytes.is_empty() {
            ctx.new_buffer_checked(4)?
        } else {
            ctx.new_buffer_with_bytes_checked(&up.rice_bytes)?
        };
        let residual_signs = ctx.new_buffer_with_bytes_checked(&up.residual_signs)?;
        let left_codes = ctx.new_buffer_with_bytes_checked(&left.codes)?;
        let left_scales_bytes = as_u8_u16(&left.scales_f16);
        let left_scales = ctx.new_buffer_with_bytes_checked(&left_scales_bytes)?;
        let right_codes = ctx.new_buffer_with_bytes_checked(&right.codes)?;
        let right_scales_bytes = as_u8_u16(&right.scales_f16);
        let right_scales = ctx.new_buffer_with_bytes_checked(&right_scales_bytes)?;
        let x_hidden_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&x_hidden))?;
        let x_swiglu_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&x_swiglu))?;
        let gate_out = ctx.new_buffer_checked(Q80_GATE_ROWS * 4)?;
        let up_out = ctx.new_buffer_checked(Q80_GATE_ROWS * 4)?;
        let mid_out = ctx.new_buffer_checked(Q80_HGRAVS_RANK * 4)?;
        let down_out = ctx.new_buffer_checked(Q80_DOWN_ROWS * 4)?;
        let rice_idx_out = ctx.new_buffer_checked(up.outlier_count * 4)?;
        let row_ptr = rice_q1_row_ptr(&rice_indices_oracle, up.binary.rows, up.binary.cols)?;
        let row_ptr_bytes: Vec<u8> = row_ptr.iter().flat_map(|v| v.to_le_bytes()).collect();
        let row_ptr_buf = ctx.new_buffer_with_bytes_checked(&row_ptr_bytes)?;

        // ── correctness (not timed) ──────────────────────────────────────
        write_f32(&gate_out, &vec![0.0f32; Q80_GATE_ROWS]);
        let gate_parity = ctx.dispatch_threads_timed(
            "q80_binary_group_matvec",
            (gate.rows as u32, 1, 1),
            (256, 1, 1),
            |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
        )?;
        let gate_got = read_f32(&gate_out, gate.rows);
        let gate_err = max_abs_error(&gate_oracle, &gate_got);
        if gate_err > BINARY_TOL {
            return Err(format!("gate binary_group max_abs_error={gate_err} > {BINARY_TOL}").into());
        }

        write_f32(&up_out, &vec![0.0f32; Q80_GATE_ROWS]);
        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads(
                "q80_binary_group_matvec",
                (up.binary.rows as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    encode_binary(
                        enc,
                        &up_signs,
                        &up_scales,
                        &x_hidden_buf,
                        &up_out,
                        &up.binary,
                    )
                },
            )?;
            batch.dispatch_threads(
                "q80_rice_q1_residual_apply",
                (32, 1, 1),
                (32, 1, 1),
                |enc| {
                    encode_rice_apply(
                        enc,
                        &rice_bytes,
                        &residual_signs,
                        &x_hidden_buf,
                        &up_out,
                        &up,
                    )
                },
            )?;
            Ok(())
        })?;
        let up_got = read_f32(&up_out, up.binary.rows);
        let up_err = max_abs_error(&up_oracle, &up_got);
        if up_err > UP_TOL {
            return Err(format!("up binary+rice_q1 max_abs_error={up_err} > {UP_TOL}").into());
        }

        ctx.dispatch_threads_timed(
            "q80_rice_q1_expand_indices",
            (32, 1, 1),
            (32, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&rice_bytes), 0);
                enc.set_buffer(1, Some(&rice_idx_out), 0);
                set_u32(enc, 2, up.first_index);
                set_u32(enc, 3, up.rice_k);
                set_u32(enc, 4, up.rice_bytes.len() as u32);
                set_u32(enc, 5, up.outlier_count as u32);
            },
        )?;
        let rice_got = read_u32(&rice_idx_out, up.outlier_count);
        if rice_got != rice_indices_oracle {
            return Err("rice expand indices are not bit-identical to the CPU oracle".into());
        }

        write_f32(&up_out, &vec![0.0f32; Q80_GATE_ROWS]);
        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads(
                "q80_binary_group_matvec",
                (up.binary.rows as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    encode_binary(
                        enc,
                        &up_signs,
                        &up_scales,
                        &x_hidden_buf,
                        &up_out,
                        &up.binary,
                    )
                },
            )?;
            batch.dispatch_threads(
                "q80_sparse_q1_apply_csr",
                (up.binary.rows as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    encode_csr_apply(
                        enc,
                        &rice_idx_out,
                        &row_ptr_buf,
                        &residual_signs,
                        &x_hidden_buf,
                        &up_out,
                        &up,
                    )
                },
            )?;
            Ok(())
        })?;
        let up_csr_got = read_f32(&up_out, up.binary.rows);
        let up_csr_err = max_abs_error(&up_oracle, &up_csr_got);
        if up_csr_err > UP_TOL {
            return Err(format!("up binary+csr residual max_abs_error={up_csr_err} > {UP_TOL}").into());
        }

        write_f32(&mid_out, &vec![0.0f32; Q80_HGRAVS_RANK]);
        write_f32(&down_out, &vec![0.0f32; Q80_DOWN_ROWS]);
        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads(
                "q80_hgravs01_factor_matvec",
                (right.rows as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    encode_factor(enc, &right_codes, &right_scales, &x_swiglu_buf, &mid_out, &right)
                },
            )?;
            batch.dispatch_threads(
                "q80_hgravs01_factor_matvec",
                (left.rows as u32, 1, 1),
                (256, 1, 1),
                |enc| encode_factor(enc, &left_codes, &left_scales, &mid_out, &down_out, &left),
            )?;
            Ok(())
        })?;
        let down_got = read_f32(&down_out, left.rows);
        let down_err = max_abs_error(&down_oracle, &down_got);
        if down_err > HGRAVS_TOL {
            return Err(format!("down hgravs01 max_abs_error={down_err} > {HGRAVS_TOL}").into());
        }

        // ── warmup (not reported) ────────────────────────────────────────
        for _ in 0..WARMUP {
            let _ = ctx.dispatch_threads_timed(
                "q80_binary_group_matvec",
                (gate.rows as u32, 1, 1),
                (256, 1, 1),
                |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
            )?;
            ctx.dispatch_batch(|batch| {
                batch.dispatch_threads(
                    "q80_binary_group_matvec",
                    (up.binary.rows as u32, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        encode_binary(
                            enc,
                            &up_signs,
                            &up_scales,
                            &x_hidden_buf,
                            &up_out,
                            &up.binary,
                        )
                    },
                )?;
                batch.dispatch_threads(
                    "q80_sparse_q1_apply_csr",
                    (up.binary.rows as u32, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        encode_csr_apply(
                            enc,
                            &rice_idx_out,
                            &row_ptr_buf,
                            &residual_signs,
                            &x_hidden_buf,
                            &up_out,
                            &up,
                        )
                    },
                )?;
                Ok(())
            })?;
            ctx.dispatch_batch(|batch| {
                batch.dispatch_threads(
                    "q80_hgravs01_factor_matvec",
                    (right.rows as u32, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        encode_factor(
                            enc,
                            &right_codes,
                            &right_scales,
                            &x_swiglu_buf,
                            &mid_out,
                            &right,
                        )
                    },
                )?;
                batch.dispatch_threads(
                    "q80_hgravs01_factor_matvec",
                    (left.rows as u32, 1, 1),
                    (256, 1, 1),
                    |enc| encode_factor(enc, &left_codes, &left_scales, &mid_out, &down_out, &left),
                )?;
                Ok(())
            })?;
        }

        // ── alternating measured reps: gate, up, down, … ─────────────────
        let mut gate_ns = Vec::with_capacity(MEASURED);
        let mut up_ns = Vec::with_capacity(MEASURED);
        let mut down_ns = Vec::with_capacity(MEASURED);
        for _ in 0..MEASURED {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_matvec",
                (gate.rows as u32, 1, 1),
                (256, 1, 1),
                |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
            )?;
            gate_ns.push(require_gpu_ns(t.gpu_duration_us, "gate")?);

            let t = ctx.dispatch_batch_timed(|batch| {
                batch.dispatch_threads(
                    "q80_binary_group_matvec",
                    (up.binary.rows as u32, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        encode_binary(
                            enc,
                            &up_signs,
                            &up_scales,
                            &x_hidden_buf,
                            &up_out,
                            &up.binary,
                        )
                    },
                )?;
                batch.dispatch_threads(
                    "q80_sparse_q1_apply_csr",
                    (up.binary.rows as u32, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        encode_csr_apply(
                            enc,
                            &rice_idx_out,
                            &row_ptr_buf,
                            &residual_signs,
                            &x_hidden_buf,
                            &up_out,
                            &up,
                        )
                    },
                )?;
                Ok(())
            })?;
            up_ns.push(require_gpu_ns(t.gpu_duration_us, "up")?);

            let t = ctx.dispatch_batch_timed(|batch| {
                batch.dispatch_threads(
                    "q80_hgravs01_factor_matvec",
                    (right.rows as u32, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        encode_factor(
                            enc,
                            &right_codes,
                            &right_scales,
                            &x_swiglu_buf,
                            &mid_out,
                            &right,
                        )
                    },
                )?;
                batch.dispatch_threads(
                    "q80_hgravs01_factor_matvec",
                    (left.rows as u32, 1, 1),
                    (256, 1, 1),
                    |enc| encode_factor(enc, &left_codes, &left_scales, &mid_out, &down_out, &left),
                )?;
                Ok(())
            })?;
            down_ns.push(require_gpu_ns(t.gpu_duration_us, "down")?);
        }

        let mut gate_sorted = gate_ns.clone();
        gate_sorted.sort_unstable();
        let mut up_sorted = up_ns.clone();
        up_sorted.sort_unstable();
        let mut down_sorted = down_ns.clone();
        down_sorted.sort_unstable();
        let gate_med = gate_sorted[MEASURED / 2];
        let up_med = up_sorted[MEASURED / 2];
        let down_med = down_sorted[MEASURED / 2];
        let organ_sum = gate_med + up_med + down_med;
        let routed_token_ns = organ_sum.saturating_mul(LAYERS).saturating_mul(TOP_K);

        // One rice-serial sample so the bind-time expand decision is measured,
        // not assumed. Not part of the per-token number.
        write_f32(&up_out, &vec![0.0f32; Q80_GATE_ROWS]);
        let rice_serial = ctx.dispatch_batch_timed(|batch| {
            batch.dispatch_threads(
                "q80_binary_group_matvec",
                (up.binary.rows as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    encode_binary(
                        enc,
                        &up_signs,
                        &up_scales,
                        &x_hidden_buf,
                        &up_out,
                        &up.binary,
                    )
                },
            )?;
            batch.dispatch_threads(
                "q80_rice_q1_residual_apply",
                (32, 1, 1),
                (32, 1, 1),
                |enc| {
                    encode_rice_apply(
                        enc,
                        &rice_bytes,
                        &residual_signs,
                        &x_hidden_buf,
                        &up_out,
                        &up,
                    )
                },
            )?;
            Ok(())
        })?;
        let rice_serial_ns = require_gpu_ns(rice_serial.gpu_duration_us, "up_rice_serial")?;

        let gate_bytes = gate.signs.len() + gate.scales_f16.len() * 2;
        let up_bytes = up.binary.signs.len()
            + up.binary.scales_f16.len() * 2
            + up.rice_bytes.len()
            + 4
            + 2
            + up.residual_signs.len();
        let down_bytes = left.codes.len()
            + left.scales_f16.len() * 2
            + right.codes.len()
            + right.scales_f16.len() * 2;
        let mid_temp_bytes = Q80_HGRAVS_RANK * 4;

        let receipt = json!({
            "schema": SCHEMA,
            "status": "DECODE_KERNEL_EXISTS_COMPONENT_PARITY_NOT_RUNTIME_OR_COHERENCE",
            "lane": "q80-decode-kernels",
            "baseline_ns_per_token": 403000000,
            "baseline_label": "DIRTY_ENGINEERING",
            "baseline_source": "receipts/ascent-2026-08-16/Q80_BASELINE_2026_08_16.json full-token uniform-Q4",
            "result_ns_per_token": routed_token_ns,
            "result_label": "DIRTY_ENGINEERING",
            "result_scope": "48*10*(gate+up+down) routed mixed-decode GPU only",
            "next_bottleneck": {
                "organ": "down_proj hgravs01 two-stage factor matvec",
                "median_ns": down_med,
                "token_scale_ns": down_med.saturating_mul(LAYERS).saturating_mul(TOP_K),
                "why": "serial per-row 3-bit factor decode; R is 160x512 then L is 2048x160. Two dispatches, 640-byte mid."
            },
            "measurement_label": "DIRTY_ENGINEERING",
            "timing_authority": "MTLCommandBuffer.GPUEndTime - GPUStartTime after wait_until_completed",
            "device_name": ctx.device_name(),
            "geometry": {
                "gate_up_shape": [Q80_GATE_ROWS, Q80_GATE_COLS],
                "down_shape": [Q80_DOWN_ROWS, Q80_DOWN_COLS],
                "hgravs_rank": Q80_HGRAVS_RANK,
                "hgravs_bits": Q80_HGRAVS_BITS,
                "hgravs_group_size": Q80_HGRAVS_GROUP_SIZE,
                "binary_group_size": Q80_BINARY_GROUP_SIZE,
                "rice_outlier_ratio": Q80_RICE_Q1_OUTLIER_RATIO,
                "rice_outlier_count": up.outlier_count,
                "rice_k": up.rice_k,
                "note": "Qwen3-Coder-Next routed-expert geometry. Weights are a deterministic packed artifact, not a BF16 parent slice. Kernel is graded against the artifact oracle."
            },
            "pack_lane_contract": {
                "gate_proj": {
                    "codec": "binary_group",
                    "container_magic": "HGRAVB01",
                    "schema": "hawking.gravity.binary_sign_scale.v1",
                    "kernel": "q80_binary_group_matvec",
                    "body": "fp16 scales[groups] || LSB-first sign bits",
                    "group_size": 128
                },
                "up_proj": {
                    "codec": "binary + rice_q1_rms sparse residual @2%",
                    "container_magic": "HGRAVR02",
                    "schema": "hawking.gravity.binary_outlier_residual.v2",
                    "kernels": [
                        "q80_binary_group_matvec",
                        "q80_rice_q1_expand_indices",
                        "q80_sparse_q1_apply_csr",
                        "q80_rice_q1_residual_apply"
                    ],
                    "body": "binary scales+signs || u32 first_index || rice(diffs) || fp16 rms || 1-bit residual signs",
                    "rice": "unary quotient of 1-bits, then 0, then k LSBs; LSB-first bitstream"
                },
                "down_proj": {
                    "codec": "hgravs01_r160_b3",
                    "container_magic": "HGRAVS01",
                    "schema": "hawking.gravity.activation_weighted_svd_low_rank.v1",
                    "kernel": "q80_hgravs01_factor_matvec (R then L)",
                    "execution": "y = L @ (R @ x); mid[rank] only temporary",
                    "factor_body": "fp16 scales[groups] || packed unsigned codes",
                    "bits": 3,
                    "group_size": 64,
                    "rank": 160
                },
                "must_not": "packed -> dense (rows x cols) reconstruction -> matvec",
                "note": "Pack lane must emit these containers. Kernels consume bodies after magic+header."
            },
            "correctness": {
                "gate": {
                    "gate_kind": if gate_err == 0.0 { "bit_identical_f32" } else { "numeric_equivalence_serial_f32" },
                    "max_abs_error": gate_err,
                    "tolerance": BINARY_TOL,
                    "passed": gate_err <= BINARY_TOL
                },
                "up": {
                    "gate_kind": "numeric_equivalence_serial_f32",
                    "rice_apply_max_abs_error": up_err,
                    "csr_apply_max_abs_error": up_csr_err,
                    "tolerance": UP_TOL,
                    "passed": up_err <= UP_TOL && up_csr_err <= UP_TOL,
                    "note": "rice apply and CSR apply are both graded against the same artifact oracle. CSR add order matches sorted-index serial apply."
                },
                "rice_expand": {
                    "gate_kind": "bit_identical_u32_indices",
                    "passed": true,
                    "tolerance": RICE_EXPAND_TOL
                },
                "down": {
                    "gate_kind": "numeric_equivalence_two_stage_f32",
                    "max_abs_error": down_err,
                    "tolerance": HGRAVS_TOL,
                    "passed": down_err <= HGRAVS_TOL,
                    "oracle": "two_stage_L_R_x_of_packed_factors_not_dense_W"
                },
                "fallbacks": 0,
                "dense_reconstruction_on_token_path": false
            },
            "cost": {
                "gate_proj": {
                    "decode_ops_per_weight": 1,
                    "dram_bytes_this_dispatch": gate_bytes + Q80_GATE_COLS * 4 + Q80_GATE_ROWS * 4,
                    "packed_weight_bytes": gate_bytes,
                    "temporary_bytes": 0,
                    "dispatches": 1,
                    "fuses_with_surrounding_op": false,
                    "gpu_ns": spread(&gate_ns),
                    "median_ns_per_token_this_organ": gate_med
                },
                "up_proj": {
                    "decode_ops_per_weight": "1 binary + 0.02 residual apply",
                    "per_token_kernels": ["q80_binary_group_matvec", "q80_sparse_q1_apply_csr"],
                    "dram_bytes_this_dispatch": up_bytes + up.outlier_count * 4 + (up.binary.rows + 1) * 4 + Q80_GATE_COLS * 4 + Q80_GATE_ROWS * 4,
                    "packed_weight_bytes": up_bytes,
                    "temporary_bytes": up.outlier_count * 4 + (up.binary.rows + 1) * 4,
                    "temporary_kind": "bind-time expanded u32 indices + CSR row_ptr; not dense W",
                    "dispatches": 2,
                    "fuses_with_surrounding_op": false,
                    "gpu_ns": spread(&up_ns),
                    "median_ns_per_token_this_organ": up_med,
                    "rice_per_token_serial_ns_one_shot": rice_serial_ns,
                    "rice_per_token_serial_rejected_as_token_path": true,
                    "rice_expand_role": "bind-time only; q80_rice_q1_expand_indices is format-pure and bit-identical"
                },
                "down_proj": {
                    "decode_ops_per_factor_element": 1,
                    "dram_bytes_this_dispatch": down_bytes + Q80_DOWN_COLS * 4 + mid_temp_bytes + Q80_DOWN_ROWS * 4,
                    "packed_weight_bytes": down_bytes,
                    "temporary_bytes": mid_temp_bytes,
                    "temporary_kind": "mid[rank=160] f32, not dense W",
                    "dispatches": 2,
                    "fuses_with_surrounding_op": false,
                    "gpu_ns": spread(&down_ns),
                    "median_ns_per_token_this_organ": down_med
                },
                "routed_expert_wave_estimate": {
                    "layers": LAYERS,
                    "top_k": TOP_K,
                    "organs_per_expert": 3,
                    "formula": "48 * 10 * (gate_med + up_med + down_med)",
                    "median_ns_per_token": routed_token_ns,
                    "label": "DIRTY_ENGINEERING",
                    "claim_boundary": "routed-expert mixed-decode GPU time only; excludes attention, shared expert, router, host, and dispatch of a full token graph"
                }
            },
            "reps": {
                "protocol": "warmup=2 then 6 interleaved (gate, up, down) pairs",
                "gate_gpu_ns": gate_ns,
                "up_gpu_ns": up_ns,
                "down_gpu_ns": down_ns
            },
            "parity_warmup_gpu_ns_not_a_claim": {
                "gate_first_dispatch_us": gate_parity.gpu_duration_us,
                "note": "first dispatch includes possible pipeline compile; excluded from measured reps"
            },
            "claim_boundary": {
                "decode_kernel_exists": true,
                "artifact_packed": false,
                "coherence_generation_tested": false,
                "not_full_token_runtime": true,
                "not_bf16_parent_parity": true,
                "cpu_wall_is_not_gpu_time": true
            }
        });

        let pretty = serde_json::to_string_pretty(&receipt)?;
        println!("{pretty}");
        if let Some(path) = args.out {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, pretty)?;
        }
        Ok(())
    }
}
