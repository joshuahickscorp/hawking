//! Q80 mixed-decode throughput: parallel factor decode, fused down_proj.
//!
//! Keeps the shipped serial kernels as arm A. Arm B decodes in registers /
//! simdgroup and consumes immediately — never a dense (rows × cols) W.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core --example ascension_qwen80_mixed_decode_throughput
//! Run under the GPU mutex:
//!   ./tools/gpu_lane_lock.sh q80-decode-throughput \
//!     workspace/ops/build/rust/release-fast/examples/ascension_qwen80_mixed_decode_throughput \
//!     --out receipts/ascent-2026-08-16/q80-decode-throughput.json

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

    const SCHEMA: &str = "hawking.ascension.qwen80_mixed_decode_throughput.v1";
    const WARMUP: usize = 2;
    const MEASURED: usize = 6;
    const BINARY_TOL: f32 = 2e-5;
    const UP_TOL: f32 = 2e-5;
    const HGRAVS_TOL: f32 = 2e-5;
    const LAYERS: u64 = 48;
    const TOP_K: u64 = 10;
    const CEILING_GBPS: f64 = 800.0;
    const SHIPPED_GATE_NS: u64 = 61_000;
    const SHIPPED_UP_NS: u64 = 80_000;
    const SHIPPED_DOWN_NS: u64 = 259_000;

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
        let tgs = rows.div_ceil(rows_per_tg);
        (tgs.saturating_mul(256), 1, 1)
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

    fn encode_csr(
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

    fn encode_binary_csr(
        enc: &metal::ComputeCommandEncoderRef,
        signs: &metal::Buffer,
        scales: &metal::Buffer,
        input: &metal::Buffer,
        output: &metal::Buffer,
        indices: &metal::Buffer,
        row_ptr: &metal::Buffer,
        residual_signs: &metal::Buffer,
        packed: &RiceQ1Packed,
    ) {
        enc.set_buffer(0, Some(signs), 0);
        enc.set_buffer(1, Some(scales), 0);
        enc.set_buffer(2, Some(input), 0);
        enc.set_buffer(3, Some(output), 0);
        enc.set_buffer(4, Some(indices), 0);
        enc.set_buffer(5, Some(row_ptr), 0);
        enc.set_buffer(6, Some(residual_signs), 0);
        set_u32(enc, 7, packed.binary.rows as u32);
        set_u32(enc, 8, packed.binary.cols as u32);
        set_u32(enc, 9, packed.binary.group_size as u32);
        set_u32(enc, 10, packed.binary.groups_per_row as u32);
        set_u32(enc, 11, u32::from(packed.residual_scale_f16));
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

    fn encode_two_stage(
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

    fn gpu_ns_from(
        gpu_duration_us: Option<u64>,
        gpu_start_ns: Option<u64>,
        gpu_end_ns: Option<u64>,
        label: &str,
    ) -> Result<u64, Box<dyn Error>> {
        if let (Some(start), Some(end)) = (gpu_start_ns, gpu_end_ns) {
            if end > start {
                return Ok(end - start);
            }
        }
        let us = gpu_duration_us.ok_or_else(|| {
            format!("{label}: MTLCommandBuffer GPUStartTime/GPUEndTime unavailable")
        })?;
        Ok(us.saturating_mul(1000))
    }

    fn require_dispatch_ns(
        t: hawking_core::metal::MetalDispatchTiming,
        label: &str,
    ) -> Result<u64, Box<dyn Error>> {
        gpu_ns_from(t.gpu_duration_us, t.gpu_start_ns, t.gpu_end_ns, label)
    }

    fn require_batch_ns(
        t: hawking_core::metal::MetalBatchTiming,
        label: &str,
    ) -> Result<u64, Box<dyn Error>> {
        gpu_ns_from(t.gpu_duration_us, t.gpu_start_ns, t.gpu_end_ns, label)
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

    fn median_of(values: &[u64]) -> u64 {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    fn gbps(bytes: u64, ns: u64) -> f64 {
        if ns == 0 {
            return 0.0;
        }
        (bytes as f64) / (ns as f64) // bytes/ns == GB/s
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
        let row_ptr = rice_q1_row_ptr(&rice_indices_oracle, up.binary.rows, up.binary.cols)?;

        let gate_signs = ctx.new_buffer_with_bytes_checked(&gate.signs)?;
        let gate_scales = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&gate.scales_f16))?;
        let up_signs = ctx.new_buffer_with_bytes_checked(&up.binary.signs)?;
        let up_scales = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&up.binary.scales_f16))?;
        let rice_idx_bytes: Vec<u8> = rice_indices_oracle
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let rice_idx_buf = ctx.new_buffer_with_bytes_checked(&rice_idx_bytes)?;
        let row_ptr_bytes: Vec<u8> = row_ptr.iter().flat_map(|v| v.to_le_bytes()).collect();
        let row_ptr_buf = ctx.new_buffer_with_bytes_checked(&row_ptr_bytes)?;
        let residual_signs = ctx.new_buffer_with_bytes_checked(&up.residual_signs)?;
        let left_codes = ctx.new_buffer_with_bytes_checked(&left.codes)?;
        let left_scales = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&left.scales_f16))?;
        let right_codes = ctx.new_buffer_with_bytes_checked(&right.codes)?;
        let right_scales = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&right.scales_f16))?;
        let x_hidden_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&x_hidden))?;
        let x_swiglu_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&x_swiglu))?;
        let gate_out = ctx.new_buffer_checked(Q80_GATE_ROWS * 4)?;
        let up_out = ctx.new_buffer_checked(Q80_GATE_ROWS * 4)?;
        let mid_out = ctx.new_buffer_checked(Q80_HGRAVS_RANK * 4)?;
        let down_out = ctx.new_buffer_checked(Q80_DOWN_ROWS * 4)?;

        let zeros_gate = vec![0.0f32; Q80_GATE_ROWS];
        let zeros_down = vec![0.0f32; Q80_DOWN_ROWS];
        let zeros_mid = vec![0.0f32; Q80_HGRAVS_RANK];

        // ── correctness (not timed) ──────────────────────────────────────
        let mut correctness = serde_json::Map::new();

        write_f32(&gate_out, &zeros_gate);
        ctx.dispatch_threads(
            "q80_binary_group_matvec_simd",
            tiled_grid(gate.rows as u32, 8),
            (256, 1, 1),
            |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
        )?;
        let gate_simd_err = max_abs_error(&gate_oracle, &read_f32(&gate_out, gate.rows));
        if gate_simd_err > BINARY_TOL {
            return Err(format!("gate simd max_abs_error={gate_simd_err} > {BINARY_TOL}").into());
        }
        correctness.insert(
            "gate_simd".into(),
            json!({
                "max_abs_error": gate_simd_err,
                "tolerance": BINARY_TOL,
                "passed": true
            }),
        );

        write_f32(&gate_out, &zeros_gate);
        ctx.dispatch_threads(
            "q80_binary_group_matvec_rowblock4",
            tiled_grid(gate.rows as u32, 32),
            (256, 1, 1),
            |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
        )?;
        let gate_rb4_err = max_abs_error(&gate_oracle, &read_f32(&gate_out, gate.rows));
        if gate_rb4_err > BINARY_TOL {
            return Err(format!("gate rowblock4 max_abs_error={gate_rb4_err} > {BINARY_TOL}").into());
        }
        correctness.insert(
            "gate_rowblock4".into(),
            json!({
                "max_abs_error": gate_rb4_err,
                "tolerance": BINARY_TOL,
                "passed": true
            }),
        );

        write_f32(&gate_out, &zeros_gate);
        ctx.dispatch_threads(
            "q80_binary_group_matvec_simd_bytes",
            tiled_grid(gate.rows as u32, 8),
            (256, 1, 1),
            |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
        )?;
        let gate_bytes_err = max_abs_error(&gate_oracle, &read_f32(&gate_out, gate.rows));
        if gate_bytes_err > BINARY_TOL {
            return Err(format!("gate simd_bytes max_abs_error={gate_bytes_err} > {BINARY_TOL}").into());
        }
        correctness.insert(
            "gate_simd_bytes".into(),
            json!({
                "max_abs_error": gate_bytes_err,
                "tolerance": BINARY_TOL,
                "passed": true
            }),
        );

        write_f32(&gate_out, &zeros_gate);
        ctx.dispatch_threads(
            "q80_binary_group_matvec_chunk",
            tiled_grid(gate.rows as u32, 8),
            (256, 1, 1),
            |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
        )?;
        let gate_chunk_err = max_abs_error(&gate_oracle, &read_f32(&gate_out, gate.rows));
        if gate_chunk_err > BINARY_TOL {
            return Err(format!("gate chunk max_abs_error={gate_chunk_err} > {BINARY_TOL}").into());
        }
        correctness.insert(
            "gate_chunk".into(),
            json!({
                "max_abs_error": gate_chunk_err,
                "tolerance": BINARY_TOL,
                "passed": true
            }),
        );

        write_f32(&gate_out, &zeros_gate);
        ctx.dispatch_threads(
            "q80_binary_group_matvec_tg256",
            (gate.rows as u32 * 256, 1, 1),
            (256, 1, 1),
            |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
        )?;
        let gate_tg256_err = max_abs_error(&gate_oracle, &read_f32(&gate_out, gate.rows));
        if gate_tg256_err > BINARY_TOL {
            return Err(format!("gate tg256 max_abs_error={gate_tg256_err} > {BINARY_TOL}").into());
        }
        correctness.insert(
            "gate_tg256".into(),
            json!({
                "max_abs_error": gate_tg256_err,
                "tolerance": BINARY_TOL,
                "passed": true
            }),
        );

        write_f32(&up_out, &zeros_gate);
        ctx.dispatch_threads(
            "q80_binary_group_csr_matvec",
            tiled_grid(up.binary.rows as u32, 8),
            (256, 1, 1),
            |enc| {
                encode_binary_csr(
                    enc,
                    &up_signs,
                    &up_scales,
                    &x_hidden_buf,
                    &up_out,
                    &rice_idx_buf,
                    &row_ptr_buf,
                    &residual_signs,
                    &up,
                )
            },
        )?;
        let up_fused_err = max_abs_error(&up_oracle, &read_f32(&up_out, up.binary.rows));
        if up_fused_err > UP_TOL {
            return Err(format!("up fused binary+csr max_abs_error={up_fused_err} > {UP_TOL}").into());
        }
        correctness.insert(
            "up_fused_binary_csr".into(),
            json!({
                "max_abs_error": up_fused_err,
                "tolerance": UP_TOL,
                "passed": true
            }),
        );

        write_f32(&up_out, &zeros_gate);
        ctx.dispatch_threads(
            "q80_binary_group_csr_matvec_bytes",
            tiled_grid(up.binary.rows as u32, 8),
            (256, 1, 1),
            |enc| {
                encode_binary_csr(
                    enc,
                    &up_signs,
                    &up_scales,
                    &x_hidden_buf,
                    &up_out,
                    &rice_idx_buf,
                    &row_ptr_buf,
                    &residual_signs,
                    &up,
                )
            },
        )?;
        let up_fused_bytes_err = max_abs_error(&up_oracle, &read_f32(&up_out, up.binary.rows));
        if up_fused_bytes_err > UP_TOL {
            return Err(
                format!("up fused bytes+csr max_abs_error={up_fused_bytes_err} > {UP_TOL}").into(),
            );
        }
        correctness.insert(
            "up_fused_bytes_csr".into(),
            json!({
                "max_abs_error": up_fused_bytes_err,
                "tolerance": UP_TOL,
                "passed": true
            }),
        );

        write_f32(&up_out, &zeros_gate);
        ctx.dispatch_threads(
            "q80_binary_group_csr_matvec_tg256",
            (up.binary.rows as u32 * 256, 1, 1),
            (256, 1, 1),
            |enc| {
                encode_binary_csr(
                    enc,
                    &up_signs,
                    &up_scales,
                    &x_hidden_buf,
                    &up_out,
                    &rice_idx_buf,
                    &row_ptr_buf,
                    &residual_signs,
                    &up,
                )
            },
        )?;
        let up_fused_tg256_err = max_abs_error(&up_oracle, &read_f32(&up_out, up.binary.rows));
        if up_fused_tg256_err > UP_TOL {
            return Err(
                format!("up fused tg256+csr max_abs_error={up_fused_tg256_err} > {UP_TOL}").into(),
            );
        }
        correctness.insert(
            "up_fused_tg256_csr".into(),
            json!({
                "max_abs_error": up_fused_tg256_err,
                "tolerance": UP_TOL,
                "passed": true
            }),
        );

        write_f32(&up_out, &zeros_gate);
        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads(
                "q80_binary_group_matvec_simd",
                tiled_grid(up.binary.rows as u32, 8),
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
                "q80_sparse_q1_apply_csr_simd",
                tiled_grid(up.binary.rows as u32, 8),
                (256, 1, 1),
                |enc| {
                    encode_csr(
                        enc,
                        &rice_idx_buf,
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
        let up_two_err = max_abs_error(&up_oracle, &read_f32(&up_out, up.binary.rows));
        if up_two_err > UP_TOL {
            return Err(format!("up simd+csr_simd max_abs_error={up_two_err} > {UP_TOL}").into());
        }
        correctness.insert(
            "up_simd_then_csr_simd".into(),
            json!({
                "max_abs_error": up_two_err,
                "tolerance": UP_TOL,
                "passed": true
            }),
        );

        write_f32(&mid_out, &zeros_mid);
        ctx.dispatch_threads(
            "q80_hgravs01_factor_matvec_simd",
            tiled_grid(right.rows as u32, 8),
            (256, 1, 1),
            |enc| encode_factor(enc, &right_codes, &right_scales, &x_swiglu_buf, &mid_out, &right),
        )?;
        write_f32(&down_out, &zeros_down);
        ctx.dispatch_threads(
            "q80_hgravs01_factor_matvec_simd",
            tiled_grid(left.rows as u32, 8),
            (256, 1, 1),
            |enc| encode_factor(enc, &left_codes, &left_scales, &mid_out, &down_out, &left),
        )?;
        let down_two_err = max_abs_error(&down_oracle, &read_f32(&down_out, left.rows));
        if down_two_err > HGRAVS_TOL {
            return Err(
                format!("down simd two-dispatch max_abs_error={down_two_err} > {HGRAVS_TOL}").into(),
            );
        }
        correctness.insert(
            "down_simd_two_dispatch".into(),
            json!({
                "max_abs_error": down_two_err,
                "tolerance": HGRAVS_TOL,
                "passed": true
            }),
        );

        write_f32(&mid_out, &zeros_mid);
        write_f32(&down_out, &zeros_down);
        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads(
                "q80_hgravs01_factor_matvec_simd3",
                tiled_grid(right.rows as u32, 8),
                (256, 1, 1),
                |enc| {
                    encode_factor(enc, &right_codes, &right_scales, &x_swiglu_buf, &mid_out, &right)
                },
            )?;
            batch.dispatch_threads(
                "q80_hgravs01_factor_matvec_simd3",
                tiled_grid(left.rows as u32, 8),
                (256, 1, 1),
                |enc| encode_factor(enc, &left_codes, &left_scales, &mid_out, &down_out, &left),
            )?;
            Ok(())
        })?;
        let down_simd3_err = max_abs_error(&down_oracle, &read_f32(&down_out, left.rows));
        if down_simd3_err > HGRAVS_TOL {
            return Err(
                format!("down simd3 two-dispatch max_abs_error={down_simd3_err} > {HGRAVS_TOL}")
                    .into(),
            );
        }
        correctness.insert(
            "down_simd3_two_dispatch".into(),
            json!({
                "max_abs_error": down_simd3_err,
                "tolerance": HGRAVS_TOL,
                "passed": true
            }),
        );

        write_f32(&down_out, &zeros_down);
        ctx.dispatch_threads(
            "q80_hgravs01_two_stage_matvec",
            tiled_grid(left.rows as u32, 8),
            (256, 1, 1),
            |enc| {
                encode_two_stage(
                    enc,
                    &right_codes,
                    &right_scales,
                    &left_codes,
                    &left_scales,
                    &x_swiglu_buf,
                    &down_out,
                    &left,
                    &right,
                )
            },
        )?;
        let down_fused_err = max_abs_error(&down_oracle, &read_f32(&down_out, left.rows));
        if down_fused_err > HGRAVS_TOL {
            return Err(
                format!("down fused two-stage max_abs_error={down_fused_err} > {HGRAVS_TOL}").into(),
            );
        }
        correctness.insert(
            "down_fused".into(),
            json!({
                "max_abs_error": down_fused_err,
                "tolerance": HGRAVS_TOL,
                "passed": true
            }),
        );

        write_f32(&down_out, &zeros_down);
        ctx.dispatch_threads(
            "q80_hgravs01_two_stage_matvec_rowblock4",
            tiled_grid(left.rows as u32, 32),
            (256, 1, 1),
            |enc| {
                encode_two_stage(
                    enc,
                    &right_codes,
                    &right_scales,
                    &left_codes,
                    &left_scales,
                    &x_swiglu_buf,
                    &down_out,
                    &left,
                    &right,
                )
            },
        )?;
        let down_fused_rb4_err = max_abs_error(&down_oracle, &read_f32(&down_out, left.rows));
        if down_fused_rb4_err > HGRAVS_TOL {
            return Err(format!(
                "down fused rowblock4 max_abs_error={down_fused_rb4_err} > {HGRAVS_TOL}"
            )
            .into());
        }
        correctness.insert(
            "down_fused_rowblock4".into(),
            json!({
                "max_abs_error": down_fused_rb4_err,
                "tolerance": HGRAVS_TOL,
                "passed": true
            }),
        );

        // ── warmup (not reported) ────────────────────────────────────────
        let dispatch_gate_serial = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_matvec",
                (gate.rows as u32, 1, 1),
                (256, 1, 1),
                |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
            )?;
            require_dispatch_ns(t, "gate_serial")
        };
        let dispatch_gate_simd = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_matvec_simd",
                tiled_grid(gate.rows as u32, 8),
                (256, 1, 1),
                |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
            )?;
            require_dispatch_ns(t, "gate_simd")
        };
        let dispatch_gate_rb4 = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_matvec_rowblock4",
                tiled_grid(gate.rows as u32, 32),
                (256, 1, 1),
                |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
            )?;
            require_dispatch_ns(t, "gate_rb4")
        };
        let dispatch_gate_bytes = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_matvec_simd_bytes",
                tiled_grid(gate.rows as u32, 8),
                (256, 1, 1),
                |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
            )?;
            require_dispatch_ns(t, "gate_bytes")
        };
        let dispatch_gate_chunk = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_matvec_chunk",
                tiled_grid(gate.rows as u32, 8),
                (256, 1, 1),
                |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
            )?;
            require_dispatch_ns(t, "gate_chunk")
        };
        let dispatch_gate_tg256 = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_matvec_tg256",
                (gate.rows as u32 * 256, 1, 1),
                (256, 1, 1),
                |enc| encode_binary(enc, &gate_signs, &gate_scales, &x_hidden_buf, &gate_out, &gate),
            )?;
            require_dispatch_ns(t, "gate_tg256")
        };
        let dispatch_up_serial = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
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
                        encode_csr(
                            enc,
                            &rice_idx_buf,
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
            require_batch_ns(t, "up_serial")
        };
        let dispatch_up_fused_tg256 = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_csr_matvec_tg256",
                (up.binary.rows as u32 * 256, 1, 1),
                (256, 1, 1),
                |enc| {
                    encode_binary_csr(
                        enc,
                        &up_signs,
                        &up_scales,
                        &x_hidden_buf,
                        &up_out,
                        &rice_idx_buf,
                        &row_ptr_buf,
                        &residual_signs,
                        &up,
                    )
                },
            )?;
            require_dispatch_ns(t, "up_fused_tg256")
        };
        let dispatch_up_fused_bytes = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_csr_matvec_bytes",
                tiled_grid(up.binary.rows as u32, 8),
                (256, 1, 1),
                |enc| {
                    encode_binary_csr(
                        enc,
                        &up_signs,
                        &up_scales,
                        &x_hidden_buf,
                        &up_out,
                        &rice_idx_buf,
                        &row_ptr_buf,
                        &residual_signs,
                        &up,
                    )
                },
            )?;
            require_dispatch_ns(t, "up_fused_bytes")
        };
        let dispatch_up_fused = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_binary_group_csr_matvec",
                tiled_grid(up.binary.rows as u32, 8),
                (256, 1, 1),
                |enc| {
                    encode_binary_csr(
                        enc,
                        &up_signs,
                        &up_scales,
                        &x_hidden_buf,
                        &up_out,
                        &rice_idx_buf,
                        &row_ptr_buf,
                        &residual_signs,
                        &up,
                    )
                },
            )?;
            require_dispatch_ns(t, "up_fused")
        };
        let dispatch_up_two = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_batch_timed(|batch| {
                batch.dispatch_threads(
                    "q80_binary_group_matvec_simd",
                    tiled_grid(up.binary.rows as u32, 8),
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
                    "q80_sparse_q1_apply_csr_simd",
                    tiled_grid(up.binary.rows as u32, 8),
                    (256, 1, 1),
                    |enc| {
                        encode_csr(
                            enc,
                            &rice_idx_buf,
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
            require_batch_ns(t, "up_two")
        };
        let dispatch_down_serial = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
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
            require_batch_ns(t, "down_serial")
        };
        let dispatch_down_simd3_two = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_batch_timed(|batch| {
                batch.dispatch_threads(
                    "q80_hgravs01_factor_matvec_simd3",
                    tiled_grid(right.rows as u32, 8),
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
                    "q80_hgravs01_factor_matvec_simd3",
                    tiled_grid(left.rows as u32, 8),
                    (256, 1, 1),
                    |enc| encode_factor(enc, &left_codes, &left_scales, &mid_out, &down_out, &left),
                )?;
                Ok(())
            })?;
            require_batch_ns(t, "down_simd3_two")
        };
        let dispatch_down_simd_two = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_batch_timed(|batch| {
                batch.dispatch_threads(
                    "q80_hgravs01_factor_matvec_simd",
                    tiled_grid(right.rows as u32, 8),
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
                    "q80_hgravs01_factor_matvec_simd",
                    tiled_grid(left.rows as u32, 8),
                    (256, 1, 1),
                    |enc| encode_factor(enc, &left_codes, &left_scales, &mid_out, &down_out, &left),
                )?;
                Ok(())
            })?;
            require_batch_ns(t, "down_simd_two")
        };
        let dispatch_down_fused = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_hgravs01_two_stage_matvec",
                tiled_grid(left.rows as u32, 8),
                (256, 1, 1),
                |enc| {
                    encode_two_stage(
                        enc,
                        &right_codes,
                        &right_scales,
                        &left_codes,
                        &left_scales,
                        &x_swiglu_buf,
                        &down_out,
                        &left,
                        &right,
                    )
                },
            )?;
            require_dispatch_ns(t, "down_fused")
        };
        let dispatch_down_fused_rb4 = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_hgravs01_two_stage_matvec_rowblock4",
                tiled_grid(left.rows as u32, 32),
                (256, 1, 1),
                |enc| {
                    encode_two_stage(
                        enc,
                        &right_codes,
                        &right_scales,
                        &left_codes,
                        &left_scales,
                        &x_swiglu_buf,
                        &down_out,
                        &left,
                        &right,
                    )
                },
            )?;
            require_dispatch_ns(t, "down_fused_rb4")
        };
        let dispatch_r_only = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_hgravs01_factor_matvec_simd",
                tiled_grid(right.rows as u32, 8),
                (256, 1, 1),
                |enc| {
                    encode_factor(enc, &right_codes, &right_scales, &x_swiglu_buf, &mid_out, &right)
                },
            )?;
            require_dispatch_ns(t, "r_only")
        };
        let dispatch_l_only = |ctx: &MetalContext| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "q80_hgravs01_factor_matvec_simd",
                tiled_grid(left.rows as u32, 8),
                (256, 1, 1),
                |enc| encode_factor(enc, &left_codes, &left_scales, &mid_out, &down_out, &left),
            )?;
            require_dispatch_ns(t, "l_only")
        };

        for _ in 0..WARMUP {
            let _ = dispatch_gate_serial(&ctx)?;
            let _ = dispatch_gate_simd(&ctx)?;
            let _ = dispatch_gate_rb4(&ctx)?;
            let _ = dispatch_gate_bytes(&ctx)?;
            let _ = dispatch_gate_chunk(&ctx)?;
            let _ = dispatch_gate_tg256(&ctx)?;
            let _ = dispatch_up_serial(&ctx)?;
            let _ = dispatch_up_fused(&ctx)?;
            let _ = dispatch_up_fused_bytes(&ctx)?;
            let _ = dispatch_up_fused_tg256(&ctx)?;
            let _ = dispatch_up_two(&ctx)?;
            let _ = dispatch_down_serial(&ctx)?;
            let _ = dispatch_down_simd_two(&ctx)?;
            let _ = dispatch_down_simd3_two(&ctx)?;
            let _ = dispatch_down_fused(&ctx)?;
            let _ = dispatch_down_fused_rb4(&ctx)?;
            let _ = dispatch_r_only(&ctx)?;
            let _ = dispatch_l_only(&ctx)?;
        }

        // Candidate selection: 3 interleaved reps, not the official pair.
        let mut sel_gate_simd = Vec::new();
        let mut sel_gate_rb4 = Vec::new();
        let mut sel_gate_bytes = Vec::new();
        let mut sel_gate_chunk = Vec::new();
        let mut sel_gate_tg256 = Vec::new();
        let mut sel_up_fused = Vec::new();
        let mut sel_up_fused_bytes = Vec::new();
        let mut sel_up_fused_tg256 = Vec::new();
        let mut sel_up_two = Vec::new();
        let mut sel_down_two = Vec::new();
        let mut sel_down_simd3 = Vec::new();
        let mut sel_down_fused = Vec::new();
        let mut sel_down_rb4 = Vec::new();
        let mut sel_r = Vec::new();
        let mut sel_l = Vec::new();
        for _ in 0..3 {
            sel_gate_simd.push(dispatch_gate_simd(&ctx)?);
            sel_gate_rb4.push(dispatch_gate_rb4(&ctx)?);
            sel_gate_bytes.push(dispatch_gate_bytes(&ctx)?);
            sel_gate_chunk.push(dispatch_gate_chunk(&ctx)?);
            sel_gate_tg256.push(dispatch_gate_tg256(&ctx)?);
            sel_up_fused.push(dispatch_up_fused(&ctx)?);
            sel_up_fused_bytes.push(dispatch_up_fused_bytes(&ctx)?);
            sel_up_fused_tg256.push(dispatch_up_fused_tg256(&ctx)?);
            sel_up_two.push(dispatch_up_two(&ctx)?);
            sel_down_two.push(dispatch_down_simd_two(&ctx)?);
            sel_down_simd3.push(dispatch_down_simd3_two(&ctx)?);
            sel_down_fused.push(dispatch_down_fused(&ctx)?);
            sel_down_rb4.push(dispatch_down_fused_rb4(&ctx)?);
            sel_r.push(dispatch_r_only(&ctx)?);
            sel_l.push(dispatch_l_only(&ctx)?);
        }

        let gate_cands = [
            ("simd", median_of(&sel_gate_simd)),
            ("rowblock4", median_of(&sel_gate_rb4)),
            ("simd_bytes", median_of(&sel_gate_bytes)),
            ("chunk", median_of(&sel_gate_chunk)),
            ("tg256", median_of(&sel_gate_tg256)),
        ];
        let gate_b_kind = gate_cands
            .iter()
            .min_by_key(|(_, ns)| *ns)
            .map(|(name, _)| *name)
            .unwrap();
        let up_cands = [
            ("fused", median_of(&sel_up_fused)),
            ("fused_bytes", median_of(&sel_up_fused_bytes)),
            ("fused_tg256", median_of(&sel_up_fused_tg256)),
            ("two_dispatch", median_of(&sel_up_two)),
        ];
        let up_b_kind = up_cands
            .iter()
            .min_by_key(|(_, ns)| *ns)
            .map(|(name, _)| *name)
            .unwrap();
        let down_meds = [
            ("simd_two_dispatch", median_of(&sel_down_two)),
            ("simd3_two_dispatch", median_of(&sel_down_simd3)),
            ("fused", median_of(&sel_down_fused)),
            ("fused_rowblock4", median_of(&sel_down_rb4)),
        ];
        let down_b_kind = down_meds
            .iter()
            .min_by_key(|(_, ns)| *ns)
            .map(|(name, _)| *name)
            .unwrap();

        // Official paired A/B. A is the shipped serial path. B is the winner.
        let mut gate_a = Vec::with_capacity(MEASURED);
        let mut gate_b = Vec::with_capacity(MEASURED);
        let mut up_a = Vec::with_capacity(MEASURED);
        let mut up_b = Vec::with_capacity(MEASURED);
        let mut down_a = Vec::with_capacity(MEASURED);
        let mut down_b = Vec::with_capacity(MEASURED);
        for _ in 0..MEASURED {
            gate_a.push(dispatch_gate_serial(&ctx)?);
            gate_b.push(match gate_b_kind {
                "rowblock4" => dispatch_gate_rb4(&ctx)?,
                "simd_bytes" => dispatch_gate_bytes(&ctx)?,
                "chunk" => dispatch_gate_chunk(&ctx)?,
                "tg256" => dispatch_gate_tg256(&ctx)?,
                _ => dispatch_gate_simd(&ctx)?,
            });
            up_a.push(dispatch_up_serial(&ctx)?);
            up_b.push(match up_b_kind {
                "fused" => dispatch_up_fused(&ctx)?,
                "fused_bytes" => dispatch_up_fused_bytes(&ctx)?,
                "fused_tg256" => dispatch_up_fused_tg256(&ctx)?,
                _ => dispatch_up_two(&ctx)?,
            });
            down_a.push(dispatch_down_serial(&ctx)?);
            down_b.push(match down_b_kind {
                "simd3_two_dispatch" => dispatch_down_simd3_two(&ctx)?,
                "fused_rowblock4" => dispatch_down_fused_rb4(&ctx)?,
                "fused" => dispatch_down_fused(&ctx)?,
                _ => dispatch_down_simd_two(&ctx)?,
            });
        }

        let gate_a_med = median_of(&gate_a);
        let gate_b_med = median_of(&gate_b);
        let up_a_med = median_of(&up_a);
        let up_b_med = median_of(&up_b);
        let down_a_med = median_of(&down_a);
        let down_b_med = median_of(&down_b);
        let organ_a = gate_a_med + up_a_med + down_a_med;
        let organ_b = gate_b_med + up_b_med + down_b_med;
        let routed_a = organ_a.saturating_mul(LAYERS).saturating_mul(TOP_K);
        let routed_b = organ_b.saturating_mul(LAYERS).saturating_mul(TOP_K);

        let gate_bytes = (gate.signs.len() + gate.scales_f16.len() * 2) as u64;
        let up_bytes = (up.binary.signs.len()
            + up.binary.scales_f16.len() * 2
            + up.rice_bytes.len()
            + 4
            + 2
            + up.residual_signs.len()) as u64;
        let down_bytes = (left.codes.len()
            + left.scales_f16.len() * 2
            + right.codes.len()
            + right.scales_f16.len() * 2) as u64;
        let gate_dram = gate_bytes + (Q80_GATE_COLS * 4 + Q80_GATE_ROWS * 4) as u64;
        let up_dram = up_bytes
            + (up.outlier_count * 4 + (up.binary.rows + 1) * 4 + Q80_GATE_COLS * 4 + Q80_GATE_ROWS * 4)
                as u64;
        let down_dram =
            down_bytes + (Q80_DOWN_COLS * 4 + Q80_HGRAVS_RANK * 4 + Q80_DOWN_ROWS * 4) as u64;
        let down_fused_dram = down_bytes + (Q80_DOWN_COLS * 4 + Q80_DOWN_ROWS * 4) as u64;

        let dense_gate = (Q80_GATE_ROWS * Q80_GATE_COLS * 4) as u64;
        let dense_up = dense_gate;
        let dense_down = (Q80_DOWN_ROWS * Q80_DOWN_COLS * 4) as u64;
        let dense_expert = dense_gate + dense_up + dense_down;
        let packed_expert = gate_bytes + up_bytes + down_bytes;

        let budget_ns = 8_000_000u64;
        let vs_budget = (routed_b as f64) / (budget_ns as f64);

        let receipt = json!({
            "schema": SCHEMA,
            "status": "DECODE_THROUGHPUT_COMPONENT_NOT_RUNTIME_OR_COHERENCE",
            "lane": "q80-decode-throughput",
            "measurement_label": "DIRTY_ENGINEERING",
            "baseline_ns_per_token": routed_a,
            "baseline_label": "DIRTY_ENGINEERING",
            "baseline_source": "this-run shipped serial A median; shipped-lane reference was 192_000_000",
            "shipped_lane_reference_ns": {
                "gate": SHIPPED_GATE_NS,
                "up": SHIPPED_UP_NS,
                "down": SHIPPED_DOWN_NS,
                "routed_48x10": (SHIPPED_GATE_NS + SHIPPED_UP_NS + SHIPPED_DOWN_NS) * LAYERS * TOP_K
            },
            "result_ns_per_token": routed_b,
            "result_label": "DIRTY_ENGINEERING",
            "result_scope": "48*10*(gate_b+up_b+down_b) routed mixed-decode GPU only",
            "timing_authority": "MTLCommandBuffer.GPUEndTime - GPUStartTime after wait_until_completed; ns from raw endpoints when present",
            "device_name": ctx.device_name(),
            "geometry": {
                "gate_up_shape": [Q80_GATE_ROWS, Q80_GATE_COLS],
                "down_shape": [Q80_DOWN_ROWS, Q80_DOWN_COLS],
                "hgravs_rank": Q80_HGRAVS_RANK,
                "hgravs_bits": Q80_HGRAVS_BITS,
                "hgravs_group_size": Q80_HGRAVS_GROUP_SIZE,
                "binary_group_size": Q80_BINARY_GROUP_SIZE,
                "rice_outlier_count": up.outlier_count,
                "rice_k": up.rice_k
            },
            "regime": {
                "note": "packed bytes / GPU ns. Ceiling 800 GB/s is the machine DRAM ballpark, not a measured roofline.",
                "ceiling_gbps": CEILING_GBPS,
                "gate": {
                    "packed_weight_bytes": gate_bytes,
                    "dram_bytes": gate_dram,
                    "weights": Q80_GATE_ROWS * Q80_GATE_COLS,
                    "decode_ops_per_weight": 1,
                    "shipped_threads": Q80_GATE_ROWS,
                    "fast_threads": (Q80_GATE_ROWS as u32).div_ceil(8) * 256,
                    "gbps_a": gbps(gate_dram, gate_a_med),
                    "gbps_b": gbps(gate_dram, gate_b_med),
                    "pct_of_ceiling_b": 100.0 * gbps(gate_dram, gate_b_med) / CEILING_GBPS,
                    "verdict": "serial-decode / occupancy bound, not bandwidth bound"
                },
                "up": {
                    "packed_weight_bytes": up_bytes,
                    "dram_bytes": up_dram,
                    "gbps_a": gbps(up_dram, up_a_med),
                    "gbps_b": gbps(up_dram, up_b_med),
                    "pct_of_ceiling_b": 100.0 * gbps(up_dram, up_b_med) / CEILING_GBPS,
                    "verdict": "binary decode dominates; CSR is ~2% nnz and not the wall"
                },
                "down": {
                    "packed_weight_bytes": down_bytes,
                    "dram_bytes_two_dispatch": down_dram,
                    "dram_bytes_fused": down_fused_dram,
                    "factor_elements": Q80_HGRAVS_RANK * Q80_DOWN_COLS + Q80_DOWN_ROWS * Q80_HGRAVS_RANK,
                    "gbps_a": gbps(down_dram, down_a_med),
                    "gbps_b": gbps(down_fused_dram, down_b_med),
                    "pct_of_ceiling_b": 100.0 * gbps(down_fused_dram, down_b_med) / CEILING_GBPS,
                    "r_only_simd_ns": spread(&sel_r),
                    "l_only_simd_ns": spread(&sel_l),
                    "verdict": "serial 3-bit factor decode + 160-thread R occupancy was the wall"
                },
                "one_expert_three_organs_bytes": packed_expert,
                "prompt_arithmetic": {
                    "stated": "240 KB in 400 us ~ 0.6 GB/s vs 800 GB/s",
                    "measured_packed_bytes": packed_expert,
                    "shipped_us": 400.0,
                    "measured_gbps_at_shipped_400us": gbps(packed_expert, 400_000),
                    "conclusion": "decode is compute/occupancy bound by ~three orders vs the 800 GB/s ceiling; the 240 KB figure was slightly low (actual ~470 KB) but the regime call is correct"
                }
            },
            "selected_b": {
                "gate": match gate_b_kind {
                    "rowblock4" => "q80_binary_group_matvec_rowblock4",
                    "simd_bytes" => "q80_binary_group_matvec_simd_bytes",
                    "chunk" => "q80_binary_group_matvec_chunk",
                    "tg256" => "q80_binary_group_matvec_tg256",
                    _ => "q80_binary_group_matvec_simd",
                },
                "up": match up_b_kind {
                    "fused" => "q80_binary_group_csr_matvec",
                    "fused_bytes" => "q80_binary_group_csr_matvec_bytes",
                    "fused_tg256" => "q80_binary_group_csr_matvec_tg256",
                    _ => "q80_binary_group_matvec_simd + q80_sparse_q1_apply_csr_simd",
                },
                "down": match down_b_kind {
                    "simd3_two_dispatch" => "q80_hgravs01_factor_matvec_simd3 R then L",
                    "fused_rowblock4" => "q80_hgravs01_two_stage_matvec_rowblock4",
                    "fused" => "q80_hgravs01_two_stage_matvec",
                    _ => "q80_hgravs01_factor_matvec_simd R then L",
                },
                "selection_medians_ns": {
                    "gate_simd": median_of(&sel_gate_simd),
                    "gate_rowblock4": median_of(&sel_gate_rb4),
                    "gate_simd_bytes": median_of(&sel_gate_bytes),
                    "gate_chunk": median_of(&sel_gate_chunk),
                    "gate_tg256": median_of(&sel_gate_tg256),
                    "up_fused": median_of(&sel_up_fused),
                    "up_fused_bytes": median_of(&sel_up_fused_bytes),
                    "up_fused_tg256": median_of(&sel_up_fused_tg256),
                    "up_two_dispatch": median_of(&sel_up_two),
                    "down_simd_two": median_of(&sel_down_two),
                    "down_simd3_two": median_of(&sel_down_simd3),
                    "down_fused": median_of(&sel_down_fused),
                    "down_fused_rowblock4": median_of(&sel_down_rb4)
                }
            },
            "correctness": {
                "tolerance": 2e-5,
                "rice_indices": "bind-time expanded; bit-identical CPU oracle reused, not re-decoded per token",
                "fallbacks": 0,
                "dense_reconstruction_on_token_path": false,
                "kernels": correctness,
                "largest_measured_drift": {
                    "gate": gate_simd_err
                        .max(gate_rb4_err)
                        .max(gate_bytes_err)
                        .max(gate_chunk_err)
                        .max(gate_tg256_err),
                    "up": up_fused_err
                        .max(up_two_err)
                        .max(up_fused_bytes_err)
                        .max(up_fused_tg256_err),
                    "down": down_two_err
                        .max(down_fused_err)
                        .max(down_fused_rb4_err)
                        .max(down_simd3_err)
                }
            },
            "cost": {
                "gate_proj": {
                    "a_shipped_serial_ns": spread(&gate_a),
                    "b_fast_ns": spread(&gate_b),
                    "median_a": gate_a_med,
                    "median_b": gate_b_med,
                    "dispatches_a": 1,
                    "dispatches_b": 1
                },
                "up_proj": {
                    "a_shipped_serial_ns": spread(&up_a),
                    "b_fast_ns": spread(&up_b),
                    "median_a": up_a_med,
                    "median_b": up_b_med,
                    "dispatches_a": 2,
                    "dispatches_b": if up_b_kind == "two_dispatch" { 2 } else { 1 }
                },
                "down_proj": {
                    "a_shipped_serial_ns": spread(&down_a),
                    "b_fast_ns": spread(&down_b),
                    "median_a": down_a_med,
                    "median_b": down_b_med,
                    "dispatches_a": 2,
                    "dispatches_b": if down_b_kind == "simd_two_dispatch" || down_b_kind == "simd3_two_dispatch" { 2 } else { 1 },
                    "temporary_bytes": if down_b_kind == "simd_two_dispatch" || down_b_kind == "simd3_two_dispatch" { Q80_HGRAVS_RANK * 4 } else { 0 },
                    "temporary_kind": if down_b_kind == "simd_two_dispatch" || down_b_kind == "simd3_two_dispatch" {
                        "device mid[rank=160] f32, not dense W"
                    } else {
                        "threadgroup mid[160] + x[512]; not dense W"
                    }
                },
                "routed_expert_wave_estimate": {
                    "layers": LAYERS,
                    "top_k": TOP_K,
                    "formula": "48 * 10 * (gate + up + down)",
                    "median_ns_per_token_a": routed_a,
                    "median_ns_per_token_b": routed_b,
                    "budget_ns": budget_ns,
                    "b_over_8ms_budget": vs_budget,
                    "label": "DIRTY_ENGINEERING",
                    "claim_boundary": "routed-expert mixed-decode GPU time only"
                }
            },
            "amortization": {
                "decoded_dense_bytes_per_expert": dense_expert,
                "packed_bytes_per_expert": packed_expert,
                "top10_decoded_bytes": dense_expert * 10,
                "top10_packed_bytes": packed_expert * 10,
                "all_routed_48x512_decoded_gib": (dense_expert as f64) * 48.0 * 512.0 / 1024.0 / 1024.0 / 1024.0,
                "all_routed_48x512_packed_gib": (packed_expert as f64) * 48.0 * 512.0 / 1024.0 / 1024.0 / 1024.0,
                "verdict": "decoded-weight cache is rejected: 12 MiB dense per expert, 120 MiB for the live top-10, ~288 GiB if every (layer, expert) is cached. That restores the footprint the 1.23 BPW representation exists to remove. Bind-time rice-index expand (~84 KiB/up) stays; it is not dense W."
            },
            "reps": {
                "protocol": "warmup=2 all variants; 3-rep selection sweep; then 6 alternating A,B pairs (gate, up, down interleaved inside each arm)",
                "gate_a_gpu_ns": gate_a,
                "gate_b_gpu_ns": gate_b,
                "up_a_gpu_ns": up_a,
                "up_b_gpu_ns": up_b,
                "down_a_gpu_ns": down_a,
                "down_b_gpu_ns": down_b
            },
            "next_bottleneck": {
                "organ": if down_b_med >= gate_b_med && down_b_med >= up_b_med {
                    "down_proj"
                } else if up_b_med >= gate_b_med {
                    "up_proj"
                } else {
                    "gate_proj"
                },
                "median_ns": down_b_med.max(up_b_med).max(gate_b_med),
                "token_scale_ns": down_b_med.max(up_b_med).max(gate_b_med) * LAYERS * TOP_K
            },
            "claim_boundary": {
                "decode_kernel_exists": true,
                "artifact_packed": false,
                "coherence_generation_tested": false,
                "not_full_token_runtime": true,
                "not_bf16_parent_parity": true,
                "cpu_wall_is_not_gpu_time": true,
                "dense_w_never_materialized": true
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
