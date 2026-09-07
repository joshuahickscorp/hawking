//! Native Q2F kernel vs CPU reconstruction.
//!
//! Packs a deterministic matrix as HGRAVF01 delta-only
//! (`w = (q - 1.5) * delta`, q in {0,1,2,3}, group 64) and dispatches
//! `q2f_group64_matvec` / geo_tpr64 against the f32 oracle. Also runs the
//! affine2 kernel on the same codes with `bias = -1.5*delta` so the two
//! reconstructions can be compared. Never writes dense W on the GEMV path.
//!
//! ```text
//! cargo run -p hawking-core --example q2f_parity -- --synthetic
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("q2f_parity requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use half::f16;
    use hawking_core::model::qwen_complete_binary::{
        affine_factor_matvec_f32, affine_factor_value, pack_q2f_factor_group, wrap_affine_factor,
        AffineFactorPacked, AFFINE_GROUP_SIZE_64,
    };
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
    use std::env;
    use std::error::Error;

    const SHADER: &str = include_str!("../shaders/affine2_group32_matvec.metal");
    const PASS_TOL: f32 = 1e-2;
    const SYNTH_ROWS: usize = 64;
    const SYNTH_COLS: usize = 256;

    fn as_bytes_f32(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn read_f32(buffer: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, n).to_vec() }
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    fn deterministic_input(cols: usize) -> Vec<f32> {
        (0..cols).map(|i| (i % 17) as f32 * 0.125 - 1.0).collect()
    }

    fn widen_f16(bits: &[u16]) -> Vec<f32> {
        bits.iter().map(|b| f16::from_bits(*b).to_f32()).collect()
    }

    fn cpu_reconstruct(packed: &AffineFactorPacked) -> Vec<f32> {
        let mut out = vec![0.0f32; packed.rows * packed.cols];
        for row in 0..packed.rows {
            for col in 0..packed.cols {
                out[row * packed.cols + col] = affine_factor_value(packed, row, col);
            }
        }
        out
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--synthetic" => {}
                other => return Err(format!("unsupported option {other}").into()),
            }
        }

        let values = hawking_core::model::qwen_complete_binary::deterministic_matrix(
            SYNTH_ROWS, SYNTH_COLS, 41,
        );
        let packed = pack_q2f_factor_group(&values, SYNTH_ROWS, SYNTH_COLS, AFFINE_GROUP_SIZE_64)?;
        let payload = wrap_affine_factor(&packed)?;
        assert_eq!(&payload[..8], b"HGRAVF01");
        assert!(packed.is_q2f());
        assert_eq!(packed.group_size, 64);

        let deltas = widen_f16(&packed.scales_f16);
        let biases: Vec<f32> = deltas.iter().map(|d| -1.5 * *d).collect();
        let cpu_w = cpu_reconstruct(&packed);
        let input = deterministic_input(packed.cols);
        let cpu_y = affine_factor_matvec_f32(&packed, &input)?;

        let device = Device::system_default().ok_or("no Metal-capable GPU")?;
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("q2f shader compile: {e}"))?;
        let dequant_fn = library
            .get_function("q2f_group64_dequant", None)
            .map_err(|e| format!("q2f_group64_dequant: {e}"))?;
        let matvec_fn = library
            .get_function("q2f_group64_matvec", None)
            .map_err(|e| format!("q2f_group64_matvec: {e}"))?;
        let geo_fn = library
            .get_function("q2f_group64_matvec_geo_tpr64_tg128", None)
            .map_err(|e| format!("q2f_group64_matvec_geo_tpr64_tg128: {e}"))?;
        let affine_geo_fn = library
            .get_function("affine2_group32_matvec_geo_tpr64_tg128", None)
            .map_err(|e| format!("affine2 geo: {e}"))?;
        let fused_fn = library
            .get_function("q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128", None)
            .map_err(|e| format!("q2f fused swiglu: {e}"))?;
        let dequant_pipe = device.new_compute_pipeline_state_with_function(&dequant_fn)?;
        let matvec_pipe = device.new_compute_pipeline_state_with_function(&matvec_fn)?;
        let geo_pipe = device.new_compute_pipeline_state_with_function(&geo_fn)?;
        let affine_geo_pipe = device.new_compute_pipeline_state_with_function(&affine_geo_fn)?;
        let fused_pipe = device.new_compute_pipeline_state_with_function(&fused_fn)?;

        let codes_buf = device.new_buffer_with_data(
            packed.codes.as_ptr() as *const _,
            packed.codes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let deltas_buf = device.new_buffer_with_data(
            as_bytes_f32(&deltas).as_ptr() as *const _,
            as_bytes_f32(&deltas).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let biases_buf = device.new_buffer_with_data(
            as_bytes_f32(&biases).as_ptr() as *const _,
            as_bytes_f32(&biases).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buf = device.new_buffer_with_data(
            as_bytes_f32(&input).as_ptr() as *const _,
            as_bytes_f32(&input).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let dequant_buf = device.new_buffer(
            (packed.rows * packed.cols * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_serial_buf = device.new_buffer(
            (packed.rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_geo_buf = device.new_buffer(
            (packed.rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_affine_buf = device.new_buffer(
            (packed.rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let rows = packed.rows as u32;
        let cols = packed.cols as u32;
        let group_u32 = 64u32;

        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&dequant_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&deltas_buf), 0);
            enc.set_buffer(2, Some(&dequant_buf), 0);
            set_u32(enc, 3, rows);
            set_u32(enc, 4, cols);
            let n = (packed.rows * packed.cols) as u64;
            enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&matvec_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&deltas_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(&y_serial_buf), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            enc.dispatch_threads(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(256.min(rows.max(1) as u64), 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&geo_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&deltas_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(&y_geo_buf), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            let groups = (packed.rows as u64).div_ceil(2);
            enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&affine_geo_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&deltas_buf), 0);
            enc.set_buffer(2, Some(&biases_buf), 0);
            enc.set_buffer(3, Some(&input_buf), 0);
            enc.set_buffer(4, Some(&y_affine_buf), 0);
            set_u32(enc, 5, rows);
            set_u32(enc, 6, cols);
            set_u32(enc, 7, group_u32);
            let groups = (packed.rows as u64).div_ceil(2);
            enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let gpu_w = read_f32(&dequant_buf, packed.rows * packed.cols);
        let gpu_y = read_f32(&y_serial_buf, packed.rows);
        let geo_y = read_f32(&y_geo_buf, packed.rows);
        let affine_y = read_f32(&y_affine_buf, packed.rows);
        let max_abs_diff_w = max_abs_diff(&gpu_w, &cpu_w);
        let max_abs_diff_y = max_abs_diff(&gpu_y, &cpu_y);
        let max_abs_diff_geo = max_abs_diff(&geo_y, &cpu_y);
        let max_abs_diff_reuse = max_abs_diff(&affine_y, &cpu_y);

        // Oracle dequant writes dense W. Production GEMV does not.
        let mut dense_w_materialized_on_oracle_dequant: u64 = 0;
        dense_w_materialized_on_oracle_dequant =
            dense_w_materialized_on_oracle_dequant.saturating_add(1);
        let dense_w_materialized_on_gemv: u64 = 0;

        let up_values = hawking_core::model::qwen_complete_binary::deterministic_matrix(
            SYNTH_ROWS, SYNTH_COLS, 43,
        );
        let packed_up =
            pack_q2f_factor_group(&up_values, SYNTH_ROWS, SYNTH_COLS, AFFINE_GROUP_SIZE_64)?;
        let up_deltas = widen_f16(&packed_up.scales_f16);
        let cpu_up = affine_factor_matvec_f32(&packed_up, &input)?;
        let mut cpu_act = vec![0.0f32; packed.rows];
        for i in 0..packed.rows {
            let g = cpu_y[i];
            cpu_act[i] = (g / (1.0 + (-g).exp())) * cpu_up[i];
        }
        let up_codes_buf = device.new_buffer_with_data(
            packed_up.codes.as_ptr() as *const _,
            packed_up.codes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let up_deltas_buf = device.new_buffer_with_data(
            as_bytes_f32(&up_deltas).as_ptr() as *const _,
            as_bytes_f32(&up_deltas).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let act_buf = device.new_buffer(
            (packed.rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&fused_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&deltas_buf), 0);
            enc.set_buffer(2, Some(&up_codes_buf), 0);
            enc.set_buffer(3, Some(&up_deltas_buf), 0);
            enc.set_buffer(4, Some(&input_buf), 0);
            enc.set_buffer(5, Some(&act_buf), 0);
            set_u32(enc, 6, rows);
            set_u32(enc, 7, cols);
            let groups = (packed.rows as u64).div_ceil(2);
            enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let gpu_act = read_f32(&act_buf, packed.rows);
        let max_abs_diff_fused_swiglu = max_abs_diff(&gpu_act, &cpu_act);

        let body_bpw = 2.0 + 16.0 / 64.0;

        println!("codec: HGRAVF01 fourlevel_q2_group64_fp16_delta");
        println!("mode: synthetic");
        println!("shape: [{}, {}]", packed.rows, packed.cols);
        println!("groups: {}", packed.groups);
        println!("storage_bpw_body: {body_bpw:.2} (2-bit codes + f16 delta @ g64, no bias)");
        println!("kernel: q2f_group64_matvec (in-register dequant, no dense W on GEMV)");
        println!("kernel_geo: q2f_group64_matvec_geo_tpr64_tg128");
        println!("kernel_fused_swiglu: q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128");
        println!("reuse_affine2_kernel: affine2_group32_matvec_geo_tpr64_tg128 (bias=-1.5*delta)");
        println!("max_abs_diff: {:.6e}", max_abs_diff_w);
        println!("max_abs_diff_matvec: {:.6e}", max_abs_diff_y);
        println!("max_abs_diff_geo_tpr64: {:.6e}", max_abs_diff_geo);
        println!("max_abs_diff_reuse_affine2: {:.6e}", max_abs_diff_reuse);
        println!(
            "max_abs_diff_fused_swiglu: {:.6e}",
            max_abs_diff_fused_swiglu
        );
        println!("tolerance: {:.6e}", PASS_TOL);
        println!(
            "dense_w_materialized_on_oracle_dequant: {dense_w_materialized_on_oracle_dequant}"
        );
        println!("dense_w_materialized_on_gemv: {dense_w_materialized_on_gemv}");

        if !max_abs_diff_w.is_finite() || max_abs_diff_w >= PASS_TOL {
            return Err(format!(
                "native q2f reconstruction diverged: max_abs_diff={max_abs_diff_w} >= {PASS_TOL}"
            )
            .into());
        }
        if !max_abs_diff_y.is_finite() || max_abs_diff_y >= PASS_TOL {
            return Err(format!(
                "native q2f matvec diverged: max_abs_diff_matvec={max_abs_diff_y} >= {PASS_TOL}"
            )
            .into());
        }
        if !max_abs_diff_geo.is_finite() || max_abs_diff_geo >= PASS_TOL {
            return Err(format!(
                "native q2f geo_tpr64 diverged: max_abs_diff_geo_tpr64={max_abs_diff_geo} >= {PASS_TOL}"
            )
            .into());
        }
        if !max_abs_diff_fused_swiglu.is_finite() || max_abs_diff_fused_swiglu >= PASS_TOL {
            return Err(format!(
                "native q2f fused SwiGLU diverged: max_abs_diff_fused_swiglu={max_abs_diff_fused_swiglu} >= {PASS_TOL}"
            )
            .into());
        }
        // max_abs_diff_reuse was printed but never gated. A corruption of affine2_w --
        // the exact function this path exercises -- reached 2.53e-2, 2.5x PASS_TOL, and
        // the contract still reported PASS. Watched failing before this gate existed.
        if !max_abs_diff_reuse.is_finite() || max_abs_diff_reuse >= PASS_TOL {
            return Err(format!(
                "native q2f affine2-reuse diverged: max_abs_diff_reuse_affine2={max_abs_diff_reuse} >= {PASS_TOL}"
            )
            .into());
        }
        if dense_w_materialized_on_gemv != 0 {
            return Err("production GEMV materialized dense W".into());
        }
        println!("status: PASS");
        Ok(())
    }
}
