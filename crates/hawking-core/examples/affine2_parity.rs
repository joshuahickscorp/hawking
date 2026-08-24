//! Native affine-2 kernel vs CPU reconstruction.
//!
//! Default `--synthetic` packs a deterministic matrix as HGRAVF01
//! (`w = q * scale + bias`, unsigned q in {0,1,2,3}, group 64) and
//! dispatches the group32-family kernels (`affine2_group32_matvec` /
//! geo_tpr64 / dequant) against the f32 oracle. `--group 32` still
//! exercises the original group. Does not call mlx or mlx_lm. Never
//! writes dense W on the GEMV path.
//!
//! ```text
//! cargo run -p hawking-core --example affine2_parity -- --synthetic --group 64
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("affine2_parity requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use half::f16;
    use hawking_core::model::qwen_complete_binary::{
        affine_factor_matvec_f32, affine_factor_value, pack_affine_factor_group, wrap_affine_factor,
        AffineFactorPacked, AFFINE_GROUP_SIZE, AFFINE_GROUP_SIZE_64,
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

    fn as_bytes_u32(values: &[u32]) -> &[u8] {
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
        (0..cols)
            .map(|i| (i % 17) as f32 * 0.125 - 1.0)
            .collect()
    }

    fn codes_as_u32(packed: &AffineFactorPacked) -> Vec<u32> {
        packed
            .codes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
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
        let mut group = AFFINE_GROUP_SIZE_64;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--synthetic" => {}
                "--group" => {
                    let value = args.next().ok_or("missing value after --group")?;
                    group = value.parse::<usize>()?;
                    if group != AFFINE_GROUP_SIZE && group != AFFINE_GROUP_SIZE_64 {
                        return Err(format!("group must be 32 or 64, got {group}").into());
                    }
                }
                "--artifact" => {
                    let _ = args.next();
                    eprintln!("affine2_parity: --artifact ignored; running --synthetic");
                }
                other => return Err(format!("unsupported option {other}").into()),
            }
        }

        let values = hawking_core::model::qwen_complete_binary::deterministic_matrix(
            SYNTH_ROWS, SYNTH_COLS, 41,
        );
        let packed = pack_affine_factor_group(&values, SYNTH_ROWS, SYNTH_COLS, group)?;
        let payload = wrap_affine_factor(&packed)?;
        assert_eq!(&payload[..8], b"HGRAVF01");
        assert_eq!(packed.group_size, group);

        let codes = codes_as_u32(&packed);
        let scales = widen_f16(&packed.scales_f16);
        let biases = widen_f16(&packed.biases_f16);
        let cpu_w = cpu_reconstruct(&packed);
        let input = deterministic_input(packed.cols);
        let cpu_y = affine_factor_matvec_f32(&packed, &input)?;

        let device = Device::system_default().ok_or("no Metal-capable GPU")?;
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("affine2 shader compile: {e}"))?;
        let dequant_fn = library
            .get_function("affine2_group32_dequant", None)
            .map_err(|e| format!("affine2_group32_dequant: {e}"))?;
        let matvec_fn = library
            .get_function("affine2_group32_matvec", None)
            .map_err(|e| format!("affine2_group32_matvec: {e}"))?;
        let geo_fn = library
            .get_function("affine2_group32_matvec_geo_tpr64_tg128", None)
            .map_err(|e| format!("affine2_group32_matvec_geo_tpr64_tg128: {e}"))?;
        let dequant_pipe = device.new_compute_pipeline_state_with_function(&dequant_fn)?;
        let matvec_pipe = device.new_compute_pipeline_state_with_function(&matvec_fn)?;
        let geo_pipe = device.new_compute_pipeline_state_with_function(&geo_fn)?;

        let codes_buf = device.new_buffer_with_data(
            as_bytes_u32(&codes).as_ptr() as *const _,
            as_bytes_u32(&codes).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scales_buf = device.new_buffer_with_data(
            as_bytes_f32(&scales).as_ptr() as *const _,
            as_bytes_f32(&scales).len() as u64,
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

        let rows = packed.rows as u32;
        let cols = packed.cols as u32;
        let group_u32 = group as u32;

        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&dequant_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&scales_buf), 0);
            enc.set_buffer(2, Some(&biases_buf), 0);
            enc.set_buffer(3, Some(&dequant_buf), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            set_u32(enc, 6, group_u32);
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
            enc.set_buffer(1, Some(&scales_buf), 0);
            enc.set_buffer(2, Some(&biases_buf), 0);
            enc.set_buffer(3, Some(&input_buf), 0);
            enc.set_buffer(4, Some(&y_serial_buf), 0);
            set_u32(enc, 5, rows);
            set_u32(enc, 6, cols);
            set_u32(enc, 7, group_u32);
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
            enc.set_buffer(1, Some(&scales_buf), 0);
            enc.set_buffer(2, Some(&biases_buf), 0);
            enc.set_buffer(3, Some(&input_buf), 0);
            enc.set_buffer(4, Some(&y_geo_buf), 0);
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
        let max_abs_diff_w = max_abs_diff(&gpu_w, &cpu_w);
        let max_abs_diff_y = max_abs_diff(&gpu_y, &cpu_y);
        let max_abs_diff_geo = max_abs_diff(&geo_y, &cpu_y);
        let body_bpw = 2.0 + 16.0 / group as f64 + 16.0 / group as f64;

        println!("codec: HGRAVF01 affine_q2_group{group}");
        println!("mode: synthetic");
        println!("shape: [{}, {}]", packed.rows, packed.cols);
        println!("groups: {}", packed.groups);
        println!(
            "storage_bpw_body: {body_bpw:.1} (2-bit codes + f16 scale + f16 bias @ g{group})"
        );
        println!("kernel: affine2_group32_matvec (in-register dequant, no dense W on GEMV)");
        println!("kernel_geo: affine2_group32_matvec_geo_tpr64_tg128");
        println!("max_abs_diff: {:.6e}", max_abs_diff_w);
        println!("max_abs_diff_matvec: {:.6e}", max_abs_diff_y);
        println!("max_abs_diff_geo_tpr64: {:.6e}", max_abs_diff_geo);
        println!("tolerance: {:.6e}", PASS_TOL);
        println!("dense_w_materialized_on_gemv: 0");

        if !max_abs_diff_w.is_finite() || max_abs_diff_w >= PASS_TOL {
            return Err(format!(
                "native affine2 reconstruction diverged: max_abs_diff={max_abs_diff_w} >= {PASS_TOL}"
            )
            .into());
        }
        if !max_abs_diff_y.is_finite() || max_abs_diff_y >= PASS_TOL {
            return Err(format!(
                "native affine2 matvec diverged: max_abs_diff_matvec={max_abs_diff_y} >= {PASS_TOL}"
            )
            .into());
        }
        if !max_abs_diff_geo.is_finite() || max_abs_diff_geo >= PASS_TOL {
            return Err(format!(
                "native affine2 geo_tpr64 diverged: max_abs_diff_geo_tpr64={max_abs_diff_geo} >= {PASS_TOL}"
            )
            .into());
        }
        println!("status: PASS");
        Ok(())
    }
}
