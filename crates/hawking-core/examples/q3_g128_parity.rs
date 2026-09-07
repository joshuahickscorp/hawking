//! Native HGRAVU01 q3 g128 geo kernel vs CPU reconstruction.
//!
//! Packs a deterministic matrix as grouped-absmax q3 / group 128 and
//! dispatches `qwen_uniform_q3_group128_matvec_geo_tpr64_tg128` against the
//! f32 reconstruct-then-GEMV oracle. Packed bytes stay packed. Never writes
//! dense W on the GEMV path.
//!
//! ```text
//! cargo run -p hawking-core --example q3_g128_parity -- --synthetic
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("q3_g128_parity requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::metal::all_shader_sources;
    use hawking_core::model::qwen_complete_binary::{
        deterministic_matrix, pack_uniform_factor, uniform_factor_matvec_f32,
    };
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
    use std::env;
    use std::error::Error;

    const PASS_TOL: f32 = 2e-2;
    const SYNTH_ROWS: usize = 64;
    const SYNTH_COLS: usize = 256;
    const GROUP: usize = 128;
    const BITS: u8 = 3;
    const KERNEL: &str = "qwen_uniform_q3_group128_matvec_geo_tpr64_tg128";

    fn as_bytes_f32(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn as_bytes_u16(values: &[u16]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 2) }
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

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--synthetic" => {}
                other => return Err(format!("unsupported option {other}").into()),
            }
        }

        let values = deterministic_matrix(SYNTH_ROWS, SYNTH_COLS, 41);
        let packed = pack_uniform_factor(&values, SYNTH_ROWS, SYNTH_COLS, BITS, GROUP)?;
        assert_eq!(packed.bits, BITS);
        assert_eq!(packed.group_size, GROUP);
        assert_eq!(packed.bound, 3);
        let input = deterministic_input(packed.cols);
        let cpu_y = uniform_factor_matvec_f32(&packed, &input)?;

        let device = Device::system_default().ok_or("no Metal-capable GPU")?;
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let src = all_shader_sources();
        let library = device
            .new_library_with_source(&src, &opts)
            .map_err(|e| format!("q3 g128 shader compile: {e}"))?;
        let geo_fn = library
            .get_function(KERNEL, None)
            .map_err(|e| format!("{KERNEL}: {e}"))?;
        let geo_pipe = device.new_compute_pipeline_state_with_function(&geo_fn)?;

        let codes_buf = device.new_buffer_with_data(
            packed.codes.as_ptr() as *const _,
            packed.codes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scales_buf = device.new_buffer_with_data(
            as_bytes_u16(&packed.scales_f16).as_ptr() as *const _,
            as_bytes_u16(&packed.scales_f16).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buf = device.new_buffer_with_data(
            as_bytes_f32(&input).as_ptr() as *const _,
            as_bytes_f32(&input).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_geo_buf = device.new_buffer(
            (packed.rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let rows = packed.rows as u32;
        let cols = packed.cols as u32;
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&geo_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&scales_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(&y_geo_buf), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            set_u32(enc, 6, GROUP as u32);
            set_u32(enc, 7, u32::from(BITS));
            set_u32(enc, 8, u32::from(packed.bound));
            let groups = (packed.rows as u64).div_ceil(2);
            enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let gpu_y = read_f32(&y_geo_buf, packed.rows);
        let max_abs = max_abs_diff(&cpu_y, &gpu_y);
        let status = if max_abs <= PASS_TOL { "PASS" } else { "FAIL" };
        println!("kernel: {KERNEL}");
        println!("status: {status}");
        println!("max_abs_diff: {max_abs}");
        println!("rows: {}", packed.rows);
        println!("cols: {}", packed.cols);
        println!("bits: {BITS}");
        println!("group_size: {GROUP}");
        println!("dense_w_materialized: 0");
        println!("oracle: reconstruct_then_matvec_cpu");
        if max_abs > PASS_TOL {
            return Err(format!("q3 g128 geo max_abs_diff {max_abs} > {PASS_TOL}").into());
        }
        Ok(())
    }
}
