//! G003 cheapest discriminator: name the resource that holds Q80 mixed matvec
//! at 2.57 GB/s. Isolated live organs, occupancy, uncompressed same-launch.
//!
//!   cargo build --release -p hawking-core --example q80_matvec_discriminator
//!   ./tools/gpu_lane_lock.sh q80-matvec-diagnosis \
//!     $CARGO_TARGET_DIR/release/examples/q80_matvec_discriminator \
//!     --out receipts/ascent-2026-08-16/Q80_MATVEC_DISCRIMINATOR.json

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
    use hawking_core::model::qwen80_mixed_catalog::Qwen80MixedStreamingCatalog;
    use hawking_core::model::qwen_complete_binary::{
        binary_group_matvec_f32, deterministic_input, max_abs_error, uniform_factor_matvec_f32,
        MixedPackedTensor, UniformFactorPacked,
    };
    use metal::{CompileOptions, ComputePipelineState, Device, MTLSize};
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    const SCHEMA: &str = "hawking.ascension.q80_matvec_discriminator.v1";
    const SHADER: &str = include_str!("q80_matvec_discriminator.metal");
    const DEFAULT_ROOT: &str = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/mixed-1p5-v1";
    const HONEST_CEILING_GBPS: f64 = 411.51;
    const CONTROL_TGS: u32 = 60;
    const CONTROL_TPTG: u32 = 256;
    const CONTROL_ITERS: u32 = 4096;
    const CONTROL_BYTES: u64 = 256 * 1024 * 1024;
    const WARMUP: usize = 2;
    const REPS: usize = 3;
    const Q8_TOL: f32 = 2e-4;
    const BIN_TOL: f32 = 2e-5;

    struct Args {
        root: PathBuf,
        out: PathBuf,
        skip_lm_head: bool,
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let mut root = PathBuf::from(DEFAULT_ROOT);
        let mut out = PathBuf::from("receipts/ascent-2026-08-16/Q80_MATVEC_DISCRIMINATOR.json");
        let mut skip_lm_head = false;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--root" => root = PathBuf::from(args.next().ok_or("missing --root")?),
                "--out" => out = PathBuf::from(args.next().ok_or("missing --out")?),
                "--skip-lm-head" => skip_lm_head = true,
                other => return Err(format!("unsupported option {other}").into()),
            }
        }
        Ok(Args {
            root,
            out,
            skip_lm_head,
        })
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

    fn median(values: &[u64]) -> u64 {
        let mut v = values.to_vec();
        v.sort_unstable();
        v[v.len() / 2]
    }

    fn gbps(bytes: u64, ns: u64) -> f64 {
        if ns == 0 {
            return 0.0;
        }
        (bytes as f64) / (ns as f64) * 1e9 / 1e9
    }

    fn occupancy_json(pipe: &ComputePipelineState, rows: u32, tptg: u32, threads: u32) -> Value {
        let max_tg = pipe.max_total_threads_per_threadgroup();
        let width = pipe.thread_execution_width();
        let tg_mem = pipe.static_threadgroup_memory_length();
        let tgs = rows.div_ceil(tptg.max(1));
        json!({
            "threads_per_threadgroup_launched": tptg,
            "threadgroups": tgs,
            "threads_launched": threads,
            "thread_execution_width": width,
            "max_total_threads_per_threadgroup": max_tg,
            "static_threadgroup_memory_length": tg_mem,
            "occupancy_vs_1024_from_registers_or_tgmem": max_tg as f64 / 1024.0,
            "occupancy_proxy_vs_control_15360": threads as f64 / (CONTROL_TGS as f64 * CONTROL_TPTG as f64),
            "register_count_exposed": false,
            "register_pressure_limits_tg": max_tg < 1024,
        })
    }

    fn gpu_ns_from_cmd(cmd: &metal::CommandBufferRef) -> Result<u64, Box<dyn Error>> {
        use metal::objc::{msg_send, sel, sel_impl};
        let start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
        let end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
        if end > start && start > 0.0 {
            Ok(((end - start) * 1e9) as u64)
        } else {
            Err("missing GPU timestamp".into())
        }
    }

    struct DiscLib {
        device: Device,
        queue: metal::CommandQueue,
        lib: metal::Library,
    }

    impl DiscLib {
        fn compile(device: &Device) -> Result<Self, Box<dyn Error>> {
            let opts = CompileOptions::new();
            let lib = device
                .new_library_with_source(SHADER, &opts)
                .map_err(|e| format!("discriminator shader compile: {e}"))?;
            Ok(Self {
                device: device.clone(),
                queue: device.new_command_queue(),
                lib,
            })
        }

        fn pipe(&self, name: &str) -> Result<ComputePipelineState, Box<dyn Error>> {
            let f = self
                .lib
                .get_function(name, None)
                .map_err(|e| format!("{name}: {e}"))?;
            self.device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}").into())
        }

        fn time(
            &self,
            pipe: &ComputePipelineState,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl Fn(&metal::ComputeCommandEncoderRef),
        ) -> Result<u64, Box<dyn Error>> {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            gpu_ns_from_cmd(cmd)
        }
    }

    fn time_shipped(
        ctx: &MetalContext,
        name: &str,
        grid: (u32, u32, u32),
        tg: (u32, u32, u32),
        encode: impl Fn(&metal::ComputeCommandEncoderRef),
    ) -> Result<u64, Box<dyn Error>> {
        let t = ctx.dispatch_threads_timed(name, grid, tg, encode)?;
        match (t.gpu_start_ns, t.gpu_end_ns) {
            (Some(s), Some(e)) if e > s => Ok(e - s),
            _ => t
                .gpu_duration_us
                .map(|us| us.saturating_mul(1000))
                .ok_or_else(|| format!("{name}: no GPU timestamp").into()),
        }
    }

    fn unique_f32(rows: usize, cols: usize) -> Vec<f32> {
        let n = rows.saturating_mul(cols);
        let mut w = vec![0.0f32; n];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = ((i as u32).wrapping_mul(1_664_525).wrapping_add(101_390_4223) >> 9) as f32
                * (1.0 / 8_388_608.0)
                - 0.5;
        }
        w
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

    fn measure_ns(
        warmup: usize,
        reps: usize,
        mut once: impl FnMut() -> Result<u64, Box<dyn Error>>,
    ) -> Result<Vec<u64>, Box<dyn Error>> {
        for _ in 0..warmup {
            let _ = once()?;
        }
        let mut out = Vec::with_capacity(reps);
        for _ in 0..reps {
            out.push(once()?);
        }
        Ok(out)
    }

    fn variant_json(
        name: &str,
        ns: &[u64],
        packed_bytes: u64,
        dram_bytes: u64,
        occupancy: Value,
        max_abs: Option<f32>,
        note: &str,
    ) -> Value {
        let med = median(ns);
        json!({
            "kernel": name,
            "gpu_ns": ns,
            "median_gpu_ns": med,
            "packed_gbps": gbps(packed_bytes, med),
            "dram_gbps": gbps(dram_bytes, med),
            "frac_of_honest_ceiling_packed": gbps(packed_bytes, med) / HONEST_CEILING_GBPS,
            "occupancy": occupancy,
            "max_abs_error": max_abs,
            "note": note,
        })
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let catalog = Qwen80MixedStreamingCatalog::open(&args.root)?;

        let ctx = MetalContext::new()?;
        let device_name = ctx.device_name();
        let disc = DiscLib::compile(ctx.device())?;

        let shipped_names = [
            "q80_binary_group_matvec",
            "q80_binary_group_matvec_simd",
            "q80_binary_group_matvec_tg256",
            "q80_hgravs01_factor_matvec",
            "q80_hgravs01_factor_matvec_simd",
            "q80_hgravs01_factor_matvec_simd3",
            "gk_matvec_binary",
            "gk_matvec_binary_simd",
            "gk_matvec_hgravs",
            "gk_matvec_hgravs_simd",
        ];
        let mut shipped_occ = serde_json::Map::new();
        for name in shipped_names {
            let pipe = ctx.pipeline(name)?;
            shipped_occ.insert(
                name.to_string(),
                json!({
                    "thread_execution_width": pipe.thread_execution_width(),
                    "max_total_threads_per_threadgroup": pipe.max_total_threads_per_threadgroup(),
                    "static_threadgroup_memory_length": pipe.static_threadgroup_memory_length(),
                    "occupancy_vs_1024": pipe.max_total_threads_per_threadgroup() as f64 / 1024.0,
                    "register_pressure_limits_tg": pipe.max_total_threads_per_threadgroup() < 1024,
                }),
            );
        }

        let disc_names = [
            "disc_stream_control",
            "disc_f32_serial",
            "disc_f32_simd",
            "disc_f32_tg256",
            "disc_x_only_serial",
            "disc_q8_bit_serial",
            "disc_q8_byte_serial",
            "disc_q8_byte_simd",
            "disc_q8_byte_tg256",
            "disc_load_only_serial",
        ];
        let mut disc_occ = serde_json::Map::new();
        let mut disc_pipes = std::collections::HashMap::new();
        for name in disc_names {
            let pipe = disc.pipe(name)?;
            disc_occ.insert(
                name.to_string(),
                json!({
                    "thread_execution_width": pipe.thread_execution_width(),
                    "max_total_threads_per_threadgroup": pipe.max_total_threads_per_threadgroup(),
                    "static_threadgroup_memory_length": pipe.static_threadgroup_memory_length(),
                    "occupancy_vs_1024": pipe.max_total_threads_per_threadgroup() as f64 / 1024.0,
                    "register_pressure_limits_tg": pipe.max_total_threads_per_threadgroup() < 1024,
                }),
            );
            disc_pipes.insert(name, pipe);
        }

        let control_nthreads = CONTROL_TGS * CONTROL_TPTG;
        let mut control_data = vec![0u8; CONTROL_BYTES as usize];
        for (i, b) in control_data.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31).wrapping_add(7);
        }
        let control_buf = ctx.new_buffer_with_bytes_checked(&control_data)?;
        let control_out = ctx.new_buffer_checked(control_nthreads as usize * 4)?;
        let control_pipe = &disc_pipes["disc_stream_control"];
        let control_ns = measure_ns(WARMUP, REPS, || {
            disc.time(
                control_pipe,
                (control_nthreads, 1, 1),
                (CONTROL_TPTG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&control_buf), 0);
                    enc.set_buffer(1, Some(&control_out), 0);
                    set_u32(enc, 2, CONTROL_BYTES as u32);
                    set_u32(enc, 3, CONTROL_ITERS);
                },
            )
        })?;
        let control_traffic = control_nthreads as u64 * CONTROL_ITERS as u64 * 16;
        let control_med = median(&control_ns);
        let control_gbps = gbps(control_traffic, control_med);

        let mut organs = vec![
            (
                "binary_gate_512x2048",
                "model.layers.0.mlp.experts.0.gate_proj.weight",
            ),
            (
                "hgravs_down_2048x512",
                "model.layers.0.mlp.experts.0.down_proj.weight",
            ),
            (
                "q8_qkvz_12288x2048",
                "model.layers.0.linear_attn.in_proj_qkvz.weight",
            ),
            (
                "q8_out_2048x4096",
                "model.layers.0.linear_attn.out_proj.weight",
            ),
            (
                "q8_ba_64x2048",
                "model.layers.0.linear_attn.in_proj_ba.weight",
            ),
            (
                "q8_router_512x2048",
                "model.layers.0.mlp.gate.weight",
            ),
            (
                "q8_gqa_q_8192x2048",
                "model.layers.3.self_attn.q_proj.weight",
            ),
        ];
        if !args.skip_lm_head {
            organs.push(("q8_lm_head_151936x2048", "lm_head.weight"));
        }

        let mut organ_receipts = serde_json::Map::new();
        for (label, name) in organs {
            let packed = catalog.load_packed(name)?;
            let row = catalog.require_row(name)?;
            match packed {
                MixedPackedTensor::Binary(bin) => {
                    organ_receipts.insert(
                        label.to_string(),
                        measure_binary(&ctx, &disc, &disc_pipes, name, &bin, row.nbytes)?,
                    );
                }
                MixedPackedTensor::Hgravs { left, right } => {
                    organ_receipts.insert(
                        format!("{label}_R"),
                        measure_uniform(
                            &ctx,
                            &disc,
                            &disc_pipes,
                            &format!("{name}#R"),
                            &right,
                            "hgravs_R_3bit",
                        )?,
                    );
                    organ_receipts.insert(
                        format!("{label}_L"),
                        measure_uniform(
                            &ctx,
                            &disc,
                            &disc_pipes,
                            &format!("{name}#L"),
                            &left,
                            "hgravs_L_3bit",
                        )?,
                    );
                }
                MixedPackedTensor::Uniform8(u) => {
                    organ_receipts.insert(
                        label.to_string(),
                        measure_uniform(&ctx, &disc, &disc_pipes, name, &u, "uniform8")?,
                    );
                }
                MixedPackedTensor::Residual(_) => {
                    organ_receipts.insert(
                        label.to_string(),
                        json!({"skipped": "residual organ; binary half is the gate twin"}),
                    );
                }
            }
        }

        let diagnosis = diagnose(&organ_receipts, control_gbps);

        let receipt = json!({
            "schema": SCHEMA,
            "lane": "q80-matvec-diagnosis",
            "measurement_label": "DIRTY_ENGINEERING",
            "gpu_time_authority": "MTLCommandBuffer.GPUEndTime-GPUStartTime after wait",
            "device_name": device_name,
            "root": args.root,
            "honest_ceiling_gbps": HONEST_CEILING_GBPS,
            "control": {
                "kernel": "disc_stream_control",
                "median_gpu_ns": control_med,
                "gpu_ns": control_ns,
                "gbps": control_gbps,
                "traffic_bytes": control_traffic,
                "threadgroups": CONTROL_TGS,
                "threads_per_threadgroup": CONTROL_TPTG,
            },
            "shipped_pipeline_occupancy": shipped_occ,
            "diagnostic_pipeline_occupancy": disc_occ,
            "organs": organ_receipts,
            "diagnosis": diagnosis,
            "sibling_reconstruction_lane_premise": {
                "stated": "mixed codecs (binary/rice/low-rank) need expensive per-weight reconstruction; that is why mixed is 5.9x slower per byte than Q4",
                "mass_fact_from_catalog": "attention + lm_head on mixed-1p5-v1 are HGRAVU01 q8, not binary/rice/hgravs. Routed mixed organs are ~0.10 GB of the 2.218 GB token.",
            },
        });

        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&args.out, serde_json::to_string_pretty(&receipt)?)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        Ok(())
    }

    fn measure_binary(
        ctx: &MetalContext,
        disc: &DiscLib,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        name: &str,
        packed: &hawking_core::model::qwen_complete_binary::BinaryGroupPacked,
        nbytes: u64,
    ) -> Result<Value, Box<dyn Error>> {
        let rows = packed.rows as u32;
        let cols = packed.cols as u32;
        let x = deterministic_input(packed.cols);
        let oracle = binary_group_matvec_f32(packed, &x)?;
        let signs = ctx.new_buffer_with_bytes_checked(&packed.signs)?;
        let scales = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&packed.scales_f16))?;
        let x_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&x))?;
        let out = ctx.new_buffer_checked(packed.rows * 4)?;
        let zeros = vec![0.0f32; packed.rows];
        let w = unique_f32(packed.rows, packed.cols);
        let w_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&w))?;
        let packed_bytes = (packed.signs.len() + packed.scales_f16.len() * 2) as u64;
        let f32_bytes = (packed.rows * packed.cols * 4) as u64;
        let dram = packed_bytes + (packed.cols * 4 + packed.rows * 4) as u64;
        let serial_grid = (rows, 1, 1);
        let serial_tg = (256u32, 1, 1);
        let simd_grid = (rows.div_ceil(8) * 256, 1, 1);
        let tg256_grid = (rows * 256, 1, 1);

        let bind_bin = |enc: &metal::ComputeCommandEncoderRef| {
            enc.set_buffer(0, Some(&signs), 0);
            enc.set_buffer(1, Some(&scales), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&out), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            set_u32(enc, 6, packed.group_size as u32);
            set_u32(enc, 7, packed.groups_per_row as u32);
        };

        write_f32(&out, &zeros);
        let _ = time_shipped(
            ctx,
            "q80_binary_group_matvec",
            serial_grid,
            serial_tg,
            |enc| bind_bin(enc),
        )?;
        let got = read_f32(&out, packed.rows);
        let err_q80 = max_abs_error(&got, &oracle);

        write_f32(&out, &zeros);
        let _ = time_shipped(ctx, "gk_matvec_binary", serial_grid, serial_tg, |enc| {
            bind_bin(enc)
        })?;
        let err_gk = max_abs_error(&read_f32(&out, packed.rows), &oracle);

        let q80_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            time_shipped(
                ctx,
                "q80_binary_group_matvec",
                serial_grid,
                serial_tg,
                |enc| bind_bin(enc),
            )
        })?;
        let gk_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            time_shipped(ctx, "gk_matvec_binary", serial_grid, serial_tg, |enc| {
                bind_bin(enc)
            })
        })?;
        let gk_simd_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            time_shipped(ctx, "gk_matvec_binary_simd", simd_grid, serial_tg, |enc| {
                bind_bin(enc)
            })
        })?;
        let f32_serial_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            disc.time(&pipes["disc_f32_serial"], serial_grid, serial_tg, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out), 0);
                set_u32(enc, 3, rows);
                set_u32(enc, 4, cols);
            })
        })?;
        let f32_simd_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            disc.time(&pipes["disc_f32_simd"], simd_grid, serial_tg, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out), 0);
                set_u32(enc, 3, rows);
                set_u32(enc, 4, cols);
            })
        })?;
        let f32_tg256_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            disc.time(&pipes["disc_f32_tg256"], tg256_grid, serial_tg, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out), 0);
                set_u32(enc, 3, rows);
                set_u32(enc, 4, cols);
            })
        })?;
        let x_only_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            disc.time(&pipes["disc_x_only_serial"], serial_grid, serial_tg, |enc| {
                enc.set_buffer(0, Some(&x_buf), 0);
                enc.set_buffer(1, Some(&out), 0);
                set_u32(enc, 2, rows);
                set_u32(enc, 3, cols);
            })
        })?;

        let q80_pipe = ctx.pipeline("q80_binary_group_matvec")?;
        let gk_pipe = ctx.pipeline("gk_matvec_binary")?;
        let gk_simd_pipe = ctx.pipeline("gk_matvec_binary_simd")?;
        Ok(json!({
            "name": name,
            "kind": "binary_group",
            "shape": [packed.rows, packed.cols],
            "catalog_nbytes": nbytes,
            "packed_weight_bytes": packed_bytes,
            "uncompressed_f32_bytes": f32_bytes,
            "correctness": {
                "q80_binary_group_matvec_max_abs": err_q80,
                "gk_matvec_binary_max_abs": err_gk,
                "tolerance": BIN_TOL,
                "passed": err_q80 <= BIN_TOL && err_gk <= BIN_TOL,
            },
            "variants": [
                variant_json("q80_binary_group_matvec", &q80_ns, packed_bytes, dram,
                    occupancy_json(&q80_pipe, rows, 256, rows), Some(err_q80),
                    "shipped serial 1-thread/row"),
                variant_json("gk_matvec_binary", &gk_ns, packed_bytes, dram,
                    occupancy_json(&gk_pipe, rows, 256, rows), Some(err_gk),
                    "gk family serial; same association"),
                variant_json("gk_matvec_binary_simd", &gk_simd_ns, packed_bytes, dram,
                    occupancy_json(&gk_simd_pipe, rows, 256, rows.div_ceil(8)*256), None,
                    "gk family 1-SG/row"),
                variant_json("disc_f32_serial", &f32_serial_ns, f32_bytes, f32_bytes + (cols as u64+rows as u64)*4,
                    occupancy_json(&pipes["disc_f32_serial"], rows, 256, rows), None,
                    "uncompressed SAME launch"),
                variant_json("disc_f32_simd", &f32_simd_ns, f32_bytes, f32_bytes + (cols as u64+rows as u64)*4,
                    occupancy_json(&pipes["disc_f32_simd"], rows, 256, rows.div_ceil(8)*256), None,
                    "uncompressed coalesced"),
                variant_json("disc_f32_tg256", &f32_tg256_ns, f32_bytes, f32_bytes + (cols as u64+rows as u64)*4,
                    occupancy_json(&pipes["disc_f32_tg256"], rows, 256, rows*256), None,
                    "uncompressed 256 threads/row"),
                variant_json("disc_x_only_serial", &x_only_ns, (cols*4) as u64, (cols*4) as u64,
                    occupancy_json(&pipes["disc_x_only_serial"], rows, 256, rows), None,
                    "no weight traffic; issue/latency of the serial loop"),
            ],
        }))
    }

    fn measure_uniform(
        ctx: &MetalContext,
        disc: &DiscLib,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        name: &str,
        packed: &UniformFactorPacked,
        kind: &str,
    ) -> Result<Value, Box<dyn Error>> {
        let rows = packed.rows as u32;
        let cols = packed.cols as u32;
        let x = deterministic_input(packed.cols);
        let oracle = uniform_factor_matvec_f32(packed, &x)?;
        let codes = ctx.new_buffer_with_bytes_checked(&packed.codes)?;
        let scales = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&packed.scales_f16))?;
        let x_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&x))?;
        let out = ctx.new_buffer_checked(packed.rows * 4)?;
        let zeros = vec![0.0f32; packed.rows];
        let w = unique_f32(packed.rows, packed.cols);
        let w_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&w))?;
        let packed_bytes = (packed.codes.len() + packed.scales_f16.len() * 2) as u64;
        let f32_bytes = (packed.rows * packed.cols * 4) as u64;
        let dram = packed_bytes + (packed.cols * 4 + packed.rows * 4) as u64;
        let serial_grid = (rows, 1, 1);
        let serial_tg = (256u32, 1, 1);
        let simd_grid = (rows.div_ceil(8) * 256, 1, 1);
        let tg256_grid = (rows * 256, 1, 1);

        write_f32(&out, &zeros);
        let _ = time_shipped(ctx, "q80_hgravs01_factor_matvec", serial_grid, serial_tg, |enc| {
            encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
        })?;
        let err_q80 = max_abs_error(&read_f32(&out, packed.rows), &oracle);
        write_f32(&out, &zeros);
        let _ = time_shipped(ctx, "gk_matvec_hgravs", serial_grid, serial_tg, |enc| {
            encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
        })?;
        let err_gk = max_abs_error(&read_f32(&out, packed.rows), &oracle);

        let q80_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            time_shipped(ctx, "q80_hgravs01_factor_matvec", serial_grid, serial_tg, |enc| {
                encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
            })
        })?;
        let gk_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            time_shipped(ctx, "gk_matvec_hgravs", serial_grid, serial_tg, |enc| {
                encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
            })
        })?;
        let gk_simd_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            time_shipped(ctx, "gk_matvec_hgravs_simd", simd_grid, serial_tg, |enc| {
                encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
            })
        })?;
        let bit_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            disc.time(&pipes["disc_q8_bit_serial"], serial_grid, serial_tg, |enc| {
                encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
            })
        })?;
        let byte_ns = if packed.bits == 8 {
            Some(measure_ns(WARMUP, REPS, || {
                write_f32(&out, &zeros);
                disc.time(&pipes["disc_q8_byte_serial"], serial_grid, serial_tg, |enc| {
                    encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
                })
            })?)
        } else {
            None
        };
        let byte_simd_ns = if packed.bits == 8 {
            Some(measure_ns(WARMUP, REPS, || {
                write_f32(&out, &zeros);
                disc.time(&pipes["disc_q8_byte_simd"], simd_grid, serial_tg, |enc| {
                    encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
                })
            })?)
        } else {
            None
        };
        let byte_tg_ns = if packed.bits == 8 {
            Some(measure_ns(WARMUP, REPS, || {
                write_f32(&out, &zeros);
                disc.time(&pipes["disc_q8_byte_tg256"], tg256_grid, serial_tg, |enc| {
                    encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
                })
            })?)
        } else {
            None
        };
        let load_ns = if packed.bits == 8 {
            Some(measure_ns(WARMUP, REPS, || {
                write_f32(&out, &zeros);
                disc.time(&pipes["disc_load_only_serial"], serial_grid, serial_tg, |enc| {
                    enc.set_buffer(0, Some(&codes), 0);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(&out), 0);
                    set_u32(enc, 3, rows);
                    set_u32(enc, 4, cols);
                })
            })?)
        } else {
            None
        };
        let f32_serial_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            disc.time(&pipes["disc_f32_serial"], serial_grid, serial_tg, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out), 0);
                set_u32(enc, 3, rows);
                set_u32(enc, 4, cols);
            })
        })?;
        let f32_simd_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            disc.time(&pipes["disc_f32_simd"], simd_grid, serial_tg, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out), 0);
                set_u32(enc, 3, rows);
                set_u32(enc, 4, cols);
            })
        })?;
        let f32_tg256_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            disc.time(&pipes["disc_f32_tg256"], tg256_grid, serial_tg, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out), 0);
                set_u32(enc, 3, rows);
                set_u32(enc, 4, cols);
            })
        })?;
        let x_only_ns = measure_ns(WARMUP, REPS, || {
            write_f32(&out, &zeros);
            disc.time(&pipes["disc_x_only_serial"], serial_grid, serial_tg, |enc| {
                enc.set_buffer(0, Some(&x_buf), 0);
                enc.set_buffer(1, Some(&out), 0);
                set_u32(enc, 2, rows);
                set_u32(enc, 3, cols);
            })
        })?;

        let q80_pipe = ctx.pipeline("q80_hgravs01_factor_matvec")?;
        let gk_pipe = ctx.pipeline("gk_matvec_hgravs")?;
        let gk_simd_pipe = ctx.pipeline("gk_matvec_hgravs_simd")?;
        let mut variants = vec![
            variant_json(
                "q80_hgravs01_factor_matvec",
                &q80_ns,
                packed_bytes,
                dram,
                occupancy_json(&q80_pipe, rows, 256, rows),
                Some(err_q80),
                "shipped serial; Q8 uses 8-iteration bit extract",
            ),
            variant_json(
                "gk_matvec_hgravs",
                &gk_ns,
                packed_bytes,
                dram,
                occupancy_json(&gk_pipe, rows, 256, rows),
                Some(err_gk),
                "gk family serial",
            ),
            variant_json(
                "gk_matvec_hgravs_simd",
                &gk_simd_ns,
                packed_bytes,
                dram,
                occupancy_json(&gk_simd_pipe, rows, 256, rows.div_ceil(8) * 256),
                None,
                "gk family 1-SG/row, wide extract",
            ),
            variant_json(
                "disc_q8_bit_serial",
                &bit_ns,
                packed_bytes,
                dram,
                occupancy_json(&pipes["disc_q8_bit_serial"], rows, 256, rows),
                None,
                "same bit-serial extract, diagnostic lib",
            ),
            variant_json(
                "disc_f32_serial",
                &f32_serial_ns,
                f32_bytes,
                f32_bytes + (cols as u64 + rows as u64) * 4,
                occupancy_json(&pipes["disc_f32_serial"], rows, 256, rows),
                None,
                "uncompressed SAME 1-thread/row launch",
            ),
            variant_json(
                "disc_f32_simd",
                &f32_simd_ns,
                f32_bytes,
                f32_bytes + (cols as u64 + rows as u64) * 4,
                occupancy_json(&pipes["disc_f32_simd"], rows, 256, rows.div_ceil(8) * 256),
                None,
                "uncompressed coalesced",
            ),
            variant_json(
                "disc_f32_tg256",
                &f32_tg256_ns,
                f32_bytes,
                f32_bytes + (cols as u64 + rows as u64) * 4,
                occupancy_json(&pipes["disc_f32_tg256"], rows, 256, rows * 256),
                None,
                "uncompressed 256 threads/row",
            ),
            variant_json(
                "disc_x_only_serial",
                &x_only_ns,
                (cols * 4) as u64,
                (cols * 4) as u64,
                occupancy_json(&pipes["disc_x_only_serial"], rows, 256, rows),
                None,
                "serial loop, no weight traffic",
            ),
        ];
        if let Some(ns) = byte_ns {
            variants.insert(
                4,
                variant_json(
                    "disc_q8_byte_serial",
                    &ns,
                    packed_bytes,
                    dram,
                    occupancy_json(&pipes["disc_q8_byte_serial"], rows, 256, rows),
                    None,
                    "byte load, SAME 1-thread/row launch",
                ),
            );
        }
        if let Some(ns) = byte_simd_ns {
            variants.push(variant_json(
                "disc_q8_byte_simd",
                &ns,
                packed_bytes,
                dram,
                occupancy_json(&pipes["disc_q8_byte_simd"], rows, 256, rows.div_ceil(8) * 256),
                None,
                "byte load + coalesced",
            ));
        }
        if let Some(ns) = byte_tg_ns {
            variants.push(variant_json(
                "disc_q8_byte_tg256",
                &ns,
                packed_bytes,
                dram,
                occupancy_json(&pipes["disc_q8_byte_tg256"], rows, 256, rows * 256),
                None,
                "byte load + 256 threads/row",
            ));
        }
        if let Some(ns) = load_ns {
            variants.push(variant_json(
                "disc_load_only_serial",
                &ns,
                packed.codes.len() as u64,
                packed.codes.len() as u64 + (cols as u64) * 4,
                occupancy_json(&pipes["disc_load_only_serial"], rows, 256, rows),
                None,
                "load packed bytes, no scale/decode",
            ));
        }

        Ok(json!({
            "name": name,
            "kind": kind,
            "shape": [packed.rows, packed.cols],
            "bits": packed.bits,
            "group_size": packed.group_size,
            "bound": packed.bound,
            "packed_weight_bytes": packed_bytes,
            "uncompressed_f32_bytes": f32_bytes,
            "correctness": {
                "q80_hgravs01_factor_matvec_max_abs": err_q80,
                "gk_matvec_hgravs_max_abs": err_gk,
                "tolerance": if packed.bits == 8 { Q8_TOL } else { BIN_TOL },
                "passed": err_q80 <= Q8_TOL && err_gk <= Q8_TOL,
            },
            "variants": variants,
        }))
    }

    fn diagnose(organs: &serde_json::Map<String, Value>, control_gbps: f64) -> Value {
        let mut notes = Vec::new();
        let mut q8_serial = Vec::new();
        let mut q8_f32_serial = Vec::new();
        let mut q8_f32_occ = Vec::new();
        let mut q8_byte_serial = Vec::new();
        for (name, organ) in organs {
            let Some(vars) = organ.get("variants").and_then(|v| v.as_array()) else {
                continue;
            };
            let kind = organ.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            let find = |k: &str| {
                vars.iter().find(|v| v.get("kernel").and_then(|s| s.as_str()) == Some(k))
            };
            if let Some(v) = find("q80_hgravs01_factor_matvec") {
                if kind == "uniform8" {
                    q8_serial.push((
                        name.clone(),
                        v.get("packed_gbps").and_then(|x| x.as_f64()).unwrap_or(0.0),
                    ));
                }
            }
            if let Some(v) = find("disc_f32_serial") {
                q8_f32_serial.push((
                    name.clone(),
                    v.get("packed_gbps").and_then(|x| x.as_f64()).unwrap_or(0.0),
                ));
            }
            if let Some(v) = find("disc_f32_tg256") {
                q8_f32_occ.push((
                    name.clone(),
                    v.get("packed_gbps").and_then(|x| x.as_f64()).unwrap_or(0.0),
                ));
            }
            if let Some(v) = find("disc_q8_byte_serial") {
                q8_byte_serial.push((
                    name.clone(),
                    v.get("packed_gbps").and_then(|x| x.as_f64()).unwrap_or(0.0),
                ));
            }
        }
        let mean = |xs: &[(String, f64)]| {
            if xs.is_empty() {
                0.0
            } else {
                xs.iter().map(|(_, g)| *g).sum::<f64>() / xs.len() as f64
            }
        };
        let shipped = mean(&q8_serial);
        let f32_serial = mean(&q8_f32_serial);
        let f32_occ = mean(&q8_f32_occ);
        let byte_serial = mean(&q8_byte_serial);
        let access_pattern = f32_serial < control_gbps * 0.15 && f32_occ > f32_serial * 3.0;
        let reconstruction_dominant = shipped > 0.0 && f32_serial > shipped * 3.0;
        let bitloop_dominant = byte_serial > shipped * 2.0 && shipped > 0.0;
        let resource = if reconstruction_dominant && !access_pattern {
            "issue / reconstruction ALU"
        } else if access_pattern && !reconstruction_dominant {
            "occupancy + uncoalesced 1-thread-per-row access"
        } else if access_pattern && bitloop_dominant {
            "occupancy/access primary; Q8 bit-extract is a secondary issue tax"
        } else if access_pattern && reconstruction_dominant {
            "both: serial access pattern and reconstruction"
        } else {
            "unresolved from means; inspect per-organ variants"
        };
        notes.push(format!(
            "Q8 shipped mean {shipped:.2} GB/s; f32 serial {f32_serial:.2}; f32 tg256 {f32_occ:.2}; q8-byte serial {byte_serial:.2}; control {control_gbps:.2}"
        ));
        json!({
            "saturated_resource": resource,
            "q8_shipped_mean_gbps": shipped,
            "f32_serial_mean_gbps": f32_serial,
            "f32_tg256_mean_gbps": f32_occ,
            "q8_byte_serial_mean_gbps": byte_serial,
            "access_pattern_indicated": access_pattern,
            "reconstruction_indicated": reconstruction_dominant,
            "bitloop_indicated": bitloop_dominant,
            "notes": notes,
        })
    }
}
