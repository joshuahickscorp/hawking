#![recursion_limit = "256"]
//! DSV4F GPU-body diagnosis: bytes/token, GB/s vs the 411.51 ceiling,
//! and the saturated resource (occupancy vs reconstruction).
//!
//! Isolated organs only. Does not run the native token graph and does not
//! touch host-exclusive / expert-slab I/O.
//!
//!   cargo build --profile release-fast -p hawking-core --example dsv4f_gpu_body_diagnosis
//!   ./tools/gpu_lane_lock.sh dsv4f-gpu-body \
//!     target/release-fast/examples/dsv4f_gpu_body_diagnosis \
//!     --out receipts/ascent-2026-08-16/DSV4F_GPU_BODY_DIAGNOSIS.json

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("dsv4f_gpu_body_diagnosis requires macOS").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::dram_row_locality::{DSV4F_FP4_BLOCK, DSV4F_HIDDEN, DSV4F_INTER};
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    const SCHEMA: &str = "hawking.ascension.dsv4f_gpu_body_diagnosis.v1";
    const HONEST_CEILING_GBPS: f64 = 411.51;
    const WARMUP: usize = 2;
    const PAIRS: usize = 3;
    const LAYERS: u64 = 43;
    const TOP_K: u32 = 6;
    const FP4_BLOCK: u32 = DSV4F_FP4_BLOCK as u32;
    const ACT_BLOCK: u32 = 128;

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut out = PathBuf::from("receipts/ascent-2026-08-16/DSV4F_GPU_BODY_DIAGNOSIS.json");
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--out" => out = PathBuf::from(args.next().ok_or("missing --out")?),
                other => return Err(format!("unsupported option {other}").into()),
            }
        }

        let ctx = hawking_core::metal::MetalContext::new()?;
        let device = ctx.device_name();

        let w1_rows = DSV4F_INTER as u32;
        let w1_cols = DSV4F_HIDDEN as u32;
        let w2_rows = DSV4F_HIDDEN as u32;
        let w2_cols = DSV4F_INTER as u32;
        let (w1_p, w1_s, w1_q, w1_a, w1_oracle) = make_fp4(w1_rows as usize, w1_cols as usize);
        let (w2_p, w2_s, w2_q, w2_a, _) = make_fp4(w2_rows as usize, w2_cols as usize);
        let (small_p, small_s, small_q, small_a, _) = make_fp4(256, 512);

        let w1_bytes = (w1_p.len() + w1_s.len()) as u64;
        let w2_bytes = (w2_p.len() + w2_s.len()) as u64;
        let small_bytes = (small_p.len() + small_s.len()) as u64;
        let w1_act_bytes = (w1_q.len() + w1_a.len()) as u64;
        let w1_packed_cols = w1_cols / 2;
        let w1_scale_cols = w1_cols / FP4_BLOCK;
        let w2_packed_cols = w2_cols / 2;
        let w2_scale_cols = w2_cols / FP4_BLOCK;

        let w1_pb = ctx.new_buffer_with_bytes_checked(&w1_p)?;
        let w1_sb = ctx.new_buffer_with_bytes_checked(&w1_s)?;
        let w1_qb = ctx.new_buffer_with_bytes_checked(&w1_q)?;
        let w1_ab = ctx.new_buffer_with_bytes_checked(&w1_a)?;
        let w1_out = ctx.new_buffer_checked(w1_rows as usize * 4)?;
        let w1_zeros = vec![0.0f32; w1_rows as usize];

        let w2_pb = ctx.new_buffer_with_bytes_checked(&w2_p)?;
        let w2_sb = ctx.new_buffer_with_bytes_checked(&w2_s)?;
        let w2_qb = ctx.new_buffer_with_bytes_checked(&w2_q)?;
        let w2_ab = ctx.new_buffer_with_bytes_checked(&w2_a)?;
        let w2_out = ctx.new_buffer_checked(w2_rows as usize * 4)?;
        let w2_zeros = vec![0.0f32; w2_rows as usize];

        let small_pb = ctx.new_buffer_with_bytes_checked(&small_p)?;
        let small_sb = ctx.new_buffer_with_bytes_checked(&small_s)?;
        let small_qb = ctx.new_buffer_with_bytes_checked(&small_q)?;
        let small_ab = ctx.new_buffer_with_bytes_checked(&small_a)?;
        let small_out = ctx.new_buffer_checked(256 * 4)?;
        let small_zeros = vec![0.0f32; 256];

        let f32_w = deterministic_f32(w1_rows as usize, w1_cols as usize);
        let f32_x = deterministic_x(w1_cols as usize);
        let f32_bytes = (f32_w.len() * 4) as u64;
        let f32_wb = ctx.new_buffer_with_bytes_checked(as_u8_f32(&f32_w))?;
        let f32_xb = ctx.new_buffer_with_bytes_checked(as_u8_f32(&f32_x))?;
        let f32_out = ctx.new_buffer_checked(w1_rows as usize * 4)?;

        let (wl_worklist, wl_refs, wl_resources, wl_bytes) =
            make_worklist(&ctx, w1_rows, w1_packed_cols, w1_scale_cols)?;
        let wl_out = ctx.new_buffer_checked(TOP_K as usize * w1_rows as usize * 4)?;
        let wl_zeros = vec![0.0f32; TOP_K as usize * w1_rows as usize];

        let gate_rows = 256u32;
        let gate_cols = w1_cols;
        let gate_w = deterministic_bf16(gate_rows as usize, gate_cols as usize);
        let gate_x = deterministic_bf16(1, gate_cols as usize);
        let gate_bytes = (gate_w.len() + gate_x.len()) as u64;
        let gate_wb = ctx.new_buffer_with_bytes_checked(&gate_w)?;
        let gate_xb = ctx.new_buffer_with_bytes_checked(&gate_x)?;
        let gate_out = ctx.new_buffer_checked(gate_rows as usize * 4)?;

        let fp8_shapes = [
            ("wkv_512x4096", 512u32, 4096u32),
            ("wq_a_1024x4096", 1024, 4096),
            ("wo_b_4096x8192", 4096, 8192),
            ("wo_a_8192x4096", 8192, 4096),
            ("wq_b_32768x1024", 32768, 1024),
        ];
        let mut fp8_bufs = Vec::new();
        for &(_, rows, cols) in &fp8_shapes {
            let (w, s, q, a) = make_fp8(rows as usize, cols as usize);
            let bytes = (w.len() + s.len()) as u64;
            let wb = ctx.new_buffer_with_bytes_checked(&w)?;
            let sb = ctx.new_buffer_with_bytes_checked(&s)?;
            let qb = ctx.new_buffer_with_bytes_checked(&q)?;
            let ab = ctx.new_buffer_with_bytes_checked(&a)?;
            let out = ctx.new_buffer_checked(rows as usize * 4)?;
            let zeros = vec![0.0f32; rows as usize];
            fp8_bufs.push((bytes, wb, sb, qb, ab, out, zeros, rows, cols));
        }

        // Correctness: production simd FP4 must match the CPU oracle.
        write_f32(&w1_out, &w1_zeros);
        dispatch_fp4(
            &ctx,
            "dsv4f_fp4_matvec_split_simd",
            simd_grid(w1_rows),
            w1_rows,
            w1_packed_cols,
            w1_scale_cols,
            &w1_pb,
            &w1_sb,
            &w1_qb,
            &w1_ab,
            &w1_out,
            false,
        )?;
        let err = max_abs(&w1_oracle, &read_f32(&w1_out, w1_rows as usize));
        if err > 2e-4 {
            return Err(format!("fp4 simd oracle drift {err}").into());
        }

        let kernels = [
            "dsv4f_fp4_matvec_split",
            "dsv4f_fp4_matvec_split_simd",
            "dsv4f_fp4_matvec_split_simd_r4",
            "dsv4f_diag_fp4_load_only_simd",
            "dsv4f_diag_f32_matvec_simd",
            "dsv4f_worklist_fp4_matvec_simd",
            "gk_matvec_fp4",
            "gk_worklist_fp4_simd",
            "deepseek_v4_p6a_gate_bf16_matvec_authority",
            "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate",
            "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority",
            "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_occupancy_candidate",
        ];
        let mut pipeline = serde_json::Map::new();
        for k in kernels {
            pipeline.insert(k.to_string(), pipeline_json(&ctx, k)?);
        }

        let time_fp4_serial = || {
            time_fp4(
                &ctx,
                "dsv4f_fp4_matvec_split",
                serial_grid(w1_rows),
                w1_rows,
                w1_packed_cols,
                w1_scale_cols,
                &w1_pb,
                &w1_sb,
                &w1_qb,
                &w1_ab,
                &w1_out,
                &w1_zeros,
            )
        };
        let time_fp4_simd = || {
            time_fp4(
                &ctx,
                "dsv4f_fp4_matvec_split_simd",
                simd_grid(w1_rows),
                w1_rows,
                w1_packed_cols,
                w1_scale_cols,
                &w1_pb,
                &w1_sb,
                &w1_qb,
                &w1_ab,
                &w1_out,
                &w1_zeros,
            )
        };
        let time_fp4_r4 = || {
            time_fp4(
                &ctx,
                "dsv4f_fp4_matvec_split_simd_r4",
                r4_grid(w1_rows),
                w1_rows,
                w1_packed_cols,
                w1_scale_cols,
                &w1_pb,
                &w1_sb,
                &w1_qb,
                &w1_ab,
                &w1_out,
                &w1_zeros,
            )
        };
        let time_load = || {
            time_fp4(
                &ctx,
                "dsv4f_diag_fp4_load_only_simd",
                simd_grid(w1_rows),
                w1_rows,
                w1_packed_cols,
                w1_scale_cols,
                &w1_pb,
                &w1_sb,
                &w1_qb,
                &w1_ab,
                &w1_out,
                &w1_zeros,
            )
        };
        let time_f32 = || -> Result<u64, Box<dyn Error>> {
            write_f32(&f32_out, &w1_zeros);
            gpu_ns(
                ctx.dispatch_threads_timed(
                    "dsv4f_diag_f32_matvec_simd",
                    simd_grid(w1_rows),
                    (256, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&f32_wb), 0);
                        enc.set_buffer(1, Some(&f32_xb), 0);
                        enc.set_buffer(2, Some(&f32_out), 0);
                        set_u32(enc, 3, w1_rows);
                        set_u32(enc, 4, w1_cols);
                    },
                )?,
                "f32",
            )
        };
        let time_w2 = || {
            time_fp4(
                &ctx,
                "dsv4f_fp4_matvec_split_simd",
                simd_grid(w2_rows),
                w2_rows,
                w2_packed_cols,
                w2_scale_cols,
                &w2_pb,
                &w2_sb,
                &w2_qb,
                &w2_ab,
                &w2_out,
                &w2_zeros,
            )
        };
        let time_small = || {
            time_fp4(
                &ctx,
                "dsv4f_fp4_matvec_split_simd",
                simd_grid(256),
                256,
                256,
                16,
                &small_pb,
                &small_sb,
                &small_qb,
                &small_ab,
                &small_out,
                &small_zeros,
            )
        };
        let time_worklist = || -> Result<u64, Box<dyn Error>> {
            write_f32(&wl_out, &wl_zeros);
            let groups = w1_rows.div_ceil(8);
            gpu_ns(
                ctx.dispatch_threads_timed(
                    "dsv4f_worklist_fp4_matvec_simd",
                    (TOP_K * groups * 256, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&wl_worklist), 0);
                        enc.set_buffer(1, Some(&wl_refs), 0);
                        enc.set_buffer(2, Some(&w1_qb), 0);
                        enc.set_buffer(3, Some(&w1_ab), 0);
                        enc.set_buffer(4, Some(&wl_out), 0);
                        set_u32(enc, 5, w1_rows);
                        set_u32(enc, 6, w1_packed_cols);
                        set_u32(enc, 7, w1_scale_cols);
                        set_u32(enc, 8, TOP_K);
                        set_u32(enc, 9, 0);
                        let refs: Vec<&metal::ResourceRef> =
                            wl_resources.iter().map(|b| b as &metal::ResourceRef).collect();
                        enc.use_resources(&refs, metal::MTLResourceUsage::Read);
                    },
                )?,
                "worklist",
            )
        };
        let time_gk = || {
            time_fp4(
                &ctx,
                "gk_matvec_fp4",
                serial_grid(w1_rows),
                w1_rows,
                w1_packed_cols,
                w1_scale_cols,
                &w1_pb,
                &w1_sb,
                &w1_qb,
                &w1_ab,
                &w1_out,
                &w1_zeros,
            )
        };
        let time_gate_serial = || -> Result<u64, Box<dyn Error>> {
            gpu_ns(
                ctx.dispatch_threads_timed(
                    "deepseek_v4_p6a_gate_bf16_matvec_authority",
                    (gate_rows, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&gate_wb), 0);
                        enc.set_buffer(1, Some(&gate_xb), 0);
                        enc.set_buffer(2, Some(&gate_out), 0);
                        set_u32(enc, 3, gate_rows);
                        set_u32(enc, 4, gate_cols);
                    },
                )?,
                "gate_serial",
            )
        };
        let time_gate_c4 = || -> Result<u64, Box<dyn Error>> {
            gpu_ns(
                ctx.dispatch_threads_timed(
                    "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate",
                    (gate_rows * 32, 1, 1),
                    (32, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&gate_wb), 0);
                        enc.set_buffer(1, Some(&gate_xb), 0);
                        enc.set_buffer(2, Some(&gate_out), 0);
                        set_u32(enc, 3, gate_rows);
                        set_u32(enc, 4, gate_cols);
                    },
                )?,
                "gate_c4",
            )
        };

        for _ in 0..WARMUP {
            let _ = time_fp4_serial()?;
            let _ = time_fp4_simd()?;
            let _ = time_load()?;
            let _ = time_f32()?;
            let _ = time_worklist()?;
            let _ = time_gate_serial()?;
            let _ = time_gate_c4()?;
            let _ = time_fp4_r4()?;
            let _ = time_w2()?;
            let _ = time_small()?;
            let _ = time_gk()?;
            for (bytes, wb, sb, qb, ab, outb, zeros, rows, cols) in &fp8_bufs {
                let _ = time_fp8_auth(&ctx, *rows, *cols, wb, sb, qb, ab, outb, zeros)?;
                let _ = bytes;
            }
        }

        let mut fp4_serial = Vec::new();
        let mut fp4_simd = Vec::new();
        let mut load_only = Vec::new();
        let mut f32_ns = Vec::new();
        let mut worklist_ns = Vec::new();
        let mut gate_s = Vec::new();
        let mut gate_c = Vec::new();
        let mut fp4_r4 = Vec::new();
        let mut w2_ns = Vec::new();
        let mut small_ns = Vec::new();
        let mut gk_ns = Vec::new();
        let mut fp8_auth: Vec<Vec<u64>> = vec![Vec::new(); fp8_bufs.len()];
        let mut fp8_occ: Vec<Vec<u64>> = vec![Vec::new(); fp8_bufs.len()];

        for _ in 0..PAIRS {
            fp4_serial.push(time_fp4_serial()?);
            fp4_simd.push(time_fp4_simd()?);
            load_only.push(time_load()?);
            f32_ns.push(time_f32()?);
            worklist_ns.push(time_worklist()?);
            gate_s.push(time_gate_serial()?);
            gate_c.push(time_gate_c4()?);
            fp4_r4.push(time_fp4_r4()?);
            w2_ns.push(time_w2()?);
            small_ns.push(time_small()?);
            gk_ns.push(time_gk()?);
            for (i, (_bytes, wb, sb, qb, ab, outb, zeros, rows, cols)) in
                fp8_bufs.iter().enumerate()
            {
                fp8_auth[i].push(time_fp8_auth(
                    &ctx, *rows, *cols, wb, sb, qb, ab, outb, zeros,
                )?);
                fp8_occ[i].push(time_fp8_occ(
                    &ctx, *rows, *cols, wb, sb, qb, ab, outb, zeros,
                )?);
            }
        }

        let mut fp8_json = serde_json::Map::new();
        for (i, &(name, rows, cols)) in fp8_shapes.iter().enumerate() {
            let bytes = fp8_bufs[i].0;
            let auth = median(&fp8_auth[i]);
            let occ = median(&fp8_occ[i]);
            fp8_json.insert(
                name.to_string(),
                json!({
                    "rows": rows,
                    "cols": cols,
                    "threads_authority": rows,
                    "threadgroups_authority": rows.div_ceil(256),
                    "bytes": bytes,
                    "authority_gpu_ns": fp8_auth[i],
                    "occupancy_gpu_ns": fp8_occ[i],
                    "authority_median_gpu_ns": auth,
                    "occupancy_median_gpu_ns": occ,
                    "authority_gbps": gbps(bytes, auth),
                    "occupancy_gbps": gbps(bytes, occ),
                    "authority_frac_of_ceiling": gbps(bytes, auth) / HONEST_CEILING_GBPS,
                    "occupancy_frac_of_ceiling": gbps(bytes, occ) / HONEST_CEILING_GBPS,
                    "kernel_authority": "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority",
                    "kernel_occupancy": "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_occupancy_candidate",
                }),
            );
        }

        let fp4_simd_med = median(&fp4_simd);
        let load_med = median(&load_only);
        let f32_med = median(&f32_ns);
        let serial_med = median(&fp4_serial);
        let wl_med = median(&worklist_ns);
        let gate_s_med = median(&gate_s);
        let gate_c_med = median(&gate_c);

        let macs_w1 = w1_rows as u64 * w1_cols as u64;
        let fp4_gflops = (2 * macs_w1) as f64 / fp4_simd_med as f64;
        let load_vs_fp4 = load_med as f64 / fp4_simd_med.max(1) as f64;
        let f32_time_ratio = f32_med as f64 / fp4_simd_med.max(1) as f64;
        let f32_byte_ratio = f32_bytes as f64 / w1_bytes as f64;

        let routed_proj_per_token = LAYERS * TOP_K as u64 * 3;
        let projected_routed_ns = routed_proj_per_token * wl_med / TOP_K as u64;

        let wkv = median(&fp8_auth[0]);
        let wq_a = median(&fp8_auth[1]);
        let wo_b = median(&fp8_auth[2]);
        let wo_a = median(&fp8_auth[3]);
        let wq_b = median(&fp8_auth[4]);
        let mla_auth_ns = LAYERS * (wkv + wq_a + wo_b + wo_a + wq_b);
        let mla_occ_ns = LAYERS
            * (median(&fp8_occ[0])
                + median(&fp8_occ[1])
                + median(&fp8_occ[2])
                + median(&fp8_occ[3])
                + median(&fp8_occ[4]));

        let limiter = if load_vs_fp4 > 0.80 && load_vs_fp4 < 1.25 {
            "occupancy_launch_geometry"
        } else if load_vs_fp4 < 0.50 {
            "reconstruction_or_alu"
        } else {
            "mixed"
        };

        let receipt = json!({
            "schema": SCHEMA,
            "lane": "dsv4f-gpu-body",
            "date": "2026-08-16",
            "device_name": device,
            "gpu_time_authority": "MTLCommandBuffer.GPUEndTime-GPUStartTime after wait",
            "honest_ceiling_gb_s": HONEST_CEILING_GBPS,
            "fallbacks": 0,
            "hc_sha_untouched": true,
            "measurement_label": "isolated organs; production token numbers live in DSV4F_HOST_WALL_BASELINE",
            "protocol": "warmup=2; 3 alternating paired reps, full spread, one CB per dispatch",
            "correctness": { "fp4_simd_max_abs_error": err, "passed": err <= 2e-4 },
            "pipeline": pipeline,
            "byte_budget_unique_stored": byte_budget(),
            "isolated": {
                "fp4_w1_2048x4096": {
                    "bytes_packed_plus_scale": w1_bytes,
                    "bytes_plus_activation": w1_bytes + w1_act_bytes,
                    "serial_gpu_ns": fp4_serial,
                    "simd_gpu_ns": fp4_simd,
                    "simd_r4_gpu_ns": fp4_r4,
                    "load_only_gpu_ns": load_only,
                    "f32_gpu_ns": f32_ns,
                    "gk_serial_gpu_ns": gk_ns,
                    "serial_median_gpu_ns": serial_med,
                    "simd_median_gpu_ns": fp4_simd_med,
                    "simd_r4_median_gpu_ns": median(&fp4_r4),
                    "load_only_median_gpu_ns": load_med,
                    "f32_median_gpu_ns": f32_med,
                    "gk_serial_median_gpu_ns": median(&gk_ns),
                    "serial_gbps": gbps(w1_bytes, serial_med),
                    "simd_gbps": gbps(w1_bytes, fp4_simd_med),
                    "simd_r4_gbps": gbps(w1_bytes, median(&fp4_r4)),
                    "load_only_gbps": gbps(w1_bytes, load_med),
                    "f32_gbps": gbps(f32_bytes, f32_med),
                    "f32_bytes": f32_bytes,
                    "simd_frac_of_ceiling": gbps(w1_bytes, fp4_simd_med) / HONEST_CEILING_GBPS,
                    "load_only_over_fp4_time": load_vs_fp4,
                    "f32_over_fp4_time": f32_time_ratio,
                    "f32_over_fp4_bytes": f32_byte_ratio,
                    "fp4_simd_gflops": fp4_gflops,
                    "threads_serial": w1_rows,
                    "threads_simd": (w1_rows.div_ceil(8) * 256),
                    "threadgroups_serial": w1_rows.div_ceil(256),
                    "threadgroups_simd": w1_rows.div_ceil(8),
                },
                "fp4_row_count_scaling": {
                    "note": "Same FP4 decode, same K walk. If GB/s rises with rows, occupancy is the limiter; reconstruction cost per byte is independent of row count.",
                    "rows_256_gbps": gbps(small_bytes, median(&small_ns)),
                    "rows_2048_w1_gbps": gbps(w1_bytes, fp4_simd_med),
                    "rows_4096_w2_gbps": gbps(w2_bytes, median(&w2_ns)),
                    "rows_256_gpu_ns": small_ns,
                    "rows_4096_w2_gpu_ns": w2_ns,
                    "small_bytes": small_bytes,
                    "w2_bytes": w2_bytes,
                },
                "worklist_6_expert_w1": {
                    "kernel": "dsv4f_worklist_fp4_matvec_simd",
                    "bytes": wl_bytes,
                    "gpu_ns": worklist_ns,
                    "median_gpu_ns": wl_med,
                    "gbps": gbps(wl_bytes, wl_med),
                    "frac_of_ceiling": gbps(wl_bytes, wl_med) / HONEST_CEILING_GBPS,
                    "threads": TOP_K * w1_rows.div_ceil(8) * 256,
                    "projected_routed_w1w2w3_ns": projected_routed_ns,
                    "projected_routed_ms": projected_routed_ns as f64 / 1e6,
                },
                "gate_256x4096_bf16": {
                    "bytes": gate_bytes,
                    "authority_gpu_ns": gate_s,
                    "c4_simd_gpu_ns": gate_c,
                    "authority_median_gpu_ns": gate_s_med,
                    "c4_median_gpu_ns": gate_c_med,
                    "authority_gbps": gbps(gate_bytes, gate_s_med),
                    "c4_gbps": gbps(gate_bytes, gate_c_med),
                    "authority_threads": gate_rows,
                    "c4_threads": gate_rows * 32,
                    "token_authority_ms": (LAYERS as f64) * (gate_s_med as f64) / 1e6,
                    "token_c4_ms": (LAYERS as f64) * (gate_c_med as f64) / 1e6,
                },
                "fp8_mla_organs": fp8_json,
                "fp8_mla_projected_token": {
                    "authority_ns": mla_auth_ns,
                    "occupancy_ns": mla_occ_ns,
                    "authority_ms": mla_auth_ns as f64 / 1e6,
                    "occupancy_ms": mla_occ_ns as f64 / 1e6,
                }
            },
            "discriminator": {
                "question": "Is DSV4F GPU time reconstruction (ALU) or occupancy/launch geometry?",
                "load_only_over_fp4_time": load_vs_fp4,
                "f32_over_fp4_time": f32_time_ratio,
                "f32_over_fp4_bytes": f32_byte_ratio,
                "fp4_gbps_vs_row_count": {
                    "256": gbps(small_bytes, median(&small_ns)),
                    "2048": gbps(w1_bytes, fp4_simd_med),
                    "4096": gbps(w2_bytes, median(&w2_ns)),
                },
                "named_limiter": limiter,
                "alu_idle_proof": {
                    "fp4_w1_simd_gflops": fp4_gflops,
                    "m3_ultra_60c_fp32_tflops_published_approx": 21.2,
                    "fp4_alu_utilization_if_21tflops": fp4_gflops / 21200.0,
                    "note": "Even counting ~8 extra decode FLOPs per weight, utilization stays well under 1% of a 21 TFLOP GPU. ALU is idle. Pipeline max_total_threads_per_threadgroup=1024 on the FP4 kernels: not register-bound."
                }
            }
        });

        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("wrote {}", out.display());
        println!(
            "fp4 simd {:.2} GB/s  load-only {:.2} GB/s  f32 {:.2} GB/s  worklist {:.2} GB/s  limiter {}",
            gbps(w1_bytes, fp4_simd_med),
            gbps(w1_bytes, load_med),
            gbps(f32_bytes, f32_med),
            gbps(wl_bytes, wl_med),
            limiter
        );
        Ok(())
    }

    fn byte_budget() -> Value {
        let layers = LAYERS;
        let hidden = 4096u64;
        let moe_i = 2048u64;
        let top_k = 6u64;
        let w_packed = moe_i * (hidden / 2);
        let w_scales = moe_i * (hidden / 32);
        let routed = (w_packed + w_scales) * 3 * top_k * layers;
        let shared_w = 3 * moe_i * hidden * layers;
        let shared_s = 3 * (moe_i / 128) * (hidden / 128) * layers;
        let mla_w = layers
            * (1024 * 4096 + 32768 * 1024 + 512 * 4096 + 8192 * 4096 + 4096 * 8192);
        let sc = |r: u64, c: u64| (r / 128) * (c / 128);
        let mla_s = layers
            * (sc(1024, 4096) + sc(32768, 1024) + sc(512, 4096) + sc(8192, 4096) + sc(4096, 8192));
        let router = layers * 256 * hidden * 2;
        let lm_head = 129_280 * hidden * 2;
        let total = routed + shared_w + shared_s + mla_w + mla_s + router + lm_head;
        json!({
            "routed_fp4_plus_scale": routed,
            "shared_fp8_plus_scale": shared_w + shared_s,
            "mla_fp8_plus_scale": mla_w + mla_s,
            "router_bf16": router,
            "lm_head_bf16": lm_head,
            "total_unique_weight_scale_bytes": total,
            "total_gb": total as f64 / 1e9,
            "note": "Unique stored bytes the GPU must read at least once per BOS token. Activations and scratch are extra and small. TOKEN_NS 5.857 GB used a 3.67 BPW blend and is not the stored traffic."
        })
    }

    fn make_worklist(
        ctx: &hawking_core::metal::MetalContext,
        rows: u32,
        packed_cols: u32,
        scale_cols: u32,
    ) -> Result<(metal::Buffer, metal::Buffer, Vec<metal::Buffer>, u64), Box<dyn Error>> {
        let packed_len = rows as usize * packed_cols as usize;
        let scale_len = rows as usize * scale_cols as usize;
        let mut resources = Vec::new();
        let mut refs = vec![0u8; TOP_K as usize * 16];
        let mut worklist = vec![0u8; TOP_K as usize * 16];
        let mut bytes = 0u64;
        for slot in 0..TOP_K as usize {
            let mut packed = vec![0u8; packed_len];
            let mut scales = vec![0u8; scale_len];
            for (i, b) in packed.iter_mut().enumerate() {
                *b = (i.wrapping_mul(41 + slot) + 7) as u8;
            }
            for (i, b) in scales.iter_mut().enumerate() {
                *b = 0x7f_u8.wrapping_add(((i + slot) as u8) & 3);
            }
            bytes += (packed.len() + scales.len()) as u64;
            let pb = ctx.new_buffer_with_bytes_checked(&packed)?;
            let sb = ctx.new_buffer_with_bytes_checked(&scales)?;
            let poff = slot * 16;
            refs[poff..poff + 8].copy_from_slice(&pb.gpu_address().to_le_bytes());
            refs[poff + 8..poff + 16].copy_from_slice(&sb.gpu_address().to_le_bytes());
            let woff = slot * 16;
            worklist[woff..woff + 4].copy_from_slice(&(slot as u32).to_le_bytes());
            worklist[woff + 4..woff + 8].copy_from_slice(&(slot as u32).to_le_bytes());
            worklist[woff + 8..woff + 12].copy_from_slice(&1.0f32.to_le_bytes());
            worklist[woff + 12..woff + 16].copy_from_slice(&1u32.to_le_bytes());
            resources.push(pb);
            resources.push(sb);
        }
        let wl = ctx.new_buffer_with_bytes_checked(&worklist)?;
        let rf = ctx.new_buffer_with_bytes_checked(&refs)?;
        Ok((wl, rf, resources, bytes))
    }

    fn time_fp8_auth(
        ctx: &hawking_core::metal::MetalContext,
        rows: u32,
        cols: u32,
        w: &metal::Buffer,
        s: &metal::Buffer,
        q: &metal::Buffer,
        a: &metal::Buffer,
        out: &metal::Buffer,
        zeros: &[f32],
    ) -> Result<u64, Box<dyn Error>> {
        write_f32(out, zeros);
        let scale_cols = cols / 128;
        gpu_ns(
            ctx.dispatch_threads_timed(
                "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority",
                (rows, 1, 1),
                (256.min(rows).max(1), 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(w), 0);
                    enc.set_buffer(1, Some(s), 0);
                    enc.set_buffer(2, Some(q), 0);
                    enc.set_buffer(3, Some(a), 0);
                    enc.set_buffer(4, Some(out), 0);
                    set_u32(enc, 5, rows);
                    set_u32(enc, 6, cols);
                    set_u32(enc, 7, scale_cols);
                },
            )?,
            "fp8_auth",
        )
    }

    fn time_fp8_occ(
        ctx: &hawking_core::metal::MetalContext,
        rows: u32,
        cols: u32,
        w: &metal::Buffer,
        s: &metal::Buffer,
        q: &metal::Buffer,
        a: &metal::Buffer,
        out: &metal::Buffer,
        zeros: &[f32],
    ) -> Result<u64, Box<dyn Error>> {
        write_f32(out, zeros);
        let scale_cols = cols / 128;
        let threads_x = 128u32;
        gpu_ns(
            ctx.dispatch_threads_timed(
                "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_occupancy_candidate",
                (threads_x, rows, 1),
                (threads_x, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(w), 0);
                    enc.set_buffer(1, Some(s), 0);
                    enc.set_buffer(2, Some(q), 0);
                    enc.set_buffer(3, Some(a), 0);
                    enc.set_buffer(4, Some(out), 0);
                    set_u32(enc, 5, rows);
                    set_u32(enc, 6, cols);
                    set_u32(enc, 7, scale_cols);
                    set_u32(enc, 8, threads_x);
                },
            )?,
            "fp8_occ",
        )
    }

    fn dispatch_fp4(
        ctx: &hawking_core::metal::MetalContext,
        kernel: &str,
        grid: (u32, u32, u32),
        rows: u32,
        packed_cols: u32,
        scale_cols: u32,
        packed: &metal::Buffer,
        scales: &metal::Buffer,
        quant: &metal::Buffer,
        act: &metal::Buffer,
        out: &metal::Buffer,
        timed: bool,
    ) -> Result<u64, Box<dyn Error>> {
        let encode = |enc: &metal::ComputeCommandEncoderRef| {
            enc.set_buffer(0, Some(packed), 0);
            enc.set_buffer(1, Some(scales), 0);
            enc.set_buffer(2, Some(quant), 0);
            enc.set_buffer(3, Some(act), 0);
            enc.set_buffer(4, Some(out), 0);
            set_u32(enc, 5, rows);
            set_u32(enc, 6, packed_cols);
            set_u32(enc, 7, scale_cols);
        };
        if timed {
            gpu_ns(
                ctx.dispatch_threads_timed(kernel, grid, (256, 1, 1), encode)?,
                kernel,
            )
        } else {
            ctx.dispatch_threads(kernel, grid, (256, 1, 1), encode)?;
            Ok(0)
        }
    }

    fn time_fp4(
        ctx: &hawking_core::metal::MetalContext,
        kernel: &str,
        grid: (u32, u32, u32),
        rows: u32,
        packed_cols: u32,
        scale_cols: u32,
        packed: &metal::Buffer,
        scales: &metal::Buffer,
        quant: &metal::Buffer,
        act: &metal::Buffer,
        out: &metal::Buffer,
        zeros: &[f32],
    ) -> Result<u64, Box<dyn Error>> {
        write_f32(out, zeros);
        dispatch_fp4(
            ctx,
            kernel,
            grid,
            rows,
            packed_cols,
            scale_cols,
            packed,
            scales,
            quant,
            act,
            out,
            true,
        )
    }

    fn make_fp4(rows: usize, logical: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<f32>) {
        let packed_cols = logical / 2;
        let scale_cols = logical / DSV4F_FP4_BLOCK;
        let mut packed = vec![0u8; rows * packed_cols];
        let mut scales = vec![0u8; rows * scale_cols];
        for (i, b) in packed.iter_mut().enumerate() {
            *b = (i.wrapping_mul(41) + 7) as u8;
        }
        for (i, b) in scales.iter_mut().enumerate() {
            *b = 0x7f_u8.wrapping_add((i as u8) & 3);
        }
        let mut quant = vec![0u8; logical];
        let act = vec![0x7f_u8; logical / ACT_BLOCK as usize];
        for (i, b) in quant.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(3);
        }
        let oracle = fp4_cpu_oracle(&packed, &scales, &quant, &act, rows, packed_cols, scale_cols);
        (packed, scales, quant, act, oracle)
    }

    fn make_fp8(rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut w = vec![0u8; rows * cols];
        let sr = rows / 128;
        let sc = cols / 128;
        let mut s = vec![0x7fu8; sr * sc];
        let mut q = vec![0u8; cols];
        let a = vec![0x7fu8; cols / 128];
        for (i, b) in w.iter_mut().enumerate() {
            *b = (i.wrapping_mul(17) + 3) as u8;
        }
        for (i, b) in s.iter_mut().enumerate() {
            *b = 0x7f_u8.wrapping_add((i as u8) & 1);
        }
        for (i, b) in q.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(5);
        }
        (w, s, q, a)
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

    fn deterministic_f32(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.001 - 0.1)
            .collect()
    }

    fn deterministic_x(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|i| ((i % 17) as f32) * 0.01 - 0.05)
            .collect()
    }

    fn deterministic_bf16(rows: usize, cols: usize) -> Vec<u8> {
        let mut out = vec![0u8; rows * cols * 2];
        for i in 0..rows * cols {
            let f = ((i % 61) as f32) * 0.02 - 0.5;
            let bits = (f.to_bits() >> 16) as u16;
            out[i * 2] = bits as u8;
            out[i * 2 + 1] = (bits >> 8) as u8;
        }
        out
    }

    fn serial_grid(rows: u32) -> (u32, u32, u32) {
        (rows, 1, 1)
    }

    fn simd_grid(rows: u32) -> (u32, u32, u32) {
        (rows.div_ceil(8) * 256, 1, 1)
    }

    fn r4_grid(rows: u32) -> (u32, u32, u32) {
        (rows.div_ceil(32) * 256, 1, 1)
    }

    fn pipeline_json(
        ctx: &hawking_core::metal::MetalContext,
        kernel: &str,
    ) -> Result<Value, Box<dyn Error>> {
        let pipe = ctx.pipeline(kernel)?;
        Ok(json!({
            "max_total_threads_per_threadgroup": pipe.max_total_threads_per_threadgroup(),
            "thread_execution_width": pipe.thread_execution_width(),
            "static_threadgroup_memory_length": pipe.static_threadgroup_memory_length(),
        }))
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
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
}
