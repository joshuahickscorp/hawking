//! G003: does wiring `gk_*_simd` still help after recon-fuse occupancy tiles?
//!
//! Isolated live mixed-1p5-v1 organs. Alternating pairs of the shipping
//! tile (tg256 / simd_bytes / simd3) vs `gk_*_simd`. GPU ns is
//! MTLCommandBuffer GPUEnd-GPUStart after wait.
//!
//!   cargo build --profile release-fast -p hawking-core --example q80_gk_simd_after_recon
//!   ./tools/gpu_lane_lock.sh q80-wire-gk-simd \
//!     $CARGO_TARGET_DIR/release-fast/examples/q80_gk_simd_after_recon \
//!     --out receipts/ascent-2026-08-16/Q80_GK_SIMD_AFTER_RECON.json

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
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    const SCHEMA: &str = "hawking.ascension.q80_gk_simd_after_recon.v1";
    const DEFAULT_ROOT: &str = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/mixed-1p5-v1";
    const HONEST_CEILING_GBPS: f64 = 411.51;
    const WARMUP: usize = 2;
    const PAIRS: usize = 3;
    const Q8_TOL: f32 = 2e-4;
    const BIN_TOL: f32 = 2e-5;

    struct Args {
        root: PathBuf,
        out: PathBuf,
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let mut root = PathBuf::from(DEFAULT_ROOT);
        let mut out = PathBuf::from("receipts/ascent-2026-08-16/Q80_GK_SIMD_AFTER_RECON.json");
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--root" => root = PathBuf::from(args.next().ok_or("missing --root")?),
                "--out" => out = PathBuf::from(args.next().ok_or("missing --out")?),
                other => return Err(format!("unsupported option {other}").into()),
            }
        }
        Ok(Args { root, out })
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
        (bytes as f64) / (ns as f64)
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

    fn shipping_factor_kernel(bits: u8, cols: u32) -> (&'static str, (u32, u32, u32), &'static str) {
        let rows_grid = |rows: u32| (rows.div_ceil(8).saturating_mul(256).max(256), 1, 1);
        if bits == 8 {
            if cols >= 2048 {
                (
                    "q80_uniform8_matvec_tg256",
                    (0, 1, 1),
                    "shipping Q8 tg256 (256 threads/row, byte extract)",
                )
            } else {
                (
                    "q80_uniform8_matvec_simd_bytes",
                    rows_grid(0),
                    "shipping Q8 simd_bytes (1 SG/row, byte extract)",
                )
            }
        } else if bits == 3 {
            (
                "q80_hgravs01_factor_matvec_simd3",
                rows_grid(0),
                "shipping 3-bit simd3 (1 SG/row, 8-unpack)",
            )
        } else {
            (
                "q80_hgravs01_factor_matvec_simd",
                rows_grid(0),
                "shipping hgravs simd (1 SG/row, wide extract)",
            )
        }
    }

    fn factor_grid(name: &str, rows: u32) -> (u32, u32, u32) {
        if name.ends_with("tg256") {
            (rows.saturating_mul(256), 1, 1)
        } else {
            (rows.div_ceil(8).saturating_mul(256).max(256), 1, 1)
        }
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

    fn pair_stats(base: &[u64], ours: &[u64]) -> Value {
        let n = base.len().min(ours.len());
        let mut ratios = Vec::with_capacity(n);
        for i in 0..n {
            if ours[i] > 0 {
                ratios.push(base[i] as f64 / ours[i] as f64);
            }
        }
        let base_med = median(base);
        let ours_med = median(ours);
        json!({
            "base_gpu_ns": base,
            "ours_gpu_ns": ours,
            "base_median_ns": base_med,
            "ours_median_ns": ours_med,
            "base_min_ns": *base.iter().min().unwrap_or(&0),
            "base_max_ns": *base.iter().max().unwrap_or(&0),
            "ours_min_ns": *ours.iter().min().unwrap_or(&0),
            "ours_max_ns": *ours.iter().max().unwrap_or(&0),
            "base_spread_ns": base.iter().max().copied().unwrap_or(0)
                .saturating_sub(base.iter().min().copied().unwrap_or(0)),
            "ours_spread_ns": ours.iter().max().copied().unwrap_or(0)
                .saturating_sub(ours.iter().min().copied().unwrap_or(0)),
            "speedup_x_median": if ours_med == 0 { 0.0 } else { base_med as f64 / ours_med as f64 },
            "pair_speedup_x": ratios,
        })
    }

    fn measure_binary(
        ctx: &MetalContext,
        name: &str,
        packed: &hawking_core::model::qwen_complete_binary::BinaryGroupPacked,
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
        let packed_bytes = (packed.signs.len() + packed.scales_f16.len() * 2) as u64;
        let ship_name = "q80_binary_group_matvec_tg256";
        let ship_grid = (rows.saturating_mul(256), 1, 1);
        let simd_name = hawking_core::decode_family::MATVEC_BINARY_SIMD;
        let simd_grid = (rows.div_ceil(8).saturating_mul(256).max(256), 1, 1);
        let tg = (256u32, 1, 1);
        let bind = |enc: &metal::ComputeCommandEncoderRef| {
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
        let _ = time_shipped(ctx, ship_name, ship_grid, tg, bind)?;
        let err_ship = max_abs_error(&read_f32(&out, packed.rows), &oracle);
        write_f32(&out, &zeros);
        let _ = time_shipped(ctx, simd_name, simd_grid, tg, bind)?;
        let err_simd = max_abs_error(&read_f32(&out, packed.rows), &oracle);

        for _ in 0..WARMUP {
            write_f32(&out, &zeros);
            let _ = time_shipped(ctx, ship_name, ship_grid, tg, bind)?;
            write_f32(&out, &zeros);
            let _ = time_shipped(ctx, simd_name, simd_grid, tg, bind)?;
        }
        let mut base = Vec::with_capacity(PAIRS);
        let mut ours = Vec::with_capacity(PAIRS);
        for _ in 0..PAIRS {
            write_f32(&out, &zeros);
            base.push(time_shipped(ctx, ship_name, ship_grid, tg, bind)?);
            write_f32(&out, &zeros);
            ours.push(time_shipped(ctx, simd_name, simd_grid, tg, bind)?);
        }

        let stats = pair_stats(&base, &ours);
        let ours_med = stats["ours_median_ns"].as_u64().unwrap_or(0);
        Ok(json!({
            "name": name,
            "kind": "binary_group",
            "shape": [packed.rows, packed.cols],
            "packed_weight_bytes": packed_bytes,
            "shipping_kernel": ship_name,
            "ours_kernel": simd_name,
            "shipping_note": "recon-fuse default: tg256, 8 signs/byte in-register",
            "ours_note": "gk_matvec_binary_simd 1-SG/row, 1 sign/lane",
            "correctness": {
                "shipping_max_abs": err_ship,
                "gk_simd_max_abs": err_simd,
                "tolerance": BIN_TOL,
                "passed": err_ship <= BIN_TOL && err_simd <= BIN_TOL,
            },
            "pairs": stats,
            "shipping_packed_gbps": gbps(packed_bytes, stats["base_median_ns"].as_u64().unwrap_or(0)),
            "ours_packed_gbps": gbps(packed_bytes, ours_med),
            "honest_ceiling_gbps": HONEST_CEILING_GBPS,
        }))
    }

    fn measure_uniform(
        ctx: &MetalContext,
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
        let packed_bytes = (packed.codes.len() + packed.scales_f16.len() * 2) as u64;
        let (ship_name, _, ship_note) = shipping_factor_kernel(packed.bits, cols);
        let ship_grid = factor_grid(ship_name, rows);
        let simd_name = hawking_core::decode_family::MATVEC_HGRAVS_SIMD;
        let simd_grid = factor_grid(simd_name, rows);
        let tg = (256u32, 1, 1);
        let bind = |enc: &metal::ComputeCommandEncoderRef| {
            encode_factor(enc, &codes, &scales, &x_buf, &out, packed);
        };
        let tol = if packed.bits == 8 { Q8_TOL } else { BIN_TOL };

        write_f32(&out, &zeros);
        let _ = time_shipped(ctx, ship_name, ship_grid, tg, bind)?;
        let err_ship = max_abs_error(&read_f32(&out, packed.rows), &oracle);
        write_f32(&out, &zeros);
        let _ = time_shipped(ctx, simd_name, simd_grid, tg, bind)?;
        let err_simd = max_abs_error(&read_f32(&out, packed.rows), &oracle);

        for _ in 0..WARMUP {
            write_f32(&out, &zeros);
            let _ = time_shipped(ctx, ship_name, ship_grid, tg, bind)?;
            write_f32(&out, &zeros);
            let _ = time_shipped(ctx, simd_name, simd_grid, tg, bind)?;
        }
        let mut base = Vec::with_capacity(PAIRS);
        let mut ours = Vec::with_capacity(PAIRS);
        for _ in 0..PAIRS {
            write_f32(&out, &zeros);
            base.push(time_shipped(ctx, ship_name, ship_grid, tg, bind)?);
            write_f32(&out, &zeros);
            ours.push(time_shipped(ctx, simd_name, simd_grid, tg, bind)?);
        }

        let stats = pair_stats(&base, &ours);
        let ours_med = stats["ours_median_ns"].as_u64().unwrap_or(0);
        Ok(json!({
            "name": name,
            "kind": kind,
            "bits": packed.bits,
            "bound": packed.bound,
            "group_size": packed.group_size,
            "shape": [packed.rows, packed.cols],
            "packed_weight_bytes": packed_bytes,
            "shipping_kernel": ship_name,
            "ours_kernel": simd_name,
            "shipping_note": ship_note,
            "ours_note": "gk_matvec_hgravs_simd 1-SG/row, wide extract (still bit-unpacks Q8)",
            "correctness": {
                "shipping_max_abs": err_ship,
                "gk_simd_max_abs": err_simd,
                "tolerance": tol,
                "passed": err_ship <= tol && err_simd <= tol,
            },
            "pairs": stats,
            "shipping_packed_gbps": gbps(packed_bytes, stats["base_median_ns"].as_u64().unwrap_or(0)),
            "ours_packed_gbps": gbps(packed_bytes, ours_med),
            "honest_ceiling_gbps": HONEST_CEILING_GBPS,
        }))
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let catalog = Qwen80MixedStreamingCatalog::open(&args.root)?;
        let ctx = MetalContext::new()?;
        let organs = [
            (
                "binary_gate_512x2048",
                "model.layers.0.mlp.experts.0.gate_proj.weight",
            ),
            (
                "hgravs_down_2048x512",
                "model.layers.0.mlp.experts.0.down_proj.weight",
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
                "q8_shared_gate_512x2048",
                "model.layers.0.mlp.shared_expert.gate_proj.weight",
            ),
            (
                "q8_shared_up_512x2048",
                "model.layers.0.mlp.shared_expert.up_proj.weight",
            ),
            (
                "q8_shared_down_2048x512",
                "model.layers.0.mlp.shared_expert.down_proj.weight",
            ),
            (
                "q8_out_2048x4096",
                "model.layers.0.linear_attn.out_proj.weight",
            ),
            (
                "q8_qkvz_12288x2048",
                "model.layers.0.linear_attn.in_proj_qkvz.weight",
            ),
        ];

        let mut organ_receipts = serde_json::Map::new();
        for (label, name) in organs {
            eprintln!("organ {label}");
            let packed = catalog.load_packed(name)?;
            match packed {
                MixedPackedTensor::Binary(bin) => {
                    organ_receipts.insert(label.to_string(), measure_binary(&ctx, name, &bin)?);
                }
                MixedPackedTensor::Hgravs { left, right } => {
                    organ_receipts.insert(
                        format!("{label}_R"),
                        measure_uniform(&ctx, &format!("{name}#R"), &right, "hgravs_R_3bit")?,
                    );
                    organ_receipts.insert(
                        format!("{label}_L"),
                        measure_uniform(&ctx, &format!("{name}#L"), &left, "hgravs_L_3bit")?,
                    );
                }
                MixedPackedTensor::Uniform8(u) => {
                    organ_receipts.insert(
                        label.to_string(),
                        measure_uniform(&ctx, name, &u, "uniform8")?,
                    );
                }
                MixedPackedTensor::Residual(_) => {
                    organ_receipts.insert(
                        label.to_string(),
                        json!({
                            "skipped": "residual CSR has no gk_*_simd equivalent; fused csr_tg256 stays"
                        }),
                    );
                }
            }
        }

        let mut any_win = false;
        let mut all_passed = true;
        let mut projected = serde_json::Map::new();
        let counts: [(&str, u64); 10] = [
            ("q8_shared_gate_512x2048", 48),
            ("q8_shared_up_512x2048", 48),
            ("q8_shared_down_2048x512", 48),
            ("q8_out_2048x4096", 48),
            ("q8_ba_64x2048", 36),
            ("q8_router_512x2048", 48),
            ("q8_qkvz_12288x2048", 36),
            ("hgravs_down_2048x512_R", 480),
            ("hgravs_down_2048x512_L", 480),
            ("binary_gate_512x2048", 480),
        ];
        let mut ship_sum = 0.0f64;
        let mut ours_sum = 0.0f64;
        for (key, count) in counts {
            if let Some(org) = organ_receipts.get(key) {
                if org.get("skipped").is_some() {
                    continue;
                }
                let base = org["pairs"]["base_median_ns"].as_u64().unwrap_or(0) as f64;
                let ours = org["pairs"]["ours_median_ns"].as_u64().unwrap_or(0) as f64;
                let speedup = org["pairs"]["speedup_x_median"].as_f64().unwrap_or(0.0);
                if speedup > 1.05 {
                    any_win = true;
                }
                if org["correctness"]["passed"].as_bool() == Some(false) {
                    all_passed = false;
                }
                let ship_ms = base * count as f64 / 1e6;
                let ours_ms = ours * count as f64 / 1e6;
                ship_sum += ship_ms;
                ours_sum += ours_ms;
                projected.insert(
                    key.to_string(),
                    json!({
                        "count_per_token": count,
                        "shipping_isolated_ns": base,
                        "gk_simd_isolated_ns": ours,
                        "shipping_token_ms": ship_ms,
                        "gk_simd_token_ms": ours_ms,
                        "speedup_x": speedup,
                    }),
                );
            }
        }

        let receipt = json!({
            "schema": SCHEMA,
            "lane": "q80-wire-gk-simd",
            "vehicle": "mixed-1p5-v1",
            "measurement_label": "DIRTY_ENGINEERING",
            "gpu_time_authority": "MTLCommandBuffer.GPUEndTime-GPUStartTime after wait",
            "device_name": ctx.device_name(),
            "root": args.root,
            "protocol": "warmup=2; 3 alternating BASE(shipping tile) then OURS(gk_*_simd) pairs per organ",
            "question": "Does gk_*_simd still help after in-register occupancy tiles deleted the serial bit-walk?",
            "correctness_all_passed": all_passed,
            "any_organ_speedup_gt_1_05": any_win,
            "verdict": if !all_passed {
                "INVALID_NUMERICS"
            } else if any_win {
                "PARTIAL_OR_WIN"
            } else {
                "NEGATIVE_SUBSUMED"
            },
            "organs": organ_receipts,
            "token_projection_isolated_ms": {
                "method": "isolated organ median GPU ns * count per token. Not a wall. Residual up omitted.",
                "organs": projected,
                "shipping_sum_ms": ship_sum,
                "gk_simd_sum_ms": ours_sum,
            },
        });

        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&args.out, serde_json::to_string_pretty(&receipt)?)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        Ok(())
    }
}
