//! Packed-matvec occupancy: reproduce the 230x gap, name the limiter, sweep fixes.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core --example matvec_occupancy
//! Measure (GPU mutex required):
//!   ./tools/gpu_lane_lock.sh matvec-occupancy-230x \
//!     workspace/ops/build/rust/release-fast/examples/matvec_occupancy \
//!     --out receipts/ascent-2026-08-16/matvec-occupancy-230x.json

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("matvec_occupancy requires macOS").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::dram_row_locality::{
        q4_weight_from_split, DSV4F_FP4_BLOCK, DSV4F_HIDDEN, DSV4F_INTER, Q80_BINARY_GROUP,
        Q80_GATE_COLS, Q80_GATE_ROWS, Q80_LAYERS, Q80_TOP_K,
    };
    use hawking_core::model::qwen_complete_binary::{
        binary_group_matvec_f32, deterministic_input, deterministic_matrix, pack_binary_group,
        pack_uniform_q4_group64, parse_uniform_q4_header, UNIFORM_Q4_GROUP_SIZE,
    };
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    const SCHEMA: &str = "hawking.ascension.matvec_occupancy.v1";
    const WARMUP: usize = 2;
    const SELECT_REPS: usize = 3;
    const PAIRS: usize = 3;
    const Q4_TOL: f32 = 2e-5;
    const BINARY_TOL: f32 = 2e-5;
    const FP4_TOL: f32 = 2e-4;
    const CONTROL_THREADS: u32 = 256 * 60;
    const CEILING_GBPS: f64 = 819.0;
    const EXPERT_BUDGET_NS: u64 = 8_000_000;

    struct Variant {
        name: &'static str,
        kernel: &'static str,
        rows_per_tg: u32,
        threads_per_row: u32,
        note: &'static str,
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut out = PathBuf::from("receipts/ascent-2026-08-16/matvec-occupancy-230x.json");
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--out" => {
                    out = PathBuf::from(args.next().ok_or("missing --out value")?);
                }
                other => return Err(format!("unsupported option {other}").into()),
            }
        }

        let ctx = hawking_core::metal::MetalContext::new()?;
        let device = ctx.device_name();

        let gate_w = deterministic_matrix(Q80_GATE_ROWS, Q80_GATE_COLS, 1);
        let x = deterministic_input(Q80_GATE_COLS);
        let (q4_payload, _) = pack_uniform_q4_group64(&gate_w, &[Q80_GATE_ROWS, Q80_GATE_COLS])?;
        let q4_header = parse_uniform_q4_header(&q4_payload)?;
        let mut q4_scales = Vec::with_capacity(q4_header.groups);
        for g in 0..q4_header.groups {
            q4_scales.push(u16::from_le_bytes([
                q4_payload[q4_header.scale_offset + g * 2],
                q4_payload[q4_header.scale_offset + g * 2 + 1],
            ]));
        }
        let q4_codes = q4_payload[q4_header.sign_offset..].to_vec();
        let q4_oracle: Vec<f32> = (0..Q80_GATE_ROWS)
            .map(|row| {
                (0..Q80_GATE_COLS)
                    .map(|col| {
                        q4_weight_from_split(&q4_scales, &q4_codes, Q80_GATE_COLS, row, col) * x[col]
                    })
                    .sum()
            })
            .collect();
        let q4_bytes = (q4_codes.len() + q4_scales.len() * 2) as u64;
        let q4_groups = (Q80_GATE_COLS / UNIFORM_Q4_GROUP_SIZE) as u32;

        let gate_bin = pack_binary_group(&gate_w, Q80_GATE_ROWS, Q80_GATE_COLS, Q80_BINARY_GROUP)?;
        let bin_oracle = binary_group_matvec_f32(&gate_bin, &x)?;
        let bin_bytes = (gate_bin.signs.len() + gate_bin.scales_f16.len() * 2) as u64;

        let fp4_rows = 256usize;
        let fp4_logical = 512usize;
        let (fp4_packed, fp4_scales, fp4_quant, fp4_act, fp4_oracle) =
            make_fp4(fp4_rows, fp4_logical);
        let fp4_bytes = (fp4_packed.len() + fp4_scales.len()) as u64;
        let fp4_packed_cols = fp4_logical / 2;
        let fp4_scale_cols = fp4_logical / DSV4F_FP4_BLOCK;

        let full_rows = DSV4F_INTER;
        let full_logical = DSV4F_HIDDEN;
        let (full_packed, full_scales, full_quant, full_act, full_oracle) =
            make_fp4(full_rows, full_logical);
        let full_bytes = (full_packed.len() + full_scales.len()) as u64;
        let full_packed_cols = full_logical / 2;
        let full_scale_cols = full_logical / DSV4F_FP4_BLOCK;

        let q4_code_buf = ctx.new_buffer_with_bytes_checked(&q4_codes)?;
        let q4_scale_buf = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&q4_scales))?;
        let x_buf = ctx.new_buffer_with_bytes_checked(as_u8_f32(&x))?;
        let out_buf = ctx.new_buffer_checked(Q80_GATE_ROWS * 4)?;
        let zeros = vec![0.0f32; Q80_GATE_ROWS];

        let bin_sign_buf = ctx.new_buffer_with_bytes_checked(&gate_bin.signs)?;
        let bin_scale_buf = ctx.new_buffer_with_bytes_checked(&as_u8_u16(&gate_bin.scales_f16))?;

        let fp4_p_buf = ctx.new_buffer_with_bytes_checked(&fp4_packed)?;
        let fp4_s_buf = ctx.new_buffer_with_bytes_checked(&fp4_scales)?;
        let fp4_q_buf = ctx.new_buffer_with_bytes_checked(&fp4_quant)?;
        let fp4_a_buf = ctx.new_buffer_with_bytes_checked(&fp4_act)?;
        let fp4_out = ctx.new_buffer_checked(fp4_rows * 4)?;
        let fp4_zeros = vec![0.0f32; fp4_rows];

        let full_p_buf = ctx.new_buffer_with_bytes_checked(&full_packed)?;
        let full_s_buf = ctx.new_buffer_with_bytes_checked(&full_scales)?;
        let full_q_buf = ctx.new_buffer_with_bytes_checked(&full_quant)?;
        let full_a_buf = ctx.new_buffer_with_bytes_checked(&full_act)?;
        let full_out = ctx.new_buffer_checked(full_rows * 4)?;
        let full_zeros = vec![0.0f32; full_rows];

        let q4_variants = [
            Variant {
                name: "serial",
                kernel: "qwen_uniform_q4_group64_matvec",
                rows_per_tg: 256,
                threads_per_row: 1,
                note: "shipped component path: one thread walks the whole K",
            },
            Variant {
                name: "simdgroup",
                kernel: "qwen_uniform_q4_group64_matvec_simdgroup",
                rows_per_tg: 8,
                threads_per_row: 32,
                note: "existing 1 row/simdgroup, scalar weight_at",
            },
            Variant {
                name: "simdgroup_r4",
                kernel: "qwen_uniform_q4_group64_matvec_simdgroup_rowblock4",
                rows_per_tg: 32,
                threads_per_row: 32,
                note: "existing 4 rows/simdgroup, scalar weight_at",
            },
            Variant {
                name: "simdgroup_r8",
                kernel: "qwen_uniform_q4_group64_matvec_simdgroup_rowblock8",
                rows_per_tg: 64,
                threads_per_row: 32,
                note: "existing 8 rows/simdgroup, scalar weight_at",
            },
            Variant {
                name: "x64",
                kernel: "qwen_uniform_q4_group64_matvec_simdgroup_x64",
                rows_per_tg: 4,
                threads_per_row: 64,
                note: "existing 64 threads/row",
            },
            Variant {
                name: "vecgroup",
                kernel: "qwen_uniform_q4_group64_matvec_vecgroup",
                rows_per_tg: 8,
                threads_per_row: 32,
                note: "1 row/simdgroup, one code-byte per lane per group",
            },
            Variant {
                name: "vecgroup_x",
                kernel: "qwen_uniform_q4_group64_matvec_vecgroup_x",
                rows_per_tg: 8,
                threads_per_row: 32,
                note: "vecgroup + X tile in threadgroup memory",
            },
            Variant {
                name: "vecgroup_r4",
                kernel: "qwen_uniform_q4_group64_matvec_vecgroup_r4",
                rows_per_tg: 32,
                threads_per_row: 32,
                note: "4 rows/simdgroup, vectorized group decode",
            },
            Variant {
                name: "vecgroup_r4_x",
                kernel: "qwen_uniform_q4_group64_matvec_vecgroup_r4_x",
                rows_per_tg: 32,
                threads_per_row: 32,
                note: "4 rows/simdgroup + X tile",
            },
            Variant {
                name: "vecgroup_x64",
                kernel: "qwen_uniform_q4_group64_matvec_vecgroup_x64",
                rows_per_tg: 4,
                threads_per_row: 64,
                note: "vectorized group decode at 64 threads/row",
            },
        ];

        let bin_variants = [
            Variant {
                name: "serial",
                kernel: "q80_binary_group_matvec",
                rows_per_tg: 256,
                threads_per_row: 1,
                note: "shipped mixed-decode gate: one thread per row",
            },
            Variant {
                name: "simd",
                kernel: "q80_binary_group_matvec_simd",
                rows_per_tg: 8,
                threads_per_row: 32,
                note: "existing 1 row/simdgroup",
            },
            Variant {
                name: "rowblock4",
                kernel: "q80_binary_group_matvec_rowblock4",
                rows_per_tg: 32,
                threads_per_row: 32,
                note: "existing 4 rows/simdgroup",
            },
            Variant {
                name: "simd_bytes",
                kernel: "q80_binary_group_matvec_simd_bytes",
                rows_per_tg: 8,
                threads_per_row: 32,
                note: "existing 8-weight byte unpack",
            },
            Variant {
                name: "tg256",
                kernel: "q80_binary_group_matvec_tg256",
                rows_per_tg: 1,
                threads_per_row: 256,
                note: "existing 256 threads/row",
            },
        ];

        let fp4_variants = [
            Variant {
                name: "serial",
                kernel: "dsv4f_fp4_matvec_split",
                rows_per_tg: 256,
                threads_per_row: 1,
                note: "isolated split kernel used by the 230x measurement",
            },
            Variant {
                name: "simd",
                kernel: "dsv4f_fp4_matvec_split_simd",
                rows_per_tg: 8,
                threads_per_row: 32,
                note: "1 row/simdgroup, 32-wide FP4 block",
            },
            Variant {
                name: "simd_r4",
                kernel: "dsv4f_fp4_matvec_split_simd_r4",
                rows_per_tg: 32,
                threads_per_row: 32,
                note: "4 rows/simdgroup, shared activations",
            },
        ];

        let mut correctness = serde_json::Map::new();
        for v in &q4_variants {
            let err = dispatch_q4(
                &ctx,
                v,
                Q80_GATE_ROWS as u32,
                Q80_GATE_COLS as u32,
                q4_groups,
                &q4_code_buf,
                &q4_scale_buf,
                &x_buf,
                &out_buf,
                &zeros,
                false,
            )
            .map(|_| max_abs(&q4_oracle, &read_f32(&out_buf, Q80_GATE_ROWS)))?;
            correctness.insert(
                format!("q4_{}", v.name),
                json!({ "max_abs_error": err, "passed": err <= Q4_TOL }),
            );
            if err > Q4_TOL {
                return Err(format!("q4 {} drift {err} > {Q4_TOL}", v.name).into());
            }
        }
        for v in &bin_variants {
            let err = dispatch_bin(
                &ctx,
                v,
                Q80_GATE_ROWS as u32,
                Q80_GATE_COLS as u32,
                Q80_BINARY_GROUP as u32,
                gate_bin.groups_per_row as u32,
                &bin_sign_buf,
                &bin_scale_buf,
                &x_buf,
                &out_buf,
                &zeros,
                false,
            )
            .map(|_| max_abs(&bin_oracle, &read_f32(&out_buf, Q80_GATE_ROWS)))?;
            correctness.insert(
                format!("bin_{}", v.name),
                json!({ "max_abs_error": err, "passed": err <= BINARY_TOL }),
            );
            if err > BINARY_TOL {
                return Err(format!("binary {} drift {err} > {BINARY_TOL}", v.name).into());
            }
        }
        for v in &fp4_variants {
            let err = dispatch_fp4(
                &ctx,
                v,
                fp4_rows as u32,
                fp4_packed_cols as u32,
                fp4_scale_cols as u32,
                &fp4_p_buf,
                &fp4_s_buf,
                &fp4_q_buf,
                &fp4_a_buf,
                &fp4_out,
                &fp4_zeros,
                false,
            )
            .map(|_| max_abs(&fp4_oracle, &read_f32(&fp4_out, fp4_rows)))?;
            correctness.insert(
                format!("fp4_reduced_{}", v.name),
                json!({ "max_abs_error": err, "passed": err <= FP4_TOL }),
            );
            if err > FP4_TOL {
                return Err(format!("fp4 reduced {} drift {err} > {FP4_TOL}", v.name).into());
            }
            let full_err = dispatch_fp4(
                &ctx,
                v,
                full_rows as u32,
                full_packed_cols as u32,
                full_scale_cols as u32,
                &full_p_buf,
                &full_s_buf,
                &full_q_buf,
                &full_a_buf,
                &full_out,
                &full_zeros,
                false,
            )
            .map(|_| max_abs(&full_oracle, &read_f32(&full_out, full_rows)))?;
            correctness.insert(
                format!("fp4_full_{}", v.name),
                json!({ "max_abs_error": full_err, "passed": full_err <= FP4_TOL }),
            );
            if full_err > FP4_TOL {
                return Err(format!("fp4 full {} drift {full_err} > {FP4_TOL}", v.name).into());
            }
        }

        for _ in 0..WARMUP {
            for v in &q4_variants {
                let _ = time_q4(
                    &ctx,
                    v,
                    q4_groups,
                    &q4_code_buf,
                    &q4_scale_buf,
                    &x_buf,
                    &out_buf,
                    &zeros,
                )?;
            }
            for v in &bin_variants {
                let _ = time_bin(
                    &ctx,
                    v,
                    gate_bin.groups_per_row as u32,
                    &bin_sign_buf,
                    &bin_scale_buf,
                    &x_buf,
                    &out_buf,
                    &zeros,
                )?;
            }
            for v in &fp4_variants {
                let _ = time_fp4(
                    &ctx,
                    v,
                    fp4_rows as u32,
                    fp4_packed_cols as u32,
                    fp4_scale_cols as u32,
                    &fp4_p_buf,
                    &fp4_s_buf,
                    &fp4_q_buf,
                    &fp4_a_buf,
                    &fp4_out,
                    &fp4_zeros,
                )?;
            }
        }

        let mut q4_select = serde_json::Map::new();
        let mut q4_best = "serial";
        let mut q4_best_med = u64::MAX;
        for v in &q4_variants {
            let mut ns = Vec::new();
            for _ in 0..SELECT_REPS {
                ns.push(time_q4(
                    &ctx,
                    v,
                    q4_groups,
                    &q4_code_buf,
                    &q4_scale_buf,
                    &x_buf,
                    &out_buf,
                    &zeros,
                )?);
            }
            let med = median(&ns);
            q4_select.insert(
                v.name.to_string(),
                variant_json(v, Q80_GATE_ROWS as u32, q4_bytes, &ns, &ctx)?,
            );
            if med < q4_best_med {
                q4_best_med = med;
                q4_best = v.name;
            }
        }
        let q4_winner = q4_variants
            .iter()
            .find(|v| v.name == q4_best)
            .ok_or("q4 winner missing")?;
        let q4_serial = q4_variants
            .iter()
            .find(|v| v.name == "serial")
            .ok_or("q4 serial missing")?;

        let mut bin_select = serde_json::Map::new();
        let mut bin_best = "serial";
        let mut bin_best_med = u64::MAX;
        for v in &bin_variants {
            let mut ns = Vec::new();
            for _ in 0..SELECT_REPS {
                ns.push(time_bin(
                    &ctx,
                    v,
                    gate_bin.groups_per_row as u32,
                    &bin_sign_buf,
                    &bin_scale_buf,
                    &x_buf,
                    &out_buf,
                    &zeros,
                )?);
            }
            let med = median(&ns);
            bin_select.insert(
                v.name.to_string(),
                variant_json(v, Q80_GATE_ROWS as u32, bin_bytes, &ns, &ctx)?,
            );
            if med < bin_best_med {
                bin_best_med = med;
                bin_best = v.name;
            }
        }
        let bin_winner = bin_variants
            .iter()
            .find(|v| v.name == bin_best)
            .ok_or("bin winner missing")?;
        let bin_serial = bin_variants
            .iter()
            .find(|v| v.name == "serial")
            .ok_or("bin serial missing")?;

        let mut fp4_select = serde_json::Map::new();
        let mut fp4_best = "serial";
        let mut fp4_best_med = u64::MAX;
        for v in &fp4_variants {
            let mut ns = Vec::new();
            for _ in 0..SELECT_REPS {
                ns.push(time_fp4(
                    &ctx,
                    v,
                    fp4_rows as u32,
                    fp4_packed_cols as u32,
                    fp4_scale_cols as u32,
                    &fp4_p_buf,
                    &fp4_s_buf,
                    &fp4_q_buf,
                    &fp4_a_buf,
                    &fp4_out,
                    &fp4_zeros,
                )?);
            }
            let med = median(&ns);
            fp4_select.insert(
                v.name.to_string(),
                variant_json(v, fp4_rows as u32, fp4_bytes, &ns, &ctx)?,
            );
            if med < fp4_best_med {
                fp4_best_med = med;
                fp4_best = v.name;
            }
        }
        let fp4_winner = fp4_variants
            .iter()
            .find(|v| v.name == fp4_best)
            .ok_or("fp4 winner missing")?;
        let fp4_serial = fp4_variants
            .iter()
            .find(|v| v.name == "serial")
            .ok_or("fp4 serial missing")?;

        let mut full_select = serde_json::Map::new();
        let mut full_best = "serial";
        let mut full_best_med = u64::MAX;
        for v in &fp4_variants {
            let mut ns = Vec::new();
            for _ in 0..SELECT_REPS {
                ns.push(time_fp4(
                    &ctx,
                    v,
                    full_rows as u32,
                    full_packed_cols as u32,
                    full_scale_cols as u32,
                    &full_p_buf,
                    &full_s_buf,
                    &full_q_buf,
                    &full_a_buf,
                    &full_out,
                    &full_zeros,
                )?);
            }
            let med = median(&ns);
            full_select.insert(
                v.name.to_string(),
                variant_json(v, full_rows as u32, full_bytes, &ns, &ctx)?,
            );
            if med < full_best_med {
                full_best_med = med;
                full_best = v.name;
            }
        }
        let full_winner = fp4_variants
            .iter()
            .find(|v| v.name == full_best)
            .ok_or("full winner missing")?;
        let full_serial_v = fp4_serial;

        let mut q4_a = Vec::new();
        let mut q4_b = Vec::new();
        let mut bin_a = Vec::new();
        let mut bin_b = Vec::new();
        let mut fp4_a = Vec::new();
        let mut fp4_b = Vec::new();
        let mut full_a = Vec::new();
        let mut full_b = Vec::new();
        for _ in 0..PAIRS {
            q4_a.push(time_q4(
                &ctx,
                q4_serial,
                q4_groups,
                &q4_code_buf,
                &q4_scale_buf,
                &x_buf,
                &out_buf,
                &zeros,
            )?);
            q4_b.push(time_q4(
                &ctx,
                q4_winner,
                q4_groups,
                &q4_code_buf,
                &q4_scale_buf,
                &x_buf,
                &out_buf,
                &zeros,
            )?);
            bin_a.push(time_bin(
                &ctx,
                bin_serial,
                gate_bin.groups_per_row as u32,
                &bin_sign_buf,
                &bin_scale_buf,
                &x_buf,
                &out_buf,
                &zeros,
            )?);
            bin_b.push(time_bin(
                &ctx,
                bin_winner,
                gate_bin.groups_per_row as u32,
                &bin_sign_buf,
                &bin_scale_buf,
                &x_buf,
                &out_buf,
                &zeros,
            )?);
            fp4_a.push(time_fp4(
                &ctx,
                fp4_serial,
                fp4_rows as u32,
                fp4_packed_cols as u32,
                fp4_scale_cols as u32,
                &fp4_p_buf,
                &fp4_s_buf,
                &fp4_q_buf,
                &fp4_a_buf,
                &fp4_out,
                &fp4_zeros,
            )?);
            fp4_b.push(time_fp4(
                &ctx,
                fp4_winner,
                fp4_rows as u32,
                fp4_packed_cols as u32,
                fp4_scale_cols as u32,
                &fp4_p_buf,
                &fp4_s_buf,
                &fp4_q_buf,
                &fp4_a_buf,
                &fp4_out,
                &fp4_zeros,
            )?);
            full_a.push(time_fp4(
                &ctx,
                full_serial_v,
                full_rows as u32,
                full_packed_cols as u32,
                full_scale_cols as u32,
                &full_p_buf,
                &full_s_buf,
                &full_q_buf,
                &full_a_buf,
                &full_out,
                &full_zeros,
            )?);
            full_b.push(time_fp4(
                &ctx,
                full_winner,
                full_rows as u32,
                full_packed_cols as u32,
                full_scale_cols as u32,
                &full_p_buf,
                &full_s_buf,
                &full_q_buf,
                &full_a_buf,
                &full_out,
                &full_zeros,
            )?);
        }

        let probe_bytes: usize = 64 * 1024 * 1024;
        let iters: u32 = 4096;
        let probe = vec![0x5au8; probe_bytes];
        let probe_buf = ctx.new_buffer_with_bytes_checked(&probe)?;
        let probe_out = ctx.new_buffer_checked(CONTROL_THREADS as usize * 4)?;
        let time_probe = |stride: u32| -> Result<u64, Box<dyn Error>> {
            let t = ctx.dispatch_threads_timed(
                "dram_row_locality_read_reduce",
                (CONTROL_THREADS, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&probe_buf), 0);
                    enc.set_buffer(1, Some(&probe_out), 0);
                    set_u32(enc, 2, probe_bytes as u32);
                    set_u32(enc, 3, stride);
                    set_u32(enc, 4, iters);
                },
            )?;
            gpu_ns(t, "probe")
        };
        let seq_stride = CONTROL_THREADS * 16;
        let conflict_stride = 8192u32 + 64;
        for _ in 0..WARMUP {
            let _ = time_probe(seq_stride)?;
            let _ = time_probe(conflict_stride)?;
        }
        let mut seq = Vec::new();
        let mut conflict = Vec::new();
        for _ in 0..PAIRS {
            seq.push(time_probe(seq_stride)?);
            conflict.push(time_probe(conflict_stride)?);
        }
        // Each of CONTROL_THREADS issues `iters` float4 (16-byte) loads.
        let probe_moved = u64::from(CONTROL_THREADS) * u64::from(iters) * 16;

        let q4_a_med = median(&q4_a);
        let q4_b_med = median(&q4_b);
        let routed_a = (Q80_LAYERS as u64) * (Q80_TOP_K as u64) * 3 * q4_a_med;
        let routed_b = (Q80_LAYERS as u64) * (Q80_TOP_K as u64) * 3 * q4_b_med;

        let diagnosis = json!({
            "q80_q4_gate_serial": diagnose("qwen_uniform_q4_group64_matvec", Q80_GATE_ROWS as u32, Q80_GATE_COLS as u32, 1, 256, q4_bytes, 2),
            "q80_mixed_binary_serial": diagnose("q80_binary_group_matvec", Q80_GATE_ROWS as u32, Q80_GATE_COLS as u32, 1, 256, bin_bytes, 1),
            "dsv4f_fp4_reduced_serial": diagnose("dsv4f_fp4_matvec_split", fp4_rows as u32, fp4_logical as u32, 1, 256, fp4_bytes, 4),
            "dsv4f_fp4_full_w1_serial": diagnose("dsv4f_fp4_matvec_split", full_rows as u32, full_logical as u32, 1, 256, full_bytes, 4),
            "limiter": "launch geometry / one-thread-per-row serialization. 512-row organs launch 2 threadgroups; the 256-row FP4 probe launches 1. The control launches 60. Register spill is not required to explain a 200-800x gap against a same-box 466-533 GB/s read-reduce.",
            "bytes_per_thread_note": "serial Q4: 1088 packed bytes + the whole 8 KiB X vector per thread. Existing simdgroup kernels still call weight_at per column, reloading the group scale 64 times.",
            "arith_to_load": {
                "q4_mac_per_packed_byte": (Q80_GATE_ROWS * Q80_GATE_COLS) as f64 / q4_bytes as f64,
                "binary_mac_per_packed_byte": (Q80_GATE_ROWS * Q80_GATE_COLS) as f64 / bin_bytes as f64,
            }
        });

        let receipt = json!({
            "schema": SCHEMA,
            "lane": "matvec-occupancy-230x",
            "measurement_label": "DIRTY_ENGINEERING",
            "device_name": device,
            "gpu_time_authority": "MTLCommandBuffer.GPUEndTime-GPUStartTime after wait",
            "fallbacks": 0,
            "dense_weight_materialized": false,
            "oracle": "artifact packed decode, not BF16 parent",
            "correctness": {
                "tolerance_q4": Q4_TOL,
                "tolerance_binary": BINARY_TOL,
                "tolerance_fp4": FP4_TOL,
                "kernels": correctness,
            },
            "reproduce": {
                "control_sequential_gbps": gbps(probe_moved, median(&seq)),
                "control_conflict_gbps": gbps(probe_moved, median(&conflict)),
                "control_sequential_gpu_ns": seq,
                "control_conflict_gpu_ns": conflict,
                "q80_q4_gate_serial_gbps": gbps(q4_bytes, q4_a_med),
                "q80_mixed_binary_serial_gbps": gbps(bin_bytes, median(&bin_a)),
                "dsv4f_fp4_reduced_serial_gbps": gbps(fp4_bytes, median(&fp4_a)),
                "dsv4f_fp4_full_w1_serial_gbps": gbps(full_bytes, median(&full_a)),
                "gap_q4_vs_sequential_control": gbps(probe_moved, median(&seq)) / gbps(q4_bytes, q4_a_med).max(1e-12),
                "note": "Same-box control vs packed kernels. DIRTY_ENGINEERING: other lanes may be running."
            },
            "diagnosis": diagnosis,
            "selection": {
                "protocol": "warmup=2; 3 reps per variant (not paired); then 3 alternating A/B pairs of serial vs winner",
                "q4": q4_select,
                "binary": bin_select,
                "fp4_reduced": fp4_select,
                "fp4_full_w1": full_select,
            },
            "paired": {
                "q80_q4_gate": paired_arm("serial", q4_winner.name, q4_bytes, &q4_a, &q4_b),
                "q80_mixed_binary_gate": paired_arm("serial", bin_winner.name, bin_bytes, &bin_a, &bin_b),
                "dsv4f_fp4_reduced": paired_arm("serial", fp4_winner.name, fp4_bytes, &fp4_a, &fp4_b),
                "dsv4f_fp4_full_w1": paired_arm("serial", full_winner.name, full_bytes, &full_a, &full_b),
            },
            "projection": {
                "formula": "48 * 10 * 3 * organ_median",
                "claim_boundary": "isolated Q4 organ GPU time * 1440. Not a full token wall. Attention/shared/router/lm_head use the same component kernel.",
                "serial_ns_per_token": routed_a,
                "winner_ns_per_token": routed_b,
                "winner_kernel": q4_winner.kernel,
                "distance_from_8ms_expert_budget": (routed_b as f64) / (EXPERT_BUDGET_NS as f64),
                "at_100_gbps_ns": ((q4_bytes as f64) / 100.0 * 1440.0) as u64,
            },
            "pipeline": {
                "q4_serial": pipeline_json(&ctx, q4_serial.kernel)?,
                "q4_winner": pipeline_json(&ctx, q4_winner.kernel)?,
                "fp4_serial": pipeline_json(&ctx, fp4_serial.kernel)?,
                "fp4_winner": pipeline_json(&ctx, fp4_winner.kernel)?,
            }
        });

        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("wrote {}", out.display());
        println!(
            "control seq {:.2} GB/s  q4 serial {:.2} -> {} {:.2} GB/s  routed {} -> {} ns",
            gbps(probe_moved, median(&seq)),
            gbps(q4_bytes, q4_a_med),
            q4_winner.name,
            gbps(q4_bytes, q4_b_med),
            routed_a,
            routed_b
        );
        Ok(())
    }

    fn diagnose(
        kernel: &str,
        rows: u32,
        cols: u32,
        threads_per_row: u32,
        tg: u32,
        packed_bytes: u64,
        decode_ops: u32,
    ) -> Value {
        let threads = rows * threads_per_row;
        let threadgroups = threads.div_ceil(tg);
        json!({
            "kernel": kernel,
            "rows": rows,
            "cols": cols,
            "threads": threads,
            "threadgroups": threadgroups,
            "threads_per_threadgroup": tg,
            "simdgroups_per_threadgroup": tg / 32,
            "occupancy_proxy_vs_control": threads as f64 / CONTROL_THREADS as f64,
            "bytes_read_per_thread_packed": packed_bytes as f64 / threads as f64,
            "bytes_read_per_thread_x_if_reloaded": (cols as u64 * 4) as f64,
            "arithmetic_to_load_mac_per_packed_byte": (rows as u64 * cols as u64) as f64 / packed_bytes as f64,
            "decode_ops_per_weight_approx": decode_ops,
            "limiter": if threadgroups <= 8 {
                "launch geometry / insufficient parallel work"
            } else {
                "enough threadgroups; remaining cost is decode / X reuse / vector width"
            }
        })
    }

    fn variant_json(
        v: &Variant,
        rows: u32,
        bytes: u64,
        ns: &[u64],
        ctx: &hawking_core::metal::MetalContext,
    ) -> Result<Value, Box<dyn Error>> {
        let med = median(ns);
        let (grid, tg) = grid_for(v, rows);
        let threads = grid.0;
        Ok(json!({
            "kernel": v.kernel,
            "note": v.note,
            "threads": threads,
            "threadgroups": threads.div_ceil(tg.0),
            "threads_per_threadgroup": tg.0,
            "simdgroups_per_threadgroup": tg.0 / 32,
            "threads_per_row": v.threads_per_row,
            "occupancy_proxy_vs_control": threads as f64 / CONTROL_THREADS as f64,
            "bytes_read_per_thread_packed": bytes as f64 / threads.max(1) as f64,
            "gpu_ns": ns,
            "median_gpu_ns": med,
            "gbps": gbps(bytes, med),
            "pipeline": pipeline_json(ctx, v.kernel)?,
        }))
    }

    fn paired_arm(a_name: &str, b_name: &str, bytes: u64, a: &[u64], b: &[u64]) -> Value {
        json!({
            "A": a_name,
            "B": b_name,
            "A_gpu_ns": a,
            "B_gpu_ns": b,
            "A_median_gpu_ns": median(a),
            "B_median_gpu_ns": median(b),
            "A_gbps": gbps(bytes, median(a)),
            "B_gbps": gbps(bytes, median(b)),
            "A_frac_of_ceiling": gbps(bytes, median(a)) / CEILING_GBPS,
            "B_frac_of_ceiling": gbps(bytes, median(b)) / CEILING_GBPS,
            "bytes": bytes,
            "measurement_label": "DIRTY_ENGINEERING",
            "gpu_time_authority": "MTLCommandBuffer.GPUEndTime-GPUStartTime after wait",
        })
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

    fn grid_for(v: &Variant, rows: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
        if v.threads_per_row == 1 && v.rows_per_tg == 256 {
            return ((rows, 1, 1), (256, 1, 1));
        }
        if v.threads_per_row == 256 {
            return ((rows * 256, 1, 1), (256, 1, 1));
        }
        let groups = rows.div_ceil(v.rows_per_tg);
        ((groups * 256, 1, 1), (256, 1, 1))
    }

    fn dispatch_q4(
        ctx: &hawking_core::metal::MetalContext,
        v: &Variant,
        rows: u32,
        cols: u32,
        groups_per_row: u32,
        codes: &metal::Buffer,
        scales: &metal::Buffer,
        x: &metal::Buffer,
        out: &metal::Buffer,
        zeros: &[f32],
        timed: bool,
    ) -> Result<u64, Box<dyn Error>> {
        write_f32(out, zeros);
        let (grid, tg) = grid_for(v, rows);
        let encode = |enc: &metal::ComputeCommandEncoderRef| {
            enc.set_buffer(0, Some(codes), 0);
            enc.set_buffer(1, Some(scales), 0);
            enc.set_buffer(2, Some(x), 0);
            enc.set_buffer(3, Some(out), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            set_u32(enc, 6, groups_per_row);
        };
        if timed {
            gpu_ns(ctx.dispatch_threads_timed(v.kernel, grid, tg, encode)?, v.kernel)
        } else {
            ctx.dispatch_threads(v.kernel, grid, tg, encode)?;
            Ok(0)
        }
    }

    fn time_q4(
        ctx: &hawking_core::metal::MetalContext,
        v: &Variant,
        groups_per_row: u32,
        codes: &metal::Buffer,
        scales: &metal::Buffer,
        x: &metal::Buffer,
        out: &metal::Buffer,
        zeros: &[f32],
    ) -> Result<u64, Box<dyn Error>> {
        dispatch_q4(
            ctx,
            v,
            Q80_GATE_ROWS as u32,
            Q80_GATE_COLS as u32,
            groups_per_row,
            codes,
            scales,
            x,
            out,
            zeros,
            true,
        )
    }

    fn dispatch_bin(
        ctx: &hawking_core::metal::MetalContext,
        v: &Variant,
        rows: u32,
        cols: u32,
        group_size: u32,
        groups_per_row: u32,
        signs: &metal::Buffer,
        scales: &metal::Buffer,
        x: &metal::Buffer,
        out: &metal::Buffer,
        zeros: &[f32],
        timed: bool,
    ) -> Result<u64, Box<dyn Error>> {
        write_f32(out, zeros);
        let (grid, tg) = grid_for(v, rows);
        let encode = |enc: &metal::ComputeCommandEncoderRef| {
            enc.set_buffer(0, Some(signs), 0);
            enc.set_buffer(1, Some(scales), 0);
            enc.set_buffer(2, Some(x), 0);
            enc.set_buffer(3, Some(out), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            set_u32(enc, 6, group_size);
            set_u32(enc, 7, groups_per_row);
        };
        if timed {
            gpu_ns(ctx.dispatch_threads_timed(v.kernel, grid, tg, encode)?, v.kernel)
        } else {
            ctx.dispatch_threads(v.kernel, grid, tg, encode)?;
            Ok(0)
        }
    }

    fn time_bin(
        ctx: &hawking_core::metal::MetalContext,
        v: &Variant,
        groups_per_row: u32,
        signs: &metal::Buffer,
        scales: &metal::Buffer,
        x: &metal::Buffer,
        out: &metal::Buffer,
        zeros: &[f32],
    ) -> Result<u64, Box<dyn Error>> {
        dispatch_bin(
            ctx,
            v,
            Q80_GATE_ROWS as u32,
            Q80_GATE_COLS as u32,
            Q80_BINARY_GROUP as u32,
            groups_per_row,
            signs,
            scales,
            x,
            out,
            zeros,
            true,
        )
    }

    fn dispatch_fp4(
        ctx: &hawking_core::metal::MetalContext,
        v: &Variant,
        rows: u32,
        packed_cols: u32,
        scale_cols: u32,
        packed: &metal::Buffer,
        scales: &metal::Buffer,
        quant: &metal::Buffer,
        act: &metal::Buffer,
        out: &metal::Buffer,
        zeros: &[f32],
        timed: bool,
    ) -> Result<u64, Box<dyn Error>> {
        write_f32(out, zeros);
        let (grid, tg) = grid_for(v, rows);
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
            gpu_ns(ctx.dispatch_threads_timed(v.kernel, grid, tg, encode)?, v.kernel)
        } else {
            ctx.dispatch_threads(v.kernel, grid, tg, encode)?;
            Ok(0)
        }
    }

    fn time_fp4(
        ctx: &hawking_core::metal::MetalContext,
        v: &Variant,
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
        dispatch_fp4(
            ctx, v, rows, packed_cols, scale_cols, packed, scales, quant, act, out, zeros, true,
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
        let act = vec![0x7f_u8; logical / 128];
        for (i, b) in quant.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(3);
        }
        let oracle = fp4_cpu_oracle(&packed, &scales, &quant, &act, rows, packed_cols, scale_cols);
        (packed, scales, quant, act, oracle)
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

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    fn as_u8_u16(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
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
