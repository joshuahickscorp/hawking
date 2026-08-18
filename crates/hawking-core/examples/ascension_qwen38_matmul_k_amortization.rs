//! Does emitting K positions per weight sweep actually amortize the sweep?
//!
//! NX_TPS_FRONTIER.json establishes that one full weight sweep per token puts
//! 100 TPS out of reach for every representation that has shown capability --
//! even capability-dead uniform q2 only reaches 97 TPS with compute charged at
//! zero. The single axis that moves that wall is K > 1: decode each weight
//! once, multiply it into K accumulators.
//!
//! That is a claim about the hardware, not about arithmetic, and it fails if
//! the K-column kernel stops being bandwidth-bound. This measures it.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core \
//!     --example ascension_qwen38_matmul_k_amortization
//! Measure (GPU mutex required):
//!   ./tools/gpu_lane_lock.sh matmul-k-amortization \
//!     workspace/ops/build/rust/release-fast/examples/ascension_qwen38_matmul_k_amortization \
//!     --out receipts/ascent-2026-08-16/NX_MATMUL_K_AMORTIZATION.json

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
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    const SCHEMA: &str = "hawking.nos.matmul_k_amortization.v1";
    // Qwen3.8 hidden. Every MLP and attention GEMV in the genome has this as
    // its contraction length, so the per-row byte cost matches production.
    const COLS: usize = 5120;
    const GROUP: usize = 64;
    const CODE_BYTES_PER_GROUP: usize = 32;
    const REPS: usize = 5;
    const KS: [usize; 4] = [1, 2, 4, 8];
    const RK: [(usize, usize); 8] =
        [(2, 2), (2, 4), (4, 2), (4, 4), (4, 8), (8, 4), (8, 8), (16, 4)];

    fn row_bytes() -> usize {
        (COLS / GROUP) * CODE_BYTES_PER_GROUP
    }

    /// Deterministic code bytes. Content is irrelevant to bandwidth and the
    /// correctness checks below are content-agnostic, but a fixed stream keeps
    /// reruns comparable.
    fn codes_for(rows: usize) -> Vec<u8> {
        let n = rows * row_bytes();
        let mut v = vec![0u8; n];
        let mut s: u32 = 0x9e3779b9;
        for b in v.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 24) as u8;
        }
        v
    }

    fn scales_for(rows: usize) -> Vec<u8> {
        let n = rows * (COLS / GROUP);
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let f = 0.001_f32 + (i % 17) as f32 * 0.0001_f32;
            out.extend_from_slice(&half_bits(f).to_le_bytes());
        }
        out
    }

    fn half_bits(f: f32) -> u16 {
        // Only used for small positive scales; a full f32->f16 is overkill.
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = ((bits >> 13) & 0x3ff) as u16;
        if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7bff
        } else {
            sign | ((exp as u16) << 10) | mant
        }
    }

    fn median(mut v: Vec<u64>) -> u64 {
        v.sort_unstable();
        v[v.len() / 2]
    }

    struct Args {
        out: Option<PathBuf>,
        payloads_mib: Vec<usize>,
    }

    fn parse_args() -> Args {
        let mut out = None;
        let mut payloads_mib = vec![1024usize, 4096usize];
        let mut it = env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--out" => out = it.next().map(PathBuf::from),
                "--payload-mib" => {
                    payloads_mib = it
                        .next()
                        .unwrap_or_default()
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
                other => panic!("unknown flag {other}"),
            }
        }
        Args { out, payloads_mib }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args();
        let ctx = MetalContext::new()?;
        let mut results: Vec<Value> = Vec::new();

        for &mib in &args.payloads_mib {
            let target_bytes = mib * 1024 * 1024;
            // Rows are a multiple of 2 because the kernel puts 2 rows in a TG.
            let rows = ((target_bytes / row_bytes()) / 2) * 2;
            let payload_bytes = rows * row_bytes();
            eprintln!(
                "payload {mib} MiB -> rows={rows} codes={:.3} GB",
                payload_bytes as f64 / 1e9
            );

            let codes = ctx.new_buffer_with_bytes_checked(&codes_for(rows))?;
            let scales = ctx.new_buffer_with_bytes_checked(&scales_for(rows))?;
            let groups_per_row = (COLS / GROUP) as u32;

            // Position-interleaved activations: input[col * K + k].
            let kmax = *KS.iter().max().unwrap();
            let mut xin = vec![0f32; COLS * kmax];
            for col in 0..COLS {
                for k in 0..kmax {
                    // Distinct per k so a kernel that ignores k cannot pass.
                    xin[col * kmax + k] = ((col % 31) as f32 - 15.0) * (1.0 + k as f32);
                }
            }
            // Uniform variant: every position identical, so every output
            // column must equal the K=1 answer exactly.
            let xuni: Vec<f32> = (0..COLS * kmax)
                .map(|i| ((i / kmax) % 31) as f32 - 15.0)
                .collect();

            let mut by_k: Vec<Value> = Vec::new();
            let mut k1_uniform_out: Vec<f32> = Vec::new();
            let mut k1_per_position: Vec<Vec<f32>> = Vec::new();
            let mut k1_median_ns = 0u64;

            for &k in KS.iter() {
                let kernel =
                    format!("qwen_uniform_q4_group64_matmul_k{k}_geo_tpr64_tg128");
                let out_len = rows * k * 4;
                let output = ctx.new_buffer_checked(out_len)?;
                let grid = (((rows as u32 + 1) / 2) * 128, 1, 1);
                let tg = (128u32, 1, 1);

                // --- distinct-per-position run, timed ---
                let packed_x: Vec<f32> = (0..COLS * k)
                    .map(|i| {
                        let col = i / k;
                        let kk = i % k;
                        ((col % 31) as f32 - 15.0) * (1.0 + kk as f32)
                    })
                    .collect();
                let xbuf = ctx.new_buffer_with_bytes_checked(bytemuck_cast(&packed_x))?;

                let mut gpu_ns: Vec<u64> = Vec::new();
                for _ in 0..REPS {
                    let t = ctx.dispatch_threads_timed(&kernel, grid, tg, |enc| {
                        enc.set_buffer(0, Some(&codes), 0);
                        enc.set_buffer(1, Some(&scales), 0);
                        enc.set_buffer(2, Some(&xbuf), 0);
                        enc.set_buffer(3, Some(&output), 0);
                        let r = rows as u32;
                        let c = COLS as u32;
                        enc.set_bytes(4, 4, &r as *const u32 as *const _);
                        enc.set_bytes(5, 4, &c as *const u32 as *const _);
                        enc.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
                    })?;
                    let (s, e) = (t.gpu_start_ns, t.gpu_end_ns);
                    match (s, e) {
                        (Some(s), Some(e)) if e > s => gpu_ns.push(e - s),
                        _ => return Err(format!("{kernel}: driver gave no GPU timestamp").into()),
                    }
                }
                let med = median(gpu_ns.clone());
                let out_vals = read_f32(&output, rows * k);
                if k == 1 {
                    k1_median_ns = med;
                    k1_per_position.push(out_vals.clone());
                }
                // Column k of the K-run must equal a dedicated K=1 run whose
                // input is that position's activations.
                let mut per_position_max_err = 0f32;
                if k > 1 {
                    for kk in 0..k {
                        let xk: Vec<f32> = (0..COLS)
                            .map(|col| ((col % 31) as f32 - 15.0) * (1.0 + kk as f32))
                            .collect();
                        let xb = ctx.new_buffer_with_bytes_checked(bytemuck_cast(&xk))?;
                        let ob = ctx.new_buffer_checked(rows * 4)?;
                        ctx.dispatch_threads_timed(
                            "qwen_uniform_q4_group64_matmul_k1_geo_tpr64_tg128",
                            grid,
                            tg,
                            |enc| {
                                enc.set_buffer(0, Some(&codes), 0);
                                enc.set_buffer(1, Some(&scales), 0);
                                enc.set_buffer(2, Some(&xb), 0);
                                enc.set_buffer(3, Some(&ob), 0);
                                let r = rows as u32;
                                let c = COLS as u32;
                                enc.set_bytes(4, 4, &r as *const u32 as *const _);
                                enc.set_bytes(5, 4, &c as *const u32 as *const _);
                                enc.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
                            },
                        )?;
                        let refv = read_f32(&ob, rows);
                        for row in 0..rows {
                            let d = (out_vals[row * k + kk] - refv[row]).abs();
                            if d > per_position_max_err {
                                per_position_max_err = d;
                            }
                        }
                    }
                }

                // --- uniform-position control: all K columns must agree ---
                let xu: Vec<f32> = (0..COLS * k).map(|i| ((i / k) % 31) as f32 - 15.0).collect();
                let xub = ctx.new_buffer_with_bytes_checked(bytemuck_cast(&xu))?;
                let oub = ctx.new_buffer_checked(rows * k * 4)?;
                ctx.dispatch_threads_timed(&kernel, grid, tg, |enc| {
                    enc.set_buffer(0, Some(&codes), 0);
                    enc.set_buffer(1, Some(&scales), 0);
                    enc.set_buffer(2, Some(&xub), 0);
                    enc.set_buffer(3, Some(&oub), 0);
                    let r = rows as u32;
                    let c = COLS as u32;
                    enc.set_bytes(4, 4, &r as *const u32 as *const _);
                    enc.set_bytes(5, 4, &c as *const u32 as *const _);
                    enc.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
                })?;
                let uni = read_f32(&oub, rows * k);
                if k == 1 {
                    k1_uniform_out = uni.clone();
                }
                let mut uniform_max_err = 0f32;
                for row in 0..rows {
                    for kk in 0..k {
                        let d = (uni[row * k + kk] - k1_uniform_out[row]).abs();
                        if d > uniform_max_err {
                            uniform_max_err = d;
                        }
                    }
                }

                let gb_s = payload_bytes as f64 / med as f64;
                let k1_serial_ns = k1_median_ns * k as u64;
                by_k.push(json!({
                    "k": k,
                    "kernel": kernel,
                    "median_gpu_ns": med,
                    "gpu_ns_reps": gpu_ns,
                    "code_bytes_moved": payload_bytes,
                    "achieved_gb_s": gb_s,
                    "ns_per_position": med as f64 / k as f64,
                    "k1_serial_ns_for_same_k_positions": k1_serial_ns,
                    "amortization_x": k1_serial_ns as f64 / med as f64,
                    "uniform_position_max_abs_err": uniform_max_err,
                    "per_position_max_abs_err": per_position_max_err,
                    "uniform_control_holds": uniform_max_err == 0.0,
                    "per_position_control_holds": k == 1 || per_position_max_err == 0.0,
                }));
                eprintln!(
                    "  K={k}: {:.3} ms, {:.1} GB/s, {:.3} ms/position, amortization {:.2}x, \
                     uniform_err={uniform_max_err:.3e} per_pos_err={per_position_max_err:.3e}",
                    med as f64 / 1e6,
                    gb_s,
                    med as f64 / k as f64 / 1e6,
                    k1_median_ns as f64 * k as f64 / med as f64,
                );
                let _ = &k1_per_position;
                let _ = &xin;
                let _ = &xuni;
            }
            // R x K tiling: R rows share one activation load, so activation
            // bytes per code byte fall from 8K to 8K/R.
            let mut by_rk: Vec<Value> = Vec::new();
            for &(r, k) in RK.iter() {
                let kernel =
                    format!("qwen_uniform_q4_group64_matmul_r{r}k{k}_geo_tpr64_tg128");
                let output = ctx.new_buffer_checked(rows * k * 4)?;
                let tgs = (rows + 2 * r - 1) / (2 * r);
                let grid = ((tgs * 128) as u32, 1, 1);
                let tg = (128u32, 1, 1);
                let packed_x: Vec<f32> = (0..COLS * k)
                    .map(|i| {
                        let col = i / k;
                        let kk = i % k;
                        ((col % 31) as f32 - 15.0) * (1.0 + kk as f32)
                    })
                    .collect();
                let xbuf = ctx.new_buffer_with_bytes_checked(bytemuck_cast(&packed_x))?;
                let mut gpu_ns: Vec<u64> = Vec::new();
                for _ in 0..REPS {
                    let t = ctx.dispatch_threads_timed(&kernel, grid, tg, |enc| {
                        enc.set_buffer(0, Some(&codes), 0);
                        enc.set_buffer(1, Some(&scales), 0);
                        enc.set_buffer(2, Some(&xbuf), 0);
                        enc.set_buffer(3, Some(&output), 0);
                        let rr = rows as u32;
                        let c = COLS as u32;
                        enc.set_bytes(4, 4, &rr as *const u32 as *const _);
                        enc.set_bytes(5, 4, &c as *const u32 as *const _);
                        enc.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
                    })?;
                    match (t.gpu_start_ns, t.gpu_end_ns) {
                        (Some(s), Some(e)) if e > s => gpu_ns.push(e - s),
                        _ => return Err(format!("{kernel}: no GPU timestamp").into()),
                    }
                }
                let med = median(gpu_ns.clone());
                let vals = read_f32(&output, rows * k);
                // Column k must equal a dedicated K=1 run on that position.
                let mut per_position_max_err = 0f32;
                for kk in 0..k {
                    let xk: Vec<f32> = (0..COLS)
                        .map(|col| ((col % 31) as f32 - 15.0) * (1.0 + kk as f32))
                        .collect();
                    let xb = ctx.new_buffer_with_bytes_checked(bytemuck_cast(&xk))?;
                    let ob = ctx.new_buffer_checked(rows * 4)?;
                    ctx.dispatch_threads_timed(
                        "qwen_uniform_q4_group64_matmul_k1_geo_tpr64_tg128",
                        (((rows as u32 + 1) / 2) * 128, 1, 1),
                        tg,
                        |enc| {
                            enc.set_buffer(0, Some(&codes), 0);
                            enc.set_buffer(1, Some(&scales), 0);
                            enc.set_buffer(2, Some(&xb), 0);
                            enc.set_buffer(3, Some(&ob), 0);
                            let rr = rows as u32;
                            let c = COLS as u32;
                            enc.set_bytes(4, 4, &rr as *const u32 as *const _);
                            enc.set_bytes(5, 4, &c as *const u32 as *const _);
                            enc.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
                        },
                    )?;
                    let refv = read_f32(&ob, rows);
                    for row in 0..rows {
                        let d = (vals[row * k + kk] - refv[row]).abs();
                        if d > per_position_max_err {
                            per_position_max_err = d;
                        }
                    }
                }
                let act_bytes = rows * COLS * k * 4;
                by_rk.push(json!({
                    "r": r,
                    "k": k,
                    "kernel": kernel,
                    "median_gpu_ns": med,
                    "gpu_ns_reps": gpu_ns,
                    "achieved_code_gb_s": payload_bytes as f64 / med as f64,
                    "activation_bytes": act_bytes,
                    "activation_bytes_per_code_byte": act_bytes as f64 / payload_bytes as f64
                        / r as f64,
                    "ns_per_position": med as f64 / k as f64,
                    "amortization_x": (k1_median_ns * k as u64) as f64 / med as f64,
                    "per_position_max_abs_err": per_position_max_err,
                    "per_position_control_holds": per_position_max_err == 0.0,
                }));
                eprintln!(
                    "  R={r} K={k}: {:.3} ms, code {:.1} GB/s, {:.3} ms/position, \
                     amortization {:.2}x, err={per_position_max_err:.3e}",
                    med as f64 / 1e6,
                    payload_bytes as f64 / med as f64,
                    med as f64 / k as f64 / 1e6,
                    (k1_median_ns * k as u64) as f64 / med as f64,
                );
            }

            results.push(json!({
                "payload_mib": mib,
                "rows": rows,
                "cols": COLS,
                "code_bytes": payload_bytes,
                "by_k": by_k,
                "by_rk": by_rk,
            }));
        }

        let doc = json!({
            "schema": SCHEMA,
            "question": "Does a K-column geo_tpr64 kernel move the same bytes in the same time?",
            "why": "NX_TPS_FRONTIER.json: at K=1 no capability-preserving representation reaches \
                    100 TPS on this machine. K>1 is the only axis that moves the weight-movement wall.",
            "method": "Same thread map, launch geometry and per-weight arithmetic as \
                       qwen_uniform_q4_group64_matvec_geo_tpr64_tg128. Each decoded weight is \
                       multiplied into K accumulators instead of one. Timing is \
                       GPUEndTime-GPUStartTime on completed command buffers, median of 5.",
            "controls": {
                "uniform_position": "all K activation columns identical; every output column must \
                                     equal the K=1 answer exactly (catches a kernel that writes \
                                     one column K times)",
                "per_position": "column k compared against a dedicated K=1 run on that position's \
                                 activations (catches a kernel that collapses k)",
            },
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait",
            "reps": REPS,
            "results": results,
            "commit": std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .unwrap_or_default()
                .trim()
                .to_owned(),
        });

        if let Some(out) = args.out {
            if let Some(p) = out.parent() {
                fs::create_dir_all(p)?;
            }
            fs::write(&out, serde_json::to_string_pretty(&doc)? + "\n")?;
            eprintln!("wrote {}", out.display());
        } else {
            println!("{}", serde_json::to_string_pretty(&doc)?);
        }
        Ok(())
    }

    fn bytemuck_cast(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    fn read_f32(buf: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
    }
}
