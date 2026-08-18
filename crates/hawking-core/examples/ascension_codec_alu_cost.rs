//! What does a codec cost per WEIGHT, not per byte?
//!
//! DENSITY_LEADER_SPEED.json measured the coherent q3 artifact running 10.9%
//! SLOWER than G0 while moving 17.4% fewer bytes, and NX_MATMUL_K_AMORTIZATION
//! fitted the geo_tpr64 K sweep to T(K) ~ (decode_ops + K), i.e. the GEMV is
//! ALU-issue bound. Together those say bytes are the wrong currency on this
//! machine and decode ops per weight is the right one. Nobody has measured it
//! per codec.
//!
//! This does, with the confound removed: identical rows, identical cols,
//! identical launch geometry, identical thread map. The ONLY difference between
//! the two dispatches is how many ALU ops it takes to turn stored bits into a
//! float. Element count is held equal, so byte counts differ -- which is the
//! point. A codec that stores fewer bytes but costs more ops shows up here as a
//! higher ps/element.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core \
//!     --example ascension_codec_alu_cost
//! Measure (GPU mutex required):
//!   ./tools/gpu_lane_lock.sh codec-alu-cost \
//!     workspace/ops/build/rust/release-fast/examples/ascension_codec_alu_cost \
//!     --out receipts/ascent-2026-08-16/CODEC_ALU_COST.json

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

    const SCHEMA: &str = "hawking.nos.codec_alu_cost.v1";
    const COLS: usize = 5120;
    const GROUP: usize = 64;
    const REPS: usize = 5;

    struct Codec {
        name: &'static str,
        kernel: &'static str,
        code_bytes_per_group: usize,
        bits: u32,
        bound: u32,
        /// q3 kernel takes (group_size, bits, bound) where q4 takes groups_per_row.
        hgravu_abi: bool,
    }

    fn codecs() -> Vec<Codec> {
        vec![
            Codec {
                name: "q4_group64",
                kernel: "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                code_bytes_per_group: 32,
                bits: 4,
                bound: 8,
                hgravu_abi: false,
            },
            Codec {
                name: "q4_group64_hgravu",
                kernel: "qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128",
                code_bytes_per_group: 32,
                bits: 4,
                bound: 7,
                hgravu_abi: true,
            },
            Codec {
                name: "q4_group64_contiguous",
                kernel: "qwen_uniform_q4_group64_matvec_contig_tg128",
                code_bytes_per_group: 32, bits: 4, bound: 8, hgravu_abi: false,
            },
            Codec {
                name: "q3_group64_hgravu",
                kernel: "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128",
                code_bytes_per_group: 24,
                bits: 3,
                bound: 3,
                hgravu_abi: true,
            },
            Codec {
                name: "q3_group64_alignedload",
                kernel: "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128_alignedload",
                code_bytes_per_group: 24, bits: 3, bound: 3, hgravu_abi: true,
            },
            Codec { name: "binary_planes_k1", kernel: "qwen_binary_planes_k1_matvec_geo_tpr64_tg128",
                    code_bytes_per_group: 8, bits: 1, bound: 0, hgravu_abi: false },
            Codec { name: "binary_planes_k2", kernel: "qwen_binary_planes_k2_matvec_geo_tpr64_tg128",
                    code_bytes_per_group: 16, bits: 1, bound: 0, hgravu_abi: false },
            Codec { name: "binary_planes_k3", kernel: "qwen_binary_planes_k3_matvec_geo_tpr64_tg128",
                    code_bytes_per_group: 24, bits: 1, bound: 0, hgravu_abi: false },
        ]
    }

    fn median(mut v: Vec<u64>) -> u64 {
        v.sort_unstable();
        v[v.len() / 2]
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut out: Option<PathBuf> = None;
        let mut rows_list = vec![262_144usize, 1_048_576usize];
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--out" => out = it.next().map(PathBuf::from),
                "--rows" => {
                    rows_list = it
                        .next()
                        .unwrap_or_default()
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect()
                }
                o => panic!("unknown flag {o}"),
            }
        }

        let ctx = MetalContext::new()?;
        let x: Vec<f32> = (0..COLS).map(|c| ((c % 31) as f32) - 15.0).collect();
        let xbuf = ctx.new_buffer_with_bytes_checked(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, COLS * 4)
        })?;

        let mut results: Vec<Value> = Vec::new();
        for &rows in &rows_list {
            let elements = rows * COLS;
            let groups_per_row = COLS / GROUP;
            let mut per_codec: Vec<Value> = Vec::new();
            eprintln!("rows={rows} elements={elements}");

            for c in codecs() {
                let code_bytes = rows * groups_per_row * c.code_bytes_per_group;
                let mut codes = vec![0u8; code_bytes];
                let mut s: u32 = 0x9e37_79b9;
                for b in codes.iter_mut() {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    *b = (s >> 24) as u8;
                }
                let scale_slots = rows * groups_per_row
                    * if c.name.starts_with("binary_planes") {
                        c.code_bytes_per_group / 8
                    } else { 1 };
                let mut scales = Vec::with_capacity(scale_slots * 2);
                for i in 0..scale_slots {
                    // small positive f16 scales
                    let h: u16 = 0x1c00u16.wrapping_add((i % 13) as u16);
                    scales.extend_from_slice(&h.to_le_bytes());
                }
                let cbuf = ctx.new_buffer_with_bytes_checked(&codes)?;
                let sbuf = ctx.new_buffer_with_bytes_checked(&scales)?;
                let obuf = ctx.new_buffer_checked(rows * 4)?;
                let grid = ((((rows as u32) + 1) / 2) * 128, 1, 1);
                let tg = (128u32, 1, 1);

                let mut gpu_ns = Vec::new();
                for _ in 0..REPS {
                    let t = ctx.dispatch_threads_timed(c.kernel, grid, tg, |enc| {
                        enc.set_buffer(0, Some(&cbuf), 0);
                        enc.set_buffer(1, Some(&sbuf), 0);
                        enc.set_buffer(2, Some(&xbuf), 0);
                        enc.set_buffer(3, Some(&obuf), 0);
                        let r = rows as u32;
                        let co = COLS as u32;
                        enc.set_bytes(4, 4, &r as *const u32 as *const _);
                        enc.set_bytes(5, 4, &co as *const u32 as *const _);
                        if c.hgravu_abi {
                            let g = GROUP as u32;
                            enc.set_bytes(6, 4, &g as *const u32 as *const _);
                            enc.set_bytes(7, 4, &c.bits as *const u32 as *const _);
                            enc.set_bytes(8, 4, &c.bound as *const u32 as *const _);
                        } else {
                            let gpr = groups_per_row as u32;
                            enc.set_bytes(6, 4, &gpr as *const u32 as *const _);
                        }
                    })?;
                    match (t.gpu_start_ns, t.gpu_end_ns) {
                        (Some(a), Some(b)) if b > a => gpu_ns.push(b - a),
                        _ => return Err(format!("{}: no GPU timestamp", c.kernel).into()),
                    }
                }
                let med = median(gpu_ns.clone());
                let ps_per_elem = med as f64 * 1000.0 / elements as f64;
                let gb_s = code_bytes as f64 / med as f64;
                eprintln!(
                    "  {:<20} {:8.3} ms  {:7.2} ps/elem  codes {:6.3} GB @ {:6.1} GB/s",
                    c.name,
                    med as f64 / 1e6,
                    ps_per_elem,
                    code_bytes as f64 / 1e9,
                    gb_s
                );
                per_codec.push(json!({
                    "codec": c.name, "kernel": c.kernel, "bits": c.bits,
                    "code_bytes_per_group": c.code_bytes_per_group,
                    "code_bytes": code_bytes, "elements": elements,
                    "median_gpu_ns": med, "gpu_ns_reps": gpu_ns,
                    "ps_per_element": ps_per_elem, "code_gb_s": gb_s,
                    "bytes_per_element": c.code_bytes_per_group as f64 / GROUP as f64,
                }));
            }
            results.push(json!({"rows": rows, "cols": COLS, "elements": elements,
                                "by_codec": per_codec}));
        }

        // G044: the roofline knee, MEASURED. Sweep arithmetic intensity and find
        // where achieved bandwidth stops being flat.
        let mut roofline: Vec<Value> = Vec::new();
        {
            let n_vec: usize = 64 * 1024 * 1024; // 1 GiB of float4
            let bytes = n_vec * 16;
            let inbuf = ctx.new_buffer_checked(bytes)?;
            let outbuf = ctx.new_buffer_checked(n_vec * 4)?;
            eprintln!("roofline sweep: {} MiB streamed per point", bytes / (1024 * 1024));
            for k in [1u32, 2, 4, 8, 16, 32, 64, 128, 256] {
                let mut gpu_ns = Vec::new();
                for _ in 0..REPS {
                    let t = ctx.dispatch_threads_timed(
                        "hawking_roofline_sweep_f4",
                        (n_vec as u32, 1, 1),
                        (256, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&inbuf), 0);
                            enc.set_buffer(1, Some(&outbuf), 0);
                            let n = n_vec as u32;
                            enc.set_bytes(2, 4, &n as *const u32 as *const _);
                            enc.set_bytes(3, 4, &k as *const u32 as *const _);
                        },
                    )?;
                    match (t.gpu_start_ns, t.gpu_end_ns) {
                        (Some(a), Some(b)) if b > a => gpu_ns.push(b - a),
                        _ => return Err("roofline: no GPU timestamp".into()),
                    }
                }
                let med = median(gpu_ns.clone());
                let gb_s = bytes as f64 / med as f64;
                let flops = (n_vec as f64) * (k as f64) * 8.0;
                let gflop_s = flops / med as f64;
                let intensity = k as f64 / 2.0;
                eprintln!(
                    "  K={k:<4} intensity {intensity:6.1} flop/B   {gb_s:7.1} GB/s   {gflop_s:8.1} GFLOP/s"
                );
                roofline.push(json!({
                    "k_ops": k, "flops_per_byte": intensity, "bytes": bytes,
                    "median_gpu_ns": med, "achieved_gb_s": gb_s, "achieved_gflop_s": gflop_s,
                }));
            }
        }

        let doc = json!({
            "schema": SCHEMA,
            "roofline_sweep": roofline,
            "roofline_method": "hawking_roofline_sweep_f4 streams 1 GiB as float4 and performs K \
                                FMAs per vector on two independent chains before folding every \
                                result into the output, so nothing is eliminated and the measurement \
                                is throughput rather than dependency latency. flops/byte = K/2.",
            "question": "Per WEIGHT, not per byte, what does each codec cost on this GPU?",
            "why": "DENSITY_LEADER_SPEED.json: the q3 artifact runs 10.9% slower than G0 on 17.4% \
                    fewer bytes. NX_MATMUL_K_AMORTIZATION fitted geo_tpr64 to T(K) ~ (decode_ops + K), \
                    an ALU-issue bound. If that is right, ps/element is the currency and bytes are not.",
            "control": "Identical rows, cols, launch geometry and thread map across codecs. Element \
                        count held EQUAL so byte counts differ -- a codec that stores fewer bytes but \
                        costs more ops appears as a higher ps/element. q4_group64_hgravu is a same-\
                        width control: it stores exactly as many bytes as q4_group64 and differs only \
                        in ABI and offset convention, so any gap between those two is measurement \
                        noise plus ABI, not codec width.",
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait",
            "reps": REPS,
            "results": results,
            "commit": std::process::Command::new("git").args(["rev-parse","HEAD"]).output().ok()
                .and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default().trim().to_owned(),
        });
        if let Some(p) = out {
            if let Some(d) = p.parent() { fs::create_dir_all(d)?; }
            fs::write(&p, serde_json::to_string_pretty(&doc)? + "\n")?;
            eprintln!("wrote {}", p.display());
        } else {
            println!("{}", serde_json::to_string_pretty(&doc)?);
        }
        Ok(())
    }
}
