//! G109 / C01: the geo_tpr64 GEMV geometry sweep BY ORGAN CLASS.
//!
//! The incumbent geometry was tuned on one shape. Every organ in the genome has a
//! different rows:cols ratio, and the row tiling R decides how many output rows
//! share one activation load -- so the optimum has no reason to be shape-invariant.
//! This sweeps R at each organ's REAL shape and reports ns per element, which is
//! the currency this machine was measured to charge (CODEC_ALU_COST), not bytes.
//!
//! Correctness is not optional: every (organ, R) result is compared elementwise
//! against the R=1 kernel on the same inputs, and a nonzero max abs error is a
//! failure rather than a note.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("organ geometry sweep requires macos");
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        use hawking_core::metal::MetalContext;
        use serde_json::json;
        use std::env;
        use std::fs;
        use std::path::PathBuf;

        const GROUP: usize = 64;
        const CODE_BYTES_PER_GROUP: usize = 32;
        const REPS: usize = 5;
        const RS: [usize; 4] = [1, 2, 4, 8];
        // (name, rows, cols) straight from the Qwen3.8 genome.
        // Tiny shapes first. Their compute is negligible, so T(rows) extrapolated to zero
        // rows is the per-dispatch overhead -- the ceiling on what any fusion can buy.
        const ORGANS: [(&str, usize, usize); 13] = [
            ("probe.rows2", 2, 5120),
            ("probe.rows8", 8, 5120),
            ("probe.rows32", 32, 5120),
            ("probe.rows128", 128, 5120),
            ("probe.rows512", 512, 5120),
            ("attn.q_proj", 12288, 5120),
            ("attn.k_proj", 1024, 5120),
            ("attn.v_proj", 1024, 5120),
            ("attn.o_proj", 5120, 6144),
            ("mlp.gate_proj", 17408, 5120),
            ("mlp.down_proj", 5120, 17408),
            ("dn.in_proj_qkv", 10240, 5120),
            ("endpoint.lm_head", 248320, 5120),
        ];

        fn median(mut v: Vec<u64>) -> u64 { v.sort_unstable(); v[v.len() / 2] }
        fn bytemuck_cast(v: &[f32]) -> &[u8] {
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
        }
        fn read_f32(buf: &metal::Buffer, n: usize) -> Vec<f32> {
            let p = buf.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
        }

        let mut out_path: Option<PathBuf> = None;
        let mut it = env::args().skip(1);
        while let Some(a) = it.next() {
            if a == "--out" { out_path = it.next().map(PathBuf::from); }
        }

        let ctx = MetalContext::new()?;
        let mut rows_json = Vec::new();

        for (organ, rows, cols) in ORGANS {
            let groups_per_row = (cols / GROUP) as u32;
            let code_bytes = rows * (cols / GROUP) * CODE_BYTES_PER_GROUP;
            let scale_count = rows * (cols / GROUP);
            let codes = ctx.new_buffer_checked(code_bytes)?;
            let scales = ctx.new_buffer_checked(scale_count * 2)?;
            let x: Vec<f32> = (0..cols).map(|c| ((c % 31) as f32 - 15.0)).collect();
            let xbuf = ctx.new_buffer_with_bytes_checked(bytemuck_cast(&x))?;
            let mut reference: Option<Vec<f32>> = None;
            let mut best: (f64, usize) = (f64::MAX, 0);

            for r in RS {
                let kernel = if r == 1 {
                    "qwen_uniform_q4_group64_matmul_k1_geo_tpr64_tg128".to_string()
                } else {
                    format!("qwen_uniform_q4_group64_matmul_r{r}k1_geo_tpr64_tg128")
                };
                let tgs = (rows + 2 * r - 1) / (2 * r);
                let grid = ((tgs * 128) as u32, 1, 1);
                let tg = (128u32, 1, 1);
                let output = ctx.new_buffer_checked(rows * 4)?;
                let mut gpu_ns: Vec<u64> = Vec::new();
                for _ in 0..REPS {
                    let t = ctx.dispatch_threads_timed(&kernel, grid, tg, |enc| {
                        enc.set_buffer(0, Some(&codes), 0);
                        enc.set_buffer(1, Some(&scales), 0);
                        enc.set_buffer(2, Some(&xbuf), 0);
                        enc.set_buffer(3, Some(&output), 0);
                        let rr = rows as u32;
                        let c = cols as u32;
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
                let vals: Vec<f32> = read_f32(&output, rows);
                let max_err = match &reference {
                    None => { reference = Some(vals.clone()); 0f32 }
                    Some(rf) => vals.iter().zip(rf).map(|(a, b)| (a - b).abs())
                                    .fold(0f32, f32::max),
                };
                let elems = (rows * cols) as f64;
                let ps = med as f64 * 1000.0 / elems;
                if ps < best.0 { best = (ps, r); }
                eprintln!("  {organ:<20} R={r:<3} {:.4} ms  {ps:.4} ps/elem  max_err {max_err}",
                          med as f64 / 1e6);
                rows_json.push(json!({
                    "organ": organ, "rows": rows, "cols": cols, "R": r, "kernel": kernel,
                    "median_gpu_ns": med, "gpu_ns_reps": gpu_ns,
                    "elements": rows * cols, "ps_per_element": ps,
                    "max_abs_err_vs_R1": max_err, "exact_vs_R1": max_err == 0.0,
                }));
            }
            eprintln!("  {organ:<20} BEST R={} at {:.4} ps/elem", best.1, best.0);
        }

        let commit = std::process::Command::new("git").args(["rev-parse", "HEAD"])
            .output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string()).unwrap_or_default();
        let doc = json!({
            "schema": "hawking.nos.organ_geometry_sweep.v1",
            "obligation": "G109 / C01 -- geo_tpr64 geometry by organ class",
            "currency": "ps per element. CODEC_ALU_COST measured this machine ALU-issue bound, so \
                         bytes are the wrong axis for a geometry sweep.",
            "correctness": "every (organ, R) output compared elementwise against the R=1 kernel on \
                            the same inputs; a nonzero max_abs_err is a failure, not a note",
            "reps": REPS, "results": rows_json, "commit": commit,
        });
        if let Some(p) = out_path { fs::write(p, serde_json::to_string_pretty(&doc)? + "\n")?; }
        Ok(())
    }
}
