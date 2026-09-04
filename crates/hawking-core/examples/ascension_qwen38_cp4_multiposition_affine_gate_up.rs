//! CP4 — multi-position batching for the organ that actually dominates.
//!
//! CP3 (`receipts/runtime/CP3_RESULT.md`) measured 2.474x, bit-identical, from
//! putting K positions through ONE weight sweep on the uniform-q4 DeltaNet
//! in_proj. It could not reach the largest organ. The live resident dispatches
//! `qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128` for
//! `mlp_gate_up` (35.9% of per-step GPU ns) and a group32 affine matvec for
//! `mlp_down` (20.9%), and no affine-q2 kernel in any shader was multi-position
//! -- 56.8% of the step had no batching kernel at all.
//!
//! This measures the affine RxK gate_up kernels added to
//! `shaders/q80_mixed_decode.metal` against the production affine matvec, at the
//! REAL gate_up geometry read from the artifact's own header:
//!
//!   {"representation":"affine_q2_group64_fp16_scale_bias","shape":[17408,5120],
//!    "group_size":64,"code_bytes":22282240,"scale_bytes":2785280,"bias_bytes":2785280}
//!
//! CP3 makes a falsifiable prediction this run exists to test. The affine kernel
//! accumulates gate AND up together, so it holds 2*R*K accumulators against the
//! q4 kernel's R*K, plus a third half-plane read per group for the bias:
//!
//!   live floats/thread ~= 2*R*K + K + 6*R
//!     r2k2 22    r2k4 32    r4k2 42    r4k4 60
//!
//! CP3's q4 curve won at 32 live floats and collapsed by 50-56, so the knee here
//! should land at (2,4) or (4,2), NOT (4,4). If r4k4 wins anyway the register
//! story from CP3 is wrong, and that is worth more than the speedup.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core \
//!     --example ascension_qwen38_cp4_multiposition_affine_gate_up
//! Correctness (safe under GPU contention):
//!   .../ascension_qwen38_cp4_multiposition_affine_gate_up --out /tmp/cp4.json
//! Timing (REQUIRES zero loaded 27B residents, checked by RSS):
//!   ./tools/gpu_lane_lock.sh cp4-affine \
//!     .../ascension_qwen38_cp4_multiposition_affine_gate_up --timing --reps 9 \
//!     --out receipts/runtime/CP4_MULTIPOSITION_AFFINE_GATE_UP.json

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
    use std::time::Instant;

    const SCHEMA: &str = "hawking.cp4.multiposition_affine_gate_up.v1";

    // From the artifact's own hgrafv01 header, not from a constant in this repo.
    const ROWS: usize = 17_408;
    const COLS: usize = 5_120;
    const GROUP: usize = 64;
    const CODE_BYTES: usize = 22_282_240; // rows*cols*2 bits
    const PLANE_HALVES: usize = 1_392_640; // rows * groups_per_row
    const SEGMENT_BYTES: u64 = 27_853_103;
    const HEADER_BYTES: usize = 303; // 8 magic + 4 len + 291 json

    /// `--reverse` walks this backwards. CP3b established that the first cells
    /// of any sweep sit on a DVFS ramp, so an absolute ns/position is ordered,
    /// not intrinsic. The ratio survives because both arms of a cell run
    /// adjacently -- but a claim that compares CELLS to each other (which is
    /// exactly what "where is the knee" asks) must be checked in both orders.
    const GRID: &[(usize, usize)] =
        &[(1, 1), (2, 1), (4, 1), (2, 2), (2, 4), (4, 2), (4, 4)];

    const BASELINE: &str = "qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128";

    fn kernel_for(r: usize, k: usize) -> String {
        format!("qwen_affine_q2_group64_matmul_gate_up_r{r}k{k}_geo_tpr64_tg128")
    }

    /// Real artifact code bytes. Every u8 is a valid quad of 2-bit codes and the
    /// unpack is branchless, so content affects neither timing nor the A-vs-B
    /// comparison; both arms read the identical buffer. What this buys is that
    /// the payload is not invented.
    fn codes_from_artifact(root: &PathBuf) -> (Vec<u8>, Value) {
        let segments = root.join("segments");
        let mut chosen: Option<PathBuf> = None;
        if let Ok(rd) = fs::read_dir(&segments) {
            let mut v: Vec<PathBuf> = rd
                .filter_map(|e| e.ok())
                .filter(|e| e.metadata().map(|m| m.len() == SEGMENT_BYTES).unwrap_or(false))
                .map(|e| e.path())
                .collect();
            v.sort();
            chosen = v.into_iter().next();
        }
        if let Some(path) = chosen {
            if let Ok(bytes) = fs::read(&path) {
                if bytes.len() >= HEADER_BYTES + CODE_BYTES {
                    return (
                        bytes[HEADER_BYTES..HEADER_BYTES + CODE_BYTES].to_vec(),
                        json!({
                            "source": "real_artifact_segment",
                            "path": path.to_string_lossy(),
                            "segment_bytes": SEGMENT_BYTES,
                            "header_bytes_skipped": HEADER_BYTES,
                            "code_bytes_used": CODE_BYTES,
                        }),
                    );
                }
            }
        }
        (
            deterministic(CODE_BYTES),
            json!({"source": "deterministic_fallback",
                   "reason": format!("no {SEGMENT_BYTES}-byte .hgrafv01 segment under {}",
                                     segments.to_string_lossy())}),
        )
    }

    fn deterministic(n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        let mut s: u32 = 0x9e37_79b9;
        for b in v.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (s >> 24) as u8;
        }
        v
    }

    /// Small normals. Random f16 BITS would give NaN/Inf and denormals, which
    /// both poison equality and can take a slow arithmetic path.
    fn halves(n: usize, base: f32, step: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            out.extend_from_slice(&half_bits(base + (i % 13) as f32 * step).to_le_bytes());
        }
        out
    }

    fn half_bits(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = ((bits >> 13) & 0x3ff) as u16;
        if exp <= 0 { sign } else if exp >= 31 { sign | 0x7bff }
        else { sign | ((exp as u16) << 10) | mant }
    }

    fn median(mut v: Vec<u64>) -> u64 { v.sort_unstable(); v[v.len() / 2] }
    fn cast(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }
    fn read_f32(b: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() }
    }
    fn activation(t: usize) -> Vec<f32> {
        (0..COLS).map(|c| ((c % 29) as f32 - 14.0) * (1.0 + t as f32) * 0.0625).collect()
    }

    struct Args { artifact_root: PathBuf, out: Option<PathBuf>, reps: usize, timing: bool, reverse: bool }

    fn parse_args() -> Args {
        let mut artifact_root = PathBuf::from(env::var("HOME").unwrap_or_default())
            .join("noetic").join("NOETIC_PARENT_A");
        let (mut out, mut reps, mut timing) = (None, 7usize, false);
        let mut reverse = false;
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--artifact-root" => artifact_root = PathBuf::from(it.next().unwrap_or_default()),
                "--out" => out = it.next().map(PathBuf::from),
                "--reps" => reps = it.next().unwrap_or_default().parse().unwrap_or(7),
                "--timing" => timing = true,
                "--reverse" => reverse = true,
                o => panic!("unknown flag {o}"),
            }
        }
        Args { artifact_root, out, reps, timing, reverse }
    }

    struct Bufs {
        gc: metal::Buffer, gs: metal::Buffer, gb: metal::Buffer,
        uc: metal::Buffer, us: metal::Buffer, ub: metal::Buffer,
    }

    /// Baseline: K dispatches of the PRODUCTION affine matvec, one per position,
    /// each on its own command buffer -- what the per-token prefill loop does.
    fn arm_seq(
        ctx: &MetalContext, w: &Bufs,
        xs: &[metal::Buffer], go: &[metal::Buffer], uo: &[metal::Buffer],
    ) -> Result<u64, Box<dyn Error>> {
        let grid = ((((ROWS as u32) + 1) / 2) * 128, 1, 1);
        let tg = (128u32, 1, 1);
        let mut gpu = 0u64;
        for i in 0..xs.len() {
            let t = ctx.dispatch_threads_timed(BASELINE, grid, tg, |enc| {
                enc.set_buffer(0, Some(&w.gc), 0); enc.set_buffer(1, Some(&w.gs), 0);
                enc.set_buffer(2, Some(&w.gb), 0); enc.set_buffer(3, Some(&w.uc), 0);
                enc.set_buffer(4, Some(&w.us), 0); enc.set_buffer(5, Some(&w.ub), 0);
                enc.set_buffer(6, Some(&xs[i]), 0);
                enc.set_buffer(7, Some(&go[i]), 0); enc.set_buffer(8, Some(&uo[i]), 0);
                let r = ROWS as u32; let c = COLS as u32;
                enc.set_bytes(9, 4, &r as *const u32 as *const _);
                enc.set_bytes(10, 4, &c as *const u32 as *const _);
            })?;
            match (t.gpu_start_ns, t.gpu_end_ns) {
                (Some(s), Some(e)) if e > s => gpu += e - s,
                _ => return Err(format!("{BASELINE}: no GPU timestamp").into()),
            }
        }
        Ok(gpu)
    }

    #[allow(clippy::too_many_arguments)]
    fn arm_mp(
        ctx: &MetalContext, kernel: &str, r: usize, w: &Bufs,
        xi: &metal::Buffer, go: &metal::Buffer, uo: &metal::Buffer,
    ) -> Result<u64, Box<dyn Error>> {
        let tgs = (ROWS + 2 * r - 1) / (2 * r);
        let grid = ((tgs * 128) as u32, 1, 1);
        let tg = (128u32, 1, 1);
        let t = ctx.dispatch_threads_timed(kernel, grid, tg, |enc| {
            enc.set_buffer(0, Some(&w.gc), 0); enc.set_buffer(1, Some(&w.gs), 0);
            enc.set_buffer(2, Some(&w.gb), 0); enc.set_buffer(3, Some(&w.uc), 0);
            enc.set_buffer(4, Some(&w.us), 0); enc.set_buffer(5, Some(&w.ub), 0);
            enc.set_buffer(6, Some(xi), 0);
            enc.set_buffer(7, Some(go), 0); enc.set_buffer(8, Some(uo), 0);
            let rr = ROWS as u32; let c = COLS as u32;
            enc.set_bytes(9, 4, &rr as *const u32 as *const _);
            enc.set_bytes(10, 4, &c as *const u32 as *const _);
        })?;
        match (t.gpu_start_ns, t.gpu_end_ns) {
            (Some(s), Some(e)) if e > s => Ok(e - s),
            _ => Err(format!("{kernel}: no GPU timestamp").into()),
        }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args();
        let ctx = MetalContext::new()?;
        let (codes, provenance) = codes_from_artifact(&args.artifact_root);
        let w = Bufs {
            gc: ctx.new_buffer_with_bytes_checked(&codes)?,
            gs: ctx.new_buffer_with_bytes_checked(&halves(PLANE_HALVES, 0.0020, 0.00011))?,
            gb: ctx.new_buffer_with_bytes_checked(&halves(PLANE_HALVES, -0.0030, 0.00007))?,
            uc: ctx.new_buffer_with_bytes_checked(&codes)?,
            us: ctx.new_buffer_with_bytes_checked(&halves(PLANE_HALVES, 0.0017, 0.00013))?,
            ub: ctx.new_buffer_with_bytes_checked(&halves(PLANE_HALVES, -0.0021, 0.00009))?,
        };

        let kmax = GRID.iter().map(|c| c.1).max().unwrap();
        let acts: Vec<Vec<f32>> = (0..kmax).map(activation).collect();
        let xb: Vec<metal::Buffer> = acts.iter()
            .map(|a| ctx.new_buffer_with_bytes_checked(cast(a)))
            .collect::<Result<_, _>>()?;
        let gref: Vec<metal::Buffer> = (0..kmax)
            .map(|_| ctx.new_buffer_checked(ROWS * 4)).collect::<Result<_, _>>()?;
        let uref: Vec<metal::Buffer> = (0..kmax)
            .map(|_| ctx.new_buffer_checked(ROWS * 4)).collect::<Result<_, _>>()?;
        arm_seq(&ctx, &w, &xb, &gref, &uref)?;
        let gr: Vec<Vec<f32>> = gref.iter().map(|b| read_f32(b, ROWS)).collect();
        let ur: Vec<Vec<f32>> = uref.iter().map(|b| read_f32(b, ROWS)).collect();

        let mut rows_out: Vec<Value> = Vec::new();
        let mut order: Vec<(usize, usize)> = GRID.to_vec();
        if args.reverse { order.reverse(); }
        for &(r, k) in &order {
            let kernel = kernel_for(r, k);
            let t0 = Instant::now();
            let mut xi = vec![0f32; COLS * k];
            for c in 0..COLS { for kk in 0..k { xi[c * k + kk] = acts[kk][c]; } }
            let interleave_ns = t0.elapsed().as_nanos() as u64;
            let xib = ctx.new_buffer_with_bytes_checked(cast(&xi))?;
            let gob = ctx.new_buffer_checked(ROWS * k * 4)?;
            let uob = ctx.new_buffer_checked(ROWS * k * 4)?;

            arm_mp(&ctx, &kernel, r, &w, &xib, &gob, &uob)?;
            let gg = read_f32(&gob, ROWS * k);
            let gu = read_f32(&uob, ROWS * k);

            let mut max_rel = 0f64;
            for row in 0..ROWS {
                for kk in 0..k {
                    for (got, refv) in [(gg[row * k + kk], gr[kk][row]),
                                        (gu[row * k + kk], ur[kk][row])] {
                        let d = (got as f64 - refv as f64).abs();
                        let rel = d / (refv as f64).abs().max(1e-6);
                        if rel > max_rel { max_rel = rel; }
                    }
                }
            }
            let tol = 1e-4f64;
            let valid = max_rel <= tol;

            // NEGATIVE CONTROL: a check never seen to fail is not evidence.
            let mut neg = 0f64;
            if k > 1 {
                for row in 0..ROWS {
                    for kk in 0..k {
                        let refv = gr[(kk + 1) % k][row] as f64;
                        let rel = (gg[row * k + kk] as f64 - refv).abs() / refv.abs().max(1e-6);
                        if rel > neg { neg = rel; }
                    }
                }
            }
            let neg_rejected = if k > 1 { json!(neg > tol) }
                               else { json!("NOT_APPLICABLE_AT_K1") };

            // UNIFORM-POSITION CONTROL: all K columns identical must reproduce
            // the K=1 answer. Catches a kernel writing one column K times.
            let xu: Vec<f32> = (0..COLS * k).map(|i| acts[0][i / k]).collect();
            let xub = ctx.new_buffer_with_bytes_checked(cast(&xu))?;
            let gou = ctx.new_buffer_checked(ROWS * k * 4)?;
            let uou = ctx.new_buffer_checked(ROWS * k * 4)?;
            arm_mp(&ctx, &kernel, r, &w, &xub, &gou, &uou)?;
            let guu = read_f32(&gou, ROWS * k);
            let mut uni = 0f64;
            for row in 0..ROWS {
                for kk in 0..k {
                    let refv = gr[0][row] as f64;
                    let rel = (guu[row * k + kk] as f64 - refv).abs() / refv.abs().max(1e-6);
                    if rel > uni { uni = rel; }
                }
            }
            let uniform_holds = uni <= tol;

            let (mut sq, mut mq) = (Vec::new(), Vec::new());
            if args.timing && valid {
                let xs: Vec<metal::Buffer> = xb.iter().take(k).cloned().collect();
                let go: Vec<metal::Buffer> = (0..k)
                    .map(|_| ctx.new_buffer_checked(ROWS * 4)).collect::<Result<_, _>>()?;
                let uo: Vec<metal::Buffer> = (0..k)
                    .map(|_| ctx.new_buffer_checked(ROWS * 4)).collect::<Result<_, _>>()?;
                for rep in 0..(args.reps + 1) {
                    let a = arm_seq(&ctx, &w, &xs, &go, &uo)?;
                    let b = arm_mp(&ctx, &kernel, r, &w, &xib, &gob, &uob)?;
                    if rep == 0 { continue; }
                    sq.push(a); mq.push(b);
                }
            }
            let ni = || Value::String("NOT_INSTRUMENTED".into());
            let (sm, mm, sp, nps, npm) = if sq.is_empty() {
                (ni(), ni(), ni(), ni(), ni())
            } else {
                let s = median(sq.clone()) as f64;
                let m = median(mq.clone()) as f64;
                (json!(s as u64), json!(m as u64), json!(s / m),
                 json!(s / k as f64), json!(m / k as f64))
            };

            rows_out.push(json!({
                "r": r, "k": k, "kernel": kernel,
                "status": if valid { "MEASURED" } else { "CORRECTNESS_FAILED" },
                "valid": valid,
                "max_rel_err": max_rel, "tolerance": tol,
                "negative_control_rejected": neg_rejected,
                "negative_control_max_rel": if k > 1 { json!(neg) } else { Value::Null },
                "uniform_control_holds": uniform_holds,
                "uniform_control_max_rel": uni,
                "dispatch_count_seq": k, "dispatch_count_mp": 1,
                "seq_gpu_ns_reps": sq, "mp_gpu_ns_reps": mq,
                "seq_gpu_ns_median": sm, "mp_gpu_ns_median": mm,
                "ns_per_position_seq": nps, "ns_per_position_mp": npm,
                "speedup_x": sp,
                "code_bytes_moved": 2 * CODE_BYTES,
                "scale_plus_bias_bytes": 4 * PLANE_HALVES * 2,
                "activation_bytes_moved": COLS * k * 4,
                "output_bytes": 2 * ROWS * k * 4,
                "interleave_transform_ns": interleave_ns,
                "temporary_bytes": COLS * k * 4 + 2 * ROWS * k * 4,
                "flops": 2.0 * (2 * ROWS * COLS * k) as f64,
                "threadgroup_bytes": 32 * r * k,
                "accumulators_per_thread": 2 * r * k,
                "live_floats_per_thread": 2 * r * k + k + 6 * r,
            }));
            eprintln!("r{r}k{k}  valid={valid} max_rel={max_rel:.3e} \
neg_rejected={neg_rejected} uniform={uniform_holds}");
        }

        let doc = json!({
            "schema": SCHEMA,
            "organ": "mlp_gate_up",
            "organ_share_of_step_gpu_ns": 0.359,
            "organ_share_source": "receipts/headless/_ORGAN_BANDWIDTH_raw.json isolated_organs",
            "codec": "affine_q2_group64_fp16_scale_bias (.hgrafv01), from the artifact's own header",
            "geometry": {"rows": ROWS, "cols": COLS, "group_size": GROUP,
                         "code_bytes_per_tensor": CODE_BYTES,
                         "scale_halves_per_tensor": PLANE_HALVES},
            "code_provenance": provenance,
            "baseline_kernel": BASELINE,
            "baseline_note": "K separate command buffers, one per position, which is what the \
per-token prefill loop at qwen38_hybrid_decode.rs:7566 physically does. The production path uses \
the SWIGLU-fused variant; the non-fused pair variant is used here so gate and up stay separable \
for the correctness check, and both arms use it, so the comparison isolates batching.",
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
            "cp3_prediction_under_test": "CP3's q4 curve won at 32 live floats/thread and collapsed \
by 50-56. Here live floats = 2*R*K + K + 6*R, so r2k4=32 and r4k4=60. The knee should land at (2,4) \
or (4,2), NOT (4,4). If r4k4 wins anyway, CP3's register story is wrong.",
            "timing_enabled": args.timing,
            "sweep_order": if args.reverse { "REVERSED" } else { "forward" },
            "reps": if args.timing { json!(args.reps) } else { Value::String("NOT_INSTRUMENTED".into()) },
            "warmup_discarded": 1,
            "results": rows_out,
        });
        match args.out {
            Some(p) => {
                if let Some(d) = p.parent() { fs::create_dir_all(d)?; }
                fs::write(&p, serde_json::to_string_pretty(&doc)? + "\n")?;
                eprintln!("wrote {}", p.display());
            }
            None => println!("{}", serde_json::to_string_pretty(&doc)?),
        }
        Ok(())
    }
}
