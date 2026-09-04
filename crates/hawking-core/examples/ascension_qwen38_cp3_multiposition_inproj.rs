//! CP3 — one organ, one layer, physical: does multi-position batching survive
//! at the REAL tensor shape?
//!
//! Standing evidence going in (do not re-derive):
//!
//! * `receipts/headless/_ORGAN_BANDWIDTH_raw.json` ranks the organs of a real
//!   step by GPU ns, 7 reps, GPUStartTime/GPUEndTime, noop control 125 ns:
//!   mlp_gate_up 35.9%, deltanet 24.1%, mlp_down 20.9%, q4_remainder 7.7%,
//!   gqa 6.5%, lm_head 3.8%, sampling 1.2%, embedding 0.0%.
//! * `receipts/ascent-2026-08-18/K_AMORTIZATION_FRESH.json` measured the
//!   multi-position `matmul_rRkK` family at cols=5120 on a 4 GiB SYNTHETIC
//!   payload: peak r4k4 = 2.564x per position, K-alone dies by K=8 (0.832x).
//!
//! What that evidence does NOT establish, and this harness measures:
//!
//! 1. The synthetic sweep used rows = 1,677,720. A real Qwen3.8 tensor has
//!    rows = 16,384. That is a 102x smaller row count, so occupancy, tail
//!    effects and the launch:work ratio are all different. A win at 4 GiB is
//!    not a win at 42 MB until it is measured at 42 MB.
//! 2. The `matmul_rRkK` family exists for `qwen_uniform_q4` ONLY. The live
//!    resident's own dispatch record shows mlp_gate_up running
//!    `qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128` and
//!    mlp_down running `qwen_affine_q2_group32_matvec_geo_tpr64_tg128` -- and
//!    `grep "kernel void qwen_affine_q2.*matmul"` over every shader returns
//!    nothing. So 56.8% of per-step GPU ns runs on a codec with ZERO
//!    multi-position kernels. This harness therefore targets the largest organ
//!    that DOES run uniform q4: the DeltaNet in_proj, 24.1%, rows 16384 x
//!    cols 5120, of which 48 real segments sit on disk at 44,564,520 B each.
//!
//! The result here is a precondition for the affine work: if multi-position
//! collapses at real row counts, authoring an affine multi-position kernel
//! would be building on a refuted premise.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core \
//!     --example ascension_qwen38_cp3_multiposition_inproj
//! Correctness only (safe under GPU contention):
//!   .../ascension_qwen38_cp3_multiposition_inproj --out /tmp/cp3_correct.json
//! Timing (REQUIRES a quiet GPU lane):
//!   ./tools/gpu_lane_lock.sh cp3-multiposition \
//!     .../ascension_qwen38_cp3_multiposition_inproj --timing --reps 9 \
//!     --out receipts/headless/CP3_MULTIPOSITION_INPROJ.json

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

    const SCHEMA: &str = "hawking.cp3.multiposition_inproj.v1";

    /// Real Qwen3.8 DeltaNet in_proj_qkvz geometry. QWEN38_QKVZ_ROWS = 16384,
    /// QWEN38_HIDDEN = 5120, uniform-q4 group 64. A segment on disk is
    /// rows * groups_per_row * 34 + 40 = 44,564,520 B, and 48 of them exist
    /// (one per DeltaNet layer).
    const ROWS: usize = 16_384;
    const COLS: usize = 5_120;
    const GROUP: usize = 64;
    const CODE_BYTES_PER_GROUP: usize = 32;
    const SEGMENT_BYTES: u64 = 44_564_520;

    /// The grid this sweep intends to cover.
    ///
    /// `--reverse` walks it backwards. That is CP3b's discriminator, not a
    /// convenience: the first pass measured the R=1 cells (which run first) at
    /// ~128,750 ns/position against a steady ~74,500 across the twelve later
    /// cells. Two mechanisms explain that equally well -- the sequential arm
    /// re-reading the same 41.9 MB tensor K times back to back and hitting
    /// cache (which a real 64-layer prefill loop would NOT get, making the win
    /// a floor), or first-cells warm-up / DVFS ramp (making 2.474x the honest
    /// number). Reversing the order separates them: under warm-up the high cost
    /// follows the ORDER, under cache it follows the CELL.
    const GRID: &[(usize, usize)] = &[
        (1, 1), (1, 2), (1, 4), (1, 8),
        (2, 1), (4, 1), (8, 1), (16, 1),
        (2, 2), (2, 4), (2, 8),
        (4, 2), (4, 4), (4, 8),
        (8, 2), (8, 4), (8, 8),
        (16, 2), (16, 4), (16, 8),
    ];

    /// Kernel name for an (R, K) cell, or None when the shipped shader has no
    /// such instantiation. A cell with no kernel is REPORTED, never dropped.
    fn kernel_for(r: usize, k: usize) -> Option<String> {
        let have_rk = matches!(
            (r, k),
            (2, 1) | (4, 1) | (8, 1) | (16, 1)
                | (2, 2) | (2, 4) | (4, 2) | (4, 4)
                | (4, 8) | (8, 4) | (8, 8) | (16, 4)
        );
        if r == 1 && matches!(k, 1 | 2 | 4 | 8) {
            return Some(format!(
                "qwen_uniform_q4_group64_matmul_k{k}_geo_tpr64_tg128"
            ));
        }
        if have_rk {
            return Some(format!(
                "qwen_uniform_q4_group64_matmul_r{r}k{k}_geo_tpr64_tg128"
            ));
        }
        None
    }

    const BASELINE_KERNEL: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128";

    fn row_bytes() -> usize {
        (COLS / GROUP) * CODE_BYTES_PER_GROUP
    }

    /// Code bytes taken from a REAL artifact segment when one is available.
    ///
    /// Every u8 is a valid pair of q4 nibbles, so raw artifact bytes can be
    /// used directly without parsing the codec container. The unpack in these
    /// kernels is branchless -- shift, mask, integer-to-float, multiply, FMA --
    /// so timing does not depend on code CONTENT at all; what this buys is
    /// removing the objection that the payload was invented. Both arms read the
    /// identical buffer, so correctness is content-agnostic either way.
    fn codes_from_artifact(root: &PathBuf) -> (Vec<u8>, Value) {
        let need = ROWS * row_bytes();
        let segments = root.join("segments");
        let mut chosen: Option<PathBuf> = None;
        if let Ok(rd) = fs::read_dir(&segments) {
            let mut names: Vec<PathBuf> = rd
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.metadata().map(|m| m.len() == SEGMENT_BYTES).unwrap_or(false)
                })
                .map(|e| e.path())
                .collect();
            names.sort();
            chosen = names.into_iter().next();
        }
        match chosen {
            Some(path) => match fs::read(&path) {
                Ok(bytes) if bytes.len() >= 40 + need => {
                    let v = bytes[40..40 + need].to_vec();
                    (
                        v,
                        json!({
                            "source": "real_artifact_segment",
                            "path": path.to_string_lossy(),
                            "segment_bytes": SEGMENT_BYTES,
                            "header_bytes_skipped": 40,
                            "code_bytes_used": need,
                            "note": "raw artifact bytes used as the q4 code plane; \
the container was not parsed. Every u8 is a valid nibble pair and the unpack is \
branchless, so content affects neither timing nor the A-vs-B comparison."
                        }),
                    )
                }
                _ => (deterministic_codes(need), json!({
                    "source": "deterministic_fallback",
                    "reason": "segment present but shorter than the code plane",
                })),
            },
            None => (
                deterministic_codes(need),
                json!({
                    "source": "deterministic_fallback",
                    "reason": format!(
                        "no {SEGMENT_BYTES}-byte segment under {}",
                        segments.to_string_lossy()
                    ),
                }),
            ),
        }
    }

    fn deterministic_codes(n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        let mut s: u32 = 0x9e37_79b9;
        for b in v.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (s >> 24) as u8;
        }
        v
    }

    /// Small positive normals. Random f16 BITS would produce NaN/Inf and
    /// denormals, which both poison an equality comparison and can take a slow
    /// arithmetic path -- so scales are generated, not scavenged.
    fn scales_for(rows: usize) -> Vec<u8> {
        let n = rows * (COLS / GROUP);
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let f = 0.001_f32 + (i % 17) as f32 * 0.000_1_f32;
            out.extend_from_slice(&half_bits(f).to_le_bytes());
        }
        out
    }

    fn half_bits(f: f32) -> u16 {
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

    fn cast(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    fn read_f32(buf: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
    }

    /// Position t's activation vector, contiguous, exactly as a per-token path
    /// already has it. Distinct per position so a kernel that collapses k
    /// cannot pass.
    fn activation(t: usize) -> Vec<f32> {
        (0..COLS)
            .map(|col| ((col % 31) as f32 - 15.0) * (1.0 + t as f32) * 0.125)
            .collect()
    }

    struct Args {
        artifact_root: PathBuf,
        out: Option<PathBuf>,
        reps: usize,
        timing: bool,
        reverse: bool,
    }

    fn parse_args() -> Args {
        let mut artifact_root = PathBuf::from(
            env::var("HOME").unwrap_or_default(),
        )
        .join("noetic")
        .join("NOETIC_PARENT_A");
        let mut out = None;
        let mut reps = 7usize;
        let mut timing = false;
        let mut reverse = false;
        let mut it = env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--artifact-root" => {
                    artifact_root = PathBuf::from(it.next().unwrap_or_default())
                }
                "--out" => out = it.next().map(PathBuf::from),
                "--reps" => reps = it.next().unwrap_or_default().parse().unwrap_or(7),
                "--timing" => timing = true,
                "--reverse" => reverse = true,
                other => panic!("unknown flag {other}"),
            }
        }
        Args { artifact_root, out, reps, timing, reverse }
    }

    struct Arm {
        gpu_ns: u64,
        host_ns: u64,
        dispatches: usize,
    }

    /// Arm A -- the baseline that must be retained. K successive dispatches of
    /// the PRODUCTION batch-1 matvec, one per position, each on its own command
    /// buffer, which is exactly what a per-token prefill loop does.
    fn arm_sequential(
        ctx: &MetalContext,
        codes: &metal::Buffer,
        scales: &metal::Buffer,
        xs: &[metal::Buffer],
        outs: &[metal::Buffer],
        gpr: u32,
    ) -> Result<Arm, Box<dyn Error>> {
        let grid = ((((ROWS as u32) + 1) / 2) * 128, 1, 1);
        let tg = (128u32, 1, 1);
        let host = Instant::now();
        let mut gpu = 0u64;
        for (x, o) in xs.iter().zip(outs.iter()) {
            let t = ctx.dispatch_threads_timed(BASELINE_KERNEL, grid, tg, |enc| {
                enc.set_buffer(0, Some(codes), 0);
                enc.set_buffer(1, Some(scales), 0);
                enc.set_buffer(2, Some(x), 0);
                enc.set_buffer(3, Some(o), 0);
                let r = ROWS as u32;
                let c = COLS as u32;
                enc.set_bytes(4, 4, &r as *const u32 as *const _);
                enc.set_bytes(5, 4, &c as *const u32 as *const _);
                enc.set_bytes(6, 4, &gpr as *const u32 as *const _);
            })?;
            match (t.gpu_start_ns, t.gpu_end_ns) {
                (Some(s), Some(e)) if e > s => gpu += e - s,
                _ => return Err(format!("{BASELINE_KERNEL}: no GPU timestamp").into()),
            }
        }
        Ok(Arm {
            gpu_ns: gpu,
            host_ns: host.elapsed().as_nanos() as u64,
            dispatches: xs.len(),
        })
    }

    /// Arm B -- one dispatch covering K positions.
    #[allow(clippy::too_many_arguments)]
    fn arm_multiposition(
        ctx: &MetalContext,
        kernel: &str,
        r: usize,
        codes: &metal::Buffer,
        scales: &metal::Buffer,
        xi: &metal::Buffer,
        out: &metal::Buffer,
        gpr: u32,
    ) -> Result<Arm, Box<dyn Error>> {
        let tgs = (ROWS + 2 * r - 1) / (2 * r);
        let grid = ((tgs * 128) as u32, 1, 1);
        let tg = (128u32, 1, 1);
        let host = Instant::now();
        let t = ctx.dispatch_threads_timed(kernel, grid, tg, |enc| {
            enc.set_buffer(0, Some(codes), 0);
            enc.set_buffer(1, Some(scales), 0);
            enc.set_buffer(2, Some(xi), 0);
            enc.set_buffer(3, Some(out), 0);
            let rr = ROWS as u32;
            let c = COLS as u32;
            enc.set_bytes(4, 4, &rr as *const u32 as *const _);
            enc.set_bytes(5, 4, &c as *const u32 as *const _);
            enc.set_bytes(6, 4, &gpr as *const u32 as *const _);
        })?;
        let gpu = match (t.gpu_start_ns, t.gpu_end_ns) {
            (Some(s), Some(e)) if e > s => e - s,
            _ => return Err(format!("{kernel}: no GPU timestamp").into()),
        };
        Ok(Arm {
            gpu_ns: gpu,
            host_ns: host.elapsed().as_nanos() as u64,
            dispatches: 1,
        })
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args();
        let ctx = MetalContext::new()?;
        let gpr = (COLS / GROUP) as u32;

        let (code_bytes, code_provenance) = codes_from_artifact(&args.artifact_root);
        let codes = ctx.new_buffer_with_bytes_checked(&code_bytes)?;
        let scales = ctx.new_buffer_with_bytes_checked(&scales_for(ROWS))?;
        let code_plane_bytes = code_bytes.len();
        let scale_plane_bytes = ROWS * (COLS / GROUP) * 2;

        let kmax = GRID.iter().map(|c| c.1).max().unwrap();
        let acts: Vec<Vec<f32>> = (0..kmax).map(activation).collect();

        // Ground truth per position, from the PRODUCTION kernel, computed once.
        let x_bufs: Vec<metal::Buffer> = acts
            .iter()
            .map(|a| ctx.new_buffer_with_bytes_checked(cast(a)))
            .collect::<Result<_, _>>()?;
        let ref_bufs: Vec<metal::Buffer> = (0..kmax)
            .map(|_| ctx.new_buffer_checked(ROWS * 4))
            .collect::<Result<_, _>>()?;
        arm_sequential(&ctx, &codes, &scales, &x_bufs, &ref_bufs, gpr)?;
        let reference: Vec<Vec<f32>> =
            ref_bufs.iter().map(|b| read_f32(b, ROWS)).collect();

        let mut rows_out: Vec<Value> = Vec::new();

        let mut order: Vec<(usize, usize)> = GRID.to_vec();
        if args.reverse {
            order.reverse();
        }
        for &(r, k) in &order {
            let Some(kernel) = kernel_for(r, k) else {
                rows_out.push(json!({
                    "r": r, "k": k, "kernel": Value::Null, "valid": false,
                    "status": "NOT_INSTANTIATED",
                    "reason": "the shipped qwen_uniform_q4.metal has no QWEN_UNIFORM_Q4_MATMUL_RK \
instantiation for this cell. Adding one edits a production shader that is compiled at resident \
startup, so it is deferred to CP3b rather than done under a live resident.",
                    "threadgroup_bytes_if_added": 16 * r * k,
                    "accumulators_per_thread_if_added": r * k,
                    "live_floats_per_thread_if_added": r * k + 2 * k + 2 * r,
                }));
                continue;
            };

            // --- interleave: input[col * K + k]. This transform is a REAL cost
            // --- a production path would pay, so it is timed, not hidden.
            let t_xform = Instant::now();
            let mut xi = vec![0f32; COLS * k];
            for col in 0..COLS {
                for kk in 0..k {
                    xi[col * k + kk] = acts[kk][col];
                }
            }
            let interleave_ns = t_xform.elapsed().as_nanos() as u64;
            let xi_buf = ctx.new_buffer_with_bytes_checked(cast(&xi))?;
            let out_buf = ctx.new_buffer_checked(ROWS * k * 4)?;

            // ---------------- correctness ----------------
            arm_multiposition(&ctx, &kernel, r, &codes, &scales, &xi_buf, &out_buf, gpr)?;
            let got = read_f32(&out_buf, ROWS * k);

            let mut max_abs = 0f64;
            let mut max_rel = 0f64;
            for row in 0..ROWS {
                for kk in 0..k {
                    let a = got[row * k + kk] as f64;
                    let b = reference[kk][row] as f64;
                    let d = (a - b).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                    let denom = b.abs().max(1e-6);
                    if d / denom > max_rel {
                        max_rel = d / denom;
                    }
                }
            }
            let tol = 1e-4f64;
            let valid = max_rel <= tol;

            // NEGATIVE CONTROL: compare column kk against position kk+1's
            // answer. A check that has never been seen to fail is not evidence.
            let mut neg_rel = 0f64;
            let neg_applicable = k > 1;
            if neg_applicable {
                for row in 0..ROWS {
                    for kk in 0..k {
                        let a = got[row * k + kk] as f64;
                        let b = reference[(kk + 1) % k][row] as f64;
                        let denom = b.abs().max(1e-6);
                        let rel = (a - b).abs() / denom;
                        if rel > neg_rel {
                            neg_rel = rel;
                        }
                    }
                }
            }
            let negative_control_rejected = if neg_applicable {
                Value::Bool(neg_rel > tol)
            } else {
                Value::String("NOT_APPLICABLE_AT_K1".into())
            };

            // UNIFORM-POSITION CONTROL: every column identical -> every output
            // column must equal the K=1 answer. Catches a kernel that writes
            // one column K times.
            let xu: Vec<f32> = (0..COLS * k).map(|i| acts[0][i / k]).collect();
            let xu_buf = ctx.new_buffer_with_bytes_checked(cast(&xu))?;
            let ou_buf = ctx.new_buffer_checked(ROWS * k * 4)?;
            arm_multiposition(&ctx, &kernel, r, &codes, &scales, &xu_buf, &ou_buf, gpr)?;
            let gu = read_f32(&ou_buf, ROWS * k);
            let mut uni_rel = 0f64;
            for row in 0..ROWS {
                for kk in 0..k {
                    let a = gu[row * k + kk] as f64;
                    let b = reference[0][row] as f64;
                    let denom = b.abs().max(1e-6);
                    let rel = (a - b).abs() / denom;
                    if rel > uni_rel {
                        uni_rel = rel;
                    }
                }
            }
            let uniform_control_holds = uni_rel <= tol;

            // ---------------- timing ----------------
            let (mut seq_reps, mut mp_reps) = (Vec::new(), Vec::new());
            let (mut seq_host, mut mp_host) = (Vec::new(), Vec::new());
            if args.timing && valid {
                let seq_x: Vec<&metal::Buffer> = x_bufs.iter().take(k).collect();
                let seq_x: Vec<metal::Buffer> = seq_x.into_iter().cloned().collect();
                let seq_o: Vec<metal::Buffer> = (0..k)
                    .map(|_| ctx.new_buffer_checked(ROWS * 4))
                    .collect::<Result<_, _>>()?;
                // Alternate A B A B so drift hits both arms equally. The first
                // pair is warmup and is discarded.
                for rep in 0..(args.reps + 1) {
                    let a = arm_sequential(&ctx, &codes, &scales, &seq_x, &seq_o, gpr)?;
                    let b = arm_multiposition(
                        &ctx, &kernel, r, &codes, &scales, &xi_buf, &out_buf, gpr,
                    )?;
                    if rep == 0 {
                        continue;
                    }
                    seq_reps.push(a.gpu_ns);
                    seq_host.push(a.host_ns);
                    mp_reps.push(b.gpu_ns);
                    mp_host.push(b.host_ns);
                }
            }

            let code_moved = code_plane_bytes as f64;
            let act_moved = (COLS * k * 4) as f64;
            let out_moved = (ROWS * k * 4) as f64;
            let flops = 2.0 * (ROWS * COLS * k) as f64;

            let (seq_med, mp_med, speedup, ns_pos_seq, ns_pos_mp, code_gbs, tot_gbs, ai) =
                if !seq_reps.is_empty() {
                    let s = median(seq_reps.clone()) as f64;
                    let m = median(mp_reps.clone()) as f64;
                    let moved = code_moved + (scale_plane_bytes as f64) + act_moved + out_moved;
                    (
                        json!(s as u64),
                        json!(m as u64),
                        json!(s / m),
                        json!(s / k as f64),
                        json!(m / k as f64),
                        json!(code_moved / m),
                        json!(moved / m),
                        json!(flops / moved),
                    )
                } else {
                    let ni = || Value::String("NOT_INSTRUMENTED".into());
                    (ni(), ni(), ni(), ni(), ni(), ni(), ni(), json!(flops / (code_moved + act_moved + out_moved)))
                };

            rows_out.push(json!({
                "r": r,
                "k": k,
                "kernel": kernel,
                "status": if valid { "MEASURED" } else { "CORRECTNESS_FAILED" },
                "valid": valid,
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "tolerance": tol,
                "negative_control_rejected": negative_control_rejected,
                "negative_control_max_rel": if neg_applicable { json!(neg_rel) } else { Value::Null },
                "uniform_control_holds": uniform_control_holds,
                "uniform_control_max_rel": uni_rel,
                "dispatch_count_seq": k,
                "dispatch_count_mp": 1,
                "seq_gpu_ns_reps": seq_reps,
                "mp_gpu_ns_reps": mp_reps,
                "seq_host_ns_reps": seq_host,
                "mp_host_ns_reps": mp_host,
                "seq_gpu_ns_median": seq_med,
                "mp_gpu_ns_median": mp_med,
                "ns_per_position_seq": ns_pos_seq,
                "ns_per_position_mp": ns_pos_mp,
                "speedup_x": speedup,
                "code_bytes_moved": code_plane_bytes,
                "scale_bytes_moved": scale_plane_bytes,
                "activation_bytes_moved": act_moved as u64,
                "output_bytes": out_moved as u64,
                "activation_bytes_per_code_byte": act_moved / code_moved,
                "interleave_transform_ns": interleave_ns,
                "temporary_bytes": (COLS * k * 4 + ROWS * k * 4) as u64,
                "achieved_code_gb_s": code_gbs,
                "achieved_total_gb_s": tot_gbs,
                "flops": flops,
                "arithmetic_intensity_flops_per_byte": ai,
                "threadgroup_bytes": 16 * r * k,
                "accumulators_per_thread": r * k,
                "live_floats_per_thread": r * k + 2 * k + 2 * r,
            }));
            eprintln!(
                "r{r}k{k:<3} valid={valid} max_rel={max_rel:.3e} \
neg_rejected={negative_control_rejected} uniform={uniform_control_holds}"
            );
        }

        let doc = json!({
            "schema": SCHEMA,
            "organ": "deltanet_in_proj_qkvz",
            "organ_share_of_step_gpu_ns": 0.241,
            "organ_share_source": "receipts/headless/_ORGAN_BANDWIDTH_raw.json isolated_organs",
            "why_not_gate_up": "mlp_gate_up is the larger organ at 35.9%, but the live resident \
dispatches qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128 for it and NO affine-q2 \
multi-position kernel exists in any shader. The matmul_rRkK family is uniform-q4 only. Retargeting \
to the largest uniform-q4 organ is what makes a CP3 measurement possible at all.",
            "geometry": {
                "rows": ROWS, "cols": COLS, "group_size": GROUP,
                "groups_per_row": gpr,
                "code_plane_bytes": code_plane_bytes,
                "scale_plane_bytes": scale_plane_bytes,
            },
            "code_provenance": code_provenance,
            "baseline_kernel": BASELINE_KERNEL,
            "baseline_note": "K separate command buffers, one per position, which is what the \
per-token prefill loop at qwen38_hybrid_decode.rs:7566 physically does. The production DeltaNet \
path additionally FUSES qkv into one dispatch; that fusion is a different axis and is deliberately \
not folded into this comparison, which isolates batching alone.",
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
            "timing_enabled": args.timing,
            "sweep_order": if args.reverse { "REVERSED" } else { "forward" },
            "reps": if args.timing { json!(args.reps) } else { Value::String("NOT_INSTRUMENTED".into()) },
            "warmup_discarded": 1,
            "results": rows_out,
        });

        match args.out {
            Some(p) => {
                if let Some(d) = p.parent() {
                    fs::create_dir_all(d)?;
                }
                fs::write(&p, serde_json::to_string_pretty(&doc)? + "\n")?;
                eprintln!("wrote {}", p.display());
            }
            None => println!("{}", serde_json::to_string_pretty(&doc)?),
        }
        Ok(())
    }
}
