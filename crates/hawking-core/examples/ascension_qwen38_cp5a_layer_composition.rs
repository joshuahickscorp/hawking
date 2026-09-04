//! CP5a — do the per-organ wins COMPOSE when the organs share one command buffer?
//!
//! CP3 (2.474x, uniform-q4 in_proj), CP4 (2.49x, affine gate_up) and CP4b
//! (2.27x, affine down_proj) each measured ONE organ alone, each dispatch on its
//! own command buffer. `receipts/runtime/CP5_DECOMPOSITION.md` names the
//! assumption that projection rests on and does not prove:
//!
//!   "that per-organ speedups compose. Organs measured alone do not share a
//!    command buffer, and production runs one CB per token."
//!
//! This is the cheapest measurement that can falsify it. Both arms use a
//! TokenCommandBuffer, which is the production shape:
//!
//!   SEQUENTIAL  K command buffers, each holding the three organ dispatches for
//!               ONE position -- exactly what the per-token prefill loop does.
//!   BATCHED     ONE command buffer holding three multi-position dispatches that
//!               together cover all K positions.
//!
//! Three weight-heavy organs at their REAL shapes and one layer's worth of each:
//!   in_proj   uniform q4    rows 16384  cols  5120
//!   gate_up   affine q2     rows 17408  cols  5120   (pair: gate and up)
//!   down      affine q2     rows  5120  cols 17408
//!
//! NOT a layer. The norms, causal conv, recurrent state and attention are absent,
//! and the recurrence is the part that cannot be batched at all (9.8% of the
//! step). This measures composition of the batchable weight-heavy organs, which
//! is the specific claim in question, and nothing more.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core \
//!     --example ascension_qwen38_cp5a_layer_composition
//! Timing (REQUIRES zero loaded 27B residents, checked by RSS):
//!   ./tools/gpu_lane_lock.sh cp5a .../ascension_qwen38_cp5a_layer_composition \
//!     --timing --reps 9 --out receipts/runtime/CP5A_COMPOSITION.json

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
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    const SCHEMA: &str = "hawking.cp5a.layer_composition.v1";

    // in_proj, uniform q4 group 64
    const IN_ROWS: usize = 16_384;
    const IN_COLS: usize = 5_120;
    const IN_CODE_BYTES: usize = IN_ROWS * (IN_COLS / 64) * 32;
    const IN_SCALE_HALVES: usize = IN_ROWS * (IN_COLS / 64);
    // gate_up and down, affine q2 group 64
    const GU_ROWS: usize = 17_408;
    const GU_COLS: usize = 5_120;
    const DN_ROWS: usize = 5_120;
    const DN_COLS: usize = 17_408;
    const AFF_CODE_BYTES: usize = 22_282_240; // same element count either way
    const AFF_HALVES: usize = 1_392_640;

    const R: usize = 4;
    const K: usize = 4; // the peak cell in all three of CP3, CP4 and CP4b

    const SEQ_INPROJ: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128";
    const SEQ_GATEUP: &str = "qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128";
    const SEQ_DOWN: &str = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128";
    const MP_INPROJ: &str = "qwen_uniform_q4_group64_matmul_r4k4_geo_tpr64_tg128";
    const MP_GATEUP: &str = "qwen_affine_q2_group64_matmul_gate_up_r4k4_geo_tpr64_tg128";
    const MP_DOWN: &str = "qwen_affine_q2_matmul_r4k4_geo_tpr64_tg128";

    fn deterministic(n: usize, seed: u32) -> Vec<u8> {
        let mut v = vec![0u8; n];
        let mut s = seed;
        for b in v.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (s >> 24) as u8;
        }
        v
    }

    fn half_bits(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = ((bits >> 13) & 0x3ff) as u16;
        if exp <= 0 { sign } else if exp >= 31 { sign | 0x7bff }
        else { sign | ((exp as u16) << 10) | mant }
    }

    fn halves(n: usize, base: f32, step: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            out.extend_from_slice(&half_bits(base + (i % 13) as f32 * step).to_le_bytes());
        }
        out
    }

    fn cast(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }
    fn read_f32(b: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() }
    }
    fn median(mut v: Vec<u64>) -> u64 { v.sort_unstable(); v[v.len() / 2] }

    struct W {
        in_codes: metal::Buffer, in_scales: metal::Buffer,
        gc: metal::Buffer, gs: metal::Buffer, gb: metal::Buffer,
        uc: metal::Buffer, us: metal::Buffer, ub: metal::Buffer,
        dc: metal::Buffer, ds: metal::Buffer, db: metal::Buffer,
    }

    struct Out {
        inp: Vec<metal::Buffer>, gate: Vec<metal::Buffer>,
        up: Vec<metal::Buffer>, down: Vec<metal::Buffer>,
    }

    fn u32b(enc: &metal::ComputeCommandEncoderRef, idx: u64, v: u32) {
        enc.set_bytes(idx, 4, &v as *const u32 as *const _);
    }

    /// One command buffer holding the three organs for ONE position -- the shape
    /// the per-token prefill loop actually runs.
    fn seq_one_position(
        ctx: &MetalContext, w: &W, k: usize,
        x_in: &metal::Buffer, x_gu: &metal::Buffer, x_dn: &metal::Buffer, o: &Out,
    ) -> Result<(u64, u64), Box<dyn Error>> {
        let mut tcb = TokenCommandBuffer::new(ctx);
        let tg = (128u32, 1, 1);
        tcb.dispatch_threads(SEQ_INPROJ, ((((IN_ROWS as u32) + 1) / 2) * 128, 1, 1), tg, |enc| {
            enc.set_buffer(0, Some(&w.in_codes), 0); enc.set_buffer(1, Some(&w.in_scales), 0);
            enc.set_buffer(2, Some(x_in), 0); enc.set_buffer(3, Some(&o.inp[k]), 0);
            u32b(enc, 4, IN_ROWS as u32); u32b(enc, 5, IN_COLS as u32);
            u32b(enc, 6, (IN_COLS / 64) as u32);
        })?;
        tcb.dispatch_threads(SEQ_GATEUP, ((((GU_ROWS as u32) + 1) / 2) * 128, 1, 1), tg, |enc| {
            enc.set_buffer(0, Some(&w.gc), 0); enc.set_buffer(1, Some(&w.gs), 0);
            enc.set_buffer(2, Some(&w.gb), 0); enc.set_buffer(3, Some(&w.uc), 0);
            enc.set_buffer(4, Some(&w.us), 0); enc.set_buffer(5, Some(&w.ub), 0);
            enc.set_buffer(6, Some(x_gu), 0);
            enc.set_buffer(7, Some(&o.gate[k]), 0); enc.set_buffer(8, Some(&o.up[k]), 0);
            u32b(enc, 9, GU_ROWS as u32); u32b(enc, 10, GU_COLS as u32);
        })?;
        tcb.dispatch_threads(SEQ_DOWN, ((((DN_ROWS as u32) + 1) / 2) * 128, 1, 1), tg, |enc| {
            enc.set_buffer(0, Some(&w.dc), 0); enc.set_buffer(1, Some(&w.ds), 0);
            enc.set_buffer(2, Some(&w.db), 0); enc.set_buffer(3, Some(x_dn), 0);
            enc.set_buffer(4, Some(&o.down[k]), 0);
            u32b(enc, 5, DN_ROWS as u32); u32b(enc, 6, DN_COLS as u32); u32b(enc, 7, 64);
        })?;
        let d = tcb.dispatch_count() as u64;
        let t = tcb.commit_and_wait_timed()?;
        Ok((t.gpu_ns.ok_or("no GPU timestamp on the sequential CB")?, d))
    }

    /// ONE command buffer covering all K positions of all three organs.
    fn batched_all_positions(
        ctx: &MetalContext, w: &W,
        xi_in: &metal::Buffer, xi_gu: &metal::Buffer, xi_dn: &metal::Buffer,
        oi: &metal::Buffer, og: &metal::Buffer, ou: &metal::Buffer, od: &metal::Buffer,
    ) -> Result<(u64, u64), Box<dyn Error>> {
        let mut tcb = TokenCommandBuffer::new(ctx);
        let tg = (128u32, 1, 1);
        let g = |rows: usize| (((rows + 2 * R - 1) / (2 * R) * 128) as u32, 1u32, 1u32);
        tcb.dispatch_threads(MP_INPROJ, g(IN_ROWS), tg, |enc| {
            enc.set_buffer(0, Some(&w.in_codes), 0); enc.set_buffer(1, Some(&w.in_scales), 0);
            enc.set_buffer(2, Some(xi_in), 0); enc.set_buffer(3, Some(oi), 0);
            u32b(enc, 4, IN_ROWS as u32); u32b(enc, 5, IN_COLS as u32);
            u32b(enc, 6, (IN_COLS / 64) as u32);
        })?;
        tcb.dispatch_threads(MP_GATEUP, g(GU_ROWS), tg, |enc| {
            enc.set_buffer(0, Some(&w.gc), 0); enc.set_buffer(1, Some(&w.gs), 0);
            enc.set_buffer(2, Some(&w.gb), 0); enc.set_buffer(3, Some(&w.uc), 0);
            enc.set_buffer(4, Some(&w.us), 0); enc.set_buffer(5, Some(&w.ub), 0);
            enc.set_buffer(6, Some(xi_gu), 0);
            enc.set_buffer(7, Some(og), 0); enc.set_buffer(8, Some(ou), 0);
            u32b(enc, 9, GU_ROWS as u32); u32b(enc, 10, GU_COLS as u32);
        })?;
        tcb.dispatch_threads(MP_DOWN, g(DN_ROWS), tg, |enc| {
            enc.set_buffer(0, Some(&w.dc), 0); enc.set_buffer(1, Some(&w.ds), 0);
            enc.set_buffer(2, Some(&w.db), 0); enc.set_buffer(3, Some(xi_dn), 0);
            enc.set_buffer(4, Some(od), 0);
            u32b(enc, 5, DN_ROWS as u32); u32b(enc, 6, DN_COLS as u32); u32b(enc, 7, 64);
        })?;
        let d = tcb.dispatch_count() as u64;
        let t = tcb.commit_and_wait_timed()?;
        Ok((t.gpu_ns.ok_or("no GPU timestamp on the batched CB")?, d))
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut out_path: Option<PathBuf> = None;
        let mut reps = 7usize;
        let mut timing = false;
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--out" => out_path = it.next().map(PathBuf::from),
                "--reps" => reps = it.next().unwrap_or_default().parse().unwrap_or(7),
                "--timing" => timing = true,
                o => panic!("unknown flag {o}"),
            }
        }
        let ctx = MetalContext::new()?;
        let w = W {
            in_codes: ctx.new_buffer_with_bytes_checked(&deterministic(IN_CODE_BYTES, 0x9e37_79b9))?,
            in_scales: ctx.new_buffer_with_bytes_checked(&halves(IN_SCALE_HALVES, 0.0015, 0.0001))?,
            gc: ctx.new_buffer_with_bytes_checked(&deterministic(AFF_CODE_BYTES, 0x1234_5678))?,
            gs: ctx.new_buffer_with_bytes_checked(&halves(AFF_HALVES, 0.0020, 0.00011))?,
            gb: ctx.new_buffer_with_bytes_checked(&halves(AFF_HALVES, -0.0030, 0.00007))?,
            uc: ctx.new_buffer_with_bytes_checked(&deterministic(AFF_CODE_BYTES, 0x2345_6789))?,
            us: ctx.new_buffer_with_bytes_checked(&halves(AFF_HALVES, 0.0017, 0.00013))?,
            ub: ctx.new_buffer_with_bytes_checked(&halves(AFF_HALVES, -0.0021, 0.00009))?,
            dc: ctx.new_buffer_with_bytes_checked(&deterministic(AFF_CODE_BYTES, 0x3456_789a))?,
            ds: ctx.new_buffer_with_bytes_checked(&halves(AFF_HALVES, 0.0019, 0.00012))?,
            db: ctx.new_buffer_with_bytes_checked(&halves(AFF_HALVES, -0.0025, 0.00008))?,
        };

        // Per-position activations, and the interleaved form the batched kernels want.
        let act = |cols: usize, t: usize| -> Vec<f32> {
            (0..cols).map(|c| ((c % 29) as f32 - 14.0) * (1.0 + t as f32) * 0.0625).collect()
        };
        let mk = |cols: usize| -> Result<(Vec<metal::Buffer>, metal::Buffer), Box<dyn Error>> {
            let per: Vec<Vec<f32>> = (0..K).map(|t| act(cols, t)).collect();
            let bufs: Vec<metal::Buffer> = per.iter()
                .map(|a| ctx.new_buffer_with_bytes_checked(cast(a)))
                .collect::<Result<_, _>>()?;
            let mut il = vec![0f32; cols * K];
            for c in 0..cols { for k in 0..K { il[c * K + k] = per[k][c]; } }
            Ok((bufs, ctx.new_buffer_with_bytes_checked(cast(&il))?))
        };
        let (x_in, xi_in) = mk(IN_COLS)?;
        let (x_gu, xi_gu) = mk(GU_COLS)?;
        let (x_dn, xi_dn) = mk(DN_COLS)?;

        let o = Out {
            inp: (0..K).map(|_| ctx.new_buffer_checked(IN_ROWS * 4)).collect::<Result<_, _>>()?,
            gate: (0..K).map(|_| ctx.new_buffer_checked(GU_ROWS * 4)).collect::<Result<_, _>>()?,
            up: (0..K).map(|_| ctx.new_buffer_checked(GU_ROWS * 4)).collect::<Result<_, _>>()?,
            down: (0..K).map(|_| ctx.new_buffer_checked(DN_ROWS * 4)).collect::<Result<_, _>>()?,
        };
        let oi = ctx.new_buffer_checked(IN_ROWS * K * 4)?;
        let og = ctx.new_buffer_checked(GU_ROWS * K * 4)?;
        let ou = ctx.new_buffer_checked(GU_ROWS * K * 4)?;
        let od = ctx.new_buffer_checked(DN_ROWS * K * 4)?;

        // ---- correctness: every position of every organ, against the sequential arm
        let mut seq_d = 0u64;
        for k in 0..K {
            let (_, d) = seq_one_position(&ctx, &w, k, &x_in[k], &x_gu[k], &x_dn[k], &o)?;
            seq_d += d;
        }
        let (_, bat_d) = batched_all_positions(&ctx, &w, &xi_in, &xi_gu, &xi_dn, &oi, &og, &ou, &od)?;
        let refs = [
            ("in_proj", &o.inp, &oi, IN_ROWS),
            ("gate", &o.gate, &og, GU_ROWS),
            ("up", &o.up, &ou, GU_ROWS),
            ("down", &o.down, &od, DN_ROWS),
        ];
        let mut worst = 0f64;
        let mut per_organ = Vec::new();
        for (name, seq, bat, rows) in refs {
            let b = read_f32(bat, rows * K);
            let mut m = 0f64;
            for k in 0..K {
                let s = read_f32(&seq[k], rows);
                for row in 0..rows {
                    let d = (b[row * K + k] as f64 - s[row] as f64).abs();
                    let rel = d / (s[row] as f64).abs().max(1e-6);
                    if rel > m { m = rel; }
                }
            }
            if m > worst { worst = m; }
            per_organ.push(json!({"organ": name, "max_rel_err": m}));
            eprintln!("{name:<8} max_rel={m:.3e}");
        }
        let valid = worst <= 1e-4;

        // ---- timing, alternated, first rep discarded
        let (mut sq, mut bq) = (Vec::new(), Vec::new());
        if timing && valid {
            for rep in 0..(reps + 1) {
                let mut s = 0u64;
                for k in 0..K {
                    s += seq_one_position(&ctx, &w, k, &x_in[k], &x_gu[k], &x_dn[k], &o)?.0;
                }
                let b = batched_all_positions(&ctx, &w, &xi_in, &xi_gu, &xi_dn, &oi, &og, &ou, &od)?.0;
                if rep == 0 { continue; }
                sq.push(s); bq.push(b);
            }
        }
        let ni = || Value::String("NOT_INSTRUMENTED".into());
        let (sm, bm, sp) = if sq.is_empty() { (ni(), ni(), ni()) } else {
            let s = median(sq.clone()) as f64; let b = median(bq.clone()) as f64;
            (json!(s as u64), json!(b as u64), json!(s / b))
        };

        let doc = json!({
            "schema": SCHEMA,
            "question": "Do the per-organ multi-position wins COMPOSE when the organs share ONE \
command buffer, which is the shape production runs?",
            "arms": {
                "sequential": "K TokenCommandBuffers, each holding the three organ dispatches for one \
position -- what the per-token prefill loop at qwen38_hybrid_decode.rs:7566 does",
                "batched": "ONE TokenCommandBuffer holding three multi-position dispatches covering \
all K positions",
            },
            "not_a_layer": "norms, causal conv, recurrent state and attention are absent. The \
recurrence is 9.8% of the step and cannot be batched at all; this measures composition of the \
batchable weight-heavy organs, which is the claim in question, and nothing more.",
            "R": R, "K": K,
            "organs": {
                "in_proj": {"kernel_seq": SEQ_INPROJ, "kernel_mp": MP_INPROJ, "rows": IN_ROWS, "cols": IN_COLS},
                "gate_up": {"kernel_seq": SEQ_GATEUP, "kernel_mp": MP_GATEUP, "rows": GU_ROWS, "cols": GU_COLS},
                "down":    {"kernel_seq": SEQ_DOWN,   "kernel_mp": MP_DOWN,   "rows": DN_ROWS, "cols": DN_COLS},
            },
            "gpu_timestamp_authority": "TokenCommandBuffer::commit_and_wait_timed, GPUEndTime-GPUStartTime on the completed command buffer",
            "correctness": {"max_rel_err": worst, "tolerance": 1e-4, "valid": valid, "per_organ": per_organ},
            "dispatches_sequential": seq_d,
            "dispatches_batched": bat_d,
            "command_buffers_sequential": K,
            "command_buffers_batched": 1,
            "seq_gpu_ns_reps": sq, "batched_gpu_ns_reps": bq,
            "seq_gpu_ns_median": sm, "batched_gpu_ns_median": bm,
            "speedup_x": sp,
            "reps": if timing { json!(reps) } else { ni() },
            "warmup_discarded": 1,
        });
        match out_path {
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
