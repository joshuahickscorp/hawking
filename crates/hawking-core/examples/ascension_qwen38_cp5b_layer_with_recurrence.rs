//! CP5b — a DeltaNet layer shape, with the REAL recurrence serial inside the
//! batched command buffer.
//!
//! CP5a showed the three weight-heavy organs compose at 2.458x when they share
//! one command buffer. It deliberately left out the part that cannot be batched:
//! the recurrent state update, which is 9.8% of the step
//! (`receipts/runtime/CP5_DECOMPOSITION.md`). The open question it names is
//! whether a chunk of K positions survives CROSSING A LAYER with the recurrence
//! still stepping one position at a time, or whether the recurrence
//! re-serialises the projections around it.
//!
//! Structurally it should not: the recurrence consumes K projected positions and
//! produces K outputs, all available before the next layer's projections begin.
//! This measures that instead of asserting it.
//!
//!   SEQUENTIAL  K command buffers, each = in_proj -> recurrence -> gate_up ->
//!               down for ONE position. 4 dispatches x K.
//!   BATCHED     ONE command buffer = batched in_proj -> K serial recurrence
//!               dispatches -> batched gate_up -> batched down. 3 + K dispatches.
//!
//! The recurrence is `qwen38_gated_delta_decode_vi_simd`, the real kernel, at the
//! real geometry (48 value heads, key_dim 128, value_dim 128) and the production
//! launch shape (kd, heads, vd) / (kd, 1, 1) with 512 bytes of threadgroup
//! scratch, read off `encode_gated_delta_fused_ba` in the runtime.
//!
//! The state buffer is MUTATED by the recurrence and the update is order
//! dependent, so both arms reset it to the same bytes before they run and both
//! step the same K positions in the same order. That is what makes the outputs
//! comparable at all.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core \
//!     --example ascension_qwen38_cp5b_layer_with_recurrence
//! Timing (REQUIRES zero loaded 27B residents, checked by RSS):
//!   ./tools/gpu_lane_lock.sh cp5b .../ascension_qwen38_cp5b_layer_with_recurrence \
//!     --timing --reps 9 --out receipts/runtime/CP5B_LAYER_WITH_RECURRENCE.json
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

    const SCHEMA: &str = "hawking.cp5b.layer_with_recurrence.v1";

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

    // Real DeltaNet geometry: QWEN38_LINEAR_VALUE_HEADS / KEY_HEAD_DIM / VALUE_HEAD_DIM.
    const HEADS: usize = 48;
    const KD: usize = 128;
    const VD: usize = 128;
    const REC: &str = "qwen38_gated_delta_decode_vi_simd";
    const REC_TG_BYTES: u64 = 512;

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

    /// The recurrence's own buffers. `state` is MUTATED by the kernel and the
    /// update is order dependent, so it is reset from `state_seed` before each
    /// arm; without that the second arm would start from the first arm's state
    /// and the comparison would be meaningless.
    struct Rec {
        state: metal::Buffer,
        state_seed: Vec<u8>,
        q: Vec<metal::Buffer>,
        k: Vec<metal::Buffer>,
        v: Vec<metal::Buffer>,
        decay: metal::Buffer,
        beta: metal::Buffer,
        out_seq: Vec<metal::Buffer>,
        out_bat: Vec<metal::Buffer>,
    }

    fn dispatch_rec(
        tcb: &mut TokenCommandBuffer<'_>,
        r: &Rec,
        k: usize,
        out: &metal::Buffer,
    ) -> Result<(), Box<dyn Error>> {
        tcb.dispatch_threads(
            REC,
            (KD as u32, HEADS as u32, VD as u32),
            (KD as u32, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&r.state), 0);
                enc.set_buffer(1, Some(&r.q[k]), 0);
                enc.set_buffer(2, Some(&r.k[k]), 0);
                enc.set_buffer(3, Some(&r.v[k]), 0);
                enc.set_buffer(4, Some(&r.decay), 0);
                enc.set_buffer(5, Some(&r.beta), 0);
                enc.set_buffer(6, Some(out), 0);
                u32b(enc, 7, HEADS as u32);
                u32b(enc, 8, KD as u32);
                u32b(enc, 9, VD as u32);
                enc.set_threadgroup_memory_length(0, REC_TG_BYTES);
            },
        )?;
        Ok(())
    }

    fn reset_state(r: &Rec) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                r.state_seed.as_ptr(),
                r.state.contents() as *mut u8,
                r.state_seed.len(),
            );
        }
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
    #[allow(clippy::too_many_arguments)]
    fn seq_one_position(
        ctx: &MetalContext, w: &W, r: &Rec, k: usize,
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
        dispatch_rec(&mut tcb, r, k, &r.out_seq[k])?;
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
    #[allow(clippy::too_many_arguments)]
    fn batched_all_positions(
        ctx: &MetalContext, w: &W, r: &Rec,
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
        // The part that cannot be batched, still one dispatch per position, but
        // now INSIDE the same command buffer as the batched projections.
        for k in 0..K {
            dispatch_rec(&mut tcb, r, k, &r.out_bat[k])?;
        }
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
        // --- the recurrence's buffers, at the real DeltaNet geometry
        let f32b = |v: &[f32]| -> Result<metal::Buffer, Box<dyn Error>> {
            Ok(ctx.new_buffer_with_bytes_checked(cast(v))?)
        };
        let seed: Vec<f32> = (0..HEADS * KD * VD)
            .map(|i| ((i % 97) as f32 - 48.0) * 1e-4)
            .collect();
        let state_seed = cast(&seed).to_vec();
        let rec = Rec {
            state: f32b(&seed)?,
            state_seed,
            q: (0..K).map(|t| f32b(&(0..HEADS * KD)
                .map(|i| ((i % 31) as f32 - 15.0) * (1.0 + t as f32) * 1e-3).collect::<Vec<_>>()))
                .collect::<Result<_, _>>()?,
            k: (0..K).map(|t| f32b(&(0..HEADS * KD)
                .map(|i| ((i % 29) as f32 - 14.0) * (1.0 + t as f32) * 1e-3).collect::<Vec<_>>()))
                .collect::<Result<_, _>>()?,
            v: (0..K).map(|t| f32b(&(0..HEADS * VD)
                .map(|i| ((i % 23) as f32 - 11.0) * (1.0 + t as f32) * 1e-3).collect::<Vec<_>>()))
                .collect::<Result<_, _>>()?,
            // Decay in (0,1) and beta small, which is the regime the gated delta
            // rule actually runs in; a decay of 1.0 would make the state a plain
            // sum and hide an ordering bug.
            decay: f32b(&(0..HEADS).map(|i| 0.90 + (i % 7) as f32 * 0.01).collect::<Vec<_>>())?,
            beta: f32b(&(0..HEADS).map(|i| 0.05 + (i % 5) as f32 * 0.01).collect::<Vec<_>>())?,
            out_seq: (0..K).map(|_| ctx.new_buffer_checked(HEADS * VD * 4)).collect::<Result<_, _>>()?,
            out_bat: (0..K).map(|_| ctx.new_buffer_checked(HEADS * VD * 4)).collect::<Result<_, _>>()?,
        };

        let oi = ctx.new_buffer_checked(IN_ROWS * K * 4)?;
        let og = ctx.new_buffer_checked(GU_ROWS * K * 4)?;
        let ou = ctx.new_buffer_checked(GU_ROWS * K * 4)?;
        let od = ctx.new_buffer_checked(DN_ROWS * K * 4)?;

        // ---- correctness: every position of every organ, against the sequential arm
        let mut seq_d = 0u64;
        reset_state(&rec);
        for k in 0..K {
            let (_, d) = seq_one_position(&ctx, &w, &rec, k, &x_in[k], &x_gu[k], &x_dn[k], &o)?;
            seq_d += d;
        }
        reset_state(&rec);
        let (_, bat_d) = batched_all_positions(&ctx, &w, &rec, &xi_in, &xi_gu, &xi_dn, &oi, &og, &ou, &od)?;
        let refs = [
            ("in_proj", &o.inp, &oi, IN_ROWS),
            ("gate", &o.gate, &og, GU_ROWS),
            ("up", &o.up, &ou, GU_ROWS),
            ("down", &o.down, &od, DN_ROWS),
        ];
        let mut worst = 0f64;
        let mut per_organ = Vec::new();
        // The recurrence output, position by position. If the serial section
        // inside the batched CB stepped the state in a different order, or saw a
        // stale state, this is where it shows.
        {
            let mut m = 0f64;
            for k in 0..K {
                let a = read_f32(&rec.out_seq[k], HEADS * VD);
                let b = read_f32(&rec.out_bat[k], HEADS * VD);
                for i in 0..a.len() {
                    let d = (a[i] as f64 - b[i] as f64).abs();
                    let rel = d / (a[i] as f64).abs().max(1e-6);
                    if rel > m { m = rel; }
                }
            }
            if m > worst { worst = m; }
            per_organ.push(json!({"organ": "recurrence", "max_rel_err": m}));
            eprintln!("{:<8} max_rel={m:.3e}", "recurrence");
        }
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
                reset_state(&rec);
                for k in 0..K {
                    s += seq_one_position(&ctx, &w, &rec, k, &x_in[k], &x_gu[k], &x_dn[k], &o)?.0;
                }
                reset_state(&rec);
                let b = batched_all_positions(&ctx, &w, &rec, &xi_in, &xi_gu, &xi_dn, &oi, &og, &ou, &od)?.0;
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
            "includes_the_recurrence": {
                "kernel": REC,
                "geometry": {"heads": HEADS, "key_dim": KD, "value_dim": VD},
                "launch": "(kd, heads, vd) / (kd,1,1), 512 B threadgroup -- read off the runtime's encode_gated_delta_fused_ba",
                "note": "one dispatch per position in BOTH arms; in the batched arm those K dispatches sit INSIDE the same command buffer as the batched projections. The state buffer is reset from the same seed before each arm because the update is order dependent.",
            },
            "still_not_a_layer": "norms, causal conv, the gated RMSNorm and out_proj are absent, as is attention. This is the DeltaNet layer's weight-heavy organs plus its real recurrence, which is what the layer-crossing question turns on.",
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
