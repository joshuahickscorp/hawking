
use std::time::Instant;

use rayon::prelude::*;
use strand_decode_kernel::block_walk::gate_proto::{machine_stamp, synth_encoded};
use strand_decode_kernel::fused::fused_gemm;
use strand_decode_kernel::gemv_par::decode_q12_par;
use strand_decode_kernel::histogram_gemv::{
    dot_q12_reference, histogram_dot_q12, histogram_gemm, histogram_gemm_scalar,
    histogram_matvec, histogram_stats,
};
use strand_quant::encode::{n_sub_blocks, EncodedTensor};
use strand_quant::TrellisConfig;

fn pack_codes(codes: &[u8]) -> Vec<u8> {
    let total_bits = 6 * codes.len();
    let mut bytes = vec![0u8; total_bits.div_ceil(8)];
    let mut cursor = 0usize;
    for &c in codes {
        for b in 0..6 {
            if (c >> b) & 1 == 1 {
                bytes[(cursor + b) >> 3] |= 1u8 << ((cursor + b) & 7);
            }
        }
        cursor += 6;
    }
    bytes
}

fn synth_varied_scales(total: usize, k: u32, block_len: usize) -> EncodedTensor {
    let mut enc = synth_encoded(total, k, block_len);
    let mut ctr = 0usize;
    for blk in &mut enc.blocks {
        let n_sub = n_sub_blocks(blk.n as usize);
        let codes: Vec<u8> = (0..n_sub)
            .map(|_| {
                let c = 16 + (ctr % 48) as u8; 
                ctr += 1;
                c
            })
            .collect();
        blk.sub_scales = pack_codes(&codes);
    }
    enc
}

fn synth_x(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.0713).cos()).collect()
}

fn synth_xq(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| {
            let h = (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xD1B5);
            ((h >> 40) as i32 & 0x1FFF) - 4096
        })
        .collect()
}

fn baseline_gemm(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    out_f: usize,
    in_f: usize,
    xs: &[f32],
    b: usize,
) -> Vec<f32> {
    let w = decode_q12_par(enc, cfg);
    let inv = 1.0f32 / 4096.0;
    let mut y = vec![0.0f32; out_f * b];
    y.par_chunks_mut(b).enumerate().for_each(|(o, yo)| {
        let row = &w[o * in_f..(o + 1) * in_f];
        for (bb, slot) in yo.iter_mut().enumerate() {
            let x = &xs[bb * in_f..(bb + 1) * in_f];
            let mut acc = 0.0f32;
            for i in 0..in_f {
                acc += (row[i] as f32) * inv * x[i];
            }
            *slot = acc;
        }
    });
    y
}

fn bench<F: FnMut() -> Vec<f32>>(label: &str, reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        if dt < best {
            best = dt;
        }
    }
    println!("  {label:<42} {:>8.1} ms", best * 1e3);
    best
}

#[allow(clippy::too_many_arguments)]
fn run_variant(
    vname: &str,
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    out_f: usize,
    in_f: usize,
    xs: &[f32],
    xq: &[i32],
    bench_perf: bool,
) {
    let total = out_f * in_f;

    let t0 = Instant::now();
    let y_h = histogram_dot_q12(enc, cfg, None, out_f, in_f, xq);
    let t_hist_int = t0.elapsed().as_secs_f64();
    let y_r = dot_q12_reference(enc, cfg, None, out_f, in_f, xq);
    assert_eq!(
        y_h, y_r,
        "[{vname}] INTEGER HISTOGRAM != REFERENCE DOT — the order-free regroup is broken"
    );
    println!(
        "  [{vname}] integer histogram dot: BIT-EQUAL to reference i64 dot over {} rows ✓ \
         (serial walk {:.0} ms — exactness witness, not a perf path)",
        out_f,
        t_hist_int * 1e3
    );

    const ORDER_SLACK: f32 = 1e-3;
    const ABS_SLACK: f32 = 1e-4;
    let x1 = &xs[..in_f];
    let y_hm = histogram_matvec(enc, cfg, None, out_f, in_f, x1);
    let y_f1 = fused_gemm(enc, cfg, None, out_f, in_f, x1, 1);
    let wq = decode_q12_par(enc, cfg);
    let inv = 1.0f32 / 4096.0;
    let mut worst_frac = 0.0f64;
    for o in 0..out_f {
        let row = &wq[o * in_f..(o + 1) * in_f];
        let mut absum = 0.0f32;
        for i in 0..in_f {
            absum += ((row[i] as f32) * inv * x1[i]).abs();
        }
        let bound = ORDER_SLACK * absum + ABS_SLACK;
        let err = (y_hm[o] - y_f1[o]).abs();
        assert!(
            err <= bound,
            "[{vname}] float histogram beyond the order bound at row {o}: err {err:.3e} > bound {bound:.3e}"
        );
        let frac = (err / bound) as f64;
        if frac > worst_frac {
            worst_frac = frac;
        }
    }
    println!(
        "  [{vname}] float vs fused (B=1): within the principled reorder bound \
         (worst err = {:.1}% of 1e-3·Σ|w·x| budget) ✓ — values identical, add tree differs (documented caveat)",
        worst_frac * 100.0
    );
    for &b in &[4usize, 16, 64] {
        let xb = &xs[..b * in_f];
        let y_n = histogram_gemm(enc, cfg, None, out_f, in_f, xb, b);
        let y_s = histogram_gemm_scalar(enc, cfg, None, out_f, in_f, xb, b);
        for (i, (a, s)) in y_n.iter().zip(y_s.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                s.to_bits(),
                "[{vname}] NEON bucket-add != scalar histogram at flat index {i} (B={b})"
            );
        }
    }
    
    let yb4 = histogram_gemm(enc, cfg, None, out_f, in_f, &xs[..4 * in_f], 4);
    for b in 0..4 {
        let y1 = histogram_matvec(enc, cfg, None, out_f, in_f, &xs[b * in_f..(b + 1) * in_f]);
        for o in (0..out_f).step_by(97) {
            assert_eq!(
                yb4[o * 4 + b].to_bits(),
                y1[o].to_bits(),
                "[{vname}] gemm col {b} row {o} != matvec"
            );
        }
    }
    println!(
        "  [{vname}] NEON==scalar bit-equal at B∈{{4,16,64}}; columns bit-equal to matvec ✓"
    );

    let s = histogram_stats(enc, cfg, in_f);
    println!(
        "  [{vname}] stats: {} regions | avg region {:.1} weights | avg occupied {:.1} levels \
         | mul ratio {:.3} ({} muls/col vs {} direct)",
        s.regions,
        s.avg_region_len(),
        s.occupied as f64 / s.regions.max(1) as f64,
        s.mul_ratio(),
        s.occupied,
        s.weights
    );

    if !bench_perf {
        return;
    }
    println!("  [{vname}] ADVISORY bench (contended-box numbers; identity is the deliverable):");
    let mut lines: Vec<(usize, f64, f64, f64, f64)> = Vec::new();
    for &b in &[1usize, 16, 64] {
        let xb = &xs[..b * in_f];
        let t_bb = bench(&format!("[{vname}] baseline two-pass (B={b})"), 3, || {
            baseline_gemm(enc, cfg, out_f, in_f, xb, b)
        });
        let t_f = bench(&format!("[{vname}] fused_gemm (B={b})"), 3, || {
            fused_gemm(enc, cfg, None, out_f, in_f, xb, b)
        });
        let t_hs = bench(&format!("[{vname}] histogram scalar (B={b})"), 3, || {
            histogram_gemm_scalar(enc, cfg, None, out_f, in_f, xb, b)
        });
        let t_hn = bench(&format!("[{vname}] histogram NEON (B={b})"), 3, || {
            histogram_gemm(enc, cfg, None, out_f, in_f, xb, b)
        });
        lines.push((b, t_bb, t_f, t_hs, t_hn));
    }
    for (b, t_bb, t_f, t_hs, t_hn) in &lines {
        println!(
            "  B={b:<2}: hist-NEON {:.1} ms | hist-scalar {:.1} ms | vs fused {:.2}x | vs baseline {:.2}x | {:.2} GMAC/s",
            t_hn * 1e3,
            t_hs * 1e3,
            t_f / t_hn,
            t_bb / t_hn,
            (total * b) as f64 / t_hn / 1e9
        );
    }
}

fn run_point(name: &str, cfg: &TrellisConfig, out_f: usize, in_f: usize) {
    let total = out_f * in_f;
    println!(
        "\n== {name}: k={} L={} ({} states), {out_f}x{in_f} = {:.1}M weights ==",
        cfg.k_bits,
        cfg.l_bits,
        cfg.num_states(),
        total as f64 / 1e6
    );
    let max_b = 64usize;
    let mut xs = Vec::with_capacity(max_b * in_f);
    for b in 0..max_b {
        xs.extend(synth_x(in_f, b as f32 * 11.3 + 0.5));
    }
    let xq = synth_xq(in_f);

    let enc_unit = synth_encoded(total, cfg.k_bits, 256);
    run_variant("unit-scale", &enc_unit, cfg, out_f, in_f, &xs, &xq, true);

    let enc_var = synth_varied_scales(total, cfg.k_bits, 256);
    run_variant("varied-scale", &enc_var, cfg, out_f, in_f, &xs, &xq, true);
}

fn strand_delta_alive() -> bool {
    std::process::Command::new("pgrep")
        .args(["-f", "strand-delta"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn main() {
    let no_wait = std::env::args().any(|a| a == "--no-wait");
    if !no_wait {
        while strand_delta_alive() {
            println!("strand-delta measurement job is running — waiting 60 s (pass --no-wait to override)…");
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
    }

    println!(
        "gate-histogram — DENDRITIC SUMMATION: histogram-accumulate GEMV \
         (integer path EXACT/order-free; float path order-caveat vs fused)"
    );
    println!("{}", machine_stamp());

    let (out_f, in_f) = (18944usize, 3584usize);
    run_point("3-bit deploy (THE PRIZE)", &TrellisConfig::for_bpw(3.0), out_f, in_f);
    run_point("2-bit reopen (expected DEAD here)", &TrellisConfig::for_bpw_l(2.0, 12), out_f, in_f);
}
