use std::time::Instant;

use strand_decode_kernel::block_walk::gate_proto::synth_encoded;
use strand_decode_kernel::gemv::decode_q12_fast;
use strand_decode_kernel::gemv_par::decode_q12_par;
use strand_decode_kernel::interleave::{decode_q12_interleave, decode_q12_interleave_par};
use strand_quant::TrellisConfig;

fn bench<F: FnMut() -> Vec<i32>>(label: &str, total: usize, reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        assert_eq!(out.len(), total);
        std::hint::black_box(&out);
        if dt < best {
            best = dt;
        }
    }
    let mws = total as f64 / best / 1e6;
    println!("  {label:<28} {:>9.1} Mw/s  ({:.1} ms)", mws, best * 1e3);
    mws
}

fn run_point(name: &str, cfg: &TrellisConfig, total: usize) {
    println!("\n== {name}: k={} L={} ({} states), {:.1}M weights ==", cfg.k_bits, cfg.l_bits, cfg.num_states(), total as f64 / 1e6);
    let enc = synth_encoded(total, cfg.k_bits, 256);

    let want = decode_q12_fast(&enc, cfg);
    assert_eq!(decode_q12_interleave::<2>(&enc, cfg), want, "S=2 diverged");
    assert_eq!(decode_q12_interleave::<4>(&enc, cfg), want, "S=4 diverged");
    assert_eq!(decode_q12_interleave::<6>(&enc, cfg), want, "S=6 diverged");
    assert_eq!(decode_q12_interleave::<8>(&enc, cfg), want, "S=8 diverged");
    assert_eq!(decode_q12_interleave::<16>(&enc, cfg), want, "S=16 diverged");
    assert_eq!(decode_q12_par(&enc, cfg), want, "par baseline diverged");
    assert_eq!(decode_q12_interleave_par::<4>(&enc, cfg), want, "par S=4 diverged");
    assert_eq!(decode_q12_interleave_par::<8>(&enc, cfg), want, "par S=8 diverged");
    println!("  determinism: 8/8 variants byte-identical ✓");

    let base = bench("single fast (baseline)", total, 3, || decode_q12_fast(&enc, cfg));
    let s2 = bench("interleave S=2", total, 3, || decode_q12_interleave::<2>(&enc, cfg));
    let s4 = bench("interleave S=4", total, 3, || decode_q12_interleave::<4>(&enc, cfg));
    let s6 = bench("interleave S=6", total, 3, || decode_q12_interleave::<6>(&enc, cfg));
    let s8 = bench("interleave S=8", total, 3, || decode_q12_interleave::<8>(&enc, cfg));
    let s16 = bench("interleave S=16", total, 3, || decode_q12_interleave::<16>(&enc, cfg));
    let best_s = [(2, s2), (4, s4), (6, s6), (8, s8), (16, s16)].into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    println!("  single-core verdict: S={} = {:.2}x over baseline", best_s.0, best_s.1 / base);

    let parb = bench("rayon par (baseline)", total, 3, || decode_q12_par(&enc, cfg));
    let p4 = bench("interleave_par S=4", total, 3, || decode_q12_interleave_par::<4>(&enc, cfg));
    let p8 = bench("interleave_par S=8", total, 3, || decode_q12_interleave_par::<8>(&enc, cfg));
    let bestp = p4.max(p8);
    println!("  all-core verdict: {:.2}x over rayon baseline ({:.2} Gw/s; bandwidth flip ≈ 156 Gw/s)", bestp / parb, bestp / 1e3);
}

fn main() {
    let total = 18944usize * 3584;
    println!("gate-interleave — G0: multi-stream scalar ILP decode");
    println!("(advisory if run during the marathon; definitive numbers re-run serially)");

    run_point("3-bit deploy", &TrellisConfig::for_bpw(3.0), total);
    run_point("2-bit reopen", &TrellisConfig::for_bpw_l(2.0, 12), total);
}
