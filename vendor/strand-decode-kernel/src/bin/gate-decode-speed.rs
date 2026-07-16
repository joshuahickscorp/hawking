use std::time::Instant;

use strand_decode_kernel::block_walk::gate_proto::synth_encoded;
use strand_decode_kernel::gemv::decode_q12_fast;
use strand_decode_kernel::gemv_par::{decode_q12_par, decode_q12_simd};
use strand_quant::TrellisConfig;

fn best_secs<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let warm = f();
    std::mem::drop(warm);
    let mut best = f64::INFINITY;
    let mut keep = f();
    for _ in 0..reps {
        let t = Instant::now();
        let r = f();
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
        keep = r;
    }
    (best, keep)
}

fn measure_read_bw_bytes_per_s() -> f64 {
    let n_bytes = 256 * 1024 * 1024usize;
    let mut buf = vec![0u8; n_bytes];
    let mut s: u64 = 0xD1B5_4A32_D192_ED03;
    for b in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s & 0xFF) as u8;
    }
    let buf = std::hint::black_box(buf);
    let (secs, sum) = best_secs(8, || {
        let b = std::hint::black_box(buf.as_slice());

        let mut acc = [0u64; 8];
        for c in b.chunks_exact(64) {
            for (a, w) in acc.iter_mut().zip(c.chunks_exact(8)) {
                *a = a.wrapping_add(u64::from_le_bytes(w.try_into().unwrap()));
            }
        }
        acc.iter().fold(0u64, |x, &y| x.wrapping_add(y))
    });
    std::hint::black_box(sum);
    if !secs.is_finite() || secs <= 0.0 {
        return f64::NAN;
    }
    n_bytes as f64 / secs
}

fn main() {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let rayon_threads = rayon::current_num_threads();

    let cfg = TrellisConfig::for_bpw(3.0);
    assert_eq!(cfg.k_bits, 3);
    assert_eq!(cfg.block_len, 256);
    let k = cfg.k_bits;

    let rows = 18944usize;
    let cols = 3584usize;
    let total = rows * cols;
    let enc = synth_encoded(total, k, cfg.block_len);
    let n_blocks = enc.blocks.len();

    println!("STRAND CPU decode-speed gate  (3-bit, k={k}, L={}, block_len={})", cfg.l_bits, cfg.block_len);
    println!("tensor: {rows} x {cols} = {} weights ({:.2} Mw) in {n_blocks} blocks; logical CPUs={cores}, rayon threads={rayon_threads}", total, total as f64 / 1e6);
    println!();

    let ref_fast = decode_q12_fast(&enc, &cfg);
    let par = decode_q12_par(&enc, &cfg);
    let simd = decode_q12_simd(&enc, &cfg);
    assert_eq!(par, ref_fast, "PAR decode != fast at bench size — bit-identity broken");
    assert_eq!(simd, ref_fast, "SIMD decode != fast at bench size — bit-identity broken");
    println!("bit-identity @ bench size: PAR == SIMD == fast  ... OK ({} weights checked)", ref_fast.len());
    std::hint::black_box(&ref_fast);
    println!();

    let reps = 12;
    let mw = |secs: f64| (total as f64 / secs) / 1e6;

    let (s_fast, o_fast) = best_secs(reps, || decode_q12_fast(&enc, &cfg));
    std::hint::black_box(&o_fast);
    let (s_par, o_par) = best_secs(reps, || decode_q12_par(&enc, &cfg));
    std::hint::black_box(&o_par);
    let (s_simd, o_simd) = best_secs(reps, || decode_q12_simd(&enc, &cfg));
    std::hint::black_box(&o_simd);

    let mw_fast = mw(s_fast);
    let mw_par = mw(s_par);
    let mw_simd = mw(s_simd);

    println!("{:<26} {:>10} {:>11} {:>9}", "path", "Mw/s", "Gw/s", "speedup");
    println!("{}", "-".repeat(60));
    println!("{:<26} {mw_fast:>10.1} {:>11.3} {:>9}", "single-thread (fast)", mw_fast / 1e3, "1.00x");
    println!("{:<26} {mw_par:>10.1} {:>11.3} {:>8.2}x", "parallel (rayon)", mw_par / 1e3, mw_par / mw_fast);
    println!("{:<26} {mw_simd:>10.1} {:>11.3} {:>8.2}x", "SIMD (NEON 4-block)", mw_simd / 1e3, mw_simd / mw_fast);
    println!();

    let ceiling_mw = mw_fast * cores as f64;
    let pct_ceiling = mw_par / ceiling_mw * 100.0;
    println!("compute ceiling (cores x single-core): {:.1} Mw/s ({:.3} Gw/s); parallel reaches {:.0}% of it", ceiling_mw, ceiling_mw / 1e3, pct_ceiling);

    let gbs_fast = mw_fast * 1e6 * 4.0 / 1e9;
    let gbs_par = mw_par * 1e6 * 4.0 / 1e9;
    println!("decoded output write rate: single {:.2} GB/s, parallel {:.2} GB/s (Q12 = 4 B/weight)", gbs_fast, gbs_par);
    println!();

    let bw_bytes = measure_read_bw_bytes_per_s();
    let bw_gbs = bw_bytes / 1e9;
    let payload_bpw_bytes = (k as f64) / 8.0;
    let table_bpw_bytes = 16.0 / cfg.block_len as f64;
    let on_disk_bpw = payload_bpw_bytes + table_bpw_bytes;
    let flip_gw_measured = bw_bytes / on_disk_bpw / 1e9;
    let flip_gw_042 = bw_bytes / 0.42 / 1e9;

    println!("measured streaming-read BW: {bw_gbs:.1} GB/s");
    println!("on-disk weight traffic: {:.4} B/weight (payload {:.3} + 16B/{}-block table {:.4})", on_disk_bpw, payload_bpw_bytes, cfg.block_len, table_bpw_bytes);
    println!("compute->bandwidth FLIP point: {:.2} Gw/s @ {:.4} B/w  ( {:.2} Gw/s @ 0.42 B/w headline )", flip_gw_measured, on_disk_bpw, flip_gw_042);

    let par_gw = mw_par / 1e3;
    let headroom = flip_gw_measured / par_gw;
    println!();
    if par_gw < flip_gw_measured {
        println!("verdict: parallel decode {:.3} Gw/s is {:.0}x BELOW the {:.2} Gw/s flip point => still COMPUTE-BOUND", par_gw, headroom, flip_gw_measured);
        println!("         (the byte savings are not yet the bottleneck; more cores / wider SIMD still help.)");
    } else {
        println!("verdict: parallel decode {:.3} Gw/s has REACHED the {:.2} Gw/s flip point => now BANDWIDTH-BOUND", par_gw, flip_gw_measured);
    }
}
