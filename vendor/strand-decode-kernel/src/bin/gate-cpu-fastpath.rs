
use std::time::Instant;

use strand_decode_kernel::gemv::decode_q12_fast;
use strand_quant::decode::decode_lean;
use strand_quant::encode::{encode_tensor, EncodedTensor};
use strand_quant::TrellisConfig;

const Q4K_BYTES_PER_WEIGHT: f64 = 0.5625;

fn decode_then_gemv(
    f: impl Fn(&EncodedTensor, &TrellisConfig) -> Vec<i32>,
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    rows: usize,
    cols: usize,
    x: &[f32],
    y: &mut [f32],
) {
    let w = f(enc, cfg);
    let inv = 1.0f32 / 4096.0;
    for o in 0..rows {
        let row = &w[o * cols..(o + 1) * cols];
        let mut acc = 0.0f32;
        for i in 0..cols {
            acc += (row[i] as f32) * inv * x[i];
        }
        y[o] = acc;
    }
}

fn main() {
    
    let cfg = TrellisConfig::for_bpw(3.0);
    assert_eq!((cfg.k_bits, cfg.l_bits), (3, 7), "deploy point is k=3,L=7");

    let bench_rows = 256usize; 
    let shapes: &[(&str, usize)] = &[
        ("attn_o   cols=3584", 3584),
        ("ffn_up    cols=18944", 18944),
        ("ffn_down  cols=3584", 3584),
    ];

    println!("STRAND CPU decode→GEMV gate — 3-bit deploy (k=3, L=7), real encoded tensors");
    println!("(decode is integer-only, float-free; the only float is the Q12·(1/4096)·x MAC)\n");

    print!("correctness: decode_q12_fast == decode_lean (bit-for-bit) ... ");
    {
        let mut all_ok = true;
        for &(_name, cols) in shapes {
            let n = bench_rows * cols;
            let weights: Vec<f32> =
                (0..n).map(|i| ((i as f32) * 0.0007).sin() * 0.6).collect();
            let enc = encode_tensor(&weights, &cfg);
            let fast = decode_q12_fast(&enc, &cfg);
            let refr = decode_lean(&enc, &cfg);
            if fast != refr {
                all_ok = false;
                eprintln!("\nCORRECTNESS FAIL on cols={cols}: fast decode != decode_lean");
            }
        }
        if !all_ok {
            eprintln!("ABORT: not reporting throughput for an incorrect decode.");
            std::process::exit(1);
        }
        println!("PASS");
    }

    println!(
        "\n{:<22} {:>8} {:>12} {:>12} {:>9} {:>10} {:>9}  GEMV fast",
        "shape", "rows", "decode ref", "decode fast", "speedup", "B/weight", "vs Q4_K",
    );
    println!("{}", "-".repeat(108));

    for &(name, cols) in shapes {
        let rows = bench_rows;
        let n = rows * cols;
        let weights: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0007).sin() * 0.6).collect();
        let enc = encode_tensor(&weights, &cfg);
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.01).cos()).collect();
        let mut y = vec![0.0f32; rows];

        let bpw = enc.total_bpw(&cfg);
        let bytes_per_weight = bpw / 8.0;

        let _ = decode_lean(&enc, &cfg);
        let _ = decode_q12_fast(&enc, &cfg);

        let iters = 40usize;
        let t0 = Instant::now();
        for _ in 0..iters {
            let w = decode_lean(&enc, &cfg);
            std::hint::black_box(&w);
        }
        let secs_ref = t0.elapsed().as_secs_f64();
        let mw_ref = (n * iters) as f64 / secs_ref / 1e6;

        let t1 = Instant::now();
        for _ in 0..iters {
            let w = decode_q12_fast(&enc, &cfg);
            std::hint::black_box(&w);
        }
        let secs_fast = t1.elapsed().as_secs_f64();
        let mw_fast = (n * iters) as f64 / secs_fast / 1e6;

        let t2 = Instant::now();
        for _ in 0..iters {
            decode_then_gemv(decode_q12_fast, &enc, &cfg, rows, cols, &x, &mut y);
            std::hint::black_box(&y);
        }
        let secs_gemv = t2.elapsed().as_secs_f64();
        let mw_gemv = (n * iters) as f64 / secs_gemv / 1e6;

        let speedup = mw_fast / mw_ref;
        let dens = if bytes_per_weight < Q4K_BYTES_PER_WEIGHT {
            "BEATS"
        } else {
            "loses"
        };
        println!(
            "{name:<22} {rows:>8} {mw_ref:>9.1} M {mw_fast:>10.1} M {speedup:>8.2}x {bytes_per_weight:>9.4} {dens:>9} {mw_gemv:>7.1} M"
        );
    }

    println!("\nlegend:");
    println!("  decode ref/fast = Mweights/s of integer decode (decode_lean vs decode_q12_fast).");
    println!("  B/weight        = on-disk payload + ALL side info per weight (enc.total_bpw/8).");
    println!("  vs Q4_K         = beats if B/weight < {Q4K_BYTES_PER_WEIGHT} (Q4_K's 4.5 bpw iso-ish quality).");
    println!("  GEMV fast       = Mweights/s of the full decode→matvec (fast decode + Q12·1/4096·x MAC).");
    println!("\nThis path needs NO GPU and NO float unit for the decode; it ships today on CPU,");
    println!("WASM, phone, or MCU and is bit-identical across all of them (the determinism moat).");
}
