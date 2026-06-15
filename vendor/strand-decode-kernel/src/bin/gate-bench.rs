
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gate-bench: Metal is macOS-only; nothing to measure here.");
}

#[cfg(target_os = "macos")]
fn main() {
    use strand_decode_kernel::metal::{BlockEntry, StrandGpu};
    use strand_quant::codebook::codebook_lut;

    let Some(gpu) = StrandGpu::new() else {
        eprintln!("gate-bench: no Metal device; skipping.");
        return;
    };

    let peak_bps = gpu.bench_peak_bw(64 * 1024 * 1024, 50); 
    let peak_gbs = peak_bps / 1e9;
    println!("device peak streaming-read BW (measured): {peak_gbs:.1} GB/s\n");

    let (k, l) = (3u32, 7u32);
    let lut: Vec<i32> = codebook_lut(l).to_vec();

    let shapes: &[(&str, u32, u32)] = &[
        ("attn_o 3584^2", 3584, 3584),
        ("ffn_up 3584x18944", 3584, 18944),
        ("ffn_down 18944x3584", 18944, 3584),
    ];

    println!(
        "{:<22} {:>9} {:>8} {:>7} {:>9} {:>7}  {}",
        "shape", "GPU us", "GB/s", "%peak", "B/weight", "ops/B", "verdict"
    );
    println!("{}", "-".repeat(82));

    for &(name, rows, cols) in shapes {
        let bpr = cols / 256;
        
        let payload_bits = rows as u64 * cols as u64 * k as u64;
        let payload_bytes = ((payload_bits + 7) / 8) as usize;
        let payload = vec![0xA5u8; payload_bytes + 8];

        let mut tbl = Vec::with_capacity((rows * bpr) as usize);
        for r in 0..rows as u64 {
            for b in 0..bpr as u64 {
                let bit_off = (r * cols as u64 + b * 256) * k as u64;
                tbl.push(BlockEntry {
                    bit_offset: bit_off as u32,
                    init_state: 0,
                    scale_q: 1 << 16,
                    eff: [1 << 16; 8],
                    n: 256,
                    d: 1,
                    _pad: 0,
                });
            }
        }
        let x_rht = vec![1.0f32; cols as usize];

        let secs = gpu.bench_gemv(&payload, &tbl, &lut, rows, cols, k, l, &x_rht, 30);

        let table_bytes = tbl.len() * std::mem::size_of::<BlockEntry>();
        let model_bytes = payload_bytes + table_bytes;
        let total_bytes = model_bytes + cols as usize * 4 + rows as usize * 4; 
        let achieved_gbs = total_bytes as f64 / secs / 1e9;
        let pct_peak = achieved_gbs / peak_gbs * 100.0;
        let b_per_weight = model_bytes as f64 / (rows as f64 * cols as f64);
        let ops_per_byte = (rows as f64 * cols as f64 * 10.0) / total_bytes as f64; 
        let verdict = if pct_peak >= 60.0 {
            "BW-BOUND"
        } else if pct_peak >= 35.0 {
            "marginal"
        } else {
            "COMPUTE-BOUND"
        };
        println!(
            "{name:<22} {:>9.1} {achieved_gbs:>8.1} {pct_peak:>6.0}% {b_per_weight:>9.3} {ops_per_byte:>7.1}  {verdict}",
            secs * 1e6
        );
    }

    println!();
    println!("read: %peak >=60 => BANDWIDTH-bound, byte savings convert to speed (ship the GPU path).");
    println!("      %peak <35  => COMPUTE/latency-bound (the Q3_K trap): fix aligned reads / table size first.");
    println!("B/weight INCLUDES the 52-B/256-block GPU table; Q4_K is ~0.5625 B/weight at iso-ish quality.");
}
