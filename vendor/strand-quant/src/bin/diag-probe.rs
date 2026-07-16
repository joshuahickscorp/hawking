use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use strand_quant::{decode_tensor, encode_tensor, TrellisConfig};

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn read_bf16(path: &Path, name: &str) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    let hl = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let json = std::str::from_utf8(&bytes[8..8 + hl]).unwrap();
    let ds = 8 + hl;

    let key = format!("\"{name}\"");
    let p = json.find(&key).unwrap();
    let rest = &json[p..];
    let off_p = rest.find("data_offsets").unwrap();
    let b1 = rest[off_p..].find('[').unwrap() + off_p;
    let b2 = rest[b1..].find(']').unwrap() + b1;
    let nums: Vec<usize> = rest[b1 + 1..b2].split(',').map(|t| t.trim().parse().unwrap()).collect();
    let raw = &bytes[ds + nums[0]..ds + nums[1]];
    raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}

fn mse(a: &[f32], b: &[f32]) -> (f64, f64) {
    let n = a.len().min(b.len());
    let mut se = 0.0;
    let mut pw = 0.0;
    for i in 0..n {
        let d = a[i] as f64 - b[i] as f64;
        se += d * d;
        pw += (a[i] as f64) * (a[i] as f64);
    }
    (se / n as f64, (se / pw).sqrt() * 100.0)
}

fn main() {
    let safetensors = std::env::args_os().nth(1).map(PathBuf::from).or_else(|| std::env::var_os("STRAND_DIAG_SAFETENSORS").map(PathBuf::from)).unwrap_or_else(|| {
        eprintln!("usage: diag-probe <model.safetensors>  # or set STRAND_DIAG_SAFETENSORS");
        std::process::exit(64);
    });
    let t0 = Instant::now();
    let gt = read_bf16(&safetensors, "model.layers.23.mlp.down_proj.weight");
    println!("tensor n={} std={:.5}", gt.len(), {
        let m = gt.iter().map(|&x| x as f64).sum::<f64>() / gt.len() as f64;
        (gt.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / gt.len() as f64).sqrt()
    });
    println!("(Q4_K reference on this tensor: ~7.45% rel, 4.5 bpw, scale+min per 32 weights)\n");
    println!("{:>5} {:>5} {:>5} {:>8} {:>11} {:>9}", "k", "L", "blk", "tot_bpw", "mse", "rel%");

    for &k in &[4u32] {
        for &l in &[10u32, 12] {
            for &blk in &[256usize, 64, 32, 16] {
                let cfg = TrellisConfig::new(l, k, blk);
                let enc = encode_tensor(&gt, &cfg);
                let recon = decode_tensor(&enc, &cfg);
                let (m, rel) = mse(&gt, &recon);
                println!("{:>5} {:>5} {:>5} {:>8.3} {:>11.3e} {:>8.2}%", cfg.k_bits, cfg.l_bits, cfg.block_len, enc.total_bpw(&cfg), m, rel);
            }
        }
    }
    eprintln!("\n[diag wall {:.1?}]", t0.elapsed());
}
