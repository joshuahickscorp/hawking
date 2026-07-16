use strand_quant::decode::decode_tensor_fixed;
use strand_quant::{encode_tensor, TrellisConfig};

fn main() {
    let w: Vec<f32> = (0..8192)
        .map(|i| {
            let x = i as f32;
            (x * 0.011).sin() * 0.7 + (x * 0.0007).cos() * 0.3 + ((x as i64 * 2654435761) % 1000) as f32 / 3000.0
        })
        .collect();
    let cfg = TrellisConfig::for_bpw(4.0);
    let enc = encode_tensor(&w, &cfg);
    let q = decode_tensor_fixed(&enc, &cfg);
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in &q {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    println!("n={} fnv1a={:016x} first={:?} last={:?}", q.len(), h, &q[..4], &q[q.len() - 4..]);
}
