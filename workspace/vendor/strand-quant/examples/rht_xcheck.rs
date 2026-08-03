
use strand_quant::{rht_forward, RhtConfig};

fn rht_seed_for(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h | 1
}

fn main() {
    let mut a = std::env::args().skip(1);
    let n: usize = a.next().expect("in_features").parse().unwrap();
    let name = a.next().expect("tensor name");
    let seed = rht_seed_for(&name);
    
    let x: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.001 - 0.5 + ((i as f32) * 0.05).sin() * 0.1)
        .collect();
    let cfg = RhtConfig::from_seed(seed);
    let y = rht_forward(&x, &cfg);
    println!("SEED {seed}");
    let parts: Vec<String> = y.iter().map(|v| format!("{v:.8e}")).collect();
    println!("{}", parts.join(" "));
}
