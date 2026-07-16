pub fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn normal_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let u1 = ((splitmix64(&mut s) >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        out.push((r * th.cos()) as f32);
        out.push((r * th.sin()) as f32);
    }
    out.truncate(n);
    out
}

pub fn outlier_shaped(n: usize, seed: u64) -> Vec<f32> {
    let mut w = normal_vec(n, seed);

    for (i, v) in w.iter_mut().enumerate() {
        if i % 17 == 0 {
            *v = *v * v.abs() * 1.7;
        }
    }
    let mut mags: Vec<f32> = w.iter().map(|v| v.abs()).collect();
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let thresh = mags[(n / 100).max(1) - 1];
    for v in w.iter_mut() {
        if v.abs() >= thresh {
            *v = 0.0;
        }
    }
    w
}

pub fn rel_rms(w: &[f32], r: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&a, &b) in w.iter().zip(r.iter()) {
        let d = (a - b) as f64;
        num += d * d;
        den += (a as f64) * (a as f64);
    }
    (num / den.max(1e-30)).sqrt()
}

pub fn is_quantizable_linear(name: &str, shape: &[u64]) -> bool {
    if shape.len() != 2 {
        return false;
    }
    let proj = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"];
    proj.iter().any(|p| name.ends_with(p))
}

pub fn rht_seed_for(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h | 1
}
