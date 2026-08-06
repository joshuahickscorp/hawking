#![allow(dead_code)]
use std::path::{Path, PathBuf};
pub fn weights_path_qwen() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}
pub fn weights_path_deepseek() -> PathBuf {
    PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
}
pub fn weights_path_qwen_0_5b() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
}
pub fn require_path(p: impl AsRef<Path>, label: &str) -> Option<PathBuf> {
    let p = p.as_ref();
    if p.exists() {
        Some(p.to_path_buf())
    } else {
        eprintln!("skipping {label}: {} not found", p.display());
        None
    }
}
pub fn find_gguf_with_tags(tags: &[&str]) -> Option<PathBuf> {
    let dir = PathBuf::from("../../models");
    for e in std::fs::read_dir(&dir).ok()?.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("gguf") {
            continue;
        }
        let name = p.file_name()?.to_str()?.to_lowercase();
        if tags.iter().all(|t| name.contains(t)) {
            return Some(p);
        }
    }
    None
}
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}
pub fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}
pub fn lcg_f32(n: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let span = hi - lo;
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
            lo + (x as f32 / u32::MAX as f32) * span
        })
        .collect()
}
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}
pub fn worst_violation(actual: &[f32], reference: &[f32], atol: f32, rtol: f32) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
        let excess = (a - r).abs() - (atol + rtol * r.abs());
        if excess > worst {
            worst = excess;
            worst_i = i;
        }
    }
    (worst, worst_i)
}
pub fn rel_l2(reference: &[f32], got: &[f32]) -> f64 {
    let num: f64 = reference
        .iter()
        .zip(got)
        .map(|(&r, &g)| ((r - g) as f64).powi(2))
        .sum();
    let den: f64 = reference
        .iter()
        .map(|&r| (r as f64).powi(2))
        .sum::<f64>()
        .max(1e-30);
    (num / den).sqrt()
}
pub const ATOL: f32 = 1e-3;
pub fn make_q4k_bytes_pcg(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    use half::f16;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;
    let n_blocks = rows * (cols / 256);
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
        let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}
pub fn make_q4k_predec(
    rows: usize,
    cols: usize,
    seed: u32,
    scale_mul: f32,
    scale_add: f32,
) -> (Vec<u8>, Vec<f32>) {
    let bpr = cols / 256;
    let w: Vec<u8> = (0..rows * bpr * 144)
        .map(|i| ((i as u32).wrapping_mul(2_246_822_519).wrapping_add(seed)) as u8)
        .collect();
    let s: Vec<f32> = (0..rows * bpr * 16)
        .map(|i| {
            let v = ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32
                / u32::MAX as f32;
            v * scale_mul + scale_add
        })
        .collect();
    (w, s)
}
pub fn make_q4k_predec_pm05(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    make_q4k_predec(rows, cols, seed, 1.0, -0.5)
}
pub fn make_q4k_predec_pm1(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    make_q4k_predec(rows, cols, seed, 2.0, -1.0)
}
pub fn make_q4k_predec_pm025(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    make_q4k_predec(rows, cols, seed, 0.5, -0.25)
}
pub fn hash16_tokens(ids: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for &id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())[..16].to_string()
}
pub fn check_or_pin_hash(pin_path: &Path, label: &str, actual_hash: &str) {
    let actual_line = format!("{label}: {actual_hash}\n");
    let existing = std::fs::read_to_string(pin_path).unwrap_or_default();
    match existing
        .lines()
        .find(|l| l.starts_with(&format!("{label}:")))
    {
        None => {
            let mut all = existing;
            all.push_str(&actual_line);
            std::fs::write(pin_path, all).expect("write pin");
            eprintln!("PINNED first hash for {label}: {actual_hash}");
        }
        Some(prior) => assert_eq!(
            prior.trim(),
            actual_line.trim(),
            "greedy hash drift for {label}"
        ),
    }
}
#[cfg(target_os = "macos")]
use hawking_core::metal::{MetalContext, PinnedBuffer};
#[cfg(target_os = "macos")]
use once_cell::sync::Lazy;
#[cfg(target_os = "macos")]
pub fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}
#[cfg(target_os = "macos")]
pub fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}
#[cfg(target_os = "macos")]
pub fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}
#[cfg(target_os = "macos")]
pub fn new_f16_buf_from_f32(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    use half::f16;
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    ctx.new_buffer_with_bytes(&bytes)
}
#[cfg(target_os = "macos")]
pub fn new_f16_buf(ctx: &MetalContext, data: &[half::f16]) -> PinnedBuffer {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    ctx.new_buffer_with_bytes(&bytes)
}
pub fn f16_round_trip(data: &[f32]) -> Vec<f32> {
    use half::f16;
    data.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}
pub fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    use half::f16;
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_le_bytes())
        .collect()
}
#[cfg(target_os = "macos")]
pub fn run_predec_pair_combined(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    g_scales: &[f32],
    u_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    dispatch: impl FnOnce(
        &mut hawking_core::metal::TokenCommandBuffer,
        &PinnedBuffer,
        usize,
        &PinnedBuffer,
        &PinnedBuffer,
        &PinnedBuffer,
        &PinnedBuffer,
        &PinnedBuffer,
    ),
) -> (Vec<f32>, Vec<f32>) {
    use hawking_core::metal::TokenCommandBuffer;
    let w_bytes = rows * (cols / 256) * 144;
    let mut combined = Vec::with_capacity(wg.len() + wu.len());
    combined.extend_from_slice(wg);
    combined.extend_from_slice(wu);
    let combined_buf = ctx.new_buffer_with_bytes(&combined);
    let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
    let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
    let x_buf = new_f32_buf(ctx, x);
    let yg_buf = ctx.new_buffer(rows * 4);
    let yu_buf = ctx.new_buffer(rows * 4);
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        dispatch(
            &mut tcb,
            &combined_buf,
            w_bytes,
            &gs_buf,
            &us_buf,
            &x_buf,
            &yg_buf,
            &yu_buf,
        );
        tcb.commit_and_wait().expect("pair commit");
    }
    (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
}
pub fn with_env_vars<R>(pairs: &[(&str, Option<&str>)], f: impl FnOnce() -> R) -> R {
    let prior: Vec<(String, Option<std::ffi::OsString>)> = pairs
        .iter()
        .map(|(k, _)| ((*k).to_string(), std::env::var_os(k)))
        .collect();
    for (k, v) in pairs {
        match v {
            Some(val) => std::env::set_var(k, val),
            None => std::env::remove_var(k),
        }
    }
    let out = f();
    for (k, v) in prior {
        match v {
            Some(val) => std::env::set_var(&k, val),
            None => std::env::remove_var(&k),
        }
    }
    out
}
