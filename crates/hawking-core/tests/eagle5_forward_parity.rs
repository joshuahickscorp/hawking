use hawking_core::speculate::eagle5::Eagle5Head;
use serde::Deserialize;
use std::path::PathBuf;
fn b64_decode(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => panic!("invalid b64 char: {c:?}"),
        };
        buf = (buf << 6) | (v as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
        }
    }
    out
}
#[derive(Deserialize)]
#[allow(dead_code)]
struct Fixture {
    schema: String,
    hidden_dim: usize,
    vocab_size: usize,
    n_heads: usize,
    num_blocks: usize,
    prev_token: u32,
    residual_b64: String,
    intermediate_b64: String,
    logits_b64: String,
    argmax: u32,
    top_k: usize,
    top_indices: Vec<u32>,
    top_values: Vec<f32>,
    logits_l2: f32,
}
fn decode_f32(s: &str, expected_len: usize) -> Vec<f32> {
    let raw = b64_decode(s);
    assert_eq!(raw.len(), expected_len * 4, "expected {expected_len} f32s = {} bytes, got {}", expected_len * 4, raw.len());
    let mut out = vec![0.0_f32; expected_len];
    let src = raw.as_ptr() as *const f32;
    // SAFETY: bounds-checked above. We require little-endian host
    unsafe { std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), expected_len) };
    out
}
fn head_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("HAWKING_Q3B_HEAD") {
        let pp = PathBuf::from(p);
        if pp.exists() {
            return Some(pp);
        }
    }
    let home = std::env::var_os("HOME")?;
    let candidate = PathBuf::from(home).join("Downloads/head_final.safetensors");
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}
fn q1p5_head_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("HAWKING_Q1P5_HEAD") {
        let pp = PathBuf::from(p);
        if pp.exists() {
            return Some(pp);
        }
    }
    let home = std::env::var_os("HOME")?;
    let candidate = PathBuf::from(home).join("Downloads/hawking_export/heads/q1p5_eagle6_long.safetensors");
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}
#[test]
#[ignore = "needs HAWKING_Q3B_HEAD or ~/Downloads/head_final.safetensors"]
fn eagle6_forward_matches_pytorch_q3b() {
    let head = head_path().expect("set HAWKING_Q3B_HEAD or place head at ~/Downloads/");
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eagle5_parity_q3b.json");
    let raw = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| panic!("read {}: {e}", fixture_path.display()));
    let f: Fixture = serde_json::from_str(&raw).expect("parse fixture");
    assert_eq!(f.schema, "eagle5-forward-parity-v1");
    let hidden = f.hidden_dim;
    let vocab = f.vocab_size;
    let residual = decode_f32(&f.residual_b64, hidden);
    let intermediate = decode_f32(&f.intermediate_b64, hidden);
    let py_logits = decode_f32(&f.logits_b64, vocab);
    let h = Eagle5Head::load_from_safetensors(&head, hidden, vocab).expect("load head");
    use std::time::Instant;
    let mut timings = Vec::with_capacity(8);
    let mut rust_logits = Vec::new();
    for _ in 0..8 {
        let t0 = Instant::now();
        let l = h.forward_logits(f.prev_token, &residual, &intermediate).expect("Trained head must return logits");
        timings.push(t0.elapsed());
        rust_logits = l;
    }
    timings.sort();
    let median = timings[timings.len() / 2];
    let min = timings[0];
    let max = timings[timings.len() - 1];
    assert!(median.as_millis() < 200, "forward_single_step median {}ms exceeds 200ms perf gate", median.as_millis());
    assert_eq!(rust_logits.len(), vocab, "rust logits length wrong: {} != {vocab}", rust_logits.len());
    let rust_argmax = rust_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    assert_eq!(rust_argmax, f.argmax, "top-1 argmax mismatch: rust={} pytorch={}", rust_argmax, f.argmax);
    let mut l_inf = 0.0_f32;
    let mut l2_sq = 0.0_f64;
    let mut l2_rust = 0.0_f64;
    for i in 0..vocab {
        let diff = rust_logits[i] - py_logits[i];
        l_inf = l_inf.max(diff.abs());
        l2_sq += (diff as f64) * (diff as f64);
        l2_rust += (rust_logits[i] as f64) * (rust_logits[i] as f64);
    }
    let l2 = l2_sq.sqrt();
    let l2_rust = l2_rust.sqrt() as f32;
    const L_INF_TOL: f32 = 5e-2;
    assert!(l_inf <= L_INF_TOL, "L_inf parity violation: {l_inf:.4e} > {L_INF_TOL:.4e}");
    let l2_rel = ((l2_rust - f.logits_l2).abs() / f.logits_l2).abs();
    assert!(l2_rel < 0.01, "logits L2 disagrees by {:.2}%; rust={} pytorch={}", l2_rel * 100.0, l2_rust, f.logits_l2);
    let mut rust_top_k: Vec<(usize, f32)> = rust_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    rust_top_k.select_nth_unstable_by(f.top_k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    rust_top_k.truncate(f.top_k);
    rust_top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    use std::collections::HashSet;
    let rust_set: HashSet<u32> = rust_top_k.iter().take(8).map(|(i, _)| *i as u32).collect();
    let py_set: HashSet<u32> = f.top_indices.iter().take(8).copied().collect();
    let overlap = rust_set.intersection(&py_set).count();
    assert!(overlap >= 7, "top-8 overlap too small: {overlap}/8 (rust={rust_set:?} pytorch={py_set:?})");
}
#[test]
#[ignore = "needs HAWKING_Q1P5_HEAD or ~/Downloads/hawking_export/heads/q1p5_eagle6_long.safetensors"]
fn eagle6_forward_matches_pytorch_q1p5() {
    let head = q1p5_head_path().expect("set HAWKING_Q1P5_HEAD or place q1p5_eagle6_long.safetensors at ~/Downloads/hawking_export/heads/");
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eagle5_parity_q1p5.json");
    let raw = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| panic!("read {}: {e}", fixture_path.display()));
    let f: Fixture = serde_json::from_str(&raw).expect("parse fixture");
    assert_eq!(f.schema, "eagle5-forward-parity-v1");
    let hidden = f.hidden_dim;
    let vocab = f.vocab_size;
    let residual = decode_f32(&f.residual_b64, hidden);
    let intermediate = decode_f32(&f.intermediate_b64, hidden);
    let py_logits = decode_f32(&f.logits_b64, vocab);
    let h = Eagle5Head::load_from_safetensors(&head, hidden, vocab).expect("load q1p5 head");
    let rust_logits = h.forward_logits(f.prev_token, &residual, &intermediate).expect("Trained head must return logits");
    assert_eq!(rust_logits.len(), vocab);
    let rust_argmax = rust_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    assert_eq!(rust_argmax, f.argmax, "q1p5 top-1 argmax mismatch: rust={} pytorch={}", rust_argmax, f.argmax);
    let mut l_inf = 0.0_f32;
    let mut l2_sq = 0.0_f64;
    let mut l2_rust = 0.0_f64;
    for i in 0..vocab {
        let diff = rust_logits[i] - py_logits[i];
        l_inf = l_inf.max(diff.abs());
        l2_sq += (diff as f64) * (diff as f64);
        l2_rust += (rust_logits[i] as f64) * (rust_logits[i] as f64);
    }
    let l2 = l2_sq.sqrt();
    let l2_rust = l2_rust.sqrt() as f32;
    const L_INF_TOL: f32 = 1e-1;
    assert!(l_inf <= L_INF_TOL, "q1p5 L_inf parity violation: {l_inf:.4e} > {L_INF_TOL:.4e}");
    let l2_rel = ((l2_rust - f.logits_l2).abs() / f.logits_l2).abs();
    assert!(l2_rel < 0.01, "q1p5 logits L2 disagrees by {:.2}%; rust={} pytorch={}", l2_rel * 100.0, l2_rust, f.logits_l2,);
    use std::collections::HashSet;
    let mut rust_top_k: Vec<(usize, f32)> = rust_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    rust_top_k.select_nth_unstable_by(f.top_k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    rust_top_k.truncate(f.top_k);
    rust_top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rust_set: HashSet<u32> = rust_top_k.iter().take(8).map(|(i, _)| *i as u32).collect();
    let py_set: HashSet<u32> = f.top_indices.iter().take(8).copied().collect();
    let overlap = rust_set.intersection(&py_set).count();
    assert!(overlap >= 7, "q1p5 top-8 overlap too small: {overlap}/8 (rust={rust_set:?} pytorch={py_set:?})");
}
