
use crate::block_walk::{block_init_state, exceeds_max_sub, SideInfo, WordReader};
use crate::loader::StrandModel;
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::{decode_lean, reconstruct_q};
use strand_quant::encode::{EncodedTensor, SUB_BLOCK};
use strand_quant::TrellisConfig;

pub fn decode_tensor_q12(model: &StrandModel, name: &str) -> Option<Vec<i32>> {
    let hdr = model.tensor_header(name)?;
    let cfg = model.config_for(hdr);
    let enc = model.encoded_tensor(name)?;
    Some(decode_lean(&enc, &cfg))
}

pub fn decode_q12_fast(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_q12_fast_with_lut(enc, cfg, codebook_lut(cfg.l_bits))
}

#[allow(unsafe_code)]
pub fn decode_q12_fast_with_lut(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32> {
    if cfg.vec_dim() > 1 {
        return strand_quant::decode::decode_lean_with_lut(enc, cfg, lut);
    }

    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let num_states = cfg.num_states();
    debug_assert_eq!(lut.len(), num_states, "scalar LUT must have num_states entries");
    
    let fold = SUB_BLOCK >= num_states;

    if exceeds_max_sub(enc) {
        return strand_quant::decode::decode_lean_with_lut(enc, cfg, lut);
    }

    let mut out: Vec<i32> = Vec::with_capacity(enc.total);
    let mut reader = WordReader::new(&enc.bits, 0);
    let has_affine = enc.has_affine_min;
    let mut folded: Vec<i32> = Vec::new();

    for blk in &enc.blocks {
        let n = blk.n as usize;

        let side = SideInfo::hoist(blk, has_affine);
        let (eff, off) = (side.eff(), side.off());
        let n_sub = side.n_sub;

        let mut state =
            block_init_state(blk, &enc.bits, out.len() * (k as usize), cfg, enc.tail_biting);

        if fold {
            
            folded.clear();
            folded.resize(n_sub * num_states, 0);
            for (sb, &es) in eff.iter().enumerate() {
                let base = sb * num_states;
                for s in 0..num_states {
                    folded[base + s] = reconstruct_q(es, lut[s]);
                }
            }
            let mut i = 0usize;
            for (sb, &o) in off.iter().enumerate() {
                let base = sb * num_states;
                let end = (i + SUB_BLOCK).min(n);
                while i < end {
                    let sym = reader.pop(k) & input_mask;
                    state = ((state << k) | sym) & mask;
                    
                    let v = unsafe { *folded.get_unchecked(base + state) };
                    out.push(v + o);
                    i += 1;
                }
            }
        } else {
            let lut_ptr = lut.as_ptr();
            let mut i = 0usize;
            for (&es, &o) in eff.iter().zip(off.iter()) {
                let end = (i + SUB_BLOCK).min(n);
                while i < end {
                    let sym = reader.pop(k) & input_mask;
                    state = ((state << k) | sym) & mask;
                    
                    let q = unsafe { *lut_ptr.add(state) };
                    out.push(reconstruct_q(es, q) + o);
                    i += 1;
                }
            }
        }
    }
    out
}

pub fn decode_tensor_q12_fast(model: &StrandModel, name: &str) -> Option<Vec<i32>> {
    let hdr = model.tensor_header(name)?;
    let cfg = model.config_for(hdr);
    let enc = model.encoded_tensor(name)?;
    Some(decode_q12_fast(&enc, &cfg))
}

pub fn matvec_named_fast(model: &StrandModel, name: &str, x: &[f32]) -> Option<Vec<f32>> {
    let hdr = model.tensor_header(name)?;
    if hdr.shape.len() < 2 {
        return None;
    }
    let out_features = hdr.shape[0] as usize;
    let in_features = hdr.shape[1] as usize;
    if x.len() != in_features {
        return None;
    }
    let cfg = model.config_for(hdr);
    let enc = model.encoded_tensor(name)?;
    let w = decode_q12_fast(&enc, &cfg);
    if w.len() != out_features * in_features {
        return None;
    }
    let inv = 1.0f32 / 4096.0;
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = 0.0f32;
        for i in 0..in_features {
            acc += (row[i] as f32) * inv * x[i];
        }
        y[o] = acc;
    }
    Some(y)
}

pub fn matvec_named(model: &StrandModel, name: &str, x: &[f32]) -> Option<Vec<f32>> {
    let hdr = model.tensor_header(name)?;
    if hdr.shape.len() < 2 {
        return None;
    }
    let out_features = hdr.shape[0] as usize;
    let in_features = hdr.shape[1] as usize;
    if x.len() != in_features {
        return None;
    }
    let cfg = model.config_for(hdr);
    let enc = model.encoded_tensor(name)?;
    let w = decode_lean(&enc, &cfg);
    if w.len() != out_features * in_features {
        return None;
    }

    let inv = 1.0f32 / 4096.0;
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = 0.0f32;
        for i in 0..in_features {
            acc += (row[i] as f32) * inv * x[i];
        }
        y[o] = acc;
    }
    Some(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use strand_quant::decode::{decode_lean as ref_decode_lean, decode_tensor_fixed};
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts, EncodedTensor};
    use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
    use strand_quant::TrellisConfig;

    fn write_tiny_v2(
        name: &str,
        rows: u64,
        cols: u64,
        cfg: &TrellisConfig,
        enc: &EncodedTensor,
    ) -> std::path::PathBuf {
        let shape = [rows, cols];
        let pt = PackedTensorV2 {
            base: PackedTensor {
                name,
                shape: &shape,
                rht_seed: 0,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc,
            },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write_strand_v2");
        let mut path = std::env::temp_dir();
        let pid = std::process::id();
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("strand_gemv_{name}_{pid}_{uniq}.strand"));
        let mut f = std::fs::File::create(&path).expect("create temp .strand");
        f.write_all(&buf).expect("write temp .strand");
        f.sync_all().ok();
        path
    }

    #[test]
    fn decode_tensor_q12_matches_decode_lean() {
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let path = write_tiny_v2("w", 4, 256, &cfg, &enc);

        let model = StrandModel::open(&path).expect("open");
        let q_runtime = decode_tensor_q12(&model, "w").expect("decode_tensor_q12");
        let q_ref = ref_decode_lean(&enc, &cfg);
        assert_eq!(q_runtime, q_ref, "v2 runtime decode != decode_lean");
        assert_eq!(q_runtime, decode_tensor_fixed(&enc, &cfg));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn matvec_named_matches_lib_matvec() {
        let (rows, cols) = (4usize, 256usize);
        let weights: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.123).sin()).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let path = write_tiny_v2("w", rows as u64, cols as u64, &cfg, &enc);

        let model = StrandModel::open(&path).expect("open");
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.07).cos()).collect();

        let y_runtime = matvec_named(&model, "w", &x).expect("matvec_named");
        let y_ref = crate::matvec(&enc, &cfg, None, rows, cols, &x);
        assert_eq!(y_runtime.len(), y_ref.len());
        for o in 0..rows {
            assert!(
                (y_runtime[o] - y_ref[o]).abs() < 1e-6,
                "row {o}: {} vs {}",
                y_runtime[o],
                y_ref[o]
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn matvec_named_rejects_bad_x_len() {
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let path = write_tiny_v2("w", 4, 256, &cfg, &enc);

        let model = StrandModel::open(&path).expect("open");
        assert!(matvec_named(&model, "w", &[0.0f32; 7]).is_none());
        assert!(matvec_named(&model, "missing", &[0.0f32; 256]).is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fast_decode_is_bit_identical() {
        let configs = [
            (TrellisConfig::for_bpw(3.0), false),
            (TrellisConfig::for_bpw(2.0), false),
            (TrellisConfig::for_bpw(4.0), false),
            (TrellisConfig::for_bpw_l(2.0, 5), true),
            (TrellisConfig::for_bpw_l(4.0, 4), true),
            (TrellisConfig::for_bpw_l(3.0, 5), true),
        ];
        for (cfg, fold_expected) in configs {
            assert_eq!(
                strand_quant::encode::SUB_BLOCK >= cfg.num_states(),
                fold_expected,
                "fold gate mismatch for L={}",
                cfg.l_bits
            );
            for seed in 0..96u64 {
                let n = 1 + (seed as usize * 37) % 2048;
                let w: Vec<f32> = (0..n)
                    .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
                    .collect();
                let variants = [
                    encode_tensor(&w, &cfg),
                    encode_tensor_with(
                        &w,
                        &cfg,
                        &EncodeOpts { tail_biting: true, ..Default::default() },
                    ),
                    encode_tensor_with(
                        &w,
                        &cfg,
                        &EncodeOpts {
                            affine_min: true,
                            ..Default::default()
                        },
                    ),
                    encode_tensor_with(
                        &w,
                        &cfg,
                        &EncodeOpts {
                            tail_biting: true,
                            affine_min: true,
                            ..Default::default()
                        },
                    ),
                ];
                for enc in &variants {
                    assert_eq!(
                        decode_q12_fast(enc, &cfg),
                        ref_decode_lean(enc, &cfg),
                        "fast decode diverged: L={} k={} n={} seed={} tail={} affine={}",
                        cfg.l_bits,
                        cfg.k_bits,
                        n,
                        seed,
                        enc.tail_biting,
                        enc.has_affine_min
                    );
                }
            }
        }
    }

    #[test]
    fn fast_decode_tensor_q12_matches_reference_via_v2() {
        for &(rows, cols) in &[(4u64, 256u64), (3u64, 512u64), (7u64, 768u64)] {
            let n = (rows * cols) as usize;
            let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
            let cfg = TrellisConfig::for_bpw(3.0);
            let enc = encode_tensor(&weights, &cfg);
            let path = write_tiny_v2("w", rows, cols, &cfg, &enc);

            let model = StrandModel::open(&path).expect("open");
            let q_fast = decode_tensor_q12_fast(&model, "w").expect("decode_tensor_q12_fast");
            let q_ref = decode_tensor_q12(&model, "w").expect("decode_tensor_q12");
            assert_eq!(q_fast, q_ref, "fast vs reference mmap decode (rows={rows} cols={cols})");
            assert_eq!(q_fast, decode_tensor_fixed(&enc, &cfg), "fast vs integer reference");

            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn matvec_named_fast_matches_reference() {
        let (rows, cols) = (5usize, 512usize);
        let weights: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.123).sin()).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let path = write_tiny_v2("w", rows as u64, cols as u64, &cfg, &enc);

        let model = StrandModel::open(&path).expect("open");
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.07).cos()).collect();

        let y_fast = matvec_named_fast(&model, "w", &x).expect("matvec_named_fast");
        let y_ref = matvec_named(&model, "w", &x).expect("matvec_named");
        assert_eq!(y_fast.len(), y_ref.len());
        for o in 0..rows {
            assert_eq!(y_fast[o].to_bits(), y_ref[o].to_bits(), "row {o} y mismatch");
        }
        assert!(matvec_named_fast(&model, "w", &[0.0f32; 7]).is_none());
        assert!(matvec_named_fast(&model, "missing", &x).is_none());

        let _ = std::fs::remove_file(&path);
    }
}
