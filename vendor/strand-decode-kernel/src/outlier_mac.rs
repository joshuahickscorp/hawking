#[cfg(test)]
use crate::gemv::decode_q12_fast;
use crate::gemv::decode_q12_fast_with_lut;
use crate::loader::StrandModel;
use strand_quant::rht::{rht_forward_cols_inplace, rht_forward_rows_inplace, rht_inverse_cols_inplace, rht_inverse_rows_inplace, RhtConfig};

fn in_features_of(shape: &[u64]) -> Option<usize> {
    shape.last().map(|&d| d as usize)
}

pub fn patched_weights(model: &StrandModel, name: &str) -> Result<Vec<f32>, String> {
    let hdr = model.tensor_header(name).ok_or_else(|| format!("outlier_mac: no tensor {name:?}"))?;
    let cfg = model.config_for(hdr);
    let enc = model.encoded_tensor_checked(name)?;
    let q12 = decode_q12_fast_with_lut(&enc, &cfg, model.lut_for(name)?);

    let mut w: Vec<f32> = q12.iter().map(|&q| (q as f32) * (1.0 / 4096.0)).collect();

    if hdr.has_rht_seed {
        let in_features = in_features_of(&hdr.shape).ok_or_else(|| format!("outlier_mac: tensor {name:?} has empty shape"))?;
        let rcfg = RhtConfig::from_seed(hdr.rht_seed);
        if hdr.rht_cols {
            rht_inverse_cols_inplace(&mut w, &rcfg, in_features);
        } else {
            rht_inverse_rows_inplace(&mut w, &rcfg, in_features);
        }
    }

    if let Some(wire) = model.outlier(name) {
        for (i, v) in wire.dequant_vals() {
            let i = i as usize;
            if i >= w.len() {
                return Err(format!("outlier_mac: tensor {name:?} outlier index {i} out of range ({})", w.len()));
            }
            w[i] = v;
        }
    }
    Ok(w)
}

pub fn bulk_weights(model: &StrandModel, name: &str) -> Result<Vec<f32>, String> {
    let hdr = model.tensor_header(name).ok_or_else(|| format!("outlier_mac: no tensor {name:?}"))?;
    let cfg = model.config_for(hdr);
    let enc = model.encoded_tensor_checked(name)?;
    let q12 = decode_q12_fast_with_lut(&enc, &cfg, model.lut_for(name)?);
    let mut w: Vec<f32> = q12.iter().map(|&q| (q as f32) * (1.0 / 4096.0)).collect();
    if hdr.has_rht_seed {
        let in_features = in_features_of(&hdr.shape).ok_or_else(|| format!("outlier_mac: tensor {name:?} has empty shape"))?;
        let rcfg = RhtConfig::from_seed(hdr.rht_seed);
        if hdr.rht_cols {
            rht_inverse_cols_inplace(&mut w, &rcfg, in_features);
        } else {
            rht_inverse_rows_inplace(&mut w, &rcfg, in_features);
        }
    }
    Ok(w)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlierResidual {
    pub row: u32,
    pub col: u32,

    pub resid: f32,
}

pub fn outlier_residuals(model: &StrandModel, name: &str) -> Result<Vec<OutlierResidual>, String> {
    let Some(wire) = model.outlier(name) else {
        return Ok(Vec::new());
    };
    let hdr = model.tensor_header(name).ok_or_else(|| format!("outlier_mac: no tensor {name:?}"))?;
    let in_features = in_features_of(&hdr.shape).ok_or_else(|| format!("outlier_mac: tensor {name:?} has empty shape"))?;
    let bulk = bulk_weights(model, name)?;
    let mut out = Vec::with_capacity(wire.entries.len());
    for (i, val) in wire.dequant_vals() {
        let i = i as usize;
        if i >= bulk.len() {
            return Err(format!("outlier_mac: tensor {name:?} outlier index {i} out of range ({})", bulk.len()));
        }
        out.push(OutlierResidual { row: (i / in_features) as u32, col: (i % in_features) as u32, resid: val - bulk[i] });
    }
    Ok(out)
}

pub fn matvec_patched(model: &StrandModel, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
    let hdr = model.tensor_header(name).ok_or_else(|| format!("outlier_mac: no tensor {name:?}"))?;
    if hdr.shape.len() < 2 {
        return Err(format!("outlier_mac: tensor {name:?} is not 2-D"));
    }
    let out_features = hdr.shape[0] as usize;
    let in_features = hdr.shape[1] as usize;
    if x.len() != in_features {
        return Err(format!("outlier_mac: x has {} elements, tensor {name:?} expects {in_features}", x.len()));
    }
    let w = patched_weights(model, name)?;
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = 0.0f32;
        for i in 0..in_features {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
    Ok(y)
}

pub fn matvec_rht(model: &StrandModel, name: &str, x: &[f32], residuals: Option<&[OutlierResidual]>) -> Result<Vec<f32>, String> {
    let hdr = model.tensor_header(name).ok_or_else(|| format!("outlier_mac: no tensor {name:?}"))?;
    if hdr.shape.len() < 2 {
        return Err(format!("outlier_mac: tensor {name:?} is not 2-D"));
    }
    let out_features = hdr.shape[0] as usize;
    let in_features = hdr.shape[1] as usize;
    if x.len() != in_features {
        return Err(format!("outlier_mac: x has {} elements, tensor {name:?} expects {in_features}", x.len()));
    }
    let cfg = model.config_for(hdr);
    let enc = model.encoded_tensor_checked(name)?;
    let q12 = decode_q12_fast_with_lut(&enc, &cfg, model.lut_for(name)?);
    if q12.len() != out_features * in_features {
        return Err(format!("outlier_mac: tensor {name:?} decoded {} weights, shape says {}", q12.len(), out_features * in_features));
    }

    let inv = 1.0f32 / 4096.0;
    let mut y = vec![0.0f32; out_features];
    if hdr.has_rht_seed && hdr.rht_cols {
        // Column-sign RHT: the activation transform is row-independent, so compute it
        // ONCE and reuse it for every output row (the cheap-serving win; the per-row
        // path below is the row-RHT wall this avoids — see
        // single_rotation_recipe_diverges_for_multirow).
        let rcfg = RhtConfig::from_seed(hdr.rht_seed);
        let mut tx = x.to_vec();
        rht_forward_cols_inplace(&mut tx, &rcfg, in_features);
        for o in 0..out_features {
            let qrow = &q12[o * in_features..(o + 1) * in_features];
            let mut acc = 0.0f32;
            for i in 0..in_features {
                acc += (qrow[i] as f32) * inv * tx[i];
            }
            y[o] = acc;
        }
    } else if hdr.has_rht_seed {
        let rcfg = RhtConfig::from_seed(hdr.rht_seed);
        let mut x_rht = Vec::with_capacity(out_features * in_features);
        for _ in 0..out_features {
            x_rht.extend_from_slice(x);
        }
        rht_forward_rows_inplace(&mut x_rht, &rcfg, in_features);
        for o in 0..out_features {
            let qrow = &q12[o * in_features..(o + 1) * in_features];
            let xrow = &x_rht[o * in_features..(o + 1) * in_features];
            let mut acc = 0.0f32;
            for i in 0..in_features {
                acc += (qrow[i] as f32) * inv * xrow[i];
            }
            y[o] = acc;
        }
    } else {
        for o in 0..out_features {
            let qrow = &q12[o * in_features..(o + 1) * in_features];
            let mut acc = 0.0f32;
            for i in 0..in_features {
                acc += (qrow[i] as f32) * inv * x[i];
            }
            y[o] = acc;
        }
    }

    let computed;
    let res: &[OutlierResidual] = match residuals {
        Some(r) => r,
        None => {
            computed = outlier_residuals(model, name)?;
            &computed
        }
    };
    for r in res {
        y[r.row as usize] += r.resid * x[r.col as usize];
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use strand_quant::encode::encode_tensor;
    use strand_quant::format::{write_strand_v2, write_strand_v2_rht, PackedTensor, PackedTensorV2};
    use strand_quant::outlier_wire::{append_outl, idx_bits_for, OutlierWire};
    use strand_quant::rht::{rht_forward, rht_forward_cols, rht_forward_rows, RhtConfig};
    use strand_quant::TrellisConfig;

    fn rht_seed_for(name: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h | 1
    }

    fn test_weights(n: usize, seed: u64) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect()
    }

    fn bake_fixture(name: &str, rows: usize, cols: usize, outlier_pct: f64, use_rht: bool, use_cols: bool) -> (std::path::PathBuf, Vec<f32>) {
        let cfg = TrellisConfig::for_bpw_l(2.0, 8);
        let gt = test_weights(rows * cols, 0xC0FFEE);
        let n = gt.len();
        let ob = 8u32;

        let outliers: Option<(Vec<usize>, Vec<f32>, Vec<i32>, f32)> = if outlier_pct > 0.0 {
            let k = ((outlier_pct / 100.0) * n as f64).round() as usize;
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| gt[b].abs().partial_cmp(&gt[a].abs()).unwrap_or(std::cmp::Ordering::Equal));
            let idx: Vec<usize> = order[..k].to_vec();
            let omax = idx.iter().fold(0f32, |m, &i| m.max(gt[i].abs())).max(1e-12);
            let levels = ((1i64 << (ob - 1)) - 1) as f32;
            let vals: Vec<f32> = idx.iter().map(|&i| (gt[i] / omax * levels).round() / levels * omax).collect();
            let codes: Vec<i32> = idx.iter().map(|&i| (gt[i] / omax * levels).round() as i32).collect();
            Some((idx, vals, codes, omax))
        } else {
            None
        };
        let mut bulk = gt.clone();
        if let Some((idx, ..)) = &outliers {
            for &i in idx {
                bulk[i] = 0.0;
            }
        }
        let seed = rht_seed_for(name);
        let work = if use_cols {
            rht_forward_cols(&bulk, &RhtConfig::from_seed(seed), cols)
        } else if use_rht {
            rht_forward_rows(&bulk, &RhtConfig::from_seed(seed), cols)
        } else {
            bulk.clone()
        };
        let mut enc = encode_tensor(&work, &cfg);
        enc.has_rht_seed = use_rht || use_cols;

        let q12 = strand_quant::decode::decode_tensor_fixed(&enc, &cfg);
        let mut recon: Vec<f32> = q12.iter().map(|&q| (q as f32) * (1.0 / 4096.0)).collect();
        if use_cols {
            rht_inverse_cols_inplace(&mut recon, &RhtConfig::from_seed(seed), cols);
        } else if use_rht {
            rht_inverse_rows_inplace(&mut recon, &RhtConfig::from_seed(seed), cols);
        }
        if let Some((idx, vals, ..)) = &outliers {
            for (&i, &v) in idx.iter().zip(vals.iter()) {
                recon[i] = v;
            }
        }

        let shape = [rows as u64, cols as u64];
        let pt = PackedTensorV2 {
            base: PackedTensor {
                name,
                shape: &shape,
                rht_seed: if use_rht || use_cols { seed } else { 0 },
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc: &enc,
            },
            block_len: cfg.block_len as u32,
        };
        let buf = if use_cols { write_strand_v2_rht(&[pt], [0u8; 32], true, false, &[true]).expect("write v2 cols") } else { write_strand_v2(&[pt], [0u8; 32], true).expect("write v2") };
        let mut path = std::env::temp_dir();
        path.push(format!("strand_outlmac_{name}_{}_{}.strand", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
        let mut f = std::fs::File::create(&path).expect("create temp .strand");
        f.write_all(&buf).expect("write temp .strand");
        f.sync_all().ok();
        if let Some((idx, _vals, codes, omax)) = outliers {
            let wire = OutlierWire::from_selection(n, idx, codes, omax, ob);
            assert_eq!(wire.idx_bits, idx_bits_for(n));
            append_outl(&path, &[Some(wire)]).expect("append outl");
        }
        (path, recon)
    }

    #[test]
    fn patched_decode_byte_equals_recon_path() {
        for &(rows, cols, pct, rht) in &[(4usize, 256usize, 1.0f64, true), (4, 256, 1.0, false), (3, 512, 2.0, true), (4, 256, 0.0, true)] {
            let name = "model.layers.0.mlp.down_proj.weight";
            let (path, recon) = bake_fixture(name, rows, cols, pct, rht, false);
            let model = StrandModel::open(&path).expect("open");
            let got = patched_weights(&model, name).expect("patched_weights");
            assert_eq!(got.len(), recon.len());
            let bits_got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let bits_want: Vec<u32> = recon.iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits_got, bits_want, "patched decode != recon path (rows={rows} cols={cols} pct={pct} rht={rht})");
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn patched_and_rht_macs_agree() {
        let name = "model.layers.1.self_attn.q_proj.weight";
        let (rows, cols) = (4usize, 256usize);
        let (path, recon) = bake_fixture(name, rows, cols, 1.5, true, false);
        let model = StrandModel::open(&path).expect("open");
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.07).cos()).collect();

        let mut y_ref = vec![0.0f32; rows];
        for o in 0..rows {
            let row = &recon[o * cols..(o + 1) * cols];
            let mut acc = 0.0f32;
            for i in 0..cols {
                acc += row[i] * x[i];
            }
            y_ref[o] = acc;
        }

        let y_pat = matvec_patched(&model, name, &x).expect("matvec_patched");
        for o in 0..rows {
            assert_eq!(y_pat[o].to_bits(), y_ref[o].to_bits(), "matvec_patched must be bit-equal to the recon GEMV (row {o})");
        }

        let res = outlier_residuals(&model, name).expect("residuals");
        assert!(!res.is_empty(), "fixture must exercise the sparse term");
        let y_rht = matvec_rht(&model, name, &x, Some(&res)).expect("matvec_rht");
        let y_rht2 = matvec_rht(&model, name, &x, None).expect("matvec_rht (recompute)");
        assert_eq!(y_rht, y_rht2, "residual precompute must not change the result");
        for o in 0..rows {
            let scale = y_ref[o].abs().max(1.0);
            assert!((y_rht[o] - y_ref[o]).abs() / scale < 1e-4, "matvec_rht diverged at row {o}: {} vs {}", y_rht[o], y_ref[o]);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn single_rotation_recipe_diverges_for_multirow() {
        let name = "model.layers.2.mlp.gate_proj.weight";
        let (rows, cols) = (4usize, 256usize);
        let (path, recon) = bake_fixture(name, rows, cols, 0.0, true, false);
        let model = StrandModel::open(&path).expect("open");
        let hdr = model.tensor_header(name).unwrap();
        let cfg = model.config_for(hdr);
        let enc = model.encoded_tensor_checked(name).unwrap();
        let q12 = decode_q12_fast(&enc, &cfg);
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin() + 0.1).collect();

        let mut y_ref = vec![0.0f32; rows];
        for o in 0..rows {
            let row = &recon[o * cols..(o + 1) * cols];
            y_ref[o] = row.iter().zip(&x).map(|(w, xv)| w * xv).sum();
        }

        let x_rht = rht_forward(&x, &RhtConfig::from_seed(hdr.rht_seed));
        let inv = 1.0f32 / 4096.0;
        let mut y_naive = vec![0.0f32; rows];
        for o in 0..rows {
            let qrow = &q12[o * cols..(o + 1) * cols];
            y_naive[o] = qrow.iter().zip(&x_rht).map(|(&q, xv)| (q as f32) * inv * xv).sum();
        }

        assert!((y_naive[0] - y_ref[0]).abs() / y_ref[0].abs().max(1.0) < 1e-2, "row 0 should roughly match (it shares the sign prefix)");

        let worst = (1..rows).map(|o| (y_naive[o] - y_ref[o]).abs() / y_ref[o].abs().max(1e-3)).fold(0.0f32, f32::max);
        assert!(
            worst > 0.05,
            "single-rotation recipe unexpectedly matched multirow output \
             (worst rel err {worst}) — did the encoder's sign derivation change?"
        );

        let y_fixed = matvec_rht(&model, name, &x, None).expect("matvec_rht");
        for o in 0..rows {
            let scale = y_ref[o].abs().max(1.0);
            assert!((y_fixed[o] - y_ref[o]).abs() / scale < 1e-4, "matvec_rht must match the reference at row {o}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn col_rht_single_rotation_serves_all_rows() {
        // The complement of single_rotation_recipe_diverges_for_multirow: with COLUMN-sign
        // RHT the activation transform is row-independent, so ONE rht_forward(x) serves every
        // output row — and the production matvec_rht col path reproduces the spatial GEMV.
        let name = "model.layers.3.mlp.down_proj.weight";
        let (rows, cols) = (4usize, 256usize);
        let (path, recon) = bake_fixture(name, rows, cols, 0.0, false, true);
        let model = StrandModel::open(&path).expect("open");
        let hdr = model.tensor_header(name).unwrap();
        assert!(hdr.rht_cols, "col archive must carry the rht_cols flag (bit 3)");
        assert!(hdr.has_rht_seed, "col archive must still carry the RHT seed");
        let cfg = model.config_for(hdr);
        let enc = model.encoded_tensor_checked(name).unwrap();
        let q12 = decode_q12_fast(&enc, &cfg);
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin() + 0.1).collect();

        // spatial reference: bake_fixture already applied rht_inverse_cols to `recon`.
        let mut y_ref = vec![0.0f32; rows];
        for o in 0..rows {
            let row = &recon[o * cols..(o + 1) * cols];
            y_ref[o] = row.iter().zip(&x).map(|(w, xv)| w * xv).sum();
        }

        // ONE shared activation transform, reused for every row (the cheap-serving win).
        let x_rht = rht_forward(&x, &RhtConfig::from_seed(hdr.rht_seed));
        let inv = 1.0f32 / 4096.0;
        let worst = (0..rows)
            .map(|o| {
                let qrow = &q12[o * cols..(o + 1) * cols];
                let y: f32 = qrow.iter().zip(&x_rht).map(|(&q, xv)| (q as f32) * inv * xv).sum();
                (y - y_ref[o]).abs() / y_ref[o].abs().max(1e-3)
            })
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-2, "col-RHT single rotation must serve ALL rows (worst rel err {worst})");

        // and the production decoder col path agrees with the spatial reference.
        let y_fixed = matvec_rht(&model, name, &x, None).expect("matvec_rht");
        for o in 0..rows {
            let scale = y_ref[o].abs().max(1.0);
            assert!((y_fixed[o] - y_ref[o]).abs() / scale < 1e-3, "matvec_rht col path diverged at row {o}: {} vs {}", y_fixed[o], y_ref[o]);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn errors_not_panics() {
        let name = "model.layers.0.mlp.up_proj.weight";
        let (path, _recon) = bake_fixture(name, 4, 256, 1.0, true, false);
        let model = StrandModel::open(&path).expect("open");
        assert!(matvec_patched(&model, name, &[0.0; 7]).is_err());
        assert!(matvec_patched(&model, "missing", &[0.0; 256]).is_err());
        assert!(matvec_rht(&model, name, &[0.0; 7], None).is_err());
        assert!(patched_weights(&model, "missing").is_err());
        let _ = std::fs::remove_file(&path);
    }
}
