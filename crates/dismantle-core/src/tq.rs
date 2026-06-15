//! TQ (Trellis-Quant) — dismantle's deterministic sub-4-bit weight-serving project,
//! behind the `tq` feature. Reads `.tq` artifacts and serves them on CPU.
//!
//! TQ is the dismantle-side integration of the absorbed `strand-quant` codec: a
//! `.tq` file is the strand-quant `STR2` wire format (the extension is TQ's project
//! identity; the on-disk magic stays `STR2`). This module is the CPU serving
//! reference — integer-deterministic Q12 decode (delegated to `strand-quant`) plus
//! the activation-RHT matvec, mirroring `vendor/strand-decode-kernel/outlier_mac.rs`.
//! It is the **contract dismantle's Metal GEMV must reproduce bit-for-bit** (wiring
//! recipe Steps 5-9): decode the trellis-coded weights to Q12, then serve them
//! against the RHT-transformed activation. The GPU bitslice kernel is staged; this
//! CPU path is the parity oracle it will be gated against.
//!
//! Float only ever appears in the final MAC and the activation transform — the Q12
//! decode itself is integer-only and bit-identical across CPU/GPU/WASM (the
//! determinism moat). Gated behind the `tq` cargo feature so the default dismantle
//! build is byte-identical (no `strand-quant` dep pulled in).

use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::EncodedTensor;
use strand_quant::rht::{
    rht_forward_cols_inplace, rht_forward_rows_inplace, rht_inverse_cols_inplace, RhtConfig,
};
use strand_quant::TrellisConfig;

/// Canonical file extension for a TQ artifact (no leading dot). The baker writes
/// `<name>.tq`; the loader recognises it. Centralised so the project can be
/// rebranded by changing this one constant.
pub const TQ_EXT: &str = "tq";

/// Float scale of a decoded Q12 weight: `weight = q12 / 2^QUANTILE_SHIFT`.
/// Matches `strand_quant::decode::decode_tensor`'s private `Q12_TO_F32` exactly
/// (derived from the same `QUANTILE_SHIFT`, never a hard-coded 4096).
#[inline]
pub fn q12_to_f32() -> f32 {
    1.0 / (1u32 << strand_quant::QUANTILE_SHIFT) as f32
}

/// Activation-RHT serving mode for a `.strand` tensor, read from the v2
/// per-tensor flag byte (bit0 `has_rht_seed`, bit3 `rht_cols`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RhtMode {
    /// No RHT (flag bit0 clear): serve the weights directly.
    None,
    /// Per-row RHT (bit0 set, bit3 clear): every output row needs its own
    /// activation sign pattern — the per-row serving wall (~1 tok/s).
    Rows,
    /// Per-column RHT (bit0 + bit3 set): the activation transform is
    /// row-independent, so it is computed ONCE and reused for every row — the
    /// cheap serving path (the col-RHT win this whole track is built around).
    Cols,
}

impl RhtMode {
    /// Decode the serving mode from the two `.strand` v2 header flags.
    #[inline]
    pub fn from_flags(has_rht_seed: bool, rht_cols: bool) -> Self {
        match (has_rht_seed, rht_cols) {
            (false, _) => RhtMode::None,
            (true, false) => RhtMode::Rows,
            (true, true) => RhtMode::Cols,
        }
    }
}

/// Decode a STRAND-encoded tensor to its integer-deterministic Q12 weights
/// (row-major, `out_features * in_features`). Thin, float-free wrapper over the
/// absorbed `strand-quant` integer decode — the bit-identical path.
#[inline]
pub fn decode_q12(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_tensor_fixed(enc, cfg)
}

/// `y = decode(W) · serve(x)` for one projection: the Q12 weights times the
/// activation, with the RHT serving transform applied per `mode`. Mirrors
/// `outlier_mac::matvec_rht`:
/// - `Cols`: transform `x` once with `rht_forward_cols` and reuse for all rows;
/// - `Rows`: replicate `x` per row and transform each with its own sign pattern;
/// - `None`: serve directly.
///
/// `q12` is the decoded weight matrix (`decode_q12`), `rht_seed` is the per-tensor
/// seed from the `.strand` header (ignored when `mode == None`).
pub fn matvec_rht(
    q12: &[i32],
    x: &[f32],
    out_features: usize,
    in_features: usize,
    mode: RhtMode,
    rht_seed: u64,
) -> Vec<f32> {
    assert_eq!(
        q12.len(),
        out_features * in_features,
        "q12 has {} weights, expected out*in = {}",
        q12.len(),
        out_features * in_features
    );
    assert_eq!(
        x.len(),
        in_features,
        "x len {} != in_features {in_features}",
        x.len()
    );
    let inv = q12_to_f32();
    let mut y = vec![0.0f32; out_features];
    match mode {
        RhtMode::Cols => {
            let rcfg = RhtConfig::from_seed(rht_seed);
            let mut tx = x.to_vec();
            rht_forward_cols_inplace(&mut tx, &rcfg, in_features);
            for o in 0..out_features {
                let row = &q12[o * in_features..(o + 1) * in_features];
                let mut acc = 0.0f32;
                for i in 0..in_features {
                    acc += (row[i] as f32) * inv * tx[i];
                }
                y[o] = acc;
            }
        }
        RhtMode::Rows => {
            let rcfg = RhtConfig::from_seed(rht_seed);
            let mut x_rht = Vec::with_capacity(out_features * in_features);
            for _ in 0..out_features {
                x_rht.extend_from_slice(x);
            }
            rht_forward_rows_inplace(&mut x_rht, &rcfg, in_features);
            for o in 0..out_features {
                let row = &q12[o * in_features..(o + 1) * in_features];
                let xr = &x_rht[o * in_features..(o + 1) * in_features];
                let mut acc = 0.0f32;
                for i in 0..in_features {
                    acc += (row[i] as f32) * inv * xr[i];
                }
                y[o] = acc;
            }
        }
        RhtMode::None => {
            for o in 0..out_features {
                let row = &q12[o * in_features..(o + 1) * in_features];
                let mut acc = 0.0f32;
                for i in 0..in_features {
                    acc += (row[i] as f32) * inv * x[i];
                }
                y[o] = acc;
            }
        }
    }
    y
}

/// Apply 1%-outlier OVERWRITES (not adds) onto decoded Q12 weights in place:
/// `q12[idx] = val`. Mirrors `outlier_mac` (`w[i] = v`) — the sparse top-|w|
/// pre-RHT values restored after decode. `outliers` is `(flat_index, q12_value)`.
pub fn apply_outlier_overwrites(q12: &mut [i32], outliers: &[(usize, i32)]) {
    for &(idx, v) in outliers {
        if idx < q12.len() {
            q12[idx] = v;
        }
    }
}

/// One decode-ready tensor parsed from a `.strand` v2 archive: the integer
/// `EncodedTensor` plus the metadata needed to decode and serve it.
pub struct StrandTensor {
    /// Tensor name (e.g. `blk.0.ffn_down.weight`).
    pub name: String,
    /// Output features (rows) = `shape[0]`.
    pub out_features: usize,
    /// Input features (cols) = `shape[1]`.
    pub in_features: usize,
    /// Trellis decode config (L / k / block_len / vec_dim) for this tensor.
    pub cfg: TrellisConfig,
    /// Activation-RHT serving mode, from the header flag byte.
    pub rht_mode: RhtMode,
    /// Per-tensor RHT seed from the header (meaningful when `rht_mode != None`).
    pub rht_seed: u64,
    enc: EncodedTensor,
}

impl StrandTensor {
    /// Decode this tensor to its integer-deterministic Q12 weights.
    pub fn decode_q12(&self) -> Vec<i32> {
        decode_q12(&self.enc, &self.cfg)
    }

    /// `y = decode(W) · serve(x)` — decode then serve with this tensor's RHT mode.
    /// Convenience wrapper; a caller serving many tokens should `decode_q12` once
    /// and reuse it across `matvec_rht` calls rather than re-decoding per token.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        let q12 = self.decode_q12();
        matvec_rht(
            &q12,
            x,
            self.out_features,
            self.in_features,
            self.rht_mode,
            self.rht_seed,
        )
    }
}

/// Parse a `.strand` v2 archive (the whole file's bytes) into decode-ready
/// tensors. Reads the lean header (for the per-tensor `rht_cols` flag, which the
/// payload parse does not carry) and the SDSQ-applied tensor payloads, then zips
/// them by index. Errors on a header/payload count mismatch or a non-2-D tensor.
pub fn read_strand(buf: &[u8]) -> Result<Vec<StrandTensor>, String> {
    let header = strand_quant::format::read_strand_v2_header(buf)?;
    let owned = strand_quant::sideinfo_wire::read_strand_v2_applied(buf)?;
    if header.tensors.len() != owned.len() {
        return Err(format!(
            "strand reader: header lists {} tensors but payload has {}",
            header.tensors.len(),
            owned.len()
        ));
    }
    let mut out = Vec::with_capacity(owned.len());
    for (h, t) in header.tensors.into_iter().zip(owned) {
        if h.shape.len() < 2 {
            return Err(format!(
                "strand reader: tensor {:?} is not 2-D (shape {:?})",
                h.name, h.shape
            ));
        }
        let out_features = h.shape[0] as usize;
        let in_features = h.shape[1] as usize;
        let mut cfg = TrellisConfig::new(
            t.base.l_bits as u32,
            t.base.k_bits as u32,
            t.block_len as usize,
        );
        cfg.vec_dim = (t.base.vec_dim as u32).max(1);
        out.push(StrandTensor {
            name: h.name,
            out_features,
            in_features,
            cfg,
            rht_mode: RhtMode::from_flags(h.has_rht_seed, h.rht_cols),
            rht_seed: h.rht_seed,
            enc: t.base.enc,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::encode::encode_tensor;

    fn synth_w(n: usize) -> Vec<f32> {
        (0..n).map(|k| ((k as f32) * 0.013).sin() * 0.1).collect()
    }
    fn synth_x(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.07).cos()).collect()
    }

    #[test]
    fn decode_is_deterministic_and_matches_float_decode() {
        let (out_f, in_f) = (4usize, 256usize);
        let w = synth_w(out_f * in_f);
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&w, &cfg);

        // Integer decode is deterministic (the moat).
        let q12a = decode_q12(&enc, &cfg);
        let q12b = decode_q12(&enc, &cfg);
        assert_eq!(q12a, q12b, "Q12 decode must be deterministic");
        assert_eq!(q12a.len(), out_f * in_f);

        // matvec_rht(None) must equal serving strand-quant's own float decode.
        let x = synth_x(in_f);
        let y = matvec_rht(&q12a, &x, out_f, in_f, RhtMode::None, 0);
        let wf = strand_quant::decode::decode_tensor(&enc, &cfg);
        for o in 0..out_f {
            let mut acc = 0.0f32;
            for i in 0..in_f {
                acc += wf[o * in_f + i] * x[i];
            }
            assert!(
                (y[o] - acc).abs() <= 1e-4 * (1.0 + acc.abs()),
                "row {o}: q12 matvec {} vs float-decode matvec {}",
                y[o],
                acc
            );
        }
    }

    #[test]
    fn col_rht_one_transform_serves_all_rows() {
        // The §4 serving contract dismantle's kernel must honour:
        //   <W_row, rht_forward_cols(x)> == <rht_inverse_cols(W_row), x>
        // i.e. serving column-rotated weights with ONE activation transform equals
        // the un-rotated matvec. Proven here on strand-quant's own RHT primitives,
        // inside dismantle's tree (mirrors outlier_mac's col_rht test).
        let (out_f, in_f) = (5usize, 128usize);
        let q12: Vec<i32> = (0..out_f * in_f)
            .map(|k| ((k.wrapping_mul(1103515245).wrapping_add(12345)) % 2048) as i32 - 1024)
            .collect();
        let x = synth_x(in_f);
        let seed = strand_quant::gate_utils::rht_seed_for("blk.0.ffn_down.weight");

        let y_serve = matvec_rht(&q12, &x, out_f, in_f, RhtMode::Cols, seed);

        let inv = q12_to_f32();
        let rcfg = RhtConfig::from_seed(seed);
        let mut y_ref = vec![0.0f32; out_f];
        for o in 0..out_f {
            let mut wr: Vec<f32> = q12[o * in_f..(o + 1) * in_f]
                .iter()
                .map(|&q| q as f32 * inv)
                .collect();
            rht_inverse_cols_inplace(&mut wr, &rcfg, in_f);
            let mut acc = 0.0f32;
            for i in 0..in_f {
                acc += wr[i] * x[i];
            }
            y_ref[o] = acc;
        }
        for o in 0..out_f {
            assert!(
                (y_serve[o] - y_ref[o]).abs() <= 1e-3 * (1.0 + y_ref[o].abs()),
                "row {o}: col-RHT serve {} vs un-rotated ref {}",
                y_serve[o],
                y_ref[o]
            );
        }
    }

    #[test]
    fn outlier_overwrites_replace_not_add() {
        let mut q12 = vec![10i32; 8];
        apply_outlier_overwrites(&mut q12, &[(2, -500), (5, 999)]);
        assert_eq!(q12[2], -500, "outlier must overwrite, not add");
        assert_eq!(q12[5], 999);
        assert_eq!(q12[0], 10, "non-outlier untouched");
    }

    #[test]
    fn strand_file_round_trip_preserves_q12() {
        use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
        let (out_f, in_f) = (6usize, 256usize);
        let w = synth_w(out_f * in_f);
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&w, &cfg);
        // Reference: decode straight from the in-memory EncodedTensor.
        let q12_direct = decode_q12(&enc, &cfg);

        // Write a real `.strand` v2 archive, then read it back through the reader.
        let shape = [out_f as u64, in_f as u64];
        let packed = PackedTensorV2 {
            base: PackedTensor {
                name: "blk.0.ffn_down.weight",
                shape: &shape,
                rht_seed: 0,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc: &enc,
            },
            block_len: cfg.block_len as u32,
        };
        let bytes = write_strand_v2(&[packed], [0u8; 32], true).expect("write_strand_v2");

        let tensors = read_strand(&bytes).expect("read_strand");
        assert_eq!(tensors.len(), 1);
        let t = &tensors[0];
        assert_eq!(t.name, "blk.0.ffn_down.weight");
        assert_eq!((t.out_features, t.in_features), (out_f, in_f));
        assert_eq!(t.rht_mode, RhtMode::None);

        // The integer decode read back from the wire is BIT-IDENTICAL to the
        // direct decode — the determinism moat survives the file round-trip.
        assert_eq!(t.decode_q12(), q12_direct, "file decode != direct decode");

        // And serving through the reader matches the module-level matvec.
        let x = synth_x(in_f);
        assert_eq!(
            t.matvec(&x),
            matvec_rht(&q12_direct, &x, out_f, in_f, RhtMode::None, 0)
        );
    }
}
