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
//!
//! ## What today's baked artifacts actually exercise
//!
//! The serving surface here covers all three [`RhtMode`]s and the 1%-outlier OUTL
//! section, but the artifacts the baker emits today only drive a subset of it — the
//! rest is reference code the Metal kernel will be gated against once those modes
//! ship:
//!
//! - **`RhtMode::None`** — the mode every baked `.tq` uses today. Decode → optional
//!   OUTL overwrite → plain dot. Fully exercised, including OUTL, by the unit tests.
//! - **`RhtMode::Cols`** — the cheap-serving column-RHT path (one activation
//!   transform reused across all rows). Exercised end-to-end *through a real archive*
//!   by [`tests::col_rht_file_round_trip_serves_unrotated`]; not yet emitted by the
//!   baker, so it is reference-only at the artifact level.
//! - **`RhtMode::Rows`** — per-row RHT, the ~1 tok/s serving wall. **Eval-only**: it
//!   materialises an `out_features * in_features` activation buffer (see the `NOTE`
//!   on [`matvec_rht`]) and no baked artifact uses it.
//! - **OUTL** — the 1%-outlier overwrite. Parsed by [`read_strand`] when present and
//!   applied by [`StrandTensor::matvec`] / [`StrandTensor::decode_q12`]. Correct for
//!   every mode (outliers live in the un-rotated weight domain and are applied there
//!   — see [`StrandTensor::matvec`]), but baked OUTL `.tq`s are `RhtMode::None` today,
//!   which is the only path with a file-level test. See the q12-quantisation caveat
//!   on [`StrandTensor::outliers`].

use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::EncodedTensor;
use strand_quant::outlier_wire::read_outl_bytes;
use strand_quant::rht::{
    rht_forward_cols_inplace, rht_forward_rows_inplace, rht_inverse_cols_inplace,
    rht_inverse_rows_inplace, RhtConfig,
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
/// seed from the `.strand` header (ignored when `mode == None`). This serves the
/// *bulk* weights only — OUTL outlier overwrites are applied by [`StrandTensor`],
/// which knows the un-rotated domain (this free function takes raw rotated q12).
///
/// NOTE (Rows is eval-only — O(rows·cols) activation buffer): the `Rows` branch
/// allocates an `out_features * in_features` f32 buffer (replicate `x` per row, then
/// `rht_forward_rows`) — hundreds of MB on a real tensor. It is **not** streamed per
/// row because the per-row sign pattern is `sign_at(seed, o*in_features + j)` keyed
/// on the *global* flat offset, and `strand-quant` exposes no public primitive that
/// takes that offset (`rht_forward_inplace` only reproduces row 0's signs;
/// `sign_at`/`fwht_inplace` are private). Streaming would mean re-deriving that
/// private sign schedule by hand — a correctness hazard the audit explicitly says not
/// to guess at. Since `Rows` is the ~1 tok/s serving wall that no baked artifact uses
/// (the whole track targets `Cols`), the O(rows·cols) buffer is left as-is: it is the
/// eval/reference path, not a hot one. If `Rows` ever needs to serve cheaply, the fix
/// is an offset-taking sign API upstream in `strand-quant::rht`, not a hand-rolled
/// copy of the sign loop here.
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
///
/// `flat_index` is the **un-rotated** row-major position (`row*in_features + col`),
/// so this is only equivalent to the reference overwrite when the Q12 it patches is
/// itself in the un-rotated domain — i.e. `RhtMode::None`, or after an explicit
/// `rht_inverse`. [`StrandTensor::matvec`] guarantees that domain before calling in.
pub fn apply_outlier_overwrites(q12: &mut [i32], outliers: &[(usize, i32)]) {
    for &(idx, v) in outliers {
        if idx < q12.len() {
            q12[idx] = v;
        }
    }
}

/// One decode-ready tensor parsed from a `.strand` v2 archive: the integer
/// `EncodedTensor` plus the metadata needed to decode and serve it.
#[derive(Debug)]
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
    /// 1%-outlier overwrites for this tensor (`(flat_index, q12_value)`), parsed from
    /// the archive's OUTL section if it has one; empty otherwise. Indices are
    /// un-rotated row-major positions; values are the OUTL float restored value
    /// re-quantised to Q12 (`round(val * 2^QUANTILE_SHIFT)`).
    ///
    /// Caveat (q12 re-quantisation): OUTL stores its restored values as arbitrary
    /// floats (`code / levels * omax`), not Q12 multiples. Re-quantising to Q12 to
    /// fit the integer overwrite contract (and the eventual Q12 GPU kernel) loses up
    /// to ½ LSB (`1 / 2^QUANTILE_SHIFT`) per outlier vs the reference float overwrite
    /// in `outlier_mac::patched_weights`. This keeps the determinism moat (everything
    /// the kernel touches is integer Q12) at the cost of that bounded rounding; the
    /// OUTL serving test asserts agreement only to within this Q12 grid.
    pub outliers: Vec<(usize, i32)>,
    enc: EncodedTensor,
}

impl StrandTensor {
    /// Decode this tensor to its integer-deterministic Q12 weights, with any OUTL
    /// outlier overwrites applied **in the un-rotated weight domain**.
    ///
    /// For `RhtMode::None` the raw decode is already un-rotated, so the overwrite is a
    /// direct `q12[idx] = val` (via [`apply_outlier_overwrites`]). For `Cols`/`Rows`
    /// the raw decode is rotated, so a direct overwrite would corrupt the q-domain;
    /// this inverse-rotates to the un-rotated domain, overwrites there, and rotates
    /// back so the returned q12 is still the rotated weights the serving path expects
    /// (i.e. the OUTL-patched equivalent of [`decode_q12_raw`]). With no outliers this
    /// is exactly [`decode_q12_raw`].
    pub fn decode_q12(&self) -> Vec<i32> {
        let mut q12 = self.decode_q12_raw();
        if self.outliers.is_empty() {
            return q12;
        }
        match self.rht_mode {
            RhtMode::None => {
                // Raw decode is the un-rotated domain — overwrite directly.
                apply_outlier_overwrites(&mut q12, &self.outliers);
            }
            RhtMode::Cols | RhtMode::Rows => {
                // Outliers live in the un-rotated domain; the rotated q12 does not.
                // Round-trip through floats: inverse-rotate → overwrite → re-rotate.
                let inv = q12_to_f32();
                let scale = (1u32 << strand_quant::QUANTILE_SHIFT) as f32;
                let mut w: Vec<f32> = q12.iter().map(|&q| q as f32 * inv).collect();
                let rcfg = RhtConfig::from_seed(self.rht_seed);
                match self.rht_mode {
                    RhtMode::Cols => rht_inverse_cols_inplace(&mut w, &rcfg, self.in_features),
                    RhtMode::Rows => rht_inverse_rows_inplace(&mut w, &rcfg, self.in_features),
                    RhtMode::None => unreachable!(),
                }
                for &(idx, v) in &self.outliers {
                    if idx < w.len() {
                        w[idx] = v as f32 * inv;
                    }
                }
                match self.rht_mode {
                    RhtMode::Cols => rht_forward_cols_inplace(&mut w, &rcfg, self.in_features),
                    RhtMode::Rows => rht_forward_rows_inplace(&mut w, &rcfg, self.in_features),
                    RhtMode::None => unreachable!(),
                }
                for (q, &wv) in q12.iter_mut().zip(w.iter()) {
                    *q = (wv * scale).round() as i32;
                }
            }
        }
        q12
    }

    /// Pure trellis decode to Q12, **without** OUTL overwrites — the raw rotated
    /// weights the GPU bitslice kernel must reproduce bit-for-bit. Callers serving
    /// many tokens that have already folded outliers elsewhere use this; most callers
    /// want [`decode_q12`](Self::decode_q12).
    pub fn decode_q12_raw(&self) -> Vec<i32> {
        decode_q12(&self.enc, &self.cfg)
    }

    /// `y = decode(W) · serve(x)` — decode then serve with this tensor's RHT mode,
    /// including any OUTL outlier overwrites.
    ///
    /// With no outliers this is the plain `decode_q12_raw` → [`matvec_rht`] path
    /// (the cheap col-RHT serving win is preserved). With outliers present it serves
    /// in the **un-rotated** domain — the only domain the OUTL overwrite is defined in
    /// — by inverse-rotating the bulk weights once, overwriting, and doing a plain
    /// dot. For `Cols`/`Rows` that trades the per-token cheap-serving path for
    /// correctness; today this only fires for `RhtMode::None` (the only mode baked),
    /// where un-rotated == rotated and there is no such trade.
    ///
    /// Convenience wrapper; a caller serving many tokens should hoist the decode and
    /// reuse it across calls rather than re-decoding per token.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        if self.outliers.is_empty() {
            let q12 = self.decode_q12_raw();
            return matvec_rht(
                &q12,
                x,
                self.out_features,
                self.in_features,
                self.rht_mode,
                self.rht_seed,
            );
        }

        // OUTL present: serve in the un-rotated weight domain (where the overwrites are
        // defined). For None this is a direct overwrite on the already-un-rotated q12;
        // for Cols/Rows we inverse-rotate the bulk float weights first.
        let inv = q12_to_f32();
        if self.rht_mode == RhtMode::None {
            let mut q12 = self.decode_q12_raw();
            apply_outlier_overwrites(&mut q12, &self.outliers);
            return matvec_rht(
                &q12,
                x,
                self.out_features,
                self.in_features,
                RhtMode::None,
                self.rht_seed,
            );
        }

        let q12 = self.decode_q12_raw();
        let mut w: Vec<f32> = q12.iter().map(|&q| q as f32 * inv).collect();
        let rcfg = RhtConfig::from_seed(self.rht_seed);
        match self.rht_mode {
            RhtMode::Cols => rht_inverse_cols_inplace(&mut w, &rcfg, self.in_features),
            RhtMode::Rows => rht_inverse_rows_inplace(&mut w, &rcfg, self.in_features),
            RhtMode::None => unreachable!(),
        }
        for &(idx, v) in &self.outliers {
            if idx < w.len() {
                w[idx] = v as f32 * inv;
            }
        }
        let mut y = vec![0.0f32; self.out_features];
        for o in 0..self.out_features {
            let row = &w[o * self.in_features..(o + 1) * self.in_features];
            let mut acc = 0.0f32;
            for i in 0..self.in_features {
                acc += row[i] * x[i];
            }
            y[o] = acc;
        }
        y
    }
}

/// Parse a `.strand` v2 archive (the whole file's bytes) into decode-ready
/// tensors. Reads the lean header (for the per-tensor `rht_cols` flag, which the
/// payload parse does not carry), the SDSQ-applied tensor payloads, and the optional
/// OUTL outlier section, then zips them by index. Errors on a header/payload count
/// mismatch, a non-2-D tensor, or an `enc.total` that disagrees with the shape (F7
/// fail-fast: a truncated/malformed `.tq` is caught here, not as a later panic in
/// [`matvec_rht`]'s `out_features * in_features` slice).
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
    // Optional 1%-outlier section. Strict parse: a present-but-corrupt OUTL is an
    // error here rather than a silent "no outliers" (which would serve wrong weights).
    let outl = read_outl_bytes(buf, true)?;
    if let Some(sec) = &outl {
        if sec.tensors.len() != header.tensors.len() {
            return Err(format!(
                "strand reader: OUTL lists {} tensors but header has {}",
                sec.tensors.len(),
                header.tensors.len()
            ));
        }
    }
    let q12_scale = (1u32 << strand_quant::QUANTILE_SHIFT) as f32;
    let mut out = Vec::with_capacity(owned.len());
    for (ti, (h, t)) in header.tensors.into_iter().zip(owned).enumerate() {
        if h.shape.len() < 2 {
            return Err(format!(
                "strand reader: tensor {:?} is not 2-D (shape {:?})",
                h.name, h.shape
            ));
        }
        let out_features = h.shape[0] as usize;
        let in_features = h.shape[1] as usize;
        // F7: the trellis payload must hold exactly out*in weights. A truncated or
        // shape-mismatched archive trips here with the tensor name, before any later
        // `q12[o*in..(o+1)*in]` slice panics in matvec_rht.
        let expected = out_features.checked_mul(in_features).ok_or_else(|| {
            format!(
                "strand reader: tensor {:?} shape {out_features}x{in_features} overflows usize",
                h.name
            )
        })?;
        if t.base.enc.total != expected {
            return Err(format!(
                "strand reader: tensor {:?} has {} decoded weights but shape {out_features}x{in_features} needs {expected} (truncated or malformed .tq)",
                h.name, t.base.enc.total
            ));
        }
        let mut cfg = TrellisConfig::new(
            t.base.l_bits as u32,
            t.base.k_bits as u32,
            t.block_len as usize,
        );
        cfg.vec_dim = (t.base.vec_dim as u32).max(1);
        // F2: fold any OUTL record for this tensor into `(flat_index, q12_value)`.
        // OUTL stores restored values as floats; re-quantise to Q12 to keep the
        // integer-overwrite contract (see `StrandTensor::outliers` for the caveat).
        let outliers: Vec<(usize, i32)> = outl
            .as_ref()
            .and_then(|sec| sec.tensors.get(ti))
            .and_then(|w| w.as_ref())
            .map(|wire| {
                wire.dequant_vals()
                    .map(|(idx, val)| (idx as usize, (val * q12_scale).round() as i32))
                    .collect()
            })
            .unwrap_or_default();
        out.push(StrandTensor {
            name: h.name,
            out_features,
            in_features,
            cfg,
            rht_mode: RhtMode::from_flags(h.has_rht_seed, h.rht_cols),
            rht_seed: h.rht_seed,
            outliers,
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

    /// F7: a tensor whose declared shape needs more weights than the payload holds
    /// must fail in `read_strand` (named, as an Err), not panic later in matvec_rht.
    /// We forge the mismatch by editing the on-disk shape `shape[1]` to be one larger
    /// than the baked tensor actually encodes.
    #[test]
    fn reader_fails_fast_on_shape_total_mismatch() {
        use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
        let (out_f, in_f) = (4usize, 256usize);
        let w = synth_w(out_f * in_f);
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&w, &cfg);
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
        let mut bytes = write_strand_v2(&[packed], [0u8; 32], true).expect("write_strand_v2");
        // A clean archive must parse.
        assert!(read_strand(&bytes).is_ok(), "baseline archive should read");

        // Corrupt shape[1]: in_features 256 -> 257 so out*in no longer equals total.
        // Layout per write_strand_v2: [magic4|ver4|hdr_bytes4|n4|flags4|sha32|reserved4]
        // = 56 bytes, then per tensor [name_len4|name|ndim4|dims(8 each)|...]. dims
        // start at 56 + 4 + name_len + 4.
        let name_len = "blk.0.ffn_down.weight".len();
        let dims_at = 56 + 4 + name_len + 4;
        let shape1_at = dims_at + 8; // shape[0] is dims_at..+8, shape[1] is next.
        let bad = 257u64.to_le_bytes();
        bytes[shape1_at..shape1_at + 8].copy_from_slice(&bad);

        let err = read_strand(&bytes).expect_err("shape/total mismatch must Err");
        assert!(
            err.contains("blk.0.ffn_down.weight") && err.contains("malformed"),
            "F7 error must name the tensor and flag malformed: {err}"
        );
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

    /// Task 4 — the FIRST end-to-end exercise of `RhtMode::Cols` through a real
    /// archive: bake column-rotated weights, write them with
    /// `write_strand_v2_rht(.., rht_cols=&[true])`, read them back, assert the reader
    /// derives `RhtMode::Cols`, and assert serving (`.matvec`) equals the un-rotated
    /// spatial reference (decode → `rht_inverse_cols` per row → dot x).
    ///
    /// `has_rht_seed` (flag bit0) is what promotes the tensor from `None` to a
    /// rotated mode; the writer sources it from `enc.has_rht_seed`, NOT from the
    /// `rht_cols` slice (which only sets bit3). So the encoder must stamp
    /// `enc.has_rht_seed = true` for the col mode to be reachable — mirrors how the
    /// strand-decode-kernel `bake_fixture` sets it.
    #[test]
    fn col_rht_file_round_trip_serves_unrotated() {
        use strand_quant::format::{write_strand_v2_rht, PackedTensor, PackedTensorV2};
        let name = "blk.0.ffn_down.weight";
        let (out_f, in_f) = (5usize, 256usize); // in_features 256-aligned (deploy-strict).
        let seed = strand_quant::gate_utils::rht_seed_for(name);

        // Bulk (un-rotated) ground-truth weights, then column-rotate before encoding.
        let bulk = synth_w(out_f * in_f);
        let work = strand_quant::rht::rht_forward_cols(&bulk, &RhtConfig::from_seed(seed), in_f);
        let cfg = TrellisConfig::for_bpw(3.0);
        let mut enc = encode_tensor(&work, &cfg);
        enc.has_rht_seed = true; // promote: flag bit0 (without it the mode is None).

        let shape = [out_f as u64, in_f as u64];
        let packed = PackedTensorV2 {
            base: PackedTensor {
                name,
                shape: &shape,
                rht_seed: seed,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc: &enc,
            },
            block_len: cfg.block_len as u32,
        };
        // rht_cols=&[true] sets bit3; enc.has_rht_seed set bit0 → reader sees Cols.
        let bytes = write_strand_v2_rht(&[packed], [0u8; 32], true, false, &[true])
            .expect("write_strand_v2_rht");

        let tensors = read_strand(&bytes).expect("read_strand");
        assert_eq!(tensors.len(), 1);
        let t = &tensors[0];
        assert_eq!(
            t.rht_mode,
            RhtMode::Cols,
            "col archive must read back as RhtMode::Cols"
        );
        assert_eq!(t.rht_seed, seed, "seed must survive the round trip");
        assert!(t.outliers.is_empty(), "no OUTL section was written");

        // Spatial reference: decode the rotated q12, un-rotate per row with
        // rht_inverse_cols, then dot the raw activation — the contract Cols serving
        // must reproduce with ONE shared rht_forward_cols(x).
        let x = synth_x(in_f);
        let q12 = t.decode_q12();
        let inv = q12_to_f32();
        let rcfg = RhtConfig::from_seed(seed);
        let mut y_ref = vec![0.0f32; out_f];
        for o in 0..out_f {
            let mut wr: Vec<f32> = q12[o * in_f..(o + 1) * in_f]
                .iter()
                .map(|&q| q as f32 * inv)
                .collect();
            rht_inverse_cols_inplace(&mut wr, &rcfg, in_f);
            y_ref[o] = wr.iter().zip(&x).map(|(w, xv)| w * xv).sum();
        }

        let y_serve = t.matvec(&x);
        for o in 0..out_f {
            assert!(
                (y_serve[o] - y_ref[o]).abs() <= 1e-3 * (1.0 + y_ref[o].abs()),
                "row {o}: Cols file-serve {} vs un-rotated ref {}",
                y_serve[o],
                y_ref[o]
            );
        }
    }

    /// Task 2 (F2) — OUTL wired end-to-end through a real archive on the only mode
    /// that is baked today (`RhtMode::None`): bake bulk weights with the top-|w|
    /// entries zeroed, append a real OUTL section restoring them, read the archive,
    /// and assert (a) `read_strand` surfaces the outliers and (b) `.matvec` applies
    /// them. The reference is the un-rotated GEMV over the OUTL-patched weights, with
    /// outlier values on the Q12 grid (`StrandTensor::outliers` re-quantises floats to
    /// Q12, so equality is asserted to within that grid, not bit-exact).
    #[test]
    fn outl_file_round_trip_serves_patched_none() {
        use std::io::Write as _;
        use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
        use strand_quant::outlier_wire::{append_outl, idx_bits_for, OutlierWire};

        let name = "blk.0.ffn_down.weight";
        let (out_f, in_f) = (4usize, 256usize);
        let n = out_f * in_f;
        let gt = synth_w(n);

        // Pick the top-|w| 1% as outliers; quantise them to `ob` bits like the baker.
        let k = ((1.0f64 / 100.0) * n as f64).round().max(1.0) as usize;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            gt[b]
                .abs()
                .partial_cmp(&gt[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let idx: Vec<usize> = order[..k].to_vec();
        let ob = 8u32;
        let omax = idx.iter().fold(0f32, |m, &i| m.max(gt[i].abs())).max(1e-12);
        let levels = ((1i64 << (ob - 1)) - 1) as f32;
        let codes: Vec<i32> = idx
            .iter()
            .map(|&i| (gt[i] / omax * levels).round() as i32)
            .collect();
        // Restored float values exactly as OUTL `dequant_vals` will reproduce them.
        let restored: Vec<f32> = codes.iter().map(|&c| (c as f32) / levels * omax).collect();

        // Bulk weights = ground truth with the outlier positions zeroed (None mode:
        // no rotation, so the bulk IS the encoded tensor).
        let mut bulk = gt.clone();
        for &i in &idx {
            bulk[i] = 0.0;
        }
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&bulk, &cfg);

        let shape = [out_f as u64, in_f as u64];
        let packed = PackedTensorV2 {
            base: PackedTensor {
                name,
                shape: &shape,
                rht_seed: 0,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc: &enc,
            },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2(&[packed], [0u8; 32], true).expect("write_strand_v2");

        // OUTL is appended to a file (no in-memory append API), so round-trip on disk.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "tq_outl_{}_{}.tq",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        {
            let mut f = std::fs::File::create(&path).expect("create temp .tq");
            f.write_all(&buf).expect("write temp .tq");
            f.sync_all().ok();
        }
        let wire = OutlierWire::from_selection(n, idx.clone(), codes.clone(), omax, ob);
        assert_eq!(wire.idx_bits, idx_bits_for(n));
        append_outl(&path, &[Some(wire)]).expect("append outl");
        let bytes = std::fs::read(&path).expect("re-read .tq");
        let _ = std::fs::remove_file(&path);

        // The reader must surface the outliers, on the Q12 grid. OutlierWire sorts
        // entries by index, so build a sorted (index, restored-float) reference to
        // compare against rather than the |w|-sorted selection order.
        let tensors = read_strand(&bytes).expect("read_strand with OUTL");
        let t = &tensors[0];
        assert_eq!(t.rht_mode, RhtMode::None);
        assert_eq!(t.outliers.len(), k, "OUTL must be parsed into `outliers`");
        let scale = (1u32 << strand_quant::QUANTILE_SHIFT) as f32;
        let mut want: Vec<(usize, f32)> =
            idx.iter().copied().zip(restored.iter().copied()).collect();
        want.sort_unstable_by_key(|&(i, _)| i);
        for (&(oi, ov), &(gi, gv)) in t.outliers.iter().zip(want.iter()) {
            assert_eq!(oi, gi, "outlier index mismatch");
            assert_eq!(
                ov,
                (gv * scale).round() as i32,
                "outlier q12 value mismatch"
            );
        }

        // decode_q12() must show the patched values at the outlier positions, and the
        // bulk decode value (NOT the patched one) everywhere else.
        let patched = t.decode_q12();
        let raw = t.decode_q12_raw();
        for &(oi, ov) in &t.outliers {
            assert_eq!(
                patched[oi], ov,
                "decode_q12 must overwrite the outlier position"
            );
        }
        // A position that is NOT an outlier must be untouched vs the raw decode.
        let non_outlier = (0..n).find(|i| !idx.contains(i)).unwrap();
        assert_eq!(
            patched[non_outlier], raw[non_outlier],
            "bulk weight must be unchanged"
        );

        // Serving reference: un-rotated GEMV over the Q12-grid patched weights.
        let x = synth_x(in_f);
        let y_serve = t.matvec(&x);
        let y_ref = matvec_rht(&patched, &x, out_f, in_f, RhtMode::None, 0);
        for o in 0..out_f {
            assert_eq!(
                y_serve[o], y_ref[o],
                "row {o}: OUTL serve must equal patched-q12 GEMV"
            );
        }

        // And the patched serve must differ from the un-patched (bulk-only) serve —
        // proves the OUTL term is actually live, not a no-op.
        let y_bulk = matvec_rht(&raw, &x, out_f, in_f, RhtMode::None, 0);
        let max_delta = (0..out_f)
            .map(|o| (y_serve[o] - y_bulk[o]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-6,
            "OUTL overwrite must change the output (was {max_delta})"
        );
    }

    /// OUTL on a `Cols` archive: outliers live in the un-rotated domain, so the
    /// `Cols` serve must inverse-rotate, overwrite there, and dot — matching a
    /// reference that reconstructs un-rotated patched weights independently. This
    /// guards the `decode_q12`/`matvec` Cols+OUTL domain handling even though no such
    /// artifact is baked today.
    #[test]
    fn outl_cols_serves_in_unrotated_domain() {
        use std::io::Write as _;
        use strand_quant::format::{write_strand_v2_rht, PackedTensor, PackedTensorV2};
        use strand_quant::outlier_wire::{append_outl, OutlierWire};

        let name = "blk.0.ffn_down.weight";
        let (out_f, in_f) = (4usize, 256usize);
        let n = out_f * in_f;
        let seed = strand_quant::gate_utils::rht_seed_for(name);
        let gt = synth_w(n);

        // Outlier selection (top-|w| 1%), quantised like the baker.
        let k = ((1.0f64 / 100.0) * n as f64).round().max(1.0) as usize;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            gt[b]
                .abs()
                .partial_cmp(&gt[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let idx: Vec<usize> = order[..k].to_vec();
        let ob = 8u32;
        let omax = idx.iter().fold(0f32, |m, &i| m.max(gt[i].abs())).max(1e-12);
        let levels = ((1i64 << (ob - 1)) - 1) as f32;
        let codes: Vec<i32> = idx
            .iter()
            .map(|&i| (gt[i] / omax * levels).round() as i32)
            .collect();
        let inv = q12_to_f32();

        // Un-rotated bulk (outliers zeroed), column-rotated for encoding.
        let mut bulk = gt.clone();
        for &i in &idx {
            bulk[i] = 0.0;
        }
        let rcfg = RhtConfig::from_seed(seed);
        let work = strand_quant::rht::rht_forward_cols(&bulk, &rcfg, in_f);
        let cfg = TrellisConfig::for_bpw(3.0);
        let mut enc = encode_tensor(&work, &cfg);
        enc.has_rht_seed = true;

        let shape = [out_f as u64, in_f as u64];
        let packed = PackedTensorV2 {
            base: PackedTensor {
                name,
                shape: &shape,
                rht_seed: seed,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc: &enc,
            },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2_rht(&[packed], [0u8; 32], true, false, &[true])
            .expect("write_strand_v2_rht");

        let mut path = std::env::temp_dir();
        path.push(format!(
            "tq_outl_cols_{}_{}.tq",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        {
            let mut f = std::fs::File::create(&path).expect("create temp .tq");
            f.write_all(&buf).expect("write temp .tq");
            f.sync_all().ok();
        }
        let wire = OutlierWire::from_selection(n, idx.clone(), codes, omax, ob);
        append_outl(&path, &[Some(wire)]).expect("append outl");
        let bytes = std::fs::read(&path).expect("re-read .tq");
        let _ = std::fs::remove_file(&path);

        let tensors = read_strand(&bytes).expect("read_strand cols+OUTL");
        let t = &tensors[0];
        assert_eq!(t.rht_mode, RhtMode::Cols);
        assert_eq!(t.outliers.len(), k);

        // Independent reference: rebuild un-rotated patched weights from the rotated
        // raw decode, overwrite outliers on the Q12 grid, then plain GEMV.
        let raw = t.decode_q12_raw();
        let mut w: Vec<f32> = raw.iter().map(|&q| q as f32 * inv).collect();
        rht_inverse_cols_inplace(&mut w, &rcfg, in_f);
        for &(oi, ov) in &t.outliers {
            w[oi] = ov as f32 * inv;
        }
        let x = synth_x(in_f);
        let mut y_ref = vec![0.0f32; out_f];
        for o in 0..out_f {
            y_ref[o] = w[o * in_f..(o + 1) * in_f]
                .iter()
                .zip(&x)
                .map(|(wv, xv)| wv * xv)
                .sum();
        }

        let y_serve = t.matvec(&x);
        for o in 0..out_f {
            assert!(
                (y_serve[o] - y_ref[o]).abs() <= 1e-3 * (1.0 + y_ref[o].abs()),
                "row {o}: Cols+OUTL serve {} vs un-rotated patched ref {}",
                y_serve[o],
                y_ref[o]
            );
        }
    }
}
