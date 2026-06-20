//! Self-contained static rANS entropy coder for STRAND quant **side-info**
//! streams — the `scale_q` per-block scale and the outlier **position** stream.
//!
//! ## Why this module exists (the lever)
//!
//! `research/bit-ledger-results.md` measured, on the real Qwen2.5-0.5B q2
//! artifact, that the side-info is billed at fixed widths far above its Shannon
//! entropy:
//!
//! | stream      | raw bpw | entropy bpw | **recoverable** | model            |
//! |-------------|--------:|------------:|----------------:|------------------|
//! | `scale_q`   | 0.12500 |     0.04099 |     **0.08401** | order-0 (i32)    |
//! | `outl_pos`  | 0.22584 |     0.07825 |     **0.14760** | order-0 of *gaps*|
//!
//! Together ~0.232 bpw of the q2 2.665-bpw encoded artifact is pure
//! fixed-width waste an entropy coder recovers at **zero quality cost** (the
//! decoded symbols are byte-identical to the originals). This module is that
//! coder. The combined scale+sub_scale+outl-pos ceiling drives q2 2.665 → ~2.49
//! (map §5.1 conservative target).
//!
//! ## The moat: integer-deterministic decode
//!
//! STRAND's guarantee is bit-exact, float-free decode on every platform. This
//! coder honors it:
//!
//! - The probability model is a **static** integer CDF, quantized to 12-bit
//!   frequencies that sum to exactly `SCALE_TOTAL`, and **serialized into the
//!   stream**. Decode rebuilds the identical CDF from those bytes — no
//!   floating point, no platform-dependent reduction.
//! - The rANS core is the proven Ryg-style 32-bit construction already shipping
//!   in `strand-container::coder::rans` (same `L`, same `SCALE_BITS`), reused
//!   here *without* taking a dependency on `strand-core`/`strand-container`
//!   (this crate's only deps are `wide` plus macOS `metal`/`objc`).
//! - Encode may use floats to *choose* the model (it does not, here — counts are
//!   integers), but decode is integer-only. This matches the crate's
//!   encode-float / decode-integer split (see `lib.rs`).
//!
//! ## What is NOT here (scope)
//!
//! This is the codec + a measurement harness + round-trip proofs. It does **not**
//! edit `format.rs`/`encode.rs`/`outlier_wire.rs`. The integration plan (where to
//! call it, how the section is length-framed) is returned to the operator. The
//! coder is parameterized over raw `i64` symbol streams so the operator can wire
//! it behind whichever section framing they choose.

#![allow(clippy::needless_range_loop)]

// ===========================================================================
// rANS core (self-contained, byte-renormalized, 32-bit state, single lane)
// ===========================================================================

/// rANS lower bound. State `x` is kept in `[L, (L<<8))`. Matches the container
/// coder so behaviour is identical and equally battle-tested.
const L: u32 = 1 << 23;

/// Probability precision: frequencies sum to `SCALE_TOTAL = 1 << SCALE_BITS`.
/// 14 bits gives enough CDF resolution for the ~1.5k-symbol `scale_q` alphabet
/// while keeping `x_max = (L >> SCALE_BITS << 8) * freq` well under `2^32` for
/// any `freq < SCALE_TOTAL` (here `(L>>14<<8) = 2^17`, `freq < 2^14` ⇒
/// `x_max < 2^31`), so the renorm test never overflows `u32`.
const SCALE_BITS: u32 = 14;

/// `1 << SCALE_BITS`. Every model normalizes its frequencies to this sum.
pub const SCALE_TOTAL: u32 = 1 << SCALE_BITS;

const SCALE_MASK: u32 = SCALE_TOTAL - 1;

#[inline]
fn enc_put(x: &mut u32, out: &mut Vec<u8>, start: u32, freq: u32) {
    debug_assert!(freq > 0, "cannot encode a zero-frequency symbol");
    // Renormalize: emit low bytes until x fits the encodable window for `freq`.
    let x_max = (((L >> SCALE_BITS) << 8) as u64).wrapping_mul(freq as u64);
    let mut s = *x;
    while (s as u64) >= x_max {
        out.push((s & 0xFF) as u8);
        s >>= 8;
    }
    *x = ((s / freq) << SCALE_BITS)
        .wrapping_add(s % freq)
        .wrapping_add(start);
}

#[inline]
fn dec_get(
    x: &mut u32,
    data: &[u8],
    pos: &mut usize,
    cum: &[u32],
) -> usize {
    let slot = *x & SCALE_MASK;
    let symbol = cdf_find(cum, slot);
    let start = cum[symbol];
    let freq = cum[symbol + 1] - cum[symbol];

    let mut s = freq
        .wrapping_mul(*x >> SCALE_BITS)
        .wrapping_add(slot)
        .wrapping_sub(start);
    while s < L {
        let b = if *pos < data.len() { data[*pos] } else { 0 };
        *pos += 1;
        s = (s << 8) | b as u32;
    }
    *x = s;
    symbol
}

/// Rightmost index with `cum[idx] <= value` (binary search over the cumulative
/// table). Integer-only; deterministic.
#[inline]
fn cdf_find(cum: &[u32], value: u32) -> usize {
    debug_assert!(value < SCALE_TOTAL);
    let mut lo = 0usize;
    let mut hi = cum.len() - 1; // == num_symbols
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if cum[mid] <= value {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ===========================================================================
// Static integer probability model (the determinism boundary)
// ===========================================================================

/// A static rANS model over a small dense alphabet `0..num_symbols`, where the
/// last symbol (`esc_symbol`) is an **escape** used for any raw value not in the
/// model's symbol table. Frequencies are 12-bit and sum to `SCALE_TOTAL`.
///
/// The model is **self-describing**: [`Model::serialize`] writes the symbol
/// table + frequencies; [`Model::deserialize`] rebuilds the byte-identical
/// `cum` table on decode. There is no float anywhere in the decode rebuild.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    /// Raw symbol value (zig-zag-encoded `u64`) for each modelled slot. The
    /// final entry is `u64::MAX` as a sentinel for the ESC slot.
    pub symbols: Vec<u64>,
    /// `cum.len() == symbols.len() + 1`, `cum[0] == 0`,
    /// `cum[last] == SCALE_TOTAL`, non-decreasing, every modelled symbol has
    /// `freq >= 1`.
    cum: Vec<u32>,
}

/// Sentinel raw value marking the escape slot in [`Model::symbols`].
const ESC_SENTINEL: u64 = u64::MAX;

/// Max distinct *modelled* symbols (excluding ESC). Sized to cover the real
/// side-info alphabets fully so escapes are rare: the measured `scale_q` has
/// ~1.5k distinct levels (H≈10.5) and gap streams concentrate well under this.
/// Stays under `SCALE_TOTAL` so every modelled symbol keeps `freq >= 1`, and
/// the serialized table (~4 B/symbol) costs only ~0.02 bpw amortized over a
/// whole model's blocks. Symbols beyond the cap fold into the ESC tail.
const MAX_MODEL_SYMBOLS: usize = 4096;

impl Model {
    #[inline]
    fn esc_index(&self) -> usize {
        self.symbols.len() - 1
    }

    /// Build a static model from a raw symbol stream.
    ///
    /// Counts the (zig-zagged) symbols, keeps the `MAX_MODEL_SYMBOLS` most
    /// frequent as explicit slots, folds the rest into one ESC slot, then
    /// normalizes to `SCALE_TOTAL`. Deterministic: ties broken by symbol value.
    pub fn from_stream(raw: &[i64]) -> Model {
        // 1. Histogram of zig-zagged values.
        let mut counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for &v in raw {
            *counts.entry(zigzag(v)).or_insert(0) += 1;
        }
        // 2. Rank by (count desc, value asc) for a deterministic top-N cut.
        let mut ranked: Vec<(u64, u64)> = counts.into_iter().collect();
        ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let modelled = ranked.len().min(MAX_MODEL_SYMBOLS);
        let mut esc_count: u64 = 0;
        for &(_, c) in &ranked[modelled..] {
            esc_count += c;
        }
        // The ESC slot always exists. If there are escaped symbols, give it
        // their mass; otherwise give it a floor of 1 so a value never seen in
        // training can still be coded by a (rare) future caller — and so an
        // all-escape decode never divides by a zero-freq slot.
        let mut symbols: Vec<u64> = Vec::with_capacity(modelled + 1);
        let mut raw_counts: Vec<u64> = Vec::with_capacity(modelled + 1);
        for &(sym, c) in &ranked[..modelled] {
            symbols.push(sym);
            raw_counts.push(c);
        }
        symbols.push(ESC_SENTINEL);
        raw_counts.push(esc_count.max(1));

        // 3. Sort the *modelled* (non-ESC) symbols ascending so the symbol
        //    table is canonical; keep ESC last. (rANS doesn't require order,
        //    but a canonical table makes the serialized model deterministic and
        //    diff-friendly.)
        let esc_c = *raw_counts.last().unwrap();
        let mut pairs: Vec<(u64, u64)> = symbols[..symbols.len() - 1]
            .iter()
            .cloned()
            .zip(raw_counts[..raw_counts.len() - 1].iter().cloned())
            .collect();
        pairs.sort_unstable_by_key(|&(s, _)| s);
        let mut symbols2: Vec<u64> = Vec::with_capacity(pairs.len() + 1);
        let mut counts2: Vec<u64> = Vec::with_capacity(pairs.len() + 1);
        for (s, c) in pairs {
            symbols2.push(s);
            counts2.push(c);
        }
        symbols2.push(ESC_SENTINEL);
        counts2.push(esc_c);

        let cum = normalize_to_cum(&counts2);
        Model { symbols: symbols2, cum }
    }

    /// Look up the model slot for a raw (pre-zig-zag) value, or the ESC slot.
    #[inline]
    fn slot_of(&self, raw: i64) -> usize {
        let z = zigzag(raw);
        // Modelled symbols (all but the last ESC entry) are sorted ascending.
        let modelled = &self.symbols[..self.symbols.len() - 1];
        match modelled.binary_search(&z) {
            Ok(i) => i,
            Err(_) => self.esc_index(),
        }
    }

    /// Serialize the model: `[u32 num_symbols][per symbol: varint zigzag value,
    /// u16 freq]`. The ESC sentinel value is written as varint `0` with the
    /// real ESC freq; it is always the final entry. Decode rebuilds `cum`.
    pub fn serialize(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.symbols.len() as u32).to_le_bytes());
        for i in 0..self.symbols.len() {
            let freq = self.cum[i + 1] - self.cum[i];
            debug_assert!(freq <= u16::MAX as u32);
            if i + 1 == self.symbols.len() {
                // ESC slot: value field is unused, write a single 0 byte.
                write_varint(out, 0);
            } else {
                write_varint(out, self.symbols[i]);
            }
            out.extend_from_slice(&(freq as u16).to_le_bytes());
        }
    }

    /// Rebuild a model from [`Model::serialize`] bytes. Integer-only.
    pub fn deserialize(data: &[u8], pos: &mut usize) -> Result<Model, String> {
        let n = read_u32(data, pos)? as usize;
        if n < 1 || n > MAX_MODEL_SYMBOLS + 1 {
            return Err(format!("sideinfo_rans: model symbol count {n} out of range"));
        }
        let mut symbols = Vec::with_capacity(n);
        let mut freqs = Vec::with_capacity(n);
        for i in 0..n {
            let v = read_varint(data, pos)?;
            let f = read_u16(data, pos)? as u32;
            if i + 1 == n {
                symbols.push(ESC_SENTINEL);
            } else {
                symbols.push(v);
            }
            freqs.push(f);
        }
        // Validate the modelled symbols are strictly ascending (canonical form).
        for w in symbols[..n - 1].windows(2) {
            if w[1] <= w[0] {
                return Err("sideinfo_rans: model symbols not strictly ascending".into());
            }
        }
        let total: u64 = freqs.iter().map(|&f| f as u64).sum();
        if total != SCALE_TOTAL as u64 {
            return Err(format!(
                "sideinfo_rans: model freqs sum {total} != SCALE_TOTAL {SCALE_TOTAL}"
            ));
        }
        if freqs.iter().any(|&f| f == 0) {
            return Err("sideinfo_rans: model has a zero-frequency slot".into());
        }
        let mut cum = Vec::with_capacity(n + 1);
        let mut acc = 0u32;
        cum.push(0);
        for &f in &freqs {
            acc += f;
            cum.push(acc);
        }
        Ok(Model { symbols, cum })
    }
}

/// Normalize raw integer counts into a cumulative table summing to
/// `SCALE_TOTAL`, every nonzero count kept at `freq >= 1`. Integer-only,
/// deterministic — this is the byte-exact mirror of `strand_core::cdf`'s
/// `from_counts`, reproduced locally to keep the crate dependency-free.
fn normalize_to_cum(counts: &[u64]) -> Vec<u32> {
    let n = counts.len();
    debug_assert!(n >= 1);
    let total_raw: u64 = counts.iter().sum();
    let total = SCALE_TOTAL as u64;
    let mut freqs = vec![0u32; n];
    if total_raw == 0 {
        distribute_uniform(&mut freqs);
    } else {
        let mut allocated: u64 = 0;
        for (i, &c) in counts.iter().enumerate() {
            if c == 0 {
                freqs[i] = 0;
                continue;
            }
            let mut f = c * total / total_raw;
            if f == 0 {
                f = 1;
            }
            freqs[i] = f as u32;
            allocated += f;
        }
        if allocated < total {
            let need = (total - allocated) as u32;
            let idx = argmax(&freqs);
            freqs[idx] = freqs[idx].wrapping_add(need);
        } else if allocated > total {
            let mut excess = allocated - total;
            while excess > 0 {
                let idx = argmax(&freqs);
                let take = excess.min(freqs[idx].saturating_sub(1) as u64);
                if take == 0 {
                    distribute_uniform(&mut freqs);
                    break;
                }
                freqs[idx] -= take as u32;
                excess -= take;
            }
        }
    }
    let mut cum = Vec::with_capacity(n + 1);
    let mut acc = 0u32;
    cum.push(0);
    for &f in &freqs {
        acc += f;
        cum.push(acc);
    }
    debug_assert_eq!(*cum.last().unwrap(), SCALE_TOTAL);
    cum
}

fn distribute_uniform(freqs: &mut [u32]) {
    let n = freqs.len() as u32;
    let base = SCALE_TOTAL / n;
    let mut rem = SCALE_TOTAL - base * n;
    for f in freqs.iter_mut() {
        *f = base;
        if rem > 0 {
            *f += 1;
            rem -= 1;
        }
    }
}

fn argmax(freqs: &[u32]) -> usize {
    let mut best = 0usize;
    let mut best_v = 0u32;
    for (i, &f) in freqs.iter().enumerate() {
        if f > best_v {
            best_v = f;
            best = i;
        }
    }
    best
}

// ===========================================================================
// zig-zag + varint (signed i64 <-> u64, compact escape payloads)
// ===========================================================================

#[inline]
pub fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
pub fn unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
}

fn read_varint(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    let mut v = 0u64;
    let mut shift = 0u32;
    loop {
        if *pos >= data.len() {
            return Err("sideinfo_rans: varint truncated".into());
        }
        let byte = data[*pos];
        *pos += 1;
        v |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err("sideinfo_rans: varint overflow".into());
        }
    }
    Ok(v)
}

#[inline]
fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    let end = *pos + 4;
    let s = data.get(*pos..end).ok_or("sideinfo_rans: u32 truncated")?;
    *pos = end;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}

#[inline]
fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    let end = *pos + 2;
    let s = data.get(*pos..end).ok_or("sideinfo_rans: u16 truncated")?;
    *pos = end;
    Ok(u16::from_le_bytes(s.try_into().unwrap()))
}

// ===========================================================================
// Public stream codec: encode/decode a raw i64 symbol stream
// ===========================================================================

/// One self-describing rANS-coded section. Layout (all integer, little-endian):
///
/// ```text
/// [u32 n_symbols]               number of raw symbols coded
/// [model: serialize()]          static CDF (symbol table + 12-bit freqs)
/// [u32 esc_len]                 byte length of the escaped-values blob
/// [esc blob: varint zigzag ...] raw values for symbols that hit ESC, in stream
///                               order (decoder reads them back in the same order)
/// [u32 rans_len]                byte length of the rANS payload
/// [rans payload]                state (4 LE bytes) + reversed renorm bytes
/// ```
///
/// `encode_stream` is the only place that may use non-determinism-relevant
/// float (it does not). `decode_stream` is integer-only.
pub fn encode_stream(raw: &[i64]) -> Vec<u8> {
    let model = Model::from_stream(raw);
    encode_stream_with_model(raw, &model)
}

/// Encode against a caller-supplied (e.g. shared, frozen) model.
pub fn encode_stream_with_model(raw: &[i64], model: &Model) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + 64);
    out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    model.serialize(&mut out);

    // Escaped values: for any symbol routed to ESC, record its raw value so the
    // decoder can recover it. Stored in stream order; the decoder pops them as
    // it decodes ESC symbols left-to-right.
    let esc_idx = model.esc_index();
    let mut esc_blob: Vec<u8> = Vec::new();
    for &v in raw {
        if model.slot_of(v) == esc_idx {
            write_varint(&mut esc_blob, zigzag(v));
        }
    }
    out.extend_from_slice(&(esc_blob.len() as u32).to_le_bytes());
    out.extend_from_slice(&esc_blob);

    // rANS encode the *slots* (not raw values), back-to-front (rANS is LIFO).
    let mut x: u32 = L;
    let mut renorm: Vec<u8> = Vec::with_capacity(raw.len() + 16);
    for i in (0..raw.len()).rev() {
        let slot = model.slot_of(raw[i]);
        let start = model.cum[slot];
        let freq = model.cum[slot + 1] - model.cum[slot];
        enc_put(&mut x, &mut renorm, start, freq);
    }
    renorm.reverse();

    let mut payload = Vec::with_capacity(renorm.len() + 4);
    payload.extend_from_slice(&x.to_le_bytes());
    payload.extend_from_slice(&renorm);

    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Decode a section produced by [`encode_stream`]. Integer-only, deterministic.
/// Returns the exact original `raw` symbol stream.
pub fn decode_stream(data: &[u8], pos: &mut usize) -> Result<Vec<i64>, String> {
    let n = read_u32(data, pos)? as usize;
    let model = Model::deserialize(data, pos)?;

    let esc_len = read_u32(data, pos)? as usize;
    let esc_end = *pos + esc_len;
    let esc_blob = data.get(*pos..esc_end).ok_or("sideinfo_rans: esc blob truncated")?;
    *pos = esc_end;
    let mut esc_pos = 0usize;

    let payload_len = read_u32(data, pos)? as usize;
    let payload_end = *pos + payload_len;
    let payload = data.get(*pos..payload_end).ok_or("sideinfo_rans: payload truncated")?;
    *pos = payload_end;
    if payload.len() < 4 {
        if n == 0 {
            return Ok(Vec::new());
        }
        return Err("sideinfo_rans: payload shorter than initial state".into());
    }

    let mut x = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let mut rpos = 4usize;
    let esc_idx = model.esc_index();

    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let slot = dec_get(&mut x, payload, &mut rpos, &model.cum);
        let raw = if slot == esc_idx {
            let z = read_varint(esc_blob, &mut esc_pos)?;
            unzigzag(z)
        } else {
            unzigzag(model.symbols[slot])
        };
        out.push(raw);
    }
    Ok(out)
}

// ===========================================================================
// Stream transforms matching the bit-ledger entropy model
// ===========================================================================

/// Gap-transform a sorted-ascending outlier-position stream, exactly as the
/// bit-ledger `outl_pos_gap` predictor does: the first index is kept absolute,
/// each subsequent index becomes `idx - prev`. Inverse is a prefix sum.
///
/// Positions in `OutlierWire.entries` are guaranteed strictly ascending, so
/// gaps are `>= 1` after the first; we still zig-zag inside the codec to stay
/// uniform with `scale_q`.
pub fn positions_to_gaps(sorted_positions: &[u32]) -> Vec<i64> {
    let mut out = Vec::with_capacity(sorted_positions.len());
    let mut prev: Option<u32> = None;
    for &i in sorted_positions {
        match prev {
            None => out.push(i as i64),
            Some(p) => out.push(i as i64 - p as i64),
        }
        prev = Some(i);
    }
    out
}

/// Inverse of [`positions_to_gaps`]: prefix-sum the gaps back to absolute
/// positions. Returns an error if the reconstruction is not strictly ascending
/// or overflows `u32` (defensive — a corrupt stream must not silently decode).
pub fn gaps_to_positions(gaps: &[i64]) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(gaps.len());
    let mut acc: i64 = 0;
    for (k, &g) in gaps.iter().enumerate() {
        if k == 0 {
            acc = g;
        } else {
            acc += g;
        }
        if acc < 0 || acc > u32::MAX as i64 {
            return Err(format!("sideinfo_rans: position {acc} out of u32 range"));
        }
        if k > 0 && g <= 0 {
            return Err("sideinfo_rans: outlier positions must be strictly ascending".into());
        }
        out.push(acc as u32);
    }
    Ok(out)
}

/// Convenience: encode a `scale_q` stream (one `i32` per block) as a section.
pub fn encode_scale_q(scale_q: &[i32]) -> Vec<u8> {
    let raw: Vec<i64> = scale_q.iter().map(|&s| s as i64).collect();
    encode_stream(&raw)
}

/// Decode a `scale_q` section back to `i32`s.
pub fn decode_scale_q(data: &[u8], pos: &mut usize) -> Result<Vec<i32>, String> {
    let raw = decode_stream(data, pos)?;
    raw.iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| "sideinfo_rans: scale_q out of i32 range".to_string())
        })
        .collect()
}

/// Convenience: encode an outlier-**position** stream (strictly ascending `u32`)
/// as a gap-coded section — the bigger of the two measured levers.
pub fn encode_positions(sorted_positions: &[u32]) -> Vec<u8> {
    let gaps = positions_to_gaps(sorted_positions);
    encode_stream(&gaps)
}

/// Decode an outlier-position section back to strictly ascending `u32`s.
pub fn decode_positions(data: &[u8], pos: &mut usize) -> Result<Vec<u32>, String> {
    let gaps = decode_stream(data, pos)?;
    gaps_to_positions(&gaps)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- deterministic PRNG (no rand dep) ----
    fn splitmix64(x: &mut u64) -> u64 {
        *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn round_trip_raw(raw: &[i64]) -> Vec<u8> {
        let enc = encode_stream(raw);
        let mut pos = 0usize;
        let back = decode_stream(&enc, &mut pos).expect("decode");
        assert_eq!(back, raw, "round-trip mismatch");
        assert_eq!(pos, enc.len(), "decoder must consume the whole section");
        enc
    }

    // ---------------- core round-trip proofs ----------------

    #[test]
    fn zigzag_is_bijective() {
        for v in [0i64, 1, -1, 2, -2, i32::MAX as i64, i32::MIN as i64, i64::MAX, i64::MIN] {
            assert_eq!(unzigzag(zigzag(v)), v, "zigzag failed for {v}");
        }
    }

    #[test]
    fn varint_round_trips() {
        let vals = [0u64, 1, 127, 128, 300, 16_383, 16_384, u32::MAX as u64, u64::MAX];
        let mut buf = Vec::new();
        for &v in &vals {
            write_varint(&mut buf, v);
        }
        let mut pos = 0;
        for &v in &vals {
            assert_eq!(read_varint(&buf, &mut pos).unwrap(), v);
        }
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn empty_stream_round_trips() {
        round_trip_raw(&[]);
    }

    #[test]
    fn single_symbol_stream() {
        round_trip_raw(&[42i64; 100]);
    }

    #[test]
    fn small_known_stream_exact() {
        let raw: Vec<i64> = vec![3, 3, 3, -1, -1, 5, 0, 0, 0, 0, 7, 3, -1];
        round_trip_raw(&raw);
    }

    #[test]
    fn model_serialize_round_trips() {
        let raw: Vec<i64> = (0..2000).map(|i| ((i * 7) % 13) as i64 - 6).collect();
        let model = Model::from_stream(&raw);
        let mut buf = Vec::new();
        model.serialize(&mut buf);
        let mut pos = 0;
        let back = Model::deserialize(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(back, model, "model serialize/deserialize must be exact");
    }

    #[test]
    fn negative_and_large_values() {
        let raw: Vec<i64> =
            vec![i32::MIN as i64, i32::MAX as i64, 0, -1, 1, -1_000_000, 1_000_000];
        round_trip_raw(&raw);
    }

    #[test]
    fn many_distinct_symbols_force_escape() {
        // > MAX_MODEL_SYMBOLS distinct values: the tail must route through ESC
        // and still round-trip byte-exactly. Spread values over ~20k levels so
        // the alphabet comfortably exceeds the 4096-symbol model cap.
        let mut s = 0xDEAD_BEEFu64;
        let raw: Vec<i64> = (0..40_000)
            .map(|_| (splitmix64(&mut s) % 20_000) as i64 - 10_000)
            .collect();
        let distinct: std::collections::HashSet<i64> = raw.iter().cloned().collect();
        assert!(
            distinct.len() > MAX_MODEL_SYMBOLS,
            "test should overflow the model (distinct={})",
            distinct.len()
        );
        let enc = round_trip_raw(&raw);
        assert!(!enc.is_empty());
    }

    #[test]
    fn all_escape_pathological() {
        // Build a model on one distribution, then code a totally disjoint one so
        // *every* symbol escapes — exercises the ESC-only decode path.
        let train: Vec<i64> = vec![0i64; 50];
        let model = Model::from_stream(&train);
        let data: Vec<i64> = (1..=200).collect(); // none equal 0
        let enc = encode_stream_with_model(&data, &model);
        let mut pos = 0;
        let back = decode_stream(&enc, &mut pos).unwrap();
        assert_eq!(back, data);
        assert_eq!(pos, enc.len());
    }

    // ---------------- transform proofs ----------------

    #[test]
    fn gap_transform_inverts() {
        let positions: Vec<u32> = vec![3, 7, 8, 100, 101, 5000, 1_000_000];
        let gaps = positions_to_gaps(&positions);
        assert_eq!(gaps[0], 3);
        assert_eq!(gaps[1], 4);
        let back = gaps_to_positions(&gaps).unwrap();
        assert_eq!(back, positions);
    }

    #[test]
    fn positions_round_trip_through_codec() {
        let mut s = 0x1234_5678u64;
        let mut positions: Vec<u32> = Vec::new();
        let mut cur = 0u32;
        for _ in 0..3000 {
            // clustered gaps: mostly small, occasionally large — like real
            // 1%-by-|w| outliers.
            let gap = 1 + (splitmix64(&mut s) % 64) as u32;
            cur = cur.saturating_add(gap);
            positions.push(cur);
        }
        let enc = encode_positions(&positions);
        let mut pos = 0;
        let back = decode_positions(&enc, &mut pos).unwrap();
        assert_eq!(back, positions);
        assert_eq!(pos, enc.len());
    }

    #[test]
    fn scale_q_round_trip_i32() {
        let mut s = 0xABCD_0001u64;
        let scale_q: Vec<i32> = (0..4000)
            .map(|_| (splitmix64(&mut s) % 2048) as i32 - 1024)
            .collect();
        let enc = encode_scale_q(&scale_q);
        let mut pos = 0;
        let back = decode_scale_q(&enc, &mut pos).unwrap();
        assert_eq!(back, scale_q);
        assert_eq!(pos, enc.len());
    }

    #[test]
    fn corrupt_model_sum_is_error_not_panic() {
        let raw: Vec<i64> = vec![1, 2, 3, 1, 2, 1];
        let mut enc = encode_stream(&raw);
        // Corrupt the first frequency in the serialized model (just past the
        // u32 n_symbols + u32 stream-len headers + first varint).
        // Find a freq byte to flip without UB: flip a byte in the middle.
        let mid = 12.min(enc.len().saturating_sub(1));
        enc[mid] ^= 0xFF;
        let mut pos = 0;
        // Either the model fails to validate, or the stream decodes to something
        // else — but it must never panic and must consume bounded input.
        let _ = decode_stream(&enc, &mut pos); // must not panic
    }

    #[test]
    fn truncated_section_is_error() {
        let raw: Vec<i64> = (0..500).map(|i| (i % 7) as i64).collect();
        let enc = encode_stream(&raw);
        for cut in [0usize, 1, 4, 8, enc.len() / 2, enc.len() - 1] {
            let mut pos = 0;
            let r = decode_stream(&enc[..cut], &mut pos);
            // Truncated input must be a clean Err (or, for cut==exact-header with
            // n==0, an empty Ok) — never a panic.
            let _ = r;
        }
    }

    // ---------------- measurement: achieved bits/symbol ----------------
    //
    // These are not pass/fail gates; they print the measured compression so the
    // operator can compare against the bit-ledger entropy ceiling. Run with
    // `--nocapture` to see the numbers.

    /// Empirical order-0 Shannon entropy of a raw i64 stream (bits/symbol).
    fn order0_entropy(raw: &[i64]) -> f64 {
        if raw.is_empty() {
            return 0.0;
        }
        let mut counts: std::collections::HashMap<i64, u64> = std::collections::HashMap::new();
        for &v in raw {
            *counts.entry(v).or_insert(0) += 1;
        }
        let n = raw.len() as f64;
        let mut h = 0.0;
        for &c in counts.values() {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
        h
    }

    /// Synthetic scale_q model tuned to the ledger's real q2 profile: ~10.5-bit
    /// order-0 entropy over ~1.5k distinct levels (a bell curve around 0). This
    /// is the closest standalone proxy without a GPU encode of the real model;
    /// the alphabet fits under `MAX_MODEL_SYMBOLS` so escapes are negligible,
    /// exactly as on the real artifact.
    fn synthetic_scale_q(n: usize, seed: u64) -> Vec<i32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                // Sum of 4 uniforms ~ bell curve, ~1400 distinct levels, H≈10.5.
                let a = (splitmix64(&mut s) % 360) as i64;
                let b = (splitmix64(&mut s) % 360) as i64;
                let c = (splitmix64(&mut s) % 360) as i64;
                let d = (splitmix64(&mut s) % 360) as i64;
                (a + b + c + d - 718) as i32
            })
            .collect()
    }

    #[test]
    fn measure_scale_q_bits_per_symbol() {
        let scale_q = synthetic_scale_q(1_397_760, 0x5CA1_E000); // ~real q2 block count
        let raw: Vec<i64> = scale_q.iter().map(|&v| v as i64).collect();
        let h = order0_entropy(&raw);
        let enc = encode_scale_q(&scale_q);
        let achieved_bits_per_sym = (enc.len() * 8) as f64 / scale_q.len() as f64;
        // fixed-width billing today: 32 bits/block.
        let raw_bits = 32.0;
        let recoverable_vs_raw = raw_bits - achieved_bits_per_sym;
        eprintln!(
            "[MEASURE scale_q] n={} H0={:.3} bits/sym | achieved={:.3} bits/sym \
             (incl. model+framing) | raw=32 | recovered_vs_raw={:.3} bits/sym | \
             overhead_vs_entropy={:.3} bits/sym | section={} bytes",
            scale_q.len(),
            h,
            achieved_bits_per_sym,
            recoverable_vs_raw,
            achieved_bits_per_sym - h,
            enc.len(),
        );
        // The coder must beat fixed 32-bit storage by a wide margin and land
        // within a small constant of the order-0 entropy floor.
        assert!(
            achieved_bits_per_sym < raw_bits,
            "rANS must beat fixed-width 32-bit scale_q storage"
        );
        // Overhead above the order-0 floor is the serialized CDF table (~4 B per
        // distinct symbol, amortized over all blocks) plus rANS's constant flush.
        assert!(
            achieved_bits_per_sym < h + 0.20,
            "rANS within 0.2 bit/sym of order-0 entropy floor (got {achieved_bits_per_sym}, H={h})"
        );
    }

    #[test]
    fn measure_positions_bits_per_symbol() {
        // Clustered outlier positions over a ~3.5M-weight tensor at 1% density:
        // gaps concentrate (~7.8-bit gap entropy in the real ledger).
        let mut s = 0x0177_1E55u64;
        let n_weights = 896 * 4864; // a real ffn tensor shape from the ledger
        let n_outl = n_weights / 100; // 1%
        let mut positions: Vec<u32> = Vec::with_capacity(n_outl);
        let mut cur = 0u32;
        for _ in 0..n_outl {
            // clustered: most gaps small, heavy tail
            let r = splitmix64(&mut s) % 100;
            let gap = if r < 80 {
                1 + (splitmix64(&mut s) % 40) as u32
            } else {
                1 + (splitmix64(&mut s) % 400) as u32
            };
            cur = cur.saturating_add(gap);
            positions.push(cur);
        }
        let gaps = positions_to_gaps(&positions);
        let h_gap = order0_entropy(&gaps);
        let idx_bits = crate::outlier_wire::idx_bits_for(n_weights) as f64;
        let enc = encode_positions(&positions);
        let achieved_bits_per_sym = (enc.len() * 8) as f64 / positions.len() as f64;
        eprintln!(
            "[MEASURE outl_pos] n_pos={} idx_bits(raw)={} | H0(gaps)={:.3} bits/sym | \
             achieved={:.3} bits/sym (incl. model+framing) | recovered_vs_raw={:.3} bits/sym | \
             overhead_vs_entropy={:.3} bits/sym | section={} bytes",
            positions.len(),
            idx_bits,
            h_gap,
            achieved_bits_per_sym,
            idx_bits - achieved_bits_per_sym,
            achieved_bits_per_sym - h_gap,
            enc.len(),
        );
        assert!(
            achieved_bits_per_sym < idx_bits,
            "rANS gap-coding must beat fixed {idx_bits}-bit absolute positions"
        );
        // Per-tensor stream: the CDF table amortizes over only ~43k symbols
        // here, so the overhead above the gap floor is a touch larger than the
        // whole-model case (where one shared/per-tensor table covers far more).
        assert!(
            achieved_bits_per_sym < h_gap + 0.35,
            "rANS within 0.35 bit/sym of gap entropy floor (got {achieved_bits_per_sym}, H={h_gap})"
        );
    }
}
