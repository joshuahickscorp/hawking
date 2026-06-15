
use std::borrow::Cow;

/// How the per-state Gaussian codebook is sourced.
///
/// The codebook is a deterministic function of `L` either way; this only selects
/// *how* an entry is obtained, not *what* it is. Variant A
/// ([`ComputedAcklam`](CodebookMode::ComputedAcklam)) reproduces the frozen table
/// **byte-for-byte** (contract-tested in [`crate::codebook`]), so it requires no
/// re-encoding and never changes a decoded weight — it just trades a `2^L`-entry
/// LUT gather for a few integer ALU ops (a size / portability / occupancy win on
/// bandwidth-bound decoders).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CodebookMode {
    /// Borrow the frozen `&'static` integer LUT (the historical default).
    #[default]
    StoredLut,
    /// Compute each entry with the bit-exact integer Acklam path (no gather).
    /// Byte-identical to [`StoredLut`](CodebookMode::StoredLut) under Variant A.
    ComputedAcklam,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrellisConfig {

    pub l_bits: u32,

    pub k_bits: u32,

    pub block_len: usize,

    pub vec_dim: u32,
    /// How the codebook is sourced (stored LUT vs computed). Not on the wire —
    /// the codebook is a pure function of `L`, so this is a runtime choice only.
    pub codebook_mode: CodebookMode,
}

impl TrellisConfig {
    
    pub const MIN_L: u32 = 4;
    
    pub const MAX_L: u32 = 14;
    
    pub const MAX_K: u32 = 4;

    pub fn new(l_bits: u32, k_bits: u32, block_len: usize) -> Self {
        let l_bits = l_bits.clamp(Self::MIN_L, Self::MAX_L);
        let k_bits = k_bits.clamp(1, Self::MAX_K);
        
        let l_bits = l_bits.max(k_bits);
        let block_len = block_len.max(1);
        TrellisConfig {
            l_bits,
            k_bits,
            block_len,

            vec_dim: 1,
            codebook_mode: CodebookMode::StoredLut,
        }
    }

    /// Return a copy with the codebook source set to `mode`. Under
    /// [`CodebookMode::ComputedAcklam`] the decoded output is byte-identical to
    /// the default, so this is purely a decode-path implementation choice.
    #[must_use]
    pub fn with_codebook_mode(mut self, mode: CodebookMode) -> Self {
        self.codebook_mode = mode;
        self
    }

    /// The per-state codebook for this config, sourced per [`Self::codebook_mode`].
    ///
    /// `StoredLut` borrows the frozen `&'static` table (zero alloc); `ComputedAcklam`
    /// materialises the identical values via the integer Acklam path. Returned as
    /// a [`Cow`] so the default path stays allocation-free and both share one shape.
    pub fn codebook(&self) -> Cow<'static, [i32]> {
        match self.codebook_mode {
            CodebookMode::StoredLut => Cow::Borrowed(crate::codebook::codebook_lut(self.l_bits)),
            CodebookMode::ComputedAcklam => {
                Cow::Owned(crate::codebook::codebook_lut_computed(self.l_bits))
            }
        }
    }

    pub const MAX_VEC_DIM: u32 = 8;

    pub fn with_vec_dim(mut self, vec_dim: u32) -> Self {
        self.vec_dim = vec_dim.clamp(1, Self::MAX_VEC_DIM);
        self
    }

    pub fn for_bpw(target_bpw: f64) -> Self {
        
        let k = target_bpw.round().clamp(1.0, Self::MAX_K as f64) as u32;
        
        let l = (k + 4).min(Self::MAX_L);
        Self::new(l, k, 256)
    }

    pub fn for_bpw_quality(target_bpw: f64) -> Self {
        let k = target_bpw.round().clamp(1.0, Self::MAX_K as f64) as u32;
        let l = (k + 6).min(Self::MAX_L);
        Self::new(l, k, 256)
    }

    pub fn for_bpw_l(target_bpw: f64, l_bits: u32) -> Self {
        let k = target_bpw.round().clamp(1.0, Self::MAX_K as f64) as u32;
        Self::new(l_bits, k, 256)
    }

    #[inline]
    pub fn num_states(&self) -> usize {
        1usize << self.l_bits
    }

    #[inline]
    pub fn state_mask(&self) -> usize {
        self.num_states() - 1
    }

    #[inline]
    pub fn num_inputs(&self) -> usize {
        1usize << self.k_bits
    }

    #[inline]
    pub fn vec_dim(&self) -> usize {
        (self.vec_dim as usize).max(1)
    }

    #[inline]
    pub fn num_steps(&self, n: usize) -> usize {
        n.div_ceil(self.vec_dim())
    }

    #[inline]
    pub fn lut_len(&self) -> usize {
        self.num_states() * self.vec_dim()
    }

    #[inline]
    pub fn next_state(&self, state: usize, input: usize) -> usize {
        ((state << self.k_bits) | (input & (self.num_inputs() - 1))) & self.state_mask()
    }
}

pub fn push_bits(bits: &mut Vec<u8>, bit_cursor: &mut usize, value: usize, nbits: u32) {
    for i in 0..nbits {
        let bit = (value >> i) & 1;
        let byte_idx = *bit_cursor >> 3;
        let in_byte = *bit_cursor & 7;
        if byte_idx >= bits.len() {
            bits.push(0);
        }
        if bit != 0 {
            bits[byte_idx] |= 1u8 << in_byte;
        }
        *bit_cursor += 1;
    }
}

#[inline]
pub fn read_bits(bytes: &[u8], start_bit: usize, nbits: u32) -> usize {
    let mut acc = 0usize;
    for i in 0..nbits as usize {
        let bit_idx = start_bit + i;
        let byte_idx = bit_idx >> 3;
        let in_byte = bit_idx & 7;
        let bit = if byte_idx < bytes.len() {
            ((bytes[byte_idx] >> in_byte) & 1) as usize
        } else {
            0
        };
        acc |= bit << i;
    }
    acc
}
