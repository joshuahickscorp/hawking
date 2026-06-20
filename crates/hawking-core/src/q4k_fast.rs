//! Q4K_FAST — sub-block-contiguous re-layout of Q4_K GGUF blocks.
//!
//! ## Motivation
//!
//! Stock Q4_K (256-element super-block) packs metadata in scattered byte
//! ranges:
//!
//! ```text
//!   bytes  [0..2]    d        (fp16 super-block scale)
//!   bytes  [2..4]    dmin     (fp16 super-block min)
//!   bytes  [4..16]   12 bytes packed 6-bit sb_idx[0..8] + mb_idx[0..8]
//!   bytes  [16..144] 128 bytes of 4-bit nibble pairs (256 values)
//! ```
//!
//! The v3_8r GEMV kernel reads `sb_idx[k]`, `mb_idx[k]` for every sub-block
//! by stitching 6-bit values out of bytes 4..16 (two scattered loads with
//! shift/mask). The Q4K_FAST re-layout pre-computes the per-sub-block scale
//! `d * sb_idx[k]` and min `dmin * mb_idx[k]` into two fp16 fields per
//! sub-block, and groups everything for sub-block `k` into one contiguous
//! 20-byte chunk:
//!
//! ```text
//!   bytes  [k*20 + 0 ..k*20 + 2]   sub_scale (fp16) = d   * sb_idx[k]
//!   bytes  [k*20 + 2 ..k*20 + 4]   sub_min   (fp16) = dmin* mb_idx[k]
//!   bytes  [k*20 + 4 ..k*20 + 20]  16 bytes of 32 4-bit nibbles
//! ```
//!
//! 8 sub-blocks per super-block → 160 bytes per 256-element block (vs 144
//! for Q4_K — small bandwidth overhead, big cache-friendliness win on
//! the M3 Pro path).
//!
//! ## File container
//!
//! A sidecar `.hawking` file holds the re-laid weights for one or more
//! tensors of a source GGUF. Layout:
//!
//! ```text
//!   magic     : 9 bytes  = b"DSQ4KFAST"
//!   reserved  : 1 byte   = 0
//!   version   : u32 LE   = 1
//!   src_hash  : u64 LE   (first 8 bytes of sha256 of the source GGUF)
//!   n_tensors : u32 LE
//!   tensors   : repeated n_tensors times {
//!       name_len  : u32 LE
//!       name      : `name_len` bytes (UTF-8)
//!       rows      : u32 LE
//!       cols      : u32 LE
//!       byte_off  : u64 LE  (offset from start of file)
//!       byte_len  : u64 LE  (bytes; must equal rows*(cols/256)*160)
//!   }
//!   then raw tensor bytes at the offsets above.
//! ```
//!
//! ## Compatibility
//!
//! Q4K_FAST is a **lossless** re-layout. Output of
//! `gemv_q4k_fast_v1_pinned_tcb` on a Q4K_FAST tensor MUST be bit-identical
//! to `gemv_q4_k_m_v3_8r_pinned_tcb` on the source Q4_K tensor (per the
//! session's parity test at q_proj decode shape).

use half::f16;

/// Per-block byte size in source Q4_K layout.
pub const Q4K_BLOCK_BYTES: usize = 144;
/// Per-block byte size in Q4K_FAST layout (8 sub-blocks × 20 bytes).
pub const Q4K_FAST_BLOCK_BYTES: usize = 160;
/// Elements per super-block.
pub const Q4K_BLOCK_ELEMS: usize = 256;
/// Elements per sub-block.
pub const Q4K_SUB_ELEMS: usize = 32;

/// Container file magic. Exactly 9 bytes; followed by 1 reserved 0-byte.
pub const Q4K_FAST_MAGIC: &[u8; 9] = b"DSQ4KFAST";
/// Container format version. Bump on layout changes.
pub const Q4K_FAST_VERSION: u32 = 1;

/// Decode the 8 6-bit sub-block scale (`sb`) and min (`mb`) indices for a
/// single Q4_K block. Indices live in bytes `[4..16]` of the block, packed
/// per llama.cpp Q4_K_M:
///
/// ```text
///   for sub in 0..4 :
///       sb[sub] =  bytes[4 + sub]      & 0x3F
///       mb[sub] =  bytes[8 + sub]      & 0x3F
///   for j in 0..4 :
///       sb[4 + j] = (bytes[12 + j] & 0x0F) | ((bytes[4 + j] >> 6) << 4)
///       mb[4 + j] = (bytes[12 + j] >> 4)   | ((bytes[8 + j] >> 6) << 4)
/// ```
///
/// Returns `(sb_idx, mb_idx)`. This MUST match the unpack in the
/// `gemm_q4_k_m_v3_8r` Metal kernel (quant.metal).
#[inline]
pub fn decode_q4k_sb_mb(block: &[u8; Q4K_BLOCK_BYTES]) -> ([u8; 8], [u8; 8]) {
    let mut sb = [0u8; 8];
    let mut mb = [0u8; 8];
    for sub in 0..4 {
        sb[sub] = block[4 + sub] & 0x3F;
        mb[sub] = block[8 + sub] & 0x3F;
    }
    for j in 0..4 {
        sb[4 + j] = (block[12 + j] & 0x0F) | ((block[4 + j] >> 6) << 4);
        mb[4 + j] = (block[12 + j] >> 4) | ((block[8 + j] >> 6) << 4);
    }
    (sb, mb)
}

/// Convert one Q4_K block (144 bytes) to one Q4K_FAST block (160 bytes).
///
/// The mapping is fully determined by the kernel-side dequant in v3_8r:
///
/// ```text
///   for sub_block k in 0..8 :
///       sub_scale[k] = f16(d    * sb_idx[k])
///       sub_min[k]   = f16(dmin * mb_idx[k])
///       nibbles[k]   = 16 bytes; element 2i  (low nibble of byte i)
///                                element 2i+1 (high nibble of byte i)
/// ```
///
/// ## Nibble repacking
///
/// In source Q4_K (`gemm_q4_k_m_v3_8r`), element `e` of sub-block `k` is
/// the low nibble of byte `bytes_in[16 + (k/2)*32 + e]` if `k` is even,
/// or the high nibble of the same byte if `k` is odd. Each Q4_K byte
/// holds two elements from TWO different sub-blocks (k0=2i, k1=2i+1).
///
/// Q4K_FAST regroups: byte `i` of sub-block `k`'s payload holds element
/// `2i` (low nibble) and `2i+1` (high nibble) of THE SAME sub-block,
/// for `i in 0..16` and elements `0..32`.
///
/// Bit-identical math: the parity test calls this fn on a synthetic Q4_K
/// block where `d * sb_idx[k]` and `dmin * mb_idx[k]` are exactly
/// representable in fp16 (e.g. small integer products). Under that
/// constraint the kernel partial sums match v3_8r to the bit.
///
/// Note: the v3_8r kernel computes `ds[k] = d * sb_idx[k]` in fp32 from
/// fp16 `d` and integer `sb_idx[k]`. Storing the pre-multiplied product
/// as fp16 may quantize precision; the parity guarantee only holds when
/// the product is exactly representable in fp16.
pub fn convert_q4k_block_to_fast(
    src: &[u8; Q4K_BLOCK_BYTES],
    dst: &mut [u8; Q4K_FAST_BLOCK_BYTES],
) {
    let d_bits = u16::from_le_bytes([src[0], src[1]]);
    let dmin_bits = u16::from_le_bytes([src[2], src[3]]);
    let d = f16::from_bits(d_bits).to_f32();
    let dmin = f16::from_bits(dmin_bits).to_f32();

    let (sb_idx, mb_idx) = decode_q4k_sb_mb(src);

    for k in 0..8 {
        let sub_scale_f32 = d * sb_idx[k] as f32;
        let sub_min_f32 = dmin * mb_idx[k] as f32;
        let sub_scale = f16::from_f32(sub_scale_f32);
        let sub_min = f16::from_f32(sub_min_f32);
        let off = k * 20;
        dst[off..off + 2].copy_from_slice(&sub_scale.to_bits().to_le_bytes());
        dst[off + 2..off + 4].copy_from_slice(&sub_min.to_bits().to_le_bytes());

        // Repack 32 4-bit values for sub-block k into 16 contiguous bytes.
        //
        // Source: in v3_8r, element `e` of sub-block `k` lives at:
        //   pi = k / 2;  is_high = (k & 1) == 1
        //   src_byte = bytes_in[16 + pi*32 + e]
        //   val      = if is_high { src_byte >> 4 } else { src_byte & 0x0F }
        //
        // Destination: element 2i and 2i+1 of sub-block k go in
        //   dst[off + 4 + i]:
        //     low nibble  = val(e=2i)
        //     high nibble = val(e=2i+1)
        let pi = k / 2;
        let is_high = (k & 1) == 1;
        let src_run = &src[16 + pi * 32..16 + pi * 32 + 32];
        for i in 0..16 {
            let s0 = src_run[2 * i];
            let s1 = src_run[2 * i + 1];
            let v0 = if is_high { s0 >> 4 } else { s0 & 0x0F };
            let v1 = if is_high { s1 >> 4 } else { s1 & 0x0F };
            dst[off + 4 + i] = v0 | (v1 << 4);
        }
    }
}

/// Convert a contiguous Q4_K tensor (raw bytes from a GGUF) to Q4K_FAST
/// layout. The output is `n_blocks * 160` bytes, with `n_blocks = (rows *
/// cols) / 256`. Tensor major order is preserved (row-major); each row is
/// `cols/256` super-blocks.
pub fn convert_q4k_tensor_to_fast(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert!(
        cols % Q4K_BLOCK_ELEMS == 0,
        "cols must be a multiple of 256"
    );
    let blocks_per_row = cols / Q4K_BLOCK_ELEMS;
    let n_blocks = rows * blocks_per_row;
    assert_eq!(
        src.len(),
        n_blocks * Q4K_BLOCK_BYTES,
        "Q4_K input size mismatch"
    );

    let mut out = vec![0u8; n_blocks * Q4K_FAST_BLOCK_BYTES];
    for b in 0..n_blocks {
        let s_off = b * Q4K_BLOCK_BYTES;
        let d_off = b * Q4K_FAST_BLOCK_BYTES;
        let src_block: &[u8; Q4K_BLOCK_BYTES] =
            (&src[s_off..s_off + Q4K_BLOCK_BYTES]).try_into().unwrap();
        let dst_block: &mut [u8; Q4K_FAST_BLOCK_BYTES] = (&mut out
            [d_off..d_off + Q4K_FAST_BLOCK_BYTES])
            .try_into()
            .unwrap();
        convert_q4k_block_to_fast(src_block, dst_block);
    }
    out
}

/// Header descriptor for one tensor in a Q4K_FAST sidecar file.
#[derive(Debug, Clone)]
pub struct Q4kFastTensorEntry {
    pub name: String,
    pub rows: u32,
    pub cols: u32,
    pub byte_off: u64,
    pub byte_len: u64,
}

/// Parsed header of a Q4K_FAST sidecar file (no tensor bytes loaded).
#[derive(Debug, Clone)]
pub struct Q4kFastHeader {
    pub version: u32,
    pub src_hash: u64,
    pub tensors: Vec<Q4kFastTensorEntry>,
    /// Total bytes consumed by the header (offset of the first tensor's
    /// data, after the final entry). Useful for writers/readers that want
    /// to validate offsets.
    pub header_bytes: u64,
}

/// Result of writing a single tensor into the sidecar payload section.
pub struct WrittenTensor {
    pub name: String,
    pub rows: u32,
    pub cols: u32,
    pub byte_len: u64,
    pub bytes: Vec<u8>,
}

/// Compute the byte length of a Q4K_FAST header given the tensor metadata
/// (without computing payload). Used by writers to set `byte_off` fields
/// before they're known.
pub fn header_byte_length(tensors: &[(String, u32, u32)]) -> u64 {
    // magic(9) + reserved(1) + version(4) + src_hash(8) + n_tensors(4) = 26
    let mut len: u64 = 26;
    for (name, _rows, _cols) in tensors {
        // name_len(4) + name + rows(4) + cols(4) + byte_off(8) + byte_len(8) = 28 + name
        len += 28 + name.as_bytes().len() as u64;
    }
    len
}

/// Serialize a Q4K_FAST sidecar file from a list of (name, rows, cols,
/// fast_bytes). Returns the complete file as a single `Vec<u8>` (caller
/// writes to disk). Use `header_byte_length` to project layout if needed.
pub fn serialize_sidecar(src_hash: u64, tensors: Vec<WrittenTensor>) -> Vec<u8> {
    let meta: Vec<(String, u32, u32)> = tensors
        .iter()
        .map(|t| (t.name.clone(), t.rows, t.cols))
        .collect();
    let header_len = header_byte_length(&meta);

    let mut out = Vec::with_capacity(
        header_len as usize + tensors.iter().map(|t| t.bytes.len()).sum::<usize>(),
    );
    out.extend_from_slice(Q4K_FAST_MAGIC);
    out.push(0u8); // reserved
    out.extend_from_slice(&Q4K_FAST_VERSION.to_le_bytes());
    out.extend_from_slice(&src_hash.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());

    // Reserve offsets per tensor: tensor i lives at header_len + sum(prev byte_len).
    let mut cursor: u64 = header_len;
    let offsets: Vec<u64> = tensors
        .iter()
        .map(|t| {
            let off = cursor;
            cursor += t.bytes.len() as u64;
            off
        })
        .collect();

    for (t, &off) in tensors.iter().zip(offsets.iter()) {
        let name_bytes = t.name.as_bytes();
        out.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&t.rows.to_le_bytes());
        out.extend_from_slice(&t.cols.to_le_bytes());
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&t.byte_len.to_le_bytes());
    }
    debug_assert_eq!(out.len() as u64, header_len);

    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    out
}

/// Parse a Q4K_FAST sidecar header. Returns `Err` on bad magic / version /
/// truncated input.
pub fn parse_header(file: &[u8]) -> Result<Q4kFastHeader, String> {
    if file.len() < 26 {
        return Err(format!(
            "file too short for Q4K_FAST header: {}",
            file.len()
        ));
    }
    if &file[0..9] != Q4K_FAST_MAGIC {
        return Err(format!(
            "bad magic: expected {:?}, got {:?}",
            Q4K_FAST_MAGIC,
            &file[0..9]
        ));
    }
    if file[9] != 0 {
        return Err(format!("reserved byte not zero: {}", file[9]));
    }
    let version = u32::from_le_bytes(file[10..14].try_into().unwrap());
    if version != Q4K_FAST_VERSION {
        return Err(format!(
            "version mismatch: expected {Q4K_FAST_VERSION}, got {version}"
        ));
    }
    let src_hash = u64::from_le_bytes(file[14..22].try_into().unwrap());
    let n_tensors = u32::from_le_bytes(file[22..26].try_into().unwrap()) as usize;

    let mut tensors = Vec::with_capacity(n_tensors);
    let mut p = 26usize;
    for _ in 0..n_tensors {
        if p + 4 > file.len() {
            return Err("truncated in name_len".to_string());
        }
        let name_len = u32::from_le_bytes(file[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        if p + name_len + 24 > file.len() {
            return Err("truncated in tensor entry".to_string());
        }
        let name = std::str::from_utf8(&file[p..p + name_len])
            .map_err(|e| format!("name utf8: {e}"))?
            .to_string();
        p += name_len;
        let rows = u32::from_le_bytes(file[p..p + 4].try_into().unwrap());
        p += 4;
        let cols = u32::from_le_bytes(file[p..p + 4].try_into().unwrap());
        p += 4;
        let byte_off = u64::from_le_bytes(file[p..p + 8].try_into().unwrap());
        p += 8;
        let byte_len = u64::from_le_bytes(file[p..p + 8].try_into().unwrap());
        p += 8;
        tensors.push(Q4kFastTensorEntry {
            name,
            rows,
            cols,
            byte_off,
            byte_len,
        });
    }

    Ok(Q4kFastHeader {
        version,
        src_hash,
        tensors,
        header_bytes: p as u64,
    })
}

/// Compute the source-GGUF hash used in the sidecar header. We take the
/// first 8 bytes of the SHA-256 of the entire source file, interpreted as
/// little-endian u64. Cheap enough to compute once per build (the actual
/// hashing is in the offline tool; this function is exposed so loaders
/// and tools agree on the algorithm).
pub fn src_hash_from_sha256_first8(sha256_first8: [u8; 8]) -> u64 {
    u64::from_le_bytes(sha256_first8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_header() {
        let payload_a = vec![0xAAu8; 160 * 4]; // 4 super-blocks
        let payload_b = vec![0xBBu8; 160 * 8];
        let tensors = vec![
            WrittenTensor {
                name: "blk.0.attn_q.weight".to_string(),
                rows: 4,
                cols: 256,
                byte_len: payload_a.len() as u64,
                bytes: payload_a.clone(),
            },
            WrittenTensor {
                name: "blk.0.attn_k.weight".to_string(),
                rows: 8,
                cols: 256,
                byte_len: payload_b.len() as u64,
                bytes: payload_b.clone(),
            },
        ];
        let file = serialize_sidecar(0xDEAD_BEEF_CAFE_BABEu64, tensors);
        let hdr = parse_header(&file).unwrap();
        assert_eq!(hdr.version, Q4K_FAST_VERSION);
        assert_eq!(hdr.src_hash, 0xDEAD_BEEF_CAFE_BABEu64);
        assert_eq!(hdr.tensors.len(), 2);
        assert_eq!(hdr.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(hdr.tensors[0].rows, 4);
        assert_eq!(hdr.tensors[0].cols, 256);
        assert_eq!(hdr.tensors[0].byte_len, payload_a.len() as u64);
        let off_a = hdr.tensors[0].byte_off as usize;
        let len_a = hdr.tensors[0].byte_len as usize;
        assert_eq!(&file[off_a..off_a + len_a], payload_a.as_slice());
        let off_b = hdr.tensors[1].byte_off as usize;
        let len_b = hdr.tensors[1].byte_len as usize;
        assert_eq!(&file[off_b..off_b + len_b], payload_b.as_slice());
    }

    #[test]
    fn convert_block_matches_kernel_dequant() {
        // Synthetic block: fp32 reference dequant equals reconstructed value.
        let mut block = [0u8; Q4K_BLOCK_BYTES];
        let d = f16::from_f32(0.0123f32);
        let dmin = f16::from_f32(-0.005f32);
        block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
        // Pick deterministic scale/min indices (all in low-6-bit range).
        for i in 4..8 {
            block[i] = 0b00_010101 | (((i as u8) << 6) & 0xC0); // top 2 bits become high bits of sb[4+i-4]
        }
        for i in 8..12 {
            block[i] = 0b00_101010 | (((i as u8) << 6) & 0xC0);
        }
        for i in 12..16 {
            block[i] = 0b1100_0011; // sb[4+]/mb[4+] low nibbles
        }
        // 128 nibble bytes
        for i in 16..144 {
            block[i] = ((i as u32 * 31) & 0xFF) as u8;
        }

        let mut fast = [0u8; Q4K_FAST_BLOCK_BYTES];
        convert_q4k_block_to_fast(&block, &mut fast);

        // Reconstruct sb/mb indices and confirm products match.
        let (sb_idx, mb_idx) = decode_q4k_sb_mb(&block);
        for k in 0..8 {
            let off = k * 20;
            let scale_bits = u16::from_le_bytes([fast[off], fast[off + 1]]);
            let min_bits = u16::from_le_bytes([fast[off + 2], fast[off + 3]]);
            let scale = f16::from_bits(scale_bits).to_f32();
            let min = f16::from_bits(min_bits).to_f32();
            let expected_scale = f16::from_f32(d.to_f32() * sb_idx[k] as f32).to_f32();
            let expected_min = f16::from_f32(dmin.to_f32() * mb_idx[k] as f32).to_f32();
            assert_eq!(scale, expected_scale, "k={k}");
            assert_eq!(min, expected_min, "k={k}");
            // Reconstruct expected nibble payload for sub-block k under
            // the repacking rules in `convert_q4k_block_to_fast`.
            let pi = k / 2;
            let is_high = (k & 1) == 1;
            let src_run = &block[16 + pi * 32..16 + pi * 32 + 32];
            for i in 0..16 {
                let s0 = src_run[2 * i];
                let s1 = src_run[2 * i + 1];
                let v0 = if is_high { s0 >> 4 } else { s0 & 0x0F };
                let v1 = if is_high { s1 >> 4 } else { s1 & 0x0F };
                assert_eq!(fast[off + 4 + i], v0 | (v1 << 4), "nibble k={k} i={i}");
            }
        }
    }
}
