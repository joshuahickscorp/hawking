//! Minimal NPZ (numpy `.npz`) reader for `np.savez`-produced archives.
//!
//! Targets the EAGLE-4 checkpoint format produced by `eagle4/eagle4.py:136`:
//! `np.savez(path, **flat)` — uncompressed ZIP STORED of `.npy` entries,
//! fortran_order=False, dtypes `<f4` / `<i4` (one `<i2` would also be
//! handled by the same path).
//!
//! Scope is intentionally small:
//!
//! - **ZIP STORED only.** `np.savez_compressed` (DEFLATE) is rejected.
//! - **No ZIP64.** Files > 4 GB are rejected. EAGLE-4 best.npz is ~300 MB.
//! - `fortran_order=True` is transposed on read for 1D/2D arrays (path-to-90
//!   step 6 — MLX writes `mx.transpose(...).save(...)` with this flag set
//!   on the resulting v2lite_frozen.npz / eagle4 checkpoint downstream).
//! - **Limited dtypes**: `<f4` (f32), `<f2` (f16), `<i4` (i32), `<i8` (i64).
//!
//! The loader reads the whole file into memory. EAGLE-4 checkpoints are a
//! few hundred MB; the integration call path is one-shot at engine init.

use crate::{Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDtype {
    F32,
    F16,
    I32,
    I64,
}

impl NpyDtype {
    fn from_descr(descr: &str) -> Result<Self> {
        match descr {
            "<f4" => Ok(NpyDtype::F32),
            "<f2" => Ok(NpyDtype::F16),
            "<i4" => Ok(NpyDtype::I32),
            "<i8" => Ok(NpyDtype::I64),
            other => Err(Error::Model(format!(
                "npz: unsupported dtype descr '{}' (supported: <f4 <f2 <i4 <i8)",
                other
            ))),
        }
    }

    fn item_size(&self) -> usize {
        match self {
            NpyDtype::F32 | NpyDtype::I32 => 4,
            NpyDtype::F16 => 2,
            NpyDtype::I64 => 8,
        }
    }
}

#[derive(Debug)]
pub struct NpyArray {
    pub dtype: NpyDtype,
    pub shape: Vec<usize>,
    /// Raw little-endian bytes, length == numel * dtype.item_size().
    pub data: Vec<u8>,
}

impl NpyArray {
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Decode the array as `Vec<f32>`. f32 passes through; f16 upcasts;
    /// integer dtypes error.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        let n = self.numel();
        match self.dtype {
            NpyDtype::F32 => {
                if self.data.len() != n * 4 {
                    return Err(Error::Model(format!(
                        "npz: f32 array byte mismatch (numel={}, bytes={})",
                        n,
                        self.data.len()
                    )));
                }
                let mut out = Vec::with_capacity(n);
                for chunk in self.data.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(out)
            }
            NpyDtype::F16 => {
                if self.data.len() != n * 2 {
                    return Err(Error::Model(format!(
                        "npz: f16 array byte mismatch (numel={}, bytes={})",
                        n,
                        self.data.len()
                    )));
                }
                let mut out = Vec::with_capacity(n);
                for chunk in self.data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(half_to_f32(bits));
                }
                Ok(out)
            }
            _ => Err(Error::Model(format!(
                "npz: as_f32 on integer dtype {:?}",
                self.dtype
            ))),
        }
    }

    /// Decode a 0-d or 1-element array as a single f32 (handles
    /// `residual_gate` shape `(1,)` and `calib_proj.bias` shape `(1,)`).
    pub fn as_f32_scalar(&self) -> Result<f32> {
        if self.numel() != 1 {
            return Err(Error::Model(format!(
                "npz: as_f32_scalar on array of {} elements",
                self.numel()
            )));
        }
        Ok(self.as_f32()?[0])
    }

    /// Decode a 0-d i32 scalar (handles `__step__`).
    pub fn as_i32_scalar(&self) -> Result<i32> {
        if self.numel() != 1 || self.dtype != NpyDtype::I32 {
            return Err(Error::Model(format!(
                "npz: as_i32_scalar on shape {:?} dtype {:?}",
                self.shape, self.dtype
            )));
        }
        Ok(i32::from_le_bytes([
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[3],
        ]))
    }
}

/// Read an `.npz` file from disk and return all arrays keyed by their
/// archive name (the `.npy` suffix is stripped).
pub fn read_npz<P: AsRef<Path>>(path: P) -> Result<HashMap<String, NpyArray>> {
    let mut buf = Vec::new();
    File::open(path.as_ref())?.read_to_end(&mut buf)?;
    parse_npz(&buf)
}

/// Same as `read_npz` but parses from an in-memory buffer. Useful for
/// unit tests that build a synthetic archive.
pub fn parse_npz(buf: &[u8]) -> Result<HashMap<String, NpyArray>> {
    let entries = parse_zip_central_directory(buf)?;
    let mut out = HashMap::with_capacity(entries.len());
    for entry in entries {
        let raw = read_zip_entry_data(buf, &entry)?;
        let arr = parse_npy(raw)?;
        let name = entry.name.trim_end_matches(".npy").to_string();
        out.insert(name, arr);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ZIP STORED central-directory parsing (no ZIP64, no compression).
// ---------------------------------------------------------------------------

const SIG_EOCD: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
const SIG_CDH: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
const SIG_LFH: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];

struct ZipEntry {
    name: String,
    compressed_size: u32,
    local_header_offset: u32,
}

fn parse_zip_central_directory(buf: &[u8]) -> Result<Vec<ZipEntry>> {
    let eocd_pos = find_eocd(buf)?;
    if buf.len() < eocd_pos + 22 {
        return Err(Error::Model("npz: truncated EOCD".into()));
    }
    let eocd = &buf[eocd_pos..];
    let total_entries = u16::from_le_bytes([eocd[10], eocd[11]]) as usize;
    let cd_size = u32::from_le_bytes([eocd[12], eocd[13], eocd[14], eocd[15]]) as usize;
    let cd_offset = u32::from_le_bytes([eocd[16], eocd[17], eocd[18], eocd[19]]) as usize;

    if cd_offset == 0xFFFF_FFFF || cd_size == 0xFFFF_FFFF {
        return Err(Error::Model(
            "npz: ZIP64 archives not supported (file > 4 GB?)".into(),
        ));
    }
    if cd_offset + cd_size > buf.len() {
        return Err(Error::Model("npz: central directory bounds out of range".into()));
    }

    let mut entries = Vec::with_capacity(total_entries);
    let mut p = cd_offset;
    let cd_end = cd_offset + cd_size;
    while p < cd_end {
        if p + 46 > buf.len() || buf[p..p + 4] != SIG_CDH {
            return Err(Error::Model(format!(
                "npz: bad central-directory header at offset {}",
                p
            )));
        }
        let compression = u16::from_le_bytes([buf[p + 10], buf[p + 11]]);
        if compression != 0 {
            return Err(Error::Model(format!(
                "npz: ZIP entry uses compression={} (only STORED=0 supported; \
                 use np.savez, not np.savez_compressed)",
                compression
            )));
        }
        let compressed_size = u32::from_le_bytes([buf[p + 20], buf[p + 21], buf[p + 22], buf[p + 23]]);
        let name_len = u16::from_le_bytes([buf[p + 28], buf[p + 29]]) as usize;
        let extra_len = u16::from_le_bytes([buf[p + 30], buf[p + 31]]) as usize;
        let comment_len = u16::from_le_bytes([buf[p + 32], buf[p + 33]]) as usize;
        let local_header_offset = u32::from_le_bytes([buf[p + 42], buf[p + 43], buf[p + 44], buf[p + 45]]);
        let name_start = p + 46;
        let name_end = name_start + name_len;
        if name_end > buf.len() {
            return Err(Error::Model("npz: central-directory name truncated".into()));
        }
        let name = std::str::from_utf8(&buf[name_start..name_end])
            .map_err(|e| Error::Model(format!("npz: non-utf8 entry name: {}", e)))?
            .to_string();
        entries.push(ZipEntry {
            name,
            compressed_size,
            local_header_offset,
        });
        p = name_end + extra_len + comment_len;
    }
    if entries.len() != total_entries {
        return Err(Error::Model(format!(
            "npz: parsed {} central-directory entries, EOCD claimed {}",
            entries.len(),
            total_entries
        )));
    }
    Ok(entries)
}

fn find_eocd(buf: &[u8]) -> Result<usize> {
    // EOCD is 22 bytes minimum; comment field ≤ 65535 bytes. Search back
    // from end of file.
    let max_search = (1usize << 16) + 22;
    let start = buf.len().saturating_sub(max_search);
    for i in (start..buf.len().saturating_sub(3)).rev() {
        if buf[i..i + 4] == SIG_EOCD {
            return Ok(i);
        }
    }
    Err(Error::Model(
        "npz: end-of-central-directory record not found (file truncated?)".into(),
    ))
}

fn read_zip_entry_data<'a>(buf: &'a [u8], entry: &ZipEntry) -> Result<&'a [u8]> {
    let lfh_start = entry.local_header_offset as usize;
    if lfh_start + 30 > buf.len() || buf[lfh_start..lfh_start + 4] != SIG_LFH {
        return Err(Error::Model(format!(
            "npz: bad local file header for {} at offset {}",
            entry.name, lfh_start
        )));
    }
    let lfh = &buf[lfh_start..];
    let name_len = u16::from_le_bytes([lfh[26], lfh[27]]) as usize;
    let extra_len = u16::from_le_bytes([lfh[28], lfh[29]]) as usize;
    let data_start = lfh_start + 30 + name_len + extra_len;
    let data_end = data_start + entry.compressed_size as usize;
    if data_end > buf.len() {
        return Err(Error::Model(format!(
            "npz: entry data for {} truncated (need {}..{}, have {})",
            entry.name,
            data_start,
            data_end,
            buf.len()
        )));
    }
    Ok(&buf[data_start..data_end])
}

// ---------------------------------------------------------------------------
// NPY file format (NEP 1) parser.
// ---------------------------------------------------------------------------

fn parse_npy(buf: &[u8]) -> Result<NpyArray> {
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        return Err(Error::Model("npz: entry missing \\x93NUMPY magic".into()));
    }
    let major = buf[6];
    let minor = buf[7];
    let (hdr_len_bytes, hdr_start) = match major {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10usize),
        2 | 3 => {
            if buf.len() < 12 {
                return Err(Error::Model("npz: truncated v2/3 npy header length".into()));
            }
            (
                u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
                12usize,
            )
        }
        _ => {
            return Err(Error::Model(format!(
                "npz: unsupported npy version {}.{}",
                major, minor
            )));
        }
    };
    let hdr_end = hdr_start + hdr_len_bytes;
    if hdr_end > buf.len() {
        return Err(Error::Model("npz: truncated npy header".into()));
    }
    let header = std::str::from_utf8(&buf[hdr_start..hdr_end])
        .map_err(|e| Error::Model(format!("npz: non-utf8 npy header: {}", e)))?;

    let descr = extract_string_field(header, "descr")
        .ok_or_else(|| Error::Model(format!("npz: header missing 'descr': {:?}", header)))?;
    let fortran_order = extract_bool_field(header, "fortran_order")
        .ok_or_else(|| Error::Model(format!("npz: header missing 'fortran_order': {:?}", header)))?;
    let shape = extract_shape(header)
        .ok_or_else(|| Error::Model(format!("npz: header missing 'shape': {:?}", header)))?;

    let dtype = NpyDtype::from_descr(&descr)?;
    let numel: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
    let nbytes = numel * dtype.item_size();
    let data_start = hdr_end;
    let data_end = data_start + nbytes;
    if data_end > buf.len() {
        return Err(Error::Model(format!(
            "npz: npy data truncated (header says {} bytes, have {})",
            nbytes,
            buf.len() - data_start
        )));
    }
    let raw = &buf[data_start..data_end];

    // fortran_order=True: data is stored column-major. To present a
    // uniform C-order view to callers, transpose on read. Supported for
    // 0-d, 1-d (no-op), and 2-d arrays — covers all the eagle4 / MLX
    // shapes we currently see (mx.transpose(...).save() lands here).
    let data = if fortran_order && shape.len() >= 2 {
        if shape.len() != 2 {
            return Err(Error::Model(format!(
                "npz: fortran_order with ndim={} not yet supported",
                shape.len()
            )));
        }
        let r = shape[0];
        let c = shape[1];
        let isz = dtype.item_size();
        let mut out = vec![0u8; r * c * isz];
        for i in 0..r {
            for j in 0..c {
                // Fortran source byte offset: (i + j * r) * isz
                // C destination offset:        (i * c + j) * isz
                let src = (i + j * r) * isz;
                let dst = (i * c + j) * isz;
                out[dst..dst + isz].copy_from_slice(&raw[src..src + isz]);
            }
        }
        out
    } else {
        raw.to_vec()
    };

    Ok(NpyArray {
        dtype,
        shape,
        data,
    })
}

fn extract_string_field(header: &str, key: &str) -> Option<String> {
    // Looking for: 'key': '...value...'
    let needle = format!("'{}':", key);
    let kpos = header.find(&needle)?;
    let after = &header[kpos + needle.len()..];
    let q1 = after.find('\'')?;
    let rest = &after[q1 + 1..];
    let q2 = rest.find('\'')?;
    Some(rest[..q2].to_string())
}

fn extract_bool_field(header: &str, key: &str) -> Option<bool> {
    let needle = format!("'{}':", key);
    let kpos = header.find(&needle)?;
    let after = header[kpos + needle.len()..].trim_start();
    if after.starts_with("True") {
        Some(true)
    } else if after.starts_with("False") {
        Some(false)
    } else {
        None
    }
}

fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let needle = "'shape':";
    let kpos = header.find(needle)?;
    let after = &header[kpos + needle.len()..];
    let lp = after.find('(')?;
    let rp = after[lp..].find(')')?;
    let inside = &after[lp + 1..lp + rp];
    let mut out = Vec::new();
    for tok in inside.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.parse::<usize>().ok()?);
    }
    Some(out)
}

fn half_to_f32(bits: u16) -> f32 {
    // IEEE 754 binary16 → binary32. Standard branchless-ish conversion.
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f = match exp {
        0 => {
            if mant == 0 {
                (sign as u32) << 31
            } else {
                // Subnormal: renormalize.
                let mut m = mant as u32;
                let mut e: i32 = -14;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
            }
        }
        0x1f => ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13),
        _ => ((sign as u32) << 31) | (((exp as u32 + 112) as u32) << 23) | ((mant as u32) << 13),
    };
    f32::from_bits(f)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory NPZ (ZIP STORED) containing one `<f4` array.
    fn build_synthetic_npz(name: &str, shape: &[usize], data: &[f32]) -> Vec<u8> {
        let numel: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        assert_eq!(numel, data.len(), "synthetic npz: data len mismatch");

        // 1. NPY payload (v1 header).
        let shape_str = if shape.is_empty() {
            "()".to_string()
        } else if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let inner = shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({})", inner)
        };
        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}", shape_str);
        // NPY spec: total preamble (magic+version+hlen+header) must be 64-byte aligned.
        let preamble = 6 + 2 + 2 + header.len();
        let pad = (64 - (preamble % 64)) % 64;
        for _ in 0..pad.saturating_sub(1) {
            header.push(' ');
        }
        header.push('\n');
        let hlen = header.len() as u16;

        let mut npy = Vec::new();
        npy.extend_from_slice(b"\x93NUMPY");
        npy.push(1);
        npy.push(0);
        npy.extend_from_slice(&hlen.to_le_bytes());
        npy.extend_from_slice(header.as_bytes());
        for v in data {
            npy.extend_from_slice(&v.to_le_bytes());
        }

        // 2. ZIP STORED entry wrapping the NPY.
        let zip_name = format!("{}.npy", name);
        let crc = crc32(&npy);
        let usize_npy = npy.len() as u32;

        let mut zip = Vec::new();
        // Local file header
        zip.extend_from_slice(&SIG_LFH); // signature
        zip.extend_from_slice(&[20, 0]); // version needed
        zip.extend_from_slice(&[0, 0]); // flags
        zip.extend_from_slice(&[0, 0]); // compression = STORED
        zip.extend_from_slice(&[0, 0, 0, 0]); // mod time/date
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&usize_npy.to_le_bytes()); // compressed size
        zip.extend_from_slice(&usize_npy.to_le_bytes()); // uncompressed size
        zip.extend_from_slice(&(zip_name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&[0, 0]); // extra len
        let lfh_offset = 0u32;
        zip.extend_from_slice(zip_name.as_bytes());
        zip.extend_from_slice(&npy);

        // Central directory header
        let cd_start = zip.len() as u32;
        zip.extend_from_slice(&SIG_CDH);
        zip.extend_from_slice(&[20, 0]); // version made by
        zip.extend_from_slice(&[20, 0]); // version needed
        zip.extend_from_slice(&[0, 0]); // flags
        zip.extend_from_slice(&[0, 0]); // compression
        zip.extend_from_slice(&[0, 0, 0, 0]); // mod time/date
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&usize_npy.to_le_bytes()); // compressed
        zip.extend_from_slice(&usize_npy.to_le_bytes()); // uncompressed
        zip.extend_from_slice(&(zip_name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&[0, 0]); // extra len
        zip.extend_from_slice(&[0, 0]); // comment len
        zip.extend_from_slice(&[0, 0]); // disk
        zip.extend_from_slice(&[0, 0]); // int attrs
        zip.extend_from_slice(&[0, 0, 0, 0]); // ext attrs
        zip.extend_from_slice(&lfh_offset.to_le_bytes());
        zip.extend_from_slice(zip_name.as_bytes());
        let cd_size = zip.len() as u32 - cd_start;

        // EOCD
        zip.extend_from_slice(&SIG_EOCD);
        zip.extend_from_slice(&[0, 0]); // disk
        zip.extend_from_slice(&[0, 0]); // disk with cd
        zip.extend_from_slice(&1u16.to_le_bytes()); // entries on this disk
        zip.extend_from_slice(&1u16.to_le_bytes()); // total entries
        zip.extend_from_slice(&cd_size.to_le_bytes());
        zip.extend_from_slice(&cd_start.to_le_bytes());
        zip.extend_from_slice(&[0, 0]); // comment len
        zip
    }

    fn crc32(data: &[u8]) -> u32 {
        // Standard zlib CRC-32. Slow per-byte impl is fine for tests.
        let mut crc: u32 = 0xFFFF_FFFF;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    #[test]
    fn parse_synthetic_1d_array() {
        let data = vec![1.0_f32, -2.0, 3.5, 0.25];
        let buf = build_synthetic_npz("residual_gate", &[4], &data);
        let parsed = parse_npz(&buf).expect("parse");
        let arr = parsed.get("residual_gate").expect("entry");
        assert_eq!(arr.dtype, NpyDtype::F32);
        assert_eq!(arr.shape, vec![4]);
        assert_eq!(arr.as_f32().unwrap(), data);
    }

    #[test]
    fn parse_synthetic_2d_array() {
        let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let buf = build_synthetic_npz("block.attn_norm", &[2, 3], &data);
        let parsed = parse_npz(&buf).expect("parse");
        let arr = parsed.get("block.attn_norm").expect("entry");
        assert_eq!(arr.shape, vec![2, 3]);
        assert_eq!(arr.numel(), 6);
        assert_eq!(arr.as_f32().unwrap(), data);
    }

    #[test]
    fn scalar_shape_decodes() {
        let buf = build_synthetic_npz("residual_gate", &[1], &[0.05_f32]);
        let parsed = parse_npz(&buf).expect("parse");
        let arr = parsed.get("residual_gate").unwrap();
        assert!((arr.as_f32_scalar().unwrap() - 0.05).abs() < 1e-7);
    }

    #[test]
    fn dtype_descr_rejected() {
        assert!(matches!(
            NpyDtype::from_descr("<f8"),
            Err(Error::Model(_))
        ));
        assert!(matches!(
            NpyDtype::from_descr(">f4"),
            Err(Error::Model(_))
        ));
    }

    #[test]
    fn half_to_f32_basic() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0); // fp16 1.0
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert!((half_to_f32(0x3555) - 1.0 / 3.0).abs() < 1e-3);
    }
}
