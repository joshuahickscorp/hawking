//! Content hash is the object identity.
//!
//! Two ingestions of identical bytes produce one object (same [`ContentHash`])
//! and two independent [`crate::schema::ObjectRef`] records pointing at it.
//! The reverse is also true: distinct bytes never share a hash under blake3.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Read;

/// Fixed streaming chunk size for hashing and persistence.
///
/// A multi-gigabyte object is never loaded whole: each stage reads at most
/// [`CHUNK_SIZE`] bytes of working buffer at a time.
pub const CHUNK_SIZE: usize = 256 * 1024; // 256 KiB

/// blake3 content hash — the sole identity of an object body.
///
/// Wire form: `blake3:<64-hex>`. Stable across processes and platforms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentHash(pub String);

impl ContentHash {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Hash the full slice. Prefer [`hash_reader`] for large bodies.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let hex = blake3::hash(bytes).to_hex();
        Self(format!("blake3:{hex}"))
    }

    /// Stream-hash a reader. Peak buffer is [`CHUNK_SIZE`] regardless of length.
    ///
    /// Returns `(hash, size_bytes, peak_buffer_bytes)` so callers can prove the
    /// streaming bound in tests.
    pub fn of_reader<R: Read>(mut reader: R) -> std::io::Result<(Self, u64, usize)> {
        let mut hasher = blake3::Hasher::new();
        let mut buf = vec![0u8; CHUNK_SIZE];
        let mut size: u64 = 0;
        let mut peak: usize = 0;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            peak = peak.max(n);
            hasher.update(&buf[..n]);
            size += n as u64;
        }
        let hex = hasher.finalize().to_hex();
        Ok((Self(format!("blake3:{hex}")), size, peak))
    }

    pub fn is_well_formed(&self) -> bool {
        self.0.starts_with("blake3:") && self.0.len() == "blake3:".len() + 64
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ContentHash {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for ContentHash {
    fn from(s: String) -> Self {
        Self(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn same_bytes_same_hash() {
        let a = ContentHash::of_bytes(b"hello-you-object");
        let b = ContentHash::of_bytes(b"hello-you-object");
        assert_eq!(a, b);
        assert!(a.is_well_formed());
    }

    #[test]
    fn different_bytes_different_hash() {
        let a = ContentHash::of_bytes(b"alpha");
        let b = ContentHash::of_bytes(b"beta");
        assert_ne!(a, b);
    }

    #[test]
    fn reader_matches_bytes_and_bounds_buffer() {
        let payload = vec![0xABu8; CHUNK_SIZE * 3 + 17];
        let from_slice = ContentHash::of_bytes(&payload);
        let (from_reader, size, peak) =
            ContentHash::of_reader(Cursor::new(payload.clone())).unwrap();
        assert_eq!(from_slice, from_reader);
        assert_eq!(size, payload.len() as u64);
        assert!(peak <= CHUNK_SIZE);
    }
}
