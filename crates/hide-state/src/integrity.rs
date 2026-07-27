//! Integrity digests for capsule payloads.
//!
//! A capsule records the digest of its payload so a reader can prove the bytes
//! it loaded are the bytes that were sealed. Both algorithms the Bible names
//! (sec 23) are supported: sha256 and blake3. Each produces a fixed 32-byte
//! digest, so the in-memory form is a `[u8; 32]` tagged by the algorithm that
//! produced it. On the wire the digest serializes as a self-describing tagged
//! hex string of the form `algo:hex`, so a serialized capsule carries the
//! algorithm alongside the bytes and never needs an out-of-band convention.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as _, Sha256};

/// The digest algorithm that produced an [`Integrity`] value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegrityAlgo {
    Sha256,
    Blake3,
}

impl IntegrityAlgo {
    /// The lowercase tag used in the serialized `algo:hex` form.
    pub fn tag(self) -> &'static str {
        match self {
            IntegrityAlgo::Sha256 => "sha256",
            IntegrityAlgo::Blake3 => "blake3",
        }
    }

    /// Parse an algorithm tag, returning `None` for an unknown tag.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "sha256" => Some(IntegrityAlgo::Sha256),
            "blake3" => Some(IntegrityAlgo::Blake3),
            _ => None,
        }
    }
}

/// A payload digest tagged by the algorithm that produced it. Both supported
/// algorithms yield 32 bytes, so the digest is a fixed array rather than a
/// variable buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Integrity {
    pub algo: IntegrityAlgo,
    pub digest: [u8; 32],
}

impl Integrity {
    /// Compute the digest of `bytes` with the given algorithm.
    pub fn compute(algo: IntegrityAlgo, bytes: &[u8]) -> Self {
        match algo {
            IntegrityAlgo::Sha256 => Self::sha256(bytes),
            IntegrityAlgo::Blake3 => Self::blake3(bytes),
        }
    }

    /// Compute a sha256 digest of `bytes`.
    pub fn sha256(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let out = hasher.finalize();
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&out);
        Integrity {
            algo: IntegrityAlgo::Sha256,
            digest,
        }
    }

    /// Compute a blake3 digest of `bytes`.
    pub fn blake3(bytes: &[u8]) -> Self {
        Integrity {
            algo: IntegrityAlgo::Blake3,
            digest: *blake3::hash(bytes).as_bytes(),
        }
    }

    /// Recompute the digest of `bytes` with this value's algorithm and return
    /// whether it matches. This is the check a reader runs to accept or reject
    /// a payload.
    pub fn verify(&self, bytes: &[u8]) -> bool {
        Self::compute(self.algo, bytes) == *self
    }

    /// Render as the self-describing `algo:hex` form.
    pub fn to_tagged_hex(&self) -> String {
        let mut s = String::with_capacity(7 + 64);
        s.push_str(self.algo.tag());
        s.push(':');
        for byte in self.digest {
            s.push_str(&format!("{byte:02x}"));
        }
        s
    }

    /// Parse the `algo:hex` form, returning `None` on any malformed input.
    pub fn from_tagged_hex(s: &str) -> Option<Self> {
        let (tag, hex) = s.split_once(':')?;
        let algo = IntegrityAlgo::from_tag(tag)?;
        if hex.len() != 64 {
            return None;
        }
        let mut digest = [0u8; 32];
        for (i, slot) in digest.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Integrity { algo, digest })
    }
}

impl Serialize for Integrity {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.to_tagged_hex())
    }
}

impl<'de> Deserialize<'de> for Integrity {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Integrity::from_tagged_hex(&s)
            .ok_or_else(|| D::Error::custom("invalid integrity tagged-hex digest"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_and_blake3_differ_and_are_stable() {
        let bytes = b"synthetic capsule payload";
        let a = Integrity::sha256(bytes);
        let b = Integrity::blake3(bytes);
        assert_eq!(a.algo, IntegrityAlgo::Sha256);
        assert_eq!(b.algo, IntegrityAlgo::Blake3);
        assert_ne!(a.digest, b.digest);
        // Deterministic recompute.
        assert_eq!(a, Integrity::sha256(bytes));
        assert_eq!(b, Integrity::blake3(bytes));
    }

    #[test]
    fn verify_accepts_original_rejects_mutated() {
        let bytes = vec![1u8, 2, 3, 4, 5];
        for algo in [IntegrityAlgo::Sha256, IntegrityAlgo::Blake3] {
            let integ = Integrity::compute(algo, &bytes);
            assert!(integ.verify(&bytes));
            let mut flipped = bytes.clone();
            flipped[2] ^= 0x01;
            assert!(!integ.verify(&flipped));
        }
    }

    #[test]
    fn tagged_hex_roundtrips() {
        for algo in [IntegrityAlgo::Sha256, IntegrityAlgo::Blake3] {
            let integ = Integrity::compute(algo, b"abc");
            let text = integ.to_tagged_hex();
            assert!(text.starts_with(algo.tag()));
            assert_eq!(Integrity::from_tagged_hex(&text), Some(integ));
        }
        assert_eq!(Integrity::from_tagged_hex("nope"), None);
        assert_eq!(Integrity::from_tagged_hex("md5:00"), None);
    }

    #[test]
    fn serde_is_a_plain_string() {
        let integ = Integrity::blake3(b"payload");
        let json = serde_json::to_string(&integ).unwrap();
        assert!(json.starts_with("\"blake3:"));
        let back: Integrity = serde_json::from_str(&json).unwrap();
        assert_eq!(integ, back);
    }
}
