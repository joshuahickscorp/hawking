use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use crate::sha256::sha256;

pub const MANIFEST_VERSION: u32 = 1;

/// Domain/version for the tensor identity digest used by quantize-model.
/// Bump whenever the identity serialization changes.
pub const TENSOR_IDENTITY_SCHEMA: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestEntry {
    pub name: String,
    pub bits: u32,

    pub sha256_hex: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    pub config_key: String,
    pub entries: BTreeMap<String, ManifestEntry>,
}

pub fn hash_tensor_bytes(bytes: &[u8]) -> [u8; 32] {
    sha256(bytes)
}

pub fn hash_f32(vals: &[f32]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    sha256(&bytes)
}

/// Hash tensor values together with their exact shape under a versioned domain.
/// Shape binding matters even when the flattened bytes are identical: row/column
/// geometry (including `in_features`) changes RHT and block interpretation.
pub fn hash_f32_tensor(vals: &[f32], shape: &[u64]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(48 + shape.len() * 8 + vals.len() * 4);
    bytes.extend_from_slice(b"hawking.strand.tensor-identity\0");
    bytes.extend_from_slice(&TENSOR_IDENTITY_SCHEMA.to_le_bytes());
    bytes.extend_from_slice(&(shape.len() as u64).to_le_bytes());
    for &dim in shape {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    bytes.extend_from_slice(&(vals.len() as u64).to_le_bytes());
    for &value in vals {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256(&bytes)
}

pub fn hex32(d: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

pub fn should_skip(name: &str, tensor_bytes_hash: &[u8; 32], bits: u32, config_key: &str, prior: Option<&Manifest>) -> bool {
    let Some(m) = prior else { return false };
    if m.config_key != config_key {
        return false;
    }
    match m.entries.get(name) {
        Some(e) => e.bits == bits && e.sha256_hex == hex32(tensor_bytes_hash),
        None => false,
    }
}

impl Manifest {
    pub fn new(config_key: impl Into<String>) -> Self {
        Manifest { config_key: config_key.into(), entries: BTreeMap::new() }
    }

    pub fn record(&mut self, name: &str, tensor_bytes_hash: &[u8; 32], bits: u32) {
        assert!(!name.contains('"') && !name.contains('\\'), "tensor name {name:?} needs JSON escaping — not supported");
        self.entries.insert(name.to_string(), ManifestEntry { name: name.to_string(), bits, sha256_hex: hex32(tensor_bytes_hash) });
    }

    pub fn lookup(&self, name: &str) -> Option<&ManifestEntry> {
        self.entries.get(name)
    }

    pub fn to_json(&self) -> String {
        assert!(!self.config_key.contains('"') && !self.config_key.contains('\\'), "config_key needs JSON escaping — not supported");
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"version\": {},\n", MANIFEST_VERSION));
        s.push_str(&format!("  \"config_key\": \"{}\",\n", self.config_key));
        s.push_str("  \"tensors\": [\n");
        for (i, e) in self.entries.values().enumerate() {
            if i > 0 {
                s.push_str(",\n");
            }
            s.push_str(&format!("    {{\"name\": \"{}\", \"bits\": {}, \"sha256\": \"{}\"}}", e.name, e.bits, e.sha256_hex));
        }
        s.push_str("\n  ]\n}\n");
        s
    }

    pub fn from_json(s: &str) -> Option<Manifest> {
        if extract_uint(s, "version")? != MANIFEST_VERSION as u64 {
            return None;
        }
        let config_key = extract_str(s, "config_key")?;
        let mut m = Manifest::new(config_key);

        let arr_start = s.find("\"tensors\"")?;
        let body = &s[arr_start..];
        let b = body.as_bytes();
        let mut i = 0usize;
        while i < b.len() {
            if b[i] == b'{' {
                let start = i;
                let mut depth = 0usize;
                while i < b.len() {
                    match b[i] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                let obj = &body[start..i];
                let (name, bits, sha) = (extract_str(obj, "name")?, extract_uint(obj, "bits")?, extract_str(obj, "sha256")?);
                if sha.len() != 64 || !sha.bytes().all(|c| c.is_ascii_hexdigit()) {
                    return None;
                }
                m.entries.insert(name.clone(), ManifestEntry { name, bits: bits as u32, sha256_hex: sha });
            } else {
                i += 1;
            }
        }
        Some(m)
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Option<Manifest>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(None);
        }
        let s = fs::read_to_string(path)?;
        Ok(Manifest::from_json(&s))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, self.to_json())?;
        fs::rename(&tmp, path)
    }
}

fn extract_str(obj: &str, field: &str) -> Option<String> {
    let pat = format!("\"{field}\"");
    let p = obj.find(&pat)? + pat.len();
    let rest = &obj[p..];
    let q1 = rest.find('"')?;
    let after = &rest[q1 + 1..];
    let q2 = after.find('"')?;
    Some(after[..q2].to_string())
}

fn extract_uint(obj: &str, field: &str) -> Option<u64> {
    let pat = format!("\"{field}\"");
    let p = obj.find(&pat)? + pat.len();
    let rest = &obj[p..];
    let rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ':');
    rest.split(|c: char| !c.is_ascii_digit()).next().and_then(|t| t.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(bytes: &[u8]) -> [u8; 32] {
        hash_tensor_bytes(bytes)
    }

    #[test]
    fn hash_is_deterministic_and_sensitive() {
        let a = h(b"hello tensor");
        assert_eq!(a, h(b"hello tensor"));
        assert_ne!(a, h(b"hello tensoR"));
        assert_ne!(a, h(b"hello tensor "));
    }

    #[test]
    fn hash_f32_matches_le_byte_serialization() {
        let vals = [1.0f32, -2.5, 0.0, f32::MIN_POSITIVE];
        let mut bytes = Vec::new();
        for v in &vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(hash_f32(&vals), h(&bytes));

        let mut vals2 = vals;
        vals2[1] = f32::from_bits(vals2[1].to_bits() ^ 1);
        assert_ne!(hash_f32(&vals), hash_f32(&vals2));
    }

    #[test]
    fn tensor_identity_binds_shape_and_in_features() {
        let vals: Vec<f32> = (0..12).map(|i| i as f32 * 0.25).collect();
        let a = hash_f32_tensor(&vals, &[3, 4]);
        assert_eq!(a, hash_f32_tensor(&vals, &[3, 4]));
        assert_ne!(a, hash_f32_tensor(&vals, &[2, 6]));
        assert_ne!(a, hash_f32_tensor(&vals, &[12]));
    }

    #[test]
    fn should_skip_requires_everything_to_match() {
        let key = "bits=2,l=12,rht=true,out=1.0@8";
        let mut m = Manifest::new(key);
        let hash = h(b"frozen down_proj bytes");
        m.record("model.layers.0.mlp.down_proj.weight", &hash, 2);

        assert!(should_skip("model.layers.0.mlp.down_proj.weight", &hash, 2, key, Some(&m)));

        assert!(!should_skip("model.layers.0.mlp.down_proj.weight", &hash, 2, key, None));

        let other = h(b"the shadow moved");
        assert!(!should_skip("model.layers.0.mlp.down_proj.weight", &other, 2, key, Some(&m)));

        assert!(!should_skip("model.layers.0.mlp.down_proj.weight", &hash, 3, key, Some(&m)));

        assert!(!should_skip("model.layers.0.mlp.down_proj.weight", &hash, 2, "bits=2,l=13,rht=true,out=1.0@8", Some(&m)));

        assert!(!should_skip("model.layers.1.mlp.down_proj.weight", &hash, 2, key, Some(&m)));
    }

    #[test]
    fn json_roundtrip_is_lossless() {
        let mut m = Manifest::new("bits=2,l=12");
        m.record("model.layers.0.mlp.down_proj.weight", &h(b"a"), 2);
        m.record("model.layers.1.self_attn.q_proj.weight", &h(b"b"), 3);
        let s = m.to_json();
        let back = Manifest::from_json(&s).expect("parse own output");
        assert_eq!(m, back);

        assert_eq!(s, back.to_json());
    }

    #[test]
    fn load_missing_or_garbage_is_a_cache_miss() {
        assert!(Manifest::load("/nonexistent/path/manifest.json").unwrap().is_none());
        assert!(Manifest::from_json("not json at all").is_none());
        assert!(Manifest::from_json("{\"version\": 999, \"config_key\": \"x\", \"tensors\": []}").is_none());

        assert!(Manifest::from_json(
            "{\"version\": 1, \"config_key\": \"k\", \"tensors\": [\
             {\"name\": \"t\", \"bits\": 2, \"sha256\": \"deadbeef\"}]}"
        )
        .is_none());
    }

    #[test]
    fn save_load_roundtrip_on_disk() {
        let dir = std::env::temp_dir().join(format!("strand-encode-cache-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.json");
        let mut m = Manifest::new("bits=2,l=12,out=1.0@8");
        let hash = h(b"tensor bytes");
        m.record("model.layers.5.mlp.down_proj.weight", &hash, 2);
        m.save(&path).unwrap();
        let back = Manifest::load(&path).unwrap().expect("manifest present");
        assert_eq!(m, back);
        assert!(should_skip("model.layers.5.mlp.down_proj.weight", &hash, 2, "bits=2,l=12,out=1.0@8", Some(&back)));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_manifest_never_skips() {
        let m = Manifest::new("k");
        assert!(!should_skip("anything", &h(b"x"), 2, "k", Some(&m)));
        assert_eq!(m.lookup("anything"), None);
    }
}
