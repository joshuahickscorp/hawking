use std::collections::HashMap;
#[cfg(all(feature = "native-execution", target_os = "macos"))]
use std::fs::File;
#[cfg(all(feature = "native-execution", target_os = "macos"))]
use std::io::Read as _;
#[cfg(all(feature = "native-execution", target_os = "macos"))]
use std::os::unix::fs::MetadataExt;
#[cfg(feature = "native-execution")]
use std::path::Path;
#[cfg(all(feature = "native-execution", target_os = "macos"))]
use std::path::PathBuf;

#[cfg(all(feature = "native-execution", target_os = "macos"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    dev: u64,
    ino: u64,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
}

#[cfg(all(feature = "native-execution", target_os = "macos"))]
fn file_identity(path: &Path) -> std::io::Result<FileIdentity> {
    let metadata = std::fs::metadata(path)?;
    Ok(metadata_identity(&metadata))
}

#[cfg(all(feature = "native-execution", target_os = "macos"))]
fn metadata_identity(metadata: &std::fs::Metadata) -> FileIdentity {
    FileIdentity { dev: metadata.dev(), ino: metadata.ino(), size: metadata.size(), mtime_sec: metadata.mtime(), mtime_nsec: metadata.mtime_nsec() }
}

enum SafeTensorBacking {
    Owned(Vec<u8>),
    #[cfg(all(feature = "native-execution", target_os = "macos"))]
    Mapped(memmap2::Mmap),
}

impl SafeTensorBacking {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            #[cfg(all(feature = "native-execution", target_os = "macos"))]
            Self::Mapped(bytes) => bytes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StTensor {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub off_start: usize,
    pub off_end: usize,
}

pub struct SafeTensors {
    backing: SafeTensorBacking,
    pub data_start: usize,
    pub tensors: HashMap<String, StTensor>,
    pub order: Vec<String>,
    #[cfg(all(feature = "native-execution", target_os = "macos"))]
    source_identity: Option<(PathBuf, FileIdentity)>,
}

impl SafeTensors {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Ok(Self::from_backing(SafeTensorBacking::Owned(bytes)))
    }

    #[cfg(feature = "native-execution")]
    pub fn open_preallocated(path: &Path) -> std::io::Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let before = file_identity(path)?;
            let mut file = File::open(path)?;
            let fd_before = metadata_identity(&file.metadata()?);
            let after_open = file_identity(path)?;
            if before != fd_before || fd_before != after_open {
                return Err(std::io::Error::other("source identity changed while opening owned input"));
            }
            let len = usize::try_from(fd_before.size).map_err(|_| std::io::Error::other("source length does not fit usize"))?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(len).map_err(|error| std::io::Error::other(format!("cannot reserve {len} source bytes: {error}")))?;
            bytes.resize(len, 0);
            file.read_exact(&mut bytes)?;
            let fd_after = metadata_identity(&file.metadata()?);
            let path_after = file_identity(path)?;
            if fd_before != fd_after || fd_after != path_after {
                return Err(std::io::Error::other("source identity changed during owned input read"));
            }
            let mut tensors = Self::from_backing(SafeTensorBacking::Owned(bytes));
            tensors.source_identity = Some((path.to_path_buf(), fd_after));
            return Ok(tensors);
        }
        #[cfg(not(target_os = "macos"))]
        {
            let bytes = crate::native_io::read_preallocated(path)?;
            Ok(Self::from_backing(SafeTensorBacking::Owned(bytes)))
        }
    }

    #[cfg(all(feature = "native-execution", target_os = "macos"))]
    pub fn open_mmap(path: &Path) -> std::io::Result<Self> {
        let before = file_identity(path)?;
        let file = File::open(path)?;
        let fd_identity = metadata_identity(&file.metadata()?);
        // Safety: the mapping is read-only and is owned by `SafeTensors` for at
        // least as long as every slice returned by `raw`/`bytes`.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let after = file_identity(path)?;
        if before != fd_identity || fd_identity != after || mmap.len() as u64 != after.size {
            return Err(std::io::Error::other("source identity changed while establishing mmap"));
        }
        let mut tensors = Self::from_backing(SafeTensorBacking::Mapped(mmap));
        tensors.source_identity = Some((path.to_path_buf(), fd_identity));
        Ok(tensors)
    }

    #[cfg(all(feature = "native-execution", target_os = "macos"))]
    pub fn verify_source_identity(&self) -> std::io::Result<()> {
        let Some((path, expected)) = &self.source_identity else {
            return Ok(());
        };
        let actual = file_identity(path)?;
        if actual != *expected || self.backing.as_slice().len() as u64 != actual.size {
            return Err(std::io::Error::other(format!("native source identity changed: {}", path.display())));
        }
        Ok(())
    }

    fn from_backing(backing: SafeTensorBacking) -> Self {
        let bytes = backing.as_slice();
        let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let json = bytes[8..8 + hdr_len].to_vec();
        let data_start = 8 + hdr_len;
        let (tensors, order) = parse_header(&json);
        SafeTensors {
            backing,
            data_start,
            tensors,
            order,
            #[cfg(all(feature = "native-execution", target_os = "macos"))]
            source_identity: None,
        }
    }

    pub fn raw(&self, t: &StTensor) -> &[u8] {
        &self.backing.as_slice()[self.data_start + t.off_start..self.data_start + t.off_end]
    }

    pub fn bytes(&self) -> &[u8] {
        self.backing.as_slice()
    }

    pub fn to_f32(&self, t: &StTensor) -> Vec<f32> {
        let raw = self.raw(t);
        match t.dtype.as_str() {
            "BF16" => raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            other => panic!("unsupported safetensors dtype {other}"),
        }
    }
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e = -1i32;
        let mut m = mant;
        loop {
            e += 1;
            m <<= 1;
            if m & 0x0400 != 0 {
                break;
            }
        }
        let mant32 = (m & 0x03ff) << 13;
        let exp32 = ((127 - 15 - e) as u32) << 23;
        f32::from_bits(sign | exp32 | mant32)
    } else if exp == 0x1f {
        f32::from_bits(sign | 0x7f80_0000 | (mant << 13))
    } else {
        let exp32 = ((exp as i32 - 15 + 127) as u32) << 23;
        let mant32 = mant << 13;
        f32::from_bits(sign | exp32 | mant32)
    }
}

pub fn parse_header(json: &[u8]) -> (HashMap<String, StTensor>, Vec<String>) {
    let s = std::str::from_utf8(json).expect("header is utf8");
    let b = s.as_bytes();
    let mut map = HashMap::new();
    let mut order = Vec::new();
    let mut i = 0usize;

    fn read_string(b: &[u8], mut i: usize) -> (String, usize) {
        debug_assert_eq!(b[i], b'"');
        i += 1;
        let start = i;
        while b[i] != b'"' {
            i += 1;
        }
        (String::from_utf8_lossy(&b[start..i]).into_owned(), i + 1)
    }

    while i < b.len() {
        if b[i] != b'"' {
            i += 1;
            continue;
        }
        let (key, ni) = read_string(b, i);
        i = ni;
        while i < b.len() && b[i] != b'{' && b[i] != b'"' && b[i] != b'[' {
            i += 1;
        }
        if key == "__metadata__" {
            let mut depth = 0;
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
            continue;
        }
        if i >= b.len() || b[i] != b'{' {
            continue;
        }
        let obj_start = i;
        let mut depth = 0;
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
        let obj = &s[obj_start..i];
        let dtype = extract_str_field(obj, "dtype").unwrap_or_default();
        let shape = extract_num_array(obj, "shape");
        let offs = extract_num_array(obj, "data_offsets");
        if offs.len() == 2 {
            map.insert(key.clone(), StTensor { dtype, shape, off_start: offs[0] as usize, off_end: offs[1] as usize });
            order.push(key);
        }
    }
    (map, order)
}

pub fn extract_str_field(obj: &str, field: &str) -> Option<String> {
    let pat = format!("\"{field}\"");
    let p = obj.find(&pat)? + pat.len();
    let rest = &obj[p..];
    let q1 = rest.find('"')?;
    let after = &rest[q1 + 1..];
    let q2 = after.find('"')?;
    Some(after[..q2].to_string())
}

pub fn extract_num_array(obj: &str, field: &str) -> Vec<u64> {
    let pat = format!("\"{field}\"");
    let Some(p) = obj.find(&pat) else {
        return Vec::new();
    };
    let rest = &obj[p + pat.len()..];
    let Some(lb) = rest.find('[') else {
        return Vec::new();
    };
    let Some(rb) = rest.find(']') else {
        return Vec::new();
    };
    rest[lb + 1..rb].split(',').filter_map(|t| t.trim().parse::<u64>().ok()).collect()
}

#[cfg(all(test, feature = "native-execution", target_os = "macos"))]
mod native_tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn mmap_identity_guard_detects_source_change() {
        let path = std::env::temp_dir().join(format!("strand-mmap-identity-{}-{}.safetensors", std::process::id(), std::thread::current().name().unwrap_or("test")));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(b"{}");
        std::fs::write(&path, &bytes).unwrap();
        let mapped = SafeTensors::open_mmap(&path).unwrap();
        assert_eq!(mapped.bytes(), bytes);
        let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[0]).unwrap();
        file.sync_all().unwrap();
        assert!(mapped.verify_source_identity().is_err());
        drop(mapped);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn owned_identity_guard_detects_source_change() {
        let path = std::env::temp_dir().join(format!("strand-owned-identity-{}-{}.safetensors", std::process::id(), std::thread::current().name().unwrap_or("test")));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(b"{}");
        std::fs::write(&path, &bytes).unwrap();
        let owned = SafeTensors::open_preallocated(&path).unwrap();
        let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[0]).unwrap();
        file.sync_all().unwrap();
        assert!(owned.verify_source_identity().is_err());
        let _ = std::fs::remove_file(path);
    }
}
