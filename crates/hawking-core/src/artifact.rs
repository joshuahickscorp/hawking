//! Core A `.gravity` authority: bin/layout, one-shot owned ABI, sort_keys headers.
macro_rules! le_u16 {
    ($b:expr, $o:expr) => {
        u16::from_le_bytes($b[$o..$o + 2].try_into().unwrap())
    };
}
macro_rules! le_u32 {
    ($b:expr, $o:expr) => {
        u32::from_le_bytes($b[$o..$o + 4].try_into().unwrap())
    };
}
macro_rules! le_u64 {
    ($b:expr, $o:expr) => {
        u64::from_le_bytes($b[$o..$o + 8].try_into().unwrap())
    };
}
macro_rules! require_magic {
    ($b:expr, $magic:expr, $label:expr) => {{
        let b: &[u8] = $b;
        let magic: &[u8] = $magic;
        if b.len() < magic.len() || &b[..magic.len()] != magic {
            Err(crate::Error::Gravity(format!(
                "bad {} magic {:?}, expected {magic:?}",
                $label,
                b.get(..magic.len()).unwrap_or(&[])
            )))
        } else {
            Ok(())
        }
    }};
}
macro_rules! checked_end {
    ($offset:expr, $bytes:expr, $limit:expr, $label:expr) => {{
        let offset: u64 = $offset;
        let bytes: u64 = $bytes;
        let limit: u64 = $limit;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| crate::Error::Gravity(format!("{}: ovf", $label)))?;
        if end > limit {
            return Err(crate::Error::Gravity(format!("{}: past {limit}", $label)));
        }
        end
    }};
}
macro_rules! section_end {
    ($base:expr, $count:expr, $unit:expr, $label:expr) => {{
        let base: usize = $base;
        let count: usize = $count;
        let unit: usize = $unit;
        let add = count
            .checked_mul(unit)
            .ok_or_else(|| crate::Error::Gravity(format!("{}: size ovf", $label)))?;
        base.checked_add(add)
            .ok_or_else(|| crate::Error::Gravity(format!("{}: end ovf", $label)))?
    }};
}

#[path = "artifact_aap.rs"]
pub(crate) mod aap;
#[path = "artifact_pq.rs"]
pub mod pq;

pub(crate) mod bin {
    pub(in crate::artifact) fn hex_encode(bytes: &[u8]) -> String {
        const H: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            s.push(H[(b >> 4) as usize] as char);
            s.push(H[(b & 0xf) as usize] as char);
        }
        s
    }
    pub(in crate::artifact) fn canonical_json(value: &serde_json::Value) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        write_canonical(&mut out, value);
        out
    }
    fn write_canonical(out: &mut Vec<u8>, value: &serde_json::Value) {
        use serde_json::Value;
        match value {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(true) => out.extend_from_slice(b"true"),
            Value::Bool(false) => out.extend_from_slice(b"false"),
            Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
            Value::String(s) => out.extend_from_slice(serde_json::to_string(s).unwrap().as_bytes()),
            Value::Array(a) => {
                out.push(b'[');
                for (i, el) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    write_canonical(out, el);
                }
                out.push(b']');
            }
            Value::Object(m) => {
                out.push(b'{');
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.extend_from_slice(serde_json::to_string(k).unwrap().as_bytes());
                    out.push(b':');
                    write_canonical(out, &m[k]);
                }
                out.push(b'}');
            }
        }
    }
}
mod container {
    use super::bin::{canonical_json, hex_encode};
    use crate::{Error, Result};
    use memmap2::Mmap;
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::path::Path;
    const MAGIC: &[u8; 8] = b"GRAVITY\0";
    const MAX_FORMAT_VERSION: u32 = 1;
    const PREFIX_LEN: usize = 20;
    const HEADER_SCHEMA: &str = "hawking.gravity.shard_header.v1";
    const FORMAT_VERSION: u32 = 1;
    #[derive(Debug, Clone, Deserialize)]
    pub struct TensorDescriptor {
        pub name: String,
        #[serde(default)]
        pub codec: String,
        pub offset: u64,
        pub bytes: u64,
        pub sha256: String,
        #[serde(default)]
        pub shape: Vec<u64>,
        #[serde(default)]
        pub elements: u64,
    }
    #[derive(Deserialize)]
    struct GravityHeader {
        tensors: Vec<TensorDescriptor>,
    }
    pub struct GravityShard {
        mmap: Mmap,
        pub(super) body_offset: u64,
        pub(crate) tensors: HashMap<String, TensorDescriptor>,
        tensor_order: Vec<String>,
        pub extra: serde_json::Value,
        pub(super) full_header: serde_json::Value,
    }
    impl GravityShard {
        pub fn open(path: &Path) -> Result<GravityShard> {
            let f = File::open(path)?;
            // Safety: read-only for GravityShard lifetime; truncation is caller error.
            let mmap = unsafe { Mmap::map(&f)? };
            let (header_end, header_value) = parse_prefix_header(&mmap)?;
            let header: GravityHeader = serde_json::from_value(header_value.clone())
                .map_err(|e| Error::Gravity(format!("tensors parse: {e}")))?;
            let mut extra = header_value.clone();
            if let Some(obj) = extra.as_object_mut() {
                obj.remove("tensors");
            }
            let mut tensors = HashMap::with_capacity(header.tensors.len());
            let mut tensor_order = Vec::with_capacity(header.tensors.len());
            for d in header.tensors {
                if tensors.contains_key(&d.name) {
                    return Err(Error::Gravity(format!("dup tensor {:?}", d.name)));
                }
                tensor_order.push(d.name.clone());
                tensors.insert(d.name.clone(), d);
            }
            Ok(GravityShard {
                mmap,
                body_offset: header_end,
                tensors,
                tensor_order,
                extra,
                full_header: header_value,
            })
        }
        pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
            self.tensor_order.iter().map(String::as_str)
        }
        pub fn descriptor(&self, name: &str) -> Option<&TensorDescriptor> {
            self.tensors.get(name)
        }
        fn payload_range(&self, name: &str, len: Option<u64>) -> Result<(usize, usize)> {
            let d = self
                .descriptor(name)
                .ok_or_else(|| Error::Gravity(format!("no tensor {name:?}")))?;
            let take = len.unwrap_or(d.bytes);
            if take > d.bytes {
                return Err(Error::Gravity(format!(
                    "{name}: payload {} < {take}",
                    d.bytes
                )));
            }
            let start = self
                .body_offset
                .checked_add(d.offset)
                .ok_or_else(|| Error::Gravity(format!("{name}: off ovf")))?;
            let end = checked_end!(start, take, self.mmap.len() as u64, name);
            Ok((start as usize, end as usize))
        }
        /// Two ContainerLookup scopes; no TensorDescriptor clone on the hot path.
        pub fn read_tensor(&self, name: &str, verify_hash: bool) -> Result<Vec<u8>> {
            use crate::cost_ledger::{self, Bucket};
            let d = {
                let _lookup = cost_ledger::Scope::new(Bucket::ContainerLookup);
                self.descriptor(name)
                    .ok_or_else(|| Error::Gravity(format!("no tensor {name:?}")))?
            };
            let start = self
                .body_offset
                .checked_add(d.offset)
                .ok_or_else(|| Error::Gravity(format!("{name}: off ovf")))?;
            let end = checked_end!(start, d.bytes, self.mmap.len() as u64, name);
            let payload = {
                let _lookup = cost_ledger::Scope::new(Bucket::ContainerLookup);
                &self.mmap[start as usize..end as usize]
            };
            if verify_hash {
                let _v = cost_ledger::Scope::new(Bucket::ArtifactVerificationAndSha);
                cost_ledger::record_sha_verification();
                let hex = hex_encode(Sha256::digest(payload).as_slice());
                if hex != d.sha256 {
                    return Err(Error::Gravity(format!(
                        "{name}: sha256 mismatch: expected {}, got {hex}",
                        d.sha256
                    )));
                }
            }
            cost_ledger::record_allocation(payload.len() as u64);
            Ok(payload.to_vec())
        }
        pub(crate) fn read_tensor_for_prefix(
            &self,
            name: &str,
            prefix_len: usize,
        ) -> Result<Vec<u8>> {
            use crate::cost_ledger::{self, Bucket};
            let _l = cost_ledger::Scope::new(Bucket::ContainerLookup);
            let (start, end) = self.payload_range(name, Some(prefix_len as u64))?;
            cost_ledger::record_allocation((end - start) as u64);
            Ok(self.mmap[start..end].to_vec())
        }
    }
    fn parse_prefix_header(mmap: &Mmap) -> Result<(u64, serde_json::Value)> {
        if mmap.len() < PREFIX_LEN {
            return Err(Error::Gravity(format!("prefix short: {}", mmap.len())));
        }
        require_magic!(&mmap[..8], MAGIC, "gravity")?;
        let format_version = le_u32!(mmap, 8);
        if format_version > MAX_FORMAT_VERSION {
            return Err(Error::Gravity(format!("bad version {format_version}")));
        }
        let header_len = le_u64!(mmap, 12);
        let header_end = checked_end!(PREFIX_LEN as u64, header_len, mmap.len() as u64, "header");
        let header_value = serde_json::from_slice(&mmap[PREFIX_LEN..header_end as usize])
            .map_err(|e| Error::Gravity(format!("hdr parse: {e}")))?;
        Ok((header_end, header_value))
    }
    pub(super) fn plan_shard(
        payloads: &[(serde_json::Value, &[u8])],
        meta: &serde_json::Value,
    ) -> Result<(serde_json::Value, Vec<u8>)> {
        let mut body_hasher = Sha256::new();
        let mut tensors = Vec::with_capacity(payloads.len());
        let mut offset = 0u64;
        let mut seen = HashSet::with_capacity(payloads.len());
        for (descriptor, blob) in payloads {
            body_hasher.update(blob);
            let mut entry = match descriptor.clone() {
                serde_json::Value::Object(m) => m,
                _ => return Err(Error::Gravity("desc not object".into())),
            };
            let name = entry
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !name.is_empty() && !seen.insert(name.clone()) {
                return Err(Error::Gravity(format!("write: dup tensor {name:?}")));
            }
            entry.insert("offset".into(), offset.into());
            entry.insert("bytes".into(), (blob.len() as u64).into());
            entry.insert(
                "sha256".into(),
                hex_encode(Sha256::digest(blob).as_slice()).into(),
            );
            tensors.push(serde_json::Value::Object(entry));
            offset = offset
                .checked_add(blob.len() as u64)
                .ok_or_else(|| Error::Gravity("write: body off ovf".into()))?;
        }
        let meta_or = |k: &str| {
            meta.get(k)
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}))
        };
        let header = serde_json::json!({
            "schema": HEADER_SCHEMA,
            "format_version": FORMAT_VERSION,
            "model": meta_or("model"),
            "architecture": meta_or("architecture"),
            "tokenizer": meta_or("tokenizer"),
            "compression": meta_or("compression"),
            "shard": meta_or("shard"),
            "integrity": {
                "body_sha256": hex_encode(body_hasher.finalize().as_slice()),
                "tensor_count": tensors.len() as u64
            },
            "tensors": tensors,
        });
        Ok((header.clone(), canonical_json(&header)))
    }
    pub(super) fn commit_shard(
        path: &Path,
        payloads: &[(serde_json::Value, &[u8])],
        encoded: &[u8],
    ) -> Result<()> {
        let mut tmp = path.as_os_str().to_os_string();
        tmp.push(".tmp");
        let tmp = Path::new(&tmp).to_path_buf();
        let write_res = (|| {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)
                .map_err(|e| Error::Gravity(format!("{}: create tmp: {e}", tmp.display())))?;
            f.write_all(MAGIC)
                .and_then(|_| f.write_all(&FORMAT_VERSION.to_le_bytes()))
                .and_then(|_| f.write_all(&(encoded.len() as u64).to_le_bytes()))
                .and_then(|_| f.write_all(encoded))
                .map_err(|e| Error::Gravity(format!("{}: write tmp: {e}", tmp.display())))?;
            for (_, blob) in payloads {
                f.write_all(blob)
                    .map_err(|e| Error::Gravity(format!("{}: write body: {e}", tmp.display())))?;
            }
            drop(f);
            std::fs::rename(&tmp, path).map_err(|e| {
                Error::Gravity(format!(
                    "{}: rename from {}: {e}",
                    path.display(),
                    tmp.display()
                ))
            })
        })();
        if let Err(e) = write_res {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    }
    pub(super) fn verify(path: &Path) -> Result<serde_json::Value> {
        let shard = GravityShard::open(path)?;
        let claimed_body = shard
            .extra
            .pointer("/integrity/body_sha256")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let file_len = shard.mmap.len() as u64;
        if shard.body_offset > file_len {
            return Err(Error::Gravity(format!(
                "{}: body_off {} > {file_len}",
                path.display(),
                shard.body_offset
            )));
        }
        let physical_body = &shard.mmap[shard.body_offset as usize..];
        let physical_len = physical_body.len() as u64;
        let mut ordered: Vec<&TensorDescriptor> = shard.tensors.values().collect();
        ordered.sort_by_key(|t| (t.offset, t.name.as_str()));
        let mut cursor = 0u64;
        for d in &ordered {
            if d.offset != cursor {
                let kind = if d.offset < cursor { "overlap" } else { "gap" };
                return Err(Error::Gravity(format!(
                    "{}: {kind} tensor {:?} (offset {} cursor {cursor})",
                    path.display(),
                    d.name,
                    d.offset
                )));
            }
            cursor = checked_end!(d.offset, d.bytes, physical_len, &d.name);
        }
        if cursor != physical_len {
            return Err(Error::Gravity(format!(
                "{}: trailing or undeclared body bytes: declared {cursor}, physical {physical_len}",
                path.display()
            )));
        }
        let body_ok = hex_encode(Sha256::digest(physical_body).as_slice()) == claimed_body;
        let bad: Vec<String> = ordered
            .iter()
            .filter(|d| {
                let span = &physical_body[d.offset as usize..(d.offset + d.bytes) as usize];
                hex_encode(Sha256::digest(span).as_slice()) != d.sha256
            })
            .map(|d| d.name.clone())
            .collect();
        let rate = |entries: &[&TensorDescriptor]| {
            let elements: u64 = entries.iter().map(|t| t.elements).sum();
            if elements == 0 {
                0.0
            } else {
                entries.iter().map(|t| t.bytes).sum::<u64>() as f64 * 8.0 / elements as f64
            }
        };
        let carried: Vec<_> = ordered.iter().copied().filter(|t| t.elements > 0).collect();
        let compressed: Vec<_> = carried
            .iter()
            .copied()
            .filter(|t| !t.codec.starts_with("native."))
            .collect();
        let (observed_packed, observed_complete) = (rate(&compressed), rate(&carried));
        let compression = shard
            .extra
            .get("compression")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let claim = |k: &str, obs: f64| compression.get(k).and_then(|v| v.as_f64()).unwrap_or(obs);
        let (claimed_packed, claimed_complete) = (
            claim("packed_bpw", observed_packed),
            claim("complete_bpw", observed_complete),
        );
        let packed_ok = (observed_packed - claimed_packed).abs() < 1e-6;
        let complete_ok = (observed_complete - claimed_complete).abs() < 1e-6;
        let empty: Vec<String> = ordered
            .iter()
            .filter(|t| t.bytes == 0)
            .map(|t| t.name.clone())
            .collect();
        let ok = body_ok && bad.is_empty() && packed_ok && complete_ok && empty.is_empty();
        Ok(serde_json::json!({
            "path": path.display().to_string(),
            "format_version": shard.extra.get("format_version").and_then(|v| v.as_u64()).unwrap_or(1),
            "tensors": shard.tensors.len(),
            "body_bytes": physical_len,
            "body_sha256_ok": body_ok,
            "bad_tensors": bad,
            "tensors_without_payload": empty,
            "observed_packed_bpw": observed_packed,
            "claimed_packed_bpw": claimed_packed,
            "observed_complete_bpw": observed_complete,
            "claimed_complete_bpw": claimed_complete,
            "rate_self_consistent": packed_ok && complete_ok,
            "packed_rate_self_consistent": packed_ok,
            "complete_rate_self_consistent": complete_ok,
            "ok": ok,
        }))
    }
    pub(super) fn read_header(path: &Path) -> Result<serde_json::Value> {
        let mmap = unsafe { Mmap::map(&File::open(path)?)? };
        Ok(parse_prefix_header(&mmap)?.1)
    }
}
mod support {
    use crate::{Error, Result};
    use half::f16;
    pub fn widen_native(codec: &str, blob: &[u8]) -> Result<Vec<f32>> {
        let dtype = codec.split_once('.').map(|(_, d)| d).unwrap_or("");
        let unit = match dtype {
            "bf16" | "f16" => 2usize,
            "f32" => 4usize,
            other => {
                return Err(Error::Gravity(format!(
                    "bad native dtype {other:?} (codec {codec:?})"
                )))
            }
        };
        if blob.len() % unit != 0 {
            return Err(Error::Gravity(format!(
                "native.{dtype} len {} % {unit}",
                blob.len()
            )));
        }
        Ok(match dtype {
            "bf16" => blob
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "f16" => blob
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => blob
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        })
    }
}
mod ffi {
    use super::container::{commit_shard, plan_shard, read_header, verify, GravityShard};
    use super::pq::pack_indices;
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_uint};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::Path;
    use std::ptr;
    use std::slice;
    /// Owned sink: `Box<[u8]>` thin ptr+len; free with `hawking_artifact_free`.
    struct Out {
        owned: bool,
        buf: *mut u8,
        len: *mut usize,
        ptr: *mut *mut u8,
    }
    impl Out {
        /// Validate both slots and clear them before any fallible work.
        unsafe fn owned_ready(ptr: *mut *mut u8, len: *mut usize) -> Result<Self, c_int> {
            if ptr.is_null() || len.is_null() {
                return Err(1);
            }
            unsafe {
                *ptr = ptr::null_mut();
                *len = 0;
            }
            Ok(Self {
                owned: true,
                buf: ptr::null_mut(),
                len,
                ptr,
            })
        }
    }
    unsafe fn deliver(out: Out, bytes: Vec<u8>) -> c_int {
        if out.owned {
            if bytes.len() > (isize::MAX as usize) {
                return 6;
            }
            let n = bytes.len();
            let raw = Box::into_raw(bytes.into_boxed_slice());
            unsafe {
                *out.ptr = raw as *mut u8;
                *out.len = n;
            }
            return 0;
        }
        if out.len.is_null() {
            return 1;
        }
        let need = bytes.len();
        if out.buf.is_null() || unsafe { *out.len } == 0 || need > unsafe { *out.len } {
            unsafe { *out.len = need };
            return 5;
        }
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), out.buf, need);
            *out.len = need;
        }
        0
    }
    unsafe fn deliver_json(out: Out, v: &serde_json::Value) -> c_int {
        match serde_json::to_vec(v) {
            Ok(b) => deliver(out, b),
            Err(_) => 4,
        }
    }
    macro_rules! catch_code {
        ($b:expr) => {
            match catch_unwind(AssertUnwindSafe($b)) {
                Ok(c) => c,
                Err(_) => 8,
            }
        };
    }
    /// C path borrow lives only for the duration of `f` (no manufactured lifetime).
    unsafe fn with_c_path<T>(path: *const c_char, f: impl FnOnce(&Path) -> T) -> Result<T, c_int> {
        if path.is_null() {
            return Err(1);
        }
        let cstr = unsafe { CStr::from_ptr(path) };
        let s = cstr.to_str().map_err(|_| 2)?;
        Ok(f(Path::new(s)))
    }
    unsafe fn do_pack(indices: *const u32, n: usize, bits: c_uint, out: Out) -> c_int {
        if indices.is_null() || n > (isize::MAX as usize) {
            return if indices.is_null() { 1 } else { 6 };
        }
        match pack_indices(unsafe { slice::from_raw_parts(indices, n) }, bits) {
            Ok(p) => deliver(out, p),
            Err(_) => 3,
        }
    }
    unsafe fn do_path_json(
        path: *const c_char,
        out: Out,
        f: impl FnOnce(&Path) -> Result<serde_json::Value, c_int>,
    ) -> c_int {
        match unsafe { with_c_path(path, f) } {
            Ok(Ok(v)) => deliver_json(out, &v),
            Ok(Err(c)) => c,
            Err(c) => c,
        }
    }
    unsafe fn do_write(
        path: *const c_char,
        meta_json: *const u8,
        meta_len: usize,
        body: *const u8,
        body_len: usize,
        out: Out,
    ) -> c_int {
        if meta_json.is_null() {
            return 1;
        }
        if meta_len > (isize::MAX as usize) || body_len > (isize::MAX as usize) {
            return 6;
        }
        let path_buf = match unsafe { with_c_path(path, |p| p.to_path_buf()) } {
            Ok(p) => p,
            Err(c) => return c,
        };
        let path = path_buf.as_path();
        let req: serde_json::Value =
            match serde_json::from_slice(unsafe { slice::from_raw_parts(meta_json, meta_len) }) {
                Ok(v) => v,
                Err(_) => return 3,
            };
        let (Some(lengths), Some(tensors)) = (
            req.get("payload_lengths").and_then(|v| v.as_array()),
            req.get("tensors").and_then(|v| v.as_array()),
        ) else {
            return 3;
        };
        if lengths.len() != tensors.len() {
            return 6;
        }
        let body = if body.is_null() {
            if body_len != 0 {
                return 1;
            }
            &[][..]
        } else {
            unsafe { slice::from_raw_parts(body, body_len) }
        };
        let mut offset = 0usize;
        let mut payloads = Vec::with_capacity(tensors.len());
        for (desc, len_v) in tensors.iter().zip(lengths.iter()) {
            let Some(len) = len_v.as_u64().and_then(|n| usize::try_from(n).ok()) else {
                return 3;
            };
            let Some(end) = offset.checked_add(len).filter(|&e| e <= body.len()) else {
                return 6;
            };
            payloads.push((desc.clone(), &body[offset..end]));
            offset = end;
        }
        if offset != body.len() {
            return 6;
        }
        let (header, encoded) = match plan_shard(&payloads, &req) {
            Ok(v) => v,
            Err(_) => return 3,
        };
        if !out.owned {
            let need = match serde_json::to_vec(&header) {
                Ok(b) => b.len(),
                Err(_) => return 4,
            };
            let probe = !out.len.is_null()
                && (out.buf.is_null() || unsafe { *out.len } == 0 || unsafe { *out.len } < need);
            if probe {
                unsafe { *out.len = need };
                return 5;
            }
        }
        if commit_shard(path, &payloads, &encoded).is_err() {
            return 3;
        }
        if !out.owned && (out.buf.is_null() || out.len.is_null()) {
            return 0;
        }
        deliver_json(out, &header)
    }
    unsafe fn do_tensor(
        path: *const c_char,
        name: *const c_char,
        verify_hash: c_int,
        out: Out,
    ) -> c_int {
        if name.is_null() {
            return 1;
        }
        let path_buf = match unsafe { with_c_path(path, |p| p.to_path_buf()) } {
            Ok(p) => p,
            Err(c) => return c,
        };
        let path = path_buf.as_path();
        let name = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => return 2,
        };
        let shard = match GravityShard::open(path) {
            Ok(s) => s,
            Err(_) => return 3,
        };
        let d = match shard.descriptor(name) {
            Some(d) => d,
            None => return 7,
        };
        if !out.owned && (out.buf.is_null() || out.len.is_null() || unsafe { *out.len } == 0) {
            if out.len.is_null() {
                return 1;
            }
            unsafe { *out.len = d.bytes as usize };
            return 5;
        }
        match shard.read_tensor(name, verify_hash != 0) {
            Ok(b) => deliver(out, b),
            Err(_) => 3,
        }
    }
    #[no_mangle]
    pub extern "C" fn hawking_artifact_abi_version() -> c_uint {
        1
    }
    #[no_mangle]
    pub unsafe extern "C" fn hawking_artifact_free(ptr: *mut u8, len: usize) {
        if ptr.is_null() || len > (isize::MAX as usize) {
            return;
        }
        drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len)) });
    }
    /// Dual ABI: `$buf` is copy-buffer (compat); `$owned` prevalidates then clears slots.
    #[rustfmt::skip]
    macro_rules! dual {
        (path_json $buf:ident, $owned:ident, $op:expr) => {
            #[no_mangle]
            pub unsafe extern "C" fn $buf(path: *const c_char, out: *mut u8, out_len: *mut usize) -> c_int {
                catch_code!(|| do_path_json(path, Out { owned: false, buf: out, len: out_len, ptr: ptr::null_mut() }, $op))
            }
            #[no_mangle]
            pub unsafe extern "C" fn $owned(path: *const c_char, out_ptr: *mut *mut u8, out_len: *mut usize) -> c_int {
                catch_code!(|| match Out::owned_ready(out_ptr, out_len) {
                    Ok(out) => do_path_json(path, out, $op),
                    Err(c) => c,
                })
            }
        };
        (pack $buf:ident, $owned:ident) => {
            #[no_mangle]
            pub unsafe extern "C" fn $buf(indices: *const u32, n_indices: usize, bits: c_uint, out: *mut u8, out_len: *mut usize) -> c_int {
                catch_code!(|| do_pack(indices, n_indices, bits, Out { owned: false, buf: out, len: out_len, ptr: ptr::null_mut() }))
            }
            #[no_mangle]
            pub unsafe extern "C" fn $owned(indices: *const u32, n_indices: usize, bits: c_uint, out_ptr: *mut *mut u8, out_len: *mut usize) -> c_int {
                catch_code!(|| match Out::owned_ready(out_ptr, out_len) {
                    Ok(out) => do_pack(indices, n_indices, bits, out),
                    Err(c) => c,
                })
            }
        };
        (write $buf:ident, $owned:ident) => {
            #[no_mangle]
            pub unsafe extern "C" fn $buf(
                path: *const c_char, meta_json: *const u8, meta_len: usize,
                body: *const u8, body_len: usize, header_out: *mut u8, header_len: *mut usize,
            ) -> c_int {
                catch_code!(|| do_write(path, meta_json, meta_len, body, body_len, Out { owned: false, buf: header_out, len: header_len, ptr: ptr::null_mut() }))
            }
            #[no_mangle]
            pub unsafe extern "C" fn $owned(
                path: *const c_char, meta_json: *const u8, meta_len: usize,
                body: *const u8, body_len: usize, out_ptr: *mut *mut u8, out_len: *mut usize,
            ) -> c_int {
                catch_code!(|| match Out::owned_ready(out_ptr, out_len) {
                    Ok(out) => do_write(path, meta_json, meta_len, body, body_len, out),
                    Err(c) => c,
                })
            }
        };
        (tensor $buf:ident, $owned:ident) => {
            #[no_mangle]
            pub unsafe extern "C" fn $buf(path: *const c_char, name: *const c_char, verify_hash: c_int, out: *mut u8, out_len: *mut usize) -> c_int {
                catch_code!(|| do_tensor(path, name, verify_hash, Out { owned: false, buf: out, len: out_len, ptr: ptr::null_mut() }))
            }
            #[no_mangle]
            pub unsafe extern "C" fn $owned(path: *const c_char, name: *const c_char, verify_hash: c_int, out_ptr: *mut *mut u8, out_len: *mut usize) -> c_int {
                catch_code!(|| match Out::owned_ready(out_ptr, out_len) {
                    Ok(out) => do_tensor(path, name, verify_hash, out),
                    Err(c) => c,
                })
            }
        };
    }
    dual!(path_json hawking_verify, hawking_verify_owned, |p| verify(p).map_err(|_| 3));
    dual!(path_json hawking_read_header, hawking_read_header_owned, |p| read_header(p).map_err(|_| 3));
    dual!(path_json hawking_open_shard, hawking_open_shard_owned, |p| {
        GravityShard::open(p).map(|s| serde_json::json!({"body_offset": s.body_offset, "header": s.full_header})).map_err(|_| 3)
    });
    dual!(pack hawking_pack_indices, hawking_pack_indices_owned);
    dual!(write hawking_write_shard, hawking_write_shard_owned);
    dual!(tensor hawking_read_tensor, hawking_read_tensor_owned);
}
pub use aap::{
    activation_aware_sections, parse_activation_aware_header, ActivationAwareHeader,
    ActivationAwareSide, ActivationAwareTensor,
};
pub use container::{GravityShard, TensorDescriptor};
pub use pq::{
    parse_pq_header, pq_matvec, pq_matvec_f64_authority, pq_row, pq_sections, PqHeader, PqTensor,
};
pub use support::widen_native;
