//! Deterministic, header-only safetensors inventory for Gravity admission.
//!
//! This is deliberately below model execution.  It reads no tensor payloads,
//! sorts files and tensor names, validates byte geometry, and refuses malformed
//! sources.  Release-facing byte ledgers can use this common owner instead of
//! reimplementing a slightly different Python scan for every specimen.

use crate::{Error, Result};
use serde_json::Value;
use std::{
    cmp::Ordering,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// One source tensor, expressed relative to its shard payload (not its file).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsTensor {
    pub shard: PathBuf,
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub payload_offset: u64,
    pub payload_bytes: u64,
}

/// Validated accounting for one safetensors shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsShardInventory {
    pub path: PathBuf,
    /// Entire on-disk file length, including the eight-byte header-length word.
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub payload_bytes: u64,
    pub tensors: Vec<SafetensorsTensor>,
}

/// Deterministic accounting for every `*.safetensors` file directly in a directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsInventory {
    pub root: PathBuf,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub payload_bytes: u64,
    pub tensor_count: u64,
    pub shards: Vec<SafetensorsShardInventory>,
    /// Tensor lookup table in lexical tensor-name order. Duplicate names across
    /// shards are refused during scan rather than resolved by incidental order.
    pub tensors: Vec<SafetensorsTensor>,
}

impl SafetensorsInventory {
    /// Scan direct child shards in lexical filename order.
    pub fn scan_dir(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let mut paths = Vec::new();
        for entry in fs::read_dir(&root).map_err(|error| {
            Error::Gravity(format!(
                "cannot list safetensors directory {}: {error}",
                root.display()
            ))
        })? {
            let entry = entry.map_err(|error| {
                Error::Gravity(format!(
                    "cannot inspect safetensors directory {}: {error}",
                    root.display()
                ))
            })?;
            let path = entry.path();
            if entry
                .file_type()
                .map_err(|error| {
                    Error::Gravity(format!("cannot stat {}: {error}", path.display()))
                })?
                .is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension == "safetensors")
            {
                paths.push(path);
            }
        }
        paths.sort();
        if paths.is_empty() {
            return Err(Error::Gravity(format!(
                "no direct .safetensors shards in {}",
                root.display()
            )));
        }

        let mut shards = Vec::with_capacity(paths.len());
        let mut tensors = Vec::new();
        let mut file_bytes = 0u64;
        let mut header_bytes = 0u64;
        let mut payload_bytes = 0u64;
        let mut tensor_count = 0u64;
        for path in paths {
            let shard = Self::scan_shard(&path)?;
            file_bytes = checked_add(file_bytes, shard.file_bytes, "total file bytes")?;
            header_bytes = checked_add(header_bytes, shard.header_bytes, "total header bytes")?;
            payload_bytes = checked_add(payload_bytes, shard.payload_bytes, "total payload bytes")?;
            tensor_count = checked_add(tensor_count, shard.tensors.len() as u64, "tensor count")?;
            tensors.extend(shard.tensors.iter().cloned());
            shards.push(shard);
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        for pair in tensors.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(Error::Gravity(format!(
                    "duplicate tensor name {} across safetensors shards in {}",
                    pair[0].name,
                    root.display()
                )));
            }
        }
        Ok(Self {
            root,
            file_bytes,
            header_bytes,
            payload_bytes,
            tensor_count,
            shards,
            tensors,
        })
    }

    /// Find a source tensor by its exact name in O(log n) time.
    pub fn tensor(&self, name: &str) -> Option<&SafetensorsTensor> {
        self.tensors
            .binary_search_by(|candidate| candidate.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tensors[index])
    }

    /// Read a bounded range from one tensor payload.
    ///
    /// The caller's range is relative to the tensor, never the shard file. The
    /// method verifies the requested geometry before opening a payload and
    /// never falls back to a dense or alternate source.
    pub fn read_tensor_range(&self, name: &str, offset: u64, bytes: usize) -> Result<Vec<u8>> {
        let tensor = self.tensor(name).ok_or_else(|| {
            Error::Gravity(format!(
                "tensor {name} is absent from {}",
                self.root.display()
            ))
        })?;
        let requested = bytes as u64;
        let range_end = offset
            .checked_add(requested)
            .ok_or_else(|| Error::Gravity(format!("tensor {name} read range overflows u64")))?;
        if range_end > tensor.payload_bytes {
            return Err(Error::Gravity(format!(
                "tensor {name} read [{offset}, {range_end}) exceeds {} payload bytes",
                tensor.payload_bytes
            )));
        }
        let shard = self
            .shards
            .iter()
            .find(|shard| shard.path.file_name() == Some(tensor.shard.as_os_str()))
            .ok_or_else(|| {
                Error::Gravity(format!(
                    "tensor {name} points to missing shard {}",
                    tensor.shard.display()
                ))
            })?;
        let file_offset = shard
            .header_bytes
            .checked_add(tensor.payload_offset)
            .and_then(|value| value.checked_add(offset))
            .ok_or_else(|| Error::Gravity(format!("tensor {name} file offset overflows u64")))?;
        let mut file = File::open(&shard.path).map_err(|error| {
            Error::Gravity(format!("cannot open {}: {error}", shard.path.display()))
        })?;
        file.seek(SeekFrom::Start(file_offset)).map_err(|error| {
            Error::Gravity(format!(
                "cannot seek tensor {name} in {}: {error}",
                shard.path.display()
            ))
        })?;
        let mut data = vec![0u8; bytes];
        file.read_exact(&mut data).map_err(|error| {
            Error::Gravity(format!(
                "cannot read tensor {name} in {}: {error}",
                shard.path.display()
            ))
        })?;
        Ok(data)
    }

    /// Scan one shard and require every payload byte to be described exactly once.
    pub fn scan_shard(path: impl AsRef<Path>) -> Result<SafetensorsShardInventory> {
        let path = path.as_ref().to_path_buf();
        let file_bytes = fs::metadata(&path)
            .map_err(|error| Error::Gravity(format!("cannot stat {}: {error}", path.display())))?
            .len();
        let mut file = File::open(&path)
            .map_err(|error| Error::Gravity(format!("cannot open {}: {error}", path.display())))?;
        let mut length = [0u8; 8];
        file.read_exact(&mut length).map_err(|error| {
            Error::Gravity(format!(
                "cannot read header length from {}: {error}",
                path.display()
            ))
        })?;
        let header_json_bytes = u64::from_le_bytes(length);
        if header_json_bytes == 0 || header_json_bytes > MAX_HEADER_BYTES {
            return Err(Error::Gravity(format!(
                "implausible safetensors header length {header_json_bytes} in {}",
                path.display()
            )));
        }
        let header_bytes = checked_add(8, header_json_bytes, "header length")?;
        if header_bytes > file_bytes {
            return Err(Error::Gravity(format!(
                "header ({header_bytes} bytes) exceeds file ({file_bytes} bytes) in {}",
                path.display()
            )));
        }
        let mut raw = vec![0u8; header_json_bytes as usize];
        file.read_exact(&mut raw).map_err(|error| {
            Error::Gravity(format!("cannot read header in {}: {error}", path.display()))
        })?;
        let object = serde_json::from_slice::<Value>(&raw)
            .map_err(|error| {
                Error::Gravity(format!(
                    "invalid safetensors JSON in {}: {error}",
                    path.display()
                ))
            })?
            .as_object()
            .cloned()
            .ok_or_else(|| {
                Error::Gravity(format!(
                    "safetensors header is not an object in {}",
                    path.display()
                ))
            })?;

        let payload_bytes = file_bytes - header_bytes;
        let shard_name = path.file_name().map(PathBuf::from).ok_or_else(|| {
            Error::Gravity(format!(
                "safetensors path lacks a filename: {}",
                path.display()
            ))
        })?;
        let mut tensors = Vec::new();
        for (name, descriptor) in object {
            if name == "__metadata__" {
                continue;
            }
            tensors.push(parse_tensor(&shard_name, &name, &descriptor)?);
        }
        tensors.sort_by(
            |left, right| match left.payload_offset.cmp(&right.payload_offset) {
                Ordering::Equal => left.name.cmp(&right.name),
                order => order,
            },
        );

        let mut cursor = 0u64;
        for tensor in &tensors {
            if tensor.payload_offset != cursor {
                return Err(Error::Gravity(format!(
                    "payload offsets are not contiguous at tensor {} in {}: expected {cursor}, found {}",
                    tensor.name, path.display(), tensor.payload_offset
                )));
            }
            cursor = checked_add(cursor, tensor.payload_bytes, "tensor payload range")?;
            if cursor > payload_bytes {
                return Err(Error::Gravity(format!(
                    "tensor {} exceeds payload in {}: end {cursor}, payload {payload_bytes}",
                    tensor.name,
                    path.display()
                )));
            }
        }
        if cursor != payload_bytes {
            return Err(Error::Gravity(format!(
                "tensor ranges do not close payload in {}: described {cursor}, payload {payload_bytes}",
                path.display()
            )));
        }
        Ok(SafetensorsShardInventory {
            path,
            file_bytes,
            header_bytes,
            payload_bytes,
            tensors,
        })
    }
}

fn parse_tensor(shard: &Path, name: &str, descriptor: &Value) -> Result<SafetensorsTensor> {
    let descriptor = descriptor.as_object().ok_or_else(|| {
        Error::Gravity(format!(
            "tensor {name} in {} is not an object",
            shard.display()
        ))
    })?;
    let dtype = descriptor
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Gravity(format!("tensor {name} in {} lacks dtype", shard.display())))?
        .to_owned();
    let shape = descriptor
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Gravity(format!("tensor {name} in {} lacks shape", shard.display())))?
        .iter()
        .map(|dimension| {
            dimension.as_u64().ok_or_else(|| {
                Error::Gravity(format!(
                    "tensor {name} in {} has a non-integer shape",
                    shard.display()
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let offsets = descriptor
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Error::Gravity(format!(
                "tensor {name} in {} lacks data_offsets",
                shard.display()
            ))
        })?;
    if offsets.len() != 2 {
        return Err(Error::Gravity(format!(
            "tensor {name} in {} has non-pair data_offsets",
            shard.display()
        )));
    }
    let offset = offsets[0].as_u64().ok_or_else(|| {
        Error::Gravity(format!(
            "tensor {name} in {} has invalid data_offsets[0]",
            shard.display()
        ))
    })?;
    let end = offsets[1].as_u64().ok_or_else(|| {
        Error::Gravity(format!(
            "tensor {name} in {} has invalid data_offsets[1]",
            shard.display()
        ))
    })?;
    if end < offset {
        return Err(Error::Gravity(format!(
            "tensor {name} in {} has descending data_offsets",
            shard.display()
        )));
    }
    let elements = shape.iter().try_fold(1u64, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| {
            Error::Gravity(format!(
                "tensor {name} in {} shape overflows",
                shard.display()
            ))
        })
    })?;
    let expected_bytes = bytes_for_elements(elements, &dtype).ok_or_else(|| {
        Error::Gravity(format!(
            "tensor {name} in {} has unsupported dtype {dtype}",
            shard.display()
        ))
    })?;
    let actual_bytes = end - offset;
    if actual_bytes != expected_bytes {
        return Err(Error::Gravity(format!(
            "tensor {name} in {} byte geometry mismatches dtype/shape: expected {expected_bytes}, found {actual_bytes}",
            shard.display()
        )));
    }
    Ok(SafetensorsTensor {
        shard: shard.to_path_buf(),
        name: name.to_owned(),
        dtype,
        shape,
        payload_offset: offset,
        payload_bytes: actual_bytes,
    })
}

fn bytes_for_elements(elements: u64, dtype: &str) -> Option<u64> {
    let bits = match dtype {
        "BOOL" | "U8" | "I8" => 8,
        "U16" | "I16" | "F16" | "BF16" => 16,
        "U32" | "I32" | "F32" => 32,
        "U64" | "I64" | "F64" => 64,
        "F8_E4M3FN" | "F8_E5M2" | "F8_E4M3FNUZ" | "F8_E5M2FNUZ" => 8,
        "F6_E2M3FN" | "F6_E3M2FN" => 6,
        "F4_E2M1FN" => 4,
        _ => return None,
    };
    let total_bits = elements.checked_mul(bits)?;
    total_bits.checked_add(7)?.checked_div(8)
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| Error::Gravity(format!("{label} overflows u64")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_shard(path: &Path, header: Value, payload_bytes: usize) {
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(
            &(0..payload_bytes)
                .map(|value| value as u8)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }

    #[test]
    fn scans_in_lexical_order_and_closes_bytes() {
        let directory = tempdir().unwrap();
        write_shard(
            &directory.path().join("z.safetensors"),
            serde_json::json!({
                "z": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
            }),
            4,
        );
        write_shard(
            &directory.path().join("a.safetensors"),
            serde_json::json!({
                "b": {"dtype": "BF16", "shape": [2], "data_offsets": [4, 8]},
                "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}
            }),
            8,
        );
        let inventory = SafetensorsInventory::scan_dir(directory.path()).unwrap();
        assert_eq!(inventory.shards.len(), 2);
        assert_eq!(
            inventory.shards[0].path.file_name().unwrap(),
            "a.safetensors"
        );
        assert_eq!(inventory.tensor_count, 3);
        assert_eq!(inventory.tensors.len(), 3);
        assert_eq!(inventory.payload_bytes, 12);
        assert_eq!(
            inventory.file_bytes,
            inventory.header_bytes + inventory.payload_bytes
        );
        assert_eq!(inventory.shards[0].tensors[0].name, "a");
        assert_eq!(inventory.tensor("z").unwrap().dtype, "F32");
        assert_eq!(inventory.read_tensor_range("a", 1, 2).unwrap(), vec![1, 2]);
        assert!(inventory.read_tensor_range("a", 4, 1).is_err());
    }

    #[test]
    fn rejects_noncontiguous_or_mismatched_tensor_geometry() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("bad.safetensors");
        write_shard(
            &path,
            serde_json::json!({
                "bad": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}
            }),
            5,
        );
        let error = SafetensorsInventory::scan_shard(&path).unwrap_err();
        assert!(error.to_string().contains("not contiguous"));

        write_shard(
            &path,
            serde_json::json!({
                "bad": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}
            }),
            4,
        );
        let error = SafetensorsInventory::scan_shard(&path).unwrap_err();
        assert!(error.to_string().contains("byte geometry mismatches"));
    }

    #[test]
    fn rejects_empty_directory_fail_closed() {
        let directory = tempdir().unwrap();
        let error = SafetensorsInventory::scan_dir(directory.path()).unwrap_err();
        assert!(error.to_string().contains("no direct .safetensors"));
    }
}
