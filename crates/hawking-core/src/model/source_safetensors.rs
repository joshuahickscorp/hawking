//! Shared bounded safetensors-header reader for streamed Qwen source indexes.
//!
//! This is intentionally a parsing leaf: it reads only the header, retains the
//! source-declared descriptor fields, and does not impose file-byte closure,
//! dtype admission, or offset-order policy. Those are model-index decisions.

use crate::{Error, Result};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::Read, path::Path};

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
pub(super) struct SafetensorsHeader {
    pub(super) header_nbytes: u64,
    pub(super) tensors: HashMap<String, SafetensorsTensorInfo>,
}

#[derive(Debug)]
pub(super) struct SafetensorsTensorInfo {
    pub(super) dtype: String,
    pub(super) shape: Vec<usize>,
    pub(super) data_offsets: (u64, u64),
}

#[inline]
fn model_err(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

/// Read the source-declared safetensors header without inspecting its payload.
///
/// The parser deliberately preserves the Qwen source-index contract: headers
/// must be non-empty and at most 64 MiB, `__metadata__` is ignored, and each
/// named descriptor must provide dtype, usize-representable shape, and a pair
/// of u64 data offsets. Model-specific callers decide dtype and range policy.
pub(super) fn read_safetensors_header(path: &Path) -> Result<SafetensorsHeader> {
    let mut file =
        File::open(path).map_err(|e| model_err(format!("cannot open {}: {e}", path.display())))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| {
        model_err(format!(
            "cannot read header length of {}: {e}",
            path.display()
        ))
    })?;
    let header_nbytes = u64::from_le_bytes(len_buf);
    if header_nbytes == 0 || header_nbytes > MAX_HEADER_BYTES {
        return Err(model_err(format!(
            "implausible safetensors header length {header_nbytes} in {}",
            path.display()
        )));
    }
    let mut raw = vec![0u8; header_nbytes as usize];
    file.read_exact(&mut raw)
        .map_err(|e| model_err(format!("cannot read header of {}: {e}", path.display())))?;
    let value: Value = serde_json::from_slice(&raw).map_err(|e| {
        model_err(format!(
            "safetensors header JSON invalid in {}: {e}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        model_err(format!(
            "safetensors header is not an object in {}",
            path.display()
        ))
    })?;
    let mut tensors = HashMap::new();
    for (name, info_v) in object {
        if name == "__metadata__" {
            continue;
        }
        let info = info_v
            .as_object()
            .ok_or_else(|| model_err(format!("tensor {name} header is not an object")))?;
        let dtype = info
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| model_err(format!("tensor {name} lacks dtype")))?
            .to_string();
        let shape = info
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| model_err(format!("tensor {name} lacks shape")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| model_err(format!("tensor {name} has non-integer shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| model_err(format!("tensor {name} lacks data_offsets")))?;
        if offsets.len() != 2 {
            return Err(model_err(format!(
                "tensor {name} data_offsets is not a pair"
            )));
        }
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| model_err(format!("tensor {name} data_offsets[0] invalid")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| model_err(format!("tensor {name} data_offsets[1] invalid")))?;
        tensors.insert(
            name.clone(),
            SafetensorsTensorInfo {
                dtype,
                shape,
                data_offsets: (begin, end),
            },
        );
    }
    Ok(SafetensorsHeader {
        header_nbytes,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_safetensors(path: &Path, header: Value, payload: &[u8]) -> usize {
        let header = serde_json::to_vec(&header).expect("serialize header");
        let mut file = File::create(path).expect("create shard");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("write header length");
        file.write_all(&header).expect("write header");
        file.write_all(payload).expect("write payload");
        header.len()
    }

    fn write_source_shard(root: &Path, header: Value, payload: &[u8], weight_map: Value) {
        let shard_name = "model-00001-of-00001.safetensors";
        write_safetensors(&root.join(shard_name), header, payload);
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": weight_map})).expect("serialize source index"),
        )
        .expect("write source index");
    }

    #[test]
    fn reads_declared_descriptor_fields_and_skips_metadata() {
        let directory = tempdir().expect("tempdir");
        let path = directory.path().join("source.safetensors");
        let header_nbytes = write_safetensors(
            &path,
            json!({
                "__metadata__": {"format": "pt"},
                "weight": {
                    "dtype": "BF16",
                    "shape": [2, 3],
                    "data_offsets": [4, 16]
                },
                "control": {
                    "dtype": "I64",
                    "shape": [1],
                    "data_offsets": [0, 4]
                }
            }),
            &[0; 16],
        );

        let header = read_safetensors_header(&path).expect("read header");
        assert_eq!(header.header_nbytes, header_nbytes as u64);
        assert_eq!(header.tensors.len(), 2);
        assert!(!header.tensors.contains_key("__metadata__"));
        let weight = header.tensors.get("weight").expect("weight descriptor");
        assert_eq!(weight.dtype, "BF16");
        assert_eq!(weight.shape, vec![2, 3]);
        assert_eq!(weight.data_offsets, (4, 16));
        let control = header.tensors.get("control").expect("control descriptor");
        assert_eq!(control.dtype, "I64");
    }

    #[test]
    fn rejects_empty_oversized_and_truncated_headers() {
        let directory = tempdir().expect("tempdir");
        let path = directory.path().join("bad.safetensors");

        std::fs::write(&path, b"short").expect("write short prefix");
        let error = read_safetensors_header(&path).expect_err("short prefix must fail");
        assert!(matches!(error, Error::Model(_)));
        assert!(error.to_string().contains("cannot read header length of"));

        std::fs::write(&path, 0u64.to_le_bytes()).expect("write empty length");
        let error = read_safetensors_header(&path).expect_err("empty header must fail");
        assert!(error
            .to_string()
            .contains("implausible safetensors header length 0"));

        std::fs::write(&path, (MAX_HEADER_BYTES + 1).to_le_bytes()).expect("write large length");
        let error = read_safetensors_header(&path).expect_err("large header must fail");
        assert!(error
            .to_string()
            .contains("implausible safetensors header length 67108865"));

        let mut truncated = Vec::new();
        truncated.extend_from_slice(&4u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        std::fs::write(&path, truncated).expect("write truncated header");
        let error = read_safetensors_header(&path).expect_err("truncated header must fail");
        assert!(error.to_string().contains("cannot read header of"));
    }

    #[test]
    fn rejects_invalid_json_and_nonobject_header_roots() {
        let directory = tempdir().expect("tempdir");
        let path = directory.path().join("bad.safetensors");

        let mut invalid_json = Vec::new();
        invalid_json.extend_from_slice(&4u64.to_le_bytes());
        invalid_json.extend_from_slice(b"nope");
        std::fs::write(&path, invalid_json).expect("write invalid json");
        let error = read_safetensors_header(&path).expect_err("invalid JSON must fail");
        assert!(matches!(error, Error::Model(_)));
        assert!(error
            .to_string()
            .contains("safetensors header JSON invalid in"));

        write_safetensors(&path, json!(["not", "an", "object"]), &[]);
        let error = read_safetensors_header(&path).expect_err("nonobject root must fail");
        assert!(error
            .to_string()
            .contains("safetensors header is not an object in"));
    }

    #[test]
    fn rejects_malformed_descriptor_fields_with_model_errors() {
        let directory = tempdir().expect("tempdir");
        let path = directory.path().join("bad.safetensors");
        let cases = [
            (json!({"bad": []}), "tensor bad header is not an object"),
            (
                json!({"bad": {"shape": [1], "data_offsets": [0, 2]}}),
                "tensor bad lacks dtype",
            ),
            (
                json!({"bad": {"dtype": "BF16", "data_offsets": [0, 2]}}),
                "tensor bad lacks shape",
            ),
            (
                json!({"bad": {"dtype": "BF16", "shape": [1.5], "data_offsets": [0, 2]}}),
                "tensor bad has non-integer shape",
            ),
            (
                json!({"bad": {"dtype": "BF16", "shape": [1]}}),
                "tensor bad lacks data_offsets",
            ),
            (
                json!({"bad": {"dtype": "BF16", "shape": [1], "data_offsets": [0]}}),
                "tensor bad data_offsets is not a pair",
            ),
            (
                json!({"bad": {"dtype": "BF16", "shape": [1], "data_offsets": ["start", 2]}}),
                "tensor bad data_offsets[0] invalid",
            ),
        ];

        for (header, expected) in cases {
            write_safetensors(&path, header, &[]);
            let error = read_safetensors_header(&path).expect_err("bad descriptor must fail");
            assert!(matches!(error, Error::Model(_)));
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn qwen30_and_qwen80_indexes_read_the_same_bf16_source_payload() {
        let directory = tempdir().expect("tempdir");
        let payload = [0x80, 0x3f];
        write_source_shard(
            directory.path(),
            json!({
                "__metadata__": {"format": "pt"},
                "bf16": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
            }),
            &payload,
            json!({"bf16": "model-00001-of-00001.safetensors"}),
        );

        let qwen30 =
            crate::model::qwen30_source_bf16_layer_major::SourceBf16Index::open(directory.path())
                .expect("open Qwen30 source index");
        assert_eq!(qwen30.tensor_count(), 1);
        assert_eq!(
            qwen30.read_raw("bf16").expect("read Qwen30 payload"),
            payload
        );

        let qwen80 =
            crate::model::qwen80_source_bf16_layer_major::SourceBf16Index::open(directory.path())
                .expect("open Qwen80 source index");
        assert_eq!(qwen80.tensor_count(), 1);
        assert_eq!(
            qwen80.read_raw("bf16").expect("read Qwen80 payload"),
            payload
        );
        assert_eq!(qwen80.bytes_read_total(), payload.len() as u64);
    }

    #[test]
    fn qwen_indexes_preserve_their_mixed_control_policy() {
        let directory = tempdir().expect("tempdir");
        write_source_shard(
            directory.path(),
            json!({
                "bf16": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                "control": {"dtype": "U32", "shape": [1], "data_offsets": [2, 6]}
            }),
            &[0x80, 0x3f, 1, 0, 0, 0],
            json!({
                "bf16": "model-00001-of-00001.safetensors",
                "control": "model-00001-of-00001.safetensors"
            }),
        );

        let qwen30 = match crate::model::qwen30_source_bf16_layer_major::SourceBf16Index::open(
            directory.path(),
        ) {
            Ok(_) => panic!("Qwen30 must reject a non-BF16 mapped tensor"),
            Err(error) => error,
        };
        assert!(matches!(qwen30, Error::Model(_)));
        assert!(qwen30
            .to_string()
            .contains("tensor control dtype U32 is not BF16"));

        let qwen80 =
            crate::model::qwen80_source_bf16_layer_major::SourceBf16Index::open(directory.path())
                .expect("Qwen80 must skip its mixed control tensor");
        assert_eq!(qwen80.tensor_count(), 1);
        let error = qwen80
            .read_raw("control")
            .expect_err("Qwen80 control must stay outside BF16 source view");
        assert!(matches!(error, Error::Model(_)));
        assert!(error
            .to_string()
            .contains("source index lacks tensor control"));
    }
}
