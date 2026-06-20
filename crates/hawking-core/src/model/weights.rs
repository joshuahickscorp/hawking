//! Shared GGUF weight-loader helpers for the dense/MoE model loaders.
//!
//! Consolidates the byte-identical `TensorRef` + dequant helpers that were
//! copy-pasted across qwen_dense / deepseek_v2 / phi3 / llama / gemma2. The
//! mixtral loader keeps its own chunked `TensorRef` (it carries extra
//! rows/cols/chunk fields) but still shares the byte-only `dequant_f32` /
//! `dequant_f16` here. The per-model `dequant_ref_into` stays local because it
//! reads through each loader's own mmap handle.

use crate::gguf::{GgmlType, GgufFile};
use crate::{quant, Error, Result};
use half::f16;

/// Pointer into the mmap'd GGUF for one tensor. Cheap to clone; the dequant
/// happens on demand into a caller-owned buffer.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub offset: usize,
    pub byte_size: usize,
    pub dtype: GgmlType,
    pub n_elems: usize,
}

/// Build a `TensorRef` for a named tensor (errors if the tensor is absent).
pub(crate) fn tensor_ref(g: &GgufFile, name: &str) -> Result<TensorRef> {
    let info = g
        .tensor(name)
        .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
    let n_elems: usize = info.dims.iter().product::<u64>() as usize;
    Ok(TensorRef {
        offset: info.data_offset as usize,
        byte_size: info.byte_size as usize,
        dtype: info.dtype,
        n_elems,
    })
}

/// Dequantize a named tensor to `f32` (errors if absent).
pub(crate) fn dequant_f32(g: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let info = g
        .tensor(name)
        .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
    let bytes = g.tensor_bytes(name).unwrap();
    quant::dequant_to_f32(info, bytes)
}

/// Dequantize a named tensor to `f32`, returning `None` if it is absent.
pub(crate) fn dequant_f32_opt(g: &GgufFile, name: &str) -> Result<Option<Vec<f32>>> {
    if g.tensor(name).is_some() {
        Ok(Some(dequant_f32(g, name)?))
    } else {
        Ok(None)
    }
}

/// Dequantize a named tensor to `f16` (errors if absent).
pub(crate) fn dequant_f16(g: &GgufFile, name: &str) -> Result<Vec<f16>> {
    let info = g
        .tensor(name)
        .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
    let bytes = g.tensor_bytes(name).unwrap();
    quant::dequant_to_f16(info, bytes)
}
