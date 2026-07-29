//! Shared GGUF weight-loader helpers for the dense/MoE model loaders.
//!
//! Consolidates the byte-identical `TensorRef` + dequant helpers that were
//! copy-pasted across qwen_dense / deepseek_v2 / llama. Architecturally
//! distinct forward paths stay in their modules; only identical byte helpers
//! live here.

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

/// Dequant a `TensorRef` from a GGUF mmap into `buf`, resizing in place.
/// Identical across llama / qwen_dense / deepseek_v2; each engine passes its
/// own mmap slice.
pub(crate) fn dequant_ref_into(mmap: &[u8], t: &TensorRef, buf: &mut Vec<f32>) -> Result<()> {
    if buf.len() != t.n_elems {
        buf.resize(t.n_elems, 0.0);
    }
    let bytes = &mmap[t.offset..t.offset + t.byte_size];
    quant::dequant_into(t.dtype, bytes, buf)
}
