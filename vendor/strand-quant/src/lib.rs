#![cfg_attr(not(target_os = "macos"), forbid(unsafe_code))]

pub mod c2_final;
pub mod codebook;
pub mod debias;
pub mod decode;
pub mod encode;
pub mod encode_cache;
pub mod gate_utils;
pub mod safetensor_io;

pub mod fano;
pub mod format;
pub mod learned_codebook;
pub mod lut_tables;
pub mod outlier_wire;

#[cfg(any(feature = "native-execution", feature = "metal-rht-probe"))]
pub mod native_io;

#[cfg(feature = "ordered-pipeline")]
pub mod ordered_pipeline;

#[cfg(target_os = "macos")]
pub(crate) mod gpu_types;
#[cfg(target_os = "macos")]
pub(crate) mod metal_backend;

#[cfg(all(target_os = "macos", feature = "metal-rht-probe"))]
pub mod metal_rht_probe;

#[cfg(target_os = "macos")]
pub mod metal_encode;
pub mod provenance;
pub mod provenance_io;
pub mod rht;
pub mod rslt;
pub mod selfdesc;
pub mod sha256;
pub mod sideinfo_rans;
pub mod sideinfo_wire;
pub mod trellis;

#[cfg(any(test, kani))]
mod proofs;
#[cfg(test)]
mod tests;

pub use codebook::{quantile_lut, QUANTILE_SHIFT};
pub use decode::decode_tensor;
pub use encode::{encode_tensor, encode_tensor_opts, encode_tensor_with, EncodeOpts, EncodedTensor};
#[cfg(feature = "block-parallel")]
pub use encode::{encode_tensor_with_block_parallel, encode_tensor_with_lut_block_parallel, BlockParallelConfig, BlockParallelError};
pub use rht::{rht_forward, rht_forward_rows, rht_inverse, rht_inverse_rows, RhtConfig};
pub use trellis::{CodebookMode, TrellisConfig};
