
#![cfg_attr(
    not(any(target_os = "macos", feature = "cuda")),
    forbid(unsafe_code)
)]

pub mod c2_final;
pub mod codebook;
pub mod debias;
pub mod decode;
pub mod encode;
pub mod gate_utils;
pub mod safetensor_io;
pub mod encode_cache;

pub mod fano;
pub mod format;
pub mod learned_codebook;
pub mod lut_tables;
pub mod outlier_wire;

#[cfg(any(target_os = "macos", feature = "cuda"))]
pub(crate) mod gpu_types;
#[cfg(target_os = "macos")]
pub(crate) mod metal_backend;

#[cfg(target_os = "macos")]
pub mod metal_encode;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_backend;
pub mod provenance;
pub mod provenance_io;
pub mod rht;
pub mod rslt;
pub mod selfdesc;
pub mod sha256;
pub mod sideinfo_rans;
pub mod sideinfo_wire;
pub mod trellis;

#[cfg(test)]
mod tests;
#[cfg(any(test, kani))]
mod proofs;

pub use codebook::{quantile_lut, QUANTILE_SHIFT};
pub use decode::decode_tensor;
pub use encode::{
    encode_tensor, encode_tensor_opts, encode_tensor_with, EncodeOpts, EncodedTensor,
};
pub use rht::{
    rht_forward, rht_forward_rows, rht_inverse, rht_inverse_rows, RhtConfig,
};
pub use trellis::{CodebookMode, TrellisConfig};
