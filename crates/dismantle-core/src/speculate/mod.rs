//! Speculative decoding: n-gram draft, EAGLE5 trained head, and shared-expert draft paths.
//!
//! The shared-expert path uses DeepSeek-V2/V3's 2 always-active experts as a free draft,
//! then runs routed experts as verification. Tokens where they agree are accepted; mismatches roll back.

pub mod eagle5;
pub mod eagle5_forward;
pub mod shared;
pub mod safetensors_io;
pub mod user_ngram;
