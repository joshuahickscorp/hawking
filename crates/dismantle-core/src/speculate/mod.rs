//! Speculative decoding via shared experts — wedge 4.
//!
//! DeepSeek-V2/V3 has 2 shared experts that run on every token. We
//! use shared-expert-only output as a free draft, then run routed
//! experts as verification. Tokens where shared and routed agree are
//! accepted from the draft; mismatches roll back.
//!
//! Lands in Phase 4.5. Gated on ≥0.7 acceptance rate; if empirically
//! lower we ship as `--speculate` opt-in and don't headline it.

pub mod eagle5;
pub mod eagle5_forward;
pub mod shared;
pub mod safetensors_io;
pub mod user_ngram;
