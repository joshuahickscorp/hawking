//! Hawking speculative decoding pack (extracted from hawking-core / NUCLEAR PASTA).
//! Leaf crate: own Error, inlined argmax; hawking-core depends on it (no cycle).

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("model: {0}")]
    Model(String),
    #[error("not yet implemented: {0}")]
    Unimplemented(&'static str),
}
pub type Result<T> = std::result::Result<T, Error>;

/// argmax over f32 (inlined from hawking-core kernels).
pub fn argmax_f32(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v { best = i; best_v = v; }
    }
    best as u32
}

// Speculative decoding: n-gram draft, EAGLE5 trained head, and shared-expert draft paths.
// The shared-expert path uses DeepSeek-V2/V3's 2 always-active experts as a free draft,
// then runs routed experts as verification. Agreed tokens accepted; mismatches roll back.

pub mod cross_tokenizer;
pub mod eagle5;
pub mod eagle5_forward;
pub mod eagle_proposer;
pub mod governor;
pub mod parallel_draft;
pub mod policy;
pub mod proposal;
pub mod replay_oracle;
pub mod retrieval;
pub mod router;
pub mod safetensors_io;
pub mod shared;
pub mod suffix_array;
pub mod suffix_automaton;
pub mod tree;
pub mod user_ngram;
pub mod verifier;
