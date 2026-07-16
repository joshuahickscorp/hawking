//! Speculative decoding: n-gram draft, EAGLE5 trained head, and shared-expert draft paths.
//!
//! The shared-expert path uses DeepSeek-V2/V3's 2 always-active experts as a free draft,
//! then runs routed experts as verification. Tokens where they agree are accepted; mismatches roll back.

#[path = "speculate_cross_tokenizer.rs"]
pub mod cross_tokenizer;
#[path = "speculate_eagle5.rs"]
pub mod eagle5;
#[path = "speculate_eagle5_forward.rs"]
pub mod eagle5_forward;
#[path = "speculate_eagle_proposer.rs"]
pub mod eagle_proposer;
#[path = "speculate_governor.rs"]
pub mod governor;
#[path = "speculate_parallel_draft.rs"]
pub mod parallel_draft;
#[path = "speculate_policy.rs"]
pub mod policy;
#[path = "speculate_proposal.rs"]
pub mod proposal;
#[path = "speculate_replay_oracle.rs"]
pub mod replay_oracle;
#[path = "speculate_retrieval.rs"]
pub mod retrieval;
#[path = "speculate_router.rs"]
pub mod router;
#[path = "speculate_safetensors_io.rs"]
pub mod safetensors_io;
#[path = "speculate_shared.rs"]
pub mod shared;
#[path = "speculate_suffix_array.rs"]
pub mod suffix_array;
#[path = "speculate_suffix_automaton.rs"]
pub mod suffix_automaton;
#[path = "speculate_tree.rs"]
pub mod tree;
#[path = "speculate_user_ngram.rs"]
pub mod user_ngram;
#[path = "speculate_verifier.rs"]
pub mod verifier;
