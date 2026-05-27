//! End-to-end loader test against the real Eagle6 safetensors heads
//! produced by `colab/finish_q3b_reconciliation.ipynb`.
//!
//! These tests are `#[ignore]`d by default because they need the heads
//! on disk at a specific path. To run them manually after downloading
//! the heads from Drive:
//!
//!   DISMANTLE_Q3B_HEAD=/path/to/q3b_eagle6_long.safetensors \
//!   DISMANTLE_Q1P5_HEAD=/path/to/q1p5_eagle6_long.safetensors \
//!   cargo test --release --test eagle5_trained_head_load -- --ignored
//!
//! Or to point at the user's local Downloads:
//!
//!   DISMANTLE_Q3B_HEAD=$HOME/Downloads/head_final.safetensors \
//!     cargo test --release --test eagle5_trained_head_load \
//!     trained_head_q3b_loads -- --ignored
//!
//! Validates: file opens, metadata parses, all expected tensors are
//! present at the expected shapes. Does NOT exercise the (still
//! placeholder) forward pass.

use dismantle_core::speculate::eagle5::Eagle5Head;
use std::path::PathBuf;

const Q3B_HIDDEN: usize = 2048;
const Q1P5_HIDDEN: usize = 1536;
const QWEN_VOCAB: usize = 151_936;

fn head_path(env_var: &str) -> Option<PathBuf> {
    std::env::var_os(env_var).map(PathBuf::from).filter(|p| p.exists())
}

#[test]
#[ignore = "needs DISMANTLE_Q3B_HEAD=/path/to/q3b_eagle6_long.safetensors"]
fn trained_head_q3b_loads() {
    let path = head_path("DISMANTLE_Q3B_HEAD")
        .expect("set DISMANTLE_Q3B_HEAD to the q3b safetensors path");
    let head = Eagle5Head::load_from_safetensors(&path, Q3B_HIDDEN, QWEN_VOCAB)
        .expect("q3b head must load");
    assert_eq!(head.hidden(), Q3B_HIDDEN);
    assert_eq!(head.vocab(), QWEN_VOCAB);
    // Sanity: propose() with the placeholder forward pass returns K
    // ids in-vocab without panicking. Quality of these drafts is near
    // zero until the real Eagle6 forward lands; we're only proving
    // the loader → propose dispatch is wired.
    let mut h = head;
    let drafts = h.propose(0, 4);
    assert_eq!(drafts.len(), 4);
    for d in &drafts {
        assert!((*d as usize) < QWEN_VOCAB, "draft id out of vocab");
    }
}

#[test]
#[ignore = "needs DISMANTLE_Q1P5_HEAD=/path/to/q1p5_eagle6_long.safetensors"]
fn trained_head_q1p5_loads() {
    let path = head_path("DISMANTLE_Q1P5_HEAD")
        .expect("set DISMANTLE_Q1P5_HEAD to the q1p5 safetensors path");
    let head = Eagle5Head::load_from_safetensors(&path, Q1P5_HIDDEN, QWEN_VOCAB)
        .expect("q1p5 head must load (note: 2-block — exercises extra_blocks.* path)");
    assert_eq!(head.hidden(), Q1P5_HIDDEN);
    assert_eq!(head.vocab(), QWEN_VOCAB);
    let mut h = head;
    let drafts = h.propose(0, 4);
    assert_eq!(drafts.len(), 4);
}

#[test]
#[ignore = "needs DISMANTLE_Q3B_HEAD env"]
fn trained_head_rejects_wrong_hidden() {
    let path = head_path("DISMANTLE_Q3B_HEAD")
        .expect("set DISMANTLE_Q3B_HEAD to the q3b safetensors path");
    // Pass deliberately-wrong hidden — loader must refuse.
    let err = Eagle5Head::load_from_safetensors(&path, Q3B_HIDDEN + 1, QWEN_VOCAB);
    assert!(err.is_err(), "loader must reject hidden_dim mismatch");
}
