//! A prefix checkpoint must not be restorable across a `reset()`.
//!
//! The checkpoint carries `rec_state` and `conv_state`. It does NOT carry the KV
//! cache, and `reset()` zeroes that cache. Restoring across a reset leaves the
//! 16 full-attention layers reading an EMPTY cache while the recurrent state
//! insists a long prefix has already been consumed.
//!
//! Measured before the guard, 256-token prefix on the real resident: the next
//! sampled token was 4242 after reset+restore, against 358 for the honest walk.
//! `restore_prefix` documented only the prefix-token requirement and said
//! nothing about KV integrity, so nothing downstream could detect it.
//!
//! Needs the resident artifact; skips cleanly when it is absent rather than
//! passing vacuously, and says so.

#![cfg(target_os = "macos")]

use hawking_core::model::qwen38_hybrid_decode::Qwen38HybridDecodeSession;
use std::path::PathBuf;

fn artifact() -> Option<PathBuf> {
    let p = std::env::var("HAWKING_TEST_ARTIFACT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("noetic/NOETIC_PARENT_A")
        });
    if p.join("MIX_REPORT.json").is_file() {
        Some(p)
    } else {
        None
    }
}

#[test]
fn restore_refuses_across_reset_but_not_within_an_epoch() {
    let Some(root) = artifact() else {
        eprintln!("SKIP: no resident artifact; this test needs one and is NOT vacuously passing");
        return;
    };
    let mut s = match Qwen38HybridDecodeSession::open(&root, 512) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: could not open session: {e}");
            return;
        }
    };

    s.reset();
    for t in 0..8u32 {
        s.step_unmeasured(1000 + t).expect("step");
    }
    let ck = s.prefix_checkpoint().expect("checkpoint");

    // THE NEGATIVE CONTROL, first: within its own epoch the checkpoint must
    // still work. A blanket refusal would look safe while making the feature
    // useless, and would pass the assertion below for the wrong reason.
    s.restore_prefix(&ck)
        .expect("a checkpoint must still restore within the epoch it was taken in");

    // Across a reset it must refuse rather than answer differently.
    s.reset();
    let err = s
        .restore_prefix(&ck)
        .expect_err("restoring across reset() must be refused: the KV cache it depends on is gone");
    let msg = format!("{err}");
    assert!(
        msg.contains("epoch"),
        "the refusal must name the cause so a caller can act on it, got: {msg}"
    );
}
