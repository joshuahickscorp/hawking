//! W-M1-2 parity gate: the `Engine` checkpoint/fork seam round-trips the RWKV-7
//! recurrent state with NO re-prefill. Model-gated (skips without a GGUF),
//! mirroring `rwkv7_parity.rs`. The byte-level round-trip is covered
//! unconditionally by the `state_serde_tests` unit tests in `model/rwkv7.rs`;
//! this gate adds the end-to-end "restored state reproduces bit-identical
//! next-token logits" guarantee that the M1 fork/handoff work builds on.

use hawking_core::model::rwkv7::RwkvSeven;
use hawking_core::{Engine, EngineConfig};
use std::path::PathBuf;

fn locate_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HAWKING_RWKV7_F32_GGUF") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    for cand in [
        "/tmp/rwkv_ref/rwkv7-04-f32.gguf",
        "../../models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf",
    ] {
        let p = PathBuf::from(cand);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[test]
fn rwkv7_checkpoint_roundtrip_bit_identical_logits() {
    let Some(model) = locate_model() else {
        eprintln!(
            "skipping rwkv7_checkpoint_roundtrip: no RWKV-7 GGUF \
             (set HAWKING_RWKV7_F32_GGUF or place the Q4_K model under models/)"
        );
        return;
    };

    let mut engine = RwkvSeven::load(&model, EngineConfig::default()).expect("load rwkv7");
    engine.reset_kv_for_test();
    engine
        .forward_tokens_for_test(&[10, 20, 30], &[0, 1, 2])
        .expect("prefill a few tokens");

    // Snapshot the state; a fork must equal the snapshot at the same point.
    let cp = engine.save_checkpoint().expect("save_checkpoint");
    let fork = engine.fork_state().expect("fork_state");
    assert_eq!(
        cp, fork,
        "fork_state must equal the save_checkpoint snapshot"
    );

    // Step one token from the checkpoint, then restore and step the same token:
    // the restored state must reproduce bit-identical next-token logits.
    let l1 = engine
        .forward_tokens_for_test(&[42], &[3])
        .expect("step from checkpoint")
        .pop()
        .expect("one logit row");
    engine.load_checkpoint(&cp).expect("load_checkpoint");
    let l2 = engine
        .forward_tokens_for_test(&[42], &[3])
        .expect("step from restored state")
        .pop()
        .expect("one logit row");

    assert_eq!(
        l1, l2,
        "restored checkpoint must reproduce bit-identical next-token logits (no re-prefill)"
    );
}
