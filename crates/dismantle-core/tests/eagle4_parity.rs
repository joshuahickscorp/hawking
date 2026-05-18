//! EAGLE-4 head parity test — Rust vs Python at atol=1e-3 fp16.
//!
//! Scaffolded by the eagle4 wiring handoff. The body is `#[ignore]`'d and
//! gated on `EAGLE4_PARITY_TEST=1` because:
//!   1. It requires `eagle4/checkpoints/best.npz` to be present locally
//!      (gitignored — too large for the repo).
//!   2. It shells out to `python eagle4/eagle4.py eval ...` for the
//!      reference forward, which needs the eagle4 venv set up.
//!
//! See `reports/path_to_90/eagle4_wiring_handoff.md § Step 5` for the
//! protocol this test implements.
//!
//! **The body is intentionally unimplemented.** This file ships now so the
//! next session has a concrete landing site rather than a green-field
//! start. The test signature, gating, and protocol comment are stable;
//! the next session fills in the four steps marked `TODO(eagle4-wiring)`.
//!
//! When implemented and `Eagle4Head::from_npz` + `Eagle4Head::propose`
//! both work, run:
//!
//! ```bash
//! EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --test eagle4_parity -- --ignored
//! ```

use std::env;

#[test]
#[ignore = "Requires eagle4/checkpoints/best.npz + python venv; gate via EAGLE4_PARITY_TEST=1"]
fn eagle4_head_parity_rust_vs_python() {
    if env::var("EAGLE4_PARITY_TEST").ok().as_deref() != Some("1") {
        eprintln!(
            "skipping eagle4 parity test; set EAGLE4_PARITY_TEST=1 and \
             ensure eagle4/checkpoints/best.npz exists"
        );
        return;
    }

    // TODO(eagle4-wiring) — step 1:
    // Load the trained head via Eagle4Head::from_npz("eagle4/checkpoints/best.npz").
    // That method currently returns Err(Unimplemented) — implementing it is
    // the bulk of this wiring step. NPZ format is ZIP of NPY files; one
    // small `npz.rs` module under crates/dismantle-core/src/util/ can
    // parse it without external deps. Key naming from
    // eagle4.py::_flat_params — see Eagle4Weights's docstring at
    // crates/dismantle-core/src/speculate/eagle4_head.rs:67-92.

    // TODO(eagle4-wiring) — step 2:
    // Pick a deterministic fixture. Two options:
    //   (a) Generate synthetically via fixed-seed RNG (no eagle4 dependency
    //       on data shards). Simpler. Use this for the first pass.
    //   (b) Read 10 records from eagle4/data/heldout/shard_00000.parquet
    //       if eagle4's held-out data is checked in. Truer-to-life, but
    //       requires parquet reader in Rust (parquet2 or arrow-rs crate).
    //   Recommend (a) for the first parity test; add (b) as a follow-up
    //   once Eagle4Head forward is solid.

    // TODO(eagle4-wiring) — step 3:
    // Run dismantle's Eagle4Head::propose on each fixture record. Capture
    // token_logits[V=102400], mask_logits[26*64], calib_logit per record.

    // TODO(eagle4-wiring) — step 4:
    // Run Python reference via std::process::Command:
    //     python eagle4/eagle4.py eval \
    //         --ckpt eagle4/checkpoints/best.npz \
    //         --frozen eagle4/v2lite_frozen.npz \
    //         --parquet <fixture path or stdin> \
    //         --max-records 10
    // The current eagle4.py eval prints metrics but doesn't emit per-record
    // raw logits — a small flag (`--dump-logits /path/to/out.npz`) might
    // need to land in eagle4.py first. Coordinate with the user before
    // adding it.

    // TODO(eagle4-wiring) — step 5:
    // Diff:
    //   - token_argmax: must match exactly per record
    //   - mask_logits: atol 1e-3 fp16
    //   - calib_logit: atol 1e-3 fp16
    //   - draft_hidden: atol 1e-3 fp16 (optional intermediate check —
    //     helpful for debugging if argmaxes diverge)

    panic!(
        "eagle4 parity test body is unimplemented — see \
         reports/path_to_90/eagle4_wiring_handoff.md § Step 5 \
         and the TODO(eagle4-wiring) markers in this file"
    );
}

#[test]
fn parity_test_module_compiles() {
    // Trivial test so the file isn't entirely #[ignore]'d. Confirms the
    // module compiles in the workspace's normal test profile.
    assert_eq!(2 + 2, 4);
}
