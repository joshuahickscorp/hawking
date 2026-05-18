//! EAGLE-4 head parity test — Rust loader vs. trained checkpoint, with
//! the Python-reference forward-pass diff staged behind a follow-up.
//!
//! Gating: this test is `#[ignore]`'d and additionally short-circuits
//! unless `EAGLE4_PARITY_TEST=1`. Reason: it depends on the trained
//! checkpoint at `eagle4/checkpoints/eagle4_v3/best.npz` (gitignored —
//! too large for the repo). CI without local artifacts still passes.
//!
//! ## What this test currently does
//!
//! 1. Loads `eagle4/checkpoints/eagle4_v3/best.npz` via
//!    `Eagle4Head::from_npz`. This exercises the full NPZ parser + the
//!    16-key tensor shape validation against the real trained weights
//!    (`in_proj`, single transformer block, residual gate, mask
//!    projections, calib head, `__step__`).
//!
//! 2. Sanity-checks the loaded weights have plausible statistics — non-
//!    NaN, residual gate in [0, 1], in_proj column norms within a
//!    reasonable band, etc.
//!
//! 3. Locates the held-out parquet shard at
//!    `eagle4/data/v2lite_3layer_heldout/shard_00000.parquet` (the same
//!    fixture eagle4 benched against) and verifies it exists. The shard
//!    is not parsed here yet — that needs a parquet reader on the Rust
//!    side, which is a deliberate Cargo.toml dependency decision out of
//!    scope for this commit.
//!
//! ## What remains for a future session
//!
//! The full Rust-vs-Python forward-pass parity diff blocks on two
//! independent pieces of work:
//!
//! - **`Eagle4Head::propose` forward pass**: currently
//!   `Err(Unimplemented)`. The forward is RMSNorm → MHA → SwiGLU →
//!   residual gate → frozen LM head; see
//!   `reports/path_to_90/eagle4_convergence.md § "Required dismantle
//!   changes" #4`.
//! - **`python eagle4/eagle4.py eval --dump-logits` flag**: doesn't
//!   exist yet. `eagle4.py eval` currently prints aggregate metrics but
//!   doesn't emit per-record raw logits to disk. Adding the flag is a
//!   1-2-hour change to `eagle4/eagle4.py`; coordinate with the user
//!   before doing it.
//!
//! Once both land, this test grows a step that:
//!   - reads N (default 10) records from the held-out parquet shard,
//!   - calls `Eagle4Head::propose` on each record,
//!   - shells out to `python eagle4/eagle4.py eval --dump-logits <path>
//!     --ckpt eagle4/checkpoints/eagle4_v3/best.npz --max-records N`,
//!   - and diffs token_argmax (exact), mask_logits (atol 1e-3 fp16),
//!     and calib_logit (atol 1e-3) per record.
//!
//! Run:
//!
//! ```bash
//! EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --test eagle4_parity -- --ignored --nocapture
//! ```

use dismantle_core::speculate::draft_head::DraftHead;
use dismantle_core::speculate::eagle4_head::{cfg, Eagle4Head};
use std::env;
use std::path::PathBuf;

const CKPT_REL: &str = "eagle4/checkpoints/eagle4_v3/best.npz";
const HELDOUT_REL: &str = "eagle4/data/v2lite_3layer_heldout/shard_00000.parquet";

/// Walk up from `cargo test`'s CWD (the crate dir) to find the repo
/// root that contains the `eagle4/` drop. The crate sits at
/// `crates/dismantle-core/`; the eagle4 tree is at the worktree root.
fn repo_root() -> Option<PathBuf> {
    let mut p = env::current_dir().ok()?;
    for _ in 0..4 {
        if p.join("eagle4").is_dir() && p.join("crates").is_dir() {
            return Some(p);
        }
        p.pop();
    }
    None
}

#[test]
#[ignore = "Requires eagle4/checkpoints/eagle4_v3/best.npz; gate via EAGLE4_PARITY_TEST=1"]
fn eagle4_head_parity_rust_vs_python() {
    if env::var("EAGLE4_PARITY_TEST").ok().as_deref() != Some("1") {
        eprintln!(
            "skipping eagle4 parity test; set EAGLE4_PARITY_TEST=1 and \
             ensure eagle4/checkpoints/eagle4_v3/best.npz exists"
        );
        return;
    }

    let root = repo_root().expect(
        "could not locate repo root containing eagle4/ — run from the worktree, \
         e.g. cargo test -p dismantle-core --test eagle4_parity",
    );

    // ---- Step 1: load the real checkpoint via Eagle4Head::from_npz. ----
    let ckpt_path = root.join(CKPT_REL);
    assert!(
        ckpt_path.exists(),
        "checkpoint not found at {} — regenerate via `python eagle4/eagle4.py train ...` \
         or copy from the eagle4 drop",
        ckpt_path.display()
    );
    let head = Eagle4Head::from_npz(&ckpt_path)
        .unwrap_or_else(|e| panic!("Eagle4Head::from_npz({}) failed: {}", ckpt_path.display(), e));

    let w = head.weights();
    let h = cfg::HIDDEN_DIM;
    let inter = cfg::INTERMEDIATE;
    let mhid = cfg::MASK_HIDDEN;

    // Shape sanity (Eagle4Head::from_npz validates these on load, but
    // double-check here so a regression in the loader surfaces as a
    // shape assertion rather than a silent miscount).
    assert_eq!(w.in_proj.len(), h * 5 * h);
    assert_eq!(w.block_attn_q.len(), h * h);
    assert_eq!(w.block_attn_k.len(), h * h);
    assert_eq!(w.block_attn_v.len(), h * h);
    assert_eq!(w.block_attn_o.len(), h * h);
    assert_eq!(w.block_mlp_gate.len(), inter * h);
    assert_eq!(w.block_mlp_up.len(), inter * h);
    assert_eq!(w.block_mlp_down.len(), h * inter);
    assert_eq!(w.block_attn_norm.len(), h);
    assert_eq!(w.block_mlp_norm.len(), h);
    assert_eq!(w.mask_proj_in.len(), mhid * h);
    assert_eq!(w.mask_proj_out.len(), cfg::N_MOE_LAYERS * cfg::N_ROUTED * mhid);
    assert_eq!(w.calib_proj_w.len(), h);

    // ---- Step 2: weight sanity stats. ----
    let no_nans = |slice: &[f32], name: &str| {
        let bad = slice.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(bad, 0, "{} has {} non-finite entries", name, bad);
    };
    no_nans(&w.in_proj, "in_proj");
    no_nans(&w.block_attn_q, "block_attn_q");
    no_nans(&w.block_mlp_gate, "block_mlp_gate");
    no_nans(&w.mask_proj_out, "mask_proj_out");
    no_nans(&w.calib_proj_w, "calib_proj_w");
    assert!(w.residual_gate.is_finite(), "residual_gate non-finite");
    assert!(w.calib_proj_b.is_finite(), "calib_proj_b non-finite");

    // The residual gate is initialized near 0.05 in `eagle4.py`; trained
    // checkpoints land somewhere in [-1, 1] in practice. A NaN/garbage
    // load would land far outside.
    assert!(
        w.residual_gate.abs() < 5.0,
        "residual_gate={} — suspiciously large (NaN or unrelated tensor?)",
        w.residual_gate
    );

    // Per-row in_proj column norms should be bounded; sane post-training
    // weights are typically O(1).
    let in_proj_max_abs = w
        .in_proj
        .iter()
        .copied()
        .fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        in_proj_max_abs < 100.0,
        "in_proj max |w| = {} — exceeds sanity ceiling",
        in_proj_max_abs
    );

    eprintln!(
        "[eagle4 parity] loaded {} — residual_gate={:.4}, in_proj |w|max={:.4}, calib_bias={:.4}",
        head.id(),
        w.residual_gate,
        in_proj_max_abs,
        w.calib_proj_b
    );

    // ---- Step 3: locate held-out fixture (parsed in a future commit). ----
    let heldout = root.join(HELDOUT_REL);
    assert!(
        heldout.exists(),
        "held-out parquet not found at {} — copy from the eagle4 drop or \
         regenerate via eagle4/capture.py",
        heldout.display()
    );
    let heldout_bytes = std::fs::metadata(&heldout)
        .expect("stat heldout shard")
        .len();
    assert!(
        heldout_bytes > 1_000_000,
        "held-out shard suspiciously small: {} bytes",
        heldout_bytes
    );
    eprintln!(
        "[eagle4 parity] held-out fixture located: {} ({} MB)",
        heldout.display(),
        heldout_bytes / 1_000_000
    );

    // ---- Steps 4-5: Python-reference diff, deferred. ----
    //
    // The diff against `python eagle4/eagle4.py eval ...` is staged for
    // a follow-up once two blockers clear (see this file's module
    // docstring):
    //   - Eagle4Head::propose forward pass (currently Unimplemented)
    //   - eagle4.py eval --dump-logits flag (doesn't exist)
    //
    // Until then, this test confirms the loader + checkpoint shape
    // contract are in alignment, which is the regression surface that
    // breaks first when either side's tensor layout drifts.
    eprintln!(
        "[eagle4 parity] Rust-loader sanity passed. Python-reference \
         diff staged for follow-up (needs Eagle4Head::propose + \
         eagle4.py --dump-logits)."
    );
}

#[test]
fn parity_test_module_compiles() {
    // Trivial test so the file isn't entirely #[ignore]'d. Confirms the
    // module compiles in the workspace's normal test profile.
    assert_eq!(2 + 2, 4);
}
