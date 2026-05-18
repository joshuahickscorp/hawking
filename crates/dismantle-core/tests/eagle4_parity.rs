//! EAGLE-4 head parity test — Rust `Eagle4Head::forward_full` vs
//! `eagle4/eagle4.py eval` Python reference, atol=1e-3 fp16.
//!
//! Gating: `#[ignore]`'d AND short-circuits unless
//! `EAGLE4_PARITY_TEST=1`. The test depends on three local artifacts
//! (all gitignored — too large for the repo):
//!
//! - `eagle4/checkpoints/eagle4_v3/best.npz`   — trained head checkpoint
//! - `eagle4/v2lite_frozen.npz`                 — frozen V2-Lite tensors
//! - `eagle4/data/v2lite_3layer_heldout/shard_00000.parquet` — held-out
//!   fixture (same data eagle4.py was benched against)
//!
//! Plus a Python with `mlx-lm` installed. Discovery order for the
//! python binary:
//!
//! 1. `EAGLE4_PYTHON` env var if set (full path)
//! 2. `~/Downloads/eagle4/.venv/bin/python3` (the in-source-dir venv)
//! 3. `python3` from `$PATH`
//!
//! ## What this test does
//!
//! 1. Loads `Eagle4Head::from_npz(best.npz)` + attaches frozen weights
//!    from `v2lite_frozen.npz` via `Eagle4FrozenWeights::from_npz`.
//! 2. Spawns `python eagle4/eagle4.py eval --dump-logits /tmp/eagle4_ref.npz
//!    --max-records 10 ...` to produce a per-record reference NPZ
//!    containing token_logits, mask_logits, draft_hidden, calib_logit
//!    plus the input hiddens the head saw.
//! 3. For each of the 10 records, calls
//!    `Eagle4Head::forward_full(prev_token, h_low, h_mid, h_high, h_shared)`
//!    against the SAME inputs the Python eval used (`h_*` columns of the
//!    reference NPZ).
//! 4. Diffs Rust vs Python per record:
//!    - `token_argmax` — must match exactly per record;
//!    - `mask_logits` (1664 floats) at atol=1e-3;
//!    - `calib_logit` (scalar) at atol=1e-3;
//!    - `draft_hidden` (2048 floats) at atol=1e-3.
//!
//! Run:
//!
//! ```bash
//! EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --release \
//!     --test eagle4_parity -- --ignored --nocapture
//! ```

use dismantle_core::speculate::draft_head::DraftHead;
use dismantle_core::speculate::eagle4_head::{cfg, Eagle4FrozenWeights, Eagle4Head};
use dismantle_core::util::npz::read_npz;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CKPT_REL: &str = "eagle4/checkpoints/eagle4_v3/best.npz";
const FROZEN_REL: &str = "eagle4/v2lite_frozen.npz";
const HELDOUT_REL: &str = "eagle4/data/v2lite_3layer_heldout/shard_00000.parquet";
const SCRIPT_REL: &str = "eagle4/eagle4.py";
const N_RECORDS: usize = 10;
const ATOL_FP16: f32 = 1.0e-3;

/// Walk up from `cargo test`'s CWD to find the worktree root that
/// contains both `eagle4/` and `crates/`.
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

/// Resolve which python binary to use. EAGLE4_PYTHON overrides;
/// the in-source-dir venv is the default fallback.
fn resolve_python() -> String {
    if let Ok(p) = env::var("EAGLE4_PYTHON") {
        if !p.is_empty() {
            return p;
        }
    }
    let home = env::var("HOME").unwrap_or_else(|_| "/Users/scammermike".to_string());
    let venv = format!("{home}/Downloads/eagle4/.venv/bin/python3");
    if Path::new(&venv).exists() {
        return venv;
    }
    "python3".to_string()
}

/// Run the eagle4 eval subprocess with --dump-logits and return the
/// path it wrote.
fn run_python_dump(root: &Path, out_npz: &Path) {
    let py = resolve_python();
    let status = Command::new(&py)
        .arg(root.join(SCRIPT_REL))
        .arg("eval")
        .arg("--ckpt")
        .arg(root.join(CKPT_REL))
        .arg("--frozen")
        .arg(root.join(FROZEN_REL))
        .arg("--parquet")
        .arg(root.join(HELDOUT_REL))
        .arg("--max-records")
        .arg(N_RECORDS.to_string())
        .arg("--dump-logits")
        .arg(out_npz)
        .current_dir(root)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "failed to spawn python: {e}\n  python = {py}\n  set EAGLE4_PYTHON to a python3 \
                 with mlx-lm installed"
            )
        });
    assert!(
        status.success(),
        "python eagle4/eagle4.py eval exited with {:?}",
        status.code()
    );
    assert!(
        out_npz.exists(),
        "python eval finished but did not write {}",
        out_npz.display()
    );
}

fn argmax(slice: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

#[test]
#[ignore = "Requires eagle4 artifacts + mlx-lm Python; gate via EAGLE4_PARITY_TEST=1"]
fn eagle4_head_parity_rust_vs_python() {
    if env::var("EAGLE4_PARITY_TEST").ok().as_deref() != Some("1") {
        eprintln!(
            "skipping eagle4 parity test; set EAGLE4_PARITY_TEST=1 and ensure \
             eagle4/checkpoints/eagle4_v3/best.npz + eagle4/v2lite_frozen.npz + \
             eagle4/data/v2lite_3layer_heldout/shard_00000.parquet exist"
        );
        return;
    }

    let root = repo_root().expect(
        "could not locate repo root containing eagle4/ — run cargo test from \
         the worktree",
    );

    // ---- 1. Load the Rust-side head + frozen weights. ----
    let ckpt_path = root.join(CKPT_REL);
    let frozen_path = root.join(FROZEN_REL);
    let heldout_path = root.join(HELDOUT_REL);
    for (label, p) in [
        ("checkpoint", &ckpt_path),
        ("frozen", &frozen_path),
        ("heldout", &heldout_path),
    ] {
        assert!(
            p.exists(),
            "{} not found at {} — copy from the eagle4 drop or regenerate",
            label,
            p.display()
        );
    }

    eprintln!("[eagle4 parity] loading Rust-side head + frozen ...");
    let mut head = Eagle4Head::from_npz(&ckpt_path)
        .unwrap_or_else(|e| panic!("Eagle4Head::from_npz failed: {e}"));
    let frozen = Eagle4FrozenWeights::from_npz(&frozen_path)
        .unwrap_or_else(|e| panic!("Eagle4FrozenWeights::from_npz failed: {e}"));
    head.set_frozen(frozen);
    eprintln!(
        "[eagle4 parity] loaded {} (residual_gate={:.4})",
        head.id(),
        head.weights().residual_gate
    );

    // ---- 2. Run the Python reference dump. ----
    let dump_dir = env::temp_dir().join("eagle4_parity_dump");
    std::fs::create_dir_all(&dump_dir).ok();
    let dump_path = dump_dir.join("ref.npz");
    let _ = std::fs::remove_file(&dump_path);
    eprintln!(
        "[eagle4 parity] running python reference → {}",
        dump_path.display()
    );
    run_python_dump(&root, &dump_path);

    // ---- 3. Read the dumped reference. ----
    let mut ref_npz = read_npz(&dump_path).expect("read /tmp/.../ref.npz");

    fn take_f32(
        bag: &mut std::collections::HashMap<String, dismantle_core::util::npz::NpyArray>,
        k: &str,
        expected_shape: &[usize],
    ) -> Vec<f32> {
        let arr = bag
            .remove(k)
            .unwrap_or_else(|| panic!("ref NPZ missing key '{k}'"));
        assert_eq!(
            arr.shape, expected_shape,
            "ref NPZ '{}': shape {:?}, expected {:?}",
            k, arr.shape, expected_shape
        );
        arr.as_f32().unwrap_or_else(|e| panic!("{k} as_f32: {e}"))
    }
    fn take_i32(
        bag: &mut std::collections::HashMap<String, dismantle_core::util::npz::NpyArray>,
        k: &str,
        expected_shape: &[usize],
    ) -> Vec<i32> {
        let arr = bag
            .remove(k)
            .unwrap_or_else(|| panic!("ref NPZ missing key '{k}'"));
        assert_eq!(
            arr.shape, expected_shape,
            "ref NPZ '{}': shape {:?}, expected {:?}",
            k, arr.shape, expected_shape
        );
        let n: usize = arr.shape.iter().product();
        assert_eq!(arr.data.len(), n * 4, "i32 array byte mismatch for {k}");
        let mut out = Vec::with_capacity(n);
        for chunk in arr.data.chunks_exact(4) {
            out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out
    }

    let h = cfg::HIDDEN_DIM;
    let vocab = cfg::VOCAB;
    let mhid_out = cfg::N_MOE_LAYERS * cfg::N_ROUTED;

    let ref_token_logits = take_f32(&mut ref_npz, "token_logits", &[N_RECORDS, vocab]);
    let ref_mask_logits = take_f32(
        &mut ref_npz,
        "mask_logits",
        &[N_RECORDS, cfg::N_MOE_LAYERS, cfg::N_ROUTED],
    );
    let ref_calib_logit = take_f32(&mut ref_npz, "calib_logit", &[N_RECORDS]);
    let ref_draft_hidden = take_f32(&mut ref_npz, "draft_hidden", &[N_RECORDS, h]);
    let ref_prev_token = take_i32(&mut ref_npz, "prev_token", &[N_RECORDS]);
    let ref_h_low = take_f32(&mut ref_npz, "h_low", &[N_RECORDS, h]);
    let ref_h_mid = take_f32(&mut ref_npz, "h_mid", &[N_RECORDS, h]);
    let ref_h_high = take_f32(&mut ref_npz, "h_high", &[N_RECORDS, h]);
    let ref_h_shared = take_f32(&mut ref_npz, "h_shared", &[N_RECORDS, h]);

    // ---- 4. For each record, run Rust forward_full + diff. ----
    let mut worst_mask = (0.0f32, 0usize, 0usize); // (val, record, idx)
    let mut worst_calib = (0.0f32, 0usize);
    let mut worst_dh = (0.0f32, 0usize, 0usize);
    let mut argmax_mismatches = 0usize;

    for rec in 0..N_RECORDS {
        let prev = ref_prev_token[rec] as u32;
        let lo = rec * h..(rec + 1) * h;
        let mo = rec * mhid_out..(rec + 1) * mhid_out;
        let vo = rec * vocab..(rec + 1) * vocab;

        let out = head
            .forward_full(
                prev,
                &ref_h_low[lo.clone()],
                &ref_h_mid[lo.clone()],
                &ref_h_high[lo.clone()],
                &ref_h_shared[lo.clone()],
            )
            .unwrap_or_else(|e| panic!("forward_full record {rec}: {e}"));

        // Token argmax — must match exactly.
        let rust_arg = argmax(&out.token_logits);
        let ref_arg = argmax(&ref_token_logits[vo.clone()]);
        if rust_arg != ref_arg {
            argmax_mismatches += 1;
            eprintln!(
                "[eagle4 parity] record {rec}: token argmax MISMATCH \
                 rust={} python={} (rust_logit={:.4} python_logit={:.4})",
                rust_arg,
                ref_arg,
                out.token_logits[rust_arg],
                ref_token_logits[vo.start + ref_arg],
            );
        }

        // mask_logits diff.
        let (md, mi) = max_abs_diff(&out.mask_logits, &ref_mask_logits[mo.clone()]);
        if md > worst_mask.0 {
            worst_mask = (md, rec, mi);
        }

        // calib_logit diff.
        let cd = (out.calib_logit - ref_calib_logit[rec]).abs();
        if cd > worst_calib.0 {
            worst_calib = (cd, rec);
        }

        // draft_hidden diff.
        let (dd, di) = max_abs_diff(&out.draft_hidden, &ref_draft_hidden[lo.clone()]);
        if dd > worst_dh.0 {
            worst_dh = (dd, rec, di);
        }
    }

    eprintln!(
        "[eagle4 parity] worst diffs over {} records (atol={}):\n  \
         mask_logits  max|Δ| = {:.4e}  at (record={}, idx={})\n  \
         calib_logit  max|Δ| = {:.4e}  at  record={}\n  \
         draft_hidden max|Δ| = {:.4e}  at (record={}, idx={})\n  \
         argmax mismatches: {}/{}",
        N_RECORDS,
        ATOL_FP16,
        worst_mask.0,
        worst_mask.1,
        worst_mask.2,
        worst_calib.0,
        worst_calib.1,
        worst_dh.0,
        worst_dh.1,
        worst_dh.2,
        argmax_mismatches,
        N_RECORDS,
    );

    assert_eq!(
        argmax_mismatches, 0,
        "{}/{} record(s) had token-argmax mismatches between Rust and Python",
        argmax_mismatches, N_RECORDS
    );
    assert!(
        worst_mask.0 <= ATOL_FP16,
        "mask_logits exceeds atol={}: worst |Δ|={:.4e} at (rec={}, idx={})",
        ATOL_FP16,
        worst_mask.0,
        worst_mask.1,
        worst_mask.2,
    );
    assert!(
        worst_calib.0 <= ATOL_FP16,
        "calib_logit exceeds atol={}: worst |Δ|={:.4e} at rec={}",
        ATOL_FP16,
        worst_calib.0,
        worst_calib.1,
    );
    assert!(
        worst_dh.0 <= ATOL_FP16,
        "draft_hidden exceeds atol={}: worst |Δ|={:.4e} at (rec={}, idx={})",
        ATOL_FP16,
        worst_dh.0,
        worst_dh.1,
        worst_dh.2,
    );
}

#[test]
fn parity_test_module_compiles() {
    assert_eq!(2 + 2, 4);
}
