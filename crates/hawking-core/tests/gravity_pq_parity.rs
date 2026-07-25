//! Parity test: `hawking_core::gravity::pq_matvec` against the reference
//! `gravity-pq` fixtures in `tests/fixtures/gravity_pq/` (produced by the
//! Python `pq_execute` reference on the on-disk artifact).
//!
//! For each fixture (R0/R2/R4): parse the payload header and check it
//! against `manifest.json`, then run `pq_matvec` and compare against the
//! recorded `y` within `1e-4 + 1e-4*|y|` per element.

use std::path::{Path, PathBuf};

use hawking_core::gravity::{parse_pq_header, pq_matvec};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ManifestFixture {
    fixture: String,
    rows: u64,
    cols: u64,
    dim: u64,
    k: u64,
    subspaces: u64,
    sub: u64,
    nchunk: u64,
    index_bits: u64,
    rotate: bool,
    seed: u64,
    blob_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<ManifestFixture>,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_pq")
}

fn read_f32_le(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{path:?}: byte length {} not a multiple of 4",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn pq_matvec_matches_reference_for_all_fixtures() {
    let dir = fixtures_dir();
    let manifest: Manifest = {
        let text = std::fs::read_to_string(dir.join("manifest.json")).expect("read manifest");
        serde_json::from_str(&text).expect("parse manifest")
    };
    assert_eq!(manifest.fixtures.len(), 3, "expected R0/R2/R4 fixtures");

    for row in &manifest.fixtures {
        let blob_path = dir.join(format!("{}.bin", row.fixture));
        let payload = std::fs::read(&blob_path).unwrap_or_else(|e| panic!("read {blob_path:?}: {e}"));

        assert_eq!(
            payload.len() as u64,
            row.blob_bytes,
            "{}: file length {} != manifest blob_bytes {}",
            row.fixture,
            payload.len(),
            row.blob_bytes
        );

        // 1. Header fields must match the manifest row exactly.
        let h = parse_pq_header(&payload)
            .unwrap_or_else(|e| panic!("{}: parse_pq_header: {e}", row.fixture));
        assert_eq!(h.rows as u64, row.rows, "{}: rows", row.fixture);
        assert_eq!(h.cols as u64, row.cols, "{}: cols", row.fixture);
        assert_eq!(h.sub as u64, row.sub, "{}: sub", row.fixture);
        assert_eq!(h.nchunk as u64, row.nchunk, "{}: nchunk", row.fixture);
        assert_eq!(h.bits as u64, row.index_bits, "{}: index_bits", row.fixture);
        assert_eq!((h.rotate != 0), row.rotate, "{}: rotate", row.fixture);
        assert_eq!(h.seed as u64, row.seed, "{}: seed", row.fixture);
        assert_eq!(h.s as u64, row.subspaces, "{}: subspaces", row.fixture);
        assert_eq!(h.card as u64, row.k, "{}: k (card)", row.fixture);
        assert_eq!(h.d as u64, row.dim, "{}: dim", row.fixture);

        // 2. pq_matvec must match the reference y within tolerance.
        let x = read_f32_le(&dir.join(format!("{}.x.f32", row.fixture)));
        let y_ref = read_f32_le(&dir.join(format!("{}.y.f32", row.fixture)));
        assert_eq!(x.len() as u64, row.cols, "{}: x length", row.fixture);
        assert_eq!(y_ref.len() as u64, row.rows, "{}: y length", row.fixture);

        let y_got = pq_matvec(&payload, &x)
            .unwrap_or_else(|e| panic!("{}: pq_matvec: {e}", row.fixture));
        assert_eq!(y_got.len(), y_ref.len(), "{}: output length", row.fixture);

        // Worst element by tolerance margin (diff - tol), so the reported
        // culprit is the true worst violator, not just the largest raw diff.
        let mut worst = (0usize, f32::NEG_INFINITY, 0f32, 0f32, 0f32);
        for (i, (&a, &b)) in y_got.iter().zip(y_ref.iter()).enumerate() {
            let diff = (a - b).abs();
            let tol = 1e-4 + 1e-4 * b.abs();
            let margin = diff - tol;
            if margin > worst.1 {
                worst = (i, margin, diff, tol, a);
            }
            assert!(
                diff <= tol,
                "{}: element {i}: got {a}, want {b}, |diff|={diff} > tol={tol}",
                row.fixture
            );
        }
        let (idx, _, diff, tol, got_val) = worst;
        assert!(
            diff <= tol,
            "{}: worst element {idx}: got {got_val}, want {}, |diff|={diff} > tol={tol}",
            row.fixture,
            y_ref[idx]
        );
    }
}
