//! Spine A — derive an EFFECTIVE-CONTEXT multiplier from the `.tq` (strand v2)
//! artifact's REAL, measured weight compression. Never a hardcoded number: we
//! read the actual per-tensor bit budget out of the file and turn the measured
//! compression ratio into a conservative, clearly-estimated multiplier.
//!
//! Honesty: weight compression does not *directly* enlarge the context window.
//! It frees RAM that can instead hold a longer KV cache / sequence, so a larger
//! effective window becomes affordable. We surface the freed-memory ratio as an
//! ESTIMATE (capped), and the runtime presents it as "context expanded ~Nx",
//! never as a guaranteed token count. If the file is absent or unparseable, the
//! multiplier is 1.0 — i.e. we make no claim.

use std::path::Path;

/// Reference precision the compression ratio is measured against (fp16 weights).
const REFERENCE_BPW: f32 = 16.0;
/// Cap so an aggressive quant never inflates the headline beyond what is honest.
const MAX_MULTIPLIER: f32 = 8.0;

/// Measured compression facts read from a `.tq` artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct TqContextInfo {
    /// Effective bits-per-weight measured from the artifact (payload bits / weights).
    pub bpw: f32,
    /// Weight compression vs fp16 (`REFERENCE_BPW / bpw`).
    pub compression_ratio: f32,
    /// Conservative effective-context multiplier derived from the compression,
    /// clamped to `[1.0, MAX_MULTIPLIER]`.
    pub multiplier: f32,
    /// Always true today: this is a derived estimate, not a guaranteed ceiling.
    pub estimated: bool,
}

/// Turn a measured bits-per-weight into a conservative context multiplier.
/// Pure + unit-tested. `bpw <= 0`, non-finite, or `>= REFERENCE_BPW` yields 1.0
/// (no inflation — we never claim expansion we cannot justify).
pub fn bpw_to_multiplier(bpw: f32) -> f32 {
    if !bpw.is_finite() || bpw <= 0.0 || bpw >= REFERENCE_BPW {
        return 1.0;
    }
    (REFERENCE_BPW / bpw).clamp(1.0, MAX_MULTIPLIER)
}

/// Read a `.tq` (strand v2) file and derive its context info from the REAL
/// measured compression. Returns `None` (make no claim) when the file is
/// missing, not a strand v2 artifact, or unparseable — the caller then treats
/// the multiplier as 1.0.
pub fn read_tq_context(path: &Path) -> Option<TqContextInfo> {
    let buf = std::fs::read(path).ok()?;
    let header = strand_quant::format::read_strand_v2_header(&buf).ok()?;
    let mut weights: u128 = 0;
    let mut payload_bytes: u128 = 0;
    for t in &header.tensors {
        weights += t.total as u128;
        payload_bytes += t.payload_bytes as u128;
    }
    if weights == 0 {
        return None;
    }
    let bpw = (payload_bytes as f64 * 8.0 / weights as f64) as f32;
    Some(TqContextInfo {
        bpw,
        compression_ratio: REFERENCE_BPW / bpw.max(f32::MIN_POSITIVE),
        multiplier: bpw_to_multiplier(bpw),
        estimated: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplier_tracks_compression_and_is_capped() {
        // ~3 bpw (typical Q4-ish trellis) => ~5.3x, within the cap.
        assert!((bpw_to_multiplier(3.0) - 16.0 / 3.0).abs() < 1e-4);
        // Very aggressive 1 bpw would be 16x => capped to MAX_MULTIPLIER.
        assert_eq!(bpw_to_multiplier(1.0), MAX_MULTIPLIER);
    }

    #[test]
    fn no_claim_on_degenerate_or_uncompressed() {
        assert_eq!(bpw_to_multiplier(16.0), 1.0, "fp16 => no expansion");
        assert_eq!(bpw_to_multiplier(0.0), 1.0, "degenerate => no claim");
        assert_eq!(bpw_to_multiplier(f32::NAN), 1.0, "non-finite => no claim");
        assert_eq!(bpw_to_multiplier(-2.0), 1.0, "negative => no claim");
    }

    #[test]
    fn missing_file_makes_no_claim() {
        assert_eq!(read_tq_context(Path::new("/no/such/file.tq")), None);
    }
}
