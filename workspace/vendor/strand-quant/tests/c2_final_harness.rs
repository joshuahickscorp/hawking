//! Standalone test harness for `src/c2_final.rs` — the synthesized C2 coder
//! (attempt-4 mode-adaptive engine + attempt-0 sub_scale graft).
//!
//! `c2_final` is intentionally NOT yet declared in `lib.rs` (wiring it is a
//! shared-file edit left to the operator, exactly like its `sideinfo_rans` and
//! `c2_attempt_4` siblings). To exercise the **real shipping source** — including
//! its inline `#[cfg(test)] mod tests` byte-exact round-trip + MEASURE suite —
//! without touching any shared file, this integration test `#[path]`-includes the
//! module source directly. Because this is a `--test` target, `cfg(test)` is set,
//! so the module's own `mod tests` is compiled and its `#[test]` fns run here.
//!
//! Run:  cargo test -p strand-quant --test c2_final_harness -- --nocapture

#[path = "../src/c2_final.rs"]
mod c2_final;

use c2_final::{
    decode_positions, decode_scale_q, decode_sub_scales, encode_positions, encode_scale_q,
    encode_sub_scales,
};

#[test]
fn public_api_all_three_streams_round_trip() {
    // scale_q
    let scale_q: [i32; 12] = [0, 1, -1, 100, 100, 100, -50, 0, 1, 1, -32768, 32767];
    let enc = encode_scale_q(&scale_q);
    let mut pos = 0usize;
    assert_eq!(decode_scale_q(&enc, &mut pos).unwrap(), scale_q);
    assert_eq!(pos, enc.len());

    // outlier positions
    let positions: [u32; 7] = [3, 7, 8, 100, 101, 5000, 1_000_000];
    let enc = encode_positions(&positions);
    let mut pos = 0usize;
    assert_eq!(decode_positions(&enc, &mut pos).unwrap(), positions);
    assert_eq!(pos, enc.len());

    // sub_scales (the grafted lever): unpacked 6-bit codes, alphabet 0..64
    let codes: Vec<u8> = vec![63, 63, 62, 63, 60, 63, 63, 31, 63, 63, 0, 63];
    let enc = encode_sub_scales(&codes);
    let mut pos = 0usize;
    assert_eq!(decode_sub_scales(&enc, &mut pos).unwrap(), codes);
    assert_eq!(pos, enc.len());
}
