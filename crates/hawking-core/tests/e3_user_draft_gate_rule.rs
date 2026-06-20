//! Track 0/9 lock-in (DESIGN MIRROR): pins the E3 inline-pair default rule.
//!
//! The live gate is an INLINE `OnceLock`-cached closure in
//! `crates/hawking-core/src/model/qwen_dense.rs` (the `ffn_pair_2r_inline`
//! binding). It is NOT exposed as a pure fn and extracting one would mean
//! editing that file, which is out of scope here. So this test pins a TINY pure
//! MIRROR of the same boolean rule. If you ever change the live gate, change
//! `e3_default` below to match — the two must stay in lockstep.
//!
//! Live rule being mirrored (verbatim semantics):
//!   let explicit = std::env::var_os("HAWKING_QWEN_PAIR_2R_INLINE")
//!                      .map(|v| v != "0");          // Some(true) for any non-"0"
//!   // when unset, default-ON for plain decode but default-OFF under user-draft:
//!   explicit.unwrap_or_else(|| !crate::env_on("HAWKING_QWEN_USER_DRAFT"))
//! where `env_on(x)` is true IFF the var is exactly "1".
//!
//! Gate (CPU, no GPU, no env, no model):
//!   cargo test -p hawking-core --test e3_user_draft_gate_rule

/// Pure mirror of the qwen_dense `ffn_pair_2r_inline` default decision.
///
/// * `user_draft_on` — the value of `env_on("HAWKING_QWEN_USER_DRAFT")`
///   (true only when the var is exactly "1").
/// * `explicit` — the resolved explicit override:
///   `var_os("HAWKING_QWEN_PAIR_2R_INLINE").map(|v| v != "0")`
///   (None = unset; Some(false) only for "0"; Some(true) for any other value).
///
/// Returns whether E3 (the 2r inline pair) is ON.
fn e3_default(user_draft_on: bool, explicit: Option<bool>) -> bool {
    explicit.unwrap_or(!user_draft_on)
}

#[test]
fn explicit_override_always_wins() {
    // An explicit HAWKING_QWEN_PAIR_2R_INLINE pins the result regardless of
    // user-draft (power user accepting the draft bit-identity trade).
    assert!(
        e3_default(true, Some(true)),
        "explicit=1 forces E3 ON even under user-draft"
    );
    assert!(e3_default(false, Some(true)), "explicit=1 forces E3 ON");
    assert!(!e3_default(false, Some(false)), "explicit=0 forces E3 OFF");
    assert!(
        !e3_default(true, Some(false)),
        "explicit=0 forces E3 OFF even without user-draft"
    );
}

#[test]
fn unset_default_is_off_under_user_draft_on_otherwise() {
    // THE load-bearing invariant (the 2026-06-07 regression this guards):
    // with no explicit override, E3 is ON for plain decode and OFF when the
    // user n-gram draft is active (keeps forward_tokens_verify bit-identical).
    assert!(
        e3_default(false, None),
        "unset + no user-draft => E3 ON (+9.6%)"
    );
    assert!(
        !e3_default(true, None),
        "unset + user-draft ON => E3 OFF (draft stays lossless)"
    );
}
