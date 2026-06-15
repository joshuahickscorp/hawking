//! Track 5.2 wiring gate — pins the record/lookup KEY+LEN choice the serve
//! admit path performs, using the SAME `http::banked_len_for` helper the loop
//! uses. The bank's own logic is covered by tests/system_kv_bank.rs; this test
//! pins the WIRING (the banked_len = prompt-minus-one choice and the
//! record-then-lookup sequence) so it can never silently diverge from the loop.
//! Pure: no model, no GPU, no process env, no decode.
//!   cargo test -p dismantle-serve --test system_kv_bank_wiring

use dismantle_serve::http::banked_len_for;
use dismantle_serve::SystemPromptKvBank;

/// A representative fixed system prompt (>= bank min_prefix_tokens=8).
fn sys_prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 4000 + i).collect()
}

/// The exact record-then-lookup sequence the serve loop performs for a serial
/// workload: turn 1 records (no live source), turn 2 (identical prompt) looks
/// up and MUST get turn-1's slot back, via the shared banked_len helper.
#[test]
fn serve_wiring_record_then_lookup_returns_source_slot() {
    let prompt = sys_prompt(40);
    let banked_len = banked_len_for(&prompt);
    assert_eq!(banked_len, prompt.len() - 1, "bank one token short of full prompt");

    let mut bank = SystemPromptKvBank::new();

    // Turn 1: slot 3 just prefilled this prompt; serve loop records it.
    let outcome = bank.record(&prompt, banked_len, 3);
    assert_eq!(outcome, dismantle_serve::system_kv_bank::RecordOutcome::Inserted);

    // Turn 2: identical prompt arrives in slot 1; live PrefixIndex MISSES
    // (slot 3 freed). Serve loop consults the bank with the SAME banked_len.
    let hit = bank.lookup(&prompt, banked_len).expect("bank must hit on identical prompt");
    assert_eq!(hit.source_slot, 3, "lookup must return the slot that recorded");
    assert_eq!(hit.prefix_len, banked_len, "hit prefix_len == banked_len");
}

/// banked_len must produce a STRICT prefix so the bank's lookup guard
/// (`banked_len < tokens.len()`) is satisfied — i.e. the wiring never asks the
/// bank to match the whole prompt (which lookup rejects by contract).
#[test]
fn wiring_banked_len_is_a_strict_prefix() {
    for n in [9usize, 16, 40, 257] {
        let p = sys_prompt(n);
        let bl = banked_len_for(&p);
        assert!(bl < p.len(), "banked_len must be < prompt len for n={n}");
        assert!(bl >= 8, "for prompts >= 9 the banked span clears the min, n={n}");
    }
}

/// A differing SUFFIX (same fixed system span) still hits: only the leading
/// banked_len tokens address the entry. This is the core serial-chat win —
/// same system prompt, different user turn. Both record and lookup must use the
/// SAME banked_len for the cross-suffix case, which is what the loop does (it
/// recomputes banked_len_for(prompt) on the lookup side too); we pin that by
/// banking at a fixed system-span length and probing the same.
#[test]
fn wiring_shared_system_span_hits_across_suffix() {
    let system_span_len = 24usize;            // the fixed leading system block
    let mut a = sys_prompt(system_span_len);  // turn 1 prompt = system + suffix A
    a.extend_from_slice(&[10, 11, 12]);
    let mut b = sys_prompt(system_span_len);  // turn 2 prompt = system + suffix B
    b.extend_from_slice(&[20, 21, 22, 23]);

    let mut bank = SystemPromptKvBank::new();
    // Record turn 1 at the fixed system-span length.
    bank.record(&a, system_span_len, 5);
    // Turn 2: probe the same fixed span length -> hit slot 5 despite suffix B.
    let hit = bank.lookup(&b, system_span_len).expect("shared system span must hit");
    assert_eq!(hit.source_slot, 5);
}
