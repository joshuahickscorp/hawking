use hawking_serve::http::banked_len_for;
use hawking_serve::SystemPromptKvBank;
fn sys_prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 4000 + i).collect()
}
#[test]
fn serve_wiring_record_then_lookup_returns_source_slot() {
    let prompt = sys_prompt(40);
    let banked_len = banked_len_for(&prompt);
    assert_eq!(
        banked_len,
        prompt.len() - 1,
        "bank one token short of full prompt"
    );
    let mut bank = SystemPromptKvBank::new();
    let outcome = bank.record(&prompt, banked_len, 3);
    assert_eq!(
        outcome,
        hawking_serve::system_kv_bank::RecordOutcome::Inserted
    );
    let hit = bank
        .lookup(&prompt, banked_len)
        .expect("bank must hit on identical prompt");
    assert_eq!(
        hit.source_slot, 3,
        "lookup must return the slot that recorded"
    );
    assert_eq!(hit.prefix_len, banked_len, "hit prefix_len == banked_len");
}
#[test]
fn wiring_banked_len_is_a_strict_prefix() {
    for n in [9usize, 16, 40, 257] {
        let p = sys_prompt(n);
        let bl = banked_len_for(&p);
        assert!(bl < p.len(), "banked_len must be < prompt len for n={n}");
        assert!(
            bl >= 8,
            "for prompts >= 9 the banked span clears the min, n={n}"
        );
    }
}
#[test]
fn wiring_shared_system_span_hits_across_suffix() {
    let system_span_len = 24usize; // the fixed leading system block
    let mut a = sys_prompt(system_span_len); // turn 1 prompt = system + suffix A
    a.extend_from_slice(&[10, 11, 12]);
    let mut b = sys_prompt(system_span_len); // turn 2 prompt = system + suffix B
    b.extend_from_slice(&[20, 21, 22, 23]);
    let mut bank = SystemPromptKvBank::new();
    bank.record(&a, system_span_len, 5);
    let hit = bank
        .lookup(&b, system_span_len)
        .expect("shared system span must hit");
    assert_eq!(hit.source_slot, 5);
}
