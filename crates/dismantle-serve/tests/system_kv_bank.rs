//! Track 5.2 gate — SystemPromptKvBank key/hit/miss/eviction logic.
//! Pure data structure; no model, no GPU, no process env.
//!   cargo test -p dismantle-serve --test system_kv_bank

use dismantle_serve::{BankConfig, SystemPromptKvBank};
use dismantle_serve::system_kv_bank::RecordOutcome;

fn sys_prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + i).collect()
}

#[test]
fn hash_is_stable_and_prefix_sensitive() {
    let toks = sys_prompt(40);
    // Same span, same length -> same hash.
    assert_eq!(
        SystemPromptKvBank::hash_prefix(&toks, 16),
        SystemPromptKvBank::hash_prefix(&toks, 16)
    );
    // Same tokens, different banked length -> different address.
    assert_ne!(
        SystemPromptKvBank::hash_prefix(&toks, 16),
        SystemPromptKvBank::hash_prefix(&toks, 24)
    );
    // A differing token inside the span changes the hash.
    let mut toks2 = toks.clone();
    toks2[5] = 99999;
    assert_ne!(
        SystemPromptKvBank::hash_prefix(&toks, 16),
        SystemPromptKvBank::hash_prefix(&toks2, 16)
    );
    // A token AFTER the span does NOT (only the leading `prefix_len` matter).
    let mut toks3 = toks.clone();
    toks3[20] = 88888;
    assert_eq!(
        SystemPromptKvBank::hash_prefix(&toks, 16),
        SystemPromptKvBank::hash_prefix(&toks3, 16)
    );
}

#[test]
fn record_then_hit_returns_source_slot() {
    let mut bank = SystemPromptKvBank::new();
    let system = sys_prompt(32);
    // Slot 3 prefilled a 32-tok system prompt.
    assert_eq!(bank.record(&system, 32, 3), RecordOutcome::Inserted);
    // A NEW request (system + user tail) probes the 32-tok leading span.
    let mut req = system.clone();
    req.extend([7u32, 8, 9]);
    let hit = bank.lookup(&req, 32).expect("banked system prefix must hit");
    assert_eq!(hit.source_slot, 3);
    assert_eq!(hit.prefix_len, 32);
    let s = bank.stats();
    assert_eq!(s.lookups, 1);
    assert_eq!(s.hits, 1);
}

#[test]
fn miss_on_different_system_prompt() {
    let mut bank = SystemPromptKvBank::new();
    bank.record(&sys_prompt(32), 32, 1);
    let other: Vec<u32> = (0..40u32).map(|i| 5000 + i).collect();
    assert!(bank.lookup(&other, 32).is_none());
    assert_eq!(bank.stats().hits, 0);
    assert_eq!(bank.stats().lookups, 1);
}

#[test]
fn never_matches_full_prompt() {
    // Strict-prefix rule: banked_len == prompt len must miss (decode loop
    // needs a real last_id), mirroring the RAM/disk prefix tiers.
    let mut bank = SystemPromptKvBank::new();
    let p = sys_prompt(16);
    bank.record(&p, 16, 0);
    assert!(bank.lookup(&p, 16).is_none(), "must not match whole prompt");
    let mut longer = p.clone();
    longer.push(42);
    assert!(bank.lookup(&longer, 16).is_some(), "strict prefix hits");
}

#[test]
fn too_short_is_rejected_both_ways() {
    let cfg = BankConfig { min_prefix_tokens: 8, max_entries: 64 };
    let mut bank = SystemPromptKvBank::with_config(cfg);
    let p = sys_prompt(20);
    // Recording a 4-tok prefix is rejected.
    assert_eq!(bank.record(&p, 4, 0), RecordOutcome::TooShort);
    assert_eq!(bank.len(), 0);
    // Looking up below the floor misses without touching entries.
    bank.record(&p, 12, 0);
    assert!(bank.lookup(&p, 4).is_none());
}

#[test]
fn refresh_updates_source_slot() {
    let mut bank = SystemPromptKvBank::new();
    let system = sys_prompt(24);
    assert_eq!(bank.record(&system, 24, 2), RecordOutcome::Inserted);
    // The same system prompt is later served from slot 5 -> refresh, not dup.
    assert_eq!(bank.record(&system, 24, 5), RecordOutcome::Updated);
    assert_eq!(bank.len(), 1);
    let mut req = system.clone();
    req.push(1);
    assert_eq!(bank.lookup(&req, 24).unwrap().source_slot, 5);
}

#[test]
fn lru_eviction_caps_entries_and_keeps_newest() {
    let cfg = BankConfig { min_prefix_tokens: 4, max_entries: 2 };
    let mut bank = SystemPromptKvBank::with_config(cfg);
    // Bank 4 distinct system prompts; cap is 2.
    for i in 0..4u32 {
        let p: Vec<u32> = (0..8u32).map(|j| i * 100 + j).collect();
        bank.record(&p, 8, i);
    }
    assert!(bank.len() <= 2, "entry cap enforced");
    assert!(bank.stats().evictions >= 2);
    // The newest two (i=2, i=3) survive; the oldest (i=0) is gone.
    let p0: Vec<u32> = (0..9u32).map(|j| 0 * 100 + j).collect(); // 8-tok span + tail
    assert!(bank.lookup(&p0, 8).is_none(), "oldest evicted");
    let p3: Vec<u32> = (0..9u32).map(|j| 3 * 100 + j).collect();
    assert_eq!(bank.lookup(&p3, 8).unwrap().source_slot, 3, "newest survives");
}

#[test]
fn lru_touch_on_hit_protects_entry() {
    let cfg = BankConfig { min_prefix_tokens: 4, max_entries: 2 };
    let mut bank = SystemPromptKvBank::with_config(cfg);
    let a: Vec<u32> = (0..8u32).map(|j| 10 + j).collect();
    let b: Vec<u32> = (0..8u32).map(|j| 20 + j).collect();
    bank.record(&a, 8, 0);
    bank.record(&b, 8, 1);
    // Touch `a` so it is now more-recently-used than `b`.
    let mut a_q = a.clone();
    a_q.push(1);
    assert!(bank.lookup(&a_q, 8).is_some());
    // Insert a 3rd -> `b` (now LRU) is evicted, `a` survives.
    let c: Vec<u32> = (0..8u32).map(|j| 30 + j).collect();
    bank.record(&c, 8, 2);
    assert!(bank.lookup(&a_q, 8).is_some(), "touched entry protected");
    let mut b_q = b.clone();
    b_q.push(1);
    assert!(bank.lookup(&b_q, 8).is_none(), "untouched LRU evicted");
}

#[test]
fn forget_slot_invalidates_its_entries() {
    let mut bank = SystemPromptKvBank::new();
    let p = sys_prompt(16);
    bank.record(&p, 16, 7);
    let mut q = p.clone();
    q.push(1);
    assert!(bank.lookup(&q, 16).is_some());
    assert_eq!(bank.forget_slot(7), 1);
    assert!(bank.lookup(&q, 16).is_none(), "forgotten slot no longer routable");
}
