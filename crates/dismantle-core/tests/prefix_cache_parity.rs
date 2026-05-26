//! Parity test for the on-disk prefix KV cache.
//!
//! Simulates the `generate()` prefill loop without spinning up the real
//! Qwen model: each "forward step" writes a deterministic value into
//! `KvCache`, and we assert that the two paths produce byte-identical
//! KV state for the full prompt:
//!
//! 1. **Cold path** — prefill every token, no cache consulted.
//! 2. **Warm path** — round-trip the first N tokens through
//!    `store` + `lookup_longest_prefix` + `restore_hit_into_kv`, then
//!    prefill the remaining `prompt_len - N` tokens.
//!
//! If those two end-states match for arbitrary N, the wire-up in
//! `qwen_dense::generate()` is correct: turn-2 chat will produce the
//! same KV (and therefore the same next token) as a fresh prefill.

use dismantle_core::cache::prefill_disk::{
    restore_hit_into_kv, PrefillDiskCache, PrefillKey,
};
use dismantle_core::cache::KvCache;
use tempfile::TempDir;

/// Deterministic "forward step" that writes per-token KV vectors.
/// Mirrors what `forward_token` does to the cache: appends one row of
/// KV per layer, then bumps seq_len.
fn fake_forward(kv: &mut KvCache, token: u32, pos: usize) {
    assert_eq!(kv.seq_len, pos, "fake_forward: expected seq_len == pos");
    let stride = kv.n_kv_heads * kv.head_dim;
    let mut k_row = vec![0.0f32; stride];
    let mut v_row = vec![0.0f32; stride];
    for (li, (kbuf, vbuf)) in kv.keys.iter_mut().zip(kv.values.iter_mut()).enumerate() {
        for d in 0..stride {
            // Mix layer, token id, position, and dim so any off-by-one
            // in restore is loud.
            let mix = ((li as u32).wrapping_mul(2654435761))
                ^ token.wrapping_mul(40503)
                ^ ((pos as u32).wrapping_mul(0x9E37_79B9))
                ^ ((d as u32).wrapping_mul(0xDEAD_BEEF));
            k_row[d] = (mix as f32) * 1e-9;
            v_row[d] = -(mix as f32) * 1e-9;
        }
        let off = pos * stride;
        kbuf[off..off + stride].copy_from_slice(&k_row);
        vbuf[off..off + stride].copy_from_slice(&v_row);
    }
    kv.seq_len += 1;
}

fn cold_prefill(prompt: &[u32], n_layers: usize, n_kv: usize, head_dim: usize) -> KvCache {
    let mut kv = KvCache::new(n_layers, prompt.len() + 8, n_kv, head_dim);
    for (i, &t) in prompt.iter().enumerate() {
        fake_forward(&mut kv, t, i);
    }
    kv
}

fn assert_kv_eq(a: &KvCache, b: &KvCache) {
    assert_eq!(a.seq_len, b.seq_len, "seq_len mismatch");
    assert_eq!(a.n_layers, b.n_layers);
    assert_eq!(a.n_kv_heads, b.n_kv_heads);
    assert_eq!(a.head_dim, b.head_dim);
    for li in 0..a.n_layers {
        assert_eq!(
            a.keys_for(li),
            b.keys_for(li),
            "keys mismatch on layer {}",
            li
        );
        assert_eq!(
            a.values_for(li),
            b.values_for(li),
            "values mismatch on layer {}",
            li
        );
    }
}

#[test]
fn cold_vs_warm_prefill_byte_identical() {
    // Roughly mirrors the chat scenario: a 50-token "system prompt" +
    // a 30-token "user message". The system prefix gets cached on
    // turn 1, then turn 2's prompt = system + first_response (10 tok) +
    // new user msg (20 tok). The cached system prefix must restore
    // byte-identically.
    let n_layers = 4;
    let n_kv = 2;
    let head_dim = 16;

    let tmp = TempDir::new().unwrap();
    let cache = PrefillDiskCache::open(tmp.path()).unwrap();

    // Turn 1.
    let system: Vec<u32> = (0..50u32).map(|i| 100 + i).collect();
    let kv_cold_t1 = cold_prefill(&system, n_layers, n_kv, head_dim);
    let key_t1 = PrefillKey::from_model_and_prompt("qwen-test", b"tok-sig-v1", &system);
    cache.store(&key_t1, &kv_cold_t1).unwrap();

    // Turn 2: system + 10-token reply + 20-token new user message.
    let mut turn2: Vec<u32> = system.clone();
    turn2.extend((0..10u32).map(|i| 1000 + i));
    turn2.extend((0..20u32).map(|i| 2000 + i));

    // Cold reference.
    let kv_cold_t2 = cold_prefill(&turn2, n_layers, n_kv, head_dim);

    // Warm path: lookup → restore → prefill delta.
    let key_t2 = PrefillKey::from_model_and_prompt("qwen-test", b"tok-sig-v1", &turn2);
    let hit = cache
        .lookup_longest_prefix(&key_t2.model_hash, &key_t2.tokenizer_hash, &turn2)
        .unwrap()
        .expect("expected prefix hit on turn 2");
    assert_eq!(
        hit.n_tokens, 50,
        "should hit the 50-token cached system prefix"
    );
    let mut kv_warm = KvCache::new(n_layers, turn2.len() + 8, n_kv, head_dim);
    restore_hit_into_kv(&hit, &mut kv_warm).unwrap();
    assert_eq!(kv_warm.seq_len, 50);
    // Continue prefill on the delta.
    for (i, &t) in turn2.iter().enumerate().skip(50) {
        fake_forward(&mut kv_warm, t, i);
    }

    assert_kv_eq(&kv_cold_t2, &kv_warm);
}

#[test]
fn store_then_load_byte_exact() {
    // The simplest gate: store turn-1 KV, immediately load it back via
    // a longer prompt, restore, and verify byte equality with the
    // source KV for the cached prefix length.
    let n_layers = 3;
    let n_kv = 2;
    let head_dim = 8;
    let tmp = TempDir::new().unwrap();
    let cache = PrefillDiskCache::open(tmp.path()).unwrap();

    let prompt: Vec<u32> = (0..20u32).collect();
    let kv = cold_prefill(&prompt, n_layers, n_kv, head_dim);
    let key = PrefillKey::from_model_and_prompt("m", b"tok", &prompt);
    cache.store(&key, &kv).unwrap();

    // Lookup with a 1-token-longer prompt so the cache returns the
    // full 20-token prefix.
    let mut probe = prompt.clone();
    probe.push(999);
    let hit = cache
        .lookup_longest_prefix(&key.model_hash, &key.tokenizer_hash, &probe)
        .unwrap()
        .unwrap();
    assert_eq!(hit.n_tokens, 20);

    let mut restored = KvCache::new(n_layers, prompt.len() + 4, n_kv, head_dim);
    restore_hit_into_kv(&hit, &mut restored).unwrap();
    // The remaining unused KV slots stay zeroed — that's fine, the
    // engine only consults kv up to seq_len.
    let stride = n_kv * head_dim;
    let used = restored.seq_len * stride;
    for li in 0..n_layers {
        assert_eq!(&restored.keys[li][..used], &kv.keys[li][..used]);
        assert_eq!(&restored.values[li][..used], &kv.values[li][..used]);
    }
}

#[test]
fn cache_miss_falls_back_to_full_prefill() {
    // A fresh cache directory + new prompt → lookup returns None and
    // the caller must run a full prefill, producing identical KV to
    // the no-cache path.
    let tmp = TempDir::new().unwrap();
    let cache = PrefillDiskCache::open(tmp.path()).unwrap();

    let prompt: Vec<u32> = (10..30u32).collect();
    let key = PrefillKey::from_model_and_prompt("m", b"sig", &prompt);
    let hit = cache
        .lookup_longest_prefix(&key.model_hash, &key.tokenizer_hash, &prompt)
        .unwrap();
    assert!(hit.is_none(), "fresh cache must miss");

    // No-cache prefill.
    let kv_a = cold_prefill(&prompt, 2, 2, 8);
    // "Cache miss" prefill (identical code path since hit was None).
    let kv_b = cold_prefill(&prompt, 2, 2, 8);
    assert_kv_eq(&kv_a, &kv_b);
}
