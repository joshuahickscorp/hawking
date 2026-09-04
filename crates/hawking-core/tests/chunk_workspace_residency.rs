//! Residency under a chunked prefill, which `CP5C_RESULT.md` lists as untested.
//!
//! It is the one item on that list settleable by arithmetic rather than a
//! harness, because the workspace accounting already separates what scales with
//! a chunk from what does not.

use hawking_core::model::qwen38_hybrid_decode::{
    qwen38_chunk_workspace_bytes, qwen38_workspace_bytes,
};

#[test]
fn a_chunk_of_one_matches_the_per_token_path() {
    let base = qwen38_workspace_bytes(8192).expect("base");
    let one = qwen38_chunk_workspace_bytes(8192, 1).expect("chunk 1");
    assert_eq!(
        one.total_bytes, base.total_bytes,
        "chunk=1 must allocate exactly what the per-token path does; it does not, \
so the split between scaling and non-scaling classes is wrong"
    );
}

#[test]
fn only_the_activations_scale() {
    let k4 = qwen38_chunk_workspace_bytes(8192, 4).expect("k4");
    let k128 = qwen38_chunk_workspace_bytes(8192, 128).expect("k128");
    // The two classes a chunk cannot change.
    assert_eq!(k4.deltanet_state_bytes, k128.deltanet_state_bytes);
    assert_eq!(k4.gqa_kv_bytes, k128.gqa_kv_bytes);
    assert_eq!(k4.terminal_bytes, k128.terminal_bytes);
    // And the one it does.
    assert_eq!(k128.scaled_activation_bytes, 32 * k4.scaled_activation_bytes);
}

#[test]
fn the_terminal_head_is_not_multiplied_by_the_chunk() {
    // logits is 248,320 f32 -- 993 KB, the majority of the 1.69 MB activation
    // total. Prefill needs it for the LAST position of a chunk only, so scaling
    // it would overstate the requirement by more than half.
    let k = qwen38_chunk_workspace_bytes(8192, 64).expect("k64");
    assert!(
        k.terminal_bytes > 900_000,
        "the terminal head should dominate a single position's activations; got {}",
        k.terminal_bytes
    );
    assert!(
        k.per_position_bytes < k.terminal_bytes,
        "per-position bytes ({}) should be SMALLER than the terminal head ({}) \
once logits are excluded; if not, the exclusion did not happen",
        k.per_position_bytes,
        k.terminal_bytes
    );
}

#[test]
fn residency_does_not_constrain_the_chunk_size() {
    // The measured resident weight authority is 10,554,259,456 B on a 96 GiB
    // machine. If a chunk's workspace growth were a meaningful fraction of that,
    // chunk size would be a residency decision rather than a kernel one.
    const RESIDENT_WEIGHT_BYTES: usize = 10_554_259_456;
    let base = qwen38_chunk_workspace_bytes(8192, 1).expect("k1");
    for chunk in [4usize, 8, 16, 32, 64, 128] {
        let w = qwen38_chunk_workspace_bytes(8192, chunk).expect("chunk");
        let growth = w.total_bytes - base.total_bytes;
        assert!(
            growth * 100 < RESIDENT_WEIGHT_BYTES,
            "chunk {chunk} grows the workspace by {growth} B, over 1% of the \
resident weights -- at that point chunk size becomes a residency decision"
        );
    }
}

// ---------------------------------------------------------------------------
// The ALLOCATOR, not just the arithmetic. `qwen38_chunk_workspace_bytes` above
// computes what a chunked workspace would need; these pin that
// `Qwen38HybridWorkspace::allocate_chunked` actually allocates it.
//
// A private allocator with no caller is the disease this repo keeps
// rediscovering -- registration is not reachability -- so the probe that makes
// these tests possible is itself the call site.

#[cfg(target_os = "macos")]
mod device_allocation {
    use hawking_core::model::qwen38_hybrid_decode::qwen38_probe_chunk_workspace_bytes;

    #[test]
    fn chunk_of_one_allocates_exactly_what_the_per_token_path_does() {
        let (base, chunked) = match qwen38_probe_chunk_workspace_bytes(2048, 1) {
            Ok(v) => v,
            // No Metal device in this environment is not a test failure.
            Err(_) => return,
        };
        assert_eq!(
            base, chunked,
            "chunk=1 allocated {chunked} against the per-token path's {base}; a foundation that \
quietly differs at K=1 makes every later comparison suspect"
        );
    }

    #[test]
    fn growth_is_bounded_and_only_the_activations_move() {
        let (base, k1) = match qwen38_probe_chunk_workspace_bytes(2048, 1) {
            Ok(v) => v,
            Err(_) => return,
        };
        let (_, k4) = match qwen38_probe_chunk_workspace_bytes(2048, 4) {
            Ok(v) => v,
            Err(_) => return,
        };
        assert_eq!(base, k1);
        assert!(k4 > k1, "chunk=4 must allocate more than chunk=1");
        // Per-position activations are ~0.7 MB once the terminal head is
        // excluded, against a workspace dominated by the 268 MB KV cache and
        // 157 MB of per-layer state. Tripling the per-position part must not
        // come close to doubling the whole.
        assert!(
            k4 < k1 * 2,
            "chunk=4 more than doubled the workspace ({k1} -> {k4}); either the \
per-layer state or the KV cache is being scaled with the chunk, and neither should be"
        );
    }
}
