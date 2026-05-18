# Phase 2 multi-layer hidden capture — design note for next session

**Status:** design only. Not implemented in current session due to scope.

## What to build

Multi-layer hidden state capture for EAGLE-3 multi-layer fusion training.
Capture hidden states at layers `{2, 14, 24}` of DeepSeek-V2-Lite's 27
transformer layers (per EAGLE-3 paper recipe; see `stage3_c1/architecture.md`
+ deep research Q7).

Paper-measured impact (Table 2): +15-20% wall speedup, +14-19% τ.

## Implementation sketch

### Files to touch

```
crates/dismantle-core/src/metal/decode_arena.rs   — new Vec<PinnedBuffer> for capture slots
crates/dismantle-core/src/model/deepseek_v2.rs    — layer loop blit + new trait method
crates/dismantle-core/src/engine.rs               — trait method declaration
crates/dismantle/src/main.rs                      — capture-hidden CLI: --capture-layers
tools/training/capture_hidden.py                  — DCAP v2 reader (Python)
tools/training/mlx_eagle/data.py                  — load N hiddens per record
tools/training/mlx_eagle/model.py                 — concat-then-linear N hiddens into head input
```

### DecodeArena addition

```rust
pub multi_layer_capture_buf: Vec<PinnedBuffer>,  // len = n_capture_layers
pub multi_layer_capture_indices: Vec<usize>,     // which layer each slot captures
```

Allocate via new `set_capture_layers(&mut self, ctx, layers: Vec<usize>)` method.

### Layer loop change

After each `encode_layer(li)`, if `li` is in `capture_indices`:
```rust
let slot = capture_indices.position(|&x| x == li).unwrap();
let dst = &arena.multi_layer_capture_buf[slot];
// dst = x_buf + ffn_out_buf  (the full post-layer-li residual)
copy_buffer_bytes(arena.x_buf, dst);
add_inplace_metal_tcb(tcb, dst, arena.ffn_out_buf, hidden);
```

This does NOT modify x_buf (the next layer continues normally — its
phase1 add_inplace will add ffn_out_buf into x_buf as designed).

### New trait method

```rust
fn forward_token_multi_hidden_for_test(
    &mut self,
    token: u32,
    pos: usize,
    capture_layers: &[usize],
) -> Result<Vec<Vec<f32>>>  // returns [layer_idx][hidden]
```

### DCAP v2 binary format

Header (16 bytes):
- bytes 0..4: magic = b"DCAP"
- bytes 4..8: version = 2 (was 1)
- bytes 8..12: hidden_dim
- bytes 12..14: n_hiddens_per_record (u16)
- bytes 14..16: padding (u16)

Records (per token):
- u16 id_len + utf8 id
- u32 pos + u32 prev_token + u32 next_token
- N × hidden_dim × 2 bytes (N hiddens packed f16)

For N=3 layers, V2-Lite hidden=2048: each record is ~24 KB hidden + ~14 B
metadata = ~24 KB. 100K samples × ~120 tokens × 24 KB = ~280 GB total.

That's huge. Compression via zstd parquet should bring this to ~150 GB.

Alternative: capture fp8 hiddens instead of fp16. Half the size at modest
quality loss. Or capture only 2 layers (mid + final) instead of 3. Or
sample every-other-token instead of every token.

### Sanity check before full capture

Implement a 10-sample smoke that:
1. Runs the new multi-hidden capture on 10 known prompts
2. Reads back the binary
3. Verifies: shape correct, no NaN/Inf, hidden norms in expected range
4. Confirms: layer 24's hidden ≈ pre-final-norm state (compare to existing
   forward_token_with_hidden_for_test output)

### Recommended next-session order

1. Implement arena change (~30 min)
2. Implement layer loop blit (~1-2 hr — most error-prone)
3. Implement trait method + read-back (~30 min)
4. Test against synthetic 10-sample (~30 min)
5. DCAP v2 binary format + Python reader (~1 hr)
6. End-to-end smoke (~30 min)
7. Re-prep capture and restart

Total: ~4-6 hours focused. Save until rested.

## What NOT to do

- Don't try to optimize binary format on first pass (zstd parquet at conversion time is enough)
- Don't add fp8 storage initially — fp16 first
- Don't refactor existing forward paths beyond what the blit requires
- Don't change the Wedge M C-1 or batched_tcb paths in this same pass — they're alternate forward paths we don't use for capture
