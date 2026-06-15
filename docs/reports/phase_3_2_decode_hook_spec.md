# Phase 3.2 — forward_token_greedy_tcb decode hook (EXACT spec)

Stream router-files delivers the Router (backend/router.rs) + the mod decl
+ the two-leg parity test. The decode-path wiring is delivered as THIS
spec (not as qwen_dense edits) so the orchestrator applies it AFTER the
f16kv-dispatch stream lands, with zero conflict. Re-grep every anchor
before applying; line numbers are guidance, the verbatim text is the
contract.

Target: `crates/dismantle-core/src/model/qwen_dense.rs`, fn
`forward_token_greedy_tcb(&mut self, token: u32, pos: usize) -> Result<u32>`
(opens at ~:3437). `ctx: &MetalContext` is in scope; `h`, `head_dim`,
`n_heads`, `n_kv_heads`, `kv_dim`, `eps` are bound at ~:3629-3636;
`pos_u32`, `theta` are local in the per-layer loop.

## Composition proof (why this is conflict-free with f16kv)

The f16kv decode hook (sibling stream) lives in the MHA / KV-cache region
(~:4148-:4200: kv append + `mha_decode_f32_tcb` :4185 / `mha_decode_flash_f32_tcb`
:4170, plus `ensure_f16_kv` / `kv_f16_layer_byte_offset`). The two ops THIS
spec routes are strictly UPSTREAM of that region:
  - pre-loop layer-0 rmsnorm: ~:3811 (before the layer loop even starts),
  - Q/K rope: ~:4127 / ~:4137 (before KV append at :4148, before MHA).
Different, non-overlapping line ranges; rmsnorm/rope feed the K/V that
f16kv later caches. Apply order is irrelevant to correctness — apply this
AFTER f16kv as instructed and the edits do not touch the same text.

## Step 0 — read the lever once + build the Router once

The function already reads its default-off levers near the top; the flash
lever is the nearest sibling:

```rust
        let flash_attn = crate::env_on("DISMANTLE_QWEN_FLASH_ATTN");
```  (~:3770)

Immediately AFTER that line, add (verbatim insertion, no existing text
changed):

```rust
        // Phase 3.2: per-op CPU-fallback router. DISMANTLE_FORCE_CPU_OP is
        // a string lever (rmsnorm|rope), DEFAULT-UNSET. Unset ⇒ the router
        // forces nothing ⇒ every op below dispatches the identical Metal
        // kernel it did pre-router (golden hash unchanged). The router
        // owns a cheap Arc-backed MetalBackend over this ctx.
        let op_router = crate::backend::router::Router::from_env(
            crate::backend::metal::MetalBackend::new(ctx.clone()),
        );
```

NOTE for the orchestrator: `backend::metal` is currently `mod metal;`
(private) in backend/mod.rs. router-files' mod-decl edit adds
`#[cfg(target_os=\"macos\")] pub mod router;` and, to let the call site name
the backend, the SAME edit promotes metal to `#[cfg(target_os=\"macos\")]
pub mod metal;` (see the edits[] entry). If you prefer not to expose
`backend::metal`, an equivalent: have `Router::from_ctx(ctx.clone())` build
the MetalBackend internally — then the call site is
`Router::from_ctx(ctx.clone())` and metal can stay private. Either is fine;
the edits[] take the `pub mod metal` route because it is the smaller diff
and keeps Router free of a Metal-context constructor.

`ctx.clone()` is cheap (MetalContext is #[derive(Clone)], Arc-backed,
metal/mod.rs:345). Building the router does NOT open a command buffer; the
token's TCB is still the existing `let mut tcb = TokenCommandBuffer::new(ctx);`
at :3794 — leave that line exactly as is.

## Step 1 — route the pre-loop layer-0 rmsnorm (the flagship)

The orchestrator must hand the router the SAME `&mut MetalRecorder` the
rest of the token records into. The decode path currently records into a
bare `TokenCommandBuffer` named `tcb` (`let mut tcb = TokenCommandBuffer::new(ctx);`
:3794), NOT a `MetalRecorder`. There are two equivalent ways to bridge;
pick ONE and apply it consistently to BOTH the rmsnorm and rope sites:

  (Bridge α — recommended, smallest blast radius) Wrap the existing `tcb`
  in a `MetalRecorder` for the routed call only, then unwrap. Because
  `MetalRecorder{ tcb }` is a by-value newtype with `pub(crate) tcb` and
  the router swaps the field in place on a fallback, you must move `tcb`
  in and move it back out so subsequent raw `kernels::*_tcb(&mut tcb, ..)`
  calls see the (possibly fresh) TCB:

  Replace the rmsnorm dispatch at :3811-:3818, which currently reads
  EXACTLY:

```rust
        kernels::rmsnorm_metal_buf_tcb(
            &mut tcb,
            &arena.x_buf,
            layer0_attn_norm,
            eps,
            h,
            &arena.x_norm_buf,
        )?;
```

  with:

```rust
        {
            // Phase 3.2: route layer-0 pre-norm through the op router. With
            // the lever unset this is byte-for-byte the rmsnorm_metal_buf_tcb
            // dispatch above; with DISMANTLE_FORCE_CPU_OP=rmsnorm it flushes
            // `tcb`, runs kernels::rmsnorm on the shared buffers, and resumes
            // on a fresh TCB. Move `tcb` into the recorder and back so the
            // (possibly replaced) command buffer is the one the rest of the
            // token keeps recording into.
            let mut rec = crate::backend::metal::MetalRecorder { tcb };
            op_router.rmsnorm(
                &mut rec,
                &arena.x_buf,
                layer0_attn_norm,
                &arena.x_norm_buf,
                eps,
                h,
            )?;
            tcb = rec.tcb;
        }
```

  (`MetalRecorder.tcb` is `pub(crate)`; qwen_dense is in the same crate, so
  naming the field is legal. The struct field-init form requires `tcb` be
  reachable — it is, via `crate::backend::metal::MetalRecorder`. The
  mod-decl edit's `pub mod metal` makes the path nameable.)

This is the ONLY rmsnorm site to route in 3.2. Do NOT route the in-loop
FUSED `add_rmsnorm_fused_tcb` at :4272 / :4577 — its CPU fallback must also
fold the residual add (`x += attn_out` before the norm), which is a
follow-up; leave those two sites untouched.

## Step 2 — route the Q and K rope (proves a second, in-place op)

The Q rope at :4127-:4136 currently reads EXACTLY:

```rust
            kernels::rope_q_f32_inplace_tcb(
                &mut tcb,
                &arena.q_buf,
                n_heads,
                head_dim,
                0,
                head_dim,
                pos_u32,
                theta,
            )?;
```

Replace with (full-head ⇒ RopeLayout::Full; the router's CPU fallback
loops n_heads × rope_inplace, matching the GPU interleaved kernel):

```rust
            {
                let mut rec = crate::backend::metal::MetalRecorder { tcb };
                op_router.rope(
                    &mut rec,
                    &arena.q_buf,
                    crate::backend::RopeLayout::Full { n_heads, head_dim },
                    pos_u32,
                    theta,
                )?;
                tcb = rec.tcb;
            }
```

The K rope at :4137-:4146 currently reads EXACTLY:

```rust
            kernels::rope_q_f32_inplace_tcb(
                &mut tcb,
                &arena.k_token_buf,
                n_kv_heads,
                head_dim,
                0,
                head_dim,
                pos_u32,
                theta,
            )?;
```

Replace with:

```rust
            {
                let mut rec = crate::backend::metal::MetalRecorder { tcb };
                op_router.rope(
                    &mut rec,
                    &arena.k_token_buf,
                    crate::backend::RopeLayout::Full { n_heads: n_kv_heads, head_dim },
                    pos_u32,
                    theta,
                )?;
                tcb = rec.tcb;
            }
```

## Correctness / parity notes the orchestrator must hold

- UNSET path is bit-identical: `Router::supports` returns true for every
  op when forced is None, so each routed call is exactly
  `primary.rmsnorm/rope(...)` = the same `kernels::*_tcb` dispatch, and the
  flush/replace branch is dead. The single `let mut rec = ...; tcb = rec.tcb;`
  shuffle is a move of the same TCB value (no commit, no extra CB) ⇒ the
  command-buffer structure and dispatch count are unchanged ⇒ golden
  greedy-64 hash b480cc10faf9a8ec holds. Gate with integration_greedy_64
  AND cpu_fallback_parity Leg A after applying.
- FORCED path (DISMANTLE_FORCE_CPU_OP=rmsnorm|rope) is atol 1e-3 only and
  splits the per-token CB on each forced op (flush→CPU→fresh TCB). Expected
  and asserted by cpu_fallback_parity Leg B as correctness, never tps.
- Apply this spec AFTER the f16kv-dispatch stream. The edits here touch
  only :3811 (rmsnorm) and :4127/:4137 (rope); f16kv touches :4148+ (KV +
  MHA). No shared text. If f16kv shifts line numbers, re-anchor on the
  verbatim `kernels::rmsnorm_metal_buf_tcb(` / `kernels::rope_q_f32_inplace_tcb(`
  call blocks above (they are unique at these three sites).
- Bridge β (alternative, if the orchestrator prefers not to construct
  MetalRecorder inline at three sites): migrate the whole token to record
  through a single `let mut rec = op_router.recorder();` at :3794 (replacing
  `let mut tcb = TokenCommandBuffer::new(ctx);`) and change every raw
  `&mut tcb` in the function to `&mut rec.tcb`, then the routed calls take
  `&mut rec` directly and the commit at :4821 becomes `rec.commit_and_wait()?`.
  That is a larger diff (touches ~40 `&mut tcb` sites) but removes the
  per-site wrap/unwrap. Bridge α is recommended for 3.2 because it is
  surgical and keeps the f16kv composition trivially non-overlapping.
