# Q8 KV hunk-port plan — 2026-05-23

Source patch: `reports/patches/session_C_q8_kv_wiring.patch` (2610 lines, 17
files). M2 of the overnight 6h chain confirmed `git apply --3way` rejects 17
files on current main, so we port by hand. This document is the **read-only
plan**; no code is changed by this session.

## Headline

The patch is **much smaller in effective scope than its line count suggests**.
The kernel-level work (Metal shaders, low-level Rust wrappers, parity tests)
has already landed on main. What remains is a thin **wiring layer**: CLI flag
→ EngineConfig → DeepSeekV2 state field → 3 dispatcher branches in the hot
path, plus 2 small TCB-wrapper helpers to author.

## Current-main inventory (verified by grep)

Already present:
| Artifact | Location | Status |
|---|---|---|
| `mla_decode_kernel_q8kv` shader | `shaders/attn.metal:169` | ✅ landed |
| `kv_append_q8_0_f32` shader | `shaders/attn.metal:584` | ✅ landed |
| `mla_decode_q8kv_metal` Rust standalone | `kernels/mod.rs:1694` | ✅ landed |
| `kv_append_q8_0_f32_metal` Rust standalone | `kernels/mod.rs:1813` | ✅ landed |
| `quantize_q8_0` CPU helper | `quant/mod.rs:142` | ✅ landed |
| `mla_q8kv_microbench.rs` test | `tests/` | ✅ landed (was untracked-in-patch) |
| `q8_kv_parity.rs` test | `tests/` | ✅ landed (was untracked-in-patch) |

Missing on main (port targets):
| Artifact | Notes |
|---|---|
| `EngineConfig::q8_kv: bool` field | engine.rs ~line 29 (after memory_limit_mb) |
| `--q8-kv` CLI flag + plumbing | main.rs (6 hunks total) |
| `DeepSeekV2.q8_kv_enabled: bool` field | deepseek_v2.rs ~line 194 |
| `DeepSeekV2.mla_c_kv_q8_gpu: Vec<PinnedBuffer>` field | same |
| Allocator block (build()) | deepseek_v2.rs ~line 822 (post-MLA-mirror alloc) |
| `q8_kv_sync_prefix()` helper | deepseek_v2.rs (new method on impl) |
| **`mla_decode_q8kv_and_o_proj_arena_tcb`** TCB wrapper | kernels/mod.rs (NEW — model after the f32 sibling at line 4005) |
| **`kv_append_q8_0_f32_tcb`** TCB wrapper | kernels/mod.rs (NEW — model after `kv_append_f32_tcb`) |
| 3 dispatch-site branches (mla_decode ×2 + kv_append ×1) | deepseek_v2.rs hot path |
| 2 prefix-sync call sites | deepseek_v2.rs (after f32 mirror upload) |
| `shader_hash` bump | profiles/deepseek-v2-lite-q4.m3pro18.json |

## Port order (low-risk → high-risk)

The whole thing is one logical commit. **Don't split** — the field is dead
until the dispatcher branches consume it, and the dispatcher branches don't
compile without the TCB wrappers. But during local construction, stage in
this order for clean compile checkpoints.

### Step 1 — EngineConfig field
File: `crates/dismantle-core/src/engine.rs`
Change: add `pub q8_kv: bool` after `memory_limit_mb` and `q8_kv: false` to
`Default`. Pure data. Compiles standalone.

### Step 2 — Two new TCB wrappers in kernels/mod.rs
File: `crates/dismantle-core/src/kernels/mod.rs`
- `pub fn mla_decode_q8kv_and_o_proj_arena_tcb(...)` — copy the f32 sibling
  at line 4005 verbatim, swap `c_kv` param type comment to "Q8_0 packed
  bytes", change the dispatch kernel name to `"mla_decode_kernel_q8kv"`. The
  Metal kernel takes the same args (q, c_kv, k_pe, kv_b_proj, out + 7
  uniforms + same 3 threadgroup allocations). The o_proj GEMV tail is
  identical. ~50 lines.
- `pub fn kv_append_q8_0_f32_tcb(...)` — model after `kv_append_f32_tcb`
  (search for `fn kv_append_f32_tcb` in the file). The shader is
  `"kv_append_q8_0_f32"` (already in attn.metal:584). The destination
  buffer for c_kv is the Q8 buffer (row stride = `n_blocks * 34`), not the
  f32 buffer (row stride = `kv_lora_rank * 4`). ~40 lines.

Both compile standalone once Step 1 lands.

### Step 3 — DeepSeekV2 struct fields + allocator
File: `crates/dismantle-core/src/model/deepseek_v2.rs`
- Add `q8_kv_enabled: bool` and `mla_c_kv_q8_gpu: Vec<PinnedBuffer>` to the
  struct definition (~line 194 per patch).
- In `build()` after the f32 MLA-cache allocation block, add the cfg-gated
  Q8 buffer-allocation block from patch lines 1428–1455. Wrap macOS-only;
  non-macOS gets `vec![]`. Print the `[q8-kv] requested but disabled` line
  when the flag is set but conditions don't match.
- Add the two new fields to the struct constructor (~line 919 in patch).
- Compile gate: full crate must still build with `--q8-kv` un-plumbed.

### Step 4 — `q8_kv_sync_prefix` helper
File: same.
Lifted from patch lines 1471–1502 verbatim. Pure CPU-side quantize + buffer
upload loop. No callers yet — compiles as dead code on its own.

### Step 5 — Hot-path dispatcher branches (3 sites + 2 sync calls)
File: same.
- `mla_decode_and_o_proj_arena_tcb` → wrap in `if self.q8_kv_enabled &&
  !self.mla_c_kv_q8_gpu.is_empty() { q8 variant } else { f32 variant }`.
  Two call sites: patch lines 1519, 1585, 1624. (Three sites, not two —
  one is in `decode_streaming_into_tcb`, one in the batched-prefill path,
  one in the streaming-prefill-into-decode path.)
- `kv_append_f32_tcb` → same pattern at patch line 1655. One site.
- Add `self.q8_kv_sync_prefix(...)?` after each of the 2 f32-mirror upload
  blocks (patch lines 1511, 1577, 1616 — that's 3 sites despite my "2"
  comment above; verify by grepping the current file for the f32 upload
  block).

### Step 6 — CLI plumbing
File: `crates/dismantle/src/main.rs`
- `--q8-kv` flag on the `Cmd::Generate` enum variant (patch lines 2497–2505).
- Pass-through through `generate_main(...)` signature and body (3 hunks).
- Mirror the field into `EngineConfig { ..., q8_kv }`.

### Step 7 — Profile shader-hash bump
File: `profiles/deepseek-v2-lite-q4.m3pro18.json`
Bump `shader_hash` so the profile-gate validates. The patch's hash
(`05ac3c172932cfe7f6b0b327`) is stale; **compute the new hash** from the
current attn.metal content (use the same tool the gate uses — check
`profile.rs` for how it derives the hash). Do NOT hardcode the patch's
value.

## Files in the patch we IGNORE during port

Skip these — they have already converged or are noise:

| File | Why skip |
|---|---|
| `shaders/attn.metal` (+204 -0) | kernel + helper already landed |
| `quant/mod.rs` (+457 -20) | quantize_q8_0 + helpers already landed |
| `kernels/mod.rs` (+405 -115) | mla_decode_q8kv_metal + kv_append_q8_0_f32_metal already landed; the +405 reflects 14 months of incidental kernel additions in the patch's source branch. Only the 2 TCB wrappers (Step 2) are net new |
| `tests/phase1_kernel_parity.rs`, `v034_*`, `v1c_*`, `v1l_*` | tests already on main or removed |
| `tests/mla_q8kv_microbench.rs`, `q8_kv_parity.rs` | already on main as files |
| `cache/mod.rs`, `cache/prefill_disk.rs`, `attn/mod.rs`, `expert_cache.rs`, `kernel_bench.rs` | per-file +0 to +6 net; mostly delete-noise that current main already cleared |
| `docs/kernels.md` | cosmetic |

The "17 files diverge" count in M2 is structural — only **4 source files +
1 profile** carry actual port work: engine.rs, kernels/mod.rs, main.rs,
model/deepseek_v2.rs, and the profile JSON.

## Validation strategy (post-port)

In order, GATE on each. If any gate fails, hold the port and diagnose.

1. **Compile gate.** `cargo build -p dismantle-core --release` must pass
   with zero new warnings beyond what main already has.
2. **Test gate.** `cargo test -p dismantle-core --release q8_kv` must pass
   (`q8_kv_parity.rs` exists on main and is the canonical kernel-level
   parity check).
3. **CLI gate.** `cargo run --release -p dismantle -- generate --q8-kv
   --weights … --prompt "hello" --max-new-tokens 3` must print 3 tokens
   without panic and the `[q8-kv]` disabled-message must NOT fire on a
   normal Metal/MLA-ready model.
4. **Token-parity gate (greedy, bit-identical, 3 tok).** Same prompt,
   T=0.0, max-new=3, with and without `--q8-kv`. Token IDs must match.
   (Q8 noise diverges later; the per-token KV is fresh at the new slot, so
   the first 3 stay identical at greedy.)
5. **Kernel-name gate.** Run with `--trace-dispatch` (or whatever the
   current trace flag is) and confirm `mla_decode_kernel_q8kv` and
   `kv_append_q8_0_f32` appear in the dispatch log when `--q8-kv` is on,
   and `mla_decode_kernel` / `kv_append_f32` do NOT.
6. **Paired smoke (TRIALS=15+).** `tools/bench/microbench_levers.sh` (or
   equivalent paired-bench harness) with --q8-kv ON vs OFF, decode 64-tok,
   prompt "Once upon a time". Per memory `q8_kv_runtime_landed.md` the
   earlier sibling-worktree port saw +1.6 / +2.1 / +2.5 % at 16 / 64 / 256
   tok. Reproduce within σ and we're good.

If gates 1–5 pass but gate 6 is flat or negative, do **not** revert. Q8 KV
amortizes — its win grows with context length. Re-run at 256 and 1024 tok
before disposition.

## Open questions worth flagging to the user before commit

These need an explicit OK and shouldn't be guessed:

1. **Shader hash recomputation.** The profile JSON bump needs the current
   hash, not the patch's stale value. Confirm `profile.rs::compute_shader_hash`
   (or equivalent) is the right derivation.
2. **Default-off vs default-on.** Patch defaults `q8_kv: false`. That's
   safe. If the user wants `--q8-kv` to be the new default at any point,
   that's a separate change.
3. **Spec-decode interaction.** The patch's port does NOT touch the
   speculate code path. Spec-decode K≥1 paths likely call the same
   `mla_decode_and_o_proj_arena_tcb` site, so they'll branch naturally —
   but worth confirming by grep at port time.

## Estimated effort

- Reading the f32 sibling functions: 30 min
- Writing the 2 new TCB wrappers: 1 h
- DeepSeekV2 struct + allocator: 30 min
- 3 dispatcher branches + 2 sync calls: 45 min
- main.rs + engine.rs glue: 30 min
- Profile hash bump: 15 min
- Local validation gates 1–5: 1 h
- Paired bench gate 6: 1 h (TRIALS=15 in clean window)

**Total: ~5–6 h of single-session focused work.** Smaller than the
patch's 2610-line surface suggests; the heavy lifting already shipped.

## Why not just patch-port the unchanged hunks

The patch's +405 in `kernels/mod.rs` and +457 in `quant/mod.rs` are mostly
**stale context** — symbols and surrounding code that look "added" only
because the patch's base branch lagged main by months. Re-applying them
would risk *unwinding* later improvements. Stick to the wiring layer.
