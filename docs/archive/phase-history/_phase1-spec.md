# Phase 1 — Spec

**Status:** locked.
**Adopted:** 2026-04-28.
**Predecessor:** Phase 0 closed (CPU reference path, 0.30 tok/s decode,
[docs/competitive_audit.md](docs/competitive_audit.md)).
**Goal:** lift dismantle off CPU GEMV and onto Metal for the largest
single-op surfaces. Naive Metal GEMV — no fused dequant yet (that's
Phase 1 / Wedge 2 in ROADMAP), no single-launch MoE (Wedge 1, Phase
2). Just real Metal dispatch where today the model layer calls
`gemv_f32` on host slices.

## Locked decision rules

These are *not* up for debate during a haul. Reopening them requires
an attended session that updates this spec.

1. **Numerical correctness is mandatory; performance is observed.**
   Every Metal kernel landed must match its CPU reference within
   `atol=1e-3` fp16 on a fixed input. A correct-but-slower kernel
   passes the gate. (Perf wedges are subsequent hauls.)
2. **Gates run linearly.** No reordering, no parallelism within a
   haul. The order in this doc is the order.
3. **No deferral.** If a gate fails, write a blocked doc — do not
   "I'll come back to it" or "loosen the tolerance". The gate
   either passes or halts.
4. **No peel-onion fixing.** If gate G1.2 fails because of a bug in
   `kernels::gemv_f32`, do not refactor it inside G1.2's haul. Halt,
   write the blocked doc, the next attended session adopts the fix
   into a bundled patch. Phase 0 taught us that mid-haul "while I'm
   here" patches blow scope.
5. **Hybrid halt threshold:**
   - **G1.1 (Metal scaffold): 1 halt = end haul.** Every later gate
     depends on it.
   - **G1.2 / G1.3 / G1.4 (GEMV ports): 2 halts in this group =
     end haul.** First port-failure: write blocked doc, continue
     to next *independent* item. Second port-failure: end haul,
     write closeout.
6. **One infrastructure retry per haul.** Permitted: clean Metal
   pipeline cache + rebuild release binary, OR clear `~/.cache/dismantle/`
   if it exists. Not permitted: invasive cargo cache wipe, OS
   reboot, model-file re-download. After one infrastructure retry,
   the next halt ends the haul.
7. **4 hr hard ceiling per haul.** Per-gate soft 60 min (45 impl +
   15 validation). On hard ceiling, halt with `reason: ceiling`.

## The four gates

```
G1.1   Metal scaffold + rmsnorm round-trip
G1.2   LM-head GEMV → Metal
G1.3   attention o_proj GEMV → Metal
G1.4   ffn_gate_inp GEMV → Metal
```

### G1.1 — Metal scaffold + rmsnorm round-trip

**Goal:** prove `MetalContext` can dispatch a real kernel from Rust
and produce numerically correct output. Foundation for all later
gates; if this fails no GEMV port can validate.

**Surface:** the existing `rmsnorm` kernel in `shaders/common.metal`
(already a real MSL kernel, not a stub).

**PASS criteria:** `cargo test --test phase1_kernel_parity test_rmsnorm_matches_cpu`
returns 0; CPU reference and Metal output diff `< atol=1e-3` fp16
across a 4096-element fixed input.

**FAIL → HALT.** No item 2–4 attempts.

### G1.2 — LM-head GEMV → Metal

**Goal:** replace the LM-head GEMV in `model/deepseek_v2.rs` with a
Metal dispatch. This is the biggest single CPU op per token: vocab
size 102400 × hidden 2048 = ~210M MACs *per token*.

**Surface:** new `gemv_f16` kernel in `shaders/common.metal`; host
binding in `kernels/`; call site in `forward_token`'s LM-head
branch.

**PASS criteria:** `dismantle generate ... --max-new-tokens 5 --temperature 0`
produces same first 5 token IDs as the locked-baseline at
`_phase1_token_baseline.hashes`. (Token IDs, not text — exact byte
match.)

**FAIL** (token IDs differ): write blocked doc, count toward GEMV
group's 2-halt budget.

### G1.3 — attention o_proj GEMV → Metal

**Goal:** replace `o_proj` GEMV in MLA attention's tail. Per-token
op of size hidden 2048 × (n_heads × v_head_dim) = 2048 × 2048 = 4M
MACs.

**Surface:** new `gemv_f32` (or reuse fp16) in `shaders/attn.metal`;
call site at `attention()`'s tail.

**PASS criteria:** smoke generation produces non-empty UTF-8 of ≥1
char; first 5 token IDs match baseline.

**FAIL → blocked doc; count toward 2-halt budget.**

### G1.4 — ffn_gate_inp GEMV → Metal

**Goal:** replace the gate-logit GEMV in MoE FFN: shape `n_routed_experts × hidden`
= 64 × 2048 = 131k MACs. Tiny but exercises MoE-shaped tensors so the
kernel-pack proof-point lands.

**Surface:** new GEMV in `shaders/moe.metal`; call site in
`ffn()`'s MoE branch.

**PASS criteria:** smoke generation produces non-empty UTF-8;
**routing decisions match** for first 5 tokens (i.e., the same
top-K experts are picked at temp=0). Routing change implies gate
logits are wrong.

**FAIL → blocked doc; count toward 2-halt budget.**

## Cross-gate invariants (checked between gates)

The agent checks these *between* each gate. Any failure halts the
haul.

- `cargo test --workspace --lib` — all 15 prior tests still pass.
- `./target/release/dismantle generate --weights ./models/deepseek-v2-lite-q4.gguf --prompt "Once upon a time" --max-new-tokens 8 --temperature 0 --max-stall-ms 60000` — exits 0 with non-empty stdout.
- `tests/correctness/phase1_kernel_parity.rs` — every previously-passing parity test still passes.
- `_phase1_kernel_baseline.hashes` — every entry that was already there still hashes the same.

## Per-gate budget

```
G1.1: 60 min total (45 impl + 15 validation)
G1.2: 60 min total
G1.3: 60 min total
G1.4: 60 min total
total soft: 240 min
hard ceiling: 240 min × ~1× retry headroom ≈ 4 hr
```

If the haul exceeds 4 hr, the agent writes a blocked doc with
`reason: ceiling` and halts cleanly even if mid-item.

## Memory rule

Per CLAUDE.md: no process > 8 GB resident. If a Metal kernel test
or a generate call peaks above 8 GB (sample via `ps -o rss=`),
that's a regression and a halt condition. Phase 0 baseline is ~2 GB
resident (lazy expert dequant) — anything 4× that means we
accidentally re-introduced eager dequant somewhere.

## Co-existence mode (memory-aware)

dismantle hauls may run **concurrent with another GPU/RAM-heavy
process** (e.g., slm training). M3 Pro 18 GB has unified memory:
GPU and CPU share one pool. The haul protocol when co-existing:

### CE-1. Pre-flight memory probe

Before the haul starts, `tools/haul/coexist.sh probe` reads:
- `vm_stat` for free + speculative + inactive pages
- macOS `memory_pressure` for system-wide pressure (Normal / Warn / Critical)

Returns 0 (safe), 1 (degraded), 2 (critical). Haul aborts before
starting if pressure is Critical. If pressure is Warn, haul starts
but with reduced concurrency (single Metal command queue, no
parallel pipeline state compilation).

### CE-2. Per-item gate

Before each Item N, the agent calls `coexist.sh probe`:
- If 0: proceed.
- If 1 (degraded): sleep 30s, retry up to 5 times. If still
  degraded after 2.5 min, mark item as "deferred for memory" and
  continue to next item; counts as a halt only at end-of-haul if
  no item ever completed.
- If 2 (critical): wait 5 min for recovery; if still critical,
  halt the haul cleanly with `reason: memory_pressure_critical`.

### CE-3. Resource hygiene

All dismantle subprocesses inside the haul are launched with:

```
nice -n 19 taskpolicy -b  ./target/release/dismantle ...
```

This gives slm (or whatever the heavy process is) first dibs on
CPU and marks dismantle as "background" QoS. macOS's resource
arbiter then deprioritizes our reads/writes/Metal queues vs the
foreground job. Trade: dismantle gets slower; nothing crashes.

### CE-4. Synthetic-first validation

Where possible, gates validate against **synthetic kernel inputs**
(small fixed arrays, no model load). Integration smoke (model load
+ generate) is the *secondary* validation, gated on `coexist.sh probe`
returning 0. Specifically:

- **Always-run synthetic parity tests** (in `tests/correctness/`):
  - `phase1_kernel_parity::test_rmsnorm_matches_cpu` — 4096-element
    fixed input, no model.
  - `phase1_kernel_parity::test_gemv_f16_matches_cpu` — 1024×1024
    f16 matrix, fixed-seed input. No model.
- **Conditionally-run integration smoke** (in
  `tests/correctness/phase1_token_regression.rs`): loads the
  9 GB model + 3-token greedy generation. Skipped (with
  `#[ignore]` plus runtime `coexist.sh probe` check) if probe
  returns ≥ 1.

A gate with passing synthetic parity but skipped integration smoke
is recorded as `PASS-PARITY-ONLY`. Phase 1 closes only when at
least one haul has both passed; subsequent hauls (after memory
pressure drops) verify integration.

### CE-5. RSS sentinel

A background watchdog (~1 sec sample) checks the dismantle process's
RSS. If RSS exceeds **5 GB** at any moment, the watchdog kills the
process and halts that item with `reason: rss_ceiling`. Phase 0
baseline is ~2 GB; 5 GB is a generous 2.5× headroom that catches
silent eager-dequant regressions.

### CE-6. Inter-item cool-down

30 sec sleep between items lets the macOS scheduler rebalance pages
and gives slm training a clean breathing window. Negligible cost
per haul (4 items × 30s = 2 min) for substantial robustness.

### CE-7. Reduced validation token count

Token-baseline regression validates **3 tokens** in co-existence
mode (not 5). Cuts the per-validation cost by ~40% on the slow
Phase-0 CPU path, with no loss of regression sensitivity (the model
is deterministic at temp=0; if 3 tokens match then the forward
pass is consistent).

## Out of scope (do not attempt this haul)

- **Wedge 2 — fused Q4_K_M dequant.** Naive Metal GEMV here uses
  the existing `dequant_to_f32` materialization on a per-call basis.
  Slower than llama.cpp; that's the next haul's win.
- **Wedge 1 — single-launch MoE.** Per-expert dispatch is fine.
- **Attention beyond `o_proj`.** No flash-attention, no fused
  softmax-attention, no MLA decompress kernel. Phase 3 work.
- **GPU sampling.** CPU sampling stays.
- **GGUF re-mmap on init.** Lazy dequant continues.

## What "Phase 1 closed" means

Phase 1 is closed when:
- All 4 gates G1.1–G1.4 are PASS.
- `_phase1_closeout.md` is written.
- `cargo test --workspace --lib` passes both legacy 15 and new
  parity tests.
- Smoke generation still produces coherent text.
- Token-output baseline still matches.

Phase 1 not closed → Phase 2 prep does not start.
