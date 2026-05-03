# Phase 2 — Spec

**Status:** locked.
**Adopted:** 2026-04-29.
**Predecessor:** Phase 1 closed by haul 3 attempt 4 — 8 Metal kernels
parity-attested, full forward path Metal-resident, 50-prompt token
baseline locked at `_phase1_token_baseline_50.hashes`. Phase-1 perf
gate (B4) deferred to Phase 2 Wedge 3 because post-A1 dec_tps ~0.13
makes the ratio assertion meaningless until pinning + MLA migration
land.
**Goal:** close the ROADMAP Phase-2 acceptance gate:

> ≥2× prefill tok/s on 1024-token prompts vs. Phase 1; no decode
> regression. Decode tok/s ≥0.9× MLX on the same model.

Three speed wedges, ranked by impact ÷ cost (see
`_phase2_speed_followups.md`):

| Wedge | Item | Est. decode uplift | Est. attended impl | Phase-2 haul |
|------:|------|-------------------:|-------------------:|--------------|
| 1B    | MLA / Q-LoRA gemvs onto Metal | ~1.3× | landed 2026-04-29 | Haul 1 |
| 1A    | weight-pinning (persistent device-side buffers) | ~1.4× | ~1 day | Haul 2 (with W2) |
| 2     | FlashDMoE batched expert dispatch | ~15-30× | ~3-5 days | Haul 2 |
| 3     | perf-gate attestation (already tooled in haul 3) | n/a | 0 | bundled with Haul 1 + Haul 2 |

W1A was originally scoped for Haul 1 alongside W1B but pulled — the
1-day attended surgery (Buffer fields on Layer/DeepSeekV2 plus
five new `*_pinned` kernel entry points) doesn't fit a single
session; it ships with Haul 2 alongside W2, which is also days of
attended work.

Phase-2 closes when both decode-tps gates and the prefill gate hold.

## Locked decision rules

These do not change during a haul. Reopening requires an attended
session that updates this spec.

1. **Numerical correctness is mandatory; performance is observed.**
   Every Metal kernel landed must match its CPU reference within
   `atol=1e-3` fp16 on a fixed input, AND every haul must end with
   `dismantle-token-regression _phase1_token_baseline_50.hashes`
   green (50/50). The perf gate (W3) is the *separate* claim — its
   failure means "Phase 2 not closed yet," not "the haul broke."
2. **Gates run linearly.** No reordering, no parallelism within a
   haul. Manifest order is execution order.
3. **No deferral, no peel-onion fixing.** If a gate fails, halt and
   write a blocked doc — do not loosen tolerance, do not "while I'm
   here" patch the underlying bug.
4. **Per-layer halt budget, 1-halt-ends on impl.** Each W1A / W1B / W2
   layer is a small focused diff; any internal failure is the
   finding. Audit stays record-and-continue. Closeout always runs.
5. **4 hr hard ceiling per haul.** Per-gate soft 60 min. The Phase-1
   waiver (haul 3 ran overnight) does not carry over — Phase 2
   hauls fit in 4 hr by design.
6. **One infrastructure retry per haul.** Permitted: clean Metal
   pipeline cache + rebuild release binary, OR clear
   `~/.cache/dismantle/` if it exists. After one infrastructure
   retry, the next halt ends the haul.

## Halt-budget framework

Per-layer rules, codified in `tools/haul/run-gates.sh`'s case block:

| Layer prefix | Rule |
|--------------|------|
| `pre-flight` | 1 halt = end haul (everything depends on it) |
| `impl-W*`    | 1 halt within layer = end haul (focused-diff invariant) |
| `audit`      | record-and-continue (drift logged, never ends haul) |
| `closeout`   | always runs |

The W3 perf-gate sub-layer is the exception: bench gates
(`bench-decode`) get one continue-on-halt; only the
`perf-ratio-assert` failure halts the haul (with
`reason: phase2_perf_below_threshold`). Pattern matches haul 3's
`impl-B4`.

## Memory rule

Per CLAUDE.md: no dismantle process > 8 GB resident under normal
operation. Phase-1 baseline ~3 GB; weight-pinning (W1A) is *expected*
to grow this to ~9 GB resident. The 5 GB RSS sentinel from Phase 1
gets bumped to **10 GB for Phase 2** because pinning intentionally
holds the working set resident — that's the whole point. Anything
above 10 GB is a leak.

## Co-existence mode (Phase 2)

slm finished pre-haul-3; Phase 2 hauls run with no foreground GPU
contender. The full Phase-1 co-existence machinery (CE-1 to CE-7)
becomes:

- **CE-1 pre-flight probe**: still runs; expected return = 0 (safe).
  Halt only on critical (2). With no slm, critical = real OOM
  pressure from something else, which is a halt regardless.
- **CE-2 per-item gate**: probe still runs but the degraded-retry
  path is unreachable in practice. Keep it for resilience; cost is
  a sub-second probe per gate.
- **CE-3 resource hygiene**: `nice -n 19 taskpolicy -b` was *already
  stripped* in haul 3 because slm-coexistence wasn't needed.
  Phase 2 hauls run dismantle at default QoS (foreground, full
  P-core access). This is locked.
- **CE-4 synthetic-first**: all impl-layer parity tests are
  synthetic by design. Token regression is the integration smoke;
  it always runs. No `PASS-PARITY-ONLY` recording in Phase 2.
- **CE-5 RSS sentinel**: 10 GB ceiling per memory rule above.
- **CE-6 inter-item cool-down**: `HAUL_COOLDOWN_S=0` is the Phase-2
  default. The 30s sleep was for slm rebalance; without slm it's
  pure waste.
- **CE-7 reduced validation**: 3-token regression stays as the full
  attestation. The audit's "1-token at impl-layer end, full 3-token
  at AU5" optimization is a Phase-2 *opportunity* — not adopted
  this spec because the wall-clock saving (~36 min) doesn't justify
  the lost signal (argmax flip at token 2/3 is a real failure mode
  per haul 3 attempt 1's p004 divergence).

## Process deltas inherited from the audit

The audit (chat session 2026-04-29 leading to this spec) identified
four process changes that ship with Phase 2:

1. **`HAUL_COOLDOWN_S=0` default** (codified above as CE-6).
2. **Drop AU1 `verify-evidence phase2`** — pre-flight already
   attests. Audit retains AU2/AU3/AU4/AU5.
3. **Pre-flight uses `cargo build --release --workspace --tests`**
   instead of plain `cargo-build`. Compiles all integration test
   targets once so each later `cargo test` skips the link step.
4. **`evidence-archive` pre-flight step** — runs *before*
   `verify-evidence`, auto-moves any prior haul's record-and-continue
   evidence (audit-layer halts, self-improve halts) to
   `tools/haul/_evidence_archive/<source-haul>/`. Mechanically
   removes the "haul-3 attempt 1 halted on stale haul-2 evidence"
   failure mode.

Three more were identified but require runner edits and are deferred
to a focused tooling haul:

- Skip verify-pass for deterministic validators (~50% wall savings)
- `RESUME_FROM=<gate-id>` flag (cuts attempt-loop cost)
- Per-validator-kind "deterministic" attribute

These aren't blockers for Phase 2 hauls; they're refinements that
pay off proportionally to attempt-loop friction.

## Out of scope (do not attempt this phase)

- **GPU sampling.** Phase 2.5 work per ROADMAP.
- **Qwen3-MoE second-architecture validation.** Phase 3.
- **Continuous batching / serve.** Phase 4.
- **Speculative decoding.** Phase 4.5.
- **mmap KV cache.** Phase 5.

## What "Phase 2 closed" means

Phase 2 is closed when:
- All W1A / W1B / W2 parity tests are PASS at atol=1e-3.
- 50-prompt token regression is PASS post-W2.
- W3 perf-ratio-assert is PASS: prefill ≥2× Phase 1, decode tok/s
  ≥0.9× MLX (ROADMAP gate).
- `_phase2_haul${N}_attempt${M}_closeout.md` written for each haul
  attempted; the haul that lands W2 also lands the perf gate.
- `cargo test --workspace --lib` still passes; Phase-1 parity
  re-attest (AU5 against `phase1_kernel_parity`) still green.

Phase 2 not closed → no Phase 3 prep starts.
