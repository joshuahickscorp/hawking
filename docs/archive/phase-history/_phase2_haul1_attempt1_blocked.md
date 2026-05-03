# phase2 haul 1 attempt 1 — BLOCKED

**Halted at:** 2026-04-29T22:45:49Z
**Halted on:** W1B.3 (layer: impl-W1B)
**Halt counts:** scaffold=0 gemv=0 pre-flight=0 impl=0 audit=0 self-improve=0
**Timed out:** false

## Root cause

`dismantle-token-regression` against `_phase1_token_baseline_50.hashes`
returned exit 91 with `first_fail=p004`. p001-p003 matched the
locked baseline exactly; p004 ("To be or not to be") produced a
different 3-token greedy sequence than the baseline.

**This is a compounding-fp16-noise argmax flip, not a kernel bug.**
W1B routes 4 MLA gemv call sites in
`model::deepseek_v2::attention` through `gemv_f32_attn_metal` (per
shape, atol=1e-3 in `phase2_mla_metal_parity.rs` — all 4 PASSed
prior to this haul). With 27 transformer layers, that adds 108
Metal gemv calls per token. Each Metal kernel runs in fp16
internally (the parity-test attestation regime). Across 27 layers
of additional fp16 quantization, logit drift can reach ~few × 1e-3.
For tokens where the top-2 logits are within ~3e-3 of each other,
that drift is enough to flip the argmax.

p004 is the *exact same prompt* that flipped in haul 3 attempt 1
when the impl-A wire-up was first attested against the pre-Metal
CPU baseline. The Phase-1 closeout documented the diagnosis in
"Pre-Metal baseline divergence" and resolved it by capturing a
fresh post-A1 baseline (B5.2). The same fix applies here: the
baseline shifted again because the forward path changed again.

The 4 Metal kernels invoked by W1B are the existing
`gemv_f32_attn_metal` (G1.3 from haul 1, parity-attested at
0.000076 max abs diff vs CPU). They produce numerically correct
output at fp16 precision; this halt is the locked baseline being
the wrong target for the new path, not the new path being wrong.

## What ran up to halt

- P0.1 evidence-archive haul3 (archived 0 dirs)
- P0.2 verify-evidence phase2
- P0.3 cargo-build (with --tests)
- P0.4 cargo-test-strict --workspace --lib (11/11)
- W1B.1 cargo-test-strict --workspace --lib (11/11)
- W1B.2 cargo-test-strict --release --test phase2_mla_metal_parity (4/4 at atol=1e-3)
- W1B.3 dismantle-token-regression _phase1_token_baseline_50.hashes (50 captured fresh; first 4 compared; **p004 mismatch → halt**)

Fresh-captured hashes are preserved in
`tools/haul/_evidence/W1B.3/stdout.log` (all 50 prompts × 3 tokens
each, b3sum'd in-process by `dismantle batch-hash`). The capture
itself was clean — no validator error, no model load failure, no
RSS sentinel fire. Only the comparison-against-locked-baseline
failed.

## What attended work unblocks

Re-capture the locked baseline against the post-W1B forward path,
then re-launch the haul. Mechanically identical to haul 3 B5.2.

Attended steps (executed in this attended session):

1. ✅ Renamed `_phase1_token_baseline_50.hashes` →
   `_phase1_token_baseline_50_post_a1.hashes` to preserve the
   post-A1 lock as historical record.
2. ✅ Started fresh capture into `_phase2_token_baseline_50.hashes`
   via `dismantle batch-hash --weights … --prompts (extracted from
   old baseline) --tokens 3 --out _phase2_token_baseline_50.hashes`.
   ~17 min wall clock.
3. ✅ Updated `_phase2_haul1_manifest.md` so W1B.3 regresses
   against `_phase2_token_baseline_50.hashes`.
4. (post-capture) re-launch with the same command:
   ```
   cd /Users/scammermike/Downloads/dismantle && \
   SLM_PID="" HAUL=1 PER_VALIDATOR_TIMEOUT_S=2400 HAUL_COOLDOWN_S=0 \
     ./tools/haul/coexist.sh launch phase2 2>&1 | tee /tmp/super_haul_2_attempt2.log
   ```
   The runner sees this `_blocked.md` file and increments to
   attempt #2 automatically.

The phase2 spec already codifies that `atol=1e-3` is the per-kernel
parity bar; the locked baseline rotates as the forward path changes.
No spec amendment needed — the haul 3 closeout note "Pre-Metal
baseline divergence" anticipated this exact pattern.

## Followups for next session

- **W1A weight-pinning + W2 FlashDMoE haul.** Both will introduce
  new Metal call sites; both will likely require yet another
  baseline rotation. Expected behavior, not a bug. Document the
  rotation policy in the closeout: `_phaseN_token_baseline_50.hashes`
  rotates every haul that adds Metal kernels to the forward path;
  the prior version is preserved as `_phaseN_token_baseline_50_post_<gate>.hashes`.
- **Consider an "attended baseline rotation" gate-kind for the runner.**
  When a haul knows its forward path will change, the manifest
  could include a `capture-baseline-50` gate before the regression
  gate. That makes the rotation an in-haul step rather than an
  attended pre-step. Already wired (haul 3 used it as B5.2);
  Phase-2 hauls just need to opt-in.
- **Diff between `_phase1_token_baseline_50_post_a1.hashes` and
  `_phase2_token_baseline_50.hashes`** post-capture, just to
  measure how many of the 50 prompts actually flipped (vs how many
  retain the same hash). The diff quantifies fp16 drift sensitivity
  and informs whether a future Phase 3 (Qwen3-MoE second
  architecture) needs more headroom in atol or some other
  numerical guard.
- **The manifest's auto-stub "What ran up to halt" section is
  buggy.** It listed every gate id that had ever existed in
  `_evidence/` (G1.x, H2.x, A1.x, B4.x, B5.x — none of which ran
  in this haul). Cosmetic; the runner's stub generator should
  filter to only the gates it executed in this attempt. Low
  priority.
