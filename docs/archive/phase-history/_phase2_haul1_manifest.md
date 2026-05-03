# Phase 2 — Haul 1 manifest (super-haul: W1B MLA Metal + perf gate)

The first Phase-2 haul. Bundles two impl layers + audit + closeout
into a single launch. W1A weight-pinning was originally scoped for
this haul but pulled to Phase-2 Haul 2 alongside W2 (FlashDMoE) —
both are multi-day attended sprints; bundling them keeps Haul 1
single-session-friendly.

Layers, in execution order:

1. **Pre-flight** — auto-archive prior-haul record-and-continue
   evidence, re-attest haul-3, build all targets, lib tests green.
2. **Impl W1B — MLA / Q-LoRA gemvs onto Metal.** Four MLA gemv call
   sites in `model::deepseek_v2::attention` (q_a_proj, q_b_proj,
   kv_a_proj_with_mqa, kv_b_proj) routed through
   `gemv_f32_attn_dispatch`. Estimated decode uplift ~1.3×.
   **Implementation landed 2026-04-29 in this attended session.**
3. **Impl W3 — perf gate.** Dismantle / llama.cpp / MLX
   decode-tps suites; ratio assertion against ROADMAP gate.
4. **Audit** — clippy / fmt / parity / lib re-attest.
5. **Closeout** — always runs.

W1A and W2 land in `_phase2_haul2_manifest.md` (TBD); both require
days of attended surgery (Buffer-lifetime work for W1A; new shader
+ parity test for W2).

## Cumulative decode estimate

Post-haul-3 baseline: ~0.13 dec_tps. After W1B alone:

```
0.13 × 1.3 (W1B MLA Metal) ≈ 0.17 dec_tps
```

llama.cpp Metal sits at ~30 dec_tps on the same model. **r_llama after
this haul ≈ 0.006**, well below the Phase-2 gate (`r_mlx ≥ 0.9`).
That's the expected finding — W3 perf gate will halt with
`reason: perf_below_threshold`, and *that's the data point* that
quantifies how much W1A + W2 (Haul 2) needs to deliver.

This haul is not trying to close Phase 2. It's establishing the
post-W1B baseline so Wedge-2's unlock is measured against
something honest, AND validating that the runner machinery
(evidence-archive + impl-W3 layer + perf-ratio-assert with
generalized gate ids) lands cleanly.

## Halt budgets

| Layer | Halt rule |
|-------|-----------|
| pre-flight | 1 halt = end haul |
| impl-W1B | 1 halt in {W1B.1..W1B.3} = end haul |
| impl-W3 | 1st halt in {W3.1..W3.3} continues; W3.4 (ratio assert) halt = end haul; 2nd halt anywhere = end haul |
| audit | record-and-continue |
| closeout | always runs |

The W3 asymmetry mirrors haul-3's `impl-B4` pattern: a single bench
trial flake (e.g., MLX cold-start outlier) shouldn't kill the haul,
but the ratio verdict IS the perf claim.

## Time discipline

- Per-item soft ceiling: 60 min
- Haul hard ceiling: 4 hr (no waiver this haul; per spec)
- Expected wall clock: ~45-60 min if green, ~30 min more per halt

Wall-clock budget breakdown:

| Layer | Gates | Est. wall | Notes |
|-------|------:|----------:|-------|
| pre-flight | 4 | ~3 min | evidence-archive + verify + build + lib tests |
| impl-W1B | 3 | ~25 min | parity ~1 min + token regression ~17 min + verify-pass doubles cargo-test |
| impl-W3 | 4 | ~12 min | 3 bench-decode @ **1 trial** × 64 tokens + ratio assert (5-trial config timed out at 40 min in attempt 2; 1-trial fits well under the timeout while still capturing real ratio numbers) |
| audit | 4 | ~5 min | clippy + fmt + parity re-attest (AU1 dropped) |
| closeout | 1 | <1s | noop |
| **total** | **16** | **~60 min** | |

## Pre-launch state (what's already landed in this attended session)

Already in place when this manifest launches:

- ✅ **W1B implementation.** 4 call sites in
  `crates/dismantle-core/src/model/deepseek_v2.rs::attention()`
  switched from raw `gemv_f32` to `self.gemv_f32_attn_dispatch`
  (lines 694, 697, 708, 729).
- ✅ **W1B parity test.**
  `crates/dismantle-core/tests/phase2_mla_metal_parity.rs` — 4
  shape-specific tests (q_a_proj 1536×2048, q_b_proj 3072×1536,
  kv_a_proj 576×2048, kv_b_proj 2048×512). All 4 PASS at
  atol=1e-3 fp16; max diff observed ~3e-4.
- ✅ **MLX comparator model pre-pulled.** 8.2 GB cached at
  `~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx`.
  `mlx_lm.generate` 0.31.3 in PATH.
- ✅ **Runner edits.** `tools/haul/run-gates.sh` now has:
  - `evidence-archive` validator kind (auto-archives evidence dirs
    whose `post.json` reports `exit_code != 0` to
    `tools/haul/_evidence_archive/<source-haul-id>/`).
  - `impl-W1A`, `impl-W1B`, `impl-W3` layer cases.
    `impl-W1A` is wired but unused this haul; left in place for
    Haul 2.
  - `perf-ratio-assert` generalized to take optional gate-id args
    `<g-dis> <g-lla> <g-mlx>` (defaults preserve haul-3
    backwards-compat with B4.2/B4.3/B4.4).
- ✅ **Phase 1 baselines unchanged.** Lib tests 11/11, phase1 parity
  8/8, build clean — verified post-W1B.

## Pre-launch lint (sanity check before launch)

Static checks the audit identified as attempt-loop preventers. None
should fail given the pre-launch state above:

```bash
# 1. Spec exists
[[ -f _phase2-spec.md ]] || echo "MISSING _phase2-spec.md"

# 2. Manifest at templated path
[[ -f _phase2_haul1_manifest.md ]] || echo "MISSING _phase2_haul1_manifest.md"

# 3. All layer names in manifest have matching cases in run-gates.sh
for layer in pre-flight impl-W1B impl-W3 audit closeout; do
  grep -q "^[[:space:]]*${layer})" tools/haul/run-gates.sh || echo "MISSING case for layer: $layer"
done

# 4. Cargo test filters resolve (parity test compiles)
cargo test --release --test phase2_mla_metal_parity --no-run --quiet >/dev/null 2>&1 \
  && echo "phase2_mla_metal_parity: OK" \
  || echo "phase2_mla_metal_parity: COMPILE FAIL"

# 5. MLX comparator model cached
[[ -d ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx ]] \
  && echo "mlx model: cached" \
  || echo "mlx model: MISSING"

# 6. mlx_lm.generate in PATH
which mlx_lm.generate > /dev/null && echo "mlx_lm.generate: OK" || echo "mlx_lm.generate: MISSING"

# 7. Locked baseline present
[[ -f _phase1_token_baseline_50.hashes ]] || echo "MISSING _phase1_token_baseline_50.hashes"
```

If everything prints OK / no MISSING lines, launch.

## Launch

```bash
cd /Users/scammermike/Downloads/dismantle && \
SLM_PID="" \
  HAUL=1 \
  PER_VALIDATOR_TIMEOUT_S=2400 \
  HAUL_COOLDOWN_S=0 \
  ./tools/haul/coexist.sh launch phase2 2>&1 | tee /tmp/super_haul_2.log
```

Notes:
- `SLM_PID=""` — no slm to coexist with.
- `HAUL=1` selects this manifest (`_phase2_haul1_manifest.md`).
- `HAUL_COOLDOWN_S=0` per spec § CE-6.
- `PER_VALIDATOR_TIMEOUT_S=2400` (40 min) generous for token
  regression's ~17-min run + its verify-pass.
- Pipe to `tee` so the runner output is preserved at
  `/tmp/super_haul_2.log` for paste-back / async review.

## Gate runner manifest

```
# layer: pre-flight
P0.1 evidence-archive haul3
P0.2 verify-evidence phase2
P0.3 cargo-build
P0.4 cargo-test-strict --workspace --lib

# layer: impl-W1B
W1B.1 cargo-test-strict --workspace --lib
W1B.2 cargo-test-strict --release --test phase2_mla_metal_parity
W1B.3 dismantle-token-regression _phase2_token_baseline_50.hashes

# layer: impl-W3
W3.1 bench-decode dismantle 1
W3.2 bench-decode llamacpp 1
W3.3 bench-decode mlx 1
W3.4 perf-ratio-assert 0.9 0.9 W3.1 W3.2 W3.3

# layer: audit
AU2 cargo-test-strict --workspace --lib
AU3 cargo-clippy-baseline 30
AU4 cargo-fmt-check
AU5 cargo-test-strict --release --test phase1_kernel_parity

# layer: closeout
Z1 noop super-closeout
```

Notes on the fenced block:
- AU1 dropped per spec (P0.2 already attests).
- `verify-evidence phase2` runs against the existing
  `_evidence/` tree. Without phase2 gates already attested it's a
  no-op-ish pass; once W1B / W3 land their evidence, it becomes the
  re-attestation gate.

## Closeout

On haul end the runner writes
`_phase2_haul1_attempt${N}_blocked.md` (on halt) per CLAUDE.md tone-
of-artifacts rule. An attended-session agent then writes
`_phase2_haul1_attempt${N}_closeout.md` covering:
- Per-layer pass/fail table
- W1B parity diffs (max abs vs CPU ref per shape)
- Measured pre-W1B vs post-W1B dec_tps from W3.1 vs the haul-3
  closeout's ~0.13 baseline
- W3 perf-ratio output (the actual Phase-2 gate measurement) —
  expected to halt at `perf_below_threshold` since W1A/W2 aren't
  in this haul
- Audit drift summary
- Phase-2 Haul-2 (W1A + W2) scoping informed by the W3 baseline

## Risks acknowledged

- **W3 will halt with `perf_below_threshold`.** Expected outcome.
  W1B alone doesn't close the Phase-2 gate; `r_mlx` will be ~0.01
  vs the 0.9 threshold. The halt is the data point, not a failure.
  Closeout records the ratio numbers, then Haul 2 (W1A + W2) is
  scoped against them.
- **`evidence-archive` is a new validator kind, first run.** Risk
  class is "vacuous PASS." The implementation auto-detects
  evidence dirs with `post.json::exit_code != 0` and moves them to
  `_evidence_archive/<haul-id>/`. If the script silently no-ops on
  no matches, the haul proceeds (correct). If it fails on a
  malformed evidence dir, pre-flight halts (1-halt-end behavior).
- **Verify-pass-skip-for-deterministic-validators not shipped.**
  This haul still pays full verify-pass cost on cargo-test gates
  (~50% wall clock overhead). The audit identified this as a
  later runner refinement.
- **MLX bench first-run cold-start.** Despite the pre-pull, the
  first MLX trial loads the model from disk into MLX's runtime
  caches. Bench harness drops the first trial; ~30s warm-up
  built into the suite.
- **W1B's parity test passes but the production path passes
  weights through `&[f32]` slices (no pinning).** That's by
  design — W1B is an "off-CPU" haul, not a "stop-allocating-on-
  every-call" haul. W1A delivers the latter. The W3 perf number
  measured here will reflect the un-pinned cost.

## Cross-references

- Operating contract: `CLAUDE.md`
- Phase-2 spec (locked rules): `_phase2-spec.md`
- Speed-item triage: `_phase2_speed_followups.md`
- Phase-1 closure record: `_phase1_haul3_attempt4_closeout.md`
- Locked correctness baseline: `_phase2_token_baseline_50.hashes` (post-W1B; previous post-A1 baseline preserved as `_phase1_token_baseline_50_post_a1.hashes`)
- W1B implementation: `crates/dismantle-core/src/model/deepseek_v2.rs::attention`
- W1B parity test: `crates/dismantle-core/tests/phase2_mla_metal_parity.rs`
- Existing kernel parity tests: `crates/dismantle-core/tests/phase1_kernel_parity.rs`
- MetalContext API: `crates/dismantle-core/src/metal/mod.rs`
- Runner: `tools/haul/run-gates.sh`
- Historical (replaced by this manifest): `_phase2_wedge1_manifest.md`
