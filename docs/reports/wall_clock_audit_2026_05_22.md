# Wall-clock audit — 2026-05-22

Looked across memory + reports for past-run findings that imply
wall-clock waste in future sessions. Applied the easy wins inline; the
rest are queued as concrete recommendations.

## Summary — top sources of past wall-clock waste

| pattern | typical loss | mitigation |
|---|---|---|
| Investing in a lever before measuring its cost share | 1 session per false-target (e.g. LM-head simdmat at 4% not 70%) | Cost-share gate before any wedge |
| Designing levers before calibration | days of design on dead angles (eagle5-routing, Q8-KV-layer-diff) | Calibration analysis FIRST |
| Re-spawning known-dead levers (ICB, MoE megakernel, serial-dispatch, sumy-trick v3) | 1-3 sessions each | `reports/dead_levers.md` registry (NEW, this audit) |
| In-loop benches gated on contaminated absolute tps | every gated wedge wastes its bench | quick_bench.sh `--strict` mode (LANDED, this audit) |
| Profile shader-hash drift breaking existing tests | minutes per test run × repeated | Test-helper that builds in-memory profile (in tree as the pattern; one extraction wedge from being a 1-line helper) |
| Corpus duplicate-row from watchdog restart bug | ~60% of training compute wasted on dups | Trainer dedup pass (LANDED, this audit) |
| Spec-decode K=1 regression undiagnosed | eagle5 v2 training (5-10 h) is wall-clock-wasted if runtime can't consume the head | Diagnose `path-to-100-repath` Track 2 BEFORE the overnight train |
| Stale profile pin | every `cargo test` that loads V2-Lite via the on-disk profile fails | Rotate profile (one autotune run; recommendation only) |

## What landed in this audit

### Trainer + τ-eval dedup pass

[tools/training/eagle5_train.py](tools/training/eagle5_train.py) and
[tools/training/eagle5_tau_eval.py](tools/training/eagle5_tau_eval.py)
now drop duplicate rows by token-fingerprint before windowing. The
corpus at `artifacts/calibration/v2_lite_corpus/` has ~1,500-2,000
unique sequences in 4,512 rows because `iter_chat_sequences` restarts
at row 0 on every watchdog launch (per [[corpus-complete-analysis-landed]]).

**Wall-clock saving:** if the corpus is ~60% duplicates, an overnight
training run that was 8 h now lands in ~3.2 h with the same number of
unique gradient steps. Override with `--no-dedup` if you specifically
want the duplication for some reason.

The τ-eval dedup matters too: without it, the head's "accept rate" on
duplicates is artificially high (the head saw them in training), so
τ is inflated and may mislead the ship gate.

### Bench `--strict` mode

[tools/bench/quick_bench.sh](tools/bench/quick_bench.sh) now accepts
`--strict` (or `STRICT=1` env var). With either, the script exits 64
if Claude.app is running. Previously it just warned; past sessions
recorded contaminated absolute numbers anyway and gated on them, then
spent the next session diagnosing the "regression" that didn't exist.

For paired-delta runs the warning is sufficient (per
[[feedback-bench-with-claude-open]]); strict mode is for absolute-tps
gates only.

### Dead-lever registry

[reports/dead_levers.md](reports/dead_levers.md) — consolidated index
of nine documented dead levers (eagle5-routing, f16-residual, ICB,
MoE-megakernel, MoE-serial-dispatch, Phase-Y-sumy-trick-v3,
Q5_0-simd_shuffle, Q8-KV-layer-diff, LM-head-simdmat-as-tps-lever)
with the killing evidence and resurrection check for each.

Includes a **pre-spawn checklist** at the bottom — 4 mandatory
questions to ask before opening any new wedge:

1. Is this lever in the dead-lever doc? Read its resurrection check.
2. Have you measured the cost share?
3. Does it gate on a calibration insight? Run the analysis FIRST.
4. Does it depend on a downstream that's already regressing?

Past sessions failed all four checks more than once. The registry
makes it cheap to do them.

## What's still to apply (queued recommendations, not in this commit)

### Rotate the on-disk kernel profile

The pinned `profiles/deepseek-v2-lite-q4.m3pro18.json` has
`shader_hash=cf91e98f20d00a9c15d2e0b2` but the current shader source
hashes to `05ac3c172932cfe7f6b0b327`. Every test that loads the on-disk
profile fails fast with `kernel profile shader hash mismatch`. The
existing `tests/integration_greedy_64.rs` has been broken on this for
some unknown number of commits. My `vocab_prune_parity.rs` works around
it by building the profile in-memory.

**Action:** when convenient, regenerate via:
```
./target/release/dismantle autotune \
    --weights models/deepseek-v2-lite-q4.gguf \
    --profile m3-pro-18gb \
    --max-hours 0.5 \
    --out profiles/deepseek-v2-lite-q4.m3pro18.json
```

(or just hand-edit the `shader_hash` field; deterministic-candidate
selection doesn't change). Saves ~13 s of profile-build per test run.

### Extract the in-memory profile helper

Both `vocab_prune_parity.rs` and (if it were patched to skip the on-disk
profile) `integration_greedy_64.rs` could share a single helper:

```rust
pub fn fresh_test_profile(weights: &Path) -> KernelProfile {
    let gguf = GgufFile::open(weights).expect("open gguf");
    build_deterministic_profile(&gguf, &AutotuneOptions::default())
}
```

Lives wherever fits — `crates/dismantle-core/src/profile.rs` behind a
`#[cfg(any(test, feature = "test-helpers"))]` gate is reasonable. Saves
copy-paste cost across future parity tests.

### Train-before-runtime guard for eagle5 v2

`reports/eagle5_v2_wiring_handoff.md` §9 documents the runtime
dependency. The user's overnight train (5-10 h) is wall-clock-wasted
if the spec-decode runtime can't consume the head. **Recommendation:**
do the [[path-to-100-repath]] Track 2 Step 2A diagnosis FIRST — locate
the 8.9-tps eagle4-K=1 tax via the `EAGLE4_BACKEND={metal,cpu}` sweep +
`DISMANTLE_SPEC_LOG=1` per-step trace. That diagnosis is ~1 h. If it
shows a structural blocker (capture-buffer-required-for-chain-seed
type), then training eagle5 v2 is premature; if it shows a fixable
overhead, train in parallel with the fix.

Adding this as an explicit "Phase 1.5" step in the path-to-50 workflow
saves potentially 10 h of overnight compute if the runtime is unfixable.

### Per-kernel time breakdown as a hard pre-flight

[[per-kernel-time-breakdown-2026-05-20]] is the current authoritative
snapshot. Any new wedge that claims "this will save N% wall" should
identify which entry in that breakdown contains the savings. If the
addressed entry is < N%, the wedge is dead before it starts.

Should be linked from every wedge design doc as a required cross-ref;
will start that convention now.

## Feedback memory

A short feedback memory will be added at
`~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/feedback_wall_clock_audit.md`
referencing this doc, so future sessions surface the audit pattern
without re-reading old session logs.
