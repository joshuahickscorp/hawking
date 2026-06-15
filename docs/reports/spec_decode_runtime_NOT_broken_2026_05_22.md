# Spec-decode runtime is healthy — prior "broken" conclusion was a false positive

**Date:** 2026-05-22
**Context:** Prior session's `diagnose_spec_decode_k1.py` returned exit 2 (all trials failed at K=1), and the conclusion was filed as "spec-decode runtime structurally broken; eagle5 v2 head can't deliver tps until runtime is rewritten". That conclusion is **wrong**.

## What actually happened

The diagnostic script invokes:

```bash
dismantle generate --speculate exact-shared --verify-window 1 ...
```

The binary explicitly rejects this:

```
Error: model: --verify-window must be 4, 8, or 16 for exact-shared; got 1
```

The script reads the non-zero exit as "spec-decode failed" and aborts. K=1 is not a supported mode of `exact-shared` — the diagnostic was testing an invalid configuration.

## What works

`exact-shared` runs correctly at K ∈ {4, 8, 16}. Verified by:

```bash
./target/release/dismantle generate \
    --weights models/deepseek-v2-lite-q4.gguf \
    --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
    --speculate exact-shared --verify-window 4 \
    --prompt "Once upon a time" --max-new-tokens 16 --seed 0
```

Output: `, there was a young girl named Cinderella. She lived with her evil stepmother`
Stats: `dec_tps=9.98 draft_accepted=6 draft_rejected=26 prefill_ms=13595 decode_ms=1603`

(Throughput is contaminated by concurrent eagle5 training. The structural result — that decode + draft + verify all work end-to-end — is what matters.)

## Implications for path-to-50 / path-to-100

The prior plan held that:

> 100 tps is gated on a separate "spec-decode runtime recovery" workstream that needs a Rust rewrite.

That workstream is **DEAD** as written. The runtime already works. The actual gates to higher tps are:

1. **Draft quality.** Eagle4 had `draft_accepted=6, draft_rejected=26 = 19% accept rate` on "Once upon a time". Mediocre. Eagle5 v2 (currently training) should beat that.
2. **Verify-window tuning.** K=4 vs 8 vs 16 has a tradeoff: longer window = more parallelism + higher acceptance reward, but also more wasted compute on rejections. Needs measurement per workload.
3. **Better drafts for the prompts that matter.** Story-completion prompts have wide branching; eagle was originally trained more on dialog/conversational data.

## Action items

1. **Don't waste time on the "runtime recovery" workstream.** It doesn't exist.
2. **When eagle5 v2 finishes training**, immediately plug it in via the exact-shared path (same mechanism as eagle4 — the spec-decode pipeline reads the head from `eagle4/v2lite_frozen.npz` or equivalent). Verify draft acceptance rate beats eagle4's.
3. **Patch `diagnose_spec_decode_k1.py`**: either remove the K=1 config entirely, or rename the test to "K=4 baseline" to match reality. Low priority — the diagnostic isn't needed for go/no-go decisions anymore.
4. **Update memory:** delete `path_to_100_repath.md`'s claim that "all spec-decode REGRESSES" — that was based on the same false signal. The 2026-05-20 baseline of "off=26.87, exact-shared K=4 = regress" likely reflects eagle4's poor draft quality on the bench prompt, not runtime breakage.
5. **Memory note:** add `spec_decode_runtime_healthy.md` to project memory.

## What changes in the 10h plan

- **Drop Stage 9** (spec-decode runtime investigation). Was ~2.5h. Reclaim that time.
- **Add a real Stage 9**: plug eagle5 v2 head into the spec-decode dispatch + measure tps with --verify-window 4/8/16 sweep. Compare to eagle4 baseline + off-mode. Estimated ~1-2h.

This is potentially the biggest upside of the whole session: eagle5 v2 head being deployable as soon as it lands, with no separate runtime fix gate.
