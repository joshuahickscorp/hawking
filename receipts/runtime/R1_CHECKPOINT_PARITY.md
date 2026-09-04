# R1 — prefix checkpoint: semantic gate PASSED

Run 2026-09-04 on the sealed body `NOETIC_PARENT_A`, resident stopped so only one
copy of the 27B model was resident.

```
crates/hawking-core/examples/qwen38_prefix_checkpoint_parity.rs
  --artifact  /Users/scammermike/noetic/NOETIC_PARENT_A
  --tokenizer .../tokenizer.json  --max-new 16
```

## Result

```
checkpoint: position=11  rec_state=37,748,736 f32  conv_state=1,474,560 f32
prefix tokens      : 11
full prompt tokens : 24
baseline generated : [271, 248068, 198, 760, 1156, 369, 9859, 883, 264, 3234, 314, 264, 8978, 6297, 13, 6558]
resumed  generated : [271, 248068, 198, 760, 1156, 369, 9859, 883, 264, 3234, 314, 264, 8978, 6297, 13, 6558]
prefill steps      : baseline 24 vs resumed 13 (skipped 11)

PARITY: BIT-IDENTICAL.
```

## Why this run mattered

A read-only scout raised a real structural concern: `restore_prefix()` rewrites
`rec_state`, `conv_state` and `position` but never touches the GQA KV buffers.
Its soundness rests on GQA KV at position i being a pure function of
`(token[i], i)`. `session.reset()` zeroes those buffers, so an intervening
request could refill positions `0..len2` with different tokens, and a later
restore would set position correctly while GQA held another request's KV --
`INVALIDATION_TOO_NARROW`, which R1's negative-control law makes a FAIL.

The scout correctly reported it as UNKNOWN rather than asserting it, noting no
receipt anywhere recorded this harness passing.

The harness stages exactly that adversarial case: capture the checkpoint,
`session.reset()`, generate on "Completely unrelated text about weather.", and
only then restore. It came back bit-identical. The concern is resolved by
measurement, not by argument, and this file is the receipt that was missing.

## R1 clauses satisfied by this run

| clause | status |
|---|---|
| warm processes fewer physical prefix positions than cold | YES, 13 vs 24 |
| semantic verification passes across cold and warm | YES, bit-identical |

Still outstanding for R1: prefix-mutation invalidates the right region,
suffix-mutation does not invalidate the prefix, repeated pairs, bookkeeping cost
reported, and the two prompt tok/s figures kept separate.
