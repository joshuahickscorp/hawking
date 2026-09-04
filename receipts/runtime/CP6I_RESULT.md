# CP6i -- the chunked prompt wall: 1.35-1.40x, capability-equivalent

2026-09-04 · **G019 acceptance met** · resident `NOETIC_PARENT_A`, quiet lane

CP6d measured the prompt wall the sequential way -- one `step()` per prompt
token. G019's acceptance is capability equivalence **and** a lower complete
prompt wall against that. Everything since has been organs, kernels and layers.
This is the wall.

## Result -- 128 prompt tokens, all 64 layers, full resident

```
  chunk 4   sampled 95726 == 95726  EQUIVALENT   3947.2 ms -> 2811.3 ms   32.43 -> 45.53 tok/s   1.404x  [conservative 1.372x]
  chunk 4   sampled 95726 == 95726  EQUIVALENT   3627.9 ms -> 2691.6 ms   35.28 -> 47.55 tok/s   1.348x  [conservative 1.339x]
  chunk 2   sampled 95726 == 95726  EQUIVALENT   3824.1 ms -> 3375.2 ms   33.47 -> 37.92 tok/s   1.133x  [conservative 1.065x]
```

**ACCEPTANCE: PASS -- equivalent AND lower wall.** Reproducible within 4% at
chunk 4. The sequential baseline is retained and remains the default path;
nothing was removed to reach this.

## What the equivalence check earned

It failed first, and that is the point. The first run reported:

```
  sampled token   sequential Some(220)   chunked Some(6443)   DIVERGED
  speedup 1.353x   ACCEPTANCE: FAIL -- not capability-equivalent
```

A **1.353x that was wrong**. A timing-only harness -- which is what CP6a, CP6b,
CP6c and CP6g all were -- would have reported it as a win. Two defects were
behind it, and the discriminator that separated them was running at **chunk=1**,
where the chunked path must be semantically identical to sequential:

```
  chunk 1   sampled 1186 == 1186   EQUIVALENT
```

K=1 equivalent, K=4 diverged. That localised the fault to something that only
exists when K>1, not to the encoder wiring.

**Defect 1 -- shared RoPE index and KV slot.** `encode_gqa` reads
`self.position` for the rotary index and the KV cache offset. `prefill_chunk`
advanced position once per CALL, so all K positions of a chunk landed on one
slot. Fixed by cloning the `MetalContext` so the command buffer borrows the
context rather than `self`, which frees `&mut self` to set `self.position =
base + k` before each mixer.

**Defect 2 -- an intra-chunk data race, and the one that actually mattered.**
The first fix changed nothing: still token 6443, byte for byte. The dispatches
in the chunk were never serialised, so within a layer position k's gather
overwrote `workspace.hidden` while k-1's mixer was still reading it, and k's
mixer read the recurrent state before k-1 had written it. `encode_full_token`
guards this with `begin_serial_group`; `prefill_chunk` did not. Adding it made
the arms equivalent **and cost nothing measurable** (1.346x to 1.349x) -- the
dependencies were real, so ordering them removed no parallelism that existed.

This defect was present in CP6g's arm B too. CP6g measured only time, so it
could not see it, and its 1.27-1.33x layer figure was measured on a racing
encoder. The prompt-wall numbers above are from the serialised path.

## And a bias in the harness

The first stable-looking runs had arm A (sequential) always first. That hands
arm B a warmer GPU, and since the sequential arm is the longer one, the bias
points **toward the result being claimed**. Its spread showed it: 3609 / 4611 /
4272 ms across runs, 28%.

Alternating the order -- A then B on even reps, B then A on odd -- moved chunk 4
from 1.472x to 1.404x and tightened the baseline to 2% (3919-4001 ms). The
correction went the honest direction, and the tightening confirms the
instability was the ordering artifact rather than the machine.

## The mechanism

The mixer is genuinely per-position and was **not** batched. Per layer, each
position is gathered into `hidden`, run through the mixer exactly as it runs
today, and scattered into a K-wide residual; the MLP then runs **once** for all
K. The 2K glue dispatches are paid. Only the last position's logits are computed
-- CP6h measured that head at 4.3% of a token, so it is a small part of this,
not the mechanism.

So the standing blocker -- "batch the mixer" -- was never required.

## Standing

| checkpoint | what it established |
|---|---|
| CP6e | interleaved norms, bit-identical, both tails |
| CP6f | the chunked MLP is correct on real weights; widening K costs nothing |
| CP6g | the hybrid layer wins with the glue paid (on a racing encoder) |
| CP6h | prefill's discarded LM head is real and only 4.3%; not built |
| CP6i | **the prompt wall, 1.35-1.40x, capability-equivalent** |

Not established: chunk > 4 (the fused swiglu family is instantiated only to
(4,4)), prompts beyond 128, and the decode wall, which this does not touch.
