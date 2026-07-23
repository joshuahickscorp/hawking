# GLM-5.2 next-parent transfer packet

Sealed from Generation B's pilot. GLM-5.2 is one rung, not a destination, and what follows
is what the next parent should inherit, what it should never rerun unchanged, and what is
still open.

Status: `RESEARCH_RUNG_COMPLETE_AT_PILOT`. No full traversal was run, on purpose. Section
13 forbids streaming 1.51 TB to reproduce a failure the pilot closed, and the pilot closed
four.

---

## 1. The parent-specific law

GLM-5.2's routed-expert path, measured on real weights against sealed teacher capsules,
block output cosine at the window's own complete rate:

```text
0.3306 BPW -> 0.041
0.4990     -> 0.086
0.7531     -> 0.154
0.8931     -> 0.269
2.0169     -> 0.700      above the one-bit law, diagnostic only
```

Reaching cosine 0.50, where the compact block still carries half the teacher's direction,
takes roughly **1.5 bits per weight**. The campaign's law allows 1.0, where the curve sits
near 0.30. Replicated on a second window in a different region of the model.

The dense path is not bound: layer 0 clears the floor at cosine 0.709 under the same codec
at a comparable rate. The failure is specific to the routed experts, which carry 97.492
percent of the weight.

Directive taxonomy: **EXPERT_FUNCTION_BOUND**.

## 2. Methods to inherit

```text
complete BPW as the only judged rate, reconciled against physical bytes
every declared tensor physically stored, verified, before any eviction
protected organs carried natively at source precision, billed from the bytes stored
teacher capture on real corpus records, split-disjoint, every domain
trajectory measurement against a sealed capsule, not weight-space error
selection scored as set overlap, never as cosine over ids
a bounded above-ceiling oracle to separate rate-bound from family-bound
dependency windows derived from the tensor index, never shard order
```

The container, `hawking.gravity.container.v1`, transfers unchanged.

## 3. Methods never to rerun unchanged

```text
weight-space product quantization at sub-bit rates on a large MoE expert path
per-tensor low-rank factorization as a codec
a shared basis across a layer's experts
asymmetric BIT allocation as a rescue when one role holds most of the weight
```

The first three were measured here at a matched rate and landed within 0.116 to 0.157
block output cosine. The fourth is closed by arithmetic rather than by measurement.

Two of these were already dead in `docs/dead_levers.md` for a small dense parent, killed
2026-05-30 and 2026-05-31. This run reproduced the kill independently on a 753B MoE parent
without re-litigating it: per-tensor low rank at rank 71 reached mean weight error 0.9497
and the shared basis at rank 280 reached 0.9746. The activation-weighted reframe was
**not** run, because it is a recorded Type-1 kill and the house rules forbid re-opening it.

## 4. Best complete rate and half-bit status

```text
best physically exact candidate      0.7531 complete BPW, verified, complete coverage
its trajectory                       block output cosine 0.154, far below the 0.50 floor
half-bit candidate                   0.4990 complete BPW, exact, cosine 0.086
half-bit verdict                     NOT PROMOTED, trajectory not reachable
one-third candidate                  0.3306 complete BPW, exact, cosine 0.041
```

The half-bit rate has no single legal geometry on these shapes: single-subspace PQ at
power-of-two k steps straight over the 0.48 to 0.50 band, so it was reached by positional
allocation between rank 128 at dim 16 and rank 16 at dim 8. That construction transfers;
the result does not.

## 5. First causal divergence

Attention output is already at cosine 0.163 before the MoE runs. The expert path then
collapses to 0.0007. The router keeps its weight magnitudes, cosine 0.9995, and loses its
choices: top-8 expert set agreement is 0.119. Natively carried organs are untouched, with
indexer scores at 0.988, which is the protected-tensor policy working.

So the divergence is not one site. Both the attention projections and the expert
projections fail as soon as they are compressed, and the organs that are not compressed
are fine. The clean reading is that **what is compressed dies and what is carried lives**,
at every rate the law permits.

## 6. Doctor and base prior

Untested here. A Doctor correction competes for the same bits, and at these rates there
are no spare bits: the expert path cannot reach 1.0186 BPW even when every other role is
given zero. A Doctor allocation on this parent would have to come out of the expert budget
that is already 1.5x short.

## 7. Streaming throughput and amplification

```text
Xet sustained, serial                1698 Mbps
Xet sustained, 4 concurrent files    1984 Mbps
Xet sustained, 8 concurrent files    2001 Mbps
selected                             8 files, 1.178x over serial, knee between 4 and 8
```

This **corrects** the inherited note that parallelism does not raise throughput on this
link. It raises it 17.8 percent and then saturates.

```text
teacher capture, full sparse MoE layer     22.7 s
pack one MoE layer at k=8192               ~700 s
pack one MoE layer at low rank             ~80 s
pack one MoE layer, shared-basis student   ~66 s
source-byte amplification                  1.00x, no shard fetched twice
```

## 8. Adapter requirements

The propagation harness needs only a `tensor(name)` source, so any new representation
becomes measurable by supplying a decoder. Both families added in this run plugged into
the same path, which is why a difference between them cannot be the reader.

The container does not yet carry **side information shared across tensors**. A shared
basis, a runtime table, or a serialized Doctor state has no home in v1, which describes
tensors and not side channels. Section 5.2 requires runtime tables and Doctor state to be
billed, so the next parent needs this before either can ship. The shared-basis student in
this run wrote to a sidecar and billed from real bytes as a stopgap.

## 9. Recommended first pilot windows for the next parent

```text
the globals window plus the first dense layer      cheap, and it is the control that shows
                                                   whether a failure is model-wide
the last dense layer and the first sparse layer    the transition is where dense results
                                                   stop transferring
one complete IndexShare or attention-sharing group with its owner
the final layer plus the head
```

Run the above-ceiling oracle **first**, not last. One rung at twice the target rate tells
you whether the family can work at all, for the cost of one window, and it would have
saved most of this run's compute had it been run before the rate ladder.

## 10. What is still open

```text
a student fitted against the block's OUTPUT rather than its weights, which is what
section 6.1 actually proposes and which every family tested here is not
the IndexShare-aware attention student
IndexShare selection itself, untested: a 256-position batch against index_topk 2048
returns all keys, so selection was trivially complete at every rate
```

The output-fitted student is a training problem, not a factorization problem. It cannot
be reached by any closed-form decomposition, and the two closed-form directions that
looked like it, data-free and activation-weighted low rank, are both recorded kills.
