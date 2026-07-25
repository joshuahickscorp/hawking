# Hawking Motherload Completion — Session Report

```text
endpoint:  IN_PROGRESS  (not HAWKING_ODYSSEY_READY)
gates:     5 / 20 closed
blocker:   the GLM-5.2 source traversal is at 70/282 and running
```

This is the honest position. The endpoint is not reached, and the reason is a single
hard dependency: eleven of the twenty terminal gates require a complete local GLM
artifact, and the traversal that produces it is 25% through 1.507 TB. Everything that
does not depend on it is built, measured, and green.

---

## What is finished and verified

### The `.gravity` runtime executes a complete token

Three independent decoders agree on what a `gravity-pq` payload means: the numpy
reference that defines the codec, a Rust CPU path, and a Metal kernel.

```text
CPU  vs numpy oracle    max |logit diff|  2.61e-05
GPU  vs numpy oracle    max |logit diff|  6.68e-06
argmax and top-5        exact on both
```

Graded against what the artifact *encodes*, not against the BF16 parent — a sub-bit
artifact is lossy by construction and would fail that test while being a perfectly
correct execution of what it stores.

### Measured base throughput — `BASE_TRUE_TPS`, no acceleration

```text
ctx    128    prefill 116.5    decode 105.8 tok/s    ttft 47.1 ms
ctx    512    prefill  92.8    decode  68.8 tok/s    ttft  8.7 ms
ctx   2048    prefill  48.8    decode  29.2 tok/s    ttft 18.9 ms
ctx   8192    prefill  19.1    decode  13.3 tok/s    ttft 59.4 ms

cold load 680 ms · warm 368 ms · 135.7 MB resident vs a 135.6 MB artifact
1 command buffer and 210 dispatches per token at every context
```

Started at 34.6 tok/s at ctx 128 with 65 command buffers per token. Collapsing the token
into one command buffer gave 3.2x; moving attention onto the device fixed the decay with
context. End-to-end generation is 60.7 tok/s after making decode incremental, and
incremental decode is bit-identical to replaying the prefix — 0.0 difference, not "within
tolerance".

### The production path is closed

`load_engine` dispatches on the container's magic bytes, so a `.gravity` artifact reaches
`hawking-core` through the same reviewed registry as every other architecture, and streams
tokens out. No source weights are opened. A file with a `.gravity` extension over GGUF
bytes is asserted to be rejected — the registry routes on what a file *is*.

### GLM-5.2 architecture adapter

MLA, DSA, IndexShare, the `noaux_tc` router, grouped expert selection, routed and shared
experts. Agrees with the numpy oracle to **3.84e-06**, with argmax, top-5 and the DSA key
selection all exact, on a tiny model carrying the flagship's exact semantics — including a
layer that must reuse the previous layer's index rather than recompute it.

### Prometheus — all 14 Revision 3 §7 components

Eight measure, five are gated with named gates, one seals. A gated stage cannot be
constructed without naming its gate: that is enforced in the type.

The equal-budget check found a live defect. Math, General and Random land on 46.70 GB;
Uniform landed 3.56 GB **lighter**, which would have handed the conditioned arms a win
they did not earn. The budget is now fixed and the rate solved for it, as exact rationals.
All four arms match to 0.0175%. **Uniform needs 2.53 bits per weight to spend what the
conditioned arms spend at 1.0 with a natively-carried embedding and head.**

### Model Odyssey — prepared, fenced, not started

86 dry-run checks, zero failures. `ODYSSEY_LAUNCH_AUTHORIZED` is `false`, the builder
reads that file and never writes it, and the selftest proves both directions: rebuilding
leaves it false, and flipping it to true makes validation **fail**. Lean `v4.15.0` and
Mathlib `v4.15.0` pinned to concrete revisions; `latest` is rejected.

---

## Two findings worth stating plainly

### The first rate failed its own screening gate

The Llama-3.2-1B instrument at 0.877 BPW executes correctly and produces nothing usable:

```text
prompt      "The capital of France is"
continues   " settle settle settle settle settle ..."
oracle top5  settled, settles, settle, ewise, booster
```

Five inflections of one wrong word — from the numpy oracle reading the same container.
The runtime is faithfully reproducing a collapsed model. Had this been graded against the
BF16 parent it would have been filed as a decoder bug and the real finding lost.

This bounds nothing about the flagship. A dense 1B has little redundancy to spend; a 744B
MoE has expert topology, which is the redundancy the whole ladder exists to exploit.

### The traversal was three windows from a disk stall, silently

The eviction gate had authorized nothing since the run began. Every fetched shard was
being retained, the deferred set grew 12 → 22 → 33, and the controller reported `RUNNING`
with zero faults throughout — silence looked exactly like health.

The gate was right to refuse. Every teacher capsule on disk was sealed against eight ids
from a SHA-256 stream, uniform over the vocabulary; calibration has since moved to 256
real corpus tokens. So the check stayed and the capsules were archived with a withdrawal
receipt, and the chain was re-seeded from layer 0.

```text
before   0 / 33 shards authorized for eviction
after   33 / 33 authorized, 0 refused
result  free disk 274.6 -> 404.8 GiB at the next window boundary
```

---

## Gate status

| gate | state | condition |
|---|---|---|
| M09 | **green** | Prometheus architecture and profiles implemented |
| M14 | **green** | sandbox, roles, Ledger, verifiers, Tribunal, retrieval scaffolded |
| M15 | **green** | Lean/Mathlib and evidence environment pinned |
| M16 | **green** | Odyssey dry-run validation passes |
| M17 | **green** | `ODYSSEY_LAUNCH_AUTHORIZED` remains false |
| M01 | running | traversal 70/282, W004, 0 faults, eviction now firing |
| M04 | running | adapter green on fixture; flagship artifact pending |
| M05 | running | base measured on the instrument; GLM numbers pending |
| M10 | running | plans byte-matched; retention deliberately null |
| M12 | running | deterministic half sealed; two metrics gated on a served model |
| M13 | running | training bundle complete; substrate gated on M11 |
| M18 | running | eviction verified firing |
| M02 M03 M06 M07 M08 M11 M19 M20 | open | all downstream of the traversal |

---

## Exact next steps

1. The traversal completes to 282/282 (about five hours at the current rate, detached and
   self-sustaining now that eviction fires).
2. Assemble one complete GLM `.gravity` artifact under
   `~/Library/Application Support/Hawking/Models/GLM-5.2/<revision>/`. The packer
   currently writes to `~/Desktop/GLM52-Gravity-SubBit`, which §3.2 forbids as a final
   location.
3. Run the GLM adapter against the flagship artifact and the numpy oracle at scale.
4. Walk the parity ladder H09 → H04, screening gate first, and seal the lowest passing
   rate.
5. ~~Implement `Engine` for the gravity runtime.~~ **Done.** The GPU model now owns its
   Metal context and locks a `Mutex`, so it is `Send + Sync`; `load_engine` dispatches on
   container magic and streams tokens through the reviewed registry. A compile-time
   assertion fails if `Send + Sync` ever regresses.
6. Give the gravity engine a batch-slot path. `hawking serve --weights <.gravity>` starts,
   loads through the registry, and `/v1/models` correctly reports the artifact's own model
   id. Generation does not work yet, and the reason is exact rather than vague: every
   serve path is continuous-batching, so it needs `encode_prompt_for_batch` **and**
   `prefill_slot`, and `prefill_slot` plants per-layer KV into the multiseq arena while the
   gravity runtime keeps its KV in its own per-layer device buffers. That is a real
   integration, not a shim.

   The trait documents a fallback to plain `generate` for engines without batching. The
   server does not implement that fallback — it errors at admit. Worth noting because
   implementing only `encode_prompt_for_batch` makes it *worse*: the request is admitted,
   `prefill_slot` fails, and the caller gets an empty completion with a 200. That was
   observed and backed out; claiming half a capability is worse than claiming none.

   HIDE itself lives on `build/hide-impl-2026-07-19`, not in this worktree.
7. Claim A at equal bytes, using the solved rates already sealed in
   `PROMETHEUS_ARCHITECTURE.json`.

## Next-chat launch command

Odyssey remains fenced. Nothing in this session authorized it, and nothing in the package
can. When the substrate exists and the fence is deliberately opened:

```bash
printf 'true\n' > odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED && python3.12 odyssey/training/run.py T0
```
