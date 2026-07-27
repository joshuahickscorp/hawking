# Hawking Motherload Completion — Session Report

```text
endpoint:  IN_PROGRESS  (not HAWKING_ODYSSEY_READY)
gates:     9 / 20 closed
position:  the flagship traversal, assembly, and adapter proof are DONE.
           the open question is whether 0.883 BPW preserves capability —
           measurement corrected mid-session, verdict pending.
```

The GLM-5.2 source traversal completed 282/282 partway through this session. What used to
be the hard dependency blocking eleven gates is now itself resolved into three separate,
independently-answered questions: does the runtime execute the artifact correctly (yes,
sealed), is the artifact's bytes complete (yes, sealed), and does the artifact preserve
capability at the rate it was packed (open — see below, including a correction made
in-flight).

---

## What is finished and verified

### The complete local GLM-5.2 artifact exists

282/282 shards traversed, 0 faults across the entire run. Assembled to
`~/Library/Application Support/Hawking/Models/GLM-5.2/<revision>/General-R0/` per §3.2 —
not the packer's working directory, which the campaign forbids as a final location.
Coverage graded against the *official* 59,585-tensor index at the pinned revision, not
against the packer's own output:

```text
COMPACT_PAYLOAD             59,003
PROTECTED_NATIVE_PAYLOAD       582
MISSING / UNDECLARED / MISPLACED    0 / 0 / 0

physical size        83.14 GB
whole-model BPW        0.882888   (753,329,940,480 logical elements, exact)
```

Shards are hardlinked from the packer's output, not copied — confirmed 2 links on disk,
so this is the one 83 GB, not a second copy of it.

### The `.gravity` runtime executes a complete token — proven on the real artifact, not a proxy

Three independent decoders agree on what a `gravity-pq` payload means: the numpy
reference that defines the codec, a Rust CPU path, and a Metal kernel. That was true on
fixtures from early in the session; what's new is that it's now proven on the actual
744B-parameter flagship:

```text
Rust adapter vs. independent numpy oracle, same 282 real shards, 3 tokens, 78 layers:

  argmax                9540 both sides
  top5                  [9540, 86755, 67480, 112441, 122609] both sides, same order
  final DSA selection   [1, 2, 0] both sides, same order
  max |logit diff|      6.08e-05 across 154,880 outputs (float32 reassociation noise,
                         scales correctly from the fixture's 2.6e-6 over 4 layers)
```

Getting a Rust adapter to run this at all needed a real fix, not just a bigger machine:
eager whole-model decode (what worked fine for a 135 MB Llama fixture) would need
150+ GB of RAM for this artifact, because MoE sparsity means only 8 of 256 experts
activate per layer but eager decode would have processed all of them, and PQ indices
widen to roughly 2x their packed size once decoded. `GravityWeights::open_dir` now
indexes which shard owns which tensor and decodes nothing until asked; shards open
lazily and stay open. **Peak resident memory: 6.9 GB.**

The Python oracle got the same fix in the same session, for the same reason:
`GravityGlmSource` does a real matvec against packed bytes (`pq_execute(artifact, x)`)
instead of the fixture-era approach of densifying a full `[rows, cols]` array one
one-hot column at a time — which would have needed 1,872 tensor reads, each fully
densified, for a single token.

### Measured base throughput — `BASE_TRUE_TPS`, Llama instrument, no acceleration

```text
ctx    128    prefill 116.5    decode 105.8 tok/s    ttft 47.1 ms
ctx    512    prefill  92.8    decode  68.8 tok/s    ttft  8.7 ms
ctx   2048    prefill  48.8    decode  29.2 tok/s    ttft 18.9 ms
ctx   8192    prefill  19.1    decode  13.3 tok/s    ttft 59.4 ms

cold load 680 ms · warm 368 ms · 135.7 MB resident vs a 135.6 MB artifact
1 command buffer and 210 dispatches per token at every context
```

Started at 34.6 tok/s at ctx 128 with 65 command buffers per token; collapsing the token
into one command buffer gave 3.2x, moving attention onto the device fixed the decay with
context. Incremental decode is bit-identical to replaying the prefix — 0.0 difference.

GLM has no GPU-resident path yet (CPU-only, lazy: ~30–140 s/token depending on which of
the numpy/Rust paths). That is a measured, real number, not a placeholder — but it is a
correctness measurement, not what M05 asks for GLM specifically. A GPU-resident GLM
adapter (Metal kernels for MLA/DSA/256-expert MoE dispatch) is unbuilt and is the actual
size of the remaining M05/M07 work, not a small gap.

### The production path is closed, and stops one honest step short of serving

`load_engine` dispatches on the container's magic bytes, so a `.gravity` artifact reaches
`hawking-core` through the same reviewed registry as every other architecture, and
streams tokens out end to end for the Llama instrument. `hawking serve --weights
<.gravity>` starts and `/v1/models` correctly reports the artifact's own model id.
Generation does not work yet: every serve path is continuous-batching and needs
`prefill_slot`, which plants per-layer KV into a shared multiseq arena, while the gravity
runtime keeps KV in its own per-layer device buffers — a real integration, not a shim.
Implementing only half of the needed trait methods was tried and reverted, because it
turns a loud, correct refusal into a silent empty completion with a 200 status.

### GLM-5.2 architecture adapter: MLA, DSA, IndexShare, `noaux_tc` router, MoE

Agrees with the numpy oracle to 3.84e-06 on a fixture carrying the flagship's exact
semantics (a layer that must reuse the previous layer's DSA index rather than
recompute it), and — see above — to 6.08e-05 on the real 77 GB artifact itself.

### Prometheus — all 14 Revision 3 §7 components, deterministic spine complete

Eight measure, five are gated with named gates (all correctly requiring a *served*
model — S0.8 intervention probes, cartography membership — which the CPU-only GLM
forward cannot practically supply; re-running after the artifact was assembled produced
byte-identical output, confirming these stages never depended on artifact existence,
only on serving speed).

The equal-budget check found a live defect before it could bias a result: Math, General
and Random land on 46.70 GB, but Uniform as originally written was 3.56 GB *lighter* —
which would have hand Uniform an unearned disadvantage. Solved as exact rationals; all
four arms now match to 0.0175%. **Uniform needs 2.53 bits per weight to spend what the
conditioned arms spend at 1.0 with a natively-carried embedding and head.**

`GLM52_LOGICAL_WEIGHT_LEDGER.json`'s tensor denominator (753,329,940,480) and source-byte
total (1,506,659,919,872) both match the real assembled artifact and the official
manifest exactly — cross-validated from three independent sources.

### Sovereignty sealed for both artifacts that exist

The Llama instrument (single shard) and now the GLM flagship (282 shards) — which needed
a real extension, not a workaround: a single shard's `body_sha256` doesn't name a
282-shard model, so `build_continuity_manifest_multi_shard` hashes the assembler's own
manifest (every tensor's owning shard, the synthesized architecture, the coverage
verdict) instead, and refuses to seal if that manifest's coverage isn't `COMPLETE` —
proven with a synthetic incomplete fixture. `hidden_intervention_rate` 0.0,
`model_continuity_rate` 1.0, `attribution_completeness` 1.0 for both. `false_refusal_rate`
and `boundary_error_rate` remain correctly gated on a served model.

### Model Odyssey — prepared, fenced, not started

86 dry-run checks, zero failures. `ODYSSEY_LAUNCH_AUTHORIZED` is `false`, the builder
reads that file and never writes it, and the selftest proves both directions: rebuilding
leaves it false, and flipping it to true makes validation **fail**. Lean `v4.15.0` and
Mathlib `v4.15.0` pinned to concrete revisions; `latest` is rejected.

---

## The screening gate: a real result, then a real correction

### The Llama-3.2-1B instrument fails outright — and bounds nothing about the flagship

```text
prompt      "The capital of France is"
continues   " settle settle settle settle settle ..."
oracle top5  settled, settles, settle, ewise, booster
```

The numpy oracle agreed this is the artifact's real behavior, not a runtime defect. A
dense 1B model has little redundancy to spend; a 744B MoE has expert topology, which is
the redundancy the whole compression program exists to exploit. This result never
claimed to bound the flagship, and didn't.

### The flagship's first screening run also collapsed — on a malformed prompt

Running the same style of test on the real 77 GB artifact, greedy-decoding "The capital
of France is" as raw text produced the identical token three times running:
`" Fired Fired Fired"` (token id 74242). Decisive-looking by the parity contract's own
rule (degenerate repetition ends evaluation), and a FAIL verdict was sealed to M03 and
committed.

**That verdict was wrong, and was retracted before further action was taken on it.**
GLM-5.2 is chat/instruction-tuned. Its own pinned `chat_template.jinja` requires every
input to open with `[gMASK]<sop>`, wrap user content in `<|user|>...<|assistant|>` role
markers, and by default insert a reasoning-effort system turn before generation — none of
which the first run's raw-text prompt included. A chat model fed a bare completion is out
of distribution for reasons that have nothing to do with how many bits its weights carry.
The retraction is recorded in `HAWKING_MODEL_FEEL_PARITY_RESULTS.json` alongside the
original entry, not silently overwritten.

The screening driver now renders through the model's own template via jinja2
(`add_generation_prompt=True, enable_thinking=False`), with special tokens verified
atomic in the tokenizer's vocab (`[gMASK]`, `<sop>`, `<|user|>`, `<|assistant|>`,
`<think>`, `</think>` all come back as single ids in the 154822–154842 range, not
sub-word-split) before trusting the render.

### Corrected run: in progress at time of writing

First token of the properly-formatted run was `' brisk'` — not the immediate exact-repeat
that the malformed prompt produced, which is itself informative (the template mattered),
but one token is not a verdict. The run is mid-flight; **M03 has no sealed rate verdict
as of this report.** The next session should read the completed
`HAWKING_MODEL_FEEL_PARITY_RESULTS.json` before doing anything else rate-related, and
either seal M03 on the corrected result or continue the greedy run if it was interrupted.

### The traversal was three windows from a disk stall, silently

Separately: the eviction gate had authorized nothing since the run began. Every fetched
shard was being retained, the deferred set grew 12 → 22 → 33, and the controller reported
`RUNNING` with zero faults throughout — silence looked exactly like health. The gate was
right to refuse: every teacher capsule on disk was sealed against a retired 8-token
calibration. Capsules were archived with a withdrawal receipt (not deleted) and the chain
re-seeded from layer 0. Result: `0/33 → 33/33` shards authorized, 130 GiB freed, and the
traversal ran to completion afterward without further intervention.

---

## Gate status

| gate | state | condition |
|---|---|---|
| M01 | **green** | traversal 282/282, 0 faults |
| M02 | **green** | complete local GLM artifact assembled, 0 coverage defects |
| M04 | **green** | adapter proven on the real flagship, not just a fixture |
| M09 | **green** | Prometheus architecture, all 14 components |
| M12 | **green** | sovereignty sealed for both artifacts |
| M14 | **green** | Odyssey sandbox/roles/Ledger/Tribunal/retrieval |
| M15 | **green** | Lean/Mathlib pinned |
| M16 | **green** | Odyssey dry-run: 86/86 |
| M17 | **green** | launch fence false, provably |
| M03 | running | screening gate corrected mid-session; result pending |
| M05 | running | Llama instrument measured; GLM needs a GPU adapter that doesn't exist yet |
| M10 | running | plans byte-matched; retention needs a served model |
| M13 | running | training bundle complete; substrate selection gated on M11 |
| M18 | running | eviction verified firing; traversal completed cleanly after |
| M06 M07 M08 M11 M19 M20 | open | downstream of M03's verdict and a GPU GLM path |

---

## Exact next steps, in order

1. **Read the corrected screening result.** Check
   `HAWKING_MODEL_FEEL_PARITY_RESULTS.json` and
   `/private/tmp/.../scratchpad/screening_result_v2.json` (or re-run
   `tools/condense/glm52_flagship_screening.py` if the process didn't survive session
   end — it was detached but this machine's scratch dir is session-scoped).
2. **Seal M03** on whatever that result honestly shows. If it passes: R0/0.883 BPW is the
   lowest rate tested and is the selected General artifact unless a lower pilot is worth
   trying. If it fails: per the campaign's own contingency, do not abandon — bracket
   upward (H10/H12/H15). Re-packing a higher rate from R0's already-lossy representation
   would compound error rather than match packing from BF16 directly, so this needs
   source bytes beyond what's resident; a targeted pilot (not a blind full 1.5 TB
   re-fetch) is the responsible first move, consistent with the campaign's own "pilot
   windows... targeted source refetch" language.
3. **Build a GPU-resident GLM adapter** (Metal kernels for MLA + DSA + 256-expert MoE
   dispatch, mirroring what exists for Llama). This is the real remaining size of M05 and
   M07 — not a small integration, comparable in scope to everything built for the Llama
   GPU path this session, but with genuine added complexity from expert routing.
4. **Finish HIDE's `prefill_slot`** for the gravity engine once a GPU path exists to
   serve.
5. **Claim A at equal bytes**, using the solved rates already sealed in
   `PROMETHEUS_ARCHITECTURE.json`, once a served model can run the S0.8 intervention
   probes fast enough to be practical.

## Next-chat launch command

Odyssey remains fenced. Nothing in this session authorized it, and nothing in the
package can. When the substrate exists and the fence is deliberately opened:

```bash
printf 'true\n' > odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED && python3.12 odyssey/training/run.py T0
```
