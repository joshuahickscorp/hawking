# QWEN TEXTBOOK V1

The first Odyssey specimen, written down as science a later model can inherit.

**Specimen.** `huihui-ai/Huihui-Qwen3.8-27B-abliterated`, bf16, 26895998464 [receipts/headless/WHOLE_MODEL_NATIVE.json#parent_params] parameters, registered as `qwen38-huihui-bf16-P0` with structural identity `c72e6dbb668f33b8c3a764d22d5a3b4c33f9940b3a2479a0fdde17c12f26553a` [receipts/headless/MODEL_REGISTRY.json#candidates.qwen38-huihui-bf16-P0.artifact.identity].

Every quantitative claim below carries an inline citation of the form
`[receipt#json.path]`. `tools/headless/textbook_trace.py` opens each receipt, resolves
each path, and exits non-zero on any claim that is untraceable or disagrees. A number in
this document that the checker cannot confirm is a bug in this document.

**What this is not.** It is not a campaign log, and it is not a list of everything that
happened. It is only the part that transfers. Where a result is specific to this
checkpoint or this machine, it says so, because inheriting a local number as a law is the
failure this document exists to prevent.

---

## 1. ArchitectureGenome

Dense hybrid transformer: GQA attention, a DeltaNet recurrent organ, SwiGLU MLP, RMSNorm,
BPE tokenizer, untied head. The organ set the compiler must serve is recorded in
`receipts/headless/ORGAN_LIBRARY.json` and reachable through
`tools/headless/organ_library.py`.

The whole model compiles to a heterogeneous body: not one bit rate, but a per-organ
allocation. That is the single most important structural fact in this textbook, and
section 4 gives the physical reason for it.

## 2. The MLP information floor

**2.25 bpw [receipts/headless/BYTES_FRONTIER.json#baseline.active_bpw], coordinate-robust under the tested transforms.**

The MLP body reaches 2.25 bpw [receipts/headless/BYTES_FRONTIER.json#representations.0.active_bpw] with a four-level fitted affine codec at group 64 (`q2f_g64`) and stays coherent there. Below that, on this organ, nothing tested composed
coherently. Four structurally independent attacks agree:

| attack | result |
|---|---|
| binary at ~1.25 bpw | physically fast, generation-injured [receipts/headless/ONEBIT_FAMILIES.json#verdict.decision] |
| protected islands healing | 0 [receipts/headless/BINARY_HEALING.json#finding.n_that_reached_coherent_generation] of 4 candidates reached coherent generation |
| shared basis | kernel competent, density dead below the floor [receipts/headless/SHARED_BASIS_COHERENT.json#finding.reason] |
| low-rank / shared-K hybrid | no correction under the budget restored held-out activations [receipts/headless/HYBRID_OPERATOR.json#finding.reason] |

A coordinate-transform reopening did not move it
[receipts/headless/COORDINATE_TRANSFORM_PROBE.json#ROTATION_MOVES_BARRIER].

**Inheritance rule.** 2.25 is a Qwen number until a second model measures its own. What
transfers is not the value: it is the *method* — fit four levels by least squares per
group, hold group 64, and test coherence on held-out real activations rather than on
weight-space error.

## 3. Other organ floors

Measured per organ, taking the cheapest candidate that still survives held-out activations:

| organ | floor (complete EBPW) | codec |
|---|---|---|
| mlp_gate_up / mlp_down | 2.25 | q2f_g64 |
| gqa_attention | 3.126550820024315 [receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.gqa_attention.candidates.5.complete_ebpw] | ws_rtn_q3_g128 |
| deltanet | 3.1365379214627285 [receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.deltanet.candidates.5.complete_ebpw] | ws_rtn_q3_g128 |
| embedding / output | 3.125 [receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.embedding_output.candidates.3.complete_ebpw] | ws_rtn_q3_g128 |

## 4. The interaction law that makes heterogeneity necessary

**Organs do not share a floor, and the discontinuity is in the same place for three of
them.**

Attention, the recurrent organ, and embedding/output all survive held-out activations at
q3_g128 and all fail at q2f_g64 — the same codec the MLP survives:

- gqa_attention survives at 3.126550820024315 [receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.gqa_attention.candidates.5.complete_ebpw] and fails at 2.2515978145705065 [receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.gqa_attention.candidates.6.complete_ebpw]
- deltanet survives at 3.1365379214627285 [receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.deltanet.candidates.5.complete_ebpw] and fails at 2.2618875554464477 [receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.deltanet.candidates.6.complete_ebpw]
- embedding/output survives at 3.125 [receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.embedding_output.candidates.3.complete_ebpw] and fails at 2.25 [receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.embedding_output.candidates.4.complete_ebpw]

The MLP crosses that boundary and the others do not. A uniform bit rate therefore either
wastes bits on the MLP or breaks everything else. This is why the winning body is
heterogeneous, and it is the strongest candidate in this textbook for a law that
generalizes: **test the floor per organ, never per model.**

## 5. The whole-model executable

Complete 2.5969567265364937 EBPW [receipts/headless/WHOLE_MODEL_NATIVE.json#compile.complete_ebpw],
active 2.4492357186951987 EBPW per token [receipts/headless/WHOLE_MODEL_NATIVE.json#compile.active_ebpw_per_token],
8234330016 [receipts/headless/WHOLE_MODEL_NATIVE.json#compile.active_bytes_per_token] active bytes per token,
964 [receipts/headless/WHOLE_MODEL_NATIVE.json#decode.dispatches_per_token] dispatches per token,
median 30171291 [receipts/headless/WHOLE_MODEL_NATIVE.json#decode.median_gpu_ns_per_token] GPU ns per token.
Coherent: 16 [receipts/headless/WHOLE_MODEL_NATIVE.json#decode.coherence.n_unique_ids] unique token ids over 16 new tokens.

**This body is NOT capability-qualified, and the word "coherent" above was earned by a
gate that could not fail.** The coherence check behind it is 16 tokens with
`n_unique_ids > 2`. The body still looks fine at 16 tokens and degenerates at roughly 30.
Scored on the full capability suite against the artifact production actually runs, it
takes 0 [receipts/headless/QWEN_CAPABILITY_QUALIFICATION.json#results.3.overall.passed] of 43.
Its first score was 3, and all three of those points were the hygiene axis — a
`must_not_contain("</think>")` check that this body passes because it NEVER EMITS
`</think>`: it never leaves the reasoning block, so a "do not leak your reasoning" test
cannot fail on it. Scoring an unterminated think block as no reply at all takes it to
zero. On "What is 17 + 4?" it emits two tokens, both newline.

The 3.1393-EBPW body, built from the same parent by the same binary with the same
tokenizer, answers `21` and writes a correct code block, and scores 27
[receipts/headless/QWEN_CAPABILITY_QUALIFICATION.json#results.2.overall.passed] of 43.

So the density figures in this section are real and the executable they describe does not
work. Every organ in it passed its own held-out probe; the composition did not. That is
the most important thing in this textbook, and it is why §4's per-organ law is stated as
necessary rather than sufficient: **local adequacy does not compose.**

**Reproduction caveat.** The artifact these numbers were first measured on no longer
exists: the builder defaults its artifact root inside the repository and the original run
happened in a temporary worktree that was later removed. The figures above are from the
clean-room rebuild, which is reproducible.
See `receipts/headless/QWEN_CLEAN_REBUILD.json#closure_gaps`.

## 6. Machine laws (this box, not the architecture)

- Device theoretical roof 819.0 GB/s [receipts/headless/WHOLE_MODEL_NATIVE.json#three_roofs.DEVICE_THEORETICAL]
- Device measured sustained roof 778.8 GB/s [receipts/headless/BANDWIDTH_ROOF.json#anchor_roof.correction.new_roof_gb_s]
- Model-reachable roof 690.8064962384027 GB/s [receipts/headless/WHOLE_MODEL_NATIVE.json#three_roofs.MODEL_REACHABLE]

**The model-reachable roof is a property of this executable in this regime, not of the
machine.** It changes when the representation changes. It must never be copied into
another model's ledger. The device roofs are machine constants and may be reused on this
box only.

Current achievement against that roof: 272.22650823855514 GB/s [receipts/headless/WHOLE_MODEL_NATIVE.json#three_roofs.current_achieved_gb_s], a fraction of 0.39407056783757816 [receipts/headless/WHOLE_MODEL_NATIVE.json#three_roofs.current_fraction_of_model_reachable].
The executable is not bandwidth-saturated; the remaining gap is execution, not density.

## 7. Structural elimination

Nothing was eliminable in this architecture, and the reason is measurable:

Q heads are near-orthogonal, mean cosine 0.04382772212914319
[receipts/headless/STRUCTURAL_ELIMINATION.json#attention_heads.headline.q_mean_cosine_all_layers], K/V/O likewise;
there are no dead MLP channels and no near-identity layers
[receipts/headless/STRUCTURAL_ELIMINATION.json#verdict.one_line].

**Inheritance rule.** Head sharing is refuted *here* because the heads are orthogonal
*here*. The transferable artifact is the measurement — compute head cosine before
proposing head removal, on every new model.

## 8. Representation genome

`q2f_g64` (four-level fitted affine, group 64) on the MLP; `ws_rtn_q3_g128` (grouped
absmax RTN) on attention, DeltaNet and embedding/output. The full family space, including
every family still UNTESTED here, is in `receipts/headless/REPRESENTATION_LIBRARY.json`,
queryable through `tools/headless/representation_library.py`.

**Permanent law, measured:** fewer stored bits is not fewer nanoseconds. A representation
must be evaluated with a competent kernel or its density number means nothing. The shared
basis is the proof in both directions — its density did not survive, and yet a competent
fused kernel dropped its cost dramatically [receipts/headless/SHARED_BASIS_KERNEL.json#finding.reason].

## 9. Kernel genome

17 [receipts/headless/KERNEL_LIBRARY.json#n_kernels] qualified kernels, all complete under
the field-completeness checker, in `receipts/headless/KERNEL_LIBRARY.json`. Graph-collapsed
operators are tracked separately in `receipts/headless/SUPEROPERATOR_LIBRARY.json`.

**Refuted, do not re-run:** one universal megakernel. An 8-layer f16 fused kernel measured
4.4x slower than the unfused sequence. Build operator families with evidence, not one
monolith.

**Caveat on the correctness contract.** 9 [receipts/headless/KERNEL_LIBRARY.json#n_kernels_without_a_runnable_contract]
of the 17 kernels have no runnable parity binary in this repository. Their correctness is
undemonstrated, not assumed.

## 10. Negative science

40 [receipts/headless/NOETIC_NEGATIVE_SCIENCE.json#counts.total] recorded failures, every
one at MODEL_SPECIFIC level, none promoted:
0 [receipts/headless/NOETIC_NEGATIVE_SCIENCE.json#counts.by_level.GENERAL_PHYSICAL] general-physical.
Each carries model, organ, technique, representation, kernel, machine, capability, physical
reason and a reopening condition. Query it before designing an experiment.

**Promotion law.** A single model's failure never prunes a technique anywhere. The store
mechanically refuses a promotion that lacks independent measurements on distinct models.

## 11. Doctor prescriptions that came out of this specimen

1. Measure the floor per organ, not per model. Section 4 is why.
2. Test coherence on held-out real activations. Weight-space error ranked candidates
   wrongly on this specimen at every rate.
3. Evaluate a representation only with a competent kernel (section 8).
4. Compute head cosine before proposing head elimination (section 7).
5. When a body is fast and injured, measure how broad the injury is before spending on
   healing — here it was uniform, and small islands could not pay for it.

## 12. What a future model should NOT inherit

- The value 2.25. Inherit the method, measure the value.
- The model-reachable roof. It is per-executable.
- The refutations of binary, shared basis, low-rank healing, shared-K, rotations, head
  removal, state merging, and binary-as-draft. All are MODEL_SPECIFIC. They rank a search;
  they do not close one.

## 13. Final executable recipe

MLP gate/up/down on all layers at `q2f_g64`; attention, DeltaNet and embedding/output at
`ws_rtn_q3_g128`; native GEMV kernels with in-register dequantization and no dense weight
materialization; fused gate_up_swiglu where the intermediate is not observable. Built by
`tools/headless/whole_model_native.py`. Reproduction status and its open gaps:
`receipts/headless/QWEN_CLEAN_REBUILD.json`.
