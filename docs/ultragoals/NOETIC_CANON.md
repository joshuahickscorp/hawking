# NOETIC CANON

Durable doctrine distilled from steers S001-S018 of the Noetic one-bit ascension
campaign and its parent (Hawking headless completion). Canonization here means the
laws are made permanent, not that the steer log is deleted (S013: "Canonization is
done by VERIFYING, not by deleting"). Each law below was earned by a measured
failure; the failure is kept beside the law because a law without its scar gets
deleted by the next person.

Source of record: `~/.claude/ultragoal/hawking-noetic-onebit/STEERS.md`.

---

## I. Representation and information

**1. The question is what information must exist, not how to quantize.** (S017 §1)
Do not build a "1-bit quantizer". `W in {-1,+1}` or sign+scale is one baseline among
many. The frontier may need residual islands, shared bases, structural codebooks,
routed coefficients, cross-layer dictionaries, generated coefficients, sparse
exceptions, protected high-precision islands, recurrent state, or fitted operators.
A component may cost 0 bits/weight locally and another 4 bits; global physical
information is what is scored.

**2. EBPW is measured from the full executable closure, with nothing hidden.**
(S017 §39) `EBPW = 8 * MODEL_SPECIFIC_BYTES / PARENT_PARAMETER_COUNT`, parent params
= 26,895,998,464. No model-specific bytes may hide in shader constants, generated
source, routing tables, caches, runtime blobs, MLX arrays, or external files. The
information accounting scans all byte alignments and counts embedded weight-like
bytes; an 18-byte header once defeated a naive scan.

**3. Storage, active-fused and active-cached BPW are three different answers to the
same structure.** (S017 §7) `ACTIVE_BYTES_PER_TOKEN` and `DRAM_BYTES_PER_TOKEN` rank
above stored size. A representation storing 1 EBPW but reading 5 EBPW/token can be
worse than one storing 2 and reading 1.2. Doctor optimizes `useful function /
(active bytes * token_ns)`.

**4. Local adequacy does not compose.** (S017 §28; parent campaign) Cosine is
scale-invariant, so `0.01*W` scores 1.000000 on every axis — use a scale-aware
metric. As density drops, local cosine stops meaning anything. Qualification is a
ladder: local functional probe -> held-out activation -> adjacent layers -> short
chain -> complete organ -> complete token -> coherent generation -> capability. A
candidate that fails a rung is KILLED there; UNREACHED and FAILED are different
states and may not be conflated. Ternary at 1.85 bpw was named CANON on an
organ-local screen and then flipped the whole-model argmax (10895 vs 9714). A screen
verdict is not a model verdict.

**5. The floors are tradeable.** (S017 §26, §27) Per-organ quantization floors
(measured: DeltaNet 4.125, GQA 4.25, embed/output 4.125, MLP 2.50) describe the
UNIFORM allocation only. At equal total bytes a global allocator beats uniform
(0.8256 vs 0.7247 weighted scale-aware, span 1.250-4.367 bpw). Allocate globally:
one organ may get 0.3 EBPW and another 4 if the global result is better. Some tiny
fraction may need protected high-precision islands; that is fine.

---

## II. Kernel and execution

**6. A representation cannot be condemned until its native kernel is competent.**
(S017 §5) The 2-bit affine kernel looked slower than q4 while moving FEWER bytes,
because a bind-time `group_size` put a non-constant integer divide on every tile.
Specializing it moved unfused decode 26.84 -> 32.84 tok/s with the representation
unchanged. Before recording any low-density candidate REFUTED on speed, screen its
kernel for the illusion-producing defects: runtime integer divide, runtime-sized
loop, data-dependent inner branch, bind-time shape param. The screen is necessary,
not sufficient — a kernel can pass every check and still be slow for an unnamed
reason, and that limit is reported, not hidden.

**7. Specialize on what is actually static; do not explode variants.** (S017 §16,
§17) Group size, hidden dim, rank, codebook size, tile geometry, representation type.
A compile-time constant divide is a shift. Do not reuse q4 tile geometry for a 2-bit
or structured operator — it may want radically different geometry. Use measured
specialization value, not genericity that produces divides and branches.

**8. Fusion is not automatically a win.** (S017 §13; parent campaign) The
gate_up_swiglu fusion proved graph-level fusion is valuable, but an 8-layer fused
megakernel measured 4.4x SLOWER. Search graph motifs the source framework only
describes separately (norm+proj, proj+activation, gate+up+SwiGLU, route+lookup+accum,
low-rank+sparse correction, QKV combos, dequant+matvec), fuse to fewer dispatches,
and MEASURE each — the megakernel negative stands as evidence.

**9. Dispatch annihilation is measured, not assumed.** (S017 §14) 756 is not assumed
close to optimal. A DISPATCH_LEDGER names every dispatch per token with operator,
bytes, FLOPs, launch overhead, dependencies, fusion candidacy and frequency, ranked
by (overhead + memory traffic + synchronization). Measured: residual+RMSNorm fusion
cut 756 -> 628 with token ids unchanged and parity 0.

---

## III. The roof (S018)

**10. The machine's roof is 819 GB/s spec peak, and it must be measured, not
inherited.** This box is an Apple M3 Ultra, 96 GB, 60 GPU cores, spec peak memory
bandwidth 819 GB/s. The campaign reported every roofline conclusion against
`ANCHOR_ROOF_GB_S = 595.9` (72.8% of spec), a constant with no provenance in this
campaign. Traced: `G072_MULTI_PLANE_GEMV.json` set 595.9 as the reference "these
kernels are scored against" for ONE family
(`qwen_binary_planes_k{1,2,3}_matvec_geo_tpr64_tg128`); `Genesis.m3ultra.nx` promoted
it to `measured_roof_gb_s` under the machine genome; three design files hardcoded it;
later receipts called it "the measured ceiling of the hardware". A kernel family's
scoring reference became the machine's roof across three honest hops. This is Law 6
one level up: a family's kernel was not competent, so its peak was mistaken for the
limit.

**11. There is no hardware bandwidth counter on this box.** `MTLDevice.counterSets`
contains only `timestamp`/`GPUTimestamp`; `DRAM_READ_BYTES` is ABSENT. Every GB/s
figure in the campaign (468.9, 289.9, 231.5, and 595.9 itself) is
bytes-believed-read over GPU time, not a counter. A roof claim must be settled by a
microbenchmark whose byte count is true by construction, never by dividing believed
bytes. Production bar: 775 GB/s minimum on the decode path; if unreachable, prove the
wall by measurement and name the limiting factor — do not assert it.

---

## IV. Parallelism and production

**12. One immutable resident body, many sessions.** (S017 §9-§12) Do not load
multiple full model copies to get concurrency. Share representation, kernels, Metal
pipelines, codebooks, static routing; isolate context, KV/state, sampling, session
state. Measured: Metal working set for one shared body vs four copies is 1.0398 vs
57.96 GB. Concurrency ceiling on this class of body is ~1.32x regardless of topology.

**13. The production optimum is verified accepted useful work per wall second — not
tokens, not stream count.** (S017 §8, §32, §44) Measured and decisive: at c=4 the q4
incumbent wins at 669.2 verified WU/hour against the leader's 491.5, DESPITE the
leader's higher aggregate tok/s (29.03 vs 26.60). A config with lower tok/s wins when
it truncates less and reaches its closing think-tag inside the token budget. Verified
WUs must be measured through a deterministic content verifier (parse the JSON, parse
the Python with ast, reject anything still inside `<think>`), never derived from
tok/s; a token-generator negative control must be rejected.

**14. State can exceed the weights before the weights are the problem.** (S017 §34,
§35) Per-session workspace measured 192 MB at seq 256 (DeltaNet state 156.9 MB, GQA
KV 33.6 MB) and grows with context while the shared body does not. At 32K x 4
sessions, session state (16.59 GiB) exceeds MODEL_BYTES (13.32 GiB). Track production
footprint as MODEL_BYTES + SESSION_STATE_BYTES. Prefill is a separate question from
decode (§35): a decode-tuned representation can have poor prefill, and long-context
TTFT must not be sacrificed for decode tok/s. DeltaNet recurrent state is NOT
prefix-shareable across diverged suffixes — it is a summary, not a cache.

---

## V. Method (the laws that keep the science honest)

**15. Causal benchmark law: a benchmark a no-op would also pass is invalid.** (S017
§37) Every GPU/speed claim carries proof the changed thing ran: kernel identity,
dispatch count, a candidate-specific sentinel, a no-op control that must NOT score,
and a deliberately-bad control that must be rejected. This is not theoretical — a
self-optimization harness once compared two arms running identical code because
sys.path ordering made the inspected file differ from the executed module; its
REFUSED verdicts were vacuous.

**16. Enough reps to separate the arms, or report NOT SEPARATED.** A dispatch lane
reported +12.45% tok/s from 2 reps per arm where the deliberately-broken control
posted the FASTEST number of the three, under live GPU contention from a second 27B.
Use >=7 reps, report min/median/max, and if the arms' ranges overlap say they are not
separated rather than quoting a mean delta. Report cold and warm separately; a single
Metal run is page-cache confounded and a tight spread is itself evidence.

**17. An adversary is required for every frontier claim.** (S017 §38) The adversary
has refuted real claims: a tok/s outside its own recorded band with a hardcoded
`dense_w_materialized=0`; a "468.9 GB/s DRAM" figure that a non-streaming kernel would
report identically; an organ floor that charged lm_head at the wrong rate. Anything
unmeasured is ABSENT with a reason, never an estimate presented as a number. Label
measurements honestly (DIRTY_ENGINEERING vs CLEAN_CANDIDATE).

**18. Promotion is not discovery.** (S017 §43) A frontier candidate is not a resident
model. The 3.1393-EBPW leader (NOETIC_PARENT_A, closure
7921a6a27e0561...) is sealed and immutable; all density work happens in disposable
children. ~1 EBPW is a research pressure, not a promotion condition and not a floor.

**19. Stop law: continue while the frontier moves; stop Goodharting the number when
it does not.** (S017 §45) If ~1 EBPW proves incoherent after multiple structurally
distinct families, record the frontier (LOWEST_SCREEN_SURVIVOR,
LOWEST_CHAIN_SURVIVOR, LOWEST_GENERATION_COHERENT, LOWEST_CAPABILITY_SURVIVOR,
FASTEST_COHERENT, FASTEST_PRODUCTION — which may be different artifacts) and stop.
1.37 EBPW being the coherent frontier is a valid scientific result.

---

## VI. Operational (parent campaign, S001-S016)

**20. Expand the platform before self-optimizing it, and gate self-optimization
behind a measured trust threshold.** (S002, S005) The stop law: after each tranche,
ask whether it materially increases capability before continuing.

**21. Census before scope; verify before delete.** (S013, S014) A directory named for
deletion is inspected first (`git log --diff-filter=D` when an audit says ABSENT). The
GGUF is deletable only after its replacement is VERIFIED. Canonization is verifying,
not deleting.

**22. Saturate Grok across an independent frontier; serial is the waste.** (S003,
parent) Maintain a concurrent frontier of independent audits/lanes rather than one
serial chain. Headroom is measured (load vs core count and free disk), not assumed.
Lanes finish uncommitted and sparse — preserve-then-cleanup always, three-way merge
modifications against the lane base, never `git add -A` (it once tried to add the
1.6 GB nested visionmcp repo).

---

## VII. Genome libraries (S023 §12, §13, §14, §34, §70)

Originally generated by the retired `tools/headless/genome_libraries.py` producer
from sealed receipts. Not
hand-authored. Distillation path: raw experiment → receipt → genome → organ
law → architecture law → canon. Every law below cites its evidence.

**23. Organ, kernel and representation knowledge lives in generated libraries.**
(S023 §12–14, §70) `receipts/headless/ORGAN_LIBRARY.json`,
`KERNEL_LIBRARY.json`, `REPRESENTATION_LIBRARY.json`. One OrganGenome per
named organ (embed, gqa_attention, deltanet, mlp_gate_up, mlp_down, lm_head,
sampling); one KernelGenome per qualified kernel; one entry per representation
family. A library entry that cites a receipt which does not exist (on disk or
in git) is a failed test.

**24. Fewer stored bits is not fewer nanoseconds.** (N032;
`receipts/headless/BYTES_FRONTIER.json`) binary_g64 at 1.25 bpw did move
COMPLETE_TOKEN_NS toward the 729.7 model-reachable roof (delta 4.116e6 ns).
ternary_5in8_g64 (1.85), shared_binary_k2 (0.531), and
binary_residual_sparse_2pct (2.216) all stored fewer bits than q2f 2.25 and
did not: extra arithmetic ate the byte win. C1/C2/C3/C5 independently sealed
shared-basis / tensor / low-rank+sparse / structured-transform as
NOT_WORTH_BUILDING on Qwen3.8.

**25. Conversational model names are not identities.** (S023 §34;
`receipts/headless/ODYSSEY_QUEUE_RECOVERED.json`) Recover the Odyssey queue
from disk. Reconcile frontier family names (Qwen3.8, DeepSeek V4 Flash, GLM
5.x, T5V4, Kimi K3) against sealed source admissions. An unresolved shorthand
stays UNRESOLVED with `repository: null`. Never fabricate a Hugging Face repo
id. T5V4 is UNRESOLVED on this disk. GLM 5.x resolves to `zai-org/GLM-5.2`,
not Odyssey patients O010/O012 (those are GLM-4.5).
