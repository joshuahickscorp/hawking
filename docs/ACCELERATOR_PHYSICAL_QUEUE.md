# Accelerator physical qualification queue

`tools/accelerator/physical_qualification.py` is the concrete candidate frontier
for the current Qwen27 and Flash campaign. It is separate from the
architecture-repatriation queue: the atlas records reusable behaviors, while
this queue records the exact model/kernel mutation that can be measured.

Build or validate it through HCLI:

```console
python3 -m hcli agentos accelerator-physical-queue \
  --repo-root /Users/scammermike/Downloads/hawking \
  --emit receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json

python3 -m hcli agentos accelerator-physical-queue \
  --validate receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json
```

HCLI also owns explicit status progression. It accepts only the controlled
transitions for the current row, carries forward receipt evidence, rebuilds
the ready WorkUnits, and refreshes the queue fingerprint:

```console
python3 -m hcli agentos accelerator-physical-queue \
  --queue receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json \
  --candidate-id qwen27-fast-profile \
  --advance-status PROTECTED_REJECT \
  --evidence receipts/headless/example-protected-receipt.json \
  --emit receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json
```

When the protected runner has emitted a receipt, HCLI can import its declared
metrics and bind the receipt to the candidate's exact child mutation. Missing
footprint/read/synchronization fields remain null and therefore cannot satisfy
the protected-pass gate:

```console
python3 -m hcli agentos accelerator-physical-queue \
  --queue receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json \
  --candidate-id qwen27-fast-profile \
  --advance-status READY_PROTECTED \
  --receipt receipts/headless/protected-qwen27-fast.json \
  --emit receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json
```

The example is illustrative only: a protected rejection requires an actual
receipt. A pass or rejection cannot be recorded without evidence, and a
`BLOCKED` row requires a reason. Rejected/blocked candidates return only to
`STATIC_ONLY`, so they cannot silently re-enter the runnable queue.

Each row carries the baseline, exact child mutation, expected work/byte/GPU
mechanism, parity and capability contracts, argv-only diagnostic/protected
commands, dependencies, and one of the controlled statuses:

`STATIC_ONLY → READY_DIAGNOSTIC → DIAGNOSTIC_PASS → READY_PROTECTED →
PROTECTED_PASS → INTEGRATED`

Every row also carries conservative `scope_tags`. `MODEL_LOCAL` means the
implementation is intentionally specimen-specific; `BACKEND_FAMILY` means
the runtime seam may transfer within the backend family; and
`GENERIC_CANDIDATE` means the shared implementation is a transfer hypothesis,
not a law. `GENERIC_VERIFIED` is reserved for an integrated protected
cross-scope result and is absent from the current frontier. Generic candidates
must carry explicit transfer evidence before promotion.

Reject states and `BLOCKED` are durable scars, not missing data. A physical pass
must add receipt evidence before the status can advance. Queue construction is
side-effect free and records the physical state as `UNKNOWN`; it never starts a
resident or infers GPU/ANE timing.

The shared Metal substrate keeps pipeline handles local to an uncached batch
and also supports resident reuse across token/organ batches. It elides
redundant pipeline-state binds while an ordered or concurrent encoder
continues using the same exact kernel. New/per-dispatch diagnostic encoders
still bind explicitly, so this is a host-encoding reduction with unchanged
dispatch topology. The Qwen27 row `qwen27-pipeline-state-elision` exposes an
explicit `=0` control and `=1` candidate; its benefit remains to be measured by
a protected A/B. The hot TCB path also snapshots whether the cost ledger is
active, so unrecorded dispatches do not pay clock reads, and compares sticky
pipeline IDs instead of allocating a short-lived kernel-name string on each
change. The separate Qwen27 row `qwen27-pipeline-cache-reuse` controls moving
warmed pipeline handles across token command buffers; `=0` restores an empty
per-token cache, so its lock-elision benefit is independently measurable.
The dependent `qwen27-pipeline-id-resolution` row then exposes
`HAWKING_METAL_PIPELINE_ID_RESOLUTION=1` versus `=0`: the fast path resolves a
warmed handle through its stable integer ID and resident vector, while the
control performs the historical second name-map lookup. This removes host
lookup work only; pipeline identity, dispatch topology, arithmetic, and
resident bytes are unchanged.
Qwen38 sessions likewise prebuild immutable layer tensor names at
attach and resolve named GEMVs from one catalog lookup, removing per-token
formatting and redundant map probes; mixed HQ38M20 sessions probe the native
mixed catalog before the explicitly declared Q4 fallback islands. These are
host-ceremony reductions only:
no GPU timing or parity claim is inferred from source inspection.

The shared Metal encoder-label path is also explicitly queued. The Qwen27 row
`qwen27-encoder-label-elision` and blocked Flash row
`flash-encoder-label-elision` use `HAWKING_METAL_ENCODER_LABEL_ELISION=1`
versus `=0`: fast production contexts skip ordinary labels, while trace and
physical-capture contexts still label encoders. The control restores the old
per-encoder label calls, so any host-time benefit remains a protected A/B
question rather than a source-level timing claim.

The Qwen27 `qwen27-commit-timing-elision` row isolates the finalization seam
for callers that use `TokenCommandBuffer::commit_and_wait` without requesting
a timing result. `HAWKING_METAL_COMMIT_TIMING_ELISION=1` uses the plain commit
and fence path in an untraced production context, omitting CPU clocks, GPU
timeline queries, and the extra post-fence status message; `=0` restores the
historical fully-diagnostic path. Timed APIs, the cost ledger, and diagnostic
trace modes are unchanged, and command topology remains identical.

The separate `qwen27-resident-untimed-decode` row carries that seam through the
long-lived Genesis proposer. `HAWKING_QWEN38_SERVE_UNTIMED=1` selects a serving
API that omits per-token timing-vector allocation and host clocks while keeping
the token graph, state transition, and sampling order unchanged. It is
`STATIC_ONLY` until a protected resident A/B proves complete-token wall time,
capability, and the required physical metrics; the measured `generate_greedy`
path remains the qualification authority.

The Flash native graph has the same cache-reuse seam at its per-organ
`CommandBatch` boundary, the fullseq source-authority executor carries the
same resident handles across repeated positions, and the reusable P6 MoE graph
now carries them across its four adjacent batches per layer. The
`flash-pipeline-cache-reuse` row exposes
`HAWKING_FLASH_PIPELINE_CACHE_REUSE=1` versus `=0`; the candidate moves warmed
pipeline handles across all three surfaces while leaving dispatch order,
command buffers, buffers, and arithmetic unchanged. It remains blocked with
the other Flash candidates until a source-independent NX consumer and
protected complete-token capability exist.

The dependent `flash-pipeline-id-resolution` row uses
`HAWKING_METAL_PIPELINE_ID_RESOLUTION=1` versus `=0` after pipeline-cache reuse
is enabled. It removes the second name-map probe by resolving the resident
handle through its stable batch ID; the control restores the historical lookup
and all Flash source, dispatch, and capability boundaries remain unchanged.

The fullseq Flash executor also caches the prepared 43-layer source catalog
and bounded static RoPE controls under `flash-fullseq-catalog-cache`. The `=1`
path is bound to the exact prepared manifest-file SHA-256, validates the
tokenizer/config anchors once at preparation, packs the source-derived
cosine/sine rows into resident table buffers, and selects each position by a
byte offset instead of copying a static table before every position. It still
checks the position-specific empty compressed-KV rule; `=0` restores
per-position catalog plus RoPE-anchor admission. This is another
host-validation and control-reuse reduction, not a relaxed source-integrity
gate.

The shared P4B-attention/P7-FFN mHC-pre path also has an explicit Flash A/B candidate,
`HAWKING_DSV4F_MHC_PRE_SIMD=1`. It assigns one SIMD-group to each of the 24
source mHC rows in both bounded graphs, stages the wide RMS reduction, and keeps the small nonlinear
Sinkhorn/control section source-ordered. Because the wide FP32 reductions are
re-associated, the one-thread `deepseek_v4_p7_mhc_ffn_pre_authority` kernel
and the existing P4B authority remain the defaults and `=0` control. The candidate is blocked until bounded
BF16/control/output coherence and protected latency evidence exist; it is not
a Flash NX or sub-1-BPW claim.

The P6 graph also has a topology-only candidate,
`HAWKING_DSV4F_P6_SINGLE_CB=1`. It keeps the 60 device dispatches and their
explicit wave boundaries, but records the dependent down/combine waves in the
same command buffer as gate/up/SwiGLU, removing one CPU-visible commit/wait.
On learned-bias layers, the route-ID readback and source residency load remain
the hard boundary; the switch removes the second post-load body fence (3 to 2
command buffers). `=0` preserves the historical control. It remains blocked
until a matched source-parity and protected latency receipt proves that
Metal's hazard ordering is preserved.

`HAWKING_DSV4F_P6_PREFIX_CONCURRENT=1` is a separate prefix-wave candidate.
The Gate reduction and BF16-to-E4M3FN quantizer both read the same input but
write disjoint buffers, so they can share one concurrent encoder before the
route kernel's dependency boundary. Dispatches and bytes are unchanged; the
structural target is 10 to 9 compute encoders. It remains blocked until the
matched source parity/coherence and protected latency receipt is complete.

The P6 BF16-to-E4M3FN quantizer has a separate candidate,
`HAWKING_DSV4F_P6_ACT_QUANT_SIMD=1`. It assigns one SIMD-group to each
128-wide block and uses packed loads/stores while retaining the exact
E4M3FN/UE8M0 byte contract. `=0` keeps the serial authority. The candidate is
blocked until every quantized byte and scale byte matches the source oracle and
the protected latency receipt is complete.

The P6 routed-expert FP4 matvecs have a separate candidate,
`HAWKING_DSV4F_P6_FP4_SIMD=1`. It uses a 64-lane x 4-row threadgroup, packed
`uchar4` loads, and SIMDgroup split-K partials for the 18 routed W1/W3/W2
dispatches in a full MoE layer. `=0` keeps the serial source-native authority;
the candidate remains blocked until NumericParity/coherence and protected
latency evidence cover both W1/W3 and W2 shapes.

The P6 shared-expert FP8 matvecs have a separate candidate,
`HAWKING_DSV4F_P6_FP8_SIMD=1`. It uses a 256-threadgroup with eight SIMDgroups
to split the source 128-wide activation blocks for the three shared W1/W3/W2
dispatches per full MoE layer, combining block partials in source block order.
`=0` keeps the serial source-native authority; the candidate remains blocked
until NumericParity/coherence and protected latency evidence cover both shared
W1/W3 and W2 shapes.

The learned-bias P6 route also has a host-ceremony candidate,
`HAWKING_DSV4F_P6_LEARNED_READER_REUSE=1`. It retains only the admitted,
seal-checked metadata reader across route changes, removing repeated manifest
and index admission while still creating a fresh bounded expert cache, reading
the selected bundles, and uploading all six GPU expert slots. `=0` re-admits
the reader for the matched authority control. This is a host-latency A/B, not
a GPU arithmetic or Flash qualification claim.

Its dependent `HAWKING_DSV4F_P6_LEARNED_EXPERT_CACHE_REUSE=1` row keeps the
reader-reuse setting fixed and retains only the exact six-bundle bounded source
cache. Overlapping learned routes can then avoid repeating source chunk
materialization; a route with new experts evicts within that same capacity.
This candidate does not retain decoded weights or change GPU dispatches, and
must be measured separately from reader-admission reuse.

P6 also exposes a higher-leverage routed epilogue fusion under
`HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED=1`. The fixed-six indirect record
launch pairs each routed FP4 W1/W3 reduction with the exact two BF16
round-trips, clamp/SwiGLU, and device route weight, writing the existing
per-expert BF16 scratch buffers. Its scalar authority-shaped candidate reduces
P6 batch 1 from 38 to 9 dispatches and the fixed hash graph from 60 to 31;
`HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD=1` is a separate eight-SIMDgroup
occupancy A/B against scalar fusion. Both candidates declare indirect weight
reads and scratch writes explicitly; neither changes the source bytes or the
authority path, and both remain blocked until source parity/coherence and
protected complete-token latency evidence exist.

The routed P6 down wave has a separate fixed-six candidate,
`HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED=1`. It leaves the per-expert E4M3FN
QAT wave and shared-expert path untouched, then consumes each routed expert's
already-quantized activation and FP4 W2 weights in one indirect launch,
writing the existing BF16 buffers used by the source-order combine. This
reduces the fixed hash graph from 60 to 49 dispatches and P6 batch 2 from 22
to 11 structurally. It is independent of the gate/up fusion; the two can be
composed only after separate parity and protected-latency evidence.

`HAWKING_DSV4F_P6_FP4_DOWN_BF16_SIMD=1` is a separate occupancy sibling of
that down-fusion row. It keeps the same fixed-six indirect launch and output
contract, but assigns one of eight SIMDgroups in a 256-threadgroup to each
output row and splits each source 32-value FP4 block across lanes. The scalar
fused path remains the control; this sibling is blocked until source parity,
coherence, and protected complete-token latency evidence are available.

Each candidate also carries a `measurements` object. Its required fields cover
total NX bytes, resident bytes, active/read/transient bytes per token, GPU ns,
complete accepted-token wall ns, dispatches, synchronization, TPS, and
fallbacks. Use `--measurements metrics.json` when recording a protected pass;
the queue rejects `PROTECTED_PASS` or `INTEGRATED` while any required metric is
missing.

The `qwen27-q2f-splitk4` row isolates the biasless Q2F split-K geometry through
`HAWKING_Q2F_GEO`; the older `HAWKING_AFFINE2_GEO` value remains the compatibility
inheritance path. The `qwen27-affine2-splitk4-vec` and
`qwen27-q2f-splitk4-vec` rows are the N035 protected A/B candidates. They use
the explicit geometry controls so biasful affine Q2 and biasless Q2F launches
can be measured independently, and remain `STATIC_ONLY` until Metal compilation
and a native parity/latency run succeed.

The `qwen27-q4-vecgroup-x64` row exposes the existing
`Qwen38MatvecKernel::VecgroupX64` sibling through
`HAWKING_QWEN38_Q4_GEO=vecgroup_x64`. It applies only to standalone uniform-Q4
group-64 GEMVs; affine-Q2/Q2F and fused Q4 kernels keep their own bindings.
The control is the same fast profile with `HAWKING_QWEN38_Q4_GEO=tpr64`.
This is an occupancy A/B: packed bytes, dispatch count, and output buffers are
unchanged, and no speed or parity result is implied until a protected receipt
records them.

The Flash source-BF16 row now reaches the active native lm_head and streamed
source wrappers. `HAWKING_FLASH_BF16_VEC4=1` selects source-order-preserving
packed loads; `HAWKING_FLASH_BF16_GEO=1` selects the separate SIMD-group
reduction candidate and takes precedence over VEC4. The scalar kernel remains
the authority/default. The native graph receipt records the selected kernel so
an eventual protected receipt can prove that the requested mutation actually
ran. These controls still do not imply Flash NX or complete-token qualification.

The compact Flash MoE row adds `HAWKING_FLASH_MOE_VEC4=1` as a separate,
source-order-preserving packed-load candidate for compact gate/up/SwiGLU and
direct down-to-HyperConnection accumulation. It leaves dispatch topology,
resident bytes, route order, and diagnostic outputs unchanged. It must be
qualified independently from `HAWKING_FLASH_MOE_GEO=1`, whose SIMD reduction
has a different association; neither candidate is promoted without source
parity and protected complete-token evidence.

The routed FP4 gate/up candidate is separately opt-in through
`HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED=1`. It replaces the two FP4 matvec
outputs, two FP32-to-BF16 casts, and standalone routed SwiGLU with one fused
gate/up-to-BF16 kernel. The fused implementation preserves the authority
accumulation order and performs the same explicit BF16 round-trip before the
clamp and route-weighted epilogue. The scalar/five-dispatch path remains the
control; the candidate is blocked until a source-independent Flash NX
protected run proves parity, GPU ns, and complete-token metrics.

The shared FP8 gate/up candidate is separately opt-in through
`HAWKING_DSV4F_FP8_SHARED_GATE_UP_SWIGLU_FUSED=1`. It applies the same
five-dispatch-to-one fusion to the shared FP8 expert path in both the native
graph and reusable P6: the two source-order FP8 reductions feed explicit BF16
round-trips and the existing clamped SwiGLU epilogue in one kernel. On P6 this
also removes the now-empty cast/SwiGLU waves when the routed gate/up fusion is
enabled, without changing the down/combine dependency boundary. The control
keeps the FP8 matvecs, casts, and standalone SwiGLU; the candidate remains
blocked until source-independent Flash NX parity and protected complete-token
metrics are available.

The shared FP8 down/combine candidate is separately opt-in through
`HAWKING_DSV4F_FP8_SHARED_DOWN_COMBINE_FUSED=1`. It computes the shared FP8 W2
row, performs the source-required BF16 round-trip, and adds the six already
BF16 routed rows in numeric-expert order before writing the final BF16 output.
This reduces the shared W2/cast/combine boundary from three dispatches to one
in both native Flash and reusable P6; routed W2/casts and shared activation
quantization remain explicit predecessors. The control keeps the original
three-dispatch sequence. The candidate remains blocked until source-independent
parity, hazard validation, and protected complete-token latency evidence exist.

The P6 down-wave quantizer also has a fixed-seven indirect candidate,
`HAWKING_DSV4F_P6_BATCHED_DOWN_QAT=1`. It packs the six routed and one shared
BF16-to-E4M3FN/E8M0 activation-quantization launches into one dispatch while
retaining the same seven quantized/scales buffers and exact 128-value block
arithmetic. The isolated structural target is 60 to 54 P6 dispatches; the
compute-encoder count is unchanged because the existing seven launches already
share one concurrent wave. It remains blocked until source-independent parity,
resource-hazard validation, and protected complete-token timing exist.

The composed `flash-p6-fused-epilogue-stack` row turns on the existing
single-command-buffer, concurrent-prefix, routed gate/up, routed down, shared
gate/up, shared down/combine, and fixed-seven down-QAT controls together. Its
closed P6 structural target is 60 to 7 dispatches, 10 to 4 compute encoders,
and 2 to 1 command buffer on the hash path. The fixed-seven down-QAT producer
and full routed/shared down-combine consumer share one explicit serial encoder
with a resource barrier; the other independent waves retain their concurrent
boundaries. This is an interaction measurement, not an additive performance
claim; it remains blocked until the full stack passes matched source
parity/coherence and protected complete-token timing.

For the same guarded fused configuration, P6 no longer reserves full-size
authority scratch for writers that are disabled by the selected fusion. Those
fields retain one-byte non-null placeholders for ABI safety, while the live
SwiGLU, QAT, and final-output buffers remain unchanged. The full stack removes
344022 bytes of dormant scratch structurally; learned-bias preparation also
uses one-byte FP4 placeholders until the device-selected six-expert route is
known, avoiding a roughly 75 MiB pre-route weight-allocation peak. These are
allocation facts, not physical speed claims, and must still be checked in the
protected receipt.

The `flash-meta-sub1-coherent` row registers the separate functional
representation frontier. Its `meta_bpw` target (`0.8871807728336929`) is a
teacher-constrained description budget for expert-local latent programs,
frequency-tiered n-gram generation, residual repair, and protected exact
islands. It is not `physical_ebpw`, serialized bytes, residency, or a measured
active-bytes/token result. The row is intentionally `BLOCKED` until a
source-independent serializer/loader, direct native consumer, held-out
coherence evidence, and protected complete-token accelerator measurements
exist; no sub-1 physical claim is made. The offline screen is now bound to the
source-authority receipt `receipts/headless/FLASH_META_TEACHER_L4.json` and its
`model.language_model.layers.4.mlp_input` F32 capture. It requires dense
source-BF16 execution through the prefix and layer 4, validates every dynamic
top-K route row, and fits against the capture's route union; an older raw
`[streams, hidden]` state probe cannot be reused as an expert-input surface.
If the host cannot expose a Metal-capable GPU, the capture emits
`receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json` with zero
teacher rows and promotion disabled; that boundary is evidence of blocked
capture only, never a substitute for source-authority rows.

The fullseq attention capture also has a default-off encoder-topology A/B:
`HAWKING_DSV4F_FULLSEQ_ORDERED_ENCODER=1` folds its dependent dispatch chain
into the existing ordered encoder while preserving the 22/25 dispatch count
and one command-buffer boundary. Capture rows record both encoder and dispatch
counts; parity must be established before considering promotion. It is tracked
as `flash-fullseq-ordered-encoder` even while the source-independent Flash NX
executable remains blocked.

The full-attention organ also registers `HAWKING_FLASH_QKV_GQA_FUSED=1`. The
candidate folds source-BF16 Q/K/V projection, Q/K normalization and RoPE, and
the current-slot KV-cache writes into the existing head-local launch. Raw Q/K/V
diagnostic buffers remain populated, the cache geometry is unchanged, and the
scalar norm reductions retain their authority order. The matched control keeps
the triple projection and Q/K transform launches separate; this candidate is
still blocked until source-independent Flash NX parity and protected
complete-token latency evidence exist.

The companion Qwen27 token budget is emitted through HCLI:

```console
python3 -m hcli agentos qwen27-token-budget \
  --repo-root /Users/scammermike/Downloads/hawking \
  --emit receipts/headless/QWEN27_TOKEN_NS_BUDGET.json
```

It is a plan-only budget: the static byte denominator is retained, while cold,
warm, first-token, steady-state, GPU, and complete-token fields remain null.

Flash rows remain blocked at the complete source-independent NX boundary. Their
source/oracle commands are retained for planning and diagnosis, but cannot be
read as Flash NX or complete-token qualification.
