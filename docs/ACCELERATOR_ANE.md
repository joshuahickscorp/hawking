# Apple Neural Engine lane

Hawking treats the Neural Engine as a measured third compute domain behind the
Physical Graph Compiler. Metal remains the primary Flash path; this lane may
only promote an organ when it lowers measured complete useful work (including
transfer and synchronization), never from nominal utilization.

The artifact boundary is permanent: Gravity is the search process, NR (`.nr`)
is the portable representation/transient shard layer, and NX (`.nx`) is the
machine-bound executable. ANE placement belongs in NX/Physical Graph evidence,
never in an NR shard.

## Public boundary

The lane uses only supported Core ML APIs: `MLComputeDevice`, `MLModelConfiguration`,
`MLProgram`, `MLComputePlan`, and (when justified) `MLState`. Private or
undocumented interfaces, jailbreaks, firmware hooks, and direct ANE control are
explicitly forbidden. Apple owns scheduling and placement; Hawking records the
plan and runtime evidence.

## Current artifacts

```text
tools/accelerator/run_ane_lane.sh
tools/accelerator/ane_probe.swift
tools/accelerator/ane_micrograph_author.py
receipts/headless/APPLE_ANE_DEVICE_PROFILE.json
receipts/headless/APPLE_ANE_ATLAS.json
receipts/headless/ACCELERATOR_SCOREBOARD.json
hcli/ane_provider.py
```

Run the smallest probe with:

```bash
tools/accelerator/run_ane_lane.sh
```

When a public Core ML compiled model is available, pass its `.mlmodelc`
directory (or an input `.mlmodel`/`.mlpackage` that the Apple runtime can
compile):

```bash
tools/accelerator/run_ane_lane.sh /path/to/model.mlmodelc
```

The probe then uses `MLComputePlan` to record each MLProgram operation's
supported devices, preferred device, and estimated cost weight. Those are
placement-plan facts only; they do not establish runtime latency or energy.

The probe discovers CPU/GPU/Neural Engine devices. With a model it uses only
public `MLModel.compileModel(at:)` and `MLComputePlan.load(contentsOf:configuration:)`.
Without a model, the atlas author emits Flash-shaped MLProgram micrograph
manifests (matmul/GEMV/GEMM, normalization, activations, softmax/SDPA,
convolution, gather/scatter, top-k, and a fused projection gate). The bundled
`workspace/ops/ane/python/coremltools/modelrunner/ModelRunner/add_model.mlmodelc`
fixture has also been used as a generic public-plan sanity check: the profile
may therefore be `PLAN_READY` for its single `ios16.mul` operation, which is
CPU-preferred while supporting CPU and Neural Engine. It is not Flash-shaped.
The Command Line Tools Python package still cannot author the Flash atlas blob,
so Flash compilation, placement, latency, energy, and parity remain
`NOT_MEASURED` rather than being inferred from graph syntax.

The platform scoreboard is a small derived view over these and the current
Flash/Qwen receipts. It carries benchmark state and keeps absent complete-work
metrics unknown; it is not itself an ANE measurement or a promotion receipt.

## GPU critical-path candidate

The strongest current Flash GPU candidate is exposed separately from the ANE
lane. Setting `HAWKING_FLASH_HC_ROUTER_FUSED=1` selects the source-BF16
`qwen_next_hyperconnection_input_fused_with_block_router_topk` organ in the
full-attention and stateful Noetic graphs. It keeps the MLP input in
threadgroup memory while producing the block logits, router logits, shared
scalar, route IDs, and normalized top-k weights. This removes the MLP-input
global write/read edge and two old routing dispatches; the established device
outputs remain populated for parity and downstream expert ABI. The candidate
is opt-in and must still pass a compiled Metal parity/AB run before any
latency or promotion claim. `HAWKING_FLASH_ROUTER_TOPK_FUSED=1` remains the
less aggressive standalone router/top-k comparison.

## Promotion gate

`ANEProvider` is evidence-only. `plan_ready`, measured atlas rows, source-parity
organ results, and a complete-token wall-time measurement are all required before
an ANE candidate can be promoted. Until then the Physical Graph Compiler may
describe ANE as a candidate, but it must retain the Metal route.

The current Flash critical path is likewise source-bound: the layer-3 full-
attention organ probe is [FLASH_NOETIC_FULL_ATTENTION_LAYER3_ORGAN.json](/Users/scammermike/Downloads/hawking/receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER3_ORGAN.json).
It passes HyperConnection read-mix, Q/K/V projection, Q/K norm + RoPE, causal
GQA, sigmoid gating, O projection, the first HyperConnection combine, routed /
shared MoE, and the second HyperConnection combine on the exact layer-0..2
state handoff. Layers 4..6 also pass source-BF16 parity through explicit state
artifacts. This is still not a complete-token or ANE result.

The next bounded step is to run the same manifests with a full Xcode/Core ML
toolchain, inspect each operation with `MLComputePlan`, then measure CPU+ANE,
CPU+GPU, and (where useful) concurrent Metal+ANE. A stateful DeltaNet probe using
`MLState` follows only if an isolated organ demonstrates a real complete-token
benefit.
