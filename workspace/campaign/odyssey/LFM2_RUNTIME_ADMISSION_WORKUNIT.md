# LFM2 runtime admission and activation capture

## Objective

Admit `LiquidAI--LFM2-24B-A2B@a3bbacd91a67` to one bounded **research-only**
Hawking runtime session, capture real route-conditioned inputs for its
layer-2 expert organ, then release it and restore `KIMI_P0_OPERATIONAL`.
This work unit is an execution-enablement and evidence-capture task. It does
not create a normal user-facing model, promote LFM2, mutate its weights, or
make a capability/TPS claim.

## Authority and invariants

- `hawkingd` remains the only production/process root. No direct
  `mlx_lm.server`, Python, or provider root may be launched beside it.
- The normal HCLI/OpenWebUI selector remains Gravity artifacts only. LFM2 is a
  temporary Research-surface body and must not appear in the ordinary menu.
- Use the installed MLX-LM LFM2 implementation only as a disclosed
  compatibility bridge. Do not label it Hawking-native and do not use
  llama.cpp.
- Preserve KIMI P0's frozen identity and bounded authority. Stop its provider
  only through hawkingd supervision; retain/recover its session state and
  restart it after the LFM2 release check.

## Entry evidence

- `receipts/future/ODYSSEY_MOE_TRANSFER_CLUSTER_OPEN_20260911.json`
- `receipts/future/LFM2_ARCHITECTURE_RECOGNITION_20260911.json`
- `workspace/campaign/odyssey/MODELLAKE_SCHEDULING_AUTHORITY.json`

The 45 GB activation-capture guard passed with only about 4.3 GB headroom.
The 64 GB conversion-style path is STOP. Therefore LFM2 and P0 must be
serialized, not concurrently resident.

## Required work

1. Add or activate a hawkingd-owned **research provider** path for the local
   LFM2 safetensors snapshot. The provider must be a child of the existing
   root and must be rejected if a second provider would be created.
2. Prove readiness with a small deterministic generation and record the exact
   backend, model path, config hash, PID ancestry, memory, and swap.
3. Capture bounded real forward inputs and routed expert IDs for layer 2.
   Capture enough rows for held-out allocation, with the source prompts,
   route distribution, tensor/module path, and row counts recorded.
4. Explicitly account for LFM2's `feed_forward.expert_bias` and
   `conv.conv.weight` structures. If the compatibility backend cannot execute
   either, stop and record the named lowering blocker; do not substitute a
   dense or generic MLP approximation.
5. Release the LFM2 child through hawkingd. Verify no orphan remains, no swap
   remains, and P0 is restored under the same daemon root with its one-provider
   invariant intact.
6. Emit a receipt and corpus under `workspace/campaign/odyssey/activations/`.
   On success the next WorkUnit is activation-aware binary/ternary or
   vector-coded residual plus sparse salient repair on the captured organ.

## Gates

The work unit passes only when all of these are observed:

- one sovereign hawkingd process tree throughout;
- LFM2 readiness and one real forward path;
- captured route-conditioned activation corpus;
- no unaccounted expert-bias or convolution fallback;
- clean LFM2 release and P0 restoration;
- zero swap or an explicit physical STOP receipt.

Otherwise emit `BLOCKED` with the exact missing lowering/runtime capability,
the attempted path, observed process ancestry, memory state, and a reopen
condition. A static-only capture or raw externally launched server is not a
pass.
