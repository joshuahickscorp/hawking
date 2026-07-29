# GLM-5.2 route-population census

Route-metadata census from retained teacher top-k indices only. Not a compression run. Not a capability claim.

## Top-level fences (all remain false)

- `route_population_evidence_sufficient_for_rank_assignment`: **False**
- `within_target_bpw_for_proven_complete_assignment`: **False**
- `full_traversal_authorized`: **False**

## Coverage

- Covered routed layers: **3–77** (75 layers × 256 = 19200 experts)
- Missing: **layer 78** (256 experts UNOBSERVED, never imputed)
- Per-layer route sum: **32768** (OK)
- Duplicate overlapping copies: **12** layers, all byte-identical = **True**
- Loaded top-k members: **87**; sealed array hashes present/verified: **87/87**; all match = **True**
- Whole capsule hash recomputed: **False** (bind sealed capsule hashes only)

## Evidence bands (arithmetic anchors, not quality labels)

- Anchors: promotion-grade min route **2577**; low-traffic diagnostic **205**
- `PROMOTION_PANEL_ROUTE_RANGE`: **133**
- `BETWEEN_PILOT_ANCHORS`: **2131**
- `BELOW_LOW_TRAFFIC_ANCHOR`: **16607**
- `ZERO_ROUTE`: **329**
- `UNOBSERVED`: **256**

## Global route-count quantiles (covered only)

- min/p50/max: **0.0 / 29.0 / 3644.0**
- mean±std: **128.00 ± 362.96**

## Byte scenarios

### anchor_assignment_scenario (incomplete, never authorizing)

- rank-64 experts: **133**
- rank-128 experts: **2131**
- unresolved experts: **17192**
- known-rank total bytes: **41,195,721,216**

### rank128_for_all_nonpromotion_bound (byte envelope, not quality)

- rank-64 / rank-128: **133 / 19323**
- total bytes: **122,339,760,640**
- complete BPW: **95577938/73567377** (1.299189)
- within 49/50: **False**

### native_for_unresolved_bound

- rank-64 / rank-128 / native: **133 / 2131 / 17192**
- total bytes: **1,339,148,259,840**
- complete BPW: **348736526/24522459** (14.221108)
- within 49/50: **False**
- components reconcile: **True**

## Comparison with sealed max rank-128 under 49/50

- Sealed max rank-128 experts: **6583**
- Non-promotion bound n_rank128: **19323** (exceeds sealed max: **True**)

## Next safe action

Run a bounded real-weight pilot on selected representatives spanning BELOW_LOW_TRAFFIC_ANCHOR and BETWEEN_PILOT_ANCHORS experts across early/middle/late layers, and test a representation designed for zero/rare routes. Do not rehydrate from this census alone. No full traversal is authorized.

## Safety

- `RAMANUJAN_RESEARCH_AUTHORIZED`: **False**
- `HIDE_KERNEL_TURN`: **False**
- `ODYSSEY_LAUNCH_AUTHORIZED`: **False**
- `full_parent_traversal_started`: **False**
- `full_traversal_authorized`: **False**
- `capable_artifact_claimed`: **False**
- `MOP_touched`: **False**

Receipt sha256: `d14e463f9f068e99586222211876cfebd6a0ed6b7372b06515412a780c570da1`
