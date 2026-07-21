# Kimi K2.6 Long-Run Final Report

## Outcome

The tested `<=0.98`-BPW linear repair region is closed. The best defensible representation remains `P1_DUAL_PATH_RECOVERY_R16X2_LOCAL_F1_ONLY` at `0.908590952555` complete BPW with F1 cosine `0.913441658`. No candidate earned F2 promotion.

Management stopped before the four-hour boundary under the explicitly permitted disk-floor-risk condition. The hard floor was raised by `50 GiB`, from `32 GiB` to `82 GiB`; current free space is below that floor. No active result was abandoned, and the sole Kimi source, controller, and MOP were preserved.

## Time and experiment accounting

| Measure | Value |
|---|---:|
| Wall-clock managed | `01:23:53.33` |
| Valid scientific compute | `00:03:24.75` |
| Invalid/retry scientific compute | `00:06:04.00` |
| Total scientific compute | `00:09:28.75` |
| Waiting, diagnosis, verification, and management | `01:14:24.58` |
| Completed experiments | `13` |
| Held-out tokens represented in closure | `8672` |

## Operational guard at stop

| Guard | Result |
|---|---|
| Hard floor | `82 GiB` (`88046829568` bytes) |
| Free disk | `39.25 GiB` |
| Deficit | `42.75 GiB` |
| Controller | PID `13332`, heartbeat age `2.3s`, lease matched `True` |
| Sole source | `True`; `96` files verified |
| MOP | protected `True`, device `16777233`, inode `1233332` |
| Heavy Apple jobs | `1` |
| Guard audit | `FAIL` with failures `DISK_FLOOR_RISK` |

## Causal result

Primary diagnosis: **upstream compact-state error is primary; route drift is a secondary conditional amplifier after a margin crossing**.

| Causal stage | Evidence | Classification |
|---|---|---|
| Compact perturbation | Layer-1 route-set agreement is `1.0`; compact output is installed after that router. | Perturbation precedes the layer-2 router. |
| Hidden-state drift | Natural layer-2 relative L2 is `0.096221`. | Upstream drift exists before route selection. |
| Router-margin crossing | Held-out route change `19.767%`; pooled `21.637%` with 95% CI `18.567%`–`24.854%`. | Strong correlate and conditional cause. |
| Expert entry/exit | First crossed token exits `[266]` and enters `[278]`. | Concrete expert-set intervention point. |
| Weighted-MoE damage | First material MoE divergence has relative L2 `0.176907` while routes still match. | Route mismatch is not necessary for material damage. |
| Residual propagation | First irrecoverable residual cosine `0.983837`, next-layer cosine `0.987549`. | Damage propagates beyond the MoE block. |
| Later rescue | Teacher weighted-MoE substitution rescues `36.2%`; teacher hidden restoration rescues `100.0%`. | Hidden restoration identifies the causal upper bound. |
| F2 | The only promoted upstream row failed independent replication and slightly worsened layer 3. | No F2-promotable repair. |

## Layer routing atlas

| Layer/path | Route agreement | Hidden / routing evidence |
|---|---:|---|
| Layer 1 natural | `1.000` | 8th-vs-9th margin mean `0.002505` |
| Layer 2 natural | `0.802` | Jaccard `0.950`, rank concordance `0.969` |
| Layer 3 natural | `0.674` | Relative L2 `0.085443` |
| Layer 3 forced teacher indices | `0.744` | Relative L2 `0.082038` |
| Layer 3 forced indices + weights | `0.744` | Relative L2 `0.077875` |
| Layer 3 teacher MoE | `0.767` | Relative L2 `0.042068` |
| Layer 3 hidden restore | `1.000` | Relative L2 `0.0` |

## Intervention matrix

| Intervention | LR01 rescue | LR08 rescue | Stratified result |
|---|---:|---:|---|
| Natural student | `0.000%` | `0.000%` |  |
| Force teacher indices | `8.576%` | `14.635%` |  |
| Force teacher indices + weights | `10.721%` | `16.251%` | mismatch rescue 46.333%; match rescue 0.940% |
| Substitute teacher weighted-MoE | `35.191%` | `36.211%` |  |
| Teacher state through compact router | `57.278%` | `49.314%` |  |
| Restore teacher hidden state | `100.000%` | `100.000%` |  |

The decisive stratification is route-conditioned: indices-plus-weights rescue `46.333%` after a route crossing but only `0.940%` when routes already match. There are `33` material-error tokens with exact route sets still matched.

## Treatment frontier and exact physical allocation

The retained parent uses `4022298` compact-base bytes, `974848` Doctor bytes, and `4669` header bytes: `5001815` bytes total.

| Candidate | Bytes | Complete BPW | Extra allocation | Held-out paired improvement (95% CI) | Decision |
|---|---:|---:|---|---|---|
| `FIRST_DIVERGENCE_PROTECTION_R12` | `5189192` | `0.942628406343` | first_divergence_state_repair_bytes=187377 | -0.010357730 [-0.017299975, -0.004250580], n=61 | `RETIRE_HELDOUT_HARM` |
| `PRE_ROUTER_STATE_R24` | `5361130` | `0.973861331031` | pre_router_state_repair_bytes=359315 | -0.008711499 [-0.015937196, -0.002103438], n=61 | `RETIRE_HELDOUT_HARM` |
| `LOW_MARGIN_ROUTER_R24` | `5184912` | `0.941850934710` | conditional_router_repair_bytes=183097 | -0.008124208 [-0.014875668, -0.002095556], n=61 | `RETIRE_HELDOUT_HARM` |
| `WEIGHTED_MOE_OUTPUT_R24` | `5361151` | `0.973865145729` | weighted_moe_output_repair_bytes=359336 | -0.009026688 [-0.011761604, -0.005226642], n=61 | `RETIRE_HELDOUT_HARM` |
| `HYBRID_R12X2` | `5375982` | `0.976559230259` | post_moe_repair_bytes=186464, pre_router_repair_bytes=186464, repair_header_bytes=1239 | -0.018527407 [-0.027155238, -0.009294006], n=61 | `RETIRE_HELDOUT_HARM` |
| `POST_MOE_HIDDEN_CV_R12_S025` | `5189121` | `0.942615509033` | parent_bytes=5001815, post_moe_hidden_repair_bytes=187306 | +0.000135046 [-0.000342189, +0.000718405], n=289 | `RETIRE_SHRINKAGE_SURVIVOR_UNTOUCHED_FAILURE` |
| `UPSTREAM_RESIDUAL_CV` | `5045649` | `0.916553497314` | parent_header_bytes=4669, upstream_residual_repair_bytes=43834 | +0.000006182 [-0.000203388, +0.000206387], n=1383 | `RETIRE_UPSTREAM_RESIDUAL_REPLICATION_FAILED` |

## Replication and falsification

- The five physical repair rows all harmed typical tokens on the original held-out split and the `309`-token new split, including adversarial low-margin contexts.
- The post-MoE shrinkage survivor failed untouched validation and then showed mean harm `-0.001721730` on `2572` tokens.
- `UPSTREAM_RESIDUAL_CV` first improved F2 by +0.000136857 [+0.000009130, +0.000293509], n=1397, but its independent replication was +0.000006182 [-0.000203388, +0.000206387], n=1383; it was retired.
- Causal intervention ordering replicated on `2636` large-intervention tokens and the balanced 256-token stratification.

## Proven conclusions

- Compact perturbation exists before the layer-2 router.
- Low within-split router margin increases route-crossing risk but is not sufficient.
- Material residual damage occurs with exact top-8 route sets still matched.
- Teacher route indices plus weights produce only a minority causal rescue.
- Teacher hidden restoration is a full causal rescue at the tested block.
- All five <=0.98-BPW repair rows harm the typical token on two splits.
- The calibration-CV shrinkage survivor fails untouched low-margin validation.

## Correlated, not proven

- Norm-weighted hidden error often improves under low-rank repair.
- That norm-weighted improvement does not establish capability preservation.

## Negative results

- `lr02_all_rows_typical_token_harm`: `true`
- `lr03_replication_all_rows_typical_token_harm`: `true`
- `lr05_shrinkage_untouched_failure`: `"RETIRE_SHRINKAGE_SURVIVOR_UNTOUCHED_FAILURE"`
- `lr07_large_shrinkage_mean_harm`: `{"ci95_high": -0.0016059813282804122, "ci95_low": -0.0018371859467881666, "mean": -0.0017217304847242302, "n": 2572}`
- `upstream_first_f2_gain`: `{"ci95_high": 0.0002935092009441647, "ci95_low": 9.129537938965344e-06, "mean": 0.00013685700509953196, "n": 1397}`
- `upstream_replication_failure`: `{"ci95_high": 0.00020638734722794167, "ci95_low": -0.00020338789260992003, "mean": 6.1818665541325976e-06, "n": 1383}`

## Unresolved questions

- Whether a nonlinear token-conditional repair can avoid the observed domain tradeoff.
- Whether a representation below the compact expert, rather than a downstream Doctor, can remove the upstream state error at equal BPW.

## What Codex changed between runs and why

| After run | Manager decision | Scientific reason |
|---|---|---|
| `LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS` | `ADVANCE_TO_CAUSALLY_TARGETED_PHYSICAL_BRACKET` | Use measured first-divergence margins and intervention rescue to allocate the remaining 0.98-BPW bytes among pre-router and weighted-MoE repair, not generic compression. |
| `LR02_CAUSALLY_TARGETED_PHYSICAL_BRACKET` | `NO_PHYSICAL_ROW_PROMOTED_TESTED_REGION_RETIRED` | Falsify the closure with a new-split boundary test before declaring the <=0.98 region closed. |
| `LR03_NEW_SPLIT_ADVERSARIAL_REPLICATION` | `REPLICATED_NO_PROMOTABLE_REPAIR` | Run shrinkage/rank boundary falsification from calibration-only cross-validation, then validate the single preregistered winner on another untouched split. |
| `LR04_CALIBRATION_ONLY_SHRINKAGE_BOUNDARY` | `ADVANCE_SINGLE_PREREGISTERED_ROW_TO_UNTOUCHED_SPLIT` | Test exactly the selected installed payload and shrinkage on untouched contexts. |
| `LR05_UNTOUCHED_SHRINKAGE_VALIDATION` | `RETIRE_SHRINKAGE_SURVIVOR_UNTOUCHED_FAILURE` | The regularized post-MoE family is closed; use remaining time for causal replication. |
| `LR06_POOLED_CAUSAL_LAW` | `CLOSE_TESTED_LOW_RANK_REPAIR_REGION_PENDING_LARGE_HELDOUT_FALSIFICATION` | Challenge the pooled law on a larger, longer-context held-out set with no tuning. |
| `LR07_LARGE_HELDOUT_FALSIFICATION` | `STRENGTHEN_POOLED_CAUSAL_LAW` | Repeat the large suite with a new prompt ordering/seed and intervention-stratified subset. |
| `LR08_LARGE_INTERVENTION_REPLICATION` | `REPLICATE_MIXED_STATE_FIRST_ROUTE_SECONDARY_CAUSALITY` | Estimate intervention rescue by stratum and test the causal ordering under a new seed. |
| `LR09_STRATIFIED_RESCUE_CONFIDENCE` | `ESTABLISH_CONDITIONAL_ROUTE_AMPLIFICATION_LAW` | Auction remaining physical bits upstream at the compact expert output, where drift begins. |
| `LR10_UPSTREAM_REPRESENTATION_BYTE_AUCTION` | `PROMOTE_TO_HELDOUT_F2` | Install the frozen upstream residual in cached LR01 layer-1 states and run held-out F2. |
| `LR11_UPSTREAM_RESIDUAL_HELDOUT_F2` | `PROMOTE_UPSTREAM_RESIDUAL_TO_REPLICATION` | Replicate the promoted upstream representation on a new split and seed. |
| `LR12_UPSTREAM_RESIDUAL_REPLICATION` | `RETIRE_UPSTREAM_RESIDUAL_REPLICATION_FAILED` | The microscopic F1 gain does not buy robust F2; close the tested linear representation region. |
| `LR13_REGION_CLOSURE_AUDIT` | `TESTED_LINEAR_REPAIR_REGION_CLOSED` | Hold guards through the required wall-clock boundary, then report closure. |

## Chronological ledger

| # | Event | Experiment | Start | End | Compute seconds | Decision | Seal |
|---:|---|---|---|---|---:|---|---|
| 1 | `BASELINE_AUDIT` | `LR00_BASELINE_AUDIT` | `2026-07-21T16:52:21Z` | `2026-07-21T16:52:21Z` | `0.000` | `ADVANCE_TO_HELDOUT_CONTROL` | `ef7cb4c6023b` |
| 2 | `EXPERIMENT_START` | `LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS` | `2026-07-21T17:00:32Z` | `` | `0.000` | `` | `56fbc9ff9d45` |
| 3 | `EXPERIMENT_FAULT` | `LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS` | `` | `2026-07-21T17:02:12Z` | `33.000` | `RETRY_AFTER_REFERENCE_ORDER_FIX` | `01ddf99b282d` |
| 4 | `EXPERIMENT_START` | `LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS` | `2026-07-21T17:02:32Z` | `` | `0.000` | `` | `bff5ec8aa062` |
| 5 | `EXPERIMENT_FAULT` | `LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS` | `2026-07-21T17:02:32Z` | `2026-07-21T17:08:28Z` | `319.000` | `RETRY_WITH_RESIDENT_FORWARD_AS_NATURAL_REFERENCE` | `35584335f219` |
| 6 | `EXPERIMENT_START` | `LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS` | `2026-07-21T17:08:36Z` | `` | `0.000` | `` | `797e20b701c2` |
| 7 | `EXPERIMENT_FAULT` | `LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS` | `2026-07-21T17:08:36Z` | `2026-07-21T17:11:37Z` | `11.000` | `FINALIZE_FROM_RAW_CHECKPOINT_WITHOUT_REPEATING_PARENT_OR_EXPERT_FORWARD` | `39dfd49cad84` |
| 8 | `EXPERIMENT_START` | `LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS` | `2026-07-21T17:11:45Z` | `` | `0.000` | `` | `726dbda1bc51` |
| 9 | `EXPERIMENT_COMPLETE` | `LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS` | `2026-07-21T17:11:45Z` | `2026-07-21T17:11:46Z` | `0.763` | `ADVANCE_TO_CAUSALLY_TARGETED_PHYSICAL_BRACKET` | `7fb72de14a92` |
| 10 | `EXPERIMENT_START` | `LR02_CAUSALLY_TARGETED_PHYSICAL_BRACKET` | `2026-07-21T17:19:18Z` | `` | `0.000` | `` | `88bac7dd70bb` |
| 11 | `EXPERIMENT_FAULT` | `LR02_CAUSALLY_TARGETED_PHYSICAL_BRACKET` | `2026-07-21T17:19:18Z` | `2026-07-21T17:21:04Z` | `1.000` | `RETRY_WITH_AUTHORITATIVE_86_TOKEN_BATCH_AND_POST_FORWARD_SLICE` | `dc33c2578505` |
| 12 | `EXPERIMENT_START` | `LR02_CAUSALLY_TARGETED_PHYSICAL_BRACKET` | `2026-07-21T17:21:18Z` | `` | `0.000` | `` | `7d3ad8f01c67` |
| 13 | `EXPERIMENT_COMPLETE` | `LR02_CAUSALLY_TARGETED_PHYSICAL_BRACKET` | `2026-07-21T17:21:18Z` | `2026-07-21T17:21:22Z` | `4.153` | `NO_PHYSICAL_ROW_PROMOTED_TESTED_REGION_RETIRED` | `1424002ad96a` |
| 14 | `EXPERIMENT_START` | `LR03_NEW_SPLIT_ADVERSARIAL_REPLICATION` | `2026-07-21T17:26:06Z` | `` | `0.000` | `` | `70fae34eb72a` |
| 15 | `EXPERIMENT_COMPLETE` | `LR03_NEW_SPLIT_ADVERSARIAL_REPLICATION` | `2026-07-21T17:26:06Z` | `2026-07-21T17:26:29Z` | `23.011` | `REPLICATED_NO_PROMOTABLE_REPAIR` | `a33bfb8827c0` |
| 16 | `EXPERIMENT_START` | `LR04_CALIBRATION_ONLY_SHRINKAGE_BOUNDARY` | `2026-07-21T17:29:36Z` | `` | `0.000` | `` | `333a0c61ac31` |
| 17 | `EXPERIMENT_COMPLETE` | `LR04_CALIBRATION_ONLY_SHRINKAGE_BOUNDARY` | `2026-07-21T17:29:36Z` | `2026-07-21T17:29:39Z` | `2.643` | `ADVANCE_SINGLE_PREREGISTERED_ROW_TO_UNTOUCHED_SPLIT` | `84ddf28f3bf5` |
| 18 | `EXPERIMENT_START` | `LR05_UNTOUCHED_SHRINKAGE_VALIDATION` | `2026-07-21T17:32:39Z` | `` | `0.000` | `` | `c4e949e1e212` |
| 19 | `EXPERIMENT_COMPLETE` | `LR05_UNTOUCHED_SHRINKAGE_VALIDATION` | `2026-07-21T17:32:39Z` | `2026-07-21T17:32:53Z` | `13.687` | `RETIRE_SHRINKAGE_SURVIVOR_UNTOUCHED_FAILURE` | `4298c7c90b0b` |
| 20 | `EXPERIMENT_START` | `LR06_POOLED_CAUSAL_LAW` | `2026-07-21T17:36:12Z` | `` | `0.000` | `` | `6fc817043b64` |
| 21 | `EXPERIMENT_COMPLETE` | `LR06_POOLED_CAUSAL_LAW` | `2026-07-21T17:36:12Z` | `2026-07-21T17:36:12Z` | `0.429` | `CLOSE_TESTED_LOW_RANK_REPAIR_REGION_PENDING_LARGE_HELDOUT_FALSIFICATION` | `7e3a8530ad96` |
| 22 | `EXPERIMENT_START` | `LR07_LARGE_HELDOUT_FALSIFICATION` | `2026-07-21T17:39:53Z` | `` | `0.000` | `` | `bbfe5b4770ae` |
| 23 | `EXPERIMENT_COMPLETE` | `LR07_LARGE_HELDOUT_FALSIFICATION` | `2026-07-21T17:39:53Z` | `2026-07-21T17:40:19Z` | `26.000` | `STRENGTHEN_POOLED_CAUSAL_LAW` | `521e1f89cc9c` |
| 24 | `EXPERIMENT_START` | `LR08_LARGE_INTERVENTION_REPLICATION` | `2026-07-21T17:43:25Z` | `` | `0.000` | `` | `ef6b9e163d10` |
| 25 | `EXPERIMENT_COMPLETE` | `LR08_LARGE_INTERVENTION_REPLICATION` | `2026-07-21T17:43:25Z` | `2026-07-21T17:43:54Z` | `28.136` | `REPLICATE_MIXED_STATE_FIRST_ROUTE_SECONDARY_CAUSALITY` | `0fc45dae177d` |
| 26 | `EXPERIMENT_START` | `LR09_STRATIFIED_RESCUE_CONFIDENCE` | `2026-07-21T17:46:51Z` | `` | `0.000` | `` | `5b7244f56793` |
| 27 | `EXPERIMENT_COMPLETE` | `LR09_STRATIFIED_RESCUE_CONFIDENCE` | `2026-07-21T17:46:51Z` | `2026-07-21T17:46:54Z` | `2.945` | `ESTABLISH_CONDITIONAL_ROUTE_AMPLIFICATION_LAW` | `024ed1dd40bf` |
| 28 | `EXPERIMENT_START` | `LR10_UPSTREAM_REPRESENTATION_BYTE_AUCTION` | `2026-07-21T17:50:04Z` | `` | `0.000` | `` | `12440ecbe17d` |
| 29 | `EXPERIMENT_COMPLETE` | `LR10_UPSTREAM_REPRESENTATION_BYTE_AUCTION` | `2026-07-21T17:50:04Z` | `2026-07-21T17:50:10Z` | `6.376` | `PROMOTE_TO_HELDOUT_F2` | `8885c635b37c` |
| 30 | `EXPERIMENT_START` | `LR11_UPSTREAM_RESIDUAL_HELDOUT_F2` | `2026-07-21T17:53:29Z` | `` | `0.000` | `` | `584b1a83db73` |
| 31 | `EXPERIMENT_COMPLETE` | `LR11_UPSTREAM_RESIDUAL_HELDOUT_F2` | `2026-07-21T17:53:29Z` | `2026-07-21T17:54:31Z` | `61.636` | `PROMOTE_UPSTREAM_RESIDUAL_TO_REPLICATION` | `5cd0767fff37` |
| 32 | `EXPERIMENT_START` | `LR12_UPSTREAM_RESIDUAL_REPLICATION` | `2026-07-21T17:56:38Z` | `` | `0.000` | `` | `930676e54002` |
| 33 | `EXPERIMENT_COMPLETE` | `LR12_UPSTREAM_RESIDUAL_REPLICATION` | `2026-07-21T17:56:38Z` | `2026-07-21T17:57:11Z` | `33.744` | `RETIRE_UPSTREAM_RESIDUAL_REPLICATION_FAILED` | `7c67e5447b67` |
| 34 | `EXPERIMENT_START` | `LR13_REGION_CLOSURE_AUDIT` | `2026-07-21T18:00:15Z` | `` | `0.000` | `` | `7889f491ed5e` |
| 35 | `EXPERIMENT_COMPLETE` | `LR13_REGION_CLOSURE_AUDIT` | `2026-07-21T18:00:15Z` | `2026-07-21T18:00:16Z` | `1.230` | `TESTED_LINEAR_REPAIR_REGION_CLOSED` | `2025ee964deb` |
| 36 | `GUARD_THRESHOLD_CHANGE` | `LR14_DISK_FLOOR_POLICY` | `2026-07-21T18:14:12Z` | `2026-07-21T18:14:12Z` | `0.000` | `STOP_EARLY_PRESERVE_SOURCE_CONTROLLER_AND_MOP` | `9dcf74d6e5d8` |

## Evidence seals

- `KIMI_K26_LONG_RUN_BASELINE_AUDIT.json`: `2935c1ffc9833f44ac6127818d804097e589ba8431899244038768546d2baf31`
- `KIMI_K26_LONG_RUN_CONTROL_CAUSAL.json`: `b7acbc36f72d48b318b0cd52ccb2f524d7b29a4455b08a033ccbfab6bd90812d`
- `KIMI_K26_LONG_RUN_INTERVENTION_REPLICATION.json`: `3c3bd634ef7cfcbb634018ee990d3ec499b6910c712af277379e626735eb5eda`
- `KIMI_K26_LONG_RUN_LARGE_FALSIFICATION.json`: `0930b2c83fb55a894cd00d558e7dcd1a83497c12737a1691c522eddfa49dda55`
- `KIMI_K26_LONG_RUN_POOLED_CAUSAL_LAW.json`: `663c70385107c20c2da80bd0ae510b950b3fb174b63915a33fe42b0b029fefc0`
- `KIMI_K26_LONG_RUN_REPAIR_BRACKET.json`: `454402d139356a1abcfa426ace236f5701a56cf968bff04c9ff26c5634145c6a`
- `KIMI_K26_LONG_RUN_REPLICATION.json`: `07136a12d4394a6830fc7cbc697c90e612231f53ac4284bbffb767543d840935`
- `KIMI_K26_LONG_RUN_SHRINKAGE_BOUNDARY.json`: `9093a6b74c7e3881dd117f24cf4e4651cd7c89938929f18d03ade42464e5024a`
- `KIMI_K26_LONG_RUN_STRATIFIED_RESCUE.json`: `df3c4cc62fff7f9e3c25113d076e1603da0d5b64bb1551513b5406281476b97e`
- `KIMI_K26_LONG_RUN_UNTOUCHED_VALIDATION.json`: `5ef10b448d17722510b1741b3fc368ab7ce7bfb02caf8b4fef541af35164a6c8`
- `KIMI_K26_LONG_RUN_UPSTREAM_AUCTION.json`: `f98e81213a7525008ed5130799d7abd35b6c19428c84ef31ca5e1da8cdf37caf`
- `KIMI_K26_LONG_RUN_UPSTREAM_F2.json`: `254122fff882c136ec93f6f56cefd7beaeecc5aea58381ef69ee5828d3d7ffbb`
- `KIMI_K26_LONG_RUN_UPSTREAM_REPLICATION.json`: `810a67028e0232795d67473cc9529700d7c7b9677a7342d44f416b7fb13b97a1`

## Next experiment

Test a representation-side nonlinear structural allocation at F0/F1 that directly reduces compact expert-output state error before any router. Do not spend more bits on generic downstream low-rank Doctor paths; require disjoint-score F1 evidence before F2.

Do not resume heavy work until at least the reported disk deficit has been recovered without deleting the sole Kimi source or MOP.

```text
wall-clock managed: 01:23:53.33
scientific compute: 00:09:28.75
waiting/verification: 01:14:24.58
experiments completed: 13
best candidate/BPW: P1_DUAL_PATH_RECOVERY_R16X2_LOCAL_F1_ONLY / 0.908590952555
F2 result: no promotable candidate; upstream residual replication CI crossed zero and layer 3 worsened
primary causal diagnosis: UPSTREAM_STATE_PRIMARY_ROUTE_SECONDARY_CONDITIONAL_AMPLIFIER
strongest rescue: teacher hidden restoration / 100.0%
replication status: causal ordering replicated; all tested physical repairs retired
controller PID/heartbeat/lease: 13332 / current / matched
commit pushed: True 15373ec59dbda67417d5411c4adebdf654e85196
next experiment: nonlinear representation-side structural allocation at F0/F1 after disk recovery
```
