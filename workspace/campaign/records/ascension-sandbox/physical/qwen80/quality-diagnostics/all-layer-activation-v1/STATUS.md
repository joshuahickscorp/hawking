# Q80 all-layer activation capture — readiness

**Verdict:** `ALL_LAYER_ACTIVATION_CAPTURE_NOT_YET_POSSIBLE`

## Headline (null first, families later)

Per-layer nulls: **not measured** — no all-layer (or any multi-token) activation
matrix exists for Q80. This is not a null-trap finding; it is a **missing instrument** finding.

## Baseline artifact (admitted, low fidelity)

| field | value |
|---|---|
| complete_physical_bpw | **1.1331148544404688** |
| ceiling | 1.5 |
| tensor_count | 74391 |
| status | `CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED` |

## Why all-layer capture is blocked

1. **GQA full-layer same-runtime encode absent** (ready 0/12). Layers [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47] refuse at CPU preflight before lease. Authority: `QWEN80_MULTI_LAYER_GQA_ENCODE_GAP_20260809T210000Z.json` status `EARNED_NEGATIVE_GQA_FULL_LAYER_SAME_RUNTIME_ENCODE_ABSENT`.
2. **Device multi-layer earned prefix is L0..L2 only** (3/48 DeltaNet).
3. **Existing L0/L1/multi-layer captures are single-token component parity** — second residual sha + max_abs_error only; no router-input f32 rows for fit.
4. **No Q80 streamed BF16 source teacher** and no multi-token CPU hybrid capture chain.

## Cheapest honest path

| goal | path | ready now? |
|---|---|---|
| instrument calibration on 3 layers | Metal multi-token L0..L2 after new capture binary | path exists; binary absent |
| **coherence packing** | Metal all-48 after GQA encode + broad capture binary | **no** |

Fitting on L0-only or on component captures is **refused** (Q30 coverage failure class).

## What lands when GQA encode is ready

1. Owner runs Metal broad all-layer capture (see `RUN_CAPTURE.command.txt`).
2. `python3 -m lab.operators.q80_activation_null_first_report` — per-layer nulls **before** any family row.
3. `python3 -m lab.operators.ascension_qwen80_activation_weighted_svd_repack` — surplus-first under 1.5 BPW with coverage receipt.
4. Owner admits and serves on a **new** port; text generation required before any coherence claim.

## Claim boundary

- No server started, no exclusive GPU lease, no TPS.
- No family table invented.
- Negative result is the valid deliverable for this open question.

## Commit status

Worktree files are complete on disk. Sandbox blocked writes to `Downloads/hawking/.git` (index.lock Operation not permitted). Run `./APPLY_COMMIT.sh` from this worktree under an unsandboxed/gate shell to seal the branch commit.
