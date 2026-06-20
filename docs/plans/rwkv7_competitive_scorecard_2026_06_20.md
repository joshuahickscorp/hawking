# RWKV-7 Competitive Scorecard
**Date:** 2026-06-20 14:54 UTC

## Verdict Gates

| Lane | Verdict | Evidence |
|---|---|---|
| Quality | **PENDING** | Current checkpoint PPL 11.45; final G1a checkpoint not reached yet. |
| Low-bit quant | **PENDING** | No exported TQ artifact yet. |
| Draft accept | **PENDING** | No custom draft eval logs yet. |
| Spec physics | **PENDING** | Draft eval logs are not available to the hardening model yet. |
| llama.cpp comparison | **PENDING** | No clean-room llama head-to-head log parsed. |

## G1a State

| Metric | Value |
|---|---:|
| step | 90 / 150 |
| loss | 2.7420 |
| loss_ema | 5.9178 |
| watcher PPL | 11.45 |
| recent min/step | 17.9 |
| ETA remaining | 17.9h |

## Draft Sweep

| Variant | Step | PPL | Accept rate | Params M |
|---|---:|---:|---:|---:|
| draft_35m_probe | pending | pending | pending | pending |
| draft_50m_probe | pending | pending | pending | pending |
| draft_75m_probe | pending | pending | pending | pending |
| draft_100m | pending | pending | pending | pending |
| draft_150m | pending | pending | pending | pending |
| draft_200m | pending | pending | pending | pending |
| draft_300m | pending | pending | pending | pending |

## llama.cpp Head-to-Head

- Log: `/Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/llama_qwen_head_to_head.log`
- Single-stream hawking: pending tok/s
- Single-stream llama.cpp: pending tok/s
- Ratio: pendingx

## Low-Bit Quant

- Artifact count: 0
- Latest v2 report: `/Users/scammermike/Downloads/hawking/docs/plans/g1a_v2_expansion_results_2026_06_20.md`
- TQ loader pass: False
- TQ bench pass: False

## Spec Physics

| Variant | Effective TPS | vs llama | Status |
|---|---:|---:|---|
| draft_35m_probe | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_50m_probe | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_75m_probe | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_100m | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_150m | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_200m | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_300m | pending | pending | **PENDING**: no accept-rate eval yet |

## Development Backlog

1. Finish G1a and keep the watcher PPL gate as the quality source of truth.
2. Finish TQ export/loader/dispatch after G1a final if the quality gate passes.
3. Let the 100/150/200/300M sweep finish, then extend the winner instead of all four.
4. Do not shrink yet: no evaluated draft clears the spec physics gate.
5. First improve accept rate with target-logit KD, then re-run this hardening pass.
6. Once a draft passes, launch the nearest smaller configured probe with `DRAFT_VARIANTS="draft_75m_probe draft_50m_probe"` before scaling anything up.
7. Run `G1A_V2_LLAMA_BASELINE=1` in a clean room; this is the claim gate.

## Claim Rule

Do not claim a llama.cpp win until quality, low-bit quant, draft accept, spec physics, and clean-room llama head-to-head are all green in this report.
