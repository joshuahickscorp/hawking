# Q30 broader activation capture — lane verdict

**Status:** `BLOCKED_ON_METAL_CAPTURE_GATE_PROFILE_REQUIRED`  
**Base:** `13a87091` (activation-aware probe integrated)  
**Purpose:** lower the constant-mean null with a broader real L0 activation set, then re-price families

## Headline

1. **Baseline null (three-prompt HCLI capture), reported first:** mean high-hit null **0.957** (prior lane quoted **0.942** on under-ceiling family rows). Still a null trap.
2. **Broader capture input is prepared and source-bound:** **32 probes**, **3929 tokens**, domains mix code/prose/JSON/multi-turn/long-context/math/instruction. Not yet executed on Metal.
3. **Metal capture refused in this sandboxed executor:** `metal: no Metal-capable GPU` (MTL device = nil). Production server `:18430` stayed ready; no lease taken; incomplete run dir removed.
4. **Family re-price and BPW reachability for coherence:** **cannot be priced on this evidence** until the broad capture runs. Do not invent numbers past the measured baseline.

## Why broaden (from prior lane, unchanged)

- Incumbent `raw_weight_low_rank_q`: mean surplus **−0.155**, beats null on **0.00** of high-hit experts.
- `activation_weighted_svd_low_rank_q`: surplus **+0.039**, beats null **1.00**, weight cosine worse (0.464 vs 0.681).
- Coherence still not bought: mean null **0.942**, best under-ceiling surplus thin; even at **~4.87 BPW** surplus only **+0.022**.
- Stated limit: **three prompts** → near-constant expert outputs → enormous null.

## Capture provenance (prepared)

| item | value |
|---|---|
| input | `requests/QWEN30_BROAD_ACTIVATION_L0_ROUTE_CAPTURE_INPUT_901a24bdcfc6c1d2.json` |
| input sha256 | `8c5ff2d8490f716b1a14fe04c3606472dca5d0da9df20356abe804dedde3151c` |
| tokenizer | source Qwen3-Coder `tokenizer.json` (sha in PREPARE_PROVENANCE) |
| chat shape | one user message, no system, no tools (`<|im_start|>user…assistant`) |
| probes | 32 |
| total tokens | 3929 |
| domains | code 7, prose 5, structured 5, multi_turn 3, long_context 2, math 3, instruction 3, dialogue 1, list 1, mixed 2 |
| claim | diagnostic activation pricing only; not HCLI/coherence/TPS/capability |
| protocol | broad schema additive; three-probe HCLI schema **unchanged** (still exactly 3 protected probes) |

## Null-first table

| capture set | probe count | total tokens | mean null high-hit | verdict |
|---|---:|---:|---:|---|
| baseline three-prompt HCLI | 3 | ~1115 | **0.957** | null trap (prior 0.942 class) |
| broad activation v1 | 32 | 3929 | **not measured** | Metal blocked |

If the broad null does **not** fall materially below **0.942** (roughly ≥0.05 drop), that is the headline finding: capture strategy still wrong.

## Exact owner command (serialized Metal / gate profile)

```bash
/Users/scammermike/.claude-grok/worktrees/q30-broader-capture-20260809-154213/workspace/ops/build/rust/debug/examples/ascension_qwen30_current_hcli_layer0_route_capture \
  --manifest /Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity/QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json \
  --expected-manifest-seal-sha256 3321a99d719e70499663b7bfebe14dd6c732bfc533bb05b9277eb398e44d6357 \
  --expected-source-audit-seal-sha256 00ed3e495416c2cbafbcdb7800528e15f243b1a13f5f4af13240109c8fc69f7b \
  --expected-source-revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
  --input-json /Users/scammermike/.claude-grok/worktrees/q30-broader-capture-20260809-154213/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1/requests/QWEN30_BROAD_ACTIVATION_L0_ROUTE_CAPTURE_INPUT_901a24bdcfc6c1d2.json \
  --output-dir /Users/scammermike/.claude-grok/worktrees/q30-broader-capture-20260809-154213/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1/runs/8c5ff2d8490f716b_94c3f75f83dce25a \
  --max-seq-len 2048
```

Then (CPU-only; no GPU):

```bash
cd /Users/scammermike/.claude-grok/worktrees/q30-broader-capture-20260809-154213
lab/operators/q30_broad_activation_after_capture.sh \
  workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1/runs/8c5ff2d8490f716b_94c3f75f83dce25a
```

**Serialize against production server if dual Q30 Metal residency is unsafe.** Do not restart `:18430` from this lane unless you choose to.

## BPW / coherence answer (on evidence available now)

**Cannot state a BPW where activation-aware families achieve both positive surplus over a low null and operator recovery.**

- Prior lane (three-prompt): not under 1.5 BPW; high-hit joint surplus+operator not on the grid through ~4.9 BPW.
- This lane: broad null and re-price **not measured** (Metal blocked). Extrapolation refused.

## Artifacts

- `lab/operators/q30_broad_activation_route_capture_prepare.py` — corpus + tokenized input
- `lab/operators/q30_activation_null_first_report.py` — null-before-families
- `lab/operators/q30_broad_activation_after_capture.sh` — null then existing family probe
- `crates/hawking-core/examples/ascension_qwen30_current_hcli_layer0_route_capture.rs` — additive broad schema (min 12 probes); legacy 3-probe path intact
- `null-first/NULL_BASELINE_THREE_PROMPT.json`
- `RUN_CAPTURE.command.txt`, `STATUS.md`, this file

## Claim boundary

- Diagnostic preparation + baseline null only until Metal capture completes
- No gate weakened
- No full-model pack
- No server restart, no exclusive lease from this executor
- Negative / blocked results are the deliverable when that is the evidence
