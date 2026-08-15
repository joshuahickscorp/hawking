# Ascension 30B Parity Ladder Plan

**Status:** PLAN + HARNESS SCAFFOLD ONLY  
**Gate:** Proto-Frankenstein offload complete before any Qwen download / Gravity work  
**Bible:** HAWKING_ASCENSION_BIBLE §8 (Bootstrap model 1 — Qwen3-Coder-30B), §29–§31  
**Scaffold code:** `lab/operators/ascension_parity_ladder.py`  
**Scaffold tests:** `lab/tests/test_ascension_parity_ladder_scaffold.py`

---

## 1. Purpose

Build the **exact-model** Gravity path for `Qwen/Qwen3-Coder-30B-A3B-Instruct`
as the first self-optimization vehicle and first production HCLI executor.

This document plans the **parity ladder** and **Gravity ladder** only. It does
**not** authorize streaming weights, Metal kernels against Qwen tensors, or
BASE_TRUE_TPS claims. Those start after Proto-Frankenstein offload frees the
machine and a revision is pin-sealed at download time.

---

## 2. Role of the 30B (bible §8)

| Role | Meaning |
|------|---------|
| Executor | Primary coding / agent workhorse |
| First self-optimization vehicle | Self-TG gauntlet subject (see TG plan) |
| Lighter scientific instrument | Cheaper architecture experiments than 80B |
| First production HCLI model | Chat / tool / JSON surface after P11+ |

Family key for the shared harness: **`QWEN3_MOE`**.

---

## 3. Reference discipline (DSV4F tonight)

This ladder is **the same methodology** run against DeepSeek-V4-Flash tonight.
Do not invent a second vocabulary.

| DSV4F pattern | Reference | 30B generalization |
|---------------|-----------|--------------------|
| `NumericParityV21Only` | `crates/hawking-core/src/gravity_deepseek_v4_p4b_device.rs` | `ParityClassification.NUMERIC_PARITY_V2_1_ONLY` |
| Complete-token stages | `lab/operators/deepseek_v4_gravity.py::_COMPLETE_TOKEN_PROFILE_STAGES` | `QWEN3_MOE_STAGES` |
| Profile tests | `tools/condense/tests/test_deepseek_v4_complete_token_profile.py` | `CompleteTokenProfiler` + TG plan |
| `BASE_TRUE_TPS_WITHHELD` | child baseline / scoreboard receipts | default scoreboard cells |
| `PASS_FULL_STACK` | `frankenstein_teacher_forced_executor.py` | `RungStatus.PASS_FULL_STACK` |
| `claim_boundary` | every sealed DSV4F receipt | `default_claim_boundary()` |
| fallback=0 + real GPU | seal paths / metal blocks | `promote_rung_status()` |
| Honest refusal | no fake full runtime | `REJECT_WEIGHTS_ABSENT` until stream |

---

## 4. Exact-model operator surface (build list)

From bible §8 — each item becomes a Gravity/kernel contract after stream:

```text
tokenizer/template
embedding
RMSNorm
QKV
RoPE
KV
attention
router/top-8
expert gather
gate/up
activation
down
route combine
residual
final norm
lm_head
top-k/sampling
HCLI streaming
```

Scaffold complete-token stages: `QWEN3_MOE_STAGES` in
`lab/operators/ascension_parity_ladder.py`.

---

## 5. Parity ladder (P0–P13)

| Rung | Name | Weights | GPU | fallback=0 | Notes |
|------|------|---------|-----|------------|-------|
| P0 | tokenizer/template | no* | no | no | *tokenizer files only; pin SHA |
| P1 | embedding/norm | yes | no† | no | †device path optional early |
| P2 | QKV/RoPE/KV | yes | preferred | no | |
| P3 | attention | yes | preferred | no | |
| P4 | router/top-k | yes | preferred | no | top-8 for 30B MoE |
| P5 | one expert | yes | preferred | no | |
| P6 | full MoE | yes | preferred | no | + shared expert |
| P7 | one layer | yes | preferred | no | residual complete |
| P8 | early/middle/late | yes | preferred | no | depth sampling |
| P9 | first token | yes | **yes** | **yes** | Numeric V2.1 → then full |
| P10 | continuation/full logits | yes | **yes** | **yes** | |
| P11 | tool/JSON/edit behavior | yes | **yes** | **yes** | capability protected |
| P12 | long generation | yes | **yes** | **yes** | |
| P13 | restart/reload | yes | **yes** | **yes** | session + weight reload |

### Promotion rules (from DSV4F)

1. `fallback_count != 0` → `REJECT_FALLBACK_NONZERO` (never promote).
2. GPU rung with `gpu_dispatches == 0` → `REJECT_NO_REAL_GPU_DISPATCH`.
3. Numeric pass without full residual chain → `PASS_NUMERIC_V2_1_ONLY`
   (not exact-storage, not full stack).
4. Full residual + capability + fallback=0 + real GPU → `PASS_FULL_STACK`.
5. No weights → `REJECT_WEIGHTS_ABSENT` / `SCAFFOLD_PENDING` (current state).

Stub test functions (ready to fill after stream):

- `test_p0_tokenizer_template_scaffold` … `test_p13_restart_reload_scaffold`
  in `lab/tests/test_ascension_parity_ladder_scaffold.py`.

---

## 6. Gravity ladder

Produce, in order:

```text
source authority
→ quality anchor
→ performance anchor
→ Gravity equilibrium artifact
```

**Target:** lowest capable, runnable equilibrium.

**Forbidden:** a universal 1.5-BPW requirement. Per-model equilibrium may land
above or below any single BPW number; the harness hard-codes
`universal_1_5_bpw_required: false`.

---

## 7. Model ladder + rotation (bible §30–§31)

Pipeline (shared with 80B and later families):

```text
DISCOVER → PREFLIGHT → RESEARCH_DISTINCTION → DOWNLOAD_STREAM → GRAVITY
→ LOAD → PARITY → CAPABILITY → PROFILE → OPTIMIZE → REVIEW → REPORT
→ SEAL → EVICT → ROTATE
```

**Current phase for 30B:** `PREFLIGHT` (harness scaffold). Next after
Proto-Frankenstein offload: `DOWNLOAD_STREAM` with revision pin + SHA seal.

**Rotate when:**

- **A.** model descends ≥1 named TG rung, **or**
- **B.** two materially different optimization architectures fail **and**
  same-model roofline is measured **and** bottleneck is sealed **and**
  the smallest next representation change is named.

At TG3: always stop for human/controller review (see TG plan).

---

## 8. Receipt schemas (scaffold)

| Schema | Purpose |
|--------|---------|
| `hawking.ascension.parity_ladder_receipt.v1` | Full 14-rung inventory |
| `hawking.ascension.parity_rung_receipt.v1` | One rung |
| `hawking.ascension.gravity_ladder_receipt.v1` | Gravity stage track |
| `hawking.ascension.model_ladder_pipeline.v1` | DISCOVER…ROTATE cursor |

All sealed via `lab.receipts.seal` / `verify`.

---

## 9. Work sequence (when gated open)

1. Pin HF revision; seal tokenizer + config SHA (P0).
2. Stream weights under floor policy; no concurrent Frankenstein retention.
3. Build source-authority oracle (CPU) for P1–P7.
4. Device path with Numeric Parity V2.1 only until exact-storage sealed.
5. P8 depth sampling → P9 first token → P10 continuation.
6. P11 capability suite (tool/JSON/edit) under protected gates.
7. P12 long gen + P13 restart.
8. Gravity equilibrium artifact; enter Self-TG (TG plan).
9. Stop at TG3 for human review.

---

## 10. Non-goals (this document / this session)

- No Qwen download or Xet stream.
- No Gravity against real 30B weights.
- No BASE_TRUE_TPS measurement or TG rung claim.
- No edits to `lab/operators/frankenstein_*` or DSV4F crate sources.
- No push / PR / remote / detached daemon / committed venv.

---

## 11. Acceptance for the scaffold (done when)

- [x] Family-parameterized harness exists (`ParityLadderHarness`).
- [x] P0–P13 stub tests named and sealed with honest refusal.
- [x] Gravity ladder + pipeline receipts seal.
- [x] Claim boundaries forbid universal 1.5 BPW and live work claims.
- [ ] Live P0 after tokenizer on disk (post-gate).
- [ ] Live P1–P13 against streamed weights (post-gate).
