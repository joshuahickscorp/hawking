# Ascension 80B Hybrid Architecture Plan

**Status:** PLAN + HARNESS SCAFFOLD ONLY  
**Gate:** Proto-Frankenstein offload; prefer 30B P0–P7 green before heavy 80B stream  
**Bible:** HAWKING_ASCENSION_BIBLE §9 (Bootstrap model 2 — Qwen3-Coder-Next 80B), §29–§31  
**Scaffold code:** `lab/operators/ascension_parity_ladder.py` (`QWEN3_NEXT`)  
**Scaffold tests:** `lab/tests/test_ascension_parity_ladder_scaffold.py` (state gates)

---

## 1. Purpose

Treat **Qwen3-Coder-Next (80B-class)** as a **distinct architecture family**, not
a larger 30B. It is the reviewer, stronger architecture challenger, and second
self-optimization vehicle.

Build an **exact-model production path before family generalization**. Port every
applicable 30B win; do not assume attention/MoE kernels transfer unchanged.

---

## 2. Role of the 80B (bible §9)

| Role | Meaning |
|------|---------|
| Reviewer | Adversarial / stronger check on 30B outputs and kernels |
| Architecture challenger | Hybrid state-space + gated attention stress |
| Second self-optimization vehicle | Independent Self-TG subject |

Family key: **`QWEN3_NEXT`**.

HF identity is pinned at stream time (do not leave floating `main`). Scaffold
placeholder: `Qwen/Qwen3-Coder-Next` in `BOOTSTRAP_TARGETS` — replace with the
exact official Instruct revision SHA when admitted.

---

## 3. Why this is not “30B with more experts”

| Dimension | QWEN3_MOE (30B) | QWEN3_NEXT (80B) |
|-----------|-----------------|------------------|
| Attention | Standard GQA/MHA path | Hybrid **3× DeltaNet / 1× gated attention** |
| State | KV-centric | **Gated DeltaNet state** + KV |
| Routing | top-8 (scaffold default) | **512-expert top-10** + shared expert |
| Kernels | MoE + attention | DeltaNet update + gated attention + MoE |
| Extra gates | P0–P13 only | P0–P13 **+ SG0–SG6 state gates** |

Family kernel architecture (bible §29) later generalizes into `QWEN3_NEXT`
and `STATE_SPACE_HYBRID` megakernel grammars — only after exact-model success.

---

## 4. Required architecture work (bible §9)

```text
Qwen3-Next parser
Gated DeltaNet state
DeltaNet update kernels
hybrid 3-DeltaNet / 1-gated-attention schedule
gated attention
512-expert top-10 routing
shared expert
state/KV management
hybrid command graph
Gravity support
```

Scaffold constant: `QWEN3_NEXT_ARCHITECTURE_REQUIREMENTS`.

### Hybrid schedule contract

Per layer slot index `i` in the repeating unit:

```text
if i % 4 == 3:  gated attention (+ KV)
else:           Gated DeltaNet update
```

(Exact layer map is **source-bound** after parser admission — do not hard-code
counts from rumor; seal the schedule from config/source.)

---

## 5. Required state gates (SG0–SG6)

| Gate | Name | What must be proven |
|------|------|---------------------|
| SG0 | state_initialization | Zero/init contract for Gated DeltaNet state |
| SG1 | chunk_recurrent_equivalence | Chunked update ≡ recurrent update (parity) |
| SG2 | incremental_decode_parity | Step decode ≡ full recompute at boundary |
| SG3 | context_extension | Extend context without silent state corruption |
| SG4 | restart_reset | Restart/reset clears or reloads state correctly |
| SG5 | long_generation_stability | State stays finite / coherent over long gen |
| SG6 | state_memory_accounting | Bytes for state + KV ledgered (no silent growth) |

Scaffold receipts: `ParityLadderHarness.stub_state_gate_receipt` with schema
`hawking.ascension.state_gate_receipt.v1`.

Without weights → `REJECT_WEIGHTS_ABSENT` (current tests).

---

## 6. Complete-token stages (Next-specific)

`QWEN3_NEXT_STAGES` extends the MoE inventory with:

```text
qkv_or_deltanet_proj
gated_deltanet_state
deltanet_update
hybrid_schedule_slot
gated_attention
router_top10
state_memory_accounting
```

Profiler shape reuses the DSV4F complete-token pattern
(`_DiagnosticTokenProfiler` → `CompleteTokenProfiler` in
`lab/operators/ascension_tg_gauntlet.py`).

---

## 7. Parity ladder reuse

Use the **same P0–P13 skeleton** as the 30B plan, parameterized by family:

```python
ParityLadderHarness(family=ModelFamily.QWEN3_NEXT)
```

Differences in rung **bodies** (not IDs):

| Rung | 80B specialization |
|------|--------------------|
| P2 | Project into DeltaNet and/or QKV depending on schedule slot |
| P3 | DeltaNet update **or** gated attention per schedule |
| P4 | top-10 over 512 experts |
| P6 | shared expert + top-10 combine |
| P12 / SG5 | long-gen **and** state stability |
| P13 / SG4 | restart **and** state reset |

Classification remains `NUMERIC_PARITY_V2_1_ONLY` until exact-storage e2e is
sealed (same honesty as `DeepSeekV4P4bParityClassification`).

---

## 8. Port from 30B (transfer, do not fork blindly)

Port only after 30B sealed wins:

| 30B win class | Transfer condition |
|---------------|--------------------|
| Tokenizer admission pattern | Same seal/claim_boundary discipline |
| RMSNorm / residual graph | Geometry match |
| Expert gate/up/act/down | Width/dtype match; re-parity |
| MoE gather/combine | top-k and expert count differ — rebind |
| HCLI streaming spine | Endpoint yes; template/source re-admit |
| Metal command topology lessons | Rebuild graph for hybrid schedule |
| Quant / equilibrium craft | Re-run Gravity ladder; no universal BPW |

Do **not** port attention kernels as DeltaNet substitutes.

---

## 9. Family kernel path (bible §29) — after exact-model

Only after exact 80B path:

```text
shared semantic runtime
architecture-family execution graphs   # QWEN3_NEXT
generated geometry variants
exact-model fast paths where justified
```

Kernel selection key includes: family semantics, operator grammar, tensor dims,
representation, active experts, batch/session, context regime, device generation,
memory pressure.

Promote megakernels by **complete wall time and p99**, not microbench alone.

---

## 10. Work sequence (when gated open)

1. Research distinction receipt: what new geometry does Next test? (bible §30 questions).
2. Pin revision; stream under floor; seal parser + config.
3. P0 tokenizer/template; admit chat template separately if needed.
4. Implement Gated DeltaNet state + update kernels (CPU oracle first).
5. SG0–SG2 before multi-layer claims.
6. Hybrid schedule graph; P3/P7 for both slot kinds.
7. 512-expert top-10 + shared expert (P4–P6).
8. Full layer residual → multi-layer → first token (P7–P9).
9. SG3–SG6 + P12–P13.
10. Port applicable 30B Gravity wins; new equilibrium artifact.
11. Self-TG independently; TG3 human review.

---

## 11. Non-goals (this document / this session)

- No 80B download or hybrid kernel implementation against real weights.
- No claim that 30B attention kernels satisfy Next.
- No family megakernel promotion before exact-model parity.
- No Frankenstein / DSV4F crate edits.
- No push / PR / remote.

---

## 12. Acceptance for the scaffold (done when)

- [x] `QWEN3_NEXT` family key + stages + architecture requirements.
- [x] SG0–SG6 stub receipts + tests.
- [x] Distinct complete-token inventory from MoE.
- [x] Shared P0–P13 ladder parameterization.
- [ ] Parser + source schedule seal (post-stream).
- [ ] Live state gates + hybrid graph (post-stream).
