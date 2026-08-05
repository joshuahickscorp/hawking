# Functional Transfer Program — Scaffold

**Status:** SCAFFOLD SEALED (framework/formats/harnesses real; runtime stages gated)  
**Seal companion:** `workspace/campaign/evidence/models/frankenstein/FUNCTIONAL_TRANSFER_PROGRAM.json`  
**Related:** `FRANKENSTEIN_PROGRAM.md` (multi-stage product path); this doc is the
**trained functional-transfer** lane that must exist before any honest
`PROTO_FRANKENSTEIN` capability claim.

---

## Owner's steer (settled — do not relitigate)

The hardened linear GLM → DSV4F mapping is **infrastructure and initialization,
not sufficient inheritance**. Do **not** declare `PROTO_FRANKENSTEIN` from
projected weights. Keep the mapping run; seal its output as
**`LINEAR_SUBSPACE_INITIALIZATION`** and use it for layer-cartography and bridge
initialization only.

Target framing: DSV4F already has strong raw math; GLM's contribution is
long-horizon reasoning, tool-grounded reasoning, coding/agentic reliability, and
method discipline. Adapters must be small, reversible, hash-bound, independently
ablatable, and Gravity-accounted.

---

## What the linear mapping is

| Label | Meaning |
|-------|---------|
| `LINEAR_SUBSPACE_INITIALIZATION` | Closed-form weight-space Gram/PCA + projection + residual steering module |
| **Not** `PROTO_FRANKENSTEIN_COMPLETE` | Never sealed from projection alone |

Use linear init for:

- layer-correspondence cartography warm-start
- nonlinear bridge / residual weight initialization

Do **not** use it for math capability claims, promotion, or inheritance receipts
that imply functional transfer.

---

## Seven-layer transfer stack

| # | Layer | Built now? | Gate if incomplete |
|---|-------|------------|--------------------|
| 1 | Freeze BASE_DSV4F capability/runtime/routing/HCLI baseline | **Yes** (measurable fields; rest PENDING) | — |
| 2 | Paired GLM/DSV4F evidence (disjoint memberships) | **Format + membership** | `REQUIRES_GLM_RUNTIME`, `REQUIRES_BENCHMARK_CORPUS` |
| 3 | Tokenizer-independent alignment (spans/bytes/actions/tools) | **Yes** | Capture side needs GLM |
| 4 | Layer correspondence (CKA, CCA/Procrustes, causal-trace scaffold) | **Framework** (synthetic OK) | Live GLM side: `REQUIRES_GLM_RUNTIME` |
| 5 | Reversible nonlinear bridges (norm→proj→gatedMLP→low-rank) | **Architecture + apply/revert** | Fit: `REQUIRES_TRAINING_LOOP` |
| 6 | Distilled adapters (method/decomp/formal/repair/value/route-bias) | **Architecture + bank** | Fit: `REQUIRES_TRAINING_LOOP` |
| 7 | Verified expert iteration + A–G ablation + promotion gate | **Harnesses** | `REQUIRES_VERIFIER`, training, corpus |

---

## Stages 1–12 (owner list) — scaffold mapping

1. **Baseline freeze** → `frankenstein_baseline_freeze.py`
2. **Paired evidence** → `frankenstein_trace_format.py` (capture fail-closed)
3. **Alignment** → `frankenstein_aligner.py` (never token IDs)
4. **Cartography** → `frankenstein_cartography.py`
5. **Nonlinear bridges** → `frankenstein_bridges.ReversibleBridge` (+ fit gated)
6. **Adapters** → `frankenstein_bridges.build_adapter_bank` (+ fit gated)
7. **Native routing preserved** → route-bias residual only; no GLM router copy
8. **Verified expert iteration** → `frankenstein_verifier_loop.py` (fail-closed)
9. **A–G ablation** → `frankenstein_ablation.run_ag_ablation`
10. **Promotion** → `frankenstein_promotion_gate.evaluate_promotion` → **PENDING** until evidence
11. **Reject imitation-without-proof** → promotion + AG reject rules
12. **Secondary search/tool/formal gains** → measured separately from raw-model gains

---

## A–G ablation

| Arm | Name | Composition |
|-----|------|-------------|
| **A** | BASE | DSV4F only |
| **B** | Linear init | A + `LINEAR_SUBSPACE_INITIALIZATION` (not inheritance) |
| **C** | Behavior distill | + behavior distillation objectives |
| **D** | Nonlinear bridges | + trained reversible bridges |
| **E** | Router/method adapters | + method policy / route-bias residual (native routing) |
| **F** | Expert iteration | + verified expert iteration trajectories |
| **G** | Complete Proto-Frankenstein | full functional-transfer stack |

**Reject rule (additive-not-subtractive):** any secondary regression beyond sealed
tolerance → REJECT regardless of math gain. Also REJECT checkpoints that improve
imitation but fail proof/computation/repair/transfer/hidden eval.

---

## Frozen promotion gate

Promotion requires **all** of:

- held-out math gains
- measured recovery of the GLM-vs-DSV4F gap (**≥ 70%** initial band)
- coding / tool / agent / long-context non-regression
- stable routing
- exact provenance
- Gravity byte / TPS / p99 accounting
- independent challenge

**Current verdict without live scores: `PENDING`** (honest — never fabricated ACCEPT).

---

## Reality boundary (built vs gated)

### Built now (real code)

- `LINEAR_SUBSPACE_INITIALIZATION` labels on transfer + proto-run receipts
- BASE_DSV4F baseline freeze descriptor
- Paired evidence schema, membership manager, loaders/validators
- Tokenizer-independent aligner
- Cartography on synthetic paired activations
- Bridge + adapter architectures with apply/revert + byte accounting
- A–G ablation harness + reject rules
- Promotion gate + secondary non-regression suite framework
- Verifier-loop interface
- This program doc + sealed JSON

### Runtime-gated (fail closed — no fakes)

| Gate | Missing infra | Blocks |
|------|---------------|--------|
| `REQUIRES_GLM_RUNTIME` | No local GLM-5.2 serve (~1.5 TB) | GLM trajectories, activations, logits, live cartography, GLM critique |
| `REQUIRES_TRAINING_LOOP` | Forward-only DSV4F; no backward/optimizer | Bridge/adapter fit, route policy train, expert-iteration train step |
| `REQUIRES_VERIFIER` | No Lean/tool verifier wired | Stage-8 verified expert iteration |
| `REQUIRES_BENCHMARK_CORPUS` | No frozen held-out math suite | Live eval, promotion scores, disjoint membership eval |

---

## Operators

| Module | Role |
|--------|------|
| `lab/operators/frankenstein_gates.py` | Shared labels + gates |
| `lab/operators/frankenstein_baseline_freeze.py` | Stage 1 freeze |
| `lab/operators/frankenstein_trace_format.py` | Trace schema + membership |
| `lab/operators/frankenstein_aligner.py` | Span/action/tool aligner |
| `lab/operators/frankenstein_cartography.py` | CKA/CCA/Procrustes |
| `lab/operators/frankenstein_bridges.py` | Bridges + adapters + trainer iface |
| `lab/operators/frankenstein_verifier_loop.py` | Expert-iteration iface |
| `lab/operators/frankenstein_promotion_gate.py` | Promotion + secondary suite |
| `lab/operators/frankenstein_functional_transfer.py` | Program seal |
| `lab/operators/frankenstein_ablation.py` | A–B + A–G harnesses |
| `lab/operators/frankenstein_transfer.py` | Linear mapping (relabeled init) |
| `lab/operators/frankenstein_proto_run.py` | Structural compose (not PROTO complete) |

---

## Path to an actual trained transfer

1. Stand up GLM runtime **or** import sealed remote GLM traces (hash-bound).
2. Freeze held-out math corpus with disjoint memberships; fill BASE scores.
3. Capture DSV4F paired activations via existing BOS multi-layer forward hooks.
4. Run cartography; lock functional phase map (not layer ratios).
5. Land a DSV4F training loop (or off-box fit with sealed import of modules).
6. Fit bridges/adapters on held-out functional losses (not latent cosine alone).
7. Wire verifier/tool loop; run expert iteration; accumulate verified trajectories.
8. Execute A–G with live scores; pass promotion gate; only then claim PROTO.

Until then: **scaffold only**. Projection alone is not inheritance.
