# MOP → Ramanujan Reuse Assessment

**Scope:** read-only inspection of MOP at `/Users/scammermike/Downloads/mop`.  
**Constraint honored:** no repo writes, no tests, no services, no heavy compute.  
**Outputs allowed:** this file and `/tmp/MOP_RAMANUJAN_ASSESSMENT.json` only.

**Method:** start from MOP's declared authorities (`README.md`, `ARCHITECTURE.md`, `STATUS.md`), then follow live source under `src/mop/`. Where a directory name suggests substance but only `__pycache__` remains (collapse), that is noted: bytecode is historical recovery material, not a maintained import surface.

---

## Executive judgment

**The owner's hypothesis is mostly wrong as a transfer thesis.**

MOP's "whole thing" in the maintained architecture is **not** "non-LLM inputs that replace language models for math research." It is:

1. An **experimental validity kernel** (`mop.method`, `mop.science`, experiment declarations) that forces preregistration, SESOI, budgets, stop rules, claim ceilings, producer/verifier separation, null validity, and mutation-tested defect classes.
2. A **sensing / continual-learning substrate program** (video token models, STARSS23 audio adaptive-compute beds, EMNIST/HAR moldability) whose scientific outputs are largely **nulls or non-promotable mechanics demos**.

Ramanujan needs a frozen Director LLM, Lean-grounded verification, formal search, and multi-role mathematical research memory. **MOP does not contain Lean, Mathlib, proof search, formalizers, provers, math-specific retrieval, or Ramanujan-style roles/stores.** What transfers is **governance and evidence discipline**, not the substrate.

Shared vocabulary (`evidence`, `ledger`, `verifier`, `claim`, `budget`, `gate`) is abundant and **not evidence of reuse**. Below, every positive claim names a concrete path/symbol.

---

## Live vs collapsed code (important for reuse cost)

| Path | Live `.py`? | Implication for Ramanujan |
|------|-------------|---------------------------|
| `src/mop/method/**` | Yes | Primary reusable design + portable code |
| `src/mop/science/{budget,gating,statistics}.py` | Yes | Partially portable; domain-flavored |
| `src/mop/experiments/base.py` + `registry/experiments.yaml` | Yes | Declaration schema is high value |
| `src/mop/evidence.py`, `provenance.py` | Yes | Tiny utilities; portable |
| `src/mop/beds/starss23/**` | Yes | Best end-to-end producer/verifier **pattern**, audio-specific math |
| `src/mop/substrate/**` (maintained) | Partial | Video/latent substrate; wrong domain |
| `src/mop/falsification/`, `mechanisms/`, `ladder/`, `perspectives/`, `learning/` | **No `.py`** — only `__pycache__` | Ideas recoverable from bytecode/docs; **not importable as-is** |
| `forge/`, `frontier/`, `integrated/`, `proof/` | Mostly campaign JSON + scripts | Process artifacts, not math research runtime |

`ARCHITECTURE.md` states historical code is content-addressed in `MOP_COLLAPSE_STATE.json`; exact tagged bytes are not maintained beside current implementations. Treat collapsed packages as **design fossils**, not packages to `pip install`.

---

## Component matrix

### component: Architecture — frozen large Director model
```
component: Architecture — frozen large Director model
verdict: NO_OVERLAP
mop_artifact: NONE
what_ramanujan_would_actually_do: nothing (build/choose Director outside MOP)
adaptation_cost: N/A
evidence: MOP substrate is TinyVideoSubstrate / V-JEPA-style video tokens (`src/mop/substrate/custom_model.py::TinyVideoSubstrate`, `ModelSpec`) and STARSS23 audio gates, not a frozen LLM director. README frames MOP as substrate/learning hypothesis testing, not language-model orchestration.
confidence: high
```

### component: Architecture — small trainable retriever / formalizer / prover / repair / value
```
component: Architecture — small trainable retriever/formalizer/prover/repair/value system
verdict: NO_OVERLAP
mop_artifact: NONE for formalizer/prover/repair/value-of-math. Near-miss only: starss23 count_gate as tiny trained adaptive-compute policy (`src/mop/beds/starss23/count_gate.py` referenced from count_producer); construction_search arms exist only as collapsed pycache under `src/mop/mechanisms/__pycache__/construction_search_*.pyc`.
what_ramanujan_would_actually_do: nothing importable; optionally read count_gate only as "tiny policy under matched budget" design lore
adaptation_cost: full greenfield for all five specialist roles
evidence: No Lean/Mathlib/formalization symbols in live source. `grep` for lean/mathlib/formal/prover/sorry finds no math-proof machinery. Mechanisms package has no live .py.
confidence: high
```

### component: Architecture — Lean and exact tools
```
component: Architecture — Lean and exact tools
verdict: NO_OVERLAP
mop_artifact: NONE
what_ramanujan_would_actually_do: nothing
adaptation_cost: N/A
evidence: No Lean toolchain, no mathlib bindings, no `sorry`/axiom replay gates in maintained code. MOP "exact" means exact sign-flip permutation stats (`mop.science.statistics.exact_sign_flip`) and deterministic audio referees, not formal math.
confidence: high
```

### component: Architecture — proof / program / counterexample search
```
component: Architecture — proof, program, and counterexample search
verdict: NO_OVERLAP
mop_artifact: NONE (live). Collapsed toy only: `construction_search_impl` (bytecode) — subset search over "shadow coalitions," arms `no-search` / greedy / sample / exhaustive; claim scope explicitly "deterministic programmatic mechanics only; no capability claim."
what_ramanujan_would_actually_do: nothing for Lean tree/beam search; do not port construction_search as a proof engine
adaptation_cost: full greenfield (best-first/beam Lean, SAT/SMT, symbolic algebra, evolutionary islands, etc.)
evidence: Live source has no beam/tree Lean search, no SAT/SMT, no counterexample search for math. construction_search strings recovered from pycache only.
confidence: high
```

### component: Architecture — governed roles and evidence (shared spine)
```
component: Architecture — governed roles and evidence (system spine)
verdict: REUSABLE_WITH_ADAPTATION
mop_artifact:
  - `src/mop/method/` package (contracts, gate, defects, power, report, arms, controls, bed, hypothesis, voi)
  - `src/mop/experiments/base.py` (`REQUIRED`, `validate_declaration`, `bind`, `interpret`)
  - `src/mop/beds/starss23/count_producer.py` + `count_verifier.py` + `count_referee.py` (producer / independent verifier / referee pattern)
  - `src/mop/science/statistics.py` (FORBIDDEN_CLAIM_VERBS, BOUNDED_CLAIM_VERB)
what_ramanujan_would_actually_do: port the validity kernel as Ramanujan's experiment/claim admission layer; reimplement domain checks for math; keep producer modules unable to set independent confirmation
adaptation_cost: strip STARSS23/audio/video assumptions; rebind contracts to Lean-replay and computation-reproduction evidence; map MOP stages to Ramanujan Tribunal stages; most of method/ is domain-agnostic enough to port with renames
evidence: README scientific contract; ARCHITECTURE authority #1 `mop.science` and method package docstring "Experiment validity kernel"; count_verifier refuses producer self-certification of independent_scientific_confirmation.
confidence: high
```

### component: Training arc F0–F9
```
component: Training arc F0–F9 (instrument → retrieval → formalizer/prover SFT → distillation → expert iteration → preference/repair → verifier-guided RL → Director adaptation → curriculum → checkpoint tournament)
verdict: NO_OVERLAP
mop_artifact: NONE for F0–F9. Distant process rhymes only: method stages (`mop.method.gate.SEQUENCE`, `mop.method.power.STAGES`), ladder stage machine (collapsed `stage_ladder.pyc`), campaign orchestration under `src/mop/method/runs/` and `frontier/` (experiment campaigns, not model training arcs).
what_ramanujan_would_actually_do: nothing to import for training; optionally read gate SEQUENCE as a **compute-admission** ordering (cheap checks before expensive work), not as F0–F9
adaptation_cost: full greenfield training pipeline
evidence: No Mathlib retrieval, no formalizer SFT, no Lean-verified distillation, no preference/repair training, no Director LoRA/adaptation, no checkpoint tournament over provers. MOP learning package is bytecode-only (`learning/__pycache__/backprop.pyc`, `alternatives.pyc`).
confidence: high
```

### component: Data discipline
```
component: Data discipline (open/licensed/user-owned, dedup, contamination-check, split-frozen, dispositioned teacher outputs LEAN_VERIFIED | COMPUTATIONALLY_REPRODUCED | HUMAN_REVIEWED | NEGATIVE_EXAMPLE | REJECTED)
verdict: REUSABLE_WITH_ADAPTATION (partial) | CONCEPT_ONLY for teacher disposition enum
mop_artifact:
  - `src/mop/experiments/base.py::REQUIRED` fields: source, split, unit, claim_ceiling
  - `src/mop/beds/starss23/experiments.py` — room-disjoint splits, rights framing
  - `src/mop/beds/starss23/count_verifier.py` — `rights_clean`, source_kind == "real", min reproductions
  - `src/mop/method/bed.py::unit_audit`, `leakage_audit`, `classify`
  - `src/mop/method/contracts.py::DatasetContract`, `IndependentUnitContract`, `Quantity` kinds
  - `src/mop/substrate/cache_manifest.py` — content-addressed cache form declaration (includes form.kind "math" as an enum slot only)
  - Docs: STARSS23 MIT rights narrative in `docs/mixture_of_perspectives/27_escs_starss23_counting_bed.md`
what_ramanujan_would_actually_do:
  - Port split/unit/leakage audits and sealed source identity ideas into Problem/Prior-Art ingestion
  - Port Quantity provenance kinds as a base for teacher-output disposition (not a drop-in enum)
  - Do NOT expect contamination-check or LEAN_VERIFIED disposition machinery — invent it
adaptation_cost: write contamination, license graph, split freezes, and the five teacher dispositions from scratch; adapt bed.unit_audit to problem-set partitions rather than train/tune/test units
evidence: rights_clean and claim ceilings exist; no LEAN_VERIFIED/COMPUTATIONALLY_REPRODUCED/HUMAN_REVIEWED/NEGATIVE_EXAMPLE/REJECTED enum; no dataset contamination scanner in live method/science. Quantity kinds are measured/recomputed/derived/analytic/assumed/structurally_guaranteed — related idea, different ontology.
confidence: high
```

### component: Search
```
component: Search (best-first/beam/tree Lean, proof-state dedup, premise retrieval, compiler-feedback repair, symbolic algebra, SAT/SMT, integer/graph search, counterexample search, program synthesis, evolutionary islands)
verdict: NO_OVERLAP
mop_artifact: NONE live. Collapsed: construction_search (toy subset search); substrate_evo / forge campaigns are architecture-search for sensory substrates, not math search.
what_ramanujan_would_actually_do: nothing
adaptation_cost: full greenfield
evidence: No Lean state hash dedup, no premise retrieval index, no compiler-feedback repair loop, no SAT/SMT bindings, no evolutionary island engines in maintained source. "Search" in MOP means experimental lane orchestration or toy coalition search.
confidence: high
```

### component: Roles (Director, Conjecturer, Skeptic, Formalizer, Prover, Computationalist, Cartographer, Librarian, Economist, Adversary; roles cannot promote their own claims)
```
component: Roles
verdict: CONCEPT_ONLY
mop_artifact:
  - Live role-like separation: producer vs verifier vs referee (count_* modules); VerificationContract roles evidence in `contracts.py`
  - Adjudication veto: `mop.method.defects.adjudicate` — reproduction outranks panel votes
  - Collapsed: `falsification/integrity_scaffold.pyc` strings mention AUTHORITY_ROLES, promotion authority separated from execution authority, independent reviewer distinct from operator
  - No Director/Conjecturer/Skeptic/Formalizer/Prover/... types or role graph
what_ramanujan_would_actually_do: steal the rule "producer cannot self-certify promotion / independent confirmation"; steal "reproduction vetoes consensus"; invent the ten named roles and non-self-promotion policy as new code
adaptation_cost: design role bus + claim ownership from scratch; only the self-promotion ban maps cleanly from count_verifier honesty_ok flags
evidence: count_verifier mismatches if producer sets independent_scientific_confirmation; defects D10 on consensus overriding reproduction; no multi-agent math role system in live code. perspectives/ has empty live sources.
confidence: high
```

### component: Memory (Problem, Claim, Proof-State, Counterexample, Prior-Art, Strategy, Graveyard)
```
component: Memory — seven research stores
verdict: NO_OVERLAP
mop_artifact: NONE for the seven stores. Near-miss vocabulary only:
  - Episodic/replay buffers and GDumb-style continual memory (substrate moldability domain)
  - Content-addressed evidence fabric (`integrated/MOP_EVIDENCE_FABRIC.json`, `integrated/evidence_store/*`) as **artifact content store**, not research memory
  - Method ledger is campaign state (`method/runs/ledger.py`), not Claim/Proof-State stores
what_ramanujan_would_actually_do: nothing to import; optionally use evidence_store hashing pattern for artifact durability
adaptation_cost: design seven stores greenfield
evidence: Substrate memory is lifelong learning state (buffers, consolidation), not mathematical prior-art/proof-state memory. No Graveyard/Strategy/Prior-Art schemas in method contracts.
confidence: high
```

### component: Ledger and Tribunal (tiers 0 Asserted / 1 Empirically Supported / 2 Formalized / 3 Proven; Tribunal admissibility; system never certifies own novelty)
```
component: Ledger and Tribunal
verdict: REUSABLE_WITH_ADAPTATION (governance) | CONCEPT_ONLY (tier-3 Lean replay)
mop_artifact:
  - Claim ceilings: `experiments/base.py` forces claim_ceiling.activation_allowed / scientific_promotion / independent_confirmation all False at declaration
  - Terminal classification: `method/gate.py::classify_result` (method failures before scientific verdicts; provisional_positive; mechanism_null)
  - ClaimContract: claims must bind measured_paths
  - Quantity + ResultContract + wording_check: `method/report.py::wording_check` — prose may not strengthen sealed verdict
  - Forbidden claims lists: e.g. `integrated/MOP_INTEGRATED_FORBIDDEN_CLAIMS.json`; method synthesis forbidden strings
  - Defect ledger + mutations: `method/defects.py::LEDGER`, `method/acceptance.py`
  - Verdict gate (collapsed): positive needs independent verifier receipt (`falsification/verdict_gate.pyc` strings)
  - Science claim verbs: `science/statistics.py` FORBIDDEN_CLAIM_VERBS / BOUNDED_CLAIM_VERB "consistent with"
what_ramanujan_would_actually_do:
  - Port claim ceiling, wording_check, classify_result ordering, ClaimContract, forbidden-claim registries into Tribunal
  - Map MOP classifications onto Ramanujan tiers only by policy rewrite (MOP has no Formalized/Proven Lean tiers)
  - Encode "never certifies own novelty" as extension of "local execution cannot claim independent_scientific_confirmation" (`experiments/base.py::interpret`)
adaptation_cost: implement Tier 0–3 with Lean clean-replay requirements; MOP has empirical mechanics tiers, not formal proof tiers; Tribunal role graph is new
evidence: interpret() refuses independent_scientific_confirmation from local run; count_verifier honesty rules; gate.classify_result checks method validity before scientific positive. No Lean replay, no novelty oracle.
confidence: high
```

### component: Qualification Q0–Q6
```
component: Qualification Q0–Q6 (clone/offline/replay/recovery; formal competence; hidden rediscovery; frontier variants; multi-day research; adversarial governance; human-readiness packet)
verdict: CONCEPT_ONLY (with sparse partial patterns)
mop_artifact:
  - Q0-ish: stop switch + immutable run authorities (README ops; method gate stop_switch); clean-clone questions appear in campaign synthesis JSON (fastforge), not a Q-suite
  - Independent verification / mutation attacks: `method/acceptance.py`, `gate.SEQUENCE` includes independent_reproduction, mutation_attacks
  - Adversarial defect adjudication: `defects.adjudicate`
  - Human-readiness-ish: method coverage_gate, report audit, synthesis wording checks
  - No formal competence exams, no hidden rediscovery of theorems, no multi-day math research rehearsal harness
what_ramanujan_would_actually_do: read gate SEQUENCE and acceptance mutations as a **qualification design template** for adversarial governance tests (part of Q5); build Q0–Q6 suite new
adaptation_cost: almost all new; only governance-adversarial patterns transfer as concepts
evidence: No Q0–Q6 identifiers; no Lean competence battery; studio/campaign recovery is ops recovery for MOP runs, not research rehearsal qualification.
confidence: high
```

---

## 1. The owner's hypothesis, judged

**Hypothesis:** MOP is relevant because it changes how models think, moving away from LLMs toward other inputs.

**Verdict: Reject as the primary transfer story. Accept a weaker, different story.**

| Claim | Support in MOP | Transfers to Ramanujan? |
|-------|----------------|-------------------------|
| Non-LLM sensory substrate (video/audio/continual state) is the reusable core | Live: `TinyVideoSubstrate`, STARSS23 beds, owned substrate v0 (GRU/GDumb/EWC moldability) — and much of it **null** vs strong baselines | **No.** Ramanujan's Director is a frozen LLM; the hard verifier is Lean, not a latent video encoder. |
| "Other inputs" meaning non-prose authority (receipts, sealed metrics, controls) | Strong: scientific contract, contracts kernel, forbidden claim verbs, quantity provenance | **Yes — this is the real overlap.** |
| Changing how models think (representation learning / plasticity) | Domain-specific substrate experiments | **No** for math research system internals |

**What actually transfers:** MOP's **experimental governance and evidence machinery** — the parts README already highlights: declared nulls, SESOI, multiplicity, budget, stop rule, claim ceiling, independent verifier, producer/verifier separation, nulls as valid outputs.

**What does not transfer:** the substrate research program (V-JEPA caches, STARSS23 spatial audio, EMNIST/HAR moldability, forge/frontier generation campaigns about compression/routing/plasticity).

Plain statement: **If Ramanujan forked MOP hoping for a non-LLM mathematical substrate, it would inherit the wrong machine.** If it forked (or ported) `mop.method` + declaration/verification patterns, it would inherit something rare and load-bearing for expert iteration and RL: machinery that makes self-fooling expensive.

---

## 2. Three highest-value transfers (ranked)

### 1) Experiment validity kernel + admission gate (`src/mop/method/`)
**Why worth the cost:** Expert iteration and verifier-guided RL default to self-deception (metric gaming, unpowered nulls sold as failures of the method, prose that upgrades a null). MOP encodes a pre-principal stage sequence (`gate.SEQUENCE`), contracts that fail closed without evidence, mutation-tested historical defects (`defects.LEDGER` + `acceptance.py`), and terminal classification that refuses to call a method failure a mechanism null.

**What to actually do:** Port modules with light rename: `contracts`, `gate`, `defects`, `power`, `report.wording_check`, `arms.distinctness`, `hypothesis` graph, `voi` queue. Rebind domain predicates to math beds (Lean suites, problem splits).

**Adaptation cost:** Medium — mostly domain rebinding and dropping torch-specific control proofs; structure is already abstract.

### 2) Producer / referee / independent verifier split + claim ceiling (`beds/starss23/count_*` pattern + `experiments/base.py`)
**Why worth the cost:** Ramanujan's "roles cannot promote their own claims" and "system never certifies its own novelty" are policy until code enforces them. MOP already hard-codes:
- declaration-time claim_ceiling all false (`experiments/base.py::validate_declaration`)
- local interpret refuses self independent confirmation
- count_verifier re-scores from sealed tracks and refuses producer honesty violations

**What to actually do:** Copy the three-surface pattern (producer writes sealed artifact → referee pure scorer → verifier reimplements scoring without importing producer math) onto (proposal → Lean replay → Tribunal). Keep seal + canonical hash habits from `mop.evidence`.

**Adaptation cost:** Medium-high for math artifacts; pattern is clear, scoring domain is new.

### 3) Power / SESOI / stop rules / forbidden claim language (`method/power.py`, `science/statistics.py`, experiment records)
**Why worth the cost:** F5–F7 style loops (preference, repair, RL) will generate torrents of "almost" results. MOP's decision rule is brutal and useful: **tie is null; wrong direction is failure; seeds may not be added after the fact; positive needs lower confidence bound ≥ SESOI; claim verb locked to "consistent with"** unless stronger evidence exists.

**What to actually do:** Port `power.preregistration` / `power.decide` and claim-verb rails into Ramanujan's training and evaluation promotion paths. Replace MAE-flavored SESOI with math success metrics (proof rate, repair rate) but keep the **shape**.

**Adaptation cost:** Low-medium for the decision API; high for choosing honest math SESOIs and independent units (problem families, not seeds alone).

---

## 3. What MOP has that Ramanujan's spec does not ask for — but should

These are not vocabulary matches; they are **working anti-self-fooling devices** for systems that iterate models against verifiers.

| MOP device | Artifact | Why Ramanujan needs it |
|------------|----------|------------------------|
| **SESOI + powered nulls** | `method/power.py`, `PowerContract`, science `sesoi_exceeded` | Without a smallest effect of interest, RL "improvements" and expert-iteration bumps are uninterpretable. Unpowered nulls will be misread as "prover is bad" vs "design cannot see the effect." |
| **Multiplicity policy** | experiment `multiplicity` field; familywise rules in STARSS records | F0–F9 and checkpoint tournaments create many comparisons. Uncontrolled multiplicity manufactures false positives. |
| **Stop rules & sealed continuation** | `stop` in declarations; power continuation_rule; sign-flip stop | Prevents "one more seed / one more RL run" after peeking — the classic expert-iteration failure mode. |
| **Claim ceilings** | `claim_ceiling` forced false on activation/promotion/independent confirmation | Spec says Tribunal and tiers; MOP shows **default hard ceiling** until external confirmation. Prevents the system shipping its own press release. |
| **Producer / verifier structural separation** | ARCHITECTURE + count_verifier not importing producer; `interpret` refusal | Spec says roles cannot self-promote; MOP makes separation **import-structural**, not honor-system. |
| **Null and negative as first-class outputs** | README; `classify_result` mechanism_null; registry of historical nulls | Training arcs that discard nulls retrain on survivorship. Graveyard should receive null strategies, not only failed proofs. |
| **Quantity provenance kinds** | `contracts.Quantity` / QUANTITY_KINDS | Spec dispositions teacher text; MOP also tags **numbers** as measured vs structurally_guaranteed. Stops "zero forgetting by construction" style fraud in metrics dashboards. |
| **Wording / prose rail** | `report.wording_check` | Spec says persuasive prose is not authority. MOP implements machine check that prose cannot upgrade sealed verdicts — critical for Director summaries and human-readiness packets (Q6). |
| **Defect ledger with mutation reinjection** | `defects.LEDGER`, `acceptance.py` | Every past self-fool becomes a permanent regression test. Ramanujan will accumulate governance bugs; without a ledger they recur every phase. |
| **Reproduction veto over consensus** | `defects.adjudicate` | Adversary/Skeptic panels must not vote away a reproduced Lean/tool defect. |
| **Pre-principal cheap gates** | `gate.PRE_PRINCIPAL` before expensive compute | For RL and search budgets: refuse runs that lack instrument/bed/power before burning GPU/Lean hours. |
| **Arm distinctness by execution traces** | `method/arms.py` | Prevents "Prover-A vs Prover-B" that share implementation — deadly in ablation and tournament reporting. |
| **Value-of-information queue refusing closed premises** | `method/voi.py` | Stops re-running research directions already null-closed — maps to Strategy/Graveyard discipline. |

---

## 4. What must NOT be carried over

| Do not carry | Why |
|--------------|-----|
| **Video/audio substrate stack** (`substrate/custom_model.py`, V-JEPA configs, STARSS23 FOA features, campaign2 vision beds) | Wrong problem; would distort Ramanujan architecture toward latent sensing. |
| **Owned substrate v0 conclusions as capability** (`substrate/MOP_OWNED_SUBSTRATE_V0_SPEC.md`: no candidate selected, null vs GRU+GDumb) | Domain nulls are not math priors; importing them as "memory architecture truth" would be cargo-cult. |
| **`science/gating.py` causal gate traces** | Audio/online gate inference over frames — not proof search. |
| **FLOP arm models tuned to featurize/gate/re-estimate** (`science/budget.py` STARSS-shaped arms) | Budget idea is good; the FLOP model constants and arm kinds are not. |
| **Collapsed packages as dependencies** (`falsification`, `mechanisms`, `ladder`, `perspectives`, `learning` without `.py`) | Not maintained import surface; recovery via collapse index is archaeology. |
| **Studio generation1 successor chain explosion** (`studio/generation1_successor_*.py` names) | Ops scaffolding for MOP campaigns; high complexity, low math relevance. |
| **Telegram / worker throttle / encode scheduler** | Host ops, not research logic. |
| **Forbidden claims that are MOP-specific** (e.g. "bounded compression recovers the full-memory gap") | Domain noise; rewrite forbidden list for math (novelty self-cert, sorryful proofs, undeclared axioms). |
| **Claim that MOP "verifier" is a math verifier** | MOP verifiers recompute sealed experiment artifacts; they are not Lean. |
| **Activation / Stage 3 ladder rhetoric** | MOP still fail-closed on scientific activation; do not import stage indices as Ramanujan readiness. |
| **Any assumption that nulls on sensory beds transfer to "LLMs are the wrong substrate for math"** | MOP did not test that hypothesis; Ramanujan presupposes a frozen LLM Director. |

---

## Bottom line

| Question | Answer |
|----------|--------|
| Does MOP contain a Ramanujan-ready math research stack? | **No.** |
| Does MOP contain reusable **governance** for a verifier-grounded research system? | **Yes — substantial, battle-tested, and under-specified in Ramanujan's component list.** |
| Owner hypothesis (substrate / non-LLM inputs)? | **Does not transfer.** The valuable overlap is **evidence and experimental discipline**, which README already states more accurately than the substrate narrative. |
| If forced to fork one tree? | Fork or port **`src/mop/method` + `experiments/base.py` + evidence sealing + the starss producer/verifier pattern as documentation**, not `substrate/` or `forge/`. |

**Honest summary sentence for the next phase:**  
*MOP will not give Ramanujan Lean, search, roles, or training arcs; it can give Ramanujan a rare, already-debugged refusal system that stops the research loop from promoting its own stories.*
