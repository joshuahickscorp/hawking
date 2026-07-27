# Ramanujan cognition — independent frontier ideation

**Author role:** Grok (independent pass; no coordination with any parallel agent)  
**Date:** 2026-07-26  
**Nature of document:** Design ideation only. No build plan, no repo changes, no experiments run.  
**Constraint honored:** one local machine, offline, frozen giant Director, small trainable models; every mechanism must in principle be checkable by Lean or by computation.

---

## 0. Verdict on the premise

The specified stack (frozen Director + small specialists + Lean Tier-3 + multi-role Tribunal + seven stores + proof search + external solvers) is the correct **substrate for certainty**. It is not a model of **mathematical cognition**.

That is not a romantic complaint. Cognition here means: the processes by which a mind (human or artificial) forms objects, notices structure, feels blocked, invents vocabulary, and decides which true statements are worth the cost of proving. The consensus 2025–2026 neural theorem-proving architecture optimizes a different functional: *given a formal goal, allocate search until a certificate exists or resources die*. That is necessary. It is not sufficient for research intelligence, and it is not what the historical Ramanujan was doing for most of the hours that produced his notebooks.

**What the stack is structurally unable to do**, stated without technology:

1. **Form new conceptual units that reorganize the space of proofs.** It can recombine pretrained tokens and retrieve existing lemmas; it cannot grow a living definition library whose purpose is to make later proofs short.
2. **Perceive mathematical objects as multi-channel phenomena** rather than as strings that parse into Lean syntax trees. Mathematicians “see” growth rates, symmetries, counterexample shapes, and family resemblances before they have a formal statement.
3. **Treat failure as positive structure.** A Graveyard that stores dead attempts is a log. Mathematicians extract *obstructions* — invariants, missing hypotheses, reduction barriers — and reason with those as objects.
4. **Allocate effort by significance rather than provability.** Best-first proof search ranks *ease of closing a goal*. Taste ranks *which goals, if closed, restructure the map*. Those objective functions diverge systematically.
5. **Fuse two epistemologies with different failure modes.** Lean gives local certainty over formalized claims. Computation gives global, noisy evidence over vast instance spaces. Research lives in the negotiation between them, not in either channel alone.
6. **Generate from catalogue-and-pattern rather than from proof-state expansion.** Ramanujan’s generative mode was closer to “this series wants this closed form” than to “expand tactic tree under goal G.”

If someone defends the consensus stack as complete cognition, they are committing a category error: they are identifying the *verifier loop* with the *discovery loop*. Hardy was not Ramanujan. The system needs both, and the current architecture is almost all Hardy.

The rest of this document proposes mechanisms that attack those structural gaps. They are ranked by expected research value per unit of local implementation cost. Each is specified at the level of data, representation, training signal, I/O, Lean interaction, failure mode, and falsification.

---

## 1. What mathematical cognition is (phenomenon first)

Strip the tools. Watch what happens when a working mathematician discovers something.

### 1.1 Compression under pressure

A discovery is almost always a **compression event**: many previously separate facts become consequences of a shorter object (a definition, an identity, a correspondence). The feeling of “elegance” is largely the felt reduction in description length of a fragment of the mathematical world. Proof is how compression is certified; it is not how compression is first noticed.

Implication for architecture: a system that only maximizes “probability of closed proof for the current goal” will happily prove long, local, uninteresting truths and miss short global reorganizations.

### 1.2 Multiple simultaneous presentations

The same object is held as formula, as geometry, as generating function, as dynamical system, as finite check, as asymptotic. Insight is often the sudden availability of a second presentation that makes an operation trivial which was opaque in the first. This is not “multi-agent debate.” It is **representational multiplicity with translation maps**.

Implication: an architecture whose primary working memory is a single proof state in one formal system is half-blind. Lean is one presentation — an extremely valuable one — not the object itself.

### 1.3 Working examples as perception, not as decoration

Before the general theorem, there is a cloud of computed cases, drawings, small-n tables, random instances, and failed generalizations. The cloud is not “inspiration for the LLM.” It is a **sensorium**. Patterns in the cloud are the first form of the claim. Formalization is delayed, deliberately, until the pattern stabilizes.

Ramanujan’s notebooks are extreme versions of this: identity as perceived regularity across numerical and modular phenomena, with proof deferred or absent. Naming the system after him while building only a prover-loop is an irony that should be treated as a design constraint, not a brand.

### 1.4 Obstruction as object

Experienced mathematicians spend large fractions of time on **why a route cannot work**: degree reasons, parity, functoriality failures, dimension counts, known hard cores, missing regularity. These are not merely negative search results. They are portable objects — “any proof of X must either strengthen H or invent a way around O.”

A log of failed tactics is not an obstruction. An obstruction is a reusable certificate that prunes a region of strategy space and sometimes becomes a theorem of independent interest (impossibility, lower bound, counterexample family).

### 1.5 Significance is relational, not intrinsic

A statement is significant relative to a body of problems, methods, and open claims: it connects clusters, unlocks a technique, collapses a case division, or names a phenomenon people were already bumping into. Significance is closer to **graph centrality and generative power** than to truth or to proof length.

### 1.6 Vocabulary is grown, not only used

The deepest moves invent definitions that make theorems easy. The concept is a tool, not a label. A frozen Director has a fixed token vocabulary and a fixed cloud of pretrained associations; that is a real ceiling on *conceptual* novelty even if proof search scales. The ceiling dissolves only if **definition-forming** is a first-class, trainable, verifier-grounded process living outside the frozen weights.

### 1.7 What this is not

It is not mystical intuition. It is not “let the LLM free-associate harder.” It is a claim that mathematical minds run **several coupled processes** — sensing structure, inventing vocabulary, extracting barriers, ranking worth, certifying compression — of which formal proof search is one, late, expensive process.

---

## 2. Structural diagnosis of the consensus stack

| Cognitive function | Human research | Consensus stack | Structural gap |
|---|---|---|---|
| Object perception | Multi-channel (numeric, geometric, formal, combinatorial) | Tokens + proof states + maybe embeddings of text | No non-linguistic object model |
| Concept growth | New definitions reorganize proofs | Frozen Director vocabulary; lemmas retrieved not invented as conceptual units | Conceptual ceiling |
| Failure | Obstructions as portable theory | Graveyard as storage | No positive theory of failure |
| Taste | Significance relative to a map | Provability / search heuristics | Wrong objective |
| Evidence | Dual: proof + experiment | Lean as truth; compute as side tool | No fusion epistemology |
| Generation | Catalogue, analogy, identity-from-pattern | Goal-directed proof expansion | Generative mode missing |
| Roles | Different skills and incentives | Prompted personas that cannot self-promote | Roles without distinct representations or losses |

The roles (Director, Conjecturer, Skeptic, …) as specified are **governance**, not cognition. Governance is good — claims must not self-promote — but renaming a policy head “Skeptic” does not create obstruction perception. Different roles need different **inputs, representations, and training signals**, not only different system prompts.

---

## 3. Mechanisms

### Ranking summary (expected value / local cost)

| Rank | Mechanism | EV/cost | One-line |
|---|---|---|---|
| 1 | **Definitional Library Learning (DLL)** | Highest | Grow Lean definitions that minimize future formalization cost — dissolves frozen-vocabulary ceiling without unfreezing the giant |
| 2 | **Obstruction Objects (OO)** | Very high | Promote structured failure into first-class, reusable mathematical content with a training signal |
| 3 | **Multi-Sense Object Fingerprints (MSOF)** | High | Implement “perception” of mathematical objects as fused non-linguistic embeddings |
| 4 | **Significance as Map Dynamics (SMD)** | High but slippery | Mechanical taste: predicted compression, connectivity, and generativity on the claim graph |
| 5 | **Dual-Channel Epistemic Fusion (DCEF)** | Medium-high | Calibrated negotiation between Lean certainty and computational evidence |
| 6 | **Analogy as Partial Structure Maps (APSM)** | Medium | Explicit morphism search for strategy transfer, not text similarity RAG |

---

### Mechanism 1 — Definitional Library Learning (DLL)

**Capability the stack lacks:** forming new conceptual units that reorganize proof space, without unfreezing the Director.

**Phenomenon:** Mathematicians invent definitions so that theorems become short. The definition is chosen under a pressure: *make this family of facts easy*. That is library learning under a formal cost metric, not poetry.

**Concrete mechanism:**

- **Representation.** A growing `RamanujanLib` of Lean `def` / `structure` / `class` items, each with: name, type, body, dependency footprint, and a **utility vector** (which claims it helped, proof-token savings, reuse count).
- **Consumes.** Bundles of related open or recently closed claims; proof traces; existing mathlib + local library; failed proof attempts that share structure.
- **Emits.** Candidate definitions (Lean code), plus a **rewrite plan**: how existing statements should be restated through the new concept; optional automatic refactors of in-progress proof sketches.
- **Search procedure (small models only).**
  1. Mine recurring subterms / subgoals across a claim cluster (anti-unification / common subexpression / lemma-extraction from successful proofs).
  2. Propose `def` candidates that abstract the common pattern (small formalizer / synthesizer; beam over abstraction depth).
  3. Score each candidate by **predicted net description length**:  
     `ΔL = L(proofs | old lib) − L(proofs | old lib + def) − λ · L(def)`  
     estimated by short formalization attempts on a holdout slice of the cluster, not by LLM vibes.
  4. Admit definition only if a **Tribunal rule** is met: at least k claims in the cluster have shorter closed proofs (or first-time closes) under a fixed tactic budget when the def is available, and no new undeclared axioms.
- **Training signal.**  
  - Positive: definitions that reduce total proof tokens / search nodes on held-out claims in the same family.  
  - Negative: definitions that only help the training goals (overfit abstractions), increase proof length, or are never reused.  
  - Offline proxy before any research use: replay mathlib history — given library at commit t, recover definitions introduced later that would have compressed proofs of theorems formalized in (t, t+Δ].
- **Interaction with Lean.** Definitions are real Lean declarations. Utility is measured by `lake build` / proof replay under resource caps. Director never needs new weights: it is shown the **interface** (docstring + type) of new defs as retrieved context, like any other lemma.
- **How this dissolves the frozen ceiling.** Novelty lives in the **library state**, not in Director weights. The giant’s conceptual vocabulary for *talking* stays fixed; the system’s conceptual vocabulary for *proving and stating* grows. That is the non-cheating move: Hardy’s language was English; Ramanujan’s productive novelty was in the identities and objects, not in a larger English model.

**Failure mode:**  
- **Encoding tricks** — definitions that shorten proofs by smuggling a near-proof into a `def` (Goodhart on proof length). Mitigate by counting the def body toward cost and forbidding defs whose body contains heavy proof terms without marking them as theorems.  
- **Definition spam** — thousands of one-off abbreviations. Mitigate by reuse tax and periodic library consolidation (merge/delete low-utility defs; must preserve Tier-3 replay).  
- **Opaque conceptual debt** — names that compress proofs but destroy human (and Director) readability. Mitigate by requiring a small “explainer” model to produce usage examples that typecheck, and by penalizing defs that do not improve *search* (premise retrieval hit rate), only post-hoc proof length.

**Falsification:**  
On a fixed suite of formalization targets (e.g. a held-out slice of mathlib or a curated contest/research problem set), enable DLL with a fixed compute budget. **Abandon if** total successful formalizations and total proof tokens are not better than (a) lemma-mining without new definitions and (b) larger premise retrieval alone, at p-level agreed in advance. Secondary abandon signal: >50% of admitted defs have reuse count ≤ 1 after N subsequent problems.

**Novelty honesty:**  
**Partial rename / extension of known work.** Closest ancestors: DreamCoder-style library learning; lemma synthesis and “useful lemma” mining in ITP; proof compression via abstraction; macro/definition extraction in formal libraries. What is relatively under-emphasized in the NTP consensus stack is treating **definition invention as the primary escape hatch from a frozen Director**, with an explicit ΔL admission rule and Tribunal integration. Not a virgin idea — a missing first-class module.

---

### Mechanism 2 — Obstruction Objects (OO)

**Capability the stack lacks:** perceiving failure as structured, portable mathematical content rather than as a Graveyard log.

**Phenomenon:** “This proof cannot go that way because of an invariant / a missing hypothesis / a hard core / a counterexample to a needed lemma.” Experts navigate by barriers. Novices retry the same route.

**Concrete mechanism:**

- **Representation.** An `Obstruction` record, not free text:
  ```
  Obstruction := {
    id,
    kind: InvariantMismatch | CounterexampleToLemma | ResourceExhaustionWithCore
         | MissingHypothesis | ReductionToKnownHard | TypeclassGap | OtherCertified,
    payload: Lean term | computational certificate | both,
    scope: which claim families / strategies it blocks,
    dodge: hypothesized strengthenings or alternative routes (possibly empty),
    provenance: proof-state path + solver logs that produced it,
    confidence: formal | empirical | mixed
  }
  ```
- **Consumes.** Failed proof branches, compiler errors, counterexamples, SAT unsat cores, timeout profiles, Skeptic challenges.
- **Emits.** Obstruction objects into a store that is **queried as premises** by Prover and Strategist: “given goal G and strategy S, relevant O’s.” Also emits **avoidance constraints** into search: prune or deprioritize paths matching O.scope unless dodge conditions are met.
- **Extraction pipeline (small models + solvers):**
  1. **Syntactic/compiler layer:** type errors, failed rewrites, missing instances → typed obstruction stubs.  
  2. **Counterexample layer:** hypothesis-weakening + fuzzing / QuickCheck-style / SMT → `CounterexampleToLemma` with replayable seed.  
  3. **Invariant layer:** small specialized predictors propose conserved quantities (parity, degree, modular weight, rank) and attempt to **formalize the mismatch** as a short Lean lemma: “if P then contradiction with inv.” If formalized, confidence := formal.  
  4. **Core layer:** from exhausted search, extract minimal unsatisfiable set of proof obligations (analogous to UNSAT cores) as `ResourceExhaustionWithCore` — empirical unless later formalized.
  5. **Librarian link:** map O to Prior-Art (known impossibilities, barrier theorems).
- **Training signal.**  
  - Predict which obstruction class applies to a new (goal, strategy) pair; loss against later observed failures.  
  - Reward obstructions that, when used as search constraints, **reduce nodes-to-solution** or **increase correct early aborts** on problems with known outcomes.  
  - Penalize obstructions that block a path that later succeeds (false barriers).
- **Interaction with Lean.** Best obstructions *are* Lean lemmas (`by contradiction`, invariant theorems, formalized counterexample witnesses). Empirical obstructions remain labeled non-Tier-3 and can only affect search priority and conjecture shaping, never theorem admission.

**What “perceiving an obstruction” means operationally:**  
Not storing a failure string. It means: (i) classifying the failure into a typed schema, (ii) attaching a replayable certificate when possible, (iii) binding a scope and a dodge, (iv) making that object retrievable and usable as a first-class input to planning. Perception = **structured transduction from raw failure into a reusable barrier object**.

**Failure mode:**  
- **False barriers** prune the only viable path (especially from timeouts misread as impossibility).  
- **Verbal pseudo-obstructions** — Skeptic prose that feels deep and blocks progress. Ban untyped prose from the Obstruction store; require schema + certificate kind.  
- **Overfitting to local compiler noise** rather than mathematical content.

**Falsification:**  
A/B on proof search with identical budgets: baseline Graveyard replay vs OO-constrained search. **Abandon if** OO does not improve solve rate or median nodes-to-proof on a mixed suite, or if false-barrier rate exceeds a threshold (e.g. blocks >5% of later-successful baseline paths). Independent check: human mathematicians should rate a sample of formalized O’s as “real mathematical content” above a baseline of random failure summaries — if not, the schema is wrong even if search metrics twitch.

**Novelty honesty:**  
**Strong partial rename of mature ideas, re-aimed.** CDCL conflict clauses, lemma learning in ATP, failure-driven learning, dependency analyses, and “barrier” discourse in complexity theory all instantiate pieces. The gap in the consensus *research* stack is elevating these into a **uniform, typed, Tribunal-visible obstruction ontology** that spans Lean failures and computational certificates, trained end-to-end for search and for conjecture repair — not merely logging. Conceptually not virgin; architecturally still missing as a peer of Claim/Proof-State.

---

### Mechanism 3 — Multi-Sense Object Fingerprints (MSOF)

**Capability the stack lacks:** perception of mathematical objects as something other than text and proof-state trees.

**Phenomenon:** “I see what this is” — before naming it. Recognition of family resemblance across presentations. Ramanujan-level catalogue sense is extreme multi-sense pattern memory over series, products, modular behavior, and closed forms.

**What “perception” means, implementably:**  
A mathematical object (expression, structure, claim, variety, graph family, …) is mapped to a **bundle of channels**, each a lossy sensor with known failure modes; a small encoder fuses them into a fingerprint used for retrieval, analogy, and conjecture seeding. Perception is not mystical access; it is **stable multi-channel encoding under presentation changes**.

**Concrete mechanism:**

- **Channels (senses), each producing a fixed-size vector or structured tensor:**
  1. **Formal shape:** AST / type / dependency hash of the Lean term (GNN or tree encoder).  
  2. **Numeric sample:** evaluations on random inputs, special points, modular samples; moment / histogram features; interval bounds where relevant.  
  3. **Symbolic normal form:** CAS rewrite to canonical-ish form (with timeout); rewrite-graph summary.  
  4. **Combinatorial profile:** for discrete objects — degree sequences, spectrum of adjacency, partition stats, automorphism group size estimates.  
  5. **Dynamic / iterative:** for sequences and recurrences — first N terms, generating-function padé hints, singularity sketches (cheap).  
  6. **Proof-role footprint:** how the object appears in completed proofs (hypothesis, main lemma, witness) — empty for new objects.
- **Fusion.** Small contrastive encoder: same object under different presentations (α-renaming, equivalent defs, reordered sums, different Lean encodings) → attract; known-related (mathlib instances of same class, OEIS-like families) → soft attract; random → repel.
- **Consumes.** Lean terms, CAS strings, code evaluators, optional drawings only if reduced to numeric descriptors (no vision dependency required).
- **Emits.** `Fingerprint(id)`; nearest neighbors; channel-wise novelty scores (“numeric says familiar, formal shape says alien” → high interest).
- **Training signal.**  
  - Contrastive loss on presentation variants.  
  - Supervised: predict mathlib cluster / topic tags from fingerprint alone.  
  - Downstream: improve premise selection and conjecture pairing vs text-embedding baseline.
- **Interaction with Lean.** Fingerprints never admit theorems. They only (a) retrieve premises and prior art, (b) propose “these two claims may be the same under reindexing,” (c) seed Conjecturer with neighbor-identities to test. Any proposed identity must still pass computational checks and/or Lean.

**Failure mode:**  
- **Spurious numeric coincidences** (famous near-identities, floating-point traps). Channel fusion must not let numeric similarity dominate formal type disagreement without a flag.  
- **Sense poverty** in domains hard to sample (higher infinity, pure existence). Fingerprints degrade to formal-only — must detect and abstain.  
- **Catalogue collapse** — everything embeds near everything in a narrow subdomain.

**Falsification:**  
Retrieval benchmark: given a goal, does MSOF premise retrieval beat (i) BM25 on formal text, (ii) LLM embedding of statement text, (iii) type-only matching, on proof success under fixed search budget? **Abandon if** no significant gain on at least two of three domain suites (algebra, number theory, combinatorics). Secondary: presentation-invariance test — α-rename and equivalent rewrite should preserve nearest neighbors; if not, the encoder is broken.

**Novelty honesty:**  
**Mostly recombination of existing pieces.** Contrastive encoders, GNN-on-AST, experimental mathematics sampling, OEIS-style signature search, and premise selection models all exist. The design claim is that treating these as **mandatory sensory channels of a research agent**, fused and Tribunal-constrained, is different from bolting “better RAG” onto an LLM. Scientifically incremental; architecturally decisive if the system is serious about non-LLM inputs.

---

### Mechanism 4 — Significance as Map Dynamics (SMD)

**Capability the stack lacks:** a mechanical account of mathematical taste — which *true* statements matter.

**Phenomenon:** Taste is not ineffable beauty. Practitioners prefer statements that connect fields, simplify towers of lemmas, open construction pipelines, or name a recurring obstruction. Truth is cheap relative to attention; significance is an economic quantity on a map of claims and tools.

**Concrete mechanism:**

- **Working map.** Graph G whose nodes are: Claims, Definitions, Methods/Strategies, OpenProblems, Obstructions. Edges: `depends_on`, `generalizes`, `counters`, `enabled_by`, `analogous_fingerprint`, `proved_with`.
- **Significance score** of a candidate claim C (before or after proof), computed by a small value model:

  `S(C) = w1·ΔConnectivity(C) + w2·ΔCompression(C) + w3·Generativity(C) + w4·ObstructionYield(C) − w5·Cost(C)`

  where:
  - **ΔConnectivity:** predicted increase in reachability among open claims if C is admitted at Tier-3 (or as strong conjecture). Proxy offline: mathlib import/dependency centrality of comparable lemmas.  
  - **ΔCompression:** predicted reduction in total description length of a neighborhood of Prior-Art if C is added (MDL-style; can be estimated by whether many existing lemmas become corollaries under a short tactic budget).  
  - **Generativity:** predicted number of new well-typed constructions / instances enabled.  
  - **ObstructionYield:** does proving or refuting C produce high-value OO’s either way? (two-sided information value).  
  - **Cost:** expected search cost from historical analogs.
- **Consumes.** Candidate statements from Conjecturer; graph G; historical outcomes; MSOF clusters.
- **Emits.** Priority ranking for formalization effort; “pursue / defer / kill” recommendations to Economist; never auto-admits truth.
- **Training signal (local, offline-honest):**  
  - Reconstruct mathlib / Archive of Formal Proofs growth: at time t, which added lemmas become high in-degree over the next Δ commits? Train S to rank those above forgotten lemmas.  
  - On contest corpora, weak proxy only (contests ≠ research taste) — use for calibration of Cost term, not of Connectivity.  
  - Online: after admission, track actual reuse; update S with delayed reward.
- **Interaction with Lean.** S is a scheduler over the Ledger, not a truth source. A high-S false conjecture should be cheaply killed by counterexample search; a low-S true lemma may still be proved if it is on the critical path of a high-S goal.

**Failure mode:**  
- **Fashion capture** — S learns “what mathlib already centralizes,” systematically undervaluing Ramanujan-like outsider compressions that are not yet connected. Mitigate with an **exploration bonus** for high ΔCompression in sparse fingerprint regions (novelty of location on the map).  
- **Goodhart** — optimizing for citation-like centrality produces trendy but shallow targets. Keep Compression and ObstructionYield as first-class terms.  
- **Cold start** — empty graph → S meaningless; need seed Prior-Art graph from mathlib.

**Falsification:**  
Prospective test: freeze S trained on history up to t; rank unproven or newly stated candidates; measure which ranking best predicts (i) actual later reuse after human or system formalization, (ii) human expert priority labels on a blind set. **Abandon if** S does not beat simple baselines: statement length, presence of buzzwords, random, pure proof-ease estimate. If S only predicts ease of proof, it has collapsed into the existing value head and should be deleted.

**Novelty honesty:**  
**Old idea, modern implements.** Lenat’s AM/EURISKO interestingness heuristics; MDL theories of theory choice; scientometric centrality; “curiosity” bonuses in RL. What would be new enough to matter is a **verifier-grounded, graph-native S** trained on formal library dynamics rather than LLM aesthetic preference. Without the training signal and falsification above, “taste model” is just a rename of a value head.

---

### Mechanism 5 — Dual-Channel Epistemic Fusion (DCEF)

**Capability the stack lacks:** a principled fusion of Lean’s certainty and computation’s mass evidence — the actual epistemology of modern research mathematics (and of Ramanujan–Hardy collaboration).

**Phenomenon:** Experimental mathematics produces identities trusted long before proof; formal proof sometimes certifies barren statements. Conversely, beautiful formal proofs can sit on definitions that miss the phenomenon computation revealed. The fusion is not “average the confidences.” It is **typed evidence with different promotion rights**.

**Concrete mechanism:**

- **Evidence types on every Ledger claim:**
  - `E_formal`: Tiered Lean evidence (as specified — Tier-3 = clean replay, no sorry, no smuggled axioms).  
  - `E_comp`: computational package — instance counts, bit precision, property-based tests, SAT/SMT results, interval proofs, Monte-Carlo bounds — each with **protocol hash** for replay offline.
- **Fusion model (small, calibrated):**  
  Inputs: features of E_comp + partial formalization status + MSOF stability across samples.  
  Outputs:  
  - `P(exists Tier-3 proof | evidence)`  
  - `P(exists counterexample within budget b | evidence)`  
  - recommended **next action**: more compute / attempt formal / invent def / seek obstruction / abandon.
- **Tribunal rules (non-learned, hard):**  
  - No `E_comp` alone can mint a theorem.  
  - `E_formal` Tier-3 mints theorem regardless of `E_comp` (compute may still mark “phenomenon mismatch” as a *definitional* warning — different claim).  
  - High `P(counterexample)` blocks formalization spend above a threshold unless Adversary has already searched.  
  - Identities with extreme `E_comp` and high MSOF stability get **Conjecture** status with explicit bounds (“holds for n ≤ N at precision p”) — a first-class Ledger object, not a failed theorem.
- **Training signal.**  
  - Historical formalization outcomes of experimentally discovered identities (where available).  
  - Synthetic: plant near-identities that fail at large n; plant true identities with cheap formal special cases; train calibration (ECE / reliability diagrams).  
  - Reward policies that minimize total compute+search to correct Ledger state (true proved, false refuted, unknown deferred).
- **What is built at the fusion:**  
  Not a mystical third truth value. Built are: (1) **claim typing** (theorem / bounded conjecture / refuted / open), (2) **resource allocation**, (3) **definitional alerts** when formal statement and computational phenomenon diverge (often the real discovery: wrong formalization of the right idea).
- **Interaction with Lean.** Compute can generate candidates and kill falsehoods; Lean owns necessity. Interval arithmetic or fully checked computation can sometimes *be* Tier-3 if formalized (e.g. formal interval tactics) — then channels collapse honestly into formal.

**Failure mode:**  
- **Numerology** — promoting high-precision agreement to near-certainty (classic experimental math failure). Hard Tribunal walls prevent this; watch for soft pressure to weaken walls.  
- **Compute neglect** — fusion model always says “just prove,” wasting the empirical channel.  
- **Protocol non-replay** — computational evidence that cannot be offline-reproduced is poison; discard.

**Falsification:**  
Calibration plots on held-out identity/conjecture corpora: if `P(formalizable | ·)` is badly miscalibrated (ECE above threshold), retrain or kill. Policy evaluation: dual-channel action policy vs pure-formal and pure-compute baselines on time-to-correct-Ledger-state. **Abandon if** dual policy loses to pure-formal on theorem yield *and* fails to refute false conjectures faster than pure-compute. If the only win is prettier dashboards, abandon.

**Novelty honesty:**  
**Established scientific practice, under-systematized in NTP agents.** Bailey–Borwein experimental mathematics; formal-CAS bridges; probabilistic checks in proof assistants; auto-active ITP with external provers. The novelty is making **calibrated dual epistemology** a core architectural loop with promotion rules, not an optional toolkit call from the Director.

---

### Mechanism 6 — Analogy as Partial Structure Maps (APSM)

**Capability the stack lacks:** analogy as structure-preserving map, not as embedding similarity of statement text.

**Phenomenon:** “This is like the proof that…” means: parts correspond, relations correspond, and a method transports with repair. Gentner’s structure-mapping is the right cognitive description; RAG is a degraded proxy.

**Concrete mechanism:**

- **Representation.** For a problem or theory fragment, extract a **relational schema**: sorts, operations, distinguished maps, commuting squares, exactness-like patterns, order relations — from Lean types + a small relational extractor.  
- **Search.** Find partial morphisms φ: schema_source ⇀ schema_target maximizing preserved relation instances under type compatibility (constraint solver / GNN + discrete refinement).  
- **Transfer.** Pull strategies, lemma templates, and obstruction objects along φ; instantiate; hand to repair/prover.  
- **Consumes.** Solved problems in Prior-Art with stored schemas; new goal schema; MSOF neighbors as candidates to attempt morphism search (cheap filter).  
- **Emits.** Mapped proof sketches, mapped obstructions, confidence = fraction of relations preserved + repair success rate.  
- **Training signal.** Success of transfer+repair under fixed budget; supervised pairs from mathlib where the same tactic pattern solves structurally similar lemmas.
- **Interaction with Lean.** Transferred sketches are ordinary proof terms/scripts subject to the same Tier-3 rules. Failed transfers should preferably yield OO’s (“morphism broke at exactness of …”).

**Failure mode:**  
- **Forced category-theory cosplay** — everything mapped into a fashionable schema. Restrict schema language to what extractors reliably produce; allow domain-specific schema dialects.  
- **Cost blowup** — morphism search more expensive than proving from scratch. Hard cap; APSM is advisory.  
- **Spurious morphisms** that increase repair burden.

**Falsification:**  
Cross-domain and intra-domain transfer benches: APSM-guided strategy selection vs dense retrieval of proof scripts vs random. **Abandon if** no gain in solve rate or proof length after accounting for morphism-search time. If gains appear only when source and target are textual near-duplicates, APSM has collapsed into RAG and should be deleted.

**Novelty honesty:**  
**Old cognitive science + old case-based reasoning**, sporadically attempted in ATP. Not a new philosophical idea. Still worth listing because the consensus stack almost always implements “analogy” as **embedding similarity**, which is the wrong inductive bias for mathematics. The implementable delta is partial morphisms over extracted schemas with Lean-checked repair.

---

## 4. Ramanujan as architectural constraint

The naming is not ornamental. Features of Ramanujan’s recorded cognition that pressure design:

1. **Generation without proof search.** Notebooks filled with identities far ahead of certification. → Need a generative channel driven by MSOF + catalogue memory + DCEF conjectures, not only goal expansion.  
2. **Catalogue density.** Enormous recognition memory for special forms. → MSOF + library of fingerprints of classical objects; DLL to name new ones.  
3. **Collaboration as dual roles with different virtues.** Hardy supplied skepticism, formalization, and connection to existing theory; Ramanujan supplied compression candidates. → Roles must differ in **loss functions and sensors**, not only prompts; Tribunal already matches the social structure — keep it, deepen it.  
4. **Aesthetic certainty as (fallible) internal score.** He was sometimes wrong; often right. → SMD + DCEF calibration, never self-admission.  
5. **Least like next-token proof dialogue.** → If the system’s only productive loop is Director monologue + prover, the name is false advertising.

Implication: the frontier is not a smarter frozen Director monologue. The frontier is **non-linguistic sensing, definition growth, obstruction theory, and dual-channel evidence**, with the giant used as a linguistic and tactical resource — a powerful librarian-orator, not the seat of mathematical perception.

---

## 5. What the frozen Director is good for (and not)

**Good for:** fluent decomposition, naming, informal proof sketches, translating between natural language and formal outlines, proposing tactics, writing docstrings for new defs, summarizing Prior-Art, role-play that is actually “format conversion” between stores.

**Not good for (while frozen):** minting genuinely new conceptual primitives in weight space; reliable taste; calibrated belief; perception of structure invariant to presentation; honesty about obstruction.

**Therefore:** do not unfreeze the giant on one local machine (cost and offline constraint). Put learning in small models that operate on **fingerprints, graphs, definitions, obstructions, and calibration**, all checkable.

If forced to cut scope to two mechanisms only: **DLL + OO**. They attack the deepest structural gaps (concept ceiling; failure as non-structure) and both emit Lean-checkable artifacts.

---

## 6. Anti-patterns this document refuses

- “Add a knowledge graph” without admission criteria, training signal, and falsification. (SMD uses a graph but specifies S and how it dies.)  
- “Multi-agent debate” as cognition. Debate without distinct sensors and losses is token heat.  
- “Reflection loops” that write prose into memory. Prose is not Tier-anything.  
- Unfreezing the Director or assuming datacenter RL.  
- Mechanisms whose only output is unfalsifiable narrative.  
- Claiming virginity for library learning, conflict-driven failure, or experimental mathematics.

---

## 7. Optional steelman: “the perception framing is romantic error”

A competent opponent would say: mathematics *is* formal derivation plus heuristic search; everything else is human folklore; scale the prover and the folklore falls out.

**Why that is partly right:** Tier-3 Lean is non-negotiable for the system’s value proposition. Without it, “research intelligence” becomes a blog. Many famous “intuitions” are post-hoc stories.

**Why that is not enough:**  
- Formal derivation does not choose worthwhile goals.  
- Heuristic search over proof states does not invent the definitions that change the state space.  
- Human folklore, when operationalized (experimental math, obstruction theory, library learning), has already been where many real advances came from. Ignoring it is not rigor; it is amputating sensors.  
- If pure NTP scale were sufficient, the research bottleneck would already be gone for contest-adjacent domains where scale has been applied hardest; instead we see strong provers and weak research taste.

The perception framing is an error only if left metaphorical. This document’s standard is: **perception = multi-channel encoding with invariance tests; obstruction = typed certificate; taste = map dynamics with delayed reuse reward; concept = admitted definition under ΔL.** Metaphors that cannot survive that translation should be discarded — including any of the six mechanisms that fail their falsifiers.

---

## 8. Suggested evaluation philosophy (for later human decision; not a build plan)

When these ideas are tested, optimize for **Ledger correctness per joule**, not for vibe:

1. Theorems admitted false → catastrophic (must stay ~0).  
2. False conjectures killed cheaply → good use of compute channel.  
3. Definitions that compress subsequent work → DLL working.  
4. Obstructions that speed search without false walls → OO working.  
5. High-S targets that actually get reused → SMD working.  
6. Fingerprint retrieval lift → MSOF working.

Do not declare victory because the Director tells a better story about what it did.

---

## 9. Final ordering for a human decision

If the owner wants maximum research-cognition delta under local constraints, fund in this order:

1. **DLL** — concept ceiling is the frozen Director’s real wound.  
2. **OO** — turns the Graveyard from storage into theory.  
3. **MSOF** — makes “other inputs than LLM” concrete.  
4. **SMD** — without taste, the system becomes an infinite homework machine.  
5. **DCEF** — institutionalizes experimental mathematics without contaminating theorems.  
6. **APSM** — valuable, but easiest to fake with RAG and hardest to extract schemas for; last.

Independent prediction about disagreement with a parallel agent: agreement is likely on “obstructions should be first-class” and “taste ≠ provability.” Disagreement is likely on whether **definition learning** or **multi-modal perception** is the highest-EV first move, and on how much weight to give Ramanujan-as-constraint versus Ramanujan-as-brand. This pass bets on **definitions first**, because a system that cannot grow concepts cannot be a research intelligence no matter how well it sees and ranks.

---

*End of independent ideation pass. Full argument is in this file; chat summary is intentionally not a substitute.*
