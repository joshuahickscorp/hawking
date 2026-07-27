# Ramanujan's cognitive architecture — independent pass (Claude)

Written before sight of the parallel pass.

## Diagnosis

The specified stack is LLM proposes, Lean disposes. Everything else is plumbing. Its
cognition is **entirely linguistic**: the system's only representation of a mathematical
object is a token sequence naming it, and Lean checks syntax-directed derivations over those
names. At no point does the system hold a non-linguistic representation of a mathematical
object.

A mathematician holds several. They see a graph, feel a symmetry, recognize that a sequence
*looks like* a partition count, sense that a bound *smells like* Cauchy-Schwarz. Those
representations are learned from experience with instances, not from reading definitions.

The system is named after the mathematician whose cognition least resembles the specified
architecture. Ramanujan computed obsessively, accumulated enormous experience with concrete
objects — continued fractions, series, partitions, highly composite numbers — and reported
identities he appeared to *perceive*. He frequently could not prove them. Hardy formalized.

**The stack has a formalizer and a prover and no perceiver.**

One bound on the claim, stated up front: AlphaProof and DeepSeek-Prover reached elite
competitive performance with essentially the consensus stack, no perceptual organ. That is
strong evidence against the romantic framing — for *problem solving*. Those systems solve
stated problems. Ramanujan's goal is to find statements worth making. The value functions
differ: provability versus significance. Everything below earns its keep on conjecture
generation and selection, not on proving, and should be judged there.

---

## M1 — The Computational Retina

**Capability lacked:** a non-linguistic representation of mathematical objects.

**Mechanism.** Run massive cheap computation over families of objects — integer sequences,
group elements, polynomial families, graph families, modular form coefficients, partition
statistics. Train a small self-supervised model to predict and compress them: next term,
masked value, object from partial description. The learned latent space is the perceptual
organ. Proximity in it expresses a similarity no token-level LLM similarity captures.

Consumes computational streams (the Computationalist role currently just runs tools and
discards the experience). Emits object embeddings and prediction residuals. Does not touch
Lean directly; it feeds conjecture generation, which then formalizes.

**Practical advantage that matters here:** the data-discipline constraint — open, licensed,
deduplicated, contamination-checked — is *trivially satisfied* for computed mathematical
data, because the system generates it. The licensing and contamination problems that
dominate text training simply do not arise. Offline is not a limitation for this organ; it
is its natural habitat.

**Failure mode.** Latent similarity is not mathematical relevance. The model learns surface
numerical features — magnitude, parity, growth rate — and clusters on those. This is the
default outcome of representation learning, not an edge case.

**Falsification.** Hold out known nontrivial identities (Rogers–Ramanujan, and others whose
two sides are computationally dissimilar). Does the latent space place the two sides closer
than chance, and closer than a first-k-terms cosine baseline? If a representation cannot
co-locate the two sides of an identity it was never shown, it is not perceiving structure.
Abandon on failure to beat the naive baseline.

**Novelty: partial.** Davies et al. (Nature 2021) used supervised models to surface
relationships in knot theory and representation theory; OEIS-based ML exists. Those are
supervised on chosen targets. The move here is making it self-supervised, primary, and the
residual the discovery signal.

---

## M2 — Interestingness as compression gain

**Capability lacked:** a mechanical notion of significance. Proof search optimizes
provability; nothing in the spec says which *true* statements matter.

**Mechanism.** A theorem is a compression scheme. Rogers–Ramanujan compresses two infinite
families into one statement. Score a candidate claim by the description-length reduction it
buys over the computed corpus: how much shorter does the corpus become if this fact is
known? Maintain the compression model from M1; interestingness is MDL gain, and research
targets are high-residual regions adjacent to well-compressed ones.

This gives the Economist role something to price other than compute, and the Conjecturer
something to optimize other than plausibility.

**Failure mode.** MDL rewards broad trivial restatements. Worse, the corpus is whatever you
chose to compute, so the measure finds structure only where you already looked — selection
bias is built into the objective.

**Falsification.** Rank historically known theorems by MDL gain over a computed corpus. Do
the deep ones outrank arbitrary true identities of similar length? If Rogers–Ramanujan does
not beat a random true identity, the measure is wrong and no amount of tuning saves it.

**Novelty: recombination, and I should name the ancestor.** Schmidhuber's formal theory of
creativity — interestingness as compression progress — is exactly this idea. What is new is
using it as the conjecture value function inside a Lean-grounded loop where the compression
corpus is generated by the system and the winners are formally checkable.

---

## M3 — Obstruction perception

**Capability lacked:** reasoning about why things fail. The Graveyard is storage; a
mathematician *sees* an obstruction.

**Mechanism.** Every failed proof attempt yields a (proof-state, failure) pair, and search
generates millions for free. Cluster failures into obstruction classes. Train a predictor
from proof-state to obstruction class and expected blocking depth.

Two uses. The cheap one: prune before expanding, saving large amounts of Lean time. The
interesting one: surface obstruction classes as *objects of study*. "These four hundred
failures share an obstruction" is a mathematical observation, and sometimes an impossibility
theorem in waiting.

**Failure mode.** Learns syntactic surface features of failure — this tactic fails on long
goals — rather than mathematical obstruction. And obstruction models are conservative by
construction: they may prune exactly the unusual step that would have worked, suppressing
creativity precisely where it is needed.

**Falsification.** Two tests, and the second matters more. Does it predict obstruction class
better than a depth/length baseline on held-out blocked approaches? And: measure whether
obstruction-guided pruning reduces the rate of finding proofs that required a non-obvious
step. That cost must be measured, not assumed away.

**Novelty: the pruning half exists in pieces. Obstruction classes as first-class candidate
impossibility results appear underexplored.**

---

## M4 — Stereo evidence

**Capability lacked:** the spec has a hierarchy — computation is a tool, Lean is the
verifier — where it should have a fusion.

**Mechanism.** The two have genuinely different failure modes, which is the precondition for
fusion beating either. Give every claim a two-dimensional coordinate: computational support
against formal status. The off-diagonal cells are where the value is.

- **High computational support, Lean-blocked at an identified lemma** — a *localized
  research target*. This is the most valuable object the system can produce and the current
  architecture has no name for it.
- Low computational support, formally derivable — likely vacuous or degenerate; flag it.
- **Computational counterexample against a claimed proof** — a defect in the system itself,
  a formalization error or a bug. Reproduction must outrank any amount of proof-search
  confidence, exactly as MOP's `adjudicate` puts reproduction above panel votes.

**Failure mode.** Computational support is only as good as the sampling, and structured
adversarial sampling is itself a research problem. Uniform sampling finds nothing. Also
"strong evidence, unproven" is the normal state of every hard conjecture, so the cell may
just enumerate known hard problems without adding anything.

**Falsification.** Take historical cases where computation strongly indicated a result
before it was proved. Does the fusion coordinate flag them ahead of their proof date, or
does it merely relabel the famous open problems?

**Novelty: modest.** Closest to existing practice. The contribution is making the coordinate
first-class and defining the localized-target object.

---

## M5 — Minted definitions: dissolving the frozen-Director ceiling

**Capability lacked, and this is the deep one.** A frozen Director's conceptual vocabulary
is fixed at pretraining. It can become better at proving and never better at *conceiving*.
Small adapters retune the mapping, not the ontology. Genuine mathematical advance often
requires a new *definition* first, with the theorems following — Grothendieck's entire
method.

**Mechanism.** Let the system mint definitions and give the Director names it can use. When
the retina finds a recurring latent cluster with no existing name: characterize it
computationally, attempt a Lean definition, and if the definition typechecks and the
cluster's members provably satisfy it, the concept enters the Librarian with a name, a
definition and a membership test. It is then in the retrieval corpus and can appear in
Director context.

The Director's *effective* vocabulary grows without retraining, and every new concept is
Lean-checked rather than invented jargon.

**Failure mode.** Definition spam — thousands of useless named concepts poisoning retrieval.
The MDL gate from M2 is not optional here: a definition earns a name only if it compresses.

**Falsification.** Hide a known concept from the corpus — highly composite numbers, or a
specific invariant — give the system the computational data, and see whether it mints an
equivalent definition. This is the spec's Q2 hidden-rediscovery test moved from *results* to
*concepts*, which is both harder and more informative.

**Novelty: highest of these.** It attacks the frozen-Director ceiling directly and it is
cleanly testable.

---

## M6 — Competence cartography

**Capability lacked:** the system has no model of where its own perception is reliable.

**Mechanism.** Turn the Cartographer inward. Per-domain, per-complexity calibration of both
retina and prover. Route away from blind spots, report honestly in the Q6 packet, and spend
budget on extending competence rather than exploiting it.

**Failure mode.** Self-assessment is precisely what a self-deceiving system gets wrong.

**Mitigation, and it is the MOP transfer applied inward:** competence must be *measured
externally* on held-out per-domain benchmarks, never introspected. Producer and verifier
stay structurally separate even when the subject is the system itself.

**Falsification.** Does declared competence predict held-out per-domain performance? If the
system claims competence where it has none, the organ is worse than absent.

---

## Ranking by expected value against cost

1. **M2** — cheapest, principled, supplies the value function the spec lacks outright.
2. **M1** — the foundation M2, M3 and M5 all stand on. Highest leverage.
3. **M5** — highest novelty; depends on M1 and M2.
4. **M3** — free training data, immediate compute savings, real and measurable creativity risk.
5. **M4** — cheap, modest novelty, good hygiene, defines one genuinely useful new object.
6. **M6** — low novelty, necessary for honesty rather than capability.

## The thread back to MOP

MOP's thesis was learning from non-symbolic streams. Here the non-symbolic stream is
**computation itself**. That is the real synthesis, and it is why the substrate work is not
conceptually wasted even though none of its code transfers: the principle survives, the
domain changes.
