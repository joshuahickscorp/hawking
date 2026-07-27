# Ramanujan cognitive architecture — consolidated proposal

**Status: PROPOSAL. Not canonical. Nothing in the Continuum goal set has been changed.**

Two independent ideation passes, run in parallel without sight of each other, then
reconciled. Agreement between two agents on the same brief is weak evidence; the
divergences below are where the real questions are, and they are foregrounded deliberately.

---

## The diagnosis both passes reached independently

The specified stack — frozen Director, small specialists, Lean Tier-3, Tribunal, seven
stores, proof search — is the correct **substrate for certainty**. It is not a model of
mathematical **cognition**.

Grok's framing is the sharper one and is adopted here:

> The verifier loop is not the discovery loop. **Hardy was not Ramanujan. The current
> architecture is almost all Hardy.**

Both passes independently observed that the system is named after the mathematician whose
cognition it least resembles. Ramanujan computed obsessively, accumulated enormous
experience with concrete objects, and reported identities he appeared to *perceive*, often
unable to prove them. The formalization came afterwards, frequently from someone else. The
specified architecture is a superb Hardy with no Ramanujan attached.

Stated without technology, the stack structurally cannot:

1. form new conceptual units that reorganize the space of proofs;
2. perceive mathematical objects as multi-channel phenomena rather than as strings;
3. treat failure as positive structure rather than as a log;
4. allocate effort by significance rather than by provability;
5. fuse two epistemologies whose failure modes differ;
6. generate from catalogue-and-pattern rather than from proof-state expansion.

**The bound on this claim, which both passes volunteered independently:** AlphaProof and
DeepSeek-Prover reached elite performance on the consensus stack with no perceptual organ.
That is real evidence against the romantic framing — for *problem solving*. Those systems
solve stated problems; Ramanujan's job is to find statements worth stating. Provability and
significance are different objective functions, and everything below should be judged on
conjecture generation and selection, not on proving.

---

## Where the two passes converged

Four mechanisms, arrived at independently, with the same one ranked first.

| Capability | Claude | Grok | Both ranked |
|---|---|---|---|
| Grow new Lean-checked concepts | minted definitions | Definitional Library Learning | **1st** |
| Failure as structure | obstruction perception | Obstruction Objects | high |
| Non-linguistic object perception | Computational Retina | Multi-Sense Object Fingerprints | foundational |
| Lean/computation fusion | stereo evidence | Dual-Channel Epistemic Fusion | medium |

---

## Where they diverged — and the resolution

### Divergence 1: what significance *is*

**Claude:** interestingness = MDL compression gain. A theorem is a compression scheme;
Rogers–Ramanujan compresses two infinite families into one statement. Score a claim by the
description-length reduction it buys over the computed corpus. Ancestor: Schmidhuber's
compression-progress theory of creativity.

**Grok:** significance = map dynamics. A graph of Claims, Definitions, Methods,
OpenProblems, Obstructions, scored by
`ΔConnectivity + ΔCompression + Generativity + ObstructionYield − Cost`, trained on
mathlib/AFP growth history — which lemmas actually became high in-degree.

**Grok's self-criticism is the most valuable single line in either document:**

> **Fashion capture** — S learns "what mathlib already centralizes," systematically
> undervaluing Ramanujan-like outsider compressions that are not yet connected.

A significance model trained on the history of what became central will reliably devalue
exactly the case the project is named after. Grok raised this against its own mechanism.

**Claude's counter:** map dynamics has a cold start — an empty graph makes the score
meaningless — and its training signal is a *corpus* dependency, where MDL over
self-generated computation has neither problem. The offline and licensing discipline that
constrains everything else in this system is trivially satisfied by computed mathematical
data, because the system generates it.

**Resolution: they are phase-ordered, not competing.**

- **MDL is the bootstrap measure.** It works from turn one, needs no corpus, has no cold
  start, and is trainable entirely from self-generated computation.
- **Map dynamics is the mature measure.** It needs a populated graph before it means
  anything, and it is richer once it does — connectivity and generativity are real and MDL
  does not capture them.
- Blend as the graph fills. **Keep the exploration bonus for high-compression claims in
  sparse regions permanently**, at whatever blend ratio — that term is the anti-fashion-capture
  device, and it is what protects the namesake case from being scored away.

### Divergence 2: analogy — Grok had it, Claude missed it

**Analogy as Partial Structure Maps.** "This is like the proof that…" means parts
correspond, relations correspond, and a method transports with repair. Extract relational
schemas from Lean types; search for partial morphisms maximizing preserved relations; pull
strategies, lemma templates and obstructions along the morphism; repair; verify normally.

Grok is honest that this is old cognitive science — Gentner's structure-mapping, case-based
reasoning — and the implementable delta is that **the consensus stack implements analogy as
embedding cosine similarity, which is the wrong inductive bias for mathematics.** A failed
morphism should itself yield an obstruction object ("the map broke at exactness of …").

Its own kill criterion is good: if gains appear only when source and target are textual
near-duplicates, it has collapsed into RAG and should be deleted.

This is a genuine gap in the Claude pass. Polya's *Induction and Analogy in Mathematics* is
the canonical statement of why it belongs at the centre rather than the periphery.

### Divergence 3: self-model — Claude had it, Grok missed it

**Competence cartography.** Turn the Cartographer inward: a per-domain, per-complexity map
of where the system's own perception is reliable. Route away from blind spots, spend budget
on extending competence rather than exploiting it, and report honestly in the Q6 packet.

Low novelty, and necessary for honesty rather than capability. Its failure mode is the
obvious one — self-assessment is what a self-deceiving system gets wrong — so competence
must be **measured externally on held-out per-domain benchmarks, never introspected.** This
is MOP's producer/verifier separation applied to the system's model of itself.

---

## The merged set, ranked by expected value against local cost

1. **Definitional Library Learning** — mint Lean-checked definitions that shorten future
   proofs; the Director's *effective* vocabulary grows without retraining. Falsifier: hide a
   known concept, supply only the computation, see whether an equivalent definition is
   minted. This is the spec's Q2 rediscovery test moved from results to **concepts**.
2. **Obstruction Objects** — failure becomes reusable content with a free training signal
   from search traces; obstruction classes become candidate impossibility results. Falsifier
   must include whether obstruction-guided pruning *suppresses* proofs needing a non-obvious
   step — a measured cost, not an assumed-away one.
3. **Multi-Sense Fingerprints / Computational Retina** — the perceptual organ everything
   else stands on. Falsifier: does the latent space co-locate the two sides of a held-out
   nontrivial identity, beating a first-k-terms baseline?
4. **Significance — MDL now, map dynamics later**, exploration bonus permanent. Falsifier:
   rank known theorems; if the deep ones do not outrank arbitrary true identities of similar
   length, the measure is wrong.
5. **Dual-Channel Epistemic Fusion** — the useful new object is *high computational support,
   Lean-blocked at an identified lemma* = a localized research target, which the current
   architecture cannot name. A computational counterexample against a claimed proof is a
   defect in the system, and reproduction must outrank proof-search confidence — the same
   rule MOP's `adjudicate` already enforces.
6. **Analogy as Partial Structure Maps** — morphisms over extracted schemas, not cosine.
7. **Competence Cartography** — externally measured, never introspected.

---

## The thread back to MOP

MOP's thesis was learning from non-symbolic streams. Here **the non-symbolic stream is
computation itself.** That is the real synthesis, and it is why the substrate work is not
conceptually wasted even though none of its code transfers: the principle survives, the
domain changes.

Separately and already established: MOP's `method/` layer supplies the refusal machinery —
SESOI, multiplicity policy, stop rules, claim ceilings, producer/verifier separation,
`wording_check`, `adjudicate`. Every mechanism above is a new way for the system to fool
itself, which raises the value of that transfer rather than lowering it.

---

## What has NOT been decided

- The goal set is unchanged. This is a proposal awaiting a human decision.
- No mechanism here is scheduled, costed in wall-clock, or added to the Continuum.
- Ramanujan remains Continuum step 14, gated behind `HAWKING_EVOLUTION_COMPLETE`.
- `RAMANUJAN_RESEARCH_AUTHORIZED` remains false.

Sources: `RAMANUJAN_COGNITION_CLAUDE.md` and `RAMANUJAN_COGNITION_GROK.md`, both written
before either saw the other.
