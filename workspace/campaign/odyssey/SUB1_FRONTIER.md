# SUB-1 / SUB-0.2 BPW FRONTIER PACKET (shared context for Grok novelty lanes)

QUESTION: how aggressively can Hawking Gravity self-optimize toward **sub-1 BPW**, and even
**sub-0.2 BPW (1/5 bit)** — is it reachable NOW on the current patients, and by what mechanisms?
This is a NOVELTY fanout (S004 §19/§54). Propose the most aggressive CREDIBLE mechanisms; the
deterministic engine will then generate + Doctor-qualify candidates. Usage is uncapped — be bold,
but every claim obeys the guardrails.

## HARD GUARDRAILS (non-negotiable — a proposal that violates these is worthless)
1. **Complete accounting / NO fake density.** complete_bpw counts EVERY byte attributable to the
   representation: payload + scales + biases + tables/codebooks + offsets + correction streams +
   tier/router metadata + alignment + container + mandatory reconstruction state. Report `nominal_bits`
   AND `complete_bpw`. A "0.2-bit" payload with a 3-BPW codebook is a 3-BPW representation.
2. **Activation-aware ONLY.** Hawking's hard lesson: sub-bit evaluated on gaussian/synthetic proxy
   activations is an ARTIFACT (Qwen3.8 sub-bit "wins" all collapsed on PQ proxy, output-div ~0.69).
   The ONE real sub-bit success (GLM, 0.755 cos @ **0.167 BPW**) came from REAL activations. Every
   mechanism must be evaluated against REAL forward-pass activations / held-out behavior.
3. **Doctor is authority.** Not cosine, not MSE, not one prompt. Real held-out capability, not
   memorizing the gate. A gibberish sub-0.2 artifact is still valuable IF it LOCALIZES the failed
   component — say which organ/expert/channel broke.
4. **Native execution story.** The representation must EXECUTE natively or have a credible native
   path. A tiny stored object that EXPANDS to a dense body before compute is a fake win — account the
   reconstruction bytes/token + FLOPs + decode kernel cost. Judge the executable object physically.
5. **Stored vs ACTIVE sub-1 are different wins.** For MoE, ACTIVE (touched/token) sub-1 is a legit
   NX win even if complete STORED body stays >1 BPW (bible §29). Distinguish them explicitly.

## PRIOR HAWKING SCIENCE (build on it, do not rediscover)
- GLM activation-aware: **0.755 cos @ 0.167 BPW on REAL activations** — sub-1/5 precedent EXISTS.
- Qwen3.8 gaussian-proxy sub-bit: ALL collapsed 0.5-0.8 BPW — the proxy trap.
- Q80 mixed complete 1.43 BPW at full 8-bit non-expert (binary / rice+q1 outliers / low-rank r160).
- MoE: 16x selection (top-8/128, selected/full 0.0625) is an EXECUTION lever, not a compression stat.
- Cold experts NOT universal (0 cold measured on O005/O003/O006) → route-uniform; sub-1 must come from
  structure/correction/latent, not popularity skew.
- Runtime is often kernel-bound not bandwidth-bound → fewer bytes only helps if decode is efficient.

## MECHANISM FAMILIES TO PUSH (your lane attacks one deeply)
- **Base + correction**: extreme low-bit base (ternary/binary/sub-bit) + sparse conditional repair
  (residual tier, selected hi-prec channels, expert/route-conditioned). When does a 0.3-BPW base + a
  small correction beat a naked 2-BPW codec at lower complete_bpw?
- **Structural / shared-basis**: shared base + per-expert delta, expert-family clustering, shared
  codebooks, cross-expert low-rank / dictionaries. Sub-1 from STRUCTURE across experts/layers, not
  per-weight bits. (Test held-out function — do not assume experts redundant because "expert".)
- **Latent / generated / procedural**: weights generated from a small seed + deterministic rules /
  a tiny hypernetwork / procedural reconstruction. The deepest sub-0.2 lever: GENERATE not STORE.
  Account the generator bytes + reconstruction FLOPs honestly.
- **Active sub-1 (MoE)**: per-token touched < 1 BPW via expert-gather + extreme per-expert quant +
  route-conditioned representation; stored may exceed 1, active < 1.
- **Entropy / information floor**: entropy-coded metadata, exact-rational rates, the actual
  information content of the learned function; the theoretical floor and how close we can get.
- **Matryoshka / progressive**: T0 minimal executable body at effective sub-1; tiers add fidelity.

## REQUIRED OUTPUT (structured, per proposed mechanism — aim for several)
mechanism (name + one-line) · complete_byte_accounting (exactly how sub-1/sub-0.2 is honestly reached,
what every byte is) · stored_bpw + active_bpw (separate) · expected_reachable_bpw (honest range) ·
quality_risk + which component likely limits · cheapest_falsifier (the single cheapest experiment that
would kill it) · execution_path (native? decode/reconstruction cost/token + kernel) · applicability
(archs/organs/patients it fits — MoE-expert vs dense vs hybrid-state) · confidence + why · transfer
(does it generalize across the cohort?). Cite prior art where relevant (QTIP/AQLM/QuIP#/BitNet/1-bit
PTQ/product-quantization/hypernetworks). Be terse; tables over prose. Mark UNKNOWN, never guess a number.

## AMENDMENT (2026-08-19): PER-COMPONENT EXTREME + OPPORTUNISTIC DESCENT
Target is TOTAL complete_bpw sub-1 / sub-0.2 via HETEROGENEOUS per-component budgets. A single
component (organ/expert/channel/layer) floor may go to **1/10 (0.1)** or **1/100 (0.01) BPW** — or
effectively ~0 for a component that collapses to a shared codebook index / low-rank factor /
dropped-to-constant + sparse correction / pruned-with-repair. A few protected components stay high;
MOST may collapse near-zero → the TOTAL falls far below 1. PUSH EVERY COMPONENT; be OPPORTUNISTIC
about potential (don't leave a component conservative if it might tolerate far less). The system is
SMART ENOUGH TO STOP — Doctor + false-win gates catch a component that actually breaks, so descend
aggressively (sensitivity-guided binary/geometric descent to the boundary, then repair). Complete
accounting always; a component that expands to dense before compute is not a real sub-0.1 win.
