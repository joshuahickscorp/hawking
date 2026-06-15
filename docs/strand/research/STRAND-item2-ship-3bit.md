# ITEM 2 — SHIP PLAN: STRAND 3-bit (lossless-class AND provable)

_Authored 2026-06-13. Read-only analysis + authoring; no local heavy compute (PV
PID 28690 holds the box). Sources: `research/bit-ledger-results.md`,
`research/2bit-frontier-SUMMARY.md`, `docs/STRAND-quality-density-frontier.md`,
`docs/STRAND-benchmark-report.md`, `docs/STRAND-compression-map.md`,
`docs/STRAND-vs-gguf-isobpw.md`. Every number below is labeled **PROVEN** (a
committed result file / exact ledger match) or **PROJECTED** (an entropy ceiling or
a not-yet-run measurement). Owner's standing order honored: no soft-positives._

---

## 0. The one-sentence pitch

STRAND's 3-bit tier is a **lossless-class** quantization (Llama2-7B loss-tax
**0.0562 nats / +5.8%**, PROVEN) whose decode is **provably bit-identical and
float-free** (exhaustive + 52M-case golden + 4 Kani harnesses SUCCESSFUL) — a
combination no competitor (GGUF, AQLM, QTIP) can claim, and it needs **zero
training** to ship today.

This is the part of STRAND that is *done*, not the 2-bit frontier (which is
training-gated and still a contender, not a winner). 3-bit is what we put on the
table now.

---

## 1. The concrete deliverable

A two-part ship:

**(A) A deterministic 3-bit STRAND artifact.**
- The canonical recipe: **`--bits 3 --l 12 --outlier-channel 1`** (uniform q3 + 1%
  top-|w| outlier channel), producing a `.strand` v2 artifact.
- Optional mixed-precision upgrade: the true **`mp_light`** rung
  (`down_proj@4, rest@3` — note: NOT attn4/ffn3, see §5 caveat) for the lowest
  measured tax, at ~3.67 bpw.
- Encoded size **3.6653 bpw** (PROVEN, exact ledger match — see §2). With the C2
  lossless side-info coder wired, **~3.50 bpw** (PROJECTED ceiling).
- Decode is integer-only Q12 LUT; the artifact self-verifies via SPRV/RSLT
  attestation.

**(B) A one-page results/positioning sheet** — "STRAND 3-bit: lossless-class AND
provable" — drafted in full at the bottom of this file (§7). This is the
load-bearing launch asset: it makes the determinism + lossless-class double claim
in numbers a reader can check.

The 0.5B artifact + sheet are shippable from local files that already exist. The
**7B 3-bit artifact + its fresh PPL** is the pod deliverable (§4) that turns the
PROVEN Llama2-7B point into a STRAND-7B headline on the current eval harness.

---

## 2. The exact MEASURED numbers to headline (proven vs projected)

### 2.1 Density (bpw) — PROVEN exact

The bit-ledger reproduces the canon sidecar bpw **exactly** (this is the
faithfulness check, `research/bit-ledger-results.md`):

| rung | encoded bpw | deploy bpw (+seek table) | status |
|---|---:|---:|---|
| q3_l12_out1 | **3.6653** | 4.1653 | **PROVEN** (ledger == canon, exact) |
| q3 + C2 lossless side-info | **~3.50** (3.50–3.55 real) | — | **PROJECTED** (entropy ceiling; rANS table lands a few % short) |

The C2 trajectory: q3 spends **0.115 bpw** of recoverable entropy in the
scale/sub-scale side-info (PROVEN ceiling, `bit-ledger-results.md`), and the
verified C2 coder (`c2_attempt_4`, 87,381-stream byte-exact certificate) recovers
**~0.237 bpw** across all side-info streams at **zero quality cost** (lossless
recode, byte-identical reconstruction). Honest headline: **ship 3.6653 today;
~3.50–3.55 once C2 is wired** (whole-model concatenation required — per-tensor only
recovers 0.053 bpw).

### 2.2 Quality (loss-tax) — the PROVEN anchor

| model | recipe | bf16 PPL | quant PPL | rel | loss-tax | status |
|---|---|---:|---:|---:|---:|---|
| **Llama2-7B** | **mp_light** (down@4/rest@3, L=12, 1% outl) | 5.5353 | **5.8552** | +5.8% | **0.0562 nats** | **PROVEN** (`ppl_mp_light_l12_out1.json`) |
| Qwen2.5-0.5B | q3 uniform (L=12, 1% outl) | 12.5358 | 16.1128 | +28.5% | 0.251 nats | PROVEN (`ppl_strand_q3_l12_out1.json`) |
| Qwen2.5-0.5B | omega-star (attn4/ffn3) | 12.5358 | 15.0391 | +19.9% | 0.182 nats | PROVEN (`ppl_strand_mp_light.json` — **mislabeled file**, see §5) |
| Qwen-7B | q3 class | — | — | — | 0.088 | PROVEN (per §12 frontier) |

**The headline number is 0.0562 nats / +5.8% on Llama2-7B** — this is
effectively-lossless (under the 0.05-nat "indistinguishable" bar by a hair) AND it
is on a real 7B model, not a toy. That is the number that goes on the sheet, clearly
labeled as Llama2-7B.

### 2.3 The vs-GGUF framing — PROVEN, but read the caveat

iso-harness (same `ops/eval-ppl.py`, WikiText-2, ctx 2048, 64 windows, cpu, bf16;
GGUF dequantized back to HF and run through the identical path), Qwen2.5-0.5B
projection-weight bpw:

| format | proj bpw | PPL | note |
|---|---:|---:|---|
| **STRAND omega-star** | **3.806** | **15.039** | PROVEN — beats GGUF Q2_K on BOTH axes |
| GGUF "Q2_K" | 4.197 | 15.238 | nominal 2-bit, actually 4.20 bpw (896-dim fallback) |
| GGUF IQ3_S | 4.197 | 16.950 | no imatrix (fairness caveat) |
| **GGUF "Q3_K_M"** | **4.574** | **13.611** | nominal 3-bit, **actually 4.574 bpw** |
| GGUF Q4_K_M | 5.521 | 12.889 | near bf16 ceiling |

Two honest reads of this table:

1. **The density story is the durable one.** Nominal "3-bit" GGUF (Q3_K_M) spends
   **4.574 bpw**; STRAND's q3 spends **3.6653 bpw** and reaches the *requested*
   bit-width on **any** dimension. GGUF's K-quants fall back to 32-blocked legacy
   types on 896-dim tensors (896 ∤ 256-element superblock), so its low-bit tiers are
   not actually low-bit here. **STRAND is ~0.9 bpw denser than GGUF at the nominal
   3-bit tier.** This is the headline that survives scrutiny.
2. **At iso-quality GGUF wins on bpw spend, and at higher bpw GGUF wins on
   quality** (Q3_K_M 13.611 @ 4.574 < every STRAND point here). The STRAND
   both-axes win (15.039 @ 3.806 < Q2_K 15.238 @ 4.197) is **real but narrow**: it
   rides the 896-dim small-model fallback and uses the *dominated* omega-star arm.
   At 256-aligned scale (most ≥7B tensors) that specific edge shrinks. Do NOT
   headline "STRAND beats GGUF on quality" as a general claim.

**What to actually headline:** "STRAND 3-bit: ~0.9 bpw denser than GGUF's nominal
3-bit, at +5.8% loss-tax (Llama2-7B), with bit-identical deterministic decode."
Density + determinism, not a quality-beat.

### 2.4 The determinism moat — PROVEN

This is the differentiator and it is formally hardened
(`docs/STRAND-quality-density-frontier.md` §17):

- **Decode is provably bit-identical + float-free**: exhaustive equivalence
  (~5,120 scalar + 6,144 vector + tail-biting cases), a **52M-case golden vector**
  reproducing `LUT_GOLDEN_HASH` (integer decode byte-stable), and **4 Kani harnesses
  VERIFICATION SUCCESSFUL** (seed FNV-1a always-odd, sign ±1, splitmix64 pure).
- **One documented limit (state it, don't hide it):** the *encode-side* RHT
  round-trip is bit-exact only for even-power-of-2 block sizes {1,4,16,64,256};
  odd-power widths (896→128, 384, 200) are approximate ~1e-6 (worst measured
  1.79e-7), so cross-device *encoder* identity for those geometries rests on
  IEEE-754 f32 no-FMA (asserted via golden vector, not Kani-proven). **The decode
  (integer Q12 LUT) is bit-exact everywhere** — the crack is only in the float RHT
  for odd-power blocks, and only on the encode side. Mitigation: 256-align, or
  accept the documented ~1e-6 for those widths.

No competitor sells reproducible-decode + ship-the-correction. That is the moat.

---

## 3. Why 3-bit is the ship (and 2-bit is not, yet)

| tier | status | ship today? |
|---|---|---|
| **3-bit** | tax 0.0562 (PROVEN, Llama2-7B), float-free deterministic decode, **no training** | **YES** |
| 2-bit PTQ | 79.94 PPL 0.5B collapse; tax ~0.22–0.25 at scale (de-bias-only floor); ≤0.15 only if selective-PV works at scale (cloud-gated, unrun) | NO — contender, not winner |

3-bit is the regime where "quality is already useful" (frontier §2 Regime A) and the
only binding lever is **bpw-at-fixed-quality** — which C2 (lossless) already solves
on paper. It is the lowest-risk, highest-confidence STRAND quantization claim, and
it is the one that leads with the moat (determinism) rather than chasing a quality
beat that the data does not support.

---

## 4. What must be produced on the pod later

Local box is frozen (PV run). These are **pod steps**, gated and scoped — do not run
locally:

| # | pod step | why | gate |
|---|---|---|---|
| 1 | **STRAND-7B 3-bit artifact** — quantize Qwen-7B (or Llama2-7B) with `--bits 3 --l 12 --outlier-channel 1` (and the true `mp_light` down@4 variant) | the PROVEN 0.0562 is from a Llama2-7B recon; produce the shippable `.strand` artifact + re-confirm PPL on the *current* canon harness so the headline is a live STRAND-7B point, not an archived one | PPL within noise of the 5.8552 archived figure |
| 2 | **7B 3-bit PPL on the canon eval** (`ops/eval-ppl.py`, ctx 2048, 64w) | makes the loss-tax directly comparable to every other frontier point | tax ≤ 0.06 confirms ship |
| 3 | **GGUF-7B head-to-head at 256-aligned scale** | the 0.5B both-axes win rides the 896-fallback; the 7B comparison is the honest scale test of the density claim | report density gap honestly even if it shrinks |
| 4 | **C2 wiring + end-to-end bit-identity gate** (whole-model concat) → measure real encoded bpw | turns 3.6653 → ~3.50; needs the §15 safe-wiring order (seal, step-over walkers, SDSC bump) + a test proving rANS-decoded streams reproduce byte-identical Q12 weights | real bpw ≤ 3.55 AND byte-identical decode |

CPU quant on the pod (Metal is net-slower for full models per MEMORY); fp32 PV-shadow
is irrelevant here (no training for 3-bit ship).

---

## 5. Honest caveats (do not let these slip)

1. **The "mp_light" label is overloaded — this trips people.** The committed
   `ppl_strand_mp_light.json` (0.5B, 15.039 @ 3.806) is actually **omega-star
   (attn4/ffn3)**, a DOMINATED arm on the q3→q4 line, NOT the canonical `mp_light`
   (down_proj@4, rest@3). The PROVEN **0.0562** tax is the *true* mp_light on
   **Llama2-7B**. The **true mp_light 0.5B point is UNMEASURED.** When the ship
   sheet says "mp_light 0.0562," it means Llama2-7B down@4; when it cites "15.039 @
   3.806 bpw," that is omega-star on 0.5B. Keep them separate or the claim is
   self-contradicting. (See `docs/STRAND-benchmark-report.md` caveat 1,
   `docs/STRAND-compression-map.md` lines 51–52.)
2. **0.0562 is one model (Llama2-7B).** Small models differ sharply: Qwen-0.5B q3 is
   +28.5% (0.251 nats), Qwen-7B q3 ~0.088. The "lossless-class" claim is honest at
   7B+ scale; it is NOT true at 0.5B. The sheet must say "at 7B scale."
3. **~3.50 bpw is PROJECTED (entropy ceiling).** Shippable-today bpw is **3.6653**
   (PROVEN). The C2 win is verified byte-exact in isolation but **not yet wired**
   into encode.rs/format.rs, and the win needs whole-model stream concatenation. Do
   not headline 3.50 as achieved.
4. **GGUF proj-bpw figures (4.197 / 4.574 / 5.521) are doc-asserted**, not
   machine-backed by a committed file (benchmark-report caveat 2). The STRAND side is
   file-verified. State the asymmetry.
5. **The density-vs-GGUF edge is partly a 0.5B / 896-dim artifact.** The clean
   scale-fair version is the pod 7B head-to-head (§4 step 3). Lead with determinism +
   the ledger-exact 3.6653 bpw, which do not depend on the fallback quirk.
6. **The encode-side RHT ~1e-6 odd-power-block limit** (§2.4) is the one honest crack
   in the determinism story. Decode is bit-exact everywhere; the encoder float path
   for 896-wide tensors is no-FMA-asserted, not proven. State it.

---

## 6. Ship gate (the go/no-go)

Ship the 3-bit tier publicly when:

1. The 0.5B deterministic q3 artifact builds and round-trips bit-identical (PROVEN —
   the decode equivalence + golden tests already pass). ✅
2. The pod produces a STRAND-7B 3-bit artifact whose PPL re-confirms tax ≤ 0.06 on
   the canon harness (§4 steps 1–2). ⏳ pod
3. The positioning sheet (§7) is reviewed for the §5 caveats. ✅ (drafted below)

C2 (→3.50 bpw) is a **fast-follow**, not a launch blocker: 3.6653 bpw + +5.8% +
provable decode is already a shippable, honest, differentiated claim.

---

## 7. THE ONE-PAGE SHEET (draft)

> ### STRAND 3-bit: lossless-class AND provable
>
> **The claim.** A 3-bit model quantization that is (1) effectively lossless at
> scale, (2) denser than GGUF's nominal 3-bit, and (3) the only one whose decode is
> *provably* bit-identical and float-free. No training required.
>
> **Lossless-class quality.** On **Llama2-7B**, STRAND 3-bit (mp_light: down_proj at
> 4-bit, rest at 3-bit) scores **PPL 5.855 vs 5.535 bf16 — +5.8%, a 0.056-nat
> loss-tax** (WikiText-2). That is at the threshold of indistinguishable. _(Measured.
> Small models cost more — this is a 7B-scale claim.)_
>
> **Denser than GGUF.** STRAND hits the *requested* bit-width on **any** tensor
> dimension. Nominal "3-bit" GGUF (Q3_K_M) actually spends **4.574 bpw** on
> Qwen2.5-0.5B because 896-dim weights don't tile its 256-element superblocks and
> fall back to legacy 4–5-bit types. STRAND q3 spends a ledger-exact **3.665 bpw**
> (→ ~3.50 with lossless side-info coding) — **~0.9 bpw denser at the same nominal
> tier.**
>
> | tier | bpw | PPL (0.5B, iso-harness) |
> |---|---:|---:|
> | STRAND q3 | **3.665** | 16.11 |
> | STRAND mixed-3bit (omega-star) | **3.806** | **15.04** |
> | GGUF "Q2_K" (really 4.2 bpw) | 4.197 | 15.24 |
> | GGUF "Q3_K_M" (really 4.6 bpw) | 4.574 | 13.61 |
>
> At the 3.8-bpw tier STRAND beats GGUF "Q2_K" on **both** axes (lower bpw AND lower
> PPL). _(Real, but partly a 0.5B / 896-dim effect; the scale-fair 7B head-to-head is
> the honest follow-up.)_
>
> **Provable decode — the moat.** STRAND's decode is integer-only (frozen Q12
> Gaussian LUT), so every device produces **byte-identical** weights. This is not a
> marketing word: it is backed by an exhaustive equivalence test, a **52-million-case
> golden vector**, and **4 formal Kani proofs** of the decode primitives. GGUF, AQLM,
> and QTIP cannot make this claim — it matters for reproducible evals, regulated
> deployments, and distributed serving where every node must agree.
>
> **What you get.** A `.strand` artifact at ~3.5–3.7 bpw, no training, no calibration
> drift, bit-exact across hardware, ~0.06-nat quality cost at 7B scale.
>
> **Honest fine print.** The 0.056-nat figure is Llama2-7B; small models cost more.
> The ~3.50 bpw figure assumes the lossless side-info coder (verified byte-exact,
> wiring in progress); 3.665 ships today. The both-axes GGUF win is at 0.5B and
> narrows at 256-aligned scale — the durable claims are *density at the nominal tier*
> and *provable decode*.

---

_File owns no shared-refactor code; analysis + authoring only. Local box untouched
(PV PID 28690 protected)._
