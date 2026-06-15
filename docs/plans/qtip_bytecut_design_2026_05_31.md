# QTIP trellis byte-cut design — 2026-05-31 (DESIGN ONLY, no code/commits)

> **Scope.** A design doc, not an implementation. It evaluates **QTIP-style
> trellis-coded quantization** as the *deep* axis-2 byte-cut (bible §8.1 L1.5
> resurrection, §2 axis-2 QTIP row) for the dense decode GEMV (M=1) on
> Qwen2.5-3B / M3 Pro. It is the **contrast codec** the cheaper-decode-Q3 design
> (`plans/cheaper_decode_q3_design_2026_05_31.md` §2.2) explicitly **deferred** —
> this doc is that deferral written up in full, so an attended Colab+kernel
> session can act on it cold.
>
> Per bible §3.0, the decode-kernel-microopt track is **CLOSED** (the Q4_K predec
> GEMV is at the Apple-GPU memory-model optimum; clean dec_tps ~31, ~33–34 with
> the A6.5 f16s flag). Forward progress to >34 dense tps is **fewer-bytes / spec /
> stateful ONLY**. QTIP is the deepest fewer-bytes lever: ~3.0–3.25 bits with
> near-Gaussian-optimal quality-per-bit, and — critically — a **lookup-free**
> decode that is *gather-free on the Apple GPU* (the binding feasibility
> constraint that killed the L1.5 raw-codebook).
>
> No source edited. No cargo. No GPU. No Colab. No commits. Output is this file.

---

## 0. TL;DR / recommendation

**QTIP is the surviving deep byte-cut codec** (`reports/dead_levers.md:101,109`:
the raw learned codebook L1.5 is **Type-1 dead** on the Apple gather wall; QTIP
survives "by construction" because its codes are *computed arithmetically on
contiguous bits*, not gathered from a table). This doc designs it honestly. The
verdict up front:

- **The byte-cut is large and real on paper.** ~3.0 bits vs Q4_K_M's ~4.5
  effective bits raises the dense tps wall from ~66 to ~99 (bible §2/§7.0); at
  the block level a QTIP-3.0 super-block is **~96–104 B vs Q4_K-predec-f16s's
  176 B (−41 to −45%)** and vs the f32 default's 208 B (−50 to −54%). That is a
  **bigger byte-cut than f16-predec-Q3's 128 B (−27%)** — QTIP is the deeper cut.
- **It is gated on TWO things, in order, neither of which we can clear locally:**
  1. **A quality oracle** (offline, Colab-class): KL / logit-cosine / token-
     divergence vs Q4_K_M on a **code-representative** corpus — the same metric
     class the W4A8 quality work uses (`reports/dead_levers.md:118`;
     `plans/stateful_moat_continuation_design_2026_05_31.md:196`). QTIP must
     **match-or-beat Q4_K_M quality at ~3 bits** (its whole pitch is best-in-
     class quality-per-bit). This is the gate that decides whether the byte-cut
     is *usable*, not just *small*.
  2. **A decode-cost premise** (on-GPU microbench): the trellis decode must stay
     **bandwidth-bound** on the M3. This is exactly where A8 killed every Q3
     kernel — sub-4-bit decode on this GPU is *compute-bound* unless the per-
     element work is cheap. A trellis spends **more ALU per element than
     f16-predec-Q3** (it *computes* the code instead of reading a pre-decoded
     scale), so it inherits A8's risk directly.
- **The honest framing (the cheaper-decode-Q3 doc's §2.2 logic, made the
  recommendation): build f16-predec-Q3 FIRST.** It is cheap, reuses shipped
  A6.5 machinery, and its §6 microbench settles "is *any* Q3-class format
  speed-viable on this GPU" with one M3 run and no Colab. **QTIP is worth its
  Colab investment ONLY if f16-predec-Q3 passes that gate** (QTIP then inherits a
  proven-BW-bound Q3-class kernel pattern to target). If f16-predec-Q3 *fails*
  (residual per-element ALU keeps Q3 compute-bound even at 128 B with the scale
  decode removed), that is **strong by-proxy evidence QTIP also fails** for the
  same reason — a trellis adds ALU in the same place — and QTIP is a near-free
  kill of bible-§2's QTIP row by proxy.

So QTIP is **alive under the Kill Protocol** (named cheap offline oracle in §5;
the surviving gather-free codec per the ledger) but it is **second in line behind
f16-predec-Q3 and double-gated** (quality THEN speed). This doc refuses to
pre-declare it a win; both gates decide.

---

## 1. Lever mapping (bible §8 axis-2 / §2 axis-2 / L1.5)

QTIP is **axis-2: stream fewer bytes** (bible §0 three-axis table; §7.0
mechanism 2 "shrink the numerator's element size"). It does *not* touch axis-1
(kernel efficiency — CLOSED for decode) or axis-3 (spec). Its place in the
canon:

- **Bible §2 axis-2 row:** "QTIP 3.0–3.25 lookup-free trellis — near-Gaussian-
  optimal sub-4-bit; lookup-free contiguous decode (no gather — Apple-friendly);
  spends surplus ALU for fewer bytes … ceiling ~3.0 bits → +30–40% (net, minus
  decode compute)." Confidence **M**. Oracle: "quantize on Colab; KL/PPL; **must
  be bandwidth-bound first**." Where: **Colab → M3.**
- **Bible §8.1 L1.5 (learned per-model codebook):** the raw form is the
  *danger lever* — "a learned codebook implies codebook *lookups* … exactly what
  makes IQ-quants slow on Apple (random gather, no hardware gather instruction).
  Only viable if … lookup-free codes (à la QTIP's bitshift trellis)." The kill
  ledger (§8.3.1 row L1.5) records the raw form **Type-1 dead** on the gather
  wall and names "gather-free learned code (QTIP bitshift-trellis / lattice):
  codes *computed* arithmetically on contiguous bits" as the surviving reframe —
  routed to "the GPU/quality lane (build at most one byte-cut codec)."
- **`reports/dead_levers.md:100–101,108–109`:** L1.4 low-rank and L1.5 codebook
  both end "The surviving byte-cut codec is **QTIP** (lookup-free bitshift
  trellis) — a real byte cut AND gather-free; advance it to the GPU/quality
  lane." This doc is that advancement, at the design layer.

**Build-at-most-one-byte-cut-codec rule (bible §8.2, L1.4 feasibility note):**
QTIP competes for the same byte budget as L1.4 (data-aware low-rank+residual,
oracle written-not-run at `tools/bench/oracle_dataaware_lowrank.py`) and as
f16-predec-Q3. The decision is **oracle-ranked**, not parallel: do not build two
deep byte-cut codecs. The recommended order (§0): f16-predec-Q3 (cheapest, local)
→ then QTIP **or** L1.4 by whichever offline oracle wins, **only if the Q3
microbench proved a Q3-class format is BW-bound on this GPU.**

---

## 2. Mechanism — QTIP trellis + incoherence preprocessing, and the Apple-GPU decode cost

QTIP (Quantization with Trellises and Incoherence Processing) has two halves: a
front-end (**incoherence preprocessing**, offline, Colab) and a codec (**trellis-
coded quantization**, whose *decode* runs in the hot loop on the M3). Both must
be understood to judge the byte-cut.

### 2.1 — Incoherence preprocessing (front-end, OFFLINE/Colab, exact-absorbed)

The problem QTIP's front-end solves is the same one that **blocks W4A8**: a few
**super-outlier weight/activation channels** wreck low-bit quantization (W4A8 is
held at 20% bit-identical precisely because of ~10 outlier channels with
max/mean ~23×, per the W4A8-activation-distribution memo). QTIP applies a
**random orthogonal / Hadamard rotation (RHT)** to the weight matrix (and the
paired activation space) so the rotated coefficients are **near-Gaussian and
incoherent** — no single coordinate carries outsized magnitude. A near-Gaussian
source is exactly what a trellis code quantizes near-optimally.

- This is the **same rotation front-end the bible flags for W4A8** (§7.3.c "RHT
  to unblock W4A8" / QuaRot) — "the rotation can be absorbed offline into
  adjacent linears (computational-invariance trick) for ~zero decode cost." The
  rotation is **mathematically exact** (orthogonal Q, QᵀQ=I, folded into the
  neighbouring linear's weights at quantization time). So incoherence
  preprocessing adds **no decode-time cost and no quality loss of its own** — it
  *enables* the trellis to hit its bit-rate at quality.
- **Where it runs:** Colab (bible §5 division of labor: "QTIP quantization +
  incoherence fine-tuning" is explicitly a Colab one-time artifact).
- **Honest note:** absorbing the rotation into adjacent linears is clean for the
  FFN/attention projection chain but interacts with RMSNorm placement and tied
  embeddings; the Qwen2.5-3B graph must be checked for where the rotation can be
  folded vs where it would need a runtime apply (a runtime rotation *would* add
  decode cost and is to be avoided). This is a Colab-quantizer design detail, not
  a runtime-kernel one.

### 2.2 — Trellis-coded quantization (the codec; decode runs on the M3)

Instead of quantizing each weight independently to a grid point (Q4_K/Q3_K) or
indexing a codebook (the dead L1.5 raw form), QTIP encodes a *sequence* of
weights as a **path through a trellis** — a finite-state machine where each state
emits a reconstruction value and transitions are constrained. The decoder walks
the trellis, and crucially the per-state reconstruction value is **computed from
the bitstream arithmetically**, not read from a table. QTIP's published decoders
("3INST", "1MAD" / bitshift-trellis families) reconstruct each value with a
**small fixed sequence of integer shifts / multiply-add / xor on contiguous
bits** — *no memory indirection*.

**Why this is the load-bearing Apple-GPU property (gather-free):**

- The Apple GPU has **no hardware gather instruction** (verified, bible §7.0
  corollary; `reports/dead_levers.md:108`). Any codec whose decode is
  `value = table[index]` with a per-element data-dependent `index` degenerates to
  **serialized random LUT reads** — exactly why IQ-quants are slow on Apple and
  why raw-codebook L1.5 is **Type-1 dead**.
- QTIP's trellis decode reads the **bitstream contiguously** (coalesced, like
  Q4_K's nibble plane) and **computes** each reconstruction value with ALU. There
  is no per-element gather. The layout is **gather-free by construction** — which
  is the single reason QTIP survives the L1.5 feasibility gate where the raw
  codebook died.

**The decode-time cost on the Apple GPU (the crux):** decode is bandwidth-bound
with **>20× compute headroom** (bible §7.0 corollary: Qwen2.5-3B decode is
~6.2 GFLOP/token vs a compute ceiling >1,500 tps and a BW ceiling ~78 tps). So
*in principle* "spend surplus ALU to save bytes" is net-positive — an expensive
codec is almost always worth it (bible §7.0: "spending compute to save bandwidth
is almost always net-positive, even an expensive codec"). **But the >20× headroom
is a per-token average, not a per-kernel guarantee** — A8 proved that a *specific*
sub-4-bit kernel (Q3_K inline 6-bit scale decode) can be **compute-bound** on
this GPU even though the token-level compute budget is enormous, because the
per-element ALU chain is long and branchy and serializes against the memory
stream. The trellis decode is the **same risk class**: it is *more* per-element
ALU than f16-predec-Q3 (which just reads a pre-decoded scale). The question is
not "is there compute headroom" (there is) — it is "is the trellis decode's
per-element ALU chain short enough to stay hidden behind the 96–104 B/block
memory stream." That is an on-GPU microbench question (§5.2), and it inherits
A8's caution directly: **do not assume the >20× headroom saves the trellis; A8
is the counterexample where it didn't for Q3.**

### 2.3 — Why QTIP beats f16-predec-Q3 *on bytes* (and why that's not enough)

f16-predec-Q3 (the recommended-first lever) cuts to 128 B/block by **predecoding
Q3's scales to f16 and repacking** — but it is still a *uniform affine* code
(2-bit weight + hmask high-bit + per-sub-block scale). QTIP goes deeper because:

- the **trellis is entropy-efficient** — it allocates code length to match the
  (post-rotation near-Gaussian) weight distribution, getting ~Gaussian-optimal
  distortion-per-bit, whereas Q3's uniform 3.4-bit grid is ~0.5+ bits/weight
  worse at equal distortion;
- there is **no scale-byte tax at all** in the deep-trellis form (the code is
  self-describing; scales fold into the trellis fit or a tiny per-block constant),
  so QTIP does not even pay the 32 B/block of f16 scales f16-predec-Q3 carries.

This is why QTIP's block is ~96–104 B (deeper than 128 B). **But bytes are not
the gate** — the gate is (1) does the trellis hit Q4_K_M *quality* at ~3 bits
(QTIP's whole claim, must be measured on *code*, not just lit PPL), and (2) does
the gather-free-but-ALU-heavy decode stay BW-bound on the M3. A smaller block
that decodes compute-bound loses to a larger block that decodes BW-bound — the
exact lesson of A8 (110-B fused Q3 lost to 144-B Q4 because the 110-B kernel was
compute-bound). QTIP must clear **both** gates, not just the byte count.

---

## 3. Byte-budget analysis (the numerator the bus must stream)

All numbers are **bytes per 256-element super-block**, the unit the decode GEMV
streams. Qwen2.5-3B dense decode reads ~1.9 GB of Q4_K_M weights/token; the
byte-cut acts proportionally (bible axis-2). QTIP bit-rates are the published
3.0–3.25 target; the block bytes are derived (bits × 256 / 8, plus a small
per-block constant for any residual scale/state seed).

| format | eff bits/wt | weight B | scale/seed B | **total B/block** | vs Q4_K-predec-f16s (A6.5) | vs f16-predec-Q3 (128) |
|---|---|---|---|---|---|---|
| Q4_K-predec-f32 (default) | 4.5 | 144 | 64 (16×f32) | **208** | +18% | +63% |
| Q4_K-predec-f16s (A6.5, shipped) | 4.5 | 144 | 32 (16×f16) | **176** | — | +38% |
| f16-predec-Q3 REPACKED (cheaper-decode-Q3 ⭐) | ~3.4 | 96 | 32 (16×f16) | **128** | −27% | — |
| **QTIP-3.25 (conservative)** | 3.25 | 104 | ~0–8 | **~104–112** | **−36 to −41%** | **−13 to −19%** |
| **QTIP-3.0 (target) ⭐** | 3.0 | 96 | ~0–8 | **~96–104** | **−41 to −45%** | **−19 to −25%** |
| QTIP-2.0 (likely breaks a 3B) | 2.0 | 64 | ~0–8 | **~64–72** | −59 to −64% | quality floor — NO-GO |

Plus the shape-constant `x` (cols×4) and `y` (rows×4), identical across all
kernels, so they wash out of any *relative* comparison (same accounting as
`q3k_bytecut_bench.rs:199–209`).

**Key reads:**
- **QTIP-3.0 at ~96–104 B is the deepest viable cut** — −41 to −45% vs the
  dominant A6.5 f16s `_pair` kernel (which is 46.6% of decode per the A4 profile),
  and a further −19 to −25% **below** f16-predec-Q3. On bytes alone, QTIP-3.0
  raises the dense tps wall toward ~99 (bible §7.0 table: 3.0 eff bits → ~104
  dense tps at ~88% bus).
- **The "~0–8 B scale/seed" is the deep-trellis advantage** — unlike every
  affine codec in the table, QTIP does not carry a 32–64 B scale plane; the code
  is self-describing. This is *why* it undercuts even f16-predec-Q3, which still
  pays 32 B of f16 scales.
- **QTIP-2.0 is listed only to be rejected** — bible §2/§6 and §7.0 both record
  that a 3B "degrades hard below ~3 bits"; 2-bit "likely breaks a 3B." It is a
  Type-1-ish quality wall on a 3B, not a real lever. The QTIP target is **3.0,
  not below.**

**Footprint note (separate from speed):** QTIP changes the *on-disk* model size
(it is a new weight encoding produced on Colab), unlike the f16-predec-Q3 runtime
repack which is built at load and leaves the Q3_K_M file unchanged. A QTIP-3.0
Qwen2.5-3B is ~1.2 GB of weights (vs ~1.9 GB Q4_K_M, ~1.7 GB Q3_K_M) — a genuine
footprint win on top of the bandwidth win. But that file only exists after the
Colab quantize+heal; this doc designs the *runtime decode* of it, gated on the
Colab quality oracle producing an acceptable file at all.

---

## 4. Decode-cost analysis (will the trellis stay bandwidth-bound on the M3?)

The byte-cut converts to a speed win **only if the kernel is BW-bound** (bible §2:
byte-cut "raises the tps wall proportionally — no % ceiling"). This is the same
premise f16-predec-Q3 must clear, and QTIP inherits A8's hard lesson directly.

### 4.1 — The compute-headroom argument (why it *could* work)

- **>20× compute headroom per token** (bible §7.0): the GPU's arithmetic
  throughput vastly exceeds what bandwidth demands, so spending ALU to save bytes
  is *usually* net-positive. A trellis decode of ~96–104 B/block, if its per-
  element ALU chain is short and unbranchy, is hidden behind the memory stream
  the same way Q4_K-predec's nibble-unpack+FMA is hidden behind its 176 B.
- **Gather-free = coalesced loads** (§2.2): the bitstream is read contiguously
  (32 contiguous bytes/simdgroup, like Q4_K's qs plane — A5's coalescing finding
  applies), so the *memory* side is clean; the only question is the *compute*
  side.
- QTIP's published decoders are deliberately cheap: "3INST" ≈ 3 integer
  instructions/value; "1MAD" ≈ one multiply-add + a mask. If the realized Metal
  hot loop is ~that cheap, it lands BW-bound and the −41 to −45% byte-cut
  converts to roughly that tps gain (minus the small ALU overhead).

### 4.2 — The honest residual (why A8 says it might NOT work)

This is the crux the design must not gloss — it is the **same failure mode that
killed every Q3 kernel** (cheaper-decode-Q3 §4.2, A8 record):

- **A8 measured that sub-4-bit decode on THIS GPU is compute-bound** when the
  per-element ALU chain is non-trivial. The Q3 inline 6-bit scale decode (a
  branch + nibble select + shift/mask + OR + −32 per element) ran the kernel at
  7–21 GB/s on a 150 GB/s machine — compute-starved, not bandwidth-starved.
- **A trellis decode is MORE per-element ALU than f16-predec-Q3**, which is
  itself the leaner reframe of the thing A8 killed. f16-predec-Q3 *removes* the
  6-bit scale decode by predecoding it; QTIP *re-adds* per-element compute (it
  computes the code arithmetically every element). So QTIP sits **on the wrong
  side** of the exact transform (predec) that made Q4_K optimal. It could easily
  land compute-bound for the **same reason A8 killed Q3** — and, unlike
  f16-predec-Q3, we would only discover that **after** a Colab quantize + a
  hand-written trellis kernel (high effort, same risk).
- **The trellis has sequential dependence A8's kernels did not.** A path through
  a state machine is inherently *serial* along the sequence (state[i] depends on
  state[i−1]). Unrolling / parallelizing the trellis decode across a simdgroup
  (so 32 lanes aren't serialized on one path) is a real kernel-design problem
  with no Q4_K analog — Q4_K nibbles are independent. If the trellis cannot be
  decoded in parallel across the reduction dim, it is **structurally worse** than
  Q3 for this GPU, not better.

### 4.3 — Why the cheapest signal is f16-predec-Q3's microbench (the by-proxy gate)

We do **not** have to build QTIP to get the first read on whether *any* Q3-class
sub-4-bit format is BW-bound on this GPU. The cheaper-decode-Q3 §6 microbench
answers exactly that with a **local M3 run, no Colab**:

- If **f16-predec-Q3 PASSES** (128-B kernel runs ~55%+ of peak, beats Q4_K-predec
  µs/call): a Q3-class format *can* be BW-bound on the M3 once the worst per-
  element compute is removed. QTIP then has a **proven target pattern** and its
  remaining risk is "is the *trellis's* extra ALU (vs f16-predec-Q3's pre-decoded
  scale read) small enough to stay BW-bound" — a narrower, now-worth-Colab
  question.
- If **f16-predec-Q3 FAILS** (128-B kernel still ~30% of peak, compute-bound even
  with the scale decode hoisted out): the residual per-element ALU (hmask gather
  + index math) keeps Q3-class formats compute-bound on this GPU. A trellis adds
  ALU **in the same place**, so this is **strong by-proxy evidence QTIP also
  fails** — a near-free kill of QTIP without writing a trellis kernel or running
  Colab. (The cheaper-decode-Q3 doc makes this exact by-proxy argument, §2.2:
  "If f16-predec-Q3 *fails* the gate … that is strong evidence QTIP also fails for
  the same reason — a near-free kill of P4 by proxy.")

This is why QTIP is **second in line**: f16-predec-Q3's local microbench is the
cheapest possible forecast of QTIP's speed gate, and it costs one M3 bench run.

---

## 5. Cheap offline oracle (must clear BEFORE any kernel body)

Per CLAUDE.md / bible discipline (oracle-before-kernel) QTIP has **two** offline
gates; both are offline (Colab/NumPy-class), neither needs the M3 kernel. The
quality oracle is **first** because it decides whether the byte-cut is *usable*;
the speed forecast (§4.3) is the f16-predec-Q3 microbench already designed.

### 5.0 — Free by-proxy speed forecast (do this FIRST, no new code, local)

Run the **existing** `q3k_bytecut_bench` clean and read the f32-predec-Q3 GB/s
(cheaper-decode-Q3 §6.0). This is the cheapest signal on whether *any* sub-4-bit
Q3-class kernel is BW-bound on this GPU. ~50% of peak → Q3-class can be BW-bound,
QTIP's speed gate is plausibly clearable; ~30% → the residual per-element ALU is
the wall, **QTIP's trellis adds ALU in the same place → strong NO-GO-by-proxy.**
Cost: one bench run. (This is `clean_room_batch.sh` section A — see deliverable 2.)

### 5.1 — Quality oracle (the decisive one) — `oracle_qtip_quality.py` (NEW, Colab/NumPy)

- **What it measures:** reconstruction + functional quality of a QTIP-3.0 (and
  3.25) encoding of Qwen2.5-3B vs Q4_K_M, on a **code-representative** corpus —
  the same target the other `oracle_*.py` scripts use (`oracle_lowrank_codebook.py`,
  `oracle_dataaware_lowrank.py` both target "vs Q4_K-recon" / KL vs Q4_K_M).
- **Metric class (name it explicitly):** the **logit-cosine / token-divergence**
  family the W4A8 quality work uses (`reports/dead_levers.md:118` "a logit-
  streaming quality metric — bit-identical is too strict; cosine/KL on logits may
  show acceptable quality"; `plans/stateful_moat_continuation_design_2026_05_31.md:196`
  "logit-cosine / token-divergence measurement, the same metric class the W4A8
  quality work needs"). Concretely, per layer / per token on the code corpus:
  1. **Reconstruction RMSE** of QTIP-3.0 weights vs the f16 originals, and vs
     Q4_K_M weights — the cheap first cut (a NumPy afternoon; mirrors
     `oracle_lowrank_codebook.py`'s residual-std measure). **GO floor:** QTIP-3.0
     RMSE ≤ Q4_K_M RMSE *at fewer bytes* (96–104 vs 144 weight B). If QTIP-3.0
     reconstructs *worse* than Q4_K_M despite its theoretical edge, the
     incoherence/trellis fit failed → NO-GO before any forward pass.
  2. **Logit cosine** between the QTIP-3.0 model and the f16 (and Q4_K_M) model
     on held-out code tokens — the W4A8 metric (W4A8 production held at logit-
     cosine 0.9992). **GO floor:** QTIP-3.0 logit-cosine ≥ Q4_K_M's logit-cosine
     vs f16 (QTIP must be *at least as good* as the incumbent it replaces).
  3. **Token-divergence / KL** of the next-token distribution vs f16 on the code
     corpus, and **greedy argmax-agreement rate** (what fraction of positions
     keep the same top-1 token as Q4_K_M). **GO floor:** KL ≤ Q4_K_M's KL and
     argmax-agreement ≥ Q4_K_M's, on *code*.
- **The honest precedent that makes this gate strict:** the local mixed-precision
  oracle found **imatrix-Q3 still +32% PPL for −18% bytes when requant-from-Q4**
  (`bible_execution_2026_05_30.md`; MEMORY "byte-cut needs AWQ-from-f16"). QTIP
  must beat that — and it can **only** be fairly tested **AWQ/QTIP-from-f16 on
  Colab**, not requant-from-Q4_K (requant-from-Q4 is pessimistic and would
  unfairly sink QTIP; see §6 kill respect). So the oracle's input is the **f16
  Qwen2.5-3B**, quantized to QTIP on Colab — not the shipped Q4_K_M GGUF.
- **Where:** Colab (the QTIP quantizer + incoherence fit live there, bible §5).
  The *analysis* (RMSE/cosine/KL) is NumPy and can run locally on exported
  logits, but the *quantize* step is Colab-GPU. This is why QTIP cannot be
  stood up read-only/local — unlike the f16-predec-Q3 oracle which is a pure M3
  microbench.
- **Kill rule:** if QTIP-3.0 does **not** match-or-beat Q4_K_M on the logit-
  cosine / KL / argmax triple on *code* at ~3 bits, the byte-cut is **not usable**
  and QTIP stays dead (a quality Type-1: "a 3B at ~3 bits via this codec is not
  good enough on code" is a measured property, not an effort question). No
  resurrection without a *different* named quality oracle.

### 5.2 — Decode-cost oracle (the on-GPU gate, AFTER quality clears) — extend `q3k_bytecut_bench.rs`

- **Only run if 5.1 passes AND f16-predec-Q3's §6 microbench passed** (else the
  speed premise is already settled NO-GO by proxy — §4.3). A trellis kernel is
  expensive to write; do not write it to discover quality or the Q3-class BW
  premise already failed.
- **What it measures:** a hand-written `gemm_qtip_trellis_v1` decode GEMV, benched
  as a new line in `q3k_bytecut_bench.rs` on the same 3 Qwen shapes (attn-square
  2048×2048, ffn-up 11008×2048, ffn-down 2048×11008), same WARMUP=30/ITERS=200,
  same TCB commit-and-wait = one decode-path dispatch. **Report achieved GB/s at
  ~96–104 B/block** and µs/call vs Q4_K-predec.
- **GO floor:** the trellis kernel runs **~55%+ of peak BW** (BW-bound, like
  Q4_K-predec) **and** beats Q4_K-predec µs/call on the dominant `_pair` shapes.
- **NO-GO (record Type-1):** the trellis kernel runs well under peak (~30%) → it
  is compute-bound on the trellis ALU, same death as A8's Q3 kernels — the
  gather-free-but-serial trellis decode is irreducibly compute-bound on this GPU.

---

## 6. Build plan (against the REAL files — for an attended Colab+kernel session)

**Nothing here is built in this session.** This is the plan, gated on §5. It is
strictly larger than the f16-predec-Q3 build (which reuses shipped A6.5
machinery); QTIP is **all-new** (Colab quantizer + a novel Metal kernel), which
is the §0 reason it is second in line.

### 6.1 — Colab: QTIP quantizer + incoherence fit (the artifact)
- **Where:** Colab (bible §5: "QTIP quantization + incoherence fine-tuning").
- **Input:** the **f16** Qwen2.5-3B (NOT the Q4_K_M GGUF — §5.1 / §6 kill respect:
  the byte-cut prize needs AWQ/QTIP-from-f16, requant-from-Q4 is pessimistic).
- **Output:** a QTIP-3.0 weight file + a tiny per-block constant table (if any) +
  the absorbed-rotation linears. This is the new on-disk encoding the M3 serves.
- **Gate:** §5.1 quality oracle must pass on this artifact before any kernel.

### 6.2 — Loader (Rust) — QTIP block loader + repack
- **File:** `crates/dismantle-core/src/quant/mod.rs` (the home of
  `dequant_q3_k_into:566`, `predecode_q3_k_scale_table:643`, and the q4k_fast
  repack) and/or `crates/dismantle-core/src/kernels/mod.rs`.
- **Body:** parse the QTIP block format into the GPU-resident bitstream buffer
  (contiguous, gather-free layout — the load-bearing §2.2 property). Mirror the
  q4k_fast repack's load-time, replace-not-augment buffer discipline (the
  cheaper-decode-Q3 §5.2 RSS note applies verbatim: a ~1.2 GB QTIP weight buffer
  must **replace** the served buffer, not co-reside with an f16/Q4 mmap, or it
  risks the CLAUDE.md >5 GB RSS sentinel).
- **No scale sidecar** (the deep-trellis form is self-describing) — unlike the
  Q3/Q4 predec path's separate f32/f16 scale buffer. This is a divergence from
  the predec loader pattern.

### 6.3 — Kernel (Metal) — `gemm_qtip_trellis_v1` (NEW, novel hot loop)
- **File:** `crates/dismantle-core/shaders/quant.metal`.
- **Template to study (NOT clone):** the Q4_K predec decode GEMV family —
  `gemm_q4_k_v4_predec` (`quant.metal:1957`), `gemm_q4_k_v4_predec_2r`
  (`:2153`, "run ~56% peak. 16 rows/TG (8 simdgroups × 2 rows)"), and the f16s
  variants `gemm_q4_k_v4_predec_pair_f16s` (`:2085`), `gemm_q4_k_v4_predec_2r_f16s`
  (`:2226`). These establish the **outer GEMV structure** (per-row reduction over
  blocks, simdgroup tiling, 2-row ILP, the FMA accumulate, the TCB-pinned
  dispatch). QTIP reuses *that outer structure* and replaces *only* the inner
  per-block weight decode — the nibble-unpack+scale becomes the trellis walk.
- **The novel part (no Q4_K analog):** the inner loop decodes a *path through the
  trellis* — read contiguous bits, run the 3INST/1MAD arithmetic to produce each
  reconstruction value, FMA into the accumulator. **Two hard design problems with
  no Q4_K counterpart:**
  1. **Parallelizing the serial trellis across a simdgroup** (§4.2): state[i]
     depends on state[i−1]; the kernel must either decode independent sub-blocks
     per lane (QTIP's block structure allows this if encoded that way) or accept
     serialization. **The Colab encoder must emit lane-independent sub-blocks** so
     32 simdgroup lanes decode 32 independent trellis segments in parallel — this
     is a **co-design constraint between the Colab encoder (6.1) and the kernel**,
     not a pure kernel choice.
  2. **Keeping the per-element ALU chain short** so it stays hidden behind the
     96–104 B memory stream (§4.2) — the whole BW-bound premise. Prefer the 1MAD
     decoder over heavier trellis families.
- **Register the kernel name** in `metal/mod.rs` (the match arm alongside the
  Q4_K/Q3_K predec kernels, `metal/mod.rs:442–453`).
- **Behind an opt-in flag** `DISMANTLE_QWEN_QTIP` (parallel to
  `DISMANTLE_QWEN_Q4K_PREDEC` / the proposed `DISMANTLE_QWEN_Q3_PREDEC_F16SCALES`),
  default-off, `OnceLock` pattern.

### 6.4 — Wrapper (Rust) — `gemv_qtip_trellis_pinned_tcb`
- **File:** `crates/dismantle-core/src/kernels/mod.rs`.
- **Template:** `gemv_q4_k_v4_predec_pinned_tcb` / `gemv_q3_k_v4_predec_pinned_tcb`
  (`kernels/mod.rs:1321` is the Q3 wrapper the cheaper-decode-Q3 doc clones).
- **Changes:** the `expected_bytes` guard uses the **QTIP block stride** (~96–104,
  not 110/144); **no scales_buf argument** (self-describing code); grid/TG mirror
  the Q4_K predec 2r/pair grids.

### 6.5 — Dense-path wiring + parity gate (LAST, after the microbench passes)
- A full QTIP dense Qwen path is the large piece; **wire it only after §5.2's
  microbench proves the trellis kernel is BW-bound** (don't build the big thing to
  discover the kernel loses — same discipline as cheaper-decode-Q3 §5.5).
- **Parity gate** (`tests/qtip_trellis_parity.rs`, NEW): QTIP is a **quality
  lever, NOT bit-identical** to Q4_K_M (different codec entirely). Gate the
  *kernel* against a **CPU reference QTIP decode** at `atol=1e-3 fp16` (CLAUDE.md
  verification rule) — i.e. the Metal trellis decode must match a NumPy/CPU
  trellis decode of the same bitstream. The *model-quality* gate is §5.1's
  logit-cosine/KL, run upstream on Colab; the kernel parity gate only proves the
  Metal decode is faithful to the encoder, not that the encoder is good enough
  (that is §5.1's job).

---

## 7. Apple-Silicon feasibility (gather-free is the binding constraint)

- **No hardware gather on the Apple GPU** (verified, bible §7.0 corollary;
  `reports/dead_levers.md:108`). This is the constraint that **killed the L1.5
  raw codebook (Type-1)** and that **QTIP is specifically designed to satisfy**:
  its codes are *computed arithmetically on contiguous bits*, not read from a
  table. **The layout is gather-free by construction** (§2.2) — this is the *only*
  reason QTIP is alive where the raw codebook is dead. A QTIP variant that
  reverted to a LUT (a "hybrid codebook+trellis") would **re-die on the gather
  wall** — keep the decode purely computed.
- **Coalesced contiguous loads:** the bitstream reads 32 contiguous bytes/
  simdgroup like Q4_K's qs plane (A5's coalescing finding applies) — the memory
  side is clean.
- **The serial-trellis hazard (§4.2, §6.3):** the one genuinely Apple-hostile
  risk is **not** gather — it is the trellis's **sequential state dependence**. If
  the encoder does not emit lane-independent sub-blocks, the decode serializes
  across the simdgroup and the kernel underfills the 32 lanes (worse than Q4_K's
  independent nibbles). The feasibility mitigation is a **co-design constraint on
  the Colab encoder** (emit lane-parallel sub-blocks) — flagged in §6.3 as the
  hard part. This is the QTIP-specific feasibility gate, the analog of "kill the
  raw codebook at the feasibility gate before quality."
- **Compute headroom is sufficient in principle** (>20× per token, bible §7.0)
  but **A8 is the standing counterexample** that a specific sub-4-bit kernel can
  still be compute-bound — feasibility is "the per-element ALU + the lane-
  parallelism," settled by §5.2's microbench, not assumed from the headroom.

---

## 8. Kill Protocol classification (CLAUDE.md / bible §8.3.1)

QTIP is the **named surviving reframe of the L1.5 Type-1 kill** (the raw learned
codebook). Recording per the protocol:

- **The L1.5 raw-codebook kill was Type-1** (gather form): "raw k-means index→value
  LUT = per-element random gather; no HW gather on Apple Silicon → killed at the
  feasibility gate before quality. Hardware wall, basis-independent."
  (`bible §8.3.1` L1.5 row; `dead_levers.md:108`.)
- **The reframe (named, per the protocol):** "gather-free learned code (QTIP
  bitshift-trellis / lattice): codes *computed* arithmetically on contiguous bits,
  no LUT read." (`bible §8.3.1` L1.5 row.) **QTIP IS that reframe**, designed in
  full here.
- **The reframe is ALIVE — but conditionally**, with **two named cheap oracles**:
  (1) the §5.0 free by-proxy speed forecast (re-run the existing
  `q3k_bytecut_bench`, read f32-predec-Q3 GB/s) + the f16-predec-Q3 §6 microbench;
  (2) the §5.1 quality oracle (logit-cosine/KL vs Q4_K_M on code, QTIP-from-f16 on
  Colab). The bible §8.3.1 itself flags the cheapest next check: "trellis
  reconstruction-MSE vs Q4_K at matched bits, **left for attended QTIP work (a
  wrong trellis sim is worse than none)**" — which is exactly §5.1's first cut.
- **What makes it Type-1 dead (for real):**
  - **Quality Type-1:** if §5.1 shows QTIP-3.0 cannot match Q4_K_M on
    logit-cosine/KL/argmax on *code* at ~3 bits (a 3B at 3 bits via this codec is
    not good enough on the product workload) — a measured property of the
    model+codec, not effort.
  - **Speed Type-1:** if §5.2 (or its §4.3 by-proxy forecast) shows the trellis
    kernel is irreducibly compute-bound on this GPU (~30% of peak) — the
    gather-free-but-serial-ALU trellis is too heavy per element, the same death
    A8 recorded for Q3, in the same place.
- **No resurrection on vibes:** if either oracle says NO-GO, QTIP stays dead
  unless a *new* named oracle appears (e.g. a different trellis decoder family
  with a measured cheaper ALU chain, or a trained-for/L5.1 model that makes the
  3-bit quality clear). **Never re-test a recorded Type-1.**
- **Build-at-most-one-byte-cut-codec:** QTIP competes with L1.4 (data-aware
  low-rank, oracle written-not-run) and f16-predec-Q3 for the single byte-cut
  slot (bible §8.2). The recommended resolution (§0): f16-predec-Q3 first
  (cheapest, local); then QTIP **or** L1.4 by whichever offline oracle wins,
  **only if** the Q3 microbench proved a Q3-class format is BW-bound on this GPU.

**Kills QTIP must respect (do not re-fight these):**
1. **Decode micro-opt is CLOSED for decode** (bible §3.0): QTIP is **not** a
   kernel-efficiency lever — it is axis-2 (fewer bytes). It does not reopen the
   A5/A6/A7/A10 decode-kernel-microopt kills; it changes the *numerator* (bytes),
   not the *denominator* (kernel %-of-peak). (Stage-2 simdgroup-MMA stays a
   separate LIVE *prefill* lever — also not this doc.)
2. **The byte-cut prize needs AWQ/QTIP-from-f16, NOT requant-from-Q4_K**
   (`dead_levers.md:118`; `bible_execution_2026_05_30.md`; MEMORY): the §5.1
   quality oracle MUST quantize from the **f16** Qwen, not the Q4_K_M GGUF —
   requant-from-Q4 is pessimistic (imatrix-Q3-from-Q4 was +32% PPL for −18%
   bytes) and would unfairly sink QTIP. Test it from f16 on Colab or do not test
   it.
3. **A8 Q3-class compute-boundedness** (cheaper-decode-Q3 §1, A8 record): sub-4-
   bit decode on this GPU is compute-bound unless the per-element work is cheap.
   QTIP's trellis adds ALU vs f16-predec-Q3 → it inherits this risk and is
   forecast by the f16-predec-Q3 microbench (§4.3). Do not assume the >20×
   compute headroom saves it — A8 is the counterexample.

---

## 9. Honest bottom line

**QTIP is the deepest live byte-cut codec and the named survivor of the L1.5
gather-wall kill** — ~3.0 bits, ~96–104 B/block (−41 to −45% vs the dominant A6.5
kernel, −19 to −25% deeper than f16-predec-Q3), gather-free by construction, and
the only learned-code form that clears the Apple no-hardware-gather feasibility
gate. On bytes alone it raises the dense tps wall toward ~99 (bible §7.0).

**But it is double-gated and second in line, and this doc refuses to pretend
otherwise.** (1) Its **quality** must match-or-beat Q4_K_M at 3 bits on *code*
under the logit-cosine/KL/argmax metric class — measured QTIP-from-f16 on Colab,
not requant-from-Q4 — and a 3B "degrades hard below ~3 bits," so the margin is
thin. (2) Its **decode** must stay bandwidth-bound on the M3 despite a trellis
that spends *more* per-element ALU than f16-predec-Q3 and carries a *serial*
state dependence with no Q4_K analog — the exact compute-boundedness A8 measured
for every Q3 kernel. QTIP sits on the wrong side of the predec transform that
made Q4_K optimal, and the >20× compute headroom is a per-token average A8 already
proved does not guarantee a per-kernel win.

**The cheapest next actions, in order:** (a) the §5.0 free by-proxy read — re-run
the existing `q3k_bytecut_bench` clean and look at f32-predec-Q3's GB/s (one M3
run; ~50% of peak = a Q3-class format can be BW-bound, ~30% = NO-GO-by-proxy for
QTIP too); (b) build f16-predec-Q3 first (cheap, local, reuses A6.5) and run its
§6 microbench — if it passes, QTIP has a proven target and earns its Colab
quality oracle (§5.1); if it fails, QTIP is killed by proxy without a single line
of trellis code. **Do the local Q3 microbench first; commit Colab to QTIP only
after it says a Q3-class format is BW-bound AND the QTIP-from-f16 quality oracle
clears Q4_K_M on code.**

---

### Appendix — file/line index (for the attended build session)
- Q4_K predec decode GEMV family (the outer-GEMV template to study, NOT clone):
  `shaders/quant.metal` — `gemm_q4_k_v4_predec:1957`, `gemm_q4_k_v4_predec_pair:2014`,
  `gemm_q4_k_v4_predec_pair_f16s:2085`, `gemm_q4_k_v4_predec_2r:2153`,
  `gemm_q4_k_v4_predec_2r_f16s:2226`.
- Q3_K decode + predec (the cheaper-decode-Q3 lever this contrasts):
  `gemm_q3_k_fused_v2:349`, `gemm_q3_k_v4_predec:490` (quant.metal);
  `predecode_q3_k_scale_table:643`, `dequant_q3_k_into:566` (quant/mod.rs).
- Wrapper template to clone: `gemv_q4_k_v4_predec_pinned_tcb` /
  `gemv_q3_k_v4_predec_pinned_tcb:1321` (kernels/mod.rs).
- Kernel-name registry: `metal/mod.rs:442–453`.
- Microbench to extend (the §5.2 speed gate + §5.0 free read): 
  `tests/q3k_bytecut_bench.rs` (test `q3k_bytecut_gemv_bench`, `#[ignore]`, run
  `--ignored --nocapture`; verdicts (a)/(b)/(c) `:230–259`).
- The by-proxy speed forecast + the full clean-room procedure:
  `plans/cheaper_decode_q3_design_2026_05_31.md` §6.0; `tools/bench/clean_room_batch.sh`.
- Quality-oracle metric-class precedent (logit-cosine / KL, AWQ-from-f16):
  `reports/dead_levers.md:118`; `plans/bible_execution_2026_05_30.md:46,58`;
  `plans/stateful_moat_continuation_design_2026_05_31.md:196`; W4A8 held at
  logit-cosine 0.9992 (MEMORY `w4a8_production_held`).
- Existing offline oracle scripts to mirror for `oracle_qtip_quality.py`:
  `tools/bench/oracle_lowrank_codebook.py`, `tools/bench/oracle_dataaware_lowrank.py`
  (both target reconstruction/KL vs Q4_K, with f16-from-source caveat).
- L1.5 kill record + QTIP-survivor pointer: `plans/throughput_bible_2026_05_30.md`
  §8.1 L1.5, §8.3.1 (L1.5 row); `reports/dead_levers.md:100–101,105–109`.
- Bible axis-2 QTIP row + physical floor: `throughput_bible_2026_05_30.md` §2
  (axis-2 table), §7.0 (the 3.0-bit → ~104 dense-tps floor table).
