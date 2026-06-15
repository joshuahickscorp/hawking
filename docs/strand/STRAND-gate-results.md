# STRAND gate results (2026-06-09)

> **⚠️ PARTIALLY SUPERSEDED (same day) — see `will.md` §4/§6.** The B1/C1 "sub-2-bit DEAD, do
> not re-open" verdict below was measured on **rel-RMS (the wrong metric) at low state count**
> and is **overturned**: re-screened on real PPL with high state count (`--l 12`) + the pre-RHT
> outlier channel, 2-bit moved **4194 → 210 → 80.7** (0.5B). The A1 speed verdict (decode is a
> bit-exact parallel window but ALU-bound on GPU) stands. Kept for the record of *why* the
> original framing failed — the lesson is will.md ideology §5.5/§5.6.

The decisive gates from `docs/STRAND-master-plan.md` § "Sequencing", run to a measured verdict.
Both gates the master plan flagged as make-or-break — **A1** (the SPEED revival) and **B1/C1**
(the QUALITY+DENSITY moat) — came back **DEAD**. The honest boundaries and the paths they redirect
to are below.

| gate | question | verdict | determinism | proxy | unlocks |
|---|---|---|---|---|---|
| **A1** windowed decode | is the GPU decode an embarrassingly-parallel per-weight windowed read? | **DEAD** (still compute-bound) | PASS (bit-identical) | direct (real GPU %peak) | A2 bitslicing, not the per-weight kernel |
| **B1 / C1** high-dim vector trellis | does a frozen-LUT high-dim trellis make sub-2-bit usable + close the density gap? | **DEAD** (plateaus by d≈3) | PASS (bit-identical) | rel-RMS, **not yet 7B PPL** | B3 training frontier; **no 7B PPL run warranted** |

The master plan's "honest priors going in" held on both: A1 was high-upside-but-small and it
delivered a clean negative; B1 was the make-or-break for the quality moat and it says the frozen
high-dim trellis is **not** the lever — the sub-2-bit gap is a bit-budget problem that points to
training (B3), the flagged frontier.

---

## GATE A1 — per-weight windowed decode (SPEED)

**Question (master plan A1):** STRAND's decode state is `state[i] = (state[i-1] << k | sym[i]) & mask`,
a shift register of width `l` bits whose closed form is *the `l`-bit window of the symbol stream ending
at the top of symbol `i`*. That is an **independent windowed read per weight**, not a true Viterbi
recurrence (that's the ENCODE). So the GPU "serial wall" we measured (per-block-serial kernel, 8–25 %
of peak) was plausibly a *kernel artifact*, and a per-weight-windowed decode should be embarrassingly
parallel and clear the bandwidth bar — solving GPU speed *simply*, no tropical scan.

### Verdict: **DEAD — the decode is a real windowed read, but per-weight parallelism does NOT clear the bandwidth bar. Still COMPUTE/ISSUE-bound.**

Two findings, and only the first is a "win":

1. **The window is real and bit-exact (the math the gate set out to prove).** `decode_windowed`
   (`crates/strand-decode-kernel/src/windowed.rs`) computes each weight's trellis state
   **independently** from an `L`-bit windowed read — no incremental carry from the previous weight —
   and is **BIT-IDENTICAL to `decode_lean`** over the full sweep: `k ∈ {2,3,4}`, tail-biting on/off,
   affine-min on/off, 96 seeds × structural edges (short final block, sub-block tail, u32-unaligned
   total) = **1152 cases**, plus a pinned lib unit test. No residual serial dependency survives: the
   block's `init_state` (or the tail-biting-recovered state) supplies the high bits during the
   `ceil(L/k)`-weight warm-up, after which the window is self-contained. **Full per-weight parallelism
   is mathematically available** — the master plan's structural read was correct.

2. **But it does not convert to bandwidth.** Measured on Apple M3 Pro (peak ~125 GB/s), 7B GEMV shapes,
   same upload-once + timed-loop harness/denominator as `gate-bench`:

   | shape | per-weight (per-symbol) | per-weight (single-read) | shipped per-block (this session) |
   |---|---|---|---|
   | attn_o 3584² | 6 % | 8 % | 8 % |
   | ffn_up 3584×18944 | 7 % | 10 % | 23 % |
   | ffn_down 18944×3584 | 7 % | 9 % | 10 % |

   Even the optimized single-accumulator-read variant (which cuts per-weight word loads from
   ~2·`ceil(L/k)` to ~2) tops out at **8–10 % of peak — far under the ~60 % bandwidth-bound bar** — and
   does **not** beat the shipped per-block-serial kernel. On `ffn_up` (bpr=74, where the per-block kernel
   already has good occupancy) per-weight is markedly *worse* (10 % vs 23 %).

**Why:** trading the serial chain's amortized ~1 word-load-per-11-symbols for full occupancy costs each
thread a `ceil(L/k)=3×` redundant-read + windowed-state inner loop. The per-weight op count rises
proportionally with the added occupancy, so the kernel stays **decode-ALU/issue-bound, not
bandwidth-bound** — full occupancy doesn't help when each lane does proportionally more work. This is the
same shape as dismantle's dead Q3_K kernel: byte savings that don't convert because the kernel is
compute-bound, exactly the trap `docs/STRAND-metal-decode-gate.md` warned about.

### Determinism status: **PASS.**
Every new decode path is asserted bit-identical to `decode_lean` *before* any timing — CPU twin over
1152 cases, and on the real GPU the decoded Q12 from both kernel variants is bit-identical to
`decode_lean` via a one-hot-`x` probe (STEP 2a). The gate refuses to report bandwidth if any path drifts;
none did. No shipped behavior touched (`decode_lean`, `decode_q12_fast`, `gemv.rs`, the shipped shader
all unchanged).

### Proxy honesty
This gate is **direct, not a proxy** — it measures real GPU %peak on real 7B shapes with the production
harness. There is nothing further to confirm; the windowed-kernel speed thesis is settled negative.

### Next action this unlocks (per master plan Channel A)

The master plan's A1 branch is explicit: *"If bandwidth-bound, the GPU speed path is solved simply…
Fallback: only if a real residual dependency survives, the tropical-semiring route applies."* We are in a
**third** case the plan didn't fully separate: **no residual dependency (parallelism is free), yet still
compute-bound** because the per-weight decode op-count itself is the wall. So:

1. **DO NOT ship the per-weight windowed kernel** and DO NOT re-run the Metal gate (A4) on it — it loses
   to the kernel already shipped. The simple GPU-speed revival is **off the table**.
2. **A2 (bitslicing) is now the live SPEED path, and its bar is sharper.** A1 proved the bottleneck is
   **decode ALU/issue throughput per weight**, not memory and not a serial dependency. Bitslicing (32-stream
   pure-integer batch decode, the V100 21 Gbps precedent) attacks exactly that — it amortizes the symbol-unpack
   and state-advance ALU work across 32 lanes instead of paying it per weight. Build + bench it against
   **both** the shipped per-block kernel **and** the CPU block-parallel 6× baseline (4.5 Gw/s). Gate: does
   per-lane issue pressure drop enough to become bandwidth-bound?
3. **If A2 also stays compute-bound, the honest boundary is "decode is ALU-bound, ship the CPU
   block-parallel 6× as the deploy path"** — which is already the measured CPU win and is consistent with
   `docs/STRAND-cpu-deploy.md`. The tropical/parallel-scan route (A3) does **not** apply here: A1 ruled out
   a residual serial dependency, so there is no serial chain for a scan to break — the cost is raw per-weight
   ALU, which a scan would not reduce.

**Ranked SPEED next actions:** (1) A2 bitslicing — build + bench vs shipped-GPU and CPU-6×; (2) if A2
is compute-bound too, fall back to CPU block-parallel as the shipped deploy path and close the GPU speed
channel as "ALU-bound, not bandwidth-bound."

---

## GATE B1 / C1 — higher-dimensional frozen vector trellis (QUALITY + DENSITY)

**Question (master plan B1/C1):** TCQ cost is `O(2^L·T)` — *linear in dimension* — so STRAND can push
`vec_dim` far past the d=2/3 that collapsed at sub-2-bit, into d≥4–8 where QTIP reports the
space-filling/shaping gap closes >3×. With the codebook **learned but FROZEN** to the integer Q12 LUT,
decode stays bit-deterministic. *Is the gap-closing realizable with a frozen integer LUT, or only with
float?* The single most valuable quality experiment, and simultaneously C1 (the density shaping gap).

### Verdict: **DEAD — higher dimension PLATEAUS by d≈3 and does NOT make sub-2-bit usable. The residual gap is the bit budget, not the codebook.**

With the codebook learned (deterministic Lloyd-Max, 40–50 iters) and frozen to the Q12 LUT, rel-RMS
improves only marginally with `d` and **saturates by d≈3–4**, far from a usable quantizer. Headline:
N=262144, L=12 pinned (2^L centroids held constant across the d-sweep so the book is never starved),
numbers stable across L=10/L=12 and Gaussian/heavy-tailed (Student-t(4)) weights.

#### rel-RMS (%) vs d at iso-bpw — the gate

| payload bpw | d=1 | d=2 | d=3 | d=4 | d=6 | d=8 |
|---|---|---|---|---|---|---|
| 1.00 (gauss) | 57.6 | 55.2 | 53.9 | 53.7 | 53.2 | — |
| 1.00 (heavy) | 58.2 | 54.5 | 52.9 | 52.3 | 52.3 | — |
| 1.50 (gauss) | — | 39.1 | — | 38.7 | — | — |
| 2.00 (gauss) | 28.3 | 27.7 | 27.8 | — | — | — |
| 2.00 (heavy) | 31.2 | 29.5 | 28.3 | — | — | — |
| 0.75 (gauss) | — | — | — | — | — | 61.9 |

- **Plateau is unambiguous.** At 1.0 bpw the entire d=1→6 sweep moves only ~4 pp (57.6→53.2 %) and flattens
  by d=4; at 2.0 bpw d=1/2/3 are within ~0.6 pp (≈28 %) — **d3 is even slightly worse than d2**; at 1.5 bpw
  d2≈d4. No "materially falls" trend — it asymptotes.
- **Magnitudes stay catastrophic.** ~53 % rel-RMS at 1 bpw, ~28 % at 2 bpw — both far from a usable
  quantizer, at *every* d. (For scale: the shipped 4-bit STRAND that beats Q4_K sits at ~7.8 % rel-RMS.)

#### Three controls rule out "it's just an artifact"
1. **The book is never starved.** Pinning L holds centroid count constant across the d-sweep
   (vec/centroid 8–256 everywhere). The earlier non-monotonic d6/d8 spikes in an un-pinned smoke run were
   pure undertraining and **vanish** once books are well-fit. The plateau is the genuine space-filling limit.
2. **The codebook lever is real — and is what's exhausted.** The `learn-gain` column shows the unlearned
   broadcast-Gaussian q1 LUT collapses to 71–93 %; learning recovers **+18 to +54 pp** (e.g. d2/k4 @2bpw:
   71.0 %→27.7 %). So the determinism-compatible quality lever does *real, large* work — and after spending
   it fully, the floor is still ~28 %/~53 %. The residual is bit budget, not book.
3. **Holds for standard-Gaussian and heavy-tailed weights, at L=10 and L=12.** Not a distribution or
   centroid-count artifact.

#### Density (C1) — what the vector trellis reaches
- d8 reaches **0.75 bpw payload** (floor **0.094 B/w**) — the sub-1-bit regime only the vector trellis can
  hit — but at ~61 % rel-RMS it is unusable. `total_bpw` = floor + ~0.35 bpw side info (super-scale + 8×6-bit
  sub-scales + init_state + len), e.g. d8 → **1.11 total bpw**.
- **Structural ceiling surfaced:** `MAX_K=4` forces high-`d` into low bpw (payload = `k/d`); reaching ≥1.0
  bpw at d≥8 is impossible without raising `MAX_K`. So the densest configs are also the worst-quality — there
  is no high-d sweet spot hiding above the cap.
- This directly answers C1's open question: the ~0.2–0.35 bpw space-filling gap **is** recoverable in
  principle, but the recovered shaping gain (the ~4 pp at 1 bpw) is swamped by the bit-budget floor — it does
  not move sub-2-bit into usable territory. The C1 lever is real but immaterial at this budget.

#### Practicality (for the record)
High-d encode is practical: L=12 LUT (4096 centroids) train ~11–62 s/config at 262k weights (one-time
offline), encode 4–13 s; cost is O(2^L·iters·n), linear in d (d8 is the *fastest*). Not a blocker — the
result is a quality ceiling, not a cost ceiling.

### Determinism status: **PASS — 28/28 configs, both L values.**
`decode_lean_with_lut == decode_tensor_fixed_with_lut_vec == decode_tensor_fixed_with_lut` byte-identical
for the learned frozen LUT on all 28 configs. No drift; nothing rejected. Shipped `strand-quant` suite
**78/78 green**. No shipped decode/encode behavior edited — the bin only calls public APIs
(`train_state_vector_lut`, `encode_tensor_with_lut`, the `decode_*_with_lut` family).

### Proxy honesty — **this is the one place a number is still owed**
The gate metric is **rel-RMS, a proxy for PPL**, not the 7B perplexity itself. rel-RMS is a *sound
screening* proxy: it is the same weighted NMSE that tracked the prior 4-bit/3-bit STRAND-vs-Q4_K ranking,
and a quantizer at ~28–53 % rel-RMS cannot produce usable perplexity (the shipped competitive rung is
~7.8 %). So the **screen is decisive in the negative** — a 7B PPL run would only confirm "unusable," not
overturn it. **The plateau verdict does not need the 7B run; a hypothetical *pass* would have.**

### Next action this unlocks (per master plan Channels B & C)

The master plan B1 branch: *"if B1 closes the gap with a frozen LUT, usable 2-bit/1.5-bit deterministic —
the bleeding-edge moat. If not, the honest answer is 'format+runtime ready, quality is B3/training.'"* We
are in the **"if not"** branch:

1. **DO NOT launch a 7B-PPL confirmation run on the vector trellis.** rel-RMS already settles it negative
   (proxy honesty above); spending the 7B run here would confirm "unusable," not change the verdict. The
   single-most-valuable quality experiment the plan named has been run and answered: **the frozen high-dim
   trellis is not the sub-2-bit moat.**
2. **The honest boundary, stated plainly:** STRAND's format + integer-deterministic runtime are ready and
   the determinism-compatible quality lever (learned frozen LUT) is real but **capped at d≈3**. Sub-2-bit is a
   **bit-budget** problem, not a codebook or dimension problem — consistent with the prior 4× "smaller/better
   than Q4_K quality thesis disproven" finding (`research/STRAND-quant-findings-and-playbook.md`,
   `research/STRAND-stage2.2-bench-results.md`). This is the fifth independent negative on the same thesis.
3. **The path forward for usable sub-2-bit is B3 (training), the flagged frontier** — QAT / BitNet-ternary,
   with STRAND as the *deterministic runtime* for the trained weights. This trades training-free for usable
   1-bit and is a strategic choice, not a free win; pursue only if usable sub-2-bit is a product requirement.
   B1 has now closed the door on getting there PTQ-only.
4. **C2 (side-info minimization) remains an always-worth-doing density win, independent of this verdict.**
   We sit +0.7–2.7 % above the floor, almost all of it the 80 bits/256-block of scales; predictive/shared/
   entropy-coded scales shave toward the information-theoretic minimum and are determinism-safe. This is where
   the *real* near-term density gain is, now that C1 (shaping gap via dimension) is shown immaterial at
   sub-2-bit budgets.

**Ranked QUALITY/DENSITY next actions:** (1) C2 side-info minimization — the sure, determinism-safe density
win, now the best near-term lever; (2) B3 training frontier (QAT/BitNet) — the *only* route to usable
sub-2-bit, pursued only if that regime is a product requirement; (3) **closed:** further high-d / frozen-LUT
PTQ work on sub-2-bit quality (B1/C1) — dead, do not re-open without a fundamentally new lever.

---

## Cross-gate read-through

Both decisive gates returned negative, and **both negatives are clean** (bit-identical determinism upheld,
measured on real shapes / well-fit books, with the artifact controls done). They are not failures of
execution — they are the experiments doing their job and **pruning two expensive branches**:

- **SPEED:** the GPU "serial wall" is *not* a serial dependency (A1 proved the window is free) — it is raw
  per-weight decode ALU. So the cheap revival is dead; the live shot is A2 bitslicing, and the realistic
  floor is the already-measured CPU block-parallel 6×.
- **QUALITY/DENSITY:** the sub-2-bit collapse is *not* a codebook/dimension problem (B1 spent that lever
  fully) — it is the bit budget. So PTQ-only sub-2-bit is dead; usable sub-2-bit requires training (B3), and
  the near-term density win is side-info (C2), not shaping.

The master plan's central invariant is undamaged: **the integer-only, bit-identical-across-hardware decode
held through every path in both gates** (1152 CPU cases + GPU one-hot probe for A1; 28/28 configs ×2 L-values
for B1). The moat is the determinism, and it is intact — the gates only narrow *where* the wins live, not
*whether* the runtime is sound.

### Artifacts
- **A1:** `crates/strand-decode-kernel/src/windowed.rs`, `crates/strand-decode-kernel/shaders/strand_windowed_gemv.metal`,
  `crates/strand-decode-kernel/src/bin/gate-windowed.rs` (+ one `pub mod windowed;` line in `…/src/lib.rs`).
- **B1/C1:** `crates/strand-quant/src/bin/gate-vectrellis.rs`; results
  `scratch/vt-head-l12.txt` (headline, L=12), `scratch/vt-head-l10.txt` (robustness, L=10).
  Run: `cargo run -p strand-quant --release --bin gate-vectrellis` (env `VT_N`, `VT_ITERS`, `VT_LFIX=<L>`, `VT_LWIDE`).
