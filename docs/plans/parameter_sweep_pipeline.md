# Hawking Condense — Parameter-Sweep Testing Pipeline (2026-06-23)

> Scaffolded on the 19 GB Mac while the **Mac Studio (96 GB / 1 TB / 60-core GPU)** ships.
> Runs degraded here (CPU-f16, ≤7B), full on the Studio (`mps`/f32, the whole ladder).
> Canonical design — paste-able into the 7B chat. State + tools in `tools/condense/`,
> see [[condense-7b-scale-2026-06-23]] and [[condense-32b-native-serving-2026-06-23]].

## 0. The philosophy (the contract every experiment obeys)

**Hawking's goal: match each model's parameter count to the LOWEST BIT POSSIBLE at
near-1:1 quality, via the doctor. The smallest artifact with full capability = the
highest tps.**

So the sweep is **not a fixed grid** — it is an **effective-bpw floor search** over RECIPES
(single-bake AWQ + residual STRAND b1+b2), climbing eff-bpw and stopping at the smallest viable:

```
RECIPES, ascending effective bpw:
  awq1b(1.34) awq2b(2.34) res1+1(2.68) awq3b(3.34) res2+1(3.68) awq4b(4.50) res2+2(4.68) res3+2(5.68)

for each model (smallest → largest):
    f16_ppl = measure(f16)                       # the 1:1 reference
    for recipe in RECIPES:                       # CLIMB effective bpw
        heal = bake(model, recipe)               # single = AWQ base · residual = full-rank heal
        Δ = ppl(heal)/f16_ppl - 1                # healed degradation
        record(model, recipe, Δ, eff_bpw, size, tps)
        if Δ ≤ NEAR_1to1:  floor[model] = recipe; break   # smallest viable → STOP
    emit artifact at floor[model]                # highest tps
```

Two thresholds, both reported:
- **`1:1` floor** — lowest bits with healed Δ ≤ **+2 %** (true near-lossless, "have your cake").
- **`win` floor** — lowest bits with healed Δ ≤ **llama Q4_K's ~+8 %** but at far lower bpw
  (the comparative win: *denser AND ≥ Q4_K quality*).

**The central hypothesis the ladder exists to test: the bit-floor DESCENDS as params rise**
(more redundancy → tolerates fewer bits). 0.5B floors at 3-bit (measured); the bet is 32B→2,
100B→1–2, 405B→1. If the curve holds, **the biggest models compress hardest** — that is the
most aggressive possible result, and it is what makes 1-bit-405B-on-a-Mac plausible.

## 1. Two data streams, two ceilings

The work splits into two independent measurement streams (the user's framing):

- **Stream A — CONDENSE (quality recovery).** Per `(model, bits)`: f16 ppl → AWQ-base Δ →
  doctor-healed Δ, vs llama Q4_K / MLX-4bit references. *Question: how low can the bit go
  while the doctor holds near-1:1?* Tools: `awq_bake.py → ppl_bench.py → doctor_lora.py`.
- **Stream B — SERVE (runtime of the condensed `.tq`).** Per artifact: `.tq` size, RAM,
  tok/s, and the **RAM-cliff** verdict (Hawking runs while llama Q4_K swaps). Tools:
  `condense.sh`/`tq_bake` → native GPU bitslice serve → `hawking generate`.

They have **different ceilings on the same 96 GB box** — this asymmetry IS the design:

| | needs resident | 96 GB ceiling (naive) | 96 GB ceiling (phase-2) |
|---|---|---|---|
| **Condense** | f16 model (≈ 2×params) | **~32–40B** (32B f16 ≈ 66 GB) | **~235B+** via block-wise streaming |
| **Serve** | only the `.tq` (≈ bpw/8 ×params) | **~200B @ 3-bit, ~285B @ 2-bit** | (same) |

**We can serve far bigger than we can naively condense.** Phase-2 block-wise condense
(one transformer block resident at a time, peak ≈ a few GB) decouples condense RAM from total
size — letting the **condense ceiling chase the serve ceiling**. The same box that *serves*
235B can then also *condense* it, one block at a time.

### Serve footprint (weights only = params × bpw/8, GB) — 84 GB weight budget on 96 GB

| params | 1.0 bpw | 1.34 bpw | 2.34 bpw | 3.34 bpw | 4.50 (Q4_K) |
|---|---|---|---|---|---|
| 32B  | 4   | 5   | 9   | 13  | 18  |
| 72B  | 9   | 12  | 21  | 30  | 41  |
| 120B | 15  | 20  | 35  | 50  | 68  |
| 235B | 29  | 39  | **69** | 98 ❌ | ❌ |
| 405B | **51** | **68** | 118 ❌ | ❌ | ❌ |
| 671B | 84 ⚠ | ❌ | ❌ | ❌ | ❌ |

Serve ceiling: **~200B @ 3-bit, ~285B @ 2-bit, ~500B @ 1.34-bit.** 405B needs **≤1.34 bpw** —
hence 1-bit is the enabling frontier. (⚠ 671B @ 1.0 bpw ≈ 84 GB is the absolute edge of the box.)

**Conditional:** serving anything >1.5B at low-bit requires the **GPU bitslice TQ serve path**
(`strand_bitslice_gemv_tcb`, ~90 % built, proven in RWKV-7; the CPU `matvec_rht` path
Q12-inflates to f32 → OOM). Wiring it into Qwen is the Stream-B prerequisite.

## 2. The phased plan (your "both, phased")

- **Phase 1 — full curve, fast.** Naive-resident condense ≤32B + serve up to ~200B (any `.tq`
  that fits). Produces the complete bit-floor-vs-scale curve for the spine + cross-family.
- **Phase 2 — block-wise/streamed condense.** Unlocks condensing 72B / 120B / 235B on 96 GB
  (BRECQ-style per-block QAT = also the 1-bit ceiling-breaker). Condense ceiling → serve ceiling.
- **Phase 3 — the 1-bit frontier.** If block-wise makes 1-bit viable at scale, add **405B**
  (serves only at ≤1.34 bpw) and **671B** (1.0 bpw, the edge). 1-bit-405B-on-a-Mac = the capstone.

## 3. The model ladder ("as many as possible — rigorous")

Spine + cross-family ladders so the win is shown to **generalize across architectures**, not
just Qwen. Driver reads each `config.json` at fetch time for exact `hidden`/`intermediate`
(→ the serve invariant `in_features % 256 == 0` and exact sizes); the manifest declares intent.
Full machine-readable list in `tools/condense/ladder.py`.

- **P0 — Qwen2.5 spine (the clean 144× scaling curve):** 0.5B · 1.5B · 3B · 7B · 14B · 32B · 72B.
  (0.5B `hidden=896` fails the serve invariant → condense-stream dev-probe only.)
- **P1 — cross-family generality:**
  - **Llama-3.x:** 3.2-1B · 3.2-3B · 3.1-8B · 3.3-70B · (3.1-405B → P3 frontier).
  - **Gemma-2:** 2B · 9B · 27B.
  - **Mistral:** 7B-v0.3 · Nemo-12B · Small-24B.
  - **Phi:** 3.5-mini-3.8B · 3-medium-14B.
- **P2 — 100B+ / MoE (serve-stream + phase-2 block-wise condense):**
  - **Qwen3 MoE:** 30B-A3B · 235B-A22B. **MoE is the dream case** — huge *total* params (needs
    condense to fit) but small *active* params (fast tps). 235B-A22B @ 2-bit ≈ 69 GB fits, decodes
    like a 22B. Highlight these.
  - **gpt-oss:** 20B · 120B.  **DeepSeek:** V2-Lite-16B · (V3-671B → P3 edge).
- **P3 — 1-bit frontier (gated on "1-bit viable at scale"):** Llama-3.1-**405B**, DeepSeek-V3-671B.

## 4. The recipe ladder + "extract the most"

The doctor's quality path (measured 2026-06-23, commit `3bc128a`): **residual STRAND quant**
`W ≈ STRAND_b1(W) + STRAND_b2(W − STRAND_b1(W))` — **full-rank** (captures the high-rank error
LoRA caps on), **codec-native** (no uniform-proxy transfer gap that wrecked QAT), and
**train-free** (a double bake — no 19 GB training wall → it scales). 0.5B (hardest case):
`res3+2` = **+1.6 % (≈1:1)**, `res2+2` = **+8.9 % @~4 bpw (beats Q4_K)**. This **supersedes
LoRA-KD** as the heal. The floor-search climbs effective bpw across both methods:

  `awq1b · awq2b · res1+1 · awq3b · res2+1 · awq4b · res2+2 · res3+2`  (ascending eff bpw)

**Frontier caveat (the 1-bit question stays open):** residual costs +b2 bpw, so for extreme-fit
targets — 405B serves only at ≤1.34 bpw — you **cannot afford a residual**; only a single-bake
1-bit fits. So **1-bit single-bake viability at scale is THE frontier question**; residual wins
the mid-range where there's bpw headroom to spend on quality.

**Extract the most (intensify at the floor):** AWQ alpha-sweep {0.25,0.5,0.75}; and the 7B
chat's active next step — **AWQ+residual stack** (residual on the AWQ-scaled base) — becomes a
recipe the moment it lands.

**Residual SERVE note:** serving a residual artifact at its low eff-bpw needs a **two-part
`.tq`** (decode base + residual, sum in the GEMV) — not yet built. Today residual proves
QUALITY (Stream A); single-bake `.tq` is the SERVE path (Stream B). Per-recipe recorded —
Stream A: `f16_ppl, recipe, eff_bpw, heal_delta`; Stream B: `tq_gb, fits96, llama_q4k_gb,
llama_fits, cliff, tps`.

## 5. Results schema (the output)

One JSONL row per `(model, bits, method)` cell → `reports/condense/ladder.jsonl` (append-only,
idempotent). Rendered by `sweep_render.py` → `reports/condense/MATRIX.md`:

- **Headline — bit-floor vs scale:** `model | params | 1:1-floor | win-floor | Δ@floor | tps@floor`
  + the curve (does the floor descend with scale?).
- **Stream A table — quality recovery:** every `(model,bits)`: f16 ppl, PTQ/AWQ/doctor Δ, vs-llama.
- **Stream B table — serve/cliff:** every artifact: `.tq` size, RAM, tps, fits96, vs llama Q4_K.
- **The join (the aggressive claim):** "Hawking @ `<floor>` bpw matches/beats llama Q4_K quality
  AND fits/runs at `<params>` where llama Q4_K swaps/can't load."

## 6. The harness (files in `tools/condense/`)

- **`ladder.py`** — the manifest (models, families, priority, tier) + size/tier/serve-fit math.
  `python ladder.py --plan` prints the cells; `--tsv` a flat table.
- **`sweep.py`** — the driver. Floor-search per model, both streams, **idempotent** (skips cells
  already in the JSONL), **device-aware** (`--profile here|studio` sets `DOCTOR_DEVICE/DTYPE`),
  **JIT-fetch** (downloads f16 only when about to condense; disk-gated) + **purge** (deletes
  f16/intermediate safetensors after recording numbers, keeps `.tq` + JSONL → stays within 1 TB).
  **Safe default:** prints the plan; requires `--go` to execute. Resumable.
- **`sweep_render.py`** — JSONL → `MATRIX.md`.
- **`sweep_watchdog.sh`** — detached Studio launcher (download → Stream A → Stream B per model,
  each stage logged independently so a long run never loses an earlier result).

## 7. Compute dtype — bfloat16, NOT float16 (correction 2026-06-23)

**Use `bfloat16` on both profiles.** float16 maxes at 65504; a 7B's activations exceed it on
the CPU forward → inf → **nan ppl** (observed: the first 7B watchdog run crashed with f16
ppl=nan). bfloat16 is the same **2-byte** footprint but carries the full fp32 *range* (no
overflow), and it sidesteps the MPS **float16** GQA bug (that bug is fp16-specific). The
2-byte width is also what keeps the resident-condense ceiling at ~34B — **32B at bf16 = 64 GB
fits 96 GB; at f32 it would be 128 GB and would NOT**, dropping 32B into the streamed tier.
- `here` profile  = `cpu` / `bfloat16`  (19 GB box, ≤7B feasible-slow)
- `studio` profile = `mps` / `bfloat16` (96 GB box, full ladder, 32B resident)
- **Studio caveat:** MPS bf16 is newer — sanity-check bf16-vs-f32 ppl on a small model before
  trusting the full ladder; if MPS bf16 misbehaves, fall back to f32 (32B → streamed tier).

## 8. Runnable now (19 GB) vs gated on the Studio

- **Now (scaffold-only per the user):** harness + manifest + schema built + validated in plan
  mode. The 7B watchdog ran but hit the f16→nan bug above; its dtype is now bf16 (re-run when
  ready). No new local compute launched.
- **Studio (96 GB):** Phase-1 full curve 0.5→32B condense + serve to ~200B; then Phase-2
  block-wise (72/120/235B); then Phase-3 1-bit frontier (405B/671B). Flip the profile, same
  scripts: `bash tools/condense/sweep_watchdog.sh studio`.

## 9. Disk discipline (1 TB)

f16 dirs are large (32B ≈ 66 GB, 72B ≈ 145 GB); `.tq` artifacts are tiny (all of them < ~100 GB
combined). The driver **JIT-downloads f16 → condenses → records numbers → purges f16 + the
f16-sized AWQ/heal safetensors**, keeping only the `.tq` + the JSONL row. Big models default
`keep_f16=false`; small ones `true` (cheap to re-run). Never hold two f16 giants at once.
