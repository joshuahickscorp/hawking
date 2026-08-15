# HAWKING TOURNAMENT-READINESS REPORT (2026-08-14)

Gate to open: **MANAGER_ASCENT_TOURNAMENT_ACTIVE** = coherent Gravity artifacts + native
runtime correct + **BASE_TRUE_TPS >= 100 on BOTH Q30 and Q80** (Canon Phase A; TG3 gate:
operational L_minimum 100, ascent 333). Every claim below is a measurement/receipt, not a plan.

## Headline
The tournament is REACHABLE but not close, and the blocker is NOT what the campaign assumed.
Coherence is **solved for the Q4-class representation and DEAD for sub-bit**. The remaining work
is a **TPS engineering campaign on a coherent Q4-class artifact** — porting the binary path's
already-proven optimizations to it — plus **building a coherent Q80 artifact** (only a sub-bit
Q80 exists).

## 1. COHERENCE — settled
- Q30 uniform-q4 (Q4-class): **COHERENT** (verified: "The capital of France is Paris.", "2 + 2 = 4",
  "Jupiter"; doctor6 ~0.987). A coherent Q30 artifact EXISTS.
- Q30 sub-bit activation-weighted-SVD: **DECISIVELY INCOHERENT**, not fixable by capture. Fitting X
  from the correct+dense BF16 perexpert64 capture lifts L0 cosine only +0.015 over the gibberish
  baseline; all-layer mean 0.8255 (equal-layer gmean 0.808, late layers 0.771); layer-product
  ~3.6e-5 vs the >=0.5 composition bar. The sub-bit REPRESENTATION's cosine ceiling (~0.83) is
  below the ~0.9+ needed for 48-layer directional survival. **Sub-bit ascension is closed for
  coherence.** (Negative science — do not re-run capture variants expecting coherence.)
- Q80: **no coherent artifact exists.** Only a sub-bit `qwen80_complete_binary_gravity` candidate
  is on disk (inherits the sub-bit ceiling; coherence untested but expected to fail). No stored
  Q80 generate result.

## 2. TPS — the real blocker
- Q30 coherent uniform-q4: **3.35 BASE_TRUE_TPS** (298 ms/token). ROOT CAUSE: **98 command
  buffers/token** (binary path uses 1) + 2165 dispatches + forced SerialControl kernel
  (driver:1018). 98 CB x ~3ms ~= 294ms = the entire per-token wall — the matmul is not the cost.
  ~150x off the memory roof => huge headroom.
- LEVER (proven on the binary path: device-expert-table collapses 98 CB -> 1): port the
  device-expert-table + rowblock + fused-QKV to the uniform-q4 kernel. **IN FLIGHT**
  (uniformq4-tps lane). This is the single highest-value tournament move.
- Q80 TPS on a coherent artifact: **unmeasured** (no coherent Q80 exists to measure).

## 3. RUNTIME CORRECTNESS — good
The composed fast pipeline (grok/integrate-wins-...@c79c009e2) executes Q30 with bit-identity
gates green (rowblock -96 dispatch bit-identical, geometry=0 warm, repack byte-identity 13/13,
device-fix). Native runtime is not the blocker.

## 4. REMAINING BLOCKERS + next concrete step (each)
| Blocker | State | Next concrete step |
|---|---|---|
| Q30 coherent @ 100 TPS | coherent artifact @ 3.35 TPS | port device-expert-table+rowblock to uniform-q4 (IN FLIGHT); then remaining CB/dispatch levers |
| Q80 coherent artifact | none (only sub-bit, incoherent) | build a Q80 Q4-class (uniform-q4-analog) pack; verify coherence (Paris-class gate) |
| Q80 @ 100 TPS | unmeasured | after a coherent Q80 exists, port the same fast kernel; measure |
| uniform-q4 schema drift | group128 frozen out of the current driver; group64 works | wire group128 or standardize on a coherent group64/analog |
| capability/TG receipts | not written for a coherent-fast artifact | run TG3 once a coherent-fast Q30 exists |

## 5. Bottom line
Sub-bit was the wrong ascension target for coherence. The viable tournament path is a
**coherent Q4-class artifact made fast via the already-proven binary-path kernel optimizations**,
for both Q30 (in progress) and Q80 (needs the coherent artifact built first). No new research
unknown remains on the critical path — it is engineering (kernel ports) + one artifact build (Q80).
