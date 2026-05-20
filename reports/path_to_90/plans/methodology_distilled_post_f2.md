# Methodology distillation post-F.2 — patterns 11–20

Extends `acceleration_patterns.md` (patterns 1–10 from L8). The F.1 + F.2
infra work surfaced ten more patterns that future training/optimization
phases MUST inherit. Each pattern has a concrete adoption hook.

These exist because **F.2 burned 0 wall-clock on debug spirals**: the
training run is "let it cook" because every lever was validated before
the first long run started. That is the bar.

## Pattern 11 — Capture-then-iterate for hidden-state phases

**F.1 lesson:** Plan estimated 12 hr (running V2-Lite forward to capture
hiddens). Rewrite as `eagle4/rewrite_medusa.py` window-join over already-
captured Eagle3 shards: **39 sec**. 1000× speedup.

**Pattern:** Any phase whose training input is "model X's hidden state
at position Y" — capture ONCE into parquet, then iterate training scripts
against shards. Never re-forward the base model per training run.

**Adoption hook:** Before launching a hidden-state-bound phase, audit
whether existing captured shards can be re-projected (window-join,
column-rename, residual-extraction) to produce the new training input.
If yes, use the re-projection; budget seconds, not hours.

**Apply at:** F.3 retraining (uses F.2 ckpt + medusa shards already on
disk — no recapture). Future eagle5+/medusa-v2 phases.

## Pattern 12 — Frozen-weight npz extraction unlocks GPU concurrency

**F.2 lesson:** `eagle4/medusa_head.py` doesn't load V2-Lite at all —
just the 838 MB `v2lite_frozen.npz` (LM head + output_norm). Result:
trainer runs concurrent with Claude / slm / bench loops. No clean
window required for training. Overnight training is feasible without
blocking other work.

**Pattern:** Extract whatever frozen weights the trainer needs (LM
head, embedding, normalization params) to a standalone npz. Trainer
loads npz only. No mlx-lm / no V2-Lite forward. No GPU contention
with concurrent processes.

**Adoption hook:** First step of any training phase is a 10-line
extraction script that writes the frozen weights to
`eagle4/<phase>_frozen.npz`. The trainer's `--frozen` flag points
there. The full model is only ever loaded by bench / parity scripts,
never the trainer.

**Apply at:** F.3 retrain (use existing `v2lite_frozen.npz`). Any
future distillation-from-V2-Lite phase.

## Pattern 13 — Triple validation gate: smoke → subset → full

**F.2 lesson:** Three escalating gates before commiting overnight
compute:

| Gate | Cost | Decides |
| --- | --- | --- |
| Smoke (30 steps × 2000 rows) | ~30 sec | Architecture sound? loss decreasing? head 0 top1 > 0? |
| Subset (~30 min, 100k rows × 3 epochs) | ~30 min | Does best-epoch top1 clear GO threshold? |
| Full (~10 hr, 500k rows × N epochs) | overnight | Ship metric ≥ acceptance? |

Smoke caught no bugs in F.2 — but in the failure case it would have
saved overnight wall-clock. Subset's role is the trajectory call: it
proves the architecture converges AND that hyperparams aren't pulling
in the wrong direction.

**Pattern:** Every phase that runs > 1 hr of training defines all
three gates upfront with explicit numeric thresholds at each level.
Skipping a gate forfeits the right to overnight wall-clock.

**Adoption hook:** Each phase plan's "Concrete plan" section lists
the three gates with their commands + thresholds, before the launch
command.

## Pattern 14 — In-RAM cache when data fits

**F.2 lesson:** 491k rows × hidden_high (2 GB) loaded once in 10.6 s,
then iterated across 3+ epochs without re-decoding parquet. M3 Pro
has 18 GB RAM — 2 GB cache is fine. Epoch 1+ pays zero I/O cost.

**Pattern:** Audit dataset size against available RAM (`sysctl
hw.memsize / 1e9`). If dataset fits with ≥4 GB margin, cache fully
in RAM. If it doesn't, mmap parquets and stream by column with
prefetch.

**Adoption hook:** Trainer prints `[cache] <rows> rows, <gb>GB,
load <sec>` on startup. Future phases keep this log line. If load
time exceeds 30 sec, switch to mmap.

**Apply at:** F.3 retrain (same shards, same cache). L7 parity tests
(can cache synthetic Q4_K_M weights similarly).

## Pattern 15 — Column-projected dataloader (always)

**F.2 lesson:** Shards have many columns (next_token_p0..K-1,
hidden_high_p0..K-1, hidden_low, hidden_mid, ...). Training reads
only the 16 columns it needs via `pq.read_table(columns=[...])`. I/O
drops ~10× vs reading bare.

**Pattern:** Never `pq.read_table(file)` without `columns=`. The cost
of listing the columns up front is paid back in seconds on first
load.

**Adoption hook:** Code review: any new dataloader call without
`columns=[...]` is a defect.

**Apply at:** F.3 retrain. Any future medusa-variant trainer.

## Pattern 16 — Front-load all known levers into the infra commit

**F.2 lesson:** All 10 audit levers (A.1 tied LM head … B.4 SwiGLU
adapter) were baked into the F.2 infra commit (`a25ce6a`) BEFORE
the first long run. Zero mid-flight refactors. Compare to Phase 0
of dismantle where multi-hour debug spirals were the norm — those
spirals came from levers discovered mid-flight.

**Pattern:** The infra commit precedes the training commit. The
infra commit lists each lever, its source (audit doc / prior phase /
first principles), and its expected effect. If a lever turns out not
to matter, that's data; if it would have been needed and wasn't there,
that's a wasted overnight.

**Adoption hook:** Each phase plan has an "Audit levers" section
listing the patterns baked into the infra commit. The implementation
phase ships exactly those, nothing more.

**Apply at:** F.3 Rust port (list the kernel patterns / parity gates /
schedule wiring upfront). L7 kernel rewrites (already does this via
"Concrete plan" Step 1-7).

## Pattern 17 — Per-epoch heldout eval, dump JSON

**F.2 lesson:** `medusa_head.py` runs heldout eval at every epoch
boundary, dumps `best_eval.json` when `top1_mean` improves. Mid-flight
visibility is automatic — no human-loop polling needed. The decision
to GO/retune/halt is keyed off this JSON, not off a final readout.

**Pattern:** Every trainer prints per-epoch eval to log + dumps the
best-checkpoint eval to `<ckpt-dir>/best_eval.json`. Format: per-head
metrics + means + epoch number + step count.

**Adoption hook:** Trainer signature includes `--heldout-shards N`
that reserves the last N shards from training. Eval runs in-script,
no separate invocation.

**Apply at:** F.3 retrain. Any future training script.

## Pattern 18 — Hyperparams from first principles, validated at smoke

**F.2 lesson:** lr=3e-4, B=128, K=8, head_weight_slope=1, α_kl=0.5,
β_mse=0.1 — picked from the F.2 audit + first principles (AdamW
default LR, M3 Pro batch sweet spot, balanced auxiliary weights).
Smoke validated them. No grid search. Saved ~6 hours of search wall-
clock.

**Pattern:** Hyperparam selection is reasoning, not search. Each
chosen value has a sentence-long justification in the plan doc. The
smoke gate validates the chosen set; if smoke fails, perturb one
hyperparam at a time, not all at once.

**Adoption hook:** Plan doc lists hyperparams with `value | source`
columns. PR review rejects any "I'll grid search this" placeholder.

**Apply at:** L7 kernel rewrites (ROWS_PER_TG, simd count per shader
— picked from M3 Pro arch). F.3 retrain (inherit F.2 hyperparams
unless a specific reason to change).

## Pattern 19 — Cooperative scheduling is sufficient on M3 Pro

**F.2 lesson:** `nice -n 19 taskpolicy -b /path/to/python ...` lets
the F.2 trainer coexist with Claude + bench loops without a "clean
window" gate. Per-shard rows/s dropped from 140 → 113 under
contention (~20%); that's acceptable cost for overnight elasticity.

**Pattern:** Frozen-weight trainers (Pattern 12) inherit `nice -n 19
taskpolicy -b` and skip the clean-window gate. Only bench / parity
scripts that measure dec_tps require a clean window. The two roles
do not overlap.

**Adoption hook:** Trainer launch commands ALWAYS prefix `nice -n 19
taskpolicy -b`. Bench commands document "Cmd-Q Claude first" as a
prerequisite, separately.

**Apply at:** F.3 retrain (clearly fits — trainer not bench). L7
bench (clearly does NOT fit — clean window required).

## Pattern 20 — Convergence-based epoch cap

**F.2 lesson:** Full run is 10 epochs over 500k rows. After it ships,
read `best_eval.json` for the epoch where `top1_mean` peaked. If
peak is at epoch 5–6, future F.x retrains cap at 6 epochs — 40%
wall-clock saving on every future training run in this family.

**Pattern:** First run of a new training family explores epoch count.
Subsequent runs in the same family inherit the empirical convergence
epoch + 20% safety margin. Re-explore only when the architecture
changes materially (e.g., new adapter shape, new aux loss).

**Adoption hook:** `methodology_distilled_post_f2.md` (this file)
gets appended after the F.2 full run with the observed convergence
epoch. Future plans cite it.

**Apply at:** F.3 retrain (if F.2 converges at epoch 6, F.3's
retrain caps at 7). Any future medusa-variant.

## Pre-launch checklist (extends `acceleration_patterns.md`'s 1–10)

Patterns 11–20 add to the existing checklist:

- [ ] **11.** Hidden-state input re-projected from existing shards (not re-forwarded)
- [ ] **12.** Frozen-weight npz extracted; trainer doesn't load base model
- [ ] **13.** Smoke + subset + full thresholds defined upfront
- [ ] **14.** Dataset size audited vs RAM; cache strategy chosen
- [ ] **15.** Every parquet read uses `columns=[...]`
- [ ] **16.** All known levers baked into the infra commit before training launches
- [ ] **17.** Per-epoch heldout eval dumps `best_eval.json`
- [ ] **18.** Hyperparams justified one-by-one; no grid-search placeholders
- [ ] **19.** Trainer wraps with `nice -n 19 taskpolicy -b`; clean window required only for bench
- [ ] **20.** Epoch count inherited from prior convergence data + 20% margin

20/20 means the phase ships overnight unattended to a decision-grade
answer. < 16/20 means the user should expect manual intervention.
