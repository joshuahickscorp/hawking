# Acceleration patterns — distilled from L8 iter 1→4

L8 cost 4 training iterations and ~5 hours of wall time before vector
gate at K=2 cracked the 7% chain-accept ceiling. With the patterns
below applied from the start, the same answer could have arrived in
1 iter and ~45 minutes. **Apply this checklist before launching any
training/exploration phase.**

## Pattern 1 — Mid-flight eval over end-of-run eval

**L8 lesson:** Iter 1 was projected to take 10-15 hr to a final
chain_accept readout. Mid-flight eval at step 200 ckpt answered the
architectural question in ~15 min.

**Pattern:** Every training script that saves periodic checkpoints
gets a companion `tools/<phase>_midflight.sh` that:
- Reads `latest.npz` (saved every 200 steps in eagle4.py)
- Runs the cheapest possible eval against it (reduced max_records,
  shorter prompts, fewer trials)
- Outputs structured JSONL → `reports/path_to_90/_levers/<phase>_evals.jsonl`

**Decision rule:** if mid-flight eval says the experiment is failing
at step 200, KILL. Don't pay for 5+ more hours of training to confirm.

## Pattern 2 — Smoke ≠ eval ≠ bench

Three levels of validation, each cheaper than the next was assumed
to be:

| Level | Cost | What it answers |
|---|---|---|
| Synthetic parity test | ~1 sec | Does the kernel produce correct output? |
| Mid-flight eval (K=1 metric) | ~3-5 min | Is the head learning V2-Lite's argmax? |
| Chain-decode smoke (small) | ~30-60 sec | Does chain rollout actually accept? |
| Clean-window bench (full) | ~5-10 min | Median dec_tps under no-contention conditions |

**L8 lesson:** top1_vs_target (K=1 eval) hit 0.846 at step 400 while
chain_accept (K=4 smoke) was 3.4%. Eval and smoke measure DIFFERENT
things. Always include both.

**Pattern:** For any phase, define which level is the DECISIVE metric
and which is the EARLY signal. Run early signal often, decisive metric
at meaningful inflection points.

## Pattern 3 — Param-group LR for stuck parameters

**L8 lesson:** The scalar `residual_gate` got a tiny gradient signal
because `gradient ∝ (∂loss/∂draft_h) · block_output` and block_output
was small. With shared LR=3e-4 across millions of params, the gate
got swamped.

**Pattern:** When ONE parameter (or group) is structurally getting
weak gradient signal, give it dedicated LR via in-place gradient
scaling:

```python
if param_name in grads and lr_multiplier != 1.0:
    grads[param_name] = grads[param_name] * lr_multiplier
opt.update(model, grads)
```

AdamW's per-param moment estimates correctly track the scaled
gradient. Equivalent to per-param-group LR without extra optimizer
state.

**Apply at:** any future phase where a control parameter (gate, mixing
weight, attention bias) has different gradient magnitude than the
main model parameters.

## Pattern 4 — Curriculum is risk, not always reward

**L8 lesson:** K-curriculum (K=1→4 ramp) seemed safer than K=4 from
step 0, but it actually HURT iter 3 by delaying K=4 chain-rollout
training. Vector gate trained on K≤3 rollouts couldn't generalize
to K=4 at test time.

**Pattern:** Train on the exact regime you'll test on. Curriculum
helps only when:
- The test regime is unstable/dangerous early
- You have evidence the easier regime transfers

**Apply at:** any phase introducing a new dimension (K, sequence
length, batch composition). Default to "train = test regime."

## Pattern 5 — Architectural fix before hyperparameter sweep

**L8 lesson:** Iter 1 → 2 added 3 hyperparameter patches (gate-LR-mul,
K-curriculum, aux-decay) and got 8.3% vs 7.1%. The structural fix
(scalar → vector gate) skipped past all hyperparameter improvements
and gave 33%.

**Pattern:** Before sweeping hyperparameters, identify whether the
problem is structural. Diagnostic: if a metric plateaus despite
hyperparameter changes that should move it, it's a STRUCTURE
problem, not a hyperparam problem.

**Apply at:** any phase where multiple iters in a row produce small
improvements. Step back and audit whether the architecture itself
caps the metric.

## Pattern 6 — Run smoke ×2 for noise floor

**L8 lesson:** Chain smoke variance was ~17% run-to-run on the same
ckpt. A single smoke could report 7% on one run and 8.5% on the
next.

**Pattern:** Always run smoke ≥ 2 times per ckpt, sum draft_accepted
and draft_rejected across runs, compute combined rate. Implemented in
`tools/l8_autoiter.sh:run_chain_smoke`.

**Apply at:** any phase where ship/kill decisions depend on a
stochastic metric.

## Pattern 7 — Auto-iter sequencer for multi-config sweeps

**L8 lesson:** Manual launch + manual smoke + manual decide added
~10 min of wall time per iter transition. With 4 iters this was
40 minutes of human-loop overhead.

**Pattern:** Use `tools/l8_autoiter.sh queue` (or `watch_chain`) to
sequence config experiments. Each config is a small shell script
that writes its iter_name + chain_k_for_smoke to
`l8_status.json` at launch. Autoiter handles the rest.

**Apply at:** any phase running ≥ 3 hyperparameter or architecture
configs sequentially.

## Pattern 8 — Strip-restore for user diagnostic edits

**L8 lesson:** Across 16+ commits, user's diagnostic edits in
`engine.rs`/`kernels/mod.rs`/`deepseek_v2.rs` were preserved exactly
by the strip-restore pattern: before each commit, remove the user
hunk; commit; restore the user hunk.

**Pattern:** Any session that lands many commits while user has
uncommitted diagnostic code MUST use strip-restore (or git stash
push/pop). Track the diff stat before each commit; verify it
matches expectation after.

## Pattern 9 — Compute-vs-engineering accounting

**L8 lesson (and root of the user's "I thought scaffolding was done"
question):** It's tempting to say "next phase is just compute." Make
sure that's actually true by listing concretely what code still has
to be written for the phase to deliver its projection.

**Pattern:** Every phase plan opens with a "what code is missing"
section that lists files + line-count estimates. If the answer is
"none" the phase is compute-bound. If it's nonzero, the phase has
real engineering days/weeks built in.

**Apply at:** all phase plans (forced by the template — see other
plan docs).

## Pattern 10 — Cron + nohup autoiter survives Claude session

**L8 lesson:** `nohup tools/l8_autoiter.sh watch_chain ...` runs
fully autonomously and survives the user closing this Claude
session. The cron handles narration; the autoiter handles
sequencing.

**Pattern:** Long-running phases should use BOTH cron (for periodic
narration to the user) AND a background sequencer (for state
machine actions). Either alone is incomplete: cron without
sequencer requires manual ops; sequencer without cron is opaque.

## Pre-launch checklist for any new phase

Before launching a new training/optimization phase, confirm:

- [ ] Pattern 1: mid-flight eval helper exists for this phase
- [ ] Pattern 2: smoke vs eval vs bench levels are defined
- [ ] Pattern 3: any structurally-stuck param has dedicated LR
- [ ] Pattern 4: curriculum used only with evidence it transfers
- [ ] Pattern 5: structural vs hyperparam question audited
- [ ] Pattern 6: smoke runs ×2 for variance
- [ ] Pattern 7: autoiter queue configured for ≥3-config sweeps
- [ ] Pattern 8: strip-restore active if user has uncommitted edits
- [ ] Pattern 9: "code still to write" section honest about engineering vs compute
- [ ] Pattern 10: cron + nohup sequencer both armed

10/10 means the phase can run autonomously to a decision-quality
answer. < 7/10 means the user should expect manual intervention
mid-run.
