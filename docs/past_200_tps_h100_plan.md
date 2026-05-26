# Past-200 TPS H100 Push

## Objective

Produce Colab Pro H100 artifacts that target past 200 decoded tokens/sec on
Apple Silicon through two deployable tracks:

| Track | What It Breaks | Target |
|---|---|---|
| Single-session | Reads fewer bytes per generated token | 200+ dec_tps projected with a 1.5B Qwen student, Q2/IQ2 calibration, W4A8/AWQ, and a trained Eagle5 head |
| Aggregate | Reuses the same weight read across active requests | 200+ aggregate dec_tps projected with continuous batching at slot count 4+ |

The H100 notebook is built to generate the training and calibration artifacts
for both tracks in one run-all path. Local Rust runtime work still owns the
actual Metal execution path, but the notebook removes the training blocker and
writes the exact runtime knobs the local side should consume.

## Decision Stack

### Track A: single-session 200+

1. Use `Qwen/Qwen2.5-1.5B-Instruct` as the speed target.
2. Build a calibration corpus with residual, intermediate, top-k logits, and
   per-site activation statistics.
3. Export the frozen baseline from HF weights for the student.
4. Train Eagle5 heads over a high-ROI grid:
   - 1-block baseline heads.
   - 2-block heads.
   - residual-delta variants for better multi-step hidden simulation.
   - longer row-token windows for tail acceptance.
5. Run tau and frontier policy search:
   - fixed K.
   - variable K by calibration confidence.
   - entropy/margin routing.
   - lattice oracle probe for the tree-verify path.
6. Emit AWQ smoothing and Q2/IQ2 importance artifacts.
7. Rank candidates by projected single-session TPS using the student baseline
   profile selected in the notebook.

The default notebook projection assumes the local student path uses the same
formula as the existing Qwen-3B stack:

```text
projected_tps = base_student_tps * (1 + accepted_draft_tokens) * quant_multiplier * spec_efficiency
```

Passing 200 requires one of these shapes:

| Shape | Base Student TPS | Accepted Draft Tokens | Quant Multiplier | Spec Efficiency | Projected |
|---|---:|---:|---:|---:|---:|
| Conservative 1.5B Q2 | 85 | 1.80 | 1.00 | 0.85 | 202 |
| Strong 1.5B Q2 | 95 | 2.00 | 1.00 | 0.82 | 234 |
| Strong 1.5B W4A8 | 70 | 2.60 | 1.20 | 0.82 | 248 |

### Track B: aggregate 200+

1. Keep the full Qwen-3B quality target.
2. Use the best Eagle5 + W4A8/AWQ policy from the notebook.
3. Apply continuous batching locally with `slot_count >= 4`.
4. Report aggregate TPS separately from per-session TPS.

Aggregate projection:

```text
aggregate_tps = per_session_projected_tps * active_slots * cb_efficiency
```

Example gate:

| Per-Session TPS | Slots | CB Efficiency | Aggregate |
|---:|---:|---:|---:|
| 75 | 4 | 0.75 | 225 |
| 90 | 4 | 0.72 | 259 |
| 105 | 3 | 0.72 | 227 |

## Notebook Contract

The notebook at `colab/qwen_past_200_h100.ipynb` writes:

| Artifact | Purpose |
|---|---|
| `student_corpus/` | Student Eagle5 training data and activation stats |
| `qwen15b_frozen.npz` | Student frozen baseline for Eagle5 training/eval |
| `qwen15b_awq_smoothing.json` | AWQ/W4A8 activation smoothing |
| `qwen15b_q2k_importance.npz` | Q2_K/IQ2 channel-importance artifact |
| `checkpoints/eagle5_qwen15b_*` | Student Eagle5 heads |
| `eagle5_tau_*.json` | Tau ranking per trained head |
| `eagle5_frontier_*.json` | Policy search with runtime env hints |
| `past200_summary.md` | Winning path, projected TPS, local commands |

## Runtime Levers To Wire After Artifacts Exist

| Lever | Runtime Input From Notebook |
|---|---|
| Student model path | HF model id or converted GGUF for `Qwen/Qwen2.5-1.5B-Instruct` |
| W4A8/AWQ | `qwen15b_awq_smoothing.json` |
| Sub-2-bit bake | `qwen15b_q2k_importance.npz` |
| Eagle5 trained head | `head_final.safetensors` for the winning checkpoint |
| Variable-K | `runtime_hints.variable_k.env` from frontier JSON |
| Entropy routing | `runtime_hints.entropy_routing.env` from frontier JSON |
| Lattice/tree verify | `runtime_hints.draft_lattice.env` from frontier JSON |
| Continuous batching | slot count and CB efficiency projection from `past200_summary.md` |

## Pass/Fail Gates

| Gate | Pass Condition |
|---|---|
| Student single-session | Best projected `single_session_tps >= 200` |
| Qwen-3B aggregate | Best projected `aggregate_tps >= 200` |
| Head quality | Tau rank improves over the 1-block baseline at the same target |
| Quant readiness | AWQ and Q2/IQ2 artifacts exist and include all dense projection sites |
| Runtime handoff | Summary contains concrete env vars and local bench commands |

## Local Bench Commands Emitted By The Notebook

The notebook writes exact commands into `past200_summary.md`; the shape is:

```bash
DISMANTLE_QWEN_AWQ=1 \
DISMANTLE_QWEN_AWQ_SMOOTHING=profiles/qwen15b_awq_smoothing.json \
EAGLE5_HEAD=artifacts/qwen15b_artifacts/checkpoints/<winner>/head_final.safetensors \
TRIALS=10 TOKENS=128 \
  bash tools/bench/eagle5_paired_bench.sh
```

For aggregate mode:

```bash
DISMANTLE_CB_ALPHA=1 \
DISMANTLE_MAX_BATCH_SIZE=4 \
DISMANTLE_SPEC_DECODE=eagle5 \
EAGLE5_HEAD=artifacts/qwen3b_artifacts/checkpoints/<winner>/head_final.safetensors \
  cargo run --release -p dismantle -- serve models/qwen2.5-3b-instruct-q4_k_m.gguf
```
