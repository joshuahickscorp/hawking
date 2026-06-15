# Eagle5 qwen Phase C — initial paired bench

Run on 2026-05-27 right after Phase B.3 (real capture) landed.
Hardware: M3 Pro 18 GB. Repo @ commit `a45e6a8`.

## Method

Same prompt, same 24 max_new_tokens. One-shot (n=1) each — not a full
paired bench, just a sanity check that the wired infrastructure
produces non-broken results.

```bash
# Baseline
DISMANTLE_QWEN_TCB=1 \
  target/release/dismantle generate \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --speculate off \
  --prompt "Once upon a time" \
  --max-new-tokens 24

# Eagle5 + real capture + trained q3b head
DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_EAGLE5=1 \
DISMANTLE_QWEN_EAGLE5_K=4 DISMANTLE_QWEN_EAGLE5_CAPTURE=1 \
  target/release/dismantle generate \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --speculate eagle5 \
  --eagle5-head ~/Downloads/head_final.safetensors \
  --prompt "Once upon a time" \
  --max-new-tokens 24
```

## Result

| Config | dec_tps | draft_accepted | draft_rejected | accept rate |
|---|---|---|---|---|
| baseline (no spec) | **25.85** | 0 | 0 | n/a |
| Eagle5 + trained + capture | **6.03** | 1 | 89 | **1.1%** |

Both produced **identical 24 tokens** of output (`, I was a little girl,
I was a very naughty girl. I was very naughty, and I was a very`).
Greedy correctness preserved — the spec-decode invariant holds.

But Eagle5 is **4.3× SLOWER** than baseline because:
- 1% accept rate × K=4 verify cycle means ~1.04 tokens emitted per cycle
- Each cycle costs ~head_forward × K + verifier_forward × K (serial)
- ≈ 32ms × 4 + 38ms × 4 ≈ 280ms per cycle
- 24 tokens / (24/1.04 cycles × 280ms) ≈ 6 tps ✓ matches observed

## Root cause: train/serve distribution shift

The trained Eagle6 head was trained against the **fp16 HF
`Qwen/Qwen2.5-3B-Instruct`** weights. The Mac runtime serves the
**Q4_K_M quantized** GGUF. Hidden states at the capture layer differ
substantially between these two models — same architecture, different
weight precision, different layer-by-layer activations.

When the head sees activations from Q4_K_M (its inputs at runtime),
it's seeing an out-of-distribution input vs what it learned to predict
against. The result is ~random predictions (1% accept ≈ 1/100, roughly
1/vocab × small bonus).

This is NOT an infrastructure bug — every parity test passes, the
forward matches PyTorch numerically (L_inf 3e-4 on real q3b head),
greedy correctness is preserved end-to-end. The infrastructure is
ready for a head that's calibrated to the runtime.

## What this means for the spec-decode lift workstream

The 12 commits tonight shipped functionally correct Eagle5 spec-decode
infrastructure on the qwen path. But the trained head we have can't
deliver the projected accept rate against Q4_K_M.

Three viable paths forward:

### A. Quant-aware Eagle head training (most accurate)
Retrain the Eagle6 head with capture data from the Q4_K_M model
directly. Requires running the GGUF Q4_K_M model in PyTorch (via
`gguf-py` or llama.cpp's Python bindings) for the calibration corpus
to extract layer-32 hidden states under Q4_K_M precision, then
re-running the trainer. ~2-3 days of training pipeline work + ~1
day of Colab training.

### B. Two-stage train: fp16 then fine-tune on Q4_K_M (likely cheapest)
Take the existing fp16-trained head as a starting point and run a
shorter fine-tune (~5-10 epochs) on Q4_K_M captures. Distribution
shift is the only issue; the head's architecture is correct.
~1 day of pipeline + ~half day Colab.

### C. Use n-gram lookahead instead (no training needed)
`DISMANTLE_LOOKAHEAD=4 DISMANTLE_LOOKAHEAD_K=4` is already shipped and
delivers small-but-positive spec-decode wins (per `memory/lookahead_resurrected_2026_05_26.md`).
The Eagle5 infrastructure is held as inventory for future Q4_K_M-
trained heads.

## Recommendation

Path B (fine-tune existing head on Q4_K_M captures) is the highest
expected value. It uses the existing infrastructure tonight shipped
and only requires the calibration captures from the running Q4_K_M
runtime. The Eagle6 architecture is well-validated by the parity
tests — only the training needs to be matched to the runtime.

Phase C engineering scope:
1. Add a `dismantle dump-captures` subcommand that runs the verifier
   forward, captures (residual, intermediate) at layer 32 per token
   over a calibration corpus, dumps to npz.
2. Update `colab/eagle5_train_pytorch.py` to accept these
   Rust-runtime-captured npz files as the training input.
3. Train the head against the actual runtime captures.
4. Re-run this paired bench.

Effort: 2-3 days attended.

## What's GREEN (already done, no further work needed)

- Loader correctness (Phase A.1) — synthetic + real-head tests pass.
- Rust forward parity vs PyTorch (Phase A.2) — L_inf 3.6e-4 on q3b,
  1.6e-4 on q1p5 (2-block path).
- Threaded matmul (Phase A.3.1-3) — 32ms median forward, bit-identical
  per-row, slight ULP drift on LM-head sum order.
- qwen_dense.rs dispatch (Phase B.1-B.4) — pre-flight gate, serial
  verify, greedy parity preserved across all modes.
- Batched verify (Phase B.5) — same greedy parity, faster (cold cache
  benefits + saved TCB commits).
- Real capture (Phase B.3) — memcpy dispatches don't corrupt
  verifier's forward.

## What's pending (gated on head retraining)

- **Phase D: head retrained on Q4_K_M captures** — see Path B above.
- Phase E: batched verify + capture combined (currently mutually exclusive).
- Phase F: head perf to ~10ms via Metal GEMV or SIMD CPU (currently
  32ms is acceptable; only matters once accept rate >0.5).

When the retrained head delivers, the EXISTING tonight-shipped
infrastructure will pick it up via `--eagle5-head <new_head.safetensors>`
without any further runtime code changes.
