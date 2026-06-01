# Maximal Spec-Decode 500U Summary

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (102.0 GB)
Repo: cadef09

## What This Run Optimized

- Draft/spec heads that can make one verifier pass emit multiple tokens.
- Smaller but still general Qwen students instead of fake speed-only toy paths.
- Quant/calibration inventory for future local Metal experiments.
- Offline policy ranking. Local Mac trace-dispatch remains the real proof.

## Winners

| Target | Track | Winner | Policy | Tau | Accepted/Verify | Offline TPS |
|---|---|---|---|---:|---:|---:|
| q1p5 | flagship_fast_general | q1p5_b1_fast_b1_h16_ff40_s16_lr3e-4_rd000_cw12_seed0 | fixed_k | 7.994 | 23.60 | 1869.9 |

## Local Mac Gates

These are deliberately local. Colab cannot prove the Metal runtime accepts drafts or beats the 30 tps baseline.

1. Build or fetch the target GGUF locally.
2. Build a fresh kernel profile for that GGUF.
3. Run trace-dispatch with the winning head and inspect draft_accepted/draft_rejected.
4. Only after acceptance is real, run clean paired benches.
5. Promote only a stack that beats locked predec baseline under clean bench conditions.

### q1p5 local trace

~~~bash
DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 DISMANTLE_QWEN_Q4K_PREDEC=1 ./target/release/dismantle generate --trace-dispatch --weights models/qwen2.5-1.5b-instruct-q4_k_m.gguf --prompt "Once upon a time" --max-new-tokens 64 --temperature 0.0 --kernel-profile profiles/qwen15b-instruct-q4k.m3pro18.json --speculate eagle5 --verify-window 4 --eagle5-head /content/drive/MyDrive/dismantle/maximal_spec_500u/artifacts/q1p5/checkpoints/q1p5_b1_fast_b1_h16_ff40_s16_lr3e-4_rd000_cw12_seed0/head_final.safetensors
~~~

### q1p5 paired bench after trace acceptance is nonzero

~~~bash
DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 DISMANTLE_QWEN_Q4K_PREDEC=1 WEIGHTS=models/qwen2.5-1.5b-instruct-q4_k_m.gguf PROFILE=profiles/qwen15b-instruct-q4k.m3pro18.json EAGLE5_HEAD=/content/drive/MyDrive/dismantle/maximal_spec_500u/artifacts/q1p5/checkpoints/q1p5_b1_fast_b1_h16_ff40_s16_lr3e-4_rd000_cw12_seed0/head_final.safetensors TRIALS=10 TOKENS=128 bash tools/bench/eagle5_paired_bench.sh
~~~

## Export Contents

