# Gravity external baseline matrix

Research cutoff: `2026-07-21`. Primary sources only.

QMoE is the only surveyed method clearly publishing a full giant-MoE checkpoint plus all metadata below one BPW (0.807). NanoQuant and LittleBit publish informative whole-model payload/memory figures but not canonical artifact audits.

| Method | Class / scope | Rate boundary | GLM-5.2 comparison |
|---|---|---|---|
| [BitNet b1.58 / b1.58 2B4T](https://arxiv.org/abs/2402.17764) | native ternary architecture trained from scratch; dense 0.7B-3.9B experiments; released 2.4B; ternary linear weights and per-token INT8 activations | nominal `1.5849625007`; decoded `None`; canonical `None` | architecture context only: dense 2.4B native pretraining, not GLM PTQ |
| [ParetoQ](https://arxiv.org/abs/2502.02631) | QAT initialized from floating pretrained checkpoints; dense MobileLLM 125M-1.5B and Llama-3 1/3/8B; embeddings/output stay floating | nominal `[1, 1.58, 2, 3, 4]`; decoded `None`; canonical `None` | useful below-3-bit representation evidence; not PTQ, MoE, complete storage, or GLM scale |
| [QuIP](https://arxiv.org/abs/2307.13304) | training-free Hessian-aware weight-only PTQ with incoherence and LDLQ; dense OPT 125M-66B and Llama-2 70B; W2/3/4A16 | nominal `[2, 3, 4]`; decoded `None`; canonical `None` | dense >=2-bit NVIDIA PTQ reference only |
| [QuIP#](https://arxiv.org/abs/2402.04396) | Hadamard incoherence plus E8 lattice codebooks; dense Llama 7B-70B; appendix Falcon-180B and no-FT Mixtral-8x7B | nominal `[2, 3, 4]`; decoded `None`; canonical `None` | closest established extreme lattice/VQ family with a MoE appendix, but only 2-bit and incomplete accounting |
| [AQLM](https://arxiv.org/abs/2401.06118) | additive multi-codebook weight-only PTQ; dense Llama-2 7/13/70B, Mistral-7B, Mixtral-8x7B; Mixtral gate unquantized | nominal `[1.97, 2.07]`; decoded `None`; canonical `None` | MoE/PTQ relevant, but not sub-1-bit and far smaller |
| [VPTQ](https://aclanthology.org/2024.emnlp-main.467/) | second-order vector PTQ with residual/outlier codebooks; peer-reviewed dense Llama-2/3 and Mistral through 70B; later repo scale claims are not equivalent paper evidence | nominal `[2.02, 2.26]`; decoded `None`; canonical `None` | dense >=2-bit paper evidence; repo-scale claims are unmatched |
| [GPTVQ](https://arxiv.org/abs/2402.15319) | Hessian-aware vector PTQ with EM codebooks; dense Llama/Mistral through 70B plus Mixtral-8x7B | nominal `[2.125, 4.125]`; decoded `None`; canonical `None` | PTQ/MoE relevance, but >=2-bit and non-Apple |
| [BiLLM](https://arxiv.org/abs/2402.04291) | training-free Hessian/saliency binary PTQ; dense OPT 1.3B-66B and Llama/Vicuna 7B-70B | nominal `[1.07, 1.13]`; decoded `2.88`; canonical `None` | warning that 1-bit arithmetic is not 1 physical BPW; dense only |
| [STBLLM](https://openreview.net/forum?id=6XUSDvBFkV) | training-free N:M sparse binary PTQ; dense OPT/Llama/Mistral through 70B | nominal `[0.53, 0.85]`; decoded `{'2_of_4_kernel_floor_before_metadata': 1.5, 'nanoquant_reaccount_4_of_8': 3.5, 'nanoquant_reaccount_6_of_8': 4.0, 'nanoquant_reaccount_8_of_8': 4.13}`; canonical `None` | not MoE and not physically sub-1-bit |
| [QMoE](https://proceedings.mlsys.org/paper_files/paper/2024/file/c74b624843218d9b6713fcf299d6d5e4-Paper-Conference.pdf) | retraining-free GPTQ-style expert PTQ plus lossless dictionary/entropy encoding; official BF16 SwitchTransformer base/large-128 and c2048 1.6T; ternary experts, BF16 non-experts | nominal `0.8`; decoded `None`; canonical `0.807` | critical closest prior: giant BF16-source MoE, complete sub-1 artifact and direct runtime; materially different encoder-decoder MLM, unusual zero-heavy experts, and shallow capability evidence |
| [BTC-LLM](https://aclanthology.org/2026.acl-long.1066/) | binary-codebook PTQ with learnable invertible transform; dense Llama 7B-65B, Qwen and FBI-LLM | nominal `[0.7, 1.11]`; decoded `None`; canonical `None` | peer-reviewed sub-1 claim but accounting/reproducibility insufficient for physical comparison |
| [NanoQuant](https://arxiv.org/abs/2602.06694) | low-rank binary-factor PTQ with ADMM and scale-only KL distillation; dense Llama/Qwen/Gemma/Rnj-1 0.6B-70B; no MoE support listed | nominal `[1.0, 0.8, 0.55]`; decoded `0.667`; canonical `None` | strongest dense PTQ comparator; dense 70B, gradient-calibrated, CUDA-only and not artifact-audited |
| [LittleBit](https://arxiv.org/abs/2506.13771) | low-rank binary-factor QAT; dense OPT/Llama/Phi/QwQ evaluated through 32B; embeddings/lm-head FP16 | nominal `[0.1, 1.0]`; decoded `0.927`; canonical `None` | important sub-1 QAT representation reference; not one-pass PTQ, MoE, or GLM scale |

## Claim boundary

QMoE prevents any honest first/only claim for complete sub-one-bit giant-MoE PTQ. The potential contribution is the autoregressive GLM/capability/Apple-local conjunction.

Seal: `c69b5d636944fbbfcdc76c72ce3441e323add538b94a3566b25f251f1ff49206`.
