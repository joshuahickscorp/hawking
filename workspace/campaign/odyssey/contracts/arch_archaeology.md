# ODYSSEY-I STATIC ARCHITECTURE ARCHAEOLOGY (web-enabled, no edits)

For each patient below, produce a DETERMINISTIC architecture fact-sheet from
OFFICIAL sources (HF model card, `config.json`, tech report / paper, official
repo). This is static recon that runs while weights download. Cite EVERY fact with
a URL. Mark UNKNOWN if unverified — never guess a number.

Patients:
- O000  google/gemma-3-1b-it
- O001  tiiuae/Falcon-H1-7B-Instruct
- O005  Qwen/Qwen3-30B-A3B
- O010  zai-org/GLM-4.5-Air

For EACH patient report:

IDENTITY: exact repo; latest revision/commit if pinned; tokenizer; source precision
(bf16/fp16/native-qat); license; params(total); params(active, if MoE).

ARCHITECTURE: dense / MoE / hybrid; n_layers; hidden_size; attention type
(MHA/GQA/MLA/linear/KDA); n_heads / n_kv_heads; SSM/state blocks if hybrid (which
layers, state size); MoE topology (n_experts, experts/token top-k, shared experts,
router type sigmoid/softmax, which layers are MoE vs dense); context length;
positional scheme (RoPE theta etc.); modality (text/vision/audio); vision-tower
params if multimodal; MTP / draft heads.

KNOWN QUANT / RUNTIME: does an official low-bit / MLX / GGUF release exist? native
QAT? known-good external runtime (mlx-lm / llama.cpp / vllm / transformers) with
version notes or gotchas for THIS model on Apple Silicon.

GRAVITY-RELEVANT NOVELTY: what makes this patient architecturally NOVEL vs a plain
Qwen transformer — the thing Hawking must learn from it. 1-3 bullets.

COMPILER HYPOTHESES: 2-3 cheap, falsifiable representation/execution hypotheses
worth testing. For each: the cheapest discriminator, expected physical win, Doctor
risk.

Finish with a NEXT table: `patient | first cheapest discriminator to run once
weights land`.

Be terse and factual. Tables over prose.
