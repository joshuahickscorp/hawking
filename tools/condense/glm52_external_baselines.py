#!/usr/bin/env python3.12
"""Generate the primary-source external low-bit baseline matrix for GLM-5.2."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from glm52_common import REPO_ROOT, atomic_json, atomic_text, seal, sha256_file


CUTOFF = "2026-07-21"

PAPER_PINS: dict[str, tuple[str, str]] = {
    "The Era of 1-bit LLMs": (
        "https://arxiv.org/pdf/2402.17764",
        "bcd625e89f95c39dc9143af4174978fcf2467eb9659d86f8da448d7d9366c332",
    ),
    "BitNet b1.58 2B4T": (
        "https://arxiv.org/pdf/2504.12285",
        "af78afa2409dc8c50ef16574d80081c340d93ea52656be5224640b6eec7f5f01",
    ),
    "ParetoQ": (
        "https://arxiv.org/pdf/2502.02631",
        "8c9b4ec1beaea9af616f78b12f6238275dbd4fae760544b4177df818cdc7ffa8",
    ),
    "QuIP": (
        "https://arxiv.org/pdf/2307.13304",
        "782982b03b97f6437df021d122ed15779454318032888db38d24b038e2062d93",
    ),
    "QuIP#": (
        "https://arxiv.org/pdf/2402.04396",
        "2a5c4c103ba31f4a889344d95059c40e96841761e725500829f00b5a16c3d825",
    ),
    "AQLM": (
        "https://arxiv.org/pdf/2401.06118",
        "41f0da86fb91478587ab70136b3101f79820c223244cb055bb962cc7c31347f4",
    ),
    "VPTQ": (
        "https://aclanthology.org/2024.emnlp-main.467.pdf",
        "74c41bf6249808b581378784e68e6dc15426e48abce53bb72daa287e56abc8f7",
    ),
    "GPTVQ": (
        "https://arxiv.org/pdf/2402.15319",
        "dfcfae2e7b180f82348b8e538b80c9b874f65b37fecbc924142cdbe35bfef3ed",
    ),
    "BiLLM": (
        "https://arxiv.org/pdf/2402.04291",
        "8cd1d9045c93e8cd033f131e87ed4bec932919c2a8a6719b7bc34d34b8330c6f",
    ),
    "STBLLM": (
        "https://arxiv.org/pdf/2408.01803",
        "de6890f45c441e899e2f766ad56c980db321dc8a59e79d013ca09ab6eb0bd726",
    ),
    "QMoE: Sub-1-Bit Compression of Trillion-Parameter Models": (
        "https://proceedings.mlsys.org/paper_files/paper/2024/file/c74b624843218d9b6713fcf299d6d5e4-Paper-Conference.pdf",
        "b5c60a998594fd5759e9d5a95dadf657696da0a7607c6811b6d702f4554f741b",
    ),
    "BTC-LLM": (
        "https://aclanthology.org/2026.acl-long.1066.pdf",
        "9630dd96cbcb2a7bf9d9aa85aaec1d7a25c4eb507cf56c765e68a871713d70b2",
    ),
    "NanoQuant": (
        "https://arxiv.org/pdf/2602.06694",
        "474824b66c7760856ee262eb7cfaa957ebbbba621423225f5ac661c3db9d22a3",
    ),
    "LittleBit": (
        "https://arxiv.org/pdf/2506.13771",
        "54508221ea5b5d77d98012cbd0b4f5f08351523a4a1b58d85f7f32691bf63ea8",
    ),
}

STRUCTURED_COMPARISON: dict[str, dict[str, str]] = {
    "BitNet b1.58 / b1.58 2B4T": {
        "source_or_teacher_precision": "none: native ternary training from scratch",
        "architecture_class": "DENSE",
        "largest_evaluated_scale": "3.9B experiments; 2.4B released",
        "compression_regime": "NATIVE_LOW_BIT_TRAINING",
        "weight_activation_scope": "ternary linear weights plus per-token INT8 activations",
        "physical_accounting_level": "METHOD_RATE_ONLY",
    },
    "ParetoQ": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained checkpoints",
        "architecture_class": "DENSE",
        "largest_evaluated_scale": "Llama-3 8B",
        "compression_regime": "QAT",
        "weight_activation_scope": "quantized decoder weights; embeddings/output remain floating",
        "physical_accounting_level": "PARTIAL_MODEL_SIZE",
    },
    "QuIP": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained OPT/Llama checkpoints",
        "architecture_class": "DENSE",
        "largest_evaluated_scale": "OPT 66B / Llama-2 70B",
        "compression_regime": "WEIGHT_ONLY_PTQ",
        "weight_activation_scope": "W2/3/4 A16",
        "physical_accounting_level": "QUANTIZED_MATRICES_ONLY",
    },
    "QuIP#": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained checkpoints",
        "architecture_class": "DENSE_AND_MOE",
        "largest_evaluated_scale": "Falcon 180B; Mixtral-8x7B appendix",
        "compression_regime": "WEIGHT_ONLY_PTQ_WITH_OPTIONAL_CALIBRATION_FT",
        "weight_activation_scope": "quantized weight matrices; activations floating",
        "physical_accounting_level": "QUANTIZED_MATRICES_ONLY",
    },
    "AQLM": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained checkpoints",
        "architecture_class": "DENSE_AND_MOE",
        "largest_evaluated_scale": "Llama-2 70B / Mixtral-8x7B",
        "compression_regime": "WEIGHT_ONLY_PTQ_WITH_OPTIONAL_KL_DISTILLATION",
        "weight_activation_scope": "additive-codebook weights; Mixtral router floating; activations floating",
        "physical_accounting_level": "QUANTIZED_MATRICES_ONLY",
    },
    "VPTQ": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained checkpoints",
        "architecture_class": "DENSE",
        "largest_evaluated_scale": "70B peer-reviewed",
        "compression_regime": "WEIGHT_ONLY_PTQ",
        "weight_activation_scope": "vector-quantized Transformer block weights; activations floating",
        "physical_accounting_level": "QUANTIZED_MATRICES_ONLY",
    },
    "GPTVQ": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained checkpoints",
        "architecture_class": "DENSE_AND_MOE",
        "largest_evaluated_scale": "70B / Mixtral-8x7B",
        "compression_regime": "WEIGHT_ONLY_PTQ_WITH_OPTIONAL_REFINEMENT",
        "weight_activation_scope": "vector-quantized weights; activations floating",
        "physical_accounting_level": "METHOD_FORMULA_ONLY",
    },
    "BiLLM": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained checkpoints",
        "architecture_class": "DENSE",
        "largest_evaluated_scale": "70B",
        "compression_regime": "WEIGHT_ONLY_PTQ",
        "weight_activation_scope": "binary weights; activations floating",
        "physical_accounting_level": "REACCOUNTED_DECODED_PAYLOAD_ONLY",
    },
    "STBLLM": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained checkpoints",
        "architecture_class": "DENSE",
        "largest_evaluated_scale": "70B",
        "compression_regime": "SPARSE_BINARY_WEIGHT_ONLY_PTQ",
        "weight_activation_scope": "N:M sparse binary weights; activations floating",
        "physical_accounting_level": "NOMINAL_RATE_WITH_KERNEL_FLOOR",
    },
    "QMoE": {
        "source_or_teacher_precision": "official BF16 SwitchTransformer checkpoints",
        "architecture_class": "MOE",
        "largest_evaluated_scale": "Switch c2048 1.6T",
        "compression_regime": "WEIGHT_ONLY_PTQ_PLUS_LOSSLESS_ENTROPY_CODING",
        "weight_activation_scope": "ternary experts; BF16 non-experts; activations floating",
        "physical_accounting_level": "CANONICAL_COMPLETE_ARTIFACT",
    },
    "BTC-LLM": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained checkpoints",
        "architecture_class": "DENSE",
        "largest_evaluated_scale": "Llama 65B",
        "compression_regime": "GRADIENT_OPTIMIZED_WEIGHT_ONLY_PTQ",
        "weight_activation_scope": "binary-codebook weights; activations floating",
        "physical_accounting_level": "INCOMPLETE_REPORTED_MEMORY",
    },
    "NanoQuant": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained dense checkpoints",
        "architecture_class": "DENSE",
        "largest_evaluated_scale": "70B",
        "compression_regime": "WEIGHT_ONLY_PTQ_WITH_SCALE_KL_DISTILLATION",
        "weight_activation_scope": "binary-factor weights and FP16 scales; activations floating",
        "physical_accounting_level": "WHOLE_MODEL_DECODED_PAYLOAD_NOT_CANONICAL_FILE",
    },
    "LittleBit": {
        "source_or_teacher_precision": "NOT_REPORTED_BY_PRIMARY_SOURCE; floating pretrained dense checkpoints",
        "architecture_class": "DENSE",
        "largest_evaluated_scale": "32B executed; 70B estimated only",
        "compression_regime": "QAT",
        "weight_activation_scope": "binary-factor decoder weights; FP16 embeddings/head; activations floating",
        "physical_accounting_level": "WHOLE_MODEL_DECODED_PAYLOAD_NOT_CANONICAL_FILE",
    },
}


def _source(title: str, url: str, *, kind: str = "paper", commit: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"title": title, "url": url, "kind": kind}
    if kind == "paper":
        if title not in PAPER_PINS:
            raise ValueError(f"paper source lacks a content pin: {title}")
        pdf_url, digest = PAPER_PINS[title]
        row["content_identity"] = {
            "format": "PDF",
            "retrieval_url": pdf_url,
            "sha256": digest,
            "captured_at_cutoff": CUTOFF,
        }
    if commit:
        row["commit"] = commit
    return row


ROWS: list[dict[str, Any]] = [
    {
        "method": "BitNet b1.58 / b1.58 2B4T",
        "sources": [
            _source("The Era of 1-bit LLMs", "https://arxiv.org/abs/2402.17764"),
            _source("BitNet b1.58 2B4T", "https://arxiv.org/abs/2504.12285"),
            _source("microsoft/BitNet", "https://github.com/microsoft/BitNet/tree/16da220ae2b510caff437d403288882687f44ae5", kind="code", commit="16da220ae2b510caff437d403288882687f44ae5"),
        ],
        "method_class": "native ternary architecture trained from scratch",
        "training": "2B4T uses 4T-token pretraining plus SFT/DPO; not post-training compression",
        "scope": "dense 0.7B-3.9B experiments; released 2.4B; ternary linear weights and per-token INT8 activations",
        "rate": {
            "nominal_or_method_bpw": 1.5849625007,
            "nominal_meaning": "log2(3), not a stored-file rate",
            "decoded_tensor_payload_bpw": None,
            "canonical_artifact_bpw": None,
            "physical_boundary": "GPU format packs four trits per byte: 2 physical bits per linear weight before higher-precision embeddings/other tensors",
        },
        "evaluation": "broad peer-size general, math, code, instruction and dialogue tests; no floating-teacher preservation contract",
        "runtime": "CUDA kernels and bitnet.cpp CPU runtime",
        "reproducibility": "public weights and code",
        "glm_comparability": "architecture context only: dense 2.4B native pretraining, not GLM PTQ",
    },
    {
        "method": "ParetoQ",
        "sources": [
            _source("ParetoQ", "https://arxiv.org/abs/2502.02631"),
            _source("facebookresearch/ParetoQ", "https://github.com/facebookresearch/ParetoQ/tree/7b36b8de958aada6508e620c8dc544e6ed6c39b4", kind="code", commit="7b36b8de958aada6508e620c8dc544e6ed6c39b4"),
        ],
        "method_class": "QAT initialized from floating pretrained checkpoints",
        "training": "up to 30B QAT tokens at <=2 bits on 16 GPUs",
        "scope": "dense MobileLLM 125M-1.5B and Llama-3 1/3/8B; embeddings/output stay floating",
        "rate": {"nominal_or_method_bpw": [1, 1.58, 2, 3, 4], "decoded_tensor_payload_bpw": None, "canonical_artifact_bpw": None, "physical_boundary": "effective model size includes embeddings but not complete metadata/container accounting; ternary generally requires 2 physical bits"},
        "evaluation": "WikiText-2 and eight zero-shot commonsense tasks",
        "runtime": "preliminary CPU 2-bit kernel",
        "reproducibility": "public code; paper results used internal Meta infrastructure",
        "glm_comparability": "useful below-3-bit representation evidence; not PTQ, MoE, complete storage, or GLM scale",
    },
    {
        "method": "QuIP",
        "sources": [
            _source("QuIP", "https://arxiv.org/abs/2307.13304"),
            _source("Cornell-RelaxML/QuIP", "https://github.com/Cornell-RelaxML/QuIP/tree/ac92cfc7a22f6100009e2caf53bb72257d3f3184", kind="code", commit="ac92cfc7a22f6100009e2caf53bb72257d3f3184"),
        ],
        "method_class": "training-free Hessian-aware weight-only PTQ with incoherence and LDLQ",
        "training": "calibration/Hessian computation, no model retraining",
        "scope": "dense OPT 125M-66B and Llama-2 70B; W2/3/4A16",
        "rate": {"nominal_or_method_bpw": [2, 3, 4], "decoded_tensor_payload_bpw": None, "canonical_artifact_bpw": None, "physical_boundary": "transforms/scales/unquantized tensors/packaging are not fully accounted"},
        "evaluation": "WikiText/PTB/C4 perplexity and several zero-shot tasks",
        "runtime": "OPT-66B A6000 about 81 ms/token versus 53 ms for paper's OPTQ comparison",
        "reproducibility": "public code",
        "glm_comparability": "dense >=2-bit NVIDIA PTQ reference only",
    },
    {
        "method": "QuIP#",
        "sources": [
            _source("QuIP#", "https://arxiv.org/abs/2402.04396"),
            _source("Cornell-RelaxML/quip-sharp", "https://github.com/Cornell-RelaxML/quip-sharp/tree/1d8f873e9a2a8b86b12bb1064c312c5689b77d98", kind="code", commit="1d8f873e9a2a8b86b12bb1064c312c5689b77d98"),
        ],
        "method_class": "Hadamard incoherence plus E8 lattice codebooks",
        "training": "main results use calibration fine-tuning; no-FT variant exists",
        "scope": "dense Llama 7B-70B; appendix Falcon-180B and no-FT Mixtral-8x7B",
        "rate": {"nominal_or_method_bpw": [2, 3, 4], "decoded_tensor_payload_bpw": None, "canonical_artifact_bpw": None, "physical_boundary": "codebook/sign overhead <0.01 BPW per quantized matrix, but lm-head/norms/floating tensors excluded"},
        "evaluation": "perplexity, five zero-shot tasks, limited chat generations",
        "runtime": "RTX 4090 proof-of-concept 32.74 tok/s for Llama-2-70B",
        "reproducibility": "public code and models",
        "glm_comparability": "closest established extreme lattice/VQ family with a MoE appendix, but only 2-bit and incomplete accounting",
    },
    {
        "method": "AQLM",
        "sources": [
            _source("AQLM", "https://arxiv.org/abs/2401.06118"),
            _source("Vahe1994/AQLM", "https://github.com/Vahe1994/AQLM/tree/e79a896ed6656fe4ed06193d42d004e7d0bbdbb2", kind="code", commit="e79a896ed6656fe4ed06193d42d004e7d0bbdbb2"),
        ],
        "method_class": "additive multi-codebook weight-only PTQ",
        "training": "beam/block refinement; optional 4M-16M-token full-model KL distillation",
        "scope": "dense Llama-2 7/13/70B, Mistral-7B, Mixtral-8x7B; Mixtral gate unquantized",
        "rate": {"nominal_or_method_bpw": [1.97, 2.07], "decoded_tensor_payload_bpw": None, "canonical_artifact_bpw": None, "physical_boundary": "codes, FP16 codebooks and scales for quantized matrices only; floating parameters/artifact overhead excluded"},
        "evaluation": "perplexity, zero-shot tasks, MMLU and GSM8K appendix",
        "runtime": "70B quantization 10-14 days on one A100 or 3-4 days on eight; NVIDIA/CPU inference kernels",
        "reproducibility": "public code and models",
        "glm_comparability": "MoE/PTQ relevant, but not sub-1-bit and far smaller",
    },
    {
        "method": "VPTQ",
        "sources": [
            _source("VPTQ", "https://aclanthology.org/2024.emnlp-main.467/"),
            _source("microsoft/VPTQ", "https://github.com/microsoft/VPTQ/tree/942c3151026c26a5fae62807c65c630ff19e3893", kind="code", commit="942c3151026c26a5fae62807c65c630ff19e3893"),
        ],
        "method_class": "second-order vector PTQ with residual/outlier codebooks",
        "training": "layer/end-to-end calibration refinement",
        "scope": "peer-reviewed dense Llama-2/3 and Mistral through 70B; later repo scale claims are not equivalent paper evidence",
        "rate": {"nominal_or_method_bpw": [2.02, 2.26], "decoded_tensor_payload_bpw": None, "canonical_artifact_bpw": None, "physical_boundary": "Transformer-block indices/codebooks/scales only; embeddings/head/nonlinear tensors/packaging excluded"},
        "evaluation": "perplexity and five zero-shot tasks",
        "runtime": "four A100s: about 2 h at 7B and 19 h at 70B",
        "reproducibility": "public CUDA/Triton tools; community models carry a caveat",
        "glm_comparability": "dense >=2-bit paper evidence; repo-scale claims are unmatched",
    },
    {
        "method": "GPTVQ",
        "sources": [
            _source("GPTVQ", "https://arxiv.org/abs/2402.15319"),
            _source("Qualcomm-AI-research/gptvq", "https://github.com/Qualcomm-AI-research/gptvq/tree/fd8a3ced81f767018a28a51a91af96777f885f9f", kind="code", commit="fd8a3ced81f767018a28a51a91af96777f885f9f"),
        ],
        "method_class": "Hessian-aware vector PTQ with EM codebooks",
        "training": "optional centroid or LoRA refinement",
        "scope": "dense Llama/Mistral through 70B plus Mixtral-8x7B",
        "rate": {"nominal_or_method_bpw": [2.125, 4.125], "decoded_tensor_payload_bpw": None, "canonical_artifact_bpw": None, "physical_boundary": "formula includes index/quantized-codebook overhead but not a complete model denominator or artifact"},
        "evaluation": "WikiText and five zero-shot tasks; optional GSM8K LoRA",
        "runtime": "one H100 roughly 3-11 h for 70B; Snapdragon proof on Llama-3-8B",
        "reproducibility": "algorithm public; headline mobile engine not fully released",
        "glm_comparability": "PTQ/MoE relevance, but >=2-bit and non-Apple",
    },
    {
        "method": "BiLLM",
        "sources": [
            _source("BiLLM", "https://arxiv.org/abs/2402.04291"),
            _source("Aaronhuang-778/BiLLM", "https://github.com/Aaronhuang-778/BiLLM/tree/dc137ebbf62d4b31e8a82ba6bf9e18a51a298dcb", kind="code", commit="dc137ebbf62d4b31e8a82ba6bf9e18a51a298dcb"),
        ],
        "method_class": "training-free Hessian/saliency binary PTQ",
        "training": "no retraining",
        "scope": "dense OPT 1.3B-66B and Llama/Vicuna 7B-70B",
        "rate": {"nominal_or_method_bpw": [1.07, 1.13], "decoded_tensor_payload_bpw": 2.88, "canonical_artifact_bpw": None, "physical_boundary": "headline is arithmetic weight rate; checkpoint ratios imply 1.5-1.9, while NanoQuant complete-decode re-accounting gives ~2.88 due to scales/means/bitmaps"},
        "evaluation": "perplexity, seven zero-shot tasks, dialogue examples",
        "runtime": "no optimized end-to-end binary inference kernel",
        "reproducibility": "public code",
        "glm_comparability": "warning that 1-bit arithmetic is not 1 physical BPW; dense only",
    },
    {
        "method": "STBLLM",
        "sources": [
            _source("STBLLM", "https://openreview.net/forum?id=6XUSDvBFkV"),
            _source("pprp/STBLLM", "https://github.com/pprp/STBLLM/tree/6fe628759852ffb993ad1113577ef1f118ef9a2c", kind="code", commit="6fe628759852ffb993ad1113577ef1f118ef9a2c"),
        ],
        "method_class": "training-free N:M sparse binary PTQ",
        "training": "post-training; no MoE support",
        "scope": "dense OPT/Llama/Mistral through 70B",
        "rate": {"nominal_or_method_bpw": [0.53, 0.85], "decoded_tensor_payload_bpw": {"2_of_4_kernel_floor_before_metadata": 1.5, "nanoquant_reaccount_4_of_8": 3.50, "nanoquant_reaccount_6_of_8": 4.00, "nanoquant_reaccount_8_of_8": 4.13}, "canonical_artifact_bpw": None, "physical_boundary": "own 2:4 kernel stores six bits per four original weights before scales/group metadata"},
        "evaluation": "perplexity and seven zero-shot tasks; no strong free-form suite",
        "runtime": "NVIDIA sparse tensor-core layer benchmark; not end-to-end generation",
        "reproducibility": "public code",
        "glm_comparability": "not MoE and not physically sub-1-bit",
    },
    {
        "method": "QMoE",
        "sources": [
            _source("QMoE: Sub-1-Bit Compression of Trillion-Parameter Models", "https://proceedings.mlsys.org/paper_files/paper/2024/file/c74b624843218d9b6713fcf299d6d5e4-Paper-Conference.pdf"),
            _source("IST-DASLab/qmoe", "https://github.com/IST-DASLab/qmoe/tree/9110baa9466f2a7d8590e3c5dc3a5e11f7446604", kind="code", commit="9110baa9466f2a7d8590e3c5dc3a5e11f7446604"),
        ],
        "method_class": "retraining-free GPTQ-style expert PTQ plus lossless dictionary/entropy encoding",
        "training": "post-training; one A6000, about 16 h for c2048",
        "scope": "official BF16 SwitchTransformer base/large-128 and c2048 1.6T; ternary experts, BF16 non-experts",
        "rate": {"nominal_or_method_bpw": 0.8, "decoded_tensor_payload_bpw": None, "canonical_artifact_bpw": 0.807, "source_gb": 3142.0, "artifact_gb": 158.6, "compression_ratio": 19.81, "physical_boundary": "paper states full model plus all metadata; expert natural sparsity 88.6%"},
        "evaluation": "C4 masked-LM validation loss plus Arxiv/GitHub/StackExchange/Wikipedia losses; no modern autoregressive capability suite",
        "runtime": "direct compressed inference on 4xA6000 or 8xRTX3090, under 5% over idealized BF16 lower bound",
        "reproducibility": "public compressed checkpoints, code and CUDA kernel",
        "glm_comparability": "critical closest prior: giant BF16-source MoE, complete sub-1 artifact and direct runtime; materially different encoder-decoder MLM, unusual zero-heavy experts, and shallow capability evidence",
        "closest_giant_moe_prior": True,
    },
    {
        "method": "BTC-LLM",
        "sources": [
            _source("BTC-LLM", "https://aclanthology.org/2026.acl-long.1066/"),
            _source("Chooovy/BTC-LLM", "https://github.com/Chooovy/BTC-LLM/tree/a265b9acdebd8156e545e6a16427a8635c2e7095", kind="code", commit="a265b9acdebd8156e545e6a16427a8635c2e7095"),
        ],
        "method_class": "binary-codebook PTQ with learnable invertible transform",
        "training": "block gradient optimization; not gradient-free",
        "scope": "dense Llama 7B-65B, Qwen and FBI-LLM",
        "rate": {"nominal_or_method_bpw": [0.7, 1.11], "decoded_tensor_payload_bpw": None, "canonical_artifact_bpw": None, "reported_llama2_7b_model_memory_gb_at_0_7": 0.65, "physical_boundary": "formula omits required row scales/biases and floating embeddings/head; reported memory is not an audited artifact"},
        "evaluation": "WikiText and seven zero-shot tasks",
        "runtime": "preliminary H800 layer kernel, not a matched end-to-end runtime",
        "reproducibility": "code repository at cutoff contains only a nine-byte README; no algorithm/kernels/checkpoints",
        "glm_comparability": "peer-reviewed sub-1 claim but accounting/reproducibility insufficient for physical comparison",
    },
    {
        "method": "NanoQuant",
        "sources": [
            _source("NanoQuant", "https://arxiv.org/abs/2602.06694"),
            _source("SamsungLabs/NanoQuant", "https://github.com/SamsungLabs/NanoQuant/tree/a9e0a430881ff80d83b622c3129e330dc33c04f5", kind="code", commit="a9e0a430881ff80d83b622c3129e330dc33c04f5"),
        ],
        "method_class": "low-rank binary-factor PTQ with ADMM and scale-only KL distillation",
        "training": "128x2048 calibration tokens, eight-epoch block reconstruction, one H100",
        "scope": "dense Llama/Qwen/Gemma/Rnj-1 0.6B-70B; no MoE support listed",
        "rate": {"nominal_or_method_bpw": [1.0, 0.8, 0.55], "decoded_tensor_payload_bpw": 0.667, "canonical_artifact_bpw": None, "llama2_70b_source_gb": 137.95, "llama2_70b_payload_gb_at_0_55": 5.75, "physical_boundary": "nominal rate covers decoder-block linear factors plus FP16 row/column scales; whole-model number is memory/payload, not canonical file audit"},
        "evaluation": "perplexity, six zero-shot tasks and limited qualitative generation",
        "runtime": "one H100, about 13 h at 70B; RTX3050 20.11 tok/s and 5.86 GB peak for nominal 0.55",
        "reproducibility": "open CUDA quantization/GEMV/GEMM code; no official prequantized collection identified",
        "glm_comparability": "strongest dense PTQ comparator; dense 70B, gradient-calibrated, CUDA-only and not artifact-audited",
    },
    {
        "method": "LittleBit",
        "sources": [
            _source("LittleBit", "https://arxiv.org/abs/2506.13771"),
            _source("SamsungLabs/LittleBit", "https://github.com/SamsungLabs/LittleBit/tree/933857ed1443b53fc43a875c2cf64249e3c56f0c", kind="code", commit="933857ed1443b53fc43a875c2cf64249e3c56f0c"),
        ],
        "method_class": "low-rank binary-factor QAT",
        "training": "about 1B tokens; typically 4xH100, QwQ-32B 32xA100",
        "scope": "dense OPT/Llama/Phi/QwQ evaluated through 32B; embeddings/lm-head FP16",
        "rate": {"nominal_or_method_bpw": [0.1, 1.0], "decoded_tensor_payload_bpw": 0.927, "canonical_artifact_bpw": None, "llama2_13b_source_gb": 26.06, "llama2_13b_payload_gb_at_nominal_0_55": 1.51, "physical_boundary": "nominal covers Transformer factors/scales; 70B figures are estimates and 70B QAT was not run"},
        "evaluation": "perplexity, seven zero-shot tasks and qualitative generations showing degradation at extremes",
        "runtime": "A100 custom kernel; Llama-2-7B nominal 0.1 reports 203.2 tok/s versus 82.6 FP16",
        "reproducibility": "public training/evaluation code",
        "glm_comparability": "important sub-1 QAT representation reference; not one-pass PTQ, MoE, or GLM scale",
    },
]


def build() -> dict[str, Any]:
    if len({row["method"] for row in ROWS}) != len(ROWS):
        raise ValueError("duplicate external baseline method")
    if set(STRUCTURED_COMPARISON) != {row["method"] for row in ROWS}:
        raise ValueError("structured comparison rows do not exactly match methods")
    comparison_keys = {
        "source_or_teacher_precision",
        "architecture_class",
        "largest_evaluated_scale",
        "compression_regime",
        "weight_activation_scope",
        "physical_accounting_level",
    }
    methods: list[dict[str, Any]] = []
    for row in ROWS:
        comparison = STRUCTURED_COMPARISON[row["method"]]
        if set(comparison) != comparison_keys or any(
            not isinstance(value, str) or not value for value in comparison.values()
        ):
            raise ValueError(f"invalid structured comparison: {row['method']}")
        if comparison["architecture_class"] not in {"DENSE", "MOE", "DENSE_AND_MOE"}:
            raise ValueError(f"invalid architecture class: {row['method']}")
        for source in row["sources"]:
            if source["kind"] == "paper":
                digest = source.get("content_identity", {}).get("sha256")
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise ValueError(f"paper content is not pinned: {source['title']}")
            elif source["kind"] == "code":
                if re.fullmatch(r"[0-9a-f]{40}", source.get("commit", "")) is None:
                    raise ValueError(f"code commit is not pinned: {source['title']}")
            else:
                raise ValueError(f"unsupported source kind: {source['kind']}")
        methods.append({**row, "structured_comparison": comparison})
    return seal({
        "schema": "hawking.gravity_external_baseline_matrix.v1",
        "status": "PASS_PRIMARY_SOURCE_COMPARISON",
        "research_cutoff": CUTOFF,
        "source_policy": (
            "Primary paper PDFs are content-hashed; official code repositories are "
            "commit-pinned. No secondary result is used for a method row."
        ),
        "instrument_binding": {
            "generator": "tools/condense/glm52_external_baselines.py",
            "generator_sha256": sha256_file(Path(__file__)),
            "common_sha256": sha256_file(REPO_ROOT / "tools/condense/glm52_common.py"),
            "repository_base_commit": "753c73dc0685ce470090aba3e49c62fe4a4f9b08",
            "timestamp_free_deterministic_rebuild": True,
        },
        "rate_taxonomy": {
            "nominal_or_method_bpw": "Index/arithmetic/quantized-matrix rate claimed by the method.",
            "decoded_tensor_payload_bpw": "All tensors required to reconstruct or infer, divided by original logical weights.",
            "canonical_artifact_bpw": "Actual checkpoint bytes including all tensors, scales, codebooks, bitmaps, offsets, padding, headers, manifest and configuration.",
            "ranking_rule": "Rates from different taxonomy levels are not ranked as equal evidence.",
        },
        "accounting_verdict": "QMoE is the only surveyed method clearly publishing a full giant-MoE checkpoint plus all metadata below one BPW (0.807). NanoQuant and LittleBit publish informative whole-model payload/memory figures but not canonical artifact audits.",
        "methods": methods,
        "secondary_context": [
            {"method": "ARB-LLM", "paper": "https://openreview.net/forum?id=ZU8OdDLTts", "code": "https://github.com/ZHITENGLI/ARB-LLM/tree/55701970b9d881238b45bdf3cc6daf0a96a89f7e", "nominal": "binary PTQ", "decoded_complete_estimate_bpw": "2.50-2.52 per NanoQuant"},
            {"method": "HBLLM", "paper": "https://papers.nips.cc/paper_files/paper/2025/hash/e1b45fc5715c2b2ba878ea744b5ef267-Abstract-Conference.html", "nominal_bpw": 1.08, "decoded_complete_estimate_bpw": 3.25},
        ],
        "claim_policy": {
            "safe": [
                "Gravity targets a fully auditable canonical-artifact rate below 1 BPW for official GLM-5.2 BF16 weights.",
                "QMoE is the closest prior giant-MoE baseline and already demonstrates a fully metadata-accounted 0.807-BPW checkpoint; Gravity evaluates a materially different autoregressive GLM architecture and capability regime.",
                "NanoQuant and LittleBit establish dense low-rank binary references under different training budgets, scales, hardware, and denominators.",
                "External results are contextual baselines, not a cross-paper leaderboard.",
            ],
            "unsafe": [
                "First sub-1-bit PTQ.",
                "First fully accounted giant-MoE sub-1-bit checkpoint.",
                "Better than QMoE, NanoQuant, or LittleBit without matched model, tokenizer, harness, denominator, and hardware.",
                "Treating BitNet 1.58, STBLLM 0.55, BTC index rate, or quantized-matrix BPW as complete physical artifact BPW.",
            ],
            "defensible_novelty_if_achieved": "The conjunction of official GLM-5.2 BF16 source, 753B-class autoregressive MoE, complete canonical artifact under 1 BPW, auditable metadata, capability-preserving generation evaluation, and measured Apple-local execution—not sub-1-bit compression by itself.",
        },
    })


def markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Gravity external baseline matrix",
        "",
        f"Research cutoff: `{matrix['research_cutoff']}`. Primary sources only.",
        "",
        matrix["accounting_verdict"],
        "",
        "| Method | Class / scope | Rate boundary | GLM-5.2 comparison |",
        "|---|---|---|---|",
    ]
    for row in matrix["methods"]:
        source = row["sources"][0]
        rate = row["rate"]
        rate_text = (
            f"nominal `{rate.get('nominal_or_method_bpw')}`; "
            f"decoded `{rate.get('decoded_tensor_payload_bpw')}`; "
            f"canonical `{rate.get('canonical_artifact_bpw')}`"
        )
        lines.append(
            f"| [{row['method']}]({source['url']}) | {row['method_class']}; {row['scope']} | "
            f"{rate_text} | {row['glm_comparability']} |"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "QMoE prevents any honest first/only claim for complete sub-one-bit giant-MoE PTQ. "
        "The potential contribution is the autoregressive GLM/capability/Apple-local conjunction.",
        "",
        f"Seal: `{matrix['seal_sha256']}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    matrix = build()
    atomic_json(REPO_ROOT / "GRAVITY_EXTERNAL_BASELINE_MATRIX.json", matrix)
    atomic_text(REPO_ROOT / "GRAVITY_EXTERNAL_BASELINE_MATRIX.md", markdown(matrix))
    print(json.dumps({
        "status": matrix["status"],
        "methods": len(matrix["methods"]),
        "closest_prior": next(row["method"] for row in matrix["methods"] if row.get("closest_giant_moe_prior")),
        "seal_sha256": matrix["seal_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
