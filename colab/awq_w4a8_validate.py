#!/usr/bin/env python3
"""AWQ + W4A8 fake-quantization validator.

Validates that the AWQ smoothing factors in ``profiles/qwen3b_awq_smoothing.json``
actually help W4A8 quantization quality BEFORE the smoothing tensors get wired
into dismantle's Rust loader.

Protocol per prompt:
  1. Baseline: fp16 Qwen-3B forward + greedy decode N tokens.
  2. W4A8 + AWQ: apply per-channel smoothing at load time
       (W' = W * f, runtime x' = x / f, math-equivalent in fp),
     then fake-quantize each Linear's weight to int4 (per-output-channel
     scale) and each Linear's input activation to int8 (per-tensor scale)
     — round-and-dequantize back to fp16. Forward + greedy decode N tokens.
  3. W4A8 without AWQ: same fake quant, NO smoothing. Forward + greedy decode.

Metrics, per condition (with-awq, without-awq):
  - Per-token KL divergence from baseline logits (mean + p95).
  - Greedy match rate at first-{8,16,32} tokens.

A positive AWQ signal looks like:
  with_awq.greedy_match_first_32 > without_awq.greedy_match_first_32
  AND
  with_awq.mean_kl_per_token   < without_awq.mean_kl_per_token

The 7 AWQ sites match what ``colab/mega_calibrate.py`` captures:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

Single file, pure PyTorch + numpy + transformers. No awq_ext / auto-gptq
dependency. Designed to run on Colab A100 (fp16) or T4/V100/L4 (--load-4bit).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# AWQ site names — same 7 sites the smoothing JSON has factors for.
SITE_NAMES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# 30 diverse prompts: chat, code, math, general. Hardcoded so the report is
# reproducible without a corpus dependency.
DEFAULT_PROMPTS: Tuple[str, ...] = (
    # chat (10)
    "Explain what a black hole is in two sentences.",
    "What is the capital of France?",
    "Recommend a good book about climate change.",
    "Summarize the plot of Romeo and Juliet.",
    "How do I make sourdough bread at home?",
    "What is the meaning of life?",
    "Describe the taste of an avocado.",
    "Why is the sky blue?",
    "Tell me a short joke.",
    "What are three benefits of regular exercise?",
    # code (10)
    "Write a Python function that returns the nth Fibonacci number.",
    "What does the SQL keyword JOIN do?",
    "Implement bubble sort in Rust.",
    "How do I read a file line by line in Python?",
    "Write a regular expression that matches an email address.",
    "Explain the difference between a list and a tuple in Python.",
    "What is a closure in JavaScript?",
    "Write a CUDA kernel for vector addition.",
    "How do I reverse a linked list in C?",
    "Describe what 'async/await' does in Rust.",
    # math (5)
    "What is the derivative of x^2 * sin(x)?",
    "Solve for x: 3x + 7 = 22.",
    "What is the integral of 1/x from 1 to e?",
    "If a triangle has sides 3, 4, 5, what is its area?",
    "What is the value of pi to 6 decimal places?",
    # general (5)
    "List three planets in our solar system and one fact about each.",
    "Who wrote the novel 'Pride and Prejudice'?",
    "What year did World War II end?",
    "Name a country in South America that borders Brazil.",
    "What is photosynthesis?",
)


# ──────────────────────────────────────────────────────────────────────────────
# Fake quantization primitives.
# ──────────────────────────────────────────────────────────────────────────────


def fake_quant_weight_int4(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel int4 round-trip.

    Args:
        w: weight tensor, shape (out_features, in_features).
    Returns:
        Dequantized tensor of the same shape and dtype as ``w``.
    """
    if w.ndim != 2:
        raise ValueError(f"expected 2-D weight, got shape {tuple(w.shape)}")
    orig_dtype = w.dtype
    w32 = w.to(torch.float32)
    # Per-output-channel max-abs scale; clamp to int4 range [-8, 7].
    max_abs = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / 7.0
    q = torch.round(w32 / scale).clamp_(-8, 7)
    deq = q * scale
    return deq.to(orig_dtype)


def fake_quant_act_int8(x: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor int8 round-trip on the activation.

    Per-tensor (not per-token) keeps the math simple and matches dismantle's
    current W4A8 prototype, which uses per-tensor activation scales.
    """
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    max_abs = x32.abs().amax().clamp_min(1e-8)
    scale = max_abs / 127.0
    q = torch.round(x32 / scale).clamp_(-128, 127)
    deq = q * scale
    return deq.to(orig_dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Fake-quantized linear with optional AWQ smoothing.
# ──────────────────────────────────────────────────────────────────────────────


class FakeQuantLinear(torch.nn.Module):
    """Wrap an ``nn.Linear`` with W4 + A8 fake quant + optional AWQ smoothing.

    AWQ smoothing math (per-channel factor ``f`` on the input dim):

        W' = W * f                 (precomputed once, stored as fp16)
        x' = x / max(f, eps)       (applied at runtime to incoming activations)
        y  = (W' @ x'.T).T + b     (math-equivalent to W @ x.T + b)

    With fake quant layered on top:

        x' -> fake_quant_act_int8(x')
        W' -> fake_quant_weight_int4(W')  (precomputed at construction)
    """

    def __init__(
        self,
        linear: torch.nn.Linear,
        smoothing: Optional[torch.Tensor],
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.has_bias = linear.bias is not None
        device = linear.weight.device
        dtype = linear.weight.dtype

        weight = linear.weight.detach().clone()  # (out, in)
        if smoothing is not None:
            f = smoothing.to(device=device, dtype=torch.float32)
            if f.numel() != self.in_features:
                raise ValueError(
                    f"smoothing length {f.numel()} != in_features "
                    f"{self.in_features}"
                )
            f_safe = f.clamp_min(eps)
            # W' = W * f   (broadcast along in_dim)
            weight = (weight.to(torch.float32) * f_safe.unsqueeze(0)).to(dtype)
            self.register_buffer("recip_f", (1.0 / f_safe).to(dtype))
        else:
            self.register_buffer(
                "recip_f", torch.ones(self.in_features, dtype=dtype, device=device)
            )

        # Pre-quantize weight ONCE; we keep the dequantized fp16 result so the
        # forward path is a pure fp16 matmul against fake-quant weights.
        w_q = fake_quant_weight_int4(weight)
        self.register_buffer("weight_q", w_q)

        if self.has_bias:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

        # Whether to apply AWQ smoothing on the input at runtime. If smoothing
        # is None we still scale by 1.0 (no-op) — but we keep the flag so we
        # can flip behavior cheaply.
        self.smoothed = smoothing is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Apply x' = x / f on the input dim.
        x_smoothed = x * self.recip_f if self.smoothed else x
        # Fake-quantize the (smoothed) activation to int8.
        x_q = fake_quant_act_int8(x_smoothed)
        out = torch.nn.functional.linear(x_q, self.weight_q, self.bias)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Model surgery: find Linear children matching the 7 AWQ sites and swap them.
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_layer_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Return the per-layer module list for a Qwen2 / Llama-style HF model.

    The transformer blocks live under ``model.model.layers``.
    """
    obj = model
    for attr in ("model", "layers"):
        if not hasattr(obj, attr):
            raise RuntimeError(
                f"unexpected model structure: missing .{attr} "
                f"(got {type(obj).__name__})"
            )
        obj = getattr(obj, attr)
    if not isinstance(obj, (list, torch.nn.ModuleList)):
        raise RuntimeError(
            f"expected ModuleList at .model.layers, got {type(obj).__name__}"
        )
    return list(obj)


def _get_site_parent_and_name(
    layer: torch.nn.Module, site: str
) -> Tuple[torch.nn.Module, str]:
    """Locate the parent module and attribute name for a given site in a layer.

    q/k/v/o live under ``self_attn``; gate/up/down live under ``mlp``.
    """
    if site in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return layer.self_attn, site
    if site in ("gate_proj", "up_proj", "down_proj"):
        return layer.mlp, site
    raise ValueError(f"unknown site {site!r}")


def install_fake_quant(
    model: torch.nn.Module,
    smoothing: Optional[Dict[Tuple[int, str], torch.Tensor]],
) -> int:
    """Replace each (layer, site) Linear with a ``FakeQuantLinear``.

    Args:
        model: HF causal LM (Qwen2 layout).
        smoothing: dict ``{(layer_idx, site): tensor[in_features]}`` or None
            for "no smoothing" runs.

    Returns:
        Number of Linears replaced.
    """
    layers = _resolve_layer_modules(model)
    replaced = 0
    for li, layer in enumerate(layers):
        for site in SITE_NAMES:
            parent, name = _get_site_parent_and_name(layer, site)
            lin = getattr(parent, name, None)
            if not isinstance(lin, torch.nn.Linear):
                continue
            sm = None
            if smoothing is not None:
                sm = smoothing.get((li, site))
            fq = FakeQuantLinear(lin, sm)
            setattr(parent, name, fq)
            replaced += 1
    return replaced


# ──────────────────────────────────────────────────────────────────────────────
# Smoothing JSON loading.
# ──────────────────────────────────────────────────────────────────────────────


def load_smoothing(path: Path) -> Dict[Tuple[int, str], torch.Tensor]:
    """Load ``awq-smoothing-v1`` JSON into a per-(layer, site) tensor dict."""
    if not path.exists():
        raise FileNotFoundError(f"smoothing file not found: {path}")
    with path.open() as fh:
        data = json.load(fh)
    schema = data.get("schema")
    if schema != "awq-smoothing-v1":
        raise ValueError(
            f"expected schema 'awq-smoothing-v1', got {schema!r} at {path}"
        )
    out: Dict[Tuple[int, str], torch.Tensor] = {}
    for key, values in data.get("smoothing_factors", {}).items():
        # key = layer_{L}_{site}; site may contain underscores.
        if not key.startswith("layer_"):
            continue
        body = key[len("layer_"):]
        head, _, tail = body.partition("_")
        if not head.isdigit() or not tail:
            continue
        layer = int(head)
        if tail not in SITE_NAMES:
            continue
        out[(layer, tail)] = torch.tensor(values, dtype=torch.float32)
    if not out:
        raise ValueError(f"no usable smoothing factors found in {path}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Inference + metrics.
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def greedy_decode_with_logits(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[List[int], torch.Tensor]:
    """Greedy decode and return (generated_ids, stacked_logits).

    stacked_logits has shape (max_new_tokens, vocab_size) and contains, for
    each generation step, the next-token logits BEFORE the argmax is taken.
    Stored on CPU in fp32 to make KL / softmax stable.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    generated: List[int] = []
    logits_per_step: List[torch.Tensor] = []
    cur_ids = input_ids
    cur_mask = attention_mask
    past = None

    for _ in range(max_new_tokens):
        out = model(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            past_key_values=past,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]  # (1, vocab)
        logits_per_step.append(logits.detach().to("cpu", dtype=torch.float32))
        nxt = int(torch.argmax(logits, dim=-1).item())
        generated.append(nxt)
        past = out.past_key_values

        cur_ids = torch.tensor([[nxt]], device=device, dtype=input_ids.dtype)
        if cur_mask is not None:
            cur_mask = torch.cat(
                [cur_mask, torch.ones((1, 1), device=device, dtype=cur_mask.dtype)],
                dim=1,
            )

    stacked = torch.cat(logits_per_step, dim=0)  # (T, vocab)
    return generated, stacked


def kl_per_token(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) per token where P, Q are softmax of the given logits.

    Args:
        p_logits: (T, V) reference (baseline) logits.
        q_logits: (T, V) candidate logits.
    Returns:
        Tensor of shape (T,) with KL divergence per step in nats.
    """
    p_logp = torch.nn.functional.log_softmax(p_logits.to(torch.float32), dim=-1)
    q_logp = torch.nn.functional.log_softmax(q_logits.to(torch.float32), dim=-1)
    p = p_logp.exp()
    kl = (p * (p_logp - q_logp)).sum(dim=-1)
    # Numerical floor (KL is non-negative; allow tiny negative from fp noise).
    return kl.clamp_min(0.0)


def greedy_match_at(
    baseline_ids: List[int], cand_ids: List[int], k: int
) -> float:
    """Fraction of matching argmax tokens in the first ``k`` positions."""
    k = min(k, len(baseline_ids), len(cand_ids))
    if k == 0:
        return float("nan")
    matches = sum(1 for i in range(k) if baseline_ids[i] == cand_ids[i])
    return matches / float(k)


# ──────────────────────────────────────────────────────────────────────────────
# Driver.
# ──────────────────────────────────────────────────────────────────────────────


def _load_model(model_id: str, device: torch.device, load_4bit: bool):
    """Load a fresh copy of the model. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)

    kwargs: Dict[str, object] = {"attn_implementation": "sdpa"}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = {"": device.index if device.type == "cuda" else "cpu"}
    else:
        # transformers 5.0 renamed `torch_dtype` to `dtype`. Probe and prefer
        # the new name; fall back for older transformers builds that still
        # require `torch_dtype`.
        import transformers as _hf
        _hf_major = int(str(_hf.__version__).split(".", 1)[0])
        dtype_kw = "dtype" if _hf_major >= 5 else "torch_dtype"
        kwargs[dtype_kw] = torch.float16
        kwargs["device_map"] = (
            {"": device.index} if device.type == "cuda" else None
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not load_4bit and device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model, tok


def _aggregate(per_prompt: List[Dict[str, object]]) -> Dict[str, float]:
    """Aggregate KL and match metrics across prompts."""
    all_kl = np.concatenate(
        [np.asarray(p["kl"], dtype=np.float64) for p in per_prompt]
    )
    match_8 = np.mean([float(p["match_8"]) for p in per_prompt])
    match_16 = np.mean([float(p["match_16"]) for p in per_prompt])
    match_32 = np.mean([float(p["match_32"]) for p in per_prompt])
    return {
        "mean_kl_per_token": float(np.mean(all_kl)),
        "p95_kl_per_token": float(np.percentile(all_kl, 95.0)),
        "greedy_match_first_8": float(match_8),
        "greedy_match_first_16": float(match_16),
        "greedy_match_first_32": float(match_32),
    }


def _run_condition(
    model_id: str,
    device: torch.device,
    load_4bit: bool,
    smoothing: Optional[Dict[Tuple[int, str], torch.Tensor]],
    baseline_per_prompt: List[Tuple[List[int], torch.Tensor]],
    prompts: List[str],
    tokenizer,
    max_new_tokens: int,
    tag: str,
) -> Dict[str, object]:
    """Load a fresh model, install fake-quant (with/without smoothing), run."""
    print(f"[awq_w4a8_validate] loading model for condition '{tag}'", file=sys.stderr)
    model, _ = _load_model(model_id, device, load_4bit)
    replaced = install_fake_quant(model, smoothing)
    if replaced == 0:
        raise RuntimeError(
            f"condition '{tag}' replaced 0 Linears. This usually means the "
            "model was loaded through a quantized wrapper that the fake-quant "
            "validator cannot instrument. Re-run validation in fp16 on A100+ "
            "or skip this quality gate on small GPUs."
        )
    print(
        f"[awq_w4a8_validate] '{tag}': replaced {replaced} Linears with "
        f"FakeQuantLinear (expected 7 × n_layers).",
        file=sys.stderr,
    )

    per_prompt: List[Dict[str, object]] = []
    t0 = time.time()
    for i, (prompt, (base_ids, base_logits)) in enumerate(
        zip(prompts, baseline_per_prompt)
    ):
        cand_ids, cand_logits = greedy_decode_with_logits(
            model, tokenizer, prompt, max_new_tokens, device
        )
        kl = kl_per_token(base_logits, cand_logits).numpy().tolist()
        per_prompt.append(
            {
                "kl": kl,
                "match_8": greedy_match_at(base_ids, cand_ids, 8),
                "match_16": greedy_match_at(base_ids, cand_ids, 16),
                "match_32": greedy_match_at(base_ids, cand_ids, 32),
            }
        )
        if (i + 1) % 5 == 0 or (i + 1) == len(prompts):
            print(
                f"[awq_w4a8_validate] '{tag}': prompt {i + 1}/{len(prompts)} "
                f"in {time.time() - t0:.1f}s",
                file=sys.stderr,
            )

    # Drop the model to free VRAM before the next condition loads.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return _aggregate(per_prompt)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Validate AWQ smoothing factors against a W4A8 fake-quant baseline."
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument(
        "--smoothing",
        type=Path,
        default=Path("profiles/qwen3b_awq_smoothing.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/awq_w4a8_validation_2026_05_26.json"),
    )
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", type=str, choices=("cuda", "cpu"), default="cuda")
    p.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load base model in bitsandbytes nf4 to fit on T4/V100/L4. "
             "Default off (assumes A100+ with native fp16).",
    )
    args = p.parse_args()

    if args.n_prompts <= 0:
        print("[awq_w4a8_validate] ERROR: --n-prompts must be > 0", file=sys.stderr)
        return 2
    if args.max_new_tokens <= 0:
        print("[awq_w4a8_validate] ERROR: --max-new-tokens must be > 0", file=sys.stderr)
        return 2

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print(
            "[awq_w4a8_validate] WARNING: --device cuda requested but CUDA "
            "is unavailable; falling back to CPU.",
            file=sys.stderr,
        )

    prompts = list(DEFAULT_PROMPTS[: args.n_prompts])
    if len(prompts) < args.n_prompts:
        print(
            f"[awq_w4a8_validate] WARNING: only {len(prompts)} hardcoded "
            f"prompts available; clamping --n-prompts.",
            file=sys.stderr,
        )

    print(
        f"[awq_w4a8_validate] loading smoothing factors from {args.smoothing}",
        file=sys.stderr,
    )
    smoothing = load_smoothing(args.smoothing)
    print(
        f"[awq_w4a8_validate] loaded {len(smoothing)} (layer, site) factor "
        f"vectors.",
        file=sys.stderr,
    )

    # Pass 1: fp16 baseline — load fresh, run, capture logits, drop.
    print(
        "[awq_w4a8_validate] pass 1/3: fp16 baseline forward pass",
        file=sys.stderr,
    )
    model, tokenizer = _load_model(args.model, device, args.load_4bit)
    baseline_per_prompt: List[Tuple[List[int], torch.Tensor]] = []
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        ids, logits = greedy_decode_with_logits(
            model, tokenizer, prompt, args.max_new_tokens, device
        )
        baseline_per_prompt.append((ids, logits))
        if (i + 1) % 5 == 0 or (i + 1) == len(prompts):
            print(
                f"[awq_w4a8_validate] baseline: prompt {i + 1}/{len(prompts)} "
                f"in {time.time() - t0:.1f}s",
                file=sys.stderr,
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Pass 2: W4A8 + AWQ smoothing.
    print(
        "[awq_w4a8_validate] pass 2/3: W4A8 + AWQ smoothing",
        file=sys.stderr,
    )
    with_awq = _run_condition(
        args.model, device, args.load_4bit, smoothing,
        baseline_per_prompt, prompts, tokenizer, args.max_new_tokens, "with_awq",
    )

    # Pass 3: W4A8 without smoothing (naive symmetric int4 / int8).
    print(
        "[awq_w4a8_validate] pass 3/3: W4A8 WITHOUT AWQ smoothing",
        file=sys.stderr,
    )
    without_awq = _run_condition(
        args.model, device, args.load_4bit, None,
        baseline_per_prompt, prompts, tokenizer, args.max_new_tokens, "without_awq",
    )

    delta = {
        "kl_improvement": float(
            without_awq["mean_kl_per_token"] - with_awq["mean_kl_per_token"]
        ),
        "match_improvement": float(
            with_awq["greedy_match_first_32"]
            - without_awq["greedy_match_first_32"]
        ),
    }

    payload = {
        "schema": "awq-w4a8-validation-v1",
        "model": args.model,
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "with_awq": with_awq,
        "without_awq": without_awq,
        "delta": delta,
    }
    _write_json_atomic(args.out, payload)

    print(
        f"[awq_w4a8_validate] DONE. with_awq.mean_kl="
        f"{with_awq['mean_kl_per_token']:.4f} "
        f"without_awq.mean_kl={without_awq['mean_kl_per_token']:.4f} "
        f"(Δ={delta['kl_improvement']:+.4f}); "
        f"match@32 with={with_awq['greedy_match_first_32']:.3f} "
        f"without={without_awq['greedy_match_first_32']:.3f} "
        f"(Δ={delta['match_improvement']:+.3f}) → {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
