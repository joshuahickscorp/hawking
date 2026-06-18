"""Stateful batched RWKV-7 generation for CPU — no subprocess overhead, no GPU.

Adds generate_batch() on top of an existing RWKV7Model by implementing the
per-token stateful WKV-7 step directly. This lets 8 workers each hold one
fp32 model (~1.7 GB) and process B prompts in parallel per step.

State per layer: S [B,H,D,D] (WKV-7 recurrent matrix),
                 prev_attn [B,n] (prev attn-normed input for token shift),
                 prev_ffn  [B,n] (prev ffn-normed input for token shift).
v_first [B,n] is NOT temporal state — it is the layer-0 v projection recomputed
from scratch at every new token and passed across layers within that token's forward.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ln(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm over last dim for [B, n] tensors."""
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps) * w + b


# ---------------------------------------------------------------------------
# Per-token step functions — mirror RWKV7TimeMix / RWKV7ChannelMix forward
# but for a single timestep with explicit state I/O.
# ---------------------------------------------------------------------------

def _timemix_step(tm, x_t: torch.Tensor, x_prev: torch.Tensor,
                  S: torch.Tensor, v_first_in: torch.Tensor):
    """One token step through RWKV7TimeMix.

    Args:
        tm: RWKV7TimeMix module
        x_t:        [B, n]      current attn-normed embedding
        x_prev:     [B, n]      previous attn-normed embedding (token shift)
        S:          [B, H, D, D] WKV-7 recurrent state (IN)
        v_first_in: [B, n]      value-residual cross-layer carry (from layer 0)

    Returns: (out [B,n], S_new [B,H,D,D], v_first_out [B,n])
    """
    cfg = tm.cfg
    B, n = x_t.shape
    H, D = cfg.n_head, cfg.head_dim

    sx = x_prev - x_t
    xr = x_t + sx * tm.x_r
    xw = x_t + sx * tm.x_w
    xk = x_t + sx * tm.x_k
    xv = x_t + sx * tm.x_v
    xa = x_t + sx * tm.x_a
    xg = x_t + sx * tm.x_g

    r = tm.r_proj(xr)
    w = torch.tanh(xw @ tm.w1.T) @ tm.w2.T + tm.w0
    w = torch.exp(-0.606531 * torch.sigmoid(w))
    k = tm.k_proj(xk)
    v = tm.v_proj(xv)

    if tm.layer_idx == 0:
        v_first_out = v
    else:
        g_v = torch.sigmoid((xv @ tm.v1.T) @ tm.v2.T + tm.v0)
        v = v + (v_first_in - v) * g_v
        v_first_out = v_first_in

    g = torch.sigmoid(xg @ tm.g1.T) @ tm.g2.T
    a = torch.sigmoid((xa @ tm.a1.T) @ tm.a2.T + tm.a0)

    kk = (k * tm.k_k).view(B, H, D)
    kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    kk_flat = kk.view(B, n)
    k = k + (a - 1.0) * (k * tm.k_a)

    a_op = -kk_flat
    b_op = kk_flat * a

    rh = r.view(B, H, D)
    kh = k.view(B, H, D)
    vh = v.view(B, H, D)
    wh = w.view(B, H, D)
    ah = a_op.view(B, H, D)
    bh = b_op.view(B, H, D)

    sa = torch.einsum("bhij,bhj->bhi", S, ah)
    S_new = (
        S * wh.unsqueeze(2)
        + vh.unsqueeze(3) * kh.unsqueeze(2)
        + sa.unsqueeze(3) * bh.unsqueeze(2)
    )
    out_h = torch.einsum("bhij,bhj->bhi", S_new, rh)  # [B, H, D]

    # per-head group norm
    mean = out_h.mean(-1, keepdim=True)
    var = out_h.var(-1, keepdim=True, unbiased=False)
    out_h = (out_h - mean) * torch.rsqrt(var + cfg.gn_eps)
    out = out_h.reshape(B, n) * tm.g_norm_w + tm.g_norm_b

    # bonus: += v * sum_D(k * r * r_k)
    kr = (kh * rh * tm.r_k).sum(-1, keepdim=True)  # [B, H, 1]
    out = (out.view(B, H, D) + vh * kr).reshape(B, n)

    out = tm.o_proj(out * g)
    return out, S_new, v_first_out


def _channelmix_step(ffn, x_t: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
    """One token step through RWKV7ChannelMix."""
    xk = x_t + (x_prev - x_t) * ffn.x_k
    k = torch.relu(ffn.key(xk))
    return ffn.value(k * k)  # squared-relu FFN


def _block_step(block, hidden: torch.Tensor,
                prev_a: torch.Tensor, prev_f: torch.Tensor,
                S: torch.Tensor, v_first_in: torch.Tensor):
    """One token step through RWKV7Block.

    Returns: (hidden_out, prev_a_new, prev_f_new, S_new, v_first_out)
    where prev_a_new / prev_f_new are the attn/ffn-normed inputs that
    become the token-shift x_prev on the NEXT call.
    """
    eps = block.cfg.ln_eps

    if block.layer_idx == 0:
        residual = _ln(hidden, block.pre_norm_w, block.pre_norm_b, eps)
    else:
        residual = hidden

    attn_normed = _ln(residual, block.attn_norm_w, block.attn_norm_b, eps)
    attn_out, S_new, v_first_out = _timemix_step(block.attn, attn_normed, prev_a, S, v_first_in)
    residual = residual + attn_out

    ffn_normed = _ln(residual, block.ffn_norm_w, block.ffn_norm_b, eps)
    ffn_out = _channelmix_step(block.ffn, ffn_normed, prev_f)
    residual = residual + ffn_out

    return residual, attn_normed, ffn_normed, S_new, v_first_out


# ---------------------------------------------------------------------------
# Batched generate
# ---------------------------------------------------------------------------

def generate_batch(model, tok, user_prompts: List[str],
                   max_new_tokens: int, temperature: float, base_seed: int) -> List[str]:
    """Generate one rejection sample per prompt, B prompts in parallel on CPU.

    Args:
        model:          RWKV7Model (CPU, eval mode, no grad)
        tok:            RWKV_TOKENIZER
        user_prompts:   list of B user question strings
        max_new_tokens: cap on generated tokens
        temperature:    sampling temperature (0 = greedy)
        base_seed:      int, varied per sample/step for diversity

    Returns: list of B decoded strings (no leading space, truncated at 600 chars)
    """
    B = len(user_prompts)
    cfg = model.cfg
    H, D, n = cfg.n_head, cfg.head_dim, cfg.n_embd
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    EOS = 0

    # Tokenize prompts
    ids_list = [
        [EOS] + tok.encodeBytes(f"User: {u}\n\nAssistant:".encode("utf-8"))
        for u in user_prompts
    ]
    max_p = max(len(ids) for ids in ids_list)

    # Right-pad with EOS (token 0); mask tracks real tokens
    prompt_tensor = torch.zeros(B, max_p, dtype=torch.long, device=device)
    active_mask = torch.zeros(B, max_p, dtype=torch.bool, device=device)
    for b, ids in enumerate(ids_list):
        prompt_tensor[b, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        active_mask[b, :len(ids)] = True

    # Initialize state
    S       = [torch.zeros(B, H, D, D, dtype=dtype, device=device) for _ in range(cfg.n_layer)]
    prev_a  = [torch.zeros(B, n,       dtype=dtype, device=device) for _ in range(cfg.n_layer)]
    prev_f  = [torch.zeros(B, n,       dtype=dtype, device=device) for _ in range(cfg.n_layer)]
    v_first = torch.zeros(B, n, dtype=dtype, device=device)
    hidden  = torch.zeros(B, n, dtype=dtype, device=device)

    with torch.no_grad():
        # ---- Prompt encoding with masked state update ----
        for t in range(max_p):
            active = active_mask[:, t]  # [B] bool
            if not active.any():
                break

            token_t = prompt_tensor[:, t]  # [B]
            h = model.embeddings(token_t)  # [B, n]
            vf = torch.zeros_like(v_first)

            for i, block in enumerate(model.layers):
                h, pa_new, pf_new, S_new, vf = _block_step(block, h, prev_a[i], prev_f[i], S[i], vf)
                act1 = active.view(B, 1)
                act4 = active.view(B, 1, 1, 1)
                S[i]      = torch.where(act4,  S_new,  S[i])
                prev_a[i] = torch.where(act1, pa_new, prev_a[i])
                prev_f[i] = torch.where(act1, pf_new, prev_f[i])

            hidden  = torch.where(active.view(B, 1), h,  hidden)
            v_first = torch.where(active.view(B, 1), vf, v_first)

        # Final norm + lm_head → logits for the last real prompt token
        h_norm = _ln(hidden, model.norm_w, model.norm_b, cfg.ln_eps)
        logits = model.lm_head(h_norm)  # [B, vocab]

        # ---- Autoregressive generation ----
        generated: List[List[int]] = [[] for _ in range(B)]
        finished = [False] * B

        for g in range(max_new_tokens):
            if all(finished):
                break

            next_tokens: List[int] = []
            for b in range(B):
                if finished[b]:
                    next_tokens.append(EOS)
                    continue
                logit_b = logits[b]
                if temperature < 1e-6:
                    tok_id = int(logit_b.argmax())
                else:
                    probs = torch.softmax(logit_b / temperature, dim=-1)
                    torch.manual_seed(base_seed + b * 997 + g)
                    tok_id = int(torch.multinomial(probs, 1))
                if tok_id == EOS:
                    finished[b] = True
                else:
                    generated[b].append(tok_id)
                next_tokens.append(tok_id)

            if all(finished):
                break

            token_t = torch.tensor(next_tokens, dtype=torch.long, device=device)
            h = model.embeddings(token_t)
            vf = torch.zeros_like(v_first)

            for i, block in enumerate(model.layers):
                h, prev_a[i], prev_f[i], S[i], vf = _block_step(block, h, prev_a[i], prev_f[i], S[i], vf)

            h_norm = _ln(h, model.norm_w, model.norm_b, cfg.ln_eps)
            logits = model.lm_head(h_norm)

    results = []
    for b in range(B):
        try:
            text = tok.decodeBytes(generated[b]).decode("utf-8", errors="replace").split("\n\n")[0].strip()[:600]
        except Exception:
            text = ""
        results.append(text)
    return results


def load_tokenizer(hf_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "hf_rwkv_tokenizer", str(hf_dir / "hf_rwkv_tokenizer.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RWKV_TOKENIZER(str(hf_dir / "rwkv_vocab_v20230424.txt"))
