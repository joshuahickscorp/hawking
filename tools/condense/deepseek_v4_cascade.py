#!/usr/bin/env python3.12
"""The one question: does the functional student compose across the full DeepSeek model?

Single-layer perturbations said DeepSeek is favourable, not uniformly contractive. That does
not settle composition. This runs the real thing: a functional student in every sparse MoE
layer, propagated through the whole model on the validated streamed forward, measuring where
and whether the cascade diverges from the teacher.

Two streams share each layer's weights (loaded once): the teacher runs the real MoE, the
student cascade substitutes a per-layer functional student. Each layer's student is fitted
on that layer's own teacher MoE output (the official module's, captured by a hook), so the
students are trained on the clean teacher trajectory and applied to their own drifting
cascade -- which is exactly the composition question.

    C0  teacher forward + affine upper-control cascade
    C1  random-feature student in every sparse MoE layer

Per layer: hidden skill, amplification, router top-k agreement. At the head: greedy-token
agreement through the logit lens. No broad search.

    run [SEQ_LEN] [MODE]     MODE in {student, affine}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import deepseek_v4_moe as ds
import deepseek_v4_reference as ref
import hawking_null_metric as metric
import glm52_moe_student as gs

OUT = Path(__file__).resolve().parents[2] / "reports" / "condense" / "deepseek_v4_flash"
HASH_LAYERS = 3   # 0-2 hash-routed; the learned-routing MoE layers are 3..42
HIDDEN = 1024


def _collapsed(streams):
    return streams.mean(axis=2)


def _fit_student(pre_moe_np, moe_out_np, mode: str):
    import numpy as _np
    x = pre_moe_np.reshape(-1, ds.DIM).astype(np.float32)
    y = moe_out_np.reshape(-1, ds.DIM).astype(np.float32)
    if mode == "affine":
        a = _np.concatenate([x, _np.ones((x.shape[0], 1), np.float32)], axis=1)
        ridge = 1.0
        w = _np.linalg.solve(a.T @ a + ridge * _np.eye(a.shape[1], dtype=np.float32),
                             a.T @ y)
        return ("affine", w)
    fitted = gs.fit(x, y, hidden=HIDDEN, seed=17, replaced_weights=1)
    return ("student", fitted["blob"])


def _apply_student(model, x_np):
    kind, payload = model
    x = x_np.reshape(-1, ds.DIM).astype(np.float32)
    if kind == "affine":
        import numpy as _np
        a = _np.concatenate([x, _np.ones((x.shape[0], 1), np.float32)], axis=1)
        return a @ payload
    return gs.apply_student(payload, x)


def run(seq_len: int = 128, mode: str = "student") -> dict:
    import torch
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M

    config = ref._config()
    config._attn_implementation = "eager"
    index = ds._index()
    torch.set_grad_enabled(False)
    n_layers = config.num_hidden_layers

    tokens = np.random.default_rng(41).integers(0, 129280, seq_len, dtype=np.int64)
    embed_meta, embed_raw = ds._read("embed.weight", index)
    embed = ds._bf16(embed_raw, embed_meta["shape"])
    tok_t = torch.from_numpy(tokens).long().unsqueeze(0)
    hidden0 = torch.from_numpy(embed[tokens]).to(torch.bfloat16).unsqueeze(0)
    teacher = hidden0.unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
    student = teacher.clone()
    rotary = M.DeepseekV4RotaryEmbedding(config)
    pid = torch.arange(seq_len).unsqueeze(0)
    pe = {"main": rotary(hidden0, position_ids=pid, layer_type="main"),
          "compress": rotary(hidden0, position_ids=pid, layer_type="compress")}
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    call = dict(input_ids=tok_t, position_embeddings=pe, position_ids=pid,
                attention_mask=mask, past_key_values=None)

    def block_null(layer_module, streams_in):
        # A fit-split null for the block output: fit the mean on the teacher block output
        # of this layer. Reported per layer so skill is null-corrected, not raw.
        return None

    per_layer = []
    for L in range(n_layers):
        module = M.DeepseekV4DecoderLayer(config, L).to(torch.bfloat16).eval()
        ref._load_layer(L, module, index)
        sparse = L >= HASH_LAYERS

        # Teacher stream: capture this layer's real MoE input and output.
        cap = {}
        pre = module.mlp.register_forward_pre_hook(
            lambda m, a: cap.__setitem__("in", a[0].detach().float().numpy()))
        post = module.mlp.register_forward_hook(
            lambda m, i, o: cap.__setitem__("out", o.detach().float().numpy()))
        t_topk = {}
        if sparse:
            g = module.mlp.gate.register_forward_hook(
                lambda m, i, o: t_topk.__setitem__("idx", o[2].detach().numpy()))
        teacher_in = teacher.clone()
        teacher_out = module(teacher, **call)
        pre.remove(); post.remove()
        if sparse:
            g.remove()

        # Fit this layer's student on the teacher trajectory, then run the student stream
        # with the MoE substituted. The student pass replaces the whole SparseMoeBlock
        # forward with a light version: run only the gate (for router agreement) and the
        # student, skipping the 256 experts entirely, so the student stream is cheap.
        student_in = student.clone()
        if sparse:
            model = _fit_student(cap["in"], cap["out"], mode)
            sg = {}
            original_forward = module.mlp.forward

            def light_forward(hidden_states, input_ids=None, _mlp=module.mlp, _model=model):
                _, _, indices = _mlp.gate(hidden_states)
                sg["idx"] = indices.detach().numpy()
                pred = _apply_student(_model, hidden_states.detach().float().numpy())
                return torch.from_numpy(
                    pred.reshape(tuple(hidden_states.shape))).to(hidden_states.dtype)

            module.mlp.forward = light_forward
            student_out = module(student, **call)
            module.mlp.forward = original_forward
        else:
            student_out = module(student, **call)

        # Measure. Skill of the student block output against a fit-split null on the teacher
        # block output; amplification of the collapsed-hidden deviation.
        tc = _collapsed(teacher_out).float().numpy().reshape(-1, ds.DIM)
        sc = _collapsed(student_out).float().numpy().reshape(-1, ds.DIM)
        null = metric.fit_null(tc)
        skilled = metric.score(tc, sc, null)
        entering = float(np.linalg.norm(_collapsed(student_in).float().numpy()
                                        - _collapsed(teacher_in).float().numpy())
                         / max(np.linalg.norm(_collapsed(teacher_in).float().numpy()), 1e-9))
        leaving = float(np.linalg.norm(sc - tc) / max(np.linalg.norm(tc), 1e-9))
        row = {"layer": L, "sparse": sparse,
               "block_skill": skilled["skill"], "block_skill_lower": skilled["skill_lower"],
               "block_centered_cosine": skilled["centered_cosine"],
               "entering_relative_l2": entering, "leaving_relative_l2": leaving,
               "amplification": leaving / max(entering, 1e-9) if entering > 1e-6 else None}
        if sparse and "idx" in t_topk and "idx" in sg:
            ti = t_topk["idx"].reshape(-1, ds.N_ACTIVATED)
            si = sg["idx"].reshape(-1, ds.N_ACTIVATED)
            row["router_topk_overlap"] = float(np.mean(
                [len(set(a) & set(b)) for a, b in zip(ti, si)]))
            row["router_top1_agreement"] = float((ti[:, 0] == si[:, 0]).mean())
        per_layer.append(row)
        teacher, student = teacher_out, student_out
        del module

    # Head: logit-lens greedy-token agreement.
    token = _head_agreement(teacher, student, index, config)

    diverged = next((r["layer"] for r in per_layer
                     if r["sparse"] and r["block_skill_lower"] is not None
                     and r["block_skill_lower"] <= 0.0), None)
    result = {
        "schema": "hawking.deepseek_v4.cascade.v1",
        "parent": "deepseek-ai/DeepSeek-V4-Flash", "mode": mode, "seq_len": seq_len,
        "sparse_layers": [L for L in range(n_layers) if L >= HASH_LAYERS],
        "per_layer": per_layer,
        "final_leaving_relative_l2": per_layer[-1]["leaving_relative_l2"],
        "final_block_skill": per_layer[-1]["block_skill"],
        "first_divergence_layer": diverged,
        "mean_amplification_sparse": float(np.mean(
            [r["amplification"] for r in per_layer if r.get("amplification")])),
        "min_router_top1": float(np.min(
            [r["router_top1_agreement"] for r in per_layer if "router_top1_agreement" in r])),
        "token": token,
        "not_evidence_of": "capability; bounded tokens, in-sample per-layer student fits.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"DEEPSEEK_V4_CASCADE_{mode.upper()}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float))
    return result


def _head_agreement(teacher, student, index, config):
    import torch
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    # hc_head collapses the 4-wide stream; norm; lm_head over a bounded vocab subset.
    try:
        norm_w = ds._bf16(*[ds._read("norm.weight", index)][0][::-1]) if False else \
            ds._bf16(ds._read("norm.weight", index)[1], ds._read("norm.weight", index)[0]["shape"])
        head_meta, head_raw = ds._read("head.weight", index)
        rows = 2048
        head = ds._bf16(head_raw, head_meta["shape"])[:rows]

        def logits(streams):
            collapsed = streams.mean(axis=2).float().numpy().reshape(-1, ds.DIM)
            normed = ds.rmsnorm(collapsed, norm_w)
            return normed @ head.T

        t = logits(teacher).argmax(-1)
        s = logits(student).argmax(-1)
        return {"vocab_subset_rows": rows, "positions": int(t.shape[0]),
                "greedy_token_agreement": float((t == s).mean()),
                "note": "hc_head omitted (mean-collapse proxy); a 2048-row logit-lens, not the head"}
    except Exception as error:  # noqa: BLE001
        return {"error": repr(error)[:200]}


if __name__ == "__main__":
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    mode = sys.argv[2] if len(sys.argv) > 2 else "student"
    r = run(seq_len, mode)
    print(json.dumps({k: v for k, v in r.items() if k != "per_layer"}, indent=2, default=float))
    print("\nper-layer (sparse):")
    for row in r["per_layer"]:
        if row["sparse"]:
            print(f"  L{row['layer']:02d} skill {row['block_skill']:7.4f} "
                  f"amp {row['amplification'] if row['amplification'] else 0:6.3f} "
                  f"router_top1 {row.get('router_top1_agreement', float('nan')):.3f} "
                  f"leaving_rl2 {row['leaving_relative_l2']:.4f}")
