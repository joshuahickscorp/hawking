"""A measured, output-identical fusion of the MoE block's combine tail.

Stock `DeepseekV3MoE.__call__` issues the score-weighted combine and the
shared-expert add as separate launches per layer. The router itself is already
@mx.compile'd; this tail is not. Compiling it fuses those launches.

MEASURED on O003 (Kimi-VL-A3B-Instruct, q3-g128-experts, M3 Ultra, MLX):

    isolated MoE block   4.2563 -> 3.8105 ms   -10.5%   (stock spread 0.1299)
    end-to-end decode    115.01 -> 117.27 tok/s  +2.0%   ranges DO NOT overlap
    output difference    0.000e+00 ; 4/4 generations identical

The end-to-end win is smaller than the isolated one because the tail partially
overlaps other work in a real forward. The end-to-end figure is the honest one.

Why it is worth having despite being 2%: it is free. No representation change,
no kernel, no capability cost, and the outputs are bit-identical -- so it does
not re-baseline any measurement, unlike batching, which was refused for exactly
that reason (receipts/future/O003_BATCH_EQUIVALENCE.json).
"""
from __future__ import annotations

import mlx.core as mx


@mx.compile
def _combine(y, scores, shared):
    return (y * scores[..., None]).sum(axis=-2).astype(y.dtype) + shared


def _fused_call(self, x):
    inds, scores = self.gate(x)
    y = self.switch_mlp(x, inds)
    return _combine(y, scores, self.shared_experts(x))


def apply(model) -> int:
    """Patch every DeepseekV3MoE-style block in `model`. Returns how many.

    Patches the TYPE, because Python resolves __call__ there -- an
    instance-level patch silently does nothing, which cost this campaign a
    whole ablation run.
    """
    blocks = [l.mlp for l in model.language_model.model.layers
              if hasattr(getattr(l, "mlp", None), "switch_mlp")]
    if not blocks:
        return 0
    t = type(blocks[0])
    if getattr(t, "_hawking_moe_fused", False):
        return 0
    t._hawking_stock_call = t.__call__
    t.__call__ = _fused_call
    t._hawking_moe_fused = True
    return len(blocks)


def revert(model) -> bool:
    blocks = [l.mlp for l in model.language_model.model.layers
              if hasattr(getattr(l, "mlp", None), "switch_mlp")]
    if not blocks:
        return False
    t = type(blocks[0])
    if not getattr(t, "_hawking_moe_fused", False):
        return False
    t.__call__ = t._hawking_stock_call
    t._hawking_moe_fused = False
    return True
