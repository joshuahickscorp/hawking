"""O003 does not load on the current runtime, and that is why its physical
baselines cannot be reproduced.

mlx_lm 0.31.3 rejects moonshotai/Kimi-VL-A3B-Instruct with

    ValueError: Received 27 parameters not in model:
    language_model.model.layers.<N>.self_attn.kv_b_proj.weight

kimi_vl.py takes its attention from deepseek_v3.py, whose sanitizer DOES consume
kv_b_proj -- it splits each one into `embed_q` and `unembed_out`. But that
sanitizer builds its key prefix as

    model.layers.{l}.self_attn

while the Kimi-VL checkpoint stores

    language_model.model.layers.{l}.self_attn.kv_b_proj.weight

kimi_vl.py's own sanitizer knows about the `language_model.` prefix -- it uses it
to stack the experts -- but never applies the kv_b_proj split under it. So the 27
tensors are never split, never consumed, and load_weights rejects them.

CONSEQUENCE FOR THE CAMPAIGN: the two O003 decode figures on disk, 69.97 tok/s in
O003_TPS_BASELINE.json (marginal method stated) and 87.67 tok/s quoted as
`reference.bf16_decode_tok_s` in O003_PROVENANCE_FULL.json (no method stated),
were both produced under an earlier mlx_lm. Neither is reproducible today, and
the 25% gap between them cannot be settled without this shim or a pinned version.

This applies the split under the right prefix. It is a LOADER SHIM, not a patch
to the vendored package: mlx_lm stays untouched.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any

import mlx.core as mx


def kv_b_proj_keys(snapshot: str) -> list[str]:
    """Every kv_b_proj tensor in the checkpoint, in layer order."""
    import struct
    out = []
    for f in sorted(glob.glob(snapshot.rstrip("/") + "/*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        out += [k for k in hdr if k.endswith("kv_b_proj.weight")]
    return sorted(out, key=lambda k: int(k.split("layers.")[1].split(".")[0]))


def split_kv_b_proj(weights: dict, cfg: dict, prefix: str = "language_model.model") -> dict:
    """Split each kv_b_proj into embed_q / unembed_out, as deepseek_v3 would.

    Mirrors mlx_lm.models.deepseek_v3.Model.sanitize exactly -- reshape to
    (num_heads, qk_nope_head_dim + v_head_dim, -1), take the first
    qk_nope_head_dim rows transposed as embed_q and the rest as unembed_out --
    but under the prefix the Kimi-VL checkpoint actually uses.
    """
    t = cfg.get("text_config", cfg)
    n_layers = int(t["num_hidden_layers"])
    n_heads = int(t["num_attention_heads"])
    qk_nope = int(t["qk_nope_head_dim"])
    v_head = int(t["v_head_dim"])
    head_dim = qk_nope + v_head
    moved = 0
    for l in range(n_layers):
        p = f"{prefix}.layers.{l}.self_attn"
        key = f"{p}.kv_b_proj.weight"
        if key not in weights:
            continue
        v = weights.pop(key).reshape(n_heads, head_dim, -1)
        weights[f"{p}.embed_q.weight"] = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
        weights[f"{p}.unembed_out.weight"] = mx.contiguous(v[:, qk_nope:, :])
        moved += 1
    if not moved:
        raise RuntimeError(
            f"no kv_b_proj found under prefix {prefix!r}: the shim matched nothing, "
            f"which means the checkpoint layout changed and this fix is now silent")
    return weights


def _selfcheck() -> None:
    """The split must be shape-correct and must REFUSE when it matches nothing."""
    cfg = {"text_config": {"num_hidden_layers": 2, "num_attention_heads": 4,
                           "qk_nope_head_dim": 8, "v_head_dim": 8}}
    lora = 16
    w = {f"language_model.model.layers.{l}.self_attn.kv_b_proj.weight":
         mx.zeros((4 * 16, lora)) for l in range(2)}
    out = split_kv_b_proj(dict(w), cfg)
    assert not any("kv_b_proj" in k for k in out), "kv_b_proj survived the split"
    for l in range(2):
        p = f"language_model.model.layers.{l}.self_attn"
        assert out[f"{p}.embed_q.weight"].shape == (4, lora, 8), out[f"{p}.embed_q.weight"].shape
        assert out[f"{p}.unembed_out.weight"].shape == (4, 8, lora), out[f"{p}.unembed_out.weight"].shape

    # A shim that matches nothing must SHOUT. Silently returning the weights
    # unchanged would reproduce the original failure one call later, with the
    # shim in the stack taking the blame off the real cause.
    try:
        split_kv_b_proj({}, cfg)
        raise AssertionError("a shim that matched nothing returned success")
    except RuntimeError as exc:
        assert "matched nothing" in str(exc)

    # Wrong prefix is the same failure and must also refuse.
    try:
        split_kv_b_proj(dict(w), cfg, prefix="model")
        raise AssertionError("wrong prefix returned success")
    except RuntimeError:
        pass
    print("selfcheck OK")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        snap = sys.argv[1]
        ks = kv_b_proj_keys(snap)
        print(f"{len(ks)} kv_b_proj tensors; first: {ks[0] if ks else '(none)'}")
