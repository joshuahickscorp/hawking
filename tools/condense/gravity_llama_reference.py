#!/usr/bin/env python3.12
"""Reference Llama forward that reads its weights straight out of a `.gravity` shard.

This is the oracle Gate S0.8 is graded against. "Produces correct logits" cannot mean
"matches the BF16 parent" -- a sub-bit artifact is lossy by construction and would fail
that test while being a perfectly correct execution of what it stores. It means: the
runtime computes what the ARTIFACT encodes. So the authority has to read the same bytes
and use the same codec, and that is what this does: every projection is
``gravity_forge.pq_execute`` over the payload the container holds, with no dense
reconstruction anywhere.

Numpy on purpose. It is slow, independent of the Rust/Metal path it grades, and short
enough to audit by eye -- three properties a fast oracle would cost us.

    python3.12 tools/condense/gravity_llama_reference.py ART.gravity --tokens 128000 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_pack as pack  # noqa: E402
import gravity_format as gravity  # noqa: E402
import gravity_forge as forge  # noqa: E402


class GravityWeights:
    """Lazy tensor access over one `.gravity` shard, PQ artifacts cached after first use."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.header = gravity.read_header(path)
        self.index = {t["name"]: t for t in self.header["tensors"]}
        self._cache: dict[str, object] = {}

    def artifact(self, name: str):
        if name not in self._cache:
            entry = self.index[name]
            blob = gravity.read_tensor(self.path, name, verify_hash=False)
            if entry["codec"] == "gravity-pq":
                self._cache[name] = pack.load_artifact(blob)
            else:
                dtype = entry["codec"].split(".", 1)[1]
                if dtype == "bf16":
                    u16 = np.frombuffer(blob, dtype=np.uint16).astype(np.uint32) << 16
                    arr = u16.view(np.float32)
                elif dtype == "f16":
                    arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
                else:
                    arr = np.frombuffer(blob, dtype=np.float32)
                self._cache[name] = arr.reshape([int(d) for d in entry["shape"]])
        return self._cache[name]

    def matvec(self, name: str, x: np.ndarray) -> np.ndarray:
        obj = self.artifact(name)
        if isinstance(obj, np.ndarray):
            return obj @ x
        return forge.pq_execute(obj, x)

    def row(self, name: str, index: int) -> np.ndarray:
        """One row of a tensor. For PQ this decodes just that row's chunks."""
        obj = self.artifact(name)
        if isinstance(obj, np.ndarray):
            return obj[index].astype(np.float32)
        codes = obj.config["pq_codes"]
        nchunk, S, sub, D = codes["nchunk"], codes["S"], codes["sub"], codes["D"]
        out = np.empty(D * nchunk, dtype=np.float32)
        for c in range(nchunk):
            for s in range(S):
                code = codes["indices"][index * nchunk + c, s]
                out[c * D + s * sub:(c * D + s * sub) + sub] = codes["codebooks"][s][code]
        return out


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return (x / np.sqrt(np.mean(x * x) + eps)) * weight


def llama3_inv_freq(head_dim: int, theta: float, scaling: dict | None) -> np.ndarray:
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    if not scaling or scaling.get("rope_type") != "llama3":
        return inv
    factor = float(scaling["factor"])
    low, high = float(scaling["low_freq_factor"]), float(scaling["high_freq_factor"])
    orig = float(scaling["original_max_position_embeddings"])
    wavelen = 2 * np.pi / inv
    low_wl, high_wl = orig / low, orig / high
    smooth = (orig / wavelen - low) / (high - low)
    scaled = np.where(wavelen > low_wl, inv / factor, inv)
    mid = (wavelen <= low_wl) & (wavelen >= high_wl)
    return np.where(mid, (1 - smooth) * (inv / factor) + smooth * inv, scaled)


def forward(art: Path, tokens: list[int]) -> dict:
    w = GravityWeights(art)
    arch = w.header["architecture"]
    n_layer = int(arch["num_hidden_layers"])
    n_head = int(arch["num_attention_heads"])
    n_kv = int(arch["num_key_value_heads"])
    hidden = int(arch["hidden_size"])
    head_dim = hidden // n_head
    eps = float(arch["rms_norm_eps"])
    inv_freq = llama3_inv_freq(head_dim, float(arch["rope_theta"]),
                               (w.header.get("architecture") or {}).get("rope_scaling"))
    groups = n_head // n_kv

    k_cache = [np.zeros((0, n_kv, head_dim), np.float32) for _ in range(n_layer)]
    v_cache = [np.zeros((0, n_kv, head_dim), np.float32) for _ in range(n_layer)]
    logits = None
    for pos, token in enumerate(tokens):
        x = w.row("model.embed_tokens.weight", token).copy()
        cos = np.cos(pos * inv_freq).astype(np.float32)
        sin = np.sin(pos * inv_freq).astype(np.float32)
        for layer in range(n_layer):
            p = f"model.layers.{layer}."
            h = rms_norm(x, w.artifact(p + "input_layernorm.weight"), eps)
            q = w.matvec(p + "self_attn.q_proj.weight", h).reshape(n_head, head_dim)
            k = w.matvec(p + "self_attn.k_proj.weight", h).reshape(n_kv, head_dim)
            v = w.matvec(p + "self_attn.v_proj.weight", h).reshape(n_kv, head_dim)
            # NeoX-style rotate_half: pair i with i+head_dim/2, not i with i+1.
            half = head_dim // 2
            for vec in (q, k):
                a, b = vec[:, :half].copy(), vec[:, half:].copy()
                vec[:, :half] = a * cos - b * sin
                vec[:, half:] = b * cos + a * sin
            k_cache[layer] = np.concatenate([k_cache[layer], k[None]], 0)
            v_cache[layer] = np.concatenate([v_cache[layer], v[None]], 0)
            kc, vc = k_cache[layer], v_cache[layer]
            attn = np.empty((n_head, head_dim), np.float32)
            for head in range(n_head):
                kv = head // groups
                scores = (kc[:, kv, :] @ q[head]) / np.sqrt(head_dim)
                scores -= scores.max()
                probs = np.exp(scores)
                probs /= probs.sum()
                attn[head] = probs @ vc[:, kv, :]
            x = x + w.matvec(p + "self_attn.o_proj.weight", attn.reshape(-1))
            h = rms_norm(x, w.artifact(p + "post_attention_layernorm.weight"), eps)
            gate = w.matvec(p + "mlp.gate_proj.weight", h)
            up = w.matvec(p + "mlp.up_proj.weight", h)
            act = gate / (1.0 + np.exp(-gate)) * up
            x = x + w.matvec(p + "mlp.down_proj.weight", act)
        h = rms_norm(x, w.artifact("model.norm.weight"), eps)
        head_name = ("lm_head.weight" if "lm_head.weight" in w.index
                     else "model.embed_tokens.weight")
        logits = w.matvec(head_name, h)
    return {"logits": logits, "argmax": int(np.argmax(logits)),
            "top5": [int(i) for i in np.argsort(logits)[::-1][:5]],
            "logits_head": [float(v) for v in logits[:8]],
            "tied_head": head_name == "model.embed_tokens.weight"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--tokens", type=int, nargs="+", default=[128000, 9906, 1917])
    ap.add_argument("--dump", type=Path, help="write logits as raw f32 for a parity harness")
    args = ap.parse_args()
    out = forward(args.artifact, args.tokens)
    if args.dump:
        args.dump.write_bytes(np.ascontiguousarray(out.pop("logits"), np.float32).tobytes())
    else:
        out.pop("logits")
    print(json.dumps({"tokens": args.tokens, **out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
