#!/usr/bin/env python3
"""CompositionGenome: find where locally-good representations stop composing.

Walks the composition ladder on the sealed Qwen3.8 uniform-q4-v1 artifact
against the BF16 parent, on REAL activations (never synthetic X).

  component -> held-out functional probe -> adjacent pair -> short chain
  -> organ family -> full layer -> multi-layer -> complete token loop
  -> coherent generation

Every rung is RUN, or explicitly NOT_RUN with a reason. Fidelity is scale-aware
(cosine is not enough; 0.01*W scores cosine 1). Every number is reported against
its constant-mean null. Error accumulation is measured across depth, not assumed
linear. An organ is degraded until the chain stops surviving so the boundary is
bracketed rather than hoped for.

Run (repo root):

    python3 tools/headless/noetic_composition.py

Re-execs ~/.grok-vision/bin/python when the caller lacks torch. Does not spawn a
second 27B: llama-server on :52484 stays the only resident model; artifact
tensors stream on and off the CPU one layer at a time.
"""
from __future__ import annotations

import gc
import json
import os
import struct
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any


def _reexec_vision() -> None:
    vis = Path.home() / ".grok-vision" / "bin" / "python"
    if not vis.is_file():
        return
    try:
        if Path(sys.executable).resolve() == vis.resolve():
            return
    except OSError:
        pass
    os.execv(str(vis), [str(vis), *sys.argv])


# Importing this module must NEVER replace the interpreter. `pytest` collects
# this directory under the system python; an import-time execv into the vision
# python -- which has no pytest -- silently killed the whole collection run.
if __name__ == "__main__":
    _reexec_vision()

import numpy as np  # noqa: E402

# torch is required to RUN this harness, not to import it. The codec helpers
# below are pure numpy and are unit-tested under the system python, which has
# pytest but no torch; the vision python has torch but no pytest. Importing
# torch unconditionally here meant the tests could not be collected at all.
try:  # noqa: E402
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised by the test run
    class _AbsentTorch:
        """Lets the module IMPORT without torch, but never pretends to be it.

        Module-level `@torch.no_grad()` decorators are evaluated at import, so a
        plain None stub still breaks collection. This passes decoration through
        untouched and defers the failure to the moment something actually tries
        to compute, which only main() does.
        """

        def no_grad(self):
            return lambda fn: fn

        def __getattr__(self, name):
            def _absent(*_a, **_k):
                raise RuntimeError(
                    f"torch.{name} used without torch: run this harness directly "
                    "so it re-execs into ~/.grok-vision/bin/python"
                )
            return _absent

    torch = _AbsentTorch()
    nn = _AbsentTorch()
    TORCH_AVAILABLE = False

REPO = Path(__file__).resolve().parents[2]
ART = Path(os.environ.get("QWEN38_Q4_ARTIFACT", str(Path.home() / "models/qwen38-gravity-uniform-q4-v1")))
SRC = Path(os.environ.get("QWEN38_PARENT_BF16", str(Path.home() / "models/qwen3.8-27b-abliterated-bf16")))
LLAMA = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:52484")
# A whole-model arm writes its OWN receipt. Sharing one path cost the q3 control
# run its receipt: the arm finished, wrote NOETIC_COMPOSITION.json, and the next
# thing to touch that path overwrote it. The per-organ run keeps the plain name.
_ARM = os.environ.get("HAWKING_WHOLE_MODEL_MLP_CODEC", "").strip()
RECEIPT = REPO / "receipts" / "headless" / (
    f"NOETIC_COMPOSITION_WHOLEMODEL_{_ARM.upper()}.json" if _ARM
    else "NOETIC_COMPOSITION.json"
)
NATIVE_RECEIPT = REPO / "receipts" / "headless" / "QWEN38_GRAVITY_NATIVE.json"

LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
GROUP = 64
KEY_HEADS = 16
KEY_HEAD_DIM = 128
VALUE_HEADS = 48
VALUE_HEAD_DIM = 128
VALUES_PER_KEY = 3
QKVZ_ROWS_PER_KEY = KEY_HEAD_DIM * 2 + VALUES_PER_KEY * VALUE_HEAD_DIM * 2  # 1024
BA_ROWS_PER_KEY = VALUES_PER_KEY * 2  # 6
KEY_ELEMENTS = KEY_HEADS * KEY_HEAD_DIM  # 2048
VALUE_ELEMENTS = VALUE_HEADS * VALUE_HEAD_DIM  # 6144

# Named BEFORE looking. A chain SURVIVES only if all three hold.
# scale_aware = mean_row_cosine * gain, so a 0.01-scaled artifact cannot hide.
SCALE_AWARE_OVER_NULL = 0.05
MIN_GAIN = 0.50
MAX_REL_L2 = 0.50

SHORT_CHAIN = 4  # layers 0..3 (three DeltaNet + one GQA)
MULTI_LAYER = 8  # layers 0..7
SITE_LAYERS = (0, 3)  # capture organ sites here
DEGRADE_LAYER = 0
DEGRADE_ORGAN = "mlp.down_proj"

PROMPT = (
    "Explain, in ordinary prose, how a compiler turns a for-loop into "
    "basic blocks and then into machine code."
)

SCHEMA = "hawking.headless.noetic_composition.v1"


# ---------------------------------------------------------------------------
# small utils
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return ""


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return None
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return jsonable(x.tolist())
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    return str(x)


def llama_json(path: str, body: dict | None = None, timeout: float = 60.0) -> Any:
    url = LLAMA.rstrip("/") + path
    if body is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# scale-aware fidelity  (cosine is the trap this metric exists to catch)
# ---------------------------------------------------------------------------


def _as2d(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _as2d(a)
    b = _as2d(b)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30
    return float(np.mean(num / den))


def gain_score(a: np.ndarray, b: np.ndarray) -> float:
    """min(r, 1/r) of per-row L2 norms. Cosine cannot see this (doctor _gain)."""
    a = _as2d(a)
    b = _as2d(b)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    r = nb / (na + 1e-30)
    return float(np.mean(np.minimum(r, 1.0 / (r + 1e-30))))


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = _as2d(pred)
    ref = _as2d(ref)
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-30))


def constant_mean_null(ref: np.ndarray) -> np.ndarray:
    ref = _as2d(ref)
    mu = ref.mean(axis=0, keepdims=True)
    return np.broadcast_to(mu, ref.shape).copy()


def score(pred: np.ndarray, ref: np.ndarray, *, tag: str = "") -> dict:
    """Function-space score of pred against ref, plus the constant-mean null.

    Survival rule (named before looking):
      scale_aware > null_scale_aware + 0.05
      AND gain >= 0.50
      AND rel_l2 <= 0.50
    """
    pred = _as2d(pred)
    ref = _as2d(ref)
    if pred.shape[0] == 0 or ref.shape[0] == 0:
        raise ValueError(f"score({tag!r}) got an empty tensor; the split was wrong")
    null = constant_mean_null(ref)
    cos = row_cosine(pred, ref)
    gn = gain_score(pred, ref)
    rel = rel_l2(pred, ref)
    sa = cos * gn
    n_rows = int(ref.shape[0])
    # A single row's constant-mean null IS the row. Cosine-vs-null is 1 by
    # construction and cannot be beaten. That is a degenerate baseline, not a
    # composition failure — refuse to pretend otherwise.
    null_degenerate = n_rows < 2
    n_cos = 1.0 if null_degenerate else row_cosine(null, ref)
    n_gn = 1.0 if null_degenerate else gain_score(null, ref)
    n_rel = 0.0 if null_degenerate else rel_l2(null, ref)
    n_sa = 1.0 if null_degenerate else n_cos * n_gn
    if null_degenerate:
        survives = bool(gn >= MIN_GAIN and rel <= MAX_REL_L2)
    else:
        survives = bool(
            sa > n_sa + SCALE_AWARE_OVER_NULL and gn >= MIN_GAIN and rel <= MAX_REL_L2
        )
    corr = float(np.linalg.norm(ref - pred) / (np.linalg.norm(ref) + 1e-30))  # == rel
    # alpha such that (1-alpha)*rel <= MAX_REL_L2  => residual correction dependency
    if rel <= MAX_REL_L2:
        alpha_to_survive = 0.0
    else:
        alpha_to_survive = float(1.0 - MAX_REL_L2 / (rel + 1e-30))
        alpha_to_survive = min(1.0, max(0.0, alpha_to_survive))
    return {
        "tag": tag,
        "n_rows": int(ref.shape[0]),
        "dim": int(ref.shape[1]),
        "cosine": cos,
        "gain": gn,
        "scale_aware": sa,
        "rel_l2": rel,
        "null_cosine": n_cos,
        "null_gain": n_gn,
        "null_scale_aware": n_sa,
        "null_rel_l2": n_rel,
        "cosine_minus_null": cos - n_cos,
        "scale_aware_minus_null": sa - n_sa,
        "correction_rel": corr,
        "correction_alpha_to_survive": alpha_to_survive,
        "null_degenerate": null_degenerate,
        "survives": survives,
        "healthy": survives,
        "survival_rule": (
            f"scale_aware > null_scale_aware + {SCALE_AWARE_OVER_NULL} "
            f"AND gain >= {MIN_GAIN} AND rel_l2 <= {MAX_REL_L2}"
        ),
    }


# ---------------------------------------------------------------------------
# tensor IO
# ---------------------------------------------------------------------------


class SourceBF16:
    def __init__(self, root: Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._hdr: dict[str, tuple[Path, int, dict]] = {}

    def _header(self, shard: str) -> tuple[Path, int, dict]:
        if shard in self._hdr:
            return self._hdr[shard]
        path = self.root / shard
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        self._hdr[shard] = (path, hlen, hdr)
        return self._hdr[shard]

    def load(self, src_name: str) -> np.ndarray:
        shard = self.weight_map[src_name]
        path, hlen, hdr = self._header(shard)
        meta = hdr[src_name]
        s, e = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(8 + hlen + s)
            raw = f.read(e - s)
        dt = meta["dtype"]
        if dt in ("BF16", "BFLOAT16"):
            u16 = np.frombuffer(raw, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dt in ("F32", "FLOAT32"):
            arr = np.frombuffer(raw, dtype="<f4")
        else:
            raise ValueError(f"{src_name} dtype {dt}")
        return arr.reshape(meta["shape"]).copy()

    def embed_rows(self, token_ids: list[int]) -> np.ndarray:
        name = "model.language_model.embed_tokens.weight"
        shard = self.weight_map[name]
        path, hlen, hdr = self._header(shard)
        meta = hdr[name]
        rows, hidden = meta["shape"]
        assert hidden == HIDDEN
        row_bytes = hidden * 2  # bf16
        base = 8 + hlen + meta["data_offsets"][0]
        out = np.empty((len(token_ids), hidden), dtype=np.float32)
        with open(path, "rb") as f:
            for i, tid in enumerate(token_ids):
                if tid < 0 or tid >= rows:
                    raise ValueError(f"token id {tid} outside embed {rows}")
                f.seek(base + tid * row_bytes)
                raw = f.read(row_bytes)
                u16 = np.frombuffer(raw, dtype=np.uint16)
                out[i] = (u16.astype(np.uint32) << 16).view(np.float32)
        return out


def parse_q4_header(payload: bytes) -> dict:
    if payload[:8] != b"HQ30UQ4\0":
        raise ValueError(f"q4 magic {payload[:8]!r}")
    version = struct.unpack_from("<I", payload, 8)[0]
    gs = struct.unpack_from("<I", payload, 12)[0]
    rank = struct.unpack_from("<H", payload, 16)[0]
    elems = struct.unpack_from("<Q", payload, 20)[0]
    dims = [struct.unpack_from("<I", payload, 32 + 4 * i)[0] for i in range(rank)]
    after = 32 + 4 * rank
    groups = (elems + gs - 1) // gs
    return {
        "version": version,
        "group_size": gs,
        "rank": rank,
        "elements": elems,
        "shape": dims,
        "groups": groups,
        "scale_off": after,
        "code_off": after + groups * 2,
    }


def dequant_q4(payload: bytes) -> np.ndarray:
    h = parse_q4_header(payload)
    gs = h["group_size"]
    groups = h["groups"]
    elems = h["elements"]
    scales = np.frombuffer(payload, dtype="<f2", count=groups, offset=h["scale_off"]).astype(
        np.float32
    )
    codes = np.frombuffer(
        payload, dtype=np.uint8, count=groups * (gs // 2), offset=h["code_off"]
    )
    low = (codes & 0x0F).astype(np.int16) - 8
    high = (codes >> 4).astype(np.int16) - 8
    q = np.empty(groups * gs, dtype=np.float32)
    q[0::2] = low
    q[1::2] = high
    out = (q[:elems] * np.repeat(scales, gs)[:elems]).reshape(h["shape"])
    return out


def dequant_q4_rows(payload: bytes, row_ids: list[int]) -> np.ndarray:
    """Dequant selected rows of a rank-2 [V, H] q4 without materialising W."""
    h = parse_q4_header(payload)
    shape = h["shape"]
    if len(shape) != 2:
        raise ValueError(f"row dequant needs rank-2, got {shape}")
    v, width = shape
    gs = h["group_size"]
    if width % gs != 0:
        raise ValueError(f"row width {width} not divisible by group {gs}")
    groups_per_row = width // gs
    scales = np.frombuffer(
        payload, dtype="<f2", count=h["groups"], offset=h["scale_off"]
    ).astype(np.float32)
    code_bytes_per_group = gs // 2
    out = np.empty((len(row_ids), width), dtype=np.float32)
    for i, rid in enumerate(row_ids):
        if rid < 0 or rid >= v:
            raise ValueError(f"row {rid} outside {v}")
        g0 = rid * groups_per_row
        sc = scales[g0 : g0 + groups_per_row]
        off = h["code_off"] + g0 * code_bytes_per_group
        codes = np.frombuffer(
            payload, dtype=np.uint8, count=groups_per_row * code_bytes_per_group, offset=off
        )
        low = (codes & 0x0F).astype(np.int16) - 8
        high = (codes >> 4).astype(np.int16) - 8
        q = np.empty(groups_per_row * gs, dtype=np.float32)
        q[0::2] = low
        q[1::2] = high
        out[i] = q * np.repeat(sc, gs)
    return out


def read_f32v2(payload: bytes) -> np.ndarray:
    n = int.from_bytes(payload[:8], "little")
    arr = np.frombuffer(payload, dtype="<f4", count=n, offset=8)
    if arr.size != n:
        raise ValueError(f"f32v2 n={n} got {arr.size}")
    return np.array(arr, dtype=np.float32, copy=True)


def split_fused_qkvz(fused: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of qwen38_geometry::fuse_in_proj_qkvz. fused is [16384, 5120]."""
    hidden = fused.shape[-1]
    fused = fused.reshape(KEY_HEADS, QKVZ_ROWS_PER_KEY, hidden)
    q = fused[:, 0:KEY_HEAD_DIM, :]
    k = fused[:, KEY_HEAD_DIM : 2 * KEY_HEAD_DIM, :]
    v = fused[:, 2 * KEY_HEAD_DIM : 2 * KEY_HEAD_DIM + VALUE_ELEMENTS // KEY_HEADS, :]
    z = fused[:, 2 * KEY_HEAD_DIM + VALUE_ELEMENTS // KEY_HEADS :, :]
    qkv = np.concatenate(
        [
            q.reshape(KEY_ELEMENTS, hidden),
            k.reshape(KEY_ELEMENTS, hidden),
            v.reshape(VALUE_ELEMENTS, hidden),
        ],
        axis=0,
    )
    return qkv, z.reshape(VALUE_ELEMENTS, hidden)


def split_fused_ba(fused: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of fuse_in_proj_ba: per key-head [b x 3, a x 3]. fused [96, 5120]."""
    hidden = fused.shape[-1]
    fused = fused.reshape(KEY_HEADS, BA_ROWS_PER_KEY, hidden)
    b = fused[:, :VALUES_PER_KEY, :].reshape(VALUE_HEADS, hidden)
    a = fused[:, VALUES_PER_KEY:, :].reshape(VALUE_HEADS, hidden)
    return b, a


def requantize_absmax(w: np.ndarray, bits: int, group: int = GROUP) -> np.ndarray:
    """Grouped weight codec at `bits` bits per weight. bits=4 matches HQ30UQ4.

    Signed-symmetric absmax (bound = 2^(bits-1)-1) is only a sane
    parameterization down to 3 bits. At bits=2 the bound is 1, so every weight
    below half the group max rounds to zero -- measured 829/4096 survivors,
    rel_l2 0.7219. At bits=1 the bound is 0 and the codec returns the ZERO
    TENSOR: the first boundary hunt scored `requantize-q1` identical to the
    `zeroed down_proj` control to four decimals, because it WAS that control.
    That is an artifact of the parameterization, not a property of 1-bit.

    So the low arms use the codecs those bit-widths are actually defined by:

      bits == 1  binary sign coding, w ~ alpha * sign(w), alpha = mean|w| per
                 group -- the L2-optimal scalar for a sign code.
      bits == 2  ternary {-alpha, 0, +alpha} with threshold t = 0.7 * mean|w|
                 (Ternary Weight Networks) and alpha refit as the mean |w| of
                 the kept set, rather than absmax rounding.

    Both keep the same group size, so bits still means bits.
    """
    if bits <= 0:
        return np.zeros_like(w, dtype=np.float32)
    flat = np.asarray(w, dtype=np.float32).reshape(-1)

    if bits <= 2:
        n = flat.size
        groups = (n + group - 1) // group
        pad = groups * group - n
        work = np.concatenate([flat, np.zeros(pad, dtype=np.float32)]) if pad else flat.copy()
        g = work.reshape(groups, group)
        absg = np.abs(g)
        if bits == 1:
            alpha = absg.mean(axis=1, keepdims=True)
            recon = alpha * np.where(g >= 0, 1.0, -1.0)
        else:
            thresh = 0.7 * absg.mean(axis=1, keepdims=True)
            keep = absg > thresh
            kept = np.where(keep, absg, 0.0).sum(axis=1, keepdims=True)
            cnt = keep.sum(axis=1, keepdims=True)
            alpha = np.where(cnt > 0, kept / np.maximum(cnt, 1), 0.0)
            recon = alpha * np.sign(g) * keep
        out = recon.reshape(-1)[:n]
        return out.reshape(w.shape).astype(np.float32)

    bound = (1 << (bits - 1)) - 1
    n = flat.size
    groups = (n + group - 1) // group
    pad = groups * group - n
    if pad:
        work = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        work = flat.copy()
    g = work.reshape(groups, group)
    scale = np.max(np.abs(g), axis=1) / float(bound) if bound else np.zeros(groups)
    scale = np.where(scale > 0, scale, 1.0)
    q = np.rint(g / scale[:, None]).clip(-bound - 1 if bits == 4 else -bound, bound)
    # 4-bit HQ30UQ4 uses -8..7; keep that for bits==4
    if bits == 4:
        q = np.rint(g / scale[:, None]).clip(-8, 7)
    recon = (q * scale[:, None]).reshape(-1)[:n]
    return recon.reshape(w.shape).astype(np.float32)


def lowrank(w: np.ndarray, rank: int) -> np.ndarray:
    """Truncated SVD. The only arm here that can go BELOW one bit per weight."""
    m = np.asarray(w, dtype=np.float32)
    u, sv, vt = np.linalg.svd(m, full_matrices=False)
    r = min(int(rank), sv.size)
    return (u[:, :r] * sv[:r]) @ vt[:r], r


def codec_bpw(kind: str, shape, *, bits: int = 0, group: int = GROUP,
              rank: int = 0, scale_bits: int = 16) -> float:
    """Bits per PARENT weight, counting the scales -- not just the code width.

    This is the accounting the archaeology receipt caught NR missing: a codec
    that stores a 16-bit alpha per group of 64 is not a 1-bit codec, it is a
    1.25-bit codec, and calling it 1-bit hides 25% of the payload. Every arm
    below is reported at its true cost so the bracket compares like with like.
    """
    out_f, in_f = int(shape[0]), int(shape[1])
    n = out_f * in_f
    if kind == "bits":
        if bits <= 0:
            return 0.0
        groups = (n + group - 1) // group
        return (n * bits + groups * scale_bits) / n
    if kind == "lowrank":
        return (rank * (out_f + in_f) * scale_bits) / n
    if kind == "binary_lowrank":
        groups = (n + group - 1) // group
        return (n + groups * scale_bits + rank * (out_f + in_f) * scale_bits) / n
    if kind == "scale":
        return 4.0 + scale_bits / group
    return float("nan")


class ArtifactQ4:
    def __init__(self, root: Path):
        self.root = root
        man = json.loads((root / "manifest.json").read_text())
        self.manifest = man
        self.by_name = {t["name"]: t for t in man["tensors"]}

    def _payload(self, catalog_name: str) -> bytes:
        row = self.by_name[catalog_name]
        return (self.root / "tensors" / row["artifact"]).read_bytes()

    def load(self, catalog_name: str) -> np.ndarray:
        row = self.by_name[catalog_name]
        raw = self._payload(catalog_name)
        if row["kind"] == "q4":
            arr = dequant_q4(raw)
        elif row["kind"] == "f32":
            arr = read_f32v2(raw).reshape(row["shape"])
        else:
            raise ValueError(f"unknown kind {row['kind']} for {catalog_name}")
        return arr

    def embed_rows(self, token_ids: list[int]) -> np.ndarray:
        name = "language_model.model.embed_tokens.weight"
        row = self.by_name[name]
        raw = self._payload(name)
        return dequant_q4_rows(raw, token_ids)

    def lm_head_logits(self, hidden: np.ndarray) -> np.ndarray:
        """Chunked W @ h for the q4 lm_head without a 5 GiB dense W."""
        name = "language_model.lm_head.weight"
        raw = self._payload(name)
        h = parse_q4_header(raw)
        v, width = h["shape"]
        hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
        assert hidden.size == width
        chunk = 4096
        logits = np.empty(v, dtype=np.float32)
        for start in range(0, v, chunk):
            ids = list(range(start, min(v, start + chunk)))
            w = dequant_q4_rows(raw, ids)
            logits[start : start + len(ids)] = w @ hidden
        return logits


# ---------------------------------------------------------------------------
# HF layer (official Qwen3.5 math, weights injected)
# ---------------------------------------------------------------------------


def text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    raw = json.loads((SRC / "config.json").read_text())["text_config"]
    skip = {
        "dtype",
        "mamba_ssm_dtype",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "attn_output_gate",
        "output_gate_type",
    }
    cfg = Qwen3_5TextConfig(**{k: v for k, v in raw.items() if k not in skip})
    cfg._attn_implementation = "eager"
    return cfg


def additive_causal(seqlen: int, dtype: torch.dtype) -> torch.Tensor:
    neg = torch.finfo(dtype).min
    m = torch.triu(torch.full((seqlen, seqlen), neg, dtype=dtype), diagonal=1)
    return m[None, None, :, :]


def build_layer(cfg, layer_idx: int) -> nn.Module:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    return Qwen3_5DecoderLayer(cfg, layer_idx)


def load_teacher_layer(src: SourceBF16, cfg, layer_idx: int) -> nn.Module:
    layer = build_layer(cfg, layer_idx)
    prefix = f"model.language_model.layers.{layer_idx}."
    sd = {}
    for k in layer.state_dict():
        arr = src.load(prefix + k)
        sd[k] = torch.from_numpy(np.ascontiguousarray(arr))
    missing = layer.load_state_dict(sd, strict=False)
    if missing.missing_keys:
        raise RuntimeError(f"teacher L{layer_idx} missing {missing.missing_keys}")
    layer.eval()
    return layer


# Set by --whole-model-mlp-codec. None means the ordinary q4 student.
WHOLE_MODEL_MLP_CODEC = None


def load_student_layer(art: ArtifactQ4, cfg, layer_idx: int) -> nn.Module:
    layer = build_layer(cfg, layer_idx)
    cat_prefix = f"language_model.model.layers.{layer_idx}."
    sd: dict[str, torch.Tensor] = {}
    if layer.block_type == "linear_attention":
        qkvz = art.load(cat_prefix + "linear_attn.in_proj_qkvz.weight")
        qkv, z = split_fused_qkvz(qkvz)
        ba = art.load(cat_prefix + "linear_attn.in_proj_ba.weight")
        b, a = split_fused_ba(ba)
        sd["linear_attn.in_proj_qkv.weight"] = torch.from_numpy(np.ascontiguousarray(qkv))
        sd["linear_attn.in_proj_z.weight"] = torch.from_numpy(np.ascontiguousarray(z))
        sd["linear_attn.in_proj_b.weight"] = torch.from_numpy(np.ascontiguousarray(b))
        sd["linear_attn.in_proj_a.weight"] = torch.from_numpy(np.ascontiguousarray(a))
        for k in (
            "linear_attn.out_proj.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.norm.weight",
        ):
            sd[k] = torch.from_numpy(np.ascontiguousarray(art.load(cat_prefix + k)))
        conv = art.load(cat_prefix + "linear_attn.conv1d.weight")
        # catalog [C, K, 1] (packer reshape of HF [C, 1, K])
        if conv.ndim == 3 and conv.shape[-1] == 1:
            conv = np.transpose(conv, (0, 2, 1))
        sd["linear_attn.conv1d.weight"] = torch.from_numpy(np.ascontiguousarray(conv))
    else:
        for k in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ):
            sd[k] = torch.from_numpy(np.ascontiguousarray(art.load(cat_prefix + k)))
    for k in (
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ):
        w = art.load(cat_prefix + k)
        if WHOLE_MODEL_MLP_CODEC is not None and k.startswith("mlp."):
            # Whole-model arm: the organ-local CANON in FRACTIONAL_BIT_CANON.json
            # is a per-tensor function-space screen, and its own receipt says a
            # local CANON is not a composed win. This applies the codec to EVERY
            # MLP tensor of EVERY layer so the error has somewhere to accumulate.
            w = WHOLE_MODEL_MLP_CODEC(w)
        sd[k] = torch.from_numpy(np.ascontiguousarray(w))
    missing = layer.load_state_dict(sd, strict=False)
    if missing.missing_keys:
        raise RuntimeError(f"student L{layer_idx} missing {missing.missing_keys}")
    layer.eval()
    return layer


def inject_weight(layer: nn.Module, dotted: str, arr: np.ndarray) -> None:
    parts = dotted.split(".")
    mod = layer
    for p in parts[:-1]:
        mod = getattr(mod, p)
    getattr(mod, parts[-1]).data.copy_(torch.from_numpy(np.ascontiguousarray(arr)))


@torch.no_grad()
def run_layer_sites(
    layer: nn.Module,
    x: torch.Tensor,
    pos_emb: tuple[torch.Tensor, torch.Tensor],
    causal: torch.Tensor,
    pad: torch.Tensor,
    want_swiglu: bool,
) -> dict[str, torch.Tensor]:
    residual = x
    x_n = layer.input_layernorm(x)
    if layer.block_type == "linear_attention":
        mix = layer.linear_attn(hidden_states=x_n, cache_params=None, attention_mask=pad)
    else:
        mix, _ = layer.self_attn(
            hidden_states=x_n,
            position_embeddings=pos_emb,
            attention_mask=causal,
        )
    x_mid = residual + mix
    x_pn = layer.post_attention_layernorm(x_mid)
    if want_swiglu:
        g = layer.mlp.gate_proj(x_pn)
        u = layer.mlp.up_proj(x_pn)
        sw = layer.mlp.act_fn(g) * u
        mlp_out = layer.mlp.down_proj(sw)
        sites = {
            "post_input_norm": x_n,
            "mixer_out": mix,
            "post_mixer": x_mid,
            "post_attn_norm": x_pn,
            "gate": g,
            "up": u,
            "post_swiglu": sw,
            "mlp_out": mlp_out,
        }
    else:
        mlp_out = layer.mlp(x_pn)
        sites = {
            "post_input_norm": x_n,
            "mixer_out": mix,
            "post_mixer": x_mid,
            "post_attn_norm": x_pn,
            "mlp_out": mlp_out,
        }
    y = x_mid + mlp_out
    sites["x_out"] = y
    return sites


def t_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# ladder printers
# ---------------------------------------------------------------------------


def fmt_score(s: dict) -> str:
    flag = "SURVIVES" if s.get("survives") else "FAILS"
    return (
        f"{flag}  cos={s['cosine']:.4f} (null {s['null_cosine']:.4f}, Δ{s['cosine_minus_null']:+.4f})"
        f"  gain={s['gain']:.4f}  sa={s['scale_aware']:.4f} (null {s['null_scale_aware']:.4f}, "
        f"Δ{s['scale_aware_minus_null']:+.4f})  rel_l2={s['rel_l2']:.4f}  "
        f"corr_α={s['correction_alpha_to_survive']:.3f}"
    )


def rung_record(name: str, status: str, **kw: Any) -> dict:
    d = {"rung": name, "status": status}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# main walk
# ---------------------------------------------------------------------------


def tokenize_prompt() -> tuple[list[int], dict]:
    info: dict[str, Any] = {"method": None}
    try:
        tok = llama_json("/tokenize", {"content": PROMPT}, timeout=10)
        ids = list(tok["tokens"])
        info["method"] = "llama-server /tokenize"
        info["n_tokens"] = len(ids)
        return ids, info
    except Exception as e:
        info["llama_tokenize_error"] = repr(e)
    # fallback: tokenizers lib on the parent tokenizer.json
    from tokenizers import Tokenizer

    tz = Tokenizer.from_file(str(SRC / "tokenizer.json"))
    ids = tz.encode(PROMPT).ids
    info["method"] = "tokenizers.Tokenizer parent tokenizer.json"
    info["n_tokens"] = len(ids)
    return ids, info


def rmsnorm_delta(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """HF Qwen3_5RMSNorm: y = rms(x) * (1 + w)."""
    x32 = x.astype(np.float32)
    rms = np.sqrt((x32 ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x32 / rms) * (1.0 + w.astype(np.float32))


@torch.no_grad()
def _install_whole_model_codec() -> str | None:
    """Optionally quantize EVERY layer's MLP with a named codec.

    FRACTIONAL_BIT_CANON.json reaches CANON at 1.85 bpw on an organ-local
    function-space screen and states outright that a local CANON is not a
    composed win. This is how that gets tested: same codec, every MLP tensor,
    every layer, then the full 64-layer loop with its argmax.
    """
    global WHOLE_MODEL_MLP_CODEC
    name = os.environ.get("HAWKING_WHOLE_MODEL_MLP_CODEC")
    if not name:
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fractional_bit_canon as fbc

    if name.startswith("ternary_g"):
        g = int(name.split("_g")[1])
        # d=None is the MEAN-ABS scale, not the activation-aware one. That is
        # deliberately the WEAKER arm: it needs no per-layer capture, and if the
        # weaker codec survives composition the fitted one is not worse.
        WHOLE_MODEL_MLP_CODEC = lambda w: fbc.codec_ternary(w, g=g, d=None)[0]
    elif name.startswith("q2f_g"):
        # 4-level fitted 2-bit (2.25 bpw). Brackets the whole-model boundary
        # between ternary at 1.85, which fails, and q3 at 3.25, which survives.
        g = int(name.split("_g")[1])
        WHOLE_MODEL_MLP_CODEC = lambda w: fbc._fourlevel_fitted(w, g)
    elif name.startswith("q") and "_g" in name:
        # Whole-model control arm. Without a run of a codec that is known GOOD
        # locally, a whole-model failure cannot be attributed to sub-2-bit rather
        # than to any MLP requantization compounding at all.
        bits = int(name[1:name.index("_g")])
        g = int(name.split("_g")[1])
        WHOLE_MODEL_MLP_CODEC = lambda w: requantize_absmax(w, bits, group=g)
    elif name.startswith("binary_g"):
        g = int(name.split("_g")[1])
        WHOLE_MODEL_MLP_CODEC = lambda w: requantize_absmax(w, 1, group=g)
    elif name == "zeroed":
        WHOLE_MODEL_MLP_CODEC = lambda w: np.zeros_like(w)
    else:
        raise SystemExit(f"unknown HAWKING_WHOLE_MODEL_MLP_CODEC={name!r}")
    return name


def main() -> int:
    if not TORCH_AVAILABLE:
        sys.exit("torch required: run this harness directly so it re-execs into ~/.grok-vision/bin/python")
    whole_model = _install_whole_model_codec()
    if whole_model:
        print(f"WHOLE-MODEL ARM: every layer's mlp.{{gate,up,down}}_proj -> {whole_model}")
        sys.stdout.flush()

    warnings.filterwarnings("ignore")
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ.get("NOETIC_TORCH_THREADS", "8")))
    t_all = time.time()
    watched: list[str] = []
    rungs: list[dict] = []
    notes: list[str] = []

    print("=" * 78)
    print("NOETIC COMPOSITION GENOME")
    print("=" * 78)
    print(f"generated_at  {now_iso()}")
    print(f"git_head      {git_head()[:12]}")
    print(f"artifact      {ART}")
    print(f"parent_bf16   {SRC}")
    print(f"llama         {LLAMA}")
    print(f"torch         {torch.__version__} mps_built={torch.backends.mps.is_built()} "
          f"mps_avail={torch.backends.mps.is_available()}")
    print(
        f"survival rule scale_aware > null + {SCALE_AWARE_OVER_NULL} AND "
        f"gain >= {MIN_GAIN} AND rel_l2 <= {MAX_REL_L2}"
    )
    print(
        "recovered constraints: never synthetic X; cosine is scale-invariant; "
        "raw activation cosine null ~0.898 on GLM (measured here, not assumed); "
        "local BPW is not a result without a health verdict; G035 sharing lost; "
        "do not spawn a second 27B"
    )
    sys.stdout.flush()

    if not (ART / "manifest.json").is_file():
        print("FATAL: artifact missing")
        return 2
    if not (SRC / "model.safetensors.index.json").is_file():
        print("FATAL: parent bf16 missing")
        return 2

    art = ArtifactQ4(ART)
    src = SourceBF16(SRC)
    man = art.manifest
    print(
        f"catalog       schema={man.get('schema')} tensors={man.get('tensor_count')} "
        f"q4={man.get('q4_tensors')} f32={man.get('f32_tensors')} "
        f"complete_bpw={man.get('complete_physical_bpw')} "
        f"min_q4_cosine(manifest)={man.get('min_q4_cosine')}"
    )

    # live llama identity
    llama_ok = False
    llama_model = None
    try:
        health = llama_json("/health", timeout=3)
        models = llama_json("/v1/models", timeout=3)
        llama_ok = health.get("status") == "ok"
        llama_model = (models.get("data") or models.get("models") or [{}])[0]
        llama_id = llama_model.get("id") or llama_model.get("name")
        print(f"llama health  {health} model={llama_id}")
    except Exception as e:
        print(f"llama         UNREACHABLE {e!r}")
        notes.append(f"llama-server unreachable: {e!r}")

    token_ids, tok_info = tokenize_prompt()
    # keep the prompt short enough that 64-layer CPU GEMM finishes
    max_tok = int(os.environ.get("NOETIC_MAX_TOKENS", "16"))
    if len(token_ids) > max_tok:
        token_ids = token_ids[:max_tok]
        tok_info["truncated_to"] = max_tok
    print(f"prompt tokens {len(token_ids)} via {tok_info['method']}: {token_ids[:12]}...")
    sys.stdout.flush()

    cfg = text_config()
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    # embeddings: teacher rows + student q4 rows of the SAME tokens (real, not Gaussian)
    t_emb = src.embed_rows(token_ids)
    s_emb = art.embed_rows(token_ids)
    embed_score = score(s_emb, t_emb, tag="embed_tokens rows of prompt")
    print(f"\n[pre] embed_tokens prompt rows  {fmt_score(embed_score)}")
    if not embed_score["survives"]:
        watched.append(
            f"q4 embed_tokens on the prompt rows already FAILS the survival rule "
            f"(rel_l2={embed_score['rel_l2']:.4f}). Local catalog cosine can still look fine."
        )

    x_teacher = torch.from_numpy(t_emb[None].copy())  # [1,S,H]
    x_student_embed = torch.from_numpy(s_emb[None].copy())
    bsz, seqlen, _ = x_teacher.shape
    pos = torch.arange(seqlen)[None]
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    cos, sin = rope(x_teacher, pos)
    pos_emb = (cos, sin)
    pad = torch.ones(bsz, seqlen, dtype=torch.bool)
    causal = additive_causal(seqlen, torch.float32)

    # ------------------------------------------------------------------
    # Teacher walk: real activations at every layer.
    # ------------------------------------------------------------------
    print(f"\n--- teacher BF16 walk  {LAYERS} layers, seq={seqlen} ---")
    sys.stdout.flush()
    teacher_h = [x_teacher]
    teacher_sites: dict[int, dict[str, np.ndarray]] = {}
    t0 = time.time()
    for L in range(LAYERS):
        layer = load_teacher_layer(src, cfg, L)
        want = L in SITE_LAYERS
        sites = run_layer_sites(layer, teacher_h[-1], pos_emb, causal, pad, want_swiglu=want)
        teacher_h.append(sites["x_out"].contiguous())
        if want:
            teacher_sites[L] = {k: t_np(v) for k, v in sites.items()}
        del layer, sites
        gc.collect()
        if L % 8 == 7 or L == LAYERS - 1:
            print(f"  teacher L{L:02d}  std={float(teacher_h[-1].std()):.4f}  "
                  f"elapsed {time.time()-t0:.1f}s")
            sys.stdout.flush()
    print(f"teacher walk wall {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Student walk: teacher-forced (local) AND free-running (composition).
    # ------------------------------------------------------------------
    print(f"\n--- student q4 walk  teacher-forced + free-run ---")
    sys.stdout.flush()
    student_h = x_student_embed.clone()
    depth_rows: list[dict] = []
    student_sites: dict[int, dict[str, np.ndarray]] = {}
    t0 = time.time()
    first_fail_free = None
    first_fail_local = None
    for L in range(LAYERS):
        layer = load_student_layer(art, cfg, L)
        want = L in SITE_LAYERS
        local_sites = run_layer_sites(
            layer, teacher_h[L], pos_emb, causal, pad, want_swiglu=want
        )
        free_sites = run_layer_sites(
            layer, student_h, pos_emb, causal, pad, want_swiglu=want
        )
        local_y = t_np(local_sites["x_out"])
        free_y = t_np(free_sites["x_out"])
        ref_y = t_np(teacher_h[L + 1])
        s_local = score(local_y, ref_y, tag=f"L{L}.local_full_layer")
        s_free = score(free_y, ref_y, tag=f"L{L}.free_run")
        row = {
            "layer": L,
            "mixer": "gqa" if (L + 1) % 4 == 0 else "delta_net",
            "local": s_local,
            "free": s_free,
        }
        depth_rows.append(row)
        if want:
            student_sites[L] = {k: t_np(v) for k, v in free_sites.items()}
            student_sites[f"{L}.local"] = {k: t_np(v) for k, v in local_sites.items()}
        if first_fail_local is None and not s_local["survives"]:
            first_fail_local = L
        if first_fail_free is None and not s_free["survives"]:
            first_fail_free = L
        student_h = free_sites["x_out"].contiguous()
        del layer, local_sites, free_sites
        gc.collect()
        if L % 8 == 7 or L == LAYERS - 1 or L < 8:
            print(
                f"  L{L:02d} {row['mixer']:<10} local {fmt_score(s_local)}\n"
                f"             free  {fmt_score(s_free)}"
            )
            sys.stdout.flush()
    print(f"student walk wall {time.time()-t0:.1f}s")
    print(
        f"first FAIL local_full_layer: {first_fail_local}   "
        f"first FAIL free-run: {first_fail_free}"
    )

    # error accumulation (measured, not assumed linear)
    free_rel = [r["free"]["rel_l2"] for r in depth_rows]
    local_rel = [r["local"]["rel_l2"] for r in depth_rows]
    free_sa = [r["free"]["scale_aware"] for r in depth_rows]
    diffs = [free_rel[i] - free_rel[i - 1] for i in range(1, len(free_rel))]
    monotonic = all(d >= -1e-6 for d in diffs)
    print("\n--- error accumulation (free-run rel_l2 vs teacher hidden) ---")
    print(f"{'L':>4} {'mixer':<10} {'local_rel':>10} {'free_rel':>10} {'Δfree':>10} {'free_sa':>10} {'local_ok':>8} {'free_ok':>8}")
    for i, r in enumerate(depth_rows):
        dlt = 0.0 if i == 0 else free_rel[i] - free_rel[i - 1]
        print(
            f"{r['layer']:4d} {r['mixer']:<10} {local_rel[i]:10.4f} {free_rel[i]:10.4f} "
            f"{dlt:10.4f} {free_sa[i]:10.4f} "
            f"{str(r['local']['survives']):>8} {str(r['free']['survives']):>8}"
        )
    print(
        f"monotonic_nondecreasing_free_rel_l2={monotonic}  "
        f"if false, that is a finding (error does not just pile up)"
    )
    if not monotonic:
        watched.append(
            "free-run rel_l2 is NOT monotonic across depth: compounding is not a "
            "linear pile-up. "
            + ", ".join(
                f"L{i}->{i+1} Δ={diffs[i]:+.4f}"
                for i in range(len(diffs))
                if diffs[i] < -1e-4
            )
        )

    # ------------------------------------------------------------------
    # SCALE-AWARE METRIC DEMO  (must reject 0.01*W)
    # ------------------------------------------------------------------
    print("\n--- scale-aware metric demo (must REJECT 0.01·W) ---")
    X = _as2d(teacher_sites[0]["post_attn_norm"])  # real post-attn hidden, L0, rows=tokens
    W_gate = src.load("model.language_model.layers.0.mlp.gate_proj.weight")
    Y = X @ W_gate.T
    Y_scale = X @ (0.01 * W_gate).T
    W_q4 = art.load("language_model.model.layers.0.mlp.gate_proj.weight")
    Y_q4 = X @ W_q4.T
    sc_id = score(Y, Y, tag="identity")
    sc_scale = score(Y_scale, Y, tag="0.01*W_gate")
    sc_q4 = score(Y_q4, Y, tag="q4 gate_proj on real X")
    sc_cos_only_trap = {
        "cosine_of_0.01W": sc_scale["cosine"],
        "gain_of_0.01W": sc_scale["gain"],
        "scale_aware_of_0.01W": sc_scale["scale_aware"],
        "cosine_would_accept": bool(sc_scale["cosine"] > 0.99),
        "scale_aware_rejects": (not sc_scale["survives"]),
    }
    print(f"  identity     {fmt_score(sc_id)}")
    print(f"  0.01*W_gate  {fmt_score(sc_scale)}")
    print(f"  q4 gate_proj {fmt_score(sc_q4)}")
    print(
        f"  cosine(0.01W)={sc_scale['cosine']:.6f}  (the trap: cosine is scale-invariant)  "
        f"gain={sc_scale['gain']:.6f}  scale_aware={sc_scale['scale_aware']:.6f}  "
        f"REJECTED={not sc_scale['survives']}"
    )
    if sc_scale["survives"]:
        watched.append("METRIC BLINDNESS: 0.01*W survived the named rule. The rule is wrong.")
    else:
        watched.append(
            f"scale-aware metric REJECTED 0.01*W_gate on real L0 post_attn_norm "
            f"(cosine={sc_scale['cosine']:.6f} would have accepted; "
            f"gain={sc_scale['gain']:.6f} caught it; null_cosine={sc_scale['null_cosine']:.4f})"
        )
    if sc_q4["survives"]:
        notes.append("L0 gate_proj q4 SURVIVES locally on real X (expected for grouped-absmax q4).")
    else:
        watched.append(
            f"L0 gate_proj q4 already FAILS locally on real X "
            f"(rel_l2={sc_q4['rel_l2']:.4f} sa={sc_q4['scale_aware']:.4f})"
        )

    # ------------------------------------------------------------------
    # RUNG 1: COMPONENT
    # ------------------------------------------------------------------
    rungs.append(
        rung_record(
            "component",
            "RUN",
            organ="mlp.gate_proj",
            layer=0,
            site="post_attn_norm (real teacher L0)",
            score=sc_q4,
            also={"embed_tokens_prompt_rows": embed_score},
        )
    )
    print(f"\n[1] COMPONENT  L0 mlp.gate_proj  {fmt_score(sc_q4)}")

    # ------------------------------------------------------------------
    # RUNG 2: HELD-OUT FUNCTIONAL
    # ------------------------------------------------------------------
    n = int(X.shape[0])
    cut = max(1, n // 2)
    if cut >= n:
        raise RuntimeError(f"held-out split empty: n_rows={n}")
    # evaluate on the held-out half of the real prompt tokens; this is NOT a fit
    sc_hold = score(Y_q4[cut:], Y[cut:], tag="held-out tokens of real prompt")
    sc_fit = score(Y_q4[:cut], Y[:cut], tag="seen tokens of real prompt")
    rungs.append(
        rung_record(
            "held_out_functional_probe",
            "RUN",
            note="split of REAL prompt tokens; no codec was fitted on this X (NS-014: this is eval, not a rank claim)",
            n_rows=n,
            hold_rows=n - cut,
            seen=sc_fit,
            held_out=sc_hold,
        )
    )
    print(f"[2] HELD-OUT   seen {fmt_score(sc_fit)}")
    print(f"             hold {fmt_score(sc_hold)}")
    if sc_fit["survives"] and not sc_hold["survives"]:
        watched.append("held-out tokens fail while seen tokens survive: the local fit does not transfer.")

    # ------------------------------------------------------------------
    # RUNG 3: ADJACENT PAIR  (L0 then L1, teacher-forced vs free-run)
    # ------------------------------------------------------------------
    pair_local = depth_rows[1]["local"]  # L1 applied to teacher L0 out
    pair_free = depth_rows[1]["free"]    # L1 applied to student L0 out
    rungs.append(
        rung_record(
            "adjacent_pair",
            "RUN",
            layers=[0, 1],
            teacher_forced_L1_on_teacher_L0=pair_local,
            free_run_L1_on_student_L0=pair_free,
            composition_gap_rel_l2=pair_free["rel_l2"] - pair_local["rel_l2"],
        )
    )
    print(f"[3] ADJACENT PAIR L0→L1")
    print(f"             teacher-forced L1  {fmt_score(pair_local)}")
    print(f"             free-run L1        {fmt_score(pair_free)}")
    print(f"             composition gap rel_l2 = {pair_free['rel_l2']-pair_local['rel_l2']:+.4f}")
    if pair_local["survives"] and not pair_free["survives"]:
        watched.append(
            "ADJACENT PAIR is the boundary: L1 is locally fine on teacher X and dies on student X."
        )

    # ------------------------------------------------------------------
    # RUNG 4: SHORT CHAIN  layers 0..3
    # ------------------------------------------------------------------
    chain = depth_rows[SHORT_CHAIN - 1]
    rungs.append(
        rung_record(
            "short_chain",
            "RUN",
            layers=list(range(SHORT_CHAIN)),
            free_at_end=chain["free"],
            local_at_end=chain["local"],
            free_rel_l2_path=[depth_rows[i]["free"]["rel_l2"] for i in range(SHORT_CHAIN)],
            local_rel_l2_path=[depth_rows[i]["local"]["rel_l2"] for i in range(SHORT_CHAIN)],
        )
    )
    print(f"[4] SHORT CHAIN L0..L{SHORT_CHAIN-1}  free-run end {fmt_score(chain['free'])}")
    print(f"             free rel_l2 path {[round(depth_rows[i]['free']['rel_l2'],4) for i in range(SHORT_CHAIN)]}")

    # ------------------------------------------------------------------
    # RUNG 5: ORGAN FAMILY  SwiGLU (gate, up, down) at L0
    # ------------------------------------------------------------------
    ts = teacher_sites[0]
    ss_local = student_sites[f"{0}.local"]
    fam = {
        "gate": score(ss_local["gate"], ts["gate"], tag="L0.gate"),
        "up": score(ss_local["up"], ts["up"], tag="L0.up"),
        "post_swiglu": score(ss_local["post_swiglu"], ts["post_swiglu"], tag="L0.post_swiglu"),
        "down_mlp_out": score(ss_local["mlp_out"], ts["mlp_out"], tag="L0.mlp_out"),
    }
    fam_fail = [k for k, v in fam.items() if not v["survives"]]
    rungs.append(
        rung_record(
            "organ_family",
            "RUN",
            family="SwiGLU (gate, up, silu(g)*u, down)",
            layer=0,
            site="real teacher post_attn_norm (teacher-forced student organs)",
            members=fam,
            any_fail=bool(fam_fail),
            failed_members=fam_fail,
        )
    )
    print("[5] ORGAN FAMILY  L0 SwiGLU (teacher-forced student organs)")
    for k, v in fam.items():
        print(f"             {k:<12} {fmt_score(v)}")
    if fam_fail:
        watched.append(f"SwiGLU family members failing at L0: {fam_fail}")

    # ------------------------------------------------------------------
    # RUNG 6: FULL LAYER  L0 (DeltaNet + MLP) and L3 (GQA + MLP)
    # ------------------------------------------------------------------
    full = {}
    for L in SITE_LAYERS:
        full[f"L{L}"] = {
            "mixer": "gqa" if (L + 1) % 4 == 0 else "delta_net",
            "local": depth_rows[L]["local"],
            "free": depth_rows[L]["free"],
            "mixer_out": score(
                student_sites[f"{L}.local"]["mixer_out"],
                teacher_sites[L]["mixer_out"],
                tag=f"L{L}.mixer_out.local",
            ),
            "mlp_out": score(
                student_sites[f"{L}.local"]["mlp_out"],
                teacher_sites[L]["mlp_out"],
                tag=f"L{L}.mlp_out.local",
            ),
        }
    rungs.append(rung_record("full_layer", "RUN", layers=list(SITE_LAYERS), layers_detail=full))
    print("[6] FULL LAYER")
    for k, v in full.items():
        print(f"             {k} {v['mixer']} local {fmt_score(v['local'])}")
        print(f"                 mixer_out {fmt_score(v['mixer_out'])}")
        print(f"                 mlp_out   {fmt_score(v['mlp_out'])}")
        print(f"                 free-run  {fmt_score(v['free'])}")

    # ------------------------------------------------------------------
    # RUNG 7: MULTI-LAYER  0..7
    # ------------------------------------------------------------------
    ml = depth_rows[MULTI_LAYER - 1]
    rungs.append(
        rung_record(
            "multi_layer",
            "RUN",
            layers=list(range(MULTI_LAYER)),
            free_at_end=ml["free"],
            local_at_end=ml["local"],
            free_rel_l2_path=[depth_rows[i]["free"]["rel_l2"] for i in range(MULTI_LAYER)],
            first_fail_free_in_span=next(
                (i for i in range(MULTI_LAYER) if not depth_rows[i]["free"]["survives"]),
                None,
            ),
        )
    )
    print(f"[7] MULTI-LAYER L0..L{MULTI_LAYER-1}  free-run end {fmt_score(ml['free'])}")
    print(
        f"             free rel_l2 path "
        f"{[round(depth_rows[i]['free']['rel_l2'],4) for i in range(MULTI_LAYER)]}"
    )

    # ------------------------------------------------------------------
    # RUNG 8: COMPLETE TOKEN LOOP  residual through 64 layers + lm_head argmax
    # ------------------------------------------------------------------
    print("[8] COMPLETE TOKEN LOOP  64 layers + lm_head argmax (streamed, no second 27B)")
    sys.stdout.flush()
    # final RMSNorm
    t_final_w = src.load("model.language_model.norm.weight")
    s_final_w = art.load("language_model.model.norm.weight")
    t_seq = t_np(teacher_h[-1])  # [1,S,H]
    s_seq = t_np(student_h)
    t_normed = rmsnorm_delta(t_seq, t_final_w)
    s_normed = rmsnorm_delta(s_seq, s_final_w)
    hid_score = score(s_normed, t_normed, tag="final_norm all-positions")
    print(f"             final hidden  {fmt_score(hid_score)}")

    # teacher logits via chunked bf16 lm_head
    def teacher_logits(hvec: np.ndarray) -> np.ndarray:
        name = "lm_head.weight"
        shard = src.weight_map[name]
        path, hlen, hdr = src._header(shard)
        meta = hdr[name]
        v, width = meta["shape"]
        row_bytes = width * 2
        base = 8 + hlen + meta["data_offsets"][0]
        hvec = np.asarray(hvec, dtype=np.float32).reshape(-1)
        logits = np.empty(v, dtype=np.float32)
        chunk = 4096
        with open(path, "rb") as f:
            for start in range(0, v, chunk):
                n = min(chunk, v - start)
                f.seek(base + start * row_bytes)
                raw = f.read(n * row_bytes)
                u16 = np.frombuffer(raw, dtype=np.uint16)
                w = (u16.astype(np.uint32) << 16).view(np.float32).reshape(n, width)
                logits[start : start + n] = w @ hvec
        return logits

    print("             scoring lm_head (chunked, last token) ...")
    sys.stdout.flush()
    t_logits = teacher_logits(t_normed[0, -1])
    s_logits = art.lm_head_logits(s_normed[0, -1])
    t_arg = int(np.argmax(t_logits))
    s_arg = int(np.argmax(s_logits))
    logit_score = score(s_logits[None], t_logits[None], tag="lm_head logits")
    token_agree = t_arg == s_arg
    print(f"             logits      {fmt_score(logit_score)}")
    print(f"             argmax teacher={t_arg} student={s_arg} agree={token_agree}")
    loop_survives = bool(hid_score["survives"] and token_agree)
    rungs.append(
        rung_record(
            "complete_token_loop",
            "RUN",
            note="streamed 64-layer residual + final RMSNorm + chunked lm_head; no second resident 27B",
            final_hidden=hid_score,
            logits=logit_score,
            teacher_argmax=t_arg,
            student_argmax=s_arg,
            argmax_agree=token_agree,
            survives=loop_survives,
            free_rel_l2_at_L63=depth_rows[-1]["free"]["rel_l2"],
        )
    )
    if not token_agree:
        watched.append(
            f"complete token loop: student argmax {s_arg} != teacher argmax {t_arg} "
            f"after 64 free-run layers (hidden rel_l2={hid_score['rel_l2']:.4f})"
        )

    # ------------------------------------------------------------------
    # RUNG 9: COHERENT GENERATION
    # ------------------------------------------------------------------
    gen_live: dict[str, Any] = {"status": "NOT_RUN"}
    if llama_ok:
        try:
            chat = llama_json(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": 64,
                    "temperature": 0,
                },
                timeout=120,
            )
            text = chat["choices"][0]["message"]["content"]
            gen_live = {
                "status": "RUN",
                "vehicle": "llama-server Q5_K already resident on :52484 (NOT the gravity q4 artifact)",
                "text": text,
                "n_chars": len(text),
                "has_replacement_char": "\ufffd" in text,
                "looks_like_prose": bool(text) and ("\ufffd" not in text) and len(text) > 20,
            }
            print(f"[9] COHERENT GENERATION  live Q5_K (control, already resident)")
            print(f"             excerpt: {text[:240]!r}")
        except Exception as e:
            gen_live = {"status": "ERROR", "error": repr(e)}
            print(f"[9] COHERENT GENERATION  live Q5_K ERROR {e!r}")
    else:
        print("[9] COHERENT GENERATION  live Q5_K NOT_RUN (server unreachable)")

    native_note = None
    if NATIVE_RECEIPT.is_file():
        try:
            native = json.loads(NATIVE_RECEIPT.read_text())
            excerpt = (
                ((native.get("three_part_bar") or {}).get("generated_text_excerpt"))
                or ""
            )
            native_note = {
                "path": str(NATIVE_RECEIPT),
                "bar": native.get("bar"),
                "coherence": (native.get("three_part_bar") or {}).get("coherent_compiler_prose"),
                "excerpt_head": excerpt[:80],
                "replacement_chars": excerpt.count("\ufffd"),
            }
        except Exception as e:
            native_note = {"path": str(NATIVE_RECEIPT), "error": repr(e)}

    rungs.append(
        rung_record(
            "coherent_generation",
            "NOT_RUN",
            reason=(
                "Generating from the gravity q4 artifact via the native Metal decoder "
                "would load the 14.3 GiB catalog while llama-server already holds a 27B "
                "Q5_K on :52484. Contract: do not spawn a second 27B. The live Q5_K "
                "control is RUN. Native gravity generate was already measured on this "
                "artifact in receipts/headless/QWEN38_GRAVITY_NATIVE.json "
                "(replacement-character collapse)."
            ),
            live_q5k_control=gen_live,
            native_gravity_receipt=native_note,
            python_token_loop_proxy={
                "argmax_agree": token_agree,
                "teacher_argmax": t_arg,
                "student_argmax": s_arg,
                "note": "one greedy token from streamed python math; not a 128-token generate",
            },
        )
    )
    print(
        "[9] COHERENT GENERATION  ARTIFACT native decode NOT_RUN "
        "(second 27B forbidden); cited QWEN38_GRAVITY_NATIVE.json"
    )
    if native_note and native_note.get("replacement_chars"):
        watched.append(
            f"native gravity q4 generate collapsed to U+FFFD "
            f"({native_note['replacement_chars']} replacement chars in the sealed receipt)"
        )

    # ------------------------------------------------------------------
    # FAILURE BOUNDARY: degrade L0 down_proj until the short chain dies
    # ------------------------------------------------------------------
    print("\n--- failure-boundary hunt: degrade L0 mlp.down_proj, re-run short chain ---")
    print(f"    chain = free-run layers 0..{SHORT_CHAIN-1}; only L0 down_proj is replaced")
    sys.stdout.flush()
    base_down = art.load(f"language_model.model.layers.{DEGRADE_LAYER}.mlp.down_proj.weight")
    # schedule named before looking: bit-depth then gain. Last survivor / first death brackets.
    schedule = [
        {"kind": "bits", "bits": 4, "label": "q4 (artifact down_proj, undegraded)"},
        {"kind": "bits", "bits": 3, "label": "q3 grouped-64"},
        {"kind": "bits", "bits": 2, "label": "ternary grouped-64"},
        {"kind": "bits", "bits": 1, "label": "binary grouped-64"},
        {"kind": "bits", "bits": 1, "group": 1024, "label": "binary grouped-1024"},
        # below one bit per weight: only low rank gets there without a mask
        {"kind": "binary_lowrank", "rank": 64, "label": "binary g1024 + rank-64 residual"},
        {"kind": "lowrank", "rank": 512, "label": "rank-512 (no quantization)"},
        {"kind": "lowrank", "rank": 128, "label": "rank-128 (no quantization)"},
        {"kind": "lowrank", "rank": 32, "label": "rank-32 (no quantization)"},
        {"kind": "scale", "scale": 0.25, "label": "0.25 * q4 down_proj"},
        {"kind": "scale", "scale": 0.05, "label": "0.05 * q4 down_proj"},
        {"kind": "scale", "scale": 0.01, "label": "0.01 * q4 down_proj"},
        {"kind": "scale", "scale": 0.0, "label": "zeroed down_proj"},
    ]
    boundary_rows = []
    last_survive = None
    first_die = None
    for step in schedule:
        kind = step["kind"]
        grp = int(step.get("group", GROUP))
        eff_rank = 0
        if kind == "bits":
            wh = requantize_absmax(base_down, int(step["bits"]), group=grp)
        elif kind == "lowrank":
            wh, eff_rank = lowrank(base_down, int(step["rank"]))
            wh = wh.astype(np.float32)
        elif kind == "binary_lowrank":
            coarse = requantize_absmax(base_down, 1, group=1024)
            resid, eff_rank = lowrank(base_down - coarse, int(step["rank"]))
            wh = (coarse + resid).astype(np.float32)
        else:
            wh = (float(step["scale"]) * base_down).astype(np.float32)
        step["bpw"] = codec_bpw(
            kind, base_down.shape, bits=int(step.get("bits", 0)), group=grp,
            rank=eff_rank or int(step.get("rank", 0)),
        )
        step["rel_l2_weight"] = float(
            np.linalg.norm(wh - base_down) / max(np.linalg.norm(base_down), 1e-12)
        )
        # walk short chain with injected down_proj at L0
        h = x_student_embed.clone()
        end_score = None
        for L in range(SHORT_CHAIN):
            layer = load_student_layer(art, cfg, L)
            if L == DEGRADE_LAYER:
                inject_weight(layer, "mlp.down_proj.weight", wh)
            sites = run_layer_sites(layer, h, pos_emb, causal, pad, want_swiglu=False)
            h = sites["x_out"].contiguous()
            del layer, sites
            gc.collect()
        end_score = score(t_np(h), t_np(teacher_h[SHORT_CHAIN]), tag=f"degrade:{step['label']}")
        rec = {
            "label": step["label"],
            "kind": step["kind"],
            **{k: step[k] for k in step if k not in ("kind", "label")},
            "score": end_score,
        }
        boundary_rows.append(rec)
        flag = "SURVIVES" if end_score["survives"] else "FAILS"
        bpw_s = f"{step['bpw']:.4f}bpw" if step.get("bpw") == step.get("bpw") else "  n/a   "
        print(f"    {step['label']:<34} {bpw_s:>11}  wΔ={step['rel_l2_weight']:.4f}  {flag}  {fmt_score(end_score)}")
        sys.stdout.flush()
        if end_score["survives"]:
            last_survive = rec
        elif first_die is None:
            first_die = rec

    if first_die is None:
        # chain survived the whole schedule — look harder: mix noise into down_proj
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(base_down.shape).astype(np.float32)
        noise *= float(np.linalg.norm(base_down) / (np.linalg.norm(noise) + 1e-30))
        for mix in (0.5, 1.0, 2.0, 4.0):
            wh = (base_down + mix * noise).astype(np.float32)
            h = x_student_embed.clone()
            for L in range(SHORT_CHAIN):
                layer = load_student_layer(art, cfg, L)
                if L == DEGRADE_LAYER:
                    inject_weight(layer, "mlp.down_proj.weight", wh)
                sites = run_layer_sites(layer, h, pos_emb, causal, pad, want_swiglu=False)
                h = sites["x_out"].contiguous()
                del layer, sites
                gc.collect()
            end_score = score(t_np(h), t_np(teacher_h[SHORT_CHAIN]), tag=f"degrade:noise x{mix}")
            rec = {"label": f"q4 down_proj + {mix}·matched-norm noise", "kind": "noise", "mix": mix, "score": end_score}
            boundary_rows.append(rec)
            print(f"    {rec['label']:<40} {'SURVIVES' if end_score['survives'] else 'FAILS'}  {fmt_score(end_score)}")
            if end_score["survives"]:
                last_survive = rec
            elif first_die is None:
                first_die = rec
                break

    if first_die is None:
        boundary = {
            "bracketed": False,
            "reason": (
                "short chain still survived after bit-collapse, 0-scale, and matched-norm "
                "noise on L0 down_proj. The residual stream is not bottlenecked on that organ "
                "alone at this depth — look at the free-run depth table for the real boundary."
            ),
            "last_survive": last_survive,
            "first_die": None,
        }
        watched.append(
            "boundary hunt on L0 down_proj did not kill the 4-layer chain; "
            "the residual skip is carrying the stream. Depth table is the authority."
        )
    else:
        boundary = {
            "bracketed": True,
            "organ": f"L{DEGRADE_LAYER} {DEGRADE_ORGAN}",
            "chain": f"free-run layers 0..{SHORT_CHAIN-1}",
            "last_survive": last_survive,
            "first_die": first_die,
            "bracket": (
                f"chain SURVIVES at {last_survive['label'] if last_survive else 'nothing'} "
                f"and FAILS at {first_die['label']}"
            ),
        }
        watched.append(f"bracketed failure boundary: {boundary['bracket']}")
        print(f"    BRACKET  {boundary['bracket']}")

    # If the undegraded free-run already dies at some depth, that is also a boundary.
    depth_boundary = None
    if first_fail_free is not None:
        prev = first_fail_free - 1
        depth_boundary = {
            "kind": "free_run_depth",
            "first_fail_layer": first_fail_free,
            "last_survive_layer": prev if prev >= 0 else None,
            "mixer": depth_rows[first_fail_free]["mixer"],
            "fail_score": depth_rows[first_fail_free]["free"],
            "prev_score": depth_rows[prev]["free"] if prev >= 0 else None,
        }
        watched.append(
            f"undegraded q4 free-run first FAILS the survival rule at layer {first_fail_free} "
            f"({depth_rows[first_fail_free]['mixer']}); last survive L{prev if prev>=0 else 'none'}"
        )
        print(
            f"    DEPTH BOUNDARY  undegraded free-run dies at L{first_fail_free} "
            f"(last survive L{prev if prev>=0 else 'none'})"
        )
    else:
        notes.append("undegraded q4 free-run SURVIVED all 64 layers under the named rule.")

    if first_fail_local is not None:
        watched.append(
            f"even teacher-forced local full-layer first FAILS at L{first_fail_local} "
            f"— not only a composition problem, a local one at that depth"
        )

    # ------------------------------------------------------------------
    # receipt + WHAT I WATCHED FAIL
    # ------------------------------------------------------------------
    receipt = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "artifact": {
            "root": str(ART),
            "schema": man.get("schema"),
            "complete_physical_bpw": man.get("complete_physical_bpw"),
            "q4_tensors": man.get("q4_tensors"),
            "f32_tensors": man.get("f32_tensors"),
            "min_q4_cosine_manifest": man.get("min_q4_cosine"),
            "status": man.get("status"),
        },
        "parent_bf16": str(SRC),
        "llama_server": {"url": LLAMA, "ok": llama_ok, "model": llama_model},
        "prompt": PROMPT,
        "token_ids": token_ids,
        "tokenize": tok_info,
        "survival_rule": {
            "scale_aware_over_null": SCALE_AWARE_OVER_NULL,
            "min_gain": MIN_GAIN,
            "max_rel_l2": MAX_REL_L2,
            "scale_aware_definition": "mean_row_cosine * gain, gain = mean min(r,1/r) of per-row L2 norms",
            "null": "constant-mean of the teacher reference, broadcast over rows (the GLM 0.898-class null, MEASURED per comparison, never assumed)",
        },
        "metric_rejects_scaled_artifact": sc_cos_only_trap,
        "scale_demo": {"identity": sc_id, "scaled_0.01W": sc_scale, "q4_gate": sc_q4},
        "rungs": rungs,
        "error_accumulation": {
            "assumed_linear": False,
            "measured": True,
            "monotonic_nondecreasing_free_rel_l2": monotonic,
            "free_rel_l2": free_rel,
            "local_rel_l2": local_rel,
            "free_scale_aware": free_sa,
            "first_fail_free_layer": first_fail_free,
            "first_fail_local_layer": first_fail_local,
            "per_layer": [
                {
                    "layer": r["layer"],
                    "mixer": r["mixer"],
                    "local_rel_l2": r["local"]["rel_l2"],
                    "free_rel_l2": r["free"]["rel_l2"],
                    "local_scale_aware": r["local"]["scale_aware"],
                    "free_scale_aware": r["free"]["scale_aware"],
                    "local_survives": r["local"]["survives"],
                    "free_survives": r["free"]["survives"],
                    "null_cosine_free": r["free"]["null_cosine"],
                }
                for r in depth_rows
            ],
        },
        "failure_boundary": {
            "organ_degrade": boundary,
            "undegraded_depth": depth_boundary,
            "schedule": boundary_rows,
        },
        "recovered_science_honoured": [
            "real activations only (BF16 parent hidden states of a live-tokenised prompt)",
            "no synthetic X (NS-009)",
            "scale-aware metric; cosine-only 0.01*W trap exhibited and rejected",
            "every fidelity number carries its constant-mean null",
            "storage BPW is not this measurement; this is function-space composition",
            "did not spawn a second 27B",
        ],
        "notes": notes,
        "watched_fail": watched,
        "wall_s": time.time() - t_all,
        "device": "cpu",
        "mps_available": bool(torch.backends.mps.is_available()),
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(jsonable(receipt), indent=2, sort_keys=False) + "\n")
    tmp.replace(RECEIPT)

    print("\n" + "=" * 78)
    print("## WHAT I WATCHED FAIL")
    print("=" * 78)
    if not watched:
        print("  (nothing failed — that would itself be a problem; the boundary hunt should have found one)")
    for i, w in enumerate(watched, 1):
        print(f"  {i}. {w}")
    print(f"\nreceipt {RECEIPT}  ({RECEIPT.stat().st_size} bytes)  wall {time.time()-t_all:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
