#!/usr/bin/env python3.12
"""Qwen3-235B-A22B MoE ORGAN adapter / inventory (Part VII/22).

Reads the official metadata ONLY (config.json + model.safetensors.index.json) and produces an
organ inventory that maps every tensor to an organ class, computes per-organ parameter and byte
counts, records which shard file holds each tensor (a bounded-read plan), and verifies the real
Qwen3 tensor-name scheme against the downloaded index. Nothing here loads, downloads, or runs the
438 GiB source; only the metadata is present locally (models/qwen3-235b-a22b/_meta/).

Honesty / validity boundary
  * The HF safetensors index (model.safetensors.index.json) carries name->shard mapping plus
    metadata.total_size ONLY. It does NOT carry per-tensor shapes or dtypes. Per-tensor shapes here
    are therefore DERIVED analytically from the config geometry (deterministic for qwen3_moe), not
    read from the index.
  * The derivation is CROSS-CHECKED byte-exact against the index metadata.total_size: the analytic
    parameter total * dtype_bytes must equal total_size to the byte. For the real model this holds
    exactly (235,093,634,560 params * 2 = 470,187,269,120 bytes), which validates the geometry.
  * A bounded-read stub can read TRUE shapes/bytes from the safetensors headers when the source is
    present, but the source shards are NOT local (only _meta is). That path is marked
    untested-pending-source and is never executed against the 438 GiB source here.

Verified architecture (config.json, revision ac9c66cc9b46af7306746a9250f23d47083d689e):
  qwen3_moe / Qwen3MoeForCausalLM, 94 layers, 128 experts, top-8, GQA 64 Q heads / 4 KV heads,
  head_dim 128 (decoupled: q_proj out = 64*128 = 8192, not hidden 4096), hidden 4096,
  moe_intermediate 1536, vocab 151936, bf16, tie_word_embeddings false.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Immutable pin recorded for provenance; the adapter reads local _meta, never the network.
IMMUTABLE_REVISION = "ac9c66cc9b46af7306746a9250f23d47083d689e"
HF_ID = "Qwen/Qwen3-235B-A22B"
EXPECTED_ARCH = "Qwen3MoeForCausalLM"
EXPECTED_MODEL_TYPE = "qwen3_moe"

DEFAULT_META = Path("models/qwen3-235b-a22b/_meta")

# safetensors dtype -> bytes-per-element (for the bounded-read verification path only).
_ST_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}
# config torch_dtype -> (safetensors_dtype_tag, bytes)
_TORCH_DTYPE = {
    "bfloat16": ("BF16", 2), "float16": ("F16", 2), "float32": ("F32", 4),
}
MAX_HEADER_BYTES = 200 * 1024 * 1024


class Qwen3MoeAdapterError(Exception):
    """Raised on any config/index inconsistency or unsafe source access."""


# ---------------------------------------------------------------------------
# Geometry (derived from config)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    tie_word_embeddings: bool
    torch_dtype: str
    model_type: str
    architecture: str

    @property
    def dtype_tag(self) -> str:
        return _TORCH_DTYPE.get(self.torch_dtype, ("BF16", 2))[0]

    @property
    def dtype_bytes(self) -> int:
        return _TORCH_DTYPE.get(self.torch_dtype, ("BF16", 2))[1]

    @property
    def q_out(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.num_key_value_heads * self.head_dim


def geometry_from_config(config: dict[str, Any]) -> Geometry:
    """Build Geometry from a qwen3_moe config dict. head_dim falls back to hidden/heads if absent."""
    arch = (config.get("architectures") or [None])[0]
    heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim") or (int(config["hidden_size"]) // heads))
    return Geometry(
        hidden_size=int(config["hidden_size"]),
        vocab_size=int(config["vocab_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=int(config["num_key_value_heads"]),
        head_dim=head_dim,
        num_experts=int(config["num_experts"]),
        num_experts_per_tok=int(config["num_experts_per_tok"]),
        moe_intermediate_size=int(config["moe_intermediate_size"]),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        torch_dtype=str(config.get("torch_dtype", "bfloat16")),
        model_type=str(config.get("model_type", "")),
        architecture=str(arch or ""),
    )


def load_config(meta_dir: Path = DEFAULT_META) -> dict[str, Any]:
    with open(Path(meta_dir) / "config.json") as fh:
        return json.load(fh)


def load_index(meta_dir: Path = DEFAULT_META) -> dict[str, Any]:
    with open(Path(meta_dir) / "model.safetensors.index.json") as fh:
        idx = json.load(fh)
    if "weight_map" not in idx or not isinstance(idx["weight_map"], dict):
        raise Qwen3MoeAdapterError("index missing weight_map")
    return idx


# ---------------------------------------------------------------------------
# Tensor name scheme (the real Qwen3 scheme, verified against the index)
# ---------------------------------------------------------------------------
# Organ classes are the collapsed-across-layers/experts identity of a tensor.
ORGAN_EMBED = "embed_tokens"
ORGAN_LM_HEAD = "lm_head"
ORGAN_FINAL_NORM = "model.norm"
ORGAN_Q = "self_attn.q_proj"
ORGAN_K = "self_attn.k_proj"
ORGAN_V = "self_attn.v_proj"
ORGAN_O = "self_attn.o_proj"
ORGAN_QNORM = "self_attn.q_norm"
ORGAN_KNORM = "self_attn.k_norm"
ORGAN_ILN = "input_layernorm"
ORGAN_PLN = "post_attention_layernorm"
ORGAN_ROUTER = "mlp.gate"            # the MoE router
ORGAN_EXP_GATE = "mlp.experts.gate_proj"
ORGAN_EXP_UP = "mlp.experts.up_proj"
ORGAN_EXP_DOWN = "mlp.experts.down_proj"

# Regexes over the canonical Qwen3MoeForCausalLM state-dict names.
_RE_TOP = {
    "model.embed_tokens.weight": ORGAN_EMBED,
    "lm_head.weight": ORGAN_LM_HEAD,
    "model.norm.weight": ORGAN_FINAL_NORM,
}
_RE_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)\.weight$")
_LAYER_SUFFIX = {
    "self_attn.q_proj": ORGAN_Q, "self_attn.k_proj": ORGAN_K,
    "self_attn.v_proj": ORGAN_V, "self_attn.o_proj": ORGAN_O,
    "self_attn.q_norm": ORGAN_QNORM, "self_attn.k_norm": ORGAN_KNORM,
    "input_layernorm": ORGAN_ILN, "post_attention_layernorm": ORGAN_PLN,
    "mlp.gate": ORGAN_ROUTER,
}
_RE_EXPERT = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
_EXPERT_SUFFIX = {
    "gate_proj": ORGAN_EXP_GATE, "up_proj": ORGAN_EXP_UP, "down_proj": ORGAN_EXP_DOWN,
}


def expert_name_pattern() -> str:
    """The exact routed-expert tensor-name pattern (regex, {L}=layer, {E}=expert)."""
    return r"model\.layers\.{L}\.mlp\.experts\.{E}\.(gate_proj|up_proj|down_proj)\.weight"


def router_name(layer: int) -> str:
    """The router (gate) tensor name for a given layer."""
    return f"model.layers.{layer}.mlp.gate.weight"


@dataclass(frozen=True)
class Classification:
    organ_class: str
    layer: int | None       # None for top-level (embed/lm_head/final norm)
    expert: int | None      # only set for routed-expert organs


def classify_tensor(name: str) -> Classification:
    """Map a state-dict tensor name to its organ class + (layer, expert) coordinates."""
    if name in _RE_TOP:
        return Classification(_RE_TOP[name], None, None)
    m = _RE_LAYER.match(name)
    if not m:
        raise Qwen3MoeAdapterError(f"unrecognized tensor name: {name!r}")
    layer, mid = int(m.group(1)), m.group(2)
    if mid in _LAYER_SUFFIX:
        return Classification(_LAYER_SUFFIX[mid], layer, None)
    me = _RE_EXPERT.match(mid)
    if me:
        return Classification(_EXPERT_SUFFIX[me.group(2)], layer, int(me.group(1)))
    raise Qwen3MoeAdapterError(f"unrecognized layer tensor name: {name!r}")


def expected_shape(geom: Geometry, name: str) -> tuple[int, ...]:
    """Analytic shape for a tensor name, derived from geometry (see module honesty note)."""
    cls = classify_tensor(name)
    oc, H = cls.organ_class, geom.hidden_size
    if oc in (ORGAN_EMBED, ORGAN_LM_HEAD):
        return (geom.vocab_size, H)
    if oc in (ORGAN_FINAL_NORM, ORGAN_ILN, ORGAN_PLN):
        return (H,)
    if oc == ORGAN_Q:
        return (geom.q_out, H)
    if oc in (ORGAN_K, ORGAN_V):
        return (geom.kv_out, H)
    if oc == ORGAN_O:
        return (H, geom.q_out)
    if oc in (ORGAN_QNORM, ORGAN_KNORM):
        return (geom.head_dim,)
    if oc == ORGAN_ROUTER:
        return (geom.num_experts, H)
    if oc in (ORGAN_EXP_GATE, ORGAN_EXP_UP):
        return (geom.moe_intermediate_size, H)
    if oc == ORGAN_EXP_DOWN:
        return (H, geom.moe_intermediate_size)
    raise Qwen3MoeAdapterError(f"no shape rule for organ {oc!r}")


def expected_tensor_names(geom: Geometry) -> set[str]:
    """Generate the full set of tensor names the qwen3_moe architecture must contain."""
    names: set[str] = {"model.embed_tokens.weight", "model.norm.weight"}
    if not geom.tie_word_embeddings:
        names.add("lm_head.weight")
    for L in range(geom.num_hidden_layers):
        base = f"model.layers.{L}."
        for suf in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
                    "input_layernorm", "post_attention_layernorm", "mlp.gate"):
            names.add(f"{base}{suf}.weight")
        for E in range(geom.num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names.add(f"{base}mlp.experts.{E}.{proj}.weight")
    return names


# ---------------------------------------------------------------------------
# Organ inventory
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrganTensor:
    name: str
    organ_class: str
    layer: int | None
    expert: int | None
    shape: tuple[int, ...]
    dtype: str
    param_count: int
    byte_count: int
    shard_file: str          # the bounded-read plan: which shard holds this tensor


@dataclass
class OrganInventory:
    geometry: Geometry
    tensors: list[OrganTensor]
    per_class_params: dict[str, int] = field(default_factory=dict)
    per_class_bytes: dict[str, int] = field(default_factory=dict)
    per_class_count: dict[str, int] = field(default_factory=dict)
    grand_params: int = 0
    grand_bytes: int = 0
    index_total_size: int | None = None
    shard_files: list[str] = field(default_factory=list)

    def cross_check_ok(self) -> bool:
        """True iff analytic grand bytes == index metadata.total_size (byte-exact)."""
        return self.index_total_size is not None and self.grand_bytes == self.index_total_size


def build_inventory(config: dict[str, Any], index: dict[str, Any]) -> OrganInventory:
    """Build the full organ inventory from config + index weight_map. No weights are read."""
    geom = geometry_from_config(config)
    weight_map: dict[str, str] = index["weight_map"]
    dtag, dbytes = geom.dtype_tag, geom.dtype_bytes

    tensors: list[OrganTensor] = []
    per_p: dict[str, int] = {}
    per_b: dict[str, int] = {}
    per_c: dict[str, int] = {}
    for name in sorted(weight_map):
        cls = classify_tensor(name)
        shape = expected_shape(geom, name)
        params = math.prod(shape)
        nbytes = params * dbytes
        tensors.append(OrganTensor(
            name=name, organ_class=cls.organ_class, layer=cls.layer, expert=cls.expert,
            shape=shape, dtype=dtag, param_count=params, byte_count=nbytes,
            shard_file=weight_map[name],
        ))
        per_p[cls.organ_class] = per_p.get(cls.organ_class, 0) + params
        per_b[cls.organ_class] = per_b.get(cls.organ_class, 0) + nbytes
        per_c[cls.organ_class] = per_c.get(cls.organ_class, 0) + 1

    inv = OrganInventory(
        geometry=geom, tensors=tensors,
        per_class_params=per_p, per_class_bytes=per_b, per_class_count=per_c,
        grand_params=sum(per_p.values()), grand_bytes=sum(per_b.values()),
        index_total_size=index.get("metadata", {}).get("total_size"),
        shard_files=sorted(set(weight_map.values())),
    )
    return inv


# ---------------------------------------------------------------------------
# Index-name verification
# ---------------------------------------------------------------------------
@dataclass
class NameVerification:
    ok: bool
    missing: list[str]           # expected-but-absent
    unexpected: list[str]        # present-but-unclassifiable / not expected
    n_expected: int
    n_index: int
    router_example: str
    expert_pattern: str
    router_present: bool
    q_norm_present: bool
    k_norm_present: bool


def verify_index_names(geom: Geometry, index: dict[str, Any]) -> NameVerification:
    """Assert the expected qwen3_moe tensor names exist in the index and nothing extra is present."""
    idx_names = set(index["weight_map"])
    exp_names = expected_tensor_names(geom)
    missing = sorted(exp_names - idx_names)
    unexpected = sorted(idx_names - exp_names)
    router0 = router_name(0)
    return NameVerification(
        ok=(not missing and not unexpected),
        missing=missing, unexpected=unexpected,
        n_expected=len(exp_names), n_index=len(idx_names),
        router_example=router0, expert_pattern=expert_name_pattern(),
        router_present=router0 in idx_names,
        q_norm_present="model.layers.0.self_attn.q_norm.weight" in idx_names,
        k_norm_present="model.layers.0.self_attn.k_norm.weight" in idx_names,
    )


# ---------------------------------------------------------------------------
# Top-k routing shape logic (pure numpy; exercises the MoE routing contract)
# ---------------------------------------------------------------------------
def topk_route(router_logits, top_k: int, norm_topk_prob: bool = True):
    """Given router logits [n_tokens, n_experts], return (expert_idx, weights) each [n_tokens, k].

    Mirrors Qwen3MoE routing: softmax over experts, take top-k, optionally renormalize the k
    selected probabilities (norm_topk_prob). Shape-and-contract only, not a fused kernel.
    """
    import numpy as np
    logits = np.asarray(router_logits, dtype=np.float32)
    if logits.ndim != 2:
        raise Qwen3MoeAdapterError("router_logits must be [n_tokens, n_experts]")
    n_tok, n_exp = logits.shape
    if not (1 <= top_k <= n_exp):
        raise Qwen3MoeAdapterError(f"top_k {top_k} out of range for {n_exp} experts")
    probs = _softmax(logits)
    idx = np.argsort(-probs, axis=1)[:, :top_k]                 # [n_tok, k]
    w = np.take_along_axis(probs, idx, axis=1)                  # [n_tok, k]
    if norm_topk_prob:
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-20, None)
    return idx, w


def _softmax(x):
    import numpy as np
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Bounded-read stub (safetensors header offsets) - UNTESTED, PENDING SOURCE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundedReadPlan:
    tensor: str
    shard_file: str
    shard_path: str
    status: str            # "ready" if shard present, else "untested-pending-source"
    reason: str


def plan_bounded_read(inv: OrganInventory, tensor_name: str,
                      source_dir: Path) -> BoundedReadPlan:
    """Produce (but do NOT execute) a bounded single-tensor read plan for tensor_name.

    Returns the shard the tensor lives in and whether the shard is present locally. When the shard
    is absent (the 438 GiB source is not staged), status is untested-pending-source and no bytes are
    ever read. read_one_tensor_bytes() performs the actual header-scoped read only when a caller
    explicitly opts in AND the shard is present.
    """
    match = next((t for t in inv.tensors if t.name == tensor_name), None)
    if match is None:
        raise Qwen3MoeAdapterError(f"{tensor_name!r} not in inventory")
    shard_path = Path(source_dir) / match.shard_file
    if shard_path.is_file():
        return BoundedReadPlan(tensor_name, match.shard_file, str(shard_path),
                               "ready", "shard present; call read_one_tensor_bytes to execute")
    return BoundedReadPlan(tensor_name, match.shard_file, str(shard_path),
                           "untested-pending-source",
                           f"shard {match.shard_file} not present under {source_dir}; "
                           "source is the 438 GiB weight set, intentionally not staged")


def read_one_tensor_bytes(shard_path: Path, tensor_name: str, *, max_bytes: int = 0) -> dict:
    """Read ONE tensor's header entry (and optionally its byte range) from a safetensors shard.

    Bounded: opens the shard read-only (no symlink follow), reads only the header, locates the named
    tensor, and reads at most max_bytes of its data range (0 = header/metadata only, zero data
    bytes). This is the ONLY code path that would touch a real shard; it is untested against the
    438 GiB source because the shards are not present locally.
    """
    path = Path(shard_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise Qwen3MoeAdapterError(f"shard is not a regular file: {path}")
        prefix = os.pread(fd, 8, 0)
        if len(prefix) != 8:
            raise Qwen3MoeAdapterError(f"truncated safetensors length: {path}")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > MAX_HEADER_BYTES or 8 + header_len > st.st_size:
            raise Qwen3MoeAdapterError(f"unsafe safetensors header length: {path}")
        header = os.pread(fd, header_len, 8)
        if len(header) != header_len:
            raise Qwen3MoeAdapterError(f"truncated safetensors header: {path}")
        parsed = json.loads(header)
        if tensor_name not in parsed:
            raise Qwen3MoeAdapterError(f"{tensor_name!r} not in shard header {path.name}")
        row = parsed[tensor_name]
        dtype, shape, offsets = row["dtype"], tuple(row["shape"]), row["data_offsets"]
        start, end = offsets
        elem_bytes = _ST_DTYPE_BYTES.get(dtype)
        if elem_bytes is None:
            raise Qwen3MoeAdapterError(f"unsupported dtype {dtype!r} for {tensor_name}")
        expected = math.prod(shape) * elem_bytes
        if end - start != expected:
            raise Qwen3MoeAdapterError(
                f"header extent mismatch for {tensor_name}: {end - start} != {expected}")
        data_start = 8 + header_len + start
        n = min(max_bytes, end - start) if max_bytes > 0 else 0
        sample = os.pread(fd, n, data_start) if n > 0 else b""
        return {
            "tensor": tensor_name, "dtype": dtype, "shape": shape,
            "byte_count": end - start, "data_start": data_start,
            "read_bytes": len(sample),
        }
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
_ORGAN_ORDER = [
    ORGAN_EMBED, ORGAN_Q, ORGAN_K, ORGAN_V, ORGAN_O, ORGAN_QNORM, ORGAN_KNORM,
    ORGAN_ILN, ORGAN_PLN, ORGAN_ROUTER, ORGAN_EXP_GATE, ORGAN_EXP_UP, ORGAN_EXP_DOWN,
    ORGAN_FINAL_NORM, ORGAN_LM_HEAD,
]


def render_report(inv: OrganInventory, ver: NameVerification) -> str:
    g = inv.geometry
    gib = inv.grand_bytes / 1024 ** 3
    lines = [
        f"Qwen3-235B-A22B MoE organ inventory  (rev {IMMUTABLE_REVISION})",
        f"  arch={g.architecture} model_type={g.model_type} dtype={g.torch_dtype}",
        f"  layers={g.num_hidden_layers} experts={g.num_experts} top_k={g.num_experts_per_tok} "
        f"GQA {g.num_attention_heads}Q/{g.num_key_value_heads}KV head_dim={g.head_dim} "
        f"hidden={g.hidden_size} moe_int={g.moe_intermediate_size} vocab={g.vocab_size}",
        f"  q_proj out={g.q_out} (=heads*head_dim, decoupled from hidden)  kv_proj out={g.kv_out}",
        f"  shards={len(inv.shard_files)} tensors={len(inv.tensors)}",
        "",
        f"  {'organ_class':<28}{'tensors':>9}{'params':>18}{'bytes':>18}",
    ]
    for oc in _ORGAN_ORDER:
        if oc in inv.per_class_params:
            lines.append(f"  {oc:<28}{inv.per_class_count[oc]:>9}"
                         f"{inv.per_class_params[oc]:>18,}{inv.per_class_bytes[oc]:>18,}")
    lines += [
        f"  {'-' * 71}",
        f"  {'GRAND TOTAL':<28}{len(inv.tensors):>9}{inv.grand_params:>18,}{inv.grand_bytes:>18,}",
        "",
        f"  grand_params = {inv.grand_params:,} (~{inv.grand_params / 1e9:.3f}B)",
        f"  grand_bytes  = {inv.grand_bytes:,} (~{gib:.2f} GiB)",
        f"  index total_size = {inv.index_total_size:,}"
        if inv.index_total_size is not None else "  index total_size = (absent)",
        f"  byte-exact cross-check (analytic bytes == index total_size): "
        f"{'PASS' if inv.cross_check_ok() else 'FAIL'}",
        "",
        f"  name verification: {'PASS' if ver.ok else 'FAIL'} "
        f"(expected {ver.n_expected}, index {ver.n_index}, "
        f"missing {len(ver.missing)}, unexpected {len(ver.unexpected)})",
        f"  router tensor (layer 0): {ver.router_example}  present={ver.router_present}",
        f"  expert pattern: {ver.expert_pattern}",
        f"  q_norm present={ver.q_norm_present}  k_norm present={ver.k_norm_present}",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    meta = Path(argv[1]) if len(argv) > 1 else DEFAULT_META
    config = load_config(meta)
    index = load_index(meta)
    geom = geometry_from_config(config)
    if geom.architecture and geom.architecture != EXPECTED_ARCH:
        print(f"WARNING: architecture {geom.architecture!r} != {EXPECTED_ARCH!r}", file=sys.stderr)
    inv = build_inventory(config, index)
    ver = verify_index_names(geom, index)
    print(render_report(inv, ver))
    # a bounded-read plan for one representative expert tensor (never executed here)
    plan = plan_bounded_read(inv, "model.layers.0.mlp.experts.0.gate_proj.weight", meta)
    print(f"\n  bounded-read plan [{plan.tensor}] -> {plan.shard_file}: {plan.status}")
    print(f"    {plan.reason}")
    return 0 if (inv.cross_check_ok() and ver.ok) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
