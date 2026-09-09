#!/usr/bin/env python3.12
"""KSB001 -- complete organ-partitioned byte anatomy that CLOSES EXACTLY.

Every EBPW claim is a fraction, and a fraction is only as honest as its
denominator. S013's rule: no denominator games, no omitted organs, no vision
bytes silently removed from one side only. If vision is excluded it is excluded
from BOTH numerator and source-parameter denominator, and the resulting object is
named precisely.

So this partitions every tensor into an organ, sums the bytes, and CHECKS THE SUM
AGAINST THE FILES ON DISK. A partition that does not close is a partition with a
category nobody thought of, and silently dropping the remainder is how a
denominator shrinks by accident.

Closure is exact, not approximate: safetensors files are a JSON header plus a
packed payload, so file_bytes == header_bytes + sum(tensor bytes). Both are
computed and both are reported. A residual of even one byte fails.

Two named objects come out, because they have different denominators:

  FULL          everything in the checkpoint, vision tower included
  LANGUAGE      the language tower only -- what a text-only promoted body would
                be -- with vision and its projector removed from BOTH sides

Quoting a language-tower rate against a full-model parameter count would be the
denominator game the gate exists to prevent, so the two are never mixed.

    python3 tools/future/organ_byte_anatomy.py --spec <dir>
    python3 tools/future/organ_byte_anatomy.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Ordered: the FIRST pattern that matches wins, so the specific ones lead.
ORGANS: tuple[tuple[str, str, str], ...] = (
    ("vision",            "vision", r"^vision_tower\."),
    ("mm_projector",      "vision", r"^multi_modal_projector\."),
    ("routed_experts",    "language", r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$"),
    ("shared_experts",    "language", r"\.mlp\.shared_experts\.(gate|up|down)_proj\.weight$"),
    ("router",            "language", r"\.mlp\.gate\.(weight|e_score_correction_bias)$"),
    ("dense_mlp",         "language", r"\.mlp\.(gate|up|down)_proj\.weight$"),
    ("attention_mla",     "language", r"\.self_attn\.(q_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)\.weight$"),
    ("attention_norms",   "language", r"\.self_attn\.kv_a_layernorm\.weight$"),
    ("rotary_buffers",    "language", r"\.rotary_emb\.inv_freq$"),
    ("layer_norms",       "language", r"\.(input_layernorm|post_attention_layernorm)\.weight$"),
    ("final_norm",        "language", r"^language_model\.model\.norm\.weight$|^model\.norm\.weight$"),
    ("embeddings",        "language", r"\.embed_tokens\.weight$"),
    ("lm_head",           "language", r"\.lm_head\.weight$|^lm_head\.weight$"),
)

DTYPE_BITS = {"BF16": 16, "F16": 16, "F32": 32, "F64": 64, "BOOL": 8,
              "I8": 8, "U8": 8, "I16": 16, "U16": 16, "I32": 32, "U32": 32,
              "I64": 64, "U64": 64, "F8_E4M3": 8, "F8_E5M2": 8}


def classify(name: str) -> tuple[str, str]:
    for organ, tower, pat in ORGANS:
        if re.search(pat, name):
            return organ, tower
    return "UNCLASSIFIED", "unknown"


def read_headers(spec: Path):
    """safetensors header: 8-byte little-endian length, then that many JSON bytes.
    Read directly rather than through the library so the HEADER SIZE itself is
    available -- it is real bytes on disk and closure needs it."""
    out = []
    for f in sorted(spec.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            meta = json.loads(fh.read(n))
        out.append((f, 8 + n, meta))
    return out


def anatomy(spec: Path) -> dict:
    organs: dict[str, dict] = {}
    total_tensor_bytes = total_header_bytes = total_file_bytes = 0
    unclassified: list[str] = []

    for path, hdr_bytes, meta in read_headers(spec):
        total_header_bytes += hdr_bytes
        total_file_bytes += path.stat().st_size
        for name, info in meta.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            nbytes = e - s
            numel = 1
            for d in info["shape"]:
                numel *= d
            bits = DTYPE_BITS.get(info["dtype"])
            if bits is None:
                raise ValueError(f"unknown dtype {info['dtype']} on {name}; "
                                 "refusing to guess a width in a byte-closure check")
            if numel * bits // 8 != nbytes:
                raise ValueError(f"{name}: shape/dtype implies {numel*bits//8} bytes "
                                 f"but the offsets span {nbytes}")
            organ, tower = classify(name)
            if organ == "UNCLASSIFIED":
                unclassified.append(name)
            d = organs.setdefault(organ, {"tower": tower, "bytes": 0, "params": 0,
                                          "tensors": 0, "dtypes": {}})
            d["bytes"] += nbytes
            d["params"] += numel
            d["tensors"] += 1
            d["dtypes"][info["dtype"]] = d["dtypes"].get(info["dtype"], 0) + numel
            total_tensor_bytes += nbytes

    accounted = total_tensor_bytes + total_header_bytes
    residual = total_file_bytes - accounted

    def tower(t):
        b = sum(v["bytes"] for v in organs.values() if v["tower"] == t)
        p = sum(v["params"] for v in organs.values() if v["tower"] == t)
        return {"bytes": b, "params": p,
                "source_bits_per_weight": round(b * 8 / p, 6) if p else 0.0}

    return {
        "schema": "hawking.future.organ_byte_anatomy.v1",
        "obligation": "KSB001", "steer": "S013",
        "specimen": str(spec),
        "closure": {
            "file_bytes": total_file_bytes,
            "safetensors_header_bytes": total_header_bytes,
            "tensor_payload_bytes": total_tensor_bytes,
            "accounted_bytes": accounted,
            "residual_bytes": residual,
            "closes_exactly": residual == 0},
        "unclassified_tensors": unclassified,
        "organs": {k: {**v, "bytes_pct": round(v["bytes"] * 100 / max(1, total_tensor_bytes), 4)}
                   for k, v in sorted(organs.items(), key=lambda kv: -kv[1]["bytes"])},
        "objects": {
            "FULL": {**tower("vision"), "note": "vision only"} and {
                "bytes": total_tensor_bytes + total_header_bytes,
                "params": sum(v["params"] for v in organs.values()),
                "note": "everything in the checkpoint, vision included; header charged"},
            "LANGUAGE": {**tower("language"),
                         "note": "language tower only -- vision tower and its projector "
                                 "removed from BOTH numerator and denominator"},
            "VISION_EXCLUDED": tower("vision")},
    }


def _selftest() -> int:
    """The classifier must place every real Kimi tensor name AND must not place a
    name it has never seen. A catch-all that silently absorbs the unknown would
    make the closure check unable to fail."""
    known = [
        "language_model.model.layers.3.mlp.experts.17.gate_proj.weight",
        "language_model.model.layers.3.mlp.shared_experts.down_proj.weight",
        "language_model.model.layers.3.mlp.gate.weight",
        "language_model.model.layers.3.mlp.gate.e_score_correction_bias",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.3.self_attn.kv_b_proj.weight",
        "language_model.model.layers.3.self_attn.kv_a_layernorm.weight",
        "language_model.model.layers.3.self_attn.rotary_emb.inv_freq",
        "language_model.model.layers.3.input_layernorm.weight",
        "language_model.model.norm.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "vision_tower.encoder.blocks.2.wqkv.weight",
        "multi_modal_projector.linear_1.weight",
    ]
    for n in known:
        organ, tower = classify(n)
        assert organ != "UNCLASSIFIED", f"a real tensor name fell through: {n}"
        assert tower in ("language", "vision"), (organ, tower, n)
    # Negative control: an invented organ must NOT be absorbed.
    for n in ("model.layers.3.mlp.quantum_proj.weight", "some_new_tower.thing"):
        assert classify(n)[0] == "UNCLASSIFIED", (
            f"{n} was classified, so the closure check can never surface a "
            "category nobody thought of")
    # Ordering control: the routed-expert pattern must win over dense_mlp, which
    # would otherwise swallow it and merge two organs with very different rates.
    assert classify("m.layers.1.mlp.experts.0.gate_proj.weight")[0] == "routed_experts"
    assert classify("m.layers.0.mlp.gate_proj.weight")[0] == "dense_mlp"
    # And the router must not be read as a dense gate_proj.
    assert classify("m.layers.1.mlp.gate.weight")[0] == "router"
    print("selftest OK: 14 real names classified, 2 invented names refused, "
          "expert/dense/router ordering holds")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "receipts" / "future" / "ORGAN_BYTE_ANATOMY.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.spec:
        ap.error("--spec is required unless --selftest")

    rec = anatomy(a.spec)
    c = rec["closure"]
    print(f"{'organ':<18}{'tower':<10}{'GiB':>10}{'params':>16}{'%':>8}")
    for k, v in rec["organs"].items():
        print(f"{k:<18}{v['tower']:<10}{v['bytes']/2**30:>10.3f}"
              f"{v['params']:>16,}{v['bytes_pct']:>8.2f}")
    print(f"\nfile bytes      {c['file_bytes']:,}")
    print(f"tensor payload  {c['tensor_payload_bytes']:,}")
    print(f"header bytes    {c['safetensors_header_bytes']:,}")
    print(f"residual        {c['residual_bytes']:,}   CLOSES: {c['closes_exactly']}")
    if rec["unclassified_tensors"]:
        print(f"UNCLASSIFIED    {len(rec['unclassified_tensors'])}: "
              f"{rec['unclassified_tensors'][:4]}")
    for name, o in rec["objects"].items():
        if "source_bits_per_weight" in o:
            print(f"\n{name:<16} {o['bytes']/2**30:.3f} GiB over {o['params']:,} params "
                  f"= {o['source_bits_per_weight']:.4f} source bits/weight")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")
    return 0 if c["closes_exactly"] and not rec["unclassified_tensors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
