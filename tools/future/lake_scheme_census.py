"""What can Odyssey-I actually reach, across the whole lake, without loading anything.

G034/G035. The campaign is defined over 56 bodies and 4.29 TiB, and the obvious
reading of those numbers -- that this is 4.29 TiB of bf16 parent weights waiting
to be compressed -- is WRONG for the largest bodies. Three of the four biggest are
already quantized on disk. Kimi-K3's 1453.8 GiB is packed U8 with separate scales;
GLM-5.3-Flash is F8_E4M3; DeepSeek-V4-Flash is I8. Their complete-EBPW baseline is
nowhere near 16 bits per source parameter, so a "compression ratio" measured against
their byte count would be measuring the wrong denominator.

This reads safetensors headers only -- 8-byte little-endian length, then JSON of
{dtype, shape, data_offsets} -- so the whole 4.29 TiB lake is classified in minutes
without a single payload byte being read.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import struct

FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
SCALE_SUFFIXES = (".scale", ".weight_scale", ".weight_scale_inv")
_LAYER_RE = re.compile(r"layers\.(\d+)")


def read_header(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: shorter than a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        if not 0 < n < (1 << 31):
            raise ValueError(f"{path}: implausible header length {n}; not safetensors")
        return {k: v for k, v in json.loads(fh.read(n)).items() if k != "__metadata__"}


def _walk_files(path: str):
    for root, _dirs, files in os.walk(path):
        for f in files:
            yield f


def classify(path: str) -> dict:
    """One specimen -> what OI can reach and the exact mechanism when it cannot."""
    # RECURSIVE. A non-recursive glob classified Wan2.2-T2V-A14B and
    # audio-flamingo-3 as having no safetensors at all; they keep 12 and 7 of
    # them one directory down. Two of nine "NO-SAFETENSORS" verdicts were the
    # very defect this module exists to catch.
    shards = sorted(glob.glob(path.rstrip("/") + "/**/*.safetensors", recursive=True))
    if not shards:
        present = (collections.Counter(os.path.splitext(f)[1] or "<noext>"
                                       for f in _walk_files(path))
                   if os.path.isdir(path) else collections.Counter())
        return {"klass": "NO-SAFETENSORS", "n_shards": 0, "n_tensors": 0,
                "blocked_by": "weights are " + ", ".join(f"{n}x{e}" for e, n in present.most_common(4)),
                "present_extensions": dict(present)}

    n_tensors = 0
    expert_dtypes: collections.Counter = collections.Counter()
    expert_layers: set[int] = set()
    scale_keys: set[str] = set()
    patterns: collections.Counter = collections.Counter()
    for f in shards:
        for k, e in read_header(f).items():
            n_tensors += 1
            if "expert" not in k.lower():
                continue
            if k.endswith(SCALE_SUFFIXES):
                scale_keys.add(k.rsplit(".", 1)[-1])
                continue
            expert_dtypes[e["dtype"]] += 1
            patterns[re.sub(r"\.\d+", ".N", k)] += 1
            m = _LAYER_RE.search(k)
            if m:
                expert_layers.add(int(m.group(1)))

    out = {"n_shards": len(shards), "n_tensors": n_tensors,
           "expert_dtypes": dict(expert_dtypes),
           "expert_layers": ([min(expert_layers), max(expert_layers)]
                             if expert_layers else None),
           "scale_keys": sorted(scale_keys),
           "key_pattern": patterns.most_common(1)[0][0] if patterns else None}

    if not expert_dtypes:
        out.update(klass="DENSE",
                   blocked_by="no key contains 'expert'; expert-organ anatomy is "
                              "undefined for this body -- it owes a DENSE anatomy instead")
        return out
    nonfloat = {d for d in expert_dtypes if d not in FLOAT_DTYPES}
    if nonfloat:
        out.update(klass="MOE-PREQUANTIZED",
                   blocked_by=f"expert payload is {sorted(nonfloat)} with "
                              f"{out['scale_keys'] or 'no'} scale tensors -- a spectrum over "
                              f"quantization codes measures the codebook, not the organism")
        return out
    out.update(klass="MOE-FLOAT", blocked_by="")
    return out


def census(catalog_path: str) -> dict:
    cat = json.load(open(catalog_path))
    rows = []
    for s in sorted(cat["specimens"], key=lambda x: -x["bytes"]):
        try:
            r = classify(s["path"])
        except Exception as exc:                       # a header we cannot parse is a FINDING
            r = {"klass": "HEADER-UNREADABLE", "blocked_by": f"{type(exc).__name__}: {exc}"}
        r.update(slug=s["slug"], gib=round(s["bytes"] / 2 ** 30, 1),
                 family=s["architecture_family"])
        rows.append(r)
    tally = collections.Counter(r["klass"] for r in rows)
    reachable = [r for r in rows if r["klass"] == "MOE-FLOAT"]
    return {"n": len(rows), "tally": dict(tally), "rows": rows,
            "expert_anatomy_reachable": len(reachable),
            "reachable_gib": round(sum(r["gib"] for r in reachable), 1),
            "blocked_gib": round(sum(r["gib"] for r in rows if r["klass"] != "MOE-FLOAT"), 1)}


def _selfcheck() -> None:
    """Absence must never look like a measurement of zero."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "pytorch_model.bin"), "wb").write(b"\0")
        r = classify(d)
        assert r["klass"] == "NO-SAFETENSORS", r
        assert ".bin" in r["blocked_by"], r["blocked_by"]

        # A safetensors file whose experts are int8 must be refused, not measured.
        hdr = {"model.layers.0.mlp.experts.0.w1.weight":
               {"dtype": "I8", "shape": [4, 4], "data_offsets": [0, 16]},
               "model.layers.0.mlp.experts.0.w1.scale":
               {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]}}
        blob = json.dumps(hdr).encode()
        with open(os.path.join(d, "m.safetensors"), "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)) + blob + b"\0" * 32)
        r = classify(d)
        assert r["klass"] == "MOE-PREQUANTIZED", r
        assert "I8" in r["blocked_by"] and "scale" in r["blocked_by"], r["blocked_by"]
        assert r["expert_layers"] == [0, 0], r

        # Same file with float experts is reachable.
        hdr = {"model.layers.3.mlp.experts.0.down_proj.weight":
               {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}}
        blob = json.dumps(hdr).encode()
        with open(os.path.join(d, "m.safetensors"), "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)) + blob + b"\0" * 32)
        r = classify(d)
        assert r["klass"] == "MOE-FLOAT" and r["blocked_by"] == "", r
        assert r["expert_layers"] == [3, 3], r

        # A nested layout must be FOUND, not reported as no-safetensors.
        import shutil
        nest = os.path.join(d, "low_noise_model")
        os.makedirs(nest, exist_ok=True)
        shutil.move(os.path.join(d, "m.safetensors"), os.path.join(nest, "m.safetensors"))
        r = classify(d)
        assert r["klass"] != "NO-SAFETENSORS", r
        assert r["n_shards"] == 1, r
        shutil.move(os.path.join(nest, "m.safetensors"), os.path.join(d, "m.safetensors"))

        # No experts at all is DENSE, not an empty MoE answer.
        hdr = {"model.layers.0.self_attn.q_proj.weight":
               {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}}
        blob = json.dumps(hdr).encode()
        with open(os.path.join(d, "m.safetensors"), "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)) + blob + b"\0" * 32)
        assert classify(d)["klass"] == "DENSE", classify(d)
    print("selfcheck OK")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        print(json.dumps(census(sys.argv[1]), indent=1))
