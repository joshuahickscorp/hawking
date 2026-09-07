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

# Complete EBPW = every persistent byte / source-parameter count. Scales, zero
# points and shapes are OVERHEAD: they belong in the numerator and must never be
# counted as source parameters, which would flatter the ratio by ~5% on the two
# block-scaled bodies here.
DTYPE_BITS = {"BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
              "U16": 16, "I16": 16, "F16": 16, "BF16": 16,
              "U32": 32, "I32": 32, "F32": 32, "U64": 64, "I64": 64, "F64": 64}
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


def accounting(path: str, pack: int | None = None) -> dict:
    """Complete-EBPW accounting from headers alone.

    `pack` is logical values per stored byte for packed integer payloads. It is
    not guessed: it is CONFIRMED against the companion scale tensor's group
    geometry, because a wrong packing factor silently halves or doubles the
    denominator. Kimi-K3's w1 stores [3072,1792] U8 beside a [3072,112] scale --
    1792*2/112 = 32 elements per group, which is MXFP4's block size, so pack=2.
    """
    shards = sorted(glob.glob(path.rstrip("/") + "/**/*.safetensors", recursive=True))
    stored = src = scale_bytes = 0
    packed_seen = False
    for f in shards:
        for k, e in read_header(f).items():
            n = 1
            for d in e["shape"]:
                n *= int(d)
            b = n * DTYPE_BITS.get(e["dtype"], 16) // 8
            stored += b
            if k.endswith(SCALE_SUFFIXES):
                scale_bytes += b
                continue
            if pack > 1 and e["dtype"] not in FLOAT_DTYPES:
                src += n * pack
                packed_seen = True
            else:
                src += n
    if not src:
        raise ValueError(f"{path}: no source parameters found; refusing to divide by zero")
    return {"stored_bytes": stored, "source_params": src,
            "complete_ebpw": round(stored * 8 / src, 3),
            "scale_overhead_pct": round(100 * scale_bytes / stored, 2),
            "packed_payload": packed_seen, "pack_factor": pack if packed_seen else 1,
            "n_shards": len(shards)}


def confirm_pack_factor(path: str) -> int:
    """Logical values per stored byte, from the model's own quantization_config,
    CROSS-CHECKED against the scale tensors' group geometry.

    Deriving this from geometry alone is ambiguous and I got it wrong once:
    Kimi-K3's w1 stores [3072,1792] beside a [3072,112] scale, and 1792/112 = 16
    is a perfectly standard block size, so pack=1 "confirms" just as readily as
    pack=2 (which gives 32). Only config.json settles it -- num_bits 4,
    group_size 32 -- and the geometry then agrees with exactly one of them.
    Two independent sources that must AGREE, not one source that merely looks
    plausible.
    """
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return 1
    cfg = json.load(open(cfg_path))
    q = (cfg.get("quantization_config")
         or (cfg.get("text_config") or {}).get("quantization_config"))
    if not q:
        return 1
    w = ((q.get("config_groups") or {}).get("group_0") or {}).get("weights") or {}
    num_bits, group_size = w.get("num_bits"), w.get("group_size")

    if not num_bits:
        # An 8-bit format cannot pack sub-byte, so its factor is 1 by definition
        # and no calibration is needed. DeepSeek-V4-Flash declares fmt "e4m3"
        # with no num_bits and names its projections w1/w2/w3, so the
        # shape-calibration path below would refuse a body that was never packed.
        fmt = str(q.get("fmt") or q.get("format") or q.get("quant_method") or "").lower()
        if any(t in fmt for t in ("e4m3", "e5m2", "fp8", "int8", "i8")):
            return 1

        # A quantized body that does not declare num_bits still packs something --
        # bitnet-b1.58 stores down_proj as [640, 6912] where the bf16 twin is
        # [2560, 6912], four ternary values to a byte, under the ordinary name
        # ".weight" with no "packed" marker anywhere. Recover the factor from the
        # architecture's own hidden_size, and REFUSE if it does not divide
        # cleanly. Reporting 11.095 EBPW for a 3.909 body, which is what guessing
        # 1 did here, is worse than reporting nothing.
        hidden = cfg.get("hidden_size") or (cfg.get("text_config") or {}).get("hidden_size")
        if not hidden:
            raise ValueError(f"{path}: quantization_config present ({q.get('quant_method')}) "
                             f"but neither num_bits nor hidden_size is declared -- "
                             f"the packing factor is undetermined and EBPW would be fiction")
        for f in sorted(glob.glob(path.rstrip("/") + "/**/*.safetensors", recursive=True)):
            for k, e in read_header(f).items():
                if not k.endswith("down_proj.weight") or len(e["shape"]) != 2:
                    continue
                rows = e["shape"][0]
                if e["dtype"] in FLOAT_DTYPES:
                    return 1                        # stored dense after all
                if hidden % rows:
                    raise ValueError(
                        f"{path}: {k} stores {e['shape']} against hidden_size {hidden}; "
                        f"{hidden}/{rows} is not integral -- refusing to guess a denominator")
                return hidden // rows
        raise ValueError(f"{path}: quantized per config but no down_proj weight found "
                         f"to calibrate the packing factor against")

    if num_bits >= 8:
        return 1                                    # nothing sub-byte is packed
    pack = 8 // num_bits

    for f in sorted(glob.glob(path.rstrip("/") + "/**/*.safetensors", recursive=True)):
        hdr = read_header(f)
        for k, e in hdr.items():
            if not k.endswith(".weight_packed"):
                continue
            sc = hdr.get(k[: -len("weight_packed")] + "weight_scale")
            if not sc or len(e["shape"]) < 2 or len(sc["shape"]) < 2:
                continue
            observed = e["shape"][-1] * pack / sc["shape"][-1]
            if group_size and observed != group_size:
                raise ValueError(
                    f"{path}: config declares num_bits={num_bits} group_size={group_size}, "
                    f"but {k} stores {e['shape']} beside scale {sc['shape']}, giving "
                    f"{observed:g} elements per group. The two sources disagree -- "
                    f"refusing to pick a denominator.")
            return pack
    return pack


def census(catalog_path: str) -> dict:
    cat = json.load(open(catalog_path))
    rows = []
    for s in sorted(cat["specimens"], key=lambda x: -x["bytes"]):
        try:
            r = classify(s["path"])
        except Exception as exc:                       # a header we cannot parse is a FINDING
            r = {"klass": "HEADER-UNREADABLE", "blocked_by": f"{type(exc).__name__}: {exc}"}
        if r["klass"] not in ("NO-SAFETENSORS", "HEADER-UNREADABLE"):
            try:
                r.update(accounting(s["path"], confirm_pack_factor(s["path"])))
            except Exception as exc:
                r["accounting_error"] = f"{type(exc).__name__}: {exc}"
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

        # A bf16 body must account to EXACTLY 16.000 EBPW. This is the control
        # that catches a wrong denominator: Qwen3-30B-A3B lands on 16.000.
        a = accounting(d, 1)
        assert a["complete_ebpw"] == 16.0, a
        assert a["scale_overhead_pct"] == 0.0, a

        # Scales are overhead. Adding one must RAISE ebpw above 16, never leave
        # it at 16 -- which is what counting them as source params would do.
        hdr = {"model.layers.0.mlp.experts.0.down_proj.weight":
               {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
               "model.layers.0.mlp.experts.0.down_proj.weight_scale":
               {"dtype": "F32", "shape": [4], "data_offsets": [32, 48]}}
        blob = json.dumps(hdr).encode()
        with open(os.path.join(d, "m.safetensors"), "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)) + blob + b"\0" * 48)
        a = accounting(d, 1)
        assert a["source_params"] == 16, a
        assert a["complete_ebpw"] == 24.0, a
        assert a["scale_overhead_pct"] == 33.33, a
    print("selfcheck OK")


def _lakecheck(lake: str = "/Volumes/corpdrive/hawking-modellake/specimens") -> None:
    """The twin-pair control, run against the real lake.

    bitnet-b1.58-2B-4T ships twice: once packed, once bf16. They are the SAME
    organism, so their source-parameter counts must be identical to the digit --
    which is the only reason the packed body's 4-values-per-byte factor is
    knowable at all. This control is what caught the census reporting 11.095
    EBPW for a body that actually sits at 3.908, and it will catch the next
    packing scheme the pack-factor logic does not understand.
    """
    packed = os.path.join(lake, "microsoft--bitnet-b1.58-2B-4T@04c3b9ad9361")
    dense = os.path.join(lake, "microsoft--bitnet-b1.58-2B-4T-bf16@276681394656")
    if not (os.path.isdir(packed) and os.path.isdir(dense)):
        raise SystemExit(f"LAKECHECK NOT RUN -- twin pair absent under {lake}. "
                         f"This control did not execute; do not read its silence as a pass.")
    a = accounting(packed, confirm_pack_factor(packed))
    b = accounting(dense, confirm_pack_factor(dense))
    assert a["source_params"] == b["source_params"], (
        f"twin pair disagrees: packed {a['source_params']} vs bf16 {b['source_params']} "
        f"-- the packing factor is wrong")
    assert b["complete_ebpw"] == 16.0, b
    assert a["complete_ebpw"] < 4.0, a
    print(f"lakecheck OK -- twin pair agrees at {a['source_params'] / 1e9:.3f}B params; "
          f"packed {a['complete_ebpw']} EBPW vs bf16 {b['complete_ebpw']}")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    elif "--lakecheck" in sys.argv:
        _lakecheck()
    else:
        print(json.dumps(census(sys.argv[1]), indent=1))
