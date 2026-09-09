#!/usr/bin/env python3.12
"""Specimen anatomy that needs no config.json.

Three ModelLake specimens were left open across the census because their config
could not be parsed: boltz-2 ships PyTorch Lightning .ckpt with no config at
all, moshika ships a bare model.safetensors, and HunyuanVideo's config is nested
under a subdirectory the top-level reader never looked into. In every case the
specimen was recorded as unparseable when what was unparseable was the METADATA.
The tensors were always there.

So this reads the tensors and asks the questions the disposition gate actually
needs -- complete byte anatomy, dtype census, role structure, repeated blocks --
none of which require anyone to have written a config.json in the format
transformers expects.

Complete accounting: every tensor's bytes are counted at its own dtype width,
including the ones that are not weights. A specimen's size is what is on disk,
not the part of it a loader would call parameters.
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
LAKE = Path("/Volumes/corpdrive/hawking-modellake/specimens")
OUT = Path(__file__).resolve().parents[2] / "receipts/future/ANATOMY.json"

#: F8_E8M0 was missing from an earlier table and silently defaulted to 2 bytes,
#: inflating a specimen by 8 GiB. Anything absent here raises rather than guesses.
WIDTH = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
         "F8_E8M0": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
         "torch.float64": 8, "torch.float32": 4, "torch.float16": 2,
         "torch.bfloat16": 2, "torch.int64": 8, "torch.int32": 4,
         "torch.int16": 2, "torch.int8": 1, "torch.uint8": 1, "torch.bool": 1,
         # torch's own fp8 spellings. HunyuanVideo's 12.28 GiB transformer is
         # float8_e4m3fn and the table raised on it rather than defaulting --
         # which is the table working, and is how the reader defect above was
         # confirmed fixed: a walker that was returning zero tensors cannot
         # raise on the dtype of the tensors it is not reading.
         "torch.float8_e4m3fn": 1, "torch.float8_e5m2": 1,
         "torch.float8_e4m3fnuz": 1, "torch.float8_e5m2fnuz": 1}


def width(dtype):
    key = str(dtype)
    if key not in WIDTH:
        raise KeyError(f"unknown dtype {key!r} -- add its width rather than guessing")
    return WIDTH[key]


def _tensors_in(obj, prefix=""):
    """Every tensor anywhere in a loaded container.

    NOT obj.get("state_dict", obj). HunyuanVideo's 12.28 GiB transformer is a
    DeepSpeed mp_rank file whose tensors live under "module", so that lookup
    returned the outer dict, none of whose top-level values are tensors, and the
    walker reported ZERO from a twelve-gigabyte file without raising. A silent
    zero from a container that exists is the same defect as a format gap
    recorded as a negative result -- it just hides better.
    """
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _tensors_in(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _tensors_in(v, f"{prefix}.{i}")


def walk(root):
    """(container, name, shape, dtype) for every tensor.

    Dedup is per CONTAINER, not global. boltz-2 ships boltz2_aff.ckpt and
    boltz2_conf.ckpt -- two different models, 1.92 and 2.13 GiB -- that share a
    name space, and a global dedup silently reported one of them. Two files that
    both exist on disk both count.
    """
    st = sorted(root.rglob("*.safetensors"))
    if st:
        for f in st:
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    sl = h.get_slice(k)
                    yield f.name, k, tuple(sl.get_shape()), sl.get_dtype()
        return
    for f in (sorted(root.rglob("*.ckpt")) + sorted(root.rglob("*.bin"))
              + sorted(root.rglob("*.pt"))):
        try:
            obj = torch.load(f, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(f, map_location="cpu", weights_only=False)
        # DEDUP BY STORAGE, NOT BY NAME. boltz-2's Lightning checkpoint holds
        # an EMA copy under "callbacks" that ALIASES the state_dict tensors, so
        # summing logical tensor bytes double-counts 48% of the file: 4.07 GiB
        # logical against 2.13 GiB of unique storage and 2.13 GiB on disk. The
        # unique figure matches the file to the digit, which is what makes this
        # a measurement rather than a story about why the numbers disagree.
        found = 0
        storages = set()
        for name, t in _tensors_in(obj):
            found += 1
            try:
                sid = t.untyped_storage().data_ptr()
            except Exception:
                sid = None
            if sid is not None and sid in storages:
                continue                      # an alias, already counted
            if sid is not None:
                storages.add(sid)
            yield f.name, name, tuple(t.shape), t.dtype
        if found == 0:
            raise RuntimeError(
                f"{f.name} is {f.stat().st_size/2**30:.2f} GiB and yielded no "
                f"tensors -- that is a reader defect, not a measurement")


def probe(name):
    root = next(iter(sorted(LAKE.glob(f"{name}@*"))), None)
    if root is None:
        return {"specimen": name, "error": "not on disk"}
    total_bytes = 0
    params = 0
    dtypes = Counter()
    roles = defaultdict(list)
    n = 0
    per_container = defaultdict(int)
    for container, tname, shape, dtype in walk(root):
        elems = 1
        for d in shape:
            elems *= d
        b = elems * width(dtype)
        total_bytes += b
        per_container[container] += b
        dtypes[str(dtype)] += b
        if len(shape) >= 2:
            params += elems
        roles[re.sub(r"\.\d+\.", ".<L>.", tname)].append(shape)
        n += 1
    repeated = {r: len(v) for r, v in roles.items() if len(v) >= 8}
    disk = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    _ = disk
    return {
        "specimen": name, "path": root.name, "n_tensors": n,
        "tensor_bytes": total_bytes, "tensor_gib": round(total_bytes / 2**30, 2),
        "on_disk_bytes": disk, "on_disk_gib": round(disk / 2**30, 2),
        "matrix_params_b": round(params / 1e9, 4),
        "dtype_bytes": dict(dtypes.most_common()),
        "per_container_gib": {k: round(v / 2**30, 2) for k, v in
                              sorted(per_container.items(), key=lambda kv: -kv[1])},
        "tensor_bytes_vs_disk": round(total_bytes / disk, 3) if disk else None,
        "coverage_warning": (None if not disk or total_bytes / disk > 0.7 else
            f"tensor bytes are only {total_bytes/disk:.0%} of what is on disk -- "
            f"treat this as PARTIAL until the gap is explained"),
        "distinct_roles": len(roles),
        "repeated_blocks": dict(sorted(repeated.items(), key=lambda kv: -kv[1])[:12]),
        "n_repeated_roles": len(repeated),
        "complete_ebpw_if_bf16": round(total_bytes * 8 / params, 3) if params else None,
    }


if __name__ == "__main__":
    out = [probe(a) for a in sys.argv[1:]]
    for r in out:
        if "error" in r:
            print(f"{r['specimen']}: {r['error']}"); continue
        print(f"{r['specimen']}")
        print(f"   {r['n_tensors']} tensors, {r['tensor_gib']} GiB of tensor, "
              f"{r['on_disk_gib']} GiB on disk, {r['matrix_params_b']}B matrix params")
        print(f"   dtypes: {list(r['dtype_bytes'])}")
        print(f"   {r['distinct_roles']} distinct roles, {r['n_repeated_roles']} repeated >=8x")
        for k, v in list(r["repeated_blocks"].items())[:4]:
            print(f"     {v:3d}x  {k[:70]}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
