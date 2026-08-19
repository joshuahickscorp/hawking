#!/usr/bin/env python3
"""Odyssey architecture census — deterministic, no weight load, stdlib only.

Bible §13/§24-3/§90: tools MEASURE, models interpret. Reads config.json + every
safetensors header (8-byte length prefix + JSON) to get exact tensor shapes and
byte sizes without materializing a single weight. Emits total params, stored
bytes, per-organ breakdown, MoE topology, and ACTIVE params/token (the MoE cost
metric that stored size hides).

    python3 tools/odyssey_census.py <model_dir_or_hf_snapshot> [--out packet.json]

`--self-check` runs an in-process assertion demo (no model needed).
"""
from __future__ import annotations
import json, struct, sys, glob, os, re
from pathlib import Path

# bytes per element by safetensors dtype tag
_DT = {"BOOL":1,"U8":1,"I8":1,"F8_E4M3":1,"F8_E5M2":1,"I16":2,"U16":2,"F16":2,
       "BF16":2,"I32":4,"U32":4,"F32":4,"I64":8,"U64":8,"F64":8}

def _prod(xs):
    n = 1
    for x in xs: n *= x
    return n

def read_safetensors_header(path: str) -> dict:
    """Return the tensor->meta dict from a safetensors file, reading only the header."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr

def _shard_set(d: Path) -> list:
    """Authoritative safetensors set — avoid double-counting repos that ship BOTH a
    consolidated single file AND sharded weights. Prefer the index weight_map."""
    idx = sorted(glob.glob(str(d / "*.safetensors.index.json")))
    if idx:
        wm = json.loads(Path(idx[0]).read_text()).get("weight_map", {})
        files = sorted({str(d / f) for f in wm.values()})
        if files:
            return files
    allst = sorted(glob.glob(str(d / "*.safetensors")))
    sharded = [s for s in allst if re.search(r"-\d{5}-of-\d{5}\.safetensors$", s)]
    if sharded:  # drop consolidated/single-file duplicates when shards exist
        return sharded
    return allst

def _flat_cfg(cfg: dict) -> dict:
    """Multimodal configs nest the LM under text_config/llm_config/language_config —
    surface those keys for the arch summary without losing the top level."""
    out = dict(cfg)
    for k in ("text_config","llm_config","language_config"):
        sub = cfg.get(k)
        if isinstance(sub, dict):
            for kk, vv in sub.items():
                out.setdefault(kk, vv)
    return out

def census(model_dir: str) -> dict:
    d = Path(model_dir)
    cfg = _flat_cfg(json.loads((d / "config.json").read_text()))
    shards = _shard_set(d)
    if not shards:
        raise SystemExit(f"no safetensors in {model_dir}")

    tensors = {}          # name -> (shape, dtype, params, bytes)
    for s in shards:
        for name, meta in read_safetensors_header(s).items():
            shp = meta["shape"]; dt = meta["dtype"]
            p = _prod(shp) if shp else 0
            b = _DT.get(dt, 2) * p
            tensors[name] = (shp, dt, p, b)

    total_params = sum(t[2] for t in tensors.values())
    total_bytes  = sum(t[3] for t in tensors.values())

    # organ classification by tensor-name substring (transformer-generic)
    organs = {"embed":0,"attn":0,"router":0,"expert":0,"shared_expert":0,
              "mlp_dense":0,"norm":0,"lm_head":0,"other":0}
    obytes = dict.fromkeys(organs, 0)
    def bump(k, p, b): organs[k]+=p; obytes[k]+=b
    for name,(shp,dt,p,b) in tensors.items():
        n = name.lower()
        if "embed" in n and "lm_head" not in n:               bump("embed",p,b)
        elif "lm_head" in n:                                   bump("lm_head",p,b)
        elif "gate" in n and ("router" in n or ("mlp.gate" in n and "proj" not in n) or n.endswith("gate.weight")): bump("router",p,b)
        elif "shared_expert" in n or "shared_mlp" in n:        bump("shared_expert",p,b)
        elif re.search(r"experts?\.\d+", n) or ".experts." in n: bump("expert",p,b)
        elif any(k in n for k in ("q_proj","k_proj","v_proj","o_proj","qkv","attn","q_norm","k_norm")): bump("attn",p,b)
        elif "norm" in n:                                      bump("norm",p,b)
        elif any(k in n for k in ("up_proj","down_proj","gate_proj","mlp")): bump("mlp_dense",p,b)
        else:                                                  bump("other",p,b)

    # MoE topology from config (generic across qwen3_moe / mixtral / glm / etc.)
    ne  = cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts")
    net = cfg.get("num_experts_per_tok") or cfg.get("moe_topk") or cfg.get("n_experts_per_tok")
    is_moe = bool(ne)
    active_params = None
    if is_moe and organs["expert"] and nInt(ne):
        # active/token = everything non-expert + (topk/ne) * expert params
        non_expert = total_params - organs["expert"]
        active_params = non_expert + int(organs["expert"] * (nInt(net) / nInt(ne)))

    return {
        "model_dir": str(d),
        "arch": (cfg.get("architectures") or ["?"])[0],
        "model_type": cfg.get("model_type"),
        "config": {k: cfg.get(k) for k in (
            "num_hidden_layers","hidden_size","intermediate_size","moe_intermediate_size",
            "num_attention_heads","num_key_value_heads","head_dim","num_experts",
            "num_experts_per_tok","n_routed_experts","n_shared_experts","num_local_experts",
            "vocab_size","max_position_embeddings","rope_theta","torch_dtype","tie_word_embeddings",
            "decoder_sparse_step","mlp_only_layers") if k in cfg},
        "tensor_count": len(tensors),
        "shard_count": len(shards),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "stored_bpw": round(total_bytes * 8 / total_params, 4) if total_params else None,
        "is_moe": is_moe,
        "active_params_per_token": active_params,
        "active_bytes_per_token": int(active_params * (total_bytes/total_params)) if active_params and total_params else None,
        "organs_params": organs,
        "organs_bytes": obytes,
    }

def nInt(x):
    try: return int(x)
    except (TypeError, ValueError): return 0

def _self_check():
    # synthetic 2-expert-per-4 MoE: non-expert 100, experts 400 -> active 100 + 200 = 300
    fake = {"total_params":500,"expert":400}
    non = fake["total_params"] - fake["expert"]
    active = non + int(fake["expert"] * (2/4))
    assert active == 300, active
    # dtype table sanity
    assert _DT["BF16"]==2 and _DT["F32"]==4 and _DT["F8_E4M3"]==1
    assert _prod([2,3,4])==24
    print("self-check OK")

if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check(); sys.exit(0)
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out")+1]
    r = census(sys.argv[1])
    js = json.dumps(r, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(js)
    # compact human summary
    c = r["config"]
    print(f"{r['arch']}  ({r['model_type']})")
    print(f"  layers={c.get('num_hidden_layers')} hidden={c.get('hidden_size')} "
          f"heads={c.get('num_attention_heads')}/{c.get('num_key_value_heads')}kv ctx={c.get('max_position_embeddings')}")
    if r["is_moe"]:
        print(f"  MoE: experts={c.get('num_experts') or c.get('n_routed_experts')} "
              f"top-{c.get('num_experts_per_tok') or c.get('moe_topk')} moe_ffn={c.get('moe_intermediate_size')}")
    print(f"  total_params={r['total_params']/1e9:.3f}B  stored_bytes={r['total_bytes']/1e9:.2f}GB  "
          f"stored_bpw={r['stored_bpw']}")
    if r["active_params_per_token"]:
        print(f"  ACTIVE_params/token={r['active_params_per_token']/1e9:.3f}B "
              f"({100*r['active_params_per_token']/r['total_params']:.1f}% of total)")
    print(f"  organs(GB): " + "  ".join(f"{k}={v/1e9:.2f}" for k,v in r["organs_bytes"].items() if v))
    if out: print(f"  -> {out}")
