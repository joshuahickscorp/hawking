"""Load a DeepSeek-V3-shaped MoE text tower through STOCK transformers.

Kimi-VL-A3B ships `modeling_kimi_vl.py` written against transformers 4.50.3.
This host runs 5.5.4, and remote modeling code does not survive a major version
step -- the cache and attention interfaces both moved underneath it. The usual
answers are to pin an old transformers or to fork the remote code, and both put
a second, drifting implementation of the same architecture into the world.

There is a third answer, and it is cheaper than either: Kimi's `text_config` is
the DeepSeek-V3 schema verbatim -- kv_lora_rank, qk_nope_head_dim,
qk_rope_head_dim, v_head_dim, topk_method=noaux_tc, scoring_func=sigmoid,
routed_scaling_factor, first_k_dense_replace -- and its tensor names are
DeepSeek-V3's under a `language_model.` prefix. transformers 5.5.4 has a NATIVE
deepseek_v3. So the whole port is a name map plus one shape change.

The one shape change: native DeepseekV3NaiveMoe stores experts FUSED as 3D
parameters, gate_up_proj [E, 2*inter, hidden] and down_proj [E, hidden, inter],
where the checkpoint stores each expert separately. The fused order is fixed by
the forward, which does `linear(x, gate_up_proj[e]).chunk(2, dim=-1)` -- gate
first, up second. Getting that order backwards produces a model that loads
cleanly and generates garbage, so it is asserted rather than assumed.

Fusing is not a tax, it is the point. The native path batches expert weights as
3D tensors instead of looping 64 nn.Linear modules, and it exposes hidden states
and router scores through the standard interfaces, which is what any refusal,
routing or activation study needs.

WHY THIS IS A CAPABILITY AND NOT A SCRIPT: DeepSeek-V2/V3 is a whole family --
DeepSeek itself, Kimi, and a growing list of MLA+MoE bodies. Every one of them
that ships version-skewed remote code costs the same afternoon. This costs a
name map.

    python3 tools/future/dsv3_native_loader.py --selftest        # no weights read
    python3 tools/future/dsv3_native_loader.py --spec <dir> --check
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PREFIX = "language_model."


def free_gib() -> float:
    """Free + inactive + speculative, using the page size vm_stat REPORTS.

    Hardcoding 4096 here undercounts by 4x on this machine, which reports 16384 --
    a past guard read 16.9 GiB where 72.7 were available and latched itself off
    forever. The header is the authority, never a constant."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps = 4096
    vals = {}
    for line in out.splitlines():
        if "page size of" in line:
            ps = int(line.split("page size of")[1].split()[0])
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                vals[k.strip()] = int(v)
    got = sum(vals.get(k, 0) for k in
              ("Pages free", "Pages inactive", "Pages speculative"))
    return got * ps / 2 ** 30


def swapouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return 0
# Tensors the checkpoint carries that the native model does not want. inv_freq is
# a derived rotary buffer that the native rotary embedding recomputes; carrying a
# stale one across a version step is a silent correctness bug.
DROP = re.compile(r"\.rotary_emb\.inv_freq$")


def text_config(spec: Path):
    from transformers import DeepseekV3Config
    raw = json.loads((spec / "config.json").read_text())
    if "text_config" not in raw:
        raise ValueError(f"{spec}/config.json has no text_config; not a VL-wrapped tower")
    return DeepseekV3Config(**raw["text_config"])


def build_skeleton(cfg):
    """The model on the meta device: shapes and names, no bytes."""
    import torch
    from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM
    with torch.device("meta"):
        return DeepseekV3ForCausalLM(cfg)


def checkpoint_names(spec: Path) -> set[str]:
    wm = json.loads((spec / "model.safetensors.index.json").read_text())["weight_map"]
    return {k[len(PREFIX):] for k in wm if k.startswith(PREFIX) and not DROP.search(k)}


def plan(spec: Path):
    """What the native model wants, what the checkpoint has, and how they meet.

    Returned so a caller can REFUSE a specimen instead of discovering the
    mismatch halfway through a 33 GB load."""
    cfg = text_config(spec)
    want = set(build_skeleton(cfg).state_dict().keys())
    have = checkpoint_names(spec)
    direct = want & have
    fused = {k for k in want - have
             if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj")}
    unexplained = (want - have) - fused
    # Every fused target must be reconstructible from per-expert tensors present.
    for k in fused:
        layer = k.split(".layers.")[1].split(".")[0]
        src = (["gate_proj", "up_proj"] if k.endswith("gate_up_proj") else ["down_proj"])
        for e in range(cfg.n_routed_experts):
            for s in src:
                n = f"model.layers.{layer}.mlp.experts.{e}.{s}.weight"
                if n not in have:
                    unexplained.add(f"{k} <- missing {n}")
    return {"config": cfg, "want": want, "have": have, "direct": direct,
            "fused": fused, "unexplained": unexplained,
            "unused": {k for k in have - want
                       if ".mlp.experts." not in k}}


def load(spec: Path, *, device: str = "cpu", dtype: str = "bfloat16", verbose: bool = True,
         min_free_gib: float = 12.0):
    """Materialise the tower. Streams shard by shard so peak memory is the model
    plus one shard, not the model plus a whole second copy of itself."""
    import torch
    from safetensors import safe_open

    p = plan(spec)
    if p["unexplained"]:
        raise ValueError(f"cannot map {len(p['unexplained'])} tensors: "
                         f"{sorted(p['unexplained'])[:5]}")
    cfg = p["config"]
    torch_dtype = getattr(torch, dtype)

    # MLA gives attention a value head narrower than its query/key head --
    # v_head_dim 128 against qk_head_dim 192 on this family. Torch's SDPA on MPS
    # does not honour that: it returns output shaped by the QUERY head, so
    # o_proj is handed num_heads*192 where it expects num_heads*v_head_dim.
    # Measured on a one-layer model with this exact geometry:
    #
    #     cpu/fp32/sdpa   OK        mps/fp32/sdpa   FAILS (5x3072 vs 2048x2048)
    #     cpu/fp32/eager  OK        mps/bf16/sdpa   FAILS
    #                               mps/*/eager     OK
    #
    # Here it raises, because o_proj's shape happens to catch it. On a body whose
    # v_head_dim equalled qk_head_dim the same bug would return WRONG NUMBERS in
    # silence, which is why this is pinned rather than worked around at the call
    # site. Eager is correct on MPS and costs attention speed, not correctness.
    if device.startswith("mps") and cfg.v_head_dim != (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim):
        cfg._attn_implementation = "eager"
        if verbose:
            print(f"  MLA on MPS: forcing eager attention "
                  f"(v_head_dim {cfg.v_head_dim} != qk_head_dim "
                  f"{cfg.qk_nope_head_dim + cfg.qk_rope_head_dim}; MPS sdpa returns the query head dim)")

    # Second Apple gap, same shape as the first. The default experts kernel is
    # `grouped_mm`, which counts tokens per expert with torch.histc -- and MPS
    # has no integer histogram kernel ("histogram_mps" not implemented for
    # 'Int'). `batched_mm` computes the same thing and runs. Measured on a
    # two-layer model with this geometry, bf16 on MPS:
    #
    #     grouped_mm   RuntimeError: "histogram_mps" not implemented for 'Int'
    #     batched_mm   OK, bit-identical to itself across runs
    #     naive loop   OK, max|diff| 0.0156 vs batched_mm (bf16 accumulation order)
    #
    # batched_mm is chosen over the naive loop because it is both correct here
    # and the batched path -- the naive one dispatches 64 separate linears per
    # MoE layer, which is the cost this family exists to avoid.
    if device.startswith("mps") and getattr(cfg, "_experts_implementation", None) == "grouped_mm":
        cfg._experts_implementation = "batched_mm"
        if verbose:
            print("  MoE on MPS: grouped_mm -> batched_mm (MPS has no integer histc kernel)")

    # ORDER MATTERS AND COSTS 33 GB. The skeleton is built in torch's default
    # dtype (fp32), so `to_empty(device)` then `.to(bfloat16)` allocates the fp32
    # storage FIRST -- 66 GB for this body -- and only then the 33 GB it wanted.
    # Measured: free RAM fell 57.8 -> 13.3 GiB on a machine also holding a
    # resident, which is how a load starts swapping against another process.
    # Meta tensors have no storage, so converting the dtype while still on meta
    # is free and to_empty then allocates bf16 directly.
    model = build_skeleton(cfg).to(torch_dtype)
    model.to_empty(device=device)

    # to_empty() allocates UNINITIALISED storage for buffers as well as
    # parameters, and rotary inv_freq is derived rather than stored -- no
    # checkpoint carries it. Left as-is the model would apply rope with garbage
    # frequencies and still produce fluent-looking logits, which is the worst
    # shape a bug can have. Rebuilt from the config on the target device.
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding
    model.model.rotary_emb = DeepseekV3RotaryEmbedding(config=cfg, device=device)

    params = dict(model.named_parameters())
    params.update({k: v for k, v in model.named_buffers()
                   if not k.startswith("model.rotary_emb.")})
    wm = json.loads((spec / "model.safetensors.index.json").read_text())["weight_map"]

    # Per-layer expert staging. Held only until a layer's 64 experts are complete.
    stage: dict[str, dict[int, "torch.Tensor"]] = {}
    filled = set()

    def flush(layer: str):
        inter, hid = cfg.moe_intermediate_size, cfg.hidden_size
        gu = params[f"model.layers.{layer}.mlp.experts.gate_up_proj"]
        dn = params[f"model.layers.{layer}.mlp.experts.down_proj"]
        for e in range(cfg.n_routed_experts):
            g = stage[f"{layer}:gate_proj"][e]
            u = stage[f"{layer}:up_proj"][e]
            d = stage[f"{layer}:down_proj"][e]
            # gate FIRST: the forward does chunk(2, dim=-1) on the linear output.
            gu.data[e, :inter] = g.to(torch_dtype)
            gu.data[e, inter:] = u.to(torch_dtype)
            dn.data[e] = d.to(torch_dtype)
        assert gu.shape == (cfg.n_routed_experts, 2 * inter, hid), gu.shape
        assert dn.shape == (cfg.n_routed_experts, hid, inter), dn.shape
        for s in ("gate_proj", "up_proj", "down_proj"):
            stage.pop(f"{layer}:{s}", None)
        filled.add(f"model.layers.{layer}.mlp.experts.gate_up_proj")
        filled.add(f"model.layers.{layer}.mlp.experts.down_proj")

    sw0 = swapouts()
    if verbose:
        print(f"  before load: free {free_gib():.1f} GiB", flush=True)
    shards = sorted({v for k, v in wm.items() if k.startswith(PREFIX)})
    for si, fname in enumerate(shards):
        with safe_open(str(spec / fname), framework="pt") as h:
            for full in h.keys():
                if not full.startswith(PREFIX) or DROP.search(full):
                    continue
                name = full[len(PREFIX):]
                m = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight$", name)
                if m:
                    layer, e, kind = m.group(1), int(m.group(2)), m.group(3)
                    stage.setdefault(f"{layer}:{kind}", {})[e] = h.get_tensor(full)
                    if all(len(stage.get(f"{layer}:{s}", {})) == cfg.n_routed_experts
                           for s in ("gate_proj", "up_proj", "down_proj")):
                        flush(layer)
                elif name in params:
                    params[name].data.copy_(h.get_tensor(full).to(torch_dtype))
                    filled.add(name)
        # Per-shard memory, because a 30 GiB load on a 96 GiB machine drove this
        # host into swap thrash once -- 339,647 pages out in 20s, forwards
        # ~10,000x slower than the 0.14s they should cost, and the OTHER
        # resident evicted. A load that cannot say where its memory went cannot
        # be debugged, so it says.
        import gc
        gc.collect()
        if device.startswith("mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        fg = free_gib()
        if verbose:
            print(f"  shard {si+1}/{len(shards)}  {fname}   free {fg:.1f} GiB"
                  f"   swapouts +{swapouts()-sw0}", flush=True)
        # A load must not be allowed to evict another resident. It did once:
        # forwards went ~10,000x slower than the 0.14s they should cost and the
        # other front's weights were paged out. Aborting here loses a load; not
        # aborting loses the machine, and someone else's campaign with it.
        if fg < min_free_gib:
            raise MemoryError(
                f"free memory {fg:.1f} GiB fell below the {min_free_gib:.1f} GiB "
                f"floor after shard {si+1}/{len(shards)}. Refusing to continue: "
                "this is the condition that drove the host into swap thrash. "
                "Quiesce other residents, lower the floor deliberately, or load "
                "a smaller representation.")

    missing = set(params) - filled
    if missing:
        raise ValueError(f"{len(missing)} parameters never received weights: "
                         f"{sorted(missing)[:5]}")
    model.eval()
    return model, cfg


def _selftest() -> int:
    """Shape and order checks that need no checkpoint. The order check is the one
    that matters: a swapped gate/up loads clean and generates garbage."""
    import torch
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3NaiveMoe

    cfg = DeepseekV3Config(hidden_size=8, moe_intermediate_size=4, n_routed_experts=3,
                           num_experts_per_tok=1, num_hidden_layers=1,
                           intermediate_size=16, num_attention_heads=2,
                           num_key_value_heads=2, kv_lora_rank=4, q_lora_rank=None,
                           qk_nope_head_dim=4, qk_rope_head_dim=2, v_head_dim=4,
                           vocab_size=32, n_shared_experts=1, first_k_dense_replace=0)
    moe = DeepseekV3NaiveMoe(cfg)
    E, inter, hid = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
    assert moe.gate_up_proj.shape == (E, 2 * inter, hid), moe.gate_up_proj.shape
    assert moe.down_proj.shape == (E, hid, inter), moe.down_proj.shape

    # Order is checked against an INDEPENDENT reference, not against chunk().
    # Zeroing one half proves nothing here: silu(0)*up and silu(gate)*0 are both
    # zero, so either order gives the same answer. Two distinct random matrices
    # through a nonlinear gate do not.
    torch.manual_seed(0)
    g = torch.randn(inter, hid)
    u = torch.randn(inter, hid)
    d = torch.randn(hid, inter)
    x = torch.randn(2, hid)
    idx = torch.zeros(2, 1, dtype=torch.long)
    w = torch.ones(2, 1)
    ref = (torch.nn.functional.silu(x @ g.t()) * (x @ u.t())) @ d.t()

    def run(first, second):
        with torch.no_grad():
            moe.gate_up_proj[0, :inter] = first
            moe.gate_up_proj[0, inter:] = second
            moe.down_proj[0] = d
            return moe(x, idx, w)

    y_gate_first = run(g, u)
    y_up_first = run(u, g)
    assert torch.allclose(y_gate_first, ref, atol=1e-5), (
        "placing GATE in the first half did not reproduce silu(xG)*(xU)D -- "
        "the loader's fusion order is wrong")
    assert not torch.allclose(y_up_first, ref, atol=1e-5), (
        "the two orders gave the SAME answer, so this specimen cannot tell them "
        "apart and the check above is vacuous")
    print("selftest OK: fused layout is [E, 2*inter, hidden], GATE first, "
          "verified against an independent silu(xG)*(xU)D reference")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--check", action="store_true", help="map the names, read no weights")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.spec:
        ap.error("--spec is required unless --selftest")
    p = plan(a.spec)
    print(f"native model wants   {len(p['want'])} tensors")
    print(f"checkpoint offers    {len(p['have'])} (prefix '{PREFIX}', inv_freq dropped)")
    print(f"  direct name match  {len(p['direct'])}")
    print(f"  fused from experts {len(p['fused'])}")
    print(f"  UNEXPLAINED        {len(p['unexplained'])}")
    for k in sorted(p["unexplained"])[:10]:
        print(f"      {k}")
    print(f"  checkpoint tensors the native model ignores: {len(p['unused'])}")
    return 1 if p["unexplained"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
