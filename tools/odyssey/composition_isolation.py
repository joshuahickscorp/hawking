#!/usr/bin/env python3
"""Which half of the 2.60 body broke it?

The 2.5970-EBPW heterogeneous body differs from the coherent 3.1393-EBPW body in exactly
two places, and both were validated per organ before being composed:

  MLP        affine-LS 2.5 bpw (2-bit codes + f16 scale + f16 bias)  ->  q2f 2.25 bpw
             (2-bit codes + f16 scale, NO per-group bias)
  attn/state q4 (HQ30UQ4)  ->  q3 g128/g64 (HGRAVU01)
  embed/head

Variant A holds the attention/state/embed/head side at the q4 incumbent and moves ONLY
the MLP. It needs no new packer: the builder already hardlinks any tensor whose organ
role is "leftover" straight from the q4 incumbent, which is the sealed codec.

If A is coherent, the 0.25 bpw of per-group MLP bias was not the problem and the q4->q3
drop on everything else is. If A is broken, that bias is load-bearing.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/headless"))


# Forcing an organ's role to "leftover" does NOT work and should not: the builder guards
# it with `is leftover but incumbent kind is q4; refusing hidden GEMV`, because leftover
# means an f32 passthrough and hardlinking a quantized GEMV in under that name would hide
# it from the accounting. The genome is overridden instead, which keeps the organ a real
# GEMV and only changes its bit rate.
Q4_OVERRIDE = {
    "bits": 4, "group": 64, "gemv_storage_bpw": 4.25,
    "codec": "ws_rtn_q4_g64", "family": "grouped_absmax", "container": "HGRAVU01",
}
Q4_KERNEL = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128"


# The MLP codec the SEALED body uses: four-level affine at group 64 WITH a per-group
# bias, 2.5 bpw. The builder's HGRAVF01 path is hardcoded to the no-bias q2f packer, so
# routing to it needs a swap of the pack function, not just a genome field.
MLP_AFFINE_BIAS = {
    "codec": "affine2_g64_ls", "family": "fourlevel_fitted_affine",
    "gemv_storage_bpw": 2.5, "group": 64, "bits": 2,
    "kernel": "affine2_group64_matvec_geo_tpr64_tg128",
    "container": "HGRAVF01",
}


def build_variant(out_root, hold_at_q4, mlp_with_bias=False):
    import copy
    import whole_model_native as w

    original = copy.deepcopy(w.GENOME)
    original_pack = w.pack_hgrafv01_q2f
    try:
        if mlp_with_bias:
            import affine2_g64_lsfit as af
            w.GENOME["mlp"].update(MLP_AFFINE_BIAS)

            def _pack_with_bias(weights, group):
                # compile_mix expects (payload, probe); the affine packer returns payload
                return af.pack_hgrafv01(weights, group, fit="ls"), {"fit": "ls",
                                                                    "bias": True}
            w.pack_hgrafv01_q2f = _pack_with_bias
        for role in hold_at_q4:
            if role not in w.GENOME:
                raise SystemExit(f"unknown organ role {role!r}; have {sorted(w.GENOME)}")
            w.GENOME[role].update(Q4_OVERRIDE)
            if role != "embedding":          # the embed path is a lookup, not a GEMV
                w.GENOME[role]["kernel"] = Q4_KERNEL
        t0 = time.time()
        res = w.compile_mix(out_root=Path(out_root))
        res["wall_s"] = round(time.time() - t0, 1)
        res["genome_override"] = {r: {k: w.GENOME[r][k] for k in
                                      ("codec", "bits", "group", "gemv_storage_bpw", "kernel")}
                                  for r in hold_at_q4}
        return res
    finally:
        w.GENOME.clear()
        w.GENOME.update(original)
        w.pack_hgrafv01_q2f = original_pack


PROBES = [
    ("math", "What is 17 + 4? Answer with the number only.", "21"),
    ("code", "Write a Python function that reverses a string. Reply with only a python "
             "code block.", "```python"),
    ("fact", "What is the capital of France? Answer with the city name only.", "Paris"),
]


def probe(root, tokenizer_dir, binary, max_new=220):
    from transformers import AutoTokenizer
    import tempfile, re
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    out = []
    for pid, prompt, expect in PROBES:
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        n = len(tok(text)["input_ids"])
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        f.close()
        cmd = [str(binary), "--artifact-root", str(root),
               "--tokenizer", str(Path(tokenizer_dir) / "tokenizer.json"),
               "--prompt", text, "--max-new-tokens", str(max_new),
               "--max-seq-len", str(n + max_new + 16), "--out", f.name, "--raw-prompt"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        body = json.loads(Path(f.name).read_text()) if Path(f.name).stat().st_size else {}
        Path(f.name).unlink(missing_ok=True)
        ids = body.get("new_token_ids") or []
        txt = body.get("generated_text") or ""
        run, cur, best = 1, (ids[0] if ids else None), (1 if ids else 0)
        for t in ids[1:]:
            run = run + 1 if t == cur else 1
            cur = t
            best = max(best, run)
        out.append({
            "probe": pid, "expect": expect, "exit_code": p.returncode,
            "n_new_tokens": len(ids),
            "unique_ratio": round(len(set(ids)) / len(ids), 3) if ids else 0.0,
            "longest_identical_run": best,
            "closed_think": "</think>" in txt,
            "contains_expected": expect in txt,
            "newline_only": bool(txt) and txt.strip() == "",
            "text_head": txt[:200],
        })
    passed = sum(1 for r in out if r["contains_expected"])
    return {"probes": out, "n_passed": passed, "n_probes": len(out),
            "coherent": passed == len(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--binary", required=True)
    # nargs="*" so a variant can hold NOTHING at q4 -- variant B moves only the MLP and
    # leaves every other organ at the genome default of q3.
    ap.add_argument("--hold-at-q4", nargs="*",
                    default=["deltanet", "attention_gqa", "embedding", "output"])
    ap.add_argument("--emit", required=True)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--mlp-with-bias", action="store_true",
                    help="route the MLP through the affine-with-bias packer (2.5 bpw), "
                         "which is the codec the sealed body uses")
    a = ap.parse_args()

    import whole_model_native as w
    built = None
    if not a.probe_only:
        built = build_variant(a.out_root, a.hold_at_q4, mlp_with_bias=a.mlp_with_bias)
    art = Path(built["artifact_root"]) if built else Path(a.out_root) / w.MIX_ID
    res = probe(art, a.tokenizer_dir, a.binary)
    out = {
        "schema": "hawking.headless.composition_isolation.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/composition_isolation.py",
        "obligation": "G004 follow-on — which representation change broke whole-model capability",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "variant": {"mlp": ("affine2_g64_ls 2.5 bpw (WITH per-group bias)"
                            if a.mlp_with_bias else "q2f_g64 2.25 bpw (no per-group bias)"),
                    "held_at_q4_incumbent": a.hold_at_q4,
                    "mlp_with_bias": a.mlp_with_bias},
        "artifact_root": str(art),
        "build": None if not built else {k: built.get(k) for k in
                                         ("complete_ebpw", "n_tensors", "n_hardlink",
                                          "wall_s", "storage_bpw")},
        "capability_probe": res,
        "pass": res["coherent"],
    }
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(json.dumps({"complete_ebpw": (built or {}).get("complete_ebpw"),
                      "n_hardlink": (built or {}).get("n_hardlink"),
                      "coherent": res["coherent"],
                      "n_passed": f"{res['n_passed']}/{res['n_probes']}"}, indent=1))
    for r in res["probes"]:
        print(f"  {r['probe']:5} tok={r['n_new_tokens']:4} uniq={r['unique_ratio']:5} "
              f"closed_think={str(r['closed_think']):5} got_expected={r['contains_expected']}")
        print(f"        {r['text_head'][:120]!r}")
    return 0 if res["coherent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
