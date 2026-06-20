#!/usr/bin/env python3
"""isobpw_table.py — join STRAND + GGUF PPLs and bpw into the Pareto table.

Reads research/isobpw/ppl/*.json (STRAND + GGUF, same harness) and
research/isobpw/gguf-bpw.json + STRAND quant sidecars for bpw, then prints a
markdown table sorted by bpw with the per-tier winner.
"""
import glob
import json
import os

RES = "research/isobpw"
PPL = f"{RES}/ppl"
RECON = f"{RES}/strand-recon"


def load_ppl(tag):
    for f in glob.glob(f"{PPL}/ppl_*{tag}*.json") + glob.glob(f"{PPL}/{tag}*.json"):
        try:
            d = json.load(open(f))
            if "ppl" in d:
                return d["ppl"]
        except Exception:
            pass
    return None


def strand_bpw(tag):
    side = f"{RECON}/{tag}/model.safetensors.json"
    if os.path.exists(side):
        try:
            return json.load(open(side))["aggregate"]["effective_bpw"]
        except Exception:
            pass
    # fall back to any sidecar in the recon dir
    for f in glob.glob(f"{RECON}/{tag}/*.json"):
        try:
            return json.load(open(f))["aggregate"]["effective_bpw"]
        except Exception:
            pass
    return None


def gguf_bpw():
    p = f"{RES}/gguf-bpw.json"
    out = {}
    if os.path.exists(p):
        for r in json.load(open(p)):
            name = r["file"].replace("qwen05b-", "").replace(".gguf", "")
            out[name] = r
    return out


def main():
    rows = []  # (bpw, label, ppl, fmt, note)
    gb = gguf_bpw()

    # bf16 anchor
    bf = load_ppl("bf16_anchor")
    if bf is not None:
        rows.append((16.0, "bf16 (anchor)", bf, "ref", "ceiling, not a target"))

    # GGUF
    for q in ("Q2_K", "Q3_K_M", "IQ3_S", "Q4_K_M"):
        ppl = load_ppl(f"gguf_{q}")
        b = gb.get(q, {})
        rows.append((b.get("proj_bpw"), f"GGUF {q}", ppl, "gguf",
                     f"file_bpw={b.get('file_bpw'):.3f}" if b.get("file_bpw") else ""))

    # STRAND
    for tag, label in (("q2_l12_out1", "STRAND q2 l12 out1"),
                       ("q3_l12_out1", "STRAND q3 l12 out1"),
                       ("mp_light", "STRAND mp_light (attn4/ffn3)")):
        rows.append((strand_bpw(tag), label, load_ppl(f"strand_{tag}"), "strand", ""))

    rows = [r for r in rows if r[0] is not None]
    rows.sort(key=lambda r: r[0])

    print(f"| bpw (proj) | config | PPL | format | note |")
    print(f"|---|---|---|---|---|")
    for bpw, label, ppl, fmt, note in rows:
        ps = f"{ppl:.4f}" if ppl is not None else "—"
        print(f"| {bpw:.4f} | {label} | {ps} | {fmt} | {note} |")


if __name__ == "__main__":
    main()
