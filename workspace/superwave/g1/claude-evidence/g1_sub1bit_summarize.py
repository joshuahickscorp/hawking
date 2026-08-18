#!/usr/bin/env python3
"""Summarize /tmp/g1_sub1bit_regions.json into ranking + budget tables."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

N = 26_895_998_464
G0_BPW = 4.252735126866492
G0_BYTES = 14_297_694_680
Q4_BODY = 4.25
SMALL_ELEMS = 2_645_504
SMALL_BYTES = 10_584_840


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def is_region_row(r: dict) -> bool:
    return not r.get("spot") and r["family"] != "incumbent"


def region_key(r: dict) -> tuple:
    return (r["layer"], r["class"], r.get("mixer", ""))


def main() -> None:
    d = load("/tmp/g1_sub1bit_regions.json")
    rows = d["rows"]
    amp = d["amp"]
    print(f"n_rows={d['n_rows']} wall={d['wall_s']:.1f}s rss={d['rss_max_gb']:.3f}GiB")
    print(f"capture={d['capture_sha256_self']}")

    # index q4 baseline per region
    q4 = {}
    for r in rows:
        if r.get("spot"):
            continue
        if r["codec"] == "q4_g64":
            q4[region_key(r)] = r

    cheap = [r for r in rows if is_region_row(r)]
    print(f"cheap_rows={len(cheap)} q4_regions={len(q4)}")

    # best codec per region by bits/final_err
    best = {}
    for r in cheap:
        k = region_key(r)
        if k not in best or r["bits_per_final_err"] > best[k]["bits_per_final_err"]:
            best[k] = r
    ranked = sorted(best.values(), key=lambda r: -r["bits_per_final_err"])
    print(f"regions={len(ranked)}")

    # class x depth bins
    def bin_of(L):
        if L < 0:
            return "embed"
        if L == 64:
            return "lm_head"
        if L <= 7:
            return "L00-07"
        if L <= 23:
            return "L08-23"
        if L <= 47:
            return "L24-47"
        return "L48-63"

    print("\n=== TOP 25 regions by bits_saved / final_stream_rel ===")
    print(f"{'rank':>4} {'L':>3} {'cls':8} {'codec':18} {'bpw':7} {'cos':7} {'relY':7} {'final':8} {'vsQ4':7} {'val':10} {'Gbits':8}")
    for i, r in enumerate(ranked[:25], 1):
        q = q4.get(region_key(r))
        vs = r["final_stream_rel"] / q["final_stream_rel"] if q and q["final_stream_rel"] > 0 else float("nan")
        gbits = r["bits_saved_vs_q4body"] / 1e9
        print(f"{i:4d} {r['layer']:3d} {r['class']:8} {r['codec']:18} {r['physical_bpw']:7.3f} {r['hold_cosine']:7.4f} {r['hold_rel_l2_y']:7.4f} {r['final_stream_rel']:8.4f} {vs:7.2f} {r['bits_per_final_err']:10.3e} {gbits:8.3f}")

    print("\n=== BOTTOM 15 (expensive: least bits per unit final error) ===")
    for i, r in enumerate(ranked[-15:], 1):
        q = q4.get(region_key(r))
        vs = r["final_stream_rel"] / q["final_stream_rel"] if q and q["final_stream_rel"] > 0 else float("nan")
        print(f"{i:4d} L{r['layer']:02d} {r['class']:8} {r['codec']:18} bpw={r['physical_bpw']:.3f} cos={r['hold_cosine']:.4f} relY={r['hold_rel_l2_y']:.4f} final={r['final_stream_rel']:.4f} vsQ4={vs:.2f} val={r['bits_per_final_err']:.3e}")

    # per class x bin: mean hold_rel_l2_y for each family (best not, all codecs)
    print("\n=== ASYMMETRY: mean hold_rel_l2_y by class x depth, binary ===")
    acc = defaultdict(list)
    for r in cheap:
        if r["codec"] != "binary_g128":
            continue
        acc[(r["class"], bin_of(r["layer"]))].append(r["hold_rel_l2_y"])
    classes = sorted({c for c, _ in acc})
    bins = ["embed", "L00-07", "L08-23", "L24-47", "L48-63", "lm_head"]
    hdr = f"{'class':10} " + " ".join(f"{b:10}" for b in bins)
    print(hdr)
    for c in classes:
        cells = []
        for b in bins:
            vs = acc.get((c, b), [])
            cells.append(f"{sum(vs)/len(vs):10.4f}" if vs else f"{'—':>10}")
        print(f"{c:10} " + " ".join(cells))

    print("\n=== ASYMMETRY: mean hold_cosine binary ===")
    acc = defaultdict(list)
    for r in cheap:
        if r["codec"] != "binary_g128":
            continue
        acc[(r["class"], bin_of(r["layer"]))].append(r["hold_cosine"])
    print(hdr)
    for c in classes:
        cells = []
        for b in bins:
            vs = acc.get((c, b), [])
            cells.append(f"{sum(vs)/len(vs):10.4f}" if vs else f"{'—':>10}")
        print(f"{c:10} " + " ".join(cells))

    # MLP hardening: per-layer down/gate/up binary hold cosine
    print("\n=== MLP binary residual hold_cosine by layer ===")
    for cls in ("gate", "up", "down"):
        seq = [r for r in cheap if r["class"] == cls and r["codec"] == "binary_g128"]
        seq = sorted(seq, key=lambda r: r["layer"])
        cos = [r["hold_cosine"] for r in seq]
        rely = [r["hold_rel_l2_y"] for r in seq]
        print(f"{cls:5} n={len(seq)} cos min={min(cos):.4f}@L{seq[int(np_argmin(cos))]['layer']} med={median(cos):.4f} max={max(cos):.4f}  relY min={min(rely):.4f} med={median(rely):.4f} max={max(rely):.4f}")

    # budget at thresholds
    print("\n=== BUDGET if collapse regions whose BEST cheap codec meets threshold ===")
    # For each region pick cheapest (lowest bpw) codec that meets the quality bar
    by_reg = defaultdict(list)
    for r in cheap:
        by_reg[region_key(r)].append(r)

    def collapse(pred, label):
        saved = 0.0
        bytes_after = 0
        n = 0
        elems_c = 0
        classes_c = defaultdict(int)
        for k, opts in by_reg.items():
            q = q4.get(k)
            ok = [r for r in opts if pred(r, q)]
            # each region counted once; if multiple tensors share... they don't
            # payload if collapsed: min bpw among ok; else q4 body
            elems = opts[0]["elements"]
            if ok:
                choice = min(ok, key=lambda r: r["physical_bpw"])
                saved += bits_saved(elems, choice["physical_bpw"])
                bytes_after += choice["payload_bytes"]
                n += 1
                elems_c += elems
                classes_c[opts[0]["class"]] += 1
            else:
                # stay at q4
                if q:
                    bytes_after += q["payload_bytes"]
                else:
                    bytes_after += int(math.ceil(elems * Q4_BODY / 8.0))
        # add uncaptured small tensors at f32
        # language N includes small; our rows cover GEMV+tables. small stay f32.
        covered = sum(opts[0]["elements"] for opts in by_reg.values())
        # bytes_after is only covered tensors. complete bpw:
        # (bytes_after + small_f32 + any missing) * 8 / N
        missing = N - covered - SMALL_ELEMS
        # missing should be ~0 if we covered all GEMV+tables. conv1d is in small-ish
        bpw = 8.0 * (bytes_after + SMALL_BYTES) / N
        print(f"{label:50} n_collapse={n:4d} elems={elems_c:14d} frac={elems_c/N:.4f} saved_Gbits={saved/1e9:8.3f} complete_bpw={bpw:.6f} missing_elems={missing} classes={dict(classes_c)}")
        return bpw, n, elems_c, saved

    def bits_saved(elements, bpw):
        return float(elements) * (Q4_BODY - bpw)

    collapse(lambda r, q: r["hold_rel_l2_y"] <= 0.05, "relY<=0.05")
    collapse(lambda r, q: r["hold_rel_l2_y"] <= 0.10, "relY<=0.10")
    collapse(lambda r, q: r["hold_rel_l2_y"] <= 0.20, "relY<=0.20")
    collapse(lambda r, q: r["hold_cosine"] >= 0.99, "cos>=0.99")
    collapse(lambda r, q: r["hold_cosine"] >= 0.95, "cos>=0.95")
    collapse(lambda r, q: r["hold_cosine"] >= 0.90, "cos>=0.90")
    collapse(lambda r, q: q is not None and r["final_stream_rel"] <= 2.0 * q["final_stream_rel"], "final<=2*Q4")
    collapse(lambda r, q: q is not None and r["final_stream_rel"] <= 3.0 * q["final_stream_rel"], "final<=3*Q4")
    collapse(lambda r, q: q is not None and r["final_stream_rel"] <= 5.0 * q["final_stream_rel"], "final<=5*Q4")
    collapse(lambda r, q: r["hold_rel_l2_y"] <= 0.15 and r["physical_bpw"] < 1.5, "relY<=0.15 AND bpw<1.5")
    collapse(lambda r, q: r["family"] in ("zero_plus_sign", "shared_template", "binary") and r["hold_rel_l2_y"] <= 0.20, "sub~1.2 family relY<=0.20")

    # family dominance
    print("\n=== which family wins bits/err per class ===")
    wins = defaultdict(lambda: defaultdict(int))
    for r in ranked:
        wins[r["class"]][r["codec"]] += 1
    for c, m in sorted(wins.items()):
        print(f"  {c:8} {dict(m)}")

    # late down vs early down
    print("\n=== down binary per-layer (hardening) ===")
    downs = sorted([r for r in cheap if r["class"]=="down" and r["codec"]=="binary_g128"], key=lambda r: r["layer"])
    for r in downs:
        print(f"  L{r['layer']:02d} cos={r['hold_cosine']:.4f} relY={r['hold_rel_l2_y']:.4f} relH={r['hold_rel_l2_h']:.4f} final={r['final_stream_rel']:.4f} lo_cos={r['layer_out_hold_cosine']:.4f}")

    # attention out
    print("\n=== out/o binary residual ===")
    for r in sorted([r for r in cheap if r["class"] in ("out","o") and r["codec"]=="binary_g128"], key=lambda r: r["layer"]):
        print(f"  L{r['layer']:02d} {r['class']:3} {r['residual_kind']:24} cos={r['hold_cosine']:.4f} relY={r['hold_rel_l2_y']:.4f} final={r['final_stream_rel']:.4f}")

    # endpoints
    print("\n=== embed / lm_head ===")
    for r in cheap:
        if r["class"] in ("embed", "lm_head"):
            extra = ""
            if "hold_top1_agree" in r:
                extra = f" top1={r['hold_top1_agree']:.4f}"
            print(f"  {r['class']:8} {r['codec']:18} bpw={r['physical_bpw']:.4f} cos={r['hold_cosine']:.4f} relY={r['hold_rel_l2_y']:.4f} final={r['final_stream_rel']:.4f}{extra}")

    # q4 vs cheap ratio distribution
    print("\n=== final / Q4_final for binary ===")
    ratios = []
    for r in cheap:
        if r["codec"] != "binary_g128":
            continue
        q = q4.get(region_key(r))
        if q and q["final_stream_rel"] > 0:
            ratios.append((r["final_stream_rel"] / q["final_stream_rel"], r))
    ratios.sort()
    print(f"n={len(ratios)} min={ratios[0][0]:.2f} p25={ratios[len(ratios)//4][0]:.2f} med={ratios[len(ratios)//2][0]:.2f} p75={ratios[3*len(ratios)//4][0]:.2f} max={ratios[-1][0]:.2f}")
    print("easiest vs Q4:", [(round(a,2), r["layer"], r["class"]) for a,r in ratios[:8]])
    print("hardest vs Q4:", [(round(a,2), r["layer"], r["class"]) for a,r in ratios[-8:]])

    Path("/tmp/g1_sub1bit_summary_meta.json").write_text(json.dumps({
        "n_regions": len(ranked),
        "n_cheap_rows": len(cheap),
        "wall_s": d["wall_s"],
        "rss_max_gb": d["rss_max_gb"],
    }))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def np_argmin(xs):
    m = min(xs)
    return xs.index(m)


if __name__ == "__main__":
    main()
