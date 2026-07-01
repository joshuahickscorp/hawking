#!/usr/bin/env python3.12
"""size_frontier.py — the BLACK-HOLE size axis: how many parameters can a device pull in, orthogonal
to per-weight quality. llama.cpp caps at what fits in RAM (~500B @ 1.34bpw on 84GB). Hawking's move
is to stop requiring the whole model resident — the model LIVES on the SSD and only the weights a
token touches stream through RAM (the event horizon). Three regimes, honestly separated by speed:

  RESIDENT   (fast, full speed) : whole .tq in RAM. ceiling = RAM_budget * 8 / bpw  (~500B @1.34/84GB).
  MOE-PAGED  (usable)           : only ACTIVE experts + a hot-expert cache resident; cold experts page
                                  from SSD on demand. Footprint ~ active_params, NOT total. All giant
                                  models are MoE, so this is the lever: total params bounded by SSD,
                                  speed bounded by (active .tq bytes / SSD bandwidth), lifted by cache.
  DENSE-OOC  (fits, slow)       : layer-streamed out-of-core; the whole .tq streams per token. Fits to
                                  the SSD ceiling but SSD-bandwidth bound (~0.03 tok/s at 1T) - batch/async.

Device ceilings: RESIDENT = RAM/bpw*8; STORAGE (the real black-hole limit) = SSD/bpw*8. Pushing higher:
lower bpw (raises both ceilings), MoE expert-paging + prefetch + hot cache (hides SSD latency), bigger/
faster external NVMe (ceiling scales with storage). Projection tool; the OOC pager + per-expert .tq
serve are the Rust serve build. KILL: dense-OOC below ~0.1 tok/s is not interactive (MoE or bust at scale).
"""
import sys, os, json

OUT = "reports/condense"
DEVICES = {  # name -> (RAM_GB usable for weights, SSD_TB, SSD_read_GBps)
    "studio-m2max": (84.0, 2.0, 5.0), "mbp-36": (28.0, 1.0, 5.0),
    "mbp-64": (52.0, 2.0, 5.0), "studio-m3ultra-512": (470.0, 8.0, 6.0),
}


def tq_gb(p_b, bpw): return p_b * bpw / 8.0


def analyze(total_b, active_b, bpw, dev, hot_cache_gb=20.0, kv_gb=6.0):
    ram, ssd_tb, bw = DEVICES.get(dev, DEVICES["studio-m2max"])
    ssd_gb = ssd_tb * 1000
    store = tq_gb(total_b, bpw)
    res = {"device": dev, "ram_gb": ram, "ssd_gb": ssd_gb, "ssd_bw_gbps": bw,
           "model_total_b": total_b, "active_b": active_b, "bpw": bpw,
           "tq_on_disk_gb": round(store, 1), "fits_ssd": store <= ssd_gb,
           "resident_ceiling_b": round(ram * 8 / bpw),
           "storage_ceiling_b": round(ssd_gb * 8 / bpw)}
    if store <= ram:
        res["best_regime"] = "RESIDENT"
        res["resident_gb"] = round(store + kv_gb, 1)
        res["est_tok_s"] = ">full speed (all in RAM)"
    elif active_b:                       # MoE: page cold experts, keep active + hot cache
        active_gb = tq_gb(active_b, bpw)
        resident = active_gb + hot_cache_gb + kv_gb
        res["best_regime"] = "MOE-PAGED" if store <= ssd_gb else "TOO-BIG"
        res["resident_gb"] = round(resident, 1)
        res["est_tok_s_cold"] = round(bw / active_gb, 2) if active_gb else None
        res["note"] = "only active experts stream/token; hot-expert cache lifts warm tok/s well above cold"
    else:                                # dense out-of-core
        res["best_regime"] = "DENSE-OOC" if store <= ssd_gb else "TOO-BIG"
        res["resident_gb"] = round(min(store, ram) + kv_gb, 1)
        res["est_tok_s"] = round(bw / store, 3)
        res["note"] = "whole model streams/token; SSD-bandwidth bound; batch/async, not interactive"
    return res


def report(total_b, active_b, bpw, dev):
    r = analyze(total_b, active_b, bpw, dev)
    os.makedirs(OUT, exist_ok=True)
    lbl = f"{int(total_b)}b_{dev}"
    json.dump(r, open(f"{OUT}/size_{lbl}.json", "w"), indent=2)
    print(f"[size] {total_b}B {'MoE act '+str(active_b)+'B' if active_b else 'dense'} @ {bpw}bpw on {dev} "
          f"(RAM {r['ram_gb']}GB, SSD {r['ssd_gb']:.0f}GB):", file=sys.stderr)
    print(f"  .tq on disk {r['tq_on_disk_gb']}GB  regime={r['best_regime']}  resident~{r.get('resident_gb','?')}GB  "
          f"tok/s~{r.get('est_tok_s', r.get('est_tok_s_cold','?'))}", file=sys.stderr)
    print(f"  device ceilings: RESIDENT {r['resident_ceiling_b']}B (fast) | STORAGE {r['storage_ceiling_b']}B "
          f"(out-of-core, the black-hole limit)", file=sys.stderr)
    if r["best_regime"] == "TOO-BIG":
        print(f"# KILL: {r['tq_on_disk_gb']}GB > {r['ssd_gb']:.0f}GB SSD — needs lower bpw or bigger storage", file=sys.stderr)
    elif r["best_regime"] == "DENSE-OOC" and isinstance(r.get("est_tok_s"), (int, float)) and r["est_tok_s"] < 0.1:
        print(f"# NOTE: dense-OOC at {r['est_tok_s']} tok/s is batch/async only; a MoE of this size would run ~10-20x faster", file=sys.stderr)
    return r


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if a == "--ceiling":
        dev = sys.argv[2] if len(sys.argv) > 2 else "studio-m2max"
        ram, ssd_tb, bw = DEVICES[dev]
        print(f"{dev}: RAM {ram}GB, SSD {ssd_tb}TB, {bw}GB/s")
        for bpw in (4.5, 3.34, 2.34, 1.34, 1.0):
            print(f"  {bpw}bpw: RESIDENT {round(ram*8/bpw)}B (fast) | STORAGE {round(ssd_tb*1000*8/bpw)}B (out-of-core)")
    elif a == "--help":
        print(__doc__)
    else:
        total = float(a)
        active = None; dev = "studio-m2max"; bpw = 1.34
        if "--active" in sys.argv: active = float(sys.argv[sys.argv.index("--active")+1])
        if "--bpw" in sys.argv: bpw = float(sys.argv[sys.argv.index("--bpw")+1])
        if "--device" in sys.argv: dev = sys.argv[sys.argv.index("--device")+1]
        report(total, active, bpw, dev)
