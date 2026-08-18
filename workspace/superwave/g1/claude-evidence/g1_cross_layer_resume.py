#!/usr/bin/env python3
"""Resume remaining DN classes + heads + hidden-scale fold after probe-set bug."""
from __future__ import annotations

import json
import os
import sys
import time

# import functions from the (now fixed) main script
src = open("/tmp/g1_cross_layer_structure.py").read()
ns = {}
exec(compile(src.replace("raise SystemExit(main())", "pass"), "/tmp/g1_cross_layer_structure.py", "exec"), ns)

for k, v in ns.items():
    if not k.startswith("_"):
        globals()[k] = v

OUT = ns["OUT"]
log = ns["log"]
dump = ns["dump"]
parse_all_headers = ns["parse_all_headers"]
ROOT = ns["ROOT"]
CLASSES = ns["CLASSES"]
process_class = ns["process_class"]
tensor_name = ns["tensor_name"]
head_structure_gqa_q = ns["head_structure_gqa_q"]
head_structure_dn_qkv = ns["head_structure_dn_qkv"]
load_hidden = ns["load_hidden"]
rss_gb = ns["rss_gb"]
T0 = ns["T0"]


def main():
    report = json.loads(OUT.read_text())
    done = set(report.get("classes", {}))
    log(f"resume done={sorted(done)}")
    table = parse_all_headers(ROOT)
    need_X = set(ns["PROBE_MLP"] + ns["PROBE_GQA"] + ns["PROBE_DN"])
    X_by_layer = {L: load_hidden(L) for L in sorted(need_X)}
    for spec in CLASSES:
        if spec["name"] in done:
            log(f"skip {spec['name']}")
            continue
        process_class(table, spec, X_by_layer, report)

    log("heads GQA q L3/L31/L63")
    report.setdefault("heads", {})
    report["heads"]["gqa_q"] = {}
    for L in (3, 31, 63):
        info = table[tensor_name(L, "self_attn.q_proj.weight")]
        report["heads"]["gqa_q"][str(L)] = head_structure_gqa_q(info, L)
        dump(report)
        log(f"  gqa q L{L} {report['heads']['gqa_q'][str(L)]['q_weight_cos']}")

    log("heads DN qkv L0/L32/L62")
    report["heads"]["dn_qkv"] = {}
    for L in (0, 32, 62):
        info = table[tensor_name(L, "linear_attn.in_proj_qkv.weight")]
        report["heads"]["dn_qkv"][str(L)] = head_structure_dn_qkv(info, L)
        dump(report)
        log(f"  dn qkv L{L} qcos={report['heads']['dn_qkv'][str(L)]['q_weight_cos']['mean']}")

    report["rss_max_gb"] = rss_gb()
    report["elapsed_s"] = time.time() - T0
    report["resumed"] = True
    dump(report)
    log(f"DONE elapsed={report['elapsed_s']:.1f}s rss_max={report['rss_max_gb']:.3f}GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
