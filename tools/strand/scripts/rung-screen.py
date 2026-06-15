#!/usr/bin/env python3
"""rung-screen.py — per-tensor rung screening + water-filling allocator (Tier 2a).

Design: docs/STRAND-rung-allocator-design.md. Two subcommands:

  screen    Invoke `quantize-model --measure-only` per rung config over a model dir and
            emit the per-tensor screening table (CSV): n, bits, billed bpw, rel-RMS per
            (tensor, config). rel-RMS is a WITHIN-FAMILY proxy only (will.md §5.5): this
            table ranks rungs per tensor and flags anomalies; it must NOT be used to
            compare importance across tensors — that is stage 2's job (rung-kl.py, the
            swap-KL screen; join its output here via --stage2-csv).
  allocate  Discrete water-filling over rungs (HAWQ-style reverse water-filling, will.md
            §4 queue #7): minimize avg billed bpw s.t. sum(per-tensor damage) <= target.
            Requires a true damage column (dkl_nats / dnll_nats); REFUSES rel_rms_pct
            unless --force-rel-rms (plumbing tests only).

Modes (screen): `batch` (default) = ONE invocation per (config, shard); the sidecar JSON
already carries per-tensor rows, and tensors quantize independently (per-tensor RHT seed,
per-tensor scales, per-256-block Viterbi) so the numbers are byte-identical to per-tensor
invocation at 1/168th the process+shard-read overhead. `per-tensor` = one invocation per
tensor via `--only <full tensor name>` (names are unique; --only is substring) — for spot
checks and future per-tensor flag sweeps.

Ops constraints baked in (will.md §7): STRAND_NO_GPU=1 (Metal encode watchdog SIGKILLs
7B-wide tensors; CPU SIMD encode beats the serialized GPU ~8x), nice -n 19 (the box runs
a training marathon), --threads 4 default. Never run concurrently with a QAT/PV arm on
the 18 GB box. --dry-run prints the command plan and writes NOTHING.

Examples:
  rung-screen.py screen --model-dir scratch/qwen-05b --out-dir scratch/rung-screen --dry-run
  rung-screen.py screen --model-dir scratch/qwen-05b --out-dir scratch/rung-screen
  rung-screen.py allocate --screen-csv scratch/rung-screen/rung-screen.csv \
      --stage2-csv scratch/rung-screen/rung-kl.csv --damage-col dkl_nats \
      --target-dnats 0.25 --out-dir scratch/rung-screen
"""

import argparse
import csv
import heapq
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import time

# Mirrors is_quantizable_linear in quantize-model.rs: 2-D + one of the 7 proj suffixes.
PROJ_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)
SUFFIX_ORDER = {s: i for i, s in enumerate(PROJ_SUFFIXES)}

# Canon rungs (will.md §3/§8): q2_l12_out1 = 80.7, q3_l12_out1 = 16.11, q4_l12 = 13.535
# on the 0.5B. --l 12 explicit applies per-rung via for_bpw_l. q4 carries no outlier in
# canon; in a single mixed invocation the outlier flag is global (design doc §3.3).
DEFAULT_CONFIGS = [
    {"name": "r2", "flags": "--bits 2 --l 12 --outlier-channel 1"},
    {"name": "r3", "flags": "--bits 3 --l 12 --outlier-channel 1"},
    {"name": "r4", "flags": "--bits 4 --l 12"},
]


def sanitize(name):
    """Filename-safe tensor name; mirrors sanitize_for_filename in quantize-model.rs."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def sh(cmd):
    try:
        return shlex.join(cmd)
    except AttributeError:  # < py3.8
        return " ".join(shlex.quote(c) for c in cmd)


# ---------------------------------------------------------------------------
# Model discovery (read-only: 8-byte header length + JSON header per shard)
# ---------------------------------------------------------------------------
def find_shards(model_dir):
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    shards = sorted(
        os.path.join(model_dir, f) for f in os.listdir(model_dir)
        if re.match(r"model-\d+-of-\d+\.safetensors$", f)
    )
    if not shards:
        raise SystemExit(f"[rung-screen] no model*.safetensors in {model_dir}")
    return shards


def read_header(path):
    """safetensors header: u64 LE length, then JSON {name: {dtype, shape, data_offsets}}."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen).decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr


def quantizable_tensors(shard):
    """[(name, n)] for proj linears in this shard, header-order. A tensor never spans
    shards (strand-7b-ppl.sh invariant), so per-shard lists concatenate cleanly."""
    out = []
    for name, meta in read_header(shard).items():
        shape = meta.get("shape", [])
        if len(shape) == 2 and any(name.endswith(s) for s in PROJ_SUFFIXES):
            out.append((name, shape[0] * shape[1]))
    return out


def tensor_sort_key(name):
    m = re.search(r"layers\.(\d+)\.", name)
    layer = int(m.group(1)) if m else 10 ** 6
    suf = next((SUFFIX_ORDER[s] for s in PROJ_SUFFIXES if name.endswith(s)), 99)
    return (layer, suf, name)


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def build_cmd(args, shard, out_prefix, cfg_flags, only=None):
    cmd = []
    if args.nice > 0:
        cmd += ["nice", "-n", str(args.nice)]
    # --out is REQUIRED even in measure-only: the sidecar lands at <out>.json and is
    # only written when --out is non-empty (quantize-model.rs:1248-1252).
    cmd += [args.bin, "--in", shard, "--out", out_prefix, "--measure-only"]
    flags = shlex.split(cfg_flags)
    cmd += flags
    if "--threads" not in flags:
        cmd += ["--threads", str(args.threads)]
    if only is not None:
        cmd += ["--only", only]
    return cmd


def plan_screen(args, configs, shards):
    """[(config_name, shard, out_prefix, cmd)] — the full invocation plan."""
    plan = []
    for cfg in configs:
        cdir = os.path.join(args.out_dir, cfg["name"])
        for shard in shards:
            base = os.path.basename(shard).replace(".safetensors", "")
            if args.mode == "batch":
                pref = os.path.join(cdir, f"{base}.measure")
                plan.append((cfg["name"], shard, pref, build_cmd(args, shard, pref, cfg["flags"])))
            else:  # per-tensor: identical numbers, 168x the overhead; spot-check mode
                for tname, _n in quantizable_tensors(shard):
                    pref = os.path.join(cdir, f"{base}__{sanitize(tname)}.measure")
                    plan.append((cfg["name"], shard, pref,
                                 build_cmd(args, shard, pref, cfg["flags"], only=tname)))
    return plan


def parse_sidecar(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("tensors", []), d.get("aggregate", {})


def run_screen(args):
    configs = DEFAULT_CONFIGS
    if args.configs:
        with open(args.configs) as f:
            configs = json.load(f)
        for c in configs:
            assert "name" in c and "flags" in c, f"config entry missing name/flags: {c}"
    shards = find_shards(args.model_dir)
    plan = plan_screen(args, configs, shards)

    env_note = "STRAND_NO_GPU=1 " if not args.allow_gpu else ""
    if args.dry_run:
        print(f"[rung-screen] DRY-RUN: {len(plan)} invocation(s); nothing written.")
        print(f"[rung-screen] model={args.model_dir} shards={len(shards)} "
              f"configs={[c['name'] for c in configs]} mode={args.mode}")
        if not os.path.exists(args.bin):
            print(f"[rung-screen] note: --bin {args.bin} does not exist yet "
                  f"(fine for a dry run; build or point --bin at scratch/bin/quantize-model)")
        shown = 0
        for cname, _shard, pref, cmd in plan:
            if shown < args.dry_limit:
                print(f"  [{cname}] {env_note}{sh(cmd)}")
                print(f"           -> sidecar {pref}.json")
                shown += 1
        if len(plan) > shown:
            print(f"  ... {len(plan) - shown} more (raise --dry-limit to see all)")
        print(f"[rung-screen] then: parse sidecars -> {args.out_dir}/rung-screen.csv "
              f"(+ -wide.csv, -aggregate.csv, cmds.txt)")
        return 0

    if not os.path.exists(args.bin):
        raise SystemExit(f"[rung-screen] --bin {args.bin} not found")
    env = dict(os.environ)
    if not args.allow_gpu:
        env["STRAND_NO_GPU"] = "1"

    os.makedirs(args.out_dir, exist_ok=True)
    cmd_log = open(os.path.join(args.out_dir, "cmds.txt"), "a")
    rows, aggs, failures = [], [], 0
    for cname, shard, pref, cmd in plan:
        sidecar = pref + ".json"
        os.makedirs(os.path.dirname(pref), exist_ok=True)
        if args.resume and os.path.exists(sidecar):
            try:
                tensors, agg = parse_sidecar(sidecar)
                print(f"[rung-screen] RESUME {cname} {os.path.basename(sidecar)} "
                      f"({len(tensors)} tensors)")
                rows += [(cname, shard, t) for t in tensors]
                aggs.append((cname, shard, agg))
                continue
            except Exception:
                pass  # unparseable partial -> re-run
        print(f"[rung-screen] RUN {cname}: {env_note}{sh(cmd)}", flush=True)
        cmd_log.write(env_note + sh(cmd) + "\n")
        cmd_log.flush()
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(sidecar):
            failures += 1
            print(f"[rung-screen] FAIL rc={r.returncode} ({time.time()-t0:.0f}s):\n"
                  f"{(r.stderr or r.stdout)[-600:]}", file=sys.stderr)
            continue
        tensors, agg = parse_sidecar(sidecar)
        print(f"[rung-screen] done {cname} {os.path.basename(shard)}: {len(tensors)} tensors "
              f"in {time.time()-t0:.0f}s  agg_bpw={agg.get('effective_bpw', 0):.4f} "
              f"rel-RMS={agg.get('weighted_rel_rms_pct', 0):.2f}%")
        rows += [(cname, shard, t) for t in tensors]
        aggs.append((cname, shard, agg))
    cmd_log.close()

    write_tables(args, rows, aggs)
    if failures:
        print(f"[rung-screen] {failures} invocation(s) FAILED — table is partial",
              file=sys.stderr)
    return 1 if failures else 0


def write_tables(args, rows, aggs):
    stage2 = {}
    if args.stage2_csv and os.path.exists(args.stage2_csv):
        with open(args.stage2_csv) as f:
            for r in csv.DictReader(f):
                stage2[(r["tensor"], r["config"])] = r
        print(f"[rung-screen] joined stage-2: {len(stage2)} (tensor, config) rows")

    s2cols = ["dkl_nats", "dnll_nats"] if stage2 else []
    long_path = os.path.join(args.out_dir, "rung-screen.csv")
    with open(long_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_dir", "shard", "config", "tensor", "n", "bits", "bpw",
                    "rel_rms_pct"] + s2cols)
        for cname, shard, t in sorted(rows, key=lambda r: (tensor_sort_key(r[2]["name"]), r[0])):
            extra = [stage2.get((t["name"], cname), {}).get(c, "") for c in s2cols]
            w.writerow([args.model_dir, os.path.basename(shard), cname, t["name"],
                        t["n"], t["bits"], f"{t['bpw']:.6f}", f"{t['rel_rms_pct']:.6f}"]
                       + extra)

    # Wide pivot: one row per tensor, one column group per config.
    configs = sorted({c for c, _s, _t in rows})
    by_tensor = {}
    for cname, _shard, t in rows:
        by_tensor.setdefault(t["name"], {"n": t["n"]})[cname] = t
    wide_path = os.path.join(args.out_dir, "rung-screen-wide.csv")
    with open(wide_path, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["tensor", "n"]
        for c in configs:
            hdr += [f"{c}.bits", f"{c}.bpw", f"{c}.rel_rms_pct"]
            hdr += [f"{c}.{x}" for x in s2cols]
        w.writerow(hdr)
        mono_viol = []
        for name in sorted(by_tensor, key=tensor_sort_key):
            d = by_tensor[name]
            row = [name, d["n"]]
            for c in configs:
                t = d.get(c)
                row += ([t["bits"], f"{t['bpw']:.6f}", f"{t['rel_rms_pct']:.6f}"]
                        if t else ["", "", ""])
                row += [stage2.get((name, c), {}).get(x, "") for x in s2cols]
            w.writerow(row)
            # Sanity gate (E1): rel-RMS must fall as bits rise, per tensor.
            pts = sorted((d[c]["bits"], d[c]["rel_rms_pct"]) for c in configs if c in d)
            if any(pts[i][1] < pts[i + 1][1] for i in range(len(pts) - 1)):
                mono_viol.append(name)

    agg_path = os.path.join(args.out_dir, "rung-screen-aggregate.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "shard", "quantized_weights", "effective_bpw",
                    "weighted_rel_rms_pct"])
        for cname, shard, a in aggs:
            w.writerow([cname, os.path.basename(shard), a.get("quantized_weights", ""),
                        a.get("effective_bpw", ""), a.get("weighted_rel_rms_pct", "")])

    print(f"[rung-screen] wrote {long_path} ({len(rows)} rows), {wide_path}, {agg_path}")
    if mono_viol:
        print(f"[rung-screen] MONOTONICITY VIOLATION (rel-RMS rose with bits) on "
              f"{len(mono_viol)} tensor(s): {mono_viol[:4]} — encoder/config bug, "
              f"investigate before allocating", file=sys.stderr)


# ---------------------------------------------------------------------------
# allocate — discrete water-filling on the convexified per-tensor damage curves
# ---------------------------------------------------------------------------
def load_screen(args):
    """{tensor: {"n": int, "rungs": [(config, bits, cost_bits, damage)] cost-ascending}}"""
    stage2 = {}
    if args.stage2_csv:
        with open(args.stage2_csv) as f:
            for r in csv.DictReader(f):
                stage2[(r["tensor"], r["config"])] = r
    tensors = {}
    with open(args.screen_csv) as f:
        for r in csv.DictReader(f):
            key = (r["tensor"], r["config"])
            src = stage2.get(key, r)
            dmg = src.get(args.damage_col, "")
            if dmg in ("", None):
                raise SystemExit(
                    f"[allocate] no '{args.damage_col}' for {key}; run/join stage 2 "
                    f"(rung-kl.py) or pick --damage-col present in the CSV")
            t = tensors.setdefault(r["tensor"], {"n": int(r["n"]), "rungs": []})
            t["rungs"].append((r["config"], int(r["bits"]),
                               int(r["n"]) * float(r["bpw"]), float(dmg)))
    for t in tensors.values():
        t["rungs"].sort(key=lambda x: x[2])
    return tensors


def lower_hull(rungs):
    """Keep cost-ascending points with strictly decreasing damage AND decreasing marginal
    efficiency e_j = -ddamage/dcost (the water-filling needs convex curves; dominated and
    concave points are skipped, standard convexification)."""
    pts = []
    for p in rungs:
        if pts and p[3] >= pts[-1][3]:
            continue  # dominated: more bits, no less damage
        pts.append(p)
        while len(pts) >= 3:
            (c0, d0), (c1, d1), (c2, d2) = [(q[2], q[3]) for q in pts[-3:]]
            e01 = (d0 - d1) / max(c1 - c0, 1e-12)
            e12 = (d1 - d2) / max(c2 - c1, 1e-12)
            if e12 > e01:        # middle point breaks convexity -> drop it
                pts.pop(-2)
            else:
                break
    return pts


def run_allocate(args):
    if args.damage_col == "rel_rms_pct" and not args.force_rel_rms:
        raise SystemExit(
            "[allocate] rel_rms_pct is a WITHIN-FAMILY proxy (will.md §5.5; the diag-H "
            "tell: rel-RMS fell while PPL rose). Water-filling across tensors on it is "
            "invalid. Join stage-2 damage (--stage2-csv, dkl_nats) or pass "
            "--force-rel-rms for a plumbing test.")
    tensors = load_screen(args)
    names = sorted(tensors, key=tensor_sort_key)
    total_n = sum(tensors[t]["n"] for t in names)

    hull = {t: lower_hull(tensors[t]["rungs"]) for t in names}
    state = {t: 0 for t in names}  # index into hull[t]; start = cheapest rung
    total_dmg = sum(hull[t][0][3] for t in names)
    total_cost = sum(hull[t][0][2] for t in names)

    # Heap of next upgrade per tensor, keyed by -efficiency (max damage removed per bit).
    heap = []
    def push_next(t):
        i = state[t]
        if i + 1 < len(hull[t]):
            c0, d0 = hull[t][i][2], hull[t][i][3]
            c1, d1 = hull[t][i + 1][2], hull[t][i + 1][3]
            heapq.heappush(heap, (-(d0 - d1) / max(c1 - c0, 1e-12), t, i + 1))
    for t in names:
        push_next(t)

    if args.target_dnats is None and args.budget_bpw is None:
        raise SystemExit("[allocate] need --target-dnats or --budget-bpw")
    target = None
    if args.target_dnats is not None:
        target = args.target_dnats / args.alpha   # superadditivity safety factor
    upgrades = 0
    while heap:
        if target is not None and total_dmg <= target:
            break
        neg_e, t, nxt = heapq.heappop(heap)
        if nxt != state[t] + 1:
            continue  # stale entry
        dc = hull[t][nxt][2] - hull[t][state[t]][2]
        if args.budget_bpw is not None and (total_cost + dc) / total_n > args.budget_bpw:
            continue  # does not fit; try the next-best edge
        total_dmg -= hull[t][state[t]][3] - hull[t][nxt][3]
        total_cost += dc
        state[t] = nxt
        upgrades += 1
        push_next(t)

    # Classification at the final assignment (design doc §2.3).
    n_tensors = len(names)
    tau_g = (0.3 * (args.target_dnats or total_dmg * args.alpha) / max(n_tensors, 1))
    assign, red, green = {}, [], []
    for t in names:
        cfg, bits, cost, dmg = hull[t][state[t]]
        assign[t] = {"config": cfg, "bits": bits, "n": tensors[t]["n"],
                     "bpw": cost / tensors[t]["n"], "damage": dmg}
        if dmg >= args.tau_red:
            red.append(t)
        elif dmg <= tau_g:
            green.append(t)
    avg_bpw = total_cost / total_n
    fallback_bits = min(a["bits"] for a in assign.values())

    # mp-config rules: exact dotted tensor-path substrings (minus .weight). Patterns must
    # not collide with any OTHER tensor name (substring semantics, first match wins).
    rules = []
    all_names = set(names)
    for t in names:
        if assign[t]["bits"] == fallback_bits:
            continue
        pat = t[:-len(".weight")] if t.endswith(".weight") else t
        hits = [o for o in all_names if pat in o]
        assert hits == [t], f"pattern {pat!r} is ambiguous: {hits}"
        rules.append({"pattern": pat, "bits": assign[t]["bits"]})

    pv_regex = ("^(?:" + "|".join(
        re.escape(t[:-len(".weight")] if t.endswith(".weight") else t) for t in red)
        + ")$") if red else ""

    summary = {
        "damage_col": args.damage_col, "alpha": args.alpha,
        "target_dnats": args.target_dnats, "budget_bpw": args.budget_bpw,
        "tau_green": tau_g, "tau_red": args.tau_red,
        "tensors": n_tensors, "upgrades": upgrades,
        "avg_bpw_billed": avg_bpw, "sum_damage_nats": total_dmg,
        "predicted_ln_ppl_ratio": total_dmg * args.alpha,
        "fallback_bits": fallback_bits, "rules": len(rules),
        "green": len(green), "amber": n_tensors - len(green) - len(red), "red": len(red),
        "pv_set": [t for t in red],
        "bits_histogram": {
            str(b): sum(1 for a in assign.values() if a["bits"] == b)
            for b in sorted({a["bits"] for a in assign.values()})},
    }

    if args.dry_run:
        print(f"[allocate] DRY-RUN — nothing written")
        print(json.dumps(summary, indent=2))
        print(f"[allocate] would write: mp-alloc.json ({len(rules)} rules, "
              f"fallback --bits {fallback_bits}), pv-tensors.regex "
              f"({len(red)} RED), alloc-summary.json, alloc-assign.csv")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "mp-alloc.json"), "w") as f:
        json.dump(rules, f, indent=2)
    with open(os.path.join(args.out_dir, "pv-tensors.regex"), "w") as f:
        f.write(pv_regex + "\n")
    with open(os.path.join(args.out_dir, "alloc-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "alloc-assign.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tensor", "n", "config", "bits", "bpw", "damage", "class"])
        for t in names:
            a = assign[t]
            cls = "RED" if t in red else ("GREEN" if t in green else "AMBER")
            w.writerow([t, a["n"], a["config"], a["bits"], f"{a['bpw']:.6f}",
                        f"{a['damage']:.8f}", cls])
    print(f"[allocate] avg billed bpw = {avg_bpw:.4f}  sum damage = {total_dmg:.5f} nats "
          f"(alpha-adjusted ln-ratio {total_dmg*args.alpha:.5f})")
    print(f"[allocate] classes: GREEN {len(green)} / AMBER "
          f"{n_tensors-len(green)-len(red)} / RED {len(red)} (the PV set)")
    print(f"[allocate] wrote mp-alloc.json ({len(rules)} rules, fallback --bits "
          f"{fallback_bits}), pv-tensors.regex, alloc-summary.json, alloc-assign.csv "
          f"-> {args.out_dir}")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen", help="per-tensor measure-only screening table")
    s.add_argument("--model-dir", required=True, help="HF model dir (model*.safetensors)")
    s.add_argument("--bin", default="target/release/quantize-model")
    s.add_argument("--configs", default="",
                   help='JSON list [{"name","flags"}]; default = canon r2/r3/r4 (l=12)')
    s.add_argument("--out-dir", required=True)
    s.add_argument("--mode", choices=["batch", "per-tensor"], default="batch")
    s.add_argument("--threads", type=int, default=4,
                   help="per-invocation encoder threads (busy box: keep low, nice -n 19)")
    s.add_argument("--nice", type=int, default=19, help="0 disables the nice prefix")
    s.add_argument("--allow-gpu", action="store_true",
                   help="drop STRAND_NO_GPU=1 (Metal watchdog SIGKILLs 7B-wide tensors)")
    s.add_argument("--resume", action="store_true",
                   help="reuse existing parseable sidecars (repo --resume convention)")
    s.add_argument("--stage2-csv", default="",
                   help="join rung-kl.py output (tensor,config,dkl_nats[,dnll_nats])")
    s.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    s.add_argument("--dry-limit", type=int, default=8)

    a = sub.add_parser("allocate", help="water-filling rung assignment + PV set")
    a.add_argument("--screen-csv", required=True, help="long CSV from `screen`")
    a.add_argument("--stage2-csv", default="")
    a.add_argument("--damage-col", default="dkl_nats")
    a.add_argument("--target-dnats", type=float, default=None,
                   help="quality constraint: sum damage <= target/alpha (nats of ln-PPL-ratio)")
    a.add_argument("--budget-bpw", type=float, default=None,
                   help="alternative: max avg billed bpw (walks the same frontier)")
    a.add_argument("--alpha", type=float, default=1.5,
                   help="superadditivity safety factor (E2c calibrates)")
    a.add_argument("--tau-red", type=float, default=0.05,
                   help="per-tensor nats: >= tau -> RED/collapsed -> the PV set")
    a.add_argument("--force-rel-rms", action="store_true",
                   help="allow allocating on rel_rms_pct (PLUMBING TESTS ONLY — §5.5)")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    if args.cmd == "screen":
        sys.exit(run_screen(args))
    sys.exit(run_allocate(args))


if __name__ == "__main__":
    main()
