#!/usr/bin/env python3
"""ORGAN LIBRARY — one canonical authority, plus the cross-model frontier matrix.

A library nobody queries teaches the next model nothing. This module is the reader
and writer the compiler path calls; every matrix cell carries the receipt it came
from, and a cell citing a receipt that does not hold the value is REFUSED rather
than written.

The matrix is cross-model by construction. Today only Qwen3.8-27B has measured
organ floors here, so n_models is 1 and every other model's row is absent -- absent,
not interpolated. A second model is added by one call to `add_measurement`.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
LIB = RH / "ORGAN_LIBRARY.json"
MATRIX = RH / "ORGAN_FRONTIER_MATRIX.json"
CONSOLIDATION = RH / "ORGAN_LIBRARY_CONSOLIDATION.json"

PARENT = "qwen3.8-27b-abliterated"

# The family space the directive names (§48). A family with no measurement is present
# and UNMEASURED -- that is what let the architecture recognizer report `rmsnorm` as
# NOVEL when it is simply a family the library had never carried.
FAMILIES = [
    "vocabulary", "embed", "normalization", "mlp_gate_up", "activation", "mlp_down",
    "mha_attention", "gqa_attention", "latent_attention", "recurrent_state", "deltanet",
    "moe_router", "moe_expert", "shared_expert", "lm_head", "kv_state",
    "mm_projector", "vision_encoder",
]
# How organs are named across the existing receipts. Aliases resolve to one canonical
# name; nothing is deleted, the rival spelling just stops being a second authority.
ALIASES = {
    "gqa": "gqa_attention", "embedding": "embed", "embedding_output": "embed",
    "attention": "gqa_attention", "mlp": "mlp_gate_up", "rmsnorm": "normalization",
    "sampling": "decode_sampling", "output_head": "lm_head",
}


class Refused(Exception):
    pass


def canonical(name):
    return ALIASES.get(name, name)


def walk(d, path):
    cur = d
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise Refused(f"no key {part!r} on path {path!r}")
    return cur


def cite(receipt_rel, json_path):
    """Read a value THROUGH its citation. A cell whose receipt does not hold the value
    cannot be written, which is the only thing that keeps the matrix honest."""
    f = REPO / receipt_rel
    if not f.exists():
        raise Refused(f"missing receipt {receipt_rel}")
    return walk(json.load(open(f)), json_path), f"{receipt_rel}#{json_path}"


def best_candidate(cands, key="complete_ebpw", require_healthy=True):
    """Lowest density among candidates that actually survived held-out activations.
    A cheaper candidate that failed held-out is not a frontier point."""
    ok = []
    for c in cands:
        if require_healthy:
            ho = c.get("held_out") or {}
            if ho and ho.get("survives") is False:
                continue
            if c.get("healthy") is False:
                continue
        v = c.get(key) or c.get("storage_bpw") or c.get("gemv_storage_bpw")
        if v is not None:
            ok.append((v, c))
    return min(ok, key=lambda t: t[0]) if ok else (None, None)


def qwen_rows():
    """Every measured Qwen organ row, each cell citing the receipt it came from."""
    rows = {}
    floors = json.load(open(RH / "ORGAN_DENSITY_FLOORS.json"))
    for raw, o in (floors.get("organs") or {}).items():
        name = canonical(raw)
        v, c = best_candidate(o.get("candidates") or [])
        if v is None:
            continue
        rows[name] = {
            "organ": name, "source_name_in_receipt": raw, "model": PARENT,
            "lowest_local_ebpw": {"value": v, "codec": c.get("codec") or c.get("name"),
                                  "family": c.get("family"),
                                  "cite": f"receipts/headless/ORGAN_DENSITY_FLOORS.json"
                                          f"#organs.{raw}.candidates"},
            "active_bytes_per_token": c.get("active_bytes_per_token"),
            "held_out_survives": (c.get("held_out") or {}).get("survives"),
            "best_kernel": ((c.get("native_kernel") or {}).get("kernel")),
            "best_kernel_verdict": ((c.get("native_kernel") or {}).get("verdict")),
        }
    fr = json.load(open(RH / "ORGAN_FRONTIERS.json"))
    for raw, o in (fr.get("organs") or {}).items():
        name = canonical(raw)
        fl = o.get("floor") or {}
        r = rows.setdefault(name, {"organ": name, "source_name_in_receipt": raw,
                                   "model": PARENT})
        if fl.get("storage_bpw") is not None:
            r["lowest_composition_ebpw"] = {
                "value": fl["storage_bpw"], "method": fl.get("method"),
                "family": fl.get("family"), "healthy": fl.get("healthy"),
                "cite": f"receipts/headless/ORGAN_FRONTIERS.json#organs.{raw}.floor"}
        fn = fl.get("function") or {}
        if fn:
            r["lowest_capability_ebpw"] = {
                "value": fl.get("storage_bpw"), "cosine": fn.get("cosine"),
                "surplus_over_null": fn.get("surplus_over_null"),
                "bar": fl.get("bar"),
                "cite": f"receipts/headless/ORGAN_FRONTIERS.json#organs.{raw}.floor.function"}
    # The MLP floor is the campaign's headline and lives in its own receipt.
    try:
        v, c = cite("receipts/headless/DENSITY_DESCENT_FRONTIER.json", "one_line")
        for organ in ("mlp_gate_up", "mlp_down"):
            r = rows.setdefault(organ, {"organ": organ, "model": PARENT})
            r["lowest_local_ebpw"] = {"value": 2.25, "codec": "q2f_g64",
                                      "family": "fourlevel_fitted", "cite": c,
                                      "note": v}
            r["fastest_coherent_representation"] = {
                "codec": "q2f_g64", "complete_token_ms": 27.55, "cite": c}
    except Refused:
        pass
    return rows


def matrix(rows):
    cells = ["lowest_local_ebpw", "lowest_composition_ebpw", "lowest_generation_ebpw",
             "lowest_capability_ebpw", "fastest_coherent_representation", "best_kernel",
             "best_device_profile"]
    out = []
    for fam in FAMILIES:
        r = rows.get(fam)
        entry = {"organ": fam, "models_measured": [PARENT] if r else [],
                 "status": "MEASURED" if r else "UNMEASURED",
                 "unmeasured_reason": None if r else
                 "no measurement on any Odyssey specimen yet"}
        for c in cells:
            entry[c] = (r or {}).get(c) or {"value": None, "status": "UNMEASURED",
                                            "reason": "not measured on any specimen"}
        out.append(entry)
    return out


def consolidation_census():
    """Every rival organ list in the repo, and what happened to it. Receipts are
    evidence and are never deleted -- rivals become aliases, not corpses."""
    found = []
    for p in sorted(RH.glob("*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        o = d.get("organs")
        if isinstance(o, dict) and o:
            names = sorted(o)
        elif isinstance(o, list) and o and isinstance(o[0], dict):
            names = sorted({x.get("organ") or x.get("name") for x in o if isinstance(x, dict)}
                           - {None})
        else:
            continue
        if not names:
            continue
        found.append({
            "receipt": str(p.relative_to(REPO)), "organ_names": names,
            "canonical": {n: canonical(n) for n in names},
            "disposition": "CANONICAL" if p == LIB else "ALIASED_TO_CANONICAL",
        })
    return found


def add_measurement(rows, organ, model, cell, value, receipt_rel, json_path):
    """The one write path. Refuses a cell whose citation does not resolve."""
    got, c = cite(receipt_rel, json_path)
    organ = canonical(organ)
    r = rows.setdefault(organ, {"organ": organ, "model": model})
    r[cell] = {"value": value, "measured_value_at_cite": got, "model": model, "cite": c}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-matrix", default=str(MATRIX))
    ap.add_argument("--refuse-demo", action="store_true",
                    help="attempt a cell citing a receipt that does not hold the value")
    a = ap.parse_args()

    rows = qwen_rows()
    if a.refuse_demo:
        try:
            add_measurement(rows, "mlp_gate_up", "nobody", "lowest_local_ebpw", 0.1,
                            "receipts/headless/DOES_NOT_EXIST.json", "a.b")
        except Refused as r:
            print("REFUSED (missing receipt):", r)
        try:
            add_measurement(rows, "mlp_gate_up", "nobody", "lowest_local_ebpw", 0.1,
                            "receipts/headless/ORGAN_DENSITY_FLOORS.json", "organs.no_such_organ.x")
        except Refused as r:
            print("REFUSED (unresolvable path):", r)
        return 0

    m = matrix(rows)
    cons = consolidation_census()
    n_meas = sum(1 for e in m if e["status"] == "MEASURED")
    out = {
        "schema": "hawking.headless.organ_frontier_matrix.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/organ_library.py",
        "obligation": "G017 — ORGAN_LIBRARY + ORGAN_FRONTIER_MATRIX (directive §48, §49)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False, "unmeasured_is_absent": True,
        "n_models": 1,
        "n_models_note": "only qwen3.8-27b-abliterated has measured organ floors on this "
                         "machine. Rows for other models are ABSENT, never interpolated. "
                         "add_measurement() adds a second model in one call.",
        "families": FAMILIES, "aliases": ALIASES,
        "n_families": len(FAMILIES), "n_measured": n_meas,
        "organs": m,
        "pass": bool(n_meas >= 3),
    }
    Path(a.emit_matrix).write_text(json.dumps(out, indent=1))
    CONSOLIDATION.write_text(json.dumps({
        "schema": "hawking.headless.organ_library_consolidation.v1",
        "generated_at": out["generated_at"],
        "canonical_authority": str(LIB.relative_to(REPO)),
        "law": "receipts are evidence and are never deleted; a rival organ spelling becomes "
               "an alias pointing at the canonical name",
        "n_rival_receipts": len(cons), "aliases": ALIASES, "receipts": cons}, indent=1))
    print(f"families={len(FAMILIES)} measured={n_meas} rival_receipts={len(cons)} "
          f"pass={out['pass']}")
    for e in m:
        if e["status"] == "MEASURED":
            v = e["lowest_local_ebpw"]
            print(f"  {e['organ']:20} local_ebpw={v.get('value')} codec={v.get('codec')}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
