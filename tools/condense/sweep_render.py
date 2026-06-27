#!/usr/bin/env python3.12
"""Render reports/condense/ladder.jsonl → reports/condense/MATRIX.md.

Views (docs/plans/parameter_sweep_pipeline.md §5):
  1. HEADLINE — bit-floor vs scale: lowest eff-bpw recipe at 1:1 / win, per model.
  2. Stream A — quality recovery per (model, recipe): f16 ppl, healed Δ.
  3. Stream B — serve/cliff per artifact: .tq size, fits96, vs llama Q4_K.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ladder as L

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
JSONL = os.path.join(ROOT, "reports", "condense", "ladder.jsonl")
OUT = os.path.join(ROOT, "reports", "condense", "MATRIX.md")


def pct(x):
    return f"{x*100:+.1f}%" if isinstance(x, (int, float)) else "—"

def fnum(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"

def pof(name):
    return next((m["params_b"] for m in L.MODELS if m["name"] == name), 0)


def main():
    rows = [json.loads(l) for l in open(JSONL)] if os.path.exists(JSONL) else []
    A = [r for r in rows if r.get("stream") == "A"]
    B = [r for r in rows if r.get("stream") == "B"]
    models = sorted({r["model"] for r in rows}, key=pof)
    # per-model recipe→delta (only rows with a measured heal)
    q = {}
    for r in A:
        if r.get("heal_delta") is not None:
            q.setdefault(r["model"], []).append((r.get("eff_bpw"), r["recipe"], r["heal_delta"]))

    out = ["# Hawking Condense — Parameter Sweep Matrix", "",
           "_Auto-rendered from `reports/condense/ladder.jsonl`. Contract: lowest effective "
           "bpw at near-1:1 via the doctor = smallest artifact = highest tps. Recipes climb "
           "eff-bpw across single-bake (AWQ) + residual (STRAND b1+b2). Hypothesis: the floor "
           "descends as params rise._", ""]

    # 1 ── HEADLINE ─────────────────────────────────────────────────────────────────────
    out += ["## 1. Bit-floor vs scale (the headline)", "",
            "| model | params | 1:1 floor (≤+2%) | win floor (≤+8%) | Δ@win | serve-max bpw |",
            "|---|---|---|---|---|---|"]
    for n in models:
        m = next((x for x in L.MODELS if x["name"] == n), None)
        if not m:
            continue
        cells = sorted(q.get(n, []))
        f11 = next((f"{lbl} @{eb}bpw" for eb, lbl, d in cells if d <= L.NEAR_1to1), "—")
        win = next(((f"{lbl} @{eb}bpw", d) for eb, lbl, d in cells if d <= L.WIN), None)
        hr = L.serve_headroom_bpw(m["params_b"])
        out.append(f"| {n} | {m['params_b']}B | {f11} | {win[0] if win else '— (pending)'} | "
                   f"{pct(win[1]) if win else '—'} | {hr if hr else '1.0-edge'} |")
    out += ["", "_Win-floor trending down as params rise ⇒ the scaling hypothesis holds "
            "(bigger models compress harder). The 405B/671B frontier needs a single-bake "
            "1-bit floor (residual costs too much bpw to fit)._", ""]

    # 2 ── Stream A ─────────────────────────────────────────────────────────────────────
    out += ["## 2. Stream A — condense / quality recovery", "",
            "| model | recipe | eff bpw | f16 ppl | healed Δ | status |",
            "|---|---|---|---|---|---|"]
    for r in sorted(A, key=lambda r: (pof(r.get("model")), r.get("eff_bpw") or 0)):
        if r.get("status"):
            out.append(f"| {r['model']} | {r.get('recipe','')} | {fnum(r.get('eff_bpw'))} | — | — | {r['status']} |")
        else:
            st = "ok" if r.get("heal_ppl") else r.get("error", "")
            out.append(f"| {r['model']} | {r.get('recipe','')} | {fnum(r.get('eff_bpw'))} | "
                       f"{fnum(r.get('f16_ppl'))} | {pct(r.get('heal_delta'))} | {st} |")

    # 3 ── Stream B ─────────────────────────────────────────────────────────────────────
    out += ["", "## 3. Stream B — serve / RAM-cliff", "",
            "| model | recipe | .tq GB | fits 96GB | llama Q4_K GB | llama fits | cliff | tps |",
            "|---|---|---|---|---|---|---|---|"]
    seen = set()
    for r in sorted(B, key=lambda r: (pof(r.get("model")), r.get("eff_bpw") or 0)):
        k = (r["model"], r.get("recipe"))
        if k in seen:
            continue
        seen.add(k)
        out.append(f"| {r['model']} | {r.get('recipe','')} | {r.get('tq_gb')} | "
                   f"{'✅' if r.get('fits96') else '❌'} | {r.get('llama_q4k_gb')} | "
                   f"{'✅' if r.get('llama_fits') else '❌'} | {r.get('cliff','')} | "
                   f"{r.get('tps') or r.get('tps_raw') or '—'} |")

    out += ["", "## 4. The aggressive claim (the join)", "",
            "Where win-floor fits 96 GB and llama Q4_K does not: _Hawking @ floor-bpw matches/"
            "beats llama Q4_K quality AND runs where llama swaps/can't load._", ""]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(out))
    print(f"wrote {OUT} ({len(A)} quality rows, {len(B)} serve rows, {len(models)} models)")


if __name__ == "__main__":
    main()
