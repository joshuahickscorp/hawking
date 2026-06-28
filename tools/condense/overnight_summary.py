#!/usr/bin/env python3.12
"""Assemble a clean morning RESULTS.md from the overnight condensation runs.
Parses JSON lines (from ppl_bench.py and doctor_qat.py) out of any logs/jsonl files
given on the command line, builds the recovery table, and writes
reports/condense/OVERNIGHT_RESULTS.md.

Usage: overnight_summary.py <log-or-jsonl> [more...]
"""
import sys, json, datetime, pathlib

ppl_rows = {}   # label -> ppl  (from ppl_bench.py)
doc_rows = []   # {bits, steps, ptq_ppl, qat_ppl, recovery_pct, kd?}  (from doctor_qat.py)

for path in sys.argv[1:]:
    try:
        lines = open(path, errors="ignore").read().splitlines()
    except OSError:
        continue
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "qat_ppl" in d:
            doc_rows.append(d)
        elif "ppl" in d and "label" in d:
            ppl_rows[d["label"]] = d["ppl"]

f16 = ppl_rows.get("f16")
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
L = ["# Hawking Condense — overnight results", "", f"_assembled {stamp}_", ""]


def vs(p):
    return f"+{(p/f16-1)*100:.0f}%" if (f16 and p) else "—"


# real perplexity table (actual inference, Qwen2.5-0.5B)
L += ["## Real perplexity — condensation + recovery (actual inference)", "",
      "| variant | ppl | vs f16 |", "|---|--:|--:|"]
order = ["f16", "tq3", "tq3+outl", "tq3+full", "tq2", "tq2+outl", "tq2+full",
         "tq2+doctor150+STRAND", "tq2+doctor+STRAND", "tq2+doctor", "tq3+doctor"]
seen = set()
for k in order + [k for k in ppl_rows if k not in order]:
    if k in ppl_rows and k not in seen:
        seen.add(k)
        L.append(f"| {k} | {ppl_rows[k]:.2f} | {vs(ppl_rows[k])} |")

# doctor recovery summary
if doc_rows:
    L += ["", "## The doctor (QAT/KD) — recovery", "",
          "| bits | steps | PTQ ppl | +doctor ppl | recovery |", "|--:|--:|--:|--:|--:|"]
    for d in doc_rows:
        L.append(f"| {d.get('bits')} | {d.get('steps')} | {d.get('ptq_ppl',0):,.0f} | "
                 f"{d.get('qat_ppl',0):,.1f} | {d.get('recovery_pct',0):.2f}% |")

# headline
best = ppl_rows.get("tq2+doctor150+STRAND") or ppl_rows.get("tq2+doctor+STRAND")
L += ["", "## Headline", ""]
if best and f16:
    L.append(f"> **2-bit condensed + the doctor → ppl {best:.1f}** vs f16 {f16:.1f} "
             f"(PTQ alone collapsed to {ppl_rows.get('tq2', float('nan')):.0f}). "
             f"Condensation + recovery = small AND ~1:1.")
else:
    L.append("> (recovery run still in flight — rerun this summary once the cron/insurance run completes)")
L += ["", "See `reports/sota-compare/condense_scorecard.md` for compression + the RAM cliff.", ""]

out = pathlib.Path("reports/condense/OVERNIGHT_RESULTS.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(L) + "\n")
print(f"wrote {out}")
print("\n".join(L))
