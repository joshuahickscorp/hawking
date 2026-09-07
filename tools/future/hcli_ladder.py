#!/usr/bin/env python3
"""Drive the resident DOWN a BPW ladder through HCLI (G012, G013 / S005).

The resident NEVER reports a measurement. It emits a spec string and a plan
line; the evaluator executes the spec and the numbers come from the machine.
That closes the fabrication hole recorded in this campaign -- a scorer that
rewards measurement-shaped prose selects for invented measurements, and
tools/drive_resident.py scores on the literal word "measured". Here the model
cannot fabricate a result because it is never asked for one.

S005: when a rung cannot be implemented inside the grammar, the resident must
PLAN that rung instead of stopping. So each round asks for both.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W)); sys.path.insert(0, str(W / "tools" / "future"))
import gravity_outlier_eval as G                                   # noqa: E402

PROFILE = W / "ascension_envelope.hawking.json"
# S008 section 7. A STABLE prefix: byte-identical on every call so a prefix cache
# can reuse it, and first in the prompt so nothing shifts its token positions.
# 129 resident tokens against a ~667-token usable budget -- deliberately small,
# because a manifesto here would displace the evidence it is meant to govern.
CONSTITUTION = (W / "hcli" / "CONSTITUTION.txt").read_text().rstrip() + "\n\n"
LOG = W / "receipts/future/O003_HCLI_LADDER.jsonl"
SPEC_RE = re.compile(r"^\s*SPEC:\s*(\S+)\s*$", re.M | re.I)
PLAN_RE = re.compile(r"^\s*PLAN:\s*(.+)$", re.M | re.I)


def prompt_for(evidence: list[dict], rejected: list[str] | None = None) -> str:
    lines = [f"  {e['spec']:20} {e['ebpw']:.4f} {'PASS' if e['pass'] else 'FAIL'}"
             for e in evidence]
    passing = [e for e in evidence if e["pass"]]
    best = min(passing, key=lambda e: e["ebpw"], default=None)
    return (
        CONSTITUTION
        + "O003 Kimi-VL MoE. 26 MoE layers, 64 experts, top-6. Expert organ = 90.2% of weights.\n"
        "Expert tensor shapes, which bound the group size:\n"
        "  gate_proj, up_proj  [64, 1408, 2048]  last dim 2048 -> group up to 2048\n"
        "  down_proj           [64, 2048, 1408]  last dim 1408 = 2^7*11 -> group CAPPED at 128\n"
        "  A group must DIVIDE the last dim. down_proj admits no group above 128, so a larger\n"
        "  group only helps the 2/3 of expert weights in gate/up.\n"
        "Complete EBPW measured, capability gated against the bf16 source:\n"
        + "\n".join(lines)
        + (f"\n\nLowest PASSING: {best['spec']} at {best['ebpw']:.4f}. Target is <=1.0.\n"
           if best else
           "\n\nNOTHING above passes the capability gate. Target is <=1.0. Propose a rung that "
           "trades bytes for capability, not the reverse.\n")
        + "Grammar you can execute (group is 32, 64 or 128):\n"
        "  q<bits>-g<group>-experts   bits 2..8, plain affine\n"
        "  outlier<frac>-g<group>     2-bit base + top-|w| frac kept at full precision\n"
        "  binary-g<group>            1 bit/weight, scale = mean|W| per group\n"
        "  binarypercal-g<group>      1 bit, scale fitted to per-expert activations (best at 1 bit)\n"
        "  binarypercal<frac>-g<group>  the same, plus an outlier channel\n"
        "COST, so you can compute your own proposal before making it:\n"
        "  complete EBPW = (14394851328 * bits_per_weight / 8 + 885541120) * 8 / 16407657776\n"
        "  bits_per_weight: affine = bits + 32/group;  binary = 1 + 16/group;\n"
        "                   an outlier channel ADDS frac * 32 bits per weight.\n"
        "  So a larger outlier frac RAISES EBPW. A larger group LOWERS it.\n"
        "Propose a rung whose computed EBPW is LOWER than the lowest one above.\n"
        "Do not repeat any spec above.\n"
        + ("ALREADY REJECTED this session, do not propose again:\n"
           + "".join(f"  {r}\n" for r in (rejected or [])) if rejected else "")
        + "\n"
        "Reply with EXACTLY two lines and nothing else:\n"
        "SPEC: <one spec from the grammar>\n"
        "PLAN: <one sentence: a mechanism OUTSIDE this grammar that could reach 1.0 EBPW, "
        "and which measured number above it must beat>\n"
    )


def ask(prompt: str, timeout: int = 1800) -> str:
    pf = W / "_ladder_prompt.txt"
    pf.write_text(prompt, encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-u", "-m", "hcli", "1", "--task-file", str(pf),
         "--model", str(PROFILE), "--max-cycles", "2"],
        cwd=str(W), capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


class _Skip(Exception):
    """Proposal rejected on arithmetic before any GPU time is spent."""


class Cand:
    def __init__(self, spec): self.spec, self.specimen = spec, "O003"


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    evidence = [
        {"spec": "q4-g64-experts", "ebpw": 4.3797, "pass": True},
        {"spec": "binary-g128", "ebpw": 1.4188, "pass": False},
        {"spec": "binarypercal-g128", "ebpw": 1.4188, "pass": False},
        {"spec": "binarypercal-g64", "ebpw": 1.5284, "pass": False},
        {"spec": "binarypercal-g256", "ebpw": 1.3822, "pass": False},
        {"spec": "binarypercal-g512", "ebpw": 1.3639, "pass": False},
        {"spec": "q3-g64-experts", "ebpw": 3.5024, "pass": True},
        {"spec": "q2-g64-experts", "ebpw": 2.6251, "pass": False},
        {"spec": "q2-g128-experts", "ebpw": 2.4058, "pass": False},
        {"spec": "outlier0.005-g128", "ebpw": 2.5419, "pass": False},
        {"spec": "outlier0.01-g128", "ebpw": 2.6815, "pass": False},
        {"spec": "outlier0.02-g128", "ebpw": 2.9597, "pass": False},
    ]
    seen = {e["spec"] for e in evidence}
    rejected: list[str] = []
    for rnd in range(1, rounds + 1):
        t0 = time.time()
        raw = ask(prompt_for(evidence, rejected))
        m, pm = SPEC_RE.search(raw), PLAN_RE.search(raw)
        rec = {"round": rnd, "wall_s": round(time.time() - t0, 1),
               "spec_proposed": m.group(1) if m else None,
               "plan": (pm.group(1).strip()[:400] if pm else None),
               "raw_tail": raw[-500:]}
        if not m:
            rec["outcome"] = "NO_SPEC_EMITTED"
        elif m.group(1) in seen:
            rec["outcome"] = "REPEATED_A_SPEC_ALREADY_ON_THE_TABLE"
            rejected.append(f"{m.group(1)} -> already measured, do not repeat.")
        else:
            try:
                plan = G.parse_spec(m.group(1))
                rec["parsed"] = plan
                pred = G.predict_ebpw(m.group(1))
                rec["predicted_ebpw"] = round(pred, 4)
                floor = min(e["ebpw"] for e in evidence)
                if pred >= floor:
                    rec["outcome"] = (f"WRONG_DIRECTION: predicted {pred:.4f} >= "
                                      f"current lowest {floor:.4f}, not executed")
                    rejected.append(f"{m.group(1)} -> {pred:.4f} EBPW, which is HIGHER "
                                    f"than {floor:.4f}. An outlier channel ADDS bytes.")
                    raise _Skip()
                r = G.evaluate(Cand(m.group(1)))
                rec.update({"outcome": "EXECUTED", "ebpw": r["complete_ebpw"],
                            "capability_ok": r["capability_ok"],
                            "ppl": r["capability"]["ppl"],
                            "median_4gram": r["capability"]["median_4gram_repeat"],
                            "magnitude_ratio": r["magnitude_ratio"],
                            "direction_similarity": r["direction_similarity"]})
                evidence.append({"spec": m.group(1), "ebpw": r["complete_ebpw"],
                                 "pass": r["capability_ok"]})
                seen.add(m.group(1))
            except _Skip:
                pass
            except ValueError as e:
                rec["outcome"] = f"REFUSED_UNRUNNABLE: {e}"
            except Exception as e:                       # keep failures as evidence
                rec["outcome"] = f"RUNNER_FAILURE: {type(e).__name__}: {e}"
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec.get(k) for k in
                          ("round", "spec_proposed", "outcome", "ebpw", "capability_ok",
                           "median_4gram", "wall_s")}), flush=True)
        if rec.get("plan"):
            print("   PLAN: " + rec["plan"][:220], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
