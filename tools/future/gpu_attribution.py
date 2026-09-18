#!/usr/bin/env python3.12
"""Who owns the GPU? Attribution before attribution-free claims.

EPOCH II G033. The directive's warning is exact: "DO NOT interpret visible GPU
saturation as proof that Hawking Accelerator owns the work." This box runs at
least three model servers, and only one of them is Hawking:

  * ascension_qwen38_resident  -- Hawking's sealed-3.14 resident
  * mlx_lm.server              -- MLX, a separate stack on :9999
  * ollama serve               -- Ollama, a separate stack on :11434

A saturated GPU is evidence that SOMETHING is computing. It is not evidence that
Hawking's PhysicalGraph chose and lowered the plan. This reports what is
measurable without elevated privileges and states plainly what is not, because a
GPU number with no owner is the same class of defect as a receipt whose producer
was never checked.

Deliberately NOT using sudo powermetrics: it is the only way to get per-process
GPU residency on macOS and it needs a password this must never ask for. The
honest output is "unattributed" rather than a guess.
"""
import json, subprocess, sys, time
from pathlib import Path

#: name fragment -> whether work by this process is Hawking Accelerator evidence
KNOWN = {
    "ascension_qwen38_resident": ("hawking", "sealed-3.14 resident (Rust, Metal)"),
    "hawkingd":                  ("hawking", "serve wrapper, not a compute owner"),
    "mlx_lm.server":             ("foreign", "MLX stack, separate runtime"),
    "mlx_lm":                    ("foreign", "MLX stack, separate runtime"),
    "ollama":                    ("foreign", "Ollama stack, separate runtime"),
    "comfy":                     ("foreign", "diffusion UI"),
    "python3.12 organ_gate":     ("hawking-cpu", "Odyssey CPU probe, no GPU"),
}


def _ps():
    out = subprocess.run(
        ["ps", "-eo", "pid,ppid,etime,%cpu,rss,command"],
        capture_output=True, text=True).stdout.splitlines()[1:]
    rows = []
    for line in out:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, etime, cpu, rss, cmd = parts
        rows.append({"pid": int(pid), "ppid": int(ppid), "etime": etime,
                     "cpu": float(cpu), "rss_gib": round(int(rss) / 1048576, 2),
                     "command": cmd})
    return rows


def _busy(pid, seconds=6.0, samples=3):
    """%CPU sampled over an interval. A process holding weights at ~0% is LOADED
    and IDLE, which is a different claim from 'using the GPU'."""
    vals = []
    for i in range(samples):
        r = subprocess.run(["ps", "-o", "%cpu=", "-p", str(pid)],
                           capture_output=True, text=True).stdout.strip()
        if r:
            vals.append(float(r))
        if i < samples - 1:
            time.sleep(seconds / (samples - 1))
    return vals


def attribute(window_note=""):
    rows = _ps()
    found = []
    for r in rows:
        for frag, (owner, what) in KNOWN.items():
            if frag in r["command"]:
                r = {**r, "owner": owner, "role": what}
                r["cpu_samples"] = _busy(r["pid"])
                r["verdict"] = (
                    "LOADED BUT IDLE -- holds weights, is not computing"
                    if max(r["cpu_samples"] or [0]) < 5 and r["rss_gib"] > 1
                    else "ACTIVE" if max(r["cpu_samples"] or [0]) >= 5
                    else "IDLE, HOLDS NOTHING")
                found.append(r)
                break
    hawking = [r for r in found if r["owner"] == "hawking"]
    foreign = [r for r in found if r["owner"] == "foreign"]
    return {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window_note": window_note,
        "hawking_processes": hawking,
        "foreign_model_servers": foreign,
        "MEASURABLE_WITHOUT_PRIVILEGE": [
            "which model servers exist and who owns them",
            "resident memory held by each",
            "%CPU over an interval, which separates LOADED-AND-IDLE from ACTIVE",
            "whether a foreign server has any model loaded or any client attached",
        ],
        "NOT_MEASURABLE_WITHOUT_PRIVILEGE": [
            "per-process GPU residency and utilization (sudo powermetrics only)",
            "GPU power draw",
            "dispatch counts and command-buffer occupancy from outside the process",
        ],
        "THE_RULE": (
            "A saturated GPU proves SOMETHING is computing. Hawking Accelerator "
            "evidence requires naming the owning process, its runtime, its "
            "representation, and that the plan was SELECTED AND LOWERED by "
            "Hawking. Absent that, report UNATTRIBUTED."),
    }


if __name__ == "__main__":
    rec = attribute(" ".join(sys.argv[1:]))
    for r in rec["hawking_processes"]:
        print(f"  HAWKING  pid {r['pid']:6d} {r['rss_gib']:7.2f} GiB  "
              f"cpu {r['cpu_samples']}  {r['verdict']}")
    for r in rec["foreign_model_servers"]:
        print(f"  FOREIGN  pid {r['pid']:6d} {r['rss_gib']:7.2f} GiB  "
              f"cpu {r['cpu_samples']}  {r['verdict']}  <- NOT Hawking")
    out = Path(__file__).resolve().parents[2] / "receipts/future/GPU_ATTRIBUTION.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}")
