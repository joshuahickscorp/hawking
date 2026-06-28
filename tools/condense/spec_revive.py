#!/usr/bin/env python3.12
"""spec_revive.py — revive speculative decoding ON TOP of a condensed model (the latency lane).

The speculate/ subsystem is ~80% built (eagle5 + retrieval/user-ngram drafts + governor + verifier
+ acceptance harness). It was parked on three blockers: (1) batched verify not bit-lossless at
near-ties [a hardware-independent CORRECTNESS fix], (2) eagle5 accept ~1.1% because the head was
trained against a DIFFERENT distribution than the served model, (3) on 18GB the draft+verifier+KV
contended for RAM so spec REGRESSED. Blockers 2 and 3 are dissolved by the condense baseline: the
condensed model FITS with RAM headroom for the draft (kills 3), and the condense calib + doctor-KD
data + the condensed model's OWN output distribution is exactly the capture-retrain signal (kills 2).

This orchestrates the revival in order, gating each phase. It is a runner over REAL targets
(eagle5_forward_dump.py, eagle5_train.py, the hawking-spec-acceptance-measure bin, the spec parity
tests); heavy cargo/training steps are Studio-tier and are PRINTED + run in order. --plan dry-runs
the whole sequence here without building. Density (RAM) and spec (latency) are ORTHOGONAL and stack
multiplicatively: a model that both FITS at low bpw AND decodes 2-3x faster at identical greedy output.

Usage:
  spec_revive.py <condensed.tq-or-model-dir> <label> [--head <eagle5.safetensors>] [--accept-gate 0.40]
  spec_revive.py --plan <label>     # print the gated sequence, run nothing heavy
"""
import sys, os, json, subprocess, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
ACCEPT_GATE = float(os.environ.get("SPEC_ACCEPT_GATE", "0.40"))   # below this, self-spec is not worth it
OUT_DIR = "reports/condense"

# The gated sequence (tool, args-template, note, studio_only). Each step's KILL is explicit.
PHASES = [
    ("lossless-verify gate",
     ["cargo", "test", "-p", "hawking-core", "eagle5_spec_parity", "user_draft_parity_e2e",
      "event_horizon_parity_prop", "--release"],
     "batched verify must bit-match the greedy kernel at near-ties; if RED, fix the tie-break "
     "(deterministic argmax) BEFORE anything else — spec is only allowed if it is exactly lossless.",
     True),
    ("capture condensed distribution",
     ["python3.12", "tools/eagle5_forward_dump.py", "{model}", "--out", OUT_DIR + "/{label}_speccap.safetensors"],
     "dump the CONDENSED model's hidden states + top-k logits as the capture-retrain target "
     "(this is the new baseline the laptop never had — same distribution the head must predict).",
     True),
    ("capture-retrain eagle5 head",
     ["python3.12", "tools/training/eagle5_train.py", "--capture", OUT_DIR + "/{label}_speccap.safetensors",
      "--save", OUT_DIR + "/{label}_eagle5.safetensors"],
     "retrain the draft head against the condensed distribution (fixes the 1.1% accept = fp16-head-vs-"
     "low-bit-runtime mismatch). KILL: if retrained accept stays < gate, self-spec is dead for this model.",
     True),
    ("acceptance measure",
     ["cargo", "run", "--release", "-p", "hawking", "--bin", "hawking-spec-acceptance-measure", "--",
      "--weights", OUT_DIR + "/{label}_eagle5.safetensors", "--steps", "256"],
     "measure mean per-prompt acceptance on the condensed model. GATE: mean accept >= "
     + f"{ACCEPT_GATE:.0%} or the governor will not net a tps win.",
     True),
    ("governor bench (exact-match gate)",
     ["cargo", "run", "--release", "-p", "hawking", "--", "bench", "--spec", "--exact-match", "--ppl"],
     "end-to-end tps with the governor choosing among eagle5 / retrieval / user-ngram drafts; "
     "the exact-match gate guarantees identical greedy output (the lossless contract). Report tps x-factor.",
     True),
]


def _plan(label):
    print(f"# spec_revive --plan {label}: the gated revival sequence (heavy steps run on the Studio)\n")
    for i, (name, args, note, studio) in enumerate(PHASES, 1):
        cmd = " ".join(a.replace("{label}", label).replace("{model}", f"<{label}.tq>") for a in args)
        print(f"  P{i}. {name}{'  [STUDIO]' if studio else ''}")
        print(f"      $ {cmd}")
        print(f"      -> {note}\n")
    print(f"# accept gate = {ACCEPT_GATE:.0%}; density (RAM) x spec (latency) stack multiplicatively.")
    print("# KILL (whole lane): lossless-verify cannot be made bit-exact, OR retrained accept < gate "
          "on the 7B+ substrate.")


def run(model, label, head=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    rec = {"label": label, "model": model, "phases": [], "accept_gate": ACCEPT_GATE}
    for name, args, note, _ in PHASES:
        cmd = [a.replace("{label}", label).replace("{model}", model) for a in args]
        print(f"[spec] {name}: {' '.join(cmd)}", file=sys.stderr)
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode == 0
        rec["phases"].append({"phase": name, "rc": r.returncode, "ok": ok,
                              "tail": (r.stdout or r.stderr)[-300:]})
        if name == "lossless-verify gate" and not ok:
            rec["verdict"] = "HALT: verify not lossless — fix the near-tie tie-break first"
            break
        if name == "acceptance measure":
            acc = _parse_accept(r.stdout)
            rec["accept_rate"] = acc
            if acc is not None and acc < ACCEPT_GATE:
                rec["verdict"] = f"KILL: accept {acc:.1%} < gate {ACCEPT_GATE:.0%} — self-spec not worth it"
                break
    rec.setdefault("verdict", "spec lane complete")
    out = f"{OUT_DIR}/{label}_spec.json"
    json.dump(rec, open(out, "w"), indent=2)
    print(f"[spec] {rec['verdict']} -> {out}", file=sys.stderr)
    return 0 if rec["verdict"].startswith(("spec lane", "KILL")) else 1


def _parse_accept(stdout):
    import re
    m = re.search(r"(?:mean|aggregate)[^0-9]*([0-9.]+)\s*%", stdout or "", re.I)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"accept[^0-9]*([01]\.[0-9]+)", stdout or "", re.I)
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plan":
        _plan(sys.argv[2] if len(sys.argv) > 2 else "MODEL")
    elif len(sys.argv) >= 3:
        sys.exit(run(sys.argv[1], sys.argv[2]))
    else:
        print(__doc__)
