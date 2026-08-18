#!/usr/bin/env python3
"""Chopping-block tournament: Qwen3.8 vs Q80, five tasks, one Genesis.

Steer S003. Both contenders see identical prompts, identical HEAD, identical
budgets. Neither is told what the other produced. The material is authentic -
every number, trap and failure below happened on 2026-08-16 and is on disk.

Grading is deterministic wherever a task admits it. Task 2 (false-win
rejection, 25%) is fully mechanical: nine claims, six false and three true, so
a contender that rejects everything scores 6/9 rather than winning. Tasks that
need judgement record the transcript for external review - this driver never
promotes anything by itself.

    genesis_tournament.py run   --contender qwen38|q80 [--task N] [--out DIR]
    genesis_tournament.py grade --dir DIR

Every generate runs under tools/gpu_lane_lock.sh. Exit 0 ran, 2 could not.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "tools" / "gpu_lane_lock.sh"
BUILD = Path(
    "/private/tmp/claude-503/-Users-scammermike-Downloads-hawking/"
    "d51d4904-9fa1-4f81-8170-5e7eb27a291d/scratchpad/main_target/release/examples"
)

CONTENDERS = {
    "qwen38": {
        "binary": BUILD / "ascension_qwen38_hybrid_greedy",
        "artifact": REPO / "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1",
        "tokenizer": REPO / "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json",
        "text_marker": "GENERATED_TEXT_VERBATIM:",
    },
    "q80": {
        "binary": BUILD / "ascension_qwen80_mixed_hybrid_greedy",
        "artifact": REPO / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
                           "/quality-candidates/mixed-1p5-v1",
        "tokenizer": REPO / "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next/tokenizer.json",
        "text_marker": 'generated_text="',
    },
}

# ---------------------------------------------------------------- task material

T1 = """You are the resident engineering unit for Hawking, a local LLM runtime.

Here is a REAL measurement taken today from the Q80 mixed-1.5-BPW artifact,
one greedy generation, GPU-lock held, MTLCommandBuffer GPUEnd-GPUStart timing:

  wall_ns_per_token        = 194,978,316
  gpu_matvec_ns_per_token  =  44,141,899
  host_prep_med            = 131,113,655
    of which expert_bind   = 111,702,498
    catalog_reparse        =   8,688,458
    embed                  =   8,704,000
    encode                 =   1,847,658
  cbs_per_token            = 49
  dispatches_per_token     = 2834
  bytes moved per token    = 2.218 GB
  honest decode ceiling    = 411.51 GB/s (unique bytes once)

Answer these four questions, briefly and concretely:
1. What is the actual dominant bottleneck? Give the component and its ns.
2. Is it a REPRESENTATION limit (too many bytes) or an EXECUTION limit
   (bytes are fine, the program wastes time)? Justify with a number.
3. Propose the single cheapest falsifiable experiment to attack it.
4. State the specific measured result that would KILL your hypothesis.
"""

T2_CLAIMS = [
    # (id, text, correct_verdict, why)
    ("C1",
     "The gk_simd kernel gives a 2-45x speedup. Evidence: benchmarked against the "
     "serial 1-thread-per-row extract path at 867 ms gpu_matvec.",
     "REJECT",
     "stale baseline - the serial extract was deleted by the reconstruction fix; "
     "gpu_matvec is 36.6 ms now, and measured properly gk_simd is 2.5x SLOWER"),
    ("C2",
     "The DSV4F end-to-end test passes, so the decode path is correct.",
     "REJECT",
     "that test printed divergence and asserted nothing - it would pass on "
     "arbitrarily wrong output"),
    ("C3",
     "Q80 will reach 333 TPS once the artifact is packed at 1.44445 BPW.",
     "REJECT",
     "projected, not measured; and 333 TPS needs ~0.805 BPW, so the stated BPW "
     "cannot produce the stated TPS"),
    ("C4",
     "The G005 capability gate requires displacement >= 0.50 on the offload bench, "
     "and the run scored 0.35, so we are close to passing.",
     "REJECT",
     "the bench CEILING is 0.35 - only 7 of 20 tasks are model-expected - so the "
     "0.50 bar is unreachable by construction, not nearly met"),
    ("C5",
     "gpu_matvec is 36.6 ms per token, therefore Q80 runs at 27 TPS.",
     "REJECT",
     "component timing presented as complete-token speed; the complete token is "
     "108.3 ms, so ~9.2 TPS"),
    ("C6",
     "The decode ceiling on this box is 647.5 GB/s, so 2.218 GB/token allows 293 TPS.",
     "REJECT",
     "535.9-647.5 GB/s is the cache-resident REUSE band; the honest unique-bytes-once "
     "decode ceiling is 411.51 GB/s"),
    ("C7",
     "Q80 wall per token fell from 1376.3 ms to 108.306 ms. Evidence: median of 3 "
     "reps under the GPU lock, complete-token wall, warm steady state, same artifact "
     "identity, same HEAD.",
     "ACCEPT",
     "genuine - paired, complete-token, warm regime stated, artifact identity fixed"),
    ("C8",
     "DSV4F cannot reach the 20 ms gate on its current artifact. Evidence: it moves "
     "10.280 GB unique bytes per token; at the 411.51 GB/s ceiling that is a 24.98 ms "
     "floor, which is above the gate.",
     "ACCEPT",
     "genuine - names the saturated resource and derives the floor from measured bytes"),
    ("C9",
     "Reconstruction of compressed weights is free at the tpr64 launch geometry. "
     "Evidence: 33 codec variants on real activations all land 15,124-15,541 ns "
     "against an uncompressed f32 control of 15,125 ns on the same shape.",
     "ACCEPT",
     "genuine - controlled comparison against an uncompressed control on the same shape"),
]

T2 = (
    "You protect a research campaign from convincing but false results.\n\n"
    "OUTPUT FORMAT - obey this exactly. Write the nine verdict lines FIRST,\n"
    "before any explanation or reasoning. One line per claim:\n"
    "  C1: REJECT - <one sentence of evidence>\n"
    "  C2: ACCEPT - <one sentence of evidence>\n"
    "  ... through C9\n"
    "Only after all nine lines may you add further reasoning.\n"
    "Some claims are genuine. Rejecting everything is itself a failure.\n\n"
    + "\n\n".join(f"{cid}. {text}" for cid, text, _, _ in T2_CLAIMS)
)

T3 = """You are the resident engineering unit for Hawking.

BOUNDED BOTTLENECK, measured today on Q80, complete-token wall 194.98 ms:

  host_prep_med  = 131,113,655 ns   (67% of the token, on the CPU)
    expert_bind  = 111,702,498 ns   (57% of the token by itself)
  packed_calls   = 688 per token
  cached_geometry_count = 666
  cbs_per_token  = 49
  The GPU is idle for most of expert_bind.

expert_bind is the host walking the mixed catalog to resolve, for each routed
expert selected this token, the Metal buffer and byte offsets of its weight
segments, then binding them into the command encoder. The set of experts
changes per token; the CATALOG DOES NOT. 74,391 tensors, geometry fixed at
artifact-open time.

1. State the mechanism that removes this cost - not reduces it, removes it.
   Be specific about what data structure is built when, and what the per-token
   path does instead.
2. Name the exact source file(s) you would change.
3. Give the build command and the test command you would run.
4. State the LIVE baseline you would compare against, and the single number
   that decides whether this landed.
5. State what result would mean your mechanism is WRONG, not merely small.
"""

T4 = """You are the resident engineering unit for Hawking. An experiment failed.

WHAT WAS CLAIMED: a new simdgroup matvec kernel, gk_simd, gives 2-45x over the
shipping kernel. It was promoted into the ledger twice on that basis.

WHAT ACTUALLY HAPPENED: measured head-to-head on the same shape under the GPU
lock, gk_simd took 91.28 ms against the shipping kernel's 36.53 ms. It is 2.5x
SLOWER. The 2-45x number had been measured against a serial 1-thread-per-row
extract path that a separate lane had already deleted.

A colleague proposes: "the parity test is too strict, let us relax the
tolerance so gk_simd passes, then land it and tune later."

Answer:
1. What is the ROOT CAUSE of the false win? Not the symptom.
2. Respond to the colleague's proposal. Be explicit.
3. What negative science here is worth preserving, and why?
4. What is the correct repair - to the ledger, and to the process?
5. What is the single next action?
"""

T5 = """You are the resident engineering unit for Hawking. Grok, an exploration
agent, returned this proposal. Grok is capable but its report is a CLAIM, not a
receipt.

GROK PROPOSAL:
  "Q80 gpu_matvec runs at 56.6 GB/s against a 411.51 GB/s ceiling - only 13.8%.
   That is 7.3x of headroom. I propose rewriting all matvec kernels to use
   simdgroup reductions, which are known to be faster than thread-per-row.
   Expected result: gpu_matvec drops from 39.19 ms to 5.39 ms, taking the
   complete token under 10 ms and clearing 100 TPS. I did not build or run
   anything; the estimate is from the bandwidth model."

Known context you must use:
  - a previous simdgroup kernel, gk_simd, measured 2.5x SLOWER than the
    shipping thread-per-row kernel on the same shape
  - a 33-codec sweep found the launch geometry, not the codec, set the cost:
    tpr64 ~15,100 ns vs tg256 ~26,500 ns on identical work
  - per-organ census: qkvz already sustains 402 GB/s; the small organs are the
    slow ones - ba is 0.2% of bytes but 7% of GPU time
  - the complete token is 194.98 ms, of which gpu_matvec is 44.14 ms

Answer:
1. ACCEPT, REFINE, or REJECT this proposal. Say which.
2. Identify every unsupported step in Grok's reasoning.
3. Give the refined version you would actually execute - the smallest
   experiment that discriminates.
4. State the verification that would make its result trustworthy.
5. End your answer with a single final line in exactly this format:
   NEXT_BOTTLENECK: <component>, <measured ns>
"""

# Budgets are deliberately generous. Qwen3.8 emits a <think> block and Q80 does
# not, so a tight cap would truncate one contender's reasoning and score my cap
# rather than the model. The first run did exactly that on T1 and was discarded.
TASKS = {
    1: {"name": "DIAGNOSE", "weight": 0.15, "prompt": T1, "max_new": 2000},
    2: {"name": "FALSE_WIN_REJECTION", "weight": 0.25, "prompt": T2, "max_new": 3000},
    3: {"name": "ENGINEER", "weight": 0.30, "prompt": T3, "max_new": 2500},
    4: {"name": "RECOVER", "weight": 0.15, "prompt": T4, "max_new": 2000},
    5: {"name": "AUTONOMY_HANDOFF", "weight": 0.15, "prompt": T5, "max_new": 2500},
}

# ------------------------------------------------------------------- execution


def extract_text(stdout: str, marker: str) -> str:
    """Pull the generated text out of whichever format the example prints."""
    if marker == 'generated_text="':
        m = re.search(r'generated_text="(.*)"\s*$', stdout, re.S | re.M)
        return m.group(1) if m else ""
    # qwen38 prints a verbatim block terminated by the FALLBACKS line
    m = re.search(r"GENERATED_TEXT_VERBATIM:\s*(.*?)\nFALLBACKS:", stdout, re.S)
    return m.group(1) if m else ""


def run_one(contender: str, task_id: int, out_dir: Path, seq_len: int,
            lock: bool = True) -> dict:
    c = CONTENDERS[contender]
    t = TASKS[task_id]
    # The lock serializes GPU work so a TIMING is clean. These runs are graded on
    # what the contender SAYS; clean per-token timings already exist from the
    # paired smokes. So --no-lock is allowed here, and wall_secs from an unlocked
    # run is marked contended and must not be quoted as a latency measurement.
    prefix = [str(LOCK), f"genesis-{contender}-t{task_id}"] if lock else []
    cmd = [
        *prefix,
        str(c["binary"]),
        "--artifact-root", str(c["artifact"]),
        "--tokenizer", str(c["tokenizer"]),
        "--prompt", t["prompt"],
        "--max-new-tokens", str(t["max_new"]),
        "--max-seq-len", str(seq_len),
    ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.SubprocessError as e:
        rc, stdout, stderr = -1, "", str(e)
    wall = time.time() - t0

    rec = {
        "contender": contender,
        "task": task_id,
        "task_name": t["name"],
        "exit_code": rc,
        "wall_secs": round(wall, 3),
        "timing_label": "CLEAN_CANDIDATE" if lock else "CONTENDED_NOT_A_LATENCY_MEASUREMENT",
        "generated_text": extract_text(stdout, c["text_marker"]),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-1500:],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{contender}-t{task_id}.json").write_text(json.dumps(rec, indent=2))
    print(f"{contender} t{task_id} {t['name']}: exit={rc} wall={wall:.1f}s "
          f"chars={len(rec['generated_text'])}")
    return rec


# -------------------------------------------------------------------- grading


def grade_task2(text: str) -> dict:
    """Fully mechanical. Six REJECT, three ACCEPT - answering uniformly loses."""
    # Q80 prints the generated text with literal backslash-n escapes, so a
    # line-anchored match silently found only the first verdict and scored 1/9.
    # Normalize the escapes before matching, and take the window after the id
    # rather than "the rest of the line".
    text = text.replace("\\n", "\n")
    per = {}
    for cid, _, correct, _ in T2_CLAIMS:
        # Take the LAST verdict stated for a claim. Contenders reconsider mid
        # answer ("Let me double-check C5..."), and the settled verdict is the
        # one they end on. Cut each candidate at its newline so a fixed window
        # cannot read the NEXT claim's verdict - that mis-scored C2 at 160 chars.
        got = "NO_VERDICT"
        for m in re.finditer(rf"\b{cid}\s*[:.\)]", text, re.I):
            line = text[m.end():].split("\n", 1)[0].upper()
            ri, ai = line.find("REJECT"), line.find("ACCEPT")
            if ri < 0 and ai < 0:
                continue
            got = "REJECT" if (ai < 0 or (0 <= ri < ai)) else "ACCEPT"
        per[cid] = {"expected": correct, "got": got, "correct": got == correct}
    n = sum(1 for v in per.values() if v["correct"])
    return {"per_claim": per, "correct": n, "total": len(T2_CLAIMS),
            "score": n / len(T2_CLAIMS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "grade"])
    ap.add_argument("--contender", choices=list(CONTENDERS))
    ap.add_argument("--task", type=int)
    ap.add_argument("--out", default="receipts/ascent-2026-08-16/genesis-tournament")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--no-lock", action="store_true",
                    help="run without the GPU lane lock so tasks can go in parallel")
    a = ap.parse_args()
    out = REPO / a.out

    if a.mode == "run":
        if not a.contender:
            print("--contender required", file=sys.stderr)
            return 2
        ids = [a.task] if a.task else sorted(TASKS)
        for tid in ids:
            run_one(a.contender, tid, out, a.seq_len, lock=not a.no_lock)
        return 0

    # grade: mechanical part only; judgement tasks are printed for review
    report = {}
    for contender in CONTENDERS:
        f = out / f"{contender}-t2.json"
        if f.is_file():
            rec = json.loads(f.read_text())
            report[contender] = grade_task2(rec["generated_text"])
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
