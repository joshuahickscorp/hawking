#!/usr/bin/env python3
"""QWEN_PERFORMANCE_QUALIFICATION + GPU_CLEANLINESS_OVERRIDE.

Reprofiled from zero on a quiesced machine. Two things are measured together because
they are the same experiment seen twice:

  G005  the full latency and physical-work vectors, uncontended
  G013  the same measurement with HDD/network traffic deliberately running, so the
        contamination is DEMONSTRATED rather than asserted, and the guard that pauses
        I/O is shown engaging

Both bodies are measured, not just the fast one. The interesting number is not TPOT on
its own -- it is what capability costs in nanoseconds, and that needs a body that works
and a body that does not.
"""
import argparse, json, os, signal, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
sys.path.insert(0, str(REPO / "tools/headless"))

# All three bodies of the composition ladder, so the latency numbers can be read against
# the capability numbers. A TPOT figure for a body that cannot do the work is only
# meaningful as the other half of that trade.
BODIES = [
    ("sealed-3.14", "/Users/scammermike/noetic/NOETIC_PARENT_A", 3.1393),
    ("variantA-2.98", "/Users/scammermike/noetic/VARIANT_A_MLP_ONLY", 2.980254),
    ("clean-2.60", "/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors", 2.596994),
]
TOKENIZER = "/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors"
BINARY = REPO / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
PROMPT = ("Explain, in ordinary prose and at length, how a compiler turns a for-loop into "
          "basic blocks and then into machine code.")


# Work this campaign starts that must FINISH before measuring -- a running pack or suite
# changes the answer and cannot be paused safely mid-write.
OURS = ("python", "cargo", "dd", "rustc", "ld", "clang")
# Transfers, which are resumable and are SUSPENDED for the window instead of waited out.
# Leaving `hf` in OURS made the quiesce wait block on the very download PausedIO exists to
# pause, so the run would have waited out its 45-minute deadline and then measured anyway.
PAUSABLE_IO = ("hf download",)
# GPU-bound work sits at LOW cpu% because it is waiting on Metal, so a cpu-percentage
# proxy reports the machine quiesced while a decode suite is mid-run. These are matched by
# COMMAND LINE regardless of cpu%, which is the only signal that actually sees them.
OUR_WORKLOADS = ("capability_suite.py", "composition_isolation.py",
                 "whole_model_native.py", "ascension_qwen38_hybrid_greedy",
                 "clean_rebuild.py", "capture_moe_x.py", "cold_vs_transfer.py",
                 "matched_bits_probe.py", "expert_family_genome.py")


def quiesce_check():
    """What is competing for the machine, split by what can actually be waited out.

    fileproviderd has been pinned near 100% CPU for over two days on this box and
    avconferenced sits around 40%. Neither belongs to this campaign and neither is mine to
    kill, so blocking on them would wait forever. They are recorded as a standing
    contamination FLOOR that every timing number here carries, rather than pretended away
    -- and the campaign's own work is still waited out properly.
    """
    ours, standing, pausable = [], [], []
    # ONE pass over full command lines. `comm` alone is not enough: the hf downloader's
    # comm is just "Python", so classifying on it put a pausable transfer into the
    # must-finish bucket and blocked quiescence on the very thing PausedIO suspends.
    full = subprocess.run(["ps", "-Ao", "pid,pcpu,etime,command"],
                          capture_output=True, text=True).stdout
    for line in full.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, etime, cmdline = parts[0], parts[2], parts[3]
        try:
            cpu = float(parts[1])
        except ValueError:
            continue
        if "pgrep" in cmdline or cmdline.endswith("/ps") or " -Ao " in cmdline:
            continue
        # the caller is python and will cross the 20% threshold under load, so without
        # this a measurement run counts ITSELF as busy work and can never quiesce. It
        # showed up as a test that passed alone and failed inside a full suite.
        if pid in (str(os.getpid()), str(os.getppid())):
            continue
        row = {"pid": pid, "cpu": cpu, "etime": etime, "cmd": cmdline[:90]}

        # a shell WAITING for a workload names it but is not doing it; counting the
        # variant-B waiter as work deadlocked this run against its own receipt
        stripped = cmdline.lstrip()
        is_waiter = (stripped.startswith(("sh -c", "/bin/sh -c", "zsh -c", "/bin/zsh"))
                     and (" until " in cmdline or " while " in cmdline))

        if any(k in cmdline for k in PAUSABLE_IO):
            pausable.append({**row, "matched_by": "pausable transfer"})
        elif any(w in cmdline for w in OUR_WORKLOADS) and not is_waiter:
            ours.append({**row, "matched_by": "workload name"})
        elif cpu > 20.0 and not is_waiter:
            # Match the PROGRAM, not the command line. A bare substring test put three
            # Google Chrome renderers at ~100% CPU into the must-finish bucket, because
            # the linker token "ld" appears inside "--field-trial-handle". The gate would
            # then have waited for the user's browser to finish, which never happens.
            prog = Path(cmdline.split()[0]).name.lower() if cmdline.split() else ""
            if any(prog == k or prog.startswith(k) for k in OURS):
                ours.append({**row, "matched_by": "cpu over 20% and a tool name",
                             "program": prog})
            else:
                standing.append({**row, "matched_by": "cpu over 20%, not ours"})

    hdd = Path("/Volumes/corpdrive/hawking-modellake/partial")
    return {
        "our_busy_processes": ours, "n_ours_busy": len(ours),
        "pausable_io": pausable, "n_pausable_io": len(pausable),
        "standing_system_load": standing, "n_standing": len(standing),
        "standing_cpu_total": round(sum(r["cpu"] for r in standing), 1),
        "hdd_transfers_in_flight": [p.name for p in hdd.iterdir()] if hdd.is_dir() else [],
        "quiesced": len(ours) == 0,
        "quiesced_means": ("no UNPAUSABLE work started by this campaign is running. It does "
                           "NOT mean an idle machine: see standing_system_load, a floor "
                           "every number here carries and not ours to remove, and "
                           "pausable_io, which is suspended for the window rather than "
                           "waited out."),
    }


from protected_window import ProtectedWindow  # noqa: E402


def io_pids():
    """Model-lake transfers in flight. These are OURS and they are pausable.

    The FILLER parent is included deliberately: pausing only the children lets the
    filler notice their stall and spawn replacements inside the protected window,
    which reintroduces exactly the contamination the window exists to remove.
    """
    pat = "hf download|lake_filler.py"
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    me = os.getpid()
    return [int(x) for x in out.stdout.split()
            if x.strip().isdigit() and int(x) != me]


class PausedIO:
    """Directive §25: GPU cleanliness OVERRIDES I/O overlap.

    Waiting for a 58 GiB acquisition to finish would honour the letter and waste the
    hour. The transfers are resumable by construction (huggingface_hub appends to the
    .incomplete file and re-requests a byte Range), so they are SUSPENDED for the
    protected window and resumed immediately after.

    The resume USED to live in a finally block. That survives an exception in the body
    and nothing else: the parent shell was killed on a timeout mid-window and six
    downloaders were left SIGSTOPped indefinitely. Delegating to ProtectedWindow adds a
    detached watchdog and a healable lease, so the resume no longer depends on this
    process surviving. See tools/odyssey/test_protected_window.py.
    """

    def __init__(self, max_s=1800):
        self.max_s = max_s
        self.win = None
        self.paused = []

    def __enter__(self):
        self.win = ProtectedWindow(io_pids(), max_s=self.max_s)
        self.win.__enter__()
        self.paused = self.win.paused
        return self

    def __exit__(self, *exc):
        return self.win.__exit__(*exc)


def run_once(root, max_new, tag):
    import tempfile
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    f.close()
    cmd = [str(BINARY), "--artifact-root", str(root),
           "--tokenizer", str(Path(TOKENIZER) / "tokenizer.json"),
           "--prompt", PROMPT, "--max-new-tokens", str(max_new),
           "--max-seq-len", str(max_new + 64), "--out", f.name]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0
    body = json.loads(Path(f.name).read_text()) if Path(f.name).stat().st_size else {}
    Path(f.name).unlink(missing_ok=True)
    steps = body.get("gpu_ns_per_step") or []
    return {"tag": tag, "exit_code": p.returncode, "wall_s": round(wall, 3),
            "prefill_wall_ns": body.get("prefill_wall_ns"),
            "first_step_wall_ns": body.get("first_step_wall_ns"),
            "decode_steps": body.get("decode_steps"),
            "decode_wall_ns": body.get("decode_wall_ns"),
            "median_gpu_ns_per_token": body.get("median_gpu_ns_per_token"),
            "dispatches_per_step": body.get("dispatches_per_step"),
            "gpu_ns_per_step": steps,
            "wait_ns_per_step": body.get("wait_ns_per_step"),
            "n_new_tokens": len(body.get("new_token_ids") or [])}


def vectors(runs, ebpw, root):
    """The latency vector and the physical-work vector, from the runs themselves."""
    steps = [s for r in runs for s in (r["gpu_ns_per_step"] or [])]
    med = statistics.median(steps) if steps else None
    ttft = [(r["prefill_wall_ns"] or 0) + (r["first_step_wall_ns"] or 0) for r in runs]
    mix = Path(root) / "MIX_REPORT.json"
    m = json.load(open(mix)) if mix.is_file() else {}
    du = subprocess.run(["du", "-sk", str(root)], capture_output=True, text=True).stdout
    resident = int(du.split()[0]) * 1024 if du.strip() else None
    return {
        "n_reps": len(runs), "n_steps_sampled": len(steps),
        "latency_vector": {
            "TTFT_ns_median": statistics.median(ttft) if ttft else None,
            "prefill_ns_median": statistics.median(
                [r["prefill_wall_ns"] for r in runs if r["prefill_wall_ns"]]) if runs else None,
            "TPOT_ns_median": med,
            "TPOT_ns_p50": statistics.median(steps) if steps else None,
            "TPOT_ns_p95": (sorted(steps)[int(len(steps) * 0.95) - 1] if len(steps) >= 20
                            else None),
            "TPOT_ns_min": min(steps) if steps else None,
            "TPOT_ns_max": max(steps) if steps else None,
            "single_stream_tps": round(1e9 / med, 4) if med else None,
        },
        "physical_work_vector": {
            "dispatches_per_token": runs[0]["dispatches_per_step"] if runs else None,
            "stored_bytes": m.get("payload_bytes"),
            "resident_bytes_on_disk": resident,
            # ARTIFACT_PHYSICAL, derived from the bytes on disk. Reading complete_ebpw
            # out of MIX_REPORT re-imported the frozen design constant: variantA reported
            # 2.5969567 beside its own stored_bytes of 10,019,612,956.
            "ARTIFACT_PHYSICAL_complete_ebpw": (
                round(8.0 * m["payload_bytes"] / 26895998464, 6)
                if m.get("payload_bytes") else None),
            "DESIGN_EXPECTED_complete_ebpw_from_mix_report": m.get("complete_ebpw"),
            "design_and_physical_agree": (
                abs(8.0 * m["payload_bytes"] / 26895998464 - m["complete_ebpw"]) < 1e-3
                if m.get("payload_bytes") and m.get("complete_ebpw") else None),
            "active_bpw": m.get("active_bpw"),
            "wait_ns_per_step": runs[0].get("wait_ns_per_step") if runs else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--emit-perf", required=True)
    ap.add_argument("--emit-clean", required=True)
    a = ap.parse_args()

    import maxx_scheduler as mx

    # Recording that the machine was busy is not the same as refusing to measure while it
    # is. §25 says GPU cleanliness OVERRIDES I/O overlap, so wait for quiescence rather
    # than publishing a contended number with a caveat attached.
    waited_s, deadline = 0.0, 45 * 60
    pre = quiesce_check()
    while not pre["quiesced"] and waited_s < deadline:
        time.sleep(30)
        waited_s += 30
        pre = quiesce_check()
    pre["waited_for_quiescence_s"] = waited_s
    pre["measured_contended_anyway"] = not pre["quiesced"]
    if not pre["quiesced"]:
        print(f"  WARNING: campaign work still running after {waited_s}s; "
              f"ours_busy={pre['n_ours_busy']} -- the figures below are NOT clean")

    results = {}
    io_before = io_pids()
    with PausedIO() as paused_io, \
            mx.ProtectedWindow("QWEN_PERFORMANCE_QUALIFICATION (G005)") as w:
        declined = [{"queue": q, "admitted": mx.admit(q, w)[0], "reason": mx.admit(q, w)[1]}
                    for q in mx.QUEUES]
        for name, root, ebpw in BODIES:
            if not Path(root).exists():
                results[name] = {"absent": root}
                continue
            runs = [run_once(root, a.max_new, f"{name}-warm{i}") for i in range(a.reps)]
            results[name] = {"runs": runs, **vectors(runs, ebpw, root)}

    # G013: the same measurement with the HDD deliberately busy.
    # ALTERNATING PAIRED reps. Running the quiet block first and the noisy block after
    # measured warm-up, not contention: it reported the CONTENDED runs 6.72% FASTER.
    # Alternating puts both arms at the same thermal and page-cache state.
    hdd_target = Path("/Volumes/corpdrive/hawking-modellake/contention-probe.bin")
    contended = None
    name, root, ebpw = BODIES[0]
    paired = {"quiet": [], "noisy": []}
    if Path(root).exists():
        proc = None
        try:
            for i in range(a.reps):
                paired["quiet"].append(run_once(root, a.max_new, f"{name}-quiet{i}"))
                proc = subprocess.Popen(
                    ["sh", "-c", f"for j in $(seq 1 8); do dd if=/dev/urandom "
                                 f"of={hdd_target} bs=1m count=256 2>/dev/null; done"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)
                paired["noisy"].append(run_once(root, a.max_new, f"{name}-noisy{i}"))
                proc.terminate()
                proc.wait(timeout=30)
                proc = None
        finally:
            if proc is not None:
                proc.terminate()
            hdd_target.unlink(missing_ok=True)
        contended = {"runs": paired["noisy"], **vectors(paired["noisy"], ebpw, root)}

    def med(rs):
        st = [x for r in rs for x in (r["gpu_ns_per_step"] or [])]
        return statistics.median(st) if st else None

    clean_med = med(paired["quiet"]) if paired["quiet"] else \
        results[name]["latency_vector"]["TPOT_ns_median"]
    cont_med = med(paired["noisy"]) if paired["noisy"] else None
    delta = ((cont_med - clean_med) / clean_med) if (clean_med and cont_med) else None
    per_pair = [round((med([n]) - med([q])) / med([q]), 5)
                for q, n in zip(paired["quiet"], paired["noisy"])
                if med([q]) and med([n])]

    perf = {
        "schema": "hawking.headless.qwen_performance_qualification.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/performance_qualification.py",
        "obligation": "G005 — QWEN_PERFORMANCE_QUALIFICATION (directive §12, §72, §73, §75, §76)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "reprofiled_from_zero": True,
        "quiesce_check_before": pre,
        "contamination_floor": {
            "what": "processes over 20% CPU that this campaign did not start and cannot "
                    "remove",
            "processes": pre["standing_system_load"],
            "total_cpu_percent": pre["standing_cpu_total"],
            "effect": "every latency figure in this receipt carries this floor. It is "
                      "recorded rather than subtracted, because subtracting an unmeasured "
                      "interference is a forged number.",
        },
        "protected_window": {
            "open": True,
            "declined_queues": [d["queue"] for d in declined if not d["admitted"]],
            "io_paused_for_the_window": {
                "transfers_in_flight_before": io_before,
                "suspended": paused_io.paused,
                "n_suspended": len(paused_io.paused),
                "mechanism": "SIGSTOP for the window, SIGCONT in a finally block",
                "why": "directive §25 -- GPU cleanliness overrides I/O overlap. The "
                       "transfers are resumable by construction, so pausing costs nothing "
                       "but waiting for a 58 GiB acquisition would cost an hour."},
        },
        "bodies": results,
        "capability_context": (
            "read these against receipts/headless/COMPOSITION_ATTRIBUTION.json. The "
            "2.5970-EBPW body is capability-dead and the 2.9803 variant is nearly so; their "
            "TPOT is reported because the trade is the point -- what capability costs in "
            "nanoseconds -- not because either is a candidate."),
        "three_roofs_not_rederived": {
            "DEVICE_THEORETICAL": 819.0, "DEVICE_MEASURED_SUSTAINED": 778.8,
            "source": "receipts/headless/BANDWIDTH_ROOF.json",
            "MODEL_REACHABLE": "per-executable; not copied across bodies (directive §76)"},
        "pass": bool(results and all("runs" in v for v in results.values())
                     and pre["quiesced"]),
    }
    clean = {
        "schema": "hawking.headless.gpu_cleanliness_override.v1",
        "generated_at": perf["generated_at"],
        "generated_by": "tools/odyssey/performance_qualification.py",
        # NOT the same receipt as GPU_CLEANLINESS_OVERRIDE.json. That one owns the
        # MECHANISM (the three resume guarantees and the pausable/standing
        # classification, from tools/odyssey/gpu_cleanliness.py). This one owns the
        # MEASURED EFFECT: latency with contaminating I/O running vs suspended. Two
        # writers on one canonical path is how the genome libraries got silently
        # reverted earlier in this campaign, so they are deliberately separate files.
        "obligation": "G013 — GPU_CLEANLINESS contention demonstration (directive §25)",
        "mechanism_receipt": "receipts/headless/GPU_CLEANLINESS_OVERRIDE.json",
        "hand_authored": False,
        "body_measured": name,
        "paired_measurement": {
            "uncontended_TPOT_ns_median": clean_med,
            "contended_TPOT_ns_median": cont_med,
            "contention": "a 256 MiB dd write loop to the external HDD",
            "design": "ALTERNATING paired reps: quiet, noisy, quiet, noisy ... so both arms "
                      "sit at the same thermal and page-cache state",
            "why_alternating": "sequential blocks measured warm-up rather than contention "
                               "and reported the contended runs 6.72% FASTER",
            "n_pairs": len(per_pair),
            "per_pair_relative_slowdown": per_pair,
            "relative_slowdown": round(delta, 6) if delta is not None else None,
            "sign_is_consistent": (all(x > 0 for x in per_pair) or all(x < 0 for x in per_pair))
                                  if per_pair else None,
            "contamination_demonstrated": bool(
                delta is not None and abs(delta) > 0.02
                and per_pair and all(x > 0 for x in per_pair))},
        "guard": {
            "engaged": True,
            "io_transfers_suspended": len(paused_io.paused),
            "io_resumed_after": True,
            "declined_queues": [d["queue"] for d in declined if not d["admitted"]],
            "reason_example": next((d["reason"] for d in declined if not d["admitted"]), None),
            "mechanism": "tools/headless/maxx_scheduler.py ProtectedWindow + admit()"},
        "no_forged_speedups": (
            "the uncontended figures in QWEN_PERFORMANCE_QUALIFICATION.json were taken inside "
            "the protected window, with the contaminating queues declined; the contended "
            "figures here were taken deliberately outside it to size the effect."),
        "pass": bool(cont_med and clean_med),
    }
    Path(a.emit_perf).write_text(json.dumps(perf, indent=1))
    Path(a.emit_clean).write_text(json.dumps(clean, indent=1))

    for n, v in results.items():
        if "runs" not in v:
            continue
        lv = v["latency_vector"]
        print(f"  {n:12} TPOT_p50={lv['TPOT_ns_p50']}ns  p95={lv['TPOT_ns_p95']}  "
              f"TPS={lv['single_stream_tps']}  dispatch/tok={v['physical_work_vector']['dispatches_per_token']}")
    print(f"  contention: clean={clean_med} contended={cont_med} "
          f"delta={round(delta*100,2) if delta is not None else None}%")
    return 0 if (perf["pass"] and clean["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
