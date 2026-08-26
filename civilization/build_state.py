"""Build ROADMAP_STATE.json from disk truth.

The ERA MAP is judgement and is written here in the open. Everything else --
obligation status, receipt counts, test counts, commit -- is DERIVED from disk,
because a ledger that lets a human retype a status is a ledger that drifts.
"""
import json, pathlib, re, subprocess, sys

HAWKING = pathlib.Path(__file__).resolve().parent.parent
GOAL = pathlib.Path.home() / ".claude/ultragoal/hawking-odyssey-maxx-ascension/GOAL.md"

# Judgement, stated in the open. An obligation lands where its EVIDENCE lands,
# not where its title sounds like it belongs.
ERA_MAP = {
    "I-A_AGENTOS_HCLI":   ["G083", "G076", "G013", "G014", "G015", "G030", "G031", "G063",
                           "G064", "G065"],
    "I-B_DOCTOR":         ["G077", "G016", "G017", "G018", "G019", "G020", "G021", "G035",
                           "G073"],
    "I-C_GRAVITY_NOETIC": ["G082", "G074", "G001", "G002", "G003", "G004", "G005", "G006", "G022",
                           "G023", "G032", "G033", "G034", "G036", "G037", "G038",
                           "G040", "G042", "G059",
                           "G066", "G067", "G068", "G069", "G070", "G071", "G072"],
    "I-D_ACCELERATOR":    ["G078", "G079", "G080", "G081", "G075", "G043", "G044", "G045", "G046", "G047", "G049", "G055", "G058",
                           "G060", "G062"],
    "I-E_ODYSSEY_I":      ["G007", "G008", "G009", "G010", "G011", "G012", "G024",
                           "G025", "G026", "G027", "G028", "G029", "G039", "G041",
                           "G048", "G056", "G061"],
    "II-E_GREEN_MACHINE": ["G057"],
    "IV-A_FUSION":        ["G053", "G054"],
    "IV-B_HMF_HGVAS":     ["G050"],
    "IV-D_EGPU":          ["G051", "G052"],
}
ERA_I = [k for k in ERA_MAP if k.startswith("I-")]

# S015 §II: a civilization is not complete because files exist. These NINE are the
# categories completion is weighted by; a percentage is (satisfied / 9) and nothing
# else, so it can never be computed from a file count.
EVIDENCE_CATEGORIES = ["artifact", "runtime", "adversarial_verification",
                       "negative_control", "failure_recovery", "durable_receipt",
                       "integration", "measured_useful_work", "named_boundaries"]

# Assessed against receipts on disk. Each False is a REAL gap, not a formality.
# GATES are not obligations. I-D has receipts in all nine categories and EVERY gate
# open -- the first build of this ledger reported it 100% and that was the exact
# inflation S015 §II forbids. Evidence coverage is a CEILING, never the score.
OPEN_GATES = {
  "I-A_AGENTOS_HCLI": ["S016 civilization-grade scheduler: Era/Civilization/Program/Gate "
                       "as first-class scheduler concepts -- NOT_STARTED"],
  "I-B_DOCTOR": ["gate: on an UNSEEN model, reduce a huge hypothesis space to a small "
                 "high-information experimental set and explain why -- not run"],
  "I-C_GRAVITY_NOETIC": ["G074", "G023 Noetic compiler pipeline",
                         "real-weight execution gate",
                         "EBPW namespace separation not yet permanent in code"],
  "I-D_ACCELERATOR": ["G075", "C2M T3: several real open CUDA codebases -- NOT CLAIMED",
                      "P2 CUDA differential -- blocked on NVIDIA hardware",
                      "AIR multi-backend (G058) -- Metal only"],
  "I-E_ODYSSEY_I": ["model #2 is not a Noetic executable",
                    "no real weights have executed (blocked on FAST_LOCAL_STORAGE)",
                    "G056 lake fast-forward in flight"],
}

EVIDENCE = {
  "I-A_AGENTOS_HCLI": dict(artifact=1, runtime=1, adversarial_verification=1,
      negative_control=1, failure_recovery=1, durable_receipt=1, integration=1,
      measured_useful_work=1, named_boundaries=1,
      note="the OLD gate is met. S016's civilization-grade scheduler (Era/Civilization/"
           "Program/Gate as first-class scheduler concepts) is NOT_STARTED and is NOT "
           "counted here -- it is a NEW gate, tracked as a blocker."),
  "I-B_DOCTOR": dict(artifact=1, runtime=1, adversarial_verification=0,
      negative_control=1, failure_recovery=0, durable_receipt=1, integration=1,
      measured_useful_work=0, named_boundaries=1,
      note="39-technique library and applicability matrix exist. The gate -- on an "
           "UNSEEN model reduce a huge hypothesis space and explain why -- has not "
           "been run end to end on an unseen specimen."),
  "I-C_GRAVITY_NOETIC": dict(artifact=1, runtime=1, adversarial_verification=1,
      negative_control=1, failure_recovery=0, durable_receipt=1, integration=1,
      measured_useful_work=1, named_boundaries=1,
      note="G023 NOETIC_COMPILER PIPELINE open. Real-weight execution gate open. The "
           "frozen EBPW accounting bug is found and named but the namespaces "
           "(DESIGN_EXPECTED / ARTIFACT_PHYSICAL / RUNTIME_MEASURED) are not yet "
           "permanently separated in code."),
  "I-D_ACCELERATOR": dict(artifact=1, runtime=1, adversarial_verification=1,
      negative_control=1, failure_recovery=1, durable_receipt=1, integration=1,
      measured_useful_work=1, named_boundaries=1,
      note="Every category has receipts. The GATE is still open: C2M T3 is NOT "
           "CLAIMED (no real open CUDA project runs) and P2 has no CUDA differential "
           "because no NVIDIA hardware exists. Category coverage is not gate closure."),
  "I-E_ODYSSEY_I": dict(artifact=1, runtime=1, adversarial_verification=1,
      negative_control=1, failure_recovery=0, durable_receipt=1, integration=1,
      measured_useful_work=0, named_boundaries=1,
      note="4 specimens censused, compounding MEASURED (100%/40%/0% with the Falcon "
           "zero making it meaningful). Model #2 is NOT a Noetic executable and NO "
           "REAL WEIGHTS have executed -- so 'measured useful work' is FALSE for the "
           "school's own product."),
}



NAMED_GATES = {
    "NVIDIA_CUDA_HARDWARE": "P2 differential. No local NVIDIA execution exists.",
    "FAST_LOCAL_STORAGE": ("real weights for G048. Falcon-H1-7B is 15.17 GB and the "
                           "contended USB bus measured under ~0.5 MB/s against "
                           "96-131 MiB/s quiet."),
    "SUDO_POWERMETRICS": "thermal_envelope; sudo not available to this process.",
    "SUDO_PURGE_OR_96GiB_WORKING_SET": ("a repeatable cold read. Evicting the page "
        "cache needs either `sudo purge` -- sudo is not available to this process "
        "-- or a working set larger than the 96 GiB of unified memory. Neither is "
        "available, so every cold number measured here is warm-cache contaminated "
        "and must be labelled so."),
    "XCRUN_METAL": "AOT metallib and generated-code inspection. Toolchain absent.",
}

# Which civilizations each gate actually holds up. A gate that blocks nothing is
# trivia; a gate that blocks a civilization belongs on the critical path.
GATE_BLOCKS = {
    "NVIDIA_CUDA_HARDWARE": ["I-D_ACCELERATOR"],
    "FAST_LOCAL_STORAGE": ["I-E_ODYSSEY_I", "I-C_GRAVITY_NOETIC"],
    "SUDO_POWERMETRICS": ["II-E_GREEN_MACHINE"],
    "SUDO_PURGE_OR_96GiB_WORKING_SET": ["I-D_ACCELERATOR"],
    "XCRUN_METAL": ["I-D_ACCELERATOR"],
}

# Judgement, stated in the open. Ranked by expected roadmap information gain x
# dependency unlock x probability of a decisive result, divided by wall time and
# resource conflict -- NOT by which is easiest.
NEXT_DECISIVE_GATES = [
    {"rank": 1, "civilization": "I-A_AGENTOS_HCLI",
     "gate": "CLAUDE_HCLI_DELEGATION (G064) and HCLI_ALPHA_STANDALONE (G065)",
     "why": ("steer S022, and it is now the EXECUTION-CAPACITY gate for the whole "
             "program: Grok is 402-blocked, so HCLI is the only executor that can be "
             "delegated to. G065 additionally covers the operator's Claude-limit "
             "outage, when it becomes the ONLY way work continues at all."),
     "resource": "CPU + a local model server; the 1B GGUF avoids contending with the fill"},
    {"rank": 2, "civilization": "I-C_GRAVITY_NOETIC",
     "gate": "G023 Noetic compiler pipeline -- the one BLOCKED obligation in Era I",
     "why": ("its recorded blocker was already re-verified FALSE AS STATED: a wired "
             "7,046-line native Metal routed-MoE path exists. The real blocker is much "
             "narrower -- that reader is bound to one model and one artifact family -- so "
             "the unblock is GENERALIZATION of a working reader, not a from-scratch build."),
     "resource": "CPU; no bus contention"},
    {"rank": 3, "civilization": "I-E_ODYSSEY_I",
     "gate": "real weights execute (G048)",
     "why": ("still the highest-information experiment remaining and still the same "
             "RESOURCE CONFLICT: it needs a quiesced window or a 15.17 GB stage competing "
             "with the operator-prioritised fill. Every specimen result to date is "
             "one-layer and random-weight, so nothing yet says anything about adequacy."),
     "resource": "USB bus -- CONTENDED, currently owned by the fill"},
    {"rank": 4, "civilization": "I-D_ACCELERATOR",
     "gate": "C2M T3 -- a real open CUDA project slice, not a synthetic kernel",
     "why": ("I-D has 9/9 evidence categories and 2/10 obligations. T3 is NOT CLAIMED and "
             "is the gate that separates a kernel corpus from a compiler. P2's CUDA "
             "differential stays blocked on hardware that does not exist here."),
     "resource": "CPU + GPU; zero I/O"},
]


def obligations():
    """Parse GOAL.md. Status comes from the file, never from this script."""
    text = GOAL.read_text()
    out = {}
    for m in re.finditer(r"^- \[([ x])\] (G\d+) — (.{0,70})", text, re.M):
        out[m.group(2)] = {"checked": m.group(1) == "x", "title": m.group(3).strip()}
    for m in re.finditer(r"^- \[[ x]\] (G\d+) .*?\| status: ([A-Z_]+)", text, re.M):
        if m.group(1) in out:
            out[m.group(1)]["status"] = m.group(2)
    return out



# --- LIVE STATE ------------------------------------------------------------------
# Everything below is MEASURED at build time. The directive's census requires
# finding "running work not represented in state", and a literal here would be the
# exact fiction that requirement exists to catch.

def _ps():
    return subprocess.run(["ps", "-axo", "command"], capture_output=True,
                          text=True).stdout.splitlines()


AGENT_QUIET_SECS = 300


def running_lanes():
    """Delegation lanes that are ACTUALLY alive, across BOTH executors.

    The first version of this function knew only about Grok, and it reported
    `running_lanes: 0` while three Claude workflow agents were mid-edit in this
    repo -- precisely the "running work not represented in state" defect the field
    exists to catch, committed by the detector itself. A census blind to the
    executor actually in use is not a census.

    The two executors are NOT equally observable, and the difference is recorded
    per lane rather than smoothed over:

      grok    DEFINITIVE. A live `grok` process holding the lane's task.md. Never
              the status file: swgrok documents in its own source that `grok-run
              status` carries no pid and reports long-dead lanes as running, and
              on 2026-08-25 four lanes killed by an HTTP 402 all still read "done".

      claude  HEURISTIC. A workflow agent transcript touched within
              AGENT_QUIET_SECS. There is no pid to check, and an agent can be
              legitimately quiet while one long tool call runs, so this can report
              a finished agent as alive. Labelled, not hidden.
    """
    out = []

    tasks = pathlib.Path.home() / ".claude-grok/tasks"
    if tasks.is_dir():
        cmds = _ps()
        for d in sorted(tasks.iterdir()):
            tm = d / "task.md"
            if tm.is_file() and any(c.startswith("grok ") and str(tm) in c for c in cmds):
                out.append({"lane": d.name, "executor": "grok", "alive": True,
                            "task_file": str(tm), "detection": "definitive",
                            "judged_by": "live grok process holding task.md, not a status file"})

    import time
    now = time.time()
    # RESIDENT AUTONOMOUS WORK. A launchd job was found committing to this branch
    # every five minutes while the census reported zero running lanes -- it landed a
    # commit BETWEEN two of this session's own commits. A delegation lane is not the
    # only kind of running work, and a committer the ledger cannot see is the worst
    # kind to miss.
    # Matched on the LAST token being a driver script under this repo's tools/, not
    # on the line merely mentioning one: the first version also caught this session's
    # own `zsh -c source ...` shell, and a census that over-reports is as useless as
    # one that under-reports.
    for line in _ps():
        tok = line.strip().split()
        if not tok:
            continue
        prog = tok[-1]
        if not (prog.endswith("driver.sh") and str(HAWKING) in prog):
            continue
        out.append({"lane": prog, "executor": "resident",
                        "alive": True, "detection": "definitive",
                        "judged_by": "live process in ps",
                        "commits_to_this_repo": True})

    root = pathlib.Path.home() / ".claude/projects"
    if root.is_dir():
        for wf in sorted(root.glob("*/*/subagents/workflows/wf_*")):
            for a in sorted(wf.glob("agent-*.jsonl")):
                age = now - a.stat().st_mtime
                if age < AGENT_QUIET_SECS:
                    out.append({"lane": f"{wf.name}/{a.stem}", "executor": "claude",
                                "alive": True, "transcript": str(a),
                                "detection": "heuristic",
                                "judged_by": f"transcript touched {age:.0f}s ago "
                                             f"(< {AGENT_QUIET_SECS}s); no pid exists to check, "
                                             "so a finished agent can read as alive"})
    return out


def acquisition_workers():
    """ModelLake fill workers, counted from live processes."""
    cmds = _ps()
    hf = [c for c in cmds if "/hf download " in c or c.startswith("hf download")]
    ml = [c for c in cmds if "modellake.py acquire" in c]
    filler = [c for c in cmds if "lake_filler.py" in c]
    def repo_of(c):
        m = re.search(r"--repo (\S+)", c) or re.search(r"hf download (\S+)", c)
        return m.group(1) if m else "?"
    return {"hf_download_workers": len(hf), "modellake_acquire": len(ml),
            "lake_filler": len(filler),
            "repos": sorted({repo_of(c) for c in hf + ml} - {"?"}),
            "counted_from": "ps, not a status file"}


def unresolved_retractions():
    """Receipts that record a retraction, refutation or in-place amendment.

    This corpus supersedes itself: a later receipt can overturn an earlier one. A
    ledger that does not surface that serves stale laws as current.
    """
    rec = HAWKING / "receipts/headless"
    marks = ("RETRACT", "REFUT", "AMENDED_IN_PLACE", "DID_NOT_REPRODUCE",
             "was a confound", "superseded")
    out = []
    for f in sorted(rec.glob("*.json")):
        try:
            txt = f.read_text()
        except Exception:
            continue
        hit = sorted({m for m in marks if m.lower() in txt.lower()})
        if hit:
            out.append({"receipt": f.name, "markers": hit})
    return out


def laws_since(checkpoint):
    """Receipts newer than the named checkpoint, by mtime.

    Heuristic and labelled as such: mtime is not provenance. It answers 'what
    landed since' well enough to seed the next checkpoint, and nothing stronger.
    """
    cp = HAWKING / "civilization" / checkpoint
    if not cp.is_file():
        return {"basis": "checkpoint absent", "receipts": []}
    since = cp.stat().st_mtime
    rec = HAWKING / "receipts/headless"
    new = sorted(f.name for f in rec.glob("*.json") if f.stat().st_mtime > since)
    return {"basis": "mtime newer than " + checkpoint,
            "heuristic": True,
            "why_heuristic": "mtime is not provenance; an untouched-but-amended receipt is missed",
            "receipts": new}


# Judgement, stated in the open -- like ERA_MAP. Dependencies are what one
# civilization needs FROM another before its gate can close.
DEPENDENCIES = {
    "I-C_GRAVITY_NOETIC": ["G074", 
        "I-D_ACCELERATOR: a representation is not condemned until its native "
        "execution is competent, so a Gravity floor needs an Accelerator kernel",
        "I-E_ODYSSEY_I: real weights on fast local storage (FAST_LOCAL_STORAGE gate)",
    ],
    "I-D_ACCELERATOR": ["G075", 
        "I-E_ODYSSEY_I: Accelerator work must be driven by real specimen "
        "bottlenecks, not synthetic ones",
        "EXTERNAL: NVIDIA hardware for any CUDA differential (no local execution exists)",
    ],
    "I-E_ODYSSEY_I": [
        "I-B_DOCTOR: prescription before packing, or the school runs blind experiments",
        "RESOURCE: the USB bus, currently owned by the ModelLake fill",
    ],
    "I-B_DOCTOR": ["I-E_ODYSSEY_I: more than one architecture, or the library is Qwen folklore"],
    "I-A_AGENTOS_HCLI": ["G076", ],
}


def build():
    obs = obligations()
    mapped = {g for v in ERA_MAP.values() for g in v}
    unmapped = sorted(set(obs) - mapped)
    orphan = sorted(mapped - set(obs))          # named in the map, absent from GOAL.md

    civ = {}
    for name, ids in ERA_MAP.items():
        present = [g for g in ids if g in obs]
        open_ids = [g for g in present if not obs[g]["checked"]]
        ev = EVIDENCE.get(name)
        sat = sum(ev[c] for c in EVIDENCE_CATEGORIES) if ev else None
        gates = OPEN_GATES.get(name, [])
        ob_pct = round(100 * (len(present) - len(open_ids)) / len(present), 1) if present else None
        ev_pct = round(100 * sat / len(EVIDENCE_CATEGORIES), 1) if ev else None
        civ[name] = {
            "obligations": present,
            "verified": len(present) - len(open_ids),
            "open": open_ids,
            "evidence_satisfied": sat,
            "evidence_of": len(EVIDENCE_CATEGORIES) if ev else None,
            "evidence_pct": ev_pct,
            "obligation_pct": ob_pct,
            "completion_pct": min(ev_pct, ob_pct) if (ev_pct is not None and ob_pct is not None) else None,
            "completion_basis": ("min(evidence categories satisfied / 9, obligations "
                                 "verified / total). The MINIMUM on purpose: I-D has all "
                                 "nine categories and zero verified obligations, and "
                                 "reporting 100% there is the inflation S015 §II forbids. "
                                 "Never a file count."),
            "open_gates": gates,
            "status": ("CIVILIZATION_COMPLETE" if (ev_pct == 100 and ob_pct == 100 and not gates)
                       else "INTEGRATED" if (ev_pct == 100 and not gates)
                       else "ADVERSARIALLY_VERIFIED" if (ev and ev["adversarial_verification"] and ev["negative_control"])
                       else "PHYSICALLY_RUNNING" if (ev and ev["runtime"])
                       else "BUILDING" if ev else "EXPLORING"),
            "evidence": {c: bool(ev[c]) for c in EVIDENCE_CATEGORIES} if ev else None,
            "note": ev.get("note") if ev else "Era-IV/II advance work; not scored under Era-I sovereignty.",
        }

    counts = {}
    for g, v in obs.items():
        counts[v.get("status", "UNKNOWN")] = counts.get(v.get("status", "UNKNOWN"), 0) + 1

    rec = HAWKING / "receipts/headless"

    # THE INTERPRETER IS PART OF THE MEASUREMENT. The default `python3` on this box
    # is 3.14.6 with NO mlx, where tools/accelerator reports five failures; the
    # framework 3.12 has mlx and reports them all passing. A test count without the
    # interpreter that produced it is not a measurement, so both are recorded and
    # the validator refuses a count that arrives without one.
    PY = "/usr/local/bin/python3"
    tests = subprocess.run(
        [PY, "-m", "pytest", str(HAWKING / "tools/accelerator"), "-q"],
        capture_output=True, text=True, cwd=HAWKING).stdout
    tm = re.search(r"(\d+) passed", tests)
    fm = re.search(r"(\d+) failed", tests)
    interp = subprocess.run(
        [PY, "-c", "import sys;import mlx.core as m;print(sys.version.split()[0], m.__file__)"],
        capture_output=True, text=True).stdout.strip()
    test_env = {
        "interpreter": PY,
        "resolves_to": str(pathlib.Path(PY).resolve()),
        "version_and_mlx": interp or "mlx NOT importable under this interpreter",
        "suite": "tools/accelerator",
        "failed": int(fm.group(1)) if fm else 0,
    }

    lanes = running_lanes()
    acq = acquisition_workers()
    retractions = unresolved_retractions()
    stage = pathlib.Path.home() / "noetic/stage"
    stage_bytes = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) if stage.is_dir() else 0
    stage_note = f"~/noetic/stage holds {stage_bytes/1e6:.1f} MB across {len(list(stage.rglob('*'))) if stage.is_dir() else 0} entries"

    # A blocker is only real if it is QUANTIFIED. "no runtime" and "storage slow"
    # are not blockers, they are shrugs -- the directive says so in section XII.
    blockers = []
    for gate, why in NAMED_GATES.items():
        blockers.append({"gate": gate, "quantified_as": why,
                         "blocks": GATE_BLOCKS.get(gate, [])})

    return {
        "roadmap_version": "HAWKING_CIVILIZATION_ASCENSION_V1",
        "frozen_plan": "HAWKING_SUPER_ROADMAP_FREEZE_V1_2026-08-25.md",
        "generated_from": "disk truth: GOAL.md + receipts + git + a real pytest run",
        "active_era": "I",
        "era_sovereignty": ("ERA I is sovereign. Later-era work is permitted ONLY when it "
                            "is already running, consumes an idle resource, produces "
                            "infrastructure Era I needs, or resolves an uncertainty that "
                            "changes Era-I design. It NEVER earns civilization completion."),
        "active_civilizations": ERA_I,
        "civilization_status": civ,
        "obligation_status_counts": counts,
        "obligations_total": len(obs),
        "unmapped_obligations": unmapped,
        "orphan_map_entries": orphan,
        "receipt_count": len(list(rec.glob("*.json"))),
        "accelerator_receipt_count": len(list(rec.glob("ACCELERATOR_*.json"))),
        "last_verified_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=HAWKING).stdout.strip(),
        "last_verified_test_count": int(tm.group(1)) if tm else None,
        "test_count_is_from_a_run_not_arithmetic": True,
        "test_environment": test_env,
        "named_gates": NAMED_GATES,
        # DERIVED, not typed. This field carried "4 hf download workers" as a literal
        # and drifted the moment the fill changed shape -- a ledger that lets a human
        # retype a measurement is a ledger that lies with confidence.
        "resource_ownership": {
            "USB_BUS_corpdrive": {
                "owner": "ModelLake fill -- OPERATOR PRIORITISED",
                "workers": acq,
                "consequence": ("frontier science that needs this bus must wait for a "
                                "quiesced window; zero-I/O science is unaffected"),
            },
            "GPU": "free for zero-I/O accelerator science",
            "TIER1_SSD": stage_note,
        },

        # --- directive VIII required fields ---------------------------------------
        "civilization_progress": {
            "value_pct": 1.0,
            "heuristic": True,
            "basis": ("the frozen plan's civilizational coordinate against the COMPLETE "
                      "Hawking system as denominator -- five eras, twenty-five "
                      "civilizations. It is NOT a ledger-completion statistic and must "
                      "never be recomputed from obligation or file counts."),
            "source": "HAWKING_SUPER_ROADMAP_FREEZE_V1_2026-08-25.md",
        },
        "completion_evidence": {
            name: {"categories_met": sum(EVIDENCE[name][c] for c in EVIDENCE_CATEGORIES),
                   "of": len(EVIDENCE_CATEGORIES),
                   "note": EVIDENCE[name]["note"]}
            for name in ERA_I
        },
        "blockers": blockers,
        "dependencies": DEPENDENCIES,
        "running_lanes": lanes,
        "next_decisive_gates": NEXT_DECISIVE_GATES,
        "unresolved_retractions": retractions,
        "laws_since_last_checkpoint": laws_since("ERA_I_CHECKPOINT_001.json"),
    }


if __name__ == "__main__":
    s = build()
    pathlib.Path(__file__).with_name("ROADMAP_STATE.json").write_text(json.dumps(s, indent=1))
    print(f"era {s['active_era']} | {s['obligations_total']} obligations "
          f"{s['obligation_status_counts']} | unmapped={s['unmapped_obligations']} "
          f"orphan={s['orphan_map_entries']} | tests={s['last_verified_test_count']}")
    for k in ERA_I:
        c = s["civilization_status"][k]
        print(f"  {k:20s} {c['completion_pct']:5.1f}%  {c['verified']}/{len(c['obligations'])} verified"
              f"  open={c['open']}")
