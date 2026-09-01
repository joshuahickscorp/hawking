"""Build ROADMAP_STATE.json from disk truth.

The ERA MAP is judgement and is written here in the open. Everything else --
obligation status, receipt counts, test counts, commit -- is DERIVED from disk,
because a ledger that lets a human retype a status is a ledger that drifts.
"""
import hashlib, json, os, pathlib, re, subprocess, sys

HAWKING = pathlib.Path(__file__).resolve().parent.parent
GOAL = pathlib.Path.home() / ".claude/ultragoal/hawking-odyssey-maxx-ascension/GOAL.md"
# The old freeze remains source lineage, but this is the governing execution
# decree supplied for the current campaign.  Keep the path and its digest in the
# generated ledger so a fresh process can distinguish a changed roadmap from a
# changed repository.
CANONICAL_ROADMAP = pathlib.Path.home() / "Downloads/H-ROADMAP.md"
CANONICAL_ROADMAP_VERSION = "H-ROADMAP_CRISPR_EXECUTION_SPECIFICATION_2026-08-27"
CANONICAL_CIVILIZATIONAL_COORDINATE = 0.7

# The canonical decree has five eras and twenty-five named programs.  Existing
# obligation evidence maps into this vocabulary; absent evidence stays
# NOT_STARTED instead of being inferred from implementation surface area.
CANONICAL_PROGRAMS = (
    "I-A_AGENTOS_HCLI", "I-B_DOCTOR", "I-C_GRAVITY_NOETIC", "I-D_ACCELERATOR", "I-E_ODYSSEY_I",
    "II-A_ODYSSEY_II", "II-B_NOETIC_COMPILER_V1", "II-C_PHYSICAL_GRAPH_COMPILER",
    "II-D_STATE_TOKENIZER_DECODING", "II-E_GREEN_MACHINE",
    "III-A_ODYSSEY_III", "III-B_LEARNED_PHYSICAL_COMPILER", "III-C_RESIDENT_OPTIMIZER",
    "III-D_BEYOND_DENSE_REPRESENTATION", "III-E_AUTONOMOUS_REPRODUCIBLE_SCIENCE",
    "IV-A_FUSION", "IV-B_HMF_HGVAS", "IV-C_DGX_SPARK", "IV-D_EGPU",
    "IV-E_FUSION_BRIDGE_TOPOLOGY_ASCENSION",
    "V-A_PRODUCT_SOVEREIGNTY", "V-B_DEVELOPER_PLATFORM", "V-C_CONTINUOUS_VERIFIED_IMPROVEMENT",
    "V-D_DOMINANCE_SCOREBOARD", "V-E_PERPETUAL_HAWKING",
)

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
# GATES are not obligations. I-D has receipts in all nine categories and its
# bounded source-layer parity is now passed, while the broader gates remain open.
# Evidence coverage is a CEILING, never the score.
OPEN_GATES = {
  "I-A_AGENTOS_HCLI": ["S016 civilization-grade scheduler: Era/Civilization/Program/Gate "
                       "as first-class scheduler concepts -- NOT_STARTED"],
  "I-B_DOCTOR": ["gate: on an UNSEEN model, reduce a huge hypothesis space to a small "
                 "high-information experimental set and explain why -- not run"],
  "I-C_GRAVITY_NOETIC": ["G074", "G023 Noetic compiler pipeline",
                         "real-weight execution gate",
                         "EBPW namespace separation not yet permanent in code"],
  "I-D_ACCELERATOR": ["G075", "48-layer complete-token source runtime -- NOT CLAIMED",
                      "C2M T3: project-level runtime/compiler coverage over eight diverse CUDA sources -- NOT CLAIMED",
                      "ANE MLProgram atlas: public Core ML compilation/placement/latency evidence -- NOT MEASURED",
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
      note="Every category has receipts and bounded source-BF16 parity passes on "
           "three contiguous eligible linear-attention layers on Apple Metal. Eight independent, "
           "diverse open CUDA source trees are now fully checked out and censused, "
           "two translated vector-add kernels plus one literal scalar-scale kernel and "
           "the supported host allocation/copy/launch sequence match a numpy oracle on "
           "Apple Metal; project-level C2M-T3 runtime "
           "coverage remains NOT CLAIMED, and P2 has no CUDA differential "
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
     "gate": "Qwen3.8-Flash-Next complete exact layer -> first complete native token",
             "why": ("primary resident campaign restored after the bounded Qwen3-30B-A3B "
             "uniform-Q4 control closure: finish the exact complete Flash layer, "
             "with layers 0..2, the layer-3 full-attention + MoE organ, layers 4..6, "
             "the layer-7 full-attention + MoE organ, and layers 8..9 now source-parity "
             "verified through explicit state handoffs, then continue the layer-10 "
             "boundary and earn a complete native "
             "token, and continue toward executable, capability, performance, and "
             "HCLI-resident qualification."),
     "resource": "Apple Metal + local Flash source/artifact; preserve the ModelLake fill boundary"},
    {"rank": 2, "civilization": "I-A_AGENTOS_HCLI",
     "gate": "CLAUDE_HCLI_DELEGATION (G064)",
     "why": ("steer S022, and it is now the EXECUTION-CAPACITY gate for the whole "
             "program: G065's operator-only path is now verified, while Grok is "
             "402-blocked and Claude-facing delegation remains the only unqualified "
             "execution surface."),
     "resource": "CPU + a local model server; the 1B GGUF avoids contending with the fill"},
    {"rank": 3, "civilization": "I-C_GRAVITY_NOETIC",
     "gate": "G023 Noetic compiler pipeline -- the one BLOCKED obligation in Era I",
     "why": ("its recorded blocker was already re-verified FALSE AS STATED: a wired "
             "7,046-line native Metal routed-MoE path exists. The real blocker is much "
             "narrower -- that reader is bound to one model and one artifact family -- so "
             "the unblock is GENERALIZATION of a working reader, not a from-scratch build."),
     "resource": "CPU; no bus contention"},
    {"rank": 4, "civilization": "I-E_ODYSSEY_I",
     "gate": "real weights execute (G048)",
     "why": ("still the highest-information experiment remaining and still the same "
             "RESOURCE CONFLICT: it needs a quiesced window or a 15.17 GB stage competing "
             "with the operator-prioritised fill. Every specimen result to date is "
             "one-layer and random-weight, so nothing yet says anything about adequacy."),
     "resource": "USB bus -- CONTENDED, currently owned by the fill"},
    {"rank": 5, "civilization": "I-D_ACCELERATOR",
     "gate": "C2M T3 -- project-level runtime/compiler coverage over real CUDA sources",
     "why": ("I-D has 9/9 evidence categories and 2/15 obligations. Bounded source-BF16 "
             "parity now passes on three eligible layers, and eight diverse real CUDA "
             "source trees are fully checked out and censused with two vector-add and one "
             "literal scalar-scale translated kernel plus a T1 host sequence on Apple; "
             "project-level "
             "T3 is still NOT CLAIMED and is the gate that separates a kernel corpus "
             "from a compiler. P2's CUDA "
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

    # ROADMAP WORK outside the commit lane.  The Qwen30 base uniform-Q4 packer
    # is a real G023 WorkUnit whose durable artifact lives under ~/noetic; if
    # this process is alive it must be visible in the canonical census even
    # though it does not commit to this checkout.
    for line in _ps():
        if "ascension_qwen30_uniform_q4_repack.py" not in line:
            continue
        if "python" not in line.lower():
            continue
        out.append({"lane": "G023:qwen30-base-uniform-q4-pack",
                    "executor": "roadmap",
                    "alive": True,
                    "detection": "definitive",
                    "judged_by": "live packer process in ps",
                    "commits_to_this_repo": False,
                    "artifact_root": str(pathlib.Path.home() / "noetic/MODEL2_Q4_ARTIFACT")})

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


def _sha256(path):
    """Hash the canonical roadmap when it is locally available.

    The roadmap is external to the repository by deliberate user placement, so
    absence is reported rather than replaced with a remembered digest.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def canonical_program_statuses(civilizations):
    """Map existing evidence into the five-era canonical program vocabulary.

    No source or test is treated as progress for a program with no mapped
    evidence.  This prevents the V5 ledger from turning its new denominator
    into implied completion.
    """
    statuses = {}
    for name in CANONICAL_PROGRAMS:
        current = civilizations.get(name)
        statuses[name] = {
            "status": current["status"] if current else "NOT_STARTED",
            "evidence": ("civilization_status." + name) if current else None,
            "claim_boundary": (
                "Existing Era-I evidence is mapped here; it does not close the program's "
                "canonical promotion gate."
                if current else
                "No mapped disk evidence has been evaluated for this canonical program."
            ),
        }
    return statuses


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
    program_statuses = canonical_program_statuses(civ)
    canonical_roadmap_hash = _sha256(CANONICAL_ROADMAP)
    try:
        disk_available_bytes = os.statvfs(HAWKING).f_bavail * os.statvfs(HAWKING).f_frsize
    except OSError:
        disk_available_bytes = None
    stage = pathlib.Path.home() / "noetic/stage"
    stage_bytes = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) if stage.is_dir() else 0
    stage_note = f"~/noetic/stage holds {stage_bytes/1e6:.1f} MB across {len(list(stage.rglob('*'))) if stage.is_dir() else 0} entries"

    # A blocker is only real if it is QUANTIFIED. "no runtime" and "storage slow"
    # are not blockers, they are shrugs -- the directive says so in section XII.
    blockers = []
    for gate, why in NAMED_GATES.items():
        blockers.append({"gate": gate, "quantified_as": why,
                         "blocks": GATE_BLOCKS.get(gate, [])})

    state = {
        "roadmap_version": CANONICAL_ROADMAP_VERSION,
        "roadmap_hash": canonical_roadmap_hash,
        "roadmap_path": str(CANONICAL_ROADMAP),
        "legacy_frozen_plan": "HAWKING_SUPER_ROADMAP_FREEZE_V1_2026-08-25.md",
        "frozen_plan": "H-ROADMAP.md (canonical execution decree)",
        "generated_from": "disk truth: GOAL.md + receipts + git + a real pytest run",
        "active_era": "I",
        "era_sovereignty": ("ERA I is sovereign. Later-era work is permitted ONLY when it "
                            "is already running, consumes an idle resource, produces "
                            "infrastructure Era I needs, or resolves an uncertainty that "
                            "changes Era-I design. It NEVER earns civilization completion."),
        "resident_daemon_policy": {
            "role": "supporting Hawking infrastructure, not a separate primary project",
            "law": "THE DAEMON SERVES HAWKING. HAWKING DOES NOT SERVE THE DAEMON.",
            "qualification": "organic_on_real_roadmap_need",
            "cuda_hardware": "HARDWARE_BLOCKED; port CUDA architecture to Apple Silicon/Metal",
            "unqualified_behaviors": [
                "actual heavy resident load",
                "measured UMA self-evacuation",
                "protected GPU clean-room unload/reload",
                "multi-lane physical contention",
                "long unattended mixed-workload run",
            ],
            "defect_response": "patch, verify, and return immediately to the roadmap",
            "speculative_architecture_expansion": False,
        },
        "active_civilizations": ERA_I,
        "civilization_status": civ,
        "civilizational_coordinate": CANONICAL_CIVILIZATIONAL_COORDINATE,
        "era_statuses": {
            "I": {"status": "PHYSICALLY_RUNNING", "claim_boundary": "Era I remains sovereign and incomplete."},
            "II": {"status": "EXPLORING", "claim_boundary": "Advance work is tracked but cannot earn Era-I completion."},
            "III": {"status": "NOT_STARTED", "claim_boundary": "No program has been promoted into Era III."},
            "IV": {"status": "EXPLORING", "claim_boundary": "Fusion/HMF/eGPU advance work is not an era graduation."},
            "V": {"status": "NOT_STARTED", "claim_boundary": "No product-era promotion evidence has been evaluated."},
        },
        "program_statuses": program_statuses,
        "active_odyssey": {"id": "I-E_ODYSSEY_I", "status": civ["I-E_ODYSSEY_I"]["status"]},
        "active_textbooks": {
            "Qwen3.8-27B": {"role": "control / Accelerator regression school", "evidence": "receipts/headless/QWEN27_HISTORICAL_RUNTIME_IDENTITY.json"},
            "Qwen3.8-Flash-Next": {"role": "current MoE / Noetic executable school", "status": "BOUNDED_SOURCE_PARITY_L0_L39_WITH_LAYER3_LAYER7_LAYER11_LAYER15_LAYER19_LAYER23_LAYER27_LAYER31_LAYER35_LAYER39_FULL_ATTENTION_MOE_ORGANS", "evidence": "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER39_ORGAN.json", "cross_layer_evidence": "receipts/headless/FLASH_NOETIC_MULTILAYER_LINEAR_PREFIX_L0_L2.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER3_ORGAN.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER4_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER5_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER6_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER7_ORGAN.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER8_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER9_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER10_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER11_ORGAN.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER12_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER13_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER14_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER15_ORGAN.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER16_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER17_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER18_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER19_ORGAN.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER20_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER21_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER22_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER23_ORGAN.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER24_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER25_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER26_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER27_ORGAN.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER28_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER29_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER30_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER31_ORGAN.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER32_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER33_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER34_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER35_ORGAN.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER36_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER37_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_COMPLETE_LAYER38_PREFIX_FED_NATIVE.json + receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER39_ORGAN.json", "state_handoff": "receipts/headless/FLASH_PREFIX_FED_LAYER39_STATE.f32", "next_boundary": "implement and verify the layer-40 linear-attention graph"},
            "U50DD/XCU50": {"role": "FPGA textbook #1", "status": "NOT_STARTED", "evidence": None},
        },
        "primary_campaign": {
            "model": "Qwen3.8-Flash-Next",
            "role": "primary HCLI resident candidate under investigation",
            "objective": "close first exact complete Flash layer -> first complete native token -> complete Noetic executable -> protected capability/performance qualification -> HCLI resident evaluation",
            "control_boundary": "Qwen3-30B-A3B uniform-Q4 is a bounded generalization/transfer control, not the primary resident target",
        },
        "accelerator_substrate": {
            "ane_provider": "hcli.ane_provider.ANEProvider",
            "result_envelope": "hcli.result_envelope.ResultEnvelope",
            "resource_lease_authority": "hcli.resources + resident daemon lease checks",
            "artifact_extensions": {"representation_shard": ".nr", "final_executable": ".nx", "historical_compatibility": ".gravity"},
            "device_profile": "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json",
            "atlas": "receipts/headless/APPLE_ANE_ATLAS.json",
            "scoreboard": "receipts/headless/ACCELERATOR_SCOREBOARD.json",
            "scoreboard_builder": "tools/accelerator/scoreboard.py",
            "public_api_only": True,
            "private_interface_control": "forbidden",
            "selection_authority": "measured complete useful work; ANE visibility alone never promotes placement",
        },
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
        # A count without its SCOPE is as misleading as one without its
        # interpreter. This is one directory, not the repo: tools/headless alone
        # carries 41 pre-existing failures that this number never sees.
        "test_count_scope": "tools/accelerator only, `-q`, one directory",
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
        "resource_state": {
            "disk_available_bytes": disk_available_bytes,
            "resource_ownership": "resource_ownership",
            "measurement": "statvfs at ledger build time; not a capacity forecast",
        },
        "background_jobs": {"running_lanes": lanes, "modellake": acq},
        "open_workunits": NEXT_DECISIVE_GATES,
        "blocked_workunits": blockers,
        "accepted_receipts": [
            "receipts/headless/HCLI_RESTART_RESUME.json",
            "receipts/headless/HCLI_AGENTOS_UNATTENDED_WINDOW_LONG.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER0_NATIVE_PARITY.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER2_NATIVE_PARITY.json",
            "receipts/headless/ACCELERATOR_C2M_T3_REAL_PROJECTS.json",
            "receipts/headless/G065_HCLI_ALPHA_STANDALONE.json",
            "receipts/headless/NOETIC_MODEL2_Q4_GENERALIZATION.json",
            "receipts/headless/NOETIC_MODEL2_Q4_CONTROL_PHYSICAL_TOKEN.json",
            "receipts/headless/FLASH_NOETIC_MULTILAYER_LINEAR_PREFIX_L0_L2.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER3_ORGAN.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER4_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER5_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER6_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER7_ORGAN.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER8_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER9_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER10_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER11_ORGAN.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER12_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER13_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER14_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER15_ORGAN.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER16_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER17_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER18_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER19_ORGAN.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER20_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER21_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER22_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER23_ORGAN.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER24_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER25_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER26_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER27_ORGAN.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER28_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER29_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_DISPATCH_OPTIMIZATION_PROBE_LAYER29.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER30_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER31_ORGAN.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER32_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER33_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER34_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER35_ORGAN.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER36_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER37_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER38_PREFIX_FED_NATIVE.json",
            "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER39_ORGAN.json",
            "receipts/headless/FLASH_GRAVITY_DOCTOR_CYCLE.json",
            "receipts/headless/FLASH_GRAVITY_DOCTOR_CYCLE_V3.json",
            "receipts/headless/FLASH_DOCTOR_EXPERT_BANK_SCREEN_L7_128.json",
            "receipts/headless/FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json",
            "receipts/headless/FLASH_ROUTE_CONDITIONED_SHARED_BASIS_L4.json",
            "receipts/headless/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json",
            "receipts/headless/FLASH_MARGIN_RESIDUAL_CANDIDATE_L3_L4.json",
            "receipts/headless/FLASH_MARGIN_RESIDUAL_NATIVE_PARITY_L4_L6.json",
            "receipts/headless/FLASH_ROUTE_CONDITIONED_OUTPUT_BASIS_L4.json",
            "receipts/headless/FLASH_FAST_COMPACT_L0_L7_PARITY.json",
            "receipts/headless/FLASH_FUSED_ROUTE_ACCUMULATE_PARITY_L3.json",
            "receipts/headless/FLASH_ROUTE_ARCHETYPE_SPARSE_SCREEN_L4.json",
            "receipts/headless/FLASH_FUSED_ROUTE_ACCUMULATE_L7.json",
            "receipts/headless/FLASH_FUSED_ROUTE_ACCUMULATE_PARITY_L7.json",
            "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json",
            "receipts/headless/APPLE_ANE_ATLAS.json",
            "receipts/headless/FLASH_TERMINAL_EXECUTOR_COMPILE.json",
            "receipts/headless/ACCELERATOR_SCOREBOARD.json",
        ],
        "negative_science": retractions,
        "pareto_frontiers": ["receipts/headless/PARETO_ARCHIVE.json"],
        "laws": laws_since("ERA_I_CHECKPOINT_001.json"),
        "scars": retractions,
        "benchmark_qualification": {
            "qwen27": "receipts/headless/QWEN27_RUNTIME_DIFF.json",
            "flash_next": "BOUNDED_SOURCE_PARITY_L0_L39_WITH_LAYER3_LAYER7_LAYER11_LAYER15_LAYER19_LAYER23_LAYER27_LAYER31_LAYER35_LAYER39_FULL_ATTENTION_MOE_ORGANS; layers 0..2 linear prefix, layer 3 full-attention + routed/shared MoE organ, layers 4..6 linear, layer 7 full-attention + routed/shared MoE organ, layers 8..10 linear, layer 11 full-attention + routed/shared MoE organ, layers 12..14 linear, layer 15 full-attention + routed/shared MoE organ, layers 16..18 linear, layer 19 full-attention + routed/shared MoE organ, layers 20..22 linear, layer 23 full-attention + routed/shared MoE organ, layers 24..26 linear-attention, layer 27 full-attention + routed/shared MoE organ, layers 28..30 linear-attention, layer 31 full-attention + routed/shared MoE organ, layers 32..34 linear-attention, layer 35 full-attention + routed/shared MoE organ, layers 36..38 linear-attention, and layer 39 full-attention + routed/shared MoE organ pass exact source-BF16 parity through explicit state handoffs; layers 40..47, complete-token, EBPW, and accepted-TPS promotion remain open",
        },
        "next_work": NEXT_DECISIVE_GATES,
        "last_handoff": "receipts/headless/HCLI_AGENTOS_HANDOFF.json",

        # --- directive VIII required fields ---------------------------------------
        "civilization_progress": {
            "value_pct": CANONICAL_CIVILIZATIONAL_COORDINATE,
            "heuristic": True,
            "basis": ("the frozen plan's civilizational coordinate against the COMPLETE "
                      "Hawking system as denominator -- five eras, twenty-five "
                      "civilizations. It is NOT a ledger-completion statistic and must "
                      "never be recomputed from obligation or file counts."),
            "source": str(CANONICAL_ROADMAP),
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

    # Derive the active Flash boundary from the receipts actually present on
    # disk.  This prevents the long human-readable textbook string above from
    # lagging a verified layer and keeps the state ledger truthful after each
    # physical continuation run.
    flash_receipts = []
    for pattern, kind in (
        ("FLASH_NOETIC_COMPLETE_LAYER*_PREFIX_FED_NATIVE.json", "linear-attention"),
        ("FLASH_NOETIC_FULL_ATTENTION_LAYER*_ORGAN.json", "full-attention/MoE"),
    ):
        for receipt in rec.glob(pattern):
            match = re.search(r"LAYER(\d+)", receipt.name)
            if not match:
                continue
            try:
                payload = json.loads(receipt.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") == "PASSED":
                flash_receipts.append((int(match.group(1)), kind, receipt))
    if flash_receipts:
        latest_layer, latest_kind, latest_receipt = max(flash_receipts, key=lambda row: row[0])
        latest_state = rec / f"FLASH_PREFIX_FED_LAYER{latest_layer}_STATE.f32"
        if latest_state.exists():
            textbook = state["active_textbooks"]["Qwen3.8-Flash-Next"]
            census = rec / "FLASH_ORGAN_CENSUS.json"
            try:
                census_payload = json.loads(census.read_text())
            except (OSError, json.JSONDecodeError):
                census_payload = None
            if isinstance(census_payload, dict) and census_payload.get("schema") == "hawking.flash.organ_census.v1":
                census_ref = str(census.relative_to(HAWKING))
                if census_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {census_ref}"
                if census_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(census_ref)
                textbook["organ_census"] = {
                    "receipt": census_ref,
                    "status": census_payload.get("status"),
                    "tensor_count": census_payload.get("tensor_count"),
                    "source_parameter_bytes_indexed": census_payload.get("source_parameter_bytes_indexed"),
                    "family_summary": census_payload.get("family_summary"),
                    "claim": census_payload.get("claim_boundary"),
                }
            complete_nr = rec / "FLASH_COMPLETE_V0.nr.json"
            complete_nx = rec / "FLASH_COMPLETE_V0.nx.json"
            try:
                complete_nr_payload = json.loads(complete_nr.read_text())
                complete_nx_payload = json.loads(complete_nx.read_text())
            except (OSError, json.JSONDecodeError):
                complete_nr_payload = None
                complete_nx_payload = None
            if isinstance(complete_nr_payload, dict) and complete_nr_payload.get("schema") == "hawking.flash.complete_nr.v0":
                complete_nr_ref = str(complete_nr.relative_to(HAWKING))
                if complete_nr_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {complete_nr_ref}"
                if complete_nr_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(complete_nr_ref)
                textbook["complete_nr_candidate"] = {
                    "receipt": complete_nr_ref,
                    "status": complete_nr_payload.get("status"),
                    "scope": (complete_nr_payload.get("representation") or {}).get("scope"),
                    "complete_bits_per_weight": (complete_nr_payload.get("representation") or {}).get("complete_bits_per_weight"),
                    "family_count": len((complete_nr_payload.get("representation") or {}).get("parts") or []),
                    "promotion_allowed": (complete_nr_payload.get("promotion") or {}).get("allowed"),
                    "claim": complete_nr_payload.get("claim_boundary"),
                }
            if isinstance(complete_nx_payload, dict) and complete_nx_payload.get("schema") == "hawking.flash.nx_genome.v1":
                complete_nx_ref = str(complete_nx.relative_to(HAWKING))
                if complete_nx_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {complete_nx_ref}"
                if complete_nx_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(complete_nx_ref)
                textbook["complete_nx_candidate"] = {
                    "receipt": complete_nx_ref,
                    "status": complete_nx_payload.get("status"),
                    "machine_genome": (complete_nx_payload.get("compiled_for_machine_genome") or {}).get("genome_digest"),
                    "accepted_multitoken_tps": (complete_nx_payload.get("qualification") or {}).get("accepted_multitoken_tps"),
                    "complete_system_ebpw": (complete_nx_payload.get("qualification") or {}).get("complete_system_ebpw"),
                    "resident_promotion": (complete_nx_payload.get("qualification") or {}).get("resident_promotion"),
                    "claim": complete_nx_payload.get("claim_boundary"),
                }
            complete_ledger = rec / "FLASH_COMPLETE_V0.BYTE_LEDGER.json"
            try:
                complete_ledger_payload = json.loads(complete_ledger.read_text())
            except (OSError, json.JSONDecodeError):
                complete_ledger_payload = None
            if isinstance(complete_ledger_payload, dict) and complete_ledger_payload.get("schema") == "hawking.flash.complete_byte_ledger.v1":
                ledger_ref = str(complete_ledger.relative_to(HAWKING))
                if ledger_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {ledger_ref}"
                if ledger_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(ledger_ref)
                exact = complete_ledger_payload.get("complete_exact_control") or {}
                profiled = complete_ledger_payload.get("measured_fastpath_profile") or {}
                textbook["complete_byte_ledger"] = {
                    "receipt": ledger_ref,
                    "status": complete_ledger_payload.get("status"),
                    "complete_control_ebpw": exact.get("complete_ebpw"),
                    "runtime_required_bytes": exact.get("runtime_required_bytes"),
                    "profile_source_bytes_read": profiled.get("source_bytes_read"),
                    "compact_ebpw": profiled.get("complete_ebpw"),
                    "promotion_allowed": complete_ledger_payload.get("promotion_allowed"),
                    "claim": complete_ledger_payload.get("claim_boundary"),
                }
            bank_screen = rec / "FLASH_DOCTOR_EXPERT_BANK_SCREEN.json"
            try:
                bank_screen_payload = json.loads(bank_screen.read_text())
            except (OSError, json.JSONDecodeError):
                bank_screen_payload = None
            if isinstance(bank_screen_payload, dict) and bank_screen_payload.get("schema") == "hawking.flash.doctor_expert_bank_screen.v1":
                bank_ref = str(bank_screen.relative_to(HAWKING))
                if bank_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {bank_ref}"
                if bank_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(bank_ref)
                population = bank_screen_payload.get("population") or {}
                rank = population.get("sampled_population_rank") or {}
                textbook["doctor_expert_bank_screen"] = {
                    "receipt": bank_ref,
                    "status": bank_screen_payload.get("status"),
                    "experts_sampled": len((bank_screen_payload.get("source") or {}).get("experts_sampled") or []),
                    "mean_cross_expert_cosine": population.get("cross_expert_gate_up_mean_cosine"),
                    "min_cross_expert_cosine": population.get("cross_expert_gate_up_min_cosine"),
                    "rank8_energy": rank.get("rank_8_energy"),
                    "claim": bank_screen_payload.get("claim_boundary"),
                }
            bank_screen_l7 = rec / "FLASH_DOCTOR_EXPERT_BANK_SCREEN_L7_128.json"
            try:
                bank_screen_l7_payload = json.loads(bank_screen_l7.read_text())
            except (OSError, json.JSONDecodeError):
                bank_screen_l7_payload = None
            if isinstance(bank_screen_l7_payload, dict) and bank_screen_l7_payload.get("schema") == "hawking.flash.doctor_expert_bank_screen.v1":
                bank_l7_ref = str(bank_screen_l7.relative_to(HAWKING))
                if bank_l7_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {bank_l7_ref}"
                if bank_l7_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(bank_l7_ref)
                population = bank_screen_l7_payload.get("population") or {}
                rank = population.get("sampled_population_rank") or {}
                textbook["doctor_expert_bank_screen_l7_128"] = {
                    "receipt": bank_l7_ref,
                    "status": bank_screen_l7_payload.get("status"),
                    "experts_sampled": len((bank_screen_l7_payload.get("source") or {}).get("experts_sampled") or []),
                    "mean_cross_expert_cosine": population.get("cross_expert_gate_up_mean_cosine"),
                    "min_cross_expert_cosine": population.get("cross_expert_gate_up_min_cosine"),
                    "rank8_energy": rank.get("rank_8_energy"),
                    "claim": bank_screen_l7_payload.get("claim_boundary"),
                }
            bank_screen_full = rec / "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json"
            try:
                bank_screen_full_payload = json.loads(bank_screen_full.read_text())
            except (OSError, json.JSONDecodeError):
                bank_screen_full_payload = None
            if (isinstance(bank_screen_full_payload, dict)
                    and bank_screen_full_payload.get("schema") == "hawking.flash.doctor_expert_bank_screen.v1"):
                bank_full_ref = str(bank_screen_full.relative_to(HAWKING))
                if bank_full_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {bank_full_ref}"
                if bank_full_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(bank_full_ref)
                population = bank_screen_full_payload.get("population") or {}
                rank = population.get("sampled_population_rank") or {}
                textbook["doctor_expert_bank_screen_full_l44"] = {
                    "receipt": bank_full_ref,
                    "status": bank_screen_full_payload.get("status"),
                    "experts_sampled": len((bank_screen_full_payload.get("source") or {}).get("experts_sampled") or []),
                    "mean_cross_expert_cosine": population.get("cross_expert_gate_up_mean_cosine"),
                    "min_cross_expert_cosine": population.get("cross_expert_gate_up_min_cosine"),
                    "rank8_energy": rank.get("rank_8_energy"),
                    "claim": bank_screen_full_payload.get("claim_boundary"),
                }
            ngram_screen = rec / "FLASH_DOCTOR_NGRAM_SCREEN.json"
            try:
                ngram_screen_payload = json.loads(ngram_screen.read_text())
            except (OSError, json.JSONDecodeError):
                ngram_screen_payload = None
            if isinstance(ngram_screen_payload, dict) and ngram_screen_payload.get("schema") == "hawking.flash.doctor_ngram_screen.v1":
                ngram_ref = str(ngram_screen.relative_to(HAWKING))
                if ngram_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {ngram_ref}"
                if ngram_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(ngram_ref)
                population = ngram_screen_payload.get("population") or {}
                textbook["doctor_ngram_screen"] = {
                    "receipt": ngram_ref,
                    "status": ngram_screen_payload.get("status"),
                    "shards": (ngram_screen_payload.get("source") or {}).get("shards"),
                    "mean_pairwise_row_cosine": population.get("mean_pairwise_row_cosine"),
                    "rank8_energy": population.get("rank8_energy"),
                    "claim": ngram_screen_payload.get("claim_boundary"),
                }
            # Direct routed accumulation is a reusable Accelerator law only
            # after it has repeated at more than one full-attention boundary.
            # Keep the organ receipts attached to the textbook without
            # pretending that dispatch/GPU savings are complete-token evidence.
            fused_route_rows = []
            for fused_route in sorted(rec.glob("FLASH_FUSED_ROUTE_ACCUMULATE_PARITY_L*.json")):
                try:
                    fused_route_payload = json.loads(fused_route.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if fused_route_payload.get("schema") != "hawking.flash.fused_route_accumulate_parity.v1":
                    continue
                fused_route_ref = str(fused_route.relative_to(HAWKING))
                if fused_route_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fused_route_ref}"
                if fused_route_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fused_route_ref)
                delta = fused_route_payload.get("physical_delta") or {}
                comparison = fused_route_payload.get("comparison") or {}
                fused_route_rows.append({
                    "receipt": fused_route_ref,
                    "layer": fused_route_payload.get("layer"),
                    "status": fused_route_payload.get("status"),
                    "dispatches_saved": delta.get("dispatches_saved"),
                    "gpu_reduction_fraction": delta.get("gpu_reduction_fraction"),
                    "route_ids_match": comparison.get("route_ids_match"),
                    "state_max_abs_error": (comparison.get("final_state") or {}).get("max_abs_error"),
                    "promotion_allowed": fused_route_payload.get("promotion_allowed"),
                    "claim": fused_route_payload.get("claim_boundary"),
                })
            if fused_route_rows:
                textbook["direct_route_accumulation"] = {
                    "receipts": fused_route_rows,
                    "reusable_after_repeated_organ_parity": len(fused_route_rows) >= 2,
                    "promotion_allowed": False,
                }
            archetype_screen = rec / "FLASH_ROUTE_ARCHETYPE_SPARSE_SCREEN_L4.json"
            try:
                archetype_payload = json.loads(archetype_screen.read_text())
            except (OSError, json.JSONDecodeError):
                archetype_payload = None
            if isinstance(archetype_payload, dict) and archetype_payload.get("schema") == "hawking.flash.route_archetype_sparse_screen.v1":
                archetype_ref = str(archetype_screen.relative_to(HAWKING))
                if archetype_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {archetype_ref}"
                if archetype_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(archetype_ref)
                rows = archetype_payload.get("rows") if isinstance(archetype_payload.get("rows"), list) else []
                textbook["doctor_archetype_sparse_screen"] = {
                    "receipt": archetype_ref,
                    "status": archetype_payload.get("status"),
                    "rows": len(rows),
                    "frontier_rows": len(archetype_payload.get("frontier") or []),
                    "beats_q4_rows": sum(1 for row in rows if isinstance(row, dict) and row.get("beats_q4_function") is True),
                    "claim": archetype_payload.get("claim_boundary"),
                }
            full_layers = [layer for layer in range(latest_layer + 1) if layer in {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}]
            suffix = "_".join(f"L{layer}" for layer in full_layers)
            textbook["status"] = f"BOUNDED_SOURCE_PARITY_L0_L{latest_layer}_WITH_{suffix}_FULL_ATTENTION_MOE_ORGANS"
            textbook["evidence"] = str(latest_receipt.relative_to(HAWKING))
            textbook["state_handoff"] = str(latest_state.relative_to(HAWKING))
            next_layer = latest_layer + 1
            if next_layer <= 47:
                next_kind = "full-attention/MoE" if next_layer in {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47} else "linear-attention"
                textbook["next_boundary"] = f"implement and verify the layer-{next_layer} {next_kind} graph"
            else:
                textbook["next_boundary"] = "attempt the first complete native token from the verified 48-layer Flash chain"
            for _, _, receipt in sorted(flash_receipts):
                receipt_ref = str(receipt.relative_to(HAWKING))
                if receipt_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {receipt_ref}"
                if receipt_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(receipt_ref)
            # Keep measured dispatch reductions attached to the active Flash
            # textbook without allowing an optimization receipt to masquerade
            # as a new layer boundary.
            fused_receipt = rec / "FLASH_NOETIC_COMPLETE_LAYER0_FUSED_SWIGLU.json"
            try:
                fused_payload = json.loads(fused_receipt.read_text())
            except (OSError, json.JSONDecodeError):
                fused_payload = None
            if isinstance(fused_payload, dict) and fused_payload.get("status") == "PASSED":
                fused_ref = str(fused_receipt.relative_to(HAWKING))
                if fused_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fused_ref}"
                if fused_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fused_ref)
                textbook["dispatch_optimization"] = {
                    "receipt": fused_ref,
                    "change": "shared-expert gate GEMV + up GEMV + SwiGLU fused",
                    "dispatches": fused_payload.get("execution", {}).get("dispatches"),
                    "parity": fused_payload.get("parity", {}).get("passed"),
                    "claim": "layer-0 source-BF16 capability result; not complete-token throughput",
                }
            dual_receipt = rec / "FLASH_NOETIC_COMPLETE_LAYER0_DUAL_PROJ.json"
            try:
                dual_payload = json.loads(dual_receipt.read_text())
            except (OSError, json.JSONDecodeError):
                dual_payload = None
            if isinstance(dual_payload, dict) and dual_payload.get("status") == "PASSED":
                dual_ref = str(dual_receipt.relative_to(HAWKING))
                if dual_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {dual_ref}"
                if dual_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(dual_ref)
                previous = textbook.get("dispatch_optimization", {})
                textbook["dispatch_optimization"] = {
                    "receipt": dual_ref,
                    "change": "fused shared-expert SwiGLU and paired source-BF16 projection groups",
                    "dispatches": dual_payload.get("execution", {}).get("dispatches"),
                    "baseline_dispatches": dual_payload.get("dispatch_optimization", {}).get("baseline_dispatches", 35),
                    "dispatch_reduction": dual_payload.get("dispatch_optimization", {}).get("dispatch_reduction", 5),
                    "parity": dual_payload.get("parity", {}).get("passed"),
                    "prior_receipt": previous.get("receipt"),
                    "claim": "layer-0 source-BF16 capability result; not complete-token throughput",
                }
            fused_moe = rec / "FLASH_NOETIC_COMPLETE_LAYER0_FUSED_MOE_EPILOGUE.json"
            try:
                fused_moe_payload = json.loads(fused_moe.read_text())
            except (OSError, json.JSONDecodeError):
                fused_moe_payload = None
            if isinstance(fused_moe_payload, dict) and fused_moe_payload.get("status") == "PASSED":
                fused_moe_ref = str(fused_moe.relative_to(HAWKING))
                if fused_moe_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fused_moe_ref}"
                if fused_moe_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fused_moe_ref)
                textbook["dispatch_optimization"] = {
                    "receipt": fused_moe_ref,
                    "change": "routed sum + shared sigmoid gate + MoE add fused",
                    "dispatches": fused_moe_payload.get("execution", {}).get("dispatches"),
                    "baseline_dispatches": 30,
                    "dispatch_reduction": 2,
                    "parity": fused_moe_payload.get("parity", {}).get("passed"),
                    "output_hash": fused_moe_payload.get("parity", {}).get("final_output_hash"),
                    "claim": "layer-0 source-BF16 capability result; fused MoE epilogue is physically parity-verified, not complete-token throughput",
                }
                optimization_rows = textbook.setdefault("dispatch_optimization_evidence", [])
                if not any(row.get("receipt") == fused_moe_ref for row in optimization_rows if isinstance(row, dict)):
                    optimization_rows.append({
                        "receipt": fused_moe_ref,
                        "dispatches": fused_moe_payload.get("execution", {}).get("dispatches"),
                        "baseline_dispatches": 30,
                        "dispatch_reduction": 2,
                        "parity": fused_moe_payload.get("parity", {}).get("passed"),
                    })
            fused_hc = rec / "FLASH_NOETIC_COMPLETE_LAYER0_FUSED_HC_INPUT.json"
            try:
                fused_hc_payload = json.loads(fused_hc.read_text())
            except (OSError, json.JSONDecodeError):
                fused_hc_payload = None
            if isinstance(fused_hc_payload, dict) and fused_hc_payload.get("status") == "PASSED":
                fused_hc_ref = str(fused_hc.relative_to(HAWKING))
                if fused_hc_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fused_hc_ref}"
                if fused_hc_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fused_hc_ref)
                textbook["dispatch_optimization"] = {
                    "receipt": fused_hc_ref,
                    "change": "two five-stage HyperConnection input organs fused into staged superkernels",
                    "dispatches": fused_hc_payload.get("execution", {}).get("dispatches"),
                    "baseline_dispatches": 28,
                    "dispatch_reduction": 8,
                    "parity": fused_hc_payload.get("parity", {}).get("passed"),
                    "output_hash": fused_hc_payload.get("parity", {}).get("final_output_hash"),
                    "graph_gpu_ns": fused_hc_payload.get("execution", {}).get("graph_gpu_ns"),
                    "useful_work_gate": "INCONCLUSIVE_CONCURRENT_GPU_CHAIN_RETAIN_28_DISPATCH_WINNER",
                    "claim": "layer-0 source-BF16 capability result; HC superkernel is physically parity-verified, not complete-token throughput",
                }
                optimization_rows = textbook.setdefault("dispatch_optimization_evidence", [])
                if not any(row.get("receipt") == fused_hc_ref for row in optimization_rows if isinstance(row, dict)):
                    optimization_rows.append({
                        "receipt": fused_hc_ref,
                        "dispatches": fused_hc_payload.get("execution", {}).get("dispatches"),
                        "baseline_dispatches": 28,
                        "dispatch_reduction": 8,
                        "parity": fused_hc_payload.get("parity", {}).get("passed"),
                    })
            # Keep the aggressive single-digit target explicit as a plan, not as
            # an unmeasured capability claim.  The plan is evidence-linked and
            # remains NX-ineligible until each superkernel is physically run.
            single_digit_plan = rec / "FLASH_SINGLE_DIGIT_DISPATCH_PLAN.json"
            try:
                single_digit_payload = json.loads(single_digit_plan.read_text())
            except (OSError, json.JSONDecodeError):
                single_digit_payload = None
            if isinstance(single_digit_payload, dict):
                plan_ref = str(single_digit_plan.relative_to(HAWKING))
                if plan_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {plan_ref}"
                if plan_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(plan_ref)
                textbook["single_digit_dispatch_target"] = {
                    "receipt": plan_ref,
                    "current_measured_dispatches": single_digit_payload.get("current_measured", {}).get("dispatches"),
                    "target_dispatches": single_digit_payload.get("target", {}).get("dispatches"),
                    "under_single_digit": single_digit_payload.get("target", {}).get("under_single_digit"),
                    "status": "PLAN_ONLY",
                    "promotion_allowed": False,
                    "claim": "superkernel target; no dispatch-count-only relabeling",
                }
            range_candidates = sorted(rec.glob("FLASH_SINGLE_PROCESS_CHAIN_L0_RANGE_*/layer-0/receipt.json"), key=str)
            range_receipt = range_candidates[-1] if range_candidates else rec / "FLASH_SINGLE_PROCESS_CHAIN_L0_RANGE_V2" / "layer-0" / "receipt.json"
            try:
                range_payload = json.loads(range_receipt.read_text())
            except (OSError, json.JSONDecodeError):
                range_payload = None
            if isinstance(range_payload, dict) and range_payload.get("status") == "PASSED":
                range_ref = str(range_receipt.relative_to(HAWKING))
                if range_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {range_ref}"
                if range_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(range_ref)
                embedding = range_payload.get("source", {}).get("embedding", {})
                textbook["input_io_optimization"] = {
                    "receipt": range_ref,
                    "read_mode": embedding.get("read_mode"),
                    "embedding_bytes_read": embedding.get("source_bytes"),
                    "logical_embedding_bytes": embedding.get("logical_tensor_bytes"),
                    "output_hash": range_payload.get("parity", {}).get("final_output_hash"),
                    "parity": range_payload.get("parity", {}).get("passed"),
                    "claim": "BOS row range-read preserves exact layer-0 output while avoiding full embedding-table materialization",
                }
            fused_prefix = rec / "FLASH_NOETIC_MULTILAYER_LINEAR_PREFIX_L0_L2_FUSED_SWIGLU.json"
            try:
                fused_prefix_payload = json.loads(fused_prefix.read_text())
            except (OSError, json.JSONDecodeError):
                fused_prefix_payload = None
            if isinstance(fused_prefix_payload, dict) and fused_prefix_payload.get("status") == "PASSED_LINEAR_PREFIX_BLOCKED_AT_FULL_ATTENTION":
                fused_prefix_ref = str(fused_prefix.relative_to(HAWKING))
                if fused_prefix_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fused_prefix_ref}"
                if fused_prefix_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fused_prefix_ref)
            fused_full = rec / "FLASH_NOETIC_FULL_ATTENTION_LAYER47_FUSED_SWIGLU.json"
            try:
                fused_full_payload = json.loads(fused_full.read_text())
            except (OSError, json.JSONDecodeError):
                fused_full_payload = None
            if isinstance(fused_full_payload, dict) and fused_full_payload.get("status") == "PASSED":
                fused_full_ref = str(fused_full.relative_to(HAWKING))
                if fused_full_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fused_full_ref}"
                if fused_full_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fused_full_ref)
            dual_full = rec / "FLASH_NOETIC_FULL_ATTENTION_LAYER47_DUAL_PROJ.json"
            try:
                dual_full_payload = json.loads(dual_full.read_text())
            except (OSError, json.JSONDecodeError):
                dual_full_payload = None
            if isinstance(dual_full_payload, dict) and dual_full_payload.get("status") == "PASSED":
                dual_full_ref = str(dual_full.relative_to(HAWKING))
                if dual_full_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {dual_full_ref}"
                if dual_full_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(dual_full_ref)
                optimization_rows = textbook.setdefault("dispatch_optimization_evidence", [])
                if not any(row.get("receipt") == dual_full_ref for row in optimization_rows if isinstance(row, dict)):
                    optimization_rows.append({
                        "receipt": dual_full_ref,
                        "dispatches": dual_full_payload.get("execution", {}).get("dispatches"),
                        "baseline_dispatches": dual_full_payload.get("dispatch_optimization", {}).get("baseline_dispatches"),
                        "dispatch_reduction": dual_full_payload.get("dispatch_optimization", {}).get("dispatch_reduction"),
                        "parity": dual_full_payload.get("dispatch_optimization", {}).get("parity_preserved"),
                    })
            token_attempt = rec / "FLASH_COMPLETE_TOKEN_NATIVE_ATTEMPT.json"
            try:
                token_attempt_payload = json.loads(token_attempt.read_text())
            except (OSError, json.JSONDecodeError):
                token_attempt_payload = None
            if isinstance(token_attempt_payload, dict):
                token_ref = str(token_attempt.relative_to(HAWKING))
                if token_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {token_ref}"
                if token_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(token_ref)
                textbook["complete_token_attempt"] = {
                    "receipt": token_ref,
                    "status": token_attempt_payload.get("status"),
                    "first_failure_stage": token_attempt_payload.get("first_physical_failure_boundary", {}).get("stage"),
                    "next_action": token_attempt_payload.get("first_physical_failure_boundary", {}).get("next_action"),
                }
            tokenizer_contract = rec / "FLASH_TOKENIZER_ACCEPTANCE_CONTRACT.json"
            try:
                tokenizer_contract_payload = json.loads(tokenizer_contract.read_text())
            except (OSError, json.JSONDecodeError):
                tokenizer_contract_payload = None
            if isinstance(tokenizer_contract_payload, dict) and tokenizer_contract_payload.get("schema") == "hawking.flash.tokenizer_acceptance_contract.v1":
                tokenizer_ref = str(tokenizer_contract.relative_to(HAWKING))
                if tokenizer_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {tokenizer_ref}"
                if tokenizer_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(tokenizer_ref)
                textbook["tokenizer_acceptance_contract"] = {
                    "receipt": tokenizer_ref,
                    "status": tokenizer_contract_payload.get("status"),
                    "vocab_size": (tokenizer_contract_payload.get("tokenizer") or {}).get("vocab_size"),
                    "prompt_encoded_count": (tokenizer_contract_payload.get("prompt") or {}).get("encoded_count"),
                    "terminal_token_compatible": (tokenizer_contract_payload.get("terminal_token_contract") or {}).get("native_terminal_token_compatible"),
                    "model_forward_executed": (tokenizer_contract_payload.get("execution") or {}).get("model_forward_executed"),
                    "claim": tokenizer_contract_payload.get("claim_boundary"),
                }
            terminal_token = rec / "FLASH_SOURCE_BF16_TERMINAL_TOKEN.json"
            try:
                terminal_payload = json.loads(terminal_token.read_text())
            except (OSError, json.JSONDecodeError):
                terminal_payload = None
            if isinstance(terminal_payload, dict) and terminal_payload.get("status") == "PASSED":
                terminal_ref = str(terminal_token.relative_to(HAWKING))
                if terminal_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {terminal_ref}"
                if terminal_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(terminal_ref)
                textbook["complete_token_terminal_probe"] = {
                    "receipt": terminal_ref,
                    "token_id": terminal_payload.get("terminal", {}).get("token_id"),
                    "dispatches": terminal_payload.get("execution", {}).get("dispatches"),
                    "fallback_count": terminal_payload.get("execution", {}).get("fallback_count"),
                    "claim": "first token terminal probe; one-process runtime and throughput qualification remain open",
                }
            terminal_metal = rec / "FLASH_SOURCE_BF16_TERMINAL_TOKEN_METAL_READOUT.json"
            try:
                terminal_metal_payload = json.loads(terminal_metal.read_text())
            except (OSError, json.JSONDecodeError):
                terminal_metal_payload = None
            if isinstance(terminal_metal_payload, dict) and terminal_metal_payload.get("status") == "PASSED":
                terminal_metal_ref = str(terminal_metal.relative_to(HAWKING))
                if terminal_metal_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {terminal_metal_ref}"
                if terminal_metal_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(terminal_metal_ref)
                textbook["complete_token_terminal_metal_readout"] = {
                    "receipt": terminal_metal_ref,
                    "token_id": terminal_metal_payload.get("terminal", {}).get("token_id"),
                    "dispatches": terminal_metal_payload.get("execution", {}).get("dispatches"),
                    "command_buffers": terminal_metal_payload.get("execution", {}).get("command_buffers"),
                    "readout_parity": terminal_metal_payload.get("parity", {}).get("hyperconnection_readout"),
                    "claim": "terminal readout and lm_head execute in one Metal command buffer; full 48-layer one-process runtime remains open",
                }
            chain_summary = rec / "FLASH_SINGLE_PROCESS_CHAIN_L0_L3" / "CHAIN_SUMMARY.json"
            try:
                chain_payload = json.loads(chain_summary.read_text())
            except (OSError, json.JSONDecodeError):
                chain_payload = None
            if isinstance(chain_payload, dict) and chain_payload.get("status") == "PASSED":
                chain_ref = str(chain_summary.relative_to(HAWKING))
                if chain_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {chain_ref}"
                if chain_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(chain_ref)
                chain_rows = chain_payload.get("layers", [])
                chain_dispatches = 0
                chain_gpu_ns = 0
                chain_receipts_ok = True
                for row in chain_rows:
                    if not isinstance(row, dict):
                        continue
                    try:
                        layer_receipt = HAWKING / row["receipt"]
                        layer_payload = json.loads(layer_receipt.read_text())
                        execution = layer_payload.get("execution", {})
                        chain_dispatches += int(execution.get("dispatches", 0))
                        graph_gpu = execution.get("graph_gpu_ns")
                        chain_gpu_ns += int(sum(graph_gpu) if isinstance(graph_gpu, list) and graph_gpu else execution.get("gpu_ns", 0))
                        chain_receipts_ok = chain_receipts_ok and layer_payload.get("status") == "PASSED"
                    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
                        chain_receipts_ok = False
                textbook["single_process_chain"] = {
                    "receipt": chain_ref,
                    "start_layer": chain_payload.get("start_layer"),
                    "end_layer": chain_payload.get("end_layer"),
                    "dispatches": chain_dispatches,
                    "gpu_ns": chain_gpu_ns,
                    "layer_receipts_passed": chain_receipts_ok,
                    "process_boundary": chain_payload.get("process_boundary"),
                    "state_handoff": chain_payload.get("state_handoff"),
                    "device_residency": chain_payload.get("device_residency"),
                    "complete_token": chain_payload.get("complete_token"),
                    "claim": "single OS process across layers 0..3 with explicit host state snapshots; streamed device-resident whole-token runtime remains open",
                }
            # Preserve additional resumable single-process ranges without
            # replacing the canonical L0..L3 seam. Each range is evidence of
            # process continuity only; explicit host snapshots remain a hard
            # claim boundary until the streamed executor exists.
            chain_ranges = textbook.setdefault("single_process_chain_ranges", [])
            for extra_summary in sorted(rec.glob("FLASH_SINGLE_PROCESS_CHAIN_*/CHAIN_SUMMARY.json"), key=str):
                if extra_summary == chain_summary:
                    continue
                try:
                    extra_payload = json.loads(extra_summary.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if extra_payload.get("status") != "PASSED":
                    continue
                extra_ref = str(extra_summary.relative_to(HAWKING))
                if extra_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {extra_ref}"
                if extra_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(extra_ref)
                extra_dispatches = 0
                extra_gpu_ns = 0
                extra_ok = True
                for row in extra_payload.get("layers", []):
                    if not isinstance(row, dict):
                        continue
                    try:
                        layer_payload = json.loads((HAWKING / row["receipt"]).read_text())
                        execution = layer_payload.get("execution", {})
                        extra_dispatches += int(execution.get("dispatches", 0))
                        graph_gpu = execution.get("graph_gpu_ns")
                        extra_gpu_ns += int(sum(graph_gpu) if isinstance(graph_gpu, list) and graph_gpu else execution.get("gpu_ns", 0))
                        extra_ok = extra_ok and layer_payload.get("status") == "PASSED"
                    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
                        extra_ok = False
                range_row = {
                    "receipt": extra_ref,
                    "start_layer": extra_payload.get("start_layer"),
                    "end_layer": extra_payload.get("end_layer"),
                    "dispatches": extra_dispatches,
                    "gpu_ns": extra_gpu_ns,
                    "layer_receipts_passed": extra_ok,
                    "process_boundary": extra_payload.get("process_boundary"),
                    "state_handoff": extra_payload.get("state_handoff"),
                    "device_residency": extra_payload.get("device_residency"),
                    "complete_token": extra_payload.get("complete_token"),
                }
                # A terminal probe may be attached to any resumed range (for
                # example L44-L47), rather than the historical canonical
                # terminal receipt at the root of headless/.  Register that
                # receipt explicitly so the ledger reflects the newest
                # physically received token without widening its claim to a
                # streamed resident runtime.
                terminal_rel = extra_payload.get("terminal_receipt")
                if isinstance(terminal_rel, str) and terminal_rel:
                    terminal_path = HAWKING / terminal_rel
                    try:
                        extra_terminal_payload = json.loads(terminal_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        extra_terminal_payload = None
                    if isinstance(extra_terminal_payload, dict) and extra_terminal_payload.get("status") == "PASSED":
                        terminal_ref = str(terminal_path.relative_to(HAWKING))
                        if terminal_ref not in textbook["cross_layer_evidence"]:
                            textbook["cross_layer_evidence"] += f" + {terminal_ref}"
                        if terminal_ref not in state["accepted_receipts"]:
                            state["accepted_receipts"].append(terminal_ref)
                        textbook["single_process_terminal_probe"] = {
                            "chain_receipt": extra_ref,
                            "receipt": terminal_ref,
                            "token_id": extra_terminal_payload.get("terminal", {}).get("token_id"),
                            "dispatches": extra_terminal_payload.get("execution", {}).get("dispatches"),
                            "command_buffers": extra_terminal_payload.get("execution", {}).get("command_buffers"),
                            "fallback_count": extra_terminal_payload.get("execution", {}).get("fallback_count"),
                            "readout_parity": extra_terminal_payload.get("parity", {}).get("hyperconnection_readout"),
                            "claim": (
                                f"first token terminal probe after a single-process "
                                f"L{extra_payload.get('start_layer')}..L{extra_payload.get('end_layer')} seam; "
                                "explicit host state handoffs remain; streamed device-resident runtime, "
                                "TPS, EBPW, and residency remain open"
                            ),
                        }
                chain_ranges[:] = [row for row in chain_ranges if isinstance(row, dict) and row.get("receipt") != extra_ref]
                chain_ranges.append(range_row)

            # Register the accelerated continuation separately from the
            # historical layer-by-layer chains.  This is evidence of a
            # long-lived process and grouped execution, but its explicit host
            # checkpoint seam must remain visible until device-resident state
            # is physically qualified.
            fast_summary = rec / "FLASH_FAST_CHAIN_L44_L47_V2" / "FAST_CHAIN_SUMMARY.json"
            try:
                fast_payload = json.loads(fast_summary.read_text())
            except (OSError, json.JSONDecodeError):
                fast_payload = None
            if isinstance(fast_payload, dict) and fast_payload.get("status") == "PASSED":
                fast_ref = str(fast_summary.relative_to(HAWKING))
                if fast_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fast_ref}"
                if fast_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fast_ref)
                fast_groups = fast_payload.get("groups", [])
                fast_layers = [
                    layer
                    for group in fast_groups
                    if isinstance(group, dict)
                    for layer in group.get("layers", [])
                    if isinstance(layer, dict)
                ]
                fast_full_layers = []
                fast_group_wall_ns = 0
                for group in fast_groups:
                    if not isinstance(group, dict):
                        continue
                    try:
                        group_receipt = HAWKING / str(group["receipt"])
                        group_payload = json.loads(group_receipt.read_text())
                    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
                        continue
                    group_wall = group_payload.get("elapsed_wall_ns")
                    if not isinstance(group_wall, (int, float)):
                        group_wall = group_payload.get("execution", {}).get("wall_ns")
                    if isinstance(group_wall, (int, float)):
                        fast_group_wall_ns += int(group_wall)
                    if group.get("kind") == "full_attention" and group_payload.get("status") == "PASSED":
                        fast_full_layers.append(group.get("start_layer"))
                # Compare the same layer interval against the completed golden
                # chain.  A lower dispatch count is not a speed win by itself.
                baseline_wall_ns = 0
                golden_summary = rec / "FLASH_SINGLE_PROCESS_CHAIN_L0_L47" / "CHAIN_SUMMARY.json"
                try:
                    golden_payload = json.loads(golden_summary.read_text())
                except (OSError, json.JSONDecodeError):
                    golden_payload = None
                if isinstance(golden_payload, dict):
                    for row in golden_payload.get("layers", []):
                        if not isinstance(row, dict) or not (44 <= int(row.get("layer", -1)) <= 47):
                            continue
                        try:
                            golden_receipt = HAWKING / str(row["receipt"])
                            golden_layer = json.loads(golden_receipt.read_text())
                        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
                            continue
                        value = golden_layer.get("elapsed_wall_ns")
                        if not isinstance(value, (int, float)):
                            value = golden_layer.get("execution", {}).get("wall_ns")
                        if isinstance(value, (int, float)):
                            baseline_wall_ns += int(value)
                textbook["fast_executor"] = {
                    "receipt": fast_ref,
                    "start_layer": fast_payload.get("start_layer"),
                    "end_layer": fast_payload.get("end_layer"),
                    "layers_passed": [layer.get("layer") for layer in fast_layers] + fast_full_layers,
                    "linear_dispatches": sum(int(layer.get("dispatches", 0)) for layer in fast_layers),
                    "process_boundary": fast_payload.get("process_boundary"),
                    "source_index": fast_payload.get("source_index"),
                    "metal_context": fast_payload.get("metal_context"),
                    "state_handoff": fast_payload.get("state_handoff"),
                    "elapsed_wall_ns": fast_payload.get("elapsed_wall_ns"),
                    "measured_group_wall_ns": fast_group_wall_ns,
                    "golden_overlap_wall_ns": baseline_wall_ns,
                    "wall_speedup_ratio": (baseline_wall_ns / fast_group_wall_ns) if fast_group_wall_ns else None,
                    "wall_speedup_status": (
                        "FASTER_THAN_GOLDEN_OVERLAP" if fast_group_wall_ns and baseline_wall_ns and fast_group_wall_ns < baseline_wall_ns
                        else "NO_SPEEDUP_YET" if fast_group_wall_ns and baseline_wall_ns
                        else "UNCOMPARABLE"
                    ),
                    "claim": fast_payload.get("claim_boundary"),
                }

            # Current fused/compact chain: keep it distinct from the older
            # fast-executor receipts so one-dispatch MoE savings and the
            # complete-forward terminal probe are visible without promoting
            # them to accepted generation or residency.  This is deliberately
            # path-tolerant so reruns can supersede the evidence by changing
            # only the output directory name.
            fused_summary = rec / "FLASH_FAST_FUSED_L0_L47_V1" / "FAST_CHAIN_SUMMARY.json"
            try:
                fused_payload = json.loads(fused_summary.read_text())
            except (OSError, json.JSONDecodeError):
                fused_payload = None
            if isinstance(fused_payload, dict) and fused_payload.get("schema") == "hawking.flash_fast_chain.v1" and fused_payload.get("status") == "PASSED":
                fused_ref = str(fused_summary.relative_to(HAWKING))
                if fused_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fused_ref}"
                if fused_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fused_ref)
                fused_groups = fused_payload.get("groups", [])
                fused_source_ns = 0
                fused_gpu_ns = 0
                fused_dispatches = 0
                fused_layers = []
                for group in fused_groups:
                    if not isinstance(group, dict):
                        continue
                    group_receipt = group.get("receipt")
                    try:
                        payload = json.loads((HAWKING / str(group_receipt)).read_text())
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        payload = {}
                    if group.get("kind") == "linear_attention":
                        rows = payload.get("layers", []) if isinstance(payload, dict) else []
                        for row in rows:
                            if not isinstance(row, dict):
                                continue
                            fused_layers.append(row.get("layer"))
                            fused_source_ns += int(row.get("source_load_ns", 0) or 0)
                            gpu_values = row.get("gpu_ns", [])
                            if isinstance(gpu_values, list):
                                fused_gpu_ns += int(sum(value for value in gpu_values if isinstance(value, (int, float))))
                            fused_dispatches += int(row.get("dispatches", 0) or 0)
                    elif group.get("kind") == "full_attention" and isinstance(payload, dict):
                        fused_layers.append(group.get("start_layer"))
                        execution = payload.get("execution", {})
                        fused_gpu_ns += int(execution.get("gpu_ns", 0) or 0)
                        fused_dispatches += int(execution.get("dispatches", 0) or 0)
                        fused_source_ns += int(payload.get("timing", {}).get("source_load_ns", 0) or 0)
                terminal_rel = fused_payload.get("terminal_receipt")
                terminal_payload = None
                if isinstance(terminal_rel, str) and terminal_rel:
                    terminal_path = HAWKING / terminal_rel
                    try:
                        terminal_payload = json.loads(terminal_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        terminal_payload = None
                    if isinstance(terminal_payload, dict) and terminal_payload.get("status") == "PASSED":
                        terminal_ref = str(terminal_path.relative_to(HAWKING))
                        if terminal_ref not in textbook["cross_layer_evidence"]:
                            textbook["cross_layer_evidence"] += f" + {terminal_ref}"
                        if terminal_ref not in state["accepted_receipts"]:
                            state["accepted_receipts"].append(terminal_ref)
                textbook["fused_device_resident_complete_forward"] = {
                    "receipt": fused_ref,
                    "terminal_receipt": str((HAWKING / terminal_rel).relative_to(HAWKING)) if isinstance(terminal_rel, str) and terminal_rel and (HAWKING / terminal_rel).exists() else None,
                    "start_layer": fused_payload.get("start_layer"),
                    "end_layer": fused_payload.get("end_layer"),
                    "layers": fused_layers,
                    "process_boundary": fused_payload.get("process_boundary"),
                    "device_resident": fused_payload.get("device_resident"),
                    "compact_experts": fused_payload.get("compact_experts"),
                    "fused_route_accumulate": True,
                    "elapsed_wall_ns": fused_payload.get("elapsed_wall_ns"),
                    "source_load_ns_observed": fused_source_ns,
                    "gpu_ns_observed": fused_gpu_ns,
                    "dispatches_observed": fused_dispatches,
                    "host_activation_roundtrips": 0,
                    "terminal_token_id": terminal_payload.get("terminal", {}).get("token_id") if isinstance(terminal_payload, dict) else None,
                    "promotion_allowed": False,
                    "claim": fused_payload.get("claim_boundary"),
                }

            # The OS-file-cache switch is a measured experiment, not a default
            # runtime law.  Register the latest warm/cold samples as negative
            # science when they do not improve complete wall time.
            cache_samples = []
            for cache_name in ("FLASH_FAST_CACHE_L0_L7_V1", "FLASH_FAST_CACHE_L0_L7_V2"):
                cache_summary = rec / cache_name / "FAST_CHAIN_SUMMARY.json"
                try:
                    cache_payload = json.loads(cache_summary.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(cache_payload, dict) and cache_payload.get("schema") == "hawking.flash_fast_chain.v1":
                    cache_samples.append({"receipt": str(cache_summary.relative_to(HAWKING)), "elapsed_wall_ns": cache_payload.get("elapsed_wall_ns"), "status": cache_payload.get("status")})
            if cache_samples:
                textbook["source_cache_experiment"] = {
                    "samples": cache_samples,
                    "policy": "HAWKING_SOURCE_CACHE=1",
                    "promotion_allowed": False,
                    "claim": "opt-in OS file-cache continuation was measured; it is not a default because the external-volume 0→7 samples were slower than the F_NOCACHE baseline",
                }

            # The isolated timed layer makes the wall-time denominator
            # actionable.  Keep it distinct from the grouped continuation so
            # a future source-reader change can be compared against the exact
            # same layer/state without conflating GPU or parity work.
            fast_timing_summary = rec / "FLASH_FAST_TIMING_L44_V3" / "FAST_CHAIN_SUMMARY.json"
            fast_timing_receipt = rec / "FLASH_FAST_TIMING_L44_V3" / "group-44-44" / "receipt.json"
            try:
                fast_timing_payload = json.loads(fast_timing_summary.read_text())
                fast_layer_payload = json.loads(fast_timing_receipt.read_text())
            except (OSError, json.JSONDecodeError):
                fast_timing_payload = None
                fast_layer_payload = None
            if isinstance(fast_timing_payload, dict) and isinstance(fast_layer_payload, dict) and fast_layer_payload.get("status") == "PASSED":
                timing_ref = str(fast_timing_summary.relative_to(HAWKING))
                layer_timing_ref = str(fast_timing_receipt.relative_to(HAWKING))
                for evidence_ref in (timing_ref, layer_timing_ref):
                    if evidence_ref not in textbook["cross_layer_evidence"]:
                        textbook["cross_layer_evidence"] += f" + {evidence_ref}"
                    if evidence_ref not in state["accepted_receipts"]:
                        state["accepted_receipts"].append(evidence_ref)
                execution = fast_layer_payload.get("execution", {})
                timing = fast_layer_payload.get("timing", {})
                wall_ns = fast_layer_payload.get("elapsed_wall_ns")
                source_load_ns = timing.get("source_load_ns")
                textbook["fast_executor_timing"] = {
                    "receipt": layer_timing_ref,
                    "summary": timing_ref,
                    "layer": 44,
                    "elapsed_wall_ns": wall_ns,
                    "source_bytes_read": fast_layer_payload.get("bytes", {}).get("source_payload_bytes_read"),
                    "source_load_ns": source_load_ns,
                    "device_prepare_ns": timing.get("device_prepare_ns"),
                    "graph_setup_ns": timing.get("graph_setup_ns"),
                    "cpu_oracle_ns": execution.get("source_cpu_oracle_ns"),
                    "gpu_execution_ns": sum(execution.get("graph_gpu_ns", [])) if isinstance(execution.get("graph_gpu_ns"), list) else execution.get("gpu_ns"),
                    "wall_source_fraction": (source_load_ns / wall_ns) if isinstance(source_load_ns, (int, float)) and isinstance(wall_ns, (int, float)) and wall_ns else None,
                    "next_target": "lazy/routed expert-bank reads or mmap-backed source buffers",
                    "claim": "isolated exact layer timing; no complete-token or residency promotion",
                }

            # The current linear-species timing sample complements the older
            # full-attention L44 profile.  Keep it separate so source-read
            # variance is visible rather than averaged across unlike organs.
            linear_timing_summary = rec / "FLASH_FAST_TIMING_LINEAR_L0_V1" / "FAST_CHAIN_SUMMARY.json"
            linear_timing_receipt = rec / "FLASH_FAST_TIMING_LINEAR_L0_V1" / "group-0-0" / "receipt.json"
            try:
                linear_summary_payload = json.loads(linear_timing_summary.read_text())
                linear_layer_payload = json.loads(linear_timing_receipt.read_text())
            except (OSError, json.JSONDecodeError):
                linear_summary_payload = None
                linear_layer_payload = None
            if (isinstance(linear_summary_payload, dict)
                    and isinstance(linear_layer_payload, dict)
                    and isinstance(linear_layer_payload.get("status"), str)
                    and linear_layer_payload.get("status", "").startswith("PASSED")):
                linear_summary_ref = str(linear_timing_summary.relative_to(HAWKING))
                linear_receipt_ref = str(linear_timing_receipt.relative_to(HAWKING))
                for evidence_ref in (linear_summary_ref, linear_receipt_ref):
                    if evidence_ref not in textbook["cross_layer_evidence"]:
                        textbook["cross_layer_evidence"] += f" + {evidence_ref}"
                    if evidence_ref not in state["accepted_receipts"]:
                        state["accepted_receipts"].append(evidence_ref)
                linear_execution = linear_layer_payload.get("execution", {})
                linear_timing = linear_layer_payload.get("timing", {})
                linear_rows = linear_layer_payload.get("layers", [])
                linear_row = linear_rows[0] if linear_rows and isinstance(linear_rows[0], dict) else {}
                linear_wall = linear_layer_payload.get("elapsed_wall_ns")
                linear_source_ns = linear_timing.get("source_load_ns")
                linear_gpu = linear_row.get("gpu_ns")
                if isinstance(linear_gpu, list):
                    linear_gpu = sum(value for value in linear_gpu if isinstance(value, (int, float)))
                textbook["linear_executor_timing"] = {
                    "receipt": linear_receipt_ref,
                    "summary": linear_summary_ref,
                    "layer": linear_row.get("layer", 0),
                    "elapsed_wall_ns": linear_wall,
                    "source_bytes_read": linear_layer_payload.get("source", {}).get("manifest", {}).get("bytes_read")
                    if isinstance(linear_layer_payload.get("source"), dict) else linear_execution.get("source_payload_bytes_read"),
                    "source_payload_bytes_read": linear_execution.get("source_payload_bytes_read"),
                    "source_load_ns": linear_source_ns,
                    "device_prepare_ns": linear_timing.get("device_prepare_ns"),
                    "graph_setup_ns": linear_timing.get("graph_setup_ns"),
                    "cpu_oracle_ns": linear_row.get("cpu_oracle_ns"),
                    "gpu_execution_ns": linear_gpu,
                    "dispatches": linear_execution.get("total_dispatches") or linear_row.get("dispatches"),
                    "command_buffers": linear_execution.get("total_command_buffers") or linear_row.get("command_buffers"),
                    "host_activation_roundtrips": linear_execution.get("host_activation_roundtrips"),
                    "source_cache_policy": linear_execution.get("source_cache_policy") or linear_timing.get("source_cache_policy"),
                    "wall_source_fraction": (linear_source_ns / linear_wall) if isinstance(linear_source_ns, (int, float)) and isinstance(linear_wall, (int, float)) and linear_wall else None,
                    "claim": "isolated exact linear-attention timing with device-resident handoff; no complete-token or residency promotion",
                }

            expert_bank_profile = rec / "FLASH_EXPERT_BANK_ROUTED_IO_PROFILE_L44.json"
            try:
                expert_bank_payload = json.loads(expert_bank_profile.read_text())
            except (OSError, json.JSONDecodeError):
                expert_bank_payload = None
            if isinstance(expert_bank_payload, dict) and expert_bank_payload.get("status") == "PROFILE_ONLY":
                expert_ref = str(expert_bank_profile.relative_to(HAWKING))
                if expert_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {expert_ref}"
                if expert_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(expert_ref)
                textbook["expert_bank_io_profile"] = {
                    "receipt": expert_ref,
                    "layer": expert_bank_payload.get("layer"),
                    "route_ids": expert_bank_payload.get("source", {}).get("route_ids"),
                    "full_bytes": expert_bank_payload.get("full_expert_bank_bytes"),
                    "selected_bytes": expert_bank_payload.get("selected_expert_bytes"),
                    "reduction_fraction": expert_bank_payload.get("reduction_fraction"),
                    "range_read_ns": expert_bank_payload.get("physical_range_read", {}).get("elapsed_ns"),
                    "next_gate": expert_bank_payload.get("next_gate"),
                    "claim": expert_bank_payload.get("claim_boundary"),
                }

            compact_dir = rec / "FLASH_COMPACT_L44_V3"
            if not (compact_dir / "FAST_CHAIN_SUMMARY.json").exists():
                compact_dir = rec / "FLASH_COMPACT_L44_V2"
            if not (compact_dir / "FAST_CHAIN_SUMMARY.json").exists():
                compact_dir = rec / "FLASH_COMPACT_L44_V1"
            compact_summary = compact_dir / "FAST_CHAIN_SUMMARY.json"
            compact_receipt = compact_dir / "group-44-44" / "receipt.json"
            try:
                compact_payload = json.loads(compact_summary.read_text())
                compact_layer = json.loads(compact_receipt.read_text())
            except (OSError, json.JSONDecodeError):
                compact_payload = None
                compact_layer = None
            if isinstance(compact_payload, dict) and isinstance(compact_layer, dict) and compact_layer.get("status") == "PASSED":
                compact_summary_ref = str(compact_summary.relative_to(HAWKING))
                compact_receipt_ref = str(compact_receipt.relative_to(HAWKING))
                for evidence_ref in (compact_summary_ref, compact_receipt_ref):
                    if evidence_ref not in textbook["cross_layer_evidence"]:
                        textbook["cross_layer_evidence"] += f" + {evidence_ref}"
                    if evidence_ref not in state["accepted_receipts"]:
                        state["accepted_receipts"].append(evidence_ref)
                compact_bytes = compact_layer.get("bytes", {})
                compact_execution = compact_layer.get("execution", {})
                compact_timing = compact_layer.get("timing", {})
                dense_bytes = 5187958208
                dense_wall = 72324202167
                compact_source_bytes = compact_bytes.get("source_payload_bytes_read")
                compact_wall = compact_layer.get("elapsed_wall_ns")
                textbook["compact_expert_executor"] = {
                    "receipt": compact_receipt_ref,
                    "summary": compact_summary_ref,
                    "layer": 44,
                    "compact_experts": compact_payload.get("compact_experts"),
                    "source_bytes_read": compact_source_bytes,
                    "source_load_ns": compact_timing.get("source_load_ns"),
                    "wall_ns": compact_wall,
                    "dense_control_source_bytes": dense_bytes,
                    "dense_control_wall_ns": dense_wall,
                    "source_byte_reduction_fraction": (1 - compact_source_bytes / dense_bytes) if isinstance(compact_source_bytes, (int, float)) and dense_bytes else None,
                    "wall_speedup_ratio": (dense_wall / compact_wall) if isinstance(compact_wall, (int, float)) and compact_wall else None,
                    "dispatches": compact_execution.get("dispatches"),
                    "parity": compact_layer.get("parity", {}).get("passed"),
                    "final_output_hash": compact_layer.get("parity", {}).get("final_output_hash"),
                    "promotion_allowed": False,
                    "claim": "exact compact routed-bank layer gate; complete-token, residency, and broad-layer promotion remain open",
                }

            compact_group_dir = rec / "FLASH_COMPACT_L44_L46_V3"
            if not (compact_group_dir / "FAST_CHAIN_SUMMARY.json").exists():
                compact_group_dir = rec / "FLASH_COMPACT_L44_L46_V2"
            if not (compact_group_dir / "FAST_CHAIN_SUMMARY.json").exists():
                compact_group_dir = rec / "FLASH_COMPACT_L44_L46_V1"
            compact_group_summary = compact_group_dir / "FAST_CHAIN_SUMMARY.json"
            compact_group_receipt = compact_group_dir / "group-44-46" / "receipt.json"
            try:
                compact_group_payload = json.loads(compact_group_summary.read_text())
                compact_group_layer = json.loads(compact_group_receipt.read_text())
            except (OSError, json.JSONDecodeError):
                compact_group_payload = None
                compact_group_layer = None
            if isinstance(compact_group_payload, dict) and isinstance(compact_group_layer, dict) and compact_group_layer.get("status") == "PASSED_LINEAR_PREFIX_BLOCKED_AT_FULL_ATTENTION":
                group_summary_ref = str(compact_group_summary.relative_to(HAWKING))
                group_receipt_ref = str(compact_group_receipt.relative_to(HAWKING))
                for evidence_ref in (group_summary_ref, group_receipt_ref):
                    if evidence_ref not in textbook["cross_layer_evidence"]:
                        textbook["cross_layer_evidence"] += f" + {evidence_ref}"
                    if evidence_ref not in state["accepted_receipts"]:
                        state["accepted_receipts"].append(evidence_ref)
                group_source_bytes = compact_group_layer.get("source", {}).get("manifest", {}).get("bytes")
                if not isinstance(group_source_bytes, (int, float)):
                    group_source_bytes = compact_group_layer.get("execution", {}).get("source_payload_bytes_read")
                group_wall = compact_group_layer.get("elapsed_wall_ns")
                dense_group_wall = 466544220209
                textbook["compact_group_executor"] = {
                    "receipt": group_receipt_ref,
                    "summary": group_summary_ref,
                    "start_layer": compact_group_payload.get("start_layer"),
                    "end_layer": compact_group_payload.get("end_layer"),
                    "elapsed_wall_ns": group_wall,
                    "source_payload_bytes_read": sum(int(layer.get("source_bytes_read", 0)) for layer in compact_group_layer.get("layers", []) if isinstance(layer, dict)),
                    "dense_control_group_wall_ns": dense_group_wall,
                    "wall_speedup_ratio": (dense_group_wall / group_wall) if isinstance(group_wall, (int, float)) and group_wall else None,
                    "parity": all(layer.get("status") == "PASSED" for layer in compact_group_layer.get("layers", []) if isinstance(layer, dict)),
                    "dispatches": sum(int(layer.get("dispatches", 0)) for layer in compact_group_layer.get("layers", []) if isinstance(layer, dict)),
                    "promotion_allowed": False,
                    "claim": "exact compact routed-bank 44..46 continuation; host checkpoint seam and complete-token residency remain open",
                }

            # The hot-chain profile is a receipt-level view of the protected
            # FastPath fields.  It is intentionally accepted as evidence even
            # while its exit gate is false: the host-state seam and the
            # minimum 8-layer requirement must remain visible to the ledger.
            hot_profile = rec / "FLASH_HOT_CHAIN_PROFILE_L44_L46.json"
            try:
                hot_payload = json.loads(hot_profile.read_text())
            except (OSError, json.JSONDecodeError):
                hot_payload = None
            if isinstance(hot_payload, dict) and hot_payload.get("schema") == "hawking.flash_hot_chain_profile.v1":
                hot_ref = str(hot_profile.relative_to(HAWKING))
                if hot_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {hot_ref}"
                if hot_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(hot_ref)
                textbook["hot_chain_profile"] = {
                    "receipt": hot_ref,
                    "status": hot_payload.get("status"),
                    "start_layer": hot_payload.get("start_layer"),
                    "end_layer": hot_payload.get("end_layer"),
                    "layer_count": hot_payload.get("layer_count"),
                    "complete_wall_ns": hot_payload.get("complete_wall_ns"),
                    "GPU_ns": hot_payload.get("GPU_ns"),
                    "host_roundtrip_count": hot_payload.get("host_roundtrip_count"),
                    "gate": hot_payload.get("gate"),
                    "parity_verdict": hot_payload.get("parity_verdict"),
                    "claim": hot_payload.get("claim_boundary"),
                }

            hot_device_profile = rec / "FLASH_HOT_CHAIN_PROFILE_DEVICE_L44_L46_V1.json"
            try:
                hot_device_payload = json.loads(hot_device_profile.read_text())
            except (OSError, json.JSONDecodeError):
                hot_device_payload = None
            if isinstance(hot_device_payload, dict) and hot_device_payload.get("schema") == "hawking.flash_hot_chain_profile.v1":
                hot_device_ref = str(hot_device_profile.relative_to(HAWKING))
                if hot_device_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {hot_device_ref}"
                if hot_device_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(hot_device_ref)
                textbook["device_resident_hot_chain_profile"] = {
                    "receipt": hot_device_ref,
                    "status": hot_device_payload.get("status"),
                    "start_layer": hot_device_payload.get("start_layer"),
                    "end_layer": hot_device_payload.get("end_layer"),
                    "layer_count": hot_device_payload.get("layer_count"),
                    "complete_wall_ns": hot_device_payload.get("complete_wall_ns"),
                    "GPU_ns": hot_device_payload.get("GPU_ns"),
                    "host_roundtrip_count": hot_device_payload.get("host_roundtrip_count"),
                    "gate": hot_device_payload.get("gate"),
                    "parity_verdict": hot_device_payload.get("parity_verdict"),
                    "claim": hot_device_payload.get("claim_boundary"),
                }

            hot_cross_profile = rec / "FLASH_HOT_CHAIN_PROFILE_DEVICE_L44_L47_V1.json"
            try:
                hot_cross_payload = json.loads(hot_cross_profile.read_text())
            except (OSError, json.JSONDecodeError):
                hot_cross_payload = None
            if isinstance(hot_cross_payload, dict) and hot_cross_payload.get("schema") == "hawking.flash_hot_chain_profile.v1":
                hot_cross_ref = str(hot_cross_profile.relative_to(HAWKING))
                if hot_cross_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {hot_cross_ref}"
                if hot_cross_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(hot_cross_ref)
                textbook["device_resident_cross_species_profile"] = {
                    "receipt": hot_cross_ref,
                    "status": hot_cross_payload.get("status"),
                    "start_layer": hot_cross_payload.get("start_layer"),
                    "end_layer": hot_cross_payload.get("end_layer"),
                    "layer_count": hot_cross_payload.get("layer_count"),
                    "complete_wall_ns": hot_cross_payload.get("complete_wall_ns"),
                    "GPU_ns": hot_cross_payload.get("GPU_ns"),
                    "host_roundtrip_count": hot_cross_payload.get("host_roundtrip_count"),
                    "gate": hot_cross_payload.get("gate"),
                    "parity_verdict": hot_cross_payload.get("parity_verdict"),
                    "claim": hot_cross_payload.get("claim_boundary"),
                }

            # The protected 8-layer continuation is kept as a distinct receipt:
            # it proves the minimum hot-chain length and zero required host
            # handoff, while deliberately retaining the false exit gate until
            # compact multi-species parity is deep-verified.
            hot_eight_profile = rec / "FLASH_HOT_CHAIN_PROFILE_DEVICE_L40_L47_V1.json"
            try:
                hot_eight_payload = json.loads(hot_eight_profile.read_text())
            except (OSError, json.JSONDecodeError):
                hot_eight_payload = None
            if isinstance(hot_eight_payload, dict) and hot_eight_payload.get("schema") == "hawking.flash_hot_chain_profile.v1":
                hot_eight_ref = str(hot_eight_profile.relative_to(HAWKING))
                if hot_eight_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {hot_eight_ref}"
                if hot_eight_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(hot_eight_ref)
                textbook["device_resident_eight_layer_profile"] = {
                    "receipt": hot_eight_ref,
                    "status": hot_eight_payload.get("status"),
                    "start_layer": hot_eight_payload.get("start_layer"),
                    "end_layer": hot_eight_payload.get("end_layer"),
                    "layer_count": hot_eight_payload.get("layer_count"),
                    "complete_wall_ns": hot_eight_payload.get("complete_wall_ns"),
                    "GPU_ns": hot_eight_payload.get("GPU_ns"),
                    "host_roundtrip_count": hot_eight_payload.get("host_roundtrip_count"),
                    "gate": hot_eight_payload.get("gate"),
                    "parity_verdict": hot_eight_payload.get("parity_verdict"),
                    "claim": hot_eight_payload.get("claim_boundary"),
                }

            hot_deep_profile = rec / "FLASH_HOT_CHAIN_PROFILE_DEVICE_L40_L47_DEEP_V2.json"
            try:
                hot_deep_payload = json.loads(hot_deep_profile.read_text())
            except (OSError, json.JSONDecodeError):
                hot_deep_payload = None
            if isinstance(hot_deep_payload, dict) and hot_deep_payload.get("schema") == "hawking.flash_hot_chain_profile.v1":
                hot_deep_ref = str(hot_deep_profile.relative_to(HAWKING))
                if hot_deep_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {hot_deep_ref}"
                if hot_deep_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(hot_deep_ref)
                textbook["device_resident_deep_eight_layer_profile"] = {
                    "receipt": hot_deep_ref,
                    "status": hot_deep_payload.get("status"),
                    "start_layer": hot_deep_payload.get("start_layer"),
                    "end_layer": hot_deep_payload.get("end_layer"),
                    "layer_count": hot_deep_payload.get("layer_count"),
                    "complete_wall_ns": hot_deep_payload.get("complete_wall_ns"),
                    "GPU_ns": hot_deep_payload.get("GPU_ns"),
                    "host_roundtrip_count": hot_deep_payload.get("host_roundtrip_count"),
                    "gate": hot_deep_payload.get("gate"),
                    "parity_verdict": hot_deep_payload.get("parity_verdict"),
                    "claim": hot_deep_payload.get("claim_boundary"),
                }

            hot_complete_profile = rec / "FLASH_HOT_CHAIN_PROFILE_DEVICE_L0_L47_COMPLETE_V1.json"
            try:
                hot_complete_payload = json.loads(hot_complete_profile.read_text())
            except (OSError, json.JSONDecodeError):
                hot_complete_payload = None
            if isinstance(hot_complete_payload, dict) and hot_complete_payload.get("schema") == "hawking.flash_hot_chain_profile.v1":
                hot_complete_ref = str(hot_complete_profile.relative_to(HAWKING))
                if hot_complete_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {hot_complete_ref}"
                if hot_complete_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(hot_complete_ref)
                textbook["device_resident_complete_chain_profile"] = {
                    "receipt": hot_complete_ref,
                    "status": hot_complete_payload.get("status"),
                    "start_layer": hot_complete_payload.get("start_layer"),
                    "end_layer": hot_complete_payload.get("end_layer"),
                    "layer_count": hot_complete_payload.get("layer_count"),
                    "complete_wall_ns": hot_complete_payload.get("complete_wall_ns"),
                    "GPU_ns": hot_complete_payload.get("GPU_ns"),
                    "host_roundtrip_count": hot_complete_payload.get("host_roundtrip_count"),
                    "gate": hot_complete_payload.get("gate"),
                    "parity_verdict": hot_complete_payload.get("parity_verdict"),
                    "complete_token_terminal": hot_complete_payload.get("complete_token_terminal"),
                    "claim": hot_complete_payload.get("claim_boundary"),
                }

            complete_token_measurement = rec / "FLASH_COMPLETE_TOKEN_DEVICE_RESIDENT_V1.json"
            try:
                complete_token_payload = json.loads(complete_token_measurement.read_text())
            except (OSError, json.JSONDecodeError):
                complete_token_payload = None
            if isinstance(complete_token_payload, dict) and complete_token_payload.get("schema") == "hawking.flash.complete_token_measurement.v1":
                complete_token_ref = str(complete_token_measurement.relative_to(HAWKING))
                if complete_token_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {complete_token_ref}"
                if complete_token_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(complete_token_ref)
                textbook["complete_token_terminal_measurement"] = {
                    "receipt": complete_token_ref,
                    "status": complete_token_payload.get("status"),
                    "token_id": (complete_token_payload.get("terminal_token") or {}).get("token_id"),
                    "forward_wall_ns": (complete_token_payload.get("execution") or {}).get("forward_wall_ns"),
                    "forward_gpu_ns": (complete_token_payload.get("execution") or {}).get("forward_gpu_ns"),
                    "accepted_tps": complete_token_payload.get("accepted_tps"),
                    "complete_system_ebpw": complete_token_payload.get("complete_system_ebpw"),
                    "claim": complete_token_payload.get("claim_boundary"),
                }

            stateful_tps_gate = next(
                (candidate for candidate in (
                    rec / "FLASH_STATEFUL_TPS_GATE_V14.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V13.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V12.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V11.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V10.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V9.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V8.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V7.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V6.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V5.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V4.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V3.json",
                    rec / "FLASH_STATEFUL_TPS_GATE_V2.json",
                    rec / "FLASH_STATEFUL_TPS_GATE.json",
                ) if candidate.is_file()),
                rec / "FLASH_STATEFUL_TPS_GATE.json",
            )
            try:
                stateful_tps_payload = json.loads(stateful_tps_gate.read_text())
            except (OSError, json.JSONDecodeError):
                stateful_tps_payload = None
            if isinstance(stateful_tps_payload, dict) and stateful_tps_payload.get("schema") == "hawking.flash.stateful_tps_gate.v1":
                stateful_tps_ref = str(stateful_tps_gate.relative_to(HAWKING))
                if stateful_tps_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {stateful_tps_ref}"
                if stateful_tps_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(stateful_tps_ref)
                textbook["stateful_tps_gate"] = {
                    "receipt": stateful_tps_ref,
                    "status": stateful_tps_payload.get("status"),
                    "first_physical_failure_boundary": stateful_tps_payload.get("first_physical_failure_boundary"),
                    "accepted_tokens": stateful_tps_payload.get("accepted_tokens"),
                    "accepted_tps": stateful_tps_payload.get("accepted_tps"),
                    "claim": stateful_tps_payload.get("claim_boundary"),
                }

            stateful_organ = rec / "FLASH_STATEFUL_LINEAR_ORGAN.json"
            try:
                stateful_organ_payload = json.loads(stateful_organ.read_text())
            except (OSError, json.JSONDecodeError):
                stateful_organ_payload = None
            if isinstance(stateful_organ_payload, dict) and stateful_organ_payload.get("schema") == "hawking.flash.stateful_linear_organ_probe.v1":
                organ_ref = str(stateful_organ.relative_to(HAWKING))
                if organ_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {organ_ref}"
                if organ_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(organ_ref)
                steps = stateful_organ_payload.get("steps") or []
                textbook["stateful_linear_organ"] = {
                    "receipt": organ_ref,
                    "status": stateful_organ_payload.get("status"),
                    "steps": len(steps),
                    "state_changed_between_steps": stateful_organ_payload.get("state_changed_between_steps"),
                    "first_token_parity": (steps[0].get("first_token_parity") if steps and isinstance(steps[0], dict) else None),
                    "accepted_tps": stateful_organ_payload.get("accepted_tps"),
                    "claim": stateful_organ_payload.get("claim_boundary"),
                }

            stateful_attention_organ = next(
                (candidate for candidate in (
                    rec / "FLASH_STATEFUL_CROSS_SPECIES_SEAM_V3_ATTN.json",
                    rec / "FLASH_STATEFUL_CROSS_SPECIES_SEAM_V2_ATTN.json",
                    rec / "FLASH_STATEFUL_ATTENTION_ORGAN_V2.json",
                    rec / "FLASH_STATEFUL_ATTENTION_ORGAN.json",
                ) if candidate.is_file()),
                rec / "FLASH_STATEFUL_ATTENTION_ORGAN.json",
            )
            try:
                stateful_attention_payload = json.loads(stateful_attention_organ.read_text())
            except (OSError, json.JSONDecodeError):
                stateful_attention_payload = None
            if isinstance(stateful_attention_payload, dict) and stateful_attention_payload.get("schema") == "hawking.flash.stateful_attention_organ_probe.v1":
                attention_ref = str(stateful_attention_organ.relative_to(HAWKING))
                if attention_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {attention_ref}"
                if attention_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(attention_ref)
                attention_steps = stateful_attention_payload.get("steps") or []
                textbook["stateful_attention_organ"] = {
                    "receipt": attention_ref,
                    "status": stateful_attention_payload.get("status"),
                    "layer": stateful_attention_payload.get("layer"),
                    "steps": len(attention_steps),
                    "distinct_kv_slots": stateful_attention_payload.get("distinct_kv_slots"),
                    "full_attention_mlp_epilogue": (stateful_attention_payload.get("execution") or {}).get("full_attention_mlp_epilogue"),
                    "first_step_parity": (attention_steps[0].get("first_step_parity") if attention_steps and isinstance(attention_steps[0], dict) else None),
                    "accepted_tps": stateful_attention_payload.get("accepted_tps"),
                    "claim": stateful_attention_payload.get("claim_boundary"),
                }

            cross_species_seam = next(
                (candidate for candidate in (
                    rec / "FLASH_STATEFUL_CROSS_SPECIES_SEAM_V3.json",
                    rec / "FLASH_STATEFUL_CROSS_SPECIES_SEAM_V2.json",
                    rec / "FLASH_STATEFUL_CROSS_SPECIES_SEAM.json",
                ) if candidate.is_file()),
                rec / "FLASH_STATEFUL_CROSS_SPECIES_SEAM.json",
            )
            try:
                cross_species_payload = json.loads(cross_species_seam.read_text())
            except (OSError, json.JSONDecodeError):
                cross_species_payload = None
            if isinstance(cross_species_payload, dict) and cross_species_payload.get("schema") == "hawking.flash.stateful_cross_species_seam.v1":
                seam_ref = str(cross_species_seam.relative_to(HAWKING))
                if seam_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {seam_ref}"
                if seam_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(seam_ref)
                textbook["stateful_cross_species_seam"] = {
                    "receipt": seam_ref,
                    "status": cross_species_payload.get("status"),
                    "token_ids": cross_species_payload.get("token_ids"),
                    "linear_prefix": cross_species_payload.get("linear_prefix"),
                    "attention": cross_species_payload.get("attention"),
                    "claim": cross_species_payload.get("claim_boundary"),
                }

            layer3_layer4_bridge = rec / "FLASH_STATEFUL_LAYER3_LAYER4_BRIDGE.json"
            try:
                layer3_layer4_payload = json.loads(layer3_layer4_bridge.read_text())
            except (OSError, json.JSONDecodeError):
                layer3_layer4_payload = None
            if isinstance(layer3_layer4_payload, dict) and layer3_layer4_payload.get("schema") == "hawking.flash.stateful_layer3_layer4_bridge.v1":
                bridge_ref = str(layer3_layer4_bridge.relative_to(HAWKING))
                if bridge_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {bridge_ref}"
                if bridge_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(bridge_ref)
                textbook["stateful_layer3_layer4_bridge"] = {
                    "receipt": bridge_ref,
                    "status": layer3_layer4_payload.get("status"),
                    "token_ids": layer3_layer4_payload.get("token_ids"),
                    "layer3": layer3_layer4_payload.get("layer3"),
                    "layer4": layer3_layer4_payload.get("layer4"),
                    "claim": layer3_layer4_payload.get("claim_boundary"),
                }

            layer3_layer7_bridge = rec / "FLASH_STATEFUL_LAYER3_LAYER7_BRIDGE.json"
            try:
                layer3_layer7_payload = json.loads(layer3_layer7_bridge.read_text())
            except (OSError, json.JSONDecodeError):
                layer3_layer7_payload = None
            if isinstance(layer3_layer7_payload, dict) and layer3_layer7_payload.get("schema") == "hawking.flash.stateful_layer3_layer7_bridge.v2":
                bridge_ref = str(layer3_layer7_bridge.relative_to(HAWKING))
                if bridge_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {bridge_ref}"
                if bridge_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(bridge_ref)
                textbook["stateful_layer3_layer7_bridge"] = {
                    "receipt": bridge_ref,
                    "status": layer3_layer7_payload.get("status"),
                    "token_ids": layer3_layer7_payload.get("token_ids"),
                    "layer3": layer3_layer7_payload.get("layer3"),
                    "linear_4_6": layer3_layer7_payload.get("linear_4_6"),
                    "layer7": layer3_layer7_payload.get("layer7"),
                    "claim": layer3_layer7_payload.get("claim_boundary"),
                }

            layer3_layer11_bridge = rec / "FLASH_STATEFUL_LAYER3_LAYER11_BRIDGE.json"
            try:
                layer3_layer11_payload = json.loads(layer3_layer11_bridge.read_text())
            except (OSError, json.JSONDecodeError):
                layer3_layer11_payload = None
            if isinstance(layer3_layer11_payload, dict) and layer3_layer11_payload.get("schema") == "hawking.flash.stateful_layer3_layer11_bridge.v3":
                bridge_ref = str(layer3_layer11_bridge.relative_to(HAWKING))
                if bridge_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {bridge_ref}"
                if bridge_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(bridge_ref)
                textbook["stateful_layer3_layer11_bridge"] = {
                    "receipt": bridge_ref,
                    "status": layer3_layer11_payload.get("status"),
                    "token_ids": layer3_layer11_payload.get("token_ids"),
                    "layer3": layer3_layer11_payload.get("layer3"),
                    "linear_4_6": layer3_layer11_payload.get("linear_4_6"),
                    "layer7": layer3_layer11_payload.get("layer7"),
                    "linear_8_10": layer3_layer11_payload.get("linear_8_10"),
                    "layer11": layer3_layer11_payload.get("layer11"),
                    "claim": layer3_layer11_payload.get("claim_boundary"),
                }

            complete_session = next(
                (candidate for candidate in (
                    rec / "FLASH_STATEFUL_COMPLETE_TOKEN_ACCEPTED.json",
                    rec / "FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json",
                ) if candidate.is_file()),
                rec / "FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json",
            )
            try:
                complete_session_payload = json.loads(complete_session.read_text())
            except (OSError, json.JSONDecodeError):
                complete_session_payload = None
            if isinstance(complete_session_payload, dict) and complete_session_payload.get("schema") in {
                "hawking.flash.stateful_complete_token_session.v1",
                "hawking.flash.stateful_complete_token_acceptance.v1",
            }:
                session_ref = str(complete_session.relative_to(HAWKING))
                if session_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {session_ref}"
                if session_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(session_ref)
                # The accepted-token envelope is intentionally provenance-bound
                # to the expensive dense oracle. Keep that source receipt in
                # the canonical state as well, so the rejected teacher-forced
                # candidate and its exact terminal boundary remain discoverable
                # without treating them as a new acceptance claim.
                source_session_ref = complete_session_payload.get("source_session_receipt")
                if isinstance(source_session_ref, str):
                    source_path = HAWKING / source_session_ref
                    if source_path.is_file():
                        if source_session_ref not in textbook["cross_layer_evidence"]:
                            textbook["cross_layer_evidence"] += f" + {source_session_ref}"
                        if source_session_ref not in state["accepted_receipts"]:
                            state["accepted_receipts"].append(source_session_ref)
                textbook["stateful_complete_token_session"] = {
                    "receipt": session_ref,
                    "source_session_receipt": complete_session_payload.get("source_session_receipt"),
                    "source_session_sha256": complete_session_payload.get("source_session_sha256"),
                    "status": complete_session_payload.get("status"),
                    "token_ids": complete_session_payload.get("token_ids") or (
                        (complete_session_payload.get("prompt_token_ids") or [])
                        + (complete_session_payload.get("generated_token_ids") or [])
                    ),
                    "prompt_token_ids": complete_session_payload.get("prompt_token_ids"),
                    "generated_token_ids": complete_session_payload.get("generated_token_ids"),
                    "candidate_token_id": complete_session_payload.get("candidate_token_id") or (
                        (complete_session_payload.get("generated_token_ids") or [None])[0]
                    ),
                    "accepted_generation_tokens": complete_session_payload.get("accepted_generation_tokens"),
                    "first_physical_failure_boundary": complete_session_payload.get("first_physical_failure_boundary"),
                    "claim": complete_session_payload.get("claim_boundary"),
                }

            terminal_executor_compile = rec / "FLASH_TERMINAL_EXECUTOR_COMPILE.json"
            try:
                terminal_executor_compile_payload = json.loads(terminal_executor_compile.read_text())
            except (OSError, json.JSONDecodeError):
                terminal_executor_compile_payload = None
            if (isinstance(terminal_executor_compile_payload, dict)
                    and terminal_executor_compile_payload.get("schema") == "hawking.flash.terminal_executor_compile.v1"):
                executor_ref = str(terminal_executor_compile.relative_to(HAWKING))
                if executor_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {executor_ref}"
                if executor_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(executor_ref)
                textbook["terminal_executor_compile"] = {
                    "receipt": executor_ref,
                    "status": terminal_executor_compile_payload.get("status"),
                    "command": terminal_executor_compile_payload.get("command"),
                    "architecture": terminal_executor_compile_payload.get("architecture"),
                    "physical_execution": terminal_executor_compile_payload.get("physical_execution"),
                    "claim": terminal_executor_compile_payload.get("claim_boundary"),
                }

            complete_session_timing = rec / "FLASH_STATEFUL_COMPLETE_SESSION_TIMING.json"
            try:
                complete_session_timing_payload = json.loads(complete_session_timing.read_text())
            except (OSError, json.JSONDecodeError):
                complete_session_timing_payload = None
            if isinstance(complete_session_timing_payload, dict) and complete_session_timing_payload.get("schema") == "hawking.flash.complete_session_timing_decomposition.v1":
                timing_ref = str(complete_session_timing.relative_to(HAWKING))
                if timing_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {timing_ref}"
                if timing_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(timing_ref)
                textbook["stateful_complete_session_timing"] = {
                    "receipt": timing_ref,
                    "status": complete_session_timing_payload.get("status"),
                    "totals": complete_session_timing_payload.get("totals"),
                    "top_runtime_latencies": complete_session_timing_payload.get("top_runtime_latencies"),
                    "claim": complete_session_timing_payload.get("claim_boundary"),
                }

            stateful_prefix = rec / "FLASH_STATEFUL_LINEAR_PREFIX_SESSION.json"
            try:
                stateful_prefix_payload = json.loads(stateful_prefix.read_text())
            except (OSError, json.JSONDecodeError):
                stateful_prefix_payload = None
            if isinstance(stateful_prefix_payload, dict) and stateful_prefix_payload.get("schema") == "hawking.flash.stateful_linear_prefix_session.v1":
                prefix_ref = str(stateful_prefix.relative_to(HAWKING))
                if prefix_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {prefix_ref}"
                if prefix_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(prefix_ref)
                textbook["stateful_linear_prefix_session"] = {
                    "receipt": prefix_ref,
                    "status": stateful_prefix_payload.get("status"),
                    "layer_range": stateful_prefix_payload.get("layer_range"),
                    "steps": len(stateful_prefix_payload.get("steps") or []),
                    "state_changed_layers": stateful_prefix_payload.get("state_changed_layers"),
                    "accepted_tps": stateful_prefix_payload.get("accepted_tps"),
                    "claim": stateful_prefix_payload.get("claim_boundary"),
                }

            # A bounded rerun of the same prefix records the new executor
            # ownership boundary (one source index/context for the process).
            # Keep it separate from the original canonical prefix receipt so
            # the before/after comparison remains explicit.
            stateful_prefix_reuse = rec / "FLASH_STATEFUL_LINEAR_PREFIX_SESSION_REUSE.json"
            try:
                stateful_prefix_reuse_payload = json.loads(stateful_prefix_reuse.read_text())
            except (OSError, json.JSONDecodeError):
                stateful_prefix_reuse_payload = None
            if isinstance(stateful_prefix_reuse_payload, dict) and stateful_prefix_reuse_payload.get("schema") == "hawking.flash.stateful_linear_prefix_session.v1":
                reuse_ref = str(stateful_prefix_reuse.relative_to(HAWKING))
                if reuse_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {reuse_ref}"
                if reuse_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(reuse_ref)
                textbook["stateful_linear_prefix_reuse"] = {
                    "receipt": reuse_ref,
                    "status": stateful_prefix_reuse_payload.get("status"),
                    "token_ids": stateful_prefix_reuse_payload.get("token_ids"),
                    "layer_range": stateful_prefix_reuse_payload.get("layer_range"),
                    "source_payload_bytes_read": (stateful_prefix_reuse_payload.get("execution") or {}).get("source_payload_bytes_read"),
                    "process_boundary": (stateful_prefix_reuse_payload.get("execution") or {}).get("process_boundary"),
                    "source_index_reused": (stateful_prefix_reuse_payload.get("execution") or {}).get("source_index_reused"),
                    "metal_context_reused": (stateful_prefix_reuse_payload.get("execution") or {}).get("metal_context_reused"),
                    "claim": stateful_prefix_reuse_payload.get("claim_boundary"),
                }

            route_audit = rec / "FLASH_ROUTE_STABILITY_AUDIT.json"
            try:
                route_audit_payload = json.loads(route_audit.read_text())
            except (OSError, json.JSONDecodeError):
                route_audit_payload = None
            if isinstance(route_audit_payload, dict) and route_audit_payload.get("schema") == "hawking.flash.route_stability_audit.v1":
                route_ref = str(route_audit.relative_to(HAWKING))
                if route_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {route_ref}"
                if route_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(route_ref)
                textbook["flash_route_stability_audit"] = {
                    "receipt": route_ref,
                    "status": route_audit_payload.get("status"),
                    "summary": route_audit_payload.get("summary"),
                    "claim": route_audit_payload.get("claim_boundary"),
                }

            attention_reuse = rec / "FLASH_STATEFUL_ATTENTION_REUSE.json"
            try:
                attention_reuse_payload = json.loads(attention_reuse.read_text())
            except (OSError, json.JSONDecodeError):
                attention_reuse_payload = None
            if isinstance(attention_reuse_payload, dict) and attention_reuse_payload.get("schema") == "hawking.flash.stateful_attention_organ_probe.v1":
                attention_ref = str(attention_reuse.relative_to(HAWKING))
                if attention_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {attention_ref}"
                if attention_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(attention_ref)
                textbook["stateful_attention_reuse"] = {
                    "receipt": attention_ref,
                    "status": attention_reuse_payload.get("status"),
                    "layer": attention_reuse_payload.get("layer"),
                    "execution": attention_reuse_payload.get("execution"),
                    "distinct_kv_slots": attention_reuse_payload.get("distinct_kv_slots"),
                    "claim": attention_reuse_payload.get("claim_boundary"),
                }

            route_union_parity = rec / "FLASH_ROUTE_UNION_PARITY.json"
            try:
                route_union_parity_payload = json.loads(route_union_parity.read_text())
            except (OSError, json.JSONDecodeError):
                route_union_parity_payload = None
            if isinstance(route_union_parity_payload, dict) and route_union_parity_payload.get("schema") == "hawking.flash.route_union_parity.v1":
                union_ref = str(route_union_parity.relative_to(HAWKING))
                if union_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {union_ref}"
                if union_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(union_ref)
                textbook["flash_route_union_parity"] = {
                    "receipt": union_ref,
                    "status": route_union_parity_payload.get("status"),
                    "source_payload": route_union_parity_payload.get("source_payload"),
                    "timing": route_union_parity_payload.get("timing"),
                    "comparisons": len(route_union_parity_payload.get("comparisons") or []),
                    "claim": route_union_parity_payload.get("claim_boundary"),
                }

            attention_route_union_parity = rec / "FLASH_ATTENTION_ROUTE_UNION_PARITY.json"
            try:
                attention_route_union_parity_payload = json.loads(attention_route_union_parity.read_text())
            except (OSError, json.JSONDecodeError):
                attention_route_union_parity_payload = None
            if (isinstance(attention_route_union_parity_payload, dict)
                    and attention_route_union_parity_payload.get("schema") == "hawking.flash.route_union_parity.v1"):
                attention_union_ref = str(attention_route_union_parity.relative_to(HAWKING))
                if attention_union_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {attention_union_ref}"
                if attention_union_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(attention_union_ref)
                textbook["flash_attention_route_union_parity"] = {
                    "receipt": attention_union_ref,
                    "status": attention_route_union_parity_payload.get("status"),
                    "source_payload": attention_route_union_parity_payload.get("source_payload"),
                    "timing": attention_route_union_parity_payload.get("timing"),
                    "comparisons": len(attention_route_union_parity_payload.get("comparisons") or []),
                    "claim": attention_route_union_parity_payload.get("claim_boundary"),
                }

            fast_compact_parity = rec / "FLASH_FAST_COMPACT_L0_L3_PARITY.json"
            try:
                fast_compact_parity_payload = json.loads(fast_compact_parity.read_text())
            except (OSError, json.JSONDecodeError):
                fast_compact_parity_payload = None
            if (isinstance(fast_compact_parity_payload, dict)
                    and fast_compact_parity_payload.get("schema") == "hawking.flash.fast_compact_parity.v1"):
                fast_compact_ref = str(fast_compact_parity.relative_to(HAWKING))
                if fast_compact_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fast_compact_ref}"
                if fast_compact_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fast_compact_ref)
                textbook["flash_fast_compact_parity"] = {
                    "receipt": fast_compact_ref,
                    "status": fast_compact_parity_payload.get("status"),
                    "layers": fast_compact_parity_payload.get("layers"),
                    "source_payload": fast_compact_parity_payload.get("source_payload"),
                    "timing": fast_compact_parity_payload.get("timing"),
                    "claim": fast_compact_parity_payload.get("claim_boundary"),
                }

            fast_compact_l7 = rec / "FLASH_COMPACT_L0_L7_V1" / "FAST_CHAIN_SUMMARY.json"
            try:
                fast_compact_l7_payload = json.loads(fast_compact_l7.read_text())
            except (OSError, json.JSONDecodeError):
                fast_compact_l7_payload = None
            if isinstance(fast_compact_l7_payload, dict) and fast_compact_l7_payload.get("schema") == "hawking.flash_fast_chain.v1":
                fast_compact_l7_ref = str(fast_compact_l7.relative_to(HAWKING))
                if fast_compact_l7_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fast_compact_l7_ref}"
                if fast_compact_l7_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fast_compact_l7_ref)
                textbook["flash_fast_compact_l0_l7_candidate"] = {
                    "receipt": fast_compact_l7_ref,
                    "status": fast_compact_l7_payload.get("status"),
                    "elapsed_wall_ns": fast_compact_l7_payload.get("elapsed_wall_ns"),
                    "device_resident": fast_compact_l7_payload.get("device_resident"),
                    "compact_experts": fast_compact_l7_payload.get("compact_experts"),
                    "claim": fast_compact_l7_payload.get("claim_boundary"),
                }

            fast_compact_l7_parity = rec / "FLASH_FAST_COMPACT_L0_L7_PARITY.json"
            try:
                fast_compact_l7_parity_payload = json.loads(fast_compact_l7_parity.read_text())
            except (OSError, json.JSONDecodeError):
                fast_compact_l7_parity_payload = None
            if (isinstance(fast_compact_l7_parity_payload, dict)
                    and fast_compact_l7_parity_payload.get("schema") == "hawking.flash.fast_compact_parity.v1"):
                fast_compact_l7_parity_ref = str(fast_compact_l7_parity.relative_to(HAWKING))
                if fast_compact_l7_parity_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {fast_compact_l7_parity_ref}"
                if fast_compact_l7_parity_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(fast_compact_l7_parity_ref)
                textbook["flash_fast_compact_l0_l7_parity"] = {
                    "receipt": fast_compact_l7_parity_ref,
                    "status": fast_compact_l7_parity_payload.get("status"),
                    "layers": fast_compact_l7_parity_payload.get("layers"),
                    "source_payload": fast_compact_l7_parity_payload.get("source_payload"),
                    "timing": fast_compact_l7_parity_payload.get("timing"),
                    "claim": fast_compact_l7_parity_payload.get("claim_boundary"),
                }

            attention_route_audit = rec / "FLASH_ATTENTION_ROUTE_STABILITY_AUDIT.json"
            try:
                attention_route_audit_payload = json.loads(attention_route_audit.read_text())
            except (OSError, json.JSONDecodeError):
                attention_route_audit_payload = None
            if isinstance(attention_route_audit_payload, dict) and attention_route_audit_payload.get("schema") == "hawking.flash.route_stability_audit.v1":
                attention_route_ref = str(attention_route_audit.relative_to(HAWKING))
                if attention_route_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {attention_route_ref}"
                if attention_route_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(attention_route_ref)
                textbook["flash_attention_route_stability_audit"] = {
                    "receipt": attention_route_ref,
                    "status": attention_route_audit_payload.get("status"),
                    "summary": attention_route_audit_payload.get("summary"),
                    "claim": attention_route_audit_payload.get("claim_boundary"),
                }

            timing_report = rec / "FLASH_CHAIN_TIMING_DECOMPOSITION.json"
            try:
                timing_payload = json.loads(timing_report.read_text())
            except (OSError, json.JSONDecodeError):
                timing_payload = None
            if isinstance(timing_payload, dict) and timing_payload.get("status") == "MEASURED_BASELINE":
                timing_ref = str(timing_report.relative_to(HAWKING))
                if timing_ref not in textbook["cross_layer_evidence"]:
                    textbook["cross_layer_evidence"] += f" + {timing_ref}"
                if timing_ref not in state["accepted_receipts"]:
                    state["accepted_receipts"].append(timing_ref)
                textbook["chain_timing_decomposition"] = {
                    "receipt": timing_ref,
                    "status": timing_payload.get("status"),
                    "totals": timing_payload.get("totals", {}),
                    "derived": timing_payload.get("derived", {}),
                    "claim": timing_payload.get("claim_boundary"),
                }
            remaining = f"layers {next_layer}..47" if next_layer <= 47 else "no remaining Flash layers"
            state["benchmark_qualification"]["flash_next"] = (
                f"BOUNDED_SOURCE_PARITY_L0_L{latest_layer}; verified source-BF16 "
                f"{latest_layer + 1} physical layer boundaries through explicit state handoffs; "
                f"{remaining}, complete-token, EBPW, and accepted-TPS promotion remain open"
            )
            terminal_probe = textbook.get("single_process_terminal_probe")
            if isinstance(terminal_probe, dict):
                state["benchmark_qualification"]["flash_next"] += (
                    f"; single-process L44-L47 terminal source-BF16 probe produced token "
                    f"{terminal_probe.get('token_id')} with "
                    f"{terminal_probe.get('dispatches')} native Metal dispatches; "
                    "streamed 48-layer runtime, EBPW, accepted-TPS, and residency remain open"
                )
            elif isinstance(terminal_payload, dict) and terminal_payload.get("status") == "PASSED":
                state["benchmark_qualification"]["flash_next"] += (
                    f"; terminal source-BF16 probe produced token "
                    f"{terminal_payload.get('terminal', {}).get('token_id')} with "
                    f"{terminal_payload.get('execution', {}).get('dispatches')} native Metal dispatches; "
                    "one-process runtime, EBPW, accepted-TPS, and residency remain open"
                )
            fast_executor = textbook.get("fast_executor")
            if isinstance(fast_executor, dict):
                state["benchmark_qualification"]["flash_next"] += (
                    f"; accelerated single-process L{fast_executor.get('start_layer')}..L{fast_executor.get('end_layer')} "
                    f"continuation passed {len(fast_executor.get('layers_passed', []))} layers with cached source index/context "
                    f"in {fast_executor.get('elapsed_wall_ns')} ns; host checkpoint seam, streamed device residency, "
                    "TPS, EBPW, and resident qualification remain open"
                )
            timing_decomposition = textbook.get("chain_timing_decomposition")
            if isinstance(timing_decomposition, dict):
                totals = timing_decomposition.get("totals", {})
                state["benchmark_qualification"]["flash_next"] += (
                    f"; baseline timing decomposition measured {totals.get('elapsed_wall_ns')} ns wall with "
                    f"{totals.get('gpu_execution_ns')} ns GPU and {totals.get('unattributed_wall_ns')} ns "
                    "unattributed ceremony requiring direct instrumentation"
                )
            linear_timing = textbook.get("linear_executor_timing")
            if isinstance(linear_timing, dict):
                state["benchmark_qualification"]["flash_next"] += (
                    f"; current linear-species timing sample L{linear_timing.get('layer')} measured "
                    f"{linear_timing.get('elapsed_wall_ns')} ns wall / {linear_timing.get('gpu_execution_ns')} ns GPU "
                    f"with {linear_timing.get('source_payload_bytes_read')} source bytes and "
                    f"{linear_timing.get('host_activation_roundtrips')} required host activation roundtrips; "
                    "this is diagnostic timing, not complete-token promotion"
                )
            expert_profile = textbook.get("expert_bank_io_profile")
            if isinstance(expert_profile, dict):
                state["benchmark_qualification"]["flash_next"] += (
                    f"; routed expert-bank profile measured {expert_profile.get('reduction_fraction', 0) * 100:.2f}% "
                    f"source-byte reduction ({expert_profile.get('full_bytes')} -> {expert_profile.get('selected_bytes')}) "
                    "for the verified layer-44 route set; broader-layer compact-bank qualification remains open"
                )
            compact_executor = textbook.get("compact_expert_executor")
            if isinstance(compact_executor, dict):
                state["benchmark_qualification"]["flash_next"] += (
                    f"; compact routed-bank layer-44 execution passed exact parity with "
                    f"{compact_executor.get('source_byte_reduction_fraction', 0) * 100:.2f}% fewer source bytes and "
                    f"{compact_executor.get('wall_speedup_ratio')}x wall-speed ratio versus dense control; "
                    "complete-model residency and TPS promotion remain open"
                )
            compact_group_executor = textbook.get("compact_group_executor")
            if isinstance(compact_group_executor, dict):
                state["benchmark_qualification"]["flash_next"] += (
                    f"; compact routed-bank L{compact_group_executor.get('start_layer')}..L{compact_group_executor.get('end_layer')} "
                    f"continuation passed exact parity at {compact_group_executor.get('wall_speedup_ratio')}x dense-control wall speed "
                    f"with {compact_group_executor.get('source_payload_bytes_read')} source bytes; complete-model residency remains open"
                )
            hot_chain = textbook.get("hot_chain_profile")
            if isinstance(hot_chain, dict):
                gate = hot_chain.get("gate") or {}
                state["benchmark_qualification"]["flash_next"] += (
                    f"; detached FastPath profile L{hot_chain.get('start_layer')}..L{hot_chain.get('end_layer')} "
                    f"measured {hot_chain.get('complete_wall_ns')} ns wall / {hot_chain.get('GPU_ns')} ns GPU "
                    f"with {hot_chain.get('host_roundtrip_count')} host state roundtrips; "
                    f"exit gate={gate.get('fastpath_exit_gate')}, so zero-host-handoff and >=8-layer protected profile remain open"
                )
            hot_device_chain = textbook.get("device_resident_hot_chain_profile")
            if isinstance(hot_device_chain, dict):
                gate = hot_device_chain.get("gate") or {}
                state["benchmark_qualification"]["flash_next"] += (
                    f"; device-resident L{hot_device_chain.get('start_layer')}..L{hot_device_chain.get('end_layer')} "
                    f"probe passed terminal parity at {hot_device_chain.get('complete_wall_ns')} ns wall / "
                    f"{hot_device_chain.get('GPU_ns')} ns GPU with {hot_device_chain.get('host_roundtrip_count')} "
                    f"required host roundtrips; compact per-layer parity and >=8-layer protected gate remain open"
                )
            hot_cross_chain = textbook.get("device_resident_cross_species_profile")
            if isinstance(hot_cross_chain, dict):
                state["benchmark_qualification"]["flash_next"] += (
                    f"; device-resident cross-species L{hot_cross_chain.get('start_layer')}..L{hot_cross_chain.get('end_layer')} "
                    f"profile measured {hot_cross_chain.get('complete_wall_ns')} ns wall / {hot_cross_chain.get('GPU_ns')} ns GPU "
                    f"with {hot_cross_chain.get('host_roundtrip_count')} required host roundtrips and "
                    f"parity={hot_cross_chain.get('parity_verdict')}; >=8-layer protected and deep per-layer parity remain open"
                )
            hot_eight_chain = textbook.get("device_resident_eight_layer_profile")
            if isinstance(hot_eight_chain, dict):
                gate = hot_eight_chain.get("gate") or {}
                state["benchmark_qualification"]["flash_next"] += (
                    f"; device-resident protected L{hot_eight_chain.get('start_layer')}..L{hot_eight_chain.get('end_layer')} "
                    f"8-layer profile measured {hot_eight_chain.get('complete_wall_ns')} ns wall / "
                    f"{hot_eight_chain.get('GPU_ns')} ns GPU with {hot_eight_chain.get('host_roundtrip_count')} "
                    f"required host roundtrips; exit gate={gate.get('fastpath_exit_gate')} and compact/deep parity remain open"
                )
            hot_deep_chain = textbook.get("device_resident_deep_eight_layer_profile")
            if isinstance(hot_deep_chain, dict):
                gate = hot_deep_chain.get("gate") or {}
                state["benchmark_qualification"]["flash_next"] += (
                    f"; device-resident deep-parity L{hot_deep_chain.get('start_layer')}..L{hot_deep_chain.get('end_layer')} "
                    f"8-layer profile measured {hot_deep_chain.get('complete_wall_ns')} ns wall / "
                    f"{hot_deep_chain.get('GPU_ns')} ns GPU with {hot_deep_chain.get('host_roundtrip_count')} "
                    f"required host roundtrips; bounded FastPath exit gate={gate.get('fastpath_exit_gate')}; "
                    "complete-token, TPS, EBPW, and resident promotion remain open"
                )
            hot_complete_chain = textbook.get("device_resident_complete_chain_profile")
            if isinstance(hot_complete_chain, dict):
                state["benchmark_qualification"]["flash_next"] += (
                    f"; device-resident complete L{hot_complete_chain.get('start_layer')}..L{hot_complete_chain.get('end_layer')} "
                    f"48-layer deep-parity profile measured {hot_complete_chain.get('complete_wall_ns')} ns wall / "
                    f"{hot_complete_chain.get('GPU_ns')} ns GPU with {hot_complete_chain.get('host_roundtrip_count')} "
                    f"required host roundtrips; terminal token receipt={bool((hot_complete_chain.get('complete_token_terminal') or {}).get('receipt'))}; "
                    "accepted TPS, EBPW, and resident promotion remain open"
                )
            terminal_measurement = textbook.get("complete_token_terminal_measurement")
            if isinstance(terminal_measurement, dict):
                state["benchmark_qualification"]["flash_next"] += (
                    f"; measured single native terminal token id={terminal_measurement.get('token_id')} after 48-layer device-resident forward: "
                    f"{terminal_measurement.get('forward_wall_ns')} ns wall / {terminal_measurement.get('forward_gpu_ns')} ns GPU; "
                    "accepted multi-token TPS and complete-system EBPW remain null by contract"
                )
            if isinstance(chain_payload, dict) and chain_payload.get("status") == "PASSED":
                state["benchmark_qualification"]["flash_next"] += (
                    "; single-process L0-L3 seam passed with explicit f32 host state handoffs; "
                    "streamed device-resident whole-token runtime remains open"
                )
            if next_layer <= 47:
                gate_note = (
                    f"Exact source-BF16 Flash boundaries through layer {latest_layer} are "
                    f"physically verified with explicit state handoffs. Continue at layer {next_layer}, "
                    "then earn the first complete native token; complete-token, EBPW, accepted-TPS, "
                    "and resident qualification remain open."
                )
            else:
                gate_note = (
                    f"Exact source-BF16 Flash boundaries through layer {latest_layer} are physically "
                    "verified with explicit state handoffs. The layer chain is closed; attempt the "
                    "first complete native token, then qualify EBPW, accepted-TPS, and residency."
                )
            for gate_list_name in ("next_work", "next_decisive_gates", "open_workunits"):
                gate_list = state.get(gate_list_name)
                if isinstance(gate_list, list):
                    for gate in gate_list:
                        if isinstance(gate, dict) and "Flash-Next complete exact layer" in str(gate.get("gate", "")):
                            gate["why"] = gate_note

    # The scoreboard is a compact, receipt-derived platform view.  Register its
    # identity and headline counts without copying the whole row set into the
    # civilization ledger; the source receipt remains the detailed authority.
    scoreboard_path = rec / "ACCELERATOR_SCOREBOARD.json"
    try:
        scoreboard_payload = json.loads(scoreboard_path.read_text())
    except (OSError, json.JSONDecodeError):
        scoreboard_payload = None
    if isinstance(scoreboard_payload, dict) and scoreboard_payload.get("schema") == "hawking.accelerator.scoreboard.v1":
        scoreboard_ref = str(scoreboard_path.relative_to(HAWKING))
        if scoreboard_ref not in state["accepted_receipts"]:
            state["accepted_receipts"].append(scoreboard_ref)
        textbook = state["active_textbooks"].get("Qwen3.8-Flash-Next")
        if isinstance(textbook, dict):
            evidence = textbook.setdefault("cross_layer_evidence", "")
            if scoreboard_ref not in evidence:
                textbook["cross_layer_evidence"] = f"{evidence} + {scoreboard_ref}" if evidence else scoreboard_ref
            textbook["accelerator_scoreboard"] = {
                "receipt": scoreboard_ref,
                "status": scoreboard_payload.get("status"),
                "rows": len(scoreboard_payload.get("rows") or []),
                "frontier_rows": len(scoreboard_payload.get("frontier_receipts") or []),
                "promotion_winner": (scoreboard_payload.get("physical_plan_score") or {}).get("winner"),
                "promotion_allowed": (scoreboard_payload.get("physical_plan_score") or {}).get("promotion_allowed"),
                "claim": scoreboard_payload.get("claim_boundary"),
            }
        state["accelerator_scoreboard"] = {
            "receipt": scoreboard_ref,
            "status": scoreboard_payload.get("status"),
            "rows": len(scoreboard_payload.get("rows") or []),
            "frontier_rows": len(scoreboard_payload.get("frontier_receipts") or []),
            "promotion_winner": (scoreboard_payload.get("physical_plan_score") or {}).get("winner"),
            "promotion_allowed": (scoreboard_payload.get("physical_plan_score") or {}).get("promotion_allowed"),
            "claim": scoreboard_payload.get("claim_boundary"),
        }

    return state


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
