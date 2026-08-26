"""Adversarial validator for ROADMAP_STATE.json.

Its job is NOT to confirm the ledger looks tidy. It is to REFUSE a ledger that
inflates. Every rule here exists because the failure it names is one this program
has actually committed: a percentage computed from file counts, a status claimed
while its gate is open, a test count written from arithmetic instead of a run.
"""
import json, pathlib, re, subprocess, sys

CATEGORIES = 9


def validate(state, goal_text=None):
    """Return list of violations. Empty list = the ledger is defensible."""
    v = []
    g = lambda k: state.get(k)

    if g("active_era") != "I":
        v.append(f"active_era is {g('active_era')}: Era I is sovereign until its gates close")

    for name, c in state.get("civilization_status", {}).items():
        ev, ob, comp = c.get("evidence_pct"), c.get("obligation_pct"), c.get("completion_pct")
        if ev is None:
            continue
        # THE INFLATION RULE. Reporting evidence coverage as completion is exactly
        # what the first build of this ledger did for I-D: 9/9 categories, 0/8
        # obligations, printed 100%.
        if comp != min(ev, ob):
            v.append(f"{name}: completion {comp} != min(evidence {ev}, obligations {ob}) -- inflation")
        if c.get("status") == "CIVILIZATION_COMPLETE" and c.get("open_gates"):
            v.append(f"{name}: CIVILIZATION_COMPLETE with {len(c['open_gates'])} open gates")
        # round to the SAME precision the ledger stores. Comparing a rounded value
        # against an unrounded one by exact equality reports arithmetic as a finding --
        # a defect this program sealed once already on a noisy pagein counter.
        if c.get("evidence") and round(100 * sum(c["evidence"].values()) / CATEGORIES, 1) != ev:
            v.append(f"{name}: evidence_pct does not match its own category table")
        if len(c.get("open", [])) and c.get("obligation_pct") == 100:
            v.append(f"{name}: obligation_pct 100 with {len(c['open'])} open obligations")

    # ERA SOVEREIGNTY. Later-era work is permitted when it is already running,
    # consumes an idle resource, produces infrastructure Era I needs, or resolves an
    # uncertainty that changes Era-I design -- and it NEVER earns civilization
    # completion. Without this rule that law is prose: Fusion, HMF and eGPU all
    # carry real receipts and would otherwise be free to report a percentage.
    era = g("active_era")
    for name, c in state.get("civilization_status", {}).items():
        if name.startswith(f"{era}-"):
            continue
        if name in state.get("active_civilizations", []):
            v.append(f"{name} is not in the sovereign era ({era}) but is listed active -- "
                     "later-era work does not become an active civilization")
        if c.get("completion_pct") is not None:
            v.append(f"{name} is later-era and reports completion_pct "
                     f"{c['completion_pct']} -- later-era work NEVER earns civilization "
                     "completion while Era I is sovereign")
        if c.get("status") in ("ADVERSARIALLY_VERIFIED", "INTEGRATED", "CIVILIZATION_COMPLETE"):
            v.append(f"{name} is later-era with status {c['status']} -- advance work is "
                     "tracked, not graduated")

    if state.get("unmapped_obligations"):
        v.append(f"unmapped obligations: {state['unmapped_obligations']} -- every "
                 "obligation must land in the era topology or the map is a fiction")
    if state.get("orphan_map_entries"):
        v.append(f"orphan map entries: {state['orphan_map_entries']} -- named in the map, "
                 "absent from GOAL.md")

    if not state.get("test_count_is_from_a_run_not_arithmetic"):
        v.append("test count not marked as coming from a run (S015 XI: the run is evidence)")
    if not isinstance(state.get("last_verified_test_count"), int):
        v.append("last_verified_test_count is not an integer from a real pytest run")
    # A count without its interpreter is not a measurement. The same suite reports
    # 5 failed under the default python3 (3.14, no mlx) and all passing under the
    # framework 3.12 -- so the number alone cannot be compared across runs.
    te = state.get("test_environment")
    if not te:
        v.append("test count arrives with no test_environment -- which interpreter?")
    else:
        if "mlx NOT importable" in str(te.get("version_and_mlx")):
            v.append(f"tests were counted under an interpreter without mlx "
                     f"({te.get('interpreter')}) -- that suite cannot pass there")
        if te.get("failed"):
            v.append(f"{te['failed']} tests FAILED in the run this count came from; a "
                     "passed-count published unreported failures is a half-truth")

    # Status counts must match the authority on disk, not this file.
    if goal_text is not None:
        real = {}
        for m in re.finditer(r"status: ([A-Z_]+)", goal_text):
            real[m.group(1)] = real.get(m.group(1), 0) + 1
        if real != state.get("obligation_status_counts"):
            v.append(f"status counts {state.get('obligation_status_counts')} disagree with "
                     f"GOAL.md {real} -- disk state is authority")

    if not state.get("named_gates"):
        v.append("no named gates: S015 §129 requires a blocker to name its exact missing input")

    # --- rules for the directive VIII fields ------------------------------------
    # Each of these exists because the field can lie in a specific way, and a field
    # the validator does not check is a field that will.

    # A percentage must be derivable from evidence categories or LABELLED heuristic.
    cp = state.get("civilization_progress")
    if cp is None:
        v.append("no civilization_progress: directive VIII requires the coordinate")
    elif not cp.get("heuristic"):
        v.append("civilization_progress is not labelled heuristic -- a civilizational "
                 "coordinate is not derivable from evidence categories, so it must say so")
    elif not cp.get("basis"):
        v.append("civilization_progress has no basis: an unexplained percentage is a number "
                 "somebody will later mistake for a measurement")

    # "No runtime" and "storage slow" are shrugs, not blockers. A blocker must carry a
    # quantity or name the exact absent dependency (directive XII).
    QUANTIFIED = re.compile(r"\d|absent|not available|no local|does not exist|unavailable", re.I)
    for b in state.get("blockers", []):
        q = b.get("quantified_as", "")
        if not QUANTIFIED.search(q):
            v.append(f"blocker {b.get('gate')} is not quantified: {q!r} -- directive XII "
                     "refuses 'no runtime' and 'storage slow' as blockers")
        if not b.get("blocks"):
            v.append(f"blocker {b.get('gate')} blocks nothing -- then it is trivia, not a gate")

    # Every active civilization needs a dependency entry, even an empty one. A missing
    # key is indistinguishable from an unconsidered one.
    deps = state.get("dependencies")
    if deps is None:
        v.append("no dependencies field")
    else:
        for name in state.get("active_civilizations", []):
            if name not in deps:
                v.append(f"{name} has no dependency entry -- absent is not the same as none")

    # Running lanes are re-derived HERE, independently of the builder. This is the
    # census rule the directive names: running work not represented in state. A lane
    # listed as alive whose process is gone is a stale claim, and this program has
    # already been burned by a status file reporting dead lanes as running.
    lanes = state.get("running_lanes")
    if lanes is None:
        v.append("no running_lanes field -- the census must represent running work")
    else:
        try:
            cmds = subprocess.run(["ps", "-axo", "command"], capture_output=True,
                                  text=True, timeout=20).stdout.splitlines()
        except Exception:
            cmds = None
        for L in lanes:
            ex, det = L.get("executor"), L.get("detection")
            if ex not in ("grok", "claude", "resident"):
                v.append(f"running lane {L.get('lane')} names no known executor: {ex!r}")
                continue
            if det not in ("definitive", "heuristic"):
                v.append(f"running lane {L.get('lane')} does not say how it was detected -- "
                         "a definitive pid check and an mtime guess must not read alike")
                continue

            if ex == "grok":
                tf = L.get("task_file")
                # An EMPTY task_file made `tf in c` true for every line, so a lane
                # naming no task file passed silently whenever any grok process
                # existed. Require the path before trusting the check.
                if not tf:
                    v.append(f"grok lane {L.get('lane')} names no task_file -- the pid check "
                             "cannot run and would pass vacuously")
                elif cmds is not None and L.get("alive") and not any(
                        c.startswith("grok ") and tf in c for c in cmds):
                    v.append(f"running lane {L.get('lane')} claims alive but no live process "
                             "holds its task file -- a status file is not a pid")

            elif ex == "resident" and not L.get("judged_by"):
                v.append(f"resident lane {L.get('lane')} says nothing about how it was "
                         "detected; a process that COMMITS to this repo must be named exactly")

            elif ex == "claude" and det != "heuristic":
                # There is no pid for a workflow agent. Anything claiming definitive
                # is claiming a check that does not exist.
                v.append(f"claude lane {L.get('lane')} claims {det} detection, but a workflow "
                         "agent has no pid to check -- mtime recency is heuristic")

    gates = state.get("next_decisive_gates")
    if not gates:
        v.append("no next_decisive_gates -- a ledger that cannot say what is next is a report")
    else:
        ranks = [g.get("rank") for g in gates]
        if sorted(ranks) != list(range(1, len(gates) + 1)):
            v.append(f"next_decisive_gates ranks are not a clean 1..N: {ranks}")
        for g in gates:
            if not g.get("resource"):
                v.append(f"gate rank {g.get('rank')} names no resource -- resource conflict "
                         "is half of the ranking and cannot be left implicit")

    laws = state.get("laws_since_last_checkpoint")
    if laws is None:
        v.append("no laws_since_last_checkpoint field")
    elif "mtime" in str(laws.get("basis", "")) and not laws.get("heuristic"):
        v.append("laws_since_last_checkpoint is derived from mtime but not labelled "
                 "heuristic -- mtime is not provenance")

    # ANTI-VACUITY. This receipt corpus is KNOWN to supersede itself: receipts carry
    # AMENDED_IN_PLACE markers and retracted results. A detector that finds none of
    # them across a large corpus is broken, not lucky.
    retr = state.get("unresolved_retractions")
    if retr is None:
        v.append("no unresolved_retractions field")
    elif not retr and state.get("accelerator_receipt_count", 0) > 20:
        v.append(f"zero retractions found across {state.get('accelerator_receipt_count')} "
                 "accelerator receipts -- this corpus is known to contain retracted and "
                 "amended results, so the detector is broken, not the corpus clean")

    ce = state.get("completion_evidence")
    if ce is None:
        v.append("no completion_evidence field")
    else:
        for name in state.get("active_civilizations", []):
            e = ce.get(name)
            if not e:
                v.append(f"{name} has no completion_evidence entry")
            elif not e.get("note"):
                v.append(f"{name} completion_evidence has no note -- a bare category count "
                         "is what inflated I-D to 100% once already")
    return v


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    st = json.loads((here / "ROADMAP_STATE.json").read_text())
    goal = (pathlib.Path.home() / ".claude/ultragoal/hawking-odyssey-maxx-ascension/GOAL.md").read_text()
    bad = validate(st, goal)
    print("\n".join(f"VIOLATION: {b}" for b in bad) or "ledger defensible: 0 violations")
    sys.exit(1 if bad else 0)
