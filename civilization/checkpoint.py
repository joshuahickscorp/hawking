"""Generate ERA_I_CHECKPOINT_N.json from disk truth.

ERA_I_CHECKPOINT_001 was HAND-WRITTEN. That is the defect this file closes: a
hand-written checkpoint needs the originating prompt repeated to produce the next
one, and the directive's whole deliverable is a control plane that can drive
CHECKPOINT_002 without that.

The split is explicit and is the point:

    DERIVED   obligation status, evidence categories, percentages, receipts
              landed, regressions, running lanes, resource ownership. Computed
              here from ROADMAP_STATE.json, which is itself computed from
              GOAL.md + receipts + git + a real pytest run.

    AUTHORED  what became physically true, what was refuted, what changed in the
              roadmap. A machine cannot read a receipt and know which of its
              numbers was the point. These are written by whoever runs the
              checkpoint and are REQUIRED to cite receipts, which the validator
              enforces.

An authored field that cites nothing is prose, and prose is not evidence. A
derived field a human can retype is a field that will lie with confidence --
`resource_ownership` already did exactly that, carrying "4 hf download workers"
as a literal while the fill had changed shape underneath it.
"""
import json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
STATE = HERE / "ROADMAP_STATE.json"

# An authored claim must point at something on disk. Receipt names, obligation
# ids, and repo-relative tool paths all count; an adjective does not.
# The repo's real top-level directories, named explicitly. A generic "any path"
# pattern would make this rule vacuous, and omitting `civilization/` made it
# reject correctly-cited claims about the control plane itself -- caught by
# running the real authored block through it.
CITATION = re.compile(
    r"[A-Z0-9_]+\.json"
    r"|G\d{3}"
    r"|(?:tools|receipts|civilization|crates|hcli|lab|workspace|docs)/[\w/.-]+"
    r"|test_\w+")


def previous(n):
    """The newest checkpoint before N, or None. Schema is NOT assumed stable --
    CHECKPOINT_001 was hand-written with a different shape, and a comparison that
    crashes on an older schema is a comparison nobody will run."""
    best = None
    for f in sorted(HERE.glob("ERA_I_CHECKPOINT_*.json")):
        m = re.search(r"_(\d+)\.json$", f.name)
        if not m or int(m.group(1)) >= n:
            continue
        try:
            best = (f.name, json.loads(f.read_text()))
        except Exception:
            continue
    return best


def _civ_pcts(doc):
    """Pull {civilization: completion_pct} out of either schema."""
    out = {}
    for key in ("civilizations", "civilization_status"):
        for name, c in (doc.get(key) or {}).items():
            if isinstance(c, dict) and c.get("completion_pct") is not None:
                out[name] = c["completion_pct"]
    return out


def regressions(state, prev):
    """Things that went BACKWARD. A checkpoint that can only report progress is a
    press release; this program has shipped enough retractions to know the
    difference."""
    if not prev:
        return {"basis": "no previous checkpoint", "found": []}
    name, doc = prev
    found = []

    old_t, new_t = doc.get("last_verified_test_count"), state.get("last_verified_test_count")
    if isinstance(old_t, int) and isinstance(new_t, int) and new_t < old_t:
        found.append(f"test count fell {old_t} -> {new_t}")

    old_p, new_p = _civ_pcts(doc), _civ_pcts(state)
    for civ, was in old_p.items():
        now = new_p.get(civ)
        if now is not None and now < was:
            found.append(f"{civ} completion fell {was} -> {now}")

    old_v = doc.get("obligation_status_counts", {}).get("VERIFIED")
    new_v = state.get("obligation_status_counts", {}).get("VERIFIED")
    if isinstance(old_v, int) and isinstance(new_v, int) and new_v < old_v:
        found.append(f"VERIFIED obligations fell {old_v} -> {new_v}")

    return {"basis": f"compared against {name}", "found": found}


def build(n, authored):
    state = json.loads(STATE.read_text())
    prev = previous(n)

    civs = {}
    for name in state["active_civilizations"]:
        c = state["civilization_status"][name]
        civs[name] = {
            "status": c["status"],
            "completion_pct": c["completion_pct"],
            "evidence_pct": c["evidence_pct"],
            "obligation_pct": c["obligation_pct"],
            # Every percentage traces to its category table or its obligation
            # count. Directive VIII: no percentage without what changed beneath it.
            "percentage_traces_to": {
                "evidence_pct": f"{c['evidence_satisfied']}/{c['evidence_of']} evidence categories",
                "obligation_pct": f"{c['verified']}/{len(c['obligations'])} obligations VERIFIED",
                "completion_pct": "min(evidence_pct, obligation_pct) -- the MINIMUM, so "
                                  "category breadth cannot stand in for a closed gate",
            },
            "open_obligations": c["open"],
            "open_gates": c["open_gates"],
            "note": c["note"],
        }

    return {
        "checkpoint": f"ERA_I_CHECKPOINT_{n:03d}",
        "generated_by": "civilization/checkpoint.py from ROADMAP_STATE.json",
        "hand_written": False,
        "roadmap": state["roadmap_version"],
        "active_era": state["active_era"] + " (SOVEREIGN)",
        "last_verified_commit": state["last_verified_commit"],
        "last_verified_test_count": state["last_verified_test_count"],
        "civilization_progress": state["civilization_progress"],

        "civilizations": civs,

        "evidence_gained": state["laws_since_last_checkpoint"],
        "blockers": state["blockers"],
        "dependencies": state["dependencies"],
        "unresolved_retractions_count": len(state["unresolved_retractions"]),
        "regressions": regressions(state, prev),

        "physical_resource_state": {
            "resource_ownership": state["resource_ownership"],
            "running_lanes": state["running_lanes"],
        },
        "next_decisive_wave": state["next_decisive_gates"],

        # The authored half. Required to cite; see CITATION.
        "authored": authored,
        "authored_vs_derived": (
            "Everything outside `authored` is DERIVED from ROADMAP_STATE.json, which is "
            "itself derived from GOAL.md + receipts + git + a real pytest run. The "
            "`authored` block is human/agent judgement and every claim in it must cite a "
            "receipt, obligation id or tool path -- the validator refuses it otherwise."),
    }


def validate(cp):
    """Refuse a checkpoint that inflates. Same law as validate.py: a rule nobody
    has watched refuse is decoration, so every rule here has a mutation test."""
    v = []

    if cp.get("hand_written"):
        v.append("checkpoint is hand-written -- the control plane must be able to "
                 "produce the next one without the originating prompt")

    for name, c in cp.get("civilizations", {}).items():
        if c.get("completion_pct") != min(c.get("evidence_pct", 0), c.get("obligation_pct", 0)):
            v.append(f"{name}: completion is not min(evidence, obligation) -- inflation")
        if not c.get("percentage_traces_to"):
            v.append(f"{name}: percentages do not trace to an evidence category "
                     "(directive VIII: never report a generated percentage without "
                     "showing what changed beneath it)")
        if c.get("status") == "CIVILIZATION_COMPLETE" and c.get("open_gates"):
            v.append(f"{name}: CIVILIZATION_COMPLETE with open gates")

    if cp.get("regressions", {}).get("basis") is None:
        v.append("regressions section has no basis -- 'none found' is only meaningful "
                 "against a named prior checkpoint")

    a = cp.get("authored")
    if not a:
        v.append("no authored block -- a checkpoint that reports only derived numbers "
                 "cannot say what became true")
    else:
        for field in ("what_became_physically_true", "what_was_refuted",
                      "what_changed_in_the_roadmap"):
            vals = a.get(field)
            if not vals:
                v.append(f"authored.{field} is empty -- required by directive XV")
                continue
            for claim in vals:
                if not CITATION.search(claim):
                    v.append(f"authored.{field} claim cites nothing: {claim[:70]!r} -- "
                             "prose is not evidence")

    if cp.get("civilization_progress", {}).get("heuristic") is not True:
        v.append("civilization_progress not labelled heuristic")

    if not cp.get("next_decisive_wave"):
        v.append("no next_decisive_wave -- a checkpoint that cannot say what is next "
                 "leaves the next session needing the prompt again")

    return v


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    src = HERE / f"authored_{n:03d}.json"
    if not src.is_file():
        sys.exit(f"authored block missing: {src}\nThe derived half is free; the "
                 f"judgement half is not, and a checkpoint without it is a status dump.")
    cp = build(n, json.loads(src.read_text()))
    bad = validate(cp)
    if bad:
        print("\n".join(f"VIOLATION: {b}" for b in bad))
        sys.exit(1)
    out = HERE / f"ERA_I_CHECKPOINT_{n:03d}.json"
    out.write_text(json.dumps(cp, indent=1))
    r = cp["regressions"]["found"]
    print(f"wrote {out.name} | 0 violations | regressions: {len(r)}")
    for name, c in cp["civilizations"].items():
        print(f"  {name:20s} {c['completion_pct']:5.1f}%  {c['status']}")
