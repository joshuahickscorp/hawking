"""Generate PART I of the recompiled roadmap from the audited capability graph.

PART I answers "what actually exists now?", and the directive is explicit that
nothing counts as complete because a file exists. So this does not narrate: it
reads civilization/CAPABILITY_GRAPH.json -- which the adversarial auditor built
from HEAD blobs, not from the working tree -- and prints, per capability, the
ten fields the directive names. A field with no evidence prints as absent rather
than as prose, because an unknown that reads like a sentence is how a roadmap
starts lying.

The defining property is quoted from the OLD roadmap's proof-obligation span for
that gate, so PART I inherits the obligation the campaign was actually judged
against rather than a fresh paraphrase of it.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / "civilization" / "CAPABILITY_GRAPH.json"
OLD_ROADMAP = Path.home() / "Downloads" / "H-ROADMAP.md"

# The seven organizational VIEWS the directive asks for. These are lenses on the
# existing genes, deliberately not new civilizations -- the gene stays authority.
VIEWS: list[tuple[str, str, tuple[str, ...]]] = [
    ("A", "CONTROL / HCLI / AGENTOS", ("I-A_AGENTOS_HCLI",)),
    ("B", "MODEL SCIENCE / DOCTOR / GRAVITY / NOETIC", ("I-C_GRAVITY_NOETIC",)),
    ("C", "NATIVE RUNTIME / APPLE ACCELERATOR", ("I-D_ACCELERATOR",)),
    ("D", "SCIENCE MEMORY / MODELLAKE / ODYSSEY",
     ("I-E_ODYSSEY_I", "II-A_ODYSSEY_II", "III-A_ODYSSEY_III")),
    ("E", "HARDWARE COMPILER / U50DD PREBOARD", ()),        # by id prefix, below
    ("F", "HETEROGENEOUS MACHINE / HMF / HGVAS / FUSION", ("IV-A_FUSION", "IV-B_HMF_HGVAS")),
    ("G", "PERCEPTION / PRODUCT / VMCP / THEIA / PLATFORM", ()),
]
E_PREFIXES = ("U50_", "FPGA_")
G_PREFIXES = ("VMCP_", "THEIA_")


def view_of(gate: dict) -> str:
    gid = gate["id"]
    if gid.startswith(E_PREFIXES):
        return "E"
    if gid.startswith(G_PREFIXES):
        return "G"
    gene = str(gate.get("gene"))
    for letter, _title, genes in VIEWS:
        if gene in genes:
            return letter
    return "G"


def status_of(gate: dict) -> str:
    """Map the auditor's vocabulary onto the directive's, without inflating.

    The auditor's BUILT already means wired AND accepted, so it is the only thing
    that may read as integrated. WIRED means a real non-test caller exists but the
    gate's own acceptance criterion was never demonstrated -- that is built, not
    verified-integrated, and it must not be allowed to read as the latter.
    PHYSICALLY_MEASURED is unreachable today by construction: every gate in the
    graph is evidence_tier STATIC.
    """
    status = gate["status"]
    tier = gate.get("evidence_tier")
    if status == "BUILT":
        return "PHYSICALLY_MEASURED" if tier == "HARDWARE_MEASURED" else "VERIFIED_INTEGRATED"
    return {
        "WIRED": "VERIFIED_BUILT",
        "SCAFFOLDED": "SCAFFOLDED",
        "BLOCKED_HARDWARE": "BLOCKED_HARDWARE",
        "BLOCKED_EXTERNAL": "BLOCKED_EVIDENCE",
        "ABSENT": "ABSENT",
        "UNREACHABLE": "UNREACHABLE",
    }.get(status, status)


def defining_property(gate: dict, roadmap_lines: list[str]) -> str:
    """Quote the gate's proof obligation from the old roadmap, first prose line."""
    span = gate.get("acceptance_span") or {}
    start, end = span.get("start_line"), span.get("end_line")
    if not start or not end or start > len(roadmap_lines):
        return ""
    for raw in roadmap_lines[start - 1 : min(end, len(roadmap_lines))]:
        line = raw.strip()
        if not line or line.startswith(("#", "```", "|", "---", "==")):
            continue
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\[[ xX]\]\s*", "", line)
        if len(line) > 20:
            return line
    return ""


def refs(items, limit=3):
    """Render evidence refs. Some lists hold plain strings, some hold dicts."""
    out = []
    for item in (items or [])[:limit]:
        if isinstance(item, str):
            out.append(item)
            continue
        f, ln = item.get("file"), item.get("line")
        out.append(f"{f}:{ln}" if ln and ln > 1 else str(f))
    extra = len(items or []) - len(out)
    return ", ".join(out) + (f" (+{extra} more)" if extra > 0 else "") if out else ""


def negative_control(gate: dict) -> str:
    """A negative control is a test that proves the gate can FAIL, so look for one.

    Detected by reading the cited test files for refusal/attack/mutation language
    rather than by filename: a name is a claim, the body is the evidence.
    """
    want = ("refus", "negative control", "must fail", "adversar", "attack",
            "mutation", "does not count", "is not a pass")
    found = []
    for test in gate.get("tests") or []:
        rel = test if isinstance(test, str) else test.get("file")
        path = REPO / str(rel)
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace").lower()
        except OSError:
            continue
        if any(w in text for w in want):
            found.append(str(rel))
    return ", ".join(sorted(set(found))[:2]) if found else ""


def limitations(gate: dict) -> str:
    bits = []
    for key in ("hardware_blocker", "software_blocker"):
        if gate.get(key):
            bits.append(str(gate[key]))
    accepted = gate.get("accepted") or {}
    if not accepted.get("value"):
        for ev in accepted.get("evidence") or []:
            if ev.get("note"):
                bits.append(f"acceptance not demonstrated: {str(ev['note'])[:160]}")
                break
    return " | ".join(bits)


def render() -> str:
    graph = json.loads(GRAPH.read_text())
    gates = graph["gates"]
    roadmap_lines = OLD_ROADMAP.read_text(errors="replace").splitlines() if OLD_ROADMAP.is_file() else []

    by_view: dict[str, list[dict]] = defaultdict(list)
    for gate in gates.values():
        by_view[view_of(gate)].append(gate)

    counts = Counter(status_of(g) for g in gates.values())
    out: list[str] = []
    add = out.append

    add("# PART I — VERIFIED HAWKING TODAY")
    add("")
    add("Generated by `tools/roadmap/recompile.py` from `civilization/CAPABILITY_GRAPH.json`,")
    add("which the adversarial auditor builds from HEAD blobs rather than the working tree.")
    add("Nothing here is complete because a file exists: a field with no evidence prints as")
    add("`absent`, never as prose.")
    add("")
    add("## Status census")
    add("")
    for status, n in counts.most_common():
        add(f"    {status:24} {n}")
    add("")
    measured = counts.get("PHYSICALLY_MEASURED", 0)
    add(f"PHYSICALLY_MEASURED = {measured}. Every gate in the graph is evidence_tier STATIC,")
    add("so no present capability rests on a physical measurement. This is the floor by")
    add("design, not by neglect: simulated is not measured.")
    add("")
    add("## Evidence coverage across all gates")
    add("")
    add(f"    defining property available    {sum(1 for g in gates.values() if defining_property(g, roadmap_lines)):3} / {len(gates)}")
    add(f"    real (non-test) caller         {sum(1 for g in gates.values() if g.get('runtime_caller')):3} / {len(gates)}")
    add(f"    any verifier                   {sum(1 for g in gates.values() if g.get('tests')):3} / {len(gates)}")
    add(f"    receipt cited                  {sum(1 for g in gates.values() if g.get('receipts_cited')):3} / {len(gates)}")
    add("")

    for letter, title, _genes in VIEWS:
        members = sorted(by_view.get(letter, []), key=lambda g: g["id"])
        if not members:
            continue
        add(f"## {letter}. {title}")
        add("")
        add(f"{len(members)} capabilities. " + ", ".join(
            f"{s}={n}" for s, n in Counter(status_of(g) for g in members).most_common()))
        add("")
        for gate in members:
            add(f"### {gate['id']}")
            add("")
            add(f"    STATUS              {status_of(gate)}")
            prop = defining_property(gate, roadmap_lines)
            add(f"    defining property   {prop or 'absent'}")
            add(f"    implementation      {refs(gate.get('code_refs')) or 'absent'}")
            add(f"    real caller         {refs(gate.get('runtime_caller')) or 'absent — no non-test call site'}")
            add(f"    verifier            {refs(gate.get('tests')) or 'absent — no test cites this gate'}")
            add(f"    negative control    {negative_control(gate) or 'absent — no cited test proves it can fail'}")
            add(f"    receipt             {refs(gate.get('receipts_cited')) or 'absent'}")
            add(f"    evidence level      {gate.get('evidence_tier')}")
            add(f"    limitations         {limitations(gate) or 'none recorded'}")
            wired = bool((gate.get('wired') or {}).get('value'))
            accepted = bool((gate.get('accepted') or {}).get('value'))
            add(f"    integration         wired={wired} accepted={accepted}")
            add("")
    return "\n".join(out) + "\n"


def build() -> Path:
    out = REPO / "docs" / "roadmap" / "PART_I_VERIFIED_TODAY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render())
    return out




# ---------------------------------------------------------------------------
# PART II — the optimized remaining action plan.
#
# The directive's primary objective is to drive SOFTWARE_CONNECTION_REMAINING
# toward zero, so the classification has to be DERIVED from evidence rather than
# assigned by hand -- a hand-assigned class is just an opinion about difficulty.
#
# The rule below is deliberately blunt: what is the gate actually missing?
#   missing a caller or a verifier  -> the gap is CODE. Software connection.
#   has both, acceptance not shown  -> the gap is a RUN. Experimentation.
#   blocked on absent silicon       -> hardware, and no amount of code helps.
#   blocked on an absent trained model or campaign -> long-run evidence.
#   no implementation at all AND no defining property -> unknown research.
# ---------------------------------------------------------------------------

BLOCKER_CLASSES = (
    "SOFTWARE_CONNECTION_REMAINING",
    "EXPERIMENTATION_REQUIRED",
    "LONG_RUN_EVIDENCE_REQUIRED",
    "PHYSICAL_HARDWARE_REQUIRED",
    "UNKNOWN_RESEARCH",
)


def blocker_class(gate: dict) -> tuple[str, str]:
    """(class, the exact thing that is missing). Derived, never assigned."""
    if gate.get("hardware_blocker") or gate["status"] == "BLOCKED_HARDWARE":
        wake = gate.get("wake_condition") or "the board"
        return "PHYSICAL_HARDWARE_REQUIRED", f"silicon absent; wakes on {wake}"

    if gate["status"] == "BLOCKED_EXTERNAL":
        blocker = str(gate.get("software_blocker") or "")
        # A THEIA rung needs a TRAINED MODEL, which is wall time and compute, not wiring.
        return "LONG_RUN_EVIDENCE_REQUIRED", blocker[:200] or "external substrate absent"

    wired = bool((gate.get("wired") or {}).get("value"))
    accepted = bool((gate.get("accepted") or {}).get("value"))
    has_impl = bool(gate.get("code_refs"))
    has_test = bool(gate.get("tests"))

    if not has_impl and not has_test:
        return "UNKNOWN_RESEARCH", "no implementation and no verifier exist yet"
    if not wired:
        return "SOFTWARE_CONNECTION_REMAINING", "no non-test call site reaches this capability"
    if not has_test:
        return "SOFTWARE_CONNECTION_REMAINING", "wired but nothing verifies it"
    if not accepted:
        return "EXPERIMENTATION_REQUIRED", "wired and verified; its acceptance criterion has never been run"
    return "", "already integrated"


def render_part_ii() -> str:
    graph = json.loads(GRAPH.read_text())
    gates = graph["gates"]
    rows = []
    for gate in gates.values():
        cls, missing = blocker_class(gate)
        if cls:
            rows.append((cls, gate, missing))

    counts = Counter(cls for cls, _g, _m in rows)
    out: list[str] = []
    add = out.append
    add("# PART II — OPTIMIZED REMAINING ACTION PLAN")
    add("")
    add("Organized by what BLOCKS each action, not by era. The class is derived from the")
    add("gate's own evidence -- a hand-assigned class is an opinion about difficulty, and")
    add("the whole point of this part is to separate 'I have not connected these two")
    add("components yet' from work that genuinely needs an experiment, wall time, or silicon.")
    add("")
    add("## Blocker census")
    add("")
    for cls in BLOCKER_CLASSES:
        add(f"    {cls:32} {counts.get(cls, 0)}")
    add("")
    add(f"    {'TOTAL REMAINING':32} {len(rows)}")
    add("")
    add(f"SOFTWARE_CONNECTION_REMAINING = {counts.get('SOFTWARE_CONNECTION_REMAINING', 0)} is the")
    add("number this campaign exists to drive toward zero. Every other class is honest")
    add("frontier: an experiment that must run, evidence that needs wall time, hardware that")
    add("must physically exist, or a question whose answer nobody has.")
    add("")

    for cls in BLOCKER_CLASSES:
        members = sorted((r for r in rows if r[0] == cls), key=lambda r: r[1]["id"])
        if not members:
            continue
        add(f"## {cls} ({len(members)})")
        add("")
        for _cls, gate, missing in members:
            add(f"### {gate['id']}")
            add(f"    missing             {missing}")
            add(f"    shortest verifier   {refs(gate.get('tests'), 1) or 'must be written'}")
            add(f"    implementation      {refs(gate.get('code_refs'), 2) or 'absent'}")
            deps = gate.get("dependencies") or []
            add(f"    unlocks             {len(deps)} declared dependencies")
            add("")
    return "\n".join(out) + "\n"


def build_part_ii() -> Path:
    out = REPO / "docs" / "roadmap" / "PART_II_ACTION_PLAN.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_part_ii())
    return out


if __name__ == "__main__":
    print(build())
    print(build_part_ii())
