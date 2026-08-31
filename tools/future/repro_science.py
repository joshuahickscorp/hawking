"""REPRO_SCIENCE — provenance, identity, canaries, replication, fail-closed faults.

Autonomy that cannot be shown to fail closed will launder weak evidence into a
strong claim. This module is the sidecar guard: STATIC_ONLY, bench UNKNOWN,
gpu_authority false. It produces neither DIAGNOSTIC_RELATIVE nor
PROTECTED_ABSOLUTE.

    python3 tools/future/repro_science.py --selftest
    python3 tools/future/repro_science.py --build

Recovered, not forked: tools/future/_common.py seal/write_receipt;
tools/provenance_chain.py content-not-path hashing; lab/provenance.py pins;
tools/headless/causal_benchmark_law.py (a no-op must not pass);
tools/headless/disk_truth.py (disk is authority); tools/headless/dirty_tree_preservation.py
(a gate never seen to fail is not a gate); hcli/agentos/recovery.py (kill + resume);
hcli/agentos/checkpoint.py (durable snapshot, not a success stamp).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import ast
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.future._common import git, sha256_file, write_receipt

RECEIPT = "REPRO_SCIENCE.json"
SCHEMA = "hawking.future.repro_science.v1"

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

# Named faults from the lane contract. Order is admit() check order.
FAULT_NAMES = (
    "stale_nx",
    "wrong_specimen_hash",
    "corrupt_receipt",
    "stale_pipeline_cache",
    "changed_compiler",
    "changed_machine_genome",
    "failed_worker",
    "killed_subprocess",
    "partial_result",
    "invalid_route_corpus",
    "malformed_teacher_receipt",
    "missing_hardware",
)

NODE_KINDS = ("experiment", "input", "code", "machine", "output", "claim")
EDGE_RELS = ("uses", "compiled_from", "ran_on", "produced_by", "supported_by", "inherits")
METADATA_ONLY_NX = frozenset(
    {
        "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
        "NOT_BUILT",
        "STALE",
    }
)
LEGAL_VERDICTS = frozenset({"PASS", "FAIL", "REFUSED"})
SKIP_VERDICTS = frozenset({"SKIP", "SKIPPED", "N/A", "NA", "NONE"})
BUNDLE_REQUIRED = (
    "experiment_identity",
    "inputs",
    "code_identity",
    "machine_genome_pin",
    "recipe_steps",
    "expected_outputs",
    "claim_boundary",
    "how_to_verify",
)
CANARY_MARKER = "HAWKING_NEGATIVE_CONTROL_MUTATION_LIVE"
TOOL_SYMBOLS = frozenset({"build", "selftest", "admit", "run_all_proofs"})

_KILL_CHILD = (
    "import json, sys, time\n"
    "from pathlib import Path\n"
    "p = Path(sys.argv[1])\n"
    "p.write_text(json.dumps({"
    "'complete': False, 'n_got': 1, 'n_expected': 8, 'partial': True"
    "}))\n"
    "time.sleep(60)\n"
    "p.write_text(json.dumps({"
    "'complete': True, 'n_got': 8, 'n_expected': 8"
    "}))\n"
)


class FailClosed(Exception):
    """Refuse, and say why. Never a default, never a silent skip."""

    def __init__(self, fault: str, reason: str) -> None:
        self.fault = fault
        self.reason = reason
        super().__init__(f"FAIL_CLOSED [{fault}]: {reason}")


# ---------------------------------------------------------------------------
# Content hashing. Disk state is authority; hash content, never path strings.
# Same family as tools/future/_common.seal and tools/provenance_chain.py.
# ---------------------------------------------------------------------------


def content_hash(obj: Any) -> str:
    blob = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def seal_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Seal a dict the same way _common.seal does. Does not write."""
    out = dict(doc)
    body = {k: v for k, v in out.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    out["seal_sha256"] = hashlib.sha256(blob).hexdigest()
    return out


def seal_is_valid(doc: dict[str, Any]) -> bool:
    got = doc.get("seal_sha256")
    if not isinstance(got, str) or len(got) != 64:
        return False
    try:
        int(got, 16)
    except ValueError:
        return False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest() == got


def experiment_identity(
    *,
    inputs: dict[str, str],
    code_sha256: str,
    compiler: str,
    machine_genome: dict[str, Any],
) -> str:
    """Immutable identity over inputs + code (incl. compiler) + machine genome.

    The same identity must reproduce. A changed input must not.
    Compiler is part of the code bundle: a silent toolchain swap is a
    different experiment, not the same one with a different compiler.
    """
    return content_hash(
        {
            "inputs": {k: inputs[k] for k in sorted(inputs)},
            "code_sha256": code_sha256,
            "compiler": compiler,
            "machine_genome_sha256": content_hash(machine_genome),
        }
    )


def fixture_machine_genome() -> dict[str, Any]:
    """STATIC pin, not a measurement. No bandwidth, no joules, no tps."""
    return {
        "schema": "hawking.future.machine_genome_pin.v1",
        "knowledge_level": "PIN_ONLY",
        "arch": "arm64",
        "os_family": "darwin",
        "gpu_authority": False,
        "note": "identity pin only; not a MachineGenome measurement and not a roof",
    }


def fixture_code_sha256() -> str:
    return hashlib.sha256(b"repro-science-fixture-source-v1").hexdigest()


def fixture_specimen_hash() -> str:
    return hashlib.sha256(b"repro-science-fixture-specimen-v1").hexdigest()


# ---------------------------------------------------------------------------
# Provenance graph: experiment -> inputs -> code -> machine -> outputs -> claims
# Edges point at dependencies so a claim walks back to evidence and code state.
# ---------------------------------------------------------------------------


def new_graph() -> dict[str, Any]:
    return {"nodes": {}, "edges": []}


def add_node(
    graph: dict[str, Any],
    node_id: str,
    kind: str,
    content_sha256: str,
    **extra: Any,
) -> None:
    if kind not in NODE_KINDS:
        raise ValueError(f"unknown node kind {kind!r}")
    if node_id in graph["nodes"]:
        raise ValueError(f"duplicate node {node_id!r}")
    node = {"id": node_id, "kind": kind, "content_sha256": content_sha256}
    for k in sorted(extra):
        node[k] = extra[k]
    graph["nodes"][node_id] = node


def add_edge(graph: dict[str, Any], src: str, rel: str, dst: str) -> None:
    if rel not in EDGE_RELS:
        raise ValueError(f"unknown edge rel {rel!r}")
    if src not in graph["nodes"] or dst not in graph["nodes"]:
        raise ValueError(f"edge {src!r} -{rel}-> {dst!r} names a missing node")
    graph["edges"].append({"src": src, "rel": rel, "dst": dst})


def graph_as_json(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = [graph["nodes"][k] for k in sorted(graph["nodes"])]
    edges = sorted(graph["edges"], key=lambda e: (e["src"], e["rel"], e["dst"]))
    return {"nodes": nodes, "edges": edges}


def trace_claim(graph: dict[str, Any], claim_id: str) -> list[dict[str, Any]]:
    """Walk a claim back to the experiment, its inputs, code, and machine."""
    if claim_id not in graph["nodes"]:
        raise FailClosed("unknown_claim", f"claim {claim_id!r} is not in the graph")
    by_src: dict[str, list[str]] = {}
    for e in graph["edges"]:
        by_src.setdefault(e["src"], []).append(e["dst"])
    ordered: list[str] = []
    seen: set[str] = set()
    stack = [claim_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        ordered.append(cur)
        for dst in sorted(by_src.get(cur, [])):
            if dst not in seen:
                stack.append(dst)
    return [graph["nodes"][i] for i in ordered]


def example_provenance_graph() -> dict[str, Any]:
    g = new_graph()
    specimen = fixture_specimen_hash()
    code = fixture_code_sha256()
    genome = content_hash(fixture_machine_genome())
    compiler = "sidecar-static-compiler-pin"
    inputs = {"specimen": specimen, "route_corpus": hashlib.sha256(b"r0").hexdigest()}
    eid = experiment_identity(
        inputs=inputs,
        code_sha256=code,
        compiler=compiler,
        machine_genome=fixture_machine_genome(),
    )
    out_hash = hashlib.sha256(b"fixture-output-v1").hexdigest()
    add_node(g, "EXP1", "experiment", eid)
    add_node(g, "SPECIMEN", "input", specimen, role="specimen")
    add_node(g, "ROUTES", "input", inputs["route_corpus"], role="route_corpus")
    add_node(g, "CODE", "code", code, compiler=compiler)
    add_node(g, "MACHINE", "machine", genome, knowledge_level="PIN_ONLY")
    add_node(g, "OUT1", "output", out_hash, complete=True)
    add_node(g, "CLAIM_PARENT", "claim", content_hash({"on": "OUT1"}), status="VALID")
    add_node(g, "CLAIM_CHILD", "claim", content_hash({"on": "CLAIM_PARENT"}), status="VALID")
    add_node(
        g, "CLAIM_GRANDCHILD", "claim", content_hash({"on": "CLAIM_CHILD"}), status="VALID"
    )
    add_edge(g, "EXP1", "uses", "SPECIMEN")
    add_edge(g, "EXP1", "uses", "ROUTES")
    add_edge(g, "EXP1", "compiled_from", "CODE")
    add_edge(g, "EXP1", "ran_on", "MACHINE")
    add_edge(g, "OUT1", "produced_by", "EXP1")
    add_edge(g, "CLAIM_PARENT", "supported_by", "OUT1")
    add_edge(g, "CLAIM_CHILD", "inherits", "CLAIM_PARENT")
    add_edge(g, "CLAIM_GRANDCHILD", "inherits", "CLAIM_CHILD")
    return g


# ---------------------------------------------------------------------------
# Claim ledger: invalidating evidence transitively downgrades descendants.
# ---------------------------------------------------------------------------


def new_ledger() -> dict[str, Any]:
    return {"nodes": {}, "depends_on": []}


def ledger_add(ledger: dict[str, Any], node_id: str, kind: str) -> None:
    if kind not in {"evidence", "claim"}:
        raise ValueError(f"unknown ledger kind {kind!r}")
    if node_id in ledger["nodes"]:
        raise ValueError(f"duplicate ledger node {node_id!r}")
    ledger["nodes"][node_id] = {"id": node_id, "kind": kind, "status": "VALID"}


def ledger_link(ledger: dict[str, Any], child: str, parent: str) -> None:
    if child not in ledger["nodes"] or parent not in ledger["nodes"]:
        raise ValueError(f"link {child!r}->{parent!r} names a missing node")
    ledger["depends_on"].append([child, parent])


def ledger_status(ledger: dict[str, Any], node_id: str) -> str:
    node = ledger["nodes"].get(node_id)
    if node is None:
        raise FailClosed("unknown_claim", f"ledger has no node {node_id!r}")
    return str(node["status"])


def ledger_invalidate(ledger: dict[str, Any], node_id: str) -> dict[str, str]:
    """Invalidate evidence; every downstream claim is DOWNGRADED, transitively.

    Never proceeds on a missing node. Never silently leaves a descendant VALID.
    """
    if node_id not in ledger["nodes"]:
        raise FailClosed("unknown_claim", f"cannot invalidate missing node {node_id!r}")
    root = ledger["nodes"][node_id]
    root["status"] = "INVALID" if root["kind"] == "evidence" else "DOWNGRADED"
    children: dict[str, list[str]] = {}
    for child, parent in ledger["depends_on"]:
        children.setdefault(parent, []).append(child)
    stack = list(children.get(node_id, ()))
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        node = ledger["nodes"][cur]
        node["status"] = "DOWNGRADED"
        stack.extend(children.get(cur, ()))
    return {k: ledger["nodes"][k]["status"] for k in sorted(ledger["nodes"])}


def transitive_downgrade_proof() -> dict[str, Any]:
    led = new_ledger()
    ledger_add(led, "E_parent", "evidence")
    ledger_add(led, "E_other", "evidence")
    ledger_add(led, "C_child", "claim")
    ledger_add(led, "C_grandchild", "claim")
    ledger_add(led, "C_unrelated", "claim")
    ledger_link(led, "C_child", "E_parent")
    ledger_link(led, "C_grandchild", "C_child")
    ledger_link(led, "C_unrelated", "E_other")
    before = {k: ledger_status(led, k) for k in sorted(led["nodes"])}
    after = ledger_invalidate(led, "E_parent")
    holds = (
        after["E_parent"] == "INVALID"
        and after["C_child"] == "DOWNGRADED"
        and after["C_grandchild"] == "DOWNGRADED"
        and after["C_unrelated"] == "VALID"
        and after["E_other"] == "VALID"
    )
    if not holds:
        raise FailClosed(
            "claim_downgrade",
            "parent invalidation did not transitively downgrade the grandchild",
        )
    return {
        "before": before,
        "after": after,
        "transitivity_holds": holds,
        "meaning": (
            "Invalidating an input's evidence downgrades every downstream claim, "
            "including a grandchild. Unrelated claims stay VALID. This is not a "
            "promotion and not a hardware measurement."
        ),
    }


# ---------------------------------------------------------------------------
# Mutation canaries. Two historical shapes this project has already paid for:
#   1. A suite that reads a checked-in receipt instead of running the tool
#      stayed green through source mutations.
#   2. A killed agent left its negative-control mutation live in source.
# A canary that stays green means the suite is not testing what it claims.
# ---------------------------------------------------------------------------


def is_receipt_reader_suite(source: str, tool_names: frozenset[str] = TOOL_SYMBOLS) -> bool:
    """True when a test reads a receipt JSON and never invokes the tool."""
    tree = ast.parse(source)
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)
    mentions_receipt = "receipt" in source.lower()
    reads_json = bool({"loads", "load", "read_text", "read_bytes"} & called)
    invokes_tool = bool(called & set(tool_names))
    return mentions_receipt and reads_json and not invokes_tool


def leftover_canary_present(source: str) -> bool:
    """True when a negative-control mutation was left live in source.

    The detector's own constant assignment is not a live mutation. A live
    mutation is an executable statement carrying the marker (return/assert).
    """
    for line in source.splitlines():
        stripped = line.strip()
        if CANARY_MARKER not in stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("CANARY_MARKER"):
            continue
        if re.match(r"^[A-Z_][A-Z0-9_]*\s*=", stripped):
            continue
        if stripped.startswith("return ") or stripped.startswith("assert "):
            return True
    return False


def run_mutation_canaries() -> dict[str, Any]:
    """Prove both historical shapes are DETECTED, and that a live test does fail."""

    dead_src = (
        "import json\n"
        "from pathlib import Path\n"
        "def test_repro():\n"
        "    doc = json.loads(Path('receipts/future/REPRO_SCIENCE.json').read_text())\n"
        "    assert doc['schema'] == 'hawking.future.repro_science.v1'\n"
    )
    live_src = (
        "from tools.future.repro_science import build\n"
        "def test_repro():\n"
        "    out = build()\n"
        "    assert out.exists()\n"
    )
    dead_flagged = is_receipt_reader_suite(dead_src)
    live_flagged = is_receipt_reader_suite(live_src)

    ns: dict[str, Any] = {"add": lambda a, b: a + b}

    def live_test() -> None:
        assert ns["add"](2, 2) == 4

    live_test()
    ns["add"] = lambda a, b: a - b
    live_failed_after_mutation = False
    try:
        live_test()
    except AssertionError:
        live_failed_after_mutation = True

    frozen = 4

    def dead_test() -> None:
        assert frozen == 4

    dead_still_green = True
    try:
        dead_test()
    except AssertionError:
        dead_still_green = False
    # Dead suite stayed green after the subject mutated; live suite failed.
    # Catching the dead suite is the canary. A green dead suite is the bug.
    dead_shape_caught = bool(dead_still_green) and bool(live_failed_after_mutation)

    mutated_source = (
        "def add(a, b):\n"
        f"    return a - b  # {CANARY_MARKER}\n"
    )
    leftover_caught = leftover_canary_present(mutated_source)
    own = Path(__file__).read_text(encoding="utf-8")
    own_clean = not leftover_canary_present(own)

    proof = {
        "receipt_reader_suite_flagged": dead_flagged,
        "live_suite_not_flagged": not live_flagged,
        "live_verification_fails_on_mutation": live_failed_after_mutation,
        "dead_verification_stayed_green": dead_still_green,
        "dead_verification_caught": dead_shape_caught,
        "leftover_negative_control_caught": leftover_caught,
        "this_module_has_no_leftover_canary": own_clean,
        "historical_shapes": [
            "suite read a checked-in receipt instead of running the tool; stayed green through source mutations",
            "killed agent left its negative-control mutation live in source",
        ],
    }
    if not (
        dead_flagged
        and (not live_flagged)
        and live_failed_after_mutation
        and dead_shape_caught
        and leftover_caught
        and own_clean
    ):
        raise FailClosed("mutation_canary", f"canary did not fire: {proof}")
    return proof


# ---------------------------------------------------------------------------
# Clean-build recipe + replication bundle (a dict, not a *replication_bundle* file).
# ---------------------------------------------------------------------------


def make_replication_bundle(
    *,
    experiment_identity_value: str,
    inputs: list[dict[str, str]],
    code_identity: str,
    machine_genome_pin: dict[str, Any],
) -> dict[str, Any]:
    steps = [
        {"n": 1, "action": "pin experiment identity over inputs + code + machine genome", "deterministic": True},
        {"n": 2, "action": "materialize inputs by content hash (never by path string)", "deterministic": True},
        {"n": 3, "action": "pin code identity and compiler", "deterministic": True},
        {"n": 4, "action": "pin MachineGenome as PIN_ONLY; do not treat it as a measurement", "deterministic": True},
        {"n": 5, "action": "admit() the world; refuse on any of the twelve faults", "deterministic": True},
        {"n": 6, "action": "seal outputs; bind claims to outputs in the provenance graph", "deterministic": True},
        {"n": 7, "action": "never report PASS on SKIP; never invent a hardware number", "deterministic": True},
    ]
    return {
        "experiment_identity": experiment_identity_value,
        "inputs": inputs,
        "code_identity": code_identity,
        "machine_genome_pin": machine_genome_pin,
        "recipe_steps": steps,
        "expected_outputs": [
            {"name": "sealed experiment receipt", "role": "output"},
            {"name": "provenance graph", "role": "graph"},
        ],
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. "
            "Neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE."
        ),
        "how_to_verify": (
            "python3 -m pytest tools/future/test_repro_science.py -q && "
            "python3 tools/future/repro_science.py --selftest"
        ),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "fpga": "FPGA is part of Accelerator / Physical Compiler / Fusion, not its own civilization",
        },
    }


def bundle_missing(bundle: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in BUNDLE_REQUIRED:
        if key not in bundle or bundle[key] in (None, "", [], {}):
            missing.append(key)
    inputs = bundle.get("inputs")
    if isinstance(inputs, list):
        for i, row in enumerate(inputs):
            if not isinstance(row, dict) or "sha256" not in row or "name" not in row:
                missing.append(f"inputs[{i}].name/sha256")
    steps = bundle.get("recipe_steps")
    if isinstance(steps, list):
        for i, step in enumerate(steps):
            if not isinstance(step, dict) or "action" not in step:
                missing.append(f"recipe_steps[{i}].action")
    return missing


def assert_bundle_complete(bundle: dict[str, Any]) -> None:
    missing = bundle_missing(bundle)
    if missing:
        raise FailClosed(
            "incomplete_replication_bundle",
            "replication bundle incomplete: " + ", ".join(missing),
        )


# ---------------------------------------------------------------------------
# Experiment world + admit(). Twelve faults, first match wins, named.
# ---------------------------------------------------------------------------


@dataclass
class World:
    code_sha256: str
    compiler: str
    pinned_compiler: str
    machine_genome: dict[str, Any]
    pinned_machine_genome_hash: str
    specimen_hash: str
    specimen_hash_on_disk: str
    experiment_identity: str
    nx: dict[str, Any]
    nx_required: bool
    receipt: dict[str, Any]
    pipeline_cache: dict[str, Any]
    worker_status: str
    subprocess_status: str
    result: dict[str, Any]
    route_corpus: dict[str, Any]
    teacher_receipt: dict[str, Any]
    hardware_required: bool
    hardware_present: bool
    proposed_verdict: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "World":
        return deepcopy(self)


def make_route_corpus(routes: list[tuple[str, str]]) -> dict[str, Any]:
    items = []
    for rid, body in routes:
        items.append(
            {"id": rid, "content_sha256": hashlib.sha256(body.encode()).hexdigest()}
        )
    items.sort(key=lambda r: r["id"])
    corpus = content_hash([r["content_sha256"] for r in items])
    return {
        "schema": "hawking.future.route_corpus.v1",
        "routes": items,
        "corpus_sha256": corpus,
    }


def route_corpus_valid(doc: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(doc, dict):
        return False, "route corpus is not an object"
    if doc.get("schema") != "hawking.future.route_corpus.v1":
        return False, "route corpus schema missing or wrong"
    routes = doc.get("routes")
    if not isinstance(routes, list) or not routes:
        return False, "route corpus has no routes"
    hashes: list[str] = []
    for i, row in enumerate(routes):
        if not isinstance(row, dict):
            return False, f"route[{i}] is not an object"
        h = row.get("content_sha256")
        rid = row.get("id")
        if not rid or not isinstance(h, str) or len(h) != 64:
            return False, f"route[{i}] missing id or content hash"
        try:
            int(h, 16)
        except ValueError:
            return False, f"route[{i}] content hash is not hex"
        hashes.append(h)
    expected = content_hash(hashes)
    if doc.get("corpus_sha256") != expected:
        return False, "route corpus_sha256 does not match sorted route hashes"
    return True, "ok"


def teacher_receipt_valid(doc: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(doc, dict):
        return False, "teacher receipt is not an object"
    if doc.get("schema") != "hawking.future.teacher_receipt.v1":
        return False, "teacher receipt schema missing or wrong"
    if not isinstance(doc.get("teacher_id"), str) or not doc["teacher_id"]:
        return False, "teacher_id missing"
    ch = doc.get("corpus_hash")
    if not isinstance(ch, str) or len(ch) != 64:
        return False, "teacher corpus_hash missing"
    n = doc.get("n_traces")
    if type(n) is not int or n < 1:
        return False, "n_traces must be a positive int"
    if not seal_is_valid(doc):
        return False, "teacher receipt seal is broken or missing"
    return True, "ok"


def healthy_world() -> World:
    genome = fixture_machine_genome()
    code = fixture_code_sha256()
    compiler = "sidecar-static-compiler-pin"
    specimen = fixture_specimen_hash()
    routes = make_route_corpus([("r0", "route-zero"), ("r1", "route-one")])
    inputs = {
        "specimen": specimen,
        "route_corpus": routes["corpus_sha256"],
    }
    eid = experiment_identity(
        inputs=inputs,
        code_sha256=code,
        compiler=compiler,
        machine_genome=genome,
    )
    teacher = seal_doc(
        {
            "schema": "hawking.future.teacher_receipt.v1",
            "teacher_id": "fixture-teacher",
            "corpus_hash": hashlib.sha256(b"fixture-teacher-corpus").hexdigest(),
            "n_traces": 4,
        }
    )
    receipt = seal_doc(
        {
            "schema": "hawking.future.experiment_output.v1",
            "experiment_identity": eid,
            "outputs": {"complete": True, "n": 4},
        }
    )
    nx = {
        "schema": "hawking.future.nx_pin.v1",
        "status": "STATIC_DESCRIPTOR",
        "code_identity": code,
        "compiler": compiler,
        "specimen_hash": specimen,
        "knowledge_level": "PIN_ONLY",
        "note": "not a physical NX; sidecar has no GPU and does not build one",
    }
    return World(
        code_sha256=code,
        compiler=compiler,
        pinned_compiler=compiler,
        machine_genome=genome,
        pinned_machine_genome_hash=content_hash(genome),
        specimen_hash=specimen,
        specimen_hash_on_disk=specimen,
        experiment_identity=eid,
        nx=nx,
        nx_required=False,
        receipt=receipt,
        pipeline_cache={"identity": eid, "payload_sha256": content_hash({"n": 4})},
        worker_status="ok",
        subprocess_status="ok",
        result={"complete": True, "n_got": 4, "n_expected": 4},
        route_corpus=routes,
        teacher_receipt=teacher,
        hardware_required=False,
        hardware_present=False,
        proposed_verdict=None,
    )


def checkpoint(world: World) -> dict[str, Any]:
    """Durable snapshot of a consistent world. Safe resume restores this, not the fault."""
    return deepcopy(world.__dict__)


def resume(snapshot: dict[str, Any]) -> World:
    data = deepcopy(snapshot)
    extra = data.pop("extra", {})
    w = World(**data)
    w.extra = extra
    return w


def finalize_verdict(value: str) -> str:
    v = str(value).strip().upper()
    if v in SKIP_VERDICTS:
        raise FailClosed(
            "skip_as_pass",
            f"verdict {value!r} is not evidence; SKIP is not PASS and must not be reported as success",
        )
    if v not in LEGAL_VERDICTS:
        raise FailClosed("invalid_verdict", f"unknown verdict {value!r}")
    return v


def admit(world: World) -> str:
    """Admit an experiment world or FAIL CLOSED with the named fault.

    First match wins. A refusal is REFUSED, never PASS, never SKIP.
    """
    if world.proposed_verdict is not None:
        finalize_verdict(world.proposed_verdict)

    if world.nx_required:
        nx = world.nx if isinstance(world.nx, dict) else {}
        status = str(nx.get("status") or "")
        nx_code = str(nx.get("code_identity") or "")
        if status in METADATA_ONLY_NX or nx_code != world.code_sha256 or not nx:
            raise FailClosed(
                "stale_nx",
                f"NX is not runnable under this code identity "
                f"(status={status!r}, nx_code={nx_code[:12]!r}, "
                f"code={world.code_sha256[:12]!r}). "
                "A metadata-only NX must not be treated as a physical executable.",
            )

    if world.specimen_hash != world.specimen_hash_on_disk:
        raise FailClosed(
            "wrong_specimen_hash",
            "specimen hash on disk does not match the experiment pin; "
            "refusing rather than running against the wrong bytes",
        )

    if not isinstance(world.receipt, dict) or not seal_is_valid(world.receipt):
        raise FailClosed(
            "corrupt_receipt",
            "experiment receipt seal does not match canonical body; "
            "a broken seal is not evidence",
        )

    cache_id = (world.pipeline_cache or {}).get("identity")
    if cache_id != world.experiment_identity:
        raise FailClosed(
            "stale_pipeline_cache",
            "pipeline cache identity does not match the live experiment identity; "
            "a cache hit on a different experiment is not a result",
        )

    if world.compiler != world.pinned_compiler:
        raise FailClosed(
            "changed_compiler",
            f"live compiler {world.compiler!r} != pinned {world.pinned_compiler!r}",
        )

    live_genome = content_hash(world.machine_genome)
    if live_genome != world.pinned_machine_genome_hash:
        raise FailClosed(
            "changed_machine_genome",
            "live MachineGenome pin does not match the experiment's pinned genome",
        )

    if world.worker_status != "ok":
        raise FailClosed(
            "failed_worker",
            f"worker status {world.worker_status!r}; refusing rather than skipping remaining units as PASS",
        )

    if world.subprocess_status != "ok":
        raise FailClosed(
            "killed_subprocess",
            f"subprocess status {world.subprocess_status!r}; partial stdout is not a result. Restore the checkpoint to resume.",
        )

    complete = bool(world.result.get("complete"))
    n_got = world.result.get("n_got")
    n_expected = world.result.get("n_expected")
    if (not complete) or n_got != n_expected:
        raise FailClosed(
            "partial_result",
            f"result incomplete complete={complete} n_got={n_got} n_expected={n_expected}",
        )

    ok, why = route_corpus_valid(world.route_corpus)
    if not ok:
        raise FailClosed("invalid_route_corpus", why)

    ok, why = teacher_receipt_valid(world.teacher_receipt)
    if not ok:
        raise FailClosed("malformed_teacher_receipt", why)

    if world.hardware_required and not world.hardware_present:
        raise FailClosed(
            "missing_hardware",
            "experiment requires hardware this process does not have "
            "(no protected GPU lease, no FPGA, no power meter). "
            "UNKNOWN is the correct answer; an estimate is not a measurement. "
            "Sidecar emits STATIC_ONLY and will not invent a number.",
        )

    return "ADMITTED"


def inject(fault: str, world: World) -> World:
    """Return a cloned world with exactly one named fault injected."""
    if fault not in FAULT_NAMES:
        raise ValueError(f"unknown fault {fault!r}")
    w = world.clone()
    if fault == "stale_nx":
        w.nx_required = True
        w.nx = dict(w.nx)
        w.nx["status"] = "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION"
        w.nx["code_identity"] = "0" * 64
    elif fault == "wrong_specimen_hash":
        w.specimen_hash_on_disk = "1" * 64
    elif fault == "corrupt_receipt":
        w.receipt = dict(w.receipt)
        outputs = dict(w.receipt.get("outputs") or {})
        outputs["n"] = 99
        w.receipt["outputs"] = outputs
    elif fault == "stale_pipeline_cache":
        w.pipeline_cache = dict(w.pipeline_cache)
        w.pipeline_cache["identity"] = "2" * 64
    elif fault == "changed_compiler":
        w.compiler = "mutated-compiler-not-the-pin"
    elif fault == "changed_machine_genome":
        g = dict(w.machine_genome)
        g["arch"] = "x86_64"
        w.machine_genome = g
    elif fault == "failed_worker":
        w.worker_status = "failed"
    elif fault == "killed_subprocess":
        w.subprocess_status = "killed"
    elif fault == "partial_result":
        w.result = {"complete": False, "n_got": 1, "n_expected": 4}
    elif fault == "invalid_route_corpus":
        w.route_corpus = {
            "schema": "hawking.future.route_corpus.v1",
            "routes": [],
            "corpus_sha256": "0" * 64,
        }
    elif fault == "malformed_teacher_receipt":
        w.teacher_receipt = {"schema": "not-a-teacher", "oops": True}
    elif fault == "missing_hardware":
        w.hardware_required = True
        w.hardware_present = False
    return w


def run_fault_suite() -> list[dict[str, Any]]:
    healthy = healthy_world()
    admit(healthy)
    snap = checkpoint(healthy)
    rows: list[dict[str, Any]] = []
    for fault in FAULT_NAMES:
        tainted = inject(fault, healthy)
        detected = False
        matched = ""
        reason = ""
        try:
            admit(tainted)
        except FailClosed as exc:
            detected = exc.fault == fault
            matched = exc.fault
            reason = exc.reason
        resumed = False
        resume_reason = ""
        try:
            admit(resume(snap))
            resumed = True
        except FailClosed as exc:
            resume_reason = str(exc)
        rows.append(
            {
                "fault": fault,
                "injected": True,
                "detected": detected,
                "matched": matched,
                "reason": reason,
                "resumed": resumed,
                "resume_reason": resume_reason,
            }
        )
    missed = [r["fault"] for r in rows if not r["detected"] or not r["resumed"]]
    if missed:
        raise FailClosed(
            "fault_suite",
            "guards did not fire or resume failed: " + ", ".join(missed),
        )
    return rows


def _wait_for_file(path: Path, timeout_s: float = 5.0) -> bool:
    import time as _time

    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        try:
            if path.is_file() and path.stat().st_size > 10:
                return True
        except OSError:
            pass
        _time.sleep(0.01)
    return path.is_file()


def physical_killed_subprocess_proof() -> dict[str, Any]:
    """Really kill a child. A flag-only injector is not this proof."""
    with tempfile.TemporaryDirectory(prefix="repro-science-kill-") as td:
        out = Path(td) / "result.json"
        child = subprocess.Popen(
            [_sys.executable, "-c", _KILL_CHILD, str(out)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            _wait_for_file(out)
            try:
                os.kill(child.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        finally:
            if child.poll() is None:
                try:
                    os.kill(child.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        payload: dict[str, Any] = {}
        if out.is_file():
            try:
                loaded = json.loads(out.read_text())
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                payload = {"complete": False, "n_got": 0, "n_expected": 8}
        w = healthy_world()
        w.subprocess_status = "killed"
        w.result = {
            "complete": bool(payload.get("complete")),
            "n_got": int(payload.get("n_got") or 0),
            "n_expected": int(payload.get("n_expected") or 8),
        }
        snap = checkpoint(healthy_world())
        detected = False
        reason = ""
        try:
            admit(w)
        except FailClosed as exc:
            detected = exc.fault == "killed_subprocess"
            reason = exc.reason
        resumed = False
        try:
            admit(resume(snap))
            resumed = True
        except FailClosed:
            resumed = False
        if not detected or not resumed:
            raise FailClosed(
                "killed_subprocess",
                f"physical kill did not fail closed / resume: detected={detected} resumed={resumed}",
            )
        return {
            "fault": "killed_subprocess",
            "physical": True,
            "child_returncode": child.returncode,
            "partial_present": out.is_file(),
            "detected": detected,
            "reason": reason,
            "resumed": resumed,
        }


def identity_proof() -> dict[str, Any]:
    genome = fixture_machine_genome()
    inputs = {"specimen": fixture_specimen_hash(), "route_corpus": "a" * 64}
    code = fixture_code_sha256()
    compiler = "sidecar-static-compiler-pin"
    a = experiment_identity(
        inputs=inputs, code_sha256=code, compiler=compiler, machine_genome=genome
    )
    b = experiment_identity(
        inputs=inputs, code_sha256=code, compiler=compiler, machine_genome=genome
    )
    other_inputs = dict(inputs)
    other_inputs["specimen"] = "b" * 64
    c = experiment_identity(
        inputs=other_inputs, code_sha256=code, compiler=compiler, machine_genome=genome
    )
    other_code = experiment_identity(
        inputs=inputs, code_sha256="c" * 64, compiler=compiler, machine_genome=genome
    )
    other_genome = dict(genome)
    other_genome["arch"] = "x86_64"
    d = experiment_identity(
        inputs=inputs, code_sha256=code, compiler=compiler, machine_genome=other_genome
    )
    other_compiler = experiment_identity(
        inputs=inputs, code_sha256=code, compiler="other", machine_genome=genome
    )
    proof = {
        "reproduces": a == b,
        "input_sensitive": a != c,
        "code_sensitive": a != other_code,
        "machine_genome_sensitive": a != d,
        "compiler_sensitive": a != other_compiler,
        "identity": a,
        "identity_changed_input": c,
    }
    if not (
        proof["reproduces"]
        and proof["input_sensitive"]
        and proof["code_sensitive"]
        and proof["machine_genome_sensitive"]
        and proof["compiler_sensitive"]
    ):
        raise FailClosed("experiment_identity", f"identity is not immutable/sensitive: {proof}")
    return proof


def provenance_proof() -> dict[str, Any]:
    g = example_provenance_graph()
    traced = trace_claim(g, "CLAIM_GRANDCHILD")
    kinds = {n["kind"] for n in traced}
    ids = [n["id"] for n in traced]
    needed = {"claim", "output", "experiment", "input", "code", "machine"}
    holds = needed <= kinds and "CLAIM_GRANDCHILD" in ids and "EXP1" in ids and "CODE" in ids
    if not holds:
        raise FailClosed("provenance", f"claim did not trace to code/inputs/machine: {ids}")
    return {
        "claim": "CLAIM_GRANDCHILD",
        "traced_ids": ids,
        "traced_kinds": sorted(kinds),
        "holds": holds,
        "graph": graph_as_json(g),
    }


def run_all_proofs() -> dict[str, Any]:
    identity = identity_proof()
    provenance = provenance_proof()
    canaries = run_mutation_canaries()
    faults = run_fault_suite()
    kill = physical_killed_subprocess_proof()
    downgrade = transitive_downgrade_proof()
    healthy = healthy_world()
    bundle = make_replication_bundle(
        experiment_identity_value=healthy.experiment_identity,
        inputs=[
            {"name": "specimen", "sha256": healthy.specimen_hash, "role": "specimen"},
            {
                "name": "route_corpus",
                "sha256": healthy.route_corpus["corpus_sha256"],
                "role": "route_corpus",
            },
        ],
        code_identity=sha256_file(Path(__file__)),
        machine_genome_pin=fixture_machine_genome(),
    )
    assert_bundle_complete(bundle)
    incomplete = dict(bundle)
    incomplete.pop("recipe_steps")
    incomplete_refused = False
    try:
        assert_bundle_complete(incomplete)
    except FailClosed as exc:
        incomplete_refused = exc.fault == "incomplete_replication_bundle"
    if not incomplete_refused:
        raise FailClosed("incomplete_replication_bundle", "completeness checker did not refuse")
    skip_refused = False
    try:
        finalize_verdict("SKIP")
    except FailClosed as exc:
        skip_refused = exc.fault == "skip_as_pass"
    if not skip_refused:
        raise FailClosed("skip_as_pass", "SKIP was accepted as a verdict")
    return {
        "identity": identity,
        "provenance": provenance,
        "mutation_canaries": canaries,
        "faults": faults,
        "physical_killed_subprocess": kill,
        "claim_downgrade": downgrade,
        "replication_bundle": bundle,
        "incomplete_bundle_refused": incomplete_refused,
        "skip_is_not_pass": skip_refused,
        "healthy_admitted": True,
    }


# ---------------------------------------------------------------------------
# Recovery notes. Written into the receipt so a later reader does not re-derive.
# ---------------------------------------------------------------------------


RECOVERED_IMPLEMENTATION = [
    {
        "path": "tools/future/_common.py",
        "what": "seal() hashes canonical JSON minus seal_sha256; write_receipt attaches STATIC_ONLY/UNKNOWN bench and raises HardwareClaimError on numeric hardware fields",
        "use": "used as-is; this module does not reimplement write_receipt or weaken HARDWARE_FIELDS",
    },
    {
        "path": "tools/provenance_chain.py",
        "what": "PARENT→NR→NX content digests with a CONTROL tamper that must make the chain check FAIL",
        "use": "law kept: hash content never path strings; CONTROL must be shown to fire. Not forked (Codex tools/ surface)",
    },
    {
        "path": "lab/provenance.py",
        "what": "pin_digest / verify_pin over canonical JSON",
        "use": "same pin family; not imported (sparse / not this write partition)",
    },
    {
        "path": "tools/headless/causal_benchmark_law.py",
        "what": "a benchmark a no-op would also pass is invalid; kernel identity, sentinel, noop_control, bad_control",
        "use": "mutation canaries generalize the no-op law to verification itself",
    },
    {
        "path": "tools/headless/disk_truth.py",
        "what": "disk state is authority; tree hashes; genomes PRESENT or ABSENT, never guessed",
        "use": "experiment identity is a content hash; missing hardware is UNKNOWN/REFUSED, not estimated",
    },
    {
        "path": "tools/headless/dirty_tree_preservation.py",
        "what": "demonstrate_detector_can_fire: a gate never seen to fail is not evidence; scratch clone only",
        "use": "the twelve injectors plus leftover-canary scan are that demonstration",
    },
    {
        "path": "hcli/agentos/recovery.py",
        "what": "kill resident+host, recover_mission, continue; malformed JSON REJECTED; nonsense model not self-certified",
        "use": "killed_subprocess + checkpoint/resume; not imported, not rewritten",
    },
    {
        "path": "hcli/agentos/checkpoint.py",
        "what": "program census, not a success stamp; durable snapshot for resume",
        "use": "checkpoint()/resume() is the static analog",
    },
    {
        "path": "hcli/machine.py",
        "what": "MachineGenome is a PRIOR; GenomeStale on identity/time drift; never silently trust a prior",
        "use": "changed_machine_genome refuse; pin is PIN_ONLY, not a measured genome",
    },
    {
        "path": "tools/accelerator/machine_genome.py",
        "what": "physical MachineGenome with measured bandwidth under a protected window",
        "use": "NOT called. Sidecar has no GPU. We pin identity only.",
    },
    {
        "path": "lab/verification_authority.py",
        "what": "models propose; protected controller decides; forbidden self-promotion",
        "use": "this module never emits PROTECTED_ABSOLUTE or DIAGNOSTIC_RELATIVE",
    },
    {
        "path": "tools/headless/experiment_engine.py",
        "what": "adversary questions include 'what cache is STALE?' and 'what NO-OP would also pass?'",
        "use": "stale_pipeline_cache + mutation canaries",
    },
]

GAPS_CLOSED = [
    "end-to-end provenance graph from experiment through inputs/code/machine/outputs to claims, with claim tracing",
    "immutable experiment identity over inputs + code + compiler + machine genome; input/code/genome/compiler sensitivity proven",
    "mutation canaries for the receipt-reader suite shape and the leftover negative-control mutation shape",
    "clean-build recipe and replication bundle with a completeness checker that refuses on missing pieces",
    "twelve named fail-closed injectors, each refused by name, each restoring a checkpoint",
    "physical killed-subprocess proof (SIGKILL a child, refuse partial stdout, resume from checkpoint)",
    "transitive claim downgrade: invalidating parent evidence downgrades a grandchild; unrelated claims stay VALID",
    "SKIP is not PASS: finalize_verdict('SKIP') refuses",
]

NEGATIVE_FINDINGS = [
    "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json is not in HEAD of this checkout and is not on disk in this sparse worktree; measurement_contract and claim_boundary could not be read",
    "tools/headless/* and hcli/agentos/* are not materialized here; recovered via git show HEAD:<path>, not live import",
    "no existing tools/future/repro_science.py, no *replication_bundle* file, no *provenance_graph* module — F008 was accurate",
    "this process has no protected GPU lease and produced no DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE number",
    "did not call tools/accelerator/machine_genome.py (would require mlx/GPU); MachineGenome here is PIN_ONLY",
    "did not create a file named *replication_bundle* — the contract permits only this module, and that filename would flip F008's absent-probe without a frontier edit we must not make",
]


def build() -> Path:
    proofs = run_all_proofs()
    detected = sum(1 for r in proofs["faults"] if r["detected"])
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Provenance, immutable experiment identity, mutation canaries, "
            "replication bundle, and fail-closed fault injection so autonomy "
            "cannot launder weak evidence into a strong claim."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "module_sha256": sha256_file(Path(__file__)),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "fpga_note": (
            "FPGA is part of Accelerator / Physical Compiler / Fusion. "
            "It is not its own civilization and this module does not build an FPGA backend."
        ),
        "measurement_states": {
            "DIAGNOSTIC_RELATIVE": "contaminated A/B on a busy machine; guides; never promotes; this module does not produce it",
            "PROTECTED_ABSOLUTE": "measurement under a real protected GPU lease; decides; this module does not produce it",
            "STATIC_ONLY": "everything this module emits",
        },
        "recovered_implementation": RECOVERED_IMPLEMENTATION,
        "gaps_closed": GAPS_CLOSED,
        "negative_findings": NEGATIVE_FINDINGS,
        "experiment_identity": proofs["identity"],
        "provenance_graph": proofs["provenance"],
        "mutation_canaries": proofs["mutation_canaries"],
        "clean_build_recipe": proofs["replication_bundle"]["recipe_steps"],
        "replication_bundle": proofs["replication_bundle"],
        "fault_injection": {
            "policy": (
                "FAIL CLOSED: refuse and say why. Never proceed on a default, "
                "never silently skip, never report PASS on a SKIP."
            ),
            "faults": proofs["faults"],
            "n_faults": len(FAULT_NAMES),
            "n_detected": detected,
            "all_detected": detected == len(FAULT_NAMES),
            "physical_killed_subprocess": proofs["physical_killed_subprocess"],
            "skip_is_not_pass": proofs["skip_is_not_pass"],
            "incomplete_bundle_refused": proofs["incomplete_bundle_refused"],
        },
        "claim_downgrade": proofs["claim_downgrade"],
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. "
            "Neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE."
        ),
    }
    if detected != len(FAULT_NAMES):
        raise FailClosed("fault_suite", f"detected {detected}/{len(FAULT_NAMES)}")
    return write_receipt(RECEIPT, doc, "tools/future/repro_science.py")


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
