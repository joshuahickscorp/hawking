#!/usr/bin/env python3
"""A WorkUnit whose acceptance rests on VMCP-produced evidence.

The boundary this proves, and the one it must not cross:

    AgentOS WorkUnit -> VMCP capability -> evidence -> deterministic verifier
    -> AgentOS decision

VMCP owns sensory state. AgentOS owns mission state and the mutation decision.
A recon pass established that `visionmcp` already contains a scheduler, a queue,
retry and promotion logic in its LABORATORY profile -- that profile is a second
AgentOS and is deliberately not used here. Only `profile="core"` and the capture
bus are touched, and nothing in this file lets VMCP decide whether work is done.

The capability used is the generic file eye: observe a real file, content-address
it, and let a verifier ask whether the recorded evidence still supports a claim
about that file. That claim can fail, which is the whole point -- the negative
control mutates the file after capture and the same verifier must then refuse.

    python3 hcli/agentos/vmcp/hcli_integration.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "visionmcp" / "src"))

RESULTS: List[Dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


class FileEye:
    """A deterministic sensor over a file on disk.

    Implements `visionmcp.perception.bus.SensorAdapter`. Deliberately generic and
    target-independent: it records what a file IS -- size, digest, mode, mtime --
    and never interprets it. Everything about the capture identity is derived
    from the request, so observing the same unchanged file twice reuses the
    existing capture rather than minting a second one.
    """

    name = "file.eye"
    version = "1"

    def normalize_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(str(target["path"])).resolve()
        # The id is the path, NOT the content: a capture is of a location at a
        # moment. Content identity lives in the artifact digest, which is what
        # makes staleness detectable at all.
        return {"id": f"file:{path}", "path": str(path)}

    def normalize_config(
        self, target: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"hash": str(config.get("hash", "sha256")), "max_bytes": int(config.get("max_bytes", 8_000_000))}

    def environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {"adapter": self.name, "adapter_version": self.version, "hash": config["hash"]}

    def capture(self, target: Dict[str, Any], config: Dict[str, Any], sink: Any) -> Any:
        from visionmcp.perception.bus import CaptureOutcome

        path = Path(target["path"])
        limitations: List[str] = []
        if not path.is_file():
            # Absent is a CLASSIFICATION, never an empty success. A caller must
            # be able to tell "file is not there" from "file is there and empty".
            return CaptureOutcome(
                summary={"present": False, "path": str(path)},
                limitations=["TARGET_ABSENT"],
            )
        data = path.read_bytes()
        if len(data) > config["max_bytes"]:
            data = data[: config["max_bytes"]]
            limitations.append("TRUNCATED_TO_MAX_BYTES")
        digest = hashlib.sha256(data).hexdigest()
        stat = path.stat()
        sink("bytes", data, "application/octet-stream", {"sha256": digest})
        return CaptureOutcome(
            summary={
                "present": True,
                "path": str(path),
                "size": stat.st_size,
                "mode": oct(stat.st_mode),
                "sha256": digest,
            },
            limitations=limitations,
        )


def _project(project_root: Path):
    """Open the project, creating it (and its schema) the first time.

    `ProjectStore(root)` is the raw constructor and does NOT initialise the
    schema -- `create()` does. Using the constructor gets you a store whose
    first query dies on `no such table: observation_captures`.
    """
    from visionmcp.projects.store import ProjectStore

    if (project_root / "project.db").is_file():
        return ProjectStore.open(project_root)
    project_root.mkdir(parents=True, exist_ok=True)
    return ProjectStore.create(project_root, name="hcli-vmcp-integration")


def observe_file(project_root: Path, path: Path) -> Dict[str, Any]:
    from visionmcp.perception.bus import AdapterRegistry, CaptureBus

    project = _project(project_root)
    registry = AdapterRegistry()
    registry.register(FileEye())
    bus = CaptureBus(project, registry)
    return bus.observe(
        "file.eye",
        {"path": str(path)},
        {},
        rights_decision="local-file-owned-by-operator",
    )


def verify_capture(project_root: Path, capture_id: str) -> Dict[str, Any]:
    from visionmcp.perception.bus import AdapterRegistry, CaptureBus

    bus = CaptureBus(_project(project_root), AdapterRegistry())
    return bus.verify(capture_id)


# The verifier a WorkUnit runs. It is a real command with a real exit code, and
# it consults VMCP evidence rather than re-reading the file itself.
_VERIFIER = r'''
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])   # visionmcp/src
from visionmcp.perception.bus import AdapterRegistry, CaptureBus
from visionmcp.projects.store import ProjectStore
from pathlib import Path

project_root, capture_id, claimed_path = Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
bus = CaptureBus(ProjectStore.open(project_root), AdapterRegistry())

record = bus.get(capture_id)
if not record or record.get("status") != "COMPLETE":
    print("NO_EVIDENCE: capture missing or incomplete"); raise SystemExit(2)

verified = bus.verify(capture_id)
if not verified.get("valid"):
    print(f"EVIDENCE_INVALID: {verified}"); raise SystemExit(3)

summary = record.get("summary") or {}
if isinstance(summary, str):
    summary = json.loads(summary)

# BIND THE CAPTURE TO THE SUBJECT. Without this a capture of file A
# supports a claim about file B -- a forgery canary demonstrated exactly
# that replay against this seam and it came back SUPPORTED, with
# subject_matches_recorded_target false. Content identity is not subject
# identity: two files with the same bytes are still two files, and a
# verifier that only compares digests will accept evidence gathered about
# something else entirely.
recorded_path = summary.get("path")
if not recorded_path:
    print("NO_SUBJECT: capture does not record what it observed")
    raise SystemExit(5)
if Path(recorded_path).resolve() != claimed_path.resolve():
    print(f"SUBJECT_MISMATCH: evidence is about {recorded_path}, "
          f"claim is about {claimed_path}")
    raise SystemExit(6)

observed = summary.get("sha256")
actual = hashlib.sha256(claimed_path.read_bytes()).hexdigest()
if observed != actual:
    print(f"EVIDENCE_STALE: observed {observed[:16]} but the file is now {actual[:16]}")
    raise SystemExit(4)
print(f"SUPPORTED: {observed[:16]} still describes {claimed_path.name}")
'''


def main() -> int:
    from hcli.mission import Mission
    from hcli.workunit import WorkUnit

    with tempfile.TemporaryDirectory(prefix="vmcp-integration-") as tmp:
        root = Path(tmp)
        project_root = root / "project"
        subject = root / "subject.txt"
        subject.write_text("the claim under observation\n", encoding="utf-8")

        # ---- VMCP capability: observe -----------------------------------
        record = observe_file(project_root, subject)
        capture_id = record.get("capture_id")
        check(
            "VMCP capture completed and is content-addressed",
            record.get("status") == "COMPLETE" and bool(capture_id),
            f"capture_id={str(capture_id)[:16]} status={record.get('status')} "
            f"authority={record.get('authority')}",
        )
        verified = verify_capture(project_root, capture_id)
        check(
            "VMCP verifies its own capture against the artifact store",
            bool(verified.get("valid")),
            f"valid={verified.get('valid')}",
        )
        check(
            "capture is reused rather than duplicated for an unchanged file",
            bool(observe_file(project_root, subject).get("reused")),
            "second observe of the same unchanged file returned reused=True",
        )

        # Write the verifier to a file rather than inlining it with -c. The
        # executor runs verifiers through `shell=True`, and a JSON-quoted Python
        # source string does not survive shell word-splitting -- the command
        # failed there while running fine standalone, which is the confusing
        # kind of failure to design out rather than debug twice.
        verifier_py = root / "vmcp_verifier.py"
        verifier_py.write_text(_VERIFIER, encoding="utf-8")
        verifier_cmd = (
            f'{sys.executable} {verifier_py} '
            f'{REPO_ROOT / "visionmcp/src"} '
            f'{project_root} {capture_id} {subject}'
        )

        class NullEngine:
            active = False

            def execute_workunit(self, wu: Any, context: Any) -> Dict[str, Any]:
                return {"kind": "answer", "validation": {"ok": False, "reason": "CPU_ONLY"}}

        def run_once(ws: Path) -> Dict[str, str]:
            unit = WorkUnit(
                id="claim.file_unchanged",
                role="validate",
                description="the observed file still matches the evidence VMCP recorded",
                preferred_backend="cpu",
                resource_class="TEST",
                verifier=verifier_cmd,
            )
            mission = Mission(
                str(ws),
                engine=NullEngine(),
                units={unit.id: unit},
                goal="accept only what VMCP evidence supports",
                runtime_count=1,
                quiet=True,
                no_progress_threshold=4,
            )
            mission.run()
            return {u.id: u.status for u in mission.scheduler.units.values()}

        # ---- positive: evidence supports the claim ----------------------
        positive = run_once(root / "ws-pos")
        check(
            "a WorkUnit is ACCEPTED on VMCP evidence",
            positive.get("claim.file_unchanged") == "completed",
            f"{positive}",
        )

        # ---- negative control: mutate the subject after capture ---------
        subject.write_text("the claim has been tampered with\n", encoding="utf-8")
        negative = run_once(root / "ws-neg")
        check(
            "NEGATIVE CONTROL: the same WorkUnit is REFUSED once the evidence is stale",
            negative.get("claim.file_unchanged") != "completed",
            f"{negative}",
        )

        # ---- replay control ---------------------------------------------
        # A forgery canary against this same seam found replay UNDETECTED:
        # a capture of one file supported a claim about another with the same
        # bytes. Identical content is the HARD case -- a digest comparison
        # cannot tell the two apart, so only subject binding can.
        subject.write_text("the claim under observation\n", encoding="utf-8")
        twin = root / "twin.txt"
        twin.write_text("the claim under observation\n", encoding="utf-8")
        twin_record = observe_file(project_root, twin)
        replay_cmd = (
            f'{sys.executable} {verifier_py} '
            f'{REPO_ROOT / "visionmcp/src"} '
            f'{project_root} {twin_record.get("capture_id")} {subject}'
        )
        replay = subprocess.run(
            replay_cmd, shell=True, capture_output=True, text=True, timeout=300
        )
        check(
            "REPLAY CONTROL: evidence about one file cannot support a claim "
            "about another with identical bytes",
            replay.returncode != 0 and "SUBJECT_MISMATCH" in (replay.stdout or ""),
            f"exit={replay.returncode} {(replay.stdout or '').strip()[:120]}",
        )

        # ---- the boundary ------------------------------------------------
        import visionmcp.mcp.factory as factory

        server = factory.create_server(profile="core")
        # Ask the ToolManager for the real tool names. An earlier version read a
        # `_tools` attribute that does not exist and "passed" on an empty list,
        # and the version after that listed instance attributes -- both proved
        # nothing. `list_tools()` is the actual surface.
        tools: List[str] = []
        manager = getattr(server, "_tool_manager", None)
        lister = getattr(manager, "list_tools", None)
        if callable(lister):
            tools = sorted(str(getattr(t, "name", t)) for t in lister())
        queue_like = [
            t for t in tools
            if any(w in t for w in ("enqueue", "workflow", "schedule", "promote", "dispatch"))
        ]
        check(
            "the core VMCP profile exposes no scheduler, queue or promotion",
            bool(tools) and not queue_like,
            f"{len(tools)} core tools, none of them a queue: {tools}"
            if tools
            else "INCONCLUSIVE: could not enumerate the core server's tools, so this proves nothing",
        )
        check(
            "the acquire capability the integration depends on is actually exposed",
            "vision.observe" in tools and "vision.verify" in tools,
            f"vision.observe and vision.verify present among {len(tools)} tools",
        )

    failed = [r for r in RESULTS if not r["ok"]]
    out = REPO_ROOT / "receipts/headless/VMCP_AGENTOS_INTEGRATION.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gate": "VMCP_AGENTOS_INTEGRATION",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_head": subprocess.check_output(
                    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
                "seam": {
                    "acquire": "visionmcp.perception.bus.CaptureBus.observe",
                    "verify": "CaptureBus.verify + the capture's recorded digest",
                    "profile": "core -- the laboratory profile owns a queue and is a second AgentOS",
                },
                "boundary_held": "VMCP produced evidence; the deterministic verifier read it; AgentOS "
                "made the acceptance decision. VMCP holds no mission state and decided nothing.",
                "results": RESULTS,
                "failed": [r["name"] for r in failed],
                "result": "PASS" if not failed else "FAIL",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    print(f"receipt: {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
