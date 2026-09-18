"""Hawking's compact local perception acts.

The public perception surface is one Hawking organ, not a separate runtime.
It provides real local file observation, terminal capture where the host permits
it, tool diagnosis, and a behavior proof matrix.  Acts that do not yet have a
local implementation remain explicitly PARKED with a concrete Hawking-owned
reopening condition.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


RECEIPT = "PERCEPTION_ACTS.json"
SCHEMA = "hawking.perception.acts.v1"
VERSION = 1
RECORDED_BY = "hawking/perception/acts.py"
DISPOSITION_SCHEMA = "hawking.perception.disposition.v1"
WAKE_SCHEMA = "hawking.audit.wake_condition.v1"
WAKE_REQUIRED_KIND = "call"

NINE_ACTS: tuple[str, ...] = (
    "see",
    "hold",
    "open",
    "know",
    "make",
    "check",
    "fix",
    "keep",
    "prove",
)
CONNECTED_ACTS: frozenset[str] = frozenset({"see", "hold", "know", "check", "prove"})
PARKED_ACTS: frozenset[str] = frozenset(set(NINE_ACTS) - set(CONNECTED_ACTS))

CLAIM_BOUNDARY = (
    "Local-file classification, local tool receipts, and isolated behavior fixtures "
    "on this host. Browser, 3D, repair, and artifact-creation acts remain PARKED "
    "until Hawking implements and verifies their local owners."
)


def _file_observe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .file_eye import observe

    return observe(*args, **kwargs)


def _pty_capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .terminal import capture

    return capture(*args, **kwargs)


def _pty_probe() -> dict[str, Any]:
    from .terminal import probe

    return probe()


def _doctor_profile(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .doctor import profile

    return profile(*args, **kwargs)


def _doctor_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .doctor import report

    return report(*args, **kwargs)


def _behavior_matrix(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from hawking.behavior import run_matrix

    return run_matrix(*args, **kwargs)


def _wake(
    *,
    kind: str,
    required_symbol: str,
    predicate: str,
    blocker: str,
    missing_dependency: str,
) -> dict[str, Any]:
    return {
        "schema": WAKE_SCHEMA,
        "kind": kind,
        "required_kind": WAKE_REQUIRED_KIND,
        "required_symbol": required_symbol,
        "required_caller_prefix": "hawking/",
        "predicate": predicate,
        "blocker": blocker,
        "missing_dependency": missing_dependency,
        "evidence_tier": "STATIC",
    }


def _parked_act_wake(act: str) -> dict[str, Any]:
    table = {
        "open": (
            "PERCEPTION_OPEN_UNIMPLEMENTED",
            "hawking.perception.open_asset",
            "a tested Hawking implementation opens a declared local encoded or binary target",
            "Hawking has no local opener for this act yet",
            "a verified Hawking perception opener",
        ),
        "make": (
            "PERCEPTION_MAKE_UNIMPLEMENTED",
            "hawking.perception.create_artifact",
            "a tested Hawking implementation emits a declared local artifact with a receipt",
            "Hawking has no local artifact-generation act yet",
            "a verified Hawking perception artifact creator",
        ),
        "fix": (
            "PERCEPTION_FIX_UNIMPLEMENTED",
            "hawking.perception.repair_asset",
            "a tested Hawking repair act consumes a declared residual and produces a verified result",
            "Hawking has no local perception repair loop yet",
            "a verified Hawking perception repair owner",
        ),
        "keep": (
            "PERCEPTION_KEEP_UNIMPLEMENTED",
            "hawking.perception.store.ProjectStore.record_observation",
            "a tested Hawking act persists a declared perception result through ProjectStore",
            "the generic store exists, but no compact KEEP act owns its contract yet",
            "a verified Hawking KEEP contract",
        ),
    }
    kind, symbol, predicate, blocker, dependency = table[act]
    return _wake(
        kind=kind,
        required_symbol=symbol,
        predicate=predicate,
        blocker=blocker,
        missing_dependency=dependency,
    )


def _organ_table() -> list[dict[str, Any]]:
    pty = _pty_probe()
    pty_wake = None
    if not pty.get("ok"):
        pty_wake = _wake(
            kind="PTY_OPEN_DENIED",
            required_symbol="hawking.perception.terminal.capture",
            predicate="a Hawking terminal capture runs after openpty succeeds; a pipe is not a PTY",
            blocker=str(pty.get("blocker") or "openpty unavailable"),
            missing_dependency="a process able to allocate a real PTY",
        )
    return [
        {
            "id": "perception.file_eye",
            "disposition": "CONNECTED",
            "execution": "REAL",
            "symbol": "hawking.perception.file_eye.observe",
            "description": "magic/header classification and content identity for a local file",
            "wake": None,
        },
        {
            "id": "perception.terminal",
            "disposition": "CONNECTED" if pty.get("ok") else "PARKED",
            "execution": "REAL" if pty.get("ok") else "BLOCKED",
            "symbol": "hawking.perception.terminal.capture",
            "description": "real PTY capture only; a blocked host never becomes a simulated terminal",
            "wake": pty_wake,
        },
        {
            "id": "perception.behavior",
            "disposition": "CONNECTED",
            "execution": "REAL",
            "symbol": "hawking.behavior.run_matrix",
            "description": "BHV-01..23 isolated behavior matrix with a five-axis local scorer",
            "wake": None,
        },
        {
            "id": "perception.doctor",
            "disposition": "CONNECTED",
            "execution": "REAL",
            "symbol": "hawking.perception.doctor.profile",
            "description": "bounded local tool diagnosis with network and dangerous-command refusal",
            "wake": None,
        },
        *[
            {
                "id": f"perception.{act}",
                "disposition": "PARKED",
                "execution": "UNIMPLEMENTED",
                "symbol": None,
                "description": f"Hawking {act} act has no verified local owner yet",
                "wake": _parked_act_wake(act),
            }
            for act in sorted(PARKED_ACTS)
        ],
    ]


def disposition() -> dict[str, Any]:
    """Describe the live Hawking perception surface without false affordances."""
    acts: list[dict[str, Any]] = []
    for act in NINE_ACTS:
        connected = act in CONNECTED_ACTS
        acts.append(
            {
                "act": act,
                "disposition": "CONNECTED" if connected else "PARKED",
                "symbol": f"hawking.perception.{act}" if connected else None,
                "evidence_tier": "FUNCTIONAL_SIM" if connected else "STATIC",
                "wake": None if connected else _parked_act_wake(act),
                "empty_success": False,
                "looked": connected,
            }
        )
    return {
        "schema": DISPOSITION_SCHEMA,
        "subsystem": "hawking.perception",
        "acts": acts,
        "organs": _organ_table(),
        "claim_boundary": CLAIM_BOUNDARY,
        "empty_success_rule": "A PARKED act returns a named wake, never an empty success.",
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
    }


def see(
    path: str | os.PathLike[str] | None = None,
    *,
    max_bytes: int = 8_000_000,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Observe a local file or, with ``organ=pty``, a real terminal session."""
    args = dict(arguments or {})
    if path is not None and "path" not in args:
        args["path"] = str(path)
    args["max_bytes"] = int(args.get("max_bytes") or max_bytes)
    if str(args.get("organ") or "").strip().lower() in {"pty", "terminal"}:
        return _pty_capture(arguments=args)
    return _file_observe(path, max_bytes=int(args["max_bytes"]), arguments=args)


def hold(
    path: str | os.PathLike[str] | None = None,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Observe and bind local content identity."""
    observed = see(path, arguments=arguments)
    observed["act"] = "hold"
    observed["asset_id"] = f"sha256:{observed['sha256']}" if observed.get("present") else None
    observed["bound"] = bool(observed.get("present"))
    return observed


def know(
    path: str | os.PathLike[str] | None = None,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare content identity without conflating it with path identity."""
    args = dict(arguments or {})
    observed = see(path, arguments=args)
    observed["act"] = "know"
    observed["identity_kind"] = "content_sha256"
    other = args.get("other_path")
    if other:
        twin = see(other, arguments={"path": other, "max_bytes": args.get("max_bytes")})
        observed["other"] = {key: twin.get(key) for key in ("path", "sha256", "present")}
        observed["same_bytes"] = bool(
            observed.get("present")
            and twin.get("present")
            and observed.get("sha256") == twin.get("sha256")
        )
        observed["same_subject"] = bool(
            observed.get("path")
            and twin.get("path")
            and Path(str(observed["path"])).resolve() == Path(str(twin["path"])).resolve()
        )
    observed["subject_identity"] = "path"
    observed["content_identity"] = "sha256"
    return observed


def check(
    path: str | os.PathLike[str] | None = None,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded local tool check or verify an explicit file digest."""
    args = dict(arguments or {})
    organ = str(args.get("organ") or "").strip().lower()
    if organ in {"tool_doctor", "doctor"} or args.get("argv") or args.get("command"):
        return _doctor_report(arguments=args) if args.get("report") else _doctor_profile(arguments=args)
    observed = see(path, arguments=args)
    observed["act"] = "check"
    claimed = args.get("expected_sha256") or args.get("sha256")
    if not observed.get("present"):
        observed.update(ok=False, reason="target not present")
    elif not claimed:
        observed.update(ok=False, reason="expected_sha256 is required; a missing claim is not a pass")
    else:
        observed.update(
            ok=str(claimed) == str(observed.get("sha256")),
            expected_sha256=str(claimed),
            reason="match" if str(claimed) == str(observed.get("sha256")) else "EVIDENCE_STALE",
        )
    return observed


def prove(
    path: str | os.PathLike[str] | None = None,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove a behavior matrix or a file canary with a real RED then GREEN."""
    args = dict(arguments or {})
    organ = str(args.get("organ") or "").strip().lower()
    if organ in {"behavior", "behavior_lab", "bhv"} or args.get("fixtures") is not None:
        return _behavior_matrix(arguments=args)
    payload = b"hawking-perception-canary\n"
    source = path if path is not None else args.get("path")
    if source and Path(str(source)).is_file():
        payload = Path(str(source)).read_bytes()
    with tempfile.TemporaryDirectory(prefix="hawking-perception-") as tmp:
        subject = Path(tmp) / "subject.bin"
        subject.write_bytes(payload)
        baseline = see(subject)
        subject.write_bytes(payload + b"\x00")
        mutated = see(subject)
        subject.write_bytes(payload)
        restored = see(subject)
    red = bool(baseline.get("sha256") and baseline.get("sha256") != mutated.get("sha256"))
    green = bool(red and restored.get("sha256") == baseline.get("sha256"))
    return {
        "act": "prove",
        "status": "CONNECTED",
        "ok": bool(red and green),
        "red": red,
        "green": green,
        "baseline_sha256": baseline.get("sha256"),
        "mutated_sha256": mutated.get("sha256"),
        "restored_sha256": restored.get("sha256"),
        "empty_success": False,
        "looked": True,
        "evidence_tier": "FUNCTIONAL_SIM",
    }


def _parked_response(act: str) -> dict[str, Any]:
    wake = _parked_act_wake(act)
    return {
        "act": act,
        "status": "PARKED",
        "ok": False,
        "looked": False,
        "empty_success": False,
        "results": None,
        "items": None,
        "wake": wake,
        "wake_condition": wake["predicate"],
        "missing_dependency": wake["missing_dependency"],
        "evidence_tier": "STATIC",
    }


def compact_surface(act: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Route one compact Hawking perception act to its local owner."""
    name = str(act or "").strip().lower()
    args = arguments or {}
    if name in {"", "disposition", "status"}:
        return {"act": "disposition", "status": "CONNECTED", "ok": True, "result": disposition()}
    handlers = {"see": see, "hold": hold, "know": know, "check": check, "prove": prove}
    if name in handlers:
        return handlers[name](arguments=args)
    if name in PARKED_ACTS:
        return _parked_response(name)
    return {
        "act": name or None,
        "status": "UNKNOWN_ACT",
        "ok": False,
        "looked": False,
        "empty_success": False,
        "known_acts": [*NINE_ACTS, "disposition"],
        "evidence_tier": "STATIC",
    }


def selftest() -> dict[str, Any]:
    """Exercise the connected organs and prove parked acts cannot look successful."""
    with tempfile.TemporaryDirectory(prefix="hawking-perception-selftest-") as tmp:
        path = Path(tmp) / "sample.txt"
        path.write_text("hawking perception\n", encoding="utf-8")
        seen = see(path)
        held = hold(path)
        checked = check(path, arguments={"expected_sha256": seen.get("sha256")})
        stale = check(path, arguments={"expected_sha256": "0" * 64})
        proof = prove(path)
    if not (seen.get("present") and held.get("bound") and checked.get("ok")):
        raise AssertionError("local file acts did not produce a bound, verified observation")
    if stale.get("ok") or not proof.get("ok"):
        raise AssertionError("digest or RED/GREEN proof control did not bind")
    for act in PARKED_ACTS:
        parked = compact_surface(act)
        if parked.get("status") != "PARKED" or parked.get("empty_success"):
            raise AssertionError(f"{act} did not honestly remain parked")
    matrix = _behavior_matrix()
    if matrix.get("n") != 23 or matrix.get("n_ok") != 23:
        raise AssertionError(f"behavior matrix did not complete: {matrix.get('residuals')}")
    return {
        "ok": True,
        "file_kind": seen.get("kind"),
        "behavior_fixtures": matrix.get("n_ok"),
        "behavior_verdict": (matrix.get("verdict") or {}).get("outcome"),
        "terminal": _pty_probe(),
    }


def build() -> Path:
    """Write an explicit local perception receipt when an operator requests it."""
    from tools.future._common import write_receipt

    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "BUILT_NOT_PROMOTED",
        "promoted": False,
        "disposition": disposition(),
        "selftest": selftest(),
        "claim_boundary": CLAIM_BOUNDARY,
        "resident_callable": {
            "entry_point": "python3 -m hawking.perception.acts --build",
            "symbol": "hawking.perception.compact_surface",
        },
        "gpu_authority": False,
        "weights_modified": False,
    }
    return write_receipt(RECEIPT, payload, RECORDED_BY)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--disposition", action="store_true")
    parser.add_argument("--act")
    parser.add_argument("--path")
    parser.add_argument("--organ")
    parser.add_argument("--args", default="", help="JSON object merged into the act arguments")
    args = parser.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    if args.act:
        payload = json.loads(args.args) if args.args else {}
        if not isinstance(payload, dict):
            raise ValueError("--args must decode to a JSON object")
        if args.path:
            payload["path"] = args.path
        if args.organ:
            payload["organ"] = args.organ
        print(json.dumps(compact_surface(args.act, payload), indent=2, sort_keys=True))
        return 0
    if args.disposition:
        print(json.dumps(disposition(), indent=2, sort_keys=True))
        return 0
    if args.build:
        print(build())
        return 0
    parser.print_help()
    return 2


__all__ = [
    "CLAIM_BOUNDARY",
    "CONNECTED_ACTS",
    "DISPOSITION_SCHEMA",
    "NINE_ACTS",
    "PARKED_ACTS",
    "RECEIPT",
    "SCHEMA",
    "build",
    "check",
    "compact_surface",
    "disposition",
    "hold",
    "know",
    "main",
    "prove",
    "see",
    "selftest",
]


if __name__ == "__main__":
    raise SystemExit(main())
