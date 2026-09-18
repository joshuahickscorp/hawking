"""Deterministic coverage for Hawking's machine-authority boundary."""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from hawking.goal import infer_required_capabilities
from hawking.native_owner import (
    NativeHelperReleaseStore,
    NativeOwnerError,
    inspect_code_identity,
)
from hawking.permissions import (
    PermissionRequired,
    PermissionsService,
    READY,
    UNAVAILABLE,
    normalize_capabilities,
)
from hawking.serve import HAWKING_WEB_ENDPOINTS, make_handler
from hawking.web_launcher import REQUIRED_ENDPOINTS
from hawking.workunit import WorkUnit


class _NativeFixture:
    def __init__(self, state: str = READY) -> None:
        self.state = state
        self.calls: list[tuple[str, dict]] = []

    def call(self, operation: str, payload: dict) -> dict:
        self.calls.append((operation, payload))
        return {
            "schema": "hawking.permissions.v1",
            "helper_version": "fixture",
            "capabilities": {
                str(name): {
                    "state": self.state,
                    "kind": "tcc",
                    "owner": "fixture-only",
                    "evidence": {"probe": "deterministic fixture"},
                }
                for name in payload.get("capabilities", [])
            },
        }


def test_capability_names_are_normalized_without_losing_targets() -> None:
    assert normalize_capabilities([" accessibility ", "AUTOMATION:Finder", "accessibility"]) == [
        "ACCESSIBILITY",
        "AUTOMATION:Finder",
    ]
    required = infer_required_capabilities(
        "Use AXPress on the Finder window and write the result to the external drive.",
        explicit=["automation:Finder"],
    )
    assert "ACCESSIBILITY" in required
    assert "AUTOMATION:Finder" in required
    assert "EXTERNAL_DRIVE_READ" in required
    assert "EXTERNAL_DRIVE_WRITE" in required


def test_injected_native_runtime_is_explicit_and_persisted_as_observation(tmp_path: Path) -> None:
    runtime = _NativeFixture()
    service = PermissionsService(tmp_path, runtime=runtime)
    snapshot = service.status(["ACCESSIBILITY", "SCREEN_CAPTURE"], refresh=True)

    assert snapshot["capabilities"]["ACCESSIBILITY"]["state"] == READY
    assert snapshot["capabilities"]["SCREEN_CAPTURE"]["state"] == READY
    assert snapshot["code_identity"]["state"] == "EXPLICIT_RUNTIME"
    assert snapshot["code_identity"]["owner_mode"] == "injected_runtime"
    assert runtime.calls and runtime.calls[0][0] == "permissions.status"
    assert service.cache_path.is_file()
    assert "OPENROUTER_API_KEY" not in service.cache_path.read_text(encoding="utf-8")


def test_machine_ready_keeps_core_gate_scoped_but_reports_specialist_state(tmp_path: Path) -> None:
    report = PermissionsService(tmp_path, runtime=_NativeFixture()).machine_ready()
    assert report["required"] == ["WORKSPACE_READ", "WORKSPACE_WRITE", "NETWORK", "PROCESS_EXECUTION"]
    assert "ACCESSIBILITY" in report["capabilities"]
    assert report["capabilities"]["ACCESSIBILITY"]["state"] == READY
    assert report["machine_ready"] is True


def test_missing_native_owner_is_not_claimed_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hawking.permissions.resolve_native_helper",
        lambda _workspace: (_ for _ in ()).throw(RuntimeError("no stable helper fixture")),
    )
    snapshot = PermissionsService(tmp_path).status(["ACCESSIBILITY"], refresh=True)

    assert snapshot["native"]["state"] == UNAVAILABLE
    assert snapshot["capabilities"]["ACCESSIBILITY"]["state"] == "MISSING"
    with pytest.raises(PermissionRequired) as error:
        PermissionsService(tmp_path).require(["ACCESSIBILITY"])
    assert error.value.to_dict()["failure_class"] == "PERMISSION_BLOCKED"
    assert error.value.to_dict()["missing"][0]["capability"] == "ACCESSIBILITY"


def test_permission_required_does_not_dispatch_a_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hawking import permissions as permission_module
    from hawking import workunit_owner as owner_module

    class DeniedService:
        def __init__(self, _workspace: Path) -> None:
            pass

        def require(self, required, *, external_roots=None):
            del external_roots
            names = list(required)
            snapshot = {"checked_at": 123.0, "capabilities": {}}
            raise PermissionRequired(
                [{"capability": names[0], "state": "MISSING"}],
                required=names,
                snapshot=snapshot,
            )

    monkeypatch.setattr(permission_module, "PermissionsService", DeniedService)
    monkeypatch.setattr(
        owner_module,
        "resolve_live_worker",
        lambda _model: (_ for _ in ()).throw(AssertionError("provider dispatch was reached")),
    )
    contract_path = tmp_path / "goal.contract.json"
    contract_path.write_text(json.dumps({
        "workunit_id": "PERMISSION-FIXTURE-1",
        "goal_id": "GOAL-PERMISSION",
        "goal_mode": "worker_research",
        "objective": "observe one authorized desktop application",
        "acceptance": ["permission preflight is durable"],
        "required_capabilities": ["ACCESSIBILITY"],
        "exact_next_action": "recheck Accessibility, then resume this WorkUnit",
    }), encoding="utf-8")

    with pytest.raises(PermissionRequired):
        owner_module.WorkunitOwner(tmp_path).start(
            contract_path=contract_path,
            workunit_id="PERMISSION-FIXTURE-1",
            background=False,
        )
    record = owner_module.WorkunitOwner(tmp_path).load("PERMISSION-FIXTURE-1")
    assert record.state == "BLOCKED"
    assert record.classification == "PERMISSION_BLOCKED"
    assert record.worker_status == "KILLED"
    assert record.exact_next_action == "recheck Accessibility, then resume this WorkUnit"
    assert any(item.get("kind") == "permission_blocked" for item in record.failures)


def test_workunit_permission_contract_round_trips() -> None:
    unit = WorkUnit(
        id="permission-round-trip",
        role="verifier",
        description="check a capability",
        required_capabilities=["accessibility"],
        capability_lease={"schema": "hawking.goal.capability_lease.v1", "required": ["ACCESSIBILITY"]},
    )
    restored = WorkUnit.from_dict(unit.to_dict())
    assert restored.required_capabilities == ["ACCESSIBILITY"]
    assert restored.capability_lease["required"] == ["ACCESSIBILITY"]


def test_native_release_store_refuses_unsigned_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "helper"
    candidate.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    candidate.chmod(0o755)
    identity = inspect_code_identity(candidate)
    assert identity["stable_owner"] is False

    store = NativeHelperReleaseStore(tmp_path / "owner")
    staged = store.stage(candidate)
    assert staged["candidate"] is not None
    with pytest.raises(NativeOwnerError, match="refusing to promote"):
        store.activate()


class _Backend:
    def complete(self, _payload, timeout=None):  # pragma: no cover - no cognition in this fixture
        raise AssertionError("permission route fixture must not invoke cognition")


def test_hweb_permission_route_is_local_and_uses_gateway_workspace(tmp_path: Path) -> None:
    assert "/hawking/web/permissions" in HAWKING_WEB_ENDPOINTS
    assert "/hawking/web/permissions" in REQUIRED_ENDPOINTS
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            _Backend(),
            "fixture",
            greedy=False,
            health={"status": "ok", "resident": "fixture"},
            stores={"state_root": str(tmp_path)},
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/hawking/web/permissions",
            timeout=10,
        ) as response:
            payload = json.loads(response.read())
        assert payload["schema"] == "hawking.machine.authority.v1"
        assert payload["capabilities"]["WORKSPACE_READ"]["state"] == READY
        assert payload["capabilities"]["WORKSPACE_WRITE"]["state"] == READY
        assert payload["snapshot"]["workspace"] == str(tmp_path.resolve())
    finally:
        server.shutdown()
        server.server_close()
