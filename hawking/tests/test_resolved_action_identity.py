"""Resolved-action transport stays bound to the admitted Gravity record.

These tests use a tiny on-disk Gravity registry and a structurally valid native
profile. They deliberately do not launch a provider, resident, HTTP server,
or browser: the boundary under test is the signed-in-data shape passed between
the catalog, CLI, Web launcher, and resident owner.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

from http.server import ThreadingHTTPServer

import pytest

import hawking.catalog as catalog_module
from hawking.catalog import (
    ResolvedAction,
    resolve_action,
    resolved_action_from_json,
    resolved_action_from_wire,
    resolved_action_json,
)


ARTIFACT_ID = "TEST_RESOLVED_ACTION"
KIMI_P0_ARTIFACT_ID = "KIMI_P0_OPERATIONAL"


def _write_registry(
    registry: Path,
    model: Path,
    *,
    revision: str = "a" * 64,
    actions: tuple[str, ...] = ("execute", "serve", "web"),
    status: str = "ADMITTED",
    artifact_id: str = ARTIFACT_ID,
    authority: dict[str, object] | None = None,
) -> None:
    artifact: dict[str, object] = {
        "id": artifact_id,
        "path": str(model),
        "kind": "noetic_native",
        "status": status,
        # This is intentionally a registry declaration, not a claim that
        # the tiny fixture's bytes hash to this value.
        "revision": revision,
        "revision_basis": "fixture_registry_declared_revision",
        "supported_actions": list(actions),
        "role": "resolved-action test fixture",
    }
    if authority is not None:
        artifact["authority"] = authority
    registry.write_text(json.dumps({
        "schema": catalog_module.GRAVITY_REGISTRY_SCHEMA,
        "status": catalog_module.GRAVITY_REGISTRY_STATUS,
        "artifacts": [artifact],
    }), encoding="utf-8")


@pytest.fixture
def gravity_artifact(monkeypatch, tmp_path):
    """Install one admitted artifact without consulting the real ModelLake."""
    model = tmp_path / "hawking-native.fixture.json"
    model.write_text(json.dumps({
        "profile_schema": "hawking.provider.profile.v1",
        "provider": "native",
        "runtime": "hawking-native",
        "resident_identity": "test-native",
    }), encoding="utf-8")
    registry = tmp_path / "gravity-artifacts.json"
    _write_registry(registry, model)
    monkeypatch.setattr(catalog_module, "GRAVITY_REGISTRY", registry)
    monkeypatch.setattr(catalog_module, "MODELLAKE", tmp_path / "absent-modellake")
    return {"model": model, "registry": registry}


def _action(kind: str, gravity_artifact, artifact_id: str = ARTIFACT_ID):
    resolved = resolve_action(artifact_id, kind)
    assert resolved is not None
    return resolved


def test_gravity_action_round_trips_and_revalidates(gravity_artifact):
    action = _action("execute", gravity_artifact)

    wire = action.to_wire()
    assert wire["schema"] == "hawking.resolved_action.v1"
    assert wire["action"] == "execute"
    assert wire["execution_intent"] == "interactive_execute"
    assert wire["artifact"]["path"] == str(gravity_artifact["model"].resolve())
    assert wire["artifact"]["revision_basis"] == "fixture_registry_declared_revision"
    assert wire["catalog"]["identity"].startswith("gravity-registry:")
    assert wire["grant"]["admitted"] is True
    assert wire["grant"]["supported_actions"] == ["execute", "serve", "web"]

    # Structural parsing alone has no private Body cache; revalidation restores
    # it from the current Gravity authority before an action can run.
    parsed = ResolvedAction.from_wire(wire)
    assert parsed.body is None
    revalidated = resolved_action_from_json(resolved_action_json(action))
    assert revalidated.to_wire() == wire
    assert revalidated.body is not None


def test_tampered_wire_is_refused_before_catalog_reuse(gravity_artifact):
    action = _action("serve", gravity_artifact)
    tampered = deepcopy(action.to_wire())
    tampered["artifact"]["revision"] = "b" * 64

    with pytest.raises(ValueError, match="grant digest"):
        ResolvedAction.from_wire(tampered)

    bad_binding = deepcopy(action.to_wire())
    bad_binding["reuse_binding"]["catalog_revision"] = "b" * 64
    with pytest.raises(ValueError, match="reuse binding"):
        ResolvedAction.from_wire(bad_binding)

    # Rust accepts only canonical lowercase SHA-256 fields. The Python parser
    # must enforce that same wire contract rather than quietly normalizing an
    # input that Rust would refuse.
    uppercase = deepcopy(action.to_wire())
    uppercase["artifact"]["revision"] = "A" * 64
    with pytest.raises(ValueError, match="incomplete or not admitted"):
        ResolvedAction.from_wire(uppercase)


def test_stale_or_revoked_gravity_binding_is_refused(gravity_artifact):
    action = _action("web", gravity_artifact)
    wire = action.to_wire()

    _write_registry(
        gravity_artifact["registry"], gravity_artifact["model"], revision="b" * 64
    )
    with pytest.raises(PermissionError, match="no longer matches"):
        resolved_action_from_wire(wire)

    # A status revocation removes the body from the execution catalog entirely.
    _write_registry(
        gravity_artifact["registry"], gravity_artifact["model"], status="REVOKED"
    )
    with pytest.raises(PermissionError, match="no longer names one admitted"):
        resolved_action_from_wire(wire)


def test_ungranted_or_unknown_actions_are_refused(gravity_artifact):
    _write_registry(
        gravity_artifact["registry"], gravity_artifact["model"], actions=("execute",)
    )

    assert _action("execute", gravity_artifact).action == "execute"
    with pytest.raises(PermissionError, match="not admitted for action"):
        resolve_action(ARTIFACT_ID, "serve")
    with pytest.raises(ValueError, match="unsupported Hawking action"):
        resolve_action(ARTIFACT_ID, "invent")


def test_resolve_only_cli_emits_the_full_action_contract(gravity_artifact, capsys):
    from hawking.use import main as use_main

    assert use_main([ARTIFACT_ID, "--resolve-only", "--action", "execute"]) == 0
    wire = json.loads(capsys.readouterr().out)
    assert wire["schema"] == "hawking.resolved_action.v1"
    assert wire["action"] == "execute"
    assert wire["artifact"]["path"] == str(gravity_artifact["model"].resolve())
    assert set(wire) >= {
        "schema", "action", "execution_intent", "artifact", "catalog", "grant",
        "reuse_binding",
    }


def test_terminal_cli_accepts_only_an_execute_action(gravity_artifact):
    from hawking.cli import parse_hawking_args

    execute = _action("execute", gravity_artifact)
    args = parse_hawking_args([
        "--resolved-action-json", resolved_action_json(execute), "inspect the binding",
    ])
    assert args.model == execute.path
    assert args.resolved_action.to_wire() == execute.to_wire()

    serve = _action("serve", gravity_artifact)
    with pytest.raises(SystemExit) as refused:
        parse_hawking_args([
            "--resolved-action-json", resolved_action_json(serve), "inspect the binding",
        ])
    assert refused.value.code == 2


def test_controller_uses_the_revalidated_execute_action_without_another_selector(
    gravity_artifact, tmp_path
):
    from hawking.controller import Controller

    action = _action("execute", gravity_artifact)

    class Registry:
        selected = None

        def resolve(self, **_kwargs):
            raise AssertionError("Controller re-ran ModelRegistry for a resolved action")

    controller = Controller(tmp_path, resolved_action=action, registry=Registry())
    try:
        assert controller.model == action.path
        assert controller.session.model == action.path
        assert controller.engine.model_name == action.path
        assert controller.status()["resolved_action"] == action.to_wire()
    finally:
        controller.shutdown()


def test_controller_normalizes_a_legacy_admitted_model_path(gravity_artifact, tmp_path):
    from hawking.controller import Controller

    action = _action("execute", gravity_artifact)
    controller = Controller(tmp_path, model=action.path)
    try:
        assert controller.resolved_action is not None
        assert controller.resolved_action.to_wire() == action.to_wire()
        assert controller.model == action.path
    finally:
        controller.shutdown()


def test_web_launcher_passes_full_action_json_not_a_bare_model(
    monkeypatch, gravity_artifact, tmp_path
):
    import hawking.web as web

    action = _action("web", gravity_artifact)
    captured = {}

    class FakeProcess:
        pid = 77

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(web.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        web.shutil, "which",
        lambda name: str(tmp_path / "hawkingd") if name == "hawkingd" else None,
    )

    proc, _log = web.start_surface(
        "ignored-by-resolved-action", "127.0.0.1", 8011, tmp_path / "logs",
        resolved_action=action,
    )
    assert proc.pid == 77
    argv = captured["argv"]
    assert "--model" not in argv
    index = argv.index("--resolved-action-json")
    assert json.loads(argv[index + 1]) == action.to_wire()


def test_web_same_path_with_bad_binding_posts_full_action(monkeypatch, gravity_artifact):
    import hawking.use as use
    from hawking.web import reconcile_requested_resident

    action = _action("web", gravity_artifact)
    stale = deepcopy(action.to_wire())
    stale["reuse_binding"]["grant_digest"] = "b" * 64
    calls = []

    def fake_post(url, body, timeout):
        calls.append((url, body, timeout))
        return 200, {
            "resident": action.name,
            "switched": True,
            "resolved_action": action.to_wire(),
        }

    monkeypatch.setattr(use, "post_json", fake_post)
    updated = reconcile_requested_resident(
        {"resident": action.name, "model": action.path, "resolved_action": stale},
        action,
        "http://127.0.0.1:8011/v1",
        5.0,
    )

    assert updated["resolved_action"] == action.to_wire()
    assert calls == [(
        "http://127.0.0.1:8011/v1/switch",
        {"resolved_action": action.to_wire()},
        5.0,
    )]


def test_resident_noop_switch_keeps_the_resolved_action(gravity_artifact):
    from hawking.serve import Resident

    action = _action("serve", gravity_artifact)

    class FakeBackend:
        pass

    resident = Resident(action.body, FakeBackend(), resolved_action=action)
    result = resident.switch(action)
    assert result["switched"] is False
    assert result["resident"] == action.name
    assert result["resolved_action"] == action.to_wire()


def test_web_only_resident_keeps_its_web_action_while_completing(gravity_artifact):
    """A Web grant authorizes the resident that answers for that Web surface."""
    from hawking.serve import Resident

    _write_registry(
        gravity_artifact["registry"], gravity_artifact["model"], actions=("web",)
    )
    action = _action("web", gravity_artifact)

    class FakeBackend:
        def __init__(self):
            self.calls = []

        def complete(self, payload, timeout=None):
            self.calls.append((payload, timeout))
            return "web answer"

    backend = FakeBackend()
    resident = Resident(action.body, backend, resolved_action=action)
    assert resident.complete({"messages": [{"role": "user", "content": "hello"}]}) == "web answer"
    assert backend.calls == [
        ({"messages": [{"role": "user", "content": "hello"}]}, None)
    ]
    assert resident.resolved_action.action == "web"


def test_serve_rejects_an_execute_contract_before_spawning(monkeypatch, gravity_artifact):
    import hawking.runtime_iface as runtime_iface
    from hawking.serve import build_server

    execute = _action("execute", gravity_artifact)

    def must_not_spawn(*_args, **_kwargs):
        raise AssertionError("serve tried to spawn an execute-only contract")

    monkeypatch.setattr(runtime_iface, "make_backend_for_model", must_not_spawn)
    with pytest.raises(PermissionError, match="cannot start a resident surface"):
        build_server(execute.path, resolved_action=execute)


def test_kimi_write_surface_is_refused_before_provider_spawn(
    monkeypatch, gravity_artifact
):
    """A bounded Kimi artifact cannot turn ``hawking build`` into a write door."""
    import hawking.runtime_iface as runtime_iface
    from hawking.serve import build_server

    _write_registry(
        gravity_artifact["registry"],
        gravity_artifact["model"],
        artifact_id=KIMI_P0_ARTIFACT_ID,
        authority={
            "admitted": ["read/research HAWKING work"],
            "withheld": ["autonomous repository writing"],
        },
    )
    kimi = _action("serve", gravity_artifact, KIMI_P0_ARTIFACT_ID)

    def must_not_construct_provider(*_args, **_kwargs):
        raise AssertionError("write-withheld Kimi reached provider construction")

    monkeypatch.setattr(runtime_iface, "make_backend_for_model", must_not_construct_provider)
    with pytest.raises(PermissionError, match="does not grant repository-write authority"):
        build_server(kimi.path, resolved_action=kimi, write=True)


def test_write_resident_cannot_switch_to_kimi_before_stopping_current_provider(
    monkeypatch, gravity_artifact
):
    """A generic write resident cannot bypass the same artifact policy by switch."""
    import hawking.runtime_iface as runtime_iface
    from hawking.catalog import Body
    from hawking.serve import Resident

    _write_registry(
        gravity_artifact["registry"],
        gravity_artifact["model"],
        artifact_id=KIMI_P0_ARTIFACT_ID,
        authority={
            "admitted": ["read/research HAWKING work"],
            "withheld": ["autonomous repository writing"],
        },
    )
    kimi = _action("serve", gravity_artifact, KIMI_P0_ARTIFACT_ID)

    class CurrentBackend:
        def stop(self):
            raise AssertionError("write-policy refusal stopped the current provider")

    resident = Resident(
        Body(name="GENERIC_WRITER", path="/tmp/generic", kind="mlx"),
        CurrentBackend(),
        write=True,
    )
    monkeypatch.setattr(
        runtime_iface,
        "make_backend_for_model",
        lambda *_args, **_kwargs: pytest.fail(
            "write-policy refusal constructed the Kimi provider"
        ),
    )

    with pytest.raises(PermissionError, match="does not grant repository-write authority"):
        resident.switch(kimi)


def test_kimi_read_resident_cannot_be_requalified_as_a_writer(gravity_artifact):
    """A later tool qualification cannot elevate an already-loaded Kimi body."""
    from hawking.serve import Resident

    _write_registry(
        gravity_artifact["registry"],
        gravity_artifact["model"],
        artifact_id=KIMI_P0_ARTIFACT_ID,
        authority={
            "admitted": ["read/research HAWKING work"],
            "withheld": ["autonomous repository writing"],
        },
    )
    kimi = _action("serve", gravity_artifact, KIMI_P0_ARTIFACT_ID)
    resident = Resident(kimi.body, object())

    with pytest.raises(PermissionError, match="does not grant repository-write authority"):
        resident.qualify_tools(write=True)


def test_explicitly_allowed_generic_writer_can_build_a_write_surface(
    monkeypatch, gravity_artifact
):
    """The Kimi guard is artifact-scoped; ordinary admitted writers still work."""
    import hawking.runtime_iface as runtime_iface
    import hawking.serve as serve

    _write_registry(
        gravity_artifact["registry"],
        gravity_artifact["model"],
        authority={
            "admitted": ["repository writing"],
            "withheld": [],
        },
    )
    writer = _action("serve", gravity_artifact)

    class FakeBackend:
        def __init__(self):
            self.spawned = False

        def spawn(self):
            self.spawned = True

        def ready(self, _timeout):
            return True

        def stop(self):
            return {"gone": True}

    class FakeHttpd:
        def __init__(self, address, handler):
            self.address = address
            self.handler = handler

    backend = FakeBackend()
    monkeypatch.setattr(runtime_iface, "make_backend_for_model", lambda *_a, **_kw: backend)
    monkeypatch.setattr(serve, "ThreadingHTTPServer", FakeHttpd)

    httpd, identity, _health = serve.build_server(
        writer.path,
        host="127.0.0.1",
        port=0,
        ready_timeout=0.01,
        resolved_action=writer,
        write=True,
    )

    assert backend.spawned is True
    assert identity == ARTIFACT_ID
    assert httpd.backend.write_authority is True


def test_switch_endpoint_carries_a_resolved_action_and_health_publishes_it(
    gravity_artifact
):
    from hawking.serve import Resident, make_handler

    serve_action = _action("serve", gravity_artifact)
    execute_action = _action("execute", gravity_artifact)

    class FakeBackend:
        pass

    resident = Resident(
        serve_action.body, FakeBackend(), resolved_action=serve_action,
    )
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            resident,
            serve_action.name,
            greedy=False,
            health={"status": "ok"},
        ),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            health = json.loads(response.read())
        assert health["resolved_action"] == serve_action.to_wire()
        assert health["resident_binding"] == serve_action.binding

        request = urllib.request.Request(
            base + "/v1/switch",
            data=json.dumps({"resolved_action": execute_action.to_wire()}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=5)
        assert refused.value.code == 403
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "include_model",
    [
        False,
        True,
    ],
)
def test_current_resident_refuses_a_revoked_action_before_completion(
    monkeypatch, gravity_artifact, include_model
):
    """An omitted or same-name request cannot extend a revoked resident grant."""
    import hawking.runtime_iface as runtime_iface
    from hawking.serve import Resident

    action = _action("serve", gravity_artifact)

    class FakeBackend:
        def __init__(self):
            self.completions = []
            self.stops = 0

        def complete(self, payload, timeout=None):
            self.completions.append((payload, timeout))
            return object()

    backend = FakeBackend()
    resident = Resident(action.body, backend, resolved_action=action)

    _write_registry(
        gravity_artifact["registry"], gravity_artifact["model"], status="REVOKED"
    )

    def no_replacement_resident(*_args, **_kwargs):
        raise AssertionError("a stale current action attempted to start a replacement")

    monkeypatch.setattr(runtime_iface, "make_backend_for_model", no_replacement_resident)
    payload = {"messages": [{"role": "user", "content": "is this current?"}]}
    if include_model:
        payload["model"] = action.name

    with pytest.raises((LookupError, PermissionError)):
        resident.complete(payload)
    assert backend.completions == []
    assert backend.stops == 0


@pytest.mark.parametrize("include_model", [False, True])
def test_current_resident_reloads_a_changed_binding_before_completion(
    monkeypatch, gravity_artifact, include_model
):
    """A changed current revision may run only after a replacement resident is ready."""
    import hawking.runtime_iface as runtime_iface
    from hawking.serve import Resident

    action = _action("serve", gravity_artifact)

    class FakeBackend:
        def __init__(self, answer):
            self.answer = answer
            self.completions = []
            self.stops = 0
            self.spawned = False

        def complete(self, payload, timeout=None):
            self.completions.append((payload, timeout))
            return self.answer

        def stop(self):
            self.stops += 1
            return {"gone": True}

        def spawn(self):
            self.spawned = True

        def ready(self, _timeout):
            return True

    old_backend = FakeBackend("stale answer")
    replacement = FakeBackend("current answer")
    resident = Resident(action.body, old_backend, resolved_action=action)
    _write_registry(
        gravity_artifact["registry"], gravity_artifact["model"], revision="b" * 64
    )
    monkeypatch.setattr(
        runtime_iface, "make_backend_for_model", lambda *_args, **_kwargs: replacement
    )
    monkeypatch.setattr(
        Resident, "qualify_tools", lambda self: {"qualified": False},
    )
    payload = {"messages": [{"role": "user", "content": "is this current?"}]}
    if include_model:
        payload["model"] = action.name

    assert resident.complete(payload) == "current answer"
    assert old_backend.stops == 1
    assert old_backend.completions == []
    assert replacement.spawned is True
    assert replacement.completions == [
        ({"messages": [{"role": "user", "content": "is this current?"}]}, None)
    ]
    assert resident.resolved_action.artifact_revision == "b" * 64


def test_build_server_keeps_a_direct_web_action_in_health(monkeypatch, gravity_artifact):
    """The direct serve adapter must not erase the Web action it was handed."""
    import hawking.runtime_iface as runtime_iface
    import hawking.serve as serve

    action = _action("web", gravity_artifact)

    class FakeBackend:
        def __init__(self):
            self.spawned = False

        def spawn(self):
            self.spawned = True

        def ready(self, _timeout):
            return True

        def stop(self):
            return {}

    backend = FakeBackend()

    class FakeHttpd:
        def __init__(self, address, handler):
            self.address = address
            self.handler = handler

    monkeypatch.setattr(runtime_iface, "make_backend_for_model", lambda *_a, **_kw: backend)
    monkeypatch.setattr(serve, "ThreadingHTTPServer", FakeHttpd)

    httpd, identity, health = serve.build_server(
        action.path,
        host="127.0.0.1",
        port=0,
        ready_timeout=0.01,
        resolved_action=action,
    )

    assert backend.spawned is True
    assert identity == action.name
    assert httpd.backend.resolved_action.to_wire() == action.to_wire()
    assert health["resolved_action"] == action.to_wire()
    assert health["resident_binding"] == action.binding
