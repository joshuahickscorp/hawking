"""Native-only logical role routing for the canonical hawkingd surface."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from hawking.canonical_runtime import CanonicalRoleRouter, RuntimeRoleUnavailable
from hawking.catalog import Body


def _authority(tmp_path):
    path = tmp_path / "scheduling.json"
    path.write_text(json.dumps({
        "schema": "hawking.modellake.scheduling_authority.v2",
        "authority": {"status": "ACTIVE"},
        "campaign": {"phases": [{
            "id": "STAR_SERIES",
            "roles": {"PULSAR": {}, "MAGNETAR": {}, "THEMIS": {}},
        }]},
    }), encoding="utf-8")
    return path


def _body(*, name="PULSAR_R1", kind="noetic_native", role="pulsar",
          qualification="RESIDENT_QUALIFIED", bound=True):
    detail = {"revision_basis": "fixture"}
    if bound:
        detail["hawkingd_roles"] = {
            role: {
                "qualification": qualification,
                "evidence": ["receipts/future/PULSAR_R1_QUALIFICATION.json"],
            },
        }
    return Body(
        name=name,
        path=f"/tmp/{name}.native.json",
        kind=kind,
        source="gravity",
        revision="a" * 64,
        admitted=True,
        supported_actions=("execute", "serve", "web"),
        detail=detail,
    )


def test_pulsar_is_bound_only_to_explicit_native_qualified_artifact(tmp_path):
    body = _body()
    router = CanonicalRoleRouter(
        bodies_provider=lambda: [body], scheduling_authority=_authority(tmp_path),
    )

    route = router.route("pulsar")

    assert route.role == "pulsar"
    assert route.action.name == "PULSAR_R1"
    assert route.action.kind == "noetic_native"
    assert router.route("PULSAR_R1").action.has_same_binding(route.action)


def test_any_admitted_native_artifact_is_directly_addressable_without_role_claim(tmp_path):
    body = _body(name="OTHER_HAWKING_MODEL", bound=False)
    router = CanonicalRoleRouter(
        bodies_provider=lambda: [body], scheduling_authority=_authority(tmp_path),
    )

    route = router.route("OTHER_HAWKING_MODEL")

    assert route.role == "artifact"
    assert route.action.name == "OTHER_HAWKING_MODEL"
    rows = {row["id"]: row for row in router.public_models()}
    assert rows["OTHER_HAWKING_MODEL"]["hawking"]["immutable_artifact"] is True
    with pytest.raises(RuntimeRoleUnavailable, match="pulsar"):
        router.route("pulsar")


def test_auto_is_deterministic_and_never_falls_back_to_a_different_role(tmp_path):
    body = _body()
    router = CanonicalRoleRouter(
        bodies_provider=lambda: [body], scheduling_authority=_authority(tmp_path),
    )

    assert router.route("hawking-auto", {"objective": "small repository edit"}).role == "pulsar"
    with pytest.raises(RuntimeRoleUnavailable, match="magnetar"):
        router.route("hawking-auto", {"objective": "novel representation hypothesis"})


def test_mlx_cannot_claim_a_canonical_role_even_with_a_binding(tmp_path):
    mlx = _body(kind="mlx")
    router = CanonicalRoleRouter(
        bodies_provider=lambda: [mlx], scheduling_authority=_authority(tmp_path),
    )

    with pytest.raises(RuntimeRoleUnavailable, match="native"):
        router.route("pulsar")

    rows = {row["id"]: row for row in router.public_models()}
    assert rows["pulsar"]["hawking"]["availability"] == "WITHHELD"
    assert "KIMI_P0_OPERATIONAL" not in rows


def test_unbound_legacy_provider_name_is_not_a_public_selector(tmp_path):
    router = CanonicalRoleRouter(
        bodies_provider=lambda: [_body(bound=False)],
        scheduling_authority=_authority(tmp_path),
    )

    with pytest.raises(RuntimeRoleUnavailable) as exc:
        router.route("KIMI_P0_OPERATIONAL")

    assert exc.value.code == "MODEL_NOT_CANONICAL"


def test_bad_scheduling_authority_withholds_all_roles(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    router = CanonicalRoleRouter(
        bodies_provider=lambda: [_body()], scheduling_authority=bad,
    )

    with pytest.raises(RuntimeRoleUnavailable, match="unexpected schema"):
        router.route("pulsar")


def test_waiting_canonical_daemon_owns_the_api_without_loading_a_legacy_body():
    from hawking.serve import DEFAULT_MODEL, DEFAULT_PORT, build_canonical_server

    assert DEFAULT_PORT == 8014
    assert DEFAULT_MODEL == "hawking-auto"
    httpd, identity, health = build_canonical_server(
        "pulsar", host="127.0.0.1", port=0, ready_timeout=0.01,
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert identity == "hawkingd-awaiting-native"
        assert health["status"] == "waiting_for_native_runtime"
        with urllib.request.urlopen(base + "/api/runtime/roles", timeout=5) as response:
            roles = json.loads(response.read())
        assert roles["owner"] == "hawkingd"
        assert all(row["state"] == "WITHHELD" for row in roles["roles"])
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
            models = json.loads(response.read())
        ids = {row["id"] for row in models["data"]}
        assert ids == {"KIMI_P0_OPERATIONAL"}
        request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({
                "model": "pulsar",
                "messages": [{"role": "user", "content": "hello"}],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=5)
        assert refused.value.code == 503
        body = json.loads(refused.value.read())
        assert body["error"]["type"] == "runtime_role_unavailable"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
