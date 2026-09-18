from __future__ import annotations

import os
import io
from pathlib import Path
import urllib.error

import pytest

from hawking import remote_cognition
from hawking import web_launcher


def _state() -> dict:
    return {
        "pid": 4242,
        "current_release": "web-test123456",
        "health": {"owner": {"daemon": "hawkingd", "pid": 4242}},
    }


def test_public_router_web_and_build_bypass_legacy_hcli_and_kimi():
    router = Path("/Users/scammermike/.local/share/hawking/current/hawking-router")
    if not router.is_file():
        pytest.skip("installed public router is not present on this host")
    text = router.read_text(encoding="utf-8")
    assert 'if [ "$cmd" = "web" ] || [ "$cmd" = "build" ]; then' in text
    assert "-m hcli" not in text
    assert "kimi-p0" not in text.lower()
    assert "-m hawking.web_launcher" in text
    assert "--profile build" in text


def test_healthy_hawkingd_is_reused_without_restart_or_model_start(monkeypatch, tmp_path):
    monkeypatch.setattr(web_launcher, "openrouter_credential_available", lambda: True)
    monkeypatch.setattr(web_launcher, "_cloud_roster_status", lambda *args, **kwargs: (3, 3))
    seen = []

    def snapshot(base, *, timeout):
        seen.append((base, timeout))
        return _state()

    def refused_restart(*args, **kwargs):
        raise AssertionError("healthy hawkingd must not restart")

    opened = []
    result = web_launcher.launch(
        root=tmp_path,
        no_browser=False,
        opener=lambda url: opened.append(url) or True,
        snapshot=snapshot,
        restart=refused_restart,
    )
    assert result == 0
    assert seen == [("http://127.0.0.1:8014", 8.0)]
    assert opened == ["http://hawking.localhost:8014/hawking/web/read"]


def test_current_web_contract_includes_durable_attachment_route():
    assert "/hawking/web/attachments" in web_launcher.REQUIRED_ENDPOINTS


def test_current_web_contract_includes_markdown_export_route():
    assert "/hawking/web/markdown" in web_launcher.REQUIRED_ENDPOINTS


def test_launcher_requires_current_auto_orchestration_contract(monkeypatch):
    monkeypatch.setattr(web_launcher, "_json_get", lambda *_args, **_kwargs: {
        "owner": {"daemon": "hawkingd", "pid": 4242},
        "endpoints": list(web_launcher.REQUIRED_ENDPOINTS),
        "hweb_capability_projection_revision": web_launcher.HWEB_CAPABILITY_PROJECTION_REVISION,
        "hweb_builder_session_revision": web_launcher.HWEB_BUILDER_SESSION_REVISION,
    })
    with pytest.raises(web_launcher.WebLaunchError, match="Auto orchestration"):
        web_launcher._contract_snapshot("http://127.0.0.1:8014")


def test_controlled_stop_identifies_waiting_canonical_daemon(monkeypatch):
    from hawking import web

    monkeypatch.setattr(
        web,
        "_get_json",
        lambda *_args, **_kwargs: {
            "state": "waiting_for_native_runtime",
            "owner": {"daemon": "hawkingd", "pid": 4242},
        },
    )
    assert web.daemon_health_for_stop("127.0.0.1", 8014)["owner"]["pid"] == 4242


def test_incompatible_surface_uses_controlled_recovery_callback(monkeypatch, tmp_path):
    monkeypatch.setattr(web_launcher, "openrouter_credential_available", lambda: True)
    monkeypatch.setattr(web_launcher, "_cloud_roster_status", lambda *args, **kwargs: (3, 3))
    calls = []

    def snapshot(base, *, timeout):
        calls.append(("probe", base))
        if len(calls) == 1:
            raise web_launcher.WebLaunchError("incompatible")
        return _state()

    def recover(root, *, host, port, timeout):
        calls.append(("recover", root, host, port))
        return _state()

    assert web_launcher.launch(
        root=tmp_path,
        no_browser=True,
        snapshot=snapshot,
        restart=recover,
    ) == 0
    assert calls == [
        ("probe", "http://127.0.0.1:8014"),
        ("recover", tmp_path, "127.0.0.1", 8014),
        ("probe", "http://127.0.0.1:8014"),
    ]


def test_keychain_only_resolver_uses_canonical_service_and_current_account(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HAWKING_OPENROUTER_KEYCHAIN_SERVICE", raising=False)
    monkeypatch.delenv("HAWKING_OPENROUTER_KEYCHAIN_ACCOUNT", raising=False)
    calls = []

    class Completed:
        returncode = 0
        stdout = "keychain-test-value\n"

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return Completed()

    monkeypatch.setattr(remote_cognition.subprocess, "run", run)
    resolver = remote_cognition._keychain_resolver_from_environment()
    assert resolver is not None
    assert resolver() == "keychain-test-value"
    argv, kwargs = calls[0]
    assert argv[:6] == ["security", "find-generic-password", "-s", "OpenRouter", "-a", os.environ.get("USER")]
    assert argv[-1] == "-w"
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_cli_web_dispatches_to_current_launcher(monkeypatch):
    import hawking.cli as cli

    monkeypatch.setattr(cli, "warn_if_stale", lambda: None)
    called = []
    import hawking.web_launcher as launcher

    monkeypatch.setattr(launcher, "main", lambda argv: called.append(argv) or 19)
    assert cli.main(["web", "--no-browser"]) == 19
    assert called == [["--no-browser"]]


def test_cli_chat_dispatches_to_the_same_current_launcher(monkeypatch):
    import hawking.cli as cli

    monkeypatch.setattr(cli, "warn_if_stale", lambda: None)
    called = []
    import hawking.web_launcher as launcher

    monkeypatch.setattr(launcher, "main", lambda argv: called.append(argv) or 19)
    assert cli.main(["chat", "--no-browser"]) == 19
    assert called == [["--no-browser"]]


def test_cli_without_a_verb_opens_the_same_current_launcher(monkeypatch):
    import hawking.cli as cli

    monkeypatch.setattr(cli, "warn_if_stale", lambda: None)
    called = []
    import hawking.web_launcher as launcher

    monkeypatch.setattr(launcher, "main", lambda argv: called.append(argv) or 19)
    assert cli.main([]) == 19
    assert called == [[]]


def test_cli_build_dispatches_to_current_launcher_not_legacy_web(monkeypatch):
    import hawking.cli as cli

    monkeypatch.setattr(cli, "warn_if_stale", lambda: None)
    called = []
    import hawking.web_launcher as launcher

    monkeypatch.setattr(launcher, "main", lambda argv: called.append(argv) or 23)
    assert cli.main(["build", "--no-browser"]) == 23
    assert called == [["--profile", "build", "--no-browser"]]


def test_builder_launch_mints_handoff_without_printing_its_secret(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(web_launcher, "openrouter_credential_available", lambda: True)
    monkeypatch.setattr(web_launcher, "_cloud_roster_status", lambda *args, **kwargs: (3, 3))
    monkeypatch.setattr(
        web_launcher, "_json_post",
        lambda *args, **kwargs: {"handoff_url": "/hawking/web/build/web-test/opaque-token"},
    )
    opened = []
    assert web_launcher.launch(
        root=tmp_path, profile="build", no_browser=False,
        opener=lambda url: opened.append(url) or True,
        snapshot=lambda *args, **kwargs: _state(),
    ) == 0
    assert opened == ["http://hawking.localhost:8014/hawking/web/build/web-test/opaque-token"]
    output = capsys.readouterr().out
    assert "opaque-token" not in output
    assert "kimi-p0" not in output.lower()


def test_builder_session_http_rejection_preserves_safe_status_and_reason(monkeypatch):
    response = io.BytesIO(b'{"error":{"message":"builder workspace does not match this Hawking daemon"}}')

    def reject(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:8014/hawking/web/build/session",
            400,
            "Bad Request",
            {},
            response,
        )

    monkeypatch.setattr(web_launcher.urllib.request, "urlopen", reject)
    with pytest.raises(web_launcher.WebLaunchError) as raised:
        web_launcher._json_post(
            "http://127.0.0.1:8014/hawking/web/build/session",
            {"workspace": "/workspace"},
        )
    assert str(raised.value) == (
        "Hawking builder session mint failed (HTTP 400): "
        "builder workspace does not match this Hawking daemon"
    )


def test_launcher_result_is_secret_free(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(web_launcher, "openrouter_credential_available", lambda: True)
    monkeypatch.setattr(web_launcher, "_cloud_roster_status", lambda *args, **kwargs: (3, 3))
    web_launcher.launch(root=tmp_path, no_browser=True, snapshot=lambda *a, **k: _state())
    output = capsys.readouterr().out
    assert "OPENROUTER_API_KEY" not in output
    assert "Authorization" not in output
    assert "keychain-test-value" not in output
