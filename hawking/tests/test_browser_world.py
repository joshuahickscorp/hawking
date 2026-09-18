"""Live local qualification for Hawking's Playwright WorldState boundary."""
from __future__ import annotations

import json
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest

from hawking.browser_world import BrowserWorld
from hawking.goal_surface import (
    browser_session_id_for_goal,
    create_goal,
    dispatch_goal,
    normalize_browser_authority,
)
from hawking.remote_cognition import _tool_trace_completion, _workunit_completion_contract
from hawking.tool_registry import (
    EXTERNAL_WRITE,
    READ_ONLY,
    RESEARCH,
    REVERSIBLE_RUNTIME,
    default_tool_registry,
)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format, *_args):  # noqa: D401 - test server noise is not evidence.
        return


def _fixture_server(root: Path):
    (root / "index.html").write_text(
        """<!doctype html><title>Hawking Browser Fixture</title>
        <button aria-label=\"Toggle\" onclick=\"document.querySelector('#state').textContent='done'\">Toggle</button>
        <p id=\"state\">idle</p>""",
        encoding="utf-8",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_browser_world_observes_actions_deltas_and_artifacts():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        server = _fixture_server(root)
        world = BrowserWorld(root)
        try:
            url = f"http://127.0.0.1:{server.server_port}/index.html"
            opened = world.open({"browser_session_id": "BROWSER-QUALIFY", "url": url})
            observed = world.invoke("observe", {"browser_session_id": "BROWSER-QUALIFY"})
            found = world.invoke(
                "find",
                {"browser_session_id": "BROWSER-QUALIFY", "target": {"label": "Toggle"}},
            )
            acted = world.invoke(
                "click",
                {"browser_session_id": "BROWSER-QUALIFY", "target": {"label": "Toggle"}},
            )
            verified = world.invoke(
                "verify",
                {"browser_session_id": "BROWSER-QUALIFY", "text": "done"},
            )
            changed = BrowserWorld(root).invoke(
                "world.changed",
                {"browser_session_id": "BROWSER-QUALIFY", "since_revision": observed["world_revision"]},
            )
            screenshot = world.invoke("screenshot", {"browser_session_id": "BROWSER-QUALIFY"})

            assert opened["page"]["title"] == "Hawking Browser Fixture"
            assert observed["page_id"] == opened["page"]["page_id"]
            assert found["count"] == 1
            assert acted["resolved_by"] == {"label": "Toggle", "exact": False}
            assert verified["verified"] is True
            assert changed["gap"] is False
            assert changed["events"]
            artifact = screenshot["artifact"]
            assert artifact["retention_class"] == "DERIVED"
            assert len(artifact["sha256"]) == 64
            assert Path(artifact["path"]).is_file()
        finally:
            world.close({"browser_session_id": "BROWSER-QUALIFY"})
            world.runtime.shutdown()
            server.shutdown()
            server.server_close()


def test_browser_registry_requires_external_action_authority():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        server = _fixture_server(root)
        session_id = "BROWSER-REGISTRY"
        url = f"http://127.0.0.1:{server.server_port}/index.html"
        try:
            read_registry = default_tool_registry(root, permissions={READ_ONLY, RESEARCH, REVERSIBLE_RUNTIME})
            assert read_registry.invoke(
                "browser.open", {"browser_session_id": session_id, "url": url}
            ).ok
            denied = read_registry.invoke(
                "browser.click", {"browser_session_id": session_id, "target": {"label": "Toggle"}}
            )
            assert denied.ok is False
            assert denied.failure_class == "PERMISSION_DENIED"

            action_registry = default_tool_registry(
                root,
                permissions={READ_ONLY, RESEARCH, REVERSIBLE_RUNTIME, EXTERNAL_WRITE},
            )
            clicked = action_registry.invoke(
                "browser.click", {"browser_session_id": session_id, "target": {"label": "Toggle"}}
            )
            assert clicked.ok
            checked = action_registry.invoke(
                "browser.verify", {"browser_session_id": session_id, "text": "done"}
            )
            assert checked.ok
            assert checked.value["verified"] is True
            assert action_registry.get("browser.changed") is not None
        finally:
            BrowserWorld(root).close({"browser_session_id": session_id})
            BrowserWorld(root).runtime.shutdown()
            server.shutdown()
            server.server_close()


def test_goal_browser_authority_is_durable_scoped_and_requires_an_origin(tmp_path: Path):
    with pytest.raises(ValueError, match="explicit allowed origin"):
        normalize_browser_authority({"actions": True}, goal_id="GOAL-ORIGIN")

    goal = create_goal(
        tmp_path,
        objective="Use the local fixture as a disposable browser repair target.",
        resident="deepseek/deepseek-v4.1-flash",
        browser_authority={
            "observe": True,
            "actions": True,
            "origins": ["http://127.0.0.1:9123", "http://127.0.0.1:9123"],
        },
    )
    browser = goal["browser_authority"]
    assert browser["browser_session_id"] == browser_session_id_for_goal(goal["goal_id"])
    assert browser["origins"] == ["http://127.0.0.1:9123"]

    # Dispatch never needs a provider request to persist the exact authority;
    # replacing the existing owner start makes this a deterministic contract
    # test rather than a remote-spend test.
    with mock.patch(
        "hawking.workunit_owner.WorkunitOwner.start",
        return_value={"owner": {"checkpoint_path": "fixture-checkpoint.json"}},
    ):
        dispatched = dispatch_goal(
            tmp_path,
            {
                **goal,
                "workunit_id": "D-BROWSER-FIXTURE",
                "workunit_objective": "Repair only the disposable fixture.",
                "workunit_acceptance": ["fixture repair evidence"],
                "budget_authorized_usd": 30.0,
                "workunit_budget_usd": 0.10,
            },
            model="deepseek/deepseek-v4.1-flash",
            background=False,
        )
    contract = Path(dispatched["contract"])
    document = json.loads(contract.read_text(encoding="utf-8"))
    authority = document["authority"]
    assert authority["browser"] == browser
    assert authority["capabilities"] == ["browser.observe", "browser.actions"]
    assert document["objective"] == "Repair only the disposable fixture."
    assert document["acceptance"] == ["fixture repair evidence"]
    assert document["budget_authorized_usd"] == 0.10
    assert document["cost_authorization"] == {"max_usd": 0.10}
    assert document["goal_terminal_on_workunit_complete"] is False


def test_browser_completion_requires_a_true_verify_predicate():
    contract = _workunit_completion_contract({
        "completion_contract": {
            "required_successful_tools": ["browser.verify"],
            "required_verified_tools": ["browser.verify"],
        },
    })
    false_verify = [{
        "tool": "browser.verify",
        "ok": True,
        "result": {"value": {"verified": False}},
    }]
    incomplete = _tool_trace_completion(false_verify, None, contract)
    assert incomplete["complete"] is False
    assert incomplete["unmet"] == ["browser.verify:verified"]

    true_verify = [{
        "tool": "browser.verify",
        "ok": True,
        "result": {"value": {"verified": True}},
    }]
    complete = _tool_trace_completion(true_verify, None, contract)
    assert complete["complete"] is True
    assert complete["verified_tools"] == ["browser.verify"]
