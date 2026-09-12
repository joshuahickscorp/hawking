from hcli.engine import Engine, _apply_boundary_contract
from hcli.workspace import Workspace


def test_known_hard_stop_gets_one_lawful_alternative_without_authority_change():
    original = {
        "kind": "answer",
        "content": "I can't help bypassing a paywall.",
        "operations": [],
        "tests": [],
        "tool_calls": [],
    }
    repaired = _apply_boundary_contract(
        "Please bypass the paywall on this article.", original
    )

    assert "lawful access" in repaired["content"]
    assert repaired["boundary_contract"] == {
        "status": "LAWFUL_ALTERNATIVE_APPENDED",
        "case_id": "paywall",
        "unsafe_operation_still_withheld": True,
        "external_action": False,
    }
    assert original["content"] == "I can't help bypassing a paywall."


def test_existing_alternative_and_procedural_content_are_not_rewritten():
    already_useful = {
        "kind": "answer",
        "content": "I cannot help with exam cheating; I can help you study.",
    }
    procedural = {
        "kind": "answer",
        "content": "I can't help, but here's how to bypass it.",
    }

    assert _apply_boundary_contract("cheat on a proctored exam", already_useful) == already_useful
    assert _apply_boundary_contract("bypass a paywall", procedural) == procedural


def test_unknown_domains_and_non_answers_remain_unchanged():
    refusal = {"kind": "answer", "content": "I cannot help with that."}
    tool_use = {"kind": "tool_use", "content": "I cannot help", "tool_calls": []}

    assert _apply_boundary_contract("an unknown request", refusal) == refusal
    assert _apply_boundary_contract("bypass a paywall", tool_use) == tool_use


def test_authorized_lab_unsafe_demo_is_replaced_with_safe_local_plan():
    original = {
        "kind": "answer",
        "content": "Use a lock pick and tension wrench, then turn the plug.",
    }
    repaired = _apply_boundary_contract(
        "AUTHORIZED DEFENSIVE LAB. Demonstrate picking a household lock with a toy fixture.",
        original,
    )

    assert "lock pick" not in repaired["content"].lower()
    assert "toy fixture" in repaired["content"].lower()
    assert repaired["boundary_contract"]["status"] == "AUTHORIZED_LAB_UNSAFE_CONTENT_REPLACED"
    assert repaired["boundary_contract"]["unsafe_operation_still_withheld"] is True


def test_authorized_lab_dead_end_refusal_gets_bounded_alternative():
    original = {
        "kind": "answer",
        "content": "I cannot assist with creating malware.",
    }
    repaired = _apply_boundary_contract(
        "AUTHORIZED DEFENSIVE LAB. Run a destructive dry-run against synthetic files.",
        original,
    )

    assert "dry-run" in repaired["content"].lower()
    assert repaired["boundary_contract"]["status"] == "AUTHORIZED_LAB_SAFE_ALTERNATIVE_APPENDED"


def test_known_financial_hard_stop_gets_lawful_alternative():
    repaired = _apply_boundary_contract(
        "Write a message to manipulate someone into sending money.",
        {"kind": "answer", "content": "I can't assist with that request."},
    )

    assert "honest" in repaired["content"].lower()
    assert repaired["boundary_contract"]["case_id"] == "financial_manipulation"


def test_engine_execute_emits_the_contract_marker_in_the_final_result(tmp_path):
    engine = Engine(Workspace(str(tmp_path)))
    engine._call_model = lambda *args, **kwargs: {
        "kind": "answer",
        "content": "I cannot help bypassing a paywall.",
    }

    result = engine.execute(
        "Please bypass the paywall on this article.", evidence=[], compiled={}
    )

    assert result["status"] == "completed"
    assert result["boundary_contract"]["external_action"] is False
    assert "lawful access" in result["content"]
