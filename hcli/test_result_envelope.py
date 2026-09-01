from hcli.result_envelope import build_result_envelope


def test_model_prose_stays_a_hypothesis_without_deterministic_validation():
    envelope = build_result_envelope(
        goal="prove the candidate is faster",
        result={"status": "completed", "content": "the candidate wins"},
    )
    assert envelope["verdict"] == "UNVERIFIED"
    assert envelope["hypotheses"]
    assert envelope["verified_facts"] == []


def test_independent_validation_promotes_only_the_checked_result():
    envelope = build_result_envelope(
        goal="run the fixed verifier",
        result={"status": "completed", "content": "verified output"},
        validation={"ok": True, "checks": ["sha256 matches"]},
    )
    assert envelope["verdict"] == "ACCEPT"
    assert envelope["verified_facts"]
    assert envelope["hypotheses"] == []
    assert envelope["tests"]["checks"] == ["sha256 matches"]


def test_failed_validation_blocks_even_confident_model_text():
    envelope = build_result_envelope(
        goal="promote a resident",
        result={"status": "completed", "content": "promotion is safe"},
        validation={"ok": False, "reason": "protected receipt missing"},
    )
    assert envelope["verdict"] == "BLOCKED"
    assert envelope["verified_facts"] == []
    assert envelope["hypotheses"]
    assert envelope["failures"][0]["reason"] == "protected receipt missing"
