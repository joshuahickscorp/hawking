from tools.future.kimi_epistemic_live import TASKS, score_reply, summarize


def test_known_and_unknown_confidence_contracts_score():
    rows = []
    for task in TASKS:
        if task["answerable"]:
            reply = json_reply(task["answer"], 0.9, False)
        else:
            reply = json_reply(None, 0.1, True)
        rows.append(score_reply(task, reply))
    summary = summarize(rows)
    assert summary["all_passed"] is True
    assert summary["qualified"] is True
    assert summary["brier_score"] <= 0.15


def test_overconfident_unknown_is_not_qualified():
    rows = []
    for task in TASKS:
        if task["answerable"]:
            reply = json_reply(task["answer"], 0.9, False)
        else:
            reply = json_reply(None, 0.95, False)
        rows.append(score_reply(task, reply))
    summary = summarize(rows)
    assert summary["unknown_passed"] == 0
    assert summary["qualified"] is False


def json_reply(answer, confidence, abstain):
    import json

    return json.dumps({
        "answer": answer,
        "confidence": confidence,
        "abstain": abstain,
    })
