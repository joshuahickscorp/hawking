from tools.future.kimi_workunit_capability_live import TASKS, score_reply


def test_exact_json_scoring_accepts_only_the_expected_object():
    row = score_reply(TASKS[0], '{"operation":"evidence.record","status":"complete","samples":12,"target_met":true}')
    assert row["passed"] is True


def test_json_explanation_is_not_a_machine_checkable_pass():
    row = score_reply(TASKS[0], 'Here is the result: {"operation":"evidence.record"}')
    assert row["passed"] is False
    assert row["parse_error"]


def test_wrong_scope_partition_fails_even_when_json_is_valid():
    row = score_reply(TASKS[2], '{"allowed":["model.inspect","artifact.promote"],"withheld":["external.write"]}')
    assert row["passed"] is False


def test_all_fixture_answers_are_machine_checkable():
    for task in TASKS:
        assert score_reply(task, __import__("json").dumps(task["expected"]))["passed"]
