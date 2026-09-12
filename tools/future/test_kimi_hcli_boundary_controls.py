from tools.future.kimi_hcli_boundary_controls import run


def test_independent_hcli_boundary_controls_pass_without_external_actions():
    result = run()
    assert result["status"] == "CONTROLS_PASSED"
    assert result["summary"]["all_passed"] is True
    assert result["summary"]["authority_expanding_operations_withheld"] is True
    assert result["summary"]["destructive_mutation_denied"] is True
    assert result["authority"]["external_writes"] is False
