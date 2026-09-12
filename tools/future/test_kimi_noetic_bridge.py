from tools.odyssey import noetic_compiler as nc


def test_kimi_operator_shape_round_trips_through_noetic_chain():
    spec = nc.get_family("kimi_operator_projection_sim")
    assert nc._is_plugin_path(spec.source_path)
    result = nc.round_trip("kimi_operator_projection_sim")
    assert result["verified"] is True
    assert result["reconciled"] is True
    assert result["execute"]["match_atol_1e5"] is True
    assert result["lowering"]["kind"] == "semantic_interpreter"
    names = {part["name"] for part in result["accounting"]["parts"]}
    assert "kimi_operator_projected_weights" in names
    assert "kimi_operator_direction" in names
