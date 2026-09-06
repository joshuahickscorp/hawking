from tools.future.gravity_outlier_eval import down_proj_scale_overhead_bpw


def test_down_proj_scale_overhead_bpw_128():
    assert down_proj_scale_overhead_bpw(128) == 0.041666666666666664


def test_down_proj_scale_overhead_bpw_64():
    assert down_proj_scale_overhead_bpw(64) == 0.08333333333333333
