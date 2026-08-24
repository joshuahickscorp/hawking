"""The competence screen must separate the two kernels that differ only in the defect.

`affine2_group32_matvec_geo_tpr64_tg128` and its `_runtime_div` twin are the same
kernel; the twin was deliberately kept as the slow A/B arm. Measured, the
runtime-divide body was 1.37x the specialized one, and specializing moved the
unfused decode 26.84 -> 32.84 tok/s. So the screen has a real positive and
negative control sitting in the repo, and it must get both right.

Watched failing first, three times:
  v1 matched any `/` followed by a name and flagged 238 of 565 kernels on tokens
     like `float`, `sum` and `rms` -- floating-point division, not the defect.
  v2 restricted to declared integers but still flagged `kSplit`, a
     `constexpr uint kSplit = 2u`, which the compiler turns into a shift.
  v3 still flagged the SPECIALIZED geo kernel, which dispatches on
     `if (group_size == 32u)` into literal arms -- the very fix the law is
     built on. A screen that condemns its own anchor is worse than no screen.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "headless" / "KERNEL_COMPETENCE.json"


def _receipt():
    if not RECEIPT.is_file():
        subprocess.run([sys.executable, str(REPO / "tools/headless/kernel_competence.py")],
                       cwd=REPO, capture_output=True, timeout=300)
    return json.loads(RECEIPT.read_text())


def _verdict(d, name):
    for f in d["per_file"]:
        for k in f["kernels"]:
            if k["kernel"] == name:
                return k["verdict"]
    return None


def test_the_slow_arm_is_flagged():
    v = _verdict(_receipt(), "affine2_group32_matvec_geo_tpr64_tg128_runtime_div")
    assert v == "DEFECTIVE", f"the measured-1.37x runtime-divide arm must be flagged, got {v}"


def test_the_specialized_arm_is_not_flagged():
    v = _verdict(_receipt(), "affine2_group32_matvec_geo_tpr64_tg128")
    assert v != "DEFECTIVE", (
        f"the specialized kernel measured at 32.84 tok/s must NOT be called defective, got {v}. "
        "A screen that condemns its own anchor is worse than no screen."
    )


def test_screen_discriminates_rather_than_flagging_everything():
    c = _receipt()["counts"]["by_verdict"]
    n = _receipt()["counts"]["kernels"]
    assert c.get("DEFECTIVE", 0) < n * 0.25, (
        "flagging a quarter of all kernels is as useless as flagging none"
    )
    assert c.get("CLEAR", 0) > 0, "if nothing is clear the screen cannot clear anything"


def test_law_and_anchor_are_recorded():
    d = _receipt()
    assert "CANNOT BE CONDEMNED" in d["law"]
    a = d["measured_anchor"]
    assert a["runtime_div_vs_specialized"] == 1.37
    assert a["unfused_tok_s_before"] < a["unfused_tok_s_after"]


def test_limits_are_stated():
    lim = _receipt()["what_this_screen_cannot_do"]
    assert len(lim) >= 3
    assert any("necessary" in x.lower() for x in lim), "must say it is not sufficient"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print(f"ok  {n}")
    print("5/5 passed")
