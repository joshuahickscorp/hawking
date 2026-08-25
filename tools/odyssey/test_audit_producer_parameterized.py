"""G023 chain step-1 pins: the audit producer must be able to audit model #2."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
pytest.importorskip("lab.operators.ascension_qwen30_physical_campaign")
from lab.operators import ascension_qwen30_physical_campaign as apc  # noqa: E402

M2 = "Qwen/Qwen3-30B-A3B"
M2_REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def test_defaults_are_unchanged():
    """Parameterizing must not alter what the operator did before."""
    c = apc.Qwen30PhysicalCampaign(model_dir=Path("/tmp"), root=Path("/tmp"))
    assert c.source_repository == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert c.source_revision == "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
    assert c.source_audit_path.name == "QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json"
    assert c.status_path.name == "QWEN30_REAL_CAMPAIGN_STATUS.json"


def test_the_old_module_constants_still_resolve():
    """Anything importing the old names keeps working."""
    assert apc.SOURCE_REPOSITORY == apc.DEFAULT_SOURCE_REPOSITORY
    assert apc.SOURCE_REVISION == apc.DEFAULT_SOURCE_REVISION


def test_it_can_now_be_pointed_at_model_2():
    c = apc.Qwen30PhysicalCampaign(
        model_dir=Path("/tmp"), root=Path("/tmp"),
        source_repository=M2, source_revision=M2_REV, audit_prefix="QWEN30BASE")
    assert c.source_repository == M2
    assert c.source_revision == M2_REV


def test_a_second_model_cannot_overwrite_the_first_audit():
    """The audit filename was fixed, so two models would collide in one root."""
    a = apc.Qwen30PhysicalCampaign(model_dir=Path("/tmp"), root=Path("/tmp"))
    b = apc.Qwen30PhysicalCampaign(model_dir=Path("/tmp"), root=Path("/tmp"),
                                   audit_prefix="QWEN30BASE")
    assert a.source_audit_path != b.source_audit_path
    assert a.status_path != b.status_path


def test_no_read_site_still_uses_the_module_constant():
    """Three sites read SOURCE_REPOSITORY/SOURCE_REVISION directly, which is why the
    operator could only ever audit one model."""
    src = (REPO / "lab/operators/ascension_qwen30_physical_campaign.py").read_text()
    body = src[src.index("class Qwen30PhysicalCampaign"):]
    assert '"repository": SOURCE_REPOSITORY' not in body
    assert '"revision": SOURCE_REVISION' not in body
    assert body.count('"repository": self.source_repository') == 3


def test_the_cli_exposes_the_new_arguments():
    p = apc.build_parser()
    ns = p.parse_args(["--source-repository", M2, "--source-revision", M2_REV,
                       "--audit-prefix", "QWEN30BASE", "once"])
    assert ns.source_repository == M2
    assert ns.audit_prefix == "QWEN30BASE"
