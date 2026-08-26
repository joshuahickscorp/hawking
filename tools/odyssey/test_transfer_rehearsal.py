"""A head start is only transfer if the inputs prove it was not smuggled."""
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import transfer_rehearsal as trh

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts/headless/QWEN_TRANSFER_REHEARSAL.json"


def test_audit_catches_a_deliberate_smuggle():
    tool = str(Path(__file__).parent / "transfer_rehearsal.py")
    p = subprocess.run([sys.executable, tool, "--emit", "/dev/null", "--smuggle-demo"],
                       capture_output=True, text=True)
    out = json.loads(p.stdout)
    assert out["clean"] is False and out["n_forbidden_reads"] >= 1
    assert any("noetic" in f for f in out["forbidden_reads"])


def test_the_real_rehearsal_read_nothing_forbidden():
    a = json.load(open(R))["input_audit"]
    assert a["clean"] is True
    assert a["n_forbidden_reads"] == 0
    assert a["n_repo_reads_outside_allowlist"] == 0
    assert a["n_opens_recorded"] > 0, "an audit that recorded no opens proves nothing"


def test_forbidden_prefixes_cover_qwen_private_state():
    a = json.load(open(R))["input_audit"]
    joined = " ".join(a["forbidden_prefixes"])
    for must in ("noetic", "models", "workspace", "artifacts"):
        assert must in joined, must


def test_plan_seeds_organs_and_reuses_kernels():
    p = json.load(open(R))["plan"]
    assert p["n_organs_recognized"] >= 5
    for o in p["organ_plan"]:
        assert o["seeded_representations"], o["organ"]
        assert o["reusable_kernels"], o["organ"]


def test_prior_failures_actually_moved_the_search():
    p = json.load(open(R))["plan"]
    assert p["n_prior_failures_applied"] > 0
    demoted = [a for a in p["prior_failures_that_shape_the_search"]
               if a["effect"].startswith("demoted")]
    assert demoted, "no branch was moved by prior science"


def test_roofs_are_not_copied_into_the_new_specimen():
    d = json.load(open(R))["plan"]["device_genome_init"]
    assert "NOT copied" in d["seeded_from"]
    assert "690.8" not in json.dumps(d)
