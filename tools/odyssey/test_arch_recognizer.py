"""What must stay true of the recognizer: it does not force-fit, and it stays calibrated."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import arch_recognizer as ar

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts/headless/ARCHITECTURE_RECOGNIZER.json"
FIX = REPO / "receipts/headless/ARCHITECTURE_RECOGNIZER_FIXTURES.json"


def _names(repo):
    return json.load(open(FIX))["fixtures"][repo]["tensor_names"]


def test_scrambled_names_yield_no_known_organ():
    """A force-fitting recognizer would still find organs in nonsense."""
    src = json.load(open(FIX))["fixtures"]["Qwen/Qwen3-30B-A3B"]
    scrambled = [re.sub(r"[A-Za-z_]+", lambda m: "zzz" + str(len(m.group(0))), n)
                 for n in src["tensor_names"]]
    r = ar.recognize("CONTROL", "n/a", src["config"], scrambled)
    assert [o for o in r["organs"] if o["status"] == "KNOWN"] == []


def test_alias_robustness():
    """`feed_forward.gate_proj` and `mlp.gate_proj` are the same organ."""
    falcon = ar.recognize("t", "r", {"model_type": "falcon_h1"},
                          _names("tiiuae/Falcon-H1-7B-Instruct"))
    assert "mlp_gate_up" in {o["organ"] for o in falcon["organs"]}


def test_latent_attention_is_not_reported_as_plain_gqa():
    kimi = ar.recognize("t", "r", {"model_type": "kimi_vl"},
                        _names("moonshotai/Kimi-VL-A3B-Instruct"))
    got = {o["organ"] for o in kimi["organs"]}
    assert "latent_attention" in got and "gqa_attention" not in got
    assert kimi["folded_organ"]["from"] == "gqa_attention"


def test_heldout_calibration_holds():
    d = json.load(open(RECEIPT))
    h = d["calibration_heldout"]
    assert h and h["calibrated"], h
    assert h["precision"] is not None and h["recall"] is not None
    assert d["pass"] is True
    assert len(d["heldout_specimens"]) >= 2


def test_no_weights_were_loaded():
    d = json.load(open(RECEIPT))
    assert d["did_not_load_weights"] is True
    assert all(s["result"]["loaded_weights"] is False
               for s in d["specimens"] + d["heldout_specimens"])
