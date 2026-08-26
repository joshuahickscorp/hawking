"""G073: a capability receipt must name the body it graded.

Every CAPABILITY_noetic-*.json on disk records target="http://127.0.0.1:8080"
and a hand-typed label -- the llama endpoint DEFAULT, for bodies loaded from
--artifact-root. The leaderboard was a table of labels.

Each test below breaks the identity on purpose and requires a refusal, plus the
anti-vacuity partner: a check that refuses everything is not a check.
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import capability_suite as C


def _noetic(tmp, no_think=True):
    root = tmp / "body"; root.mkdir(parents=True, exist_ok=True)
    (root / "model-00001.hq30uq4").write_bytes(b"\x00" * 4096)
    (root / "tokenizer.json").write_text('{"v":1}')
    (root / "chat_template.jinja").write_text("{{ x }}")
    binp = tmp / "greedy"; binp.write_bytes(b"\x7fELF-ish")
    return argparse.Namespace(
        backend="noetic", artifact_root=str(root), noetic_binary=str(binp),
        tokenizer_dir=None, model_path=None, endpoint="http://127.0.0.1:8080",
        no_think=no_think, default_system=None)


def test_a_noetic_run_names_its_artifact(tmp_path):
    """Anti-vacuity: without this, a validator refusing EVERYTHING passes the rest."""
    ok, why = C.identity_is_sufficient(C.artifact_identity(_noetic(tmp_path)))
    assert ok, why


def test_a_served_endpoint_is_REFUSED(tmp_path):
    """The exact shape of every noetic receipt on disk: a port, not a body."""
    a = argparse.Namespace(backend="llama", endpoint="http://127.0.0.1:8080",
                           model_path=None, artifact_root=None, no_think=False,
                           default_system=None)
    ok, why = C.identity_is_sufficient(C.artifact_identity(a))
    assert not ok and "does not disclose" in why, why


def test_the_TEMPLATE_ARM_is_recorded(tmp_path):
    """G067 measured the SAME body at 0/43 and 14/43 across these two arms.
    A score whose receipt omits the arm is not comparable to another score."""
    a = C.artifact_identity(_noetic(tmp_path, no_think=True))
    b = C.artifact_identity(_noetic(tmp_path, no_think=False))
    assert a["chat_template_arm"] == "pre_closed_think"
    assert b["chat_template_arm"] == "open_think"
    assert a["chat_template_arm"] != b["chat_template_arm"]


def test_a_DIFFERENT_BODY_gets_a_DIFFERENT_IDENTITY(tmp_path):
    """The inventory hash must actually discriminate, or it is decoration."""
    a = C.artifact_identity(_noetic(tmp_path / "a"))
    b_args = _noetic(tmp_path / "b")
    p = pathlib.Path(b_args.artifact_root) / "model-00001.hq30uq4"
    p.write_bytes(b"\x00" * 8192)                      # different SIZE
    b = C.artifact_identity(b_args)
    assert a["artifact_inventory_sha"] != b["artifact_inventory_sha"]
    assert a["artifact_bytes"] != b["artifact_bytes"]


def test_a_SAME_LENGTH_EDIT_is_INVISIBLE_and_the_receipt_SAYS_SO(tmp_path):
    """The honest limit, pinned. Sizes are not a content hash and the field
    that says so must stay true, or a reader will over-trust the identity."""
    a_args = _noetic(tmp_path / "a")
    a = C.artifact_identity(a_args)
    p = pathlib.Path(a_args.artifact_root) / "model-00001.hq30uq4"
    p.write_bytes(b"\xff" * 4096)                      # SAME size, different bytes
    b = C.artifact_identity(a_args)
    assert a["artifact_inventory_sha"] == b["artifact_inventory_sha"], "expected blind"
    assert a["identity_is_content_hash"] is False
    assert "same-length edit is invisible" in a["identity_note"]


def test_the_binary_and_template_ARE_content_hashed(tmp_path):
    """Small files are cheap, so these are real hashes and must discriminate."""
    a_args = _noetic(tmp_path / "a")
    a = C.artifact_identity(a_args)
    (pathlib.Path(a_args.artifact_root) / "chat_template.jinja").write_text("{{ y }}")
    (pathlib.Path(a_args.noetic_binary)).write_bytes(b"\x7fELF-OTHER")
    b = C.artifact_identity(a_args)
    assert a["chat_template_sha256_16"] != b["chat_template_sha256_16"]
    assert a["binary_sha256_16"] != b["binary_sha256_16"]
