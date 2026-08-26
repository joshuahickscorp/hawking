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


# --- harness health: a crash is not a score --------------------------------

def _err(n, kind="ModuleNotFoundError"):
    return [{"text": "", "finish_reason": f"ERROR:{kind}: No module named 'transformers'"}
            for _ in range(n)]


def test_an_all_error_run_is_REFUSED_not_scored_zero():
    """The real event: 43 calls raised under python3.14 and the receipt said
    0/43 rate 0.0 -- identical in a summary line to the 2.60 body's genuine
    0/43. A score of zero is a claim about a model; a suite that never reached
    one has no claim to make."""
    h = C.harness_health(_err(43))
    assert h["every_call_errored"] and not h["scoreable"]
    assert "HARNESS failure" in h["verdict"]
    assert h["error_kinds"] == ["ModuleNotFoundError"]


def test_a_REAL_ZERO_is_still_scoreable():
    """Anti-vacuity, and the distinction that matters: the 2.60 body under the
    open-<think> arm really does score 0/43 while ANSWERING every time. That
    must stay a score, or the refusal has eaten a real finding."""
    real = [{"text": "<think>\nreasoning that never ends", "finish_reason": "length"}
            for _ in range(43)]
    h = C.harness_health(real)
    assert h["scoreable"] and not h["every_call_errored"]
    assert h["empty_replies"] == 0


def test_a_PARTIAL_failure_still_scores_and_is_counted():
    """One flaky call must not void 42 real ones -- but the count is reported
    so a reader can see the run was not clean."""
    h = C.harness_health(_err(3) + [{"text": "Paris", "finish_reason": "stop"}] * 40)
    assert h["scoreable"] and h["errored"] == 3 and h["calls"] == 43


def test_empty_replies_are_counted_SEPARATELY_from_errors():
    """An empty reply with a clean finish_reason is the model producing nothing,
    which IS a capability fact; an empty reply after a raise is not."""
    h = C.harness_health([{"text": "", "finish_reason": "stop"}] * 5)
    assert h["scoreable"] and h["errored"] == 0 and h["empty_replies"] == 5


# --------------------------------------------------------------- machine_state
# Two unpaired capability runs of the SAME artifact differed by 19% of wall and it
# took reading per-repetition spread, by hand, to establish that one arm had shared
# the machine (ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json). These pin the field that
# now records it, so the next reader does not have to infer it.

def _responses(spreads):
    """One item per entry; each maps id -> the wall times of its repetitions."""
    return [{"id": k, "rep": i, "wall_s": w}
            for k, ws in spreads.items() for i, w in enumerate(ws)]


def test_repetition_spread_separates_a_contended_run_from_a_quiet_one():
    """The signal itself. A graph or model change cannot make a kernel more
    REPEATABLE, so scatter across repetitions is the machine, not the model."""
    C.machine_state._before = {"quiet": True, "max_rss_gib": 0.1}
    contended = C.machine_state(_responses({"a": [5.45, 5.45, 7.88], "b": [15.0, 17.9, 12.3]}))
    quiet = C.machine_state(_responses({"a": [5.22, 5.24, 5.21], "b": [11.55, 11.53, 11.52]}))
    assert contended["worst_repetition_spread_pct"] > 30, contended
    assert quiet["worst_repetition_spread_pct"] < 2, quiet
    # and the worst offender is NAMED, not just counted
    assert next(iter(contended["repetition_spread_pct_by_item"])) == "a"


def test_a_missing_BEFORE_sample_is_NOT_a_recorded_machine_state():
    """'I could not look' must not read as 'I looked and found nothing'. Returning
    recorded=True with a null quiescence field is exactly that."""
    C.machine_state._before = None
    out = C.machine_state(_responses({"a": [1.0, 1.0]}))
    assert out["recorded"] is False and "before" in out["why"], out


def test_the_field_is_recorded_when_a_sample_EXISTS():
    """Anti-vacuity. A machine_state that always refused would pass the test above
    and record nothing forever."""
    C.machine_state._before = {"quiet": False, "max_rss_gib": 33.7}
    out = C.machine_state(_responses({"a": [1.0, 1.02]}))
    assert out["recorded"] is True
    assert out["quiescence_before"]["max_rss_gib"] == 33.7
    assert out["quiet_at_both_samples"] is False, "before was not quiet"


def test_a_single_repetition_yields_NO_spread_rather_than_a_fake_zero():
    """One sample has no spread. Reporting 0.0% would read as a perfectly quiet
    machine, which is the flattering direction."""
    C.machine_state._before = {"quiet": True}
    out = C.machine_state(_responses({"a": [1.0]}))
    assert out["worst_repetition_spread_pct"] is None, out


# ------------------------------------------------- measurement weight / vacuity
# 43 cases are 11 items. A five-point move is ONE item weighted five, and reading
# it as five independent successes is the error these pin against.

def test_43_cases_are_11_items_and_the_field_says_so():
    import json
    a = C.REPO / "receipts/headless/CAPABILITY_sealed-3.14-binB-fused4-swiglu.json"
    b = C.REPO / "receipts/headless/CAPABILITY_sealed-3.14-binB-fused4-NOTHINK.json"
    if not (a.is_file() and b.is_file()):
        return
    for p in (a, b):
        m = C.measurement_weight(json.loads(p.read_text())["per_item"])
        assert m["cases"] == 43 and m["distinct_items"] == 11, m
        assert m["every_repeat_identical"] is True, (
            "greedy decode at temperature 0 should make every repeat identical; "
            f"if this fails the suite has found real nondeterminism: {m}")


def test_the_weight_field_can_report_NONDETERMINISM():
    """Anti-vacuity. every_repeat_identical == True is only a finding if the field
    can come back False."""
    per_item = {"x": {"repeats": 2, "results": [
        {"completion_tokens": 4, "reply_head": "a", "pass": True},
        {"completion_tokens": 4, "reply_head": "b", "pass": False}]}}
    m = C.measurement_weight(per_item)
    assert m["every_repeat_identical"] is False, m
    assert m["distinct_outputs_per_item"]["x"] == 2


def test_the_no_think_leak_check_is_MOSTLY_VACUOUS_and_that_is_recorded():
    """`no-think-leak` (capability_suite.py:228-231) forbids "<think>", "</think>"
    and "reasoning_content". But call_noetic:305 SPLITS the raw output on
    "</think>" and scores only what follows, so both of the first two needles are
    removed by the harness before the predicate ever sees the text, and the third
    is an API field name a CLI backend never emits. It scores 3/3 on BOTH template
    arms.

    NOT fully vacuous, and the difference is load-bearing: :304's `unterminated`
    guard makes a reply that never leaves the think block score as EMPTY, and
    must_not_contain refuses empty. That guard exists because the 2.5970-EBPW body
    once passed this exact check BY never finishing thinking -- the comment at
    :299-303 says so. What remains vacuous is the leak check itself.

    Kept rather than deleted: it is 3 of 43 on BOTH arms, so it is common-mode and
    cannot touch the arm comparison -- but both absolute scores carry it, and that
    is visible here rather than counted silently as capability."""
    split = lambda raw: (raw.split("</think>", 1)[1] if "</think>" in raw else raw).strip()
    leaked = split("<think>\nreasoning\n</think>\nok")
    assert "<think>" not in leaked and "</think>" not in leaked, leaked
    ok, why = C.must_not_contain("<think>", "</think>", "reasoning_content")(leaked, {})
    assert ok, f"the harness strips the needles it checks for; only empty can fail: {why}"

    # the half that is NOT vacuous, and the reason the guard was added
    empty_ok, empty_why = C.must_not_contain("<think>", "</think>")("", {})
    assert not empty_ok and "empty" in empty_why.lower(), (empty_ok, empty_why)

    # and it CAN still fail on a real leak after the split
    bad_ok, _ = C.must_not_contain("<think>", "</think>")("ok <think> oops", {})
    assert not bad_ok, "a think block AFTER </think> must still be caught"


def test_machine_state_carries_the_canonical_bench_block():
    """S032 §3: a capability receipt and an accelerator receipt must state the
    machine the same way, or a reader has to learn two vocabularies for one fact."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "accelerator"))
    import bench
    import capability_suite as CS

    loud = {"quiet": False, "method": "enumerate", "contenders": [{"comm": "x"}],
            "n_contenders": 1, "max_rss_gib": 39.0}
    CS.machine_state._before = loud
    st = CS.machine_state([{"id": "a", "wall_s": 1.0}, {"id": "a", "wall_s": 1.4}])
    assert st["recorded"] is True
    assert st["bench"]["state"] == "CONTENDED", st["bench"]
    assert st["bench"]["machine"] and st["bench"]["recorded_at"]


def test_machine_state_cannot_report_QUIESCED_without_a_before_sample():
    import capability_suite as CS
    CS.machine_state._before = None
    st = CS.machine_state([])
    assert st["recorded"] is False and "before" in st["why"]
