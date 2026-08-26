"""Mutation tests for the HCLI resident seal.

Same law as every validator in this program: one that nobody has watched REFUSE
is indistinguishable from one that always accepts. Every test here breaks the
seal on purpose and requires a refusal, and the first one is the anti-vacuity
guard without which all the others would pass on a validator that refused
everything.
"""
import copy, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import resident_seal as S


def good():
    return S.build()


def test_the_real_seal_is_defensible():
    """ANTI-VACUITY. Without this, a validator that refused everything would make
    every other test in this file pass."""
    assert S.validate(good())["status"] == "SEALED"


def test_every_field_is_SOURCED_and_a_typed_one_is_REFUSED():
    """The defect this exists against is not a missing field -- that is visible --
    it is a field somebody typed. civilization/build_state.py already carried the
    literal '4 hf download workers' and was wrong the moment the fill changed."""
    s = good(); s["fields"]["complete_ebpw"] = 3.1393     # a bare value, no source
    try:
        S.validate(s)
    except S.Refused as e:
        assert "no source" in str(e); return
    raise AssertionError("a typed field survived")


def test_a_field_citing_a_receipt_that_does_not_exist_is_REFUSED():
    s = good()
    s["fields"]["capability_score"]["source"] = "receipts/headless/NO_SUCH_RECEIPT.json#overall"
    try:
        S.validate(s)
    except S.Refused as e:
        assert "does not" in str(e); return
    raise AssertionError("a dangling citation survived")


def test_a_field_citing_a_receipt_KEY_that_does_not_exist_is_REFUSED():
    """Weaker to catch than a missing file and more likely: the receipt is real and
    the pointer has rotted."""
    s = good()
    s["fields"]["capability_score"]["source"] = (
        "receipts/headless/CAPABILITY_sealed-3.14-binB-fused4-NOTHINK.json#no_such_key")
    try:
        S.validate(s)
    except S.Refused:
        return
    raise AssertionError("a rotted json pointer survived")


def test_a_capability_score_WITHOUT_ITS_ARM_is_REFUSED():
    """The whole reason this seal exists. THE SAME ARTIFACT scores 30/43 and 35/43
    on two chat-template arms, so a score without its arm names nothing -- and five
    prior receipts recorded exactly that."""
    s = good(); s["fields"]["chat_template_arm"]["value"] = None
    try:
        S.validate(s)
    except S.Refused as e:
        assert "arm" in str(e); return
    raise AssertionError("an armless capability score survived")


def test_a_SPEED_CLAIM_with_UNKNOWN_bench_state_is_REFUSED_unless_PROVISIONAL():
    """S032 §3: a speed claim without auditable machine state is provisional.
    UNKNOWN IS NOT QUIET."""
    s = good(); s["fields"]["bench_state"]["value"] = "UNKNOWN"
    try:
        S.validate(s)
    except S.Refused as e:
        assert "UNKNOWN" in str(e) and "PROVISIONAL" in str(e)
    else:
        raise AssertionError("a speed claim on an unknown machine survived as SEALED")
    # and the escape hatch works, so the rule is not simply a ban
    s["status"] = "PROVISIONAL"
    assert S.validate(s)["status"] == "PROVISIONAL"


def test_an_invented_bench_state_is_REFUSED():
    s = good(); s["fields"]["bench_state"]["value"] = "probably fine"
    try:
        S.validate(s)
    except S.Refused as e:
        assert "bench_state" in str(e); return
    raise AssertionError("an invented bench state survived")


def test_a_NONZERO_FALLBACK_COUNT_is_REFUSED():
    """A resident that falls back to a dense path is not the resident that was
    measured. There is no threshold here on purpose."""
    s = good(); s["fields"]["fallbacks"]["value"] = 1
    try:
        S.validate(s)
    except S.Refused as e:
        assert "fallback" in str(e); return
    raise AssertionError("a falling-back resident sealed")


def test_dropping_ANY_required_field_is_REFUSED():
    """All 23, not a sample -- a seal list that is only spot-checked is one whose
    holes are wherever nobody looked."""
    for k in S.REQUIRED:
        s = good(); s["fields"].pop(k)
        try:
            S.validate(s)
        except S.Refused as e:
            assert k in str(e), (k, str(e))
        else:
            raise AssertionError(f"a seal missing {k!r} survived")


def test_ABSENT_needs_a_reason_and_is_otherwise_ACCEPTED():
    """ABSENT-with-a-reason is legal -- it is how the accelerator receipt schema
    records an identity that does not apply. ABSENT without one is not."""
    s = good(); s["fields"]["runtime_commit"] = {"value": None, "source": "ABSENT"}
    try:
        S.validate(s)
    except S.Refused as e:
        assert "reason" in str(e)
    else:
        raise AssertionError("a reasonless ABSENT survived")
    s["fields"]["runtime_commit"] = S.absent("not a git checkout")
    assert S.validate(s)["status"] == "SEALED"


def test_the_seal_records_BOTH_ARMS_not_only_the_flattering_one():
    """A seal that silently re-binds to the better configuration is unauditable
    against the five prior receipts that used the other one."""
    cap = good()["fields"]["capability_score"]["value"]
    assert cap["best_arm"]["passed"] == 35 and cap["best_arm"]["arm"] == "pre_closed_think"
    assert cap["arm_previous_receipts_used"]["passed"] == 30
    assert cap["arm_previous_receipts_used"]["arm"] == "open_think"


def test_the_written_seal_on_disk_still_validates():
    """The file anyone reads is the file that must hold, not the object build()
    happens to return in this process."""
    p = S.RH / "HCLI_RESIDENT_SEAL.json"
    if not p.is_file():
        return
    assert S.validate(json.loads(p.read_text()))["resident"] == "sealed-3.14"
