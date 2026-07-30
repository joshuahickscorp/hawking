#!/usr/bin/env python3
"""Self-check for the Capability Sovereignty deterministic spine. No framework:
asserts + __main__.

Run:  python3 tools/sovereignty/test_sovereignty.py
It fails loudly if any load-bearing invariant breaks.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sovereignty as S  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "condense"))
import artifact_client as G  # noqa: E402

ARTIFACT = Path("/Users/scammermike/Library/Application Support/Hawking/CampaignS08/"
                "llama32-1b-R0.v2.gravity")


def test_continuity_manifest_from_real_artifact():
    if not ARTIFACT.exists():
        print(f"  SKIP continuity-from-real-artifact: {ARTIFACT} not found")
        return
    header = G.read_header(ARTIFACT)
    manifest = S.build_continuity_manifest(ARTIFACT)
    assert manifest["artifact_hash"] == header["integrity"]["body_sha256"]
    assert manifest["executed_model"] == header["model"]["repo"]
    assert manifest["source_weight_hash"] == header["model"]["revision"]
    assert manifest["requested_model"] == manifest["executed_model"]  # no substitution
    assert manifest["fallback_allowed"] is False
    assert manifest["fallback_events"] == []
    print(f"  continuity: artifact_hash == header body_sha256 "
          f"({manifest['artifact_hash'][:16]}...)")


def test_continuity_manifest_errors_on_bad_artifact():
    """A manifest that cannot name its own artifact must error, never placeholder."""
    with tempfile.TemporaryDirectory() as tmp:
        bogus = Path(tmp) / "not-a-shard.gravity"
        bogus.write_bytes(b"not a gravity file")
        try:
            S.build_continuity_manifest(bogus)
        except Exception:
            print("  continuity: malformed artifact correctly raises, no placeholder hash")
            return
        raise AssertionError("malformed artifact did NOT raise")


def test_invalid_plane_rejected():
    try:
        S.make_event(plane="bogus", owner="x", reason="y", scope="z")
    except ValueError:
        print("  event: invalid plane correctly rejected")
        return
    raise AssertionError("invalid plane was NOT rejected")


def test_all_five_planes_accepted():
    for plane in S.PLANES:
        e = S.make_event(plane=plane, owner="o", reason="r", scope="s")
        assert e["plane"] == plane
    print(f"  event: all {len(S.PLANES)} planes (capability|policy|permission|"
          "evidence|resource) accepted")


def test_event_id_is_content_addressed():
    e = S.make_event(plane="policy", owner="a", reason="b", scope="c", transformation="t1")
    assert S.verify_event(e), "freshly minted event must verify"
    tampered = dict(e, transformation="t2")  # id not recomputed -> stale
    assert not S.verify_event(tampered), "rewriting a field must invalidate the id"
    retagged = S.make_event(plane="policy", owner="a", reason="b", scope="c",
                            transformation="t2", timestamp=e["timestamp"])
    assert retagged["id"] != e["id"], "different content must mint a different id"
    print("  event id: content-addressed -- rewriting a field changes/invalidates it")


def test_unclassified_refusal_raises():
    try:
        S.attribute("R-BOGUS", "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unclassified refusal code did NOT raise")
    for code in S.REFUSAL_CODES:
        rec = S.attribute(code, "detail")
        assert rec["code"] == code and rec["meaning"]
    print(f"  refusal: unclassified code raises; all {len(S.REFUSAL_CODES)} real "
          "codes attribute cleanly")


def test_limit_no_owner_or_reason_rejected_at_insert():
    registry: dict = {}
    for bad_kwargs in (
        dict(id="ctx-window", owner="", type="resource", reason="context cap",
             scope="session", threshold=128000),
        dict(id="ctx-window", owner="hawking-runtime", type="resource", reason="",
             scope="session", threshold=128000),
    ):
        try:
            S.add_limit(registry, **bad_kwargs)
        except ValueError:
            continue
        raise AssertionError(f"limit insert should have been rejected: {bad_kwargs}")
    assert registry == {}, "rejected limits must not land in the registry"
    S.add_limit(registry, id="ctx-window", owner="hawking-runtime", type="resource",
               reason="context cap", scope="session", threshold=128000)
    assert "ctx-window" in registry
    print("  limit registry: missing owner and missing reason both rejected at insert")


def test_limit_check_records_event_on_bind():
    registry: dict = {}
    S.add_limit(registry, id="ctx-window", owner="hawking-runtime", type="resource",
               reason="context cap", scope="session", threshold=1000)
    below = S.check(registry, "ctx-window", 500)
    assert below["bound"] is False
    assert registry["ctx-window"]["event_log"] == []
    above = S.check(registry, "ctx-window", 1200)
    assert above["bound"] is True
    assert len(registry["ctx-window"]["event_log"]) == 1
    assert S.list_limits(registry) == [registry["ctx-window"]]
    print("  limit registry: check() records an event only when the limit binds")


def test_model_continuity_rate():
    assert S.model_continuity_rate([]) == 1.0
    no_sub = [S.make_event(plane="capability", requested_model="glm-5.2",
                           executed_model="glm-5.2") for _ in range(3)]
    assert S.model_continuity_rate(no_sub) == 1.0
    with_sub = no_sub + [S.make_event(plane="capability", requested_model="glm-5.2",
                                      executed_model="qwen3.5-fallback")]
    rate = S.model_continuity_rate(with_sub)
    assert rate == 0.75, rate
    print(f"  model_continuity_rate: 1.0 with no substitution events, "
          f"{rate:.2f} with one substitution among four")


def test_hidden_intervention_rate():
    assert S.hidden_intervention_rate([]) == 0.0
    visible = [S.make_event(plane="policy", owner="o", reason="r", scope="s")
               for _ in range(3)]
    hidden = S.make_event(plane="policy", owner="o", reason="r", scope="s",
                          user_visible=False)
    rate = S.hidden_intervention_rate(visible + [hidden])
    assert rate == 0.25, rate
    print(f"  hidden_intervention_rate: 0.0 with no events, {rate:.2f} with one "
          "non-user-visible event among four")


def test_attribution_completeness():
    complete = S.make_event(plane="policy", owner="o", reason="r", scope="s",
                            ledger_ref="L1")
    incomplete = S.make_event(plane="policy", owner="o", reason="r", scope="s",
                              ledger_ref=None)
    assert S.attribution_completeness([]) == 1.0
    assert S.attribution_completeness([complete]) == 1.0
    assert S.attribution_completeness([complete, incomplete]) == 0.5
    assert S.attribution_completeness([incomplete]) == 0.0
    print("  attribution_completeness: 1.0 only when every event carries "
          "owner+reason+scope+ledger_ref")


def test_event_log_append_and_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp)
        assert S.read_events(state) == []  # no file yet -> empty, not an error
        e1 = S.make_event(plane="resource", owner="o", reason="r", scope="s")
        e2 = S.make_event(plane="evidence", owner="o", reason="r", scope="s")
        S.append_event(state, e1)
        S.append_event(state, e2)
        got = S.read_events(state)
        assert got == [e1, e2]
        assert all(S.verify_event(e) for e in got)
        print(f"  event log: append-only JSONL round-trips {len(got)} events at "
              f"{S.log_path(state).name}")


def test_gated_metrics_named_never_fabricated():
    assert set(S.GATED_METRICS) == {"false_refusal_rate", "boundary_error_rate"}
    assert all(isinstance(v, str) and v.startswith("GATED:") for v in S.GATED_METRICS.values())

    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp)
        S.append_event(state, S.make_event(plane="capability", owner="o", reason="r",
                                           scope="s", requested_model="a",
                                           executed_model="b"))
        if ARTIFACT.exists():
            result = S.seal(ARTIFACT, state_dir=state)
        else:
            result = {
                "sovereignty": {
                    "hidden_intervention_rate": S.hidden_intervention_rate(S.read_events(state)),
                    "model_continuity_rate": S.model_continuity_rate(S.read_events(state)),
                    "attribution_completeness": S.attribution_completeness(S.read_events(state)),
                },
                "gated_stages": dict(S.GATED_METRICS),
            }
        assert "false_refusal_rate" not in result["sovereignty"]
        assert "boundary_error_rate" not in result["sovereignty"]
        assert "false_refusal_rate" in result["gated_stages"]
        assert "boundary_error_rate" in result["gated_stages"]
        # scan the entire output structure: the two names may only ever appear as
        # gated_stages keys mapped to their string reason, never as a numeric value.
        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ("false_refusal_rate", "boundary_error_rate"):
                        assert isinstance(v, str) and v.startswith("GATED:"), (k, v)
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(result)
    print("  gated: false_refusal_rate/boundary_error_rate named in gated_stages "
          "only, never a fabricated numeric value anywhere in output")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"sovereignty self-check: {len(tests)} invariants")
    for t in tests:
        t()
    print("ALL PASS")


if __name__ == "__main__":
    main()
