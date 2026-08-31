"""Pins for tools/future/hmf_objects.py -- overlay, not a second fabric."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.future import hmf_objects as ho
from tools.future._common import RECEIPTS


def _clean(identity: str = "obj.0", *, device: str = "APPLE_DOMAIN_0") -> ho.ManagedObject:
    obj = ho.ManagedObject(identity=ho.HgvasRef(identity))
    ho.establish_clean(
        obj,
        location=ho.Location(ho.MemoryTier.UMA, device),
        visibility={device},
        evidence="test fixture",
        digest="ab" * 16,
        payload=b"payload-bytes",
    )
    return obj


def _humf():
    acc = Path(__file__).resolve().parents[2] / "tools" / "accelerator"
    if str(acc) not in sys.path:
        sys.path.insert(0, str(acc))
    import humf  # type: ignore  # noqa: E402
    return humf


# ------------------------------------------------------------------ receipt


def test_build_emits_sealed_receipt():
    out = ho.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "HMF_MANAGED_OBJECTS.json"
    assert doc["schema"] == "hawking.future.hmf_objects.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert all(v.startswith("PASS") for v in doc["selftest"].values())


def test_selftest_is_build():
    assert ho.selftest() == ho.build()


# ------------------------------------------------------------------ identity


def test_new_object_is_unknown_on_every_axis():
    obj = ho.ManagedObject(identity=ho.HgvasRef("fresh"))
    assert obj.state is ho.ObjectState.UNKNOWN
    assert obj.trust is ho.Trust.UNKNOWN
    assert obj.device_visibility == "UNKNOWN"
    assert obj.location.memory_tier is ho.MemoryTier.UNKNOWN
    assert is_unknown(obj)


def test_identity_fields_are_all_present():
    obj = _clean()
    d = obj.to_dict()
    for key in (
        "identity", "owner", "semantic_representation",
        "physical_materialization", "location", "device_visibility",
        "version", "trust", "state",
    ):
        assert key in d
    assert d["identity"]["object_id"] == "obj.0"
    assert d["location"]["memory_tier"] == "UMA"
    assert d["location"]["device"] == "APPLE_DOMAIN_0"
    assert d["device_visibility"] == ["APPLE_DOMAIN_0"]


def test_hgvas_refuses_native_pointer():
    with pytest.raises(ho.HgvasError, match="native pointer"):
        ho.HgvasRef("0xabc123def")
    with pytest.raises(ho.HgvasError, match="native-pointer residue"):
        ho.HgvasRef("metal_buffer_9")
    ho.HgvasRef("weights.layer0.qkv")  # must not raise


def test_hgvas_refuses_negative_offset():
    with pytest.raises(ho.HgvasError):
        ho.HgvasRef("ok", byte_offset=-1)


# ------------------------------------------------------------------ coherence tri-state


def is_unknown(obj: ho.ManagedObject, **kw) -> bool:
    return ho.is_coherent(obj, **kw) is ho.Coherence.UNKNOWN


def test_is_coherent_is_tri_state_not_bool():
    obj = _clean()
    v = ho.is_coherent(obj)
    assert v is ho.Coherence.COHERENT
    assert type(v) is ho.Coherence
    with pytest.raises(ho.CoherenceBooleanError):
        bool(v)
    with pytest.raises(ho.CoherenceBooleanError):
        _ = v == True  # noqa: E712
    with pytest.raises(ho.CoherenceBooleanError):
        _ = v == False  # noqa: E712


def test_boolean_if_on_is_coherent_always_raises():
    """Accidental `if is_coherent(obj):` must not silently treat UNKNOWN as True
    (str Enum default) or as False. A guard nobody watched fail is not a guard."""
    cases = [
        _clean(),
        ho.ManagedObject(identity=ho.HgvasRef("u")),
        _stale(),
    ]
    for obj in cases:
        with pytest.raises(ho.CoherenceBooleanError):
            if ho.is_coherent(obj):  # noqa: SIM108
                raise AssertionError("bool collapse must not succeed")


def _stale() -> ho.ManagedObject:
    obj = _clean("stale.0")
    ho.invalidate(obj, reason="test invalidate")
    return obj


def test_clean_trusted_is_coherent():
    obj = _clean()
    assert ho.is_coherent(obj) is ho.Coherence.COHERENT
    assert ho.is_coherent(obj, on_device="APPLE_DOMAIN_0") is ho.Coherence.COHERENT


def test_stale_is_not_coherent():
    obj = _stale()
    assert obj.state is ho.ObjectState.STALE
    assert ho.is_coherent(obj) is ho.Coherence.NOT_COHERENT


def test_dirty_is_not_coherent():
    obj = _clean("dirty.0")
    ho.mark_written(obj, device="APPLE_DOMAIN_0")
    assert obj.state is ho.ObjectState.DIRTY
    assert ho.is_coherent(obj) is ho.Coherence.NOT_COHERENT
    assert obj.physical.digest is None
    assert obj.version == 1


def test_unknown_state_is_unknown_coherence():
    obj = ho.ManagedObject(identity=ho.HgvasRef("u.state"))
    assert obj.state is ho.ObjectState.UNKNOWN
    assert ho.is_coherent(obj) is ho.Coherence.UNKNOWN


def test_clean_with_trust_unknown_is_unknown_coherence():
    """THE gap versus HUMF valid_copies(): CLEAN is not knowledge of coherence."""
    obj = _clean("trust.u")
    obj.trust = ho.Trust.UNKNOWN
    assert obj.state is ho.ObjectState.CLEAN
    assert ho.is_coherent(obj) is ho.Coherence.UNKNOWN


def test_asserted_trust_is_unknown_coherence_not_coherent():
    obj = _clean("asserted.0")
    obj.trust = ho.Trust.ASSERTED
    assert ho.is_coherent(obj) is ho.Coherence.UNKNOWN


def test_unknown_visibility_is_unknown_coherence():
    obj = _clean("vis.u")
    obj.device_visibility = "UNKNOWN"
    assert ho.is_coherent(obj) is ho.Coherence.UNKNOWN


def test_device_not_in_visibility_is_not_coherent():
    obj = _clean()
    assert ho.is_coherent(obj, on_device="FPGA_HBM_0") is ho.Coherence.NOT_COHERENT


# ------------------------------------------------------------------ consume gate (NEGATIVE CONTROL)


def test_require_coherent_refuses_unknown():
    obj = ho.ManagedObject(identity=ho.HgvasRef("need.u"))
    with pytest.raises(ho.CoherenceRequiredError, match="is_coherent=UNKNOWN"):
        ho.require_coherent(obj, reader="unit-test")


def test_require_coherent_refuses_not_coherent():
    obj = _stale()
    with pytest.raises(ho.CoherenceRequiredError, match="is_coherent=NOT_COHERENT"):
        ho.require_coherent(obj, reader="unit-test")


def test_require_coherent_accepts_coherent():
    ho.require_coherent(_clean(), reader="unit-test")


def test_consume_for_kernel_refuses_unknown():
    """NEGATIVE CONTROL: a path requiring coherence must actually fire."""
    obj = ho.ManagedObject(identity=ho.HgvasRef("k.u"))
    with pytest.raises(ho.CoherenceRequiredError, match="kernel:gemm"):
        ho.consume_for_kernel(obj, kernel="gemm", device="APPLE_DOMAIN_0")


def test_read_payload_refuses_unknown():
    obj = ho.ManagedObject(identity=ho.HgvasRef("p.u"))
    with pytest.raises(ho.CoherenceRequiredError):
        ho.read_payload(obj)
    assert ho.read_payload(_clean()) == b"payload-bytes"


def test_every_consume_path_refuses_the_same_unknown_object():
    obj = ho.ManagedObject(identity=ho.HgvasRef("all.u"))
    with pytest.raises(ho.CoherenceRequiredError):
        ho.require_coherent(obj, reader="a")
    with pytest.raises(ho.CoherenceRequiredError):
        ho.consume_for_kernel(obj, kernel="k", device="APPLE_DOMAIN_0")
    with pytest.raises(ho.CoherenceRequiredError):
        ho.read_payload(obj)


# ------------------------------------------------------------------ kernel boundary (NEGATIVE CONTROL)


def test_kernel_boundary_without_sync_moves_clean_to_unknown():
    """NEGATIVE CONTROL: CLEAN must not remain CLEAN across an unsynced kernel."""
    obj = _clean("k.bound")
    assert obj.state is ho.ObjectState.CLEAN
    ho.cross_kernel_boundary(obj, synchronized=False, access=ho.KernelAccess.READ)
    assert obj.state is ho.ObjectState.UNKNOWN
    assert obj.state is not ho.ObjectState.CLEAN
    assert ho.is_coherent(obj) is ho.Coherence.UNKNOWN
    with pytest.raises(ho.CoherenceRequiredError):
        ho.consume_for_kernel(obj, kernel="decode", device="APPLE_DOMAIN_0")


def test_kernel_boundary_without_sync_does_not_leave_clean_even_on_write_access():
    obj = _clean("k.write")
    ho.cross_kernel_boundary(obj, synchronized=False, access=ho.KernelAccess.WRITE)
    assert obj.state is ho.ObjectState.UNKNOWN


def test_kernel_boundary_with_sync_read_stays_clean():
    obj = _clean("k.sync")
    ho.cross_kernel_boundary(obj, synchronized=True, access=ho.KernelAccess.READ)
    assert obj.state is ho.ObjectState.CLEAN
    assert ho.is_coherent(obj) is ho.Coherence.COHERENT


def test_kernel_boundary_with_sync_unknown_access_is_unknown():
    obj = _clean("k.uacc")
    ho.cross_kernel_boundary(obj, synchronized=True, access=ho.KernelAccess.UNKNOWN)
    assert obj.state is ho.ObjectState.UNKNOWN
    assert ho.is_coherent(obj) is ho.Coherence.UNKNOWN


def test_kernel_boundary_with_sync_write_is_asserted_not_trusted():
    obj = _clean("k.ww")
    prior = obj.version
    ho.cross_kernel_boundary(obj, synchronized=True, access=ho.KernelAccess.WRITE)
    assert obj.state is ho.ObjectState.CLEAN
    assert obj.trust is ho.Trust.ASSERTED
    assert obj.version == prior + 1
    # fence is not a digest -- we do not *know* the bytes
    assert ho.is_coherent(obj) is ho.Coherence.UNKNOWN
    with pytest.raises(ho.CoherenceRequiredError):
        ho.consume_for_kernel(obj, kernel="store", device="APPLE_DOMAIN_0")


# ------------------------------------------------------------------ migrate / invalidate / digest


def test_migrate_without_digest_is_unknown_not_clean():
    obj = _clean("mig.0")
    ho.migrate(obj, ho.Location(ho.MemoryTier.HBM, "FPGA_HBM_0"))
    assert obj.state is ho.ObjectState.UNKNOWN
    assert obj.location.device == "FPGA_HBM_0"
    assert obj.device_visibility == "UNKNOWN"
    assert ho.is_coherent(obj) is ho.Coherence.UNKNOWN
    with pytest.raises(ho.CoherenceRequiredError):
        ho.consume_for_kernel(obj, kernel="hbm_read", device="FPGA_HBM_0")


def test_migrate_with_matching_digest_is_clean_at_dest():
    obj = _clean("mig.ok")
    digest = obj.physical.digest
    ho.migrate(obj, ho.Location(ho.MemoryTier.HBM, "FPGA_HBM_0"), digest=digest)
    assert obj.state is ho.ObjectState.CLEAN
    assert obj.location.device == "FPGA_HBM_0"
    assert ho.is_coherent(obj, on_device="FPGA_HBM_0") is ho.Coherence.COHERENT


def test_migrate_mismatched_digest_is_unknown():
    obj = _clean("mig.bad")
    ho.migrate(obj, ho.Location(ho.MemoryTier.HBM, "FPGA_HBM_0"), digest="00" * 16)
    assert obj.state is ho.ObjectState.UNKNOWN


def test_invalidate_is_not_coherent():
    obj = _clean("inv.0")
    ho.invalidate(obj, reason="capacity eviction")
    assert obj.state is ho.ObjectState.STALE
    assert ho.is_coherent(obj) is ho.Coherence.NOT_COHERENT


def test_device_digest_without_provider_is_unknown():
    obj = _clean("dig.0")
    result = ho.device_digest(obj, provider=None)
    assert result.status == "UNKNOWN"
    assert result.digest is None
    # absence of a digest does not invent one, and does not mark CLEAN worse
    assert obj.state is ho.ObjectState.CLEAN


def test_device_digest_mismatch_is_stale_not_unknown():
    obj = _clean("dig.bad")

    class Lying:
        resident_digest_path = "compute"

        def digest_resident(self, key: str) -> str:
            return "ff" * 16

    result = ho.device_digest(obj, provider=Lying())
    assert result.status == "MISMATCH"
    assert obj.state is ho.ObjectState.STALE
    assert ho.is_coherent(obj) is ho.Coherence.NOT_COHERENT


def test_device_digest_match_is_verified():
    obj = _clean("dig.ok")

    class Honest:
        resident_digest_path = "compute"

        def digest_resident(self, key: str) -> str:
            return "ab" * 16

    result = ho.device_digest(obj, provider=Honest())
    assert result.status == "VERIFIED"
    assert result.path == "compute"


def test_device_digest_provider_error_is_unknown():
    obj = _clean("dig.err")

    class Hang:
        def digest_resident(self, key: str) -> str:
            raise TimeoutError("probe timed out")

    result = ho.device_digest(obj, provider=Hang())
    assert result.status == "UNKNOWN"
    assert obj.state is ho.ObjectState.UNKNOWN


def test_establish_clean_refuses_empty_evidence():
    obj = ho.ManagedObject(identity=ho.HgvasRef("no.ev"))
    with pytest.raises(ho.ManagedObjectError, match="without evidence"):
        ho.establish_clean(
            obj,
            location=ho.Location(ho.MemoryTier.UMA, "APPLE_DOMAIN_0"),
            visibility={"APPLE_DOMAIN_0"},
            evidence="",
            digest="ab" * 16,
        )
    assert obj.state is ho.ObjectState.UNKNOWN


def test_illegal_transition_fails_closed():
    """NEGATIVE CONTROL: CLEAN is not a legal successor of UNKNOWN on
    transition(). The only way to earn CLEAN is establish_clean()."""
    obj = ho.ManagedObject(identity=ho.HgvasRef("legal.0"))
    assert obj.state is ho.ObjectState.UNKNOWN
    with pytest.raises(ho.ManagedObjectError, match="illegal object-state"):
        obj.transition(ho.ObjectState.CLEAN)
    assert obj.state is ho.ObjectState.UNKNOWN
    assert ho.is_coherent(obj) is ho.Coherence.UNKNOWN
    dirty = _clean("legal.dirty")
    ho.mark_written(dirty, device="APPLE_DOMAIN_0")
    with pytest.raises(ho.ManagedObjectError, match="illegal object-state"):
        dirty.transition(ho.ObjectState.CLEAN)


# ------------------------------------------------------------------ move vs recompute


def test_unknown_transfer_is_undecidable():
    d = ho.move_vs_recompute("UNKNOWN", 1.0)
    assert d.decision is ho.Decision.UNDECIDABLE
    assert d.transfer_cost == "UNKNOWN"
    assert d.recompute_cost == 1.0


def test_unknown_recompute_is_undecidable():
    d = ho.move_vs_recompute(1.0, "UNKNOWN")
    assert d.decision is ho.Decision.UNDECIDABLE


def test_none_cost_is_treated_as_unknown_not_zero():
    d = ho.move_vs_recompute(None, 1.0)
    assert d.decision is ho.Decision.UNDECIDABLE


def test_never_guesses_that_moving_is_cheaper():
    """A known transfer cost and an unknown recompute must NOT produce MOVE."""
    d = ho.move_vs_recompute(0.001, "UNKNOWN")
    assert d.decision is not ho.Decision.MOVE
    assert d.decision is ho.Decision.UNDECIDABLE


def test_recompute_can_beat_transfer():
    d = ho.move_vs_recompute(5.0, 1.0)
    assert d.decision is ho.Decision.RECOMPUTE


def test_move_wins_only_when_both_known_and_cheaper():
    d = ho.move_vs_recompute(1.0, 5.0)
    assert d.decision is ho.Decision.MOVE


def test_tie_is_decidable_but_not_a_guess():
    d = ho.move_vs_recompute(2.0, 2.0)
    assert d.decision is ho.Decision.TIE


def test_both_unavailable_is_neither():
    d = ho.move_vs_recompute("UNAVAILABLE", "UNAVAILABLE")
    assert d.decision is ho.Decision.NEITHER


def test_recompute_only_option():
    d = ho.move_vs_recompute("UNAVAILABLE", 3.0)
    assert d.decision is ho.Decision.RECOMPUTE


def test_negative_cost_refused():
    with pytest.raises(ho.ManagedObjectError, match="cost must be"):
        ho.move_vs_recompute(-1.0, 1.0)


# ------------------------------------------------------------------ HUMF adapter: extend, do not fork


def test_from_humf_clean_trust_unknown_is_not_reported_coherent():
    """HUMF valid_copies() names a CLEAN copy even when trust is UNKNOWN.
    The overlay must not call that coherent."""
    humf = _humf()
    o = humf.HumfObject("W_overlay", "tensor", 8, "f32")
    o.place(humf.Materialization(
        "APPLE_UM", "dense_f32", "row_major", 32, humf.State.CLEAN,
        payload=b"\x00" * 32, trust="UNKNOWN",
    ))
    assert o.valid_copies() == ["APPLE_UM"]
    overlay = ho.from_humf_object(o)
    assert overlay.state is ho.ObjectState.CLEAN
    assert overlay.trust is ho.Trust.UNKNOWN
    assert ho.is_coherent(overlay) is ho.Coherence.UNKNOWN
    with pytest.raises(ho.CoherenceRequiredError):
        ho.consume_for_kernel(overlay, kernel="use", device="APPLE_DOMAIN_0")


def test_from_humf_clean_trusted_apple_um_is_coherent():
    humf = _humf()
    o = humf.HumfObject("W_ok", "tensor", 8, "f32")
    o.place(humf.Materialization(
        "APPLE_UM", "dense_f32", "row_major", 32, humf.State.CLEAN,
        payload=b"\x00" * 32, trust="TRUSTED",
    ))
    o.content_digest = "cd" * 16
    overlay = ho.from_humf_object(o)
    assert overlay.location.memory_tier is ho.MemoryTier.UMA
    assert overlay.location.device == "APPLE_DOMAIN_0"
    assert ho.is_coherent(overlay) is ho.Coherence.COHERENT


def test_apply_kernel_boundary_to_humf_moves_clean_to_unknown():
    """Extension point: the overlay drives HUMF's existing CLEAN->UNKNOWN edge."""
    humf = _humf()
    o = humf.HumfObject("W_k", "tensor", 8, "f32")
    o.place(humf.Materialization(
        "APPLE_UM", "dense_f32", "row_major", 32, humf.State.CLEAN,
        payload=b"\x00" * 32,
    ))
    assert o.valid_copies() == ["APPLE_UM"]
    ho.apply_kernel_boundary_to_humf(o, "APPLE_UM", synchronized=False)
    assert o.materializations["APPLE_UM"].state is humf.State.UNKNOWN
    assert o.materializations["APPLE_UM"].trust == "UNKNOWN"
    assert o.valid_copies() == []
    overlay = ho.from_humf_object(o)
    assert overlay.state is ho.ObjectState.UNKNOWN
    with pytest.raises(ho.CoherenceRequiredError):
        ho.require_coherent(overlay, reader="after-humf-kernel-boundary")


def test_apply_kernel_boundary_to_humf_with_sync_leaves_clean():
    humf = _humf()
    o = humf.HumfObject("W_ks", "tensor", 8, "f32")
    o.place(humf.Materialization(
        "APPLE_UM", "dense_f32", "row_major", 32, humf.State.CLEAN,
        payload=b"\x00" * 32,
    ))
    ho.apply_kernel_boundary_to_humf(o, "APPLE_UM", synchronized=True)
    assert o.materializations["APPLE_UM"].state is humf.State.CLEAN


def test_move_vs_recompute_from_humf_recipe_without_cost_is_undecidable():
    """HUMF plan_acquire would skip recompute and pick TRANSFER. We refuse."""
    humf = _humf()
    o = humf.HumfObject("W_r", "tensor", 8, "f32", recompute_cost_s=None)
    o.recompute = lambda: b"x" * 32
    d = ho.move_vs_recompute_from_humf(o, transfer_cost=0.001)
    assert d.recompute_cost == "UNKNOWN"
    assert d.decision is ho.Decision.UNDECIDABLE
    assert d.decision is not ho.Decision.MOVE


def test_move_vs_recompute_from_humf_no_recipe_no_cost_is_move():
    humf = _humf()
    o = humf.HumfObject("W_n", "tensor", 8, "f32", recompute_cost_s=None)
    d = ho.move_vs_recompute_from_humf(o, transfer_cost=0.001)
    assert d.recompute_cost == "UNAVAILABLE"
    assert d.decision is ho.Decision.MOVE


def test_hmf_is_still_the_fabric_not_a_rival():
    recovered = ho.recover_humf_surface()
    assert recovered["imported"] is True
    assert recovered["has_is_coherent"] is False
    assert recovered["has_kernel_boundary"] is False
    assert recovered["has_require_coherent"] is False
    assert recovered.get("hmf_is_humf") is True


def test_illegal_humf_projection_of_unknown_name():
    with pytest.raises(ho.ManagedObjectError, match="unmapped HUMF"):
        ho.project_humf_state("NOT_A_STATE")


# ------------------------------------------------------------------ version / location


def test_version_increments_on_write_not_on_invalidate():
    obj = _clean("ver.0")
    assert obj.version == 0
    ho.mark_written(obj, device="APPLE_DOMAIN_0")
    assert obj.version == 1
    ho.invalidate(obj, reason="test")
    assert obj.version == 1


def test_receipt_names_recovered_humf_paths():
    doc = json.loads(ho.build().read_text())
    rec = doc["recovered_implementation"]["hmf_humf"]
    assert rec["humf_path"] == "tools/accelerator/humf.py"
    assert rec["hmf_path"] == "tools/accelerator/hmf.py"
    assert rec["has_is_coherent"] is False
