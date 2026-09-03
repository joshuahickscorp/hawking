"""The dashboard must never take itself down, and must never invent a number."""
from __future__ import annotations

from tools.future import hawking_health as H


def test_every_section_is_present_and_none_crashed():
    doc = H.report()
    for section in ("roadmap", "acceptance", "measurement_provenance",
                    "tool_friction", "odyssey_wall"):
        assert section in doc, section
        assert "unavailable" not in doc[section], (section, doc[section])


def test_one_broken_instrument_does_not_take_the_dashboard_down(monkeypatch):
    """A dashboard that dies on its weakest reader gets deleted, and then nobody
    watches any of the numbers."""
    def boom():
        raise RuntimeError("instrument exploded")
    monkeypatch.setattr(H, "friction", boom)
    doc = H.report()
    assert "unavailable" in doc["tool_friction"]
    assert "instrument exploded" in doc["tool_friction"]["unavailable"]
    assert "unavailable" not in doc["roadmap"], "one failure leaked into another section"


def test_the_wall_projection_is_still_refused():
    """If this ever reports a number, a component acquired a measured duration and
    the claim must be re-derived rather than inherited."""
    assert "REFUSED" in H.report()["odyssey_wall"]["projection"]


def test_it_reads_and_never_writes():
    import inspect
    src = inspect.getsource(H)
    for forbidden in ("write_text", "write_receipt", "unlink", "rmtree"):
        assert forbidden not in src, forbidden
