"""The Python process view is a compatibility skin over Rust HCLI authority."""
from __future__ import annotations

from hcli.processes import Process, live_processes, render, summary


def test_summary_comes_from_the_native_process_authority():
    value = summary()
    assert set(value) >= {"count", "total_rss_bytes", "by_class", "roles", "processes"}
    assert value["count"] == len(value["processes"])
    assert value["total_rss_bytes"] == sum(
        process["rss_bytes"] for process in value["processes"]
    )
    for process in value["processes"]:
        assert isinstance(process["pid"], int) and process["pid"] > 0
        assert isinstance(process["role"], str) and process["role"]
        assert isinstance(process["safe_to_stop"], bool)
        assert process["memory_source"] in {"phys_footprint", "rss"}


def test_list_and_summary_share_the_native_record_shape():
    rows = [process.to_dict() for process in live_processes()]
    direct = summary()["processes"]
    assert {process["pid"] for process in rows} == {
        process["pid"] for process in direct
    }
    if rows:
        assert set(rows[0]) == set(direct[0])


def test_renderer_survives_a_host_with_nothing_running():
    assert render([]) == "no Hawking processes visible on this host"


def test_render_never_exceeds_its_width():
    text = render(width=60)
    assert all(len(line) <= 60 for line in text.splitlines()), text


def test_fallback_is_labelled_in_the_compatibility_renderer():
    fell_back = [
        Process(
            pid=1,
            ppid=0,
            rss_bytes=1234567,
            cpu_percent=0.0,
            elapsed="1:00",
            role="resident-body",
            process_class="ESSENTIAL_PERSISTENT",
            safe_to_stop=False,
            purpose="p",
            command="c",
            memory_source="rss",
        )
    ]
    text = render(fell_back, width=200)
    assert "rss fallback" in text and "under-reports" in text, text

    measured = [
        Process(
            pid=1,
            ppid=0,
            rss_bytes=1234567,
            cpu_percent=0.0,
            elapsed="1:00",
            role="resident-body",
            process_class="ESSENTIAL_PERSISTENT",
            safe_to_stop=False,
            purpose="p",
            command="c",
            memory_source="phys_footprint",
        )
    ]
    assert "rss fallback" not in render(measured, width=200)
