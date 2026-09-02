"""Every Hawking process gets a role, and a wrong role is worse than none.

The argv shapes below are the real ones observed on this host on 2026-09-01, not
invented fixtures. The one that matters most is the near-miss: a 3 MB process
carrying `--artifact-root` was labelled `resident-body` beside the actual 1.18 GB
body, which is exactly the kind of confident-but-wrong label this view exists to
replace.
"""
from __future__ import annotations

from hcli.processes import _body_of, _classify, live_processes, render, summary

RESIDENT_BODY = (
    "/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/release-fast/"
    "examples/ascension_qwen38_resident --artifact-root "
    "/Users/scammermike/noetic/NOETIC_PARENT_A --tokenizer "
    "/Users/scammermike/noetic/NOETIC_PARENT_A/tokenizer.json --max-seq-len 8192 "
    "--resident-identity sealed-3.14"
)
SUPERVISOR = (
    "/opt/homebrew/.../Python -m hcli.agentos.resident --supervise "
    "/tmp/ws/.hcli/resident/state.json"
)
SOVEREIGN = (
    "/Users/scammermike/.venvs/hawking-aider/bin/python3 "
    "tools/future/hcli_sovereign.py --run --minutes 600"
)
WATCHER = "/Library/.../Python /Users/scammermike/Downloads/hawking/tools/odyssey/modellake_watch.py --poll-secs 0.10"
DOWNLOAD = (
    "/Library/.../Python /Library/.../bin/hf download Qwen/Qwen3-Coder-30B-A3B-Instruct "
    "config.json --revision b2cff646eb4b --local-dir /Volumes/corpdrive/x --max-workers 16"
)


def _role(command: str):
    meta = _classify(command)
    return None if meta is None else meta[0]


def test_each_real_process_shape_gets_its_role():
    assert _role(RESIDENT_BODY) == "resident-body"
    assert _role(SUPERVISOR) == "resident-supervisor"
    assert _role(SOVEREIGN) == "sovereign-loop"
    assert _role(WATCHER) == "modellake-watcher"
    assert _role(DOWNLOAD) == "modellake-download"


def test_a_flag_alone_does_not_make_something_the_resident_body():
    """The near-miss that shipped a wrong label once already."""
    probe = "/Library/.../Python tools/accelerator/some_probe.py --artifact-root /x/y"
    assert _role(probe) != "resident-body"
    gate = "/bin/sh -c 'run_gate --artifact-root /x/y --resident-identity sealed-3.14'"
    # A shell wrapper is not the body either; the executable must be a resident.
    assert _role(gate) != "resident-body"


def test_unrelated_processes_are_not_claimed():
    for command in (
        "/usr/bin/python3 -m pip install requests",
        "/Applications/Safari.app/Contents/MacOS/Safari",
        "node /Users/x/.claude-grok/v2/contract.mjs",
        "",
    ):
        assert _classify(command) is None, command


def test_the_downloader_is_the_only_safe_to_stop_persistent_thing():
    """Stop-safety is the field an operator acts on, so it must not be sloppy."""
    assert _classify(DOWNLOAD)[2] is True, "a resumable download is safe to stop"
    for command in (RESIDENT_BODY, SUPERVISOR, SOVEREIGN, WATCHER):
        assert _classify(command)[2] is False, command


def test_model_identity_is_read_as_data_not_from_the_executable_name():
    """The body's identity comes off argv. The binary's name is not the truth."""
    assert _body_of(RESIDENT_BODY) == "sealed-3.14"
    assert _body_of(DOWNLOAD) == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert _body_of(SUPERVISOR) is None


def test_summary_and_render_survive_a_host_with_nothing_running():
    """Called on any machine, including one where Hawking is not running."""
    assert render([]) == "no Hawking processes visible on this host"
    live = summary()
    assert set(live) >= {"count", "total_rss_bytes", "by_class", "processes"}
    assert live["count"] == len(live["processes"])


def test_render_never_exceeds_its_width():
    text = render(width=60)
    assert all(len(line) <= 60 for line in text.splitlines()), text




def test_memory_comes_from_footprint_not_rss():
    """RSS is not Activity Monitor's Memory column, and here it is not close.

    Measured on this host: the resident body reported rss=1.19 GB against a
    phys_footprint of 12 GB, and two `hf download` children reported 2.93 and
    1.73 GB against 29.84 GB each in Activity Monitor. RSS counts resident pages
    and ignores what the compressor holds for the process; with tens of GB
    compressed system-wide that is most of the footprint. Reporting RSS made a
    box under real memory pressure look idle, which is the one thing this view
    exists to prevent.
    """
    import os

    from hcli.processes import _footprint_bytes

    procs = live_processes()
    if not procs:
        return  # nothing running is a valid host state, not a failure

    # THE ASSERTION THAT BITES. An earlier version of this test accepted either
    # source, so reverting the fix to `measured = None` left it green while the
    # view silently went back to under-reporting by ~10x. Where the platform CAN
    # give a footprint, the view MUST use it; hosts without the tool still skip.
    available = _footprint_bytes(os.getpid())
    if available is not None:
        assert any(p.memory_source == "phys_footprint" for p in procs), (
            "footprint is available on this host but every process fell back to "
            "rss, which under-reports memory by roughly ten-fold"
        )
    sources = {p.memory_source for p in procs}
    assert sources <= {"phys_footprint", "rss"}, sources
    # Every process must SAY which metric it used, so a silent fallback to the
    # under-reporting one can be seen rather than believed.
    for proc in procs:
        assert proc.memory_source in ("phys_footprint", "rss")
        assert proc.to_dict()["memory_source"] == proc.memory_source


def test_the_fallback_is_labelled_in_the_rendered_view():
    """An rss fallback must announce itself; a quiet one is the old bug back."""
    from hcli.processes import Process

    fell_back = [Process(
        pid=1, ppid=0, rss_bytes=1234567, cpu_percent=0.0, elapsed="1:00",
        role="resident-body", process_class="ESSENTIAL_PERSISTENT",
        safe_to_stop=False, purpose="p", command="c", memory_source="rss",
    )]
    text = render(fell_back, width=200)
    assert "rss fallback" in text and "under-reports" in text, text

    measured = [Process(
        pid=1, ppid=0, rss_bytes=1234567, cpu_percent=0.0, elapsed="1:00",
        role="resident-body", process_class="ESSENTIAL_PERSISTENT",
        safe_to_stop=False, purpose="p", command="c",
        memory_source="phys_footprint",
    )]
    assert "rss fallback" not in render(measured, width=200)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all green")


def test_acquisition_disk_check_binds_to_where_bytes_actually_land():
    """Three answers, two of them wrong, before this was right.

    The storage check first read the REPO's filesystem (reported 278 GiB for a
    download destined elsewhere), then HF_HUB_CACHE (reported 406 GiB on the
    internal SSD). Both are overridden in practice: the ModelLake watcher
    launches every acquisition with --local-dir pointing at the external volume,
    so that is where the bytes go and that is the only mount whose free space
    can authorize a 200 GiB acquisition.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT := __import__("pathlib").Path(
        __file__).resolve().parents[1]))
    from hcli import acquisition

    disk = (acquisition.propose() or {}).get("disk") or {}
    destination = str(disk.get("destination") or "")
    if not destination:
        return  # a host without the lake mounted is not a failure here
    mount = str(disk.get("mount") or disk.get("filesystem") or "")
    assert "hawking-modellake" in destination, (
        f"acquisition resolved {destination!r}, which is not where the watcher's "
        f"--local-dir sends the bytes"
    )
    # The decisive assertion: the mount must not be the boot volume when the
    # lake lives on an external drive.
    assert not mount.startswith("/System/Volumes/Data"), (
        f"storage check bound to the boot volume ({mount}) for a download that "
        f"lands on {destination}"
    )


def test_a_claimed_body_is_never_reaped():
    """ppid==1 alone is NOT evidence of abandonment, and reaping on it is fatal.

    The resident supervisor DAEMONISES its body deliberately, so a perfectly
    healthy in-use 11 GB model also sits at ppid 1. If the reaper keyed on ppid
    alone it would kill the live resident every time a new hcli process started
    -- turning a memory-leak fix into an outage. Only a body no live resident
    state file claims may be reaped.
    """
    from hcli import processes as P
    from hcli.processes import Process

    fake = Process(
        pid=999999, ppid=1, rss_bytes=11 * 1024 ** 3, cpu_percent=1.0,
        elapsed="1:00", role="resident-body", process_class="ESSENTIAL_PERSISTENT",
        safe_to_stop=False, purpose="p", command=RESIDENT_BODY,
    )
    real_live, real_claimed = P.live_processes, P._claimed_worker_pids
    try:
        P.live_processes = lambda **kw: [fake]

        # Claimed by a live resident -> invisible to the reaper.
        P._claimed_worker_pids = lambda: {999999}
        assert P.orphaned_resident_bodies() == [], (
            "reaper offered to kill a body the resident state file claims"
        )
        assert P.reap_orphaned_bodies(dry_run=True)["found"] == []

        # THE MUTATION THIS TEST EXISTS FOR: drop the claim check and the same
        # live body becomes a reap target. If this half passes while the half
        # above also passes, the guard is load-bearing.
        P._claimed_worker_pids = lambda: set()
        assert [p.pid for p in P.orphaned_resident_bodies()] == [999999]
    finally:
        P.live_processes, P._claimed_worker_pids = real_live, real_claimed


def test_reaping_never_blocks_a_runtime_from_starting():
    """Reclaiming memory must never be the reason hcli fails to boot.

    A NameError in the reporting line escaped the guard during development and
    killed the caller. The guard now covers the report too.
    """
    from hcli import processes as P
    from hcli.runtime import _reap_orphans_once

    real = P.reap_orphaned_bodies
    try:
        P.reap_orphaned_bodies = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        _reap_orphans_once()  # must not raise
    finally:
        P.reap_orphaned_bodies = real
