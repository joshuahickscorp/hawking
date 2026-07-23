#!/usr/bin/env python3.12
"""Detached Qwen downloader with a WRITE-CADENCE disk floor.

snapshot_download has no disk-space awareness, and the supervisor's disk guard only samples once per
60s launchd tick - so a fast multi-worker download can undershoot the reserve toward 0 between ticks and
crash the box. This worker runs the download in a background thread while the main thread polls free disk
every few seconds and hard-exits the whole process group the instant free < reserve. The partial shard is
left for snapshot_download to resume on the next launch.

argv: repo revision local_dir max_workers reserve_gb [poll_seconds]
"""
import os
import shutil
import sys
import threading

ALLOW = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"]


def main() -> int:
    repo, rev, local_dir, workers, reserve_gb = sys.argv[1:6]
    workers = int(workers)
    reserve_gb = float(reserve_gb)
    poll = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0

    from huggingface_hub import snapshot_download  # imported here so --help/self-check stays light

    done = threading.Event()
    err: list[BaseException] = []

    def _dl() -> None:
        try:
            snapshot_download(repo, revision=rev, local_dir=local_dir,
                              max_workers=workers, allow_patterns=ALLOW)
        except BaseException as exc:  # noqa: BLE001 - surfaced via exit code, not swallowed
            err.append(exc)
        finally:
            done.set()

    threading.Thread(target=_dl, daemon=True).start()
    while not done.wait(poll):
        free_gb = shutil.disk_usage(local_dir).free / 1e9
        if free_gb < reserve_gb:
            sys.stderr.write(f"[qwen-dl] disk floor breached: {free_gb:.1f} GB < {reserve_gb} GB; aborting\n")
            sys.stderr.flush()
            os._exit(3)   # immediate: kill the download threads too, before the next write fills the disk
    if err:
        sys.stderr.write(f"[qwen-dl] download error: {type(err[0]).__name__}: {err[0]}\n")
        return 1
    return 0


def _selfcheck() -> None:
    # The watchdog's one job: exit the instant free < reserve. Prove the comparison + poll loop fire
    # without a real download (stub snapshot_download, fake a shrinking disk, assert os._exit is called).
    import types
    calls = {"exit": None}
    fake_free = [200.0, 200.0, 30.0]  # third sample breaches a 40 GB reserve
    mod = sys.modules[__name__]
    orig_du, orig_exit = shutil.disk_usage, os._exit
    hf = types.ModuleType("huggingface_hub")
    hf.snapshot_download = lambda *a, **k: __import__("time").sleep(60)  # never finishes on its own
    sys.modules["huggingface_hub"] = hf
    shutil.disk_usage = lambda p: types.SimpleNamespace(free=fake_free.pop(0) * 1e9 if fake_free else 30.0 * 1e9)

    def _fake_exit(code):
        calls["exit"] = code
        raise SystemExit(code)  # unwind out of main() in the test instead of really exiting
    os._exit = _fake_exit
    try:
        sys.argv = ["worker", "repo", "rev", "/tmp", "4", "40", "0"]
        try:
            main()
        except SystemExit:
            pass
        assert calls["exit"] == 3, f"expected os._exit(3) on disk breach, got {calls['exit']}"
        print("selfcheck ok: aborts on disk-floor breach")
    finally:
        shutil.disk_usage, os._exit = orig_du, orig_exit
        del sys.modules["huggingface_hub"]


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "selfcheck":
        _selfcheck()
    else:
        raise SystemExit(main())
