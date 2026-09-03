#!/usr/bin/env python3
"""What the Odyssey funnel can afford to read, measured rather than assumed.

The corpus does not live on the machine that computes over it. It lives on an
external USB drive whose platters, not whose link, are the wall: the enclosure
negotiates 120 Gb/s and delivers ~122 MB/s. At that rate one pass over the
whole corpus is most of a working day, and three Odysseys that each deep-read
every specimen cannot fit in twelve hours by a factor of two and a half.

What rescues the schedule is that the cheap stages do not need weight VALUES.
A safetensors header carries every tensor name, shape, dtype and offset in a
few kilobytes, so census, clustering and structural probing read megabytes to
stand for terabytes. tools/odyssey/specimen_open.py already refuses weight
bytes on that path; this measures what the refusal is worth.

The deep stages do need values, and the lever there is that the internal SSD
is ~24x the external drive. Stage a survivor once and every later Odyssey
reads it for free. So the real budget is not time -- it is how much of the
corpus survives to deep probing, capped by staging space.

Run it to get the numbers for the drive that is actually attached.
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import time

CATALOG = pathlib.Path("receipts/future/modellake-index/catalog.json")
BUDGET_HOURS = 12.0


def specimens() -> list[dict]:
    raw = json.loads(CATALOG.read_text())
    return raw if isinstance(raw, list) else raw["specimens"]


def measure_read_bps(path: pathlib.Path, cap: int = 1 << 31) -> float:
    """Sequential read rate, from the file the corpus is actually made of."""
    want = min(cap, path.stat().st_size)
    read = 0
    start = time.monotonic()
    with open(path, "rb") as handle:
        while read < want:
            chunk = handle.read(1 << 23)
            if not chunk:
                break
            read += len(chunk)
    return read / max(time.monotonic() - start, 1e-9)


def header_census(entries: list[dict]) -> tuple[int, int, int, float]:
    """Read every header we can reach. Returns (ok, header_bytes, bodies, secs)."""
    ok = header_bytes = bodies = 0
    start = time.monotonic()
    for entry in entries:
        root = pathlib.Path(entry.get("path", ""))
        if not root.is_dir():
            continue
        shard = next(iter(sorted(root.glob("*.safetensors"))), None)
        if shard is None:
            continue
        try:
            with open(shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                if length > (1 << 28):
                    continue
                handle.read(length)
        except OSError:
            continue
        ok += 1
        header_bytes += 8 + length
        bodies += int(entry.get("bytes") or 0)
    return ok, header_bytes, bodies, time.monotonic() - start


def main() -> int:
    entries = specimens()
    total = sum(int(e.get("bytes") or 0) for e in entries)
    reachable = [e for e in entries if pathlib.Path(e.get("path", "")).is_dir()]
    if not reachable:
        print("no specimen directory is reachable; is the corpus drive mounted?")
        return 1

    probe = next(
        (p for e in reachable for p in sorted(pathlib.Path(e["path"]).glob("*.safetensors"))
         if p.stat().st_size > (1 << 30)),
        None,
    )
    if probe is None:
        print("no shard large enough to measure a sustained read rate")
        return 1
    bps = measure_read_bps(probe)

    ok, header_bytes, bodies, secs = header_census(reachable)
    budget_bytes = BUDGET_HOURS * 3600 * bps

    print(f"corpus            {len(entries)} specimens, {total / 1e12:.2f} TB")
    print(f"reachable         {len(reachable)}")
    print(f"sequential read   {bps / 1e6:.0f} MB/s  (measured on {probe.name})")
    print(f"one full pass     {total / bps / 3600:.1f} h")
    print(f"{BUDGET_HOURS:.0f}h buys          {budget_bytes / 1e12:.2f} TB"
          f"  = {budget_bytes / max(total, 1):.2f} corpus passes")
    print()
    print(f"header census     {ok} specimens in {secs:.1f}s, {header_bytes / 1e6:.1f} MB read")
    print(f"                  stands for {bodies / 1e12:.2f} TB  "
          f"(1 : {bodies / max(header_bytes, 1):,.0f})")
    print()
    deep = budget_bytes / max(total, 1)
    print(f"VERDICT: three deep passes over the whole corpus need "
          f"{3 * total / bps / 3600:.1f} h, which is "
          f"{3 * total / bps / 3600 / BUDGET_HOURS:.1f}x the {BUDGET_HOURS:.0f} h budget.")
    print(f"         Cheap stages are free. The budget is a SURVIVOR budget:")
    print(f"         at most {deep:.2f} corpus-equivalents may reach deep probing,")
    print(f"         and each survivor must be staged once and re-read from SSD.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
