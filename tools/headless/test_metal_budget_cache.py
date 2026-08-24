"""metal_device() must not compile Swift on every call.

The latency ledger ranked MemGate.consider(refresh_metal=True) third at 277.7 ms.
The cause was metal_device() writing a temp .swift file and running `swift` on
every uncached call, measured at 229 ms median, to re-read values that are
constants of the machine. The one dynamic field it returned,
currentAllocatedSize, comes from a throwaway subprocess and describes that
helper's allocation rather than ours, so the refresh never bought freshness.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metal_budget


def test_repeat_calls_are_cached_not_respawned():
    metal_budget.metal_device()  # warm
    t0 = time.perf_counter()
    for _ in range(20):
        metal_budget.metal_device()
    per_call_ms = (time.perf_counter() - t0) * 1000 / 20
    assert per_call_ms < 5.0, (
        f"{per_call_ms:.1f} ms per cached call: a Swift compile is ~229 ms, so this "
        "is respawning rather than caching"
    )


def test_force_still_reprobes():
    metal_budget.metal_device()
    metal_budget._DEVICE_CACHE = {"source": "sentinel", "recommendedMaxWorkingSetSize": 1}
    assert metal_budget.metal_device()["source"] == "sentinel", "cache must be honoured"
    assert metal_budget.metal_device(force=True)["source"] != "sentinel", (
        "force=True must bypass the cache, otherwise the escape hatch is fake"
    )


def test_reported_budget_is_a_real_number():
    d = metal_budget.metal_device(force=True)
    wss = d["recommendedMaxWorkingSetSize"]
    assert wss > 0, "a zero working set would make every admission decision meaningless"
    assert "source" in d and d["source"], "must say whether it measured or estimated"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("3/3 passed")
