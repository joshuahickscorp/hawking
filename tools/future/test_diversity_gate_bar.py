"""The diversity bar must stay inside the metric it measures. (G020)

median_4gram_repeat is a FRACTION of 4-grams that repeat: its range is [0, 1].
The bar was `3.0 x this specimen's dense median r4`, which is unbounded, so any
parent repeating more than a third of its 4-grams got a bar ABOVE 1.0 and an
axis that could not reject anything. Qwen3-0.6B's bar was 1.0857 and its
deliberately inverted control -- 82% of 4-grams repeating -- is recorded in
G017_ORGAN_ALLOCATION.json as diversity_ok: TRUE.

G020's acceptance: "The gate must REJECT the G017 INVERTED generations it
currently passes." This asserts exactly that, on the live gate, using the arms
from that receipt.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "future"))

from organ_allocation import gate_from_reference  # noqa: E402

# From receipts/future/G017_ORGAN_ALLOCATION.json
QWEN_DENSE_R4 = 0.3619
QWEN_DENSE_PPL = 4.7342
ARMS = [
    ("dense parent", 0.3619, 4.7342, True),
    ("arm b", 0.6923, 6.1738, False),
    ("arm c", 0.7103, 5.2278, False),
    ("inverted control", 0.8224, 12.84, False),
]
# O003's reference, which must be UNAFFECTED -- a cap that tightens a sound gate
# would silently rewrite every verdict in the capability cliff.
O003_DENSE_R4 = 0.0538


def main() -> int:
    fails = []
    ref = {"r4_full": QWEN_DENSE_R4, "ppl_full": QWEN_DENSE_PPL}

    for name, r4, ppl, want_div in ARMS:
        g = gate_from_reference({"r4_full": r4, "ppl_full": ppl}, ref)
        if bool(g["diversity_ok"]) != want_div:
            fails.append(f"{name} (r4={r4}): diversity_ok={g['diversity_ok']}, want {want_div} "
                         f"[bar {g['gate_r4_max']}]")

    bar = gate_from_reference({"r4_full": 0.0, "ppl_full": 1.0}, ref)["gate_r4_max"]
    if bar >= 1.0:
        fails.append(f"the bar {bar} is at or above the metric's ceiling -- nothing can fail it")

    # O003 must not move.
    o = gate_from_reference({"r4_full": 0.0, "ppl_full": 1.0},
                            {"r4_full": O003_DENSE_R4, "ppl_full": 2.8936})
    if abs(o["gate_r4_max"] - 0.1614) > 1e-3:
        fails.append(f"O003's bar moved to {o['gate_r4_max']}; it must stay 0.1614 -- a cap that "
                     f"tightens a sound gate rewrites the capability cliff")

    for f in fails:
        print("FAIL:", f)
    print(f"{'FAILED' if fails else 'PASS'} — G017's collapsed arms rejected, its dense parent "
          f"accepted, O003 untouched")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
