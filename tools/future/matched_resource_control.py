"""Matched-resource controls for a two-arm comparison. (G028, and G012's OIII)

The G028 audit found five of six OIII pieces built and this one absent
everywhere -- grep across tools/ and hcli/ returned nothing. It is the piece
that decides whether an NX win is REAL: a candidate measured with more memory, a
warmer cache or a freer GPU than its parent has not beaten the parent, it has
been measured more kindly.

This campaign has already paid for that. From the session scars: arm A always
ran first and the ratio read 1.472x; alternating the order took it to 1.404x and
the baseline spread from 28% to 2%. The bias was worth 5% of the headline
number, in the direction of the claim.

Nothing here runs an arm. It decides the ORDER, states the conditions both arms
must share, and refuses a comparison whose arms did not actually share them --
so the discipline exists before there is a candidate to apply it to, which is
what "preparation is parallel, execution is earned" means.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Conditions that must MATCH between arms, and the tolerance each is allowed.
# A tolerance of 0 means exact.
MATCHED_CONDITIONS: Dict[str, float] = {
    "prompt_tokens": 0,          # different work is not a comparison
    "concurrency": 0,
    "max_seq_len": 0,
    "runtime": 0,                # string equality, checked separately
    "free_gb_at_start": 4.0,     # host memory the arm could have used
    "resident_rss_gb": 1.0,
}
STRING_CONDITIONS = ("runtime", "machine", "workload")


class UnmatchedArms(ValueError):
    """The arms did not share the conditions the comparison assumes."""


def alternating_order(arms: Sequence[str], repeats: int) -> List[str]:
    """A run order that cancels position bias in the ratio.

    Not A,A,B,B and not one A followed by one B. Position effects here are
    monotone -- a DVFS ramp, a warming cache, KV growth -- so the correction is
    to give each arm the same distribution of positions. ABBA does that for two
    arms at each repeat; a plain ABAB leaves A permanently earlier.
    """
    if len(arms) != 2:
        raise ValueError(f"matched control is a TWO-arm design; got {len(arms)}")
    a, b = arms
    out: List[str] = []
    for i in range(max(1, int(repeats))):
        out.extend([a, b, b, a] if i % 2 == 0 else [b, a, a, b])
    return out


def position_balance(order: Sequence[str]) -> Dict[str, float]:
    """Mean position of each arm. Equal means position cannot explain a delta."""
    seen: Dict[str, List[int]] = {}
    for i, arm in enumerate(order):
        seen.setdefault(arm, []).append(i)
    return {arm: sum(pos) / len(pos) for arm, pos in seen.items()}


def check_matched(arm_a: Dict[str, Any], arm_b: Dict[str, Any]) -> Dict[str, Any]:
    """Did these two arms actually share their conditions?

    Returns the verdict rather than raising, so a caller can record an UNMATCHED
    comparison as a finding instead of losing it.
    """
    mismatches: List[Dict[str, Any]] = []
    missing: List[str] = []
    for key, tol in MATCHED_CONDITIONS.items():
        if key in STRING_CONDITIONS:
            continue
        a, b = arm_a.get(key), arm_b.get(key)
        if a is None or b is None:
            missing.append(key)
            continue
        if abs(float(a) - float(b)) > float(tol):
            mismatches.append({"condition": key, "a": a, "b": b, "tolerance": tol})
    for key in STRING_CONDITIONS:
        a, b = arm_a.get(key), arm_b.get(key)
        if a is None or b is None:
            missing.append(key)
        elif str(a) != str(b):
            mismatches.append({"condition": key, "a": a, "b": b, "tolerance": "exact"})

    return {
        "matched": not mismatches and not missing,
        "mismatches": mismatches,
        # A condition NOBODY RECORDED is not a matched condition. Treating an
        # absent field as agreement is how an unmatched comparison passes for a
        # matched one -- the same shape as "absence of a field is not absence of
        # a parent" in the NX contract.
        "unrecorded": sorted(set(missing)),
        "verdict": ("MATCHED" if not mismatches and not missing else
                    "UNMATCHED" if mismatches else "UNVERIFIABLE -- conditions not recorded"),
    }


def require_matched(arm_a: Dict[str, Any], arm_b: Dict[str, Any]) -> None:
    r = check_matched(arm_a, arm_b)
    if not r["matched"]:
        raise UnmatchedArms(
            f"{r['verdict']}: mismatches={r['mismatches']} unrecorded={r['unrecorded']}. "
            f"A candidate measured under different conditions than its parent has not beaten "
            f"the parent."
        )


def _selftest() -> List[str]:
    fails = []
    base = {"prompt_tokens": 512, "concurrency": 1, "max_seq_len": 8192,
            "runtime": "hawking-native sealed-3.14", "machine": "m3-ultra",
            "workload": "greedy-96", "free_gb_at_start": 60.0, "resident_rss_gb": 11.5}

    if not check_matched(dict(base), dict(base))["matched"]:
        fails.append("identical arms are not reported matched")

    # Different work is not a comparison.
    b = dict(base); b["prompt_tokens"] = 1024
    r = check_matched(base, b)
    if r["matched"] or r["mismatches"][0]["condition"] != "prompt_tokens":
        fails.append("a prompt-length difference was not caught")

    # Memory within tolerance is fine; outside is not.
    b = dict(base); b["free_gb_at_start"] = 62.0
    if not check_matched(base, b)["matched"]:
        fails.append("a 2 GB difference inside the 4 GB tolerance was rejected")
    b["free_gb_at_start"] = 40.0
    if check_matched(base, b)["matched"]:
        fails.append("a 20 GB memory advantage was accepted as matched")

    # AN ABSENT CONDITION IS NOT AGREEMENT.
    b = dict(base); b.pop("resident_rss_gb")
    r = check_matched(base, b)
    if r["matched"]:
        fails.append("an UNRECORDED condition was treated as a matched one")
    if "resident_rss_gb" not in r["unrecorded"]:
        fails.append("the unrecorded condition was not named")

    # Order: each arm must average the same position.
    order = alternating_order(["A", "B"], repeats=2)
    bal = position_balance(order)
    if abs(bal["A"] - bal["B"]) > 1e-9:
        fails.append(f"alternating order leaves a position bias: {bal}")
    if order[:4] != ["A", "B", "B", "A"]:
        fails.append(f"first repeat is not ABBA: {order[:4]}")
    # and the naive orders it replaces must NOT balance
    if abs(position_balance(["A", "A", "B", "B"])["A"]
           - position_balance(["A", "A", "B", "B"])["B"]) < 1e-9:
        fails.append("the AABB control balances, so this test cannot detect bias at all")
    if abs(position_balance(["A", "B", "A", "B"])["A"]
           - position_balance(["A", "B", "A", "B"])["B"]) < 1e-9:
        fails.append("ABAB balances, so alternation would be unnecessary")

    try:
        require_matched(base, {"prompt_tokens": 1})
        fails.append("require_matched accepted arms that share nothing")
    except UnmatchedArms:
        pass
    return fails


if __name__ == "__main__":
    fails = _selftest()
    for f in fails:
        print("SELFTEST FAIL:", f)
    print(f"selftest: {'FAILED' if fails else 'PASS'} — matching, absent conditions, and "
          f"position balance")
    raise SystemExit(1 if fails else 0)
