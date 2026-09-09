"""LOCAL-anatomy allocation must differ from the GLOBAL prior. (G021, S010 17)

S010 17: compare UNIFORM, GLOBAL-PRIOR ALLOCATION and LOCAL-ANATOMY ALLOCATION
at matched complete EBPW. The campaign has the first two -- UNIFORM,
ORGAN_WEIGHTED (the global prior) and INVERTED (its control). There is no LOCAL
policy: order_organs_by_deficit and per_param_weights both read the module-level
PRIOR_DEFICIT, so every arm is ordered by the aggregate.

The precondition is already settled (G021_PRECONDITION_HOLDS_ON_TWO_BODIES):
two measured bodies both invert the global prior on k/q and down/gate. This
pins what the code must therefore do -- produce a DIFFERENT byte map from the
same budget -- because an experiment whose two policies allocate identically
cannot discriminate no matter how carefully it is run.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "future"))

import organ_allocation as oa  # noqa: E402

# Qwen3-Embedding-0.6B, from .hcli receipt 843bfae7 (round 23), corroborated by
# rounds 22 and 24. NOT in the ledger: those rounds each measured it and none
# recorded it, so this is round-receipt evidence and is cited as such.
LOCAL_DEFICIT = {
    "k_proj": 41.09, "q_proj": 39.19, "o_proj": 23.53,
    "v_proj": 21.70, "down_proj": 21.32, "gate_proj": 20.29,
}


def main() -> int:
    fails = []

    if "LOCAL" not in oa.POLICIES:
        fails.append("LOCAL is not a policy -- S010 17 asks for three, and there are two plus a control")

    try:
        g_order = oa.order_organs_by_deficit()
        l_order = oa.order_organs_by_deficit(LOCAL_DEFICIT)
    except TypeError as exc:
        print("FAIL: order_organs_by_deficit takes no deficit map:", exc)
        print("FAILED — the local policy cannot be expressed")
        return 1

    # The orders must differ on the two pairs both measured bodies invert.
    def before(order, a, b):
        return order.index(a) < order.index(b)

    if not before(g_order, "q_proj", "k_proj"):
        fails.append(f"the GLOBAL order does not put q before k: {g_order}")
    if not before(l_order, "k_proj", "q_proj"):
        fails.append(f"the LOCAL order does not put k before q: {l_order}")
    if "gate_proj" in l_order and "down_proj" in l_order:
        if not before(g_order, "gate_proj", "down_proj"):
            fails.append("the GLOBAL order does not put gate before down")
        if not before(l_order, "down_proj", "gate_proj"):
            fails.append("the LOCAL order does not put down before gate")

    # And the WEIGHTS must actually differ, not just the ordering -- a policy
    # that reorders and lands on the same byte map discriminates nothing.
    gw = oa.per_param_weights("ORGAN_WEIGHTED")
    lw = oa.per_param_weights("LOCAL", LOCAL_DEFICIT)
    shared = set(gw) & set(lw)
    if not shared:
        fails.append("the two policies share no organ -- they cannot be compared")
    elif all(abs(gw[k] - lw[k]) < 1e-9 for k in shared):
        fails.append("LOCAL and GLOBAL weights are identical on every shared organ")

    # UNIFORM must remain flat, and LOCAL must not silently become UNIFORM.
    uw = oa.per_param_weights("UNIFORM")
    if len(set(round(v, 9) for v in uw.values())) != 1:
        fails.append("UNIFORM is no longer flat")
    if len(set(round(lw[k], 9) for k in lw)) == 1:
        fails.append("LOCAL produced a flat weight map -- it is UNIFORM wearing another name")

    # A LOCAL policy with no deficits given must REFUSE, not fall back to the
    # global prior. Falling back would run a global-prior arm and label it local.
    try:
        oa.per_param_weights("LOCAL")
        fails.append("LOCAL with no deficit map did not refuse -- it would run the "
                     "global prior under a local label")
    except (ValueError, TypeError):
        pass

    for f in fails:
        print("FAIL:", f)
    print(f"{'FAILED' if fails else 'PASS'} — LOCAL exists, differs from GLOBAL on both inverted "
          f"pairs, and refuses without its own anatomy")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
