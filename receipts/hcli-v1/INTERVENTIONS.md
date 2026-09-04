# Human intervention ledger

One JSON object per line. The metric reads this; nothing writes it automatically,
because a human action that nothing recorded is exactly the kind that inflates an
autonomy number.

    {"ts": 1788500000, "kind": "HUMAN_RESTARTS", "causal": false, "note": "..."}

`kind` is one of HUMAN_GOAL_WRITES, HUMAN_RESTARTS, HUMAN_PATCHES,
HUMAN_TARGET_HINTS, HUMAN_VERIFIER_CHANGES, EXTERNAL_MODEL_SCOUTS,
EXTERNAL_MODEL_REPAIRS.

`causal` is the load-bearing field. A human watching is not a human fixing:

  * observation      causal=false   read a receipt, ran the metric, tailed a log
  * intervention     causal=false   restarted a daemon, resubmitted the same goal
  * CAUSAL REPAIR    causal=true    changed code, changed the verifier, or told
                                    the model where to look, in a way without
                                    which the unit would not have been accepted

Writing the goal is causal=false: a goal is the task, not the answer. Naming the
exact line, anchor or mutation IS causal=true -- that is doing the work.
