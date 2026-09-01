#!/usr/bin/env python3
"""Mission-kernel compaction, graded on continuity not similarity.

receipts/future/HCLI_MISSION_KERNEL.json is the resident's durable memory.
This module reduces it by dropping repeated prose, obsolete deliberation and
degenerate tails, then asks whether a reader of the compacted kernel could
continue the mission correctly.

THE GRADE IS NOT TEXTUAL SIMILARITY. A near-identical kernel that lost one
scar fails; a rewritten kernel that kept every verdict, scar, target, tried
param and wake condition passes.

Negative controls (remove a live hypothesis, remove a scar, alter the
target, remove the latest refutation, remove a wake condition) must each
be caught. A compactor that passes its own corrupted input is worthless.

This module does not choose a hypothesis, a representation, or an
experiment. It copies load-bearing state and discards the rest.

    python3 tools/future/hcli_compactor.py --build
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import HARDWARE_FIELDS, REPO, git, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/hcli_compactor.py"
RECEIPT_NAME = "HCLI_COMPACTOR.json"
KERNEL_REL = "receipts/future/HCLI_MISSION_KERNEL.json"
SCHEMA = "hawking.future.hcli_compactor.v1"

REQUIRED_KERNEL_FIELDS = ("objective", "measured_state", "hypotheses", "scars")
HYPOTHESIS_REQUIRED = ("id", "verdict")

# Continuity loss kinds. The evaluator names what a reader would get wrong,
# not how much the JSON changed.
LIVE_HYPOTHESIS = "live_hypothesis"
SCAR = "scar"
CURRENT_TARGET = "current_target"
LATEST_REFUTATION = "latest_refutation"
WAKE_CONDITION = "wake_condition"
OBJECTIVE = "objective"
MEASURED_STATE = "measured_state"
TRIED_PARAMS = "tried_params"
UNSUPPORTED_REQUEST = "unsupported_request"
NEXT_WORK = "next_work"
ACTIVE_WORK = "active_work"
HYPOTHESIS_VERDICT = "hypothesis_verdict"
MEASUREMENT = "measurement"
OBSERVATION = "observation"

REFUTED_VERDICTS = frozenset({"REFUTED", "FALSIFIED", "BURIED"})

# The not_capability string is copied onto every PERTURB result. One copy
# at kernel level is enough for a reader; the rest is repeated prose.
CAVEAT_KEY = "not_capability"


class CompactorRefused(RuntimeError):
    """The kernel is missing or malformed; compaction will not invent state."""


def _nbytes(obj: Any) -> int:
    return len(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _as_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return obj
    raise CompactorRefused(f"expected a list, got {type(obj).__name__}")


def _worktree_roots() -> list[Path]:
    text = git("worktree", "list", "--porcelain")
    roots: list[Path] = []
    for line in text.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line[len("worktree "):]))
    return roots


def kernel_path() -> Path:
    """The live kernel is untracked and may live in the canonical worktree.

    Sparse-checkout lanes do not copy untracked files, so REPO / KERNEL_REL
    can be absent here while the sovereign loop is still writing it next door.
    We read; we never write that path.
    """
    here = REPO / KERNEL_REL
    if here.is_file():
        return here
    found: list[Path] = []
    for root in _worktree_roots():
        p = root / KERNEL_REL
        if p.is_file() and p != here:
            found.append(p)
    if not found:
        raise CompactorRefused(
            f"{KERNEL_REL} is not on disk. The mission kernel IS the "
            "resident's memory; compacting a missing kernel would invent it"
        )
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0]


def load_kernel() -> dict[str, Any]:
    p = kernel_path()
    last_err: Exception | None = None
    for _ in range(3):
        try:
            raw = p.read_text()
        except OSError as exc:
            raise CompactorRefused(f"{KERNEL_REL} cannot be read: {exc}") from exc
        try:
            k = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_err = exc
            time.sleep(0.05)
            continue
        if not isinstance(k, dict):
            raise CompactorRefused(f"{KERNEL_REL} is not a JSON object")
        return k
    raise CompactorRefused(f"{KERNEL_REL} is not valid JSON: {last_err}")


def require_kernel(k: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(k, dict):
        raise CompactorRefused("kernel is not a JSON object")
    missing = [f for f in REQUIRED_KERNEL_FIELDS if f not in k]
    if missing:
        raise CompactorRefused(
            f"kernel is missing {missing}; refusing to default load-bearing state"
        )
    hyps = k["hypotheses"]
    if not isinstance(hyps, list):
        raise CompactorRefused("hypotheses is not a list")
    for i, h in enumerate(hyps):
        if not isinstance(h, dict):
            raise CompactorRefused(f"hypotheses[{i}] is not an object")
        for f in HYPOTHESIS_REQUIRED:
            if f not in h:
                raise CompactorRefused(
                    f"hypotheses[{i}] is missing {f}; a hypothesis without a "
                    "verdict is not durable memory"
                )
    if not isinstance(k["scars"], list):
        raise CompactorRefused("scars is not a list")
    if not isinstance(k["measured_state"], dict):
        raise CompactorRefused("measured_state is not an object")
    if not isinstance(k["objective"], str) or not k["objective"].strip():
        raise CompactorRefused("objective is missing")
    return k


def _scar_key(s: Any) -> str:
    if isinstance(s, str):
        return s
    return _canon(s)


def _hyp_key(h: dict[str, Any]) -> tuple[Any, Any]:
    return (h.get("id"), h.get("claim"))


def _param_key(p: dict[str, Any] | None) -> str:
    p = p or {}
    return f"{p.get('tensor')}/L{p.get('layer')}/{p.get('side')}/{p.get('fraction')}"


def hypothesis_verdicts(k: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for h in _as_list(k.get("hypotheses")):
        if isinstance(h, dict) and "id" in h and "verdict" in h:
            out[str(h["id"])] = str(h["verdict"])
    return out


def latest_refutation(k: dict[str, Any]) -> dict[str, Any] | None:
    refs = [
        h for h in _as_list(k.get("hypotheses"))
        if isinstance(h, dict) and str(h.get("verdict", "")).upper() in REFUTED_VERDICTS
    ]
    return copy.deepcopy(refs[-1]) if refs else None


def live_hypotheses(k: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer an explicit key, even if empty, so a corruption that empties it is visible."""
    if "live_hypotheses" in k:
        return [h for h in _as_list(k.get("live_hypotheses")) if isinstance(h, dict)]
    for it in reversed(_as_list(k.get("iterations"))):
        if not isinstance(it, dict):
            continue
        lh = it.get("live_hypotheses")
        if lh:
            return [h for h in _as_list(lh) if isinstance(h, dict)]
    open_ = [
        h for h in _as_list(k.get("hypotheses"))
        if isinstance(h, dict)
        and str(h.get("verdict", "")).upper() not in REFUTED_VERDICTS
        and str(h.get("verdict", "")).upper() not in {"CONFIRMED", "ACCEPTED"}
    ]
    return open_


def current_target(k: dict[str, Any]) -> Any:
    if "current_target" in k and k["current_target"] not in (None, ""):
        return k["current_target"]
    if k.get("frontier"):
        return k["frontier"]
    raise CompactorRefused(
        "kernel has no current_target and no frontier; refusing to invent one"
    )


def wake_conditions(k: dict[str, Any]) -> list[str]:
    if "wake_conditions" in k:
        return [str(w) for w in _as_list(k.get("wake_conditions"))]
    wakes: list[str] = []
    for d in _as_list(k.get("harness_defects_found_and_fixed")):
        wakes.append(str(d))
    for s in _as_list(k.get("scars")):
        if isinstance(s, dict) and s.get("reopen_condition"):
            wakes.append(str(s["reopen_condition"]))
    return wakes


def tried_params(k: dict[str, Any]) -> list[str]:
    if "tried_params" in k:
        return [str(x) for x in _as_list(k.get("tried_params"))]
    out: list[str] = []
    for it in _as_list(k.get("iterations")):
        if not isinstance(it, dict):
            continue
        for r in _as_list(it.get("results")):
            if isinstance(r, dict) and r.get("params"):
                out.append(_param_key(r["params"]))
    return out


def unsupported_requests(k: dict[str, Any]) -> list[Any]:
    if "unsupported_requests" in k:
        return list(_as_list(k.get("unsupported_requests")))
    out: list[Any] = []
    for it in _as_list(k.get("iterations")):
        if not isinstance(it, dict):
            continue
        v = it.get("validation") or {}
        if isinstance(v, dict):
            for rej in _as_list(v.get("rejected")):
                out.append(rej)
    return out


def active_work(k: dict[str, Any]) -> list[Any]:
    if "active_work" in k:
        return list(_as_list(k.get("active_work")))
    for it in reversed(_as_list(k.get("iterations"))):
        if not isinstance(it, dict):
            continue
        ran = [
            {"type": r.get("type"), "params": r.get("params")}
            for r in _as_list(it.get("results"))
            if isinstance(r, dict) and r.get("ran")
        ]
        if ran:
            return ran
    return []


def next_work(k: dict[str, Any]) -> list[Any]:
    if "next_work" in k:
        return list(_as_list(k.get("next_work")))
    return [
        {
            "id": h.get("id"),
            "cheapest_falsifier": h.get("cheapest_falsifier"),
        }
        for h in live_hypotheses(k)
    ]


def observations(k: dict[str, Any]) -> list[Any]:
    return list(_as_list(k.get("observations")))


def measurements(k: dict[str, Any]) -> list[tuple[Any, Any, Any]]:
    out: list[tuple[Any, Any, Any]] = []
    for it in _as_list(k.get("iterations")):
        if not isinstance(it, dict):
            continue
        for r in _as_list(it.get("results")):
            if not isinstance(r, dict) or not r.get("ran"):
                continue
            res = r.get("result") if isinstance(r.get("result"), dict) else {}
            out.append((
                _param_key(r.get("params") if isinstance(r.get("params"), dict) else {}),
                res.get("damage"),
                res.get("hidden_cosine_after_2_layers"),
            ))
    return out


def continuity_surface(k: dict[str, Any]) -> dict[str, Any]:
    """Everything a reader needs to continue. Not the prose around it."""
    return {
        "objective": k.get("objective"),
        "current_target": current_target(k) if (k.get("current_target") or k.get("frontier")) else None,
        "measured_state": k.get("measured_state"),
        "hypothesis_verdicts": hypothesis_verdicts(k),
        "scars": [_scar_key(s) for s in _as_list(k.get("scars"))],
        "live_hypotheses": [_canon(h) for h in live_hypotheses(k)],
        "latest_refutation_id": (latest_refutation(k) or {}).get("id"),
        "tried_params": tried_params(k),
        "unsupported_requests": [_canon(x) for x in unsupported_requests(k)],
        "active_work": [_canon(x) for x in active_work(k)],
        "next_work": [_canon(x) for x in next_work(k)],
        "wake_conditions": wake_conditions(k),
        "observations": [_canon(x) for x in observations(k)],
        "measurements": measurements(k),
    }


def evaluate_continuity(authority: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Could a reader of candidate continue as if they had authority?

    Authority is the pre-compaction kernel. Grading candidate against itself
    (or against the corrupted input a worthless compactor was handed) is not
    this function's job; the caller passes the real kernel as authority.
    """
    require_kernel(authority)
    if not isinstance(candidate, dict):
        return {
            "ok": False,
            "could_continue": False,
            "grade": "FAILED_CONTINUITY",
            "losses": [{"kind": "candidate", "missing": "candidate is not an object"}],
            "loss_kinds": ["candidate"],
            "not_the_grade": "textual similarity is not the grade",
        }
    auth = continuity_surface(authority)
    try:
        cand = continuity_surface(candidate)
    except CompactorRefused as exc:
        return {
            "ok": False,
            "could_continue": False,
            "grade": "FAILED_CONTINUITY",
            "losses": [{"kind": "candidate", "missing": str(exc)}],
            "loss_kinds": ["candidate"],
            "not_the_grade": "textual similarity is not the grade",
        }

    losses: list[dict[str, Any]] = []

    if auth["objective"] != cand["objective"]:
        losses.append({"kind": OBJECTIVE, "missing": auth["objective"]})
    if _canon(auth["current_target"]) != _canon(cand["current_target"]):
        losses.append({"kind": CURRENT_TARGET, "missing": auth["current_target"]})
    if _canon(auth["measured_state"]) != _canon(cand["measured_state"]):
        losses.append({"kind": MEASURED_STATE, "missing": "measured_state"})

    for hid, verdict in auth["hypothesis_verdicts"].items():
        got = cand["hypothesis_verdicts"].get(hid)
        if got != verdict:
            losses.append({
                "kind": HYPOTHESIS_VERDICT,
                "missing": hid,
                "authority_verdict": verdict,
                "candidate_verdict": got,
            })

    auth_scars = set(auth["scars"])
    cand_scars = set(cand["scars"])
    for s in sorted(auth_scars - cand_scars):
        losses.append({"kind": SCAR, "missing": s})

    auth_live = set(auth["live_hypotheses"])
    cand_live = set(cand["live_hypotheses"])
    for h in sorted(auth_live - cand_live):
        losses.append({"kind": LIVE_HYPOTHESIS, "missing": h})

    lid = auth["latest_refutation_id"]
    if lid is not None:
        got = cand["hypothesis_verdicts"].get(lid)
        auth_v = auth["hypothesis_verdicts"].get(lid)
        if got != auth_v:
            losses.append({
                "kind": LATEST_REFUTATION,
                "missing": lid,
                "authority_verdict": auth_v,
                "candidate_verdict": got,
            })

    auth_wakes = set(auth["wake_conditions"])
    cand_wakes = set(cand["wake_conditions"])
    for w in sorted(auth_wakes - cand_wakes):
        losses.append({"kind": WAKE_CONDITION, "missing": w})

    auth_tried = set(auth["tried_params"])
    cand_tried = set(cand["tried_params"])
    for t in sorted(auth_tried - cand_tried):
        losses.append({"kind": TRIED_PARAMS, "missing": t})

    auth_unsup = set(auth["unsupported_requests"])
    cand_unsup = set(cand["unsupported_requests"])
    for u in sorted(auth_unsup - cand_unsup):
        losses.append({"kind": UNSUPPORTED_REQUEST, "missing": u})

    if set(auth["next_work"]) - set(cand["next_work"]):
        for n in sorted(set(auth["next_work"]) - set(cand["next_work"])):
            losses.append({"kind": NEXT_WORK, "missing": n})

    if set(auth["active_work"]) - set(cand["active_work"]):
        for a in sorted(set(auth["active_work"]) - set(cand["active_work"])):
            losses.append({"kind": ACTIVE_WORK, "missing": a})

    auth_obs = set(auth["observations"])
    cand_obs = set(cand["observations"])
    for o in sorted(auth_obs - cand_obs):
        losses.append({"kind": OBSERVATION, "missing": o})

    auth_m = Counter(auth["measurements"])
    cand_m = Counter(cand["measurements"])
    if auth_m - cand_m:
        losses.append({
            "kind": MEASUREMENT,
            "missing": [list(m) for m in (auth_m - cand_m).elements()],
        })

    kinds = []
    for loss in losses:
        if loss["kind"] not in kinds:
            kinds.append(loss["kind"])
    ok = not losses
    return {
        "ok": ok,
        "could_continue": ok,
        "grade": "CONTINUITY" if ok else "FAILED_CONTINUITY",
        "losses": losses,
        "loss_kinds": kinds,
        "n_losses": len(losses),
        "not_the_grade": "textual similarity is not the grade",
    }


def _is_empty_degenerate(it: dict[str, Any]) -> bool:
    if not isinstance(it, dict):
        return True
    ran = [
        r for r in _as_list(it.get("results"))
        if isinstance(r, dict) and r.get("ran")
    ]
    if ran:
        return False
    if it.get("parsed") is False:
        return True
    if it.get("degenerated") is True:
        return True
    return False


def _lift_caveat(results: list[Any], discarded: dict[str, Any]) -> str | None:
    texts: list[str] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        res = r.get("result")
        if isinstance(res, dict) and CAVEAT_KEY in res and res[CAVEAT_KEY]:
            texts.append(str(res[CAVEAT_KEY]))
    if not texts:
        return None
    # Only lift when every copy is the same string; mixed caveats stay put.
    if len(set(texts)) != 1:
        return None
    caveat = texts[0]
    n = 0
    for r in results:
        if not isinstance(r, dict):
            continue
        res = r.get("result")
        if isinstance(res, dict) and CAVEAT_KEY in res:
            discarded["repeated_prose"]["n"] += 1
            discarded["repeated_prose"]["bytes"] += _nbytes({CAVEAT_KEY: res[CAVEAT_KEY]})
            del res[CAVEAT_KEY]
            n += 1
    return caveat if n else None


def _strip_nulls(obj: Any, discarded: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        out = {}
        for key, val in obj.items():
            if val is None:
                discarded["repeated_prose"]["n"] += 1
                discarded["repeated_prose"]["bytes"] += _nbytes({key: None})
                continue
            out[key] = _strip_nulls(val, discarded)
        return out
    if isinstance(obj, list):
        return [_strip_nulls(x, discarded) for x in obj]
    return obj


def _strip_iteration(
    it: dict[str, Any],
    *,
    keep_deliberation: bool,
    discarded: dict[str, Any],
) -> dict[str, Any]:
    keep_always = {
        "n", "parsed", "degenerated", "results", "results_summary",
        "validation", "t_s",
    }
    keep_last = {"belief_update", "live_hypotheses"}
    out: dict[str, Any] = {}
    for key, val in it.items():
        if key in keep_always or (keep_deliberation and key in keep_last):
            out[key] = copy.deepcopy(val)
            continue
        if key in keep_last:
            discarded["obsolete_deliberation"]["n"] += 1
            discarded["obsolete_deliberation"]["bytes"] += _nbytes({key: val})
            continue
        # Telemetry and leftover prose: discard.
        discarded["repeated_prose"]["n"] += 1
        discarded["repeated_prose"]["bytes"] += _nbytes({key: val})
    if "results" in out:
        out["results"] = _strip_nulls(out["results"], discarded)
        # results_summary is derived from results; keep it only on the last
        # turn (context_pack reads iterations[-1].results_summary).
        if not keep_deliberation and "results_summary" in out:
            discarded["repeated_prose"]["n"] += 1
            discarded["repeated_prose"]["bytes"] += _nbytes(
                {"results_summary": out["results_summary"]}
            )
            del out["results_summary"]
    return out


def compact_with_stats(k: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    require_kernel(k)
    discarded = {
        "repeated_prose": {"n": 0, "bytes": 0},
        "obsolete_deliberation": {"n": 0, "bytes": 0},
        "degenerate_tails": {"n": 0, "bytes": 0},
    }
    before = _nbytes(k)

    out: dict[str, Any] = {}
    copy_keys = (
        "schema", "resident_mode", "objective", "frontier", "measured_state",
        "authority", "executable_work_types", "scars",
        "scars_bind_methods_not_goals", "hypotheses", "observations",
        "tried_params", "created_unix", "updated_unix",
        "harness_defects_found_and_fixed",
    )
    for key in copy_keys:
        if key in k:
            out[key] = copy.deepcopy(k[key])

    # Materialise load-bearing fields the original stores implicitly so a
    # reader (and the evaluator) does not have to reconstruct them. Values
    # are copied from the kernel; nothing is chosen.
    out["current_target"] = copy.deepcopy(current_target(k))
    out["live_hypotheses"] = copy.deepcopy(live_hypotheses(k))
    out["wake_conditions"] = list(wake_conditions(k))
    out["unsupported_requests"] = copy.deepcopy(unsupported_requests(k))
    out["active_work"] = copy.deepcopy(active_work(k))
    out["next_work"] = copy.deepcopy(next_work(k))
    if "tried_params" not in out:
        out["tried_params"] = tried_params(k)

    iterations = [it for it in _as_list(k.get("iterations")) if isinstance(it, dict)]
    kept_full: list[dict[str, Any]] = []
    for it in iterations:
        if _is_empty_degenerate(it):
            discarded["degenerate_tails"]["n"] += 1
            discarded["degenerate_tails"]["bytes"] += _nbytes(it)
            continue
        kept_full.append(it)

    stripped: list[dict[str, Any]] = []
    last_i = len(kept_full) - 1
    caveats: list[str] = []
    for i, it in enumerate(kept_full):
        row = _strip_iteration(it, keep_deliberation=(i == last_i), discarded=discarded)
        if isinstance(row.get("results"), list):
            caveat = _lift_caveat(row["results"], discarded)
            if caveat:
                caveats.append(caveat)
        stripped.append(row)
    out["iterations"] = stripped
    if caveats and len(set(caveats)) == 1:
        out["measurement_caveat"] = caveats[0]

    after = _nbytes(out)
    if before <= 0:
        raise CompactorRefused("authority kernel serialised to zero bytes")
    stats = {
        "bytes_before": before,
        "bytes_after": after,
        "bytes_saved": before - after,
        "compression_ratio": round(after / before, 4),
        "discarded_by_category": discarded,
        "n_iterations_before": len(iterations),
        "n_iterations_after": len(stripped),
    }
    return out, stats


def compact(k: dict[str, Any]) -> dict[str, Any]:
    compacted, _stats = compact_with_stats(k)
    return compacted


def corrupt_remove_live_hypothesis(k: dict[str, Any]) -> dict[str, Any]:
    k = copy.deepcopy(k)
    live = live_hypotheses(k)
    if not live:
        raise CompactorRefused("no live hypothesis to remove")
    k["live_hypotheses"] = live[1:]
    return k


def corrupt_remove_scar(k: dict[str, Any]) -> dict[str, Any]:
    k = copy.deepcopy(k)
    scars = list(_as_list(k.get("scars")))
    if not scars:
        raise CompactorRefused("no scar to remove")
    k["scars"] = scars[:-1]
    return k


def corrupt_alter_target(k: dict[str, Any]) -> dict[str, Any]:
    k = copy.deepcopy(k)
    k["current_target"] = "__ALTERED_TARGET__"
    k["frontier"] = "__ALTERED_TARGET__"
    return k


def corrupt_remove_latest_refutation(k: dict[str, Any]) -> dict[str, Any]:
    k = copy.deepcopy(k)
    ref = latest_refutation(k)
    if ref is None or "id" not in ref:
        raise CompactorRefused("no refutation to remove")
    rid = ref["id"]
    k["hypotheses"] = [
        h for h in _as_list(k.get("hypotheses"))
        if not (isinstance(h, dict) and h.get("id") == rid)
    ]
    return k


def corrupt_remove_wake_condition(k: dict[str, Any]) -> dict[str, Any]:
    k = copy.deepcopy(k)
    wakes = list(wake_conditions(k))
    if not wakes:
        raise CompactorRefused("no wake condition to remove")
    k["wake_conditions"] = wakes[1:]
    return k


CORRUPTIONS: dict[str, Any] = {
    "remove_live_hypothesis": corrupt_remove_live_hypothesis,
    "remove_scar": corrupt_remove_scar,
    "alter_target": corrupt_alter_target,
    "remove_latest_refutation": corrupt_remove_latest_refutation,
    "remove_wake_condition": corrupt_remove_wake_condition,
}

PRIMARY_LOSS = {
    "remove_live_hypothesis": LIVE_HYPOTHESIS,
    "remove_scar": SCAR,
    "alter_target": CURRENT_TARGET,
    "remove_latest_refutation": LATEST_REFUTATION,
    "remove_wake_condition": WAKE_CONDITION,
}


def apply_corruption(k: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in CORRUPTIONS:
        raise CompactorRefused(f"unknown corruption {name!r}")
    return CORRUPTIONS[name](k)


def negative_controls(authority: dict[str, Any], compacted: dict[str, Any]) -> dict[str, Any]:
    """Each corruption must be caught against the REAL kernel, not against itself."""
    rows: dict[str, Any] = {}
    for name in CORRUPTIONS:
        bad = apply_corruption(compacted, name)
        vs_authority = evaluate_continuity(authority, bad)
        vs_self = evaluate_continuity(bad, compact(bad)) if _can_compact(bad) else {
            "ok": None, "loss_kinds": [],
        }
        rows[name] = {
            "caught": vs_authority["ok"] is False
                      and PRIMARY_LOSS[name] in vs_authority["loss_kinds"],
            "primary_loss": PRIMARY_LOSS[name],
            "loss_kinds": vs_authority["loss_kinds"],
            "n_losses": vs_authority["n_losses"],
            "worthless_self_grade_would_pass": vs_self.get("ok") is True,
        }
    return rows


def _can_compact(k: dict[str, Any]) -> bool:
    try:
        require_kernel(k)
        return True
    except CompactorRefused:
        return False


def build() -> dict[str, Any]:
    authority = load_kernel()
    compacted, stats = compact_with_stats(authority)
    continuity = evaluate_continuity(authority, compacted)
    controls = negative_controls(authority, compacted)
    all_caught = all(row["caught"] for row in controls.values())
    return {
        "schema": SCHEMA,
        "question": (
            "Can mission state be compacted without the resident losing "
            "what it knows?"
        ),
        "answer": (
            "YES - a reader of the compacted kernel can continue correctly"
            if continuity["ok"] and all_caught else
            "NO - continuity failed or a negative control was not caught"
        ),
        "authority": "receipts/future/HCLI_MISSION_KERNEL.json",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "kernel_resolved_to": str(kernel_path()),
        "bytes_before": stats["bytes_before"],
        "bytes_after": stats["bytes_after"],
        "bytes_saved": stats["bytes_saved"],
        "compression_ratio": stats["compression_ratio"],
        "discarded_by_category": stats["discarded_by_category"],
        "n_iterations_before": stats["n_iterations_before"],
        "n_iterations_after": stats["n_iterations_after"],
        "continuity": continuity,
        "negative_controls": controls,
        "all_negative_controls_caught": all_caught,
        "preserved": {
            "objective": compacted.get("objective"),
            "current_target": compacted.get("current_target"),
            "hypothesis_verdicts": hypothesis_verdicts(compacted),
            "n_scars": len(compacted.get("scars") or []),
            "scars": compacted.get("scars"),
            "n_live_hypotheses": len(compacted.get("live_hypotheses") or []),
            "tried_params": compacted.get("tried_params"),
            "n_wake_conditions": len(compacted.get("wake_conditions") or []),
            "n_unsupported_requests": len(compacted.get("unsupported_requests") or []),
        },
        "compacted_kernel": _cite_hardware(compacted),
        "what_this_module_does_not_do": (
            "choose the next hypothesis, representation, or experiment. "
            "The resident owns those. Compaction copies load-bearing state "
            "and discards repeated prose, obsolete deliberation, and "
            "degenerate tails."
        ),
        "grade_is_not": "textual similarity",
    }


def _cite_hardware(node: Any) -> Any:
    """Copy the kernel, but CITE its hardware numbers instead of restating them.

    The kernel legitimately carries measured_state.tps and token_ms: that is
    what the context pack tells the resident about the body it is reasoning
    over, and a stale one is how it spent hours believing 36.644 when the
    promotion measured 45.566. But this module is a SIDECAR with no GPU
    authority, so `write_receipt` refuses any receipt that states a hardware
    number as its own -- and it was right to refuse this one.

    Both laws hold at once by keeping the value where the resident reads it
    (the live kernel, untouched) and replacing it here with a pointer to the
    receipt that does have the authority. Continuity is unaffected: the
    continuity grade compares the in-memory compacted kernel against the
    authoritative one, not this rendering.
    """
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                out[key] = None
                out[f"{key}_cited_from"] = node.get("refreshed_from") or node.get(
                    "source"
                ) or "unattributed: kernel carried no refreshed_from"
            else:
                out[key] = _cite_hardware(value)
        return out
    if isinstance(node, list):
        return [_cite_hardware(v) for v in node]
    return node


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({
        "compression_ratio": doc["compression_ratio"],
        "continuity": doc["continuity"]["grade"],
        "all_negative_controls_caught": doc["all_negative_controls_caught"],
        "discarded_by_category": doc["discarded_by_category"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
