"""The generation pass must REFUSE on real evidence, not perform a refusal.

`refused_on_evidence: 0` was not a broken filter. The loop asked the negative
index whether a python module name was a dead hypothesis family, which is a
category error the index can only ever answer no to. And its other work sources
-- the frontier, and the Codex candidate queue -- are pruned before the sidecar
ever sees them, so consuming those can honestly report zero refusals forever.

The danger in fixing that is the obvious one: proposing only ideas already known
to be dead, so the counter goes up and nothing was decided. These tests are
weighted against exactly that -- the proposal space must be the fixed taxonomy,
most of it must SURVIVE, and every rejection must cite a scar file that exists.
"""
import json

import pytest

from tools.future import autonomy_run as ar
from tools.future import autonomy_trial as at
from tools.future import negative_index as ni
from tools.future._common import REPO, git


def _source_resolves(src: str) -> bool:
    """A citation must resolve. Sparse checkout absence is not git absence."""
    if not src:
        return False
    if (REPO / src).exists():
        return True
    listed = git("ls-tree", "-r", "--name-only", "HEAD", "--", src)
    return src in listed.splitlines() or listed.strip() == src


def test_proposal_space_is_the_fixed_taxonomy_not_the_set_of_dead_ideas():
    """NEGATIVE CONTROL against circularity.

    If the generator drew its families from the scars, every proposal would die
    and the pass would decide nothing. The taxonomy is a fixed vocabulary, so
    most proposals must survive.
    """
    assert set(ar.FAMILY_TAXONOMY) == set(ni.FAMILY_SLUGS)
    scars = ni.ingest()
    families_with_scars = {
        s.hypothesis_family for s in scars if s.hypothesis_family != ni.UNRECORDED
    }
    unscarred = set(ar.FAMILY_TAXONOMY) - families_with_scars
    assert unscarred, "every proposable family carries a scar; the space is circular"


def test_live_parents_are_canonical_slugs_or_no_refusal_can_ever_fire():
    """A scar is model-targeted: a wrong slug silently refuses nothing, forever."""
    for parent in ar.LIVE_PARENTS:
        assert ni.canon_model(parent) == parent, f"{parent} is not a canonical slug"


def test_the_grid_kills_some_and_spares_most():
    scars = ni.ingest()
    dead = alive = 0
    for parent in ar.LIVE_PARENTS:
        for organ in sorted(set(ar.fs.SCHOOL_ORGAN_SLUG.values())):
            for fam in ar.FAMILY_TAXONOMY:
                if ni.refuse_if_dead(
                    {"model": parent, "organ": organ, "hypothesis_family": fam}, scars
                ):
                    dead += 1
                else:
                    alive += 1
    assert dead > 0, "the filter never fires; a refusal would be theatre"
    assert alive > dead, "more than half the space is dead; the proposals are pre-selected"


def test_every_rejection_cites_a_scar_source_that_exists_on_disk():
    """A citation that does not resolve is an assertion wearing a receipt's clothes."""
    scars = ni.ingest()
    checked = 0
    for parent in ar.LIVE_PARENTS:
        for organ in ("routed_experts", "router", "attention", "mlp"):
            for fam in ar.FAMILY_TAXONOMY:
                dead = ni.refuse_if_dead(
                    {"model": parent, "organ": organ, "hypothesis_family": fam}, scars
                )
                if not dead:
                    continue
                src = str(dead.get("source_path") or "")
                assert src, "refusal with no source_path cannot be cited"
                assert _source_resolves(src), f"cited scar source is absent: {src}"
                assert dead.get("reopen_condition"), "a scar with no reopen condition is a wall"
                checked += 1
                if checked >= 25:
                    return
    assert checked, "no refusal was produced to check"


def test_a_short_run_emits_idea_rejected_the_judge_can_read(tmp_path):
    """End to end: the driver must emit the event shape the trial judge scores."""
    tl = tmp_path / "tl.json"
    ar.run(trial="15m", duration_s=45, timeline=tl)
    doc = json.loads(tl.read_text())
    rejected = [e for e in doc["events"] if e["kind"] == "idea_rejected"]
    assert rejected, "no idea_rejected event; the reject condition cannot be met"
    for event in rejected[:20]:
        assert event["payload"].get("idea"), "rejection did not name the idea"
        cites = event.get("cites") or []
        assert cites and all(c for c in cites), "rejection cited nothing"
    assert doc["summary"]["hypotheses_still_live"] > doc["summary"]["refused_on_evidence"]


def test_run_never_emits_an_idle_event(tmp_path):
    tl = tmp_path / "tl2.json"
    ar.run(trial="15m", duration_s=30, timeline=tl)
    kinds = {e["kind"] for e in json.loads(tl.read_text())["events"]}
    assert not (kinds & {"idle", "awaiting_instructions", "all_tasks_complete"})


def test_the_metal_blocker_is_measured_not_quoted():
    """A repeated blocker line was half false and it scoped the whole campaign.

    "no Metal-capable GPU and no Metal compiler" was carried from a blocker list
    into three places in this driver. The GPU is an M3 Ultra and it is present;
    what is absent is the offline shader compiler. A missing GPU would block
    every physical measurement, while a missing offline compiler blocks
    precompilation -- different scopes, different work unblocked.
    """
    from tools.future import hardware_doctor as hwd

    state = hwd.metal_state()
    why = ar._metal_why()
    assert state["is_a_measurement"] is False, "this is a capability probe, not a timing"
    assert state["runtime_source_compilation"] in {"AVAILABLE", "UNKNOWN"}, (
        "AVAILABLE only after the probe exercised it; UNKNOWN means not run, "
        "never a guess that it fails"
    )
    if state["gpu_present"]:
        assert "no Metal-capable GPU" not in why
        assert state["chip"] in why
    assert ("compiler is absent" in why) == (not state["offline_metal_compiler"])


def test_the_driver_speaks_the_lane_vocabulary_the_frontier_actually_uses():
    """Invented lane names made the frontier's own work silently unreachable.

    The driver declared CPU_ANALYSIS / CPU_VERIFY / CPU_REPRESENTATION / DISK_IO.
    No frontier item requires any of those, so `required_lanes <= available` was
    false for all 31 NEXT_WORK items and next_work() and refill() returned an
    empty list on every call. The loop still had work -- it queued capabilities
    directly -- so nothing looked broken, and the frontier's own work never ran.
    """
    from tools.future import frontiers as frontiers_mod

    assert set(ar.AVAILABLE_LANES) <= set(frontiers_mod.THIS_HOST_LANES)
    assert set(ar.BLOCKED_LANES) == set(frontiers_mod.HARDWARE_LANES)
    assert not (set(ar.AVAILABLE_LANES) & set(ar.BLOCKED_LANES))
    assert frontiers_mod.next_work(ar.AVAILABLE_LANES), (
        "the frontier yields no work for these lanes; the vocabulary is wrong again"
    )


def test_refill_is_exercised_without_waiting_for_starvation(tmp_path):
    """A loop that only asks for work at zero never refills in a full window.

    The 1h trial queued seven multi-hundred-GB verifications and so never once
    reached the end of its queue.
    """
    tl = tmp_path / "tl.json"
    ar.run(trial="15m", duration_s=40, timeline=tl)
    doc = json.loads(tl.read_text())
    kinds = [e["kind"] for e in doc["events"]]
    assert "result_ingested" in kinds, "the judge scores result_ingested, not receipt_ingested"
    refills = [e for e in doc["events"] if e["kind"] == "work_refilled"]
    ingests = [e for e in doc["events"] if e["kind"] == "result_ingested"]
    assert refills, "no refill happened inside the window"
    assert min(e["t_s"] for e in ingests) < max(e["t_s"] for e in refills), (
        "a refill must follow an ingested result to count as refilling after work"
    )
    for event in refills:
        assert event["payload"]["unit_ids"], "a refill that added nothing is not a refill"


def test_an_invalidated_run_is_recorded_and_never_reported_as_a_result():
    """Improving the test and claiming the original interval is the failure here."""
    doc = json.loads(ar.build().read_text())
    rows = doc["invalidated_runs"]
    assert rows, "the killed 1h run must be on the record"
    for row in rows:
        assert row["verdict"] == "INVALIDATED_BY_SUBSTRATE_MUTATION"
        assert row["why"] and row["kept"], "say what was lost and what survives"


def test_the_timeline_the_judge_reads_is_sealed(tmp_path):
    """An unsealed timeline can be edited after the fact by the thing being judged.

    The trial timeline and the mission state were written straight to disk, so
    they carried no seal, no bench block and no gpu_authority field. The
    adversarial attacker flagged all three as P0. The verdict of every autonomy
    trial rests on this file.
    """
    from tools.future._common import seal

    tl = tmp_path / "tl.json"
    ar.run(trial="15m", duration_s=40, timeline=tl)
    doc = json.loads(tl.read_text())

    assert doc.get("seal_sha256"), "the timeline is unsealed"
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"

    claimed = doc["seal_sha256"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    assert seal(dict(body))["seal_sha256"] == claimed, "the seal does not verify"

    # NEGATIVE CONTROL: the seal must actually detect an edit, or it is decoration.
    body["summary"] = dict(body["summary"], launched=99999)
    assert seal(dict(body))["seal_sha256"] != claimed, "an edited timeline still verified"


def test_mission_state_is_sealed_too():
    from tools.future._common import RECEIPTS, seal

    p = RECEIPTS / "AUTONOMY_MISSION_STATE.json"
    if not p.is_file():
        return  # written by a run; the timeline test above carries the guarantee
    doc = json.loads(p.read_text())
    assert doc.get("seal_sha256"), "mission state is unsealed"
    assert doc["bench"]["state"] == "UNKNOWN"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    assert seal(dict(body))["seal_sha256"] == doc["seal_sha256"]


def test_the_trial_queue_is_a_composed_mix_not_whatever_the_connector_exposes():
    """A queue of every bound capability is a checklist, not a trial.

    trial_workload composes a declared mix -- fast specimen science, one
    genuinely long unit, a negative-science query that can refuse, a
    multi-fidelity screen, an HCLI self-optimization unit, an Odyssey-II
    transfer and an Odyssey-III attack -- plus, for the longer trials, pairs
    where one unit's result legitimately reprioritizes another. Replanning is
    what separates a resident from an executor and cannot be demonstrated on
    work that has no dependencies.
    """
    from tools.future import trial_workload as twl

    for trial in ("1h", "3h", "6h"):
        composed = twl.compose(trial)
        units = composed["units"]
        assert units, f"{trial} composed no units"
        # every composed unit must name a module the connector can actually run
        for unit in units:
            module = unit.get("module")
            if module:
                assert module in ar.orch.BINDINGS, (
                    f"{trial} composed {module}, which no binding names"
                )
        roles = {u.get("mix_role") for u in units}
        assert len(roles) > 1, f"{trial} mix collapsed to a single role: {roles}"

    # The longer trials must be able to demonstrate replanning at all.
    assert twl.compose("3h")["n_replan_pairs"] > 0
    assert twl.compose("6h")["n_replan_pairs"] > 0


def test_a_composed_unit_that_needs_a_blocked_resource_sleeps(monkeypatch):
    """Parked, never dropped, and never quietly run anyway."""
    from tools.future import trial_workload as twl

    composed = twl.compose("6h")
    for unit in composed.get("sleeping") or []:
        assert unit.get("resource_class") or unit.get("required_lanes")
        assert unit.get("id")
    for unit in composed["units"]:
        assert not unit.get("gpu_authority"), "a runnable composed unit claimed GPU authority"


def test_detached_started_without_a_live_process_is_refused():
    """NEGATIVE CONTROL: a started event with no live pid is the hardcoded-True failure."""
    doc = {"events": []}
    with pytest.raises(ar.EmitRefused, match="no live pid"):
        ar.emit_detached_started(doc, {"job_id": "ghost"}, t_s=0)
    with pytest.raises(ar.EmitRefused, match="not alive"):
        ar.emit_detached_started(
            doc, {"job_id": "ghost", "pid": 2147483647, "launched_at": 0.0}, t_s=0
        )
    assert not [e for e in doc.get("events") or [] if e.get("kind") == "detached_started"]


def test_detached_started_requires_a_live_pid(tmp_path):
    proc = __import__("subprocess").Popen(
        [__import__("sys").executable, "-c", "import time; time.sleep(8)"]
    )
    try:
        doc = ar.emit_detached_started(
            {"events": []},
            {"job_id": "live-job", "pid": proc.pid, "launched_at": 1.5},
            t_s=0,
        )
        event = doc["events"][-1]
        assert event["kind"] == "detached_started"
        assert event["payload"]["job_id"] == "live-job"
        assert event["payload"]["pid"] == proc.pid
    finally:
        proc.kill()
        proc.wait()


def test_priority_altered_with_unchanged_order_is_refused():
    """NEGATIVE CONTROL: emitting before == after would score a reorder that did not happen."""
    doc = {"events": []}
    same = ["a", "b", "c"]
    with pytest.raises(ar.EmitRefused, match="before == after"):
        ar.emit_priority_altered(doc, same, list(same), t_s=0, cites=["receipts/future/X.json"])
    assert not [e for e in doc.get("events") or [] if e.get("kind") == "priority_altered"]
    doc = ar.emit_priority_altered(
        {"events": []}, ["a", "b"], ["b", "a"], t_s=0, cites=["receipts/future/X.json"]
    )
    event = doc["events"][-1]
    assert event["kind"] == "priority_altered"
    assert event["payload"]["before"] == ["a", "b"]
    assert event["payload"]["after"] == ["b", "a"]
    assert event["payload"]["before"] != event["payload"]["after"]


def test_negative_science_refusal_requires_the_scar_source_path():
    """NEGATIVE CONTROL: a refusal that cites nothing refused nothing."""
    doc = {"events": []}
    query = {"model": "qwen3-80b", "organ": "routed_experts", "hypothesis_family": "x"}
    with pytest.raises(ar.EmitRefused, match="source_path"):
        ar.emit_negative_science_refusal(doc, {"scar_id": "scar-1", "refused": True}, query, t_s=0)
    assert not [e for e in doc.get("events") or [] if e.get("kind") == "negative_science_refusal"]

    scars = ni.ingest()
    dead = None
    for parent in ar.LIVE_PARENTS:
        for organ in ("routed_experts", "router", "attention"):
            for fam in ar.FAMILY_TAXONOMY:
                dead = ni.refuse_if_dead(
                    {"model": parent, "organ": organ, "hypothesis_family": fam}, scars
                )
                if dead and dead.get("source_path"):
                    break
            if dead and dead.get("source_path"):
                break
        if dead and dead.get("source_path"):
            break
    assert dead and dead.get("source_path"), "no real scar refusal available to cite"
    src = dead["source_path"]
    assert _source_resolves(src), f"cited scar source is absent: {src}"
    doc = ar.emit_negative_science_refusal(
        {"events": []},
        dead,
        {"model": dead.get("model"), "organ": dead.get("organ"),
         "hypothesis_family": dead.get("hypothesis_family")},
        t_s=0,
    )
    event = doc["events"][-1]
    assert event["kind"] == "negative_science_refusal"
    assert event["payload"]["source_path"] == src
    assert src in (event.get("cites") or [])


def test_long_detached_jobs_are_ranked_ahead_of_capabilities():
    """Priority 0 is highest. `prio or 99` used to send long jobs to the back."""
    long_job = {
        "long_subprocess": True,
        "capability": "specimen_verify.py",
        "launch": "detached",
        "composed_unit_id": "WU.TORTURE.NO_WAIT.specimen_verify",
    }
    cap_a = {"capability": "negative_index.py", "composed_unit_id": "WU.NEG"}
    cap_b = {"capability": "hcli_self_profile.py", "composed_unit_id": "WU.HCLI"}
    assert ar._detach_priority(long_job) == 0
    assert ar._detach_priority(cap_a) == 2
    ranked = ar.rank_detachable([cap_a, long_job, cap_b])
    assert ranked[0] is long_job
    assert [j["composed_unit_id"] for j in ranked][0] == "WU.TORTURE.NO_WAIT.specimen_verify"


def test_reorder_from_evidence_mutates_the_queue_or_returns_none():
    """The event is a report of a mutation. A no-op must not claim a reorder."""
    cause = {
        "capability": "negative_index.py",
        "composed_unit_id": "WU.CAUSE.neg",
        "frontier_id": "FT.VERIFICATION.negative-index",
    }
    effect = {
        "capability": "ngram_school.py",
        "composed_unit_id": "WU.EFFECT.ngram",
        "frontier_id": "FT.MODEL_REPRESENTATION.ngram-school",
    }
    other = {
        "capability": "freshness.py",
        "frontier_id": "FT.EXPERIMENT_TURNAROUND.refresh",
    }
    queue = [cause, other, effect]
    pairs = [{
        "cause_module": "negative_index.py",
        "effect_module": "ngram_school.py",
        "cause_frontier_id": "FT.VERIFICATION.negative-index",
        "effect_frontier_id": "FT.MODEL_REPRESENTATION.ngram-school",
    }]
    changed = ar.reorder_queue_from_evidence(queue, 1, cause, pairs)
    assert changed is not None
    before, after = changed
    assert before != after
    assert queue[1]["capability"] == "ngram_school.py"
    assert ar.reorder_queue_from_evidence(queue, 1, cause, pairs) is None


def _overlap_from_timestamps(events):
    """Interval overlap from started_at/finished_at, not from adjacency of kinds."""
    starts, ends = {}, {}
    for event in events:
        payload = event.get("payload") or {}
        jid = payload.get("job_id")
        if not jid:
            continue
        # started_at / finished_at are epoch seconds; t_s is trial-relative.
        # Falling back from one to the other compares 1788141337.2 against 35.0
        # and reports a real overlap as negative. Only epoch stamps are used;
        # a job with no finish stamp is treated as still open.
        if event.get("kind") == "detached_started":
            stamp = payload.get("started_at")
            if stamp is not None:
                starts[jid] = float(stamp)
        elif event.get("kind") in {"detached_completed", "detached_failed"}:
            stamp = payload.get("finished_at")
            if stamp is not None:
                ends[jid] = float(stamp)
    jobs = list(starts)
    for i, a in enumerate(jobs):
        for b in jobs[i + 1 :]:
            sa, sb = starts[a], starts[b]
            ea = ends.get(a, sa + 1e9)
            eb = ends.get(b, sb + 1e9)
            overlap = min(ea, eb) - max(sa, sb)
            if overlap > 0:
                return True, a, b, overlap
    return False, None, None, 0.0


@pytest.fixture(scope="module")
def short_loop_timeline(tmp_path_factory):
    """One real driver loop. The four condition tests all read this run.

    trial=15m uses the same emit sites as 30m; power_torture.compose() alone
    can consume a short duration_s before the loop ever starts.
    """
    tl = tmp_path_factory.mktemp("short_loop") / "timeline.json"
    ar.run(trial="15m", duration_s=25, timeline=tl)
    return json.loads(tl.read_text())


def test_refill_work_emits_work_refilled_after_an_ingest(short_loop_timeline):
    """eval_refill_work: work_refilled after min result_ingested, with unit ids."""
    doc = short_loop_timeline
    verdict = at.eval_refill_work(at.TimelineView(doc, "15m"))
    assert verdict["met"], verdict.get("detail")
    ingests = [e for e in doc["events"] if e["kind"] == "result_ingested"]
    refills = [e for e in doc["events"] if e["kind"] == "work_refilled"]
    assert ingests and refills
    t_ing = min(int(e.get("t_s") or 0) for e in ingests)
    later = [e for e in refills if int(e.get("t_s") or 0) > t_ing]
    assert later, "refill did not follow an ingest (same-second refill is not after)"
    for event in later:
        payload = event.get("payload") or {}
        ids = list(payload.get("unit_ids") or []) + list(event.get("cites") or [])
        assert ids or payload.get("unit"), "a refill that added nothing is not a refill"


def test_overlap_detached_work_starts_two_live_jobs(short_loop_timeline):
    """eval_overlap_detached_work: two detached_started intervals open at once."""
    doc = short_loop_timeline
    events = doc["events"]
    verdict = at.eval_overlap_detached_work(at.TimelineView(doc, "15m"))
    assert verdict["met"], verdict.get("detail")
    started = [e for e in events if e["kind"] == "detached_started"]
    assert len(started) >= 2, "need two real detached jobs, not a single handle"
    for event in started:
        pid = event["payload"].get("pid")
        assert isinstance(pid, int) and pid > 0
        assert event["payload"].get("job_id")
        assert event["payload"].get("started_at")
    overlapped, _a, _b, overlap = _overlap_from_timestamps(events)
    assert overlapped, "detached_started events were adjacent, not an open interval"
    assert overlap > 0


def test_use_negative_science_emits_query_and_refusal(short_loop_timeline):
    """eval_use_negative_science: query/refusal with query, source_path, or cites."""
    doc = short_loop_timeline
    verdict = at.eval_use_negative_science(at.TimelineView(doc, "15m"))
    assert verdict["met"], verdict.get("detail")
    events = doc["events"]
    refusals = [e for e in events if e["kind"] == "negative_science_refusal"]
    queries = [e for e in events if e["kind"] == "negative_science_query"]
    assert queries, "consulted the index without emitting the query the judge scores"
    assert refusals, "refused ideas without emitting negative_science_refusal"
    for event in queries:
        payload = event.get("payload") or {}
        assert payload.get("query") or event.get("cites") or payload.get("source_path")
    for event in refusals[:20]:
        src = event["payload"].get("source_path")
        assert src, "refusal carried no scar source path"
        assert _source_resolves(src), f"cited scar source is absent: {src}"
        assert event.get("cites")


def test_alter_priority_from_evidence_reorders_the_remaining_queue(short_loop_timeline):
    """eval_alter_priority_from_evidence: before != after lists, citing evidence."""
    doc = short_loop_timeline
    verdict = at.eval_alter_priority_from_evidence(at.TimelineView(doc, "15m"))
    assert verdict["met"], verdict.get("detail")
    altered = [e for e in doc["events"] if e["kind"] == "priority_altered"]
    assert altered, "no priority_altered; the reorder condition cannot be met"
    for event in altered:
        before, after = event["payload"].get("before"), event["payload"].get("after")
        assert isinstance(before, list) and isinstance(after, list)
        assert before != after
        assert event.get("cites")


def test_held_for_refill_is_remaining_and_inflight_not_history():
    """Historical queue items are done, not held. That was the 30m refill-dry lie."""
    queue = [
        {"frontier_id": "FT.DONE", "capability": "a.py"},
        {"frontier_id": "FT.REMAINING", "capability": "b.py"},
    ]
    held = ar.held_for_refill(queue, 1, [{"frontier_id": "FT.FLYING"}])
    assert "FT.DONE" not in held
    assert "FT.REMAINING" in held
    assert "FT.FLYING" in held
    assert ar.held_for_refill(queue, 2, []) == set()
    assert ar.held_for_refill(queue, 0, []) == {"FT.DONE", "FT.REMAINING"}


def test_idle_justified_is_refused_without_a_handle_or_while_novel_work_exists():
    """NEGATIVE CONTROL: the justification event cannot paper over the defect."""
    doc = {"events": []}
    with pytest.raises(ar.EmitRefused, match="no open handle"):
        ar.emit_idle_justified(doc, asked=[], waiting_on=[], t_s=0)
    with pytest.raises(ar.EmitRefused, match="novel work"):
        ar.emit_idle_justified(
            doc,
            asked=[{"frontier_id": "FT.X", "returned": "novel"}],
            waiting_on=[{"job_id": "j1", "pid": 1}],
            t_s=0,
        )
    assert not [e for e in doc.get("events") or [] if e.get("kind") == at.IDLE_JUSTIFIED_KIND]


def test_idle_justified_carries_the_refill_survey_and_the_wait():
    doc = ar.emit_idle_justified(
        {"events": []},
        asked=[
            {"frontier_id": "FT.A", "returned": "already_run", "capability": "freshness.py"},
            {"frontier_id": "FT.B", "returned": "no_safe_capability"},
        ],
        waiting_on=[{"job_id": "spec", "pid": 22827, "unit_id": "WU.TORTURE.NO_WAIT.specimen_verify"}],
        t_s=88,
    )
    event = doc["events"][-1]
    assert event["kind"] == at.IDLE_JUSTIFIED_KIND
    payload = event["payload"]
    assert "refill returned no novel work" in payload["why"]
    assert payload["frontiers_asked"] == ["FT.A", "FT.B"]
    assert payload["n_novel"] == 0
    assert payload["waiting_on"][0]["job_id"] == "spec"
    assert payload["returned"][0]["returned"] == "already_run"


def test_short_loop_passes_no_idle_while_work_exists(short_loop_timeline):
    """Acceptance: a real 25s driver run must PASS the new evaluator."""
    doc = short_loop_timeline
    verdict = at.eval_no_idle_while_work_exists(at.TimelineView(doc, "15m"))
    assert verdict["met"], verdict.get("detail")
    kinds = {e["kind"] for e in doc["events"]}
    assert "idle" not in kinds
    assert "awaiting_instructions" not in kinds
    assert "all_tasks_complete" not in kinds
    for event in doc["events"]:
        if event["kind"] != at.IDLE_JUSTIFIED_KIND:
            continue
        payload = event.get("payload") or {}
        assert payload.get("why")
        assert isinstance(payload.get("returned"), list)
        assert payload.get("waiting_on")
        assert payload.get("n_novel") == 0
