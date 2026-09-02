"""Worker context compiler: packets must not dump the root goal.

These tests were first run against the pre-fix tree (Mission._unit_context
appends ``GOAL: {self.goal}`` and Engine._build_model_payload wraps that
again). Failures from that run are the evidence the assertions are real.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.engine import Engine, EngineError, _SYSTEM_PROMPT
from hcli.events import EventBus
from hcli.goal import GoalCompiler
from hcli.mission import Mission
from hcli.workunit import WorkUnit
from hcli.workspace import Workspace

BANNED = [
    "THIS IS THE GOVERNING ULTRAGOAL.",
    "Do not give each child the full giant parent context.",
    "88. CORE DOCTRINE",
    "Q30 Ascension",
]

TOKENIZE_URL = "http://127.0.0.1:8080/tokenize"

_ULTRAGOAL_CANDIDATES = (
    REPO / "docs/ultragoals/HCLI_SUPER_AGENT_OS.md",
    Path("/Users/scammermike/Downloads/hawking-copy/docs/ultragoals/HCLI_SUPER_AGENT_OS.md"),
)


def _load_ultragoal() -> str:
    for path in _ULTRAGOAL_CANDIDATES:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        "ultragoal fixture not on disk; looked in: "
        + ", ".join(str(p) for p in _ULTRAGOAL_CANDIDATES)
    )


def tokenize(content: str) -> list:
    payload = json.dumps({"content": content}).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(8):
        req = urllib.request.Request(
            TOKENIZE_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            tokens = data.get("tokens")
            if not isinstance(tokens, list):
                raise AssertionError(f"tokenizer returned no tokens list: {data!r}")
            return tokens
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    # The live tokenizer is the honest way to count, but it is an external
    # service and this suite must not fail because a model server happens to
    # be down. Fall back to a deterministic character proxy and say so: the
    # invariant under test is a RATIO between two prompts, and every
    # measurement in a given comparison uses the same estimator.
    raise _TokenizerUnavailable(str(last_error))


def _first_unit(compiled):
    """The DAG's first unit, whatever it is called.

    Unit ids used to be the fixed pair implement/validate. Since the DAG is
    derived from the obligations, a multi-obligation goal names its units after
    the obligations they discharge (G001, G002, ...), so a test that hardcodes
    "implement" is asserting the old fixed-template behaviour rather than the
    packet contents it means to check.
    """
    units = compiled["workunits"].units
    return units[next(iter(units))]


class _TokenizerUnavailable(RuntimeError):
    """Raised internally so callers can fall back to the character proxy."""


def _tokens_or_proxy(content: str):
    """Token count from the live tokenizer, else a same-estimator proxy."""
    try:
        return len(tokenize(content)), "live-tokenizer"
    except _TokenizerUnavailable:
        # ~3.7 chars/token measured on this model; the exact constant does not
        # matter because it cancels in the ratio.
        return max(1, round(len(content) / 3.7)), "char-proxy"


def _engine(root: Path) -> Engine:
    cfg_dir = root / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    from hcli.config import Config

    runtime = type("Runtime", (), {})()
    runtime.index = 0
    runtime.pid = 1
    runtime.port = 18765
    runtime.active = True
    pool = type("Pool", (), {})()
    pool.runtimes = [runtime]
    return Engine(
        workspace=Workspace(str(root)),
        event_bus=EventBus(),
        runtime_provider=lambda: pool,
        runtime_state_provider=lambda: pool,
        runtime_count=1,
        model_name="local",
        config=Config(str(root), global_path=str(cfg_dir / "global-config.json")),
    )


def _bare_mission(root_goal: str, compiled) -> Mission:
    mission = object.__new__(Mission)
    mission.goal = root_goal
    mission.phase = "running"
    mission._compiled = compiled
    mission._steering = None
    return mission


class TestWorkerPacketCompiler(unittest.TestCase):
    def test_worker_prompt_excludes_unrelated_ultragoal_sections(self):
        root = _load_ultragoal()
        compiler = GoalCompiler()
        compiled = compiler.compile(root)
        wu = _first_unit(compiled)
        mission = _bare_mission(root, compiled)
        ctx = mission._unit_context(wu)
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp))
            payload = engine._build_model_payload(
                ctx["prompt"], evidence=[], compiled=compiled
            )
        user = payload["messages"][1]["content"]
        blob = payload["messages"][0]["content"] + user

        self.assertNotIn("GOAL:", user)
        self.assertTrue(all(s not in user for s in BANNED), user[:800])
        self.assertNotIn(root, user)
        self.assertIn(wu.id, user)
        n_root, how_root = _tokens_or_proxy(root)
        n_worker, how_worker = _tokens_or_proxy(blob)
        self.assertEqual(how_root, how_worker, "both sides must use one estimator")
        self.assertLess(
            n_worker,
            n_root * 0.25,
            (n_worker, n_root),
        )
        self.assertLessEqual(blob.count("MODELS THINK."), 1)

    def test_supplied_units_path_still_excludes_root_from_prompt(self):
        root = _load_ultragoal()
        compiler = GoalCompiler()
        compiled = compiler.compile(root)
        wu = _first_unit(compiled)
        units = {
            "implement": WorkUnit(
                id="implement",
                role="implementation",
                description=wu.description,
            ),
            "validate": WorkUnit(
                id="validate",
                role="validation",
                description="Validate the implementation against the active acceptance criteria.",
                dependencies=["implement"],
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            mission = Mission(
                tmp,
                units=units,
                goal=root,
                quiet=True,
                no_progress_threshold=100,
            )
            ctx = mission._unit_context(units["implement"])
        self.assertNotIn(root, ctx["prompt"])
        self.assertNotIn("GOAL:", ctx["prompt"])
        self.assertIn("implement", ctx["prompt"])
        self.assertNotIn("goal", ctx)

    def test_worker_prompt_includes_six_components(self):
        root = _load_ultragoal()
        compiler = GoalCompiler()
        compiled = compiler.compile(root)
        # Take the DAG's own first two units rather than the old fixed
        # implement/validate names, which no longer exist for a
        # multi-obligation goal.
        _dag_units = list(compiled["workunits"].units.values())
        _first, _second = _dag_units[0], _dag_units[min(1, len(_dag_units) - 1)]
        wu = _second
        with tempfile.TemporaryDirectory() as tmp:
            units = {
                _first.id: WorkUnit(
                    id=_first.id,
                    role="implementation",
                    description=_first.description,
                    status="completed",
                ),
                _second.id: WorkUnit(
                    id=_second.id,
                    role="validation",
                    description=wu.description,
                    dependencies=[_first.id],
                ),
            }
            mission = Mission(
                tmp,
                units=units,
                goal=root,
                quiet=True,
                no_progress_threshold=100,
            )
            mission.phase = "running"
            ctx = mission._unit_context(units[_second.id])
            engine = _engine(Path(tmp))
            payload = engine._build_model_payload(
                ctx["prompt"], evidence=[], compiled=compiled
            )
        user = payload["messages"][1]["content"]
        self.assertIn("PHASE:", user)
        self.assertIn("running", user)
        self.assertIn(_second.id, user)
        self.assertIn("INVARIANTS:", user)
        self.assertIn("ACCEPTANCE:", user)
        self.assertIn("EVIDENCE_PATHS:", user)
        self.assertIn("NEIGHBORHOOD:", user)
        self.assertIn(_first.id, user)

    def test_doctrine_not_duplicated_in_worker_blob(self):
        root = _load_ultragoal()
        compiled = GoalCompiler().compile(root)
        wu = _first_unit(compiled)
        mission = _bare_mission(root, compiled)
        ctx = mission._unit_context(wu)
        with tempfile.TemporaryDirectory() as tmp:
            payload = _engine(Path(tmp))._build_model_payload(
                ctx["prompt"], evidence=[], compiled=compiled
            )
        blob = payload["messages"][0]["content"] + payload["messages"][1]["content"]
        self.assertEqual(blob.count("MODELS THINK."), 1)
        self.assertEqual(blob.count("TOOLS KNOW."), 1)
        self.assertEqual(blob.count("CONTEXT IS A CACHE."), 1)
        self.assertEqual(blob.count("DISK STATE IS AUTHORITY."), 1)
        self.assertEqual(payload["messages"][1]["content"].count("GOAL:"), 0)

    def test_root_goal_reachable_by_reference(self):
        root = _load_ultragoal()
        compiled = GoalCompiler().compile(root)
        wu = _first_unit(compiled)
        with tempfile.TemporaryDirectory() as tmp:
            mission = Mission(
                tmp,
                units={
                    "implement": WorkUnit(
                        id="implement",
                        role="implementation",
                        description=wu.description,
                    )
                },
                goal=root,
                quiet=True,
                no_progress_threshold=100,
            )
            mission.checkpoint()
            ctx = mission._unit_context(mission.scheduler.units["implement"])
            prompt = ctx["prompt"]
            self.assertNotIn(root, prompt)
            self.assertIn("ROOT_REF:", prompt)
            ref_line = [
                line for line in prompt.splitlines() if line.startswith("ROOT_REF:")
            ][0]
            ref = ref_line.split("ROOT_REF:", 1)[1].strip().split("#", 1)[0]
            path = Path(ref)
            self.assertTrue(path.is_file(), ref)
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk.get("goal"), root)

    def test_compile_worker_context_rejects_goal_header(self):
        from hcli.goal import WorkerPacket, compile_worker_context

        wu = WorkUnit(id="u1", role="implementation", description="do the thing")
        compiled = {
            "invariants": ["must not dump the parent"],
            "acceptance_criteria": ["tests pass"],
            "referenced_files": [],
        }
        packet = compile_worker_context(
            wu,
            compiled,
            phase="running",
            units={"u1": wu},
            steering=["[constraint] stay in this file"],
        )
        self.assertIsInstance(packet, WorkerPacket)
        self.assertNotIn("GOAL:", packet.prompt)
        self.assertIn("u1", packet.prompt)
        with self.assertRaises(ValueError):
            WorkerPacket(
                unit_id="x",
                phase="running",
                workunit="x implementation do it",
                invariants=(),
                acceptance=(),
                neighborhood=(),
                evidence_paths=(),
                steering=(),
                prompt="GOAL: dumped",
            )

    def test_knowledge_steers_are_excluded_constraints_kept(self):
        from hcli.goal import compile_worker_context
        from hcli.steering import SteerEvent

        wu = WorkUnit(id="u1", role="work", description="edit foo.py")
        compiled = {
            "invariants": ["never write secrets"],
            "acceptance_criteria": ["foo.py tests pass"],
            "referenced_files": ["foo.py"],
        }
        events = [
            SteerEvent(
                id="s1",
                text="trivia about the weather",
                session_id="s",
                timestamp=0.0,
                kind="knowledge",
            ),
            SteerEvent(
                id="s2",
                text="must use pathlib",
                session_id="s",
                timestamp=0.0,
                kind="constraint",
            ),
            SteerEvent(
                id="s3",
                text="the previous patch inverted the sign",
                session_id="s",
                timestamp=0.0,
                kind="correction",
            ),
        ]
        packet = compile_worker_context(
            wu,
            compiled,
            phase="running",
            units={"u1": wu},
            steering=events,
        )
        # Operator steers of every kind reach the worker. Knowledge is not
        # promoted into INVARIANTS (those are constraints + compiler rules).
        self.assertIn("trivia about the weather", packet.prompt)
        self.assertNotIn("trivia about the weather", "\n".join(packet.invariants))
        self.assertIn("must use pathlib", packet.prompt)
        self.assertTrue(any("must use pathlib" in item for item in packet.invariants))
        self.assertIn("previous patch inverted the sign", packet.prompt)
        self.assertEqual(packet.evidence_paths, ("foo.py",))

    def test_over_cap_drops_neighborhood_before_constraints(self):
        from hcli.goal import compile_worker_context

        wu = WorkUnit(
            id="child",
            role="work",
            description="touch target.py",
            dependencies=["dep0", "dep1", "dep2"],
        )
        units = {
            "child": wu,
            "dep0": WorkUnit(id="dep0", role="work", description="d0", status="completed"),
            "dep1": WorkUnit(id="dep1", role="work", description="d1", status="failed"),
            "dep2": WorkUnit(id="dep2", role="work", description="d2", status="pending"),
        }
        constraint = "CONSTRAINT_MUST_SURVIVE: never drop this constraint " + ("X" * 200)
        compiled = {
            "invariants": ["must keep this invariant " + ("I" * 80) for _ in range(40)],
            "acceptance_criteria": ["tests pass " + ("A" * 40) for _ in range(20)],
            "referenced_files": ["target.py"],
        }
        packet = compile_worker_context(
            wu,
            compiled,
            phase="running",
            units=units,
            steering=[constraint],
            char_cap=800,
        )
        self.assertLessEqual(len(packet.prompt), 800 + 8)
        self.assertIn("CONSTRAINT_MUST_SURVIVE", packet.prompt)
        self.assertIn("never drop this constraint", packet.prompt)
        # Neighborhood details (status + receipt) are the first casualty.
        self.assertNotIn("status=failed", packet.prompt)
        self.assertNotIn("status=completed", packet.prompt)

    def test_scars_surface_causal_reason_across_dag_siblings(self):
        """A resident restarting mid-mission must know WHY things died, not
        just a receipt pointer, and must see same-role dead ends even when
        they are not a direct DAG dependency of the unit about to run."""
        from hcli.goal import compile_worker_context

        wu = WorkUnit(
            id="child",
            role="work",
            description="touch target.py",
            dependencies=["dep0"],
        )
        dep0 = WorkUnit(
            id="dep0",
            role="work",
            description="d0",
            status="failed",
            failure_context={"validation": {"reason": "assertion mismatch on shape"}},
        )
        # Same-role dead end that is NOT a dependency of `wu` at all — the
        # exact gap this closes: a mission-sibling scar outside the DAG edge.
        scar = WorkUnit(
            id="sibling_attempt",
            role="work",
            description="tried a different approach",
            status="failed",
            failure_context={
                "reason": "wrong file targeted",
                "failure_signature": "deadbeefcafefeed",
            },
        )
        # A failed unit of a different role is noise here, not signal.
        other_role_failure = WorkUnit(
            id="unrelated_role_failure",
            role="research",
            description="looked at something else",
            status="failed",
            failure_context={"reason": "irrelevant to this role"},
        )
        units = {
            "child": wu,
            "dep0": dep0,
            "sibling_attempt": scar,
            "unrelated_role_failure": other_role_failure,
        }
        compiled = {"invariants": [], "acceptance_criteria": [], "referenced_files": []}
        packet = compile_worker_context(
            wu, compiled, phase="running", units=units, steering=[]
        )
        # Direct dependency: the causal reason must survive, not just an id
        # and a receipt pointer.
        self.assertIn("dep0", packet.prompt)
        self.assertIn("assertion mismatch on shape", packet.prompt)
        # Cross-sibling scar: `sibling_attempt` is not a dependency of `wu`
        # at all, yet its id and WHY it died must still reach the packet.
        self.assertIn("sibling_attempt", packet.prompt)
        self.assertIn("wrong file targeted", packet.prompt)
        self.assertIn("deadbeefcafefeed", packet.prompt)
        # A failed unit of a different role stays out.
        self.assertNotIn("unrelated_role_failure", packet.prompt)
        self.assertNotIn("irrelevant to this role", packet.prompt)

    def test_scars_are_bounded_not_an_archive(self):
        """Bounded model context stays bounded: no growing buffer of every
        failure the mission has ever produced."""
        from hcli.goal import MAX_SCAR_LINES, compile_worker_context

        wu = WorkUnit(id="child", role="work", description="touch target.py")
        units = {"child": wu}
        total = MAX_SCAR_LINES + 5
        for i in range(total):
            units[f"attempt{i}"] = WorkUnit(
                id=f"attempt{i}",
                role="work",
                description=f"attempt {i}",
                status="failed",
                failure_context={"reason": f"reason {i}"},
            )
        compiled = {"invariants": [], "acceptance_criteria": [], "referenced_files": []}
        packet = compile_worker_context(
            wu, compiled, phase="running", units=units, steering=[]
        )
        shown = sum(1 for i in range(total) if f"attempt{i}" in packet.prompt)
        self.assertEqual(shown, MAX_SCAR_LINES)

    def test_checkpoint_persists_compiled_ir_without_inlining_goal(self):
        root = _load_ultragoal()
        with tempfile.TemporaryDirectory() as tmp:
            mission = Mission(
                tmp,
                units={
                    "implement": WorkUnit(
                        id="implement",
                        role="implementation",
                        description="implement the unit",
                    )
                },
                goal=root,
                quiet=True,
                no_progress_threshold=100,
            )
            path = mission.checkpoint()
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data.get("goal"), root)
            compiled = data.get("compiled")
            self.assertIsInstance(compiled, dict)
            self.assertIn("invariants", compiled)
            self.assertIn("acceptance_criteria", compiled)
            self.assertIn("referenced_files", compiled)
            blob = json.dumps(compiled)
            self.assertNotIn("THIS IS THE GOVERNING ULTRAGOAL.", blob)
            resumed = Mission.from_workspace(
                tmp, engine=None, quiet=True, no_progress_threshold=100
            )
            self.assertIsNotNone(resumed._compiled)
            ctx = resumed._unit_context(resumed.scheduler.units["implement"])
            self.assertNotIn(root, ctx["prompt"])

    def test_execute_fallback_is_hard_error(self):
        from hcli.executors import dispatch_workunit

        class ExecuteOnly:
            def execute(self, prompt):
                self.seen = prompt
                return {"validation": {"ok": True}}

        wu = WorkUnit(id="u1", role="work", description="x")
        with self.assertRaises(EngineError) as ctx:
            dispatch_workunit(
                ExecuteOnly(),
                wu,
                {"prompt": "compiled packet", "evidence_paths": []},
            )
        self.assertIn("execute_workunit", str(ctx.exception))

    def test_consult_uses_packet_prompt_not_root_goal(self):
        from hcli.executors import consult_worker
        from hcli.goal import compile_worker_context

        wu = WorkUnit(id="u1", role="work", description="edit foo.py")
        packet = compile_worker_context(
            wu,
            {
                "invariants": ["must not leak"],
                "acceptance_criteria": ["foo.py tests pass"],
                "referenced_files": ["foo.py"],
            },
            phase="running",
            units={"u1": wu},
            steering=[],
        )
        seen = {}

        class Bridge:
            def consult(self, prompt, **kwargs):
                seen["prompt"] = prompt
                seen["kwargs"] = kwargs
                return "handle"

        root = "THIS IS THE GOVERNING ULTRAGOAL. " * 50
        result = consult_worker(Bridge(), packet)
        self.assertEqual(result, "handle")
        self.assertEqual(seen["prompt"], packet.prompt)
        self.assertNotIn(root, seen["prompt"])
        self.assertNotIn("GOAL:", seen["prompt"])


if __name__ == "__main__":
    unittest.main()
