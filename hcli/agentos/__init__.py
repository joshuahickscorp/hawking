"""AgentOS: Goal, obligations, WorkUnit DAG, scheduler, repair,
mutation authority, verifier orchestration, mission persistence, steering.

Canonical implementation modules remain ``hcli.goal``, ``hcli.workunit``,
``hcli.scheduler``, ``hcli.mission``, ``hcli.ledger``, ``hcli.steering``,
``hcli.mutation``, ``hcli.verifier_pipeline``, ``hcli.dag_store``,
``hcli.executors``, ``hcli.resources``. Re-exported here so ownership is
an importable package, not a comment — the class objects are the same.
"""
from hcli.dag_store import DagStore
from hcli.executors import WorkUnitExecutor
from hcli.goal import GoalCompiler, WorkerPacket, compile_worker_context
from hcli.ledger import Ledger
from hcli.mission import Mission
from hcli.mutation import MutationError
from hcli.resources import MutationLock, ResourceClass, ResourceLimits
from hcli.scheduler import Scheduler
from hcli.steering import SteeringQueue
from hcli.verifier_pipeline import command_is_admissible
from hcli.workunit import WorkUnit

__all__ = [
    "DagStore",
    "GoalCompiler",
    "Ledger",
    "Mission",
    "MutationError",
    "MutationLock",
    "ResourceClass",
    "ResourceLimits",
    "Scheduler",
    "SteeringQueue",
    "WorkUnit",
    "WorkUnitExecutor",
    "WorkerPacket",
    "command_is_admissible",
    "compile_worker_context",
]
