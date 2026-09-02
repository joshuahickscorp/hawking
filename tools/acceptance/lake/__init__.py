"""ModelLake + Qwen27 acceptance. Lake is read-only; never retire/move/rewrite specimens."""

from tools.acceptance.lake.common import GATES, RECEIPT_SCHEMA, WORKTREE

__all__ = ["GATES", "RECEIPT_SCHEMA", "WORKTREE"]
