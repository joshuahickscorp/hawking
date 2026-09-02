"""VMCP / AgentOS acceptance lane (roadmap §8 and APPENDIX E).

Each gate is demonstrated by calling an implementing symbol, not by
importing a module. Verdicts are ACCEPTED or BLOCKED; criteria are not
weakened.
"""

from tools.acceptance.vmcp.run import run_all, write_receipts

__all__ = ["run_all", "write_receipts"]
