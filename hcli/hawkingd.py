"""`hawkingd` -- the long-lived Hawking daemon.

`hcli` is the client; hawkingd is what keeps running after the client exits.
The split matters for more than tidiness: `ps` should describe the process, and
"python3 -m hcli.agentos.resident --supervise <state>" describes the
interpreter that happens to host it. The name is deliberately model-neutral. A
supervisor may hold several bodies, in parallel or one after another, so
naming the daemon for whichever model is loaded today would be wrong tomorrow.

This module exists rather than pointing the shim straight at
``hcli.agentos.resident`` because ``hcli.agentos.__init__`` already imports
that module, so ``-m hcli.agentos.resident`` re-executes an
already-imported module and Python warns about it on every single launch:

    RuntimeWarning: 'hcli.agentos.resident' found in sys.modules after import
    of package 'hcli.agentos', but prior to execution of ...

Ownership of a running daemon is decided by pid plus process start token
(``hcli.resources.process_start_token``), never by argv, so an incumbent
launched under the older name stays owned across this change.
"""
from __future__ import annotations

import sys
from typing import List, Optional

from hcli.agentos.resident import daemon_main


def main(argv: Optional[List[str]] = None) -> int:
    return daemon_main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
