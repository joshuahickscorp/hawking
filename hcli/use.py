"""`hcli use` -- see admitted Gravity bodies, and change which one is answering.

The switch happens over the SAME connection. Open WebUI reads `/v1/models` for
its dropdown and sends `model` on every request, so a running browser session
follows the change with no reconnect, no restart and no new port. `hcli use` and
the dropdown are two doors onto one mechanism rather than two mechanisms.

WHY A COMMAND WHEN THE DROPDOWN ALREADY WORKS. Loading a body costs seconds to
minutes. Discovering that inside your first message -- watching a chat sit there
with no explanation -- is a bad way to learn it. `hcli use qwen3-14b` pays the
cost deliberately, in a place where a progress line makes sense, and reports
what it swapped from.

WITH NO SURFACE RUNNING it still lists the catalog, because "what can I run?" is
a fair question to ask before starting anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .catalog import Body, catalog, missing_sources, resolve
from .serve import DEFAULT_BASE_URL


def _get(url: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


def _post(url: str, body: Dict[str, Any], timeout: float) -> tuple:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except Exception:
            return exc.code, {"error": {"message": exc.reason}}


def loaded_name(base: str) -> Optional[str]:
    health = _get(base.rstrip("/") + "/health")
    if isinstance(health, dict):
        return health.get("resident")
    return None


def render(bodies: List[Body], current: Optional[str], notes: List[str]) -> str:
    lines = []
    width = max((len(b.name) for b in bodies), default=10)
    for body in bodies:
        mark = "*" if body.name == current else " "
        size = f"{body.bytes / 1e9:6.1f} GB" if body.bytes else " " * 9
        extra = ""
        if body.detail.get("ebpw"):
            extra = f"  ebpw {body.detail['ebpw']}"
        lines.append(f" {mark} {body.name:<{width}}  {body.kind:<14}{size}{extra}")
    if current:
        lines.append("")
        lines.append(f" * = answering now")
    else:
        lines.append("")
        lines.append(" nothing is loaded. `hcli web` or `hcli serve` starts one.")
    for note in notes:
        lines.append(f" note: {note}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hcli use",
        description="List Hawking bodies, or switch the one that is answering.")
    ap.add_argument("model", nargs="?", default=None,
                    help="name or path to switch to; omit to list")
    ap.add_argument("--base", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--research", action="store_true",
                    help="include the advanced ModelLake research catalog")
    ap.add_argument("--timeout", type=float, default=1800.0)
    a = ap.parse_args(list(argv or []))

    try:
        bodies = catalog(research=a.research)
    except Exception as exc:
        print(f"could not read the catalog: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    current = loaded_name(a.base)

    if not a.model:
        if a.json:
            print(json.dumps({
                "loaded": current,
                "bodies": [b.to_openai(loaded=(b.name == current)) for b in bodies],
                "notes": missing_sources()}, indent=2))
        else:
            print(render(bodies, current, missing_sources()))
        return 0

    try:
        target = resolve(a.model, bodies)
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if target is None:
        print(f"no body named {a.model!r}. `hcli use` lists what is available.",
              file=sys.stderr)
        return 2

    if current is None:
        print(f"nothing is running on {a.base}, so there is nothing to switch.\n"
              f"Start it already loaded with:\n"
              f"    python -m hcli web --model {target.path}", file=sys.stderr)
        return 2
    if current == target.name:
        print(f"{target.name} is already answering.")
        return 0

    print(f"switching {current} -> {target.name} ...", flush=True)
    status, body = _post(a.base.rstrip("/") + "/v1/switch",
                         {"model": target.name}, a.timeout)
    if status != 200:
        message = (body.get("error") or {}).get("message") or body
        print(f"REFUSED ({status}): {message}", file=sys.stderr)
        return 2
    print(json.dumps(body) if a.json
          else f"now answering: {body.get('resident')}"
               f"  (was {body.get('from', current)})")
    print("open browser sessions follow this change; no reconnect needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
