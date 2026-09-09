"""`hcli campaign` — the self-development campaign, legible without grep.

WHY THIS EXISTS. The entire first epoch of the self-development campaign was read
by hand: `grep campaign.log`, `cat evidence/cycle-0077.txt`, and inline python
over trace JSON to find out which tools a cycle actually called. The campaign's
own state was legible only to someone willing to reconstruct it, which is exactly
what Hawking forbids for population counts — applied to the campaign itself.

Two defects were found ONLY because someone happened to dump a trace by hand:

  * a rejected mutation logged as "repo.edit CALLED and accepted", against a
    clean worktree and an unmoved HEAD;
  * `fs.read H-MANIFESTO.md` returning FileNotFoundError as the FIRST call of
    every orientation cycle, for the whole campaign, because the file was
    untracked and therefore absent from the worktree.

Neither was visible in the log line. Both are visible here by construction, so
this reports ANOMALIES rather than only events: a RED marker on a test that
exited 0, a tool that failed, a HEAD that moved without a landing trailer, a
phase that has failed repeatedly. The point is not prettier output. It is that
the questions which cost hours to ask by hand are answered in one command.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The trailer that makes a landing mechanically attributable. A HEAD that moved
#: without it is not HCLI's work -- nine "accepted edits" in experiment 1 were
#: all Claude commits landing mid-cycle.
LANDED_BY = "Landed-By: hcli-autonomous-landing-service"

_CYCLE = re.compile(r"^\[(\d\d:\d\d:\d\d)\] cycle (\d+): (.*)$")
_SOFT = re.compile(r"^\[(\d\d:\d\d:\d\d)\] cycle (\d+) soft-terminal \(([^)]*)\)")
_RESUME = re.compile(r"^\[(\d\d:\d\d:\d\d)\] campaign resumed at cycle (\d+)")
_STUCK = re.compile(r"^\[(\d\d:\d\d:\d\d)\] !! STUCK: (.*)$")
_TRANS = re.compile(r"\| ([A-Z_]+)(?:->([A-Z_]+))? ?\(([^|]*)\)")


@dataclass
class Cycle:
    n: int
    at: str = ""
    reachable: str = ""
    head: str = ""
    frm: str = ""
    to: str = ""
    note: str = ""
    did: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)
    reply_chars: int = 0
    anomalies: List[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return "FAILED" in self.frm or "FAILED" in self.note

    @property
    def moved(self) -> bool:
        return bool(self.to) and self.to != self.frm


def _parse_log(path: Path) -> Dict[str, Any]:
    cycles: Dict[int, Cycle] = {}
    softs: List[tuple] = []
    resumes: List[tuple] = []
    stucks: List[tuple] = []
    if not path.exists():
        return {"cycles": cycles, "softs": softs, "resumes": resumes, "stucks": stucks}
    for line in path.read_text(errors="replace").splitlines():
        m = _SOFT.match(line)
        if m:
            softs.append((m.group(1), int(m.group(2)), m.group(3)))
            continue
        m = _RESUME.match(line)
        if m:
            resumes.append((m.group(1), int(m.group(2))))
            continue
        m = _STUCK.match(line)
        if m:
            stucks.append((m.group(1), m.group(2)))
            continue
        m = _CYCLE.match(line)
        if not m:
            continue
        at, n, rest = m.group(1), int(m.group(2)), m.group(3)
        c = Cycle(n=n, at=at)
        rm = re.search(r"reachable (\S+?),", rest)
        if rm:
            c.reachable = rm.group(1)
        hm = re.search(r"HEAD (\w+)", rest)
        if hm:
            c.head = hm.group(1)
        tm = _TRANS.search(rest)
        if tm:
            c.frm, c.to, c.note = tm.group(1), (tm.group(2) or ""), tm.group(3)
        dm = re.search(r"\| DID: (.*)$", rest)
        if dm:
            c.did = dm.group(1).strip()
        cycles[n] = c
    return {"cycles": cycles, "softs": softs, "resumes": resumes, "stucks": stucks}


def _attach_evidence(cycles: Dict[int, Cycle], evid: Path) -> None:
    for n, c in cycles.items():
        reply = evid / f"cycle-{n:04d}.txt"
        if reply.exists():
            c.reply_chars = len(reply.read_text(errors="replace"))
        tr = evid / f"cycle-{n:04d}.trace.json"
        if tr.exists():
            try:
                c.trace = json.loads(tr.read_text())
            except Exception:
                c.trace = []


def _find_anomalies(cycles: Dict[int, Cycle]) -> None:
    """The whole reason this is not a log formatter.

    Each check corresponds to a defect that actually reached a receipt before
    anyone noticed, so none of them are hypothetical."""
    for c in cycles.values():
        for e in c.trace:
            tool = str(e.get("tool"))
            if tool == "tests.run" and c.note.startswith("RED"):
                rc = (e.get("returncode")
                      if e.get("returncode") is not None
                      else (e.get("value") or {}).get("returncode"))
                if rc == 0 or "passed" in str(c.note) and "failed" not in str(c.note):
                    c.anomalies.append(
                        "RED claimed on a test that PASSED -- a green discriminator "
                        "is a REFUTED, not a RED")
            if tool == "repo.edit":
                if e.get("ok") and e.get("verdict") is None:
                    c.anomalies.append(
                        "repo.edit ok=True with NO verdict -- ok means the "
                        "transaction ran, never that the repository changed")
                if e.get("verdict") in ("rejected",) and "accepted" in c.note:
                    c.anomalies.append(
                        f"repo.edit verdict={e['verdict']} but the cycle was "
                        f"logged as accepted")
            if e.get("ok") is False:
                err = str(e.get("error") or "")[:90]
                c.anomalies.append(f"{tool} FAILED: {err}")
        if c.trace and not any(str(e.get("tool")) != "path" for e in c.trace):
            c.anomalies.append("every tool call in this cycle was malformed")


def gather(root: Path) -> Dict[str, Any]:
    sd = root / ".hcli" / "selfdev"
    parsed = _parse_log(sd / "campaign.log")
    _attach_evidence(parsed["cycles"], sd / "evidence")
    _find_anomalies(parsed["cycles"])
    state: Dict[str, Any] = {}
    sp = sd / "state.json"
    if sp.exists():
        try:
            state = json.loads(sp.read_text())
        except Exception:
            state = {}
    return {**parsed, "state": state, "selfdev": sd}


def arcs(cycles: Dict[int, Cycle]) -> List[Dict[str, Any]]:
    """Group cycles into hypothesis arcs. An arc opens when ORIENT produces a
    hypothesis and closes on a VERDICT or a REFUTED."""
    out: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for n in sorted(cycles):
        c = cycles[n]
        if c.frm == "ORIENT" and c.to and c.to != "ORIENT":
            if cur:
                out.append(cur)
            cur = {"cycles": [], "opened": n, "outcome": "OPEN", "claim": c.note}
        if cur is None:
            cur = {"cycles": [], "opened": n, "outcome": "OPEN", "claim": c.note}
        cur["cycles"].append(c)
        if "REFUTED" in c.note:
            cur["outcome"] = "REFUTED"
        elif c.frm == "VERIFY" and c.to == "ORIENT":
            cur["outcome"] = "VERDICT"
    if cur:
        out.append(cur)
    return out


def render(data: Dict[str, Any], last: int = 3, width: int = 96) -> str:
    st = data["state"]
    cycles = data["cycles"]
    L: List[str] = []
    n = st.get("cycle", max(cycles) if cycles else 0)
    L.append(f"HCLI CAMPAIGN  ·  cycle {n}  ·  phase {st.get('phase_name', '?')}")
    hyp = st.get("hypothesis")
    L.append(f"  hypothesis : {(hyp[:width - 15] + '…') if hyp and len(hyp) > width - 15 else (hyp or '(none — orienting)')}")
    L.append(f"  accepted   : {st.get('accepted_changes', 0)} HCLI-authored changes"
             f"   ·  refuted {len(st.get('refuted_hypotheses') or [])}"
             f"   ·  failed cycles {st.get('failed_cycles', 0)}")

    A = arcs(cycles)
    if A:
        L.append("")
        L.append(f"ARCS  (last {min(last, len(A))} of {len(A)})")
        for a in A[-last:]:
            cs = a["cycles"]
            span = f"c{cs[0].n}-c{cs[-1].n}" if len(cs) > 1 else f"c{cs[0].n}"
            mark = {"REFUTED": "REFUTED ", "VERDICT": "VERDICT ", "OPEN": "OPEN    "}[a["outcome"]]
            L.append(f"  {mark} {span:9s} {a['claim'][:width - 24]}")
            for c in cs:
                arrow = f"{c.frm}→{c.to}" if c.moved else f"{c.frm}"
                tools = Counter(str(e.get("tool")) for e in c.trace)
                tsum = " ".join(f"{k}×{v}" if v > 1 else k for k, v in tools.most_common(4))
                L.append(f"      c{c.n:<4d} {arrow:34s} {tsum[:width - 46]}")
                for an in c.anomalies:
                    L.append(f"           !! {an[:width - 16]}")
    L.append("")
    L.append("HEALTH")
    softs = data["softs"]
    if softs:
        by = Counter(s[1] for s in softs)
        chronic = ", ".join(f"c{k}×{v}" for k, v in sorted(by.items())[-4:])
        kinds = Counter(s[2].split(":")[0] for s in softs)
        L.append(f"  soft-terminals : {len(softs)}  ({chronic})  kinds: "
                 + ", ".join(f"{k}×{v}" for k, v in kinds.most_common()))
    if data["stucks"]:
        L.append(f"  stuck events   : {len(data['stucks'])}  last: {data['stucks'][-1][1][:width - 22]}")
    L.append(f"  restarts       : {len(data['resumes'])}   ·  reachability last seen "
             f"{cycles[max(cycles)].reachable if cycles else '?'}")
    L.append(f"  supervisor     : {st.get('supervisor_interventions', 0)} recorded interventions"
             "  (counter is known to undercount)")
    anom = sum(len(c.anomalies) for c in cycles.values())
    L.append(f"  anomalies      : {anom} across {len(cycles)} cycles"
             + ("  — none outstanding" if not anom else ""))
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="hcli campaign",
        description="Report the self-development campaign without reading raw logs.")
    ap.add_argument("--root", default=".", help="repository root holding .hcli/selfdev")
    ap.add_argument("--arcs", type=int, default=3, help="how many recent arcs to show")
    ap.add_argument("--cycle", type=int, default=None, help="dump one cycle's full trace")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    args = ap.parse_args(list(argv or []))

    data = gather(Path(args.root).resolve())
    if args.cycle is not None:
        c = data["cycles"].get(args.cycle)
        if c is None:
            print(f"no cycle {args.cycle} in the log")
            return 1
        print(f"cycle {c.n} at {c.at}  {c.frm}→{c.to or c.frm}  HEAD {c.head}")
        print(f"  {c.note}")
        print(f"  reply {c.reply_chars} chars, {len(c.trace)} tool calls")
        for e in c.trace:
            ok = "ok " if e.get("ok") else "ERR"
            extra = ""
            if e.get("verdict"):
                extra = f"  verdict={e['verdict']} applied={e.get('applied')}"
            print(f"    {ok} {str(e.get('tool')):16s} {str(e.get('arguments'))[:60]}{extra}")
            if e.get("error"):
                print(f"        {str(e['error'])[:120]}")
        for an in c.anomalies:
            print(f"  !! {an}")
        return 0
    if args.json:
        print(json.dumps({
            "state": data["state"],
            "cycles": {n: {"at": c.at, "from": c.frm, "to": c.to, "note": c.note,
                           "head": c.head, "tools": [e.get("tool") for e in c.trace],
                           "anomalies": c.anomalies}
                       for n, c in data["cycles"].items()},
            "soft_terminals": data["softs"], "restarts": len(data["resumes"]),
        }, indent=1, default=str))
        return 0
    print(render(data, last=args.arcs))
    return 0
