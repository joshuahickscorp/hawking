"""Write HCLI_SELF_HOSTING_LAUNCH.json from live probes, never from assertions.

The point of this receipt is that it cannot be written optimistically. Every
field is the observed result of a command run at the moment of writing. A
capability that cannot be demonstrated records HOW it failed; it does not record
``false`` and it does not record nothing.

Prior campaigns have been damaged by exactly the opposite artifact: a receipt
full of booleans somebody set by hand, which then outlived the state it claimed
to describe. So:

* no probe takes an argument saying what its answer should be
* a probe that raises records the exception, and the receipt still writes
* ``status`` is derived by counting, not by declaration

Run: ``PYTHONPATH=. python3 tools/future/hcli_self_hosting_launch.py``
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
RECEIPT = REPO / "receipts" / "future" / "HCLI_SELF_HOSTING_LAUNCH.json"

# Probes get a real budget: a resident start or a suite run is not instant, and
# a timeout that fires is recorded as a timeout, not silently as a failure.
DEFAULT_TIMEOUT_S = 120.0


def _run(argv: List[str], *, timeout: float = DEFAULT_TIMEOUT_S, cwd: Path = REPO) -> Dict[str, Any]:
    """Run one command and record what actually happened."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {
            "argv": argv, "exit": None, "elapsed_s": round(time.monotonic() - started, 3),
            "stdout": "", "stderr": f"TIMEOUT after {timeout}s",
        }
    except OSError as exc:
        return {
            "argv": argv, "exit": None, "elapsed_s": round(time.monotonic() - started, 3),
            "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "argv": argv,
        "exit": proc.returncode,
        "elapsed_s": round(time.monotonic() - started, 3),
        # Bounded: a receipt is evidence, not a log file.
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-2000:],
    }


def _hcli(*args: str, timeout: float = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
    return _run([PY, "-m", "hcli", *args], timeout=timeout)


# --- probes ---------------------------------------------------------------
# Each returns a dict. It must contain enough for a reader to disagree with it.


def probe_cli_launch() -> Dict[str, Any]:
    """The CLI enters and exits without a human."""
    result = _run([PY, "-m", "hcli"], timeout=90)
    # Driving stdin is what actually proves it reaches the prompt; --help does not.
    proc = _run([PY, "-c",
        "import subprocess,sys,os;"
        "e=dict(os.environ); e['PYTHONPATH']=%r;" % str(REPO) +
        "p=subprocess.run([%r,'-m','hcli']," % PY +
        "input='/exit\\n',capture_output=True,text=True,timeout=80,env=e,cwd=%r);" % str(REPO) +
        "print('EXIT',p.returncode);print(p.stdout[-1200:])"], timeout=100)
    return {"help_probe": result, "interactive_exit": proc}


def probe_neutral_cwd_discovery() -> Dict[str, Any]:
    """Run the installed shim from a directory that is not the repo."""
    shim = shutil.which("hcli")
    if not shim:
        return {"shim": None, "note": "no hcli on PATH"}
    neutral = Path("/tmp")
    return {
        "shim": shim,
        "status_from_neutral_cwd": _run([shim, "agentos", "status"], cwd=neutral, timeout=90),
    }


def probe_resident() -> Dict[str, Any]:
    return {"status": _hcli("agentos", "resident", "status", timeout=60)}


def probe_typed_tools() -> Dict[str, Any]:
    """The tool surface is the resident's hands. Count and name it.

    Parsed in the child, not here: ``_run`` keeps only the tail of stdout so a
    receipt stays evidence rather than a log, and the full tool listing is far
    longer than that tail. Parsing the truncated text reported 0 tools on a
    machine that has 41.

    Counting them is not the same as the resident being able to CALL them -
    ``reachable_from_workunit`` is the question that actually matters, and it is
    answered by grepping the execution path rather than by asking the registry.
    """
    listing = _run([PY, "-c",
        "import json;from hcli.tool_registry import default_tool_registry;"
        "r=default_tool_registry(%r, repo_root=%r);" % (str(REPO), str(REPO)) +
        "print(json.dumps(sorted(str(t.get('name')) for t in r.discover())))"
    ], timeout=90)
    names: List[str] = []
    if listing.get("exit") == 0:
        try:
            names = json.loads(listing["stdout"].strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            listing["parse_error"] = f"{type(exc).__name__}: {exc}"
    if not names:  # fall back to the CLI surface, counting rather than parsing
        cli = _run([PY, "-c",
            "import subprocess,sys,os,json;"
            "e=dict(os.environ);e['PYTHONPATH']=%r;" % str(REPO) +
            "p=subprocess.run([%r,'-m','hcli','agentos','tools']," % PY +
            "capture_output=True,text=True,timeout=80,env=e,cwd=%r);" % str(REPO) +
            "d=json.loads(p.stdout);t=d.get('tools') if isinstance(d,dict) else d;"
            "print(json.dumps(sorted(str(x.get('name')) for x in t)))"
        ], timeout=100)
        listing["cli_fallback"] = cli
        if cli.get("exit") == 0:
            try:
                names = json.loads(cli["stdout"].strip().splitlines()[-1])
            except (ValueError, IndexError):
                pass
    return {
        "count": len(names),
        "names": names,
        "reachable_from_workunit": _tool_reachability(),
        "probe": listing,
    }


def _tool_reachability() -> Dict[str, Any]:
    """Can a WorkUnit actually reach the registry, or is it 41 unreachable tools?

    A tool surface the executor never calls is a catalogue, not a capability.
    This greps the execution path rather than trusting the registry's own count.
    """
    out: Dict[str, Any] = {}
    for rel in ("hcli/mission.py", "hcli/executors.py", "hcli/engine.py",
                "hcli/agentos/resident.py"):
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except OSError as exc:
            out[rel] = f"UNREADABLE: {exc}"
            continue
        out[rel] = {
            "ToolRegistry": text.count("ToolRegistry"),
            "invoke_calls": text.count(".invoke("),
        }
    wired = any(
        isinstance(v, dict) and (v["ToolRegistry"] or v["invoke_calls"])
        for v in out.values()
    )
    return {"per_file": out, "any_call_site": wired}


def probe_test_suite() -> Dict[str, Any]:
    env_path = str(REPO) + os.pathsep + str(REPO / "tools" / "haider")
    started = time.monotonic()
    env = dict(os.environ)
    env["PYTHONPATH"] = env_path
    try:
        proc = subprocess.run(
            [PY, "-m", "pytest", "tools/haider/", "hcli/", "-q"],
            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1200,
        )
        tail = proc.stdout.strip().splitlines()[-1:] or [""]
        return {"exit": proc.returncode, "summary": tail[0],
                "elapsed_s": round(time.monotonic() - started, 1)}
    except subprocess.TimeoutExpired:
        return {"exit": None, "summary": "TIMEOUT", "elapsed_s": round(time.monotonic() - started, 1)}


def probe_paste_cache() -> Dict[str, Any]:
    """Round-trip a large paste through the real module, in a scratch root."""
    script = (
        "import json,tempfile,hashlib;"
        "from hcli.paste_cache import PasteCache;"
        "d=tempfile.mkdtemp();c=PasteCache(d);"
        "blob=('x'*79+chr(10))*2600;"
        "r=c.store(blob);"
        "back=c.get(r.id);"
        "print(json.dumps({'size':r.size,'lines':r.lines,'kind':r.kind,"
        "'byte_identical':back==blob,'context_ref':r.context_ref(),"
        "'context_ref_len':len(r.context_ref()),'listed':len(c.list())}))"
    )
    return _run([PY, "-c", script], timeout=90)


def probe_git_identity() -> Dict[str, Any]:
    return {
        "head": _run(["git", "rev-parse", "HEAD"], timeout=30),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=30),
        "dirty_paths": _run(["git", "status", "--porcelain"], timeout=60),
    }


def probe_modellake() -> Dict[str, Any]:
    return {
        "watcher_processes": _run(["pgrep", "-fl", "modellake_watch"], timeout=30),
        # `pgrep -c` is not a macOS flag; it printed a usage error that read as
        # "no downloads running" on a box that had three.
        "download_processes": _run(
            ["/bin/sh", "-c", "pgrep -f 'hf download' | wc -l"], timeout=30
        ),
    }


def probe_roadmap() -> Dict[str, Any]:
    path = REPO / "civilization" / "ROADMAP_STATE.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "obligation_status_counts": data.get("obligation_status_counts"),
        "obligations_total": data.get("obligations_total"),
        "active_odyssey": data.get("active_odyssey"),
        "active_era": data.get("active_era"),
        "next_work_gates": [w.get("gate") for w in (data.get("next_work") or [])][:5],
    }


def probe_memory_admission() -> Dict[str, Any]:
    """The gate that held the resident at cycles=0. Show what it reads now."""
    script = (
        "import json,time;"
        "from hcli.machine import host_snapshot;"
        "from hcli.agentos.resident import memory_decision;"
        "host_snapshot();time.sleep(2);s=host_snapshot();"
        "d=memory_decision(s);"
        "print(json.dumps({'pressure':s['pressure'],'free_bytes':s['free_bytes'],"
        "'swap_used_bytes':s['swap_used_bytes'],"
        "'swap_highwater_bytes':s.get('swap_highwater_bytes'),"
        "'safe':d['safe'],'reasons':d['reasons']}))"
    )
    return _run([PY, "-c", script], timeout=90)


def probe_landing_authority() -> Dict[str, Any]:
    """Can HCLI land a verified change, and does the governance hold?

    Driven against a SCRATCH repo, never this one. The interesting answer is not
    that a clean case lands - it is that each refusal fires by name.
    """
    script = (
        "import json,subprocess,sys,tempfile;"
        "from pathlib import Path;"
        "from hcli.landing import IntegrationVerifier, LandingProposal, LandingService;"
        "t=Path(tempfile.mkdtemp());"
        "[subprocess.run(c,check=True,capture_output=True) for c in ("
        "['git','init','-q','-b','main',str(t)],"
        "['git','-C',str(t),'config','user.email','p@p'],"
        "['git','-C',str(t),'config','user.name','p'])];"
        "(t/'a.txt').write_text('one\\n');"
        "(t/'t_x.py').write_text('def test_ok():\\n    assert 1==1\\n');"
        "subprocess.run(['git','-C',str(t),'add','-A'],check=True,capture_output=True);"
        "subprocess.run(['git','-C',str(t),'commit','-qm','i'],check=True,capture_output=True);"
        "(t/'a.txt').write_text('two\\n');"
        "mk=lambda **k: LandingProposal(repo_root=t,branch='main',"
        "allowed_paths=k.get('paths',('a.txt',)),"
        "test_command=k.get('cmd',(sys.executable,'-m','pytest','t_x.py','-q')),"
        "message='m');"
        "v=IntegrationVerifier();"
        "out={'tautology_refused':v.check(mk(cmd=('true',))).reason,"
        "'governance_refused':v.check(mk(paths=('hcli/landing.py',))).reason,"
        "'clean_landed':None};"
        "r=LandingService().land(mk());"
        "out['clean_landed']=bool(r.landed);out['sha']=(r.commit_sha or '')[:8];"
        "print(json.dumps(out))"
    )
    return _run([PY, "-c", script], timeout=180)


def probe_processes() -> Dict[str, Any]:
    """Role, RSS and stop-safety for every live Hawking process."""
    return _run([PY, "-c",
        "import json;from hcli.processes import summary;print(json.dumps(summary()))"
    ], timeout=60)


def probe_new_verbs() -> Dict[str, Any]:
    """Odyssey and the ANE lab: REGISTERED, not merely importable.

    The first version of this probe checked `__import__(mod)` and reported
    "importable", which a verifier correctly called out: importability is a much
    weaker property than callability, and conflating them let this receipt
    assert "resident-callable" while its own 45-tool listing contained no
    odyssey entry at all. A resident reaches a capability exactly one way -
    WorkUnit.tool -> _run_tool -> ToolRegistry.invoke - so registration in the
    registry is the property that decides, and it is what is checked here."""
    return _run([PY, "-c",
        "import json;"
        "from hcli.tool_registry import default_tool_registry as d;"
        "names=sorted(t['name'] for t in d(%r, repo_root=%r).discover());"
        "want={'odyssey':'odyssey.','forbidden_fruit':'forbidden_fruit.',"
        "'landing':'git.land.','escalation':'frontier.','swarm':'grok.swarm.'};"
        "reg={k:[n for n in names if n.startswith(v)] for k,v in want.items()};"
        "print(json.dumps({'registered':reg,"
        "'unregistered':[k for k,v in reg.items() if not v],"
        "'tool_count':len(names),'tools':names}))" % (str(REPO), str(REPO))
    ], timeout=90)


def probe_sovereign_mission() -> Dict[str, Any]:
    """The live SUB2 loop: is it advancing, and is it parsing?"""
    script = (
        "import json;"
        "d=json.load(open(%r));"
        "it=d.get('iterations') or [];"
        "recent=it[-12:];"
        "print(json.dumps({'iterations':len(it),"
        "'updated_unix':d.get('updated_unix'),"
        "'live_hypotheses':len(d.get('live_hypotheses') or []),"
        "'scars':len(d.get('scars') or []),"
        "'harness_defects_self_found':len(d.get('harness_defects_found_and_fixed') or []),"
        "'recent_parsed':sum(1 for x in recent if x.get('parsed')),"
        "'recent_degenerated':sum(1 for x in recent if x.get('degenerated')),"
        "'lifetime_parsed':sum(1 for x in it if x.get('parsed'))}))"
        % str(REPO / "receipts" / "future" / "HCLI_MISSION_KERNEL.json")
    )
    return _run([PY, "-c", script], timeout=60)


PROBES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "cli_launch": probe_cli_launch,
    "neutral_cwd_discovery": probe_neutral_cwd_discovery,
    "resident": probe_resident,
    "typed_tools": probe_typed_tools,
    "test_suite": probe_test_suite,
    "paste_cache": probe_paste_cache,
    "git_identity": probe_git_identity,
    "modellake": probe_modellake,
    "roadmap": probe_roadmap,
    "memory_admission": probe_memory_admission,
    "landing_authority": probe_landing_authority,
    "processes": probe_processes,
    "new_verbs": probe_new_verbs,
    "sovereign_mission": probe_sovereign_mission,
}


def main() -> int:
    observations: Dict[str, Any] = {}
    failures: List[str] = []
    for name, probe in PROBES.items():
        print(f"probing {name} ...", flush=True)
        try:
            observations[name] = probe()
        except Exception as exc:  # a probe that dies is data, not a crash
            observations[name] = {"probe_error": f"{type(exc).__name__}: {exc}"}
            failures.append(name)

    receipt = {
        "schema": "hawking.hcli.self_hosting_launch.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recorded_by": "tools/future/hcli_self_hosting_launch.py",
        "note": (
            "Every field below is the observed result of a command run at write "
            "time. Nothing here is a declared capability. A probe that failed "
            "records how it failed rather than reporting false."
        ),
        "python": PY,
        "repo": str(REPO),
        "probes_run": len(PROBES),
        "probes_errored": failures,
        "observations": observations,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(RECEIPT)
    print(f"\nwrote {RECEIPT}")
    print(f"{len(PROBES) - len(failures)}/{len(PROBES)} probes completed without raising")
    if failures:
        print("probe errors:", ", ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
