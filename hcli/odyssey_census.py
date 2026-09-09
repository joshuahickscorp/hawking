"""`hcli census` — HCLI surveys ModelLake specimens as bounded WorkUnits.

EPOCH II G015/G016. 26 of 45 registered specimens are censused but never
transfer-tested, and every deep pass so far was driven by Claude reading configs
by hand. This moves the cheap, high-volume half of that work to the resident.

WHY A WORKUNIT AND NOT A CHAT. The same body ran 78 self-development cycles on
the open-ended chat path without landing anything, and wrote a function AND its
test on the bounded WorkUnit path (390842354) — because there, a reply that is
not a result FAILS THE UNIT. That asymmetry is the whole design:

    MEASURE MECHANICALLY  ->  CLASSIFY WITH COGNITION  ->  VERIFY NUMERICALLY

The tool measures the bytes. The model supplies judgement it cannot get from
arithmetic — architecture family, execution class, likely bottlenecks, which
prior Laws apply, what to do next. And then the verifier checks the model's
NUMERIC claims against the measurement it was given, so a confident paragraph
with invented parameter counts is rejected rather than filed.

That last gate is the reason this can take load off the supervisor at all. A
survey that cannot be trusted without re-reading it saves nobody any time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

#: What a census verdict must contain. Missing any of these fails the unit --
#: the model does not get to choose which questions it felt like answering.
REQUIRED = (
    "architecture_family",     # dense / MoE / SSM / RWKV / hybrid / diffusion / encoder / vision / audio
    "execution_class",         # autoregressive decode / encoder pass / diffusion steps / streaming ...
    "dominant_organs",         # which tensors carry the mass
    "state_or_kv",             # what persistent state execution needs
    "likely_bottleneck",       # bandwidth / compute / dispatch / state ...
    "relevant_laws",           # which prior rules apply to this body
    "disposition",             # the S011 disposition vocabulary
    "next_experiment",         # highest-information next step, or "none, dominated"
)

DISPOSITIONS = {
    "REJECTED", "DOMINATED", "DATA-ONLY", "RESEARCH SPECIMEN", "PARETO CANDIDATE",
    "HCLI RESIDENT CANDIDATE", "PULSAR CANDIDATE", "MAGNETAR CANDIDATE",
    "CEPHEID CANDIDATE", "HARDWARE / REPRESENTATION RESEARCH SPECIMEN",
    "SPECIALIST", "DEEP ODYSSEY REQUIRED",
}


@dataclass
class Unit:
    specimen: str
    facts: Dict[str, Any]
    verdict: Optional[Dict[str, Any]] = None
    status: str = "pending"
    reason: str = ""


def build_prompt(u: Unit) -> str:
    """Everything the body needs and nothing it should invent.

    The measured numbers are stated so the model does not have to guess them,
    and it is told they will be checked — a contract it can satisfy is the point
    of the whole Epoch I substrate campaign."""
    f = u.facts
    return (
        "You are classifying ONE model specimen for the Hawking ModelLake census.\n"
        "The measurements below were taken mechanically from the specimen's own "
        "tensors. They are FACTS. Do not restate them incorrectly; every number "
        "you echo is checked against them and a mismatch fails this unit.\n\n"
        f"SPECIMEN: {u.specimen}\n"
        f"  tensors            : {f.get('n_tensors')}\n"
        f"  tensor bytes       : {f.get('tensor_gib')} GiB\n"
        f"  on disk            : {f.get('on_disk_gib')} GiB\n"
        f"  matrix params      : {f.get('matrix_params_b')} B\n"
        f"  dtypes by bytes    : {list((f.get('dtype_bytes') or {}))}\n"
        f"  complete EBPW      : {f.get('complete_ebpw_if_bf16')}\n"
        f"  distinct roles     : {f.get('distinct_roles')}\n"
        f"  repeated blocks    : {list((f.get('repeated_blocks') or {}))[:8]}\n"
        f"  config model_type  : {f.get('config_model_type')}\n"
        f"  config architectures: {f.get('config_architectures')}\n"
        f"  config signals     : {f.get('config_signals')}\n\n"
        "Reply with ONLY a JSON object, no prose before or after, with exactly "
        "these keys:\n"
        + "".join(f"  {k}\n" for k in REQUIRED)
        + f"\n`disposition` must be one of: {sorted(DISPOSITIONS)}\n"
        "`execution_class` describes HOW COMPUTE FLOWS, not what the model is "
        "for. Valid answers look like: autoregressive decode with KV cache; "
        "iterative diffusion denoising; bidirectional encoder pass; recurrent "
        "state scan; streaming full-duplex; encoder-decoder. 'instruct', 'chat' "
        "and 'high-throughput inference' are NOT execution classes and fail.\n"
        "`architecture_family` must be consistent with config model_type above.\n"
        "`matrix_params_b` and `tensor_gib` may be echoed; if you echo them they "
        "must match the facts above.\n"
        "`relevant_laws` is a list naming prior Hawking rules that apply to a "
        "body of this shape. `next_experiment` names the single highest-"
        "information next step, or the string 'none, dominated'.\n"
    )


def parse_verdict(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """The model's JSON, however it chose to wrap it.

    Small bodies fence their JSON, prepend a sentence, or emit it inside a
    tool-call tag. Refusing all of that would be measuring the dialect rather
    than the judgement, so the object is extracted; what is NOT forgiven is a
    missing or wrong ANSWER."""
    if not text:
        return None, "empty reply"
    m = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.S)
    if not m:
        return None, "no JSON object in the reply -- prose does not satisfy this unit"
    try:
        return json.loads(m.group(0)), ""
    except ValueError as exc:
        return None, f"the JSON object does not parse: {exc}"


def verify(u: Unit, verdict: Dict[str, Any]) -> Tuple[bool, str]:
    """The gate. Numeric claims are checked against the measurement."""
    missing = [k for k in REQUIRED if not verdict.get(k)]
    if missing:
        return False, f"missing required keys: {missing}"
    disp = str(verdict["disposition"]).upper().strip()
    if not any(d in disp for d in DISPOSITIONS):
        return False, (f"disposition {verdict['disposition']!r} is not one of the "
                       f"accepted terms")
    # ANTI-HALLUCINATION. If it echoed a number, the number must be right.
    for key, tol in (("matrix_params_b", 0.01), ("tensor_gib", 0.02)):
        if key in verdict:
            try:
                claimed = float(verdict[key])
            except (TypeError, ValueError):
                return False, f"{key} was echoed as a non-number: {verdict[key]!r}"
            actual = float(u.facts.get(key) or 0)
            if actual and abs(claimed - actual) / actual > tol:
                return False, (f"{key} claimed {claimed} against a measured "
                               f"{actual} -- the survey invented a number")
    if len(str(verdict["next_experiment"])) < 8:
        return False, "next_experiment is not an answer"
    return True, ""


def run_unit(u: Unit, complete: Callable[[str], str], attempts: int = 2) -> Unit:
    """One bounded unit. Prose fails it; a wrong number fails it."""
    last = ""
    for attempt in range(attempts):
        prompt = build_prompt(u)
        if last:
            prompt += (f"\nYOUR PREVIOUS REPLY FAILED THIS UNIT: {last}\n"
                       "Emit only the JSON object.\n")
        verdict, why = parse_verdict(complete(prompt))
        if verdict is None:
            last = why
            continue
        ok, why = verify(u, verdict)
        if ok:
            u.verdict, u.status, u.reason = verdict, "accepted", ""
            return u
        last = why
    u.status, u.reason = "failed", last
    return u


def pending_specimens(repo: Path) -> List[str]:
    """Registered specimens with no census verdict yet."""
    census = repo / "receipts/odyssey-i/MODELLAKE_CENSUS_2026-09-09.json"
    out: List[str] = []
    if not census.exists():
        return out
    done = {p.stem.replace("HCLI_CENSUS_", "")
            for p in (repo / "receipts/hcli-census").glob("HCLI_CENSUS_*.json")}
    for s in json.loads(census.read_text())["specimens"]:
        name = s["name"].split("@")[0]
        if name not in done:
            out.append(s["name"])
    return out


def write_receipt(repo: Path, u: Unit) -> Path:
    d = repo / "receipts/hcli-census"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"HCLI_CENSUS_{u.specimen.split('@')[0]}.json"
    p.write_text(json.dumps({
        "specimen": u.specimen,
        "author": "hcli",
        "note": ("classification by the resident; the measurements were taken "
                 "mechanically and every numeric claim in the verdict was checked "
                 "against them before this receipt was written"),
        "measured_facts": u.facts,
        "verdict": u.verdict,
    }, indent=1))
    return p


# ---------------------------------------------------------------- runner ---
def http_completer(endpoint: str, model: str, max_tokens: int = 900,
                   timeout: float = 300.0) -> Callable[[str], str]:
    """An OpenAI-compatible endpoint as a completer.

    Endpoint-agnostic on purpose. The census does not care whether the body
    behind the URL is the 27B resident on :8011 or a 1.2B on an MLX server that
    was spawned for this batch and will be killed after it -- which is what makes
    a POPULATION of agents possible rather than one privileged resident."""
    import json as _json
    import urllib.request

    def complete(prompt: str) -> str:
        body = _json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = _json.load(r)
        ch = d.get("choices") or []
        if not ch:
            raise RuntimeError(f"no choices: {str(d)[:200]}")
        return ch[0]["message"]["content"] or ""

    return complete


def measure(repo: Path, specimen: str) -> Dict[str, Any]:
    """Mechanical anatomy. The model never measures; it only classifies.

    The config's own `model_type` and `architectures` are included because they
    are FACTS on disk, not judgements. The first census run asked the body to
    name the architecture family unaided and it called two diffusion language
    models "LLM" and "LLaMA-variant" -- guessing something that was written down
    three directories away. Facts are handed over; judgement is asked for."""
    import importlib.util
    spec_path = repo / "tools/future/anatomy_from_bytes.py"
    spec = importlib.util.spec_from_file_location("_anat", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    facts = mod.probe(specimen.split("@")[0])
    root = next(iter(sorted(mod.LAKE.glob(f"{specimen.split('@')[0]}@*"))), None)
    if root is not None:
        for cfg in [root / "config.json", *sorted(root.glob("*/config.json"))]:
            if cfg.exists():
                try:
                    c = json.loads(cfg.read_text())
                except Exception:
                    continue
                facts["config_model_type"] = c.get("model_type")
                facts["config_architectures"] = c.get("architectures")
                facts["config_signals"] = {
                    k: c.get(k) for k in
                    ("num_hidden_layers", "hidden_size", "num_experts",
                     "num_local_experts", "state_size", "diffusion_steps",
                     "is_encoder_decoder", "vocab_size")
                    if c.get(k) is not None}
                break
    return facts


def survey(repo: Path, specimens: List[str], complete: Callable[[str], str],
           agent: str = "unknown", on_result: Optional[Callable[[Unit], None]] = None
           ) -> List[Unit]:
    done: List[Unit] = []
    for name in specimens:
        try:
            facts = measure(repo, name)
        except Exception as exc:
            u = Unit(specimen=name, facts={}, status="unmeasurable",
                     reason=f"{type(exc).__name__}: {exc}")
            done.append(u)
            if on_result:
                on_result(u)
            continue
        if facts.get("error"):
            u = Unit(specimen=name, facts=facts, status="unmeasurable",
                     reason=str(facts["error"]))
            done.append(u)
            if on_result:
                on_result(u)
            continue
        u = run_unit(Unit(specimen=name, facts=facts), complete)
        if u.status == "accepted":
            u.verdict["_agent"] = agent
            write_receipt(repo, u)
        done.append(u)
        if on_result:
            on_result(u)
    return done


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="hcli census",
        description="Survey ModelLake specimens as bounded WorkUnits. Prose fails a unit.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--endpoint", default="http://localhost:8011/v1/chat/completions")
    ap.add_argument("--model", default="sealed-3.14")
    ap.add_argument("--agent", default=None, help="label for the receipt (defaults to --model)")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--specimen", action="append", default=None)
    ap.add_argument("--pending", action="store_true", help="list what is left and exit")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="qualify this agent against known answers and exit")
    ap.add_argument("--skip-calibration", action="store_true",
                    help="survey without qualifying (verdicts are marked unqualified)")
    args = ap.parse_args(list(argv or []))

    repo = Path(args.root).resolve()
    todo = args.specimen or pending_specimens(repo)[:args.limit]
    if args.pending:
        rest = pending_specimens(repo)
        print(f"{len(rest)} specimens with no HCLI census verdict:")
        for n in rest:
            print(f"  {n}")
        return 0
    if not todo:
        print("nothing pending")
        return 0

    agent = args.agent or args.model
    complete = http_completer(args.endpoint, args.model)

    if not args.skip_calibration:
        print(f"calibrating {agent} against {len(CALIBRATION)} known answers")

        def cal_report(name, ok, why):
            print(f"  {'PASS' if ok else 'FAIL'} {name.split('@')[0][:44]:44s} {why[:80]}")

        passed, detail = calibrate(repo, complete, on_result=cal_report)
        d = repo / "receipts/hcli-census"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"CALIBRATION_{agent.replace('/', '_')}.json").write_text(
            json.dumps({"agent": agent, "passed": passed, "results": detail}, indent=1))
        if not passed:
            print(f"\n{agent} is NOT QUALIFIED to survey: it missed a body Epoch I "
                  f"already measured. No verdicts written.")
            return 2
        print(f"  {agent} qualified\n")

    if args.calibrate_only:
        return 0
    print(f"census: {len(todo)} units  agent={agent}  endpoint={args.endpoint}")
    import time as _t
    t0 = _t.time()

    def report(u: Unit) -> None:
        mark = {"accepted": "OK ", "failed": "FAIL", "unmeasurable": "SKIP"}[u.status]
        detail = ""
        if u.status == "accepted":
            detail = f"{u.verdict['architecture_family']} / {u.verdict['disposition']}"
        else:
            detail = u.reason[:90]
        print(f"  {mark} {u.specimen.split('@')[0][:44]:44s} {detail}")

    units = survey(repo, todo, complete, agent=agent, on_result=report)
    ok = sum(1 for u in units if u.status == "accepted")
    dt = _t.time() - t0
    print(f"\n{ok}/{len(units)} accepted in {dt:.0f}s "
          f"({dt / max(1, len(units)):.0f}s per unit, agent={agent})")
    return 0 if ok else 1


# ------------------------------------------------------------ calibration ---
#: Specimens whose answers Epoch I already MEASURED. An agent that cannot
#: classify a body we have already characterised is not qualified to classify
#: one we have not, and the first census run proved that gap is real: it called
#: two diffusion language models "LLM" and "LLaMA-variant" while passing every
#: numeric check. Keys are keyword sets -- the answer must CONTAIN one, so
#: wording is free and the claim is not.
CALIBRATION: Dict[str, Dict[str, Any]] = {
    "Dream-org--Dream-v0-Instruct-7B": {
        "architecture_family": {"diffusion"},
        "execution_class": {"diffusion", "denois", "iterative"},
    },
    "GSAI-ML--iLLaDA-8B-Base": {
        "architecture_family": {"diffusion"},
        "execution_class": {"diffusion", "denois", "iterative"},
    },
    "microsoft--bitnet-b1.58-2B-4T": {
        "architecture_family": {"ternary", "bitnet", "low-bit", "low bit"},
        "execution_class": {"autoregressive", "decode"},
    },
    "answerdotai--ModernBERT-large": {
        "architecture_family": {"encoder", "bert"},
        "execution_class": {"encoder", "bidirectional"},
    },
}


def calibrate(repo: Path, complete: Callable[[str], str],
              on_result: Optional[Callable[[str, bool, str], None]] = None
              ) -> Tuple[bool, List[Dict[str, Any]]]:
    """Qualify an agent against known answers. Returns (passed, detail).

    ALL of them must pass. This is a competence gate, not a score: a body that
    gets three of four right will get the fourth kind wrong on a specimen nobody
    has checked, and the whole point of delegating the survey is not having to
    re-read it."""
    out: List[Dict[str, Any]] = []
    for name, expect in CALIBRATION.items():
        try:
            facts = measure(repo, name)
        except Exception as exc:
            out.append({"specimen": name, "passed": False,
                        "why": f"unmeasurable: {type(exc).__name__}: {exc}"})
            continue
        u = run_unit(Unit(specimen=name, facts=facts), complete)
        if u.status != "accepted":
            out.append({"specimen": name, "passed": False,
                        "why": f"unit failed: {u.reason}", "verdict": u.verdict})
            if on_result:
                on_result(name, False, u.reason)
            continue
        bad = []
        for field_, words in expect.items():
            got = str(u.verdict.get(field_, "")).lower()
            if not any(w in got for w in words):
                bad.append(f"{field_}={u.verdict.get(field_)!r} lacks any of {sorted(words)}")
        ok = not bad
        out.append({"specimen": name, "passed": ok, "why": "; ".join(bad),
                    "verdict": u.verdict})
        if on_result:
            on_result(name, ok, "; ".join(bad))
    return all(r["passed"] for r in out), out
