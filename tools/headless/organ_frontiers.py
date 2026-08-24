#!/usr/bin/env python3
"""ORGAN_FRONTIERS: measured floors for DeltaNet, GQA, embedding/output.

The MLP frontier (uniform sub-2-bit FAILS at 1.85 bpw with an argmax flip,
SURVIVES at 2.25) is cited and then locked out. It is not a prior on any
other organ.

Each organ is scored on REAL held-out activations from capture_diverse2.
Information (weight / table structure) and function (held-out Y) are
separate ledgers. Storage BPW and active BPW are both billed; scales are
counted; every number has a null; the GO metric is required to reject
0.01*W (cosine does not).

    python3 tools/headless/organ_frontiers.py
    python3 -m pytest tools/headless/test_organ_frontiers.py -q

Does not load a second 27B: parent BF16 tensors stream one at a time;
the gravity Q4 artifact is never opened as a model; llama-server is not
used as a teacher.
"""
from __future__ import annotations

import gc
import json
import math
import os
import struct
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"
RECEIPT = ROOT / "receipts" / "headless" / "ORGAN_FRONTIERS.json"
SCHEMA = "hawking.headless.organ_frontiers.v1"

HIDDEN = 5120
VOCAB = 248320
SCALE_BITS = 16
F16_BPW = 16.0
HEADER_BYTES = 64
LOG2_3 = math.log2(3.0)
TRIT_PACK_5IN8 = 8.0 / 5.0  # 1.6 code bpw; + 16/64 = 1.85 with scales
SCALE_TRAP = 0.01
CHUNK = 512
SEED = 20260823
GAIN_HEALTH = 0.50
BAR_Q4 = 0.990  # GQA / mixer high-sensitivity bar (ATTENTION_FLOOR_REFIT)
SURPLUS_MIN = 0.02
REL_FRO_LOCAL_MAX = 0.50  # composition "survives locally" (not Q4-equivalent)
HESSIAN_ROWS = 2048

# Geometry (qwen38_geometry.rs). Not re-derived.
DN_K_HEADS = 16
DN_VPK = 3
DN_K_DIM = 128
DN_V_DIM = 128
DN_LAYERS = 48
GQA_HEADS = 24
GQA_KV_HEADS = 4
GQA_HEAD_DIM = 256
GQA_LAYERS_N = 16
REC_ELEMS_PER_LAYER = DN_K_HEADS * DN_VPK * DN_K_DIM * DN_V_DIM  # 786432

# Probe layers. GQA is every 4th starting at 3; DN otherwise.
DN_PROBE = (0, 32)
GQA_PROBE = (3, 63)
SHARE_DN = (0, 1, 32)

PARENT_CANDIDATES = [
    Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16"),
    ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16",
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/records/runs/qwen38-27b/bf16"),
]
CAPTURE_CANDIDATES = [
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/phaseB/capture_diverse2"),
    ROOT / "workspace/campaign/phaseB/capture_diverse2",
]
TOKENIZER_CANDIDATES = [
    Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16/tokenizer.json"),
]

# Cited MLP whole-model bracket. Do not transfer.
MLP_FAIL_BPW = 1.85  # ternary 5-in-8 g64; argmax flip
MLP_SURVIVE_BPW = 2.25  # q2 g64; argmax agrees
MLP_FAIL_RECEIPT = "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json"
MLP_SURVIVE_RECEIPT = "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json"
ATTN_REFIT_RECEIPT = "receipts/headless/ATTENTION_FLOOR_REFIT.json"
ORGAN_CENSUS_RECEIPT = "receipts/headless/NOETIC_ORGAN_CENSUS.json"
DN_DESIGN_RECEIPT = "receipts/headless/NOETIC_DELTANET_DESIGN.json"

# Hardcoded capture families (capture_diverse2.py). Code family is file-derived
# and is reconstructed best-effort; other families are literal.
_PROSE = [
    "The Antikythera mechanism, recovered from a Roman-era shipwreck, is an ancient Greek analog computer built to predict eclipses and the positions of the planets. Its intricate bronze gears, some with teeth counted in the dozens, encoded astronomical cycles with a precision that would not be matched in Europe for well over a thousand years, and its very existence forced historians to revise their assumptions about the technological ceiling of the ancient world.",
    "Deep beneath the ocean surface, hydrothermal vents spew superheated, mineral-rich water into near-freezing darkness. Around them thrive ecosystems that depend not on sunlight but on chemosynthesis, where bacteria convert hydrogen sulfide into energy. Giant tube worms, ghostly crabs, and heat-tolerant microbes form food webs entirely independent of the sun, suggesting that life could arise on worlds we once dismissed as barren.",
    "The Library of Alexandria was less a single building than an idea: that all the knowledge of the known world could be gathered, copied, and cross-referenced in one place. Scholars there measured the circumference of the Earth, catalogued the stars, and edited the texts of Homer. Its gradual decline, through fire, funding cuts, and neglect, is a reminder that institutions of memory are fragile and must be actively sustained.",
    "Photosynthesis is arguably the most important chemical reaction on the planet. In the chloroplasts of plants and algae, light energy splits water molecules, releasing the oxygen that fills our atmosphere and fixing carbon into the sugars that feed nearly every food chain. The process is astonishingly inefficient in raw energetic terms, yet its cumulative output over billions of years transformed a lifeless rock into a living world.",
    "The construction of the transcontinental railroad reshaped a continent. Crews working from opposite coasts blasted tunnels through granite, bridged canyons, and laid track across deserts, often in brutal conditions. When the final golden spike was driven, a journey that had taken months by wagon collapsed to under a week, knitting distant markets together and accelerating the settlement, and the upheaval, of the American interior.",
    "Sleep, long treated as mere downtime, turns out to be a period of intense biological housekeeping. During deep sleep the brain flushes metabolic waste, consolidates memories from the day, and recalibrates hormones that govern appetite and stress. Chronic sleep deprivation is now linked to impaired judgment, weakened immunity, and long-term disease, making rest not a luxury but a physiological necessity.",
    "The domestication of wild grasses into wheat, rice, and maize was among the most consequential events in human history. By selecting for larger seeds that clung to the stalk rather than scattering, early farmers slowly rewired entire species to depend on human cultivation. In turn, reliable harvests allowed permanent settlements, dense populations, and the specialization of labor that made cities possible.",
    "Auroras form when charged particles from the sun, funneled by Earth's magnetic field toward the poles, collide with gases in the upper atmosphere. Oxygen glows green and red, nitrogen glows blue and violet, and the resulting curtains of light ripple across the polar sky. What appears to be a serene spectacle is in fact the visible edge of a violent interaction between our planet and the star it orbits.",
    "Glass is a peculiar material, neither fully solid nor liquid but an amorphous state in which molecules are frozen in disorder. Ancient craftsmen learned to melt sand into transparent panes long before anyone understood the physics involved. Today the same substance carries the internet as optical fiber, bends light in telescopes, and forms the screens through which much of modern life is now mediated.",
    "The eradication of smallpox stands as one of medicine's greatest triumphs. A coordinated global vaccination campaign, tracking outbreaks village by village, cornered a virus that had killed hundreds of millions across recorded history. By nineteen eighty the disease existed only in laboratory freezers, proving that with enough coordination, humanity could deliberately drive a pathogen to extinction.",
    "Coffee began as a shrub in the highlands of Ethiopia and spread along trade routes to become a global ritual. In the coffeehouses of seventeenth-century Europe it lubricated conversation, commerce, and revolution, earning the nickname penny universities for the ideas exchanged over a cheap cup. The humble bean quietly reorganized daily rhythms around a mild stimulant.",
    "Bridges are exercises in managing invisible forces. A suspension bridge hangs its roadway from cables that transfer enormous loads to towers and anchorages, converting the downward pull of gravity into tension and compression distributed across the structure. Engineers must account for wind, temperature, and resonance, lest a gentle oscillation grow, as it once did at Tacoma Narrows, into catastrophic collapse.",
]
_MATH = [
    "Consider the quadratic equation two x squared minus four x minus six equals zero. Dividing through by two gives x squared minus two x minus three, which factors as x minus three times x plus one. The roots are therefore x equals three and x equals negative one, and their sum, negative b over a, equals two, while their product, c over a, equals negative three, consistent with Vieta's formulas.",
    "To find the area under the curve y equals x squared from zero to three, we integrate. The antiderivative of x squared is x cubed over three. Evaluating from zero to three gives twenty seven over three minus zero, which equals nine. Geometrically this is the accumulated area between the parabola and the horizontal axis across that interval.",
    "A geometric series with first term a and common ratio r, where the absolute value of r is less than one, converges to a over one minus r. For example, one half plus one quarter plus one eighth and so on sums to one, since a equals one half and r equals one half, giving one half over one half.",
    "The probability of drawing two aces in a row from a standard deck without replacement is four over fifty two times three over fifty one. That product equals twelve over two thousand six hundred fifty two, which reduces to one over two hundred twenty one, a little under half a percent.",
    "By the Pythagorean theorem, a triangle with legs of length nine and twelve has a hypotenuse whose square equals eighty one plus one hundred forty four, which is two hundred twenty five. The square root of two hundred twenty five is fifteen, so the hypotenuse measures exactly fifteen units.",
    "The factorial of five is five times four times three times two times one, which equals one hundred twenty. Factorials grow explosively; ten factorial already exceeds three million, which is why they appear in counting problems where order matters, such as permutations.",
    "Logarithms turn multiplication into addition. Because ten to the third is one thousand, the base ten logarithm of one thousand is three. Likewise the logarithm of a product equals the sum of the logarithms, a property that once made slide rules and log tables indispensable for calculation.",
    "The derivative measures instantaneous rate of change. For the function f of x equals sine x, the derivative is cosine x, so the slope of the sine curve at zero is one, and at pi over two, where sine peaks, the slope is zero, reflecting the momentary flatness at the crest.",
    "A system of two linear equations, x plus y equals ten and x minus y equals four, can be solved by addition. Adding the equations eliminates y and gives two x equals fourteen, so x equals seven, and back-substitution yields y equals three.",
    "The mean of the numbers four, eight, fifteen, sixteen, and twenty three is their sum, sixty six, divided by five, which equals thirteen point two. The median, the middle value when sorted, is fifteen, illustrating how mean and median can diverge in a small sample.",
]
_MULTI = [
    "La revolution industrielle a transforme les societes europeennes en deplacant des millions de personnes des campagnes vers les villes. Les usines ont impose de nouveaux rythmes de travail regles par l'horloge plutot que par le soleil, et les conditions difficiles ont fini par susciter des mouvements ouvriers reclamant des journees plus courtes et des salaires plus justes.",
    "El descubrimiento de la penicilina por Alexander Fleming ocurrio casi por accidente cuando noto que un moho contaminante mataba las bacterias en una placa de cultivo. Aquel hallazgo fortuito abrio la era de los antibioticos y salvo incontables vidas, aunque el uso excesivo posterior ha impulsado la aparicion de bacterias resistentes.",
    "Die Entwicklung der Schriftsprache zaehlt zu den wichtigsten kulturellen Errungenschaften der Menschheit. Mit dem Schreiben konnten Gesetze, Vertraege und Geschichten ueber Generationen hinweg bewahrt werden, ohne allein auf das Gedaechtnis angewiesen zu sein, und Verwaltung sowie Handel wurden ueber grosse Entfernungen moeglich.",
    "La biodiversidad de los arrecifes de coral rivaliza con la de las selvas tropicales. Miles de especies de peces, moluscos y crustaceos dependen de estas estructuras vivas, que sin embargo son extremadamente sensibles a los cambios de temperatura del agua, lo que las convierte en indicadores tempranos del calentamiento global.",
    "L'exploration spatiale a commence comme une competition entre deux superpuissances mais est devenue peu a peu une entreprise collaborative. La Station spatiale internationale, assemblee en orbite par plusieurs nations, symbolise cette cooperation et sert de laboratoire pour etudier les effets de l'apesanteur sur le corps humain.",
    "O ciclo da agua conecta oceanos, atmosfera e continentes. A evaporacao eleva o vapor de agua, que se condensa em nuvens e retorna como chuva ou neve, alimentando rios e aquiferos. Esse movimento continuo redistribui a agua doce pelo planeta e sustenta praticamente toda a vida terrestre.",
    "Die Alpen entstanden durch die Kollision der afrikanischen und der europaeischen Kontinentalplatte, ein Prozess, der ueber Millionen von Jahren Gestein auffaltete und Gipfel emporhob. Gletscher formten spaeter die Taeler, und noch heute bewegt sich das Gebirge langsam, waehrend Erosion es unablaessig abtraegt.",
    "La imprenta de Gutenberg multiplico la difusion del conocimiento al hacer posible producir libros en cantidades antes inimaginables. Textos que solo existian en unos pocos manuscritos copiados a mano pudieron circular por toda Europa, alfabetizando a nuevas clases sociales y acelerando la revolucion cientifica.",
]
_INST = [
    "Explain, step by step, how to implement a least-recently-used cache with constant time lookups and updates. Describe the data structures you would combine, why a hash map alone is insufficient, and how a doubly linked list lets you evict the oldest entry in constant time when capacity is exceeded.",
    "Write a clear set of instructions for safely deprecating a public API endpoint. Cover announcing the timeline, adding warning headers, providing a migration guide, monitoring remaining traffic, and choosing a final shutdown date, emphasizing how to avoid breaking downstream consumers who upgrade slowly.",
    "Describe how to design a database schema for a library that tracks books, members, and loans. Specify the tables, the primary and foreign keys, and how you would enforce that a single physical copy cannot be checked out to two members at the same time.",
    "Outline a procedure for debugging an intermittent failure that only appears in production. Include how to add structured logging, reproduce the load pattern, isolate the failing component, and confirm a fix without simply restarting the service and hoping the problem stays away.",
    "Provide guidance on writing a good commit message. Explain why the subject line should be short and imperative, why the body should describe the reason for the change rather than the mechanics, and how referencing the issue being solved helps future maintainers understand the history.",
    "Explain how to convert a recursive function that computes Fibonacci numbers into an iterative one, and then into a memoized version. Compare the time and space complexity of each approach and describe when the extra memory of memoization is justified by the speedup.",
    "Describe the steps to set up automated testing for a web application, from choosing unit versus integration tests, to running them on every pull request, to gating merges on their success, and explain why flaky tests undermine the entire value of the suite.",
    "Write instructions for reducing the memory footprint of a program that loads a large dataset. Cover streaming instead of loading everything at once, using compact data types, releasing objects promptly, and measuring actual usage rather than guessing where the memory goes.",
]
_DIALOGUE = [
    "User: I think my sourdough starter is dead, it hasn't risen in two days. Assistant: Before giving up, try a rescue feeding. Discard all but a spoonful, then feed it equal weights of flour and lukewarm water and keep it somewhere warm around twenty five degrees. If you see even small bubbles within twelve hours, the yeast is dormant, not dead, and consistent daily feedings should revive it.",
    "User: Why does my code run fast locally but slowly in production? Assistant: The usual suspects are differences in data size, network latency between services, and cold caches. Locally you probably hit a small dataset and a warm disk cache, while production queries larger tables across the network. Start by adding timing around each stage and comparing where the wall clock actually goes.",
    "User: Should I learn calculus before linear algebra for machine learning? Assistant: For most practical machine learning, linear algebra is the more immediately useful foundation, since models are built on vectors and matrices. You need only a working grasp of derivatives and the chain rule for training, so a light pass through calculus alongside a solid grounding in linear algebra tends to serve people best.",
    "User: My houseplant's leaves are turning yellow. Assistant: Yellowing usually points to watering problems. If the soil is soggy and the yellowing starts at the bottom, you are likely overwatering and the roots need to dry out. If the leaves are crisp and the soil is bone dry, it is the opposite. Check the top inch of soil and adjust before assuming it needs fertilizer.",
    "User: How do I stay motivated on a long project? Assistant: Break it into pieces small enough that finishing one feels achievable in a single sitting, and track visible progress so you can see the accumulation. Motivation tends to follow action rather than precede it, so lowering the barrier to starting each day matters more than waiting to feel inspired.",
    "User: What actually happens when I clear my browser cache? Assistant: You delete the local copies of images, scripts, and pages your browser saved to load sites faster. The next visit re-downloads everything fresh, which fixes problems caused by stale files but temporarily makes pages load a little slower until the cache refills.",
    "User: Is it worth buying a mechanical keyboard? Assistant: If you type for hours a day, many people find the tactile feedback and durability worth it, and the switches let you tune feel and sound to preference. If you mostly browse or type occasionally, the benefit is more about enjoyment than productivity, so it comes down to how much you value the experience.",
    "User: How do noise-cancelling headphones work? Assistant: They use small microphones to sample the ambient sound, then generate an inverted sound wave that destructively interferes with the incoming noise. This works best on steady low-frequency drones like engine hum, and less well on sudden or high-pitched sounds, which is why they excel on planes but not in noisy offices.",
]

HARDCODED_FAMILIES = {
    "prose": _PROSE,
    "math": _MATH,
    "multilingual": _MULTI,
    "instruction": _INST,
    "dialogue": _DIALOGUE,
}


# ---------------------------------------------------------------------------
# import-safe utilities (no torch, no execv)
# ---------------------------------------------------------------------------

def grouped_storage_bpw(bits: int, group: int) -> float:
    """Codes + one f16 scale per group. Scales counted."""
    return float(bits) + float(SCALE_BITS) / float(group)


def ternary_5in8_storage_bpw(group: int = 64) -> float:
    return TRIT_PACK_5IN8 + float(SCALE_BITS) / float(group)


def binary_storage_bpw(group: int = 64) -> float:
    return 1.0 + float(SCALE_BITS) / float(group)


def lowrank_f16_bpw(rows: int, cols: int, rank: int) -> float:
    n_w = rows * cols
    return F16_BPW * rank * (rows + cols) / n_w


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_json(rel: str):
    p = ROOT / rel
    if p.is_file():
        return json.loads(p.read_text()), f"disk:{rel}"
    try:
        raw = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT, timeout=60)
        return json.loads(raw), f"git:HEAD:{rel}"
    except Exception:
        alt = Path("/Users/scammermike/Downloads/hawking-copy") / rel
        if alt.is_file():
            return json.loads(alt.read_text()), f"copy:{rel}"
        return None, f"missing:{rel}"


def j(x):
    if isinstance(x, dict):
        return {k: j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
    except Exception:
        pass
    return str(x)


def numbered(value, *, status: str, null, unit=None, formula=None, source=None, note=None):
    rec = {"value": j(value), "status": status, "null": j(null)}
    if unit is not None:
        rec["unit"] = unit
    if formula is not None:
        rec["formula"] = formula
    if source is not None:
        rec["source"] = source
    if note is not None:
        rec["note"] = note
    return rec


def find_parent() -> Path:
    for p in PARENT_CANDIDATES:
        if (p / "model.safetensors.index.json").is_file():
            return p
    raise FileNotFoundError("qualified parent bf16 not found")


def find_capture() -> Path:
    for p in CAPTURE_CANDIDATES:
        if (p / "L00.f16").is_file() or (p / "L0.f16").is_file():
            return p
    raise FileNotFoundError("real post_attn_norm capture not found")


def find_tokenizer() -> Path | None:
    for p in TOKENIZER_CANDIDATES:
        if p.is_file():
            return p
    parent = None
    try:
        parent = find_parent()
    except FileNotFoundError:
        return None
    t = parent / "tokenizer.json"
    return t if t.is_file() else None


_INDEX_CACHE: dict | None = None


def weight_index(parent: Path) -> dict[str, str]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = json.loads((parent / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
    return _INDEX_CACHE


def load_tensor(parent: Path, name: str):
    import numpy as np

    shard = parent / weight_index(parent)[name]
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return np.array(f32.reshape(shape), dtype=np.float32, copy=True)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape).copy()
    raise ValueError(f"{name} dtype {dtype}")


def load_tensor_f16(parent: Path, name: str):
    """BF16 parent tensor as float16 (2 bytes/elem). Does not keep f32."""
    import numpy as np

    shard = parent / weight_index(parent)[name]
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    if meta["dtype"] != "BF16":
        raise ValueError(f"{name} dtype {meta['dtype']}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return np.array(f32.reshape(tuple(meta["shape"])), dtype=np.float16)


def tensor_name(layer: int, kind: str) -> str:
    return f"model.language_model.layers.{layer}.{kind}"


def capture_path(cap: Path, layer: int) -> Path:
    for name in (f"L{layer:02d}.f16", f"L{layer}.f16"):
        p = cap / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no capture for layer {layer} in {cap}")


def load_X(cap: Path, layer: int):
    import numpy as np

    p = capture_path(cap, layer)
    raw = np.fromfile(p, dtype=np.float16)
    if raw.size % HIDDEN != 0:
        raise ValueError(f"{p} size {raw.size} not divisible by hidden {HIDDEN}")
    X = raw.reshape(-1, HIDDEN).astype(np.float32)
    if X.shape[0] < 256:
        raise ValueError(f"{p} only {X.shape[0]} rows; refusing a toy capture")
    return X


def split_from_manifest(manifest: dict, n_tokens: int):
    import numpy as np

    if manifest.get("manifest"):
        fit, hold = [], []
        for m in manifest["manifest"]:
            sl = np.arange(m["row_start"], m["row_start"] + m["n_tokens"])
            (hold if m.get("split") == "hold" else fit).append(sl)
        return np.concatenate(fit), np.concatenate(hold)
    n_hold = max(256, n_tokens // 5)
    return np.arange(0, n_tokens - n_hold), np.arange(n_tokens - n_hold, n_tokens)


def _write(obj: dict) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(j(obj), indent=2) + "\n")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def gemm(a, b):
    import numpy as np

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    try:
        import torch

        return (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()
    except Exception:
        return a @ b


def x_wt(X, W, chunk: int = CHUNK):
    import numpy as np

    n = X.shape[0]
    out_dim = W.shape[0]
    if n <= chunk:
        return gemm(X, W.T)
    y = np.empty((n, out_dim), dtype=np.float32)
    for i in range(0, n, chunk):
        y[i : i + chunk] = gemm(X[i : i + chunk], W.T)
    return y


def row_cosine(A, B) -> float:
    import numpy as np

    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    ok = den > 1e-20
    if not np.any(ok):
        return 0.0
    return float((num[ok] / den[ok]).mean())


def min_row_cosine(A, B) -> float:
    import numpy as np

    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    ok = den > 1e-20
    if not np.any(ok):
        return 0.0
    return float((num[ok] / den[ok]).min())


def rel_fro(A, B) -> float:
    import numpy as np

    na = np.linalg.norm(A)
    if na == 0:
        return float("nan")
    return float(np.linalg.norm(A - B) / na)


def gain_score(A, B) -> float:
    import numpy as np

    def ratio(axis):
        na = np.linalg.norm(A, axis=axis)
        nb = np.linalg.norm(B, axis=axis)
        r = nb / (na + 1e-30)
        return np.minimum(r, 1.0 / (r + 1e-30))

    return float(min(np.mean(ratio(1)), ratio(0).min()))


def constant_mean_null(Y) -> float:
    import numpy as np

    mu = Y.mean(axis=0, keepdims=True)
    return row_cosine(Y, np.broadcast_to(mu, Y.shape))


def score_pair(Y, Yh) -> dict:
    cos = row_cosine(Y, Yh)
    null = constant_mean_null(Y)
    gain = gain_score(Y, Yh)
    rf = rel_fro(Y, Yh)
    return {
        "rel_fro": rf,
        "cosine": cos,
        "cosine_min_row": min_row_cosine(Y, Yh),
        "gain": gain,
        "scale_aware": cos * gain,
        "null": null,
        "beats_null": bool(cos > null),
        "surplus_over_null": cos - null,
        "n_rows": int(Y.shape[0]),
        "dim": int(Y.shape[1]),
    }


def q4_healthy(sc: dict) -> tuple[bool, str]:
    if sc["cosine"] < BAR_Q4:
        return False, f"hold_cosine {sc['cosine']:.6f} < {BAR_Q4}"
    if sc["gain"] < GAIN_HEALTH:
        return False, f"gain {sc['gain']:.6f} < {GAIN_HEALTH}"
    if sc["surplus_over_null"] < SURPLUS_MIN:
        return False, (
            f"cosine {sc['cosine']:.6f} within {SURPLUS_MIN} of null {sc['null']:.6f}"
        )
    return True, "Q4-equivalent: cosine>=0.990, gain>=0.50, surplus>=0.02"


def local_survives(sc: dict) -> tuple[bool, str]:
    """Looser composition-style local survival. Not the mixer Q4 bar."""
    if sc["gain"] < GAIN_HEALTH:
        return False, f"gain {sc['gain']:.6f} < {GAIN_HEALTH}"
    if sc["rel_fro"] > REL_FRO_LOCAL_MAX:
        return False, f"rel_fro {sc['rel_fro']:.6f} > {REL_FRO_LOCAL_MAX}"
    if sc["scale_aware"] < sc["null"] * sc["gain"] + 0.05 and sc["surplus_over_null"] < 0.05:
        # scale_aware of the constant-mean null is not stored; require surplus
        return False, f"surplus {sc['surplus_over_null']:.6f} < 0.05"
    if not sc["beats_null"]:
        return False, "does not beat constant-mean null"
    return True, "local_survives (gain, rel_fro, beats_null); NOT Q4-equivalent"


def participation_ratio(energy) -> float:
    import numpy as np

    e = np.asarray(energy, dtype=np.float64)
    e = e[e > 0]
    if e.size == 0:
        return float("nan")
    s = e.sum()
    return float((s * s) / ((e * e).sum() + 1e-30))


def snap_f16(x):
    import numpy as np

    return x.astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# codecs (scales counted)
# ---------------------------------------------------------------------------

def as_groups(W, g: int):
    import numpy as np

    rows, cols = W.shape
    if cols % g != 0:
        raise ValueError(f"cols {cols} not divisible by group {g}")
    return np.ascontiguousarray(W, dtype=np.float32).reshape(rows, cols // g, g)


def bill_grouped(n_w: int, bits: float, n_scales: int, extra_bits: float = 0.0) -> dict:
    scale_bits = float(n_scales) * SCALE_BITS
    storage_bits = float(bits) * n_w + scale_bits + float(extra_bits)
    storage_bpw = storage_bits / n_w
    return {
        "n_weights": int(n_w),
        "code_bpw": float(bits),
        "n_scales": int(n_scales),
        "scale_bits": SCALE_BITS,
        "scale_bpw": scale_bits / n_w,
        "storage_bits": storage_bits,
        "storage_bpw": storage_bpw,
        "active_fused_bpw": storage_bpw,
        "active_cached_f16_bpw": F16_BPW,
        "scales_counted": True,
        "note": (
            "storage includes codes + f16 scales. fused_active = storage for an "
            "in-register dequant matvec. cached_f16_active = 16 if W_hat is densified."
        ),
    }


def bill_factors(rows: int, cols: int, rank: int) -> dict:
    n_w = rows * cols
    factor_elems = rank * (rows + cols)
    storage_bits = F16_BPW * factor_elems
    storage_bpw = storage_bits / n_w
    return {
        "n_weights": int(n_w),
        "rank": int(rank),
        "factor_elems": int(factor_elems),
        "storage_bits": storage_bits,
        "storage_bpw": storage_bpw,
        "active_fused_bpw": storage_bpw,
        "active_cached_f16_bpw": F16_BPW,
        "scales_counted": True,
        "note": (
            "f16 factors, no header. two-GEMM active = storage. densified W_hat = 16 active bpw."
        ),
    }


def ws_rtn(W, bits: int, group: int = 64):
    import numpy as np

    if bits <= 1:
        raise ValueError("bits<=1 is the degenerate absmax trap")
    G = as_groups(W, group)
    bound = (1 << (bits - 1)) - 1
    amax = np.max(np.abs(G), axis=-1, keepdims=True)
    scale = snap_f16(np.where(amax > 0, amax / float(bound), np.ones_like(amax)))
    q = np.clip(np.rint(G / np.where(scale > 0, scale, 1.0)), -bound - 1, bound)
    What = (q * scale).reshape(W.shape).astype(np.float32)
    acc = bill_grouped(int(W.size), bits, int(W.shape[0] * (W.shape[1] // group)))
    acc["method"] = "ws_rtn"
    acc["bits"] = bits
    acc["group"] = group
    return What, acc


def aa_diag_scale(W, bits: int, group: int, d):
    """Function-space grouped quant: per-group scale from diag(X.T X)."""
    import numpy as np

    if bits <= 1:
        raise ValueError("bits<=1 is the degenerate absmax trap")
    G = as_groups(W, group)
    bound = (1 << (bits - 1)) - 1
    dd = d.reshape(W.shape[1] // group, group).astype(np.float32)
    dd = np.broadcast_to(dd[None, :, :], G.shape)
    a = np.abs(G)
    den = dd.sum(axis=-1, keepdims=True)
    # energy-weighted absmax proxy: weighted mean-abs * headroom to bound
    meanabs = (dd * a).sum(axis=-1, keepdims=True) / np.maximum(den, 1e-30)
    amax = np.max(a, axis=-1, keepdims=True)
    # blend: start at absmax, shrink toward energy-weighted if the Hessian is peaked
    scale0 = np.where(amax > 0, amax / float(bound), np.ones_like(amax))
    scale_aa = np.where(meanabs > 0, meanabs * 2.0 / float(bound), scale0)
    scale = snap_f16(0.5 * scale0 + 0.5 * scale_aa)
    q = np.clip(np.rint(G / np.where(scale > 0, scale, 1.0)), -bound - 1, bound)
    What = (q * scale).reshape(W.shape).astype(np.float32)
    acc = bill_grouped(int(W.size), bits, int(W.shape[0] * (W.shape[1] // group)))
    acc["method"] = "fs_diagH"
    acc["bits"] = bits
    acc["group"] = group
    return What, acc


def binary_meanabs(W, group: int = 64, d=None):
    import numpy as np

    G = as_groups(W, group)
    s = np.where(G >= 0.0, 1.0, -1.0).astype(np.float32)
    if d is None:
        scale = np.mean(np.abs(G), axis=-1, keepdims=True)
    else:
        dd = d.reshape(W.shape[1] // group, group).astype(np.float32)
        dd = np.broadcast_to(dd[None, :, :], G.shape)
        den = dd.sum(axis=-1, keepdims=True)
        scale = (dd * np.abs(G)).sum(axis=-1, keepdims=True) / np.maximum(den, 1e-30)
    scale = snap_f16(scale)
    What = (s * scale).reshape(W.shape).astype(np.float32)
    acc = bill_grouped(int(W.size), 1.0, int(W.shape[0] * (W.shape[1] // group)))
    acc["method"] = "binary_aa" if d is not None else "binary_meanabs"
    acc["group"] = group
    return What, acc


def ternary_fit(W, group: int = 64, d=None, iters: int = 3):
    import numpy as np

    G = as_groups(W, group)
    a = np.abs(G)
    if d is None:
        dd = np.ones_like(G)
    else:
        dd = d.reshape(W.shape[1] // group, group).astype(np.float32)
        dd = np.broadcast_to(dd[None, :, :], G.shape)
    den0 = dd.sum(axis=-1, keepdims=True)
    s = (dd * a).sum(axis=-1, keepdims=True) / np.maximum(den0, 1e-30)
    p = None
    for _ in range(iters):
        p = np.where(a > (s / 2.0), np.where(G >= 0.0, 1.0, -1.0), 0.0).astype(np.float32)
        m = p != 0
        den = (dd * m).sum(axis=-1, keepdims=True)
        num = (dd * a * m).sum(axis=-1, keepdims=True)
        s = np.where(den > 0, num / np.maximum(den, 1e-12), s)
    s = snap_f16(s)
    What = (s * p).reshape(W.shape).astype(np.float32)
    n_sc = int(W.shape[0] * (W.shape[1] // group))
    acc = bill_grouped(int(W.size), TRIT_PACK_5IN8, n_sc)
    acc["method"] = "ternary_aa" if d is not None else "ternary"
    acc["group"] = group
    acc["storage_bpw_packed2"] = 2.0 + SCALE_BITS / group
    acc["storage_bpw_5in8"] = acc["storage_bpw"]
    return What, acc


def input_pcs(X, max_rank: int, seed: int = SEED):
    """Uncentered right singular vectors of real X. Exact Gram when cheap."""
    import numpy as np

    n, p = X.shape
    r = min(int(max_rank), n, p)
    Xt = np.ascontiguousarray(X, dtype=np.float32)
    if p <= 8192:
        G = gemm(Xt.T, Xt)
        evals, evecs = np.linalg.eigh(G)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        V = np.ascontiguousarray(evecs[:, order], dtype=np.float32)
        s = np.sqrt(evals).astype(np.float32)
        fro2 = float(evals.sum())
        method = "exact_gram_eigh"
        return V[:, :r], s[:r], fro2, method
    rng = np.random.default_rng(seed)
    over = min(r + 16, p, n)
    Q = rng.standard_normal((p, over)).astype(np.float32)
    Y = gemm(Xt, Q)
    Qout, _ = np.linalg.qr(Y, mode="reduced")
    for _ in range(2):
        Qout, _ = np.linalg.qr(gemm(Xt.T, Qout), mode="reduced")
        Qout, _ = np.linalg.qr(gemm(Xt, Qout), mode="reduced")
    B = gemm(Qout.T, Xt)
    _u, s, Vh = np.linalg.svd(B, full_matrices=False)
    V = np.ascontiguousarray(Vh[:r].T, dtype=np.float32)
    fro2 = float((Xt.astype(np.float64) ** 2).sum())
    return V, s[:r].astype(np.float32), fro2, "randomized_svd_niter2"


def project_W(W, V_k):
    """W_hat = (W @ V_k) @ V_k.T  — activation-aware / shared-basis."""
    WV = gemm(W, V_k)
    return gemm(WV, V_k.T)


def aa_rank_hat(W, V, k: int):
    return project_W(W, V[:, :k])


def _silu(x):
    import numpy as np

    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def _sigmoid(x):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fuse_q38_qkvz(qkv, z):
    import numpy as np

    key_heads, key_dim, vpk, value_dim, hidden = 16, 128, 3, 128, HIDDEN
    key_elements = key_heads * key_dim
    value_rows = vpk * value_dim
    qkvz_rows_per_key = key_dim * 2 + value_rows * 2
    fused = np.empty((key_heads * qkvz_rows_per_key, hidden), dtype=np.float32)
    for kh in range(key_heads):
        dst = kh * qkvz_rows_per_key
        q_src = kh * key_dim
        k_src = key_elements + kh * key_dim
        v_src = key_elements * 2 + kh * value_rows
        z_src = kh * value_rows
        fused[dst : dst + key_dim] = qkv[q_src : q_src + key_dim]
        fused[dst + key_dim : dst + 2 * key_dim] = qkv[k_src : k_src + key_dim]
        fused[dst + 2 * key_dim : dst + 2 * key_dim + value_rows] = qkv[
            v_src : v_src + value_rows
        ]
        fused[dst + 2 * key_dim + value_rows : dst + qkvz_rows_per_key] = z[
            z_src : z_src + value_rows
        ]
    return fused


def deltanet_out_proxy(X, W_qkvz):
    import numpy as np

    y = x_wt(X, W_qkvz)
    value_rows = DN_VPK * DN_V_DIM
    per_key = DN_K_DIM * 2 + value_rows * 2
    y3 = y.reshape(X.shape[0], DN_K_HEADS, per_key)
    v = y3[:, :, DN_K_DIM * 2 : DN_K_DIM * 2 + value_rows].reshape(X.shape[0], -1)
    z = y3[:, :, DN_K_DIM * 2 + value_rows :].reshape(X.shape[0], -1)
    return np.ascontiguousarray(v * _silu(z), dtype=np.float32)


def gqa_out_proxy(X, W_q, W_v):
    import numpy as np

    qg = x_wt(X, W_q).reshape(X.shape[0], GQA_HEADS, 2, GQA_HEAD_DIM)
    gate = _sigmoid(qg[:, :, 1, :])
    v = x_wt(X, W_v).reshape(X.shape[0], GQA_KV_HEADS, GQA_HEAD_DIM)
    v_rep = np.repeat(v, GQA_HEADS // GQA_KV_HEADS, axis=1)
    return np.ascontiguousarray(
        (v_rep * gate).reshape(X.shape[0], GQA_HEADS * GQA_HEAD_DIM), dtype=np.float32
    )


def eval_linear(W, Wh, X_hold, Y=None) -> dict:
    if Y is None:
        Y = x_wt(X_hold, W)
    Yh = x_wt(X_hold, Wh)
    sc = score_pair(Y, Yh)
    ok, reason = q4_healthy(sc)
    loc, loc_r = local_survives(sc)
    sc["q4_equivalent"] = ok
    sc["q4_reason"] = reason
    sc["local_survives"] = loc
    sc["local_reason"] = loc_r
    return sc


def pack_candidate(name, family, acc, function_score, information_score=None, extra=None):
    rec = {
        "name": name,
        "family": family,
        "storage_bpw": acc.get("storage_bpw"),
        "active_fused_bpw": acc.get("active_fused_bpw"),
        "active_cached_f16_bpw": acc.get("active_cached_f16_bpw", F16_BPW),
        "scales_counted": bool(acc.get("scales_counted", True)),
        "accounting": acc,
        "function": function_score,
        "information": information_score,
    }
    if extra:
        rec.update(extra)
    return rec


def cheapest_healthy(cands, *, key="q4_equivalent") -> dict | None:
    ok = []
    for c in cands:
        fn = c.get("function") or {}
        if fn.get(key):
            bpw = c.get("storage_bpw")
            if bpw is not None:
                ok.append((float(bpw), c))
    if not ok:
        return None
    ok.sort(key=lambda t: t[0])
    return ok[0][1]


def _cand_tensor(c: dict) -> str | None:
    t = c.get("tensor")
    if t:
        return str(t)
    name = c.get("name") or ""
    if name.startswith("organ_vsiluz"):
        return "organ_vsiluz"
    if name.startswith("lm_head"):
        return "lm_head"
    if name.startswith("embed"):
        return "embed_tokens"
    return None


def per_tensor_floors(cands, *, key="q4_equivalent") -> dict:
    """Cheapest HEALTHY storage_bpw per tensor. Organ floor is the MAX of these."""
    by: dict[str, dict] = {}
    for c in cands:
        t = _cand_tensor(c)
        if not t:
            continue
        fn = c.get("function") or {}
        if not fn.get(key):
            continue
        bpw = c.get("storage_bpw")
        if bpw is None:
            continue
        prev = by.get(t)
        if prev is None or float(bpw) < float(prev["storage_bpw"]):
            by[t] = c
    return by


def gated_organ_candidate(cands, *, key="q4_equivalent") -> dict | None:
    """Organ floor is gated by the most expensive required tensor, not the cheapest lucky one."""
    by = per_tensor_floors(cands, key=key)
    if not by:
        return None
    return max(by.values(), key=lambda c: float(c["storage_bpw"]))


# ---------------------------------------------------------------------------
# tokenizer / hot-cold
# ---------------------------------------------------------------------------

def encode_prompt(tok, text: str) -> list[int]:
    enc = tok.encode(text)
    ids = list(enc.ids)
    return ids


def reconstruct_token_ids(tok, manifest: dict) -> dict:
    """Best-effort alignment of hardcoded families to the capture manifest."""
    by_fam: dict[str, list[dict]] = {}
    for m in manifest.get("manifest") or []:
        by_fam.setdefault(m["family"], []).append(m)

    out = {
        "aligned_families": [],
        "failed_families": [],
        "fit_ids": [],
        "hold_ids": [],
        "n_tokens_aligned": 0,
        "alignment": "tokenizer.json of the qualified parent; mlx_lm used the same file",
    }
    bos = 248044
    for fam, prompts in HARDCODED_FAMILIES.items():
        rows = by_fam.get(fam) or []
        if len(rows) != len(prompts):
            out["failed_families"].append(
                {"family": fam, "reason": f"prompt count {len(prompts)} vs manifest {len(rows)}"}
            )
            continue
        fam_ok = True
        fam_ids = []
        for prompt, row in zip(prompts, rows):
            ids = encode_prompt(tok, prompt)
            if len(ids) == row["n_tokens"]:
                fam_ids.append((row, ids))
                continue
            if ids and ids[0] == bos and len(ids) - 1 == row["n_tokens"]:
                fam_ids.append((row, ids[1:]))
                continue
            if len(ids) + 1 == row["n_tokens"]:
                fam_ids.append((row, [bos] + ids))
                continue
            fam_ok = False
            out["failed_families"].append(
                {
                    "family": fam,
                    "prompt_idx": row.get("prompt_idx"),
                    "got": len(ids),
                    "want": row["n_tokens"],
                }
            )
            break
        if not fam_ok:
            continue
        out["aligned_families"].append(fam)
        for row, ids in fam_ids:
            bucket = out["hold_ids"] if row.get("split") == "hold" else out["fit_ids"]
            bucket.extend(int(i) for i in ids)
            out["n_tokens_aligned"] += len(ids)
    return out


# ---------------------------------------------------------------------------
# embed helpers
# ---------------------------------------------------------------------------

def row_score_table(W, Wh, ids) -> dict:
    import numpy as np

    if len(ids) == 0:
        return {"n": 0, "note": "no ids"}
    idx = np.asarray(ids, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < W.shape[0])]
    Y = W[idx].astype(np.float32, copy=False)
    Yh = Wh[idx].astype(np.float32, copy=False)
    sc = score_pair(Y, Yh)
    ok, reason = q4_healthy(sc)
    loc, loc_r = local_survives(sc)
    sc["q4_equivalent"] = ok
    sc["q4_reason"] = reason
    sc["local_survives"] = loc
    sc["local_reason"] = loc_r
    sc["n_ids"] = int(idx.size)
    sc["n_unique"] = int(np.unique(idx).size)
    return sc


def kmeans(sample, k: int, iters: int = 6, seed: int = SEED):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = sample.shape[0]
    k = min(k, n)
    pick = rng.choice(n, size=k, replace=False)
    c = sample[pick].astype(np.float32, copy=True)
    for _ in range(iters):
        # assign
        # c: k x d, sample: n x d → dist via ||x||^2 + ||c||^2 - 2 x c.T
        x2 = (sample.astype(np.float32) ** 2).sum(1, keepdims=True)
        c2 = (c ** 2).sum(1)
        dots = gemm(sample.astype(np.float32), c.T)
        dist = x2 + c2[None, :] - 2.0 * dots
        lab = dist.argmin(1)
        for i in range(k):
            m = lab == i
            if np.any(m):
                c[i] = sample[m].mean(0)
    return c


def assign_codes(X, C):
    import numpy as np

    x2 = (X.astype(np.float32) ** 2).sum(1, keepdims=True)
    c2 = (C.astype(np.float32) ** 2).sum(1)
    dots = gemm(X.astype(np.float32), C.astype(np.float32).T)
    dist = x2 + c2[None, :] - 2.0 * dots
    return dist.argmin(1).astype(np.int32)


def rsvd_rows(W_f16, rank: int, seed: int = SEED, niter: int = 2):
    """Randomized SVD of a tall f16 table. Returns U (n x r) f32, S, Vt (r x d)."""
    import numpy as np

    n, d = W_f16.shape
    r = min(rank, n, d)
    rng = np.random.default_rng(seed)
    over = min(r + 8, d)
    Q = rng.standard_normal((d, over)).astype(np.float32)
    # Y = W @ Q, chunked
    Y = np.empty((n, over), dtype=np.float32)
    step = 8192
    for i in range(0, n, step):
        sl = W_f16[i : i + step].astype(np.float32)
        Y[i : i + step] = gemm(sl, Q)
    Qout, _ = np.linalg.qr(Y, mode="reduced")
    for _ in range(niter):
        # Z = W.T @ Qout
        Z = np.zeros((d, Qout.shape[1]), dtype=np.float32)
        for i in range(0, n, step):
            sl = W_f16[i : i + step].astype(np.float32)
            Z += gemm(sl.T, Qout[i : i + step])
        Qz, _ = np.linalg.qr(Z, mode="reduced")
        Y = np.empty((n, Qz.shape[1]), dtype=np.float32)
        for i in range(0, n, step):
            sl = W_f16[i : i + step].astype(np.float32)
            Y[i : i + step] = gemm(sl, Qz)
        Qout, _ = np.linalg.qr(Y, mode="reduced")
    B = np.zeros((Qout.shape[1], d), dtype=np.float32)
    for i in range(0, n, step):
        sl = W_f16[i : i + step].astype(np.float32)
        B += gemm(Qout[i : i + step].T, sl)
    U_b, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = gemm(Qout, U_b[:, :r])
    return U[:, :r], s[:r].astype(np.float32), Vt[:r]


# ---------------------------------------------------------------------------
# organ campaigns
# ---------------------------------------------------------------------------

def diag_energy(X):
    import numpy as np

    return (X.astype(np.float32) ** 2).sum(0)


def run_codec_ladder(W, X_fit, X_hold, *, organ, tensor, hess=False):
    """Grouped codecs + binary/ternary, weight-space and diag-H function space."""
    d = diag_energy(X_fit)
    Y = x_wt(X_hold, W)
    cands = []
    specs = [
        ("q4_g64", 4, 64, "incumbent_shipping"),
        ("q4_g128", 4, 128, "claimed_attn_floor_arithmetic"),
        ("q3_g64", 3, 64, "mlp_coherent_point_NOT_a_prior"),
        ("q2_g64", 2, 64, "mlp_survive_2.25_NOT_a_prior"),
    ]
    for name, bits, g, role in specs:
        if W.shape[1] % g != 0:
            continue
        for fn, tag in ((ws_rtn, "ws_rtn"), (lambda W, bits, group: aa_diag_scale(W, bits, group, d), "fs_diagH")):
            Wh, acc = fn(W, bits, g) if tag == "ws_rtn" else fn(W, bits, g)
            acc["role"] = role
            sc = eval_linear(W, Wh, X_hold, Y=Y)
            print(
                f"    {organ} {tensor} {tag}_{name} cos={sc['cosine']:.4f} "
                f"gain={sc['gain']:.3f} rel={sc['rel_fro']:.3f} "
                f"st={acc['storage_bpw']:.3f} q4={sc['q4_equivalent']}",
                flush=True,
            )
            cands.append(
                pack_candidate(
                    f"{tag}_{name}",
                    "grouped_absmax",
                    acc,
                    sc,
                    extra={"organ": organ, "tensor": tensor, "role": role},
                )
            )
            del Wh
    for tag, fn in (("binary_ws", lambda: binary_meanabs(W, 64, None)), ("binary_aa", lambda: binary_meanabs(W, 64, d))):
        Wh, acc = fn()
        sc = eval_linear(W, Wh, X_hold, Y=Y)
        print(
            f"    {organ} {tensor} {tag} cos={sc['cosine']:.4f} gain={sc['gain']:.3f} "
            f"st={acc['storage_bpw']:.3f} q4={sc['q4_equivalent']}",
            flush=True,
        )
        cands.append(pack_candidate(tag, "binary", acc, sc, extra={"organ": organ, "tensor": tensor}))
        del Wh
    for tag, fn in (("ternary_ws", lambda: ternary_fit(W, 64, None)), ("ternary_aa", lambda: ternary_fit(W, 64, d))):
        Wh, acc = fn()
        sc = eval_linear(W, Wh, X_hold, Y=Y)
        print(
            f"    {organ} {tensor} {tag} cos={sc['cosine']:.4f} gain={sc['gain']:.3f} "
            f"st={acc['storage_bpw']:.3f} q4={sc['q4_equivalent']}",
            flush=True,
        )
        cands.append(pack_candidate(tag, "ternary", acc, sc, extra={"organ": organ, "tensor": tensor}))
        del Wh
    trap = eval_linear(W, (SCALE_TRAP * W).astype("float32"), X_hold, Y=Y)
    cands.append(
        pack_candidate(
            "scale_001W",
            "control",
            {
                "storage_bpw": F16_BPW,
                "active_fused_bpw": F16_BPW,
                "active_cached_f16_bpw": F16_BPW,
                "scales_counted": True,
            },
            trap,
            extra={"organ": organ, "tensor": tensor, "trap": True},
        )
    )
    zero = eval_linear(W, W * 0, X_hold, Y=Y)
    cands.append(
        pack_candidate(
            "zero",
            "control",
            {
                "storage_bpw": 0.0,
                "active_fused_bpw": 0.0,
                "active_cached_f16_bpw": F16_BPW,
                "scales_counted": True,
            },
            zero,
            extra={"organ": organ, "tensor": tensor, "deletion": True},
        )
    )
    return cands


def run_rank_ladder(W, V, s, fro2, X_hold, *, organ, tensor, ks=None):
    rows, cols = W.shape
    if ks is None:
        ks = (16, 64, 128, 256, 512, 1024)
    ks = [k for k in ks if k <= V.shape[1]]
    energy = (s.astype("float64") ** 2)
    total = float(fro2) + 1e-30
    Y = x_wt(X_hold, W)
    cands = []
    for k in ks:
        Wh = aa_rank_hat(W, V, k)
        acc = bill_factors(rows, cols, k)
        sc = eval_linear(W, Wh, X_hold, Y=Y)
        print(
            f"    {organ} {tensor} aa_rank_{k} cos={sc['cosine']:.4f} "
            f"gain={sc['gain']:.3f} st={acc['storage_bpw']:.4f} q4={sc['q4_equivalent']}",
            flush=True,
        )
        captured = float(energy[:k].sum()) / total
        cands.append(
            pack_candidate(
                f"aa_rank_{k}",
                "activation_aware_lowrank",
                acc,
                sc,
                information_score={
                    "input_energy_captured": captured,
                    "rank": k,
                    "null_energy_uniform": k / cols,
                },
                extra={"organ": organ, "tensor": tensor},
            )
        )
        del Wh
    return cands


def head_pairwise(W_heads):
    """W_heads: [H, D, C] → mean/min pairwise row-flattened cosine."""
    import numpy as np

    h = W_heads.shape[0]
    flat = W_heads.reshape(h, -1).astype(np.float32)
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-30)
    G = gemm(flat, flat.T)
    iu = np.triu_indices(h, 1)
    vals = G[iu]
    return {
        "n_heads": int(h),
        "n_pairs": int(vals.size),
        "mean_cosine": float(vals.mean()) if vals.size else None,
        "min_cosine": float(vals.min()) if vals.size else None,
        "max_cosine": float(vals.max()) if vals.size else None,
        "null_independent": 0.0,
        "note": "pairwise cosine of flattened head matrices. Independent heads → ~0.",
    }


def shared_head_basis(W_heads, X_hold, k: int):
    """Stack heads, shared column PCs of the stacked W (information), score function."""
    import numpy as np

    h, d, c = W_heads.shape
    stacked = W_heads.reshape(h * d, c)
    # column basis of stacked W via Gram c x c
    G = gemm(stacked.T, stacked)
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    V = np.ascontiguousarray(evecs[:, order[:k]], dtype=np.float32)
    What_stack = project_W(stacked, V)
    What = What_stack.reshape(h * d, c)
    W = W_heads.reshape(h * d, c)
    sc = eval_linear(W, What, X_hold)
    acc = bill_factors(h * d, c, k)
    # shared V is paid once; per-head coefficients are (h*d)*k
    # bill_factors already uses rank*(rows+cols) which is the independent-style
    # shared-column bill: k*c + (h*d)*k — SAME formula, shared V is the cols term.
    acc["shared"] = True
    return What, acc, sc


def campaign_deltanet(parent, cap, fit_idx, hold_idx, cited) -> dict:
    import numpy as np

    print("\n## DELTANET")
    layers_data = {}
    all_cands = []
    X_cache = {}
    pcs_cache = {}
    W_qkv_cache = {}
    W_z_cache = {}
    W_out_cache = {}

    for layer in sorted(set(DN_PROBE + SHARE_DN)):
        X = load_X(cap, layer)
        X_cache[layer] = X
        print(f"  L{layer} X{tuple(X.shape)}")

    for layer in DN_PROBE:
        X = X_cache[layer]
        X_fit, X_hold = X[fit_idx], X[hold_idx]
        V, s, fro2, method = input_pcs(X_fit, 1024, seed=SEED + layer)
        pcs_cache[layer] = (V, s, fro2, method)
        print(f"  L{layer} input PCA {method} erank={participation_ratio(s.astype('float64')**2):.1f}")

        W_qkv = load_tensor(parent, tensor_name(layer, "linear_attn.in_proj_qkv.weight"))
        W_z = load_tensor(parent, tensor_name(layer, "linear_attn.in_proj_z.weight"))
        W_out = load_tensor(parent, tensor_name(layer, "linear_attn.out_proj.weight"))
        W_qkv_cache[layer] = W_qkv
        W_z_cache[layer] = W_z
        W_out_cache[layer] = W_out

        W_qkvz = fuse_q38_qkvz(W_qkv, W_z)
        # information: Q vs K alignment
        q = W_qkv[: DN_K_HEADS * DN_K_DIM]
        k = W_qkv[DN_K_HEADS * DN_K_DIM : 2 * DN_K_HEADS * DN_K_DIM]
        qh = q.reshape(DN_K_HEADS, DN_K_DIM, HIDDEN)
        kh = k.reshape(DN_K_HEADS, DN_K_DIM, HIDDEN)
        qk_pair = []
        for h in range(DN_K_HEADS):
            a = qh[h].ravel()
            b = kh[h].ravel()
            qk_pair.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))

        info = {
            "in_proj_qkv_shape": list(W_qkv.shape),
            "in_proj_z_shape": list(W_z.shape),
            "out_proj_shape": list(W_out.shape),
            "qkvz_fused_shape": list(W_qkvz.shape),
            "input_pca_method": method,
            "input_participation_ratio": participation_ratio(s.astype("float64") ** 2),
            "input_energy_top16": float((s[:16].astype("float64") ** 2).sum() / (fro2 + 1e-30)),
            "input_energy_top256": float((s[:256].astype("float64") ** 2).sum() / (fro2 + 1e-30)),
            "qk_head_cosine_mean": float(np.mean(qk_pair)),
            "qk_head_cosine_min": float(np.min(qk_pair)),
            "qk_head_cosine_null": 0.0,
            "q_head_pairwise": head_pairwise(qh),
            "k_head_pairwise": head_pairwise(kh),
        }

        c_qkv = run_codec_ladder(W_qkv, X_fit, X_hold, organ="deltanet", tensor=f"L{layer}.in_proj_qkv")
        c_z = run_codec_ladder(W_z, X_fit, X_hold, organ="deltanet", tensor=f"L{layer}.in_proj_z")
        # out_proj X is the mixer proxy, not post_attn_norm
        Xo = deltanet_out_proxy(X, W_qkvz)
        Xo_fit, Xo_hold = Xo[fit_idx], Xo[hold_idx]
        c_out = run_codec_ladder(W_out, Xo_fit, Xo_hold, organ="deltanet", tensor=f"L{layer}.out_proj")
        r_qkv = run_rank_ladder(W_qkv, V, s, fro2, X_hold, organ="deltanet", tensor=f"L{layer}.in_proj_qkv")
        r_z = run_rank_ladder(W_z, V, s, fro2, X_hold, organ="deltanet", tensor=f"L{layer}.in_proj_z")
        Vo, so, fro2o, _ = input_pcs(Xo_fit, 1024, seed=SEED + 100 + layer)
        r_out = run_rank_ladder(W_out, Vo, so, fro2o, Xo_hold, organ="deltanet", tensor=f"L{layer}.out_proj")

        # organ function: v*silu(z) under compressed in_proj (not just GEMV)
        organ_fn = []
        Y_org = deltanet_out_proxy(X_hold, W_qkvz)
        for bits, g, name in ((4, 64, "q4_g64"), (4, 128, "q4_g128"), (2, 64, "q2_g64"), (3, 64, "q3_g64")):
            Wh_q, acc_q = aa_diag_scale(W_qkv, bits, g, diag_energy(X_fit))
            Wh_z, acc_z = aa_diag_scale(W_z, bits, g, diag_energy(X_fit))
            Wh_f = fuse_q38_qkvz(Wh_q, Wh_z)
            Yh = deltanet_out_proxy(X_hold, Wh_f)
            sc = score_pair(Y_org, Yh)
            ok, reason = q4_healthy(sc)
            loc, loc_r = local_survives(sc)
            sc["q4_equivalent"] = ok
            sc["q4_reason"] = reason
            sc["local_survives"] = loc
            sc["local_reason"] = loc_r
            # bill the fused qkvz payload
            n_w = int(W_qkv.size + W_z.size)
            storage_bpw = (acc_q["storage_bits"] + acc_z["storage_bits"]) / n_w
            organ_fn.append(
                {
                    "name": f"organ_vsiluz_fs_diagH_{name}",
                    "family": "organ_function_not_gemv",
                    "storage_bpw": storage_bpw,
                    "active_fused_bpw": storage_bpw,
                    "active_cached_f16_bpw": F16_BPW,
                    "scales_counted": True,
                    "function": sc,
                    "note": "f(X)=v*silu(z); recurrent mix is NOT in this proxy. Real X_hold.",
                }
            )
            del Wh_q, Wh_z, Wh_f, Yh
        for k in (64, 256, 512, 1024):
            if k > V.shape[1]:
                continue
            Wh_q = aa_rank_hat(W_qkv, V, k)
            Wh_z = aa_rank_hat(W_z, V, k)
            Wh_f = fuse_q38_qkvz(Wh_q, Wh_z)
            Yh = deltanet_out_proxy(X_hold, Wh_f)
            sc = score_pair(Y_org, Yh)
            ok, reason = q4_healthy(sc)
            loc, loc_r = local_survives(sc)
            sc["q4_equivalent"] = ok
            sc["q4_reason"] = reason
            sc["local_survives"] = loc
            sc["local_reason"] = loc_r
            acc = bill_factors(W_qkv.shape[0] + W_z.shape[0], HIDDEN, k)
            organ_fn.append(
                {
                    "name": f"organ_vsiluz_aa_rank_{k}",
                    "family": "organ_function_lowrank",
                    "storage_bpw": acc["storage_bpw"],
                    "active_fused_bpw": acc["active_fused_bpw"],
                    "active_cached_f16_bpw": F16_BPW,
                    "scales_counted": True,
                    "function": sc,
                }
            )
            del Wh_q, Wh_z, Wh_f, Yh

        layer_cands = c_qkv + c_z + c_out + r_qkv + r_z + r_out + organ_fn
        all_cands.extend(layer_cands)
        layers_data[str(layer)] = {
            "information": info,
            "n_candidates": len(layer_cands),
            "out_proxy_note": "v*silu(z) from fused in_proj; not the recurrent S mix",
        }
        print(
            f"  L{layer} qkv q4_eq={[c['name'] for c in c_qkv if (c.get('function') or {}).get('q4_equivalent')]}"
        )
        gc.collect()

    # shared input basis across DN layers (function-space, not G035 weight-space)
    print("  shared input basis L0/L1/L32")
    Xs = [X_cache[L][fit_idx] for L in SHARE_DN]
    Xcat = np.concatenate(Xs, axis=0)
    Vsh, ssh, fro2sh, msh = input_pcs(Xcat, 1024, seed=SEED + 7)
    shared_rows = []
    for layer in SHARE_DN:
        if layer not in W_qkv_cache:
            W_qkv_cache[layer] = load_tensor(
                parent, tensor_name(layer, "linear_attn.in_proj_qkv.weight")
            )
        W = W_qkv_cache[layer]
        X_hold = X_cache[layer][hold_idx]
        for k in (256, 512, 1024):
            if k > Vsh.shape[1]:
                continue
            Wh = aa_rank_hat(W, Vsh, k)
            sc = eval_linear(W, Wh, X_hold)
            acc = bill_factors(W.shape[0], W.shape[1], k)
            # shared V paid once across 48 layers: amortised storage
            amort_elems = k * HIDDEN + DN_LAYERS * W.shape[0] * k
            amort_bpw = F16_BPW * amort_elems / (DN_LAYERS * W.size)
            shared_rows.append(
                {
                    "layer": layer,
                    "rank": k,
                    "per_layer_storage_bpw": acc["storage_bpw"],
                    "amortised_48_storage_bpw": amort_bpw,
                    "active_fused_bpw": acc["storage_bpw"],
                    "function": sc,
                    "input_pca": msh,
                }
            )
            del Wh
    shared_healthy = [
        r for r in shared_rows if (r.get("function") or {}).get("q4_equivalent")
    ]

    # state capacity (arithmetic, confirmed this run)
    qkvz_elems = 16384 * HIDDEN  # fused rows
    in_proj_qkv_bytes_f32 = 10240 * HIDDEN * 4
    rec_bytes = REC_ELEMS_PER_LAYER * 4
    state = {
        "rec_elems_per_layer": REC_ELEMS_PER_LAYER,
        "rec_bytes_f32_per_layer": rec_bytes,
        "rec_bytes_f32_48": rec_bytes * DN_LAYERS,
        "in_proj_qkv_bytes_f32_per_layer": in_proj_qkv_bytes_f32,
        "capacity_ratio_state_over_qkv": rec_bytes / in_proj_qkv_bytes_f32,
        "null": "if ratio >= 1, state could store W; measured ratio is << 1",
        "reading": (
            "S stores the image of W, not W. Rank-1 update is many-to-one. "
            "State CONTENT is reused; state TRAFFIC is re-read every token. "
            "A recurrent organ still streams in_proj/out_proj every decode step."
        ),
        "source_geometry": "crates/hawking-core/src/model/qwen38_geometry.rs",
        "cited_design": cited.get("dn_design_verdict"),
    }

    floor_cand = gated_organ_candidate(all_cands, key="q4_equivalent")
    local_cand = gated_organ_candidate(all_cands, key="local_survives")
    per_t = per_tensor_floors(all_cands, key="q4_equivalent")
    # sub-1 candidates that survive locally (the "opportunity")
    sub1 = [
        c
        for c in all_cands
        if c.get("storage_bpw") is not None
        and c["storage_bpw"] < 1.0
        and (c.get("function") or {}).get("q4_equivalent")
    ]
    sub1_local = [
        c
        for c in all_cands
        if c.get("storage_bpw") is not None
        and c["storage_bpw"] < 1.0
        and (c.get("function") or {}).get("local_survives")
    ]

    why = [
        "Do not treat DeltaNet as a dense MLP. The organ has a native recurrent op; "
        "TOKEN_NS already attributes the 10.6% component to that op plus helpers.",
        "State cannot store in_proj: rec_state/in_proj_qkv capacity ratio is "
        f"{rec_bytes / in_proj_qkv_bytes_f32:.4f}.",
        "Q and K heads are not a free shared basis "
        f"(mean pairwise Q-head cosine and Q-K cosine measured per layer).",
        "Activation-aware low-rank and grouped function-space codecs are scored on "
        "held-out real X and on the organ map v*silu(z). Sub-1 Q4-equivalent "
        f"clears: {len(sub1)}.",
        "MLP 1.85/2.25 is not this organ's floor.",
    ]

    return {
        "status": "MEASURED",
        "capture_site": "post_attn_norm (real; wrong residual vs input_layernorm, same honesty class as ATTENTION_FLOOR_REFIT)",
        "physical_cited": cited.get("dn_physical"),
        "information": {
            "state_capacity": state,
            "layers": {k: v["information"] for k, v in layers_data.items()},
            "shared_input_basis": {
                "layers": list(SHARE_DN),
                "method": msh,
                "n_fit_concat": int(Xcat.shape[0]),
                "rows": shared_rows,
                "n_q4_equivalent": len(shared_healthy),
                "null": "shared basis that loses to per-layer on hold Y is not a win",
            },
        },
        "function": {
            "site": "held-out post_attn_norm GEMV of in_proj_qkv/z and out_proj; organ map v*silu(z)",
            "n_hold": int(len(hold_idx)),
            "bar_q4_equivalent": "cosine>=0.990 AND gain>=0.50 AND surplus_over_null>=0.02",
            "bar_local": "gain>=0.50 AND rel_fro<=0.50 AND beats null (NOT Q4-equivalent)",
            "n_candidates": len(all_cands),
            "n_q4_equivalent": sum(
                1 for c in all_cands if (c.get("function") or {}).get("q4_equivalent")
            ),
            "n_sub1_q4_equivalent": len(sub1),
            "n_sub1_local_survives": len(sub1_local),
        },
        "candidates": all_cands,
        "per_tensor_floors": {
            t: {
                "method": c.get("name"),
                "storage_bpw": c.get("storage_bpw"),
                "active_fused_bpw": c.get("active_fused_bpw"),
                "function_cosine": (c.get("function") or {}).get("cosine"),
                "null": (c.get("function") or {}).get("null"),
            }
            for t, c in per_t.items()
        },
        "floor": _floor_record(
            floor_cand,
            organ="deltanet",
            bar="q4_equivalent",
            fallback_note=(
                "Organ floor is MAX over required tensors of each tensor's cheapest "
                "Q4-equivalent candidate (gated by the worst projection). Not the "
                "cheapest lucky tensor. MLP 1.85/2.25 is not this organ's floor."
            ),
        ),
        "local_survival_cheapest": _floor_record(
            local_cand, organ="deltanet", bar="local_survives", fallback_note="no local survivor"
        ),
        "why": why,
        "mlp_not_used_as_prior": True,
    }


def campaign_gqa(parent, cap, fit_idx, hold_idx, cited) -> dict:
    import numpy as np

    print("\n## GQA")
    all_cands = []
    layers_data = {}
    W_cache = {}
    X_cache = {}

    for layer in GQA_PROBE:
        X = load_X(cap, layer)
        X_cache[layer] = X
        X_fit, X_hold = X[fit_idx], X[hold_idx]
        V, s, fro2, method = input_pcs(X_fit, 1024, seed=SEED + 50 + layer)
        Wq = load_tensor(parent, tensor_name(layer, "self_attn.q_proj.weight"))
        Wk = load_tensor(parent, tensor_name(layer, "self_attn.k_proj.weight"))
        Wv = load_tensor(parent, tensor_name(layer, "self_attn.v_proj.weight"))
        Wo = load_tensor(parent, tensor_name(layer, "self_attn.o_proj.weight"))
        W_cache[layer] = {"q": Wq, "k": Wk, "v": Wv, "o": Wo}

        # q_proj is 24 heads × (q, gate) × 256 × 5120
        qg = Wq.reshape(GQA_HEADS, 2, GQA_HEAD_DIM, HIDDEN)
        q_heads = qg[:, 0]
        gate_heads = qg[:, 1]
        k_heads = Wk.reshape(GQA_KV_HEADS, GQA_HEAD_DIM, HIDDEN)
        v_heads = Wv.reshape(GQA_KV_HEADS, GQA_HEAD_DIM, HIDDEN)

        info = {
            "q_shape": list(Wq.shape),
            "k_shape": list(Wk.shape),
            "v_shape": list(Wv.shape),
            "o_shape": list(Wo.shape),
            "input_pca_method": method,
            "input_participation_ratio": participation_ratio(s.astype("float64") ** 2),
            "input_energy_top16": float((s[:16].astype("float64") ** 2).sum() / (fro2 + 1e-30)),
            "input_energy_top256": float((s[:256].astype("float64") ** 2).sum() / (fro2 + 1e-30)),
            "q_head_pairwise": head_pairwise(q_heads),
            "gate_head_pairwise": head_pairwise(gate_heads),
            "k_head_pairwise": head_pairwise(k_heads),
            "v_head_pairwise": head_pairwise(v_heads),
            "kv_group": GQA_HEADS // GQA_KV_HEADS,
            "note": (
                "GQA already shares K/V 6-way in the cache. Head-redundancy of Q is "
                "the remaining sharing question. Pairwise cosine ~1 would mean "
                "duplicate heads; ~0 means an aligned basis is not free."
            ),
        }

        # shared Q-head basis at several ranks
        shared_q = []
        for k in (64, 128, 256, 512):
            _Wh, acc, sc = shared_head_basis(q_heads, X_hold, k)
            shared_q.append(
                {
                    "rank": k,
                    "storage_bpw": acc["storage_bpw"],
                    "active_fused_bpw": acc["active_fused_bpw"],
                    "function": sc,
                    "what": "shared column basis of 24 Q heads (information of W, scored on hold Y=X W_q^T)",
                }
            )
        info["shared_q_head_basis"] = shared_q

        Xo = gqa_out_proxy(X, Wq, Wv)
        Xo_fit, Xo_hold = Xo[fit_idx], Xo[hold_idx]
        c_q = run_codec_ladder(Wq, X_fit, X_hold, organ="gqa", tensor=f"L{layer}.q_proj")
        c_k = run_codec_ladder(Wk, X_fit, X_hold, organ="gqa", tensor=f"L{layer}.k_proj")
        c_v = run_codec_ladder(Wv, X_fit, X_hold, organ="gqa", tensor=f"L{layer}.v_proj")
        c_o = run_codec_ladder(Wo, Xo_fit, Xo_hold, organ="gqa", tensor=f"L{layer}.o_proj")
        r_q = run_rank_ladder(Wq, V, s, fro2, X_hold, organ="gqa", tensor=f"L{layer}.q_proj")
        r_k = run_rank_ladder(Wk, V, s, fro2, X_hold, organ="gqa", tensor=f"L{layer}.k_proj")
        r_v = run_rank_ladder(Wv, V, s, fro2, X_hold, organ="gqa", tensor=f"L{layer}.v_proj")
        Vo, so, fro2o, _ = input_pcs(Xo_fit, 1024, seed=SEED + 80 + layer)
        r_o = run_rank_ladder(Wo, Vo, so, fro2o, Xo_hold, organ="gqa", tensor=f"L{layer}.o_proj")

        # KV compression: low-rank K and V as a cache operator (function of X)
        kv_rows = []
        for k in (16, 32, 64, 128, 256):
            if k > V.shape[1]:
                continue
            for tag, W in (("k", Wk), ("v", Wv)):
                Wh = aa_rank_hat(W, V, k)
                sc = eval_linear(W, Wh, X_hold)
                acc = bill_factors(W.shape[0], W.shape[1], k)
                kv_rows.append(
                    {
                        "proj": tag,
                        "rank": k,
                        "storage_bpw": acc["storage_bpw"],
                        "active_fused_bpw": acc["active_fused_bpw"],
                        "function": sc,
                    }
                )
                del Wh

        layer_cands = c_q + c_k + c_v + c_o + r_q + r_k + r_v + r_o
        all_cands.extend(layer_cands)
        layers_data[str(layer)] = {
            "information": info,
            "kv_state_compression": kv_rows,
        }
        print(
            f"  L{layer} q_proj q4_eq={[c['name'] for c in c_q if (c.get('function') or {}).get('q4_equivalent')]}"
        )
        gc.collect()

    # cross-layer shared input basis L3 vs L63
    print("  shared input basis L3/L63")
    Xcat = np.concatenate([X_cache[L][fit_idx] for L in GQA_PROBE], axis=0)
    Vsh, ssh, fro2sh, msh = input_pcs(Xcat, 1024, seed=SEED + 9)
    shared_rows = []
    for layer in GQA_PROBE:
        W = W_cache[layer]["q"]
        X_hold = X_cache[layer][hold_idx]
        for k in (256, 512, 1024):
            Wh = aa_rank_hat(W, Vsh, k)
            sc = eval_linear(W, Wh, X_hold)
            acc = bill_factors(W.shape[0], W.shape[1], k)
            amort_elems = k * HIDDEN + GQA_LAYERS_N * W.shape[0] * k
            amort_bpw = F16_BPW * amort_elems / (GQA_LAYERS_N * W.size)
            shared_rows.append(
                {
                    "layer": layer,
                    "rank": k,
                    "per_layer_storage_bpw": acc["storage_bpw"],
                    "amortised_16_storage_bpw": amort_bpw,
                    "function": sc,
                }
            )
            del Wh

    floor_cand = gated_organ_candidate(all_cands, key="q4_equivalent")
    per_t = per_tensor_floors(all_cands, key="q4_equivalent")
    refit = cited.get("attn_refit") or {}
    why_holds = [
        "ATTENTION_FLOOR_REFIT already re-fit grouped-absmax in function space: "
        "60/60 matched-bit rows favoured Hessian-optimal fitting, yet the composed "
        "GQA organ floor held at 4.125. Cited, not re-derived.",
        "This lane searched structure that grouped-absmax refit cannot see: Q-head "
        "redundancy, shared Q-head column basis, cross-layer shared input PCs, "
        "activation-aware low-rank on Q/K/V/O, KV low-rank as a cache operator, "
        "binary/ternary with scales counted.",
        "If Q heads are not copies (pairwise cosine far from 1) a shared-head "
        "operator is not a free lunch. If AA rank needed for 0.990 is near hidden, "
        "low-rank cannot undercut Q4 at the attention bar.",
        "K/V are already grouped 6-way; further KV compression is a NEW operator "
        "and is scored on hold Y of k_proj/v_proj, not assumed from GQA grouping.",
        "MLP 1.85/2.25 is not this organ's floor.",
    ]
    return {
        "status": "MEASURED",
        "capture_site": "post_attn_norm (real; same honesty class as ATTENTION_FLOOR_REFIT)",
        "physical_cited": cited.get("gqa_physical"),
        "cited_grouped_absmax_floor": refit,
        "information": {
            "layers": {k: v["information"] for k, v in layers_data.items()},
            "kv_state_compression": {k: v["kv_state_compression"] for k, v in layers_data.items()},
            "shared_input_basis": {
                "layers": list(GQA_PROBE),
                "method": msh,
                "rows": shared_rows,
            },
        },
        "function": {
            "site": "held-out post_attn_norm GEMV of q/k/v; o_proj on real-derived GQA-repeat(v)*sigmoid(q_gate)",
            "n_hold": int(len(hold_idx)),
            "bar_q4_equivalent": "cosine>=0.990 AND gain>=0.50 AND surplus_over_null>=0.02",
            "n_candidates": len(all_cands),
            "n_q4_equivalent": sum(
                1 for c in all_cands if (c.get("function") or {}).get("q4_equivalent")
            ),
        },
        "candidates": all_cands,
        "per_tensor_floors": {
            t: {
                "method": c.get("name"),
                "storage_bpw": c.get("storage_bpw"),
                "active_fused_bpw": c.get("active_fused_bpw"),
                "function_cosine": (c.get("function") or {}).get("cosine"),
                "null": (c.get("function") or {}).get("null"),
            }
            for t, c in per_t.items()
        },
        "floor": _floor_record(
            floor_cand,
            organ="gqa",
            bar="q4_equivalent",
            fallback_note=(
                "Organ floor is MAX over q/k/v/o of each projection's cheapest "
                "Q4-equivalent candidate. Grouped-absmax refit held at 4.125; "
                "this lane asks whether structure below that bar clears 0.990."
            ),
            cited_fallback=refit.get("fs_hess_organ_floor_storage_bpw"),
        ),
        "why_the_floor_holds": why_holds,
        "mlp_not_used_as_prior": True,
    }


def campaign_embed(parent, cap, fit_idx, hold_idx, tok_pack, cited) -> dict:
    import numpy as np

    print("\n## EMBEDDING / OUTPUT")
    W_e = load_tensor_f16(parent, "model.language_model.embed_tokens.weight")
    print(f"  embed f16 {tuple(W_e.shape)}")
    assert W_e.shape == (VOCAB, HIDDEN)

    fit_ids = tok_pack.get("fit_ids") or []
    hold_ids = tok_pack.get("hold_ids") or []
    counts = Counter(fit_ids + hold_ids)
    unique_obs = sorted(counts)
    rare_hold = [t for t in hold_ids if counts[t] == 1]
    freq_hold = [t for t in hold_ids if counts[t] >= 4]
    hot = [t for t, n in counts.items() if n >= 2]
    print(
        f"  tokens aligned={tok_pack.get('n_tokens_aligned')} "
        f"unique={len(unique_obs)} hold={len(hold_ids)} rare_hold={len(rare_hold)} "
        f"families={tok_pack.get('aligned_families')}"
    )

    # information: row-norm stats, sample pairwise, participation via rSVD
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(VOCAB, size=4096, replace=False)
    sample = W_e[sample_idx].astype(np.float32)
    U, s_e, Vt = rsvd_rows(W_e, 256, seed=SEED)
    erank = participation_ratio(s_e.astype("float64") ** 2)
    # tied? sample rows vs lm_head
    print("  loading lm_head f16 for tie / function")
    W_h = load_tensor_f16(parent, "lm_head.weight")
    assert W_h.shape == (VOCAB, HIDDEN)
    # row cosine of a 4096-row sample + of observed tokens
    def table_row_cosine(A, B, idx):
        a = A[idx].astype(np.float32)
        b = B[idx].astype(np.float32)
        return row_cosine(a, b), min_row_cosine(a, b)

    sample_cos, sample_min = table_row_cosine(W_e, W_h, sample_idx)
    if unique_obs:
        obs_idx = np.asarray(unique_obs, dtype=np.int64)
        obs_cos, obs_min = table_row_cosine(W_e, W_h, obs_idx)
    else:
        obs_cos, obs_min = None, None

    info = {
        "tie_word_embeddings_config": False,
        "embed_shape": [VOCAB, HIDDEN],
        "lm_head_shape": [VOCAB, HIDDEN],
        "rsvd_rank256_participation_ratio": erank,
        "rsvd_top16_energy": float((s_e[:16].astype("float64") ** 2).sum() / ((s_e.astype("float64") ** 2).sum() + 1e-30)),
        "tie_sample_4096_mean_row_cosine": sample_cos,
        "tie_sample_4096_min_row_cosine": sample_min,
        "tie_observed_mean_row_cosine": obs_cos,
        "tie_observed_min_row_cosine": obs_min,
        "null_tie_cosine": 0.0,
        "reading": (
            "config.tie_word_embeddings is false. Measured row cosine of embed vs "
            "lm_head says whether tying is approximately free. Cosine near 1 would "
            "halve table storage; cosine far from 1 forbids tying at the Q4 bar."
        ),
        "tokenizer_alignment": {
            "aligned_families": tok_pack.get("aligned_families"),
            "failed_families": tok_pack.get("failed_families"),
            "n_tokens_aligned": tok_pack.get("n_tokens_aligned"),
            "n_unique_observed": len(unique_obs),
            "n_hold": len(hold_ids),
            "n_rare_hold_count1": len(rare_hold),
            "n_freq_hold_count_ge4": len(freq_hold),
        },
    }

    print("  embed rSVD k=512 (ranks sliced from one factorization)")
    U512, s512, Vt512 = rsvd_rows(W_e, 512, seed=SEED)

    cands = []
    for k in (32, 64, 128, 256, 512):
        print(f"  embed rSVD slice k={k}", flush=True)
        Wh = gemm(U512[:, :k] * s512[:k][None, :], Vt512[:k]).astype(np.float16)
        acc = bill_factors(VOCAB, HIDDEN, k)
        fn = {
            "hold_gather": row_score_table(W_e, Wh, hold_ids),
            "rare_hold_gather": row_score_table(W_e, Wh, rare_hold),
            "freq_hold_gather": row_score_table(W_e, Wh, freq_hold),
            "random_vocab_4096": row_score_table(W_e, Wh, sample_idx.tolist()),
        }
        # gate the candidate's headline function on RARE tokens when we have them
        headline = fn["rare_hold_gather"] if rare_hold else fn["hold_gather"]
        cands.append(
            pack_candidate(
                f"embed_rsvd_{k}",
                "table_lowrank",
                acc,
                headline,
                extra={"function_slices": fn, "organ": "embedding"},
            )
        )
        del Wh
        gc.collect()

    # row codebook
    print("  embed codebook k-means")
    fit_sample_n = 8192
    fit_rows = rng.choice(VOCAB, size=fit_sample_n, replace=False)
    # hold out 20% of vocab from centroid FIT so unseen rows are a real hold
    vocab_perm = rng.permutation(VOCAB)
    vocab_fit = vocab_perm[: int(0.8 * VOCAB)]
    vocab_hold = vocab_perm[int(0.8 * VOCAB) :]
    km_sample = W_e[rng.choice(vocab_fit, size=min(8192, vocab_fit.size), replace=False)].astype(
        np.float32
    )
    for k_cb in (256,):
        print(f"    k={k_cb}")
        C = kmeans(km_sample, k_cb, iters=5, seed=SEED + k_cb)
        # assign in chunks
        codes = np.empty((VOCAB,), dtype=np.int32)
        step = 8192
        for i in range(0, VOCAB, step):
            sl = W_e[i : i + step].astype(np.float32)
            codes[i : i + step] = assign_codes(sl, C)
        Wh = C[codes].astype(np.float16)
        # accounting: f16 codebook + log2(k) index bits per row
        n_w = VOCAB * HIDDEN
        index_bits = VOCAB * math.log2(k_cb)
        cb_bits = k_cb * HIDDEN * F16_BPW
        storage_bits = index_bits + cb_bits
        acc = {
            "n_weights": n_w,
            "n_centroids": k_cb,
            "index_bits": index_bits,
            "codebook_bits": cb_bits,
            "storage_bits": storage_bits,
            "storage_bpw": storage_bits / n_w,
            "active_fused_bpw": F16_BPW,  # gather one reconstructed row = 16*H bits, billed vs table
            "active_bytes_per_token": HIDDEN * 2,  # f16 row after lookup
            "active_cached_f16_bpw": F16_BPW,
            "scales_counted": True,
            "note": (
                "storage = codebook + per-row index. Active at decode is one gathered "
                "row (embed table is not streamed). Report both."
            ),
        }
        fn = {
            "hold_gather": row_score_table(W_e, Wh, hold_ids),
            "rare_hold_gather": row_score_table(W_e, Wh, rare_hold),
            "freq_hold_gather": row_score_table(W_e, Wh, freq_hold),
            "unseen_vocab_rows": row_score_table(W_e, Wh, vocab_hold[:4096].tolist()),
            "fit_vocab_rows": row_score_table(W_e, Wh, vocab_fit[:4096].tolist()),
        }
        headline = fn["rare_hold_gather"] if rare_hold else fn["hold_gather"]
        cands.append(
            pack_candidate(
                f"embed_codebook_{k_cb}",
                "row_codebook",
                acc,
                headline,
                extra={"function_slices": fn, "organ": "embedding"},
            )
        )
        # sparse exceptions: replace worst 1% of FIT residuals with f16 originals
        # (exceptions are a storage add-on)
        print(f"    k={k_cb} + top1% exceptions")
        # residual energy on a 16k fit sample
        probe = vocab_fit[:16384]
        R = W_e[probe].astype(np.float32) - Wh[probe].astype(np.float32)
        energy = (R ** 2).sum(1)
        n_exc = max(1, int(0.01 * VOCAB))
        # approximate worst-1% using this sample's threshold
        thresh = float(np.quantile(energy, 0.99))
        # apply on probe only would understate; scan all in chunks
        exc_idx = []
        for i in range(0, VOCAB, step):
            sl = W_e[i : i + step].astype(np.float32)
            rec = Wh[i : i + step].astype(np.float32)
            e = ((sl - rec) ** 2).sum(1)
            hit = np.where(e >= thresh)[0] + i
            exc_idx.append(hit)
        exc_idx = np.concatenate(exc_idx)
        if exc_idx.size > n_exc:
            # keep the worst n_exc
            # recompute energy for candidates
            e_all = []
            for i in range(0, exc_idx.size, 8192):
                ix = exc_idx[i : i + 8192]
                e_all.append(((W_e[ix].astype(np.float32) - Wh[ix].astype(np.float32)) ** 2).sum(1))
            e_all = np.concatenate(e_all)
            keep = np.argpartition(e_all, -n_exc)[-n_exc:]
            exc_idx = exc_idx[keep]
        Wh2 = Wh.copy()
        Wh2[exc_idx] = W_e[exc_idx]
        extra_bits = int(exc_idx.size) * HIDDEN * F16_BPW + int(exc_idx.size) * 32  # row id
        acc2 = dict(acc)
        acc2["storage_bits"] = storage_bits + extra_bits
        acc2["storage_bpw"] = acc2["storage_bits"] / n_w
        acc2["n_exceptions"] = int(exc_idx.size)
        acc2["exception_frac"] = float(exc_idx.size) / VOCAB
        fn2 = {
            "hold_gather": row_score_table(W_e, Wh2, hold_ids),
            "rare_hold_gather": row_score_table(W_e, Wh2, rare_hold),
            "freq_hold_gather": row_score_table(W_e, Wh2, freq_hold),
            "unseen_vocab_rows": row_score_table(W_e, Wh2, vocab_hold[:4096].tolist()),
        }
        headline = fn2["rare_hold_gather"] if rare_hold else fn2["hold_gather"]
        cands.append(
            pack_candidate(
                f"embed_codebook_{k_cb}_exc1pct",
                "row_codebook_sparse_exceptions",
                acc2,
                headline,
                extra={"function_slices": fn2, "organ": "embedding"},
            )
        )
        del Wh, Wh2, codes, C
        gc.collect()

    # hot f16 / cold codebook-256
    print("  hot/cold split")
    C = kmeans(km_sample, 256, iters=4, seed=SEED + 3)
    codes = np.empty((VOCAB,), dtype=np.int32)
    for i in range(0, VOCAB, 8192):
        codes[i : i + 8192] = assign_codes(W_e[i : i + 8192].astype(np.float32), C)
    Wh = C[codes].astype(np.float16)
    hot_idx = np.asarray(sorted(set(hot + hold_ids + fit_ids)), dtype=np.int64)
    Wh[hot_idx] = W_e[hot_idx]
    n_w = VOCAB * HIDDEN
    storage_bits = (
        256 * HIDDEN * F16_BPW
        + VOCAB * 8.0
        + int(hot_idx.size) * HIDDEN * F16_BPW
        + int(hot_idx.size) * 32
    )
    acc = {
        "n_weights": n_w,
        "n_hot": int(hot_idx.size),
        "storage_bits": storage_bits,
        "storage_bpw": storage_bits / n_w,
        "active_fused_bpw": F16_BPW,
        "active_bytes_per_token": HIDDEN * 2,
        "active_cached_f16_bpw": F16_BPW,
        "scales_counted": True,
        "note": "hot tokens (observed in capture) stored f16; cold rows are codebook-256",
    }
    fn = {
        "hold_gather": row_score_table(W_e, Wh, hold_ids),
        "rare_hold_gather": row_score_table(W_e, Wh, rare_hold),
        "freq_hold_gather": row_score_table(W_e, Wh, freq_hold),
        "unseen_vocab_rows": row_score_table(W_e, Wh, vocab_hold[:4096].tolist()),
    }
    # hot/cold *stores* observed tokens exactly, so hold gather is near-perfect
    # if hold tokens were marked hot. Rare UNSEEN is the lexical test.
    headline = fn["unseen_vocab_rows"]
    headline_hold = fn["hold_gather"]
    cands.append(
        pack_candidate(
            "embed_hot_f16_cold_codebook256",
            "hot_cold",
            acc,
            headline,
            extra={
                "function_slices": fn,
                "organ": "embedding",
                "hold_gather_is_exact_if_hot": headline_hold,
                "note": (
                    "Hold tokens are in the hot set by construction, so hold gather "
                    "is not the lexical test. Unseen vocab rows are."
                ),
            },
        )
    )
    del Wh

    # grouped absmax along hidden, on a 16k-row slice for function (full table
    # q4 would be 2.5GB f32). Information bill is for the FULL table.
    print("  grouped absmax on 16k-row slice (bill is full-table)")
    slice_idx = np.concatenate(
        [
            np.asarray(unique_obs[: min(4000, len(unique_obs))], dtype=np.int64)
            if unique_obs
            else np.array([], dtype=np.int64),
            sample_idx[:2048],
            vocab_hold[:2048],
        ]
    )
    slice_idx = np.unique(slice_idx)
    Wslice = W_e[slice_idx].astype(np.float32)
    # dummy X = identity-like: for a table, function of gather IS row reconstruction,
    # so score_pair(Wslice, What_slice) is the function.
    for bits, g, name in ((4, 64, "q4_g64"), (4, 128, "q4_g128"), (3, 64, "q3_g64"), (2, 64, "q2_g64")):
        What, acc = ws_rtn(Wslice, bits, g)
        # rebill for full vocab
        acc_full = bill_grouped(VOCAB * HIDDEN, bits, VOCAB * (HIDDEN // g))
        sc = score_pair(Wslice, What)
        ok, reason = q4_healthy(sc)
        loc, loc_r = local_survives(sc)
        sc["q4_equivalent"] = ok
        sc["q4_reason"] = reason
        sc["local_survives"] = loc
        sc["local_reason"] = loc_r
        # rare subset inside the slice
        rare_in = [t for t in rare_hold if t in set(slice_idx.tolist())]
        map_pos = {int(t): i for i, t in enumerate(slice_idx.tolist())}
        rare_fn = None
        if rare_in:
            ri = np.array([map_pos[t] for t in rare_in], dtype=np.int64)
            rare_fn = score_pair(Wslice[ri], What[ri])
            ok_r, _ = q4_healthy(rare_fn)
            rare_fn["q4_equivalent"] = ok_r
        headline = rare_fn if rare_fn is not None else sc
        cands.append(
            pack_candidate(
                f"embed_ws_rtn_{name}_slice",
                "grouped_absmax_table",
                acc_full,
                headline,
                extra={
                    "function_slices": {"slice_all": sc, "rare_in_slice": rare_fn},
                    "organ": "embedding",
                    "slice_n": int(slice_idx.size),
                    "note": "codec fit is weight-space RTN on the slice; bill is full-table",
                },
            )
        )
        del What

    # 0.01*W trap on embed gather of hold ids
    if hold_ids:
        Y = W_e[np.asarray(hold_ids, dtype=np.int64)].astype(np.float32)
        trap = score_pair(Y, SCALE_TRAP * Y)
        trap_ok = bool(trap["cosine"] > 0.99 and trap["gain"] < 0.05 and trap["rel_fro"] > 0.9)
    else:
        trap = None
        trap_ok = None

    # lm_head function on last-layer X, observed+cold vocab
    print("  lm_head function-space on L63 X")
    X = load_X(cap, 63)
    X_hold = X[hold_idx]
    # cap hold rows for the big GEMM
    n_logit = min(256, X_hold.shape[0])
    Xh = X_hold[:n_logit]
    obs = np.unique(np.asarray(hold_ids + fit_ids, dtype=np.int64)) if (hold_ids or fit_ids) else sample_idx[:2048]
    cold = vocab_hold[:4096]
    mix = np.unique(np.concatenate([obs[:4096], cold]))
    Wmix = W_h[mix].astype(np.float32)
    Y = gemm(Xh, Wmix.T)
    # candidates for lm_head: tie to embed, low-rank, q4 slice of mix
    lm_cands = []
    # tie: use embed rows as lm_head
    Y_tie = gemm(Xh, W_e[mix].astype(np.float32).T)
    sc_tie = score_pair(Y, Y_tie)
    ok, reason = q4_healthy(sc_tie)
    loc, loc_r = local_survives(sc_tie)
    sc_tie["q4_equivalent"] = ok
    sc_tie["q4_reason"] = reason
    sc_tie["local_survives"] = loc
    sc_tie["local_reason"] = loc_r
    lm_cands.append(
        {
            "name": "lm_head_tied_to_embed",
            "family": "tied_representation",
            "storage_bpw": 0.0,  # incremental: reuse embed table
            "storage_bpw_note": "incremental vs storing a second table; embed still billed",
            "active_fused_bpw": F16_BPW,
            "active_bytes_per_token": VOCAB * 2,  # still a full vocab GEMV unless fused otherwise
            "scales_counted": True,
            "function": sc_tie,
            "vocab_mix_n": int(mix.size),
            "n_hold_rows": int(n_logit),
        }
    )
    # low-rank lm_head on mix: project mix rows onto embed rSVD? use input PCs of X
    Vx, sx, fro2x, mx = input_pcs(X[fit_idx], 512, seed=SEED + 11)
    for k in (64, 128, 256, 512):
        # W_hat[mix] = W[mix] @ Vk @ Vk.T
        Wh_mix = project_W(Wmix, Vx[:, :k])
        Yh = gemm(Xh, Wh_mix.T)
        sc = score_pair(Y, Yh)
        ok, reason = q4_healthy(sc)
        loc, loc_r = local_survives(sc)
        sc["q4_equivalent"] = ok
        sc["q4_reason"] = reason
        sc["local_survives"] = loc
        sc["local_reason"] = loc_r
        acc = bill_factors(VOCAB, HIDDEN, k)
        lm_cands.append(
            {
                "name": f"lm_head_aa_rank_{k}",
                "family": "activation_aware_lowrank",
                "storage_bpw": acc["storage_bpw"],
                "active_fused_bpw": acc["active_fused_bpw"],
                "active_cached_f16_bpw": F16_BPW,
                "scales_counted": True,
                "function": sc,
                "vocab_mix_n": int(mix.size),
            }
        )
        del Wh_mix, Yh
    # grouped q4/q3/q2 on the mix (proxy for the table)
    for bits, g, name in ((4, 64, "q4_g64"), (3, 64, "q3_g64"), (2, 64, "q2_g64")):
        What, acc = ws_rtn(Wmix, bits, g)
        Yh = gemm(Xh, What.T)
        sc = score_pair(Y, Yh)
        ok, reason = q4_healthy(sc)
        loc, loc_r = local_survives(sc)
        sc["q4_equivalent"] = ok
        sc["q4_reason"] = reason
        sc["local_survives"] = loc
        sc["local_reason"] = loc_r
        acc_full = bill_grouped(VOCAB * HIDDEN, bits, VOCAB * (HIDDEN // g))
        # rare mix columns
        rare_set = set(rare_hold)
        rare_cols = [i for i, t in enumerate(mix.tolist()) if t in rare_set]
        rare_sc = None
        if rare_cols:
            rc = np.asarray(rare_cols, dtype=np.int64)
            rare_sc = score_pair(Y[:, rc], Yh[:, rc])
            ok_r, _ = q4_healthy(rare_sc)
            rare_sc["q4_equivalent"] = ok_r
        lm_cands.append(
            {
                "name": f"lm_head_ws_rtn_{name}_mix",
                "family": "grouped_absmax",
                "storage_bpw": acc_full["storage_bpw"],
                "active_fused_bpw": acc_full["active_fused_bpw"],
                "active_cached_f16_bpw": F16_BPW,
                "scales_counted": True,
                "function": rare_sc if rare_sc is not None else sc,
                "function_all_mix": sc,
                "function_rare_cols": rare_sc,
                "vocab_mix_n": int(mix.size),
            }
        )
        del What, Yh

    # greedy argmax among FULL vocab on a handful of hold rows (true lexical act)
    print("  lm_head full-vocab argmax on 16 hold rows")
    n_arg = min(16, X_hold.shape[0])
    teacher_am = []
    # compute teacher argmax in vocab chunks to bound memory
    for i in range(n_arg):
        x = X_hold[i]
        best = -1
        best_v = -1e30
        step = 8192
        for r0 in range(0, VOCAB, step):
            sl = W_h[r0 : r0 + step].astype(np.float32)
            logits = sl @ x
            j = int(logits.argmax())
            v = float(logits[j])
            if v > best_v:
                best_v = v
                best = r0 + j
        teacher_am.append(best)
    # student: q4 of a random 8k-row is NOT full vocab. Instead q4-reconstruct
    # the 16 teacher rows + nearby? We need a real compressed table student.
    # Use ws_rtn on 16k random rows is not argmax-legal.
    # Do q4 on the MIX only for argmax-among-mix, stated as such.
    What_mix, _ = ws_rtn(Wmix, 4, 64)
    student_am_mix = []
    teacher_am_mix = []
    for i in range(n_arg):
        x = Xh[i] if i < Xh.shape[0] else X_hold[i]
        tlog = Wmix @ x
        slog = What_mix @ x
        teacher_am_mix.append(int(mix[int(tlog.argmax())]))
        student_am_mix.append(int(mix[int(slog.argmax())]))
    argmax_mix_agree = float(np.mean([a == b for a, b in zip(teacher_am_mix, student_am_mix)]))

    floor_rare = cheapest_healthy(cands, key="q4_equivalent")
    # also a mean-hold floor that ignores rare gating
    hold_gated = []
    for c in cands:
        sl = (c.get("function_slices") or {}).get("hold_gather")
        if sl and sl.get("q4_equivalent"):
            hold_gated.append(c)
        elif (c.get("function") or {}).get("q4_equivalent") and c.get("family") == "grouped_absmax_table":
            hold_gated.append(c)
    hold_floor = None
    if hold_gated:
        hold_floor = min(hold_gated, key=lambda c: float(c["storage_bpw"]))

    why = [
        "Embedding is a gather of one row per token (active 2720 B at f16); lm_head "
        "is a full-vocab GEMV every token (active streams the table). Storage and "
        "active therefore disagree by orders of magnitude — report both.",
        "tie_word_embeddings is false. Tying is a measured cosine, not a config flag.",
        "A codebook that looks cheap on mean row cosine can still destroy rare tokens. "
        "The organ floor is gated on rare/unseen rows, not on the hot mean.",
        "MLP 1.85/2.25 is not this organ's floor.",
    ]
    return {
        "status": "MEASURED",
        "capture_site": {
            "embed": "tokenizer-aligned hold token ids from capture_diverse2 hardcoded families (real ids, not gaussian)",
            "lm_head": "L63 post_attn_norm hold X (real distribution; not final RMSNorm — stated)",
        },
        "physical_cited": {
            "embedding": cited.get("embed_physical"),
            "output": cited.get("output_physical"),
        },
        "information": info,
        "function": {
            "embed_gather": "row cosine of W[ids] vs What[ids] on hold / rare / frequent / unseen vocab",
            "lm_head": "Y = X_hold @ W[mix].T vs What, mix = observed tokens + cold sample; plus mix-argmax on 16 rows",
            "n_hold_ids": len(hold_ids),
            "n_rare_hold": len(rare_hold),
            "lm_head_argmax_mix_agree_q4": argmax_mix_agree,
            "lm_head_teacher_fullvocab_argmax_16": teacher_am,
            "null_argmax_agree": 1.0 / VOCAB,
        },
        "scale_trap_embed_gather": {
            "score": trap,
            "rejects_scaled_artifact": trap_ok,
            "pass_if": "cosine~1 and gain~0.01 and rel_fro~0.99",
        },
        "candidates": cands,
        "lm_head_candidates": lm_cands,
        "floor": _floor_record(
            floor_rare,
            organ="embedding_output",
            bar="q4_equivalent_on_rare_or_unseen",
            fallback_note=(
                "If no cheap codebook/low-rank clears rare/unseen at 0.990, the floor "
                "is grouped-absmax of the table (q4_g64=4.25 or q4_g128=4.125), with "
                "embed ACTIVE still one gathered row."
            ),
        ),
        "mean_hold_floor_not_lexical": _floor_record(
            hold_floor, organ="embedding_output", bar="q4_equivalent_on_hold_mean", fallback_note="none"
        ),
        "why": why,
        "mlp_not_used_as_prior": True,
        "did_not_load_second_27b": True,
    }


def _floor_record(cand, *, organ, bar, fallback_note, cited_fallback=None) -> dict:
    if cand is None:
        return {
            "status": "MEASURED",
            "organ": organ,
            "bar": bar,
            "storage_bpw": cited_fallback,
            "active_fused_bpw": cited_fallback,
            "active_cached_f16_bpw": F16_BPW,
            "method": None,
            "healthy": False,
            "candidate": None,
            "note": fallback_note,
            "null": "a missing healthy candidate is not a 0-bpw floor",
        }
    fn = cand.get("function") or {}
    return {
        "status": "MEASURED",
        "organ": organ,
        "bar": bar,
        "storage_bpw": cand.get("storage_bpw"),
        "active_fused_bpw": cand.get("active_fused_bpw"),
        "active_cached_f16_bpw": cand.get("active_cached_f16_bpw", F16_BPW),
        "scales_counted": cand.get("scales_counted", True),
        "method": cand.get("name"),
        "family": cand.get("family"),
        "healthy": bool(fn.get("q4_equivalent") or fn.get("local_survives")),
        "function": {
            "cosine": fn.get("cosine"),
            "gain": fn.get("gain"),
            "rel_fro": fn.get("rel_fro"),
            "null": fn.get("null"),
            "surplus_over_null": fn.get("surplus_over_null"),
            "q4_equivalent": fn.get("q4_equivalent"),
            "q4_reason": fn.get("q4_reason"),
        },
        "candidate": {k: cand[k] for k in ("name", "family", "storage_bpw", "active_fused_bpw") if k in cand},
        "note": fallback_note,
        "null": fn.get("null"),
    }


def recompute_floors(doc: dict) -> dict:
    """Recompute gated organ floors from stored candidates. Safe to run on a finished receipt."""
    organs = doc.get("organs") or {}
    if "deltanet" in organs:
        o = organs["deltanet"]
        cands = o.get("candidates") or []
        per_t = per_tensor_floors(cands, key="q4_equivalent")
        o["per_tensor_floors"] = {
            t: {
                "method": c.get("name"),
                "storage_bpw": c.get("storage_bpw"),
                "active_fused_bpw": c.get("active_fused_bpw"),
                "function_cosine": (c.get("function") or {}).get("cosine"),
                "null": (c.get("function") or {}).get("null"),
            }
            for t, c in per_t.items()
        }
        o["floor"] = _floor_record(
            gated_organ_candidate(cands, key="q4_equivalent"),
            organ="deltanet",
            bar="q4_equivalent",
            fallback_note=(
                "Organ floor is MAX over required tensors of each tensor's cheapest "
                "Q4-equivalent candidate (gated by the worst projection)."
            ),
        )
        o["local_survival_cheapest"] = _floor_record(
            gated_organ_candidate(cands, key="local_survives"),
            organ="deltanet",
            bar="local_survives",
            fallback_note="no local survivor",
        )
    if "gqa" in organs:
        o = organs["gqa"]
        cands = o.get("candidates") or []
        per_t = per_tensor_floors(cands, key="q4_equivalent")
        o["per_tensor_floors"] = {
            t: {
                "method": c.get("name"),
                "storage_bpw": c.get("storage_bpw"),
                "active_fused_bpw": c.get("active_fused_bpw"),
                "function_cosine": (c.get("function") or {}).get("cosine"),
                "null": (c.get("function") or {}).get("null"),
            }
            for t, c in per_t.items()
        }
        refit = o.get("cited_grouped_absmax_floor") or {}
        o["floor"] = _floor_record(
            gated_organ_candidate(cands, key="q4_equivalent"),
            organ="gqa",
            bar="q4_equivalent",
            fallback_note=(
                "Organ floor is MAX over q/k/v/o of each projection's cheapest "
                "Q4-equivalent candidate."
            ),
            cited_fallback=refit.get("fs_hess_organ_floor_storage_bpw")
            or refit.get("gqa_organ_floor_fs_hess"),
        )
    if "embedding_output" in organs:
        o = organs["embedding_output"]
        cands = o.get("candidates") or []
        lm = o.get("lm_head_candidates") or []
        per_e = per_tensor_floors(cands, key="q4_equivalent")
        per_l = per_tensor_floors(lm, key="q4_equivalent")
        o["per_tensor_floors"] = {
            "embed": {
                t: {
                    "method": c.get("name"),
                    "storage_bpw": c.get("storage_bpw"),
                    "active_fused_bpw": c.get("active_fused_bpw"),
                    "function_cosine": (c.get("function") or {}).get("cosine"),
                    "null": (c.get("function") or {}).get("null"),
                }
                for t, c in per_e.items()
            },
            "lm_head": {
                t: {
                    "method": c.get("name"),
                    "storage_bpw": c.get("storage_bpw"),
                    "active_fused_bpw": c.get("active_fused_bpw"),
                    "function_cosine": (c.get("function") or {}).get("cosine"),
                    "null": (c.get("function") or {}).get("null"),
                }
                for t, c in per_l.items()
            },
        }
        embed_floor = gated_organ_candidate(cands, key="q4_equivalent")
        lm_floor = gated_organ_candidate(lm, key="q4_equivalent")
        o["embed_floor"] = _floor_record(
            embed_floor,
            organ="embedding_output",
            bar="q4_equivalent_on_rare_or_unseen",
            fallback_note="embed table floor; active is one gathered row",
        )
        o["lm_head_floor"] = _floor_record(
            lm_floor,
            organ="embedding_output",
            bar="q4_equivalent",
            fallback_note="lm_head GEMV floor on hold X vs observed+cold mix",
        )
        # organ summary: gated by the more expensive of the two tables
        pick = None
        for cand in (embed_floor, lm_floor):
            if cand is None or cand.get("storage_bpw") is None:
                continue
            if pick is None or float(cand["storage_bpw"]) > float(pick["storage_bpw"]):
                pick = cand
        o["floor"] = _floor_record(
            pick,
            organ="embedding_output",
            bar="q4_equivalent_on_rare_or_unseen",
            fallback_note=(
                "Summary storage floor is MAX(embed table, lm_head) at the Q4 bar. "
                "Embed ACTIVE is one gathered row (table not streamed). lm_head ACTIVE "
                "streams the table. See embed_floor and lm_head_floor."
            ),
        )
        # Embed decode does not stream the table.
        ef = o.get("embed_floor") or {}
        st = ef.get("storage_bpw")
        if st is not None:
            o["floor"]["embed_active_bytes_per_token"] = HIDDEN * float(st) / 8.0
            o["floor"]["embed_active_note"] = (
                "gather one row: active_bytes = hidden * storage_bpw / 8. "
                "The table itself is not in the decode stream."
            )
            o["embed_floor"]["active_bytes_per_token"] = HIDDEN * float(st) / 8.0
            o["embed_floor"]["table_not_streamed"] = True
        lf = o.get("lm_head_floor") or {}
        lst = lf.get("storage_bpw")
        if lst is not None:
            o["lm_head_floor"]["active_bytes_per_token"] = VOCAB * HIDDEN * float(lst) / 8.0
            o["lm_head_floor"]["table_streamed_every_token"] = True
    floors = {
        k: (organs[k].get("floor") or {}).get("storage_bpw")
        for k in ("deltanet", "gqa", "embedding_output")
        if k in organs
    }
    v = doc.get("verdict") or {}
    v["floors_storage_bpw"] = floors
    v["floor_rule"] = (
        "per-organ floor = max over required tensors of cheapest Q4-equivalent "
        "storage_bpw on held-out real activations. Not the cheapest lucky tensor. "
        "Not the MLP 1.85/2.25 bracket."
    )
    doc["verdict"] = v
    return doc


def load_citations() -> dict:
    census, src_c = git_json(ORGAN_CENSUS_RECEIPT)
    refit, src_r = git_json(ATTN_REFIT_RECEIPT)
    dn, src_d = git_json(DN_DESIGN_RECEIPT)
    ternary, src_t = git_json(MLP_FAIL_RECEIPT)
    q2f, src_q = git_json(MLP_SURVIVE_RECEIPT)
    organs = (census or {}).get("organs") or {}
    v = (refit or {}).get("verdict") or {}
    return {
        "census_source": src_c,
        "refit_source": src_r,
        "dn_design_source": src_d,
        "mlp_fail_source": src_t,
        "mlp_survive_source": src_q,
        "embed_physical": (organs.get("embedding") or {}).get("physical"),
        "gqa_physical": (organs.get("attention_gqa") or {}).get("physical"),
        "dn_physical": (organs.get("deltanet") or {}).get("physical"),
        "output_physical": (organs.get("output") or {}).get("physical"),
        "dn_design_verdict": (dn or {}).get("verdict"),
        "attn_refit": {
            "decision": v.get("decision"),
            "n_matched_rows": v.get("n_matched_rows"),
            "fs_hess_beats_ws_rel_fro": v.get("fs_hess_beats_ws_rel_fro"),
            "organ_floor_moved_below_4.125": v.get("organ_floor_moved_below_4.125"),
            "gqa_organ_floor_fs_hess": (v.get("gqa_organ_floor_fs_hess") or {}).get(
                "organ_floor_storage_bpw"
            ),
            "deltanet_organ_floor_fs_hess": (v.get("deltanet_organ_floor_fs_hess") or {}).get(
                "organ_floor_storage_bpw"
            ),
            "fs_hess_organ_floor_storage_bpw": (v.get("gqa_organ_floor_fs_hess") or {}).get(
                "organ_floor_storage_bpw"
            ),
            "reason": v.get("reason"),
            "source": ATTN_REFIT_RECEIPT,
        },
        "mlp": {
            "fail_bpw": MLP_FAIL_BPW,
            "survive_bpw": MLP_SURVIVE_BPW,
            "fail_receipt": MLP_FAIL_RECEIPT,
            "survive_receipt": MLP_SURVIVE_RECEIPT,
            "fail_argmax": ((ternary or {}).get("rungs") or [{}])[-1]
            if ternary
            else None,
        },
    }


def main() -> int:
    t_all = time.time()
    if VISION_PY.is_file() and Path(sys.executable).resolve() != VISION_PY.resolve():
        os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])

    try:
        import torch

        torch.set_num_threads(min(12, os.cpu_count() or 8))
        torch_s = f"{torch.__version__} mps={torch.backends.mps.is_available()}"
    except Exception as e:
        torch_s = f"unavailable ({e})"

    parent = find_parent()
    cap = find_capture()
    tok_path = find_tokenizer()
    cited = load_citations()
    manifest = {}
    mp = cap / "manifest.json"
    if mp.is_file():
        manifest = json.loads(mp.read_text())
    X0 = load_X(cap, 0)
    n_tokens = int(X0.shape[0])
    fit_idx, hold_idx = split_from_manifest(manifest, n_tokens)
    del X0

    tok_pack = {
        "aligned_families": [],
        "failed_families": ["tokenizer_unavailable"],
        "fit_ids": [],
        "hold_ids": [],
        "n_tokens_aligned": 0,
    }
    if tok_path is not None:
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tok_path))
            tok_pack = reconstruct_token_ids(tok, manifest)
        except Exception as e:
            tok_pack["failed_families"] = [{"reason": f"tokenizer failed: {type(e).__name__}: {e}"}]

    print("ORGAN FRONTIERS")
    print("=" * 72)
    print(f"git_head: {git_head()}")
    print(f"python:   {sys.executable}")
    print(f"torch:    {torch_s}")
    print(f"parent:   {parent}")
    print(f"capture:  {cap}  n={n_tokens} fit={len(fit_idx)} hold={len(hold_idx)}")
    print(f"tokenizer aligned families: {tok_pack.get('aligned_families')}")
    print("teacher:  qualified parent BF16 tensors, one at a time. no second 27B.")
    print()

    results = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head(),
        "python": sys.executable,
        "torch": torch_s,
        "parent": str(parent),
        "did_not_load_second_27b": True,
        "question": (
            "What is each of DeltaNet, GQA, and embedding/output's OWN measured "
            "information/function floor on real held-out activations, without "
            "transferring the MLP 1.85-fail / 2.25-survive bracket?"
        ),
        "mlp_not_extrapolated": {
            "do_not_transfer": True,
            "fail_bpw": numbered(
                MLP_FAIL_BPW,
                status="CITED",
                null="not a floor for DeltaNet/GQA/embed",
                unit="bpw",
                source=MLP_FAIL_RECEIPT,
                note="whole-model uniform ternary; argmax flipped (9714 vs 10895)",
            ),
            "survive_bpw": numbered(
                MLP_SURVIVE_BPW,
                status="CITED",
                null="not a floor for DeltaNet/GQA/embed",
                unit="bpw",
                source=MLP_SURVIVE_RECEIPT,
                note="whole-model uniform q2 g64; argmax agreed",
            ),
            "applies_to": "mlp / whole-model uniform mix, not these three organs",
        },
        "capture": {
            "path": str(cap),
            "site": "post_attn_norm",
            "n_tokens": n_tokens,
            "n_fit": int(len(fit_idx)),
            "n_hold": int(len(hold_idx)),
            "hidden": HIDDEN,
            "not_gaussian": True,
            "not_llama_server_teacher": True,
            "split_rule": manifest.get("split_rule"),
            "manifest_families": manifest.get("families"),
            "source_note": (
                "Phase-B capture_diverse2: real BF16 parent MLX full-model forward. "
                "Wrong residual point vs input_layernorm for mixer in-proj; real "
                "distribution. Known-invalid class is Gaussian-proxy; this is not that."
            ),
            "tokenizer": str(tok_path) if tok_path else None,
            "token_alignment": tok_pack,
        },
        "accounting_rules": {
            "scales_counted": True,
            "q4_g64_storage_bpw": grouped_storage_bpw(4, 64),
            "q4_g128_storage_bpw": grouped_storage_bpw(4, 128),
            "q3_g64_storage_bpw": grouped_storage_bpw(3, 64),
            "q2_g64_storage_bpw": grouped_storage_bpw(2, 64),
            "ternary_5in8_g64_storage_bpw": ternary_5in8_storage_bpw(64),
            "binary_g64_storage_bpw": binary_storage_bpw(64),
            "rule": "A 16-bit scale per group of 64 is 0.25 bpw, not free. Report storage AND active, or neither.",
        },
        "quality_bars": {
            "q4_equivalent": "hold cosine >= 0.990 AND gain >= 0.50 AND surplus_over_null >= 0.02",
            "local_survives": "gain >= 0.50 AND rel_fro <= 0.50 AND beats constant-mean null; NOT Q4-equivalent",
            "embed_lexical": "rare/unseen row cosine at the same 0.990 bar; mean over hot tokens is not sufficient",
            "cosine_is_not_go": True,
        },
        "organs": {},
        "scale_trap": {},
        "verdict": {},
        "what_i_watched_fail": [],
        "wall_s": None,
    }

    # scale trap on GQA q_proj L3 (mixer) — required instrument
    print("## SCALE TRAP")
    Wq = load_tensor(parent, tensor_name(3, "self_attn.q_proj.weight"))
    X3 = load_X(cap, 3)
    Y = x_wt(X3[hold_idx], Wq)
    trap = score_pair(Y, SCALE_TRAP * Y)
    ident = score_pair(Y, Y)
    rejects = bool(trap["cosine"] > 0.99 and trap["gain"] < 0.05 and trap["rel_fro"] > 0.9)
    print(
        f"  identity cosine={ident['cosine']:.6f} gain={ident['gain']:.6f} rel_fro={ident['rel_fro']:.6f}"
    )
    print(
        f"  0.01*W   cosine={trap['cosine']:.6f} gain={trap['gain']:.6f} rel_fro={trap['rel_fro']:.6f} rejects={rejects}"
    )
    results["scale_trap"] = {
        "site": "L3.self_attn.q_proj on real hold X",
        "identity": ident,
        "scaled_0p01": trap,
        "rejects_scaled_artifact": rejects,
        "pass_if": "cosine~1 and gain~0.01 and rel_fro~0.99. GO uses gain+rel_fro+0.99-cosine, never cosine alone.",
        "null": "a metric that accepts 0.01*W is not a GO metric",
    }
    results["what_i_watched_fail"].append(
        f"0.01*W on L3 q_proj: cosine={trap['cosine']:.6f} (blind) gain={trap['gain']:.6f} "
        f"rel_fro={trap['rel_fro']:.6f} (rejects={rejects})"
    )
    if not rejects:
        results["verdict"] = {"decision": "NO-GO", "reason": "scale trap failed"}
        results["wall_s"] = time.time() - t_all
        _write(results)
        return 2
    del Wq, X3, Y
    gc.collect()
    _write(results)

    results["organs"]["deltanet"] = campaign_deltanet(parent, cap, fit_idx, hold_idx, cited)
    _write(results)
    gc.collect()

    results["organs"]["gqa"] = campaign_gqa(parent, cap, fit_idx, hold_idx, cited)
    _write(results)
    gc.collect()

    results["organs"]["embedding_output"] = campaign_embed(
        parent, cap, fit_idx, hold_idx, tok_pack, cited
    )
    results = recompute_floors(results)
    _write(results)

    floors = {
        k: (results["organs"][k].get("floor") or {}).get("storage_bpw")
        for k in ("deltanet", "gqa", "embedding_output")
    }
    results["verdict"] = {
        "decision": "THREE_INDEPENDENT_FLOORS",
        "floors_storage_bpw": floors,
        "do_not_transfer_mlp": True,
        "mlp_fail_bpw": MLP_FAIL_BPW,
        "mlp_survive_bpw": MLP_SURVIVE_BPW,
        "attention_refit_floor_moved": (cited.get("attn_refit") or {}).get(
            "organ_floor_moved_below_4.125"
        ),
        "reading": (
            "Each organ has its own measured floor with evidence on real hold "
            "activations. Information and function are separate. A cheap storage "
            "number without a health verdict is not a result."
        ),
    }
    results["citations"] = {
        "mlp_fail": MLP_FAIL_RECEIPT,
        "mlp_survive": MLP_SURVIVE_RECEIPT,
        "attention_floor_refit": ATTN_REFIT_RECEIPT,
        "organ_census": ORGAN_CENSUS_RECEIPT,
        "deltanet_design": DN_DESIGN_RECEIPT,
        "loaded_from": {
            "census": cited.get("census_source"),
            "refit": cited.get("refit_source"),
            "dn_design": cited.get("dn_design_source"),
        },
    }
    results["wall_s"] = time.time() - t_all
    _write(results)
    print()
    print(f"WROTE {RECEIPT}  wall={results['wall_s']:.1f}s")
    print(f"floors {floors}")
    return 0


# ---------------------------------------------------------------------------
# unit tests (also imported by test_organ_frontiers.py)
# ---------------------------------------------------------------------------

def test_grouped_bpw_counts_scales():
    assert abs(grouped_storage_bpw(4, 64) - 4.25) < 1e-12
    assert abs(grouped_storage_bpw(4, 128) - 4.125) < 1e-12
    assert abs(grouped_storage_bpw(2, 64) - 2.25) < 1e-12
    assert abs(grouped_storage_bpw(3, 64) - 3.25) < 1e-12
    assert abs(ternary_5in8_storage_bpw(64) - (TRIT_PACK_5IN8 + 0.25)) < 1e-12
    assert abs(binary_storage_bpw(64) - 1.25) < 1e-12
    assert grouped_storage_bpw(4, 64) != 4.0


def test_lowrank_bpw_glm_shape():
    # GLM up_proj [2048, 6144] k=16 → 0.16667
    bpw = lowrank_f16_bpw(2048, 6144, 16)
    assert abs(bpw - 0.16666666666666666) < 1e-9


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--self-test", "--unit"):
        test_grouped_bpw_counts_scales()
        test_lowrank_bpw_glm_shape()
        print("unit tests passed")
        sys.exit(0)
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        raise
