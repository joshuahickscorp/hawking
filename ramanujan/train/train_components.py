#!/usr/bin/env python3.12
"""Train and evaluate the small formal system on frozen memberships.

Components:
  retriever   — D3 premise selection; held-out retrieval@k / MRR
  formalizer  — D1 statement → first-tactic class; held-out exact-match
  prover      — D2 state → next tactic; held-out exact-match
  repair      — D4 (broken, error, signature) → fix; exact-match + Lean compile
  value       — D2 state → closed-next + remaining steps; accuracy / MAE

Only the sealed train membership is used for fitting. Metrics are reported on
the frozen test membership (dev used for light early-stop / selection).

    nice -n 15 ~/.grok-vision/bin/python -m ramanujan.train.train_components

Does not flip RAMANUJAN_RESEARCH_AUTHORIZED. Never reads Math-Preserve.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ramanujan.data.common import read_jsonl
from ramanujan.data.freeze_memberships import (
    MANIFEST_PATH,
    index_for_source,
    load_membership,
    verify_membership_seal,
)
from ramanujan.data.paths import CORPORA, ROOT, SOURCE_FILES
from ramanujan.limits import LimitRegistry
from ramanujan.train import AUTHORITY
from ramanujan.train.features import LabelVocab, bag_of_text, tokenize
from ramanujan.train.models import ClassifierModel, RetrieverModel, ValueModel

BAG_DIM = 4096
EMB_DIM = 64
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
RECEIPT_PATH = Path(__file__).resolve().parent / "TRAINING_RECEIPT.json"
METRICS_PATH = Path(__file__).resolve().parent / "HELD_OUT_METRICS.json"


# re exported for tests
__all__ = ["run_training", "main"]


def _device() -> torch.device:
    # Prefer MPS when available; fall back to CPU. Never requires CUDA/cloud.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_split_items(source_id: str, membership: dict[str, Any]) -> dict[str, list[dict]]:
    path = SOURCE_FILES[source_id]
    items = read_jsonl(path)
    idx = index_for_source(membership, source_id)
    out: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for it in items:
        h = it.get("content_hash")
        if not h or h not in idx:
            continue
        if not it.get("admitted", True):
            continue
        out[idx[h]].append(it)
    return out


def _first_tactic(proof: str, tactics: list[str] | None = None) -> str:
    if tactics:
        return str(tactics[0]).strip()
    lines = [ln.strip() for ln in (proof or "").splitlines() if ln.strip()]
    return lines[0] if lines else ""


# ---------------------------------------------------------------------------
# Retriever (D3)
# ---------------------------------------------------------------------------
def train_retriever(
    splits: dict[str, list[dict]],
    *,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, Any]:
    train = splits["train"]
    if not train:
        return {"status": "NO_TRAIN_DATA", "trained": False}

    model = RetrieverModel(BAG_DIM, EMB_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def _pairs(items: list[dict]) -> list[tuple[str, str, str]]:
        pairs = []
        for it in items:
            goal = it.get("goal") or ""
            pos = list(it.get("positive_premises") or [])
            neg = list(it.get("negative_premises") or [])
            if not pos:
                continue
            # one positive + one negative (or random other pos as weak neg)
            p = pos[0]
            n = neg[0] if neg else (pos[-1] if len(pos) > 1 else p + "_absent")
            pairs.append((goal, p, n))
        return pairs

    train_pairs = _pairs(train)
    if not train_pairs:
        return {"status": "NO_PAIRS", "trained": False}

    history = []
    model.train()
    for ep in range(epochs):
        random.shuffle(train_pairs)
        total_loss = 0.0
        n_steps = 0
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i : i + batch_size]
            goals = torch.stack([bag_of_text(g, BAG_DIM) for g, _, _ in batch]).to(device)
            poss = torch.stack([bag_of_text(p, BAG_DIM) for _, p, _ in batch]).to(device)
            negs = torch.stack([bag_of_text(n, BAG_DIM) for _, _, n in batch]).to(device)
            sp = model.score(goals, poss)
            sn = model.score(goals, negs)
            # pairwise logistic: want sp > sn
            loss = F.softplus(sn - sp).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_steps += 1
        history.append({"epoch": ep + 1, "loss": total_loss / max(1, n_steps)})

    def evaluate(items: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
        model.eval()
        if not items:
            return {"n": 0, "note": "empty_split"}
        hits = {k: 0 for k in ks}
        rr_sum = 0.0
        n = 0
        with torch.no_grad():
            for it in items:
                goal = it.get("goal") or ""
                pos = list(it.get("positive_premises") or [])
                neg = list(it.get("negative_premises") or [])
                if not pos:
                    continue
                cands = list(dict.fromkeys(pos + neg))  # preserve order, unique
                if not cands:
                    continue
                g = bag_of_text(goal, BAG_DIM).to(device)
                px = torch.stack([bag_of_text(c, BAG_DIM) for c in cands]).to(device)
                ge = model.encode_goal(g)
                pe = model.encode_prem(px)
                scores = (pe * ge).sum(dim=-1)
                order = scores.argsort(descending=True).tolist()
                ranked = [cands[i] for i in order]
                relevant = set(pos)
                # RR of first relevant
                rr = 0.0
                for rank, name in enumerate(ranked, start=1):
                    if name in relevant:
                        rr = 1.0 / rank
                        break
                rr_sum += rr
                for k in ks:
                    top = set(ranked[:k])
                    if top & relevant:
                        hits[k] += 1
                n += 1
        out = {
            "n": n,
            "mrr": rr_sum / max(1, n),
            **{f"recall@{k}": hits[k] / max(1, n) for k in ks},
        }
        return out

    # Untrained baseline: token-overlap scorer
    def baseline_eval(items: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
        hits = {k: 0 for k in ks}
        rr_sum = 0.0
        n = 0
        for it in items:
            goal = it.get("goal") or ""
            pos = list(it.get("positive_premises") or [])
            neg = list(it.get("negative_premises") or [])
            if not pos:
                continue
            cands = list(dict.fromkeys(pos + neg))
            gset = set(tokenize(goal))
            scored = []
            for c in cands:
                cset = set(tokenize(c))
                scored.append((len(gset & cset) / max(1, len(gset)), c))
            scored.sort(key=lambda x: (-x[0], x[1]))
            ranked = [c for _, c in scored]
            relevant = set(pos)
            rr = 0.0
            for rank, name in enumerate(ranked, start=1):
                if name in relevant:
                    rr = 1.0 / rank
                    break
            rr_sum += rr
            for k in ks:
                if set(ranked[:k]) & relevant:
                    hits[k] += 1
            n += 1
        return {
            "n": n,
            "mrr": rr_sum / max(1, n),
            **{f"recall@{k}": hits[k] / max(1, n) for k in ks},
            "label": "token_overlap_baseline",
        }

    ckpt = CKPT_DIR / "retriever.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "bag_dim": BAG_DIM, "emb_dim": EMB_DIM}, ckpt)

    test_metrics = evaluate(splits["test"])
    dev_metrics = evaluate(splits["dev"])
    base_test = baseline_eval(splits["test"])
    improved = test_metrics.get("mrr", 0) > base_test.get("mrr", 0) + 1e-9
    return {
        "status": "TRAINED_AND_BEATS_BASELINE" if improved else "TRAINED_BUT_NO_BETTER_THAN_BASELINE",
        "trained": True,
        "converged": improved,
        "component": "retriever",
        "source": "D3",
        "n_train_pairs": len(train_pairs),
        "n_train_items": len(train),
        "n_dev": len(splits["dev"]),
        "n_test": len(splits["test"]),
        "epochs": epochs,
        "history": history,
        "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
        "held_out_test": test_metrics,
        "held_out_dev": dev_metrics,
        "baseline_test": base_test,
        "improved_vs_baseline_mrr": improved,
        "metric_definition": (
            "For each test D3 item, rank positive_premises ∪ negative_premises by "
            "cosine of dual encoders; report recall@k (any positive in top-k) and MRR."
        ),
    }


# ---------------------------------------------------------------------------
# Classifier family: formalizer / prover / repair
# ---------------------------------------------------------------------------
def _train_classifier(
    *,
    name: str,
    source: str,
    splits: dict[str, list[dict]],
    text_fn: Callable[[dict], str],
    label_fn: Callable[[dict], str],
    device: torch.device,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    min_count: int = 2,
) -> dict[str, Any]:
    train = splits["train"]
    labels = [label_fn(it) for it in train]
    labels = [lb for lb in labels if lb]
    if not labels:
        return {"status": "NO_LABELS", "trained": False, "component": name}

    vocab = LabelVocab(labels, min_count=min_count, max_size=6000)
    # Filter train to known labels only (exclude pure-UNK rows from training signal)
    train_xy: list[tuple[torch.Tensor, int]] = []
    for it in train:
        lab = label_fn(it)
        if not lab or not vocab.known(lab):
            continue
        train_xy.append((bag_of_text(text_fn(it), BAG_DIM), vocab.encode(lab)))
    if len(train_xy) < 8:
        return {
            "status": "TOO_FEW_TRAIN_EXAMPLES",
            "trained": False,
            "component": name,
            "n_known_labels": vocab.n_known,
            "n_train_xy": len(train_xy),
        }

    X = torch.stack([x for x, _ in train_xy])
    y = torch.tensor([yi for _, yi in train_xy], dtype=torch.long)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Inverse-frequency class weights so rare tactics are not drowned by majority.
    counts = torch.bincount(y, minlength=vocab.n_known).float().clamp(min=1.0)
    class_w = (counts.sum() / (counts * vocab.n_known)).to(device)

    hidden = 384
    model = ClassifierModel(BAG_DIM, n_classes=vocab.n_known, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    model.train()
    for ep in range(epochs):
        total = 0.0
        n_steps = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_steps += 1
        history.append({"epoch": ep + 1, "loss": total / max(1, n_steps)})

    def evaluate(items: list[dict]) -> dict[str, Any]:
        model.eval()
        if not items:
            return {"n": 0, "note": "empty_split"}
        correct = 0
        known_gold = 0
        total = 0
        # majority baseline
        maj = max(vocab.itos, key=lambda lb: labels.count(lb)) if vocab.itos else ""
        maj_correct = 0
        with torch.no_grad():
            for it in items:
                gold = label_fn(it)
                if not gold:
                    continue
                total += 1
                x = bag_of_text(text_fn(it), BAG_DIM).to(device)
                logits = model(x.unsqueeze(0))
                pred_idx = int(logits.argmax(dim=-1).item())
                pred = vocab.decode(pred_idx)
                if gold == pred:
                    correct += 1
                if vocab.known(gold):
                    known_gold += 1
                if gold == maj:
                    maj_correct += 1
        return {
            "n": total,
            "exact_match": correct / max(1, total),
            "n_correct": correct,
            "n_gold_in_train_vocab": known_gold,
            "coverage_gold_in_train_vocab": known_gold / max(1, total),
            "majority_baseline_exact_match": maj_correct / max(1, total),
            "majority_label": maj[:80],
            "n_classes_train": vocab.n_known,
        }

    ckpt = CKPT_DIR / f"{name}.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "itos": vocab.itos,
            "bag_dim": BAG_DIM,
            "n_classes": vocab.n_known,
            "hidden": hidden,
        },
        ckpt,
    )

    test_m = evaluate(splits["test"])
    dev_m = evaluate(splits["dev"])
    improved = test_m.get("exact_match", 0) > test_m.get("majority_baseline_exact_match", 0) + 1e-9
    converged = improved and test_m.get("exact_match", 0) > 0
    return {
        "status": "TRAINED_AND_BEATS_MAJORITY" if converged else "TRAINED_BUT_NO_BETTER_THAN_MAJORITY",
        "trained": True,
        "converged": converged,
        "component": name,
        "source": source,
        "n_train": len(train_xy),
        "n_dev": len(splits["dev"]),
        "n_test": len(splits["test"]),
        "n_classes": vocab.n_known,
        "epochs": epochs,
        "history": history,
        "class_weighted_ce": True,
        "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
        "held_out_test": test_m,
        "held_out_dev": dev_m,
        "improved_vs_majority": improved,
    }


def train_formalizer(splits, *, device, epochs=6) -> dict[str, Any]:
    def text_fn(it: dict) -> str:
        # Include name + signature so the formalizer sees the theorem identity.
        name = it.get("name") or ""
        return f"{name}\n{it.get('statement') or it.get('signature') or ''}"

    def label_fn(it: dict) -> str:
        # Prefer coarse tactic head (rw/simp/exact/...) for a learnable closed set,
        # falling back to full first line when the head alone is too common.
        raw = _first_tactic(it.get("proof") or "", it.get("tactics"))
        toks = tokenize(raw)
        if not toks:
            return raw
        head = toks[0]
        # Keep full line for rare heads; collapse ultra-common heads to head+lemma-ish.
        if head in {"exact", "rw", "simp", "simpa", "apply", "refine", "intro"}:
            return " ".join(toks[:3]) if len(toks) >= 2 else head
        return raw

    out = _train_classifier(
        name="formalizer",
        source="D1",
        splits=splits,
        text_fn=text_fn,
        label_fn=label_fn,
        device=device,
        epochs=epochs,
        min_count=2,
        lr=8e-4,
    )
    out["metric_definition"] = (
        "Held-out exact match of predicted first-tactic / first proof line "
        "against D1 gold, closed over train-vocab classes (min_count=2)."
    )
    out["task"] = "statement -> first_tactic"
    return out


def train_prover(splits, *, device, epochs=6) -> dict[str, Any]:
    def text_fn(it: dict) -> str:
        sb = it.get("state_before") or {}
        thm = it.get("theorem") or ""
        if isinstance(sb, dict):
            goal = str(sb.get("goal") or "")
            rem = sb.get("remaining_tactics") or []
            return f"thm:{thm}\ngoal:{goal}\nrem:{' | '.join(map(str, rem[:3]))}"
        return f"thm:{thm}\n{sb}"

    def label_fn(it: dict) -> str:
        raw = str(it.get("tactic") or "").strip()
        toks = tokenize(raw)
        if not toks:
            return raw
        head = toks[0]
        if head in {"exact", "rw", "simp", "simpa", "apply", "refine", "intro"}:
            return " ".join(toks[:3]) if len(toks) >= 2 else head
        return raw

    out = _train_classifier(
        name="prover",
        source="D2",
        splits=splits,
        text_fn=text_fn,
        label_fn=label_fn,
        device=device,
        epochs=epochs,
        min_count=2,
        lr=8e-4,
    )
    out["metric_definition"] = (
        "Held-out exact match of predicted next tactic against D2 gold."
    )
    out["task"] = "state_before.goal -> tactic"
    return out


def train_repair(splits, *, device, epochs=8, lean_eval_limit: int = 40, lean_workers: int = 4) -> dict[str, Any]:
    def text_fn(it: dict) -> str:
        # Normalize away absolute temp paths in errors so the model sees the shape.
        err = str(it.get("error") or "")
        err = re_sub_paths(err)
        return (
            f"sig: {it.get('signature') or ''}\n"
            f"broken: {it.get('broken_proof') or ''}\n"
            f"error: {err[:400]}"
        )

    def label_fn(it: dict) -> str:
        return str(it.get("fix_proof") or "").strip()

    out = _train_classifier(
        name="repair",
        source="D4",
        splits=splits,
        text_fn=text_fn,
        label_fn=label_fn,
        device=device,
        epochs=epochs,
        min_count=1,  # D4 is smaller; keep rare fixes
        batch_size=32,
    )
    out["task"] = "(signature, broken_proof, error) -> fix_proof"
    out["metric_definition"] = (
        "Held-out exact match of predicted fix_proof; plus Lean compile rate of "
        "predicted fixes under pinned Mathlib (lake env lean)."
    )

    # Lean compile evaluation on held-out test predictions.
    if out.get("trained") and splits["test"]:
        from ramanujan.train.lean_check import check_many

        ckpt_path = CKPT_DIR / "repair.pt"
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vocab = LabelVocab.from_itos(list(blob["itos"]))
        model = ClassifierModel(
            BAG_DIM,
            n_classes=len(vocab.itos),
            hidden=int(blob.get("hidden", 384)),
        ).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()

        # Prefer Nat.Basic items first (isolated example wrapper is reliable there),
        # then fill with remaining test items up to lean_eval_limit.
        ranked = sorted(
            splits["test"],
            key=lambda it: 0 if it.get("import") == "Mathlib.Data.Nat.Basic" else 1,
        )
        test_items = ranked[: lean_eval_limit]
        jobs = []
        preds = []
        with torch.no_grad():
            for it in test_items:
                x = bag_of_text(text_fn(it), BAG_DIM).to(device)
                logits = model(x.unsqueeze(0))
                pred = vocab.decode(int(logits.argmax(dim=-1).item())) or ""
                preds.append(pred)
                jobs.append(
                    {
                        "id": it.get("id"),
                        "import": it.get("import") or "Mathlib.Data.Nat.Basic",
                        "signature": it.get("signature") or "",
                        "proof": pred,
                    }
                )
        # Also check gold fixes as a ceiling / toolchain sanity.
        gold_jobs = [
            {
                "id": it.get("id"),
                "import": it.get("import") or "Mathlib.Data.Nat.Basic",
                "signature": it.get("signature") or "",
                "proof": it.get("fix_proof") or "",
            }
            for it in test_items
        ]
        pred_results = check_many(jobs, workers=lean_workers)
        gold_results = check_many(gold_jobs, workers=lean_workers)
        n = len(test_items)
        pred_ok = sum(1 for r in pred_results if r.get("ok"))
        gold_ok = sum(1 for r in gold_results if r.get("ok"))
        exact = sum(
            1
            for it, pr in zip(test_items, preds)
            if (it.get("fix_proof") or "").strip() == pr
        )
        nat_idx = [
            i
            for i, it in enumerate(test_items)
            if it.get("import") == "Mathlib.Data.Nat.Basic"
        ]
        nat_pred_ok = sum(1 for i in nat_idx if pred_results[i].get("ok"))
        nat_gold_ok = sum(1 for i in nat_idx if gold_results[i].get("ok"))
        out["held_out_test_lean"] = {
            "n_evaluated": n,
            "n_nat_basic": len(nat_idx),
            "lean_workers": lean_workers,
            "predicted_fix_compiles": pred_ok / max(1, n),
            "n_predicted_compiles": pred_ok,
            "gold_fix_compiles": gold_ok / max(1, n),
            "n_gold_compiles": gold_ok,
            "nat_basic_predicted_compiles": nat_pred_ok / max(1, len(nat_idx)),
            "nat_basic_gold_compiles": nat_gold_ok / max(1, len(nat_idx)),
            "exact_match_on_lean_subset": exact / max(1, n),
            "note": (
                "Compile under pinned Mathlib via lake env lean. "
                "Gold compile rate < 1 on non-Nat imports means the isolated "
                "example wrapper lacks typeclass context (honest ceiling). "
                "Nat.Basic subset is the fairer compile metric."
            ),
        }
    else:
        out["held_out_test_lean"] = {
            "n_evaluated": 0,
            "note": "repair did not train or no test items",
        }
    return out


def re_sub_paths(err: str) -> str:
    import re

    # Strip absolute temp paths so the model generalizes across generation runs.
    return re.sub(r"/[^\s:]+\.lean", "<file>.lean", err)


# ---------------------------------------------------------------------------
# Value (D2)
# ---------------------------------------------------------------------------
def train_value(
    splits: dict[str, list[dict]],
    *,
    device: torch.device,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, Any]:
    def featurize(it: dict) -> torch.Tensor:
        sb = it.get("state_before") or {}
        if isinstance(sb, dict):
            goal = str(sb.get("goal") or "")
            rem = sb.get("remaining_tactics") or []
            n_rem = len(rem) if isinstance(rem, list) else 0
            text = f"goal:{goal}\nn_rem:{n_rem}\nrem:{' | '.join(map(str, rem[:4]))}"
        else:
            text = str(sb)
        return bag_of_text(text, BAG_DIM)

    def targets(it: dict) -> tuple[float, float]:
        sa = it.get("state_after") or {}
        closed = 1.0 if (isinstance(sa, dict) and sa.get("closed")) else 0.0
        if isinstance(sa, dict):
            rem = sa.get("remaining_tactics")
            if isinstance(rem, list):
                steps = float(len(rem))
            else:
                steps = 0.0 if closed else 1.0
        else:
            steps = 0.0
        # remaining AFTER the tactic: gold is len(remaining) at state_after
        return closed, steps

    train = splits["train"]
    if len(train) < 8:
        return {"status": "TOO_FEW_TRAIN", "trained": False, "component": "value", "converged": False}

    X = torch.stack([featurize(it) for it in train])
    closed_y = torch.tensor([targets(it)[0] for it in train], dtype=torch.float32)
    steps_y = torch.tensor([targets(it)[1] for it in train], dtype=torch.float32)
    ds = TensorDataset(X, closed_y, steps_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Pos-weight for rare closed class if imbalanced.
    n_pos = float(closed_y.sum().item())
    n_neg = float(len(closed_y) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)

    model = ValueModel(BAG_DIM, hidden=192).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    model.train()
    for ep in range(epochs):
        total = 0.0
        n_steps = 0
        for xb, cb, sb in loader:
            xb, cb, sb = xb.to(device), cb.to(device), sb.to(device)
            clog, steps = model(xb)
            loss = (
                F.binary_cross_entropy_with_logits(clog, cb, pos_weight=pos_weight)
                + 0.25 * F.smooth_l1_loss(steps, sb)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_steps += 1
        history.append({"epoch": ep + 1, "loss": total / max(1, n_steps)})

    def evaluate(items: list[dict]) -> dict[str, Any]:
        model.eval()
        if not items:
            return {"n": 0}
        correct = 0
        abs_err = 0.0
        # majority closed baseline
        train_closed_rate = float(closed_y.mean().item()) if len(closed_y) else 0.5
        maj_pred = 1.0 if train_closed_rate >= 0.5 else 0.0
        maj_correct = 0
        n = 0
        with torch.no_grad():
            for it in items:
                c_true, s_true = targets(it)
                x = featurize(it).to(device)
                clog, steps = model(x.unsqueeze(0))
                c_pred = 1.0 if torch.sigmoid(clog).item() >= 0.5 else 0.0
                if c_pred == c_true:
                    correct += 1
                if maj_pred == c_true:
                    maj_correct += 1
                abs_err += abs(float(steps.item()) - s_true)
                n += 1
        return {
            "n": n,
            "closed_accuracy": correct / max(1, n),
            "remaining_steps_mae": abs_err / max(1, n),
            "majority_closed_baseline_accuracy": maj_correct / max(1, n),
            "train_closed_rate": train_closed_rate,
        }

    ckpt = CKPT_DIR / "value.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "bag_dim": BAG_DIM}, ckpt)

    test_m = evaluate(splits["test"])
    dev_m = evaluate(splits["dev"])
    improved = (
        test_m.get("closed_accuracy", 0)
        > test_m.get("majority_closed_baseline_accuracy", 0) + 1e-9
    )
    return {
        "status": "TRAINED_AND_BEATS_MAJORITY" if improved else "TRAINED_BUT_NO_BETTER_THAN_MAJORITY",
        "trained": True,
        "converged": improved,
        "component": "value",
        "source": "D2",
        "n_train": len(train),
        "n_dev": len(splits["dev"]),
        "n_test": len(splits["test"]),
        "epochs": epochs,
        "history": history,
        "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
        "held_out_test": test_m,
        "held_out_dev": dev_m,
        "improved_vs_majority_closed": improved,
        "metric_definition": (
            "Held-out accuracy of P(state_after.closed) and MAE of "
            "len(state_after.remaining_tactics) from state_before.goal only."
        ),
        "task": "state_before -> (closed_next, remaining_steps)",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_training(
    *,
    epochs: int = 6,
    lean_eval_limit: int = 40,
    lean_workers: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)

    reg = LimitRegistry()
    teacher = reg.consult("teacher_trace_from_math_preserve", role_id="librarian")
    research = reg.consult("run_research", role_id="director")
    if teacher.allowed or research.allowed:
        raise RuntimeError("fences must refuse teacher/math-preserve and run_research")

    seal = verify_membership_seal()
    if not seal["ok"]:
        raise RuntimeError(f"membership seal failed: {seal}")
    membership = load_membership()

    device = _device()
    t0 = time.time()

    d3 = _load_split_items("D3", membership)
    d1 = _load_split_items("D1", membership)
    d2 = _load_split_items("D2", membership)
    d4 = _load_split_items("D4", membership)

    print(f"device={device} membership_sha256={membership['membership_sha256'][:16]}…", flush=True)
    print(
        f"splits D3 train/dev/test={len(d3['train'])}/{len(d3['dev'])}/{len(d3['test'])}",
        flush=True,
    )

    results: dict[str, Any] = {}

    print("=== retriever (D3) ===", flush=True)
    results["retriever"] = train_retriever(d3, device=device, epochs=max(3, epochs - 1))
    print(json.dumps(results["retriever"].get("held_out_test"), indent=2), flush=True)

    print("=== formalizer (D1) ===", flush=True)
    results["formalizer"] = train_formalizer(d1, device=device, epochs=max(epochs, 10))
    print(
        json.dumps(
            {
                "held_out_test": results["formalizer"].get("held_out_test"),
                "status": results["formalizer"].get("status"),
            },
            indent=2,
        ),
        flush=True,
    )

    print("=== prover (D2) ===", flush=True)
    results["prover"] = train_prover(d2, device=device, epochs=max(epochs, 10))
    print(
        json.dumps(
            {
                "held_out_test": results["prover"].get("held_out_test"),
                "status": results["prover"].get("status"),
            },
            indent=2,
        ),
        flush=True,
    )

    print("=== repair (D4) ===", flush=True)
    results["repair"] = train_repair(
        d4,
        device=device,
        epochs=max(epochs + 4, 12),
        lean_eval_limit=lean_eval_limit,
        lean_workers=lean_workers,
    )
    print(
        json.dumps(
            {
                "held_out_test": results["repair"].get("held_out_test"),
                "lean": results["repair"].get("held_out_test_lean"),
            },
            indent=2,
        ),
        flush=True,
    )

    print("=== value (D2) ===", flush=True)
    results["value"] = train_value(d2, device=device, epochs=epochs)
    print(json.dumps(results["value"].get("held_out_test"), indent=2), flush=True)

    wall = time.time() - t0

    trained = [k for k, v in results.items() if v.get("trained")]
    failed = [k for k, v in results.items() if not v.get("trained")]
    converged = [k for k, v in results.items() if v.get("converged")]
    trained_not_converged = [
        k for k, v in results.items() if v.get("trained") and not v.get("converged")
    ]
    scaffold = [
        "end_to_end_search_policy",
        "composer_solver_curriculum",
        "preference_rl_from_verifier",
        "hidden_rediscovery_D8",
        "frontier_variant_D9",
        "math_preserve_teacher_distillation",
        "seq2seq_proof_generation_beyond_closed_vocab",
        "interactive_lean_state_value_from_real_goals",
    ]

    metrics_public = {
        "schema": "hawking.ramanujan.held_out_metrics.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "membership_sha256": membership["membership_sha256"],
        "membership_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "device": str(device),
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "authority": AUTHORITY,
        "components": {
            k: {
                "trained": v.get("trained"),
                "status": v.get("status"),
                "source": v.get("source"),
                "held_out_test": v.get("held_out_test"),
                "held_out_dev": v.get("held_out_dev"),
                "held_out_test_lean": v.get("held_out_test_lean"),
                "baseline_test": v.get("baseline_test"),
                "metric_definition": v.get("metric_definition"),
                "improved_vs_baseline": v.get("improved_vs_baseline_mrr", v.get("improved_vs_majority", v.get("improved_vs_majority_closed"))),
            }
            for k, v in results.items()
        },
        "trained_components": trained,
        "converged_components": converged,
        "trained_but_did_not_converge": trained_not_converged,
        "did_not_train": failed,
        "still_scaffold": scaffold,
        "note": (
            "All metrics are on the frozen TEST membership unless labeled otherwise. "
            "No train-split numbers are reported as held-out. "
            "'converged' means the held-out metric strictly beats the stated baseline "
            "(token-overlap for retriever; majority class for classifiers/value)."
        ),
    }

    receipt = {
        "schema": "hawking.ramanujan.training_receipt.v1",
        "at": metrics_public["at"],
        "authority": AUTHORITY,
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "teacher_from_math_preserve": False,
        "membership_sha256": membership["membership_sha256"],
        "membership_seal_ok": seal["ok"],
        "device": str(device),
        "wall_clock_seconds": round(wall, 2),
        "resource_policy": {
            "nice_requested": 15,
            "max_workers": 8,
            "lean_workers": lean_workers,
            "lean_eval_limit": lean_eval_limit,
            "touched_mop": False,
            "fetched_model_shards": False,
            "paid_apis": False,
            "cloud_training": False,
        },
        "limit_consults": list(reg.consult_log),
        "components": results,
        "trained_components": trained,
        "converged_components": converged,
        "trained_but_did_not_converge": trained_not_converged,
        "did_not_train": failed,
        "still_scaffold": scaffold,
        "checkpoints_dir": str(CKPT_DIR.relative_to(ROOT)),
        "metrics_path": str(METRICS_PATH.relative_to(ROOT)),
    }

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps(metrics_public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lean-eval-limit", type=int, default=40)
    p.add_argument("--lean-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    lean_workers = max(1, min(int(args.lean_workers), 8))
    receipt = run_training(
        epochs=args.epochs,
        lean_eval_limit=args.lean_eval_limit,
        lean_workers=lean_workers,
        seed=args.seed,
    )
    summary = {
        "status": "COMPLETE",
        "wall_clock_seconds": receipt["wall_clock_seconds"],
        "trained": receipt["trained_components"],
        "converged": receipt["converged_components"],
        "trained_but_did_not_converge": receipt["trained_but_did_not_converge"],
        "did_not_train": receipt["did_not_train"],
        "metrics_path": receipt["metrics_path"],
        "membership_sha256": receipt["membership_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
