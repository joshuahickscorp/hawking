"""Phase F.2 — medusa K-head trainer + per-head top-{1,4,10} eval.

Implements all the speed + quality levers identified in the F.2 audit:

  Speed
    A. Tied LM head & output_norm to V2-Lite frozen weights (no per-head
       Linear→vocab; per-head adapter is hidden→hidden only).
    B. Batched K dispatch: stack K adapter outputs into (K,B,hidden) and
       run one big GEMM through the tied lm_head.
    C. In-RAM cache of hidden_high + targets after first epoch.
    D. Column-projected dataloader (read only what each head consumes).
    E. No V2-Lite forward — just the frozen npz.
    F. Single optimizer step over all K heads (one loss scalar).

  Quality
    A.1 MSE auxiliary on hidden_high_+i  (denser signal than discrete CE).
    A.2 Per-head loss weighting  w_k = 1 + k/K  (compensate easy-head
        gradient dominance).
    B.3 KL distillation from frozen lm_head at +i positions  (soft-label
        signal; biggest single quality lever per the audit).
    B.4 Per-head feed-forward adapter — RMSNorm → SwiGLU → residual gate
        — richer than Linear→SiLU→Linear per the plan skeleton. (Real
        attention block would need windowed dataloader; left as a future
        upgrade if MLP plateau is hit.)

  Eval
    Reports per-head top-{1,4,10} accuracy on a held-out shard so F.5
    can make K_inference choices on real branch-quality data, not just
    top-1.

  CLI
    train:  python medusa_head.py train --parquet '...medusa/*.parquet' \\
                --frozen v2lite_frozen.npz --ckpt-dir checkpoints/medusa_v1
    eval:   python medusa_head.py eval  --ckpt   checkpoints/medusa_v1/best.npz \\
                --frozen v2lite_frozen.npz --parquet '...medusa/shard_00060.parquet'

The training script never instantiates V2-Lite, so it can run with
other GPU work (Claude, bench loops) active. The audit's "no contention
window required" property is by construction.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as mxoptim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import pyarrow.parquet as pq

HIDDEN_DIM = 2048
VOCAB = 102_400
RMS_EPS = 1e-6
SENTINEL_TOKEN = -1


# ---------------------------------------------------------------------------
# Adapter primitive (per-head)
# ---------------------------------------------------------------------------
class _SwiGLU(nn.Module):
    def __init__(self, h: int, i: int):
        super().__init__()
        self.gate = nn.Linear(h, i, bias=False)
        self.up = nn.Linear(h, i, bias=False)
        self.down = nn.Linear(i, h, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class _MedusaAdapter(nn.Module):
    """RMSNorm → SwiGLU → residual.  Refines hidden_high into a +i-aware
    prediction in the same hidden space; tied lm_head then projects to vocab.

    `gate` starts at 0.05 so the adapter is near-identity at init — the
    untrained head predicts roughly what V2-Lite itself would predict at
    position p, and training learns the +i refinement. Same warm-start
    trick eagle4 uses.
    """

    def __init__(self, hidden: int = HIDDEN_DIM, intermediate: int = HIDDEN_DIM * 2):
        super().__init__()
        self.norm = mx.ones((hidden,))
        self.mlp = _SwiGLU(hidden, intermediate)
        self.gate = mx.array([0.05])

    def __call__(self, h):
        h_norm = mx.fast.rms_norm(h, self.norm, RMS_EPS)
        return h + self.gate * self.mlp(h_norm)


# ---------------------------------------------------------------------------
# MedusaHead — K parallel adapters + tied frozen lm_head
# ---------------------------------------------------------------------------
class MedusaHead(nn.Module):
    def __init__(self, lm_head: mx.array, output_norm: mx.array, K: int = 8,
                 intermediate: int = HIDDEN_DIM * 2):
        super().__init__()
        self._lm_head = lm_head
        self._output_norm = output_norm
        self.adapters = [_MedusaAdapter(HIDDEN_DIM, intermediate) for _ in range(K)]
        self.K = K

    def trainable_parameters(self):
        p = self.parameters()
        for k in ("_lm_head", "_output_norm"):
            p.pop(k, None)
        return p

    def refine(self, h_high):
        """(B, hidden) → (K, B, hidden)  refined per-head."""
        return mx.stack([a(h_high) for a in self.adapters], axis=0)

    def project(self, refined):
        """(K, B, hidden) → (K, B, vocab)  through tied output_norm + lm_head."""
        normed = mx.fast.rms_norm(refined, self._output_norm, RMS_EPS)
        return normed @ self._lm_head

    def __call__(self, h_high):
        refined = self.refine(h_high)
        logits = self.project(refined)
        return refined, logits


def build_head(frozen_npz: Path, K: int = 8) -> MedusaHead:
    z = np.load(frozen_npz, allow_pickle=False)
    lm_head = mx.array(z["lm_head"])  # (hidden, vocab) fp16
    output_norm = mx.array(z["output_norm"])  # (hidden,) fp32
    return MedusaHead(lm_head, output_norm, K=K)


# ---------------------------------------------------------------------------
# Dataloader — column-projected, in-RAM cached
# ---------------------------------------------------------------------------
@dataclass
class CachedShards:
    """All training tensors held in RAM after first read.

    For K=8 and 500k rows: hidden_high = 4 GB, hidden_high_+i × 7 = 28 GB
    target_tokens = 32 MB.  Total ~32 GB — too large for 18 GB RAM.
    So in practice we cache hidden_high + targets only and re-read
    hidden_high_+i columns per epoch (cheap thanks to column projection).
    """
    hidden_high: np.ndarray  # (N, hidden) fp16
    targets: np.ndarray  # (N, K) int32, with -1 sentinel for tail rows
    shard_paths: list[Path] = field(default_factory=list)
    K: int = 8


def _shard_columns_targets(K: int) -> list[str]:
    return ["next_token"] + [f"next_token_p{i}" for i in range(1, K)]


def _shard_columns_teacher(K: int) -> list[str]:
    return ["hidden_high"] + [f"hidden_high_p{i}" for i in range(1, K)]


def load_cached_targets(shards: list[Path], K: int) -> CachedShards:
    """First-pass load: hidden_high (current position) + per-head target tokens.
    The +i teacher hiddens are loaded per-epoch (column-projected) so RAM
    holds only one epoch's worth of teacher data peak."""
    t0 = time.time()
    cols_t = _shard_columns_targets(K)
    hh_chunks = []
    tgt_chunks = []
    rows = 0
    for sp in shards:
        t = pq.read_table(sp, columns=["hidden_high"] + cols_t)
        n = t.num_rows
        # Convert hidden_high bytes column to (n, HIDDEN_DIM) fp16 ndarray
        hh = np.frombuffer(b"".join(t["hidden_high"].to_pylist()), dtype=np.float16).reshape(n, HIDDEN_DIM)
        hh_chunks.append(hh)
        tgt = np.stack([t[c].to_numpy(zero_copy_only=False) for c in cols_t], axis=1).astype(np.int32)
        tgt_chunks.append(tgt)
        rows += n
    hidden_high = np.concatenate(hh_chunks, axis=0)
    targets = np.concatenate(tgt_chunks, axis=0)
    elapsed = time.time() - t0
    print(
        f"[cache] {rows} rows, hidden_high={hidden_high.nbytes/1e9:.2f}GB "
        f"targets={targets.nbytes/1e6:.1f}MB, load {elapsed:.1f}s",
        flush=True,
    )
    return CachedShards(hidden_high=hidden_high, targets=targets,
                        shard_paths=list(shards), K=K)


def _load_teacher_hiddens_shard(sp: Path, K: int) -> np.ndarray:
    """(n, K, hidden) fp16 — hidden_high (i=0) and hidden_high_p1..p{K-1}.
    Column-projected; only reads what's needed."""
    cols = _shard_columns_teacher(K)
    t = pq.read_table(sp, columns=cols)
    n = t.num_rows
    stacks = []
    for c in cols:
        hh = np.frombuffer(b"".join(t[c].to_pylist()), dtype=np.float16).reshape(n, HIDDEN_DIM)
        stacks.append(hh)
    return np.stack(stacks, axis=1)  # (n, K, hidden)


# ---------------------------------------------------------------------------
# Loss — CE + KL + MSE, per-head weighted, masked by tail sentinel
# ---------------------------------------------------------------------------
def medusa_loss(head: MedusaHead, h_high_batch: mx.array, targets_batch: mx.array,
                teacher_hiddens_batch: mx.array,
                head_weights: mx.array, alpha_kl: float, beta_mse: float,
                ) -> tuple[mx.array, dict]:
    """Compute per-head CE + KL + MSE losses, masked, weighted.

    Args:
      h_high_batch: (B, hidden) fp16
      targets_batch: (B, K) int32 with -1 sentinel for invalid
      teacher_hiddens_batch: (B, K, hidden) fp16  — true hidden_high_+i
      head_weights: (K,) fp32  — multiplier per head's contributions
      alpha_kl: scalar  — KL term strength
      beta_mse: scalar  — MSE aux strength
    Returns (loss, stats) where stats has CE/KL/MSE component logs.
    """
    K = head.K
    # Forward: refined (K, B, hidden), student_logits (K, B, vocab)
    refined, student_logits_fp16 = head(h_high_batch)

    # Use fp32 for loss math to avoid fp16 overflow on big GEMMs
    student_logits = student_logits_fp16.astype(mx.float32)

    # Per-head valid mask: targets != -1   shape (K, B) after transpose
    targets_kb = targets_batch.T  # (K, B)
    valid_kb = (targets_kb != SENTINEL_TOKEN).astype(mx.float32)
    targets_clamped = mx.maximum(targets_kb, 0)  # avoid OOB at sentinel positions

    # ---- Cross-entropy hard-label ----------------------------------------
    # student_logits: (K, B, V); targets_clamped: (K, B)
    log_probs = student_logits - mx.logsumexp(student_logits, axis=-1, keepdims=True)
    # gather target logprobs
    target_lp = mx.take_along_axis(log_probs, targets_clamped[..., None], axis=-1).squeeze(-1)
    # ce per row, masked
    ce_per = -target_lp * valid_kb  # (K, B)
    valid_per_head = mx.maximum(valid_kb.sum(axis=1), 1.0)  # (K,)
    ce_per_head = ce_per.sum(axis=1) / valid_per_head  # (K,)

    # ---- KL distillation from frozen teacher_hiddens -----------------------
    # teacher_hiddens_batch: (B, K, hidden)  → transpose to (K, B, hidden)
    teacher_h_kb = teacher_hiddens_batch.transpose(1, 0, 2)
    teacher_normed = mx.fast.rms_norm(teacher_h_kb, head._output_norm, RMS_EPS)
    teacher_logits = (teacher_normed @ head._lm_head).astype(mx.float32)
    teacher_log_probs = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    teacher_probs = mx.exp(teacher_log_probs)
    # KL(teacher || student) = sum_v teacher_p * (teacher_logp - student_logp)
    kl_per_pos = (teacher_probs * (teacher_log_probs - log_probs)).sum(axis=-1)  # (K, B)
    kl_per = kl_per_pos * valid_kb
    kl_per_head = kl_per.sum(axis=1) / valid_per_head  # (K,)

    # ---- MSE aux on refined vs teacher_hidden ------------------------------
    # refined: (K, B, hidden); teacher_h_kb: (K, B, hidden)
    refined_f32 = refined.astype(mx.float32)
    teacher_f32 = teacher_h_kb.astype(mx.float32)
    mse_per_pos = ((refined_f32 - teacher_f32) ** 2).mean(axis=-1)  # (K, B)
    mse_per = mse_per_pos * valid_kb
    mse_per_head = mse_per.sum(axis=1) / valid_per_head  # (K,)

    # ---- Combine ----------------------------------------------------------
    per_head_total = ce_per_head + alpha_kl * kl_per_head + beta_mse * mse_per_head
    total = (head_weights * per_head_total).sum()

    stats = {
        "ce": ce_per_head,
        "kl": kl_per_head,
        "mse": mse_per_head,
        "valid_per_head": valid_per_head,
    }
    return total, stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    parquet_glob: str
    frozen: Path
    ckpt_dir: Path
    K: int = 8
    batch_size: int = 128
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.0
    alpha_kl: float = 0.5
    beta_mse: float = 0.1
    head_weight_slope: float = 1.0
    seed: int = 0
    log_every: int = 50
    max_rows: int | None = None
    heldout_shards: int = 2


def _expand_parquet(g: str) -> list[Path]:
    paths = sorted(Path(p) for p in _glob.glob(g))
    if not paths:
        raise SystemExit(f"no parquet matched {g}")
    return paths


def train(cfg: TrainConfig) -> None:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    mx.random.seed(cfg.seed)

    all_shards = _expand_parquet(cfg.parquet_glob)
    train_shards = all_shards[: -cfg.heldout_shards] if cfg.heldout_shards else all_shards
    heldout_shards = all_shards[-cfg.heldout_shards:] if cfg.heldout_shards else []
    print(
        f"[train] {len(train_shards)} train shards, {len(heldout_shards)} heldout",
        flush=True,
    )

    head = build_head(cfg.frozen, K=cfg.K)
    mx.eval(head.parameters())

    # head weights: w_k = 1 + slope * k / (K-1)
    if cfg.K > 1:
        head_weights_np = np.array(
            [1.0 + cfg.head_weight_slope * k / (cfg.K - 1) for k in range(cfg.K)],
            dtype=np.float32,
        )
    else:
        head_weights_np = np.ones(1, dtype=np.float32)
    head_weights = mx.array(head_weights_np)
    print(f"[train] head_weights = {head_weights_np.tolist()}", flush=True)

    optim = mxoptim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)

    # First-pass cache: hidden_high + targets
    cache = load_cached_targets(train_shards, cfg.K)
    if cfg.max_rows is not None and cfg.max_rows < cache.hidden_high.shape[0]:
        cache.hidden_high = cache.hidden_high[: cfg.max_rows]
        cache.targets = cache.targets[: cfg.max_rows]
        print(f"[train] truncated cache to max_rows={cfg.max_rows}", flush=True)
    N = cache.hidden_high.shape[0]
    print(f"[train] N={N} rows  K={cfg.K}  B={cfg.batch_size}", flush=True)

    # Teacher hiddens: load shard-at-a-time per epoch (column-projected)
    # We'll keep a list of (shard_path, row_offset, row_count) so per-step
    # we can fetch the teacher block for a contiguous slice from one shard.
    shard_offsets: list[tuple[int, int]] = []
    off = 0
    for sp in train_shards:
        n = pq.read_metadata(sp).num_rows
        shard_offsets.append((off, n))
        off += n
    if off != N and cfg.max_rows is None:
        print(f"[warn] shard rows {off} != cache rows {N}", flush=True)

    def _loss_only(h, t, teach):
        loss, _ = medusa_loss(
            head, h, t, teach, head_weights, cfg.alpha_kl, cfg.beta_mse,
        )
        return loss

    grad_fn = nn.value_and_grad(head, _loss_only)

    best_eval_top1 = -1.0
    step = 0
    t_train0 = time.time()

    for epoch in range(cfg.epochs):
        # Re-load teacher hiddens shard-by-shard (column projection makes this cheap)
        for s_idx, sp in enumerate(train_shards):
            shard_t0 = time.time()
            teacher = _load_teacher_hiddens_shard(sp, cfg.K)  # (n, K, hidden)
            off, n = shard_offsets[s_idx]
            if cfg.max_rows is not None:
                if off >= cfg.max_rows:
                    break
                n = min(n, cfg.max_rows - off)
                teacher = teacher[:n]
            local_h = cache.hidden_high[off : off + n]
            local_t = cache.targets[off : off + n]

            # Shuffle within shard
            perm = np.random.permutation(n)
            for bi in range(0, n - cfg.batch_size + 1, cfg.batch_size):
                bidx = perm[bi : bi + cfg.batch_size]
                h_batch = mx.array(local_h[bidx])
                t_batch = mx.array(local_t[bidx])
                teach_batch = mx.array(teacher[bidx])
                loss, grads = grad_fn(h_batch, t_batch, teach_batch)
                optim.update(head, grads)
                mx.eval(head.parameters(), optim.state, loss)
                step += 1
                if step % cfg.log_every == 0:
                    el = time.time() - t_train0
                    rate = step * cfg.batch_size / max(el, 1e-3)
                    print(
                        f"[train] e{epoch} s{step} shard={s_idx} loss={float(loss):.4f} "
                        f"rows/s={rate:.0f} elapsed={el:.0f}s",
                        flush=True,
                    )
            del teacher
            print(
                f"[train] e{epoch} shard {s_idx+1}/{len(train_shards)} done "
                f"({time.time()-shard_t0:.1f}s)",
                flush=True,
            )

        # Per-epoch heldout eval
        if heldout_shards:
            eval_stats = evaluate(head, heldout_shards, cfg.K, cfg.batch_size,
                                  max_rows_per_shard=4000)
            top1_mean = float(np.mean(eval_stats["top1"]))
            print(
                f"[eval] e{epoch} top1_mean={top1_mean:.3f} "
                f"per_head_top1={['%.3f' % t for t in eval_stats['top1']]}",
                flush=True,
            )
            ckpt_path = cfg.ckpt_dir / f"epoch_{epoch:02d}.npz"
            save_ckpt(head, ckpt_path)
            if top1_mean > best_eval_top1:
                best_eval_top1 = top1_mean
                save_ckpt(head, cfg.ckpt_dir / "best.npz")
                with open(cfg.ckpt_dir / "best_eval.json", "w") as f:
                    json.dump({"epoch": epoch, "top1_mean": top1_mean,
                               "per_head": eval_stats}, f, indent=2,
                              default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    print(f"[train] done in {time.time()-t_train0:.1f}s, best top1_mean={best_eval_top1:.3f}",
          flush=True)


def save_ckpt(head: MedusaHead, path: Path) -> None:
    out = {}
    for name, p in tree_flatten(head.trainable_parameters()):
        out[name] = np.array(p)
    np.savez(path, **out)


def load_ckpt(head: MedusaHead, path: Path) -> None:
    z = np.load(path, allow_pickle=False)
    items = [(k, mx.array(z[k])) for k in z.files]
    new_params = tree_unflatten(items)
    head.update(new_params)


# ---------------------------------------------------------------------------
# Eval — per-head top-{1,4,10}
# ---------------------------------------------------------------------------
def evaluate(head: MedusaHead, shards: list[Path], K: int, batch_size: int = 256,
             max_rows_per_shard: int | None = None) -> dict:
    """Return per-head top-{1,4,10} accuracies on the given shards."""
    correct = {tk: np.zeros(K, dtype=np.int64) for tk in (1, 4, 10)}
    valid = np.zeros(K, dtype=np.int64)
    for sp in shards:
        cols = _shard_columns_teacher(K)[:1] + _shard_columns_targets(K)
        t = pq.read_table(sp, columns=cols)
        n = t.num_rows
        if max_rows_per_shard is not None:
            n = min(n, max_rows_per_shard)
        hh = np.frombuffer(b"".join(t["hidden_high"].to_pylist()[:n]),
                           dtype=np.float16).reshape(n, HIDDEN_DIM)
        tgt_cols = _shard_columns_targets(K)
        tgt = np.stack([t[c].to_numpy(zero_copy_only=False)[:n] for c in tgt_cols], axis=1).astype(np.int32)

        for bi in range(0, n, batch_size):
            sl = slice(bi, min(bi + batch_size, n))
            h_batch = mx.array(hh[sl])
            _, logits = head(h_batch)
            logits_np = np.array(logits.astype(mx.float32))  # (K, b, V)
            tgt_batch = tgt[sl]  # (b, K)
            for k in range(K):
                lg = logits_np[k]  # (b, V)
                t_k = tgt_batch[:, k]
                mask = t_k != SENTINEL_TOKEN
                if not mask.any():
                    continue
                lg_m = lg[mask]
                t_m = t_k[mask]
                # top-1
                top1 = lg_m.argmax(axis=-1)
                correct[1][k] += int((top1 == t_m).sum())
                # top-4 and top-10 via argpartition for speed
                for tk in (4, 10):
                    part = np.argpartition(-lg_m, kth=tk - 1, axis=-1)[:, :tk]
                    hit = (part == t_m[:, None]).any(axis=-1)
                    correct[tk][k] += int(hit.sum())
                valid[k] += int(mask.sum())

    out = {f"top{tk}": (correct[tk] / np.maximum(valid, 1)).tolist() for tk in (1, 4, 10)}
    out["valid_per_head"] = valid.tolist()
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_train(args) -> int:
    cfg = TrainConfig(
        parquet_glob=args.parquet,
        frozen=args.frozen,
        ckpt_dir=args.ckpt_dir,
        K=args.K,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha_kl=args.alpha_kl,
        beta_mse=args.beta_mse,
        head_weight_slope=args.head_weight_slope,
        seed=args.seed,
        log_every=args.log_every,
        max_rows=args.max_rows,
        heldout_shards=args.heldout_shards,
    )
    train(cfg)
    return 0


def _cmd_eval(args) -> int:
    head = build_head(args.frozen, K=args.K)
    load_ckpt(head, args.ckpt)
    shards = _expand_parquet(args.parquet)
    print(f"[eval] {len(shards)} shards", flush=True)
    stats = evaluate(head, shards, args.K, args.batch_size,
                     max_rows_per_shard=args.max_rows_per_shard)
    print(json.dumps(stats, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="medusa_head")
    sub = p.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train")
    tp.add_argument("--parquet", required=True, help="glob pattern for medusa shards")
    tp.add_argument("--frozen", type=Path, required=True)
    tp.add_argument("--ckpt-dir", type=Path, required=True)
    tp.add_argument("--K", type=int, default=8)
    tp.add_argument("--batch-size", type=int, default=128)
    tp.add_argument("--epochs", type=int, default=10)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--weight-decay", type=float, default=0.0)
    tp.add_argument("--alpha-kl", type=float, default=0.5)
    tp.add_argument("--beta-mse", type=float, default=0.1)
    tp.add_argument("--head-weight-slope", type=float, default=1.0,
                    help="0 = uniform, 1 = late heads 2x weight, larger = more aggressive")
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--log-every", type=int, default=50)
    tp.add_argument("--max-rows", type=int, default=None,
                    help="cap total training rows (for smoke tests)")
    tp.add_argument("--heldout-shards", type=int, default=2)
    tp.set_defaults(func=_cmd_train)

    ep = sub.add_parser("eval")
    ep.add_argument("--ckpt", type=Path, required=True)
    ep.add_argument("--frozen", type=Path, required=True)
    ep.add_argument("--parquet", required=True)
    ep.add_argument("--K", type=int, default=8)
    ep.add_argument("--batch-size", type=int, default=256)
    ep.add_argument("--max-rows-per-shard", type=int, default=None)
    ep.set_defaults(func=_cmd_eval)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
