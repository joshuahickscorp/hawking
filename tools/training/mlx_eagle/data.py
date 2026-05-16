"""
data.py — parquet -> MLX batch iterator for EAGLE-3 head training.

Reads `training_data/c2_hidden/eagle3_v0/shard_*.parquet` (DCAP records
converted by `tools/training/capture_hidden.py to-parquet`) and yields
batches of shape:

  prev_tokens        : int32[B, S]
  target_hidden      : float32[B, S, H]    (cast from packed f16 bytes)
  target_next_tokens : int32[B, S]
  loss_mask          : float32[B, S]       (1.0 for positions to score,
                                            0.0 for skipped / padded)

The (B, S) shape is a vector-batching convenience: EAGLE-3 sees each
(prev_token, target_hidden, next_token) tuple INDEPENDENTLY, so each row
of each batch is a self-contained training signal. We just stack S rows
per sample for throughput. S=16 means 16x fewer Python-loop iterations
per gradient step than S=1.

Position-weighted loss support
------------------------------
The 5K capture analysis (5k_capture_results.md §"Hidden L2 by position
bucket") found that BOS-adjacent positions (0..9) have ~17% smaller
hidden L2 norm than deeper positions — the model genuinely has less
context to encode in those positions and the head has correspondingly
less to predict from. To prevent these low-information positions from
dominating early training, the loader supports `skip_bos_positions=N`
which zeros the loss mask for positions 0..N-1. Recommended N=3.

Memory profile
--------------
Each batch at B=16, S=16, H=2048 (float32 hidden) is:
  prev_tokens   : 16 * 16 * 4   = 1 KB
  target_hidden : 16 * 16 * 2048 * 4 = 2 MB
  next_tokens   : 16 * 16 * 4   = 1 KB
  loss_mask     : 16 * 16 * 4   = 1 KB
  ~total per batch: ~2 MB

PyArrow streams the parquet in row-groups (~64K rows each), so peak
loader memory is bounded regardless of total shard size.

Usage
-----
    from tools.training.mlx_eagle.data import ParquetBatchIterator

    it = ParquetBatchIterator(
        parquet_paths=["training_data/c2_hidden/eagle3_v0/shard_000.parquet"],
        batch_size=16,
        seq_len=16,
        hidden_dim=2048,
        skip_bos_positions=3,
        shuffle=True,
        seed=20260516,
    )
    for step, batch in enumerate(it.iter_epoch()):
        # batch is a dict of mx.array.
        pass
"""

from __future__ import annotations

import dataclasses
import pathlib
import random
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np


# Lazy MLX import — module is importable for review without MLX installed.
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# In-memory record schema (per-position, NOT per-sample)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class _Record:
    """One captured (prev, hidden, next) triple at a specific corpus position.

    Stored as raw numpy arrays so collation into batches is just np.stack.
    """

    sample_id: str
    pos: int
    prev_token: int
    next_token: int
    hidden_f16: bytes  # exactly 2 * H bytes; decoded lazily at batch time


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------
class ParquetBatchIterator:
    """Yields MLX-ready batches from one or more DCAP parquet shards.

    Records are loaded into RAM up-front (the 5K shard is ~1.7 GB
    parquet -> ~2 GB pyarrow Table; the 55K extension will be ~17 GB
    parquet -> ~20 GB pyarrow Table which still fits in 18 GB unified
    memory once the dismantle engine is NOT loaded simultaneously).

    For larger shards, switch to row-group streaming via
    `iter_batches_streaming()` (TODO — not implemented; the in-memory
    approach is fine through 100K samples).
    """

    def __init__(
        self,
        parquet_paths: Sequence[str | pathlib.Path],
        batch_size: int = 16,
        seq_len: int = 16,
        hidden_dim: int = 2048,
        skip_bos_positions: int = 3,
        shuffle: bool = True,
        seed: int = 20260516,
        drop_last: bool = True,
    ) -> None:
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pyarrow not installed — pip install pyarrow"
            ) from e

        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.parquet_paths = [pathlib.Path(p) for p in parquet_paths]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.skip_bos_positions = skip_bos_positions
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        print(
            f"[data] loading {len(self.parquet_paths)} parquet shard(s)…",
            file=sys.stderr,
        )
        self.records: List[_Record] = self._load_all_records()
        print(
            f"[data] loaded {len(self.records):,} records from "
            f"{len(self.parquet_paths)} shard(s)",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_all_records(self) -> List[_Record]:
        import pyarrow.parquet as pq

        all_records: List[_Record] = []
        expected_hidden_bytes = self.hidden_dim * 2
        for path in self.parquet_paths:
            if not path.exists():
                raise FileNotFoundError(path)
            pf = pq.ParquetFile(str(path))
            md = pf.schema_arrow.metadata or {}
            shard_hd = int(md.get(b"hidden_dim", str(self.hidden_dim).encode()))
            if shard_hd != self.hidden_dim:
                raise ValueError(
                    f"{path}: shard hidden_dim={shard_hd} != configured {self.hidden_dim}"
                )
            for batch in pf.iter_batches(
                batch_size=8192,
                columns=["sample_id", "pos", "prev_token", "next_token", "hidden_f16"],
            ):
                sids = batch.column("sample_id").to_pylist()
                poses = batch.column("pos").to_pylist()
                prevs = batch.column("prev_token").to_pylist()
                nexts = batch.column("next_token").to_pylist()
                hbs = batch.column("hidden_f16").to_pylist()
                for sid, p, prev, nx, hb in zip(sids, poses, prevs, nexts, hbs):
                    if len(hb) != expected_hidden_bytes:
                        raise ValueError(
                            f"{path} sid={sid} pos={p}: hidden bytes "
                            f"{len(hb)} != expected {expected_hidden_bytes}"
                        )
                    all_records.append(
                        _Record(
                            sample_id=sid,
                            pos=int(p),
                            prev_token=int(prev),
                            next_token=int(nx),
                            hidden_f16=hb,
                        )
                    )
        return all_records

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of full batches per epoch."""
        records_per_batch = self.batch_size * self.seq_len
        n_batches = len(self.records) // records_per_batch
        if not self.drop_last and len(self.records) % records_per_batch:
            n_batches += 1
        return n_batches

    def iter_epoch(self, epoch: int = 0) -> Iterator[Dict[str, "mx.array"]]:
        """Yield one epoch's worth of batches.

        Shuffles per-epoch with a deterministic seed derived from
        `self.seed + epoch`.
        """
        if mx is None:
            raise ImportError("MLX not installed — pip install mlx")

        rng = random.Random(self.seed + epoch)
        idx = list(range(len(self.records)))
        if self.shuffle:
            rng.shuffle(idx)

        records_per_batch = self.batch_size * self.seq_len
        i = 0
        while i + records_per_batch <= len(idx):
            slice_idx = idx[i : i + records_per_batch]
            i += records_per_batch
            yield self._collate(slice_idx)

        if not self.drop_last and i < len(idx):
            # Pad the final short batch with zeros, mask out the padding.
            remaining = idx[i:]
            yield self._collate(remaining, pad_to=records_per_batch)

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------
    def _collate(
        self, slice_idx: Sequence[int], pad_to: Optional[int] = None
    ) -> Dict[str, "mx.array"]:
        B = self.batch_size
        S = self.seq_len
        H = self.hidden_dim

        n_real = len(slice_idx)
        total = pad_to if pad_to is not None else n_real
        assert total == B * S, f"collate slice {total} != B*S {B*S}"

        prev_arr = np.zeros((total,), dtype=np.int32)
        next_arr = np.zeros((total,), dtype=np.int32)
        pos_arr = np.zeros((total,), dtype=np.int32)
        hidden_arr = np.zeros((total, H), dtype=np.float32)
        mask = np.zeros((total,), dtype=np.float32)

        for k, ri in enumerate(slice_idx):
            r = self.records[ri]
            prev_arr[k] = r.prev_token
            next_arr[k] = r.next_token
            pos_arr[k] = r.pos
            hidden_arr[k] = np.frombuffer(r.hidden_f16, dtype=np.float16).astype(
                np.float32
            )
            # Loss mask: include this position unless it's in the BOS-warmup band.
            if r.pos >= self.skip_bos_positions:
                mask[k] = 1.0

        # Reshape (B*S, ...) -> (B, S, ...)
        return {
            "prev_tokens": mx.array(prev_arr.reshape(B, S)),
            "target_hidden": mx.array(hidden_arr.reshape(B, S, H)),
            "target_next_tokens": mx.array(next_arr.reshape(B, S)),
            "loss_mask": mx.array(mask.reshape(B, S)),
            "positions": mx.array(pos_arr.reshape(B, S)),  # for diagnostics
        }

    # ------------------------------------------------------------------
    # Numpy-only debug iterator (works without MLX for smoke testing)
    # ------------------------------------------------------------------
    def iter_epoch_numpy(
        self, epoch: int = 0
    ) -> Iterator[Dict[str, np.ndarray]]:
        """MLX-free iterator returning numpy arrays. Use for smoke tests."""
        rng = random.Random(self.seed + epoch)
        idx = list(range(len(self.records)))
        if self.shuffle:
            rng.shuffle(idx)
        records_per_batch = self.batch_size * self.seq_len
        i = 0
        while i + records_per_batch <= len(idx):
            slice_idx = idx[i : i + records_per_batch]
            i += records_per_batch
            yield self._collate_numpy(slice_idx)

    def _collate_numpy(self, slice_idx: Sequence[int]) -> Dict[str, np.ndarray]:
        """Numpy-only version of _collate. Same shapes, no MLX import."""
        B = self.batch_size
        S = self.seq_len
        H = self.hidden_dim
        total = len(slice_idx)
        assert total == B * S

        prev_arr = np.zeros((total,), dtype=np.int32)
        next_arr = np.zeros((total,), dtype=np.int32)
        pos_arr = np.zeros((total,), dtype=np.int32)
        hidden_arr = np.zeros((total, H), dtype=np.float32)
        mask = np.zeros((total,), dtype=np.float32)
        for k, ri in enumerate(slice_idx):
            r = self.records[ri]
            prev_arr[k] = r.prev_token
            next_arr[k] = r.next_token
            pos_arr[k] = r.pos
            hidden_arr[k] = np.frombuffer(r.hidden_f16, dtype=np.float16).astype(
                np.float32
            )
            if r.pos >= self.skip_bos_positions:
                mask[k] = 1.0
        return {
            "prev_tokens": prev_arr.reshape(B, S),
            "target_hidden": hidden_arr.reshape(B, S, H),
            "target_next_tokens": next_arr.reshape(B, S),
            "loss_mask": mask.reshape(B, S),
            "positions": pos_arr.reshape(B, S),
        }


# ---------------------------------------------------------------------------
# Streaming variant — for shards larger than RAM
# ---------------------------------------------------------------------------
class StreamingParquetBatchIterator:
    """Memory-friendly variant of ParquetBatchIterator for shards >> RAM.

    Streams parquet via `pyarrow.parquet.ParquetFile.iter_batches()` in a
    background thread, accumulates records into a sliding shuffle buffer,
    and emits batches to the main thread via a bounded Queue.

    Key differences vs the in-memory iterator:
      - **Memory bounded** to roughly `prefetch * batch_size * seq_len *
        (hidden_dim * 2 + 24)` bytes — a few MB regardless of shard size.
        20 GB parquet runs in <100 MB of loader RAM.
      - **Shuffling is approximate** — records are shuffled within a
        rolling buffer (`shuffle_buffer`, default 64 K records) rather
        than globally. Good enough for SGD; matches torch.IterableDataset
        + shuffle-buffer pattern.
      - **One pass per epoch** — re-instantiate or call .reset() to restart.
      - **Prefetch depth** is `prefetch` batches; default 2 keeps the
        compute pipeline fed without blowing RAM.

    Same batch dict schema as ParquetBatchIterator (prev_tokens,
    target_hidden, target_next_tokens, loss_mask, positions).
    """

    def __init__(
        self,
        parquet_paths: Sequence[str | pathlib.Path],
        batch_size: int = 16,
        seq_len: int = 16,
        hidden_dim: int = 2048,
        skip_bos_positions: int = 3,
        shuffle_buffer: int = 65536,
        prefetch: int = 2,
        seed: int = 20260516,
        row_group_batch: int = 8192,
    ) -> None:
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as e:
            raise ImportError("pyarrow not installed — pip install pyarrow") from e
        if batch_size <= 0 or seq_len <= 0 or hidden_dim <= 0:
            raise ValueError("batch_size / seq_len / hidden_dim must be positive")
        if prefetch < 1:
            raise ValueError("prefetch must be >= 1")
        self.parquet_paths = [pathlib.Path(p) for p in parquet_paths]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.skip_bos_positions = skip_bos_positions
        self.shuffle_buffer = shuffle_buffer
        self.prefetch = prefetch
        self.seed = seed
        self.row_group_batch = row_group_batch
        # Estimate total batches by summing num_rows across files.
        import pyarrow.parquet as pq
        total_rows = 0
        for p in self.parquet_paths:
            md = pq.ParquetFile(str(p)).metadata
            total_rows += md.num_rows
        self._total_rows = total_rows
        print(
            f"[data-stream] {len(self.parquet_paths)} shard(s), "
            f"{total_rows:,} total rows, buffer={shuffle_buffer:,}, "
            f"prefetch={prefetch}",
            file=sys.stderr,
        )

    def __len__(self) -> int:
        return self._total_rows // (self.batch_size * self.seq_len)

    def iter_epoch(self, epoch: int = 0) -> Iterator[Dict[str, "mx.array"]]:
        """Spawn a background producer; yield MLX-cast batches to caller."""
        if mx is None:
            raise ImportError("MLX not installed — pip install mlx")
        for npbatch in self._iter_numpy_batches(epoch):
            yield {
                "prev_tokens": mx.array(npbatch["prev_tokens"]),
                "target_hidden": mx.array(npbatch["target_hidden"]),
                "target_next_tokens": mx.array(npbatch["target_next_tokens"]),
                "loss_mask": mx.array(npbatch["loss_mask"]),
                "positions": mx.array(npbatch["positions"]),
            }

    def iter_epoch_numpy(self, epoch: int = 0) -> Iterator[Dict[str, np.ndarray]]:
        """MLX-free streaming iterator for smoke tests / non-training use."""
        yield from self._iter_numpy_batches(epoch)

    def _iter_numpy_batches(
        self, epoch: int
    ) -> Iterator[Dict[str, np.ndarray]]:
        import queue
        import threading
        import pyarrow.parquet as pq

        rng = random.Random(self.seed + epoch)
        records_per_batch = self.batch_size * self.seq_len

        q: "queue.Queue[Optional[Dict[str, np.ndarray]]]" = queue.Queue(
            maxsize=self.prefetch
        )
        stop_flag = threading.Event()

        def producer():
            try:
                buffer: List[_Record] = []
                expected_hb = self.hidden_dim * 2
                for path in self.parquet_paths:
                    pf = pq.ParquetFile(str(path))
                    for batch in pf.iter_batches(
                        batch_size=self.row_group_batch,
                        columns=["sample_id", "pos", "prev_token", "next_token", "hidden_f16"],
                    ):
                        if stop_flag.is_set():
                            return
                        sids = batch.column("sample_id").to_pylist()
                        poses = batch.column("pos").to_pylist()
                        prevs = batch.column("prev_token").to_pylist()
                        nexts = batch.column("next_token").to_pylist()
                        hbs = batch.column("hidden_f16").to_pylist()
                        for sid, p, prev, nx, hb in zip(sids, poses, prevs, nexts, hbs):
                            buffer.append(
                                _Record(
                                    sample_id=sid, pos=int(p),
                                    prev_token=int(prev), next_token=int(nx),
                                    hidden_f16=hb,
                                )
                            )
                            # Once buffer is full, drain into batches.
                            while len(buffer) >= self.shuffle_buffer:
                                # Take a random batch from the buffer and refill from the tail.
                                idxs = rng.sample(range(len(buffer)), records_per_batch)
                                idxs_set = set(idxs)
                                batch_recs = [buffer[i] for i in idxs]
                                buffer = [r for i, r in enumerate(buffer) if i not in idxs_set]
                                npbatch = self._collate_records(batch_recs)
                                q.put(npbatch)
                                if stop_flag.is_set():
                                    return
                # Flush remaining buffer.
                rng.shuffle(buffer)
                while len(buffer) >= records_per_batch:
                    batch_recs = buffer[:records_per_batch]
                    buffer = buffer[records_per_batch:]
                    npbatch = self._collate_records(batch_recs)
                    q.put(npbatch)
                    if stop_flag.is_set():
                        return
                # Sentinel.
                q.put(None)
            except Exception as e:
                # Propagate via the queue so consumer raises.
                q.put({"__error__": str(e)})  # type: ignore[arg-type]

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                if isinstance(item, dict) and "__error__" in item:
                    raise RuntimeError(f"producer error: {item['__error__']}")
                yield item
        finally:
            stop_flag.set()
            t.join(timeout=5)

    def _collate_records(
        self, recs: Sequence[_Record]
    ) -> Dict[str, np.ndarray]:
        B = self.batch_size
        S = self.seq_len
        H = self.hidden_dim
        total = len(recs)
        prev_arr = np.empty(total, dtype=np.int32)
        next_arr = np.empty(total, dtype=np.int32)
        pos_arr = np.empty(total, dtype=np.int32)
        hidden_arr = np.empty((total, H), dtype=np.float32)
        mask = np.zeros(total, dtype=np.float32)
        for k, r in enumerate(recs):
            prev_arr[k] = r.prev_token
            next_arr[k] = r.next_token
            pos_arr[k] = r.pos
            hidden_arr[k] = np.frombuffer(r.hidden_f16, dtype=np.float16).astype(np.float32)
            if r.pos >= self.skip_bos_positions:
                mask[k] = 1.0
        return {
            "prev_tokens": prev_arr.reshape(B, S),
            "target_hidden": hidden_arr.reshape(B, S, H),
            "target_next_tokens": next_arr.reshape(B, S),
            "loss_mask": mask.reshape(B, S),
            "positions": pos_arr.reshape(B, S),
        }


# ---------------------------------------------------------------------------
# CLI: smoke-test the loader against a real shard
# ---------------------------------------------------------------------------
def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--parquet",
        default="training_data/c2_hidden/eagle3_v0/shard_000.parquet",
        nargs="+",
        help="One or more parquet shard paths.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--skip-bos-positions", type=int, default=3)
    p.add_argument("--n-batches", type=int, default=3, help="How many batches to dump")
    p.add_argument("--numpy-only", action="store_true",
                   help="Use numpy iterator (skip MLX import)")
    args = p.parse_args()

    paths = args.parquet if isinstance(args.parquet, list) else [args.parquet]
    it = ParquetBatchIterator(
        parquet_paths=paths,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        skip_bos_positions=args.skip_bos_positions,
        shuffle=True,
    )
    print(f"[smoke] full epoch: {len(it):,} batches "
          f"(records/batch = {args.batch_size * args.seq_len})")

    iter_fn = it.iter_epoch_numpy if args.numpy_only else it.iter_epoch
    for step, batch in enumerate(iter_fn()):
        if step >= args.n_batches:
            break
        # Convert MLX -> numpy for display if needed
        def _np(x):
            return x if isinstance(x, np.ndarray) else np.array(x)
        prev = _np(batch["prev_tokens"])
        hidden = _np(batch["target_hidden"])
        nxt = _np(batch["target_next_tokens"])
        mask = _np(batch["loss_mask"])
        pos = _np(batch["positions"])
        print(f"\n[batch {step}]")
        print(f"  prev_tokens     : shape={prev.shape}    sample[0,:5]={prev[0,:5].tolist()}")
        print(f"  target_hidden   : shape={hidden.shape}  L2[0,0]={np.linalg.norm(hidden[0,0]):.2f}  finite={np.isfinite(hidden).all()}")
        print(f"  next_tokens     : shape={nxt.shape}     sample[0,:5]={nxt[0,:5].tolist()}")
        print(f"  loss_mask       : shape={mask.shape}    fraction_masked={1 - mask.mean():.3f}  (BOS skip = pos<{args.skip_bos_positions})")
        print(f"  positions       : shape={pos.shape}     min={pos.min()} max={pos.max()}")

    return 0


if __name__ == "__main__":
    sys.exit(_main())
