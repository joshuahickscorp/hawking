#!/usr/bin/env python3.12
"""FIXTURE: pure-numpy toy MLP for exercising Odyssey trainer apparatus.

This is not a real model, not a student of Math-Preserve, and learns nothing of
scientific value. A few thousand float32 parameters exist so that checkpoints,
resume, and receipts have a real state blob to content-address.

Label: FIXTURE. Never cite results from this model as training evidence.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA = "hawking.odyssey.toy_model.fixture.v1"
FIXTURE_LABEL = "FIXTURE: pure-numpy toy MLP; not a real student; never trained anything real"


@dataclass
class ToyConfig:
    """Tiny dimensions on purpose — apparatus proof, not learning."""

    vocab_size: int = 32
    d_model: int = 16
    d_hidden: int = 24
    n_classes: int = 8
    seed: int = 0

    def n_params(self) -> int:
        # emb + W1 + b1 + W2 + b2
        return (
            self.vocab_size * self.d_model
            + self.d_model * self.d_hidden
            + self.d_hidden
            + self.d_hidden * self.n_classes
            + self.n_classes
        )


class ToyMLP:
    """Two-layer bag-of-ids classifier with SGD. Fully deterministic given seed."""

    def __init__(self, cfg: ToyConfig | None = None):
        self.cfg = cfg or ToyConfig()
        rng = np.random.RandomState(self.cfg.seed)
        # Xavier-ish scale so steps are stable without torch.
        s = 1.0 / np.sqrt(self.cfg.d_model)
        self.emb = (rng.randn(self.cfg.vocab_size, self.cfg.d_model) * s).astype(np.float32)
        self.W1 = (rng.randn(self.cfg.d_model, self.cfg.d_hidden) * s).astype(np.float32)
        self.b1 = np.zeros(self.cfg.d_hidden, dtype=np.float32)
        self.W2 = (rng.randn(self.cfg.d_hidden, self.cfg.n_classes) * s).astype(np.float32)
        self.b2 = np.zeros(self.cfg.n_classes, dtype=np.float32)
        # Optimizer momentum buffers (SGD+momentum) — part of resume state.
        self.v_emb = np.zeros_like(self.emb)
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        self.step = 0
        self.rng_state = rng.get_state()

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "emb": self.emb,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

    def optimizer_state(self) -> dict[str, np.ndarray]:
        return {
            "v_emb": self.v_emb,
            "v_W1": self.v_W1,
            "v_b1": self.v_b1,
            "v_W2": self.v_W2,
            "v_b2": self.v_b2,
        }

    def n_params(self) -> int:
        return int(sum(p.size for p in self.parameters().values()))

    def _set_rng(self) -> np.random.RandomState:
        rng = np.random.RandomState(0)
        rng.set_state(self.rng_state)
        return rng

    def _batch(self, batch_size: int = 4, seq_len: int = 8) -> tuple[np.ndarray, np.ndarray]:
        rng = self._set_rng()
        x = rng.randint(0, self.cfg.vocab_size, size=(batch_size, seq_len))
        y = rng.randint(0, self.cfg.n_classes, size=(batch_size,))
        self.rng_state = rng.get_state()
        return x.astype(np.int64), y.astype(np.int64)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        # Mean-pool embeddings → ReLU MLP → logits.
        h = self.emb[token_ids].mean(axis=1)  # (B, d)
        z = h @ self.W1 + self.b1
        a = np.maximum(z, 0.0)
        logits = a @ self.W2 + self.b2
        return logits

    def train_step(self, lr: float = 0.05, momentum: float = 0.9) -> dict[str, float]:
        """One deterministic SGD+momentum step on a synthetic batch."""
        x, y = self._batch()
        B = x.shape[0]
        # Forward
        emb_tok = self.emb[x]  # (B, T, d)
        h = emb_tok.mean(axis=1)
        z = h @ self.W1 + self.b1
        a = np.maximum(z, 0.0)
        logits = a @ self.W2 + self.b2
        # Softmax CE
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        probs = ex / ex.sum(axis=1, keepdims=True)
        loss = float(-np.mean(np.log(probs[np.arange(B), y] + 1e-12)))
        # Backward
        dlogits = probs
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B
        dW2 = a.T @ dlogits
        db2 = dlogits.sum(axis=0)
        da = dlogits @ self.W2.T
        dz = da * (z > 0)
        dW1 = h.T @ dz
        db1 = dz.sum(axis=0)
        dh = dz @ self.W1.T
        # Mean-pool backward into embeddings.
        T = x.shape[1]
        demb = np.zeros_like(self.emb)
        for b in range(B):
            for t in range(T):
                demb[x[b, t]] += dh[b] / T

        def _sgd(param: np.ndarray, grad: np.ndarray, vel: np.ndarray) -> None:
            vel *= momentum
            vel += grad
            param -= lr * vel

        _sgd(self.emb, demb, self.v_emb)
        _sgd(self.W1, dW1, self.v_W1)
        _sgd(self.b1, db1, self.v_b1)
        _sgd(self.W2, dW2, self.v_W2)
        _sgd(self.b2, db2, self.v_b2)
        self.step += 1
        return {"loss": loss, "step": float(self.step)}

    def state_dict(self) -> dict[str, Any]:
        """Canonical state for content-addressed checkpoints.

        wall_clock is deliberately excluded: identity is the training state.
        """
        def arr(a: np.ndarray) -> list:
            return a.astype(np.float32).reshape(-1).tolist()

        # RNG state is a tuple; serialize portably.
        # numpy RandomState: ('MT19937', key_array, pos, has_gauss, cached_gaussian)
        rs = self.rng_state
        rng_ser = {
            "kind": str(rs[0]),
            "key": [int(x) for x in rs[1].tolist()],
            "pos": int(rs[2]),
            "has_gauss": int(rs[3]),
            "cached_gaussian": float(rs[4]),
        }
        return {
            "schema": SCHEMA,
            "fixture_label": FIXTURE_LABEL,
            "cfg": {
                "vocab_size": self.cfg.vocab_size,
                "d_model": self.cfg.d_model,
                "d_hidden": self.cfg.d_hidden,
                "n_classes": self.cfg.n_classes,
                "seed": self.cfg.seed,
            },
            "step": int(self.step),
            "params": {k: arr(v) for k, v in self.parameters().items()},
            "optimizer": {k: arr(v) for k, v in self.optimizer_state().items()},
            "rng": rng_ser,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ToyMLP":
        cfg_d = state["cfg"]
        cfg = ToyConfig(**cfg_d)
        m = cls(cfg)
        shapes = {
            "emb": (cfg.vocab_size, cfg.d_model),
            "W1": (cfg.d_model, cfg.d_hidden),
            "b1": (cfg.d_hidden,),
            "W2": (cfg.d_hidden, cfg.n_classes),
            "b2": (cfg.n_classes,),
        }
        for k, shape in shapes.items():
            setattr(m, k, np.asarray(state["params"][k], dtype=np.float32).reshape(shape))
        for k, shape in shapes.items():
            setattr(m, f"v_{k}", np.asarray(state["optimizer"][f"v_{k}"], dtype=np.float32).reshape(shape))
        m.step = int(state["step"])
        rs = state["rng"]
        key = np.asarray(rs["key"], dtype=np.uint32)
        m.rng_state = (rs["kind"], key, int(rs["pos"]), int(rs["has_gauss"]), float(rs["cached_gaussian"]))
        return m

    def state_sha256(self) -> str:
        """Bit-identity of the full training state (params + opt + rng + step)."""
        import json

        raw = json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def weights_sha256(self) -> str:
        """Hash of parameter tensors only (float32 bytes, sorted keys)."""
        h = hashlib.sha256()
        for k in sorted(self.parameters()):
            h.update(k.encode())
            h.update(self.parameters()[k].astype(np.float32).tobytes())
        return h.hexdigest()


def synthetic_batch_stream(seed: int, n: int, cfg: ToyConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """FIXTURE data stream — not a real corpus."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        x = rng.randint(0, cfg.vocab_size, size=(4, 8)).astype(np.int64)
        y = rng.randint(0, cfg.n_classes, size=(4,)).astype(np.int64)
        out.append((x, y))
    return out
