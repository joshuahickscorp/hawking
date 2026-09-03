#!/usr/bin/env python3
"""doctor/repack — EVALUATE and SEAL as two verbs.

    python -m lab.operators.doctor_repack evaluate --budget 1.0
    python -m lab.operators.doctor_repack seal --budget 1.0
    python -m lab.operators.doctor_repack ladder --budgets 1.5,1.25,1.0,0.85,0.75 --seal-lowest
    python -m lab.operators.doctor_repack build-index --from-selection PATH

EVALUATE is pure CPU over the compact score index (target <50 ms per budget).
SEAL writes payloads + manifest, delta-only against a prior seal.
"""
from __future__ import annotations

import sys

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
