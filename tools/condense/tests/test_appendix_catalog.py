from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools" / "condense" / "appendix_catalog.py"
SPEC = importlib.util.spec_from_file_location("appendix_catalog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_catalog_covers_all_25_sectors_without_execution() -> None:
    catalog = MODULE.build_catalog()
    assert catalog == MODULE.build_catalog()
    assert catalog["execution_supported"] is False
    sectors = catalog["sectors"]
    assert [item["number"] for item in sectors] == list(range(1, 26))
    assert len({item["id"] for item in sectors}) == 25
    assert all(item["mutates_active_corpus"] is False for item in sectors)


def test_catalog_has_a_gate_and_tq_disposition_for_every_sector() -> None:
    for item in MODULE.build_catalog()["sectors"]:
        assert item["next_gate"]
        assert item["tq_relevance"] in {"direct", "indirect", "training"}
        assert set(item["currencies"]) <= MODULE.CURRENCIES
        assert item["hawking_state"] in MODULE.STATES
