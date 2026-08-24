"""One module identity for the HCLI control plane (incident F24).

Python loaded the same files under two dotted names:

* ``hcli.engine``               with sys.path = <repo>/tools/haider
* ``tools.haider.hcli.engine``  with sys.path = <repo>

Those were different module objects, so a monkeypatch on one was
invisible to the other. Watched failing before the migration:

    a.Engine is b.Engine -> False  (same __file__, two class objects)

This test is the lock: the fossil dotted name must not resolve, and
sys.modules must not hold a second engine module for that path.
"""
from __future__ import annotations

import importlib
import sys
import unittest

# Built so a mechanical ``tools.haider.hcli`` -> ``hcli`` rewrite cannot
# delete the name this test is guarding against.
_FOSSIL_ENGINE = "tools" + ".haider.hcli.engine"


class TestOneModuleIdentity(unittest.TestCase):
    def test_canonical_engine_imports(self):
        import hcli.engine as canonical

        self.assertEqual(canonical.__name__, "hcli.engine")
        self.assertTrue(hasattr(canonical, "Engine"))

    def test_fossil_name_raises_import_error(self):
        import hcli.engine as canonical

        with self.assertRaises(ImportError):
            importlib.import_module(_FOSSIL_ENGINE)

        self.assertEqual(canonical.__name__, "hcli.engine")

    def test_sys_modules_has_one_engine(self):
        import hcli.engine as canonical

        engines = [
            name
            for name, mod in sys.modules.items()
            if mod is not None
            and getattr(mod, "__file__", None) == canonical.__file__
            and name.endswith(".engine")
            and "tests" not in name
        ]
        self.assertIn("hcli.engine", engines)
        self.assertNotIn(_FOSSIL_ENGINE, engines)


if __name__ == "__main__":
    unittest.main()
