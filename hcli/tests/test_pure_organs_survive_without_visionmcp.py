"""The salvage set must not need the foreign package.

Sublation replaces a runtime dependency on visionmcp with hawking-native
capability. Four modules under hcli/agentos/vmcp/ already qualify: receipt,
pty_eye, tool_doctor and file_eye import stdlib and each other, nothing else.

Grep alone does not prove this -- those files mention "visionmcp" 0-12 times
each in docstrings, and disposition.py mentions it 48 times while importing it
zero times. Prose is not a dependency. The only honest check is to make the
package unreachable and see whether the import still works, which is the same
acceptance step docs/VMCP_SUBLATION_CONTRACT.md requires of the port itself.
"""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PURE = ("receipt", "pty_eye", "tool_doctor", "file_eye")


@pytest.mark.parametrize("mod", PURE)
def test_no_module_in_the_salvage_set_imports_visionmcp(mod):
    src = (REPO / "hcli" / "agentos" / "vmcp" / f"{mod}.py").read_text()
    for i, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        assert not (s.startswith("import visionmcp") or s.startswith("from visionmcp")), (
            f"{mod}.py:{i} imports the foreign package: {s}")
        assert 'import_module("visionmcp' not in s, f"{mod}.py:{i}: {s}"


def test_they_import_with_the_package_made_unreachable(tmp_path):
    """The real check: block visionmcp from sys.path entirely, in a COLD
    interpreter, and require every pure organ to still import.

    Run in a subprocess because this process may already have the package
    cached in sys.modules, which would make the test pass for the wrong reason.
    """
    code = (
        "import sys, importlib\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        # Make any attempt to import the foreign package fail loudly.
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'visionmcp' or name.startswith('visionmcp.'):\n"
        "            raise ImportError('visionmcp is blocked by this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        f"for m in {PURE!r}:\n"
        "    importlib.import_module('hcli.agentos.vmcp.' + m)\n"
        "print('OK')\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0 and "OK" in p.stdout, (
        f"a pure organ needs visionmcp after all:\n{p.stderr[-1500:]}")


def test_the_blocker_actually_blocks():
    """Negative control, with a BASELINE -- without one this proves nothing.

    An earlier version of this test only showed that `import visionmcp` fails
    with the blocker installed. But the package is a separate gitignored
    checkout that is not on sys.path anyway, so it fails either way: the test
    could not distinguish "the blocker worked" from "the package was never
    reachable". A control that cannot fail for the reason it claims is not a
    control.

    So: put the real source root on sys.path, prove the import SUCCEEDS, then
    install the blocker and prove it stops. If the checkout is absent the
    baseline is unavailable and the test skips loudly rather than passing on a
    vacuous half.
    """
    src = REPO / "visionmcp" / "src"
    if not (src / "visionmcp" / "__init__.py").is_file():
        pytest.skip(f"no visionmcp checkout at {src}; baseline unavailable")

    baseline = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(src)!r}); import visionmcp; print('IMPORTED')"],
        capture_output=True, text=True)
    assert "IMPORTED" in baseline.stdout, (
        f"baseline failed: visionmcp is not importable even unblocked, so the "
        f"blocker below would prove nothing.\n{baseline.stderr[-600:]}")

    blocked = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(src)!r})\n"
         "class B:\n"
         "    def find_spec(self, name, path=None, target=None):\n"
         "        if name == 'visionmcp' or name.startswith('visionmcp.'):\n"
         "            raise ImportError('blocked')\n"
         "        return None\n"
         "sys.meta_path.insert(0, B())\n"
         "try:\n"
         "    import visionmcp; print('NOT_BLOCKED')\n"
         "except ImportError:\n"
         "    print('BLOCKED')\n"],
        capture_output=True, text=True)
    assert "BLOCKED" in blocked.stdout, (
        f"the blocker does not block a REACHABLE package: {blocked.stdout} "
        f"{blocked.stderr[-400:]}")
