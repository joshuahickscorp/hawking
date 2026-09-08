"""A search result must name the SOURCE, not the search engine's redirect.

Measured: every url in a web.search result was
`https://www.bing.com/ck/a?!&&p=...&u=a1<base64>` -- Bing's click-tracking
wrapper -- while the tool stamped its own output `confidence:
"source-links-extracted"`. Nothing downstream could cite a source, dedupe by
domain, or tell two results apart, and the provenance the campaign requires for
web research was a hostname belonging to the engine.

The destination is sitting in the `u` parameter, base64 after an `a1` prefix.
"""
from __future__ import annotations

import unittest

from hcli.tool_registry import _bing_destination


class TestBingRedirectIsDecoded(unittest.TestCase):
    def test_a_redirect_resolves_to_its_destination(self):
        # u = "a1" + urlsafe base64 of the real URL
        import base64
        real = "https://docs.openwebui.com/getting-started/env-configuration"
        token = "a1" + base64.urlsafe_b64encode(real.encode()).decode().rstrip("=")
        wrapped = f"https://www.bing.com/ck/a?!&&p=abc&u={token}&ntb=1"
        self.assertEqual(_bing_destination(wrapped), real)

    def test_a_direct_url_is_returned_unchanged(self):
        direct = "https://docs.openwebui.com/"
        self.assertEqual(_bing_destination(direct), direct)

    def test_an_undecodable_redirect_falls_back_to_the_original(self):
        # Never invent a URL: if the wrapper cannot be decoded, the wrapper is
        # still the honest answer.
        bad = "https://www.bing.com/ck/a?!&&p=abc&u=a1@@@notbase64@@@"
        self.assertTrue(_bing_destination(bad).startswith("https://www.bing.com/ck/"))

    def test_a_redirect_with_no_u_parameter_is_unchanged(self):
        bare = "https://www.bing.com/ck/a?!&&p=abc"
        self.assertEqual(_bing_destination(bare), bare)


if __name__ == "__main__":
    unittest.main()
