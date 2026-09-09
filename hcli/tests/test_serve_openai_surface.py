"""The wire contract Open WebUI actually depends on.

Driven against the real handler over a real socket with a fake backend, because
the failure this guards is a SHAPE failure -- a missing `created`, a stream flag
answered with a JSON body instead of SSE -- and none of those are visible to a
test that calls the functions directly.
"""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from hcli.serve import (
    GREEDY_OK,
    chat_payload,
    make_handler,
    models_payload,
    profile_is_greedy,
    sampler_refusal,
    stream_frames,
)


class _Result:
    text = "batched matmul reuses each loaded weight across many rows."
    finish_reason = "stop"
    prompt_tokens = 11
    completion_tokens = 9
    total_tokens = 20
    degraded: list = []
    raw: dict = {}


class _Backend:
    def __init__(self):
        self.seen = []

    def complete(self, payload, timeout=None):
        self.seen.append(payload)
        return _Result()


def _serve(greedy=True):
    backend = _Backend()
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(backend, "sealed-3.14", greedy=greedy,
                     health={"status": "ok", "resident": "sealed-3.14"}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, backend, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(base, body, path="/v1/chat/completions"):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


class TestOpenAISurface(unittest.TestCase):
    def setUp(self):
        self.httpd, self.backend, self.base = _serve()
        self.addCleanup(self.httpd.shutdown)

    def test_models_carries_the_fields_the_picker_reads(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
            data = json.loads(r.read())
        self.assertEqual(data["object"], "list")
        row = data["data"][0]
        self.assertEqual(row["id"], "sealed-3.14")
        # A missing `created` renders as "Invalid Date" in the Open WebUI picker.
        self.assertIsInstance(row["created"], int)
        self.assertEqual(row["owned_by"], "hawking")

    def test_non_stream_answer_is_openai_shaped(self):
        code, ctype, raw = _post(self.base, {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200, raw)
        body = json.loads(raw)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertIn("batched matmul", body["choices"][0]["message"]["content"])
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"]["total_tokens"], 20)

    def test_stream_true_returns_SSE_not_json(self):
        # The defect this pins: answering a streaming request with a JSON body
        # leaves the browser waiting forever on a stream that never frames.
        code, ctype, raw = _post(
            self.base, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(code, 200, raw)
        self.assertIn("text/event-stream", ctype)
        text = raw.decode()
        self.assertTrue(text.startswith("data: "), text[:80])
        self.assertIn("[DONE]", text)
        deltas = [json.loads(line[6:])
                  for line in text.splitlines()
                  if line.startswith("data: ") and "[DONE]" not in line]
        self.assertEqual(deltas[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertIn("batched matmul", deltas[1]["choices"][0]["delta"]["content"])
        self.assertEqual(deltas[-1]["choices"][0]["finish_reason"], "stop")
        for d in deltas:
            self.assertEqual(d["object"], "chat.completion.chunk")

    def test_the_stream_flag_never_reaches_the_resident(self):
        _post(self.base, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertNotIn("stream", self.backend.seen[-1])

    def test_a_sampler_request_is_refused_with_the_accepted_value(self):
        code, _, raw = _post(self.base, {
            "messages": [{"role": "user", "content": "hi"}], "temperature": 0.8})
        self.assertEqual(code, 400)
        message = json.loads(raw)["error"]["message"]
        self.assertIn("temperature=0.0", message)
        self.assertIn("Advanced Params", message, "the refusal does not say where to fix it")

    def test_greedy_defaults_are_accepted(self):
        code, _, raw = _post(self.base, {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0, "top_p": 1, "n": 1})
        self.assertEqual(code, 200, raw)

    def test_no_messages_says_the_shape_it_wants(self):
        code, _, raw = _post(self.base, {})
        self.assertEqual(code, 400)
        self.assertIn("'role'", json.loads(raw)["error"]["message"])

    def test_an_unknown_route_names_the_ones_that_exist(self):
        code, _, raw = _post(self.base, {}, path="/v1/completions")
        self.assertEqual(code, 404)
        self.assertIn("/v1/chat/completions", json.loads(raw)["error"]["message"])


class TestNonGreedyBackendDoesNotRefuse(unittest.TestCase):
    def test_an_mlx_specimen_may_sample(self):
        httpd, _, base = _serve(greedy=False)
        self.addCleanup(httpd.shutdown)
        code, _, raw = _post(base, {
            "messages": [{"role": "user", "content": "hi"}], "temperature": 0.8})
        self.assertEqual(code, 200, raw)


class TestGreedyDetection(unittest.TestCase):
    def test_do_sample_false_is_greedy(self, ):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "profile.json"
            p.write_text(json.dumps({"generation": {"do_sample": False}}))
            self.assertTrue(profile_is_greedy(str(p)))

    def test_a_sampling_profile_is_not_greedy(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "profile.json"
            p.write_text(json.dumps({"generation": {"do_sample": True, "temperature": 0.7}}))
            self.assertFalse(profile_is_greedy(str(p)))

    def test_the_real_sealed_profile_is_greedy(self):
        import pathlib
        profile = pathlib.Path(__file__).resolve().parents[1] / "hawking-native.sealed-3.14.json"
        self.assertTrue(profile.is_file(), profile)
        self.assertTrue(profile_is_greedy(str(profile)),
                        "the sealed profile decodes argmax and must be detected as greedy")

    def test_true_is_not_accepted_as_the_integer_one(self):
        # `True == 1`, so a bare membership test would wave top_k=True through.
        self.assertIsNotNone(sampler_refusal({"top_k": True}))
        self.assertIsNone(sampler_refusal({"top_k": 1}))

    def test_unset_samplers_are_fine(self):
        self.assertIsNone(sampler_refusal({}))
        self.assertIsNone(sampler_refusal({k: None for k in GREEDY_OK}))


class TestPayloadHelpers(unittest.TestCase):
    def test_single_chunk_streaming_is_labelled_in_the_answer(self):
        body = chat_payload(_Result(), "sealed-3.14", request_id="chatcmpl-x")
        self.assertEqual(body["hawking"]["streaming"], "single-chunk")

    def test_frames_end_with_done(self):
        frames = list(stream_frames(_Result(), "m", request_id="r"))
        self.assertEqual(frames[-1], b"data: [DONE]\n\n")

    def test_models_payload_shape(self):
        self.assertEqual(models_payload("m")["data"][0]["id"], "m")


if __name__ == "__main__":
    unittest.main()
