import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p0_tool_bridge as bridge


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages):
        if not self.responses:
            return json.dumps({"type": "final", "content": "done"}), None
        return self.responses.pop(0), None


class ProtocolTests(unittest.TestCase):
    def test_valid_tool(self):
        obj = bridge.parse_json_protocol('{"type":"tool","name":"git.status","args":{}}')
        self.assertEqual(obj["type"], "tool")
        self.assertEqual(obj["name"], "git.status")

    def test_valid_final(self):
        obj = bridge.parse_json_protocol('{"type":"final","content":"ok"}')
        self.assertEqual(obj["content"], "ok")

    def test_valid_spawn(self):
        obj = bridge.parse_json_protocol('{"type":"spawn","role":"scout","task":"find","context":{}}')
        self.assertEqual(obj["role"], "scout")

    def test_malformed_prose(self):
        with self.assertRaises(bridge.ProtocolError):
            bridge.parse_json_protocol("I will use git.status")

    def test_malformed_code_fence(self):
        with self.assertRaises(bridge.ProtocolError):
            bridge.parse_json_protocol('```json\n{"type":"final","content":"ok"}\n```')

    def test_unknown_type(self):
        with self.assertRaises(bridge.ProtocolError):
            bridge.parse_json_protocol('{"type":"maybe"}')

    def test_unknown_tool(self):
        with self.assertRaises(bridge.ProtocolError):
            bridge.parse_json_protocol('{"type":"tool","name":"fs.write","args":{}}')


class RepositoryGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "sub"))
        with open(os.path.join(self.root, "file.txt"), "w", encoding="utf-8") as f:
            f.write("x")
        self.guard = bridge.RepositoryGuard(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_inside(self):
        resolved = self.guard.resolve("sub")
        self.assertEqual(resolved, os.path.realpath(os.path.join(self.root, "sub")))

    def test_outside(self):
        with self.assertRaises(bridge.ToolError):
            self.guard.resolve("../outside")

    def test_dotdot_escape(self):
        with self.assertRaises(bridge.ToolError):
            self.guard.resolve("sub/../../outside")

    def test_symlink_escape(self):
        outside = tempfile.TemporaryDirectory()
        try:
            link = os.path.join(self.root, "link")
            os.symlink(outside.name, link)
            with self.assertRaises(bridge.ToolError):
                self.guard.resolve("link")
        finally:
            outside.cleanup()


class CommandAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "tools", "haider"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_git_status_ok(self):
        bridge.admit_command("git status", self.root)

    def test_git_diff_path_ok(self):
        bridge.admit_command("git diff -- path", self.root)

    def test_git_log_ok(self):
        bridge.admit_command("git log -n 5 --oneline", self.root)

    def test_git_reset_hard_rejected(self):
        with self.assertRaises(bridge.ToolError):
            bridge.admit_command("git reset --hard", self.root)

    def test_git_clean_rejected(self):
        with self.assertRaises(bridge.ToolError):
            bridge.admit_command("git clean -fd", self.root)

    def test_rg_ok(self):
        bridge.admit_command("rg --color=never -n pattern tools", self.root)

    def test_rg_path_escape_rejected(self):
        with self.assertRaises(bridge.ToolError):
            bridge.admit_command("rg pattern ../outside", self.root)

    def test_cargo_check_ok(self):
        bridge.admit_command("cargo check --workspace", self.root)

    def test_cargo_test_ok(self):
        bridge.admit_command("cargo test -- --exact foo", self.root)

    def test_python_admitted(self):
        bridge.admit_command("python3 tools/haider/test_p0_tool_bridge.py", self.root)

    def test_python_unittest_discover_ok(self):
        bridge.admit_command("python3 -m unittest discover -s tools/haider -p test_*.py", self.root)

    def test_python_unittest_module_ok(self):
        bridge.admit_command("python3 -m unittest tools.haider.test_p0_tool_bridge", self.root)

    def test_python_rejected(self):
        with self.assertRaises(bridge.ToolError):
            bridge.admit_command("python3 /tmp/x.py", self.root)

    def test_sudo_rejected(self):
        with self.assertRaises(bridge.ToolError):
            bridge.admit_command("sudo git status", self.root)

    def test_rm_rejected(self):
        with self.assertRaises(bridge.ToolError):
            bridge.admit_command("rm -rf /", self.root)


class TruncationTests(unittest.TestCase):
    def test_truncate_lines(self):
        text = "\n".join(f"line{i}" for i in range(1000))
        out, truncated = bridge.truncate_text(text, max_chars=10000, max_lines=10)
        self.assertTrue(truncated)
        self.assertIn("...[truncated]", out)

    def test_no_truncate(self):
        out, truncated = bridge.truncate_text("small", max_chars=100, max_lines=10)
        self.assertFalse(truncated)
        self.assertEqual(out, "small")


class ToolExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "tools", "haider"))
        self.guard = bridge.RepositoryGuard(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fs_read_truncation(self):
        path = os.path.join(self.root, "big.txt")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(1000):
                f.write(f"line{i}\n")
        executor = bridge.ToolExecutor(self.guard, max_output_lines=10)
        obs = executor.execute("fs.read", {"path": "big.txt"})
        self.assertTrue(obs["truncated"])
        self.assertEqual(obs["lines"], 10)

    def test_tool_timeout(self):
        script = os.path.join(self.root, "tools", "haider", "sleep.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write("import time\ntime.sleep(2)\n")
        exe = os.path.basename(sys.executable)
        python_cmd = "python3" if exe.startswith("python3") else "python"
        executor = bridge.ToolExecutor(self.guard, timeout=0.2)
        obs = executor.execute(
            "shell.run_safe",
            {"command": f"{python_cmd} tools/haider/sleep.py", "timeout": 0.2},
        )
        self.assertEqual(obs["exit_code"], 124)
        self.assertFalse(obs["ok"])


class ModelTimeoutTests(unittest.TestCase):
    def test_model_timeout(self):
        def opener(req, timeout=None):
            raise TimeoutError()

        client = bridge.ModelClient(
            api_base="http://127.0.0.1:9999/v1",
            model="test",
            api_key="key",
            timeout=1,
            opener=opener,
        )
        with self.assertRaises(bridge.ModelError):
            client.chat([{"role": "user", "content": "hi"}])


class WatchdogTests(unittest.TestCase):
    def test_repeated_failed_call_stops(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            root = tmp.name
            guard = bridge.RepositoryGuard(root)
            executor = bridge.ToolExecutor(guard)
            responses = [
                json.dumps({"type": "tool", "name": "fs.read", "args": {"path": "missing"}})
                for _ in range(3)
            ]
            client = FakeClient(responses)
            session = bridge.Session(
                "parent",
                "task",
                client,
                executor,
                guard,
                max_turns=5,
                emit=lambda _msg: None,
            )
            result = session.run()
            self.assertFalse(result["ok"])
            self.assertEqual(session.stats["tool_calls"], 3)
        finally:
            tmp.cleanup()


class WorkerNormalizationTests(unittest.TestCase):
    def test_json(self):
        packet = bridge.normalize_worker_output(
            "scout",
            json.dumps({"claims": ["a"], "files": ["b"]}),
        )
        self.assertEqual(packet["claims"], ["a"])
        self.assertEqual(packet["files"], ["b"])

    def test_prose(self):
        packet = bridge.normalize_worker_output("scout", "found stuff")
        self.assertEqual(packet["claims"], ["found stuff"])

    def test_empty(self):
        packet = bridge.normalize_worker_output("adversary", "")
        self.assertEqual(packet["claims"], [])


class LaneValidationTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(bridge.validate_lanes(3), 3)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            bridge.validate_lanes(4)


if __name__ == "__main__":
    unittest.main()
