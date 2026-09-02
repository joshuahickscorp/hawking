import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hcli.engine import Engine
from hcli.controller import Controller
from hcli.workspace import Workspace
from hcli.events import Event


class TestEngineConnector(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.workspace_path = Path(self.tmpdir.name)
        self.workspace_path.mkdir(exist_ok=True)
        (self.workspace_path / "README.md").write_text("# Test")
        (self.workspace_path / "main.py").write_text("print('hello')")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_read_only_answer_path(self):
        """Proves Controller.execute -> Engine for read-only requests."""
        workspace = Workspace(str(self.workspace_path))
        controller = Controller(workspace=workspace, model="test-model")
        
        # Mock the engine's model client to return a read-only answer
        mock_response = {
            "kind": "answer",
            "content": "The README says # Test",
            "operations": [],
            "tests": []
        }
        
        with patch.object(controller.engine, '_call_model', return_value=mock_response):
            result = controller.execute("read README.md")
            
        self.assertEqual(result["kind"], "answer")
        self.assertIn("README", result["content"])
        self.assertEqual(result["operations"], [])
        
        # Verify no mutation occurred
        self.assertEqual((self.workspace_path / "main.py").read_text(), "print('hello')")

    def test_mutation_path(self):
        """Proves mutation path with structured result parsing."""
        workspace = Workspace(str(self.workspace_path))
        controller = Controller(workspace=workspace, model="test-model")
        
        mock_response = {
            "kind": "mutation",
            "content": "Updated main.py",
            "operations": [
                {
                    "op": "replace",
                    "path": "main.py",
                    "old_text": "print('hello')",
                    "new_text": "print('world')"
                }
            ],
            "tests": []
        }
        
        with patch.object(controller.engine, '_call_model', return_value=mock_response):
            result = controller.execute("change hello to world in main.py")
            
        self.assertEqual(result["kind"], "mutation")
        self.assertEqual(len(result["operations"]), 1)
        
        # Verify mutation was applied
        self.assertEqual((self.workspace_path / "main.py").read_text(), "print('world')")

    def test_rollback_on_validation_failure(self):
        """Proves rollback on validation failure."""
        workspace = Workspace(str(self.workspace_path))
        controller = Controller(workspace=workspace, model="test-model")
        
        mock_response = {
            "kind": "mutation",
            "content": "Bad mutation",
            "operations": [
                {
                    "op": "replace",
                    "path": "main.py",
                    "old_text": "print('hello')",
                    "new_text": "def broken():\n    pass"
                }
            ],
            "tests": []
        }
        
        with patch.object(controller.engine, '_call_model', return_value=mock_response):
            with patch.object(controller.engine, '_validate', return_value=False):
                result = controller.execute("make a bad change")
                
        self.assertEqual(result["kind"], "mutation")
        self.assertTrue(result.get("rolled_back", False))
        
        # Verify rollback restored original
        self.assertEqual((self.workspace_path / "main.py").read_text(), "print('hello')")

    def test_receipt_creation(self):
        """Proves durable receipt creation."""
        workspace = Workspace(str(self.workspace_path))
        controller = Controller(workspace=workspace, model="test-model")
        
        mock_response = {
            "kind": "answer",
            "content": "Test answer",
            "operations": [],
            "tests": []
        }
        
        with patch.object(controller.engine, '_call_model', return_value=mock_response):
            controller.execute("test receipt")
            
        receipts_dir = self.workspace_path / ".hcli" / "receipts"
        self.assertTrue(receipts_dir.exists())
        
        receipt_files = list(receipts_dir.glob("*.json"))
        self.assertGreaterEqual(len(receipt_files), 1)
        
        # Verify receipt content
        with open(receipt_files[0]) as f:
            receipt = json.load(f)
            
        self.assertIn("goal", receipt)
        self.assertIn("model", receipt)
        self.assertIn("status", receipt)
        self.assertIn("timestamps", receipt)

    def test_workspace_escape_rejection(self):
        """Proves workspace escape rejection."""
        workspace = Workspace(str(self.workspace_path))
        controller = Controller(workspace=workspace, model="test-model")
        
        mock_response = {
            "kind": "mutation",
            "content": "Escape attempt",
            "operations": [
                {
                    "op": "replace",
                    "path": "../outside.py",
                    "old_text": "x",
                    "new_text": "y"
                }
            ],
            "tests": []
        }
        
        with patch.object(controller.engine, '_call_model', return_value=mock_response):
            result = controller.execute("escape workspace")
            
        self.assertTrue(result.get("error", "").lower().find("escape") >= 0 or result.get("rolled_back", False))

    def test_git_mutation_rejection(self):
        """Proves .git mutation rejection."""
        workspace = Workspace(str(self.workspace_path))
        controller = Controller(workspace=workspace, model="test-model")
        
        mock_response = {
            "kind": "mutation",
            "content": "Git mutation",
            "operations": [
                {
                    "op": "replace",
                    "path": ".git/config",
                    "old_text": "a",
                    "new_text": "b"
                }
            ],
            "tests": []
        }
        
        with patch.object(controller.engine, '_call_model', return_value=mock_response):
            result = controller.execute("mutate git")
            
        self.assertTrue(result.get("error", "").lower().find("git") >= 0 or result.get("rolled_back", False))

    def test_hidden_reasoning_never_rendered(self):
        """Proves hidden reasoning is never rendered in final response."""
        workspace = Workspace(str(self.workspace_path))
        controller = Controller(workspace=workspace, model="test-model")
        
        mock_response = {
            "kind": "answer",
            "content": "Final answer",
            "operations": [],
            "tests": [],
            "reasoning_content": "I think... let me consider...",
            "hidden_reasoning": "step 1, step 2"
        }
        
        with patch.object(controller.engine, '_call_model', return_value=mock_response):
            result = controller.execute("test reasoning")
            
        self.assertEqual(result["content"], "Final answer")
        self.assertNotIn("reasoning_content", result)
        self.assertNotIn("hidden_reasoning", result)

    def test_cancellation_safety(self):
        """Proves cancellation safety."""
        workspace = Workspace(str(self.workspace_path))
        controller = Controller(workspace=workspace, model="test-model")
        
        mock_response = {
            "kind": "mutation",
            "content": "Cancelled",
            "operations": [
                {
                    "op": "replace",
                    "path": "main.py",
                    "old_text": "print('hello')",
                    "new_text": "print('cancelled')"
                }
            ],
            "tests": []
        }
        
        with patch.object(controller.engine, '_call_model', return_value=mock_response):
            controller.cancel()
            result = controller.execute("test cancellation")
            
        self.assertTrue(result.get("cancelled", False) or result.get("error", "").lower().find("cancel") >= 0)

    def test_receipt_does_not_start_runtime_pool(self):
        """Receipt provenance must inspect state, never create inference state."""
        workspace = Workspace(str(self.workspace_path))
        controller = Controller(
            workspace=workspace,
            model="test-model",
        )

        mock_response = {
            "kind": "answer",
            "content": "read-only",
            "operations": [],
            "tests": [],
        }

        with patch.object(
            controller.engine,
            "_call_model",
            return_value=mock_response,
        ):
            result = controller.execute("read README.md")

        self.assertEqual(result["kind"], "answer")
        self.assertIsNone(controller.runtime_pool)


if __name__ == "__main__":
    unittest.main()