import unittest
from hcli.perception.store import ProjectStore
class TestProjectStore(unittest.TestCase):
    def test_create_empty_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            ProjectStore.create("", "")
    def test_create_non_empty_name_works(self):
        store = ProjectStore.create("p", "data")
        self.assertIsNotNone(store)
if __name__ == '__main__':
    unittest.main()
# Ensure the test fails when empty name is passed
