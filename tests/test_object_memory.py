
import tempfile
import time
import unittest
from pathlib import Path

from go2.object_memory import ObjectMemory
from go2.executor import Executor


class ObjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.memory_path = Path(self.temp_dir.name) / "memory.json"

    def test_memory_survives_reload(self):
        memory = ObjectMemory(self.memory_path)
        memory.remember("chair", 1.5, 2.0, color="blue")

        reloaded = ObjectMemory(self.memory_path)
        result = reloaded.recall("chair")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["x"], 1.5)
        self.assertEqual(result[0]["y"], 2.0)
        self.assertEqual(result[0]["color"], "blue")

    def test_memory_matches_labels_case_insensitively(self):
        memory = ObjectMemory(self.memory_path)
        memory.remember("Chair", 1.0, 2.0)

        self.assertEqual(len(memory.recall("chair")), 1)

    def test_memory_updates_moved_object_position(self):
        memory = ObjectMemory(self.memory_path)
        memory.remember("chair", 1.0, 2.0, color="blue")
        memory.remember("chair", 1.2, 2.1, color="blue")

        result = memory.recall("chair")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["x"], 1.2)
        self.assertEqual(result[0]["y"], 2.1)

    def test_memory_keeps_separate_chairs(self):
        memory = ObjectMemory(self.memory_path)
        memory.remember("chair", 1.0, 2.0, color="blue")
        memory.remember("chair", 4.0, 5.0, color="blue")

        result = memory.recall("chair")

        self.assertEqual(len(result), 2)

    def test_memory_updates_seen_count_and_timestamp(self):
        memory = ObjectMemory(self.memory_path)
        memory.remember("chair", 1.0, 2.0, color="blue")
        first = memory.recall("chair")[0]
        first_timestamp = first["last_seen"]

        time.sleep(0.01)
        memory.remember("chair", 1.0, 2.0, color="blue")
        updated = memory.recall("chair")[0]

        self.assertEqual(updated["times_seen"], 2)
        self.assertGreater(updated["last_seen"], first_timestamp)

    def test_executor_reloads_persistent_memory(self):
        first = Executor(memory_path=self.memory_path)
        first.object_memory.remember(
            "red ball", 1.5, 2.0, color="red"
        )

        restarted = Executor(memory_path=self.memory_path)
        result = restarted._recall("red ball")

        self.assertIsNotNone(result)
        self.assertEqual(result[1], 1.5)
        self.assertEqual(result[2], 2.0)


if __name__ == "__main__":
    unittest.main()
