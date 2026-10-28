
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "command_ui")]

from go2.executor import Executor
from go2.object_memory import ObjectMemory
from go2.perception import Detection
from tests.fakes import FakeRobot

try:
    import server  # noqa: E402  (command_ui/server.py)
except ImportError:                     # flask not installed: the console's own requirements
    server = None


def seen(label, color, x, y):
    return Detection(id="obj1", label=label, color=color, distance_m=x, bearing_deg=0., x_m=x, y_m=y,
                     distance_source="lidar", confidence=.9)


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
        first.object_memory.remember("sports ball", 1.5, 2.0, color="red")   # what the detector really says

        restarted = Executor(memory_path=self.memory_path)
        result = restarted._recall("red ball")

        self.assertIsNotNone(result)
        self.assertEqual(result[1], 1.5)
        self.assertEqual(result[2], 2.0)

    def test_a_moved_object_is_recalled_where_it_was_last_seen(self):
        executor = Executor(memory_path=self.memory_path)
        executor._remember([seen("sports ball", "red", 2.0, .5)], (0., 0., 0.))
        time.sleep(.01)
        executor._remember([seen("sports ball", "red", 4.0, -1.0)], (0., 0., 0.))   # it has moved
        _, x, y = executor._recall("red ball")
        self.assertEqual((round(x, 1), round(y, 1)), (4.0, -1.0))
        restarted = Executor(memory_path=self.memory_path)                          # and after a restart
        _, x, y = restarted._recall("red ball")
        self.assertEqual((round(x, 1), round(y, 1)), (4.0, -1.0))

    def test_look_results_dont_list_the_saved_memory(self):
        # go2-ft-04 was trained on look without it (#14 leaves extra look data for #16).
        executor = Executor(memory_path=self.memory_path)
        self.addCleanup(executor.close)
        executor.object_memory.remember("chair", 1.0, 2.0)
        robot = FakeRobot()
        robot.connect()
        self.addCleanup(robot.close)
        executor.attach(robot)
        self.assertNotIn("remembered_objects", executor.look())

    @unittest.skipIf(server is None, "console requirements not installed")
    def test_one_memory_file_per_scene_and_none_on_the_real_robot(self):
        def args(**kw):
            return SimpleNamespace(**{"robot": None, "scene": "room", "seed": None, **kw})
        room, house = server.memory_file(args()), server.memory_file(args(scene="house"))
        seeded = server.memory_file(args(scene="house", seed=3))
        self.assertEqual(len({room, house, seeded}), 3)
        self.assertEqual(seeded.name, "house-seed3.json")
        self.assertIsNone(server.memory_file(args(robot="en7")))

    def test_an_executor_without_a_path_keeps_no_file(self):
        executor = Executor()
        executor._remember([seen("chair", "blue", 1.0, 0.)], (0., 0., 0.))
        self.assertIsNone(executor.object_memory.path)

    def test_save_false_delays_disk_write(self):
        memory = ObjectMemory(self.memory_path)

        memory.remember("chair", 1.0, 2.0, save=False)
        self.assertFalse(self.memory_path.exists())

        memory.save()
        self.assertTrue(self.memory_path.exists())

if __name__ == "__main__":
    unittest.main()
