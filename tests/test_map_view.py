import base64
import sys
from pathlib import Path
import unittest
import numpy as np
from go2.executor import Executor
from go2.map_view import map_view
from go2.room_map import OccupancyGrid, FREE, OCCUPIED
from tests.fakes import FakeRobot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"command_ui"))
try:
    import server
except ImportError:
    server = None


class MapViewTests(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.robot.connect()
        self.addCleanup(self.robot.close)
        self.executor = Executor()
        self.executor.attach(self.robot)
        self.addCleanup(self.executor.close)

    def test_map_is_bounded_preserves_obstacles_and_world_origin(self):
        m = self.executor.room_map = OccupancyGrid(.05, 400, grow=True)
        m.mark_obstacles([[11., -12.]])
        result = map_view(self.executor, max_pixels=32)
        g = result["grid"]
        cells = np.frombuffer(base64.b64decode(g["data"]), dtype=np.uint8).reshape(g["height"], g["width"])
        self.assertLessEqual(max(cells.shape), 32)
        col, row = np.floor((np.array([11., -12.])-g["origin_m"])/g["resolution_m"]).astype(int)
        self.assertEqual(cells[row, col], 2)
        self.assertFalse(any(c[0] in ("move", "lidar", "camera") for c in self.robot.calls))

    def test_named_places_and_missing_objects_share_map_coordinates(self):
        self.executor.places.remember("door <test>", 1., 2., .3)
        item = self.executor.object_memory.remember("laptop", -1., .5)
        for stamp in (0., .2, .4):
            self.executor.object_memory.observe_empty([item["id"]], observed_at=stamp)
        result = map_view(self.executor)
        self.assertEqual(result["places"][0]["name"], "door <test>")
        self.assertEqual(result["places"][0]["x"], 1.)
        self.assertEqual(result["objects"][0]["state"], "missing")
        self.assertEqual(result["frame"], "map")

    def test_disconnected_map_is_readable_with_error(self):
        self.robot.close()
        result = map_view(self.executor)
        self.assertIsNone(result["robot"])
        self.assertIn("disconnected", result["error"])

    @unittest.skipIf(server is None, "console dependencies not installed")
    def test_map_endpoint_is_read_only_and_uses_existing_host_guard(self):
        from go2.mock_model import MockModel
        console = server.Console(MockModel(), self.robot)
        self.addCleanup(console.close)
        client = server.make_app(console, 5050).test_client()
        response = client.get("/api/map", base_url="http://127.0.0.1:5050")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["grid"]["encoding"], "base64-u8")
        self.assertEqual(client.get("/api/map", base_url="http://evil.test:5050").status_code, 403)
        self.assertEqual(client.post("/api/map", base_url="http://127.0.0.1:5050").status_code, 405)
        self.assertIsNone(console.executor.last_frame)
