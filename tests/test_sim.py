"""Integration: the executor driving the real MuJoCo simulation (headless). Skipped without mujoco."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

try:
    import mujoco  # noqa: F401
    import torch  # noqa: F401
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False

from go2.executor import Executor  # noqa: E402
from go2.tools import resolve  # noqa: E402
from tests.fakes import needs_camera, wait_for  # noqa: E402


@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from go2.adapters.sim import SimAdapter
        cls.sim = SimAdapter(python=sys.executable, scene="room", headless=True)
        cls.sim.connect()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.reset()
        self.events = []
        self.executor = Executor(on_event=self.events.append)
        self.executor.attach(self.sim)

    def tearDown(self):
        self.executor.close()

    def run_tool(self, name, args, timeout=40):
        before = len(self.events)
        result = self.executor.submit(resolve(name, args), "g", 1)
        if result[0]["status"] != "queued":
            return result[0]
        self.assertTrue(wait_for(lambda: len(self.events) > before, timeout))
        return self.events[-1]

    def test_move_and_turn_accuracy(self):
        start = self.sim.state()
        task = self.run_tool("move", {"direction": "forward", "distance_m": 1.5})
        self.assertEqual(task.status, "completed")
        self.assertAlmostEqual(self.sim.state().x - start.x, 1.5, delta=.08)
        task = self.run_tool("turn", {"direction": "left", "angle_deg": 90})
        self.assertEqual(task.status, "completed")
        self.assertAlmostEqual(math.degrees(self.sim.state().yaw), 90, delta=8)

    def test_top_speed_move_reaches_four_metres(self):
        task = self.run_tool("move", {"direction": "forward", "distance_m": 4})   # 0.8 m/s, open path
        self.assertEqual(task.status, "completed")
        self.assertAlmostEqual(task.detail["moved_m"], 4, delta=.125)   # 12 cm either way, in whole centimetres

    def test_wall_stops_motion_without_contact(self):
        task = self.run_tool("move", {"direction": "backward", "distance_m": 2})   # west wall is ~1.4 m behind
        self.assertEqual(task.status, "blocked")
        self.assertGreaterEqual(task.detail["blocked"]["clearance_m"], .08, "stopped with a real gap")
        x = self.sim.state().x
        self.assertGreater(x - .36, -3.95 + .05, "rear edge never reached the wall face")
        self.assertTrue(wait_for(lambda: self.sim.state().posture == "standing", 3))
        self.assertIsNone(self.sim.state().fault, "no collision or fall")

    def test_postures_and_trick(self):
        for name, args, posture in (("posture", {"name": "sit"}, "sitting"), ("posture", {"name": "stand"}, "standing"),
                                    ("posture", {"name": "lie_down"}, "lying"), ("trick", {"name": "hello"}, "standing")):
            task = self.run_tool(name, args)
            self.assertEqual(task.status, "completed", (name, args, task.detail))
            self.assertEqual(self.sim.state().posture, posture)

    def test_acrobatics_refused_in_simulation(self):
        result = self.run_tool("acrobatic", {"name": "front_flip"})
        self.assertEqual(result["status"], "rejected")

    @needs_camera
    def test_look_finds_the_red_ball(self):
        report = self.executor.look()
        balls = [o for o in report["objects"] if o["label"] == "sports ball" and o["color"] == "red"]
        self.assertEqual(len(balls), 1, report)
        # Ground truth: ball centre (0.6, 1.3), radius 0.12; robot at (-2.5, 0) facing +x.
        self.assertAlmostEqual(balls[0]["distance_m"], math.hypot(3.1, 1.3) - .12, delta=.25)
        self.assertGreater(balls[0]["bearing_deg"], 10)
        self.assertEqual(balls[0]["distance_source"], "lidar")
        self.assertIsNotNone(report["free_space_m"]["front"])

    @needs_camera
    def test_yolo_mode_detects_scanned_objects(self):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            self.skipTest("ultralytics not installed")
        self.sim.detector = "yolo"
        try:
            self.sim.reset(-.5, 0., 0.)      # facing the chair, backpack and toy dog
            self.executor.attach(self.sim)
            report = self.executor.look()
        finally:
            self.sim.detector = "truth"
        labels = {o["label"] for o in report.get("objects", [])}
        self.assertTrue(labels, report)
        self.assertTrue(labels & {"chair", "backpack", "dog", "bowl", "sports ball", "truck"}, labels)
        self.assertTrue(any(o["distance_m"] for o in report["objects"]))

    @needs_camera
    def test_approach_stops_at_the_requested_gap(self):
        task = self.run_tool("approach", {"target": "red ball", "stop_distance_m": .5}, timeout=90)
        self.assertEqual(task.status, "completed", task.detail)
        self.assertAlmostEqual(task.detail["final_gap_m"], .5, delta=.2)

    @needs_camera
    def test_approach_walks_toward_an_object_it_cannot_range_yet(self):
        looks = [0]

        def no_distance_at_first(detections):
            looks[0] += 1
            if looks[0] <= 2:
                for d in detections:
                    d.distance_m = d.x_m = d.y_m = None
                    d.distance_source = "unknown"
            return detections
        self.executor.detection_filter = no_distance_at_first
        task = self.run_tool("approach", {"target": "red ball"}, timeout=120)
        self.assertEqual(task.status, "completed", task.detail)
        self.assertGreater(looks[0], 2)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class LidarTests(unittest.TestCase):
    def test_floor_is_seen_far_from_the_world_origin(self):
        import mujoco
        import world as W
        from sensors import Lidar
        model, _ = W.build("empty")
        data = mujoco.MjData(model)
        lidar = Lidar(model)
        free = model.jnt_qposadr[0]
        for x, y in [(0, 0), (26.6, -14.95), (60, 40)]:     # the second was ~30.5 m out and saw nothing
            data.qpos[free:free + 3] = (x, y, .32)
            data.qpos[free + 3:free + 7] = (1, 0, 0, 0)
            mujoco.mj_forward(model, data)
            self.assertGreater(len(lidar.scan(data)), 1500, f"floor visible at ({x}, {y})")
