"""Integration (#22): the executor on the MuJoCo "hazards" platform — a 0.5 m ledge, stairs going down and a
glass pane the LiDAR can't see. Headless; skipped without mujoco/torch."""
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

from go2 import executor as X  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.tools import resolve  # noqa: E402
from tests.fakes import wait_for  # noqa: E402

LEDGE_X, STAIRS_Y, GLASS_X = 1.0, 2.0, -1.0     # see simulation/hazards.py
HALF_LENGTH = .36


@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class HazardsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from go2.adapters.sim import SimAdapter
        cls.sim = SimAdapter(python=sys.executable, scene="hazards", headless=True)
        cls.sim.connect()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.events = []
        self.executor = Executor(on_event=self.events.append)
        self.executor.attach(self.sim)

    def tearDown(self):
        self.executor.close()

    def place(self, x, y, yaw):
        self.sim.reset(x, y, yaw)
        self.assertTrue(wait_for(lambda: self.sim.state().posture == "standing", 5))

    def run_tool(self, name, args, timeout=40):
        before = len(self.events)
        result = self.executor.submit(resolve(name, args), "g", 1)
        if result[0]["status"] != "queued":
            return result[0]
        self.assertTrue(wait_for(lambda: len(self.events) > before, timeout))
        return self.events[-1]

    def assert_safe(self):
        s = self.sim.state()
        self.assertIsNone(s.fault, "no fall or collision")
        self.assertAlmostEqual(s.height, .32, delta=.06, msg="still standing on the platform")

    def test_height_is_measured_from_the_platform(self):
        self.place(-2.5, 0, 0)
        self.assert_safe()

    def test_walking_on_the_platform_is_not_mistaken_for_a_drop(self):
        self.place(-2.5, 0, 0)
        task = self.run_tool("move", {"direction": "forward", "distance_m": 1.5})
        self.assertEqual(task.status, "completed", task.detail)
        self.assert_safe()

    def test_stops_before_the_ledge(self):
        self.place(-.3, 0, 0)
        task = self.run_tool("move", {"direction": "forward", "distance_m": 3})
        self.assertEqual(task.status, "blocked")
        self.assertEqual(task.detail["blocked"]["reason"], "drop ahead")
        self.assertLess(self.sim.state().x + HALF_LENGTH, LEDGE_X, "front of the body is still over the platform")
        self.assert_safe()

    def test_stops_before_the_stairs_going_down(self):
        self.place(-2.0, .6, math.pi / 2)
        task = self.run_tool("move", {"direction": "forward", "distance_m": 3})
        self.assertEqual(task.status, "blocked")
        self.assertEqual(task.detail["blocked"]["reason"], "drop ahead")
        self.assertLess(self.sim.state().y + HALF_LENGTH, STAIRS_Y)
        self.assert_safe()

    def test_glass_pane_is_caught_by_the_stall_check(self):
        self.place(-2.6, -1.3, 0)
        task = self.run_tool("move", {"direction": "forward", "distance_m": 2.5})
        self.assertEqual(task.status, "blocked", task.detail)
        self.assertTrue(task.detail["blocked"].get("stall"), task.detail)
        self.assertEqual(task.detail["blocked"]["reason"], X.STALLED)
        self.assertLess(self.sim.state().x + HALF_LENGTH, GLASS_X, "did not push through")
        self.assert_safe()
        retry = self.run_tool("move", {"direction": "forward", "distance_m": 2.5})
        self.assertEqual(retry.get("status") if isinstance(retry, dict) else retry.status, "blocked")



@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class OpenFloorTests(unittest.TestCase):
    """No false stalls: short side-steps on an empty floor, where there is nothing to bump into. The legs take a
    moment to get going, which used to look like a stall in about one side-step in three."""

    def test_short_side_steps_never_stall(self):
        from go2.adapters.sim import SimAdapter
        sim = SimAdapter(python=sys.executable, scene="empty", headless=True)
        sim.connect()
        self.addCleanup(sim.close)
        events = []
        executor = Executor(on_event=events.append)
        executor.attach(sim)
        self.addCleanup(executor.close)
        sim.reset(0, 0, 0)
        self.assertTrue(wait_for(lambda: sim.state().posture == "standing", 5))
        stalls = []
        for i in range(20):
            n = len(events)
            executor.submit(resolve("move", {"direction": "right" if i % 2 else "left", "distance_m": .25}), "g", 1)
            self.assertTrue(wait_for(lambda: len(events) > n, 20))
            if (events[-1].detail.get("blocked") or {}).get("stall"):
                stalls.append(events[-1].detail.get("moved_m"))
            wait_for(lambda: sim.state().speed < .05, 2)
        self.assertEqual(stalls, [], f"{len(stalls)} of 20 side-steps stopped as stalls")


if __name__ == "__main__":
    unittest.main()
