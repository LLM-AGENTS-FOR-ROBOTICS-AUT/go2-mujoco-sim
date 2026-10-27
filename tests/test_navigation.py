"""go_to plans a route around obstacles on the room map (#14): with the kinematic FakeRobot, and in the MuJoCo
room (headless, no camera needed, so it also runs on CI)."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "simulation")]

from go2.executor import Executor  # noqa: E402
from go2.tools import resolve  # noqa: E402
from tests.fakes import FakeRobot, wait_for  # noqa: E402

try:
    import mujoco  # noqa: F401
    import torch  # noqa: F401
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False


class GoToRouteTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.executor = Executor(on_event=self.events.append)
        self.addCleanup(self.executor.close)

    def attach(self, **kwargs):
        robot = FakeRobot(**kwargs)
        robot.connect()
        self.addCleanup(robot.close)
        self.executor.attach(robot)
        return robot

    def go_to(self, x, y=0., timeout=60):
        before = len(self.events)
        result = self.executor.submit(resolve("go_to", {"x_m": x, "y_m": y}), "g", 1)
        if result[0]["status"] not in ("queued",):
            return result[0]
        self.assertTrue(wait_for(lambda: len(self.events) > before, timeout), "go_to did not finish")
        return self.events[-1]

    def test_goes_around_a_box_in_the_way(self):
        robot = self.attach(obstacles=[(1.2, 1.6, -.4, .4)])
        task = self.go_to(3.0)
        self.assertEqual(task.status, "completed", task.detail)
        self.assertLess(task.detail["remaining_m"], .15)
        self.assertGreaterEqual(len(task.detail["route"]), 2, "went via a waypoint beside the box")
        self.assertAlmostEqual(robot.x, 3.0, delta=.15)

    def test_a_clear_straight_line_needs_no_route(self):
        robot = self.attach()
        task = self.go_to(2.0)
        self.assertEqual(task.status, "completed", task.detail)
        self.assertNotIn("route", task.detail)
        self.assertAlmostEqual(robot.x, 2.0, delta=.15)

    def test_a_wall_with_no_way_round_still_stops_safely(self):
        robot = self.attach(obstacles=[(1.2, 1.4, -30, 30)])       # far too long to go round
        task = self.go_to(3.0)
        self.assertEqual(task.status, "blocked", task.detail)
        self.assertLess(robot.x + .36, 1.2, "never touched the wall")

    def test_going_back_to_the_start_goes_around_too(self):
        robot = self.attach(obstacles=[(1.2, 1.6, -.4, .4)])
        self.assertEqual(self.go_to(3.0).status, "completed")
        before = len(self.events)
        self.executor.submit(resolve("go_to", {"target": "start"}), "g2", 2)
        self.assertTrue(wait_for(lambda: len(self.events) > before, 60))
        task = self.events[-1]
        self.assertEqual(task.status, "completed", task.detail)
        self.assertLess(math.hypot(robot.x, robot.y), .2)


@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class RoomRouteTests(unittest.TestCase):
    """In the MuJoCo room: from just west of the cardboard box, go_to a point just east of it."""

    def test_walks_around_the_box_in_the_room(self):
        from go2.adapters.sim import SimAdapter
        sim = SimAdapter(python=sys.executable, scene="room", headless=True)
        sim.connect()
        self.addCleanup(sim.close)
        events = []
        executor = Executor(on_event=events.append)
        executor.attach(sim)
        self.addCleanup(executor.close)
        sim.reset(-1.0, -1.1, 0.)                       # the box is at (0.2, -1.1), straight ahead
        self.assertTrue(wait_for(lambda: sim.state().posture == "standing", 5))
        executor.submit(resolve("go_to", {"x_m": 2.1, "y_m": 0}), "g", 1)
        self.assertTrue(wait_for(lambda: events, 90))
        task = events[-1]
        s = sim.state()
        self.assertEqual(task.status, "completed", task.detail)
        self.assertIn("route", task.detail)
        self.assertLess(math.hypot(s.x - 1.1, s.y + 1.1), .2, f"ended at ({s.x:.2f}, {s.y:.2f})")
        self.assertIsNone(s.fault, "no collision or fall")


if __name__ == "__main__":
    unittest.main()
