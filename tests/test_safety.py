"""Ledge and stall protection (#22): drops in the LiDAR cloud, and moves stopped by something the LiDAR can't
see. Synthetic clouds and the kinematic FakeRobot; the MuJoCo hazards scene is in test_hazards.py."""
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2 import executor as X  # noqa: E402
from go2 import perception as P  # noqa: E402
from go2.agent import describe_outcome  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.tools import resolve  # noqa: E402
from tests.fakes import FakeRobot, wait_for  # noqa: E402


def cloud(z_of, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-3, 4, n), rng.uniform(-3, 3, n)
    return np.stack([x, y, z_of(x, y)], 1)


class DropPerceptionTests(unittest.TestCase):
    def test_a_ledge_ahead_is_a_drop_at_the_edge(self):
        c = cloud(lambda x, y: np.where(x < 1.0, 0., -.5))         # floor ends 1 m ahead, 0.5 m drop
        self.assertAlmostEqual(P.drop_along(c, 1, 0), 1.0 - P.HALF_LENGTH, delta=.1)
        self.assertAlmostEqual(P.clearance_along(c, 1, 0), 1.0 - P.HALF_LENGTH, delta=.1)
        self.assertEqual(P.obstacle_along(c, 1, 0), math.inf, "a drop is not an obstacle")
        self.assertLess(P.observed_along(c, 1, 0), 1.0, "the lower floor does not count as seeing the corridor")

    def test_the_issue_example_is_no_longer_clear(self):
        rng = np.random.default_rng(0)
        x, y = rng.uniform(-1, 4, 6000), rng.uniform(-1.5, 1.5, 6000)
        c = np.stack([x, y, np.where(x < 1.0, 0.0, -0.5)], 1)
        self.assertLess(P.clearance_along(c, 1, 0), 1.0)
        self.assertIsNotNone(P.sector_free_space(c)["front"])
        z = np.where((x > 1.0) & (x < 1.3) & (abs(y) < .3), 0.3, 0.0)
        self.assertAlmostEqual(P.clearance_along(np.stack([x, y, z], 1), 1, 0), .64, delta=.05, msg="boxes as before")

    def test_drops_behind_and_to_the_side(self):
        c = cloud(lambda x, y: np.where((x > -1.2) & (y > -1.0), 0., -.5))
        self.assertAlmostEqual(P.drop_along(c, -1, 0), 1.2 - P.HALF_LENGTH, delta=.1)
        self.assertAlmostEqual(P.drop_along(c, 0, -1), 1.0 - P.HALF_WIDTH, delta=.1)
        self.assertEqual(P.drop_along(c, 1, 0), math.inf)
        self.assertEqual(P.drop_along(c, 0, 1), math.inf)

    def test_stairs_going_down_are_a_drop(self):
        c = cloud(lambda x, y: np.where(x < 1.0, 0., -.125 * np.ceil((x - 1.0) / .3)))
        self.assertAlmostEqual(P.drop_along(c, 1, 0), 1.0 - P.HALF_LENGTH, delta=.1)

    def test_a_flat_floor_with_noise_never_looks_like_a_drop(self):
        c = cloud(lambda x, y: np.random.default_rng(1).normal(0, .015, len(x)))
        for a in range(0, 360, 45):
            self.assertEqual(P.drop_along(c, math.cos(math.radians(a)), math.sin(math.radians(a))), math.inf)
        self.assertEqual(P.turn_clearance(c), math.inf)
        self.assertTrue(all(v is None for v in P.sector_free_space(c).values()))

    def test_turning_checks_for_floor_under_the_turning_circle(self):
        c = cloud(lambda x, y: np.where(x < .6, 0., -.5))
        self.assertLess(P.turn_clearance(c), .25)
        self.assertEqual(P.turn_clearance(cloud(lambda x, y: np.zeros_like(x))), math.inf)

    def test_look_gives_the_distance_to_the_edge(self):
        c = cloud(lambda x, y: np.where(x < 1.5, 0., -.5))
        free = P.sector_free_space(c)
        self.assertAlmostEqual(free["front"], 1.5 - P.HALF_LENGTH, delta=.1)
        self.assertIsNone(free["back"])


class StallAndLedgeExecutorTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.executor = Executor(on_event=self.events.append)
        self.robot = None

    def tearDown(self):
        self.executor.close()
        if self.robot:
            self.robot.close()

    def attach(self, **kwargs):
        self.robot = FakeRobot(**kwargs)
        self.robot.connect()
        self.executor.attach(self.robot)
        return self.robot

    def submit(self, name, args, group="g1"):
        return self.executor.submit(resolve(name, args), group, 1)

    def finished(self, count=1, timeout=25):
        self.assertTrue(wait_for(lambda: len(self.events) >= count, timeout), "task did not finish")
        return self.events[-1]

    def test_walking_toward_a_ledge_stops_before_the_edge(self):
        robot = self.attach(drops=[(1.5, 9, -9, 9)])
        self.submit("move", {"direction": "forward", "distance_m": 3})
        task = self.finished()
        self.assertEqual(task.status, "blocked")
        self.assertEqual(task.detail["blocked"]["reason"], "drop ahead")
        self.assertLess(robot.x + P.HALF_LENGTH, 1.5, "front of the body still on the platform")
        self.assertGreater(robot.x, .3, "it walked most of the way")
        self.assertIn("drop ahead", describe_outcome(task, [task]))

    def test_backwards_and_sideways_ledges_stop_too(self):
        robot = self.attach(drops=[(-9, -1.4, -9, 9), (-9, 9, -9, -1.2)])
        self.submit("move", {"direction": "backward", "distance_m": 3})
        task = self.finished()
        self.assertEqual(task.detail["blocked"]["reason"], "drop behind")
        self.assertGreater(robot.x - P.HALF_LENGTH, -1.4)
        self.submit("move", {"direction": "right", "distance_m": 3}, group="g2")
        task = self.finished(2)
        self.assertEqual(task.detail["blocked"]["reason"], "drop to the right")
        self.assertGreater(robot.y - P.HALF_WIDTH, -1.2)

    def test_a_move_starting_at_the_edge_is_refused(self):
        robot = self.attach(drops=[(.55, 9, -9, 9)])
        result = self.submit("move", {"direction": "forward", "distance_m": 1})
        self.assertEqual(result[0]["status"], "blocked")
        self.assertIn("drop ahead", result[0]["error"])
        self.assertAlmostEqual(robot.x, 0, delta=.02)

    def test_look_reports_the_edge_as_a_distance(self):
        self.attach(drops=[(1.5, 9, -9, 9)])
        free = self.executor.look()["free_space_m"]
        self.assertIsInstance(free["front"], float)
        self.assertAlmostEqual(free["front"], 1.5 - P.HALF_LENGTH, delta=.15)
        self.assertEqual(free["back"], "clear")

    def test_glass_stops_the_robot_quickly_and_it_backs_off(self):
        robot = self.attach(glass=[(1.2, 1.25, -1, 1)])
        self.submit("move", {"direction": "forward", "distance_m": 2.5})
        task = self.finished()
        self.assertEqual(task.status, "blocked")
        blocked = task.detail["blocked"]
        self.assertTrue(blocked.get("stall"))
        self.assertEqual(blocked["reason"], X.STALLED)
        self.assertIsNotNone(robot.contact_at)
        stop_times = [t for t in robot.stop_times if t > robot.contact_at]
        self.assertLess(stop_times[0] - robot.contact_at, .6, "stopped within about half a second of contact")
        self.assertGreater(blocked.get("backed_off_m", 0), .1, "backed away from the glass")
        self.assertLess(robot.x + .36, 1.2 - .08)
        text = describe_outcome(task, [task])
        self.assertNotIn("without touching", text)
        self.assertIn("backed off", text)

    def test_asking_again_does_not_walk_back_into_the_glass(self):
        robot = self.attach(glass=[(1.2, 1.25, -1, 1)])
        self.submit("move", {"direction": "forward", "distance_m": 2.5})
        self.finished()
        before = robot.x
        result = self.submit("move", {"direction": "forward", "distance_m": 2.5}, group="g2")
        if result[0]["status"] != "blocked":
            self.finished(2)
        task = self.events[-1] if result[0]["status"] != "blocked" else None
        reason = result[0].get("error", "") if task is None else task.detail["blocked"]["reason"]
        self.assertIn("something I bumped into earlier", reason)
        self.assertLess(robot.x - before, .2, "it did not walk back into the glass")

    def test_a_timed_move_into_glass_is_blocked_not_completed(self):
        self.attach(glass=[(.9, .95, -1, 1)])
        self.submit("move", {"direction": "forward", "duration_s": 5})
        task = self.finished()
        self.assertEqual(task.status, "blocked")
        self.assertTrue(task.detail["blocked"]["stall"])

    def test_slow_moves_turns_and_approach_creep_never_stall_on_open_floor(self):
        robot = self.attach()
        for i, (name, args) in enumerate([("move", {"direction": "forward", "distance_m": 1, "speed_mps": .2}),
                                          ("turn", {"direction": "left", "angle_deg": 90}),
                                          ("move", {"direction": "left", "distance_m": .5}),
                                          ("go_to", {"x_m": -1, "y_m": .5})]):
            self.submit(name, args, group=f"g{i}")
            task = self.finished(i + 1)
            self.assertEqual(task.status, "completed", f"{name} {args}: {task.detail}")
        self.assertIsNone(robot.contact_at)

    def test_a_slow_simulator_is_not_mistaken_for_a_stall(self):
        robot = self.attach()
        self.submit("move", {"direction": "forward", "distance_m": 2})
        time.sleep(1.2)
        robot.paused_until = time.monotonic() + .7     # no progress for 0.7 s, but the speed reading stays
        task = self.finished()
        self.assertEqual(task.status, "completed", task.detail)

    def test_a_slow_start_is_not_mistaken_for_a_stall(self):
        robot = self.attach()
        robot.start_lag = .35                          # legs barely move for the first 0.35 s, as in MuJoCo
        for i in range(8):
            self.submit("move", {"direction": "right" if i % 2 else "left", "distance_m": .25}, group=f"g{i}")
            task = self.finished(i + 1)
            self.assertEqual(task.status, "completed", f"side-step {i}: {task.detail}")

    def test_starting_against_glass_is_still_caught(self):
        robot = self.attach(glass=[(.38, .43, -1, 1)])  # the nose is 2 cm from the pane
        robot.start_lag = .2
        began = time.monotonic()
        self.submit("move", {"direction": "forward", "distance_m": 2})
        task = self.finished()
        self.assertEqual(task.status, "blocked", task.detail)
        self.assertTrue(task.detail["blocked"].get("stall"))
        stopped = min(t for t in robot.stop_times if t > began)       # before backing off
        self.assertLess(stopped - began, 2.5, "the stall check started even though it never got going")

    def test_bump_spots_are_forgotten_after_a_while(self):
        self.attach(glass=[(1.2, 1.25, -1, 1)])
        self.submit("move", {"direction": "forward", "distance_m": 2.5})
        self.finished()
        self.assertEqual(len(self.executor.bumps), 1)
        self.executor.bumps[0] = (time.monotonic() - X.BUMP_MEMORY_S - 1, self.executor.bumps[0][1])
        s = self.robot.state()
        self.assertEqual(len(self.executor._with_bumps(np.zeros((0, 3)), s)), 0)


if __name__ == "__main__":
    unittest.main()
