import math
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from go2 import perception as P
from go2.executor import Executor, Task
from go2.tools import resolve
from go2.room_map import OccupancyGrid, FREE, OCCUPIED
from go2.exploration import _safe_cells
from tests.fakes import FakeRobot, wait_for


class ExplorationExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.events = []
        self.robot = FakeRobot()
        self.robot.connect()
        self.addCleanup(self.robot.close)
        self.executor = Executor(on_event=self.events.append, memory_path=self.temp.name + "/memory.json")
        self.executor.attach(self.robot)
        self.addCleanup(self.executor.close)

    def test_world_viewpoint_runs_through_worker_and_reaches_position(self):
        result = self.executor.submit_viewpoint((.65, .2, .25))
        self.assertEqual(result["status"], "queued", result)
        self.assertTrue(wait_for(lambda: self.events, 12))
        task = self.events[-1]
        self.assertEqual(task.status, "completed", task.detail)
        self.assertLess(math.hypot(self.robot.x - .65, self.robot.y - .2), .15)
        self.assertAlmostEqual(self.robot.yaw, .25, delta=.15)
        self.assertIsNotNone(task.detail["room"])

    def test_stop_cancels_viewpoint_without_resuming(self):
        self.executor.submit_viewpoint((2., 0., 0.))
        self.assertTrue(wait_for(lambda: self.robot.calls and self.robot.last_move, 3))
        self.executor.stop("test cancellation")
        self.assertTrue(wait_for(lambda: self.events, 3))
        self.assertEqual(self.events[-1].status, "stopped")
        stopped = self.robot.x
        time.sleep(.4)
        self.assertLess(abs(self.robot.x - stopped), .1)

    def test_viewpoint_corrects_lateral_drift_and_checks_each_corrected_direction(self):
        self._check_viewpoint_directions()

    def test_viewpoint_checks_each_corrected_direction_with_delayed_scans(self):
        self._check_viewpoint_directions(scan_delay=.075)

    def _check_viewpoint_directions(self, scan_delay=0.):
        native_move, native_state = self.robot.move, self.robot.state
        clearance = self.executor._clearance
        checked, unsafe, maximum = [], [], [0.]
        def observe():
            state = native_state()
            maximum[0] = max(maximum[0], abs(state.y))
            return state
        def check(ux, uy, turning, **kwargs):
            result = clearance(ux, uy, turning, **kwargs)
            # Alternate slow and fast scans to move the send and scan deadlines
            # apart; constant latency can leave the two timers in lockstep.
            if scan_delay and len(checked) % 2 == 0:
                time.sleep(scan_delay)
            checked.append((ux, uy, turning))
            return result
        def drift(vx, vy, wz):
            speed = math.hypot(vx, vy)
            if speed > .01:
                direction = (vx/speed, vy/speed, False)
                if (speed > .150001 or not checked or
                        not np.allclose(checked[-1], direction, atol=1e-6)):
                    unsafe.append((vx, vy, checked[-1:] ))
                vy += .035  # repeatable walking bias, independent of command
            native_move(vx, vy, wz)
        self.robot.state = observe
        self.robot.move = drift
        self.executor._clearance = check
        self.executor.submit_viewpoint((1.5, 0., 0.), timeout_s=30)
        self.assertTrue(wait_for(lambda: self.events, 35))
        self.assertEqual(self.events[-1].status, "completed", self.events[-1].detail)
        self.assertLess(maximum[0], .09, 'stay within the route tracking allowance')
        self.assertEqual(unsafe, [], 'check a corrected direction before commanding it')

    def test_viewpoint_at_current_position_can_turn_without_a_translation_leg(self):
        self.executor.submit_viewpoint((0., 0., .4))
        self.assertTrue(wait_for(lambda: self.events, 6))
        self.assertEqual(self.events[-1].status, "completed", self.events[-1].detail)
        self.assertAlmostEqual(self.robot.yaw, .4, delta=.15)

    def test_short_budget_fails_and_stops_motion(self):
        self.executor.submit_viewpoint((2., 0., 0.), timeout_s=.2)
        self.assertTrue(wait_for(lambda: self.events, 3))
        self.assertEqual(self.events[-1].status, "failed")
        self.assertIn("timed out", self.events[-1].detail["error"])
        self.assertIsNone(self.robot.last_move)

    def test_attach_clears_map_and_room_identities(self):
        self.executor.room_map.mark_obstacles([[1., 0.]])
        self.executor.attach(self.robot)
        self.assertTrue((self.executor.room_map.grid == -1).all())
        self.assertIsNone(self.executor.room_at(0, 0))

    def test_viewpoint_does_not_fall_back_to_unknown_space(self):
        # This goal is well beyond every initial LiDAR observation.
        self.executor.submit_viewpoint((8., 0., 0.))
        self.assertTrue(wait_for(lambda: self.events, 3))
        self.assertEqual(self.events[-1].status, "blocked")
        self.assertAlmostEqual(self.robot.x, 0., delta=.05)

    def test_viewpoint_query_does_not_move_robot(self):
        self.executor.exploration_viewpoint([])
        self.assertFalse(any(call[0] == "move" for call in self.robot.calls))

    def test_viewpoint_query_prefers_a_fine_normal_route_before_recovery(self):
        normal, recovery = (-1., 1., .5), (.3, 0., 0.)
        def selector(context, visited):
            if context.planning_resolution_m == .05 and context.turn_clearance_m == .55:
                return normal
            return recovery if getattr(context, "allow_recovery", True) else None
        with patch("go2.executor.next_viewpoint", side_effect=selector):
            self.assertEqual(self.executor.exploration_viewpoint([]), normal)
        self.assertFalse(any(call[0] == "move" for call in self.robot.calls))

    def test_blocked_normal_route_recovers_to_turning_space_before_fresh_goal(self):
        m = OccupancyGrid(resolution=.05, size=160)
        m.grid[:] = OCCUPIED
        r, c = np.indices(m.grid.shape)
        x = (c + m._min_col + .5) * m.resolution
        y = (r + m._min_row + .5) * m.resolution
        m.grid[(x > -.6) & (x < .9) & (abs(y) < .3)] = FREE
        m.grid[(x > -2.8) & (x < -.6) & (abs(y) < 1.2)] = FREE
        m.grid[(x > -.6) & (x < -.3)] = OCCUPIED
        m.grid[(x > -.6) & (x < -.3) & (abs(y) < .25)] = FREE
        with self.robot.lock:
            self.robot.x = .3
        self.executor.room_map = m
        # Isolate route/recovery behavior from the fake's open-field sensor.
        self.executor._refresh_cloud = self.robot.state
        self.executor._clearance = lambda *a, **kw: math.inf
        self.executor.submit_viewpoint((-1.8, 0., 0.), timeout_s=30)
        self.assertTrue(wait_for(lambda: self.events, 35))
        task = self.events[-1]
        self.assertEqual(task.status, "blocked", task.detail)
        self.assertIn("fresh viewpoint", task.detail["error"])
        self.assertLess(self.robot.x, -1.)
        self.assertTrue(_safe_cells(m, .55)[m.world_to_cell(self.robot.x, self.robot.y)])
        self.assertIsNone(self.robot.last_move)

    def test_flat_sim_map_keeps_world_floor_when_local_ground_is_a_low_object(self):
        self.robot.scene = "house"
        original = self.robot.state
        self.robot.state = lambda: replace(original(), height=.26, extra={"world_z": .3})
        # Raw sensor returns are expressed about a base at z=.3, not about the
        # spurious local ground reference at z=.04.
        self.robot.lidar = lambda: (np.array([[1., 0., -.3], [1.5, 0., -.26]]), self.robot.state())
        m = self.executor.room_map
        with patch.object(m, "mark_scan", wraps=m.mark_scan) as mark:
            self.executor._refresh_cloud()
        points = mark.call_args.args[1]
        self.assertAlmostEqual(points[0, 2], 0.)
        self.assertAlmostEqual(points[1, 2], .04)
        self.assertEqual(m.grid[m.world_to_cell(1., 0.)], 0)

    def test_known_route_detours_when_the_live_initial_direction_is_blocked(self):
        self._check_live_route_start(known_only=True)

    def test_general_route_detours_when_the_live_initial_direction_is_blocked(self):
        self._check_live_route_start(known_only=False)

    def test_route_uses_map_evidence_observed_by_live_probes(self):
        self.executor.room_map.grid[:] = -1
        def observe_floor(*args, **kwargs):
            self.executor.room_map.grid[:] = FREE
            return math.inf
        self.executor._clearance = observe_floor
        self.assertEqual(self.executor._route(self.robot.state(), 1., 0., known_only=True), [])

    def _check_live_route_start(self, known_only):
        self.executor.room_map.grid[:] = 0
        def clearance(ux, uy, turning):
            return 0. if ux > .9 and abs(uy) < .4 else math.inf
        self.executor._clearance = clearance
        route = self.executor._route(self.robot.state(), 1., 0., known_only=known_only)
        self.assertTrue(route)
        x, y = route[0]
        distance = math.hypot(x, y)
        self.assertGreater(clearance(x / distance, y / distance, False), self.executor._margin(.2))

    def test_off_centre_pose_does_not_turn_clear_axes_into_blocked_diagonals(self):
        s=replace(self.robot.state(),x=.038,y=.046,yaw=0.)
        self.executor._clearance=lambda ux,uy,turning,**kw: math.inf if min(abs(ux),abs(uy))<.01 else 0.
        forbidden=self.executor._exploration_forbidden(s)
        for step in ((-1,0),(1,0),(0,-1),(0,1)):
            self.assertNotIn(step,forbidden)
        self.assertIn((1,1),forbidden)

    def test_approach_reaches_nearby_route_bend_before_replanning(self):
        task = Task(resolve("approach", {"target": "laptop", "stop_distance_m": .5})[0],
                    "test", self.executor.generation)
        with patch.object(self.executor, "_route", return_value=[(.06, 0.), (2., 0.)]):
            self.assertTrue(self.executor._approach_route(
                task, threading.Event(), self.robot.state(), 3., 0., .5))
        self.assertGreater(self.robot.x, .03, "a near bend must not become a no-op replan")

    def test_approach_routes_around_furniture_and_reacquires_target(self):
        self.robot.obstacles = [(1.1, 1.6, -.4, .4)]
        e = self.executor
        target = (3., 0.)
        looks_without_target = []

        def look():
            for _ in range(3):
                e._refresh_cloud()
            s = self.robot.state()
            dx, dy = target[0] - s.x, target[1] - s.y
            c, sn = math.cos(s.yaw), math.sin(s.yaw)
            bx, by = c * dx + sn * dy, -sn * dx + c * dy
            bearing = math.degrees(math.atan2(by, bx))
            if abs(bearing) > 65:
                e.last_detections = []
                looks_without_target.append(1)
            else:
                det = P.Detection("laptop1", "laptop", None, math.hypot(bx, by), bearing,
                                  bx, by, "lidar", 1.)
                e.last_detections = [(det, (s.x, s.y, s.yaw))]
            return {}

        e.look = look
        e.submit(resolve("approach", {"target": "laptop", "stop_distance_m": .5}), "test", 1)
        self.assertTrue(wait_for(lambda: self.events, 60), "approach did not finish")
        task = self.events[-1]
        self.assertEqual(task.status, "completed", task.detail)
        self.assertIn("route", task.detail)
        self.assertAlmostEqual(task.detail["final_gap_m"], .5, delta=.25)
        self.assertLess(abs(self.robot.y), .6)


if __name__ == "__main__":
    unittest.main()
