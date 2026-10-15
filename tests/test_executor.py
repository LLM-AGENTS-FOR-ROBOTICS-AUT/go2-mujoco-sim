import math
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2 import limits as L  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.tools import resolve  # noqa: E402
from tests.fakes import FakeRobot, wait_for  # noqa: E402


class ExecutorTests(unittest.TestCase):
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

    def submit(self, name, args, turn=1, group="g1"):
        return self.executor.submit(resolve(name, args), group, turn)

    def finished(self, count=1, timeout=20):
        self.assertTrue(wait_for(lambda: len(self.events) >= count, timeout), "task did not finish")
        return self.events[-1]

    def test_move_completes_by_measured_distance(self):
        robot = self.attach()
        result = self.submit("move", {"direction": "forward", "distance_m": 1})
        self.assertEqual(result[0]["status"], "queued")
        task = self.finished()
        self.assertEqual(task.status, "completed")
        self.assertAlmostEqual(robot.x, 1.0, delta=.06)
        self.assertAlmostEqual(task.detail["moved_m"], 1.0, delta=.06)
        self.assertEqual(robot.calls[-1], ("stop_move",))

    def test_turn_completes_by_measured_angle(self):
        robot = self.attach()
        self.submit("turn", {"direction": "right", "angle_deg": 90})
        task = self.finished()
        self.assertEqual(task.status, "completed")
        self.assertAlmostEqual(math.degrees(robot.yaw), -90, delta=5)

    def test_queued_steps_run_in_order(self):
        robot = self.attach()
        commands = resolve("move", {"direction": "forward", "distance_m": .5}) + resolve("turn", {"direction": "left"})
        self.executor.submit(commands, "g", 1)
        self.finished(2)
        self.assertEqual([t.command.tool for t in self.events], ["move", "turn"])
        self.assertAlmostEqual(robot.x, .5, delta=.06)
        self.assertAlmostEqual(math.degrees(robot.yaw), 90, delta=5)

    def test_stop_preempts_running_and_queued_work(self):
        robot = self.attach()
        commands = resolve("move", {"direction": "forward", "distance_m": 4}) + resolve("turn", {})
        self.executor.submit(commands, "g", 1)
        self.assertTrue(wait_for(lambda: robot.v[0] > .3))
        result = self.executor.stop("user")
        self.assertTrue(result["stop_sent"])
        self.assertEqual(result["cancelled_queued"], 1)
        task = self.finished()
        self.assertEqual(task.status, "stopped")
        time.sleep(.8)
        self.assertLess(abs(robot.v[0]), .02, "robot must not resume after stop")
        self.assertLess(robot.x, 3.0)
        self.assertEqual(self.executor.snapshot()["queue"], [])

    def test_blocked_path_is_refused_before_moving(self):
        robot = self.attach(obstacles=[(.5, .9, -.5, .5)])
        commands = resolve("move", {"direction": "forward", "distance_m": 2}) + resolve("turn", {})
        results = self.executor.submit(commands, "g", 1)
        task = self.finished()
        self.assertEqual(task.status, "blocked")
        self.assertLess(robot.x, .05)
        self.assertEqual(task.detail["cancelled_after"], 1, "later steps of the same request are cancelled")
        # The tool call hears about it straight away (not "queued"), so it is not announced again later.
        self.assertEqual(results[0]["status"], "blocked")
        self.assertIn("path is blocked", results[0]["error"])
        self.assertEqual(results[1], {"status": "cancelled", "action": "turn left 90° at 29°/s",
                                      "reason": "previous step blocked"})
        self.assertTrue(task.reported)

    def test_a_motion_that_starts_is_reported_as_queued(self):
        self.attach(obstacles=[(2.0, 2.4, -.5, .5)])
        result = self.submit("move", {"direction": "forward", "distance_m": 4})
        self.assertEqual(result[0]["status"], "queued")
        task = self.finished()
        self.assertEqual(task.status, "blocked")       # blocked later, while moving: announced as usual
        self.assertFalse(task.reported)

    def test_status_reports_what_was_done_since_connecting(self):
        self.attach(obstacles=[(2.4, 2.8, -.5, .5)])
        self.submit("move", {"direction": "forward", "distance_m": 1})
        self.finished()
        self.submit("turn", {"direction": "left", "angle_deg": 90}, group="g2")
        self.finished(2)
        self.submit("turn", {"direction": "right", "angle_deg": 90}, group="g3")
        self.finished(3)
        self.submit("move", {"direction": "forward", "distance_m": 2}, group="g4")
        self.finished(4)
        done = self.executor.status_report()["done_so_far"]
        self.assertEqual(done["actions"], 4)
        self.assertAlmostEqual(done["walked_m"], 1.5, delta=.3)        # 1 m, then blocked short of the box
        self.assertAlmostEqual(done["turned_deg"], 180, delta=12)
        self.assertEqual(done["recent"][0], "move forward 1 m at 0.20 m/s: completed")
        self.assertTrue(done["recent"][-1].startswith("move forward 2 m at 0.40 m/s: blocked after"), done["recent"][-1])
        self.assertIn("(obstacle ahead)", done["recent"][-1])
        self.attach()
        self.assertEqual(self.executor.status_report()["done_so_far"]["actions"], 0, "a new robot starts afresh")

    def test_blocked_reports_name_the_direction_of_travel(self):
        robot = self.attach(obstacles=[(-.5, .5, 1.6, 2.0), (-2.4, -2.0, -.5, 2.0)])
        self.submit("move", {"direction": "left", "distance_m": 3})
        task = self.finished()
        self.assertEqual(task.detail["blocked"]["reason"], "obstacle to the left")
        self.submit("move", {"direction": "backward", "distance_m": 4}, group="g2")
        task = self.finished(2)
        self.assertEqual(task.detail["blocked"]["reason"], "obstacle behind")
        self.assertLess(robot.x - .36, -1.0)

    def test_obstacle_stops_motion_before_contact(self):
        robot = self.attach(obstacles=[(2.0, 2.4, -.5, .5)])
        self.submit("move", {"direction": "forward", "distance_m": 4})
        task = self.finished()
        self.assertEqual(task.status, "blocked")
        front_edge = robot.x + .36
        self.assertLess(front_edge, 2.0, "stopped before the obstacle face")
        self.assertGreater(front_edge, 1.4, "did not stop absurdly early")

    def test_side_obstacles_do_not_block_forward_motion(self):
        robot = self.attach(obstacles=[(.3, 2.5, .5, .8), (.3, 2.5, -.8, -.5)])   # a corridor
        self.submit("move", {"direction": "forward", "distance_m": 2})
        self.assertEqual(self.finished().status, "completed")
        self.assertAlmostEqual(robot.x, 2, delta=.06)

    def test_unseen_direction_is_never_assumed_free(self):
        robot = self.attach(blind_behind=True)
        self.submit("move", {"direction": "backward", "distance_m": 1})
        task = self.finished()
        self.assertEqual(task.status, "blocked")
        self.assertIn("can't see", task.detail["blocked"]["reason"])
        self.assertGreater(robot.x, -.05)
        self.submit("move", {"direction": "forward", "distance_m": .5}, group="g2")
        self.assertEqual(self.finished(2).status, "completed", "directions it can see still work")

    def test_stale_telemetry_stops_the_robot(self):
        robot = self.attach()
        self.submit("move", {"direction": "forward", "distance_m": 3})
        self.assertTrue(wait_for(lambda: robot.v[0] > .2))
        robot.freeze = True
        task = self.finished()
        self.assertEqual(task.status, "failed")
        self.assertIn("telemetry", task.detail["error"])
        self.assertEqual(robot.calls[-1], ("stop_move",))

    def test_auto_stand_before_walking(self):
        robot = self.attach(posture="lying")
        self.submit("move", {"direction": "forward", "distance_m": .5})
        task = self.finished()
        self.assertEqual(task.status, "completed")
        self.assertTrue(task.detail.get("stood_up_first"))
        self.assertIn(("sport", "stand", {}), robot.calls)

    def test_arming_gate_for_real_robots(self):
        self.attach(requires_arming=True)
        result = self.submit("move", {"direction": "forward"})
        self.assertEqual(result[0]["status"], "rejected")
        self.assertIn("disarmed", result[0]["error"])
        self.assertEqual(self.submit("get_status", {})[0]["status"], "done", "queries still work while disarmed")
        self.executor.set_armed(True)
        self.assertEqual(self.submit("move", {"direction": "forward", "distance_m": .3})[0]["status"], "queued")

    def test_acrobatics_need_confirmation_in_a_later_turn(self):
        robot = self.attach()
        first = self.submit("acrobatic", {"name": "front_flip"}, turn=1)
        self.assertEqual(first[0]["status"], "needs_confirmation")
        same_turn = self.submit("acrobatic", {"name": "front_flip", "confirmed": True}, turn=1)
        self.assertEqual(same_turn[0]["status"], "needs_confirmation", "model cannot confirm for the user")
        confirmed = self.submit("acrobatic", {"name": "front_flip", "confirmed": True}, turn=2)
        self.assertEqual(confirmed[0]["status"], "queued")
        self.finished()
        self.assertIn(("sport", "front_flip", {"confirmed": True}), robot.calls)

    def test_confirmation_without_a_prior_request_asks_first(self):
        self.attach()
        result = self.submit("acrobatic", {"name": "back_flip", "confirmed": True}, turn=3)
        self.assertEqual(result[0]["status"], "needs_confirmation")
        self.assertEqual(self.submit("acrobatic", {"name": "back_flip", "confirmed": True}, turn=4)[0]["status"], "queued")

    def test_confirmations_expire_and_stop_clears_them(self):
        self.attach()
        self.submit("acrobatic", {"name": "front_jump"}, turn=1)
        self.executor.stop("user")
        self.assertEqual(self.submit("acrobatic", {"name": "front_jump", "confirmed": True}, turn=2)[0]["status"],
                         "needs_confirmation")
        self.executor.confirmations["front_jump"] = (2, time.monotonic() - L.CONFIRMATION_TTL_S - 1)
        self.assertEqual(self.submit("acrobatic", {"name": "front_jump", "confirmed": True}, turn=3)[0]["status"],
                         "needs_confirmation")

    def test_unsupported_commands_are_refused_up_front(self):
        self.attach(unsupported={"back_flip"})
        result = self.submit("acrobatic", {"name": "back_flip"})
        self.assertEqual(result[0]["status"], "rejected")

    def test_go_to_point_and_back_to_start(self):
        robot = self.attach()
        self.submit("go_to", {"x_m": 1.5, "y_m": 1.0})
        self.assertEqual(self.finished().status, "completed")
        self.assertAlmostEqual(robot.x, 1.5, delta=.15)
        self.assertAlmostEqual(robot.y, 1.0, delta=.15)
        self.submit("go_to", {"target": "start"}, group="g2")
        task = self.finished(2)
        self.assertEqual(task.status, "completed")
        self.assertLess(math.hypot(robot.x, robot.y), .15)

    def test_short_target_behind_reverses_instead_of_turning(self):
        robot = self.attach()
        self.submit("go_to", {"x_m": -.5})
        self.assertEqual(self.finished().status, "completed")
        self.assertAlmostEqual(robot.x, -.5, delta=.08)
        self.assertLess(abs(math.degrees(robot.yaw)), 5)

    def test_status_report_is_plain_language(self):
        self.attach()
        report = self.submit("get_status", {})[0]
        self.assertEqual(report["position"], "right where it started")
        self.assertEqual(report["doing"], "nothing")

    def test_matching_tolerates_colour_drift_and_suggests_closest(self):
        from go2.perception import Detection
        self.attach()
        def det(i, label, color, x, y):
            return Detection(f"obj{i}", label, color, (x * x + y * y) ** .5, 0., x, y, "lidar", .9)
        pose = (0., 0., 0.)
        self.executor.last_detections = [(det(1, "box", "brown", 2., 0.), pose), (det(2, "sports ball", "red", 3., 1.), pose)]
        match, closest = self.executor._match("orange box")          # orange ~ brown under different light
        self.assertEqual(match[0].id, "obj1")
        match, closest = self.executor._match("red ball")
        self.assertEqual(match[0].label, "sports ball")
        match, closest = self.executor._match("blue box")            # no blue box: suggest the brown one
        self.assertIsNone(match)
        self.assertEqual(closest.label, "box")
        self.assertIsNone(self.executor._match("unicorn")[0])

    def test_matching_works_with_any_detector_labels(self):
        from go2.perception import Detection
        self.attach()
        pose = (0., 0., 0.)
        labels = ["crate", "basketball", "blue water bottle", "shoe", "coffee table"]
        self.executor.last_detections = [(Detection(f"obj{i}", label, None, 2. + i, 0., 2. + i, 0., "lidar", .9), pose)
                                         for i, label in enumerate(labels)]
        for words, label in [("the crate", "crate"), ("basketball", "basketball"), ("water bottle", "blue water bottle"),
                             ("my shoes", "shoe"), ("coffee table", "coffee table"), ("the table", "coffee table")]:
            self.assertEqual(self.executor._match(words)[0][0].label, label, words)
        # A different thing of the same kind is only offered as the closest match.
        self.executor.last_detections = [(Detection("obj1", "side table", None, 2., 0., 2., 0., "lidar", .9), pose),
                                         (Detection("obj2", "sports ball", "red", 3., 0., 3., 0., "lidar", .9), pose)]
        match, closest = self.executor._match("coffee table")
        self.assertIsNone(match)
        self.assertEqual(closest.label, "side table")
        self.assertEqual(self.executor._match("tennis ball")[0][0].label, "sports ball")   # COCO's generic label
        self.assertEqual(self.executor._match("the big wooden table")[0][0].label, "side table")

    def test_locked_target_is_tracked_by_position_not_colour(self):
        from go2.perception import Detection
        self.attach()
        pose = (0., 0., 0.)
        self.executor.last_detections = [(Detection("obj1", "box", "brown", 2., 0., 2., 0., "lidar", .9), pose),
                                         (Detection("obj2", "box", "orange", 4., 0., 0., 4., "lidar", .9), pose)]
        tracked = self.executor._track(("box", 2.05, 0.1))
        self.assertEqual(tracked[0].id, "obj1")
        self.assertIsNone(self.executor._track(("box", -3., -3.)))

    def test_look_in_the_same_request_waits_for_the_turn(self):
        self.attach()
        self.robot.blind_behind = False
        results = self.executor.submit(resolve("turn", {"direction": "left", "angle_deg": 90}) + resolve("look", {}), "g1", 1)
        self.assertEqual(results[0]["status"], "queued")
        self.assertEqual(self.executor.history[-1].status, "completed")      # the turn finished before looking
        self.assertAlmostEqual(self.robot.yaw, math.pi / 2, delta=.1)
        # A look from a later request doesn't wait for motion that is still going.
        self.executor.submit(resolve("move", {"direction": "forward", "distance_m": 1}), "g2", 2)
        started = time.monotonic()
        self.executor.submit(resolve("look", {}), "g3", 3)
        self.assertLess(time.monotonic() - started, 1.)
        self.executor.stop("test done")

    def test_object_ids_stay_the_same_across_looks(self):
        from go2.perception import Detection
        self.attach()
        first = [Detection("obj1", "chair", "red", 2., 0., 2., 0., "lidar", .9),
                 Detection("obj2", "bucket", "blue", 3., 0., 3., 1., "lidar", .9)]
        self.executor._stable_ids(first, (0., 0., 0.))
        self.executor._remember(first, (0., 0., 0.))
        bucket = next(d.id for d in first if d.label == "bucket")
        # After walking 1 m forward the bucket is seen first and a new thing appears; ids keep their meaning.
        second = [Detection("obj1", "bucket", "blue", 2., 0., 2.02, 1., "lidar", .9),
                  Detection("obj2", "lamp", None, 4., 0., 4., -1., "lidar", .9)]
        self.executor._stable_ids(second, (1., 0., 0.))
        self.assertEqual(second[0].id, bucket)
        self.assertNotIn(second[1].id, {d.id for d in first})
        self.executor._remember(second, (1., 0., 0.))
        self.assertEqual(self.executor._recall(bucket)[0].label, "bucket")

    def test_recently_seen_objects_are_remembered(self):
        from go2.perception import Detection
        self.attach()
        self.executor._remember([Detection("obj1", "sports ball", "red", 2., 0., 2., 0., "lidar", .9)], (1., 1., 1.5708))
        det, wx, wy = self.executor._recall("red ball")
        self.assertAlmostEqual(wx, 1., places=2)
        self.assertAlmostEqual(wy, 3., places=2)

    def test_no_robot_is_a_clear_rejection(self):
        result = self.submit("move", {"direction": "forward"})
        self.assertEqual(result[0]["status"], "rejected")
        self.assertIn("No robot is connected", result[0]["error"])

    def test_stop_from_another_thread_while_worker_sends_moves(self):
        robot = self.attach()
        self.submit("move", {"direction": "left", "distance_m": 3})
        self.assertTrue(wait_for(lambda: robot.v[1] > .2))
        threading.Thread(target=self.executor.stop).start()
        self.finished()
        time.sleep(.7)
        self.assertLess(abs(robot.v[1]), .02)


if __name__ == "__main__":
    unittest.main()
